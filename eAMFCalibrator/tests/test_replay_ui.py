"""The v21 replay page's server: a match's PLAY_OVERs off an export (prod's quote at each), the
states v21 prices and the ones it can't, the nudge toward hand-set expected points, the whole-match
chart, the API's answers and the page served."""

import csv
import datetime as dt
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from .. import config, replay_ui as ru, snowflake_io

HISTORY_COLUMNS = ["MATCH_CODE", "SPORT_CODE", "STREAM_NUMBER", "SCHEDULED_START_TIME_UTC", "PLAYER_1_HANDLE",
                   "PLAYER_1_TEAM", "PLAYER_2_HANDLE", "PLAYER_2_TEAM", "PLAYER_1_FINAL_SCORE",
                   "PLAYER_2_FINAL_SCORE"]
SETUP = {}


def setUpModule():
    from eAMFModel import v21
    from eAMFModel.tests.fakes import _matches
    from eAMFModel.tests.test_v21 import _history
    matches = _matches(n=40, seed=3)
    hist = _history(matches)
    SETUP["model"] = v21.build(matches, hist, mode="tilt", min_records=20, verbose=False, refine=False,
                               key="lean", blend=40.0, field=0.2)
    SETUP["plain"] = v21.build(matches, hist, mode="full", min_records=20, verbose=False, refine=False)
    SETUP["matches"], SETUP["hist"] = matches, hist
    SETUP["dir"] = tempfile.TemporaryDirectory()
    d = SETUP["dir"].name
    rows = []
    for code in sorted(matches)[:2]:                   # prod quoted the first two matches, every PLAY_OVER
        for r in matches[code]:
            lead = int(r["score_p1"]) - int(r["score_p2"])
            p = min(0.95, max(0.05, 0.5 + lead / 30))
            rows.append(dict(r, prob_50=f"{p:.3f}", live_50="1", line_52="-2.5", prob_52=f"{p - 0.05:.3f}",
                             live_52="1", line_54="38.5", prob_54="0.5", live_54="0", bet_state="BET_UNSUSPEND"))
    SETUP["export"] = os.path.join(d, "export.csv")
    with open(SETUP["export"], "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    SETUP["history"] = os.path.join(d, "history.csv")
    with open(SETUP["history"], "w", newline="") as fh:
        w = csv.DictWriter(fh, HISTORY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(dict(h, SPORT_CODE="AF", STREAM_NUMBER="1") for h in hist)


def tearDownModule():
    SETUP["dir"].cleanup()


def replay(model="model", prior=None, history=True, paths=200):
    src = ru.ExportSource(SETUP["export"], SETUP["history"] if history else None)
    return ru.Replay(src, SETUP[model], prior, paths=paths, model_file="m.pkl")


class FixedPrior:
    """A prior with set expected points, keeping what it was asked."""

    def __init__(self, means):
        self.points, self.asked = means, []

    def means(self, rows, results=None):
        self.asked.append((rows, results))
        return {r["MATCH_CODE"]: self.points[r["MATCH_CODE"]] for r in rows if r["MATCH_CODE"] in self.points}

    def describe(self):
        return "fixed"


class TestHelpers(unittest.TestCase):

    def test_sides_clocks_and_prod_quotes(self):
        self.assertEqual(ru._side("TEAM_A", "home"), "home")
        self.assertEqual(ru._side("TEAM_A", "away"), "away")
        self.assertEqual(ru._side("TEAM_B", ""), "away")
        self.assertIsNone(ru._side("", "home"))
        self.assertEqual(ru._clock("236"), "3:56")
        self.assertEqual(ru._clock(""), "")
        row = dict(prob_50="0.6", live_50="1", prob_52="0.45", line_52="-3.5", live_52="0", prob_54="",
                   prematch_prob_50="0.55", prematch_prob_52="0.5", prematch_line_52="-1.5")
        self.assertEqual(ru._prod(row), {"50": dict(p=0.6, line=None, live=True),
                                         "52": dict(p=0.45, line=-3.5, live=False)})
        self.assertEqual(ru._prod(row, "prematch_"), {"50": dict(p=0.55, line=None, live=True),
                                                      "52": dict(p=0.5, line=-1.5, live=True)})


class TestReplay(unittest.TestCase):

    def test_meta_and_a_day_s_matches(self):
        rp = replay()
        meta = rp.meta()
        self.assertTrue(meta["nudge"])
        self.assertEqual((meta["mode"], meta["source_kind"]), ("tilt", "Export"))
        self.assertEqual((meta["first_day"], meta["last_day"]), ("2026-10-01", "2026-10-01"))
        day = rp.matches("2026-10-01")
        self.assertEqual(len(day), 40)
        self.assertEqual(day[0]["start"], "00:00")
        self.assertEqual(rp.matches("2026-10-02"), [])
        bare = replay(history=False)                       # no history: the export's matches
        self.assertEqual([m["code"] for m in bare.matches("2026-10-01")], sorted(SETUP["matches"])[:2])
        self.assertFalse(replay("plain").meta()["nudge"])

    def test_a_match_loaded_its_timeline_and_the_states_v21_prices(self):
        code = sorted(SETUP["matches"])[0]
        rows = SETUP["matches"][code]
        view = replay().load(code)
        self.assertEqual(len(view["timeline"]), len(rows))
        self.assertEqual(view["final"], [int(rows[-1]["final_p1"]), int(rows[-1]["final_p2"])])
        self.assertEqual(view["prematch"]["54"], dict(p=0.5, line=30.5, live=True))
        self.assertEqual((view["prior"], view["prior_note"]), (None, None))
        priced = [t for t in view["timeline"] if t["priced"]]
        self.assertEqual(view["n_priced"], len(priced))
        self.assertGreater(len(priced), len(rows) // 2)
        kinds = {t["play"] for t in view["timeline"] if not t["priced"]}
        self.assertIn("TOUCHDOWN", kinds)                  # a conversion still to come: no v21 state
        first = view["timeline"][0]
        self.assertEqual((first["period"], first["clock"], first["offense"]), (1, "3:56", "home"))
        self.assertEqual(first["prod"]["54"]["live"], False)
        with self.assertRaises(ValueError):
            replay().load("NOPE")

    def test_the_prior_s_points_read_before_the_match(self):
        code = sorted(SETUP["matches"])[1]
        prior = FixedPrior({code: (21.456, 17.0)})
        view = replay(prior=prior).load(code)
        self.assertEqual(view["prior"], [21.46, 17.0])
        rows, results = prior.asked[0]
        self.assertEqual(rows[0]["MATCH_CODE"], code)
        start = dt.datetime.fromisoformat(view["start"])
        self.assertTrue(all(dt.datetime.fromisoformat(r["SCHEDULED_START_TIME_UTC"]) < start for r in results))
        other = sorted(SETUP["matches"])[0]
        self.assertEqual(replay(prior=prior).load(other)["prior_note"], "Not in prior")

    def test_prices_at_the_kick_off_a_state_and_none_where_v21_has_no_state(self):
        rp = replay()
        code = sorted(SETUP["matches"])[0]
        view = rp.load(code)
        kick = rp.price(code, None, [20.0, 14.0], 0.75)
        self.assertTrue(kick["priced"])
        self.assertEqual(kick["prod"], view["prematch"])
        ml = kick["moneyline"]
        self.assertAlmostEqual(ml["home"] + ml["away"], 1.0, delta=0.06)  # less the tie's share
        self.assertEqual(len(kick["margin_dist"]["p"]), 2 * ru.MARGIN_SHOWN + 1)
        self.assertEqual(len(kick["total_dist"]["p"]), ru.TOTAL_SHOWN + 1)
        self.assertEqual(len(kick["spread"]["ladder"]), 2 * ru.SPREAD_LADDER + 1)
        self.assertEqual(set(kick["at_prod"]), {"50", "52", "54"})
        self.assertAlmostEqual(kick["expected"]["home"] - kick["expected"]["away"], kick["expected"]["margin"],
                               places=1)
        state = next(t for t in view["timeline"] if t["priced"] and t["message"] > 100)
        p = rp.price(code, state["message"], [20.0, 14.0], 0.75)
        self.assertTrue(p["priced"])
        self.assertEqual(p["prod"], state["prod"])
        self.assertEqual(p["at_prod"]["50"], p["moneyline"]["home"])
        self.assertEqual(p, rp.price(code, state["message"], [20.0, 14.0], 0.75))   # the same draws again
        lost = next(t for t in view["timeline"] if not t["priced"])
        q = rp.price(code, lost["message"], [20.0, 14.0], 0.75)
        self.assertFalse(q["priced"])
        self.assertEqual(q["why"], "No v21 state")
        with self.assertRaises(ValueError):
            rp.price(code, 99999)

    def test_the_nudge_moves_the_kick_off_toward_the_points_by_the_share(self):
        rp = replay(paths=300)
        code = sorted(SETUP["matches"])[0]
        own = rp.price(code, None, None)["nudge"]
        self.assertIsNone(own["alpha"])
        self.assertEqual(own["own"], own["nudged"])
        target = [own["own"][0] + 6.0, own["own"][1] - 3.0]
        full = rp.price(code, None, target, 1.0)["nudge"]
        half = rp.price(code, None, target, 0.5)["nudge"]
        self.assertGreater(full["alpha"][0], 0)
        self.assertLess(full["alpha"][1], 0)
        self.assertLess(abs(full["nudged"][0] - target[0]), 2.0)
        self.assertLess(abs(half["nudged"][0] - (own["own"][0] + 3.0)), 2.0)
        self.assertLess(half["alpha"][0], full["alpha"][0])
        self.assertEqual(rp.price(code, None, target, 0.0)["nudge"]["alpha"], None)   # share 0: no nudge
        up = rp.price_all(code, target, 1.0, 200)["points"]
        down = rp.price_all(code, [own["own"][0] - 6.0, own["own"][1] + 3.0], 1.0, 200)["points"]
        self.assertGreater(up[0]["home"], down[0]["home"])
        self.assertGreater(sum(p["margin"] for p in up), sum(p["margin"] for p in down))
        plain = replay("plain")                            # no strength direction: v21's own means
        n = plain.price(code, None, target, 1.0)["nudge"]
        self.assertFalse(n["available"])
        self.assertIsNone(n["alpha"])
        self.assertEqual(n["own"], n["nudged"])

    def test_the_whole_match_priced_at_once(self):
        rp = replay()
        code = sorted(SETUP["matches"])[0]
        view = rp.load(code)
        out = rp.price_all(code, [20.0, 14.0], 0.75, 100)
        pts = out["points"]
        self.assertEqual(len(pts), view["n_priced"] + 1)
        self.assertIsNone(pts[0]["message"])
        self.assertEqual([p["message"] for p in pts[1:]], [t["message"] for t in view["timeline"] if t["priced"]])
        self.assertTrue(all(0 <= p["home"] <= 1 for p in pts))
        last = pts[-1]                                     # late on, the margin is near the scoreboard's
        row = next(t for t in view["timeline"] if t["message"] == last["message"])
        self.assertLess(abs(last["margin"] - (row["home"] - row["away"])), 10)


class TestSources(unittest.TestCase):

    def test_snowflake_reads_the_match_s_window_and_puts_the_cutoffs_back(self):
        seen = {}

        def snaps(cur, codes, with_handles=False):
            seen.update(codes=codes, handles=with_handles, window=(config.CUTOFF_START, config.CUTOFF_END))
            return {codes[0]: [{"message": "7"}]}, {}

        was = config.CUTOFF_START, config.CUTOFF_END
        src = ru.SnowflakeSource(object(), SETUP["history"])
        with mock.patch.object(snowflake_io, "_play_over_snapshots", snaps):
            got = src.snapshots("AF001", dt.datetime(2026, 10, 1, 13, 0))
        self.assertEqual(got, [{"message": "7"}])
        self.assertEqual(seen, dict(codes=["AF001"], handles=True,
                                    window=("2026-09-30 13:00:00", "2026-10-03 13:00:00")))
        self.assertEqual((config.CUTOFF_START, config.CUTOFF_END), was)
        self.assertEqual(len(src.history()), 40)


class TestApi(unittest.TestCase):

    def test_respond(self):
        rp = replay()
        code = sorted(SETUP["matches"])[0]
        self.assertEqual(ru.respond(rp, "/api/meta", {})[0], 200)
        status, got = ru.respond(rp, "/api/matches", {"day": "2026-10-01"})
        self.assertEqual((status, len(got["matches"])), (200, 40))
        self.assertEqual(ru.respond(rp, "/api/load", {})[0], 400)
        self.assertEqual(ru.respond(rp, "/api/load", {"code": f" {code} "})[1]["code"], code)
        status, kick = ru.respond(rp, "/api/price", {"code": code, "means": ["20", 14], "share": 0.5})
        self.assertEqual(kick["nudge"]["target"], [20.0, 14.0])
        status, kick = ru.respond(rp, "/api/price", {"code": code, "means": [20, ""]})
        self.assertIsNone(kick["nudge"]["target"])         # a side left empty: no expected points
        status, out = ru.respond(rp, "/api/price_all", {"code": code, "chart_paths": 50})
        self.assertEqual(out["paths"], 50)
        self.assertEqual(ru.respond(rp, "/api/nope", {"code": code})[0], 404)

    def test_the_page_and_the_calls_over_http(self):
        rp = replay()
        server = ThreadingHTTPServer(("127.0.0.1", 0), ru.handler_for(rp))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"

        def post(path, body):
            req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        try:
            with urllib.request.urlopen(base + "/") as r:
                page = r.read().decode()
            self.assertIn("<title>", page)
            self.assertIn("/api/price", page)
            with urllib.request.urlopen(base + "/api/meta") as r:
                self.assertTrue(json.loads(r.read())["nudge"])
            status, got = post("/api/load", {"code": "NOPE"})
            self.assertEqual(status, 400)
            self.assertIn("no PLAY_OVERs", got["error"])
            status, got = post("/api/load", {"code": sorted(SETUP["matches"])[0]})
            self.assertEqual(status, 200)
            self.assertGreater(len(got["timeline"]), 10)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
