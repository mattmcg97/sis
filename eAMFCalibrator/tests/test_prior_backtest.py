"""The prior backtest: reading each arm's expected points, prod's pre-match total, and a run graded
against prod with and without the rematch layer."""

import csv
import datetime as dt
import os
import random
import tempfile
import unittest
from unittest import mock

from .. import bets, config, feature_layer, prior_backtest as pb, snowflake_io

T0 = dt.datetime(2026, 9, 1, 18, 0)


class TestRead(unittest.TestCase):

    def write(self, d, name, text):
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_every_source_shape(self):
        with tempfile.TemporaryDirectory() as d:
            cad = self.write(d, "cadence_matches.csv",
                             "MatchId,Time,P1,P2,S1,S2,GameOfDay,trim:hl60:game|1,trim:hl60:game|2\n"
                             "AF1,2026-09-01 18:00:00,A,B,21,14,1,17.5,15.25\n")
            nb2 = self.write(d, "nb2.csv", "MATCH_CODE,Pred_P1_Points,Pred_P2_Points\nAF1,16,18\n")
            pvs = self.write(d, "pvs.csv", "MATCH_CODE,raw_home,raw_away,means_home\nAF1,15,15.5,9\n")
            bad = self.write(d, "bad.csv", "MATCH_CODE,x\nAF1,1\n")
            self.assertEqual(pb.read_points(f"{cad}@trim:hl60:game", [], []), {"AF1": (17.5, 15.25)})
            self.assertEqual(pb.read_points(nb2, [], []), {"AF1": (16.0, 18.0)})
            self.assertEqual(pb.read_points(pvs, [], []), {"AF1": (15.0, 15.5)})
            with self.assertRaises(SystemExit):
                pb.read_points(bad, [], [])

    def test_an_nb2_csv_carries_its_own_score_model(self):
        with tempfile.TemporaryDirectory() as d:
            nb2 = self.write(d, "nb2.csv", "Match Id,Pred_P1_Points,Pred_P2_Points,P1_Effective_Alpha,"
                                           "P2_Effective_Alpha,Shared_Gamma_Fraction\nAF1,16,18,0.12,0.1,0.45\n")
            plain = self.write(d, "plain.csv", "MATCH_CODE,exp_p1,exp_p2\nAF1,16,18\n")
            points, own = pb.read_arm(nb2, [], [])
            self.assertEqual(points, {"AF1": (16.0, 18.0)})
            self.assertEqual(own, {"AF1": ("nb2", (16.0, 18.0, 0.12, 0.1, 0.45))})
            self.assertEqual(pb.read_arm(plain, [], [])[1], {})

    def test_an_arms_mode(self):
        self.assertEqual(pb.parse_arm("x.csv@trim:hl60:game#fitted"), ("x.csv@trim:hl60:game", "fitted"))
        self.assertEqual(pb.parse_arm("rolling:d"), ("rolling:d", None))
        with self.assertRaises(SystemExit):
            pb.parse_arm("x.csv#poisson")

    def test_each_mode_prices_a_match(self):
        r = dict(match="AF1", start=T0, margin=3.0, total=36.0, move_margin=1.0, move_total=-2.0,
                 margin_adj=4.0, total_adj=34.0, actual_margin=7, actual_total=35)
        prod = {"AF1": {"p1": 0.6, "line": 35.5, "over": 0.5}}
        normal, _ = pb.readings("m", [r], prod, "normal")
        own, _ = pb.readings("m", [r], prod, "own", {"AF1": ("nb2", (19.5, 16.5, 0.1, 0.1, 0.4))}, draws=20000)
        for got in (normal, own):
            p, t, po = got["AF1"]["m"]
            p2, t2, po2 = got["AF1"]["m+layer"]
            self.assertTrue(0.5 < p < p2 < 1)                  # the layer adds a point to the margin
            self.assertEqual((t, t2), (36.0, 34.0))
            self.assertGreater(po, po2)                         # and takes two off the total
        self.assertNotAlmostEqual(normal["AF1"]["m"][0], own["AF1"]["m"][0], places=3)
        p_prod = pb.readings(pb.PROD, [r], prod)[0]["AF1"]
        self.assertEqual(p_prod[pb.PROD], (0.6, 35.5, 0.5))

    def test_prods_closing_quote_is_the_last_before_the_first_play(self):
        def q(minutes, p, msg, line=None, market=50):
            return ("AF1", market, T0 + dt.timedelta(minutes=minutes), 100.0 * p, None,
                    None if line is None else f"Line {line}", msg, "open", "true")
        tl = bets.timeline([q(-60, 0.5, None), q(-5, 0.6, 1), q(-1, 0.62, 2), q(2, 0.9, 4),
                            q(-60, 0.5, None, 40.5, 54), q(-5, 0.55, 1, 38.5, 54), q(2, 0.7, 4, 38.5, 54)])
        self.assertEqual(pb.prematch_p1(tl, "AF1"), 0.5)                                 # as read before
        self.assertEqual(pb.prematch_p1(tl, "AF1", first_play=3), 0.62)                  # below the first play
        self.assertEqual(pb.prematch_p1(tl, "AF1", first_play=2), 0.6)
        self.assertEqual(pb.prematch_p1(tl, "AF1", kickoff=T0), 0.62)                    # no play feed: kick-off
        self.assertEqual(pb.prematch_total(tl, "AF1", 3, T0), (38.5, 0.55))
        self.assertEqual(pb.prematch_total(tl, "AF1"), (40.5, 0.5))
        pre = bets.closing_timeline(tl, {"AF1": 3}, {})
        self.assertEqual(pb.prematch_p1(pre, "AF1"), 0.62)                               # messages dropped
        self.assertEqual(pre[("AF1", 50)][1][-1], (None, 0.62, None, True))
        self.assertEqual(len(pre[("AF1", 50)][0]), 3)                                    # the in-play one out

    def test_prods_closing_total_and_its_over(self):
        def q(line, p, msg=None):
            return ("AF1", 54, T0, 100.0 * p, None, f"Line {line}", msg, "open", "true")
        tl = bets.timeline([q(36.5, 0.48), q(37.5, 0.51), q(40.5, 0.6, msg=9)])
        self.assertEqual(pb.prematch_total(tl, "AF1"), (37.5, 0.51))
        self.assertEqual(pb.prematch_total(tl, "AF2"), (None, None))


def fixture():
    """(history, prod's quotes by match, the arm's CSV lines): six pairs meeting three times a night
    for 24 nights, prod opening at evens and closing on a pre-game message."""
    rnd = random.Random(4)
    hist, quotes, arm = [], [], ["MATCH_CODE,exp_p1,exp_p2"]
    k = 0
    for d in range(24):
        for pair in range(6):
            a, b = f"A{pair}", f"B{pair}"
            um, ut = rnd.gauss(0, 4), rnd.gauss(0, 5)             # each pair's day
            for j in range(3):
                code = f"AF{k:05d}"
                start = T0 + dt.timedelta(days=d, minutes=40 * j + pair)
                mu1, mu2 = 17 + rnd.gauss(0, 2), 17 + rnd.gauss(0, 2)
                s1 = max(0, round(mu1 + um / 2 + ut / 2 + rnd.gauss(0, 7)))
                s2 = max(0, round(mu2 - um / 2 + ut / 2 + rnd.gauss(0, 7)))
                hist.append(dict(MATCH_CODE=code, SCHEDULED_START_TIME_UTC=str(start), PLAYER_1_HANDLE=a,
                                 PLAYER_2_HANDLE=b, PLAYER_1_TEAM="Ravens", PLAYER_2_TEAM="Lions",
                                 PLAYER_1_FINAL_SCORE=str(s1), PLAYER_2_FINAL_SCORE=str(s2)))
                p = pb.rematch_layer.win_chance(mu1 - mu2)
                opening, closing = start - dt.timedelta(minutes=60), start - dt.timedelta(minutes=5)
                line = round(mu1 + mu2) + 0.5
                # prod opens at evens with no message, closes on a pre-game message, then goes in play
                quotes += [(code, 50, opening, 50.0, None, None, None, "open", "true"),
                           (code, 54, opening, 50.0, None, f"Line {line + 5}", None, "open", "true"),
                           (code, 50, closing, 100 * p, None, None, 1, "open", "true"),
                           (code, 54, closing, 50.0, None, f"Line {line}", 1, "open", "true"),
                           (code, 50, start + dt.timedelta(minutes=2), 90.0, None, None, 5, "open", "true")]
                arm.append(f"{code},{mu1:.3f},{mu2:.3f}")
                k += 1
    return hist, defaultdict_list(quotes), arm


class TestRun(unittest.TestCase):

    def test_prod_and_an_arm_each_with_the_layer(self):
        hist, by, arm = fixture()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "arm.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(arm) + "\n")
            with mock.patch.object(snowflake_io, "fetch_history", lambda cur: hist), \
                    mock.patch.object(snowflake_io, "fetch_quotes",
                                      lambda cur, s, codes: [q for c in codes for q in by.get(c, [])]), \
                    mock.patch.object(snowflake_io, "fetch_first_plays",     # half by the play feed, half by kick-off
                                      lambda cur, codes: {c: 3 for c in codes if int(c[2:]) % 2 == 0}), \
                    mock.patch.object(config, "MATCH_CHUNK_SIZE", 200):
                out = pb.run(None, d, [("model", path)], "2026-09-01 00:00:00", None)
                ctx = pb.prepare(None, "2026-09-01 00:00:00")
                with open(out, encoding="utf-8") as fh:
                    text = fh.read()
                with open(os.path.join(d, "prior_backtest.csv"), newline="", encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
        for arm_name in ("prod ", "model ", "model+layer", "model+infer", "model+both"):
            self.assertIn(arm_name, text)
        for gone in ("prod+layer", "prod-early"):                  # prod is a third party's: the benchmark only
            self.assertNotIn(gone, text)
        # prod read at its closing quote (a pre-game message), not its opening one (no message)
        closing = next(q[3] / 100 for q in by["AF00000"] if q[1] == 50 and q[6] == 1)
        self.assertAlmostEqual(ctx["prod"]["AF00000"]["p1"], closing)
        self.assertEqual(ctx["early"]["AF00000"]["p1"], 0.5)
        # +both: a rematch moves as the rematch layer does, a first meeting as the inference layer does
        for r in rows:
            twin = "model+infer" if r["meeting"] == "1st" else "model+layer"
            self.assertEqual(r["model+both_p_home"], r[f"{twin}_p_home"])
            self.assertEqual(r["model+both_total"], r[f"{twin}_total"])
        self.assertIn("Rematches", text)
        self.assertIn("The layers' last fits", text)
        self.assertIn("NB2's construction fitted to its errors", text)
        self.assertNotIn("+feat", text)                            # no stats: no feature layer

    def test_with_the_gamers_stats_the_feature_layer(self):
        hist, by, arm = fixture()
        with tempfile.TemporaryDirectory() as d:
            path, stats = os.path.join(d, "arm.csv"), os.path.join(d, "player_match_stats.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(arm) + "\n")
            write_stats(stats, hist, arm)
            with mock.patch.object(snowflake_io, "fetch_history", lambda cur: hist), \
                    mock.patch.object(snowflake_io, "fetch_quotes",
                                      lambda cur, s, codes: [q for c in codes for q in by.get(c, [])]), \
                    mock.patch.object(snowflake_io, "fetch_first_plays", lambda cur, codes: {}), \
                    mock.patch.object(feature_layer, "MIN_TRAIN", 60), \
                    mock.patch.object(config, "MATCH_CHUNK_SIZE", 200), mock.patch("builtins.print"):
                out = pb.run(None, d, [("model", path)], "2026-09-01 00:00:00", None, features=stats)
                with open(out, encoding="utf-8") as fh:
                    text = fh.read()
                missing = pb.run(None, os.path.join(d, "again"), [("model", path)], "2026-09-01 00:00:00", None,
                                 features=os.path.join(d, "nothing.csv"))
                with open(missing, encoding="utf-8") as fh:
                    text_missing = fh.read()
        for name in ("model+feat", "model+both+feat", "feature layer, on the arm alone (+feat)",
                     "feature layer, on +both (+both+feat)", "points_for"):
            self.assertIn(name, text)
        self.assertNotIn("prod+feat", text)                         # the benchmark is not layered
        self.assertNotIn("+feat", text_missing)                     # no stats file: no feature layer

    def test_the_feature_layer_on_a_rows_errors(self):
        hist, by, arm = fixture()
        with tempfile.TemporaryDirectory() as d:
            path, stats = os.path.join(d, "arm.csv"), os.path.join(d, "player_match_stats.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(arm) + "\n")
            write_stats(stats, hist, arm)
            with mock.patch.object(snowflake_io, "fetch_history", lambda cur: hist), \
                    mock.patch.object(snowflake_io, "fetch_quotes",
                                      lambda cur, s, codes: [q for c in codes for q in by.get(c, [])]), \
                    mock.patch.object(snowflake_io, "fetch_first_plays", lambda cur, codes: {}), \
                    mock.patch.object(feature_layer, "MIN_TRAIN", 60), \
                    mock.patch.object(config, "MATCH_CHUNK_SIZE", 200), mock.patch("builtins.print"):
                ctx = pb.prepare(None, "2026-09-01 00:00:00", features=stats)
                _, _, rows, layers = pb.fit_arm("model", path, ctx)
                _, _, prod_rows, prod_layers = pb.fit_arm(pb.PROD, None, ctx)
        self.assertIn("feature", layers)
        self.assertNotIn("feature", prod_layers)
        first_day = [r for r in rows if r["start"].date() == T0.date()]
        self.assertTrue(first_day and all(r["feat_margin"] == 0.0 and r["feat_total"] == 0.0 for r in first_day))
        self.assertTrue(any(r["feat_margin"] != 0.0 for r in rows))
        for r in rows:                                              # +both+feat is +both and the stacked fit
            if r["start"].date() == T0.date():
                self.assertEqual((r["bf_margin"], r["bf_total"]), (r["both_margin"], r["both_total"]))


def write_stats(path, hist, arm):
    """A player_match_stats.csv for the fixture's matches: each side's yards a play off its expected
    points (what the arm already knows) and its scoring (what it doesn't)."""
    mus = {line.split(",")[0]: tuple(map(float, line.split(",")[1:])) for line in arm[1:]}
    cols = ["match_code", "start", "side", "player", "opponent", "points", "points_against", "scouting", "snaps",
            "detail_snaps", "plays", "yards", "yards_known", "opp_yards", "opp_yards_known", "third_conv",
            "third_att", "rz_td", "rz_trips", "turnovers", "tod", "top_s"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for h in hist:
            mu1, mu2 = mus[h["MATCH_CODE"]]
            s1, s2 = int(h["PLAYER_1_FINAL_SCORE"]), int(h["PLAYER_2_FINAL_SCORE"])
            for side, me, them, mu, opp_mu, pts, against in (
                    ("home", h["PLAYER_1_HANDLE"], h["PLAYER_2_HANDLE"], mu1, mu2, s1, s2),
                    ("away", h["PLAYER_2_HANDLE"], h["PLAYER_1_HANDLE"], mu2, mu1, s2, s1)):
                w.writerow([h["MATCH_CODE"], h["SCHEDULED_START_TIME_UTC"], side, me, them, pts, against, 1, 40,
                            40, 40, round(40 * (4 + mu / 4)), 40, round(40 * (4 + opp_mu / 4)), 40, 4, 8, 2, 3,
                            1, 0, 480])


def defaultdict_list(quotes):
    out = {}
    for q in quotes:
        out.setdefault(q[0], []).append(q)
    return out


if __name__ == "__main__":
    unittest.main()
