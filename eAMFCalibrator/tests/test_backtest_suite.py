"""The backtest suite: its settings, the audit of the daily fits and the gamers' stats (each catching
a planted leak), the layers' scramble check (catching one), and a run end to end with its controls."""

import csv
import datetime as dt
import json
import os
import tempfile
import unittest
from unittest import mock

from .. import (backtest_suite as bs, config, feature_layer, gamer_features, prior_backtest as pb,
                prior_replay as pr, rematch_layer, snowflake_io)
from . import test_prior_backtest as tpb

T0 = tpb.T0
SINCE = "2026-09-01 00:00:00"


class TestSettings(unittest.TestCase):

    def test_defaults_a_file_and_the_arms(self):
        cfg = bs.load_config()
        self.assertEqual(cfg["nb2_half_lives"], [30, 60, 120])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w") as fh:
                json.dump({"nb2_half_lives": [45], "glmer": "", "arms": {"x": "x.csv#normal"}}, fh)
            cfg = bs.load_config(path)
            self.assertEqual(bs.arm_list(cfg, "out"),
                             [("nb2-hl45", f"rolling:{os.path.join('out', 'nb2_hl45')}"), ("x", "x.csv#normal")])
            with open(path, "w") as fh:
                json.dump({"half_lives": [45]}, fh)
            with self.assertRaises(SystemExit):
                bs.load_config(path)
        with self.assertRaises(SystemExit):                            # a control's name
            bs.arm_list(dict(cfg, arms={"echo": "e.csv"}), "out")
        self.assertEqual(bs.window_days("2026-09-01 00:00:00", "2026-10-15 00:00:00"),
                         (dt.date(2026, 9, 1), dt.date(2026, 10, 14)))


def write_day(directory, day, starts, half_life=30, level_starts=()):
    """A prior-daily NB2 fit for a day, as nb2_prior writes one: level.json and the training files."""
    d = os.path.join(directory, day.isoformat())
    os.makedirs(os.path.join(d, "level"), exist_ok=True)
    with open(os.path.join(d, "level.json"), "w") as fh:
        json.dump({"before": dt.datetime.combine(day, dt.time()).isoformat(), "half_life_days": half_life,
                   "scale": 1.0, "league_average": [17, 17]}, fh)
    for path, ts in ((os.path.join(d, "history.csv"), starts), (os.path.join(d, "level", "history.csv"), level_starts)):
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["MATCH_CODE", "SCHEDULED_START_TIME_UTC"])
            w.writerows([f"m{k}", str(t)] for k, t in enumerate(ts))


class TestAuditFits(unittest.TestCase):

    def test_a_fit_on_its_own_days_results_fails(self):
        first, last = dt.date(2026, 9, 10), dt.date(2026, 9, 12)
        with tempfile.TemporaryDirectory() as d:
            for k in range(3):
                day = first + dt.timedelta(days=k)
                before = dt.datetime.combine(day, dt.time())
                starts = [before - dt.timedelta(hours=h) for h in (1, 30)]
                if k == 1:
                    starts.append(before + dt.timedelta(hours=2))      # planted: a result from the day itself
                write_day(d, day, starts, level_starts=[before - dt.timedelta(days=8)])
            with open(os.path.join(d, "rolling.json"), "w") as fh:
                json.dump({"prior": "nb2", "options": {"half_life": 30},
                           "days": [(first + dt.timedelta(days=k)).isoformat() for k in range(3)]}, fh)
            audit = bs.Audit()
            with mock.patch("builtins.print"):
                bs.audit_rolling(audit, "nb2-hl30", d, first, last + dt.timedelta(days=1), 30)
            text = "\n".join(audit.lines)
            self.assertIn("PASS  nb2-hl30: fitted at its half-life", text)
            self.assertIn("FAIL  nb2-hl30: every fit on results before its day: 1 not: 2026-09-11", text)
            self.assertIn("WARN  nb2-hl30: a fit for every day: 1 days without one", text)   # the 13th
            audit = bs.Audit()
            with mock.patch("builtins.print"):
                bs.audit_rolling(audit, "nb2-hl60", d, first, last, 60)
            self.assertEqual(audit.count["FAIL"], 3)                   # the setting, the leak, the half-life
            audit = bs.Audit()
            with mock.patch("builtins.print"):
                bs.audit_rolling(audit, "nb2-hl30", os.path.join(d, "none"), first, last, 30)
            self.assertIn("run the fits step", audit.lines[0])


    def test_a_fit_on_an_older_history_fails(self):
        day = dt.date(2026, 9, 10)
        before = dt.datetime.combine(day, dt.time())
        history = sorted(before - dt.timedelta(minutes=10 * k) for k in range(1, 201))
        with tempfile.TemporaryDirectory() as d:
            write_day(d, day, history[:190])                           # the fit missed the last 10 results
            with open(os.path.join(d, "rolling.json"), "w") as fh:
                json.dump({"prior": "nb2", "options": {"half_life": 30}, "days": [day.isoformat()]}, fh)
            audit = bs.Audit()
            with mock.patch("builtins.print"):
                bs.audit_rolling(audit, "nb2-hl30", d, day, day, 30, history)
            self.assertIn("FAIL  nb2-hl30: every fit on all the results before its day: 1 fitted on an older "
                          "history (results seen of those now known): 2026-09-10 (190 of 200)", "\n".join(audit.lines))
            write_day(d, day, history[:199])                           # one late result is within the 1%
            audit = bs.Audit()
            with mock.patch("builtins.print"):
                bs.audit_rolling(audit, "nb2-hl30", d, day, day, 30, history)
            self.assertIn("PASS  nb2-hl30: every fit on all the results before its day", "\n".join(audit.lines))


class TestHistoryAndFits(unittest.TestCase):

    def test_a_history_csv_is_taken_and_only_the_days_it_covers_are_fitted(self):
        rows = [dict(MATCH_CODE=f"m{k}", SPORT_CODE="AF", STREAM_NUMBER="1",
                     SCHEDULED_START_TIME_UTC=f"2026-09-{1 + k // 4:02d} {10 + k % 4}:00:00", PLAYER_1_HANDLE="A",
                     PLAYER_1_TEAM="X", PLAYER_2_HANDLE="B", PLAYER_2_TEAM="Y", PLAYER_1_FINAL_SCORE="21",
                     PLAYER_2_FINAL_SCORE="14") for k in range(40)]                  # 1 to 10 Sep
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "mine.csv")
            with open(src, "w", newline="") as fh:
                w = csv.DictWriter(fh, snowflake_io.HISTORY_COLUMNS)
                w.writeheader()
                w.writerows(rows)
            out = os.path.join(d, "suite")
            os.makedirs(out)
            with mock.patch("builtins.print"):
                bs.step_history(None, out, src)                        # no Snowflake: cur None
            self.assertTrue(os.path.exists(os.path.join(out, bs.HISTORY)))
            from eAMFModel import rolling_prior
            calls = []
            with mock.patch.object(rolling_prior, "fit", lambda h, prior, directory, first, last, **kw:
                                   calls.append((directory, first, last, kw["options"]))), \
                    mock.patch("builtins.print"):
                bs.step_fits(dict(bs.load_config(), nb2_half_lives=[30, 60]), out, dt.date(2026, 9, 5),
                             dt.date(2026, 9, 20))
            self.assertEqual([c[1:] for c in calls], [(dt.date(2026, 9, 5), dt.date(2026, 9, 10), {"half_life": 30}),
                                                      (dt.date(2026, 9, 5), dt.date(2026, 9, 10), {"half_life": 60})])
            with mock.patch("builtins.print"), self.assertRaises(SystemExit):
                bs.step_fits(bs.load_config(), out, dt.date(2026, 9, 12), dt.date(2026, 9, 20))


class TestQueryCache(unittest.TestCase):

    class Cursor:
        def __init__(self):
            self.runs = 0
            self.description = [("A",), ("B",)]

        def execute(self, sql, params):
            self.runs += 1

        def fetchall(self):
            return [(1, dt.datetime(2026, 9, 1, 12, 0)), (2, None)]

    def test_a_query_is_asked_once_and_read_back_after(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(config, "FETCH_DISK_CACHE", d):
            cur = self.Cursor()
            first = snowflake_io.fetch_all(cur, "SELECT A, B\n  FROM T WHERE X = %s", (1,))
            again = snowflake_io.fetch_all(cur, "SELECT A, B FROM T   WHERE X = %s", (1,))   # spacing aside
            self.assertEqual(cur.runs, 1)
            self.assertEqual(first, again)
            self.assertEqual(snowflake_io.fetch_all(None, "SELECT A, B FROM T WHERE X = %s", (1,)), first)
            snowflake_io.fetch_all(cur, "SELECT A, B FROM T WHERE X = %s", (2,))              # other params
            self.assertEqual(cur.runs, 2)
            with self.assertRaises(SystemExit):                                                 # not cached, no cursor
                snowflake_io.fetch_all(None, "SELECT A FROM T", ())
        cur = self.Cursor()                                                                     # cache off
        snowflake_io.fetch_all(cur, "SELECT 1", ())
        snowflake_io.fetch_all(cur, "SELECT 1", ())
        self.assertEqual(cur.runs, 2)

    def test_the_suite_connects_only_when_a_query_needs_it(self):
        from .. import __main__ as cli
        with mock.patch.object(snowflake_io, "get_connection") as connect:
            cur = cli._LazyCursor()
            cur.close()
            self.assertFalse(connect.called)
            cur = cli._LazyCursor()
            cur.execute("SELECT 1", ())
            self.assertTrue(connect.called and cur.opened)
            cur.close()


class TestAuditFeatures(unittest.TestCase):

    def stats(self, d):
        hist, _, arm = tpb.fixture()
        path = os.path.join(d, "stats.csv")
        tpb.write_stats(path, hist, arm)
        window = [r for r in hist]
        return path, window

    def test_the_daily_reading_passes_and_a_leaky_one_is_caught(self):
        first, last = T0.date(), T0.date() + dt.timedelta(days=23)
        with tempfile.TemporaryDirectory() as d:
            path, window = self.stats(d)
            audit = bs.Audit()
            with mock.patch("builtins.print"):
                bs.audit_features(audit, path, window, first, last)
            text = "\n".join(audit.lines)
            self.assertIn("PASS  the gamers' stats: nothing from a later day", text)
            self.assertIn("PASS  the gamers' stats cover the window", text)
            self.assertIn("WARN  the gamers' stats carry the reconciliation counts", text)
            real = gamer_features.daily

            def leaky(rows, half_life=50, prior_matches=20, since=None):
                out = real(rows, half_life, prior_matches, since)       # every match also read the whole file
                total = sum(float(r["points"]) for r in rows)
                for f in out.values():
                    if f["home"]["points_for"] is not None:
                        f["home"]["points_for"] += total * 1e-9
                return out

            audit = bs.Audit()
            with mock.patch.object(gamer_features, "daily", leaky), mock.patch("builtins.print"):
                bs.audit_features(audit, path, window, first, last)
            self.assertIn("FAIL  the gamers' stats: nothing from a later day", "\n".join(audit.lines))
            audit = bs.Audit()
            with mock.patch("builtins.print"):
                bs.audit_features(audit, os.path.join(d, "none.csv"), window, first, last)
            self.assertIn("run player-stats", audit.lines[0])


def bet_rows(hist, by):
    """Pre-match bets on every match from day 8: a moneyline 3 minutes before kick-off at prod's
    closing price, and a total 4 minutes before at its closing line."""
    out = []
    for h in hist:
        start = dt.datetime.fromisoformat(h["SCHEDULED_START_TIME_UTC"])
        if start < T0 + dt.timedelta(days=8):
            continue
        s1, s2 = int(h["PLAYER_1_FINAL_SCORE"]), int(h["PLAYER_2_FINAL_SCORE"])
        p = next(q[3] / 100 for q in by[h["MATCH_CODE"]] if q[1] == 50 and q[6] == 1)
        line = float(next(q[5] for q in by[h["MATCH_CODE"]] if q[1] == 54 and q[6] == 1).split()[1])
        for t, mt, odds, won, ln in ((start - dt.timedelta(minutes=3), 1, 1 / (p * 1.05),
                                      None if s1 == s2 else s1 > s2, None),
                                     (start - dt.timedelta(minutes=4), 3, 1 / (0.5 * 1.05), s1 + s2 > line, line)):
            stake, odds = 10.0, round(odds, 4)
            revenue = stake - stake * odds if won is True else 0.0 if won is None else stake
            out.append((len(out), h["MATCH_CODE"], t, mt, 1, odds, stake, revenue, ln, 1, "OPX", "No",
                        "VIP" if len(out) % 5 == 0 else "Standard"))
    return out


BET_COLS = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID", "ODDS", "STAKE_GBP",
            "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER", "OPERATOR_NAME", "BET_IN_PLAY",
            "CUSTOMER_TEMPERATURE"]


class Fake:
    """The fixture's data behind every fetch the suite makes."""

    def __init__(self, d):
        self.hist, self.by, arm = tpb.fixture()
        self.arm = os.path.join(d, "arm.csv")
        with open(self.arm, "w", encoding="utf-8") as fh:
            fh.write("\n".join(arm) + "\n")
        self.stats = os.path.join(d, "player_match_stats.csv")
        tpb.write_stats(self.stats, self.hist, arm)
        self.bets = bet_rows(self.hist, self.by)
        self.patches = [
            mock.patch.object(snowflake_io, "fetch_history", lambda cur: self.hist),
            mock.patch.object(snowflake_io, "fetch_quotes",
                              lambda cur, s, codes: [q for c in codes for q in self.by.get(c, [])]),
            mock.patch.object(snowflake_io, "fetch_first_plays", lambda cur, codes: {}),
            mock.patch.object(pr, "fetch_all", lambda cur, sql, params: (BET_COLS, self.bets)),
            mock.patch.object(feature_layer, "MIN_TRAIN", 60),
            mock.patch.object(config, "MATCH_CHUNK_SIZE", 200),
            mock.patch("builtins.print"),
        ]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


class TestLeakCheck(unittest.TestCase):

    def test_the_layers_pass_and_a_peeking_layer_is_caught(self):
        with tempfile.TemporaryDirectory() as d, Fake(d) as f:
            ctx = pb.prepare(None, SINCE, features=f.stats)
            audit = bs.Audit()
            bs.leak_check(audit, ctx, "model", f.arm)
            self.assertEqual(audit.count["FAIL"], 0, audit.lines)
            self.assertEqual(audit.count["PASS"], 3)
            self.assertIn("later ones changed", audit.lines[0])
            real = rematch_layer.walk_forward

            def peeking(items, burn_in_days=7):                 # every move nudged by the window's mean total
                rows, layer = real(items, burn_in_days)
                done = [it["actual_total"] for it in items if it["actual_total"] is not None]
                for r in rows:
                    r["move_margin"] += 1e-6 * sum(done) / len(done)
                return rows, layer

            audit = bs.Audit()
            ctx = pb.prepare(None, SINCE, features=f.stats)
            with mock.patch.object(rematch_layer, "walk_forward", peeking):
                bs.leak_check(audit, ctx, "model", f.arm)
            self.assertIn("FAIL  model: the layers read nothing from a later day", audit.lines[0])


class TestRun(unittest.TestCase):

    def test_the_suite_end_to_end(self):
        with tempfile.TemporaryDirectory() as d, Fake(d) as f:
            out = os.path.join(d, "suite")
            os.makedirs(out)
            with open(os.path.join(out, bs.HISTORY), "w", newline="") as fh:
                w = csv.DictWriter(fh, snowflake_io.HISTORY_COLUMNS, extrasaction="ignore")
                w.writeheader()
                w.writerows(dict(h, SPORT_CODE="AF", STREAM_NUMBER="1") for h in f.hist)
            cfg = dict(bs.load_config(), nb2_half_lives=[], glmer="", arms={"model": f.arm}, features=f.stats,
                       draws=2000, boot=50,
                       periods=[["first fortnight", None, "2026-09-15"], ["then", "2026-09-15", None]])
            path = bs.run(None, out, cfg, SINCE, "2026-09-25 00:00:00",
                          steps=["audit", "backtest", "replay", "summary"])
            with open(path) as fh:
                text = fh.read()
            with open(os.path.join(out, bs.AUDIT)) as fh:
                audit = fh.read()
            with open(os.path.join(out, bs.MANIFEST)) as fh:
                man = json.load(fh)
            with open(os.path.join(out, "suite_summary.csv"), newline="") as fh:
                ranked = list(csv.DictReader(fh))
            # the independent grading agrees with prior-backtest's own
            variants, per, rows = bs.grade_backtest(os.path.join(out, "prior_backtest.csv"),
                                                    dt.datetime.fromisoformat(man["burn_from"]))
        self.assertNotIn("FAIL", audit)
        for check in ("the gamers' stats: nothing from a later day", "model: the layers read nothing from a later day",
                      "model: the layers' refit is the same (determinism)", "model: prices the window"):
            self.assertIn(f"PASS  {check}", audit)
        self.assertIn("PASS  echo grades as prod", text)
        self.assertIn("PASS  shuffled grades below model", text)
        for name in ("model+both+feat", "first fortnight", "The controls", "The audit: ", "total"):
            self.assertIn(name, text)
        self.assertEqual(set(man["steps"]), {"audit", "backtest", "replay", "summary"})
        self.assertEqual(man["audit"]["FAIL"], 0)
        names = [r["variant"] for r in ranked]
        self.assertEqual(names[0], "prod")
        self.assertNotIn("echo", names[1:-2])                           # the controls come after the ranking
        ranked_only = [r for r in ranked if r["rank"] not in ("",)]
        self.assertEqual([float(r["ml_vs_prod"]) for r in ranked_only],
                         sorted(float(r["ml_vs_prod"]) for r in ranked_only))
        self.assertTrue(all(r["bets_change"] != "" for r in ranked_only))
        self.assertEqual(len(variants), 1 + 6 * 3)                      # prod; model, echo, shuffled with each layer
        g = pb.grade([r["MATCH_CODE"] for r in rows],
                     {v: {r["MATCH_CODE"]: (float(r[f"{v}_p_home"]), float(r[f"{v}_total"]),
                                            None if r[f"{v}_p_over"] == "" else float(r[f"{v}_p_over"]))
                          for r in rows} for v in ("prod", "model")},
                     {r["MATCH_CODE"]: (int(r["margin"]), int(r["total"]), float(r["prod_line"])) for r in rows})
        model = next(r for r in ranked if r["variant"] == "model")
        self.assertAlmostEqual(float(model["ml_ll"]), pb._mean(g["model"]["ll"]), places=6)
        self.assertAlmostEqual(float(model["ml_vs_prod"]), pb._paired(g["prod"]["ll"], g["model"]["ll"])[0], places=6)

    def test_an_unknown_step_and_a_failed_audit_stop(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = dict(bs.load_config(), nb2_half_lives=[], glmer="", arms={"x": os.path.join(d, "missing.csv")},
                       features="none")
            with self.assertRaises(SystemExit):
                bs.run(None, d, cfg, SINCE, None, steps=["fit"])
            with mock.patch("builtins.print"), self.assertRaises(SystemExit) as stop:
                bs.run(None, d, cfg, SINCE, "2026-09-03 00:00:00", steps=["audit", "summary"])
            self.assertIn("failures", str(stop.exception))
            with open(os.path.join(d, bs.AUDIT)) as fh:
                text = fh.read()
            self.assertIn("FAIL  the history", text)
            self.assertIn("FAIL  x: source", text)


if __name__ == "__main__":
    unittest.main()
