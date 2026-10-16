"""NB2's historical pass: each gamer's matches before the day, every match priced by its own day's
fit with NB2's own score model, only the days the history covers fitted, and the month-by-month
grading."""

import csv
import datetime as dt
import json
import os
import tempfile
import unittest
from unittest import mock

from .. import nb2_history as nh, prior_backtest as pb


def hist_row(code, start, p1, p2, s1, s2, t1="Bills", t2="Jets"):
    return dict(MATCH_CODE=code, SPORT_CODE="AF", STREAM_NUMBER="1", SCHEDULED_START_TIME_UTC=start,
                PLAYER_1_HANDLE=p1, PLAYER_1_TEAM=t1, PLAYER_2_HANDLE=p2, PLAYER_2_TEAM=t2,
                PLAYER_1_FINAL_SCORE="" if s1 is None else str(s1), PLAYER_2_FINAL_SCORE="" if s2 is None else str(s2))


HISTORY = [hist_row("m1", "2026-01-01 10:00:00", "A", "B", 21, 14),
           hist_row("m2", "2026-01-01 10:40:00", "A", "B", 10, 17),
           hist_row("m3", "2026-01-02 09:00:00", "a", "C", 24, 24),
           hist_row("m4", "2026-01-02 09:30:00", "B", "C", 3, 28),
           hist_row("m5", "2026-01-03 12:00:00", "A", "C", None, None)]          # not settled


def write_rolling(d, days):
    os.makedirs(d, exist_ok=True)
    for day in days:
        os.makedirs(os.path.join(d, day), exist_ok=True)
        with open(os.path.join(d, day, "level.json"), "w") as fh:
            json.dump({"before": f"{day}T00:00:00", "fitted_on": 100, "scale": 1.02,
                       "league_average": [17, 17]}, fh)
    with open(os.path.join(d, "rolling.json"), "w") as fh:
        json.dump({"prior": "nb2", "options": {}, "days": days}, fh)


class TestPricing(unittest.TestCase):

    def test_experience_counts_the_settled_matches_before_the_day(self):
        seen = nh.experience(HISTORY)
        self.assertEqual(seen[("A", dt.date(2026, 1, 1))], 0)
        self.assertEqual(seen[("A", dt.date(2026, 1, 2))], 2)       # a's two of the 1st (any case)
        self.assertEqual(seen[("B", dt.date(2026, 1, 2))], 2)
        self.assertEqual(seen[("C", dt.date(2026, 1, 2))], 0)
        self.assertNotIn(("A", dt.date(2026, 1, 3)), seen)           # m5 isn't settled

    def test_each_match_by_its_own_days_fit_and_nb2s_score_model(self):
        settled = [r for r in HISTORY if r["PLAYER_1_FINAL_SCORE"]]
        points = {"m1": (20.0, 16.0), "m2": (19.0, 17.0), "m3": (18.0, 18.0), "m4": (14.0, 22.0)}
        own = {c: ("nb2", p + (0.15, 0.15, 0.4)) for c, p in points.items() if c != "m4"}
        with tempfile.TemporaryDirectory() as d:
            write_rolling(d, ["2026-01-01", "2026-01-02"])
            with mock.patch.object(pb, "read_rolling", lambda directory, rows, history: (points, own)):
                rows = nh.price_rows(d, settled, HISTORY, draws=4000)
        by = {r["MATCH_CODE"]: r for r in rows}
        self.assertEqual(list(by), ["m1", "m2", "m3", "m4"])
        self.assertEqual((by["m1"]["fit_day"], by["m3"]["fit_day"]), ("2026-01-01", "2026-01-02"))
        self.assertEqual((by["m2"]["meeting"], by["m2"]["home_matches_before"]), (2, 0))
        self.assertEqual(by["m3"]["home"], "A")
        self.assertEqual((by["m3"]["home_matches_before"], by["m3"]["away_matches_before"]), (2, 0))
        self.assertEqual((by["m1"]["margin"], by["m1"]["total"], by["m1"]["exp_margin"]), (7, 35, 4.0))
        for r in rows:
            self.assertAlmostEqual(r["p_home_win"] + r["p_draw"] + r["p_away_win"], 1.0, places=4)
            self.assertAlmostEqual(r["p_home"], r["p_home_win"] + r["p_draw"] / 2, places=4)
        self.assertGreater(by["m1"]["p_home"], 0.55)
        self.assertGreater(by["m1"]["p_draw"], 0)                       # NB2's own: discrete scores tie
        self.assertEqual(by["m4"]["p_draw"], 0.0)                        # no score model of its own: the normal
        self.assertIsNone(by["m4"]["P1_Effective_Alpha"])
        self.assertLess(by["m4"]["p_home"], 0.5)
        self.assertEqual((by["m1"]["fit_matches"], by["m1"]["level_scale"]), (100, 1.02))


class TestRun(unittest.TestCase):

    def test_only_the_covered_days_are_fitted_and_the_csv_reads_as_an_arm(self):
        calls = []

        def fit(history, prior, directory, first, last, workers=4, options=None):
            calls.append((first, last, options))
            write_rolling(directory, [(first + dt.timedelta(days=k)).isoformat()
                                      for k in range((last - first).days + 1)])

        points = {"m1": (20.0, 16.0), "m2": (19.0, 17.0), "m3": (18.0, 18.0), "m4": (14.0, 22.0)}
        own = {c: ("nb2", p + (0.15, 0.15, 0.4)) for c, p in points.items()}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "history.csv")
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, list(HISTORY[0]))
                w.writeheader()
                w.writerows(HISTORY)
            from eAMFModel import rolling_prior
            with mock.patch.object(rolling_prior, "fit", fit), \
                    mock.patch.object(pb, "read_rolling", lambda directory, rows, history: (
                        {c: v for c, v in points.items() if c in {r["MATCH_CODE"] for r in rows}}, own)), \
                    mock.patch("builtins.print"):
                out = nh.run(d, "2026-01-01 00:00:00", "2026-01-31 00:00:00", path, half_life=30, workers=2)
                with self.assertRaises(SystemExit):                          # wholly after the history
                    nh.run(d, "2026-02-01 00:00:00", None, path)
            with open(out, newline="") as fh:
                rows = list(csv.DictReader(fh))
            with open(os.path.join(d, nh.TXT_NAME)) as fh:
                text = fh.read()
            self.assertTrue(os.path.isdir(os.path.join(d, "nb2_history_hl30")))
            points_read, own_read = pb.read_arm(out, [], [])               # prior-backtest reads it as NB2
        self.assertEqual(calls, [(dt.date(2026, 1, 1), dt.date(2026, 1, 2), {"half_life": 30})])   # m5's day: unsettled
        self.assertEqual([r["MATCH_CODE"] for r in rows], ["m1", "m2", "m3", "m4"])
        self.assertEqual(points_read["m1"], (20.0, 16.0))
        self.assertEqual(own_read["m1"][0], "nb2")
        for part in ("2026-01", "all", "calibration"):
            self.assertIn(part, text)


if __name__ == "__main__":
    unittest.main()
