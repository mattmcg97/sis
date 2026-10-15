"""The pre-match bet replay: an arm's chance of each selection at its line, prod's moved by the rematch
layer, the bets placed before a gamer's last result was in, and a run on fake bets."""

import csv
import datetime as dt
import os
import random
import tempfile
import unittest
from statistics import NormalDist
from unittest import mock

import numpy as np

from .. import config, prior_replay as pr, rematch_layer, score_model, snowflake_io

T0 = dt.datetime(2026, 9, 1, 18, 0)


class TestChances(unittest.TestCase):

    def test_each_selection_at_its_line(self):
        b = score_model.Book(np.array([10, 7, 3, 14]), np.array([7, 7, 10, 0]))   # margins 3, 0, -7, 14
        self.assertAlmostEqual(pr.arm_chance(b, 50, None), 2 / 3)                # the tie is a push
        self.assertAlmostEqual(pr.arm_chance(b, 51, None), 1 / 3)
        self.assertAlmostEqual(pr.arm_chance(b, 52, -2.5), 3 / 4)
        self.assertAlmostEqual(pr.arm_chance(b, 52, 3), 1 / 3)                   # the 3 pushes
        self.assertAlmostEqual(pr.arm_chance(b, 53, 2.5), 1 / 4)                 # the away margin over 2.5
        with mock.patch.object(config, "SPREAD_RESOLUTION", "complement"):
            self.assertAlmostEqual(pr.arm_chance(b, 53, -2.5), 1 / 4)            # the NO of 52 at -2.5
        self.assertAlmostEqual(pr.arm_chance(b, 54, 13.5), 3 / 4)                # totals 17, 14, 13, 14
        self.assertAlmostEqual(pr.arm_chance(b, 55, 14), 1 / 2)                  # the 14s push
        self.assertIsNone(pr.arm_chance(b, 54, None))
        self.assertAlmostEqual(pr.arm_chance(b, 50, None, dm=-3.5), 1 / 4)       # the layer's move
        self.assertAlmostEqual(pr.arm_chance(b, 54, 16.5, dt_=1.0), 1 / 4)

    def test_the_normal_reading(self):
        n = score_model.Normal(0.0, 35.0)
        self.assertAlmostEqual(pr.arm_chance(n, 50, None), 0.5)
        self.assertGreater(pr.arm_chance(n, 52, -2.5), 0.5)
        self.assertAlmostEqual(pr.arm_chance(n, 52, -2.5) + pr.arm_chance(n, 53, 2.5), 1.0)
        self.assertAlmostEqual(pr.arm_chance(n, 54, 35.0), 0.5)
        self.assertLess(pr.arm_chance(n, 55, 30.0), 0.5)

    def test_prod_moved_by_the_layer(self):
        self.assertAlmostEqual(pr.moved_chance(0.6, 50), 0.6)
        up = pr.moved_chance(0.6, 50, dm=2.0)
        self.assertGreater(up, 0.6)
        self.assertAlmostEqual(up + pr.moved_chance(0.4, 51, dm=2.0), 1.0)
        # prior-backtest's prod+layer: the margin its moneyline implies, moved
        self.assertAlmostEqual(up, rematch_layer.win_chance(rematch_layer.implied_margin(0.6) + 2.0))
        self.assertAlmostEqual(pr.moved_chance(0.5, 54, dt_=2.0), NormalDist().cdf(2.0 / 12.0))
        self.assertAlmostEqual(pr.moved_chance(0.5, 55, dt_=2.0), 1 - NormalDist().cdf(2.0 / 12.0))
        self.assertIsNone(pr.moved_chance(None, 50))


class TestTiming(unittest.TestCase):

    def test_a_bet_before_the_last_result_was_in(self):
        hist = [dict(MATCH_CODE="AF1", SCHEDULED_START_TIME_UTC=str(T0), PLAYER_1_HANDLE="a", PLAYER_2_HANDLE="c",
                     PLAYER_1_FINAL_SCORE="21", PLAYER_2_FINAL_SCORE="14")]
        kick = pr.previous_kickoffs(hist)
        meet = dict(home="A", away="B", start=T0 + dt.timedelta(minutes=40))
        self.assertFalse(pr.results_in(meet, T0 + dt.timedelta(minutes=30), kick))      # AF1's not in until 36
        self.assertTrue(pr.results_in(meet, T0 + dt.timedelta(minutes=37), kick))
        self.assertTrue(pr.results_in(dict(meet, home="D"), T0 + dt.timedelta(minutes=30), kick))
        self.assertFalse(pr.results_in(meet, None, kick))
        self.assertEqual([pr.minute_band(m) for m in (2, 10, 30, 90, None)],
                         ["under 5 min", "5-15 min", "15-60 min", "an hour or more", "unknown"])

    def test_the_candidates_quote_where_prod_did_before_kick_off(self):
        t1 = T0 + dt.timedelta(minutes=1)
        tl = {("AF1", 50): ([T0, t1], [(None, 0.6, None, True), (5, 0.62, None, True)]),
              ("AF2", 50): ([T0], [(None, 0.5, None, True)])}
        a, b = pr.priced_timelines(tl, {"AF1"}, lambda m, k, line, p: (0.9999, p / 2), 2)
        self.assertEqual(a, {("AF1", 50): ([T0], [(None, 1 - pr.PROB_FLOOR, None, True)])})   # held; in play out
        self.assertEqual(b, {("AF1", 50): ([T0], [(None, 0.3, None, True)])})


class TestRun(unittest.TestCase):

    def test_the_bets_replayed(self):
        rnd = random.Random(5)
        hist, quotes, arm, bet_rows = [], [], ["MATCH_CODE,exp_p1,exp_p2"], []
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID", "ODDS",
                "STAKE_GBP", "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER", "OPERATOR_NAME",
                "BET_IN_PLAY", "CUSTOMER_TEMPERATURE"]

        def bet(code, t, mt, sel, odds, won, line=None, in_play="No"):
            stake, odds = 10.0, round(odds, 4)
            revenue = stake - stake * odds if won is True else 0.0 if won is None else stake
            bet_rows.append((len(bet_rows), code, t, mt, sel, odds, stake, revenue, line, 1, "OPX",
                             in_play, "VIP" if len(bet_rows) % 5 == 0 else "Standard"))
        k = 0
        for d in range(24):
            for pair in range(6):
                a, b = f"A{pair}", f"B{pair}"
                um, ut = rnd.gauss(0, 4), rnd.gauss(0, 5)
                for j in range(3):
                    code = f"AF{k:05d}"
                    start = T0 + dt.timedelta(days=d, minutes=40 * j + pair)
                    mu1, mu2 = 17 + rnd.gauss(0, 2), 17 + rnd.gauss(0, 2)
                    s1 = max(0, round(mu1 + um / 2 + ut / 2 + rnd.gauss(0, 7)))
                    s2 = max(0, round(mu2 - um / 2 + ut / 2 + rnd.gauss(0, 7)))
                    hist.append(dict(MATCH_CODE=code, SCHEDULED_START_TIME_UTC=str(start), PLAYER_1_HANDLE=a,
                                     PLAYER_2_HANDLE=b, PLAYER_1_TEAM="Ravens", PLAYER_2_TEAM="Lions",
                                     PLAYER_1_FINAL_SCORE=str(s1), PLAYER_2_FINAL_SCORE=str(s2)))
                    p = rematch_layer.win_chance(mu1 - mu2)
                    line = round(mu1 + mu2) + 0.5
                    pre = start - dt.timedelta(minutes=10)
                    quotes += [(code, 50, pre, 100 * p, None, None, None, "open", "true"),
                               (code, 51, pre, 100 * (1 - p), None, None, None, "open", "true"),
                               (code, 54, pre, 50.0, None, f"Line {line}", None, "open", "true"),
                               (code, 55, pre, 50.0, None, f"Line {line}", None, "open", "true"),
                               (code, 50, start + dt.timedelta(minutes=1), 100 * p, None, None, 3, "open", "true")]
                    arm.append(f"{code},{mu1:.3f},{mu2:.3f}")
                    if d >= 8:
                        bet(code, start - dt.timedelta(minutes=3), 1, 1, 1 / (p * 1.05),
                            None if s1 == s2 else s1 > s2)
                        bet(code, start - dt.timedelta(minutes=8), 3, 1, 1 / (0.5 * 1.05),
                            s1 + s2 > line, line=line)
                        bet(code, start + dt.timedelta(minutes=2), 1, 2, 1 / ((1 - p) * 1.05),
                            None if s1 == s2 else s2 > s1, in_play="Yes")
                        if j == 0:                                # flagged pre-match, placed in play
                            bet(code, start + dt.timedelta(minutes=1), 1, 1, 1 / (p * 1.05),
                                None if s1 == s2 else s1 > s2)
                    k += 1
        by = {}
        for q in quotes:
            by.setdefault(q[0], []).append(q)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "arm.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(arm) + "\n")
            with mock.patch.object(snowflake_io, "fetch_history", lambda cur: hist), \
                    mock.patch.object(snowflake_io, "fetch_quotes",
                                      lambda cur, s, codes: [q for c in codes for q in by.get(c, [])]), \
                    mock.patch.object(pr, "fetch_all", lambda cur, sql, params: (cols, bet_rows)), \
                    mock.patch.object(config, "MATCH_CHUNK_SIZE", 200), mock.patch("builtins.print"):
                out = pr.run(None, d, [("model", path)], "2026-09-01 00:00:00", None, draws=2000, n_boot=50)
            with open(out, encoding="utf-8") as fh:
                text = fh.read()
            with open(os.path.join(d, "prior_replay.csv"), newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
        for name in ("prod+layer", "model ", "model+layer"):
            self.assertIn(name, text)
        for part in ("Clean bets", "Clean bets, moneyline", "Clean bets, total", "by meeting", "Every pre-match bet",
                     "The rematch layers' last fits"):
            self.assertIn(part, text)
        n_matches = 16 * 18                                   # every match from day 8, two pre-match bets each
        self.assertEqual(len(rows), 2 * n_matches)            # the in-play bets left out, and the late ones
        self.assertIn(f"left out: {n_matches // 3:,} bets flagged pre-match but placed at or after kick-off", text)
        clean = [r for r in rows if r["clean"] == "True"]
        # the moneyline bets 3 minutes before kick-off are clean; the totals 8 minutes before are clean
        # only on a pair's first game of the night (the game before's result is in 36 minutes after it)
        self.assertEqual(sum(r["market"] == "moneyline" for r in clean), n_matches)
        self.assertEqual(sum(r["market"] == "total" for r in clean), n_matches // 3)
        for r in rows:
            p, odds = float(r["stream_prob"]), float(r["odds"])
            for name in ("prod+layer", "model", "model+layer"):
                self.assertAlmostEqual(float(r[f"{name}_odds"]), odds * p / float(r[f"{name}_prob"]), places=3)
            if r["meeting"] == "1st":
                self.assertAlmostEqual(float(r["prod+layer_revenue"]), float(r["revenue"]), places=6)
                self.assertEqual(r["model_odds"], r["model+layer_odds"])     # no earlier meeting: no move


if __name__ == "__main__":
    unittest.main()
