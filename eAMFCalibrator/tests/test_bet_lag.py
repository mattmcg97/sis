"""Tests for bet_lag: the lag read match by match off the lines and the odds."""

import datetime as dt
import os
import random
import tempfile
import unittest

from .. import bet_lag, bets

T0 = dt.datetime(2026, 9, 20, 20, 0, 0)
GRID = list(range(-10, 31))


def _timeline(match, moves):
    """A total (market 54) whose line moves at these seconds, each move one step up, and a flat 50%
    moneyline (50): the shape bets.timeline returns."""
    out = {}
    times, quotes = [], []
    line = 40.5
    t = 0
    for k in range(0, 600, 5):
        if k in moves:
            line += 1
        times.append(T0 + dt.timedelta(seconds=k))
        quotes.append((k, 0.5, line, True))
    out[(match, 54)] = (times, quotes)
    out[(match, 50)] = ([T0], [(1, 0.5, None, True)])
    return out


def _bet(match, second, line, op="OP", period=1, odds=1.9, market_type=3, selection=1):
    return bets.Bet(bet_id=f"{match}{second}", match_code=match, time=T0 + dt.timedelta(seconds=second),
                    market_type=market_type, selection=selection, odds=odds, stake=10.0, revenue=0.0,
                    line=line, period=period, extra={"OPERATOR_NAME": op, "BET_IN_PLAY": "Yes"})


class TestLag(unittest.TestCase):
    def setUp(self):
        self.old = bets.config.BET_GROUP_COLUMN
        bets.config.BET_GROUP_COLUMN = "OPERATOR_NAME"

    def tearDown(self):
        bets.config.BET_GROUP_COLUMN = self.old

    def world(self):
        """Two matches, their operator 9s and 3s behind prod; bets every 2s around each line move."""
        rng = random.Random(1)
        tl, all_bets = {}, []
        for match, lag in (("AF001200926", 9), ("AF002200926", 3)):
            moves = (100, 250, 400)
            tl.update(_timeline(match, moves))
            step = bet_lag.line_steps(tl)[(match, 54)]
            for second in range(60, 560, 2):
                line = bet_lag.line_at(step, T0 + dt.timedelta(seconds=second - lag))
                all_bets.append(_bet(match, second, line))
            all_bets += [_bet(match, s, None, market_type=1, odds=1.9 + 0.01 * rng.random())
                         for s in range(60, 560, 20)]
        return tl, all_bets

    def test_each_match_reads_its_own_lag_off_the_lines(self):
        tl, all_bets = self.world()
        bl = bet_lag.per_bet(all_bets, bet_lag.line_steps(tl), tl, {}, GRID)
        ms = bet_lag.per_match(bl, GRID, {"OP": 5})
        self.assertEqual(ms[("AF001200926", "OP")].best, 9)
        lo, hi = ms[("AF002200926", "OP")].plateau
        self.assertTrue(lo <= 3 <= hi and hi - lo <= 1)
        m = ms[("AF001200926", "OP")]
        lo, hi = max(m.lower), min(m.upper)
        self.assertTrue(lo <= 9 <= hi and hi - lo <= 1)
        self.assertAlmostEqual(m.agree[GRID.index(9)], 1.0)
        self.assertLess(m.agree[GRID.index(5)], 1.0)

    def test_held_out_a_lag_per_match_beats_one_lag(self):
        tl, all_bets = self.world()
        bl = bet_lag.per_bet(all_bets, bet_lag.line_steps(tl), tl, {}, GRID)
        n, one, own = bet_lag.held_out(bl, GRID, {"OP": 5})["OP"]
        self.assertGreater(n, 400)
        self.assertAlmostEqual(own, 1.0)
        self.assertLess(one, own)

    def test_a_bet_on_the_old_line_after_a_move_says_at_least(self):
        tl = _timeline("M", (100,))
        step = bet_lag.line_steps(tl)[("M", 54)]
        b = bet_lag.per_bet([_bet("M", 107, 40.5)], {("M", 54): step}, tl, {}, GRID)[0]
        self.assertEqual(bet_lag._bounds(b, GRID), (8, None))
        b = bet_lag.per_bet([_bet("M", 103, 41.5)], {("M", 54): step}, tl, {}, GRID)[0]
        self.assertEqual(bet_lag._bounds(b, GRID), (None, 3))

    def test_the_summary_csv_and_pages(self):
        tl, all_bets = self.world()
        bl = bet_lag.per_bet(all_bets, bet_lag.line_steps(tl), tl, {}, GRID)
        ms = bet_lag.per_match(bl, GRID, {"OP": 5})
        lags = {"OP": bets.Lag(5, 10)}
        lines = bet_lag.summary(ms, bl, GRID, lags, bet_lag.held_out(bl, GRID, {"OP": 5}))
        text = "\n".join(lines)
        self.assertIn("one lag per match?", text)
        with tempfile.TemporaryDirectory() as d:
            bet_lag.write_csv(os.path.join(d, "bets_lag.csv"), ms, GRID, {"OP": 5})
            paths = bet_lag.write_pages(d, ms, bl, tl, GRID, {"OP": 5}, lines)
            self.assertEqual([os.path.basename(p) for p in paths], ["bets_lag.html", "bets_lag_2026-09-20.html"])
            with open(paths[1], encoding="utf-8") as fh:
                page = fh.read()
            self.assertIn("<svg", page)
            self.assertIn("AF002200926", page)


if __name__ == "__main__":
    unittest.main()
