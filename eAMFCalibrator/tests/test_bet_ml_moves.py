"""The moneyline movement audit: prod's price either side of every moneyline bet."""

import os
import tempfile
import unittest
from unittest import mock

from .. import bet_ml_moves as mm, bets, config
from .test_bets import MONEYLINE, PRICES, _checks_for, at, bet, odds_for, quote


class TestPath(unittest.TestCase):

    def test_prod_s_chance_of_the_side_either_side(self):
        tl = bets.timeline(MONEYLINE)                     # home 0.40 at 0, 0.60 at 100, 0.45 at 200, 0.70 at 300, 0.50 at 400
        b = bet("M1", 150, odds=odds_for(0.6))
        r = bets.join([b], {}, {}, tl, {}, None, {"M1": (21, 17)})[0]
        p = mm.path(b, r, tl)
        self.assertAlmostEqual(p["price"], 0.60)
        self.assertEqual((p["backs"], p["price_band"], p["side"]), ("favourite", "0.5-0.65", "Home"))
        self.assertAlmostEqual(p["move_m120"], 0.40 - 0.60)          # it shortened into the bet
        self.assertEqual(p["way_m120"], mm.FOLLOWED)
        self.assertEqual(p["way_p10"], mm.HELD)
        self.assertAlmostEqual(p["move_p60"], 0.45 - 0.60)           # and drifted after it
        self.assertEqual(p["way_p60"], mm.AGAINST)
        self.assertAlmostEqual(p["move_p300"], 0.50 - 0.60)
        away = bet("M1", 150, selection=2, odds=odds_for(0.4))
        tl2 = bets.timeline(MONEYLINE + [quote("M1", 51, t, 1 - p_) for t, p_ in PRICES])
        r2 = bets.join([away], {}, {}, tl2, {}, None, {"M1": (21, 17)})[0]
        q = mm.path(away, r2, tl2)
        self.assertEqual((q["backs"], q["way_p60"]), ("underdog", mm.FOR))    # the away side shortened after

    def test_the_bands(self):
        self.assertEqual(mm.band(0.1, mm.PRICE_BANDS), "0-0.2")
        self.assertEqual(mm.band(0.99, mm.PRICE_BANDS), "0.8-1.01")
        self.assertEqual(mm.band(3000, mm.STAKE_BANDS), "2000+")
        self.assertIsNone(mm.band(None, mm.STAKE_BANDS))


class TestRun(unittest.TestCase):

    def test_the_run_writes_the_report(self):
        from .. import snowflake_io
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER",
                "OPERATOR_NAME", "BET_IN_PLAY", "CUSTOMER_TEMPERATURE", "CUSTOMER_NAME_HASH"]
        rows = [(i, "M1", at(c + 7 + k), 1, 1, odds_for(p), 10 + k, 10 if k % 2 else -9, None, 2, "FANDUEL", "Yes",
                 "Restricted" if k % 3 == 0 else "Standard", f"C{k % 4}")
                for k in range(12) for i, (c, p) in enumerate(PRICES[1:], start=100 * k)]
        rows += [(9000 + k, "M1", at(-60 + k), 1, 1, odds_for(0.4), 5, 5, None, 0, "FANDUEL", "No", "VIP", "C9")
                 for k in range(40)]
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(bets, "fetch_all", return_value=(cols, rows)), \
                mock.patch.object(snowflake_io, "fetch_quotes", return_value=MONEYLINE), \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 17)}), \
                mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.object(config, "LAG_RANGE", (-20, 30)), mock.patch("builtins.print"):
            paths = mm.run(None, d, min_bets=5)
            with open(paths[0], encoding="utf-8") as fh:
                text = fh.read()
            self.assertTrue(os.path.exists(paths[1]))
        for part in ("MONEYLINE MOVES", "IN PLAY", "where the money came", "by the customer temperature",
                     "prod's price either side of the bet", "the book's margin by what prod did in the 60s after",
                     "THE CUSTOMERS WITH THE MOST MONEYLINE STAKE"):
            self.assertIn(part, text)


if __name__ == "__main__":
    unittest.main()
