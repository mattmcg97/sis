"""Bets split by the pre-match favourite, the score at the bet and the side backed."""

import datetime as dt
import unittest

from .. import bet_favourite as bf, bets, config, markets

T0 = dt.datetime(2026, 9, 15, 12)


def bet(market=1, selection=1, stake=10.0, revenue=10.0, odds=1.9, in_play="Yes", period=1,
        temp="Standard", code="M1"):
    return bets.Bet("x", code, T0, market, selection, odds, stake, revenue, period=period,
                    extra={config.BET_VIP_COLUMN: temp, config.BET_IN_PLAY_COLUMN: in_play})


def row(b, message=20, prob=0.6, result=bets.LOST, on_line=True):
    return dict(match_code=b.match_code, feed_market=b.feed_market, market=markets.market_group(b.feed_market),
                message=message, stream_prob=prob, on_prod_line=on_line, odds=b.odds, stake=b.stake,
                revenue=b.revenue, result=result, _bet=b)


# prod's moneyline on M1: P1 at 62% pre-match (no message), then in play
TL = {("M1", 50): ([T0, T0], [(None, 0.62, None, True), (20, 0.75, None, True)]),
      ("M2", 50): ([T0], [(None, 0.51, None, True)])}
# M1: P2 scores 3 at message 10, P1 scores 7 at message 15
SCORES = {"M1": ([10, 15], [(0, 3), (7, 3)])}


class TestFavourite(unittest.TestCase):

    def test_the_favourite_is_prods_last_prematch_moneyline(self):
        self.assertEqual(bf.prematch_favourite(TL, "M1"), (True, 0.62))
        self.assertIsNone(bf.prematch_favourite(TL, "M2"))           # within 2% of evens
        self.assertIsNone(bf.prematch_favourite(TL, "M3"))
        tl = {("M4", 50): ([T0], [(5, 0.30, None, True)])}            # never quoted pre-match
        self.assertEqual(bf.prematch_favourite(tl, "M4"), (False, 0.70))

    def test_phase_state_and_side(self):
        self.assertEqual(bf.phase_of(bet(in_play="No")), "pre-match")
        self.assertEqual(bf.phase_of(bet(period=2)), "Q2")
        self.assertEqual(bf.phase_of(bet(period=5)), "OT")
        self.assertIsNone(bf.phase_of(bet(period=None)))
        self.assertEqual([bf.state_of(x) for x in (-10, -3, 0, 7, 9)], list(bf.STATES))
        self.assertEqual(bf.side_of(50, True), bf.FAV)
        self.assertEqual(bf.side_of(51, True), bf.DOG)
        self.assertEqual(bf.side_of(52, False), bf.DOG)
        self.assertEqual(bf.side_of(53, False), bf.FAV)
        self.assertEqual(bf.side_of(55, True), bf.UNDER)

    def test_a_bet_is_tagged_with_the_lead_it_saw(self):
        rows = bf.tag([row(bet(selection=2, temp="Restricted"), message=16, prob=0.25, result=bets.WON),
                       row(bet(), message=12),
                       row(bet(in_play="No"), message=None),
                       row(bet(code="M2"))], TL, SCORES)
        self.assertEqual(len(rows), 3)                                # M2 has no favourite
        dog, early, pre = rows
        self.assertEqual((dog["state"], dog["fav_lead"], dog["side"], dog["group"]),
                         ("fav up 1-8", 4, bf.DOG, "Restricted"))
        self.assertEqual((early["state"], early["fav_lead"]), ("fav down 1-8", -3))
        self.assertEqual((pre["phase"], pre["state"]), ("pre-match", "level"))
        self.assertAlmostEqual(dog["expected_revenue"], 10.0 * (1 - 1.9 * 0.25))
        self.assertEqual(dog["won"], 1.0)

    def test_the_cell_sums_margin_and_expected_margin(self):
        rows = bf.tag([row(bet(stake=10.0, revenue=10.0), prob=0.5),
                       row(bet(stake=30.0, revenue=-27.0), prob=0.5, result=bets.WON),
                       row(bet(stake=5.0, revenue=5.0), prob=None)], TL, SCORES)
        c = bf.cell(rows)
        self.assertEqual(c[:4], [3, 45.0, 45.0, -12.0])
        self.assertAlmostEqual(c[5] / c[4], 1 - 1.9 * 0.5)            # priced: the first two
        self.assertAlmostEqual(c[7] / c[8], 0.75)                      # won 30 of 40 staked
        text = "\n".join(bf.report(rows, min_bets=1))
        self.assertIn("Q1 / fav up 1-8", text)
        self.assertIn("The favourite ahead in Q1 and Q2", text)


if __name__ == "__main__":
    unittest.main()


class TestRun(unittest.TestCase):

    def test_the_command_runs_off_the_fetches(self):
        import contextlib
        import io
        import os
        import tempfile
        from unittest import mock
        from .. import snowflake_io
        c = config.BET_COLUMNS
        cols = [c["id"], c["match"], c["time"], c["market_type"], c["selection"], c["odds"], c["stake"],
                c["revenue"], c["line"], c["period"], config.BET_VIP_COLUMN, config.BET_IN_PLAY_COLUMN]
        raw = [(1, "M1", T0 + dt.timedelta(minutes=5), 1, 2, 3.0, 10.0, 10.0, None, 1, "Restricted", "Yes"),
               (2, "M1", T0 - dt.timedelta(minutes=5), 1, 1, 1.6, 20.0, -12.0, None, 0, "VIP", "No")]
        quotes = [("M1", 50, T0 - dt.timedelta(minutes=30), 62.0, 1.6, "PLAYER 1 to win", None, "OPEN", "true"),
                  ("M1", 50, T0 + dt.timedelta(minutes=4), 75.0, 1.3, "PLAYER 1 to win", 20, "OPEN", "true"),
                  ("M1", 51, T0 + dt.timedelta(minutes=4), 25.0, 4.0, "PLAYER 2 to win", 20, "OPEN", "true")]
        scores = [("M1", 15, 1, None, 7, 7, 0)]
        with tempfile.TemporaryDirectory() as out, \
                mock.patch.object(bets, "fetch_all", return_value=(cols, raw)), \
                mock.patch.object(snowflake_io, "fetch_quotes", return_value=quotes), \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 10)}), \
                mock.patch.object(snowflake_io, "fetch_scores", return_value=scores), \
                contextlib.redirect_stdout(io.StringIO()) as printed:
            path = bf.run(None, out, min_bets=1)
            self.assertTrue(os.path.exists(path))
            self.assertTrue(os.path.exists(os.path.join(out, "bets_favourite.txt")))
        self.assertIn("2 bets on matches with a pre-match favourite", printed.getvalue())
        self.assertIn("Q1 / fav up 1-8", printed.getvalue())
