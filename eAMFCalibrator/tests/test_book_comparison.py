"""The book comparison: acceptance by segment, the two-book test and the intervals."""

import unittest
from unittest import mock

from .. import book_comparison as br, bets, config, markets


def row(temp="Standard", odds=2.0, cand_odds=2.0, stake=10.0, result=bets.LOST, match="M1",
        market=markets.MONEYLINE, same_line=True, in_play=True):
    won = result == bets.WON
    return {config.BET_VIP_COLUMN: temp, "odds": odds, "candidate_odds": cand_odds, "stake": stake,
            "result": result, "match_code": match, "market": market, "same_line": same_line,
            "in_play": in_play, "simulated": True,
            "revenue": stake - stake * odds if won else stake,
            "candidate_revenue": stake - stake * cand_odds if won else stake}


class TestAcceptance(unittest.TestCase):

    def test_segments(self):
        self.assertEqual(br.segment(row("Restricted")), br.SHARP)
        self.assertEqual(br.segment(row("VIP")), br.RECREATIONAL)
        with mock.patch.object(config, "SIM_SHARP_GROUPS", ("Restricted", "VIP")):
            self.assertEqual(br.segment(row("VIP")), br.SHARP)

    def test_a_sharp_bets_only_while_an_edge_is_left(self):
        sharp = row("Restricted", odds=2.0, cand_odds=1.8)
        self.assertEqual(br.scale(sharp, 0.05), 0.0)                       # 1.05 / 2 x 1.8 < 1
        self.assertAlmostEqual(br.scale(row("Restricted", cand_odds=2.0), 0.05), 1.0)
        self.assertAlmostEqual(br.scale(row("Restricted", cand_odds=2.1), 0.05), (1.05 * 1.05 - 1) / 0.05)
        self.assertEqual(br.scale(row("Restricted", cand_odds=3.0), 0.05), config.SIM_MAX_SCALE)

    def test_everyone_else_bets_as_placed_unless_given_an_elasticity(self):
        self.assertEqual(br.scale(row("VIP", cand_odds=1.5), 0.05), 1.0)
        with mock.patch.object(config, "SIM_ELASTICITY", 1.0):
            self.assertAlmostEqual(br.scale(row("VIP", cand_odds=1.5), 0.05), 0.75)

    def test_the_edge_is_the_bettors_return_shrunk_toward_the_segment(self):
        rows = [row("Restricted", result=bets.WON)] * 3 + [row("Restricted")] \
            + [row("Restricted", market=markets.TOTAL)] * 4
        e = br.edges(rows, prior=0)
        self.assertAlmostEqual(e[(br.SHARP, markets.MONEYLINE)], (3 * 10 - 10) / 40)    # +50%
        self.assertAlmostEqual(e[(br.SHARP, markets.TOTAL)], -1.0)
        self.assertAlmostEqual(e[(br.SHARP, None)], (30 - 10 - 40) / 80)
        shrunk = br.edges(rows, prior=4)[(br.SHARP, markets.MONEYLINE)]
        self.assertAlmostEqual(shrunk, (4 * 0.5 + 4 * -0.25) / 8)

    def test_a_dropped_bet_leaves_the_candidate(self):
        rows = [row("Restricted", cand_odds=1.5, result=bets.WON), row("VIP", cand_odds=1.5)]
        acc = br.accepted(rows, {(br.SHARP, markets.MONEYLINE): 0.05})
        self.assertEqual([(s, v) for _, s, v in acc], [(0.0, 0.0), (10.0, 10.0)])


class TestTwoBooks(unittest.TestCase):

    def test_the_bettor_takes_the_better_price_on_the_same_line(self):
        rows = [row(cand_odds=2.2, result=bets.WON), row(cand_odds=1.8), row(cand_odds=2.5, same_line=False)]
        t = br.two_books(rows)
        self.assertEqual([bk for _, bk, _, _ in t], ["candidate", "prod"])
        self.assertAlmostEqual(t[0][3], 10 - 22.0)


class TestIntervals(unittest.TestCase):

    def test_margins_and_a_paired_difference(self):
        a = {f"M{i}": [10.0, 1.0] for i in range(30)}
        c = {f"M{i}": [10.0, 2.0] for i in range(30)}
        b = br.boot({"a": a, "c": c}, n=200)
        self.assertAlmostEqual(b["a"][0], 10.0)
        self.assertAlmostEqual(b["a"][1], 10.0)                         # no spread: every match alike
        d = br.diff(b, "a", "c")
        self.assertEqual(d, (10.0, 10.0, 10.0))

    def test_the_report_runs(self):
        rows = [row("Restricted", cand_odds=2.1, result=bets.WON, match=f"M{i}") for i in range(5)] \
            + [row("VIP", cand_odds=1.9, match=f"M{i}") for i in range(5)]
        with mock.patch.object(config, "SIM_BOOT", 50):
            text = "\n".join(br.report([("v10", rows)]))
        self.assertIn("== v10 ==", text)
        self.assertIn("two books", text)
        self.assertIn("sharp", text)
        with mock.patch.object(config, "SIM_BOOT", 50):
            page = br.html(br.compute([("v10", rows)]), lambda x: str(x))
        self.assertIn("<h3>Book comparison</h3>", page)
        self.assertEqual(page.count("<table"), 2)
        self.assertIn("<th>Candidate takes</th>", page)
        self.assertEqual(br.html([], str), "")


if __name__ == "__main__":
    unittest.main()
