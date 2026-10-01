"""Totals bets against prod's line either side of them: the sign, the probability where the line
held, the candidate's lean, and restricted against everyone else."""

import unittest

from .. import bet_checks, bet_totals, bets, config
from .test_bets import at, bet, quote

# prod's total: 44.5 at 0 (P(over) 0.50), the price firms at 50 (0.55), the line goes to 46.5 at 100
PROD = [quote("M1", 54, 0, 0.50, 44.5, message=1), quote("M1", 55, 0, 0.50, 44.5, message=1),
        quote("M1", 54, 50, 0.55, 44.5, message=2), quote("M1", 55, 50, 0.45, 44.5, message=2),
        quote("M1", 54, 100, 0.50, 46.5, message=3), quote("M1", 55, 100, 0.50, 46.5, message=3)]
# a model already at 46.5 by message 2
CAND = [quote("M1", 54, 0, 0.50, 44.5, message=1), quote("M1", 55, 0, 0.50, 44.5, message=1),
        quote("M1", 54, 50, 0.50, 46.5, message=2), quote("M1", 55, 50, 0.50, 46.5, message=2),
        quote("M1", 54, 100, 0.50, 46.5, message=3), quote("M1", 55, 100, 0.50, 46.5, message=3)]


def checks():
    rows = [("M1", q[6], None, None, None, None, None, None, None, None) for q in PROD]
    rows.append(("M1", 0, None, "FIRST_QUARTER_STARTED", None, None, None, None, None, None))
    return bet_checks.Checks(bet_checks.build_feeds(rows), bet_checks.message_times(PROD), {})


def total(seconds, selection, temperature="Restricted", **kw):
    return bet("M1", seconds, 3, selection, line=44.5, odds=1.9,
               **{config.BET_VIP_COLUMN: temperature}, **kw)


def build(bs, cand=None):
    tl, ch = bets.timeline(PROD), checks()
    results = []
    for name, rows in ([("m", cand)] if cand else []):
        ctl = bets.timeline(rows)
        results.append((name, bets.join(bs, {}, {}, tl, bets.quote_index(rows), ctl, {"M1": (24, 21)},
                                         ch), ctl))
    base = results[0][1] if results else bets.join(bs, {}, {}, tl, {}, None, {"M1": (24, 21)}, ch)
    return bet_totals.build(bs, base, results, tl, ch)


class TestSign(unittest.TestCase):

    def test_a_line_rising_after_an_over_is_their_way_and_after_an_under_against(self):
        over, under = build([total(60, 1), total(60, 2)])
        self.assertEqual((over["level_p60"], over["way_p60"]), (2.0, bet_totals.FOR))
        self.assertEqual((under["level_p60"], under["way_p60"]), (-2.0, bet_totals.AGAINST))

    def test_where_the_line_held_the_probability_carries_the_move(self):
        (over,) = build([total(60, 1)])
        # 30s before the bet: same line, P(over) 0.50 against 0.55 at the bet -- the price firmed
        # into the bet, so the over bettor followed the move
        self.assertEqual(over["level_m30"], 0.0)
        self.assertAlmostEqual(over["plevel_m30"], -0.05)
        self.assertEqual(over["way_m30"], bet_totals.FOLLOWED)

    def test_a_line_that_moved_has_no_probability_level(self):
        (over,) = build([total(60, 1)])
        self.assertIsNone(over["plevel_p60"])

    def test_the_operators_lag_reads_the_bet_against_the_price_it_saw(self):
        tl, ch = bets.timeline(PROD), checks()
        b = total(110, 1)
        base = bets.join([b], {"FANDUEL": bets.Lag(20.0)}, {}, tl, {}, None, {}, ch)
        (r,) = bet_totals.build([b], base, [], tl, ch)
        self.assertEqual(r["stream_line"], 44.5)           # seen at 90s, before the move
        self.assertEqual((r["level_bet"], r["way_bet"]), (2.0, bet_totals.FOR))

    def test_restricted_is_the_temperature_and_everyone_else_the_rest(self):
        a, b, c = build([total(60, 1), total(60, 1, "VIP"), total(60, 1, None)])
        self.assertEqual([r["group"] for r in (a, b, c)],
                         [bet_totals.RESTRICTED, bet_totals.OTHERS, bet_totals.OTHERS])


class TestModel(unittest.TestCase):

    def test_a_model_already_at_the_line_prod_moved_to_leans_their_way_and_sits_nearer(self):
        (over,) = build([total(60, 1)], cand=CAND)
        self.assertTrue(over["m_ok"])
        self.assertEqual((over["m_line_edge"], over["m_lean"]), (2.0, bet_totals.FOR))
        self.assertEqual(over["m_nearer_p60"], "nearer")

    def test_at_the_same_line_the_lean_is_the_probability(self):
        cand = [q if q[6] != 2 else q[:3] + ((60.0 if q[1] == 54 else 40.0), None, "Line 44.5") + q[6:]
                for q in PROD]
        (over,) = build([total(60, 1)], cand=cand)
        self.assertAlmostEqual(over["m_prob_edge"], 0.05)
        self.assertEqual(over["m_lean"], bet_totals.FOR)


class TestSummaries(unittest.TestCase):

    def test_the_standard_error_is_clustered_by_match(self):
        rows = [dict(match_code="A", x=1.0), dict(match_code="A", x=1.0),
                dict(match_code="B", x=-1.0), dict(match_code="B", x=-1.0)]
        n, m, se = bet_totals.mean_se(rows, "x")
        self.assertEqual((n, m), (4, 0.0))
        self.assertAlmostEqual(se, 1.0)        # two clusters of +-2 -> sqrt(8 * 2) / 4

    def test_the_report_runs_on_both_groups(self):
        rows = build([total(60, 1, revenue=-9.0), total(60, 2, "Standard", revenue=10.0)], cand=CAND)
        lines = bet_totals.report(rows, min_bets=1)
        text = "\n".join(lines)
        self.assertIn(bet_totals.RESTRICTED, text)
        self.assertIn(bet_totals.OTHERS, text)
        self.assertIn("<svg", bet_totals.page(lines, [r for r in rows if bet_totals.in_scope(r)], "w"))


class TestBaselineAndGamers(unittest.TestCase):

    def test_the_baseline_is_the_operators_that_send_restricted_accounts(self):
        bs = [total(60, 1), total(60, 1, "Standard"), total(60, 1, None, operator="HARDROCK")]
        kept, ops = bet_totals.same_operators(bs)
        self.assertEqual((len(kept), ops), (2, ["FANDUEL"]))

    def test_with_no_restricted_account_every_operator_stays(self):
        bs = [total(60, 1, "Standard"), total(60, 1, None, operator="HARDROCK")]
        self.assertEqual(len(bet_totals.same_operators(bs)[0]), 2)

    def test_each_gamer_gets_every_bet_of_his_matches_and_the_miss_is_line_less_final(self):
        rows = build([total(60, 1), total(60, 2), total(60, 2, "Standard")], cand=CAND)
        bet_totals.add_players(rows, {"M1": {"PLAYER_1_HANDLE": "ann", "PLAYER_2_HANDLE": "bob"}})
        table = {k: v for k, *v in bet_totals.players(rows, ["m"])}
        self.assertEqual(set(table), {"ann", "bob"})
        (n, _, _, under), (others, _), miss, ((k, cand_miss, prod_same, _),) = table["ann"]
        self.assertEqual((n, under, others), (2, 50.0, 1))
        self.assertAlmostEqual(miss, 44.5 - 45)          # prod's line at the bet, final 24-21
        self.assertAlmostEqual(cand_miss, 46.5 - 45)     # the model already at 46.5
        self.assertAlmostEqual(prod_same, 44.5 - 45)     # prod on the same bets
        self.assertEqual(rows[0]["matchup"], "ann v bob")



def game(match, minutes, p1, p2, s1=21, s2=14):
    return dict(MATCH_CODE=match, SCHEDULED_START_TIME_UTC=f"2026-09-20 {minutes // 60:02d}:{minutes % 60:02d}:00",
                PLAYER_1_HANDLE=p1, PLAYER_2_HANDLE=p2, PLAYER_1_FINAL_SCORE=str(s1),
                PLAYER_2_FINAL_SCORE=str(s2))


class TestSessions(unittest.TestCase):

    # ann and bob play three matches 40 minutes apart, break five hours, then one more;
    # cat joins bob's second match only
    HISTORY = [game("A1", 0, "ann", "bob"), game("A2", 40, "ann", "cat"), game("A3", 80, "ann", "bob"),
               game("A4", 380, "ann", "bob", 10, 7)]

    def test_a_gap_over_the_break_starts_a_new_session(self):
        pos = bet_totals.sessions(self.HISTORY, gap=240)
        self.assertEqual(pos[("ann", "A1")], (1, 3, 3))
        self.assertEqual(pos[("ann", "A3")], (3, 1, 3))
        self.assertEqual(pos[("ann", "A4")], (1, 1, 1))
        self.assertEqual(pos[("bob", "A3")], (2, 1, 2))

    def test_the_match_is_labelled_by_how_many_of_its_gamers_are_on_their_last(self):
        pos = bet_totals.sessions(self.HISTORY, gap=240)
        self.assertEqual(bet_totals.session_end(pos[("ann", "A3")], pos[("bob", "A3")]),
                         bet_totals.LAST_BOTH)
        self.assertEqual(bet_totals.session_end(pos[("ann", "A2")], pos[("cat", "A2")]),
                         bet_totals.LAST_ONE)
        self.assertEqual(bet_totals.session_end(pos[("ann", "A1")], pos[("bob", "A1")]),
                         bet_totals.SECOND)

    def test_scoring_is_read_against_the_gamers_own_means(self):
        pos = bet_totals.sessions(self.HISTORY, gap=240)
        scoring = bet_totals.session_scoring(self.HISTORY, pos)
        # A4 (17 points) against ann's mean (35 * 3 + 17) / 4 and bob's (35 * 2 + 17) / 3
        n, m, _ = scoring[bet_totals.LAST_BOTH]
        ann, bob = (35 * 3 + 17) / 4, (35 * 2 + 17) / 3
        self.assertEqual(n, 2)                                 # A3 and A4
        self.assertAlmostEqual(m, ((35 - (ann + bob) / 2) + (17 - (ann + bob) / 2)) / 2)


if __name__ == "__main__":
    unittest.main()
