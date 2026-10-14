"""Rematches the same day: the meetings numbered within the gamers' session, the teams against the
first meeting, each bet read against the first winner, and prod's price of him then and now."""

import datetime as dt
import math
import random
import unittest

from .. import bet_rematches as br, bets, config
from .test_bets import bet

T0 = dt.datetime(2026, 10, 1, 18, 0)


def game(code, minutes, home, away, s1, s2, t1="Ravens", t2="Lions"):
    return dict(MATCH_CODE=code, SCHEDULED_START_TIME_UTC=str(T0 + dt.timedelta(minutes=minutes)),
                PLAYER_1_HANDLE=home, PLAYER_2_HANDLE=away, PLAYER_1_TEAM=t1, PLAYER_2_TEAM=t2,
                PLAYER_1_FINAL_SCORE=str(s1), PLAYER_2_FINAL_SCORE=str(s2))


# ANN and BOB meet three times in one night (ANN wins the first by 10), with ANN v CAT in between;
# the next night, after an eleven-hour break, they meet again: a new day's first meeting
HISTORY = [game("G1", 0, "ANN", "BOB", 24, 14),
           game("G2", 50, "ANN", "CAT", 7, 3),
           game("G3", 100, "BOB", "ANN", 10, 17, t1="Lions", t2="Ravens"),     # the same teams, BOB at home
           game("G4", 150, "ANN", "BOB", 20, 21, t1="Lions", t2="Ravens"),     # swapped
           game("G5", 800, "ANN", "BOB", 3, 0, t1="Bills", t2="Lions")]


class TestMeetings(unittest.TestCase):

    def setUp(self):
        self.m = br.day_meetings(HISTORY)

    def test_meetings_are_numbered_within_the_day(self):
        self.assertEqual([self.m[g]["meeting"] for g in ("G1", "G3", "G4", "G5")], ["1st", "2nd", "3rd", "1st"])
        self.assertEqual((self.m["G2"]["meeting"], self.m["G2"]["pair"]), ("1st", "ANN v CAT"))
        self.assertEqual(self.m["G3"]["first"]["match"], "G1")
        self.assertEqual(self.m["G4"]["prev"]["match"], "G3")

    def test_the_teams_against_the_first_meeting(self):
        self.assertEqual((self.m["G3"]["teams"], self.m["G4"]["teams"]), ("same teams", "swapped teams"))
        self.assertIsNone(self.m["G1"]["teams"])
        h = HISTORY[:1] + [game("G9", 60, "ANN", "BOB", 3, 0, t1="Bills", t2="Lions")]
        self.assertEqual(br.day_meetings(h)["G9"]["teams"], "teams changed")

    def test_the_first_meetings_winner_and_margin(self):
        first = self.m["G4"]["first"]
        self.assertEqual((first["winner"], first["loser"], first["margin"], first["total"]), ("ANN", "BOB", 10, 38))
        self.assertEqual(br.margin_band(first["margin"]), "8-13")
        self.assertEqual([br.margin_band(x) for x in (0, 3, 4, 21)], ["draw", "1-3", "4-7", "21+"])

    def test_a_bet_reads_against_the_first_winner(self):
        m = self.m["G3"]                                   # BOB at home, ANN away; ANN won the first
        on_bob = bet("G3", 0, 1, 1, stake=10.0, revenue=10.0)
        on_ann = bet("G3", 0, 1, 2, stake=10.0, revenue=-9.0, odds=1.9)
        over = bet("G3", 0, 3, 1, line=40.5)
        rows = bets.join([on_bob, on_ann, over], {}, {}, {}, {}, None, {})
        r = [br.bet_row(b, j, m) for b, j in zip((on_bob, on_ann, over), rows)]
        self.assertEqual([x["side"] for x in r], [br.LOSER, br.WINNER, "Over"])
        self.assertEqual((r[1]["won"], r[1]["first_band"], r[1]["meeting"]), (1.0, "8-13", "2nd"))

    def test_prods_price_of_the_first_winner_then_and_now(self):
        def q(match, p):
            return (match, 50, T0, 100.0 * p, None, None, None, "open", "true")
        tl = bets.timeline([q("G1", 0.55), q("G3", 0.30)])          # G3: BOB (home) at 30%, so ANN 70%
        row = br.match_row(self.m["G3"], [], tl)
        self.assertAlmostEqual(row["fw_p_first"], 0.55)
        self.assertAlmostEqual(row["fw_p_now"], 0.70)
        self.assertEqual(row["fw_won"], 1.0)
        self.assertAlmostEqual(row["fw_res"], 0.30)
        self.assertEqual(br.match_row(self.m["G4"], [], tl)["fw_won"], 0.0)


class TestLinesAndFits(unittest.TestCase):

    def test_meetings_carry_the_day_and_the_earlier_ones(self):
        m = br.day_meetings(HISTORY)
        self.assertEqual(m["G4"]["earlier"], ["G1", "G3"])
        self.assertEqual(m["G4"]["day_key"], m["G1"]["day_key"])
        self.assertNotEqual(m["G5"]["day_key"], m["G1"]["day_key"])

    def test_prods_closing_total_line(self):
        def q(match, line, msg=None):
            return (match, 54, T0, 50.0, None, f"Line {line}", msg, "open", "true")
        tl = bets.timeline([q("G1", 38.5), q("G1", 40.5), q("G1", 44.5, msg=7)])
        self.assertEqual(br.prematch_line(tl, "G1"), 40.5)               # the last before kick-off
        self.assertEqual(br.prematch_line(bets.timeline([q("G2", 33.5, msg=3)]), "G2"), 33.5)
        self.assertIsNone(br.prematch_line(tl, "G9"))

    def test_the_rematch_row_reads_the_lines_and_the_surprise(self):
        def q(match, market, p, line=None):
            return (match, market, T0, 100.0 * p, None, None if line is None else f"Line {line}", None,
                    "open", "true")
        tl = bets.timeline([q("G1", 50, 0.5), q("G1", 54, 0.5, 35.5), q("G3", 50, 0.5), q("G3", 54, 0.5, 37.5)])
        row = br.match_row(br.day_meetings(HISTORY)["G3"], [], tl)
        self.assertEqual((row["line_first"], row["line_now"]), (35.5, 37.5))
        self.assertAlmostEqual(row["first_miss"], 38 - 35.5)              # G1: 24-14
        self.assertAlmostEqual(row["miss"], 27 - 37.5)                    # G3: 10-17
        self.assertEqual(row["first_miss_band"], "first within 3")
        self.assertAlmostEqual(row["fw_first_surprise"], 10.0)            # ANN by 10 at evens
        self.assertAlmostEqual(row["fw_margin_miss"], 7.0)                # ANN by 7 again at evens

    def test_the_logistic_fit_recovers_a_planted_slope(self):
        rnd = random.Random(2)
        xs = [rnd.gauss(0, 1) for _ in range(4000)]
        offs = [rnd.gauss(0, 0.5) for _ in xs]
        ys = [1.0 if rnd.random() < 1 / (1 + math.exp(-(o + 0.1 + 0.4 * x))) else 0.0 for x, o in zip(xs, offs)]
        a, b, sa, sb = br.logistic_offset(xs, ys, offs)
        self.assertAlmostEqual(a, 0.1, delta=3 * sa)
        self.assertAlmostEqual(b, 0.4, delta=3 * sb)

    def test_the_layer_on_prods_prices(self):
        def q(match, market, p, line=None):
            return (match, market, T0, 100.0 * p, None, None if line is None else f"Line {line}", None,
                    "open", "true")
        meets = br.day_meetings(HISTORY)
        tl = bets.timeline([x for g in ("G1", "G2", "G3", "G4", "G5") for x in (q(g, 50, 0.5), q(g, 54, 0.5, 30.5))])
        rows, layer = br.prod_layer(meets, sorted(meets), tl, min_days=0)
        self.assertEqual(len(rows), 5)
        self.assertEqual(sorted(r["n_earlier"] for r in rows), [0, 0, 0, 1, 2])
        text = "\n".join(br.layer_lines(rows, layer, 0))
        self.assertIn("2nd meetings", text)


class TestReport(unittest.TestCase):

    def test_the_report_over_many_rematches(self):
        rnd = random.Random(5)
        hist, minute = [], 0
        names = ["A", "B", "C", "D"]
        for k in range(300):
            h, a = rnd.sample(names, 2)
            hist.append(game(f"M{k}", minute, h, a, rnd.randint(0, 35), rnd.randint(0, 35),
                             t1=rnd.choice(("Ravens", "Lions")), t2=rnd.choice(("Bills", "Chiefs"))))
            minute += 40 if k % 12 else 400
        meets = br.day_meetings(hist)
        brows, mrows = [], []
        for code, m in meets.items():
            rows = []
            for j in range(rnd.randint(0, 6)):
                b = bet(code, j, rnd.choice((1, 2, 3)), rnd.choice((1, 2)), stake=50.0,
                        revenue=rnd.choice((50.0, -45.0)), odds=1.9, line=40.5, in_play=rnd.choice(("Yes", "No")),
                        **{config.BET_VIP_COLUMN: rnd.choice(("VIP", "Restricted", "Standard"))})
                (j_row,) = bets.join([b], {}, {}, {}, {}, None, {})
                j_row["stream_prob"] = 0.5
                rows.append(br.bet_row(b, j_row, m))
            brows += rows
            mrows.append(br.match_row(m, rows, {}))
        text = "\n".join(br.report(brows, mrows))
        for heading in ("Every match by its meeting of the day", "by the first meeting's margin of victory",
                        "Same teams (against the first meeting)", "Handicap:", "Total: over and under",
                        "The 25 rematches", "Rematch totals against prod's line", "How far to move a rematch"):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
