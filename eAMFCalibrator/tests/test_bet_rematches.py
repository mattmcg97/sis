"""Rematches the same day: the meetings numbered within the gamers' session, the teams against the
first meeting, each bet read against the first winner, and prod's price of him then and now."""

import datetime as dt
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
                        "The 25 rematches"):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
