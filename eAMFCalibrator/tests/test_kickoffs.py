"""kickoffs: how kick-offs land, the half-time double, and prod's total on a kick."""

import csv
import os
import tempfile
import unittest

from .. import kickoffs, scouting


def _row(msg, period, clock, kind, offense, score, nxt_off=None, nxt_field=None, messages="", line=None,
         p_over=0.5, final=(14, 10), field=""):
    r = {f: "" for f in scouting.EXPORT_FIELDS}
    r.update(match_code="AF1", message=msg, period=period, clock_seconds=clock, play_kind=kind, offense=offense,
             scrimmage="0" if kind in ("KICKOFF", "CONVERSION") else "1", team_a_side="home",
             score_p1=score[0], score_p2=score[1], score_p1_at_start=score[0], score_p2_at_start=score[1],
             next_offense=nxt_off or offense, next_field_position="" if nxt_field is None else nxt_field,
             play_messages=messages, final_p1=final[0], final_p2=final[1], field_position=field,
             home_handle="ALPHA", away_handle="BETA", prematch_line_54=40.5, prematch_prob_54=0.5,
             prematch_prob_50=0.7)
    if line is not None:
        r.update(line_54=line, prob_54=p_over, live_54=1, line_52=-3.5, prob_52=0.5, live_52=1)
    return r


ROWS = [
    # ALPHA (home, TEAM_A, the favourite) receives the opening kick at its 25: prod drops to 39.5
    _row(10, 1, 236, "KICKOFF", "TEAM_A", (0, 0), nxt_field=25, line=39.5),
    _row(20, 1, 200, "TOUCHDOWN", "TEAM_A", (6, 0), messages="TOUCHDOWN_TEAM_A", line=44.5),
    _row(21, 1, 200, "CONVERSION", "TEAM_A", (7, 0), line=44.5, nxt_off="TEAM_B"),
    # ALPHA kicks a touchback: BETA at the 20, no clock; prod's line holds
    _row(30, 1, 200, "KICKOFF", "TEAM_B", (7, 0), nxt_field=20, line=44.5),
    _row(40, 2, 100, "SCRIMMAGE", "TEAM_B", (7, 0), line=43.5, field=30),
    _row(50, 2, 60, "FIELD_GOAL", "TEAM_B", (7, 3), messages="FIELD_GOAL_GOOD_TEAM_B", line=42.5, nxt_off="TEAM_A"),
    # the second half: BETA (which kicked the game off) receives, returned to its 30 in 4 seconds,
    # and scores on that drive -- the double
    _row(60, 3, 236, "KICKOFF", "TEAM_B", (7, 3), nxt_field=30, line=40.5),
    _row(70, 3, 200, "TOUCHDOWN", "TEAM_B", (7, 9), messages="TOUCHDOWN_TEAM_B", line=40.5),
    _row(71, 3, 200, "CONVERSION", "TEAM_B", (7, 10), line=40.5, nxt_off="TEAM_A"),
    _row(80, 4, 10, "SCRIMMAGE", "TEAM_A", (14, 10), line=24.5),
]


class TestKickoffs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "export.csv")
        with open(self.path, "w", newline="") as fh:
            w = csv.DictWriter(fh, scouting.EXPORT_FIELDS)
            w.writeheader()
            w.writerows(ROWS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_kicks_are_classed_and_their_drives_counted(self):
        (code, rows), = kickoffs._matches(self.path)
        m = kickoffs.Match(code, rows)
        ks = kickoffs.kicks(m)
        self.assertEqual([(k["period"], k["receiver"], k["kind"], k["start"]) for k in ks],
                         [(1, "home", kickoffs.RETURNED, 25), (1, "away", kickoffs.TOUCHBACK, 20),
                          (3, "away", kickoffs.RETURNED, 30)])
        self.assertEqual([k["drive_points"] for k in ks], [7, 3, 7])
        self.assertEqual((ks[0]["line_before"], ks[0]["line_kick"]), (40.5, 39.5))    # pre-match -> kick
        self.assertEqual((ks[1]["line_before"], ks[1]["line_kick"]), (44.5, 44.5))
        self.assertEqual(ks[2]["line_before"], 42.5)                          # Q2's last quote
        self.assertTrue(ks[0]["receiver_favourite"])
        self.assertEqual(ks[1]["kicker_handle"], "ALPHA")

    def test_the_half_time_double(self):
        (code, rows), = kickoffs._matches(self.path)
        m = kickoffs.Match(code, rows)
        h = kickoffs.half_time(m, kickoffs.kicks(m))
        self.assertEqual((h["r"], h["r_handle"], h["late_q2_r"], h["q3_first_drive_r"], h["double"]),
                         ("away", "BETA", 3, 7, True))
        self.assertEqual((h["q2_points_r"], h["q2_points_k"]), (3, 0))
        self.assertEqual(h["two_min_line"], 43.5)

    def test_run_writes_the_tables(self):
        text = "\n".join(kickoffs.run(self.path, self.tmp.name, min_games=1))
        self.assertIn("3 kick-offs", text)
        self.assertIn("THE HALF-TIME DOUBLE", text)
        self.assertIn("opening kick (pre-match)", text)
        for f in ("kickoffs.txt", "kickoffs_kicks.csv", "kickoffs_matches.csv", "kickoffs_gamers.csv"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp.name, f)), f)


if __name__ == "__main__":
    unittest.main()
