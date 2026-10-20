"""xgb.drives: snaps and drives off a hand-made export."""

import csv
import os
import tempfile
import unittest
from unittest import mock

from .. import drives

HEAD = list(drives.KEEP)


def play(code, msg, kind, offense, down="", dist="", field="", clock=200, period=1, messages="", board=(0, 0),
         final=(7, 3)):
    """One export row; `board` is the score (home, away) as the play starts and as it ends."""
    return dict(match_code=code, message=msg, period=period, clock_seconds=clock, play_kind=kind,
                team_a_side="home", offense=offense, down=down, distance=dist, field_position=field,
                play_messages=messages, final_p1=final[0], final_p2=final[1], home_handle="ANN", away_handle="BOB",
                score_p1=board[0], score_p2=board[1], score_p1_at_start=board[0], score_p2_at_start=board[1],
                prematch_line_52=-2.5, prematch_prob_52=0.5, prematch_line_54=40.5, prematch_prob_54=0.5,
                prematch_prob_50=0.55)


# M1. TEAM_A (home): 25 -> 35 -> 85 -> touchdown; kick-off; TEAM_B: 30 -> 28 -> 28 -> punt; TEAM_A:
# 20 -> 25, intercepted; TEAM_B: 60 -> field goal from 70; kick-off.
M1 = [
    play("M1", 1, "KICKOFF", "TEAM_B", clock=240),
    play("M1", 2, "SCRIMMAGE", "TEAM_A", 1, 10, 25, 235),
    play("M1", 3, "SCRIMMAGE", "TEAM_A", 1, 10, 35, 228),
    play("M1", 4, "SCRIMMAGE", "TEAM_A", 1, 10, 85, 220),
    play("M1", 5, "TOUCHDOWN", "TEAM_A", 1, 10, 85, 214, messages="TOUCHDOWN_TEAM_A", board=(6, 0)),
    play("M1", 6, "CONVERSION", "TEAM_A", clock=214, messages="EXTRA_POINT_GOOD_TEAM_A", board=(6, 0)),
    play("M1", 7, "KICKOFF", "TEAM_A", clock=214, board=(7, 0)),
    play("M1", 8, "SCRIMMAGE", "TEAM_B", 1, 10, 30, 210, board=(7, 0)),
    play("M1", 9, "SCRIMMAGE", "TEAM_B", 2, 12, 28, 204, board=(7, 0)),
    play("M1", 10, "SCRIMMAGE", "TEAM_B", 3, 12, 28, 198, board=(7, 0)),
    play("M1", 11, "PUNT", "TEAM_B", 4, 10, 30, 192, messages="PUNT_TEAM_B", board=(7, 0)),
    play("M1", 12, "SCRIMMAGE", "TEAM_A", 1, 10, 20, 186, board=(7, 0)),
    play("M1", 13, "SCRIMMAGE", "TEAM_A", 2, 5, 25, 180, board=(7, 0)),
    play("M1", 14, "SCRIMMAGE", "TEAM_B", 1, 10, 60, 174, messages="POSSESSION_TEAM_B", board=(7, 0)),
    play("M1", 15, "FIELD_GOAL", "TEAM_B", 1, 10, 70, 168, messages="FIELD_GOAL_GOOD_TEAM_B", board=(7, 0)),
    play("M1", 16, "KICKOFF", "TEAM_B", clock=168, board=(7, 3)),
]

# M2. The touchdown's message lands late, so its play is filed as a scrimmage: TEAM_A 30 -> 60, scores
# from the 60; then TEAM_B fails on an unflagged 4th and 2 and TEAM_A takes over.
M2 = [
    play("M2", 1, "KICKOFF", "TEAM_B", clock=240, final=(7, 0)),
    play("M2", 2, "SCRIMMAGE", "TEAM_A", 1, 10, 30, 235, final=(7, 0)),
    play("M2", 3, "SCRIMMAGE", "TEAM_A", 1, 10, 60, 228, final=(7, 0)),
    play("M2", 4, "CONVERSION", "TEAM_A", clock=222, messages="EXTRA_POINT_GOOD_TEAM_A", board=(6, 0), final=(7, 0)),
    play("M2", 5, "KICKOFF", "TEAM_A", clock=222, board=(7, 0), final=(7, 0)),
    play("M2", 6, "SCRIMMAGE", "TEAM_B", 4, 2, 40, 216, board=(7, 0), final=(7, 0)),
    play("M2", 7, "SCRIMMAGE", "TEAM_A", 1, 10, 61, 210, board=(7, 0), final=(7, 0)),
]


class TestDrives(unittest.TestCase):

    def stats(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "export.csv"), os.path.join(d, "out.csv")
            with open(src, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=HEAD)
                w.writeheader()
                for r in M1 + M2:
                    w.writerow({k: r[k] for k in HEAD})
            with mock.patch("builtins.print"):
                drives.run(src, dst, verbose=False)
            with open(dst) as fh:
                return {r["match_code"]: r for r in csv.DictReader(fh)}

    def test_drives_and_their_ends(self):
        s = self.stats()["M1"]
        g = lambda k: int(s[k])
        self.assertEqual((g("h_drives"), g("h_td"), g("h_turnover"), g("h_punt")), (2, 1, 1, 0))
        self.assertEqual((g("a_drives"), g("a_punt"), g("a_fg"), g("a_turnover")), (2, 1, 1, 0))
        self.assertEqual((g("h_rz_trips"), g("h_rz_td")), (1, 1))
        self.assertEqual((g("h_drive_pts"), g("a_drive_pts")), (7, 3))
        self.assertEqual(g("h_start_field") + g("a_start_field"), 25 + 20 + 30 + 60)
        self.assertEqual(s["prematch_line_54"], "40.5")

    def test_snaps(self):
        s = self.stats()["M1"]
        g = lambda k: int(float(s[k]))
        # home: +10, +50, +15 (the touchdown), +5, then intercepted
        self.assertEqual((g("h_snaps"), g("h_gains"), g("h_yards")), (5, 4, 80))
        self.assertEqual((g("h_success"), g("h_big20"), g("h_big40")), (3, 1, 1))
        # away: -2, 0 on 3rd and 12, +2 to the punt; +10 to the field goal
        self.assertEqual((g("a_snaps"), g("a_yards"), g("a_loss")), (4, 10, 1))
        self.assertEqual((g("a_third_n"), g("a_third_conv")), (1, 0))
        self.assertEqual(g("h_seconds"), 7 + 8 + 6 + 6 + 6)

    def test_a_late_touchdown_message_and_an_unflagged_4th_down(self):
        s = self.stats()["M2"]
        g = lambda k: int(float(s[k]))
        self.assertEqual((g("h_drives"), g("h_td"), g("h_half_end"), g("h_drive_pts")), (2, 1, 1, 7))
        self.assertEqual((g("h_yards"), g("h_big40")), (30 + 40, 1))        # the scoring snap gains the 40 left
        self.assertEqual((g("a_drives"), g("a_downs"), g("a_turnover")), (1, 1, 0))
        self.assertEqual((g("a_fourth_n"), g("a_fourth_conv")), (1, 0))


if __name__ == "__main__":
    unittest.main()
