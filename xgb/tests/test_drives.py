"""xgb.drives: snaps and drives off a hand-made export."""

import csv
import os
import tempfile
import unittest

from .. import drives

HEAD = list(drives.KEEP)


def play(msg, kind, offense, down="", dist="", field="", clock=200, period=1, messages=""):
    return dict(match_code="M1", message=msg, period=period, clock_seconds=clock, play_kind=kind,
                team_a_side="home", offense=offense, down=down, distance=dist, field_position=field,
                play_messages=messages, final_p1=7, final_p2=3, home_handle="ANN", away_handle="BOB",
                prematch_line_52=-2.5, prematch_prob_52=0.5, prematch_line_54=40.5, prematch_prob_54=0.5,
                prematch_prob_50=0.55)


# TEAM_A (home): 25 -> 35 -> 85 -> touchdown; kick-off; TEAM_B: 30 -> 28 -> 28 -> punt; TEAM_A: 20 -> 25,
# intercepted; TEAM_B: 60 -> field goal from 70; kick-off.
ROWS = [
    play(1, "KICKOFF", "TEAM_B", clock=240),
    play(2, "SCRIMMAGE", "TEAM_A", 1, 10, 25, 235),
    play(3, "SCRIMMAGE", "TEAM_A", 1, 10, 35, 228),
    play(4, "SCRIMMAGE", "TEAM_A", 1, 10, 85, 220),
    play(5, "TOUCHDOWN", "TEAM_A", 1, 10, 85, 214, messages="TOUCHDOWN_TEAM_A"),
    play(6, "CONVERSION", "TEAM_A", clock=214, messages="EXTRA_POINT_GOOD_TEAM_A"),
    play(7, "KICKOFF", "TEAM_A", clock=214),
    play(8, "SCRIMMAGE", "TEAM_B", 1, 10, 30, 210),
    play(9, "SCRIMMAGE", "TEAM_B", 2, 12, 28, 204),
    play(10, "SCRIMMAGE", "TEAM_B", 3, 12, 28, 198),
    play(11, "PUNT", "TEAM_B", 4, 10, 30, 192, messages="PUNT_TEAM_B"),
    play(12, "SCRIMMAGE", "TEAM_A", 1, 10, 20, 186),
    play(13, "SCRIMMAGE", "TEAM_A", 2, 5, 25, 180),
    play(14, "SCRIMMAGE", "TEAM_B", 1, 10, 60, 174, messages="POSSESSION_TEAM_B"),
    play(15, "FIELD_GOAL", "TEAM_B", 1, 10, 70, 168, messages="FIELD_GOAL_GOOD_TEAM_B"),
    play(16, "KICKOFF", "TEAM_B", clock=168),
]


class TestDrives(unittest.TestCase):

    def stats(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "export.csv"), os.path.join(d, "out.csv")
            with open(src, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=HEAD)
                w.writeheader()
                for r in ROWS:
                    w.writerow({k: r[k] for k in HEAD})
            from unittest import mock
            with mock.patch("builtins.print"):
                drives.run(src, dst, verbose=False)
            with open(dst) as fh:
                return next(csv.DictReader(fh))

    def test_drives_and_their_ends(self):
        s = self.stats()
        g = lambda k: int(s[k])
        self.assertEqual((g("h_drives"), g("h_td"), g("h_turnover"), g("h_punt")), (2, 1, 1, 0))
        self.assertEqual((g("a_drives"), g("a_punt"), g("a_fg"), g("a_turnover")), (2, 1, 1, 0))
        self.assertEqual((g("h_rz_trips"), g("h_rz_td")), (1, 1))
        self.assertEqual(g("h_start_field") + g("a_start_field"), 25 + 20 + 30 + 60)
        self.assertEqual(s["prematch_line_54"], "40.5")

    def test_snaps(self):
        s = self.stats()
        g = lambda k: int(float(s[k]))
        # home: +10, +50, +15 (the touchdown), +5, then intercepted
        self.assertEqual((g("h_snaps"), g("h_gains"), g("h_yards")), (5, 4, 80))
        self.assertEqual((g("h_success"), g("h_big20"), g("h_big40")), (3, 1, 1))
        # away: -2, 0 on 3rd and 12, +2 to the punt; +10 to the field goal
        self.assertEqual((g("a_snaps"), g("a_yards"), g("a_loss")), (4, 10, 1))
        self.assertEqual((g("a_third_n"), g("a_third_conv")), (1, 0))
        self.assertEqual(g("h_seconds"), 7 + 8 + 6 + 6 + 6)


if __name__ == "__main__":
    unittest.main()
