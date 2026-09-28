"""A version's points still to come against reality: the summaries."""

import unittest

import numpy as np

from .. import remaining
from ..pricer import AWAY, HOME, GameState


def pmf(**mass):
    p = np.zeros(remaining.MAX_REMAINING + 1)
    for k, v in mass.items():
        p[int(k[1:])] = v
    return p


class TestRemaining(unittest.TestCase):

    def test_the_game_state_is_read_off_the_margin_and_the_ball(self):
        s = lambda h, a, off: GameState(offense=off, down=1, distance=10, field_position=25,
                                        home_score=h, away_score=a, period=4, elapsed_in_period=0.0,
                                        clock_seconds=60)
        self.assertEqual(remaining.game_state(s(7, 7, HOME)), "level")
        self.assertEqual(remaining.game_state(s(14, 7, HOME)), "1 score, leader has ball")
        self.assertEqual(remaining.game_state(s(14, 7, AWAY)), "1 score, trailer has ball")
        self.assertEqual(remaining.game_state(s(7, 21, AWAY)), "2+ scores, leader has ball")

    def test_the_version_is_set_against_what_really_came(self):
        # two Q4 snapshots, level: the version gives a field goal half the time, reality one in two
        rows = [("A", 1, "Q4", "level", 20, 3, pmf(p0=0.5, p3=0.5)),
                ("B", 1, "Q4", "level", 20, 7, pmf(p0=0.5, p3=0.5))]
        text = "\n".join(remaining.summary(rows))
        self.assertIn("Q4 level", text)
        # real: =3 50%, =7 50%; version: =3 50%, =7 0% -> -50.0 at 7, mean 1.5 against 5.0
        self.assertIn("-50.0", text)
        self.assertIn("-3.50", text)
        cal = "\n".join(remaining.over_calibration(rows, thresholds=(0.5, 3.5)))
        # P(>0.5): version 50%, real 100% -> -50.0; P(>3.5): version 0%, real 50% -> -50.0
        self.assertIn("-50.0  -50.0", cal)

    def test_the_match_day_is_read_off_the_code(self):
        self.assertEqual(str(remaining.day("AF012170926")), "2026-09-17")


if __name__ == "__main__":
    unittest.main()


class TestBuildUntil(unittest.TestCase):

    def test_until_builds_only_on_matches_before_the_date(self):
        import csv
        import os
        import tempfile
        from unittest import mock
        from .. import __main__ as cli, playover
        rows = {"AF001090926": [{"match_code": "AF001090926"}], "AF002100926": [{"match_code": "AF002100926"}],
                "AF003110926": [{"match_code": "AF003110926"}]}
        seen = {}

        def fake_build(matches, out, **kw):
            seen["codes"], seen["before"] = sorted(matches), kw.get("before")
        with mock.patch.object(playover, "load", return_value=rows), \
                mock.patch("eAMFModel.v6.build", side_effect=fake_build), \
                mock.patch("eAMFModel.nb2_prior.load_history", return_value=[]), \
                mock.patch("builtins.print"):
            cli.main(["v6-build", "x.csv", "--half", "all", "--out", tempfile.mkdtemp(),
                      "--history", "h.csv", "--until", "2026-09-10"])
        self.assertEqual(seen["codes"], ["AF001090926"])
        self.assertEqual(str(seen["before"])[:10], "2026-09-10")


class TestDrivePoints(unittest.TestCase):

    def rows(self):
        def r(msg, kind, off, p1, p2, period=3):
            return dict(message=msg, play_kind=kind, offense=off, score_p1=p1, score_p2=p2,
                        period=period, team_a_side="home", final_p1=20, final_p2=10)
        return [r(1, "SCRIMMAGE", "TEAM_A", 10, 7), r(2, "SCRIMMAGE", "TEAM_A", 10, 7),
                r(3, "FIELD_GOAL", "TEAM_A", 10, 7), r(4, "KICKOFF", "TEAM_A", 13, 7),
                r(5, "SCRIMMAGE", "TEAM_B", 13, 7), r(6, "PUNT", "TEAM_B", 13, 7),
                r(7, "SCRIMMAGE", "TEAM_A", 13, 7), r(8, "SCRIMMAGE", "TEAM_A", 13, 7, period=5)]

    def test_the_drive_ends_at_the_kick_off_the_change_of_hands_or_the_half(self):
        rows = self.rows()
        self.assertEqual(remaining.real_drive_points(rows, 0, HOME), 3)     # the field goal
        self.assertEqual(remaining.real_drive_points(rows, 4, AWAY), 0)     # punted away
        self.assertEqual(remaining.real_drive_points(rows, 6, HOME), 0)     # the half turned
