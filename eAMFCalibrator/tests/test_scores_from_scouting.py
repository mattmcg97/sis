"""Score changes rebuilt from SCOUTING_FULL for matches SCORE_CHANGES lost."""

import unittest
from unittest import mock

from .. import config, scouting, snowflake_io as io


class Cursor:
    """SCORE_CHANGES has AF1 only; the scouting query answers AF2's messages."""

    def __init__(self):
        self.sql = []

    def execute(self, sql, params=()):
        self.sql.append((sql, params))
        if "SCORE_CHANGES" in sql:
            self.rows = [("AF1", 10, 1, 7, 0, 7, 0)]
        else:
            self.rows = [("AF2", 1, "FIRST_QUARTER_STARTED", None),
                         ("AF2", 5, None, "TOUCHDOWN_TEAM_B"),
                         ("AF2", 6, None, "EXTRA_POINT_GOOD_TEAM_B"),
                         ("AF2", 9, "SECOND_QUARTER_STARTED", None),
                         ("AF2", 12, None, "FIELD_GOAL_GOOD_TEAM_A"),
                         ("AF2", 14, None, "FIELD_GOAL_MISSED_TEAM_A"),
                         ("AF2", 20, None, "SAFETY_AWARDED_TEAM_A"),
                         ("AF2", 20, None, "PLAY_OVER")]
        self.description = [("C%d" % k,) for k in range(len(self.rows[0]))]

    def fetchall(self):
        return list(self.rows)


class TestScoresFromScouting(unittest.TestCase):
    def setUp(self):
        table = scouting.Table("DB.S.SCOUTING_FULL", {"FILE_TIME": "TIMESTAMP_NTZ"})
        self.patches = [mock.patch.object(io, "scouting_table", lambda cur: table),
                        mock.patch.object(config, "FETCH_CACHE", False)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()

    def test_a_match_without_score_rows_gets_them_from_its_scoring_messages(self):
        with mock.patch.object(config, "SCORES_FROM_SCOUTING", True):
            got = io.fetch_scores(Cursor(), ["AF1", "AF2"])
        self.assertEqual(got, [("AF1", 10, 1, 7, 0, 7, 0),
                               ("AF2", 5, 1, 0, 6, 0, 6), ("AF2", 6, 1, 0, 1, 0, 7),
                               ("AF2", 12, 2, 3, 0, 3, 7), ("AF2", 20, 2, 2, 0, 5, 7)])

    def test_the_rebuilt_rows_resolve_team_a_as_home(self):
        with mock.patch.object(config, "SCORES_FROM_SCOUTING", True):
            got = io.fetch_scores(Cursor(), ["AF2"])
        rows = [(None, 5, None, None, "TOUCHDOWN_TEAM_B"), (None, 12, None, None, "FIELD_GOAL_GOOD_TEAM_A")]
        self.assertEqual(scouting.team_a_side(rows, got), "home")

    def test_off_leaves_score_changes_as_it_is(self):
        cur = Cursor()
        with mock.patch.object(config, "SCORES_FROM_SCOUTING", False):
            got = io.fetch_scores(cur, ["AF1", "AF2"])
        self.assertEqual(got, [("AF1", 10, 1, 7, 0, 7, 0)])
        self.assertEqual(len(cur.sql), 1)


if __name__ == "__main__":
    unittest.main()
