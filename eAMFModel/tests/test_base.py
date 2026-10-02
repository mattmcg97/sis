"""The pieces every version shares: reading a PLAY_OVER snapshot into a game state, player
profiles, the stream's market descriptions and the league's 4th-down curves. No data files.

Run with:  python -m unittest eAMFModel.tests.test_base
"""

import unittest

from .. import drive, players, playover, stream
from ..state import AWAY, HOME, ML_HOME, SPREAD_AWAY, SPREAD_HOME, TOTAL_OVER, TOTAL_UNDER


class TestDriveCurves(unittest.TestCase):
    def test_go_probability_follows_the_data(self):
        p = drive.DriveParams()
        self.assertGreater(drive.go_probability(p, 35, 1), 0.8)     # 4th-and-1, own 35
        self.assertLess(drive.go_probability(p, 35, 15), 0.35)
        self.assertGreater(drive.go_probability(p, 35, 3, 1.0), drive.go_probability(p, 35, 3))
        self.assertAlmostEqual(drive.fg_make(p, 72), 0.949, places=2)   # a 45-yarder
        self.assertLess(drive.field_goal_share(p, 40), 0.5)
        self.assertGreater(drive.field_goal_share(p, 70), 0.5)


class TestPlayOver(unittest.TestCase):
    def snap(self, **kw):
        r = {f: "" for f in ("match_code", "message", "period", "clock_seconds", "play_kind",
                              "team_a_side", "offense", "down", "distance", "field_position",
                              "next_offense", "next_down", "next_distance",
                              "next_field_position", "score_p1", "score_p2",
                              "score_p1_at_start", "score_p2_at_start", "play_messages",
                              "opening_offense")}
        r.update(match_code="AF1", message="20", period="2", clock_seconds="150",
                 play_kind="SCRIMMAGE", team_a_side="away", offense="TEAM_B", down="1",
                 distance="10", field_position="40", next_offense="TEAM_B", next_down="2",
                 next_distance="4", next_field_position="46", score_p1="7", score_p2="0",
                 score_p1_at_start="7", score_p2_at_start="0", opening_offense="TEAM_A")
        r.update({k: str(v) for k, v in kw.items()})
        return r

    def test_scrimmage_takes_the_row(self):
        state, _ = playover.state_for(self.snap())
        self.assertEqual((state.offense, state.down, state.distance, state.field_position),
                         (HOME, 1, 10, 40))       # TEAM_A is away, so TEAM_B is home
        self.assertTrue(state.snap_confirmed)
        self.assertEqual(state.clock_seconds, 150)
        self.assertEqual(state.opening_receiver, AWAY)
        state, _ = playover.state_for(self.snap(), playover.NEXT)
        self.assertEqual((state.down, state.field_position), (2, 46))

    def test_a_turnover_row_is_the_new_side(self):
        state, _ = playover.state_for(self.snap(offense="TEAM_A", field_position="55"))
        self.assertEqual((state.offense, state.field_position), (AWAY, 55))

    def test_touchdown_scorer_from_the_message_and_points_added(self):
        # A pick six: the touchdown message names the side that scored.
        state, _ = playover.state_for(self.snap(play_kind="TOUCHDOWN",
                                                play_messages="POSSESSION_TEAM_A|TOUCHDOWN_TEAM_A"))
        self.assertEqual(state.pending_conversion, AWAY)
        self.assertEqual((state.home_score, state.away_score), (7, 6))
        self.assertEqual(state.offense, HOME)
        state, _ = playover.state_for(self.snap(play_kind="TOUCHDOWN", score_p1="13",
                                                play_messages="TOUCHDOWN_TEAM_B"))
        self.assertEqual(state.home_score, 13)   # already there: not added twice

    def test_conversion_gives_the_other_side_the_ball(self):
        state, _ = playover.state_for(self.snap(play_kind="CONVERSION", score_p1="13",
                                                score_p1_at_start="13",
                                                play_messages="EXTRA_POINT_GOOD_TEAM_B"))
        self.assertEqual((state.home_score, state.offense), (14, AWAY))
        self.assertIsNone(state.down)

    def test_field_goals(self):
        state, _ = playover.state_for(self.snap(play_kind="FIELD_GOAL",
                                                play_messages="FIELD_GOAL_GOOD_TEAM_B"))
        self.assertEqual((state.home_score, state.offense), (10, AWAY))
        state, _ = playover.state_for(self.snap(play_kind="FIELD_GOAL", offense="TEAM_A",
                                                field_position="30",
                                                play_messages="FIELD_GOAL_MISSED_TEAM_B"))
        self.assertEqual((state.offense, state.field_position, state.home_score), (AWAY, 30, 7))

    def test_unresolved_team(self):
        state, why = playover.state_for(self.snap(team_a_side=""))
        self.assertIsNone(state)
        self.assertEqual(why, "team_a_unresolved")


def _row(msg, clock_s, kind="SCRIMMAGE", offense="TEAM_A", down="1", distance="10",
         field="30", period="1", p1="0", p2="0"):
    return {"message": str(msg), "period": period, "clock_seconds": str(clock_s),
            "play_kind": kind, "offense": offense, "down": down, "distance": distance,
            "field_position": field, "score_p1": p1, "score_p2": p2}


class TestPlayers(unittest.TestCase):
    def match(self, gap, offense="TEAM_A"):
        rows, clock_s = [], 240
        for i in range(12):
            rows.append(_row(10 + i, clock_s, offense=offense))
            clock_s -= gap
        return rows

    def test_slow_and_fast_players(self):
        matches = {"AF1": self.match(30), "AF2": self.match(15)}
        handles = {"AF1": ("SLOW", "X"), "AF2": ("FAST", "Y")}
        book = players.build(matches, handles)
        self.assertGreater(book.profile("SLOW").pace, 1.0)
        self.assertLess(book.profile("FAST").pace, 1.0)
        self.assertEqual(book.profile("NOBODY").pace, 1.0)

    def test_aggression_reads_4th_down_choices(self):
        def fourth(went):
            a = _row(10, 200, down="4", distance="3", field="40")
            b = _row(11, 190, kind="SCRIMMAGE" if went else "PUNT")
            return [a, b]
        matches = {f"AF{i}": fourth(i % 5 != 0) for i in range(40)}
        handles = {code: ("BOLD", "X") for code in matches}
        book = players.build(matches, handles)
        self.assertGreater(book.profile("BOLD").aggression, 0.0)


class TestStream(unittest.TestCase):
    def test_descriptions_round_trip(self):
        for market, line in ((SPREAD_HOME, -2.5), (SPREAD_AWAY, 3.5), (TOTAL_OVER, 38.5),
                             (TOTAL_UNDER, 41.0)):
            self.assertEqual(stream._parse_line(stream.description(market, line)), line)
        self.assertIsNone(stream._parse_line(stream.description(ML_HOME, None)))


if __name__ == "__main__":
    unittest.main()
