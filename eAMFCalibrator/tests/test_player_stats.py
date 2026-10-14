"""The h2hggl stats rebuilt from SCOUTING_FULL and the history: one short match worked by hand."""

import unittest

from .. import player_stats as ps


def row(msg, clock=None, status=None, kind=None, team=None, down=None, dist=None, field=None):
    """A scouting.fetch_scouting row: match, message, clock, status, in-play message, offence, down,
    distance, field position, file time."""
    return ("M1", msg, clock, status, kind, team, down, dist, field, None)


# TEAM_A (home) takes the kick at its 25 and drives: 6, 2, a 3rd and 2 made for 17, 35 into the red
# zone, then a 15-yard touchdown; the two-point try fails. TEAM_B: 15 yards on a play with a foul by
# TEAM_A, then a 4th and 5 turned over on downs for 2. TEAM_A punts from its 58 to TEAM_B's 20 (net 22).
ROWS = [row(1, 240, "FIRST_QUARTER_STARTED"),
        row(2, 240, kind="KICKOFF_TEAM_B"), row(3, 240, kind="PLAY_STARTED", team="TEAM_B"),
        row(4, 235, kind="PLAY_OVER", team="TEAM_A", down=1, dist=10, field=25),
        row(5, 235, kind="PLAY_STARTED", team="TEAM_A", down=1, dist=10, field=25),
        row(6, 225, kind="PLAY_OVER", team="TEAM_A", down=2, dist=4, field=31),
        row(7, 220, kind="PLAY_STARTED", team="TEAM_A", down=2, dist=4, field=31),
        row(8, 210, kind="PLAY_OVER", team="TEAM_A", down=3, dist=2, field=33),
        row(9, 205, kind="PLAY_STARTED", team="TEAM_A", down=3, dist=2, field=33),
        row(10, 200, kind="PLAY_OVER", team="TEAM_A", down=1, dist=10, field=50),
        row(11, 195, kind="PLAY_STARTED", team="TEAM_A", down=1, dist=10, field=50),
        row(12, 190, kind="PLAY_OVER", team="TEAM_A", down=1, dist=10, field=85),
        row(13, 188, kind="PLAY_STARTED", team="TEAM_A", down=1, dist=10, field=85),
        row(14, 185, kind="TOUCHDOWN_TEAM_A"),
        row(15, 185, kind="PLAY_OVER", team="TEAM_A"),
        row(16, 185, kind="PLAY_STARTED", team="TEAM_A"),
        row(17, 185, kind="TWO_POINT_CONVERSION_FAILED_TEAM_A"),
        row(18, 185, kind="PLAY_OVER", team="TEAM_A"),
        row(19, 185, kind="KICKOFF_TEAM_A"), row(20, 185, kind="PLAY_STARTED", team="TEAM_A"),
        row(21, 180, kind="PLAY_OVER", team="TEAM_B", down=1, dist=10, field=25),
        row(22, 178, kind="PLAY_STARTED", team="TEAM_B", down=1, dist=10, field=25),
        row(23, 175, kind="FOUL_TEAM_A"),
        row(24, 170, kind="PLAY_OVER", team="TEAM_B", down=4, dist=5, field=40),
        row(25, 168, kind="PLAY_STARTED", team="TEAM_B", down=4, dist=5, field=40),
        row(26, 162, kind="TURNOVER_ON_DOWNS_TEAM_B"),
        row(27, 160, kind="PLAY_OVER", team="TEAM_A", down=1, dist=10, field=58),
        row(28, 158, kind="PUNT_TEAM_A"),
        row(29, 158, kind="PLAY_STARTED", team="TEAM_A", down=1, dist=10, field=58),
        row(30, 150, kind="PLAY_OVER", team="TEAM_B", down=1, dist=10, field=20)]
SCORES = [("M1", 15, None, None, None, 6, 0)]
INFO = {"MATCH_CODE": "M1", "SCHEDULED_START_TIME_UTC": "2026-10-01 12:00:00", "PLAYER_1_HANDLE": "ANN",
        "PLAYER_2_HANDLE": "BOB", "PLAYER_1_TEAM": "Ravens", "PLAYER_2_TEAM": "Lions"}


class TestMatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        home, away = ps.match_rows("M1", INFO, ROWS, SCORES, ("6", "0"))
        cls.home, cls.away = home, away

    def test_who_and_the_result(self):
        self.assertEqual((self.home["player"], self.home["opponent"], self.home["result"], self.home["points"]),
                         ("ANN", "BOB", "w", 6))
        self.assertEqual((self.away["player"], self.away["result"], self.away["points_against"]), ("BOB", "l", 6))
        self.assertEqual(self.home["scouting"], 1)

    def test_the_drive_to_the_touchdown(self):
        h = self.home
        self.assertEqual((h["plays"], h["yards_known"], h["yards"]), (5, 5, 6 + 2 + 17 + 35 + 15))
        self.assertEqual((h["third_att"], h["third_conv"], h["fourth_att"]), (1, 1, 0))
        self.assertEqual((h["rz_trips"], h["rz_td"], h["rz_fg"]), (1, 1, 0))
        self.assertEqual((h["tds"], h["twopt_att"], h["twopt_good"]), (1, 1, 0))
        self.assertEqual((h["punts"], h["punts_spotted"], h["punt_yards"]), (1, 1, 22))
        self.assertEqual(h["opp_yards"], 15 + 2)

    def test_the_other_side_turns_it_over_on_downs(self):
        a = self.away
        self.assertEqual((a["plays"], a["yards"], a["clean_plays"], a["yards_clean"]), (2, 17, 1, 2))
        self.assertEqual((a["fourth_att"], a["fourth_conv"], a["tod"], a["tod_msgs"]), (1, 0, 1, 1))
        self.assertEqual((a["rz_trips"], a["turnovers"]), (0, 0))

    def test_fouls_go_to_the_team_on_the_message(self):
        self.assertEqual((self.home["fouls"], self.home["fouls_other"]), (1, 0))
        self.assertEqual((self.away["fouls"], self.away["fouls_other"]), (0, 1))

    def test_time_of_possession_is_the_clock_each_side_had_the_ball(self):
        # ANN: the kick's 5s, then 10+15+10+10+5 on the drive, and the punt's 10
        self.assertEqual(self.home["top_s"], 5 + 10 + 15 + 10 + 10 + 5 + 10)
        # BOB: the kick's 5s, 10 and 10 on his snaps, then the 150s left after the punt
        self.assertEqual(self.away["top_s"], 5 + 10 + 10 + 150)

    def test_a_match_without_scouting_keeps_its_result(self):
        home, away = ps.match_rows("M1", INFO, [], [], ("6", "0"))
        self.assertEqual((home["scouting"], home["result"], home.get("plays")), (0, "w", None))


class TestPlayers(unittest.TestCase):

    def test_results_all_time(self):
        hist = [dict(MATCH_CODE=f"G{k}", SCHEDULED_START_TIME_UTC=f"2026-10-0{k + 1} 12:00:00",
                     PLAYER_1_HANDLE="ann", PLAYER_2_HANDLE="BOB", PLAYER_1_FINAL_SCORE=str(s1),
                     PLAYER_2_FINAL_SCORE=str(s2)) for k, (s1, s2) in enumerate([(21, 7), (14, 14), (21, 3), (28, 0), (3, 7)])]
        r = ps.results(hist)["ANN"]
        self.assertEqual((r["matchesPlayed"], r["matchesWon"], r["matchesLost"], r["matchesDrawn"]), (5, 3, 1, 1))
        self.assertEqual((r["matchesWinPct"], r["matchLongestWinningStreak"], r["matchWinningStreak"]), (60, 2, 0))
        self.assertEqual(r["form"], ["l", "w", "w", "d", "w"])            # newest first
        self.assertAlmostEqual(r["avgPoints"], 87 / 5)

    def test_the_stats_and_the_check(self):
        home, away = ps.match_rows("M1", INFO, ROWS, SCORES, ("6", "0"))
        ours = ps.player_stats([home, away])
        ann = ours["ANN"]
        self.assertEqual((ann["matches_scouted"], ann["matches_detail"]), (1, 1))
        self.assertEqual((ann["redZonEfficiencyPct"], ann["thirdDownConversionPct"]), (100.0, 100.0))
        self.assertAlmostEqual(ann["avgYardsPlay"], 75 / 5)
        self.assertAlmostEqual(ann["avgTimeOfPossession"], 65 / 60)
        self.assertEqual((ours["BOB"]["fourthDownConversionPct"], ours["BOB"]["avgTurnoverOnDowns"]), (0.0, 1.0))
        site = {"ANN": {"participantName": "ANN", "matchesPlayed": "1", "avgYardsPlay": "15.0",
                        "matchForm": '["w"]'}}
        ours["ANN"].update(matchesPlayed=1, form=["w"])
        text = "\n".join(ps.check_lines(site, ours))
        self.assertIn("matchesPlayed", text)
        self.assertIn("100%", text)


if __name__ == "__main__":
    unittest.main()
