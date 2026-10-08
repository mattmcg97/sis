"""halves: half splits, the second half by half-time margin, touchdown lengths, per gamer."""

import csv
import os
import tempfile
import unittest

from .. import halves, scouting


def _row(code, msg, period, kind, offense, score, nxt=(None, None, None), clock=200, messages="",
         side="home", home="ALPHA", away="BETA", final=(14, 10)):
    r = {f: "" for f in scouting.EXPORT_FIELDS}
    r.update(match_code=code, message=msg, period=period, play_kind=kind, offense=offense,
             scrimmage="0" if kind in ("KICKOFF", "CONVERSION") else "1", clock_seconds=clock,
             score_p1=score[0], score_p2=score[1], play_messages=messages, team_a_side=side,
             home_handle=home, away_handle=away, final_p1=final[0], final_p2=final[1])
    r["next_offense"], r["next_field_position"], r["next_start_clock"] = (
        "" if v is None else v for v in nxt)
    return r


ROWS = [
    # AF1: ALPHA (TEAM_A, home) scores a 60-yard TD on a 75-yard, 2-play drive; 7-3 at half time;
    # BETA's 20-yard TD takes the lead, ALPHA's 50-yarder takes it back: two lead changes
    _row("AF1", 10, 1, "KICKOFF", "TEAM_B", (0, 0), ("TEAM_A", 25, 230), clock=232),
    _row("AF1", 20, 1, "SCRIMMAGE", "TEAM_A", (0, 0), ("TEAM_A", 40, 220), clock=225),
    _row("AF1", 30, 1, "TOUCHDOWN", "TEAM_A", (6, 0), ("TEAM_A", 98, 205), clock=210,
         messages="PLAY_STARTED|TOUCHDOWN_TEAM_A|PLAY_OVER"),
    _row("AF1", 35, 1, "CONVERSION", "TEAM_A", (7, 0), ("TEAM_B", 30, 200), messages="EXTRA_POINT_GOOD_TEAM_A"),
    _row("AF1", 40, 2, "FIELD_GOAL", "TEAM_B", (7, 3), ("TEAM_B", 80, 150), messages="FIELD_GOAL_GOOD_TEAM_B"),
    _row("AF1", 50, 3, "TOUCHDOWN", "TEAM_B", (7, 9), ("TEAM_B", 98, 140), clock=145,
         messages="TOUCHDOWN_TEAM_B"),
    _row("AF1", 51, 3, "CONVERSION", "TEAM_B", (7, 10), ("TEAM_A", 50, 100), messages="EXTRA_POINT_GOOD_TEAM_B"),
    _row("AF1", 60, 4, "TOUCHDOWN", "TEAM_A", (13, 10), ("TEAM_A", 98, 50), clock=90,
         messages="TOUCHDOWN_TEAM_A"),
    _row("AF1", 61, 4, "CONVERSION", "TEAM_A", (14, 10), messages="EXTRA_POINT_GOOD_TEAM_A"),
    # AF2: TEAM_A is the away gamer (BETA); 0-0 at half time, BETA's field goal in Q3
    _row("AF2", 10, 2, "SCRIMMAGE", "TEAM_A", (0, 0), ("TEAM_A", 70, 100), side="away", final=(0, 3)),
    _row("AF2", 20, 3, "FIELD_GOAL", "TEAM_A", (0, 3), side="away", final=(0, 3),
         messages="FIELD_GOAL_GOOD_TEAM_A"),
]


class TestHalves(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "export.csv")
        with open(self.path, "w", newline="") as fh:
            w = csv.DictWriter(fh, scouting.EXPORT_FIELDS)
            w.writeheader()
            w.writerows(ROWS)

    def tearDown(self):
        self.tmp.cleanup()

    def test_halves_lead_changes_and_touchdowns(self):
        got = {code: halves.read_match(code, rows) for code, rows in halves._matches(self.path)}
        m1, tds = got["AF1"]
        self.assertEqual((m1["h1_home"], m1["h1_away"], m1["h2_home"], m1["h2_away"]), (7, 3, 7, 7))
        self.assertEqual((m1["ht_band"], m1["lead_changes_h2"]), ("3-4", 2))
        self.assertEqual([(t["gamer"], t["half"], t["how"], t["length"]) for t in tds],
                         [("ALPHA", 1, "offence", 60), ("BETA", 2, "offence", 20), ("ALPHA", 2, "offence", 50)])
        self.assertEqual((tds[0]["drive_yards"], tds[0]["drive_plays"], tds[0]["drive_seconds"]), (75, 2, 20))
        m2, tds2 = got["AF2"]
        self.assertEqual((m2["home"], m2["away"], m2["ht_band"], m2["h2_away"]), ("ALPHA", "BETA", "level", 3))
        self.assertEqual(tds2, [])

    def test_run_writes_the_league_and_the_gamers(self):
        lines = halves.run(self.path, self.tmp.name, min_games=1)
        text = "\n".join(lines)
        self.assertIn("2 matches", text)
        self.assertRegex(text, r"\n3-4\s+1\s+14\.0\s+2\.00\s+100%\s+100%\s+0%")
        with open(os.path.join(self.tmp.name, "halves_gamers.csv")) as fh:
            g = {r["gamer"]: r for r in csv.DictReader(fh)}
        self.assertEqual((g["ALPHA"]["games"], g["ALPHA"]["tds"], g["ALPHA"]["td_mean_yards"]), ("2", "2", "55.0"))
        self.assertEqual((g["ALPHA"]["won_leading_3_4"], g["BETA"]["won_trailing_3_4"]), ("1.0", "0.0"))
        self.assertEqual(g["ALPHA"]["td_allowed_mean_yards"], "20.0")
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "halves_teams.csv")))

    def test_teams_and_gamer_team_pairs(self):
        # AF2's history row has the handles the other way round: the teams follow the handles
        history = os.path.join(self.tmp.name, "history.csv")
        with open(history, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["MATCH_CODE", "PLAYER_1_HANDLE", "PLAYER_1_TEAM", "PLAYER_2_HANDLE", "PLAYER_2_TEAM"])
            w.writerow(["AF1", "ALPHA", "Detroit Lions", "BETA", "Minnesota Vikings"])
            w.writerow(["AF2", "BETA", "Chicago Bears", "ALPHA", "Detroit Lions"])
        text = "\n".join(halves.run(self.path, self.tmp.name, min_games=1, history_path=history,
                                    min_pair_games=1))
        self.assertIn("TEAMS (2 of 2 matches", text)
        with open(os.path.join(self.tmp.name, "halves_teams.csv")) as fh:
            t = {r["team"]: r for r in csv.DictReader(fh)}
        self.assertEqual((t["Detroit Lions"]["games"], t["Detroit Lions"]["tds"], t["Detroit Lions"]["td_mean_yards"],
                          t["Detroit Lions"]["td_allowed_mean_yards"]), ("2", "2", "55.0", "20.0"))
        self.assertEqual((t["Minnesota Vikings"]["td_mean_yards"], t["Chicago Bears"]["games"],
                          t["Chicago Bears"]["tds"]), ("20.0", "1", "0"))
        with open(os.path.join(self.tmp.name, "halves_gamer_teams.csv")) as fh:
            p = {(r["gamer"], r["team"]): r for r in csv.DictReader(fh)}
        self.assertEqual(set(p), {("ALPHA", "Detroit Lions"), ("BETA", "Minnesota Vikings"), ("BETA", "Chicago Bears")})
        self.assertEqual((p[("ALPHA", "Detroit Lions")]["games"], p[("ALPHA", "Detroit Lions")]["n_td_fit"]), ("2", "2"))

    def test_td_effects_split_gamer_from_team(self):
        tds = [{"how": "offence", "length": 20 + g + t, "gamer": gn, "team": tn, "opponent": "X", "opp_team": "Y"}
               for gn, g in (("G1", 3), ("G2", -3)) for tn, t in (("T1", 2), ("T2", -2))]
        eff, sd, fitted = halves.td_effects(tds)
        self.assertAlmostEqual(eff["gamer"]["G1"], 3)
        self.assertAlmostEqual(eff["team"]["T2"], -2)
        self.assertAlmostEqual(sd, 0)
        self.assertEqual([round(f, 6) for _, f in fitted], [25, 21, 19, 15])


if __name__ == "__main__":
    unittest.main()
