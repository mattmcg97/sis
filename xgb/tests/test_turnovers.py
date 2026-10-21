"""xgb.turnovers: a hand-made match's turnovers, turnovers on downs, punts and touchdowns counted as the sim
reads its snaps, each side's own, and the whole report off a small export."""

import csv
import datetime as dt
import os
import tempfile
import unittest

from .. import turnovers as tv


def row(msg, kind, offense, down, dist, field, clock, messages="", side="home"):
    """One PLAY_OVER row as the export has it: the play that just closed, and the state for the next snap."""
    return dict(match_code="M1", message=str(msg), period="1", clock_seconds=str(clock), play_kind=kind,
                offense=offense, down=str(down), distance=str(dist), field_position=str(field), play_messages=messages,
                score_p1="0", score_p2="0", team_a_side=side)


def match(side="home"):
    """TEAM_A: a drive intercepted; TEAM_B: four downs short; TEAM_A: a punt; TEAM_B: a touchdown; TEAM_A: a
    drive lost on a row the feed also gave a TEAM_A touchdown message (suspect); TEAM_B: a snap."""
    return [row(1, "KICKOFF", "TEAM_A", 1, 10, 25, 236, side=side),
            row(2, "SCRIMMAGE", "TEAM_A", 2, 4, 31, 220, side=side),
            row(3, "SCRIMMAGE", "TEAM_B", 1, 10, 60, 204, "POSSESSION_TEAM_B", side=side),       # intercepted
            row(4, "SCRIMMAGE", "TEAM_B", 2, 10, 60, 188, side=side),
            row(5, "SCRIMMAGE", "TEAM_B", 3, 10, 60, 172, side=side),
            row(6, "SCRIMMAGE", "TEAM_B", 4, 10, 60, 156, side=side),
            row(7, "TURNOVER_ON_DOWNS", "TEAM_A", 1, 10, 40, 150, side=side),                     # on downs
            row(8, "SCRIMMAGE", "TEAM_A", 4, 5, 45, 134, side=side),
            row(9, "PUNT", "TEAM_B", 1, 10, 20, 120, "POSSESSION_TEAM_B", side=side),             # TEAM_A punts
            row(10, "SCRIMMAGE", "TEAM_B", 1, 10, 95, 104, side=side),
            row(11, "TOUCHDOWN", "TEAM_B", 1, 10, 85, 98, "TOUCHDOWN_TEAM_B", side=side),
            row(12, "CONVERSION", "TEAM_B", 1, 10, 35, 98, "EXTRA_POINT_GOOD_TEAM_B", side=side),
            row(13, "KICKOFF", "TEAM_A", 1, 10, 25, 94, "KICKOFF_TEAM_B|POSSESSION_TEAM_A", side=side),
            row(14, "SCRIMMAGE", "TEAM_A", 2, 10, 25, 78, side=side),
            row(15, "SCRIMMAGE", "TEAM_B", 1, 10, 70, 62, "TOUCHDOWN_TEAM_A|POSSESSION_TEAM_B", side=side),
            row(16, "SCRIMMAGE", "TEAM_B", 2, 10, 70, 46, side=side)]


class TestCount(unittest.TestCase):

    def test_a_hand_made_match(self):
        got = tv.count(list(reversed(match())))                 # put back in message order
        a = dict(drives=3, snaps=5, turnovers=2, downs=0, punts=1, tds=0)
        b = dict(drives=3, snaps=7, turnovers=0, downs=1, punts=0, tds=1)
        self.assertEqual({k: got[f"h_{k}"] for k in tv.COUNTS}, a)          # TEAM_A at home
        self.assertEqual({k: got[f"a_{k}"] for k in tv.COUNTS}, b)
        self.assertEqual(got["suspect"], 1)
        away = tv.count(match(side="away"))                     # TEAM_A away: the sides swap
        self.assertEqual({k: away[f"a_{k}"] for k in tv.COUNTS}, a)
        self.assertEqual({k: away[f"h_{k}"] for k in tv.COUNTS}, b)

    def test_the_drives_as_the_sim_counts_them(self):
        from eAMFModel import sim14 as s14
        from eAMFModel.tests.fakes import _matches
        from eAMFModel.v21_report import real_drives
        for code, rows in _matches(n=10, seed=4).items():
            got = tv.count(rows)
            recs = s14.snap_records(rows)
            self.assertEqual(got["h_drives"] + got["a_drives"], real_drives(rows), code)
            self.assertEqual(got["h_snaps"] + got["a_snaps"], len(recs), code)
            self.assertEqual(got["h_turnovers"] + got["a_turnovers"],
                             sum(r["kind"] in (s14.TURNOVER, s14.DEF_TD) for r in recs), code)
            self.assertEqual(got["h_punts"] + got["a_punts"], sum(r["play_kind"] == "PUNT" for r in rows), code)
            self.assertEqual(got["h_tds"] + got["a_tds"], sum(r["play_kind"] == "TOUCHDOWN" for r in rows), code)


class TestRun(unittest.TestCase):

    def test_the_report_off_a_small_export(self):
        from eAMFCalibrator.tests.test_pace_prior import write_export
        from eAMFModel.tests.fakes import _matches
        matches = _matches(n=320, seed=6)
        start = dt.datetime(2026, 8, 20)
        with tempfile.TemporaryDirectory() as d:
            export, history = os.path.join(d, "scouting_playover.csv"), os.path.join(d, "history.csv")
            write_export(export, matches)
            with open(history, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["MATCH_CODE", "SPORT_CODE", "STREAM_NUMBER", "SCHEDULED_START_TIME_UTC", "PLAYER_1_HANDLE",
                            "PLAYER_1_TEAM", "PLAYER_2_HANDLE", "PLAYER_2_TEAM", "PLAYER_1_FINAL_SCORE",
                            "PLAYER_2_FINAL_SCORE"])
                for i, (code, rows) in enumerate(sorted(matches.items())):
                    t = start + dt.timedelta(hours=2 * i)
                    w.writerow([code, "AF", 1, t.strftime("%Y-%m-%d %H:%M:%S"), f"G{i % 6}", f"T{i % 4}",
                                f"G{(i + 2) % 6}", f"T{(i + 1) % 4}", rows[-1]["final_p1"], rows[-1]["final_p2"]])
            out = os.path.join(d, "out")
            logged = []
            L = tv.run(export, history, "2026-09-07", "2026-09-17", out, log=logged.append)
            again = []
            tv.run(export, history, "2026-09-07", "2026-09-17", out, log=again.append)
            with open(os.path.join(out, "turnovers_matches.csv"), newline="") as fh:
                table = list(csv.DictReader(fh))
        text = "\n".join(L)
        for part in ("1. A match's drives against its turnovers", "drives against turnovers: r",
                     "the snaps a drive", "2. A gamer's turnovers", "turnovers a drive given up",
                     "3. Before the match", "expected turnovers: r", "the pace's miss", "4. The margin"):
            self.assertIn(part, text)
        self.assertIn("320 matches counted off", logged[0])
        self.assertIn("counted before", again[0])
        self.assertEqual(len(table), 320)
        self.assertTrue(any(r["exp_turnovers"] for r in table))
        self.assertEqual(table[0]["exp_turnovers"], "")             # before the scored span: not priced


if __name__ == "__main__":
    unittest.main()
