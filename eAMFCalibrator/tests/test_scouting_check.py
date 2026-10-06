"""scouting_check: each match day's scheduled matches against SCOUTING_FULL, judged on a baseline."""

import datetime as dt
import os
import tempfile
import unittest

from .. import scouting, scouting_check as sc

D = dt.date
T = dt.datetime
COLUMNS = {"MATCH_CODE": "TEXT", "EVENT_MESSAGE_COUNT": "NUMBER", scouting.CLOCK: "NUMBER",
           scouting.STATUS: "TEXT", scouting.MESSAGE: "TEXT", scouting.TEAM: "TEXT",
           scouting.DOWN: "NUMBER", scouting.DIST: "NUMBER", scouting.FIELD: "TEXT",
           "FILE_TIME": "TIMESTAMP_NTZ", "FILE_LOADED": "TIMESTAMP_NTZ"}
MATCH_COLS = ["MATCH_CODE", "MESSAGES", "DISTINCT_COUNTS", "FIRST_COUNT", "LAST_COUNT", "FIRST_TIME",
              "LAST_TIME", "LAST_LOADED", "RAW_ROWS", "PLAY_OVERS", "PLAY_STARTS", "STARTED",
              "ENDED", "QUARTERS", "PO_FIELD", "PO_TEAM", "PO_DOWN", "PO_DD", "DD", "CLOCK_FILLED", "POINTS_A",
              "POINTS_B"]


def match(code, when, messages=300, first=1, gaps=0, plays=60, ended=1, quarters=4, field=60,
          clock=300, points=(21, 14), raw=None, dd=None):
    dd = plays if dd is None else dd
    return (code, messages, messages - gaps, first, first + messages - 1, when,
            when + dt.timedelta(minutes=20), when + dt.timedelta(minutes=25), raw or messages,
            plays, plays, 1, ended, quarters, field, field, field, dd, dd, clock, points[0], points[1])


DETAIL_COLS = ["K", "N", "MATCHES", "N_PO", "DOWN_OK", "DIST_OK", "DD", "FIELD_OK", "TEAM_OK", "DD_PO",
               "FIELD_PO", "TEAM_PO", "MATCHES_DD", "FIRST_DD", "LAST_DD", "LAST_LOADED"]


def detail(when, n_po, dd_po, last_dd=None, matches=10):
    """A detail_sql row: n_po PLAY_OVERs (of 6x as many messages), dd_po with down & distance."""
    return (when, 6 * n_po, matches, n_po, dd_po, dd_po, dd_po, n_po, n_po, dd_po, n_po, n_po,
            matches if dd_po else 0, when if dd_po else None, last_dd, when)


class FakeCursor:
    """Answers the check's queries from canned rows."""

    def __init__(self, expected, matches, fills, kinds, days=(), hours=(), values=()):
        self.answers = {"expected": (["MATCH_CODE", "S", "STREAM", "P1", "P2"], expected),
                        "matches": (MATCH_COLS, matches), "fills": fills, "kinds": kinds,
                        "days": (DETAIL_COLS, list(days)), "hours": (DETAIL_COLS, list(hours)),
                        "values": (["DAY", "COL", "VAL", "N"], list(values))}
        self.sql = []

    def execute(self, sql, params=()):
        self.sql.append(sql)
        if "SPORT_CODE" in sql:
            key = "expected"
        elif "AS MESSAGES" in sql:
            key = "matches"
        elif "MATCHES_DD" in sql:
            key = "hours" if "DATE_TRUNC" in sql else "days"
        elif "AS COL" in sql:
            key = "values"
        elif "N_PO" in sql:
            key = "fills"
        else:
            key = "kinds"
        cols, self.rows = self.answers[key]
        self.description = [(c,) for c in cols]

    def fetchall(self):
        return list(self.rows)


class TestScoutingCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = scouting.Table("DB.S.SCOUTING_FULL", COLUMNS)
        self.now = T(2026, 9, 25, 12)

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, expected, matches, fills=None, kinds=None, **detail_rows):
        checked = [c for c in COLUMNS]
        fills = fills or (["DAY", "N", "N_PO"] + [f"{p}{k}" for k in range(len(checked)) for p in "CP"], [])
        cur = FakeCursor(expected, matches, fills, kinds or (["DAY", "KIND", "N"], []), **detail_rows)
        lines = sc.run(cur, self.table, D(2026, 9, 21), D(2026, 9, 25), D(2026, 9, 23),
                       self.tmp.name, now=self.now)
        return cur, lines

    def test_flags_each_kind_of_hole_against_the_baseline(self):
        day = lambda d, h=12: T(2026, 9, d, h)
        expected = [(f"AF{k}", day(d), "1", 21, 14) for k, d in
                    ((1, 21), (2, 21), (3, 22), (4, 23), (5, 23), (6, 24), (7, 24), (8, 24), (9, 25))]
        expected.append(("AF10", day(25, 23), "2", None, None))         # later today: not judged
        matches = [match("AF1", day(21)), match("AF2", day(21)), match("AF3", day(22)),
                   match("AF4", day(23), gaps=5, raw=320),                      # holes in the sequence
                   match("AF5", day(23), ended=0, quarters=3, points=(14, 14)),  # cut off in Q3
                   match("AF6", day(24), field=0, dd=0),                        # detail gone
                   match("AF7", day(24), messages=100, plays=20, first=200),    # most of it lost
                   match("AF9", day(25, 10), ended=0, points=(7, 0)),           # started 2h ago
                   match("AF11", day(24))]                                      # not in EVENT
        _, lines = self._run(expected, matches)
        import csv
        with open(os.path.join(self.tmp.name, "scouting_check_matches.csv")) as fh:
            got = {r["match_code"]: r for r in csv.DictReader(fh)}
        self.assertEqual(got["AF1"]["problems"], "")
        self.assertEqual(got["AF4"]["problems"], "GAPS")
        self.assertIn("DUPLICATES", got["AF4"]["flags"])
        self.assertEqual(got["AF5"]["problems"], "NO_END QUARTERS SCORE")
        self.assertEqual(got["AF6"]["problems"], "NO_DETAIL NO_DOWN")
        self.assertEqual(got["AF7"]["problems"], "HEAD_MISSING FEW_MESSAGES FEW_PLAYS")
        self.assertEqual(got["AF8"]["problems"], "MISSING")
        self.assertEqual(got["AF9"]["flags"], "LIVE")
        self.assertEqual(got["AF10"]["flags"], "NO_FINAL")
        self.assertEqual(got["AF11"]["problems"], "")
        self.assertIn("NOT_SCHEDULED", got["AF11"]["flags"])

        with open(os.path.join(self.tmp.name, "scouting_check_days.csv")) as fh:
            days = {r["day"]: r for r in csv.DictReader(fh)}
        self.assertEqual((days["2026-09-21"]["complete"], days["2026-09-21"]["period"]), ("2", "baseline"))
        self.assertEqual((days["2026-09-24"]["scheduled"], days["2026-09-24"]["in_scouting"],
                          days["2026-09-24"]["complete"], days["2026-09-24"]["incomplete"],
                          days["2026-09-24"]["missing"]), ("3", "3", "1", "2", "1"))
        self.assertEqual(days["2026-09-25"]["live"], "1")
        text = "\n".join(lines)
        self.assertIn("Days with a missing or incomplete match: 2026-09-23, 2026-09-24", text)
        self.assertIn("Missing matches by stream: 1 1", text)

    def test_reports_columns_and_message_kinds_that_change_after_the_break(self):
        checked = list(COLUMNS)
        field = checked.index(scouting.FIELD)
        cols = ["DAY", "N", "N_PO"] + [f"{p}{k}" for k in range(len(checked)) for p in "CP"]

        def fill(day, field_po):
            vals = {f"C{k}": 100 for k in range(len(checked))}
            vals.update({f"P{k}": 20 for k in range(len(checked))})
            vals[f"P{field}"] = field_po
            vals[f"C{field}"] = 80 + field_po
            return [day, 100, 20] + [vals[c] for c in cols[3:]]

        fills = (cols, [fill(D(2026, 9, d), 20 if d < 23 else 0) for d in range(21, 26)])
        kinds = (["DAY", "KIND", "N"],
                 [(D(2026, 9, d), "PLAY_OVER", 60) for d in range(21, 26)]
                 + [(D(2026, 9, d), "POSSESSION_TEAM_A", 10) for d in (21, 22)])
        expected = [(f"AF{d}", T(2026, 9, d, 8), "1", 21, 14) for d in range(21, 26)]
        matches = [match(f"AF{d}", T(2026, 9, d, 8)) for d in range(21, 26)]
        cur, lines = self._run(expected, matches, fills, kinds)
        text = "\n".join(lines)
        self.assertRegex(text, rf"{scouting.FIELD}\s+PLAY_OVER\s+baseline\s+100%\s+first off 2026-09-23")
        self.assertNotRegex(text, rf"{scouting.CLOCK}\s+")
        self.assertRegex(text, r"POSSESSION_TEAM_A\s+baseline\s+10\.00/match\s+first off 2026-09-23")
        self.assertNotIn("PLAY_OVER     ", text.split("Message kinds")[1])
        # the per-match query reads a day either side, so a match over midnight is whole
        self.assertTrue(any("QUALIFY ROW_NUMBER()" in q for q in cur.sql))

    def test_down_and_distance_day_by_day_and_the_hour_it_stopped(self):
        days = [detail(D(2026, 9, d), 600, 600, T(2026, 9, d, 23, 50)) for d in (21, 22, 23)]
        days += [detail(D(2026, 9, 24), 600, 250, T(2026, 9, 24, 9, 58)), detail(D(2026, 9, 25), 600, 0)]
        hours = [detail(T(2026, 9, 24, h), 50, 50 if h < 10 else 0) for h in range(8, 13)]
        values = ([(D(2026, 9, 24), scouting.DOWN, v, n) for v, n in (("(none)", 350), ("1", 100), ("2", 80))]
                  + [(D(2026, 9, 25), scouting.DOWN, "(none)", 600)])
        expected = [(f"AF{d}", T(2026, 9, d, 8), "1", 21, 14) for d in range(21, 26)]
        matches = [match(f"AF{d}", T(2026, 9, d, 8), dd=60 if d < 24 else 0) for d in range(21, 26)]
        _, lines = self._run(expected, matches, days=days, hours=hours, values=values)
        text = "\n".join(lines)
        self.assertIn("down & distance on 100% of PLAY_OVERs", text)
        self.assertRegex(text, r"2026-09-24\s+3,600\s+250\s+7%\s+600\s+250\s+42%")
        self.assertRegex(text, r"2026-09-25\s+3,600\s+0\s+0%\s+600\s+0\s+0%")
        self.assertIn("Last message with a down and distance: 2026-09-24 09:58; since then 150 PLAY_OVERs"
                      " in 3 hours", text)
        self.assertRegex(text, r"2026-09-24 09:00\s+100% \(50\)\s+->\s+2026-09-24 10:00\s+0% \(50\)")
        self.assertIn("2026-09-24  (none) 58%  1 17%  2 13%", text)
        import csv
        with open(os.path.join(self.tmp.name, "scouting_check_matches.csv")) as fh:
            got = {r["match_code"]: r["problems"] for r in csv.DictReader(fh)}
        self.assertEqual((got["AF22"], got["AF24"]), ("", "NO_DOWN"))
        with open(os.path.join(self.tmp.name, "scouting_check_detail_hours.csv")) as fh:
            self.assertEqual([r["dd_po_share"] for r in csv.DictReader(fh)], ["1.0", "1.0", "0.0", "0.0", "0.0"])


if __name__ == "__main__":
    unittest.main()
