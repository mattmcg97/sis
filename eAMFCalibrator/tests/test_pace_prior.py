"""The pace prior: drives counted off the export as the sim counts them, the daily ridge priced off the
results in by each midnight, the tables written with the pace, the walkthrough's pace step, and v21
reading the pace off its prior table."""

import csv
import datetime as dt
import os
import random
import tempfile
import unittest
from unittest import mock

from .. import pace_prior as pp


def write_export(path, matches, split=None):
    """The matches' rows as the scouting export has them, a match at a time; `split`: a match whose rows are
    cut in two, its second half written at the end."""
    cols = sorted({k for rows in matches.values() for r in rows for k in r})
    tail = []
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        for code, rows in matches.items():
            if code == split:
                half = len(rows) // 2
                w.writerows(rows[:half])
                tail = rows[half:]
            else:
                w.writerows(rows)
        w.writerows(tail)


def history_of(days=240, per_day=30, seed=5, start=dt.datetime(2026, 3, 1)):
    """Matches between eight gamers, each of known pace (FAST's matches about 3 drives quicker than SLOW's);
    {match: drives} and the history rows."""
    rng = random.Random(seed)
    pace = {"FAST": 1.5, "SLOW": -1.5, **{f"G{i}": 0.0 for i in range(6)}}
    names = sorted(pace)
    history, drives = [], {}
    for d in range(days):
        for k in range(per_day):
            a, b = rng.sample(names, 2)
            t = start + dt.timedelta(days=d, minutes=40 * k)
            code = f"M{d:03d}{k:02d}"
            history.append(dict(MATCH_CODE=code, SCHEDULED_START_TIME_UTC=t.strftime("%Y-%m-%d %H:%M:%S"),
                                PLAYER_1_HANDLE=a, PLAYER_2_HANDLE=b, PLAYER_1_TEAM=f"T{rng.randrange(4)}",
                                PLAYER_2_TEAM=f"T{rng.randrange(4)}", PLAYER_1_FINAL_SCORE="20",
                                PLAYER_2_FINAL_SCORE="17"))
            drives[code] = max(4, int(round(10 + pace[a] + pace[b] + rng.gauss(0, 2))))
    return history, drives


class TestDrives(unittest.TestCase):

    def test_counted_as_the_sim_counts_them(self):
        from eAMFModel.tests.fakes import _matches
        from eAMFModel.v21_report import real_drives
        matches = _matches(n=12, seed=4)
        warned = []
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "scouting_playover.csv")
            write_export(path, matches)
            got = pp.count_drives(path)
            write_export(path, matches, split="AF003")
            cut = pp.count_drives(path, warn=warned.append)
        want = {c: real_drives(rows) for c, rows in matches.items()}
        self.assertEqual(got, want)
        self.assertTrue(all(n > 3 for n in got.values()))
        self.assertLess(cut["AF003"], want["AF003"])            # its first half alone
        self.assertEqual({c: n for c, n in cut.items() if c != "AF003"}, {c: n for c, n in want.items() if c != "AF003"})
        self.assertIn("1 matches' rows not together", warned[0])

    def test_counted_once_while_the_export_is_the_same(self):
        from eAMFModel.tests.fakes import _matches
        history, _ = history_of(days=20, per_day=20)
        with tempfile.TemporaryDirectory() as d:
            export = os.path.join(d, "x.csv")
            write_export(export, _matches(n=3, seed=4))
            first, _ = pp.run(history, export, dt.date(2026, 3, 15), dt.date(2026, 3, 16), d, log=lambda s: None)
            with mock.patch.object(pp, "count_drives", side_effect=AssertionError("recounted")):
                logged = []
                pp.run(history, export, dt.date(2026, 3, 15), dt.date(2026, 3, 16), d, log=logged.append)
            self.assertIn("kept", logged[0])
            os.utime(export, (1, 1))                        # a new export: counted again
            with mock.patch.object(pp, "count_drives", return_value={}) as counted:
                pp.run(history, export, dt.date(2026, 3, 15), dt.date(2026, 3, 16), d, log=lambda s: None)
            self.assertEqual(counted.call_count, 1)
        self.assertEqual(first, {})                         # three matches' drives: too few to fit


class TestWalk(unittest.TestCase):

    def test_each_day_priced_off_the_results_in_by_its_midnight(self):
        history, drives = history_of()
        first, last = dt.date(2026, 9, 1), dt.date(2026, 9, 3)
        pace = pp.walk(history, drives, first, last)
        today = [h["MATCH_CODE"] for h in history if h["SCHEDULED_START_TIME_UTC"].startswith("2026-09-02")]
        self.assertTrue(set(today) <= set(pace))
        # the day's own results, and every later one, change nothing of that day's prices
        later = {c: (n + 7 if h["SCHEDULED_START_TIME_UTC"] >= "2026-09-02" else n)
                 for h in history for c, n in [(h["MATCH_CODE"], drives[h["MATCH_CODE"]])]}
        moved = pp.walk(history, later, first, last)
        for c in today:
            self.assertAlmostEqual(moved[c][0], pace[c][0], places=9)
        # a result 36 minutes old is in at midnight, a later one isn't
        just = [h for h in history if h["SCHEDULED_START_TIME_UTC"] < "2026-09-02"][-1]
        t = dt.datetime.strptime(just["SCHEDULED_START_TIME_UTC"], "%Y-%m-%d %H:%M:%S")
        self.assertTrue(t + pp.RESULT_LAG <= dt.datetime(2026, 9, 2))
        bumped = dict(drives, **{just["MATCH_CODE"]: drives[just["MATCH_CODE"]] + 20})
        self.assertNotAlmostEqual(pp.walk(history, bumped, first, last)[today[0]][0], pace[today[0]][0], places=6)

    def test_the_quick_gamers_matches_priced_quicker(self):
        history, drives = history_of()
        pace = pp.walk(history, drives, dt.date(2026, 9, 1), dt.date(2026, 9, 10))
        by = {h["MATCH_CODE"]: {h["PLAYER_1_HANDLE"], h["PLAYER_2_HANDLE"]} for h in history}
        fast = [pace[c][0] for c in pace if "FAST" in by[c] and "SLOW" not in by[c]]
        slow = [pace[c][0] for c in pace if "SLOW" in by[c] and "FAST" not in by[c]]
        self.assertGreater(sum(fast) / len(fast) - sum(slow) / len(slow), 2.0)
        league = {round(lg, 9) for _, lg in pace.values()}
        self.assertEqual(len(league), 10)                   # one league level a day
        for lg in league:
            self.assertAlmostEqual(lg, 10.0, delta=0.6)

    def test_too_few_results_and_nothing_priced(self):
        history, drives = history_of(days=12, per_day=10)
        self.assertEqual(pp.walk(history, drives, dt.date(2026, 3, 5), dt.date(2026, 3, 12)), {})
        lines = pp.lines({}, drives)
        self.assertIn("0 matches priced", lines[0])


class TestTables(unittest.TestCase):

    def test_the_pace_written_into_a_table_and_read_back_by_v21(self):
        from eAMFModel import v21_stream as vs
        with tempfile.TemporaryDirectory() as d:
            table = os.path.join(d, "avg_form4_w05.csv")
            with open(table, "w", newline="") as fh:
                fh.write("MATCH_CODE,home_pts,away_pts\nA,20,17\nB,22,19\nC,18,18\n")
            with self.assertRaises(SystemExit) as no_pace:
                vs.drives_table(None, table)
            self.assertIn("pace-prior", str(no_pace.exception))
            # A and B both a drive above their own day's league, which rose a drive between them
            n = pp.add_to_table(table, {"A": (11.0, 10.0), "B": (12.0, 11.0)})
            with open(table, newline="") as fh:
                rows = list(csv.DictReader(fh))
            got = vs.drives_table(None, table)
            with mock.patch.dict(os.environ, {"EAMF_V21_DRIVES": table}):
                self.assertEqual(vs.drives_table(None, None), got)
        self.assertEqual(n, 2)
        self.assertEqual([r["exp_drives"] for r in rows], ["11.0", "12.0", ""])
        self.assertEqual(rows[0]["home_pts"], "20")
        self.assertEqual(set(got), {"A", "B"})
        # each against its day's league, at the table's average league (10.5): A 11/10, B 12/11
        self.assertAlmostEqual(got["A"], 11.0 / 10.0 * 10.5)
        self.assertAlmostEqual(got["B"], 12.0 / 11.0 * 10.5)

    def test_a_table_without_league_drives_read_as_it_is(self):
        from eAMFModel import v21_stream as vs
        with tempfile.TemporaryDirectory() as d:
            table = os.path.join(d, "pace_eff_prior.csv")
            with open(table, "w", newline="") as fh:
                fh.write("MATCH_CODE,exp_drives\nA,12.5\nB,7.5\n")
            self.assertEqual(vs.drives_table(table, "some/build"), {"A": 12.5, "B": 7.5})
            with self.assertRaises(SystemExit):
                vs.drives_table(None, d)                   # a build's directory, no pace table named


class TestWalkthrough(unittest.TestCase):

    def test_the_walkthrough_writes_the_pace_into_every_table(self):
        import json
        from eAMFModel import nb2_prior, rolling_prior
        from .. import prior_backtest as pb, walkthrough as wt
        from . import test_prior_backtest as tpb
        from .test_nb2_history import write_rolling
        from .test_prior_offline import write_history
        from .test_walkthrough import cadence_runner
        hist, _, arm = tpb.fixture()
        mus = {line.split(",")[0]: tuple(map(float, line.split(",")[1:])) for line in arm[1:]}

        def fit(history, prior, directory, first, last, workers=4, verbose=True, options=None, on_fit=None):
            write_rolling(directory, [(first + dt.timedelta(days=k)).isoformat() for k in range((last - first).days + 1)])

        def read_rolling(directory, rows, history, progress=None):
            codes = {r["MATCH_CODE"] for r in rows}
            return ({c: mus[c] for c in codes}, {c: ("nb2", mus[c] + (0.15, 0.15, 0.4)) for c in codes})

        seen = {}

        def pace_run(history, export, first, last, cache_dir, on_match=None, on_day=None, log=print):
            seen.update(export=export, first=first, last=last, cache=cache_dir)
            log("  drives: 2 matches off the export")
            on_day(first)
            return {"AF00100": (11.2, 10.4), "AF00300": (9.6, 10.4)}, {"AF00100": 12, "AF00300": 9}

        with tempfile.TemporaryDirectory() as d:
            history = os.path.join(d, "history.csv")
            write_history(history, hist)
            out = os.path.join(d, "wt")
            lines = []
            with mock.patch.object(rolling_prior, "fit", fit), mock.patch.object(pb, "read_rolling", read_rolling), \
                    mock.patch.object(nb2_prior.Prematch, "exists",
                                      staticmethod(lambda p: os.path.exists(os.path.join(p, "level.json")))), \
                    mock.patch.object(pp, "run", pace_run), \
                    mock.patch("builtins.print", lambda *a, **k: lines.append(" ".join(map(str, a)))):
                path = wt.run(out, history, "2026-09-01 00:00:00", cores=2, draws=500,
                              cadence_runner=cadence_runner(arm, hist), scouting="scouting_playover.csv")
            with open(path) as fh:
                manifest = json.load(fh)
            tables = {}
            for name, _ in wt.VARIANTS:
                with open(manifest["variants"][name]["file"], newline="") as fh:
                    tables[name] = {r["MATCH_CODE"]: r for r in csv.DictReader(fh)}
            with open(os.path.join(out, "walkthrough.txt")) as fh:
                text = fh.read()
        for t in tables.values():
            self.assertEqual((t["AF00100"]["exp_drives"], t["AF00100"]["league_drives"]), ("11.2", "10.4"))
            self.assertEqual(t["AF00000"]["exp_drives"], "")
        self.assertEqual(seen["cache"], os.path.join(out, "pace"))
        self.assertEqual(seen["export"], "scouting_playover.csv")
        self.assertIn("pace", manifest["seconds"])
        self.assertEqual(manifest["pace_matches"], 2)
        self.assertIn("pace: 2 matches priced", text)
        self.assertIn("== the pace", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
