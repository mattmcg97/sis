"""strength_check: which kinds of play a side's strength moves, the sim's snaps against the real ones,
and the day's strength draw against fixed strengths."""

import csv
import datetime as dt
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from .. import sim14, strength_check as sc, v14
from .fakes import _matches
from .test_prior_variants import FakePrior


class TestClasses(unittest.TestCase):

    def test_each_snap_in_one_class(self):
        kind = [sim14.TURNOVER, sim14.GAIN, sim14.GAIN, sim14.GAIN, sim14.GAIN, sim14.GAIN, sim14.GAIN, sim14.GAIN]
        gain = [5, -3, 0, 4, 12, 25, 60, 50]
        dist = [10, 10, 10, 10, 10, 10, 10, 10]
        field = [30, 30, 30, 30, 30, 30, 30, 70]           # the last: 50 yards drawn at the 70, 30 to the goal
        self.assertEqual(list(sc.snap_class(kind, gain, dist, field)), [0, 1, 2, 3, 4, 5, 6, 5])

    def test_real_snaps_by_side(self):
        rows = _matches(n=1)["AF000"]
        counts = sc.real_counts(rows)
        recs = [r for r in sim14.snap_records(rows) if r["period"] <= 4 and not sim14.is_kneel(r)]
        self.assertEqual(counts.sum(), len(recs))
        self.assertEqual(counts[0].sum(), sum(r["offense"] == "TEAM_A" for r in recs))


class TestInputs(unittest.TestCase):

    def test_only_the_windows_matches_are_read(self):
        history = [{"MATCH_CODE": "AF000", "SCHEDULED_START_TIME_UTC": "2026-10-03 12:00:00",
                    "PLAYER_1_FINAL_SCORE": "21", "PLAYER_2_FINAL_SCORE": "14"},
                   {"MATCH_CODE": "AF001", "SCHEDULED_START_TIME_UTC": "2026-09-03 12:00:00",
                    "PLAYER_1_FINAL_SCORE": "21", "PLAYER_2_FINAL_SCORE": "14"},
                   {"MATCH_CODE": "AF002", "SCHEDULED_START_TIME_UTC": "2026-10-04 12:00:00",
                    "PLAYER_1_FINAL_SCORE": "", "PLAYER_2_FINAL_SCORE": ""}]
        self.assertEqual(sc.window_codes(history, dt.date(2026, 10, 1), dt.date(2026, 10, 12)), {"AF000"})
        matches = _matches(n=3)
        rows = [r for rs in matches.values() for r in rs][::-1]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snaps.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            got = sc.load_snapshots(path, {"AF001"})
        self.assertEqual(list(got), ["AF001"])
        self.assertEqual([r["message"] for r in got["AF001"]], [r["message"] for r in matches["AF001"]])

    def test_the_command(self):
        from .. import __main__ as cli
        args = cli.build_parser().parse_args(["strength-check", "--since", "2026-10-01", "--no-matches"])
        self.assertEqual((args.model, args.paths, args.no_matches), ("v14_model", 2000, True))


class TestStrengthCheck(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches()
        cls.tables = sim14.Tables.build(cls.matches, min_records=20)
        cls.tables.strength_game, cls.tables.strength_league = 0.02, 0.0
        cls.tables.strength_theta = np.array([-1.0, 0.0, 1.0])
        cls.tables.strength_slope = np.array([1.0, 1.0, 1.0])
        cls.grid = v14.PriorGrid.build(cls.tables, n_paths=60)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = os.path.join(self.tmp.name, "v14_model")
        os.makedirs(self.model)
        self.tables.save(os.path.join(self.model, "v14tables.npz"))
        self.grid.save(os.path.join(self.model, "v14grid.npz"))
        with open(os.path.join(self.model, v14.PRIOR_FILE), "w") as fh:
            json.dump({"prior": "glmer"}, fh)
        FakePrior.build([], os.path.join(self.model, "glmer"), dt.datetime(2026, 10, 1))
        self.patches = [mock.patch.dict(v14.PRIORS, {"glmer": FakePrior}),
                        mock.patch.object(v14, "HOLD_PATHS", 40)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_the_sweep_steps_one_side_and_leaves_the_switches(self):
        boom, tilt = sim14.BOOM, sim14.TILT_SCALE
        rows = sc.sweep(self.model, n_paths=40)
        self.assertEqual(len(rows), len(sc.SWEEP_POINTS) * len(sc.CHANNELS))
        self.assertEqual((sim14.BOOM, sim14.TILT_SCALE), (boom, tilt))
        built = [r for r in rows if r[1] == "as built"]
        self.assertLess(built[0][2][0], built[-1][2][0])               # stronger for more points
        self.assertEqual(built[0][3]["counts"].shape, (2, sc.N_CLASSES))
        self.assertIn("A. FROM KICK-OFF", "\n".join(sc.sweep_lines(rows)))

    def test_the_matches_real_against_the_draw_and_fixed(self):
        history = [{"MATCH_CODE": code, "PLAYER_1_HANDLE": "ALPHA", "PLAYER_2_HANDLE": "BRAVO",
                    "PLAYER_1_TEAM": "Lions", "PLAYER_2_TEAM": "Bears", "STREAM_NUMBER": "1",
                    "SCHEDULED_START_TIME_UTC": "2026-10-03 12:00:00",
                    "PLAYER_1_FINAL_SCORE": rows[-1]["final_p1"], "PLAYER_2_FINAL_SCORE": rows[-1]["final_p2"]}
                   for code, rows in list(self.matches.items())[:4]]
        snaps = {r["MATCH_CODE"]: self.matches[r["MATCH_CODE"]] for r in history}
        got = sc.run(self.model, history, snaps, dt.date(2026, 10, 1), dt.date(2026, 10, 12), n_paths=40)
        self.assertEqual(len(got), 4)
        x = got[0]
        self.assertEqual(set(x["sims"]), {"draw", "fixed"})
        self.assertEqual(x["real"].shape, (2, sc.N_CLASSES))
        self.assertGreater(x["sims"]["draw"]["counts"].sum(), 0)
        self.assertEqual(x["sims"]["draw"]["games"], 40)
        text = "\n".join(sc.report([], got, dt.date(2026, 10, 1), dt.date(2026, 10, 12)))
        for s in ("EXPECTED POINTS", "EXPECTED MARGIN", "STRENGTH DRAW AGAINST FIXED", "log loss, fixed less draw",
                  "draw-real"):
            self.assertIn(s, text)
        path = os.path.join(self.tmp.name, "s.csv")
        sc.write_csv(path, got)
        with open(path) as fh:
            self.assertEqual(len(list(csv.reader(fh))), 1 + 2 * 4)

    def test_fixed_strengths_draw_nothing(self):
        fixed = sc.fixed_tables(self.tables)
        theta, own, game = sim14.strength_draw(fixed, (0.1, -0.1), [0.01, 0.01])
        self.assertEqual(tuple(theta), (0.1, -0.1))
        self.assertFalse(np.any(own) or np.any(game))
        self.assertTrue(len(self.tables.strength_slope))                # the build's own untouched
        theta, own, game = sim14.strength_draw(self.tables, (0.1, -0.1), [0.01, 0.01])
        self.assertTrue(np.all(np.asarray(theta) < (0.1, -0.1)))          # moved down to keep the mean


if __name__ == "__main__":
    unittest.main()
