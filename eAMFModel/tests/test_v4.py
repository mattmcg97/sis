"""v4: v3's simulation with in-game quarter calibration, common random
numbers and player profiles (sim4.py, v4.py, v4_stream.py)."""

import csv
import inspect
import os
import random
import tempfile
import unittest

import numpy as np

from .. import sim, sim4, v4, v4_stream
from .test_v3 import _matches


def _with_handles(matches):
    rng = random.Random(7)
    names = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    for rows in matches.values():
        home, away = rng.sample(names, 2)
        for r in rows:
            r["home_handle"], r["away_handle"] = home, away
    return matches


class TestCommonRandomNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = sim4.Tables.build(_matches(), min_records=20)

    def _two_of(self, common):
        st = sim4.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim4.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 14, 10
        return sim4.simulate(self.tables, st, 300, np.random.default_rng(1), common=common)

    def test_the_same_state_twice_plays_the_same_games(self):
        h, a = self._two_of(True)
        self.assertTrue((h[0] == h[1]).all() and (a[0] == a[1]).all())
        h, a = self._two_of(False)
        self.assertFalse((h[0] == h[1]).all() and (a[0] == a[1]).all())

    def test_the_stream_is_uniform_and_repeatable(self):
        path, step = np.arange(20000), np.full(20000, 3)
        u = sim4._uniform(12345, path, step, 7)
        self.assertTrue(((u >= 0) & (u < 1)).all())
        self.assertAlmostEqual(float(u.mean()), 0.5, delta=0.01)
        self.assertTrue((u == sim4._uniform(12345, path, step, 7)).all())
        self.assertFalse((u == sim4._uniform(12345, path, step, 8)).all())

    def test_v3_is_left_as_it_was(self):
        self.assertNotIn("common", inspect.signature(sim.simulate).parameters)


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.matches = _with_handles(_matches(40))
        v4.build(cls.matches, cls.tmp.name, grid_paths=60, verbose=False)
        cls.tables = sim4.Tables.load(os.path.join(cls.tmp.name, "v4tables.npz"))
        cls.grid = v4.PriorGrid.load(os.path.join(cls.tmp.name, "v4grid.npz"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_writes_the_model_and_the_profiles(self):
        for name in ("v4tables.npz", "v4grid.npz", "v4players.json"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp.name, name)), name)
        book = v4.players_book(self.tmp.name)
        self.assertEqual(set(book.players), {"ALPHA", "BRAVO", "CHARLIE", "DELTA"})

    def test_the_quarter_fit_on_real_states_converges_when_asked(self):
        # kept as a diagnostic (the build only reports it): on fixed states
        # and thetas the fit does bring the quarters to what games scored
        import copy
        tables = copy.deepcopy(self.tables)
        items = v4.quarter_start_states(self.matches, self.grid)
        v4.fit_period_theta_states(tables, items, rounds=6, n_paths=300)
        _, got, real = v4.fit_period_theta_states(tables, items, rounds=1, n_paths=300)
        for q in real:
            self.assertAlmostEqual(got[q], real[q], delta=max(0.4, 0.08 * real[q]))

    def test_the_recent_total_shade(self):
        import datetime as dt
        matches = {}
        for i in range(60):
            day = dt.date(2026, 9, 1) + dt.timedelta(days=i % 10)
            total = 20 if i % 4 else 40            # 25% go over a line of 30.5
            matches[f"AF{i:03d}"] = [{"match_code": f"AF{i:03d}", "file_time": f"{day} 10:00:00",
                                       "prematch_line_52": "0.5", "prematch_prob_52": "0.5",
                                       "prematch_line_54": "30.5", "prematch_prob_54": "0.5",
                                       "prematch_prob_50": "0.5",
                                       "final_p1": str(total // 2), "final_p2": str(total - total // 2)}]
        shade, n, rate, prod = v4.recent_total_shade(matches, days=7, prior_n=200)
        recent = [i for i in range(60) if i % 10 >= 3]          # the last 7 of the 10 days
        self.assertEqual(n, len(recent))
        self.assertAlmostEqual(rate, sum(i % 4 == 0 for i in recent) / len(recent), places=9)
        self.assertAlmostEqual(prod, 0.5, places=9)
        self.assertAlmostEqual(shade, (rate - 0.5) * n / (n + 200), places=9)
        self.assertEqual(v4.recent_total_shade({}, 7)[0], 0.0)

    def test_the_shade_lowers_the_prior_level(self):
        import copy
        grid = copy.deepcopy(self.grid)
        grid.total_shade = 0.0
        plain = sum(grid.fit(0.5, 0.5, 30.5, 0.5, 0.5))
        grid.total_shade = -0.08
        self.assertLess(sum(grid.fit(0.5, 0.5, 30.5, 0.5, 0.5)), plain)

    def test_handles_come_off_the_export(self):
        rows = next(iter(self.matches.values()))
        self.assertEqual(v4.handles_of(rows), (rows[0]["home_handle"], rows[0]["away_handle"]))
        self.assertIsNone(v4.handles_of([{"home_handle": "", "away_handle": None}]))

    def test_grades_a_snapshot_file_with_profiles(self):
        rng = random.Random(1)
        for rows in self.matches.values():
            for r in rows[::5]:
                total = int(r["final_p1"]) + int(r["final_p2"])
                r.update(line_54="30.5", prob_54=str(round(rng.uniform(0.3, 0.7), 3)), live_54="1",
                         outcome_54=str(int(total > 30.5)))
        path = os.path.join(self.tmp.name, "snaps.csv")
        fields = list(next(iter(self.matches.values()))[0])
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fields)
            w.writeheader()
            for rows in self.matches.values():
                w.writerows(rows)
        graded, _ = v4.run(path, os.path.join(self.tmp.name, "v4tables.npz"),
                           os.path.join(self.tmp.name, "v4grid.npz"),
                           [v4.Variant("v4"), v4.Variant("plain", profiles=False, pace=False)],
                           n_paths=100, workers=1)
        self.assertGreater(len(graded), 50)
        for _, row, probs in graded:
            self.assertTrue(0 < probs["v4"] < 1 and 0 < probs["plain"] < 1)

    def test_the_stream_finds_the_model_and_quotes(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        prod = [(code, 54, None, 50.0, 2.0, "Total points over 30.5", int(r["message"]), "OPEN", "true")
                for r in rows]
        quotes = v4_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
        self.assertTrue(quotes)
        self.assertTrue(all(q[1] == 54 and 0 < q[3] < 100 for q in quotes))

    def test_a_missing_model_says_how_to_build_it(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(SystemExit) as caught:
                v4_stream.model_paths(empty)
        self.assertIn("v4-build", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
