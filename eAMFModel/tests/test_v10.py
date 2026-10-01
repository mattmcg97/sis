"""v10: v9 (copied as it was) learning the day's scoring from the game so far -- each simulated
game counts by how likely its draws make the points each side has scored (sim10.py, v10.py,
v10_stream.py)."""

import copy
import os
import tempfile
import unittest

import numpy as np

from .. import players, sim9, sim10, v9, v10, v10_stream
from .test_v3 import _matches


def _with_form(tables, game=0.04):
    """The tables with the day's form on: a shared swing of `game` (variance in log points)."""
    tables = copy.deepcopy(tables)
    _, _, slope = v10.fixed_strength_spread(tables, n_paths=400)
    tables.strength_theta, tables.strength_slope = v10.FORM_GRID.copy(), slope
    tables.strength_game, tables.strength_league = game, 0.0
    return tables


def _state(period, clock, home, away):
    st = sim10.Start(1)
    st.period[:], st.clock[:], st.phase[:] = period, clock, sim10.SCRIM
    st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, home, away
    return st


class TestV10(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches()
        base = sim10.Tables.build(cls.matches, min_records=20)
        cls.tables = _with_form(base)
        cls.grid = v10.PriorGrid.build(cls.tables, n_paths=300)

    def test_the_draws_are_the_ones_the_simulation_plays(self):
        # sim10 draws through path_draws; sim9 the old way: the same games, path for path
        t9 = copy.deepcopy(self.tables)
        st = _state(2, 100.0, 10, 7)
        st.theta[0], st.strength[0], st.strength_game[0] = sim10.strength_draw(
            self.tables, [0.0, 0.0], [0.02, 0.02])
        a = sim10.simulate(self.tables, st, 400, np.random.default_rng(1), seed=11)
        b = sim9.simulate(t9, st, 400, np.random.default_rng(1), seed=11)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
        # and a path's shared draw is what moves its points: higher draw, more points
        st = _state(1, 240.0, 0, 0)
        st.theta[0], st.strength[0], st.strength_game[0] = sim10.strength_draw(
            self.tables, [0.0, 0.0], [0.0, 0.0])
        h, a = sim10.simulate(self.tables, st, 1500, np.random.default_rng(1), seed=11)
        _, game = sim10.path_draws(11, 1500)
        self.assertGreater(np.corrcoef(game, h[0] + a[0])[0, 1], 0.1)

    def test_a_slow_start_weights_the_low_draws_and_a_fast_one_the_high(self):
        _, game = sim10.path_draws(5, 2000)
        e = v10.expected_points(self.grid, (0.0, 0.0))
        half = e * v10.elapsed_share(self.tables, 3, 240.0)
        slow = v10.learn_weights(self.tables, self.grid, (0.0, 0.0), (0.0, 0.0), 3, 240.0,
                                 (0, 0), 5, 2000)
        fast = v10.learn_weights(self.tables, self.grid, (0.0, 0.0), (0.0, 0.0), 3, 240.0,
                                 np.round(half * 2), 5, 2000)
        self.assertLess(float(slow @ game), -0.05)
        self.assertGreater(float(fast @ game), 0.05)
        for w in (slow, fast):
            self.assertAlmostEqual(float(w.sum()), 1.0)
            self.assertGreaterEqual(1 / (np.square(w).sum() * len(w)), v10.LEARN_MIN_ESS - 1e-6)

    def test_nothing_is_learnt_before_a_snap_or_without_form(self):
        self.assertIsNone(v10.learn_weights(self.tables, self.grid, (0, 0), (0, 0), 1, 240.0,
                                            (0, 0), 5, 100))
        flat = sim10.Tables.build(self.matches, min_records=20)
        self.assertIsNone(v10.learn_weights(flat, self.grid, (0, 0), (0, 0), 3, 240.0, (0, 0), 5, 100))
        self.assertEqual(v10.elapsed_share(self.tables, 5, 100.0), 1.0)
        self.assertAlmostEqual(v10.elapsed_share(self.tables, 3, 240.0),
                               float(self.tables._learn_quarters[:2].sum()
                                     / self.tables._learn_quarters.sum()))

    def _price(self, home, away, learn):
        state = v10.GameState(period=3, elapsed_in_period=0.0, home_score=home, away_score=away,
                              offense=v10.HOME, down=1, field_position=25, distance=10,
                              clock_seconds=240.0)
        prof = (players.Profile(), players.Profile())
        variant = v10.Variant("v10", learn=learn)
        (mp_, tp), = v10.price_states(self.tables, (0.0, 0.0), variant, [], True, [state], [1], prof,
                                      1500, np.random.default_rng(3), seed=21, grid=self.grid)
        return float(tp @ np.arange(len(tp))) - home - away

    def test_the_points_to_come_follow_the_first_half(self):
        e = v10.expected_points(self.grid, (0.0, 0.0)) * v10.elapsed_share(self.tables, 3, 240.0)
        slow, fast = (0, 0), tuple(int(x) for x in np.round(e * 2))
        self.assertLess(self._price(*slow, learn=True), self._price(*slow, learn=False))
        self.assertGreater(self._price(*fast, learn=True), self._price(*fast, learn=False))

    def test_with_learning_off_v10_prices_as_v9(self):
        state = v10.GameState(period=2, elapsed_in_period=140.0, home_score=10, away_score=3,
                              offense=v10.HOME, down=2, field_position=40, distance=6,
                              clock_seconds=100.0)
        prof = (players.Profile(), players.Profile())
        t9 = copy.deepcopy(self.tables)
        a = v10.price_states(self.tables, (0.0, 0.0), v10.Variant("v10", learn=False), [], True,
                             [state], [1], prof, 400, np.random.default_rng(3), seed=4, grid=self.grid)
        b = v9.price_states(t9, (0.0, 0.0), v9.Variant("v9"), [], True, [state], [1], prof, 400,
                            np.random.default_rng(3), seed=4)
        self.assertTrue(np.array_equal(a[0][0], b[0][0]) and np.array_equal(a[0][1], b[0][1]))

    def test_a_weight_of_nothing_prices_as_v9_and_the_weight_rides_on_the_variant(self):
        from unittest import mock
        self.assertIsNone(v10.learn_weights(self.tables, self.grid, (0, 0), (0, 0), 3, 240.0, (0, 0),
                                            5, 100, weight=0.0))
        with mock.patch.dict(os.environ, {"EAMF_V10_LEARN_WEIGHT": "2.5"}):
            self.assertEqual(v10.Variant("v10").learn_weight, 2.5)
        self.assertEqual(v10.Variant("v10", learn_weight=0.5).learn_weight, 0.5)

    def test_a_v9_build_stands_in_for_v10s_own(self):
        with tempfile.TemporaryDirectory() as d:
            for f in ("v9tables.npz", "v9grid.npz", "v9players.json"):
                open(os.path.join(d, f), "w").close()
            tables, grid = v10_stream.model_paths(d)
            self.assertEqual((os.path.basename(tables), os.path.basename(grid)),
                             ("v9tables.npz", "v9grid.npz"))
            self.assertEqual(os.path.basename(v10.build_file(d, "v10players.json")), "v9players.json")
            open(os.path.join(d, "v10players.json"), "w").close()
            self.assertEqual(os.path.basename(v10.build_file(d, "v10players.json")), "v10players.json")


if __name__ == "__main__":
    unittest.main()
