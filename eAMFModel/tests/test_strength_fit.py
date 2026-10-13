"""v14's strength refit: the lead read with TEAM_A on either side, the build's own boom and tilt,
the strength fit on real snaps by expected points, and the day's spread fitted through the sim."""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from .. import players, sim14, v14
from .fakes import _matches, _with_handles


class TestTeamASide(unittest.TestCase):

    def row(self, side):
        return {"score_p1": "14", "score_p2": "7", "team_a_side": side}

    def test_the_lead_follows_team_a_side(self):
        self.assertEqual(sim14._margin(self.row("home"), "TEAM_A"), 7)
        self.assertEqual(sim14._margin(self.row("home"), "TEAM_B"), -7)
        self.assertEqual(sim14._margin(self.row("away"), "TEAM_A"), -7)     # TEAM_A is the away side, down 7
        self.assertEqual(sim14._margin(self.row("away"), "TEAM_B"), 7)
        self.assertEqual(sim14._margin(self.row(None), "TEAM_A"), 7)        # not known: home, as before

    def test_the_same_game_with_team_a_away_reads_the_same(self):
        rows = _matches(n=1)["AF000"]
        # the same game written the other way: TEAM_A the away side, the board home's first
        other = [dict(r, team_a_side="away", score_p1=r["score_p2"], score_p2=r["score_p1"],
                      final_p1=r["final_p2"], final_p2=r["final_p1"]) for r in rows]
        self.assertEqual([r["margin"] for r in sim14.snap_records(rows)],
                         [r["margin"] for r in sim14.snap_records(other)])
        np.testing.assert_array_equal(sim14.real_snap_counts(rows)[::-1], sim14.real_snap_counts(other))


class TestBuildsOwnStrength(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.matches = _with_handles(_matches(n=40))
        cls.tables = sim14.Tables.build(cls.matches, min_records=20)

    def test_kept_with_the_tables(self):
        t = self.tables
        self.assertEqual(sim14.strength_mech(t), (sim14.BOOM, sim14.TILT_SCALE))
        with tempfile.TemporaryDirectory() as d:
            t.boom, t.tilt_scale = 0.75, 1.0
            t.theta_grid, t.form_grid, t.spread_scale = np.linspace(-1, 1, 5), np.linspace(-0.5, 0.5, 3), (0.25, 2.0)
            path = os.path.join(d, "t.npz")
            t.save(path)
            got = sim14.Tables.load(path)
        t.boom = t.tilt_scale = t.theta_grid = t.form_grid = t.spread_scale = None
        self.assertEqual(sim14.strength_mech(got), (0.75, 1.0))
        np.testing.assert_allclose(got.theta_grid, np.linspace(-1, 1, 5))
        np.testing.assert_allclose(got.form_grid, np.linspace(-0.5, 0.5, 3))
        self.assertEqual(got.spread_scale, (0.25, 2.0))
        np.testing.assert_allclose(v14.theta_grid_of(got), np.linspace(-1, 1, 5))
        np.testing.assert_allclose(v14.theta_grid_of(self.tables), v14.GRID)

    def test_the_sim_plays_the_tables_tilt(self):
        start = sim14.Start(1)
        start.theta[:] = (0.6, 0.0)
        points = []
        for tilt in (0.0, 1.0):
            self.tables.tilt_scale = tilt
            home, _ = sim14.simulate(self.tables, start, 300, np.random.default_rng(1), seed=1)
            points.append(home.mean())
        self.tables.tilt_scale = None
        self.assertGreater(points[1], points[0] + 1.0)          # strength counts only with the tilt on

    def test_the_grid_spans_the_tables_strengths(self):
        self.tables.theta_grid = np.array([-0.5, 0.0, 0.5])
        try:
            grid = v14.PriorGrid.build(self.tables, n_paths=20)
        finally:
            self.tables.theta_grid = None
        np.testing.assert_allclose(grid.grid, [-0.5, 0.0, 0.5])
        self.assertEqual(grid.margin.shape[:2], (3, 3))


class TestStrengthFit(unittest.TestCase):

    def test_theta_at_reads_the_curve_and_runs_on_past_its_ends(self):
        curve = (np.array([-1.0, 0.0, 1.0]), np.exp(np.array([2.0, 2.5, 3.0])))
        self.assertAlmostEqual(v14.theta_at(curve, np.exp(2.75)), 0.5, places=4)
        self.assertAlmostEqual(v14.theta_at(curve, np.exp(3.5)), 2.0, places=3)
        self.assertAlmostEqual(v14.theta_at(curve, np.exp(1.5)), -2.0, places=3)

    def test_deviance(self):
        real = np.array([[10.0, 30.0, 60.0], [20.0, 20.0, 60.0]])
        self.assertAlmostEqual(v14.snap_deviance(real, real * 7), 0.0)
        self.assertGreater(v14.snap_deviance(real, real[::-1]), 1.0)

    def test_the_fit_sets_the_build_own_strength(self):
        m = _with_handles(_matches(n=50))
        tables = sim14.Tables.build(m, min_records=20)
        priors = {c: (14.0 + (i % 5), 16.0 - (i % 3)) for i, c in enumerate(m)}
        handles = {c: v14.handles_of(rows) for c, rows in m.items()}
        prof = v14.match_profiles(m, players.Book(), None, handles, {})
        with mock.patch.multiple(v14, STRENGTH_BOOMS=(0.0, 3.0), STRENGTH_TILTS=(0.5, 1.0), STRENGTH_MIN_MATCHES=10,
                                 STRENGTH_PATHS=8, STRENGTH_GRID_PATHS=30, STRENGTH_SWEEP_PATHS=40, BUILD_WORKERS=1), \
                mock.patch("builtins.print"):
            tried = v14.fit_strength(tables, m, priors, prof)
            self.assertIsNone(v14.fit_strength(sim14.Tables.build(m, min_records=20), dict(list(m.items())[:5]),
                                               priors, prof))
        best = min(tried.values(), key=lambda r: r["deviance"])
        self.assertEqual((tables.boom, tables.tilt_scale), (best["boom"], best["tilt"]))
        self.assertGreaterEqual(len(tried), 4)                   # the four, and the steps around the best
        self.assertEqual(len(tables.theta_grid), v14.GRID_N)
        self.assertTrue(np.all(np.diff(tables.theta_grid) > 0))
        self.assertEqual(len(tables.form_grid), 7)
        r = tried[(0.0, 0.5)]
        self.assertEqual(r["real"].shape, (len(v14.STRENGTH_BANDS) + 1, sim14.N_CLASSES))
        self.assertEqual(r["real"].sum(), sum(sim14.real_snap_counts(rows).sum() for rows in m.values()))


class TestSpreadFit(unittest.TestCase):

    def setUp(self):
        self.m = _with_handles(_matches(n=30))
        self.tables = sim14.Tables.build(self.m, min_records=20)
        self.tables.strength_game, self.tables.strength_league = 0.02, 0.005
        self.tables.strength_theta = np.array([-1.0, 0.0, 1.0])
        self.tables.strength_slope = np.array([1.0, 1.0, 1.0])
        self.handles = {c: v14.handles_of(rows) for c, rows in self.m.items()}
        self.held = {c: v14.Held((0.0, 0.0), (players.Profile(), players.Profile())) for c in self.m}

    def test_the_fit_scales_the_swing_and_own_form(self):
        form = {"ALPHA": 0.01, "BRAVO": 0.0}
        with mock.patch.multiple(v14, SPREAD_MATCHES=30, SPREAD_PATHS=10, STRENGTH_MIN_MATCHES=10,
                                 BUILD_WORKERS=1, SPREAD_GAME=(0.0, 1.0), SPREAD_OWN=(1.0, 2.0)), \
                mock.patch("builtins.print"):
            scores = v14.fit_spread(self.tables, self.m, self.held, form, self.handles,
                                    {c: (16.0, 15.0) for c in self.m})
        g, o = self.tables.spread_scale
        self.assertIn((g, 1.0), scores)
        self.assertIn((g, o), scores)
        self.assertAlmostEqual(self.tables.strength_game, 0.02 * g)
        self.assertAlmostEqual(self.tables.strength_league, 0.005 * o)
        self.assertAlmostEqual(form["ALPHA"], 0.01 * o)
        rps_m, rps_t, *cover = scores[(g, o)]
        self.assertTrue(rps_m > 0 and rps_t > 0 and all(0 <= c <= 1 for c in cover))

    def test_later_form_fits_keep_the_scale(self):
        self.tables.spread_scale = (0.5, 2.0)
        sides = [("P", 1.0, 17.0, 16.0, f"M{i // 2}") for i in range(400)]
        with mock.patch.object(v14, "fixed_strength_spread", return_value=(None, None, np.ones(3))), \
                mock.patch.object(v14, "fit_form", return_value=(0.02, 0.004, {"P": 0.01})):
            form = v14._fit_form(self.tables, sides)
        self.assertAlmostEqual(self.tables.strength_game, 0.01)
        self.assertAlmostEqual(self.tables.strength_league, 0.008)
        self.assertEqual(form, {"P": 0.02})


if __name__ == "__main__":
    unittest.main()
