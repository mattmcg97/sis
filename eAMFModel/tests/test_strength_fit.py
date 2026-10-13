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

    def test_no_gain_plays_sit_below_the_short_gains(self):
        t = self.tables
        self.assertIsNotNone(t.n_zero_stop)
        for key in range(len(t.count)):
            for lo, n, to, loss, zero, fail in (
                    (t.start[key], t.n_stop[key], t.n_to_stop[key], t.n_loss_stop[key], t.n_zero_stop[key],
                     t.n_fail_stop[key]),
                    (t.start[key] + t.n_stop[key], t.count[key] - t.n_stop[key], t.n_to_run[key],
                     t.n_loss_run[key], t.n_zero_run[key], t.n_fail_run[key])):
                if not n:
                    continue
                g = t.gain[lo:lo + n]
                k = t.kind[lo:lo + n]
                zeros = slice(to + loss, to + loss + zero)
                self.assertTrue(np.all((k[zeros] == sim14.GAIN) & (g[zeros] == 0)))
                shorts = slice(to + loss + zero, fail)
                self.assertTrue(np.all(g[shorts] != 0) or not len(g[shorts]))

    def test_kept_with_the_tables(self):
        t = self.tables
        self.assertEqual(sim14.strength_mech(t), (sim14.BOOM, sim14.TILT_SCALE))
        self.assertEqual(sim14.strength_convert(t), sim14.CONVERT)
        with tempfile.TemporaryDirectory() as d:
            t.boom, t.tilt_scale, t.convert = 0.75, 1.0, 1.5
            t.theta_grid, t.form_grid, t.spread_scale = np.linspace(-1, 1, 5), np.linspace(-0.5, 0.5, 3), (0.25, 2.0)
            path = os.path.join(d, "t.npz")
            t.save(path)
            got = sim14.Tables.load(path)
        t.boom = t.tilt_scale = t.convert = t.theta_grid = t.form_grid = t.spread_scale = None
        self.assertEqual(sim14.strength_mech(got), (0.75, 1.0))
        self.assertEqual(sim14.strength_convert(got), 1.5)
        np.testing.assert_array_equal(got.n_zero_stop, t.n_zero_stop)
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


class TestConversion(unittest.TestCase):

    def parts(self, conv, n=40000):
        rng = np.random.default_rng(0)
        u, one = rng.random(n), np.ones(n)
        to, loss, zero, fail, top = (np.full(n, x) for x in (0.03, 0.08, 0.15, 0.55, 0.04))
        w = sim14.zone_warp(u, one, to, loss, fail, top, one, one, one, zero, None if conv is None else one * conv)
        edges = [0.03, 0.11, 0.26, 0.55, 0.96]
        return np.bincount(np.searchsorted(edges, w, side="right"), minlength=6) / n

    def test_only_short_gains_and_first_downs_trade(self):
        base, more = self.parts(None), self.parts(2.0)
        np.testing.assert_allclose(base, [0.03, 0.08, 0.15, 0.29, 0.41, 0.04], atol=0.01)
        np.testing.assert_allclose(more[[0, 1, 2, 5]], base[[0, 1, 2, 5]], atol=0.004)
        # odds of a first down against a short gain doubled, the two together kept
        self.assertAlmostEqual(more[4] / more[3], 2 * base[4] / base[3], delta=0.06)
        self.assertAlmostEqual(more[3] + more[4], base[3] + base[4], delta=0.004)

    def test_strength_converts_in_the_sim(self):
        m = _with_handles(_matches(n=40))
        tables = sim14.Tables.build(m, min_records=20)
        start = sim14.Start(1)
        start.theta[:] = (0.6, 0.0)
        tables.boom, tables.tilt_scale = 0.0, 0.0                # strength through the conversion alone
        got = []
        for c in (0.0, 2.0):
            tables.convert = c
            stats = {"_snaps": []}
            home, _ = sim14.simulate(tables, start, 300, np.random.default_rng(1), seed=1, stats=stats)
            counts = sim14.sim_snap_counts(stats["_snaps"])[0]
            got.append((home.mean(), counts / counts.sum()))
        self.assertGreater(got[1][0], got[0][0] + 1.0)
        self.assertLess(got[1][1][3], got[0][1][3])                    # fewer short gains
        self.assertAlmostEqual(got[1][1][2], got[0][1][2], delta=0.02)  # as many no gains


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
        axes = (("boom", (-0.5, 0.0)), ("tilt", (0.5, 1.0)), ("convert", (0.0, 1.0)))
        with mock.patch.multiple(v14, STRENGTH_AXES=axes, STRENGTH_START=(0.0, 0.5, 0.0), STRENGTH_CYCLES=1,
                                 STRENGTH_MIN_MATCHES=10, STRENGTH_PATHS=8, STRENGTH_GRID_PATHS=30,
                                 STRENGTH_SWEEP_PATHS=40, BUILD_WORKERS=1), \
                mock.patch("builtins.print"):
            tried = v14.fit_strength(tables, m, priors, prof)
            self.assertIsNone(v14.fit_strength(sim14.Tables.build(m, min_records=20), dict(list(m.items())[:5]),
                                               priors, prof))
        best = min(tried.values(), key=lambda r: r["deviance"])
        self.assertEqual((tables.boom, tables.tilt_scale, tables.convert), (best["boom"], best["tilt"], best["convert"]))
        self.assertIn((0.0, 0.5, 0.0), tried)                    # the start
        self.assertIn((sim14.BOOM, sim14.TILT_SCALE, sim14.CONVERT), tried)   # the old setting, for the print
        self.assertTrue(any(k[0] == -0.5 for k in tried))        # the boom may go below 0
        self.assertEqual(len(tables.theta_grid), v14.GRID_N)
        self.assertTrue(np.all(np.diff(tables.theta_grid) > 0))
        self.assertEqual(len(tables.form_grid), 7)
        r = tried[(0.0, 0.5, 0.0)]
        n, prior, got = r["points"].sum(0)
        self.assertEqual(n, 2 * len(m))
        self.assertLess(abs(got - prior) / n, abs(prior / n) * 0.25)      # held near the priors
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


class TestTheBuildFitsIt(unittest.TestCase):
    """The strength fit runs in the joint fit's first round, after the grid's responses."""

    def test_a_build_with_priors_and_handles_fits_its_strength(self):
        from .test_prior_variants import FakePrior
        import datetime as dt
        m = _with_handles(_matches(n=24))
        history = [{"MATCH_CODE": c, "SPORT_CODE": "AF", "STREAM_NUMBER": "1",
                    "SCHEDULED_START_TIME_UTC": "2026-09-0%d 12:00:00" % (1 + i % 9),
                    "PLAYER_1_HANDLE": rows[0]["home_handle"], "PLAYER_1_TEAM": "Lions",
                    "PLAYER_2_HANDLE": rows[0]["away_handle"], "PLAYER_2_TEAM": "Bears",
                    "PLAYER_1_FINAL_SCORE": rows[0]["final_p1"], "PLAYER_2_FINAL_SCORE": rows[0]["final_p2"]}
                   for i, (c, rows) in enumerate(m.items())]
        axes = (("boom", (0.0,)), ("tilt", (0.5,)), ("convert", (0.0, 1.0)))
        calls = []
        real_fit = v14.fit_strength

        def spy(tables, *a, **kw):
            calls.append(tables.boom)
            return real_fit(tables, *a, **kw)

        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(v14.PRIORS, {"glmer": FakePrior}), \
                mock.patch.multiple(v14, PRIOR_SHRINK=False, STRENGTH_AXES=axes, STRENGTH_START=(0.0, 0.5, 0.0),
                                    STRENGTH_CYCLES=1, STRENGTH_MIN_MATCHES=5, STRENGTH_PATHS=4,
                                    STRENGTH_GRID_PATHS=20, STRENGTH_SWEEP_PATHS=20, JOINT_ROUNDS=1,
                                    fit_strength=spy):
            v14.build(m, d, grid_paths=40, verbose=False, history=history, before=dt.datetime(2026, 9, 20),
                      prior="glmer")
            t = sim14.Tables.load(os.path.join(d, "v14tables.npz"))
            grid = v14.PriorGrid.load(os.path.join(d, "v14grid.npz"))
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(t.boom)
        self.assertIn(t.convert, (0.0, 1.0))
        np.testing.assert_allclose(grid.grid, t.theta_grid)       # the final grid on the fitted strength


if __name__ == "__main__":
    unittest.main()
