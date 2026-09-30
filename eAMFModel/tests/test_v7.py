"""v7: v4 (copied as it was) plus play calling -- clock-stopping or
clock-running plays called by game state, the clock each uses by state,
and the rubber band (sim7.py, v7.py, v7_stream.py)."""

import csv
import datetime as dt
import inspect
import json
import os
import random
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np

from .. import glmer_prior, nb2_prior, players, pricer, sim, sim4, sim5, sim6, sim7, v7, v7_stream
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
        cls.tables = sim7.Tables.build(_matches(), min_records=20)

    def _two_of(self, common):
        st = sim7.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim7.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 14, 10
        return sim7.simulate(self.tables, st, 300, np.random.default_rng(1), common=common)

    def test_the_same_state_twice_plays_the_same_games(self):
        h, a = self._two_of(True)
        self.assertTrue((h[0] == h[1]).all() and (a[0] == a[1]).all())
        h, a = self._two_of(False)
        self.assertFalse((h[0] == h[1]).all() and (a[0] == a[1]).all())

    def test_the_stream_is_uniform_and_repeatable(self):
        path, step = np.arange(20000), np.full(20000, 3)
        u = sim7._uniform(12345, path, step, 7)
        self.assertTrue(((u >= 0) & (u < 1)).all())
        self.assertAlmostEqual(float(u.mean()), 0.5, delta=0.01)
        self.assertTrue((u == sim7._uniform(12345, path, step, 7)).all())
        self.assertFalse((u == sim7._uniform(12345, path, step, 8)).all())

    def test_v3_is_left_as_it_was(self):
        self.assertNotIn("common", inspect.signature(sim.simulate).parameters)


class TestPlayCalling(unittest.TestCase):
    """The play call (a clock-stopping play or one that keeps the clock
    running), the clock each uses and the rubber band, by game state."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(60)
        cls.tables = sim7.Tables.build(cls.matches, min_records=20)

    def test_the_cells_agree(self):
        for period, clock, lead in ((1, 240, 0), (2, 119.5, -3), (3, 1, 21), (4, 80, 9), (5, 200, -9)):
            self.assertEqual(sim7.cell_index(period, clock, lead),
                             int(sim7._cells_np(np.array([period]), np.array([float(clock)]),
                                                np.array([lead]))[0]))
        self.assertEqual(sim7.cell_index(4, 0.5, 30), sim7.N_CELLS - 1)

    def test_each_bin_lays_out_its_stopping_plays_first(self):
        t = self.tables
        self.assertTrue(((0 <= t.n_stop) & (t.n_stop <= t.count)).all())
        for key in range(0, sim7.N_KEYS, 13):
            a, n, m = t.start[key], t.count[key], t.n_stop[key]
            secs = t.seconds[a:a + n]
            self.assertTrue((secs[:m] <= sim7.STOP_SECONDS).all())
            self.assertTrue((secs[m:] > sim7.STOP_SECONDS).all())

    def test_the_fit_finds_a_planted_state_effect(self):
        import copy
        snaps = [r for rows in self.matches.values() for r in sim7.snap_records(rows)]
        target = sim7.cell_index(3, 150, 5)
        planted = []
        for r in snaps:
            r = dict(r)
            if sim7.cell_index(r["period"], r["clock"], r["margin"]) == target \
                    and r["seconds"] > sim7.STOP_SECONDS:
                r["seconds"] += 8                  # leaders here take 8 s longer
            planted.append(r)
        t = copy.deepcopy(self.tables)
        sim7.fit_play_calling(t, planted)
        sim7.fit_play_calling(self.tables, snaps)
        gained = t.sec_shift[sim7.RUNNING, target] - self.tables.sec_shift[sim7.RUNNING, target]
        n = sum(1 for r in snaps if sim7.cell_index(r["period"], r["clock"], r["margin"]) == target
                and r["seconds"] > sim7.STOP_SECONDS and r["clock"] >= sim7.UNCUT_SECONDS
                and not r["fresh"])
        self.assertGreater(n, 10)
        # the 8 s, shrunk toward 0 by SECONDS_PRIOR snaps (a fresh possession's snap is left out)
        self.assertAlmostEqual(gained, 8 * n / (n + sim7.SECONDS_PRIOR), places=6)
        self.assertLess(abs(t.sec_shift[sim7.STOP, target] - self.tables.sec_shift[sim7.STOP, target]), 1e-9)

    def test_the_band_pulls_in_every_quarter(self):
        self.assertTrue(v7.BAND_BY_QUARTER)
        shift = v7._band_shift(np.array([0.1, 0.2, 0.3]))
        quarter = np.arange(sim7.N_CELLS) // (sim7.CLOCK_CELLS * sim7.LEAD_CELLS)
        ahead = np.arange(sim7.N_CELLS) % sim7.LEAD_CELLS == 4          # ahead by 9+
        behind = np.arange(sim7.N_CELLS) % sim7.LEAD_CELLS == 0
        for q, pull in enumerate((0.1, 0.1, 0.2, 0.3)):             # the fourth included
            self.assertTrue(np.allclose(shift[(quarter == q) & ahead], -pull * 14 / v7.BAND_LEAD))
            self.assertTrue(np.allclose(shift[(quarter == q) & behind], pull * 14 / v7.BAND_LEAD))

    def test_the_band_is_fitted_by_quarter(self):
        import copy
        tables = copy.deepcopy(self.tables)
        grid = v7.PriorGrid.build(tables, n_paths=60)
        items = v7.in_game_states(self.matches, grid)
        pull, got, real = v7.fit_rubber_band(tables, items, rounds=3, n_paths=60)
        self.assertEqual(len(pull), 3)
        # each pull brings its own states' comeback to the real one
        for g, r in zip(got, real):
            self.assertAlmostEqual(g, r, delta=0.08)
        self.assertEqual({st.period for st, *_ in items} >= {1, 2, 3, 4}, True)

    def test_distinct_streams_keep_their_luck_between_calls(self):
        st = sim7.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim7.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 14, 10
        a = sim7.simulate(self.tables, st, 200, np.random.default_rng(1), seed=5, distinct=True)
        b = sim7.simulate(self.tables, st, 200, np.random.default_rng(2), seed=5, distinct=True)
        self.assertTrue(np.array_equal(a[0], b[0]))               # the same luck, call to call
        self.assertFalse(np.array_equal(a[0][0], a[0][1]))        # its own for each state

    def test_the_fourth_quarter_gets_the_items_switched_on_for_it(self):
        q4 = np.arange(sim7.N_CELLS) // (sim7.CLOCK_CELLS * sim7.LEAD_CELLS) == 3
        old = sim7.PLAY_CALLING_Q4
        try:
            sim7.PLAY_CALLING_Q4 = {"stop"}
            self.assertTrue(sim7._quarter_mask("stop")[q4].all())
            self.assertFalse(sim7._quarter_mask("band")[q4].any())
            self.assertTrue(sim7._quarter_mask("band")[~q4].all())
            self.assertFalse(sim7._quarter_mask()[q4].any())
            self.assertTrue((v7._band_shift(np.array([0.3, 0.3]))[q4] == 0).all())
            sim7.PLAY_CALLING_Q4 = set()
            t = sim7.Tables.build(self.matches, min_records=20)
            self.assertTrue((t.stop_shift[q4] == 0).all() and (t.sec_shift[:, q4] == 0).all()
                            and (t.eff_shift[q4] == 0).all())
            self.assertTrue(np.abs(t.sec_shift[:, ~q4]).sum() > 0)
        finally:
            sim7.PLAY_CALLING_Q4 = old

    def test_red_zone_cells_split_quarter_lead_and_the_ten(self):
        self.assertEqual(sim7.rz_index(1, -20, 70), 0)
        self.assertEqual(sim7.rz_index(1, -20, 95), 1)
        self.assertEqual(sim7.rz_index(4, 20, 95), sim7.N_RZ - 1)
        self.assertEqual(list(sim7.rz_index(np.array([3, 5]), np.array([0, 0]), np.array([80, 80]))),
                         [(2 * sim7.LEAD_CELLS + 2) * 2, (3 * sim7.LEAD_CELLS + 2) * 2])

    def test_held_touchdowns_turn_into_field_goals_not_fewer_first_downs(self):
        import copy
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 3, 200.0, sim7.SCRIM
        st.team[:], st.y[:], st.dist[:], st.home[:], st.away[:] = 0, 80, 10, 10, 10
        runs = {}
        for hold in (0.0, 0.8):
            t = copy.deepcopy(self.tables)
            t.rz_hold = np.full(sim7.N_RZ, hold)
            stats = {}
            h, _ = sim7.simulate(t, st, 4000, np.random.default_rng(1), seed=5, one_drive=True,
                                 stats=stats)
            runs[hold] = (np.mean(h[0] - 10 >= 6), np.mean(h[0] - 10 == 3), stats)
        self.assertLess(runs[0.8][0], runs[0.0][0] - 0.05)
        self.assertGreater(runs[0.8][1], runs[0.0][1])
        self.assertGreater(runs[0.8][2]["snaps"], runs[0.0][2]["snaps"])

    def test_the_red_zone_fit_round_trips(self):
        old = sim7.RED_ZONE_FIT
        try:
            sim7.RED_ZONE_FIT = True
            t = sim7.Tables.build(self.matches, min_records=20)
        finally:
            sim7.RED_ZONE_FIT = old
        self.assertEqual(t.rz_hold.shape, (sim7.N_RZ,))
        self.assertTrue(((t.rz_hold >= 0) & (t.rz_hold <= sim7.HOLD_GRID[-1])).all())
        with tempfile.TemporaryDirectory() as d:
            t.save(os.path.join(d, "t.npz"))
            back = sim7.Tables.load(os.path.join(d, "t.npz"))
        self.assertTrue(np.array_equal(back.rz_hold, t.rz_hold))

    def _q3_leader_with_ball(self, n=3000, **shift):
        import copy
        t = copy.deepcopy(self.tables)
        for name, (cells, value) in shift.items():
            arr = getattr(t, name)
            if arr.ndim == 2:
                arr[sim7.RUNNING, cells] += value
            else:
                arr[cells] += value
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 3, 200.0, sim7.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 30, 17, 10
        h, a = sim7.simulate(t, st, n, np.random.default_rng(1), seed=5)
        return (h + a).mean(), (h - a).mean()

    def test_a_leader_milking_the_clock_leaves_fewer_points(self):
        lead = [c for c in range(sim7.N_CELLS) if c % sim7.LEAD_CELLS >= 3 and c // (sim7.CLOCK_CELLS * sim7.LEAD_CELLS) >= 2]
        plain, _ = self._q3_leader_with_ball()
        milked, _ = self._q3_leader_with_ball(sec_shift=(lead, 15.0))
        self.assertLess(milked, plain - 0.5)

    def test_the_rubber_band_pulls_a_lead_back(self):
        trail = [c for c in range(sim7.N_CELLS) if c % sim7.LEAD_CELLS <= 1]
        lead = [c for c in range(sim7.N_CELLS) if c % sim7.LEAD_CELLS >= 3]
        _, plain = self._q3_leader_with_ball()
        _, banded = self._q3_leader_with_ball(eff_shift=(trail, 0.4))
        self.assertLess(banded, plain - 0.3)
        _, eased = self._q3_leader_with_ball(eff_shift=(lead, -0.4))
        self.assertLess(eased, plain - 0.3)

    def test_v4s_tables_play_as_v4(self):
        t4 = sim4.Tables.build(self.matches, min_records=20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            t4.save(path)
            t5 = sim7.Tables.load(path)
        # a game that cannot reach overtime, which v7 plays by the real rules
        st = sim4.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim4.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 38, 0
        a = sim4.simulate(t4, st, 400, np.random.default_rng(1), seed=9)
        b = sim7.simulate(t5, st, 400, np.random.default_rng(1), seed=9)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))


class TestBackedUp(unittest.TestCase):
    """Inside the own 10 each snap draws a safety, a defensive touchdown, a
    turnover or a touchdown from its yard line's own rates; after a safety
    the scorer gets two points and the free kick."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim7.Tables.build(_matches(60), min_records=20)

    def _table(self, **at_the_one):
        import copy
        t = copy.deepcopy(self.tables)
        t.backed = np.zeros((sim7.BACKED_UP + 1, len(sim7.BACKED_OUTCOMES)))
        for name, p in at_the_one.items():
            t.backed[1, sim7.BACKED_OUTCOMES.index(name)] = p
        t.backed_return = None
        return t

    @staticmethod
    def _on_the_one(n=1):
        st = sim7.Start(n)
        st.period[:], st.clock[:], st.phase[:] = 2, 200.0, sim7.SCRIM
        st.team[:], st.y[:], st.down[:], st.dist[:] = 0, 1, 1, 10
        return st

    def test_each_yard_line_keeps_its_own_rate_pulled_toward_the_curve(self):
        records = ([(1, sim7.B_SAFETY, None)] * 12 + [(1, None, None)] * 188
                   + [(5, sim7.B_SAFETY, None)] * 2 + [(5, None, None)] * 198
                   + [(3, sim7.B_TURNOVER, -10)] * 30 + [(3, None, None)] * 170)
        hazard, returns = sim7.fit_backed_up(records)
        self.assertAlmostEqual(hazard[1, sim7.B_SAFETY], 0.06, delta=0.012)
        self.assertAlmostEqual(hazard[5, sim7.B_SAFETY], 0.01, delta=0.008)
        # the 2, never seen, sits on the curve between them
        self.assertGreater(hazard[1, sim7.B_SAFETY], hazard[2, sim7.B_SAFETY])
        self.assertGreater(hazard[2, sim7.B_SAFETY], hazard[5, sim7.B_SAFETY])
        self.assertAlmostEqual(hazard[3, sim7.B_TURNOVER], 0.15, delta=0.03)
        self.assertEqual(list(returns), [-10] * 30)

    def test_the_real_safety_rows_are_found(self):
        rows = [dict(play_kind="SCRIMMAGE", down="2", field_position="2", period="2", offense="TEAM_A",
                     play_messages=""),
                dict(play_kind="PUNT", down="1", field_position="20", period="2", offense="TEAM_A",
                     play_messages="SAFETY_TEAM_B"),
                dict(play_kind="SCRIMMAGE", down="1", field_position="40", period="2", offense="TEAM_B",
                     play_messages="")]
        self.assertEqual(sim7.backed_up_snaps(rows), [(2, sim7.B_SAFETY, None)])

    def test_a_snap_on_the_one_comes_to_its_yard_lines_outcomes(self):
        t = self._table(safety=0.2, def_td=0.1, td=0.1)
        n = 20000
        h, a = sim7.simulate(t, self._on_the_one(), n, np.random.default_rng(1), max_steps=1, seed=9)
        self.assertAlmostEqual(float((a == 2).mean()), 0.2, delta=0.015)
        self.assertAlmostEqual(float((a == 6).mean()), 0.1, delta=0.01)
        self.assertAlmostEqual(float((h == 6).mean()), 0.1, delta=0.01)
        # and nothing else scores: the other snaps stay in the field of play
        self.assertAlmostEqual(float(((h == 0) & (a == 0)).mean()), 0.6, delta=0.015)

    def test_after_a_safety_the_scorer_receives_the_free_kick(self):
        t = self._table(safety=1.0)
        t.safety_kick = np.array([99], dtype=np.int32)     # the free kick lands on the kicker's 1
        stats = {}
        h, a = sim7.simulate(t, self._on_the_one(), 2000, np.random.default_rng(1), max_steps=6,
                             seed=9, stats=stats)
        self.assertTrue((a >= 2).all())
        self.assertTrue((h == 0).all())
        # from the 1 the scorer of the safety scores again, often
        self.assertGreater(float((a >= 8).mean()), 0.3)

    def test_old_tables_have_no_backed_up_outcomes(self):
        t = self._table()
        path = os.path.join(tempfile.mkdtemp(), "t.npz")
        t.backed = None
        t.save(path)
        self.assertIsNone(sim7.Tables.load(path).backed)
        t = self._table(safety=0.3)
        t.save(path)
        self.assertAlmostEqual(float(sim7.Tables.load(path).backed[1, sim7.B_SAFETY]), 0.3)


class TestStrengthSpread(unittest.TestCase):
    """Each simulated game's offenses drawn around the prior: a shared game
    draw that moves the total, each player's own form, the same draw for
    path k of every snapshot of a match, and the fit of both off results."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(40)
        cls.tables = sim7.Tables.build(cls.matches, min_records=20)

    def _kickoff(self, n=1, own=0.0, game=0.0):
        st = sim7.Start(n)
        st.team[:] = 0
        st.kicks_second_half[:] = 1
        st.strength[:] = own
        st.strength_game[:] = game
        return st

    def _spread(self, **kw):
        h, a = sim7.simulate(self.tables, self._kickoff(**kw), 4000, np.random.default_rng(1), seed=5)
        return float((h + a)[0].std()), float((h - a)[0].std())

    def test_the_game_draw_widens_the_total_not_the_margin(self):
        total0, margin0 = self._spread()
        total1, margin1 = self._spread(game=0.3)
        self.assertGreater(total1, total0 * 1.08)
        self.assertGreater(total1 / total0 - 1.0, 2 * (margin1 / margin0 - 1.0))

    def test_a_players_own_form_widens_the_margin(self):
        _, margin0 = self._spread()
        _, margin1 = self._spread(own=0.3)
        self.assertGreater(margin1, margin0 * 1.05)

    def test_the_draw_is_the_same_for_every_snapshot(self):
        st = self._kickoff(2, own=0.2, game=0.2)
        h, a = sim7.simulate(self.tables, st, 500, np.random.default_rng(1), seed=5)
        self.assertTrue(np.array_equal(h[0], h[1]) and np.array_equal(a[0], a[1]))
        h2, _ = sim7.simulate(self.tables, st, 500, np.random.default_rng(9), seed=5)
        self.assertTrue(np.array_equal(h, h2))

    def test_the_draw_keeps_the_expected_points(self):
        import copy
        tables = copy.deepcopy(self.tables)
        _, _, slope = v7.fixed_strength_spread(tables, n_paths=600)
        tables.strength_theta, tables.strength_slope = v7.FORM_GRID.copy(), slope
        tables.strength_game = 0.04
        theta, own, game = sim7.strength_draw(tables, [0.0, 0.0], [0.03, 0.0])
        self.assertTrue(own[0] > 0 and own[1] == 0 and game[0] > 0)
        st = self._kickoff()
        fixed = sim7.simulate(tables, st, 6000, np.random.default_rng(2), common=False)[0].mean()
        st.theta[0], st.strength[0], st.strength_game[0] = theta, own, game
        drawn = sim7.simulate(tables, st, 6000, np.random.default_rng(2), common=False)[0].mean()
        self.assertAlmostEqual(drawn / fixed, 1.0, delta=0.04)

    def test_the_fit_finds_the_game_and_the_volatile_player(self):
        rng = np.random.default_rng(4)
        sides = []
        for m in range(6000):
            p1, p2 = rng.choice(["STEADY1", "STEADY2", "STEADY3", "WILD"], 2, replace=False)
            g = rng.normal(0, 0.15)
            for k, p in enumerate((p1, p2)):
                own = rng.normal(0, 0.25) if p == "WILD" else 0.0
                sides.append((p, 1.0, 20.0 * np.exp(g + own), 20.0, f"M{m}"))
        game, league, form = v7.fit_form(sides, np.zeros(3), np.zeros(4))
        self.assertAlmostEqual(np.sqrt(game), 0.15, delta=0.03)
        self.assertGreater(form["WILD"], 0.03)
        self.assertLess(max(form["STEADY1"], form["STEADY2"], form["STEADY3"]), 0.01)

    def test_a_players_form_rides_in_the_book(self):
        import tempfile
        book = players.Book()
        book.players["WILD"] = players.Profile(form=0.05)
        with tempfile.TemporaryDirectory() as d:
            book.save(os.path.join(d, "p.json"))
            back = players.Book.load(os.path.join(d, "p.json"))
        self.assertEqual(back.profile("wild").form, 0.05)
        self.assertIsNone(back.profile("NOBODY").form)

    def test_old_tables_play_with_the_strengths_fixed(self):
        t = sim7.Tables()
        theta, own, game = sim7.strength_draw(t, [0.1, -0.1], [0.05, 0.05])
        self.assertTrue(np.allclose(theta, [0.1, -0.1]) and not own.any() and not game.any())


class TestLateGame(unittest.TestCase):
    """v7: a trailing side's late 4th downs from a fitted table, and big leads as situations of
    their own."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(60)
        cls.tables = sim7.Tables.build(cls.matches, min_records=20)

    def _fourth(self, behind):
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 4, 60.0, sim7.SCRIM
        st.team[:], st.down[:], st.dist[:], st.y[:] = 0, 4, 8, 80
        st.home[:], st.away[:] = 10, 10 + behind
        return st

    def test_the_table_sets_what_a_trailing_side_does(self):
        import copy
        t = copy.deepcopy(self.tables)
        t.late_fourth = np.zeros_like(t.late_fourth)
        t.late_fourth[..., 1] = 1.0
        h, _ = sim7.simulate(t, self._fourth(10), 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertGreater(float((h == 13).mean()), 0.8)
        t.late_fourth[..., 1], t.late_fourth[..., 0] = 0.0, 1.0
        h, _ = sim7.simulate(t, self._fourth(10), 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertEqual(float((h == 13).mean()), 0.0)

    def test_the_fitted_table_is_shaped_and_sums_to_one(self):
        lf = self.tables.late_fourth
        self.assertEqual(lf.shape, (2, len(sim7.LATE_DEFICITS) + 1, len(sim7.KICK_RANGES) + 1, 3))
        self.assertTrue(np.allclose(lf.sum(3), 1.0))

    def test_with_its_own_switches_off_v7_plays_as_v6(self):
        saved = (sim7.FRESH_CLOCK, sim7.Q4_MODES, sim7.SETTLE_FIT, sim7.GO_AHEAD, sim7.PLAY_CALLING_Q4)
        sim7.FRESH_CLOCK = sim7.Q4_MODES = sim7.SETTLE_FIT = sim7.GO_AHEAD = False
        sim7.PLAY_CALLING_Q4 = set()
        try:
            t7 = sim7.Tables.build(self.matches, min_records=20)
            t6 = sim6.Tables.build(self.matches, min_records=20)
            st = sim7.Start(4)
            st.period[:], st.clock[:], st.phase[:] = [3, 4, 4, 4], [150.0, 150.0, 50.0, 25.0], sim7.SCRIM
            st.team[:], st.y[:], st.home[:], st.away[:] = 0, [50, 50, 80, 75], [3, 17, 10, 10], [10, 10, 16, 13]
            st.down[:] = [1, 2, 4, 4]
            st.fresh[:] = [True, False, True, False]
            a = sim7.simulate(t7, st, 300, np.random.default_rng(1), seed=9)
            b = sim6.simulate(t6, st, 300, np.random.default_rng(1), seed=9)
        finally:
            sim7.FRESH_CLOCK, sim7.Q4_MODES, sim7.SETTLE_FIT, sim7.GO_AHEAD, sim7.PLAY_CALLING_Q4 = saved
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_with_both_off_v6_plays_as_v5(self):
        import copy
        saved = (sim7.BIG_LEAD, sim7.FOURTH_JOINT, sim7.OT_RULES, sim7.PLAY_CALLING_Q4,
                 sim7.RED_ZONE_FIT, sim7.FRESH_CLOCK, sim7.Q4_MODES, sim7.SETTLE_FIT, sim7.GO_AHEAD)
        sim7.BIG_LEAD, sim7.FOURTH_JOINT, sim7.OT_RULES, sim7.PLAY_CALLING_Q4, sim7.RED_ZONE_FIT = \
            None, False, False, set(), False
        sim7.FRESH_CLOCK = sim7.Q4_MODES = sim7.SETTLE_FIT = sim7.GO_AHEAD = False
        try:
            t6 = sim7.Tables.build(self.matches, min_records=20)
            t6.late_fourth = None
            t5 = sim5.Tables.build(self.matches, min_records=20)
            st = sim7.Start(3)
            st.period[:], st.clock[:], st.phase[:] = 4, 150.0, sim7.SCRIM
            st.team[:], st.y[:], st.home[:], st.away[:] = 0, 50, [3, 17, 10], [10, 10, 30]
            a = sim7.simulate(t6, st, 300, np.random.default_rng(1), seed=9)
            b = sim5.simulate(t5, st, 300, np.random.default_rng(1), seed=9)
        finally:
            (sim7.BIG_LEAD, sim7.FOURTH_JOINT, sim7.OT_RULES, sim7.PLAY_CALLING_Q4,
             sim7.RED_ZONE_FIT, sim7.FRESH_CLOCK, sim7.Q4_MODES, sim7.SETTLE_FIT, sim7.GO_AHEAD) = saved
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_big_leads_have_their_own_situations(self):
        self.assertEqual(sim7.mode_of(3, 100.0, 5), 3)
        self.assertEqual(sim7.mode_of(3, 100.0, 12), 8)
        self.assertEqual(sim7.mode_of(4, 60.0, 12), 9)
        self.assertEqual(sim7.mode_of(4, 60.0, 12, None), 4)
        modes = sim7._modes_np(np.array([3, 4, 4]), np.array([100.0, 60.0, 60.0]), np.array([12, 12, 3]), 9)
        self.assertEqual(list(modes), [8, 9, 4])
        self.assertEqual(self.tables.big_lead, sim7.BIG_LEAD)


class TestV7(unittest.TestCase):
    """v7: a fresh possession's first snap is played with the clock stopped, the fourth quarter
    has situations of its own, a side a touchdown behind late goes for it, and would-be
    touchdowns are held back where the simulation scores too often."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(60)
        cls.tables = sim7.Tables.build(cls.matches, min_records=20)

    def _snap(self, n=1, fresh=False, period=3, clock=150.0, y=40, down=1, dist=10, home=10, away=10):
        st = sim7.Start(n)
        st.period[:], st.clock[:], st.phase[:] = period, clock, sim7.SCRIM
        st.team[:], st.y[:], st.down[:], st.dist[:] = 0, y, down, dist
        st.home[:], st.away[:], st.fresh[:] = home, away, fresh
        return st

    def test_a_fresh_possession_is_marked_on_its_row(self):
        row = lambda kind, messages="": dict(play_kind=kind, play_messages=messages)
        for kind in ("KICKOFF", "PUNT", "TURNOVER_ON_DOWNS", "FIELD_GOAL"):
            self.assertTrue(sim7.is_fresh(row(kind)))
        self.assertTrue(sim7.is_fresh(row("SCRIMMAGE", "PASS_TEAM_A|POSSESSION_TEAM_B")))
        self.assertFalse(sim7.is_fresh(row("SCRIMMAGE", "PASS_TEAM_A")))
        rows = next(iter(self.matches.values()))
        recs = sim7.snap_records(rows)
        kicks = {int(a["message"]) for a in rows if a["play_kind"] == "KICKOFF"}
        by_msg = {int(b["message"]): int(a["message"]) for a, b in zip(rows, rows[1:])}
        self.assertTrue(any(r["fresh"] for r in recs))
        self.assertTrue(all(r["fresh"] for r in recs if by_msg[r["message"]] in kicks))

    def test_a_fresh_snap_draws_a_clock_stopped_play(self):
        import copy
        t = copy.deepcopy(self.tables)
        t.stop_shift[:] = 0.0
        t.n_fresh, t.n_fresh_stop = t.n_stop.copy(), t.n_stop.copy()
        for fresh, want in ((True, 0.9), (False, 0.0)):
            stats = {}
            sim7.simulate(t, self._snap(200, fresh), 5, np.random.default_rng(1), max_steps=1, stats=stats)
            share = stats.get("stop_calls", 0) / stats["snaps"]
            if fresh:
                self.assertGreater(share, want)
            else:
                self.assertEqual(share, want)

    def test_kicks_turnovers_and_quarter_starts_leave_the_next_snap_fresh(self):
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 3, 200.0, sim7.KICK
        stats = {}
        sim7.simulate(self.tables, st, 300, np.random.default_rng(1), max_steps=2, stats=stats)
        self.assertEqual(stats["fresh_snaps"], stats["snaps"])
        stats = {}
        sim7.simulate(self.tables, self._snap(1), 300, np.random.default_rng(1), max_steps=1, stats=stats)
        self.assertEqual(stats.get("fresh_snaps", 0), 0)
        stats = {}
        sim7.simulate(self.tables, self._snap(1, clock=0.5), 300, np.random.default_rng(1), max_steps=3,
                      stats=stats)
        self.assertGreater(stats["fresh_snaps"], 0)

    def test_the_clock_is_fitted_on_snaps_that_follow_a_scrimmage_play(self):
        import copy
        snaps = [r for rows in self.matches.values() for r in sim7.snap_records(rows)]
        planted = [dict(r, seconds=59.0) if r["fresh"] else r for r in snaps]
        a, b = copy.deepcopy(self.tables), copy.deepcopy(self.tables)
        sim7.fit_play_calling(a, snaps)
        sim7.fit_play_calling(b, planted)
        self.assertTrue(np.allclose(a.sec_shift, b.sec_shift))
        self.assertTrue(np.allclose(a.stop_shift, b.stop_shift))

    def test_the_fourth_quarter_has_situations_of_its_own(self):
        self.assertEqual([sim7.mode_of(4, 200.0, m, 9, True) for m in (3, 12, 0, -3, -12)],
                         [10, 11, 12, 13, 14])
        self.assertEqual([sim7.mode_of(4, 200.0, m, 9) for m in (3, 12, 0, -3, -12)], [3, 8, 5, 5, 5])
        self.assertEqual(sim7.mode_of(4, 100.0, 3, 9, True), 4)       # its last two minutes as before
        self.assertEqual(sim7.mode_of(4, 100.0, -3, 9, True), 6)
        self.assertEqual(sim7.mode_of(4, 100.0, -12, 9, True), 15)    # but two scores behind is its own
        self.assertEqual(sim7.mode_of(3, 200.0, 3, 9, True), 3)
        self.assertEqual(sim7.mode_of(3, 200.0, -12, 9, True), 5)
        period, clock = np.array([4, 4, 4, 4, 4, 4, 3]), np.array([200.0, 200, 200, 200, 100, 100, 200])
        margin = np.array([3, 12, 0, -12, -3, -12, -12])
        self.assertEqual(list(sim7._modes_np(period, clock, margin, 9, True)), [10, 11, 12, 14, 6, 15, 5])
        self.assertTrue(self.tables.q4_modes)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            self.tables.save(path)
            back = sim7.Tables.load(path)
        self.assertTrue(back.q4_modes)
        self.assertTrue(np.array_equal(back.n_fresh, self.tables.n_fresh))

    def test_a_side_a_touchdown_behind_in_the_last_minute_goes_for_it(self):
        import copy
        t = copy.deepcopy(self.tables)
        t.late_fourth = np.zeros_like(t.late_fourth)
        t.late_fourth[..., 1] = 1.0                        # the table would kick
        st = self._snap(1, period=4, clock=50.0, y=80, down=4, dist=8, home=10, away=16)
        h, _ = sim7.simulate(t, st, 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertEqual(float((h == 13).mean()), 0.0)
        saved = sim7.GO_AHEAD
        sim7.GO_AHEAD = False
        try:
            h, _ = sim7.simulate(t, st, 400, np.random.default_rng(1), seed=3, max_steps=1)
        finally:
            sim7.GO_AHEAD = saved
        self.assertGreater(float((h == 13).mean()), 0.8)
        st = self._snap(1, period=4, clock=100.0, y=80, down=4, dist=8, home=10, away=16)
        h, _ = sim7.simulate(t, st, 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertGreater(float((h == 13).mean()), 0.8)     # with time left, the table decides

    def test_a_hold_redraws_would_be_touchdowns(self):
        import copy
        t = copy.deepcopy(self.tables)
        st = self._snap(1, y=95, dist=5)
        h0, _ = sim7.simulate(t, st, 2000, np.random.default_rng(1), seed=3, max_steps=1)
        t.td_hold[:] = 0.8
        h1, _ = sim7.simulate(t, st, 2000, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertLess(float((h1 == 16).mean()), 0.6 * float((h0 == 16).mean()))

    def test_the_settle_fit_holds_back_touchdowns_where_real_snaps_score_less(self):
        import copy
        t = copy.deepcopy(self.tables)
        grid = v7.PriorGrid.build(t, n_paths=60)
        items = v7.settle_states(self.matches, grid)
        self.assertTrue(items and all(0 <= c < sim7.N_SETTLE for c, *_ in items))
        cells = {}
        for c, *_ in items:
            cells[c] = cells.get(c, 0) + 1
        busy = max((c for c in cells if c % 4 >= 2), key=cells.get)     # inside the 30
        planted = [(c, f, th, False if c == busy else td) for c, f, th, td in items]
        fitted = v7.fit_settle(t, planted, rounds=3, n_paths=40, prior=0.0)
        n, real, before, after = fitted[busy]
        self.assertEqual(real, 0.0)
        self.assertGreater(t.td_hold[busy], 0.3)
        self.assertLess(after, before)
        self.assertTrue(((t.td_hold >= 0) & (t.td_hold <= sim7.SETTLE_MAX)).all())


class TestOvertime(unittest.TestCase):
    """Overtime as the feed shows it played: each side has the ball once,
    then the game ends the moment one side leads; one or two behind after
    a touchdown once the other side has had the ball, a side goes for two."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim7.Tables.build(_matches(60), min_records=20)

    def test_a_game_ends_once_both_have_had_the_ball_and_one_leads(self):
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 240.0, sim7.KICK
        st.team[:], st.home[:], st.away[:] = 0, 20, 20
        stats = {}
        h, a = sim7.simulate(self.tables, st, 4000, np.random.default_rng(1), seed=3, stats=stats)
        margin, points = np.abs(h - a).ravel(), (h + a - 40).ravel()
        self.assertGreater(stats["ot_decided"], 0)
        # decided by the first lead after both possessions: never by more
        # than a converted touchdown bar a defensive score, and no shoot-outs
        self.assertGreater(float((margin <= 8).mean()), 0.97)
        self.assertLess(float((points >= 17).mean()), 0.03)

    def test_one_behind_after_a_touchdown_it_goes_for_two(self):
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 150.0, sim7.CONV
        st.team[:], st.home[:], st.away[:] = 1, 27, 26      # the away side's six: one behind
        h, a = sim7.simulate(self.tables, st, 3000, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertTrue(set(np.unique(a)) <= {26, 28})        # never the kick to tie
        st.home[:] = 20                                       # six to lead: kicks as usual
        h, a = sim7.simulate(self.tables, st, 3000, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertIn(27, set(np.unique(a)))


class TestOvertimeRules(unittest.TestCase):
    """A touchdown that wins overtime ends it with no conversion, and a side behind once the other
    has had the ball never punts or kicks short of a tie."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim7.Tables.build(_matches(60), min_records=20)

    def test_a_walk_off_touchdown_ends_the_game_without_its_kick(self):
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 240.0, sim7.KICK
        st.team[:], st.home[:], st.away[:] = 0, 20, 20
        stats = {}
        h, a = sim7.simulate(self.tables, st, 6000, np.random.default_rng(1), seed=3, stats=stats,
                             common=False)
        points = (h + a - 40).ravel()
        self.assertGreater(stats.get("ot_walk_off", 0), 0)
        # a field goal answered by a touchdown ends at 9, never 10 with the kick
        self.assertLess(float((points == 10).mean()), 0.005)

    def test_level_before_the_other_side_has_had_the_ball_it_goes_as_real_overtimes_do(self):
        import copy
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 200.0, sim7.SCRIM
        st.team[:], st.home[:], st.away[:] = 0, 20, 20
        st.down[:], st.dist[:], st.y[:] = 4, 5, 80
        for ot_go in (0.9, 0.1):
            t = copy.deepcopy(self.tables)
            t.ot_go = ot_go
            stats = {}
            sim7.simulate(t, st, 4000, np.random.default_rng(1), seed=3, stats=stats, max_steps=1,
                          common=False)
            self.assertAlmostEqual(stats.get("fourth_go", 0) / 4000, ot_go, delta=0.03)
            self.assertEqual(stats.get("punt", 0), 0)

    def test_behind_by_a_touchdown_after_the_other_side_had_the_ball_it_goes_for_it(self):
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 150.0, sim7.SCRIM
        st.team[:], st.home[:], st.away[:] = 1, 27, 20      # the home side scored first
        st.down[:], st.dist[:], st.y[:] = 4, 10, 30
        stats = {}
        sim7.simulate(self.tables, st, 2000, np.random.default_rng(1), seed=3, stats=stats,
                      max_steps=1)
        self.assertEqual(stats.get("punt", 0) + stats.get("fg", 0), 0)
        self.assertEqual(stats["fourth_go"], 2000)


class TestFourthDownFit(unittest.TestCase):
    """The league's go curve, the part-of-game shifts and each player's own go and kick shifts, fitted
    together."""

    def test_the_fit_finds_the_bold_player_against_the_league(self):
        rng = np.random.default_rng(5)
        dp = sim7.DriveParams()
        recs = []
        for i in range(6000):
            y, t = int(rng.integers(20, 95)), int(rng.integers(1, 12))
            who = ["BOLD", "CALM", "MID"][i % 3]
            z = sim7._go_basis([y], [t])[0] @ np.array(dp.go_coef) + {"BOLD": 1.0, "CALM": -1.0, "MID": 0.0}[who]
            went = rng.random() < 1 / (1 + np.exp(-z))
            fg_range = 100 - y + 17 <= 55
            choice = "go" if went else ("fg" if fg_range and rng.random() < 0.9 else "punt")
            recs.append((0, 3, y, t, choice, who))
        go_coef, go_shift, go_p, kick_coef, kick_shift, kick_p = sim7.fit_fourth_downs(recs, dp)
        self.assertGreater(go_p["BOLD"] - go_p["MID"], 0.7)
        self.assertLess(go_p["CALM"] - go_p["MID"], -0.7)
        self.assertEqual(set(kick_p), {"BOLD", "CALM", "MID"})
        self.assertEqual(go_shift.shape, (4, 7))

    def test_the_fitted_curves_ride_in_the_tables(self):
        t = sim7.Tables.build(_matches(40), min_records=20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            t.save(path)
            back = sim7.Tables.load(path)
        self.assertEqual(back.drive.go_coef, t.drive.go_coef)
        self.assertEqual(back.drive.fg_kick_coef, t.drive.fg_kick_coef)
        self.assertNotEqual(t.drive.go_coef, sim7.DriveParams().go_coef)

    def test_the_offense_handle_follows_team_a(self):
        self.assertEqual(sim7._offense_handle({"offense": "TEAM_A", "team_a_side": "home"}, ("H", "A")), "H")
        self.assertEqual(sim7._offense_handle({"offense": "TEAM_A", "team_a_side": "away"}, ("H", "A")), "A")
        self.assertEqual(sim7._offense_handle({"offense": "TEAM_B", "team_a_side": "home"}, ("H", "A")), "A")
        self.assertIsNone(sim7._offense_handle({"offense": "TEAM_B"}, None))


class TestSecondHalfKick(unittest.TestCase):
    """The opening receiver kicks the second half; where the feed did not
    say who received, a coin per path."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim7.Tables.build(_matches(60), min_records=20)

    def _margin(self, kicks):
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 2, 0.0, sim7.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 30, 10, 10
        st.kicks_second_half[:] = kicks
        h, a = sim7.simulate(self.tables, st, 4000, np.random.default_rng(1), seed=21)
        return float((h - a).mean())

    def test_an_unknown_opening_is_a_coin(self):
        home_kicks, away_kicks, unknown = self._margin(0), self._margin(1), self._margin(-1)
        self.assertLess(home_kicks, away_kicks)             # receiving the half is worth points
        self.assertLess(home_kicks, unknown)
        self.assertLess(unknown, away_kicks)
        self.assertEqual(int(sim7.Start(1).kicks_second_half[0]), -1)

    def test_a_snapshot_without_the_opening_says_so(self):
        state = mock.Mock(pending_conversion=None, down=None, offense=sim7_home(), period=1,
                          clock_seconds=200.0, home_score=0, away_score=0, opening_receiver=None)
        self.assertEqual(v7.start_from(state)["kicks_second_half"], -1)
        state.opening_receiver = sim7_home()
        self.assertEqual(v7.start_from(state)["kicks_second_half"], 0)


def sim7_home():
    return v7.HOME


def sim7_profile():
    from .. import players
    return players.Profile()


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.matches = _with_handles(_matches(40))
        # the in-play shift and the quarter-start fit are off by default; the
        # build is tested with them on
        saved = v7.IN_PLAY_FIT, v7.QUARTER_START_FIT
        v7.IN_PLAY_FIT = v7.QUARTER_START_FIT = True
        try:
            v7.build(cls.matches, cls.tmp.name, grid_paths=60, verbose=False)
        finally:
            v7.IN_PLAY_FIT, v7.QUARTER_START_FIT = saved
        cls.tables = sim7.Tables.load(os.path.join(cls.tmp.name, "v7tables.npz"))
        cls.grid = v7.PriorGrid.load(os.path.join(cls.tmp.name, "v7grid.npz"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_writes_the_model_and_the_profiles(self):
        for name in ("v7tables.npz", "v7grid.npz", "v7players.json"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp.name, name)), name)
        book = v7.players_book(self.tmp.name)
        self.assertEqual(set(book.players), {"ALPHA", "BRAVO", "CHARLIE", "DELTA"})
        # each player's go and kick shifts come from the joint 4th-down fit
        self.assertTrue(any(p.aggression != 0.0 for p in book.players.values()))
        self.assertTrue(all(isinstance(p.kick, float) for p in book.players.values()))

    def test_a_quarters_points_run_to_the_next_quarters_first_row(self):
        # the feed posts a quarter's last score on the next quarter's first
        # row: that score belongs to the quarter before
        base = dict(team_a_side="home", offense="TEAM_A", play_kind="SCRIMMAGE", down="1",
                    distance="10", field_position="25", clock_seconds="200", play_messages="",
                    final_p1="17", final_p2="10", match_code="AFX")
        rows = [dict(base, message="1", period="1", score_p1="0", score_p2="0"),
                dict(base, message="2", period="1", score_p1="7", score_p2="0"),
                dict(base, message="3", period="2", score_p1="10", score_p2="0"),
                dict(base, message="4", period="3", score_p1="10", score_p2="3"),
                dict(base, message="5", period="4", score_p1="17", score_p2="10")]
        got = {q: pts for q, _, _, pts in v7.quarter_start_states({"AFX": rows}, self.grid)}
        self.assertEqual(got, {1: 10, 2: 3, 3: 14, 4: 0})

    def test_the_two_minute_mark_states(self):
        base = dict(team_a_side="home", offense="TEAM_A", play_kind="SCRIMMAGE", down="1",
                    distance="10", field_position="25", play_messages="", final_p1="17",
                    final_p2="10", match_code="AFX")
        rows = [dict(base, message="1", period="2", clock_seconds="200", score_p1="0", score_p2="0"),
                dict(base, message="2", period="2", clock_seconds="118", score_p1="3", score_p2="0"),
                dict(base, message="3", period="2", clock_seconds="30", score_p1="3", score_p2="0"),
                dict(base, message="4", period="3", clock_seconds="240", score_p1="10", score_p2="0"),
                dict(base, message="5", period="4", clock_seconds="100", score_p1="10", score_p2="10")]
        got = {q: pts for q, _, _, pts in v7.late_start_states({"AFX": rows}, self.grid)}
        self.assertEqual(got, {2: 7, 4: 7})

    def test_the_build_fits_the_quarters_and_the_two_minute_drills(self):
        import copy
        self.assertFalse(v7.QUARTER_START_FIT)          # off by default (see v7.py)
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = 0.0          # this class's build has the in-play shift on too
        quarters, lates = v7.fit_quarter_levels(
            tables, v7.quarter_start_states(self.matches, self.grid),
            v7.late_start_states(self.matches, self.grid), rounds=1, n_paths=200)
        for real, got in list(quarters.values()) + list(lates.values()):
            self.assertAlmostEqual(got, real, delta=max(0.6, 0.12 * real))
        self.assertEqual(set(lates), {2, 4})
        self.assertTrue(np.any(self.tables.late_theta))

    def test_the_two_minute_level_is_only_the_last_two_minutes(self):
        import copy
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = 0.0
        tables.late_theta[:] = 0.0
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:], st.y[:] = 2, 239.0, sim7.SCRIM, 30
        plain = sim7.simulate(tables, st, 3000, np.random.default_rng(1), seed=4, max_steps=2)
        tables.late_theta[2] = 1.0
        early = sim7.simulate(tables, st, 3000, np.random.default_rng(1), seed=4, max_steps=2)
        self.assertTrue(np.array_equal(plain[0], early[0]))    # 4:00 left: untouched
        st.clock[:] = 100.0
        tables.late_theta[2] = 0.0
        a = sim7.simulate(tables, st, 3000, np.random.default_rng(1), seed=4)
        tables.late_theta[2] = 1.0
        b = sim7.simulate(tables, st, 3000, np.random.default_rng(1), seed=4)
        self.assertGreater(float((b[0] + b[1]).mean()), float((a[0] + a[1]).mean()))

    def test_the_quarter_fit_on_real_states_converges_when_asked(self):
        # on fixed states and thetas the fit brings the quarters to what
        # games scored
        import copy
        tables = copy.deepcopy(self.tables)
        items = v7.quarter_start_states(self.matches, self.grid)
        v7.fit_period_theta_states(tables, items, rounds=6, n_paths=300)
        _, got, real = v7.fit_period_theta_states(tables, items, rounds=1, n_paths=300)
        for q in real:
            self.assertAlmostEqual(got[q], real[q], delta=max(0.4, 0.08 * real[q]))

    def test_the_rest_of_game_fit_leaves_what_games_really_left(self):
        import copy
        tables = copy.deepcopy(self.tables)
        items = v7.rest_of_game_states(self.matches, self.grid, every=3)
        q3 = sim7.SEGMENTS.index("Q3")
        self.assertTrue({it[0][0] for it in items} >= {1, 2, 3, q3, 5, 6})
        # games that left a point fewer than they did from every state in
        # the third quarter: its in-play scoring comes down, most of the way
        lower = [(c, st, th, pts - 1 if c[0] == q3 else pts) for c, st, th, pts in items]
        before = tables.inplay_theta.copy()
        segs, bands = v7.fit_rest_of_game(tables, lower, n_paths=120, rounds=6, max_states=600)
        real, simulated_before, after = segs[q3]
        self.assertLess(abs(after - real), 0.5 * abs(simulated_before - real))
        self.assertLess(tables.inplay_theta[q3].mean(), before[q3].mean() - 0.02)
        for (g, band), (r, b, a) in bands.items():
            self.assertIn(band, range(sim7.N_BANDS))
        # and the kickoff scoring, the pre-match's, is untouched
        self.assertTrue((tables.period_theta == self.tables.period_theta).all())

    def test_the_in_play_shift_is_only_in_play(self):
        import copy
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = -0.5
        st = sim7.Start(1)
        st.period[:], st.clock[:], st.phase[:], st.y[:] = 3, 200.0, sim7.SCRIM, 30
        runs = {flag: sim7.simulate(tables, st, 3000, np.random.default_rng(1), seed=3, in_play=flag)
                for flag in (False, True)}
        total = {flag: float((h + a).mean()) for flag, (h, a) in runs.items()}
        self.assertLess(total[True], total[False] - 1.0)
        self.assertFalse(v7.IN_PLAY_FIT)
        # a build with it on fitted it and saved it
        self.assertTrue(np.abs(self.tables.inplay_theta[1:7]).sum() > 0)

    def test_the_in_play_shift_moves_the_total_and_not_the_margin(self):
        import copy
        from .. import playover
        rows = v7.resolve_sides(list(self.matches.values())[0])
        states, messages = [], []
        for r in rows[10:40:5]:
            st, _ = playover.state_for(r)
            if st is not None:
                states.append(st)
                messages.append(int(r["message"]))
        prof = (sim7_profile(), sim7_profile())
        books = {}
        for name, shift in (("plain", 0.0), ("shifted", -0.6)):
            tables = copy.deepcopy(self.tables)
            tables.inplay_theta[:] = shift
            books[name] = v7.price_states(tables, (0.0, 0.0), v7.Variant("v7"), [], True, states,
                                          messages, prof, 400, np.random.default_rng(2), seed=11)
        x = np.arange(len(books["plain"][0][1]))
        for (mp0, tp0), (mp1, tp1) in zip(books["plain"], books["shifted"]):
            self.assertTrue(np.allclose(mp0, mp1))
            self.assertLess((tp1 * x).sum(), (tp0 * x).sum())

    def test_nothing_of_gameplais_is_read(self):
        # every prod price column moved far off: v7's priors, states and
        # prices come out the same
        moved = {c: [dict(r, prematch_line_52="-20.5", prematch_prob_52="0.9",
                          prematch_line_54="80.5", prematch_prob_54="0.9", prematch_prob_50="0.99",
                          line_52="-20.5", prob_52="0.9", line_54="80.5", prob_54="0.9",
                          prob_50="0.99") for r in rows]
                 for c, rows in self.matches.items()}
        for fn, k in ((v7.in_game_states, 1), (v7.rest_of_game_states, 2),
                      (v7.quarter_start_states, 2)):
            a, b = fn(self.matches, self.grid), fn(moved, self.grid)
            self.assertTrue(a)
            self.assertEqual([x[k] for x in a], [x[k] for x in b])
            self.assertTrue(all(tuple(x[k]) == v7.LEAGUE_THETA for x in a))
        rows = next(iter(self.matches.values()))
        a = v7_stream.match_books(self.tables, self.grid, v7.Variant("v7"), rows, 60,
                                  np.random.default_rng(1))
        b = v7_stream.match_books(self.tables, self.grid, v7.Variant("v7"), moved[rows[0]["match_code"]],
                                  60, np.random.default_rng(1))
        self.assertTrue(a)
        for (_, mp1, tp1), (_, mp2, tp2) in zip(a, b):
            self.assertTrue(np.array_equal(mp1, mp2) and np.array_equal(tp1, tp2))
        for gone in ("prior_lines", "recent_total_shade"):
            self.assertFalse(hasattr(v7, gone))
        self.assertFalse(hasattr(v7.PriorGrid, "fit"))

    def test_the_build_needs_history_on_the_command_line(self):
        from ..__main__ import main
        path = os.path.join(self.tmp.name, "snaps_cli.csv")
        with open(path, "w") as fh:
            fh.write("match_code\n")
        with self.assertRaises(SystemExit) as caught:
            main(["v7-build", path, "--out", os.path.join(self.tmp.name, "cli")])
        self.assertIn("--history", str(caught.exception))

    def test_handles_come_off_the_export(self):
        rows = next(iter(self.matches.values()))
        self.assertEqual(v7.handles_of(rows), (rows[0]["home_handle"], rows[0]["away_handle"]))
        self.assertIsNone(v7.handles_of([{"home_handle": "", "away_handle": None}]))

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
        graded, _ = v7.run(path, os.path.join(self.tmp.name, "v7tables.npz"),
                           os.path.join(self.tmp.name, "v7grid.npz"),
                           [v7.Variant("v7"), v7.Variant("plain", profiles=False, pace=False)],
                           n_paths=100, workers=1)
        self.assertGreater(len(graded), 50)
        for _, row, probs in graded:
            self.assertTrue(0 < probs["v7"] < 1 and 0 < probs["plain"] < 1)

    def test_the_stream_finds_the_model_and_quotes(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        prod = [(code, 54, None, 50.0, 2.0, "Total points over 30.5", int(r["message"]), "OPEN", "true")
                for r in rows]
        quotes = v7_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
        self.assertTrue(quotes)
        self.assertTrue(all(q[1] == 54 and 0 < q[3] < 100 for q in quotes))

    def test_fit_means_finds_the_thetas_that_score_the_points(self):
        g = self.grid
        x_m = np.arange(g.margin.shape[-1]) - v7.MARGIN_MAX
        x_t = np.arange(g.total.shape[-1])
        for i, j in ((8, 8), (4, 11), (12, 6)):
            margin, total = (g.margin[i, j] * x_m).sum(), (g.total[i, j] * x_t).sum()
            got = v7.fit_means(g, (total + margin) / 2, (total - margin) / 2)
            self.assertAlmostEqual(got[0], g.grid[i], delta=0.02)
            self.assertAlmostEqual(got[1], g.grid[j], delta=0.02)
        low, high = v7.fit_means(g, 14, 14), v7.fit_means(g, 24, 24)
        self.assertLess(sum(low), sum(high))

    def test_a_fixed_seed_prices_a_match_the_same_every_time(self):
        rows = next(iter(self.matches.values()))
        a = v7_stream.match_books(self.tables, self.grid, v7.Variant("v7"), rows, 80,
                                  np.random.default_rng(1), means=(20.0, 14.0))
        b = v7_stream.match_books(self.tables, self.grid, v7.Variant("v7"), rows, 80,
                                  np.random.default_rng(99), means=(20.0, 14.0))
        self.assertTrue(a)
        for (m1, mp1, tp1), (m2, mp2, tp2) in zip(a, b):
            self.assertEqual(m1, m2)
            self.assertTrue(np.array_equal(mp1, mp2) and np.array_equal(tp1, tp2))

    def test_with_nb2_means_prods_pre_match_quotes_are_not_read(self):
        rows = next(iter(self.matches.values()))
        other = [dict(r, prematch_line_52="-20.5", prematch_prob_52="0.9",
                      prematch_line_54="80.5", prematch_prob_54="0.9", prematch_prob_50="0.99",
                      line_52="-20.5", prob_52="0.9", line_54="80.5", prob_54="0.9")
                 for r in rows]
        args = (self.tables, self.grid, v7.Variant("v7"))
        a = v7_stream.match_books(*args, rows, 80, np.random.default_rng(1), means=(20.0, 14.0))
        b = v7_stream.match_books(*args, other, 80, np.random.default_rng(1), means=(20.0, 14.0))
        for (_, mp1, tp1), (_, mp2, tp2) in zip(a, b):
            self.assertTrue(np.array_equal(mp1, mp2) and np.array_equal(tp1, tp2))

    def _snapshot_file(self):
        path = os.path.join(self.tmp.name, "snaps.csv")
        if not os.path.exists(path):
            self.test_grades_a_snapshot_file_with_profiles()
        return path

    def test_a_model_with_nb2_takes_its_prior_from_there(self):
        path = self._snapshot_file()
        tables, grid = (os.path.join(self.tmp.name, n) for n in ("v7tables.npz", "v7grid.npz"))
        codes = sorted(self.matches)[:6]

        class Fake:
            league = (17.0, 17.0)

            def __init__(self, level):
                self.level = level

            def means(self, schedule, n_sims=0):
                return {r["MATCH_CODE"]: self.level for r in schedule}

        history = [{"MATCH_CODE": c} for c in codes]
        with mock.patch.object(v7, "prematch_model", return_value=Fake((12.0, 12.0))):
            with self.assertRaises(SystemExit) as caught:
                v7.run(path, tables, grid, [v7.Variant("v7")], matches=codes, workers=1)
            self.assertIn("--history", str(caught.exception))
            low, _ = v7.run(path, tables, grid, [v7.Variant("v7")], matches=codes, n_paths=100,
                            workers=1, history=history)
        with mock.patch.object(v7, "prematch_model", return_value=Fake((26.0, 26.0))):
            high, _ = v7.run(path, tables, grid, [v7.Variant("v7")], matches=codes, n_paths=100,
                             workers=1, history=history)
        over = lambda graded: np.mean([p["v7"] for _, r, p in graded if r.market_id == 54])
        self.assertLess(over(low), over(high))
        # no prediction for a match: skipped, never priced off prod
        _, skipped = v7.run(path, tables, grid, [v7.Variant("v7")], matches=codes, n_paths=50,
                            workers=1, priors={codes[0]: (20.0, 14.0)})
        self.assertGreater(skipped["no_prematch_prediction"], 0)

    def test_the_stream_with_nb2_needs_match_info_and_falls_back_to_the_league(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        prod = [(code, 54, None, 50.0, 2.0, "Total points over 30.5", int(r["message"]), "OPEN", "true")
                for r in rows]

        class Fake:
            league = (17.0, 17.0)
            asked = None

            def means(self, schedule, n_sims=0):
                Fake.asked = schedule
                return {}

        with mock.patch.object(v7, "prematch_model", return_value=Fake()):
            with self.assertRaises(SystemExit):
                v7_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
            quotes = v7_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50,
                                                  workers=1, match_info=[])
        self.assertTrue(quotes)
        self.assertEqual(Fake.asked, [])


    def test_the_even_line_is_the_half_point_nearest_even_money(self):
        pmf = np.zeros(80)
        pmf[[38, 41, 44, 47]] = [0.2, 0.25, 0.3, 0.25]
        # P(> 38.5) = 0.8, P(> 41.5 .. 43.5) = 0.55, P(> 44.5) = 0.25: of the
        # three 0.55 lines, the one nearest the mean (42.8)
        self.assertEqual(v7.even_line(pmf, 0), 42.5)
        margin = np.zeros(2 * v7.MARGIN_MAX + 1)
        margin[v7.MARGIN_MAX + np.array([-3, 3, 7])] = [0.3, 0.45, 0.25]
        # -2.5 .. 2.5 all give P(home by more) = 0.7: the one nearest the mean (2.2)
        self.assertEqual(v7.even_line(margin, v7.MARGIN_MAX), 2.5)
        line = v7.even_line(margin, v7.MARGIN_MAX)
        self.assertLessEqual(abs(v7.market_prob(52, line, margin, pmf) - 0.5), 0.2 + 1e-9)

    def test_the_stream_moves_its_own_line_with_the_game(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        prod = [(code, m, None, 50.0, 2.0, v7_stream.description(m, 30.5 if m in (54, 55) else 0.5),
                 int(r["message"]), "OPEN", "true") for r in rows for m in (52, 53, 54, 55)]
        own = v7_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=100, workers=1)
        at_prod = v7_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=100,
                                               workers=1, lines="prod")
        parse = v7_stream._parse_line
        self.assertEqual({parse(q[5]) for q in at_prod if q[1] == 54}, {30.5})
        totals = [(q[6], parse(q[5])) for q in own if q[1] == 54]
        self.assertGreater(len({t for _, t in totals}), 1)         # it moves with the game
        self.assertTrue(all(t % 1 == 0.5 for _, t in totals))
        # never below the points already on the board
        board = {int(r["message"]): int(r["score_p1"] or 0) + int(r["score_p2"] or 0) for r in rows}
        for message, line in totals:
            on_board = board[max(m for m in board if m <= message)]
            self.assertGreater(line, on_board - 1)

    def test_the_kick_off_is_priced_off_the_pre_match_means(self):
        rng = np.random.default_rng(3)
        prof = (players.Profile(), players.Profile())
        strong = v7.price_kickoff(self.tables, v7.prior_theta(self.grid, (26.0, 12.0)), v7.Variant("v7"),
                                  prof, 600, rng, seed=1)
        weak = v7.price_kickoff(self.tables, v7.prior_theta(self.grid, (12.0, 26.0)), v7.Variant("v7"),
                                prof, 600, rng, seed=1)
        for margin, total in (strong, weak):
            self.assertAlmostEqual(margin.sum(), 1.0, places=6)
            self.assertAlmostEqual(total.sum(), 1.0, places=6)
        home_win = lambda book: v7.market_prob(50, 0.0, book[0], book[1])
        self.assertGreater(home_win(strong), 0.5)
        self.assertLess(home_win(weak), 0.5)

    def test_pre_match_and_pre_play_rows_get_the_kick_off_price(self):
        margin = np.zeros(2 * v7.MARGIN_MAX + 1)
        margin[v7.MARGIN_MAX + 7] = 1.0
        total = np.zeros(v7.TOTAL_MAX + 1)
        total[40] = 1.0
        kick_margin = np.zeros_like(margin)
        kick_margin[v7.MARGIN_MAX - 3] = 1.0
        row = lambda m: ("M", 50, None, 50.0, 2.0, "PLAYER 1 to win", m, "OPEN", "true")
        prod = [row(None), row(3), row(7), row(12)]
        out = v7_stream.quote_rows("M", [(10, margin, total)], prod, first_play_message=5,
                                   kickoff=(kick_margin, total))
        got = {q[6]: q[3] for q in out}
        self.assertEqual(set(got), {None, 3, 12})           # 7: the first play is under way
        self.assertEqual((got[None], got[3], got[12]), (0.01, 0.01, 99.99))
        self.assertEqual({q[6] for q in v7_stream.quote_rows("M", [(10, margin, total)], prod, 5)},
                         {12})

    def test_a_match_whose_side_is_not_known_still_gets_its_pre_match_price(self):
        rows = [dict(r, team_a_side="") for r in next(iter(self.matches.values()))]
        code = rows[0]["match_code"]
        prod = [(code, 50, None, 50.0, 2.0, "PLAYER 1 to win", None, "OPEN", "true")] + \
               [(code, 50, None, 50.0, 2.0, "PLAYER 1 to win", int(r["message"]), "OPEN", "true")
                for r in rows]
        quotes = v7_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
        self.assertEqual([q[6] for q in quotes], [None])

    def test_a_missing_model_says_how_to_build_it(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(SystemExit) as caught:
                v7_stream.model_paths(empty)
        self.assertIn("v7-build", str(caught.exception))


_FIT_STUB = """
    import argparse, csv, os
    from nb2_prior_names import RATINGS
    p = argparse.ArgumentParser(); p.add_argument("--input"); a = p.parse_args()
    rows = list(csv.DictReader(open(a.input)))
    home = sum(float(r["PLAYER_1_FINAL_SCORE"]) for r in rows) / len(rows)
    away = sum(float(r["PLAYER_2_FINAL_SCORE"]) for r in rows) / len(rows)
    for name in RATINGS.values():
        open(name, "w").write(f"{home},{away},{len(rows)}")
"""

_PREDICT_STUB = """
    import argparse, csv
    p = argparse.ArgumentParser()
    for k in ("schedule", "stream-col", "n-sims", "leaderboard", "teams", "streams", "joint"):
        p.add_argument("--" + k)
    a = p.parse_args()
    home, away, _ = open(a.leaderboard).read().split(",")
    with open("NB2_joint_schedule_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Match Id", "Pred_P1_Points", "Pred_P2_Points", "Prediction_Status"])
        for r in csv.DictReader(open(a.schedule)):
            status = "UNKNOWN_PLAYER" if r["Player 1 Name"] == "NOBODY" else "OK"
            w.writerow([r["Match Id"], home, away, status])
"""


class TestNB2Prior(unittest.TestCase):
    """The bridge to nb2/'s scripts, run against stand-ins that fit a plain
    average, so what is tested is the plumbing: which history is fitted
    on, the walk-forward level scale and the league-average fallback."""

    BEFORE = dt.datetime(2026, 9, 20)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        stub = os.path.join(self.tmp.name, "nb2")
        os.makedirs(stub)
        with open(os.path.join(stub, "nb2_prior_names.py"), "w") as fh:
            fh.write(f"RATINGS = {nb2_prior.RATINGS!r}\n")
        with open(os.path.join(stub, nb2_prior.FIT_SCRIPT), "w") as fh:
            fh.write("import sys; sys.path.insert(0, %r)\n" % stub + textwrap.dedent(_FIT_STUB))
        with open(os.path.join(stub, nb2_prior.PREDICT_SCRIPT), "w") as fh:
            fh.write(textwrap.dedent(_PREDICT_STUB))
        self.patches = [mock.patch.object(nb2_prior, "NB2_DIR", stub),
                        mock.patch.object(nb2_prior, "_need_libraries", lambda: None)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def history(self):
        rows = []
        for d in range(55):                       # Aug 1 .. Sep 24, two a day
            day = dt.datetime(2026, 8, 1) + dt.timedelta(days=d)
            late = day >= self.BEFORE - dt.timedelta(days=7)
            for k in range(2):
                rows.append({"MATCH_CODE": f"AF{d:02d}{k}", "SPORT_CODE": "AF", "STREAM_NUMBER": "1",
                             "SCHEDULED_START_TIME_UTC": f"{day + dt.timedelta(hours=k):%Y-%m-%d %H:%M:%S}",
                             "PLAYER_1_HANDLE": "ALPHA", "PLAYER_1_TEAM": "A",
                             "PLAYER_2_HANDLE": "BRAVO", "PLAYER_2_TEAM": "B",
                             "PLAYER_1_FINAL_SCORE": "24" if late else "20",
                             "PLAYER_2_FINAL_SCORE": "16" if late else "14"})
        rows[3]["PLAYER_1_FINAL_SCORE"] = ""       # unsettled: never fitted on
        return rows

    def test_builds_on_the_history_before_the_cut_off_only(self):
        history = self.history()
        out = os.path.join(self.tmp.name, "model")
        pre = nb2_prior.Prematch.build(history, out, self.BEFORE)
        fitted = [r for r in history if r["PLAYER_1_FINAL_SCORE"]
                  and nb2_prior._start(r) < self.BEFORE]
        self.assertEqual(pre.meta["fitted_on"], len(fitted))
        # the level: fitted before Sep 13 (20 + 14), scored on Sep 13-19 (24 + 16)
        self.assertEqual(pre.meta["scale_matches"], 14)
        self.assertAlmostEqual(pre.meta["scale_raw_ratio"], 40 / 34, places=9)
        self.assertAlmostEqual(pre.scale, 1 + (40 / 34 - 1) * 14 / (14 + nb2_prior.SCALE_PRIOR),
                               places=9)
        home = sum(float(r["PLAYER_1_FINAL_SCORE"]) for r in fitted) / len(fitted)
        away = sum(float(r["PLAYER_2_FINAL_SCORE"]) for r in fitted) / len(fitted)
        schedule = [dict(history[-1], MATCH_CODE="NEW1"),
                    dict(history[-1], MATCH_CODE="NEW2", PLAYER_1_HANDLE="NOBODY")]
        means = nb2_prior.Prematch(out).means(schedule)
        self.assertAlmostEqual(means["NEW1"][0], home * pre.scale, places=6)
        self.assertAlmostEqual(means["NEW1"][1], away * pre.scale, places=6)
        self.assertEqual(means["NEW2"], pre.league)          # NB2 cannot price it: unscaled
        self.assertTrue(nb2_prior.Prematch.exists(out))
        self.assertFalse(nb2_prior.Prematch.exists(self.tmp.name))

    def test_v7_build_with_history_writes_the_prematch_model(self):
        from .test_v3 import _matches
        matches = _matches(12)
        out = os.path.join(self.tmp.name, "v7")
        v7.build(matches, out, grid_paths=40, verbose=False, history=self.history(),
                 before=self.BEFORE)
        self.assertIsNotNone(v7.prematch_model(out))
        self.assertIsNotNone(v7.prematch_model(os.path.join(out, "v7tables.npz")))

    def test_a_failing_script_says_what_went_wrong(self):
        with open(os.path.join(nb2_prior.NB2_DIR, nb2_prior.FIT_SCRIPT), "w") as fh:
            fh.write("raise SystemExit('bad input')\n")
        with self.assertRaises(SystemExit) as caught:
            nb2_prior.fit(self.history(), os.path.join(self.tmp.name, "x"))
        self.assertIn("bad input", str(caught.exception))

    def test_load_history_keeps_the_sport_and_checks_the_columns(self):
        path = os.path.join(self.tmp.name, "h.csv")
        rows = self.history()[:3] + [dict(self.history()[4], SPORT_CODE="BB")]
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, nb2_prior.HISTORY_FIELDS)
            w.writeheader()
            w.writerows(rows)
        self.assertEqual(len(nb2_prior.load_history(path)), 3)
        with open(path, "w", newline="") as fh:
            fh.write("MATCH_CODE,SPORT_CODE\nAF1,AF\n")
        with self.assertRaises(SystemExit):
            nb2_prior.load_history(path)


_GLMER_FIT_STUB = """
    import argparse, csv, json, os
    p = argparse.ArgumentParser()
    for k in ("history", "before", "out"):
        p.add_argument("--" + k)
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.history)))
    home = sum(float(r["PLAYER_1_FINAL_SCORE"]) for r in rows) / len(rows)
    away = sum(float(r["PLAYER_2_FINAL_SCORE"]) for r in rows) / len(rows)
    open(os.path.join(a.out, "model.rds"), "w").write(f"{home},{away}")
    json.dump({"feature_set": "form", "weighting": "hl60", "mode": "global", "form_half_life": 10,
               "before": a.before, "fitted_on": len(rows)},
              open(os.path.join(a.out, "model_info.json"), "w"))
"""

_GLMER_PREDICT_STUB = """
    import argparse, csv, json, os
    p = argparse.ArgumentParser()
    for k in ("model", "schedule", "history", "n-sims", "out"):
        p.add_argument("--" + k)
    a = p.parse_args()
    json.dump(vars(a), open(os.path.join(os.path.dirname(a.out), "args.json"), "w"))
    home, away = open(a.model).read().split(",")
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "Pred_P1_Points", "Pred_P2_Points", "Prediction_Status"])
        for r in csv.DictReader(open(a.schedule)):
            if r["PLAYER_1_HANDLE"] != "NOBODY":
                w.writerow([r["MATCH_CODE"], home, away, "OK"])
"""


class TestGlmerPrior(unittest.TestCase):
    """The bridge to glmer/'s R scripts, run against Python stand-ins (so R isn't needed): the
    history it fits on, what predict.R is handed, the league-average fallback, and a v7 build
    that records which prior it prices off."""

    BEFORE = dt.datetime(2026, 9, 20)
    history = TestNB2Prior.history

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        stub = os.path.join(self.tmp.name, "glmer")
        os.makedirs(stub)
        with open(os.path.join(stub, glmer_prior.FIT_SCRIPT), "w") as fh:
            fh.write(textwrap.dedent(_GLMER_FIT_STUB))
        with open(os.path.join(stub, glmer_prior.PREDICT_SCRIPT), "w") as fh:
            fh.write(textwrap.dedent(_GLMER_PREDICT_STUB))
        self.patches = [mock.patch.object(glmer_prior, "GLMER_DIR", stub),
                        mock.patch.object(glmer_prior, "RSCRIPT", sys.executable)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_builds_on_the_history_before_the_cut_off_only(self):
        history = self.history()
        out = os.path.join(self.tmp.name, "model")
        pre = glmer_prior.Prematch.build(history, out, self.BEFORE)
        fitted = [r for r in history if r["PLAYER_1_FINAL_SCORE"]
                  and nb2_prior._start(r) < self.BEFORE]
        self.assertEqual(pre.meta["fitted_on"], len(fitted))
        self.assertEqual(pre.meta["model"]["fitted_on"], len(fitted))
        self.assertEqual(pre.scale, 1.0)
        home = sum(float(r["PLAYER_1_FINAL_SCORE"]) for r in fitted) / len(fitted)
        away = sum(float(r["PLAYER_2_FINAL_SCORE"]) for r in fitted) / len(fitted)
        schedule = [dict(history[-1], MATCH_CODE="NEW1"),
                    dict(history[-1], MATCH_CODE="NEW2", PLAYER_1_HANDLE="NOBODY")]
        means = glmer_prior.Prematch(out).means(schedule)
        self.assertAlmostEqual(means["NEW1"][0], home, places=6)
        self.assertAlmostEqual(means["NEW1"][1], away, places=6)
        self.assertEqual(means["NEW2"], pre.league)           # not priced: the league's average
        self.assertTrue(glmer_prior.Prematch.exists(out))
        self.assertFalse(glmer_prior.Prematch.exists(self.tmp.name))
        self.assertIn("glmer form / hl60", pre.describe())

    def test_predict_gets_the_models_history_and_skips_the_simulation(self):
        out = os.path.join(self.tmp.name, "model")
        pre = glmer_prior.Prematch.build(self.history(), out, self.BEFORE)
        last = self.history()[-1]
        pre.means([dict(last, MATCH_CODE="NEW1")], n_sims=1000)
        with open(os.path.join(out, "predict", "args.json")) as fh:
            args = json.load(fh)
        self.assertEqual(args["n_sims"], "0")                 # expected points only
        self.assertEqual(os.path.normcase(args["history"]),
                         os.path.normcase(os.path.abspath(os.path.join(out, glmer_prior.HISTORY))))
        with open(args["schedule"], newline="") as fh:
            row = next(csv.DictReader(fh))
        # finals the rows carry reach the form features; the R side reads only earlier matches
        self.assertEqual(row["PLAYER_1_FINAL_SCORE"], last["PLAYER_1_FINAL_SCORE"])

    def test_v7_build_prices_off_the_prior_it_was_built_with(self):
        out = os.path.join(self.tmp.name, "v7")
        v7.build(_matches(12), out, grid_paths=40, verbose=False, history=self.history(),
                 before=self.BEFORE, prior="glmer")
        self.assertIsInstance(v7.prematch_model(out), glmer_prior.Prematch)
        self.assertIsInstance(v7.prematch_model(os.path.join(out, "v7tables.npz")),
                              glmer_prior.Prematch)
        self.assertFalse(os.path.exists(os.path.join(out, "nb2")))
        with open(os.path.join(out, v7.PRIOR_FILE)) as fh:
            self.assertEqual(json.load(fh), {"prior": "glmer"})

    def test_a_model_built_before_the_choice_still_reads_nb2(self):
        old = os.path.join(self.tmp.name, "old")
        os.makedirs(os.path.join(old, "nb2"))
        with mock.patch.object(nb2_prior.Prematch, "exists", return_value=True), \
                mock.patch.object(nb2_prior.Prematch, "__init__", return_value=None):
            self.assertIsInstance(v7.prematch_model(old), nb2_prior.Prematch)
        self.assertIsNone(v7.prematch_model(self.tmp.name))  # nothing built here

    def test_an_unknown_prior_is_refused(self):
        with self.assertRaises(ValueError):
            v7.build(_matches(2), os.path.join(self.tmp.name, "x"), verbose=False, prior="elo")

    def test_a_failing_script_says_what_went_wrong(self):
        with open(os.path.join(glmer_prior.GLMER_DIR, glmer_prior.FIT_SCRIPT), "w") as fh:
            fh.write("raise SystemExit('bad input')\n")
        with self.assertRaises(SystemExit) as caught:
            glmer_prior.fit(self.history(), os.path.join(self.tmp.name, "x"), self.BEFORE)
        self.assertIn("bad input", str(caught.exception))

    def test_without_r_it_says_how_to_get_it(self):
        with mock.patch.object(glmer_prior, "RSCRIPT", None), \
                mock.patch.dict(os.environ, {"RSCRIPT": ""}), \
                mock.patch.object(glmer_prior.shutil, "which", return_value=None), \
                mock.patch.object(glmer_prior.sys, "platform", "linux"):
            with self.assertRaises(SystemExit) as caught:
                glmer_prior.rscript()
        self.assertIn("install_packages.R", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
