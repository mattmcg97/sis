"""v10: a fresh copy of v9 (sim10.py, v10.py, v10_stream.py), the base for the next changes."""

import csv
import datetime as dt
import json
import math
import os
import random
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np

from .. import glmer_prior, nb2_prior, players, sim9, sim10, v10, v10_stream
from .fakes import _matches, _with_handles


class TestV10IsV9(unittest.TestCase):
    """v10 starts as v9 copied as it was: the same tables and the same games."""

    def test_v10_plays_as_v9(self):
        matches = _matches()
        t10 = sim10.Tables.build(matches, min_records=20)
        t9 = sim9.Tables.build(matches, min_records=20)
        st = sim10.Start(4)
        st.period[:], st.clock[:], st.phase[:] = [2, 3, 4, 4], [30.0, 150.0, 50.0, 25.0], sim10.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, [50, 50, 80, 75], [3, 17, 10, 10], [10, 10, 16, 13]
        st.down[:] = [1, 2, 1, 4]
        a = sim10.simulate(t10, st, 300, np.random.default_rng(1), seed=9)
        b = sim9.simulate(t9, st, 300, np.random.default_rng(1), seed=9)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))


class TestCommonRandomNumbers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = sim10.Tables.build(_matches(), min_records=20)

    def _two_of(self, common):
        st = sim10.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim10.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 14, 10
        return sim10.simulate(self.tables, st, 300, np.random.default_rng(1), common=common)

    def test_the_same_state_twice_plays_the_same_games(self):
        h, a = self._two_of(True)
        self.assertTrue((h[0] == h[1]).all() and (a[0] == a[1]).all())
        h, a = self._two_of(False)
        self.assertFalse((h[0] == h[1]).all() and (a[0] == a[1]).all())

    def test_the_stream_is_uniform_and_repeatable(self):
        path, step = np.arange(20000), np.full(20000, 3)
        u = sim10._uniform(12345, path, step, 7)
        self.assertTrue(((u >= 0) & (u < 1)).all())
        self.assertAlmostEqual(float(u.mean()), 0.5, delta=0.01)
        self.assertTrue((u == sim10._uniform(12345, path, step, 7)).all())
        self.assertFalse((u == sim10._uniform(12345, path, step, 8)).all())


class TestPlayCalling(unittest.TestCase):
    """The play call (a clock-stopping play or one that keeps the clock
    running), the clock each uses and the rubber band, by game state."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(60)
        cls.tables = sim10.Tables.build(cls.matches, min_records=20)

    def test_the_cells_agree(self):
        for period, clock, lead in ((1, 240, 0), (2, 119.5, -3), (3, 1, 21), (4, 80, 9), (5, 200, -9)):
            self.assertEqual(sim10.cell_index(period, clock, lead),
                             int(sim10._cells_np(np.array([period]), np.array([float(clock)]),
                                                np.array([lead]))[0]))
        self.assertEqual(sim10.cell_index(4, 0.5, 30), sim10.N_CELLS - 1)

    def test_each_bin_lays_out_its_stopping_plays_first(self):
        t = self.tables
        self.assertTrue(((0 <= t.n_stop) & (t.n_stop <= t.count)).all())
        for key in range(0, sim10.N_KEYS, 13):
            a, n, m = t.start[key], t.count[key], t.n_stop[key]
            secs = t.seconds[a:a + n]
            self.assertTrue((secs[:m] <= sim10.STOP_SECONDS).all())
            self.assertTrue((secs[m:] > sim10.STOP_SECONDS).all())

    def test_the_fit_finds_a_planted_state_effect(self):
        import copy
        snaps = [r for rows in self.matches.values() for r in sim10.snap_records(rows)]
        target = sim10.cell_index(3, 150, 5)
        planted = []
        for r in snaps:
            r = dict(r)
            if sim10.cell_index(r["period"], r["clock"], r["margin"]) == target \
                    and r["seconds"] > sim10.STOP_SECONDS:
                r["seconds"] += 8                  # leaders here take 8 s longer
            planted.append(r)
        t = copy.deepcopy(self.tables)
        sim10.fit_play_calling(t, planted)
        sim10.fit_play_calling(self.tables, snaps)
        gained = t.sec_shift[sim10.RUNNING, target] - self.tables.sec_shift[sim10.RUNNING, target]
        n = sum(1 for r in snaps if sim10.cell_index(r["period"], r["clock"], r["margin"]) == target
                and r["seconds"] > sim10.STOP_SECONDS and r["clock"] >= sim10.UNCUT_SECONDS
                and not r["fresh"])
        self.assertGreater(n, 10)
        # the 8 s, shrunk toward 0 by SECONDS_PRIOR snaps (a fresh possession's snap is left out)
        self.assertAlmostEqual(gained, 8 * n / (n + sim10.SECONDS_PRIOR), places=6)
        self.assertLess(abs(t.sec_shift[sim10.STOP, target] - self.tables.sec_shift[sim10.STOP, target]), 1e-9)

    def test_the_band_pulls_in_every_quarter(self):
        self.assertTrue(v10.BAND_BY_QUARTER)
        shift = v10._band_shift(np.array([0.1, 0.2, 0.3]))
        quarter = np.arange(sim10.N_CELLS) // (sim10.CLOCK_CELLS * sim10.LEAD_CELLS)
        ahead = np.arange(sim10.N_CELLS) % sim10.LEAD_CELLS == 4          # ahead by 9+
        behind = np.arange(sim10.N_CELLS) % sim10.LEAD_CELLS == 0
        for q, pull in enumerate((0.1, 0.1, 0.2, 0.3)):             # the fourth included
            self.assertTrue(np.allclose(shift[(quarter == q) & ahead], -pull * 14 / v10.BAND_LEAD))
            self.assertTrue(np.allclose(shift[(quarter == q) & behind], pull * 14 / v10.BAND_LEAD))

    def test_the_band_is_fitted_by_quarter(self):
        import copy
        tables = copy.deepcopy(self.tables)
        grid = v10.PriorGrid.build(tables, n_paths=60)
        items = v10.in_game_states(self.matches, grid)
        pull, got, real = v10.fit_rubber_band(tables, items, rounds=3, n_paths=60)
        self.assertEqual(len(pull), 3)
        # each pull brings its own states' comeback to the real one
        for g, r in zip(got, real):
            self.assertAlmostEqual(g, r, delta=0.08)
        self.assertEqual({st.period for st, *_ in items} >= {1, 2, 3, 4}, True)

    def test_distinct_streams_keep_their_luck_between_calls(self):
        st = sim10.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim10.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 14, 10
        a = sim10.simulate(self.tables, st, 200, np.random.default_rng(1), seed=5, distinct=True)
        b = sim10.simulate(self.tables, st, 200, np.random.default_rng(2), seed=5, distinct=True)
        self.assertTrue(np.array_equal(a[0], b[0]))               # the same luck, call to call
        self.assertFalse(np.array_equal(a[0][0], a[0][1]))        # its own for each state

    def test_the_fourth_quarter_gets_the_items_switched_on_for_it(self):
        q4 = np.arange(sim10.N_CELLS) // (sim10.CLOCK_CELLS * sim10.LEAD_CELLS) == 3
        old = sim10.PLAY_CALLING_Q4
        try:
            sim10.PLAY_CALLING_Q4 = {"stop"}
            self.assertTrue(sim10._quarter_mask("stop")[q4].all())
            self.assertFalse(sim10._quarter_mask("band")[q4].any())
            self.assertTrue(sim10._quarter_mask("band")[~q4].all())
            self.assertFalse(sim10._quarter_mask()[q4].any())
            self.assertTrue((v10._band_shift(np.array([0.3, 0.3]))[q4] == 0).all())
            sim10.PLAY_CALLING_Q4 = set()
            t = sim10.Tables.build(self.matches, min_records=20)
            self.assertTrue((t.stop_shift[q4] == 0).all() and (t.sec_shift[:, q4] == 0).all()
                            and (t.eff_shift[q4] == 0).all())
            self.assertTrue(np.abs(t.sec_shift[:, ~q4]).sum() > 0)
        finally:
            sim10.PLAY_CALLING_Q4 = old

    def test_red_zone_cells_split_quarter_lead_and_the_ten(self):
        self.assertEqual(sim10.rz_index(1, -20, 70), 0)
        self.assertEqual(sim10.rz_index(1, -20, 95), 1)
        self.assertEqual(sim10.rz_index(4, 20, 95), sim10.N_RZ - 1)
        self.assertEqual(list(sim10.rz_index(np.array([3, 5]), np.array([0, 0]), np.array([80, 80]))),
                         [(2 * sim10.LEAD_CELLS + 2) * 2, (3 * sim10.LEAD_CELLS + 2) * 2])

    def test_held_touchdowns_turn_into_field_goals_not_fewer_first_downs(self):
        import copy
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 3, 200.0, sim10.SCRIM
        st.team[:], st.y[:], st.dist[:], st.home[:], st.away[:] = 0, 80, 10, 10, 10
        runs = {}
        for hold in (0.0, 0.8):
            t = copy.deepcopy(self.tables)
            t.rz_hold = np.full(sim10.N_RZ, hold)
            stats = {}
            h, _ = sim10.simulate(t, st, 4000, np.random.default_rng(1), seed=5, one_drive=True,
                                 stats=stats)
            runs[hold] = (np.mean(h[0] - 10 >= 6), np.mean(h[0] - 10 == 3), stats)
        self.assertLess(runs[0.8][0], runs[0.0][0] - 0.05)
        self.assertGreater(runs[0.8][1], runs[0.0][1])
        self.assertGreater(runs[0.8][2]["snaps"], runs[0.0][2]["snaps"])

    def test_the_red_zone_fit_round_trips(self):
        old = sim10.RED_ZONE_FIT
        try:
            sim10.RED_ZONE_FIT = True
            t = sim10.Tables.build(self.matches, min_records=20)
        finally:
            sim10.RED_ZONE_FIT = old
        self.assertEqual(t.rz_hold.shape, (sim10.N_RZ,))
        self.assertTrue(((t.rz_hold >= 0) & (t.rz_hold <= sim10.HOLD_GRID[-1])).all())
        with tempfile.TemporaryDirectory() as d:
            t.save(os.path.join(d, "t.npz"))
            back = sim10.Tables.load(os.path.join(d, "t.npz"))
        self.assertTrue(np.array_equal(back.rz_hold, t.rz_hold))

    def _q3_leader_with_ball(self, n=3000, **shift):
        import copy
        t = copy.deepcopy(self.tables)
        for name, (cells, value) in shift.items():
            arr = getattr(t, name)
            if arr.ndim == 2:
                arr[sim10.RUNNING, cells] += value
            else:
                arr[cells] += value
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 3, 200.0, sim10.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 30, 17, 10
        h, a = sim10.simulate(t, st, n, np.random.default_rng(1), seed=5)
        return (h + a).mean(), (h - a).mean()

    def test_a_leader_milking_the_clock_leaves_fewer_points(self):
        lead = [c for c in range(sim10.N_CELLS) if c % sim10.LEAD_CELLS >= 3 and c // (sim10.CLOCK_CELLS * sim10.LEAD_CELLS) >= 2]
        plain, _ = self._q3_leader_with_ball()
        milked, _ = self._q3_leader_with_ball(sec_shift=(lead, 15.0))
        self.assertLess(milked, plain - 0.5)

    def test_the_rubber_band_pulls_a_lead_back(self):
        trail = [c for c in range(sim10.N_CELLS) if c % sim10.LEAD_CELLS <= 1]
        lead = [c for c in range(sim10.N_CELLS) if c % sim10.LEAD_CELLS >= 3]
        _, plain = self._q3_leader_with_ball()
        _, banded = self._q3_leader_with_ball(eff_shift=(trail, 0.4))
        self.assertLess(banded, plain - 0.3)
        _, eased = self._q3_leader_with_ball(eff_shift=(lead, -0.4))
        self.assertLess(eased, plain - 0.3)


class TestBackedUp(unittest.TestCase):
    """Inside the own 10 each snap draws a safety, a defensive touchdown, a
    turnover or a touchdown from its yard line's own rates; after a safety
    the scorer gets two points and the free kick."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim10.Tables.build(_matches(60), min_records=20)

    def _table(self, **at_the_one):
        import copy
        t = copy.deepcopy(self.tables)
        t.backed = np.zeros((sim10.BACKED_UP + 1, len(sim10.BACKED_OUTCOMES)))
        for name, p in at_the_one.items():
            t.backed[1, sim10.BACKED_OUTCOMES.index(name)] = p
        t.backed_return = None
        return t

    @staticmethod
    def _on_the_one(n=1):
        st = sim10.Start(n)
        st.period[:], st.clock[:], st.phase[:] = 2, 200.0, sim10.SCRIM
        st.team[:], st.y[:], st.down[:], st.dist[:] = 0, 1, 1, 10
        return st

    def test_each_yard_line_keeps_its_own_rate_pulled_toward_the_curve(self):
        records = ([(1, sim10.B_SAFETY, None)] * 12 + [(1, None, None)] * 188
                   + [(5, sim10.B_SAFETY, None)] * 2 + [(5, None, None)] * 198
                   + [(3, sim10.B_TURNOVER, -10)] * 30 + [(3, None, None)] * 170)
        hazard, returns = sim10.fit_backed_up(records)
        self.assertAlmostEqual(hazard[1, sim10.B_SAFETY], 0.06, delta=0.012)
        self.assertAlmostEqual(hazard[5, sim10.B_SAFETY], 0.01, delta=0.008)
        # the 2, never seen, sits on the curve between them
        self.assertGreater(hazard[1, sim10.B_SAFETY], hazard[2, sim10.B_SAFETY])
        self.assertGreater(hazard[2, sim10.B_SAFETY], hazard[5, sim10.B_SAFETY])
        self.assertAlmostEqual(hazard[3, sim10.B_TURNOVER], 0.15, delta=0.03)
        self.assertEqual(list(returns), [-10] * 30)

    def test_the_real_safety_rows_are_found(self):
        rows = [dict(play_kind="SCRIMMAGE", down="2", field_position="2", period="2", offense="TEAM_A",
                     play_messages=""),
                dict(play_kind="PUNT", down="1", field_position="20", period="2", offense="TEAM_A",
                     play_messages="SAFETY_TEAM_B"),
                dict(play_kind="SCRIMMAGE", down="1", field_position="40", period="2", offense="TEAM_B",
                     play_messages="")]
        self.assertEqual(sim10.backed_up_snaps(rows), [(2, sim10.B_SAFETY, None)])

    def test_a_snap_on_the_one_comes_to_its_yard_lines_outcomes(self):
        t = self._table(safety=0.2, def_td=0.1, td=0.1)
        n = 20000
        h, a = sim10.simulate(t, self._on_the_one(), n, np.random.default_rng(1), max_steps=1, seed=9)
        self.assertAlmostEqual(float((a == 2).mean()), 0.2, delta=0.015)
        self.assertAlmostEqual(float((a == 6).mean()), 0.1, delta=0.01)
        self.assertAlmostEqual(float((h == 6).mean()), 0.1, delta=0.01)
        # and nothing else scores: the other snaps stay in the field of play
        self.assertAlmostEqual(float(((h == 0) & (a == 0)).mean()), 0.6, delta=0.015)

    def test_after_a_safety_the_scorer_receives_the_free_kick(self):
        t = self._table(safety=1.0)
        t.safety_kick = np.array([99], dtype=np.int32)     # the free kick lands on the kicker's 1
        stats = {}
        h, a = sim10.simulate(t, self._on_the_one(), 2000, np.random.default_rng(1), max_steps=6,
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
        self.assertIsNone(sim10.Tables.load(path).backed)
        t = self._table(safety=0.3)
        t.save(path)
        self.assertAlmostEqual(float(sim10.Tables.load(path).backed[1, sim10.B_SAFETY]), 0.3)


class TestStrengthSpread(unittest.TestCase):
    """Each simulated game's offenses drawn around the prior: a shared game
    draw that moves the total, each player's own form, the same draw for
    path k of every snapshot of a match, and the fit of both off results."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(40)
        cls.tables = sim10.Tables.build(cls.matches, min_records=20)

    def _kickoff(self, n=1, own=0.0, game=0.0):
        st = sim10.Start(n)
        st.team[:] = 0
        st.kicks_second_half[:] = 1
        st.strength[:] = own
        st.strength_game[:] = game
        return st

    def _spread(self, **kw):
        h, a = sim10.simulate(self.tables, self._kickoff(**kw), 4000, np.random.default_rng(1), seed=5)
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
        h, a = sim10.simulate(self.tables, st, 500, np.random.default_rng(1), seed=5)
        self.assertTrue(np.array_equal(h[0], h[1]) and np.array_equal(a[0], a[1]))
        h2, _ = sim10.simulate(self.tables, st, 500, np.random.default_rng(9), seed=5)
        self.assertTrue(np.array_equal(h, h2))

    def test_the_draw_keeps_the_expected_points(self):
        import copy
        tables = copy.deepcopy(self.tables)
        _, _, slope = v10.fixed_strength_spread(tables, n_paths=600)
        tables.strength_theta, tables.strength_slope = v10.FORM_GRID.copy(), slope
        tables.strength_game = 0.04
        theta, own, game = sim10.strength_draw(tables, [0.0, 0.0], [0.03, 0.0])
        self.assertTrue(own[0] > 0 and own[1] == 0 and game[0] > 0)
        st = self._kickoff()
        fixed = sim10.simulate(tables, st, 6000, np.random.default_rng(2), common=False)[0].mean()
        st.theta[0], st.strength[0], st.strength_game[0] = theta, own, game
        drawn = sim10.simulate(tables, st, 6000, np.random.default_rng(2), common=False)[0].mean()
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
        game, league, form = v10.fit_form(sides, np.zeros(3), np.zeros(4))
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
        t = sim10.Tables()
        theta, own, game = sim10.strength_draw(t, [0.1, -0.1], [0.05, 0.05])
        self.assertTrue(np.allclose(theta, [0.1, -0.1]) and not own.any() and not game.any())


class TestLateGame(unittest.TestCase):
    """v10: a trailing side's late 4th downs from a fitted table, and big leads as situations of
    their own."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(60)
        cls.tables = sim10.Tables.build(cls.matches, min_records=20)

    def _fourth(self, behind):
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 4, 60.0, sim10.SCRIM
        st.team[:], st.down[:], st.dist[:], st.y[:] = 0, 4, 8, 80
        st.home[:], st.away[:] = 10, 10 + behind
        return st

    def test_the_table_sets_what_a_trailing_side_does(self):
        import copy
        t = copy.deepcopy(self.tables)
        t.late_fourth = np.zeros_like(t.late_fourth)
        t.late_fourth[..., 1] = 1.0
        h, _ = sim10.simulate(t, self._fourth(10), 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertGreater(float((h == 13).mean()), 0.8)
        t.late_fourth[..., 1], t.late_fourth[..., 0] = 0.0, 1.0
        h, _ = sim10.simulate(t, self._fourth(10), 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertEqual(float((h == 13).mean()), 0.0)

    def test_the_fitted_table_is_shaped_and_sums_to_one(self):
        lf = self.tables.late_fourth
        self.assertEqual(lf.shape, (2, len(sim10.LATE_DEFICITS) + 1, len(sim10.KICK_RANGES) + 1, 3))
        self.assertTrue(np.allclose(lf.sum(3), 1.0))

    def test_big_leads_have_their_own_situations(self):
        self.assertEqual(sim10.mode_of(3, 100.0, 5), 3)
        self.assertEqual(sim10.mode_of(3, 100.0, 12), 8)
        self.assertEqual(sim10.mode_of(4, 60.0, 12), 9)
        self.assertEqual(sim10.mode_of(4, 60.0, 12, None), 4)
        modes = sim10._modes_np(np.array([3, 4, 4]), np.array([100.0, 60.0, 60.0]), np.array([12, 12, 3]), 9)
        self.assertEqual(list(modes), [8, 9, 4])
        self.assertEqual(self.tables.big_lead, sim10.BIG_LEAD)


class TestClockToTheEnd(unittest.TestCase):
    """v10: the clock is fitted to the end of each quarter, and the trailing team's timeouts count."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches()
        cls.tables = sim10.Tables.build(cls.matches, min_records=20)

    def _row(self, message, period, clock, down=1, kind="SCRIMMAGE", offense="TEAM_A"):
        return {"message": str(message), "period": str(period), "clock_seconds": str(clock),
                "play_kind": kind, "down": str(down), "distance": "10", "field_position": "40",
                "offense": offense, "score_p1": "7", "score_p2": "3", "play_messages": ""}

    def test_a_snap_that_runs_the_quarter_out_is_kept_as_censored(self):
        rows = [self._row(1, 2, 90), self._row(2, 2, 30), self._row(3, 3, 240, kind="KICKOFF"),
                self._row(4, 3, 200), self._row(5, 3, 150), self._row(6, 4, 20)]
        ends = sim10.clock_end_records(rows)
        self.assertEqual([(e["period"], e["seconds"]) for e in ends], [(2, 30.0), (3, 150.0), (4, 20.0)])
        self.assertEqual(sim10.clock_end_records([self._row(1, 1, 10, down=4), self._row(2, 2, 240)]), [])

    def test_quarters_that_run_out_teach_the_last_slice_to_stop_the_clock(self):
        t = self.tables
        key = int(np.argmax(np.minimum(t.n_stop, t.count - t.n_stop)))
        rec = lambda clock, secs: dict(period=2, clock=clock, margin=0, key=key, seconds=secs, fresh=False)
        cell = sim10.cell_index(2, 20.0, 0)
        run = next(s for s in t.seconds[t.start[key] + t.n_stop[key]:t.start[key] + t.count[key]] if s > 20)
        stop = [rec(20.0, 5.0)] * 40
        ran_out = [rec(20.0, 20.0)] * 40            # censored: took all 20 seconds or more
        self.assertGreater(run, 20)
        dropped = sim10._fit_stop_censored(t, stop, [])[cell]
        with_ends = sim10._fit_stop_censored(t, stop, ran_out)[cell]
        as_runs = sim10._fit_stop_censored(t, stop + [rec(20.0, 18.0)] * 40, [])[cell]
        # a snap that took all 20 seconds left was a clock-running play: dropping it overstates
        # the clock-stopped share
        self.assertLess(with_ends, dropped)
        self.assertAlmostEqual(with_ends, as_runs, delta=0.2)

    def test_real_timeouts_are_placed_before_their_snaps(self):
        from .. import playover
        rows = [self._row(10, 2, 100), self._row(20, 2, 70), self._row(30, 2, 40), self._row(40, 3, 240)]
        by = {"M": rows}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "timeouts.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["match_code", "message", "period", "caller", "prev_play"])
                w.writerow(["M", 12, 2, "TEAM_B", "in play"])
                w.writerow(["M", 33, 2, "TEAM_A", "incomplete"])
            self.assertEqual(playover.annotate_timeouts(by, path), 2)
        self.assertEqual([r["timeout_after"] for r in rows], ["TEAM_B", "", "TEAM_A", ""])
        self.assertEqual([(r["timeouts_used_a"], r["timeouts_used_b"]) for r in rows],
                         [(0, 0), (0, 1), (0, 1), (0, 0)])        # the new half starts afresh
        rec = dict(period=2, offense="TEAM_A")
        sim10._timeouts_at(rec, rows[1])
        self.assertEqual((rec["timeouts_left"], rec.get("timeout_role")), ((3, 2), None))
        sim10._timeouts_at(rec, rows[0])
        self.assertEqual(rec["timeout_role"], sim10.DEFENCE)
        rec = dict(period=2, offense="TEAM_A")
        sim10._timeouts_at(rec, rows[2])                          # after an incompletion: not needed
        self.assertIsNone(rec.get("timeout_role"))

    def test_call_rates_come_from_the_running_clocks_real_sides_stopped(self):
        base = dict(period=4, clock=50.0, margin=3, fresh=False, timeouts_left=(3, 3))
        snaps = ([dict(base, seconds=35.0)] * 60 + [dict(base, seconds=4.0, timeout_role=sim10.DEFENCE)] * 40
                 + [dict(base, seconds=5.0)] * 50)               # stopped by themselves: not counted
        p = sim10.fit_timeout_calls(snaps)
        got = p[sim10.call_index(4, 50.0, -3, sim10.DEFENCE)]
        self.assertAlmostEqual(float(got), 0.4, delta=0.05)
        self.assertLess(float(p[sim10.call_index(4, 50.0, 3, sim10.OFFENCE)]), 0.05)
        self.assertEqual(float(sim10.fit_timeout_calls([dict(base, timeouts_left=None, seconds=30.0)]).sum()), 0.0)

    def test_without_timeouts_known_the_build_uses_the_real_rates(self):
        t = sim10.Tables.build(self.matches, min_records=20)
        self.assertTrue(np.array_equal(t.call_p, np.array(sim10.DEFAULT_CALL_P)))
        self.assertFalse(t.call_fitted)
        d = np.array(sim10.DEFAULT_CALL_P)
        self.assertGreater(d[0, sim10.OFFENCE, 2, -1], 0.5)      # Q2, last 40 s: the side with the ball
        self.assertGreater(d[1, sim10.DEFENCE, 1, 3], 0.4)       # Q4, trailing without it, 2:00-1:20

    def test_overtime_fits_its_own_clock_stops(self):
        t = self.tables
        key = int(np.argmax(np.minimum(t.n_stop, t.count - t.n_stop)))
        rec = lambda secs: dict(period=5, clock=100.0, margin=0, key=key, seconds=secs)
        many_stops = sim10.fit_ot_stop(t, [rec(5.0)] * 80 + [rec(30.0)] * 20, np.zeros(sim10.N_CELLS))
        few_stops = sim10.fit_ot_stop(t, [rec(5.0)] * 20 + [rec(30.0)] * 80, np.zeros(sim10.N_CELLS))
        sl = int(sim10._slice_of(100.0))
        self.assertGreater(many_stops[sl], few_stops[sl])
        self.assertEqual(float(sim10.fit_ot_stop(t, [], np.zeros(sim10.N_CELLS)).sum()), 0.0)

    def test_overtime_has_two_timeouts_a_side_called_less_often(self):
        rec = dict(period=5, offense="TEAM_A")
        sim10._timeouts_at(rec, {"timeouts_used_a": 1, "timeouts_used_b": 0, "offense": "TEAM_A"})
        self.assertEqual(rec["timeouts_left"], (1, 2))
        base = dict(period=5, clock=50.0, margin=0, fresh=False, timeouts_left=(2, 2))
        p = np.full(sim10.N_CALL, 0.5)
        none = sim10.fit_ot_call_scale([dict(base, seconds=35.0)] * 200, p)
        some = sim10.fit_ot_call_scale([dict(base, seconds=35.0)] * 180
                                      + [dict(base, seconds=4.0, timeout_role=sim10.DEFENCE)] * 20, p)
        self.assertLess(none, 0.02)
        self.assertAlmostEqual(some, 20 / 200, delta=0.02)
        self.assertEqual(sim10.fit_ot_call_scale([], p), sim10.OT_CALL_DEFAULT)
        t = self.tables
        saved = t.call_p.copy(), t.ot_call_scale
        try:
            t.call_p[:] = 1.0
            t.ot_call_scale = 1.0
            st = sim10.Start(1)
            st.period[:], st.clock[:], st.phase[:] = 5, 200.0, sim10.SCRIM
            st.team[:], st.y[:], st.down[:], st.home[:], st.away[:] = 0, 30, 1, 17, 17
            stats = {}
            sim10.simulate(t, st, 200, np.random.default_rng(1), seed=4, stats=stats)
            self.assertGreater(stats.get("timeouts", 0), 0)
            self.assertLessEqual(stats.get("timeouts", 0), 200 * 2 * 2 * sim10.MAX_OT)
        finally:
            t.call_p, t.ot_call_scale = saved

    def test_overtime_carries_on_until_both_sides_have_had_the_ball(self):
        t = self.tables
        st = sim10.Start(2)
        # a side leads as the first overtime period runs out, but the side behind has yet to have
        # the ball / the score is level: play carries on into the next period, same ball and spot
        st.period[:], st.clock[:], st.phase[:] = 5, 3.0, sim10.SCRIM
        st.team[:], st.y[:], st.down[:], st.home[:], st.away[:] = [1, 0], 40, 1, [20, 17], [17, 17]
        stats = {}
        sim10.simulate(t, st, 200, np.random.default_rng(1), seed=4, stats=stats)
        self.assertGreater(stats.get("ot_carried", 0), 0)
        saved = sim10.OT_CARRY
        try:
            sim10.OT_CARRY = False
            stats = {}
            sim10.simulate(t, st, 200, np.random.default_rng(1), seed=4, stats=stats)
            self.assertEqual(stats.get("ot_carried", 0), 0)
        finally:
            sim10.OT_CARRY = saved
        h, a = sim10.simulate(t, st, 400, np.random.default_rng(1), seed=4)
        self.assertLess(float((h == a).mean()), 0.02)            # overtime almost never ends level

    def test_a_play_stopped_by_a_timeout_gets_a_running_play_s_time(self):
        base = dict(key=7, mode=1, down=2)
        snaps = [dict(base, seconds=33.0)] * 10 + [dict(base, seconds=3.0, timeout_role=sim10.OFFENCE)]
        sim10.natural_seconds(snaps)
        self.assertEqual((snaps[-1]["seconds"], snaps[-1]["seconds_real"]), (33.0, 3.0))

    def test_a_kneel_is_a_leader_s_small_loss_in_the_fourth_quarter(self):
        rec = dict(period=4, margin=3, down=1, kind=sim10.GAIN, gain=-1)
        self.assertTrue(sim10.is_kneel(rec))
        self.assertFalse(sim10.is_kneel(dict(rec, margin=-3)))
        self.assertFalse(sim10.is_kneel(dict(rec, period=3)))
        self.assertFalse(sim10.is_kneel(dict(rec, gain=4)))
        self.assertFalse(sim10.is_kneel(dict(rec, down=4)))
        big, d, cb = sim10.kneel_index(np.array([3, 12]), np.array([1, 3]), np.array([230.0, 10.0]))
        self.assertEqual((list(big), list(d), list(cb)), ([0, 1], [0, 2], [0, 5]))

    def test_the_leader_kneels_as_often_as_the_table_says(self):
        t = self.tables
        saved = t.kneel_p.copy(), t.call_p.copy()
        try:
            t.call_p[:] = 0.0
            t.kneel_p[:] = 1.0
            stats = {}
            sim10.simulate(t, self._late_lead(200.0), 200, np.random.default_rng(1), seed=4,
                          stats=stats, max_steps=1)
            self.assertEqual(stats.get("kneel_plays", 0), 200)
            t.kneel_p[:] = 0.0
            stats = {}
            sim10.simulate(t, self._late_lead(200.0), 200, np.random.default_rng(1), seed=4,
                          stats=stats, max_steps=1)
            self.assertEqual(stats.get("kneel_plays", 0), 0)
        finally:
            t.kneel_p, t.call_p = saved

    def _late_lead(self, clock, down=1, timeouts=-1):
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 4, clock, sim10.SCRIM
        st.team[:], st.y[:], st.down[:], st.home[:], st.away[:] = 0, 40, down, 21, 17
        st.timeouts[:, 1] = timeouts
        return st

    def test_the_leader_kneels_it_out_only_when_the_timeouts_cannot_stop_it(self):
        t = self.tables
        t.call_p[:] = 0.6
        for left, clock, ends in ((0, 150.0, True), (3, 150.0, False), (3, 35.0, True), (2, 90.0, False)):
            stats = {}
            h, a = sim10.simulate(t, self._late_lead(clock, timeouts=left), 200, np.random.default_rng(1),
                                 seed=4, stats=stats, max_steps=1)
            self.assertEqual(stats.get("kneel", 0) == 200, ends, (left, clock))

    def test_a_side_spends_only_the_timeouts_it_has(self):
        t = self.tables
        t.call_p[:] = 1.0
        stats = {}
        sim10.simulate(t, self._late_lead(115.0, timeouts=2), 300, np.random.default_rng(1), seed=4,
                      stats=stats)
        self.assertGreater(stats.get("timeouts", 0), 0)
        self.assertLessEqual(stats.get("timeouts", 0), (2 + 3) * 300)
        t.call_p[:] = 0.0
        stats = {}
        sim10.simulate(t, self._late_lead(115.0, timeouts=2), 300, np.random.default_rng(1), seed=4,
                      stats=stats)
        self.assertEqual(stats.get("timeouts", 0), 0)

    def test_timeouts_give_the_trailing_team_more_of_the_clock(self):
        t = self.tables
        pts = {}
        for p in (0.0, 1.0):
            t.call_p[:] = 0.0
            t.call_p[1, sim10.DEFENCE] = p
            h, a = sim10.simulate(t, self._late_lead(118.0, down=2, timeouts=3), 2000,
                                 np.random.default_rng(1), seed=4)
            pts[p] = float((h + a).mean())
        t.call_p[:] = 0.0
        t.call_p[1, sim10.DEFENCE] = 1e-9                         # on, but calls nothing
        h, a = sim10.simulate(t, self._late_lead(118.0, down=2, timeouts=3), 2000,
                             np.random.default_rng(1), seed=4)
        self.assertGreater(pts[1.0], float((h + a).mean()))

    def test_the_export_s_timeouts_become_each_side_s_timeouts_left(self):
        row = {"timeouts_used_a": "1", "timeouts_used_b": "3", "team_a_side": "away", "period": "4"}
        self.assertEqual(v10.timeouts_left(row), (0, 2))
        self.assertEqual(v10.timeouts_left(dict(row, team_a_side="home")), (2, 0))
        self.assertEqual(v10.timeouts_left(dict(row, period="5")), (0, 1))      # two a side in overtime
        self.assertEqual(v10.timeouts_left({"period": "4", "team_a_side": "home"}), (-1, -1))
        state = v10.FreshState(**{f.name: None for f in v10.fields(v10.GameState)}, timeouts=(2, 0))
        self.assertEqual(state.timeouts, (2, 0))

    def test_in_q2_the_side_with_the_ball_stops_it_at_the_end(self):
        t = self.tables
        t.call_p[:] = 0.0
        t.call_p[0, sim10.OFFENCE, :, -1] = 1.0                   # Q2, last 40 seconds, with the ball
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 2, 30.0, sim10.SCRIM
        st.team[:], st.y[:], st.down[:], st.home[:], st.away[:] = 0, 30, 1, 7, 7
        stats = {}
        sim10.simulate(t, st, 300, np.random.default_rng(1), seed=4, stats=stats)
        self.assertGreater(stats.get("timeouts", 0), 0)
        st.timeouts[:] = [0, 3]
        stats = {}
        sim10.simulate(t, st, 300, np.random.default_rng(1), seed=4, stats=stats, max_steps=1)
        self.assertEqual(stats.get("timeouts", 0), 0)

    def test_the_tables_keep_the_timeouts(self):
        t = self.tables
        t.call_p = np.random.default_rng(2).random(sim10.N_CALL)
        with tempfile.TemporaryDirectory() as d:
            t.save(os.path.join(d, "t.npz"))
            u = sim10.Tables.load(os.path.join(d, "t.npz"))
        self.assertTrue(np.allclose(u.call_p, t.call_p))


class TestV9Changes(unittest.TestCase):
    """v10: recent weeks weigh more in the touchdown and clock fits; timeout plays take their real
    time; the clock's state shifts are fitted on plays nobody stopped."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches()

    def test_with_its_own_switches_off_v10_plays_as_v8(self):
        from .. import sim8
        saved = (sim10.NATURAL_SHIFT, sim10.TO_PLAY_SECONDS)
        sim10.NATURAL_SHIFT = sim10.TO_PLAY_SECONDS = False
        try:
            t9 = sim10.Tables.build(self.matches, min_records=20)
            t8 = sim8.Tables.build(self.matches, min_records=20)
            st = sim10.Start(4)
            st.period[:], st.clock[:], st.phase[:] = [2, 4, 4, 4], [30.0, 150.0, 50.0, 25.0], sim10.SCRIM
            st.team[:], st.y[:], st.home[:], st.away[:] = 0, [50, 50, 80, 75], [3, 17, 10, 10], [10, 10, 16, 13]
            st.down[:] = [1, 2, 1, 4]
            a = sim10.simulate(t9, st, 300, np.random.default_rng(1), seed=9)
            b = sim8.simulate(t8, st, 300, np.random.default_rng(1), seed=9)
        finally:
            sim10.NATURAL_SHIFT, sim10.TO_PLAY_SECONDS = saved
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_matches_weigh_less_with_age(self):
        recs = [dict(match="AF001100926"), dict(match="AF001030926"), dict(match="X")]
        sim10.age_weights(recs, dt.date(2026, 9, 10), 7.0)
        self.assertEqual([round(r["w"], 3) for r in recs], [1.0, 0.5, 1.0])
        sim10.age_weights(recs, None, 7.0)
        self.assertEqual([r["w"] for r in recs], [1.0, 1.0, 1.0])
        self.assertEqual(sim10.match_date("AF001010926"), dt.date(2026, 9, 1))

    def test_recent_weeks_pull_the_clock_fit(self):
        t = sim10.Tables.build(self.matches, min_records=20)
        key = int(np.argmax(np.minimum(t.n_stop, t.count - t.n_stop)))
        base = dict(period=4, clock=150.0, margin=3, key=key, fresh=False)
        old = [dict(base, seconds=5.0, w=0.1)] * 50
        new = [dict(base, seconds=35.0, w=1.0)] * 50
        cell = sim10.cell_index(4, 150.0, 3)
        shift = sim10._fit_stop_censored(t, old + new, [])[cell]
        even = sim10._fit_stop_censored(t, [dict(r, w=1.0) for r in old + new], [])[cell]
        self.assertLess(shift, even)                       # the recent running plays count for more

    def test_the_touchdown_fit_weighs_recent_matches_more(self):
        from .. import v10 as v10mod
        quick = v10mod.PriorGrid.build(sim10.Tables.build(self.matches, min_records=20), n_paths=200)
        some = {code: [dict(r, file_time="2026-09-01 10:00:00") for r in rows]
                for code, rows in list(self.matches.items())[:3]}
        states = v10mod.settle_states(some, quick, as_of=dt.date(2026, 9, 15), half_life=7.0)
        self.assertTrue(states and all(len(s) == 5 for s in states))
        self.assertTrue(all(abs(s[4] - 0.25) < 1e-9 for s in states))     # two half-lives old
        flat = v10mod.settle_states(some, quick)
        self.assertTrue(all(s[4] == 1.0 for s in flat))

    def test_timeout_plays_take_their_real_time_by_caller(self):
        snaps = ([dict(period=4, margin=7, timeout_role=sim10.DEFENCE, seconds=12.0)] * 40
                 + [dict(period=4, margin=-7, timeout_role=sim10.OFFENCE, seconds=5.0)] * 40
                 + [dict(period=4, margin=0, seconds=30.0)] * 40)
        q = sim10.fit_timeout_seconds(snaps)
        self.assertEqual(q[1, sim10.DEFENCE, 0, 10], 12.0)          # the side behind, on defence
        self.assertEqual(q[1, sim10.OFFENCE, 0, 10], 5.0)
        self.assertTrue(np.isnan(q[0]).all())
        self.assertEqual(list(sim10.caller_side([-3, 0, 4])), [0, 1, 2])

    def test_a_called_timeout_takes_the_fitted_time(self):
        t = sim10.Tables.build(self.matches, min_records=20)
        t.call_p[:] = 0.0
        t.call_p[1, sim10.DEFENCE] = 1.0
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 4, 100.0, sim10.SCRIM
        st.team[:], st.y[:], st.down[:], st.home[:], st.away[:] = 0, 40, 1, 21, 17
        st.timeouts[:, 1] = 3
        used = {}
        for secs in (3.0, 20.0):
            t.to_secs[:] = secs
            stats = {}
            sim10.simulate(t, st, 300, np.random.default_rng(1), seed=4, stats=stats, max_steps=1)
            used[secs] = stats.get("clock_used", 0) / max(1, stats.get("snaps", 1))
        self.assertGreater(used[20.0], used[3.0])


class TestV9(unittest.TestCase):
    """v10: a fresh possession's first snap is played with the clock stopped, the fourth quarter
    has situations of its own, a side a touchdown behind late goes for it, and would-be
    touchdowns are held back where the simulation scores too often."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(60)
        cls.tables = sim10.Tables.build(cls.matches, min_records=20)

    def _snap(self, n=1, fresh=False, period=3, clock=150.0, y=40, down=1, dist=10, home=10, away=10):
        st = sim10.Start(n)
        st.period[:], st.clock[:], st.phase[:] = period, clock, sim10.SCRIM
        st.team[:], st.y[:], st.down[:], st.dist[:] = 0, y, down, dist
        st.home[:], st.away[:], st.fresh[:] = home, away, fresh
        return st

    def test_a_fresh_possession_is_marked_on_its_row(self):
        row = lambda kind, messages="": dict(play_kind=kind, play_messages=messages)
        for kind in ("KICKOFF", "PUNT", "TURNOVER_ON_DOWNS", "FIELD_GOAL"):
            self.assertTrue(sim10.is_fresh(row(kind)))
        self.assertTrue(sim10.is_fresh(row("SCRIMMAGE", "PASS_TEAM_A|POSSESSION_TEAM_B")))
        self.assertFalse(sim10.is_fresh(row("SCRIMMAGE", "PASS_TEAM_A")))
        rows = next(iter(self.matches.values()))
        recs = sim10.snap_records(rows)
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
            sim10.simulate(t, self._snap(200, fresh), 5, np.random.default_rng(1), max_steps=1, stats=stats)
            share = stats.get("stop_calls", 0) / stats["snaps"]
            if fresh:
                self.assertGreater(share, want)
            else:
                self.assertEqual(share, want)

    def test_kicks_turnovers_and_quarter_starts_leave_the_next_snap_fresh(self):
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 3, 200.0, sim10.KICK
        stats = {}
        sim10.simulate(self.tables, st, 300, np.random.default_rng(1), max_steps=2, stats=stats)
        self.assertEqual(stats["fresh_snaps"], stats["snaps"])
        stats = {}
        sim10.simulate(self.tables, self._snap(1), 300, np.random.default_rng(1), max_steps=1, stats=stats)
        self.assertEqual(stats.get("fresh_snaps", 0), 0)
        stats = {}
        sim10.simulate(self.tables, self._snap(1, clock=0.5), 300, np.random.default_rng(1), max_steps=3,
                      stats=stats)
        self.assertGreater(stats["fresh_snaps"], 0)

    def test_the_clock_is_fitted_on_snaps_that_follow_a_scrimmage_play(self):
        import copy
        snaps = [r for rows in self.matches.values() for r in sim10.snap_records(rows)]
        planted = [dict(r, seconds=59.0) if r["fresh"] else r for r in snaps]
        a, b = copy.deepcopy(self.tables), copy.deepcopy(self.tables)
        sim10.fit_play_calling(a, snaps)
        sim10.fit_play_calling(b, planted)
        self.assertTrue(np.allclose(a.sec_shift, b.sec_shift))
        self.assertTrue(np.allclose(a.stop_shift, b.stop_shift))

    def test_the_fourth_quarter_has_situations_of_its_own(self):
        self.assertEqual([sim10.mode_of(4, 200.0, m, 9, True) for m in (3, 12, 0, -3, -12)],
                         [10, 11, 12, 13, 14])
        self.assertEqual([sim10.mode_of(4, 200.0, m, 9) for m in (3, 12, 0, -3, -12)], [3, 8, 5, 5, 5])
        self.assertEqual(sim10.mode_of(4, 100.0, 3, 9, True), 4)       # its last two minutes as before
        self.assertEqual(sim10.mode_of(4, 100.0, -3, 9, True), 6)
        self.assertEqual(sim10.mode_of(4, 100.0, -12, 9, True), 15)    # but two scores behind is its own
        self.assertEqual(sim10.mode_of(3, 200.0, 3, 9, True), 3)
        self.assertEqual(sim10.mode_of(3, 200.0, -12, 9, True), 5)
        period, clock = np.array([4, 4, 4, 4, 4, 4, 3]), np.array([200.0, 200, 200, 200, 100, 100, 200])
        margin = np.array([3, 12, 0, -12, -3, -12, -12])
        self.assertEqual(list(sim10._modes_np(period, clock, margin, 9, True)), [10, 11, 12, 14, 6, 15, 5])
        self.assertTrue(self.tables.q4_modes)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            self.tables.save(path)
            back = sim10.Tables.load(path)
        self.assertTrue(back.q4_modes)
        self.assertTrue(np.array_equal(back.n_fresh, self.tables.n_fresh))

    def test_a_side_a_touchdown_behind_in_the_last_minute_goes_for_it(self):
        import copy
        t = copy.deepcopy(self.tables)
        t.late_fourth = np.zeros_like(t.late_fourth)
        t.late_fourth[..., 1] = 1.0                        # the table would kick
        st = self._snap(1, period=4, clock=50.0, y=80, down=4, dist=8, home=10, away=16)
        h, _ = sim10.simulate(t, st, 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertEqual(float((h == 13).mean()), 0.0)
        saved = sim10.GO_AHEAD
        sim10.GO_AHEAD = False
        try:
            h, _ = sim10.simulate(t, st, 400, np.random.default_rng(1), seed=3, max_steps=1)
        finally:
            sim10.GO_AHEAD = saved
        self.assertGreater(float((h == 13).mean()), 0.8)
        st = self._snap(1, period=4, clock=100.0, y=80, down=4, dist=8, home=10, away=16)
        h, _ = sim10.simulate(t, st, 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertGreater(float((h == 13).mean()), 0.8)     # with time left, the table decides

    def test_a_hold_redraws_would_be_touchdowns(self):
        import copy
        t = copy.deepcopy(self.tables)
        st = self._snap(1, y=95, dist=5)
        h0, _ = sim10.simulate(t, st, 2000, np.random.default_rng(1), seed=3, max_steps=1)
        t.td_hold[:] = 0.8
        h1, _ = sim10.simulate(t, st, 2000, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertLess(float((h1 == 16).mean()), 0.6 * float((h0 == 16).mean()))

    def test_the_settle_fit_holds_back_touchdowns_where_real_snaps_score_less(self):
        import copy
        t = copy.deepcopy(self.tables)
        grid = v10.PriorGrid.build(t, n_paths=60)
        items = v10.settle_states(self.matches, grid)
        self.assertTrue(items and all(0 <= c < sim10.N_SETTLE for c, *_ in items))
        cells = {}
        for c, *_ in items:
            cells[c] = cells.get(c, 0) + 1
        busy = max((c for c in cells if c % 4 >= 2), key=cells.get)     # inside the 30
        planted = [(c, f, th, False if c == busy else td, *rest) for c, f, th, td, *rest in items]
        fitted = v10.fit_settle(t, planted, rounds=3, n_paths=40, prior=0.0)
        n, real, before, after = fitted[busy]
        self.assertEqual(real, 0.0)
        self.assertGreater(t.td_hold[busy], 0.3)
        self.assertLess(after, before)
        self.assertTrue(((t.td_hold >= 0) & (t.td_hold <= sim10.SETTLE_MAX)).all())


class TestOvertime(unittest.TestCase):
    """Overtime as the feed shows it played: each side has the ball once,
    then the game ends the moment one side leads; one or two behind after
    a touchdown once the other side has had the ball, a side goes for two."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim10.Tables.build(_matches(60), min_records=20)

    def test_a_game_ends_once_both_have_had_the_ball_and_one_leads(self):
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 240.0, sim10.KICK
        st.team[:], st.home[:], st.away[:] = 0, 20, 20
        stats = {}
        h, a = sim10.simulate(self.tables, st, 4000, np.random.default_rng(1), seed=3, stats=stats)
        margin, points = np.abs(h - a).ravel(), (h + a - 40).ravel()
        self.assertGreater(stats["ot_decided"], 0)
        # decided by the first lead after both possessions: never by more
        # than a converted touchdown bar a defensive score, and no shoot-outs
        self.assertGreater(float((margin <= 8).mean()), 0.97)
        self.assertLess(float((points >= 17).mean()), 0.03)

    def test_one_behind_after_a_touchdown_it_goes_for_two(self):
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 150.0, sim10.CONV
        st.team[:], st.home[:], st.away[:] = 1, 27, 26      # the away side's six: one behind
        h, a = sim10.simulate(self.tables, st, 3000, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertTrue(set(np.unique(a)) <= {26, 28})        # never the kick to tie
        st.home[:] = 20                                       # six to lead: kicks as usual
        h, a = sim10.simulate(self.tables, st, 3000, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertIn(27, set(np.unique(a)))


class TestOvertimeRules(unittest.TestCase):
    """A touchdown that wins overtime ends it with no conversion, and a side behind once the other
    has had the ball never punts or kicks short of a tie."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim10.Tables.build(_matches(60), min_records=20)

    def test_a_walk_off_touchdown_ends_the_game_without_its_kick(self):
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 240.0, sim10.KICK
        st.team[:], st.home[:], st.away[:] = 0, 20, 20
        stats = {}
        h, a = sim10.simulate(self.tables, st, 6000, np.random.default_rng(1), seed=3, stats=stats,
                             common=False)
        points = (h + a - 40).ravel()
        self.assertGreater(stats.get("ot_walk_off", 0), 0)
        # a field goal answered by a touchdown ends at 9, never 10 with the kick
        self.assertLess(float((points == 10).mean()), 0.005)

    def test_level_before_the_other_side_has_had_the_ball_it_goes_as_real_overtimes_do(self):
        import copy
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 200.0, sim10.SCRIM
        st.team[:], st.home[:], st.away[:] = 0, 20, 20
        st.down[:], st.dist[:], st.y[:] = 4, 5, 80
        for ot_go in (0.9, 0.1):
            t = copy.deepcopy(self.tables)
            t.ot_go = ot_go
            stats = {}
            sim10.simulate(t, st, 4000, np.random.default_rng(1), seed=3, stats=stats, max_steps=1,
                          common=False)
            self.assertAlmostEqual(stats.get("fourth_go", 0) / 4000, ot_go, delta=0.03)
            self.assertEqual(stats.get("punt", 0), 0)

    def test_behind_by_a_touchdown_after_the_other_side_had_the_ball_it_goes_for_it(self):
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 150.0, sim10.SCRIM
        st.team[:], st.home[:], st.away[:] = 1, 27, 20      # the home side scored first
        st.down[:], st.dist[:], st.y[:] = 4, 10, 30
        stats = {}
        sim10.simulate(self.tables, st, 2000, np.random.default_rng(1), seed=3, stats=stats,
                      max_steps=1)
        self.assertEqual(stats.get("punt", 0) + stats.get("fg", 0), 0)
        self.assertEqual(stats["fourth_go"], 2000)


class TestFourthDownFit(unittest.TestCase):
    """The league's go curve, the part-of-game shifts and each player's own go and kick shifts, fitted
    together."""

    def test_the_fit_finds_the_bold_player_against_the_league(self):
        rng = np.random.default_rng(5)
        dp = sim10.DriveParams()
        recs = []
        for i in range(6000):
            y, t = int(rng.integers(20, 95)), int(rng.integers(1, 12))
            who = ["BOLD", "CALM", "MID"][i % 3]
            z = sim10._go_basis([y], [t])[0] @ np.array(dp.go_coef) + {"BOLD": 1.0, "CALM": -1.0, "MID": 0.0}[who]
            went = rng.random() < 1 / (1 + np.exp(-z))
            fg_range = 100 - y + 17 <= 55
            choice = "go" if went else ("fg" if fg_range and rng.random() < 0.9 else "punt")
            recs.append((0, 3, y, t, choice, who))
        go_coef, go_shift, go_p, kick_coef, kick_shift, kick_p = sim10.fit_fourth_downs(recs, dp)
        self.assertGreater(go_p["BOLD"] - go_p["MID"], 0.7)
        self.assertLess(go_p["CALM"] - go_p["MID"], -0.7)
        self.assertEqual(set(kick_p), {"BOLD", "CALM", "MID"})
        self.assertEqual(go_shift.shape, (4, 7))

    def test_the_fitted_curves_ride_in_the_tables(self):
        t = sim10.Tables.build(_matches(40), min_records=20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            t.save(path)
            back = sim10.Tables.load(path)
        self.assertEqual(back.drive.go_coef, t.drive.go_coef)
        self.assertEqual(back.drive.fg_kick_coef, t.drive.fg_kick_coef)
        self.assertNotEqual(t.drive.go_coef, sim10.DriveParams().go_coef)

    def test_the_offense_handle_follows_team_a(self):
        self.assertEqual(sim10._offense_handle({"offense": "TEAM_A", "team_a_side": "home"}, ("H", "A")), "H")
        self.assertEqual(sim10._offense_handle({"offense": "TEAM_A", "team_a_side": "away"}, ("H", "A")), "A")
        self.assertEqual(sim10._offense_handle({"offense": "TEAM_B", "team_a_side": "home"}, ("H", "A")), "A")
        self.assertIsNone(sim10._offense_handle({"offense": "TEAM_B"}, None))


class TestSecondHalfKick(unittest.TestCase):
    """The opening receiver kicks the second half; where the feed did not
    say who received, a coin per path."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim10.Tables.build(_matches(60), min_records=20)

    def _margin(self, kicks):
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 2, 0.0, sim10.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 30, 10, 10
        st.kicks_second_half[:] = kicks
        h, a = sim10.simulate(self.tables, st, 4000, np.random.default_rng(1), seed=21)
        return float((h - a).mean())

    def test_an_unknown_opening_is_a_coin(self):
        home_kicks, away_kicks, unknown = self._margin(0), self._margin(1), self._margin(-1)
        self.assertLess(home_kicks, away_kicks)             # receiving the half is worth points
        self.assertLess(home_kicks, unknown)
        self.assertLess(unknown, away_kicks)
        self.assertEqual(int(sim10.Start(1).kicks_second_half[0]), -1)

    def test_a_snapshot_without_the_opening_says_so(self):
        state = mock.Mock(pending_conversion=None, down=None, offense=sim10_home(), period=1,
                          clock_seconds=200.0, home_score=0, away_score=0, opening_receiver=None)
        self.assertEqual(v10.start_from(state)["kicks_second_half"], -1)
        state.opening_receiver = sim10_home()
        self.assertEqual(v10.start_from(state)["kicks_second_half"], 0)


def sim10_home():
    return v10.HOME


def sim10_profile():
    from .. import players
    return players.Profile()


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.matches = _with_handles(_matches(40))
        # the in-play shift and the quarter-start fit are off by default; the
        # build is tested with them on
        saved = v10.IN_PLAY_FIT, v10.QUARTER_START_FIT
        v10.IN_PLAY_FIT = v10.QUARTER_START_FIT = True
        try:
            v10.build(cls.matches, cls.tmp.name, grid_paths=60, verbose=False)
        finally:
            v10.IN_PLAY_FIT, v10.QUARTER_START_FIT = saved
        cls.tables = sim10.Tables.load(os.path.join(cls.tmp.name, "v10tables.npz"))
        cls.grid = v10.PriorGrid.load(os.path.join(cls.tmp.name, "v10grid.npz"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_writes_the_model_and_the_profiles(self):
        for name in ("v10tables.npz", "v10grid.npz", "v10players.json"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp.name, name)), name)
        book = v10.players_book(self.tmp.name)
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
        got = {q: pts for q, _, _, pts in v10.quarter_start_states({"AFX": rows}, self.grid)}
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
        got = {q: pts for q, _, _, pts in v10.late_start_states({"AFX": rows}, self.grid)}
        self.assertEqual(got, {2: 7, 4: 7})

    def test_the_build_fits_the_quarters_and_the_two_minute_drills(self):
        import copy
        self.assertFalse(v10.QUARTER_START_FIT)          # off by default (see v10.py)
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = 0.0          # this class's build has the in-play shift on too
        quarters, lates = v10.fit_quarter_levels(
            tables, v10.quarter_start_states(self.matches, self.grid),
            v10.late_start_states(self.matches, self.grid), rounds=1, n_paths=200)
        for real, got in list(quarters.values()) + list(lates.values()):
            self.assertAlmostEqual(got, real, delta=max(0.6, 0.12 * real))
        self.assertEqual(set(lates), {2, 4})
        self.assertTrue(np.any(self.tables.late_theta))

    def test_the_two_minute_level_is_only_the_last_two_minutes(self):
        import copy
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = 0.0
        tables.late_theta[:] = 0.0
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:], st.y[:] = 2, 239.0, sim10.SCRIM, 30
        plain = sim10.simulate(tables, st, 3000, np.random.default_rng(1), seed=4, max_steps=2)
        tables.late_theta[2] = 1.0
        early = sim10.simulate(tables, st, 3000, np.random.default_rng(1), seed=4, max_steps=2)
        self.assertTrue(np.array_equal(plain[0], early[0]))    # 4:00 left: untouched
        st.clock[:] = 100.0
        tables.late_theta[2] = 0.0
        a = sim10.simulate(tables, st, 3000, np.random.default_rng(1), seed=4)
        tables.late_theta[2] = 1.0
        b = sim10.simulate(tables, st, 3000, np.random.default_rng(1), seed=4)
        self.assertGreater(float((b[0] + b[1]).mean()), float((a[0] + a[1]).mean()))

    def test_the_quarter_fit_on_real_states_converges_when_asked(self):
        # on fixed states and thetas the fit brings the quarters to what
        # games scored
        import copy
        tables = copy.deepcopy(self.tables)
        items = v10.quarter_start_states(self.matches, self.grid)
        v10.fit_period_theta_states(tables, items, rounds=6, n_paths=300)
        _, got, real = v10.fit_period_theta_states(tables, items, rounds=1, n_paths=300)
        for q in real:
            self.assertAlmostEqual(got[q], real[q], delta=max(0.4, 0.08 * real[q]))

    def test_the_rest_of_game_fit_leaves_what_games_really_left(self):
        import copy
        tables = copy.deepcopy(self.tables)
        items = v10.rest_of_game_states(self.matches, self.grid, every=3)
        q3 = sim10.SEGMENTS.index("Q3")
        self.assertTrue({it[0][0] for it in items} >= {1, 2, 3, q3, 5, 6})
        # games that left a point fewer than they did from every state in
        # the third quarter: its in-play scoring comes down, most of the way
        lower = [(c, st, th, pts - 1 if c[0] == q3 else pts) for c, st, th, pts in items]
        before = tables.inplay_theta.copy()
        segs, bands = v10.fit_rest_of_game(tables, lower, n_paths=120, rounds=6, max_states=600)
        real, simulated_before, after = segs[q3]
        self.assertLess(abs(after - real), 0.5 * abs(simulated_before - real))
        self.assertLess(tables.inplay_theta[q3].mean(), before[q3].mean() - 0.02)
        for (g, band), (r, b, a) in bands.items():
            self.assertIn(band, range(sim10.N_BANDS))
        # and the kickoff scoring, the pre-match's, is untouched
        self.assertTrue((tables.period_theta == self.tables.period_theta).all())

    def test_the_in_play_shift_is_only_in_play(self):
        import copy
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = -0.5
        st = sim10.Start(1)
        st.period[:], st.clock[:], st.phase[:], st.y[:] = 3, 200.0, sim10.SCRIM, 30
        runs = {flag: sim10.simulate(tables, st, 3000, np.random.default_rng(1), seed=3, in_play=flag)
                for flag in (False, True)}
        total = {flag: float((h + a).mean()) for flag, (h, a) in runs.items()}
        self.assertLess(total[True], total[False] - 1.0)
        self.assertFalse(v10.IN_PLAY_FIT)
        # a build with it on fitted it and saved it
        self.assertTrue(np.abs(self.tables.inplay_theta[1:7]).sum() > 0)

    def test_the_in_play_shift_moves_the_total_and_not_the_margin(self):
        import copy
        from .. import playover
        rows = v10.resolve_sides(list(self.matches.values())[0])
        states, messages = [], []
        for r in rows[10:40:5]:
            st, _ = playover.state_for(r)
            if st is not None:
                states.append(st)
                messages.append(int(r["message"]))
        prof = (sim10_profile(), sim10_profile())
        books = {}
        for name, shift in (("plain", 0.0), ("shifted", -0.6)):
            tables = copy.deepcopy(self.tables)
            tables.inplay_theta[:] = shift
            books[name] = v10.price_states(tables, (0.0, 0.0), v10.Variant("v10"), [], True, states,
                                          messages, prof, 400, np.random.default_rng(2), seed=11)
        x = np.arange(len(books["plain"][0][1]))
        for (mp0, tp0), (mp1, tp1) in zip(books["plain"], books["shifted"]):
            self.assertTrue(np.allclose(mp0, mp1))
            self.assertLess((tp1 * x).sum(), (tp0 * x).sum())

    def test_nothing_of_gameplais_is_read(self):
        # every prod price column moved far off: v10's priors, states and
        # prices come out the same
        moved = {c: [dict(r, prematch_line_52="-20.5", prematch_prob_52="0.9",
                          prematch_line_54="80.5", prematch_prob_54="0.9", prematch_prob_50="0.99",
                          line_52="-20.5", prob_52="0.9", line_54="80.5", prob_54="0.9",
                          prob_50="0.99") for r in rows]
                 for c, rows in self.matches.items()}
        for fn, k in ((v10.in_game_states, 1), (v10.rest_of_game_states, 2),
                      (v10.quarter_start_states, 2)):
            a, b = fn(self.matches, self.grid), fn(moved, self.grid)
            self.assertTrue(a)
            self.assertEqual([x[k] for x in a], [x[k] for x in b])
            self.assertTrue(all(tuple(x[k]) == v10.LEAGUE_THETA for x in a))
        rows = next(iter(self.matches.values()))
        a = v10_stream.match_books(self.tables, self.grid, v10.Variant("v10"), rows, 60,
                                  np.random.default_rng(1))
        b = v10_stream.match_books(self.tables, self.grid, v10.Variant("v10"), moved[rows[0]["match_code"]],
                                  60, np.random.default_rng(1))
        self.assertTrue(a)
        for (_, mp1, tp1), (_, mp2, tp2) in zip(a, b):
            self.assertTrue(np.array_equal(mp1, mp2) and np.array_equal(tp1, tp2))
        for gone in ("prior_lines", "recent_total_shade"):
            self.assertFalse(hasattr(v10, gone))
        self.assertFalse(hasattr(v10.PriorGrid, "fit"))

    def test_the_build_needs_history_on_the_command_line(self):
        from ..__main__ import main
        path = os.path.join(self.tmp.name, "snaps_cli.csv")
        with open(path, "w") as fh:
            fh.write("match_code\n")
        with self.assertRaises(SystemExit) as caught:
            main(["v10-build", path, "--out", os.path.join(self.tmp.name, "cli")])
        self.assertIn("--history", str(caught.exception))

    def test_handles_come_off_the_export(self):
        rows = next(iter(self.matches.values()))
        self.assertEqual(v10.handles_of(rows), (rows[0]["home_handle"], rows[0]["away_handle"]))
        self.assertIsNone(v10.handles_of([{"home_handle": "", "away_handle": None}]))

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
        graded, _ = v10.run(path, os.path.join(self.tmp.name, "v10tables.npz"),
                           os.path.join(self.tmp.name, "v10grid.npz"),
                           [v10.Variant("v10"), v10.Variant("plain", profiles=False, pace=False)],
                           n_paths=100, workers=1)
        self.assertGreater(len(graded), 50)
        for _, row, probs in graded:
            self.assertTrue(0 < probs["v10"] < 1 and 0 < probs["plain"] < 1)

    def test_the_stream_finds_the_model_and_quotes(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        prod = [(code, 54, None, 50.0, 2.0, "Total points over 30.5", int(r["message"]), "OPEN", "true")
                for r in rows]
        quotes = v10_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
        self.assertTrue(quotes)
        self.assertTrue(all(q[1] == 54 and 0 < q[3] < 100 for q in quotes))

    def test_fit_means_finds_the_thetas_that_score_the_points(self):
        g = self.grid
        x_m = np.arange(g.margin.shape[-1]) - v10.MARGIN_MAX
        x_t = np.arange(g.total.shape[-1])
        for i, j in ((8, 8), (4, 11), (12, 6)):
            margin, total = (g.margin[i, j] * x_m).sum(), (g.total[i, j] * x_t).sum()
            got = v10.fit_means(g, (total + margin) / 2, (total - margin) / 2)
            self.assertAlmostEqual(got[0], g.grid[i], delta=0.02)
            self.assertAlmostEqual(got[1], g.grid[j], delta=0.02)
        low, high = v10.fit_means(g, 14, 14), v10.fit_means(g, 24, 24)
        self.assertLess(sum(low), sum(high))

    def test_a_fixed_seed_prices_a_match_the_same_every_time(self):
        rows = next(iter(self.matches.values()))
        a = v10_stream.match_books(self.tables, self.grid, v10.Variant("v10"), rows, 80,
                                  np.random.default_rng(1), means=(20.0, 14.0))
        b = v10_stream.match_books(self.tables, self.grid, v10.Variant("v10"), rows, 80,
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
        args = (self.tables, self.grid, v10.Variant("v10"))
        a = v10_stream.match_books(*args, rows, 80, np.random.default_rng(1), means=(20.0, 14.0))
        b = v10_stream.match_books(*args, other, 80, np.random.default_rng(1), means=(20.0, 14.0))
        for (_, mp1, tp1), (_, mp2, tp2) in zip(a, b):
            self.assertTrue(np.array_equal(mp1, mp2) and np.array_equal(tp1, tp2))

    def _snapshot_file(self):
        path = os.path.join(self.tmp.name, "snaps.csv")
        if not os.path.exists(path):
            self.test_grades_a_snapshot_file_with_profiles()
        return path

    def test_a_model_with_nb2_takes_its_prior_from_there(self):
        path = self._snapshot_file()
        tables, grid = (os.path.join(self.tmp.name, n) for n in ("v10tables.npz", "v10grid.npz"))
        codes = sorted(self.matches)[:6]

        class Fake:
            league = (17.0, 17.0)

            def __init__(self, level):
                self.level = level

            def means(self, schedule, n_sims=0, results=None):
                return {r["MATCH_CODE"]: self.level for r in schedule}

        history = [{"MATCH_CODE": c} for c in codes]
        with mock.patch.object(v10, "prematch_model", return_value=Fake((12.0, 12.0))):
            with self.assertRaises(SystemExit) as caught:
                v10.run(path, tables, grid, [v10.Variant("v10")], matches=codes, workers=1)
            self.assertIn("--history", str(caught.exception))
            low, _ = v10.run(path, tables, grid, [v10.Variant("v10")], matches=codes, n_paths=100,
                            workers=1, history=history)
        with mock.patch.object(v10, "prematch_model", return_value=Fake((26.0, 26.0))):
            high, _ = v10.run(path, tables, grid, [v10.Variant("v10")], matches=codes, n_paths=100,
                             workers=1, history=history)
        over = lambda graded: np.mean([p["v10"] for _, r, p in graded if r.market_id == 54])
        self.assertLess(over(low), over(high))
        # no prediction for a match: skipped, never priced off prod
        _, skipped = v10.run(path, tables, grid, [v10.Variant("v10")], matches=codes, n_paths=50,
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

            def means(self, schedule, n_sims=0, results=None):
                Fake.asked = schedule
                return {}

        with mock.patch.object(v10, "prematch_model", return_value=Fake()):
            with self.assertRaises(SystemExit):
                v10_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
            quotes = v10_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50,
                                                  workers=1, match_info=[])
        self.assertTrue(quotes)
        self.assertEqual(Fake.asked, [])

    def test_a_prior_that_follows_results_prices_each_prematch_quote_off_what_was_known(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        start = dt.datetime(2026, 9, 1, 12, 0)
        early, late = start - dt.timedelta(minutes=50), start - dt.timedelta(minutes=5)
        line = "Total points over 40.5"
        prod = [(code, 54, t, 50.0, 2.0, line, None, "OPEN", "true") for t in (early, late)]
        prod += [(code, 54, None, 50.0, 2.0, line, int(r["message"]), "OPEN", "true")
                 for r in rows[5:8]]
        teams = {"SPORT_CODE": "AF", "STREAM_NUMBER": "1", "PLAYER_1_TEAM": "T1",
                 "PLAYER_2_TEAM": "T2"}
        info = [dict(teams, MATCH_CODE=code, SCHEDULED_START_TIME_UTC="2026-09-01 12:00:00",
                     PLAYER_1_HANDLE="ann", PLAYER_2_HANDLE="bob")]
        # ann's previous match, from 10:48 (its result in by 11:24), is still on 50 minutes before
        # kick-off and over 5 minutes before
        history = [dict(teams, MATCH_CODE="PREV", SCHEDULED_START_TIME_UTC="2026-09-01 10:48:00",
                        PLAYER_1_HANDLE="ann", PLAYER_2_HANDLE="cat", PLAYER_1_FINAL_SCORE="20",
                        PLAYER_2_FINAL_SCORE="10")]

        class Fake:
            league = (17.0, 17.0)
            FOLLOWS_RESULTS = True
            asked = []

            def means(self, schedule, n_sims=0, results=None):
                Fake.asked.append((schedule, results))
                return {r["MATCH_CODE"]: (10.0, 10.0) if "@" in r["MATCH_CODE"] else (30.0, 30.0)
                        for r in schedule}

        price = lambda **kw: v10_stream.quotes_for_matches(
            {code: rows}, prod, self.tmp.name, n_paths=200, workers=1, match_info=info,
            lines=v10_stream.PROD_LINES, **kw)
        with mock.patch.object(v10, "prematch_model", return_value=Fake()):
            quotes = price(history=history)
            pre_only = price(history=history, prematch_only=True)
            frozen = price()
        before = {q[2]: q[3] for q in quotes if q[6] is None}
        self.assertLess(before[early], 20.0)          # priced off ann's form before PREV's result
        self.assertGreater(before[late], before[early] + 30)          # and after it
        schedule, results = Fake.asked[0]
        self.assertIs(results, history)
        self.assertEqual([r["MATCH_CODE"] for r in schedule], [code, code + "@1"])
        self.assertEqual(schedule[1]["SCHEDULED_START_TIME_UTC"], "2026-09-01 10:47:59")
        self.assertTrue(any(q[6] is not None for q in quotes))
        # prematch_only: the quotes before kick-off alone, the same prices
        self.assertEqual({q[6] for q in pre_only}, {None})
        self.assertEqual({q[2]: q[3] for q in pre_only}, before)
        # no results known when pricing: every quote off the kick-off's
        self.assertEqual([r["MATCH_CODE"] for r in Fake.asked[2][0]], [code])
        self.assertEqual(len({q[3] for q in frozen if q[6] is None}), 1)


    def test_the_even_line_is_the_half_point_nearest_even_money(self):
        pmf = np.zeros(80)
        pmf[[38, 41, 44, 47]] = [0.2, 0.25, 0.3, 0.25]
        # P(> 38.5) = 0.8, P(> 41.5 .. 43.5) = 0.55, P(> 44.5) = 0.25: of the
        # three 0.55 lines, the one nearest the mean (42.8)
        self.assertEqual(v10.even_line(pmf, 0), 42.5)
        margin = np.zeros(2 * v10.MARGIN_MAX + 1)
        margin[v10.MARGIN_MAX + np.array([-3, 3, 7])] = [0.3, 0.45, 0.25]
        # -2.5 .. 2.5 all give P(home by more) = 0.7: the one nearest the mean (2.2)
        self.assertEqual(v10.even_line(margin, v10.MARGIN_MAX), 2.5)
        line = v10.even_line(margin, v10.MARGIN_MAX)
        self.assertLessEqual(abs(v10.market_prob(52, line, margin, pmf) - 0.5), 0.2 + 1e-9)

    def test_the_anchored_line_moves_prods_only_as_far_as_the_band(self):
        pmf = np.zeros(80)
        pmf[30:60] = 1 / 30                      # flat on 30..59: P(> x.5) = (59 - x) / 30
        anchored = lambda start, pmf=pmf, offset=0: v10.anchored_line(pmf, offset, start, band=0.11)
        self.assertEqual(anchored(44.5), 44.5)                           # 0.5: kept
        self.assertEqual(anchored(46.5), 46.5)                           # 0.43: inside 39-61%
        self.assertEqual(anchored(49.5), 47.5)                           # 0.33: down to 0.4
        self.assertEqual(anchored(36.5), 41.5)                           # 0.77: up to 0.6
        self.assertEqual(anchored(45.0), 45.0)                           # a whole line stays whole
        self.assertEqual(v10.anchored_line(pmf, 0, 36.5, band=0.02), 44.5)
        margin = np.zeros(2 * v10.MARGIN_MAX + 1)
        margin[v10.MARGIN_MAX - 10:v10.MARGIN_MAX + 10] = 1 / 20         # -10..9
        self.assertEqual(anchored(6.5, margin, v10.MARGIN_MAX), 1.5)
        self.assertEqual(anchored(-8.5, margin, v10.MARGIN_MAX), -2.5)

    def test_the_held_line_stays_until_the_even_line_is_far_or_the_price_is(self):
        pmf = np.zeros(80)
        pmf[30:60] = 1 / 30                      # even 44.5
        self.assertEqual(v10.held_line(pmf, 0, None), 44.5)
        self.assertEqual(v10.held_line(pmf, 0, 43.5), 43.5)              # 1 point, 0.53
        self.assertEqual(v10.held_line(pmf, 0, 42.5), 44.5)              # 2 points away
        self.assertEqual(v10.held_line(pmf, 0, 42.5, move=3.0), 42.5)    # 0.57, inside 35-65%
        self.assertEqual(v10.held_line(pmf, 0, 39.5, move=9.0), 44.5)    # 0.67, outside

    def test_the_key_line_sits_in_the_gap_between_the_spikes(self):
        pmf = np.zeros(60)
        pmf[[24, 27, 31, 34, 38]] = [0.15, 0.30, 0.25, 0.20, 0.10]   # lumps of 3 and 4
        # P(over): 26.5 0.85 | 27.5 .. 30.5 0.55 | 31.5 .. 33.5 0.30: the band 40-60% holds
        # 27.5 .. 30.5; 28.5 and 29.5 have nothing either side, and 28.5 is the first
        self.assertEqual(v10.key_line(pmf, 0), 28.5)
        self.assertEqual(v10.even_line(pmf, 0), 30.5)     # the even line (nearest the mean, 30.05) sits on 31
        self.assertEqual(v10.key_line(pmf, 0, kept=27.5), 28.5)     # 0.30 next to 27.5: too heavy
        self.assertEqual(v10.key_line(pmf, 0, kept=29.5), 29.5)     # as light: kept
        self.assertEqual(v10.key_line(pmf, 0, kept=27.5, hold=0.5), 27.5)
        self.assertEqual(v10.key_line(pmf, 0, kept=33.5, hold=0.5), 28.5)   # outside the band
        spike = np.zeros(60)
        spike[30] = 1.0                                             # nothing in the band: even
        self.assertEqual(v10.key_line(spike, 0), v10.even_line(spike, 0))
        margin = np.zeros(2 * v10.MARGIN_MAX + 1)
        margin[v10.MARGIN_MAX + np.array([-7, -3, 3, 7])] = [0.2, 0.3, 0.3, 0.2]
        self.assertEqual(v10.key_line(margin, v10.MARGIN_MAX, band=0.0), -1.5)   # not -2.5, on the 3

    def test_the_stream_quotes_every_line_rule_off_one_book(self):
        margin = np.zeros(2 * v10.MARGIN_MAX + 1)
        margin[v10.MARGIN_MAX - 10:v10.MARGIN_MAX + 8] = 1 / 18          # -10..7: even -1.5
        total = np.zeros(v10.TOTAL_MAX + 1)
        total[30:58] = 1 / 28                                            # 30..57: even 43.5
        row = lambda m, line, msg: ("M", m, None, 50.0, 2.0, v10_stream.description(m, line), msg,
                                    "OPEN", "true")
        prod = [row(52, 6.5, 1), row(53, -6.5, 1), row(54, 49.5, 1), row(55, 49.5, 1)]
        lines = lambda mode, books=((1, margin, total),), rows=prod: {
            (q[6], q[1]): v10_stream._parse_line(q[5])
            for q in v10_stream.quote_rows("M", list(books), rows, lines=mode)}
        self.assertEqual(lines("prod"), {(1, 52): 6.5, (1, 53): -6.5, (1, 54): 49.5, (1, 55): 49.5})
        # prod's lines stepped toward even until P(over) and P(home covers) reach 40%
        self.assertEqual(lines("anchored"), {(1, 52): -0.5, (1, 53): 0.5, (1, 54): 45.5, (1, 55): 45.5})
        self.assertEqual(lines("even"), {(1, 52): -1.5, (1, 53): 1.5, (1, 54): 43.5, (1, 55): 43.5})
        # hyst keeps 43.5 while the book drifts a point, and moves once it is two away
        later, far = np.roll(total, 1), np.roll(total, 2)
        books = ((1, margin, total), (2, margin, later), (3, margin, far))
        rows = [row(54, 49.5, m) for m in (1, 2, 3)]
        self.assertEqual(lines("hyst", books, rows), {(1, 54): 43.5, (2, 54): 43.5, (3, 54): 45.5})
        self.assertEqual(lines("even", books, rows), {(1, 54): 43.5, (2, 54): 44.5, (3, 54): 45.5})
        # own (key numbers): a flat book has no spikes, so the line nearest 50%, kept while the
        # book drifts and the line stays in the band
        self.assertEqual(lines("own", books, rows), {(1, 54): 43.5, (2, 54): 43.5, (3, 54): 43.5})
        self.assertEqual(lines("own"), {(1, 52): -1.5, (1, 53): 1.5, (1, 54): 43.5, (1, 55): 43.5})
        with self.assertRaises(ValueError):
            lines("nearest")

    def test_the_stream_moves_its_own_line_with_the_game(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        prod = [(code, m, None, 50.0, 2.0, v10_stream.description(m, 30.5 if m in (54, 55) else 0.5),
                 int(r["message"]), "OPEN", "true") for r in rows for m in (52, 53, 54, 55)]
        own = v10_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=100, workers=1)
        at_prod = v10_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=100,
                                               workers=1, lines="prod")
        parse = v10_stream._parse_line
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
        strong = v10.price_kickoff(self.tables, v10.prior_theta(self.grid, (26.0, 12.0)), v10.Variant("v10"),
                                  prof, 600, rng, seed=1)
        weak = v10.price_kickoff(self.tables, v10.prior_theta(self.grid, (12.0, 26.0)), v10.Variant("v10"),
                                prof, 600, rng, seed=1)
        for margin, total in (strong, weak):
            self.assertAlmostEqual(margin.sum(), 1.0, places=6)
            self.assertAlmostEqual(total.sum(), 1.0, places=6)
        home_win = lambda book: v10.market_prob(50, 0.0, book[0], book[1])
        self.assertGreater(home_win(strong), 0.5)
        self.assertLess(home_win(weak), 0.5)

    def test_pre_match_and_pre_play_rows_get_the_kick_off_price(self):
        margin = np.zeros(2 * v10.MARGIN_MAX + 1)
        margin[v10.MARGIN_MAX + 7] = 1.0
        total = np.zeros(v10.TOTAL_MAX + 1)
        total[40] = 1.0
        kick_margin = np.zeros_like(margin)
        kick_margin[v10.MARGIN_MAX - 3] = 1.0
        row = lambda m: ("M", 50, None, 50.0, 2.0, "PLAYER 1 to win", m, "OPEN", "true")
        prod = [row(None), row(3), row(7), row(12)]
        out = v10_stream.quote_rows("M", [(10, margin, total)], prod, first_play_message=5,
                                   kickoff=(kick_margin, total))
        got = {q[6]: q[3] for q in out}
        self.assertEqual(set(got), {None, 3, 12})           # 7: the first play is under way
        self.assertEqual((got[None], got[3], got[12]), (0.01, 0.01, 99.99))
        self.assertEqual({q[6] for q in v10_stream.quote_rows("M", [(10, margin, total)], prod, 5)},
                         {12})

    def test_a_match_whose_side_is_not_known_still_gets_its_pre_match_price(self):
        rows = [dict(r, team_a_side="") for r in next(iter(self.matches.values()))]
        code = rows[0]["match_code"]
        prod = [(code, 50, None, 50.0, 2.0, "PLAYER 1 to win", None, "OPEN", "true")] + \
               [(code, 50, None, 50.0, 2.0, "PLAYER 1 to win", int(r["message"]), "OPEN", "true")
                for r in rows]
        quotes = v10_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
        self.assertEqual([q[6] for q in quotes], [None])

    def test_a_missing_model_says_how_to_build_it(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(SystemExit) as caught:
                v10_stream.model_paths(empty)
        self.assertIn("v10-build", str(caught.exception))


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

    def test_v9_build_with_history_writes_the_prematch_model(self):
        from .fakes import _matches
        matches = _matches(12)
        out = os.path.join(self.tmp.name, "v10")
        v10.build(matches, out, grid_paths=40, verbose=False, history=self.history(),
                 before=self.BEFORE)
        self.assertIsNotNone(v10.prematch_model(out))
        self.assertIsNotNone(v10.prematch_model(os.path.join(out, "v10tables.npz")))

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
    history it fits on, what predict.R is handed, the league-average fallback, and a v10 build
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

    def test_v9_build_prices_off_the_prior_it_was_built_with(self):
        out = os.path.join(self.tmp.name, "v10")
        v10.build(_matches(12), out, grid_paths=40, verbose=False, history=self.history(),
                 before=self.BEFORE, prior="glmer")
        self.assertIsInstance(v10.prematch_model(out), glmer_prior.Prematch)
        self.assertIsInstance(v10.prematch_model(os.path.join(out, "v10tables.npz")),
                              glmer_prior.Prematch)
        self.assertFalse(os.path.exists(os.path.join(out, "nb2")))
        with open(os.path.join(out, v10.PRIOR_FILE)) as fh:
            self.assertEqual(json.load(fh), {"prior": "glmer"})

    def test_a_model_built_before_the_choice_still_reads_nb2(self):
        old = os.path.join(self.tmp.name, "old")
        os.makedirs(os.path.join(old, "nb2"))
        with mock.patch.object(nb2_prior.Prematch, "exists", return_value=True), \
                mock.patch.object(nb2_prior.Prematch, "__init__", return_value=None):
            self.assertIsInstance(v10.prematch_model(old), nb2_prior.Prematch)
        self.assertIsNone(v10.prematch_model(self.tmp.name))  # nothing built here

    def test_an_unknown_prior_is_refused(self):
        with self.assertRaises(ValueError):
            v10.build(_matches(2), os.path.join(self.tmp.name, "x"), verbose=False, prior="elo")

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


class TestPriorPace(unittest.TestCase):
    """v10: the pre-match prior's expected points carry each player's pace already, so the
    starting strengths are fitted with the pace response taken out."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim10.Tables.build(_matches(), min_records=20)
        cls.grid = v10.PriorGrid.build(cls.tables, n_paths=300)
        cls.grid.pace_home, cls.grid.pace_away = v10.fit_pace_response(cls.tables, n_paths=600)

    def test_the_response_is_one_at_league_pace_and_falls_with_slower_play(self):
        n = len(v10.PACE_POINTS)
        for table in (self.grid.pace_home, self.grid.pace_away):
            self.assertAlmostEqual(table[n // 2, n // 2], 1.0)
            self.assertGreater(table[0, n // 2], table[-1, n // 2])    # home's pace
            self.assertGreater(table[n // 2, 0], table[n // 2, -1])    # away's pace
        self.assertEqual(self.grid.pace_response(1.0, 1.0), (1.0, 1.0))
        self.assertEqual(v10.PriorGrid(self.grid.margin, self.grid.total).pace_response(0.9, 0.9),
                         (1.0, 1.0))

    def test_fast_players_start_weaker_so_the_points_match_the_prior(self):
        means = (19.0, 16.0)
        plain = v10.prior_theta(self.grid, means)
        league = v10.prior_theta(self.grid, means, (players.Profile(), players.Profile()))
        self.assertEqual(plain, league)
        fast = v10.prior_theta(self.grid, means, (players.Profile(pace=0.88), players.Profile(pace=0.88)))
        self.assertLess(fast[0], plain[0])
        self.assertLess(fast[1], plain[1])
        with mock.patch.object(v10, "PACE_NEUTRAL", False):
            self.assertEqual(v10.prior_theta(self.grid, means, (players.Profile(pace=0.88),) * 2), plain)
        self.assertEqual(v10.prior_theta(self.grid, None, (players.Profile(pace=0.88),) * 2),
                         v10.LEAGUE_THETA)

    def test_the_grid_keeps_its_pace_response(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.npz")
            self.grid.save(path)
            back = v10.PriorGrid.load(path)
        self.assertTrue(np.array_equal(back.pace_home, self.grid.pace_home))
        self.assertEqual(back.pace_response(0.9, 1.1), self.grid.pace_response(0.9, 1.1))


class TestPriorShrink(unittest.TestCase):
    """v10: the pre-match prior's totals and margins are pulled toward the build's average by the
    slopes real ones showed on it out of sample, pooled over several refits."""

    SHRINK = {"total_slope": 0.8, "margin_slope": 1.0, "total_centre": 35.0, "margin_centre": 0.0}
    BEFORE = dt.datetime(2026, 9, 20)

    def test_the_total_moves_and_the_margin_holds(self):
        h, a = v10.shrink_means((25.0, 20.0), self.SHRINK)
        self.assertAlmostEqual(h + a, 35.0 + 0.8 * 10.0)
        self.assertAlmostEqual(h - a, 5.0)
        h, a = v10.shrink_means((17.5, 17.5), self.SHRINK)
        self.assertAlmostEqual(h + a, 35.0)

    GAMERS = [f"G{i}" for i in range(6)]

    def history(self, n_days=30, margin=1.0, newcomer=0):
        """Six gamers, established by a season well before the stretches, then n_days of 30 matches
        a day; `newcomer` matches in the last days between a debutant and one of them, predicted
        12 points apart and played level."""
        import random
        rng = random.Random(4)
        rows, preds = [], {}
        old = f"{self.BEFORE - dt.timedelta(days=80):%Y-%m-%d %H:%M:%S}"
        for k in range(120):
            rows.append({"MATCH_CODE": f"OLD{k:03d}", "SCHEDULED_START_TIME_UTC": old,
                         "PLAYER_1_HANDLE": self.GAMERS[k % 6], "PLAYER_2_HANDLE": self.GAMERS[(k + 1) % 6],
                         "PLAYER_1_FINAL_SCORE": "17", "PLAYER_2_FINAL_SCORE": "17"})
        for d in range(n_days):
            day = self.BEFORE - dt.timedelta(days=n_days - d)
            for k in range(30):
                code = f"AF{d:02d}{k:02d}"
                pt, pm = rng.uniform(27, 43), rng.uniform(-6, 6)
                rt = 35.0 + 0.7 * (pt - 35.0) + rng.gauss(0, 1.0)
                rm = margin * pm + rng.gauss(0, 1.0)
                preds[code] = ((pt + pm) / 2, (pt - pm) / 2)
                a, b = rng.sample(self.GAMERS, 2)
                rows.append({"MATCH_CODE": code, "SCHEDULED_START_TIME_UTC": f"{day:%Y-%m-%d %H:%M:%S}",
                             "PLAYER_1_HANDLE": a, "PLAYER_2_HANDLE": b,
                             "PLAYER_1_FINAL_SCORE": f"{(rt + rm) / 2:.2f}",
                             "PLAYER_2_FINAL_SCORE": f"{(rt - rm) / 2:.2f}"})
        for k in range(newcomer):
            code = f"NEW{k:02d}"
            day = self.BEFORE - dt.timedelta(days=1) + dt.timedelta(minutes=k)
            preds[code] = (11.5, 23.5)
            rows.append({"MATCH_CODE": code, "SCHEDULED_START_TIME_UTC": f"{day:%Y-%m-%d %H:%M:%S}",
                         "PLAYER_1_HANDLE": "DEBUT", "PLAYER_2_HANDLE": self.GAMERS[k % 6],
                         "PLAYER_1_FINAL_SCORE": "17", "PLAYER_2_FINAL_SCORE": "17"})
        return rows, preds

    def test_the_slope_is_read_out_of_sample(self):
        rows, preds = self.history()
        seen = {"before": [], "n": 0}

        def fit(history, out_dir, before=None):
            seen["before"].append(before)

        def predict(d, schedule, *a, **kw):
            seen["n"] += len(schedule)
            return {r["MATCH_CODE"]: preds[r["MATCH_CODE"]] for r in schedule}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(nb2_prior, "fit", side_effect=fit), \
                mock.patch.object(nb2_prior, "predict", side_effect=predict):
            got = v10.fit_shrink(rows, "nb2", self.BEFORE, (35.0, 0.5), d, scale=1.0)
            # a refit at the start of each 14-day stretch; 30 days of history fill two and a bit
            step = dt.timedelta(days=v10.SHRINK_DAYS)
            self.assertEqual(sorted(seen["before"]), [self.BEFORE - 3 * step, self.BEFORE - 2 * step,
                                                      self.BEFORE - step])
            self.assertEqual(seen["n"], 30 * 30)
            self.assertEqual((got["matches"], got["windows"], len(got["each"])), (900, 3, 3))
            self.assertAlmostEqual(got["total_slope"], 0.7, delta=0.03)
            self.assertLess(got["total_se"], 0.02)
            self.assertAlmostEqual(got["margin_raw"], 1.0, delta=0.05)
            self.assertAlmostEqual(got["margin_slope"], 1.0, delta=0.05)
            self.assertEqual((got["total_centre"], got["margin_centre"]), (35.0, 0.5))
            halved = v10.fit_shrink(rows, "nb2", self.BEFORE, (35.0, 0.5), d, scale=2.0)
            self.assertAlmostEqual(halved["total_raw"], got["total_raw"] / 2)
            self.assertIsNone(v10.fit_shrink(rows[:100], "nb2", self.BEFORE, (35.0, 0.5), d))
            # margins that spread too wide are pulled in too, unless switched off
            rows, preds = self.history(margin=0.6)
            got = v10.fit_shrink(rows, "nb2", self.BEFORE, (35.0, 0.5), d)
            self.assertAlmostEqual(got["margin_raw"], 0.6, delta=0.03)
            pull = 1 - (got["margin_raw"] + got["margin_se"])          # toward 1 by its error
            self.assertAlmostEqual(got["margin_slope"], 1 - v10.MARGIN_WEIGHT * pull)
            with mock.patch.object(v10, "MARGIN_WEIGHT", 1.0):
                full = v10.fit_shrink(rows, "nb2", self.BEFORE, (35.0, 0.5), d)
            self.assertAlmostEqual(full["margin_slope"], 1 - pull)
            with mock.patch.object(v10, "MARGIN_SHRINK", False):
                self.assertEqual(v10.fit_shrink(rows, "nb2", self.BEFORE, (35.0, 0.5), d)
                                 ["margin_slope"], 1.0)

    def test_a_newcomer_s_first_matches_are_not_read(self):
        rows, preds = self.history(newcomer=25)
        seen = []

        def predict(d, schedule, *a, **kw):
            seen.extend(r["MATCH_CODE"] for r in schedule)
            return {r["MATCH_CODE"]: preds[r["MATCH_CODE"]] for r in schedule}
        played = v10.experience(rows)
        self.assertEqual(played["OLD000"], 0)
        self.assertEqual(played["NEW00"], 0)
        self.assertEqual(played["NEW24"], 24)
        self.assertGreaterEqual(min(played[c] for c in played if c.startswith("AF")), 40)
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(nb2_prior, "fit"), \
                mock.patch.object(nb2_prior, "predict", side_effect=predict):
            got = v10.fit_shrink(rows, "nb2", self.BEFORE, (35.0, 0.5), d)
            self.assertFalse(any(c.startswith("NEW") for c in seen))
            self.assertAlmostEqual(got["margin_raw"], 1.0, delta=0.05)
            with mock.patch.object(v10, "SHRINK_MIN_EXPERIENCE", 0):
                swung = v10.fit_shrink(rows, "nb2", self.BEFORE, (35.0, 0.5), d)
            self.assertLess(swung["margin_raw"], 0.85)          # 25 debut matches drag it down

    def test_the_slopes_pool_within_windows_and_move_toward_one_by_their_error(self):
        # two windows on different levels, each with slope 2: pooled within them, 2 exactly
        b, se = v10.pooled_slope([([0, 1, 2], [10, 12, 14]), ([5, 6, 7], [0, 2, 4])])
        self.assertAlmostEqual(b, 2.0)
        self.assertAlmostEqual(se, 0.0)
        self.assertEqual(v10.pooled_slope([([3, 3, 3], [1, 2, 3])]), (1.0, math.inf))
        b, se = v10.pooled_slope([([0, 1, 2, 3], [0, 2, 1, 3])])
        self.assertAlmostEqual(b, 0.8)
        self.assertAlmostEqual(se, math.sqrt(1.8 / 2 / 5))           # rss 1.8 on 4 - 2 points
        self.assertAlmostEqual(v10.toward_one(0.6, 0.1), 0.7)
        self.assertEqual(v10.toward_one(0.95, 0.1), 1.0)
        self.assertAlmostEqual(v10.toward_one(1.3, 0.1), 1.2)
        self.assertEqual(v10.toward_one(1.05, 0.1), 1.0)

    def test_the_build_s_prematch_model_is_shrunk(self):
        class Pre:
            league = (17.5, 17.5)
            scale = 1.0

            def means(self, schedule, **kw):
                return {r["MATCH_CODE"]: (25.0, 20.0) for r in schedule}
        shrunk = v10.ShrunkPrematch(Pre(), self.SHRINK)
        (h, a), = shrunk.means([{"MATCH_CODE": "X"}]).values()
        self.assertAlmostEqual(h + a, 43.0)
        self.assertEqual(shrunk.league, (17.5, 17.5))
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(nb2_prior.Prematch, "exists", return_value=True), \
                    mock.patch.object(nb2_prior.Prematch, "__init__", return_value=None):
                self.assertNotIsInstance(v10.prematch_model(d), v10.ShrunkPrematch)
                with open(os.path.join(d, v10.SHRINK_FILE), "w") as fh:
                    json.dump(self.SHRINK, fh)
                self.assertIsInstance(v10.prematch_model(d), v10.ShrunkPrematch)
                with mock.patch.object(v10, "PRIOR_SHRINK", False):
                    self.assertNotIsInstance(v10.prematch_model(d), v10.ShrunkPrematch)


if __name__ == "__main__":
    unittest.main()
