"""v6: v4 (copied as it was) plus play calling -- clock-stopping or
clock-running plays called by game state, the clock each uses by state,
and the rubber band (sim6.py, v6.py, v6_stream.py)."""

import csv
import datetime as dt
import inspect
import os
import random
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np

from .. import nb2_prior, players, pricer, sim, sim4, sim5, sim6, v6, v6_stream
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
        cls.tables = sim6.Tables.build(_matches(), min_records=20)

    def _two_of(self, common):
        st = sim6.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim6.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 14, 10
        return sim6.simulate(self.tables, st, 300, np.random.default_rng(1), common=common)

    def test_the_same_state_twice_plays_the_same_games(self):
        h, a = self._two_of(True)
        self.assertTrue((h[0] == h[1]).all() and (a[0] == a[1]).all())
        h, a = self._two_of(False)
        self.assertFalse((h[0] == h[1]).all() and (a[0] == a[1]).all())

    def test_the_stream_is_uniform_and_repeatable(self):
        path, step = np.arange(20000), np.full(20000, 3)
        u = sim6._uniform(12345, path, step, 7)
        self.assertTrue(((u >= 0) & (u < 1)).all())
        self.assertAlmostEqual(float(u.mean()), 0.5, delta=0.01)
        self.assertTrue((u == sim6._uniform(12345, path, step, 7)).all())
        self.assertFalse((u == sim6._uniform(12345, path, step, 8)).all())

    def test_v3_is_left_as_it_was(self):
        self.assertNotIn("common", inspect.signature(sim.simulate).parameters)


class TestPlayCalling(unittest.TestCase):
    """The play call (a clock-stopping play or one that keeps the clock
    running), the clock each uses and the rubber band, by game state."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(60)
        cls.tables = sim6.Tables.build(cls.matches, min_records=20)

    def test_the_cells_agree(self):
        for period, clock, lead in ((1, 240, 0), (2, 119.5, -3), (3, 1, 21), (4, 80, 9), (5, 200, -9)):
            self.assertEqual(sim6.cell_index(period, clock, lead),
                             int(sim6._cells_np(np.array([period]), np.array([float(clock)]),
                                                np.array([lead]))[0]))
        self.assertEqual(sim6.cell_index(4, 0.5, 30), sim6.N_CELLS - 1)

    def test_each_bin_lays_out_its_stopping_plays_first(self):
        t = self.tables
        self.assertTrue(((0 <= t.n_stop) & (t.n_stop <= t.count)).all())
        for key in range(0, sim6.N_KEYS, 13):
            a, n, m = t.start[key], t.count[key], t.n_stop[key]
            secs = t.seconds[a:a + n]
            self.assertTrue((secs[:m] <= sim6.STOP_SECONDS).all())
            self.assertTrue((secs[m:] > sim6.STOP_SECONDS).all())

    def test_the_fit_finds_a_planted_state_effect(self):
        import copy
        snaps = [r for rows in self.matches.values() for r in sim6.snap_records(rows)]
        target = sim6.cell_index(3, 150, 5)
        planted = []
        for r in snaps:
            r = dict(r)
            if sim6.cell_index(r["period"], r["clock"], r["margin"]) == target \
                    and r["seconds"] > sim6.STOP_SECONDS:
                r["seconds"] += 8                  # leaders here take 8 s longer
            planted.append(r)
        t = copy.deepcopy(self.tables)
        sim6.fit_play_calling(t, planted)
        sim6.fit_play_calling(self.tables, snaps)
        gained = t.sec_shift[sim6.RUNNING, target] - self.tables.sec_shift[sim6.RUNNING, target]
        n = sum(1 for r in snaps if sim6.cell_index(r["period"], r["clock"], r["margin"]) == target
                and r["seconds"] > sim6.STOP_SECONDS and r["clock"] >= sim6.UNCUT_SECONDS)
        self.assertGreater(n, 10)
        # the 8 s, shrunk toward 0 by SECONDS_PRIOR snaps
        self.assertAlmostEqual(gained, 8 * n / (n + sim6.SECONDS_PRIOR), places=6)
        self.assertLess(abs(t.sec_shift[sim6.STOP, target] - self.tables.sec_shift[sim6.STOP, target]), 1e-9)

    def test_the_band_pulls_in_every_quarter(self):
        self.assertTrue(v6.BAND_BY_QUARTER)
        shift = v6._band_shift(np.array([0.1, 0.2, 0.3]))
        quarter = np.arange(sim6.N_CELLS) // (sim6.CLOCK_CELLS * sim6.LEAD_CELLS)
        ahead = np.arange(sim6.N_CELLS) % sim6.LEAD_CELLS == 4          # ahead by 9+
        behind = np.arange(sim6.N_CELLS) % sim6.LEAD_CELLS == 0
        for q, pull in enumerate((0.1, 0.1, 0.2, 0.3)):             # the fourth included
            self.assertTrue(np.allclose(shift[(quarter == q) & ahead], -pull * 14 / v6.BAND_LEAD))
            self.assertTrue(np.allclose(shift[(quarter == q) & behind], pull * 14 / v6.BAND_LEAD))

    def test_the_band_is_fitted_by_quarter(self):
        import copy
        tables = copy.deepcopy(self.tables)
        grid = v6.PriorGrid.build(tables, n_paths=60)
        items = v6.in_game_states(self.matches, grid)
        pull, got, real = v6.fit_rubber_band(tables, items, rounds=3, n_paths=60)
        self.assertEqual(len(pull), 3)
        # each pull brings its own states' comeback to the real one
        for g, r in zip(got, real):
            self.assertAlmostEqual(g, r, delta=0.08)
        self.assertEqual({st.period for st, *_ in items} >= {1, 2, 3, 4}, True)

    def test_distinct_streams_keep_their_luck_between_calls(self):
        st = sim6.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim6.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 14, 10
        a = sim6.simulate(self.tables, st, 200, np.random.default_rng(1), seed=5, distinct=True)
        b = sim6.simulate(self.tables, st, 200, np.random.default_rng(2), seed=5, distinct=True)
        self.assertTrue(np.array_equal(a[0], b[0]))               # the same luck, call to call
        self.assertFalse(np.array_equal(a[0][0], a[0][1]))        # its own for each state

    def test_the_fourth_quarter_is_left_to_the_end_game_tables(self):
        q4 = np.arange(sim6.N_CELLS) // (sim6.CLOCK_CELLS * sim6.LEAD_CELLS) == 3
        t = self.tables
        self.assertTrue((t.stop_shift[q4] == 0).all() and (t.sec_shift[:, q4] == 0).all()
                        and (t.eff_shift[q4] == 0).all())
        self.assertTrue(np.abs(t.sec_shift[:, ~q4]).sum() > 0)
        self.assertTrue((v6._band_shift(np.array([0.3, 0.3]))[q4] == 0).all())

    def _q3_leader_with_ball(self, n=3000, **shift):
        import copy
        t = copy.deepcopy(self.tables)
        for name, (cells, value) in shift.items():
            arr = getattr(t, name)
            if arr.ndim == 2:
                arr[sim6.RUNNING, cells] += value
            else:
                arr[cells] += value
        st = sim6.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 3, 200.0, sim6.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 30, 17, 10
        h, a = sim6.simulate(t, st, n, np.random.default_rng(1), seed=5)
        return (h + a).mean(), (h - a).mean()

    def test_a_leader_milking_the_clock_leaves_fewer_points(self):
        lead = [c for c in range(sim6.N_CELLS) if c % sim6.LEAD_CELLS >= 3 and c // (sim6.CLOCK_CELLS * sim6.LEAD_CELLS) >= 2]
        plain, _ = self._q3_leader_with_ball()
        milked, _ = self._q3_leader_with_ball(sec_shift=(lead, 15.0))
        self.assertLess(milked, plain - 0.5)

    def test_the_rubber_band_pulls_a_lead_back(self):
        trail = [c for c in range(sim6.N_CELLS) if c % sim6.LEAD_CELLS <= 1]
        lead = [c for c in range(sim6.N_CELLS) if c % sim6.LEAD_CELLS >= 3]
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
            t5 = sim6.Tables.load(path)
        # a game that cannot reach overtime, which v6 plays by the real rules
        st = sim4.Start(2)
        st.period[:], st.clock[:], st.phase[:] = 3, 150.0, sim4.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 40, 38, 0
        a = sim4.simulate(t4, st, 400, np.random.default_rng(1), seed=9)
        b = sim6.simulate(t5, st, 400, np.random.default_rng(1), seed=9)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))


class TestBackedUp(unittest.TestCase):
    """Inside the own 10 each snap draws a safety, a defensive touchdown, a
    turnover or a touchdown from its yard line's own rates; after a safety
    the scorer gets two points and the free kick."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim6.Tables.build(_matches(60), min_records=20)

    def _table(self, **at_the_one):
        import copy
        t = copy.deepcopy(self.tables)
        t.backed = np.zeros((sim6.BACKED_UP + 1, len(sim6.BACKED_OUTCOMES)))
        for name, p in at_the_one.items():
            t.backed[1, sim6.BACKED_OUTCOMES.index(name)] = p
        t.backed_return = None
        return t

    @staticmethod
    def _on_the_one(n=1):
        st = sim6.Start(n)
        st.period[:], st.clock[:], st.phase[:] = 2, 200.0, sim6.SCRIM
        st.team[:], st.y[:], st.down[:], st.dist[:] = 0, 1, 1, 10
        return st

    def test_each_yard_line_keeps_its_own_rate_pulled_toward_the_curve(self):
        records = ([(1, sim6.B_SAFETY, None)] * 12 + [(1, None, None)] * 188
                   + [(5, sim6.B_SAFETY, None)] * 2 + [(5, None, None)] * 198
                   + [(3, sim6.B_TURNOVER, -10)] * 30 + [(3, None, None)] * 170)
        hazard, returns = sim6.fit_backed_up(records)
        self.assertAlmostEqual(hazard[1, sim6.B_SAFETY], 0.06, delta=0.012)
        self.assertAlmostEqual(hazard[5, sim6.B_SAFETY], 0.01, delta=0.008)
        # the 2, never seen, sits on the curve between them
        self.assertGreater(hazard[1, sim6.B_SAFETY], hazard[2, sim6.B_SAFETY])
        self.assertGreater(hazard[2, sim6.B_SAFETY], hazard[5, sim6.B_SAFETY])
        self.assertAlmostEqual(hazard[3, sim6.B_TURNOVER], 0.15, delta=0.03)
        self.assertEqual(list(returns), [-10] * 30)

    def test_the_real_safety_rows_are_found(self):
        rows = [dict(play_kind="SCRIMMAGE", down="2", field_position="2", period="2", offense="TEAM_A",
                     play_messages=""),
                dict(play_kind="PUNT", down="1", field_position="20", period="2", offense="TEAM_A",
                     play_messages="SAFETY_TEAM_B"),
                dict(play_kind="SCRIMMAGE", down="1", field_position="40", period="2", offense="TEAM_B",
                     play_messages="")]
        self.assertEqual(sim6.backed_up_snaps(rows), [(2, sim6.B_SAFETY, None)])

    def test_a_snap_on_the_one_comes_to_its_yard_lines_outcomes(self):
        t = self._table(safety=0.2, def_td=0.1, td=0.1)
        n = 20000
        h, a = sim6.simulate(t, self._on_the_one(), n, np.random.default_rng(1), max_steps=1, seed=9)
        self.assertAlmostEqual(float((a == 2).mean()), 0.2, delta=0.015)
        self.assertAlmostEqual(float((a == 6).mean()), 0.1, delta=0.01)
        self.assertAlmostEqual(float((h == 6).mean()), 0.1, delta=0.01)
        # and nothing else scores: the other snaps stay in the field of play
        self.assertAlmostEqual(float(((h == 0) & (a == 0)).mean()), 0.6, delta=0.015)

    def test_after_a_safety_the_scorer_receives_the_free_kick(self):
        t = self._table(safety=1.0)
        t.safety_kick = np.array([99], dtype=np.int32)     # the free kick lands on the kicker's 1
        stats = {}
        h, a = sim6.simulate(t, self._on_the_one(), 2000, np.random.default_rng(1), max_steps=6,
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
        self.assertIsNone(sim6.Tables.load(path).backed)
        t = self._table(safety=0.3)
        t.save(path)
        self.assertAlmostEqual(float(sim6.Tables.load(path).backed[1, sim6.B_SAFETY]), 0.3)


class TestStrengthSpread(unittest.TestCase):
    """Each simulated game's offenses drawn around the prior: a shared game
    draw that moves the total, each player's own form, the same draw for
    path k of every snapshot of a match, and the fit of both off results."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(40)
        cls.tables = sim6.Tables.build(cls.matches, min_records=20)

    def _kickoff(self, n=1, own=0.0, game=0.0):
        st = sim6.Start(n)
        st.team[:] = 0
        st.kicks_second_half[:] = 1
        st.strength[:] = own
        st.strength_game[:] = game
        return st

    def _spread(self, **kw):
        h, a = sim6.simulate(self.tables, self._kickoff(**kw), 4000, np.random.default_rng(1), seed=5)
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
        h, a = sim6.simulate(self.tables, st, 500, np.random.default_rng(1), seed=5)
        self.assertTrue(np.array_equal(h[0], h[1]) and np.array_equal(a[0], a[1]))
        h2, _ = sim6.simulate(self.tables, st, 500, np.random.default_rng(9), seed=5)
        self.assertTrue(np.array_equal(h, h2))

    def test_the_draw_keeps_the_expected_points(self):
        import copy
        tables = copy.deepcopy(self.tables)
        _, _, slope = v6.fixed_strength_spread(tables, n_paths=600)
        tables.strength_theta, tables.strength_slope = v6.FORM_GRID.copy(), slope
        tables.strength_game = 0.04
        theta, own, game = sim6.strength_draw(tables, [0.0, 0.0], [0.03, 0.0])
        self.assertTrue(own[0] > 0 and own[1] == 0 and game[0] > 0)
        st = self._kickoff()
        fixed = sim6.simulate(tables, st, 6000, np.random.default_rng(2), common=False)[0].mean()
        st.theta[0], st.strength[0], st.strength_game[0] = theta, own, game
        drawn = sim6.simulate(tables, st, 6000, np.random.default_rng(2), common=False)[0].mean()
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
        game, league, form = v6.fit_form(sides, np.zeros(3), np.zeros(4))
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
        t = sim6.Tables()
        theta, own, game = sim6.strength_draw(t, [0.1, -0.1], [0.05, 0.05])
        self.assertTrue(np.allclose(theta, [0.1, -0.1]) and not own.any() and not game.any())


class TestLateGame(unittest.TestCase):
    """v6: a trailing side's late 4th downs from a fitted table, and big leads as situations of
    their own."""

    @classmethod
    def setUpClass(cls):
        cls.matches = _matches(60)
        cls.tables = sim6.Tables.build(cls.matches, min_records=20)

    def _fourth(self, behind):
        st = sim6.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 4, 60.0, sim6.SCRIM
        st.team[:], st.down[:], st.dist[:], st.y[:] = 0, 4, 8, 80
        st.home[:], st.away[:] = 10, 10 + behind
        return st

    def test_the_table_sets_what_a_trailing_side_does(self):
        import copy
        t = copy.deepcopy(self.tables)
        t.late_fourth = np.zeros_like(t.late_fourth)
        t.late_fourth[..., 1] = 1.0
        h, _ = sim6.simulate(t, self._fourth(10), 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertGreater(float((h == 13).mean()), 0.8)
        t.late_fourth[..., 1], t.late_fourth[..., 0] = 0.0, 1.0
        h, _ = sim6.simulate(t, self._fourth(10), 400, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertEqual(float((h == 13).mean()), 0.0)

    def test_the_fitted_table_is_shaped_and_sums_to_one(self):
        lf = self.tables.late_fourth
        self.assertEqual(lf.shape, (len(sim6.LATE_DEFICITS) + 1, len(sim6.KICK_RANGES) + 1, 3))
        self.assertTrue(np.allclose(lf.sum(2), 1.0))

    def test_with_both_off_v6_plays_as_v5(self):
        import copy
        big = sim6.BIG_LEAD
        sim6.BIG_LEAD = None
        try:
            t6 = sim6.Tables.build(self.matches, min_records=20)
        finally:
            sim6.BIG_LEAD = big
        t6.late_fourth = None
        t5 = sim5.Tables.build(self.matches, min_records=20)
        st = sim6.Start(3)
        st.period[:], st.clock[:], st.phase[:] = 4, 150.0, sim6.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 50, [3, 17, 10], [10, 10, 30]
        a = sim6.simulate(t6, st, 300, np.random.default_rng(1), seed=9)
        b = sim5.simulate(t5, st, 300, np.random.default_rng(1), seed=9)
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_big_leads_have_their_own_situations(self):
        self.assertEqual(sim6.mode_of(3, 100.0, 5), 3)
        self.assertEqual(sim6.mode_of(3, 100.0, 12), 8)
        self.assertEqual(sim6.mode_of(4, 60.0, 12), 9)
        self.assertEqual(sim6.mode_of(4, 60.0, 12, None), 4)
        modes = sim6._modes_np(np.array([3, 4, 4]), np.array([100.0, 60.0, 60.0]), np.array([12, 12, 3]), 9)
        self.assertEqual(list(modes), [8, 9, 4])
        self.assertEqual(self.tables.big_lead, sim6.BIG_LEAD)


class TestOvertime(unittest.TestCase):
    """Overtime as the feed shows it played: each side has the ball once,
    then the game ends the moment one side leads; one or two behind after
    a touchdown once the other side has had the ball, a side goes for two."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim6.Tables.build(_matches(60), min_records=20)

    def test_a_game_ends_once_both_have_had_the_ball_and_one_leads(self):
        st = sim6.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 240.0, sim6.KICK
        st.team[:], st.home[:], st.away[:] = 0, 20, 20
        stats = {}
        h, a = sim6.simulate(self.tables, st, 4000, np.random.default_rng(1), seed=3, stats=stats)
        margin, points = np.abs(h - a).ravel(), (h + a - 40).ravel()
        self.assertGreater(stats["ot_decided"], 0)
        # decided by the first lead after both possessions: never by more
        # than a converted touchdown bar a defensive score, and no shoot-outs
        self.assertGreater(float((margin <= 8).mean()), 0.97)
        self.assertLess(float((points >= 17).mean()), 0.03)

    def test_one_behind_after_a_touchdown_it_goes_for_two(self):
        st = sim6.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 5, 150.0, sim6.CONV
        st.team[:], st.home[:], st.away[:] = 1, 27, 26      # the away side's six: one behind
        h, a = sim6.simulate(self.tables, st, 3000, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertTrue(set(np.unique(a)) <= {26, 28})        # never the kick to tie
        st.home[:] = 20                                       # six to lead: kicks as usual
        h, a = sim6.simulate(self.tables, st, 3000, np.random.default_rng(1), seed=3, max_steps=1)
        self.assertIn(27, set(np.unique(a)))


class TestSecondHalfKick(unittest.TestCase):
    """The opening receiver kicks the second half; where the feed did not
    say who received, a coin per path."""

    @classmethod
    def setUpClass(cls):
        cls.tables = sim6.Tables.build(_matches(60), min_records=20)

    def _margin(self, kicks):
        st = sim6.Start(1)
        st.period[:], st.clock[:], st.phase[:] = 2, 0.0, sim6.SCRIM
        st.team[:], st.y[:], st.home[:], st.away[:] = 0, 30, 10, 10
        st.kicks_second_half[:] = kicks
        h, a = sim6.simulate(self.tables, st, 4000, np.random.default_rng(1), seed=21)
        return float((h - a).mean())

    def test_an_unknown_opening_is_a_coin(self):
        home_kicks, away_kicks, unknown = self._margin(0), self._margin(1), self._margin(-1)
        self.assertLess(home_kicks, away_kicks)             # receiving the half is worth points
        self.assertLess(home_kicks, unknown)
        self.assertLess(unknown, away_kicks)
        self.assertEqual(int(sim6.Start(1).kicks_second_half[0]), -1)

    def test_a_snapshot_without_the_opening_says_so(self):
        state = mock.Mock(pending_conversion=None, down=None, offense=sim6_home(), period=1,
                          clock_seconds=200.0, home_score=0, away_score=0, opening_receiver=None)
        self.assertEqual(v6.start_from(state)["kicks_second_half"], -1)
        state.opening_receiver = sim6_home()
        self.assertEqual(v6.start_from(state)["kicks_second_half"], 0)


def sim6_home():
    return v6.HOME


def sim6_profile():
    from .. import players
    return players.Profile()


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.matches = _with_handles(_matches(40))
        # the in-play shift and the quarter-start fit are off by default; the
        # build is tested with them on
        saved = v6.IN_PLAY_FIT, v6.QUARTER_START_FIT
        v6.IN_PLAY_FIT = v6.QUARTER_START_FIT = True
        try:
            v6.build(cls.matches, cls.tmp.name, grid_paths=60, verbose=False)
        finally:
            v6.IN_PLAY_FIT, v6.QUARTER_START_FIT = saved
        cls.tables = sim6.Tables.load(os.path.join(cls.tmp.name, "v6tables.npz"))
        cls.grid = v6.PriorGrid.load(os.path.join(cls.tmp.name, "v6grid.npz"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_writes_the_model_and_the_profiles(self):
        for name in ("v6tables.npz", "v6grid.npz", "v6players.json"):
            self.assertTrue(os.path.exists(os.path.join(self.tmp.name, name)), name)
        book = v6.players_book(self.tmp.name)
        self.assertEqual(set(book.players), {"ALPHA", "BRAVO", "CHARLIE", "DELTA"})

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
        got = {q: pts for q, _, _, pts in v6.quarter_start_states({"AFX": rows}, self.grid)}
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
        got = {q: pts for q, _, _, pts in v6.late_start_states({"AFX": rows}, self.grid)}
        self.assertEqual(got, {2: 7, 4: 7})

    def test_the_build_fits_the_quarters_and_the_two_minute_drills(self):
        import copy
        self.assertFalse(v6.QUARTER_START_FIT)          # off by default (see v6.py)
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = 0.0          # this class's build has the in-play shift on too
        quarters, lates = v6.fit_quarter_levels(
            tables, v6.quarter_start_states(self.matches, self.grid),
            v6.late_start_states(self.matches, self.grid), rounds=1, n_paths=200)
        for real, got in list(quarters.values()) + list(lates.values()):
            self.assertAlmostEqual(got, real, delta=max(0.6, 0.12 * real))
        self.assertEqual(set(lates), {2, 4})
        self.assertTrue(np.any(self.tables.late_theta))

    def test_the_two_minute_level_is_only_the_last_two_minutes(self):
        import copy
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = 0.0
        tables.late_theta[:] = 0.0
        st = sim6.Start(1)
        st.period[:], st.clock[:], st.phase[:], st.y[:] = 2, 239.0, sim6.SCRIM, 30
        plain = sim6.simulate(tables, st, 3000, np.random.default_rng(1), seed=4, max_steps=2)
        tables.late_theta[2] = 1.0
        early = sim6.simulate(tables, st, 3000, np.random.default_rng(1), seed=4, max_steps=2)
        self.assertTrue(np.array_equal(plain[0], early[0]))    # 4:00 left: untouched
        st.clock[:] = 100.0
        tables.late_theta[2] = 0.0
        a = sim6.simulate(tables, st, 3000, np.random.default_rng(1), seed=4)
        tables.late_theta[2] = 1.0
        b = sim6.simulate(tables, st, 3000, np.random.default_rng(1), seed=4)
        self.assertGreater(float((b[0] + b[1]).mean()), float((a[0] + a[1]).mean()))

    def test_the_quarter_fit_on_real_states_converges_when_asked(self):
        # on fixed states and thetas the fit brings the quarters to what
        # games scored
        import copy
        tables = copy.deepcopy(self.tables)
        items = v6.quarter_start_states(self.matches, self.grid)
        v6.fit_period_theta_states(tables, items, rounds=6, n_paths=300)
        _, got, real = v6.fit_period_theta_states(tables, items, rounds=1, n_paths=300)
        for q in real:
            self.assertAlmostEqual(got[q], real[q], delta=max(0.4, 0.08 * real[q]))

    def test_the_rest_of_game_fit_leaves_what_games_really_left(self):
        import copy
        tables = copy.deepcopy(self.tables)
        items = v6.rest_of_game_states(self.matches, self.grid, every=3)
        q3 = sim6.SEGMENTS.index("Q3")
        self.assertTrue({it[0][0] for it in items} >= {1, 2, 3, q3, 5, 6})
        # games that left a point fewer than they did from every state in
        # the third quarter: its in-play scoring comes down, most of the way
        lower = [(c, st, th, pts - 1 if c[0] == q3 else pts) for c, st, th, pts in items]
        before = tables.inplay_theta.copy()
        segs, bands = v6.fit_rest_of_game(tables, lower, n_paths=60, rounds=4, max_states=600)
        real, simulated_before, after = segs[q3]
        self.assertLess(abs(after - real), 0.5 * abs(simulated_before - real))
        self.assertLess(tables.inplay_theta[q3].mean(), before[q3].mean() - 0.02)
        for (g, band), (r, b, a) in bands.items():
            self.assertIn(band, range(sim6.N_BANDS))
        # and the kickoff scoring, the pre-match's, is untouched
        self.assertTrue((tables.period_theta == self.tables.period_theta).all())

    def test_the_in_play_shift_is_only_in_play(self):
        import copy
        tables = copy.deepcopy(self.tables)
        tables.inplay_theta[:] = -0.5
        st = sim6.Start(1)
        st.period[:], st.clock[:], st.phase[:], st.y[:] = 3, 200.0, sim6.SCRIM, 30
        runs = {flag: sim6.simulate(tables, st, 3000, np.random.default_rng(1), seed=3, in_play=flag)
                for flag in (False, True)}
        total = {flag: float((h + a).mean()) for flag, (h, a) in runs.items()}
        self.assertLess(total[True], total[False] - 1.0)
        self.assertFalse(v6.IN_PLAY_FIT)
        # a build with it on fitted it and saved it
        self.assertTrue(np.abs(self.tables.inplay_theta[1:7]).sum() > 0)

    def test_the_in_play_shift_moves_the_total_and_not_the_margin(self):
        import copy
        from .. import playover
        rows = v6.resolve_sides(list(self.matches.values())[0])
        states, messages = [], []
        for r in rows[10:40:5]:
            st, _ = playover.state_for(r)
            if st is not None:
                states.append(st)
                messages.append(int(r["message"]))
        prof = (sim6_profile(), sim6_profile())
        books = {}
        for name, shift in (("plain", 0.0), ("shifted", -0.6)):
            tables = copy.deepcopy(self.tables)
            tables.inplay_theta[:] = shift
            books[name] = v6.price_states(tables, (0.0, 0.0), v6.Variant("v6"), [], True, states,
                                          messages, prof, 400, np.random.default_rng(2), seed=11)
        x = np.arange(len(books["plain"][0][1]))
        for (mp0, tp0), (mp1, tp1) in zip(books["plain"], books["shifted"]):
            self.assertTrue(np.allclose(mp0, mp1))
            self.assertLess((tp1 * x).sum(), (tp0 * x).sum())

    def test_nothing_of_gameplais_is_read(self):
        # every prod price column moved far off: v6's priors, states and
        # prices come out the same
        moved = {c: [dict(r, prematch_line_52="-20.5", prematch_prob_52="0.9",
                          prematch_line_54="80.5", prematch_prob_54="0.9", prematch_prob_50="0.99",
                          line_52="-20.5", prob_52="0.9", line_54="80.5", prob_54="0.9",
                          prob_50="0.99") for r in rows]
                 for c, rows in self.matches.items()}
        for fn, k in ((v6.in_game_states, 1), (v6.rest_of_game_states, 2),
                      (v6.quarter_start_states, 2)):
            a, b = fn(self.matches, self.grid), fn(moved, self.grid)
            self.assertTrue(a)
            self.assertEqual([x[k] for x in a], [x[k] for x in b])
            self.assertTrue(all(tuple(x[k]) == v6.LEAGUE_THETA for x in a))
        rows = next(iter(self.matches.values()))
        a = v6_stream.match_books(self.tables, self.grid, v6.Variant("v6"), rows, 60,
                                  np.random.default_rng(1))
        b = v6_stream.match_books(self.tables, self.grid, v6.Variant("v6"), moved[rows[0]["match_code"]],
                                  60, np.random.default_rng(1))
        self.assertTrue(a)
        for (_, mp1, tp1), (_, mp2, tp2) in zip(a, b):
            self.assertTrue(np.array_equal(mp1, mp2) and np.array_equal(tp1, tp2))
        for gone in ("prior_lines", "recent_total_shade"):
            self.assertFalse(hasattr(v6, gone))
        self.assertFalse(hasattr(v6.PriorGrid, "fit"))

    def test_the_build_needs_history_on_the_command_line(self):
        from ..__main__ import main
        path = os.path.join(self.tmp.name, "snaps_cli.csv")
        with open(path, "w") as fh:
            fh.write("match_code\n")
        with self.assertRaises(SystemExit) as caught:
            main(["v6-build", path, "--out", os.path.join(self.tmp.name, "cli")])
        self.assertIn("--history", str(caught.exception))

    def test_handles_come_off_the_export(self):
        rows = next(iter(self.matches.values()))
        self.assertEqual(v6.handles_of(rows), (rows[0]["home_handle"], rows[0]["away_handle"]))
        self.assertIsNone(v6.handles_of([{"home_handle": "", "away_handle": None}]))

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
        graded, _ = v6.run(path, os.path.join(self.tmp.name, "v6tables.npz"),
                           os.path.join(self.tmp.name, "v6grid.npz"),
                           [v6.Variant("v6"), v6.Variant("plain", profiles=False, pace=False)],
                           n_paths=100, workers=1)
        self.assertGreater(len(graded), 50)
        for _, row, probs in graded:
            self.assertTrue(0 < probs["v6"] < 1 and 0 < probs["plain"] < 1)

    def test_the_stream_finds_the_model_and_quotes(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        prod = [(code, 54, None, 50.0, 2.0, "Total points over 30.5", int(r["message"]), "OPEN", "true")
                for r in rows]
        quotes = v6_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
        self.assertTrue(quotes)
        self.assertTrue(all(q[1] == 54 and 0 < q[3] < 100 for q in quotes))

    def test_fit_means_finds_the_thetas_that_score_the_points(self):
        g = self.grid
        x_m = np.arange(g.margin.shape[-1]) - v6.MARGIN_MAX
        x_t = np.arange(g.total.shape[-1])
        for i, j in ((8, 8), (4, 11), (12, 6)):
            margin, total = (g.margin[i, j] * x_m).sum(), (g.total[i, j] * x_t).sum()
            got = v6.fit_means(g, (total + margin) / 2, (total - margin) / 2)
            self.assertAlmostEqual(got[0], g.grid[i], delta=0.02)
            self.assertAlmostEqual(got[1], g.grid[j], delta=0.02)
        low, high = v6.fit_means(g, 14, 14), v6.fit_means(g, 24, 24)
        self.assertLess(sum(low), sum(high))

    def test_a_fixed_seed_prices_a_match_the_same_every_time(self):
        rows = next(iter(self.matches.values()))
        a = v6_stream.match_books(self.tables, self.grid, v6.Variant("v6"), rows, 80,
                                  np.random.default_rng(1), means=(20.0, 14.0))
        b = v6_stream.match_books(self.tables, self.grid, v6.Variant("v6"), rows, 80,
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
        args = (self.tables, self.grid, v6.Variant("v6"))
        a = v6_stream.match_books(*args, rows, 80, np.random.default_rng(1), means=(20.0, 14.0))
        b = v6_stream.match_books(*args, other, 80, np.random.default_rng(1), means=(20.0, 14.0))
        for (_, mp1, tp1), (_, mp2, tp2) in zip(a, b):
            self.assertTrue(np.array_equal(mp1, mp2) and np.array_equal(tp1, tp2))

    def _snapshot_file(self):
        path = os.path.join(self.tmp.name, "snaps.csv")
        if not os.path.exists(path):
            self.test_grades_a_snapshot_file_with_profiles()
        return path

    def test_a_model_with_nb2_takes_its_prior_from_there(self):
        path = self._snapshot_file()
        tables, grid = (os.path.join(self.tmp.name, n) for n in ("v6tables.npz", "v6grid.npz"))
        codes = sorted(self.matches)[:6]

        class Fake:
            league = (17.0, 17.0)

            def __init__(self, level):
                self.level = level

            def means(self, schedule, n_sims=0):
                return {r["MATCH_CODE"]: self.level for r in schedule}

        history = [{"MATCH_CODE": c} for c in codes]
        with mock.patch.object(v6, "prematch_model", return_value=Fake((12.0, 12.0))):
            with self.assertRaises(SystemExit) as caught:
                v6.run(path, tables, grid, [v6.Variant("v6")], matches=codes, workers=1)
            self.assertIn("--history", str(caught.exception))
            low, _ = v6.run(path, tables, grid, [v6.Variant("v6")], matches=codes, n_paths=100,
                            workers=1, history=history)
        with mock.patch.object(v6, "prematch_model", return_value=Fake((26.0, 26.0))):
            high, _ = v6.run(path, tables, grid, [v6.Variant("v6")], matches=codes, n_paths=100,
                             workers=1, history=history)
        over = lambda graded: np.mean([p["v6"] for _, r, p in graded if r.market_id == 54])
        self.assertLess(over(low), over(high))
        # no prediction for a match: skipped, never priced off prod
        _, skipped = v6.run(path, tables, grid, [v6.Variant("v6")], matches=codes, n_paths=50,
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

        with mock.patch.object(v6, "prematch_model", return_value=Fake()):
            with self.assertRaises(SystemExit):
                v6_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
            quotes = v6_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50,
                                                  workers=1, match_info=[])
        self.assertTrue(quotes)
        self.assertEqual(Fake.asked, [])


    def test_the_even_line_is_the_half_point_nearest_even_money(self):
        pmf = np.zeros(80)
        pmf[[38, 41, 44, 47]] = [0.2, 0.25, 0.3, 0.25]
        # P(> 38.5) = 0.8, P(> 41.5 .. 43.5) = 0.55, P(> 44.5) = 0.25: of the
        # three 0.55 lines, the one nearest the mean (42.8)
        self.assertEqual(v6.even_line(pmf, 0), 42.5)
        margin = np.zeros(2 * v6.MARGIN_MAX + 1)
        margin[v6.MARGIN_MAX + np.array([-3, 3, 7])] = [0.3, 0.45, 0.25]
        # -2.5 .. 2.5 all give P(home by more) = 0.7: the one nearest the mean (2.2)
        self.assertEqual(v6.even_line(margin, v6.MARGIN_MAX), 2.5)
        line = v6.even_line(margin, v6.MARGIN_MAX)
        self.assertLessEqual(abs(v6.market_prob(52, line, margin, pmf) - 0.5), 0.2 + 1e-9)

    def test_the_stream_moves_its_own_line_with_the_game(self):
        rows = next(iter(self.matches.values()))
        code = rows[0]["match_code"]
        prod = [(code, m, None, 50.0, 2.0, v6_stream.description(m, 30.5 if m in (54, 55) else 0.5),
                 int(r["message"]), "OPEN", "true") for r in rows for m in (52, 53, 54, 55)]
        own = v6_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=100, workers=1)
        at_prod = v6_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=100,
                                               workers=1, lines="prod")
        parse = v6_stream._parse_line
        self.assertEqual({parse(q[5]) for q in at_prod if q[1] == 54}, {30.5})
        totals = [(q[6], parse(q[5])) for q in own if q[1] == 54]
        self.assertGreater(len({t for _, t in totals}), 1)         # it moves with the game
        self.assertTrue(all(t % 1 == 0.5 for _, t in totals))
        # never below the points already on the board
        board = {int(r["message"]): int(r["score_p1"] or 0) + int(r["score_p2"] or 0) for r in rows}
        for message, line in totals:
            on_board = board[max(m for m in board if m <= message)]
            self.assertGreater(line, on_board - 1)

    def test_a_missing_model_says_how_to_build_it(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(SystemExit) as caught:
                v6_stream.model_paths(empty)
        self.assertIn("v6-build", str(caught.exception))


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

    def test_v6_build_with_history_writes_the_prematch_model(self):
        from .test_v3 import _matches
        matches = _matches(12)
        out = os.path.join(self.tmp.name, "v6")
        v6.build(matches, out, grid_paths=40, verbose=False, history=self.history(),
                 before=self.BEFORE)
        self.assertIsNotNone(v6.prematch_model(out))
        self.assertIsNotNone(v6.prematch_model(os.path.join(out, "v6tables.npz")))

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


if __name__ == "__main__":
    unittest.main()
