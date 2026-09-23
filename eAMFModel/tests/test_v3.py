"""v3: the play-by-play simulation (sim.py) and its pricing (v3.py), on
synthetic PLAY_OVER exports shaped like eAMFCalibrator scouting's."""

import random
import unittest

import numpy as np

from .. import playover, sim, v3
from ..pricer import AWAY, HOME, GameState

MARKET_COLUMNS = {f"{k}_{m}": "" for m in (50, 51, 52, 53, 54, 55)
                  for k in ("line", "prob", "live", "outcome")}


def _fake_match(code, rng, lead_seconds=28.0, strength=(0.0, 0.0)):
    """A toy game on the real clock. The offense ahead in the second half
    uses `lead_seconds` a snap (it milks the clock), everyone else 16."""
    rows = []
    msg = [0]
    score = {"TEAM_A": 0, "TEAM_B": 0}
    other = {"TEAM_A": "TEAM_B", "TEAM_B": "TEAM_A"}

    def emit(period, clock, kind, offense, down, dist, field, messages=""):
        msg[0] += 7
        r = dict(MARKET_COLUMNS, match_code=code, message=str(msg[0]), period=str(period),
                 clock_seconds=str(max(0, int(clock))), play_kind=kind, offense=offense,
                 down=str(down), distance=str(dist), field_position=str(field),
                 score_p1=str(score["TEAM_A"]), score_p2=str(score["TEAM_B"]),
                 play_messages=messages, team_a_side="home", opening_offense="TEAM_A",
                 prematch_line_52="0.5", prematch_prob_52="0.5", prematch_line_54="30.5",
                 prematch_prob_54="0.5", prematch_prob_50="0.5")
        rows.append(r)

    offense = "TEAM_A"
    for period in (1, 2, 3, 4):
        clock = 240.0
        if period in (1, 3):
            offense = "TEAM_A" if period == 1 else "TEAM_B"
            clock -= 4
            emit(period, clock, "KICKOFF", offense, 1, 10, 25)
        y, down, dist = 25, 1, 10
        while clock > 0:
            margin = score[offense] - score[other[offense]]
            if down == 4:
                if y >= 60:
                    clock -= 5
                    good = rng.random() < 0.8
                    if good:
                        score[offense] += 3
                    emit(period, clock, "FIELD_GOAL", offense, 1, 10, 35,
                         ("FIELD_GOAL_GOOD_" if good else "FIELD_GOAL_MISSED_") + offense)
                else:
                    clock -= 10
                    offense = other[offense]
                    y, down, dist = 30, 1, 10
                    emit(period, clock, "PUNT", offense, 1, 10, y, "POSSESSION_" + offense)
                    continue
                if clock <= 0:
                    break
                clock -= 4
                kicker = offense
                offense = other[offense]
                y, down, dist = 25, 1, 10
                emit(period, clock, "KICKOFF", offense, 1, 10, y, f"KICKOFF_{kicker}|POSSESSION_{offense}")
                continue
            q = strength[0] if offense == "TEAM_A" else strength[1]
            gain = int(rng.choice([-2, 0, 0, 3, 4, 6, 8, 12, 15, 25]) * (1 + q))
            clock -= lead_seconds if (period >= 3 and margin > 0) else 16.0
            if rng.random() < 0.02:
                offense = other[offense]
                y, down, dist = 100 - y, 1, 10
                emit(period, clock, "SCRIMMAGE", offense, down, dist, y, "POSSESSION_" + offense)
                continue
            y += gain
            if y >= 100:
                score[offense] += 6
                emit(period, clock, "TOUCHDOWN", offense, 1, 10, 85, "TOUCHDOWN_" + offense)
                score[offense] += 1
                emit(period, clock, "CONVERSION", offense, 1, 10, 35, "EXTRA_POINT_GOOD_" + offense)
                clock -= 4
                kicker = offense
                offense = other[offense]
                y, down, dist = 25, 1, 10
                emit(period, clock, "KICKOFF", offense, 1, 10, y, f"KICKOFF_{kicker}|POSSESSION_{offense}")
                continue
            y = max(1, y)
            if gain >= dist:
                down, dist = 1, min(10, 100 - y)
            else:
                down, dist = down + 1, dist - gain
            emit(period, clock, "SCRIMMAGE", offense, down, dist, y)
    for r in rows:
        r["final_p1"], r["final_p2"] = str(score["TEAM_A"]), str(score["TEAM_B"])
    return rows


def _matches(n=120, seed=3, **kw):
    rng = random.Random(seed)
    return {f"AF{i:03d}": _fake_match(f"AF{i:03d}", rng, **kw) for i in range(n)}


class TestTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matches = _matches()
        cls.tables = sim.Tables.build(cls.matches, min_records=20)

    def test_every_situation_has_results(self):
        t = self.tables
        self.assertEqual(len(t.start), sim.N_KEYS)
        self.assertTrue((t.count > 0).all())
        self.assertTrue(((t.success >= 0) & (t.success <= 1)).all())

    def test_bins_run_worst_to_best(self):
        t = self.tables
        for key in range(0, sim.N_KEYS, 17):
            s, n = t.start[key], t.count[key]
            # turnovers first, then gains (ordered against the distance to go)
            is_gain = (t.kind[s:s + n] == sim.GAIN).astype(int)
            self.assertTrue((np.diff(is_gain) >= 0).all())

    def test_milking_is_in_the_leader_tables(self):
        t = self.tables
        lead = sim.key_index(sim.MODES.index("lead"), 1, 10, 30)
        even = sim.key_index(sim.MODES.index("q1"), 1, 10, 30)
        mean = lambda k: t.seconds[t.start[k]:t.start[k] + t.count[k]].mean()
        self.assertGreater(mean(lead), mean(even) + 5)

    def test_rows_missing_fields_are_skipped_not_fatal(self):
        # real exports have rows with no quarter (a feed that missed the
        # quarter-start status), no clock, no scores or no down and distance
        matches = _matches(30, seed=9)
        rng = random.Random(4)
        fields = ["period", "clock_seconds", "down", "distance", "field_position", "offense",
                  "score_p1", "score_p2", "play_messages"]
        for rows in matches.values():
            for r in rows:
                if rng.random() < 0.2:
                    for f in rng.sample(fields, 3):
                        r[f] = ""
        next(iter(matches.values()))[0]["period"] = ""
        for r in list(matches.values())[1]:
            r["period"] = ""
        t = sim.Tables.build(matches, min_records=20)
        self.assertGreater(t.n_snaps, 100)
        self.assertEqual(len(sim.quarter_points(matches)), 4)

    def test_save_and_load(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.npz")
            self.tables.save(path)
            back = sim.Tables.load(path)
            self.assertTrue((back.gain == self.tables.gain).all())
            self.assertTrue((back.go_for_two == self.tables.go_for_two).all())


class TestSimulate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = sim.Tables.build(_matches(), min_records=20)

    def _start(self, n=1, **fields):
        st = sim.Start(n)
        for k, v in fields.items():
            getattr(st, k)[:] = v
        return st

    def test_games_finish_and_are_settled(self):
        st = self._start(team=1)
        stats = {}
        h, a = sim.simulate(self.tables, st, 500, np.random.default_rng(0), stats=stats)
        self.assertTrue((h >= 0).all() and (a >= 0).all())
        self.assertGreater((h + a).mean(), 5)
        # overtime is played while level (up to MAX_OT periods)
        self.assertLess((h == a).mean(), 0.02)

    def test_efficiency_means_points(self):
        st = self._start(2, team=1)
        st.theta[0] = (0.3, 0.0)
        st.theta[1] = (-0.3, 0.0)
        h, a = sim.simulate(self.tables, st, 1500, np.random.default_rng(1))
        self.assertGreater(h[0].mean(), h[1].mean() + 2)

    def test_leader_kneels_it_out(self):
        st = self._start(period=4, clock=30.0, phase=sim.SCRIM, team=0, down=1, dist=10, y=40,
                         home=17, away=14)
        h, a = sim.simulate(self.tables, st, 300, np.random.default_rng(2))
        self.assertTrue((h == 17).all() and (a == 14).all())

    def test_efficient_leader_bleeds_the_clock(self):
        # up 7 with the ball, three minutes left: the better the leader moves
        # the chains, the less time the other side gets -- fewer points for it
        st = self._start(2, period=4, clock=180.0, phase=sim.SCRIM, team=0, down=1, dist=10,
                         y=25, home=21, away=14)
        st.theta[0] = (0.4, 0.0)
        st.theta[1] = (-0.4, 0.0)
        h, a = sim.simulate(self.tables, st, 3000, np.random.default_rng(3))
        self.assertLess((a[0] - 14).mean(), (a[1] - 14).mean())

    def test_conversion_then_kickoff(self):
        st = self._start(period=2, clock=100.0, phase=sim.CONV, team=0, home=6, away=0)
        stats = {}
        h, a = sim.simulate(self.tables, st, 200, np.random.default_rng(4), stats=stats)
        self.assertTrue((h >= 6).all())


class TestV3Pricing(unittest.TestCase):
    def test_push_is_graded_out(self):
        pmf = np.zeros(2 * v3.MARGIN_MAX + 1)
        pmf[v3.MARGIN_MAX + 3] = 0.5          # margin 3
        pmf[v3.MARGIN_MAX - 3] = 0.5          # margin -3
        self.assertAlmostEqual(float(v3.prob_above(pmf, v3.MARGIN_MAX, 3.0)), 0.0)
        self.assertAlmostEqual(float(v3.prob_above(pmf, v3.MARGIN_MAX, 2.5)), 0.5)
        pmf[v3.MARGIN_MAX + 3] = 0.25
        pmf[v3.MARGIN_MAX + 7] = 0.25
        self.assertAlmostEqual(float(v3.prob_above(pmf, v3.MARGIN_MAX, 3.0)), 0.25 / 0.75)

    def test_markets_mirror(self):
        rng = np.random.default_rng(0)
        h, a = rng.integers(0, 40, 5000), rng.integers(0, 40, 5000)
        m, t = v3._distributions(h, a)
        for line in (-3.5, 2.5, 7.0):
            self.assertAlmostEqual(float(v3.market_prob(52, line, m, t)),
                                   1 - float(v3.market_prob(53, -line, m, t)), places=9)
        self.assertAlmostEqual(float(v3.market_prob(54, 40.5, m, t)),
                               1 - float(v3.market_prob(55, 40.5, m, t)), places=9)

    def test_prior_fit_finds_the_favourite(self):
        tables = sim.Tables.build(_matches(60), min_records=20)
        grid = v3.PriorGrid.build(tables, n_paths=300, grid=np.round(np.linspace(-0.4, 0.4, 5), 3))
        th = grid.fit(6.5, 0.5, 30.5, 0.5, 0.75)
        self.assertGreater(th[0], th[1])

    def test_efficiency_moves_with_first_downs_not_points(self):
        tables = sim.Tables.build(_matches(60), min_records=20)
        key = sim.key_index(0, 1, 10, 30)
        up, down = v3.Efficiency(tables, (0.0, 0.0), 50), v3.Efficiency(tables, (0.0, 0.0), 50)
        for _ in range(10):
            up.add(0, key, True)
            down.add(0, key, False)
        self.assertGreater(up.theta()[0], 0)
        self.assertLess(down.theta()[0], 0)
        self.assertEqual(up.theta()[1], 0)      # the other side is untouched
        self.assertLess(up.sd()[0], up.sd()[1])

    def test_start_from_a_touchdown_and_a_conversion(self):
        base = dict(period=2, elapsed_in_period=0.0, home_score=6, away_score=0, clock_seconds=90.0,
                    opening_receiver=HOME)
        td = v3.start_from(GameState(offense=AWAY, pending_conversion=HOME, **base))
        self.assertEqual((td["phase"], td["team"]), (sim.CONV, 0))
        kick = v3.start_from(GameState(offense=AWAY, **base))
        self.assertEqual((kick["phase"], kick["team"]), (sim.KICK, 0))
        snap = v3.start_from(GameState(offense=AWAY, down=3, distance=4, field_position=60,
                                       snap_confirmed=True, **base))
        self.assertEqual((snap["phase"], snap["team"], snap["y"]), (sim.SCRIM, 1, 60))
        self.assertEqual(td["kicks_second_half"], 0)

    def test_grades_a_snapshot_file(self):
        import csv
        import os
        import tempfile
        matches = _matches(24)
        rng = random.Random(1)
        for rows in matches.values():
            for r in rows[::5]:
                total = int(r["final_p1"]) + int(r["final_p2"])
                r.update(line_54="30.5", prob_54=str(round(rng.uniform(0.3, 0.7), 3)), live_54="1",
                         outcome_54=str(int(total > 30.5)))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "snaps.csv")
            fields = list(next(iter(matches.values()))[0])
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fields)
                w.writeheader()
                for rows in matches.values():
                    w.writerows(rows)
            tables_path, grid_path = v3.build(playover.load(path), d, grid_paths=100, verbose=False)
            graded, skipped = v3.run(path, tables_path, grid_path,
                                     [v3.Variant("v3", react=False), v3.Variant("react", kappa=50)],
                                     n_paths=100, workers=1)
        self.assertGreater(len(graded), 50)
        for _, row, probs in graded:
            self.assertEqual(row.market_id, 54)
            self.assertTrue(0 < probs["v3"] < 1 and 0 < probs["react"] < 1)


if __name__ == "__main__":
    unittest.main()
