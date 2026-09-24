"""v4: v3's simulation with in-game quarter calibration, common random
numbers and player profiles (sim4.py, v4.py, v4_stream.py)."""

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

from .. import nb2_prior, sim, sim4, v4, v4_stream
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

    def test_fit_means_finds_the_thetas_that_score_the_points(self):
        g = self.grid
        x_m = np.arange(g.margin.shape[-1]) - v4.MARGIN_MAX
        x_t = np.arange(g.total.shape[-1])
        for i, j in ((8, 8), (4, 11), (12, 6)):
            margin, total = (g.margin[i, j] * x_m).sum(), (g.total[i, j] * x_t).sum()
            got = v4.fit_means(g, (total + margin) / 2, (total - margin) / 2)
            self.assertAlmostEqual(got[0], g.grid[i], delta=0.02)
            self.assertAlmostEqual(got[1], g.grid[j], delta=0.02)
        low, high = v4.fit_means(g, 14, 14), v4.fit_means(g, 24, 24)
        self.assertLess(sum(low), sum(high))

    def test_a_fixed_seed_prices_a_match_the_same_every_time(self):
        rows = next(iter(self.matches.values()))
        a = v4_stream.match_books(self.tables, self.grid, v4.Variant("v4"), rows, 80,
                                  np.random.default_rng(1), means=(20.0, 14.0))
        b = v4_stream.match_books(self.tables, self.grid, v4.Variant("v4"), rows, 80,
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
        args = (self.tables, self.grid, v4.Variant("v4"))
        a = v4_stream.match_books(*args, rows, 80, np.random.default_rng(1), means=(20.0, 14.0))
        b = v4_stream.match_books(*args, other, 80, np.random.default_rng(1), means=(20.0, 14.0))
        for (_, mp1, tp1), (_, mp2, tp2) in zip(a, b):
            self.assertTrue(np.array_equal(mp1, mp2) and np.array_equal(tp1, tp2))

    def _snapshot_file(self):
        path = os.path.join(self.tmp.name, "snaps.csv")
        if not os.path.exists(path):
            self.test_grades_a_snapshot_file_with_profiles()
        return path

    def test_a_model_with_nb2_takes_its_prior_from_there(self):
        path = self._snapshot_file()
        tables, grid = (os.path.join(self.tmp.name, n) for n in ("v4tables.npz", "v4grid.npz"))
        codes = sorted(self.matches)[:6]

        class Fake:
            league = (17.0, 17.0)

            def __init__(self, level):
                self.level = level

            def means(self, schedule, n_sims=0):
                return {r["MATCH_CODE"]: self.level for r in schedule}

        history = [{"MATCH_CODE": c} for c in codes]
        with mock.patch.object(v4, "prematch_model", return_value=Fake((12.0, 12.0))):
            with self.assertRaises(SystemExit) as caught:
                v4.run(path, tables, grid, [v4.Variant("v4")], matches=codes, workers=1)
            self.assertIn("--history", str(caught.exception))
            low, _ = v4.run(path, tables, grid, [v4.Variant("v4")], matches=codes, n_paths=100,
                            workers=1, history=history)
        with mock.patch.object(v4, "prematch_model", return_value=Fake((26.0, 26.0))):
            high, _ = v4.run(path, tables, grid, [v4.Variant("v4")], matches=codes, n_paths=100,
                             workers=1, history=history)
        over = lambda graded: np.mean([p["v4"] for _, r, p in graded if r.market_id == 54])
        self.assertLess(over(low), over(high))
        # no prediction for a match: skipped, never priced off prod
        _, skipped = v4.run(path, tables, grid, [v4.Variant("v4")], matches=codes, n_paths=50,
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

        with mock.patch.object(v4, "prematch_model", return_value=Fake()):
            with self.assertRaises(SystemExit):
                v4_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50, workers=1)
            quotes = v4_stream.quotes_for_matches({code: rows}, prod, self.tmp.name, n_paths=50,
                                                  workers=1, match_info=[])
        self.assertTrue(quotes)
        self.assertEqual(Fake.asked, [])


    def test_a_missing_model_says_how_to_build_it(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(SystemExit) as caught:
                v4_stream.model_paths(empty)
        self.assertIn("v4-build", str(caught.exception))


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

    def test_v4_build_with_history_writes_the_prematch_model(self):
        from .test_v3 import _matches
        matches = _matches(12)
        out = os.path.join(self.tmp.name, "v4")
        v4.build(matches, out, grid_paths=40, verbose=False, history=self.history(),
                 before=self.BEFORE)
        self.assertIsNotNone(v4.prematch_model(out))
        self.assertIsNotNone(v4.prematch_model(os.path.join(out, "v4tables.npz")))
        self.assertEqual(v4.PriorGrid.load(os.path.join(out, "v4grid.npz")).total_shade, 0.0)

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
