"""prior-vs-sim: the pre-match prior's own price against v13's kick-off off it, graded on the results."""

import datetime as dt
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np

from .. import __main__ as cli
from .. import prior_check, v13

DAY = dt.datetime(2026, 10, 1, 12)


def row(code, minutes, s1, s2, p1="ANN", p2="BOB"):
    t = DAY + dt.timedelta(minutes=minutes)
    return {"MATCH_CODE": code, "SCHEDULED_START_TIME_UTC": f"{t:%Y-%m-%d %H:%M:%S}", "PLAYER_1_HANDLE": p1,
            "PLAYER_2_HANDLE": p2, "PLAYER_1_TEAM": "A", "PLAYER_2_TEAM": "B", "STREAM_NUMBER": "1",
            "PLAYER_1_FINAL_SCORE": str(s1), "PLAYER_2_FINAL_SCORE": str(s2)}


def point_books(margin, total):
    """A book certain of one margin and one total."""
    mp_, tp = np.zeros(2 * v13.MARGIN_MAX + 1), np.zeros(v13.TOTAL_MAX + 1)
    mp_[margin + v13.MARGIN_MAX] = 1.0
    tp[total] = 1.0
    return mp_, tp


class Fit:
    def __init__(self, s_obs, s_match):
        self.meta = {"model": {"sigma_obs": s_obs, "sigma_match": s_match}}


class TestRawBooks(unittest.TestCase):

    def test_the_score_model_keeps_the_expected_points(self):
        mp_, tp = prior_check.raw_books((24.0, 18.0), 0.38, 0.2, n=60000)
        k = np.arange(len(mp_)) - v13.MARGIN_MAX
        self.assertAlmostEqual(float((k * mp_).sum()), 6.0, delta=0.25)
        self.assertAlmostEqual(float((np.arange(len(tp)) * tp).sum()), 42.0, delta=0.4)
        plain_m, plain_t = prior_check.raw_books((24.0, 18.0), 0.0, 0.0, n=60000)
        sd = lambda p, ks: float(np.sqrt((p * (ks - (ks * p).sum()) ** 2).sum()))
        kt = np.arange(len(tp))
        self.assertGreater(sd(tp, kt), sd(plain_t, kt) + 2)        # glmer's noise widens the total
        self.assertAlmostEqual(sd(plain_t, kt), np.sqrt(42), delta=0.2)

    def test_sigmas_come_from_the_fit_that_prices_the_day(self):
        self.assertEqual(prior_check.sigmas(Fit(0.38, 0.2), DAY.date()), (0.38, 0.2))
        rolling = mock.Mock(spec=["fit_for"])
        rolling.fit_for.return_value = Fit(0.3, 0.1)
        self.assertEqual(prior_check.sigmas(rolling, DAY.date()), (0.3, 0.1))
        rolling.fit_for.assert_called_with(DAY.date())
        self.assertEqual(prior_check.sigmas(mock.Mock(spec=[]), DAY.date()), (0.0, 0.0))

    def test_the_base_prior_is_under_every_layer(self):
        base = mock.Mock(spec=["means"])
        layered = mock.Mock(spec=["pre"], pre=mock.Mock(spec=["pre"], pre=base))
        self.assertIs(prior_check.base_prior(layered), base)
        self.assertIs(prior_check.base_prior(base), base)


class TestGrade(unittest.TestCase):

    def results(self):
        # two matches: the home side wins the first by 7 (total 41), loses the second by 3 (total 37)
        out = []
        for (s1, s2), right in (((24, 17), (7, 41)), ((17, 20), (-3, 37))):
            sure = point_books(*right)
            wrong = point_books(-right[0], right[1] + 10)
            out.append(dict(row=row("M", 0, s1, s2), finals=(s1, s2), raw_means=(20, 20), means=(20, 20),
                            books={"raw": wrong, "raw+": wrong, "sim": sure}))
        return out

    def test_a_book_sure_of_the_result_scores_perfectly(self):
        summary, tables = prior_check.grade(self.results())
        sim, raw = summary["sim"], summary["raw"]
        self.assertLess(sim["log_loss"], 1e-6)
        self.assertAlmostEqual(sim["brier"], 0.0)
        self.assertEqual(sim["margin"]["bias"], 0.0)
        self.assertEqual(sim["margin"]["rps"], 0.0)
        self.assertEqual(sim["total"]["in50"], 1.0)
        self.assertGreater(raw["log_loss"], 10)
        self.assertAlmostEqual(raw["brier"], 1.0)
        self.assertAlmostEqual(raw["total"]["bias"], -10.0)
        self.assertEqual(raw["margin"]["in80"], 0.0)
        self.assertEqual(sum(n for _, _, n in tables["sim"]), 2)

    def test_pushes_leave_the_moneyline(self):
        res = self.results() + [dict(row=row("P", 0, 20, 20), finals=(20, 20), raw_means=(20, 20), means=(20, 20),
                                     books={m: point_books(0, 40) for m in prior_check.METHODS})]
        summary, _ = prior_check.grade(res)
        self.assertEqual((summary["sim"]["n"], summary["sim"]["ml_n"]), (3, 2))

    def test_the_report_and_the_csv(self):
        lines = prior_check.report(self.results(), DAY.date(), DAY.date())
        text = "\n".join(lines)
        self.assertIn("2 settled matches", text)
        self.assertIn("moneyline log loss", text)
        self.assertIn("total RPS", text)
        self.assertIn("v13 sim", text)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.csv")
            prior_check.write_csv(path, self.results())
            with open(path) as fh:
                head, first = fh.readline(), fh.readline()
        self.assertIn("sim_home_win", head)
        self.assertTrue(first.startswith("M,2026-10-01 12:00:00,ANN,BOB,24,17,20,20,20,20"))


class TestRun(unittest.TestCase):

    def test_both_prices_off_the_same_matches_as_of_kick_off(self):
        history = [row("OLD", -1440 * 3, 20, 10), row("M1", 0, 24, 17), row("M2", 60, 10, 30),
                   row("LATER", 1440 * 5, 9, 9)]
        base = mock.Mock(spec=["means", "meta"], meta={"model": {"sigma_obs": 0.3, "sigma_match": 0.1}})
        base.means.side_effect = lambda rows, results=None: {r["MATCH_CODE"]: (20.0, 18.0) for r in rows}
        pre = mock.Mock(spec=["means", "pre"], pre=base)
        pre.means.side_effect = lambda rows, results=None: {r["MATCH_CODE"]: (21.0, 17.0) for r in rows}
        seen = []

        def worker(job):
            seen.append(job)
            return {code: point_books(4, 38) for code, _, _ in job[1]}
        with mock.patch.object(prior_check.v13_stream, "model_paths", return_value=("t.npz", "g.npz")), \
                mock.patch.object(prior_check.v13, "prematch_model", return_value=pre), \
                mock.patch.object(prior_check, "profiles", side_effect=lambda d, rows: {r["MATCH_CODE"]: None for r in rows}), \
                mock.patch.object(prior_check, "_sim_worker", side_effect=worker):
            out = prior_check.run("build", history, DAY.date(), DAY.date() + dt.timedelta(days=1))
        self.assertEqual([x["row"]["MATCH_CODE"] for x in out], ["M1", "M2"])
        self.assertEqual(out[0]["raw_means"], (20.0, 18.0))
        self.assertEqual(out[0]["means"], (21.0, 17.0))
        self.assertEqual([code for code, _, means in seen[0][1]], ["M1", "M2"])
        self.assertEqual(seen[0][1][0][2], (21.0, 17.0))                  # the sim prices off the build's means
        self.assertIs(pre.means.call_args.kwargs["results"], history)      # form reads every result before
        self.assertEqual(set(out[0]["books"]), set(prior_check.METHODS))

    def test_the_command(self):
        with tempfile.TemporaryDirectory() as d:
            hist = os.path.join(d, "h.csv")
            with open(hist, "w") as fh:
                fh.write("MATCH_CODE,SPORT_CODE,STREAM_NUMBER,SCHEDULED_START_TIME_UTC,PLAYER_1_HANDLE,PLAYER_1_TEAM,"
                         "PLAYER_2_HANDLE,PLAYER_2_TEAM,PLAYER_1_FINAL_SCORE,PLAYER_2_FINAL_SCORE\n"
                         "M1,AF,1,2026-10-01 12:00:00,ANN,A,BOB,B,24,17\n")
            res = [dict(row=row("M1", 0, 24, 17), finals=(24.0, 17.0), raw_means=(20, 18), means=(21, 17),
                        books={m: point_books(7, 41) for m in prior_check.METHODS})]
            out = io.StringIO()
            with mock.patch.object(prior_check, "run", return_value=res) as run, redirect_stdout(out):
                cli.main(["prior-vs-sim", "--model", "b", "--history", hist, "--since", "2026-10-01",
                          "--until", "2026-10-12", "--workers", "2", "--out", os.path.join(d, "o")])
            self.assertEqual(run.call_args.args[2:], (dt.date(2026, 10, 1), dt.date(2026, 10, 12)))
            self.assertEqual(run.call_args.kwargs["workers"], 2)
            self.assertIn("1 settled matches", out.getvalue())
            self.assertTrue(os.path.exists(os.path.join(d, "o", "prior_vs_sim.csv")))


if __name__ == "__main__":
    unittest.main()


class TestV14(unittest.TestCase):
    """prior-vs-sim on a v14 build: v14's own kick-off, named for it."""

    def test_a_v14_build_is_priced_by_v14(self):
        from .. import v14, v14_stream
        history = [row("M1", 0, 24, 17)]
        pre = mock.Mock(spec=["means"])
        pre.means.side_effect = lambda rows, results=None: {r["MATCH_CODE"]: (21.0, 17.0) for r in rows}
        seen = []

        def worker(job):
            seen.append(job)
            return {code: point_books(4, 38) for code, _, _ in job[1]}
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "v14tables.npz"), "w").close()
            self.assertEqual(prior_check.version_of(d), "v14")
            with mock.patch.object(v14_stream, "model_paths", return_value=("t.npz", "g.npz")), \
                    mock.patch.object(v14, "prematch_model", return_value=pre) as made, \
                    mock.patch.object(prior_check, "profiles", side_effect=lambda dd, rows: {r["MATCH_CODE"]: None for r in rows}), \
                    mock.patch.object(prior_check, "_sim_worker", side_effect=worker):
                out = prior_check.run(d, history, DAY.date(), DAY.date())
        self.assertTrue(made.called)
        self.assertEqual(seen[0][4], "v14")
        self.assertIn("v14 sim", "\n".join(prior_check.report(out, DAY.date(), DAY.date(), "v14")))
        self.assertEqual(prior_check.version_of("no_such_build"), "v13")
