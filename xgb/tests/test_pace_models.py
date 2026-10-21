"""xgb.pace_models: the mixed model's REML against lmer, against the likelihood written out in full, and
the walk's as-of stats never reading a result before it is in."""

import datetime as dt
import math
import unittest

import numpy as np
import pandas as pd

from .. import pace_models as pm

# lmer(drives ~ stream + (1 | p1) + (1 | t1), weights = w, REML = TRUE) on synthetic(), lme4 1.1-35.1:
# lam = sigma^2 / each effect's variance, and the fixed effects
LMER_LAM = (10.875957, 414.311692)
LMER_FIXED = (9.2273116, 0.0320092, 0.1301514)


def synthetic(n=900, gamers=40, teams=8, seed=7):
    """Matches with known pace: gamer sd 1, team sd 0.4, noise sd 3, three streams, weights by age."""
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1.0, gamers)
    t = rng.normal(0, 0.4, teams)
    p1, p2 = rng.integers(0, gamers, n), rng.integers(0, gamers, n)
    p2 = np.where(p2 == p1, (p2 + 1) % gamers, p2)
    t1, t2 = rng.integers(0, teams, n), rng.integers(0, teams, n)
    stream = rng.integers(1, 4, n)
    age = rng.uniform(0, 180, n)
    start = pd.Timestamp("2026-09-01") - pd.to_timedelta(age, unit="D")
    drives = 10 + 0.3 * (stream == 2) - 0.2 * (stream == 3) + g[p1] + g[p2] + t[t1] + t[t2] + rng.normal(0, 3, n)
    m = pd.DataFrame({"code": [f"M{i}" for i in range(n)], "start": start, "p1": [f"G{i:02d}" for i in p1],
                      "p2": [f"G{i:02d}" for i in p2], "t1": [f"T{i}" for i in t1], "t2": [f"T{i}" for i in t2],
                      "stream": stream.astype(str), "drives": drives})
    m["w"] = pm.weights(m.start, pd.Timestamp("2026-09-01"))
    return m


def reml_in_full(X, Z, y, w, lam):
    """-2 REML log-likelihood straight off V = s2 (W^-1 + Z Lam^-1 Z'), s2 profiled out, up to the same
    constant as pm.reml (which drops sum log w): the dense textbook form, for small problems."""
    n, p = X.shape
    H = np.diag(1.0 / w) + Z @ np.diag(1.0 / lam) @ Z.T
    Hi = np.linalg.inv(H)
    XtHiX = X.T @ Hi @ X
    b = np.linalg.solve(XtHiX, X.T @ Hi @ y)
    r = y - X @ b
    rss = float(r @ Hi @ r)
    return ((n - p) * math.log(rss / (n - p)) + np.linalg.slogdet(H)[1] + np.linalg.slogdet(XtHiX)[1]
            + float(np.log(w).sum()))


class TestMixed(unittest.TestCase):

    def test_single_membership_matches_lmer(self):
        m = synthetic()
        d = pm.Design(m, stream=True, gamer_sides=("p1",), team_sides=("t1",))
        beta, log_lam = pm.mixed_fit(d, d.matrix(m), m.drives.values, m.w.values)
        np.testing.assert_allclose(np.exp(log_lam), LMER_LAM, rtol=5e-3)
        np.testing.assert_allclose(beta[:d.p], LMER_FIXED, rtol=1e-4)

    def test_reml_is_the_likelihood_written_out(self):
        m = synthetic(n=160, gamers=12, teams=4, seed=3)
        d = pm.Design(m, stream=True)                   # multi-membership: what lmer can't fit
        X = d.matrix(m).toarray()
        fixed, Z = X[:, :d.p], X[:, d.p:]
        y, w = m.drives.values, m.w.values
        for lam_g, lam_t in ((9.0, 56.0), (30.0, 200.0), (2.0, 10.0)):
            lam = d.penalty(lam_g, lam_t)[d.p:]
            got = pm.reml(d, d.matrix(m), y, w, np.log([lam_g, lam_t]))
            want = reml_in_full(fixed, Z, y, w, lam)
            self.assertAlmostEqual(got - want, 0.0, places=6)   # same constant: the difference is 0

    def test_multi_membership_finds_the_pace_it_was_given(self):
        m = synthetic(n=6000, gamers=60, teams=10, seed=11)
        d = pm.Design(m, stream=True)
        _, log_lam = pm.mixed_fit(d, d.matrix(m), m.drives.values, m.w.values)
        lam_g, lam_t = np.exp(log_lam)
        self.assertAlmostEqual(lam_g, 9.0, delta=3.5)      # 3^2 / 1^2
        self.assertGreater(lam_t, 20.0)                    # 3^2 / 0.4^2 = 56, ten teams: loosely

    def test_ridge_is_the_mixed_model_at_fixed_lam(self):
        m = synthetic(n=400)
        d = pm.Design(m, stream=True)
        X = d.matrix(m)
        ridge = pm.ridge_fit(d, X, m.drives.values, m.w.values, 20.0, 80.0)
        A = (X.T @ np.diag(m.w.values) @ X.toarray()) + np.diag(d.penalty(20.0, 80.0, lam_fixed=0.0))
        mixed = np.linalg.solve(A, X.T @ (m.w.values * m.drives.values))
        np.testing.assert_allclose(ridge, mixed, atol=1e-3)


class TestRolling(unittest.TestCase):

    def test_only_results_in_by_then(self):
        t0 = pd.Timestamp("2026-09-01 12:00")
        m = pd.DataFrame({"start": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2)],
                          "p1": ["A", "B", "A"], "p2": ["B", "C", "C"], "drives": [10.0, 12.0, 14.0]})
        r = pm.Rolling(m, ("p1", "p2"))
        self.assertEqual(r.at("A", t0)[0], 0)                                  # its own match isn't in
        self.assertEqual(r.at("A", t0 + pm.RESULT_LAG - dt.timedelta(seconds=1))[0], 0)
        self.assertEqual(r.at("A", t0 + pm.RESULT_LAG)[:2], (1, 10.0))
        n, last15, last50, ewma = r.at("B", t0 + pd.Timedelta(hours=3))
        self.assertEqual((n, last15, last50), (2, 11.0, 11.0))
        a = 1 - 2 ** (-1 / pm.EWMA_MATCHES)
        self.assertAlmostEqual(ewma, 10.0 + a * 2.0)
        self.assertTrue(np.isnan(r.at("Z", t0)[1]))


if __name__ == "__main__":
    unittest.main()
