"""The joint score models: NB2's shared-gamma scores (their means, variances and covariance as NB2's
own formulas give them), glmer's log-normal ones, reading a book at a shift, and fitting NB2's
dispersion and shared fraction to a prior's errors."""

import unittest

import numpy as np

from .. import score_model as sm


class TestDraws(unittest.TestCase):

    def test_nb2_scores_have_nb2s_moments(self):
        mu1, mu2, a1, a2, shared = 19.0, 17.0, 0.12, 0.09, 0.45
        s1, s2 = sm.nb2_draws(mu1, mu2, a1, a2, shared, n=400000, seed=1)
        self.assertAlmostEqual(s1.mean(), mu1, delta=0.1)
        self.assertAlmostEqual(s2.mean(), mu2, delta=0.1)
        self.assertAlmostEqual(s1.var() / (mu1 + a1 * mu1 ** 2), 1.0, delta=0.03)
        # nb2/NB2_schedule_predict.joint_moments: cov = shared * mu1 * mu2 * min(alpha)
        cov = np.cov(s1, s2)[0, 1]
        self.assertAlmostEqual(cov / (shared * mu1 * mu2 * min(a1, a2)), 1.0, delta=0.06)

    def test_covariance_tightens_the_margin_and_widens_the_total(self):
        alone = sm.Book(*sm.nb2_draws(18, 18, 0.1, 0.1, 0.0, n=200000, seed=2))
        shared = sm.Book(*sm.nb2_draws(18, 18, 0.1, 0.1, 0.6, n=200000, seed=2))
        self.assertLess(shared.margin.std(), alone.margin.std())
        self.assertGreater(shared.total.std(), alone.total.std())

    def test_glmers_shared_match_noise_makes_the_scores_covary(self):
        s1, s2 = sm.lognormal_draws(18.0, 16.0, 0.3, 0.2, n=200000, seed=3)
        self.assertAlmostEqual(s1.mean(), 18.0, delta=0.15)
        self.assertGreater(np.corrcoef(s1, s2)[0, 1], 0.1)
        t1, t2 = sm.lognormal_draws(18.0, 16.0, 0.3, 0.0, n=200000, seed=3)
        self.assertLess(abs(np.corrcoef(t1, t2)[0, 1]), 0.01)


class TestBook(unittest.TestCase):

    def test_ties_split_and_a_shift_moves_the_price(self):
        b = sm.Book(np.array([10, 7, 3, 14]), np.array([7, 7, 10, 0]))     # margins 3, 0, -7, 14
        self.assertAlmostEqual(b.p_home(), (2 + 0.5) / 4)
        self.assertAlmostEqual(b.p_home(-3.5), 1 / 4)                        # 3 - 3.5 < 0
        self.assertAlmostEqual(b.p_over(16.5), 1 / 4)                        # totals 17, 14, 13, 14
        self.assertAlmostEqual(b.p_over(14), 1 / 2)                          # the two 14s are pushes
        self.assertAlmostEqual(b.mean_total(1.0), 15.5)

    def test_the_normal_reading(self):
        n = sm.Normal(0.0, 35.0)
        self.assertAlmostEqual(n.p_home(), 0.5)
        self.assertAlmostEqual(n.p_over(35.0), 0.5)
        self.assertGreater(n.p_home(2.0), 0.5)

    def test_one_seed_per_match(self):
        a = sm.book("nb2", (18, 17, 0.1, 0.1, 0.4), "AF1", draws=1000)
        b = sm.book("nb2", (18, 17, 0.1, 0.1, 0.4), "AF1", draws=1000)
        self.assertEqual(a.p_home(), b.p_home())
        with self.assertRaises(ValueError):
            sm.book("other", (), "AF1")


class TestFit(unittest.TestCase):

    def test_the_method_of_moments_recovers_nb2s_parameters(self):
        rng = np.random.default_rng(4)
        rows = []
        for _ in range(4000):
            mu1, mu2 = rng.uniform(12, 24), rng.uniform(12, 24)
            s1, s2 = sm.nb2_draws(mu1, mu2, 0.1, 0.1, 0.4, n=1, seed=int(rng.integers(1 << 30)))
            rows.append((mu1, mu2, float(s1[0]), float(s2[0])))
        alpha, shared = sm.fit_nb2(rows)
        self.assertAlmostEqual(alpha, 0.1, delta=0.02)
        self.assertAlmostEqual(shared, 0.4, delta=0.12)
        self.assertEqual(sm.fit_nb2(rows[:50]), (None, None))


if __name__ == "__main__":
    unittest.main()
