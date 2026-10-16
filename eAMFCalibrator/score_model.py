"""Joint score models: a match's two scores simulated together, so the moneyline, the total's over
at any line and the margin come from one distribution, with the scores' covariance in it.

  nb2        NB2's own construction (nb2/NB2_schedule_predict.py): each side negative binomial
             (mean mu, dispersion alpha: variance mu + alpha mu^2), the two sharing a gamma pace
             factor -- r_i = 1/alpha_i, k = shared * min(r_1, r_2), U ~ Gamma(k), V_i ~ Gamma(r_i - k),
             lambda_i = (mu_i / r_i)(U + V_i), Y_i ~ Poisson(lambda_i). Each score stays exactly NB2;
             the shared U makes them covary (a fast game lifts both), which tightens the margin and
             widens the total
  lognormal  glmer's own (glmer/predict.R, eAMFModel/prior_check.raw_books): each side Poisson about
             exp(eta + its own normal noise (sd s_obs) + the match's (sd s_match)), eta set so the
             mean is the expected points; the shared match noise is glmer's covariance
  normal     the margin and the total as normals about the expected values (MARGIN_SD, TOTAL_SD):
             no covariance, no shape, no ties; for comparison

fit_nb2() fits a single dispersion and shared fraction to a prior's own errors by the method of
moments -- a side's squared error less its mean is alpha mu^2, the two sides' errors' product is
shared * alpha * mu_1 mu_2 -- so a prior with no score model of its own (glmer trimmed of its noise
terms) can be priced with NB2's.

A moneyline splits ties half and half, as NB2 does; the over leaves out pushes, and so does
p_margin (the home margin over a line), which prices a bet's moneyline or spread. A shift (the
rematch layer's move) moves the simulated margin or total before it is read, so the layer moves the
model's own price. Every match draws from its own seed (off its match code): each arm and each
variant of a match sees the same randomness, so paired comparisons measure the models, not the
simulation.
"""

import hashlib
import math
from statistics import NormalDist

import numpy as np

MARGIN_SD = 9.35
TOTAL_SD = 12.0
DRAWS = 40000
ALPHA_FLOOR = 1e-4
_N = NormalDist()


def seed_of(match):
    return int(hashlib.sha1(str(match).encode()).hexdigest()[:8], 16)


def nb2_draws(mu1, mu2, a1, a2, shared, n=DRAWS, seed=0):
    """(home, away) scores under NB2's shared-gamma model."""
    rng = np.random.default_rng(seed)
    a1, a2 = max(a1, ALPHA_FLOOR), max(a2, ALPHA_FLOOR)
    r1, r2 = 1.0 / a1, 1.0 / a2
    k = min(max(shared, 0.0), 0.999) * min(r1, r2)
    u = rng.gamma(k, 1.0, n) if k > 1e-12 else np.zeros(n)
    v1 = rng.gamma(max(r1 - k, 1e-12), 1.0, n)
    v2 = rng.gamma(max(r2 - k, 1e-12), 1.0, n)
    return rng.poisson((mu1 / r1) * (u + v1)), rng.poisson((mu2 / r2) * (u + v2))


def lognormal_draws(mu1, mu2, s_obs, s_match, n=DRAWS, seed=0):
    """(home, away) scores under glmer's model: Poisson about a log-normal mean, a shared match noise."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0.0, s_match, n) if s_match else np.zeros(n)
    out = []
    for mu in (mu1, mu2):
        eta = math.log(max(mu, 0.1)) - (s_obs ** 2 + s_match ** 2) / 2
        noise = rng.normal(0.0, s_obs, n) if s_obs else np.zeros(n)
        out.append(rng.poisson(np.exp(np.clip(eta + shared + noise, -15, 15))))
    return out[0], out[1]


class Book:
    """A match's simulated margins and totals, read at any shift: the moneyline, the over at a
    line, the expected total."""

    def __init__(self, s1, s2):
        self.margin = (s1 - s2).astype(float)
        self.total = (s1 + s2).astype(float)

    def p_home(self, shift=0.0):
        m = self.margin + shift
        return float(np.mean(m > 0) + 0.5 * np.mean(m == 0))

    def p_over(self, line, shift=0.0):
        t = self.total + shift
        over, push = float(np.mean(t > line)), float(np.mean(t == line))
        return over / (1 - push) if push < 1 else None

    def p_margin(self, line, shift=0.0):
        """The home margin over `line`, pushes left out: a bet's moneyline (line 0) or spread."""
        m = self.margin + shift
        over, push = float(np.mean(m > line)), float(np.mean(m == line))
        return over / (1 - push) if push < 1 else None

    def mean_total(self, shift=0.0):
        return float(np.mean(self.total)) + shift


class Normal:
    """The normal reading: the moneyline and the over off the expected margin and total."""

    def __init__(self, margin, total, margin_sd=MARGIN_SD, total_sd=TOTAL_SD):
        self.m, self.t, self.msd, self.tsd = margin, total, margin_sd, total_sd

    def p_home(self, shift=0.0):
        return _N.cdf((self.m + shift) / self.msd)

    def p_over(self, line, shift=0.0):
        return 1 - _N.cdf((line - self.t - shift) / self.tsd)

    def p_margin(self, line, shift=0.0):
        return 1 - _N.cdf((line - self.m - shift) / self.msd)

    def mean_total(self, shift=0.0):
        return self.t + shift


def book(kind, params, match, draws=DRAWS):
    """The pricing object for one match: kind nb2 (mu1, mu2, alpha1, alpha2, shared), lognormal
    (mu1, mu2, s_obs, s_match) or normal (margin, total)."""
    if kind == "normal":
        return Normal(*params)
    if kind == "nb2":
        return Book(*nb2_draws(*params, n=draws, seed=seed_of(match)))
    if kind == "lognormal":
        return Book(*lognormal_draws(*params, n=draws, seed=seed_of(match)))
    raise ValueError(f"no score model {kind!r}")


def fit_nb2(rows):
    """(alpha, shared) fitted to a prior's errors by the method of moments, off rows of (mu1, mu2,
    score1, score2); (None, None) with too few."""
    rows = [r for r in rows if None not in r]
    if len(rows) < 100:
        return None, None
    num = sum((y - mu) ** 2 - mu for mu1, mu2, y1, y2 in rows for mu, y in ((mu1, y1), (mu2, y2)))
    den = sum(mu * mu for mu1, mu2, _, _ in rows for mu in (mu1, mu2))
    alpha = max(num / den, ALPHA_FLOOR) if den else ALPHA_FLOOR
    cov = sum((y1 - mu1) * (y2 - mu2) for mu1, mu2, y1, y2 in rows)
    scale = sum(alpha * mu1 * mu2 for mu1, mu2, _, _ in rows)
    shared = min(max(cov / scale, 0.0), 0.95) if scale else 0.0
    return alpha, shared
