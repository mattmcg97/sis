"""Team strength: the pre-match view, and how the scoreboard moves it.

TOP DOWN. The pre-match spread and total fix each side's expected points,

    home = (total + spread) / 2,   away = (total - spread) / 2

(spread = expected home margin, the market-52 line). Dividing by the number
of drives a team gets turns that into expected points per drive -- the
level the drive chain is tuned to.

Then the game is allowed to disagree. Each side's rate carries a multiplier
theta with a Gamma(k, k) prior (mean 1). Points are lumpy, so they are
counted in "scores" of `points_per_score` points, and by now a side was
expected to have put up `expected * elapsed` of them. Gamma-Poisson gives

    theta | game  ~  Gamma(k + scored / v,  k + expected_by_now / v)

so a side outscoring its prior pulls its rate up, by an amount that grows
with the evidence and fades with k. k is the one dial between "trust the
pre-match number" (large k) and "trust what is happening" (small k): the
two model versions differ there. Only the scoreboard is used, not drive
counts, so a drive the feed mis-splits cannot move the strength.

The posterior is carried as a few equal-weight quantiles, not just its
mean, so uncertainty about how good a side really is widens the remaining
points distribution instead of being thrown away.
"""

import math
from dataclasses import dataclass

# Standard normal quantiles at the midpoints of n equal-probability slices.
_SLICES = {
    1: (0.0,),
    3: (-0.9674, 0.0, 0.9674),
    5: (-1.2816, -0.5244, 0.0, 0.5244, 1.2816),
    7: (-1.4652, -0.7916, -0.3661, 0.0, 0.3661, 0.7916, 1.4652),
}


@dataclass(frozen=True)
class Prior:
    home_points: float
    away_points: float

    @classmethod
    def from_lines(cls, spread, total, floor=3.0):
        """Spread is the expected home margin; total the expected combined."""
        home = max(floor, (total + spread) / 2.0)
        away = max(floor, (total - spread) / 2.0)
        return cls(home, away)


def gamma_quantile(shape, rate, z):
    """Wilson-Hilferty: the Gamma(shape, rate) quantile at normal quantile z."""
    c = 1.0 / (9.0 * shape)
    x = shape * max(0.0, 1.0 - c + z * math.sqrt(c)) ** 3
    return x / rate


def posterior(k, scored, expected_by_now, points_per_score):
    """(shape, rate) of theta after the game so far."""
    return (k + scored / points_per_score, k + expected_by_now / points_per_score)


def multipliers(k, scored, expected_by_now, points_per_score, n_slices=5):
    """[(weight, theta)], equal weights, renormalised to the posterior mean.

    Wilson-Hilferty is a touch biased for small shapes, so the nodes are
    rescaled to hit the exact posterior mean -- that keeps the expected
    points on the line the posterior says, whatever n_slices is.
    """
    if k is None or math.isinf(k):
        return [(1.0, 1.0)]
    shape, rate = posterior(k, scored, expected_by_now, points_per_score)
    zs = _SLICES[n_slices]
    nodes = [gamma_quantile(shape, rate, z) for z in zs]
    mean = shape / rate
    avg = sum(nodes) / len(nodes)
    if avg > 0:
        nodes = [x * mean / avg for x in nodes]
    w = 1.0 / len(nodes)
    return [(w, x) for x in nodes]
