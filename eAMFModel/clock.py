"""How much of the game is left, from the only clock the feed has.

There is no game clock in the play feed -- just PERIOD_NUMBER and
EVENT_MESSAGE_COUNT. Messages arrive roughly per snap, so messages elapsed
in a period stand in for time elapsed in it. Period lengths vary (the end of
each half runs long: stoppages), so each period gets its own length, and
the time left in the CURRENT period is the conditional expectation

    E[L - e | L > e],   L ~ Normal(mean_p, sd_p)

given e messages already played in it. That residual shrinks smoothly as the
period runs long but never quite reaches zero until the period number turns
over, which is the right shape: the model cannot see the clock, so it should
not price a late 4th quarter as if it knew exactly how many snaps remain.

Everything is expressed as a fraction of an average regulation game, which
is the unit the possession count scales by.
"""

import math
from dataclasses import dataclass

REGULATION_PERIODS = 4


@dataclass(frozen=True)
class ClockParams:
    # Messages per period, mean and spread. Fitted from 262 matches of
    # directional_pairs (period boundaries bracketed by the first and last
    # snapshot either side, midpoint taken, bracket noise removed from the
    # spread). The 4th has no closing bracket; its length is set from the
    # match's last snapshot plus the typical gap.
    means: tuple = (71.0, 122.0, 81.0, 120.0)
    sds: tuple = (20.0, 29.0, 24.0, 32.0)
    # Share of a game's scoring drives that fall in each period. Not equal:
    # the 4th scores about half what the others do (games are settled, the
    # leader runs the clock, the last drive runs out of time). Fitted to
    # outcomes in fit.py; must sum to 1.
    shares: tuple = (0.27, 0.30, 0.25, 0.18)

    @property
    def regulation(self):
        return sum(self.means)


def _phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _upper_tail(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def residual(mean, sd, elapsed):
    """E[L - e | L > e] for L ~ N(mean, sd)."""
    z = (elapsed - mean) / sd
    tail = _upper_tail(z)
    if tail < 1e-12:
        # Mills-ratio asymptote: sd / z, for a period that is running very long.
        return sd / max(z, 1e-9)
    return mean - elapsed + sd * _phi(z) / tail


def residual_variance(mean, sd, elapsed):
    """Var[L - e | L > e] for L ~ N(mean, sd): the truncated-normal variance."""
    z = (elapsed - mean) / sd
    tail = _upper_tail(z)
    if tail < 1e-12:
        return (sd / max(z, 1e-9)) ** 2
    lam = _phi(z) / tail
    return sd * sd * max(0.0, 1.0 + z * lam - lam * lam)


def half_of(period):
    return 1 if period <= 2 else 2


@dataclass(frozen=True)
class Remaining:
    """What is left, in shares of a whole game's drives: this half, and
    the whole next half if there is one, each with its variance (from not
    knowing how long the periods will run)."""
    this_half: float
    this_half_var: float
    next_half: float
    next_half_var: float

    @property
    def fraction(self):
        return self.this_half + self.next_half


def _period_left(params, i, elapsed):
    """(share, variance) still to come in period i, given messages elapsed."""
    per_message = params.shares[i] / params.means[i]
    left = residual(params.means[i], params.sds[i], elapsed) * per_message
    var = residual_variance(params.means[i], params.sds[i], elapsed) * per_message ** 2
    return left, var


def _period_full(params, i):
    per_message = params.shares[i] / params.means[i]
    return params.shares[i], (params.sds[i] * per_message) ** 2


def remaining(params, period, elapsed_in_period):
    """Share of the game's drives left (and its variance), split by half.

    None for the period means "not started": the whole game is ahead.
    Overtime has nothing left in regulation.
    """
    if period is None:
        period, elapsed_in_period = 1, 0.0
    if period > REGULATION_PERIODS:
        return Remaining(0.0, 0.0, 0.0, 0.0)
    period = max(1, int(period))
    elapsed = max(0.0, float(elapsed_in_period or 0))
    i = period - 1
    mean, var = _period_left(params, i, elapsed)
    if period % 2 == 1:           # 1st or 3rd: the rest of the half is ahead too
        m, v = _period_full(params, i + 1)
        mean, var = mean + m, var + v
    nxt = nxt_var = 0.0
    if period <= 2:
        for j in (2, 3):
            m, v = _period_full(params, j)
            nxt, nxt_var = nxt + m, nxt_var + v
    return Remaining(mean, var, nxt, nxt_var)


def fraction_remaining(params, period, elapsed_in_period):
    """Share of a regulation game's scoring still to come (0 in overtime)."""
    return remaining(params, period, elapsed_in_period).fraction
