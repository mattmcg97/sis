"""How much of the game is left, from the only clock the feed has.

There is no game clock in the play feed -- just PERIOD_NUMBER and
EVENT_MESSAGE_COUNT. Messages arrive roughly per snap, so messages stand in
for time.

The unit is the HALF. The second quarter is the first carried on, and the
fourth the third: possession runs across the quarter break. Scoring is not
spread evenly over a half: the share of a half's scoring still to come at
game-time fraction u of it is

    R(u) = (1 - u) ^ (1 + a)

a = 0 is flat; a > 0 is a smooth slope down through the whole half -- no
step at the quarter break -- steeper in the second half, as games get
settled, leaders run the clock and the last drive runs out of time.

Game time u comes from messages, anchored on the quarter break, which IS a
known point of game time (u = 0.5) even though the quarters' message
counts are far from equal (the second quarter of each half runs long on
stoppages and hurry-up). Within a quarter, u moves with messages elapsed
over the quarter's length L -- not known, so averaged over L ~ Normal(mean,
sd) given the quarter has already run e messages (L > e). That keeps late
prices honest about not seeing the clock: the share left shrinks smoothly
as the quarter runs long but never hits zero until the period turns over.

A pure messages-since-the-half-began clock was tried and lost to this on
held-out matches (the quarter break carries real information); so was the
earlier per-quarter clock with a step in scoring between quarters, which
is kept as mode "quarter" for comparison.

Everything is a share of a whole regulation game's scoring drives, the
unit the drive count scales by.
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

    # "half" (the default) or "quarter".
    mode: str = "half"
    # Share of the game's scoring drives in each half, and how steeply
    # each half's scoring falls away (a in R(u) above). Fitted on half of
    # 262 matches, checked on the other: near flat through the first half,
    # a clear fall through the second, which the drive in progress follows
    # too (params.current_intensity).
    half_shares: tuple = (0.57, 0.43)
    half_slopes: tuple = (-0.2, 0.5)

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


def half_left(params, period, elapsed, points=48):
    """(mean, variance) of the share of this half's scoring still to come.

    Game time through the half is anchored on the quarter break: the first
    quarter of the half runs u from 0 to 0.5, the second from 0.5 to 1,
    each at the rate its own (unknown) message length L allows. So u =
    (quarter - 1) / 2 + e / (2 L), averaged over L > e.
    """
    half = half_of(period)
    i = period - 1
    mean_l, sd_l = params.means[i], params.sds[i]
    a = params.half_slopes[half - 1]
    offset = 0.0 if period % 2 == 1 else 0.5
    if elapsed <= 0:
        r = (1.0 - offset) ** (1.0 + a)
        return r, 0.0
    lo = max(elapsed, mean_l - 4 * sd_l)
    hi = max(mean_l + 4 * sd_l, elapsed + 3 * sd_l)
    step = (hi - lo) / points
    w_sum = m1 = m2 = 0.0
    for k in range(points):
        length = lo + (k + 0.5) * step
        w = _phi((length - mean_l) / sd_l)
        u = offset + 0.5 * min(1.0, elapsed / length)
        r = (1.0 - u) ** (1.0 + a)
        w_sum += w
        m1 += w * r
        m2 += w * r * r
    if w_sum <= 0:
        return 0.0, 0.0
    mean = m1 / w_sum
    return mean, max(0.0, m2 / w_sum - mean * mean)


def remaining(params, period, elapsed_in_period, elapsed_in_half=None):
    """Share of the game's drives left (and its variance), split by half.

    None for the period means "not started": the whole game is ahead.
    Overtime has nothing left in regulation. `elapsed_in_half` (messages
    since the half's first) is what the half clock runs on; without it the
    quarter's elapsed is taken, plus an average first quarter in the 2nd
    or 4th.
    """
    if period is None:
        period, elapsed_in_period, elapsed_in_half = 1, 0.0, 0.0
    if period > REGULATION_PERIODS:
        return Remaining(0.0, 0.0, 0.0, 0.0)
    if params.mode == "half":
        half = half_of(period)
        mean, var = half_left(params, max(1, int(period)),
                              max(0.0, float(elapsed_in_period or 0)))
        share = params.half_shares[half - 1]
        nxt = params.half_shares[1] if half == 1 else 0.0
        return Remaining(share * mean, share * share * var, nxt, 0.0)
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


def intensity(params, period, elapsed_in_period):
    """Scoring rate now, relative to the half's average: w(u) = (1+a)(1-u)^a,
    at the expected position u in the half (quarter-anchored game time).
    Second half only -- the first's slope is slight and the per-drive
    effect of it did not hold up; 1 on the quarter clock or in overtime."""
    if (params.mode != "half" or period is None or period > REGULATION_PERIODS
            or period <= 2):
        return 1.0
    period = max(1, int(period))
    a = params.half_slopes[half_of(period) - 1]
    share, _ = half_left(params, period, max(0.0, float(elapsed_in_period or 0)))
    u = 1.0 - max(0.0, share) ** (1.0 / (1.0 + a))
    return (1.0 + a) * max(0.0, 1.0 - u) ** a


def messages_left_in_half(params, period, elapsed_in_period):
    """Expected CLOCK left in the half, in messages: the rest of this
    quarter (given how long it has run) plus the next one if this is the
    half's first. Time, not scoring -- what decides whether a drive has
    room to finish. None / overtime: 0."""
    if period is None:
        period, elapsed_in_period = 1, 0.0
    if period > REGULATION_PERIODS:
        return 0.0
    i = max(1, int(period)) - 1
    left = residual(params.means[i], params.sds[i], max(0.0, float(elapsed_in_period or 0)))
    if period % 2 == 1:
        left += params.means[i + 1]
    return left


def fraction_remaining(params, period, elapsed_in_period, elapsed_in_half=None):
    """Share of a regulation game's scoring still to come (0 in overtime)."""
    return remaining(params, period, elapsed_in_period, elapsed_in_half).fraction
