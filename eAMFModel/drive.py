"""The current drive, as a play-by-play Markov chain.

State is (down, field position, line to gain). Field position is yards from
the offense's own goal line, 1-99, which is how INPLAY_FIELD_POSITION_PERIOD
reports it: the kickoff spot reads 35, a touchdown is crossing 100. Each
snap is one transition:

  turnover          drive over, no points
  gain g            g drawn from the play kernel (losses, no-gains, short
                    gains, big plays); reaching 100 is a touchdown,
                    reaching the line to gain is a fresh 1st-and-10
  4th down          go for it with the fitted probability for that down,
                    distance and field (shifted by the player's aggression),
                    otherwise kick: a field goal from range, a punt before it

The chain is solved EXACTLY, backwards, into the probability that the drive
ends in a touchdown or a field goal from every state -- no simulation. Every
series has a fixed line to gain L, a first down always moves L forward, and
within a series only the down changes, so ordering states by (L descending,
down descending) visits every state after all the states it can reach.

Offensive quality q scales the play kernel (the mean of ordinary gains and
the rate of big plays). The chain is solved on a small grid of q and
interpolated, and q itself is set so the value of a fresh drive matches the
team's expected points per drive -- which is what the top-down strength
model hands down. So the drive chain supplies the SHAPE of how a drive's
chances move with down, distance and field position; the level comes from
the game-level view.
"""

import bisect
import math
from dataclasses import dataclass, field

from . import dist

FIELD = 100

# 4th-down policies. NORMAL kicks in range, goes on 4th-and-short past
# go_min_field, punts otherwise. Late in a game the side behind changes
# its mind: MUST_SCORE never punts (a punt ends the game) but still kicks
# when a field goal is worth having; NEED_TD never punts and never kicks,
# because three points do not catch up.
NORMAL = "normal"
# Players' 4th-down aggression (log-odds shift on going for it) is solved on
# this grid and interpolated.
AGGRESSION_GRID = (-2.0, -1.0, 0.0, 1.0, 2.0)
MUST_SCORE = "must_score"
NEED_TD = "need_td"
POLICIES = (NORMAL, MUST_SCORE, NEED_TD)


@dataclass(frozen=True)
class DriveParams:
    # The play kernel, fitted by maximum likelihood to 38,000 real snaps
    # (1,160 matches of SCOUTING_FULL PLAY_OVER snapshots, the train half):
    # for every snap more than two minutes from the end of its half, did the
    # possession end in a touchdown, a field goal or neither, against what
    # this chain says from that down, distance and field -- with the fitted
    # 4th-down behaviour below in the chain (fitting it with NFL-style
    # punting instead cost 4,900 log-likelihood points). Madden plays big,
    # and the red zone roughly halves every gain.
    p_turnover: float = 0.0265      # interception or fumble, per snap
    p_zero: float = 0.0766          # incompletion / stuffed
    p_loss: float = 0.1991          # sack or tackle for loss
    loss_max: int = 7               # losses uniform on 1..loss_max
    gain_mean: float = 19.09        # ordinary gains: 1 + geometric
    p_big: float = 0.1369           # big plays, per snap
    big_mean: float = 14.59         # big plays: 10 + geometric
    # 4th down, fitted to 7,400 real 4th-down decisions and 3,000 field-goal
    # attempts in SCOUTING_FULL (fit.fit_fourth_down). Madden players go for
    # it far more than the NFL: 4th-and-1 from their own 35 about 85% of the
    # time, 4th-and-6 about half the time. Logistic in the log of the
    # distance and the field position (see go_probability); a kick is a
    # field goal rather than a punt from about the opponent's 45 on; and the
    # kickers are good -- 95% at 45 yards, 50% near 63.
    go_coef: tuple = (-1.4844, -0.1921, 12.7965, -10.4955, -2.3300, 0.1111, 1.1210)
    fg_kick_coef: tuple = (-24.8943, 47.7511)
    make_coef: tuple = (1.7102, 14.8557, -27.0659)
    fg_max_distance: int = 75       # nothing is tried from further than this
    start_field: int = 28           # typical fresh-drive start (the feed's kickoff mean)
    redzone_field: int = 80         # from here on the field compresses...
    redzone_scale: float = 0.55     # ...and gains shrink by this (fitted, as above)
    quality_grid: tuple = (0.35, 0.45, 0.56, 0.68, 0.8, 0.9, 1.0, 1.12, 1.26, 1.42,
                          1.6, 1.85, 2.2, 2.7)


def kick_distance(field_position):
    return FIELD - field_position + 17


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def fg_make(params, field_position):
    """P(a field goal from here is good): kick distance is 117 - field."""
    distance = kick_distance(field_position)
    if distance > params.fg_max_distance:
        return 0.0
    d = distance / 100.0
    a, b, c = params.make_coef
    return _sigmoid(a + b * d + c * d * d)


def go_probability(params, field_position, distance, aggression=0.0):
    """P(going for it on 4th down) for an average player, shifted in log
    odds by the player's `aggression`."""
    u = field_position / 100.0
    lt = math.log(max(1, distance))
    red = 1.0 if field_position >= 80 else 0.0
    c = params.go_coef
    z = c[0] + c[1] * lt + c[2] * u + c[3] * u * u + c[4] * lt * u + c[5] * red + c[6] * lt * red
    return _sigmoid(z + aggression)


def field_goal_share(params, field_position):
    """P(a 4th-down kick is a field goal rather than a punt)."""
    a, b = params.fg_kick_coef
    return _sigmoid(a + b * field_position / 100.0)


def gain_kernel(params, quality, scale=1.0):
    """Per-snap outcome: (p_turnover, {gain: prob}) with gains in yards.

    Quality multiplies the ordinary gain mean and the big-play rate, and
    divides the rates of the snaps that go nowhere -- turnovers and losses
    in full, incompletions by its square root -- so a better offense both
    gains more and stalls less. Ordinary gains take what is left.
    """
    p_big = min(0.35, params.p_big * quality * scale)
    mean = max(1.05, params.gain_mean * quality * scale)
    p_turnover = params.p_turnover / quality
    p_zero = params.p_zero / math.sqrt(quality)
    p_loss = params.p_loss / quality
    p_ordinary = 1.0 - (p_turnover + p_zero + p_loss + p_big)
    if p_ordinary <= 0.0:
        raise ValueError("play kernel has no room for ordinary gains")
    kernel = {}
    kernel[0] = p_zero
    for yards in range(1, params.loss_max + 1):
        kernel[-yards] = kernel.get(-yards, 0.0) + p_loss / params.loss_max
    # 1 + Geometric(success rate r) has mean 1/r.
    r = 1.0 / mean
    for g in range(1, FIELD + 1):
        kernel[g] = kernel.get(g, 0.0) + p_ordinary * r * (1 - r) ** (g - 1)
    rb = 1.0 / (params.big_mean - 9)
    for g in range(10, FIELD + 1):
        kernel[g] = kernel.get(g, 0.0) + p_big * rb * (1 - rb) ** (g - 10)
    # Whatever the truncation at 100 lost is a touchdown from anywhere.
    lost = 1.0 - p_turnover - sum(kernel.values())
    kernel[FIELD] = kernel.get(FIELD, 0.0) + max(0.0, lost)
    return p_turnover, kernel


class Chain:
    """Solved drive chain for one quality: P(TD), P(FG) from any state."""

    def __init__(self, params, quality, policy=NORMAL, aggression=0.0):
        self.params = params
        self.quality = quality
        self.policy = policy
        self.aggression = aggression
        self.p_turnover, kernel = gain_kernel(params, quality)
        rz_turnover, rz_kernel = gain_kernel(params, quality, params.redzone_scale)
        low = -params.loss_max
        # gains[i] = P(gain == low + i), gains index up to +FIELD.
        self.low = low
        self.gains = [kernel.get(g, 0.0) for g in range(low, FIELD + 1)]
        self.rz_gains = [rz_kernel.get(g, 0.0) for g in range(low, FIELD + 1)]
        self.rz_turnover = rz_turnover
        self._solve()

    def _solve(self):
        p = self.params
        # td[d][L][y], fg[d][L][y]: d 1..4, L 2..100, y 1..L-1.
        td = [[None] * (FIELD + 1) for _ in range(5)]
        fg = [[None] * (FIELD + 1) for _ in range(5)]
        # fresh first down at y: (1, y, min(y + 10, 100)); TD at y >= 100.
        # fresh_td[y], fresh_fg[y] for y in 1..100+.
        size = FIELD + FIELD + 2
        fresh_td = [0.0] * size
        fresh_fg = [0.0] * size
        for y in range(FIELD, size):
            fresh_td[y] = 1.0
        low = self.low
        open_field = (self.gains, 1.0 - self.p_turnover)
        red_zone = (self.rz_gains, 1.0 - self.rz_turnover)
        for L in range(FIELD, 1, -1):
            for d in range(4, 0, -1):
                row_td = [0.0] * L
                row_fg = [0.0] * L
                nxt_td = td[d + 1][L] if d < 4 else None
                nxt_fg = fg[d + 1][L] if d < 4 else None
                for y in range(1, L):
                    t = L - y
                    go = 1.0          # share of the time it goes for it
                    kick_fg = 0.0     # value from kicking instead (FG only; a punt is 0)
                    if d == 4:
                        make = fg_make(p, y) if self.policy != NEED_TD else 0.0
                        if self.policy == NORMAL:
                            go = go_probability(p, y, t, self.aggression)
                            kick_fg = (1.0 - go) * field_goal_share(p, y) * make
                        elif self.policy == MUST_SCORE and make >= 0.5:
                            go, kick_fg = 0.0, make
                        # NEED_TD, and MUST_SCORE out of range: always go
                    gains, keep = red_zone if y >= p.redzone_field else open_field
                    # Conversions: gains g >= t land on a fresh first down at y + g.
                    a_td = a_fg = 0.0
                    start = t - low
                    for i in range(start, len(gains)):
                        w = gains[i]
                        if w:
                            y2 = y + low + i
                            a_td += w * fresh_td[y2]
                            a_fg += w * fresh_fg[y2]
                    # Short of the line: next down, same line to gain.
                    if nxt_td is not None:
                        for i in range(0, start):
                            w = gains[i]
                            if w:
                                y2 = max(1, y + low + i)
                                a_td += w * nxt_td[y2]
                                a_fg += w * nxt_fg[y2]
                    row_td[y] = go * keep * a_td
                    row_fg[y] = go * keep * a_fg + kick_fg
                td[d][L] = row_td
                fg[d][L] = row_fg
            # Every state that can reach a fresh first down with line L is done.
            y = L - 10 if L < FIELD else None
            if L < FIELD:
                if 1 <= y < FIELD:
                    fresh_td[y] = td[1][L][y]
                    fresh_fg[y] = fg[1][L][y]
            else:
                for y in range(90, FIELD):
                    fresh_td[y] = td[1][FIELD][y]
                    fresh_fg[y] = fg[1][FIELD][y]
        self.td, self.fg = td, fg

    def value(self, down, field_position, distance):
        """(P(TD), P(FG)) for this drive from (down, field, distance)."""
        down = min(4, max(1, int(down)))
        y = min(FIELD - 1, max(1, int(field_position)))
        t = max(1, int(distance))
        L = min(FIELD, y + t)
        return self.td[down][L][y], self.fg[down][L][y]

    def fresh(self, field_position=None):
        y = self.params.start_field if field_position is None else field_position
        return self.value(1, y, 10)


def touchdown_points(q6, q8):
    """Points from a touchdown, extra point or two included."""
    return {6: q6, 7: 1.0 - q6 - q8, 8: q8}


def outcome_pmf(p_td, p_fg, q6, q8):
    """Drive points pmf over {0, 3, 6, 7, 8}."""
    pmf = [0.0] * 9
    pmf[0] = max(0.0, 1.0 - p_td - p_fg)
    pmf[3] = p_fg
    for pts, w in touchdown_points(q6, q8).items():
        pmf[pts] += p_td * w
    return pmf


def expected_points(p_td, p_fg, q6, q8):
    return dist.mean(outcome_pmf(p_td, p_fg, q6, q8))


@dataclass
class DriveModel:
    """The chain on a grid of qualities, interpolated in between."""
    params: DriveParams = field(default_factory=DriveParams)
    q6: float = 0.05
    q8: float = 0.043

    def __post_init__(self):
        self._chains = {}
        self._fresh_ep = None

    def chain(self, quality, policy=NORMAL, aggression=0.0):
        key = (quality, policy, aggression)
        if key not in self._chains:
            self._chains[key] = Chain(self.params, quality, policy, aggression)
        return self._chains[key]

    def _bracket(self, quality):
        grid = self.params.quality_grid
        q = min(grid[-1], max(grid[0], quality))
        i = bisect.bisect_left(grid, q)
        if i < len(grid) and grid[i] == q:
            return grid[i], grid[i], 0.0
        lo, hi = grid[i - 1], grid[i]
        return lo, hi, (q - lo) / (hi - lo)

    def value(self, quality, down, field_position, distance, policy=NORMAL, aggression=0.0):
        if policy != NORMAL or aggression == 0.0:
            return self._value_q(quality, down, field_position, distance, policy, 0.0)
        g = AGGRESSION_GRID
        x = min(g[-1], max(g[0], aggression))
        i = max(1, min(len(g) - 1, bisect.bisect_left(g, x)))
        lo, hi = g[i - 1], g[i]
        w = (x - lo) / (hi - lo)
        a = self._value_q(quality, down, field_position, distance, policy, lo)
        b = self._value_q(quality, down, field_position, distance, policy, hi)
        return (a[0] * (1 - w) + b[0] * w, a[1] * (1 - w) + b[1] * w)

    def _value_q(self, quality, down, field_position, distance, policy, aggression):
        lo, hi, w = self._bracket(quality)
        a = self.chain(lo, policy, aggression).value(down, field_position, distance)
        if w == 0.0:
            return a
        b = self.chain(hi, policy, aggression).value(down, field_position, distance)
        return (a[0] * (1 - w) + b[0] * w, a[1] * (1 - w) + b[1] * w)

    def fresh_ep_curve(self):
        """[(quality, EP of a fresh drive)] on the grid, increasing."""
        if self._fresh_ep is None:
            self._fresh_ep = [
                (q, expected_points(*self.chain(q).fresh(), self.q6, self.q8))
                for q in self.params.quality_grid]
        return self._fresh_ep

    def quality_for(self, ep_per_drive):
        """The quality whose fresh drive is worth ep_per_drive points."""
        curve = self.fresh_ep_curve()
        if ep_per_drive <= curve[0][1]:
            return curve[0][0]
        if ep_per_drive >= curve[-1][1]:
            return curve[-1][0]
        for (q0, e0), (q1, e1) in zip(curve, curve[1:]):
            if e0 <= ep_per_drive <= e1:
                return q0 + (q1 - q0) * (ep_per_drive - e0) / (e1 - e0)
        return curve[-1][0]

    def pmf(self, quality, down, field_position, distance, policy=NORMAL, aggression=0.0):
        p_td, p_fg = self.value(quality, down, field_position, distance, policy, aggression)
        return outcome_pmf(p_td, p_fg, self.q6, self.q8)
