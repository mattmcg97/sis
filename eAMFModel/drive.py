"""The current drive, as a play-by-play Markov chain.

State is (down, field position, line to gain). Field position is yards from
the offense's own goal line, 1-99, which is how INPLAY_FIELD_POSITION_PERIOD
reports it: the kickoff spot reads 35, a touchdown is crossing 100. Each
snap is one transition:

  turnover          drive over, no points
  gain g            g drawn from the play kernel (losses, no-gains, short
                    gains, big plays); reaching 100 is a touchdown,
                    reaching the line to gain is a fresh 1st-and-10
  4th down          kick a field goal if in range, go for it on 4th and
                    short past midfield-ish, otherwise punt (no points)

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
MUST_SCORE = "must_score"
NEED_TD = "need_td"
POLICIES = (NORMAL, MUST_SCORE, NEED_TD)


@dataclass(frozen=True)
class DriveParams:
    p_turnover: float = 0.018       # interception or fumble, per snap
    p_zero: float = 0.22            # incompletion / stuffed
    p_loss: float = 0.08            # sack or tackle for loss
    loss_max: int = 7               # losses uniform on 1..loss_max
    gain_mean: float = 10.0         # ordinary gains: 1 + geometric
    p_big: float = 0.18             # big plays, per snap
    big_mean: float = 32.0          # big plays: 10 + geometric
    fg_max_distance: int = 57       # longest attempt; kick = 117 - field
    fg_mid: float = 52.0            # 50% make distance
    fg_scale: float = 4.0           # steepness of the make curve
    go_max_togo: int = 2            # 4th-and-this-or-less gets gone for...
    go_min_field: int = 45          # ...from here on, when not kicking
    start_field: int = 25           # typical fresh-drive start, for the level
    quality_grid: tuple = (0.35, 0.45, 0.56, 0.68, 0.8, 0.9, 1.0, 1.12, 1.26, 1.42,
                          1.6, 1.85, 2.2, 2.7)


def kick_distance(field_position):
    return FIELD - field_position + 17


def fg_make(params, field_position):
    distance = kick_distance(field_position)
    if distance > params.fg_max_distance:
        return 0.0
    return 1.0 / (1.0 + math.exp((distance - params.fg_mid) / params.fg_scale))


def gain_kernel(params, quality):
    """Per-snap outcome: (p_turnover, {gain: prob}) with gains in yards.

    Quality multiplies the ordinary gain mean and the big-play rate, and
    divides the rates of the snaps that go nowhere -- turnovers and losses
    in full, incompletions by its square root -- so a better offense both
    gains more and stalls less. Ordinary gains take what is left.
    """
    p_big = min(0.35, params.p_big * quality)
    mean = max(1.05, params.gain_mean * quality)
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

    def __init__(self, params, quality, policy=NORMAL):
        self.params = params
        self.quality = quality
        self.policy = policy
        self.p_turnover, kernel = gain_kernel(params, quality)
        low = -params.loss_max
        # gains[i] = P(gain == low + i), gains index up to +FIELD.
        self.low = low
        self.gains = [kernel.get(g, 0.0) for g in range(low, FIELD + 1)]
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
        gains, low = self.gains, self.low
        keep = 1.0 - self.p_turnover
        for L in range(FIELD, 1, -1):
            for d in range(4, 0, -1):
                row_td = [0.0] * L
                row_fg = [0.0] * L
                nxt_td = td[d + 1][L] if d < 4 else None
                nxt_fg = fg[d + 1][L] if d < 4 else None
                for y in range(1, L):
                    t = L - y
                    if d == 4:
                        make = fg_make(p, y) if self.policy != NEED_TD else 0.0
                        if self.policy == NORMAL:
                            if make > 0 and not (t <= p.go_max_togo and make < 0.9):
                                row_fg[y] = make
                                continue
                            if not (t <= p.go_max_togo and y >= p.go_min_field):
                                continue          # punt
                        elif self.policy == MUST_SCORE and make >= 0.5:
                            row_fg[y] = make
                            continue
                        # otherwise: go for it
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
                    row_td[y] = keep * a_td
                    row_fg[y] = keep * a_fg
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

    def chain(self, quality, policy=NORMAL):
        key = (quality, policy)
        if key not in self._chains:
            self._chains[key] = Chain(self.params, quality, policy)
        return self._chains[key]

    def _bracket(self, quality):
        grid = self.params.quality_grid
        q = min(grid[-1], max(grid[0], quality))
        i = bisect.bisect_left(grid, q)
        if i < len(grid) and grid[i] == q:
            return grid[i], grid[i], 0.0
        lo, hi = grid[i - 1], grid[i]
        return lo, hi, (q - lo) / (hi - lo)

    def value(self, quality, down, field_position, distance, policy=NORMAL):
        lo, hi, w = self._bracket(quality)
        a = self.chain(lo, policy).value(down, field_position, distance)
        if w == 0.0:
            return a
        b = self.chain(hi, policy).value(down, field_position, distance)
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

    def pmf(self, quality, down, field_position, distance, policy=NORMAL):
        p_td, p_fg = self.value(quality, down, field_position, distance, policy)
        return outcome_pmf(p_td, p_fg, self.q6, self.q8)
