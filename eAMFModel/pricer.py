"""From a game state to prices: the exact possession-level Markov chain.

What is left of the game is a run of alternating drives:

    the current drive (offense)  ->  defense  ->  offense  ->  ...

The number of drive starts still to come is Poisson, with mean
2 * drives_per_team * (share of the game left, from clock.py), less the part
of the current drive already used. The team NOT on offense gets the first
of them, so R more drives split ceil(R/2) to the defense and floor(R/2) to
the offense. Every drive's points come from the drive chain (drive.py):
the current one from its actual down, distance and field position, the
later ones as fresh drives. Each side's quality is its posterior strength
(strength.py) carried as a few quantiles.

All of that is combined by convolution, EXACTLY: the result is the full
distribution of the final score -- margin and total -- and every market is
read off it. No simulation, so the same state always gets the same price,
and a price moves only because the state did. A regulation tie goes to
overtime, which settles on a field goal (the finals fit), split by each
side's chance of scoring a drive first.
"""

import math
from dataclasses import dataclass, replace
from typing import Optional

from . import clock, dist, strength
from .drive import MUST_SCORE, NEED_TD, NORMAL, DriveModel

HOME = "home"
AWAY = "away"

# Market ids, as GAMEPLAI publishes them.
ML_HOME, ML_AWAY, SPREAD_HOME, SPREAD_AWAY, TOTAL_OVER, TOTAL_UNDER = 50, 51, 52, 53, 54, 55
MARKET_IDS = (ML_HOME, ML_AWAY, SPREAD_HOME, SPREAD_AWAY, TOTAL_OVER, TOTAL_UNDER)

POINTS_CAP = 160
MIN_MASS = 1e-12

_PRIOR_CACHE = {}


@dataclass(frozen=True)
class GameState:
    period: Optional[int]
    elapsed_in_period: float
    home_score: int
    away_score: int
    offense: Optional[str] = None     # HOME, AWAY, or None when not known
    down: Optional[int] = None
    field_position: Optional[int] = None
    distance: Optional[int] = None
    drive_age: Optional[float] = None       # messages since this drive began
    elapsed_in_half: Optional[float] = None  # messages since the half began
    clock_seconds: Optional[float] = None    # game clock left in the quarter, when known
    pending_conversion: Optional[str] = None # scored a TD, the extra point still to come
    opening_receiver: Optional[str] = None  # who had the ball first

    @property
    def has_snap(self):
        return (self.offense in (HOME, AWAY) and self.down is not None
                and self.field_position is not None and self.distance is not None
                and 1 <= self.down <= 4 and 0 < self.field_position < 100
                and self.distance > 0)


class Book:
    """The final-score distribution, and every price read off it."""

    def __init__(self, margin, total, fraction_left):
        self.margin = margin          # dist.Signed over home - away
        self.total = total            # list, pmf over home + away
        self.fraction_left = fraction_left

    @property
    def p_home(self):
        return self.margin.prob_above(0) + 0.5 * self.margin.prob_at(0)

    @staticmethod
    def _no_push(p_above, p_at):
        # A push has no 0/1 outcome and is graded out, so the probability
        # that means something is the one conditional on no push.
        rest = 1.0 - p_at
        return p_above / rest if rest > 1e-12 else 0.5

    def prob(self, market_id, line=None):
        """Probability, 0-1, that this selection wins at this line."""
        if market_id == ML_HOME:
            return self.p_home
        if market_id == ML_AWAY:
            return 1.0 - self.p_home
        if line is None:
            return None
        if market_id in (SPREAD_HOME, SPREAD_AWAY):
            # 52: home margin > line.  53: away margin > line.
            if market_id == SPREAD_HOME:
                above = self.margin.prob_above(line)
                at = self.margin.prob_at(int(line)) if float(line).is_integer() else 0.0
            else:
                above = self.margin.prob_below(-line)
                at = self.margin.prob_at(int(-line)) if float(line).is_integer() else 0.0
            return self._no_push(above, at)
        if market_id in (TOTAL_OVER, TOTAL_UNDER):
            at = 0.0
            if float(line).is_integer() and 0 <= int(line) < len(self.total):
                at = self.total[int(line)]
            over = sum(p for k, p in enumerate(self.total) if k > line)
            under = sum(p for k, p in enumerate(self.total) if k < line)
            return self._no_push(over if market_id == TOTAL_OVER else under, at)
        return None

    def fair_line(self, market_id):
        """The half-point line whose price sits closest to evens."""
        if market_id in (ML_HOME, ML_AWAY):
            return None
        if market_id in (TOTAL_OVER, TOTAL_UNDER):
            centre = dist.mean(self.total)
        else:
            centre = self.margin.mean() * (1 if market_id == SPREAD_HOME else -1)
        best = None
        for step in range(-12, 13):
            line = math.floor(centre) + 0.5 + step
            gap = abs(self.prob(market_id, line) - 0.5)
            if best is None or gap < best[0]:
                best = (gap, line)
        return best[1]


def _truncate(pmf, in_time):
    """Scale a drive's scoring by the chance the clock lets it finish."""
    if in_time >= 1.0:
        return pmf
    out = [x * in_time for x in pmf]
    out[0] += 1.0 - in_time
    return out


class Model:
    """One configured version of the pricer."""

    def __init__(self, params):
        self.params = params
        self.drives = DriveModel(params.drive, params.q6, params.q8)
        self._fresh = {}

    # -- building blocks -------------------------------------------------

    def fresh_pmf(self, quality, policy=NORMAL):
        key = (round(quality, 4), policy)
        if key not in self._fresh:
            self._fresh[key] = self.drives.pmf(quality, 1, self.params.drive.start_field,
                                               10, policy)
        return self._fresh[key]

    def side_qualities(self, prior_points, scored, fraction_left):
        """[(weight, quality)] for one side, from its posterior strength."""
        p = self.params
        per_drive = prior_points / p.drives_per_team
        expected_by_now = prior_points * (1.0 - fraction_left)
        nodes = strength.multipliers(p.prior_strength, scored, expected_by_now,
                                     p.points_per_score, p.n_slices)
        return [(w, self.drives.quality_for(per_drive * theta)) for w, theta in nodes]

    def _team_pmfs(self, qualities, n_max, current=None, policy=NORMAL, scale=1.0):
        """out[n] = pmf of points from n more fresh drives (after `current`),
        each drive's scoring scaled by `scale`."""
        out = [None] * (n_max + 1)
        mixes = [[] for _ in range(n_max + 1)]
        for w, q in qualities:
            base = dist.point(0) if current is None else current(q)
            fresh = _truncate(self.fresh_pmf(q, policy), scale)
            acc = base
            for n in range(n_max + 1):
                if n:
                    acc = dist.convolve(acc, fresh, POINTS_CAP)
                mixes[n].append((w, acc))
        for n in range(n_max + 1):
            out[n] = dist.mix(mixes[n])
        return out

    # -- the pre-match anchor --------------------------------------------

    def fit_prior(self, spread_line, total_line, ml_home=None, spread_home=None,
                  over=None, rounds=2, steps=24, on_clock=False):
        """The Prior whose kickoff book reproduces the pre-match prices.

        Half-point lines throw information away -- a -2.5 at 46% is not a
        -2.5 at 54% -- so where the pre-match model's probabilities are to
        hand, each side's expected points are moved until this model, at
        kickoff, quotes what it quoted: the total until P(over the line)
        matches, the margin until the moneyline (or failing that the spread
        at its line) matches. Monotone in each, so bisection; two rounds
        settle the small interaction between the two.

        `on_clock`: fit on the game-clock path (the pricer's structure
        differs a little with the clock known), for pricing off SCOUTING_FULL.

        At kickoff nothing has been scored and nothing was expected yet, so
        the prior strength k plays no part: versions that differ only in k
        share the answer, cached.
        """
        key = (replace(self.params, prior_strength=0.0), spread_line, total_line,
               ml_home, spread_home, over, rounds, steps, on_clock)
        if key in _PRIOR_CACHE:
            return _PRIOR_CACHE[key]
        prior = self._fit_prior(spread_line, total_line, ml_home, spread_home, over,
                                rounds, steps, on_clock)
        _PRIOR_CACHE[key] = prior
        return prior

    def _fit_prior(self, spread_line, total_line, ml_home, spread_home, over, rounds, steps,
                   on_clock=False):
        total, margin = float(total_line), float(spread_line)
        kickoff = GameState(period=1, elapsed_in_period=0.0, home_score=0, away_score=0,
                            clock_seconds=clock.QUARTER_SECONDS if on_clock else None)

        def book(t, m):
            return self.book(strength.Prior.from_lines(m, t), kickoff)

        def solve(lo, hi, value, target):
            # value() increases in its argument.
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                if value(mid) < target:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)

        for _ in range(rounds):
            if over is not None and 0.02 < over < 0.98:
                total = solve(total_line - 20, total_line + 20,
                              lambda t: book(t, margin).prob(TOTAL_OVER, total_line), over)
            if ml_home is not None and 0.02 < ml_home < 0.98:
                margin = solve(spread_line - 20, spread_line + 20,
                               lambda m: book(total, m).p_home, ml_home)
            elif spread_home is not None and 0.02 < spread_home < 0.98:
                margin = solve(spread_line - 20, spread_line + 20,
                               lambda m: book(total, m).prob(SPREAD_HOME, spread_line),
                               spread_home)
        return strength.Prior.from_lines(margin, total)

    # -- the price -------------------------------------------------------

    def book(self, prior, state):
        p = self.params
        clock_left = self._drives_of_clock(state)
        future_scale = 1.0
        if state.clock_seconds is not None:
            left, future_scale = self._on_the_clock(state, clock_left)
        else:
            left = clock.remaining(p.clock, state.period, state.elapsed_in_period,
                                   state.elapsed_in_half)
        f = left.fraction
        home_q = self.side_qualities(prior.home_points, state.home_score, f)
        away_q = self.side_qualities(prior.away_points, state.away_score, f)

        if state.period is not None and state.period > clock.REGULATION_PERIODS:
            return self._overtime(state, home_q, away_q, f)

        kickoff = (state.has_snap and state.down == 1 and state.distance == 10
                   and state.field_position == p.kickoff_field)
        # The half's clock can end the current drive before it does. Clock
        # left is in messages (time, not scoring share), counted in average
        # drives' worth; the time a drive still needs is `drive_time` of a
        # whole drive, scaled by how much of the field is left to cover --
        # a drive on the 1 needs a snap or two, one from its own 20 all of
        # it (see _in_time).
        policies, kill = self._endgame(state, clock_left)
        # The half's scoring falls away (clock.py); the drive in progress
        # shares in that, not just the count of drives to come.
        rate_now = clock.intensity(p.clock, state.period, state.elapsed_in_period,
                                   state.clock_seconds)
        if state.clock_seconds is not None and state.period in (3, 4):
            shares = p.clock.half_shares
            rate_now *= shares[1] / shares[0]
        rate_now = min(1.0, rate_now)
        slow = rate_now ** p.current_intensity
        if state.has_snap and not kickoff:
            offense = state.offense
            need = min(1.3, max(0.1, (100 - state.field_position) / 75.0))
            in_time = self._in_time(clock_left, need) * slow
            keep = 1.0 - p.kill_score if kill.get(offense) == "kneel" else 1.0
            current = self._current_drive(state, in_time * keep, policies[offense])
            part = self._part_left(state) + (p.kill_time if kill.get(offense) else 0.0)
            branches = [(1.0, offense, current, part)]
        elif state.offense in (HOME, AWAY):
            offense = state.offense
            in_time = self._in_time(clock_left, 1.0) * slow
            keep = 1.0 - p.kill_score if kill.get(offense) == "kneel" else 1.0
            part = 1.0 + (p.kill_time if kill.get(offense) else 0.0)
            branches = [(1.0, offense, self._fresh_drive(in_time * keep, policies[offense]),
                         part)]
        else:
            in_time = self._in_time(clock_left, 1.0) * slow
            branches = [(0.5, side, self._fresh_drive(in_time, policies[side]), 1.0)
                        for side in (HOME, AWAY)]

        # Who gets the ball first in the second half: whoever did not get it
        # first in the game. Unknown -> both, half each.
        if left.next_half > 0 and state.opening_receiver in (HOME, AWAY):
            second = [(1.0, AWAY if state.opening_receiver == HOME else HOME)]
        elif left.next_half > 0:
            second = [(0.5, HOME), (0.5, AWAY)]
        else:
            second = [(1.0, None)]

        margin = dist.Signed.empty(-POINTS_CAP, POINTS_CAP)
        total = [0.0] * (2 * POINTS_CAP + 1)
        tied = {}
        rate = 2.0 * p.drives_per_team
        next_half = self.drive_count(rate * left.next_half, rate * rate * left.next_half_var)
        for weight, offense, current, part in branches:
            # The current drive is the offense's; the rest of this half
            # alternates starting with the defense.
            # On the real clock the count's spread also carries how long the
            # current drive will take: a quick turnover hands the ball back
            # with time on it.
            drive_var = ((p.drive_time_cv * p.drive_time * part) ** 2
                         if state.clock_seconds is not None else 0.0)
            this_half_after = self.drive_count(
                max(0.0, rate * left.this_half - part),
                rate * rate * left.this_half_var + drive_var)
            counts = {}
            for w2, receiver in second:
                for r1, p1 in enumerate(this_half_after):
                    off_n, def_n = r1 // 2, (r1 + 1) // 2
                    for r2, p2 in enumerate(next_half if receiver else [1.0]):
                        if receiver is None or r2 == 0:
                            o, d_ = off_n, def_n
                        elif receiver == offense:
                            o, d_ = off_n + (r2 + 1) // 2, def_n + r2 // 2
                        else:
                            o, d_ = off_n + r2 // 2, def_n + (r2 + 1) // 2
                        key = (o, d_)
                        counts[key] = counts.get(key, 0.0) + w2 * p1 * p2
            n_off = max(o for o, _ in counts)
            n_def = max(d_ for _, d_ in counts)
            off_q, def_q = (home_q, away_q) if offense == HOME else (away_q, home_q)
            defense = AWAY if offense == HOME else HOME
            off = self._team_pmfs(off_q, n_off, current, policies[offense], future_scale)
            dfn = self._team_pmfs(def_q, n_def, None, policies[defense], future_scale)
            if state.pending_conversion in (HOME, AWAY):
                extra = self._conversion_pmf()
                if state.pending_conversion == offense:
                    off = [dist.convolve(pmf, extra, POINTS_CAP) for pmf in off]
                else:
                    dfn = [dist.convolve(pmf, extra, POINTS_CAP) for pmf in dfn]
            # For each offense count, the defense's points mixed over its counts.
            by_off = {}
            for (o, d_), pr in counts.items():
                if pr > MIN_MASS:
                    by_off.setdefault(o, []).append((pr, dfn[d_]))
            for o, mixture in by_off.items():
                mass = sum(pr for pr, _ in mixture)
                other = dist.mix([(pr / mass, pmf) for pr, pmf in mixture])
                home, away = (off[o], other) if offense == HOME else (other, off[o])
                self._accumulate(margin, total, tied, weight * mass, home, away, state)

        self._settle_ties(margin, total, tied, home_q, away_q)
        return Book(margin, total, f)

    def drive_count(self, mean, clock_var):
        """Distribution of how many drives start: a discretised normal.

        Variance is the game's own (count_dispersion per expected drive)
        plus what the unknown clock adds -- so late on, when the model
        cannot see how much time is left, the count stays honestly vague.
        """
        if mean <= 1e-9 and clock_var <= 1e-9:
            return [1.0]
        var = max(0.05, self.params.count_dispersion * mean + clock_var)
        sd = math.sqrt(var)
        hi = int(mean + 6 * sd) + 1
        w = [math.exp(-0.5 * (k - mean) ** 2 / var) for k in range(hi + 1)]
        return dist.normalise(w)

    def _part_left(self, state):
        """How much of the current drive is still to play, in drives."""
        p = self.params
        if state.drive_age is None:
            return p.current_drive_share
        drive_messages = p.clock.regulation / (2.0 * p.drives_per_team)
        return min(1.0, max(0.15, 1.0 - state.drive_age / drive_messages))

    def _current_drive(self, state, in_time=1.0, policy=NORMAL):
        down, field, togo = state.down, state.field_position, state.distance

        def pmf(quality):
            return _truncate(self.drives.pmf(quality, down, field, togo, policy), in_time)
        return pmf

    def _fresh_drive(self, in_time=1.0, policy=NORMAL):
        def pmf(quality):
            return _truncate(self.fresh_pmf(quality, policy), in_time)
        return pmf

    def _on_the_clock(self, state, clock_left):
        """(Remaining, scale of later drives' scoring) off the real clock.

        With the game clock known, how many drives are left is a matter of
        TIME: this half's clock in drives' worth, and a whole half more if
        this is the first. The second half's lower scoring then comes off
        what each drive is worth, not off how many there are: its drives
        score at half_shares[1] / half_shares[0] of the first half's, and
        through the half at the rate w(u) (clock.py), so a later drive from
        position u is worth the average of w over what is left, (1 - u)^a.
        """
        p = self.params
        rate = 2.0 * p.drives_per_team
        if state.period is None:
            return clock.Remaining(0.5, 0.0, 0.5, 0.0), 1.0
        if state.period > clock.REGULATION_PERIODS:
            return clock.Remaining(0.0, 0.0, 0.0, 0.0), 1.0
        shares = p.clock.half_shares
        second_level = shares[1] / shares[0] if shares[0] > 0 else 1.0
        this_half = clock_left / rate
        if state.period <= 2:
            n_this, n_next = clock_left, 0.5 * rate
            scale = (n_this + n_next * second_level) / max(1e-9, n_this + n_next)
            return clock.Remaining(this_half, 0.0, 0.5, 0.0), min(1.0, scale)
        a = p.clock.half_slopes[1]
        u = clock.position_in_half(state.period, state.clock_seconds)
        scale = second_level * max(0.0, 1.0 - u) ** a
        return clock.Remaining(this_half, 0.0, 0.0, 0.0), min(1.0, scale)

    def _conversion_pmf(self):
        """Points still to come from a touchdown's conversion: 0, 1 or 2."""
        q6, q8 = self.params.q6, self.params.q8
        return [q6, 1.0 - q6 - q8, q8]

    def _drives_of_clock(self, state):
        """Clock left in the half, in average drives' worth: seconds when
        the feed gives the game clock, messages when it does not."""
        p = self.params
        if state.clock_seconds is not None:
            if state.period is None or state.period > clock.REGULATION_PERIODS:
                return 0.0
            seconds = max(0.0, min(float(state.clock_seconds), clock.QUARTER_SECONDS))
            if state.period % 2 == 1:
                seconds += clock.QUARTER_SECONDS
            regulation = clock.REGULATION_PERIODS * clock.QUARTER_SECONDS
            return seconds * 2.0 * p.drives_per_team / regulation
        messages = clock.messages_left_in_half(p.clock, state.period, state.elapsed_in_period)
        return messages * 2.0 * p.drives_per_team / p.clock.regulation

    def _in_time(self, clock_left, need):
        """P(the drive finishes before the half does).

        The clock a drive still needs is normal around `drive_time` of a
        whole drive's worth, scaled by `need`, with spread `drive_time_cv`
        of that. Not exponential: a drive with half the game ahead of it
        always has time, and one on the 1 needs a snap or two.
        """
        p = self.params
        t = p.drive_time * need
        if t <= 0:
            return 1.0
        z = (clock_left - t) / (p.drive_time_cv * t)
        return 0.5 * math.erfc(-z / math.sqrt(2.0))

    def _endgame(self, state, clock_left):
        """End-of-game behaviour: ({side: 4th-down policy}, {side: kill}).

        Late in the second half -- fewer than `late_drives` drives' worth of
        clock left -- the two sides stop playing the same game:

          behind by 1-3    MUST_SCORE: never punts, still kicks
          behind by 4-8    NEED_TD: never punts, never kicks (3 is no use)
          behind by 9+     MUST_SCORE: needs more than one score, any will do
          ahead, on the ball
                           runs the clock: the drive eats `kill_time` more
                           drives' worth of time, so fewer are left for the
                           other side; ahead by `kill_lead` or more it
                           kneels, scoring `kill_score` less
        """
        p = self.params
        policies = {HOME: NORMAL, AWAY: NORMAL}
        kill = {}
        if state.period is None or state.period < 3 or state.period > clock.REGULATION_PERIODS:
            return policies, kill
        if clock_left >= p.late_drives:
            return policies, kill
        margin = state.home_score - state.away_score
        for side, lead in ((HOME, margin), (AWAY, -margin)):
            if lead < 0:
                behind = -lead
                policies[side] = NEED_TD if 4 <= behind <= 8 else MUST_SCORE
            elif lead > 0:
                kill[side] = "kneel" if lead >= p.kill_lead else "run"
        return policies, kill

    @staticmethod
    def _accumulate(margin, total, tied, weight, home, away, state):
        hs, as_ = state.home_score, state.away_score
        base = hs - as_
        tbase = hs + as_
        b_items = [(j, y) for j, y in enumerate(away) if y > MIN_MASS]
        for i, x in enumerate(home):
            if x <= MIN_MASS:
                continue
            wx = weight * x
            for j, y in b_items:
                m = wx * y
                d = base + i - j
                t = tbase + i + j
                if -POINTS_CAP <= d <= POINTS_CAP:
                    margin.masses[d + POINTS_CAP] += m
                if t < len(total):
                    total[t] += m
                    if d == 0:
                        tied[t] = tied.get(t, 0.0) + m

    def _first_score_home(self, home_q, away_q):
        """P(home scores first) in sudden death, coin toss for the ball."""
        def score_rate(qs):
            pmf = dist.mix([(w, self.fresh_pmf(q)) for w, q in qs])
            return 1.0 - pmf[0]
        a, b = score_rate(home_q), score_rate(away_q)
        both_miss = (1 - a) * (1 - b)
        home_first = a / (1 - both_miss)
        away_first = b / (1 - both_miss)
        return 0.5 * home_first + 0.5 * (1 - away_first)

    def _settle_ties(self, margin, total, tied, home_q, away_q):
        """A regulation tie goes to overtime and is won by a field goal."""
        tie = margin.prob_at(0)
        if tie <= 0:
            return
        home_share = self._first_score_home(home_q, away_q)
        d = self.params.ot_margin
        margin.masses[POINTS_CAP] = 0.0
        margin.add(d, tie * home_share)
        margin.add(-d, tie * (1 - home_share))
        for t, x in tied.items():
            total[t] -= x
            if t + d < len(total):
                total[t + d] += x

    def _overtime(self, state, home_q, away_q, f):
        margin = dist.Signed.empty(-POINTS_CAP, POINTS_CAP)
        total = [0.0] * (2 * POINTS_CAP + 1)
        diff = state.home_score - state.away_score
        now = state.home_score + state.away_score
        d = self.params.ot_margin
        if diff != 0:
            margin.add(diff, 1.0)
            total[now] = 1.0
            return Book(margin, total, 0.0)
        home_share = self._first_score_home(home_q, away_q)
        if state.has_snap:
            # This drive first, then sudden death with the ball turned over.
            qs = home_q if state.offense == HOME else away_q
            cur = dist.mix([(w, self.drives.pmf(q, state.down, state.field_position,
                                                state.distance)) for w, q in qs])
            scores_now = 1.0 - cur[0]
            other_q = away_q if state.offense == HOME else home_q
            other = 1.0 - dist.mix([(w, self.fresh_pmf(q)) for w, q in other_q])[0]
            mine = 1.0 - dist.mix([(w, self.fresh_pmf(q)) for w, q in qs])[0]
            other_wins = other / (1 - (1 - other) * (1 - mine))
            off_wins = scores_now + (1 - scores_now) * (1 - other_wins)
            home_share = off_wins if state.offense == HOME else 1 - off_wins
        margin.add(d, home_share)
        margin.add(-d, 1 - home_share)
        total[now + d] = 1.0
        return Book(margin, total, 0.0)
