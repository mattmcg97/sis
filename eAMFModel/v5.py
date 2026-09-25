"""v5: price PLAY_OVER snapshots with the play-by-play simulation (sim5.py).

v5 starts as a copy of v4 (v4.py is left as it is: NB2's pre-match
prior, common random numbers with a fixed seed per match, player profiles,
its own even lines) and adds play calling by game state (sim5.py):

  * the play call -- one that stops the clock (an incompletion or out of
    bounds: a pass, mostly) or one that keeps it running (a run, a
    completion in bounds) -- as likely as real offenses make it in that
    quarter, 40-second slice of it and lead, with the yards then drawn
    from real plays of that kind;
  * the clock each kind of play uses in that state (a leader milking the
    play clock, a trailer hurrying);
  * the rubber band: offenses' efficiency by state, from their real
    first-down success, and a pull per half (fit_rubber_band) solved so
    that from real in-game states the simulated share of a lead that comes
    back by the end matches the real one. With the play-level fit it
    came out at zero: leads already come back as fast as in real games.

The fourth quarter is left to v4's end-game tables: on held-out weeks
every shift cost a little there. See the eAMFModel README for the tests.

What follows is v3's description, which still holds.

v3: price PLAY_OVER snapshots with the play-by-play simulation (sim.py).

What a v3 price knows about the two players:

  prior       each offense's efficiency theta, set so that the simulation
              played from kickoff scores each side's expected points from
              our own pre-match model, NB2 (`PriorGrid`, `fit_means`). v5
              reads nothing of GAMEPLAI's: its prices are only what v5 is
              compared with;
  efficiency  how the offense has actually done at moving the chains --
              first-down success on each snap against the league's rate in
              the same situation -- which moves its theta (`Efficiency`).
              Scoring does not: points already on the board are points, and
              the price of what is to come reacts to how the players have
              moved the ball, not to how many of their drives happened to
              end in the end zone;
  profiles    optionally, each player's 4th-down aggression and pace
              (players.Book).

The simulation does the rest: the situation (a lead, the clock) decides how
the rest of the game is played, so a leader who keeps moving the chains
bleeds the clock and the total dies with it.
"""

import math
import multiprocessing as mp
from collections import Counter

import numpy as np

from . import nb2_prior, playover, players, sim5 as sim
from .pricer import HOME

GRID = np.round(np.linspace(-0.8, 0.8, 17), 3)
MARGIN_MAX = 100
TOTAL_MAX = 160
KAPPA = 40.0          # prior precision on theta, in Fisher information (fitted: fit_kappa)


# --------------------------------------------------------------------------
# The prior: theta from NB2's expected points (prior_theta)

def _distributions(home, away):
    """(margin pmf over -MARGIN_MAX..MARGIN_MAX, total pmf over 0..TOTAL_MAX)."""
    m = np.clip(home - away, -MARGIN_MAX, MARGIN_MAX) + MARGIN_MAX
    t = np.clip(home + away, 0, TOTAL_MAX)
    mp_ = np.bincount(m, minlength=2 * MARGIN_MAX + 1) / len(m)
    tp = np.bincount(t, minlength=TOTAL_MAX + 1) / len(t)
    return mp_, tp


def _graded(side, at):
    """P(side | not a push); an even 0.5 when a push is all that is left."""
    rest = 1 - at
    return np.where(rest > 1e-12, side / np.maximum(1e-12, rest), 0.5)


def even_line(pmf, offset):
    """The half-point line nearest to even money: the L = k + 0.5 whose
    P(X > L) is closest to a half (the median, rounded to a half point, so
    a line never pushes). Where several lines are equally near -- no
    outcome falls between them, as late in a game -- the one nearest the
    mean."""
    cdf = np.cumsum(pmf)[:-1]
    gap = np.abs(cdf - 0.5)
    # never a line below everything left to happen (an over already won):
    # a dead game -- no more points -- is quoted just above the score
    gap = np.where(cdf <= 1e-12, np.inf, gap) if (cdf > 1e-12).any() else gap
    best = np.flatnonzero(gap <= gap.min() + 1e-12)
    x = np.arange(len(pmf)) - offset
    mean = float((pmf * x).sum() / max(1e-300, pmf.sum()))
    k = best[np.argmin(np.abs(best - offset + 0.5 - mean))]
    return float(k - offset) + 0.5


def prob_above(pmf, offset, line):
    """P(X > line) with a push graded out, off a pmf whose index i is X = i - offset."""
    x = np.arange(pmf.shape[-1]) - offset
    above = pmf[..., x > line].sum(-1)
    at = pmf[..., x == line].sum(-1) if float(line).is_integer() else 0.0
    return _graded(above, at)


def prob_below(pmf, offset, line):
    x = np.arange(pmf.shape[-1]) - offset
    below = pmf[..., x < line].sum(-1)
    at = pmf[..., x == line].sum(-1) if float(line).is_integer() else 0.0
    return _graded(below, at)


def market_prob(market, line, margin_pmf, total_pmf):
    if market == 50:
        return prob_above(margin_pmf, MARGIN_MAX, 0.0)
    if market == 51:
        return prob_below(margin_pmf, MARGIN_MAX, 0.0)
    if market == 52:
        return prob_above(margin_pmf, MARGIN_MAX, line)
    if market == 53:
        return prob_below(margin_pmf, MARGIN_MAX, -line)
    if market == 54:
        return prob_above(total_pmf, 0, line)
    if market == 55:
        return prob_below(total_pmf, 0, line)
    raise ValueError(market)


class PriorGrid:
    """The simulation from kickoff on a grid of (theta_home, theta_away):
    margin and total distributions at each point, interpolated between."""

    def __init__(self, margin, total, grid=GRID):
        self.margin, self.total, self.grid = margin, total, grid

    @classmethod
    def build(cls, tables, n_paths=6000, seed=0, grid=GRID, theta_sd=0.0, **sim_kw):
        """`theta_sd`: how far each path's efficiencies are drawn around the
        grid point -- what the pre-match price does not know."""
        g = len(grid)
        start = sim.Start(g * g)
        th, ta = np.meshgrid(grid, grid, indexing="ij")
        start.theta = np.stack([th.ravel(), ta.ravel()], axis=1)
        rng = np.random.default_rng(seed)
        start.team = rng.integers(0, 2, g * g).astype(np.int8)        # who kicks: a coin
        start.kicks_second_half = 1 - start.team                     # the receiver kicks the 2nd half
        sds = np.full((g * g, 2), theta_sd) if theta_sd else None
        home, away = sim.simulate(tables, start, n_paths, rng, theta_sd=sds, **sim_kw)
        margin = np.zeros((g, g, 2 * MARGIN_MAX + 1))
        total = np.zeros((g, g, TOTAL_MAX + 1))
        for i in range(g * g):
            a, b = divmod(i, g)
            margin[a, b], total[a, b] = _distributions(home[i], away[i])
        return cls(margin, total, grid)

    def save(self, path):
        np.savez_compressed(path, margin=self.margin, total=self.total, grid=self.grid)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(z["margin"], z["total"], z["grid"])


def _surface_fn(grid, step):
    fine = np.arange(grid[0], grid[-1] + 1e-9, step)
    gi = np.interp(fine, grid, np.arange(len(grid)))
    lo = np.minimum(len(grid) - 2, gi.astype(int))
    w = gi - lo

    def surface(v):
        return (v[lo][:, lo] * np.outer(1 - w, 1 - w) + v[lo + 1][:, lo] * np.outer(w, 1 - w)
                + v[lo][:, lo + 1] * np.outer(1 - w, w) + v[lo + 1][:, lo + 1] * np.outer(w, w))
    return fine, surface


def fit_means(grid, home_points, away_points, step=0.01):
    """(theta_home, theta_away) whose simulated games average these points
    for each side -- the prior from a pre-match model's expected scores
    (NB2's), matched on the log scale."""
    fine, surface = _surface_fn(grid.grid, step)
    x_m = np.arange(grid.margin.shape[-1]) - MARGIN_MAX
    x_t = np.arange(grid.total.shape[-1])
    e_margin = surface((grid.margin * x_m).sum(-1))
    e_total = surface((grid.total * x_t).sum(-1))
    e_home, e_away = (e_total + e_margin) / 2, (e_total - e_margin) / 2
    err = (np.log(np.maximum(0.1, e_home)) - np.log(max(0.1, home_points))) ** 2 \
        + (np.log(np.maximum(0.1, e_away)) - np.log(max(0.1, away_points))) ** 2
    i, j = np.unravel_index(np.argmin(err), err.shape)
    return float(fine[i]), float(fine[j])


LEAGUE_THETA = (0.0, 0.0)


def prior_theta(grid, means):
    """A match's (theta_home, theta_away): off our pre-match model's
    expected points (NB2's) when given, else league-average offenses.
    Nothing of GAMEPLAI's is ever read -- its prices are only what v5 is
    compared with."""
    return LEAGUE_THETA if means is None else fit_means(grid, *means)


def resolve_sides(match_rows):
    """The rows with TEAM_A's side filled in. SCOUTING_FULL's TEAM_A is
    PLAYER_1, the home side, on every row the probe has seen (142,000 of
    142,000), so a match whose side the scores could not confirm -- no
    scoring message to check it against yet -- is taken as home rather
    than dropped."""
    if match_rows and all(r.get("team_a_side") in ("home", "away") for r in match_rows):
        return match_rows
    return [r if r.get("team_a_side") in ("home", "away") else dict(r, team_a_side="home")
            for r in match_rows]


# --------------------------------------------------------------------------
# Efficiency: first-down success, snap by snap

class Efficiency:
    """Each offense's theta, moved from its prior by its first-down success.

    Under the simulation's tilt an offense with efficiency theta converts a
    snap whose league success rate is f with probability f ** exp(-theta).
    Each snap adds its score and Fisher information; theta is one Newton
    step from the prior, shrunk by `kappa` snaps' worth of information:

        theta = theta0 + sum(score) / (sum(information) + kappa)
    """

    def __init__(self, tables, theta0, kappa=KAPPA):
        self.tables = tables
        self.theta0 = np.array(theta0, dtype=float)
        self.kappa = kappa
        self.score = np.zeros(2)
        self.info = np.zeros(2)

    def expected(self, side, key, period=None):
        """P(first down) for this side's prior theta, with the league's
        offset for the quarter as the simulation applies it."""
        f = min(0.995, max(0.005, self.tables.success_of(key)))
        offset = self.tables.period_theta[min(int(period), 5)] if period else 0.0
        return f ** math.exp(-(self.theta0[side] + offset))

    def add(self, side, key, success, period=None):
        p = min(0.995, max(0.005, self.expected(side, key, period)))
        lp = math.log(p)
        self.score[side] += -lp * (success - p) / (1 - p)
        self.info[side] += lp * lp * p / (1 - p)

    def theta(self):
        return self.theta0 + self.score / (self.info + self.kappa)

    def sd(self):
        return 1.0 / np.sqrt(self.info + self.kappa)


def fit_kappa(tables, grid, matches, kappas=(5, 10, 20, 40, 80, 160, 320, 1e9)):
    """Predictive log likelihood of every snap's first-down success from the
    snaps before it, for each kappa. Returns {kappa: mean log likelihood}."""
    out = {}
    per_match = []
    for code, rows in matches.items():
        rows = resolve_sides(rows)
        theta0 = LEAGUE_THETA
        a_home = rows[0]["team_a_side"] == "home"
        recs = [(0 if (r["offense"] == "TEAM_A") == a_home else 1, r["key"], r["success"],
                 r["period"]) for r in sim.snap_records(rows)]
        per_match.append((theta0, recs))
    for k in kappas:
        ll, n = 0.0, 0
        for theta0, recs in per_match:
            eff = Efficiency(tables, theta0, k)
            for side, key, success, period in recs:
                th = eff.theta()[side] + tables.period_theta[min(period, 5)]
                f = min(0.995, max(0.005, tables.success_of(key)))
                p = min(0.995, max(0.005, f ** math.exp(-th)))
                ll += math.log(p if success else 1 - p)
                n += 1
                eff.add(side, key, success, period)
        out[k] = ll / max(1, n)
    return out


# --------------------------------------------------------------------------
# Snapshots

def start_from(state):
    """sim.Start fields for one playover.state_for GameState."""
    side = lambda s: 0 if s == HOME else 1
    if state.pending_conversion is not None:
        phase, team = sim.CONV, side(state.pending_conversion)
        down, dist, y = 1, 10, 25
    elif state.down is None:
        phase, team = sim.KICK, 1 - side(state.offense)
        down, dist, y = 1, 10, 25
    else:
        phase, team = sim.SCRIM, side(state.offense)
        down, dist, y = state.down, max(1, state.distance), state.field_position
    return dict(period=state.period, clock=state.clock_seconds, phase=phase, team=team, down=down,
                dist=dist, y=min(99, max(1, y)), home=state.home_score, away=state.away_score,
                kicks_second_half=side(state.opening_receiver) if state.opening_receiver else -1)


def _fill(start, i, fields):
    for k, v in fields.items():
        getattr(start, k)[i] = v


class Variant:
    """One way of running v5 (what it reacts to). By default: no in-game
    efficiency update (it measured as noise), player profiles on."""

    def __init__(self, name, kappa=KAPPA, react=False, theta_sd=False, profiles=True,
                 pace=True, sim_kw=None, grid=None):
        """`theta_sd`: draw each path's efficiencies from the posterior (the
        prior grid in `grid` must then have been built with sd 1/sqrt(kappa))."""
        self.name, self.kappa, self.react, self.grid = name, kappa, react, grid
        self.theta_sd, self.profiles, self.pace = theta_sd, profiles, pace
        self.sim_kw = sim_kw or {}


def match_seed(match_code):
    """A fixed seed per match for the common random numbers."""
    import zlib
    return zlib.crc32(str(match_code).encode("utf-8"))


def price_states(tables, theta0, variant, snaps, a_home, states, messages, prof, n_paths, rng,
                 seed=None):
    """(margin pmf, total pmf) for each state, in message order.

    `snaps` are the match's sim.snap_records (for the efficiency update),
    `states` playover.state_for GameStates at `messages`, `prof` the two
    players' players.Profile."""
    v = variant
    start = sim.Start(len(messages))
    sds = np.zeros((len(messages), 2))
    eff = Efficiency(tables, theta0, v.kappa)
    k = 0
    for i, msg in enumerate(messages):
        while k < len(snaps) and snaps[k]["message"] <= msg:
            s = snaps[k]
            eff.add(0 if (s["offense"] == "TEAM_A") == a_home else 1, s["key"], s["success"],
                    s["period"])
            k += 1
        _fill(start, i, start_from(states[i]))
        start.theta[i] = eff.theta() if v.react else theta0
        sds[i] = eff.sd() if v.react else 1.0 / math.sqrt(v.kappa)
        if v.profiles:
            start.aggression[i] = (prof[0].aggression, prof[1].aggression)
        if v.pace:
            start.pace[i] = (prof[0].pace, prof[1].pace)
    kw = dict(theta_sd=sds if v.theta_sd else None, seed=seed, **v.sim_kw)
    home, away = sim.simulate(tables, start, n_paths, rng, **kw)
    books = [_distributions(home[i], away[i]) for i in range(len(messages))]
    if not np.any(tables.inplay_theta):
        return books
    # the total from the in-play run (fit_rest_of_game), on the same luck:
    # scoring shifted to what really comes from inside a game moves the
    # total's line and leaves the margin -- moneyline and spread -- as they
    # were, where the shift had cost a little
    home, away = sim.simulate(tables, start, n_paths, rng, in_play=True, **kw)
    return [(mp_, _distributions(home[i], away[i])[1]) for i, (mp_, _) in enumerate(books)]


def pool_context():
    """fork where there is one (Linux, macOS); spawn on Windows, which has no
    fork. Workers are top-level functions with picklable jobs, so either works."""
    return mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")


def _grade_matches(job):
    (matches, tables_path, grid_path, variants, n_paths, seed, book, handles, require_live,
     priors) = job
    tables = sim.Tables.load(tables_path)
    grids = {None: PriorGrid.load(grid_path)}
    for v in variants:
        if v.grid and v.grid not in grids:
            grids[v.grid] = PriorGrid.load(v.grid)
    graded, skipped = [], Counter()
    rng = np.random.default_rng(seed)
    for code, match_rows in matches:
        match_rows = resolve_sides(match_rows)
        if priors is not None:
            # our own pre-match model (NB2): nothing of GAMEPLAI's is read
            means = priors.get(code)
            if means is None:
                skipped["no_prematch_prediction"] += 1
                continue
            theta0s = {g: fit_means(grid, *means) for g, grid in grids.items()}
        else:
            # a model without NB2: league-average offenses, never prod's quotes
            theta0s = {g: LEAGUE_THETA for g in grids}
        a_home = match_rows[0]["team_a_side"] == "home"
        snaps = sim.snap_records(match_rows)
        pair = (handles.get(code) if handles else None) or handles_of(match_rows)
        prof = ((book.profile(pair[0]), book.profile(pair[1])) if (book and pair)
                else (players.Profile(), players.Profile()))
        # the rows to price, and their states
        todo = []
        for row in playover.rows_for_match(match_rows):
            if row.prod_outcome is None:
                skipped["push_or_unresolved"] += 1
                continue
            if require_live and not row.prod_live:
                skipped["prod_not_live"] += 1
                continue
            state, why = playover.state_for(row.source)
            if state is None:
                skipped[why] += 1
                continue
            todo.append((row, state))
        if not todo:
            continue
        messages = sorted({row.message for row, _ in todo})
        index = {m: i for i, m in enumerate(messages)}
        states = {}
        for row, state in todo:
            states[row.message] = state
        dists = {v.name: price_states(tables, theta0s[v.grid], v, snaps, a_home,
                                      [states[m] for m in messages], messages, prof, n_paths, rng,
                                      seed=match_seed(code))
                 for v in variants}
        for row, state in todo:
            i = index[row.message]
            probs = {}
            for v in variants:
                mpmf, tpmf = dists[v.name][i]
                probs[v.name] = float(np.clip(market_prob(row.market_id, row.prod_line or 0.0,
                                                          mpmf, tpmf), 0.001, 0.999))
            row.period = state.period
            row.clock_seconds = state.clock_seconds
            row.source = None
            graded.append((code, row, probs))
    return graded, skipped


def handles_of(match_rows):
    """(home handle, away handle) off the export's own columns, or None."""
    r = match_rows[0] if match_rows else {}
    home, away = (r.get("home_handle") or "").strip(), (r.get("away_handle") or "").strip()
    return (home.upper(), away.upper()) if home and away else None


# Each quarter's scoring level (off by default: see below). fit_period_theta sets it from kickoff with
# league-average offenses; held out, from real quarter starts with NB2's
# priors, that left the fourth quarter (and overtime) about half a point
# too high and the first 0.3-0.7 too low, and every total line priced in
# the third quarter too high with it. So the build fits the levels again
# from real quarter starts: from the first row of each quarter of every
# game built on, the simulation scores in that quarter what was scored --
# and, with a level of their own, from the two-minute mark of each half
# to its end (fit_quarter_levels): Q2's two-minute drill scores about 9 of
# the quarter's 13 points, which one level for the whole quarter misses.
# Held out, it fixed the third quarter's totals but cost moneyline and
# spread about 0.001 Brier in both weeks tested, with no gain on the total,
# so it is off: v5-build keeps the kickoff fit.
QUARTER_START_FIT = False
QUARTER_ROUNDS = 6
QUARTER_PATHS = 150


def quarter_start_states(matches, grid, priors=None):
    """(quarter, GameState, theta0, points really scored in the rest of that
    quarter) at the first PLAY_OVER of every quarter 1-4 of every match.
    theta0: prior_theta off `priors` (match -> NB2's expected points; a
    match missing from them is left out), league average without.

    A quarter's points run to the next period's first row, or to the final
    score after the fourth: a quarter's last row often does not yet show
    its last score, which the feed posts on the next quarter's first row."""
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        rows = resolve_sides(rows)
        if rows[0].get("final_p1") in ("", None) or rows[0].get("final_p2") in ("", None):
            continue
        final = int(float(rows[0]["final_p1"])) + int(float(rows[0]["final_p2"]))
        theta0 = prior_theta(grid, None if priors is None else priors[code])
        firsts = {}
        for r in rows:
            p = r["period"]
            if p and p.isdigit() and p not in firsts and r["score_p1"] and r["score_p2"]:
                firsts[p] = r
        for q in (1, 2, 3, 4):
            if str(q) not in firsts:
                continue
            nxt = next((firsts[p] for p in sorted(firsts, key=int) if int(p) > q), None)
            if nxt is None and q < 4:
                continue                         # the feed stops before the quarter ends
            state, _ = playover.state_for(firsts[str(q)])
            if state is None:
                continue
            end = int(nxt["score_p1"]) + int(nxt["score_p2"]) if nxt is not None else final
            out.append((q, state, theta0, end - state.home_score - state.away_score))
    return out


def late_start_states(matches, grid, priors=None):
    """(quarter 2 or 4, GameState, theta0, points really scored from there to
    the quarter's end) at the first row of the last two minutes of each
    half -- counted, as in quarter_start_states, to the next period's first
    row or the final score."""
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        rows = resolve_sides(rows)
        if rows[0].get("final_p1") in ("", None) or rows[0].get("final_p2") in ("", None):
            continue
        final = int(float(rows[0]["final_p1"])) + int(float(rows[0]["final_p2"]))
        theta0 = prior_theta(grid, None if priors is None else priors[code])
        for q in (2, 4):
            late = next((r for r in rows if r["period"] == str(q) and r["clock_seconds"]
                         and float(r["clock_seconds"]) <= sim.LATE and r["score_p1"] and r["score_p2"]),
                        None)
            if late is None:
                continue
            nxt = next((r for r in rows if r["period"] and r["period"].isdigit() and int(r["period"]) > q
                        and r["score_p1"] and r["score_p2"]), None)
            if nxt is None and q < 4:
                continue
            state, _ = playover.state_for(late)
            if state is None:
                continue
            end = int(nxt["score_p1"]) + int(nxt["score_p2"]) if nxt is not None else final
            out.append((q, state, theta0, end - state.home_score - state.away_score))
    return out


def fit_quarter_levels(tables, quarter_items, late_items, rounds=6, n_paths=150, seed=0,
                       verbose=False):
    """tables.period_theta and tables.late_theta so that, from real quarter
    starts, the simulation scores in each quarter what was scored, and from
    the two-minute mark of each half, what was scored from there. Returns
    ({quarter: (real, simulated)}, {half's quarter: (real, simulated)})."""
    def prepared(items):
        by_q = {}
        for it in items:
            by_q.setdefault(it[0], []).append(it)
        out = {}
        for q, its in by_q.items():
            start = sim.Start(len(its))
            for i, (_, state, theta0, _) in enumerate(its):
                _fill(start, i, start_from(state))
                start.theta[i] = theta0
            out[q] = (start, float(np.mean([it[3] for it in its])))
        return out

    def scored(start, q):
        stats = {}
        sim.simulate(tables, start, n_paths, np.random.default_rng(seed + q), stats=stats,
                     common=False)
        began = float((start.home + start.away).sum()) * n_paths
        return (stats[f"points_by_q{q}"] - began) / (len(start.period) * n_paths)

    quarters, lates = prepared(quarter_items), prepared(late_items)
    got_q, got_l = {}, {}
    for rnd in range(rounds):
        for q, (start, real) in sorted(quarters.items()):
            got_q[q] = scored(start, q)
            tables.period_theta[q] += (real - got_q[q]) / (1.8 * max(1.0, real))
        for q, (start, real) in sorted(lates.items()):
            got_l[q] = scored(start, q)
            tables.late_theta[q] += (real - got_l[q]) / (1.8 * max(1.0, real))
        tables.period_theta[5] = tables.period_theta[4]
        if verbose:
            print(f"    round {rnd}: " + "  ".join(f"Q{q} {quarters[q][1]:.2f}/{got_q[q]:.2f}"
                                             for q in sorted(quarters))
                  + "  last 2:00 " + "  ".join(f"Q{q} {lates[q][1]:.2f}/{got_l[q]:.2f}"
                                             for q in sorted(lates)))
    return ({q: (quarters[q][1], got_q[q]) for q in quarters},
            {q: (lates[q][1], got_l[q]) for q in lates})


def fit_period_theta_states(tables, items, rounds=6, n_paths=300, seed=0, verbose=False):
    """tables.period_theta so that, from the start of each quarter of real
    games, the simulation scores in that quarter what they scored."""
    by_q = {}
    for it in items:
        by_q.setdefault(it[0], []).append(it)
    for rnd in range(rounds):
        got, real = {}, {}
        for q, its in by_q.items():
            start = sim.Start(len(its))
            for i, (_, state, theta0, _) in enumerate(its):
                _fill(start, i, start_from(state))
                start.theta[i] = theta0
            stats = {}
            sim.simulate(tables, start, n_paths, np.random.default_rng(seed), stats=stats,
                         common=False, in_play=True)
            began = sum(st.home_score + st.away_score for _, st, _, _ in its) * n_paths
            got[q] = (stats[f"points_by_q{q}"] - began) / (len(its) * n_paths)
            real[q] = float(np.mean([it[3] for it in its]))
        if verbose:
            print(f"    round {rnd}: " + "  ".join(f"Q{q} real {real[q]:.2f} sim {got[q]:.2f}"
                                             for q in sorted(real)))
        for q in real:
            tables.period_theta[q] += (real[q] - got[q]) / (1.8 * max(1.0, real[q]))
        tables.period_theta[5] = tables.period_theta[4]
    return tables.period_theta.copy(), got, real


def in_game_states(matches, grid, every=5, priors=None):
    """Every `every`-th scrimmage PLAY_OVER of each match, with the match's
    prior (prior_theta: NB2's when `priors` are given, a match missing from
    them left out; league average without) and how far the offense's lead
    really moved from there to the final whistle:
    [(GameState, theta0, offense's lead, real change in it)]."""
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        rows = resolve_sides(rows)
        if rows[0].get("final_p1") in ("", None):
            continue
        theta0 = prior_theta(grid, None if priors is None else priors[code])
        final = int(float(rows[0]["final_p1"])) - int(float(rows[0]["final_p2"]))
        for r in rows[::every]:
            state, _ = playover.state_for(r)
            if state is None or state.down is None or state.pending_conversion is not None \
                    or not 1 <= state.period <= 4:
                continue
            sign = 1 if state.offense == HOME else -1
            lead = sign * (state.home_score - state.away_score)
            out.append((state, theta0, lead, sign * final - lead))
    return out


BAND_LEAD = 7.0          # a score: the band's pull is per score of lead
# One pull for the first half, one for the third quarter and one for the
# fourth. Fitted as one pull for each half and left out of the fourth
# quarter, the second half's whole comeback was pushed into the third:
# held out, v5's third-quarter trailers came back half a point more than
# real ones, and scored 0.7-0.9 points more, while its fourth-quarter
# trailers came back 0.2-0.3 less. (A pull for each first-half quarter on
# its own came out wild off the first quarter's few real leads.)
BAND_BY_QUARTER = True
BAND_GROUPS = np.array([0, 0, 1, 2])        # quarter 1-4 -> its pull


def _band_shift(pull, cells=None):
    """Each cell's efficiency shift from the band's pull: an offense ahead by
    L plays pull * L / BAND_LEAD worse (behind, that much better), the lead
    taken at its cell's middle and capped at three scores. Three pulls
    (BAND_GROUPS): the first half, the third quarter, the fourth; two (as it
    was): one for each half, in the play-calling quarters only."""
    cells = np.arange(sim.N_CELLS) if cells is None else cells
    quarter = cells // (sim.CLOCK_CELLS * sim.LEAD_CELLS)
    mid = np.array([-14.0, -4.0, 0.0, 4.0, 14.0])[cells % sim.LEAD_CELLS]
    pull = np.asarray(pull)
    if len(pull) == 3:
        return -pull[BAND_GROUPS[quarter]] * mid / BAND_LEAD
    return -pull[quarter // 2] * mid / BAND_LEAD * sim._quarter_mask()[cells]


def _slopes(changes, leads, groups, n_groups=2):
    """By group (half, or quarter): the least-squares slope of the offense's
    lead change on its lead, leads capped at three scores."""
    out = np.full(n_groups, np.nan)
    x_all = np.clip(leads, -21, 21)
    for h in range(n_groups):
        on = groups == h
        if on.sum() < 2:
            continue
        x, y = x_all[on], changes[on]
        out[h] = ((x - x.mean()) * (y - y.mean())).sum() / max(1e-9, ((x - x.mean()) ** 2).sum())
    return out


def fit_rubber_band(tables, items, rounds=4, n_paths=200, seed=0, verbose=False):
    """The rubber band, fitted where it shows: from real in-game states,
    how fast a lead comes back by the end -- the slope of the offense's
    lead change on its lead -- simulated against real, solved for the pull
    (see _band_shift) by the secant method, on top of the snap-level fit.
    Each state plays luck of its own, the same for every trial pull.

    By quarter (BAND_BY_QUARTER, BAND_GROUPS): one pull at a time from the
    last, each on its own states with the later pulls held -- a lead's fate
    early on runs through all that follows, so the pulls solved together
    only chase each other. By half, as it was: both together.
    Returns (pull, simulated slopes, real slopes)."""
    base = tables.eff_shift.copy()
    n = 3 if BAND_BY_QUARTER else 2
    leads = np.array([lead for _, _, lead, _ in items], dtype=float)
    changes = np.array([ch for *_, ch in items], dtype=float)
    groups = np.array([BAND_GROUPS[st.period - 1] if n == 3 else (0 if st.period <= 2 else 1)
                       for st, *_ in items])
    real = _slopes(changes, leads, groups, n)
    sign = np.array([1 if st.offense == HOME else -1 for st, *_ in items], dtype=float)

    def start_for(sel):
        start = sim.Start(len(sel))
        for k, i in enumerate(sel):
            st, theta0, _, _ = items[i]
            _fill(start, k, start_from(st))
            start.theta[k] = theta0
        return start

    def simulated(pull, sel, start):
        tables.eff_shift = base + _band_shift(pull)
        home, away = sim.simulate(tables, start, n_paths, np.random.default_rng(seed), seed=seed + 1,
                                  distinct=True)
        return _slopes(sign[sel] * (home - away).mean(1) - leads[sel], leads[sel], groups[sel], n)

    def secant(pull, which, sel, start):
        """pull[which] so that the states `sel` come back as real ones do."""
        p0 = pull.copy()
        g0 = simulated(p0, sel, start)
        p1 = pull + 0.1 * which
        g1 = simulated(p1, sel, start)
        for rnd in range(rounds):
            if verbose:
                print(f"    round {rnd}: pull {p1} slopes real {real} simulated {g1}")
            moved = np.abs(p1 - p0) > 1e-6
            d = np.where(moved, (g1 - g0) / np.where(moved, p1 - p0, 1.0), -0.1)
            d = np.minimum(np.nan_to_num(d, nan=-0.1), -0.01)   # more pull, more comes back
            step = np.where(which > 0, np.clip(np.nan_to_num((real - g1) / d), -0.3, 0.3), 0.0)
            p0, g0 = p1, g1
            # negative when the snap-level fit already brings leads back too fast
            p1 = np.clip(p1 + step, -1.0, 1.0)
            g1 = simulated(p1, sel, start)
        return p1, g1

    if n == 2:
        every = np.arange(len(items))
        pull, got = secant(np.zeros(2), np.ones(2), every, start_for(every))
    else:
        pull, got = np.zeros(n), np.zeros(n)
        for q in range(n - 1, -1, -1):
            sel = np.flatnonzero(groups == q)
            if len(sel) < 50:
                continue
            which = np.zeros(n)
            which[q] = 1.0
            pull, g = secant(pull, which, sel, start_for(sel))
            got[q] = g[q]
    tables.eff_shift = base + _band_shift(pull)
    return pull, got, real



# --------------------------------------------------------------------------
# Points still to come, from real in-game states
#
# fit_period_theta sets each quarter's scoring from kickoff, with league-
# average offenses: the simulation from kickoff scores what the league
# scores in each quarter. Priced from inside a game it did not quite. From
# real states, with each match's prior as pricing gives it, v5 left too few
# points from the second quarter (the last two minutes of the half score
# more than the tables give them) and too many from the third (0.3-0.9),
# in the games it was built on and in the next week's alike, and about two
# too many from overtime. So on top of the kickoff fit each segment of the
# game (sim.SEGMENTS: the quarters, the last two minutes of each half
# apart, overtime) gets an in-play shift, tables.inplay_theta, used only
# when pricing from inside a game: from real states in a segment the
# simulation must leave, on average, the points really still to come --
# first by segment, then by the scoreline's margin within it (close games
# bleed the clock; blowouts score in garbage time). The last segments go
# first, since every earlier state's rest of the game runs through them.
# Only the total is priced off the shifted run (price_states): moneyline
# and spread keep the plain one, where the shift cost a little. The
# pre-match is left alone -- the prior grid is simulated from kickoff
# without the shift -- so a match's prior cannot chase the fit.
#
# Off by default (v5-build --in-play): the mean moved the right way in
# every held-out test, but the total's Brier gained in one week and lost
# in the next, as the pre-match level was right or not (README).

IN_PLAY_FIT = False      # fit tables.inplay_theta in the build (v5-build --in-play)
REST_EVERY = 4           # every 4th PLAY_OVER of each match
REST_PATHS = 60
REST_MAX_STATES = 4000   # per segment, spread across the matches
REST_MIN_STATES = 100    # fewer states than this in a segment: no shift
REST_MIN_CELL = 150      # ... in a segment's margin band: the segment's shift
REST_BY_BAND = False     # fit each margin band after its segment (overfits: see README)
REST_SEGMENTS = tuple(range(1, sim.N_SEGMENTS))   # the segments fitted; the rest keep 0
REST_ROUNDS = 3


def rest_of_game_states(matches, grid, priors=None, every=REST_EVERY):
    """[((segment, margin band), GameState, theta0, real points still to
    come)] from every `every`-th PLAY_OVER of each match. theta0 is the
    match's prior as pricing sets it (prior_theta: NB2's expected points,
    `priors`, match -> (home, away), when given; league average without)."""
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        rows = resolve_sides(rows)
        if rows[0].get("final_p1") in ("", None) or rows[0].get("final_p2") in ("", None):
            continue
        theta0 = prior_theta(grid, None if priors is None else priors[code])
        final = int(float(rows[0]["final_p1"])) + int(float(rows[0]["final_p2"]))
        for r in rows[::every]:
            state, _ = playover.state_for(r)
            if state is None or state.period < 1:
                continue
            cell = (sim.inplay_segment(state.period, state.clock_seconds),
                    sim.margin_band(state.home_score - state.away_score))
            out.append((cell, state, theta0, final - state.home_score - state.away_score))
    return out


def fit_rest_of_game(tables, items, n_paths=REST_PATHS, rounds=REST_ROUNDS, seed=0,
                     max_states=REST_MAX_STATES, verbose=False):
    """tables.inplay_theta so that from real states the simulation (in
    play) leaves the points really still to come: first by segment of the
    game, then each margin band within it. Returns {segment: (real,
    simulated before, simulated after)} and {(segment, band): the same}."""

    def prepare(groups, least):
        out = {}
        for key, its in groups.items():
            if len(its) < least:
                continue
            if len(its) > max_states:
                its = [its[i] for i in np.linspace(0, len(its) - 1, max_states).astype(int)]
            start = sim.Start(len(its))
            for i, (_, state, theta0, _) in enumerate(its):
                _fill(start, i, start_from(state))
                start.theta[i] = theta0
            out[key] = (start, float(np.mean(start.home + start.away)),
                        float(np.mean([it[3] for it in its])))
        return out

    def rest(prepared, key):
        # independent paths: the mean over thousands of states wants
        # thousands of games' luck, not n_paths'
        start, began, _ = prepared[key]
        home, away = sim.simulate(tables, start, n_paths,
                                  np.random.default_rng(seed + hash(key) % 1000),
                                  common=False, in_play=True)
        return float((home + away).mean()) - began

    by_seg, by_cell = {}, {}
    for it in items:
        by_seg.setdefault(it[0][0], []).append(it)
        if it[0][0] in REST_SEGMENTS:
            by_cell.setdefault(it[0], []).append(it)
    segs = prepare(by_seg, REST_MIN_STATES)
    cells = prepare(by_cell, REST_MIN_CELL)
    order = sorted(segs, reverse=True)
    real_after = {g: segs[g][2] for g in segs}

    def fit(prepared, keys, apply):
        first = {}
        for rnd in range(rounds):
            for key in keys:
                got = rest(prepared, key)
                first.setdefault(key, got)
                g = key if isinstance(key, int) else key[0]
                k = order.index(g)
                # the segment's own points, roughly: what is left from it
                # less what is left from the next; a point of scoring moves
                # with theta at about 1.8 x them
                own = got - (real_after[order[k - 1]] if k else 0.0)
                step = float(np.clip((prepared[key][2] - got) / (1.8 * max(1.0, own)), -0.3, 0.3))
                apply(key, step)
                if verbose:
                    print(f"    round {rnd} {key}: real {prepared[key][2]:.2f} simulated {got:.2f}")
        return first

    def by_segment(g, step):
        tables.inplay_theta[g, :] += step

    def by_cell_(key, step):
        tables.inplay_theta[key[0], key[1]] += step

    seg_before = fit(segs, [g for g in order if g in REST_SEGMENTS], by_segment)
    cell_keys = sorted(cells, key=lambda c: (-c[0], c[1])) if REST_BY_BAND else []
    cell_before = fit(cells, cell_keys, by_cell_)
    if 7 not in segs and 7 in REST_SEGMENTS:
        tables.inplay_theta[7] = tables.inplay_theta[5]        # overtime plays like the 4th
    return ({g: (segs[g][2], seg_before[g], rest(segs, g)) for g in sorted(segs) if g in seg_before},
            {c: (cells[c][2], cell_before[c], rest(cells, c)) for c in cell_keys})

def match_day(match_rows):
    """The match's date: its first row's file time, else the code's DDMMYY."""
    import datetime as dt
    stamp = (match_rows[0].get("file_time") or "")[:10] if match_rows else ""
    try:
        return dt.date.fromisoformat(stamp)
    except ValueError:
        pass
    code = (match_rows[0].get("match_code") or "") if match_rows else ""
    try:
        return dt.date(2000 + int(code[-2:]), int(code[-4:-2]), int(code[-6:-4]))
    except (ValueError, IndexError):
        return None



def in_game_check(tables, grid, matches, n_paths=300, seed=0, priors=None):
    """From the first PLAY_OVER of each quarter of these matches: the points
    real games scored in the rest of that quarter against what the
    simulation expects (quarter_start_states)."""
    items = quarter_start_states(matches, grid, priors)
    _, got, real = fit_period_theta_states(tables, items, rounds=1, n_paths=n_paths, seed=seed)
    return got, real


# --------------------------------------------------------------------------
# How much the day's strengths are unknown
#
# v5 played every simulated game with the two offenses fixed at the match's
# prior. The points still to come then came out too narrow: held out, real
# results fell in the bottom tenth of v5's distribution 10.5-14.6% of the
# time and in the top tenth 10-12%, and the real spread ran 3-12% wider
# than v5's (widest where the pre-match knew least). So each simulated game
# draws its offenses around the prior (sim5: tables.strength_sd), and the
# build fits how far: on the last two weeks of the games built on, each
# priced with an NB2 fitted only on what came before them -- as pricing
# sees a match in the days after a build -- the real squared misses must
# equal the simulated variance. One week swung the fit (0.19 before Sep 3,
# 0.13 before Sep 10); two or three weeks give 0.20 and 0.17.
# The fourth quarter is left out of the target: too little is left in it
# for the day's strengths to matter (its narrowness is its own).

STRENGTH_FIT = True
STRENGTH_HOLDOUT_DAYS = 14
STRENGTH_PATHS = 200
STRENGTH_MAX = 0.5


def spread_ratio(tables, items, sd, n_paths=STRENGTH_PATHS, seed=0):
    """Real squared misses of the points still to come over the simulated
    variance, from these states (rest_of_game_states' items) with strength
    sd `sd`: 1 when the simulation is as wide as real games."""
    start = sim.Start(len(items))
    for i, (_, state, theta0, _) in enumerate(items):
        _fill(start, i, start_from(state))
        start.theta[i] = theta0
    real = np.array([it[3] for it in items], dtype=float)
    home, away = sim.simulate(tables, start, n_paths, np.random.default_rng(seed), seed=seed + 1,
                              distinct=True, strength_sd=sd)
    rest = (home + away) - (start.home + start.away)[:, None]
    return float(((real - rest.mean(1)) ** 2).mean() / max(1e-9, rest.var(1).mean()))


def fit_strength_sd(tables, items, rounds=6, hi=STRENGTH_MAX):
    """tables.strength_sd so that spread_ratio is 1 (none when the fixed
    strengths are already wide enough). Returns (ratio at 0, sd, ratio)."""
    at0 = spread_ratio(tables, items, 0.0)
    if at0 <= 1.0:
        tables.strength_sd = 0.0
        return at0, 0.0, at0
    lo, top = 0.0, hi
    for _ in range(rounds):
        mid = (lo + top) / 2
        if spread_ratio(tables, items, mid) > 1.0:
            lo = mid
        else:
            top = mid
    tables.strength_sd = (lo + top) / 2
    return at0, tables.strength_sd, spread_ratio(tables, items, tables.strength_sd)


def strength_states(matches, grid, history, out_dir, days=STRENGTH_HOLDOUT_DAYS):
    """The last `days` days of these matches, each priced with an NB2 fitted
    only on the history before them: rest_of_game_states' items in the
    first three quarters."""
    import datetime as dt
    import os
    dated = {c: match_day(rows) for c, rows in matches.items()}
    if all(d is None for d in dated.values()):
        return []
    last = max(d for d in dated.values() if d is not None)
    first = last - dt.timedelta(days=days - 1)
    recent = {c: rows for c, rows in matches.items() if dated[c] is not None and dated[c] >= first}
    if not recent:
        return []
    pre = nb2_prior.Prematch.build(history, os.path.join(out_dir, "nb2_strength"),
                                   dt.datetime.combine(first, dt.time()))
    priors = pre.means([r for r in history if r["MATCH_CODE"] in recent])
    return [it for it in rest_of_game_states(recent, grid, priors) if it[1].period <= 3]


def build(matches, out_dir, grid_paths=6000, verbose=True, handles=None, history=None,
          before=None):
    """Tables, the league's efficiency by quarter (from kickoff, as v3), the
    prior grid, our own pre-match model (NB2, off `history`) and the player
    profiles from {match: export rows}; written to out_dir as v5tables.npz,
    v5grid.npz, nb2/ and v5players.json (when any handles are known).

    Nothing of GAMEPLAI's is read: every fit that needs a match's prior
    takes NB2's expected points for it, or league-average offenses when
    there is no history (a model for tests; v5-build insists on one).
    Returns (tables path, grid path)."""
    import copy
    import os
    os.makedirs(out_dir, exist_ok=True)
    pre = priors = None
    if history is not None:
        # our own pre-match model: NB2 fitted on the history before the day
        # after the last match built on (or `before`)
        import datetime as dt
        if before is None:
            days = [match_day(rows) for rows in matches.values()]
            last = max(d for d in days if d is not None)
            before = dt.datetime.combine(last + dt.timedelta(days=1), dt.time())
        pre = nb2_prior.Prematch.build(history, os.path.join(out_dir, "nb2"), before)
        priors = pre.means([r for r in history if r["MATCH_CODE"] in matches])
    tables = sim.Tables.build(matches)
    real = sim.quarter_points(matches)
    offsets, got = sim.fit_period_theta(tables, real)
    # the rubber band, from real in-game states (with a quick grid for
    # each match's prior), then the quarters again
    items = in_game_states(matches, PriorGrid.build(tables, n_paths=max(500, grid_paths // 4)),
                           priors=priors)
    if len(items) >= 200 and "band" in sim.PLAY_CALLING:
        pull, band_got, band_real = fit_rubber_band(tables, items)
        offsets, got = sim.fit_period_theta(tables, real)
        if verbose:
            names = (["first half", "Q3", "Q4"] if len(pull) == 3
                     else ["first half", "second half"])
            print("  rubber band: of every point of lead, how much comes back by the end, "
                  "real / simulated -- " + ", ".join(
                      f"{nm} {-r:.3f} / {-g:.3f}" for nm, r, g in zip(names, band_real, band_got))
                  + "; pull per score " + " / ".join(f"{x:.2f}" for x in pull))
    if verbose:
        print(f"  {tables.n_snaps:,} snaps in the tables; points by quarter real "
              + " / ".join(f"{x:.2f}" for x in real) + ", simulated from kickoff "
              + " / ".join(f"{x:.2f}" for x in got))
    if QUARTER_START_FIT:
        # each quarter's scoring again, from real quarter starts with each
        # match's prior as pricing sets it (see QUARTER_START_FIT)
        quick = PriorGrid.build(tables, n_paths=max(500, grid_paths // 4))
        before_q = tables.period_theta.copy()
        fitted_q, fitted_l = fit_quarter_levels(
            tables, quarter_start_states(matches, quick, priors),
            late_start_states(matches, quick, priors), rounds=QUARTER_ROUNDS, n_paths=QUARTER_PATHS)
        if verbose:
            print("  points in each quarter from real quarter starts, real / simulated: "
                  + "; ".join(f"Q{q} {r:.2f} / {g:.2f}" for q, (r, g) in sorted(fitted_q.items()))
                  + "; from the two-minute mark: "
                  + "; ".join(f"Q{q} {r:.2f} / {g:.2f}" for q, (r, g) in sorted(fitted_l.items()))
                  + "; scoring by quarter moved " + " / ".join(
                      f"{d:+.3f}" for d in (tables.period_theta - before_q)[1:5])
                  + ", last two minutes of each half " + " / ".join(
                      f"{tables.late_theta[q]:+.3f}" for q in (2, 4)))
    if verbose and tables.backed is not None:
        print("  backed up, per snap on the own 1 / 2 / 3 / 4 / 5: "
              + "; ".join(f"{name.replace('_', ' ')} "
                          + " / ".join(f"{100 * tables.backed[y, o]:.1f}" for y in range(1, 6))
                          for o, name in enumerate(sim.BACKED_OUTCOMES)) + " %")
    if STRENGTH_FIT and history is not None:
        # how far each simulated game's offenses are drawn around the prior
        # (see STRENGTH_FIT), before the grid, which is simulated with it
        items = strength_states(matches, PriorGrid.build(tables, n_paths=max(500, grid_paths // 4)),
                                history, out_dir)
        if len(items) >= 200:
            at0, sd, ratio = fit_strength_sd(tables, items)
            if verbose:
                print(f"  width of the points still to come, real / simulated variance (last "
                      f"{STRENGTH_HOLDOUT_DAYS} days, NB2 fitted before them): {at0:.3f} with the "
                      f"strengths fixed -> strength sd {sd:.3f}, {ratio:.3f}")
    tables_path = os.path.join(out_dir, "v5tables.npz")
    grid_path = os.path.join(out_dir, "v5grid.npz")
    grid = PriorGrid.build(tables, n_paths=grid_paths)
    grid.save(grid_path)
    if IN_PLAY_FIT:
        # the in-play shift by segment (see fit_rest_of_game), off each
        # match's prior as pricing will set it
        rest, bands = fit_rest_of_game(tables, rest_of_game_states(matches, grid, priors))
        if verbose and rest:
            print("  points still to come from real in-game states, real / simulated before -> "
                  "after the in-play shift: "
                  + "; ".join(f"{sim.SEGMENTS[g]} {r:.2f} / {b:.2f} -> {a:.2f}"
                              for g, (r, b, a) in rest.items()))
            if bands:
                print("    by margin: " + "; ".join(
                    f"{sim.SEGMENTS[g]} {sim.MARGIN_BANDS[m]} {r:.2f} / {b:.2f} -> {a:.2f}"
                    for (g, m), (r, b, a) in sorted(bands.items())))
    tables.save(tables_path)
    if verbose:
        sim_q, real_q = in_game_check(copy.deepcopy(tables), grid, matches, priors=priors)
        print("  in-game check, points in each quarter from real quarter-start states: real "
              + " / ".join(f"{real_q[q]:.2f}" for q in sorted(real_q)) + ", simulated "
              + " / ".join(f"{sim_q[q]:.2f}" for q in sorted(sim_q)))
    if pre is not None:
        if verbose:
            m = pre.meta
            ratio = f"{m['scale_raw_ratio']:.3f}" if m["scale_raw_ratio"] else "n/a"
            print(f"  pre-match: NB2 fitted on {m['fitted_on']:,} matches before {before:%Y-%m-%d};"
                  f" its totals ran {ratio} x under the last {nb2_prior.SCALE_DAYS} days'"
                  f" ({m['scale_matches']} matches) -> expected points scaled {m['scale']:.3f}")
    handles = dict(handles or {})
    for code, rows in matches.items():
        handles.setdefault(code, handles_of(rows))
    handles = {c: h for c, h in handles.items() if h}
    if handles:
        book = players.build(matches, handles)
        book.save(os.path.join(out_dir, "v5players.json"))
        if verbose:
            print(f"  player profiles: {len(book.players):,} players from {len(handles):,} matches")
    elif verbose:
        print("  player profiles: no handles in the export, none built")
    return tables_path, grid_path


def prematch_model(model_dir_or_tables):
    """The model's NB2 pre-match model, or None (built without history)."""
    import os
    d = model_dir_or_tables if os.path.isdir(model_dir_or_tables) else os.path.dirname(model_dir_or_tables)
    path = os.path.join(d, "nb2")
    return nb2_prior.Prematch(path) if nb2_prior.Prematch.exists(path) else None


def players_book(model_dir_or_tables):
    """The model's player profiles, or None."""
    import os
    d = model_dir_or_tables if os.path.isdir(model_dir_or_tables) else os.path.dirname(model_dir_or_tables)
    path = os.path.join(d, "v5players.json")
    return players.Book.load(path) if os.path.exists(path) else None


def run(path, tables_path, grid_path, variants, matches=None, n_paths=1000, workers=4,
        seed=0, book=None, handles=None, require_live=True, priors=None, history=None):
    """Grade v5 on an export. `priors`: match -> (home points, away points)
    from our own pre-match model; or `history` (AMFELO-shaped rows covering
    the graded matches) to have the model's NB2 predict them. Either way the
    prior comes only from there, never from prod's pre-match quotes."""
    if book is None:
        book = players_book(tables_path)
    by_match = playover.load(path)
    pre = prematch_model(tables_path) if priors is None else None
    if priors is None and history is not None:
        if pre is None:
            raise SystemExit("this v5 model has no NB2 pre-match model: rebuild it with --history")
        wanted = set(by_match) if matches is None else set(matches)
        priors = pre.means([r for r in history if r["MATCH_CODE"] in wanted])
    elif priors is None and pre is not None:
        raise SystemExit("this v5 model prices pre-match with NB2: pass --history (the matches'"
                         " players, teams and streams, e.g. eAMFCalibrator history's CSV)")
    codes = sorted(by_match) if matches is None else [c for c in matches if c in by_match]
    items = [(c, by_match[c]) for c in codes]
    chunks = [items[i::workers] for i in range(workers)]
    jobs = [(chunk, tables_path, grid_path, variants, n_paths, seed + i, book, handles, require_live,
             priors) for i, chunk in enumerate(chunks) if chunk]
    graded, skipped = [], Counter()
    if workers <= 1:
        results = [_grade_matches(j) for j in jobs]
    else:
        with pool_context().Pool(len(jobs)) as pool:
            results = pool.map(_grade_matches, jobs)
    for g, s in results:
        graded += g
        skipped.update(s)
    return graded, skipped
