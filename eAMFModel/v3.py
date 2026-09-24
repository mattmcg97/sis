"""v3: price PLAY_OVER snapshots with the play-by-play simulation (sim.py).

What a v3 price knows about the two players:

  prior       each offense's efficiency theta, set so that the simulation
              played from kickoff reproduces prod's pre-match spread, total
              and moneyline (`PriorGrid`);
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

from . import playover, players, sim
from .pricer import HOME

GRID = np.round(np.linspace(-0.8, 0.8, 17), 3)
MARGIN_MAX = 100
TOTAL_MAX = 160
KAPPA = 40.0          # prior precision on theta, in Fisher information (fitted: fit_kappa)


# --------------------------------------------------------------------------
# The prior: theta from prod's pre-match prices

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

    def fit(self, line52, prob52, line54, prob54, prob50=None, step=0.01):
        """(theta_home, theta_away) whose pre-match prices come closest to
        prod's, in log odds."""
        fine = np.arange(self.grid[0], self.grid[-1] + 1e-9, step)
        gi = np.interp(fine, self.grid, np.arange(len(self.grid)))
        lo = np.minimum(len(self.grid) - 2, gi.astype(int))
        w = gi - lo

        def surface(values):
            # bilinear, values: (g, g) -> (f, f)
            v = values
            a = v[lo][:, lo] * np.outer(1 - w, 1 - w) + v[lo + 1][:, lo] * np.outer(w, 1 - w) \
                + v[lo][:, lo + 1] * np.outer(1 - w, w) + v[lo + 1][:, lo + 1] * np.outer(w, w)
            return a

        logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
        err = np.zeros((len(fine), len(fine)))
        used = 0
        for market, line, prob in ((52, line52, prob52), (54, line54, prob54), (50, 0.0, prob50)):
            if prob is None or line is None:
                continue
            p = surface(market_prob(market, line, self.margin, self.total))
            err += (logit(p) - logit(prob)) ** 2
            used += 1
        if not used:
            return 0.0, 0.0                   # nothing to fit to: the league average
        i, j = np.unravel_index(np.argmin(err), err.shape)
        return float(fine[i]), float(fine[j])


def prior_lines(match_rows):
    """Prod's pre-match (spread line, P, total line, P, P(home wins)), or its
    first in-play quote on both lines when it never quoted pre-match. None
    when there is no spread or total price to fit to."""
    f = playover._float
    r = match_rows[0]
    line52, line54 = f(r.get("prematch_line_52")), f(r.get("prematch_line_54"))
    p52, p54 = f(r.get("prematch_prob_52")), f(r.get("prematch_prob_54"))
    if line52 is not None and line54 is not None and (p52 is not None or p54 is not None):
        return line52, p52, line54, p54, f(r.get("prematch_prob_50"))
    for r in match_rows:
        line52, line54 = f(r.get("line_52")), f(r.get("line_54"))
        p52, p54 = f(r.get("prob_52")), f(r.get("prob_54"))
        if line52 is not None and line54 is not None and p52 is not None and p54 is not None:
            return line52, p52, line54, p54, f(r.get("prob_50"))
    return None


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
        lines = prior_lines(rows)
        if lines is None:
            continue
        theta0 = grid.fit(*lines)
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
                kicks_second_half=side(state.opening_receiver) if state.opening_receiver else 0)


def _fill(start, i, fields):
    for k, v in fields.items():
        getattr(start, k)[i] = v


class Variant:
    """One way of running v3 (what it reacts to)."""

    def __init__(self, name, kappa=KAPPA, react=True, theta_sd=False, profiles=False,
                 pace=False, sim_kw=None, grid=None):
        """`theta_sd`: draw each path's efficiencies from the posterior (the
        prior grid in `grid` must then have been built with sd 1/sqrt(kappa))."""
        self.name, self.kappa, self.react, self.grid = name, kappa, react, grid
        self.theta_sd, self.profiles, self.pace = theta_sd, profiles, pace
        self.sim_kw = sim_kw or {}


def price_states(tables, theta0, variant, snaps, a_home, states, messages, prof, n_paths, rng):
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
    home, away = sim.simulate(tables, start, n_paths, rng,
                              theta_sd=sds if v.theta_sd else None, **v.sim_kw)
    return [_distributions(home[i], away[i]) for i in range(len(messages))]


def pool_context():
    """fork where there is one (Linux, macOS); spawn on Windows, which has no
    fork. Workers are top-level functions with picklable jobs, so either works."""
    return mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")


def _grade_matches(job):
    (matches, tables_path, grid_path, variants, n_paths, seed, book, handles, require_live) = job
    tables = sim.Tables.load(tables_path)
    grids = {None: PriorGrid.load(grid_path)}
    for v in variants:
        if v.grid and v.grid not in grids:
            grids[v.grid] = PriorGrid.load(v.grid)
    graded, skipped = [], Counter()
    rng = np.random.default_rng(seed)
    for code, match_rows in matches:
        match_rows = resolve_sides(match_rows)
        lines = prior_lines(match_rows)
        if lines is None:
            skipped["no_prior"] += 1
            continue
        theta0s = {g: grid.fit(*lines) for g, grid in grids.items()}
        a_home = match_rows[0]["team_a_side"] == "home"
        snaps = sim.snap_records(match_rows)
        pair = handles.get(code) if handles else None
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
                                      [states[m] for m in messages], messages, prof, n_paths, rng)
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


def build(matches, out_dir, grid_paths=6000, verbose=True):
    """Tables (with the league's efficiency by quarter fitted) and the prior
    grid from {match: export rows}; written to out_dir as v3tables.npz and
    v3grid.npz. Returns their paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    tables = sim.Tables.build(matches)
    real = sim.quarter_points(matches)
    offsets, got = sim.fit_period_theta(tables, real)
    if verbose:
        print(f"  {tables.n_snaps:,} snaps in the tables; points by quarter real "
              + " / ".join(f"{x:.2f}" for x in real) + ", simulated "
              + " / ".join(f"{x:.2f}" for x in got))
    tables_path = os.path.join(out_dir, "v3tables.npz")
    grid_path = os.path.join(out_dir, "v3grid.npz")
    tables.save(tables_path)
    PriorGrid.build(tables, n_paths=grid_paths).save(grid_path)
    return tables_path, grid_path


def run(path, tables_path, grid_path, variants, matches=None, n_paths=1000, workers=4,
        seed=0, book=None, handles=None, require_live=True):
    by_match = playover.load(path)
    codes = sorted(by_match) if matches is None else [c for c in matches if c in by_match]
    items = [(c, by_match[c]) for c in codes]
    chunks = [items[i::workers] for i in range(workers)]
    jobs = [(chunk, tables_path, grid_path, variants, n_paths, seed + i, book, handles, require_live)
            for i, chunk in enumerate(chunks) if chunk]
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
