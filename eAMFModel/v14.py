"""v14: prices moneyline, spread and total from inside a game by simulating the rest of it play by
play."""

import datetime as dt
import json
import math
import multiprocessing as mp
from collections import Counter, defaultdict
from dataclasses import dataclass, fields, replace as _replace

import numpy as np

from . import follow, form_layer, glmer_prior, nb2_prior, playover, players, sim14 as sim
from .state import HOME, GameState

# v14: with STRENGTH_BOOM the strengths move the points less through the draw's tilt, so a side at
# 10 points a match (the weakest prior) sits near -0.8 and one at 27 (the strongest) near +0.3:
# the grid is moved down to reach both.
GRID = np.round(np.linspace(-1.1, 0.5, 17) if sim.STRENGTH_BOOM else np.linspace(-0.8, 0.8, 17), 3)
MARGIN_MAX = 100
TOTAL_MAX = 160
KAPPA = 40.0
# The pre-match models a build can fit on --history (`prior`), each saved in its own
# subdirectory of the model; PRIOR_FILE records which one the model prices off.
PRIORS = {"nb2": nb2_prior.Prematch, "glmer": glmer_prior.Prematch}
PRIOR_FILE = "v14prior.json"


def _distributions(home, away):
    """Margin and total distributions from simulated final scores."""
    m = np.clip(home - away, -MARGIN_MAX, MARGIN_MAX) + MARGIN_MAX
    t = np.clip(home + away, 0, TOTAL_MAX)
    mp_ = np.bincount(m, minlength=2 * MARGIN_MAX + 1) / len(m)
    tp = np.bincount(t, minlength=TOTAL_MAX + 1) / len(t)
    return mp_, tp


def _graded(side, at):
    """The chance of a side once a push is taken out."""
    rest = 1 - at
    return np.where(rest > 1e-12, side / np.maximum(1e-12, rest), 0.5)


def even_line(pmf, offset):
    """The half-point line nearest to even money."""
    cdf = np.cumsum(pmf)[:-1]
    gap = np.abs(cdf - 0.5)
    gap = np.where(cdf <= 1e-12, np.inf, gap) if (cdf > 1e-12).any() else gap
    best = np.flatnonzero(gap <= gap.min() + 1e-12)
    x = np.arange(len(pmf)) - offset
    mean = float((pmf * x).sum() / max(1e-300, pmf.sum()))
    k = best[np.argmin(np.abs(best - offset + 0.5 - mean))]
    return float(k - offset) + 0.5


# Line rules other than the even line (the stream's "anchored" and "hyst" modes). Anchored: start
# at prod's line and step a point at a time toward even until P(over) is inside 50% +- ANCHOR_BAND.
# Hysteresis: keep the line the market last had until the even line is HOLD_MOVE points away or
# P(over) at the kept line leaves 50% +- HOLD_BAND. Lines are on prob_above's scale: a total, or
# the margin home must beat.
ANCHOR_BAND = 0.10
HOLD_MOVE = 2.0
HOLD_BAND = 0.15


def anchored_line(pmf, offset, start, band=ANCHOR_BAND):
    """`start` (prod's line), stepped a point at a time toward even until P(above) is inside
    50% +- band, never past the distribution's ends."""
    lo, hi = -offset, pmf.shape[-1] - 1 - offset
    x = float(start)
    for _ in range(pmf.shape[-1]):
        q = float(prob_above(pmf, offset, x))
        if q > 0.5 + band and x + 1 < hi:
            x += 1
        elif q < 0.5 - band and x - 1 > lo:
            x -= 1
        else:
            break
    return x


def held_line(pmf, offset, kept, move=HOLD_MOVE, band=HOLD_BAND):
    """The kept line while the even line is under `move` points from it and P(above) at it is
    inside 50% +- band; the even line otherwise (and with nothing kept)."""
    even = even_line(pmf, offset)
    if kept is None:
        return even
    q = float(prob_above(pmf, offset, kept))
    return kept if abs(even - kept) < move and abs(q - 0.5) <= band else even


# Key numbers: how the stream sets its own lines. Points come in lumps of 3 and 7, so the distributions are spiky: a
# line next to a spike prices far from its neighbours, and a small error in the spike moves its
# price a lot. Among half-point lines with P(over) inside 50% +- KEY_BAND, take the one with the
# least mass on the two whole numbers either side (ties: nearest 50%), and keep the market's last
# line while it is still in the band with at most KEY_HOLD more mass next to it.
KEY_BAND = 0.10
KEY_HOLD = 0.02


def key_line(pmf, offset, kept=None, band=KEY_BAND, hold=KEY_HOLD):
    """The half-point line in the gap between the spikes (see KEY_BAND); the even line when none
    has P(above) in the band."""
    pmf = np.asarray(pmf, dtype=float)
    above = 1 - np.cumsum(pmf)[:-1]                 # line k - offset + 0.5: P(above)
    mass = pmf[:-1] + pmf[1:]                       # on the two whole numbers either side
    ok = np.abs(above - 0.5) <= band + 1e-12
    if not ok.any():
        return even_line(pmf, offset)
    m = np.where(ok, mass, np.inf)
    tie = ok & (m <= m.min() + 1e-12)
    k = int(np.argmin(np.where(tie, np.abs(above - 0.5), np.inf)))
    if kept is not None and float(kept) % 1 == 0.5:
        j = int(np.floor(kept)) + offset
        if 0 <= j < len(above) and ok[j] and mass[j] <= mass[k] + hold:
            return float(kept)
    return float(k - offset) + 0.5


def prob_above(pmf, offset, line):
    """P(X > line), pushes taken out."""
    x = np.arange(pmf.shape[-1]) - offset
    above = pmf[..., x > line].sum(-1)
    at = pmf[..., x == line].sum(-1) if float(line).is_integer() else 0.0
    return _graded(above, at)


def prob_below(pmf, offset, line):
    """P(X < line), pushes taken out."""
    x = np.arange(pmf.shape[-1]) - offset
    below = pmf[..., x < line].sum(-1)
    at = pmf[..., x == line].sum(-1) if float(line).is_integer() else 0.0
    return _graded(below, at)


def market_prob(market, line, margin_pmf, total_pmf):
    """The chance of a market's selection at a line."""
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


# Builds run the long simulations (the prior grid, the points responses, the reconcile's holds) on
# BUILD_WORKERS processes, splitting the starts by row. Each row's paths draw the same random
# numbers wherever it runs, so the split changes nothing but the time.
BUILD_WORKERS = 4
_PAR = {}


def _slice_start(start, lo, hi):
    """Rows lo..hi of a Start."""
    n = len(start.period)
    sub = sim.Start(hi - lo)
    for k, v in vars(start).items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
            setattr(sub, k, v[lo:hi].copy())
    return sub


def _par_rows(bounds):
    """Worker: simulate one block of rows (and its tallies, with stats)."""
    tables, start, n_paths, seed, kw, keep = _PAR["job"]
    lo, hi = bounds
    n = len(start.period)
    kw = {k: (v[lo:hi] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for k, v in kw.items()}
    if kw.get("distinct"):
        seed = seed + int(lo)              # each block its own paths, not the first block's again
    stats = {k: ([] if isinstance(v, list) else 0) for k, v in keep.items()} if keep is not None else None
    home, away = sim.simulate(tables, _slice_start(start, lo, hi), n_paths, np.random.default_rng(seed),
                              seed=seed, stats=stats, **kw)
    if stats is not None and "_snaps" in stats:     # the audit log's paths, numbered as in one run
        stats["_snaps"] = [(x[0] + int(lo) * n_paths,) + tuple(x[1:]) for x in stats["_snaps"]]
    return home, away, stats


def par_simulate(tables, start, n_paths, seed=0, workers=None, stats=None, **kw):
    """sim.simulate (seeded) split across processes by row; the same results. With stats, every
    block's tallies are added into it (lists, such as the audit log, joined)."""
    workers = BUILD_WORKERS if workers is None else workers
    n = len(start.period)
    if workers <= 1 or n < 2 * workers:
        return sim.simulate(tables, start, n_paths, np.random.default_rng(seed), seed=seed, stats=stats, **kw)
    _PAR["job"] = (tables, start, n_paths, seed, kw, stats)
    edges = np.linspace(0, n, workers + 1).astype(int)
    try:
        with pool_context().Pool(workers) as pool:
            parts = pool.map(_par_rows, list(zip(edges[:-1], edges[1:])))
    finally:
        _PAR.clear()
    if stats is not None:
        for _, _, st in parts:
            for k, v in st.items():
                if isinstance(v, list):
                    stats.setdefault(k, []).extend(v)
                else:
                    stats[k] = stats.get(k, 0) + v
    return np.concatenate([h for h, _, _ in parts]), np.concatenate([a for _, a, _ in parts])


class PriorGrid:
    """Simulated games from kickoff on a grid of the two offenses' strengths."""

    def __init__(self, margin, total, grid=GRID, pace_home=None, pace_away=None, big_home=None,
                 big_away=None, loss_home=None, loss_away=None):
        """Hold the margin and total distributions for each grid point, and the pace and big-play
        responses (each side's points at each pair of paces or big-play rates, against both at 1)."""
        self.margin, self.total, self.grid = margin, total, grid
        self.pace_home, self.pace_away = pace_home, pace_away
        self.big_home, self.big_away = big_home, big_away
        self.loss_home, self.loss_away = loss_home, loss_away

    def loss_response(self, home_loss, away_loss):
        """How far the two sides' lost-yardage rates move each side's points from kickoff (v14)."""
        if self.loss_home is None:
            return 1.0, 1.0
        return _bilinear((self.loss_home, self.loss_away), np.log(BIG_POINTS), np.log(home_loss),
                         np.log(away_loss))

    def big_response(self, home_big, away_big):
        """How far the two sides' big-play rates move each side's points from kickoff: (home, away)
        multipliers, 1 when the grid has no big-play response."""
        if self.big_home is None:
            return 1.0, 1.0
        return _bilinear((self.big_home, self.big_away), np.log(BIG_POINTS), np.log(home_big),
                         np.log(away_big))

    def pace_response(self, home_pace, away_pace):
        """How far the two players' pace moves each side's points from kickoff: (home, away)
        multipliers, 1 when the grid has no pace response."""
        if self.pace_home is None:
            return 1.0, 1.0
        gi = np.interp([home_pace, away_pace], PACE_POINTS, np.arange(len(PACE_POINTS)))
        lo = np.minimum(len(PACE_POINTS) - 2, gi.astype(int))
        w = gi - lo
        out = []
        for table in (self.pace_home, self.pace_away):
            a, b = lo
            wa, wb = w
            out.append(float(table[a, b] * (1 - wa) * (1 - wb) + table[a + 1, b] * wa * (1 - wb)
                             + table[a, b + 1] * (1 - wa) * wb + table[a + 1, b + 1] * wa * wb))
        return tuple(out)

    @classmethod
    def build(cls, tables, n_paths=6000, seed=0, grid=GRID, theta_sd=0.0, **sim_kw):
        """Simulate from kickoff at every grid point."""
        g = len(grid)
        start = sim.Start(g * g)
        th, ta = np.meshgrid(grid, grid, indexing="ij")
        start.theta = np.stack([th.ravel(), ta.ravel()], axis=1)
        rng = np.random.default_rng(seed)
        start.team = rng.integers(0, 2, g * g).astype(np.int8)
        start.kicks_second_half = 1 - start.team
        sds = np.full((g * g, 2), theta_sd) if theta_sd else None
        home, away = par_simulate(tables, start, n_paths, seed=int(rng.integers(0, 2 ** 62)), theta_sd=sds, **sim_kw)
        margin = np.zeros((g, g, 2 * MARGIN_MAX + 1))
        total = np.zeros((g, g, TOTAL_MAX + 1))
        for i in range(g * g):
            a, b = divmod(i, g)
            margin[a, b], total[a, b] = _distributions(home[i], away[i])
        return cls(margin, total, grid)

    def save(self, path):
        """Write the grid to a file."""
        extra = ({} if self.pace_home is None else
                 dict(pace_home=self.pace_home, pace_away=self.pace_away))
        if self.big_home is not None:
            extra.update(big_home=self.big_home, big_away=self.big_away)
        if self.loss_home is not None:
            extra.update(loss_home=self.loss_home, loss_away=self.loss_away)
        np.savez_compressed(path, margin=self.margin, total=self.total, grid=self.grid, **extra)

    @classmethod
    def load(cls, path):
        """Read a grid from a file."""
        z = np.load(path)
        pace = (z["pace_home"], z["pace_away"]) if "pace_home" in z else (None, None)
        big = (z["big_home"], z["big_away"]) if "big_home" in z else (None, None)
        loss = (z["loss_home"], z["loss_away"]) if "loss_home" in z else (None, None)
        return cls(z["margin"], z["total"], z["grid"], *pace, *big, *loss)


def _bilinear(tables, points, a, b):
    """Each table (shaped len(points) x len(points)) read at (a, b), clamped to the points."""
    gi = np.interp([a, b], points, np.arange(len(points)))
    lo = np.minimum(len(points) - 2, gi.astype(int))
    wa, wb = gi - lo
    i, j = lo
    return tuple(float(t[i, j] * (1 - wa) * (1 - wb) + t[i + 1, j] * wa * (1 - wb)
                       + t[i, j + 1] * (1 - wa) * wb + t[i + 1, j + 1] * wa * wb) for t in tables)


# v10: the pre-match model's expected points already carry each player's pace -- a fast player
# scores more, and NB2 learned that from his scores -- so the sim's clock must not add it a second
# time. Each side's points at kickoff are simulated at pairs of paces (PACE_POINTS, league
# strengths) and a match's expected points are divided by its pace response before the starting
# strengths are fitted: the sim, with both paces applied, then averages the prior's points.
PACE_NEUTRAL = True
PACE_POINTS = np.array([0.85, 0.925, 1.0, 1.075, 1.15])
PACE_PATHS = 4000


def fit_pace_response(tables, n_paths=PACE_PATHS, seed=0):
    """Each side's mean points from kickoff at every pair of paces (home x away), over both at
    1: (home, away) tables shaped (len(PACE_POINTS), len(PACE_POINTS))."""
    n = len(PACE_POINTS)
    start = sim.Start(2 * n * n)
    rows = [(i, j, t) for i in range(n) for j in range(n) for t in (0, 1)]
    for k, (i, j, t) in enumerate(rows):
        start.pace[k] = (PACE_POINTS[i], PACE_POINTS[j])
        start.team[k] = t
        start.kicks_second_half[k] = 1 - t
    home, away = par_simulate(tables, start, n_paths, seed=seed)
    h = home.mean(axis=1).reshape(n, n, 2).mean(-1)
    a = away.mean(axis=1).reshape(n, n, 2).mean(-1)
    mid = n // 2
    return h / h[mid, mid], a / a[mid, mid]


# v13: a side's big-play rate (sim13.BIG_PLAYS) moves its points too -- more long plays score
# more at the same strength -- and the pre-match model's expected points already carry it (a
# big-play gamer's results are in its fit). As with pace, each side's points from kick-off are
# simulated at pairs of rates (BIG_POINTS, league strengths) and a match's expected points are
# divided by its response before the strengths are fitted: a big-play side then reaches the same
# points as the prior has it, from fewer, longer drives.
BIG_POINTS = np.exp(np.linspace(-1.0, 1.0, 5))
BIG_PATHS = 4000


def fit_big_response(tables, n_paths=BIG_PATHS, seed=0, attr="big"):
    """Each side's mean points from kickoff at every pair of big-play rates (home x away), over
    both at 1: (home, away) tables shaped (len(BIG_POINTS), len(BIG_POINTS)). With attr "loss",
    the same for the lost-yardage rates (v14)."""
    n = len(BIG_POINTS)
    start = sim.Start(2 * n * n)
    rows = [(i, j, t) for i in range(n) for j in range(n) for t in (0, 1)]
    for k, (i, j, t) in enumerate(rows):
        getattr(start, attr)[k] = (BIG_POINTS[i], BIG_POINTS[j])
        start.team[k] = t
        start.kicks_second_half[k] = 1 - t
    home, away = par_simulate(tables, start, n_paths, seed=seed)
    h = home.mean(axis=1).reshape(n, n, 2).mean(-1)
    a = away.mean(axis=1).reshape(n, n, 2).mean(-1)
    mid = n // 2
    return h / h[mid, mid], a / a[mid, mid]


def fit_big_elasticity(tables, grid, n_paths=BIG_PATHS // 2, seed=0, attr="big"):
    """How far the simulated big plays move with the rate once the strengths hold the points: the
    slope of log(big plays a snap) on log(rate), both sides at each of BIG_POINTS from league
    strengths. A rating (log odds of a big play) x is then played at rate exp(x / slope)."""
    x_t = np.arange(grid.total.shape[-1])
    mid = len(grid.grid) // 2
    league = float((grid.total[mid, mid] * x_t).sum()) / 2
    logs = []
    for b in BIG_POINTS:
        prof = players.Profile(**{attr: float(b)})
        st = sim.Start(2)
        st.team[:], st.kicks_second_half[:] = (0, 1), (1, 0)
        getattr(st, attr)[:] = b
        st.theta[:] = prior_theta(grid, (league, league), (prof, prof), pace=False)
        stats = {}
        sim.simulate(tables, st, n_paths, np.random.default_rng(seed), seed=seed, stats=stats)
        logs.append(np.log(max(1, stats.get(attr, 0)) / max(1, stats.get("snaps", 1))))
    lb = np.log(BIG_POINTS)
    slope = float(np.polyfit(lb, np.array(logs), 1)[0])
    return min(1.5, max(BIG_ELASTICITY_MIN, slope))


# With sim.STRENGTH_BOOM a side's strength works on the same big and failed plays as its big-play
# rate, so the strengths that hold its points take most of the rate back (the slope fell from 0.52
# to the old floor, 0.2): every rating was played at its odds to the fifth power, and a match of
# two big-play gamers (both at BIG_POINTS' top, e) was priced 4 points over its prior. The slope is
# held at 0.5 or more, about what it was: the ratings move the big plays about as far as before.
BIG_ELASTICITY_MIN = 0.5 if sim.STRENGTH_BOOM else 0.2


def _surface_fn(grid, step):
    """Interpolate a grid of values onto a finer grid."""
    fine = np.arange(grid[0], grid[-1] + 1e-9, step)
    gi = np.interp(fine, grid, np.arange(len(grid)))
    lo = np.minimum(len(grid) - 2, gi.astype(int))
    w = gi - lo

    def surface(v):
        """The values on the finer grid."""
        return (v[lo][:, lo] * np.outer(1 - w, 1 - w) + v[lo + 1][:, lo] * np.outer(w, 1 - w)
                + v[lo][:, lo + 1] * np.outer(1 - w, w) + v[lo + 1][:, lo + 1] * np.outer(w, w))
    return fine, surface


def fit_means(grid, home_points, away_points, step=0.01):
    """The two offenses' strengths whose simulated games average these expected points."""
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


def prior_theta(grid, means, prof=None, pace=True):
    """A match's starting strengths: from the pre-match prior's expected points, else league
    average. With the players' profiles (`prof`, home and away) and PACE_NEUTRAL, the points are
    first divided by the pace response, so the sim's clock does not count pace twice."""
    if means is None:
        return LEAGUE_THETA
    if PACE_NEUTRAL and pace and prof is not None:
        rh, ra = grid.pace_response(prof[0].pace, prof[1].pace)
        means = (means[0] / rh, means[1] / ra)
    if prof is not None:
        bh, ba = grid.big_response(prof[0].big, prof[1].big)
        lh, la = grid.loss_response(prof[0].loss, prof[1].loss)
        means = (means[0] / (bh * lh), means[1] / (ba * la))
    return fit_means(grid, *means)


# v14: hold the prior's points with every side's own rates at once. The big-play and lost-yardage
# responses are each read with the other at league, and loss size and kick-off styles have none:
# with a match's own profiles the sim's kick-off ran 0.59 points a match above the prior (league
# profiles: 0.12). So the match's kick-off is simulated at the strengths its prior gives at league
# rates, once with both sides' own rates and once at league rates on the same random numbers
# (HOLD_PATHS games each), and the prior's points are divided by each side's ratio of the two
# before the strengths are fitted. A first version stepped the strengths on the own-rate sim alone;
# its noise moved every match's expected total by about 0.6 points.
HOLD_POINTS = True
HOLD_PATHS = 2000
HOLD_RATES = ("big", "loss", "loss_size", "kick_tb", "kick_nlz")
# Held out, holding each side's points to the prior threw away what the rates say about the margin
# (with no hold -- prior_theta's responses alone -- the margin's RPS was 0.012 better than v13's,
# held 0.013 worse) while no hold left the total 0.86 points high. Scaling both sides' points alike
# from league-rate strengths let the rates' whole effect onto the margin (0.080 worse). HOLD_TOTAL
# starts where no hold does and scales both sides' prior points alike until the match's own
# kick-off sim gives the prior's total.
HOLD_TOTAL = True
# With sim.STRENGTH_BOOM a side's points climb faster than exponentially in its strength, and the
# day's strength spread (strength_draw) lifts the mean more than its lognormal allowance: the held
# kick-off ran 0.80 points a match above the prior. HOLD_AS_PRICED plays the hold sim as pricing
# plays the match -- with the day's spread and, where the in-play shift is fitted, on the total's
# in-play run.
HOLD_AS_PRICED = True
# The hold's one step (scale the prior's points by prior over simulated, refit the strengths) assumes
# the match's points move with the strengths as the grid's do; where its sides' own rates make them
# move faster it overshoots. Up to HOLD_STEPS sims: after the first step the held kick-off is
# simulated again and, until it lands within HOLD_TOL points of the prior's total, the scale is
# moved along the line through the last two (log scale, log points).
HOLD_STEPS = 3
HOLD_TOL = 0.25


def _hold_step(lk, lg, target):
    """The next log scale: along the line through the last two (log scale, log points) to the
    target's log points, or a plain ratio step where the two give no slope."""
    slope = (lg[-1] - lg[-2]) / (lk[-1] - lk[-2]) if abs(lk[-1] - lk[-2]) > 1e-6 else 0.0
    slope = slope if slope > 0.2 else 1.0
    return lk[-1] + (target - lg[-1]) / slope


def _hold_sim(tables, theta, prof, variant, n_paths, seed):
    """Each side's mean points from the match's kick-off (either side kicking) at strengths theta."""
    st = sim.Start(2)
    st.team[:], st.kicks_second_half[:] = (0, 1), (1, 0)
    st.theta[:] = theta
    if HOLD_AS_PRICED:
        for i in range(2):
            st.theta[i], st.strength[i], st.strength_game[i] = sim.strength_draw(tables, theta, _forms(tables, prof))
    st.aggression[:] = (prof[0].aggression, prof[1].aggression)
    st.kick[:] = (prof[0].kick, prof[1].kick)
    if variant.pace:
        st.pace[:] = (prof[0].pace, prof[1].pace)
    for name in HOLD_RATES:
        getattr(st, name)[:] = (getattr(prof[0], name), getattr(prof[1], name))
    home, away = sim.simulate(tables, st, n_paths, np.random.default_rng(seed), seed=seed, in_play=_priced_in_play(tables))
    return max(0.5, float(home.mean())), max(0.5, float(away.mean()))


def _forms(tables, prof):
    """Each side's form on the day as pricing reads it."""
    return [tables.strength_league if p.form is None else p.form for p in prof]


def _priced_in_play(tables):
    """Whether a hold sim plays the in-play shift, as pricing's total does (HOLD_AS_PRICED)."""
    return HOLD_AS_PRICED and bool(np.any(tables.inplay_theta))


def held_theta(tables, grid, means, prof, variant, seed=0, n_paths=None):
    """The match's starting strengths with its own rates holding the prior's points (HOLD_POINTS),
    else prior_theta."""
    if not HOLD_POINTS or means is None or prof is None or not variant.profiles:
        return prior_theta(grid, means, prof if variant.profiles else None, variant.pace)
    n = n_paths or HOLD_PATHS
    if HOLD_TOTAL:
        target = means[0] + means[1]
        theta = prior_theta(grid, means, prof, variant.pace)
        got = sum(_hold_sim(tables, theta, prof, variant, 2 * n, seed))
        k = target / got
        theta = prior_theta(grid, (means[0] * k, means[1] * k), prof, variant.pace)
        lk, lg = [0.0], [math.log(got)]
        for _ in range(HOLD_STEPS - 1):
            got = sum(_hold_sim(tables, theta, prof, variant, 2 * n, seed))
            if abs(got - target) <= HOLD_TOL:
                break
            lk.append(math.log(k)); lg.append(math.log(got))
            k = math.exp(_hold_step(lk, lg, math.log(target)))
            theta = prior_theta(grid, (means[0] * k, means[1] * k), prof, variant.pace)
        return theta
    league = tuple(_replace(p, **{name: 1.0 for name in HOLD_RATES}) for p in prof)
    theta = prior_theta(grid, means, league, variant.pace)
    own = _hold_sim(tables, theta, prof, variant, n, seed)
    base = _hold_sim(tables, theta, league, variant, n, seed)
    return prior_theta(grid, (means[0] * base[0] / own[0], means[1] * base[1] / own[1]), league, variant.pace)


# v14: the build's calibration fits play every state at league rates and the prior's strengths,
# while pricing plays each match with its sides' own rates at the held strengths. Once the profiles
# exist, the build's last step (RECONCILE) re-fits each quarter's scoring level and its last two
# minutes' from real quarter starts (fit_quarter_levels), and with RECONCILE_REST the in-play shift
# by part of the game and margin (fit_rest_of_game), each state played as pricing plays it: Held.
RECONCILE = True
RECONCILE_REST = False


@dataclass
class Held:
    """A match's held strengths with its two sides' profiles, as a calibration state's theta0."""
    theta: tuple
    prof: tuple


def _set_side(tables, start, i, theta0):
    """A calibration start's strengths: theta0 as given, or (Held) as pricing sets them, with the
    day's strength spread and both sides' profiles."""
    if not isinstance(theta0, Held):
        start.theta[i] = theta0
        return
    prof = theta0.prof
    start.theta[i], start.strength[i], start.strength_game[i] = sim.strength_draw(
        tables, theta0.theta, [tables.strength_league if p.form is None else p.form for p in prof])
    start.aggression[i] = (prof[0].aggression, prof[1].aggression)
    start.kick[i] = (prof[0].kick, prof[1].kick)
    start.pace[i] = (prof[0].pace, prof[1].pace)
    for name in HOLD_RATES:
        getattr(start, name)[i] = (getattr(prof[0], name), getattr(prof[1], name))


RECONCILE_HOLD_PATHS = 200


def held_starts(tables, grid, matches, priors, book, ratings, handles, teams, variant=None,
                n_paths=RECONCILE_HOLD_PATHS):
    """{match code: Held} for every match with a prior, its profiles as pricing builds them. With
    HOLD_TOTAL every match's kick-off is simulated in one run (n_paths games each, on paths of its
    own; the reconcile only needs the matches together right, so far fewer than pricing's)."""
    v = variant or Variant("v14")
    found = []
    for code, rows in matches.items():
        means = (priors or {}).get(code)
        if means is None:
            continue
        pair = (handles or {}).get(code) or handles_of(resolve_sides(rows))
        prof = ((book.profile(pair[0]), book.profile(pair[1])) if (book and pair)
                else (players.Profile(), players.Profile()))
        found.append((code, means, with_big(prof, ratings, pair, *side_teams(pair, (teams or {}).get(code)))))
    if not (HOLD_POINTS and HOLD_TOTAL and v.profiles) or not found:
        return {c: Held(tuple(held_theta(tables, grid, m, p, v, seed=match_seed(c))), p) for c, m, p in found}
    theta0 = [prior_theta(grid, m, p, v.pace) for _, m, p in found]
    st = sim.Start(2 * len(found))
    for k, ((_, _, prof), th) in enumerate(zip(found, theta0)):
        for i, first in ((2 * k, 0), (2 * k + 1, 1)):
            st.team[i], st.kicks_second_half[i] = first, 1 - first
            st.theta[i] = th
            if HOLD_AS_PRICED:
                st.theta[i], st.strength[i], st.strength_game[i] = sim.strength_draw(tables, th, _forms(tables, prof))
            st.aggression[i] = (prof[0].aggression, prof[1].aggression)
            st.kick[i] = (prof[0].kick, prof[1].kick)
            if v.pace:
                st.pace[i] = (prof[0].pace, prof[1].pace)
            for name in HOLD_RATES:
                getattr(st, name)[i] = (getattr(prof[0], name), getattr(prof[1], name))
    target = np.array([m[0] + m[1] for _, m, _ in found])
    lk, lg = [np.zeros(len(found))], []
    for step in range(HOLD_STEPS if HOLD_AS_PRICED else 1):
        home, away = par_simulate(tables, st, n_paths, seed=0, distinct=True, in_play=_priced_in_play(tables))
        lg.append(np.log(np.maximum(1.0, (home + away).mean(axis=1).reshape(-1, 2).mean(axis=1))))
        if step == 0:
            nk = np.log(target) - lg[0]
        else:
            nk = np.array([_hold_step([a, b], [c, d], t) for a, b, c, d, t in
                           zip(lk[-2], lk[-1], lg[-2], lg[-1], np.log(target))])
        lk.append(nk)
        theta0 = [prior_theta(grid, (m[0] * math.exp(x), m[1] * math.exp(x)), p, v.pace)
                  for (_, m, p), x in zip(found, nk)]
        if step + 1 < (HOLD_STEPS if HOLD_AS_PRICED else 1):
            for k, ((_, _, prof), th) in enumerate(zip(found, theta0)):
                for i in (2 * k, 2 * k + 1):
                    st.theta[i], st.strength[i], st.strength_game[i] = sim.strength_draw(tables, th, _forms(tables, prof))
    return {code: Held(tuple(th), prof) for (code, _, prof), th in zip(found, theta0)}


def reconcile(tables, grid, matches, priors, book, ratings, handles, teams, grid_paths, verbose=True):
    """The build's last step (RECONCILE): the scoring levels (and with RECONCILE_REST the in-play
    shift) re-fitted with every match played as pricing plays it, then the grid rebuilt on them
    (its responses kept). Returns the new grid."""
    held = held_starts(tables, grid, matches, priors, book, ratings, handles, teams)
    if sim.QUARTER_PACE:
        # the pace was fitted at league strengths before the levels; with the clock stopping after
        # failed plays (sim.STOP_LINK) it hangs on the play, so it is fitted again as pricing plays
        paced = fit_quarter_pace(tables, *pace_states(matches), held=held)
        if verbose and paced:
            real_n, before_n, after_n = paced[3:]
            print("  reconcile, snaps a quarter from the real quarter starts at the matches' own strengths, real / "
                  "simulated before -> after: " + "; ".join(f"Q{q + 1} {real_n[q]:.2f} / {before_n[q]:.2f} -> {after_n[q]:.2f}"
                                                         for q in range(4)))
        if CALL_FIT and sim.TIMEOUTS and getattr(tables, "call_fitted", False):
            scaled = fit_call_scale(tables, call_states(matches))
            if verbose and scaled:
                print("  reconcile, timeouts: " + "; ".join(f"Q{q} {r:.2f} / {b:.2f} -> {a:.2f} (rates x{s:.2f})"
                                                         for q, (n, r, b, a, s) in sorted(scaled.items())))
        if sim.MANAGE_FIT:
            two_minute = manage_states(matches)
            counted = fit_late_count(tables, two_minute)
            if verbose and counted:
                print("  reconcile, snaps from the two-minute mark: " + "; ".join(
                    f"Q{q} {r:.2f} / {b:.2f} -> {a:.2f} (x{k:.2f})" for q, (n, r, b, a, k) in sorted(counted.items())))
            managed = fit_manage(tables, two_minute)
            if verbose and managed:
                print("  reconcile, end-of-half field goals on downs 1-3: " + "; ".join(
                    f"Q{q} {r:.3f} / {b:.3f} -> {a:.3f} (clock held {p:.2f})" for q, (n, r, b, a, p) in sorted(managed.items())))
    before_q, before_l = tables.period_theta.copy(), tables.late_theta.copy()
    fitted_q, fitted_l = fit_quarter_levels(
        tables, quarter_start_states(matches, grid, priors, held=held),
        late_start_states(matches, grid, priors, held=held), rounds=QUARTER_ROUNDS, n_paths=QUARTER_PATHS)
    if verbose:
        print(f"  reconcile, {len(held):,} matches at their own rates and held strengths -- points in each"
              " quarter from real quarter starts, real / simulated: "
              + "; ".join(f"Q{q} {r:.2f} / {g:.2f}" for q, (r, g) in sorted(fitted_q.items()))
              + "; from the two-minute mark: "
              + "; ".join(f"Q{q} {r:.2f} / {g:.2f}" for q, (r, g) in sorted(fitted_l.items()))
              + "; scoring by quarter moved " + " / ".join(
                  f"{d:+.3f}" for d in (tables.period_theta - before_q)[1:5])
              + ", last two minutes " + " / ".join(
                  f"{tables.late_theta[q] - before_l[q]:+.3f}" for q in (2, 4)))
    if RECONCILE_REST:
        rest, bands = fit_rest_of_game(tables, rest_of_game_states(matches, grid, priors, held=held))
        if verbose and rest:
            print("  reconcile, points still to come from real in-game states, real / simulated before"
                  " -> after: " + "; ".join(f"{sim.SEGMENTS[g]} {r:.2f} / {b:.2f} -> {a:.2f}"
                                           for g, (r, b, a) in rest.items()))
            if bands:
                print("    by margin: " + "; ".join(
                    f"{sim.SEGMENTS[g]} {sim.MARGIN_BANDS[m]} {r:.2f} / {b:.2f} -> {a:.2f}"
                    for (g, m), (r, b, a) in sorted(bands.items())))
    new = PriorGrid.build(tables, n_paths=grid_paths)
    for name in ("pace_home", "pace_away", "big_home", "big_away", "loss_home", "loss_away"):
        setattr(new, name, getattr(grid, name))
    return new


# v10: the pre-match model's expected points spread wider than real games do. Refitted at the
# start of each of the SHRINK_WINDOWS stretches of SHRINK_DAYS before the build's cut-off and read
# on them out of sample, as the live model prices (NB2's ratings as fitted, glmer's form following
# the results), real totals and margins move total_slope and margin_slope points for each point of
# its predictions, pooled within the stretches. Each slope is moved toward 1 by its standard
# error -- only as far as the data are sure of -- and kept within SHRINK_RANGE, and every match's
# expected total and margin are pulled toward the build's average by them (MARGIN_SHRINK: the
# margin's too, by MARGIN_WEIGHT of its pull: big predicted margins come back further than small
# ones, so the mean margin's best slope, about 0.77, is stronger than the moneyline's, about 0.87;
# half keeps most of the spread's gain and leaves the moneyline as it was). The slopes are read on
# established gamers' matches only (SHRINK_MIN_EXPERIENCE
# earlier matches each): a newcomer's first matches are priced in another regime -- NB2 can't price
# them, glmer extrapolates their learning curve -- and a cohort of them swung glmer's margin slope
# from 0.78 to 0.27 in one fortnight. One fortnight alone reads a margin slope to about +-0.08;
# four, pooled, to about +-0.04.
PRIOR_SHRINK = True
MARGIN_SHRINK = True
MARGIN_WEIGHT = 0.5
SHRINK_DAYS = 14
SHRINK_WINDOWS = 4
SHRINK_WORKERS = 2               # refits run side by side (each its own process)
SHRINK_MIN_EXPERIENCE = 30
SHRINK_MIN_MATCHES = 200
SHRINK_RANGE = (0.5, 1.2)
SHRINK_FILE = "v14shrink.json"


def shrink_means(means, shrink):
    """(home, away) expected points with the total and margin pulled toward the build's
    average."""
    h, a = means
    t = shrink["total_centre"] + shrink["total_slope"] * (h + a - shrink["total_centre"])
    m = shrink["margin_centre"] + shrink["margin_slope"] * (h - a - shrink["margin_centre"])
    return (t + m) / 2, (t - m) / 2


def pooled_slope(windows):
    """Least-squares slope of y on x pooled within windows (each centred on its own means), and
    its standard error: windows is [(x, y), ...]. (1, inf) when x never varies."""
    parts = [(np.asarray(x, float), np.asarray(y, float)) for x, y in windows if len(x) >= 2]
    parts = [(x - x.mean(), y - y.mean()) for x, y in parts]
    sxx = sum(float(x @ x) for x, _ in parts)
    if sxx <= 0:
        return 1.0, math.inf
    b = sum(float(x @ y) for x, y in parts) / sxx
    rss = sum(float(((y - b * x) ** 2).sum()) for x, y in parts)
    dof = max(1, sum(len(x) for x, _ in parts) - len(parts) - 1)
    return b, math.sqrt(rss / dof / sxx)


def toward_one(slope, se):
    """A slope moved toward 1 by its standard error, never past it."""
    return min(1.0, slope + se) if slope < 1 else max(1.0, slope - se)


def experience(history):
    """match -> the fewer settled matches its two gamers had played before it."""
    seen, out = Counter(), {}
    for r in sorted(history, key=nb2_prior._start):
        gamers = [str(r.get(k) or "").strip().upper() for k in ("PLAYER_1_HANDLE", "PLAYER_2_HANDLE")]
        out[r["MATCH_CODE"]] = min(seen[g] for g in gamers)
        for g in gamers:
            seen[g] += 1
    return out


def fit_shrink(history, prior, before, centre, work_dir, scale=1.0, days=SHRINK_DAYS,
               windows=SHRINK_WINDOWS):
    """Refit the pre-match model at the start of each of `windows` stretches of `days` before the
    cut-off, predict each stretch's matches between established gamers (SHRINK_MIN_EXPERIENCE)
    and regress the real total and margin on the predicted ones, pooled within the stretches: the
    slopes (corrected by the live model's level `scale`, moved toward 1 by their standard errors,
    the margin's applied by MARGIN_WEIGHT, and kept within SHRINK_RANGE), the build's average
    (`centre`: total, margin), the matches read and each stretch's own slopes. None when fewer
    than SHRINK_MIN_MATCHES were read."""
    import os
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    module = nb2_prior if prior == "nb2" else glmer_prior
    finished = [r for r in history if r.get("PLAYER_1_FINAL_SCORE") not in ("", None)
                and r.get("PLAYER_2_FINAL_SCORE") not in ("", None)
                and nb2_prior._start(r) is not None]
    played = experience(finished)
    stretches = []
    for k in range(1, windows + 1):
        start = before - dt.timedelta(days=k * days)
        end = start + dt.timedelta(days=days)
        held = [r for r in finished if start <= nb2_prior._start(r) < end
                and played[r["MATCH_CODE"]] >= SHRINK_MIN_EXPERIENCE]
        if held:
            stretches.append((start, held))
    if sum(len(held) for _, held in stretches) < SHRINK_MIN_MATCHES:
        return None

    def read(k):
        start, held = stretches[k]
        d = os.path.join(work_dir, "shrink", str(k + 1))
        module.fit(history, d, before=start)
        pred = module.predict(d, held)
        shutil.rmtree(d, ignore_errors=True)               # read once: no need to keep the refit
        return [(pred[r["MATCH_CODE"]], float(r["PLAYER_1_FINAL_SCORE"]),
                 float(r["PLAYER_2_FINAL_SCORE"])) for r in held if r["MATCH_CODE"] in pred]

    with ThreadPoolExecutor(max(1, min(SHRINK_WORKERS, len(stretches)))) as pool:
        got = list(pool.map(read, range(len(stretches))))
    if sum(len(rows) for rows in got) < SHRINK_MIN_MATCHES:
        return None
    totals = [([p[0] + p[1] for p, _, _ in rows], [h + a for _, h, a in rows]) for rows in got]
    margins = [([p[0] - p[1] for p, _, _ in rows], [h - a for _, h, a in rows]) for rows in got]
    raw_t, se_t = (v / scale for v in pooled_slope(totals))
    raw_m, se_m = (v / scale for v in pooled_slope(margins))
    lo, hi = SHRINK_RANGE
    margin = (float(np.clip(1 - MARGIN_WEIGHT * (1 - toward_one(raw_m, se_m)), lo, hi))
              if MARGIN_SHRINK else 1.0)
    each = []
    for (start, _), rows, t, m in zip(stretches, got, totals, margins):
        each.append({"from": start.isoformat(), "matches": len(rows),
                     "total_raw": pooled_slope([t])[0] / scale,
                     "margin_raw": pooled_slope([m])[0] / scale})
    return {"total_slope": float(np.clip(toward_one(raw_t, se_t), lo, hi)), "margin_slope": margin,
            "total_raw": raw_t, "margin_raw": raw_m, "total_se": se_t, "margin_se": se_m,
            "total_centre": float(centre[0]), "margin_centre": float(centre[1]),
            "matches": sum(len(rows) for rows in got), "days": days, "windows": len(stretches),
            "from": stretches[-1][0].isoformat(), "before": before.isoformat(), "each": each}


class ShrunkPrematch:
    """A pre-match model whose expected points are pulled toward the build's average
    (shrink_means); everything else is the model's own."""

    def __init__(self, pre, shrink):
        """Wrap a fitted pre-match model and the build's shrink."""
        self.pre, self.shrink = pre, shrink

    def __getattr__(self, name):
        return getattr(self.pre, name)

    def means(self, schedule, **kw):
        """The model's expected points for each match, shrunk."""
        return {c: shrink_means(m, self.shrink) for c, m in self.pre.means(schedule, **kw).items()}


def resolve_sides(match_rows):
    """The rows with TEAM_A's side (home or away) filled in."""
    if match_rows and all(r.get("team_a_side") in ("home", "away") for r in match_rows):
        return match_rows
    return [r if r.get("team_a_side") in ("home", "away") else dict(r, team_a_side="home")
            for r in match_rows]


class Efficiency:
    """Each offense's strength updated from its first-down success in the game so far."""

    def __init__(self, tables, theta0, kappa=KAPPA):
        """Start both offenses at their prior strengths."""
        self.tables = tables
        self.theta0 = np.array(theta0, dtype=float)
        self.kappa = kappa
        self.score = np.zeros(2)
        self.info = np.zeros(2)

    def expected(self, side, key, period=None):
        """The chance of a first down at the prior strength."""
        f = min(0.995, max(0.005, self.tables.success_of(key)))
        offset = self.tables.period_theta[min(int(period), 5)] if period else 0.0
        return f ** math.exp(-(self.theta0[side] + offset))

    def add(self, side, key, success, period=None):
        """Count one snap's result."""
        p = min(0.995, max(0.005, self.expected(side, key, period)))
        lp = math.log(p)
        self.score[side] += -lp * (success - p) / (1 - p)
        self.info[side] += lp * lp * p / (1 - p)

    def theta(self):
        """Both offenses' updated strengths."""
        return self.theta0 + self.score / (self.info + self.kappa)

    def sd(self):
        """How uncertain each updated strength still is."""
        return 1.0 / np.sqrt(self.info + self.kappa)


def fit_kappa(tables, grid, matches, kappas=(5, 10, 20, 40, 80, 160, 320, 1e9)):
    """How much each snap should move the strength, chosen by likelihood."""
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


@dataclass(frozen=True)
class FreshState(GameState):
    """A game state that also says whether its next snap starts a possession with the clock
    stopped (sim.is_fresh), and each side's timeouts left in the half (-1: not known)."""
    fresh: bool = False
    timeouts: tuple = (-1, -1)


TIMEOUTS_PER_HALF = 3


def timeouts_left(row):
    """(home, away) timeouts left in the half (or overtime period, two each) from the export's
    timeouts_used_a/b, or (-1, -1) where the export has none (older exports)."""
    a, b = row.get("timeouts_used_a"), row.get("timeouts_used_b")
    side = row.get("team_a_side")
    try:
        period = int(float(row.get("period") or 0))
    except ValueError:
        return -1, -1
    if a in ("", None) or b in ("", None) or side not in ("home", "away") or period < 1:
        return -1, -1
    have = TIMEOUTS_PER_HALF if period <= 4 else sim.OT_TIMEOUTS
    left_a = max(0, have - int(float(a)))
    left_b = max(0, have - int(float(b)))
    return (left_a, left_b) if side == "home" else (left_b, left_a)


def state_for(row):
    """playover.state_for, with whether the next snap is a fresh possession."""
    state, why = playover.state_for(row)
    if state is None:
        return state, why
    return FreshState(**{f.name: getattr(state, f.name) for f in fields(GameState)},
                      fresh=sim.is_fresh(row), timeouts=timeouts_left(row)), why


def start_from(state):
    """Simulation start fields for one game state."""
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
                kicks_second_half=side(state.opening_receiver) if state.opening_receiver else -1,
                fresh=phase == sim.SCRIM and bool(getattr(state, "fresh", False)),
                timeouts=getattr(state, "timeouts", (-1, -1)))


def _fill(start, i, fields):
    """Copy start fields into row i."""
    for k, v in fields.items():
        getattr(start, k)[i] = v


class Variant:
    """One way of running v14 (what it reacts to in the game)."""

    def __init__(self, name, kappa=KAPPA, react=False, theta_sd=False, profiles=True,
                 pace=True, sim_kw=None, grid=None):
        """Set the options."""
        self.name, self.kappa, self.react, self.grid = name, kappa, react, grid
        self.theta_sd, self.profiles, self.pace = theta_sd, profiles, pace
        self.sim_kw = sim_kw or {}


def match_seed(match_code):
    """A fixed random seed for each match."""
    import zlib
    return zlib.crc32(str(match_code).encode("utf-8"))


def price_states(tables, theta0, variant, snaps, a_home, states, messages, prof, n_paths, rng,
                 seed=None):
    """Margin and total distributions at each snapshot of a match."""
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
        start.theta[i], start.strength[i], start.strength_game[i] = sim.strength_draw(
            tables, eff.theta() if v.react else theta0,
            [tables.strength_league if p.form is None else p.form for p in prof])
        sds[i] = eff.sd() if v.react else 1.0 / math.sqrt(v.kappa)
        if v.profiles:
            start.aggression[i] = (prof[0].aggression, prof[1].aggression)
            start.kick[i] = (prof[0].kick, prof[1].kick)
        if v.pace:
            start.pace[i] = (prof[0].pace, prof[1].pace)
        if v.profiles:
            start.big[i] = (prof[0].big, prof[1].big)
            start.kick_tb[i] = (prof[0].kick_tb, prof[1].kick_tb)
            start.kick_nlz[i] = (prof[0].kick_nlz, prof[1].kick_nlz)
            start.loss[i] = (prof[0].loss, prof[1].loss)
            start.loss_size[i] = (prof[0].loss_size, prof[1].loss_size)
    kw = dict(theta_sd=sds if v.theta_sd else None, seed=seed, **v.sim_kw)
    home, away = sim.simulate(tables, start, n_paths, rng, **kw)
    books = [_distributions(home[i], away[i]) for i in range(len(messages))]
    if not np.any(tables.inplay_theta):
        return books
    home, away = sim.simulate(tables, start, n_paths, rng, in_play=True, **kw)
    return [(mp_, _distributions(home[i], away[i])[1]) for i, (mp_, _) in enumerate(books)]


def price_kickoff(tables, theta0, variant, prof, n_paths, rng, seed=None):
    """Margin and total distributions from the kick-off, before anything is known of the game:
    v14's pre-match price. Each side receives the opening kick in half the paths."""
    v = variant
    start = sim.Start(2)
    start.team[:] = (0, 1)
    start.kicks_second_half[:] = (1, 0)
    sds = np.full((2, 2), 1.0 / math.sqrt(v.kappa))
    forms = [tables.strength_league if p.form is None else p.form for p in prof]
    for i in range(2):
        start.theta[i], start.strength[i], start.strength_game[i] = sim.strength_draw(
            tables, theta0, forms)
        if v.profiles:
            start.aggression[i] = (prof[0].aggression, prof[1].aggression)
            start.kick[i] = (prof[0].kick, prof[1].kick)
        if v.pace:
            start.pace[i] = (prof[0].pace, prof[1].pace)
        if v.profiles:
            start.big[i] = (prof[0].big, prof[1].big)
            start.kick_tb[i] = (prof[0].kick_tb, prof[1].kick_tb)
            start.kick_nlz[i] = (prof[0].kick_nlz, prof[1].kick_nlz)
            start.loss[i] = (prof[0].loss, prof[1].loss)
            start.loss_size[i] = (prof[0].loss_size, prof[1].loss_size)
    kw = dict(theta_sd=sds if v.theta_sd else None, seed=seed)
    half = max(1, n_paths // 2)
    home, away = sim.simulate(tables, start, half, rng, **kw)
    margin, total = _distributions(home.ravel(), away.ravel())
    if np.any(tables.inplay_theta):
        home, away = sim.simulate(tables, start, half, rng, in_play=True, **kw)
        total = _distributions(home.ravel(), away.ravel())[1]
    return margin, total


def pool_context():
    """The multiprocessing start method for this platform."""
    return mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")


def _grade_matches(job):
    """Worker: price a list of matches at every snapshot."""
    (matches, tables_path, grid_path, variants, n_paths, seed, book, handles, require_live,
     priors, teams) = job
    ratings = big_ratings(tables_path)
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
            means = priors.get(code)
            if means is None:
                skipped["no_prematch_prediction"] += 1
                continue
        pair = (handles.get(code) if handles else None) or handles_of(match_rows)
        prof = ((book.profile(pair[0]), book.profile(pair[1])) if (book and pair)
                else (players.Profile(), players.Profile()))
        prof = with_big(prof, ratings, pair, *side_teams(pair, (teams or {}).get(code)))
        if priors is not None:
            theta0s = {(v.grid, v.pace, v.profiles): held_theta(tables, grids[v.grid], means, prof, v,
                                                                seed=match_seed(code))
                       for v in variants}
        else:
            theta0s = {(v.grid, v.pace, v.profiles): LEAGUE_THETA for v in variants}
        a_home = match_rows[0]["team_a_side"] == "home"
        snaps = sim.snap_records(match_rows)
        todo = []
        for row in playover.rows_for_match(match_rows):
            if row.prod_outcome is None:
                skipped["push_or_unresolved"] += 1
                continue
            if require_live and not row.prod_live:
                skipped["prod_not_live"] += 1
                continue
            state, why = state_for(row.source)
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
        dists = {v.name: price_states(tables, theta0s[(v.grid, v.pace, v.profiles)], v, snaps, a_home,
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
    """(home handle, away handle) from the export's own columns, or None."""
    r = match_rows[0] if match_rows else {}
    home, away = (r.get("home_handle") or "").strip(), (r.get("away_handle") or "").strip()
    return (home.upper(), away.upper()) if home and away else None


QUARTER_START_FIT = False
QUARTER_ROUNDS = 6
QUARTER_PATHS = 150


def quarter_start_states(matches, grid, priors=None, held=None):
    """Real states at the start of each quarter, with the points really scored in that quarter."""
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        rows = resolve_sides(rows)
        if rows[0].get("final_p1") in ("", None) or rows[0].get("final_p2") in ("", None):
            continue
        final = int(float(rows[0]["final_p1"])) + int(float(rows[0]["final_p2"]))
        if held is not None and code not in held:
            continue
        theta0 = held[code] if held is not None else \
            prior_theta(grid, None if priors is None else priors[code])
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
                continue
            state, _ = state_for(firsts[str(q)])
            if state is None:
                continue
            end = int(nxt["score_p1"]) + int(nxt["score_p2"]) if nxt is not None else final
            out.append((q, state, theta0, end - state.home_score - state.away_score))
    return out


def late_start_states(matches, grid, priors=None, held=None):
    """Real states at the two-minute mark of each half, with the points really scored after."""
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        rows = resolve_sides(rows)
        if rows[0].get("final_p1") in ("", None) or rows[0].get("final_p2") in ("", None):
            continue
        final = int(float(rows[0]["final_p1"])) + int(float(rows[0]["final_p2"]))
        if held is not None and code not in held:
            continue
        theta0 = held[code] if held is not None else \
            prior_theta(grid, None if priors is None else priors[code])
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
            state, _ = state_for(late)
            if state is None:
                continue
            end = int(nxt["score_p1"]) + int(nxt["score_p2"]) if nxt is not None else final
            out.append((q, state, theta0, end - state.home_score - state.away_score))
    return out


def fit_quarter_levels(tables, quarter_items, late_items, rounds=6, n_paths=150, seed=0,
                       verbose=False):
    """Each quarter's scoring level fitted from real quarter starts (off by default)."""
    def prepared(items):
        """Simulation starts for a set of states."""
        by_q = {}
        for it in items:
            by_q.setdefault(it[0], []).append(it)
        out = {}
        for q, its in by_q.items():
            start = sim.Start(len(its))
            for i, (_, state, theta0, _) in enumerate(its):
                _fill(start, i, start_from(state))
                _set_side(tables, start, i, theta0)
            out[q] = (start, float(np.mean([it[3] for it in its])))
        return out

    def scored(start, q):
        """Points the simulation scores in a quarter from these starts."""
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
    """Each quarter's scoring level fitted from real quarter starts."""
    by_q = {}
    for it in items:
        by_q.setdefault(it[0], []).append(it)
    for rnd in range(rounds):
        got, real = {}, {}
        for q, its in by_q.items():
            start = sim.Start(len(its))
            for i, (_, state, theta0, _) in enumerate(its):
                _fill(start, i, start_from(state))
                _set_side(tables, start, i, theta0)
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
    """Real in-game states, with the lead each really ended with."""
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
            state, _ = state_for(r)
            if state is None or state.down is None or state.pending_conversion is not None \
                    or not 1 <= state.period <= 4:
                continue
            sign = 1 if state.offense == HOME else -1
            lead = sign * (state.home_score - state.away_score)
            out.append((state, theta0, lead, sign * final - lead))
    return out


SETTLE_ROUNDS = 3
SETTLE_OUTER = 2
SETTLE_PATHS = 40
SETTLE_PRIOR = 150.0
# v9: the settle fit weights each match by its age at the build's cut-off, halving every
# SETTLE_HALF_LIFE days (None: every match alike). Sides two scores down score less week on week
# (touchdowns a snap behind 9+ in Q4: 0.107 the week of 24 Aug, 0.096 three weeks on; level,
# flat), so an even weighting keeps them scoring as they did weeks ago.
SETTLE_HALF_LIFE = 7.0


def settle_states(matches, grid, priors=None, as_of=None, half_life=None):
    """Every real snapshot with a snap to come as a simulation start: its settle cell, its start
    fields, the match's prior strengths, whether the next play was the offense's touchdown and its
    weight (v9: halving every `half_life` days of the match's age at `as_of`; 1 without)."""
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        w = 1.0
        if half_life and as_of is not None:
            day = match_day(rows)
            if day is not None:
                w = 0.5 ** (max(0, (as_of - day).days) / half_life)
        rows = sorted(resolve_sides(rows), key=lambda r: int(r["message"]))
        theta0 = prior_theta(grid, None if priors is None else priors[code])
        for r, nxt in zip(rows, rows[1:]):
            if r["play_kind"] not in sim.SNAP_KINDS or r["period"] != nxt["period"]:
                continue
            state, _ = state_for(r)
            if state is None or not state.has_snap or state.pending_conversion is not None \
                    or state.clock_seconds is None:
                continue
            fields = start_from(state)
            lead = (state.home_score - state.away_score) * (1 if state.offense == HOME else -1)
            scorer = sim._scorer(nxt, "TOUCHDOWN_TEAM") or nxt["offense"]
            td = nxt["play_kind"] == "TOUCHDOWN" and \
                playover.side_of(scorer, r.get("team_a_side")) == state.offense
            cell = int(sim.settle_index(state.period, state.clock_seconds, lead, fields["y"]))
            out.append((cell, fields, theta0, bool(td), w))
    return out


# v11: timeouts in the last two minutes of each half. Called at the fitted rates on every running
# clock, the simulation called 1.6-1.8 timeouts from the two-minute mark of a half where real sides
# called 2.3-2.7 (80% of them stopping a running clock). CALL_FIT scales each half's call rates, by
# one factor for Q2 and one for Q4, so that played from the first snap at or inside 2:00 of every
# real half, the simulation calls as many timeouts as the real sides did from there.
CALL_FIT = True
CALL_FIT_PATHS = 60
CALL_FIT_RANGE = (0.5, 4.0)
CALL_FIT_STEPS = 10
CALL_MAX = 0.95


def call_states(matches):
    """For each real half with timeouts known, its first snap at or inside 2:00 of Q2 or Q4: (quarter,
    start fields, timeouts both sides called from there to the end of the quarter)."""
    out = []
    for code, rows in matches.items():
        rows = sorted(resolve_sides(rows), key=lambda r: int(r["message"]))
        if not rows or "timeouts_used_a" not in rows[0]:
            continue
        for q in ("2", "4"):
            in_q = [r for r in rows if r["period"] == q]
            start = None
            for r in in_q:
                if r["play_kind"] != "SCRIMMAGE" or not r["clock_seconds"] or float(r["clock_seconds"]) > 120:
                    continue
                state, _ = state_for(r)
                if state is not None and state.has_snap and state.clock_seconds is not None:
                    start = (r, state)
                    break
            if start is None:
                continue
            r, state = start
            used = lambda x: int(x.get("timeouts_used_a") or 0) + int(x.get("timeouts_used_b") or 0)
            out.append((int(q), start_from(state), max(0, used(in_q[-1]) - used(r))))
    return out


MANAGE_PATHS = 40
MANAGE_STEPS = 8


def manage_states(matches):
    """For each real half, its first snap at or inside 2:00 of Q2 or Q4: (quarter, start fields,
    field goals tried on downs 1-3 from there to the quarter's end, snaps played from there (kneels
    in; not the half's last live state, never snapped))."""
    out = []
    for code, rows in matches.items():
        rows = resolve_sides(rows)
        for q in ("2", "4"):
            in_q = [r for r in rows if r["period"] == q]
            first = next((i for i, r in enumerate(in_q) if r["play_kind"] == "SCRIMMAGE" and r["clock_seconds"]
                          and float(r["clock_seconds"]) <= sim.LATE and r["down"]), None)
            if first is None:
                continue
            state, _ = state_for(in_q[first])
            if state is None or not state.has_snap:
                continue
            kicks = sum(1 for a, b in zip(in_q[first:], in_q[first + 1:])
                        if b["play_kind"] == "FIELD_GOAL" and a["down"] and int(float(a["down"])) < 4)
            snaps = sum(1 for a, b in zip(in_q[first:], in_q[first + 1:])
                        if a["play_kind"] in sim.SNAP_KINDS and a["down"] and b["play_kind"] not in ("PUNT", "FIELD_GOAL"))
            out.append((int(q), start_from(state), kicks, snaps))
    return out


LATE_COUNT_ROUNDS = 5


def fit_late_count(tables, items, rounds=LATE_COUNT_ROUNDS, n_paths=MANAGE_PATHS, seed=0):
    """Q2's and Q4's last two minutes of Tables.quarter_pace scaled together until as many snaps
    are played from every real two-minute state (manage_states) to the half's end as really were
    (kneels in). The quarter's own fit (fit_quarter_pace) matches the whole quarter and leaves the
    last two minutes with too many snaps and the rest too few (Q2: 10.0 snaps from the real 2:00
    states to the half's end, real 9.3). Returns {quarter: (halves, real, before, after, scale)}."""
    out = {}
    late = np.arange(sim.PACE_SLICES) >= int((sim.QUARTER - sim.LATE) // sim.PACE_SLICE)
    for q in (2, 4):
        its = [it for it in items if it[0] == q]
        if len(its) < 100:
            continue
        start = sim.Start(len(its))
        for i, it in enumerate(its):
            _fill(start, i, it[1])
            start.theta[i] = LEAGUE_THETA
        real = float(np.mean([it[3] for it in its]))

        def snaps():
            stats = {"_snaps": []}
            par_simulate(tables, start, n_paths, seed=seed + q, stats=stats)
            per = np.concatenate([x[1] for x in stats["_snaps"]])
            return ((per == q).sum() + (stats.get("kneel_plays", 0) if q == 4 else 0)) / (len(its) * n_paths)

        before = got = snaps()
        total = 1.0
        for _ in range(rounds):
            scale = float(np.clip(got / max(real, 1e-3), 0.85, 1.15))
            tables.quarter_pace[q - 1, late] = np.clip(tables.quarter_pace[q - 1, late] * scale, 0.4, 2.5)
            total *= scale
            got = snaps()
        out[q] = (len(its), real, before, got, total)
    return out


def fit_manage(tables, items, n_paths=MANAGE_PATHS, steps=MANAGE_STEPS, seed=0):
    """Tables.manage_p: for Q2 and Q4, the chance a play in range that would run the half out has
    its clock held for the kick, bisected so the field goals tried on downs 1-3 from every real
    two-minute state come as often as real. Returns {quarter: (halves, real, before, after, p)}."""
    out = {}
    for q in (2, 4):
        its = [it for it in items if it[0] == q]
        if len(its) < 100:
            continue
        start = sim.Start(len(its))
        for i, it in enumerate(its):
            _fill(start, i, it[1])
            start.theta[i] = LEAGUE_THETA
        real = float(np.mean([it[2] for it in its]))
        qi = q // 2 - 1

        def kicks(p):
            tables.manage_p[qi] = p
            stats = {}
            par_simulate(tables, start, n_paths, seed=seed + q, stats=stats)
            return stats.get(f"fg_early_q{q}", 0) / (len(its) * n_paths)

        before = kicks(float(tables.manage_p[qi]))
        lo, hi = 0.0, 1.0
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if kicks(mid) < real else (lo, mid)
        p = 0.5 * (lo + hi)
        out[q] = (len(its), real, before, kicks(p), p)
    return out


PACE_FIT_ROUNDS = 5
PACE_FIT_PATHS = 6


def pace_states(matches):
    """Every match's real state at the start of each quarter (its first snap) as simulation start
    fields, the snaps really played from it to the quarter's end (kneels too; not a quarter's last
    live state, which is snapped in the next quarter or not at all), and the real seconds of every
    snap by quarter and 20-second slice (sim.pace_slice): (starts, quarters, snaps played, seconds
    summed (4, slices), snaps timed (4, slices)). The timed snaps leave out kneels, which the sim
    plays apart, and a quarter's last snap, which its end cuts short."""
    starts, quarters, played, codes = [], [], [], []
    secs, snaps = np.zeros((4, sim.PACE_SLICES)), np.zeros((4, sim.PACE_SLICES))
    for code, rows in matches.items():
        rows = resolve_sides(rows)
        for q in (1, 2, 3, 4):
            first = next((i for i, r in enumerate(rows) if r["period"] == str(q) and r["play_kind"] in sim.SNAP_KINDS
                          and r["down"] and r["clock_seconds"] and r["score_p1"] and r["score_p2"]), None)
            if first is None:
                continue
            state, _ = state_for(rows[first])
            if state is None:
                continue
            n = 0
            for a, b in zip(rows[first:], rows[first + 1:] + [None]):
                if a["period"] != str(q) or b is None or b["period"] != str(q):
                    break
                if a["play_kind"] in sim.SNAP_KINDS and a["down"] and b["play_kind"] not in ("PUNT", "FIELD_GOAL"):
                    n += 1
            starts.append(start_from(state))
            quarters.append(q)
            played.append(n)
            codes.append(code)
        for r in sim.snap_records(rows):
            if not 1 <= r["period"] <= 4 or (sim.KNEELS and sim.is_kneel(r)):
                continue
            q, k = r["period"] - 1, int(sim.pace_slice(r["clock"]))
            secs[q, k] += r["seconds"]
            snaps[q, k] += 1
    return starts, np.array(quarters), np.array(played, dtype=float), secs, snaps, codes


def fit_quarter_pace(tables, starts, quarters, played, real_secs, real_snaps, codes=None, held=None,
                     rounds=PACE_FIT_ROUNDS, n_paths=PACE_FIT_PATHS, seed=0):
    """Tables.quarter_pace, in two steps. Each quarter's snap times are scaled 20-second slice by
    slice until the snaps played from every real quarter start take the real seconds there on
    average (kneels and snaps a quarter's end cut short apart); then each quarter's whole row is
    scaled until as many snaps are played from its real starts to its end as really were (kneels in),
    so the time the quarter spends elsewhere -- kicks, kneels, a clock run out -- comes out as real.
    With held ({match code: Held}, the reconcile's), each start is played as pricing plays its match;
    otherwise at league strengths.
    Returns (real, before, after) seconds a snap (4, slices) and (real, before, after) snaps a
    quarter (4,)."""
    if len(starts) < 200:
        return None
    start = sim.Start(len(starts))
    for i, f in enumerate(starts):
        _fill(start, i, f)
        theta0 = held.get(codes[i]) if (held is not None and codes is not None) else None
        _set_side(tables, start, i, LEAGUE_THETA if theta0 is None else theta0)
    real = real_secs / np.maximum(1, real_snaps)
    real_n = np.array([played[quarters == q].mean() for q in (1, 2, 3, 4)])
    first_q = np.repeat(quarters, n_paths)

    def run():
        stats = {"_snaps": []}
        par_simulate(tables, start, n_paths, seed=seed, stats=stats)
        rows, per, clk, used = (np.concatenate([x[k] for x in stats["_snaps"]]) for k in (0, 1, 2, 9))
        on = (per >= 1) & (per <= 4) & (clk - used > 0)          # the quarter's end cut it short: no time to read
        cell = (per[on] - 1) * sim.PACE_SLICES + sim.pace_slice(clk[on])
        m = np.bincount(cell, minlength=4 * sim.PACE_SLICES)
        secs = (np.bincount(cell, used[on], 4 * sim.PACE_SLICES) / np.maximum(1, m)).reshape(4, sim.PACE_SLICES)
        own = per == first_q[rows] if len(rows) else per == 0    # snaps in the quarter the start was in
        n = np.array([(own & (per == q)).sum() / max(1, (quarters == q).sum() * n_paths) for q in (1, 2, 3, 4)])
        n[3] += stats.get("kneel_plays", 0) / max(1, (quarters == 4).sum() * n_paths)  # kneels from every start;
        return secs, n                                                                 # nearly all from Q4's own

    before, n_before = run()
    got, n_got = before, n_before
    for _ in range(rounds):
        step = np.where(real_snaps >= 30, np.clip(real / np.maximum(got, 1.0), 0.8, 1.25), 1.0)
        step[[0, 2], -2:] = 1.0     # Q1's and Q3's last 40 seconds: the clock running out (sim.CLOCK_EXPIRES)
        #                             drops the slow snaps there, so their mean time does not answer a scale
        tables.quarter_pace = np.clip(tables.quarter_pace * step, 0.5, 2.0)
        got, n_got = run()
    for _ in range(rounds):
        scale = np.clip(n_got / np.maximum(real_n, 1e-3), 0.85, 1.15)
        tables.quarter_pace = np.clip(tables.quarter_pace * scale[:, None], 0.4, 2.5)
        got, n_got = run()
    return real, before, got, real_n, n_before, n_got


def fit_call_scale(tables, items, n_paths=CALL_FIT_PATHS, seed=0):
    """Scale each half's call rates (Q2, Q4) so the simulated timeouts from the two-minute mark
    match the real ones. Returns {quarter: (halves, real, before, after, scale)}."""
    out = {}
    for q in (2, 4):
        its = [it for it in items if it[0] == q]
        if len(its) < 50:
            continue
        start = sim.Start(len(its))
        for i, (_, fields, _) in enumerate(its):
            _fill(start, i, fields)
            start.theta[i] = LEAGUE_THETA
        real = float(np.mean([it[2] for it in its]))
        qi = 0 if q == 2 else 1
        base = tables.call_p[qi].copy()

        def calls(scale):
            tables.call_p[qi] = np.minimum(base * scale, CALL_MAX)
            stats = {}
            sim.simulate(tables, start, n_paths, np.random.default_rng(seed + q), stats=stats)
            return stats.get(f"timeouts_q{q}", 0) / (len(its) * n_paths)

        before = calls(1.0)
        lo, hi = CALL_FIT_RANGE
        for _ in range(CALL_FIT_STEPS):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if calls(mid) < real else (lo, mid)
        scale = 0.5 * (lo + hi)
        out[q] = (len(its), real, before, calls(scale), scale)
    return out


# v11: a side with the ball level in the fourth quarter's last two minutes takes the kick more readily
# than the close-kick table alone gives it, and gets there more often: real drives from level snaps
# in those two minutes ended in a field goal 68% of the time and a touchdown 16% (before 10 Sep); the
# simulation, 55% and 21%. CLOSE_LEVEL fits two shifts on those drives -- the level side's kick odds
# (close_fg and close_fourth's level rows, in log odds) and its efficiency in Q4's last three
# 40-second slices -- so the drive under way, played from every real level snap there, kicks and
# scores as the real ones did.
CLOSE_LEVEL = True
CLOSE_LEVEL_PATHS = 100
# v12: v11's fit took the level side's kick odds +4.0 everywhere (the top of its grid), so it kicked
# early, with time left, where real level sides run the drive down and kick at the end (median 0
# seconds left against v11's 43 from 2:00-1:01); the side 3 behind then had time to tie. The kick
# shift now applies only to the last CLOSE_LEVEL_LATE seconds of downs 1-3 (4th downs keep their real,
# clock-sliced rates, sim12.FOURTH_CLOCK), and the efficiency shift may go either way.
# The shifts are fitted by rounds of one-at-a-time grid searches (CLOSE_LEVEL_ROUNDS) to the real shares
# of the drive under way's field goals and touchdowns. The share of these games level at the end of
# regulation is simulated and reported too, but weighs nothing (CLOSE_LEVEL_OT_WEIGHT): given weight,
# with a 4th-down shift free as well (CLOSE_LEVEL_FOURTH), the fit took both kick shifts to the edges of
# their grids for under a point of overtime (16.4% -> 15.8%, real 9.5%, before 10 Sep) and kicked
# early again -- the overtime left is in the reply, not the level side's choices.
CLOSE_LEVEL_LATE = 10.0
CLOSE_LEVEL_KICK = np.arange(0.0, 6.01, 0.5)
CLOSE_LEVEL_FOURTH = np.array([0.0])
CLOSE_LEVEL_EFF = np.arange(-0.3, 0.301, 0.05)
CLOSE_LEVEL_ROUNDS = 2
CLOSE_LEVEL_OT_WEIGHT = 0.0


def close_level_cells():
    """The efficiency cells of a level offense in the fourth quarter's last three slices."""
    return [int(sim._cells_np(np.array(4), np.array(c), np.array(0)))
            for c in (sim.QUARTER / sim.CLOCK_CELLS * k - 1.0 for k in (1, 2, 3))]


def close_level_states(matches, grid, priors=None):
    """Every real fourth-quarter snap in the last two minutes with the score level: its start fields,
    the match's prior strengths, the points the real drive under way scored and whether the match
    went to overtime."""
    from .remaining import real_drive_points
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        rows = sorted(resolve_sides(rows), key=lambda r: int(r["message"]))
        theta0 = prior_theta(grid, None if priors is None else priors[code])
        overtime = any(str(r.get("period") or "").isdigit() and int(r["period"]) >= 5 for r in rows)
        for i, r in enumerate(rows):
            if r["period"] != "4" or r["play_kind"] not in sim.SNAP_KINDS:
                continue
            state, _ = state_for(r)
            if state is None or not state.has_snap or state.clock_seconds is None \
                    or state.clock_seconds > 120 or state.home_score != state.away_score:
                continue
            real = real_drive_points(rows, i, state.offense)
            if real is not None:
                out.append((start_from(state), theta0, real, overtime))
    return out


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_close_level(tables, items, n_paths=CLOSE_LEVEL_PATHS, seed=0):
    """The level side's late kick, 4th-down kick and efficiency shifts that bring its late drives'
    field-goal and touchdown shares, and the share of games level at the end of regulation, to the
    real ones; set on the tables. Returns (states, real (fg, td, ot), before, after, kick shift,
    4th-down shift, efficiency shift)."""
    if not items or tables.close_fg is None or tables.close_fourth is None:
        return None
    start = sim.Start(len(items))
    for i, (fields, theta, *_rest) in enumerate(items):
        _fill(start, i, fields)
        start.theta[i] = theta
    team = start.team.astype(int)
    base = np.where(team == 0, start.home, start.away)[:, None]
    pts = np.array([it[2] for it in items])
    ot = np.array([bool(it[3]) if len(it) > 3 else False for it in items])
    real = (float(np.mean(pts == 3)), float(np.mean(pts >= 6)), float(np.mean(ot)))
    cells = close_level_cells()
    fg0, f40, eff0 = tables.close_fg[0].copy(), tables.close_fourth[0].copy(), tables.eff_shift[cells].copy()
    late = int(np.searchsorted(np.array(sim.CLOSE_FG_CLOCK, dtype=float), CLOSE_LEVEL_LATE, side="right")) \
        if CLOSE_LEVEL_LATE else len(sim.CLOSE_FG_CLOCK)
    if len(f40) > 2:
        open_ = np.array((0.0,) + tuple(sim.FOURTH_CLOCK)) >= 30       # cells with more than 30 seconds
    else:
        open_ = np.array([True, False])

    def set_shifts(kick, k4, eff):
        if CLOSE_LEVEL_LATE:
            tables.close_fg[0] = fg0.copy()
            tables.close_fg[0][:, :late] = 1 / (1 + np.exp(-(_logit(fg0[:, :late]) + kick)))
            tables.close_fourth[0] = np.where(open_, 1 / (1 + np.exp(-(_logit(f40) + k4))), f40)
        else:                                                           # v11: one shift for all
            tables.close_fg[0] = 1 / (1 + np.exp(-(_logit(fg0) + kick)))
            tables.close_fourth[0] = 1 / (1 + np.exp(-(_logit(f40) + kick)))
        tables.eff_shift[cells] = eff0 + eff

    def shares(kick, k4, eff):
        set_shifts(kick, k4, eff)
        home, away = sim.simulate(tables, start, n_paths, np.random.default_rng(seed), one_drive=True)
        got = np.where(team[:, None] == 0, home, away) - base
        stats = {}
        sim.simulate(tables, start, n_paths, np.random.default_rng(seed + 1), stats=stats)
        return (float(np.mean(got == 3)), float(np.mean(got >= 6)),
                stats.get("overtime", 0) / (len(items) * n_paths))

    weights = (1.0, 1.0, CLOSE_LEVEL_OT_WEIGHT)
    err = lambda sh: sum(w * (a - b) ** 2 for w, a, b in zip(weights, sh, real))
    before = shares(0.0, 0.0, 0.0)
    x = [0.0, 0.0, 0.0]
    grids = (CLOSE_LEVEL_KICK, CLOSE_LEVEL_FOURTH if CLOSE_LEVEL_LATE else np.array([0.0]), CLOSE_LEVEL_EFF)
    for _ in range(CLOSE_LEVEL_ROUNDS):
        for k, grid in enumerate(grids):
            best = min((err(shares(*(x[:k] + [v] + x[k + 1:]))), v) for v in grid)
            x[k] = float(best[1])
    after = shares(*x)
    return len(items), real, before, after, x[0], x[1], x[2]

def fit_settle(tables, items, rounds=SETTLE_ROUNDS, n_paths=SETTLE_PATHS, prior=SETTLE_PRIOR,
               seed=0):
    """Each settle cell's share of would-be touchdowns held back, so that one snap played from
    every real state scores a touchdown as often as the real next play did. A small cell's target
    is shrunk toward what the simulation already gives. Returns {cell: (snaps, real, before,
    after)} touchdown rates."""
    by = {}
    for it in items:
        by.setdefault(it[0], []).append(it)
    prepared = {}
    for cell, its in by.items():
        start = sim.Start(len(its))
        for i, it in enumerate(its):
            _fill(start, i, it[1])
            start.theta[i] = it[2]
        team = start.team.astype(int)
        base = np.where(team == 0, start.home, start.away)
        w = np.array([it[4] if len(it) > 4 else 1.0 for it in its])
        # the weighted touchdowns and the weighted count, so the target's shrinkage reads the
        # cell's effective size
        prepared[cell] = (start, team, base, float(sum(it[3] * wi for it, wi in zip(its, w))),
                          float(w.sum()), w)

    def scored(cell):
        """The simulation's touchdown rate on one snap from the cell's states, weighted alike."""
        start, team, base, _, _, w = prepared[cell]
        home, away = sim.simulate(tables, start, n_paths, np.random.default_rng(seed + cell),
                                  max_steps=1, common=False)
        own = np.where(team[:, None] == 0, home, away)
        return float((((own - base[:, None]) == 6).mean(1) * w).sum() / max(1e-12, w.sum()))

    out = {}
    for rnd in range(rounds):
        for cell, (_, _, _, real, n, _) in sorted(prepared.items()):
            got = scored(cell)
            if rnd == 0:
                out[cell] = [n, real / n, got, got]
            target = (real + prior * got) / (n + prior)
            if got > 0:
                h = tables.td_hold[cell]
                tables.td_hold[cell] = float(np.clip(1.0 - (1.0 - h) * target / got, 0.0, sim.SETTLE_MAX))
    for cell in out:
        out[cell][3] = scored(cell)
    return {c: tuple(v) for c, v in out.items()}


BAND_LEAD = 7.0
BAND_BY_QUARTER = True
BAND_GROUPS = np.array([0, 0, 1, 2])


def _band_shift(pull, cells=None):
    """Each game-state cell's strength shift from the rubber band's pull."""
    cells = np.arange(sim.N_CELLS) if cells is None else cells
    quarter = cells // (sim.CLOCK_CELLS * sim.LEAD_CELLS)
    mid = np.array([-14.0, -4.0, 0.0, 4.0, 14.0])[cells % sim.LEAD_CELLS]
    pull = np.asarray(pull)
    if len(pull) == 3:
        return -pull[BAND_GROUPS[quarter]] * mid / BAND_LEAD
    return -pull[quarter // 2] * mid / BAND_LEAD * sim._quarter_mask()[cells]


def _slopes(changes, leads, groups, n_groups=2):
    """How much of each point of lead comes back by the end, by group."""
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
    """Fit how strongly trailing sides come back, so simulated leads shrink as real ones do."""
    base = tables.eff_shift.copy()
    n = 3 if BAND_BY_QUARTER else 2
    leads = np.array([lead for _, _, lead, _ in items], dtype=float)
    changes = np.array([ch for *_, ch in items], dtype=float)
    groups = np.array([BAND_GROUPS[st.period - 1] if n == 3 else (0 if st.period <= 2 else 1)
                       for st, *_ in items])
    real = _slopes(changes, leads, groups, n)
    sign = np.array([1 if st.offense == HOME else -1 for st, *_ in items], dtype=float)

    def start_for(sel):
        """Simulation starts for a set of states."""
        start = sim.Start(len(sel))
        for k, i in enumerate(sel):
            st, theta0, _, _ = items[i]
            _fill(start, k, start_from(st))
            start.theta[k] = theta0
        return start

    def simulated(pull, sel, start):
        """How the simulated leads change with this pull."""
        tables.eff_shift = base + _band_shift(pull)
        home, away = sim.simulate(tables, start, n_paths, np.random.default_rng(seed), seed=seed + 1,
                                  distinct=True)
        return _slopes(sign[sel] * (home - away).mean(1) - leads[sel], leads[sel], groups[sel], n)

    def secant(pull, which, sel, start):
        """Solve one group's pull so its leads come back as real ones do; the iterate that comes
        nearest wins (the simulated slopes are noisy)."""
        p0 = pull.copy()
        g0 = simulated(p0, sel, start)
        p1 = pull + 0.1 * which
        g1 = simulated(p1, sel, start)
        q = int(np.argmax(which))
        tried = [(p0, g0), (p1, g1)]
        for rnd in range(rounds):
            if verbose:
                print(f"    round {rnd}: pull {p1} slopes real {real} simulated {g1}")
            moved = np.abs(p1 - p0) > 1e-6
            d = np.where(moved, (g1 - g0) / np.where(moved, p1 - p0, 1.0), -0.1)
            d = np.minimum(np.nan_to_num(d, nan=-0.1), -0.01)
            step = np.where(which > 0, np.clip(np.nan_to_num((real - g1) / d), -0.3, 0.3), 0.0)
            p0, g0 = p1, g1
            p1 = np.clip(p1 + step, -1.0, 1.0)
            g1 = simulated(p1, sel, start)
            tried.append((p1, g1))
        return min(tried, key=lambda pg: abs(np.nan_to_num(pg[1][q] - real[q], nan=9.0)))

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


IN_PLAY_FIT = False
REST_EVERY = 4
REST_PATHS = 60
REST_MAX_STATES = 4000
REST_MIN_STATES = 100
REST_MIN_CELL = 150
REST_BY_BAND = False
REST_SEGMENTS = tuple(range(1, sim.N_SEGMENTS))
REST_ROUNDS = 3


def rest_of_game_states(matches, grid, priors=None, every=REST_EVERY, held=None):
    """Real in-game states with the points really still to come."""
    out = []
    for code, rows in matches.items():
        if priors is not None and code not in priors:
            continue
        rows = resolve_sides(rows)
        if rows[0].get("final_p1") in ("", None) or rows[0].get("final_p2") in ("", None):
            continue
        if held is not None and code not in held:
            continue
        theta0 = held[code] if held is not None else \
            prior_theta(grid, None if priors is None else priors[code])
        final = int(float(rows[0]["final_p1"])) + int(float(rows[0]["final_p2"]))
        for r in rows[::every]:
            state, _ = state_for(r)
            if state is None or state.period < 1:
                continue
            cell = (sim.inplay_segment(state.period, state.clock_seconds),
                    sim.margin_band(state.home_score - state.away_score))
            out.append((cell, state, theta0, final - state.home_score - state.away_score))
    return out


def fit_rest_of_game(tables, items, n_paths=REST_PATHS, rounds=REST_ROUNDS, seed=0,
                     max_states=REST_MAX_STATES, verbose=False):
    """The in-play scoring shift for totals (off by default)."""

    def prepare(groups, least):
        """Group the states and their simulation starts."""
        out = {}
        for key, its in groups.items():
            if len(its) < least:
                continue
            if len(its) > max_states:
                its = [its[i] for i in np.linspace(0, len(its) - 1, max_states).astype(int)]
            start = sim.Start(len(its))
            for i, (_, state, theta0, _) in enumerate(its):
                _fill(start, i, start_from(state))
                _set_side(tables, start, i, theta0)
            out[key] = (start, float(np.mean(start.home + start.away)),
                        float(np.mean([it[3] for it in its])))
        return out

    def rest(prepared, key):
        """Simulated points still to come for a group."""
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
        """Solve the shifts for these groups."""
        first = {}
        for rnd in range(rounds):
            for key in keys:
                got = rest(prepared, key)
                first.setdefault(key, got)
                g = key if isinstance(key, int) else key[0]
                k = order.index(g)
                own = got - (real_after[order[k - 1]] if k else 0.0)
                step = float(np.clip((prepared[key][2] - got) / (1.8 * max(1.0, own)), -0.3, 0.3))
                apply(key, step)
                if verbose:
                    print(f"    round {rnd} {key}: real {prepared[key][2]:.2f} simulated {got:.2f}")
        return first

    def by_segment(g, step):
        """Shift one segment of the game."""
        tables.inplay_theta[g, :] += step

    def by_cell_(key, step):
        """Shift one segment and margin band."""
        tables.inplay_theta[key[0], key[1]] += step

    seg_before = fit(segs, [g for g in order if g in REST_SEGMENTS], by_segment)
    cell_keys = sorted(cells, key=lambda c: (-c[0], c[1])) if REST_BY_BAND else []
    cell_before = fit(cells, cell_keys, by_cell_)
    if 7 not in segs and 7 in REST_SEGMENTS:
        tables.inplay_theta[7] = tables.inplay_theta[5]
    return ({g: (segs[g][2], seg_before[g], rest(segs, g)) for g in sorted(segs) if g in seg_before},
            {c: (cells[c][2], cell_before[c], rest(cells, c)) for c in cell_keys})

def match_day(match_rows):
    """The date a match was played."""
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
    """Real against simulated points in each quarter, from real quarter starts."""
    items = quarter_start_states(matches, grid, priors)
    _, got, real = fit_period_theta_states(tables, items, rounds=1, n_paths=n_paths, seed=seed)
    return got, real


# v13: each side's big-play rate (sim13.BIG_PLAYS), fitted on every scrimmage snap built on. A
# snap's chance of a big play (sim13.BIG_GAIN yards or more) is its bin's league rate, moved in log
# odds by four effects: the gamer with the ball, its NFL team, and what the other gamer and the
# other team allow. Each effect is shrunk toward 0 by how far its kind really spreads beyond noise
# (fitted, as for form), and recent weeks weigh more (BIG_HALF_LIFE). A side's rate in a match is the
# odds its four effects add up to (big_rates), kept within BIG_POINTS' range. Touchdown length by
# gamer and by team, held out, needed both: neither is the other in disguise (gamers spread their
# games over many teams), and the pair adds nothing beyond them.
BIG_HALF_LIFE = 60.0
BIG_FACTORS = ("gamer", "team", "gamer_allows", "team_allows")
BIG_ROUNDS = 40
BIG_FILE = "v14big.json"


def match_teams(history):
    """{match code: (PLAYER_1 handle, PLAYER_1 team, PLAYER_2 handle, PLAYER_2 team)} off a
    match history."""
    out = {}
    for r in history or ():
        t1, t2 = (r.get("PLAYER_1_TEAM") or "").strip(), (r.get("PLAYER_2_TEAM") or "").strip()
        if t1 and t2:
            out[r["MATCH_CODE"]] = ((r.get("PLAYER_1_HANDLE") or "").strip().upper(), t1,
                                    (r.get("PLAYER_2_HANDLE") or "").strip().upper(), t2)
    return out


def side_teams(pair, teams):
    """(home team, away team) for handles `pair` off a match_teams entry, turned round when the
    history has the handles the other way; ("", "") when unknown."""
    if not teams or not pair:
        return "", ""
    h1, t1, h2, t2 = teams
    if (h2, h1) == tuple(pair) and (h1, h2) != tuple(pair):
        return t2, t1
    return t1, t2


def big_records(matches, tables, handles, teams, as_of=None, half_life=BIG_HALF_LIFE, kind="big"):
    """Every scrimmage snap of the matches as (big?, league log odds of its bin, weight, gamer,
    team, other gamer, other team). With kind "loss": did it lose yards (v14)."""
    if kind == "loss":
        rate = (tables.n_loss_stop + tables.n_loss_run) / np.maximum(1, tables.count)
        hit = sim._is_loss
    else:
        rate = (tables.n_big_stop + tables.n_big_run) / np.maximum(1, tables.count)
        hit = sim._is_big
    out = []
    for code, rows in matches.items():
        pair = handles.get(code)
        if not pair or not rows:
            continue
        home_team, away_team = side_teams(pair, teams.get(code))
        a_home = rows[0].get("team_a_side") == "home"
        day = sim.match_date(code)
        w = 1.0 if (as_of is None or day is None) else 2.0 ** (-(as_of - day).days / half_life)
        for rec in sim.snap_records(rows):
            p = rate[rec["key"]]
            if not 0 < p < 1 or rec.get("kneel") or sim.is_kneel(rec):
                continue
            home = (rec["offense"] == "TEAM_A") == a_home
            g, o = (pair[0], pair[1]) if home else (pair[1], pair[0])
            t, ot = (home_team, away_team) if home else (away_team, home_team)
            out.append((float(hit(rec)), float(np.log(p / (1 - p))), w, g, t, o, ot))
    return out


def _big_fit(y, offset, w, codes, n_levels, ridge, rounds=BIG_ROUNDS):
    """Ridge logistic regression on factor effects by one Newton step per factor per round:
    {factor: effects}, {factor: each effect's Hessian}."""
    eff = {f: np.zeros(n) for f, n in n_levels.items()}
    hess = {}
    for _ in range(rounds):
        for f, ix in codes.items():
            eta = offset + sum(eff[g][codes[g]] for g in codes)
            mu = 1.0 / (1.0 + np.exp(-eta))
            grad = np.bincount(ix, w * (y - mu), n_levels[f]) - ridge[f] * eff[f]
            h = np.bincount(ix, w * mu * (1 - mu), n_levels[f]) + ridge[f]
            eff[f] = eff[f] + grad / h
            hess[f] = h
    return eff, hess


def fit_big_ratings(records):
    """Each gamer's and team's big-play effects (log odds, with the ball and against it), shrunk:
    {"gamer": {...}, "team": {...}, "gamer_allows": {...}, "team_allows": {...}, "spread": {...}}."""
    if not records:
        return {f: {} for f in BIG_FACTORS} | {"spread": {}, "snaps": 0}
    y = np.array([r[0] for r in records])
    offset = np.array([r[1] for r in records])
    w = np.array([r[2] for r in records])
    levels, codes = {}, {}
    for k, f in enumerate(BIG_FACTORS):
        vals = [r[3 + k] or "" for r in records]
        levels[f] = sorted(set(vals))
        idx = {v: i for i, v in enumerate(levels[f])}
        codes[f] = np.array([idx[v] for v in vals])
    n_levels = {f: len(v) for f, v in levels.items()}
    # how far each kind of effect really spreads: a light fit, less its noise
    eff, hess = _big_fit(y, offset, w, codes, n_levels, {f: 1.0 for f in BIG_FACTORS})
    spread = {}
    for f in BIG_FACTORS:
        known = np.array([v != "" for v in levels[f]])
        if known.sum() < 2:
            spread[f] = 0.0
            continue
        e, h = eff[f][known], hess[f][known]
        tau2 = float(np.average((e - np.average(e, weights=h)) ** 2, weights=h) - np.average(1 / h, weights=h))
        spread[f] = max(tau2, 1e-4)
    eff, _ = _big_fit(y, offset, w, codes, n_levels, {f: 1.0 / max(spread[f], 1e-4) for f in BIG_FACTORS})
    out = {f: {lv: round(float(e), 4) for lv, e in zip(levels[f], eff[f]) if lv != ""} for f in BIG_FACTORS}
    out["spread"] = {f: round(float(np.sqrt(v)), 4) for f, v in spread.items()}
    out["snaps"] = len(records)
    return out


def big_rates(ratings, pair, home_team="", away_team=""):
    """(home, away) big-play rates for handles `pair` and the two teams: the odds each side's
    four effects add up to, over the build's elasticity (fit_big_elasticity), within BIG_POINTS.
    (1, 1) without ratings or handles."""
    if not ratings or not pair:
        return 1.0, 1.0
    get = lambda f, k: ratings.get(f, {}).get(k or "", 0.0)
    h, a = (pair[0] or "").upper(), (pair[1] or "").upper()
    xh = get("gamer", h) + get("team", home_team) + get("gamer_allows", a) + get("team_allows", away_team)
    xa = get("gamer", a) + get("team", away_team) + get("gamer_allows", h) + get("team_allows", home_team)
    k = ratings.get("elasticity", 1.0) or 1.0
    xh, xa = xh / k, xa / k
    lo, hi = np.log(BIG_POINTS[0]), np.log(BIG_POINTS[-1])
    return float(np.exp(np.clip(xh, lo, hi))), float(np.exp(np.clip(xa, lo, hi)))


def with_big(prof, ratings, pair, home_team="", away_team=""):
    """The two profiles with this match's big-play rates on them (copies), and (v14) its
    lost-yardage rates and each gamer's loss size."""
    import dataclasses
    bh, ba = big_rates(ratings, pair, home_team, away_team)
    out = [dataclasses.replace(prof[0], big=bh), dataclasses.replace(prof[1], big=ba)]
    if ratings and ratings.get("loss"):
        lh, la = big_rates(ratings["loss"], pair, home_team, away_team)
        sizes = ratings.get("loss_size", {})
        for k, (rate, handle) in enumerate(((lh, pair[0]), (la, pair[1]))):
            out[k] = dataclasses.replace(out[k], loss=rate, loss_size=float(sizes.get((handle or "").upper(), 1.0)))
    return tuple(out)


# v14: how deep each gamer's losses go. The league's losses, deepest first, are the draw inside a
# bin's losses; a side's loss_size s leans it to v ** s (sim14.zone_warp), so the mean loss over
# the league's losses at s is read off a grid and each gamer's own mean, shrunk toward the league's
# by LOSS_SIZE_PRIOR losses, picks their s.
LOSS_SIZE_PRIOR = 30.0
LOSS_SIZE_GRID = np.exp(np.linspace(-1.5, 1.5, 61))


def fit_loss_sizes(matches, handles, prior=LOSS_SIZE_PRIOR):
    """{handle: loss size}: each gamer's s, from the yards their losses lost."""
    yards, by = [], defaultdict(list)
    for code, rows in matches.items():
        pair = handles.get(code)
        if not pair or not rows:
            continue
        a_home = rows[0].get("team_a_side") == "home"
        for rec in sim.snap_records(rows):
            if sim._is_loss(rec):
                g = pair[0] if (rec["offense"] == "TEAM_A") == a_home else pair[1]
                yards.append(-rec["gain"])
                by[g].append(-rec["gain"])
    if len(yards) < 50:
        return {}
    deep = np.sort(np.array(yards, float))[::-1]
    v = (np.arange(400) + 0.5) / 400
    means = np.array([deep[np.minimum(len(deep) - 1, (v ** s * len(deep)).astype(int))].mean()
                      for s in LOSS_SIZE_GRID])
    league = float(deep.mean())
    out = {}
    for g, ys in by.items():
        target = (sum(ys) + prior * league) / (len(ys) + prior)
        out[g] = round(float(np.interp(target, means, LOSS_SIZE_GRID)), 4)
    return out


def big_ratings(model_dir_or_tables):
    """The model's big-play ratings, or None."""
    import os
    d = model_dir_or_tables if os.path.isdir(model_dir_or_tables) else os.path.dirname(model_dir_or_tables)
    path = os.path.join(d, BIG_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# v14: each kicker's touchback and no-landing-zone rates (sim14.KICK_STYLES), as multiples of the
# league's for the part of the game the kick came in: their count over what the league's rates give
# their kicks, shrunk toward 1 by KICK_STYLE_PRIOR kicks' worth of the league.
KICK_STYLE_PRIOR = 5.0


def fit_kick_styles(matches, tables, handles, prior=KICK_STYLE_PRIOR):
    """{handle: (touchback multiple, no-landing-zone multiple)} over the matches' kick-offs."""
    if tables.kick_mix is None:
        return {}
    seen = defaultdict(lambda: np.zeros(4))           # touchbacks, expected, no landing, expected
    for code, rows in matches.items():
        pair = handles.get(code)
        if not pair or not rows:
            continue
        a_home = rows[0].get("team_a_side") == "home"
        for late, tb, nlz, onside, field, secs, kicker in sim.kick_kinds(rows):
            g = pair[0] if (kicker == "TEAM_A") == a_home else pair[1]
            base_tb, base_nlz = tables.kick_mix[int(late)]
            seen[g] += (tb, base_tb, nlz, base_nlz)
    return {g: (float((v[0] + prior) / (v[1] + prior)), float((v[2] + prior) / (v[3] + prior)))
            for g, v in seen.items()}


FORM_HALF_LIFE = 60.0
FORM_DAYS = 365
FORM_GRID = np.array([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
FORM_PATHS = 2000


def _var_terms(mu):
    """Quadratic terms of a mean."""
    mu = np.asarray(mu, dtype=float)
    return np.stack([np.ones_like(mu), mu, mu * mu], axis=-1)


def _cov_terms(mh, ma):
    """Quadratic terms of two means."""
    mh, ma = np.asarray(mh, dtype=float), np.asarray(ma, dtype=float)
    return np.stack([np.ones_like(mh), mh + ma, mh * ma, mh * mh + ma * ma], axis=-1)


def fixed_strength_spread(tables, grid=FORM_GRID, n_paths=FORM_PATHS, seed=0):
    """How the simulation's scores spread with the strengths fixed, and how points move with
    strength."""
    g = len(grid)
    start = sim.Start(g * g)
    th, ta = np.meshgrid(grid, grid, indexing="ij")
    start.theta = np.stack([th.ravel(), ta.ravel()], axis=1)
    rng = np.random.default_rng(seed)
    start.team = rng.integers(0, 2, g * g).astype(np.int8)
    start.kicks_second_half = 1 - start.team
    home, away = sim.simulate(tables, start, n_paths, rng, common=False)
    mh, ma = home.mean(1), away.mean(1)
    means = np.concatenate([mh, ma])
    var = np.concatenate([home.var(1), away.var(1)])
    cov = ((home - mh[:, None]) * (away - ma[:, None])).mean(1)
    var_fn = np.linalg.lstsq(_var_terms(means), var, rcond=None)[0]
    cov_fn = np.linalg.lstsq(_cov_terms(mh, ma), cov, rcond=None)[0]
    log_home = np.log(np.maximum(0.5, mh)).reshape(g, g)
    slope = np.gradient(log_home, grid, axis=0).mean(1)
    return var_fn, cov_fn, slope


def form_sides(history, means, before, days=FORM_DAYS, half_life=FORM_HALF_LIFE):
    """Each side of each recent finished match: player, weight, points scored and NB2's expected
    points."""
    out = []
    since = before - dt.timedelta(days=days)
    for r in history:
        start = nb2_prior._start(r)
        if (r.get("PLAYER_1_FINAL_SCORE") in ("", None) or start is None
                or not since <= start < before or r["MATCH_CODE"] not in means):
            continue
        w = 2.0 ** (-(before - start).total_seconds() / 86400.0 / half_life)
        mu = means[r["MATCH_CODE"]]
        for k, side in ((0, "PLAYER_1"), (1, "PLAYER_2")):
            out.append((r[f"{side}_HANDLE"].strip().upper(), w, float(r[f"{side}_FINAL_SCORE"]),
                        mu[k], r["MATCH_CODE"]))
    return out


def fit_form(sides, var_fn, cov_fn):
    """The game's shared swing and each player's own form, fitted from results."""
    handle = np.array([s[0] for s in sides])
    w = np.array([s[1] for s in sides])
    y = np.array([s[2] for s in sides])
    mu = np.maximum(1.0, np.array([s[3] for s in sides]))
    r = ((y - mu) ** 2 - _var_terms(mu) @ var_fn) / mu ** 2
    wm = w * mu ** 2
    num = den = 0.0
    by_match = {}
    for s in sides:
        by_match.setdefault(s[4], []).append(s)
    for pair in by_match.values():
        if len(pair) == 2:
            (_, w1, y1, m1, _), (_, _, y2, m2, _) = pair
            m1, m2 = max(1.0, m1), max(1.0, m2)
            num += w1 * ((y1 - m1) * (y2 - m2) - _cov_terms(m1, m2) @ cov_fn)
            den += w1 * m1 * m2
    game = max(0.0, num / den) if den > 0 else 0.0
    league = float((wm * r).sum() / wm.sum())
    per_side = float((wm * (r - league) ** 2).sum() / wm.sum())
    raw, noise = {}, {}
    for h in np.unique(handle):
        k = handle == h
        raw[h] = float((wm[k] * r[k]).sum() / wm[k].sum())
        noise[h] = per_side * float((wm[k] ** 2).sum() / wm[k].sum() ** 2)
    names = list(raw)
    size = np.array([wm[handle == h].sum() for h in names])
    spread = float(np.average([(raw[h] - league) ** 2 for h in names], weights=size))
    tau2 = max(0.0, spread - float(np.average([noise[h] for h in names], weights=size)))
    form = {h: max(0.0, league - game + tau2 / (tau2 + noise[h]) * (raw[h] - league))
            for h in names}
    return game, max(0.0, league - game), form


def _print_settle(settled, tables):
    """The settle fit, by part of the game and the offense's lead: touchdowns per snap, real /
    simulated before -> after, and the mean share held back."""
    segs = ("Q1", "Q2", "Q2 last 2:00", "Q3", "Q4", "Q4 last 2:00")
    leads = ("behind 9+", "behind 1-8", "level", "ahead 1-8", "ahead 9+")
    print("  touchdowns per snap from real states, real / simulated before -> after the settle "
          "fit (share of would-be touchdowns held back):")
    for g, seg in enumerate(segs):
        parts = []
        for lc, lead in enumerate(leads):
            cells = [(g * sim.LEAD_CELLS + lc) * 4 + z for z in range(4)]
            rows = [settled[c] for c in cells if c in settled]
            n = sum(r[0] for r in rows)
            if not n:
                continue
            w = lambda k: sum(r[0] * r[k] for r in rows) / n
            held = sum(settled[c][0] * tables.td_hold[c] for c in cells if c in settled) / n
            parts.append(f"{lead} {w(1):.3f}/{w(2):.3f}->{w(3):.3f} ({held:.2f})")
        print(f"    {seg}: " + "; ".join(parts))


def build(matches, out_dir, grid_paths=6000, verbose=True, handles=None, history=None,
          before=None, prior="nb2"):
    """Build the model: tables, fits, prior grid, the pre-match model (`prior`: "nb2" or
    "glmer", fitted on `history`) and player profiles."""
    import copy
    import os
    import time
    clock0 = [time.time()]

    def lap(label):
        """The build's time on a stage (verbose)."""
        if verbose:
            print(f"    [{label}: {time.time() - clock0[0]:.0f}s]", flush=True)
        clock0[0] = time.time()

    if prior not in PRIORS:
        raise ValueError(f"prior must be one of {', '.join(PRIORS)}")
    os.makedirs(out_dir, exist_ok=True)
    pre = priors = None
    if history is not None:
        if before is None:
            days = [match_day(rows) for rows in matches.values()]
            last = max(d for d in days if d is not None)
            before = dt.datetime.combine(last + dt.timedelta(days=1), dt.time())
        if verbose and prior == "glmer":
            print("  pre-match: fitting the glmer model in R (a few minutes on a full history)")
        pre = PRIORS[prior].build(history, os.path.join(out_dir, prior), before)
        with open(os.path.join(out_dir, PRIOR_FILE), "w", encoding="utf-8") as fh:
            json.dump({"prior": prior}, fh)
        priors = pre.means([r for r in history if r["MATCH_CODE"] in matches], results=history)
        shrink_path = os.path.join(out_dir, SHRINK_FILE)
        if os.path.exists(shrink_path):
            os.remove(shrink_path)
        if PRIOR_SHRINK and priors:
            centre = (float(np.mean([h + a for h, a in priors.values()])),
                      float(np.mean([h - a for h, a in priors.values()])))
            if verbose:
                print(f"  prior shrink: refitting the pre-match model at the start of each of the"
                      f" {SHRINK_WINDOWS} stretches of {SHRINK_DAYS} days before the cut-off")
            shrink = fit_shrink(history, prior, before, centre, out_dir, scale=pre.scale)
            if shrink is not None:
                with open(shrink_path, "w", encoding="utf-8") as fh:
                    json.dump(shrink, fh, indent=1)
                pre = ShrunkPrematch(pre, shrink)
                priors = {c: shrink_means(m, shrink) for c, m in priors.items()}
                if verbose:
                    print(f"  prior shrink: on {shrink['matches']:,} matches out of sample, real totals move"
                          f" {shrink['total_raw']:.3f} (+-{shrink['total_se']:.3f}) and margins"
                          f" {shrink['margin_raw']:.3f} (+-{shrink['margin_se']:.3f}) a point of the"
                          f" prediction -> totals pulled {shrink['total_slope']:.3f}, margins"
                          f" {shrink['margin_slope']:.3f} toward {shrink['total_centre']:.2f} /"
                          f" {shrink['margin_centre']:+.2f}")
                    print("    by stretch: " + "; ".join(
                        f"{e['from'][:10]} {e['matches']:,} matches, totals {e['total_raw']:.2f},"
                        f" margins {e['margin_raw']:.2f}" for e in shrink["each"]))
            elif verbose:
                print(f"  prior shrink: fewer than {SHRINK_MIN_MATCHES} finished matches in the"
                      f" {SHRINK_WINDOWS * SHRINK_DAYS} days before the cut-off, none applied")
    lap("pre-match model")
    handles = dict(handles or {})
    for code, rows in matches.items():
        handles.setdefault(code, handles_of(rows))
    handles = {c: h for c, h in handles.items() if h}
    days = [d for d in (match_day(r) for r in matches.values()) if d is not None]
    as_of = (before.date() if before is not None else
             max(days) + dt.timedelta(days=1) if days else None)
    tables = sim.Tables.build(matches, handles=handles, as_of=as_of, half_life=sim.CLOCK_HALF_LIFE)
    lap("tables")
    if CALL_FIT and sim.TIMEOUTS and getattr(tables, "call_fitted", False):
        scaled = fit_call_scale(tables, call_states(matches))
        if verbose and scaled:
            print("  timeouts from the two-minute mark, real / simulated before -> after: "
                  + "; ".join(f"Q{q} {r:.2f} / {b:.2f} -> {a:.2f} (rates x{s:.2f}, {n:,} halves)"
                              for q, (n, r, b, a, s) in sorted(scaled.items())))
    if sim.QUARTER_PACE:
        paced = fit_quarter_pace(tables, *pace_states(matches))
        if verbose and paced:
            real_p, before_p, after_p, real_n, before_n, after_n = paced
            print("  snaps a quarter from the real quarter starts, real / simulated before -> after: "
                  + "; ".join(f"Q{q + 1} {real_n[q]:.2f} / {before_n[q]:.2f} -> {after_n[q]:.2f}" for q in range(4)))
            for q in range(4):
                print(f"    Q{q + 1} seconds a snap by 20-second slice, real / after: "
                      + " ".join(f"{a:.0f}/{c:.0f}" for a, c in zip(real_p[q], after_p[q]))
                      + "; x " + " ".join(f"{x:.2f}" for x in tables.quarter_pace[q]))
        if CALL_FIT and sim.TIMEOUTS and getattr(tables, "call_fitted", False):
            # the pace moves how often the clock runs before a snap, and so the timeouts: refit them
            scaled = fit_call_scale(tables, call_states(matches))
            if verbose and scaled:
                print("  timeouts again, after the pace: "
                      + "; ".join(f"Q{q} {r:.2f} / {b:.2f} -> {a:.2f} (rates x{s:.2f})"
                                  for q, (n, r, b, a, s) in sorted(scaled.items())))
    if sim.MANAGE_FIT:
        two_minute = manage_states(matches)
        if sim.QUARTER_PACE:
            counted = fit_late_count(tables, two_minute)
            if verbose and counted:
                print("  snaps from the two-minute mark to the half's end, real / simulated before -> after: "
                      + "; ".join(f"Q{q} {r:.2f} / {b:.2f} -> {a:.2f} (x{k:.2f})" for q, (n, r, b, a, k) in sorted(counted.items())))
        managed = fit_manage(tables, two_minute)
        if verbose and managed:
            print("  end-of-half field goals on downs 1-3 from the two-minute mark, real / simulated before -> "
                  "after: " + "; ".join(f"Q{q} {r:.3f} / {b:.3f} -> {a:.3f} (clock held {p:.2f})"
                                        for q, (n, r, b, a, p) in sorted(managed.items())))
    lap("timeouts and pace")
    real = sim.quarter_points(matches)
    offsets, got = sim.fit_period_theta(tables, real)
    quick = PriorGrid.build(tables, n_paths=max(500, grid_paths // 4))
    items = in_game_states(matches, quick, priors=priors)
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
    if sim.SETTLE_FIT:
        states = settle_states(matches, quick, priors, as_of=as_of, half_life=SETTLE_HALF_LIFE)
        for _ in range(SETTLE_OUTER):
            settled = fit_settle(tables, states)
            offsets, got = sim.fit_period_theta(tables, real)
        if verbose:
            _print_settle(settled, tables)
    if CLOSE_LEVEL and sim.CLOSE_FG:
        fitted = fit_close_level(tables, close_level_states(matches, quick, priors))
        if fitted:
            offsets, got = sim.fit_period_theta(tables, real)      # the scoring level, refitted after
        if verbose and fitted:
            n, drives, sim_before, sim_after, kick, k4, eff = fitted
            pct = lambda x: (f"field goal {100 * x[0]:.1f}% / touchdown {100 * x[1]:.1f}% / overtime"
                             f" {100 * x[2]:.1f}%")
            print(f"  close endings: from {n:,} real level snaps in Q4's last two minutes, real"
                  f" {pct(drives)}; simulated {pct(sim_before)} -> {pct(sim_after)} (level kick odds in"
                  f" the last {CLOSE_LEVEL_LATE:.0f}s {kick:+.1f}, 4th down {k4:+.1f}, efficiency {eff:+.2f})")
    lap("scoring levels, rubber band, settle, close endings")
    if verbose and sim.TIMEOUTS:
        if tables.call_p.any():
            source = ("fitted on the real calls given (--timeouts)" if getattr(tables, "call_fitted", False)
                      else "the real rates in sim14.DEFAULT_CALL_P (build with --timeouts to refit)")
            print(f"  timeouts: {source}; overtime, two a side each period, called at"
                  f" {tables.ot_call_scale:.2f}x the fourth quarter's rates")
            p = tables.call_p
            print("  timeouts, share of running clocks a side stops before the snap in each 40-second"
                  " slice (4:00 -> 0:00), level score: Q2 with the ball " + " ".join(f"{100 * x:.0f}" for x in p[0, 0, 2])
                  + " / without " + " ".join(f"{100 * x:.0f}" for x in p[0, 1, 2])
                  + "; Q4 trailing 1-8 with the ball " + " ".join(f"{100 * x:.0f}" for x in p[1, 0, 1])
                  + " / without " + " ".join(f"{100 * x:.0f}" for x in p[1, 1, 1]) + " %")
        else:
            print("  timeouts: off -- the leader kneels out as v7 did")
    if verbose:
        print(f"  {tables.n_snaps:,} snaps in the tables; points by quarter real "
              + " / ".join(f"{x:.2f}" for x in real) + ", simulated from kickoff "
              + " / ".join(f"{x:.2f}" for x in got))
    if QUARTER_START_FIT:
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
    if verbose and sim.FOURTH_JOINT:
        go_sd = np.std(list(tables.player_go.values())) if tables.player_go else 0.0
        kick_sd = np.std(list(tables.player_kick.values())) if tables.player_kick else 0.0
        print(f"  4th downs: the league go curve, the part-of-game shifts and {len(tables.player_go):,}"
              f" players' own go shifts fitted together (sd {go_sd:.2f} in log odds); kick rather than"
              f" punt the same way ({len(tables.player_kick):,} players, sd {kick_sd:.2f})")
    if verbose and sim.KNEELS and tables.kneel_p.any():
        print("  kneels, leader on 1st down by 40-second slice of Q4 (one score / more): "
              + " ".join(f"{100 * a:.0f}/{100 * b:.0f}" for a, b in zip(tables.kneel_p[0, 0], tables.kneel_p[1, 0]))
              + f"%; a kneel takes {np.median(tables.kneel_secs):.0f}s when nothing stops the clock")
    if verbose and tables.backed is not None:
        print("  backed up, per snap on the own 1 / 2 / 3 / 4 / 5: "
              + "; ".join(f"{name.replace('_', ' ')} "
                          + " / ".join(f"{100 * tables.backed[y, o]:.1f}" for y in range(1, 6))
                          for o, name in enumerate(sim.BACKED_OUTCOMES)) + " %")
    lap("quarter levels")
    form = {}
    if pre is not None:
        var_fn, cov_fn, slope = fixed_strength_spread(tables)
        tables.strength_theta, tables.strength_slope = FORM_GRID.copy(), slope
        cutoff = dt.datetime.fromisoformat(pre.meta["before"])
        finished = [r for r in history if r.get("PLAYER_1_FINAL_SCORE") not in ("", None)
                    and nb2_prior._start(r) is not None and nb2_prior._start(r) < cutoff
                    and nb2_prior._start(r) >= cutoff - dt.timedelta(days=FORM_DAYS)]
        sides = form_sides(history, pre.means(finished, n_sims=1000, results=history), cutoff)
        if len(sides) >= 200:
            tables.strength_game, tables.strength_league, form = fit_form(sides, var_fn, cov_fn)
            if verbose:
                sds = sorted(np.sqrt(list(form.values())))
                print(f"  form on the day, sd in log points: the game {np.sqrt(tables.strength_game):.3f}"
                      f" (both sides together), each player's own {sds[0]:.3f} to {sds[-1]:.3f}"
                      f" ({len(form)} players; {np.sqrt(tables.strength_league):.3f} for one"
                      f" with no history)")
    lap("form")
    tables_path = os.path.join(out_dir, "v14tables.npz")
    grid_path = os.path.join(out_dir, "v14grid.npz")
    grid = PriorGrid.build(tables, n_paths=grid_paths)
    lap("prior grid")
    big_path = os.path.join(out_dir, BIG_FILE)
    if os.path.exists(big_path):
        os.remove(big_path)
    ratings = None
    if sim.BIG_PLAYS and tables.n_big_stop is not None and handles:
        ratings = fit_big_ratings(big_records(matches, tables, handles, match_teams(history), as_of=as_of))
        grid.big_home, grid.big_away = fit_big_response(tables, n_paths=max(500, grid_paths * 2 // 3))
        ratings["elasticity"] = round(fit_big_elasticity(tables, grid, n_paths=max(400, grid_paths // 3)), 4)
        if sim.LOSS_PLAYS and tables.n_loss_stop is not None:
            loss = fit_big_ratings(big_records(matches, tables, handles, match_teams(history), as_of=as_of,
                                               kind="loss"))
            grid.loss_home, grid.loss_away = fit_big_response(tables, n_paths=max(500, grid_paths * 2 // 3),
                                                              attr="loss")
            loss["elasticity"] = round(fit_big_elasticity(tables, grid, n_paths=max(400, grid_paths // 3),
                                                          attr="loss"), 4)
            ratings["loss"] = loss
            ratings["loss_size"] = fit_loss_sizes(matches, handles)
            if verbose:
                _print_loss(loss, ratings["loss_size"], grid)
        with open(big_path, "w", encoding="utf-8") as fh:
            json.dump(ratings, fh, indent=1)
        if verbose:
            _print_big(ratings, grid)
    lap("big-play and loss ratings and responses")
    if PACE_NEUTRAL:
        grid.pace_home, grid.pace_away = fit_pace_response(tables, n_paths=max(500, grid_paths * 2 // 3))
        if verbose:
            n = len(PACE_POINTS)
            print("  pace response, home points at kickoff against both at 1 (home pace "
                  + " / ".join(f"{p:.3f}" for p in PACE_POINTS) + ", away at 1): "
                  + " / ".join(f"{grid.pace_home[i, n // 2]:.3f}" for i in range(n))
                  + "; at the away pace: " + " / ".join(f"{grid.pace_home[n // 2, j]:.3f}" for j in range(n)))
    lap("pace response")
    grid.save(grid_path)
    if IN_PLAY_FIT:
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
    lap("in-play fit and in-game check")
    if pre is not None:
        if verbose and prior == "glmer":
            print(f"  pre-match: {pre.describe()}")
        elif verbose:
            m = pre.meta
            ratio = f"{m['scale_raw_ratio']:.3f}" if m["scale_raw_ratio"] else "n/a"
            print(f"  pre-match: NB2 fitted on {m['fitted_on']:,} matches before {before:%Y-%m-%d};"
                  f" its totals ran {ratio} x under the last {nb2_prior.SCALE_DAYS} days'"
                  f" ({m['scale_matches']} matches) -> expected points scaled {m['scale']:.3f}")
    if handles or form:
        book = players.build(matches, handles) if handles else players.Book()
        if sim.FOURTH_JOINT:
            for h, prof in book.players.items():
                prof.aggression = tables.player_go.get(h, 0.0)
                prof.kick = tables.player_kick.get(h, 0.0)
            for h in set(tables.player_go) - set(book.players):
                book.players[h] = players.Profile(aggression=tables.player_go[h],
                                                  kick=tables.player_kick.get(h, 0.0))
        for h, f in form.items():
            book.players.setdefault(h, players.Profile()).form = f
        if sim.KICK_STYLES and tables.kick_mix is not None:
            styles = fit_kick_styles(matches, tables, handles)
            for h, (tb, nlz) in styles.items():
                prof = book.players.setdefault(h, players.Profile())
                prof.kick_tb, prof.kick_nlz = tb, nlz
            if verbose and styles:
                mix = tables.kick_mix
                top = lambda i: ", ".join(f"{h} x{v[i]:.1f}" for h, v in sorted(styles.items(), key=lambda kv: -kv[1][i])[:4])
                print(f"  kick-offs: touchback {100 * mix[0, 0]:.1f}% / no landing zone {100 * mix[0, 1]:.1f}% of"
                      f" kicks, {100 * mix[1, 0]:.1f}% / {100 * mix[1, 1]:.1f}% in Q2's and Q4's last"
                      f" {sim.KICK_LATE:.0f}s; {len(styles)} kickers, touchbacks most {top(0)}; no landing zone"
                      f" most {top(1)}")
        book.save(os.path.join(out_dir, "v14players.json"))
        if verbose:
            print(f"  player profiles: {len(book.players):,} players from {len(handles):,} matches")
    elif verbose:
        print("  player profiles: no handles in the export, none built")
    lap("player profiles and kick-offs")
    if RECONCILE and priors and handles:
        grid = reconcile(tables, grid, matches, priors, book, ratings, handles, match_teams(history),
                         grid_paths, verbose=verbose)
        grid.save(grid_path)
        tables.save(tables_path)
        if verbose:
            sim_q, real_q = in_game_check(copy.deepcopy(tables), grid, matches, priors=priors)
            print("  in-game check after the reconcile (league rates): real "
                  + " / ".join(f"{real_q[q]:.2f}" for q in sorted(real_q)) + ", simulated "
                  + " / ".join(f"{sim_q[q]:.2f}" for q in sorted(sim_q)))
    lap("reconcile")
    return tables_path, grid_path


def _print_loss(loss, sizes, grid):
    """The build's lines on the lost-yardage rates and loss sizes."""
    sp = loss["spread"]
    teams = sorted(loss.get("team", {}).items(), key=lambda kv: -kv[1])
    print(f"  losses, on {loss['snaps']:,} snaps: effects spread (sd, log odds) gamer {sp.get('gamer', 0):.3f},"
          f" team {sp.get('team', 0):.3f}, what a gamer allows {sp.get('gamer_allows', 0):.3f}, what a team"
          f" allows {sp.get('team_allows', 0):.3f}; a rating is played at its odds to the power"
          f" 1 / {loss.get('elasticity', 1.0):.2f}")
    if teams:
        print("    teams with the ball: " + ", ".join(f"{t} {x:+.2f}" for t, x in teams))
    if sizes:
        s = sorted(sizes.items(), key=lambda kv: kv[1])
        print(f"  loss size, {len(sizes)} gamers: shallowest " + ", ".join(f"{g} {v:.2f}" for g, v in s[:3])
              + "; deepest " + ", ".join(f"{g} {v:.2f}" for g, v in s[-3:]))
    n = len(BIG_POINTS)
    print("  lost-yardage response, home points at kickoff against both at 1 (home rate "
          + " / ".join(f"{p:.2f}" for p in BIG_POINTS) + "): "
          + " / ".join(f"{grid.loss_home[i, n // 2]:.3f}" for i in range(n)))


def _print_big(ratings, grid):
    """The build's lines on the big-play ratings and their points response."""
    sp = ratings["spread"]
    print(f"  big plays ({sim.BIG_GAIN}+ yards), on {ratings['snaps']:,} snaps: effects spread (sd, log odds)"
          f" gamer {sp.get('gamer', 0):.3f}, team {sp.get('team', 0):.3f}, what a gamer allows"
          f" {sp.get('gamer_allows', 0):.3f}, what a team allows {sp.get('team_allows', 0):.3f}")
    teams = ratings.get("team", {})
    if teams:
        print("    teams with the ball: " + ", ".join(
            f"{t} {x:+.2f}" for t, x in sorted(teams.items(), key=lambda kv: -kv[1])))
    n = len(BIG_POINTS)
    print(f"  big plays a snap move {ratings.get('elasticity', 1.0):.2f} of the rate once the strengths hold"
          " the points: a rating is played at its odds to the power 1 / that")
    print("  big-play response, home points at kickoff against both at 1 (home rate "
          + " / ".join(f"{p:.2f}" for p in BIG_POINTS) + ", away at 1): "
          + " / ".join(f"{grid.big_home[i, n // 2]:.3f}" for i in range(n))
          + "; at the away rate: " + " / ".join(f"{grid.big_home[n // 2, j]:.3f}" for j in range(n)))


def prematch_model(model_dir_or_tables):
    """The model's pre-match model -- NB2 or glmer, whichever it was built with -- or None."""
    import os
    d = model_dir_or_tables if os.path.isdir(model_dir_or_tables) else os.path.dirname(model_dir_or_tables)
    kind, meta = "nb2", {}           # nb2: a model built before the choice existed
    if os.path.exists(os.path.join(d, PRIOR_FILE)):
        with open(os.path.join(d, PRIOR_FILE), encoding="utf-8") as fh:
            meta = json.load(fh)
        kind = meta.get("prior", "nb2")
    path = os.path.join(d, kind)
    if meta.get("rolling"):          # refitted every day (rolling_prior; `eAMFModel prior-daily`)
        from . import rolling_prior
        pre = rolling_prior.Rolling(meta["rolling"])
    elif not PRIORS[kind].exists(path):
        return None
    else:
        pre = PRIORS[kind](path)
    shrink_path = os.path.join(d, SHRINK_FILE)
    if PRIOR_SHRINK and os.path.exists(shrink_path):
        with open(shrink_path, encoding="utf-8") as fh:
            pre = ShrunkPrematch(pre, json.load(fh))
    # each gamer's results since the fit, on top (follow.py); a rolling prior refits every day instead
    pre = follow.wrap(pre, rolling=bool(meta.get("rolling")))
    # and the post-game form layer, where the build has it switched on (form_layer.py)
    return form_layer.wrap(pre, meta)


def players_book(model_dir_or_tables):
    """The model's player profiles, or None."""
    import os
    d = model_dir_or_tables if os.path.isdir(model_dir_or_tables) else os.path.dirname(model_dir_or_tables)
    path = os.path.join(d, "v14players.json")
    return players.Book.load(path) if os.path.exists(path) else None


def run(path, tables_path, grid_path, variants, matches=None, n_paths=1000, workers=4,
        seed=0, book=None, handles=None, require_live=True, priors=None, history=None):
    """Price an export with v14 and grade it against the results."""
    if book is None:
        book = players_book(tables_path)
    by_match = playover.load(path)
    pre = prematch_model(tables_path) if priors is None else None
    if priors is None and history is not None:
        if pre is None:
            raise SystemExit("this v14 model has no pre-match model: rebuild it with --history")
        wanted = set(by_match) if matches is None else set(matches)
        priors = pre.means([r for r in history if r["MATCH_CODE"] in wanted], results=history)
    elif priors is None and pre is not None:
        raise SystemExit("this v14 model prices pre-match with its own model (NB2 or glmer): pass"
                         " --history (the matches' players, teams and streams, e.g. eAMFCalibrator"
                         " history's CSV)")
    codes = sorted(by_match) if matches is None else [c for c in matches if c in by_match]
    items = [(c, by_match[c]) for c in codes]
    chunks = [items[i::workers] for i in range(workers)]
    teams = match_teams(history)
    jobs = [(chunk, tables_path, grid_path, variants, n_paths, seed + i, book, handles, require_live,
             priors, teams) for i, chunk in enumerate(chunks) if chunk]
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
