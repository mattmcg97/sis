"""v5: prices moneyline, spread and total from inside a game by simulating the rest of it play by
play."""

import datetime as dt
import math
import multiprocessing as mp
from collections import Counter

import numpy as np

from . import nb2_prior, playover, players, sim5 as sim
from .pricer import HOME

GRID = np.round(np.linspace(-0.8, 0.8, 17), 3)
MARGIN_MAX = 100
TOTAL_MAX = 160
KAPPA = 40.0


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


class PriorGrid:
    """Simulated games from kickoff on a grid of the two offenses' strengths."""

    def __init__(self, margin, total, grid=GRID):
        """Hold the margin and total distributions for each grid point."""
        self.margin, self.total, self.grid = margin, total, grid

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
        home, away = sim.simulate(tables, start, n_paths, rng, theta_sd=sds, **sim_kw)
        margin = np.zeros((g, g, 2 * MARGIN_MAX + 1))
        total = np.zeros((g, g, TOTAL_MAX + 1))
        for i in range(g * g):
            a, b = divmod(i, g)
            margin[a, b], total[a, b] = _distributions(home[i], away[i])
        return cls(margin, total, grid)

    def save(self, path):
        """Write the grid to a file."""
        np.savez_compressed(path, margin=self.margin, total=self.total, grid=self.grid)

    @classmethod
    def load(cls, path):
        """Read a grid from a file."""
        z = np.load(path)
        return cls(z["margin"], z["total"], z["grid"])


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


def prior_theta(grid, means):
    """A match's starting strengths: from NB2's expected points, else league average."""
    return LEAGUE_THETA if means is None else fit_means(grid, *means)


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
                kicks_second_half=side(state.opening_receiver) if state.opening_receiver else -1)


def _fill(start, i, fields):
    """Copy start fields into row i."""
    for k, v in fields.items():
        getattr(start, k)[i] = v


class Variant:
    """One way of running v5 (what it reacts to in the game)."""

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
        if v.pace:
            start.pace[i] = (prof[0].pace, prof[1].pace)
    kw = dict(theta_sd=sds if v.theta_sd else None, seed=seed, **v.sim_kw)
    home, away = sim.simulate(tables, start, n_paths, rng, **kw)
    books = [_distributions(home[i], away[i]) for i in range(len(messages))]
    if not np.any(tables.inplay_theta):
        return books
    home, away = sim.simulate(tables, start, n_paths, rng, in_play=True, **kw)
    return [(mp_, _distributions(home[i], away[i])[1]) for i, (mp_, _) in enumerate(books)]


def pool_context():
    """The multiprocessing start method for this platform."""
    return mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")


def _grade_matches(job):
    """Worker: price a list of matches at every snapshot."""
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
            means = priors.get(code)
            if means is None:
                skipped["no_prematch_prediction"] += 1
                continue
            theta0s = {g: fit_means(grid, *means) for g, grid in grids.items()}
        else:
            theta0s = {g: LEAGUE_THETA for g in grids}
        a_home = match_rows[0]["team_a_side"] == "home"
        snaps = sim.snap_records(match_rows)
        pair = (handles.get(code) if handles else None) or handles_of(match_rows)
        prof = ((book.profile(pair[0]), book.profile(pair[1])) if (book and pair)
                else (players.Profile(), players.Profile()))
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
    """(home handle, away handle) from the export's own columns, or None."""
    r = match_rows[0] if match_rows else {}
    home, away = (r.get("home_handle") or "").strip(), (r.get("away_handle") or "").strip()
    return (home.upper(), away.upper()) if home and away else None


QUARTER_START_FIT = False
QUARTER_ROUNDS = 6
QUARTER_PATHS = 150


def quarter_start_states(matches, grid, priors=None):
    """Real states at the start of each quarter, with the points really scored in that quarter."""
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
                continue
            state, _ = playover.state_for(firsts[str(q)])
            if state is None:
                continue
            end = int(nxt["score_p1"]) + int(nxt["score_p2"]) if nxt is not None else final
            out.append((q, state, theta0, end - state.home_score - state.away_score))
    return out


def late_start_states(matches, grid, priors=None):
    """Real states at the two-minute mark of each half, with the points really scored after."""
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
                start.theta[i] = theta0
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
            state, _ = playover.state_for(r)
            if state is None or state.down is None or state.pending_conversion is not None \
                    or not 1 <= state.period <= 4:
                continue
            sign = 1 if state.offense == HOME else -1
            lead = sign * (state.home_score - state.away_score)
            out.append((state, theta0, lead, sign * final - lead))
    return out


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
        """Solve one group's pull so its leads come back as real ones do."""
        p0 = pull.copy()
        g0 = simulated(p0, sel, start)
        p1 = pull + 0.1 * which
        g1 = simulated(p1, sel, start)
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


IN_PLAY_FIT = False
REST_EVERY = 4
REST_PATHS = 60
REST_MAX_STATES = 4000
REST_MIN_STATES = 100
REST_MIN_CELL = 150
REST_BY_BAND = False
REST_SEGMENTS = tuple(range(1, sim.N_SEGMENTS))
REST_ROUNDS = 3


def rest_of_game_states(matches, grid, priors=None, every=REST_EVERY):
    """Real in-game states with the points really still to come."""
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
                start.theta[i] = theta0
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


def build(matches, out_dir, grid_paths=6000, verbose=True, handles=None, history=None,
          before=None):
    """Build the model: tables, fits, prior grid, NB2 and player profiles."""
    import copy
    import os
    os.makedirs(out_dir, exist_ok=True)
    pre = priors = None
    if history is not None:
        if before is None:
            days = [match_day(rows) for rows in matches.values()]
            last = max(d for d in days if d is not None)
            before = dt.datetime.combine(last + dt.timedelta(days=1), dt.time())
        pre = nb2_prior.Prematch.build(history, os.path.join(out_dir, "nb2"), before)
        priors = pre.means([r for r in history if r["MATCH_CODE"] in matches])
    tables = sim.Tables.build(matches)
    real = sim.quarter_points(matches)
    offsets, got = sim.fit_period_theta(tables, real)
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
    form = {}
    if pre is not None:
        var_fn, cov_fn, slope = fixed_strength_spread(tables)
        tables.strength_theta, tables.strength_slope = FORM_GRID.copy(), slope
        cutoff = dt.datetime.fromisoformat(pre.meta["before"])
        finished = [r for r in history if r.get("PLAYER_1_FINAL_SCORE") not in ("", None)
                    and nb2_prior._start(r) is not None and nb2_prior._start(r) < cutoff
                    and nb2_prior._start(r) >= cutoff - dt.timedelta(days=FORM_DAYS)]
        sides = form_sides(history, pre.means(finished, n_sims=1000), cutoff)
        if len(sides) >= 200:
            tables.strength_game, tables.strength_league, form = fit_form(sides, var_fn, cov_fn)
            if verbose:
                sds = sorted(np.sqrt(list(form.values())))
                print(f"  form on the day, sd in log points: the game {np.sqrt(tables.strength_game):.3f}"
                      f" (both sides together), each player's own {sds[0]:.3f} to {sds[-1]:.3f}"
                      f" ({len(form)} players; {np.sqrt(tables.strength_league):.3f} for one"
                      f" with no history)")
    tables_path = os.path.join(out_dir, "v5tables.npz")
    grid_path = os.path.join(out_dir, "v5grid.npz")
    grid = PriorGrid.build(tables, n_paths=grid_paths)
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
    if handles or form:
        book = players.build(matches, handles) if handles else players.Book()
        for h, f in form.items():
            book.players.setdefault(h, players.Profile()).form = f
        book.save(os.path.join(out_dir, "v5players.json"))
        if verbose:
            print(f"  player profiles: {len(book.players):,} players from {len(handles):,} matches")
    elif verbose:
        print("  player profiles: no handles in the export, none built")
    return tables_path, grid_path


def prematch_model(model_dir_or_tables):
    """The model's NB2 pre-match model, or None."""
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
    """Price an export with v5 and grade it against the results."""
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
