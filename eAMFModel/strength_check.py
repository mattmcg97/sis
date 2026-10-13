"""How a side's strength plays out, snap by snap, in a v14 build: the sim against the real games,
and the day's strength draw against fixed strengths.

    python -m eAMFModel strength-check --model v14_model --since 2026-10-01 --until 2026-10-12

A side's strength (theta) reaches its snaps two ways (sim14.simulate): the tilt, exp(tilt x theta),
which leans every draw toward the better end of its bin (fewer turnovers and losses, longer gains,
all at once); and the boom, exp(boom x theta) times its bin's 40+ yard plays, the same share more
of its failed plays, out of the plays between. A build fitted with v14.STRENGTH_FIT carries its own
boom and tilt (Tables.boom, Tables.tilt_scale); an older one plays sim14.BOOM and TILT_SCALE. The build's prior grid finds the strengths that
give each side its expected points. Each simulated game also draws the day's strengths around them
(sim14.strength_draw: one swing both sides share, and each player's own form). Three parts:

  A  from kick-off: one side's expected points stepped, the other at SWEEP_OTHER, league profiles,
     no draw. Its points and how its snaps split (SNAP_CLASSES), as built, with the tilt alone
     (the boom off) and the boom alone (the tilt off): which kinds of play the strength moves.
  B  the window's settled matches, each priced as the stream prices its kick-off (the held
     strengths, the players' profiles, the day's draw): every side's snaps against its real ones,
     grouped by the side's expected points and by its expected margin.
  C  the same matches with fixed strengths -- the hold run without the draw, so the prior's total
     is still met -- against the draw: the mean points against the prior's, the spread, and the
     moneyline, margin and total graded on the results (prior_check.grade).

Snaps are scrimmage plays in regulation, kneels left out. A sim snap is the real play its draw
landed on, its gain cut at the goal line; the touchdown run-on and the backed-up rules that can
change it afterwards are not applied, so the shares are what each side draws.
"""

import copy
import csv
import math
import os

import numpy as np

from . import nb2_prior, playover, prior_check, sim14 as sim, v14, v14_stream

SNAP_CLASSES = sim.SNAP_CLASSES
N_CLASSES = sim.N_CLASSES
SWEEP_POINTS = (10.0, 13.0, 16.0, 19.0, 22.0, 25.0)
SWEEP_OTHER = 16.0
SWEEP_PATHS = 4000
CHANNELS = (("as built", {}), ("tilt alone", {"boom": 0.0}), ("boom alone", {"tilt_scale": 0.0}))
POINT_BINS = (13.0, 15.0, 17.0, 19.0, 21.0)
MARGIN_BINS = (-7.0, -3.5, -1.0, 1.0, 3.5, 7.0)
DEFAULT_PATHS = 2000
METHODS = ("draw", "fixed")


snap_class = sim.snap_class
real_counts = sim.real_snap_counts


def sim_counts(log):
    """Counts by side and class, summed over every path, off a simulation's per-snap log."""
    return sim.sim_snap_counts(log)


def fixed_tables(tables):
    """The tables with the day's strength draw switched off (sim14.strength_draw keeps the strength
    as given when the tables carry no strength slope)."""
    out = copy.copy(tables)
    out.strength_slope = np.zeros(0)
    out.strength_theta = np.zeros(0)
    return out


def _kickoff_start(tables, theta0, prof, variant):
    """The two kick-off states pricing plays (each side receiving), as v14.price_kickoff sets them."""
    start = sim.Start(2)
    start.team[:] = (0, 1)
    start.kicks_second_half[:] = (1, 0)
    forms = [tables.strength_league if p.form is None else p.form for p in prof]
    for i in range(2):
        start.theta[i], start.strength[i], start.strength_game[i] = sim.strength_draw(tables, theta0, forms)
        start.aggression[i] = (prof[0].aggression, prof[1].aggression)
        start.kick[i] = (prof[0].kick, prof[1].kick)
        if variant.pace:
            start.pace[i] = (prof[0].pace, prof[1].pace)
        for name in ("big", "kick_tb", "kick_nlz", "loss", "loss_size"):
            getattr(start, name)[i] = (getattr(prof[0], name), getattr(prof[1], name))
    return start


def price_match(tables, grid, variant, code, prof, means, n_paths):
    """One match's kick-off as pricing plays it: {held strengths, margin and total books, each side's
    mean points, snap counts by side and class, games}."""
    seed = v14.match_seed(code)
    theta0 = v14.held_theta(tables, grid, means, prof, variant, seed=seed)
    start = _kickoff_start(tables, theta0, prof, variant)
    half = max(1, n_paths // 2)
    stats = {"_snaps": []}
    home, away = sim.simulate(tables, start, half, np.random.default_rng(seed), seed=seed, stats=stats)
    margin, total = v14._distributions(home.ravel(), away.ravel())
    out = dict(theta=tuple(float(t) for t in theta0), points=(float(home.mean()), float(away.mean())),
               counts=sim_counts(stats["_snaps"]), games=2 * half)
    if np.any(tables.inplay_theta):
        home2, away2 = sim.simulate(tables, start, half, np.random.default_rng(seed), seed=seed, in_play=True)
        total = v14._distributions(home2.ravel(), away2.ravel())[1]
        out["total_points"] = float(home2.mean() + away2.mean())
    else:
        out["total_points"] = out["points"][0] + out["points"][1]
    out["book"] = (margin, total)
    return out


def _match_worker(job):
    model_dir, items, n_paths = job
    tables_path, grid_path = v14_stream.model_paths(model_dir)
    tables = sim.Tables.load(tables_path)
    grid = v14.PriorGrid.load(grid_path)
    variant = v14.Variant("v14")
    fixed = fixed_tables(tables)
    return {code: {"draw": price_match(tables, grid, variant, code, prof, means, n_paths),
                   "fixed": price_match(fixed, grid, variant, code, prof, means, n_paths)}
            for code, prof, means in items}


def _sweep_worker(job):
    model_dir, theta, channel, n_paths, seed = job
    tables = sim.Tables.load(v14_stream.model_paths(model_dir)[0])
    for k, v in channel.items():                      # the build's own boom or tilt switched off
        setattr(tables, k, v)
    start = sim.Start(2)
    start.team[:] = (0, 1)
    start.kicks_second_half[:] = (1, 0)
    start.theta[:] = theta
    stats = {"_snaps": []}
    home, away = sim.simulate(tables, start, n_paths // 2, np.random.default_rng(seed), seed=seed, stats=stats)
    return dict(points=(float(home.mean()), float(away.mean())), counts=sim_counts(stats["_snaps"]),
                games=2 * (n_paths // 2))


def sweep(model_dir, n_paths=SWEEP_PATHS, workers=1, seed=0):
    """Part A: [(expected points, channel, theta, result)], the other side at SWEEP_OTHER."""
    grid = v14.PriorGrid.load(v14_stream.model_paths(model_dir)[1])
    jobs, keys = [], []
    for pts in SWEEP_POINTS:
        theta = v14.fit_means(grid, pts, SWEEP_OTHER)
        for name, channel in CHANNELS:
            jobs.append((model_dir, theta, channel, n_paths, seed))
            keys.append((pts, name, theta))
    got = _map(_sweep_worker, jobs, workers)
    return [k + (g,) for k, g in zip(keys, got)]


def _map(fn, jobs, workers):
    workers = max(1, min(workers, len(jobs)))
    if workers == 1:
        return [fn(j) for j in jobs]
    with v14.pool_context().Pool(workers) as pool:
        return pool.map(fn, jobs)


def run(model_dir, history, snapshots, since, until, n_paths=DEFAULT_PATHS, workers=1):
    """Parts B and C: per settled match in the window with real snaps, its row, finals, the prior's
    means, the real snap counts and the two pricings ("draw", "fixed")."""
    tables_path, _ = v14_stream.model_paths(model_dir)
    pre = v14.prematch_model(tables_path)
    if pre is None:
        raise SystemExit(f"{model_dir}: no pre-match model to check")
    rows = sorted((r for r in history if prior_check._finals(r) is not None and prior_check._start(r) is not None
                   and since <= prior_check._start(r).date() <= until and r["MATCH_CODE"] in snapshots),
                  key=prior_check._start)
    if not rows:
        raise SystemExit("no settled matches with snapshots in that window")
    priced = pre.means(rows, results=history)
    prof = prior_check.profiles(model_dir, rows)
    items = [(r["MATCH_CODE"], prof[r["MATCH_CODE"]], priced[r["MATCH_CODE"]]) for r in rows
             if r["MATCH_CODE"] in priced]
    workers = max(1, min(workers, len(items)))
    parts = _map(_match_worker, [(model_dir, items[k::workers], n_paths) for k in range(workers)], workers)
    sims = {code: got for part in parts for code, got in part.items()}
    out = []
    for r in rows:
        code = r["MATCH_CODE"]
        if code not in sims:
            continue
        out.append(dict(row=r, finals=prior_check._finals(r), means=tuple(priced[code]),
                        real=real_counts(snapshots[code]), sims=sims[code],
                        books={m: sims[code][m]["book"] for m in METHODS}))
    return out


# ---------------------------------------------------------------- the report

def _shares(counts):
    n = counts.sum()
    return 100 * counts / n if n else np.full(N_CLASSES, np.nan)


W = 9


def _head(first):
    return f"  {first}" + "".join(f"{c:>{W}s}" for c in ("points", "snaps") + SNAP_CLASSES)


def _row(label, points, counts, games):
    sh = _shares(counts)
    return (f"  {label}{points:>{W}.1f}{counts.sum() / max(games, 1):>{W}.1f}"
            + "".join(f"{v:>{W}.1f}" for v in sh))


def _diff(label, diff):
    return f"  {label}{'':{2 * W}s}" + "".join(f"{v:>+{W}.1f}" for v in diff)


def sweep_lines(rows):
    lines = ["", f"  A. FROM KICK-OFF: one side's expected points stepped, the other at {SWEEP_OTHER:g}"
                 " (league profiles, no strength draw)",
             "     its points and snaps a game, and how its snaps split (% of its snaps); the tilt alone"
             " has the boom off, the boom alone the tilt off, at the same strengths",
             _head(f"{'expected':>9s} {'strength':>9s} {'':12s}")]
    for pts, name, theta, g in rows:
        first = f"{pts:>9.0f} {theta[0]:>+9.2f} " if name == CHANNELS[0][0] else " " * 20
        lines.append(_row(f"{first}{name:12s}", g["points"][0], g["counts"][0], g["games"]))
    return lines


def _side_rows(results):
    """[(expected points, expected margin, real points, real counts, {method: (points, counts, games)})]
    for every side of every match."""
    out = []
    for x in results:
        for s in (0, 1):
            exp, opp = x["means"][s], x["means"][1 - s]
            sims = {m: (x["sims"][m]["points"][s], x["sims"][m]["counts"][s], x["sims"][m]["games"])
                    for m in METHODS}
            out.append((exp, exp - opp, x["finals"][s], x["real"][s], sims))
    return out


def _band(v, edges):
    return int(np.searchsorted(edges, v, side="right"))


def _band_name(i, edges, unit):
    if i == 0:
        return f"under {edges[0]:+g}" if unit == "margin" else f"under {edges[0]:g}"
    if i == len(edges):
        return f"{edges[-1]:+g} up" if unit == "margin" else f"{edges[-1]:g} up"
    a, b = edges[i - 1], edges[i]
    return f"{a:+g} to {b:+g}" if unit == "margin" else f"{a:g}-{b:g}"


def grouped_lines(results, by="points"):
    edges, col = (POINT_BINS, 0) if by == "points" else (MARGIN_BINS, 1)
    sides = _side_rows(results)
    title = ("B. EVERY SIDE BY ITS EXPECTED POINTS" if by == "points" else
             "B. EVERY SIDE BY ITS EXPECTED MARGIN (its expected points less the other side's)")
    lines = ["", f"  {title}: real against the sim as priced (the day's draw) and with fixed strengths",
             _head(f"{'band':>13s} {'sides':>6s} {'prior':>6s} {'':6s}")]
    for i in range(len(edges) + 1):
        group = [s for s in sides if _band(s[col], edges) == i]
        if not group:
            continue
        prior = np.mean([s[0] for s in group])
        real_c = sum(s[3] for s in group)
        lines.append(_row(f"{_band_name(i, edges, by):>13s} {len(group):>6d} {prior:>6.1f} {'real':>6s}",
                          float(np.mean([s[2] for s in group])), real_c, len(group)))
        for m in METHODS:
            pts = float(np.mean([s[4][m][0] for s in group]))
            cnt = sum(s[4][m][1] for s in group)
            games = float(np.mean([s[4][m][2] for s in group]))
            lines.append(_row(f"{'':13s} {'':6s} {'':6s} {m:>6s}", pts, cnt / games, len(group)))
        sim_sh = _shares(sum(s[4]["draw"][1] for s in group))
        lines.append(_diff(f"{'':13s} {'':6s} {'draw-real':>13s}", sim_sh - _shares(real_c)))
    return lines


def draw_lines(results):
    summary, _ = prior_check.grade(results, methods=METHODS)
    n = len(results)
    mean = lambda f: float(np.mean([f(x) for x in results]))
    lines = ["", f"  C. THE DAY'S STRENGTH DRAW AGAINST FIXED STRENGTHS ({n:,} matches, each priced as of its kick-off;"
                 " fixed: the hold run without the draw)",
             f"  {'':34s}{'draw':>12s}{'fixed':>12s}"]
    for label, f in (("total less the prior's", lambda x, m: x["sims"][m]["total_points"] - sum(x["means"])),
                     ("margin less the prior's", lambda x, m: (x["sims"][m]["points"][0] - x["sims"][m]["points"][1])
                      - (x["means"][0] - x["means"][1]))):
        lines.append(f"  {label:34s}" + "".join(f"{mean(lambda x: f(x, m)):>+12.2f}" for m in METHODS))
    for name, offset, real in (("margin", v14.MARGIN_MAX, lambda x: x["finals"][0] - x["finals"][1]),
                               ("total", 0, lambda x: x["finals"][0] + x["finals"][1])):
        sd = {}
        for m in METHODS:
            vals = []
            for x in results:
                pmf = x["books"][m][0 if name == "margin" else 1]
                k = np.arange(len(pmf)) - offset
                mu = (k * pmf).sum()
                vals.append(((k - mu) ** 2 * pmf).sum())
            sd[m] = math.sqrt(float(np.mean(vals)))
        expected = (lambda x: x["means"][0] - x["means"][1]) if name == "margin" else (lambda x: sum(x["means"]))
        rms = math.sqrt(mean(lambda x: (real(x) - expected(x)) ** 2))
        lines.append(f"  {name + ' sd in the sim':34s}" + "".join(f"{sd[m]:>12.2f}" for m in METHODS)
                     + f"   (real: {rms:.2f} RMS of the result less the prior's)")
    lines.append(f"  {'moneyline log loss':34s}" + "".join(f"{summary[m]['log_loss']:>12.4f}" for m in METHODS))
    lines.append(f"  {'moneyline Brier':34s}" + "".join(f"{summary[m]['brier']:>12.4f}" for m in METHODS))
    for name in ("margin", "total"):
        for key, label, fmt in (("bias", "result less expected", "{:+.2f}"), ("rps", "RPS (lower better)", "{:.3f}"),
                                ("in50", "inside central 50%", "{:.1%}"), ("in80", "inside central 80%", "{:.1%}")):
            lines.append(f"  {name + ' ' + label:34s}" + "".join(f"{fmt.format(summary[m][name][key]):>12s}"
                                                                for m in METHODS))
    d, se = prior_check.paired(summary, "fixed", "draw")
    lines.append(f"  {'log loss, fixed less draw':34s}{d:>+12.4f} +/- {se:.4f} (paired standard error)")
    return lines


def report(sweep_rows, results, since, until):
    lines = [f"  STRENGTH CHECK -- {since} to {until}"]
    if sweep_rows:
        lines += sweep_lines(sweep_rows)
    if results:
        lines += grouped_lines(results, "points") + grouped_lines(results, "margin") + draw_lines(results)
    lines += ["", "  snaps: scrimmage plays in regulation, kneels out; short: a gain short of the line; 1st <20:"
                  " a first down under 20 yards; a sim snap is the play its draw landed on, cut at the goal line"]
    return lines


def write_csv(path, results):
    """One row per side of every match: the prior, the real and simulated points, and the snap
    counts by class (the sim's a game)."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "side", "expected", "expected_margin", "real_points"]
                   + [f"{m}_points" for m in METHODS] + [f"{m}_theta" for m in METHODS]
                   + [f"real_{c}" for c in SNAP_CLASSES]
                   + [f"{m}_{c}" for m in METHODS for c in SNAP_CLASSES])
        for x in results:
            for s in (0, 1):
                sims = x["sims"]
                w.writerow([x["row"]["MATCH_CODE"], "home" if s == 0 else "away", round(x["means"][s], 3),
                            round(x["means"][s] - x["means"][1 - s], 3), x["finals"][s]]
                           + [round(sims[m]["points"][s], 3) for m in METHODS]
                           + [round(sims[m]["theta"][s], 4) for m in METHODS]
                           + [int(v) for v in x["real"][s]]
                           + [round(v / sims[m]["games"], 3) for m in METHODS for v in sims[m]["counts"][s]])


def window_codes(history, since, until):
    """The settled matches' codes in the window."""
    return {r["MATCH_CODE"] for r in history if prior_check._finals(r) is not None
            and prior_check._start(r) is not None and since <= prior_check._start(r).date() <= until}


def load_snapshots(path, codes=None):
    """The export's rows by match in message order (playover.load), keeping only the `codes` given
    as it reads."""
    if codes is None:
        return playover.load(path)
    by_match = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["match_code"] in codes:
                by_match.setdefault(r["match_code"], []).append(r)
    for rows in by_match.values():
        rows.sort(key=lambda r: playover._int(r["message"]))
    return by_match


def history_rows(path):
    return nb2_prior.load_history(path)
