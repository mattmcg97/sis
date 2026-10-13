"""How much calibration the sim keeps: the pre-match prior's own price (glmer or NB2 expected points and
its own score distribution) against the sim's kick-off price off the same prior, both graded on the
real results.

    python -m eAMFModel prior-vs-sim --model v13_glmer_1001 \
        --history eAMFCalibrator/out/match_history.csv --since 2026-10-01 --until 2026-10-12

For every settled match in the window, priced as of its kick-off (the prior's form reads every result
before it):
  raw     the prior's own: glmer's expected points and its score model -- Poisson with a log-normal
          noise for each side (sd sigma_obs) and one shared by the match (sd sigma_match), as
          glmer/predict.R prices the moneyline. A prior that saves no sigmas (NB2) draws plain
          Poisson.
  raw+    the same score model off the means the build prices with: the prior shrink and any
          follow or form layer on top. Set against raw it shows what those do; set against sim,
          what the simulation alone does.
  sim     the build's kick-off price (vN_stream.kickoff_book: v13 or v14, whichever the build is) off
          those means, with each side's profile and big-play rates: what the stream quotes before the
          first play.

Graded on the moneyline (log loss, Brier, a calibration table by decile, pushes left out), the
margin and the total: the bias and RMSE of the expected value, the ranked probability score over the
whole distribution, and how often the result lands inside the central 50% and 80% of it.
"""

import csv
import math

import numpy as np

import importlib
import os

from . import nb2_prior, players, v13, v13_stream

RAW_DRAWS = 20000
VERSIONS = ("v13", "v14")


def version_of(model_dir):
    """The version a build directory holds (its <version>tables.npz), newest first; v13 when none."""
    for v in reversed(VERSIONS):
        if os.path.exists(os.path.join(str(model_dir), f"{v}tables.npz")):
            return v
    return "v13"


def _modules(version):
    """The version's model and stream modules."""
    if version == "v13":
        return v13, v13_stream
    return (importlib.import_module(f"{__package__}.{version}"),
            importlib.import_module(f"{__package__}.{version}_stream"))
DEFAULT_PATHS = 4000


def _start(r):
    return nb2_prior._start(r)


def _finals(r):
    try:
        return float(r["PLAYER_1_FINAL_SCORE"]), float(r["PLAYER_2_FINAL_SCORE"])
    except (KeyError, TypeError, ValueError):
        return None


def base_prior(pre):
    """The prior under the build's layers (shrink, follow, form): the model itself."""
    while hasattr(pre, "pre"):
        pre = pre.pre
    return pre


def sigmas(base, day):
    """(sigma_obs, sigma_match) of the fit that prices a match on `day`: a daily prior's fit for
    that day, else the single fit; (0, 0) when the fit saves none (NB2: plain Poisson)."""
    fit = base.fit_for(day) if hasattr(base, "fit_for") else base
    m = (getattr(fit, "meta", None) or {}).get("model", {}) or {}
    return float(m.get("sigma_obs") or 0.0), float(m.get("sigma_match") or 0.0)


def raw_books(means, s_obs, s_match, n=RAW_DRAWS, seed=0):
    """Margin and total distributions (v13's shapes) from expected points under glmer's score model:
    each side Poisson around exp(eta + noise), eta set so its mean is the expected points."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0.0, s_match, n) if s_match else np.zeros(n)
    sides = []
    for mu in means:
        eta = math.log(max(mu, 0.1)) - (s_obs ** 2 + s_match ** 2) / 2
        noise = rng.normal(0.0, s_obs, n) if s_obs else np.zeros(n)
        sides.append(rng.poisson(np.exp(np.clip(eta + shared + noise, -15, 15))))
    return v13._distributions(sides[0].astype(int), sides[1].astype(int))


def _sim_worker(job):
    model_dir, items, n_paths, seed = job[:4]
    m, st = _modules(job[4] if len(job) > 4 else "v13")
    tables_path, grid_path = st.model_paths(model_dir)
    tables = m.sim.Tables.load(tables_path)
    grid = m.PriorGrid.load(grid_path)
    variant = m.Variant(job[4] if len(job) > 4 else "v13")
    rng = np.random.default_rng(seed)
    return {code: st.kickoff_book(tables, grid, variant, code, n_paths, rng, prof, means)
            for code, prof, means in items}


def profiles(model_dir, rows):
    """{match: (home profile, away profile)} with big-play rates (and v14's lost-yardage rates and
    loss sizes), as the build's stream prices them."""
    v13_, st = _modules(version_of(model_dir))
    tables_path, _ = st.model_paths(model_dir)
    book = v13_.players_book(tables_path)
    ratings = v13_.big_ratings(tables_path)
    teams = v13_.match_teams(rows)
    out = {}
    for r in rows:
        pair = (r["PLAYER_1_HANDLE"].strip().upper(), r["PLAYER_2_HANDLE"].strip().upper())
        prof = (book.profile(pair[0]), book.profile(pair[1])) if book else \
            (players.Profile(), players.Profile())
        if ratings:
            prof = v13_.with_big(prof, ratings, pair, *v13_.side_teams(pair, teams.get(r["MATCH_CODE"])))
        out[r["MATCH_CODE"]] = prof
    return out


def run(model_dir, history, since, until, n_paths=DEFAULT_PATHS, workers=1, seed=0):
    """Per match: its row, finals and the three (margin, total) books, and both sets of means."""
    version = version_of(model_dir)
    m, st = _modules(version)
    tables_path, _ = st.model_paths(model_dir)
    pre = m.prematch_model(tables_path)
    if pre is None:
        raise SystemExit(f"{model_dir}: no pre-match model to check")
    base = base_prior(pre)
    rows = sorted((r for r in history if _finals(r) is not None and _start(r) is not None
                   and since <= _start(r).date() <= until), key=_start)
    if not rows:
        raise SystemExit("no settled matches in that window")
    priced = pre.means(rows, results=history)
    raw = base.means(rows, results=history)
    prof = profiles(model_dir, rows)
    items = [(r["MATCH_CODE"], prof[r["MATCH_CODE"]], priced[r["MATCH_CODE"]]) for r in rows
             if r["MATCH_CODE"] in priced]
    workers = max(1, min(workers, len(items)))
    jobs = [(model_dir, items[k::workers], n_paths, seed + k, version) for k in range(workers)]
    if workers == 1:
        parts = [_sim_worker(j) for j in jobs]
    else:
        with m.pool_context().Pool(workers) as pool:
            parts = pool.map(_sim_worker, jobs)
    sims = {code: book for part in parts for code, book in part.items()}
    out = []
    for r in rows:
        code = r["MATCH_CODE"]
        if code not in sims or code not in raw:
            continue
        s = sigmas(base, _start(r).date())
        seed_ = m.match_seed(code)
        out.append(dict(row=r, finals=_finals(r), raw_means=raw[code], means=priced[code],
                        books={"raw": raw_books(raw[code], *s, seed=seed_),
                               "raw+": raw_books(priced[code], *s, seed=seed_),
                               "sim": sims[code]}))
    return out


METHODS = ("raw", "raw+", "sim")


def _cdf_scores(pmf, offset, actual):
    """(expected value, RPS, inside the central 50%, inside the central 80%) for one distribution
    over values k - offset."""
    k = np.arange(len(pmf)) - offset
    cdf = np.cumsum(pmf)
    rps = float(((cdf - (k >= actual)) ** 2).sum())
    lo50, hi50 = k[np.searchsorted(cdf, 0.25)], k[min(len(k) - 1, np.searchsorted(cdf, 0.75))]
    lo80, hi80 = k[np.searchsorted(cdf, 0.10)], k[min(len(k) - 1, np.searchsorted(cdf, 0.90))]
    return float((k * pmf).sum()), rps, lo50 <= actual <= hi50, lo80 <= actual <= hi80


def grade(results, buckets=10):
    """{method: summary dict} and the calibration tables."""
    summary, tables = {}, {}
    for m in METHODS:
        p, won, mstats, tstats = [], [], [], []
        for x in results:
            mp_, tp = x["books"][m]
            s1, s2 = x["finals"]
            margin, total = s1 - s2, s1 + s2
            mid = v13.MARGIN_MAX
            if margin != 0:
                push = mp_[mid]
                p.append(float(mp_[mid + 1:].sum() / max(1e-12, 1 - push)))
                won.append(1.0 if margin > 0 else 0.0)
            mstats.append(_cdf_scores(mp_, mid, margin) + (margin,))
            tstats.append(_cdf_scores(tp, 0, total) + (total,))
        p, won = np.clip(np.array(p), 1e-9, 1 - 1e-9), np.array(won)
        out = dict(n=len(results), ml_n=len(p),
                   log_loss=float(-(won * np.log(p) + (1 - won) * np.log(1 - p)).mean()),
                   brier=float(((p - won) ** 2).mean()))
        for name, st in (("margin", mstats), ("total", tstats)):
            a = np.array([s[:4] + (s[4],) for s in st], dtype=float)
            err = a[:, 4] - a[:, 0]
            out[name] = dict(bias=float(err.mean()), rmse=float(math.sqrt((err ** 2).mean())),
                             rps=float(a[:, 1].mean()), in50=float(a[:, 2].mean()), in80=float(a[:, 3].mean()))
        order = np.argsort(p)
        rows = []
        for chunk in np.array_split(order, min(buckets, len(p))):
            if len(chunk):
                rows.append((float(p[chunk].mean()), float(won[chunk].mean()), len(chunk)))
        summary[m], tables[m] = out, rows
    return summary, tables


LABELS = {"raw": "glmer raw", "raw+": "glmer + layers", "sim": "v13 sim"}


def report(results, since, until, version="v13"):
    """Lines of text: the three side by side."""
    LABELS = dict(globals()["LABELS"], sim=f"{version} sim")
    summary, tables = grade(results)
    n = summary["raw"]["n"]
    lines = [f"  {n:,} settled matches {since} to {until}, each priced as of its kick-off", "",
             f"  {'':26s}" + "".join(f"{LABELS[m]:>16s}" for m in METHODS),
             f"  {'moneyline log loss':26s}" + "".join(f"{summary[m]['log_loss']:>16.4f}" for m in METHODS),
             f"  {'moneyline Brier':26s}" + "".join(f"{summary[m]['brier']:>16.4f}" for m in METHODS)]
    for name in ("margin", "total"):
        for key, label, fmt in (("bias", "result less expected", "{:+.2f}"), ("rmse", "RMSE of expected", "{:.3f}"),
                                ("rps", "RPS (lower better)", "{:.3f}"), ("in50", "inside central 50%", "{:.1%}"),
                                ("in80", "inside central 80%", "{:.1%}")):
            lines.append(f"  {name + ' ' + label:26s}" + "".join(
                f"{fmt.format(summary[m][name][key]):>16s}" for m in METHODS))
    lines += ["", "  moneyline calibration by decile of each one's own home win chance (pushes out):"]
    lines.append(f"  {'':4s}" + "".join(f"{LABELS[m]:>24s}" for m in METHODS))
    lines.append(f"  {'':4s}" + "".join(f"{'priced':>9s}{'won':>8s}{'n':>7s}" for _ in METHODS))
    for i in range(max(len(t) for t in tables.values())):
        cells = []
        for m in METHODS:
            t = tables[m]
            cells.append(f"{t[i][0]:>9.1%}{t[i][1]:>8.1%}{t[i][2]:>7d}" if i < len(t) else " " * 24)
        lines.append(f"  {i + 1:<4d}" + "".join(cells))
    lines += ["", "  (raw: the prior's own expected points and score model; glmer + layers: the same off the"
                  f" means the build prices with -- shrink, follow or form layer; {version} sim: the kick-off price"
                  " off those means)"]
    return lines


def write_csv(path, results):
    keys = ["MATCH_CODE", "start", "home", "away", "home_points", "away_points",
            "raw_home", "raw_away", "means_home", "means_away"]
    for m in METHODS:
        keys += [f"{m}_home_win", f"{m}_margin", f"{m}_total"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        mid = v13.MARGIN_MAX
        for x in results:
            r = x["row"]
            row = [r["MATCH_CODE"], r["SCHEDULED_START_TIME_UTC"][:19], r["PLAYER_1_HANDLE"], r["PLAYER_2_HANDLE"],
                   *x["finals"], *(round(v, 3) for v in x["raw_means"]), *(round(v, 3) for v in x["means"])]
            for m in METHODS:
                mp_, tp = x["books"][m]
                k = np.arange(len(mp_)) - mid
                row += [round(float(mp_[mid + 1:].sum() / max(1e-12, 1 - mp_[mid])), 4),
                        round(float((k * mp_).sum()), 3), round(float((np.arange(len(tp)) * tp).sum()), 3)]
            w.writerow(row)
