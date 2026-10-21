"""How much calibration the sim keeps: the pre-match prior's own price (glmer or NB2 expected points and
its own score distribution) against the sim's kick-off price off the same prior, both graded on the
real results.

    python -m eAMFModel prior-vs-sim --model v14_model \
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
  sim     the build's kick-off price (v14_stream.kickoff_book) off
          those means, with each side's profile and big-play rates: what the stream quotes before the
          first play.

Graded on the moneyline (log loss, Brier, a calibration table by decile, pushes left out), the
margin and the total: the bias and RMSE of the expected value, the ranked probability score over the
whole distribution, and how often the result lands inside the central 50% and 80% of it.
"""

import csv

import numpy as np

from . import books, players, v14, v14_stream
# a prior's own score books and their grading, shared by every version's checks (books.py), under prior_check's names
from .books import (RAW_DRAWS, _cdf_scores, _finals, _start, base_prior, nb2_books, paired,  # noqa: F401
                    raw_books, sigmas, table_books)



def version_of(model_dir):
    """The version a build directory holds: v14, the one left that builds its own tables."""
    return "v14"


def _modules(version="v14"):
    """The version's model and stream modules."""
    return v14, v14_stream
DEFAULT_PATHS = 4000


def _sim_worker(job):
    model_dir, items, n_paths, seed = job[:4]
    m, st = _modules()
    tables_path, grid_path = st.model_paths(model_dir)
    tables = m.sim.Tables.load(tables_path)
    grid = m.PriorGrid.load(grid_path)
    variant = m.Variant(job[4] if len(job) > 4 else "v14")
    rng = np.random.default_rng(seed)
    return {code: st.kickoff_book(tables, grid, variant, code, n_paths, rng, prof, means)
            for code, prof, means in items}


def profiles(model_dir, rows):
    """{match: (home profile, away profile)} with big-play rates (and v14's lost-yardage rates and
    loss sizes), as the build's stream prices them."""
    m_, st = _modules()
    tables_path, _ = st.model_paths(model_dir)
    book = m_.players_book(tables_path)
    ratings = m_.big_ratings(tables_path)
    teams = m_.match_teams(rows)
    out = {}
    for r in rows:
        pair = (r["PLAYER_1_HANDLE"].strip().upper(), r["PLAYER_2_HANDLE"].strip().upper())
        prof = (book.profile(pair[0]), book.profile(pair[1])) if book else \
            (players.Profile(), players.Profile())
        if ratings:
            prof = m_.with_big(prof, ratings, pair, *m_.side_teams(pair, teams.get(r["MATCH_CODE"])))
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


def grade(results, buckets=10, methods=None):
    """books.grade, for `methods` (default METHODS)."""
    return books.grade(results, buckets, methods or METHODS)


LABELS = {"raw": "glmer raw", "raw+": "glmer + layers", "sim": "v14 sim"}


def report(results, since, until, version="v14"):
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
    for a, b in (("sim", "raw"), ("sim", "raw+"), ("raw+", "raw")):
        d, se = paired(summary, a, b)
        lines.append(f"  {'log loss, ' + LABELS[a] + ' less ' + LABELS[b]:42s}{d:>+9.4f} +/- {se:.4f}"
                     " (paired standard error)")
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
        mid = v14.MARGIN_MAX
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
