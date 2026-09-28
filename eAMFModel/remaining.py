"""A version's points still to come against what the rest of each game really made.

At every PLAY_OVER snapshot a version (v4, v5, v6) simulates the rest of the game. Its distribution
of the final total, less the points on the board, is its distribution of the points still to come.
The export's final score gives what really came. Set side by side, value by value (0, 3, 6, 7, 8,
10, 14 ...) and by quarter and game state, this is the comparison against reality -- not against
prod -- that says where the simulation puts too much or too little: a lone field goal late, a
touchdown without its kick, a second score.

  python -m eAMFModel remaining SNAPS.csv --version v6 --model v6_model --history AMFELO.csv \\
      --since 2026-09-10 --until 2026-09-22
"""

import csv
import datetime as dt
import importlib
import math
import os
from collections import defaultdict

import numpy as np

from . import playover
from .pricer import HOME

VALUES = (0, 3, 6, 7, 8, 10, 13, 14, 17, 21)
MAX_REMAINING = 60
QUARTERS = ("Q1", "Q2", "Q3", "Q4", "OT")
STATES = ("level", "1 score, leader has ball", "1 score, trailer has ball",
          "2+ scores, leader has ball", "2+ scores, trailer has ball")


def day(code):
    """The match's date off its code (AFnnnDDMMYY)."""
    s = str(code)[-6:]
    return dt.date(2000 + int(s[4:6]), int(s[2:4]), int(s[0:2]))


def quarter(period):
    return f"Q{period}" if period and period <= 4 else "OT"


def game_state(state):
    """Level (within 2), one score apart (3-8) or two or more (9+), and who has the ball."""
    margin = state.home_score - state.away_score
    if abs(margin) <= 2:
        return STATES[0]
    leader_has_it = (margin > 0) == (state.offense == HOME)
    apart = "1 score" if abs(margin) <= 8 else "2+ scores"
    return f"{apart}, {'leader' if leader_has_it else 'trailer'} has ball"


def _job(job):
    """Worker: every priceable PLAY_OVER of some matches, as (quarter, state, board, real, pmf of
    the points still to come)."""
    items, name, model_dir, n_paths, seed = job
    model = importlib.import_module(f"eAMFModel.{name}")
    stream = importlib.import_module(f"eAMFModel.{name}_stream")
    tables_path, grid_path = stream.model_paths(model_dir)
    tables = stream.sim.Tables.load(tables_path)
    grid = model.PriorGrid.load(grid_path)
    book = model.players_book(tables_path) if hasattr(model, "players_book") else None
    rng = np.random.default_rng(seed)
    out = []
    for code, snaps, means, pair in items:
        rows = model.resolve_sides([stream._as_text(r) for r in snaps])
        prof = (book.profile(pair[0]), book.profile(pair[1])) if (book and pair) else None
        books = stream.match_books(tables, grid, model.Variant(name), rows, n_paths, rng, prof, means)
        by_msg = {int(r["message"]): r for r in rows}
        for msg, _, tpmf in books:
            r = by_msg[msg]
            state, _ = playover.state_for(r)
            try:
                final = int(float(r["final_p1"])) + int(float(r["final_p2"]))
            except (TypeError, ValueError):
                continue
            board = state.home_score + state.away_score
            real = final - board
            if real < 0:
                continue
            rem = np.zeros(MAX_REMAINING + 1)
            tail = np.asarray(tpmf[board:], dtype=float)
            rem[:min(len(tail), MAX_REMAINING)] = tail[:MAX_REMAINING]
            rem[MAX_REMAINING] = tail[MAX_REMAINING:].sum() if len(tail) > MAX_REMAINING else 0.0
            s = rem.sum()
            if s <= 0:
                continue
            out.append((code, msg, quarter(state.period), game_state(state), board, real, rem / s))
    return out


def price(snapshots_path, name, model_dir, since=None, until=None, n_paths=500, workers=4,
          history=None, handles=None, limit=None):
    """[(match, message, quarter, state, points on board, points still to come, the version's pmf
    of them)] over the matches in [since, until] whose TEAM_A side is known."""
    model = importlib.import_module(f"eAMFModel.{name}")
    stream = importlib.import_module(f"eAMFModel.{name}_stream")
    tables_path, _ = stream.model_paths(model_dir)
    by_match = playover.load(snapshots_path)
    codes = [c for c in sorted(by_match)
             if (since is None or day(c) >= since) and (until is None or day(c) <= until)
             and all(str(r.get("team_a_side") or "") in ("home", "away") for r in by_match[c])]
    if limit:
        codes = codes[:limit]
    pre = model.prematch_model(tables_path) if hasattr(model, "prematch_model") else None
    means = {}
    if pre is not None:
        if history is None:
            raise SystemExit(f"this {name} model prices pre-match with NB2: pass --history")
        means = pre.means([r for r in history if r["MATCH_CODE"] in set(codes)])
    items = []
    for c in codes:
        pair = (handles or {}).get(c) or (model.handles_of(by_match[c]) if hasattr(model, "handles_of")
                                          else None)
        items.append((c, by_match[c], means.get(c, pre.league) if pre is not None else None, pair))
    workers = max(1, min(workers, len(items)))
    jobs = [(items[i::workers], name, model_dir, n_paths, i) for i in range(workers)]
    if workers == 1:
        parts = [_job(j) for j in jobs]
    else:
        with model.pool_context().Pool(workers) as pool:
            parts = pool.map(_job, jobs)
    return [x for part in parts for x in part]


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row[2], "all")].append(row)
        groups[(row[2], row[3])].append(row)
        groups[("all", "all")].append(row)
    order = {q: i for i, q in enumerate(("all",) + QUARTERS)}
    states = {s: i for i, s in enumerate(("all",) + STATES)}
    return sorted(groups.items(), key=lambda kv: (order.get(kv[0][0], 9), states.get(kv[0][1], 9)))


def summary(rows):
    """Lines of text: for each quarter and state, the real share at each value, the version's
    share minus it, the mean points still to come both ways, and the log score."""
    head = (f"  {'':32s} {'n':>6s} " + " ".join(f"{('=' + str(v)):>6s}" for v in VALUES)
            + f" {'>21':>6s}   {'mean':>6s}")
    real_lines = ["\n  what the rest of the game really made: share at each value (%), and the mean", head]
    diff_lines = ["\n  the version minus reality at each value (points of %), the mean (points), and the "
                  "log score (lower is sharper and right)",
                  head + f" {'log':>6s}"]
    for (q, st), rs in _groups(rows):
        n = len(rs)
        real = np.array([r[5] for r in rs])
        pmfs = np.stack([r[6] for r in rs])
        label = f"{q} {st}"[:32]
        real_share = [100 * np.mean(real == v) for v in VALUES] + [100 * np.mean(real > 21)]
        pred_share = [100 * pmfs[:, v].mean() for v in VALUES] + [100 * pmfs[:, 22:].sum(axis=1).mean()]
        support = np.arange(MAX_REMAINING + 1)
        pred_mean = float((pmfs * support).sum(axis=1).mean())
        idx = np.minimum(real, MAX_REMAINING).astype(int)
        log = float(-np.mean(np.log(np.maximum(pmfs[np.arange(n), idx], 1e-6))))
        real_lines.append(f"  {label:32s} {n:6,d} " + " ".join(f"{x:6.1f}" for x in real_share)
                          + f"   {real.mean():6.2f}")
        diff_lines.append(f"  {label:32s} {n:6,d} "
                          + " ".join(f"{p - r:+6.1f}" for p, r in zip(pred_share, real_share))
                          + f"   {pred_mean - real.mean():+6.2f} {log:6.3f}")
    return real_lines + diff_lines


def over_calibration(rows, thresholds=(0.5, 2.5, 3.5, 6.5, 7.5, 9.5, 10.5, 13.5, 14.5, 16.5, 17.5, 20.5)):
    """Lines of text: at each points-needed threshold, the version's mean P(over) against how often
    the rest of the game went over it, by quarter -- every snapshot, not just where its line sat."""
    lines = ["\n  P(more than N still to come): the version's mean against reality (points of %), "
             "every snapshot",
             f"  {'':6s} {'n':>6s} " + " ".join(f"{t:>6.1f}" for t in thresholds)]
    by_q = defaultdict(list)
    for r in rows:
        by_q[r[2]].append(r)
        by_q["all"].append(r)
    for q in ("all",) + QUARTERS:
        rs = by_q.get(q)
        if not rs:
            continue
        real = np.array([r[5] for r in rs])
        pmfs = np.stack([r[6] for r in rs])
        cells = []
        for t in thresholds:
            k = int(math.ceil(t))
            cells.append(f"{100 * (pmfs[:, k:].sum(axis=1).mean() - np.mean(real > t)):+6.1f}")
        lines.append(f"  {q:6s} {len(rs):6,d} " + " ".join(cells))
    lines.append("  (positive: the version gives more than N more points too often -- an over priced "
                 "too high at a line there)")
    return lines


def write_csv(path, rows, width=36):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_code", "message", "quarter", "state", "on_board", "real_remaining",
                    "pred_mean"] + [f"p{k}" for k in range(width)] + [f"p{width}plus"])
        support = np.arange(MAX_REMAINING + 1)
        for code, msg, q, st, board, real, pmf in rows:
            w.writerow([code, msg, q, st, board, real, round(float((pmf * support).sum()), 3)]
                       + [round(float(x), 5) for x in pmf[:width]] + [round(float(pmf[width:].sum()), 5)])
