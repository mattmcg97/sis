"""v3 as a stream: quote rows shaped like GAMEPLAI_STREAM's, so the
calibrator can stand v3 in for the candidate (`--candidate v3`).

v3 prices on the game clock, which only SCOUTING_FULL carries, so its input
is the PLAY_OVER snapshots eAMFCalibrator's scouting module builds (the
same rows as scouting_playover.csv), plus prod's quotes for the lines and
publish times.

Each PLAY_OVER is priced by simulation. A stream's quote stands until its
next one, so every prod quote after a snapshot gets v3's latest book,
re-read at prod's line at that message: the two streams always answer the
same question. Messages before the first snapshot get no quote.

The model itself -- the play tables and the pre-match grid -- comes from
`python -m eAMFModel v3-build` and is loaded from `model_dir`. It is never
fitted on the matches it prices here.
"""

import math
import multiprocessing as mp
import os
from bisect import bisect_right

import numpy as np

from . import playover, players, sim, v3
from .pricer import ML_AWAY, ML_HOME, MARKET_IDS
from .stream import OPEN, ProdView, description

DEFAULT_MODEL_DIR = "v3_model"
DEFAULT_PATHS = 2000


def model_paths(model_dir=None):
    """(tables, grid) paths, or SystemExit saying how to build them."""
    candidates = [model_dir] if model_dir else [
        os.environ.get("EAMF_V3_MODEL"), DEFAULT_MODEL_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DEFAULT_MODEL_DIR)]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "v3tables.npz")) \
                and os.path.exists(os.path.join(d, "v3grid.npz")):
            return os.path.join(d, "v3tables.npz"), os.path.join(d, "v3grid.npz")
    raise SystemExit(
        "v3 needs its model (play tables and pre-match grid), and none was found"
        f" in {model_dir or DEFAULT_MODEL_DIR}. Build it once from a PLAY_OVER export:\n"
        "  python -m eAMFCalibrator scouting            # writes out/scouting_playover.csv\n"
        "  python -m eAMFModel v3-build eAMFCalibrator/out/scouting_playover.csv"
        " --half all --out v3_model\n"
        "then point --v3-model (or EAMF_V3_MODEL) at it. Build it on matches before the"
        " window you calibrate, or the comparison is in-sample.")


def _as_text(row):
    """Snapshot dict -> the CSV-string shape the backtest reads."""
    return {k: "" if v is None else str(v) for k, v in row.items()}


def match_books(tables, grid, variant, match_rows, n_paths, rng, prof=None):
    """[(message, margin pmf, total pmf)] at every priceable PLAY_OVER of one
    match (export-shaped rows, message order)."""
    rows = [_as_text(r) for r in match_rows]
    lines = v3.prior_lines(rows)
    if lines is None or rows[0]["team_a_side"] not in ("home", "away"):
        return []
    theta0 = grid.fit(*lines)
    a_home = rows[0]["team_a_side"] == "home"
    states, messages = [], []
    for r in rows:
        state, _ = playover.state_for(r)
        if state is not None:
            states.append(state)
            messages.append(int(r["message"]))
    if not states:
        return []
    prof = prof or (players.Profile(), players.Profile())
    dists = v3.price_states(tables, theta0, variant, sim.snap_records(rows), a_home, states,
                            messages, prof, n_paths, rng)
    return [(m, mp_, tp) for m, (mp_, tp) in zip(messages, dists)]


def quote_rows(match_code, books, prod_quote_rows, first_play_message=None):
    """GAMEPLAI-shaped rows: v3's latest book at every prod message from the
    first snapshot on, at prod's line there."""
    if not books:
        return []
    prod = ProdView(prod_quote_rows, first_play_message)
    # lines off prod's live quotes: a message can also carry the settlement
    # of the line prod just moved off
    live = ProdView([r for r in prod_quote_rows if str(r[8]).strip().lower() == "true"],
                    first_play_message)
    keys = [b[0] for b in books]
    out = []
    for message in prod.messages():
        i = bisect_right(keys, message) - 1
        if i < 0:
            continue
        _, mpmf, tpmf = books[i]
        publish_time = prod.time_at(message)
        for market_id in MARKET_IDS:
            if market_id in (ML_HOME, ML_AWAY):
                line = None
            else:
                line = live.line_at(market_id, message)
                if line is None:
                    line = prod.line_at(market_id, message)
                if line is None:
                    continue
            p = float(v3.market_prob(market_id, 0.0 if line is None else line, mpmf, tpmf))
            if not math.isfinite(p):
                continue
            p = min(0.9999, max(0.0001, p))
            out.append((match_code, market_id, publish_time, round(100.0 * p, 2),
                        round(1.0 / p, 4), description(market_id, line), message, OPEN, "true"))
    return out


def _worker(job):
    items, tables_path, grid_path, variant, n_paths, seed = job
    tables = sim.Tables.load(tables_path)
    grid = v3.PriorGrid.load(grid_path)
    rng = np.random.default_rng(seed)
    out = []
    for match_code, snaps, prod_rows, first_play, prof in items:
        books = match_books(tables, grid, variant, snaps, n_paths, rng, prof)
        out.extend(quote_rows(match_code, books, prod_rows, first_play))
    return out


def quotes_for_matches(snapshots_by_match, prod_quote_rows, model_dir=None, n_paths=DEFAULT_PATHS,
                       workers=None, variant=None, book=None, handles=None, seed=0):
    """Quote rows for many matches.

    `snapshots_by_match`: match -> export-shaped PLAY_OVER rows (scouting's
    snapshots_for_match); `prod_quote_rows`: prod's raw quote rows.
    """
    tables_path, grid_path = model_paths(model_dir)
    variant = variant or v3.Variant("v3", react=False, profiles=bool(book), pace=bool(book))
    prod_by = {}
    for r in prod_quote_rows:
        prod_by.setdefault(r[0], []).append(r)
    items = []
    for code, snaps in snapshots_by_match.items():
        if not snaps or code not in prod_by:
            continue
        snaps = sorted(snaps, key=lambda r: int(r["message"]))
        first_play = snaps[0].get("first_play_message")
        first_play = int(first_play) if first_play not in ("", None) else None
        pair = handles.get(code) if (book and handles) else None
        prof = (book.profile(pair[0]), book.profile(pair[1])) if pair else None
        items.append((code, snaps, prod_by[code], first_play, prof))
    if not items:
        return []
    workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 1), len(items)))
    jobs = [(items[i::workers], tables_path, grid_path, variant, n_paths, seed + i)
            for i in range(workers)]
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    return [row for part in results for row in part]
