"""v4 as a stream: quote rows shaped like GAMEPLAI_STREAM's, so the
calibrator can stand v4 in for the candidate (`--candidate v4`). A copy of
v3_stream.py pointed at v4 (and at v4's player profiles when the
calibrator passes handles).

v4 prices on the game clock, which only SCOUTING_FULL carries, so its input
is the PLAY_OVER snapshots eAMFCalibrator's scouting module builds (the
same rows as scouting_playover.csv), plus prod's quotes for the lines and
publish times.

Each PLAY_OVER is priced by simulation. A stream's quote stands until its
next one, so every prod quote after a snapshot gets v4's latest book. v4
quotes its own even line off that book, moving it as the game moves (the
calibrator then scores whose line was nearer the result where the lines
differ); `lines="prod"` re-reads the book at prod's line instead, so the
two streams always answer the same question. Messages before the first
snapshot get no quote.

The model itself -- the play tables and the pre-match grid -- comes from
`python -m eAMFModel v4-build` and is loaded from `model_dir`. It is never
fitted on the matches it prices here.
"""

import math
import os
from bisect import bisect_right

import numpy as np

from . import playover, players, sim4 as sim, v4
from .pricer import ML_AWAY, ML_HOME, MARKET_IDS, SPREAD_AWAY, SPREAD_HOME
from .stream import OPEN, _parse_line, description

DEFAULT_MODEL_DIR = "v4_model"
DEFAULT_PATHS = 2000


def model_paths(model_dir=None):
    """(tables, grid) paths, or SystemExit saying how to build them."""
    candidates = [model_dir] if model_dir else [
        os.environ.get("EAMF_V4_MODEL"), DEFAULT_MODEL_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DEFAULT_MODEL_DIR)]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "v4tables.npz")) \
                and os.path.exists(os.path.join(d, "v4grid.npz")):
            return os.path.join(d, "v4tables.npz"), os.path.join(d, "v4grid.npz")
    raise SystemExit(
        "v4 needs its model (play tables and pre-match grid), and none was found"
        f" in {model_dir or DEFAULT_MODEL_DIR}. Build it once from a PLAY_OVER export:\n"
        "  python -m eAMFCalibrator scouting            # writes out/scouting_playover.csv\n"
        "  python -m eAMFModel v4-build eAMFCalibrator/out/scouting_playover.csv"
        " --half all --out v4_model\n"
        "then point --v4-model (or EAMF_V4_MODEL) at it. Build it on matches before the"
        " window you calibrate, or the comparison is in-sample.")


def _as_text(row):
    """Snapshot dict -> the CSV-string shape the backtest reads."""
    return {k: "" if v is None else str(v) for k, v in row.items()}


def match_books(tables, grid, variant, match_rows, n_paths, rng, prof=None, means=None):
    """[(message, margin pmf, total pmf)] at every priceable PLAY_OVER of one
    match (export-shaped rows, message order). `means`: (home, away)
    expected points from our pre-match model (NB2); without them the prior
    falls back to prod's pre-match quotes (a model built without history)."""
    rows = v4.resolve_sides([_as_text(r) for r in match_rows])
    if means is not None:
        theta0 = v4.fit_means(grid, *means)
    else:
        lines = v4.prior_lines(rows)
        if lines is None:
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
    dists = v4.price_states(tables, theta0, variant, sim.snap_records(rows), a_home, states,
                            messages, prof, n_paths, rng,
                            seed=v4.match_seed(rows[0].get("match_code", "")))
    return [(m, mp_, tp) for m, (mp_, tp) in zip(messages, dists)]


def paired_prod_rows(prod_quote_rows):
    """(market, message) -> the prod row the calibrator pairs a candidate
    quote with: the first live row on that message, else the first row --
    eAMFCalibrator.directional.index_by_message's rule, applied to rows in
    the order they were fetched (publish time). v4 quotes at that row's
    line, so the pair always answers one question."""
    chosen = {}
    for r in prod_quote_rows:
        if r[6] is None or r[3] is None:
            continue
        live = str(r[8]).strip().lower() == "true"
        key = (r[1], r[6])
        held = chosen.get(key)
        if held is None or (live and not held[1]):
            chosen[key] = (r, live)
    return {key: row for key, (row, _) in chosen.items()}


OWN, PROD_LINES = "own", "prod"


def quote_rows(match_code, books, prod_quote_rows, first_play_message=None, lines=OWN):
    """GAMEPLAI-shaped rows: at every (market, message) prod quoted from
    v4's first snapshot on, v4's latest book.

    lines=OWN (the default): v4 prices its own line, the even one off its
    own distribution for the state -- the margin's for the spread, the
    total's for the total -- so the line moves when the game moves it.
    lines=PROD_LINES reads the book at the line of the prod row the
    calibrator pairs with, so every pair answers one question.

    `first_play_message` is kept for callers; pre-match messages carry no
    book and so no quote."""
    if not books:
        return []
    keys = [b[0] for b in books]
    out = []
    for (market_id, message), prod_row in sorted(paired_prod_rows(prod_quote_rows).items(),
                                                 key=lambda kv: (kv[0][1], kv[0][0])):
        if market_id not in MARKET_IDS:
            continue
        i = bisect_right(keys, message) - 1
        if i < 0:
            continue
        _, mpmf, tpmf = books[i]
        if market_id in (ML_HOME, ML_AWAY):
            line = None
        elif lines == OWN:
            if market_id in (SPREAD_HOME, SPREAD_AWAY):
                home = v4.even_line(mpmf, v4.MARGIN_MAX)
                line = home if market_id == SPREAD_HOME else -home
            else:
                line = v4.even_line(tpmf, 0)
        else:
            line = _parse_line(prod_row[5])
            if line is None:
                continue
        p = float(v4.market_prob(market_id, 0.0 if line is None else line, mpmf, tpmf))
        if not math.isfinite(p):
            continue
        p = min(0.9999, max(0.0001, p))
        out.append((match_code, market_id, prod_row[2], round(100.0 * p, 2),
                    round(1.0 / p, 4), description(market_id, line), message, OPEN, "true"))
    return out


def _worker(job):
    items, tables_path, grid_path, variant, n_paths, seed, lines = job
    tables = sim.Tables.load(tables_path)
    grid = v4.PriorGrid.load(grid_path)
    rng = np.random.default_rng(seed)
    out = []
    for match_code, snaps, prod_rows, first_play, prof, means in items:
        books = match_books(tables, grid, variant, snaps, n_paths, rng, prof, means)
        out.extend(quote_rows(match_code, books, prod_rows, first_play, lines))
    return out


def quotes_for_matches(snapshots_by_match, prod_quote_rows, model_dir=None, n_paths=DEFAULT_PATHS,
                       workers=None, variant=None, book=None, handles=None, seed=0,
                       match_info=None, lines=OWN):
    """Quote rows for many matches.

    `snapshots_by_match`: match -> export-shaped PLAY_OVER rows (scouting's
    snapshots_for_match); `prod_quote_rows`: prod's raw quote rows.
    """
    tables_path, grid_path = model_paths(model_dir)
    variant = variant or v4.Variant("v4")
    if book is None:
        book = v4.players_book(tables_path)
    # the prior: our NB2 pre-match model when the model has one, for the
    # matches whose players, teams and stream are known (match_info:
    # AMFELO-shaped rows); prod's pre-match quotes only for an old model
    pre = v4.prematch_model(tables_path)
    means = {}
    if pre is not None:
        if match_info is None:
            raise SystemExit("this v4 model prices off its NB2 pre-match model, which needs each "
                             "match's players, teams and stream (match_info)")
        means = pre.means([r for r in match_info if r["MATCH_CODE"] in snapshots_by_match])
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
        pair = ((handles or {}).get(code) or v4.handles_of(snaps)) if book else None
        prof = (book.profile(pair[0]), book.profile(pair[1])) if pair else None
        items.append((code, snaps, prod_by[code], first_play, prof,
                      means.get(code, pre.league) if pre is not None else None))
    if not items:
        return []
    workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 1), len(items)))
    jobs = [(items[i::workers], tables_path, grid_path, variant, n_paths, seed + i, lines)
            for i in range(workers)]
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        with v4.pool_context().Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    return [row for part in results for row in part]
