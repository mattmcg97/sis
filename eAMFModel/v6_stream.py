"""v6 as a price stream: v6's quotes at every snapshot, shaped like prod's."""

import math
import os
from bisect import bisect_right

import numpy as np

from . import playover, players, sim6 as sim, v6
from .pricer import ML_AWAY, ML_HOME, MARKET_IDS, SPREAD_AWAY, SPREAD_HOME
from .stream import OPEN, _parse_line, description

DEFAULT_MODEL_DIR = "v6_model"
DEFAULT_PATHS = 2000


def model_paths(model_dir=None):
    """The model's tables and grid paths, or an error saying how to build them."""
    candidates = [model_dir] if model_dir else [
        os.environ.get("EAMF_V6_MODEL"), DEFAULT_MODEL_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DEFAULT_MODEL_DIR)]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "v6tables.npz")) \
                and os.path.exists(os.path.join(d, "v6grid.npz")):
            return os.path.join(d, "v6tables.npz"), os.path.join(d, "v6grid.npz")
    raise SystemExit(
        "v6 needs its model (play tables and pre-match grid), and none was found"
        f" in {model_dir or DEFAULT_MODEL_DIR}. Build it once from a PLAY_OVER export:\n"
        "  python -m eAMFCalibrator scouting            # writes out/scouting_playover.csv\n"
        "  python -m eAMFModel v6-build eAMFCalibrator/out/scouting_playover.csv"
        " --half all --out v6_model\n"
        "then point --v6-model (or EAMF_V6_MODEL) at it. Build it on matches before the"
        " window you calibrate, or the comparison is in-sample.")


def _as_text(row):
    """A snapshot dict as strings, as the export reads."""
    return {k: "" if v is None else str(v) for k, v in row.items()}


def match_books(tables, grid, variant, match_rows, n_paths, rng, prof=None, means=None):
    """v6's margin and total distributions at every priceable snapshot of one match."""
    rows = v6.resolve_sides([_as_text(r) for r in match_rows])
    theta0 = v6.prior_theta(grid, means)
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
    dists = v6.price_states(tables, theta0, variant, sim.snap_records(rows), a_home, states,
                            messages, prof, n_paths, rng,
                            seed=v6.match_seed(rows[0].get("match_code", "")))
    return [(m, mp_, tp) for m, (mp_, tp) in zip(messages, dists)]


def paired_prod_rows(prod_quote_rows):
    """The prod row each quote is paired with, per market and message."""
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
    """Prod-shaped quote rows from v6's distributions."""
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
                home = v6.even_line(mpmf, v6.MARGIN_MAX)
                line = home if market_id == SPREAD_HOME else -home
            else:
                line = v6.even_line(tpmf, 0)
        else:
            line = _parse_line(prod_row[5])
            if line is None:
                continue
        p = float(v6.market_prob(market_id, 0.0 if line is None else line, mpmf, tpmf))
        if not math.isfinite(p):
            continue
        p = min(0.9999, max(0.0001, p))
        out.append((match_code, market_id, prod_row[2], round(100.0 * p, 2),
                    round(1.0 / p, 4), description(market_id, line), message, OPEN, "true"))
    return out


def _worker(job):
    """Worker: price a list of matches."""
    items, tables_path, grid_path, variant, n_paths, seed, lines = job
    tables = sim.Tables.load(tables_path)
    grid = v6.PriorGrid.load(grid_path)
    rng = np.random.default_rng(seed)
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    out = {mode: [] for mode in modes}
    for match_code, snaps, prod_rows, first_play, prof, means in items:
        books = match_books(tables, grid, variant, snaps, n_paths, rng, prof, means)
        for mode in modes:
            out[mode].extend(quote_rows(match_code, books, prod_rows, first_play, mode))
    return out


def quotes_for_matches(snapshots_by_match, prod_quote_rows, model_dir=None, n_paths=DEFAULT_PATHS,
                       workers=None, variant=None, book=None, handles=None, seed=0,
                       match_info=None, lines=OWN):
    """Quote rows for many matches."""
    tables_path, grid_path = model_paths(model_dir)
    variant = variant or v6.Variant("v6")
    if book is None:
        book = v6.players_book(tables_path)
    pre = v6.prematch_model(tables_path)
    means = {}
    if pre is not None:
        if match_info is None:
            raise SystemExit("this v6 model prices off its NB2 pre-match model, which needs each "
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
        pair = ((handles or {}).get(code) or v6.handles_of(snaps)) if book else None
        prof = (book.profile(pair[0]), book.profile(pair[1])) if pair else None
        items.append((code, snaps, prod_by[code], first_play, prof,
                      means.get(code, pre.league) if pre is not None else None))
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    if not items:
        return [] if isinstance(lines, str) else {mode: [] for mode in modes}
    workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 1), len(items)))
    jobs = [(items[i::workers], tables_path, grid_path, variant, n_paths, seed + i, lines)
            for i in range(workers)]
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        with v6.pool_context().Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    out = {mode: [row for part in results for row in part[mode]] for mode in modes}
    return out[lines] if isinstance(lines, str) else out
