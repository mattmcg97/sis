"""v10 as a price stream: v10's quotes at every snapshot, shaped like prod's."""

import math
import os
from bisect import bisect_right

import numpy as np

from . import playover, players, sim10 as sim, v10
from .state import ML_AWAY, ML_HOME, MARKET_IDS, SPREAD_AWAY, SPREAD_HOME
from .stream import OPEN, _parse_line, confident_windows, description, side_known

DEFAULT_MODEL_DIR = "v10_model"
DEFAULT_PATHS = 2000
PREMATCH = True


def model_paths(model_dir=None):
    """The model's tables and grid paths, or an error saying how to build them."""
    candidates = [model_dir] if model_dir else [
        os.environ.get("EAMF_V10_MODEL"), DEFAULT_MODEL_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DEFAULT_MODEL_DIR)]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "v10tables.npz")) \
                and os.path.exists(os.path.join(d, "v10grid.npz")):
            return os.path.join(d, "v10tables.npz"), os.path.join(d, "v10grid.npz")
    raise SystemExit(
        "v10 needs its model (play tables and pre-match grid), and none was found"
        f" in {model_dir or DEFAULT_MODEL_DIR}. Build it once from a PLAY_OVER export:\n"
        "  python -m eAMFCalibrator scouting            # writes out/scouting_playover.csv\n"
        "  python -m eAMFModel v10-build eAMFCalibrator/out/scouting_playover.csv"
        " --half all --out v10_model\n"
        "then point --v10-model (or EAMF_V10_MODEL) at it. Build it on matches before the"
        " window you calibrate, or the comparison is in-sample.")


def _as_text(row):
    """A snapshot dict as strings, as the export reads."""
    return {k: "" if v is None else str(v) for k, v in row.items()}


def match_books(tables, grid, variant, match_rows, n_paths, rng, prof=None, means=None):
    """v10's margin and total distributions at every priceable snapshot of one match."""
    rows = v10.resolve_sides([_as_text(r) for r in match_rows])
    prof = prof or (players.Profile(), players.Profile())
    theta0 = v10.prior_theta(grid, means, prof if variant.pace else None)
    a_home = rows[0]["team_a_side"] == "home"
    states, messages = [], []
    for r in rows:
        state, _ = v10.state_for(r)
        if state is not None:
            states.append(state)
            messages.append(int(r["message"]))
    if not states:
        return []
    dists = v10.price_states(tables, theta0, variant, sim.snap_records(rows), a_home, states,
                            messages, prof, n_paths, rng,
                            seed=v10.match_seed(rows[0].get("match_code", "")))
    return [(m, mp_, tp) for m, (mp_, tp) in zip(messages, dists)]


def kickoff_book(tables, grid, variant, match_code, n_paths, rng, prof=None, means=None):
    """v10's pre-match margin and total distributions: the kick-off priced off the pre-match prior
    (NB2 or glmer)."""
    prof = prof or (players.Profile(), players.Profile())
    return v10.price_kickoff(tables, v10.prior_theta(grid, means, prof if variant.pace else None),
                             variant, prof, n_paths, rng, seed=v10.match_seed(match_code))


def _before_play(message, first_play_message):
    """Whether a prod message is pre-match: no message, or one before the first play started."""
    return message is None or (first_play_message is not None and message < first_play_message)


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


# How the stream sets a handicap or total line: its own even line, prod's, prod's moved only as far
# as it must to bring P(over) inside a band (anchored), or its own even line held until it is far
# off (hyst). v10.anchored_line and v10.held_line say how.
OWN, PROD_LINES, ANCHORED, HELD = "own", "prod", "anchored", "hyst"
LINE_MODES = (OWN, PROD_LINES, ANCHORED, HELD)


def _line(market_id, prod_row, mpmf, tpmf, lines, held=None):
    """A handicap or total market's line under a line rule; None where prod's line is needed and
    cannot be read. `held` (hyst) carries each group's line from one quote to the next."""
    spread = market_id in (SPREAD_HOME, SPREAD_AWAY)
    pmf, offset = (mpmf, v10.MARGIN_MAX) if spread else (tpmf, 0)
    sign = -1.0 if market_id == SPREAD_AWAY else 1.0           # the away line is home's, negated
    if lines == OWN:
        x = v10.even_line(pmf, offset)
    elif lines == HELD:
        key = "spread" if spread else "total"
        x = v10.held_line(pmf, offset, None if held is None else held.get(key))
        if held is not None:
            held[key] = x
    elif lines in (PROD_LINES, ANCHORED):
        prod = _parse_line(prod_row[5])
        if prod is None:
            return None
        if lines == PROD_LINES:
            return prod
        x = v10.anchored_line(pmf, offset, sign * prod)
    else:
        raise ValueError(f"no such line rule {lines!r} (there are {', '.join(LINE_MODES)})")
    return sign * x


def _quote(match_code, market_id, prod_row, mpmf, tpmf, lines, message, held=None):
    """One prod-shaped quote row off a margin and total distribution, at the line `lines` sets;
    None where it cannot be priced."""
    if market_id in (ML_HOME, ML_AWAY):
        line = None
    else:
        line = _line(market_id, prod_row, mpmf, tpmf, lines, held)
        if line is None:
            return None
    p = float(v10.market_prob(market_id, 0.0 if line is None else line, mpmf, tpmf))
    if not math.isfinite(p):
        return None
    p = min(0.9999, max(0.0001, p))
    return (match_code, market_id, prod_row[2], round(100.0 * p, 2), round(1.0 / p, 4),
            description(market_id, line), message, OPEN, "true")


def quote_rows(match_code, books, prod_quote_rows, first_play_message=None, lines=OWN, windows=None,
               kickoff=None):
    """Prod-shaped quote rows from v10's distributions, at the lines `lines` sets. With `kickoff` (the pre-match margin and
    total distributions), every prod row published before the first play started -- no message,
    or one below first_play_message -- gets v10's pre-match price, on its own message (none for
    none) and publish time."""
    keys = [b[0] for b in books]
    out, held = [], {}
    if kickoff is not None:
        for r in prod_quote_rows:
            if r[1] in MARKET_IDS and r[3] is not None and r[6] is None:
                q = _quote(match_code, r[1], r, kickoff[0], kickoff[1], lines, None, held)
                if q:
                    out.append(q)
    for (market_id, message), prod_row in sorted(paired_prod_rows(prod_quote_rows).items(),
                                                 key=lambda kv: (kv[0][1], kv[0][0])):
        if market_id not in MARKET_IDS:
            continue
        i = bisect_right(keys, message) - 1
        if i < 0:
            if kickoff is not None and _before_play(message, first_play_message):
                q = _quote(match_code, market_id, prod_row, kickoff[0], kickoff[1], lines, message,
                                   held)
                if q:
                    out.append(q)
            continue
        if windows is not None and not (keys[i] in windows and message < windows[keys[i]]):
            continue
        _, mpmf, tpmf = books[i]
        q = _quote(match_code, market_id, prod_row, mpmf, tpmf, lines, message, held)
        if q:
            out.append(q)
    return out


def _worker(job):
    """Worker: price a list of matches."""
    items, tables_path, grid_path, variant, n_paths, seed, lines = job
    tables = sim.Tables.load(tables_path)
    grid = v10.PriorGrid.load(grid_path)
    rng = np.random.default_rng(seed)
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    out = {mode: [] for mode in modes}
    for match_code, snaps, prod_rows, first_play, prof, means in items:
        kickoff = (kickoff_book(tables, grid, variant, match_code, n_paths, rng, prof, means)
                   if PREMATCH else None)
        if side_known(snaps):
            books = match_books(tables, grid, variant, snaps, n_paths, rng, prof, means)
            windows = confident_windows(snaps, [b[0] for b in books])
        else:
            books, windows = [], {}
        for mode in modes:
            out[mode].extend(quote_rows(match_code, books, prod_rows, first_play, mode, windows,
                                        kickoff))
    return out


def quotes_for_matches(snapshots_by_match, prod_quote_rows, model_dir=None, n_paths=DEFAULT_PATHS,
                       workers=None, variant=None, book=None, handles=None, seed=0,
                       match_info=None, lines=OWN):
    """Quote rows for many matches."""
    tables_path, grid_path = model_paths(model_dir)
    variant = variant or v10.Variant("v10")
    if book is None:
        book = v10.players_book(tables_path)
    pre = v10.prematch_model(tables_path)
    means = {}
    if pre is not None:
        if match_info is None:
            raise SystemExit("this v10 model prices off its own pre-match model (NB2 or glmer), which"
                             " needs each match's players, teams and stream (match_info)")
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
        pair = ((handles or {}).get(code) or v10.handles_of(snaps)) if book else None
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
        with v10.pool_context().Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    out = {mode: [row for part in results for row in part[mode]] for mode in modes}
    return out[lines] if isinstance(lines, str) else out
