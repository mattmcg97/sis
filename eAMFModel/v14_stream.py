"""v14 as a price stream: v14's quotes at every snapshot, shaped like prod's."""

import os
import time
from bisect import bisect_right

import numpy as np

from . import playover, players, sim14 as sim, v14
from .state import MARKET_IDS
# the line rules and quotes every stream shares (stream.py), under v14_stream's names
from .stream import (ANCHORED, EVEN, HELD, LINE_MODES, OPEN, OWN, PROD_LINES, _before_play, _line,  # noqa: F401
                     _parse_line, _quote, confident_windows, description, known_states, match_ends,
                     paired_prod_rows, prematch_publish_times, side_known)

DEFAULT_MODEL_DIR = "v14_model"
DEFAULT_PATHS = 2000
PREMATCH = True


def model_paths(model_dir=None):
    """The model's tables and grid paths, or an error saying how to build them."""
    candidates = [model_dir] if model_dir else [
        os.environ.get("EAMF_V14_MODEL"), DEFAULT_MODEL_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DEFAULT_MODEL_DIR)]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "v14tables.npz")) \
                and os.path.exists(os.path.join(d, "v14grid.npz")):
            return os.path.join(d, "v14tables.npz"), os.path.join(d, "v14grid.npz")
    raise SystemExit(
        "v14 needs its model (play tables and pre-match grid), and none was found"
        f" in {model_dir or DEFAULT_MODEL_DIR}. Build it once from a PLAY_OVER export:\n"
        "  python -m eAMFCalibrator scouting            # writes out/scouting_playover.csv\n"
        "  python -m eAMFModel v14-build eAMFCalibrator/out/scouting_playover.csv"
        " --half all --out v14_model\n"
        "then point --v14-model (or EAMF_V14_MODEL) at it. Build it on matches before the"
        " window you calibrate, or the comparison is in-sample.")


def _as_text(row):
    """A snapshot dict as strings, as the export reads."""
    return {k: "" if v is None else str(v) for k, v in row.items()}


def match_books(tables, grid, variant, match_rows, n_paths, rng, prof=None, means=None):
    """v14's margin and total distributions at every priceable snapshot of one match."""
    rows = v14.resolve_sides([_as_text(r) for r in match_rows])
    prof = prof or (players.Profile(), players.Profile())
    theta0 = v14.held_theta(tables, grid, means, prof, variant, seed=v14.match_seed(rows[0].get("match_code", "")))
    a_home = rows[0]["team_a_side"] == "home"
    states, messages = [], []
    for r in rows:
        state, _ = v14.state_for(r)
        if state is not None:
            states.append(state)
            messages.append(int(r["message"]))
    if not states:
        return []
    dists = v14.price_states(tables, theta0, variant, sim.snap_records(rows), a_home, states,
                            messages, prof, n_paths, rng,
                            seed=v14.match_seed(rows[0].get("match_code", "")))
    return [(m, mp_, tp) for m, (mp_, tp) in zip(messages, dists)]


def kickoff_book(tables, grid, variant, match_code, n_paths, rng, prof=None, means=None):
    """v14's pre-match margin and total distributions: the kick-off priced off the pre-match prior
    (NB2 or glmer)."""
    prof = prof or (players.Profile(), players.Profile())
    theta0 = v14.held_theta(tables, grid, means, prof, variant, seed=v14.match_seed(match_code))
    return v14.price_kickoff(tables, theta0, variant, prof, n_paths, rng, seed=v14.match_seed(match_code))


def quote_rows(match_code, books, prod_quote_rows, first_play_message=None, lines=OWN, windows=None,
               kickoff=None):
    """Prod-shaped quote rows from v14's distributions, at the lines `lines` sets. With `kickoff` (the
    pre-match margin and total distributions, or a function of the prod row giving them: what was
    known when it was published), every prod row published before the first play started -- no
    message, or one below first_play_message -- gets v14's pre-match price, on its own message
    (none for none) and publish time."""
    keys = [b[0] for b in books]
    out, held = [], {}
    kickoff_at = kickoff if callable(kickoff) else (lambda row: kickoff)
    if kickoff is not None:
        for r in prod_quote_rows:
            if r[1] in MARKET_IDS and r[3] is not None and r[6] is None:
                mpmf, tpmf = kickoff_at(r)
                q = _quote(match_code, r[1], r, mpmf, tpmf, lines, None, held)
                if q:
                    out.append(q)
    for (market_id, message), prod_row in sorted(paired_prod_rows(prod_quote_rows).items(),
                                                 key=lambda kv: (kv[0][1], kv[0][0])):
        if market_id not in MARKET_IDS:
            continue
        i = bisect_right(keys, message) - 1
        if i < 0:
            if kickoff is not None and _before_play(message, first_play_message):
                mpmf, tpmf = kickoff_at(prod_row)
                q = _quote(match_code, market_id, prod_row, mpmf, tpmf, lines, message, held)
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
    """Worker: price a list of matches; with prematch_only, the kick-off alone (no in-play books).
    `known` (publish time -> expected points): the pre-match quotes priced off what was known when
    they were published, where that differs from the kick-off's."""
    items, tables_path, grid_path, variant, n_paths, seed, lines, prematch_only = job
    tables = sim.Tables.load(tables_path)
    grid = v14.PriorGrid.load(grid_path)
    rng = np.random.default_rng(seed)
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    out = {mode: [] for mode in modes}
    for match_code, snaps, prod_rows, first_play, prof, means, known in items:
        kickoff = (kickoff_book(tables, grid, variant, match_code, n_paths, rng, prof, means)
                   if PREMATCH else None)
        if kickoff is not None and known:
            # pre-match quotes published before results the form follows came in
            kicks = {m: kickoff_book(tables, grid, variant, match_code, n_paths, rng, prof, m)
                     for m in sorted(set(known.values()))}
            at = {publish: kicks[m] for publish, m in known.items()}
            kickoff = (lambda row, at=at, own=kickoff: at.get(row[2], own))
        if side_known(snaps) and not prematch_only:
            books = match_books(tables, grid, variant, snaps, n_paths, rng, prof, means)
            windows = confident_windows(snaps, [b[0] for b in books])
        else:
            books, windows = [], {}
        for mode in modes:
            out[mode].extend(quote_rows(match_code, books, prod_rows, first_play, mode, windows,
                                        kickoff))
    return out


LAST_TIMING = {}        # the last quotes_for_matches call's seconds: prematch, simulate (and workers)


def quotes_for_matches(snapshots_by_match, prod_quote_rows, model_dir=None, n_paths=DEFAULT_PATHS,
                       workers=None, variant=None, book=None, handles=None, seed=0,
                       match_info=None, lines=OWN, history=None, prematch_only=False):
    """Quote rows for many matches. `history`: settled matches (with finals) known when pricing,
    for a pre-match model whose form follows them (glmer): the kick-off and in-play prices read
    every result before the match, and each pre-match quote only those in by its publish time
    (stream.known_states). `prematch_only`: price only the prod rows published before each match's
    first play, off the kick-off -- no in-play simulation, so a long window of pre-match bets is
    cheap."""
    tables_path, grid_path = model_paths(model_dir)
    variant = variant or v14.Variant("v14")
    if book is None:
        book = v14.players_book(tables_path)
    pre = v14.prematch_model(tables_path)
    prod_by = {}
    for r in prod_quote_rows:
        prod_by.setdefault(r[0], []).append(r)
    matches = {}
    for code, snaps in snapshots_by_match.items():
        if not snaps or code not in prod_by:
            continue
        snaps = sorted(snaps, key=lambda r: int(r["message"]))
        first_play = snaps[0].get("first_play_message")
        matches[code] = (snaps, int(first_play) if first_play not in ("", None) else None)
    means, known = {}, {}
    LAST_TIMING.clear()
    t0 = time.time()
    if pre is not None:
        if match_info is None:
            raise SystemExit("this v14 model prices off its own pre-match model (NB2 or glmer), which"
                             " needs each match's players, teams and stream (match_info)")
        rows, code_at = [r for r in match_info if r["MATCH_CODE"] in matches], {}
        if PREMATCH and history is not None and getattr(pre, "FOLLOWS_RESULTS", False) \
                and getattr(pre, "AS_OF", True):
            publish = {c: prematch_publish_times(prod_by[c], fp) for c, (_, fp) in matches.items()}
            rows, code_at = known_states(rows, publish, history, match_ends(snapshots_by_match))
        means = pre.means(rows, results=history)
        for (code, publish), at in code_at.items():
            if at != code:
                known.setdefault(code, {})[publish] = means.get(at, pre.league)
    LAST_TIMING.update(prematch=time.time() - t0, cuts=sum(len(set(k.values())) for k in known.values()))
    items = []
    ratings = v14.big_ratings(tables_path)
    teams = v14.match_teams(match_info)
    for code, (snaps, first_play) in matches.items():
        pair = ((handles or {}).get(code) or v14.handles_of(snaps)) if (book or ratings) else None
        prof = (book.profile(pair[0]), book.profile(pair[1])) if (pair and book) else None
        if pair and ratings:
            prof = v14.with_big(prof or (players.Profile(), players.Profile()), ratings, pair,
                                *v14.side_teams(pair, teams.get(code)))
        items.append((code, snaps, prod_by[code], first_play, prof,
                      means.get(code, pre.league) if pre is not None else None, known.get(code)))
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    if not items:
        return [] if isinstance(lines, str) else {mode: [] for mode in modes}
    workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 1), len(items)))
    jobs = [(items[i::workers], tables_path, grid_path, variant, n_paths, seed + i, lines,
             prematch_only) for i in range(workers)]
    t0 = time.time()
    if workers == 1:
        results = [_worker(j) for j in jobs]
    else:
        with v14.pool_context().Pool(workers) as pool:
            results = pool.map(_worker, jobs)
    LAST_TIMING.update(simulate=time.time() - t0, workers=workers)
    out = {mode: [row for part in results for row in part[mode]] for mode in modes}
    return out[lines] if isinstance(lines, str) else out
