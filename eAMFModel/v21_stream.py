"""v2.1 as a price stream: its quotes at every snapshot, shaped like prod's, for the calibrator
(`python -m eAMFCalibrator ... --candidate v21`).

Each match is priced as v21-report prices it. From kick-off the sim plays the match's gamers and
teams, each side's offense nudged NUDGE_SHARE of the way from the sim's own means to the prior's
expected points (v21.nudge_solve; the prior is a v13/v14 build's pre-match model, as v21-report's
--prior-model). In play, every PLAY_OVER the sim can start from is priced from there with the same
nudge: a snap (v21.match_starts) or a kick-off after a score.

A touchdown's PLAY_OVER, with its conversion still to come, has no book, because the sim has no
start there. Prod's quotes until the conversion go unpaired. A prod quote reads the book of the
PLAY_OVER before it while that state lasts (stream.confident_windows), and only those books are
simulated.

With an in-play strength update (v21_react: a candidate named v21-snapT or v21-ptsT), every in-play
state gets its own nudge, the kick-off's moved by what the match has shown of each side by then."""

import os
import time
import zlib
from bisect import bisect_left, bisect_right

import numpy as np

from . import playover, sim14 as s14, v14, v21, v21_react
from .state import HOME, AWAY, MARKET_IDS
from .stream import (confident_windows, known_states, match_ends, prematch_publish_times, side_known)
# the line rules, as v14's (the calibrator reads every one of LINE_MODES off one simulation)
from .v14_stream import (EVEN, HELD, LINE_MODES, OWN, PROD_LINES, ANCHORED, _before_play, _quote,  # noqa: F401
                         paired_prod_rows)

DEFAULT_MODEL_DIR = "v21_model"
DEFAULT_PRIOR_DIR = "v14_model"
DEFAULT_PATHS = 1000          # games a snapshot (the kick-off: each side receiving, so twice that)
NUDGE_SHARE = 0.75            # how far toward the prior's expected points (v21-report's --nudge-share)
PREMATCH = True
# Every match's sim is seeded by the match (_seed): its quotes are the same whichever matches it is
# priced with, so the calibrator prices several of its chunks in one pool (snowflake_io._sim_quotes).
MATCH_SEEDED = True


def _model_file(model_dir=None):
    candidates = [model_dir] if model_dir else [
        os.environ.get("EAMF_V21_MODEL"), DEFAULT_MODEL_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DEFAULT_MODEL_DIR)]
    for d in candidates:
        if not d:
            continue
        path = d if d.endswith(".pkl") else os.path.join(d, v21.MODEL_FILE)
        if os.path.exists(path):
            return path
    raise SystemExit(
        f"v21 needs its model, and none was found in {model_dir or DEFAULT_MODEL_DIR}. Build it once:\n"
        "  python -m eAMFModel v21-build --until <the window's start> --out v21_model --mode tilt"
        " --key lean --blend 40 --field 0.2 --decision-blend 25 --clock 0.2\n"
        "then point --v21-model (or EAMF_V21_MODEL) at it. Build it on matches before the window"
        " you calibrate, or the comparison is in-sample.")


def _prior_tables(prior_dir=None):
    """The prior build's tables (its pre-match model), or a prior table (a walkthrough variant's CSV,
    table_prior.py) as it is; None where none is found."""
    from . import prior_check, table_prior
    if table_prior.is_table(prior_dir):
        return prior_dir
    candidates = [prior_dir] if prior_dir else [
        os.environ.get("EAMF_V21_PRIOR"), os.environ.get("EAMF_V14_MODEL"), DEFAULT_PRIOR_DIR,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", DEFAULT_PRIOR_DIR)]
    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        version = prior_check.version_of(d)
        _, st = prior_check._modules(version)
        try:
            return st.model_paths(d)[0]
        except SystemExit:
            continue
    return None


def model_paths(model_dir=None, prior_dir=None):
    """(the prior's tables or None, v21's model file): the first is what prematch_model reads."""
    return _prior_tables(prior_dir), _model_file(model_dir)


def prematch_model(prior_tables):
    """The prior's pre-match model (its own expected points: the base under the build's layers, as
    v21-report's raw inputs), or None without a prior."""
    if prior_tables is None:
        return None
    from . import prior_check, table_prior
    if table_prior.is_table(prior_tables):        # a walkthrough variant's expected points, as they are
        return table_prior.TablePrior(prior_tables)
    version = prior_check.version_of(os.path.dirname(prior_tables))
    m, _ = prior_check._modules(version)
    pre = m.prematch_model(prior_tables)
    # the base, so a caller reads FOLLOWS_RESULTS off the fit v21 prices with: an NB2 fit under v14's
    # follow layer says it follows results, its own means don't, and history fetched for it goes unread
    return None if pre is None else _base(pre)


def _base(pre):
    from . import prior_check
    return prior_check.base_prior(pre)


def react_of(name):
    """The in-play strength update a candidate's name asks for ('v21-snap4' -> ('snap', 4.0)), or None."""
    return v21_react.variant_of(name)


def _seed(code):
    return zlib.crc32(str(code).encode())


def _as_text(row):
    return {k: "" if v is None else str(v) for k, v in row.items()}


def _side(side):
    return 0 if side == HOME else 1


def state_starts(model, rows, pair=("", ""), teams=("", "")):
    """[(PLAY_OVER message, v21 start)] in message order: every snap state (v21.match_starts) and
    every kick-off after a score (a conversion's PLAY_OVER, a made field goal's, or a missed one's
    where the kicking side keeps the kick-off), the receiving side's."""
    rows = v21._with_side(rows)
    if not rows:
        return []
    out = {}
    for i, s in v21.match_starts(model, rows, pair, teams):
        out[s14._i(rows[i]["message"])] = s
    a_home = rows[0]["team_a_side"] != "away"
    starts = sorted(r["start"] for r in v21.match_snaps(rows))
    opening = rows[0].get("opening_offense")
    first = v21._side_of(opening, a_home) if opening in ("TEAM_A", "TEAM_B") else 0
    for r in rows:
        m = s14._i(r.get("message"))
        if m is None or m in out:
            continue
        st, _ = playover.state_for(r)
        if st is None or st.pending_conversion is not None or st.down is not None \
                or st.offense not in (HOME, AWAY):
            continue
        period, clock = st.period, st.clock_seconds
        if period is None or clock is None or clock <= 0 or period > 4 + v21.MAX_OT:
            continue
        used = (s14._i(r.get("timeouts_used_a")), s14._i(r.get("timeouts_used_b")))
        have = v21.TIMEOUTS if period <= 4 else v21.OT_TIMEOUTS
        tos_a = [max(0, have - u) if u is not None else have for u in used]
        tos = tos_a if a_home else tos_a[::-1]
        out[m] = dict(period=np.array([period]), clock=np.array([float(clock)]),
                      team=np.array([_side(st.offense)]), phase=np.array([v21.KICK]),
                      down=np.array([1]), dist=np.array([10]), y=np.array([25]),
                      home=np.array([st.home_score]), away=np.array([st.away_score]),
                      prev=np.array([v21.FRESH]), elapsed=np.array([bisect_left(starts, m)]),
                      tos=np.array([tos]), kick2=np.array([1 - first]), **v21._start(model, 1, pair, teams))
    return sorted(out.items())


def _needed(keys, prod_quote_rows, windows):
    """The PLAY_OVER messages whose books a match's prod quotes read (quote_rows's own pairing)."""
    need = set()
    for (market_id, message), _ in paired_prod_rows(prod_quote_rows).items():
        if market_id not in MARKET_IDS:
            continue
        i = bisect_right(keys, message) - 1
        if i < 0:
            continue
        if keys[i] in windows and message < windows[keys[i]]:
            need.add(keys[i])
    return need


def quote_rows(match_code, books, prod_quote_rows, first_play_message=None, lines=OWN, windows=None,
               kickoff=None):
    """Prod-shaped quote rows (v14_stream.quote_rows's pairing) off v21's books: (message, margin
    pmf, total pmf), None where a book was not simulated -- no quote then."""
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
        if mpmf is None:
            continue
        q = _quote(match_code, market_id, prod_row, mpmf, tpmf, lines, message, held)
        if q:
            out.append(q)
    return out


def _kickoff(model, pair, teams, target, n_paths, code, share):
    """(kick-off margin and total distributions, the nudge alpha or None, each side's simulated mean
    and variance of its points)."""
    start = v21.kickoff_start(model, pair, teams)
    if target is not None and share and getattr(model, "nudge_dir", None) is not None:
        (h, a), alpha = v21.nudge_solve(model, start, target, n_paths, seed=_seed(code), share=share)
    else:
        (h, a), alpha = v21.simulate(model, start, n_paths, seed=_seed(code)), None
    h, a = h.ravel(), a.ravel()
    return v14._distributions(h, a), alpha, dict(mean=(h.mean(), a.mean()), var=(h.var(), a.var()))


def match_books(model, snaps, pair, teams, n_paths, code, alpha, prod_rows, react=None, kick=None):
    """(books, windows): v21's margin and total distributions at the PLAY_OVERs the match's prod
    quotes read, every other PLAY_OVER keyed with None. `react` (v21_react.spec_of): each state's
    nudge moved by the match so far (kick: the kick-off's points, for the points update)."""
    rows = [_as_text(r) for r in sorted(snaps, key=lambda r: int(r["message"]))]
    starts = state_starts(model, rows, pair, teams)
    keys = [m for m, _ in starts]
    windows = confident_windows(snaps, keys)
    need = _needed(keys, prod_rows, windows)
    sel = [(m, s) for m, s in starts if m in need]
    books = {m: (m, None, None) for m in keys}
    if sel:
        S = v21.concat_starts([s for _, s in sel])
        if alpha is not None or react:
            base = np.zeros(2) if alpha is None else np.asarray(alpha, float)
            S["nudge"] = np.tile(base, (len(S["period"]), 1))
            if react:
                extra = react[2] if len(react) > 2 else {}
                S["nudge"] = S["nudge"] + v21_react.updates(model, rows, sel, base, react[:2], pair, teams, kick,
                                                            wp=extra.get("wp"), gamers=extra.get("gamers"))
        sh, sa = v21.simulate(model, S, n_paths, seed=_seed(code) + 1)
        for k, (m, _) in enumerate(sel):
            mp_, tp = v14._distributions(sh[k], sa[k])
            books[m] = (m, mp_, tp)
    return [books[m] for m in keys], windows


def _price(model, item, n_paths, lines, prematch_only, share, react=None):
    """One match's quote rows by line mode: the kick-off (and the pre-match quotes published before
    results the prior's form follows came in), then the PLAY_OVERs prod's quotes read."""
    code, snaps, prod_rows, first_play, pair, teams, target, known = item
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    kick, alpha, kstats = _kickoff(model, pair, teams, target, n_paths, code, share)
    kickoff = kick if PREMATCH else None
    if kickoff is not None and known:
        kicks = {m: _kickoff(model, pair, teams, m, n_paths, code, share)[0] for m in sorted(set(known.values()))}
        at = {publish: kicks[m] for publish, m in known.items()}
        kickoff = (lambda row, at=at, own=kick: at.get(row[2], own))
    if side_known(snaps) and not prematch_only:
        books, windows = match_books(model, snaps, pair, teams, n_paths, code, alpha, prod_rows, react, kstats)
    else:
        books, windows = [], {}
    return {mode: quote_rows(code, books, prod_rows, first_play, mode, windows, kickoff) for mode in modes}


def _worker(job):
    """Price a list of matches, one after another (_price)."""
    items, model_file, n_paths, lines, prematch_only, share, react = job
    model = v21.Model.load(model_file)
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    out = {mode: [] for mode in modes}
    for item in items:
        got = _price(model, item, n_paths, lines, prematch_only, share, react)
        for mode in modes:
            out[mode].extend(got[mode])
    return out


_LOADED = {}            # a pool worker's model, read once for every match it is handed (_load)


def _load(model_file):
    _LOADED.clear()
    _LOADED[model_file] = v21.Model.load(model_file)


def _one(job):
    """A pool worker: one match, off the model it loaded."""
    item, model_file, n_paths, lines, prematch_only, share, react = job
    if model_file not in _LOADED:
        _load(model_file)
    return _price(_LOADED[model_file], item, n_paths, lines, prematch_only, share, react)


LAST_TIMING = {}        # the last quotes_for_matches call's seconds: prematch, simulate (and workers)


def quotes_for_matches(snapshots_by_match, prod_quote_rows, model_dir=None, n_paths=DEFAULT_PATHS,
                       workers=None, match_info=None, lines=OWN, history=None, prematch_only=False,
                       prior_dir=None, nudge_share=None, handles=None, batches=None, react=None):
    """Quote rows for many matches (v14_stream.quotes_for_matches's contract). `match_info`: each
    match's players and teams (and the prior's inputs); `history`: settled matches, for a prior whose
    form follows results; `nudge_share`: how far toward the prior's expected points (NUDGE_SHARE;
    0: the sim's own means, no prior needed); `batches`: the calibrator's chunks priced here together,
    each match's earlier pre-match states read off its own chunk's ends of play, as a call for that
    chunk alone reads them; `react`: the in-play strength update (v21_react.spec_of; None: none)."""
    share = NUDGE_SHARE if nudge_share is None else float(nudge_share)
    prior_tables, model_file = model_paths(model_dir, prior_dir)
    if react:                                        # the tables a variant reads, once with the build
        fit_it = ("fit them once on real games before the window --\n  python -m eAMFModel v21-react-fit --model"
                  " <the build> --snapshots <scouting_playover.csv> --history <match_history.csv> --until"
                  " <the window's start>")
        extra = {}
        if react[0].endswith("wpa"):
            extra["wp"] = v21_react.load_wp(model_file)
            if extra["wp"] is None:
                raise SystemExit(f"the {react[0]} update reads its win-probability table, {v21_react.WP_FILE}, beside"
                                 f" the build ({os.path.dirname(model_file)}): {fit_it}")
        if react[0].startswith("g"):
            extra["gamers"] = v21_react.load_gamers(model_file)
            if not extra["gamers"] or react[0][1:] not in extra["gamers"]:
                raise SystemExit(f"the {react[0]} update reads each gamer's own scale, {v21_react.GAMERS_FILE}, beside"
                                 f" the build ({os.path.dirname(model_file)}): {fit_it}")
        react = (react[0], react[1], extra)
    pre = prematch_model(prior_tables)
    if share and pre is None:
        raise SystemExit(
            "v21 prices toward a prior's expected points (its nudge), and no prior build was found in"
            f" {prior_dir or DEFAULT_PRIOR_DIR}: point --v21-prior-model (or EAMF_V21_PRIOR) at a v14 build"
            " made with --history, or price without the nudge (--v21-nudge-share 0)")
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
    info = {r["MATCH_CODE"]: r for r in (match_info or [])}
    means, known = {}, {}
    LAST_TIMING.clear()
    t0 = time.time()
    if share and pre is not None:
        base = _base(pre)
        rows, code_at = [r for r in (match_info or []) if r["MATCH_CODE"] in matches], {}
        if PREMATCH and history is not None and getattr(pre, "FOLLOWS_RESULTS", False) \
                and getattr(pre, "AS_OF", True):
            publish = {c: prematch_publish_times(prod_by[c], fp) for c, (_, fp) in matches.items()}
            parts = [set(b) for b in batches] if batches else [set(snapshots_by_match)]
            got_rows, code_at = [], {}
            for part in parts:                           # each chunk's matches off its own ends of play
                r_, c_ = known_states([r for r in rows if r["MATCH_CODE"] in part],
                                      {c: p for c, p in publish.items() if c in part}, history,
                                      match_ends({c: s for c, s in snapshots_by_match.items() if c in part}))
                got_rows += r_
                code_at.update(c_)
            rows = got_rows
        means = base.means(rows, results=history)
        for (code, publish), at in code_at.items():
            if at != code and at in means:
                known.setdefault(code, {})[publish] = tuple(means[at])
    LAST_TIMING.update(prematch=time.time() - t0, cuts=sum(len(set(k.values())) for k in known.values()))
    items = []
    for code, (snaps, first_play) in matches.items():
        pair, teams = v21.match_people(code, snaps, info)
        if handles and code in handles and not all(pair):
            pair = tuple((x or "").strip().upper() for x in handles[code])
        target = tuple(float(x) for x in means[code]) if code in means else None
        items.append((code, snaps, prod_by[code], first_play, pair, teams, target, known.get(code)))
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    if not items:
        return [] if isinstance(lines, str) else {mode: [] for mode in modes}
    workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 1), len(items)))
    t0 = time.time()
    if workers == 1:
        out = _worker((items, model_file, n_paths, lines, prematch_only, share, react))
    else:
        # a match at a time to whichever worker is free (a worker handed a fixed share kept the rest
        # waiting on its longest matches), the rows put back in the order the shares gave them: each
        # worker's every workers-th match, one after another
        with v14.pool_context().Pool(workers, initializer=_load, initargs=(model_file,)) as pool:
            per = pool.map(_one, [(item, model_file, n_paths, lines, prematch_only, share, react) for item in items],
                           chunksize=1)
        order = [k for i in range(workers) for k in range(i, len(items), workers)]
        out = {mode: [row for k in order for row in per[k][mode]] for mode in modes}
    LAST_TIMING.update(simulate=time.time() - t0, workers=workers)
    return out[lines] if isinstance(lines, str) else out
