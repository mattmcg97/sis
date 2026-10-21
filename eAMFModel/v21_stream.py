"""v2.1 as a price stream: its quotes at every snapshot, shaped like prod's, for the calibrator
(`python -m eAMFCalibrator ... --candidate v21`).

Each match is priced as v21-report prices it. From kick-off the sim plays the match's gamers and
teams, each side's offense nudged NUDGE_SHARE of the way from the sim's own means to the prior's
expected points (v21.nudge_solve). The prior, as v21-report's --prior-model: by default the form table
(DEFAULT_PRIOR_TABLE: the walkthrough's avg table moved half way to a rating drifting 4 points^2 a day,
as `python -m eAMFCalibrator form-prior` writes it), else a v14 build's pre-match model. In play, every PLAY_OVER the sim can start from is priced from there with the same
nudge: a snap (v21.match_starts) or a kick-off after a score.

A touchdown's PLAY_OVER, with its conversion still to come, has no book, because the sim has no
start there. Prod's quotes until the conversion go unpaired. A prod quote reads the book of the
PLAY_OVER before it while that state lasts (stream.confident_windows), and only those books are
simulated. Given `moments` (the betting simulation's: each bet's market and the prod message it reads),
only the books those bets read are simulated, and the quotes are those books' alone.

With an in-play strength update (v21_react: a candidate named v21-snapT or v21-ptsT), every in-play
state gets its own nudge, the kick-off's moved by what the match has shown of each side by then.
With the pace nudge (a candidate named v21-drives, or v21-drivesS for S of the way from the sim's own),
one tempo on both sides' time tilts is solved with the points nudge so the kick-off games' drives meet the
match's expected drives (v21.drives_solve), and carried into every in-play state as the nudge is: the
expected drives from the pre-match pace table (DEFAULT_DRIVES_TABLE, xgb/pace_eff.py's exp_drives; a match
it lacks priced as v21), relative to the table's mean, at the sim's own level (v21.drive_level)."""

import os
import re
import time
import zlib
from bisect import bisect_left, bisect_right

import numpy as np

from . import playover, progress, sim14 as s14, v14, v21, v21_react
from .state import HOME, AWAY, MARKET_IDS
from .stream import (confident_windows, known_states, match_ends, prematch_publish_times, side_known)
# the line rules, as v14's (the calibrator reads every one of LINE_MODES off one simulation)
from .v14_stream import (EVEN, HELD, LINE_MODES, OWN, PROD_LINES, ANCHORED, _before_play, _quote,  # noqa: F401
                         paired_prod_rows)

DEFAULT_MODEL_DIR = "v21_model"
DEFAULT_PRIOR_DIR = "v14_model"
# the default prior: the walkthrough's avg table, its margin moved half way to a rating drifting 4 points^2 a
# day (`python -m eAMFCalibrator form-prior`'s default); it beat the v14 glmer build on the Oct bets
DEFAULT_PRIOR_TABLE = os.path.join("eAMFCalibrator", "out", "form_prior", "priors", "avg_form4_w05.csv")
DEFAULT_PATHS = 1000          # games a snapshot (the kick-off: each side receiving, so twice that)
NUDGE_SHARE = 0.75            # how far toward the prior's expected points (v21-report's --nudge-share)
# the pace nudge's expected drives a match: xgb/pace_eff.py's table (MATCH_CODE, exp_drives)
DEFAULT_DRIVES_TABLE = os.path.join("xgb", "out", "pace_eff", "pace_eff_prior.csv")
PREMATCH = True
# Every match's sim is seeded by the match (_seed): its quotes are the same whichever matches it is
# priced with, so the calibrator prices several of its chunks in one pool (snowflake_io._sim_quotes).
MATCH_SEEDED = True
PRICES_MOMENTS = True         # quotes_for_matches takes `moments`: only the books the bets read


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


def _prior_candidates():
    """Where v21 looks for its prior when none is named, in order: $EAMF_V21_PRIOR, the form table
    (DEFAULT_PRIOR_TABLE, here or beside the package), $EAMF_V14_MODEL, ./v14_model."""
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    return [os.environ.get("EAMF_V21_PRIOR"), DEFAULT_PRIOR_TABLE, os.path.join(here, DEFAULT_PRIOR_TABLE),
            os.environ.get("EAMF_V14_MODEL"), DEFAULT_PRIOR_DIR, os.path.join(here, DEFAULT_PRIOR_DIR)]


def default_prior():
    """The prior v21 nudges toward when none is named (a table's path or a build's directory), or None."""
    return next((d for d in _prior_candidates() if d and _prior_tables(d) is not None), None)


def _prior_tables(prior_dir=None):
    """The prior build's tables (its pre-match model), or a prior table (a walkthrough variant's CSV,
    table_prior.py) as it is; None where none is found."""
    from . import prior_check, table_prior
    for d in ([prior_dir] if prior_dir else _prior_candidates()):
        if table_prior.is_table(d):
            return d
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


_DRIVES = re.compile(r"^drives(\d+(?:\.\d+)?)?$")


def drives_of(name):
    """The pace nudge a candidate's name asks for: the share of the way from the sim's own drives to the
    expected ('v21-drives' -> 1.0, 'v21-drives0.5' -> 0.5), or None."""
    _, _, tag = str(name or "").partition("-")
    m = _DRIVES.match(tag.strip().lower())
    if not m:
        return None
    share = float(m.group(1)) if m.group(1) else 1.0
    return share if share > 0 else None


def drives_table(path=None):
    """{match: expected drives} off the pace table (path, else $EAMF_V21_DRIVES, else DEFAULT_DRIVES_TABLE
    here or beside the package); stops with how to write it."""
    import csv
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    tried = [path] if path else [os.environ.get("EAMF_V21_DRIVES"), DEFAULT_DRIVES_TABLE,
                                 os.path.join(here, DEFAULT_DRIVES_TABLE)]
    for p in tried:
        if p and os.path.exists(p):
            with open(p, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if rows and "exp_drives" not in rows[0]:
                raise SystemExit(f"{p}: no exp_drives column (xgb/pace_eff.py's pace_eff_prior.csv has one)")
            return {r["MATCH_CODE"]: float(r["exp_drives"]) for r in rows if r.get("exp_drives") not in ("", None)}
    raise SystemExit("the pace nudge reads each match's expected drives, and no table was found in "
                     f"{path or ', '.join(p for p in tried if p)}: write it --\n"
                     "  python -m xgb.pace_eff --features xgb/out/features.csv --drives xgb/data/drive_stats.csv"
                     " --since <six months before> --score-from <the window's start>\n"
                     "then point --v21-drives (or EAMF_V21_DRIVES) at its pace_eff_prior.csv")


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


def _read(keys, prod_quote_rows, windows, reads):
    """The PLAY_OVER messages whose books the bets read. `reads`: (market, prod message) per bet; each
    reads the latest quote of its market at or before that message (the calibrator's bets.price_at),
    which is the book of the latest prod quote there that falls in a book's window -- none (the
    kick-off's price) when that quote is before the first PLAY_OVER."""
    by_market = {}
    for market_id, message in paired_prod_rows(prod_quote_rows):
        if market_id in MARKET_IDS:
            by_market.setdefault(market_id, []).append(int(message))
    for msgs in by_market.values():
        msgs.sort()
    need = set()
    for market_id, message in reads:
        msgs = by_market.get(int(market_id), [])
        j = bisect_right(msgs, int(message)) - 1
        while j >= 0:
            i = bisect_right(keys, msgs[j]) - 1
            if i < 0:
                break
            if keys[i] in windows and msgs[j] < windows[keys[i]]:
                need.add(keys[i])
                break
            j -= 1
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


def _kickoff(model, pair, teams, target, n_paths, code, share, drives=None):
    """(kick-off margin and total distributions, the nudge alpha or None, each side's simulated mean
    and variance of its points, and the pace nudge's tempo under "tempo"). drives: (the match's expected
    drives over the table's mean, the share of the way there), the pace nudge (v21.drives_solve) to that
    times the sim's own level (v21.drive_level), None off."""
    start = v21.kickoff_start(model, pair, teams)
    nudging = target is not None and share and getattr(model, "nudge_dir", None) is not None
    tempo = None
    if drives is not None:
        (h, a), alpha, tempo, _ = v21.drives_solve(model, start, target if nudging else None,
                                                   drives[0] * v21.drive_level(model)[0], n_paths, seed=_seed(code),
                                                   share=share, drives_share=drives[1])
        alpha = alpha if nudging else None
    elif nudging:
        (h, a), alpha = v21.nudge_solve(model, start, target, n_paths, seed=_seed(code), share=share)
    else:
        (h, a), alpha = v21.simulate(model, start, n_paths, seed=_seed(code)), None
    h, a = h.ravel(), a.ravel()
    return v14._distributions(h, a), alpha, dict(mean=(h.mean(), a.mean()), var=(h.var(), a.var()), tempo=tempo)


def match_books(model, snaps, pair, teams, n_paths, code, alpha, prod_rows, react=None, kick=None, reads=None,
                tally=None):
    """(books, windows): v21's margin and total distributions at the PLAY_OVERs the match's prod
    quotes read, every other PLAY_OVER keyed with None. `react` (v21_react.spec_of): each state's
    nudge moved by the match so far (kick: the kick-off's points, for the points update). `reads`
    ((market, prod message) per bet): only the books those bets read (_read). `tally` (a dict): the books
    simulated and the ones prod's quotes read, added to "priced" and "of". The kick-off's pace nudge
    (kick's "tempo") is carried into every state."""
    rows = [_as_text(r) for r in sorted(snaps, key=lambda r: int(r["message"]))]
    starts = state_starts(model, rows, pair, teams)
    keys = [m for m, _ in starts]
    windows = confident_windows(snaps, keys)
    need = _needed(keys, prod_rows, windows)
    of = len(need)
    if reads is not None:
        need &= _read(keys, prod_rows, windows, reads)
    if tally is not None:
        tally["priced"] = tally.get("priced", 0) + len(need)
        tally["of"] = tally.get("of", 0) + of
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
        if kick is not None and kick.get("tempo") is not None:     # the pace nudge's, as the kick-off's
            S["tempo"] = np.full(len(S["period"]), float(kick["tempo"]))
        sh, sa = v21.simulate(model, S, n_paths, seed=_seed(code) + 1)
        for k, (m, _) in enumerate(sel):
            books[m] = (m,) + tuple(v14._distributions(sh[k], sa[k]))
    return [books[m] for m in keys], windows


def _price(model, item, n_paths, lines, prematch_only, share, react=None):
    """One match's quote rows by line mode: the kick-off (and the pre-match quotes published before
    results the prior's form follows came in), then the PLAY_OVERs prod's quotes read (or only the
    bets', the item's reads). Under "_states": (books simulated, books prod's quotes read)."""
    code, snaps, prod_rows, first_play, pair, teams, target, known, reads, drives = item
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    kick, alpha, kstats = _kickoff(model, pair, teams, target, n_paths, code, share, drives)
    kickoff = kick if PREMATCH else None
    if kickoff is not None and known:
        kicks = {m: _kickoff(model, pair, teams, m, n_paths, code, share, drives)[0]
                 for m in sorted(set(known.values()))}
        at = {publish: kicks[m] for publish, m in known.items()}
        kickoff = (lambda row, at=at, own=kick: at.get(row[2], own))
    tally = {}
    if side_known(snaps) and not prematch_only:
        books, windows = match_books(model, snaps, pair, teams, n_paths, code, alpha, prod_rows, react, kstats, reads,
                                     tally)
    else:
        books, windows = [], {}
    out = {mode: quote_rows(code, books, prod_rows, first_play, mode, windows, kickoff) for mode in modes}
    out["_states"] = (tally.get("priced", 0), tally.get("of", 0))
    return out


def _worker(job, tick=None):
    """Price a list of matches, one after another (_price); tick() after each."""
    items, model_file, n_paths, lines, prematch_only, share, react = job
    model = v21.Model.load(model_file)
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    out, states = {mode: [] for mode in modes}, (0, 0)
    for item in items:
        got = _price(model, item, n_paths, lines, prematch_only, share, react)
        for mode in modes:
            out[mode].extend(got[mode])
        states = tuple(a + b for a, b in zip(states, got["_states"]))
        if tick is not None:
            tick()
    out["_states"] = states
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


def _one_at(indexed):
    """_one for (k, job): (k, its rows), so the pool can hand them back as they finish."""
    k, job = indexed
    return k, _one(job)


LAST_TIMING = {}        # the last quotes_for_matches call's seconds: prematch, simulate (and workers)


def quotes_for_matches(snapshots_by_match, prod_quote_rows, model_dir=None, n_paths=DEFAULT_PATHS,
                       workers=None, match_info=None, lines=OWN, history=None, prematch_only=False,
                       prior_dir=None, nudge_share=None, handles=None, batches=None, react=None, moments=None,
                       drives=None, drives_path=None, label=None):
    """Quote rows for many matches (v14_stream.quotes_for_matches's contract). `match_info`: each
    match's players and teams (and the prior's inputs); `history`: settled matches, for a prior whose
    form follows results; `nudge_share`: how far toward the prior's expected points (NUDGE_SHARE;
    0: the sim's own means, no prior needed); `batches`: the calibrator's chunks priced here together,
    each match's earlier pre-match states read off its own chunk's ends of play, as a call for that
    chunk alone reads them; `react`: the in-play strength update (v21_react.spec_of; None: none);
    `moments`: {match: (market, prod message) per bet}, only the in-play books those bets read simulated
    (None: every book prod's quotes read; a match without any: its pre-match prices alone); `drives`: the
    pace nudge, the share of the way from the sim's own drives to each match's expected drives (drives_of;
    None: off), those off `drives_path` (drives_table's default where None); `label`: the candidate's name
    on the simulation's progress bar."""
    share = NUDGE_SHARE if nudge_share is None else float(nudge_share)
    prior_tables, model_file = model_paths(model_dir, prior_dir)
    exp_drives = drives_table(drives_path) if drives else None
    level = float(np.mean(list(exp_drives.values()))) if exp_drives else None
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
            "v21 prices toward a prior's expected points (its nudge), and no prior was found in"
            f" {prior_dir or ', '.join([DEFAULT_PRIOR_TABLE, DEFAULT_PRIOR_DIR])}: build the default -- python -m"
            " eAMFCalibrator walkthrough --history eAMFCalibrator/out/match_history.csv --since <a month before>,"
            " then python -m eAMFCalibrator form-prior -- or point --v21-prior-model (or EAMF_V21_PRIOR) at a"
            " table or a v14 build made with --history, or price without the nudge (--v21-nudge-share 0)")
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
                known.setdefault(code, {})[publish] = tuple(float(x) for x in means[at])
    LAST_TIMING.update(prematch=time.time() - t0, cuts=sum(len(set(k.values())) for k in known.values()))
    items = []
    for code, (snaps, first_play) in matches.items():
        pair, teams = v21.match_people(code, snaps, info)
        if handles and code in handles and not all(pair):
            pair = tuple((x or "").strip().upper() for x in handles[code])
        target = tuple(float(x) for x in means[code]) if code in means else None
        reads = None if moments is None else set(moments.get(code, ()))
        pace_d = (exp_drives[code] / level, float(drives)) if exp_drives is not None and code in exp_drives else None
        items.append((code, snaps, prod_by[code], first_play, pair, teams, target, known.get(code), reads, pace_d))
    if exp_drives is not None:                       # the matches the pace table has
        LAST_TIMING.update(drives_matches=sum(1 for it in items if it[9] is not None), drives_of=len(items))
    if share and pre is not None:                    # which prior, and how many of the matches it has
        LAST_TIMING.update(prior=prior_tables, prior_matches=sum(1 for it in items if it[6] is not None),
                           matches=len(items))
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    if not items:
        return [] if isinstance(lines, str) else {mode: [] for mode in modes}
    workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 1), len(items)))
    t0 = time.time()
    bar = progress.Bar(f"{label or 'v21'}: simulating", len(items), "matches")
    try:
        if workers == 1:
            out = _worker((items, model_file, n_paths, lines, prematch_only, share, react), tick=bar.update)
            states = out.pop("_states")
        else:
            # a match at a time to whichever worker is free (a worker handed a fixed share kept the rest
            # waiting on its longest matches), the rows put back in the order the shares gave them: each
            # worker's every workers-th match, one after another
            jobs = [(item, model_file, n_paths, lines, prematch_only, share, react) for item in items]
            per = [None] * len(jobs)
            with v14.pool_context().Pool(workers, initializer=_load, initargs=(model_file,)) as pool:
                for k, got in pool.imap_unordered(_one_at, enumerate(jobs), chunksize=1):
                    per[k] = got
                    bar.update()
            order = [k for i in range(workers) for k in range(i, len(items), workers)]
            out = {mode: [row for k in order for row in per[k][mode]] for mode in modes}
            states = tuple(sum(p["_states"][j] for p in per) for j in (0, 1))
    finally:
        bar.close()
    LAST_TIMING.update(simulate=time.time() - t0, workers=workers, states=states[0], states_of=states[1],
                       moments=moments is not None)
    return out[lines] if isinstance(lines, str) else out
