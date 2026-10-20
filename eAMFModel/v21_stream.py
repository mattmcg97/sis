"""v2.1 as a price stream: its quotes at every snapshot, shaped like prod's, for the calibrator
(`python -m eAMFCalibrator ... --candidate v21`).

Each match is priced as v21-report prices it. From kick-off the sim plays the match's gamers and
teams, each side's offense nudged NUDGE_SHARE of the way from the sim's own means to the prior's
expected points (v21.nudge_solve). The prior, as v21-report's --prior-model: by default the form table
(DEFAULT_PRIOR_TABLE: the walkthrough's avg table moved half way to a rating drifting 4 points^2 a day,
as `python -m eAMFCalibrator form-prior` writes it), else a v13/v14 build's pre-match model. In play, every PLAY_OVER the sim can start from is priced from there with the same
nudge: a snap (v21.match_starts) or a kick-off after a score.

A touchdown's PLAY_OVER, with its conversion still to come, has no book, because the sim has no
start there. Prod's quotes until the conversion go unpaired. A prod quote reads the book of the
PLAY_OVER before it while that state lasts (stream.confident_windows), and only those books are
simulated. Given `moments` (the betting simulation's: each bet's market and the prod message it reads),
only the books those bets read are simulated, and the quotes are those books' alone.

With an in-play strength update (v21_react: a candidate named v21-snapT or v21-ptsT), every in-play
state gets its own nudge, the kick-off's moved by what the match has shown of each side by then. With
the second strength lever (a candidate named v21-shapeK, v21.SHAPE_TRAIL), the favourite's strength gap
is played by the score at kick-off and in every state: wider when it is behind, narrower when it is two
scores up. With the gap shrink (a candidate named v21-gapL), the prior's expected margin is taken L times
over, its total kept, before the nudge meets it: the prior's big gaps come out at about half (September)
to three quarters (October) in the real finals. With pace when ahead (a candidate named v21-pace,
v21_pace), each side's clock while it is ahead is drawn on its gamer's own extra time tilt there, the
table v21-pace-fit writes beside the build. With the late correction (a candidate named v21-late, v21_late),
each in-play book of Q4 with the sides 8 or fewer apart has the side with the ball's chance moved by its
match day's rolling shift for its lead and the time left, the table v21-late-fit writes beside the build."""

import os
import re
import time
import zlib
from bisect import bisect_left, bisect_right

import numpy as np

from . import playover, sim14 as s14, v14, v21, v21_late, v21_pace, v21_react
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


_SHAPE = re.compile(r"^shape(\d+(?:\.\d+)?)$")


def shape_of(name):
    """The second strength lever a candidate's name asks for ('v21-shape0.25' -> 0.25, v21.SHAPE_TRAIL), or None."""
    _, _, tag = str(name or "").partition("-")
    m = _SHAPE.match(tag.strip().lower())
    return float(m.group(1)) if m and float(m.group(1)) > 0 else None


_GAP = re.compile(r"^gap(\d+(?:\.\d+)?)$")


def gap_of(name):
    """The gap shrink a candidate's name asks for ('v21-gap0.75' -> 0.75: the prior's expected margin x 0.75,
    its total kept), or None (v21 itself, or 1)."""
    _, _, tag = str(name or "").partition("-")
    m = _GAP.match(tag.strip().lower())
    return float(m.group(1)) if m and float(m.group(1)) != 1.0 else None


def pace_of(name):
    """Whether a candidate's name asks for pace when ahead ('v21-pace')."""
    _, _, tag = str(name or "").partition("-")
    return tag.strip().lower() == "pace"


def late_of(name):
    """Whether a candidate's name asks for the late Q4 correction ('v21-late')."""
    _, _, tag = str(name or "").partition("-")
    return tag.strip().lower() == "late"


def late_table(model_dir):
    """{day: {cell: shift}} beside the build (v21_late.LATE_FILE); stops with how to fit it."""
    table = v21_late.load(_model_file(model_dir))
    if table is None:
        raise SystemExit(f"v21-late reads each day's late-Q4 shifts, {v21_late.LATE_FILE}, beside the build"
                         f" ({os.path.dirname(_model_file(model_dir))}): fit it on v21-endgame's late states --\n"
                         "  python -m eAMFModel v21-late-fit --model <the build> --states <v21_endgame_states.csv> ...")
    return table


def pace_table(model_dir):
    """{gamer: extra time tilt ahead} beside the build (v21_pace.PACE_FILE); stops with how to fit it."""
    table = v21_pace.load(_model_file(model_dir))
    if table is None:
        raise SystemExit(f"v21-pace reads each gamer's pace ahead, {v21_pace.PACE_FILE}, beside the build"
                         f" ({os.path.dirname(_model_file(model_dir))}): fit it once on real games before the window --\n"
                         "  python -m eAMFModel v21-pace-fit --model <the build> --until <the window's start>")
    return table


def shrunk(points, gap):
    """(home, away) expected points with their margin x gap and their total kept; as given with no gap."""
    if gap is None:
        return tuple(float(x) for x in points)
    h, a = float(points[0]), float(points[1])
    total, margin = h + a, (h - a) * float(gap)
    return ((total + margin) / 2, (total - margin) / 2)


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


def _kickoff(model, pair, teams, target, n_paths, code, share, shape=0.0, pace=None):
    """(kick-off margin and total distributions, the nudge alpha or None, each side's simulated mean
    and variance of its points). shape: the second strength lever (v21.SHAPE_TRAIL), with the nudge;
    pace: each side's extra time tilt while ahead (v21_pace), None off."""
    start = v21.kickoff_start(model, pair, teams)
    if pace is not None:
        start["pace"] = np.tile(np.asarray(pace, float), (len(start["period"]), 1))
    if target is not None and share and getattr(model, "nudge_dir", None) is not None:
        (h, a), alpha = v21.nudge_solve(model, start, target, n_paths, seed=_seed(code), share=share, shape=shape or 0.0)
    else:
        (h, a), alpha = v21.simulate(model, start, n_paths, seed=_seed(code)), None
    h, a = h.ravel(), a.ravel()
    return v14._distributions(h, a), alpha, dict(mean=(h.mean(), a.mean()), var=(h.var(), a.var()))


def match_books(model, snaps, pair, teams, n_paths, code, alpha, prod_rows, react=None, kick=None, shape=0.0,
                reads=None, tally=None, pace=None, late=None):
    """(books, windows): v21's margin and total distributions at the PLAY_OVERs the match's prod
    quotes read, every other PLAY_OVER keyed with None. `react` (v21_react.spec_of): each state's
    nudge moved by the match so far (kick: the kick-off's points, for the points update). shape: the
    second strength lever, its edge the kick-off's margin. `reads` ((market, prod message) per bet):
    only the books those bets read (_read). `tally` (a dict): the books simulated and the ones prod's
    quotes read, added to "priced" and "of". `pace`: each side's extra time tilt while ahead (v21_pace).
    `late`: the match day's late-Q4 shifts ({cell: shift}, v21_late), each Q4 book within 8 points moved by its."""
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
        if shape and alpha is not None and kick is not None:
            n = len(S["period"])
            S["edge"] = np.full(n, float(kick["mean"][0] - kick["mean"][1]))
            S["shape"] = np.full(n, float(shape))
        if pace is not None:
            S["pace"] = np.tile(np.asarray(pace, float), (len(S["period"]), 1))
        sh, sa = v21.simulate(model, S, n_paths, seed=_seed(code) + 1)
        for k, (m, s) in enumerate(sel):
            mp_, tp = v14._distributions(sh[k], sa[k])
            if late:
                team = int(s["team"][0])
                lead = int(s["home"][0] - s["away"][0]) * (1 if team == 0 else -1)
                mp_ = v21_late.adjust(mp_, team == 0, v21_late.shift_of(late, s["period"][0], s["clock"][0], lead),
                                      v14.MARGIN_MAX)
            books[m] = (m, mp_, tp)
    return [books[m] for m in keys], windows


def _price(model, item, n_paths, lines, prematch_only, share, react=None, shape=0.0):
    """One match's quote rows by line mode: the kick-off (and the pre-match quotes published before
    results the prior's form follows came in), then the PLAY_OVERs prod's quotes read (or only the
    bets', the item's reads). Under "_states": (books simulated, books prod's quotes read)."""
    code, snaps, prod_rows, first_play, pair, teams, target, known, reads, pace, late = item
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    kick, alpha, kstats = _kickoff(model, pair, teams, target, n_paths, code, share, shape, pace)
    kickoff = kick if PREMATCH else None
    if kickoff is not None and known:
        kicks = {m: _kickoff(model, pair, teams, m, n_paths, code, share, shape, pace)[0]
                 for m in sorted(set(known.values()))}
        at = {publish: kicks[m] for publish, m in known.items()}
        kickoff = (lambda row, at=at, own=kick: at.get(row[2], own))
    tally = {}
    if side_known(snaps) and not prematch_only:
        books, windows = match_books(model, snaps, pair, teams, n_paths, code, alpha, prod_rows, react, kstats, shape,
                                     reads, tally, pace, late)
    else:
        books, windows = [], {}
    out = {mode: quote_rows(code, books, prod_rows, first_play, mode, windows, kickoff) for mode in modes}
    out["_states"] = (tally.get("priced", 0), tally.get("of", 0))
    return out


def _worker(job):
    """Price a list of matches, one after another (_price)."""
    items, model_file, n_paths, lines, prematch_only, share, react, shape = job
    model = v21.Model.load(model_file)
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    out, states = {mode: [] for mode in modes}, (0, 0)
    for item in items:
        got = _price(model, item, n_paths, lines, prematch_only, share, react, shape)
        for mode in modes:
            out[mode].extend(got[mode])
        states = tuple(a + b for a, b in zip(states, got["_states"]))
    out["_states"] = states
    return out


_LOADED = {}            # a pool worker's model, read once for every match it is handed (_load)


def _load(model_file):
    _LOADED.clear()
    _LOADED[model_file] = v21.Model.load(model_file)


def _one(job):
    """A pool worker: one match, off the model it loaded."""
    item, model_file, n_paths, lines, prematch_only, share, react, shape = job
    if model_file not in _LOADED:
        _load(model_file)
    return _price(_LOADED[model_file], item, n_paths, lines, prematch_only, share, react, shape)


LAST_TIMING = {}        # the last quotes_for_matches call's seconds: prematch, simulate (and workers)


def quotes_for_matches(snapshots_by_match, prod_quote_rows, model_dir=None, n_paths=DEFAULT_PATHS,
                       workers=None, match_info=None, lines=OWN, history=None, prematch_only=False,
                       prior_dir=None, nudge_share=None, handles=None, batches=None, react=None, shape=None,
                       moments=None, gap=None, pace=False, late=False):
    """Quote rows for many matches (v14_stream.quotes_for_matches's contract). `match_info`: each
    match's players and teams (and the prior's inputs); `history`: settled matches, for a prior whose
    form follows results; `nudge_share`: how far toward the prior's expected points (NUDGE_SHARE;
    0: the sim's own means, no prior needed); `batches`: the calibrator's chunks priced here together,
    each match's earlier pre-match states read off its own chunk's ends of play, as a call for that
    chunk alone reads them; `react`: the in-play strength update (v21_react.spec_of; None: none); `shape`:
    the second strength lever's kappa (v21.SHAPE_TRAIL, shape_of; None: off); `moments`: {match: (market,
    prod message) per bet}, only the in-play books those bets read simulated (None: every book prod's
    quotes read; a match without any: its pre-match prices alone); `gap`: the gap shrink, the prior's
    expected margin x gap with its total kept, at kick-off and in every earlier pre-match state (gap_of;
    None: as the prior says); `pace`: pace when ahead, each side's clock while it is ahead on its gamer's own
    time tilt there (pace_table, v21_pace; False: off); `late`: the late Q4 correction, each match day's
    rolling shifts (late_table, v21_late; False: off)."""
    share = NUDGE_SHARE if nudge_share is None else float(nudge_share)
    prior_tables, model_file = model_paths(model_dir, prior_dir)
    paces = pace_table(model_file) if pace else None
    lates = late_table(model_file) if late else None
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
                known.setdefault(code, {})[publish] = shrunk(means[at], gap)
    LAST_TIMING.update(prematch=time.time() - t0, cuts=sum(len(set(k.values())) for k in known.values()))
    items = []
    for code, (snaps, first_play) in matches.items():
        pair, teams = v21.match_people(code, snaps, info)
        if handles and code in handles and not all(pair):
            pair = tuple((x or "").strip().upper() for x in handles[code])
        target = shrunk(means[code], gap) if code in means else None
        reads = None if moments is None else set(moments.get(code, ()))
        side_pace = v21_pace.side_pace(paces, pair) if paces is not None else None
        late_day = (v21_late.day_shifts(lates, code, info.get(code, {}).get("SCHEDULED_START_TIME_UTC"))
                    if lates is not None else None)
        items.append((code, snaps, prod_by[code], first_play, pair, teams, target, known.get(code), reads, side_pace,
                      late_day))
    if lates is not None:                            # the matches whose day the late table has
        LAST_TIMING.update(late_matches=sum(1 for it in items if it[10] is not None), late_of=len(items))
    if share and pre is not None:                    # which prior, and how many of the matches it has
        LAST_TIMING.update(prior=prior_tables, prior_matches=sum(1 for it in items if it[6] is not None),
                           matches=len(items))
    modes = (lines,) if isinstance(lines, str) else tuple(lines)
    if not items:
        return [] if isinstance(lines, str) else {mode: [] for mode in modes}
    workers = max(1, min(workers or max(1, (os.cpu_count() or 2) - 1), len(items)))
    t0 = time.time()
    if workers == 1:
        out = _worker((items, model_file, n_paths, lines, prematch_only, share, react, shape or 0.0))
        states = out.pop("_states")
    else:
        # a match at a time to whichever worker is free (a worker handed a fixed share kept the rest
        # waiting on its longest matches), the rows put back in the order the shares gave them: each
        # worker's every workers-th match, one after another
        with v14.pool_context().Pool(workers, initializer=_load, initargs=(model_file,)) as pool:
            per = pool.map(_one, [(item, model_file, n_paths, lines, prematch_only, share, react, shape or 0.0) for item in items],
                           chunksize=1)
        order = [k for i in range(workers) for k in range(i, len(items), workers)]
        out = {mode: [row for k in order for row in per[k][mode]] for mode in modes}
        states = tuple(sum(p["_states"][j] for p in per) for j in (0, 1))
    LAST_TIMING.update(simulate=time.time() - t0, workers=workers, states=states[0], states_of=states[1],
                       moments=moments is not None)
    return out[lines] if isinstance(lines, str) else out
