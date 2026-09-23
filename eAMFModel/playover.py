"""Backtest on PLAY_OVER snapshots from SCOUTING_FULL.

Input: the `scouting_playover.csv` that `python -m eAMFCalibrator scouting`
writes -- one row per PLAY_OVER message GAMEPLAI_STREAM quoted, with the
game clock, the state, the score, the final, and prod's line, probability,
liveness and outcome on each of the six markets.

Each snapshot is priced from what is known at that PLAY_OVER, on the REAL
game clock, at prod's line on each market, and scored against the same
outcome as prod. Only prod quotes that were live are compared.

What state a PLAY_OVER is priced from (state_for): the row itself, which
carries the state after the play, except after a score, where the row shows
the set-up of what follows -- then the scorer comes off the scoring message:

  TOUCHDOWN            the conversion pending for the scorer, then a fresh
                       drive for the other side
  CONVERSION, good FIELD_GOAL
                       the points on the board, a fresh drive for the other side
  missed FIELD_GOAL    the other side's ball at the spot
  SAFETY               two points, and the scorer receives
  everything else      the row's side, down, distance and field

The scoreboard at a PLAY_OVER comes from SCORE_CHANGES at or before its
message. A touchdown's six (or a field goal's three) can land a message or
two after the PLAY_OVER; when the scorer has not moved since the play
started, the points are added.
"""

import csv
from collections import Counter, defaultdict

from dataclasses import replace

from . import backtest, players, pricer
from .pricer import AWAY, HOME, GameState

# How the in-game pace enters (see _player_effects):
PACE_OFF = "off"      # no pace effect
PACE_NEWS = "news"    # only what the game shows beyond the players' profiles
PACE_FULL = "full"    # the profiles' pace too, from kickoff
# Clock evidence (seconds) before the game's own pace counts for half.
PACE_PRIOR_SECONDS = 600.0

MARKETS = (50, 51, 52, 53, 54, 55)
NEXT = "next"
OVER = "over"


def _int(v):
    try:
        return None if v in ("", None) else int(float(v))
    except ValueError:
        return None


def _float(v):
    try:
        return None if v in ("", None) else float(v)
    except ValueError:
        return None


def load(path):
    by_match = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_match[r["match_code"]].append(r)
    for rows in by_match.values():
        rows.sort(key=lambda r: _int(r["message"]))
    return by_match


def side_of(team, team_a_side):
    """TEAM_A / TEAM_B -> HOME / AWAY, given which side TEAM_A is."""
    if team not in ("TEAM_A", "TEAM_B") or team_a_side not in ("home", "away"):
        return None
    a = HOME if team_a_side == "home" else AWAY
    return a if team == "TEAM_A" else (AWAY if a == HOME else HOME)


def _message_team(messages, prefixes):
    """TEAM_A / TEAM_B off the last message starting with one of prefixes."""
    team = None
    for m in messages:
        if m.startswith(prefixes) and m[-6:] in ("TEAM_A", "TEAM_B"):
            team = m[-6:]
    return team


def _other(side):
    return AWAY if side == HOME else HOME


def state_for(r, state_mode=OVER):
    """(GameState, None) or (None, reason it cannot be priced).

    The PLAY_OVER row carries the state AFTER the play -- it matches the
    next PLAY_STARTED 95% of the time on scrimmage plays, the rest being
    penalties enforced between the two -- so by default the row is the
    state. Scores are the exception: the row then shows the set-up of what
    comes next (the conversion on the 85, the kicker on the 35), so the
    scorer comes off the scoring message and the other side is given the
    ball. `state_mode=NEXT` takes scrimmage states from the next
    PLAY_STARTED instead (a small look ahead, for comparison).
    """
    a_side = r["team_a_side"] or None
    if a_side is None:
        return None, "team_a_unresolved"
    period = _int(r["period"])
    clock_s = _float(r["clock_seconds"])
    if period is None or clock_s is None:
        return None, "no_clock"
    kind = r["play_kind"]
    home, away = _int(r["score_p1"]) or 0, _int(r["score_p2"]) or 0
    row_side = side_of(r["offense"], a_side)
    opening = side_of(r.get("opening_offense"), a_side)
    messages = [m for m in (r.get("play_messages") or "").split("|") if m]
    base = dict(period=period, elapsed_in_period=0.0, home_score=home, away_score=away,
                clock_seconds=clock_s, opening_receiver=opening)

    def snap(side, prefix=""):
        down, dist_, field = (_int(r[prefix + "down"]), _int(r[prefix + "distance"]),
                              _int(r[prefix + "field_position"]))
        if side is None or down is None or dist_ is None or field is None:
            return None, "no_state"
        return GameState(offense=side, down=down, distance=dist_, field_position=field,
                         snap_confirmed=True, **base), None

    def scored(team_prefixes, points_by_message):
        team = _message_team(messages, team_prefixes)
        scorer = side_of(team, a_side) if team else row_side
        points = 0
        for m in messages:
            for prefix, pts in points_by_message.items():
                if m.startswith(prefix):
                    points = pts
        return scorer, points

    if kind == "TOUCHDOWN":
        scorer, _ = scored(("TOUCHDOWN_TEAM",), {})
        if scorer is None:
            return None, "no_state"
        h, a = _on_the_board(r, home, away, scorer, 6)
        return GameState(offense=_other(scorer), pending_conversion=scorer,
                         **dict(base, home_score=h, away_score=a)), None
    if kind == "CONVERSION":
        scorer, points = scored(("EXTRA_POINT", "TWO_POINT"), {
            "EXTRA_POINT_GOOD": 1, "TWO_POINT_CONVERSION_SUCCESSFUL": 2})
        if scorer is None:
            return None, "no_state"
        h, a = _on_the_board(r, home, away, scorer, points) if points else (home, away)
        return GameState(offense=_other(scorer), **dict(base, home_score=h, away_score=a)), None
    if kind == "FIELD_GOAL":
        kicker, points = scored(("FIELD_GOAL_GOOD", "FIELD_GOAL_MISSED"), {"FIELD_GOAL_GOOD": 3})
        if kicker is None:
            return None, "no_state"
        if points:
            h, a = _on_the_board(r, home, away, kicker, 3)
            return GameState(offense=_other(kicker), **dict(base, home_score=h, away_score=a)), None
        if row_side == _other(kicker):
            return snap(row_side)               # missed: their ball at the spot
        return GameState(offense=_other(kicker), **base), None
    if kind == "SAFETY":
        scorer, _ = scored(("SAFETY_AWARDED",), {})
        if scorer is None:
            return None, "no_state"
        h, a = _on_the_board(r, home, away, scorer, 2)
        return GameState(offense=scorer, **dict(base, home_score=h, away_score=a)), None
    if kind == "SCRIMMAGE" and state_mode == NEXT:
        nxt = side_of(r["next_offense"], a_side)
        if nxt is not None and _int(r["next_down"]) is not None:
            return snap(nxt, "next_")
    # Scrimmage, kickoff, punt, turnover on downs: the row is the state.
    return snap(row_side)


def rows_for_match(match_rows):
    """backtest.Row objects (one per live prod market) with their states."""
    out = []
    for r in match_rows:
        for m in MARKETS:
            prob = _float(r.get(f"prob_{m}"))
            outcome = _int(r.get(f"outcome_{m}"))
            if prob is None:
                continue
            row = backtest.Row(
                match_code=r["match_code"], message=_int(r["message"]), period=_int(r["period"]),
                home_score=_int(r["score_p1"]) or 0, away_score=_int(r["score_p2"]) or 0,
                offense=None, field_position=None, down=None, distance=None, market_id=m,
                prod_line=_float(r.get(f"line_{m}")), candidate_line=None,
                prod_probability=prob, candidate_probability=None,
                prod_outcome=outcome, candidate_outcome=None,
                prod_live=_int(r.get(f"live_{m}")), candidate_live=1)
            row.kind = r["play_kind"]
            row.source = r
            out.append(row)
    return out


def prior_for(model, match_rows):
    r = match_rows[0]
    line52, line54 = _float(r.get("prematch_line_52")), _float(r.get("prematch_line_54"))
    if line52 is None or line54 is None:
        # No pre-match quote: fall back on the first snapshot's lines.
        line52, line54 = _float(r.get("line_52")), _float(r.get("line_54"))
        if line52 is None or line54 is None:
            return None
        return model.fit_prior(line52, line54, ml_home=_float(r.get("prob_50")),
                               spread_home=_float(r.get("prob_52")),
                               over=_float(r.get("prob_54")), on_clock=True)
    return model.fit_prior(line52, line54, ml_home=_float(r.get("prematch_prob_50")),
                           spread_home=_float(r.get("prematch_prob_52")),
                           over=_float(r.get("prematch_prob_54")), on_clock=True)


def _pace_by_message(match_rows, book, handles):
    """{message: (pace ratio so far, profile pace)} at each snapshot."""
    pace = (players.Pace(book, *handles, prior_seconds=PACE_PRIOR_SECONDS) if handles
            else players.Pace(book, None, None, prior_seconds=PACE_PRIOR_SECONDS))
    intervals = sorted(players.play_intervals(match_rows), key=lambda x: x[3])
    out = {}
    j = 0
    for r in match_rows:
        m = int(r["message"])
        while j < len(intervals) and intervals[j][3] <= m:
            offense, sit, seconds, _ = intervals[j]
            pace.add(offense, sit, seconds)
            j += 1
        out[m] = (pace.ratio(), pace.profile_pace())
    return out


def _player_effects(state, message, pace_table, profiles, effects):
    """The state with the players' effects on it, per `effects`
    ({"aggression": bool, "milk": bool, "pace": PACE_*})."""
    changes = {}
    home, away = profiles
    if effects.get("aggression"):
        changes.update(home_aggression=home.aggression, away_aggression=away.aggression)
    if effects.get("milk"):
        changes.update(home_milk=home.milk, away_milk=away.milk)
    mode = effects.get("pace", PACE_OFF)
    if mode != PACE_OFF and message in pace_table:
        ratio, profile_pace = pace_table[message]
        # ratio: clock used over what the profiles expected. Drives left go
        # the other way. "full" also charges the profiles' own pace against
        # the league's (the pre-match price assumed a league-average game).
        mult = 1.0 / ratio
        if mode == PACE_FULL:
            mult /= profile_pace
        changes["pace"] = mult
    return replace(state, **changes) if changes else state


def _grade_matches(job):
    """Worker: grade a list of (match_code, rows). Top level, so it pickles."""
    (matches, versions, state_mode, scrimmage_only, require_live, book, handles,
     effects) = job
    models = {name: pricer.Model(params) for name, params in versions.items()}
    graded = []
    skipped = Counter()
    for match_code, match_rows in matches:
        priors = {name: prior_for(model, match_rows) for name, model in models.items()}
        if any(p is None for p in priors.values()):
            skipped["no_prior"] += 1
            continue
        pair = handles.get(match_code) if handles else None
        profiles = ((book.profile(pair[0]), book.profile(pair[1])) if (book and pair)
                    else (players.Profile(), players.Profile()))
        pace_table = _pace_by_message(match_rows, book, pair) if book else {}
        books = {}
        for row in rows_for_match(match_rows):
            src = row.source
            if row.prod_outcome is None:
                skipped["push_or_unresolved"] += 1
                continue
            if require_live and not row.prod_live:
                skipped["prod_not_live"] += 1
                continue
            if scrimmage_only and row.kind != "SCRIMMAGE":
                skipped["not_scrimmage"] += 1
                continue
            state, why = state_for(src, state_mode)
            if state is None:
                skipped[why] += 1
                continue
            probs = {}
            for name, model in models.items():
                key = (name, row.message)
                if key not in books:
                    s = _player_effects(state, row.message, pace_table, profiles,
                                        effects.get(name, {}))
                    books[key] = model.book(priors[name], s)
                probs[name] = books[key].prob(row.market_id, row.prod_line)
            row.period = state.period
            row.clock_seconds = state.clock_seconds
            row.source = None            # the raw row is not needed past here
            graded.append((match_code, row, probs))
    return graded, skipped


def run(path, versions, state_mode=OVER, scrimmage_only=False, limit=None,
        require_live=True, workers=1, matches=None, book=None, handles=None,
        effects=None):
    """[(match, row, {version: probability})], Counter of what was skipped.

    `workers` > 1 splits the matches across processes. `matches`, if given,
    restricts to those match codes (a train / test split, say). `book`
    (players.Book) and `handles` ({match: (home, away)}) switch on player
    effects, per version as `effects[name]` = {"aggression": bool,
    "milk": bool, "pace": "off" | "news" | "full"}.
    """
    data = load(path)
    items = sorted(data.items())
    if matches is not None:
        wanted = set(matches)
        items = [item for item in items if item[0] in wanted]
    if limit is not None:
        items = items[:limit]
    job = (versions, state_mode, scrimmage_only, require_live, book, handles or {},
           effects or {})
    if workers <= 1:
        return _grade_matches((items,) + job)
    import multiprocessing
    chunks = [items[i::workers * 4] for i in range(workers * 4)]
    graded, skipped = [], Counter()
    with multiprocessing.Pool(workers) as pool:
        for g, s in pool.imap_unordered(_grade_matches, [(c,) + job for c in chunks if c]):
            graded.extend(g)
            skipped.update(s)
    graded.sort(key=lambda g: (g[0], g[1].message, g[1].market_id))
    return graded, skipped


def _on_the_board(r, home, away, scorer, points):
    """The score with this play's points in it. SCORE_CHANGES can land a
    message or two after the PLAY_OVER; if the scorer has not moved by
    `points` since the play started, add them."""
    start_home = _int(r.get("score_p1_at_start"))
    start_away = _int(r.get("score_p2_at_start"))
    if start_home is None or start_away is None:
        return home, away
    if scorer == HOME and home - start_home < points:
        return start_home + points, away
    if scorer == AWAY and away - start_away < points:
        return home, start_away + points
    return home, away
