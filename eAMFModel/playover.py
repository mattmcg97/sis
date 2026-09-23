"""Backtest on PLAY_OVER snapshots from SCOUTING_FULL.

Input: the `scouting_playover.csv` that `python -m eAMFCalibrator scouting`
writes -- one row per PLAY_OVER message GAMEPLAI_STREAM quoted, with the
game clock, the state, the score, the final, and prod's line, probability,
liveness and outcome on each of the six markets.

Each snapshot is priced from what is known at that PLAY_OVER, on the REAL
game clock, at prod's line on each market, and scored against the same
outcome as prod. Only prod quotes that were live are compared.

What state a PLAY_OVER is priced from depends on how the play ended:

  SCRIMMAGE            the next PLAY_STARTED's side, down, distance and
                       field: the result of the play, as the game shows it
                       at the whistle -- a turnover included. `--state
                       over` uses the PLAY_OVER row's own state instead
                       (and a fresh drive for the other side after a
                       turnover)
  TOUCHDOWN            the extra point still to come for the scorer, then a
                       fresh drive for the other side
  FIELD_GOAL, PUNT, TURNOVER_ON_DOWNS, SAFETY, KICKOFF, CONVERSION
                       a fresh drive for whoever starts next

The scoreboard at a PLAY_OVER comes from SCORE_CHANGES at or before its
message. A touchdown's six (or a field goal's three) can land a message or
two after the PLAY_OVER; when the scorer has not moved since the play
started, the points are added.
"""

import csv
from collections import Counter, defaultdict

from . import backtest, pricer
from .pricer import AWAY, HOME, GameState

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


def state_for(r, state_mode=NEXT):
    """(GameState, None) or (None, reason it cannot be priced)."""
    a_side = r["team_a_side"] or None
    if a_side is None:
        return None, "team_a_unresolved"
    period = _int(r["period"])
    clock_s = _float(r["clock_seconds"])
    if period is None or clock_s is None:
        return None, "no_clock"
    kind = r["play_kind"]
    home, away = _int(r["score_p1"]) or 0, _int(r["score_p2"]) or 0
    offense = side_of(r["offense"], a_side)
    nxt_offense = side_of(r["next_offense"], a_side)
    opening = side_of(r.get("opening_offense"), a_side)
    base = dict(period=period, elapsed_in_period=0.0, home_score=home, away_score=away,
                clock_seconds=clock_s, opening_receiver=opening)

    if kind == "SCRIMMAGE":
        turnover = nxt_offense is not None and offense is not None and nxt_offense != offense
        if turnover and state_mode == OVER:
            # Intercepted or fumbled: a drive for the other side is next.
            return GameState(offense=nxt_offense, **base), None
        use_next = state_mode == NEXT and nxt_offense is not None and _int(r["next_down"]) is not None
        src = "next_" if use_next else ""
        side = nxt_offense if use_next else offense
        down, dist_, field = (_int(r[src + "down"]), _int(r[src + "distance"]),
                              _int(r[src + "field_position"]))
        if side is None or down is None or dist_ is None or field is None:
            return None, "no_state"
        return GameState(offense=side, down=down, distance=dist_, field_position=field,
                         **base), None
    if kind == "TOUCHDOWN":
        if offense is None:
            return None, "no_state"
        home, away = _on_the_board(r, home, away, offense, 6)
        other = AWAY if offense == HOME else HOME
        return GameState(offense=other, pending_conversion=offense,
                         **dict(base, home_score=home, away_score=away)), None
    messages = set((r.get("play_messages") or "").split("|"))
    if kind == "FIELD_GOAL" and offense is not None and messages & {
            "FIELD_GOAL_GOOD_TEAM_A", "FIELD_GOAL_GOOD_TEAM_B"}:
        home, away = _on_the_board(r, home, away, offense, 3)
        base.update(home_score=home, away_score=away)
    # Every other kind ends with a fresh drive for whoever starts next.
    if nxt_offense is None:
        return None, "no_next_offense"
    return GameState(offense=nxt_offense, **base), None


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


def run(path, versions, state_mode=NEXT, scrimmage_only=False, limit=None,
        require_live=True):
    data = load(path)
    models = {name: pricer.Model(params) for name, params in versions.items()}
    graded = []
    skipped = Counter()
    for i, (match_code, match_rows) in enumerate(sorted(data.items())):
        if limit is not None and i >= limit:
            break
        priors = {name: prior_for(model, match_rows) for name, model in models.items()}
        if any(p is None for p in priors.values()):
            skipped["no_prior"] += 1
            continue
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
            key = row.message
            probs = {}
            for name, model in models.items():
                if (name, key) not in books:
                    books[(name, key)] = model.book(priors[name], state)
                probs[name] = books[(name, key)].prob(row.market_id, row.prod_line)
            row.period = state.period
            graded.append((match_code, row, probs))
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
