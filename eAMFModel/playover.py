"""Reading the PLAY_OVER export: one game state per snapshot, and the prod prices at it."""

import csv
from collections import Counter, defaultdict

from . import grading
from .state import AWAY, HOME, GameState

MARKETS = (50, 51, 52, 53, 54, 55)
NEXT = "next"
OVER = "over"


def _int(v):
    """An int, or None for a blank."""
    try:
        return None if v in ("", None) else int(float(v))
    except ValueError:
        return None


def _float(v):
    """A float, or None for a blank."""
    try:
        return None if v in ("", None) else float(v)
    except ValueError:
        return None


def load(path):
    """Read the export, grouped by match in message order."""
    by_match = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_match[r["match_code"]].append(r)
    for rows in by_match.values():
        rows.sort(key=lambda r: _int(r["message"]))
    return by_match


TIMEOUTS_PER_HALF = 3


def _half(period):
    """1 for the first half, 2 for the second, the period itself in overtime."""
    return 1 if period <= 2 else 2 if period <= 4 else period


def annotate_timeouts(by_match, path):
    """Mark each PLAY_OVER row with the timeouts around it, off the calibrator's timeouts.csv
    (`python -m eAMFCalibrator timeouts`): the side that called one before the next snap
    (timeout_after: TEAM_A/TEAM_B, blank for none; timeout_prev: the play before it) and how many
    each side had called in the half at the row (timeouts_used_a/b, the scouting export's own
    columns). Returns how many calls found their row."""
    calls = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            m, p = _int(r.get("message")), _int(r.get("period"))
            if m is not None and p is not None and r.get("caller") in ("TEAM_A", "TEAM_B"):
                calls[r["match_code"]].append((m, p, r["caller"], r.get("prev_play") or ""))
    found = 0
    for code, rows in by_match.items():
        mine = sorted(calls.get(code, []))
        used = Counter()
        k = 0
        for i, row in enumerate(rows):
            row["timeouts_used_a"] = used[(_half(_int(row["period"]) or 1), "TEAM_A")]
            row["timeouts_used_b"] = used[(_half(_int(row["period"]) or 1), "TEAM_B")]
            row["timeout_after"], row["timeout_prev"] = "", ""
            msg = _int(row["message"])
            nxt = _int(rows[i + 1]["message"]) if i + 1 < len(rows) else None
            while k < len(mine) and (nxt is None or mine[k][0] < nxt):
                m, p, team, prev = mine[k]
                k += 1
                if msg is None or m < msg:
                    continue
                used[(_half(p), team)] += 1
                if not row["timeout_after"]:
                    row["timeout_after"], row["timeout_prev"] = team, prev
                    found += 1
    return found


def side_of(team, team_a_side):
    """TEAM_A or TEAM_B as home or away."""
    if team not in ("TEAM_A", "TEAM_B") or team_a_side not in ("home", "away"):
        return None
    a = HOME if team_a_side == "home" else AWAY
    return a if team == "TEAM_A" else (AWAY if a == HOME else HOME)


def _message_team(messages, prefixes):
    """The team named by the last message with one of these prefixes."""
    team = None
    for m in messages:
        if m.startswith(prefixes) and m[-6:] in ("TEAM_A", "TEAM_B"):
            team = m[-6:]
    return team


def _other(side):
    """The other side."""
    return AWAY if side == HOME else HOME


def state_for(r, state_mode=OVER):
    """The game state at a snapshot, or the reason it cannot be priced."""
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
    base = dict(period=period, home_score=home, away_score=away, clock_seconds=clock_s,
                opening_receiver=opening)

    def snap(side, prefix=""):
        """The side on the ball and its field position."""
        down, dist_, field = (_int(r[prefix + "down"]), _int(r[prefix + "distance"]),
                              _int(r[prefix + "field_position"]))
        if side is None or down is None or dist_ is None or field is None:
            return None, "no_state"
        return GameState(offense=side, down=down, distance=dist_, field_position=field,
                         snap_confirmed=True, **base), None

    def scored(team_prefixes, points_by_message):
        """Points this play added for each side."""
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
            return snap(row_side)
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
    return snap(row_side)


def rows_for_match(match_rows):
    """One grading row per prod market quoted in a match, with the export row it came from."""
    out = []
    for r in match_rows:
        for m in MARKETS:
            prob = _float(r.get(f"prob_{m}"))
            if prob is None:
                continue
            row = grading.Row(
                match_code=r["match_code"], message=_int(r["message"]), period=_int(r["period"]),
                home_score=_int(r["score_p1"]) or 0, away_score=_int(r["score_p2"]) or 0,
                market_id=m, prod_line=_float(r.get(f"line_{m}")), prod_probability=prob,
                prod_outcome=_int(r.get(f"outcome_{m}")), prod_live=_int(r.get(f"live_{m}")))
            row.kind = r["play_kind"]
            row.source = r
            out.append(row)
    return out


def _on_the_board(r, home, away, scorer, points):
    """The score with this play's points added."""
    start_home = _int(r.get("score_p1_at_start"))
    start_away = _int(r.get("score_p2_at_start"))
    if start_home is None or start_away is None:
        return home, away
    if scorer == HOME and home - start_home < points:
        return start_home + points, away
    if scorer == AWAY and away - start_away < points:
        return home, start_away + points
    return home, away
