"""The scouting feed: its PLAY_OVER rows read into snaps, kicks and conversions, and the game's rules they
are read by. Shared by every version (sim14 and v14 re-export it; v2.1 reads it directly), so no version
needs another's code to read the export.

Each PLAY_OVER row carries the kind of the play that just closed and the state for the next snap (the
offense, down, distance and field position about to be played)."""

import numpy as np

# the game's rules
QUARTER = 240.0
PLAY_CLOCK = 40.0          # the play clock: a running clock's snap comes within it
OT_TIMEOUTS = 2            # each side's timeouts in each overtime period (three a half in regulation)
OFFENCE, DEFENCE = 0, 1    # who called a timeout, as the side with the ball sees it

# what came of a snap
GAIN, TURNOVER, DEF_TD = 0, 1, 2

# kick-offs that take no clock: a touchback, and one landing in no landing zone
KICK_TOUCHBACK, KICK_NO_LANDING = 20, 35

# the conversion after a touchdown: the lead after the six cut at +-CONV_MARGIN; in the fourth quarter's
# last three minutes the clock cut at CONV_CLOCK, then overtime on its own
CONV_MARGIN = 16
CONV_CLOCK = (30.0, 60.0, 120.0)
CONV_CELLS = 4 + len(CONV_CLOCK) + 1


def half_left(period, clock):
    """Seconds left in the half."""
    return clock + (QUARTER if period in (1, 3) else 0.0)


# ---------------------------------------------------------------- the rows

def _i(v):
    """An int, or None for a blank."""
    return int(float(v)) if v not in ("", None) else None


def _f(v):
    """A float, or None for a blank."""
    return float(v) if v not in ("", None) else None


def _scorer(row, prefix):
    """The team named by the last scoring message with this prefix."""
    for m in reversed((row.get("play_messages") or "").split("|")):
        if m.startswith(prefix) and m[-6:] in ("TEAM_A", "TEAM_B"):
            return m[-6:]
    return None


SNAP_KINDS = ("SCRIMMAGE", "KICKOFF", "PUNT", "TURNOVER_ON_DOWNS")
FRESH_KINDS = ("KICKOFF", "PUNT", "TURNOVER_ON_DOWNS")


def is_fresh(row):
    """Whether the next snap after this PLAY_OVER starts a possession with the clock stopped: after
    a kick-off, a punt, a turnover on downs, a missed field goal or a turnover on a scrimmage play
    (the feed's POSSESSION message)."""
    kind = row["play_kind"]
    if kind in FRESH_KINDS or kind == "FIELD_GOAL":
        return True
    return kind == "SCRIMMAGE" and any(m.startswith("POSSESSION")
                                       for m in (row.get("play_messages") or "").split("|"))


def _margin(row, team):
    """A team's lead on a row's scoreboard. The board is home's (score_p1) and away's; TEAM_A is
    whichever side team_a_side says (home when it is not known). Until v14's strength refit the
    lead was read as if TEAM_A were always home, so on a match where it is away every lead the
    tables read -- modes, settle, kneels, 4th downs, kicks, conversions -- had the wrong sign."""
    home, away = _i(row["score_p1"]), _i(row["score_p2"])
    if home is None or away is None or team not in ("TEAM_A", "TEAM_B"):
        return None
    a_home = (row.get("team_a_side") or "home") != "away"
    return (home - away) * (1 if (team == "TEAM_A") == a_home else -1)


def snap_records(rows, label=None):
    """Every scrimmage snap of a match: its state, what came of it, the clock it used and whether
    it started a possession with the clock stopped. `label(rec)`: fields a version adds to each
    record before its timeouts (sim14: its tables' mode and key)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in SNAP_KINDS or not a["down"] or not a["field_position"]:
            continue
        if not a["period"] or a["period"] != b["period"] or not a["clock_seconds"] \
                or not b["clock_seconds"] or not a["distance"]:
            continue
        seconds = _f(a["clock_seconds"]) - _f(b["clock_seconds"])
        if not 0 <= seconds <= 60:
            continue
        down, t, y = _i(a["down"]), max(1, _i(a["distance"])), _i(a["field_position"])
        rec = dict(offense=a["offense"], period=_i(a["period"]), clock=_f(a["clock_seconds"]),
                   margin=_margin(a, a["offense"]),
                   down=down, distance=t, field=y, seconds=seconds, message=_i(b["message"]),
                   kind=GAIN, gain=0, new_field=0, replay=False,
                   fresh=is_fresh(a), start=_i(a["message"]))
        if rec["margin"] is None or rec["clock"] is None or y is None or down is None:
            continue
        bk = b["play_kind"]
        if bk == "TOUCHDOWN":
            scorer = _scorer(b, "TOUCHDOWN_TEAM") or b["offense"]
            if scorer == a["offense"]:
                rec["gain"] = 100 - y
            else:
                rec["kind"] = DEF_TD
        elif bk in ("SCRIMMAGE", "TURNOVER_ON_DOWNS"):
            if not b["field_position"]:
                continue
            by = _i(b["field_position"])
            if b["offense"] == a["offense"]:
                rec["gain"] = by - y
                rec["replay"] = (_i(b["down"]) == down and by - y < t)
            elif bk == "TURNOVER_ON_DOWNS" or (down == 4 and b["play_kind"] == "SCRIMMAGE"
                                               and "TURNOVER_ON_DOWNS" in b["play_messages"]):
                rec["gain"] = 100 - by - y
            else:
                rec["kind"] = TURNOVER
                rec["new_field"] = by
        else:
            continue
        if down == 4 and bk != "TOUCHDOWN" and rec["kind"] == GAIN and b["offense"] == a["offense"] \
                and rec["gain"] < t and not rec["replay"]:
            continue
        rec["success"] = rec["kind"] == GAIN and (rec["gain"] >= t or rec["gain"] >= 100 - y)
        if label is not None:
            label(rec)
        _timeouts_at(rec, a)
        out.append(rec)
    return out




def _timeouts_at(rec, row):
    """v8: each side's timeouts left at the snap's PLAY_OVER (offense's, defense's) and who, if
    anyone, stopped the running clock before it (timeout_role), where the row carries them
    (playover.annotate_timeouts, or the scouting export's columns)."""
    a, b = _i(row.get("timeouts_used_a")), _i(row.get("timeouts_used_b"))
    if a is None or b is None or row.get("offense") not in ("TEAM_A", "TEAM_B"):
        return
    mine = (a, b) if row["offense"] == "TEAM_A" else (b, a)
    have = 3 if rec["period"] <= 4 else OT_TIMEOUTS
    rec["timeouts_left"] = (max(0, have - mine[0]), max(0, have - mine[1]))
    caller = row.get("timeout_after")
    if caller in ("TEAM_A", "TEAM_B") and row.get("timeout_prev") != "incomplete":
        rec["timeout_role"] = OFFENCE if caller == row["offense"] else DEFENCE




# ---------------------------------------------------------------- the snaps' classes

def play_class(gain, dist):
    """A drive's play as the next snap's clock sees it: 0 nothing gained, 1 a gain short of the
    line, 2 a first down, 3 a loss."""
    gain = np.asarray(gain)
    return np.where(gain == 0, 0, np.where(gain < 0, 3, np.where(gain >= np.asarray(dist), 2, 1))).astype(np.int8)


# v14: a scrimmage snap's class, as the strength fit (v14.fit_strength) and strength_check read
# them: a turnover, a loss, no gain, a gain short of the line, a first down under 20 yards, 20-39
# yards, 40 or more. Gains are cut at the goal line.
SNAP_CLASSES = ("turnover", "loss", "no gain", "short", "1st <20", "20-39", "40+")
N_CLASSES = len(SNAP_CLASSES)


def snap_class(kind, gain, dist, field):
    """Each snap's class, an index into SNAP_CLASSES."""
    kind, dist = np.asarray(kind), np.asarray(dist)
    gain = np.minimum(np.asarray(gain), 100 - np.asarray(field))
    return np.select([kind != GAIN, gain < 0, gain == 0, gain >= 40, gain >= 20, gain < dist],
                     [0, 1, 2, 6, 5, 3], 4).astype(np.int64)


def link_prev(recs):
    """Each snap's previous play in its drive (play_class), or -1: none before it in the drive, or
    not in the records."""
    by_end = {}
    for r in recs:
        by_end[r["message"]] = r
    for r in recs:
        p = by_end.get(r.get("start"))
        ok = (p is not None and not r.get("fresh") and p["offense"] == r["offense"] and p["period"] == r["period"]
              and p["kind"] == GAIN and p["gain"] < 100 - p["field"])
        r["prev"] = int(play_class(p["gain"], p["distance"])) if ok else -1
    return recs



# ---------------------------------------------------------------- kicks and conversions

def kick_records(rows, kick_tds=True):
    """Every kickoff: kept by the kicking side or not, where the side with the ball started (100: it
    scored), and the clock used. kick_tds: a kick-off returned for a touchdown, or recovered and run in
    by the kicking side, kept at field 100 (sim14's KICK_TDS; off, the row after one is skipped)."""
    out = []
    prev = None
    for r in rows:
        if kick_tds and r["play_kind"] == "TOUCHDOWN" and prev is not None \
                and prev["period"] == r["period"] and prev["play_kind"] in ("CONVERSION", "FIELD_GOAL") \
                and prev["clock_seconds"] and r["clock_seconds"] and r["period"]:
            kicker = prev["offense"]
            scorer = _scorer(r, "TOUCHDOWN_TEAM") or r["offense"]
            seconds = _f(prev["clock_seconds"]) - _f(r["clock_seconds"])
            margin, p = _margin(prev, kicker), _i(prev["period"])
            if margin is not None and p is not None and 0 <= seconds <= 60:
                desperate = p >= 4 and margin < 0 and half_left(p, _f(prev["clock_seconds"])) <= 180
                out.append((desperate, scorer == kicker, 100, seconds))
            prev = r
            continue
        if r["play_kind"] == "KICKOFF" and r["field_position"] and r["clock_seconds"]:
            if not r["period"]:
                pass
            elif prev is not None and prev["period"] == r["period"] and \
                    prev["play_kind"] in ("CONVERSION", "FIELD_GOAL") and prev["clock_seconds"]:
                kicker = prev["offense"]
                seconds = _f(prev["clock_seconds"]) - _f(r["clock_seconds"])
                margin = _margin(prev, kicker)
                p = _i(prev["period"])
                if margin is None or p is None:
                    prev = r
                    continue
                desperate = p >= 4 and margin < 0 and half_left(p, _f(prev["clock_seconds"])) <= 180
                field = _i(r["field_position"])
                if kick_tds and _scorer(r, "TOUCHDOWN_TEAM") is not None:
                    field = 100                     # recovered by the kicking side and run in
                if 0 <= seconds <= 60:
                    out.append((desperate, r["offense"] == kicker, field, seconds))
            elif prev is None or prev["period"] != r["period"]:
                seconds = QUARTER - _f(r["clock_seconds"])
                if 0 <= seconds <= 60:
                    out.append((False, False, _i(r["field_position"]), seconds))
        prev = r
    return out


def kick_decisions(rows):
    """Every punt and field goal: where from, the result and the clock used."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in SNAP_KINDS or not a["field_position"] or not a["period"] \
                or a["period"] != b["period"]:
            continue
        if not a["clock_seconds"] or not b["clock_seconds"] or not b["field_position"]:
            continue
        seconds = _f(a["clock_seconds"]) - _f(b["clock_seconds"])
        if not 0 <= seconds <= 60:
            continue
        y = _i(a["field_position"])
        if b["play_kind"] == "PUNT" and b["offense"] != a["offense"] and a["down"] == "4" \
                and "SAFETY" not in (b["play_messages"] or ""):
            out.append(("punt", y, (100 - _i(b["field_position"])) - y, seconds))
        elif b["play_kind"] == "FIELD_GOAL":
            made = "FIELD_GOAL_GOOD" in b["play_messages"]
            after = _i(b["field_position"]) if (not made and b["offense"] != a["offense"]) else None
            out.append(("fg", y, made, seconds, after))
    return out


def safety_kicks(rows):
    """Where the receiver started after each safety's free kick."""
    out = []
    for i, (a, b) in enumerate(zip(rows, rows[1:])):
        if b["play_kind"] != "PUNT" or "SAFETY" not in (b["play_messages"] or "") \
                or a["offense"] not in ("TEAM_A", "TEAM_B"):
            continue
        for c in rows[i + 2:i + 6]:
            if c["play_kind"] == "SCRIMMAGE" and c["field_position"]:
                if c["offense"] != a["offense"]:
                    out.append(_i(c["field_position"]))
                break
    return out


def conversion_phase(period, clock):
    """The part of the game for a conversion decision."""
    if period <= 2:
        return 0
    if period == 3:
        return 1
    return 3 if (period >= 5 or clock <= 180) else 2



def conversion_cell(period, clock, cuts=CONV_CLOCK):
    """The cell of the go-for-two table: the decision phase, or with `cuts` (CONV_CLOCK; sim14 passes
    its switch) the last three minutes' clock slice (4, 5, ...; 3 from 2:00 to 3:00) or overtime (the
    last cell, CONV_CELLS - 1)."""
    phase = conversion_phase(period, clock)
    if phase != 3 or not cuts:
        return phase
    if period >= 5:
        return CONV_CELLS - 1
    for i, edge in enumerate(cuts):
        if clock <= edge:
            return 4 + i
    return 3

