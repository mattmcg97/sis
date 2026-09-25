"""v5's simulation: v4's (sim4.py, copied as the starting point and left
untouched there) with play calling by game state -- see "Game states for
play calling" below: each bin's plays are split by what they did to the
clock, and the state (quarter, 40-second slice, the offense's lead)
decides which kind is called, how long it takes and how well offenses do.
Inside its own 10 an offense's snaps come to a safety, a defensive
touchdown, a turnover or a touchdown at that yard line's own real rates
(see "Backed up" below).

v4's simulation, in turn, is v3's with

  * common random numbers: every simulated path draws its randomness from
    its own stream, keyed on the path's number and how many events it has
    played since the snapshot. Two snapshots of one match, priced side by
    side, play out the same luck, so their prices differ by what changed
    in the game and not by Monte Carlo noise;
  * kneels only with room to take them, and a safety's free kick from the
    20.

What follows is v3's description.

v3: a play-by-play simulation of the rest of the game on the real clock.

Every simulated snap draws its result -- yards, turnover, and the seconds of
game clock until the next snap -- from what really happened in the same
situation in SCOUTING_FULL (eAMFCalibrator scouting's PLAY_OVER export):
the same down, distance and field zone, and the same game situation for
the offense:

    q1          the first quarter (slower and lower scoring than the second)
    first       the second quarter, before its last two minutes
    late1       the last two minutes of the first half (everyone hurries)
    lead        second half, offense ahead            (the clock is milked)
    lead_late   last two minutes, offense ahead       (it is run out)
    even        second half, offense level or behind, before the last two
    trail_late  last two minutes, offense behind      (hurry-up)
    tied_late   last two minutes, level

so the result and its clock use come together: a leader's run up the
middle in the third quarter keeps the clock going 28 seconds, a trailer's
incompletion in the last minute stops it. Nothing in the tables looks at
the players' scores except through that situation.

Each offense has an efficiency, theta: its snaps draw from the situation's
results tilted toward the good end, u' = 1 - (1 - u) ** exp(theta), which
turns the league's first-down success rate f into f ** exp(-theta). A more
efficient offense converts more first downs, keeps the ball, and -- when it
is ahead -- runs more clock doing it.

4th downs use the fitted go / field-goal / make curves (drive.py) with each
player's aggression; kickoffs, onside kicks, punts and conversions are drawn
from their own tables. A leader with the ball in the last quarter kneels
when the downs he has cover the clock left. Overtime is a timed period,
replayed while level -- and (v5) played as the feed shows it is: each side
has the ball once, and after that the game ends the moment one side leads;
a side that scores a touchdown to trail by one or two after the other has
had the ball goes for two (see "Overtime" in simulate).

`Tables.build(matches)` makes the tables; `simulate(tables, start, ...)`
plays N copies of each starting state to the end, vectorised over all of
them at once, and returns the final scores.
"""

import math
from collections import Counter, defaultdict

import numpy as np

from .drive import DriveParams

MODES = ("q1", "first", "late1", "lead", "lead_late", "even", "trail_late", "tied_late")
N_KEYS = len(MODES) * 4 * 4 * 4
MIN_RECORDS = 60
MANAGED_SECONDS = 3.0
MIN_PUNTS = 30
CONV_PRIOR = 4.0
DECISION_PRIOR = 3.0  # information units shrinking each situation's 4th-down shift to 0
QUARTER = 240.0
LATE = 120.0
MAX_OT = 3
TD_TAIL = 12.0      # yards: how a long touchdown's gain would have run on (see simulate)

GAIN, TURNOVER, DEF_TD = 0, 1, 2
SCRIM, KICK, CONV, DONE = 0, 1, 2, 3


# --------------------------------------------------------------------------
# Situations

def half_left(period, clock):
    return clock + (QUARTER if period in (1, 3) else 0.0)


def mode_of(period, clock, margin):
    late = half_left(period, clock) <= LATE and period != 1 and period != 3
    if period == 1:
        return 0
    if period == 2:
        return 2 if late else 1
    if margin > 0:
        return 4 if late else 3
    if late:
        return 6 if margin < 0 else 7
    return 5


def dist_bucket(t):
    return 0 if t <= 2 else 1 if t <= 5 else 2 if t <= 10 else 3


def zone(y):
    return 0 if y < 40 else 1 if y < 70 else 2 if y < 90 else 3


def key_index(mode, down, t, y):
    return ((mode * 4 + (min(4, max(1, down)) - 1)) * 4 + dist_bucket(t)) * 4 + zone(y)


def _modes_np(period, clock, margin):
    hl = clock + QUARTER * ((period == 1) | (period == 3))
    late = (hl <= LATE) & (period != 1) & (period != 3)
    first = period <= 2
    out = np.where(period == 1, 0, np.where(first, np.where(late, 2, 1),
                   np.where(margin > 0, np.where(late, 4, 3),
                            np.where(late, np.where(margin < 0, 6, 7), 5))))
    return out


def _keys_np(mode, down, dist, y):
    db = np.where(dist <= 2, 0, np.where(dist <= 5, 1, np.where(dist <= 10, 2, 3)))
    z = np.where(y < 40, 0, np.where(y < 70, 1, np.where(y < 90, 2, 3)))
    d = np.clip(down, 1, 4) - 1
    return ((mode * 4 + d) * 4 + db) * 4 + z



# --------------------------------------------------------------------------
# Segments of a game for the in-play scoring shift (v5.fit_rest_of_game):
# the quarters, with the last two minutes of each half apart -- the end of
# the first half scores more than the rest of the second quarter says, and
# the end of the game is its own game -- each split by the scoreline's
# margin: close games bleed the clock, blowouts score in garbage time.

SEGMENTS = ("", "Q1", "Q2", "Q2 last 2:00", "Q3", "Q4", "Q4 last 2:00", "OT")
N_SEGMENTS = len(SEGMENTS)


def inplay_segment(period, clock):
    if period >= 5:
        return 7
    late = clock <= LATE
    return {1: 1, 2: 3 if late else 2, 3: 4, 4: 6 if late else 5}[period]


MARGIN_BANDS = ("0-3", "4-8", "9-16", "17+")   # the scoreline's margin, either way
N_BANDS = len(MARGIN_BANDS)


def margin_band(margin):
    m = abs(margin)
    return 0 if m <= 3 else 1 if m <= 8 else 2 if m <= 16 else 3


def _bands_np(margin):
    m = np.abs(margin)
    return np.where(m <= 3, 0, np.where(m <= 8, 1, np.where(m <= 16, 2, 3)))


def _segments_np(period, clock):
    late = clock <= LATE
    return np.where(period >= 5, 7, np.where(period == 1, 1, np.where(period == 3, 4, np.where(
        period == 2, np.where(late, 3, 2), np.where(late, 6, 5)))))

# --------------------------------------------------------------------------
# Game states for play calling (v5)
#
# The feed never says run or pass, but it does say what a play did to the
# clock: the game clock from one snap to the next is the play alone when
# the clock stopped after it (an incompletion, out of bounds -- passes,
# mostly) and the play plus however long the offense took to pick the next
# one when it kept running (runs and completions in bounds). v5 splits
# every play of a bin by that, and decides which kind is called from the
# game state -- the quarter, the time left in it and the offense's lead --
# before drawing the yards from real plays of that kind.

STOP_SECONDS = 12.0   # a snap to the next in this or less: the clock stopped
CLOCK_CELLS = 6       # 40-second slices of a quarter
LEAD_CELLS = 5        # the offense's lead: <=-9, -8..-1, 0, 1..8, >=9
N_CELLS = 4 * CLOCK_CELLS * LEAD_CELLS
STOP_PRIOR = 40.0     # snaps' worth of shrinkage toward the bin's own mix
SECONDS_PRIOR = 40.0  # snaps' worth of shrinkage on the clock used
EFF_PRIOR = 40.0      # information units shrinking the rubber band to 0
# A snap is only kept when the next one came in the same quarter, so with
# less than this left the plays that would have run past the buzzer are
# missing and the rest look short: the clock and play-call shifts are
# fitted on snaps with at least this much of the quarter left.
UNCUT_SECONDS = 60.0
STOP, RUNNING = 0, 1
# which parts of the play-calling fit are used ("stop", "seconds", "band"):
# each can be switched off to see what it is worth on its own
PLAY_CALLING = {"stop", "seconds", "band"}
# The quarters the play-calling shifts apply in. The fourth is left to the
# end-game tables (lead_late, trail_late, tied_late) as in v4: on held-out
# weeks every shift cost a little there, and together significantly
# (spread -0.0017, total -0.0024 Brier on Sep 10-22).
PLAY_CALLING_QUARTERS = (1, 2, 3)


def _quarter_mask():
    q = np.arange(N_CELLS) // (CLOCK_CELLS * LEAD_CELLS) + 1
    return np.isin(q, PLAY_CALLING_QUARTERS)


def lead_cell(lead):
    return 0 if lead <= -9 else 1 if lead < 0 else 2 if lead == 0 else 3 if lead <= 8 else 4


def cell_index(period, clock, lead):
    """(quarter, 40-second slice of it, the offense's lead) -> 0..N_CELLS-1."""
    pq = min(max(period, 1), 4) - 1
    cb = min(CLOCK_CELLS - 1, max(0, int((QUARTER - clock) // (QUARTER / CLOCK_CELLS))))
    return (pq * CLOCK_CELLS + cb) * LEAD_CELLS + lead_cell(lead)


def _cells_np(period, clock, lead):
    pq = np.clip(period, 1, 4) - 1
    cb = np.clip(((QUARTER - clock) // (QUARTER / CLOCK_CELLS)).astype(np.int64), 0, CLOCK_CELLS - 1)
    lc = np.where(lead <= -9, 0, np.where(lead < 0, 1, np.where(lead == 0, 2, np.where(lead <= 8, 3, 4))))
    return (pq * CLOCK_CELLS + cb) * LEAD_CELLS + lc


def fit_play_calling(tables, snaps):
    """By game-state cell: (1) how much likelier a clock-stopping play is
    than its bin's own mix says (a log-odds shift), (2) how many seconds
    more or less each kind of play uses than its bin's (the huddle: a leader
    milks the play clock, a trailer hurries), (3) the rubber band -- how
    much better or worse offenses do in that state than the bin's league
    average (a tilt on the efficiency). Each is shrunk toward 0."""
    n = len(snaps)
    key = np.array([r["key"] for r in snaps])
    cell = np.array([cell_index(r["period"], r["clock"], r["margin"]) for r in snaps])
    stop = np.array([r["seconds"] <= STOP_SECONDS for r in snaps])
    secs = np.array([r["seconds"] for r in snaps])
    succ = np.array([bool(r["success"]) for r in snaps])
    uncut = np.array([r["clock"] >= UNCUT_SECONDS for r in snaps])
    ns, cnt = tables.n_stop[key].astype(float), tables.count[key].astype(float)
    base = np.log(np.clip(ns, 0.5, None) / np.clip(cnt - ns, 0.5, None))
    # (1) the stop shift: Newton steps on the penalised likelihood
    stop_shift = np.zeros(N_CELLS)
    for _ in range(10):
        p = _sigmoid(base + stop_shift[cell])
        g = np.bincount(cell, (stop - p) * uncut, N_CELLS) - STOP_PRIOR * 0.25 * stop_shift
        h = np.bincount(cell, p * (1 - p) * uncut, N_CELLS) + STOP_PRIOR * 0.25
        stop_shift += g / h
    # (2) seconds against the bin's mean for the same kind of play
    cls_mean = np.zeros((2, N_KEYS))
    for k in range(N_KEYS):
        a, m, c = tables.start[k], tables.n_stop[k], tables.count[k]
        seg = tables.seconds[a:a + c]
        cls_mean[STOP, k] = seg[:m].mean() if m else 0.0
        cls_mean[RUNNING, k] = seg[m:].mean() if c > m else 0.0
    kind = np.where(stop, STOP, RUNNING)
    resid = secs - cls_mean[kind, key]
    sec_shift = np.zeros((2, N_CELLS))
    for c_ in (STOP, RUNNING):
        on = (kind == c_) & uncut
        tot = np.bincount(cell[on], resid[on], N_CELLS)
        num = np.bincount(cell[on], minlength=N_CELLS)
        sec_shift[c_] = tot / (num + SECONDS_PRIOR)
    # (3) the rubber band: P(success) = f ** exp(-a) within the play's kind
    f = np.clip(np.where(stop, tables.stop_success[key], tables.run_success[key]), 1e-3, 1 - 1e-3)
    grid = np.linspace(-1.0, 1.0, 201)
    eff_shift = np.zeros(N_CELLS)
    for c_ in range(N_CELLS):
        on = cell == c_
        if not on.any():
            continue
        lf, s_ = np.log(f[on]), succ[on]
        pa = np.exp(np.outer(np.exp(-grid), lf))                  # grid x snaps
        ll = np.where(s_, np.log(np.clip(pa, 1e-12, 1)), np.log(np.clip(1 - pa, 1e-12, 1))).sum(1)
        eff_shift[c_] = grid[np.argmax(ll - 0.5 * EFF_PRIOR * grid ** 2)]
    on = _quarter_mask()
    tables.stop_shift = stop_shift * on if "stop" in PLAY_CALLING else np.zeros(N_CELLS)
    tables.sec_shift = sec_shift * on if "seconds" in PLAY_CALLING else np.zeros((2, N_CELLS))
    tables.eff_shift = eff_shift * on if "band" in PLAY_CALLING else np.zeros(N_CELLS)
    return n


# --------------------------------------------------------------------------
# Reading the export

def _i(v):
    return int(float(v)) if v not in ("", None) else None


def _f(v):
    return float(v) if v not in ("", None) else None


def _scorer(row, prefix):
    for m in reversed((row.get("play_messages") or "").split("|")):
        if m.startswith(prefix) and m[-6:] in ("TEAM_A", "TEAM_B"):
            return m[-6:]
    return None


SNAP_KINDS = ("SCRIMMAGE", "KICKOFF", "PUNT", "TURNOVER_ON_DOWNS")


def _margin(row, team):
    """`team`'s lead on the row's scoreboard, or None when the row lacks the
    scores or the team."""
    home, away = _i(row["score_p1"]), _i(row["score_p2"])
    if home is None or away is None or team not in ("TEAM_A", "TEAM_B"):
        return None
    return (home - away) * (1 if team == "TEAM_A" else -1)


def snap_records(rows):
    """Scrimmage snaps in one match (export rows in message order):
    dicts with the snap's state, what came of it and the clock it used."""
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
                   kind=GAIN, gain=0, new_field=0, replay=False)
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
            continue                              # a kick, or a score the feed shortened
        if down == 4 and bk != "TOUCHDOWN" and rec["kind"] == GAIN and b["offense"] == a["offense"] \
                and rec["gain"] < t and not rec["replay"]:
            continue                              # 4th-down fakes that the feed kept
        rec["success"] = rec["kind"] == GAIN and (rec["gain"] >= t or rec["gain"] >= 100 - y)
        rec["mode"] = mode_of(rec["period"], rec["clock"], rec["margin"])
        rec["key"] = key_index(rec["mode"], down, t, y)
        out.append(rec)
    return out


def kick_records(rows):
    """Kickoffs: (desperate, onside recovered, receiver's field, seconds)."""
    out = []
    prev = None
    for r in rows:
        if r["play_kind"] == "KICKOFF" and r["field_position"] and r["clock_seconds"]:
            if not r["period"]:
                pass                                  # no quarter on the row: skip it
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
                if 0 <= seconds <= 60:
                    out.append((desperate, r["offense"] == kicker, _i(r["field_position"]), seconds))
            elif prev is None or prev["period"] != r["period"]:
                seconds = QUARTER - _f(r["clock_seconds"])
                if 0 <= seconds <= 60:
                    out.append((False, False, _i(r["field_position"]), seconds))
        prev = r
    return out


def kick_decisions(rows):
    """Punts and field goals off a snap: ('punt', field, net, seconds) and
    ('fg', field, made, seconds, the receiver's field after a miss)."""
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
            # (a safety comes through as a PUNT row: the free kick that follows it)
            out.append(("punt", y, (100 - _i(b["field_position"])) - y, seconds))
        elif b["play_kind"] == "FIELD_GOAL":
            made = "FIELD_GOAL_GOOD" in b["play_messages"]
            after = _i(b["field_position"]) if (not made and b["offense"] != a["offense"]) else None
            out.append(("fg", y, made, seconds, after))
    return out


def safety_kicks(rows):
    """After each safety, the receiving side's field on its first snap: the
    free kick comes from the conceding side's 20, not the 35, so it starts
    well up the field (about its 42 against a kickoff's 26)."""
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


SAFETY_KICK_MIN = 20       # free kicks seen before they are used as they are
SAFETY_KICK_SHIFT = 16     # else: a kickoff's field moved on this far


# --------------------------------------------------------------------------
# Backed up: the ball inside the offense's own 10
#
# A snap's yards come from its bin, and the bins are wide: the first zone
# runs from the 1 to the 39. Drawn at the 1, a one-yard loss seen at the 30
# is a safety -- but offenses on their own goal line do not play as they do
# at the 30, and the snaps that really went for a safety are the ones the
# bins never see (the feed shows them as the free kick that follows). In
# SCOUTING_FULL the bins made twice the safeties that really happened on
# the 1 and 2, a tenth of the defensive touchdowns (a fumble in the end
# zone) and under half the 90-plus-yard touchdowns.
#
# So inside the 10 each snap first draws what it came to from that yard
# line's own counts -- a safety, a defensive touchdown, a turnover, a
# touchdown, or none of these -- and only a snap that came to none of them
# takes its yards from the bin, kept in the field of play.

BACKED_UP = 10             # yard lines 1..BACKED_UP draw their own outcomes
BACKED_PRIOR = 60.0        # snaps' worth of pull toward the smooth curve
BACKED_OUTCOMES = ("safety", "def_td", "turnover", "td")
B_SAFETY, B_DEF_TD, B_TURNOVER, B_TD = range(4)
BACKED_RETURNS_MIN = 20    # turnovers seen before their returns are used


def backed_up_snaps(rows):
    """Snaps on downs 1-3 from the offense's own 1 to its BACKED_UP:
    (field, outcome index or None, the new offense's field after a
    turnover relative to the spot, or None)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in SNAP_KINDS or a["down"] not in ("1", "2", "3") \
                or not a["field_position"] or a["period"] != b["period"]:
            continue
        y = _i(a["field_position"])
        if not 1 <= y <= BACKED_UP:
            continue
        messages = b["play_messages"] or ""
        bk = b["play_kind"]
        back = None
        if "SAFETY" in messages:
            outcome = B_SAFETY
        elif bk == "TOUCHDOWN":
            scorer = _scorer(b, "TOUCHDOWN_TEAM") or b["offense"]
            outcome = B_TD if scorer == a["offense"] else B_DEF_TD
        elif bk in ("SCRIMMAGE", "TURNOVER_ON_DOWNS") and b["field_position"]:
            if b["offense"] != a["offense"] and bk == "SCRIMMAGE" \
                    and "TURNOVER_ON_DOWNS" not in messages:
                outcome = B_TURNOVER
                back = _i(b["field_position"]) - (100 - y)
            else:
                outcome = None
        else:
            continue                              # a kick, or a row the feed cut short
        out.append((y, outcome, back))
    return out


def fit_backed_up(records, prior=BACKED_PRIOR):
    """(hazard, returns): hazard[y, outcome] the chance a snap on the own
    y (1..BACKED_UP; row 0 unused) comes to that outcome, and the turnover
    returns (the new offense's field against the spot). Each yard line's
    own rate is pulled toward a logistic curve in the yard line by `prior`
    snaps, since a yard line sees a hundred snaps or so a month."""
    n = np.zeros(BACKED_UP + 1)
    k = np.zeros((BACKED_UP + 1, len(BACKED_OUTCOMES)))
    for y, outcome, _ in records:
        n[y] += 1
        if outcome is not None:
            k[y, outcome] += 1
    ys = np.arange(1, BACKED_UP + 1, dtype=float)
    hazard = np.zeros_like(k)
    for o in range(len(BACKED_OUTCOMES)):
        curve = _logistic_in_y(ys, n[1:], k[1:, o])
        hazard[1:, o] = (k[1:, o] + prior * curve) / (n[1:] + prior)
    returns = np.array([b for _, o, b in records if o == B_TURNOVER and b is not None], dtype=np.int32)
    return hazard, returns


def _logistic_in_y(ys, n, k, iterations=25, ridge=1.0):
    """P(outcome) = sigmoid(a + b * (y - mean)) fitted to k of n by Newton
    steps (a light ridge on b); the pooled rate when there are no events."""
    if n.sum() == 0:
        return np.zeros_like(ys)
    if k.sum() == 0:
        return np.full_like(ys, 0.5 / (n.sum() + 1))
    x = ys - ys.mean()
    pooled = k.sum() / n.sum()
    a, b = math.log(pooled / (1 - pooled)), 0.0
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(a + b * x)))
        w = n * p * (1 - p)
        ga, gb = (k - n * p).sum(), (x * (k - n * p)).sum() - ridge * b
        haa, hab, hbb = w.sum() + 1e-9, (w * x).sum(), (w * x * x).sum() + ridge
        det = haa * hbb - hab * hab
        a += (hbb * ga - hab * gb) / det
        b += (haa * gb - hab * ga) / det
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


CONV_MARGIN = 16      # margins after the touchdown, clipped to +-this


def conversion_phase(period, clock):
    """0 first half, 1 third quarter, 2 fourth before its last three
    minutes, 3 the last three minutes and overtime."""
    if period <= 2:
        return 0
    if period == 3:
        return 1
    return 3 if (period >= 5 or clock <= 180) else 2


def conversions(rows):
    """Each conversion: (phase, the scorer's margin after the six, went
    for two, good)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] != "TOUCHDOWN" or b["play_kind"] != "CONVERSION" or not a["period"]:
            continue
        m = [x for x in (b["play_messages"] or "").split("|") if "POINT" in x and x[-6:] in ("TEAM_A", "TEAM_B")]
        if not m:
            continue
        scorer = m[-1][-6:]
        two = any("TWO_POINT" in x for x in m)
        good = any(x.startswith(("EXTRA_POINT_GOOD", "TWO_POINT_CONVERSION_SUCCESSFUL")) for x in m[-1:])
        margin = _margin(a, scorer)
        if margin is None:
            continue
        out.append((conversion_phase(_i(a["period"]), _f(a["clock_seconds"]) or 0.0),
                    max(-CONV_MARGIN, min(CONV_MARGIN, margin)), two, good))
    return out


def decision_phase(period, clock):
    """0 first half, 1 third quarter, 2 fourth before its last three
    minutes, 3 the last three minutes and overtime."""
    return conversion_phase(period, clock)


def margin_bucket(margin):
    """<=-9, -8..-4, -3..-1, 0, 1..3, 4..8, >=9 -> 0..6."""
    return (0 if margin <= -9 else 1 if margin <= -4 else 2 if margin < 0 else 3 if margin == 0
            else 4 if margin <= 3 else 5 if margin <= 8 else 6)


def _margin_bucket_np(m):
    return np.where(m <= -9, 0, np.where(m <= -4, 1, np.where(m < 0, 2, np.where(
        m == 0, 3, np.where(m <= 3, 4, np.where(m <= 8, 5, 6))))))


EARLY_FG_RANGES = (45, 55, 65)  # kick-distance buckets for early-down field goals
EARLY_FG_RANGE = EARLY_FG_RANGES[-1]
EARLY_FG_CLOCK = (5, 10, 20, 30, 45)


def _clock_bin(clock):
    for i, edge in enumerate(EARLY_FG_CLOCK):
        if clock <= edge:
            return i
    return None


def early_kicks(rows):
    """Snaps on downs 1-3 in range as a half runs out: (0 = second quarter,
    1 = fourth quarter / overtime level or behind by 1-3, clock bin, kicked)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in SNAP_KINDS or a["down"] not in ("1", "2", "3"):
            continue
        if not a["period"] or a["period"] != b["period"] or not a["field_position"] \
                or not a["clock_seconds"]:
            continue
        kick = 100 - _i(a["field_position"]) + 17
        if kick > EARLY_FG_RANGE:
            continue
        rb = int(np.searchsorted(EARLY_FG_RANGES, kick))
        p, cb = _i(a["period"]), _clock_bin(_f(a["clock_seconds"]))
        margin = _margin(a, a["offense"])
        if margin is None:
            continue
        if cb is None:
            continue
        if p == 2:
            sit = 0
        elif p >= 4 and -3 <= margin <= 0:
            sit = 1
        else:
            continue
        out.append((sit, cb, rb, b["play_kind"] == "FIELD_GOAL"))
    return out


def fourth_down_choices(rows):
    """Each 4th down: (phase, margin bucket, field, distance, choice) with
    choice 'go', 'fg' or 'punt'."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["down"] != "4" or a["play_kind"] not in SNAP_KINDS or not a["field_position"] \
                or not a["distance"]:
            continue
        if not a["period"] or a["period"] != b["period"] or not a["clock_seconds"]:
            continue
        p, c = _i(a["period"]), _f(a["clock_seconds"])
        margin = _margin(a, a["offense"])
        if margin is None:
            continue
        choice = ("punt" if b["play_kind"] == "PUNT" else "fg" if b["play_kind"] == "FIELD_GOAL"
                  else "go")
        out.append((decision_phase(p, c), margin_bucket(margin), _i(a["field_position"]),
                    max(1, _i(a["distance"])), choice, c))
    return out


def _go_logit(dp, y, t):
    u = y / 100.0
    lt = math.log(max(1, t))
    red = 1.0 if y >= 80 else 0.0
    c = dp.go_coef
    return c[0] + c[1] * lt + c[2] * u + c[3] * u * u + c[4] * lt * u + c[5] * red + c[6] * lt * red


def _fg_logit(dp, y):
    a, b = dp.fg_kick_coef
    return a + b * y / 100.0


def fit_decision_shifts(decisions, dp, prior=DECISION_PRIOR, iterations=8):
    """Log-odds shifts, by (phase, margin bucket), on the league's go-for-it
    curve and on its field-goal-or-punt split, fitted by Newton steps with
    a shrinkage prior. The last three minutes with the offense behind are
    left to the simulation's own rules (it must score)."""
    go_shift, fg_shift = np.zeros((4, 7)), np.zeros((4, 7))
    cells = defaultdict(list)
    for phase, mb, y, t, choice, clock in decisions:
        if phase == 3 and mb < 3:
            continue
        cells[(phase, mb)].append((y, t, choice))
    for (phase, mb), items in cells.items():
        a = 0.0
        for _ in range(iterations):
            num, den = 0.0, prior
            for y, t, choice in items:
                p = 1.0 / (1.0 + math.exp(-(_go_logit(dp, y, t) + a)))
                num += (choice == "go") - p
                den += p * (1 - p)
            a += num / den
        go_shift[phase, mb] = a
        kicks = [(y, choice) for y, t, choice in items if choice != "go" and 100 - y + 17 <= dp.fg_max_distance]
        b = 0.0
        for _ in range(iterations):
            num, den = 0.0, prior
            for y, choice in kicks:
                p = 1.0 / (1.0 + math.exp(-(_fg_logit(dp, y) + b)))
                num += (choice == "fg") - p
                den += p * (1 - p)
            b += num / den
        fg_shift[phase, mb] = b
    return go_shift, fg_shift


# --------------------------------------------------------------------------
# Tables

def _fallbacks(key):
    """Coarser keys to fall back on, finest first."""
    rest, z = divmod(key, 4)
    rest, db = divmod(rest, 4)
    mode, d = divmod(rest, 4)
    zc = 0 if z < 2 else 1
    base = {0: 1, 1: 1, 2: 2, 3: 5, 4: 4, 5: 5, 6: 6, 7: 6}[mode]
    return [("k", key), ("zc", mode, d, db, zc), ("mdb", mode, d, db), ("m2", base, d, db, z),
            ("m2zc", base, d, db, zc), ("any", d, db, z), ("anyzc", d, db, zc), ("d", d, db),
            ("dd", min(d, 2))]


def _group_of(rec, level):
    mode, d, db, z = rec["mode"], min(4, rec["down"]) - 1, dist_bucket(rec["distance"]), zone(rec["field"])
    zc = 0 if z < 2 else 1
    base = {0: 1, 1: 1, 2: 2, 3: 5, 4: 4, 5: 5, 6: 6, 7: 6}[mode]
    return {"k": ("k", rec["key"]), "zc": ("zc", mode, d, db, zc), "mdb": ("mdb", mode, d, db),
            "m2": ("m2", base, d, db, z),
            "m2zc": ("m2zc", base, d, db, zc), "any": ("any", d, db, z), "anyzc": ("anyzc", d, db, zc),
            "d": ("d", d, db), "dd": ("dd", min(d, 2))}[level]


class Tables:
    """Everything the simulation draws from."""

    def __init__(self):
        self.start = None     # key -> first record of its bin
        self.count = None     # key -> records in its bin
        self.success = None   # key -> league first-down success in its bin
        self.kind = self.gain = self.new_field = self.seconds = self.replay = None
        self.kick = {}        # desperate -> (onside, field, seconds) arrays
        self.punt = {}        # field bucket -> (net, seconds)
        self.safety_kick = None   # the receiver's field after a safety's free kick
        # inside the own 10: [yard line, outcome] chances (fit_backed_up) and
        # the returns after a turnover there; None: the bins alone, as v4
        self.backed = None
        self.backed_return = None
        # v5's play calling (fit_play_calling): each bin lays out its
        # clock-stopping plays first, n_stop of them, then the rest
        self.n_stop = None
        self.stop_success = self.run_success = None
        self.stop_shift = np.zeros(N_CELLS)
        self.sec_shift = np.zeros((2, N_CELLS))
        self.eff_shift = np.zeros(N_CELLS)
        self.fg_seconds = None
        self.fg_after = None
        self.go_for_two = None   # (phase, margin after the six) -> P(going for two)
        self.two_good = 0.57
        self.kick_good = 0.98
        self.drive = DriveParams()
        # The league's efficiency by quarter (index 1-4, 5 overtime) on top
        # of each offense's own, fitted so that the simulation scores what
        # the league scores in each quarter (fit_period_theta).
        self.period_theta = np.zeros(6)
        # On top of it in the last two minutes of each half (index 2 and 4):
        # Q2's two-minute drill scores three-quarters of the quarter's points,
        # more than one level for the quarter gives it (v5.fit_quarter_levels)
        self.late_theta = np.zeros(6)
        # How far each simulated game's two offenses are drawn around the
        # match's prior (v5.fit_form), as variances of log points: the
        # game's, shared by both sides, and a player's own form for one with
        # no history; strength_slope (at strength_theta) is how many log
        # points one unit of theta is worth, to turn one into the other.
        self.strength_game = 0.0
        self.strength_league = 0.0
        self.strength_theta = np.zeros(0)
        self.strength_slope = np.zeros(0)
        # On top of it when pricing a total from inside a game (simulate's
        # in_play), by segment of the game and margin band:
        # v5.fit_rest_of_game, so that from real in-game states the
        # simulation leaves the points really still to come.
        self.inplay_theta = np.zeros((N_SEGMENTS, N_BANDS))
        # 4th-down log-odds shifts by (decision phase, margin bucket)
        self.go_shift = np.zeros((4, 7))
        # behind by 1-3 in the last three minutes and in range: P(kicking to
        # tie) with more than 30 seconds left, and with 30 or fewer
        self.late_fg = np.array([0.32, 0.62])
        # P(kicking on downs 1-3) by (Q2 / Q4 level or 1-3 behind, clock bin)
        self.early_fg = np.zeros((2, len(EARLY_FG_CLOCK), len(EARLY_FG_RANGES)))
        self.fg_shift = np.zeros((4, 7))

    @classmethod
    def build(cls, matches, min_records=MIN_RECORDS, seed=0):
        """From {match: export rows in message order}."""
        snaps, kicks, decisions, conv, fourths, early, free = [], [], [], [], [], [], []
        backed = []
        for rows in matches.values():
            snaps += snap_records(rows)
            backed += backed_up_snaps(rows)
            free += safety_kicks(rows)
            kicks += kick_records(rows)
            decisions += kick_decisions(rows)
            conv += conversions(rows)
            fourths += fourth_down_choices(rows)
            early += early_kicks(rows)
        t = cls()
        levels = ["k", "zc", "mdb", "m2", "m2zc", "any", "anyzc", "d", "dd"]
        groups = {lv: defaultdict(list) for lv in levels}
        for rec in snaps:
            for lv in levels:
                groups[lv][_group_of(rec, lv)].append(rec)
        rng = np.random.default_rng(seed)
        bins, chosen = {}, []
        for key in range(N_KEYS):
            for g in _fallbacks(key):
                recs = groups[g[0]].get(g)
                if recs and len(recs) >= min_records:
                    break
            if g not in bins:
                bins[g] = recs
            chosen.append(g)
        order = list(bins)
        start, count, succ, nstop, ssucc, rsucc = {}, {}, {}, {}, {}, {}
        kind, gain, newf, secs, rep, tdf = [], [], [], [], [], []
        for g in order:
            recs = list(bins[g])
            rng.shuffle(recs)
            # clock-stopping plays first, then the rest, each worst to best
            recs.sort(key=lambda r: (r["seconds"] > STOP_SECONDS,
                                     -2000 if r["kind"] == DEF_TD else -1000 if r["kind"] == TURNOVER
                                     else r["gain"] - r["distance"]))
            start[g] = len(kind)
            count[g] = len(recs)
            succ[g] = sum(r["success"] for r in recs) / len(recs)
            stops = [r for r in recs if r["seconds"] <= STOP_SECONDS]
            nstop[g] = len(stops)
            ssucc[g] = sum(r["success"] for r in stops) / max(1, len(stops))
            rsucc[g] = (sum(r["success"] for r in recs) - sum(r["success"] for r in stops)) \
                / max(1, len(recs) - len(stops))
            for r in recs:
                kind.append(r["kind"])
                gain.append(r["gain"])
                newf.append(r["new_field"])
                secs.append(r["seconds"])
                rep.append(r["replay"])
                tdf.append(r["field"] if (r["kind"] == GAIN and r["gain"] >= 100 - r["field"]) else -1)
        t.start = np.array([start[g] for g in chosen], dtype=np.int64)
        t.count = np.array([count[g] for g in chosen], dtype=np.int64)
        t.success = np.array([succ[g] for g in chosen])
        t.n_stop = np.array([nstop[g] for g in chosen], dtype=np.int64)
        t.stop_success = np.array([ssucc[g] for g in chosen])
        t.run_success = np.array([rsucc[g] for g in chosen])
        t.kind = np.array(kind, dtype=np.int8)
        t.gain = np.array(gain, dtype=np.int32)
        t.new_field = np.array(newf, dtype=np.int32)
        t.seconds = np.array(secs)
        t.replay = np.array(rep, dtype=bool)
        t.td_from = np.array(tdf, dtype=np.int32)
        t.bins_used = Counter(g[0] for g in chosen)
        for desperate in (False, True):
            ks = [k for k in kicks if k[0] == desperate] or kicks
            t.kick[desperate] = (np.array([k[1] for k in ks], dtype=bool),
                                 np.array([k[2] for k in ks], dtype=np.int32),
                                 np.array([k[3] for k in ks]))
        t.safety_kick = (np.array(free, dtype=np.int32) if len(free) >= SAFETY_KICK_MIN
                         else _shifted_kick(t.kick[False][1]))
        punts = [d for d in decisions if d[0] == "punt"]
        for b in range(3):
            ps = [p for p in punts if _punt_bucket(p[1]) == b]
            if len(ps) < MIN_PUNTS:
                ps = punts                            # too few from here: the league's
            t.punt[b] = (np.array([p[2] for p in ps], dtype=np.int32), np.array([p[3] for p in ps]))
        fgs = [d for d in decisions if d[0] == "fg"]
        t.fg_seconds = np.array([d[3] for d in fgs]) if fgs else np.array([5.0])
        # after a miss: the receiver's field against the kicker's spot
        misses = [(d[1], d[4]) for d in fgs if d[4] is not None]
        t.fg_after = (float(np.mean([100 - a - y for y, a in misses])) if misses else 7.0)
        # Going for two: counts by phase and margin, shrunk toward the
        # phase's rate by CONV_PRIOR attempts.
        size = 2 * CONV_MARGIN + 1
        tries, twos = np.zeros((4, size)), np.zeros((4, size))
        for phase, margin, two, good in conv:
            tries[phase, margin + CONV_MARGIN] += 1
            twos[phase, margin + CONV_MARGIN] += two
        base = twos.sum(1, keepdims=True) / np.maximum(1, tries.sum(1, keepdims=True))
        t.go_for_two = (twos + CONV_PRIOR * base) / (tries + CONV_PRIOR)
        two = [good for _, _, went, good in conv if went]
        kick = [good for _, _, went, good in conv if not went]
        t.two_good = float(np.mean(two)) if two else 0.57
        t.kick_good = float(np.mean(kick)) if kick else 0.98
        t.go_shift, t.fg_shift = fit_decision_shifts(fourths, t.drive)
        n, k = np.zeros_like(t.early_fg), np.zeros_like(t.early_fg)
        for sit, cb, rb, kicked in early:
            n[sit, cb, rb] += 1
            k[sit, cb, rb] += kicked
        t.early_fg = (k + 0.5 * k.sum(2, keepdims=True) / np.maximum(1, n.sum(2, keepdims=True))) / (n + 0.5)
        for i, late in enumerate((False, True)):
            ch = [d[4] for d in fourths if d[0] == 3 and d[1] == 2 and 100 - d[2] + 17 <= 60
                  and (d[5] <= 30) == late and d[4] != "punt"]
            if ch:
                t.late_fg[i] = (sum(c == "fg" for c in ch) + 1) / (len(ch) + 2)
        t.n_snaps = len(snaps)
        if backed:
            t.backed, returns = fit_backed_up(backed)
            t.backed_return = returns if len(returns) >= BACKED_RETURNS_MIN else None
        fit_play_calling(t, snaps)
        return t

    def success_of(self, key):
        return float(self.success[key])

    def save(self, path):
        arrays = dict(start=self.start, count=self.count, success=self.success, kind=self.kind,
                      gain=self.gain, new_field=self.new_field, seconds=self.seconds,
                      replay=self.replay, td_from=self.td_from, fg_seconds=self.fg_seconds,
                      fg_after=np.array([self.fg_after]), period_theta=self.period_theta,
                      late_theta=self.late_theta,
                      strength=np.array([self.strength_game, self.strength_league]),
                      strength_theta=self.strength_theta, strength_slope=self.strength_slope,
                      go_for_two=self.go_for_two, go_shift=self.go_shift, fg_shift=self.fg_shift,
                      late_fg=self.late_fg, early_fg=self.early_fg, conv_rates=np.array([self.two_good, self.kick_good]),
                      safety_kick=self.safety_kick, n_stop=self.n_stop,
                      stop_success=self.stop_success, run_success=self.run_success,
                      stop_shift=self.stop_shift, sec_shift=self.sec_shift, eff_shift=self.eff_shift,
                      inplay_theta=self.inplay_theta)
        if self.backed is not None:
            arrays["backed"] = self.backed
            if self.backed_return is not None:
                arrays["backed_return"] = self.backed_return
        for d, (a, b, c) in self.kick.items():
            arrays[f"kick{int(d)}_onside"], arrays[f"kick{int(d)}_field"], arrays[f"kick{int(d)}_sec"] = a, b, c
        for b, (net, s) in self.punt.items():
            arrays[f"punt{b}_net"], arrays[f"punt{b}_sec"] = net, s
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        t = cls()
        for name in ("start", "count", "success", "kind", "gain", "new_field", "seconds",
                     "replay", "td_from", "fg_seconds", "go_for_two"):
            setattr(t, name, z[name])
        t.fg_after = float(z["fg_after"][0])
        t.two_good, t.kick_good = (float(x) for x in z["conv_rates"])
        if "late_theta" in z:
            t.late_theta = z["late_theta"]
        if "strength" in z:
            t.strength_game, t.strength_league = (float(x) for x in z["strength"])
            t.strength_theta, t.strength_slope = z["strength_theta"], z["strength_slope"]
        if "period_theta" in z:
            t.period_theta = z["period_theta"]
        if "go_shift" in z:
            t.go_shift, t.fg_shift = z["go_shift"], z["fg_shift"]
        if "late_fg" in z:
            t.late_fg = z["late_fg"]
        if "early_fg" in z:
            t.early_fg = z["early_fg"]
        for d in (False, True):
            t.kick[d] = (z[f"kick{int(d)}_onside"], z[f"kick{int(d)}_field"], z[f"kick{int(d)}_sec"])
        for b in range(3):
            t.punt[b] = (z[f"punt{b}_net"], z[f"punt{b}_sec"])
        t.safety_kick = z["safety_kick"] if "safety_kick" in z else _shifted_kick(t.kick[False][1])
        if "inplay_theta" in z and z["inplay_theta"].shape == (N_SEGMENTS, N_BANDS):
            t.inplay_theta = z["inplay_theta"]
        if "backed" in z:
            t.backed = z["backed"]
            t.backed_return = z["backed_return"] if "backed_return" in z else None
        if "n_stop" in z:
            t.n_stop, t.stop_success, t.run_success = z["n_stop"], z["stop_success"], z["run_success"]
            t.stop_shift, t.sec_shift, t.eff_shift = z["stop_shift"], z["sec_shift"], z["eff_shift"]
        else:                     # a v4 model's tables: one kind of play per bin, as v4
            t.n_stop = t.count.copy()
            t.stop_success = t.run_success = t.success
        return t


def _shifted_kick(fields):
    return np.clip(np.asarray(fields) + SAFETY_KICK_SHIFT, 1, 99).astype(np.int32)


def _punt_bucket(y):
    return 0 if y < 35 else 1 if y < 55 else 2


# --------------------------------------------------------------------------
# The simulation

class Start:
    """Starting states, one per snapshot, as arrays (see from_states)."""

    def __init__(self, n):
        self.period = np.ones(n, dtype=np.int32)
        self.clock = np.full(n, QUARTER)
        self.phase = np.full(n, KICK, dtype=np.int8)
        self.team = np.zeros(n, dtype=np.int8)        # offense / kicker / scorer: 0 home, 1 away
        self.down = np.ones(n, dtype=np.int32)
        self.dist = np.full(n, 10, dtype=np.int32)
        self.y = np.full(n, 25, dtype=np.int32)
        self.home = np.zeros(n, dtype=np.int32)
        self.away = np.zeros(n, dtype=np.int32)
        # who kicks the second half: the opening receiver; -1 not known (a coin)
        self.kicks_second_half = np.full(n, -1, dtype=np.int8)
        self.theta = np.zeros((n, 2))
        self.aggression = np.zeros((n, 2))
        self.pace = np.ones((n, 2))
        self.strength = np.zeros((n, 2))
        self.strength_game = np.zeros((n, 2))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0xBF58476D1CE4E5B9)
_K3 = np.uint64(0x94D049BB133111EB)


def _uniform(seed, path, step, slot):
    """U(0,1) that depends only on (seed, path, step, slot): splitmix64."""
    with np.errstate(over="ignore"):
        z = (np.uint64(seed) + path.astype(np.uint64) * _K1 + step.astype(np.uint64) * _K2
             + np.uint64(slot) * _K3)
        z ^= z >> np.uint64(30)
        z *= _K2
        z ^= z >> np.uint64(27)
        z *= _K3
        z ^= z >> np.uint64(31)
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def strength_draw(tables, theta, form):
    """(theta, own sd, game sd) for each side, in theta: the players' own
    form on the day (`form`, variances of log points) and the game's, with
    theta moved so that the draws leave each side's expected points where
    they were."""
    theta = np.asarray(theta, dtype=float)
    if not len(tables.strength_slope):
        return theta, np.zeros(2), np.zeros(2)
    slope = np.maximum(0.3, np.interp(theta, tables.strength_theta, tables.strength_slope))
    own = np.sqrt(np.maximum(0.0, np.asarray(form, dtype=float))) / slope
    game = np.sqrt(max(0.0, tables.strength_game)) / slope
    return theta - slope * (own * own + game * game) / 2.0, own, game


def simulate(tables, start, n_paths, rng=None, theta_sd=None, kneel_seconds=20.0,
             desperate_seconds=180.0, max_steps=400, stats=None, common=True, seed=None,
             in_play=False, distinct=False):
    """Play every starting state `n_paths` times to the end.

    Returns (home, away) final scores, arrays of shape (states, n_paths).
    `theta_sd` (states x 2), if given, draws each path's efficiencies around
    the starting theta -- what the posterior still does not know.

    `common` (default): path k of every state in this call draws the same
    random numbers event for event (common random numbers), so states
    priced together differ only by what differs between them. Averaging
    over the states of many matches wants `common=False`: with it on, a
    thousand states on 100 paths are 100 games' worth of luck.

    `start.strength` and `start.strength_game` (states x 2): each path's
    two offenses are drawn around the state's theta, each by its own sd and
    both by one shared draw times the game's sd, off the path's own stream
    -- with `common`, path k of every snapshot of a match plays the same
    strengths.

    `distinct` (with `common`): every path of every state has a stream of
    its own, the same from call to call -- what a fit over many states wants:
    their luck independent, and each trial value played on the same luck.

    `in_play`: price the rest of the game's points as they really come --
    each snap's efficiency takes tables.inplay_theta for its segment of the
    game and margin band (v5 reads only its totals off such a run).
    """
    rng = rng or np.random.default_rng()
    # `seed` fixes the common random numbers: the same seed for every
    # snapshot of a match means path k plays the same luck whichever call
    # prices it, so a match priced one PLAY_OVER at a time moves only when
    # the game does. None: a fresh seed from rng.
    seed = int(rng.integers(0, 2 ** 62)) if seed is None else int(seed) % (2 ** 62)
    S = len(start.period)
    P = S * n_paths
    rep = lambda a: np.repeat(a, n_paths, axis=0)
    period, clock, phase, team = rep(start.period), rep(start.clock), rep(start.phase), rep(start.team)
    down, dist, y = rep(start.down), rep(start.dist), rep(start.y)
    score = np.stack([rep(start.home), rep(start.away)], axis=1).astype(np.int32)
    kick2 = rep(start.kicks_second_half)
    theta = rep(start.theta).astype(float)
    if theta_sd is not None:
        theta = theta + rep(theta_sd) * rng.standard_normal((P, 2))
    exp_theta = np.exp(theta)
    period_exp = np.exp(tables.period_theta)
    late_exp = np.exp(tables.late_theta)
    inplay_exp = np.exp(tables.inplay_theta) if in_play else None
    agg = rep(start.aggression)
    pace = rep(start.pace)
    dp = tables.drive
    gc = dp.go_coef
    ka, kb = dp.fg_kick_coef
    ma, mb, mc = dp.make_coef

    tally = (lambda name, k: stats.__setitem__(name, stats.get(name, 0) + int(k))) \
        if stats is not None else (lambda name, k: None)
    path_no = (np.arange(P, dtype=np.int64) if distinct
               else np.tile(np.arange(n_paths, dtype=np.int64), S))   # path k of its state
    free = np.zeros(P, dtype=bool)                             # the kick is a safety's free kick
    # overtime: whose possession is running (-1 none) and which sides have
    # had theirs -- over the whole of overtime, not per period
    ot_cur = np.full(P, -1, dtype=np.int8)
    ot_done = np.zeros((P, 2), dtype=bool)
    # a state already in overtime with one side ahead: that side has had
    # the ball (the snapshot does not say more)
    ahead = (period >= 5) & (score[:, 0] != score[:, 1])
    ot_done[ahead, np.where(score[ahead, 0] > score[ahead, 1], 0, 1)] = True
    step = np.zeros(P, dtype=np.int64)                         # events since the snapshot
    # each path's own strengths around the prior: what the pre-match does
    # not know about the day (step 0 is never used by the plays)
    own_sd = getattr(start, "strength", np.zeros((S, 2)))
    game_sd = getattr(start, "strength_game", np.zeros((S, 2)))
    if np.any(own_sd) or np.any(game_sd):
        u = [_uniform(seed, path_no, step, slot) if common else rng.random(P)
             for slot in (20, 21, 22, 23)]
        r1 = np.sqrt(-2.0 * np.log(np.maximum(u[0], 1e-12)))
        r2 = np.sqrt(-2.0 * np.log(np.maximum(u[2], 1e-12)))
        own = np.stack([r1 * np.cos(2 * np.pi * u[1]), r1 * np.sin(2 * np.pi * u[1])], axis=1)
        game = (r2 * np.cos(2 * np.pi * u[3]))[:, None]
        theta = theta + rep(own_sd) * own + rep(game_sd) * game
        exp_theta = np.exp(theta)

    def rand(ix, slot):
        if not common:
            return rng.random(len(ix))
        return _uniform(seed, path_no[ix], step[ix], slot)

    def pick(ix, slot, n):
        return np.minimum(n - 1, (rand(ix, slot) * n).astype(np.int64))

    def fg_make(yy):
        d = (100 - yy + 17) / 100.0
        p = _sigmoid(ma + mb * d + mc * d * d)
        return np.where(100 - yy + 17 > dp.fg_max_distance, 0.0, p)

    def end_period(ix):
        p = period[ix]
        if stats is not None:
            for q in (1, 2, 3, 4):
                sel = ix[p == q]
                tally(f"points_by_q{q}", score[sel].sum())
                if q == 2:
                    for k in (0, 3, 7):
                        tally(f"half_margin{k}", (np.abs(score[sel, 0] - score[sel, 1]) == k).sum())
                    tally("half_fg", 0)
                tally(f"ended_q{q}", len(sel))
        # Q1 -> Q2, Q3 -> Q4: the drive carries on
        carry = (p == 1) | (p == 3)
        c = ix[carry]
        period[c] += 1
        clock[c] = QUARTER
        # half time: the opening receiver kicks (a coin when the feed did
        # not say who received)
        h = ix[p == 2]
        period[h] = 3
        clock[h] = QUARTER
        phase[h] = KICK
        team[h] = np.where(kick2[h] >= 0, kick2[h], pick(h, 18, 2))
        # end of regulation / an overtime
        e = ix[p >= 4]
        level = score[e, 0] == score[e, 1]
        more = e[level & (period[e] < 4 + MAX_OT)]
        tally("overtime", (level & (period[e] == 4)).sum())
        phase[e[~level | (period[e] >= 4 + MAX_OT)]] = DONE
        period[more] += 1
        clock[more] = QUARTER
        phase[more] = KICK
        team[more] = pick(more, 1, 2)

    def turnover(ix, field):
        team[ix] = 1 - team[ix]
        y[ix] = np.clip(field, 1, 99)
        down[ix] = 1
        dist[ix] = np.minimum(10, 100 - y[ix])

    def score_td(ix, side):
        score[ix, side] += 6
        team[ix] = side
        phase[ix] = CONV

    for _ in range(max_steps):
        live = np.flatnonzero(phase != DONE)
        if not len(live):
            break
        # Overtime, as the feed shows it played: each side has the ball once;
        # after that the game is over as soon as one side leads (a score, or
        # the other side's possession ending without one). A possession ends
        # when the other side snaps, or at a kick after a score.
        ot = live[period[live] >= 5]
        if len(ot):
            ph_ot = phase[ot]
            snap = ot[ph_ot == SCRIM]
            ended = snap[(ot_cur[snap] >= 0) & (ot_cur[snap] != team[snap])]
            ot_done[ended, ot_cur[ended]] = True
            ot_cur[snap] = team[snap]
            kick = ot[ph_ot == KICK]
            ended = kick[ot_cur[kick] >= 0]
            ot_done[ended, ot_cur[ended]] = True
            ot_cur[kick] = -1
            over = ot[(ph_ot != CONV) & ot_done[ot, 0] & ot_done[ot, 1]
                      & (score[ot, 0] != score[ot, 1])]
            phase[over] = DONE
            tally("ot_decided", len(over))
            live = np.flatnonzero(phase != DONE)
            if not len(live):
                break
        step[live] += 1
        ph = phase[live]

        # conversions (they happen even with the clock at zero): late in a
        # game players chasing it go for two
        ix = live[ph == CONV]
        if len(ix):
            s_ = team[ix]
            margin = np.clip(score[ix, s_] - score[ix, 1 - s_], -CONV_MARGIN, CONV_MARGIN)
            cp = np.where(period[ix] <= 2, 0, np.where(period[ix] == 3, 1,
                          np.where((period[ix] >= 5) | (clock[ix] <= 180), 3, 2)))
            two = rand(ix, 2) < tables.go_for_two[cp, margin + CONV_MARGIN]
            # in overtime, once the other side has had the ball: one or two
            # behind after the six, go for two (to win, or to stay alive)
            two |= (period[ix] >= 5) & ot_done[ix, 1 - s_] & ((margin == -1) | (margin == -2))
            good = rand(ix, 3) < np.where(two, tables.two_good, tables.kick_good)
            score[ix, s_] += np.where(good, np.where(two, 2, 1), 0)
            tally("two_point_tries", two.sum())
            phase[ix] = KICK

        # kickoffs
        ix = live[ph == KICK]
        if len(ix):
            out = ix[clock[ix] <= 0]
            if len(out):
                end_period(out)
            ix = ix[clock[ix] > 0]
            if len(ix):
                k = team[ix]
                margin = score[ix, k] - score[ix, 1 - k]
                hl = clock[ix] + QUARTER * ((period[ix] == 1) | (period[ix] == 3))
                desperate = (period[ix] >= 4) & (margin < 0) & (hl <= desperate_seconds)
                fk = free[ix]
                for flag in (False, True):
                    sub = ix[(desperate == flag) & ~fk]
                    if not len(sub):
                        continue
                    onside, field, secs = tables.kick[flag]
                    j = pick(sub, 4, len(field))
                    rec = team[sub]
                    team[sub] = np.where(onside[j], rec, 1 - rec)
                    y[sub] = field[j]
                    down[sub] = 1
                    dist[sub] = 10
                    clock[sub] -= secs[j]
                    phase[sub] = SCRIM
                # after a safety: a free kick from the 20, no onside
                sub = ix[fk]
                if len(sub):
                    _, _, secs = tables.kick[False]
                    j = pick(sub, 4, len(secs))
                    team[sub] = 1 - team[sub]
                    y[sub] = tables.safety_kick[pick(sub, 14, len(tables.safety_kick))]
                    down[sub] = 1
                    dist[sub] = 10
                    clock[sub] -= secs[j]
                    phase[sub] = SCRIM
                    free[sub] = False

        ix = live[ph == SCRIM]
        if not len(ix):
            continue
        out = ix[clock[ix] <= 0]
        if len(out):
            end_period(out)
        ix = ix[clock[ix] > 0]
        if not len(ix):
            continue
        o = team[ix]
        margin = score[ix, o] - score[ix, 1 - o]
        p = period[ix]
        c = clock[ix]
        hl = c + QUARTER * ((p == 1) | (p == 3))
        yy = y[ix]
        dn = down[ix]
        tt = dist[ix]

        # kneel: ahead in the last quarter with the downs to cover the clock,
        # and the room to take them -- each kneel loses a yard, so one on the
        # goal line is a safety (backed up, players run real plays instead)
        kneel = (p >= 4) & (margin > 0) & (c <= kneel_seconds * (5 - dn)) & (yy > 5 - dn)
        tally("kneel", kneel.sum())
        if kneel.any():
            kx = ix[kneel]
            clock[kx] = 0.0
            end_period(kx)

        make = fg_make(yy)
        # level or 1-2 behind late (a kick wins or ties), in range, with the downs to run the clock
        # out: bleed it and kick as time expires (what real players do)
        drain = ~kneel & (p >= 4) & (margin <= 0) & (margin >= -2) & (dn < 4) & \
            (100 - yy + 17 <= 55) & (c <= kneel_seconds * (4 - dn))
        if drain.any():
            clock[ix[drain]] = 0.0
            c = clock[ix]
        tally("drain_fg", drain.sum())
        # the clock is running out on a half and a kick would help: players
        # take it on an early down as often as they really do
        sit = np.where(p == 2, 0, np.where((p >= 4) & (margin <= 0) & (margin >= -3), 1, -1))
        cb = np.searchsorted(np.array(EARLY_FG_CLOCK, dtype=float), c)
        kd_ = 100 - yy + 17
        rb = np.minimum(np.searchsorted(np.array(EARLY_FG_RANGES, dtype=float), kd_), len(EARLY_FG_RANGES) - 1)
        early_p = np.where((sit >= 0) & (cb < len(EARLY_FG_CLOCK)) & (kd_ <= EARLY_FG_RANGE),
                           tables.early_fg[np.maximum(sit, 0), np.minimum(cb, len(EARLY_FG_CLOCK) - 1), rb], 0.0)
        fg_now = ~kneel & (drain | ((dn < 4) & (rand(ix, 5) < early_p)))
        fourth = ~kneel & ~fg_now & (dn >= 4)
        # 4th down
        u = yy / 100.0
        lt = np.log(np.maximum(1, tt))
        red = (yy >= 80).astype(float)
        z = gc[0] + gc[1] * lt + gc[2] * u + gc[3] * u * u + gc[4] * lt * u + gc[5] * red + gc[6] * lt * red
        dph = np.where(p <= 2, 0, np.where(p == 3, 1, np.where((p >= 5) | (c <= 180), 3, 2)))
        mbk = _margin_bucket_np(margin)
        pgo = _sigmoid(z + agg[ix, o] + tables.go_shift[dph, mbk])
        late_trail = (p >= 4) & (hl <= desperate_seconds) & (margin < 0)
        # overtime, behind once the other side has had the ball: a stop ends it
        late_trail |= (p >= 5) & (margin < 0) & ot_done[ix, 1 - o]
        r1, r2 = rand(ix, 6), rand(ix, 7)
        # behind late: go for it, except that within a field goal the
        # players kick to tie as often as they really do
        kick_to_tie = (margin >= -3) & (make >= 0.3) & \
            (r1 < np.where(c <= 30, tables.late_fg[1], tables.late_fg[0]))
        go = np.where(late_trail, ~kick_to_tie, r1 < pgo)
        kick_fg = ~go & np.where(late_trail, True, r2 < _sigmoid(ka + kb * u + tables.fg_shift[dph, mbk])) \
            & (make > 0)
        do_fg = fg_now | (fourth & kick_fg)
        do_punt = fourth & ~go & ~kick_fg
        play = ~kneel & ~do_fg & ~do_punt
        tally("fg", do_fg.sum())
        if stats is not None:
            for q in (1, 2, 3, 4, 5):
                tally(f"fg_q{q}", (do_fg & (np.minimum(p, 5) == q)).sum())
                tally(f"punt_q{q}", (do_punt & (np.minimum(p, 5) == q)).sum())
                tally(f"go_q{q}", (fourth & play & (np.minimum(p, 5) == q)).sum())
        tally("punt", do_punt.sum())
        tally("fourth_go", (fourth & play).sum())
        tally("fourth", fourth.sum())

        # field goals
        fx = ix[do_fg]
        if len(fx):
            j = pick(fx, 8, len(tables.fg_seconds))
            clock[fx] -= tables.fg_seconds[j]
            good = rand(fx, 9) < make[do_fg]
            tally("fg_good", good.sum())
            g = fx[good]
            score[g, team[g]] += 3
            phase[g] = KICK
            m = fx[~good]
            turnover(m, 100 - y[m] - int(round(tables.fg_after)))

        # punts
        px = ix[do_punt]
        if len(px):
            b = np.where(y[px] < 35, 0, np.where(y[px] < 55, 1, 2))
            for bucket in range(3):
                sub = px[b == bucket]
                if not len(sub):
                    continue
                net, secs = tables.punt[bucket]
                j = pick(sub, 10, len(net))
                clock[sub] -= secs[j]
                turnover(sub, 100 - (y[sub] + net[j]))

        # snaps
        sx = ix[play]
        if not len(sx):
            continue
        so = team[sx]
        lead = score[sx, so] - score[sx, 1 - so]
        mode = _modes_np(period[sx], clock[sx], lead)
        key = _keys_np(mode, down[sx], dist[sx], y[sx])
        n = tables.count[key]
        # the play call: one that stops the clock (a pass, mostly) or one
        # that keeps it running, as likely as in this state of the game
        cell = _cells_np(period[sx], clock[sx], lead)
        ns = tables.n_stop[key]
        logit = np.log(np.maximum(ns, 0.5) / np.maximum(n - ns, 0.5)) + tables.stop_shift[cell]
        p_stop = np.where(ns == 0, 0.0, np.where(ns == n, 1.0, _sigmoid(logit)))
        stops = rand(sx, 15) < p_stop
        seg0 = tables.start[key] + np.where(stops, 0, ns)
        seg_n = np.maximum(1, np.where(stops, ns, n - ns))
        # the yards: that kind of play's real results, tilted by the
        # offense's efficiency and by how offenses really do in this state
        uu = rand(sx, 11)
        tilt = exp_theta[sx, so] * period_exp[np.minimum(period[sx], 5)] * np.exp(tables.eff_shift[cell])
        late = ((period[sx] == 2) | (period[sx] == 4)) & (clock[sx] <= LATE)
        if late.any():
            tilt = tilt * np.where(late, late_exp[np.minimum(period[sx], 5)], 1.0)
        if inplay_exp is not None:
            tilt = tilt * inplay_exp[_segments_np(period[sx], clock[sx]),
                                     _bands_np(score[sx, 0] - score[sx, 1])]
        uu = 1.0 - (1.0 - uu) ** tilt
        j = seg0 + np.minimum(seg_n - 1, (uu * seg_n).astype(np.int64))
        # the clock: that play's, plus how much longer (or shorter) offenses
        # take in this state for this kind of play
        used = np.maximum(1.0, tables.seconds[j] + tables.sec_shift[np.where(stops, STOP, RUNNING), cell]) \
            * pace[sx, so]
        clock[sx] -= used
        kd = tables.kind[j]
        # inside the own 10 the yard line's own chances come first; a snap
        # that comes to none of them ("held") takes the bin's yards, kept in
        # the field of play
        held = np.zeros(len(sx), dtype=bool)
        if tables.backed is not None:
            inside = np.flatnonzero(y[sx] <= BACKED_UP)
            if len(inside):
                bx = sx[inside]
                cum = np.cumsum(tables.backed[np.clip(y[bx], 1, BACKED_UP)], axis=1)
                got = (rand(bx, 16)[:, None] >= cum).sum(1)
                tally("backed_snaps", len(bx))
                kd = kd.copy()
                kd[inside] = np.where(got < len(BACKED_OUTCOMES), -1, GAIN)
                held[inside[got == len(BACKED_OUTCOMES)]] = True
                s_ = bx[got == B_SAFETY]
                if len(s_):
                    score[s_, 1 - team[s_]] += 2
                    phase[s_] = KICK                # the free kick, from the 20
                    free[s_] = True
                tally("safety", len(s_))
                d_ = bx[got == B_DEF_TD]
                if len(d_):
                    score_td(d_, 1 - team[d_])
                tally("def_td", len(d_))
                t_ = bx[got == B_TURNOVER]
                if len(t_):
                    back = (tables.backed_return[pick(t_, 17, len(tables.backed_return))]
                            if tables.backed_return is not None else 0)
                    turnover(t_, 100 - y[t_] + back)
                tally("turnover", len(t_))
                o_ = bx[got == B_TD]
                if len(o_):
                    score_td(o_, team[o_])
                tally("td", len(o_))
        tally("snaps", len(sx))
        tally("stop_calls", stops.sum())
        if stats is not None:
            for q in (1, 2, 3, 4):
                on = period[sx] == q
                tally(f"snaps_q{q}", on.sum())
                tally(f"seconds_q{q}", tables.seconds[j[on]].sum())
                tally(f"first_q{q}", (on & (kd == GAIN) & (tables.gain[j] >= dist[sx])).sum())
        tally("snap_seconds", tables.seconds[j].sum())
        tally("turnover", (kd == TURNOVER).sum())
        tally("def_td", (kd == DEF_TD).sum())

        d = sx[kd == DEF_TD]
        if len(d):
            score_td(d, 1 - team[d])
        tvr = kd == TURNOVER
        t_ix = sx[tvr]
        if len(t_ix):
            turnover(t_ix, tables.new_field[j[tvr]])
        gsel = kd == GAIN
        gx = sx[gsel]
        if not len(gx):
            continue
        gain = tables.gain[j[gsel]]
        rp = tables.replay[j[gsel]]
        y2 = y[gx] + gain
        # A touchdown's gain is cut off at the goal line: from further back
        # the same play still scores if it would have run on far enough.
        # Madden's long gains have a heavy tail (a 30-yard gain goes 10 more
        # 62% of the time), taken here as exp(-extra / TD_TAIL).
        tdf = tables.td_from[j[gsel]]
        hg = held[gsel]
        short = (tdf >= 0) & (y[gx] < tdf) & ~hg
        if short.any():
            extra = tdf[short] - y[gx][short]
            gs = gx[short]
            runs_on = rand(gs, 12) < np.exp(-extra / TD_TAIL)
            y2[short] = np.where(runs_on, 100, y2[short] + (rand(gs, 13) * extra).astype(np.int32))
        y2 = np.where(hg, np.clip(y2, 1, 99), y2)   # backed up: its outcomes were drawn above
        gain = y2 - y[gx]                           # the yards actually made
        td = y2 >= 100
        tally("td", td.sum())
        if stats is not None:
            for q in (1, 2, 3, 4):
                tally(f"td_q{q}", (td & (period[gx] == q)).sum())
        if td.any():
            score_td(gx[td], team[gx[td]])
        sf = y2 <= 0
        tally("safety", sf.sum())
        if sf.any():
            s = gx[sf]
            score[s, 1 - team[s]] += 2
            phase[s] = KICK                         # the free kick, from the 20
            free[s] = True
        rest = ~td & ~sf
        gx, gain, rp, y2 = gx[rest], gain[rest], rp[rest], y2[rest]
        # clock management as a half runs out: in range of a kick that would
        # help, players stop the clock (spike, time out, out of bounds) and
        # leave a few seconds to take it
        pg = period[gx]
        mg = score[gx, team[gx]] - score[gx, 1 - team[gx]]
        wants = ((pg == 2) | ((pg >= 4) & (mg <= 0) & (mg >= -3))) & (100 - y2 + 17 <= EARLY_FG_RANGE)
        before = clock[gx] + used[gsel][rest]
        keep = wants & (clock[gx] < MANAGED_SECONDS) & (before > MANAGED_SECONDS + 1)
        clock[gx[keep]] = MANAGED_SECONDS
        tally("managed", keep.sum())
        first = gain >= dist[gx]
        nd = np.where(first, 1, np.where(rp, down[gx], down[gx] + 1))
        nt = np.where(first, np.minimum(10, 100 - y2), dist[gx] - gain)
        fresh = nt <= 0
        nd = np.where(fresh, 1, nd)
        nt = np.where(fresh, np.minimum(10, 100 - y2), nt)
        y[gx] = y2
        down[gx] = nd
        dist[gx] = nt
        tod = nd > 4
        tally("downs", tod.sum())
        tally("first_downs", first.sum())
        if tod.any():
            tx = gx[tod]
            turnover(tx, 100 - y[tx])

    # a path still going after max_steps (none should be) keeps its score
    tally("unfinished", (phase != DONE).sum())
    return score[:, 0].reshape(S, n_paths), score[:, 1].reshape(S, n_paths)


def fit_period_theta(tables, real_points, n_paths=30000, rounds=6, seed=1):
    """tables.period_theta so that the simulation from kickoff scores `real_points`
    (the league's mean points in each of quarters 1-4). A point of scoring
    moves with theta at about 1.8 x the quarter's points per unit."""
    for _ in range(rounds):
        got = kickoff_quarter_points(tables, n_paths, seed)
        for q in range(4):
            tables.period_theta[q + 1] += (real_points[q] - got[q]) / (1.8 * max(1.0, real_points[q]))
        tables.period_theta[5] = tables.period_theta[4]
    return tables.period_theta.copy(), got


def kickoff_quarter_points(tables, n_paths=30000, seed=1):
    """The simulation's mean points in each of quarters 1-4 from kickoff,
    league-average offenses."""
    start = Start(1)
    start.team[:] = 1
    stats = {}
    simulate(tables, start, n_paths, np.random.default_rng(seed), stats=stats)
    cum = [stats[f"points_by_q{q}"] / stats[f"ended_q{q}"] for q in (1, 2, 3, 4)]
    return np.diff([0.0] + cum)


def quarter_points(matches):
    """The league's mean points in each of quarters 1-4, off the export --
    read off each quarter's last row. That row often does not yet show the
    quarter's last score (the feed posts it on the next quarter's first
    row), so Q1 comes out about 0.65 short and Q4 0.5 long of what was
    scored. Counting to the next quarter's first row instead (as
    v5.quarter_start_states does) moved scoring from the fourth quarter to
    the first and, held out, cost moneyline and spread about 0.001 Brier in
    both weeks tested with no gain on the total: the late-game margin swings
    lean on those fourth-quarter points (see the README), so the kickoff
    fit keeps this reading."""
    ends = defaultdict(list)
    for rows in matches.values():
        last = {}
        for r in rows:
            if r["period"] in ("1", "2", "3", "4") and r["score_p1"] and r["score_p2"]:
                last[int(r["period"])] = int(r["score_p1"]) + int(r["score_p2"])
        if all(q in last for q in (1, 2, 3, 4)):
            for q in (1, 2, 3, 4):
                ends[q].append(last[q])
    cum = [float(np.mean(ends[q])) for q in (1, 2, 3, 4)]
    return list(np.diff([0.0] + cum))