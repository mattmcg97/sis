"""v14's simulation: plays the rest of a game snap by snap from real plays in the same situation."""

import datetime as dt
import math
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np

from .drive import DriveParams

MODES = ("q1", "first", "late1", "lead", "lead_late", "even", "trail_late", "tied_late", "lead_big",
         "lead_late_big", "lead_q4", "lead_big_q4", "level_q4", "trail_q4", "trail_big_q4",
         "trail_late_big")
BIG_LEAD = 9
Q4_MODES = True
N_KEYS = len(MODES) * 4 * 4 * 4
MIN_RECORDS = 60
MANAGED_SECONDS = 3.0
MIN_PUNTS = 30
CONV_PRIOR = 4.0
DECISION_PRIOR = 3.0
QUARTER = 240.0
LATE = 120.0
MAX_OT = 3
# v11: a game level at the end of the fourth quarter plays overtime from its first period. v10 sent
# it into overtime and, in the same step, counted it among the overtime periods just ended, so every
# overtime started in its second (54 real overtimes since 24 Aug: 1.4 periods each, the same as v11
# from their own kick-offs; v10's counted 2.4).
OT_FIRST_PERIOD = True
TD_TAIL = 12.0

GAIN, TURNOVER, DEF_TD = 0, 1, 2
SCRIM, KICK, CONV, DONE = 0, 1, 2, 3


def half_left(period, clock):
    """Seconds left in the half."""
    return clock + (QUARTER if period in (1, 3) else 0.0)


def mode_of(period, clock, margin, big_lead=BIG_LEAD, q4=False):
    """The game situation for an offense: quarter, late in a half, and ahead, level or behind; with
    `q4`, the fourth quarter before its last two minutes has situations of its own, and a side two
    scores behind in the fourth quarter is one of its own before and in the last two minutes."""
    late = half_left(period, clock) <= LATE and period != 1 and period != 3
    if period == 1:
        return 0
    if period == 2:
        return 2 if late else 1
    big = margin > 0 and big_lead is not None and margin >= big_lead
    far = q4 and period >= 4 and big_lead is not None and margin <= -big_lead
    if q4 and period >= 4 and not late:
        return 11 if big else 10 if margin > 0 else 12 if margin == 0 else 14 if far else 13
    if far:
        return 15
    if margin > 0:
        return (9 if big else 4) if late else (8 if big else 3)
    if late:
        return 6 if margin < 0 else 7
    return 5


def dist_bucket(t):
    """Yards to go as a bucket."""
    return 0 if t <= 2 else 1 if t <= 5 else 2 if t <= 10 else 3


def zone(y):
    """Field position as a zone."""
    return 0 if y < 40 else 1 if y < 70 else 2 if y < 90 else 3


def key_index(mode, down, t, y):
    """The table bin for a situation, down, distance and zone."""
    return ((mode * 4 + (min(4, max(1, down)) - 1)) * 4 + dist_bucket(t)) * 4 + zone(y)


def _modes_np(period, clock, margin, big_lead=BIG_LEAD, q4=False):
    """mode_of for arrays."""
    hl = clock + QUARTER * ((period == 1) | (period == 3))
    late = (hl <= LATE) & (period != 1) & (period != 3)
    first = period <= 2
    big = (margin >= big_lead) if big_lead is not None else np.zeros(np.shape(margin), dtype=bool)
    out = np.where(period == 1, 0, np.where(first, np.where(late, 2, 1),
                   np.where(margin > 0, np.where(late, np.where(big, 9, 4), np.where(big, 8, 3)),
                            np.where(late, np.where(margin < 0, 6, 7), 5))))
    if q4:
        own = (period >= 4) & ~late
        far = (margin <= -big_lead) if big_lead is not None else np.zeros(np.shape(margin), dtype=bool)
        q4_mode = np.where(margin > 0, np.where(big, 11, 10),
                           np.where(margin == 0, 12, np.where(far, 14, 13)))
        out = np.where(own, q4_mode, np.where((period >= 4) & late & far, 15, out))
    return out


def _keys_np(mode, down, dist, y):
    """key_index for arrays."""
    db = np.where(dist <= 2, 0, np.where(dist <= 5, 1, np.where(dist <= 10, 2, 3)))
    z = np.where(y < 40, 0, np.where(y < 70, 1, np.where(y < 90, 2, 3)))
    d = np.clip(down, 1, 4) - 1
    return ((mode * 4 + d) * 4 + db) * 4 + z


SEGMENTS = ("", "Q1", "Q2", "Q2 last 2:00", "Q3", "Q4", "Q4 last 2:00", "OT")
N_SEGMENTS = len(SEGMENTS)


def inplay_segment(period, clock):
    """The segment of the game a moment falls in."""
    if period >= 5:
        return 7
    late = clock <= LATE
    return {1: 1, 2: 3 if late else 2, 3: 4, 4: 6 if late else 5}[period]


MARGIN_BANDS = ("0-3", "4-8", "9-16", "17+")
N_BANDS = len(MARGIN_BANDS)


def margin_band(margin):
    """The scoreline's margin as a band."""
    m = abs(margin)
    return 0 if m <= 3 else 1 if m <= 8 else 2 if m <= 16 else 3


def _bands_np(margin):
    """margin_band for arrays."""
    m = np.abs(margin)
    return np.where(m <= 3, 0, np.where(m <= 8, 1, np.where(m <= 16, 2, 3)))


def _segments_np(period, clock):
    """inplay_segment for arrays."""
    late = clock <= LATE
    return np.where(period >= 5, 7, np.where(period == 1, 1, np.where(period == 3, 4, np.where(
        period == 2, np.where(late, 3, 2), np.where(late, 6, 5)))))


STOP_SECONDS = 12.0
CLOCK_CELLS = 6
LEAD_CELLS = 5
N_CELLS = 4 * CLOCK_CELLS * LEAD_CELLS
STOP_PRIOR = 40.0
SECONDS_PRIOR = 40.0
EFF_PRIOR = 40.0
UNCUT_SECONDS = 60.0
STOP, RUNNING = 0, 1
PLAY_CALLING = {"stop", "seconds", "band"}
PLAY_CALLING_QUARTERS = (1, 2, 3)
PLAY_CALLING_Q4 = {"stop", "seconds"}


def _quarter_mask(item=None):
    """Which game-state cells the play-calling shifts apply in: Q1-Q3, and Q4 for the items in
    PLAY_CALLING_Q4."""
    q = np.arange(N_CELLS) // (CLOCK_CELLS * LEAD_CELLS) + 1
    return np.isin(q, PLAY_CALLING_QUARTERS) | ((q == 4) & (item in PLAY_CALLING_Q4))


def lead_cell(lead):
    """The offense's lead as a bucket."""
    return 0 if lead <= -9 else 1 if lead < 0 else 2 if lead == 0 else 3 if lead <= 8 else 4


def cell_index(period, clock, lead):
    """The game-state cell: quarter, 40-second slice and the offense's lead."""
    pq = min(max(period, 1), 4) - 1
    cb = min(CLOCK_CELLS - 1, max(0, int((QUARTER - clock) // (QUARTER / CLOCK_CELLS))))
    return (pq * CLOCK_CELLS + cb) * LEAD_CELLS + lead_cell(lead)


def _cells_np(period, clock, lead):
    """cell_index for arrays."""
    pq = np.clip(period, 1, 4) - 1
    cb = np.clip(((QUARTER - clock) // (QUARTER / CLOCK_CELLS)).astype(np.int64), 0, CLOCK_CELLS - 1)
    lc = np.where(lead <= -9, 0, np.where(lead < 0, 1, np.where(lead == 0, 2, np.where(lead <= 8, 3, 4))))
    return (pq * CLOCK_CELLS + cb) * LEAD_CELLS + lc


def _running_logit(tables, key):
    """The log-odds of drawing from a bin's clock-stopped plays for a snap that follows a scrimmage
    play (the bin's fresh possessions left out), and the bin's counts without them."""
    ns = tables.n_stop[key] - tables.n_fresh_stop[key]
    n = tables.count[key] - tables.n_fresh[key]
    return np.log(np.maximum(ns, 0.5) / np.maximum(n - ns, 0.5)), ns, n


# v8: the clock is fitted right to the end of each quarter. A snap that ran the quarter out used at
# least the time that was left: it is kept as a censored observation, where before every snap with
# under a minute left was dropped and the last 40 seconds of each quarter were never fitted.
LATE_CLOCK = True
STOP_GRID = np.linspace(-4.0, 4.0, 401)


def _fit_stop_censored(tables, on, ends):
    """The clock-stopped share's shift by cell, by maximum likelihood on every snap that follows a
    scrimmage play: a stopped play, a running play, or -- for a snap that ran the quarter out with c
    seconds left -- either one that would have taken c or more."""
    base = _running_logit(tables, np.array([r["key"] for r in on] + [r["key"] for r in ends],
                                           dtype=np.int64))[0]
    cell = np.array([cell_index(r["period"], r["clock"], r["margin"]) for r in on + ends], dtype=np.int64)
    kind = np.array([0 if r["seconds"] <= STOP_SECONDS else 1 for r in on] + [2] * len(ends))
    wt = np.array([r.get("w", 1.0) for r in on] + [r.get("w", 1.0) for r in ends])
    e_stop, e_run = np.zeros(len(kind)), np.zeros(len(kind))
    for i, r in enumerate(ends, start=len(on)):
        a, m, c = tables.start[r["key"]], tables.n_stop[r["key"]], tables.count[r["key"]]
        seg = tables.seconds[a:a + c]
        e_stop[i] = float((seg[:m] >= r["clock"]).mean()) if m else 0.0
        e_run[i] = float((seg[m:] >= r["clock"]).mean()) if c > m else 0.0
    shift = np.zeros(N_CELLS)
    pen = 0.5 * STOP_PRIOR * 0.25 * STOP_GRID ** 2
    order = np.argsort(cell, kind="stable")
    bounds = np.searchsorted(cell[order], np.arange(N_CELLS + 1))
    for c_ in range(N_CELLS):
        ix = order[bounds[c_]:bounds[c_ + 1]]
        if not len(ix):
            continue
        p = np.clip(_sigmoid(base[ix, None] + STOP_GRID[None, :]), 1e-9, 1 - 1e-9)
        k = kind[ix, None]
        lik = np.where(k == 0, p, np.where(k == 1, 1 - p,
                                           p * e_stop[ix, None] + (1 - p) * e_run[ix, None]))
        shift[c_] = STOP_GRID[np.argmax((np.log(np.maximum(lik, 1e-12)) * wt[ix, None]).sum(0) - pen)]
    return shift


# v8: timeouts, three for each side each half, called as real sides call them. Before a snap,
# with the clock running after the last play, each side may stop it -- at the rate real sides do
# in the same quarter, 40-second slice, with or without the ball and at the same score, while it
# has one left (call_p: fitted on the calibrator's timeouts.csv with v8-build --timeouts, else the
# real rates in DEFAULT_CALL_P). The snap
# then takes only its own play's time. Real sides call them in the last four minutes of the
# second and fourth quarters, and hardly ever otherwise. A real play that followed a timeout is
# put back in the tables with a clock-running play's time, so the calls are not counted twice.
# A leader kneels the game out only when its downs left, less the trailing side's timeouts, each
# worth a play clock, outlast the clock.
TIMEOUTS = True
PLAY_CLOCK = 40.0
TIMEOUT_QUARTERS = (2, 4)
TIMEOUT_PRIOR = 20.0
OT_TIMEOUTS = 2            # each side's timeouts in each overtime period
OFFENCE, DEFENCE = 0, 1
N_CALL = (len(TIMEOUT_QUARTERS), 2, LEAD_CELLS, CLOCK_CELLS)


# The share of running clocks each side stopped with a timeout before the snap, while it had one,
# in 12,000 real calls over 2,300 matches (24 Aug - 22 Sep 2026: SCOUTING_FULL's TIMEOUT_CALLED
# messages, `python -m eAMFCalibrator timeouts`): what a build uses when it is given no
# timeouts.csv. [quarter 2, 4][with the ball, without it][the side's own score][slice].
DEFAULT_CALL_P = [
    [  # Q2
        [  # with the ball: down 9+, down 1-8, level, up 1-8, up 9+; 40-second slices 4:00 -> 0:00
            [0.000, 0.000, 0.006, 0.018, 0.145, 0.788],
            [0.002, 0.002, 0.000, 0.011, 0.117, 0.711],
            [0.006, 0.003, 0.000, 0.010, 0.095, 0.706],
            [0.000, 0.003, 0.000, 0.005, 0.108, 0.735],
            [0.002, 0.001, 0.000, 0.002, 0.056, 0.717],
        ],
        [  # without it: down 9+, down 1-8, level, up 1-8, up 9+; 40-second slices 4:00 -> 0:00
            [0.001, 0.005, 0.099, 0.168, 0.251, 0.130],
            [0.003, 0.011, 0.084, 0.157, 0.232, 0.128],
            [0.003, 0.013, 0.075, 0.136, 0.232, 0.126],
            [0.001, 0.008, 0.055, 0.100, 0.203, 0.107],
            [0.000, 0.007, 0.056, 0.095, 0.101, 0.056],
        ],
    ],
    [  # Q4
        [  # with the ball: down 9+, down 1-8, level, up 1-8, up 9+; 40-second slices 4:00 -> 0:00
            [0.002, 0.013, 0.052, 0.088, 0.252, 0.677],
            [0.002, 0.006, 0.007, 0.024, 0.129, 0.649],
            [0.006, 0.003, 0.001, 0.008, 0.028, 0.288],
            [0.010, 0.005, 0.007, 0.007, 0.007, 0.055],
            [0.000, 0.000, 0.005, 0.007, 0.006, 0.089],
        ],
        [  # without it: down 9+, down 1-8, level, up 1-8, up 9+; 40-second slices 4:00 -> 0:00
            [0.050, 0.082, 0.308, 0.511, 0.596, 0.652],
            [0.008, 0.065, 0.281, 0.495, 0.685, 0.813],
            [0.012, 0.037, 0.214, 0.381, 0.553, 0.618],
            [0.008, 0.017, 0.121, 0.178, 0.249, 0.190],
            [0.009, 0.007, 0.018, 0.017, 0.019, 0.035],
        ],
    ],
]


def call_index(period, clock, lead, role):
    """The timeout-call cell for a side: quarter (2 or 4), role, its own lead, 40-second slice."""
    period = np.asarray(period)
    qi = np.where(period >= 4, 1, 0)             # overtime reads the fourth quarter's
    lead = np.asarray(lead)
    lc = np.where(lead <= -9, 0, np.where(lead < 0, 1, np.where(lead == 0, 2, np.where(lead <= 8, 3, 4))))
    cb = np.clip(((QUARTER - np.asarray(clock, dtype=float)) // (QUARTER / CLOCK_CELLS)).astype(np.int64),
                 0, CLOCK_CELLS - 1)
    return qi, np.asarray(role), lc, cb


def fit_timeout_calls(snaps):
    """How often each side stops a running clock with a timeout, by cell, among the snaps whose
    clock was running (or was stopped by a timeout) while it had one left; shrunk toward the
    quarter, role and slice over all scores. Zero where nothing is known."""
    n, k = np.zeros(N_CALL), np.zeros(N_CALL)
    for r in snaps:
        left = r.get("timeouts_left")
        if left is None or r["period"] not in TIMEOUT_QUARTERS or r.get("fresh"):
            continue
        by = r.get("timeout_role")
        if by is None and r["seconds"] <= STOP_SECONDS:
            continue                          # the clock had stopped by itself
        for role in (OFFENCE, DEFENCE):
            if left[role] <= 0:
                continue
            lead = r["margin"] if role == OFFENCE else -r["margin"]
            i = call_index(r["period"], r["clock"], lead, role)
            w = r.get("w", 1.0)
            n[i] += w
            k[i] += w * (by == role)
    base = (k.sum(2, keepdims=True) + 0.1) / (n.sum(2, keepdims=True) + 1.0)
    return np.where(n.sum(2, keepdims=True) > 0, (k + TIMEOUT_PRIOR * base) / (n + TIMEOUT_PRIOR), 0.0)


OT_CALL_PRIOR = 0.2        # shrink toward a scale of 0.2 until overtime shows otherwise
OT_CALL_DEFAULT = 0.23     # all 35 real overtime calls, 24 Aug - 22 Sep


def fit_ot_call_scale(snaps, call_p):
    """Overtime's timeouts are called like the fourth quarter's but far less often (35 calls in
    75 overtime periods): the fourth quarter's rates scaled by one factor -- the real calls over
    the calls the fourth quarter's rates would make on overtime's running clocks."""
    expected = called = 0.0
    for r in snaps:
        left = r.get("timeouts_left")
        if left is None or r["period"] < 5 or r.get("fresh"):
            continue
        by = r.get("timeout_role")
        if by is None and r["seconds"] <= STOP_SECONDS:
            continue
        for role in (OFFENCE, DEFENCE):
            if left[role] > 0:
                lead = r["margin"] if role == OFFENCE else -r["margin"]
                expected += float(call_p[call_index(r["period"], r["clock"], lead, role)])
                called += by == role
    return (called + 1.0) / (expected + 1.0 / OT_CALL_PRIOR) if expected > 0 else OT_CALL_DEFAULT


# v9: the clock fits -- the clock-stopped share, each state's play time, the kneels and the
# timeout calls -- weight each match by its age at the build's cut-off, halving every
# CLOCK_HALF_LIFE days (None: every match alike). Late in the fourth quarter leaders have bled the
# clock more week on week (a running play leading by 1-8 in the last two minutes: 32.0 seconds the
# week of 24 Aug, 33.2 three weeks on); elsewhere the clock has held still.
CLOCK_HALF_LIFE = 7.0


def match_date(code):
    """A match's date off its code (...DDMMYY), or None."""
    try:
        return dt.date(2000 + int(code[-2:]), int(code[-4:-2]), int(code[-6:-4]))
    except (ValueError, TypeError, IndexError):
        return None


def age_weights(records, as_of, half_life):
    """Set r["w"] on each record: 1, or halving every `half_life` days of its match's age."""
    for r in records:
        w = 1.0
        if as_of is not None and half_life:
            day = match_date(r.get("match"))
            if day is not None:
                w = 0.5 ** (max(0, (as_of - day).days) / half_life)
        r["w"] = w


# v9: the clock-running play's time by state is fitted on the plays nobody stopped. v8 fitted it
# on every running play, the ones a timeout stopped among them with a time borrowed from the bin
# (natural_seconds): where timeouts come thick -- the leader with the ball late in the fourth
# quarter -- that pulled the state's time toward the bin's (31.6-31.9 seconds against a real
# 32.8-34.0). And a play a timeout stops takes the time real ones did: it ran in bounds, and the
# clock ran on a moment before the call (the leader's, stopped by the trailing side: 8 seconds
# against 6.3 for a stopped play), by quarter, who called it and whether the caller is behind,
# level or ahead (to_secs): the fourth quarter's trailing side, calling from defence, lets the
# clock run on a while first (10.7 seconds on average, against 6-7 elsewhere).
NATURAL_SHIFT = True
TO_PLAY_SECONDS = True
TO_QUANTILES = np.linspace(0.0, 1.0, 21)


def caller_side(lead):
    """0 behind, 1 level, 2 ahead: the calling side's own score."""
    lead = np.asarray(lead)
    return np.where(lead < 0, 0, np.where(lead == 0, 1, 2))


def fit_timeout_seconds(snaps):
    """The time a play stopped by a timeout took, as quantiles, by quarter (2; 4 and overtime),
    who called it and the caller's score: [quarter][OFFENCE, DEFENCE][behind, level, ahead] ->
    21 quantiles; NaN where too few."""
    out = np.full((2, 2, 3, len(TO_QUANTILES)), np.nan)
    pools = defaultdict(list)
    for r in snaps:
        by = r.get("timeout_role")
        secs = r.get("seconds_real", r["seconds"])
        if by is None or r["period"] < 2 or r["period"] == 3:
            continue
        lead = r["margin"] if by == OFFENCE else -r["margin"]
        pools[(1 if r["period"] >= 4 else 0, by, int(caller_side(lead)))].append(secs)
    for (qi, role, side), xs in pools.items():
        if len(xs) >= 30:
            out[qi, role, side] = np.quantile(xs, TO_QUANTILES)
    return out


def natural_seconds(snaps):
    """A real play that followed a timeout used next to nothing of the clock: give it the time of
    a clock-running play in the same bin (else situation and down), so the tables' clock is the
    one without timeouts and the simulation's own calls take it off."""
    pools = defaultdict(list)
    for r in snaps:
        if r.get("timeout_role") is None and r["seconds"] > STOP_SECONDS:
            pools[("k", r["key"])].append(r["seconds"])
            pools[("m", r["mode"], r["down"])].append(r["seconds"])
            pools[("all",)].append(r["seconds"])
    rng = np.random.default_rng(0)
    for r in snaps:
        if r.get("timeout_role") is None:
            continue
        pool = next((pools[g] for g in (("k", r["key"]), ("m", r["mode"], r["down"]), ("all",))
                     if len(pools[g]) >= 5), None)
        if pool:
            r["seconds_real"] = r["seconds"]
            r["seconds"] = float(pool[rng.integers(len(pool))])


# v8: the leader's kneel-downs in the fourth quarter are plays of their own, called as often as
# real leaders call them by down, 40-second slice of the clock and one score or more (kneel_p):
# the play tables pool a leader's plays over the whole of the quarter's end, so the kneels in them
# came as often at 1:50 as at 0:50. A kneel loses a yard or two and takes a play clock -- or next
# to nothing when the trailing team calls a timeout. Real kneels (a leader's loss of 1 to 4 yards
# on downs 1-3) leave the tables.
KNEELS = True
KNEEL_PRIOR = 10.0
KNEEL_LOSS = (-4, -1)
N_KNEEL = (2, 3, CLOCK_CELLS)


def is_kneel(rec):
    """Whether a snap was a leader's kneel-down in the fourth quarter: a loss of 1 to 4 yards on
    downs 1-3 that kept the ball."""
    return (rec["period"] == 4 and rec["margin"] > 0 and rec["down"] <= 3 and rec["kind"] == GAIN
            and KNEEL_LOSS[0] <= rec["gain"] <= KNEEL_LOSS[1])


def kneel_index(margin, down, clock):
    """The kneel cell: one score or more, down, 40-second slice."""
    big = np.asarray(margin) >= BIG_LEAD
    cb = np.clip(((QUARTER - np.asarray(clock, dtype=float)) // (QUARTER / CLOCK_CELLS)).astype(np.int64),
                 0, CLOCK_CELLS - 1)
    return big.astype(np.int64), np.clip(np.asarray(down), 1, 3) - 1, cb


def fit_kneels(snaps):
    """How often the fourth-quarter leader kneels, by cell, shrunk toward its lead and down; and
    the clock a kneel takes when nothing stops it."""
    n, k = np.zeros(N_KNEEL), np.zeros(N_KNEEL)
    secs = []
    for r in snaps:
        if r["period"] != 4 or r["margin"] <= 0 or r["down"] > 3:
            continue
        i = kneel_index(r["margin"], r["down"], r["clock"])
        w = r.get("w", 1.0)
        n[i] += w
        if r.get("kneel"):
            k[i] += w
            if r["seconds"] > STOP_SECONDS and r["clock"] >= UNCUT_SECONDS:
                secs.append(r["seconds"])
    base = (k.sum(2, keepdims=True) + 1) / (n.sum(2, keepdims=True) + 2)
    p = (k + KNEEL_PRIOR * base) / (n + KNEEL_PRIOR)
    return p, (np.array(secs) if len(secs) >= 20 else np.array([PLAY_CLOCK - 3.0]))


def _slice_of(clock):
    """The 40-second slice of a quarter, 0 (4:00) to 5 (the last 40 seconds)."""
    return np.clip(((QUARTER - np.asarray(clock, dtype=float)) // (QUARTER / CLOCK_CELLS)).astype(np.int64),
                   0, CLOCK_CELLS - 1)


def fit_ot_stop(tables, on, stop_shift):
    """v8: overtime's own shift to the clock-stopped share, by 40-second slice. Overtime reads the
    fourth quarter's cells, whose shares no longer hold the timeouts' stops (the calls add them
    back in the fourth quarter); real sides hardly call timeouts in overtime, so its clock stops
    on its own more than the fourth quarter's natural share."""
    ot = [r for r in on if r["period"] >= 5]
    shift = np.zeros(CLOCK_CELLS)
    if not ot:
        return shift
    base = (_running_logit(tables, np.array([r["key"] for r in ot], dtype=np.int64))[0]
            + stop_shift[np.array([cell_index(r["period"], r["clock"], r["margin"]) for r in ot])])
    sl = _slice_of([r["clock"] for r in ot])
    stop = np.array([r["seconds"] <= STOP_SECONDS for r in ot])
    pen = 0.5 * STOP_PRIOR * 0.25 * STOP_GRID ** 2
    for c_ in range(CLOCK_CELLS):
        ix = np.flatnonzero(sl == c_)
        if not len(ix):
            continue
        p = np.clip(_sigmoid(base[ix, None] + STOP_GRID[None, :]), 1e-9, 1 - 1e-9)
        ll = np.where(stop[ix, None], np.log(p), np.log(1 - p)).sum(0)
        shift[c_] = STOP_GRID[np.argmax(ll - pen)]
    return shift


def clock_end_records(rows):
    """v8: every scrimmage snap on downs 1-3 that ran its quarter out -- the next PLAY_OVER is in a
    later quarter, or it was the game's last -- with its state and the seconds it had left."""
    out = []
    for i, a in enumerate(rows):
        if a["play_kind"] not in SNAP_KINDS or not a["down"] or not a["field_position"] \
                or not a["distance"] or not a["period"] or not a["clock_seconds"]:
            continue
        p, c = _i(a["period"]), _f(a["clock_seconds"])
        b = rows[i + 1] if i + 1 < len(rows) else None
        if p is None or p > 4 or c is None or c <= 0:
            continue
        if b is None:
            if p != 4:
                continue
        elif not b["period"] or _i(b["period"]) == p:
            continue
        down, t, y = _i(a["down"]), max(1, _i(a["distance"])), _i(a["field_position"])
        m = _margin(a, a["offense"])
        if down is None or down >= 4 or y is None or m is None:
            continue
        mode = mode_of(p, c, m, BIG_LEAD, Q4_MODES)
        out.append(dict(period=p, clock=c, margin=m, down=down, distance=t, field=y, seconds=c,
                        key=key_index(mode, down, t, y), fresh=FRESH_CLOCK and is_fresh(a)))
    return out


def fit_play_calling(tables, snaps, ends=(), kneels=()):
    """By game state: how often clock-stopping plays are called, how long plays take, and the
    rubber band. The clock is fitted on snaps that follow a scrimmage play: a fresh possession's
    first snap is played with the clock stopped whatever the state. With LATE_CLOCK (v8) the
    snaps that ran a quarter out are kept as censored, and the clock is fitted to the last second."""
    n = len(snaps)
    on = [r for r in snaps if not r.get("fresh")]
    key = np.array([r["key"] for r in on])
    cell = np.array([cell_index(r["period"], r["clock"], r["margin"]) for r in on])
    stop = np.array([r["seconds"] <= STOP_SECONDS for r in on])
    secs = np.array([r["seconds"] for r in on])
    uncut = np.array([r["clock"] >= UNCUT_SECONDS for r in on])
    base = _running_logit(tables, key)[0]
    stop_shift = np.zeros(N_CELLS)
    for _ in range(10):
        p = _sigmoid(base + stop_shift[cell])
        g = np.bincount(cell, (stop - p) * uncut, N_CELLS) - STOP_PRIOR * 0.25 * stop_shift
        h = np.bincount(cell, p * (1 - p) * uncut, N_CELLS) + STOP_PRIOR * 0.25
        stop_shift += g / h
    if LATE_CLOCK:
        ends_on = [r for r in ends if not r.get("fresh")]
        stop_shift = _fit_stop_censored(tables, on, ends_on)
        if TIMEOUTS:
            tables.ot_stop = fit_ot_stop(tables, on, stop_shift * _quarter_mask("stop"))

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
        # a stopped play (12 seconds or less) is never cut short with more than 12 left
        sel = (kind == c_) & (uncut if c_ == RUNNING or not LATE_CLOCK
                              else np.array([r["clock"] > STOP_SECONDS for r in on]))
        if NATURAL_SHIFT:
            sel &= np.array([r.get("timeout_role") is None for r in on])
        wt = np.array([r.get("w", 1.0) for r in on])
        tot = np.bincount(cell[sel], resid[sel] * wt[sel], N_CELLS)
        num = np.bincount(cell[sel], wt[sel], N_CELLS)
        sec_shift[c_] = tot / (num + SECONDS_PRIOR)
    if LATE_CLOCK:
        # a running play in the last 40 seconds is nearly always cut short: it takes the slice
        # before's shift
        by = sec_shift[RUNNING].reshape(4, CLOCK_CELLS, LEAD_CELLS)
        by[:, CLOCK_CELLS - 1] = by[:, CLOCK_CELLS - 2]
    key = np.array([r["key"] for r in snaps])
    cell = np.array([cell_index(r["period"], r["clock"], r["margin"]) for r in snaps])
    stop = np.array([r["seconds"] <= STOP_SECONDS for r in snaps])
    succ = np.array([bool(r["success"]) for r in snaps])
    f = np.clip(np.where(stop, tables.stop_success[key], tables.run_success[key]), 1e-3, 1 - 1e-3)
    grid = np.linspace(-1.0, 1.0, 201)
    eff_shift = np.zeros(N_CELLS)
    for c_ in range(N_CELLS):
        on = cell == c_
        if not on.any():
            continue
        lf, s_ = np.log(f[on]), succ[on]
        pa = np.exp(np.outer(np.exp(-grid), lf))
        ll = np.where(s_, np.log(np.clip(pa, 1e-12, 1)), np.log(np.clip(1 - pa, 1e-12, 1))).sum(1)
        eff_shift[c_] = grid[np.argmax(ll - 0.5 * EFF_PRIOR * grid ** 2)]
    tables.stop_shift = (stop_shift * _quarter_mask("stop") if "stop" in PLAY_CALLING
                         else np.zeros(N_CELLS))
    tables.sec_shift = (sec_shift * _quarter_mask("seconds") if "seconds" in PLAY_CALLING
                        else np.zeros((2, N_CELLS)))
    tables.eff_shift = (eff_shift * _quarter_mask("band") if "band" in PLAY_CALLING
                        else np.zeros(N_CELLS))
    return n


RED_ZONE = 70
RED_ZONE_FIT = True
RED_ZONE_MODE = "hold"   # or "tilt"
RED_ZONE_PRIOR = 40.0
N_RZ = 4 * LEAD_CELLS * 2
RZ_GRID = np.linspace(-1.5, 1.0, 51)
HOLD_GRID = np.linspace(0.0, 0.8, 41)
HOLD_TRIES = 3


SETTLE_FIT = True
N_SETTLE_SEGMENTS = 6
N_SETTLE = N_SETTLE_SEGMENTS * LEAD_CELLS * 4
SETTLE_MAX = 0.8


def settle_index(period, clock, lead, y):
    """The cell a would-be touchdown is held in: the part of the game (Q1, Q2, Q2's last two
    minutes, Q3, Q4, Q4's last two minutes and overtime), the offense's lead and the field zone."""
    period, clock = np.asarray(period), np.asarray(clock)
    late = clock <= LATE
    seg = np.where(period <= 1, 0, np.where(period == 2, np.where(late, 2, 1),
                   np.where(period == 3, 3, np.where(late, 5, 4))))
    lc = np.where(lead <= -9, 0, np.where(lead < 0, 1, np.where(lead == 0, 2, np.where(lead <= 8, 3, 4))))
    yy = np.asarray(y)
    z = np.where(yy < 40, 0, np.where(yy < 70, 1, np.where(yy < 90, 2, 3)))
    return (seg * LEAD_CELLS + lc) * 4 + z


def rz_index(period, lead, y):
    """The red-zone cell: quarter, the offense's lead, and the 11-30 or inside the 10."""
    pq = np.clip(period, 1, 4) - 1
    lc = np.where(lead <= -9, 0, np.where(lead < 0, 1, np.where(lead == 0, 2, np.where(lead <= 8, 3, 4))))
    return (pq * LEAD_CELLS + lc) * 2 + (np.asarray(y) >= 90)


def _td_chance(tables, key, cell, y, tilts):
    """The chance a snap's draw scores, at each tilt: straight from the gain, or a touchdown seen
    from further back running on."""
    n, ns, a = tables.count[key], tables.n_stop[key], tables.start[key]
    logit, ns0, n0 = _running_logit(tables, key)
    ps = 0.0 if ns == 0 else 1.0 if ns == n else 1.0 / (1.0 + math.exp(-(logit + tables.stop_shift[cell])))
    out = np.zeros(len(tilts))
    for w, s0, sn in ((ps, a, ns), (1.0 - ps, a + ns, n - ns)):
        if w == 0 or sn == 0:
            continue
        g = tables.gain[s0:s0 + sn]
        tf = tables.td_from[s0:s0 + sn]
        ok = tables.kind[s0:s0 + sn] == GAIN
        ptd = np.where(ok & (g >= 100 - y), 1.0,
                       np.where(ok & (tf >= 0) & (y < tf), np.exp(-(tf - y) / TD_TAIL), 0.0))
        x = np.arange(sn + 1) / sn
        cdf = 1.0 - (1.0 - x[None, :]) ** (1.0 / tilts[:, None])
        out += w * (np.diff(cdf, axis=1) * ptd[None, :]).sum(1)
    return out


def fit_red_zone(tables, snaps):
    """How often real snaps inside the 30 score, against the bins, by quarter, lead and zone: a
    fourth-quarter leader there settles for the field goal far more than a third-quarter one."""
    tables.rz_shift, tables.rz_hold = np.zeros(N_RZ), np.zeros(N_RZ)
    if not RED_ZONE_FIT:
        return
    hold = RED_ZONE_MODE == "hold"
    grid = HOLD_GRID if hold else RZ_GRID
    ll = np.zeros((N_RZ, len(grid)))
    for r in snaps:
        y = r["field"]
        if y < RED_ZONE:
            continue
        cell = cell_index(r["period"], r["clock"], r["margin"])
        if hold:
            p = _td_chance(tables, r["key"], cell, y, np.exp([tables.eff_shift[cell]]))[0] * (1 - grid)
        else:
            p = _td_chance(tables, r["key"], cell, y, np.exp(grid + tables.eff_shift[cell]))
        p = np.clip(p, 1e-4, 1 - 1e-4)
        scored = r["kind"] == GAIN and r["gain"] >= 100 - y
        ll[int(rz_index(r["period"], r["margin"], y))] += np.log(p if scored else 1 - p)
    fitted = grid[np.argmax(ll - 0.5 * RED_ZONE_PRIOR * grid ** 2, axis=1)]
    if hold:
        tables.rz_hold = fitted
    else:
        tables.rz_shift = fitted


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
FRESH_CLOCK = True
FRESH_STOP = 0.955
FRESH_PRIOR = 20.0


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
    """A team's lead on a row's scoreboard."""
    home, away = _i(row["score_p1"]), _i(row["score_p2"])
    if home is None or away is None or team not in ("TEAM_A", "TEAM_B"):
        return None
    return (home - away) * (1 if team == "TEAM_A" else -1)


def _is_big(rec):
    """A snap that gained BIG_GAIN yards or more (v13's big plays)."""
    return rec["kind"] == GAIN and rec["gain"] >= BIG_GAIN


def _is_fail(rec):
    """A snap that failed: no first down (a turnover and a defensive touchdown too), the bottom of
    its bin's order."""
    return not _is_big(rec) and (rec["kind"] != GAIN or rec["gain"] - rec["distance"] < 0)


def snap_records(rows):
    """Every scrimmage snap of a match: its state, what came of it, the clock it used and whether
    it started a possession with the clock stopped."""
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
                   fresh=is_fresh(a))
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
        rec["mode"] = mode_of(rec["period"], rec["clock"], rec["margin"], BIG_LEAD, Q4_MODES)
        rec["key"] = key_index(rec["mode"], down, t, y)
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


# v11: a kick-off returned for a touchdown (90 since 24 Aug, about one game in 25) or recovered and
# run in by the kicking side is kept, at field 100: the side with the ball scores. v10's kick-offs
# never ended in a touchdown -- the row after one is the touchdown, not a kick-off, and was skipped.
KICK_TDS = True

# v14: kick-offs by the clock and by the kicker. In Q2's and Q4's last two minutes 13.7% of real
# kick-offs are touchbacks (the 20, no clock) and 5.8% land in no landing zone (the 35, no clock),
# against 2.8% and 2.4% the rest of the game, and the drives after a late touchback make 1.28 points
# against 2.10 after a return. Some kickers do it far more (SHROUD into no landing zone a third of
# the time, FUSE / MERLIN / FENRIR touchbacks one kick in six). v13 drew every kick from one pool.
# Now each kick draws a touchback or no landing zone with the league's chance for its part of the
# game (late: Q2 or Q4 at KICK_LATE seconds or less), times the kicking side's own (Start.kick_tb,
# Start.kick_nlz, from the players' profiles), and otherwise a kick from that part's other kicks.
KICK_STYLES = True
KICK_LATE = 120.0
KICK_TOUCHBACK, KICK_NO_LANDING = 20, 35
KICK_STYLE_MAX = 0.9
KICK_STYLE_MIN = 30          # kicks a part of the game needs before it gets its own pool


# v14: a drive carried from Q1 into Q2 (or Q3 into Q4) does not snap at 4:00 on a stopped clock.
# In every one of the feed's 4,092 carries the clock had run 3-8 seconds (median 6) before the new
# quarter's first snap, and that snap then took a running clock's time (27 seconds on average,
# 22% within 8). Played at 4:00 on a stopped clock the sim gained some 25 seconds at the start of
# each, a snap a game in Q2 (sim_audit: Q2 15.7 snaps a game, real 14.7). QUARTER_CARRY plays the
# real run-off (Tables.runoff, from quarter_runoffs) and a running clock.
QUARTER_CARRY = True
DEFAULT_RUNOFF = (6.0,)


def quarter_runoffs(rows):
    """Seconds gone from the clock before each new quarter's first snap, where a drive carries from
    Q1 into Q2 or from Q3 into Q4."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["period"] in ("1", "3") and b["period"] == str(int(a["period"]) + 1) \
                and b["play_kind"] in SNAP_KINDS and b["clock_seconds"]:
            gone = QUARTER - _f(b["clock_seconds"])
            if 0 <= gone <= 30:
                out.append(gone)
    return out


def kick_kinds(rows):
    """Every kick-off that isn't a desperate one (an onside try, behind late): (late, touchback,
    no landing zone, onside, the receiver's start, the clock used, the kicking side TEAM_A / TEAM_B)."""
    out = []
    prev = None
    for r in rows:
        if r["play_kind"] == "KICKOFF" and r["field_position"] and r["clock_seconds"] and r["period"]:
            p = _i(r["period"])
            after = prev is not None and prev["period"] == r["period"] and \
                prev["play_kind"] in ("CONVERSION", "FIELD_GOAL") and prev["clock_seconds"]
            opener = prev is None or prev["period"] != r["period"]
            if after or opener:
                start_clock = _f(prev["clock_seconds"]) if after else QUARTER
                seconds = start_clock - _f(r["clock_seconds"])
                kicker = prev["offense"] if after else ("TEAM_B" if r["offense"] == "TEAM_A" else "TEAM_A")
                margin = _margin(prev, kicker) if after else 0
                desperate = after and p >= 4 and margin is not None and margin < 0 \
                    and half_left(p, start_clock) <= 180
                field = _i(r["field_position"])
                if KICK_TDS and _scorer(r, "TOUCHDOWN_TEAM") is not None:
                    field = 100
                onside = r["offense"] == kicker
                if 0 <= seconds <= 60 and not desperate and field is not None:
                    late = p in (2, 4) and start_clock <= KICK_LATE
                    tb = field == KICK_TOUCHBACK and seconds == 0 and not onside
                    nlz = field == KICK_NO_LANDING and seconds == 0 and not onside
                    out.append((late, tb, nlz, onside, field, seconds, kicker))
        prev = r
    return out


def fit_kick_styles(kinds):
    """{late: (onside, start, seconds) of the kicks neither a touchback nor in no landing zone},
    and the league's (touchback, no landing zone) chance for each part of the game: a (2, 2) array."""
    pools, mix = {}, np.zeros((2, 2))
    for late in (False, True):
        ks = [k for k in kinds if k[0] == late]
        if len(ks) < KICK_STYLE_MIN:
            ks = kinds
        if not ks:
            return None, None
        mix[int(late)] = (np.mean([k[1] for k in ks]), np.mean([k[2] for k in ks]))
        rest = [k for k in ks if not (k[1] or k[2])] or ks
        pools[late] = (np.array([k[3] for k in rest], dtype=bool), np.array([k[4] for k in rest], dtype=np.int32),
                       np.array([k[5] for k in rest], dtype=float))
    return pools, mix


def kick_records(rows):
    """Every kickoff: kept by the kicking side or not, where the side with the ball started (100: it
    scored), and the clock used."""
    out = []
    prev = None
    for r in rows:
        if KICK_TDS and r["play_kind"] == "TOUCHDOWN" and prev is not None \
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
                if KICK_TDS and _scorer(r, "TOUCHDOWN_TEAM") is not None:
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


SAFETY_KICK_MIN = 20
SAFETY_KICK_SHIFT = 16


BACKED_UP = 10
BACKED_PRIOR = 60.0
BACKED_OUTCOMES = ("safety", "def_td", "turnover", "td")
B_SAFETY, B_DEF_TD, B_TURNOVER, B_TD = range(4)
BACKED_RETURNS_MIN = 20


def backed_up_snaps(rows):
    """Snaps from the offense's own 1 to 10 and what came of them."""
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
            continue
        out.append((y, outcome, back))
    return out


def fit_backed_up(records, prior=BACKED_PRIOR):
    """The chance of a safety, defensive touchdown, turnover or touchdown at each yard line inside
    the own 10."""
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
    """A logistic curve in yard line fitted to counts."""
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


CONV_MARGIN = 16
# v11: the conversion after a fourth-quarter touchdown in the last three minutes, by the clock. Real
# sides 7 behind who score go for two to win on 94% of tries in the last 30 seconds, 70% to 1:00, 37%
# to 2:00 and 12% to 3:00 (2,320 matches); v10 had one rate, 66%, for the whole three minutes, so it
# kicked to tie, and went to overtime, with seconds left. Cells: the three minutes split at
# CONV_CLOCK, then overtime on its own; each shrinks toward v10's one cell for them all.
CONV_CLOCK = (30.0, 60.0, 120.0)
CONV_CELLS = 4 + len(CONV_CLOCK) + 1


def conversion_cell(period, clock):
    """The cell of the go-for-two table: the decision phase, or with CONV_CLOCK set, the last
    three minutes' clock slice (4, 5, ...; 3 from 2:00 to 3:00) or overtime (the last cell)."""
    phase = conversion_phase(period, clock)
    if phase != 3 or not CONV_CLOCK:
        return phase
    if period >= 5:
        return CONV_CELLS - 1
    for i, edge in enumerate(CONV_CLOCK):
        if clock <= edge:
            return 4 + i
    return 3



def fit_go_for_two(conv):
    """P(go for two) by cell and margin after the six. The four phases shrink toward their own
    rate; the late cells (the last three minutes' slices and overtime) toward the four's last."""
    size = 2 * CONV_MARGIN + 1
    tries, twos = np.zeros((CONV_CELLS, size)), np.zeros((CONV_CELLS, size))
    for cell, margin, two, _ in conv:
        tries[cell, margin + CONV_MARGIN] += 1
        twos[cell, margin + CONV_MARGIN] += two
    tries4 = np.vstack([tries[:3], tries[3:].sum(0)])
    twos4 = np.vstack([twos[:3], twos[3:].sum(0)])
    base = twos4.sum(1, keepdims=True) / np.maximum(1, tries4.sum(1, keepdims=True))
    pooled = (twos4 + CONV_PRIOR * base) / (tries4 + CONV_PRIOR)
    return np.vstack([pooled[:3], (twos[3:] + CONV_PRIOR * pooled[3]) / (tries[3:] + CONV_PRIOR)])

def conversion_phase(period, clock):
    """The part of the game for a conversion decision."""
    if period <= 2:
        return 0
    if period == 3:
        return 1
    return 3 if (period >= 5 or clock <= 180) else 2


def conversions(rows):
    """Every conversion: when, the margin, and whether they went for two."""
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
        out.append((conversion_cell(_i(a["period"]), _f(a["clock_seconds"]) or 0.0),
                    max(-CONV_MARGIN, min(CONV_MARGIN, margin)), two, good))
    return out


def decision_phase(period, clock):
    """The part of the game for a 4th-down decision."""
    return conversion_phase(period, clock)


def margin_bucket(margin):
    """The offense's lead as a bucket."""
    return (0 if margin <= -9 else 1 if margin <= -4 else 2 if margin < 0 else 3 if margin == 0
            else 4 if margin <= 3 else 5 if margin <= 8 else 6)


def _margin_bucket_np(m):
    """margin_bucket for arrays."""
    return np.where(m <= -9, 0, np.where(m <= -4, 1, np.where(m < 0, 2, np.where(
        m == 0, 3, np.where(m <= 3, 4, np.where(m <= 8, 5, 6))))))


EARLY_FG_RANGES = (45, 55, 65)
EARLY_FG_RANGE = EARLY_FG_RANGES[-1]
EARLY_FG_CLOCK = (5, 10, 20, 30, 45)


def _clock_bin(clock):
    """Seconds left as a bin for early field goals."""
    for i, edge in enumerate(EARLY_FG_CLOCK):
        if clock <= edge:
            return i
    return None


def early_kicks(rows):
    """Snaps in range as a half runs out, and whether they kicked early."""
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


# v11: close endings. In the fourth quarter a side level or 1-3 points behind, in range on downs 1-3,
# kicks as real sides do: by how far behind (level, 1-2, 3), the down, the clock left (to 2:00) and the
# kick's length. Real sides almost never kick early with more than 45 seconds left, and one 3 behind
# hardly ever kicks to tie before the last 10; v10 instead drained the clock to 0:00 and kicked from 55
# yards whenever the downs left could run it out (the drain stays for overtime alone). A 4th down
# level, 1-2 behind or 3 behind in range in the last three minutes kicks at its own fitted rate,
# not one rate for both trailing classes and the whole-game 4th-down curve for level sides (which went
# for it on one 4th down in seven in range in the last minute; real level sides kick).
CLOSE_FG = True
CLOSE_FG_CLOCK = (5, 10, 20, 30, 45, 60, 120)
CLOSE_FG_RANGES = (35, 45, 55, 65)
CLOSE_FG_PRIOR = 5.0
CLOSE_CLASSES = ("level", "1-2 behind", "3 behind")
# A side level or 1-2 behind in range (a kick of SETTLE_RANGE yards or less) in the fourth quarter's
# last SETTLE_CLOCK seconds plays for the kick: about 3 yards and 18 seconds a play, a touchdown on
# one in ten, almost never a turnover (real plays before 10 Sep). Its plays are drawn from those real
# plays alone -- two extra play bins after the N_KEYS keyed ones, level and 1-2 behind -- where
# before they fell back to bins of every late play, most of them going for the end zone.
SETTLE_PLAYS = True
SETTLE_RANGE = 50
SETTLE_CLOCK = 120.0
SETTLE_MIN = 30


# v12: a close 4th down's kick by the clock. Real level sides in range kick on 4th down 45% of the time
# with 1:00-3:00 left, 82% from 0:31 to 1:00 and 92% and more after; going for it keeps the drive and
# the clock running, so the kick comes late. v11 had two cells, inside and outside 30 seconds, and its
# fitted level kick shift took the outside one to 98%. FOURTH_CLOCK splits the three minutes at these
# edges; each cell shrinks toward v11's cell it falls in (FOURTH_PRIOR).
FOURTH_CLOCK = (10.0, 30.0, 60.0, 120.0)
FOURTH_PRIOR = 5.0
# v12: the clock a close side's late kick takes. Real sides level or 1-3 behind kicking in the fourth
# quarter's last 2:00 let the play clock run first: from 0:40 or less, 65% of their kicks leave nothing
# on the clock (v11: the whole game's kick seconds, a median of 4); from 0:41 to 2:00, a third of them
# take 25 seconds or more. KICK_CLOCK draws those kicks' seconds from the real ones (Tables.kick_clock).
KICK_CLOCK = True
KICK_CLOCK_EDGE = 40.0
KICK_CLOCK_MIN = 20


def close_fourth_cells(clock):
    """The clock cell of a close 4th down: v11's (1 inside 30 seconds, else 0) or, with FOURTH_CLOCK,
    the slice (0 for the last 10 seconds, ...)."""
    if not FOURTH_CLOCK:
        return (np.asarray(clock) <= 30).astype(np.int64)
    return np.searchsorted(np.array(FOURTH_CLOCK), clock).astype(np.int64)


def fit_close_fourths(records):
    """P(kick) on a close 4th down by class and clock cell, from (class, clock, kicked): v11's two
    cells, then with FOURTH_CLOCK the finer slices, each shrunk toward the v11 cell it falls in."""
    n2, k2 = np.zeros((len(CLOSE_CLASSES), 2)), np.zeros((len(CLOSE_CLASSES), 2))
    for cls, c, kicked in records:
        n2[cls, int(c <= 30)] += 1
        k2[cls, int(c <= 30)] += kicked
    two = (k2 + 1.0) / (n2 + 2.0)
    if not FOURTH_CLOCK:
        return two
    cells = len(FOURTH_CLOCK) + 1
    n, k = np.zeros((len(CLOSE_CLASSES), cells)), np.zeros((len(CLOSE_CLASSES), cells))
    for cls, c, kicked in records:
        j = int(close_fourth_cells(c))
        n[cls, j] += 1
        k[cls, j] += kicked
    edges = np.array((0.0,) + tuple(FOURTH_CLOCK))
    parent = two[:, (edges < 30).astype(np.int64)]               # the v11 cell each slice falls in
    return (k + FOURTH_PRIOR * parent) / (n + FOURTH_PRIOR)


def close_kick_times(rows):
    """Field goals in the fourth quarter's last 2:00 by a side level or 1-3 behind: (clock when the
    kicking down began, seconds the kick took)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if b["play_kind"] != "FIELD_GOAL" or a["period"] != "4" or b["period"] != "4" \
                or not a["clock_seconds"] or not b["clock_seconds"]:
            continue
        margin = _margin(a, a["offense"])
        c0, c1 = _f(a["clock_seconds"]), _f(b["clock_seconds"])
        if margin is None or not -3 <= margin <= 0 or not 0 < c0 <= 120 or c1 > c0:
            continue
        out.append((c0, c0 - c1))
    return out


def fit_kick_clock(records):
    """(P(the kick takes the clock to 0) from KICK_CLOCK_EDGE or less, the seconds the kicks from
    further out took), or None with too few of either."""
    near = [used >= c0 for c0, used in records if c0 <= KICK_CLOCK_EDGE]
    far = [used for c0, used in records if c0 > KICK_CLOCK_EDGE]
    if len(near) < KICK_CLOCK_MIN or len(far) < KICK_CLOCK_MIN:
        return None
    return (sum(near) + 1.0) / (len(near) + 2.0), np.array(far, dtype=float)


def close_class(margin):
    """0 level, 1 one or two behind, 2 three behind (arrays or a number)."""
    return np.where(margin >= 0, 0, np.where(margin >= -2, 1, 2))


def close_kicks(rows):
    """Fourth-quarter snaps on downs 1-3 in range in the last 2:00, level or 1-3 behind: (class, down
    - 1, clock bin, range bin, kicked)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in SNAP_KINDS or a["down"] not in ("1", "2", "3") or a["period"] != "4" \
                or b["period"] != "4" or not a["field_position"] or not a["clock_seconds"]:
            continue
        margin = _margin(a, a["offense"])
        kick = 100 - _i(a["field_position"]) + 17
        c = _f(a["clock_seconds"])
        if margin is None or not -3 <= margin <= 0 or kick > CLOSE_FG_RANGES[-1] or c > CLOSE_FG_CLOCK[-1]:
            continue
        out.append((int(close_class(margin)), _i(a["down"]) - 1,
                    int(np.searchsorted(CLOSE_FG_CLOCK, c)), int(np.searchsorted(CLOSE_FG_RANGES, kick)),
                    b["play_kind"] == "FIELD_GOAL"))
    return out


def fit_close_fg(records, prior=CLOSE_FG_PRIOR):
    """P(field goal now) by class, down, clock bin and range bin. Each cell is shrunk toward its
    class, clock and range over the downs; that toward its class and clock; that toward the clock
    over every class -- a side 3 behind borrows from its own class first, not from level sides,
    who kick far more."""
    shape = (len(CLOSE_CLASSES), 3, len(CLOSE_FG_CLOCK), len(CLOSE_FG_RANGES))
    n, k = np.zeros(shape), np.zeros(shape)
    for cls, d, cb, rb, kicked in records:
        n[cls, d, cb, rb] += 1
        k[cls, d, cb, rb] += kicked
    p_clock = (k.sum((0, 1, 3)) + 0.05) / (n.sum((0, 1, 3)) + 1.0)
    p_cc = (k.sum((1, 3)) + prior * p_clock) / (n.sum((1, 3)) + prior)
    p_ccr = (k.sum(1) + prior * p_cc[:, :, None]) / (n.sum(1) + prior)
    return (k + prior * p_ccr[:, None]) / (n + prior)


def close_fourths(rows):
    """Fourth-quarter 4th downs level or 1-3 points behind in the last three minutes within 60 yards
    of a kick, not punted: (close class, clock, kicked)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["down"] != "4" or a["play_kind"] not in SNAP_KINDS or a["period"] != "4" \
                or b["period"] != "4" or not a["field_position"] or not a["clock_seconds"]:
            continue
        margin = _margin(a, a["offense"])
        c = _f(a["clock_seconds"])
        if margin is None or not -3 <= margin <= 0 or c > 180 or 100 - _i(a["field_position"]) + 17 > 60 \
                or b["play_kind"] == "PUNT":
            continue
        out.append((int(close_class(margin)), c, b["play_kind"] == "FIELD_GOAL"))
    return out


def fourth_down_choices(rows):
    """Every 4th down and what the offense chose: go, kick or punt."""
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
    """The league's log-odds of going for it on 4th down."""
    u = y / 100.0
    lt = math.log(max(1, t))
    red = 1.0 if y >= 80 else 0.0
    c = dp.go_coef
    return c[0] + c[1] * lt + c[2] * u + c[3] * u * u + c[4] * lt * u + c[5] * red + c[6] * lt * red


def _fg_logit(dp, y):
    """The league's log-odds of kicking a field goal on 4th down."""
    a, b = dp.fg_kick_coef
    return a + b * y / 100.0


def fit_decision_shifts(decisions, dp, prior=DECISION_PRIOR, iterations=8):
    """4th-down go and kick rates by part of the game and margin, against the league curve."""
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


FOURTH_JOINT = True
OT_RULES = True
# v8: an overtime period's clock running out ends the game only when a side leads and the side
# behind has had its possession; otherwise play carries on into the next period with the same
# ball, down and spot, as between quarters (39% of real overtimes reach a second period; none
# ended level). Before, a lead ended it at the clock and a level score kicked off again.
OT_CARRY = True
OT_KICK_RANGE = 55
OT_GO_PRIOR = 2.0


def ot_first_fourths(rows):
    """Every overtime 4th down in field-goal range by a side level or ahead: whether it went for
    it (a field goal there does not end the game while the other side has yet to have the ball)."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["down"] != "4" or a["play_kind"] not in SNAP_KINDS or not a["field_position"] \
                or not a["period"] or a["period"] != b["period"] or _i(a["period"]) < 5:
            continue
        margin = _margin(a, a["offense"])
        if margin is None or margin < 0 or 100 - _i(a["field_position"]) + 17 > OT_KICK_RANGE:
            continue
        out.append(b["play_kind"] not in ("PUNT", "FIELD_GOAL"))
    return out
CURVE_RIDGE = 0.01
PLAYER_PRIOR = 2.0


def _offense_handle(row, pair):
    """The handle of the side with the ball, from (home handle, away handle)."""
    if not pair or row.get("offense") not in ("TEAM_A", "TEAM_B"):
        return None
    a_home = (row.get("team_a_side") or "home") == "home"
    return pair[0] if (row["offense"] == "TEAM_A") == a_home else pair[1]


def fourth_down_records(rows, pair=None):
    """Every 4th down the late-trailing rules do not decide: part of the game, margin bucket,
    field, distance, choice and the handle of the side with the ball."""
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
        if margin < 0 and (p >= 5 or (p >= 4 and c <= 180)):
            continue
        choice = ("punt" if b["play_kind"] == "PUNT" else "fg" if b["play_kind"] == "FIELD_GOAL"
                  else "go")
        out.append((decision_phase(p, c), margin_bucket(margin), _i(a["field_position"]),
                    max(1, _i(a["distance"])), choice, _offense_handle(a, pair)))
    return out


def _go_basis(y, t):
    """The league go curve's terms: log distance, field position and the red zone."""
    u = np.asarray(y, dtype=float) / 100.0
    lt = np.log(np.maximum(1, np.asarray(t, dtype=float)))
    red = (np.asarray(y) >= 80).astype(float)
    return np.stack([np.ones_like(u), lt, u, u * u, lt * u, red, lt * red], axis=1)


def _ridge_logistic(basis, cells, players, outcome, cell_prior, player_prior, iterations=25):
    """Log-odds fit of `outcome` on the curve's basis, a shift per (phase, margin) cell and one per
    player, each shift shrunk toward zero: (curve coefficients, 4x7 shifts, {player: shift})."""
    names = sorted({h for h in players if h})
    col = {h: i for i, h in enumerate(names)}
    k, n = basis.shape[1], len(outcome)
    x = np.zeros((n, k + 28 + len(names)))
    x[:, :k] = basis
    x[np.arange(n), k + cells] = 1.0
    for i, h in enumerate(players):
        if h:
            x[i, k + 28 + col[h]] = 1.0
    prec = np.concatenate([np.full(k, CURVE_RIDGE), np.full(28, cell_prior),
                           np.full(len(names), player_prior)])
    beta = np.zeros(x.shape[1])
    yv = np.asarray(outcome, dtype=float)
    for _ in range(iterations):
        pr = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        grad = x.T @ (yv - pr) - prec * beta
        hess = (x * (pr * (1 - pr))[:, None]).T @ x + np.diag(prec)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-6:
            break
    shifts = beta[k:k + 28].reshape(4, 7)
    return beta[:k], shifts, {h: float(beta[k + 28 + col[h]]) for h in names}


def fit_fourth_downs(records, dp, cell_prior=DECISION_PRIOR, player_prior=PLAYER_PRIOR):
    """One fit of the 4th-down choices: the league's go curve, a shift per part of the game and
    margin, and each player's go shift, together (a player's shift is measured against the curve
    and the shifts, not the curve alone); then the same for a kick being a field goal rather than
    a punt. Returns (go coefficients, go shifts, {player: go shift}, kick coefficients, kick
    shifts, {player: kick shift})."""
    if not records:
        return tuple(dp.go_coef), np.zeros((4, 7)), {}, tuple(dp.fg_kick_coef), np.zeros((4, 7)), {}
    ph = np.array([r[0] for r in records])
    mb = np.array([r[1] for r in records])
    y = np.array([r[2] for r in records])
    t = np.array([r[3] for r in records])
    went = np.array([r[4] == "go" for r in records])
    who = [r[5] for r in records]
    go_coef, go_shift, go_players = _ridge_logistic(_go_basis(y, t), ph * 7 + mb, who, went,
                                                    cell_prior, player_prior)
    kick = np.flatnonzero(~went & (100 - y + 17 <= dp.fg_max_distance))
    if not len(kick):
        return (tuple(go_coef), go_shift, go_players, tuple(dp.fg_kick_coef), np.zeros((4, 7)), {})
    kb = np.stack([np.ones(len(kick)), y[kick] / 100.0], axis=1)
    fg = np.array([records[i][4] == "fg" for i in kick])
    kick_coef, kick_shift, kick_players = _ridge_logistic(kb, ph[kick] * 7 + mb[kick],
                                                          [who[i] for i in kick], fg,
                                                          cell_prior, player_prior)
    return tuple(go_coef), go_shift, go_players, tuple(kick_coef), kick_shift, kick_players


LATE_DEFICITS = (-3, -8, -11, -16)
KICK_RANGES = (45, 55)
LATE_PRIOR = 10.0
CHOICES = ("go", "fg", "punt")
LATE_TIME = 60.0
GO_AHEAD = True
GO_AHEAD_ONE_SCORE = 60.0
GO_AHEAD_ANY = 30.0


def late_time(clock):
    """The time left as a bin for a late 4th down: the last minute, or before it."""
    return 0 if clock <= LATE_TIME else 1


def late_band(margin):
    """A trailing side's deficit as a band: -1..-3, -4..-8, -9..-11, -12..-16, <=-17."""
    for i, edge in enumerate(LATE_DEFICITS):
        if margin >= edge:
            return i
    return len(LATE_DEFICITS)


def kick_range(y):
    """A field goal's distance from here as a range: up to 45, 46-55, longer."""
    d = 100 - y + 17
    return 0 if d <= KICK_RANGES[0] else 1 if d <= KICK_RANGES[1] else 2


def late_fourth_choices(rows):
    """Every 4th down by a trailing side in the last three minutes or overtime: the last minute or
    not, deficit band, kick range and what it chose."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["down"] != "4" or a["play_kind"] not in SNAP_KINDS or not a["field_position"]:
            continue
        if not a["period"] or a["period"] != b["period"] or not a["clock_seconds"]:
            continue
        p, c = _i(a["period"]), _f(a["clock_seconds"])
        margin = _margin(a, a["offense"])
        if margin is None or margin >= 0 or not (p >= 5 or (p == 4 and c <= 180)):
            continue
        choice = "punt" if b["play_kind"] == "PUNT" else "fg" if b["play_kind"] == "FIELD_GOAL" else "go"
        out.append((late_time(c), late_band(margin), kick_range(_i(a["field_position"])),
                    CHOICES.index(choice)))
    return out


def _fit_late_fourths_pooled(records, prior=LATE_PRIOR):
    """v6's table: by deficit band and kick range over the whole of the last three minutes."""
    n = np.zeros((len(LATE_DEFICITS) + 1, len(KICK_RANGES) + 1, len(CHOICES)))
    for _, band, rng_, choice in records:
        n[band, rng_, choice] += 1
    pooled = n.sum(0, keepdims=True)
    pooled = (pooled + 1.0) / (pooled.sum(2, keepdims=True) + len(CHOICES))
    return (n + prior * pooled) / (n.sum(2, keepdims=True) + prior)


def fit_late_fourths(records, prior=LATE_PRIOR):
    """The chance of going for it, kicking and punting by the last minute or not, deficit band and
    kick range, shrunk toward the same minute and kick range's rate over all deficits."""
    n = np.zeros((2, len(LATE_DEFICITS) + 1, len(KICK_RANGES) + 1, len(CHOICES)))
    for tb, band, rng_, choice in records:
        n[tb, band, rng_, choice] += 1
    pooled = n.sum(1, keepdims=True)
    pooled = (pooled + 1.0) / (pooled.sum(3, keepdims=True) + len(CHOICES))
    return (n + prior * pooled) / (n.sum(3, keepdims=True) + prior)


BASE_MODE = {0: 1, 1: 1, 2: 2, 3: 5, 4: 4, 5: 5, 6: 6, 7: 6, 8: 5, 9: 4, 10: 5, 11: 5, 12: 5, 13: 5,
             14: 5, 15: 6}


def _fallbacks(key):
    """Coarser table bins to fall back on, finest first."""
    rest, z = divmod(key, 4)
    rest, db = divmod(rest, 4)
    mode, d = divmod(rest, 4)
    zc = 0 if z < 2 else 1
    base = BASE_MODE[mode]
    return [("k", key), ("zc", mode, d, db, zc), ("mdb", mode, d, db), ("m2", base, d, db, z),
            ("m2zc", base, d, db, zc), ("any", d, db, z), ("anyzc", d, db, zc), ("d", d, db),
            ("dd", min(d, 2))]


def _group_of(rec, level):
    """The bin of a snap at a given level of coarseness."""
    mode, d, db, z = rec["mode"], min(4, rec["down"]) - 1, dist_bucket(rec["distance"]), zone(rec["field"])
    zc = 0 if z < 2 else 1
    base = BASE_MODE[mode]
    return {"k": ("k", rec["key"]), "zc": ("zc", mode, d, db, zc), "mdb": ("mdb", mode, d, db),
            "m2": ("m2", base, d, db, z),
            "m2zc": ("m2zc", base, d, db, zc), "any": ("any", d, db, z), "anyzc": ("anyzc", d, db, zc),
            "d": ("d", d, db), "dd": ("dd", min(d, 2))}[level]


# v13: each side's big-play rate. A side's touchdowns run longer or shorter by its gamer and its
# NFL team (Lions and Eagles about 7 yards longer than Vikings and Ravens, with gamer and opponent
# held level) without it scoring more points, and matches between big-play sides spread their totals
# about 3 points wider in sd, held out. One strength (theta) can't do that: it moves every play
# toward the top of its bin at once, and a big-play rate that took its extra big plays from the
# rest of the bin alike was undone by the strength that held its points (a rate of 1.8 kept only a
# quarter of its big plays). A big-play side is boom or bust instead: each side carries `big`, a
# snap's chance of drawing one of its bin's big plays (BIG_GAIN yards or more) is multiplied by it,
# and the same chance moves onto the bin's failed plays (no first down), out of the plays between
# -- so its points barely move, and what's left is held where the pre-match model has them by the
# strength (v13.big_response). Inside each part the tilt toward the top still applies.
BIG_PLAYS = True
BIG_GAIN = 40                  # sides differ on the long ones: 40+ yard plays run 2.9% (Vikings) to 4.5%
                               # (Lions) of snaps, 20+ yard ones 13-14% for every team
BIG_MAX = 0.9                  # a draw's chance of the bin's big plays never goes above this
BIG_MIDDLE = 0.05              # nor do the plays between keep less than this


def big_warp(u, tilt, top, bottom, big):
    """Uniform draws `u`, moved so the tilted draw lands on the top `top` share of its bin (the
    big plays) `big` times as often, the bottom `bottom` share (the failed plays) as much more
    often, and the plays between as much less (never under BIG_MIDDLE); inside each of the three
    the tilted draw keeps its shape. Where the bin has no big play, or big is 1, u comes back
    unchanged."""
    top, bottom = np.asarray(top, float), np.asarray(bottom, float)
    inv = 1.0 / np.maximum(tilt, 1e-9)
    c = np.where(top > 0, np.power(np.clip(top, 1e-12, 1.0), inv), 0.0)
    d = 1.0 - np.power(np.clip(1.0 - bottom, 0.0, 1.0), inv)
    c2 = np.minimum(BIG_MAX, c * big)
    d2 = np.clip(d + (c2 - c), 0.0, None)
    room = 1.0 - BIG_MIDDLE - c2 - d2
    over = np.minimum(room, 0.0)                    # the middle would run out: take it back evenly
    c2, d2 = c2 + over / 2, d2 + over / 2
    on = (top > 0) & (top + bottom < 1) & (c > 0) & (c2 > 0) & (1 - c - d > 0) & (big != 1.0)
    one = lambda x: np.where(on, x, 1.0)
    low = u * d / one(np.maximum(d2, 1e-12))
    mid = d + (u - d2) * (1 - c - d) / one(1 - c2 - d2)
    high = (1 - c) + (u - (1 - c2)) * c / one(c2)
    out = np.where(u < d2, low, np.where(u < 1 - c2, mid, high))
    return np.where(on, out, u)


# v14: each side's lost-yardage rate and loss size. A snap that loses yards is 8.2% of snaps; how
# often a side does it moves with its gamer (sd 0.25 in log odds), its team (0.09: the Lions most,
# the Chiefs least) and what the other gamer allows (0.16), and each holds from one fortnight to
# the next (+0.68 gamer, +0.88 team). How far it goes back moves with the gamer: their losses average
# 2.8 to 7.4 yards, the league 4.8. Each bin now keeps its plays in order: turnovers, then losses
# (biggest first), then the other failed plays, the rest, and the big plays on top. A side's `loss`
# multiplies its chance of drawing a loss, out of the plays between; its `loss_size` leans the draw
# inside the losses toward the deep end (above 1) or the shallow (below). Its points are held where
# the pre-match model has them by the strength (v14.loss_response), as for the big plays.
LOSS_PLAYS = True


def _is_loss(rec):
    """A snap that lost yards (not a turnover)."""
    return rec["kind"] == GAIN and rec["gain"] < 0


def _is_turnover(rec):
    return rec["kind"] != GAIN


def zone_warp(u, tilt, to, loss, fail, top, big, loss_rate, loss_size):
    """Uniform draws `u`, moved so the tilted draw lands on its bin's five parts -- turnovers (the
    bottom `to` share), losses (`loss`), the other failed plays (to `fail` in all), the rest, and
    the big plays (the top `top`) -- with the big plays `big` times as often and the failed plays as
    much more often (as big_warp), the losses `loss_rate` times as often again, out of the rest
    (never under BIG_MIDDLE), and inside the losses the draw leaned by `loss_size` toward the
    biggest (above 1). Inside each part the tilted draw keeps its shape; with every rate 1 and no
    loss in the bin, u comes back unchanged."""
    inv = 1.0 / np.maximum(tilt, 1e-9)
    F = lambda x: 1.0 - np.power(np.clip(1.0 - x, 0.0, 1.0), inv)
    bounds = [np.zeros_like(u), to, to + loss, fail, 1.0 - top, np.ones_like(u)]
    old = [F(np.asarray(b, float)) for b in bounds]
    q = [old[i + 1] - old[i] for i in range(5)]
    c2 = np.where(top > 0, np.minimum(BIG_MAX, q[4] * big), q[4])
    d = q[0] + q[1] + q[2]
    grow = np.where(d > 0, np.maximum(0.0, d + (c2 - q[4])) / np.where(d > 0, d, 1.0), 1.0)
    new = [q[0] * grow, q[1] * grow * loss_rate, q[2] * grow, None, c2]
    new[3] = 1.0 - new[0] - new[1] - new[2] - new[4]
    short = np.minimum(new[3] - BIG_MIDDLE, 0.0)          # the middle would run out: scale the rest
    keep = np.where(short < 0, (1.0 - BIG_MIDDLE) / np.maximum(1e-12, 1.0 - new[3]), 1.0)
    for i in (0, 1, 2, 4):
        new[i] = new[i] * keep
    new[3] = 1.0 - new[0] - new[1] - new[2] - new[4]
    cum = [np.zeros_like(u)]
    for i in range(5):
        cum.append(cum[-1] + new[i])
    out = u.copy()
    for i in range(5):
        m = (u >= cum[i]) & (u < cum[i + 1]) if i < 4 else (u >= cum[4])
        if not m.any():
            continue
        v = (u[m] - cum[i][m]) / np.maximum(new[i][m], 1e-12)
        if i == 1:
            v = np.power(np.clip(v, 0.0, 1.0), loss_size[m])
        out[m] = old[i][m] + np.clip(v, 0.0, 1.0) * q[i][m]
    return out


class Tables:
    """Everything the simulation draws from, built from real plays."""

    def __init__(self):
        """Empty tables."""
        self.start = None
        self.count = None
        self.success = None
        self.kind = self.gain = self.new_field = self.seconds = self.replay = None
        self.kick = {}
        self.punt = {}
        self.safety_kick = None
        self.backed = None
        self.backed_return = None
        self.n_stop = None
        self.n_fresh = self.n_fresh_stop = None
        self.n_big_stop = self.n_big_run = None
        self.n_fail_stop = self.n_fail_run = None
        self.n_to_stop = self.n_to_run = self.n_loss_stop = self.n_loss_run = None
        self.kick_pools, self.kick_mix = {}, None
        self.runoff = np.array(DEFAULT_RUNOFF)
        self.q4_modes = False
        self.stop_success = self.run_success = None
        self.stop_shift = np.zeros(N_CELLS)
        self.sec_shift = np.zeros((2, N_CELLS))
        self.eff_shift = np.zeros(N_CELLS)
        self.rz_shift = np.zeros(N_RZ)
        self.rz_hold = np.zeros(N_RZ)
        self.td_hold = np.zeros(N_SETTLE)
        self.call_p = np.zeros(N_CALL)
        self.ot_stop = np.zeros(CLOCK_CELLS)
        self.ot_call_scale = OT_CALL_DEFAULT
        self.to_secs = np.full((2, 2, 3, len(TO_QUANTILES)), np.nan)
        self.kneel_p = np.zeros(N_KNEEL)
        self.kneel_secs = np.array([PLAY_CLOCK - 3.0])
        self.fg_seconds = None
        self.fg_after = None
        self.go_for_two = None
        self.two_good = 0.57
        self.kick_good = 0.98
        self.drive = DriveParams()
        self.period_theta = np.zeros(6)
        self.late_theta = np.zeros(6)
        self.strength_game = 0.0
        self.strength_league = 0.0
        self.strength_theta = np.zeros(0)
        self.strength_slope = np.zeros(0)
        self.inplay_theta = np.zeros((N_SEGMENTS, N_BANDS))
        self.go_shift = np.zeros((4, 7))
        self.late_fg = np.array([0.32, 0.62])
        self.late_fourth = None
        self.big_lead = BIG_LEAD
        self.early_fg = np.zeros((2, len(EARLY_FG_CLOCK), len(EARLY_FG_RANGES)))
        self.close_fg = None
        self.close_fourth = None
        self.kick_clock = None
        self.fg_shift = np.zeros((4, 7))
        self.player_go = {}
        self.player_kick = {}
        self.ot_go = 0.5

    @classmethod
    def build(cls, matches, min_records=MIN_RECORDS, seed=0, handles=None, as_of=None,
              half_life=None):
        """Build every table from the export's matches; `handles` (match -> (home, away)) gives the
        4th-down fit each player's own go and kick shifts. v9: with `as_of` and `half_life` the
        clock fits weight each match by its age (age_weights)."""
        snaps, kicks, decisions, conv, fourths, early, free = [], [], [], [], [], [], []
        backed, late4, fourth_recs, ot_first, ends = [], [], [], [], []
        close, close4, kick_times = [], [], []
        kinds, runoff = [], []
        for code, rows in matches.items():
            runoff += quarter_runoffs(rows)
            fourth_recs += fourth_down_records(rows, (handles or {}).get(code))
            snaps_before = len(snaps)
            ot_first += ot_first_fourths(rows)
            late4 += late_fourth_choices(rows)
            snaps += snap_records(rows)
            for rec in snaps[snaps_before:]:
                rec["match"] = code
                rec["kneel"] = KNEELS and is_kneel(rec)
            new_ends = clock_end_records(rows) if LATE_CLOCK else []
            for rec in new_ends:
                rec["match"] = code
            ends += new_ends
            if not FRESH_CLOCK:
                for rec in snaps[snaps_before:]:
                    rec["fresh"] = False
            backed += backed_up_snaps(rows)
            free += safety_kicks(rows)
            kicks += kick_records(rows)
            kinds += kick_kinds(rows)
            decisions += kick_decisions(rows)
            conv += conversions(rows)
            fourths += fourth_down_choices(rows)
            early += early_kicks(rows)
            close += close_kicks(rows)
            close4 += close_fourths(rows)
            kick_times += close_kick_times(rows)
        age_weights(snaps + ends, as_of, half_life)
        t = cls()
        if TIMEOUTS:
            known = any("timeouts_left" in r for r in snaps)
            t.call_p = fit_timeout_calls(snaps) if known else np.array(DEFAULT_CALL_P)
            t.call_fitted = known
            t.ot_call_scale = fit_ot_call_scale(snaps, t.call_p) if known else OT_CALL_DEFAULT
            if TO_PLAY_SECONDS and known:
                t.to_secs = fit_timeout_seconds(snaps)
            natural_seconds(snaps)
        if KNEELS:
            t.kneel_p, t.kneel_secs = fit_kneels(snaps)
            kneels = [r for r in snaps if r["kneel"]]
            snaps = [r for r in snaps if not r["kneel"]]
        else:
            kneels = []
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
        settle = [r for r in snaps if r["period"] == 4 and r["clock"] <= SETTLE_CLOCK
                  and -2 <= r["margin"] <= 0 and r["down"] <= 3
                  and 100 - r["field"] + 17 <= SETTLE_RANGE] if SETTLE_PLAYS else []
        if len(settle) >= SETTLE_MIN:
            for c, want in enumerate((lambda m: m == 0, lambda m: m < 0)):
                recs = [r for r in settle if want(r["margin"])]
                g = ("settle", c)
                bins[g] = recs if len(recs) >= SETTLE_MIN else settle
                chosen.append(g)
        order = list(bins)
        start, count, succ, nstop, ssucc, rsucc, nfresh, nfstop = {}, {}, {}, {}, {}, {}, {}, {}
        nbig_stop, nbig_run, nfail_stop, nfail_run = {}, {}, {}, {}
        nto_stop, nto_run, nloss_stop, nloss_run = {}, {}, {}, {}
        kind, gain, newf, secs, rep, tdf = [], [], [], [], [], []
        for g in order:
            recs = list(bins[g])
            rng.shuffle(recs)
            if LOSS_PLAYS:                # turnovers, losses (biggest first), the rest, big plays on top
                recs.sort(key=lambda r: (r["seconds"] > STOP_SECONDS, BIG_PLAYS and _is_big(r),
                                         0 if _is_turnover(r) else 1 if _is_loss(r) else 2,
                                         -2000 if r["kind"] == DEF_TD else -1000 if r["kind"] == TURNOVER
                                         else r["gain"] if _is_loss(r) else r["gain"] - r["distance"]))
            else:
                recs.sort(key=lambda r: (r["seconds"] > STOP_SECONDS, BIG_PLAYS and _is_big(r),
                                         -2000 if r["kind"] == DEF_TD else -1000 if r["kind"] == TURNOVER
                                         else r["gain"] - r["distance"]))
            start[g] = len(kind)
            count[g] = len(recs)
            succ[g] = sum(r["success"] for r in recs) / len(recs)
            stops = [r for r in recs if r["seconds"] <= STOP_SECONDS]
            nstop[g] = len(stops)
            nfresh[g] = sum(1 for r in recs if r.get("fresh"))
            nbig_stop[g] = sum(1 for r in stops if _is_big(r)) if BIG_PLAYS else 0
            nbig_run[g] = (sum(1 for r in recs if _is_big(r)) - nbig_stop[g]) if BIG_PLAYS else 0
            nfail_stop[g] = sum(1 for r in stops if _is_fail(r)) if BIG_PLAYS else 0
            nfail_run[g] = (sum(1 for r in recs if _is_fail(r)) - nfail_stop[g]) if BIG_PLAYS else 0
            nto_stop[g] = sum(1 for r in stops if _is_turnover(r)) if LOSS_PLAYS else 0
            nto_run[g] = (sum(1 for r in recs if _is_turnover(r)) - nto_stop[g]) if LOSS_PLAYS else 0
            nloss_stop[g] = sum(1 for r in stops if _is_loss(r)) if LOSS_PLAYS else 0
            nloss_run[g] = (sum(1 for r in recs if _is_loss(r)) - nloss_stop[g]) if LOSS_PLAYS else 0
            nfstop[g] = sum(1 for r in stops if r.get("fresh"))
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
        t.n_fresh = np.array([nfresh[g] for g in chosen], dtype=np.int64)
        t.n_fresh_stop = np.array([nfstop[g] for g in chosen], dtype=np.int64)
        t.n_big_stop = np.array([nbig_stop[g] for g in chosen], dtype=np.int64)
        t.n_big_run = np.array([nbig_run[g] for g in chosen], dtype=np.int64)
        t.n_fail_stop = np.array([nfail_stop[g] for g in chosen], dtype=np.int64)
        t.n_fail_run = np.array([nfail_run[g] for g in chosen], dtype=np.int64)
        if LOSS_PLAYS:
            t.n_to_stop = np.array([nto_stop[g] for g in chosen], dtype=np.int64)
            t.n_to_run = np.array([nto_run[g] for g in chosen], dtype=np.int64)
            t.n_loss_stop = np.array([nloss_stop[g] for g in chosen], dtype=np.int64)
            t.n_loss_run = np.array([nloss_run[g] for g in chosen], dtype=np.int64)
        t.q4_modes = Q4_MODES
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
        if KICK_STYLES and kinds:
            t.kick_pools, t.kick_mix = fit_kick_styles(kinds)
        if runoff:
            t.runoff = np.array(runoff)
        t.safety_kick = (np.array(free, dtype=np.int32) if len(free) >= SAFETY_KICK_MIN
                         else _shifted_kick(t.kick[False][1]))
        punts = [d for d in decisions if d[0] == "punt"]
        for b in range(3):
            ps = [p for p in punts if _punt_bucket(p[1]) == b]
            if len(ps) < MIN_PUNTS:
                ps = punts
            t.punt[b] = (np.array([p[2] for p in ps], dtype=np.int32), np.array([p[3] for p in ps]))
        fgs = [d for d in decisions if d[0] == "fg"]
        t.fg_seconds = np.array([d[3] for d in fgs]) if fgs else np.array([5.0])
        misses = [(d[1], d[4]) for d in fgs if d[4] is not None]
        t.fg_after = (float(np.mean([100 - a - y for y, a in misses])) if misses else 7.0)
        t.go_for_two = fit_go_for_two(conv)
        two = [good for _, _, went, good in conv if went]
        kick = [good for _, _, went, good in conv if not went]
        t.two_good = float(np.mean(two)) if two else 0.57
        t.kick_good = float(np.mean(kick)) if kick else 0.98
        t.ot_go = (sum(ot_first) + OT_GO_PRIOR * 0.5) / (len(ot_first) + OT_GO_PRIOR)
        if FOURTH_JOINT:
            go_coef, t.go_shift, t.player_go, kick_coef, t.fg_shift, t.player_kick = \
                fit_fourth_downs(fourth_recs, t.drive)
            t.drive = replace(t.drive, go_coef=go_coef, fg_kick_coef=kick_coef)
        else:
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
        if close:
            t.close_fg = fit_close_fg(close)
        if close4:
            t.close_fourth = fit_close_fourths(close4)
        if KICK_CLOCK:
            t.kick_clock = fit_kick_clock(kick_times)
        t.n_snaps = len(snaps)
        if late4:
            t.late_fourth = fit_late_fourths(late4) if GO_AHEAD else _fit_late_fourths_pooled(late4)
        if backed:
            t.backed, returns = fit_backed_up(backed)
            t.backed_return = returns if len(returns) >= BACKED_RETURNS_MIN else None
        fit_play_calling(t, snaps, ends, kneels)
        if SETTLE_FIT:
            t.rz_shift, t.rz_hold = np.zeros(N_RZ), np.zeros(N_RZ)
        else:
            fit_red_zone(t, snaps)
        return t

    def success_of(self, key):
        """The league's first-down success rate in a bin."""
        return float(self.success[key])

    def save(self, path):
        """Write the tables to a file."""
        arrays = dict(start=self.start, count=self.count, success=self.success, kind=self.kind,
                      gain=self.gain, new_field=self.new_field, seconds=self.seconds,
                      replay=self.replay, td_from=self.td_from, fg_seconds=self.fg_seconds,
                      fg_after=np.array([self.fg_after]), period_theta=self.period_theta,
                      late_theta=self.late_theta,
                      strength=np.array([self.strength_game, self.strength_league]),
                      strength_theta=self.strength_theta, strength_slope=self.strength_slope,
                      go_for_two=self.go_for_two, go_shift=self.go_shift, fg_shift=self.fg_shift,
                      late_fg=self.late_fg, early_fg=self.early_fg, big_lead=np.array([-1 if self.big_lead is None else self.big_lead]), conv_rates=np.array([self.two_good, self.kick_good]),
                      safety_kick=self.safety_kick, n_stop=self.n_stop, n_fresh=self.n_fresh,
                      n_fresh_stop=self.n_fresh_stop, q4_modes=np.array([int(self.q4_modes)]),
                      stop_success=self.stop_success, run_success=self.run_success,
                      stop_shift=self.stop_shift, sec_shift=self.sec_shift, eff_shift=self.eff_shift,
                      rz_shift=self.rz_shift, rz_hold=self.rz_hold, td_hold=self.td_hold,
                      inplay_theta=self.inplay_theta, go_coef=np.array(self.drive.go_coef),
                      fg_kick_coef=np.array(self.drive.fg_kick_coef), ot_go=np.array([self.ot_go]),
                      call_p=self.call_p, ot_stop=self.ot_stop,
                      ot_call_scale=np.array([self.ot_call_scale]), to_secs=self.to_secs,
                      kneel_p=self.kneel_p, kneel_secs=self.kneel_secs)
        arrays["runoff"] = self.runoff
        if self.kick_mix is not None:
            arrays["kick_mix"] = self.kick_mix
            for late, (a, b, c) in self.kick_pools.items():
                arrays[f"kick_pool{int(late)}_onside"], arrays[f"kick_pool{int(late)}_field"] = a, b
                arrays[f"kick_pool{int(late)}_sec"] = c
        if self.n_big_stop is not None:
            arrays["n_big_stop"], arrays["n_big_run"] = self.n_big_stop, self.n_big_run
            arrays["n_fail_stop"], arrays["n_fail_run"] = self.n_fail_stop, self.n_fail_run
        if self.n_loss_stop is not None:
            arrays["n_to_stop"], arrays["n_to_run"] = self.n_to_stop, self.n_to_run
            arrays["n_loss_stop"], arrays["n_loss_run"] = self.n_loss_stop, self.n_loss_run
        if self.late_fourth is not None:
            arrays["late_fourth"] = self.late_fourth
        if self.close_fg is not None:
            arrays["close_fg"] = self.close_fg
        if self.close_fourth is not None:
            arrays["close_fourth"] = self.close_fourth
        if self.kick_clock is not None:
            arrays["kick_clock_zero"] = np.array([self.kick_clock[0]])
            arrays["kick_clock_used"] = self.kick_clock[1]
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
        """Read tables from a file."""
        z = np.load(path)
        t = cls()
        for name in ("start", "count", "success", "kind", "gain", "new_field", "seconds",
                     "replay", "td_from", "fg_seconds", "go_for_two"):
            setattr(t, name, z[name])
        if len(t.go_for_two) < CONV_CELLS:                    # a build from before the clock slices
            t.go_for_two = np.vstack([t.go_for_two] + [t.go_for_two[3:4]] * (CONV_CELLS - len(t.go_for_two)))
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
        if "ot_go" in z:
            t.ot_go = float(z["ot_go"][0])
        if "go_coef" in z:
            t.drive = replace(t.drive, go_coef=tuple(float(x) for x in z["go_coef"]),
                              fg_kick_coef=tuple(float(x) for x in z["fg_kick_coef"]))
        t.big_lead = int(z["big_lead"][0]) if "big_lead" in z else None
        if t.big_lead is not None and t.big_lead < 0:
            t.big_lead = None
        if "late_fourth" in z:
            t.late_fourth = z["late_fourth"]
        if "early_fg" in z:
            t.early_fg = z["early_fg"]
        if "close_fg" in z:
            t.close_fg = z["close_fg"]
        if "close_fourth" in z:
            t.close_fourth = z["close_fourth"]
        if "kick_clock_zero" in z:
            t.kick_clock = (float(z["kick_clock_zero"][0]), z["kick_clock_used"])
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
        if "rz_shift" in z:
            t.rz_shift, t.rz_hold = z["rz_shift"], z["rz_hold"]
        if "td_hold" in z and z["td_hold"].shape == (N_SETTLE,):
            t.td_hold = z["td_hold"]
        if "n_stop" in z:
            t.n_stop, t.stop_success, t.run_success = z["n_stop"], z["stop_success"], z["run_success"]
            t.stop_shift, t.sec_shift, t.eff_shift = z["stop_shift"], z["sec_shift"], z["eff_shift"]
        else:
            t.n_stop = t.count.copy()
            t.stop_success = t.run_success = t.success
        if "n_fresh" in z:
            t.n_fresh, t.n_fresh_stop = z["n_fresh"], z["n_fresh_stop"]
        else:
            t.n_fresh, t.n_fresh_stop = np.zeros_like(t.count), np.zeros_like(t.count)
        t.q4_modes = bool(z["q4_modes"][0]) if "q4_modes" in z else False
        if "call_p" in z and z["call_p"].shape == N_CALL:
            t.call_p = z["call_p"]
        if "ot_stop" in z:
            t.ot_stop = z["ot_stop"]
        if "ot_call_scale" in z:
            t.ot_call_scale = float(z["ot_call_scale"][0])
        if "to_secs" in z and z["to_secs"].ndim == 4:
            t.to_secs = z["to_secs"]
        if "kneel_p" in z:
            t.kneel_p, t.kneel_secs = z["kneel_p"], z["kneel_secs"]
        if "runoff" in z:
            t.runoff = z["runoff"]
        if "kick_mix" in z:
            t.kick_mix = z["kick_mix"]
            t.kick_pools = {bool(late): (z[f"kick_pool{late}_onside"], z[f"kick_pool{late}_field"],
                                         z[f"kick_pool{late}_sec"]) for late in (0, 1)}
        if "n_big_stop" in z:
            t.n_big_stop, t.n_big_run = z["n_big_stop"], z["n_big_run"]
            t.n_fail_stop, t.n_fail_run = z["n_fail_stop"], z["n_fail_run"]
        if "n_loss_stop" in z:
            t.n_to_stop, t.n_to_run = z["n_to_stop"], z["n_to_run"]
            t.n_loss_stop, t.n_loss_run = z["n_loss_stop"], z["n_loss_run"]
        return t


def _shifted_kick(fields):
    """A safety's free kick lands further up the field."""
    return np.clip(np.asarray(fields) + SAFETY_KICK_SHIFT, 1, 99).astype(np.int32)


def _punt_bucket(y):
    """Field position as a punt bucket."""
    return 0 if y < 35 else 1 if y < 55 else 2


class Start:
    """Starting states for the simulation, one per snapshot."""

    def __init__(self, n):
        """n kickoff states with league-average strengths."""
        self.period = np.ones(n, dtype=np.int32)
        self.clock = np.full(n, QUARTER)
        self.phase = np.full(n, KICK, dtype=np.int8)
        self.team = np.zeros(n, dtype=np.int8)
        self.down = np.ones(n, dtype=np.int32)
        self.dist = np.full(n, 10, dtype=np.int32)
        self.y = np.full(n, 25, dtype=np.int32)
        self.home = np.zeros(n, dtype=np.int32)
        self.away = np.zeros(n, dtype=np.int32)
        self.kicks_second_half = np.full(n, -1, dtype=np.int8)
        self.theta = np.zeros((n, 2))
        self.aggression = np.zeros((n, 2))
        self.kick = np.zeros((n, 2))
        self.pace = np.ones((n, 2))
        self.strength = np.zeros((n, 2))
        self.strength_game = np.zeros((n, 2))
        self.fresh = np.zeros(n, dtype=bool)
        self.timeouts = np.full((n, 2), -1, dtype=np.int8)     # -1: not known, drawn when needed
        self.big = np.ones((n, 2))                              # v13: each side's big-play rate
        self.kick_tb = np.ones((n, 2))                          # v14: each kicker's touchback and
        self.kick_nlz = np.ones((n, 2))                         # no-landing-zone rates, x the league's
        self.loss = np.ones((n, 2))                             # v14: each side's lost-yardage rate
        self.loss_size = np.ones((n, 2))                        # and how deep its losses go


def _sigmoid(z):
    """The logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0xBF58476D1CE4E5B9)
_K3 = np.uint64(0x94D049BB133111EB)


def _uniform(seed, path, step, slot):
    """A repeatable random number for (seed, path, step, slot)."""
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
    """Each side's strength spread for the day, in theta, and the start strength that keeps
    expected points."""
    theta = np.asarray(theta, dtype=float)
    if not len(tables.strength_slope):
        return theta, np.zeros(2), np.zeros(2)
    slope = np.maximum(0.3, np.interp(theta, tables.strength_theta, tables.strength_slope))
    own = np.sqrt(np.maximum(0.0, np.asarray(form, dtype=float))) / slope
    game = np.sqrt(max(0.0, tables.strength_game)) / slope
    return theta - slope * (own * own + game * game) / 2.0, own, game


def _styled_kicks(tables, sub, kicker, period, clock, kick_tb, kick_nlz, rand, pick):
    """(onside, start, seconds) for kick-offs off the kicking side's own touchback and no-landing-zone
    rates (KICK_STYLES): the league's chance for the part of the game, times the kicker's."""
    late = ((period == 2) | (period == 4)) & (clock <= KICK_LATE)
    li = late.astype(int)
    p_tb = np.minimum(KICK_STYLE_MAX, tables.kick_mix[li, 0] * kick_tb[sub, kicker])
    p_nlz = np.minimum(KICK_STYLE_MAX - p_tb, tables.kick_mix[li, 1] * kick_nlz[sub, kicker])
    u = rand(sub, 50)
    tb = u < p_tb
    nlz = ~tb & (u < p_tb + p_nlz)
    on = np.zeros(len(sub), dtype=bool)
    field = np.where(tb, KICK_TOUCHBACK, KICK_NO_LANDING).astype(np.int32)
    secs = np.zeros(len(sub))
    for lv in (False, True):
        m = (late == lv) & ~tb & ~nlz
        if m.any():
            p_on, p_field, p_secs = tables.kick_pools[lv]
            j = pick(sub[m], 4, len(p_field))
            on[m], field[m], secs[m] = p_on[j], p_field[j], p_secs[j]
    return on, field, secs


def simulate(tables, start, n_paths, rng=None, theta_sd=None, kneel_seconds=20.0,
             desperate_seconds=180.0, max_steps=400, stats=None, common=True, seed=None,
             in_play=False, distinct=False, one_drive=False):
    """Play every starting state n_paths times to the end and return the final scores; with
    one_drive, only to the end of the drive under way (the ball changes hands, a kick-off after a
    score, or the half ends)."""
    rng = rng or np.random.default_rng()
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
    kick_pl = rep(getattr(start, "kick", np.zeros((S, 2))))
    pace = rep(start.pace)
    big = rep(getattr(start, "big", np.ones((S, 2))))
    big_on = BIG_PLAYS and tables.n_big_stop is not None and bool(np.any(big != 1.0))
    kick_tb = rep(getattr(start, "kick_tb", np.ones((S, 2))))
    kick_nlz = rep(getattr(start, "kick_nlz", np.ones((S, 2))))
    styles_on = KICK_STYLES and getattr(tables, "kick_mix", None) is not None
    loss_r = rep(getattr(start, "loss", np.ones((S, 2))))
    loss_s = rep(getattr(start, "loss_size", np.ones((S, 2))))
    loss_on = LOSS_PLAYS and getattr(tables, "n_loss_stop", None) is not None \
        and bool(np.any(loss_r != 1.0) or np.any(loss_s != 1.0))
    fresh = rep(getattr(start, "fresh", np.zeros(S, dtype=bool))).astype(bool)
    tos = rep(getattr(start, "timeouts", np.full((S, 2), -1))).astype(np.int8)
    to_on = TIMEOUTS and tables.call_p.any()
    dp = tables.drive
    gc = dp.go_coef
    ka, kb = dp.fg_kick_coef
    ma, mb, mc = dp.make_coef

    tally = (lambda name, k: stats.__setitem__(name, stats.get(name, 0) + int(k))) \
        if stats is not None else (lambda name, k: None)
    path_no = (np.arange(P, dtype=np.int64) if distinct
               else np.tile(np.arange(n_paths, dtype=np.int64), S))
    free = np.zeros(P, dtype=bool)
    ot_cur = np.full(P, -1, dtype=np.int8)
    ot_done = np.zeros((P, 2), dtype=bool)
    ahead = (period >= 5) & (score[:, 0] != score[:, 1])
    ot_done[ahead, np.where(score[ahead, 0] > score[ahead, 1], 0, 1)] = True
    step = np.zeros(P, dtype=np.int64)
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
        """Random numbers for these paths."""
        if not common:
            return rng.random(len(ix))
        return _uniform(seed, path_no[ix], step[ix], slot)

    def pick(ix, slot, n):
        """A random index below n for these paths."""
        return np.minimum(n - 1, (rand(ix, slot) * n).astype(np.int64))

    def timeouts_of(ix, side):
        """The timeouts `side` has left in the half (or overtime period) on these paths; all of
        them where not known (real sides still have all three at the two-minute mark nine times
        in ten)."""
        unknown = tos[ix, side] < 0
        tos[ix[unknown], side[unknown]] = np.where(period[ix[unknown]] >= 5, OT_TIMEOUTS, 3)
        return tos[ix, side]

    def calls(ix, lead):
        """Which of these paths' sides stop the clock with a timeout before the snap, off the
        running clock: (called, side). lead: the offense's."""
        ok = np.isin(period[ix], TIMEOUT_QUARTERS) | (period[ix] >= 5)
        scale = np.where(period[ix] >= 5, tables.ot_call_scale, 1.0)
        off, dfn = team[ix].astype(np.int64), (1 - team[ix]).astype(np.int64)
        p_off = np.where(ok & (timeouts_of(ix, off) > 0),
                         scale * tables.call_p[call_index(period[ix], clock[ix], lead, OFFENCE)], 0.0)
        p_def = np.where(ok & (timeouts_of(ix, dfn) > 0),
                         scale * tables.call_p[call_index(period[ix], clock[ix], -lead, DEFENCE)], 0.0)
        u = rand(ix, 41)
        by_def = u < p_def
        by_off = ~by_def & (u < p_def + p_off)
        called = by_def | by_off
        return called, np.where(by_def, dfn, off)

    def fg_make(yy):
        """The chance a field goal from here is good."""
        d = (100 - yy + 17) / 100.0
        p = _sigmoid(ma + mb * d + mc * d * d)
        return np.where(100 - yy + 17 > dp.fg_max_distance, 0.0, p)

    def end_period(ix):
        """Move these paths to the next quarter, half or overtime, or end the game."""
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
        carry = (p == 1) | (p == 3)
        c = ix[carry]
        period[c] += 1
        if QUARTER_CARRY:
            clock[c] = QUARTER - tables.runoff[pick(c, 55, len(tables.runoff))]
            fresh[c] = False
        else:
            clock[c] = QUARTER
            fresh[c] = True
        h = ix[p == 2]
        period[h] = 3
        tos[h] = -1
        clock[h] = QUARTER
        phase[h] = KICK
        team[h] = np.where(kick2[h] >= 0, kick2[h], pick(h, 18, 2))
        e = ix[p >= 4]
        level = score[e, 0] == score[e, 1]
        if OT_CARRY:
            # regulation's end: level goes to overtime off a kick-off, otherwise it is over
            reg = e[period[e] == 4]
            lv = score[reg, 0] == score[reg, 1]
            tally("overtime", lv.sum())
            phase[reg[~lv]] = DONE
            go = reg[lv]
            period[go] = 5
            clock[go] = QUARTER
            tos[go] = OT_TIMEOUTS
            phase[go] = KICK
            team[go] = pick(go, 1, 2)
            # an overtime period's end: over only when a side leads and the side behind has had
            # its possession; otherwise play carries on into the next period, as between quarters.
            # v11: the paths that ended overtime's periods, not those just sent into overtime above
            ot_e = ix[p >= 5] if OT_FIRST_PERIOD else e[period[e] >= 5]
            behind = np.where(score[ot_e, 0] < score[ot_e, 1], 0, 1)
            decided = (score[ot_e, 0] != score[ot_e, 1]) & ot_done[ot_e, behind]
            stop = decided | (period[ot_e] >= 4 + MAX_OT)
            phase[ot_e[stop]] = DONE
            on = ot_e[~stop]
            period[on] += 1
            clock[on] = QUARTER
            tos[on] = OT_TIMEOUTS
            fresh[on] = True
            tally("ot_carried", len(on))
            return
        more = e[level & (period[e] < 4 + MAX_OT)]
        tally("overtime", (level & (period[e] == 4)).sum())
        phase[e[~level | (period[e] >= 4 + MAX_OT)]] = DONE
        period[more] += 1
        clock[more] = QUARTER
        tos[more] = OT_TIMEOUTS
        phase[more] = KICK
        team[more] = pick(more, 1, 2)

    def turnover(ix, field):
        """Give the ball to the other side at this field position."""
        team[ix] = 1 - team[ix]
        y[ix] = np.clip(field, 1, 99)
        down[ix] = 1
        dist[ix] = np.minimum(10, 100 - y[ix])
        fresh[ix] = True

    def score_td(ix, side):
        """Six points to this side, then the conversion."""
        score[ix, side] += 6
        team[ix] = side
        phase[ix] = CONV

    drive_team = team.copy()
    for _ in range(max_steps):
        if one_drive:
            phase[(phase != DONE) & ((phase == KICK) | (team != drive_team))] = DONE
        live = np.flatnonzero(phase != DONE)
        if not len(live):
            break
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

        ix = live[ph == CONV]
        if len(ix):
            s_ = team[ix]
            walk_off = OT_RULES & (period[ix] >= 5) & ot_done[ix, 1 - s_] \
                & (score[ix, s_] > score[ix, 1 - s_])
            if walk_off.any():
                phase[ix[walk_off]] = DONE
                tally("ot_walk_off", walk_off.sum())
                ix, s_ = ix[~walk_off], s_[~walk_off]
        if len(ix):
            margin = np.clip(score[ix, s_] - score[ix, 1 - s_], -CONV_MARGIN, CONV_MARGIN)
            cp = np.where(period[ix] <= 2, 0, np.where(period[ix] == 3, 1,
                          np.where((period[ix] >= 5) | (clock[ix] <= 180), 3, 2)))
            if CONV_CLOCK:
                cp = np.where(period[ix] >= 5, CONV_CELLS - 1, cp)
                for i, edge in reversed(list(enumerate(CONV_CLOCK))):
                    cp = np.where((period[ix] == 4) & (clock[ix] <= edge), 4 + i, cp)
            two = rand(ix, 2) < tables.go_for_two[cp, margin + CONV_MARGIN]
            two |= (period[ix] >= 5) & ot_done[ix, 1 - s_] & ((margin == -1) | (margin == -2))
            good = rand(ix, 3) < np.where(two, tables.two_good, tables.kick_good)
            score[ix, s_] += np.where(good, np.where(two, 2, 1), 0)
            tally("two_point_tries", two.sum())
            phase[ix] = KICK

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
                    if styles_on and not flag:
                        k_on, k_field, k_secs = _styled_kicks(tables, sub, team[sub], period[sub],
                                                              clock[sub], kick_tb, kick_nlz, rand, pick)
                    else:
                        onside, field, secs = tables.kick[flag]
                        j = pick(sub, 4, len(field))
                        k_on, k_field, k_secs = onside[j], field[j], secs[j]
                    rec = team[sub]
                    team[sub] = np.where(k_on, rec, 1 - rec)
                    y[sub] = np.minimum(99, k_field)
                    down[sub] = 1
                    dist[sub] = 10
                    clock[sub] -= k_secs
                    phase[sub] = SCRIM
                    fresh[sub] = True
                    tally("kick_touchback", ((k_field == KICK_TOUCHBACK) & (k_secs == 0) & ~k_on).sum())
                    tally("kick_no_landing", ((k_field == KICK_NO_LANDING) & (k_secs == 0) & ~k_on).sum())
                    tally("kicks", len(sub))
                    late_k = ((period[sub] == 2) | (period[sub] == 4)) & (clock[sub] + k_secs <= KICK_LATE)
                    tally("kicks_late", late_k.sum())
                    tally("kick_touchback_late", (late_k & (k_field == KICK_TOUCHBACK) & (k_secs == 0) & ~k_on).sum())
                    scored = k_field >= 100
                    tally("kick_td", scored.sum())
                    if scored.any():
                        score_td(sub[scored], team[sub[scored]])
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
                    fresh[sub] = True

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

        if to_on:
            # the fourth quarter: the leader kneels it out when its downs left, less the trailing
            # team's timeouts, each worth a play clock, outlast the clock (overtime as before)
            kneel = (p >= 5) & (margin > 0) & (c <= kneel_seconds * (5 - dn)) & (yy > 5 - dn)
            cand = (p == 4) & (margin > 0) & (yy > 5 - dn) & (c <= PLAY_CLOCK * (5 - dn))
            if cand.any():
                ci = np.flatnonzero(cand)
                left = timeouts_of(ix[ci], 1 - o[ci]).astype(np.int64)
                kneel[ci[c[ci] <= PLAY_CLOCK * np.maximum(0, 5 - dn[ci] - left)]] = True
        else:
            kneel = (p >= 4) & (margin > 0) & (c <= kneel_seconds * (5 - dn)) & (yy > 5 - dn)
        tally("kneel", kneel.sum())
        if kneel.any():
            kx = ix[kneel]
            clock[kx] = 0.0
            end_period(kx)

        make = fg_make(yy)
        close = np.zeros(len(ix), dtype=bool)
        if CLOSE_FG and tables.close_fg is not None:
            close = ~kneel & (p == 4) & (margin <= 0) & (margin >= -3) & (dn < 4)
        drain = ~kneel & ~close & (p >= 4) & (margin <= 0) & (margin >= -2) & (dn < 4) & \
            (100 - yy + 17 <= 55) & (c <= kneel_seconds * (4 - dn))
        if drain.any():
            clock[ix[drain]] = 0.0
            c = clock[ix]
        tally("drain_fg", drain.sum())
        sit = np.where(p == 2, 0, np.where((p >= 4) & (margin <= 0) & (margin >= -3), 1, -1))
        cb = np.searchsorted(np.array(EARLY_FG_CLOCK, dtype=float), c)
        kd_ = 100 - yy + 17
        rb = np.minimum(np.searchsorted(np.array(EARLY_FG_RANGES, dtype=float), kd_), len(EARLY_FG_RANGES) - 1)
        early_p = np.where((sit >= 0) & (cb < len(EARLY_FG_CLOCK)) & (kd_ <= EARLY_FG_RANGE),
                           tables.early_fg[np.maximum(sit, 0), np.minimum(cb, len(EARLY_FG_CLOCK) - 1), rb], 0.0)
        if close.any():
            ccb = np.searchsorted(np.array(CLOSE_FG_CLOCK, dtype=float), c)
            crb = np.searchsorted(np.array(CLOSE_FG_RANGES, dtype=float), kd_)
            inside = (ccb < len(CLOSE_FG_CLOCK)) & (crb < len(CLOSE_FG_RANGES))
            cp = tables.close_fg[close_class(margin), np.clip(dn - 1, 0, 2),
                                 np.minimum(ccb, len(CLOSE_FG_CLOCK) - 1),
                                 np.minimum(crb, len(CLOSE_FG_RANGES) - 1)]
            early_p = np.where(close, np.where(inside, cp, 0.0), early_p)
        fg_now = ~kneel & (drain | ((dn < 4) & (rand(ix, 5) < early_p)))
        if close.any():
            # v11: a close side that has chosen to kick, with downs enough to run the clock out,
            # runs it out first -- real go-ahead kicks from a level score in the last minute left
            # nothing on the clock three times in four (median 0 seconds)
            burn = close & fg_now & (c <= kneel_seconds * (4 - dn))
            if burn.any():
                clock[ix[burn]] = 0.0
                c = clock[ix]
            tally("close_burn", burn.sum())
        fourth = ~kneel & ~fg_now & (dn >= 4)
        u = yy / 100.0
        lt = np.log(np.maximum(1, tt))
        red = (yy >= 80).astype(float)
        z = gc[0] + gc[1] * lt + gc[2] * u + gc[3] * u * u + gc[4] * lt * u + gc[5] * red + gc[6] * lt * red
        dph = np.where(p <= 2, 0, np.where(p == 3, 1, np.where((p >= 5) | (c <= 180), 3, 2)))
        mbk = _margin_bucket_np(margin)
        pgo = _sigmoid(z + agg[ix, o] + tables.go_shift[dph, mbk])
        late_trail = (p >= 4) & (hl <= desperate_seconds) & (margin < 0)
        late_trail |= (p >= 5) & (margin < 0) & ot_done[ix, 1 - o]
        r1, r2 = rand(ix, 6), rand(ix, 7)
        kick_to_tie = (margin >= -3) & (make >= 0.3) & \
            (r1 < np.where(c <= 30, tables.late_fg[1], tables.late_fg[0]))
        go = np.where(late_trail, ~kick_to_tie, r1 < pgo)
        kick_fg = np.where(late_trail, True,
                           r2 < _sigmoid(ka + kb * u + tables.fg_shift[dph, mbk] + kick_pl[ix, o]))
        if tables.late_fourth is not None:
            band = np.select([margin >= e for e in LATE_DEFICITS], np.arange(len(LATE_DEFICITS)),
                             len(LATE_DEFICITS))
            kr = np.where(kd_ <= KICK_RANGES[0], 0, np.where(kd_ <= KICK_RANGES[1], 1, 2))
            if tables.late_fourth.ndim == 4:
                lp = tables.late_fourth[np.where(c <= LATE_TIME, 0, 1), band, kr]
            else:
                lp = tables.late_fourth[band, kr]
            table = late_trail & (margin < LATE_DEFICITS[0])
            go = np.where(table, r1 < lp[:, 0], go)
            kick_fg = np.where(table, r1 < lp[:, 0] + lp[:, 1], kick_fg)
        if GO_AHEAD:
            pointless = (p == 4) & (margin < LATE_DEFICITS[0]) & (
                (c <= GO_AHEAD_ANY) | ((margin >= LATE_DEFICITS[1]) & (c <= GO_AHEAD_ONE_SCORE)))
            go = np.where(pointless, True, go)
            kick_fg = np.where(pointless, False, kick_fg)
            tally("go_ahead", (fourth & pointless).sum())
        if OT_RULES:
            ot_first = (p >= 5) & (margin >= 0) & ~ot_done[ix, 1 - o] & (kd_ <= OT_KICK_RANGE)
            go = np.where(ot_first, r1 < tables.ot_go, go)
            kick_fg = np.where(ot_first, True, kick_fg)
            ot_must = (p >= 5) & (margin < 0) & ot_done[ix, 1 - o]
            go = np.where(ot_must & ((margin < -3) | (make <= 0)), True, go)
            kick_fg = np.where(ot_must & (margin < -3), False, kick_fg)
        if CLOSE_FG and tables.close_fourth is not None and tables.close_fourth.shape[0] == len(CLOSE_CLASSES):
            # v11: a close fourth quarter in range kicks at the fitted rate, else goes for it
            cf = fourth & (p == 4) & (margin <= 0) & (margin >= -3) & (c <= 180) & (kd_ <= 60) & (make >= 0.3)
            if cf.any():
                col = (c <= 30).astype(np.int64) if tables.close_fourth.shape[1] == 2 \
                    else np.searchsorted(np.array(FOURTH_CLOCK), c).astype(np.int64)
                kp = tables.close_fourth[close_class(margin), np.minimum(col, tables.close_fourth.shape[1] - 1)]
                kick_fg = np.where(cf, r1 < kp, kick_fg)
                go = np.where(cf, ~kick_fg, go)
        kick_fg = ~go & kick_fg & (make > 0)
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

        fx = ix[do_fg]
        if len(fx):
            j = pick(fx, 8, len(tables.fg_seconds))
            secs = tables.fg_seconds[j].astype(float)
            if KICK_CLOCK and tables.kick_clock is not None:
                # v12: a close side's late kick takes the clock as the real ones did
                fo = team[fx]
                fm = score[fx, fo] - score[fx, 1 - fo]
                fc = clock[fx]
                ck = (period[fx] == 4) & (fm <= 0) & (fm >= -3) & (fc <= 120)
                zero, used = tables.kick_clock
                near = ck & (fc <= KICK_CLOCK_EDGE)
                far = ck & ~near
                secs = np.where(near & (rand(fx, 33) < zero), fc, secs)
                secs = np.where(far, used[pick(fx, 34, len(used))], secs)
                tally("kick_clock_zero", (near & (secs >= fc)).sum())
            clock[fx] -= secs
            good = rand(fx, 9) < make[do_fg]
            tally("fg_good", good.sum())
            g = fx[good]
            score[g, team[g]] += 3
            phase[g] = KICK
            m = fx[~good]
            turnover(m, 100 - y[m] - int(round(tables.fg_after)))

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

        sx = ix[play]
        if not len(sx):
            continue
        if KNEELS and tables.kneel_p.any():
            ld = score[sx, team[sx]] - score[sx, 1 - team[sx]]
            cand = np.flatnonzero((period[sx] == 4) & (ld > 0) & (down[sx] < 4) & (y[sx] > 5))
            if len(cand):
                kp = tables.kneel_p[kneel_index(ld[cand], down[sx[cand]], clock[sx[cand]])]
                kn = cand[rand(sx[cand], 43) < kp]
                if len(kn):
                    kx = sx[kn]
                    used = tables.kneel_secs[pick(kx, 44, len(tables.kneel_secs))]
                    if to_on:
                        call, side = calls(kx, ld[kn])
                        used = np.where(call, 3.0, used)
                        tos[kx[call], side[call]] -= 1
                        tally("timeouts", call.sum())
                        if stats is not None:
                            for q in (2, 4):
                                tally(f"timeouts_q{q}", (call & (period[kx] == q)).sum())
                    clock[kx] -= used
                    y[kx] = np.maximum(1, y[kx] - 1)
                    down[kx] += 1
                    dist[kx] += 1
                    fresh[kx] = False
                    tally("kneel_plays", len(kx))
                    sx = np.delete(sx, kn)
            if not len(sx):
                continue
        so = team[sx]
        lead = score[sx, so] - score[sx, 1 - so]
        mode = _modes_np(period[sx], clock[sx], lead, tables.big_lead, tables.q4_modes)
        key = _keys_np(mode, down[sx], dist[sx], y[sx])
        if SETTLE_PLAYS and len(tables.count) >= N_KEYS + 2:
            settle = (period[sx] == 4) & (clock[sx] <= SETTLE_CLOCK) & (lead <= 0) & (lead >= -2) \
                & (down[sx] <= 3) & (100 - y[sx] + 17 <= SETTLE_RANGE)
            key = np.where(settle, N_KEYS + (lead < 0), key)
        n = tables.count[key]
        cell = _cells_np(period[sx], clock[sx], lead)
        ns = tables.n_stop[key]
        fr = fresh[sx] & tables.n_fresh.any()
        run_logit, ns0, n0 = _running_logit(tables, key)
        p_run = np.where(ns0 <= 0, 0.0, np.where(ns0 >= n0, 1.0,
                                                 _sigmoid(run_logit + tables.stop_shift[cell]
                                                          + np.where(period[sx] >= 5,
                                                                     tables.ot_stop[_slice_of(clock[sx])], 0.0))))
        p_fresh = (tables.n_fresh_stop[key] + FRESH_PRIOR * FRESH_STOP) / (tables.n_fresh[key] + FRESH_PRIOR)
        p_stop = np.where(ns == 0, 0.0, np.where(ns == n, 1.0, np.where(fr, p_fresh, p_run)))
        stops = rand(sx, 15) < p_stop
        seg0 = tables.start[key] + np.where(stops, 0, ns)
        seg_n = np.maximum(1, np.where(stops, ns, n - ns))
        uu = rand(sx, 11)
        tilt = exp_theta[sx, so] * period_exp[np.minimum(period[sx], 5)] * np.exp(tables.eff_shift[cell])
        rz = y[sx] >= RED_ZONE
        if rz.any():
            tilt = tilt * np.where(rz, np.exp(tables.rz_shift[rz_index(period[sx], lead, y[sx])]), 1.0)
        late = ((period[sx] == 2) | (period[sx] == 4)) & (clock[sx] <= LATE)
        if late.any():
            tilt = tilt * np.where(late, late_exp[np.minimum(period[sx], 5)], 1.0)
        if inplay_exp is not None:
            tilt = tilt * inplay_exp[_segments_np(period[sx], clock[sx]),
                                     _bands_np(score[sx, 0] - score[sx, 1])]
        if big_on or loss_on:
            share = np.where(stops, tables.n_big_stop[key], tables.n_big_run[key]) / seg_n
            fails = np.where(stops, tables.n_fail_stop[key], tables.n_fail_run[key]) / seg_n
            bs = big[sx, so]
            if loss_on:
                to_sh = np.where(stops, tables.n_to_stop[key], tables.n_to_run[key]) / seg_n
                loss_sh = np.where(stops, tables.n_loss_stop[key], tables.n_loss_run[key]) / seg_n
                lr, ls = loss_r[sx, so], loss_s[sx, so]
                warp = lambda u_: zone_warp(u_, tilt, to_sh, loss_sh, fails, share, bs, lr, ls)
            else:
                warp = lambda u_: big_warp(u_, tilt, share, fails, bs)
            uu = warp(uu)
        uu = 1.0 - (1.0 - uu) ** tilt
        j = seg0 + np.minimum(seg_n - 1, (uu * seg_n).astype(np.int64))
        if (rz.any() and tables.rz_hold.any()) or tables.td_hold.any():
            yy_ = y[sx]
            scores = lambda jj: (tables.kind[jj] == GAIN) & ((tables.gain[jj] >= 100 - yy_)
                                                             | (tables.td_from[jj] >= 0))
            hold = np.where(rz, tables.rz_hold[rz_index(period[sx], lead, yy_)], 0.0)
            if tables.td_hold.any():
                hold = 1.0 - (1.0 - hold) * (1.0 - tables.td_hold[settle_index(period[sx], clock[sx] + 0.0, lead, yy_)])
            held_td = scores(j) & (rand(sx, 19) < hold)
            for attempt in range(HOLD_TRIES):
                if not held_td.any():
                    break
                u2 = rand(sx, 20 + attempt)
                if big_on or loss_on:
                    u2 = warp(u2)
                u2 = 1.0 - (1.0 - u2) ** tilt
                j = np.where(held_td, seg0 + np.minimum(seg_n - 1, (u2 * seg_n).astype(np.int64)), j)
                held_td &= scores(j)
        shift = np.where(fr, 0.0, tables.sec_shift[np.where(stops, STOP, RUNNING), cell])
        used = np.maximum(1.0, tables.seconds[j] + shift) * pace[sx, so]
        if to_on:
            # the clock is running before this snap: a side may stop it with a timeout, as often
            # as real sides do there; the snap then takes only its own play's time
            win = ~stops & ~fr
            if win.any():
                wi = np.flatnonzero(win)
                call, side = calls(sx[wi], lead[wi])
                if call.any():
                    ci, dc = wi[call], side[call]
                    ns_ = tables.n_stop[key[ci]]
                    js = tables.start[key[ci]] + np.minimum(
                        np.maximum(ns_, 1) - 1, (rand(sx[ci], 42) * np.maximum(ns_, 1)).astype(np.int64))
                    # the clock stops when the play ends: it takes a stopped play's time
                    used[ci] = np.where(ns_ > 0, np.maximum(1.0, tables.seconds[js]) * pace[sx[ci], so[ci]],
                                        np.minimum(used[ci], 6.0))
                    if TO_PLAY_SECONDS and np.isfinite(tables.to_secs).any():
                        # v9: the time real plays stopped by a timeout took, by who called it
                        role = np.where(dc == so[ci], OFFENCE, DEFENCE)
                        qi = np.where(period[sx[ci]] >= 4, 1, 0)
                        caller_lead = np.where(role == OFFENCE, lead[ci], -lead[ci])
                        q = tables.to_secs[qi, role, caller_side(caller_lead)]
                        u = rand(sx[ci], 46)
                        got = np.array([np.interp(ui, TO_QUANTILES, qq) for ui, qq in zip(u, q)])
                        used[ci] = np.where(np.isfinite(got), np.maximum(1.0, got), used[ci])
                    tos[sx[ci], dc] -= 1
                    tally("timeouts", call.sum())
                    if stats is not None:
                        for q in (2, 4):
                            tally(f"timeouts_q{q}", (period[sx[ci]] == q).sum())
        clock[sx] -= used
        fresh[sx] = False
        if stats is not None and "_snaps" in stats:      # an audit's per-snap log (sim_audit)
            stats["_snaps"].append((sx.copy(), period[sx].copy(), clock[sx] + used, lead.copy(),
                                    down[sx].copy(), dist[sx].copy(), y[sx].copy(), tables.kind[j].copy(),
                                    tables.gain[j].copy(), used.copy(), stops.copy()))
        tally("clock_used", used.sum())
        tally("fresh_snaps", fr.sum())
        kd = tables.kind[j]
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
                    phase[s_] = KICK
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
        tdf = tables.td_from[j[gsel]]
        hg = held[gsel]
        short = (tdf >= 0) & (y[gx] < tdf) & ~hg
        if short.any():
            extra = tdf[short] - y[gx][short]
            gs = gx[short]
            runs_on = rand(gs, 12) < np.exp(-extra / TD_TAIL)
            y2[short] = np.where(runs_on, 100, y2[short] + (rand(gs, 13) * extra).astype(np.int32))
        y2 = np.where(hg, np.clip(y2, 1, 99), y2)
        gain = y2 - y[gx]
        td = y2 >= 100
        tally("td", td.sum())
        tally("big", (gain >= BIG_GAIN).sum())
        tally("loss", (gain < 0).sum())
        tally("loss_yards", -gain[gain < 0].sum())
        if stats is not None and td.any():
            tally("td_yards", (100 - y[gx][td]).sum())
            tally("td_long", (100 - y[gx][td] >= BIG_GAIN).sum())
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
            phase[s] = KICK
            free[s] = True
        rest = ~td & ~sf
        gx, gain, rp, y2 = gx[rest], gain[rest], rp[rest], y2[rest]
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
        reset = nt <= 0
        nd = np.where(reset, 1, nd)
        nt = np.where(reset, np.minimum(10, 100 - y2), nt)
        y[gx] = y2
        down[gx] = nd
        dist[gx] = nt
        tod = nd > 4
        tally("downs", tod.sum())
        tally("first_downs", first.sum())
        if tod.any():
            tx = gx[tod]
            turnover(tx, 100 - y[tx])

    tally("unfinished", (phase != DONE).sum())
    return score[:, 0].reshape(S, n_paths), score[:, 1].reshape(S, n_paths)


def fit_period_theta(tables, real_points, n_paths=30000, rounds=6, seed=1):
    """Each quarter's scoring level so the simulation from kickoff scores what the league scores."""
    for _ in range(rounds):
        got = kickoff_quarter_points(tables, n_paths, seed)
        for q in range(4):
            tables.period_theta[q + 1] += (real_points[q] - got[q]) / (1.8 * max(1.0, real_points[q]))
        tables.period_theta[5] = tables.period_theta[4]
    return tables.period_theta.copy(), got


def kickoff_quarter_points(tables, n_paths=30000, seed=1):
    """The simulation's average points in each quarter from kickoff."""
    start = Start(1)
    start.team[:] = 1
    stats = {}
    simulate(tables, start, n_paths, np.random.default_rng(seed), stats=stats)
    cum = [stats[f"points_by_q{q}"] / stats[f"ended_q{q}"] for q in (1, 2, 3, 4)]
    return np.diff([0.0] + cum)


def quarter_points(matches):
    """The league's average points in each quarter."""
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
