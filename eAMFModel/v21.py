"""v2.1: a snap simulation that draws each snap's time, then its distance given that time.

    python -m eAMFModel v21-build --until 2026-10-01 --out v21_full --mode full
    python -m eAMFModel v21-report --model v21_full --prior-model v14_model --since 2026-10-01 --until 2026-10-12

v2.1 stands apart from v8-v14. It reads the same snapshot export (PLAY_OVER rows) and nothing of their
fitted tables, and it has no strength: the pre-match expected points are not an input to the draws. The
sim plays from the real data alone, and v21_report sets its output against the expected points, so
where the two part shows where the draws are missing something. Nothing is ever fitted to make them
agree.

Each scrimmage snap is two draws, both from real snaps:

  time       the clock from the end of the play before to the end of this one: the wait for the snap
             (a running clock) and the play itself. Drawn from the real times of snaps alike in
             TIME_FEATURES: what the play before did to the clock (PREV: nothing gained -- mostly an
             incomplete pass --, a short gain, a first down, a loss, a fresh possession, a timeout),
             the part of the game (PHASES), the offense's lead, down, distance, field zone, the snaps
             played so far and the snaps left in the half. Bimodal by nature: a stopped clock gives a
             play's own 5-8 seconds, a running one 25-40, and the play before sets the split.
  distance   what came of the snap -- yards gained, a turnover, a defensive touchdown -- drawn from
             the real snaps alike in DIST_FEATURES, the first of them the time drawn (TIME_BUCKETS):
             a snap over in 5 seconds can only have gone so far.

Where too few real snaps share every feature (MIN_RECORDS), a draw falls back level by level to
fewer (TIME_LEVELS, DIST_LEVELS). The gamers and their teams come in one of two ways (the build's
`mode`):
  full       straight from what was seen: each draw takes, half the time, the offense's own real snaps
             (its gamer's where it has MIN_GAMER_RECORDS alike in GAMER_*_LEVELS, else its team's,
             else the league's), and half the time what the defense allowed (the defending gamer's,
             else its team's, else the league's).
  adjusted   the league's draws for every side, each leaned toward the longer or shorter end
             (u -> 1 - (1 - u) ** k, k = exp of the summed tilts) by a tilt for the offense's gamer,
             the defense's gamer and both teams, fitted on their real snaps against what the league's
             draws gave the same states, shrunk toward none by ADJUST_PRIOR snaps.

Bounds: a running clock that runs out before the snap ends the quarter; one that reaches 2:00 of Q2
or Q4 stops there and the snap takes a time from the stopped-clock part of the same draw (its real
snaps of STOPPED_MAX seconds or less); a play under way when time runs out is
finished. Real snaps cut short by either (the quarter ran out on them, or a running clock crossed
2:00) are kept in the time draw as "at least this long": drawn where more time is left than they
took, the draw moves on to a real snap at least as long. The distance stops at either end zone: past
the goal line is a touchdown, behind its own a safety; a real touchdown drawn from further back runs on
with the real chance a gain that long goes further (gain_surv).

Snaps left in the half (REMAIN) is a core split of both draws: the expected count from a table by
the half's 10-second slice, the clock running or stopped into the snap, both sides' timeouts and the
lead (remain_key), each cell shrunk toward its slice and clock's, that toward its slice's
(remain_table). Real snaps and the sim's read it the same way, off their own state -- never the count
that actually followed. The real games' counts first, then the sim's own from kick-off
(refine_remaining), the draws rebuilt on it each round (REFINE_ROUNDS).

A new set of downs is the next row's down (snap_records): spots are whole yards, so a sneak for inches
shows as a 0-yard gain that moves the chains, and a gain of the yards to go can be spotted short. The
sim takes the drawn real snap's own result where it was at the same distance, else the yards; a down
played again (a penalty) keeps its number.

The decisions (DECISIONS) are each the deciding gamer's: 4th downs (go, field goal, punt), a kick on
downs 1-3 in the half's last minute, kick-offs (returned, touchback, no landing zone, kept), going for
two and the timeouts before a running-clock snap (the defense's, then the offense's). The rules come
first, the same for every gamer: the options real sides never or all but never take in a situation,
learned off the real decisions in their full state (v21_rules, the sweep), are struck out there; with
overtime's own (ot_mask). The rest is drawn from the league's shares among real decisions alike in its
features, leaned by the gamer's own log-odds shift for each option (Choice, fit_leans): fitted on their
real decisions against the league's in the same states (those that took a struck-out option left out)
and shrunk toward none, so it carries into every state. What follows is real data too: the kicks'
fields and clock, field goals made by distance, punts' net yards and safety free kicks; a timeout moves
only the time draw. The game's own end (auto_end): in Q4, the side ahead with the ball whose downs left
at the 40-second play clock outlast the clock, the defense out of timeouts, wins there and then.
Overtime is v14's: over once both sides have had the
ball and one leads, a touchdown that puts the second side ahead ends it at once, and a period's end
decides it only when the side behind has had the ball.
"""

import datetime as dt
import math
import pickle
from bisect import bisect_left
from collections import defaultdict

import numpy as np

from . import sim14 as s14

QUARTER = 240.0
LATE = 120.0
PLAY_SECONDS = 7.0           # how long a play takes once snapped (the feed's stopped-clock median)
STOPPED_MAX = 15.0           # a snap's time no longer than this came off a stopped clock
MAX_SECONDS = 60.0
MAX_OT = 3
TIMEOUTS, OT_TIMEOUTS = 3, 2
MAX_STEPS = 900

KICK, SNAP, DONE = 0, 1, 2
GAIN, TURNOVER, DEF_TD = s14.GAIN, s14.TURNOVER, s14.DEF_TD

# the features, each a small integer
NOTHING, SHORT, FIRST, LOSS, FRESH, TIMEOUT = range(6)
PREV_NAMES = ("nothing gained", "short gain", "first down", "loss", "fresh possession", "timeout")
PHASES = ("Q1", "Q2", "Q2 2:00", "Q3", "Q4", "Q4 2:00", "OT")
MARGIN_EDGES = np.array([-8.5, -3.5, -0.5, 0.5, 3.5, 8.5])
DIST_EDGES = np.array([1.5, 3.5, 6.5, 10.5])
ZONE_EDGES = np.array([10.5, 30.5, 50.5, 70.5, 89.5, 95.5])     # the last two: the 10 to the 5, inside the 5
ELAPSED_EDGES = np.array([30.5, 60.5, 90.5])
REMAIN_EDGES = np.array([1.5, 2.5, 4.5, 7.5, 12.5])         # snaps left in the half: 0-1, 2, 3-4, 5-7, 8-12, 13+
TIME_EDGES = np.array([6.5, 9.5, 14.5, 24.5, 34.5])
TIME_BUCKETS = ("<=6s", "7-9s", "10-14s", "15-24s", "25-34s", "35s+")

SIZES = dict(prev=6, phase=7, margin=7, down=4, dist=5, zone=7, elapsed=4, remain=6, tb=6,
             lead3=3, tb2=2, dist3=3, zone3=3, rem3=3)                   # the last five: coarse ones (derive)
TIME_BASE = ("prev", "phase", "margin", "down", "dist", "zone", "elapsed", "remain")
DIST_BASE = ("tb", "down", "dist", "zone", "margin", "phase", "elapsed", "remain")
TIME_FEATURES = TIME_BASE + ("lead3",)                   # the fine features, then the coarse ones the fallbacks read
DIST_FEATURES = DIST_BASE + ("lead3", "zone3")


def _drop(features, *names):
    return tuple(f for f in features if f not in names)


# The fallbacks keep the situation -- the down, the distance, the field, the snaps left in the half (the
# sim's own expected count for the clock, whether it's running, both sides' timeouts and the lead) and
# coarsely the part of the game and the lead -- to the end: a 3rd down is never drawn from 1st downs,
# a side with time for one snap never from one with time for ten, while there are real snaps alike.
TIME_LEVELS = (TIME_BASE,
               _drop(TIME_BASE, "elapsed"),
               _drop(TIME_BASE, "elapsed", "zone"),
               ("prev", "phase", "margin", "down", "remain"),
               ("prev", "phase", "lead3", "down", "remain"),
               ("prev", "phase", "lead3", "remain"),
               ("prev", "phase", "remain"),
               ("prev", "remain"),
               ("prev",),
               ())
DIST_LEVELS = (DIST_BASE,
               _drop(DIST_BASE, "elapsed"),
               ("tb", "down", "dist", "zone", "lead3", "phase", "remain"),
               ("tb", "down", "dist", "zone", "lead3", "remain"),
               ("tb", "down", "dist", "zone", "remain"),
               ("tb", "down", "dist", "zone"),
               ("tb", "down", "dist", "zone3"),
               ("tb", "down", "dist"),
               ("tb", "down"),
               ("tb",),
               ())
# a gamer's or team's own snaps: alike in the situation, or the league's draw -- never their snaps of
# every down and distance
GAMER_TIME_FEATURES = ("who", "prev", "phase", "margin", "down", "rem3", "lead3")
GAMER_DIST_FEATURES = ("who", "tb", "down", "dist", "zone", "tb2", "dist3", "zone3")
GAMER_TIME_LEVELS = (("who", "prev", "phase", "margin", "down"), ("who", "prev", "rem3", "lead3"))
GAMER_DIST_LEVELS = (("who", "tb", "down", "dist", "zone"), ("who", "tb2", "down", "dist3", "zone3"))
MIN_RECORDS = 40
MIN_GAMER_RECORDS = 30
DEF_SHARE = 0.5                                           # full: the share of draws from what the defense allowed
MODES = ("full", "adjusted")
ADJUST_GRID = np.round(np.linspace(-1.5, 1.5, 31), 3)     # log tilts tried
ADJUST_PRIOR = 300.0
ADJUST_SWEEPS = 3
TO_VALUE, DEF_TD_VALUE = -20.0, -40.0                     # a turnover's place among the gains, for the order
SMOOTH_YARDS = 0                                          # +-yards a drawn gain is spread over (0: as drawn)
EFFECTS = ("gamer_off", "gamer_def", "team_off", "team_def")

GO, FG, PUNT = 0, 1, 2
FG_MAX = 65                   # longest kick tried (yards)
CONV_CELLS, CONV_MARGIN, CONV_CLOCK = s14.CONV_CELLS, s14.CONV_MARGIN, s14.CONV_CLOCK
RETURN, TOUCHBACK, NO_LANDING, KEPT = range(4)
KICK_TOUCHBACK, KICK_NO_LANDING = s14.KICK_TOUCHBACK, s14.KICK_NO_LANDING
LEFT_EDGES = np.array([10.5, 30.5, 60.5, 120.5, 240.5])     # seconds left in the half
EARLY_LEFT = 60.0                                            # early kicks: downs 1-3 in the half's last minute
EARLY_EDGES = np.array([5.5, 10.5, 20.5, 30.5, 45.5])
RANGE_EDGES = np.array([35.5, 45.5, 55.5])                   # the kick's length
FG_NEAR = 40.0                # a kick with this little left in the half takes its clock as the real ones did
KICK_POOL_MIN = 20

# The decisions. Each is drawn from the real ones in states alike (Bins over its features, falling back
# level by level below DECISION_MIN), and leaned by the deciding gamer's own log-odds shift for each
# option against the first (fitted on their real decisions, shrunk by DECISION_PRIOR).
DECISION_MIN = 25
DECISION_PRIOR = 3.0
DECISIONS = {
    "fourth": dict(label="4th down", options=("go for it", "field goal", "punt"), who="offense",
                   features=("togo10", "dist", "margin", "phase", "left", "lead3"), sizes=(10, 5, 7, 7, 6, 3),
                   levels=(("togo10", "dist", "margin", "phase", "left"), ("togo10", "dist", "lead3", "phase", "left"),
                           ("togo10", "dist", "lead3", "phase"), ("togo10", "dist", "phase"), ("togo10", "dist"),
                           ("togo10",), ())),
    "early_fg": dict(label="field goal on downs 1-3 in the half's last minute", options=("play on", "kick"),
                     who="offense", features=("left", "range", "down", "margin", "half"), sizes=(6, 4, 3, 7, 3),
                     levels=(("left", "range", "down", "margin", "half"), ("left", "range", "margin", "half"),
                             ("left", "range", "half"), ("left", "half"), ("left",), ())),
    "kickoff": dict(label="kick-off", options=("return", "touchback", "no landing zone", "kept"), who="kicker",
                    features=("desperate", "late", "lead"), sizes=(2, 2, 3),
                    levels=(("desperate", "late", "lead"), ("desperate", "late"), ("desperate",), ())),
    "two": dict(label="conversion", options=("kick", "go for two"), who="scorer", features=("cell", "lead"),
                sizes=(CONV_CELLS, 2 * CONV_MARGIN + 1), levels=(("cell", "lead"), ("lead",), ())),
    "to_def": dict(label="defense's timeout before a running-clock snap", options=("none", "timeout"),
                   who="defense", features=("phase", "margin", "left"), sizes=(7, 7, 6),
                   levels=(("phase", "margin", "left"), ("phase", "margin"), ("phase",), ())),
    "to_off": dict(label="offense's timeout before a running-clock snap", options=("none", "timeout"),
                   who="offense", features=("phase", "margin", "left"), sizes=(7, 7, 6),
                   levels=(("phase", "margin", "left"), ("phase", "margin"), ("phase",), ())),
}
MARGIN_NAMES = ("9+ behind", "4-8 behind", "1-3 behind", "level", "1-3 ahead", "4-8 ahead", "9+ ahead")
SLICE_NAMES = {
    "fourth": ("inside the 20, 1-3 to go", "inside the 20, 4+ to go", "20-39 out, 1-3 to go", "20-39 out, 4+ to go",
               "40-59 out, 1-3 to go", "40-59 out, 4+ to go", "60+ out, 1-3 to go", "60+ out, 4+ to go"),
    "early_fg": ("0:05 or less left", "0:06-0:10", "0:11-0:20", "0:21-0:30", "0:31-0:45", "0:46-1:00"),
    "kickoff": ("ordinary", "the half's last 2:00", "behind late in Q4"),
    "two": tuple(f"{m} after the six" for m in MARGIN_NAMES),
    "to_def": PHASES, "to_off": PHASES,
}
# the option each gamer's lean is shown for, and the real decisions a gamer needs to be shown
LEAN_SHOWN = {"fourth": (GO, 20), "early_fg": (1, 10), "kickoff": (TOUCHBACK, 20), "two": (1, 10),
              "to_def": (1, 40), "to_off": (1, 40)}

# The rules: options never or all but never taken in a situation, learned off the real decisions
# (v21_rules: the sweep), struck out for every gamer before the league's shares and any gamer's lean;
# and overtime's own (ot_mask). RULES off: the real shares alone.
RULES = True
# The game's own end: in Q4, the side ahead with the ball whose downs left, each taking the play clock,
# outlast the clock while the defense has no timeouts -- the game is over (auto_end).
AUTO_END = True
REMAIN_SLICE = 10.0           # snaps left in the half, by 10-second slice of the time left in it,
REMAIN_SLICES = int(2 * QUARTER // REMAIN_SLICE) + 1      # the clock running or not, both sides' timeouts, the lead
REMAIN_SHAPE = (3, REMAIN_SLICES, 2, 4, 4, 3)
REMAIN_PRIOR = 20.0           # a cell's own count shrunk toward its slice's (and clock's) by this many snaps
REFINE_REMAINING = True
REFINE_ROUNDS = 3             # the sim's own counts, the draws rebuilt on them, so many times over
REFINE_PATHS = 6000
SHARE_SLICE = 20.0            # points still to come, by 20-second slice of regulation (v21_report)
SHARE_SLICES = int(4 * QUARTER // SHARE_SLICE)


# ---------------------------------------------------------------- features

def phase_of(period, clock):
    period, clock = np.asarray(period), np.asarray(clock, float)
    late = clock <= LATE
    return np.select([period == 1, (period == 2) & ~late, period == 2, period == 3, (period == 4) & ~late,
                      period == 4], [0, 1, 2, 3, 4, 5], 6).astype(np.int64)


def _cut(v, edges):
    return np.searchsorted(edges, np.asarray(v, float), side="right").astype(np.int64)


def margin_b(m):
    return _cut(m, MARGIN_EDGES)


def dist_b(t):
    return _cut(t, DIST_EDGES)


def zone_b(y):
    return _cut(y, ZONE_EDGES)


def elapsed_b(n):
    return _cut(n, ELAPSED_EDGES)


def remain_b(r):
    return _cut(r, REMAIN_EDGES)


def time_b(s):
    return _cut(s, TIME_EDGES)


def left_b(s):
    return _cut(s, LEFT_EDGES)


def early_b(s):
    return _cut(s, EARLY_EDGES)


def range_b(k):
    return _cut(k, RANGE_EDGES)


def lead3(m):
    m = np.asarray(m)
    return np.where(m < 0, 0, np.where(m == 0, 1, 2)).astype(np.int64)


def togo10(y):
    return np.clip((100 - np.asarray(y)) // 10, 0, 9).astype(np.int64)


def decision_slice(name, c):
    """The report's slice (SLICE_NAMES) of each decision, off its features."""
    if name == "fourth":
        band = np.select([c["togo10"] <= 1, c["togo10"] <= 3, c["togo10"] <= 5], [0, 1, 2], 3)
        return (band * 2 + (np.asarray(c["dist"]) >= 2)).astype(np.int64)
    if name == "early_fg":
        return np.asarray(c["left"], np.int64)
    if name == "kickoff":
        return np.where(np.asarray(c["desperate"]) == 1, 2, np.asarray(c["late"])).astype(np.int64)
    if name == "two":
        return margin_b(np.asarray(c["lead"]) - CONV_MARGIN)
    return np.asarray(c["phase"], np.int64)


def down_b(d):
    return np.clip(np.asarray(d, np.int64) - 1, 0, 3)


def play_class(gain, dist):
    gain = np.asarray(gain)
    return np.where(gain == 0, NOTHING, np.where(gain < 0, LOSS, np.where(gain >= np.asarray(dist), FIRST, SHORT)))


def half_left(period, clock):
    """Seconds left in the half (regulation), in the overtime period for overtime."""
    period, clock = np.asarray(period), np.asarray(clock, float)
    return np.where((period == 1) | (period == 3), clock + QUARTER, clock)


def remain_slice(period, clock):
    """(half: 0 first, 1 second, 2 overtime, 10-second slice of the time left in it)."""
    period = np.asarray(period)
    half = np.where(period <= 2, 0, np.where(period <= 4, 1, 2))
    sl = np.clip((half_left(period, clock) // REMAIN_SLICE).astype(np.int64), 0, REMAIN_SLICES - 1)
    return half, sl


def running_of(prev):
    """Whether the clock runs into the snap off the play before (a gain short of the line, a first
    down, a loss); nothing gained (mostly an incomplete pass), a fresh possession and a timeout stop it."""
    return np.isin(np.asarray(prev), (SHORT, FIRST, LOSS)).astype(np.int64)


def remain_key(period, clock, running, own_to, opp_to, lead3):
    """The snaps-left table's index of a state: the half, its 10-second slice, the clock running or
    not, the side with the ball's timeouts and the other side's (0-3) and the lead (behind, level,
    ahead)."""
    half, sl = remain_slice(period, clock)
    to = lambda v: np.clip(np.asarray(v, np.int64), 0, 3)
    return half, sl, np.asarray(running, np.int64), to(own_to), to(opp_to), np.asarray(lead3, np.int64)


def remain_table(key, counts, prior=REMAIN_PRIOR, base=None):
    """The expected snaps left in the half for every cell of REMAIN_SHAPE off states (key) and the
    snaps that followed each (counts): each cell's own average shrunk toward its slice and clock's,
    that toward its slice's; slices with none read off their neighbours (or `base`'s)."""
    num, den = np.zeros(REMAIN_SHAPE), np.zeros(REMAIN_SHAPE)
    np.add.at(num, key, counts)
    np.add.at(den, key, 1)
    n2, s2 = den.sum((2, 3, 4, 5)), num.sum((2, 3, 4, 5))
    m2 = np.zeros(n2.shape)
    for h in range(n2.shape[0]):
        have = np.flatnonzero(n2[h] > 0)
        if len(have):
            m2[h] = np.interp(np.arange(n2.shape[1]), have, s2[h, have] / n2[h, have])
        elif base is not None:
            m2[h] = base[h].mean((1, 2, 3, 4))
        else:
            m2[h] = np.arange(n2.shape[1]) * REMAIN_SLICE / 20.0
    n3, s3 = den.sum((3, 4, 5)), num.sum((3, 4, 5))
    m3 = (s3 + prior * m2[..., None]) / (n3 + prior)
    return (num + prior * m3[..., None, None, None]) / (den + prior)


def share_slice(period, clock):
    """The 20-second slice of regulation a clock is in (0 the first; overtime the last)."""
    period, clock = np.asarray(period), np.asarray(clock, float)
    played = (np.minimum(period, 4) - 1) * QUARTER + (QUARTER - clock)
    return np.where(period >= 5, SHARE_SLICES - 1,
                    np.clip((played // SHARE_SLICE).astype(np.int64), 0, SHARE_SLICES - 1))


def ot_mask(name, n, must_score=None, other_done_close=None):
    """(decisions, options) what overtime's own rules leave, or None: a side behind once the other side
    has had the ball (must_score) can't punt, nor kick a field goal 4+ behind (lead4: the lead in
    must_score's place); after a touchdown 1 or 2 behind once the other side has had it
    (other_done_close) it goes for two."""
    if name in ("fourth", "early_fg") and must_score is not None:
        must, behind4 = must_score
        if not must.any():
            return None
        ok = np.ones((n, len(DECISIONS[name]["options"])), bool)
        if name == "fourth":
            ok[must, PUNT] = False
            ok[must & behind4, FG] = False
        else:
            ok[must & behind4, 1] = False
        return ok
    if name == "two" and other_done_close is not None and np.any(other_done_close):
        ok = np.ones((n, 2), bool)
        ok[np.asarray(other_done_close, bool), 0] = False
        return ok
    return None


def auto_end(period, clock, lead, down, def_timeouts):
    """Whether the game is over by its own end: Q4, the side with the ball ahead, its downs left at the
    play clock each outlasting the clock, the defense out of timeouts."""
    period, clock, lead = np.asarray(period), np.asarray(clock, float), np.asarray(lead)
    return (period == 4) & (lead > 0) & (np.asarray(def_timeouts) == 0) \
        & ((5 - np.asarray(down)) * s14.PLAY_CLOCK > clock)


def combine(*masks):
    """The options every mask leaves (None: no mask)."""
    got = [m for m in masks if m is not None]
    if not got:
        return None
    ok = np.logical_and.reduce(got)
    ok[~ok.any(1)] = True
    return ok


# ---------------------------------------------------------------- the draws

class Bins:
    """Real records grouped for drawing: a group for every combination of each level's features that
    holds `min_records` (every level but the last, (); with `strict` that one too), each group's records
    sorted by their value, so a draw (and a tilt) runs from the bottom of the group to the top. A draw's
    group is the first level, finest first, whose group for its features is there (strict: none if no
    level's is)."""

    def __init__(self, features, sizes, levels, codes, start, count, level_of, payload):
        self.features, self.sizes, self.levels = tuple(features), tuple(sizes), tuple(levels)
        self.codes = codes                 # per level: (its groups' codes, sorted; the first group's index)
        self.start, self.count, self.level_of, self.payload = start, count, level_of, payload

    def _dims(self, lv):
        pos = {f: i for i, f in enumerate(self.features)}
        return tuple(self.sizes[pos[f]] for f in lv)

    @classmethod
    def build(cls, features, sizes, levels, X, value, payload, min_records, strict=False):
        """X (records, features) ints; value the sort key; payload {name: array} kept with them."""
        X = np.asarray(X, np.int64).reshape(-1, len(features))
        value = np.asarray(value, float)
        n = len(X)
        sizes = tuple(int(x) for x in sizes)
        pos = {f: i for i, f in enumerate(features)}
        codes, start, count, level_of, take = [], [], [], [], []
        total = 0
        for li, lv in enumerate(levels):
            cols = [pos[f] for f in lv]
            dims = tuple(sizes[c] for c in cols)
            rc = np.ravel_multi_index(tuple(np.clip(X[:, c], 0, sizes[c] - 1) for c in cols), dims) if cols \
                else np.zeros(n, np.int64)
            need = min_records if (strict or lv) else 1
            o = np.lexsort((value, rc))
            sc = rc[o]
            uniq, first, cnt = np.unique(sc, return_index=True, return_counts=True)
            ok = cnt >= need
            codes.append((uniq[ok], len(start)))
            for u0, f0, c0 in zip(uniq[ok], first[ok], cnt[ok]):
                start.append(total)
                count.append(int(c0))
                level_of.append(li)
                take.append(o[f0:f0 + c0])                # sorted by value within the group
                total += int(c0)
        idx = np.concatenate(take) if take else np.zeros(0, np.int64)
        return cls(features, sizes, levels, codes, np.array(start, np.int64), np.array(count, np.int64),
                   np.array(level_of, np.int64), {k: np.asarray(v)[idx] for k, v in payload.items()})

    def bin_of(self, cols):
        """The group each row of features (a dict of arrays) draws from (-1: none)."""
        n = len(np.asarray(cols[self.features[0]]))
        out = np.full(n, -1, np.int64)
        for li, lv in enumerate(self.levels):
            elig, first = self.codes[li]
            todo = np.flatnonzero(out < 0)
            if not len(todo):
                break
            if not len(elig):
                continue
            if lv:
                dims = self._dims(lv)
                rc = np.ravel_multi_index(tuple(np.asarray(cols[f], np.int64)[todo] for f in lv), dims, mode="clip")
            else:
                rc = np.zeros(len(todo), np.int64)
            k = np.minimum(np.searchsorted(elig, rc), len(elig) - 1)
            hit = elig[k] == rc
            out[todo[hit]] = first + k[hit]
        return out

    def draw(self, b, u, k=None):
        """The record each draw lands on: u uniform, leaned toward the top of its group by k."""
        if k is not None:
            u = 1.0 - np.power(np.clip(1.0 - u, 0.0, 1.0), k)
        n = self.count[b]
        return self.start[b] + np.minimum(n - 1, (u * n).astype(np.int64))

    def draw_low(self, b, u, value, limit):
        """A draw from the part of each group at or under `limit` (value: the payload the group is
        sorted by); a group with none there draws as usual."""
        cache = self.__dict__.setdefault("_low", {})
        if limit not in cache:
            cache[limit] = np.add.reduceat((self.payload[value] <= limit).astype(np.int64), self.start) \
                if len(self.start) else np.zeros(0, np.int64)
        n = cache[limit][b]
        full = self.count[b]
        n = np.where(n > 0, n, full)
        return self.start[b] + np.minimum(n - 1, (u * n).astype(np.int64))

    def above(self, j, u):
        """A record at or above record j in its group (j's group read off its position), u uniform."""
        g = np.searchsorted(self.start, j, side="right") - 1
        end = self.start[g] + self.count[g]
        return j + np.minimum(end - j - 1, (u * (end - j)).astype(np.int64))

    def tilted_means(self, values, grid):
        """(bins, len(grid)) each group's mean value under each log tilt of the grid."""
        n = np.repeat(self.count, self.count)
        i = np.arange(len(values)) - np.repeat(self.start, self.count)
        out = np.zeros((len(self.count), len(grid)))
        for g, lk in enumerate(grid):
            inv = math.exp(-lk)
            G = lambda x: 1.0 - np.power(np.clip(1.0 - x, 0.0, 1.0), inv)
            w = G((i + 1) / n) - G(i / n)
            out[:, g] = np.add.reduceat(w * values, self.start) if len(values) else 0.0
        return out

    def level_shares(self, b):
        """The share of draws (their groups b) at each level."""
        lv = np.bincount(self.level_of[b[b >= 0]], minlength=len(self.levels))
        return lv / max(1, lv.sum())


def fit_leans(base, chosen, who, n_who, prior=DECISION_PRIOR, steps=8):
    """(n_who, options) each gamer's log-odds shift of each option against the first: Newton steps on
    their real decisions' likelihood around the league's shares (base: each decision's log shares),
    with a normal prior of precision `prior` pulling every shift toward none."""
    K = base.shape[1]
    S = np.zeros((n_who, K))
    Y = np.eye(K)[chosen]
    for _ in range(steps):
        z = base + S[who]
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        for k in range(1, K):
            g = np.bincount(who, Y[:, k] - p[:, k], minlength=n_who) - prior * S[:, k]
            h = np.bincount(who, p[:, k] * (1 - p[:, k]), minlength=n_who) + prior
            S[:, k] += g / h
        S[-1] = 0.0                                       # the unknown gamer: none
    return S


class Choice:
    """A decision drawn from real ones: the league's share of each option among the real decisions in
    states alike (Bins over its features), leaned by the deciding gamer's log-odds shift (lean)."""

    def __init__(self, name, bins, shares, lean, league, n, counts, ruled=0, broke=0):
        self.name, self.bins, self.shares, self.lean, self.league, self.n = name, bins, shares, lean, league, n
        self.counts = counts                              # real decisions by gamer and option
        self.ruled, self.broke = ruled, broke             # real decisions under a global rule, and how many broke it

    @staticmethod
    def _masked(base, allowed):
        """Shares with the options a rule strikes out taken away, the rest scaled up (all of them
        equal where none of the allowed ones was ever chosen)."""
        if allowed is None:
            return base
        m = base * allowed
        tot = m.sum(1, keepdims=True)
        return np.where(tot > 0, m / np.where(tot > 0, tot, 1), allowed / allowed.sum(1, keepdims=True))

    @classmethod
    def build(cls, name, X, chosen, who, n_who, min_records=DECISION_MIN, prior=DECISION_PRIOR, gamers=True,
              allowed=None):
        """allowed: (decisions, options) what the rules left each real decision (None: no rules)."""
        spec = DECISIONS[name]
        K = len(spec["options"])
        chosen, who = np.asarray(chosen, np.int64), np.asarray(who, np.int64)
        counts = np.zeros((n_who, K))
        np.add.at(counts, (who, chosen), 1)
        if not len(chosen):
            league = np.eye(K)[0]
            return cls(name, None, league[None], np.zeros((n_who, K)), league, 0, counts)
        league = np.bincount(chosen, minlength=K) / len(chosen)
        X = np.asarray(X, np.int64).reshape(len(chosen), -1)
        bins = Bins.build(spec["features"], spec["sizes"], spec["levels"], X, chosen.astype(float),
                          {"choice": chosen}, min_records)
        group = np.repeat(np.arange(len(bins.count)), bins.count)
        got = np.zeros((len(bins.count), K))
        np.add.at(got, (group, bins.payload["choice"]), 1)
        shares = (got + league) / (bins.count[:, None] + 1.0)
        bins.payload = {}
        cols = {f: X[:, i] for i, f in enumerate(spec["features"])}
        ruled = broke = 0
        keep = np.ones(len(chosen), bool)
        if allowed is not None and not allowed.all():
            under = ~allowed.all(1)
            keep = allowed[np.arange(len(chosen)), chosen]
            ruled, broke = int(under.sum()), int((~keep).sum())
        lean = np.zeros((n_who, K))
        mine = bins.bin_of(cols)
        if gamers and keep.any():
            base = cls._masked(shares[mine], None if allowed is None or allowed.all() else allowed)[keep]
            lean = fit_leans(np.log(np.maximum(base, 1e-9)), chosen[keep], who[keep], n_who, prior)
        out = cls(name, bins, shares, lean, league, len(chosen), counts, ruled, broke)
        out.real_bins = mine.astype(np.int32)              # the real decisions' groups, for gamer_rate
        return out

    def gamer_rate(self, g, opt):
        """How often gamer g would take the option over every real decision's state (their lean on the
        league's shares there)."""
        bins = getattr(self, "real_bins", None)
        if bins is None or self.bins is None:
            return float(self.league[opt])
        z = np.log(np.maximum(self.shares[bins], 1e-12)) + self.lean[g]
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return float((p[:, opt] / p.sum(1)).mean())

    def probs(self, cols, who, allowed=None):
        base = self.shares[self.bins.bin_of(cols)] if self.bins is not None else np.tile(self.shares[0], (len(who), 1))
        base = self._masked(base, allowed)
        z = np.log(np.maximum(base, 1e-12)) + self.lean[who]
        if allowed is not None:
            z = np.where(allowed, z, -np.inf)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def draw(self, cols, who, u, allowed=None):
        c = np.cumsum(self.probs(cols, who, allowed), 1)
        return np.minimum((u[:, None] > c).sum(1), c.shape[1] - 1)


# ---------------------------------------------------------------- the real snaps

def _side_of(team, a_home):
    return 0 if (team == "TEAM_A") == a_home else 1


def _with_side(rows):
    return [r if r.get("team_a_side") in ("home", "away") else dict(r, team_a_side="home") for r in rows]


def snap_records(rows):
    """Every scrimmage snap of a match, as sim14.snap_records reads them, with what the next row says
    the down became: `converted` (a new set of downs, or a touchdown) and `replay` (the same down again,
    the gain short of the line: a penalty). Spots are whole yards, so a gain of 0 can move the chains (a
    sneak for inches) and one of the yards to go can fall short; sim14 left out the 4th downs that kept
    the ball short of the line, which these are kept."""
    _i, _f = s14._i, s14._f
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in s14.SNAP_KINDS or not a["down"] or not a["field_position"]:
            continue
        if not a["period"] or a["period"] != b["period"] or not a["clock_seconds"] \
                or not b["clock_seconds"] or not a["distance"]:
            continue
        seconds = _f(a["clock_seconds"]) - _f(b["clock_seconds"])
        if not 0 <= seconds <= 60:
            continue
        down, t, y = _i(a["down"]), max(1, _i(a["distance"])), _i(a["field_position"])
        rec = dict(offense=a["offense"], period=_i(a["period"]), clock=_f(a["clock_seconds"]),
                   margin=s14._margin(a, a["offense"]), down=down, distance=t, field=y, seconds=seconds,
                   message=_i(b["message"]), kind=GAIN, gain=0, new_field=0, replay=False, converted=False,
                   fresh=s14.is_fresh(a), start=_i(a["message"]))
        if rec["margin"] is None or rec["clock"] is None or y is None or down is None:
            continue
        bk = b["play_kind"]
        if bk == "TOUCHDOWN":
            scorer = s14._scorer(b, "TOUCHDOWN_TEAM") or b["offense"]
            if scorer == a["offense"]:
                rec["gain"], rec["converted"] = 100 - y, True
            else:
                rec["kind"] = DEF_TD
        elif bk in ("SCRIMMAGE", "TURNOVER_ON_DOWNS"):
            if not b["field_position"]:
                continue
            by = _i(b["field_position"])
            if b["offense"] == a["offense"]:
                rec["gain"] = by - y
                nd = _i(b["down"])
                rec["replay"] = nd == down and by - y < t
                rec["converted"] = nd == 1
            elif bk == "TURNOVER_ON_DOWNS" or (down == 4 and bk == "SCRIMMAGE"
                                               and "TURNOVER_ON_DOWNS" in (b["play_messages"] or "")):
                rec["gain"] = 100 - by - y
            else:
                rec["kind"] = TURNOVER
                rec["new_field"] = by
        else:
            continue
        s14._timeouts_at(rec, a)
        out.append(rec)
    return out


def match_snaps(rows, pair=None, teams=None):
    """A match's scrimmage snaps with the features both draws read and who had the ball: a list of
    dicts (sim14.snap_records plus side, prev_cls, elapsed, remain, cut, play_seconds, the gamers and
    teams)."""
    rows = _with_side(rows)
    if not rows:
        return []
    a_home = rows[0]["team_a_side"] != "away"
    recs = s14.link_prev(snap_records(rows))
    for n, r in enumerate(recs):
        r["side"] = _side_of(r["offense"], a_home)
        r["elapsed"] = n
        r["prev_raw"] = FRESH if (r.get("fresh") or r["prev"] < 0) else int(r["prev"])
        r["prev_cls"] = TIMEOUT if r.get("timeout_role") is not None else r["prev_raw"]
        c, s = r["clock"], r["seconds"]
        running = r["prev_cls"] <= LOSS
        warned = running and r["period"] in (2, 4) and c > LATE and c - s < LATE
        r["cut"] = (c - s <= 0.5) or warned                   # the quarter's end or 2:00 cut it short
        r["play_seconds"] = max(1.0, s - (c - LATE)) if warned else s
        r["gamer"] = (pair[r["side"]], pair[1 - r["side"]]) if pair else ("", "")
        r["team"] = (teams[r["side"]], teams[1 - r["side"]]) if teams else ("", "")
    halves = defaultdict(list)
    for r in recs:
        halves[0 if r["period"] <= 2 else 1 if r["period"] <= 4 else 2].append(r)
    for h in halves.values():
        for k, r in enumerate(h):
            r["remain"] = len(h) - k - 1
    return recs


TEAMS = ("TEAM_A", "TEAM_B")


def _tos_left(row, team, period):
    """(the side's timeouts left, the other side's) at a row, or (-1, -1) where the row doesn't carry
    them (playover.annotate_timeouts)."""
    a, b = s14._i(row.get("timeouts_used_a")), s14._i(row.get("timeouts_used_b"))
    if a is None or b is None or team not in TEAMS:
        return -1, -1
    have = TIMEOUTS if period <= 4 else OT_TIMEOUTS
    mine, theirs = (a, b) if team == "TEAM_A" else (b, a)
    return max(0, have - mine), max(0, have - theirs)


def _prev_at(row, by_end, period):
    """The play before a PLAY_OVER as the next snap's clock sees it (PREV), off the match's snaps."""
    last = by_end.get(s14._i(row["message"]))
    if s14.is_fresh(row) or last is None or last["offense"] != row.get("offense") or last["period"] != period \
            or last["kind"] != GAIN or last["gain"] >= 100 - last["field"]:
        return FRESH
    return int(play_class(last["gain"], last["distance"]))


def decision_points(rows, recs=None):
    """A match's real decisions in their full state: {name: [dict]} for every name in DECISIONS, each
    with the deciding side (0 home, 1 away), the choice, the period, clock, seconds left in the half
    (left), the deciding side's lead, yards to the goal (togo), to go (dist), the down, both sides'
    timeouts left (own_to, opp_to; -1 unknown), the play before (prev) and the row (msg); and
    "kick_pool" (choice, desperate, late, field, seconds) for every kick-off, "fg_clock" (seconds left in
    the half, seconds used) for every field goal and "conv" (two, good) for every conversion. `recs`:
    the match's match_snaps."""
    rows = _with_side(rows)
    out = {k: [] for k in DECISIONS}
    out.update(kick_pool=[], fg_clock=[], conv=[])
    if not rows:
        return out
    recs = recs if recs is not None else match_snaps(rows)
    by_end = {r["message"]: r for r in recs}
    a_home = rows[0]["team_a_side"] != "away"
    _i, _f = s14._i, s14._f
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in s14.SNAP_KINDS or a.get("offense") not in TEAMS or not a.get("field_position") \
                or not a.get("down") or not a.get("period") or not a.get("clock_seconds"):
            continue
        p, c, y, d = _i(a["period"]), _f(a["clock_seconds"]), _i(a["field_position"]), _i(a["down"])
        m = s14._margin(a, a["offense"])
        if c <= 0 or m is None:
            continue
        side, left, same = _side_of(a["offense"], a_home), float(half_left(p, c)), a["period"] == b["period"]
        if b["play_kind"] == "FIELD_GOAL" and same and b.get("clock_seconds"):
            out["fg_clock"].append((left, c - _f(b["clock_seconds"])))
        own, opp = _tos_left(a, a["offense"], p)
        point = dict(side=side, period=p, clock=c, left=left, lead=m, togo=100 - y,
                     dist=max(1, _i(a["distance"]) or 1), down=d, own_to=own, opp_to=opp,
                     prev=_prev_at(a, by_end, p), msg=_i(a["message"]))
        if d == 4 and same and a.get("distance"):
            point["choice"] = PUNT if b["play_kind"] == "PUNT" else FG if b["play_kind"] == "FIELD_GOAL" else GO
            out["fourth"].append(point)
        elif d in (1, 2, 3) and 100 - y + 17 <= FG_MAX and (same or p in (2, 4)):
            # a half that ran out before the next row is a side that did not kick
            point["choice"] = int(b["play_kind"] == "FIELD_GOAL" and same)
            out["early_fg"].append(point)
    # kick-offs: after a conversion or a field goal (a touchdown straight after: returned or run in), and
    # each half's opening kick
    prev = None
    for r in rows:
        kind_r, got = r["play_kind"], None
        after = prev is not None and prev["period"] == r["period"] and prev["play_kind"] in ("CONVERSION", "FIELD_GOAL") \
            and bool(prev.get("clock_seconds")) and prev.get("offense") in TEAMS
        if r.get("period") and r.get("clock_seconds"):
            if after and kind_r == "TOUCHDOWN":
                kicker = prev["offense"]
                scorer = s14._scorer(r, "TOUCHDOWN_TEAM") or r["offense"]
                got = (kicker, _f(prev["clock_seconds"]), s14._margin(prev, kicker), 100,
                       KEPT if scorer == kicker else RETURN, prev)
            elif kind_r == "KICKOFF" and r.get("field_position") and r.get("offense") in TEAMS:
                if after:
                    kicker, start, at = prev["offense"], _f(prev["clock_seconds"]), prev
                    lead = s14._margin(prev, kicker)
                elif prev is None or prev["period"] != r["period"]:
                    kicker, start, at = ("TEAM_B" if r["offense"] == "TEAM_A" else "TEAM_A"), QUARTER, r
                    lead = s14._margin(r, kicker)
                else:
                    kicker = None
                if kicker:
                    field, secs = _i(r["field_position"]), start - _f(r["clock_seconds"])
                    choice = KEPT if r["offense"] == kicker else TOUCHBACK if (field == KICK_TOUCHBACK and secs == 0) \
                        else NO_LANDING if (field == KICK_NO_LANDING and secs == 0) else RETURN
                    if s14._scorer(r, "TOUCHDOWN_TEAM") is not None:
                        field = 100                       # run in from the kick
                    got = (kicker, start, lead, field, choice, at)
        if got is not None and got[2] is not None:
            kicker, start, lead, field, choice, at = got
            p, secs = _i(r["period"]), start - _f(r["clock_seconds"])
            if 0 <= secs <= 60:
                left = float(half_left(p, start))
                desperate = int(after and p >= 4 and lead < 0 and left <= 180)
                late = int(p in (2, 4) and start <= LATE)
                own, opp = _tos_left(at, kicker, p)
                out["kickoff"].append(dict(side=_side_of(kicker, a_home), choice=choice, period=p, clock=start,
                                           left=left, lead=lead, own_to=own, opp_to=opp, desperate=desperate,
                                           late=late, msg=_i(r["message"])))
                out["kick_pool"].append((choice, desperate, late, field, secs))
        prev = r
    # conversions: who scored, the lead after the six, whether they went for two and made it
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] != "TOUCHDOWN" or b["play_kind"] != "CONVERSION" or not a.get("period"):
            continue
        msgs = [x for x in (b.get("play_messages") or "").split("|") if "POINT" in x and x[-6:] in TEAMS]
        if not msgs:
            continue
        scorer = msgs[-1][-6:]
        two = any("TWO_POINT" in x for x in msgs)
        good = msgs[-1].startswith(("EXTRA_POINT_GOOD", "TWO_POINT_CONVERSION_SUCCESSFUL"))
        m = s14._margin(a, scorer)
        if m is None:
            continue
        p, c = _i(a["period"]), _f(a.get("clock_seconds")) or 0.0
        own, opp = _tos_left(a, scorer, p)
        out["two"].append(dict(side=_side_of(scorer, a_home), choice=int(two), period=p, clock=c,
                               left=float(half_left(p, c)), lead=m, own_to=own, opp_to=opp,
                               cell=s14.conversion_cell(p, c), msg=_i(a["message"])))
        out["conv"].append((two, good))
    # timeouts before a running-clock snap, with a timeout left to call: the defense's first, then
    # (the defense not calling) the offense's
    for r in recs:
        if r["prev_raw"] > LOSS or not r.get("timeouts_left"):
            continue
        base = dict(period=r["period"], clock=r["clock"], left=float(half_left(r["period"], r["clock"])),
                    togo=100 - r["field"], dist=r["distance"], down=r["down"], prev=r["prev_raw"],
                    off_margin=r["margin"], msg=r["start"])
        role = r.get("timeout_role")
        off_left, def_left = r["timeouts_left"]
        if def_left > 0:
            out["to_def"].append(dict(base, side=1 - r["side"], choice=int(role == s14.DEFENCE), lead=-r["margin"],
                                      own_to=def_left, opp_to=off_left))
        if off_left > 0 and role != s14.DEFENCE:
            out["to_off"].append(dict(base, side=r["side"], choice=int(role == s14.OFFENCE), lead=r["margin"],
                                      own_to=off_left, opp_to=def_left))
    return out


def decision_features(name, pts):
    """The draws' features (DECISIONS[name]["features"]) of real decision points, as an int array."""
    if not pts:
        return np.zeros((0, len(DECISIONS[name]["features"])), np.int64)
    g = lambda k: np.array([q[k] for q in pts])
    if name == "fourth":
        mb = margin_b(g("lead"))
        cols = [togo10(100 - g("togo")), dist_b(g("dist")), mb, phase_of(g("period"), g("clock")),
                left_b(g("left")), derive(dict(margin=mb))["lead3"]]
    elif name == "early_fg":
        p = g("period")
        cols = [early_b(g("left")), range_b(g("togo") + 17), g("down") - 1, margin_b(g("lead")),
                np.where(p == 2, 0, np.where(p == 4, 1, 2))]
    elif name == "kickoff":
        cols = [g("desperate"), g("late"), lead3(g("lead"))]
    elif name == "two":
        cols = [g("cell"), np.clip(g("lead"), -CONV_MARGIN, CONV_MARGIN) + CONV_MARGIN]
    else:
        cols = [phase_of(g("period"), g("clock")), margin_b(g("off_margin")), left_b(g("left"))]
    return np.stack([np.asarray(c, np.int64) for c in cols], axis=1)


def in_draw_window(name, pts):
    """Which real decision points v2.1's sim decides at (early kicks: downs 1-3 in the half's last
    minute of Q2, Q4 and overtime; the rest: all)."""
    if name != "early_fg":
        return np.ones(len(pts), bool)
    return np.array([(q["period"] == 2 or q["period"] >= 4) and q["left"] <= EARLY_LEFT for q in pts], bool)


def match_decisions(rows, recs=None):
    """A match's real decisions as the draws read them: {name: [(side, choice, *features)]} (only those
    the sim decides at), with decision_points' kick_pool, fg_clock and conv."""
    pts = decision_points(rows, recs)
    out = {k: pts[k] for k in ("kick_pool", "fg_clock", "conv")}
    for name in DECISIONS:
        keep = in_draw_window(name, pts[name])
        got = [q for q, k in zip(pts[name], keep) if k]
        X = decision_features(name, got)
        out[name] = [(q["side"], q["choice"]) + tuple(int(v) for v in x) for q, x in zip(got, X)]
    return out


def _turnover_shift(rec):
    """A real turnover's move of the spot, toward the defense's goal, from where it was snapped."""
    return (100 - int(rec["new_field"])) - int(rec["field"])


def _score_path(rows):
    """[(slice of regulation, home + away)] at every row with a score and a clock."""
    out = []
    for r in rows:
        p, c, h, a = s14._i(r.get("period")), s14._f(r.get("clock_seconds")), s14._i(r.get("score_p1")), \
            s14._i(r.get("score_p2"))
        if None not in (p, c, h, a):
            out.append((int(share_slice(p, c)), h + a))
    return out


def points_share(matches):
    """The share of a game's points still to come at the start of each 20-second slice of regulation
    (the last slice: overtime's), the real games' points pooled."""
    left, final = np.zeros(SHARE_SLICES), 0.0
    for rows in matches.values():
        path = _score_path(rows)
        if not path:
            continue
        end = path[-1][1]
        final += end
        at, k = 0, 0
        for sl in range(SHARE_SLICES):
            while k < len(path) and path[k][0] < sl:
                at = path[k][1]
                k += 1
            left[sl] += end - at
    return left / final if final else np.linspace(1, 0, SHARE_SLICES)


# ---------------------------------------------------------------- the model

class Model:
    """Everything v2.1 draws from."""

    def __init__(self):
        self.mode = "full"
        self.time = self.dist = None
        self.own = {}                      # full: {(who, role, "time"|"dist"): Bins}
        self.gamers, self.teams = {}, {}
        self.tilt = {}                     # adjusted: {"time"|"dist": {effect: arr}}
        self.gain_surv = np.ones(101)      # P(a positive gain reaches x yards)
        self.choices = {}                  # the decisions: {name in DECISIONS: Choice}
        self.fg_make = np.array([0.8])     # P(make) by 5-yard kick distance from 15
        self.fg_near = np.array([[5.0, 5.0]])   # (seconds left, seconds used) of real kicks FG_NEAR or less from the end
        self.fg_far = np.array([5.0])      # seconds used by the rest
        self.punts = None                  # Bins over (ybin,) payload net, seconds
        self.kick_pool = {}                # (RETURN | KEPT, desperate, late) -> (field, seconds)
        self.safety_fields = np.array([35])
        self.two_good, self.kick_good = 0.5, 0.97
        self.rules = {}                    # learned (v21_rules): {decision: [(option, {feature: (lo, hi)})]}
        self.use_rules = True
        self.remain = np.zeros(REMAIN_SHAPE)
        self.share = np.linspace(1, 0, SHARE_SLICES)
        self.meta = {}

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        m = cls()
        with open(path, "rb") as fh:
            got = pickle.load(fh)
        if "choices" not in got:
            raise SystemExit(f"{path} was built before v2.1's decisions: build it again with v21-build")
        m.__dict__.update(got)
        return m

    def gamer_index(self, handle):
        return self.gamers.get((handle or "").strip().upper(), len(self.gamers))

    def team_index(self, team):
        return self.teams.get((team or "").strip(), len(self.teams))

    def remain_at(self, period, clock, running, own_to, opp_to, lead3):
        """The expected snaps left in the half (remain_key)."""
        return self.remain[remain_key(period, clock, running, own_to, opp_to, lead3)]


MODEL_FILE = "v21model.pkl"


ZONE3 = np.array([0, 0, 0, 1, 1, 2, 2])                  # own half, the other half outside the 10, inside the 10


def late_of(period, clock):
    """The end of a half: 0 the rest, 1 Q2's or Q4's 2:00 to 0:31, 2 their last 0:30, 3 overtime."""
    period, clock = np.asarray(period), np.asarray(clock, float)
    end = (period == 2) | (period == 4)
    return np.select([period >= 5, end & (clock <= 30), end & (clock <= LATE)], [3, 2, 1], 0).astype(np.int64)


def derive(f):
    """The coarse features off the fine ones: the lead (lead3: behind, level, ahead), the clock (tb2:
    stopped, running), the distance (dist3: 1-3, 4-6, 7+) and the field (zone3)."""
    if "margin" in f:
        f["lead3"] = np.select([f["margin"] <= 2, f["margin"] == 3], [0, 1], 2).astype(np.int64)
    if "remain" in f:
        f["rem3"] = np.select([np.asarray(f["remain"]) <= 2, np.asarray(f["remain"]) <= 4], [0, 1], 2).astype(np.int64)
    if "tb" in f:
        f["tb2"] = (np.asarray(f["tb"]) >= 3).astype(np.int64)
    if "dist" in f:
        f["dist3"] = np.minimum(2, np.maximum(0, np.asarray(f["dist"]) - 1)).astype(np.int64)
    if "zone" in f:
        f["zone3"] = ZONE3[np.asarray(f["zone"])]
    return f


def remain_inputs(recs):
    """(period, clock, running, own timeouts, the other side's, lead3) of real snaps, as the sim reads
    its own (timeouts not known: three each)."""
    get = lambda k: np.array([r[k] for r in recs])
    tos = np.array([r.get("timeouts_left") or (TIMEOUTS, TIMEOUTS) for r in recs], np.int64).reshape(-1, 2)
    mb = margin_b(get("margin"))
    return (get("period"), get("clock").astype(float), running_of(get("prev_cls")), tos[:, 0], tos[:, 1],
            derive(dict(margin=mb))["lead3"])


def _features(recs, model):
    """{feature: array} for real snaps (snaps left in the half off the model's table, for each snap's
    own state)."""
    get = lambda k: np.array([r[k] for r in recs])
    period, clock = get("period"), get("clock").astype(float)
    return derive(dict(prev=get("prev_cls"), phase=phase_of(period, clock),
                       margin=margin_b(get("margin")),
                       down=down_b(get("down")), dist=dist_b(get("distance")), zone=zone_b(get("field")),
                       elapsed=elapsed_b(get("elapsed")), remain=remain_b(model.remain_at(*remain_inputs(recs))),
                       tb=time_b(get("play_seconds"))))


def _cols(feats, names):
    return np.stack([feats[f] for f in names], axis=1)


def match_people(code, rows, hist):
    """((home gamer, away gamer), (home team, away team)) off the history, else the export."""
    h = hist.get(code, {})
    r0 = rows[0] if rows else {}
    pair = ((h.get("PLAYER_1_HANDLE") or r0.get("home_handle") or "").strip().upper(),
            (h.get("PLAYER_2_HANDLE") or r0.get("away_handle") or "").strip().upper())
    teams = ((h.get("PLAYER_1_TEAM") or "").strip(), (h.get("PLAYER_2_TEAM") or "").strip())
    return pair, teams


def build(matches, history=None, mode="full", min_records=MIN_RECORDS, verbose=True, refine=REFINE_REMAINING,
          seed=0, gamer_decisions=True, rules=True):
    """The model off the export's matches ({code: rows}) and the history (gamers and teams).
    gamer_decisions=False: every gamer decides as the league does in the same state; rules=False: no
    global rules, the real shares alone."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    m = Model()
    m.mode = mode
    m.use_rules = bool(rules and RULES)
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    recs, kick_dec, safeties = [], [], []
    dec = {k: [] for k in DECISIONS}
    dec.update(kick_pool=[], fg_clock=[], conv=[])
    for code, rows in matches.items():
        rows = _with_side(rows)
        pair, teams = match_people(code, rows, hist)
        mine = match_snaps(rows, pair, teams)
        recs += mine
        got = decision_points(rows, mine)
        for name in DECISIONS:
            dec[name] += [dict(q, gamer=pair[q["side"]], code=code)
                          for q, k in zip(got[name], in_draw_window(name, got[name])) if k]
        for name in ("kick_pool", "fg_clock", "conv"):
            dec[name] += got[name]
        kick_dec += s14.kick_decisions(rows)
        safeties += s14.safety_kicks(rows)
    recs = [r for r in recs if r["period"] <= 4 + MAX_OT]
    if not recs:
        raise ValueError("no real snaps to build on")
    if verbose:
        print(f"  v2.1: {len(recs):,} real snaps from {len(matches):,} matches ({mode})", flush=True)
    for r in recs:
        for g in r["gamer"]:
            if g and g not in m.gamers:
                m.gamers[g] = len(m.gamers)
        for t in r["team"]:
            if t and t not in m.teams:
                m.teams[t] = len(m.teams)

    # snaps left in the half, from the real games first
    m.remain = remain_table(remain_key(*remain_inputs(recs)), np.array([r["remain"] for r in recs], float))
    m.share = points_share(matches)

    # positive gains' survival, touchdowns counted as at least the yards the field allowed
    g = np.array([r["gain"] for r in recs if r["kind"] == GAIN and r["gain"] > 0], np.int64)
    td = np.array([r["gain"] >= 100 - r["field"] for r in recs if r["kind"] == GAIN and r["gain"] > 0], bool)
    if len(g):
        at_risk = np.bincount(np.minimum(g, 100), minlength=101)[::-1].cumsum()[::-1]
        ended = np.bincount(np.minimum(g[~td], 100), minlength=101)
        hazard = np.where(at_risk > 0, ended / np.maximum(at_risk, 1), 0.0)
        m.gain_surv = np.concatenate([[1.0], np.cumprod(1 - hazard[:-1])])
        m.gain_surv[:2] = 1.0

    _build_draws(m, recs, min_records, verbose)

    # the decisions: the rules learned off the real ones (where an option never or all but never happens),
    # then each gamer's lean on top of the league's choices in the same states
    from . import v21_rules
    m.rules = v21_rules.learn(dec) if m.use_rules else {}
    n_who = len(m.gamers) + 1
    for name, spec in DECISIONS.items():
        got = dec[name]
        who = np.array([m.gamer_index(q["gamer"]) for q in got], np.int64)
        chosen = np.array([q["choice"] for q in got], np.int64)
        X = decision_features(name, got)
        allowed = v21_rules.mask(m.rules, name, v21_rules.fine(name, got)) if got else None
        m.choices[name] = Choice.build(name, X, chosen, who, n_who, gamers=gamer_decisions, allowed=allowed)
    # kick-offs' outcomes: the field and clock of the returned and the kept ones, by desperate and late
    pool = np.array(dec["kick_pool"], float).reshape(-1, 5)
    for kind in (RETURN, KEPT):
        for desperate in (0, 1):
            for late in (0, 1):
                wants = [(pool[:, 0] == kind) & (pool[:, 1] == desperate) & (pool[:, 2] == late),
                         (pool[:, 0] == kind) & (pool[:, 1] == desperate), pool[:, 0] == kind]
                pick = next((w for w in wants if w.sum() >= KICK_POOL_MIN), None)
                if pick is None:
                    pick = next((w for w in wants if w.any()), None)
                m.kick_pool[(kind, desperate, late)] = (pool[pick, 3].astype(np.int64), pool[pick, 4]) \
                    if pick is not None else (np.array([25 if kind == RETURN else 45]), np.array([5.0]))
    # field goals: made by distance, and the clock they take
    fgs = [d for d in kick_dec if d[0] == "fg"]
    dist_fg = np.arange(15, FG_MAX + 5, 5)
    made, tried = np.zeros(len(dist_fg)), np.zeros(len(dist_fg))
    for _, y, ok, secs, _ in fgs:
        k = min(len(dist_fg) - 1, max(0, int((100 - y + 17 - 15) // 5)))
        made[k] += bool(ok)
        tried[k] += 1
    league = made.sum() / max(1, tried.sum())
    m.fg_make = (made + 10 * league) / (tried + 10)
    clock_used = np.array(dec["fg_clock"], float).reshape(-1, 2)
    near = clock_used[:, 0] <= FG_NEAR
    if near.any():
        m.fg_near = clock_used[near]
    if (~near).any():
        m.fg_far = clock_used[~near, 1]
    punts = [d for d in kick_dec if d[0] == "punt"]
    if punts:
        P = np.array([(d[1], d[2], d[3]) for d in punts], float)
        m.punts = Bins.build(("ybin",), (5,), (("ybin",), ()), np.clip((P[:, 0] // 20).astype(int), 0, 4)[:, None],
                             P[:, 1], {"net": P[:, 1].astype(np.int64), "seconds": P[:, 2]}, 20)
    if safeties:
        m.safety_fields = np.array(safeties, np.int64)
    conv = dec["conv"]
    if conv:
        two = [g for t, g in conv if t]
        kick = [g for t, g in conv if not t]
        m.two_good = (sum(two) + 5) / (len(two) + 10)
        m.kick_good = (sum(kick) + 9) / (len(kick) + 10)
    run = dec["to_def"]
    m.meta = dict(matches=len(matches), snaps=len(recs), mode=mode, timeouts=bool(run),
                  built=dt.datetime.now().isoformat(timespec="seconds"))
    if refine:
        for r_ in range(REFINE_ROUNDS):
            before = m.remain.copy()
            refine_remaining(m, seed=seed + r_)
            _build_draws(m, recs, min_records, verbose=False)
            if verbose:
                print(f"  snaps left in the half, round {r_ + 1}: the sim's own counts, mean change"
                      f" {np.abs(m.remain - before).mean():.2f} snaps; the draws rebuilt on them", flush=True)
    if verbose:
        report_build(m, recs)
    return m


def _build_draws(m, recs, min_records=MIN_RECORDS, verbose=True):
    """The time and distance draws (and the gamers' and teams' own or their tilts) on the model's
    current table of snaps left in the half."""
    feats = _features(recs, m)
    seconds = np.array([min(MAX_SECONDS, r["seconds"]) for r in recs], float)
    cut = np.array([r["cut"] for r in recs], bool)
    time_pay = {"seconds": seconds.astype(np.float32), "cut": cut}       # compact: a record sits in a group
                                                                           # at every level that draws on it
    m.time = Bins.build(TIME_FEATURES, [SIZES[f] for f in TIME_FEATURES], TIME_LEVELS,
                        _cols(feats, TIME_FEATURES), seconds, time_pay, min_records)
    kind = np.array([r["kind"] for r in recs])
    gain = np.array([r["gain"] for r in recs], np.int64)
    field = np.array([r["field"] for r in recs], np.int64)
    shift = np.array([_turnover_shift(r) if r["kind"] == TURNOVER else 0 for r in recs], np.int64)
    dvalue = np.where(kind == DEF_TD, DEF_TD_VALUE, np.where(kind == TURNOVER, TO_VALUE, gain)).astype(float)
    made = np.array([2 if r["replay"] else int(r["converted"]) if r["kind"] == GAIN else 3 for r in recs], np.int8)
    dist_pay = {"made": made, "rdist": np.array([r["distance"] for r in recs], np.int16),
                "kind": kind.astype(np.int8), "gain": gain.astype(np.int16), "shift": shift.astype(np.int16),
                "td": (kind == GAIN) & (gain >= 100 - field), "value": dvalue.astype(np.float32)}
    m.dist = Bins.build(DIST_FEATURES, [SIZES[f] for f in DIST_FEATURES], DIST_LEVELS,
                        _cols(feats, DIST_FEATURES), dvalue, dist_pay, min_records)

    who = dict(gamer_off=np.array([m.gamer_index(r["gamer"][0]) for r in recs]),
               gamer_def=np.array([m.gamer_index(r["gamer"][1]) for r in recs]),
               team_off=np.array([m.team_index(r["team"][0]) for r in recs]),
               team_def=np.array([m.team_index(r["team"][1]) for r in recs]))
    sizes = dict(gamer_off=len(m.gamers) + 1, gamer_def=len(m.gamers) + 1,
                 team_off=len(m.teams) + 1, team_def=len(m.teams) + 1)
    m.own, m.tilt = {}, {}
    if m.mode == "full":
        for name, ids in who.items():
            kind_, role = name.split("_")
            f = dict(feats, who=ids)
            for draw, features, levels, value, pay in (
                    ("time", GAMER_TIME_FEATURES, GAMER_TIME_LEVELS, seconds, time_pay),
                    ("dist", GAMER_DIST_FEATURES, GAMER_DIST_LEVELS, dvalue, dist_pay)):
                m.own[(kind_, role, draw)] = Bins.build(
                    features, [sizes[name]] + [SIZES[x] for x in features[1:]], levels, _cols(f, features),
                    value, pay, MIN_GAMER_RECORDS, strict=True)
    else:
        timed = ~cut
        b_time = m.time.bin_of({f: feats[f][timed] for f in TIME_FEATURES})
        m.tilt["time"] = fit_tilts(m.time, m.time.payload["seconds"], b_time, seconds[timed],
                                   {k: v[timed] for k, v in who.items()}, sizes)
        m.tilt["dist"] = fit_tilts(m.dist, m.dist.payload["value"], m.dist.bin_of(feats), dvalue, who, sizes)
    m._real_bins = (m.time.bin_of(feats), m.dist.bin_of(feats))


def fit_tilts(bins, values, bin_ids, observed, who, sizes, grid=ADJUST_GRID, prior=ADJUST_PRIOR,
              sweeps=ADJUST_SWEEPS):
    """{effect: log tilt per entity}: each gamer's and team's lean of the league's draws, fitted in
    turn (each with the others held) so the snaps' mean under the summed tilt meets the real one,
    shrunk toward none by `prior` snaps."""
    E = bins.tilted_means(values, grid)
    keep = bin_ids >= 0
    bin_ids, observed = bin_ids[keep], observed[keep]
    who = {k: v[keep] for k, v in who.items()}
    step = grid[1] - grid[0]
    effects = {k: np.zeros(sizes[k]) for k in who}
    for _ in range(sweeps):
        for name, ent in who.items():
            others = sum(effects[k][who[k]] for k in who if k != name)
            loss = np.zeros((sizes[name], len(grid)))
            for g, a in enumerate(grid):
                gi = np.clip(np.round((others + a - grid[0]) / step).astype(int), 0, len(grid) - 1)
                loss[:, g] = np.bincount(ent, (observed - E[bin_ids, gi]) ** 2, minlength=sizes[name])
            n = np.bincount(ent, minlength=sizes[name])
            best = grid[np.argmin(loss, axis=1)]
            effects[name] = np.where(n > 0, best * n / (n + prior), 0.0)
            effects[name][-1] = 0.0                       # the unknown gamer or team: none
    return effects


def report_build(m, recs=None):
    """The build's lines: the draws' groups and how deep the real snaps' own states fall back."""
    real = getattr(m, "_real_bins", None)
    for i, (name, b) in enumerate((("time", m.time), ("distance", m.dist))):
        share = b.level_shares(real[i]) if real is not None else np.bincount(b.level_of, minlength=len(b.levels)) / max(1, len(b.level_of))
        print(f"  {name} draw: {len(b.count):,} groups; real snaps' states by the features they draw on: "
              + ", ".join(f"{len(b.levels[k])} {100 * share[k]:.0f}%" for k in range(len(share)) if share[k] >= 0.005))
    if m.mode == "full":
        for kind_ in ("gamer", "team"):
            for role in ("off", "def"):
                t, d = m.own.get((kind_, role, "time")), m.own.get((kind_, role, "dist"))
                if t is not None:
                    print(f"  {kind_}s' own ({'offense' if role == 'off' else 'allowed'}): {len(t.count):,} time groups,"
                          f" {len(d.count):,} distance groups")
    else:
        for kind_ in ("time", "dist"):
            for k, v in m.tilt[kind_].items():
                if len(v) > 1:
                    print(f"  {kind_} tilt by {k.replace('_', ' ')}: sd {np.std(v[:-1]):.3f}, from {v[:-1].min():+.2f}"
                          f" to {v[:-1].max():+.2f}")
    print(f"  timeouts: {'fitted from the export' if m.meta.get('timeouts') else 'none (no timeout columns)'}")
    for line in remain_lines(m):
        print(line)
    if recs is not None:
        for line in chains_lines(recs):
            print(line)
    names = {i: g for g, i in m.gamers.items()}
    if m.use_rules:
        print(f"  rules learned off the real decisions (v21_rules: never or all but never taken): "
              + ", ".join(f"{DECISIONS[k]['label'].split(' (')[0]} {len(v)}" for k, v in m.rules.items()))
    else:
        print("  no rules: the real shares alone")
    for name, ch in m.choices.items():
        spec = DECISIONS[name]
        opt, need = LEAN_SHOWN[name]
        print(f"  {spec['label']}: {ch.n:,} real, "
              + ", ".join(f"{o} {100 * p:.1f}%" for o, p in zip(spec["options"], ch.league)))
        if ch.ruled:
            print(f"    {ch.ruled:,} real ones under a rule, {ch.broke:,} took a struck-out option anyway"
                  f" (left out of the gamers' leans)")
        n_g = ch.counts.sum(1)
        ok = np.flatnonzero(n_g[:-1] >= need)
        if len(ok) and ch.lean[ok].any() and ch.league[opt] > 0:
            rate = {i: ch.gamer_rate(i, opt) for i in ok}
            order = sorted(ok, key=lambda i: rate[i])
            show = lambda ids: ", ".join(f"{names[i]} {100 * rate[i]:.1f}% ({int(n_g[i])})" for i in ids)
            print(f"    {spec['options'][opt]}: each gamer's rate over every real state (the league's"
                  f" {100 * ch.gamer_rate(len(m.gamers), opt):.1f}%), {len(ok)} gamers with {need}+:"
                  f" least {show(order[:3])}; most {show(order[::-1][:3])}")


# the snaps-left table at a few states: (label, period, clock, running, own timeouts, the other side's)
REMAIN_SHOWN = (("Q2 2:00, clock stopped, 3 timeouts each", 2, 120.0, 0, 3, 3),
                ("Q2 0:40, clock stopped, 3 timeouts each", 2, 40.0, 0, 3, 3),
                ("Q2 0:40, clock running, no timeouts", 2, 40.0, 1, 0, 0),
                ("Q4 2:00, clock stopped, 3 timeouts each", 4, 120.0, 0, 3, 3),
                ("Q4 0:40, clock stopped, 3 timeouts each", 4, 40.0, 0, 3, 3),
                ("Q4 0:40, clock running, none for the ball, 3 the other side", 4, 40.0, 1, 0, 3),
                ("Q4 0:40, clock running, no timeouts", 4, 40.0, 1, 0, 0))


def remain_lines(m):
    """The snaps-left table at REMAIN_SHOWN's states, behind / level / ahead."""
    out = ["  snaps left in the half at a few states (behind / level / ahead; buckets"
           " 0-1, 2, 3-4, 5-7, 8-12, 13+):"]
    for label, p, c, run, own_, opp_ in REMAIN_SHOWN:
        v = [float(m.remain_at(p, c, run, own_, opp_, l3)) for l3 in range(3)]
        out.append(f"    {label}: " + " / ".join(f"{x:.1f}" for x in v))
    return out


def chains_lines(recs):
    """How the real snaps' new sets of downs line up with the yards: spots are whole yards, so a gain
    short of the distance can move the chains (inches) and one of the distance can fall short; and the
    downs played again (a penalty, the gain short of the line)."""
    g = [r for r in recs if r["kind"] == GAIN]
    if not g:
        return []
    out = [f"  first downs off the next row's down: {len(g):,} real gains, "
           f"{100 * np.mean([r['replay'] for r in g]):.1f}% the same down played again"]
    for lo, hi, label in ((1, 1, "1 to go"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 99, "7+")):
        s = [r for r in g if lo <= r["distance"] <= hi and not r["replay"] and r["gain"] < 100 - r["field"]]
        if not s:
            continue
        yards = np.array([r["gain"] >= r["distance"] for r in s])
        conv = np.array([r["converted"] for r in s])
        out.append(f"    {label}: {len(s):,} snaps, first down {100 * conv.mean():.1f}% (by the yards"
                   f" {100 * yards.mean():.1f}%); short of the yards yet moved the chains {(conv & ~yards).sum():,},"
                   f" the yards yet not {(~conv & yards).sum():,}")
    return out


# ---------------------------------------------------------------- the starts

def _start(model, n, pair, teams):
    return dict(gamer=np.tile([model.gamer_index(pair[0]), model.gamer_index(pair[1])], (n, 1)),
                team_ix=np.tile([model.team_index(teams[0]), model.team_index(teams[1])], (n, 1)))


def kickoff_start(model, pair=("", ""), teams=("", "")):
    """The two kick-off states (each side receiving), with the gamers and teams."""
    return dict(period=np.array([1, 1]), clock=np.array([QUARTER, QUARTER]), team=np.array([0, 1]),
                phase=np.array([KICK, KICK]), down=np.array([1, 1]), dist=np.array([10, 10]),
                y=np.array([25, 25]), home=np.zeros(2, np.int64), away=np.zeros(2, np.int64),
                prev=np.array([FRESH, FRESH]), elapsed=np.zeros(2, np.int64),
                tos=np.full((2, 2), TIMEOUTS), kick2=np.array([1, 0]), **_start(model, 2, pair, teams))


def match_starts(model, rows, pair=("", ""), teams=("", ""), every=1):
    """[(row index, state)] at every `every`-th PLAY_OVER before a scrimmage snap, with what the action
    log says up to it: the clock's state off the play before (a fresh possession, a new quarter: stopped;
    otherwise the play's class), the snaps so far and the timeouts used. The state is a start dict of
    one."""
    rows = _with_side(rows)
    if not rows:
        return []
    a_home = rows[0]["team_a_side"] != "away"
    recs = match_snaps(rows)
    by_end = {r["message"]: r for r in recs}
    starts = sorted(r["start"] for r in recs)
    opening = rows[0].get("opening_offense")
    first = _side_of(opening, a_home) if opening in ("TEAM_A", "TEAM_B") else 0
    out, seen = [], 0
    for i, r in enumerate(rows):
        if r["play_kind"] not in s14.SNAP_KINDS or not r.get("down") or not r.get("field_position") \
                or not r.get("period") or not r.get("clock_seconds") or not r.get("distance") \
                or r.get("offense") not in ("TEAM_A", "TEAM_B"):
            continue
        period, clock = s14._i(r["period"]), s14._f(r["clock_seconds"])
        if period is None or clock is None or clock <= 0 or period > 4 + MAX_OT:
            continue
        seen += 1
        if (seen - 1) % every:
            continue
        msg = s14._i(r["message"])
        last = by_end.get(msg)
        if s14.is_fresh(r) or last is None or last["offense"] != r["offense"] or last["period"] != period \
                or last["kind"] != GAIN or last["gain"] >= 100 - last["field"]:
            prev = FRESH
        else:
            prev = int(play_class(last["gain"], last["distance"]))
        used = (s14._i(r.get("timeouts_used_a")), s14._i(r.get("timeouts_used_b")))
        have = TIMEOUTS if period <= 4 else OT_TIMEOUTS
        tos_a = [max(0, have - u) if u is not None else have for u in used]
        tos = tos_a if a_home else tos_a[::-1]
        home, away = s14._i(r["score_p1"]) or 0, s14._i(r["score_p2"]) or 0
        out.append((i, dict(period=np.array([period]), clock=np.array([clock]),
                            team=np.array([_side_of(r["offense"], a_home)]), phase=np.array([SNAP]),
                            down=np.array([s14._i(r["down"])]), dist=np.array([max(1, s14._i(r["distance"]) or 10)]),
                            y=np.array([s14._i(r["field_position"])]), home=np.array([home]),
                            away=np.array([away]), prev=np.array([prev]),
                            elapsed=np.array([bisect_left(starts, msg)]), tos=np.array([tos]),
                            kick2=np.array([1 - first]), **_start(model, 1, pair, teams))))
    return out


def concat_starts(starts):
    return {k: np.concatenate([s[k] for s in starts]) for k in starts[0]}


def _tilts(model, kind, gamer, team_ix):
    """(paths, 2) each side's tilt with the ball (adjusted builds; None otherwise)."""
    if model.mode != "adjusted" or kind not in model.tilt:
        return None
    t = model.tilt[kind]
    out = np.zeros(gamer.shape)
    for s in (0, 1):
        out[:, s] = (t["gamer_off"][gamer[:, s]] + t["gamer_def"][gamer[:, 1 - s]]
                     + t["team_off"][team_ix[:, s]] + t["team_def"][team_ix[:, 1 - s]])
    return np.exp(out)


# ---------------------------------------------------------------- the simulation

# a snap as the census reads it: its clock and the play before as at its PLAY_OVER, the clock it used to
# the next (seconds) and the time its distance was drawn on (drawn)
SNAP_LOG = ("path", "period", "clock", "prev", "seconds", "drawn", "kind", "gain", "dist", "field", "side", "down",
            "lead", "conv", "rem", "to_own", "to_opp")


def simulate(model, start, n_paths, seed=0, stats=None, max_steps=MAX_STEPS):
    """Play every start n_paths times to the end: (home, away) final scores shaped (starts, paths).
    `stats` (a dict) gathers the census: every snap (SNAP_LOG), counts and points by quarter."""
    rng = np.random.default_rng(seed)
    S = len(start["period"])
    P = S * n_paths
    rep = lambda a, t: np.repeat(np.asarray(a), n_paths, axis=0).astype(t)
    period, clock, team = rep(start["period"], np.int64), rep(start["clock"], float), rep(start["team"], np.int64)
    phase, down, dist, y = (rep(start[k], np.int64) for k in ("phase", "down", "dist", "y"))
    score = np.stack([rep(start["home"], np.int64), rep(start["away"], np.int64)], axis=1)
    prev, elapsed, tos = rep(start["prev"], np.int64), rep(start["elapsed"], np.int64), rep(start["tos"], np.int64)
    kick2, gamer, team_ix = rep(start["kick2"], np.int64), rep(start["gamer"], np.int64), rep(start["team_ix"], np.int64)
    k_time, k_dist = _tilts(model, "time", gamer, team_ix), _tilts(model, "dist", gamer, team_ix)
    ot_cur = np.full(P, -1, np.int64)
    ot_done = np.zeros((P, 2), bool)
    ahead = (period >= 5) & (score[:, 0] != score[:, 1])
    ot_done[ahead, np.where(score[ahead, 0] > score[ahead, 1], 0, 1)] = True
    st = stats
    tally = (lambda k, v: st.__setitem__(k, st.get(k, 0) + int(v))) if st is not None else (lambda k, v: None)
    if st is not None:
        st.setdefault("_snaps", [])
        st.setdefault("_dec", [])
        q_points = np.zeros((P, 5), np.int64)
    q_score = score.copy()

    from . import v21_rules
    use_rules = RULES and getattr(model, "use_rules", True)

    def state(ix, side):
        """The full state of a decision by `side` on each path (v21_rules.SWEEP_FEATURES)."""
        return dict(period=np.minimum(period[ix], 5), left=np.round(half_left(period[ix], clock[ix])).astype(np.int64),
                    lead=score[ix, side] - score[ix, 1 - side], togo=100 - y[ix], dist=dist[ix], down=down[ix],
                    own_to=tos[ix, side], opp_to=tos[ix, 1 - side], prev=prev[ix])

    def decide(name, cols, who, fine=None, ot=None, ix=None):
        """A decision for each path: the options the rules strike out in its full state (`fine`) and
        overtime's own (`ot`) taken away, the rest drawn from the real ones in the state, leaned by the
        gamer deciding."""
        learned = v21_rules.mask(model.rules, name, fine) if use_rules and fine is not None else None
        k = model.choices[name].draw(cols, who, rng.random(len(who)), combine(learned, ot))
        if st is not None:
            st["_dec"].append((name, who.copy(), k.copy(), decision_slice(name, cols), None if ix is None else ix.copy()))
        return k

    def kickoff(ix):
        recv = team[ix]
        kicker = 1 - recv
        lead = score[ix, kicker] - score[ix, recv]
        desperate = ((period[ix] >= 4) & (lead < 0) & (half_left(period[ix], clock[ix]) <= 180)).astype(np.int64)
        late = (((period[ix] == 2) | (period[ix] == 4)) & (clock[ix] <= LATE)).astype(np.int64)
        kind = decide("kickoff", dict(desperate=desperate, late=late, lead=lead3(lead)), gamer[ix, kicker],
                      fine=state(ix, kicker), ix=ix)
        tally("kickoffs", len(ix))
        for k_, spot in ((TOUCHBACK, KICK_TOUCHBACK), (NO_LANDING, KICK_NO_LANDING)):
            sel = kind == k_
            if sel.any():
                set_ball(ix[sel], recv[sel], np.full(int(sel.sum()), spot))
        for k_ in (RETURN, KEPT):
            s_ = np.flatnonzero(kind == k_)
            if not len(s_):
                continue
            field, secs = np.empty(len(s_), np.int64), np.empty(len(s_))
            for d in (0, 1):
                for l_ in (0, 1):
                    q = (desperate[s_] == d) & (late[s_] == l_)
                    if q.any():
                        pf, ps = model.kick_pool[(k_, d, l_)]
                        j = rng.integers(0, len(pf), int(q.sum()))
                        field[q], secs[q] = pf[j], ps[j]
            sel = ix[s_]
            clock[sel] = np.maximum(0.0, clock[sel] - secs)
            side = kicker[s_] if k_ == KEPT else recv[s_]
            if k_ == KEPT:
                tally("onside_kept", len(s_))
            td = field >= 100
            if td.any():
                tally("kick_tds", td.sum())
                touchdown(sel[td], side[td])
            set_ball(sel[~td], side[~td], field[~td])

    def set_ball(ix, side, field):
        team[ix] = side
        y[ix] = np.clip(field, 1, 99)
        down[ix] = 1
        dist[ix] = np.minimum(10, 100 - y[ix])
        prev[ix] = FRESH
        phase[ix] = SNAP

    def touchdown(ix, side):
        score[ix, side] += 6
        tally("touchdowns", len(ix))
        lead = score[ix, side] - score[ix, 1 - side]
        ot = period[ix] >= 5
        walk = ot & ot_done[ix, 1 - side] & (lead > 0)            # the second side goes ahead: over
        if walk.any():
            phase[ix[walk]] = DONE
            tally("ot_walk_off", walk.sum())
        ix, side, lead, ot = ix[~walk], side[~walk], lead[~walk], ot[~walk]
        if not len(ix):
            return
        p, c = period[ix], clock[ix]
        cell = np.where(p <= 2, 0, np.where(p == 3, 1, np.where(ot | (c <= 180), 3, 2)))
        if CONV_CLOCK:
            cell = np.where(ot, CONV_CELLS - 1, cell)
            for i, edge in reversed(list(enumerate(CONV_CLOCK))):
                cell = np.where((p == 4) & (c <= edge), 4 + i, cell)
        close = ot & ot_done[ix, 1 - side] & ((lead == -1) | (lead == -2))
        two = decide("two", dict(cell=cell, lead=np.clip(lead, -CONV_MARGIN, CONV_MARGIN) + CONV_MARGIN),
                     gamer[ix, side], fine=state(ix, side), ot=ot_mask("two", len(ix), other_done_close=close),
                     ix=ix) == 1
        good = rng.random(len(ix)) < np.where(two, model.two_good, model.kick_good)
        score[ix, side] += np.where(good, np.where(two, 2, 1), 0)
        tally("two_point_tries", two.sum())
        kick_to(ix, 1 - side)

    def kick_to(ix, receiver):
        team[ix] = receiver
        phase[ix] = KICK

    def field_goal(ix):
        """A field goal try: made by its length; the clock it takes as the real kicks' did (a kick with
        FG_NEAR or less left in the half from those, one that ran the clock out running this one out)."""
        tally("field_goal_tries", len(ix))
        kd = 100 - y[ix] + 17
        made = rng.random(len(ix)) < model.fg_make[np.clip((kd - 15) // 5, 0, len(model.fg_make) - 1)]
        near = half_left(period[ix], clock[ix]) <= FG_NEAR
        used = np.empty(len(ix))
        if near.any():
            j = rng.integers(0, len(model.fg_near), int(near.sum()))
            left0, used0 = model.fg_near[j, 0], model.fg_near[j, 1]
            used[near] = np.where(used0 >= left0 - 0.5, np.inf, used0)
        if (~near).any():
            used[~near] = model.fg_far[rng.integers(0, len(model.fg_far), int((~near).sum()))]
        clock[ix] = np.maximum(0.0, clock[ix] - used)
        tally("field_goals", made.sum())
        mk, ms = ix[made], ix[~made]
        score[mk, team[mk]] += 3
        kick_to(mk, 1 - team[mk])
        set_ball(ms, 1 - team[ms], np.maximum(20, 100 - y[ms]))

    def fourth(ix):
        off = team[ix]
        cols = derive(dict(togo10=togo10(y[ix]), dist=dist_b(dist[ix]), margin=margin_b(score[ix, off] - score[ix, 1 - off]),
                           phase=phase_of(period[ix], clock[ix]), left=left_b(half_left(period[ix], clock[ix]))))
        cols = {k: v for k, v in cols.items() if k in DECISIONS["fourth"]["features"]}
        lead = score[ix, off] - score[ix, 1 - off]
        must = (period[ix] >= 5) & ot_done[ix, 1 - off] & (lead < 0)
        choice = decide("fourth", cols, gamer[ix, off], fine=state(ix, off),
                        ot=ot_mask("fourth", len(ix), (must, lead <= -4)), ix=ix)
        choice = np.where((choice == FG) & (100 - y[ix] + 17 > FG_MAX), np.where(dist[ix] <= 2, GO, PUNT), choice)
        pu, fg = ix[choice == PUNT], ix[choice == FG]
        if len(pu):
            tally("punts", len(pu))
            if model.punts is not None:
                j = model.punts.draw(model.punts.bin_of(dict(ybin=np.clip(y[pu] // 20, 0, 4))), rng.random(len(pu)))
                net, secs = model.punts.payload["net"][j], model.punts.payload["seconds"][j]
            else:
                net, secs = np.full(len(pu), 40), np.full(len(pu), 8.0)
            clock[pu] = np.maximum(0.0, clock[pu] - secs)
            land = y[pu] + net
            set_ball(pu, 1 - team[pu], np.where(land >= 100, 20, 100 - land))
        if len(fg):
            field_goal(fg)
        return ix[choice == GO]

    layers = [None] + sorted(model.own) if model.mode == "full" else [None]

    def own(draw, f, ix, off, u, fields, out, stopped=False):
        """full: half the draws from the offense's own real snaps, half from what the defense
        allowed, each its gamer's where it has them, else its team's (else the league's, as drawn).
        out["_from"] and out["_j"] keep which draw (0 the league's) and record each came from."""
        if model.mode != "full" or not model.own:
            return
        allowed = rng.random(len(ix)) < DEF_SHARE
        for role, rows in (("off", ~allowed), ("def", allowed)):
            sel = np.flatnonzero(rows)
            if not len(sel):
                continue
            side = off[sel] if role == "off" else 1 - off[sel]
            for kind_, who in (("team", team_ix), ("gamer", gamer)):
                layer = model.own.get((kind_, role, draw))
                if layer is None:
                    continue
                lb = layer.bin_of(dict({k: v[sel] for k, v in f.items()}, who=who[ix[sel], side]))
                ok = lb >= 0
                if ok.any():
                    jj = layer.draw_low(lb[ok], u[sel[ok]], "seconds", STOPPED_MAX) if stopped \
                        else layer.draw(lb[ok], u[sel[ok]])
                    for name in fields:
                        out[name][sel[ok]] = layer.payload[name][jj]
                    out["_from"][sel[ok]] = layers.index((kind_, role, draw))
                    out["_j"][sel[ok]] = jj

    def remain_of(ix, prv, tos0):
        """The snaps-left bucket of each path's snap: its clock, running or not (prv), the timeouts
        each side had before any was called for it (tos0: own, other's) and the lead."""
        off = team[ix]
        lead3 = derive(dict(margin=margin_b(score[ix, off] - score[ix, 1 - off])))["lead3"]
        return remain_b(model.remain_at(period[ix], clock[ix], running_of(prv), tos0[:, 0], tos0[:, 1], lead3))

    def draw_time(ix, prv, rem, stopped=False):
        """Each snap's time; `stopped`: from the stopped-clock part of the same draw (the snap after
        the two-minute warning)."""
        off = team[ix]
        f = derive(dict(prev=prv, phase=phase_of(period[ix], clock[ix]),
                        margin=margin_b(score[ix, off] - score[ix, 1 - off]), down=down_b(down[ix]),
                        dist=dist_b(dist[ix]), zone=zone_b(y[ix]), elapsed=elapsed_b(elapsed[ix]), remain=rem))
        u = rng.random(len(ix))
        k = k_time[ix, off] if k_time is not None else None
        b = model.time.bin_of(f)
        j = model.time.draw_low(b, u, "seconds", STOPPED_MAX) if stopped else model.time.draw(b, u, k)
        out = {"seconds": model.time.payload["seconds"][j].copy(), "cut": model.time.payload["cut"][j].copy(),
               "_from": np.zeros(len(ix), np.int64), "_j": j.copy()}
        own("time", f, ix, off, u, ("seconds", "cut"), out, stopped)
        secs, cut = out["seconds"], out["cut"]
        if stopped:
            return secs
        # a real snap the quarter's end or 2:00 cut short, drawn with more time to go than it took:
        # the draw moves on to a real snap at least as long in the same group
        limit = np.where(((period[ix] == 2) | (period[ix] == 4)) & (clock[ix] > LATE),
                         clock[ix] - LATE + PLAY_SECONDS, clock[ix])
        again = np.flatnonzero(cut & (secs < limit))
        if len(again):
            u2 = rng.random(len(again))
            for li in np.unique(out["_from"][again]):
                a = out["_from"][again] == li
                b = model.time if li == 0 else model.own[layers[li]]
                secs[again[a]] = b.payload["seconds"][b.above(out["_j"][again[a]], u2[a])]
        return secs

    def draw_play(ix, secs, clk, rem):
        """What came of each snap, off the state at the snap (clk: the clock then): (kind, gain, shift,
        a new set of downs, the down played again)."""
        off = team[ix]
        f = derive(dict(tb=time_b(secs), down=down_b(down[ix]), dist=dist_b(dist[ix]), zone=zone_b(y[ix]),
                        margin=margin_b(score[ix, off] - score[ix, 1 - off]), phase=phase_of(period[ix], clk),
                        elapsed=elapsed_b(elapsed[ix]), remain=rem))
        u = rng.random(len(ix))
        k = k_dist[ix, off] if k_dist is not None else None
        j = model.dist.draw(model.dist.bin_of(f), u, k)
        names = ("kind", "gain", "shift", "td", "made", "rdist")
        out = {x: model.dist.payload[x][j].copy() for x in names}
        out["_from"], out["_j"] = np.zeros(len(ix), np.int64), j.copy()
        own("dist", f, ix, off, u, names, out)
        kind, gain, shift, td, made, rdist = (out[x] for x in names)
        if SMOOTH_YARDS:
            spread = (kind == GAIN) & (gain != 0) & ~td
            gain = np.where(spread, gain + rng.integers(-SMOOTH_YARDS, SMOOTH_YARDS + 1, len(ix)), gain)
        # a real touchdown drawn from further back runs on with the real chance a gain that long goes on
        need = 100 - y[ix]
        on = np.flatnonzero(td & (gain > 0) & (gain < need))
        if len(on):
            S_ = model.gain_surv
            g0 = np.clip(gain[on], 0, 100)
            t = rng.random(len(on)) * S_[g0]
            reach = np.searchsorted(-S_, -t, side="right") - 1          # the longest gain with S >= t
            gain[on] = np.maximum(g0, reach)
        # a new set of downs: the real snap's own, where it was to go as far (spots are whole yards: a
        # sneak for inches shows as 0 and moves the chains); else by the yards
        same = (made <= 1) & (rdist == dist[ix])
        conv = np.where(same, made == 1, gain >= dist[ix])
        return kind, gain, shift, conv, made == 2

    def play(ix):
        off = team[ix]
        # timeouts: only before a snap off a running clock, and they move only the time draw
        running = prev[ix] <= LOSS
        cols = dict(phase=phase_of(period[ix], clock[ix]), margin=margin_b(score[ix, off] - score[ix, 1 - off]),
                    left=left_b(half_left(period[ix], clock[ix])))
        call_def, call_off = np.zeros(len(ix), bool), np.zeros(len(ix), bool)
        d_ = np.flatnonzero(running & (tos[ix, 1 - off] > 0))
        if len(d_):
            call_def[d_] = decide("to_def", {k: v[d_] for k, v in cols.items()}, gamer[ix[d_], 1 - off[d_]],
                                  fine=state(ix[d_], 1 - off[d_]), ix=ix[d_]) == 1
        o_ = np.flatnonzero(running & ~call_def & (tos[ix, off] > 0))
        if len(o_):
            call_off[o_] = decide("to_off", {k: v[o_] for k, v in cols.items()}, gamer[ix[o_], off[o_]],
                                  fine=state(ix[o_], off[o_]), ix=ix[o_]) == 1
        prv = prev[ix].copy()
        tos0 = np.stack([tos[ix, off], tos[ix, 1 - off]], axis=1)      # before any is called for this snap
        tos[ix[call_def], 1 - off[call_def]] -= 1
        tos[ix[call_off], off[call_off]] -= 1
        prv[call_def | call_off] = TIMEOUT
        tally("timeouts", (call_def | call_off).sum())
        rem = remain_of(ix, prv, tos0)
        secs = draw_time(ix, prv, rem)
        ran = secs > STOPPED_MAX
        # a running clock that runs out before the snap: the period is over, no snap
        gone = ran & (secs - PLAY_SECONDS >= clock[ix])
        if gone.any():
            clock[ix[gone]] = 0.0
            tally("clock_ran_out", gone.sum())
        keep = ~gone
        ix, secs, ran, prv, tos0, rem = ix[keep], secs[keep], ran[keep], prv[keep], tos0[keep], rem[keep]
        if not len(ix):
            return
        start_clock, start_prv = clock[ix].copy(), prv.copy()
        # the two-minute warning stops a running clock at 2:00; the snap then comes off a stopped clock
        warn = ran & ((period[ix] == 2) | (period[ix] == 4)) & (clock[ix] > LATE) \
            & (clock[ix] - (secs - PLAY_SECONDS) <= LATE)
        if warn.any():
            w = ix[warn]
            clock[w] = LATE
            rem[warn] = remain_of(w, np.full(len(w), TIMEOUT), tos0[warn])    # the warning stops the clock
            secs[warn] = draw_time(w, prv[warn], rem[warn], stopped=True)
            tally("two_minute", len(w))
        snap_clock = clock[ix].copy()
        clock[ix] = np.maximum(0.0, clock[ix] - secs)
        kind, gain, shift, conv, replay = draw_play(ix, secs, snap_clock, rem)
        off = team[ix]
        elapsed[ix] += 1
        if st is not None:                    # logged as a real snap reads: PLAY_OVER to PLAY_OVER
            got = (kind == GAIN) & (conv | (gain >= 100 - y[ix])) & ~replay
            st["_snaps"].append((ix.copy(), period[ix].copy(), start_clock, start_prv, start_clock - clock[ix],
                                 secs.copy(), kind.copy(), np.minimum(gain, 100 - y[ix]), dist[ix].copy(),
                                 y[ix].copy(), off.copy(), down[ix].copy(), score[ix, off] - score[ix, 1 - off],
                                 got, rem.copy(), tos0[:, 0].copy(), tos0[:, 1].copy()))
        tally("snaps", len(ix))
        to = kind == TURNOVER
        if to.any():
            t = ix[to]
            new = 100 - (y[t] + shift[to])
            turnover(t, 1 - team[t], np.where(new < 1, 20, new))
        dt_ = kind == DEF_TD
        if dt_.any():
            d = ix[dt_]
            tally("defensive_tds", len(d))
            touchdown(d, 1 - team[d])
        g = kind == GAIN
        gx, gg, gconv, grep = ix[g], gain[g], conv[g], replay[g]
        if not len(gx):
            return
        y2 = y[gx] + gg
        td = y2 >= 100
        if td.any():
            touchdown(gx[td], team[gx[td]])
        sf = y2 <= 0
        if sf.any():
            s = gx[sf]
            tally("safeties", len(s))
            score[s, 1 - team[s]] += 2
            set_ball(s, 1 - team[s], model.safety_fields[rng.integers(0, len(model.safety_fields), len(s))])
        on = ~td & ~sf
        ox, og, oy, oc, orp = gx[on], gg[on], y2[on], gconv[on], grep[on]
        first = oc & ~orp
        prev[ox] = np.where(first, FIRST, play_class(og, dist[ox]))
        y[ox] = oy
        f, nf = ox[first], ox[~first]
        down[f] = 1
        dist[f] = np.minimum(10, 100 - y[f])
        down[nf] += np.where(orp[~first], 0, 1)                    # a down played again keeps its number
        dist[nf] = np.maximum(1, dist[nf] - og[~first])
        failed = nf[down[nf] > 4]
        if len(failed):
            tally("turnovers_on_downs", len(failed))
            set_ball(failed, 1 - team[failed], 100 - y[failed])

    def turnover(ix, side, field):
        tally("turnovers", len(ix))
        set_ball(ix, side, field)

    def end_period(ix):
        p = period[ix]
        if st is not None:
            q_points[ix, np.minimum(p, 5) - 1] += (score[ix] - q_score[ix]).sum(1)
        q_score[ix] = score[ix]
        carry = (p == 1) | (p == 3)
        c = ix[carry]
        period[c] += 1
        clock[c] = QUARTER
        prev[c] = FRESH
        h = ix[p == 2]
        period[h] = 3
        clock[h] = QUARTER
        tos[h] = TIMEOUTS
        kick_to(h, kick2[h])
        reg = ix[p == 4]
        level = score[reg, 0] == score[reg, 1]
        phase[reg[~level]] = DONE
        go = reg[level]
        tally("overtime", len(go))
        period[go] = 5
        clock[go] = QUARTER
        tos[go] = OT_TIMEOUTS
        ot_done[go], ot_cur[go] = False, -1
        kick_to(go, rng.integers(0, 2, len(go)))
        ot = ix[p >= 5]
        behind = np.where(score[ot, 0] < score[ot, 1], 0, 1)
        decided = (score[ot, 0] != score[ot, 1]) & ot_done[ot, behind]
        stop = decided | (period[ot] >= 4 + MAX_OT)
        phase[ot[stop]] = DONE
        on = ot[~stop]
        period[on] += 1
        clock[on] = QUARTER
        tos[on] = OT_TIMEOUTS
        prev[on] = FRESH

    for _ in range(max_steps):
        live = np.flatnonzero(phase != DONE)
        if not len(live):
            break
        ot = live[period[live] >= 5]
        if len(ot):                                   # overtime: whose possessions are done
            sn_ = ot[phase[ot] == SNAP]
            ended = sn_[(ot_cur[sn_] >= 0) & (ot_cur[sn_] != team[sn_])]
            ot_done[ended, ot_cur[ended]] = True
            ot_cur[sn_] = team[sn_]
            kk = ot[phase[ot] == KICK]
            ended = kk[ot_cur[kk] >= 0]
            ot_done[ended, ot_cur[ended]] = True
            ot_cur[kk] = -1
            over = ot[ot_done[ot, 0] & ot_done[ot, 1] & (score[ot, 0] != score[ot, 1])]
            phase[over] = DONE
            tally("ot_decided", len(over))
            live = np.flatnonzero(phase != DONE)
        k = live[phase[live] == KICK]
        if len(k):
            kickoff(k)
        sn = live[phase[live] == SNAP]
        sn = sn[~np.isin(sn, k)]
        if AUTO_END and len(sn):
            # the game's own end: the side ahead in Q4 can run the clock out, the defense has no timeouts
            off = team[sn]
            over = auto_end(period[sn], clock[sn], score[sn, off] - score[sn, 1 - off], down[sn], tos[sn, 1 - off])
            if over.any():
                clock[sn[over]] = 0.0
                tally("auto_end", over.sum())
                sn = sn[~over]
        if len(sn):
            # downs 1-3 in range in the half's last minute: kick now or play on
            left = half_left(period[sn], clock[sn])
            kd = 100 - y[sn] + 17
            can = (down[sn] < 4) & (kd <= FG_MAX) & (left <= EARLY_LEFT) & ((period[sn] == 2) | (period[sn] >= 4))
            e = sn[can]
            if len(e):
                off = team[e]
                cols = dict(left=early_b(left[can]), range=range_b(kd[can]), down=down[e] - 1,
                            margin=margin_b(score[e, off] - score[e, 1 - off]),
                            half=np.where(period[e] == 2, 0, np.where(period[e] == 4, 1, 2)))
                lead = score[e, off] - score[e, 1 - off]
                must = (period[e] >= 5) & ot_done[e, 1 - off] & (lead < 0)
                now = e[decide("early_fg", cols, gamer[e, off], fine=state(e, off),
                               ot=ot_mask("early_fg", len(e), (must, lead <= -4)), ix=e) == 1]
                if len(now):
                    tally("early_field_goals", len(now))
                    field_goal(now)
                    sn = sn[~np.isin(sn, now)]
            on4 = sn[down[sn] >= 4]
            go = fourth(on4) if len(on4) else on4
            plays = np.concatenate([sn[down[sn] < 4], go])
            if len(plays):
                play(plays)
        ended = np.flatnonzero((phase != DONE) & (clock <= 0.0))
        if len(ended):
            end_period(ended)
    if st is not None:
        st["went_ot"] = (period >= 5)
        st["games"] = st.get("games", 0) + P
        st["final"] = score.copy()
        st["q_points"] = q_points
        st["unfinished"] = st.get("unfinished", 0) + int((phase != DONE).sum())
    return score[:, 0].reshape(S, n_paths), score[:, 1].reshape(S, n_paths)


def decision_census(stats, n_who):
    """{name: {"slices": (slices, options), "who": (gamers + 1, options)}} counts of a sim's decisions."""
    out = {}
    for name, who, k, sl, *_ in stats.get("_dec") or []:
        K, S_ = len(DECISIONS[name]["options"]), len(SLICE_NAMES[name])
        c = out.setdefault(name, {"slices": np.zeros((S_, K)), "who": np.zeros((n_who, K))})
        np.add.at(c["slices"], (sl, k), 1)
        np.add.at(c["who"], (who, k), 1)
    return out


def real_decision_census(matches, model, history=None):
    """decision_census's counts for the real decisions of these matches ({code: rows})."""
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    n_who = len(model.gamers) + 1
    out = {name: {"slices": np.zeros((len(SLICE_NAMES[name]), len(spec["options"]))),
                  "who": np.zeros((n_who, len(spec["options"])))} for name, spec in DECISIONS.items()}
    for code, rows in matches.items():
        pair, _ = match_people(code, rows, hist)
        got = match_decisions(rows)
        for name, spec in DECISIONS.items():
            if not got[name]:
                continue
            a = np.array([r[1:] for r in got[name]], np.int64)
            who = np.array([model.gamer_index(pair[r[0]]) for r in got[name]], np.int64)
            sl = decision_slice(name, {f: a[:, 1 + i] for i, f in enumerate(spec["features"])})
            np.add.at(out[name]["slices"], (sl, a[:, 0]), 1)
            np.add.at(out[name]["who"], (who, a[:, 0]), 1)
    return out


def snap_log(stats):
    """The census's snaps as {SNAP_LOG name: array}."""
    log = stats.get("_snaps") or []
    if not log:
        return {k: np.zeros(0) for k in SNAP_LOG}
    return {k: np.concatenate([x[i] for x in log]) for i, k in enumerate(SNAP_LOG)}


def refine_remaining(model, n_paths=REFINE_PATHS, seed=0):
    """Snaps left in the half from the sim's own games: kick-offs at league, each snap's count of the
    snaps after it in its half, by remain_key (the slice, the clock running or not, the timeouts, the
    lead), shrunk as remain_table shrinks; slices the sim never reached read off the table before."""
    stats = {}
    simulate(model, kickoff_start(model), max(1, n_paths // 2), seed=seed, stats=stats)
    log = snap_log(stats)
    if not len(log["path"]):
        return model.remain
    half, sl = remain_slice(log["period"], log["clock"])
    key = log["path"].astype(np.int64) * 3 + half
    order = np.argsort(key, kind="stable")                    # snaps are logged in order: stable keeps it
    k = key[order]
    first = np.r_[True, k[1:] != k[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(k)), 0))
    ordinal = np.arange(len(k)) - group_start
    size = np.bincount(k, minlength=int(k.max()) + 1)[k]
    rem = np.empty(len(k), float)
    rem[order] = size - ordinal - 1
    lead3 = derive(dict(margin=margin_b(log["lead"])))["lead3"]
    cells = remain_key(log["period"], log["clock"], running_of(log["prev"]), log["to_own"], log["to_opp"], lead3)
    model.remain = remain_table(cells, rem, base=model.remain)
    return model.remain
