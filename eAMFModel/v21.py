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

Snaps left in the half (REMAIN) is a table by the time left in the half: the real games' average
first, then the sim's own from kick-off (refine_remaining), and the draws are rebuilt on it.

The rest is drawn from real data too: 4th downs (go, field goal, punt by field, distance, lead and
part of the game), field goals made by distance, punts' net yards, kick-offs (onside when trailing
late), conversions (going for two by v14's cells and the lead), safety free kicks and timeouts (the
defense's and the offense's calls before a running-clock snap by part of the game and lead; a timeout
moves only the time draw). Overtime is v14's: over once both sides have had the ball and one leads, a
touchdown that puts the second side ahead ends it at once, and a period's end decides it only when the
side behind has had the ball.
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
ZONE_EDGES = np.array([10.5, 30.5, 50.5, 70.5, 89.5])
ELAPSED_EDGES = np.array([30.5, 60.5, 90.5])
REMAIN_EDGES = np.array([2.5, 6.5, 12.5])
TIME_EDGES = np.array([6.5, 9.5, 14.5, 24.5, 34.5])
TIME_BUCKETS = ("<=6s", "7-9s", "10-14s", "15-24s", "25-34s", "35s+")

SIZES = dict(prev=6, phase=7, margin=7, down=4, dist=5, zone=6, elapsed=4, remain=4, tb=6)
TIME_FEATURES = ("prev", "phase", "margin", "down", "dist", "zone", "elapsed", "remain")
DIST_FEATURES = ("tb", "down", "dist", "zone", "margin", "phase", "elapsed", "remain")


def _drop(features, *names):
    return tuple(f for f in features if f not in names)


TIME_LEVELS = (TIME_FEATURES,
               _drop(TIME_FEATURES, "remain"),
               _drop(TIME_FEATURES, "remain", "elapsed"),
               _drop(TIME_FEATURES, "remain", "elapsed", "zone"),
               ("prev", "phase", "margin", "down"),
               ("prev", "phase", "margin"),
               ("prev", "phase"),
               ("prev",),
               ())
DIST_LEVELS = (DIST_FEATURES,
               _drop(DIST_FEATURES, "remain"),
               _drop(DIST_FEATURES, "remain", "elapsed"),
               ("tb", "down", "dist", "zone", "margin"),
               ("tb", "down", "dist", "zone"),
               ("tb", "dist", "zone"),
               ("tb", "zone"),
               ("tb",),
               ())
GAMER_TIME_FEATURES = ("who", "prev", "phase", "margin", "down")
GAMER_DIST_FEATURES = ("who", "tb", "down", "dist", "zone")
GAMER_TIME_LEVELS = (GAMER_TIME_FEATURES, ("who", "prev", "phase"), ("who", "prev"))
GAMER_DIST_LEVELS = (GAMER_DIST_FEATURES, ("who", "tb", "dist", "zone"), ("who", "tb"))
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

FOURTH_FEATURES = ("togo10", "dist", "margin", "phase")
FOURTH_SIZES = dict(togo10=10, dist=5, margin=7, phase=7)
FOURTH_LEVELS = (FOURTH_FEATURES, ("togo10", "dist", "margin"), ("togo10", "dist"), ("togo10",), ())
GO, FG, PUNT = 0, 1, 2
FG_MAX = 65                   # longest kick tried (yards)
CONV_CELLS, CONV_MARGIN, CONV_CLOCK = s14.CONV_CELLS, s14.CONV_MARGIN, s14.CONV_CLOCK
REMAIN_SLICE = 20.0           # snaps left in the half, by 20-second slice of the time left in it
REMAIN_SLICES = int(2 * QUARTER // REMAIN_SLICE) + 1
REFINE_REMAINING = True
REFINE_PATHS = 3000
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
    """(half: 0 first, 1 second, 2 overtime, 20-second slice of the time left in it)."""
    period = np.asarray(period)
    half = np.where(period <= 2, 0, np.where(period <= 4, 1, 2))
    sl = np.clip((half_left(period, clock) // REMAIN_SLICE).astype(np.int64), 0, REMAIN_SLICES - 1)
    return half, sl


def share_slice(period, clock):
    """The 20-second slice of regulation a clock is in (0 the first; overtime the last)."""
    period, clock = np.asarray(period), np.asarray(clock, float)
    played = (np.minimum(period, 4) - 1) * QUARTER + (QUARTER - clock)
    return np.where(period >= 5, SHARE_SLICES - 1,
                    np.clip((played // SHARE_SLICE).astype(np.int64), 0, SHARE_SLICES - 1))


# ---------------------------------------------------------------- the draws

class Bins:
    """Real records grouped for drawing: every combination of the features (sizes) points at the
    first level whose group holds `min_records` (with `strict`, none if no level does), and each
    group's records are kept sorted by their value, so a draw (and a tilt) runs from the bottom of
    the group to the top."""

    def __init__(self, features, sizes, levels, lookup, start, count, level_of, payload):
        self.features, self.sizes, self.levels = tuple(features), tuple(sizes), tuple(levels)
        self.lookup, self.start, self.count, self.level_of, self.payload = lookup, start, count, level_of, payload

    @classmethod
    def build(cls, features, sizes, levels, X, value, payload, min_records, strict=False):
        """X (records, features) ints; value the sort key; payload {name: array} kept with them."""
        X = np.asarray(X, np.int64).reshape(-1, len(features))
        value = np.asarray(value, float)
        n = len(X)
        sizes = tuple(int(x) for x in sizes)
        M = int(np.prod(sizes))
        combos = np.stack(np.unravel_index(np.arange(M), sizes), axis=1)
        pos = {f: i for i, f in enumerate(features)}
        chosen_level = np.full(M, -1, np.int64)
        chosen_code = np.zeros(M, np.int64)
        rec_codes = []
        for li, lv in enumerate(levels):
            cols = [pos[f] for f in lv]
            dims = tuple(sizes[c] for c in cols)
            rc = np.ravel_multi_index(tuple(X[:, c] for c in cols), dims) if cols else np.zeros(n, np.int64)
            cc = np.ravel_multi_index(tuple(combos[:, c] for c in cols), dims) if cols else np.zeros(M, np.int64)
            rec_codes.append(rc)
            uniq, counts = np.unique(rc, return_counts=True)
            if not len(uniq):
                continue
            k = np.clip(np.searchsorted(uniq, cc), 0, len(uniq) - 1)
            cnt = np.where(uniq[k] == cc, counts[k], 0)
            need = min_records if (strict or lv) else 1
            take = (chosen_level < 0) & (cnt >= need)
            chosen_level[take], chosen_code[take] = li, cc[take]
        key = np.where(chosen_level >= 0, chosen_level * (1 << 40) + chosen_code, -1)
        uniq, inv = np.unique(key, return_inverse=True)
        lookup = np.where(key >= 0, inv - int(uniq[0] < 0), -1).astype(np.int32)
        bins = uniq[uniq >= 0]
        order_by_level = {}
        start, count, level_of, take_idx = [], [], [], []
        total = 0
        for b in bins:
            li, code = int(b) >> 40, int(b) & ((1 << 40) - 1)
            if li not in order_by_level:
                o = np.lexsort((value, rec_codes[li]))
                order_by_level[li] = (o, rec_codes[li][o])
            o, sorted_codes = order_by_level[li]
            lo, hi = np.searchsorted(sorted_codes, code), np.searchsorted(sorted_codes, code, side="right")
            idx = o[lo:hi]                                    # already sorted by value within the code
            start.append(total)
            count.append(len(idx))
            level_of.append(li)
            take_idx.append(idx)
            total += len(idx)
        idx = np.concatenate(take_idx) if take_idx else np.zeros(0, np.int64)
        return cls(features, sizes, levels, lookup, np.array(start, np.int64), np.array(count, np.int64),
                   np.array(level_of, np.int64), {k: np.asarray(v)[idx] for k, v in payload.items()})

    def bin_of(self, cols):
        """The group each row of features (a dict of arrays) draws from (-1: none)."""
        return self.lookup[np.ravel_multi_index(tuple(np.asarray(cols[f], np.int64) for f in self.features),
                                                self.sizes, mode="clip")]

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


# ---------------------------------------------------------------- the real snaps

def _side_of(team, a_home):
    return 0 if (team == "TEAM_A") == a_home else 1


def _with_side(rows):
    return [r if r.get("team_a_side") in ("home", "away") else dict(r, team_a_side="home") for r in rows]


def match_snaps(rows, pair=None, teams=None):
    """A match's scrimmage snaps with the features both draws read and who had the ball: a list of
    dicts (sim14.snap_records plus side, prev_cls, elapsed, remain, cut, play_seconds, the gamers and
    teams)."""
    rows = _with_side(rows)
    if not rows:
        return []
    a_home = rows[0]["team_a_side"] != "away"
    recs = s14.link_prev(s14.snap_records(rows))
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


def _fourth_downs(rows):
    """(period, clock, margin, field, distance, choice) for every real 4th down."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a.get("down") != "4" or a["play_kind"] not in s14.SNAP_KINDS or not a["field_position"] \
                or not a["distance"] or not a["period"] or a["period"] != b["period"] or not a["clock_seconds"]:
            continue
        m = s14._margin(a, a["offense"])
        if m is None:
            continue
        choice = PUNT if b["play_kind"] == "PUNT" else FG if b["play_kind"] == "FIELD_GOAL" else GO
        out.append((s14._i(a["period"]), s14._f(a["clock_seconds"]), m, s14._i(a["field_position"]),
                    max(1, s14._i(a["distance"])), choice))
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
        self.fourth = None
        self.fg_make = None                # P(make) by 5-yard kick distance from 15
        self.fg_seconds = np.array([5.0])
        self.punts = None                  # Bins over (ybin,) payload net, seconds
        self.kicks = {}                    # desperate -> (kept, field, seconds)
        self.safety_fields = np.array([35])
        self.two_p = np.zeros((CONV_CELLS, 2 * CONV_MARGIN + 1))
        self.two_good, self.kick_good = 0.5, 0.97
        self.to_def = np.zeros((7, 7))
        self.to_off = np.zeros((7, 7))
        self.remain = np.zeros((3, REMAIN_SLICES))
        self.share = np.linspace(1, 0, SHARE_SLICES)
        self.meta = {}

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        m = cls()
        with open(path, "rb") as fh:
            m.__dict__.update(pickle.load(fh))
        return m

    def gamer_index(self, handle):
        return self.gamers.get((handle or "").strip().upper(), len(self.gamers))

    def team_index(self, team):
        return self.teams.get((team or "").strip(), len(self.teams))

    def remain_at(self, period, clock):
        h, sl = remain_slice(period, clock)
        return self.remain[h, sl]


MODEL_FILE = "v21model.pkl"


def _features(recs, model):
    """{feature: array} for real snaps (snaps left in the half off the model's table)."""
    get = lambda k: np.array([r[k] for r in recs])
    period, clock = get("period"), get("clock").astype(float)
    return dict(prev=get("prev_cls"), phase=phase_of(period, clock), margin=margin_b(get("margin")),
                down=down_b(get("down")), dist=dist_b(get("distance")), zone=zone_b(get("field")),
                elapsed=elapsed_b(get("elapsed")), remain=remain_b(model.remain_at(period, clock)),
                tb=time_b(get("play_seconds")))


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
          seed=0):
    """The model off the export's matches ({code: rows}) and the history (gamers and teams)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    m = Model()
    m.mode = mode
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    recs, fourths, kicks, decisions, convs, safeties = [], [], [], [], [], []
    for code, rows in matches.items():
        rows = _with_side(rows)
        pair, teams = match_people(code, rows, hist)
        recs += match_snaps(rows, pair, teams)
        fourths += _fourth_downs(rows)
        kicks += s14.kick_records(rows)
        decisions += s14.kick_decisions(rows)
        convs += s14.conversions(rows)
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
    half, sl = remain_slice(np.array([r["period"] for r in recs]), np.array([r["clock"] for r in recs], float))
    rem = np.array([r["remain"] for r in recs], float)
    num, den = np.zeros(m.remain.shape), np.zeros(m.remain.shape)
    np.add.at(num, (half, sl), rem)
    np.add.at(den, (half, sl), 1)
    m.remain = np.where(den > 0, num / np.maximum(den, 1), 0.0)
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

    # 4th downs
    if fourths:
        f = np.array(fourths, float)
        p, c = f[:, 0].astype(int), f[:, 1]
        X = np.stack([np.clip(((100 - f[:, 3]) // 10).astype(int), 0, 9), dist_b(f[:, 4]), margin_b(f[:, 2]),
                      phase_of(p, c)], axis=1)
        m.fourth = Bins.build(FOURTH_FEATURES, [FOURTH_SIZES[x] for x in FOURTH_FEATURES], FOURTH_LEVELS, X,
                              f[:, 5], {"choice": f[:, 5].astype(np.int64)}, 20)
    # field goals: made by distance, the clock they take
    fgs = [d for d in decisions if d[0] == "fg"]
    dist_fg = np.arange(15, FG_MAX + 5, 5)
    made, tried = np.zeros(len(dist_fg)), np.zeros(len(dist_fg))
    for _, y, ok, secs, _ in fgs:
        k = min(len(dist_fg) - 1, max(0, int((100 - y + 17 - 15) // 5)))
        made[k] += bool(ok)
        tried[k] += 1
    league = made.sum() / max(1, tried.sum())
    m.fg_make = (made + 10 * league) / (tried + 10)
    if fgs:
        m.fg_seconds = np.array([d[3] for d in fgs], float)
    punts = [d for d in decisions if d[0] == "punt"]
    if punts:
        P = np.array([(d[1], d[2], d[3]) for d in punts], float)
        m.punts = Bins.build(("ybin",), (5,), (("ybin",), ()), np.clip((P[:, 0] // 20).astype(int), 0, 4)[:, None],
                             P[:, 1], {"net": P[:, 1].astype(np.int64), "seconds": P[:, 2]}, 20)
    for desperate in (False, True):
        ks = [k for k in kicks if k[0] == desperate]
        if ks:
            m.kicks[desperate] = (np.array([k[1] for k in ks], bool), np.array([k[2] for k in ks], np.int64),
                                  np.array([k[3] for k in ks], float))
    if safeties:
        m.safety_fields = np.array(safeties, np.int64)
    # conversions, in v14's cells
    if convs:
        num, den = np.zeros(m.two_p.shape), np.zeros(m.two_p.shape)
        two_n = two_good = kick_n = kick_good = 0
        for cell, margin, two, good in convs:
            num[cell, int(margin) + CONV_MARGIN] += two
            den[cell, int(margin) + CONV_MARGIN] += 1
            if two:
                two_n, two_good = two_n + 1, two_good + good
            else:
                kick_n, kick_good = kick_n + 1, kick_good + good
        base = num.sum(1, keepdims=True) / np.maximum(den.sum(1, keepdims=True), 1)
        m.two_p = (num + 5 * base) / (den + 5)
        m.two_good = (two_good + 5) / (two_n + 10)
        m.kick_good = (kick_good + 9) / (kick_n + 10)
    # timeouts before a running-clock snap: the defense's call, then the offense's
    run = [r for r in recs if r["prev_raw"] <= LOSS and r.get("timeouts_left")]
    if run:
        ph = phase_of(np.array([r["period"] for r in run]), np.array([r["clock"] for r in run], float))
        mb = margin_b(np.array([r["margin"] for r in run]))
        for name, role, left in (("to_def", s14.DEFENCE, 1), ("to_off", s14.OFFENCE, 0)):
            ok = np.array([r["timeouts_left"][left] > 0 for r in run])
            called = np.array([r.get("timeout_role") == role for r in run], float)
            num, den = np.zeros((7, 7)), np.zeros((7, 7))
            np.add.at(num, (ph[ok], mb[ok]), called[ok])
            np.add.at(den, (ph[ok], mb[ok]), 1)
            base = num.sum(1, keepdims=True) / np.maximum(den.sum(1, keepdims=True), 1)
            setattr(m, name, (num + 20 * base) / (den + 20))
    m.meta = dict(matches=len(matches), snaps=len(recs), mode=mode, timeouts=bool(run),
                  built=dt.datetime.now().isoformat(timespec="seconds"))
    if refine:
        before = m.remain.copy()
        refine_remaining(m, seed=seed)
        if verbose:
            got = np.abs(m.remain - before)[before > 0]
            print(f"  snaps left in the half, the sim's own against the real games': mean change"
                  f" {got.mean() if len(got) else 0:.2f} snaps; the draws rebuilt on it", flush=True)
        _build_draws(m, recs, min_records, verbose=False)
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
    dist_pay = {"kind": kind.astype(np.int8), "gain": gain.astype(np.int16), "shift": shift.astype(np.int16),
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
        share = b.level_shares(real[i]) if real is not None else b.level_shares(b.lookup)
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
SNAP_LOG = ("path", "period", "clock", "prev", "seconds", "drawn", "kind", "gain", "dist", "field", "side")


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
        q_points = np.zeros((P, 5), np.int64)
    q_score = score.copy()

    def kickoff(ix):
        kicker = 1 - team[ix]
        desperate = (period[ix] >= 4) & (score[ix, kicker] < score[ix, 1 - kicker]) \
            & (half_left(period[ix], clock[ix]) <= 180)
        for d in (False, True):
            sel = ix[desperate == d]
            pool = model.kicks.get(d) or model.kicks.get(False)
            if not len(sel):
                continue
            if pool is None:
                set_ball(sel, team[sel], np.full(len(sel), 25))
                continue
            j = rng.integers(0, len(pool[0]), len(sel))
            kept, field, secs = pool[0][j], pool[1][j], pool[2][j]
            clock[sel] = np.maximum(0.0, clock[sel] - secs)
            off = np.where(kept, 1 - team[sel], team[sel])
            tally("kickoffs", len(sel))
            tally("onside_kept", kept.sum())
            td = field >= 100
            if td.any():
                tally("kick_tds", td.sum())
                touchdown(sel[td], off[td])
            set_ball(sel[~td], off[~td], field[~td])

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
        two = rng.random(len(ix)) < model.two_p[cell, np.clip(lead, -CONV_MARGIN, CONV_MARGIN) + CONV_MARGIN]
        two |= ot & ot_done[ix, 1 - side] & ((lead == -1) | (lead == -2))
        good = rng.random(len(ix)) < np.where(two, model.two_good, model.kick_good)
        score[ix, side] += np.where(good, np.where(two, 2, 1), 0)
        tally("two_point_tries", two.sum())
        kick_to(ix, 1 - side)

    def kick_to(ix, receiver):
        team[ix] = receiver
        phase[ix] = KICK

    def fourth(ix):
        off = team[ix]
        cols = dict(togo10=np.clip((100 - y[ix]) // 10, 0, 9), dist=dist_b(dist[ix]),
                    margin=margin_b(score[ix, off] - score[ix, 1 - off]), phase=phase_of(period[ix], clock[ix]))
        if model.fourth is None:
            choice = np.full(len(ix), PUNT)
        else:
            choice = model.fourth.payload["choice"][model.fourth.draw(model.fourth.bin_of(cols), rng.random(len(ix)))]
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
            tally("field_goal_tries", len(fg))
            kd = 100 - y[fg] + 17
            made = rng.random(len(fg)) < model.fg_make[np.clip((kd - 15) // 5, 0, len(model.fg_make) - 1)]
            clock[fg] = np.maximum(0.0, clock[fg] - model.fg_seconds[rng.integers(0, len(model.fg_seconds), len(fg))])
            tally("field_goals", made.sum())
            mk, ms = fg[made], fg[~made]
            score[mk, team[mk]] += 3
            kick_to(mk, 1 - team[mk])
            set_ball(ms, 1 - team[ms], np.maximum(20, 100 - y[ms]))
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

    def draw_time(ix, prv, stopped=False):
        """Each snap's time; `stopped`: from the stopped-clock part of the same draw (the snap after
        the two-minute warning)."""
        off = team[ix]
        f = dict(prev=prv, phase=phase_of(period[ix], clock[ix]),
                 margin=margin_b(score[ix, off] - score[ix, 1 - off]), down=down_b(down[ix]), dist=dist_b(dist[ix]),
                 zone=zone_b(y[ix]), elapsed=elapsed_b(elapsed[ix]), remain=remain_b(model.remain_at(period[ix], clock[ix])))
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

    def draw_play(ix, secs):
        off = team[ix]
        f = dict(tb=time_b(secs), down=down_b(down[ix]), dist=dist_b(dist[ix]), zone=zone_b(y[ix]),
                 margin=margin_b(score[ix, off] - score[ix, 1 - off]), phase=phase_of(period[ix], clock[ix]),
                 elapsed=elapsed_b(elapsed[ix]), remain=remain_b(model.remain_at(period[ix], clock[ix])))
        u = rng.random(len(ix))
        k = k_dist[ix, off] if k_dist is not None else None
        j = model.dist.draw(model.dist.bin_of(f), u, k)
        names = ("kind", "gain", "shift", "td")
        out = {x: model.dist.payload[x][j].copy() for x in names}
        out["_from"], out["_j"] = np.zeros(len(ix), np.int64), j.copy()
        own("dist", f, ix, off, u, names, out)
        kind, gain, shift, td = (out[x] for x in names)
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
        return kind, gain, shift

    def play(ix):
        off = team[ix]
        # timeouts: only before a snap off a running clock, and they move only the time draw
        running = prev[ix] <= LOSS
        ph = phase_of(period[ix], clock[ix])
        mb = margin_b(score[ix, off] - score[ix, 1 - off])
        u = rng.random(len(ix))
        call_def = running & (tos[ix, 1 - off] > 0) & (u < model.to_def[ph, mb])
        call_off = running & ~call_def & (tos[ix, off] > 0) & (u < model.to_def[ph, mb] + model.to_off[ph, mb])
        prv = prev[ix].copy()
        tos[ix[call_def], 1 - off[call_def]] -= 1
        tos[ix[call_off], off[call_off]] -= 1
        prv[call_def | call_off] = TIMEOUT
        tally("timeouts", (call_def | call_off).sum())
        secs = draw_time(ix, prv)
        ran = secs > STOPPED_MAX
        # a running clock that runs out before the snap: the period is over, no snap
        gone = ran & (secs - PLAY_SECONDS >= clock[ix])
        if gone.any():
            clock[ix[gone]] = 0.0
            tally("clock_ran_out", gone.sum())
        keep = ~gone
        ix, secs, ran, prv = ix[keep], secs[keep], ran[keep], prv[keep]
        if not len(ix):
            return
        start_clock, start_prv = clock[ix].copy(), prv.copy()
        # the two-minute warning stops a running clock at 2:00; the snap then comes off a stopped clock
        warn = ran & ((period[ix] == 2) | (period[ix] == 4)) & (clock[ix] > LATE) \
            & (clock[ix] - (secs - PLAY_SECONDS) <= LATE)
        if warn.any():
            w = ix[warn]
            clock[w] = LATE
            secs[warn] = draw_time(w, prv[warn], stopped=True)
            tally("two_minute", len(w))
        clock[ix] = np.maximum(0.0, clock[ix] - secs)
        kind, gain, shift = draw_play(ix, secs)
        off = team[ix]
        elapsed[ix] += 1
        if st is not None:                    # logged as a real snap reads: PLAY_OVER to PLAY_OVER
            st["_snaps"].append((ix.copy(), period[ix].copy(), start_clock, start_prv, start_clock - clock[ix],
                                 secs.copy(), kind.copy(), np.minimum(gain, 100 - y[ix]), dist[ix].copy(),
                                 y[ix].copy(), off.copy()))
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
        gx, gg = ix[g], gain[g]
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
        ox, og, oy = gx[on], gg[on], y2[on]
        first = og >= dist[ox]
        prev[ox] = play_class(og, dist[ox])
        y[ox] = oy
        f, nf = ox[first], ox[~first]
        down[f] = 1
        dist[f] = np.minimum(10, 100 - y[f])
        down[nf] += 1
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
        if len(sn):
            on4 = sn[down[sn] >= 4]
            go = fourth(on4) if len(on4) else on4
            plays = np.concatenate([sn[down[sn] < 4], go])
            if len(plays):
                play(plays)
        ended = np.flatnonzero((phase != DONE) & (clock <= 0.0))
        if len(ended):
            end_period(ended)
    if st is not None:
        st["games"] = st.get("games", 0) + P
        st["final"] = score.copy()
        st["q_points"] = q_points
        st["unfinished"] = st.get("unfinished", 0) + int((phase != DONE).sum())
    return score[:, 0].reshape(S, n_paths), score[:, 1].reshape(S, n_paths)


def snap_log(stats):
    """The census's snaps as {SNAP_LOG name: array}."""
    log = stats.get("_snaps") or []
    if not log:
        return {k: np.zeros(0) for k in SNAP_LOG}
    return {k: np.concatenate([x[i] for x in log]) for i, k in enumerate(SNAP_LOG)}


def refine_remaining(model, n_paths=REFINE_PATHS, seed=0):
    """Snaps left in the half from the sim's own games: kick-offs at league, each snap's count of the
    snaps after it in its half, by the half's 20-second slice; slices the sim reached under 20 times
    keep the real games' count."""
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
    num, den = np.zeros(model.remain.shape), np.zeros(model.remain.shape)
    np.add.at(num, (half, sl), rem)
    np.add.at(den, (half, sl), 1)
    model.remain = np.where(den >= 20, num / np.maximum(den, 1), model.remain)
    return model.remain
