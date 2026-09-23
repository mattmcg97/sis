"""v3: a play-by-play simulation of the rest of the game on the real clock.

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
replayed while level.

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
        if b["play_kind"] == "PUNT" and b["offense"] != a["offense"]:
            out.append(("punt", y, (100 - _i(b["field_position"])) - y, seconds))
        elif b["play_kind"] == "FIELD_GOAL":
            made = "FIELD_GOAL_GOOD" in b["play_messages"]
            after = _i(b["field_position"]) if (not made and b["offense"] != a["offense"]) else None
            out.append(("fg", y, made, seconds, after))
    return out


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
        if (phase == 0 and clock <= 10) or 100 - y + 17 > dp.fg_max_distance + 30:
            pass
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
        snaps, kicks, decisions, conv, fourths, early = [], [], [], [], [], []
        for rows in matches.values():
            snaps += snap_records(rows)
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
        start, count, succ = {}, {}, {}
        kind, gain, newf, secs, rep, tdf = [], [], [], [], [], []
        for g in order:
            recs = list(bins[g])
            rng.shuffle(recs)
            recs.sort(key=lambda r: (-2000 if r["kind"] == DEF_TD else -1000 if r["kind"] == TURNOVER
                                     else r["gain"] - r["distance"]))
            start[g] = len(kind)
            count[g] = len(recs)
            succ[g] = sum(r["success"] for r in recs) / len(recs)
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
        punts = [d for d in decisions if d[0] == "punt"]
        for b in range(3):
            ps = [p for p in punts if _punt_bucket(p[1]) == b] or punts
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
        return t

    def success_of(self, key):
        return float(self.success[key])

    def save(self, path):
        arrays = dict(start=self.start, count=self.count, success=self.success, kind=self.kind,
                      gain=self.gain, new_field=self.new_field, seconds=self.seconds,
                      replay=self.replay, td_from=self.td_from, fg_seconds=self.fg_seconds,
                      fg_after=np.array([self.fg_after]), period_theta=self.period_theta,
                      go_for_two=self.go_for_two, go_shift=self.go_shift, fg_shift=self.fg_shift,
                      late_fg=self.late_fg, early_fg=self.early_fg, conv_rates=np.array([self.two_good, self.kick_good]))
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
        return t


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
        self.kicks_second_half = np.zeros(n, dtype=np.int8)   # the opening receiver
        self.theta = np.zeros((n, 2))
        self.aggression = np.zeros((n, 2))
        self.pace = np.ones((n, 2))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def simulate(tables, start, n_paths, rng=None, theta_sd=None, kneel_seconds=20.0,
             desperate_seconds=180.0, max_steps=400, stats=None):
    """Play every starting state `n_paths` times to the end.

    Returns (home, away) final scores, arrays of shape (states, n_paths).
    `theta_sd` (states x 2), if given, draws each path's efficiencies around
    the starting theta -- what the posterior still does not know.
    """
    rng = rng or np.random.default_rng()
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
    agg = rep(start.aggression)
    pace = rep(start.pace)
    dp = tables.drive
    gc = dp.go_coef
    ka, kb = dp.fg_kick_coef
    ma, mb, mc = dp.make_coef

    tally = (lambda name, k: stats.__setitem__(name, stats.get(name, 0) + int(k))) \
        if stats is not None else (lambda name, k: None)

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
        # half time: the opening receiver kicks
        h = ix[p == 2]
        period[h] = 3
        clock[h] = QUARTER
        phase[h] = KICK
        team[h] = kick2[h]
        # end of regulation / an overtime
        e = ix[p >= 4]
        level = score[e, 0] == score[e, 1]
        more = e[level & (period[e] < 4 + MAX_OT)]
        tally("overtime", (level & (period[e] == 4)).sum())
        phase[e[~level | (period[e] >= 4 + MAX_OT)]] = DONE
        period[more] += 1
        clock[more] = QUARTER
        phase[more] = KICK
        team[more] = rng.integers(0, 2, len(more))

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
        ph = phase[live]

        # conversions (they happen even with the clock at zero): late in a
        # game players chasing it go for two
        ix = live[ph == CONV]
        if len(ix):
            s_ = team[ix]
            margin = np.clip(score[ix, s_] - score[ix, 1 - s_], -CONV_MARGIN, CONV_MARGIN)
            cp = np.where(period[ix] <= 2, 0, np.where(period[ix] == 3, 1,
                          np.where((period[ix] >= 5) | (clock[ix] <= 180), 3, 2)))
            two = rng.random(len(ix)) < tables.go_for_two[cp, margin + CONV_MARGIN]
            good = rng.random(len(ix)) < np.where(two, tables.two_good, tables.kick_good)
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
                for flag in (False, True):
                    sub = ix[desperate == flag]
                    if not len(sub):
                        continue
                    onside, field, secs = tables.kick[flag]
                    j = rng.integers(0, len(field), len(sub))
                    rec = team[sub]
                    team[sub] = np.where(onside[j], rec, 1 - rec)
                    y[sub] = field[j]
                    down[sub] = 1
                    dist[sub] = 10
                    clock[sub] -= secs[j]
                    phase[sub] = SCRIM

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

        # kneel: ahead in the last quarter with the downs to cover the clock
        kneel = (p >= 4) & (margin > 0) & (c <= kneel_seconds * (5 - dn))
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
        fg_now = ~kneel & (drain | ((dn < 4) & (rng.random(len(ix)) < early_p)))
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
        r1, r2 = rng.random(len(ix)), rng.random(len(ix))
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
            j = rng.integers(0, len(tables.fg_seconds), len(fx))
            clock[fx] -= tables.fg_seconds[j]
            good = rng.random(len(fx)) < make[do_fg]
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
                j = rng.integers(0, len(net), len(sub))
                clock[sub] -= secs[j]
                turnover(sub, 100 - (y[sub] + net[j]))

        # snaps
        sx = ix[play]
        if not len(sx):
            continue
        so = team[sx]
        mode = _modes_np(period[sx], clock[sx], score[sx, so] - score[sx, 1 - so])
        key = _keys_np(mode, down[sx], dist[sx], y[sx])
        n = tables.count[key]
        uu = rng.random(len(sx))
        uu = 1.0 - (1.0 - uu) ** (exp_theta[sx, so] * period_exp[np.minimum(period[sx], 5)])
        j = tables.start[key] + np.minimum(n - 1, (uu * n).astype(np.int64))
        clock[sx] -= tables.seconds[j] * pace[sx, so]
        kd = tables.kind[j]
        tally("snaps", len(sx))
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
        short = (tdf >= 0) & (y[gx] < tdf)
        if short.any():
            extra = tdf[short] - y[gx][short]
            runs_on = rng.random(len(extra)) < np.exp(-extra / TD_TAIL)
            y2[short] = np.where(runs_on, 100, y2[short] + (rng.random(len(extra)) * extra).astype(np.int32))
        td = y2 >= 100
        tally("td", td.sum())
        if stats is not None:
            for q in (1, 2, 3, 4):
                tally(f"td_q{q}", (td & (period[gx] == q)).sum())
        if td.any():
            score_td(gx[td], team[gx[td]])
        sf = y2 <= 0
        if sf.any():
            s = gx[sf]
            score[s, 1 - team[s]] += 2
            phase[s] = KICK                         # the free kick
        rest = ~td & ~sf
        gx, gain, rp, y2 = gx[rest], gain[rest], rp[rest], y2[rest]
        # clock management as a half runs out: in range of a kick that would
        # help, players stop the clock (spike, time out, out of bounds) and
        # leave a few seconds to take it
        pg = period[gx]
        mg = score[gx, team[gx]] - score[gx, 1 - team[gx]]
        wants = ((pg == 2) | ((pg >= 4) & (mg <= 0) & (mg >= -3))) & (100 - y2 + 17 <= EARLY_FG_RANGE)
        before = clock[gx] + tables.seconds[j[gsel]][rest] * pace[gx, team[gx]]
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
    start = Start(1)
    start.team[:] = 1
    for _ in range(rounds):
        stats = {}
        simulate(tables, start, n_paths, np.random.default_rng(seed), stats=stats)
        cum = [stats[f"points_by_q{q}"] / stats[f"ended_q{q}"] for q in (1, 2, 3, 4)]
        got = np.diff([0.0] + cum)
        for q in range(4):
            tables.period_theta[q + 1] += (real_points[q] - got[q]) / (1.8 * max(1.0, real_points[q]))
        tables.period_theta[5] = tables.period_theta[4]
    return tables.period_theta.copy(), got


def quarter_points(matches):
    """The league's mean points in each of quarters 1-4, off the export."""
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
