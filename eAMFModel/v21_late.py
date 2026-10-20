"""A rolling correction of v2.1's late Q4 moneyline: the sim's chance of the side with the ball against what
the real games did from the same states.

    python -m eAMFModel v21-late-fit --model v21_t10 --states eAMFCalibrator/out_v21_endgame_sep/v21_endgame_states.csv \
        eAMFCalibrator/out_v21_endgame/v21_endgame_states.csv

v21-endgame found the late game drifting. Through September a side 1-3 behind with the ball in Q4 won 34-41%
of its games a week, and the sim had it about right. From the last week of September it won more and more:
48%, then 50% and 57% in the first two weeks of October. The sim stayed at about 40%. Level sides went the
other way: the sim had them 6-9 points short in September and about right in October. No fixed correction
keeps up with that, so this one rolls.

For each match day, a shift on the log-odds of the side with the ball winning, one for each cell of its
lead (LEADS) and the time left in Q4 (TIMES). It is fitted on the late states of the WINDOW days before the
day (v21-endgame's v21_endgame_states.csv: each real Q4 state's sim chance against what really followed,
each match counting the same) and shrunk toward none by SHRINK matches. A day's own games never fit its own
shift. The table goes beside the build (LATE_FILE), and the calibrator's candidate v21-late prices with it.
Each in-play book of Q4 with the sides LEAD_MAX or fewer apart has its margin's mass either side of a level
game scaled, so the side with the ball wins at the shifted chance. The spreads move with it. The kick-off,
the earlier quarters, overtime and the totals are as v21 prices them."""

import csv
import datetime as dt
import json
import math
import os

import numpy as np

LATE_FILE = "late_q4.json"
LEADS = ((-8, -4, "4-8 behind"), (-3, -3, "3 behind"), (-2, -1, "1-2 behind"), (0, 0, "level"),
         (1, 3, "1-3 ahead"), (4, 8, "4-8 ahead"))
TIMES = ((240.0, 120.0, "4:00-2:01"), (120.0, 60.0, "2:00-1:01"), (60.0, -1.0, "1:00-0:00"))
LEAD_MAX = 8
WINDOW = 14
SHRINK = 30.0
SHIFT_MAX = 2.0


def day_of(code, scheduled=None):
    """A match's day: off its code (AF + its number + DDMMYY), else its scheduled start."""
    c = str(code or "")
    if len(c) >= 11 and c[5:11].isdigit():
        try:
            return dt.date(2000 + int(c[9:11]), int(c[7:9]), int(c[5:7]))
        except ValueError:
            pass
    if scheduled:
        try:
            return dt.date.fromisoformat(str(scheduled)[:10])
        except ValueError:
            return None
    return None


def cell_of(lead, clock):
    """The cell of a Q4 state: its lead band times its time band (LEADS x TIMES), or None outside them."""
    li = next((i for i, (a, b, _) in enumerate(LEADS) if a <= lead <= b), None)
    ti = next((i for i, (hi, lo, _) in enumerate(TIMES) if lo < clock <= hi), None)
    return None if li is None or ti is None else li * len(TIMES) + ti


def cell_name(cell):
    return f"{LEADS[cell // len(TIMES)][2]}, {TIMES[cell % len(TIMES)][2]}"


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, float)))


def read_states(paths):
    """v21-endgame's late states off its CSVs, each with its day and cell."""
    out = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lead, clock = int(float(r["lead"])), float(r["clock"])
                c, day = cell_of(lead, clock), day_of(r["code"])
                if c is None or day is None:
                    continue
                out.append(dict(code=r["code"], day=day, cell=c, lead=lead, clock=clock,
                                sim=float(r["sim_win"]), real=float(r["real_win"])))
    return out


def fit_shift(rows, shrink=SHRINK):
    """The log-odds shift that takes the sim's mean chance to the real wins, each match counting the same,
    shrunk toward none by `shrink` matches: (shift, matches)."""
    if not rows:
        return 0.0, 0
    codes = {}
    for x in rows:
        codes[x["code"]] = codes.get(x["code"], 0) + 1
    w = np.array([1.0 / codes[x["code"]] for x in rows])
    z = _logit([x["sim"] for x in rows])
    y = np.array([x["real"] for x in rows])
    delta = 0.0
    for _ in range(25):
        p = _sig(z + delta)
        h = float(np.sum(w * p * (1 - p)))
        if h <= 1e-9:
            break
        delta += float(np.sum(w * (y - p))) / h
    n = len(codes)
    return float(np.clip(delta * n / (n + shrink), -SHIFT_MAX, SHIFT_MAX)), n


def fit(states, days=None, window=WINDOW, shrink=SHRINK):
    """{day (ISO): {cell: shift}} for each of `days` (default: every day of the states with `window` days of
    states before it), each fitted on the states of the window days before it."""
    by_day = {}
    for x in states:
        by_day.setdefault(x["day"], []).append(x)
    have = sorted(by_day)
    if days is None:
        days = [d for d in have if have and (d - have[0]).days >= window]
    table = {}
    for day in days:
        past = [x for d, xs in by_day.items() if day - dt.timedelta(days=window) <= d < day for x in xs]
        cells = {}
        for x in past:
            cells.setdefault(x["cell"], []).append(x)
        table[day.isoformat()] = {c: round(fit_shift(rows, shrink)[0], 4) for c, rows in sorted(cells.items())}
    return table


def write(path, table, **meta):
    days = {d: {str(c): v for c, v in cells.items()} for d, cells in table.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(meta, days=days), fh, indent=1)


def load(model_dir):
    """{day: {cell: shift}} beside a build (LATE_FILE), or None."""
    path = os.path.join(model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir), LATE_FILE)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        got = json.load(fh)
    return {d: {int(c): float(v) for c, v in cells.items()} for d, cells in got.get("days", {}).items()}


def day_shifts(table, code, scheduled=None):
    """The shifts of a match's day ({cell: shift}), or None (no table, or the day not in it)."""
    if not table:
        return None
    day = day_of(code, scheduled)
    return None if day is None else table.get(day.isoformat())


def shift_of(shifts, period, clock, ball_lead):
    """The shift for a state: Q4, the side with the ball `ball_lead` ahead (0: none)."""
    if not shifts or int(period) != 4 or abs(int(ball_lead)) > LEAD_MAX:
        return 0.0
    c = cell_of(int(ball_lead), float(clock))
    return 0.0 if c is None else float(shifts.get(c, 0.0))


def adjust(margin_pmf, home_has_ball, shift, offset=None):
    """The home margin's pmf with its mass either side of level scaled so the side with the ball wins at its
    chance's log-odds plus `shift` (a level final, its mass kept, counting for neither)."""
    mp = np.asarray(margin_pmf, float)
    if not shift:
        return mp
    off = (len(mp) - 1) // 2 if offset is None else offset
    up, down = mp[off + 1:].sum(), mp[:off].sum()
    if up <= 0 or down <= 0:
        return mp
    p_home = up / (up + down)
    p_ball = p_home if home_has_ball else 1 - p_home
    q_ball = float(_sig(math.log(p_ball / (1 - p_ball)) + shift))
    q_home = q_ball if home_has_ball else 1 - q_ball
    out = mp.copy()
    out[off + 1:] *= q_home / p_home
    out[:off] *= (1 - q_home) / (1 - p_home)
    return out


def check(states, table):
    """On the fitted days' own states: the side with the ball's real wins, the sim's chance and the corrected
    one, by lead band (each match counting the same) -- how far the rolling shift closes the gap."""
    rows = [x for x in states if x["day"].isoformat() in table]
    out = []
    for i, (a, b, name) in enumerate(LEADS):
        g = [x for x in rows if a <= x["lead"] <= b]
        if not g:
            continue
        by = {}
        for x in g:
            fixed = float(_sig(_logit(x["sim"]) + table[x["day"].isoformat()].get(x["cell"], 0.0)))
            by.setdefault(x["code"], []).append((x["real"], x["sim"], fixed))
        m = np.array([np.mean(v, axis=0) for v in by.values()])
        out.append((name, len(by), *(100 * m.mean(0))))
    return out


def report(states, table, window, shrink):
    days = sorted(table)
    L = [f"  v2.1-LATE: a shift on the log-odds of the side with the ball winning, by its lead and Q4's time left,"
         f" fitted each day on the {window} days before it ({len(states):,} late states), shrunk by {shrink:g} matches;"
         f" {len(days)} days, {days[0] if days else '-'} to {days[-1] if days else '-'}"]
    if days:
        last = table[days[-1]]
        L.append(f"    the shifts on {days[-1]} (log-odds; +0.40 at even is about 10 points of chance):")
        for c in sorted(last):
            L.append(f"      {cell_name(c):24s} {last[c]:+.3f}")
        L.append("    on the fitted days' own states: the side with the ball wins, % real / the sim / corrected")
        for name, n, real, sim, fixed in check(states, table):
            L.append(f"      {name:11s} {n:5,d} matches  {real:5.1f} / {sim:5.1f} / {fixed:5.1f}")
    return L
