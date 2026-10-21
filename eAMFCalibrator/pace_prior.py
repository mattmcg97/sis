"""The pace prior: each match's expected drives, walk-forward, for v21's pace nudge (v21-drives), written into
the walkthrough's tables beside the expected points so the sim reads both off one table.

    python -m eAMFCalibrator walkthrough --history ... --scouting eAMFCalibrator/out/scouting_playover.csv
    python -m eAMFCalibrator pace-prior --scouting eAMFCalibrator/out/scouting_playover.csv \
        --table eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv

Each past match's drives are counted off the scouting export as the sim counts its own
(eAMFModel/v21_report.real_drives: its scrimmage snaps, a new drive at a fresh possession, a change of
side or a new half), so the pace the nudge meets and the drives the sim makes are the same quantity.
Every midnight a ridge regression of a match's drives on its two gamers' and two teams' pace (the two
gamers' effects added, and the two teams') is fitted on the results in by then (RESULT_LAG after
kick-off) over the TRAIN_DAYS before, rows weighted 2^(-days ago / HALF_LIFE), and prices that day's
matches. It's xgb/pace_eff.py's pace model, which walk-forward ranked matches' drives better than a
REML mixed model (glmer's, its shrinkage estimated) and XGBoost on rolling form (xgb/README.md, "Which
model foresees drives best?").

Two columns go into each table:
  exp_drives      the match's expected drives (both sides')
  league_drives   that day's fit's league drives a match (the weighted mean): v21 takes each match's
                  pace against its own day's league (v21_stream.drives_table), so the league's drift
                  over a span doesn't read as pace
A match priced on a day with fewer than MIN_TRAIN matches with drives behind it is left blank, and v21
prices it without the pace nudge.
"""

import csv
import datetime as dt
import os

import numpy as np

from .bet_totals import _start

RESULT_LAG = dt.timedelta(minutes=36)
TRAIN_DAYS = 180
HALF_LIFE = 60.0
SCALAR = 0.5                     # pace_eff's rows counted at half
LAMBDA = {"gamer": 45.0, "team": 180.0, "fixed": 1e-4}      # pace_eff's
MIN_TRAIN = 200
# the scouting export's columns the sim's drive count reads (feed.snap_records)
KEEP = ("match_code", "message", "period", "clock_seconds", "play_kind", "team_a_side", "offense", "down",
        "distance", "field_position", "play_messages", "score_p1", "score_p2")
COLUMNS = ("exp_drives", "league_drives")


# ---------------------------------------------------------------- the drives

def export_stamp(export):
    st = os.stat(export)
    return {"export": os.path.abspath(export), "size": st.st_size, "mtime": int(st.st_mtime)}


def each_match(export, measure, on_match=None, warn=None):
    """{match: measure(its rows)}, streamed off the export a match at a time (it is written that way; a
    match's rows met again later are dropped, with a warning), each row cut to KEEP's columns."""
    out, split = {}, set()

    def flush(code, rows):
        if code in out:
            split.add(code)
            return
        out[code] = measure(rows)
        if on_match:
            on_match(len(out))

    with open(export, newline="", encoding="utf-8") as fh:
        cur_code, cur = None, []
        for r in csv.DictReader(fh):
            code = r["match_code"]
            if code != cur_code:
                if cur_code is not None:
                    flush(cur_code, cur)
                cur_code, cur = code, []
            cur.append({k: r.get(k, "") for k in KEEP})
        if cur_code is not None:
            flush(cur_code, cur)
    if split and warn:
        warn(f"  {len(split):,} matches' rows not together in {export}: the later ones dropped")
    return out


def count_drives(export, on_match=None, warn=None):
    """{match: its drives as the sim counts them} off the export (each_match)."""
    from eAMFModel.v21_report import real_drives
    out = each_match(export, real_drives, on_match, warn)
    return {c: n for c, n in out.items() if n > 0}


def read_drives(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["MATCH_CODE"]: int(r["drives"]) for r in csv.DictReader(fh)}


def write_drives(path, drives):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "drives"])
        w.writerows(sorted(drives.items()))


# ---------------------------------------------------------------- the fit

def matches_of(history):
    """(start, code, (gamer, gamer), (team, team)) of every match in the history, in time order."""
    out = []
    for r in history:
        t = _start(r.get("SCHEDULED_START_TIME_UTC"))
        code = (r.get("MATCH_CODE") or "").strip()
        if not t or not code:
            continue
        g = tuple((r.get(f"PLAYER_{k}_HANDLE") or "").strip().upper() for k in (1, 2))
        tm = tuple((r.get(f"PLAYER_{k}_TEAM") or "").strip() for k in (1, 2))
        out.append((t, code, g, tm))
    out.sort()
    return out


def fit(train, cut):
    """The ridge on `train` [(start, code, gamers, teams, drives)] as of `cut`: (intercept, {gamer: effect},
    {team: effect}, the weighted league drives)."""
    gamers = sorted({g for _, _, gs, _, _ in train for g in gs if g})
    teams = sorted({t for _, _, _, ts, _ in train for t in ts if t})
    gi = {g: 1 + i for i, g in enumerate(gamers)}
    ti = {t: 1 + len(gamers) + i for i, t in enumerate(teams)}
    width = 1 + len(gamers) + len(teams)
    X = np.zeros((len(train), width))
    X[:, 0] = 1.0
    for k, (_, _, gs, ts, _) in enumerate(train):
        for g in gs:
            if g:
                X[k, gi[g]] += 1.0
        for t in ts:
            if t:
                X[k, ti[t]] += 1.0
    y = np.array([d for *_, d in train], float)
    age = np.array([(cut - s).total_seconds() / 86400.0 for s, *_ in train])
    w = np.power(2.0, -age / HALF_LIFE)
    w = SCALAR * w / w.mean()
    pen = np.concatenate([[LAMBDA["fixed"]], np.full(len(gamers), LAMBDA["gamer"]), np.full(len(teams), LAMBDA["team"])])
    beta = np.linalg.solve(X.T @ (w[:, None] * X) + np.diag(pen), X.T @ (w * y))
    return (float(beta[0]), {g: float(beta[i]) for g, i in gi.items()}, {t: float(beta[i]) for t, i in ti.items()},
            float(np.average(y, weights=w)))


def walk(history, drives, first, last, on_day=None):
    """{match: (expected drives, the day's league drives)} for every match from `first` to `last` (dates),
    each priced by its own midnight's fit on the results in by then."""
    from bisect import bisect_left
    ms = matches_of(history)
    starts = [m[0] for m in ms]
    played = [(t, c, g, tm, drives[c]) for t, c, g, tm in ms if c in drives]
    out = {}
    day = first
    lo = hi = 0
    while day <= last:
        cut = dt.datetime.combine(day, dt.time())
        nxt = cut + dt.timedelta(days=1)
        today = ms[bisect_left(starts, cut):bisect_left(starts, nxt)]
        while hi < len(played) and played[hi][0] + RESULT_LAG <= cut:
            hi += 1
        while lo < hi and played[lo][0] < cut - dt.timedelta(days=TRAIN_DAYS):
            lo += 1
        if today and hi - lo >= MIN_TRAIN:
            b0, ge, te, league = fit(played[lo:hi], cut)
            for _, code, g, tm in today:
                out[code] = (b0 + sum(ge.get(x, 0.0) for x in g) + sum(te.get(x, 0.0) for x in tm), league)
        if on_day:
            on_day(day)
        day += dt.timedelta(days=1)
    return out


# ---------------------------------------------------------------- the tables

def add_to_rows(rows, pace):
    """Each table row with its match's exp_drives and league_drives (blank where it has none)."""
    for r in rows:
        p = pace.get((r.get("MATCH_CODE") or "").strip())
        r["exp_drives"] = round(p[0], 3) if p else ""
        r["league_drives"] = round(p[1], 3) if p else ""
    return rows


def add_to_table(path, pace, out=None):
    """The table at `path` written again (to `out`, else in place) with exp_drives and league_drives; the
    matches it gave them."""
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        cols = list(rd.fieldnames or [])
        rows = list(rd)
    cols += [c for c in COLUMNS if c not in cols]
    add_to_rows(rows, pace)
    with open(out or path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, cols)
        w.writeheader()
        w.writerows(rows)
    return sum(1 for r in rows if r["exp_drives"] != "")


def lines(pace, drives):
    """How the expected drives went against the real ones, on the priced matches with drives (every price
    made before its match: a fair test)."""
    got = [(pace[c][0], pace[c][1], drives[c]) for c in pace if c in drives]
    out = [f"  pace: {len(pace):,} matches priced, {len(got):,} of them with drives counted"]
    if len(got) < 20:
        return out
    exp, league, real = (np.array(x, float) for x in zip(*got))
    rel = exp / league
    r = float(np.corrcoef(rel, real)[0, 1])
    out.append(f"  the expected drives {exp.mean():.2f} a match (sd {exp.std():.2f}), the real {real.mean():.2f} "
               f"(sd {real.std():.2f}); r {r:.3f} between the pace (expected over the day's league) and the real")
    edges = np.quantile(rel, [0.2, 0.4, 0.6, 0.8])
    q = np.searchsorted(edges, rel, side="right")
    out.append("  by fifth of pace, expected -> real drives: " + "  ".join(
        f"{exp[q == k].mean():.2f} -> {real[q == k].mean():.2f}" for k in range(5) if (q == k).any()))
    return out


def run(history, export, first, last, cache_dir, on_match=None, on_day=None, log=print):
    """The pace for every match from `first` to `last`: the drives counted (kept in cache_dir while the
    export is the same), the daily fits. Returns ({match: (exp_drives, league_drives)}, {match: drives})."""
    import json
    os.makedirs(cache_dir, exist_ok=True)
    path, stamp_path = os.path.join(cache_dir, "drives.csv"), os.path.join(cache_dir, "drives.json")
    stamp = export_stamp(export)
    kept = False
    if os.path.exists(path) and os.path.exists(stamp_path):
        with open(stamp_path, encoding="utf-8") as fh:
            kept = json.load(fh) == stamp
    if kept:
        drives = read_drives(path)
    else:
        drives = count_drives(export, on_match, warn=log)
        write_drives(path, drives)
        with open(stamp_path, "w", encoding="utf-8") as fh:
            json.dump(stamp, fh, indent=1)
    log(f"  drives: {len(drives):,} matches off {export}" + (" (counted before, kept)" if kept else ""))
    pace = walk(history, drives, first, last, on_day)
    with open(os.path.join(cache_dir, "pace.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "exp_drives", "league_drives", "drives"])
        for c, (e, lg) in pace.items():
            w.writerow([c, round(e, 3), round(lg, 3), drives.get(c, "")])
    return pace, drives
