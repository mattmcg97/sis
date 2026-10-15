"""The backtest suite: every pre-match prior variant over one window -- NB2 refitted daily at each
half-life, glmer, each with the rematch, inference and feature layers -- its inputs audited for
anything a match could not have known, graded against prod and replayed on the bets, with one
ranked summary.

    python -m eAMFCalibrator backtest-suite --since "2026-08-15 00:00:00" --until "2026-10-15 00:00:00"
    python -m eAMFCalibrator backtest-suite --config suite.json --steps audit,backtest,replay,summary

The steps (--steps, default all, in this order; each output kept in the suite directory, so a step
can be rerun alone):
  history    every settled match (eAMFCalibrator history's), for the daily fits: match_history.csv
  fits       eAMFModel prior-daily's NB2, refitted each day of the window, one directory a half-life
             (nb2_hl<D>/); the days already fitted are kept
  audit      the inputs checked (suite_audit.txt): every daily fit made only on results before its
             day, at its half-life; every arm's coverage of the window; the gamers' stats' coverage,
             freshness and a scramble check (below). A FAIL stops the run (--keep-going carries on)
  backtest   prior-backtest on every arm and the controls, off one read of the data, with the layers'
             leak check: prior_backtest.txt/csv
  replay     prior-replay on the same arms, off the same read and fits: prior_replay.txt/csv
  summary    every variant ranked by its moneyline against prod, with its total, its replay on the
             clean bets and each period: suite_summary.txt/csv

The checks a run makes on itself:
  scramble   the results (and the gamers' stats) from the window's middle day on shuffled between
             matches and changed at random (so their totals and averages change too), and every layer
             refitted: each match before that day must move exactly as it did -- a layer reading
             anything from a later day would differ. The same refit without the scramble must
             reproduce every move (determinism)
  echo       prod's closing moneyline and line read back as an arm (its implied margin and total, the
             normal reading): it must grade as prod does, or the matches, sides or grading are out of
             line. A diagnostic: prod is a third party's, not a candidate
  shuffled   the first NB2 arm's prices shuffled between matches of the same day: it must grade well
             below its source, or the grading isn't reading the prices
The controls are listed apart in the summary, not ranked.

The settings (--config, a JSON object; anything left out is DEFAULTS'):
  nb2_half_lives   NB2's half-lives (days), one daily NB2 arm each (nb2-hl<D>)
  nb2_mode         #MODE for the NB2 arms ("" : their own score model)
  nb2_workers      NB2 fits run side by side
  glmer            a glmer arm's source (cadence.R's PATH@PREFIX, or rolling:DIR); "" for none
  arms             any other arms, {NAME: SOURCE[#MODE]} as prior-backtest's --arm
  features         the gamers' per-match stats (player-stats' player_match_stats.csv; null: the one in
                   the output directory; "none": no feature layer)
  draws, boot      simulated scorelines a match; the replay's bootstrap resamples (null: config's)
  controls         the echo and shuffled arms
  leak_check       the scramble and determinism checks
  periods          [[label, from, until], ...] (dates; null: open): the summary's periods
"""

import csv
import datetime as dt
import json
import math
import os
import random
import subprocess
import time
from collections import Counter, defaultdict

import numpy as np

from . import config, gamer_features, prior_backtest as pb, prior_replay as pr, rematch_layer, snowflake_io
from .bet_totals import _start

STEPS = ("history", "fits", "audit", "backtest", "replay", "summary")
DEFAULTS = {
    "nb2_half_lives": [30, 60, 120],
    "nb2_mode": "",
    "nb2_workers": 2,
    "glmer": "glmer/out/cadence/cadence_matches.csv@trim:hl60_x0.5:daily",
    "arms": {},
    "features": None,
    "draws": 40000,
    "boot": None,
    "controls": True,
    "leak_check": True,
    "periods": [["before the detail gap", None, "2026-09-23"], ["the detail gap", "2026-09-23", "2026-10-09"],
                ["after the detail gap", "2026-10-09", None]],
}
HISTORY = "match_history.csv"
MANIFEST = "suite.json"
AUDIT = "suite_audit.txt"
ECHO, SHUFFLED = "echo", "shuffled"
COVERAGE_WARN = 0.95
FEATURE_COVERAGE_WARN = 0.8
LAYER_KEYS = ("move_margin", "move_total", "infer_margin", "infer_total", "both_margin", "both_total",
              "feat_margin", "feat_total", "bf_margin", "bf_total")
STAT_KEEP = ("match_code", "start", "side", "player", "opponent")      # what a scramble leaves on a stats row
ECHO_TOLERANCE = 0.0005           # log loss: the echo against prod
SHUFFLED_MARGIN = 0.005           # log loss: the shuffled arm at least this far behind its source


def load_config(path=None):
    """DEFAULTS with a JSON file's settings on top."""
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        with open(path, encoding="utf-8") as fh:
            user = json.load(fh)
        unknown = sorted(set(user) - set(DEFAULTS))
        if unknown:
            raise SystemExit(f"{path}: no setting {', '.join(unknown)}; the settings are {', '.join(DEFAULTS)}")
        cfg.update(user)
    return cfg


def nb2_dir(out_dir, half_life):
    return os.path.join(out_dir, f"nb2_hl{float(half_life):g}")


def arm_list(cfg, out_dir):
    """[(name, spec)]: an NB2 arm a half-life, the glmer arm, the others."""
    mode = f"#{cfg['nb2_mode']}" if cfg.get("nb2_mode") else ""
    arms = [(f"nb2-hl{float(h):g}", f"rolling:{nb2_dir(out_dir, h)}{mode}") for h in cfg["nb2_half_lives"]]
    if cfg.get("glmer"):
        arms.append(("glmer-daily", cfg["glmer"]))
    arms += list((cfg.get("arms") or {}).items())
    names = [n for n, _ in arms]
    clash = sorted({n for n in names if names.count(n) > 1} | ({ECHO, SHUFFLED, pb.PROD} & set(names)))
    if clash:
        raise SystemExit(f"arm names {', '.join(clash)} are taken: name them otherwise")
    return arms


def window_days(since, until):
    """(first, last) match day of [since, until): until None is today (UTC)."""
    first = _start(since).date()
    end = _start(until) if until else None
    last = (end - dt.timedelta(seconds=1)).date() if end else dt.datetime.now(dt.timezone.utc).date()
    return first, last


def _git():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


class Manifest:
    """suite.json: the settings, window, arms, each step's time, the audit's tally; kept across runs
    of the same directory."""

    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, MANIFEST)
        self.data = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                self.data = json.load(fh)

    def update(self, **kw):
        self.data.update(kw)
        self.save()

    def step(self, name, seconds):
        now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        self.data.setdefault("steps", {})[name] = {"finished": now.isoformat(timespec="seconds"),
                                                   "seconds": round(seconds, 1), "git": _git()}
        self.save()

    def save(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=1, default=str)


class Audit:
    """The checks' lines: PASS, WARN or FAIL, what was checked and what was found."""

    def __init__(self):
        self.lines, self.count = [], Counter()

    def add(self, level, check, detail=""):
        self.count[level] += 1
        line = f"  {level:4s}  {check}" + (f": {detail}" if detail else "")
        self.lines.append(line)
        print(line, flush=True)

    def info(self, text):
        self.lines.append(f"        {text}")
        print(f"        {text}", flush=True)

    def tally(self):
        return {k: self.count[k] for k in ("PASS", "WARN", "FAIL")}

    def write(self, path, header):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(header + self.lines) + "\n")


# --------------------------------------------------------------------------- history and fits

def step_history(cur, out_dir):
    rows = snowflake_io.fetch_history(cur)
    path = os.path.join(out_dir, HISTORY)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, snowflake_io.HISTORY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows):,} settled matches -> {path}", flush=True)
    return path


def step_fits(cfg, out_dir, first, last):
    from eAMFModel import nb2_prior, rolling_prior
    path = os.path.join(out_dir, HISTORY)
    if not os.path.exists(path):
        raise SystemExit(f"no {path}: run the history step first")
    history = nb2_prior.load_history(path)
    for h in cfg["nb2_half_lives"]:
        print(f"\n  NB2, half-life {float(h):g} days: a fit each day {first} to {last}", flush=True)
        rolling_prior.fit(history, "nb2", nb2_dir(out_dir, h), first, last, workers=cfg["nb2_workers"],
                          options={"half_life": h})


# --------------------------------------------------------------------------- the audit

def _starts(path):
    """Every SCHEDULED_START_TIME_UTC in a history CSV."""
    with open(path, newline="", encoding="utf-8") as fh:
        return [t for t in (_start(r.get("SCHEDULED_START_TIME_UTC")) for r in csv.DictReader(fh)) if t is not None]


def audit_history(audit, history, first, last):
    """The history: no match twice; the window's matches; how fresh it is. Returns the window's."""
    codes = Counter(r["MATCH_CODE"] for r in history)
    twice = [c for c, n in codes.items() if n > 1]
    audit.add("FAIL" if twice else "PASS", "every match once in the history",
              f"{len(twice):,} twice ({', '.join(twice[:5])})" if twice else f"{len(codes):,} matches")
    window = [r for r in history if (t := _start(r.get("SCHEDULED_START_TIME_UTC"))) and first <= t.date() <= last]
    audit.add("PASS" if window else "FAIL", "the window's settled matches", f"{len(window):,} from {first} to {last}")
    newest = max((_start(r.get("SCHEDULED_START_TIME_UTC")) for r in window), default=None)
    if newest is not None and newest.date() < last:
        audit.add("WARN", "the history reaches the window's end",
                  f"its last match is {newest:%Y-%m-%d %H:%M}, the window runs to {last}: rerun the history step")
    return window


def audit_rolling(audit, name, directory, first, last, half_life=None):
    """A prior-daily directory: its setting, its days, and every fit made on results before its day."""
    index = os.path.join(directory, "rolling.json")
    if not os.path.exists(index):
        audit.add("FAIL", f"{name}: daily fits", f"no {index}: run the fits step")
        return
    with open(index, encoding="utf-8") as fh:
        meta = json.load(fh)
    if half_life is not None:
        want = {"half_life": half_life}
        got = meta.get("options") or {}
        audit.add("PASS" if got == want else "FAIL", f"{name}: fitted at its half-life",
                  f"{got or 'the defaults'}" + ("" if got == want else f", not {want}"))
    days = {dt.date.fromisoformat(d) for d in meta.get("days", [])}
    want_days = {first + dt.timedelta(days=k) for k in range((last - first).days + 1)}
    missing = sorted(want_days - days)
    if days and min(days) > first:
        audit.add("WARN", f"{name}: a fit for the window's first day",
                  f"the first is {min(days)}: the matches before it are left out (their fit has seen them)")
    if missing:
        audit.add("WARN", f"{name}: a fit for every day", f"{len(missing)} days without one (each priced by the "
                  f"day before's fit): {', '.join(d.isoformat() for d in missing[:6])}"
                  + (" ..." if len(missing) > 6 else ""))
    else:
        audit.add("PASS", f"{name}: a fit for every day", f"{len(want_days)} days")
    late, wrong, checked = [], [], 0
    for day in sorted(days & want_days):
        d = os.path.join(directory, day.isoformat())
        before = dt.datetime.combine(day, dt.time())
        level = os.path.join(d, "level.json")
        if os.path.exists(level):
            with open(level, encoding="utf-8") as fh:
                lv = json.load(fh)
            if _start(lv.get("before")) != before or (half_life is not None
                                                     and lv.get("half_life_days", half_life) != half_life):
                wrong.append(day)
        for path, cut in ((os.path.join(d, "history.csv"), before),
                          (os.path.join(d, "level", "history.csv"), before - dt.timedelta(days=7))):
            if os.path.exists(path):
                checked += 1
                newest = max(_starts(path), default=None)
                if newest is not None and newest >= cut:
                    late.append(f"{day} ({os.path.relpath(path, d)} has {newest:%Y-%m-%d %H:%M})")
    audit.add("FAIL" if late else "PASS", f"{name}: every fit on results before its day",
              f"{len(late)} not: {'; '.join(late[:4])}" if late else f"{checked} training files read")
    if wrong:
        audit.add("FAIL", f"{name}: each fit's cut-off and half-life as its day says",
                  ", ".join(d.isoformat() for d in wrong[:6]))


def audit_csv_arm(audit, name, spec, window_codes):
    """A CSV arm: its file, and how much of the window it prices."""
    source = pb.parse_arm(spec)[0]
    path = source.partition("@")[0]
    if not os.path.exists(path):
        audit.add("FAIL", f"{name}: source", f"no {path} (drop the arm from the settings, or make it)")
        return
    try:
        points = pb.read_points(source, [], [])
    except SystemExit as e:
        audit.add("FAIL", f"{name}: source", str(e))
        return
    n = sum(1 for c in window_codes if c in points)
    share = n / len(window_codes) if window_codes else 0.0
    level = "FAIL" if not n else "WARN" if share < COVERAGE_WARN else "PASS"
    audit.add(level, f"{name}: prices the window", f"{n:,} of {len(window_codes):,} matches ({share:.0%})"
              + ("; the backtest grades only matches every arm priced" if level != "PASS" else ""))
    fits = os.path.join(os.path.dirname(path) or ".", "cadence_fits.csv")
    prefix = source.partition("@")[2]
    if prefix and os.path.exists(fits):
        with open(fits, newline="", encoding="utf-8") as fh:
            cuts = [_start(r.get("cut")) for r in csv.DictReader(fh) if r.get("arm") == prefix]
        cuts = [c for c in cuts if c is not None]
        if prefix.endswith(":daily") or prefix.count(":") < 2:
            off = [c for c in cuts if c.time() != dt.time()]
            audit.add("FAIL" if off else "PASS", f"{name}: each daily fit cut at midnight",
                      f"{len(off)} not" if off else f"{len(cuts)} fits")


def audit_features(audit, path, window, first, last, periods=DEFAULTS["periods"], seed=7):
    """The gamers' stats: there, fresh, their coverage of the window (by period), and the daily
    reading's scramble check."""
    if not path:
        audit.add("PASS", "the gamers' stats", "off (features: none)")
        return
    if not os.path.exists(path):
        audit.add("FAIL", "the gamers' stats", f"no {path}: run player-stats (or set features to \"none\")")
        return
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    newest = max((t for t in (_start(r.get("start")) for r in rows) if t), default=None)
    if newest is None or newest.date() < last - dt.timedelta(days=1):
        audit.add("WARN", "the gamers' stats reach the window's end",
                  f"the newest is {newest:%Y-%m-%d %H:%M}: rerun player-stats" if newest else "no dated rows")
    if rows and "points_msgs" not in rows[0]:
        audit.add("WARN", "the gamers' stats carry the reconciliation counts", "rerun player-stats")
    feats = gamer_features.daily(rows, since=first)
    codes = [r["MATCH_CODE"] for r in window]
    full = sum(1 for c in codes if gamer_features.margin_x(feats.get(c)))
    share = full / len(codes) if codes else 0.0
    audit.add("WARN" if share < FEATURE_COVERAGE_WARN else "PASS", "the gamers' stats cover the window",
              f"{full:,} of {len(codes):,} matches with every stat ({share:.0%})")
    detail = {r["match_code"] for r in rows if gamer_features._detail(r)}
    for label, lo, hi in periods:
        sub = [r for r in window if _in_period(_start(r["SCHEDULED_START_TIME_UTC"]), lo, hi)]
        if sub:
            audit.info(f"{label}: {sum(1 for r in sub if r['MATCH_CODE'] in detail):,} of {len(sub):,} matches "
                       "with play detail on 80%+ of snaps")
    ok, n_before, n_moved = scramble_features(rows, first, last, seed)
    audit.add("PASS" if ok else "FAIL", "the gamers' stats: nothing from a later day",
              f"stats from {mid_day(first, last)} on shuffled: {n_before:,} earlier matches read the same"
              + ("" if ok else " -- NOT")
              + (f", {n_moved:,} later ones moved" if n_moved else " (and nothing later moved: the check is idle)"))


def mid_day(first, last):
    return first + dt.timedelta(days=(last - first).days // 2)


def _jiggle(v, rnd):
    """A number scaled by a random factor (a blank or a word left as it is)."""
    try:
        return f"{float(v) * rnd.uniform(0.5, 1.5):.4f}" if v not in (None, "") else v
    except ValueError:
        return v


def scramble_rows(rows, cut, seed, start_key="start", keep=STAT_KEEP):
    """The stats rows with every row of a match from `cut` (a date) on given another such row's
    numbers (shuffled between them), each scaled at random -- so a total or a league average over
    them changes too, not only which match has which."""
    rnd = random.Random(seed)
    late = [i for i, r in enumerate(rows) if (t := _start(r.get(start_key))) and t.date() >= cut]
    donors = [rows[i] for i in late]
    rnd.shuffle(donors)
    out = list(rows)
    for i, d in zip(late, donors):
        out[i] = dict({k: _jiggle(v, rnd) for k, v in d.items() if k not in keep},
                      **{k: rows[i].get(k) for k in keep})
    return out


def scramble_features(rows, first, last, seed=7):
    """(ok, matches before the cut read the same, matches from it that moved): the stats from the
    window's middle day on shuffled between matches, the daily reading redone."""
    start = {r.get("match_code"): _start(r.get("start")) for r in rows}
    cut = mid_day(first, last)
    a = gamer_features.daily(rows, since=first)
    b = gamer_features.daily(scramble_rows(rows, cut, seed), since=first)
    before = {m for m in a if start.get(m) and start[m].date() < cut}
    ok = all(a[m] == b.get(m) for m in before)
    moved = sum(1 for m in a if m not in before and a[m] != b.get(m))
    return ok, len(before), moved


# --------------------------------------------------------------------------- the controls and the leak check

def write_controls(ctx, arms, out_dir, seed=11):
    """[(name, spec)]: the echo (prod's closing prices as an arm) and the shuffled (the first NB2
    arm's -- else the first arm's -- prices shuffled within each day), both read as normals."""
    out = []
    path = os.path.join(out_dir, "control_echo.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "exp_p1", "exp_p2"])
        for code in ctx["codes"]:
            p = ctx["prod"].get(code) or {}
            if p.get("p1") is None or p.get("line") is None:
                continue
            m, t = rematch_layer.implied_margin(p["p1"]), p["line"]
            w.writerow([code, f"{(t + m) / 2:.6f}", f"{(t - m) / 2:.6f}"])
    out.append((ECHO, f"{path}#normal"))
    source = next(((n, s) for n, s in arms if n.startswith("nb2-")), arms[0] if arms else None)
    if source:
        rows = pb.fit_arm(source[0], source[1], ctx)[2]
        by_day = defaultdict(list)
        for r in rows:
            by_day[r["start"].date()].append(r)
        rnd = random.Random(seed)
        path = os.path.join(out_dir, "control_shuffled.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["MATCH_CODE", "exp_p1", "exp_p2"])
            for day in sorted(by_day):
                day_rows = sorted(by_day[day], key=lambda r: r["match"])
                prices = [(r["margin"], r["total"]) for r in day_rows]
                rnd.shuffle(prices)
                for r, (m, t) in zip(day_rows, prices):
                    w.writerow([r["match"], f"{(t + m) / 2:.6f}", f"{(t - m) / 2:.6f}"])
        out.append((SHUFFLED, f"{path}#normal"))
    return out, (source[0] if source else None)


def scramble_window(window, cut, seed):
    """The window's matches with the finals of those from `cut` (a date) on shuffled between them,
    and each score raised by 0-14 points at random (so their averages change too)."""
    rnd = random.Random(seed)
    late = sorted(c for c, m in window.items() if m["start"].date() >= cut)
    finals = [(window[c].get("home_score"), window[c].get("away_score")) for c in late]
    rnd.shuffle(finals)
    out = dict(window)
    for c, (s1, s2) in zip(late, finals):
        out[c] = dict(window[c], home_score=None if s1 is None else s1 + rnd.randint(0, 14),
                      away_score=None if s2 is None else s2 + rnd.randint(0, 14))
    return out


def leak_check(audit, ctx, name, spec, seed=7):
    """Refit an arm's layers with the results (and the gamers' stats) from the window's middle day on
    shuffled: every earlier match must move as it did; and refit them as they were: every match must."""
    starts = [m["start"] for m in ctx["window"].values()]
    cut = mid_day(min(starts).date(), max(starts).date())
    base = {r["match"]: r for r in pb.fit_arm(name, spec, ctx)[2]}
    feats = ctx.get("features")
    if feats and ctx.get("stat_rows"):
        feats = gamer_features.daily(scramble_rows(ctx["stat_rows"], cut, seed), since=ctx["features_since"])
    shuffled = dict(ctx, window=scramble_window(ctx["window"], cut, seed), features=feats, fitted={})
    again = dict(ctx, fitted={})
    rows_s = {r["match"]: r for r in pb.fit_arm(name, spec, shuffled)[2]}
    rows_a = {r["match"]: r for r in pb.fit_arm(name, spec, again)[2]}
    keys = [k for k in LAYER_KEYS if any(k in r for r in base.values())]
    early = [c for c, r in base.items() if r["start"].date() < cut]
    differ = [c for c in early if c in rows_s and any(base[c].get(k) != rows_s[c].get(k) for k in keys)]
    moved = sum(1 for c, r in base.items() if r["start"].date() >= cut and c in rows_s
                and any(r.get(k) != rows_s[c].get(k) for k in keys))
    audit.add("FAIL" if differ else "PASS", f"{name}: the layers read nothing from a later day",
              f"results{' and stats' if feats else ''} from {cut} on shuffled: "
              + (f"{len(differ):,} of {len(early):,} earlier matches moved differently ({', '.join(differ[:4])})"
                 if differ else f"{len(early):,} earlier matches moved as before, {moved:,} later ones changed"
                 + ("" if moved else " (none: the check is idle)")))
    fit_a, fit_s = pb.fitted_by_day(list(base.values())), pb.fitted_by_day(list(rows_s.values()))
    off = [d for d in fit_a if d <= cut and fit_a[d] != fit_s.get(d)]
    audit.add("FAIL" if off else "PASS", f"{name}: the fitted score model reads nothing from a later day",
              f"{len(off)} days differ" if off else f"{sum(1 for d in fit_a if d <= cut)} days as before")
    same = set(rows_a) == set(base) and all(base[c].get(k) == rows_a[c].get(k) for c in base for k in keys)
    audit.add("PASS" if same else "FAIL", f"{name}: the layers' refit is the same (determinism)",
              f"{len(base):,} matches")


# --------------------------------------------------------------------------- the summary

def _in_period(t, lo, hi):
    if t is None:
        return False
    d = t.date()
    return (lo is None or d >= dt.date.fromisoformat(lo)) and (hi is None or d < dt.date.fromisoformat(hi))


def _num(v):
    try:
        return float(v) if v not in (None, "") else math.nan
    except ValueError:
        return math.nan


def _mean(a):
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else math.nan


def _ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _paired(d):
    d = d[~np.isnan(d)]
    if len(d) < 3:
        return math.nan, math.nan
    sd = d.std(ddof=1)
    return float(d.mean()), (float(d.mean() / (sd / math.sqrt(len(d)))) if sd else math.nan)


def grade_backtest(path, burn_from):
    """(variants, {variant: metrics}, the graded rows): every variant on the matches all of them
    priced from the burn-in, each against prod match by match."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        rows = list(reader)
    variants = [c[:-len("_p_home")] for c in cols if c.endswith("_p_home")]
    rows = [r for r in rows if _start(r["start"]) >= burn_from
            and all(r.get(f"{v}_p_home") not in ("", None) and r.get(f"{v}_total") not in ("", None) for v in variants)]
    margin = np.array([_num(r["margin"]) for r in rows])
    total = np.array([_num(r["total"]) for r in rows])
    line = np.array([_num(r["prod_line"]) for r in rows])
    y = (margin > 0).astype(float)
    ml = margin != 0
    ou_y = (total > line).astype(float)
    per = {}
    for v in variants:
        p = np.array([_num(r[f"{v}_p_home"]) for r in rows])
        t = np.array([_num(r[f"{v}_total"]) for r in rows])
        po = np.array([_num(r[f"{v}_p_over"]) for r in rows])
        ll = np.where(ml, _ll(p, y), np.nan)
        ou = np.where((total != line) & ~np.isnan(po), _ll(po, ou_y), np.nan)
        per[v] = dict(ll=ll, br=np.where(ml, (p - y) ** 2, np.nan), se=(total - t) ** 2, bias=total - t, ou=ou)
    return variants, per, rows


def metrics(per, v, mask=None):
    """One variant's numbers over the rows in `mask` (all by default), against prod."""
    m = per[v]
    base = per[pb.PROD]
    sel = (lambda a: a) if mask is None else (lambda a: a[mask])
    se = _mean(sel(m["se"]))
    out = dict(n=int(np.sum(~np.isnan(sel(m["ll"])))), ll=_mean(sel(m["ll"])), brier=_mean(sel(m["br"])),
               rmse=math.sqrt(se) if not math.isnan(se) else math.nan, bias=_mean(sel(m["bias"])),
               ou=_mean(sel(m["ou"])))
    out["d_ll"], out["t_ll"] = _paired(sel(m["ll"]) - sel(base["ll"]))
    out["d_se"], out["t_se"] = _paired(sel(m["se"]) - sel(base["se"]))
    out["d_ou"], out["t_ou"] = _paired(sel(m["ou"]) - sel(base["ou"]))
    return out


def grade_replay(path, n_boot, seed=0):
    """{variant: {"all": (change, lo, hi), market: change}}: the clean bets' margin at each variant's
    prices less prod's, in points of stake, with a 95% interval resampling matches."""
    if not os.path.exists(path):
        return {}, 0
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        rows = [r for r in reader if r.get("clean") == "True"]
    names = [c[:-len("_revenue")] for c in cols if c.endswith("_revenue") and c != "revenue"]
    if not rows:
        return {}, 0
    stake = np.array([_num(r["stake"]) for r in rows])
    rev = np.array([_num(r["revenue"]) for r in rows])
    codes = sorted({r["match_code"] for r in rows})
    at = {c: i for i, c in enumerate(codes)}
    match = np.array([at[r["match_code"]] for r in rows])
    markets = np.array([r.get("market") or "" for r in rows])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(codes), size=(n_boot, len(codes))) if n_boot else None
    s_m = np.bincount(match, stake, len(codes))
    out = {}
    for n in names:
        cand = np.array([_num(r[f"{n}_revenue"]) for r in rows])
        diff = cand - rev
        change = 100 * float(diff.sum() / stake.sum())
        lo = hi = math.nan
        if draws is not None:
            d_m = np.bincount(match, diff, len(codes))
            x = np.sort(100 * d_m[draws].sum(axis=1) / s_m[draws].sum(axis=1))
            lo, hi = float(x[int(0.025 * len(x))]), float(x[min(len(x) - 1, int(0.975 * len(x)))])
        got = {"all": (change, lo, hi)}
        for mk in sorted(set(markets)):
            sel = markets == mk
            got[mk] = 100 * float(diff[sel].sum() / stake[sel].sum()) if stake[sel].sum() else math.nan
        out[n] = got
    return out, len(rows)


def _f(x, fmt, w):
    return f"{'-':>{w}s}" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:{fmt}}".rjust(w)


def summarize(out_dir, burn_from, periods, controls=(), n_boot=None, shuffled_source=None):
    """Lines and CSV rows: every variant ranked by its moneyline log loss against prod."""
    n_boot = config.SIM_BOOT if n_boot is None else n_boot
    variants, per, rows = grade_backtest(os.path.join(out_dir, "prior_backtest.csv"), burn_from)
    replay, n_bets = grade_replay(os.path.join(out_dir, "prior_replay.csv"), n_boot)
    starts = [_start(r["start"]) for r in rows]
    first_meeting = np.array([r["meeting"] == "1st" for r in rows])
    period_masks = [(label, np.array([_in_period(t, lo, hi) for t in starts])) for label, lo, hi in periods]
    period_masks = [(label, m) for label, m in period_masks if m.any()]
    control_names = {n for n, _ in controls}
    is_control = lambda v: v.split("+")[0] in control_names
    got = {v: metrics(per, v) for v in variants}
    ranked = sorted((v for v in variants if v != pb.PROD and not is_control(v)), key=lambda v: got[v]["d_ll"])
    L = [f"  Backtest suite: {len(rows):,} matches every variant priced, from {burn_from:%Y-%m-%d} "
         f"(the layers' burn-in); {n_bets:,} clean pre-match bets replayed",
         "  ML: the moneyline's log loss (draws out); vs prod: the variant less prod, match by match (negative "
         "is better), t on it; o/u: the over at prod's line (pushes out); RMSE: the expected total's; bets: "
         "the book's margin on the clean bets at the variant's prices less at prod's, points of stake, with "
         "its 95% interval (resampling matches); then the moneyline against prod by meeting and period",
         ""]
    head = (f"  {'#':>3s} {'variant':26s} {'ML':>7s} {'vs prod':>8s} {'t':>5s} {'o/u':>7s} {'vs prod':>8s} "
            f"{'t':>5s} {'RMSE':>6s} {'bets':>6s} {'95%':>15s} {'1st':>8s} {'rematch':>8s}"
            + "".join(f" {label:>{max(8, len(label))}s}" for label, _ in period_masks))
    L.append(head)
    out_rows = []

    def line(k, v):
        m = got[v]
        rp = replay.get(v, {}).get("all", (math.nan, math.nan, math.nan))
        cuts = [metrics(per, v, first_meeting)["d_ll"], metrics(per, v, ~first_meeting)["d_ll"]]
        cuts += [metrics(per, v, mask)["d_ll"] for _, mask in period_masks]
        band = "-" if math.isnan(rp[1]) else f"[{rp[1]:+.2f},{rp[2]:+.2f}]"
        out_rows.append(dict(rank=k, variant=v, matches=m["n"], ml_ll=m["ll"], ml_vs_prod=m["d_ll"], ml_t=m["t_ll"],
                             brier=m["brier"], ou_ll=m["ou"], ou_vs_prod=m["d_ou"], ou_t=m["t_ou"], total_rmse=m["rmse"],
                             total_bias=m["bias"], bets_change=rp[0], bets_lo=rp[1], bets_hi=rp[2],
                             first_meetings_vs_prod=cuts[0], rematches_vs_prod=cuts[1],
                             **{f"{label}_vs_prod": c for (label, _), c in zip(period_masks, cuts[2:])}))
        if v == pb.PROD:
            return (f"  {'':>3s} {v[:26]:26s} {_f(m['ll'], '.4f', 7)} {'-':>8s} {'-':>5s} {_f(m['ou'], '.4f', 7)} "
                    f"{'-':>8s} {'-':>5s} {_f(m['rmse'], '.2f', 6)}")
        return (f"  {k:>3} {v[:26]:26s} {_f(m['ll'], '.4f', 7)} {_f(m['d_ll'], '+.4f', 8)} {_f(m['t_ll'], '+.1f', 5)} "
                f"{_f(m['ou'], '.4f', 7)} {_f(m['d_ou'], '+.4f', 8)} {_f(m['t_ou'], '+.1f', 5)} "
                f"{_f(m['rmse'], '.2f', 6)} {_f(rp[0], '+.2f', 6)} {band:>15s} "
                + " ".join(_f(c, '+.4f', 8) for c in cuts[:2])
                + "".join(f" {_f(c, '+.4f', max(8, len(label)))}" for (label, _), c in zip(period_masks, cuts[2:])))

    L.append(line("", pb.PROD))
    for k, v in enumerate(ranked, 1):
        L.append(line(k, v))
    if replay:
        markets = sorted({k for g in replay.values() for k in g if k != "all"})
        L += ["", "  The clean bets by market (the book's margin at the variant's prices less prod's, points):",
              f"  {'variant':26s}" + "".join(f" {mk[:10]:>10s}" for mk in markets)]
        for v in ranked:
            if v in replay:
                L.append(f"  {v[:26]:26s}" + "".join(f" {_f(replay[v].get(mk), '+.2f', 10)}" for mk in markets))
    checks = control_checks(got, control_names, shuffled_source)
    if control_names:
        L += ["", "  The controls (not candidates: checks on the pipeline):"]
        for v in variants:
            if is_control(v) and "+" not in v:
                L.append(line("", v))
        L += [f"  {c}" for c in checks]
    return L, out_rows, checks


def control_checks(got, control_names, shuffled_source):
    """Lines: PASS or FAIL for each control."""
    out = []
    if ECHO in control_names and ECHO in got:
        e = got[ECHO]
        ok = abs(e["d_ll"]) < ECHO_TOLERANCE and abs(e["d_se"]) < 0.01
        out.append(f"{'PASS' if ok else 'FAIL'}  echo grades as prod: moneyline {e['d_ll']:+.5f}, total's squared "
                   f"error {e['d_se']:+.4f} (within {ECHO_TOLERANCE} and 0.01)")
    if SHUFFLED in control_names and SHUFFLED in got and shuffled_source in got:
        gap = got[SHUFFLED]["ll"] - got[shuffled_source]["ll"]
        ok = gap > SHUFFLED_MARGIN
        out.append(f"{'PASS' if ok else 'FAIL'}  shuffled grades below {shuffled_source}: moneyline log loss "
                   f"{gap:+.4f} worse (at least {SHUFFLED_MARGIN})")
    return out


def write_summary(out_dir, L, out_rows):
    path = os.path.join(out_dir, "suite_summary.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    if out_rows:
        cols = list(out_rows[0])
        for r in out_rows:
            cols += [k for k in r if k not in cols]
        with open(os.path.join(out_dir, "suite_summary.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, cols)
            w.writeheader()
            w.writerows({k: (round(v, 6) if isinstance(v, float) and not math.isnan(v) else
                             "" if isinstance(v, float) else v) for k, v in r.items()} for r in out_rows)
    return path


# --------------------------------------------------------------------------- the run

def run(cur, out_dir, cfg, since, until=None, steps=STEPS, keep_going=False):
    """Run the suite's steps in order. Returns the summary's path (or the last step's output)."""
    os.makedirs(out_dir, exist_ok=True)
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise SystemExit(f"no step {', '.join(unknown)}; the steps are {', '.join(STEPS)}")
    steps = [s for s in STEPS if s in steps]
    first, last = window_days(since, until)
    arms = arm_list(cfg, out_dir)
    features = None if cfg.get("features") == "none" else cfg.get("features")
    man = Manifest(out_dir)
    man.update(config=cfg, since=since, until=until, first_day=first, last_day=last,
               arms=[list(a) for a in arms])
    print(f"  backtest suite: {first} to {last}, arms {', '.join(n for n, _ in arms)}; steps {', '.join(steps)}",
          flush=True)
    audit = Audit()
    state = {}

    def context():
        if "ctx" not in state:
            ctx = pb.prepare(cur, since, until, features)
            controls, source = write_controls(ctx, arms, out_dir) if cfg.get("controls") else ([], None)
            state.update(ctx=ctx, controls=controls, source=source)
            man.update(burn_from=ctx["burn_from"], controls=[list(c) for c in controls], shuffled_source=source)
        return state["ctx"]

    path = None
    for step in steps:
        t0 = time.time()
        print(f"\n== {step}", flush=True)
        if step == "history":
            path = step_history(cur, out_dir)
        elif step == "fits":
            step_fits(cfg, out_dir, first, last)
        elif step == "audit":
            history_path = os.path.join(out_dir, HISTORY)
            if not os.path.exists(history_path):
                audit.add("FAIL", "the history", f"no {history_path}: run the history step")
                window = []
            else:
                from eAMFModel import nb2_prior
                window = audit_history(audit, nb2_prior.load_history(history_path), first, last)
            codes = [r["MATCH_CODE"] for r in window]
            for name, spec in arms:
                source = pb.parse_arm(spec)[0]
                if source.startswith("rolling:"):
                    h = next((x for x in cfg["nb2_half_lives"] if name == f"nb2-hl{float(x):g}"), None)
                    audit_rolling(audit, name, source[len("rolling:"):], first, last, h)
                else:
                    audit_csv_arm(audit, name, spec, codes)
            audit_features(audit, features, window, first, last, cfg["periods"])
        elif step in ("backtest", "replay"):
            ctx = context()
            every = arms + state["controls"]
            if step == "backtest":
                if cfg.get("leak_check") and arms:
                    print("\n  the leak check: the first arm's layers refitted on the results shuffled from the "
                          "window's middle day, and as they were", flush=True)
                    leak_check(audit, ctx, *arms[0])
                path = pb.run(cur, out_dir, every, since, until, cfg["draws"], features, ctx=ctx)
            else:
                path = pr.run(cur, out_dir, every, since, until, cfg["draws"], cfg.get("boot"), features, ctx=ctx)
        elif step == "summary":
            meta = man.data
            burn = _start(str(meta.get("burn_from"))) if meta.get("burn_from") else None
            if burn is None:
                raise SystemExit("no burn-in day in suite.json: run the backtest step first")
            L, out_rows, checks = summarize(out_dir, burn, cfg["periods"], meta.get("controls") or [],
                                            cfg.get("boot"), meta.get("shuffled_source"))
            if audit.lines or os.path.exists(os.path.join(out_dir, AUDIT)):
                tally = audit.tally() if audit.lines else meta.get("audit", {})
                L += ["", f"  The audit: {tally.get('PASS', 0)} passed, {tally.get('WARN', 0)} warnings, "
                          f"{tally.get('FAIL', 0)} failed (suite_audit.txt)"]
            print("\n".join(L))
            path = write_summary(out_dir, L, out_rows)
            man.update(control_checks=checks)
        if audit.lines:
            audit.write(os.path.join(out_dir, AUDIT), [f"  Backtest suite audit, {first} to {last}"])
            man.update(audit=audit.tally())
        man.step(step, time.time() - t0)
        if audit.count["FAIL"] and step in ("audit", "backtest") and not keep_going:
            raise SystemExit(f"the audit found {audit.count['FAIL']} failures ({os.path.join(out_dir, AUDIT)}); "
                             "fix them, or rerun with --keep-going")
    return path
