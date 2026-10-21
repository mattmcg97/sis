"""The pre-match walkthrough: the best pre-match variants priced for every match of a span, walk-forward,
one table of expected points per variant, so a sim can be pointed at any of them (eAMFModel's
table_prior.py) instead of pricing the pre-match every time it's tested.

    python -m eAMFCalibrator walkthrough --history eAMFCalibrator/out/match_history.csv --since 2026-01-01

The steps, each with a progress line (done, the pace, the time left) and the run's estimated finish -- on a
terminal one line drawn over itself (eAMFModel/progress.py), sent to a file a line every LINE_EVERY seconds:
  nb2 fits     NB2 refitted every day (eAMFModel prior-daily: each fit on the results before its day)
  nb2 prices   every match priced by its own day's fit (nb2-history's nb2.csv)
  glmer fits   glmer refitted every day (glmer/cadence.R, trim:hl60_x0.5:daily: the trimmed model, each
               fit on the results in by its midnight); Rscript must be on the PATH or in $RSCRIPT
  pace         with the scouting export (--scouting): each match's expected drives, a ridge on the gamers'
               and teams' pace refitted every day (pace_prior.py), for v21's pace nudge
  layers       the rematch and inference layers fitted walk-forward on each model's own errors
  variants     each variant's expected points for every match, and its moneyline off them
  report       every variant graded against NB2 on the results (prior-offline's tables)
A step whose output is already there for the same history and span is kept (the fits day by day).

The variants (VARIANTS), each a table in priors/:
  nb2            NB2 as it is
  nb2+infer      NB2 moved by the inference layer (each gamer's form that session)
  nb2+both       NB2 moved by the rematch layer at a rematch, else the inference layer
  glmer          glmer as it is
  glmer+both     glmer moved as nb2+both
  avg            nb2+infer and glmer+both, their expected points averaged side by side (the best
                 pre-match model in prior-offline's tests)
  avg-plain      nb2 and glmer averaged
A layer moves the expected margin and total; a side's points are (total +- margin) / 2. Each table
also has the variant's moneyline: NB2's own joint score model for nb2 and the averages (each side
negative binomial with its gamers' dispersions, a shared gamma pace factor; the average's points
with NB2's dispersions), NB2's construction fitted to glmer's errors for glmer; the layered
variants read their model's book at the layer's shift, as prior-backtest does. Ties split half and
half. Every price was made before its match as it would have been live: the fits on the days
before, the layers on results in by kick-off (36 minutes after a match's kick-off). With the scouting
export every table also has the match's expected drives (exp_drives) and its day's league drives
(league_drives), the pace v21-drives meets (pace_prior.py); without it they're blank.

Point a sim at a variant with its table:
    python -m eAMFCalibrator report --candidate v21 --v21-prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv
    python -m eAMFModel v21-report ... --prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv

Writes (to --out, default out/walkthrough/): priors/<variant>.csv, nb2.csv, nb2_fits/, glmer/, pace/
(the drives counted and each match's pace), walkthrough.txt (the grading) and walkthrough.json (the
span, the inputs, the variants' files and each step's time).
"""

import csv
import datetime as dt
import json
import os
import re
import subprocess
import threading
import time

from . import nb2_history, prior_backtest as pb, prior_offline, score_model
from .bet_totals import _start

GLMER_ARM = "trim:hl60_x0.5:daily"
VARIANTS = (                     # name, ("model", its model, its layer) or ("average", two variants)
    ("nb2", ("model", "nb2", None)),
    ("nb2+infer", ("model", "nb2", "infer")),
    ("nb2+both", ("model", "nb2", "both")),
    ("glmer", ("model", "glmer", None)),
    ("glmer+both", ("model", "glmer", "both")),
    ("avg", ("average", "nb2+infer", "glmer+both")),
    ("avg-plain", ("average", "nb2", "glmer")),
)
LAYER_KEYS = {"infer": ("infer_margin", "infer_total"), "both": ("both_margin", "both_total"),
              "layer": ("move_margin", "move_total")}
DRAWS = 20000
# the first estimate of each step's pace, before the run has its own (seconds)
EST_NB2_FIT = 45.0               # a day's two NB2 fits on one worker, for 24,000 matches of history
NB2_CONTENTION = 1.5             # each worker slower with the others running beside it
EST_NB2_PRICE = 1.0              # a day's matches priced by its fit
EST_GLMER_FIT = 6.0              # a day's glmer fit on one core
EST_LAYERS = 0.002               # a match, each model
EST_VARIANTS = 0.016             # a match, every variant
EST_DRIVES = 0.004               # a match's drives counted off the scouting export
EST_PACE = 0.03                  # a day's pace fit
LINE_EVERY = 15.0                # seconds between progress lines sent to a file (a fit's line always prints)
OVERLAY_EVERY = 0.5              # seconds between redraws of the line on a terminal


def _fmt(seconds):
    seconds = max(0, int(round(seconds)))
    h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


class Progress:
    """The run's plan -- each step's estimated seconds -- and each step's progress line: done of total,
    its pace, the time it and the whole run have left, and the run's finish. A finished step's own
    pace against its estimate rescales the steps still to come. Printed to a terminal, the progress
    line is drawn over itself (`overlay`, progress.Line); elsewhere, or with `out` given, it's a line
    every LINE_EVERY seconds."""

    def __init__(self, plan, clock=time.time, now=dt.datetime.now, out=None, overlay=None):
        self.plan = dict(plan)               # step -> estimated seconds
        self.actual, self.estimated = {}, dict(plan)
        self.clock, self.now = clock, now
        self.out = out or (lambda line: print(line, flush=True))
        if overlay is None and out is None:
            from eAMFModel import progress
            overlay = progress.Line()
        self.overlay = overlay if overlay is not None and overlay.live else None
        self.t0 = clock()
        self.lock = threading.Lock()
        self.step = None

    def factor(self):
        """How the steps done ran against their estimates (bounded): the machine's pace."""
        est = sum(self.estimated[s] for s in self.actual if self.estimated[s] > 0)
        act = sum(self.actual[s] for s in self.actual if self.estimated[s] > 0)
        return min(5.0, max(0.2, act / est)) if est > 1 else 1.0

    def rest(self, after):
        """The estimated seconds of the steps after `after` that haven't run."""
        steps = list(self.plan)
        later = steps[steps.index(after) + 1:] if after in steps else steps
        return self.factor() * sum(self.plan[s] for s in later if s not in self.actual)

    def finish_text(self, left):
        return (self.now() + dt.timedelta(seconds=left)).strftime("%H:%M")

    def begin(self, step, label, total, unit, estimate=None, parallel=1):
        """A step starts: `total` units, `parallel` of them run at once (until that many are done,
        the estimate stands, since the first finish together)."""
        with self.lock:
            if estimate is not None:
                self.plan[step] = self.estimated[step] = estimate
            self.step, self.label, self.total, self.unit = step, label, total, unit
            self.parallel = max(1, parallel)
            self.done, self.ts, self.last = 0, self.clock(), 0.0
            est = self.factor() * self.plan.get(step, 0.0)
            left = est + self.rest(step)
            self._clear()
            self.out(f"\n== {label}: {total:,} {unit}, about {_fmt(est)}; the whole run about {_fmt(left)} more, "
                     f"finishing about {self.finish_text(left)}")

    def tick(self, n=1, note="", force=False):
        with self.lock:
            self.done += n
            now = self.clock()
            if self.overlay is not None:             # a terminal: the same line, redrawn
                if force or self.done >= self.total or now - self.last >= OVERLAY_EVERY:
                    self.last = now
                    self.overlay.draw(self.line(now, note))
            elif force or self.done >= self.total or now - self.last >= LINE_EVERY:
                self.last = now
                self.out(self.line(now, note))

    def _clear(self):
        if self.overlay is not None:
            self.overlay.clear()

    def line(self, now, note=""):
        elapsed = now - self.ts
        if self.done:
            # side by side, the first to finish took as long as each does, and the rest were running
            # beside it: until a round is done, the pace is `parallel` per that time
            left = elapsed / max(self.done, min(self.parallel, self.total)) * (self.total - self.done)
        else:
            left = max(0.0, self.factor() * self.plan[self.step] - elapsed)
        run_left = left + self.rest(self.step)
        pct = 100 * self.done / self.total if self.total else 100
        return (f"  {self.label}: {self.done:,}/{self.total:,} {self.unit} ({pct:.0f}%), {_fmt(elapsed)} in, "
                f"~{_fmt(left)} left; the run ~{_fmt(run_left)} left, finishing about {self.finish_text(run_left)}"
                + (f"  [{note}]" if note else ""))

    def end(self, skipped=False):
        with self.lock:
            took = self.clock() - self.ts
            # a step kept from before ran in no time: it says nothing of the machine's pace
            self.actual[self.step] = took
            if skipped:
                self.estimated[self.step] = 0.0
            self._clear()
            self.out(f"  {self.label}: {'kept from before' if skipped else 'done'} in {_fmt(took)}; "
                     f"{_fmt(self.clock() - self.t0)} since the start")


# --------------------------------------------------------------------------- the steps

def _stamp_ok(path, stamp):
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as fh:
        return json.load(fh) == stamp


def _write_stamp(path, stamp):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=1)


def history_stamp(history, first, last):
    newest = max((r.get("SCHEDULED_START_TIME_UTC") or "" for r in history), default="")
    return {"matches": len(history), "newest": newest[:19], "first": first.isoformat(), "last": last.isoformat()}


def nb2_fits(history, out_dir, first, last, workers, prog, half_life=None):
    from eAMFModel import nb2_prior, rolling_prior
    directory = os.path.join(out_dir, "nb2_fits")
    days = [first + dt.timedelta(days=k) for k in range((last - first).days + 1)]
    todo = [d for d in days if not nb2_prior.Prematch.exists(os.path.join(directory, d.isoformat()))]
    size = len(history) / 24000.0
    runs = max(1, min(workers, len(todo) or 1))
    prog.begin("nb2_fits", "NB2 fits", len(todo), "days to fit",
               len(todo) * EST_NB2_FIT * max(size, 0.2) * NB2_CONTENTION / runs, parallel=runs)
    if todo:
        rolling_prior.fit(history, "nb2", directory, first, last, workers=workers, verbose=False,
                          options={} if half_life is None else {"half_life": half_life},
                          on_fit=lambda day, s: prog.tick(note=f"{day} {s:.0f}s", force=True))
    prog.end(skipped=not todo)
    return directory


def nb2_prices(history, out_dir, fits, first, last, prog, draws):
    path = os.path.join(out_dir, "nb2.csv")
    stamp_path = os.path.join(out_dir, "nb2.json")
    stamp = dict(history_stamp(history, first, last), draws=draws)
    rows = [r for r in history if (t := _start(r.get("SCHEDULED_START_TIME_UTC"))) and first <= t.date() <= last
            and nb2_history._i(r.get("PLAYER_1_FINAL_SCORE")) is not None]
    days = len({_start(r["SCHEDULED_START_TIME_UTC"]).date() for r in rows})
    if _stamp_ok(stamp_path, stamp) and os.path.exists(path):
        prog.begin("nb2_prices", "NB2 prices", days, "days", 0.0)
        prog.end(skipped=True)
        return path
    prog.begin("nb2_prices", "NB2 prices", days, "days", days * EST_NB2_PRICE)
    priced = nb2_history.price_rows(fits, rows, history, draws,
                                    progress=lambda done, total: prog.tick(note=f"{done}/{total} fits"))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, nb2_history.COLUMNS)
        w.writeheader()
        w.writerows(priced)
    _write_stamp(stamp_path, stamp)
    prog.end()
    return path


CADENCE_TOTAL = re.compile(r"\] (\S+): (\d+) fits in (\d+) stretch")
CADENCE_LINE = re.compile(r"\] (\S+ \d+/\d+): (\d+)/(\d+) fits")
CADENCE_DONE = re.compile(r"\] (\S+ \d+/\d+): done, (\d+) fits")


def glmer_fits(history_path, history, out_dir, first, last, cores, prog, runner=None):
    """cadence.R's daily trimmed glmer over the span: the arm's source (PATH@PREFIX)."""
    directory = os.path.join(out_dir, "glmer")
    os.makedirs(directory, exist_ok=True)
    matches = os.path.join(directory, "cadence_matches.csv")
    stamp_path = os.path.join(out_dir, "glmer.json")
    stamp = dict(history_stamp(history, first, last), arm=GLMER_ARM)
    days = (last - first).days + 1
    source = f"{matches}@{GLMER_ARM}"
    if _stamp_ok(stamp_path, stamp) and os.path.exists(matches):
        prog.begin("glmer", "glmer fits", days, "days", 0.0)
        prog.end(skipped=True)
        return source
    cores = cores or max(1, (os.cpu_count() or 2) - 1)
    runs = max(1, min(cores, -(-days // 14)))       # cadence.R runs a daily arm in fortnights, side by side
    prog.begin("glmer", "glmer fits", days, "days", days * EST_GLMER_FIT / runs, parallel=runs)
    from eAMFModel import glmer_prior
    args = [glmer_prior.rscript(), os.path.join(glmer_prior.GLMER_DIR, "cadence.R"), f"--history={history_path}",
            f"--test-from={first.isoformat()}", f"--test-until={(last + dt.timedelta(days=1)).isoformat()}",
            f"--arms={GLMER_ARM}", f"--cores={cores}", "--progress=1", f"--out={directory}"]
    done_by, tail = {}, []

    def on_line(line):
        tail.append(line)
        del tail[:-30]
        m = CADENCE_TOTAL.search(line)
        if m:
            with prog.lock:
                prog.total = int(m.group(2))
            return
        m = CADENCE_DONE.search(line) or CADENCE_LINE.search(line)
        if m:
            where, n = m.group(1), int(m.group(2))
            step = n - done_by.get(where, 0)
            done_by[where] = n
            if step > 0:
                prog.tick(step, note=where)

    code = (runner or run_streaming)(args, on_line)
    if code != 0 or not os.path.exists(matches):
        raise SystemExit("glmer/cadence.R failed:\n" + "\n".join(tail[-15:]))
    _write_stamp(stamp_path, stamp)
    prog.end()
    return source


def run_streaming(args, on_line):
    """Run a command, handing each line of its output (stdout and stderr) to on_line; its exit code."""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)
    for line in proc.stdout:
        on_line(line.rstrip("\n"))
    return proc.wait()


def pace_step(history, export, out_dir, first, last, prog):
    """Each match's (expected drives, its day's league drives): pace_prior's daily ridge on the drives
    counted off the export (kept in pace/ while the export is the same)."""
    from . import pace_prior
    days = (last - first).days + 1
    prog.begin("pace", "the pace", days, "days", len(history) * EST_DRIVES + days * EST_PACE)

    def counted(n):
        if n % 500 == 0:
            prog.tick(0, note=f"{n:,} matches' drives counted")

    pace, drives = pace_prior.run(history, export, first, last, os.path.join(out_dir, "pace"), on_match=counted,
                                  on_day=lambda day: prog.tick(note=str(day)),
                                  log=lambda line: (prog._clear(), prog.out(line)))
    prog.end()
    return pace, drives


def fit_layers(ctx, arms, prog):
    n = len(ctx["codes"])
    prog.begin("layers", "the layers", len(arms), "models", len(arms) * n * EST_LAYERS)
    fitted = {}
    for name, spec in arms:
        fitted[name] = pb.fit_arm(name, spec, ctx)
        prog.tick(note=name, force=True)
    prog.end()
    return fitted


def score_of(book, own, fitted):
    """(the score model's name, its parameters) behind a model's book: NB2's own (the two gamers'
    dispersions, the shared fraction), NB2's construction fitted to the model's errors (one
    dispersion, the shared fraction), or the normal."""
    if isinstance(book, score_model.Normal):
        return "normal", ()
    if own:
        return own[0], tuple(own[1][2:])
    if fitted and fitted[0] is not None:
        return "nb2", (fitted[0], fitted[0], fitted[1])
    return "normal", ()


def _points(margin, total):
    return max(0.1, (total + margin) / 2), max(0.1, (total - margin) / 2)


def variant_rows(ctx, fitted, prog, draws=DRAWS, pace=None):
    """{variant: {match: row}}: every variant's expected points and moneyline for every match both
    models priced, and with `pace` ({match: (expected drives, league drives)}) its pace."""
    by = {name: {r["match"]: r for r in rows} for name, (_, _, rows, _) in fitted.items()}
    fits = {name: pb.fitted_by_day(rows) if mode == "fitted" else {} for name, (mode, _, rows, _) in fitted.items()}
    codes = sorted(set.intersection(*(set(v) for v in by.values())), key=lambda c: by["nb2"][c]["start"])
    prog.begin("variants", "the variants", len(codes), "matches", len(codes) * EST_VARIANTS)
    out = {name: {} for name, _ in VARIANTS}
    burn = ctx["burn_from"]
    for k, c in enumerate(codes, 1):
        m = ctx["window"][c]
        books, pts = {}, {}
        for name, (mode, own, _, _) in fitted.items():
            r = by[name][c]
            books[name] = pb.score_book(r, mode, own, fits[name], draws)
        nb2_own = fitted["nb2"][1].get(c)
        for name, spec in VARIANTS:
            if spec[0] == "model":                            # a model, as it is or layered
                _, model, layer = spec
                r = by[model][c]
                dm, dt_ = (r.get(LAYER_KEYS[layer][0], 0.0), r.get(LAYER_KEYS[layer][1], 0.0)) if layer else (0.0, 0.0)
                margin, total = r["margin"] + dm, r["total"] + dt_
                h, a = _points(margin, total)
                p = books[model].p_home(dm)
                kind, params = score_of(books[model], fitted[model][1].get(c), fits[model].get(r["start"].date()))
            else:                                             # an average of two variants' points
                x, y = pts[spec[1]], pts[spec[2]]
                h, a = (x[0] + y[0]) / 2, (x[1] + y[1]) / 2
                margin, total, dm, dt_ = h - a, h + a, None, None
                if nb2_own:
                    bk = score_model.book("nb2", (h, a) + tuple(nb2_own[1][2:]), c, draws)
                    kind, params = "nb2", tuple(nb2_own[1][2:])
                else:
                    bk, kind, params = score_model.Normal(margin, total), "normal", ()
                p = bk.p_home()
            pts[name] = (h, a)
            out[name][c] = dict(
                MATCH_CODE=c, start=m["start"].strftime("%Y-%m-%d %H:%M:%S"), home=m["home"], away=m["away"],
                home_team=m["team"].get(m["home"]), away_team=m["team"].get(m["away"]), meeting=m["k"],
                home_score=m.get("home_score"), away_score=m.get("away_score"),
                home_pts=round(h, 4), away_pts=round(a, 4), Pred_P1_Points=round(h, 4), Pred_P2_Points=round(a, 4),
                exp_margin=round(margin, 4), exp_total=round(total, 4), p_home=round(p, 5), score_model=kind,
                score_params=" ".join(f"{x:.4f}" for x in params if x is not None),
                layer_margin=None if dm is None else round(dm, 4), layer_total=None if dt_ is None else round(dt_, 4),
                burn_in=m["start"] < burn,
                exp_drives=round(pace[c][0], 3) if pace and c in pace else None,
                league_drives=round(pace[c][1], 3) if pace and c in pace else None)
        if k % 250 == 0 or k == len(codes):
            prog.tick(250 if k % 250 == 0 else k % 250)
    prog.end()
    return out


TABLE_COLUMNS = ["MATCH_CODE", "start", "home", "home_team", "away", "away_team", "meeting", "home_score",
                 "away_score", "home_pts", "away_pts", "Pred_P1_Points", "Pred_P2_Points", "exp_margin", "exp_total",
                 "p_home", "score_model", "score_params", "layer_margin", "layer_total", "burn_in", "exp_drives",
                 "league_drives"]


def file_name(variant):
    return variant.replace("+", "_") + ".csv"


def write_tables(out_dir, variants):
    d = os.path.join(out_dir, "priors")
    os.makedirs(d, exist_ok=True)
    paths = {}
    for name, rows in variants.items():
        path = os.path.join(d, file_name(name))
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, TABLE_COLUMNS)
            w.writeheader()
            w.writerows(rows[c] for c in rows)
        paths[name] = path
    return paths


def report(ctx, variants, burn_from, prog):
    prog.begin("report", "the report", 1, "report", 5.0)
    prices = {name: {c: (r["p_home"], r["exp_total"], None) for c, r in rows.items()} for name, rows in variants.items()}
    window = ctx["window"]
    results = {c: (m["home_score"] - m["away_score"], m["home_score"] + m["away_score"], None)
               for c, m in window.items() if m.get("home_score") is not None}
    L, _ = prior_offline.report(prices, results, window, burn_from, "nb2", set(), {}, {})
    # the per-arm notes and layer fits prior-offline adds are the caller's; keep the tables
    L = [x for x in L if not x.startswith("\n  How each arm") and not x.startswith("\n  The layers' last fits")]
    prog.tick(force=True)
    prog.end()
    return L


def run(out_dir, history_path, since, until=None, workers=4, cores=None, draws=DRAWS, half_life=None,
        cadence_runner=None, scouting=None):
    """The whole walkthrough (with `scouting`, the scouting export, each match's pace too). Returns
    walkthrough.json's path."""
    from eAMFModel import nb2_prior
    os.makedirs(out_dir, exist_ok=True)
    history = nb2_prior.load_history(history_path)
    settled = [t for t in (_start(r.get("SCHEDULED_START_TIME_UTC")) for r in history
                           if nb2_history._i(r.get("PLAYER_1_FINAL_SCORE")) is not None) if t]
    if not settled:
        raise SystemExit(f"{history_path}: no settled matches")
    first = _start(since).date()
    newest = max(settled)
    last = newest.date()
    if until:
        last = min(last, (_start(until) - dt.timedelta(seconds=1)).date())
    if last < first:
        raise SystemExit(f"{history_path} ends {newest:%Y-%m-%d %H:%M}, before {first}")
    days = (last - first).days + 1
    n_matches = sum(1 for t in settled if first <= t.date() <= last)
    print(f"  the walkthrough: {first} to {last} ({days} days, {n_matches:,} settled matches) off {history_path} "
          f"({len(history):,} matches, to {newest:%Y-%m-%d %H:%M}); {len(VARIANTS)} variants -> {out_dir}", flush=True)
    size = len(history) / 24000.0
    cores_ = cores or max(1, (os.cpu_count() or 2) - 1)
    plan = {"nb2_fits": days * EST_NB2_FIT * max(size, 0.2) * NB2_CONTENTION / max(1, workers),
            "nb2_prices": days * EST_NB2_PRICE,
            "glmer": days * EST_GLMER_FIT / max(1, min(cores_, -(-days // 14))),
            "layers": 2 * n_matches * EST_LAYERS, "variants": n_matches * EST_VARIANTS, "report": 5.0}
    if scouting:
        plan = dict(list(plan.items())[:3] + [("pace", len(history) * EST_DRIVES + days * EST_PACE)]
                    + list(plan.items())[3:])
    prog = Progress(plan)
    print(f"  first estimate: about {_fmt(sum(plan.values()))} in all (the fits already made are kept, so a rerun "
          f"is quicker); finishing about {prog.finish_text(sum(plan.values()))}", flush=True)
    fits = nb2_fits(history, out_dir, first, last, workers, prog, half_life)
    nb2_csv = nb2_prices(history, out_dir, fits, first, last, prog, draws)
    arms = [("nb2", nb2_csv),
            ("glmer", glmer_fits(os.path.abspath(history_path), history, os.path.abspath(out_dir), first, last, cores,
                                 prog, cadence_runner))]
    pace = drives = None
    if scouting:
        pace, drives = pace_step(history, scouting, out_dir, first, last, prog)
    ctx = prior_offline.prepare(history_path, f"{first} 00:00:00", f"{last + dt.timedelta(days=1)} 00:00:00")
    fitted = fit_layers(ctx, arms, prog)
    variants = variant_rows(ctx, fitted, prog, draws, pace)
    paths = write_tables(out_dir, variants)
    L = report(ctx, variants, ctx["burn_from"], prog)
    head = [f"  The walkthrough, {first} to {last}: each variant's expected points for every match, priced as live "
            f"(the layers move nothing before {ctx['burn_from']:%Y-%m-%d}, their burn-in)", ""]
    head += [f"  {name:12s} {len(variants[name]):7,d} matches  -> {paths[name]}" for name, _ in VARIANTS]
    head += ["", "  Point a sim at one: --v21-prior-model <the table> (eAMFCalibrator), --prior-model <the table> "
                 "(eAMFModel v21-report)", ""]
    if pace is not None:
        from . import pace_prior
        head += pace_prior.lines(pace, drives) + [f"  (the pace off {scouting}: v21-drives reads exp_drives off "
                                                  "the table it's given)", ""]
    else:
        head += ["  no pace: exp_drives blank (--scouting <scouting_playover.csv> for v21-drives)", ""]
    L = head + L + ["", "  How each model's moneyline is read:"]
    L += [f"  {name}: {pb.describe_model(mode, own, pb.fitted_by_day(rows) if mode == 'fitted' else {})}"
          for name, (mode, own, rows, _) in fitted.items()]
    L += pb.layer_lines({name: f[3] for name, f in fitted.items()})
    print("\n".join(L))
    with open(os.path.join(out_dir, "walkthrough.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    timings = {s: round(t, 1) for s, t in prog.actual.items()}
    manifest = dict(first=first.isoformat(), last=last.isoformat(), history=os.path.abspath(history_path),
                    history_matches=len(history), burn_in_until=ctx["burn_from"].strftime("%Y-%m-%d"),
                    glmer_arm=GLMER_ARM, half_life=half_life, draws=draws,
                    scouting=os.path.abspath(scouting) if scouting else None,
                    pace_matches=len(pace) if pace is not None else 0,
                    variants={name: dict(file=paths[name], spec=list(spec)) for name, spec in VARIANTS},
                    seconds=timings, finished=dt.datetime.now().isoformat(timespec="seconds"))
    path = os.path.join(out_dir, "walkthrough.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)
    print(f"\n  the walkthrough took {_fmt(time.time() - prog.t0)}", flush=True)
    return path
