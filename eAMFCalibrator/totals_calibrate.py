"""The points still to come, calibrated: a correction on top of a line's own points to come, fitted
on what the rest of the match really made.

Every line read so far -- prod's and v8/v9's -- spreads its points to come wider than the rest of the
match does, and gives a slow start more to come when it really produces less (`totals-signals`).
This fits, at the end of Q1, at half time and at the end of Q3:

    rest of the match = a + b x the line's points to come + c x points on the board + ...

on the `scouting` PLAY_OVER export (months of matches, no Snowflake), the earlier part of the matches
to fit and the latest part to test, and reports on the held-out part:

  - each coefficient; a line that priced everything takes b = 1 and leaves the rest at 0;
  - the points to come against the result (MAE), and P(over prod's line) against it (Brier), prod
    against the calibrated line. The calibrated P(over) reads the fit's own residuals, so a lumpy
    total (3, 7, 10, 14 ...) keeps its shape;
  - points to come by points on the board, real against prod's and the calibrated.

It writes the coefficients and residual quantiles to totals_calibration.json, to put in front of a
model's points to come. With --signals (a totals_signals.csv with v8/v9 in it), it fits the same
correction on each model's own points to come at the start of Q2 and Q3, cross-validated (that file
is a fortnight), so the fix can be read for the models as well as prod.
"""

import csv
import json
import math
import os
import random
from bisect import bisect_right
from collections import defaultdict

from .totals_signals import _f, _mean_se, edges_for, bin_of, ols

CHECKPOINTS = (("q1", 1, "end of Q1"), ("h1", 2, "half time"), ("q3", 3, "end of Q3"))
QUARTER = 240
FEATURES = (("needed", "prod's points to come"), ("points", "points on the board"),
            ("prematch", "prod's closing pre-match line"), ("plays", "scrimmage plays"),
            ("margin", "score margin"), ("secs_left", "game-clock seconds left"),
            ("form", "both gamers' recent form"))
SIMPLE = ("needed",)
TEST_SHARE = 1 / 3


def _num(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def match_states(rows):
    """{checkpoint: state} for one match's export rows: the last PLAY_OVER of the quarter with a
    live prod total, its board, prod's line and P(over), the plays so far, the margin and the
    game clock left in the match; with the final and the pre-match line."""
    rows = sorted(rows, key=lambda r: int(float(r["message"])))
    f1, f2 = _num(rows[0].get("final_p1")), _num(rows[0].get("final_p2"))
    if f1 is None or f2 is None:
        return {}
    final = f1 + f2
    out = {}
    plays = 0
    last = {}
    for r in rows:
        period = _num(r.get("period"))
        if period is None:
            continue
        if _num(r.get("scrimmage")) == 1:
            plays += 1
        line = _num(r.get("line_54"))
        if line is None or _num(r.get("live_54")) != 1:
            continue
        p1, p2 = _num(r.get("score_p1")) or 0.0, _num(r.get("score_p2")) or 0.0
        clock = _num(r.get("clock_seconds")) or 0.0
        last[int(period)] = dict(line=line, prob=_num(r.get("prob_54")), points=p1 + p2,
                                 margin=abs(p1 - p2), plays=plays,
                                 secs_left=(4 - int(period)) * QUARTER + clock)
    for name, period, _ in CHECKPOINTS:
        s = last.get(period)
        if s is None:
            continue
        s = dict(s, needed=s["line"] - s["points"], rest=final - s["points"], final=final,
                 prematch=_num(rows[0].get("prematch_line_54")))
        out[name] = s
    return out


def read_export(path, form=None):
    """[(match, first file time, {checkpoint: state})] off a scouting_playover.csv, read a match at a
    time (the export writes each match's rows together). `form` (match -> (form, long-run)) adds
    the gamers' recent form."""
    out = []
    cur, rows = None, []

    def flush():
        if rows:
            states = match_states(rows)
            if states:
                f = (form or {}).get(cur)
                for s in states.values():
                    s["form"] = f[0] if f else 0.0
                    s["match_code"] = cur
                out.append((cur, min(r.get("file_time") or "" for r in rows), states))

    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["match_code"] != cur:
                flush()
                cur, rows = r["match_code"], []
            rows.append(r)
    flush()
    return out


def design(s, features):
    return [1.0] + [float(s[k]) for k in features]


def usable(s, features):
    return s.get("rest") is not None and all(s.get(k) is not None for k in features)


class Fit:
    """One checkpoint's correction: the coefficients, and the residuals the probabilities read."""

    def __init__(self, train, features):
        self.features = features
        X = [design(s, features) for s in train]
        y = [s["rest"] for s in train]
        fit = ols(X, y)
        if fit is None:
            raise ValueError("singular fit")
        self.beta, self.se = fit
        self.residuals = sorted(yy - self.predict(s) for s, yy in zip(train, y))

    def predict(self, s):
        return sum(b * x for b, x in zip(self.beta, design(s, self.features)))

    def p_over(self, s, needed):
        """P(rest of the match > needed): the share of the residuals above needed - prediction."""
        cut = needed - self.predict(s)
        return 1 - bisect_right(self.residuals, cut) / len(self.residuals)

    def to_json(self):
        qs = [self.residuals[min(len(self.residuals) - 1, int(q * len(self.residuals)))]
              for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)]
        return dict(features=list(self.features), beta=self.beta, se=self.se, n=len(self.residuals),
                    residual_quantiles=dict(zip(("5", "10", "25", "50", "75", "90", "95"), qs)))


def brier(rows, prob):
    """Mean squared error of P(over prod's line) against the result, pushes left out."""
    xs = [(prob(s) - (1.0 if s["final"] > s["line"] else 0.0)) ** 2 for s in rows
          if s["final"] != s["line"] and prob(s) is not None]
    return sum(xs) / len(xs) if xs else None


def evaluate(train, test, label):
    """Lines of text and the fits for one checkpoint: the coefficients, then prod against the
    simple (points to come alone) and the full correction on the held-out matches."""
    full = []
    for k, _ in FEATURES:
        have = [s[k] for s in train if s.get(k) is not None]
        if len(have) < 0.95 * len(train) or len(set(have)) < 2:
            continue
        try:                                        # a feature that duplicates others is left out
            Fit([s for s in train if usable(s, full + [k])], full + [k])
            full.append(k)
        except ValueError:
            pass
    train_f = [s for s in train if usable(s, full)]
    test_f = [s for s in test if usable(s, full)]
    L = [f"\n  ==== {label}: fitted on {len(train_f):,} matches, tested on {len(test_f):,} ===="]
    if len(train_f) < 50 or len(test_f) < 20:
        return L + ["  too few matches"], {}
    fits = {"simple": Fit(train_f, SIMPLE), "full": Fit(train_f, full)}
    titles = dict(FEATURES)
    for name, fit in fits.items():
        L.append(f"  {name} fit: const {fit.beta[0]:+.2f}  " + "  ".join(
            f"{titles[k]} {b:+.3f}+-{2 * e:.3f}" for k, b, e in zip(fit.features, fit.beta[1:], fit.se[1:])))
    L.append(f"  held out: {'':10s} {'over':>7s} {'+-2se':>6s} {'MAE':>6s} {'Brier':>7s}   "
             "(over: the rest of the match less each line's points to come)")
    rows = [("prod", lambda s: s["needed"], lambda s: s["prob"])]
    rows += [(n, fit.predict, (lambda f: lambda s: f.p_over(s, s["needed"]))(fit))
             for n, fit in fits.items()]
    for who, pred, prob in rows:
        ov = [s["rest"] - pred(s) for s in test_f]
        m, se = _mean_se(ov)
        L.append(f"  {who:20s} {m:+7.2f} {_f(se, '.2f', 6)} {sum(map(abs, ov)) / len(ov):6.2f} "
                 f"{_f(brier(test_f, prob), '.4f', 7)}")
    edges = edges_for([s["points"] for s in test_f])
    L.append("  points to come by points on the board, held out (real against each line's)")
    L.append(f"  {'points':>21s} {'matches':>7s} {'real':>6s} {'prod':>6s} {'simple':>7s} {'full':>6s}")
    for q in sorted({bin_of(s["points"], edges) for s in test_f}):
        b = [s for s in test_f if bin_of(s["points"], edges) == q]
        pts = [s["points"] for s in b]
        avg = lambda f: sum(f(s) for s in b) / len(b)
        L.append(f"  {min(pts):10.0f}-{max(pts):<10.0f} {len(b):7,d} {avg(lambda s: s['rest']):6.1f} "
                 f"{avg(lambda s: s['needed']):6.1f} {avg(fits['simple'].predict):7.1f} "
                 f"{avg(fits['full'].predict):6.1f}")
    return L, fits


def split(matches, share=TEST_SHARE):
    """(train, test) by time: the latest `share` of the matches held out."""
    matches = sorted(matches, key=lambda m: m[1])
    cut = int(len(matches) * (1 - share))
    return matches[:cut], matches[cut:]


def report_export(matches):
    """Lines of text and {checkpoint: {fit: Fit}} for the export."""
    train, test = split(matches)
    L = [f"\n  {len(matches):,} matches: fitted on {len(train):,} to {train[-1][1][:10] if train else '-'}, "
         f"held out {len(test):,} from {test[0][1][:10] if test else '-'}"]
    out = {}
    for name, _, label in CHECKPOINTS:
        tr = [m[2][name] for m in train if name in m[2]]
        te = [m[2][name] for m in test if name in m[2]]
        lines, fits = evaluate(tr, te, label)
        L += lines
        out[name] = fits
    return L, out


# --- the models' own points to come, off totals_signals.csv ---------------------------------------

def read_signals(path):
    """totals_signals.csv rows, numbers as floats."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.append({k: (_num(v) if k != "match_code" else v) for k, v in r.items()})
    return out


def signals_models(rows):
    keys = list(rows[0]) if rows else []
    return [k[len("h1_"):-len("_needed")] for k in keys if k.startswith("h1_") and k.endswith("_needed")
            and k != "h1_needed"]


def report_signals(rows, folds=5, seed=0):
    """Lines of text: at the start of Q2 and Q3, each line's points to come (prod's, each model's)
    corrected by rest = a + b x its points to come + c x points on the board, five-fold
    cross-validated, against the line as it stands."""
    names = signals_models(rows)
    L = [f"\n  ==== the same correction on totals_signals.csv ({len(rows):,} matches), five-fold "
         "cross-validated: each line as it stands against it corrected ===="]
    for seg, label in (("q1", "start of Q2"), ("h1", "start of Q3")):
        L.append(f"\n  {label}")
        L.append(f"  {'':22s} {'MAE':>6s} {'Brier':>7s} {'b':>6s}   (b: the weight the fit puts on the "
                 "line's own points to come; 1 is a line that needs no correction)")
        for who in ["prod"] + names:
            need = f"{seg}_needed" if who == "prod" else f"{seg}_{who}_needed"
            prob = f"{seg}_prob" if who == "prod" else f"{seg}_{who}_pover"
            ss = []
            for r in rows:
                if None in (r.get(need), r.get(f"{seg}_points"), r.get("final_total"), r.get(f"{seg}_line"),
                            r.get(prob)):
                    continue
                ss.append(dict(needed=r[need], points=r[f"{seg}_points"], rest=r["final_total"] - r[f"{seg}_points"],
                               final=r["final_total"], line=r[f"{seg}_line"], prob=r[prob],
                               prod_needed=r[f"{seg}_line"] - r[f"{seg}_points"]))
            if len(ss) < 50:
                continue
            idx = list(range(len(ss)))
            random.Random(seed).shuffle(idx)
            pred, pover, bs = {}, {}, []
            for k in range(folds):
                test = set(idx[k::folds])
                train = [ss[i] for i in idx if i not in test]
                try:
                    fit = Fit(train, ("needed", "points"))
                except ValueError:                  # the line's points to come follow the board
                    fit = Fit(train, ("needed",))
                bs.append(fit.beta[1])
                for i in test:
                    pred[i] = fit.predict(ss[i])
                    pover[i] = fit.p_over(ss[i], ss[i]["prod_needed"])
            raw_mae = sum(abs(s["rest"] - s["needed"]) for s in ss) / len(ss)
            cal_mae = sum(abs(ss[i]["rest"] - pred[i]) for i in range(len(ss))) / len(ss)
            raw_b = brier(ss, lambda s: s["prob"])
            cal = [dict(s, p=pover[i]) for i, s in enumerate(ss)]
            cal_b = brier(cal, lambda s: s["p"])
            L.append(f"  {who + ' as it stands':22s} {raw_mae:6.2f} {_f(raw_b, '.4f', 7)}")
            L.append(f"  {who + ' corrected':22s} {cal_mae:6.2f} {_f(cal_b, '.4f', 7)} "
                     f"{sum(bs) / len(bs):6.2f}")
    return L


def run(export_path, out_dir, history=None, signals=None):
    """Fit and test the correction on the export (and on totals_signals.csv), print it and write
    totals_calibration.json."""
    L = []
    fits = {}
    if export_path and os.path.exists(export_path):
        form = None
        if history:
            from .totals_signals import form_by_match
            with open(history, newline="", encoding="utf-8") as fh:
                form = form_by_match(list(csv.DictReader(fh)))
        matches = read_export(export_path, form)
        print(f"  {len(matches):,} matches with a live prod total at a checkpoint in {export_path}")
        lines, fits = report_export(matches)
        L += lines
    elif export_path:
        L.append(f"\n  no export at {export_path}: run `python -m eAMFCalibrator scouting --since ...` first")
    if signals:
        L += report_signals(read_signals(signals))
    print("\n".join(L))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "totals_calibration.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({cp: {n: f.to_json() for n, f in fs.items()} for cp, fs in fits.items()}, fh, indent=1)
    with open(os.path.join(out_dir, "totals_calibration.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path
