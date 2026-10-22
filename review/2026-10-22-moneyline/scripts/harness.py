"""Shared moneyline evaluation harness: one data frame, one set of windows, one scorer.

Every model in this study is scored here, on the same matches, so the numbers compare.

    import sys; sys.path.insert(0, ML); import harness as H
    df = H.load()                       # every match, kick-off order, features + prior tables + prod + elo
    pred = H.walk_forward(df, fit, predict, *H.TEST)   # daily refits, nothing from after a day's midnight
    print(H.report(df.assign(mine=pred), ["mine"], window=H.TEST, refs=("prod", "rating")))

Data (all walk-forward already, each match's inputs known at its kick-off):
  features.csv   xgb/features.py with xgb/data/drive_stats.csv: 26,649 matches 2025-12-10 to 2026-10-18,
                 249 features: rolling form, streaks, teams, rest, sessions, head to head, the Kalman form
                 rating (rating_margin, p1_/p2_rating, _var), and from 2026-04-27 the drive stats (p1_drv_off_*,
                 p1_drv_def_*: the h2h site's red zone, 3rd down, yards per play, turnovers ... rebuilt from our
                 own play-by-play, over each gamer's last 15 and 50 matches). prod_p_home = prod's last
                 pre-match moneyline quote (market 50), a scoring benchmark only, NEVER a feature.
  prior tables   eAMFCalibrator's walkthrough (nb2, glmer, avg, ...) and form tables, 2026-09-01 to 10-18:
                 tbl_<name>_margin, tbl_<name>_total, tbl_<name>_p (their own moneyline), tbl_meeting.
  elo            AMFELO_20261019.csv's starting ratings (elo_p1, elo_p2): check for leakage before use.

y = 1 if PLAYER_1 (home) won, 0 if lost; ties (15 of 26,649) are NaN and left out of every score.
"""

import math
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

ROOT = "/home/user/sis"
ML = os.path.dirname(os.path.abspath(__file__))
FEATURES = os.path.join(ML, "features.csv")
HISTORY = os.path.join(ROOT, "eAMFCalibrator/out/match_history.csv")
ELO = os.path.join(ROOT, "AMFELO_20261019.csv")
TABLES = {
    "avg": "eAMFCalibrator/out/walkthrough/priors/avg.csv",
    "avg_plain": "eAMFCalibrator/out/walkthrough/priors/avg-plain.csv",
    "nb2": "eAMFCalibrator/out/walkthrough/priors/nb2.csv",
    "nb2_infer": "eAMFCalibrator/out/walkthrough/priors/nb2_infer.csv",
    "nb2_both": "eAMFCalibrator/out/walkthrough/priors/nb2_both.csv",
    "glmer": "eAMFCalibrator/out/walkthrough/priors/glmer.csv",
    "glmer_both": "eAMFCalibrator/out/walkthrough/priors/glmer_both.csv",
    "form": "eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv",
}
RESULT_LAG = pd.Timedelta(minutes=36)   # a result is in 36 minutes after its kick-off (the stream's)
MARGIN_SD = 9.35                        # rematch_layer's normal about the margin, for margin -> p baselines

# Windows, [start, end) as dates. Tune hyper-parameters on TUNE only; report on TEST (frozen settings).
TUNE = ("2026-06-01", "2026-08-01")
TEST = ("2026-08-01", "2026-10-19")
SEP = ("2026-09-01", "2026-10-19")      # where the walkthrough / form tables exist
DRIVES_FROM = "2026-04-27"              # play-by-play (drive stats, prod's quotes) from here
EPS = 1e-6


def _clip(p):
    return np.clip(np.asarray(p, float), EPS, 1 - EPS)


def logit(p):
    p = _clip(p)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def load(tables=True, elo=True):
    """Every match, kick-off order, with y, day, the features, prod's quote, the prior tables and elo."""
    df = pd.read_csv(FEATURES, low_memory=False)
    df["start"] = pd.to_datetime(df["start"])
    df = df.sort_values(["start", "code"], kind="stable").reset_index(drop=True)
    df["y"] = np.where(df.s1 > df.s2, 1.0, np.where(df.s1 < df.s2, 0.0, np.nan))
    df["margin"] = df.s1 - df.s2
    df["total"] = df.s1 + df.s2
    df["day"] = df.start.dt.strftime("%Y-%m-%d")
    df["prod"] = df["prod_p_home"]
    df["rating"] = _norm_p(df["rating_margin"])     # the form rating's margin read as a normal, sd MARGIN_SD
    if tables:
        for name, rel in TABLES.items():
            t = pd.read_csv(os.path.join(ROOT, rel), usecols=["MATCH_CODE", "meeting", "exp_margin", "exp_total", "p_home"])
            t = t.rename(columns={"MATCH_CODE": "code", "exp_margin": f"tbl_{name}_margin",
                                  "exp_total": f"tbl_{name}_total", "p_home": f"tbl_{name}_p", "meeting": "tbl_meeting"})
            if "tbl_meeting" in df:
                t = t.drop(columns="tbl_meeting")
            df = df.merge(t, on="code", how="left")
    if elo:
        e = pd.read_csv(ELO, usecols=["MATCH_CODE", "PLAYER_1_ELO_STARTING", "PLAYER_2_ELO_STARTING"])
        e = e.rename(columns={"MATCH_CODE": "code", "PLAYER_1_ELO_STARTING": "elo_p1", "PLAYER_2_ELO_STARTING": "elo_p2"})
        df = df.merge(e.drop_duplicates("code"), on="code", how="left")
    return df


def _norm_p(margin, sd=MARGIN_SD):
    from scipy.stats import norm
    return norm.cdf(np.asarray(margin, float) / sd)


def in_window(df, window):
    a, b = window
    return (df.day >= a) & (df.day < b)


def walk_forward(df, fit, predict, start, end, refit_days=1, train_days=None, min_train=200, verbose=False):
    """Predictions (a Series on df's index, NaN outside [start, end)) from daily refits: each day's matches
    priced by a model fitted on the matches settled (RESULT_LAG after kick-off) by that day's midnight UTC,
    optionally only the last train_days of them. fit(train_df) -> model; predict(model, day_df) -> p_home.
    The features themselves already take in results up to each kick-off (xgb/features.py), so a match late
    in the day is priced off the day's earlier results through them, never through the fit."""
    out = pd.Series(np.nan, index=df.index)
    days = sorted(df.loc[(df.day >= start) & (df.day < end), "day"].unique())
    settled_at = df.start + RESULT_LAG
    model = None
    for k, d in enumerate(days):
        if model is None or k % refit_days == 0:
            cut = pd.Timestamp(d)
            tr = settled_at <= cut
            if train_days:
                tr &= df.start >= cut - pd.Timedelta(days=train_days)
            tr &= df.y.notna()
            if tr.sum() < min_train:
                continue
            model = fit(df.loc[tr])
        idx = df.index[df.day == d]
        out.loc[idx] = np.asarray(predict(model, df.loc[idx]), float)
        if verbose and k % 10 == 0:
            print(f"  {d}: trained on {int(tr.sum()):,}", flush=True)
    return out


def calib(y, p):
    """(intercept, slope) of y on logit(p) by maximum likelihood: slope 1 is calibrated, below 1 too
    spread out (the prices should be shrunk toward evens), above 1 too timid."""
    from scipy.optimize import minimize
    x = logit(p)
    y = np.asarray(y, float)

    def nll(b):
        z = b[0] + b[1] * x
        return np.sum(np.logaddexp(0, z) - y * z)

    def grad(b):
        r = sigmoid(b[0] + b[1] * x) - y
        return np.array([r.sum(), (r * x).sum()])

    r = minimize(nll, np.array([0.0, 1.0]), jac=grad, method="BFGS")
    return float(r.x[0]), float(r.x[1])


def auc(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    ranks = pd.Series(p).rank(method="average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logloss_each(y, p):
    p = _clip(p)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def metrics(y, p):
    y = np.asarray(y, float)
    p = _clip(p)
    a, b = calib(y, p)
    return dict(n=len(y), logloss=float(logloss_each(y, p).mean()), brier=float(np.mean((p - y) ** 2)),
                auc=auc(y, p), cal_int=a, cal_slope=b, mean_p=float(p.mean()), mean_y=float(y.mean()))


def boot_diff(df, a, b, n_boot=2000, seed=0):
    """Log loss of column a less column b over the rows of df (both priced, y in), with a 95% interval
    resampling whole days: (mean difference, lo, hi). Negative: a is better."""
    d = df[df.y.notna() & df[a].notna() & df[b].notna()]
    diff = logloss_each(d.y.to_numpy(), d[a].to_numpy()) - logloss_each(d.y.to_numpy(), d[b].to_numpy())
    g = pd.DataFrame({"day": d.day.to_numpy(), "x": diff}).groupby("day")["x"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(s), size=(n_boot, len(s)))
    draws = s[k].sum(1) / c[k].sum(1)
    return float(diff.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def report(df, cols, window=TEST, refs=("prod",), common=True, n_boot=2000):
    """A text table: each column's metrics in the window, on the matches every column (and ref) prices
    when common, and its log loss less each ref's with the day-resampled 95% interval."""
    w = df[in_window(df, window) & df.y.notna()]
    allc = list(dict.fromkeys(list(cols) + [r for r in refs if r in w]))
    if common:
        w = w.dropna(subset=allc)
    lines = [f"{window[0]} to {window[1]}: {len(w):,} matches, {w.day.nunique()} days"
             + (" (priced by every column)" if common else "")]
    hdr = f"  {'model':28s} {'logloss':>8s} {'brier':>7s} {'auc':>6s} {'slope':>6s} {'int':>6s}"
    hdr += "".join(f"   vs {r:>10s} (95%)       " for r in refs)
    lines.append(hdr)
    for c in allc:
        x = w if common else w[w[c].notna()]
        m = metrics(x.y, x[c])
        s = (f"  {c:28s} {m['logloss']:8.4f} {m['brier']:7.4f} {m['auc']:6.3f} {m['cal_slope']:6.2f} "
             f"{m['cal_int']:+6.2f}")
        for r in refs:
            if r == c or r not in x:
                s += " " * 30
                continue
            mu, lo, hi = boot_diff(x, c, r, n_boot=n_boot)
            s += f"   {mu:+.4f} ({lo:+.4f},{hi:+.4f})"
        lines.append(s)
    return "\n".join(lines)


def write_preds(df, col, name, out_dir=None):
    """A model's walk-forward p_home as preds/<name>.csv (code, p_home): every agent writes these, so the
    final comparison and the stacking read them all."""
    out_dir = out_dir or os.path.join(ML, "preds")
    os.makedirs(out_dir, exist_ok=True)
    d = df.loc[df[col].notna(), ["code", col]].rename(columns={col: "p_home"})
    d.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
    return os.path.join(out_dir, f"{name}.csv")


def read_preds(df, out_dir=None):
    """df with every preds/<name>.csv merged in as column pred_<name>."""
    out_dir = out_dir or os.path.join(ML, "preds")
    for f in sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []:
        if f.endswith(".csv"):
            p = pd.read_csv(os.path.join(out_dir, f)).rename(columns={"p_home": f"pred_{f[:-4]}"})
            df = df.merge(p, on="code", how="left")
    return df
