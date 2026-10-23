"""Agent F's shared code: the panel (harness.load + drive stats), sparse gamer/team designs, a walk-forward ridge,
the scoring of an expected total and of a total distribution.

All fits are walk-forward: a day's matches are priced by a fit on the matches settled (kick-off + 36 min) by that
day's midnight UTC (harness.walk_forward's cut), the same cut coded here so that many settings run fast.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse

ML = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml"
ROOT = "/home/user/sis"
OUT = os.path.join(ML, "out", "F")
sys.path.insert(0, ML)
import harness as H  # noqa: E402

RESULT_LAG = H.RESULT_LAG
WINDOWS = {"TUNE": H.TUNE, "TEST": H.TEST, "SEP": H.SEP, "LATE": ("2026-09-22", "2026-10-19"),
           "JUNJUL": H.TUNE}


def td_fg(final, fg_seen):
    """pace_eff's: (touchdowns, field goals) a side scored: the field goals the feed saw, the rest as touchdowns."""
    if final is None or np.isnan(final):
        return np.nan, np.nan
    fg = fg_seen if not np.isnan(fg_seen) else 0.0
    rest = max(final - 3.0 * fg, 0.0)
    td, r = divmod(rest, 7.0)
    if r >= 5:
        td += 1
    elif r >= 3:
        fg += 1
    return td, fg


def panel(rebuild=False):
    """harness.load() (cached) + drive stats: D (both sides' drives), h_/a_drives, td/fg reconstructed per side."""
    path = os.path.join(OUT, "panel.pkl")
    if os.path.exists(path) and not rebuild:
        return pd.read_pickle(path)
    cache = os.path.join(OUT, "df.pkl")
    df = pd.read_pickle(cache) if os.path.exists(cache) else H.load()
    ds = pd.read_csv(os.path.join(ROOT, "xgb/data/drive_stats.csv"))
    keep = ["match_code", "h_drives", "a_drives", "h_fg", "a_fg", "h_td", "a_td", "h_seconds", "a_seconds",
            "h_snaps", "a_snaps", "h_turnover", "a_turnover", "h_punt", "a_punt", "h_half_end", "a_half_end"]
    ds = ds[keep].rename(columns={"h_td": "h_td_feed", "a_td": "a_td_feed"})
    df = df.merge(ds, left_on="code", right_on="match_code", how="left").drop(columns="match_code")
    df["D"] = df.h_drives + df.a_drives
    for s, f in (("h", "s1"), ("a", "s2")):
        tf = [td_fg(x, y) for x, y in zip(df[f].astype(float), df[f"{s}_fg"].astype(float))]
        df[f"{s}_tdx"] = [t for t, _ in tf]
        df[f"{s}_fgx"] = [g for _, g in tf]
        df.loc[df.D.isna(), [f"{s}_tdx", f"{s}_fgx"]] = np.nan
    df["settled"] = df.start + RESULT_LAG
    df["over"] = np.where(df.prod_total_line.notna(), (df.total > df.prod_total_line).astype(float), np.nan)
    df = df.copy()
    df.to_pickle(path)
    return df


# ------------------------------------------------------------------ designs

class Index:
    def __init__(self, df):
        self.g = {g: i for i, g in enumerate(sorted(set(df.p1.dropna()) | set(df.p2.dropna())))}
        self.t = {t: i for i, t in enumerate(sorted(set(df.t1.dropna()) | set(df.t2.dropna())))}
        self.s = {s: i for i, s in enumerate(sorted(df.stream.dropna().unique()))}
        self.ng, self.nt, self.ns = len(self.g), len(self.t), len(self.s)


def _onehot(n, idx_list, width):
    rows, cols = [], []
    for base, idx in idx_list:
        ok = ~np.isnan(idx)
        rows.append(np.flatnonzero(ok))
        cols.append(base + idx[ok].astype(int))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, width))


def match_design(df, ix, stream=False, teams=True):
    """[1 | streams 2.. | gamers (p1 + p2) | teams (t1 + t2)] for match-level targets (total, drives)."""
    n = len(df)
    p = 1 + ((ix.ns - 1) if stream else 0)
    width = p + ix.ng + (ix.nt if teams else 0)
    parts = [(0, np.zeros(n))]
    if stream:
        s = df.stream.map(ix.s).to_numpy(float)
        s = np.where(s == 0, np.nan, s)    # the first stream is the base
        parts.append((0, s))
    for c in ("p1", "p2"):
        parts.append((p, df[c].map(ix.g).to_numpy(float)))
    if teams:
        for c in ("t1", "t2"):
            parts.append((p + ix.ng, df[c].map(ix.t).to_numpy(float)))
    X = _onehot(n, parts, width)
    pen_kind = np.array(["fixed"] * p + ["gamer"] * ix.ng + (["team"] * ix.nt if teams else []))
    return X, pen_kind


def side_design(df, ix, stream=False):
    """Long format, home rows then away rows: [1 | home | streams | off gamer | def gamer | off team | def team]."""
    n = len(df)
    p = 2 + ((ix.ns - 1) if stream else 0)
    width = p + 2 * ix.ng + 2 * ix.nt
    g1, g2 = df.p1.map(ix.g).to_numpy(float), df.p2.map(ix.g).to_numpy(float)
    t1, t2 = df.t1.map(ix.t).to_numpy(float), df.t2.map(ix.t).to_numpy(float)
    mats = []
    for home, (go, gd, to, td) in ((1, (g1, g2, t1, t2)), (0, (g2, g1, t2, t1))):
        parts = [(0, np.zeros(n))]
        if home:
            parts.append((1, np.zeros(n)))
        if stream:
            s = df.stream.map(ix.s).to_numpy(float)
            s = np.where(s == 0, np.nan, s)
            parts.append((1, s))
        parts += [(p, go), (p + ix.ng, gd), (p + 2 * ix.ng, to), (p + 2 * ix.ng + ix.nt, td)]
        mats.append(_onehot(n, parts, width))
    X = sparse.vstack(mats).tocsr()
    pen_kind = np.array(["fixed"] * p + ["gamer"] * (2 * ix.ng) + ["team"] * (2 * ix.nt))
    return X, pen_kind


def penalty(pen_kind, lam_g, lam_t, lam_fixed=1e-4):
    return np.where(pen_kind == "gamer", lam_g, np.where(pen_kind == "team", lam_t, lam_fixed)).astype(float)


def ridge_solve(X, y, w, pen):
    Xw = X.multiply(w[:, None]).tocsr()
    A = (X.T @ Xw).toarray() + np.diag(pen)
    b = X.T @ (w * y)
    return np.linalg.solve(A, b)


def decay(age_days, hl):
    if hl is None or not np.isfinite(hl):
        return np.ones_like(age_days)
    return np.power(2.0, -age_days / hl)


def days_between(df, start, end):
    return sorted(df.loc[(df.day >= start) & (df.day < end), "day"].unique())


def walk_ridge(df, X, y, pen, start, end, hl=60.0, train_days=180, extra_w=None, level_hl=None, nrep=1,
               min_train=200, refit_days=1, pred_X=None, return_coefs=False):
    """Walk-forward ridge: for each day in [start, end), fit on rows (of X / y, nrep stacked copies of df's rows:
    the side designs) whose match settled by the day's midnight, within train_days, y not NaN, weights
    2^(-age/hl) (x extra_w) scaled to mean 1; predict the day's rows of pred_X (default X).
    level_hl: refit the intercept (column 0) on the decayed residual mean at that half-life (league level)."""
    pred_X = X if pred_X is None else pred_X
    N = len(df)
    settled = np.tile(df.settled.values, nrep)
    starts = np.tile(df.start.values, nrep)
    dayv = np.tile(df.day.values, nrep)
    yv = np.asarray(y, float)
    ok_y = ~np.isnan(yv)
    ew = np.ones(len(yv)) if extra_w is None else np.asarray(extra_w, float)
    ok_y &= ~np.isnan(ew)
    out = np.full(pred_X.shape[0], np.nan)
    coefs = {}
    beta = None
    for k, d in enumerate(days_between(df, start, end)):
        cut = np.datetime64(pd.Timestamp(d))
        if beta is None or k % refit_days == 0:
            tr = ok_y & (settled <= cut)
            if train_days:
                tr &= starts >= cut - np.timedelta64(int(train_days * 86400), "s")
            if tr.sum() // nrep < min_train:
                continue
            age = (cut - starts[tr]).astype("timedelta64[s]").astype(float) / 86400.0
            w = decay(age, hl) * ew[tr]
            w = w / w.mean()
            Xt = X[np.flatnonzero(tr)]
            beta = ridge_solve(Xt, yv[tr], w, pen)
            if level_hl:
                res = yv[tr] - Xt @ beta
                wl = decay(age, level_hl) * ew[tr]
                beta = beta.copy()
                beta[0] += np.average(res, weights=wl)
            if return_coefs:
                coefs[d] = beta.copy()
        rows = np.flatnonzero(dayv == d)
        out[rows] = pred_X[rows] @ beta
    return (out, coefs) if return_coefs else out


# ------------------------------------------------------------------ scoring of an expected total

def total_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    slope = np.polyfit(p, y, 1)[0] if p.std() > 0 else np.nan
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                bias=float(np.mean(p - y)), slope=float(slope), r=float(np.corrcoef(p, y)[0, 1]),
                sd_pred=float(p.std()))


def boot_days(day, x, n_boot=2000, seed=0):
    """Mean of x with a 95% interval resampling whole days."""
    g = pd.DataFrame({"day": day, "x": x}).groupby("day")["x"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(s), size=(n_boot, len(s)))
    draws = s[k].sum(1) / c[k].sum(1)
    return float(np.mean(x)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def total_table(df, cols, window, ref=None, common=True, target="total"):
    a, b = window
    w = df[(df.day >= a) & (df.day < b) & df[target].notna()]
    if common:
        w = w.dropna(subset=list(cols))
    lines = [f"{a} to {b}: {len(w):,} matches, {w.day.nunique()} days" + (" (common)" if common else "")]
    hdr = f"  {'model':30s}{'n':>6s}{'RMSE':>8s}{'MAE':>8s}{'bias':>7s}{'slope':>7s}{'r':>7s}{'sd':>6s}"
    if ref:
        hdr += f"   sq err vs {ref} (95% days)"
    lines.append(hdr)
    for c in cols:
        x = w if common else w[w[c].notna()]
        m = total_metrics(x[target], x[c])
        s = (f"  {c:30s}{m['n']:6d}{m['rmse']:8.3f}{m['mae']:8.3f}{m['bias']:+7.2f}{m['slope']:7.2f}{m['r']:7.3f}"
             f"{m['sd_pred']:6.2f}")
        if ref and c != ref:
            d = (x[target] - x[c]) ** 2 - (x[target] - x[ref]) ** 2
            mu, lo, hi = boot_days(x.day.values, d.values)
            s += f"   {mu:+7.3f} ({lo:+.3f},{hi:+.3f})"
        lines.append(s)
    return "\n".join(lines)


def ou_frame(df):
    """df with y := over prod's line (NaN where no line), for harness.report / boot_diff on O/U log loss."""
    d = df.copy()
    d["y"] = d["over"]
    d["prod_over"] = d["prod_total_prob"]
    return d
