"""Agent D's shared pieces: feature sets, the plain and paired designs, the early-stopped XGBoost fit,
the walk-forward driver (harness.walk_forward underneath: nothing settled after a day's midnight UTC),
the simple baselines (logistic link on rating_margin, a time-decayed ridge strength) and walk-forward
recalibration / links fitted on previous days' out-of-sample predictions only.

Run from /home/user/sis:  python <ML>/scripts/D/<script>.py
"""

import os
import sys
import time

ML = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml"
sys.path.insert(0, ML)

import harness as H  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

OUT = os.path.join(ML, "out", "D")
os.makedirs(OUT, exist_ok=True)
NTHREAD = 1      # 2 OpenMP threads ran 15x slower than 1 on the oversubscribed box (spin-waits)
RUN_FROM = "2026-04-01"      # walk-forward from here so later links / recalibrations have OOS history
RUN_TO = "2026-10-19"

IDENT = {"code", "start", "p1", "p2", "t1", "t2", "s1", "s2"}
RATING = ["p1_rating", "p2_rating", "p1_rating_var", "p2_rating_var", "rating_margin"]
CONTEXT = ["hour", "weekday", "stream"]
FORM_SIDE = ["games", "pf_all", "pa_all", "win_all", "pf_5", "pa_5", "margin_5", "total_5", "pf_15", "pa_15",
             "margin_15", "total_15", "pf_50", "pa_50", "margin_50", "total_50", "ewma_pf", "ewma_pa",
             "pf_sd_15", "margin_sd_15", "pf_max_10", "pf_min_10", "win_3", "win_10"]
FORM_MATCH = [f"exp_{k}_{w}" for w in ("5", "15", "50", "all") for k in ("margin", "total")]
H2H = ["h2h_n", "h2h_margin", "h2h_total", "h2h_wins_5"]
# match-level columns that change sign when the sides are swapped
FLIP = {"rating_margin", "h2h_margin"} | {f"exp_margin_{w}" for w in ("5", "15", "50", "all")}


def load():
    df = H.load()
    return df


def all_features(df):
    """Every features.csv feature: no identity, no result, no prod_, none of the harness extras."""
    feats = pd.read_csv(H.FEATURES, nrows=1).columns
    return [c for c in feats if c not in IDENT and not c.startswith("prod_")]


def group_of(c):
    """Feature group for importance tables (works for p1_/p2_/own_/opp_/d_ names)."""
    base = c
    for pre in ("p1_", "p2_", "own_", "opp_", "d_"):
        if base.startswith(pre):
            base = base[len(pre):]
            break
    if "drv_" in base:
        return "drives"
    if base in ("rating", "rating_var", "rating_margin", "rating_lr", "base"):
        return "rating"
    if base in ("hour", "weekday", "stream", "home", "hour_sin", "hour_cos") or base.startswith("stream_"):
        return "context"
    if base.startswith("h2h_"):
        return "h2h"
    if base in ("team_games", "team_pf", "team_pa", "teamlg_pf", "teamlg_pa") or base in ("t1", "t2", "team"):
        return "team"
    if base in ("rest_hours", "game_of_session", "game_of_day", "session_settled", "session_pf", "session_pa",
                "session_win"):
        return "rest/session"
    if base in ("p1", "p2", "gamer", "opp"):
        return "identity"
    return "form"


def feature_set(df, name):
    """Named feature subsets (all of them features.csv columns)."""
    allf = all_features(df)
    drv = [c for c in allf if "_drv_" in c]
    if name == "all":
        return allf
    if name == "nodrv":
        return [c for c in allf if c not in drv]
    if name == "ratings":
        return list(RATING)
    if name == "form":
        return [f"{s}_{k}" for s in ("p1", "p2") for k in FORM_SIDE] + FORM_MATCH
    if name == "ratings_form":
        return list(RATING) + [f"{s}_{k}" for s in ("p1", "p2") for k in FORM_SIDE] + FORM_MATCH
    raise ValueError(name)


def side_names(feats):
    return sorted({c[3:] for c in feats if c.startswith("p1_")})


# ----------------------------------------------------------------------------------------------- designs
def plain_matrix(df, feats, diffs=False, ident=False):
    """One row a match from the home side's view. Context stays ordinal (trees split hour / weekday /
    stream at any cut; stream has 3 levels). diffs adds p1_x - p2_x for every side feature."""
    X = df[feats].astype("float32").copy()
    if diffs:
        for c in side_names(feats):
            X[f"d_{c}"] = (df[f"p1_{c}"] - df[f"p2_{c}"]).astype("float32")
    if ident:
        cats = sorted(set(df.p1) | set(df.p2))
        X["p1"] = pd.Categorical(df.p1, categories=cats)
        X["p2"] = pd.Categorical(df.p2, categories=cats)
    return X


def paired_matrix(df, feats, diffs=False, ident=False):
    """Two row blocks: the home view (own = p1, home = 1) and the away view (own = p2, home = 0), with
    match features oriented to the 'own' side. The away view's label is 1 - y."""
    sides = side_names(feats)
    match_cols = [c for c in feats if not c.startswith(("p1_", "p2_"))]
    views = []
    for own, opp, home in (("p1", "p2", 1), ("p2", "p1", 0)):
        x = {"home": np.full(len(df), home, dtype="float32")}
        for c in sides:
            x[f"own_{c}"] = df[f"{own}_{c}"].to_numpy("float32")
            x[f"opp_{c}"] = df[f"{opp}_{c}"].to_numpy("float32")
            if diffs:
                x[f"d_{c}"] = (df[f"{own}_{c}"] - df[f"{opp}_{c}"]).to_numpy("float32")
        for c in match_cols:
            v = df[c].to_numpy("float32")
            if c in FLIP and not home:
                v = -v
            if c == "h2h_wins_5" and not home:      # the away side's wins among the last <=5 meetings
                v = (np.minimum(df.h2h_n.to_numpy("float32"), 5) - v).astype("float32")
            x[c] = v
        X = pd.DataFrame(x, index=df.index)
        if ident:
            cats = sorted(set(df.p1) | set(df.p2))
            X["gamer"] = pd.Categorical(df[own], categories=cats)
            X["opp"] = pd.Categorical(df[opp], categories=cats)
        views.append(X)
    return views[0], views[1]


# ----------------------------------------------------------------------------------------------- fitting
DEFAULT = dict(max_depth=3, eta=0.03, min_child_weight=50, subsample=0.8, colsample_bytree=0.5,
               reg_lambda=10.0, reg_alpha=0.0, gamma=0.0)


def xgb_params(p, objective="binary:logistic"):
    out = dict(objective=objective, tree_method="hist", nthread=NTHREAD, verbosity=0, seed=0)
    out.update(p)
    if objective == "binary:logistic":
        out.setdefault("eval_metric", "logloss")
    elif objective.startswith("reg:pseudohuber"):
        out.setdefault("eval_metric", "mphe")
    else:
        out.setdefault("eval_metric", "rmse")
    return out


def dmat(X, y=None, base=None, weight=None):
    cat = any(isinstance(X[c].dtype, pd.CategoricalDtype) for c in X.columns)
    return xgb.DMatrix(X, label=y, base_margin=base, weight=weight, enable_categorical=cat, nthread=NTHREAD)


def fit_es(X, y, start, params, objective="binary:logistic", valid_days=14, max_rounds=3000, es=100,
           base=None, min_rounds=10, refit=True):
    """Early stopping on the window's last valid_days (by kick-off), then refit on the whole window with
    the best number of trees. Returns (booster, rounds, probe_valid_pred_info)."""
    start = pd.to_datetime(pd.Series(np.asarray(start)))
    vcut = start.max() - pd.Timedelta(days=valid_days)
    va = (start > vcut).to_numpy()
    tr = ~va
    P = xgb_params(params, objective)
    d_tr = dmat(X[tr], y[tr], None if base is None else base[tr])
    d_va = dmat(X[va], y[va], None if base is None else base[va])
    probe = xgb.train(P, d_tr, max_rounds, evals=[(d_va, "v")], early_stopping_rounds=es, verbose_eval=False)
    rounds = max(min_rounds, probe.best_iteration + 1)
    if base is not None and probe.best_iteration == 0:
        rounds = 0
    if not refit:
        return probe, rounds
    if rounds == 0:
        return None, 0
    full = xgb.train(P, dmat(X, y, base), rounds)
    return full, rounds


class Log:
    def __init__(self, path):
        self.path = path
        self.fh = open(path, "a", encoding="utf-8")

    def __call__(self, s):
        print(s, flush=True)
        self.fh.write(s + "\n")
        self.fh.flush()


def wf_xgb(df, Xh, params, start, end, design="plain", Xa=None, objective="binary:logistic", target="y",
           refit_days=3, train_days=120, valid_days=14, base_h=None, base_a=None, keep_models=False,
           verbose=False, contribs_from=None):
    """Walk-forward XGBoost. design 'plain': Xh rows are matches (home view). 'paired': Xh / Xa are the
    home / away views; the model is trained on both and p(home) = (p_h + 1 - p_a) / 2 (classifier) or
    margin = (m_h - m_a) / 2 (regressor). base_*: logit offsets (classifier) on the same rows.
    Returns (pred Series, info dict with rounds per refit and kept boosters)."""
    info = {"rounds": [], "models": [], "fit_days": [], "secs": 0.0, "gain": {}, "contribs": []}
    t0 = time.time()
    is_cls = objective == "binary:logistic"

    def fit(tr):
        idx = tr.index
        if target == "y":
            yv = tr.y.to_numpy(float)
        else:
            yv = tr[target].to_numpy(float)
        st = tr.start.to_numpy()
        if design == "plain":
            X, Y, S = Xh.loc[idx], yv, st
            B = None if base_h is None else base_h.loc[idx].to_numpy(float)
        else:
            X = pd.concat([Xh.loc[idx], Xa.loc[idx]], ignore_index=True)
            Y = np.concatenate([yv, 1 - yv if is_cls else -yv])
            S = np.concatenate([st, st])
            B = None if base_h is None else np.concatenate([base_h.loc[idx].to_numpy(float),
                                                             base_a.loc[idx].to_numpy(float)])
        m, r = fit_es(X, Y, S, params, objective, valid_days=valid_days, base=B)
        info["rounds"].append(r)
        info["fit_days"].append(str(tr.start.max())[:10])
        if m is not None:
            for k, v in m.get_score(importance_type="total_gain").items():
                info["gain"][k] = info["gain"].get(k, 0.0) + v
        if keep_models:
            info["models"].append((str(tr.start.max())[:10], m))
        return m

    def pred_view(m, X, base):
        if m is None:
            out = base if base is not None else np.zeros(len(X))
            return H.sigmoid(out) if is_cls else out
        return m.predict(dmat(X, base=base))

    def predict(m, d):
        idx = d.index
        bh = None if base_h is None else base_h.loc[idx].to_numpy(float)
        ph = pred_view(m, Xh.loc[idx], bh)
        if contribs_from is not None and m is not None and d.day.iloc[0] >= contribs_from:
            c = m.predict(dmat(Xh.loc[idx], base=bh), pred_contribs=True)
            if design == "paired":
                ba_ = None if base_a is None else base_a.loc[idx].to_numpy(float)
                c = (c - m.predict(dmat(Xa.loc[idx], base=ba_), pred_contribs=True)) / 2
            info["contribs"].append(pd.DataFrame(c, index=idx, columns=list(Xh.columns) + ["BIAS"]))
        if design == "plain":
            return ph
        ba = None if base_a is None else base_a.loc[idx].to_numpy(float)
        pa = pred_view(m, Xa.loc[idx], ba)
        return (ph + 1 - pa) / 2 if is_cls else (ph - pa) / 2

    pred = H.walk_forward(df, fit, predict, start, end, refit_days=refit_days, train_days=train_days,
                          verbose=verbose)
    info["secs"] = time.time() - t0
    return pred, info


# ----------------------------------------------------------------------------------------------- simple models
def fit_logit(x, y, w=None, l2=1e-6):
    """Logistic regression of y on columns of x (intercept added), optional weights. Returns coefs."""
    x = np.column_stack([np.ones(len(y)), np.asarray(x, float).reshape(len(y), -1)])
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)

    def nll(b):
        z = x @ b
        return np.sum(w * (np.logaddexp(0, z) - y * z)) + l2 * np.sum(b[1:] ** 2)

    def grad(b):
        r = H.sigmoid(x @ b) - y
        g = x.T @ (w * r)
        g[1:] += 2 * l2 * b[1:]
        return g

    b0 = np.zeros(x.shape[1])
    return minimize(nll, b0, jac=grad, method="BFGS").x


def apply_logit(b, x):
    x = np.column_stack([np.ones(len(x)), np.asarray(x, float).reshape(len(x), -1)])
    return H.sigmoid(x @ b)


def wf_link(df, xcol, start, end, train_days=120, refit_days=1):
    """Walk-forward logistic link p = sigmoid(a + b x) fitted on the settled matches of the last train_days
    (x itself is a walk-forward feature, e.g. rating_margin)."""
    def fit(tr):
        ok = tr[xcol].notna()
        return fit_logit(tr.loc[ok, xcol].to_numpy(), tr.loc[ok, "y"].to_numpy())

    def predict(b, d):
        return apply_logit(b, d[xcol].fillna(0).to_numpy())

    return H.walk_forward(df, fit, predict, start, end, refit_days=refit_days, train_days=train_days)


def wf_on_oos(df, x, start, end, kind="platt", hist_days=60, min_n=300):
    """Walk-forward map fitted on previous days' out-of-sample values only: for each day d, matches
    settled by d's midnight with kick-off in the last hist_days and x present.
      kind 'platt': p' = sigmoid(a + b logit p)          (x is p)
      kind 'temp' : p' = sigmoid(b logit p)              (x is p; a temperature, no intercept)
      kind 'link' : p  = sigmoid(a + b x)                (x is a predicted margin)
    Returns a Series (NaN where there was too little history)."""
    x = pd.Series(np.asarray(x, float), index=df.index)
    z = H.logit(x.to_numpy()) if kind in ("platt", "temp") else x.to_numpy()
    z = pd.Series(z, index=df.index)
    out = pd.Series(np.nan, index=df.index)
    settled = df.start + H.RESULT_LAG
    days = sorted(df.loc[(df.day >= start) & (df.day < end), "day"].unique())
    for d in days:
        cut = pd.Timestamp(d)
        h = (settled <= cut) & (df.start >= cut - pd.Timedelta(days=hist_days)) & df.y.notna() & x.notna()
        idx = df.index[(df.day == d) & x.notna()]
        if h.sum() < min_n or not len(idx):
            continue
        if kind == "temp":
            yy, zz = df.y[h].to_numpy(), z[h].to_numpy()
            r = minimize(lambda b: np.sum(np.logaddexp(0, b[0] * zz) - yy * b[0] * zz), np.array([1.0]),
                         method="BFGS")
            out.loc[idx] = H.sigmoid(r.x[0] * z[idx].to_numpy())
        else:
            b = fit_logit(z[h].to_numpy(), df.y[h].to_numpy())
            out.loc[idx] = apply_logit(b, z[idx].to_numpy())
    return out


def wf_ridge_strength(df, start, end, half_life=45.0, lam=20.0, train_days=None, refit_days=1, team=False):
    """Time-decayed ridge strengths on the margin: margin = h + s[p1] - s[p2], rows weighted
    0.5 ** (age / half_life) (age in days from the cut), L2 penalty lam on the strengths. Refitted at each
    day's midnight UTC on the matches settled by then (RESULT_LAG after kick-off); returns the predicted
    margin for each match of [start, end). Same cut as harness.walk_forward, written sparse for speed."""
    import scipy.sparse as sp
    gamers = sorted(set(df.p1) | set(df.p2))
    gi = {g: i for i, g in enumerate(gamers)}
    G = len(gamers)
    i1 = df.p1.map(gi).to_numpy()
    i2 = df.p2.map(gi).to_numpy()
    n = len(df)
    A = sp.csr_matrix((np.r_[np.ones(n), np.ones(n), -np.ones(n)],
                       (np.r_[np.arange(n), np.arange(n), np.arange(n)], np.r_[np.zeros(n, int), 1 + i1, 1 + i2])),
                      shape=(n, 1 + G))
    yv = df.margin.to_numpy(float)
    ok = df.y.notna().to_numpy()
    settled = (df.start + H.RESULT_LAG).to_numpy()
    st = df.start.to_numpy()
    out = pd.Series(np.nan, index=df.index)
    days = sorted(df.loc[(df.day >= start) & (df.day < end), "day"].unique())
    beta = None
    P = np.eye(1 + G) * lam
    P[0, 0] = 1e-6
    for k, d in enumerate(days):
        cut = np.datetime64(pd.Timestamp(d))
        if beta is None or k % refit_days == 0:
            tr = ok & (settled <= cut)
            if train_days:
                tr &= st >= cut - np.timedelta64(int(train_days * 86400), "s")
            if tr.sum() >= 200:
                rows = np.flatnonzero(tr)
                age = (cut - st[rows]) / np.timedelta64(1, "D")
                w = 0.5 ** (age / half_life)
                Ar = A[rows]
                M = (Ar.T @ sp.diags(w) @ Ar).toarray() + P
                beta = np.linalg.solve(M, Ar.T @ (w * yv[rows]))
        if beta is None:
            continue
        idx = np.flatnonzero((df.day == d).to_numpy())
        out.iloc[idx] = beta[0] + beta[1 + i1[idx]] - beta[1 + i2[idx]]
    return out


def ll(df, col, window):
    w = df[H.in_window(df, window) & df.y.notna() & df[col].notna()]
    return float(H.logloss_each(w.y.to_numpy(), w[col].to_numpy()).mean()), len(w)


def save_series(df, s, name):
    """A walk-forward series keyed by match code (out/D/series_<name>.csv)."""
    pd.DataFrame({"code": df.code, "v": np.asarray(s, float)}).dropna().to_csv(
        os.path.join(OUT, f"series_{name}.csv"), index=False)


def load_series(df, name):
    s = pd.read_csv(os.path.join(OUT, f"series_{name}.csv")).set_index("code")["v"]
    return pd.Series(s.reindex(df.code).to_numpy(), index=df.index)
