"""(d) XGBoost regressor on the total, walk-forward: refitted every REFIT days at a day's midnight on the matches
settled by then (TRAIN_DAYS window), trees by early stopping on the window's last 14 days then refitted whole
(xgb/backtest.py's recipe), nthread 2. Features: features.csv's as-of-kick-off features, symmetrised (each p1_/p2_
pair as sum and |difference|, the total being symmetric) + the match-level ones; prod never.
  python xgb_total.py NAME [--stack] [--depth 4] [--mcw 30] [--start 2026-06-01] [--end 2026-10-19]
--stack adds agent F's walk-forward ridge outputs (out/F/means_ridge.pkl) as features.
Writes out/F/xgb_<NAME>.pkl (code, xgb_<NAME>).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--stack", action="store_true")
ap.add_argument("--stackk", action="store_true", help="also the online Kalman outputs (means_kalman.pkl)")
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--mcw", type=float, default=30)
ap.add_argument("--eta", type=float, default=0.03)
ap.add_argument("--train-days", type=int, default=180)
ap.add_argument("--refit", type=int, default=7)
ap.add_argument("--start", default="2026-06-01")
ap.add_argument("--end", default="2026-10-19")
ap.add_argument("--target", default="total")
a = ap.parse_args()

import xgboost as xgb  # noqa: E402

df = F.panel()
fcols = pd.read_csv(os.path.join(F.ML, "features.csv"), nrows=0).columns.tolist()
ident = {"code", "start", "p1", "p2", "t1", "t2", "s1", "s2"}
feats = [c for c in fcols if c not in ident and not c.startswith("prod_") and "_dppd_" not in c]
pairs = sorted({c[3:] for c in feats if c.startswith("p1_") and ("p2_" + c[3:]) in feats})
match_cols = [c for c in feats if not c.startswith(("p1_", "p2_"))]
X = {}
for c in pairs:
    x1, x2 = df["p1_" + c].astype(float), df["p2_" + c].astype(float)
    X["sum_" + c] = x1 + x2
    X["adif_" + c] = (x1 - x2).abs()
for c in match_cols:
    X[c] = df[c].astype(float)
X["abs_rating_margin"] = df.rating_margin.abs()
if a.stack:
    st = pd.read_pickle(os.path.join(F.OUT, "means_ridge.pkl"))
    st = st.set_index("code").reindex(df.code)
    for c in ["b_tot", "b_side", "b_side_margin", "drives", "c_prod"]:
        X["st_" + c] = st[c].values
if a.stackk:
    kk = pd.read_pickle(os.path.join(F.OUT, "means_kalman.pkl")).set_index("code").reindex(df.code)
    for c in ["k_tot", "k_c", "k_drv", "k_pts_margin", "k_tot_pvar"]:
        X["st_" + c] = kk[c].values
X = pd.DataFrame(X, index=df.index)
y = df[a.target].astype(float).values
print(f"{X.shape[1]} features", flush=True)

PARAMS = dict(objective="reg:squarederror", eta=a.eta, max_depth=a.depth, min_child_weight=a.mcw, subsample=0.8,
              colsample_bytree=0.7, reg_lambda=5.0, nthread=2, verbosity=0, seed=0, tree_method="hist")
out = np.full(len(df), np.nan)
days = F.days_between(df, a.start, a.end)
model = None
settled = df.settled.values
starts = df.start.values
for k, d in enumerate(days):
    cut = np.datetime64(pd.Timestamp(d))
    if model is None or k % a.refit == 0:
        t0 = time.time()
        tr = (settled <= cut) & (starts >= cut - np.timedelta64(a.train_days * 86400, "s")) & ~np.isnan(y)
        va = tr & (starts >= cut - np.timedelta64(14 * 86400, "s"))
        fit_rows = tr & ~va
        dtr = xgb.DMatrix(X.values[fit_rows], label=y[fit_rows])
        dva = xgb.DMatrix(X.values[va], label=y[va])
        probe = xgb.train(PARAMS, dtr, 3000, evals=[(dva, "v")], early_stopping_rounds=150, verbose_eval=False)
        rounds = max(50, probe.best_iteration + 1)
        model = xgb.train(PARAMS, xgb.DMatrix(X.values[tr], label=y[tr]), rounds)
        print(f"  {d}: trained on {int(tr.sum()):,}, {rounds} trees, {time.time() - t0:.0f}s", flush=True)
    rows = np.flatnonzero(df.day.values == d)
    out[rows] = model.predict(xgb.DMatrix(X.values[rows]))
col = f"xgb_{a.name}"
res = pd.DataFrame({"code": df.code, col: out})
res.to_pickle(os.path.join(F.OUT, f"{col}.pkl"))
df[col] = out
for w in ("TUNE", "TEST", "SEP"):
    print(F.total_table(df, [col], F.WINDOWS[w], target=a.target))
if model is not None:
    g = model.get_score(importance_type="total_gain")
    tot = sum(g.values())
    names = X.columns.tolist()
    top = sorted(g.items(), key=lambda kv: -kv[1])[:20]
    print("top gain:", ", ".join(f"{names[int(k[1:])] if k.startswith('f') and k[1:].isdigit() else k} "
                                 f"{100 * v / tot:.1f}%" for k, v in top))
