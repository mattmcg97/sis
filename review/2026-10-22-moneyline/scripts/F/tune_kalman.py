"""Tune the online Kalman filters on TUNE (06-01..07-31) RMSE (Nelder-Mead on log variance ratios), then run them
over everything with the frozen settings: out/F/kalman_tuned.json, out/F/means_kalman.pkl.
  k_tot   total, match-level
  k_pts   each side's points, side-level (off/def): total = sum, margin = difference
  k_drv   drives, match-level
  k_ppd   points a drive, side-level, R / drives: total = k_drv / 2 x (ppd_h + ppd_a)   (tuned on that total's RMSE)
  k_td, k_fg  touchdowns / field goals a drive with k_ppd's settings
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402
import kalman as K  # noqa: E402

df = F.panel()
n = len(df)
TU = ((df.day >= F.H.TUNE[0]) & (df.day < F.H.TUNE[1])).values
KEYS = ["V_g", "q_g", "V_t", "q_t", "q_mu", "V_s"]
START = dict(V_g=0.05, q_g=0.0005, V_t=0.01, q_t=0.00005, q_mu=0.001, V_s=0.01)
dr = np.column_stack([df.h_drives.values, df.a_drives.values]).astype(float)
pts = np.column_stack([df.s1.values, df.s2.values]).astype(float)
with np.errstate(invalid="ignore", divide="ignore"):
    ppd = np.where(dr > 0, pts / dr, np.nan)
    tdr = np.where(dr > 0, np.column_stack([df.h_tdx.values, df.a_tdx.values]) / dr, np.nan)
    fgr = np.where(dr > 0, np.column_stack([df.h_fgx.values, df.a_fgx.values]) / dr, np.nan)
tot = df.total.values.astype(float)
Dv = df.D.values.astype(float)


def rmse(p, y):
    ok = TU & ~np.isnan(p) & ~np.isnan(y)
    return float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2)))


def model_run(name, p, drv=None):
    if name == "k_tot":
        pr, pv = K.run(df, "match", tot, p)
        return pr, pv, rmse(pr, tot)
    if name == "k_drv":
        pr, pv = K.run(df, "match", Dv, p)
        return pr, pv, rmse(pr, Dv)
    if name == "k_pts":
        pr, pv = K.run(df, "side", pts, p)
        return pr, pv, rmse(pr.sum(1), tot)
    if name == "k_ppd":
        pr, pv = K.run(df, "side", ppd, p, w=dr)
        return pr, pv, rmse(drv / 2 * pr.sum(1), tot)


def tune(name, drv=None, iters=160):
    log = []

    def f(v):
        p = dict(zip(KEYS, np.exp(v)))
        r = model_run(name, p, drv)[2]
        log.append(dict(p, rmse=r))
        return r

    v0 = np.log([START[k] for k in KEYS])
    res = optimize.minimize(f, v0, method="Nelder-Mead", options=dict(maxfev=iters, xatol=0.05, fatol=1e-4,
                                                                       initial_simplex=None))
    best = dict(zip(KEYS, np.exp(res.x)))
    pd.DataFrame(log).to_csv(os.path.join(F.OUT, f"tune_{name}.csv"), index=False)
    print(name, "best", {k: round(v, 6) for k, v in best.items()}, "TUNE RMSE", round(res.fun, 4), flush=True)
    return best, float(res.fun)


if __name__ == "__main__":
    out = {}
    path = os.path.join(F.OUT, "kalman_tuned.json")
    res = pd.DataFrame({"code": df.code})
    for name in ("k_tot", "k_pts", "k_drv"):
        best, r = tune(name)
        out[name] = dict(setting=best, tune_rmse=r)
        json.dump(out, open(path, "w"), indent=1)
        pr, pv, _ = model_run(name, best)
        if pr.ndim == 2:
            res[f"{name}_h"], res[f"{name}_a"] = pr[:, 0], pr[:, 1]
            res[name] = pr.sum(1)
            res[f"{name}_margin"] = pr[:, 0] - pr[:, 1]
        else:
            res[name] = pr
            res[f"{name}_pvar"] = pv
    best, r = tune("k_ppd", drv=res.k_drv.values)
    out["k_ppd"] = dict(setting=best, tune_rmse=r)
    json.dump(out, open(path, "w"), indent=1)
    pr, pv, _ = model_run("k_ppd", best, res.k_drv.values)
    res["k_ppd_h"], res["k_ppd_a"] = pr[:, 0], pr[:, 1]
    res["k_c"] = res.k_drv / 2 * (res.k_ppd_h + res.k_ppd_a)
    for nm, yy in (("k_td", tdr), ("k_fg", fgr)):
        pr, _ = K.run(df, "side", yy, best, w=dr)
        res[f"{nm}_h"], res[f"{nm}_a"] = pr[:, 0], pr[:, 1]
    res.to_pickle(os.path.join(F.OUT, "means_kalman.pkl"))
    d = df[["code", "day", "total", "D", "margin"]].merge(res, on="code")
    for w in ("TUNE", "TEST", "SEP"):
        print(F.total_table(d, ["k_tot", "k_pts", "k_c"], F.WINDOWS[w], ref="k_tot"))
        print(F.total_table(d, ["k_drv"], F.WINDOWS[w], target="D"))
        print(F.total_table(d, ["k_pts_margin"], F.WINDOWS[w], target="margin"))
