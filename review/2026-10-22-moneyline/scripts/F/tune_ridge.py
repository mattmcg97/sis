"""Tune the time-decayed ridges on TUNE (06-01..07-31) RMSE of the total, coordinate search:
  b_tot   total = mu [+ stream] + gamer(p1) + gamer(p2) + team(t1) + team(t2)
  b_side  each side's points = mu + home [+ stream] + off(gamer) + def(opp gamer) + offT + defT; total = sum
  drives  D = mu [+ stream] + gamer pace (p1 + p2) + team pace (t1 + t2)        (RMSE of D)
  ppd     each side's points / its drives (drive-weighted) = side design;      (RMSE of the total D_hat/2 x sum)
Writes out/F/tune_<model>.csv and out/F/tuned.json.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402

df = F.panel()
ix = F.Index(df)
n = len(df)
TU = F.H.TUNE
tune_mask = ((df.day >= TU[0]) & (df.day < TU[1])).values


def rmse(pred, target):
    ok = tune_mask & ~np.isnan(pred) & ~np.isnan(target)
    return float(np.sqrt(np.mean((target[ok] - pred[ok]) ** 2))), int(ok.sum())


designs = {}


def get_design(kind, stream):
    key = (kind, stream)
    if key not in designs:
        designs[key] = F.match_design(df, ix, stream=stream) if kind == "match" else F.side_design(df, ix, stream=stream)
    return designs[key]


def run(model, s, drives_pred=None):
    if model in ("b_tot", "drives"):
        X, pk = get_design("match", s["stream"])
        y = df.total.values if model == "b_tot" else df.D.values
        p = F.walk_ridge(df, X, y, F.penalty(pk, s["lam_g"], s["lam_t"]), *TU, hl=s["hl"], train_days=s["td"],
                         level_hl=s["lvl"])
        return p, rmse(p, y.astype(float))
    X, pk = get_design("side", s["stream"])
    if model == "b_side":
        y = np.concatenate([df.s1.values, df.s2.values]).astype(float)
        p = F.walk_ridge(df, X, y, F.penalty(pk, s["lam_g"], s["lam_t"]), *TU, hl=s["hl"], train_days=s["td"],
                         level_hl=s["lvl"], nrep=2)
        tot = p[:n] + p[n:]
        return tot, rmse(tot, df.total.values.astype(float))
    # ppd: drive-weighted points a drive
    dr = np.concatenate([df.h_drives.values, df.a_drives.values]).astype(float)
    pts = np.concatenate([df.s1.values, df.s2.values]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(dr > 0, pts / dr, np.nan)
    ew = np.where(dr > 0, dr, np.nan)
    p = F.walk_ridge(df, X, y, F.penalty(pk, s["lam_g"], s["lam_t"]), *TU, hl=s["hl"], train_days=s["td"],
                     level_hl=s["lvl"], nrep=2, extra_w=ew)
    tot = drives_pred / 2.0 * (p[:n] + p[n:])
    return tot, rmse(tot, df.total.values.astype(float))


GRID = {
    "hl": [7, 15, 30, 45, 60, 90, 180, np.inf],
    "lam_g": [1, 2, 4, 8, 16, 32, 64, 128],
    "lam_t": [5, 20, 80, 320, 1e4],
    "td": [60, 120, 180, 365],
    "stream": [False, True],
    "lvl": [None, 3, 7, 14, 30],
}
START = {
    "b_tot": dict(hl=60, lam_g=16, lam_t=80, td=180, stream=False, lvl=None),
    "b_side": dict(hl=60, lam_g=16, lam_t=80, td=180, stream=False, lvl=None),
    "drives": dict(hl=60, lam_g=90, lam_t=360, td=180, stream=False, lvl=None),
    "ppd": dict(hl=60, lam_g=16, lam_t=80, td=180, stream=False, lvl=None),
}
PPD_SCALE = {"lam_g": [x for x in [4, 8, 16, 32, 64, 128, 256, 512]], "lam_t": [20, 80, 320, 1280, 1e4]}
DRV_SCALE = {"lam_g": [20, 45, 90, 180, 360], "lam_t": [90, 360, 1440, 1e4]}


def search(model, drives_pred=None, rounds=2):
    s = dict(START[model])
    grid = dict(GRID)
    if model == "ppd":
        grid.update(PPD_SCALE)
    if model == "drives":
        grid.update(DRV_SCALE)
    rows = []
    best = None
    seen = {}

    def ev(cand):
        key = json.dumps(cand, sort_keys=True, default=str)
        if key not in seen:
            _, (r, nn) = run(model, cand, drives_pred)
            seen[key] = r
            rows.append(dict(cand, rmse=r, n=nn))
            print(f"  {model} {cand} -> {r:.4f} ({nn})", flush=True)
        return seen[key]

    best = ev(s)
    for _ in range(rounds):
        changed = False
        for k in ["hl", "lam_g", "lam_t", "td", "stream", "lvl"]:
            for v in grid[k]:
                cand = dict(s)
                cand[k] = v
                r = ev(cand)
                if r < best - 1e-6:
                    best, s, changed = r, cand, True
        # joint hl x lam_g refinement
        for hl, lg in itertools.product(grid["hl"], grid["lam_g"]):
            cand = dict(s, hl=hl, lam_g=lg)
            r = ev(cand)
            if r < best - 1e-6:
                best, s, changed = r, cand, True
        if not changed:
            break
    pd.DataFrame(rows).to_csv(os.path.join(F.OUT, f"tune_{model}.csv"), index=False)
    return s, best


if __name__ == "__main__":
    which = sys.argv[1:] or ["b_tot", "b_side", "drives", "ppd"]
    path = os.path.join(F.OUT, "tuned.json")
    tuned = json.load(open(path)) if os.path.exists(path) else {}
    for model in which:
        dp = None
        if model == "ppd":
            sd = tuned["drives"]["setting"]
            X, pk = get_design("match", sd["stream"])
            dp = F.walk_ridge(df, X, df.D.values.astype(float), F.penalty(pk, sd["lam_g"], sd["lam_t"]), *TU,
                              hl=sd["hl"], train_days=sd["td"], level_hl=sd["lvl"])
        s, best = search(model, dp)
        tuned[model] = {"setting": s, "tune_rmse": best}
        print(model, "BEST", s, best, flush=True)
        json.dump(tuned, open(path, "w"), indent=1, default=float)
