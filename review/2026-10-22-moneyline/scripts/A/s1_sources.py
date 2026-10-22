"""Step 1: walk-forward margin sources over the whole history, and their TUNE scores.

Sources: the Kalman form rating re-run at several drifts / sigmas (with and without an online home edge),
the decayed ridge on margin (gamer + team strengths + home intercept), updated as results come in (live)
or refitted at each day's midnight (daily), and the rolling-form expected margins in features.csv.
Each source is scored on TUNE by its margin RMSE and by the log loss of a plain walk-forward link
logistic(a + b m) refitted daily on the last 120 days (training rows from 2026-01-15).
Writes out/A/sources.pkl (code + every source) and out/A/s1_tune.csv.
"""
import sys
import time

sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/A")
import common as C  # first: it pins the BLAS threads to 1 before numpy loads

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

H = C.H
TRAIN_FROM = "2026-01-15"


def rating_job(df, drift, sigma, home_rate):
    m, v1, v2 = C.kalman_rating(df, drift=drift, sigma=sigma, home_rate=home_rate)
    name = f"kal_d{drift:g}_s{sigma:g}" + ("_h" if home_rate else "")
    return {name: m, name + "_var": v1 + v2}


def ridge_job(df, hl, lg, lt, daily=False, stream_home=False, gamer_home=False):
    m = C.ridge_live(df, half_life=hl, lam_g=lg, lam_t=lt, daily=daily, stream_home=stream_home,
                     gamer_home=gamer_home)
    name = f"ridge_hl{hl:g}_g{lg:g}_t{lt:g}" + ("_daily" if daily else "") + ("_sh" if stream_home else "") \
        + ("_gh" if gamer_home else "")
    return {name: m}


def score(df, col):
    w = H.in_window(df, H.TUNE) & df.margin.notna() & df[col].notna()
    rmse = float(np.sqrt(np.mean((df.margin[w] - df[col][w]) ** 2)))
    p = C.link_walk_forward(df, [col], *H.TUNE, train_days=120, train_from=TRAIN_FROM)
    d = df.assign(_p=p)
    l, n = C.ll(d, "_p", H.TUNE)
    a, b = H.calib(d.y[H.in_window(d, H.TUNE) & d.y.notna() & d._p.notna()],
                   d._p[H.in_window(d, H.TUNE) & d.y.notna() & d._p.notna()])
    return dict(source=col, rmse=rmse, ll=l, n=n, slope=b)


def main():
    t0 = time.time()
    df = C.load(tables=False)
    slim = df[["tday", "p1", "p2", "t1", "t2", "s1", "s2", "stream", "day"]].copy()
    jobs = []
    for drift in (0.25, 0.5, 1, 2, 4, 8, 16):
        for sigma in (10, 12):
            for hr in (0.0, 0.002):
                jobs.append(delayed(rating_job)(slim, drift, sigma, hr))
    for hl in (7, 15, 30, 60, 120, 240):
        for lg in (2, 5, 15, 40):
            for lt in (20, 100):
                jobs.append(delayed(ridge_job)(slim, hl, lg, lt, daily=True))
    for hl in (7, 15, 30, 60, 120, 240):
        for lg in (5, 15):
            jobs.append(delayed(ridge_job)(slim, hl, lg, 50))
    res = Parallel(n_jobs=2, verbose=5)(jobs)
    src = pd.DataFrame({"code": df.code})
    for r in res:
        for k, v in r.items():
            src[k] = v
    for c in ("exp_margin_5", "exp_margin_15", "exp_margin_50", "exp_margin_all", "rating_margin"):
        src["fe_" + c] = df[c].to_numpy()
    src["fe_ewma_margin"] = ((df.p1_ewma_pf - df.p1_ewma_pa) - (df.p2_ewma_pf - df.p2_ewma_pa)).to_numpy() / 2
    src.to_pickle(f"{C.OUT}/sources.pkl")
    print(f"sources built in {time.time() - t0:.0f}s: {src.shape}", flush=True)
    d = df.merge(src.drop(columns=[c for c in src if c in df.columns and c != "code"]), on="code", how="left")
    cols = [c for c in src.columns if c != "code" and not c.endswith("_var")]
    base = d[["code", "start", "day", "y", "margin"]]
    rows = Parallel(n_jobs=2)(delayed(score)(base.assign(**{c: d[c]}), c) for c in cols)
    out = pd.DataFrame(rows).sort_values("ll")
    out.to_csv(f"{C.OUT}/s1_tune.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
