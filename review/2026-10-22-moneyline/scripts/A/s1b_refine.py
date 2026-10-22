"""Step 1b: a finer grid about step 1's best sources (scored on TUNE as step 1), added to out/A/sources.pkl.
Also the ridge with a home edge by stream (_sh) and by gamer (_gh), and the ridge's expected total."""
import sys
import time

sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/A")
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from s1_sources import rating_job, ridge_job, score  # noqa: E402

H = C.H


def total_job(df, hl, lg, lt):
    m = C.ridge_live(df, half_life=hl, lam_g=lg, lam_t=lt, target="total")
    return {f"rtot_hl{hl:g}_g{lg:g}_t{lt:g}": m}


def var_job(df, hl, lg, lt):
    m, v = C.ridge_live(df, half_life=hl, lam_g=lg, lam_t=lt, return_var=True)
    return {f"ridge_hl{hl:g}_g{lg:g}_t{lt:g}_pvar": v}


def main():
    t0 = time.time()
    df = C.load(tables=False)
    slim = df[["tday", "p1", "p2", "t1", "t2", "s1", "s2", "stream", "day"]].copy()
    src = pd.read_pickle(f"{C.OUT}/sources.pkl")
    jobs = []
    for drift in (0.05, 0.1, 0.15, 0.25, 0.35):
        for sigma in (8, 9, 10):
            jobs.append(delayed(rating_job)(slim, drift, sigma, 0.0))
    for hl in (20, 30, 45):
        for lg in (1, 2, 3, 5):
            jobs.append(delayed(ridge_job)(slim, hl, lg, 50))
    for lt in (10, 200):
        jobs.append(delayed(ridge_job)(slim, 30, 2, lt))
    jobs.append(delayed(ridge_job)(slim, 30, 2, 50, stream_home=True))
    jobs.append(delayed(ridge_job)(slim, 30, 2, 50, gamer_home=True))
    jobs.append(delayed(total_job)(slim, 30, 2, 50))
    jobs.append(delayed(total_job)(slim, 60, 5, 50))
    jobs.append(delayed(var_job)(slim, 30, 2, 50))
    res = Parallel(n_jobs=2, verbose=5)(jobs)
    new = []
    for r in res:
        for k, v in r.items():
            if k not in src.columns:
                src[k] = v
                new.append(k)
    src.to_pickle(f"{C.OUT}/sources.pkl")
    print(f"built in {time.time() - t0:.0f}s: {new}", flush=True)
    d = df.merge(src[["code"] + [c for c in new]], on="code", how="left")
    base = d[["code", "start", "day", "y", "margin"]]
    cols = [c for c in new if not c.endswith("_var") and not c.endswith("_pvar") and not c.startswith("rtot")]
    rows = Parallel(n_jobs=2)(delayed(score)(base.assign(**{c: d[c]}), c) for c in cols)
    out = pd.DataFrame(rows).sort_values("ll")
    old = pd.read_csv(f"{C.OUT}/s1_tune.csv")
    allr = pd.concat([old, out]).drop_duplicates("source").sort_values("ll")
    allr.to_csv(f"{C.OUT}/s1_tune.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    # the totals: TUNE rmse against the actual total, against features' exp_total_50
    for c in [k for k in new if k.startswith("rtot")]:
        w = H.in_window(d, H.TUNE) & d.total.notna()
        print(c, "TUNE total rmse", float(np.sqrt(np.mean((d.total[w] - d[c][w]) ** 2))),
              "exp_total_50", float(np.sqrt(np.nanmean((d.total[w] - d.exp_total_50[w]) ** 2))))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
