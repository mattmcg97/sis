"""Step 2: the link's training window, tuned on TUNE for the leading long-history margin sources.

logistic(a + b m) refitted every day on the matches settled by midnight: all of them since 2026-01-15
(expanding), the last N days, or all of them weighted by a half-life; plus the probit (normal with fitted
sd and intercept) and the fixed normal sd 9.35 for reference. Writes out/A/s2_window.csv.
"""
import sys

sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/A")
import common as C  # first: it pins the BLAS threads to 1 before numpy loads

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

H = C.H
TRAIN_FROM = "2026-01-15"


def run(d, col, kind, val):
    kw = dict(train_from=TRAIN_FROM)
    if kind == "days":
        kw["train_days"] = val
    elif kind == "hl":
        kw["half_life"] = val
    elif kind == "probit":
        kw["probit"] = True
        kw["train_days"] = val
    p = C.link_walk_forward(d, [col], *H.TUNE, **kw)
    d = d.assign(_p=p)
    l, n = C.ll(d, "_p", H.TUNE)
    w = H.in_window(d, H.TUNE) & d.y.notna() & d._p.notna()
    a, b = H.calib(d.y[w], d._p[w])
    return dict(source=col, kind=kind, val=val, ll=l, n=n, slope=b)


def main(cols):
    df = C.load(tables=False)
    src = pd.read_pickle(f"{C.OUT}/sources.pkl")
    df = df[["code", "start", "day", "y", "margin"]].merge(src[["code"] + cols], on="code", how="left")
    jobs = []
    for c in cols:
        jobs.append(delayed(run)(df, c, "expanding", 0))
        for v in (30, 60, 90, 120, 180):
            jobs.append(delayed(run)(df, c, "days", v))
        for v in (15, 30, 60, 120):
            jobs.append(delayed(run)(df, c, "hl", v))
        jobs.append(delayed(run)(df, c, "probit", 120))
    rows = Parallel(n_jobs=2)(jobs)
    out = pd.DataFrame(rows)
    # the fixed normal for reference
    for c in cols:
        d = df.assign(_p=H._norm_p(df[c]))
        l, n = C.ll(d, "_p", H.TUNE)
        out.loc[len(out)] = dict(source=c, kind="normal9.35", val=0, ll=l, n=n, slope=np.nan)
    out.to_csv(f"{C.OUT}/s2_window.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.sort_values(["source", "ll"]).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1].split(","))
