"""Step 7: the plain link's coefficients over time (is the shrinkage stable?) and the effective margin sd.

For each margin source: logistic(a + b m) and the probit Phi(a' + m / sd) refitted daily on the last 60 days
(the TUNE choice), every day 2026-06-01..10-18 (tables: from their first days). Prints a, b, the probit's sd
(the 'effective margin sd': the normal sd about the margin that the results say), and the 'shrink' factor
b_probit * 9.35 (1: the normal sd 9.35 is right; below 1: the margins are too spread out for it).
Writes out/A/s7_coefs.csv (every day) and prints a fortnightly summary.
"""
import sys

sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/A")
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import tables as T  # noqa: E402

H = C.H


def coefs(slim, col, start, train_from, probit):
    _, co = C.link_walk_forward(slim, [col], start, H.TEST[1], train_days=60, train_from=train_from,
                                probit=probit, keep_coefs=True, min_train=300)
    return pd.DataFrame([(d, b[0], b[1]) for d, b in sorted(co.items())], columns=["day", "a", "b"])


def main():
    cols = sys.argv[1].split(",")
    df = C.load(tables=True)
    df = T.add_ext(df)
    src = pd.read_pickle(f"{C.OUT}/sources.pkl")
    extra = [c for c in cols if c not in df.columns]
    df = df.merge(src[["code"] + extra], on="code", how="left")
    slim = df[["code", "start", "day", "y"] + cols].copy()
    jobs, keys = [], []
    for c in cols:
        first = df.loc[df[c].notna(), "day"].min()
        start = max(first, H.TUNE[0])
        tf = max(first, "2026-01-15")
        for pr in (False, True):
            jobs.append(delayed(coefs)(slim, c, start, tf, pr))
            keys.append((c, pr))
    res = Parallel(n_jobs=2)(jobs)
    out = []
    for (c, pr), r in zip(keys, res):
        r["source"] = c
        r["link"] = "probit" if pr else "logit"
        out.append(r)
    out = pd.concat(out)
    out.to_csv(f"{C.OUT}/s7_coefs.csv", index=False)
    pr = out[out.link == "probit"].copy()
    pr["sd_eff"] = 1.0 / pr.b
    pr["shrink"] = pr.b * 9.35
    lg = out[out.link == "logit"].copy()
    lg["fortnight"] = pd.to_datetime(lg.day).dt.to_period("2W").astype(str)
    pr["fortnight"] = pd.to_datetime(pr.day).dt.to_period("2W").astype(str)
    s = lg.groupby(["source", "fortnight"]).agg(a=("a", "mean"), b=("b", "mean"))
    s = s.join(pr.groupby(["source", "fortnight"]).agg(sd_eff=("sd_eff", "mean"), shrink=("shrink", "mean"),
                                                         a_probit=("a", "mean")))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print(s.round(4).to_string())
    # the margins' own spread, for reference
    for c in cols:
        w = H.in_window(df, H.TEST) & df[c].notna()
        print(f"{c}: sd of the margin on TEST {df.loc[w, c].std():.2f}")


if __name__ == "__main__":
    main()
