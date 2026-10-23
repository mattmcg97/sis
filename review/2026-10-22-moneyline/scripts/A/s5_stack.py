"""Step 5: the link on several long-history margins at once (a small stack), against the best single.

    python s5_stack.py <window spec> <best single> <stack1 a,b,c> <stack2 ...> ...

Each stack is logistic(a + sum b_k m_k), refitted daily like the single links; judged on TUNE, reported on
TEST. Writes out/A/s5_stack.csv and out/A/s5_preds.pkl.
"""
import sys

sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/A")
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

H = C.H
TRAIN_FROM = "2026-01-15"
SPAN = (H.TUNE[0], H.TEST[1])


def window_kw(spec):
    if spec.startswith("days:"):
        return dict(train_days=int(spec[5:]))
    if spec.startswith("hl:"):
        return dict(half_life=float(spec[3:]))
    return {}


def run(slim, cols, wkw):
    return C.link_walk_forward(slim, cols, *SPAN, train_from=TRAIN_FROM, keep_coefs=True, **wkw)


def main():
    wkw = window_kw(sys.argv[1])
    best = sys.argv[2]
    stacks = [s.split(",") for s in sys.argv[3:]]
    df = C.load(tables=False)
    src = pd.read_pickle(f"{C.OUT}/sources.pkl")
    allc = sorted({c for s in stacks for c in s} | {best})
    extra = [c for c in allc if c not in df.columns]
    df = df.merge(src[["code"] + extra], on="code", how="left")
    slim = df[["code", "start", "day", "y"] + allc].copy()
    specs = [[best]] + stacks
    res = Parallel(n_jobs=2)(delayed(run)(slim, s, wkw) for s in specs)
    names = ["+".join(s) for s in specs]
    d = slim[["code", "start", "day", "y"]].copy()
    for n, (p, co) in zip(names, res):
        d[n] = p
    rows = []
    for n, (p, co) in zip(names, res):
        r = dict(stack=n)
        for wn, win in (("tune", H.TUNE), ("test", H.TEST)):
            w = d[H.in_window(d, win) & d.y.notna() & d[n].notna() & d[names[0]].notna()]
            m = H.metrics(w.y, w[n])
            mu, lo, hi = H.boot_diff(w, n, names[0], n_boot=1000)
            r.update({f"{wn}_ll": m["logloss"], f"{wn}_slope": m["cal_slope"], f"{wn}_d": mu, f"{wn}_lo": lo,
                      f"{wn}_hi": hi})
        last = co[max(co)]
        r["coefs_last"] = " ".join(f"{x:+.4f}" for x in last)
        mid = co[sorted(co)[len(co) // 2]]
        r["coefs_mid"] = " ".join(f"{x:+.4f}" for x in mid)
        rows.append(r)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out.to_csv(f"{C.OUT}/s5_stack.csv", index=False)
    d.drop(columns=["start", "day", "y"]).to_pickle(f"{C.OUT}/s5_preds.pkl")


if __name__ == "__main__":
    main()
