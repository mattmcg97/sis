"""Step 3b: a stricter choice of extras than step 3's forward selection (which took any TUNE gain over 0.0002
and did not hold up on TEST): only extras whose own TUNE gain has a day-resampled 95% interval below zero are
candidates, and one is added (greedily, best first) only if its gain over the model so far is again clear of
zero on TUNE. TEST is only reported.

    python s3b_select.py <source> <window> <fast> <total> <var>     (as s3_extras.py)

Writes out/A/s3b_<source>.txt and the chosen link's predictions to out/A/s3b_<source>_preds.pkl.
"""
import sys

sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/A")
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import extras as E  # noqa: E402
from s3_extras import run_link, window_kw  # noqa: E402

H = C.H


def main():
    src, wspec = sys.argv[1], sys.argv[2]
    fast, total, var = [None if a == "-" else a for a in sys.argv[3:6]]
    wkw = window_kw(wspec)
    tag = src + "_" + wspec.replace(":", "")
    singles = pd.read_csv(f"{C.OUT}/s3_{tag}.csv")
    singles = singles[~singles.extra.str.startswith("sel")]
    cand = singles[(singles.extra != "base") & (singles.tune_hi < 0)].sort_values("tune_d")
    df = C.load(tables=False)
    s = pd.read_pickle(f"{C.OUT}/sources.pkl")
    need = [c for c in {src, fast, total, var} if c and c not in df.columns]
    df = df.merge(s[["code"] + need], on="code", how="left")
    X = E.build(df, src, fast=fast, total=total, var=var)
    slim = df[["code", "start", "day", "y"]].copy()
    lines = [f"{src} ({wspec}): candidates clear of zero on TUNE: {list(cand.extra)}"]
    spec, c_unc = (), 0.0
    cur = run_link(slim, X, spec, wkw)
    names = list(cand.extra)
    groups = [{"rematch", "rematch_full", "rm_win"}, {"stream", "stream_x"}, {"tot", "tot_x"},
              {"sess_win", "sess_margin", "sess_gap"}]
    while names:
        trials = []
        for e in names:
            if e.startswith("unc"):
                kind, cv = e.rsplit("_", 1)
                trials.append((e, spec + (kind,), float(cv)))
            else:
                trials.append((e, spec + (e,), c_unc))
        preds = Parallel(n_jobs=2)(delayed(run_link)(slim, X, sp, wkw, cu) for _, sp, cu in trials)
        res = []
        for (e, sp, cu), p in zip(trials, preds):
            d = slim.assign(new=p, cur=cur)
            w = d[H.in_window(d, H.TUNE) & d.y.notna() & d.new.notna() & d.cur.notna()]
            mu, lo, hi = H.boot_diff(w, "new", "cur", n_boot=1000)
            res.append((mu, lo, hi, e, sp, cu, p))
        res.sort(key=lambda r: r[0])
        mu, lo, hi, e, sp, cu, p = res[0]
        lines.append(f"  + {e}: TUNE {mu:+.4f} ({lo:+.4f},{hi:+.4f})")
        if hi < 0:
            spec, c_unc, cur = sp, cu, p
            names = [n for n in names if n != e and not any(e in g and n in g for g in groups)
                     and not (e.startswith("unc") and n.startswith("unc"))]
        else:
            break
    lines.append(f"  chosen: {spec} c_unc {c_unc:g}")
    base = run_link(slim, X, (), wkw)
    d = slim.assign(sel=cur, base=base)
    for wn, win in (("TUNE", H.TUNE), ("TEST", H.TEST)):
        w = d[H.in_window(d, win) & d.y.notna() & d.sel.notna() & d.base.notna()]
        mu, lo, hi = H.boot_diff(w, "sel", "base", n_boot=2000)
        m = H.metrics(w.y, w.sel)
        lines.append(f"  {wn}: chosen link {m['logloss']:.4f} (slope {m['cal_slope']:.2f}), vs the plain link "
                     f"{mu:+.4f} ({lo:+.4f},{hi:+.4f})")
    print("\n".join(lines))
    with open(f"{C.OUT}/s3b_{tag}.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n" + repr(spec) + "\n" + repr(c_unc) + "\n")
    d[["code", "sel", "base"]].to_pickle(f"{C.OUT}/s3b_{tag}_preds.pkl")


if __name__ == "__main__":
    main()
