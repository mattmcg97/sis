"""Step 3: the link's extras, one at a time and together, for one long-history margin source.

    python s3_extras.py <source> <window spec: days:N | hl:N | exp> [fast source] [total source] [var source]

Every link is refitted daily (harness.walk_forward) over TUNE and TEST in one pass; the extras are judged on
TUNE (log loss less the base link's, day-resampled 95% interval) and only reported on TEST. A forward
selection on TUNE picks the combined set. Writes out/A/s3_<source>.csv and the predictions to
out/A/s3_<source>_preds.pkl.
"""
import sys

sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/A")
import common as C  # noqa: E402  (pins BLAS threads first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import extras as E  # noqa: E402

H = C.H
TRAIN_FROM = "2026-01-15"
SPAN = (H.TUNE[0], H.TEST[1])


def window_kw(spec):
    if spec.startswith("days:"):
        return dict(train_days=int(spec[5:]))
    if spec.startswith("hl:"):
        return dict(half_life=float(spec[3:]))
    return {}


def run_link(slim, X, spec, wkw, c_unc=0.0, l2=None):
    tm = E.terms(X, spec, c_unc=c_unc)
    p = C.link_walk_forward(slim, [(lambda d, a=a: a) for a in tm], *SPAN, train_from=TRAIN_FROM, l2=l2, **wkw)
    return p


def score_vs(d, col, base, window):
    w = d[H.in_window(d, window) & d.y.notna() & d[col].notna() & d[base].notna()]
    m = H.metrics(w.y, w[col])
    mu, lo, hi = H.boot_diff(w, col, base, n_boot=1000)
    return m["logloss"], m["cal_slope"], mu, lo, hi, len(w)


def main():
    src = sys.argv[1]
    wspec = sys.argv[2]
    fast = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    total = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    var = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    wkw = window_kw(wspec)
    df = C.load(tables=False)
    s = pd.read_pickle(f"{C.OUT}/sources.pkl")
    need = [c for c in {src, fast, total, var} if c]
    extra_cols = [c for c in need if c not in df.columns]
    df = df.merge(s[["code"] + extra_cols], on="code", how="left")
    X = E.build(df, src, fast=fast, total=total, var=var)
    slim = df[["code", "start", "day", "y"]].copy()
    singles = ["rematch", "rematch_full", "rm_win", "meeting", "sess_gap", "sess_win", "sess_margin", "win3",
               "ewma_gap", "stream", "stream_x", "new_x", "h2h"]
    if fast:
        singles.append("fast_gap")
    if total:
        singles += ["tot", "tot_x"]
    specs = {"base": ()}
    for e in singles:
        specs[e] = (e,)
    uncs = []
    if var:
        for c in (0.003, 0.01, 0.03, 0.1, 0.3):
            uncs.append(("unc", c))
    for c in (5.0, 20.0, 50.0, 100.0):
        uncs.append(("unc_g", c))
    jobs = [delayed(run_link)(slim, X, sp, wkw) for sp in specs.values()]
    jobs += [delayed(run_link)(slim, X, (u,), wkw, c) for u, c in uncs]
    preds = Parallel(n_jobs=2)(jobs)
    names = list(specs) + [f"{u}_{c:g}" for u, c in uncs]
    P = pd.DataFrame({n: p for n, p in zip(names, preds)}, index=df.index)
    d = pd.concat([slim, P], axis=1)
    rows = []
    for n in names:
        r = dict(extra=n)
        for wn, win in (("tune", H.TUNE), ("test", H.TEST)):
            l, sl, mu, lo, hi, k = score_vs(d, n, "base", win)
            r.update({f"{wn}_ll": l, f"{wn}_slope": sl, f"{wn}_d": mu, f"{wn}_lo": lo, f"{wn}_hi": hi, f"{wn}_n": k})
        rows.append(r)
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(f"source {src}, window {wspec}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # forward selection on TUNE
    unc_best = None
    u = out[out.extra.str.startswith("unc")]
    if len(u):
        b = u.sort_values("tune_d").iloc[0]
        if b.tune_d < -0.0002:
            unc_best = b.extra
    cand = [e for e in singles if out.set_index("extra").loc[e, "tune_d"] < -0.0002]
    chosen = []
    cur = out.set_index("extra").loc["base", "tune_ll"]
    c_unc = 0.0
    base_spec = ()
    if unc_best:
        kind, cval = unc_best.rsplit("_", 1)
        base_spec = (kind,)
        c_unc = float(cval)
        cur = out.set_index("extra").loc[unc_best, "tune_ll"]
        chosen.append(unc_best)
    spec_now = base_spec
    while cand:
        trial = Parallel(n_jobs=2)(delayed(run_link)(slim, X, spec_now + (e,), wkw, c_unc) for e in cand)
        scores = []
        for e, p in zip(cand, trial):
            dd = slim.assign(_p=p)
            scores.append((C.ll(dd, "_p", H.TUNE)[0], e, p))
        scores.sort(key=lambda x: x[0])
        best = scores[0]
        print(f"  forward: + {best[1]} -> TUNE {best[0]:.5f} (was {cur:.5f})", flush=True)
        if best[0] < cur - 0.0002:
            cur = best[0]
            spec_now = spec_now + (best[1],)
            chosen.append(best[1])
            cand = [c for c in cand if c != best[1]]
            # drop overlapping variants of the same idea
            groups = [{"rematch", "rematch_full", "rm_win"}, {"stream", "stream_x"}, {"tot", "tot_x"},
                      {"sess_win", "sess_margin", "sess_gap"}]
            for g in groups:
                if best[1] in g:
                    cand = [c for c in cand if c not in g or c == best[1]]
        else:
            break
    print(f"  chosen on TUNE: {chosen} (c_unc {c_unc:g})")
    final = "sel"
    d[final] = run_link(slim, X, spec_now, wkw, c_unc)
    r = dict(extra=final + ":" + "+".join(chosen))
    for wn, win in (("tune", H.TUNE), ("test", H.TEST)):
        l, sl, mu, lo, hi, k = score_vs(d, final, "base", win)
        r.update({f"{wn}_ll": l, f"{wn}_slope": sl, f"{wn}_d": mu, f"{wn}_lo": lo, f"{wn}_hi": hi, f"{wn}_n": k})
    out = pd.concat([out, pd.DataFrame([r])], ignore_index=True)
    print(pd.DataFrame([r]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    tag = src + "_" + wspec.replace(":", "")
    out.to_csv(f"{C.OUT}/s3_{tag}.csv", index=False)
    keep = ["code"] + [c for c in d.columns if c not in ("code", "start", "day", "y")]
    d[keep].to_pickle(f"{C.OUT}/s3_{tag}_preds.pkl")
    with open(f"{C.OUT}/s3_{tag}_chosen.txt", "w") as fh:
        fh.write(f"{chosen}\n{c_unc}\n")


if __name__ == "__main__":
    main()
