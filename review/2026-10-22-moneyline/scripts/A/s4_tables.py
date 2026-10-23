"""Step 4: the link on the walkthrough / form prior tables' margins.

    python s4_tables.py <ridge source> <slow kalman source> <extras, comma separated or ->

For every table: logistic(a + b m) on its expected margin, logistic(a + b logit(its own p)), and the probit
(normal with fitted sd); refitted every day on the table's earlier matches (the last 60 days, the TUNE choice; the tables are short).
Two training sets: 'sep' (only the repo's tables, 2026-09-01 on: the link burns in over its first days, no
price until 300 results are in) and 'ext' (agent A's own walkthrough run from 2026-07-25 in front, when it
exists). Then the extras chosen on TUNE for the long-history sources, and stacks of the table margin with the
ridge and the ratings. Reported on SEP, 09-22..10-18 (LATE) and, with 'ext', TEST.
Writes out/A/s4_tables.csv and out/A/s4_preds.pkl.
"""
import sys

sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/A")
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import extras as E  # noqa: E402
import tables as T  # noqa: E402

H = C.H
SPAN = ("2026-07-25", H.TEST[1])
TABS = ["avg", "avg_plain", "nb2", "nb2_infer", "nb2_both", "glmer", "glmer_both", "form"]


def run(slim, terms, train, probit=False, keep=False):
    first = "2026-09-01" if train == "sep" else "2026-07-25"
    return C.link_walk_forward(slim, terms, first, SPAN[1], train_from=first, min_train=300, probit=probit,
                               keep_coefs=keep, train_days=60)


def main():
    ridge, slow = sys.argv[1], sys.argv[2]
    extras = [] if len(sys.argv) < 4 or sys.argv[3] == "-" else sys.argv[3].split(",")
    df = C.load(tables=True)
    df = T.add_ext(df)
    ext = bool(df.tbl_ext.any())
    print(f"extended tables: {ext} ({int(df.tbl_ext.sum())} rows before Sep)")
    src = pd.read_pickle(f"{C.OUT}/sources.pkl")
    df = df.merge(src[["code", ridge, slow]], on="code", how="left")
    slim = df[["code", "start", "day", "y"]].copy()
    trains = ["sep"] + (["ext"] if ext else [])
    jobs, names = [], []
    for n in TABS:
        m = df[f"tbl_{n}_margin"].to_numpy(float)
        lp = H.logit(df[f"tbl_{n}_p"].to_numpy(float)) if n != "form" or True else None
        lp = np.where(df[f"tbl_{n}_p"].notna(), lp, np.nan)
        for tr in trains:
            jobs.append(delayed(run)(slim, [lambda d, a=m: a], tr))
            names.append(f"L_{n}_{tr}")
            jobs.append(delayed(run)(slim, [lambda d, a=lp: a], tr))
            names.append(f"Lp_{n}_{tr}")
            jobs.append(delayed(run)(slim, [lambda d, a=m: a], tr, True))
            names.append(f"Pr_{n}_{tr}")
    # extras and stacks on avg (and glmer_both, form)
    Xs = {}
    for n in ("avg", "glmer_both", "form"):
        df[f"_m_{n}"] = df[f"tbl_{n}_margin"]
        Xs[n] = E.build(df, f"_m_{n}", fast="rating_margin", total=f"tbl_{n}_total")
    singles = ["rematch", "rematch_full", "meeting", "sess_gap", "sess_win", "sess_margin", "stream", "stream_x",
               "tot", "tot_x", "fast_gap", "new_x", "h2h", "win3", "ewma_gap"]
    for tr in trains:
        for e in singles:
            tm = E.terms(Xs["avg"], (e,))
            jobs.append(delayed(run)(slim, [(lambda d, a=a: a) for a in tm], tr))
            names.append(f"X_avg_{e}_{tr}")
        if extras:
            for n in ("avg", "glmer_both", "form"):
                tm = E.terms(Xs[n], tuple(extras))
                jobs.append(delayed(run)(slim, [(lambda d, a=a: a) for a in tm], tr))
                names.append(f"XS_{n}_{tr}")
        stacks = {"avg+rating": ["tbl_avg_margin", "rating_margin"], "avg+ridge": ["tbl_avg_margin", ridge],
                  "avg+slow": ["tbl_avg_margin", slow], "avg+ridge+rating": ["tbl_avg_margin", ridge, "rating_margin"],
                  "avg+form": ["tbl_avg_margin", "tbl_form_margin"],
                  "nb2_infer+glmer_both": ["tbl_nb2_infer_margin", "tbl_glmer_both_margin"],
                  "avg+ridge+slow+rating": ["tbl_avg_margin", ridge, slow, "rating_margin"]}
        for sn, cols in stacks.items():
            arrs = [df[c].to_numpy(float) for c in cols]
            jobs.append(delayed(run)(slim, [(lambda d, a=a: a) for a in arrs], tr))
            names.append(f"S_{sn}_{tr}")
            if extras:
                tm = E.terms(Xs["avg"], tuple(extras))[1:]
                jobs.append(delayed(run)(slim, [(lambda d, a=a: a) for a in arrs + tm], tr))
                names.append(f"SX_{sn}_{tr}")
    print(f"{len(jobs)} links", flush=True)
    res = Parallel(n_jobs=2, verbose=2)(jobs)
    P = pd.DataFrame(dict(zip(names, res)), index=df.index)
    keep = ["code", "start", "day", "y", "prod"] + [f"tbl_{n}_p" for n in TABS] + ["tbl_ext"]
    d = pd.concat([df[keep], P], axis=1)
    d["rating"] = df["rating"]
    d.drop(columns=["start", "day", "y", "prod"]).to_pickle(f"{C.OUT}/s4_preds.pkl")
    rows = []
    windows = [("SEP", H.SEP), ("LATE", T.LATE), ("SEPTUNE", T.SEP_TUNE)] + ([("TEST", H.TEST)] if ext else [])
    for c in names:
        tr = c.rsplit("_", 1)[1]
        r = dict(model=c)
        for wn, win in windows:
            if wn == "TEST" and tr == "sep":
                continue
            # judged against tbl_avg_p and prod on the matches all three price
            w = d[H.in_window(d, win) & d.y.notna() & d[c].notna() & d.tbl_avg_p.notna() & d["prod"].notna()]
            if len(w) < 100:
                continue
            m = H.metrics(w.y, w[c])
            mu, lo, hi = H.boot_diff(w, c, "tbl_avg_p", n_boot=1000)
            mp, lp_, hp = H.boot_diff(w, c, "prod", n_boot=1000)
            r.update({f"{wn}_n": len(w), f"{wn}_ll": m["logloss"], f"{wn}_slope": m["cal_slope"],
                      f"{wn}_vs_avg": mu, f"{wn}_avg_lo": lo, f"{wn}_avg_hi": hi, f"{wn}_vs_prod": mp,
                      f"{wn}_prod_lo": lp_, f"{wn}_prod_hi": hp})
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(f"{C.OUT}/s4_tables.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_rows", 500)
    show = ["model"] + [c for c in out.columns if c.endswith(("_ll", "_slope", "_vs_avg", "_vs_prod"))]
    print(out[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
