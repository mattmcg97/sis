"""Step 6: the final reports (harness.report, common matches, day-resampled intervals), calibration by decile,
the home / stream tests, the rematch link's coefficients, and the predictions (preds/A_*.csv).

Frozen settings (all chosen on TUNE, or for the tables on 09-01..09-21 at most):
  ridge      ridge_hl30_g2_t50_sh: decayed ridge on margin, half-life 30 days, gamer penalty 2, team 50, a home
             edge per stream, updated as results come in (36 minutes after kick-off)
  kal        kal_d0.05_s7: the Kalman form rating re-run with drift 0.05 points^2/day, sigma 7
  rating     features.csv's rating_margin (drift 4, sigma 12)
  link       logistic(a + b m [+ extras]) refitted daily on the last 60 days of settled matches
  extras     strict forward selection on TUNE: ridge none; kal + stream; rating + (ridge - rating)
  tables     the link refitted daily on the table's own earlier matches (from 2026-07-25 with agent A's walkthrough)
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
RIDGE, KAL = "ridge_hl30_g2_t50_sh", "kal_d0.05_s7"
SPAN = (H.TUNE[0], H.TEST[1])
TF = "2026-01-15"


def deciles(d, col, ref, window):
    w = d[H.in_window(d, window) & d.y.notna() & d[col].notna() & d[ref].notna()]
    out = []
    for c in (col, ref):
        q = pd.qcut(w[c], 10, labels=False, duplicates="drop")
        g = w.groupby(q).agg(n=("y", "size"), mean_p=(c, "mean"), win=("y", "mean"))
        g["gap"] = g.win - g.mean_p
        g.insert(0, "model", c)
        out.append(g.reset_index(names="decile"))
    return pd.concat(out)


def link(slim, arrs, start=SPAN[0], train_from=TF, keep=False, min_train=200):
    return C.link_walk_forward(slim, [(lambda d, a=a: a) for a in arrs], start, SPAN[1], train_days=60,
                               train_from=train_from, keep_coefs=keep, min_train=min_train)


def main(table_best, table_best_desc):
    df = C.load(tables=True)
    df = T.add_ext(df)
    src = pd.read_pickle(f"{C.OUT}/sources.pkl")
    use = [RIDGE, KAL, "ridge_hl30_g2_t50", "ridge_hl30_g2_t50_gh", "rtot_hl30_g2_t50"]
    df = df.merge(src[["code"] + use], on="code", how="left")
    slim = df[["code", "start", "day", "y"]].copy()
    d = df[["code", "start", "day", "y", "prod", "rating", "tbl_ext", "stream", "meeting"]
           + [c for c in df.columns if c.startswith("tbl_") and c.endswith("_p")]].copy()
    for name, tag in (("ridge", f"{RIDGE}_days60"), ("kal", f"{KAL}_days60"), ("rating", "fe_rating_margin_days60")):
        p = pd.read_pickle(f"{C.OUT}/s3b_{tag}_preds.pkl")
        d[f"L_{name}"] = p["base"].to_numpy()
        d[f"L_{name}_x"] = p["sel"].to_numpy()
    d["raw_ridge"] = H._norm_p(df[RIDGE])
    d["raw_kal"] = H._norm_p(df[KAL])
    d["raw_rating"] = d["rating"]
    # home: no stream edge in the ridge, a home edge per gamer, and the link's intercept by stream
    jobs = [delayed(link)(slim, [df["ridge_hl30_g2_t50"].to_numpy(float)]),
            delayed(link)(slim, [df["ridge_hl30_g2_t50_gh"].to_numpy(float)])]
    X = E.build(df, RIDGE, fast="rating_margin", total="rtot_hl30_g2_t50")
    jobs.append(delayed(link)(slim, E.terms(X, ("rematch",)), keep=True))
    jobs.append(delayed(link)(slim, E.terms(X, ("stream",)), keep=True))
    jobs.append(delayed(link)(slim, E.terms(X, ()), keep=True))
    res = Parallel(n_jobs=2)(jobs)
    d["L_ridge_nosh"], d["L_ridge_gh"] = res[0], res[1]
    d["L_ridge_rematch"], co_rm = res[2]
    d["L_ridge_stream"], co_st = res[3]
    _, co_base = res[4]
    tp = pd.read_pickle(f"{C.OUT}/s4_preds.pkl")
    tp = tp.set_index("code").loc[d.code]
    for c in tp.columns:
        if c.startswith(("L_", "Lp_", "Pr_", "S_", "SX_", "X_", "XS_")):
            d["T" + c] = tp[c].to_numpy()
    d["A_best"] = np.where(d[f"T{table_best}"].notna(), d[f"T{table_best}"], d["L_ridge"])
    L = []
    long_cols = ["raw_rating", "L_rating", "L_rating_x", "raw_kal", "L_kal", "L_kal_x", "raw_ridge", "L_ridge"]
    for wn, win in (("TUNE", H.TUNE), ("TEST", H.TEST)):
        L.append(f"\n== {wn}: the long-history margins, raw (normal sd 9.35) and linked")
        L.append(H.report(d, long_cols, window=win, refs=("prod", "L_ridge")))
    tab_cols = ["tbl_avg_p", "tbl_glmer_both_p", "tbl_form_p", "TL_avg_ext", "TL_glmer_both_ext", "TL_form_ext",
                f"T{table_best}", "L_ridge", "L_kal", "L_rating", "raw_rating"]
    tab_cols = list(dict.fromkeys(c for c in tab_cols if c in d))
    for wn, win in (("SEP", H.SEP), ("09-22..10-18", T.LATE)):
        L.append(f"\n== {wn}: the table margins (their own p and linked) beside the long-history links")
        L.append(H.report(d, tab_cols, window=win, refs=("prod", "L_ridge", "tbl_avg_p")))
    if d.tbl_ext.any():
        L.append("\n== TEST from 2026-08-01 with agent A's own walkthrough tables (Aug) in front of the repo's (Sep-Oct)")
        tc = [c for c in tab_cols if c != "tbl_form_p"]
        L.append(H.report(d, tc, window=H.TEST, refs=("prod", "L_ridge", "tbl_avg_p")))
    L.append("\n== home: the ridge's home edge by stream (_sh, chosen) against one home edge (nosh) and a home edge per "
             "gamer (gh); the link's intercept by stream on top")
    for wn, win in (("TUNE", H.TUNE), ("TEST", H.TEST)):
        L.append(H.report(d, ["L_ridge", "L_ridge_nosh", "L_ridge_gh", "L_ridge_stream"], window=win,
                          refs=("L_ridge",)))
    L.append("\n== the rematch extra on the ridge link (residual of the pair's earlier meetings that session, settled "
             "by kick-off): its coefficient (logit per point of residual) by fortnight, with the margin's")
    cr = pd.DataFrame([(k, *v) for k, v in sorted(co_rm.items())], columns=["day", "a", "b_m", "b_rm"])
    cr["fortnight"] = pd.to_datetime(cr.day).dt.to_period("2W").astype(str)
    L.append(cr.groupby("fortnight")[["a", "b_m", "b_rm"]].mean().round(4).to_string())
    cs = pd.DataFrame([(k, *v) for k, v in sorted(co_st.items())], columns=["day", "a", "b_m", "s2", "s3"])
    L.append("\n  the link's stream intercepts on the ridge (stream 2, 3 against 1), mean over TEST days: "
             + cs[cs.day >= H.TEST[0]][["a", "s2", "s3"]].mean().round(4).to_string().replace("\n", "; "))
    cb = pd.DataFrame([(k, *v) for k, v in sorted(co_base.items())], columns=["day", "a", "b"])
    L.append("  the plain ridge link: b by month " + cb.assign(mo=cb.day.str[:7]).groupby("mo").b.mean().round(4)
             .to_string().replace("\n", "; "))
    # calibration by decile
    for wn, win, col in (("TEST", H.TEST, "L_ridge"), ("SEP", H.SEP, "A_best")):
        L.append(f"\n== calibration by decile of p, {wn}: {col} and prod (each on its own deciles, common matches)")
        dd = deciles(d, col, "prod", win)
        L.append(dd.round(4).to_string(index=False))
    txt = "\n".join(L)
    print(txt)
    with open(f"{C.OUT}/s6_final.txt", "w") as fh:
        fh.write(txt + "\n")
    # predictions: 2026-06-01 .. 10-18 wherever the model prices
    w = (d.day >= "2026-06-01") & (d.day < "2026-10-19")
    dd = d[w].copy()
    out = {"A_raw_rating": "raw_rating", "A_raw_kal": "raw_kal", "A_raw_ridge": "raw_ridge",
           "A_raw_tbl_avg": "tbl_avg_p", "A_raw_tbl_glmer_both": "tbl_glmer_both_p", "A_raw_tbl_form": "tbl_form_p",
           "A_link_rating": "L_rating", "A_link_rating_gap": "L_rating_x", "A_link_kal": "L_kal",
           "A_link_kal_stream": "L_kal_x", "A_link_ridge": "L_ridge", "A_link_tbl_avg": "TL_avg_ext",
           "A_link_tbl_glmer_both": "TL_glmer_both_ext", "A_link_tbl_form": "TL_form_ext", "A_best": "A_best"}
    for name, col in out.items():
        if col in dd:
            path = H.write_preds(dd, col, name)
            print(f"  {name}: {int(dd[col].notna().sum()):,} matches -> {path}")
    d.drop(columns=["start"]).to_pickle(f"{C.OUT}/s6_all.pkl")
    print(f"\nA_best = {table_best_desc}")


if __name__ == "__main__":
    main(sys.argv[1], " ".join(sys.argv[2:]))
