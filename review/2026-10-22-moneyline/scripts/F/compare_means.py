"""Gather every expected total, build the walk-forward blends, score them (out/F/means_all.pkl, means_report.txt).

Blends: each day an OLS of the total on the members' walk-forward predictions (with an intercept) over the matches
settled by that day's midnight in the last W days (W chosen on TUNE), applied to the day's matches.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402

df = F.panel()
m = pd.read_pickle(os.path.join(F.OUT, "means_ridge.pkl"))
mx = pd.read_pickle(os.path.join(F.OUT, "mixed.pkl"))
m = m.merge(mx, on="code", how="left")
xb = pd.concat([pd.read_pickle(os.path.join(F.OUT, "xgb_base_early.pkl")).dropna().rename(columns={"xgb_base_early": "xgb_base"}),
                pd.read_pickle(os.path.join(F.OUT, "xgb_base.pkl")).dropna()])
xb = xb.groupby("code", as_index=False).first()
m = m.merge(xb[["code", "xgb_base"]], on="code", how="left")
if os.path.exists(os.path.join(F.OUT, "xgb_stack.pkl")):
    m = m.merge(pd.read_pickle(os.path.join(F.OUT, "xgb_stack.pkl")), on="code", how="left")
if os.path.exists(os.path.join(F.OUT, "means_kalman.pkl")):
    m = m.merge(pd.read_pickle(os.path.join(F.OUT, "means_kalman.pkl")), on="code", how="left")
if os.path.exists(os.path.join(F.OUT, "xgb_stack_k.pkl")):
    m = m.merge(pd.read_pickle(os.path.join(F.OUT, "xgb_stack_k.pkl")), on="code", how="left")
k = pd.read_pickle(os.path.join(F.ML, "out/C/kalman_feats.pkl"))[["code", "k_ptsall_pred_h", "k_ptsall_pred_a"]]
k["C_kal"] = k.k_ptsall_pred_h + k.k_ptsall_pred_a
m = m.merge(k[["code", "C_kal"]], on="code", how="left")
d = df[["code", "day", "start", "settled", "total", "D", "exp_total_50", "exp_total_15", "h2h_total",
        "prod_total_line"] + [c for c in df.columns if c.startswith("tbl_") and c.endswith("_total")]].merge(m, on="code")
d["naive50"] = d.exp_total_50     # features' simple expectation: each side's last-50 scoring and the other's conceding


def blend(d, members, W, start="2026-05-15", end="2026-10-19", ridge=1e-6):
    out = np.full(len(d), np.nan)
    X_all = d[members].to_numpy(float)
    ok_all = ~np.isnan(X_all).any(1)
    y = d.total.to_numpy(float)
    st, sett, dayv = d.start.values, d.settled.values, d.day.values
    for day in F.days_between(d, start, end):
        cut = np.datetime64(pd.Timestamp(day))
        tr = ok_all & ~np.isnan(y) & (sett <= cut) & (st >= cut - np.timedelta64(W * 86400, "s"))
        if tr.sum() < 300:
            continue
        Z = np.column_stack([np.ones(tr.sum()), X_all[tr]])
        A = Z.T @ Z + ridge * np.eye(Z.shape[1])
        b = np.linalg.solve(A, Z.T @ y[tr])
        rows = np.flatnonzero((dayv == day) & ok_all)
        out[rows] = np.column_stack([np.ones(len(rows)), X_all[rows]]) @ b
    return out


BLENDS = {"blend3": ["b_tot", "c_prod", "xgb_base"],
          "blend_rx": ["b_tot", "xgb_base"],
          "blend5": ["b_tot", "c_prod", "xgb_base", "b_mixed", "C_kal"]}
if "xgb_stack" in d:
    BLENDS["blend_st"] = ["b_tot", "c_prod", "xgb_stack"]
if "k_tot" in d:
    BLENDS["blend_k"] = ["k_tot", "k_c", "xgb_base"]
    BLENDS["blend_k2"] = ["k_tot", "xgb_base"]
    BLENDS["blend_k4"] = ["k_tot", "k_c", "b_tot", "xgb_base"]
if "xgb_stack_k" in d:
    BLENDS["blend_kst"] = ["k_tot", "k_c", "xgb_stack_k"]
tu = (d.day >= F.H.TUNE[0]) & (d.day < F.H.TUNE[1])
lines = []
choice = {}
for name, mem in BLENDS.items():
    best = None
    for W in (14, 28, 56, 112):
        p = blend(d, mem, W)
        ok = tu & ~np.isnan(p) & d.total.notna()
        r = float(np.sqrt(np.mean((d.total[ok] - p[ok]) ** 2)))
        lines.append(f"  {name} W={W}: TUNE RMSE {r:.4f} on {int(ok.sum())}")
        if best is None or r < best[0]:
            best = (r, W, p)
    choice[name] = best[1]
    d[name] = best[2]
d["avg3"] = d[["b_tot", "c_prod", "xgb_base"]].mean(axis=1, skipna=False)
lines.append("  chosen W: " + str(choice))
# blend weights at the end (for the notes)
for name, mem in BLENDS.items():
    W = choice[name]
    cut = pd.Timestamp("2026-10-19")
    tr = d[(d.settled <= cut) & (d.start >= cut - pd.Timedelta(days=W))].dropna(subset=mem + ["total"])
    Z = np.column_stack([np.ones(len(tr)), tr[mem].to_numpy(float)])
    b = np.linalg.lstsq(Z, tr.total.to_numpy(float), rcond=None)[0]
    lines.append(f"  {name} weights at 10-19 (W {W}): const {b[0]:+.2f}, " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(mem, b[1:])))
d.to_pickle(os.path.join(F.OUT, "means_all.pkl"))

base = ["b_tot", "c_prod", "b_mixed", "xgb_base"] + (["xgb_stack"] if "xgb_stack" in d else []) + \
       [c for c in ("k_tot", "k_pts", "k_c", "xgb_stack_k") if c in d] + \
       ["C_kal", "naive50", "avg3"] + list(BLENDS)
for wn in ("TUNE", "TEST", "SEP", "LATE"):
    lines.append("")
    lines.append(F.total_table(d, base, F.WINDOWS[wn], ref="b_tot"))
    pl = d[d.prod_total_line.notna()]
    lines.append(F.total_table(pl.assign(prod_line=pl.prod_total_line), ["prod_line", "b_tot", "xgb_base"] + list(BLENDS),
                               F.WINDOWS[wn], ref="prod_line"))
tb = [c for c in d.columns if c.startswith("tbl_") and c.endswith("_total")]
for wn in ("SEP", "LATE"):
    lines.append("")
    lines.append(F.total_table(d, tb + ["b_tot", "xgb_base"] + [c for c in ("k_tot", "k_c") if c in d] + list(BLENDS),
                               F.WINDOWS[wn], ref="tbl_glmer_both_total"))
for wn in ("TUNE", "TEST", "SEP"):
    lines.append(F.total_table(d, ["drives", "drives_pm", "drives_mixed"] + (["k_drv"] if "k_drv" in d else []),
                               F.WINDOWS[wn], ref="drives_pm", target="D"))
txt = "\n".join(lines)
print(txt)
open(os.path.join(F.OUT, "means_report.txt"), "w").write(txt + "\n")
