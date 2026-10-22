"""Item 3: does the total's spread depend on pace? On a dist_<tag> run (its walk-forward expected total and drives):
  - the fitted variance function's path (log V = a0 + a1 log(mu/35) + a2 log(muD/10.3)) and the sd it implies
  - each pace-aware form against its pace-free twin (log score, RPS, O/U at prod's line), day-resampled intervals
  - binned: the residual sd by fifth of expected drives within thirds of the expected total
  python pace_spread.py TAG MEANCOL
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402

tag, meancol = sys.argv[1], sys.argv[2]
df = F.panel()
r = pd.read_pickle(os.path.join(F.OUT, f"dist_{tag}.pkl"))
m = pd.read_pickle(os.path.join(F.OUT, "means_all.pkl"))[["code", meancol, "drives"]]
d = df[["code", "day", "total", "over", "D"]].merge(m, on="code").merge(r, on="code")
d = d.rename(columns={meancol: "mu"})
pr = pd.read_csv(os.path.join(F.OUT, f"dist_{tag}_params.csv"))
lines = [f"variance function path ({tag}): log V = a0 + a1 log(mu/35) + a2 log(muD/10.3); sd at (mu, muD)"]
for _, row in pr.iloc[::3].iterrows():
    b = json.loads(row.var_muD) if isinstance(row.var_muD, str) else row.var_muD
    bm = json.loads(row.var_mu) if isinstance(row.var_mu, str) else row.var_mu
    bc = json.loads(row.var_const) if isinstance(row.var_const, str) else row.var_const
    sd = lambda mu, D: np.sqrt(np.exp(b[0] + b[1] * np.log(mu / 35) + b[2] * np.log(D / 10.3)))
    lines.append(f"  {row.day}: const sd {np.sqrt(np.exp(bc[0])):.2f}; mu-only a1 {bm[1]:+.2f}; mu+D a1 {b[1]:+.2f} "
                 f"a2 {b[2]:+.2f}; sd(30,10.3) {sd(30, 10.3):.2f} sd(40,10.3) {sd(40, 10.3):.2f} "
                 f"sd(35,9.3) {sd(35, 9.3):.2f} sd(35,11.3) {sd(35, 11.3):.2f}")
pairs = [("normal_mu", "normal_const"), ("normal_D", "normal_const"), ("normal_muD", "normal_mu"),
         ("nb_mu", "nb_const"), ("nb_muD", "nb_mu"), ("lumpy_nb_mu", "lumpy_nb_const"), ("lumpy_nb_D", "lumpy_nb_const"),
         ("lumpy_nb_muD", "lumpy_nb_mu")]
for wn in ("TUNE", "TEST", "SEP", "LATE"):
    w = F.WINDOWS[wn]
    x = d[(d.day >= w[0]) & (d.day < w[1]) & d.total.notna()]
    lines.append(f"\n{wn}: each less its twin (95% days)        logS                       RPS                        "
                 f"O/U log loss (prod line)")
    for a_, b_ in pairs:
        if f"{a_}__logs" not in x or f"{b_}__logs" not in x:
            continue
        xx = x.dropna(subset=[f"{a_}__logs", f"{b_}__logs"])
        s = f"  {a_:14s} - {b_:14s}"
        for k in ("logs", "rps"):
            mu, lo, hi = F.boot_days(xx.day.values, (xx[f"{a_}__{k}"] - xx[f"{b_}__{k}"]).values)
            s += f"  {mu:+.4f} ({lo:+.4f},{hi:+.4f})"
        o = xx[xx.over.notna()]
        ll = lambda p: -(o.over * np.log(np.clip(p, 1e-6, 1 - 1e-6)) + (1 - o.over) * np.log(np.clip(1 - p, 1e-6, 1)))
        mu, lo, hi = F.boot_days(o.day.values, (ll(o[f"{a_}__p_over"]) - ll(o[f"{b_}__p_over"])).values)
        s += f"  {mu:+.4f} ({lo:+.4f},{hi:+.4f})"
        lines.append(s)
    xx = x.dropna(subset=["mu", "drives"])
    xx = xx.assign(e=xx.total - xx.mu, tq=pd.qcut(xx.mu, 3, labels=["low", "mid", "high"]))
    lines.append(f"  residual sd by fifth of expected drives (columns) within thirds of the expected total (rows):")
    for t_, g in xx.groupby("tq", observed=True):
        q = pd.qcut(g.drives, 5, labels=False)
        gg = g.groupby(q).agg(D=("drives", "mean"), sd=("e", "std"), n=("e", "size"))
        lines.append(f"    {t_:4s} (mu {g.mu.mean():.1f}): " + "  ".join(f"{r_.D:5.2f}: {r_.sd:5.2f}" for r_ in gg.itertuples()))
    q = pd.qcut(xx.drives, 5, labels=False)
    gg = xx.groupby(q).agg(D=("drives", "mean"), mu=("mu", "mean"), sd=("e", "std"))
    lines.append("    all (D: mu, sd): " + "  ".join(f"{r_.D:5.2f}: {r_.mu:4.1f}, {r_.sd:5.2f}" for r_ in gg.itertuples()))
    q = pd.qcut(xx.mu, 5, labels=False)
    gg = xx.groupby(q).agg(mu=("mu", "mean"), sd=("e", "std"))
    lines.append("    by fifth of mu (mu: sd): " + "  ".join(f"{r_.mu:4.1f}: {r_.sd:5.2f}" for r_ in gg.itertuples()))
txt = "\n".join(lines)
print(txt)
open(os.path.join(F.OUT, f"pace_spread_{tag}.txt"), "w").write(txt + "\n")
