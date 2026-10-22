"""Score B's preds: harness reports on TUNE/TEST/SEP/LATE (common matches, refs prod and B's best), log loss by
month and by gamer-experience bucket. Usage: python eval_B.py <best> [extra pickles...]"""
import common as C
import os, sys, glob
import numpy as np, pandas as pd

H = C.H
best = sys.argv[1]
d = C.slim()
d["mg"] = np.minimum(d.p1_games, d.p2_games)
d["exp"] = pd.cut(d.mg, [-1, 29, 200, 1e9], labels=["<30", "30-200", ">200"]).astype(str)
d["month"] = d.day.str[:7]
cols = []
for f in sorted(glob.glob(f"{C.ML}/preds/B_*.csv")):
    name = os.path.basename(f)[:-4]
    p = pd.read_csv(f).rename(columns={"p_home": name})
    d = d.merge(p, on="code", how="left")
    cols.append(name)
for pk in sys.argv[2:]:
    x = pd.read_pickle(pk)
    extra = [c for c in x.columns if c != "code" and c not in d]
    d = d.merge(x[["code"] + extra], on="code", how="left")
    cols += extra
order = [best] + [c for c in cols if c != best]
lines = []
for win, nm in [(H.TUNE, "TUNE (in-sample for the hyper-parameters)"), (H.TEST, "TEST"), (H.SEP, "SEP"),
                (C.LATE, "LATE 09-22..10-18")]:
    lines.append(f"### {nm}")
    lines.append(H.report(d, order, window=win, refs=("prod", best)))
    lines.append("")
# per-column (not common) coverage-insensitive table, TEST
lines.append("### TEST, each column on its own priced matches (not common)")
lines.append(H.report(d, order, window=H.TEST, refs=("prod",), common=False))
lines.append("")


def table(by, win):
    w = d[H.in_window(d, win) & d.y.notna()].dropna(subset=order + ["prod"])
    rows = {}
    for c in order + ["prod"]:
        rows[c] = w.groupby(by).apply(lambda g: H.logloss_each(g.y.to_numpy(), g[c].to_numpy()).mean())
    t = pd.DataFrame(rows).T
    t.loc["n"] = w.groupby(by).size()
    return t


pd.set_option("display.width", 250)
lines.append("### log loss by month, 2026-06-01..10-18 (common matches with prod)")
lines.append(table("month", C.FULLWIN).round(4).to_string())
lines.append("")
for win, nm in [(C.FULLWIN, "Jun-Oct"), (H.TUNE, "TUNE"), (H.TEST, "TEST")]:
    lines.append(f"### log loss by experience bucket (fewer games of the two gamers), {nm}")
    lines.append(table("exp", win).round(4).to_string())
    lines.append("")
txt = "\n".join(lines)
open(f"{C.OUT}/eval_B.txt", "w").write(txt)
print(txt)
