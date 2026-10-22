"""Every B variant (not just the family picks) on TUNE/TEST/SEP/LATE vs prod and B_glicko (kal3)."""
import common as C
import glob
import pandas as pd
H = C.H
d = C.slim()
bt = pd.read_pickle(f"{C.OUT}/bt_preds.pkl")
dy = pd.concat([pd.read_pickle(f).set_index("code") for f in glob.glob(f"{C.OUT}/dyn_preds_*.pkl")], axis=1)
dy = dy.loc[:, ~dy.columns.duplicated()].reset_index()
d = d.merge(bt, on="code", how="left").merge(dy, on="code", how="left")
for f in glob.glob(f"{C.OUT}/glmm/daily_*.csv"):
    if f.endswith("_theta.csv"):
        continue
    name = "glmm_" + f.split("daily_")[1][:-4]
    d = d.merge(pd.read_csv(f).rename(columns={"p": name}), on="code", how="left")
cols = [c for c in d.columns if c in bt.columns or c in dy.columns or c.startswith("glmm_")]
cols = [c for c in cols if c != "code"]
order = ["kal3"] + [c for c in cols if c != "kal3"]
out = []
for win, nm in [(H.TUNE, "TUNE"), (H.TEST, "TEST"), (H.SEP, "SEP"), (C.LATE, "LATE")]:
    out.append(f"### {nm}\n" + H.report(d, order, window=win, refs=("prod", "kal3"), n_boot=1000))
txt = "\n\n".join(out)
open(f"{C.OUT}/variants.txt", "w").write(txt)
print(txt)
