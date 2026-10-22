import common as C
import glob, os, sys
import pandas as pd
H = C.H
d = C.slim()
rows = []
for f in sorted(glob.glob(f"{C.OUT}/glmm/" + (sys.argv[1] if len(sys.argv) > 1 else "tune_*.csv"))):
    if f.endswith("_theta.csv"):
        continue
    p = pd.read_csv(f).rename(columns={"p": "g"})
    x = d.merge(p, on="code")
    w = H.in_window(x, H.TUNE) & x.y.notna()
    m = H.metrics(x.y[w], x.g[w]); m["cfg"] = os.path.basename(f)
    th = pd.read_csv(f.replace(".csv", "_theta.csv"))
    m["theta_mean"] = th.filter(like="theta").mean().round(3).to_dict()
    rows.append(m)
print(pd.DataFrame(rows).sort_values("logloss")[["cfg", "n", "logloss", "brier", "auc", "cal_slope", "theta_mean"]].to_string())
