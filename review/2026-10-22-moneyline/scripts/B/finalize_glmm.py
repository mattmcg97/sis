"""B_glmm: the daily-refit lme4 GLMM with the lowest TUNE log loss (full-window runs only)."""
import common as C
import glob, os, json
import pandas as pd
H = C.H
d = C.slim()
res = {}
for f in sorted(glob.glob(f"{C.OUT}/glmm/daily_*.csv")):
    if f.endswith("_theta.csv"):
        continue
    x = d[["code", "day", "y"]].merge(pd.read_csv(f), on="code")
    w = H.in_window(x, H.TUNE) & x.y.notna()
    res[os.path.basename(f)] = H.metrics(x.y[w], x.p[w])["logloss"]
for f in sorted(glob.glob(f"{C.OUT}/glmm/tuneonly_*.csv")):
    if f.endswith("_theta.csv"):
        continue
    x = d[["code", "day", "y"]].merge(pd.read_csv(f), on="code")
    w = H.in_window(x, H.TUNE) & x.y.notna()
    print("TUNE-only run", os.path.basename(f), round(H.metrics(x.y[w], x.p[w])["logloss"], 5))
best = min(res, key=res.get)
print({k: round(v, 5) for k, v in sorted(res.items(), key=lambda kv: kv[1])}, "->", best)
p = pd.read_csv(f"{C.OUT}/glmm/{best}").rename(columns={"p": "glmm"})
x = d[["code"]].merge(p, on="code")
H.write_preds(x, "glmm", "B_glmm")
json.dump(dict(selected=best, tune_ll=res), open(f"{C.OUT}/glmm_selected.json", "w"), indent=1)
