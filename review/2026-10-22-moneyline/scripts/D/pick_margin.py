"""Freeze the margin regressor's config from the stage-M TUNE results (best TUNE log loss after the link)."""
import json, pandas as pd
OUT = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/out/D"
r = pd.read_csv(f"{OUT}/tune_results.csv")
m = r[r.stage == "M"].sort_values("tune_ll")
print(m[["name", "tune_ll", "slope", "rounds_med"]].to_string(index=False))
best = json.loads(m.iloc[0].params)
obj = best.pop("objective")
cfg = json.load(open(f"{OUT}/configs_frozen.json"))
cfg["xgb_margin"]["objective"] = obj
cfg["xgb_margin"]["params"] = best
cfg["xgb_margin"]["picked_from"] = m.iloc[0]["name"]
json.dump(cfg, open(f"{OUT}/configs_frozen.json", "w"), indent=1)
print("xgb_margin ->", m.iloc[0]["name"], obj, best)
