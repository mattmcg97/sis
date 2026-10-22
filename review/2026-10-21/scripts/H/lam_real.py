"""pace_models' ridge walk on the real drive table at the production shrinkage and at pace_eff's effective one."""
import numpy as np, pandas as pd
from xgb import pace_models as pm
m = pm.load("eAMFCalibrator/out/match_history.csv", "xgb/data/drive_stats.csv")
res = {}
for name, lam in (("prod (90/360 eff.)", (45.0, 180.0)), ("pace_eff (45/180 eff.)", (22.5, 90.0))):
    pm.PACE_LAMBDA = {"gamer": lam[0], "team": lam[1], "fixed": 1e-4}
    f, _ = pm.walk(m, "2026-08-01", "2026-10-19", log=lambda s: None, models=("league", "ridge"))
    f = f.dropna(subset=["real"])
    p, y = f.ridge.values, f.real.values
    res[name] = (y - p) ** 2
    print(f"{name:24s} n {len(f)} RMSE {np.sqrt(((y-p)**2).mean()):.4f} r {np.corrcoef(p,y)[0,1]:.4f} slope {np.polyfit(p,y,1)[0]:.3f} sd {p.std():.3f}")
d = res["pace_eff (45/180 eff.)"] - res["prod (90/360 eff.)"]
print("sq err pace_eff - prod:", d.mean().round(4), "t", (d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))).round(2))
