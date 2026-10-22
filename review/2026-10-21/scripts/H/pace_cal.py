import numpy as np, pandas as pd
t = pd.read_csv("xgb/out/turnovers/turnovers_matches.csv", parse_dates=["start"])
t = t[t.exp_drives.notna()]
D = t.h_drives + t.a_drives
e, lg = t.exp_drives, t.league_drives
print(len(t), "matches", t.start.min(), t.start.max())
print("real mean", D.mean(), "sd", D.std(), "| exp mean", e.mean(), "sd", e.std())
for name, x in (("exp", e), ("exp/league", e / lg)):
    b = np.polyfit(x, D, 1)
    print(name, "r", np.corrcoef(x, D)[0, 1].round(3), "slope", b[0].round(3) if name == "exp" else (b[0] * lg.mean()).round(3))
q = pd.qcut(e / lg, 5, labels=False)
print(pd.DataFrame({"exp": e, "real": D, "q": q}).groupby("q").mean().round(2).T)
# what v21 does: goal = exp/league * sim level; the multiplier's spread
rel = e / lg
print("rel sd", rel.std().round(4), "-> drives sd at 9.87:", (rel.std() * 9.87).round(3))
b = np.polyfit(rel, D / D.mean(), 1)
print("slope of real/mean on rel:", b[0].round(3))
