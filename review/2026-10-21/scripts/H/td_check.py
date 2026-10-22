import numpy as np, pandas as pd
from xgb.pace_eff import td_fg
t = pd.read_csv("xgb/out/turnovers/turnovers_matches.csv", parse_dates=["start"])
d = pd.read_csv("xgb/data/drive_stats.csv")
m = t.merge(d[["match_code","h_fg","a_fg","h_td","a_td","h_turnover","a_turnover","h_drives","a_drives"]], left_on="MATCH_CODE", right_on="match_code")
m = m[m.start >= "2026-09-01"]
print("matches", len(m), "suspect sum", t.suspect.sum())
for s, f in (("h","home_final"),("a","away_final")):
    tf = [td_fg(x, y) for x, y in zip(m[f].astype(float), m[f"{s}_fg"].astype(float))]
    m[f"{s}_td_impl"] = [a for a, _ in tf]
print("sim tds/match", (m.h_tds+m.a_tds).mean(), "implied tds/match", (m.h_td_impl+m.a_td_impl).mean(), "drive_stats td", (m.h_td+m.a_td).mean())
gap = (m.h_td_impl+m.a_td_impl) - (m.h_tds+m.a_tds)
print("gap distribution", gap.value_counts().sort_index().to_dict())
# per side: does a missing TD come with an extra turnover for that side?
for s in "ha":
    g = m[f"{s}_td_impl"] - m[f"{s}_tds"]
    print(s, "side: mean turnovers when gap==0:", m.loc[g==0, f"{s}_turnovers"].mean(), " gap>=1:", m.loc[g>=1, f"{s}_turnovers"].mean(), "n gap>=1", (g>=1).sum())
print("sim drives/match", (m.h_drives_x+m.a_drives_x).mean() if "h_drives_x" in m else None)
rows = []
for s, o in (("h","a"),("a","h")):
    rows.append(pd.DataFrame({"to": m[f"{s}_turnovers"], "gap": m[f"{s}_td_impl"] - m[f"{s}_tds"], "impl": m[f"{s}_td_impl"],
                              "drv": m[f"{s}_drives_x"] if f"{s}_drives_x" in m else m[f"{s}_drives"]}))
L = pd.concat(rows)
X = np.column_stack([np.ones(len(L)), L.gap, L.impl, L.drv])
b, *_ = np.linalg.lstsq(X, L.to.values, rcond=None)
res = L.to.values - X @ b
se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (res @ res) / (len(L) - 4)))
print("side turnovers ~ 1 + td gap + implied tds + drives:", np.round(b, 3), "t", np.round(b / se, 1))
print(m.columns.tolist()[:40])
