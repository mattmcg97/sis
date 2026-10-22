import pandas as pd, numpy as np
d = pd.read_csv("eAMFCalibrator/out_bets_drives/bets_sim.csv", low_memory=False)
c="v21-drives"
s = d[(d[f"simulated_{c}"]==True) & d.market.isin(["total"]) & d.result.isin(["won","lost"])].copy()
s["won"] = (s.result=="won").astype(float)
for sel, g in s.groupby("selection"):
    print(sel, "n", len(g), "win rate", g.won.mean().round(4), "prod prob", g.stream_prob.mean().round(4), "stake-wt win", (g.won*g.stake).sum()/g.stake.sum(), "stake-wt prob", (g.stream_prob*g.stake).sum()/g.stake.sum())
# under rate at prod's line per match at the last pre-match total quote: use the bets: one per match (first bet's prod line pre-match)
pm = d[(d.market=="total") & (d.in_play==False) & d.stream_line.notna()].sort_values("bet_time").groupby("match_code").last()
tot = pm.final_p1+pm.final_p2
print("pre-match: matches", len(pm), "share under prod line", (tot < pm.stream_line).mean(), "mean total - line", (tot-pm.stream_line).mean())
