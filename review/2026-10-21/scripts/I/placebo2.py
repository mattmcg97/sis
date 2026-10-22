import pandas as pd, numpy as np, sys
import sys; sys.path.insert(0, ".")
from eAMFCalibrator import bets
d = pd.read_csv("eAMFCalibrator/out_bets_drives/bets_sim.csv", low_memory=False)
c="v21-drives"
s = d[(d[f"simulated_{c}"]==True) & d.market.isin(["total"])].copy()
def run(shift):
    out = []
    for i, r in s.iterrows():
        line = r.stream_line + shift
        res_c = bets.settle(int(r.feed_market), line, (r.final_p1, r.final_p2))
        row = dict(stream_prob=r.stream_prob, candidate_prob=r.stream_prob, odds=r.odds, stake=r.stake,
                   result=r.result, candidate_result=res_c, on_prod_line=True)
        out.append(bets.reprice(row)["candidate_revenue"])
    s["plac"] = out
    tot = 100*(s.plac.sum() - s.revenue.sum())/s.stake.sum()
    by = {sel: round(100*(g.plac.sum()-g.revenue.sum())/g.stake.sum(),2) for sel, g in s.groupby("selection")}
    return tot, by
print("stake share by selection", (s.groupby("selection").stake.sum()/s.stake.sum()).round(3).to_dict())
for sh in (-2, -1, 1, 2):
    print("constant shift", sh, run(sh))
print("v21-drives", 100*(s[f"candidate_revenue_{c}"].sum()-s.revenue.sum())/s.stake.sum(),
      {sel: round(100*(g[f"candidate_revenue_{c}"].sum()-g.revenue.sum())/g.stake.sum(),2) for sel, g in s.groupby("selection")})
