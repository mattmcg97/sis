import pandas as pd, numpy as np
d = pd.read_csv("eAMFCalibrator/out_bets_drives/bets_sim.csv", low_memory=False)
c="v21-drives"
s = d[(d[f"simulated_{c}"]==True)]
p = s[s.result=="push"]
print(p.groupby("market").size())
print(p[["market","bet_line","bet_line_prod_side","stream_line","final_p1","final_p2","result_by_score","CUSTOMER_WIN_LOSS",f"candidate_line_{c}",f"candidate_result_{c}","stake","revenue",f"candidate_revenue_{c}"]].head(20).to_string())
print("result_by_score for pushes:", p.groupby(["market","result_by_score"]).size())
print("cand revenue on push bets", p[f"candidate_revenue_{c}"].sum(), "prod", p.revenue.sum())
print("CUSTOMER_WIN_LOSS", p.CUSTOMER_WIN_LOSS.value_counts())
