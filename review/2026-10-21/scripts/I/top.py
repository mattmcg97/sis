import pandas as pd, numpy as np
d = pd.read_csv("eAMFCalibrator/out_bets_drives/bets_sim.csv", low_memory=False)
c = "v21-drives"
s = d[d[f"simulated_{c}"]==True].copy()
s["chg"] = s[f"candidate_revenue_{c}"]-s.revenue
t = s.loc[s.chg.abs().nlargest(20).index]
print(t[["match_code","in_play","market","selection","stake","odds","stream_prob",f"candidate_prob_{c}","stream_line",f"candidate_line_{c}","result",f"candidate_result_{c}","chg","CUSTOMER_TEMPERATURE","CUSTOMER_NAME_HASH"]].to_string())
print("total change", s.chg.sum())
# concentration by customer
cc = s.groupby("CUSTOMER_NAME_HASH").agg(stake=("stake","sum"), chg=("chg","sum"), n=("stake","size")).sort_values("stake", ascending=False)
print("customers", len(cc), "top 10 customers' stake share", cc.stake.head(10).sum()/cc.stake.sum(), "their change", cc.chg.head(10).sum())
print("by result flip:", s.groupby(["result", f"candidate_result_{c}"]).chg.agg(["sum","count"]))
