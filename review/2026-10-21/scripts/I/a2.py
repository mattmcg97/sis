import pandas as pd, numpy as np
d = pd.read_csv("eAMFCalibrator/out_bets_drives/bets_sim.csv", low_memory=False)
c="v21-drives"
s = d[(d[f"simulated_{c}"]==True)]
for mk in ["spread","total"]:
    m = s[s.market==mk]
    print(mk, "cand line frac parts", (m[f"candidate_line_{c}"]%1).round(2).value_counts().head(5).to_dict(), "prod", (m.stream_line%1).round(2).value_counts().head(5).to_dict())
    print("  cand prob quantiles", m[f"candidate_prob_{c}"].quantile([.05,.25,.5,.75,.95]).round(3).to_dict())
    print("  prod prob quantiles", m["stream_prob"].quantile([.05,.25,.5,.75,.95]).round(3).to_dict())
    moved = m[m[f"same_line_{c}"]!=True]
    print("  moved: cand prob mean", moved[f"candidate_prob_{c}"].mean(), "prod prob mean", moved.stream_prob.mean(), "odds ratio mean", (moved[f"candidate_odds_{c}"]/moved.odds).mean())
    # results
    print("  results prod:", moved.result.value_counts().to_dict(), " cand:", moved[f"candidate_result_{c}"].value_counts().to_dict())
