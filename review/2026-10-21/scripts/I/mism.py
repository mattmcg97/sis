import pandas as pd, numpy as np
for path, c in (("eAMFCalibrator/out_bets_drives/bets_sim.csv", "v21-drives"), ("eAMFCalibrator/out_bets_late/bets_sim.csv", "v21")):
    d = pd.read_csv(path, low_memory=False)
    s = d[d[f"simulated_{c}"]==True].copy()
    s["chg"] = s[f"candidate_revenue_{c}"]-s.revenue
    moved = s[s[f"same_line_{c}"]!=True]
    mis = moved[(moved.result.isin(["won","lost"])) & (moved.result_by_score.isin(["won","lost","push"])) & (moved.result != moved.result_by_score)]
    print(path, c, "moved-line bets", len(moved), "operator result != final-score result at prod's line:", len(mis), "stake", round(mis.stake.sum()), "change on them", round(mis.chg.sum()), "of total", round(s.chg.sum()))
    print(mis.groupby(["result","result_by_score", f"candidate_result_{c}"]).chg.agg(["count","sum"]).round(0))
    # matches involved
    print("  matches:", mis.match_code.nunique(), mis.groupby("match_code").chg.sum().round(0).sort_values().head(5).to_dict())
