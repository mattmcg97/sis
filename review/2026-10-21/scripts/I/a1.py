import pandas as pd, numpy as np
d = pd.read_csv("eAMFCalibrator/out_bets_drives/bets_sim.csv", low_memory=False)
print(len(d), d.bet_time.min(), d.bet_time.max(), d.match_code.nunique())
for c in ["v21","v21-drives"]:
    s = d[d[f"simulated_{c}"]==True]
    print(c, "simulated", len(s), "stake", s.stake.sum(), "prod margin", 100*s.revenue.sum()/s.stake.sum(), "cand", 100*s[f"candidate_revenue_{c}"].sum()/s.stake.sum())
    for mk in ["moneyline","spread","total"]:
        m = s[s.market==mk]
        same = m[m[f"same_line_{c}"]==True]; moved = m[m[f"same_line_{c}"]!=True]
        print(f"  {mk}: n={len(m)} same_line n={len(same)} moved n={len(moved)}; change same {100*(same[f'candidate_revenue_{c}'].sum()-same.revenue.sum())/m.stake.sum():+.2f} moved {100*(moved[f'candidate_revenue_{c}'].sum()-moved.revenue.sum())/m.stake.sum():+.2f}")
        if mk!="moneyline":
            dl = (moved[f"candidate_line_{c}"] - moved["stream_line"])
            print("    line diff cand-prod (moved) mean", dl.mean(), "by selection:", moved.assign(dl=dl).groupby("selection").dl.agg(["mean","count"]).to_dict())
            for sel, g in m.groupby("selection"):
                print(f"    {sel}: stake {g.stake.sum():.0f} change {100*(g[f'candidate_revenue_{c}'].sum()-g.revenue.sum())/g.stake.sum():+.2f}")
