import pandas as pd, numpy as np
for path, cands in (("eAMFCalibrator/out_bets_drives/bets_sim.csv", ("v21","v21-drives")), ("eAMFCalibrator/out_bets_late/bets_sim.csv", ("v21",))):
    d = pd.read_csv(path, low_memory=False)
    print("==", path, len(d))
    for c in cands:
        s = d[d[f"simulated_{c}"]==True]
        oc = s[f"candidate_odds_{c}"]; r = oc/s.odds
        tot = s[f"candidate_revenue_{c}"].sum() - s.revenue.sum()
        print(c, "bets", len(s), "cand odds < 1.0:", (oc<1.0).sum(), "< 1.01:", (oc<1.01).sum(), "real odds < 1.01:", (s.odds<1.01).sum(), "min cand odds", oc.min())
        lo = s[oc < 1.0]
        print("   revenue change from bets at cand odds < 1:", (lo[f"candidate_revenue_{c}"]-lo.revenue).sum(), "of total change", tot, "; won among them:", (lo[f"candidate_result_{c}"]=="won").sum())
        for k in (2, 3, 5):
            big = s[(r > k) | (r < 1/k)]
            print(f"   odds ratio beyond x{k}: {len(big)} bets, stake {big.stake.sum():,.0f}, change {(big[f'candidate_revenue_{c}']-big.revenue).sum():+,.0f}")
        print("   candidate prob min/max", s[f"candidate_prob_{c}"].min(), s[f"candidate_prob_{c}"].max(), "prod", s.stream_prob.min(), s.stream_prob.max())
        # top 10 individual contributions
        dd = (s[f"candidate_revenue_{c}"]-s.revenue)
        print("   top 20 bets' |change| share of total change:", dd.abs().nlargest(20).sum(), "sum signed top-20 by abs", dd.loc[dd.abs().nlargest(20).index].sum())
