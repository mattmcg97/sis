"""Paired difference between two candidates on the same bets (common re-priced, as bets.compare / book_comparison):
its SE clustered by match (what the bootstrap does), by day, and two-way by gamer."""
import pandas as pd, numpy as np, sys
h = pd.read_csv("eAMFCalibrator/out/match_history.csv").set_index("MATCH_CODE")
S = sys.argv[1]
runs = [(f"{S}/zips/shape/bets_sim.csv", "v21", "v21-shape0.5"), (f"{S}/zips/pace/bets_sim.csv", "v21", "v21-pace"),
        (f"{S}/zips/gap/bets_sim.csv", "v21", "v21-gap0.6"), ("eAMFCalibrator/out_bets_late/bets_sim.csv", "v21", "v21-late"),
        ("eAMFCalibrator/out_bets_drives/bets_sim.csv", "v21", "v21-drives")]
for path, a, b in runs:
    d = pd.read_csv(path, low_memory=False, usecols=lambda col: col in ("match_code","stake","revenue","market") or col.startswith("candidate_revenue_") or col.startswith("simulated_"))
    s = d[(d[f"simulated_{a}"]==True) & (d[f"simulated_{b}"]==True)].copy()
    for label, sub in (("all", s), ("moneyline", s[s.market=="moneyline"])):
        u = (sub[f"candidate_revenue_{b}"] - sub[f"candidate_revenue_{a}"]) / sub.stake.sum()
        R = u.sum()
        sub = sub.assign(u=u, g1=sub.match_code.map(h.PLAYER_1_HANDLE).str.upper(), g2=sub.match_code.map(h.PLAYER_2_HANDLE).str.upper(), day=sub.match_code.str[-6:])
        v = lambda k: (sub.groupby(k).u.sum()**2).sum()
        vm, vd = v("match_code"), v("day")
        vcgm = v("g1") + v("g2") - vm
        print(f"{b:13s} - {a} {label:9s}: {100*R:+.2f} pts; se by match {100*np.sqrt(vm):.2f} -> 95% [{100*(R-1.96*np.sqrt(vm)):+.2f}, {100*(R+1.96*np.sqrt(vm)):+.2f}]; by day {100*np.sqrt(vd):.2f}; two-way gamer {100*np.sqrt(max(vcgm,0)):.2f} -> 95% [{100*(R-1.96*np.sqrt(max(vcgm,vm))):+.2f}, {100*(R+1.96*np.sqrt(max(vcgm,vm))):+.2f}]")
