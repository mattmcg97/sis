import pandas as pd, numpy as np
h = pd.read_csv("eAMFCalibrator/out/match_history.csv").set_index("MATCH_CODE")
for path, c in (("eAMFCalibrator/out_bets_drives/bets_sim.csv", "v21-drives"), ("eAMFCalibrator/out_bets_late/bets_sim.csv", "v21")):
    d = pd.read_csv(path, low_memory=False)
    s = d[d[f"simulated_{c}"]==True].copy()
    s["chg"] = s[f"candidate_revenue_{c}"]-s.revenue
    R = s.chg.sum()/s.stake.sum()
    s["u"] = (s.chg - 0*s.stake)/s.stake.sum()   # numerator only: stake is the same both ways
    s["g1"] = s.match_code.map(h.PLAYER_1_HANDLE).str.upper(); s["g2"] = s.match_code.map(h.PLAYER_2_HANDLE).str.upper()
    s["day"] = s.match_code.str[-6:]
    s["pair"] = [ "|".join(sorted([str(a), str(b)])) for a, b in zip(s.g1, s.g2)]
    def v(key): return (s.groupby(key).u.sum()**2).sum()
    vm = v("match_code"); vd = v("day"); vp = v("pair")
    # two-way-ish: gamer clusters with each bet counted under both gamers (CGM style for overlapping membership)
    gg = pd.concat([s[["g1","u"]].rename(columns={"g1":"g"}), s[["g2","u"]].rename(columns={"g2":"g"})])
    vg = (gg.groupby("g").u.sum()**2).sum()
    # CGM: V = V_g1 + V_g2 - V_match (match nested in each)
    v1 = (s.groupby("g1").u.sum()**2).sum(); v2 = (s.groupby("g2").u.sum()**2).sum()
    vcgm = v1 + v2 - vm
    print(f"{c} change {100*R:+.2f} pts; se by match {100*np.sqrt(vm):.2f}, by day {100*np.sqrt(vd):.2f} ({s.day.nunique()} days), by gamer pair {100*np.sqrt(vp):.2f}, two-way gamer (CGM) {100*np.sqrt(max(vcgm,0)):.2f}, bets {len(s)} matches {s.match_code.nunique()} gamers {pd.concat([s.g1,s.g2]).nunique()}")
