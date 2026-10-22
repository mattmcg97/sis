"""Which bets were placed before both gamers' previous results were in (RESULT_IN = 36 min after that match's
kick-off, as prior_bets uses), and v21 / v21-drives' same-bets change on them against the rest."""
import pandas as pd, numpy as np, datetime as dt
h = pd.read_csv("eAMFCalibrator/out/match_history.csv")
h["t"] = pd.to_datetime(h.SCHEDULED_START_TIME_UTC)
rows = []
for side in ("PLAYER_1_HANDLE", "PLAYER_2_HANDLE"):
    rows.append(h[["MATCH_CODE", "t", side]].rename(columns={side: "g"}))
g = pd.concat(rows).dropna()
g["g"] = g.g.str.strip().str.upper()
g = g.sort_values(["g", "t"])
g["prev_t"] = g.groupby("g").t.shift(1)
prev = g.groupby("MATCH_CODE").prev_t.max()
gap = (g.t - g.prev_t)
print("gap between a gamer's consecutive kick-offs (minutes), quantiles:", (gap.dt.total_seconds()/60).quantile([.1,.25,.5,.75]).round(1).to_dict())
d = pd.read_csv("eAMFCalibrator/out_bets_drives/bets_sim.csv", low_memory=False)
d["bt"] = pd.to_datetime(d.bet_time, format="mixed")
d["prev_in"] = d.match_code.map(prev) + pd.Timedelta(minutes=36)
d["clean"] = d.prev_in.isna() | (d.bt >= d.prev_in)
d["start"] = d.match_code.map(h.set_index("MATCH_CODE").t)
print("matches whose previous result (kick-off + 36 min) is after their own kick-off:",
      (d.drop_duplicates("match_code").prev_in > d.drop_duplicates("match_code").start).mean().round(3))
for c in ("v21", "v21-drives"):
    s = d[d[f"simulated_{c}"] == True]
    for ip in (False, True):
        for cl in (True, False):
            x = s[(s.in_play == ip) & (s.clean == cl)]
            if not len(x): continue
            ch = 100 * (x[f"candidate_revenue_{c}"].sum() - x.revenue.sum()) / x.stake.sum()
            # bootstrap over matches
            gg = x.groupby("match_code").agg(st=("stake", "sum"), rv=("revenue", "sum"), cr=(f"candidate_revenue_{c}", "sum"))
            B = np.random.default_rng(0).multinomial(len(gg), [1 / len(gg)] * len(gg), size=1000)
            dd = 100 * (B @ gg.cr.values - B @ gg.rv.values) / (B @ gg.st.values)
            print(f"{c:11s} {'in play' if ip else 'pre-match':9s} {'clean' if cl else 'result not in':14s} bets {len(x):6d} stake {x.stake.sum():10,.0f} change {ch:+.2f} [{np.percentile(dd,2.5):+.2f}, {np.percentile(dd,97.5):+.2f}]")
