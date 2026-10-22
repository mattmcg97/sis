"""Placebo: a candidate with no information -- prod's own line moved +-k at random, prod's own probability --
re-priced on the real bets exactly as bets.reprice does (odds x prod / candidate, settled at the candidate's line)."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, ".")
from eAMFCalibrator import bets, markets
d = pd.read_csv("eAMFCalibrator/out_bets_drives/bets_sim.csv", low_memory=False)
c="v21-drives"
s = d[(d[f"simulated_{c}"]==True) & d.market.isin(["spread","total"])].copy()
rng = np.random.default_rng(0)
def run(k, seed):
    rng = np.random.default_rng(seed)
    shift = rng.choice([-k, k], size=len(s))
    out = []
    for (i, r), sh in zip(s.iterrows(), shift):
        line = r.stream_line + sh
        res_c = bets.settle(int(r.feed_market), line, (r.final_p1, r.final_p2))
        row = dict(stream_prob=r.stream_prob, candidate_prob=r.stream_prob, odds=r.odds, stake=r.stake,
                   result=r.result, candidate_result=res_c, on_prod_line=True)
        out.append(bets.reprice(row)["candidate_revenue"])
    s["plac"] = out
    ch = 100*(s.plac.sum() - s.revenue.sum())/s.stake.sum()
    # bootstrap over matches
    g = s.groupby("match_code").agg(st=("stake","sum"), rv=("revenue","sum"), pl=("plac","sum"))
    B = np.random.default_rng(1).multinomial(len(g), [1/len(g)]*len(g), size=1000)
    dd = 100*(B@g.pl.values - B@g.rv.values)/(B@g.st.values)
    per = {mk: 100*(s[s.market==mk].plac.sum()-s[s.market==mk].revenue.sum())/s[s.market==mk].stake.sum() for mk in ("spread","total")}
    return ch, np.percentile(dd,[2.5,97.5]), per
for k in (1.0, 2.0, 3.0):
    for seed in (0,1,2):
        ch, ci, per = run(k, seed)
        print(f"shift +-{k:g} seed {seed}: placebo change {ch:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]  spread {per['spread']:+.2f} total {per['total']:+.2f}")
print("v21-drives on the same bets:", 100*(s[f"candidate_revenue_{c}"].sum()-s.revenue.sum())/s.stake.sum())
