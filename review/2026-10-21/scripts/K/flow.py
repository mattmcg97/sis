import math, sys, random
from collections import defaultdict
from eAMFCalibrator import money_layer as ml

paths = ["eAMFCalibrator/out_bets_v21_4k/bets_sim.csv"]
rows = ml.read_sim_bets(paths)
cols = ml.sim_columns(paths)
print("rows", len(rows), "price cols", ml.price_columns(cols))
days = sorted({str(r.get("bet_time"))[:10] for r in rows})
print("days", days[0], days[-1], len(days))
for label, col in ml.price_columns(cols):
    bets, path = ml.flow_walk(ml.flow_bets(rows, col))
    sums = ml.flow_sums(bets)
    full = ml.flow_stats(list(sums.values()))
    print(label, "n bets", len(bets), "betas", [round(b, 2) for _, b in path])
    print("  as placed %.2f price %.2f moved %.2f change %+.2f | 2 books takes %.2f keeps %.2f diff %+.2f" % tuple(full))
    # decomposition: gain = sum s*won*(o_q - o_c); mech = sum s*q*(o_q-o_c) (if price q were right); info = sum s*(y-q)*(o_q-o_c)
    S = sum(r["stake"] for r in bets)
    mech = info = 0.0
    ll_q = ll_c = 0.0
    pos = neg = 0.0
    for r in bets:
        c = min(max(ml._sigmoid(ml._logit(r["q"]) + r["beta"] * r["x"]), 0.005), 0.995)
        o_q, o_c = r["odds"] * r["p"] / r["q"], r["odds"] * r["p"] / c
        d = o_q - o_c
        mech += r["stake"] * r["q"] * d
        info += r["stake"] * (r["y"] - r["q"]) * d
        if r["x"] > 0: pos += r["stake"] * r["y"] * d
        else: neg += r["stake"] * r["y"] * d
        y = r["y"]
        ll_q += r["stake"] * -(y * math.log(r["q"]) + (1 - y) * math.log(1 - r["q"]))
        ll_c += r["stake"] * -(y * math.log(c) + (1 - y) * math.log(1 - c))
    print("  change decomposed (pts of stake): mechanical-if-price-right %+.2f, information %+.2f; from x>0 bets %+.2f, x<=0 %+.2f" % (100*mech/S, 100*info/S, 100*pos/S, 100*neg/S))
    print("  stake-weighted log loss q %.5f moved %.5f" % (ll_q / S, ll_c / S))
