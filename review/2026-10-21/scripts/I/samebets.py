"""Same-bets reprice and two books on real-style bets: punters bet where prod is generous (informed), the
candidate is (a) equally good as prod, independent errors; (b) worse than prod (twice the noise); (c) prod
itself plus noise. Rows go through bets.reprice / bets.summarise / book_comparison.compute as in a run."""
import sys; sys.path.insert(0, ".")
import numpy as np
from eAMFCalibrator import bets, book_comparison as bc, config
config.SIM_BOOT = 300
rng = np.random.default_rng(0)
lg = lambda p: np.log(p / (1 - p)); sg = lambda z: 1 / (1 + np.exp(-z))

def book(kind, n_matches=800, moments=40, sd=0.20, thr=0.10, margin=0.085, seed=0, frac_informed=0.5):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        code = f"AF{m:04d}"
        for k in range(moments):
            z = rng.normal(0, 1.2)                     # truth, home side, logit
            ep = rng.normal(0, sd)
            zp = z + ep
            if kind == "equal": zc = z + rng.normal(0, sd)
            elif kind == "worse": zc = z + rng.normal(0, 2 * sd)
            elif kind == "prod+noise": zc = zp + rng.normal(0, sd)
            elif kind == "perfect": zc = z
            # a punter backs the side prod is generous on, when it is by more than thr (informed), else nothing
            if rng.random() < frac_informed:
                if abs(ep) < thr: continue
                side = 0 if ep < 0 else 1              # prod too low on home -> back home
            else:
                if rng.random() < 0.6: continue
                side = int(rng.random() < 0.5)         # a recreational punter: either side
            pi = sg(z) if side == 0 else 1 - sg(z)
            p = sg(zp) if side == 0 else 1 - sg(zp)
            c = sg(zc) if side == 0 else 1 - sg(zc)
            odds = 1 / (p * (1 + margin)); stake = 10.0
            won = rng.random() < pi
            res = bets.WON if won else bets.LOST
            rev = stake - (stake * odds if won else 0.0)
            row = dict(match_code=code, in_play=True, market="moneyline", stake=stake, odds=odds, revenue=rev,
                       result=res, candidate_result=res, stream_prob=p, candidate_prob=c, on_prod_line=True,
                       same_line=True, CUSTOMER_TEMPERATURE="Standard")
            row.update(bets.reprice(row))
            rows.append(row)
    return rows

for kind in ("perfect", "equal", "worse", "prod+noise"):
    rows = book(kind)
    s = bets.summarise(rows)["all"]
    e = bc.compute([(kind, rows)])[0]
    two = [r for r in e["two"] if r["label"] == "all"][0]
    acc = [r for r in e["accept"] if r["label"] == "all"][0]
    print(f"{kind:11s} bets {s[0]:6d}  prod margin {s[3]:+6.2f}%  same bets {s[5]:+6.2f}%  change {s[5]-s[3]:+6.2f} "
          f"[{acc['change'][1]:+.2f}, {acc['change'][2]:+.2f}] | two books: to cand {100*two['to_cand']:4.1f}%  cand - prod {(two['head'] or (float('nan'),)*3)[0]:+6.2f} [{(two['head'] or (float('nan'),)*3)[1]:+.2f}, {(two['head'] or (float('nan'),)*3)[2]:+.2f}]  both - alone {two['change'][0]:+.2f}")
