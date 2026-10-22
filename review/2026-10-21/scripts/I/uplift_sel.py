"""Margin uplift at the bets' moments: prod and a candidate equally good (truth + independent noise of the
same size). Ticks where prod is off are where bettors bet; the candidate is priced there alone (at_bets)."""
import sys; sys.path.insert(0, ".")
import datetime as dt, numpy as np
from scipy.stats import norm
from eAMFCalibrator import margin_uplift as mu

def make(n=600, T=60, sd=0.35, thr=0.35, seed=0):
    rng = np.random.default_rng(seed)
    prod_rows, cand_all, cand_bets, starts, finals = [], [], [], {}, {}
    t0 = dt.datetime(2026, 10, 1)
    for m in range(n):
        code = f"AF{m:03d}011026"
        start = t0 + dt.timedelta(hours=m)
        starts[code] = start
        inc = rng.normal(0, 1, T)
        mu0 = rng.normal(0, 1.0)
        path = mu0 + np.cumsum(inc)
        final = path[-1]
        finals[code] = (1.0, 0.0) if final > 0 else (0.0, 1.0)
        for k in range(T - 1):
            rem = T - 1 - k
            pi = norm.cdf(path[k] / np.sqrt(rem))
            pi = min(max(pi, 1e-4), 1 - 1e-4)
            lp = np.log(pi / (1 - pi))
            ep, ec = rng.normal(0, sd), rng.normal(0, sd)
            p = 1 / (1 + np.exp(-(lp + ep))); c = 1 / (1 + np.exp(-(lp + ec)))
            when = start + dt.timedelta(minutes=1 + 25 * k / T)
            for mk, pp, cc in ((50, p, c), (51, 1 - p, 1 - c)):
                prod_rows.append((code, mk, when, 100 * pp, None, "", k, None, "true"))
                cand_all.append((code, mk, when, 100 * cc, None, "", k, None, "true"))
                if abs(ep) > thr:
                    cand_bets.append((code, mk, when, 100 * cc, None, "", k, None, "true"))
    return prod_rows, cand_all, cand_bets, starts, finals

for seed in (0, 1, 2):
    prod_rows, cand_all, cand_bets, starts, finals = make(seed=seed)
    prod = mu.Ticks("prod").add(prod_rows, starts, finals)
    a = mu.Ticks("cand").add(cand_all, starts, finals)
    b = mu.Ticks("cand@bets").add(cand_bets, starts, finals); b.at_bets, b.max_age = True, 0.0
    for name, feed in (("every tick", a), ("bets' moments", b)):
        out = mu.compute([("prod", prod), (name, feed)], boot=300)
        st = out["cands"][0]["groups"]["Moneyline"]["all"]
        print(f"seed {seed} {name:14s} ticks {st['bets']:6d} to cand {st['to_cand']:5.1f}%  takes {st['takes']:+6.2f} keeps {st['keeps']:+6.2f}  "
              f"candidate - prod {st['head'][0]:+6.2f} [{st['head'][1]:+.2f}, {st['head'][2]:+.2f}]  prod alone {st['alone']:+.2f}  ll diff {st['ll_diff'][0]:+.4f}")
