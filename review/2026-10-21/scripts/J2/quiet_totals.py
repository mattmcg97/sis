"""The same toy for bet_totals' 'score between bet and +60s' cut: prod's total line is the median of the
final total (half-point), bettors know nothing and pick over / under at random. level = side x (line at
+60s - line at the bet), as bet_totals.level."""
import math
import numpy as np
from math import exp, lgamma, log
rng = np.random.default_rng(2)
GAME, LAM = 1200.0, 2.85 / 1200.0
def pois(k, mu):
    return exp(k * log(mu) - mu - lgamma(k + 1)) if mu > 0 else (1.0 if k == 0 else 0.0)
def line(total, tau):
    mu = 2 * LAM * max(tau, 0)
    c = 0.0
    for k in range(40):
        c += pois(k, mu)
        if c >= 0.5:
            return total + 7 * k + 0.5 if c > 0.5 else total + 7 * k + 0.5
    return total + 0.5
out = {True: [], False: []}
for g in range(3000):
    n = rng.poisson(2 * LAM * GAME); times = np.sort(rng.uniform(0, GAME, n))
    for _ in range(10):
        t = rng.uniform(0, GAME - 60)
        s0, s1 = 7 * (times <= t).sum(), 7 * (times <= t + 60).sum()
        side = rng.choice([1, -1])
        lv = side * (line(s1, GAME - t - 60) - line(s0, GAME - t))
        out[bool(s1 != s0)].append((side, lv))
for sc, rs in out.items():
    for side in (1, -1):
        x = np.array([lv for s, lv in rs if s == side])
        print(f"{'a score in between' if sc else 'no score in between'}, {'Over ' if side == 1 else 'Under'}:"
              f" mean level +60s {x.mean():+.2f} pts, their way {100 * (x > 0).mean():.0f}%, against {100 * (x < 0).mean():.0f}% (n {len(x)})")
allx = np.array([lv for rs in out.values() for _, lv in rs])
print(f"every bet: mean level {allx.mean():+.3f}")
