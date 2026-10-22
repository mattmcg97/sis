"""A calibrated in-play moneyline (exact martingale: scores are Poisson, 7 points each), bettors with NO
information who back the favourite 60% of the time. bet_ml_moves reports the +60s move 'with no score
in the 60s after' as 'prod re-pricing the same state: the drift a bettor was ahead of'. Here there is
no information at all, so any non-zero mean there is the conditioning on the future (no score)."""
import math
import numpy as np
rng = np.random.default_rng(1)
GAME = 1200.0                       # wall seconds of play
LAM = 2.85 / GAME                   # scores per second per side (~40 points a match)
KMAX = 25
from math import exp, lgamma, log
def pois(k, mu):
    return exp(k * log(mu) - mu - lgamma(k + 1)) if mu > 0 else (1.0 if k == 0 else 0.0)
def p_home(lead, tau):
    """P(home ends ahead) + half a draw, exactly, under the Poisson scoring."""
    mu = LAM * max(tau, 0.0)
    ph = [pois(k, mu) for k in range(KMAX)]
    win = draw = 0.0
    for a in range(KMAX):
        for b in range(KMAX):
            m = lead + 7 * (a - b)
            w = ph[a] * ph[b]
            if m > 0: win += w
            elif m == 0: draw += w
    return win + 0.5 * draw
rows = []
for g in range(1500):
    # the score path: event times and who scored
    n = rng.poisson(2 * LAM * GAME)
    times = np.sort(rng.uniform(0, GAME, n)); who = rng.choice([1, -1], n)
    def lead_at(t):
        return int(7 * who[times <= t].sum())
    for _ in range(20):
        t = rng.uniform(0, GAME - 60)
        L0, L1 = lead_at(t), lead_at(t + 60)
        p0, p1 = p_home(L0, GAME - t), p_home(L1, GAME - t - 60)
        fav_home = p0 >= 0.5
        back_fav = rng.random() < 0.6
        home = fav_home == back_fav
        q0, q1 = (p0, p1) if home else (1 - p0, 1 - p1)
        scored = bool(((times > t) & (times <= t + 60)).any())
        rows.append((q1 - q0, scored, back_fav, q0))
mv = np.array([r[0] for r in rows]); sc = np.array([r[1] for r in rows]); bf = np.array([r[2] for r in rows])
q0 = np.array([r[3] for r in rows])
def m(x): return f"{100 * x.mean():+.2f} pts (2se {200 * x.std() / math.sqrt(len(x)):.2f}, n {len(x)})"
print("all bets, +60s move:            ", m(mv))
print("no score in the 60s after:      ", m(mv[~sc]))
print("  backing the favourite:        ", m(mv[~sc & bf]))
print("  backing the underdog:         ", m(mv[~sc & ~bf]))
print("a score in between:             ", m(mv[sc]))
print("'their way' (>=1pt) share, no score:", f"{100*(mv[~sc] >= .01).mean():.0f}%", " against:", f"{100*(mv[~sc] <= -.01).mean():.0f}%")
