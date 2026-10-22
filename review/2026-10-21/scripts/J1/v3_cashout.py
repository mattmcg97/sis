"""Dropping cashed-out bets from the residual (bet_losers.bet_row: won is None for CASHED, so report's
scope leaves them out; bet_moments._in_scope likewise) conditions on a decision taken after the bet.
Prod exactly right; bettors cash out when their side has fallen behind at half time."""
import random

from eAMFCalibrator import bet_losers as bl

rng = random.Random(3)
rows_all, rows_kept = [], []
for m in range(3000):
    p = rng.uniform(0.3, 0.7)                     # prod's (true) pre-bet chance of home
    # half-time chance of home moves around p; final drawn from it (a martingale)
    h = min(0.98, max(0.02, p + rng.gauss(0, 0.2)))
    home_won = rng.random() < h
    for k in range(5):
        home = rng.random() < 0.5
        ps = p if home else 1 - p
        side_h = h if home else 1 - h
        won = float(home == home_won)
        odds = 0.93 / ps
        r = dict(match_code=f"M{m}", stake=10.0, revenue=10.0 - (10.0 * odds if won else 0.0), odds=odds,
                 p=ps, won=won, res=won - ps, expected=1 - odds * ps)
        rows_all.append(r)
        if not (side_h < 0.25 and rng.random() < 0.5):   # half of those behind cash out
            rows_kept.append(r)

for name, rs in (("every bet (would-be result)", rows_all), ("cash-outs left out (as ml-losers)", rows_kept)):
    s = bl.stats(rs)
    print(f"{name:36s} bets {s['bets']:6d}  resid {100 * s['res']:+.2f} pts  2se {100 * s['se']:.2f}  "
          f"margin {100 * s['margin']:+.2f}%  expected {100 * s['expected']:+.2f}%")
