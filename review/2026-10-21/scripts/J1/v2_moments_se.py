"""bet_moments.buckets' +-2se treats every bet as independent; bets of one match share its result.
Prod's price right (p true), operator margin 5%: how often is a bucket flagged red/green (margin more
than 2se from expected) when nothing is wrong?"""
import random

from eAMFCalibrator import bet_moments as bm

rng = random.Random(7)


def world(n_matches=300, bets_per=40, herd=0.8):
    rows = []
    for m in range(n_matches):
        p = rng.uniform(0.3, 0.7)               # prod's (true) chance of home
        home_won = rng.random() < p
        for _ in range(bets_per):
            home = rng.random() < (herd if p > 0.5 else 1 - herd)   # money herds onto one side
            ps = p if home else 1 - p
            odds = 0.95 / ps
            won = home == home_won
            stake = 10.0
            rows.append(dict(match_code=f"M{m}", in_play=True, excluded=False, result="won" if won else "lost",
                             stake=stake, revenue=stake - (stake * odds if won else 0.0), odds=odds,
                             stream_prob=ps, moment="between plays"))
    return rows


flag = 0
ratio = []
N = 200
margins = []
for _ in range(N):
    t = bm.buckets(world(), "moment")["between plays"]
    n, stake, rev, m, se2, exp = t
    margins.append(m - exp)
    flag += abs(m - exp) > se2
    ratio.append(se2)
import statistics
sd_true = statistics.pstdev(margins)
print(f"mean reported 2se {statistics.mean(ratio):.2f} pts; actual sd of (margin - expected) across worlds "
      f"{sd_true:.2f} pts -> true 2se {2 * sd_true:.2f}")
print(f"buckets flagged red/green with prod exactly right: {flag}/{N} (nominal ~5%)")
