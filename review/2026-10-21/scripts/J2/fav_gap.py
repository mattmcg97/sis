"""bet_favourite.cell / line: 'margin' is over settled bets (won/lost/push), 'expected' over every priced
bet whatever its result -- cash-outs and 'other' payouts included. The gap compares two bet sets."""
from eAMFCalibrator import bet_favourite as bf, bets
def row(result, revenue):
    return dict(result=result, stake=100.0, revenue=revenue, priced=True, odds=1.9, stream_prob=0.5,
                expected_revenue=100.0 * (1 - 1.9 * 0.5), won={bets.WON: 1.0, bets.LOST: 0.0}.get(result))
# 20 bets at p 0.5: 10 would win, 10 would lose; 5 of the losers are cashed out mid-game (the bettor rescued them)
rows = [row(bets.WON, -90.0) for _ in range(10)] + [row(bets.LOST, 100.0) for _ in range(5)]
rows += [row(bets.CASHED, 30.0) for _ in range(5)]   # 5 would-be losers cashed out mid-game
c = bf.cell(rows)
print(bf.HEAD); print(bf.line("all", c, c[1]))
print("margin over", c[2], "settled stake; expected over", c[4], "priced stake")
