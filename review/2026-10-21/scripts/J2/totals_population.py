"""bet_totals.population: 'expected' (what prod's probability says the book keeps) is taken over every
settled bet with a stream_prob, including bets struck at another line than prod's (on_prod_line False),
where stream_prob is prod's chance at ITS line, not the bet's."""
from eAMFCalibrator import bet_totals as bt, bets, config
def r(line_bet, on_line, won):
    # Over at the bet's line, odds 1.9; prod's line then 42.5 with P(over 42.5) = 0.5
    return {"group": bt.RESTRICTED, "result": bets.WON if won else bets.LOST, "stake": 100.0,
            "revenue": -90.0 if won else 100.0, "odds": 1.9, "stream_prob": 0.5, "stream_line": 42.5,
            "bet_line_prod_side": line_bet, "on_prod_line": on_line, "side": "Over", "in_play": True,
            "CUSTOMER_NAME_HASH": None}
# 10 bets at prod's line (5 win: fair), 10 at a stale 40.5 that wins 7 of 10
rows = [r(42.5, True, k < 5) for k in range(10)] + [r(40.5, False, k < 7) for k in range(10)]
n, c, stake, m, exp, over, ip, online = bt.population(rows)[bt.RESTRICTED]
print(f"population: margin {m:+.1f}%, expected {exp:+.1f}% (on prod's line {online:.0f}%)")
on = [x for x in rows if x["on_prod_line"]]
print("on prod's line only: margin %+.1f%%, expected %+.1f%%" % bt.money(on)[2:])
