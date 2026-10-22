"""bet_money.features' momentum ('prod's move, the last minute') compares prod's over price now with its
over price 60s ago without asking whether the line is the same: a total that jumps 44.5 -> 51.5 on a
touchdown reads as no move at all; one that holds its line while the chance drifts reads as a move."""
import datetime as dt

from eAMFCalibrator import bet_money as bm, bets

T0 = dt.datetime(2026, 10, 5, 20, 0, 0)


def q(sec, p, line, msg):
    return ("M1", 54, T0 + dt.timedelta(seconds=sec), 100.0 * p, None, f"Over {line}", msg, "open", "true")


# the line jumps 7 points 30s before t, prod's over chance re-centred at the new line
tl = bets.timeline([q(0, 0.50, 44.5, 1), q(70, 0.51, 51.5, 2)])
t = T0 + dt.timedelta(seconds=100)
p_now = bets.price_at_time(tl, "M1", 54, t)
f = bm.features(tl, None, "M1", "total", t, p_now[1], p_now[2], True)
print("line 44.5 -> 51.5 (prod's expected total +7):  momentum", round(f["momentum"], 3))

# the line holds, the chance drifts 0.50 -> 0.60: a much smaller move in expected points
tl = bets.timeline([q(0, 0.50, 44.5, 1), q(70, 0.60, 44.5, 2)])
p_now = bets.price_at_time(tl, "M1", 54, t)
f = bm.features(tl, None, "M1", "total", t, p_now[1], p_now[2], True)
print("line held at 44.5, chance 0.50 -> 0.60:         momentum", round(f["momentum"], 3))
