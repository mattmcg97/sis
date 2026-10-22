"""bet_moments / bet_losers / bet_money turn naive UTC datetimes into epoch seconds with .timestamp(),
which reads a naive datetime as the machine's LOCAL time. On a UK machine, the hour 01:00-02:00 UTC on
the clocks-back Sunday (2026-10-25) is stretched by an hour."""
import datetime as dt

a = dt.datetime(2026, 10, 25, 1, 59, 50)    # naive UTC: the price's message
b = dt.datetime(2026, 10, 25, 2, 0, 30)     # naive UTC: the bet, 40 s later
print("true gap (datetime subtraction):", (b - a).total_seconds(), "s")
print("gap via .timestamp() (as moment_of's price_age / Money.at's window):", b.timestamp() - a.timestamp(), "s")
