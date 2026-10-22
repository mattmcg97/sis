import datetime as dt, collections
from eAMFCalibrator import form_prior as fp
m = fp.read_matches("AMFELO_20261019.csv")
last, sess_start = {}, {}
n = straddle = 0
hours = collections.Counter()
for t, c, h, a, *_ in m:
    hours[t.hour] += 1
    s = False
    for g in (h, a):
        if g in last and (t - last[g]) <= dt.timedelta(hours=2):
            if sess_start[g].date() < t.date():
                s = True
        else:
            sess_start[g] = t
        last[g] = t
    if t >= dt.datetime(2026, 9, 1):
        n += 1; straddle += s
print("Sep-Oct matches", n, "whose gamer's session began on an earlier UTC day:", straddle, round(straddle / n, 3))
print("matches by UTC hour:", sorted(hours.items()))
