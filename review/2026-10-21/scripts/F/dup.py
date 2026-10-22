import collections, datetime as dt
from eAMFCalibrator import form_prior as fp
m = fp.read_matches("AMFELO_20261019.csv")
c = collections.Counter()
for t, code, h, a, *_ in m:
    c[(h, t)] += 1; c[(a, t)] += 1
d = [(k, v) for k, v in c.items() if v > 1]
print("gamer at two matches with the same kick-off:", len(d), d[:5])
# gaps below 30 min
last = {}; short = []
for t, code, h, a, *_ in m:
    for g in (h, a):
        if g in last and (t - last[g][0]) < dt.timedelta(minutes=30):
            short.append((g, last[g][1], code, (t - last[g][0]).total_seconds() / 60))
        last[g] = (t, code)
print("consecutive matches of a gamer < 30 min apart:", len(short), short[:5])
same_pair = 0
