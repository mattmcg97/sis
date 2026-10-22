import datetime as dt, collections
from eAMFCalibrator import form_prior as fp
m = fp.read_matches("AMFELO_20261019.csv")
first_day = {}
teams_first = {}
for t, c, h, a, th, ta, hs, as_ in m:
    for g in (h, a):
        first_day.setdefault(g, t.date())
    for x in (th, ta):
        teams_first.setdefault(x, t.date())
sub = [x for x in m if x[0] >= dt.datetime(2026, 9, 1)]
new = [x for x in sub if first_day[x[2]] == x[0].date() or first_day[x[3]] == x[0].date()]
newt = [x for x in sub if teams_first[x[4]] == x[0].date() or teams_first[x[5]] == x[0].date()]
print("Sep-Oct matches", len(sub), "with a gamer on his first day", len(new), "team first day", len(newt))
# gaps between a gamer's consecutive matches
gaps = collections.Counter()
last = {}
for t, c, h, a, *_ in m:
    for g in (h, a):
        if g in last:
            d = (t - last[g]).total_seconds() / 60
            if d < 240:
                gaps[int(d // 5) * 5] += 1
        last[g] = t
tot = sum(gaps.values())
print("gap (min) share among <4h gaps:", [(k, round(v / tot, 3)) for k, v in sorted(gaps.items())[:16]])
