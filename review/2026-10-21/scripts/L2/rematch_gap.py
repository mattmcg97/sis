import csv, datetime as dt
from collections import defaultdict, Counter
from eAMFCalibrator.bet_rematches import day_meetings
rows = list(csv.DictReader(open("eAMFCalibrator/out/match_history.csv", encoding="utf-8")))
meets = day_meetings(rows)
since = dt.datetime(2026, 8, 15); until = dt.datetime(2026, 10, 15)
by = defaultdict(list)
for c, m in meets.items():
    by[m["day_key"]].append((m["start"], c))
gaps = []
n_rematch = 0
for k, ms in by.items():
    ms.sort()
    for i in range(1, len(ms)):
        if not (since <= ms[i][0] < until):
            continue
        n_rematch += 1
        gaps.append((ms[i][0] - ms[i - 1][0]).total_seconds() / 60)
win = [m for m in meets.values() if since <= m["start"] < until]
print("matches in window", len(win), "rematches (2nd+ meeting)", n_rematch)
for cut in (36, 40, 45, 60):
    print(f"  gap to the previous meeting < {cut} min: {sum(g < cut for g in gaps)} ({100*sum(g < cut for g in gaps)/max(1,len(gaps)):.1f}%)")
print("  gap quantiles", sorted(gaps)[len(gaps)//10], sorted(gaps)[len(gaps)//4], sorted(gaps)[len(gaps)//2])
