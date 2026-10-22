import csv, datetime as dt, collections
rows=list(csv.DictReader(open('eAMFCalibrator/out/match_history.csv',encoding='utf-8')))
by=collections.defaultdict(list)
for r in rows:
    t=dt.datetime.fromisoformat(r['SCHEDULED_START_TIME_UTC'][:19])
    if t < dt.datetime(2026,9,1): continue
    for k in (1,2):
        by[r[f'PLAYER_{k}_HANDLE'].strip().upper()].append(t)
gaps=[]
for g,ts in by.items():
    ts.sort()
    for a,b in zip(ts,ts[1:]):
        m=(b-a).total_seconds()/60
        if m<=180: gaps.append(m)
gaps.sort()
import statistics
print(len(gaps))
for q in (0.01,0.05,0.1,0.25,0.5,0.75,0.9):
    print(q, gaps[int(q*len(gaps))])
c=collections.Counter(int(x) for x in gaps if x<60)
print(sorted(c.items())[:40])
