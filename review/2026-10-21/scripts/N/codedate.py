import csv, datetime as dt, collections
rows=list(csv.DictReader(open('eAMFCalibrator/out/match_history.csv',encoding='utf-8')))
c=collections.Counter(); ex=[]
for r in rows:
    code=r['MATCH_CODE']; s=code[-6:]
    try: d=dt.date(2000+int(s[4:6]),int(s[2:4]),int(s[0:2]))
    except Exception: c['bad']+=1; continue
    t=dt.datetime.fromisoformat(r['SCHEDULED_START_TIME_UTC'][:19])
    diff=(d - t.date()).days
    c[diff]+=1
    if diff!=0 and len(ex)<5: ex.append((code,r['SCHEDULED_START_TIME_UTC']))
print(c, ex)
# hour distribution
h=collections.Counter(dt.datetime.fromisoformat(r['SCHEDULED_START_TIME_UTC'][:19]).hour for r in rows)
print(sorted(h.items()))
