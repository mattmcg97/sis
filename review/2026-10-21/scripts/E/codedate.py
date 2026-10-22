import csv, datetime as dt, collections
from eAMFModel.remaining import day
rows = list(csv.DictReader(open("AMFELO_20261019.csv", encoding="utf-8")))
diff = collections.Counter(); hours = collections.defaultdict(collections.Counter)
bad = 0
ex = []
for r in rows:
    try:
        d = day(r["MATCH_CODE"])
    except Exception as e:
        bad += 1; continue
    t = dt.datetime.fromisoformat(r["SCHEDULED_START_TIME_UTC"][:19])
    k = (t.date() - d).days
    diff[k] += 1
    hours[k][t.hour] += 1
    if k != 0 and len(ex) < 8: ex.append((r["MATCH_CODE"], r["SCHEDULED_START_TIME_UTC"]))
print("unparsable", bad)
print("utc date - code date:", diff)
for k in hours: print(k, sorted(hours[k].items()))
print(ex)
# first/last hour of day for code-date==utc-date
