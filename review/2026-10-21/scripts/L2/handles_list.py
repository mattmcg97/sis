import csv
from collections import defaultdict
rows = list(csv.DictReader(open("eAMFCalibrator/out/match_history.csv", encoding="utf-8")))
info = defaultdict(lambda: [0, None, None, set()])
raw = defaultdict(set)
for r in rows:
    for n in ("1", "2"):
        h = r[f"PLAYER_{n}_HANDLE"]
        raw[h.strip().upper()].add(h)
        g = info[h.strip().upper()]
        g[0] += 1
        t = r["SCHEDULED_START_TIME_UTC"][:10]
        g[1] = t if g[1] is None or t < g[1] else g[1]
        g[2] = t if g[2] is None or t > g[2] else g[2]
        g[3].add(r[f"PLAYER_{n}_TEAM"])
print(len(info), "handles")
for h, (n, a, b, teams) in sorted(info.items(), key=lambda x: x[1][1]):
    print(f"{h:16s} {n:5d} {a} {b} teams={len(teams)} raw={sorted(raw[h]) if len(raw[h])>1 else ''}")
