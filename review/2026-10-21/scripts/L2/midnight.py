"""Matches whose 'as of the start of the UTC day' stats include a result not in at their kick-off
(a match that kicked off within 36 minutes before midnight, result in 36 min after kick-off)."""
import csv, datetime as dt
from collections import defaultdict
rows = list(csv.DictReader(open("eAMFCalibrator/out/match_history.csv", encoding="utf-8")))
LAG = dt.timedelta(minutes=36)
ms = []
for r in rows:
    t = dt.datetime.fromisoformat(r["SCHEDULED_START_TIME_UTC"][:19])
    ms.append((t, r["MATCH_CODE"], r["PLAYER_1_HANDLE"].strip().upper(), r["PLAYER_2_HANDLE"].strip().upper()))
ms.sort()
since, until = dt.datetime(2026, 8, 15), dt.datetime(2026, 10, 15)
win = [m for m in ms if since <= m[0] < until]
# any match of the previous day whose result came in after this match's kick-off
late_prev = defaultdict(list)   # day -> previous day's matches kicked off within LAG of midnight
for t, c, a, b in ms:
    nxt = dt.datetime.combine(t.date() + dt.timedelta(days=1), dt.time())
    if nxt - t < LAG:
        late_prev[t.date() + dt.timedelta(days=1)].append((t, a, b))
any_unknown = own_unknown = 0
for t, c, a, b in win:
    prev = [p for p in late_prev.get(t.date(), []) if p[0] + LAG > t]
    if prev:
        any_unknown += 1
        if any(x in (a, b) for p in prev for x in p[1:]):
            own_unknown += 1
print(f"{len(win)} window matches; reading a previous-day result not yet in at kick-off: "
      f"any gamer {any_unknown} ({100*any_unknown/len(win):.2f}%), one of its own gamers {own_unknown} "
      f"({100*own_unknown/len(win):.2f}%)")
