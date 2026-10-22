import csv, collections
a = {r["MATCH_CODE"]: r for r in csv.DictReader(open("AMFELO_20261019.csv", encoding="utf-8"))}
b = {r["MATCH_CODE"]: r for r in csv.DictReader(open("eAMFCalibrator/out/match_history.csv", encoding="utf-8"))}
c = collections.Counter(); ex = collections.defaultdict(list)
for code in set(a) & set(b):
    ra, rb = a[code], b[code]
    fa = (ra["PLAYER_1_FINAL_SCORE"], ra["PLAYER_2_FINAL_SCORE"]); fb = (rb["PLAYER_1_FINAL_SCORE"], rb["PLAYER_2_FINAL_SCORE"])
    day = rb["SCHEDULED_START_TIME_UTC"][:10]
    after = day >= "2026-09-23"
    if fa == fb: k = "same"
    elif fa == fb[::-1]: k = "swapped"
    else: k = "different"
    hk = "handles_same" if (ra["PLAYER_1_HANDLE"], ra["PLAYER_2_HANDLE"]) == (rb["PLAYER_1_HANDLE"], rb["PLAYER_2_HANDLE"]) else "handles_diff"
    c[(after, k, hk)] += 1
    if k != "same" and len(ex[k]) < 6: ex[k].append((code, day, fa, fb, ra["PLAYER_1_HANDLE"], rb["PLAYER_1_HANDLE"]))
for k in sorted(c): print(k, c[k])
for k, v in ex.items(): print(k, v)
print(len(set(a)-set(b)), len(set(b)-set(a)))
