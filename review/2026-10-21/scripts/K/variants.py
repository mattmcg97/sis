import csv, math
from collections import defaultdict
base = "eAMFCalibrator/out/walkthrough/priors/"
names = ["nb2", "nb2_infer", "nb2_both", "glmer", "glmer_both", "avg", "avg-plain"]
tab = {}
for n in names:
    tab[n] = {r["MATCH_CODE"]: r for r in csv.DictReader(open(base + n + ".csv", encoding="utf-8"))}
codes = set.intersection(*[set(t) for t in tab.values()])
rows = []
for c in codes:
    r = tab["nb2"][c]
    if r["burn_in"] == "True" or r["home_score"] in ("", None): continue
    hs, as_ = int(r["home_score"]), int(r["away_score"])
    if hs == as_: continue
    y = 1.0 if hs > as_ else 0.0
    ps = {n: float(tab[n][c]["p_home"]) for n in names}
    ps["nb2_both+glmer_both(p avg)"] = (ps["nb2_both"] + ps["glmer_both"]) / 2
    ps["nb2_infer+glmer_both(p avg)"] = (ps["nb2_infer"] + ps["glmer_both"]) / 2
    ps["nb2+glmer(p avg)"] = (ps["nb2"] + ps["glmer"]) / 2
    rows.append((r["start"], y, ps))
print(len(rows), "matches", min(r[0] for r in rows), max(r[0] for r in rows))
ll = defaultdict(float)
for _, y, ps in rows:
    for n, p in ps.items():
        p = min(max(p, 1e-6), 1 - 1e-6)
        ll[n] += -(y * math.log(p) + (1 - y) * math.log(1 - p))
b = ll["nb2"] / len(rows)
for n in sorted(ll, key=ll.get):
    print(f"{n:30s} {ll[n]/len(rows):.4f}  vs nb2 {ll[n]/len(rows)-b:+.4f}")
def lls(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))
for a, bname in (("nb2_infer", "nb2"), ("avg", "nb2"), ("nb2_both", "nb2_infer"), ("avg", "nb2_both+glmer_both(p avg)"), ("nb2_infer", "nb2_both")):
    d = [lls(ps[a], y) - lls(ps[bname], y) for _, y, ps in rows]
    m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    print(f"{a} - {bname}: {m:+.4f}  t {m / (sd / len(d) ** 0.5):+.1f}")
# rematch vs first meetings for nb2_infer vs nb2
for lab, keep in (("rematch", lambda c: int(tab['nb2'][c]['meeting'] or 1) > 1),):
    pass
codes_by = {}
for c in codes:
    r = tab["nb2"][c]
    if r["burn_in"] == "True" or r["home_score"] in ("", None): continue
    hs, as_ = int(r["home_score"]), int(r["away_score"])
    if hs == as_: continue
    y = 1.0 if hs > as_ else 0.0
    re = int(float(r["meeting"] or 1)) > 1
    for a, b in (("nb2_infer", "nb2"), ("nb2_both", "nb2"), ("nb2_infer", "nb2_both")):
        codes_by.setdefault((a, b, re), []).append(lls(float(tab[a][c]["p_home"]), y) - lls(float(tab[b][c]["p_home"]), y))
for k, d in sorted(codes_by.items()):
    m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    print(k, len(d), f"{m:+.4f} t {m / (sd / len(d) ** 0.5) if sd else 0:+.1f}")
