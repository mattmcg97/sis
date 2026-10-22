import csv, math
base = "eAMFCalibrator/out/walkthrough/priors/"
inf = {r["MATCH_CODE"]: r for r in csv.DictReader(open(base + "nb2_infer.csv"))}
both = {r["MATCH_CODE"]: r for r in csv.DictReader(open(base + "nb2_both.csv"))}
xs = []
for c, r in inf.items():
    b = both.get(c)
    if not b or r["burn_in"] == "True": continue
    if int(float(r["meeting"] or 1)) <= 1: continue
    if r["home_score"] == "": continue
    am = int(r["home_score"]) - int(r["away_score"])
    em = float(r["exp_margin"]) - float(r["layer_margin"])   # nb2's own margin
    xs.append((float(r["layer_margin"]), float(b["layer_margin"]), am - em, float(r["layer_total"]), float(b["layer_total"])))
n = len(xs)
sd = lambda v: (sum(x * x for x in v) / len(v)) ** 0.5
print(n, "rematches; rms move margin: infer %.3f rematch %.3f; total: infer %.3f rematch %.3f" % (sd([x[0] for x in xs]), sd([x[1] for x in xs]), sd([x[3] for x in xs]), sd([x[4] for x in xs])))
def slope(i):
    sxy = sum(x[i] * x[2] for x in xs); sxx = sum(x[i] ** 2 for x in xs)
    return sxy / sxx
print("regression of nb2's margin miss on the move (1 = right size): infer %.2f, rematch %.2f" % (slope(0), slope(1)))
for lab, want in (("first meetings", False), ("rematches", True)):
    ys = []
    for c, r in inf.items():
        if r["burn_in"] == "True" or r["home_score"] == "": continue
        if (int(float(r["meeting"] or 1)) > 1) != want: continue
        mv = float(r["layer_margin"]); mt = float(r["layer_total"])
        if mv == 0 and mt == 0: continue
        am = int(r["home_score"]) - int(r["away_score"]); at = int(r["home_score"]) + int(r["away_score"])
        em = float(r["exp_margin"]) - mv; et = float(r["exp_total"]) - mt
        ys.append((mv, am - em, mt, at - et))
    s = lambda i, j: sum(x[i] * x[j] for x in ys) / sum(x[i] ** 2 for x in ys)
    print(lab, len(ys), "inference move slope: margin %.2f total %.2f; rms margin move %.2f" % (s(0, 1), s(2, 3), (sum(x[0]**2 for x in ys)/len(ys))**0.5))
