import csv, sys, datetime as dt
from eAMFCalibrator import rematch_layer as rl, inference_layer as il
from eAMFCalibrator.bet_rematches import day_meetings
hist = list(csv.DictReader(open("eAMFCalibrator/out/match_history.csv", encoding="utf-8")))
meets = day_meetings(hist)
table = sys.argv[1] if len(sys.argv) > 1 else "eAMFCalibrator/out/walkthrough/priors/nb2.csv"
pts = {}
for r in csv.DictReader(open(table, encoding="utf-8")):
    try:
        pts[r["MATCH_CODE"]] = (float(r["Pred_P1_Points"]), float(r["Pred_P2_Points"]))
    except (KeyError, ValueError):
        pass
items = []
for code, m in meets.items():
    if code not in pts: continue
    a, b = pts[code]
    s1, s2 = m.get("home_score"), m.get("away_score")
    items.append(rl.item(code, m["start"], m["day_key"], m["home"], m["away"], m["team"][m["home"]], m["team"][m["away"]],
                         a - b, a + b, None if s1 is None or s2 is None else s1 - s2,
                         None if s1 is None or s2 is None else s1 + s2, m.get("gamer_keys")))
print(len(items), "items", min(i["start"] for i in items), max(i["start"] for i in items))
lay = il.InferenceLayer.fit(items)
print("gamer raw", {k: [round(x, 3) for x in v] for k, v in lay.gamer.items()})
print("pair raw", {k: [round(x, 3) for x in v] for k, v in lay.pair.items()})
print("sigma2", [round(x, 2) for x in lay.sigma2], "counts", lay.counts)
r = rl.Layer.fit(items)
print("rematch tau2", {k: [round(x, 3) for x in v] for k, v in r.tau2.items()}, "sigma2", [round(x, 2) for x in r.sigma2])
# what a rematch reads: total covariance of two same-team meetings of a pair, inference vs rematch
g, p = lay.use_gamer, lay.use_pair
for q, name in ((0, "margin"), (1, "total")):
    print(name, "rematch-same cov: inference", round(2 * g["same team"][q] + p["same teams"][q], 3),
          "rematch layer", round(r.tau2["same teams"][q], 3),
          "| changed: inference", round(2 * g["team changed"][q] + p["teams changed"][q], 3), "rematch", round(r.tau2["teams changed"][q], 3))
