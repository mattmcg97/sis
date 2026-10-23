"""Walk-forward tuning of the static BT on TUNE. Usage: python tune_bt.py <stage> [n_jobs]"""
import common as C
import json, sys, time, os, itertools
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import btlib as B

H = C.H
COLS = ["code", "start", "day", "y", "margin", "g1", "g2", "k1", "k2", "st", "tday"]


def run(cfg, window=H.TUNE):
    d = C.slim()[COLS]
    f, p = B.make_bt(cfg["spec"], cfg["lam"], cfg["hl"], cfg.get("mov"))
    t = time.time()
    pr = H.walk_forward(d, f, p, *window)
    w = H.in_window(d, window) & d.y.notna() & pr.notna()
    m = H.metrics(d.y[w], pr[w])
    return dict(cfg=json.dumps(cfg), n=m["n"], logloss=m["logloss"], brier=m["brier"], auc=m["auc"],
                slope=m["cal_slope"], secs=time.time() - t)


def grid(stage):
    G = []
    if stage == "g":
        for hl, lg in itertools.product([15, 30, 45, 60, 90, 120, 180, 270, None], [1, 2, 4, 8, 16]):
            G.append(dict(spec={"home": "one"}, lam={"gamer": lg}, hl=hl))
    if stage == "g2":
        for hl, lg in itertools.product([20, 25, 38, 45], [0.25, 0.5, 0.75, 1, 1.5]):
            G.append(dict(spec={"home": "one"}, lam={"gamer": lg}, hl=hl))
        for lg in [0.25, 0.5, 0.75, 1.5]:
            G.append(dict(spec={"home": "one"}, lam={"gamer": lg}, hl=30))
    if stage == "t":
        for hl, lg, lt, home in itertools.product([30, 45], [0.5, 1], [1, 3, 10, 30, 100], ["one", "stream"]):
            if home == "one" and (hl != 30 or lg != 0.5):
                continue
            G.append(dict(spec={"home": home, "team": True}, lam={"gamer": lg, "team": lt}, hl=hl))
        G.append(dict(spec={"home": "stream"}, lam={"gamer": 0.5}, hl=30))
    if stage == "x":
        base = {"home": "stream", "team": True}
        bl = {"gamer": 0.5, "team": 3}
        for hl, lg in [(20, 0.5), (25, 0.5), (25, 0.75), (30, 0.25), (30, 0.75), (38, 0.5)]:
            G.append(dict(spec=base, lam={"gamer": lg, "team": 3}, hl=hl))
        for l in [1, 3, 10, 30, 100]:
            G.append(dict(spec={**base, "gt": True}, lam={**bl, "gt": l}, hl=30))
        for l in [3, 10, 30, 100]:
            G.append(dict(spec={**base, "ghome": True}, lam={**bl, "ghome": l}, hl=30))
            G.append(dict(spec={**base, "pair": True}, lam={**bl, "pair": l}, hl=30))
    if stage == "x2":
        base = {"home": "stream", "team": True, "pair": True}
        for hl, lg, lp in [(30, 0.5, 5), (30, 0.5, 7), (30, 0.5, 15), (25, 0.5, 10), (25, 0.5, 7), (30, 0.4, 10),
                           (30, 0.65, 10), (35, 0.5, 10)]:
            G.append(dict(spec=base, lam={"gamer": lg, "team": 3, "pair": lp}, hl=hl))
    if stage == "mov":
        base = {"home": "stream", "team": True}
        for a, c in itertools.product([0.25, 0.5, 1.0], [1, 3, 7]):
            G.append(dict(spec=base, lam={"gamer": 0.5, "team": 3}, hl=30, mov=[a, c]))
        for a in [-0.25, 0.1]:
            G.append(dict(spec=base, lam={"gamer": 0.5, "team": 3}, hl=30, mov=[a, 3]))
    return G


if __name__ == "__main__":
    stage = sys.argv[1]
    nj = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    G = grid(stage) if stage != "file" else [json.loads(l) for l in open(sys.argv[3])]
    out = f"{C.OUT}/tune_bt_{stage if stage != 'file' else os.path.basename(sys.argv[3])}.csv"
    res = Parallel(n_jobs=nj, verbose=5)(delayed(run)(g) for g in G)
    r = pd.DataFrame(res).sort_values("logloss")
    r.to_csv(out, index=False)
    print(r.head(20).to_string())
