"""Frozen static-BT settings (argmin TUNE log loss within each family) run walk-forward, daily refits,
2026-06-01..10-18; preds for the families, a pickle of every variant for the tables."""
import common as C
import json, glob
import numpy as np, pandas as pd
import btlib as B

H = C.H
COLS = ["code", "start", "day", "y", "margin", "g1", "g2", "k1", "k2", "st", "tday"]
d = C.slim()[COLS]
r = pd.concat([pd.read_csv(f) for f in glob.glob(f"{C.OUT}/tune_bt_*.csv")], ignore_index=True)
r["c"] = r.cfg.map(json.loads)
r["fam"] = r.c.map(lambda c: "mov" if c.get("mov") else
                   ("+".join(k for k in ["team", "gt", "ghome", "pair"] if c["spec"].get(k)) or "gamer")
                   + ("|stream" if c["spec"].get("home") == "stream" else ""))
r = r.drop_duplicates("cfg").sort_values("logloss")
best = r.groupby("fam").head(1).set_index("fam")
print(best[["logloss", "slope", "cfg"]].to_string())
want = {"bt": "gamer", "bt_stream": "gamer|stream", "bt_team": "team|stream", "bt_team_gt": "team+gt|stream",
        "bt_team_ghome": "team+ghome|stream", "bt_team_pair": "team+pair|stream", "bt_mov": "mov"}
out = pd.DataFrame({"code": d.code})
sel = {}
for name, fam in want.items():
    if fam not in best.index:
        continue
    c = best.loc[fam, "c"]
    f, p = B.make_bt(c["spec"], c["lam"], c["hl"], c.get("mov"))
    out[name] = H.walk_forward(d, f, p, *C.FULLWIN).to_numpy()
    sel[name] = dict(cfg=c, tune_ll=float(best.loc[fam, "logloss"]))
    print(name, c, flush=True)
out.to_pickle(f"{C.OUT}/bt_preds.pkl")
json.dump(sel, open(f"{C.OUT}/bt_selected.json", "w"), indent=1)
dd = C.slim()[["code"]].merge(out, on="code")
H.write_preds(dd, "bt", "B_bt")
# the strongest static BT on TUNE among the team families is B_bt_team
tf = {k: v["tune_ll"] for k, v in sel.items() if k.startswith("bt_team")}
bt_best = min(tf, key=tf.get)
print("B_bt_team <-", bt_best)
H.write_preds(dd, bt_best, "B_bt_team")
if "bt_mov" in out:
    H.write_preds(dd, "bt_mov", "B_bt_mov")
