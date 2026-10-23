"""Pick each online family's member by TUNE log loss (from the tuning json) and write its preds."""
import common as C
import json, glob, os
import pandas as pd
H = C.H
tl = {os.path.basename(f)[9:-5]: json.load(open(f))["tune_ll"] for f in glob.glob(f"{C.OUT}/tune_dyn_*.json")}
P = pd.concat([pd.read_pickle(f).set_index("code") for f in glob.glob(f"{C.OUT}/dyn_preds_*.pkl")], axis=1)
P = P.loc[:, ~P.columns.duplicated()].reset_index()
fam = {"B_elo_tuned": lambda n: n.startswith("elo"), "B_glicko": lambda n: n.startswith("kal") or n == "kfull",
       "B_mov_elo": lambda n: n.startswith("mov_elo"), "B_mov_kal": lambda n: n.startswith("mov_kal")}
sel = {}
for out, f in fam.items():
    cands = {n: v for n, v in tl.items() if f(n) and n in P}
    best = min(cands, key=cands.get)
    sel[out] = dict(model=best, tune_ll=cands[best], candidates=dict(sorted(cands.items(), key=lambda kv: kv[1])))
    H.write_preds(P, best, out)
    print(out, "<-", best, round(cands[best], 5))
json.dump(sel, open(f"{C.OUT}/dyn_selected.json", "w"), indent=1)
