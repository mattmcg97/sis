"""Run each tuned online model (frozen TUNE settings) over all matches; keep 2026-06-01..10-18 as preds.
Also the same models with results counted only from the next UTC midnight (the static fits' information)."""
import common as C
import json, sys, glob, os
import numpy as np, pandas as pd
import dyn

H = C.H
d = C.slim()
A = dyn._arrays(d)
Am = dyn._arrays(d, midnight=True)
FUNCS = {"elo": dyn.elo, "kalman": dyn.kalman, "kalman2": dyn.kalman2, "kalman3": dyn.kalman3, "kalman_full": dyn.kalman_full}
w = H.in_window(d, C.FULLWIN)
out = pd.DataFrame({"code": d.code})
names = sys.argv[1:] or [os.path.basename(f)[9:-5] for f in sorted(glob.glob(f"{C.OUT}/tune_dyn_*.json"))]
for name in names:
    cfg = json.load(open(f"{C.OUT}/tune_dyn_{name}.json"))
    kw = dict(cfg["kw"])
    if "mov" in kw and kw["mov"] is not None:
        kw["mov"] = tuple(kw["mov"])
    fn = FUNCS[cfg["fn"]]
    p = fn(d, A=A, **kw)
    pm = fn(d, A=Am, **kw)
    out[name] = np.where(w, p, np.nan)
    out[name + "__midnight"] = np.where(w, pm, np.nan)
    print(name, "TUNE ll", round(cfg["tune_ll"], 5), flush=True)
out.to_pickle(f"{C.OUT}/dyn_preds_{'_'.join(names) if sys.argv[1:] else 'all'}.pkl")
