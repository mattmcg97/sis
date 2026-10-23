"""(b') the mixed model on the total: gamer and team "total" effects as random effects (multi-membership, the
match's two gamers on one effect each), stream fixed, the variances by REML at each daily refit (xgb/pace_models'
mixed_fit, imported read-only), rows weighted 2^(-age/30) (b_tot's TUNE half-life), 365 days. Also the drives.
Writes out/F/mixed.pkl (code, b_mixed, drives_mixed) and the REML shrinkage path.
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402
sys.path.insert(0, F.ROOT)
from xgb import pace_models as PM  # noqa: E402

df = F.panel()
m = df[["code", "start", "p1", "p2", "t1", "t2", "stream", "total", "D", "day", "settled"]].copy()
m["stream"] = m.stream.astype(str)
out = pd.DataFrame({"code": df.code, "b_mixed": np.nan, "drives_mixed": np.nan})
lams = []
for target, col, start, hl, td in (("total", "b_mixed", "2026-05-01", 30.0, 365), ("D", "drives_mixed", "2026-05-08", 15.0, 120)):
    st = (math.log(30.0), math.log(300.0))
    for d in F.days_between(m, start, "2026-10-19"):
        cut = pd.Timestamp(d)
        tr = m[(m.settled <= cut) & (m.start >= cut - pd.Timedelta(days=td)) & m[target].notna()]
        if len(tr) < 200:
            continue
        age = (cut - tr.start).dt.total_seconds().values / 86400
        w = np.power(2.0, -age / hl)
        w = w / w.mean()
        des = PM.Design(tr, stream=True)
        X = des.matrix(tr)
        beta, st = PM.mixed_fit(des, X, tr[target].values.astype(float), w, st)
        te = m[m.day == d]
        out.loc[te.index, col] = des.matrix(te) @ beta
        lams.append(dict(target=target, day=d, lam_g=float(np.exp(st[0])), lam_t=float(np.exp(st[1]))))
        if cut.day == 1:
            print(target, d, np.exp(st), flush=True)
out.to_pickle(os.path.join(F.OUT, "mixed.pkl"))
pd.DataFrame(lams).to_csv(os.path.join(F.OUT, "mixed_lams.csv"), index=False)
d = df[["code", "day", "total", "D"]].merge(out, on="code")
for w in ("TUNE", "TEST", "SEP"):
    print(F.total_table(d, ["b_mixed"], F.WINDOWS[w]))
    print(F.total_table(d, ["drives_mixed"], F.WINDOWS[w], target="D"))
