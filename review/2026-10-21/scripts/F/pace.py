import datetime as dt, numpy as np, pandas as pd
from eAMFCalibrator import pace_prior as pp
from eAMFCalibrator.tests.test_pace_prior import history_of
from xgb import pace_eff
history, drives = history_of(days=60, per_day=30)
ms = pp.matches_of(history)
cut = dt.datetime(2026, 4, 30)
train = [(t, c, g, tm, drives[c]) for t, c, g, tm in ms if t + pp.RESULT_LAG <= cut]
b0, ge, te, lg = pp.fit(train, cut)
df = pd.DataFrame(dict(start=[t for t, *_ in train], p1=[g[0] for _, _, g, _, _ in train], p2=[g[1] for _, _, g, _, _ in train],
                       t1=[tm[0] for *_, tm, _ in train], t2=[tm[1] for *_, tm, _ in train], drives=[d for *_, d in train]))
mat, beta = pace_eff.pace_fit(df, pace_eff.weights(df.start, pd.Timestamp(cut)).values / pace_eff.SCALAR)
gamers = sorted(set(df.p1) | set(df.p2))
print("pace_prior  FAST, SLOW effects:", round(ge["FAST"], 3), round(ge["SLOW"], 3))
print("pace_eff    FAST, SLOW effects:", round(beta[1 + gamers.index("FAST")], 3), round(beta[1 + gamers.index("SLOW")], 3))
