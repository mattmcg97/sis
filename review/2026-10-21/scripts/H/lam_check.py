"""pace_eff's pace ridge vs pace_prior's (production) and pace_models' 'ridge': same train rows, same cut."""
import datetime as dt
import numpy as np, pandas as pd
from xgb import pace_eff as pe, pace_models as pm
from eAMFCalibrator import pace_prior as pp

rng = np.random.default_rng(0)
n, G, T = 3000, 40, 8
g = rng.normal(0, 0.35, G); tm = rng.normal(0, 0.1, T)
p1, p2 = rng.integers(0, G, n), rng.integers(0, G, n); p2 = np.where(p2 == p1, (p2 + 1) % G, p2)
t1, t2 = rng.integers(0, T, n), rng.integers(0, T, n)
cut = pd.Timestamp("2026-09-01")
start = cut - pd.to_timedelta(rng.uniform(0.1, 179, n), unit="D")
drives = 10 + g[p1] + g[p2] + tm[t1] + tm[t2] + rng.normal(0, 2.6, n)
df = pd.DataFrame({"code": [f"M{i}" for i in range(n)], "start": start, "p1": [f"G{i}" for i in p1], "p2": [f"G{i}" for i in p2],
                   "t1": [f"T{i}" for i in t1], "t2": [f"T{i}" for i in t2], "drives": drives, "stream": "1"})
# pace_eff, exactly as walk() calls it
matrix, b_pe = pe.pace_fit(df, pe.weights(df.start, cut).values / pe.SCALAR)
eff = b_pe[1:1 + G]
# production pace_prior.fit
train = [(s.to_pydatetime(), c, (a, b), (x, y), d) for s, c, a, b, x, y, d in
         zip(df.start, df.code, df.p1, df.p2, df.t1, df.t2, df.drives)]
b0, ge, te, lg = pp.fit(train, cut.to_pydatetime())
names = sorted(f"G{i}" for i in range(G))
prod = np.array([ge[k] for k in names])
# pace_models' ridge
d = pm.Design(df, stream=False)
b_pm = pm.ridge_fit(d, d.matrix(df), df.drives.values, pm.PACE_SCALAR * pm.weights(df.start, cut), 45.0, 180.0)
pmr = b_pm[1:1 + G]
print("weights mean: pace_eff pace", (pe.weights(df.start, cut).values / pe.SCALAR).mean().round(3),
      " pace_prior", (pp.SCALAR).__round__(3), " pace_models ridge", (pm.PACE_SCALAR * pm.weights(df.start, cut)).mean().round(3))
print("gamer-effect sd: pace_eff", eff.std().round(4), " pace_prior (prod)", prod.std().round(4), " pace_models ridge", pmr.std().round(4),
      " true", g.std().round(4))
truth = np.array([g[int(k[1:])] for k in names])
for nm, v in (("pace_eff", eff), ("prod", prod)):
    print(nm, "slope of true effect on estimate", np.polyfit(v - v.mean(), truth - truth.mean(), 1)[0].round(3))
print("max |prod - pace_models ridge|", np.abs(prod - pmr).max().round(6), "; ratio pace_eff/prod sd", (eff.std() / prod.std()).round(3))
# pace_eff at lambda 90 reproduces prod?
pe.PACE_LAMBDA = {"gamer": 90.0, "team": 360.0, "fixed": 2e-4}
_, b2 = pe.pace_fit(df, pe.weights(df.start, cut).values / pe.SCALAR)
print("pace_eff at 90/360 vs prod: max diff", np.abs(b2[1:1 + G] - prod).max().round(6))
