import numpy as np, datetime as dt
from eAMFCalibrator import form_prior as fp
m = fp.read_matches("AMFELO_20261019.csv")
print(len(m), m[0][0], m[-1][0])
s = [x for x in m if x[6] is not None]
marg = np.array([x[6]-x[7] for x in s], float)
print("mean home margin", marg.mean(), "se", marg.std()/np.sqrt(len(marg)), "sd", marg.std())
for mon in sorted({x[0].strftime("%Y-%m") for x in s}):
    mm = np.array([x[6]-x[7] for x in s if x[0].strftime("%Y-%m")==mon], float)
    print(mon, len(mm), round(mm.mean(),2), round(mm.std()/np.sqrt(len(mm)),2))
for drift in (4.0,):
    v = fp.rating(m, drift)
    sub = [x for x in s if x[0] >= dt.datetime(2026,9,1)]
    pr = np.array([v[x[1]][0] for x in sub]); y = np.array([x[6]-x[7] for x in sub], float)
    X = np.c_[np.ones(len(pr)), pr]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    print("drift", drift, "Sep+ regression y = a + b*rating:", b, "mean rating", pr.mean(), "mean y", y.mean())
