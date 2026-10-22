import numpy as np, datetime as dt, math
from eAMFCalibrator import form_prior as fp
m = fp.read_matches("AMFELO_20261019.csv")
s0 = dt.datetime(2026,9,1)
sub = [x for x in m if x[6] is not None and x[0] >= s0]
y = np.array([x[6]-x[7] for x in sub], float)
views = {}
for drift in (0.01, 0.1, 0.25, 0.5, 1, 2, 4):
    v = fp.rating(m, drift); views[drift] = v
    pr = np.array([v[x[1]][0] for x in sub])
    b = np.polyfit(pr, y, 1)
    mse = np.mean((y-pr)**2)
    print(f"drift {drift:5}: slope {b[0]:.3f} rmse {math.sqrt(mse):.3f}  sd(pred) {pr.std():.2f}")
# blend: proxy table = best-calibrated slow rating
for base in (0.1, 0.25):
    T = np.array([views[base][x[1]][0] for x in sub]); R = np.array([views[4][x[1]][0] for x in sub])
    for w in (0.0, 0.5, 1.0):
        B = T + w*(R-T)
        b = np.polyfit(B, y, 1)
        print(f"proxy table drift {base}, w {w}: slope {b[0]:.3f} rmse {math.sqrt(np.mean((y-B)**2)):.3f}")
from statistics import NormalDist
N = NormalDist()
dec = y != 0
def ll(mm, sd=9.35):
    p = np.array([N.cdf(v / sd) for v in mm]); p = np.clip(p, 1e-6, 1 - 1e-6)
    yy = (y > 0).astype(float)
    return float(np.mean(-(yy * np.log(p) + (1 - yy) * np.log(1 - p))[dec]))
T = np.array([views[0.1][x[1]][0] for x in sub]); R = np.array([views[4][x[1]][0] for x in sub])
for w in (0.0, 0.25, 0.5, 1.0):
    print(f"proxy drift-0.1 table moved {w} toward drift-4 rating: ML log loss {ll(T + w*(R-T)):.4f}")
# by month, slope of drift-4 rating
for mon in ("2026-03", "2026-05", "2026-07", "2026-09", "2026-10"):
    s2 = [x for x in m if x[6] is not None and x[0].strftime("%Y-%m") == mon]
    pr = np.array([views[4][x[1]][0] for x in s2]); yy = np.array([x[6]-x[7] for x in s2], float)
    print(mon, "drift-4 slope", round(np.polyfit(pr, yy, 1)[0], 3))
