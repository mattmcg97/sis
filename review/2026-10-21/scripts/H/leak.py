"""features.build: change one result and check no feature of a match kicking off before that result is in moves."""
import copy, datetime as dt, math, random
import numpy as np
from xgb import features as fx

rng = random.Random(1)
t0 = dt.datetime(2026, 9, 1)
ms = []
gam = [f"G{i}" for i in range(8)]
for i in range(600):
    a, b = rng.sample(gam, 2)
    t = t0 + dt.timedelta(minutes=13 * i)
    ms.append(dict(code=f"M{i:04d}", t=t, stream=str(1 + i % 3), p1=a, p2=b, t1=f"T{rng.randrange(4)}", t2=f"T{rng.randrange(4)}",
                   s1=float(rng.randrange(0, 50)), s2=float(rng.randrange(0, 50))))
drv = {m["code"]: {"h": np.arange(len(fx.DRIVE_KEYS), dtype=float) + rng.random(), "a": np.arange(len(fx.DRIVE_KEYS), dtype=float),
                   "prod": {}} for m in ms}
base = fx.build(ms, None, drv)
k = 300
ms2 = copy.deepcopy(ms); ms2[k]["s1"] += 30
drv2 = copy.deepcopy(drv); drv2[ms[k]["code"]]["h"] = drv2[ms[k]["code"]]["h"] * 3
alt = fx.build(ms2, None, drv2)
in_at = ms[k]["t"] + fx.RESULT_LAG
first_changed = None
for r1, r2 in zip(base, alt):
    diff = [c for c in r1 if c not in ("s1",) and not (isinstance(r1[c], float) and math.isnan(r1[c]) and math.isnan(r2[c])) and r1[c] != r2[c]]
    if diff and r1["code"] != ms[k]["code"]:
        first_changed = (r1["start"], diff[:5]); break
print("result in at", in_at, "| first other match whose features change:", first_changed)
