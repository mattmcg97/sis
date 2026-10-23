"""Run the TUNE-frozen ridges walk-forward over the whole span and save their outputs (out/F/means_ridge.pkl):
  b_tot                 total ridge (gamer + team "total" effects), from 2026-01-15
  b_side(_h,_a,_margin) per-side points ridge (off/def gamer, off/def team, home), from 2026-01-15
  drives                expected drives (the pace ridge), from 2026-05-01
  ppd_h, ppd_a          points a drive (drive-weighted ridge, opponent-adjusted), from 2026-05-01
  td_h/a, fg_h/a        touchdowns / field goals a drive, same design and settings as ppd
  c_prod                drives / 2 x (ppd_h + ppd_a)
  drives_pm             drives at pace_models' own settings (hl 60, 180 d, lam 90/360) for reference
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402

df = F.panel()
ix = F.Index(df)
n = len(df)
tuned = json.load(open(os.path.join(F.OUT, "tuned.json")))
END = "2026-10-19"
EARLY, DRV = "2026-01-15", "2026-05-01"


def s_of(m):
    s = dict(tuned[m]["setting"])
    s["hl"] = float(s["hl"])
    return s


out = pd.DataFrame({"code": df.code})
# b_tot
s = s_of("b_tot")
X, pk = F.match_design(df, ix, stream=s["stream"])
out["b_tot"] = F.walk_ridge(df, X, df.total.values.astype(float), F.penalty(pk, s["lam_g"], s["lam_t"]), EARLY, END,
                            hl=s["hl"], train_days=s["td"], level_hl=s["lvl"])
print("b_tot done", flush=True)
# b_side
s = s_of("b_side")
Xs, pks = F.side_design(df, ix, stream=s["stream"])
y = np.concatenate([df.s1.values, df.s2.values]).astype(float)
p = F.walk_ridge(df, Xs, y, F.penalty(pks, s["lam_g"], s["lam_t"]), EARLY, END, hl=s["hl"], train_days=s["td"],
                 level_hl=s["lvl"], nrep=2)
out["b_side_h"], out["b_side_a"] = p[:n], p[n:]
out["b_side"] = p[:n] + p[n:]
out["b_side_margin"] = p[:n] - p[n:]
print("b_side done", flush=True)
# drives
s = s_of("drives")
X, pk = F.match_design(df, ix, stream=s["stream"])
out["drives"] = F.walk_ridge(df, X, df.D.values.astype(float), F.penalty(pk, s["lam_g"], s["lam_t"]), DRV, END,
                             hl=s["hl"], train_days=s["td"], level_hl=s["lvl"])
X0, pk0 = F.match_design(df, ix, stream=False)
out["drives_pm"] = F.walk_ridge(df, X0, df.D.values.astype(float), F.penalty(pk0, 90, 360), DRV, END, hl=60,
                                train_days=180)
print("drives done", flush=True)
# ppd, td, fg a drive
s = s_of("ppd")
Xs, pks = F.side_design(df, ix, stream=s["stream"])
dr = np.concatenate([df.h_drives.values, df.a_drives.values]).astype(float)
ew = np.where(dr > 0, dr, np.nan)
for name, num in (("ppd", np.concatenate([df.s1.values, df.s2.values])),
                  ("td", np.concatenate([df.h_tdx.values, df.a_tdx.values])),
                  ("fg", np.concatenate([df.h_fgx.values, df.a_fgx.values]))):
    with np.errstate(invalid="ignore", divide="ignore"):
        yy = np.where(dr > 0, num.astype(float) / dr, np.nan)
    p = F.walk_ridge(df, Xs, yy, F.penalty(pks, s["lam_g"], s["lam_t"]), DRV, END, hl=s["hl"], train_days=s["td"],
                     level_hl=s["lvl"], nrep=2, extra_w=ew)
    out[f"{name}_h"], out[f"{name}_a"] = p[:n], p[n:]
    print(name, "done", flush=True)
out["c_prod"] = out.drives / 2.0 * (out.ppd_h + out.ppd_a)
out.to_pickle(os.path.join(F.OUT, "means_ridge.pkl"))
d = df[["code", "day", "total", "D"]].merge(out, on="code")
for w in ("TUNE", "TEST", "SEP"):
    print(F.total_table(d, ["b_tot", "b_side", "c_prod"], F.WINDOWS[w], ref="b_tot"))
    print(F.total_table(d, ["drives", "drives_pm"], F.WINDOWS[w], ref="drives_pm", target="D"))
