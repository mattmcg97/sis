"""Write preds_total/F_<model>.csv: code, exp_total, sd_total, p_over (P(total > prod's line); every prod line is x.5,
so no pushes), exp_drives, and the mean model / distribution they come from.
  python write_preds.py MODELNAME DIST_TAG DIST_NAME MEANCOL
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402

name, tag, dist, meancol = sys.argv[1:5]
r = pd.read_pickle(os.path.join(F.OUT, f"dist_{tag}.pkl"))
m = pd.read_pickle(os.path.join(F.OUT, "means_all.pkl"))
df = F.panel()[["code", "day"]]
out = df.merge(r[["code", f"{dist}__mean", f"{dist}__sd", f"{dist}__p_over"]], on="code").merge(
    m[["code", meancol, "drives"]], on="code", how="left")
out = out[(out.day >= "2026-06-01") & (out.day < "2026-10-19") & out[f"{dist}__mean"].notna()]
res = pd.DataFrame({"code": out.code, "exp_total": out[meancol].round(4), "dist_mean": out[f"{dist}__mean"].round(4),
                    "sd_total": out[f"{dist}__sd"].round(4), "p_over": out[f"{dist}__p_over"].round(5),
                    "exp_drives": out.drives.round(4), "mean_model": meancol, "dist": dist})
os.makedirs(os.path.join(F.ML, "preds_total"), exist_ok=True)
path = os.path.join(F.ML, "preds_total", f"F_{name}.csv")
res.to_csv(path, index=False)
print(path, len(res), res.p_over.notna().sum(), "with a prod line")
