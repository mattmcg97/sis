"""(a) The walkthrough / form tables' own P(over prod's line), off each table's own score model (eAMFCalibrator's
prior_bets.Table: NB2's shared-gamma joint for nb2 and the averages, the normal (sd 12) where a table has none),
imported read-only, market 54 (the over). SEP only (the tables start 09-01). Writes out/F/tables_ou.pkl.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402
sys.path.insert(0, F.ROOT)
from eAMFCalibrator.prior_bets import Table  # noqa: E402
from eAMFCalibrator.prior_replay import arm_chance  # noqa: E402

df = F.panel()
d = df[(df.day >= "2026-09-01") & df.prod_total_line.notna()][["code", "prod_total_line"]]
out = pd.DataFrame({"code": d.code.values})
for name, rel in F.H.TABLES.items():
    t = Table(os.path.join(F.ROOT, rel))
    ps = []
    for c, line in zip(d.code.values, d.prod_total_line.values):
        try:
            p = arm_chance(t.book(c), 54, float(line)) if c in t.rows else None
        except Exception:
            p = None
        ps.append(np.nan if p is None else p)
    out[f"tbl_{name}_pover"] = ps
    kinds = pd.Series([t.rows[c].get("score_model") for c in d.code.values if c in t.rows]).value_counts().to_dict()
    print(name, "priced", int(np.isfinite(ps).sum()), "score models", kinds, flush=True)
out.to_pickle(os.path.join(F.OUT, "tables_ou.pkl"))
