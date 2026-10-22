"""Fakes never go for it on 4th down (always punt or kick a field goal): what does drives.py count?"""
from collections import Counter
from eAMFModel.tests.fakes import _matches
from xgb import drives as dv

tot = Counter()
for code, rows in _matches(n=40, seed=3).items():
    rows = [dict(r, score_p1_at_start=r["score_p1"], score_p2_at_start=r["score_p2"], final_p1="", final_p2="",
                 home_handle="H", away_handle="A") for r in rows]
    st = dv.match_stats(code, rows)
    for k in ("fourth_n", "fourth_conv", "snaps", "big20", "big40", "punt", "fg"):
        tot[k] += st[f"h_{k}"] + st[f"a_{k}"]
    tot["punt rows"] += sum(r["play_kind"] == "PUNT" for r in rows)
    tot["fg rows"] += sum(r["play_kind"] == "FIELD_GOAL" for r in rows)
print(dict(tot))
# real drive_stats: how many 4th-down 'attempts' against punts + field goals a side
import pandas as pd
d = pd.read_csv("xgb/data/drive_stats.csv")
for s in "ha":
    print(s, "fourth_n", d[f"{s}_fourth_n"].mean().round(3), "punts+fg+fg_miss", (d[f"{s}_punt"] + d[f"{s}_fg"] + d[f"{s}_fg_miss"]).mean().round(3),
          "downs", d[f"{s}_downs"].mean().round(3))
