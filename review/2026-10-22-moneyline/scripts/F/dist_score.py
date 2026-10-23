"""Score dist_<tag>.pkl: per window, each distribution's integer log score and RPS (vs a reference, day-resampled
interval), and P(over prod's line) against prod's own P(over) (harness.report with y := over).
  python dist_score.py TAG [--ref normal_const] [--windows TUNE,TEST,SEP,LATE] [--only a,b]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("--ref", default="normal_const")
ap.add_argument("--windows", default="TUNE,TEST,SEP,LATE")
ap.add_argument("--only", default="")
ap.add_argument("--no-ou", action="store_true")
a = ap.parse_args()

df = F.panel()
r = pd.read_pickle(os.path.join(F.OUT, f"dist_{a.tag}.pkl"))
names = sorted({c.split("__")[0] for c in r.columns if "__" in c})
if a.only:
    names = [n for n in names if n in a.only.split(",")]
d = df[["code", "day", "total", "over", "prod_total_prob", "prod_total_line"]].merge(r, on="code")
lines = []
for wn in a.windows.split(","):
    w = F.WINDOWS[wn]
    x = d[(d.day >= w[0]) & (d.day < w[1]) & d.total.notna()]
    x = x.dropna(subset=[f"{n}__logs" for n in names])
    lines.append(f"\n== {wn} {w[0]}..{w[1]}: {len(x):,} matches (every distribution), {x.day.nunique()} days")
    lines.append(f"  {'dist':24s}{'logS':>8s}{'RPS':>8s}{'sd':>7s}   logS vs {a.ref} (95%)        RPS vs {a.ref} (95%)")
    for n in names:
        s = f"  {n:24s}{x[n + '__logs'].mean():8.4f}{x[n + '__rps'].mean():8.4f}{x[n + '__sd'].mean():7.2f}"
        if n != a.ref and a.ref in names:
            for k in ("logs", "rps"):
                mu, lo, hi = F.boot_days(x.day.values, (x[f"{n}__{k}"] - x[f"{a.ref}__{k}"]).values)
                s += f"   {mu:+.4f} ({lo:+.4f},{hi:+.4f})"
        lines.append(s)
    if not a.no_ou:
        o = F.ou_frame(x)
        cols = [f"{n}__p_over" for n in names]
        o = o.rename(columns={c: c.replace("__p_over", "") for c in cols})
        lines.append("  O/U at prod's line (y = over; no pushes: every line is x.5):")
        lines.append(F.H.report(o, names, window=w, refs=("prod_over",) + ((a.ref,) if a.ref in names else ())))
txt = "\n".join(lines)
print(txt)
open(os.path.join(F.OUT, f"dist_{a.tag}_score.txt"), "w").write(txt + "\n")
