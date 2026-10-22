"""Recompute the chosen bases over the whole span (from the first fittable day), so the offset models'
training rows have walk-forward bases too.   python s01b_bases_full.py <link_days> <ridge_hl> <ridge_lam>"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

ld, hl, lam = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
ld = None if ld == "None" else int(ld)
df = load()
p = wf_link(df, "rating_margin", "2025-12-01", RUN_TO, train_days=ld)
save_series(df, p, "link_rm")
m = wf_ridge_strength(df, "2025-12-01", RUN_TO, half_life=hl, lam=lam, train_days=None)
save_series(df, m, "ridge_margin")
pr = wf_on_oos(df, m, "2025-12-01", RUN_TO, kind="link", hist_days=60)
save_series(df, pr, "ridge_p")
df["link_rm"], df["ridge_p"] = p, pr
print("coverage", df.link_rm.notna().mean(), df.ridge_p.notna().mean(), df.loc[df.ridge_p.notna(), "day"].min())
for w in (H.TUNE,):
    print(H.report(df, ["link_rm", "ridge_p"], window=w, refs=("prod", "link_rm")))
