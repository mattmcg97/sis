import numpy as np, pandas as pd
t = pd.read_csv("xgb/out/turnovers/turnovers_matches.csv")
d = pd.read_csv("xgb/data/drive_stats.csv")
m = t.merge(d, left_on="MATCH_CODE", right_on="match_code", suffixes=("_sim", "_dr"))
print(len(m), "matches in both")
for s in "ha":
    print(s, "snaps: sim", m[f"{s}_snaps_sim"].mean().round(2), " drives.py", m[f"{s}_snaps_dr"].mean().round(2),
          "| drives: sim", m[f"{s}_drives_sim"].mean().round(2), " drives.py", m[f"{s}_drives_dr"].mean().round(2),
          "| drives.py drive_plays", m[f"{s}_drive_plays"].mean().round(2))
    diff = m[f"{s}_snaps_sim"] - m[f"{s}_snaps_dr"]
    print("   snap diff per match mean", diff.mean().round(2), "corr with sim drives", np.corrcoef(diff, m[f"{s}_drives_sim"])[0,1].round(3),
          "with drives.py drives", np.corrcoef(diff, m[f"{s}_drives_dr"])[0,1].round(3))
    b = np.polyfit(m[f"{s}_drives_dr"], diff, 1); print("   diff ~ drives.py drives slope", b.round(3))
print("--- gap against the kicks a side receives (opponent's tds + fgs + punts)")
for s, o in (("h","a"),("a","h")):
    gap = m[f"{s}_snaps_sim"] - m[f"{s}_snaps_dr"]
    kicks = m[f"{o}_tds"] + m[f"{o}_fg"] + m[f"{o}_punts"]
    X = np.column_stack([np.ones(len(m)), kicks])
    b, *_ = np.linalg.lstsq(X, gap, rcond=None)
    print(s, "gap = %.2f + %.2f x kicks received; mean kicks %.2f, mean gap %.2f" % (b[0], b[1], kicks.mean(), gap.mean()))
