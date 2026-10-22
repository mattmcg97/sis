"""What the margin adds inside each results-only family: MOV version less its win/loss twin (same structure)."""
import common as C
import glob
import pandas as pd
H = C.H
d = C.slim()
bt = pd.read_pickle(f"{C.OUT}/bt_preds.pkl")
dy = pd.concat([pd.read_pickle(f).set_index("code") for f in glob.glob(f"{C.OUT}/dyn_preds_*.pkl")], axis=1)
dy = dy.loc[:, ~dy.columns.duplicated()].reset_index()
d = d.merge(bt, on="code").merge(dy, on="code")
pairs = [("bt_mov", "bt_team", "static BT (+team, stream home), MOV weights"),
         ("mov_elo", "elo_team", "Elo (+team, new-gamer K), MOV multiplier"),
         ("mov_elo_ac", "elo_team", "Elo, MOV multiplier + autocorrelation damping"),
         ("mov_kal_team", "kal_team", "Kalman (+team), MOV-scaled information"),
         ("mov_kal3", "kal3", "Kalman kal3, MOV-scaled information")]
lines = ["MOV version less win/loss twin, log loss (day-resampled 95%); negative = margin helps"]
for win, nm in [(H.TUNE, "TUNE"), (H.TEST, "TEST"), (H.SEP, "SEP"), (C.LATE, "LATE")]:
    w = d[H.in_window(d, win) & d.y.notna()]
    lines.append(f"{nm}:")
    for a, b, lab in pairs:
        mu, lo, hi = H.boot_diff(w, a, b)
        lines.append(f"  {lab:52s} {mu:+.4f} ({lo:+.4f},{hi:+.4f})")
txt = "\n".join(lines)
open(f"{C.OUT}/mov_pairs.txt", "w").write(txt)
print(txt)
