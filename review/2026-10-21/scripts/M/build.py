import os, time, csv
from eAMFModel import v21
from eAMFModel.tests.test_v21 import HISTORY, built
W = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/M/work"
t = time.time()
m = built("tilt")
m.save(os.path.join(W, v21.MODEL_FILE))
print("built", time.time() - t, "nudge_dir", m.nudge_dir is not None, "can_pace", v21.can_pace(m))
table = os.path.join(W, "avg_form4_w05.csv")
h = HISTORY[:3]
with open(table, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["MATCH_CODE", "start", "home", "home_team", "away", "away_team", "home_pts", "away_pts", "exp_drives", "league_drives"])
    for k, (r, pts, drives) in enumerate(zip(h, ((26.0, 13.0), (17.0, 17.0), (14.0, 21.0)), (11.5, 10.0, 8.5))):
        w.writerow([r["MATCH_CODE"], f"2026-10-1{k} 12:00:00", r["PLAYER_1_HANDLE"], r["PLAYER_1_TEAM"], r["PLAYER_2_HANDLE"], r["PLAYER_2_TEAM"], *pts, drives, 10.0])
print(open(table).read())
