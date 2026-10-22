import sys, pickle
import numpy as np
from eAMFModel import v21, feed
SC = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/B"
m = v21.Model.load(SC + "/tilt.pkl")
M, H = pickle.load(open(SC + "/data.pkl", "rb"))
code = sorted(M)[0]
rows = [dict(r) for r in M[code]]
# flip TEAM_A to away
for r in rows:
    r["team_a_side"] = "away"
starts = v21.match_starts(m, rows)
bad = 0
for i, s in starts[:400]:
    r = rows[i]
    a_home = False
    team = s["team"][0]
    exp_team = 0 if (r["offense"] == "TEAM_A") == a_home else 1
    if team != exp_team or s["home"][0] != int(r["score_p1"]) or s["away"][0] != int(r["score_p2"]) \
            or s["y"][0] != int(r["field_position"]) or s["down"][0] != int(r["down"]) or s["clock"][0] != float(r["clock_seconds"]):
        bad += 1
print("starts", len(starts), "mismatch", bad)
print("kick2 of starts (opening TEAM_A, A away => opening receiver is side 1 => kick2 0):", set(int(s["kick2"][0]) for _, s in starts))
# distance 0
r0 = dict(rows[5]); print(rows[5]["play_kind"], rows[5]["down"], rows[5]["distance"])
