# v21_endgame._sim_ties counts a level game's last score "in Q4's last 3:00" off its clock alone (last_score
# keeps no period): how many of those came before Q4 (no Q4 points on the path)?
import os, sys
import numpy as np
sys.path.insert(0, "/home/user/sis")
from eAMFModel import v21, v21_endgame as ve
OUT = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/C/model"
m = v21.Model.load(os.path.join(OUT, v21.MODEL_FILE))
st = {}
v21.simulate(m, v21.kickoff_start(m), 6000, seed=2, stats=st)
ot, ls, q = st["went_ot"], st["last_score"], st["q_points"]
late = ot & (ls[:, 0] >= 0) & (ls[:, 1] <= ve.CONV_LEFT)
before_q4 = late & (q[:, 3] == 0)
print("games to OT", int(ot.sum()), "| counted 'in Q4's last 3:00'", int(late.sum()),
      "| of them with no Q4 points (the levelling score came earlier)", int(before_q4.sum()))
