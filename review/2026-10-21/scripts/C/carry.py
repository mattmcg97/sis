# Does the carried snap evidence (match_evidence: alpha=None) differ from the in-match evidence's baseline
# (score_match: the kick-off nudge a0)?
import os, sys
import numpy as np
sys.path.insert(0, "/home/user/sis")
from eAMFModel import v21, v21_stream as vs, v21_react as vr, feed
from eAMFModel.tests.fakes import _matches
OUT = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/C/model"
m = v21.Model.load(os.path.join(OUT, v21.MODEL_FILE))
M = _matches(n=4, seed=21)
for code, rows in M.items():
    rows = [vs._as_text(r) for r in rows]
    last = [max(int(r["message"]) for r in rows)]
    start = v21.kickoff_start(m)
    h, a = v21.simulate(m, start, 300, seed=1)
    target = (h.mean() + 4.0, a.mean() - 4.0)          # prod rates home 4 points better than the build
    (_, _), a0 = v21.nudge_solve(m, start, target, 300, seed=1, share=0.75)
    U0, I0 = vr.snap_sums(m, rows, last, None)
    U1, I1 = vr.snap_sums(m, rows, last, a0)
    print(code, "kick-off nudge", np.round(a0, 2), "| carried U (alpha None)", np.round(U0[0], 2),
          "| U under the nudge", np.round(U1[0], 2), "| I", np.round(I1[0], 2),
          "| one-match step diff", np.round((U0[0] - U1[0]) / (I1[0] + 1.0), 2))
