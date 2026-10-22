import os, sys
import numpy as np
sys.path.insert(0, "/home/user/sis")
from eAMFModel import v21, v21_stream as vs, v21_react as vr
from eAMFModel.tests.fakes import _matches
OUT = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/C/model"
m = v21.Model.load(os.path.join(OUT, v21.MODEL_FILE))
print("nudge_slope", m.nudge_slope)
M = _matches(n=6, seed=11)
for code in list(M)[:3]:
    rows = [vs._as_text(r) for r in M[code]]
    st = vs.state_starts(m, rows)
    ks = [(msg, s) for msg, s in st if int(s["phase"][0]) == v21.KICK and int(s["period"][0]) <= 4]
    for msg, s in ks[::max(1, len(ks)//4)]:
        left = 1 - vr.elapsed_share(m, int(s["period"][0]), float(s["clock"][0]))
        out = []
        for a in (np.array([-1.0, 0.0]), np.array([1.0, 0.0])):
            S = dict(s, nudge=a[None, :])
            h, w = v21.simulate(m, S, 3000, seed=5)
            out.append(h.mean())
        print(code, "Q", int(s["period"][0]), "clock", float(s["clock"][0]), "left %.2f" % left,
              "dpts/unit %.2f" % ((out[1]-out[0])/2), "ratio to left %.2f" % ((out[1]-out[0])/2/left if left>0 else np.nan))
