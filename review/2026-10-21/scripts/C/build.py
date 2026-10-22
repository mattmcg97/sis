import contextlib, io, os, sys, time, pickle
import numpy as np
sys.path.insert(0, "/home/user/sis")
from eAMFModel import v21
from eAMFModel.tests.fakes import _matches
from eAMFModel.tests.test_v21 import _history
OUT = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/C/model"
os.makedirs(OUT, exist_ok=True)
M = _matches(n=80, seed=3)
H = _history(M)
t = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    m = v21.build(M, H, min_records=20, verbose=False, refine=False, mode="tilt",
                  key="lean", blend=40.0, field=0.2, decision_blend=25.0, clock=0.2)
m.save(os.path.join(OUT, v21.MODEL_FILE))
print("built", time.time() - t, "s; nudge_slope", m.nudge_slope, "dir", m.nudge_dir)
