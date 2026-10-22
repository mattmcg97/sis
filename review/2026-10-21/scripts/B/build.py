import sys, time, pickle, copy
sys.path.insert(0, "/home/user/sis")
import numpy as np
from eAMFModel import v21
from eAMFModel.tests.fakes import _matches
from eAMFModel.tests.test_v21 import _history
SC = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/B"
M = _matches(n=80, seed=3)
H = _history(M)
t = time.time()
m = v21.build(M, H, mode="tilt", min_records=20, verbose=False, refine=False, key="lean", blend=40, field=0.2,
              decision_blend=25, clock=0.2)
print("tilt built", time.time() - t)
m.save(SC + "/tilt.pkl")
t = time.time()
f = v21.build(M, H, mode="full", min_records=20, verbose=False, refine=False)
print("full built", time.time() - t)
f.save(SC + "/full.pkl")
pickle.dump((M, H), open(SC + "/data.pkl", "wb"))
