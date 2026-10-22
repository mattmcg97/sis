import os, sys, io, contextlib, time
sys.path.insert(0, "/home/user/sis")
from eAMFModel import v21
from eAMFModel.tests.test_v21 import MATCHES, HISTORY
D = os.path.dirname(os.path.abspath(__file__))
t = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, mode="tilt",
                  key="lean", blend=40.0, field=0.2, decision_blend=25.0, clock=0.2)
os.makedirs(os.path.join(D, "model"), exist_ok=True)
m.save(os.path.join(D, "model", v21.MODEL_FILE))
print("built", time.time() - t)
