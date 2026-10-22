import numpy as np
from eAMFModel import v21
from eAMFModel.tests.test_v21 import _state
SC = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/B"
m = v21.Model.load(SC + "/tilt.pkl")
# OT: home kicked a FG on the first possession (17-14), away then scored a TD (17-20, six on the board): the try
s = _state(m, period=5, clock=150.0, team=0, home=17, away=20)
s["phase"] = np.array([v21.KICK]); s["convert"] = np.array([1])
st = {}
h, a = v21.simulate(m, s, 2000, seed=2, stats=st)
h, a = h[0], a[0]
print("away TD to go ahead in OT's second possession: ot_walk_off", st.get("ot_walk_off"),
      "| away kicked the try in", round(((a - 20) == 1).mean() + ((a - 20) == 0).mean(), 3),
      "| home won", round((h > a).mean(), 3), "| home scored again", round((h > 17).mean(), 3))
