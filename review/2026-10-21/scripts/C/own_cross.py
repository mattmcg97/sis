# The nudge's unit: calibrate_nudge moves both sides alike; react's updates move one side, or the two
# opposite ways. Own vs cross effects at kick-off on the tests' build, and the response to an opposite pair.
import os, sys
import numpy as np
sys.path.insert(0, "/home/user/sis")
from eAMFModel import v21
OUT = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/C/model"
m = v21.Model.load(os.path.join(OUT, v21.MODEL_FILE))
start = v21.kickoff_start(m)
def means(n):
    h, a = v21.simulate(m, dict(start, nudge=np.tile(n, (2, 1))), 4000, seed=3)
    return np.array([h.mean(), a.mean()])
print("build's nudge_slope (both sides alike):", round(m.nudge_slope, 3))
for amp in (1.0, 2.0):
    both = (means([amp, amp]) - means([-amp, -amp])) / (2 * amp)
    home = (means([amp, 0]) - means([-amp, 0])) / (2 * amp)
    opp = (means([amp, -amp]) - means([-amp, amp])) / (2 * amp)
    print(f"+-{amp}: both alike {both.round(3)} | home only: own {home[0]:.3f} cross {home[1]:.3f}"
          f" | opposite (+home, -away): home {opp[0]:.3f} away {opp[1]:.3f}")
