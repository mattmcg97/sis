"""On a fake tilt build: the tilts' penalty a snap (prior x the variance of phi over all real snaps) against
the information a snap carries (phi's variance within its own state's draw), distance and time."""
import time
import numpy as np
from eAMFModel import v21
from eAMFModel.tests.fakes import _matches
from eAMFModel.tests.test_v21 import _history

orig = v21.fit_exp_tilts
def spy(H, obs, phi_b, who, sizes, prior=v21.TILT_PRIOR, rounds=v21.TILT_ROUNDS, weights=None):
    pen = phi_b[obs].var(0)
    mu = H @ phi_b
    var_in = (H @ (phi_b ** 2)) - mu ** 2
    print("  phi var over all snaps (penalty a snap / prior):", np.round(pen, 4))
    print("  mean within-state var (information a snap)     :", np.round(var_in.mean(0), 4))
    print("  effective prior in snaps                        :", np.round(prior * pen / np.maximum(var_in.mean(0), 1e-9), 0))
    return orig(H, obs, phi_b, who, sizes, prior=prior, rounds=rounds, weights=weights)
v21.fit_exp_tilts = spy
M = _matches(n=80, seed=3)
t0 = time.time()
v21.fit_model_tilts.__globals__["fit_exp_tilts"] = spy
m = v21.build(M, _history(M), min_records=20, verbose=False, refine=False, mode="tilt", key="lean", blend=40.0, field=0.2)
print(f"built in {time.time() - t0:.0f}s")
