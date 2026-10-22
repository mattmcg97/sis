"""fit_exp_tilts: does an entity with `prior` snaps move half way (as documented)?"""
import numpy as np
from eAMFModel import v21

rng = np.random.default_rng(0)
# time-like: 2-second bins of seconds/10; half the states stopped (4-10s), half running (20-40s)
secs = np.arange(4, 41, 2)
phi_b = (secs / 10.0)[:, None]
B = len(secs)
stopped = np.where(secs <= 10, 1.0, 0.0); stopped /= stopped.sum()
running = np.where(secs >= 20, 1.0, 0.0); running /= running.sum()
for n_e in (200, 1000, 5000):
    S = 20000
    state = rng.random(S) < 0.5
    H = np.where(state[:, None], stopped[None], running[None])
    ent = np.full(S, 1)                      # the unknown (last)
    ent[:n_e] = 0                            # the entity's snaps
    true = -0.5                              # quicker snaps
    th = np.where(ent == 0, true, 0.0)[:, None]
    p = H * np.exp(th @ phi_b.T); p /= p.sum(1, keepdims=True)
    obs = np.array([rng.choice(B, p=pi) for pi in p])
    who = dict(gamer_off=ent)
    sizes = dict(gamer_off=2)
    # the unpenalized MLE (prior ~0) and the penalized fit at prior 200
    mle, _ = v21.fit_exp_tilts(H, obs, phi_b, who, sizes, prior=1e-9, rounds=8)
    pen, _ = v21.fit_exp_tilts(H, obs, phi_b, who, sizes, prior=200.0, rounds=8)
    lam = 200 * phi_b[obs].var(0)[0]
    info = (np.array([((phi_b[:, 0] ** 2) * h).sum() - ((phi_b[:, 0] * h).sum()) ** 2 for h in H[:n_e]])).mean()
    print(f"entity snaps {n_e:5d}: MLE {mle['gamer_off'][0,0]:+.3f}, prior-200 fit {pen['gamer_off'][0,0]:+.3f}"
          f" -> share of the way {pen['gamer_off'][0,0]/mle['gamer_off'][0,0]:.2f}"
          f" (documented n/(n+200) = {n_e/(n_e+200):.2f}); penalty/snap {lam/200:.3f} vs info/snap {info:.3f}")
