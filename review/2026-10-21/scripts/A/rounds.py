"""fit_exp_tilts' block Newton (TILT_ROUNDS=4) where a gamer mostly plays one team: converged?"""
import numpy as np
from eAMFModel import v21

rng = np.random.default_rng(1)
B = 12
phi_b = np.linspace(-0.5, 3.0, B)[:, None]
S, G = 60000, 10
gam = rng.integers(0, G, S)
team = np.where(rng.random(S) < 0.9, gam, rng.integers(0, G, S))     # 90% of a gamer's snaps on his own team
true_g = np.linspace(-0.4, 0.4, G)
H = np.full((S, B), 1.0 / B)
th = true_g[gam][:, None]
p = H * np.exp(th @ phi_b.T); p /= p.sum(1, keepdims=True)
c = p.cumsum(1); obs = (rng.random(S)[:, None] > c).sum(1)
who = dict(gamer_off=np.r_[gam, 0][:-1], team_off=team)
sizes = dict(gamer_off=G + 1, team_off=G + 1)
for rounds in (4, 50):
    t, _ = v21.fit_exp_tilts(H, obs, phi_b, who, sizes, prior=200.0, rounds=rounds)
    print(f"rounds {rounds:2d}: gamer {np.round(t['gamer_off'][:G, 0], 3)}\n          team  {np.round(t['team_off'][:G, 0], 3)}")
    # a gamer on another team: the theta the sim would use
    print(f"          gamer 9 on team 0: {t['gamer_off'][9,0] + t['team_off'][0,0]:+.3f} (true {true_g[9]:+.3f})")
