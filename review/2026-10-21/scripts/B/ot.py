import sys, pickle
import numpy as np
from eAMFModel import v21
SC = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/B"
for name in ("tilt", "full"):
    m = v21.Model.load(SC + f"/{name}.pkl")
    G = len(m.gamers); T = len(m.teams)
    s = dict(period=np.array([5]), clock=np.array([200.0]), team=np.array([1]), phase=np.array([v21.SNAP]),
             down=np.array([1]), dist=np.array([10]), y=np.array([25]), home=np.array([14]), away=np.array([14]),
             prev=np.array([v21.FRESH]), elapsed=np.array([60]), tos=np.array([[2, 2]]), kick2=np.array([1]),
             gamer=np.array([[G, G]]), team_ix=np.array([[T, T]]))
    st = {}
    h, a = v21.simulate(m, s, 4000, seed=1, stats=st)
    h, a = h[0], a[0]
    both = (h > 14) & (a > 14)
    away_first_fg_home_wins = (a == 17) & (h > 17)
    print(name, "P(away wins)", round((a > h).mean(), 3), "P(home wins)", round((h > a).mean(), 3),
          "level", round((h == a).mean(), 3),
          "| both scored", round(both.mean(), 3), "| away 17 & home >17:", round(away_first_fg_home_wins.mean(), 3),
          "ot_walk_off", st.get("ot_walk_off"), "ot_decided", st.get("ot_decided"))
