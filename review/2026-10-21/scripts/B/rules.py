import copy
import numpy as np
from eAMFModel import v21, game as gm
from eAMFModel.tests.test_v21 import _state, _always, _fixed_time, _fixed_play
SC = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/B"
m0 = v21.Model.load(SC + "/tilt.pkl")
n_who = len(m0.gamers) + 1

# 1. missed field goal: where the other side takes over
m = copy.deepcopy(m0)
m.fg_make = np.zeros_like(m.fg_make)
m.choices["fourth"] = _always("fourth", v21.FG, n_who)
m.use_rules = False
st = {}
v21.simulate(m, _state(m, period=1, clock=200.0, down=4, dist=8, y=70), 20, stats=st, max_steps=2)
log = v21.snap_log(st)
first = np.flatnonzero(log["path"] == 0)
print("sim: missed FG from y=70 -> next snap side", log["side"][first][0], "field", log["field"][first][0])
g = gm.apply(gm.Game(phase=gm.SCRIMMAGE, side=gm.HOME, field=70, down=4, distance=8, fresh=False),
             {"type": "field_goal", "made": False})[0]
print("game.py: missed FG from field 70 ->", g.side, g.field)

# 2. auto_end on a stopped clock: 4th down, 35s left, clock stopped (incompletion), defence no timeouts
s = _state(m0, period=4, clock=35.0, down=4, dist=6, y=40, prev=v21.NOTHING, home=20, away=14)
s["tos"] = np.array([[3, 0]])
st = {}
h, a = v21.simulate(m0, s, 200, seed=1, stats=st)
print("auto_end at 4th & 6, 0:35, stopped clock, defence 0 timeouts: auto_end", st.get("auto_end"), "snaps", st.get("snaps", 0),
      "away scored", (a > 14).mean())
s["tos"] = np.array([[3, 1]])
st = {}
h, a = v21.simulate(m0, s, 2000, seed=1, stats=st)
print("same with defence 1 timeout: auto_end", st.get("auto_end"), "away scored share", (a > 14).mean(), "away wins", (a > h).mean())
s2 = _state(m0, period=4, clock=140.0, down=1, dist=10, y=40, prev=v21.FRESH, home=20, away=14)
s2["tos"] = np.array([[3, 0]])
st = {}
h, a = v21.simulate(m0, s2, 200, seed=1, stats=st)
print("1st & 10 at 2:20 after a change of possession (clock stopped), defence 0 timeouts: auto_end", st.get("auto_end"), "snaps", st.get("snaps", 0))
