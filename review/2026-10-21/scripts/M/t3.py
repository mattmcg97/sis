import os, inspect, textwrap
import numpy as np
from eAMFModel import v21, trader, game as gm
W = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/M/work"
os.environ.pop("EAMF_V21_DRIVES", None)
t = trader.V21Trader(W, paths=4000, prior=os.path.join(W, "avg_form4_w05.csv"), pace=False)
model = t.model
src = textwrap.dedent(inspect.getsource(v21.simulate))
src = src.replace("def simulate(", "def simulate_ot(")
old = "    ot_done = np.zeros((P, 2), bool)\n"
assert old in src
src = src.replace(old, old + "    if 'ot_done' in start:\n        ot_done |= rep(start['ot_done'], bool)\n")
ns = dict(vars(v21)); exec(src, ns); simulate_ot = ns["simulate_ot"]
# OT, level 7-7, home received and punted; away 1st & 10 at its own 25 (game.ot_had_ball = ["home"])
g = gm.Game(period=5, clock=180.0, home=7, away=7, phase=gm.SCRIMMAGE, side=gm.AWAY, down=1, distance=10, field=25,
            timeouts_home=2, timeouts_away=2, opening_receiver=gm.HOME, fresh=True, ot_had_ball=["home"], snaps=60, last="fresh")
pair, teams = ("G1", "G4"), ("T1", "T2")
S = t.start_of(g, pair, teams)
h, a = v21.simulate(model, S, 20000, seed=5)
S2 = dict(S, ot_done=np.array([[True, False]]))
h2, a2 = simulate_ot(model, S2, 20000, seed=5)
pw = lambda h, a: float(((h > a).sum()) / max(1, (h != a).sum()))
print("trader start (home's OT possession forgotten): P(home wins) =", round(pw(h, a), 4))
print("with home's possession marked done:           P(home wins) =", round(pw(h2, a2), 4))
# sanity: same rule from a fresh OT kick-off is unchanged by the patch
