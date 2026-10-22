# react_gamers.json records the setting each approach's scales were fitted at; updates() never reads it.
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/user/sis")
from eAMFModel import v21, v21_stream as vs, v21_react as vr
from eAMFModel.tests.fakes import _matches
OUT = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/C/model"
m = v21.Model.load(os.path.join(OUT, v21.MODEL_FILE))
real = json.load(open("/home/user/sis/v21_t10/react_gamers.json"))
print({k: (v["setting"], round(v["k"], 2)) for k, v in real.items()})
table = {"snap": dict(setting=1.0, k=real["snap"]["k"], gamers={})}
rows = [vs._as_text(r) for r in list(_matches(n=1, seed=5).values())[0]]
starts = vs.state_starts(m, rows)
for T in (1.0, 4.0, 8.0):
    base = vr.updates(m, rows, starts, None, ("snap", T))
    g = vr.updates(m, rows, starts, None, ("gsnap", T), gamers=table)
    print(f"gsnap{T:g}: accepted against a table fitted at snap1; mean |nudge| snap {np.abs(base).mean():.3f}"
          f" -> gsnap {np.abs(g).mean():.3f}, at the clip {np.mean(np.abs(g) >= v21.NUDGE_MAX - 1e-9):.1%}")
