# v21_coverage's distance-draw weights against the weights the sim draws its levels on (dist_weights).
import os, sys
import numpy as np
sys.path.insert(0, "/home/user/sis")
from eAMFModel import v21, v21_coverage as vc
from eAMFModel.tests.fakes import _matches
OUT = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/C/model"
m = v21.Model.load(os.path.join(OUT, v21.MODEL_FILE))
print("near_blended:", v21.near_blended(m), "blend", m.blend, "field", m.field)
recs = []
for code, rows in _matches(n=10, seed=9).items():
    recs += [r for r in v21.match_snaps(rows) if r["period"] <= 4 + v21.MAX_OT]
f = v21._features(recs, m)
end = f["endgame"] > 0
field = np.array([r["field"] for r in recs])
cw, _, _ = vc.kind_weights(m.dist, {k: f[k] for k in m.dist.features}, end, m.blend)
C = m.dist.chain({k: f[k] for k in m.dist.features})
W = v21.dist_weights(m, C, field)
kind = vc.level_kind(m.dist.levels, end)
K = np.stack([kind(np.full(len(recs), li), end) for li in range(len(m.dist.levels))], axis=1)
sim = np.stack([(W * (K == k)).sum(1) for k in range(4)], 1)
print("coverage's reported share by kind (full, situation, no lead, bare):", cw.mean(0).round(3))
print("the sim's own level weights by kind:                            ", sim.mean(0).round(3))
print("states whose weights differ:", np.mean(np.abs(cw - sim).sum(1) > 1e-6).round(3))
