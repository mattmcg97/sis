import pickle
import numpy as np
from eAMFModel import v21
SC = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/B"
for name in ("tilt", "full"):
    m = v21.Model.load(SC + f"/{name}.pkl")
    st = {}
    h, a = v21.simulate(m, v21.kickoff_start(m), 2000, seed=5, stats=st)
    fin = st["final"]
    print(name, "mean", h.mean(), a.mean(), "q_points match", (st["q_points"].sum(1) == fin.sum(1)).all(),
          "unfinished", st["unfinished"], "ot share", st["went_ot"].mean(), "level finals", (fin[:, 0] == fin[:, 1]).mean())
    log = v21.snap_log(st)
    print("  snaps/game", len(log["path"]) / 4000, {k: st.get(k) for k in ("touchdowns", "field_goals", "safeties", "kick_tds", "auto_end", "clock_ran_out", "two_minute", "timeouts", "late_timeouts", "punts", "turnovers")})
    # receiving side 0 vs 1 symmetric check
    print("  home recv: home-away", (h[0] - a[0]).mean(), " away recv: home-away", (h[1] - a[1]).mean())
