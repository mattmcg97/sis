"""kickoffs.half_time credits each Q2 play to the offense on its PLAY_OVER row -- the NEXT snap's offense
(feed.py's convention) -- so a punt or a turnover is credited to the side that got the ball."""
import numpy as np
from eAMFModel.tests.fakes import _matches
from eAMFCalibrator import kickoffs

ms = _matches(n=3000, seed=11)
bias, cred, true = [], [], []
for code, rows in ms.items():
    # the scouting export's extra columns, as scouting.snapshots_for_match makes them
    prev_score = ("0", "0")
    for i, r in enumerate(rows):
        r["scrimmage"] = "0" if r["play_kind"] in ("KICKOFF", "CONVERSION") else "1"
        r["next_offense"] = r["offense"]                 # the next PLAY_STARTED's offense
        r["next_field_position"] = r["field_position"]
        r["score_p1_at_start"], r["score_p2_at_start"] = prev_score
        prev_score = (r["score_p1"], r["score_p2"])
        r["home_handle"], r["away_handle"] = "H", "A"
    m = kickoffs.Match(code, rows)
    h = kickoffs.half_time(m, kickoffs.kicks(m))
    if h is None:
        continue
    # the truth: the side that ran the play is the offense on the row BEFORE it
    q2 = [(rows[i - 1], r) for i, r in enumerate(rows) if i and r["period"] == "2" and r["scrimmage"] == "1"]
    r_team = "TEAM_A" if h["r"] == "home" else "TEAM_B"
    t_r = sum(p["offense"] == r_team for p, _ in q2)
    t_k = len(q2) - t_r
    cred.append(h["q2_snaps_r"] - h["q2_snaps_k"])
    true.append(t_r - t_k)
cred, true = np.array(cred, float), np.array(true, float)
print(f"{len(cred)} matches; R less K Q2 snaps: as kickoffs counts {cred.mean():+.3f} (se {cred.std()/np.sqrt(len(cred)):.3f}),"
      f" by who ran the play {true.mean():+.3f} (se {true.std()/np.sqrt(len(true)):.3f});"
      f" matches where they differ {np.mean(cred != true):.1%}, mean |diff| {np.abs(cred-true).mean():.2f}")
