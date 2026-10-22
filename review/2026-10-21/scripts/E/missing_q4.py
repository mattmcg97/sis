import sys; sys.path.insert(0, "/home/user/sis")
from eAMFCalibrator import scouting
from eAMFModel import playover, feed

MC = "AF1"
def build(drop_status=None, drop_message=None):
    rows, m = [], [0]
    def R(status=None, msg=None, team=None, d=None, t=None, f=None, clock=240):
        m[0] += 1
        rows.append((MC, m[0], clock, status, msg, team, d, t, f, "2026-09-20 10:00:00"))
    R("THIRD_QUARTER_STARTED")
    R(msg="KICKOFF_TEAM_A"); R(msg="PLAY_STARTED", team="TEAM_B", d=1, t=10, f=35); R(msg="PLAY_OVER", team="TEAM_B", d=1, t=10, f=25, clock=236)
    y, clock = 25, 236
    for q in (3, 4):
        if q == 4:
            R("FOURTH_QUARTER_STARTED"); clock = 240
        for k in range(8):
            R(msg="PLAY_STARTED", team="TEAM_B", d=1, t=10, f=y, clock=clock)
            y, clock = y + 5, clock - 28
            R(msg="PLAY_OVER", team="TEAM_B", d=1, t=10, f=y, clock=max(clock, 0))
    out = [r for r in rows if not (drop_status and r[3] == drop_status) and r[1] != drop_message]
    return out
scores = []
for label, kw in (("complete", {}), ("FOURTH_QUARTER_STARTED lost", {"drop_status": "FOURTH_QUARTER_STARTED"})):
    snaps, _ = scouting.snapshots_for_match(MC, build(**kw), scores, (0, 0), {}, {}, require_quote=False)
    q4 = snaps[-3]
    text = {k: "" if v is None else str(v) for k, v in q4.items()}
    text["team_a_side"] = "home"
    st, _ = playover.state_for(text)
    print(f"{label}: a Q4 PLAY_OVER is read as period {st.period}, clock {st.clock_seconds}: seconds left in the "
          f"half {feed.half_left(st.period, st.clock_seconds):.0f}")
for label, kw in (("complete", {}), ("one PLAY_OVER lost (message 8)", {"drop_message": 8})):
    snaps, _ = scouting.snapshots_for_match(MC, build(**kw), scores, (0, 0), {}, {}, require_quote=False)
    text = [{k: "" if v is None else str(v) for k, v in dict(s, team_a_side="home").items()} for s in snaps]
    recs = feed.snap_records(text)
    print(f"{label}: {len(recs)} snaps; first three (gain, seconds):", [(r["gain"], r["seconds"]) for r in recs[:3]])
