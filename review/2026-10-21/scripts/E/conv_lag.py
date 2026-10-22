import sys; sys.path.insert(0, "/home/user/sis")
from eAMFModel import playover
from eAMFCalibrator import scouting

MC = "AF1"
def R(m, status=None, msg=None, team=None, d=None, t=None, f=None, clock=240):
    return (MC, m, clock, status, msg, team, d, t, f, "2026-09-20 10:00:00")
rows = [R(1, "NOT_STARTED"), R(2, "FIRST_QUARTER_STARTED"), R(3, msg="KICKOFF_TEAM_A"),
        R(4, msg="PLAY_STARTED", team="TEAM_B", d=1, t=10, f=35),
        R(5, msg="PLAY_OVER", team="TEAM_B", d=1, t=10, f=25, clock=236),
        R(8, msg="PLAY_STARTED", team="TEAM_B", d=1, t=10, f=25, clock=215),
        R(9, msg="TOUCHDOWN_TEAM_B", team="TEAM_B", clock=205),
        R(10, msg="PLAY_OVER", team="TEAM_B", d=1, t=10, f=25, clock=205),
        R(11, msg="PLAY_STARTED", team="TEAM_B", d=1, t=15, f=85, clock=205),
        R(12, msg="EXTRA_POINT_GOOD_TEAM_B", clock=205),
        R(13, msg="PLAY_OVER", team="TEAM_B", d=1, t=15, f=85, clock=205),
        R(14, msg="KICKOFF_TEAM_B"),
        R(15, msg="PLAY_STARTED", team="TEAM_A", d=1, t=10, f=35, clock=205)]
for label, scores in (("no lag (changes on the scoring messages)", [(MC, 9, 1, 0, 6, 0, 6), (MC, 12, 1, 0, 1, 0, 7)]),
                      ("TD change 3 messages late, XP change 2 late", [(MC, 12, 1, 0, 6, 0, 6), (MC, 14, 1, 0, 1, 0, 7)])):
    snaps, _ = scouting.snapshots_for_match(MC, rows, scores, (0, 7), {}, {}, require_quote=False)
    for s in snaps:
        s["team_a_side"] = "home"
    conv = [s for s in snaps if s["play_kind"] == "CONVERSION"][0]
    text = {k: "" if v is None else str(v) for k, v in conv.items()}
    st, why = playover.state_for(text)
    print(f"{label}: conversion row board {conv['score_p1']}-{conv['score_p2']}, at start "
          f"{conv['score_p1_at_start']}-{conv['score_p2_at_start']} -> state_for {st.home_score}-{st.away_score}"
          f" (true 0-7)")
