import sys; sys.path.insert(0, "/home/user/sis")
from eAMFModel import feed, v21
from eAMFModel.v21_report import real_drives, drive_starts

def R(msg, kind, off, down, dist, field, clock, period=1, msgs="", s1=0, s2=0):
    return dict(match_code="AF1", message=str(msg), period=str(period), clock_seconds=str(clock),
                play_kind=kind, offense=off, down=str(down), distance=str(dist), field_position=str(field),
                play_messages=msgs, score_p1=str(s1), score_p2=str(s2), team_a_side="home")

rows = [
    R(10, "KICKOFF", "TEAM_A", 1, 10, 25, 236, msgs="KICKOFF_TEAM_B|POSSESSION_TEAM_A"),
    R(20, "SCRIMMAGE", "TEAM_A", 1, 10, 60, 220),
    R(30, "SCRIMMAGE", "TEAM_A", 4, 3, 62, 200),
    # missed FG: TEAM_B takes over at its own 38 (the spot), 1st and 10
    R(40, "FIELD_GOAL", "TEAM_B", 1, 10, 38, 195, msgs="FIELD_GOAL_MISSED_TEAM_A"),
    # TEAM_B's first snap: an 62-yard touchdown
    R(50, "TOUCHDOWN", "TEAM_B", 1, 10, 85, 185, msgs="TOUCHDOWN_TEAM_B", s2=6),
    R(60, "CONVERSION", "TEAM_B", 1, 10, 85, 185, msgs="EXTRA_POINT_GOOD_TEAM_B", s2=7),
    R(70, "KICKOFF", "TEAM_A", 1, 10, 25, 180, msgs="KICKOFF_TEAM_B|POSSESSION_TEAM_A", s2=7),
    R(80, "SCRIMMAGE", "TEAM_A", 2, 4, 31, 170, s2=7),
]
recs = feed.snap_records(rows)
print("feed.snap_records starts:", [(r["start"], r["offense"], r["kind"], r["gain"]) for r in recs])
v = v21.snap_records(rows)
print("v21.snap_records starts: ", [(r["start"], r["offense"], r["kind"], r["gain"]) for r in v])
print("real_drives:", real_drives(rows), "(true drives: A, B (1-play TD after the missed FG), A = 3)")
print("is_fresh(FIELD_GOAL row):", feed.is_fresh(rows[3]), "but FIELD_GOAL in SNAP_KINDS:", "FIELD_GOAL" in feed.SNAP_KINDS)
print("kick_records:", feed.kick_records(rows), "(the only real kick-offs: TEAM_B's after its TD, returned to the 25)")
