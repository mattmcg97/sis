"""snap_records: a snap that ends in a safety (its PLAY_OVER the free kick's PUNT row) is dropped."""
from eAMFModel import v21, feed

def row(msg, period, clock, kind, off, down, dist, y, a=0, b=0, messages=""):
    return dict(match_code="AF900", message=str(msg), period=str(period), clock_seconds=str(clock), play_kind=kind,
                offense=off, down=str(down), distance=str(dist), field_position=str(y), score_p1=str(a),
                score_p2=str(b), play_messages=messages, team_a_side="home", opening_offense="TEAM_A")

rows = [row(7, 1, 200, "SCRIMMAGE", "TEAM_A", 2, 10, 2),
        row(14, 1, 192, "SCRIMMAGE", "TEAM_A", 3, 10, 2),       # 2nd down incomplete
        row(21, 1, 185, "PUNT", "TEAM_A", 1, 10, 20, 0, 2, "SAFETY_AWARDED_TEAM_B"),   # 3rd down: tackled in the end zone
        row(28, 1, 180, "SCRIMMAGE", "TEAM_B", 1, 10, 55, 0, 2, "POSSESSION_TEAM_B"),
        row(35, 1, 170, "SCRIMMAGE", "TEAM_B", 2, 6, 59, 0, 2)]
recs = v21.snap_records(rows)
print("snaps kept:", [(r["start"], r["down"], r["field"], r["gain"], r["kind"]) for r in recs])
print("safety kicks seen by feed.safety_kicks:", feed.safety_kicks(rows))
