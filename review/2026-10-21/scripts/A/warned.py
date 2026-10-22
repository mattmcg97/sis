"""match_snaps: a stopped-clock snap (after an incomplete pass) that crosses 2:00 is marked cut."""
from eAMFModel import v21

def row(msg, period, clock, kind, off, down, dist, y, a=0, b=0, messages=""):
    return dict(match_code="AF900", message=str(msg), period=str(period), clock_seconds=str(clock), play_kind=kind,
                offense=off, down=str(down), distance=str(dist), field_position=str(y), score_p1=str(a),
                score_p2=str(b), play_messages=messages, team_a_side="home", opening_offense="TEAM_A")

rows = [row(7, 2, 160, "SCRIMMAGE", "TEAM_A", 1, 10, 30),     # 1st and 10
        row(14, 2, 128, "SCRIMMAGE", "TEAM_A", 2, 10, 30),    # incomplete: no gain, next down (clock stops)
        row(21, 2, 121, "SCRIMMAGE", "TEAM_A", 3, 10, 30),    # incomplete again, 7s play from a stopped clock (2:08 -> 2:01)
        row(28, 2, 114, "SCRIMMAGE", "TEAM_A", 4, 10, 30)]    # incomplete again, 7s from 2:01 to 1:54
recs = v21.match_snaps(rows)
for r in recs:
    print(dict(clock=r["clock"], seconds=r["seconds"], prev_cls=v21.PREV_NAMES[r["prev_cls"]],
               clock2_running=int(v21.running_of(r["prev_cls"])), cut=r["cut"], play_seconds=r["play_seconds"],
               tb=v21.TIME_BUCKETS[int(v21.time_b(r["play_seconds"]))]))
