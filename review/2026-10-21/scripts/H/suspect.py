"""Can turnovers.count's `suspect` ever fire on an export built by scouting.snapshots_for_match?"""
import itertools
from eAMFCalibrator import scouting
from xgb import turnovers as tv

# exhaustively: every play_kind classify_play can give a PLAY_OVER whose in-play messages hold a TD message
msgs = ["TOUCHDOWN_TEAM_A", "POSSESSION_TEAM_B", "TURNOVER_ON_DOWNS_TEAM_A", "SAFETY_AWARDED_TEAM_B"]
kinds = set()
for k in range(0, len(msgs) + 1):
    for inside in itertools.combinations(msgs, k):
        if "TOUCHDOWN_TEAM_A" in inside:
            kinds.add(scouting.classify_play([], list(inside)))
print("kinds a row can have when its play_messages carry a TD message:", kinds)

# a late touchdown: TEAM_A scores from the 60 on a snap, the TD message lands after that play's PLAY_OVER
M = "M1"
raw = []
def r(msg, kind, team=None, down=None, dist=None, field=None, clock=200, status=None):
    raw.append((M, msg, clock, status, kind, team, down, dist, field, None))
r(1, None, status="FIRST_QUARTER_STARTED")
r(2, "PLAY_STARTED", "TEAM_A", 1, 10, 50, 210)
r(2.5, "PLAY_OVER", "TEAM_A", 1, 10, 60, 205)
r(2.7, "PLAY_STARTED", "TEAM_A", 1, 10, 60, 200)
r(3, "PLAY_OVER", "TEAM_B", 1, 10, 25, 190)      # state after the score (as if the other side has the ball next)
r(4, "TOUCHDOWN_TEAM_A", clock=190)              # late TD message
r(5, "PLAY_STARTED", "TEAM_A", None, None, None, 190)
r(6, "EXTRA_POINT_GOOD_TEAM_A", clock=190)
r(7, "PLAY_OVER", "TEAM_A", None, None, None, 190)
r(8, "PLAY_STARTED", "TEAM_B", 1, 10, 25, 186)
r(9, "PLAY_OVER", "TEAM_B", 2, 6, 29, 180)
scores = [(M, 4, None, 6, 0, 6, 0), (M, 6, None, 1, 0, 7, 0)]
out, dropped = scouting.snapshots_for_match(M, raw, scores, (7, 0), {}, {}, require_quote=False)
for row in out:
    row["team_a_side"] = "home"
    print(row["message"], row["play_kind"], row["offense"], row["down"], row["field_position"],
          repr(row["play_messages"]), row["score_p1"], row["score_p1_at_start"])
rows = [{k: ("" if v is None else str(v)) for k, v in row.items()} for row in out]
print("turnovers.count:", tv.count(rows))
