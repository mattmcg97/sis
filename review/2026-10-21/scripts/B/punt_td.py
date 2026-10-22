from eAMFModel import v21, feed
from eAMFModel.tests.test_v21 import _row
rows = [
    _row(10, 1, 200, "KICKOFF", "TEAM_A", 1, 10, 25),
    _row(20, 1, 190, "SCRIMMAGE", "TEAM_A", 2, 8, 27),
    _row(30, 1, 180, "SCRIMMAGE", "TEAM_A", 3, 6, 29),
    _row(40, 1, 170, "SCRIMMAGE", "TEAM_A", 4, 2, 33),
    # punt returned for a touchdown: the feed's play kind is PUNT (punt messages take precedence), the board +6 to B
    _row(50, 1, 160, "PUNT", "TEAM_B", 1, 10, 35, 0, 6, "PUNT_TEAM_A|TOUCHDOWN_TEAM_B"),
    _row(60, 1, 160, "CONVERSION", "TEAM_B", 1, 10, 35, 0, 7, "EXTRA_POINT_GOOD_TEAM_B"),
    _row(70, 1, 155, "KICKOFF", "TEAM_A", 1, 10, 25, 0, 7),
    _row(80, 1, 150, "SCRIMMAGE", "TEAM_A", 2, 10, 25, 0, 7),
]
for r in v21.snap_records(rows):
    print(r["down"], r["field"], "kind", {0: "GAIN", 1: "TURNOVER", 2: "DEF_TD"}[r["kind"]], "gain", r["gain"], "read_off_score", r.get("read_off_score"))
print("kick_decisions:", feed.kick_decisions(rows))
