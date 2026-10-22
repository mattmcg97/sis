import sys, datetime as dt; sys.path.insert(0, "/home/user/sis")
from eAMFCalibrator import scouting, prematch

MC = "AF1"
ko = dt.datetime(2026, 10, 1, 10, 0)
# prod's moneyline: 0.55 pre-match (no message / message 1), then in play 0.80 at message 40 (the home side 14-0 up)
prod = [(MC, 50, ko - dt.timedelta(minutes=5), 55.0, 1.8, "PLAYER 1", None, "OPEN", "true"),
        (MC, 50, ko + dt.timedelta(minutes=12), 80.0, 1.2, "PLAYER 1", 40, "OPEN", "true")]
# SCOUTING_FULL lost the match's head: its first surviving PLAY_STARTED is message 60 (mid first half)
rows = [(MC, 60, 120, None, "PLAY_STARTED", "TEAM_B", 1, 10, 25, "x"),
        (MC, 61, 110, None, "PLAY_OVER", "TEAM_B", 2, 6, 29, "x")]
first_play = next(r[1] for r in rows if scouting._text(r[4]) == "PLAY_STARTED")
print("export's prematch columns:", scouting._prematch(prod, first_play))
print("is_prematch(message 40, first play", first_play, "):", prematch.is_prematch(40, first_play),
      "-- published", prod[1][2] - ko, "after the scheduled kick-off")
