import os, sys
sys.path.insert(0, "/home/user/sis")
D = os.path.dirname(os.path.abspath(__file__))
from eAMFModel import v21, v21_stream as vs
from eAMFModel.tests.test_v21 import _row
model = v21.Model.load(os.path.join(D, "model", v21.MODEL_FILE))
rows = [_row(10, 1, 240, "KICKOFF", "TEAM_A", 1, 10, 25),
        _row(20, 1, 200, "SCRIMMAGE", "TEAM_A", 4, 8, 70),
        # a missed 47-yard try from the 30: the PLAY_OVER still shows the kicker on offense
        _row(30, 1, 195, "FIELD_GOAL", "TEAM_A", 4, 8, 70, messages="FIELD_GOAL_MISSED_TEAM_A"),
        _row(40, 1, 190, "SCRIMMAGE", "TEAM_B", 1, 10, 30)]
for m, s in vs.state_starts(model, rows):
    print(m, "phase", int(s["phase"][0]), "(KICK=%d SNAP=%d)" % (v21.KICK, v21.SNAP), "team", int(s["team"][0]), "y", int(s["y"][0]))
