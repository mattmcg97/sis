"""bet_moments._states: the feed's safety message is SAFETY_AWARDED_TEAM_x, _base gives SAFETY_AWARDED,
which `kind in ("FIELD_GOAL_GOOD", "SAFETY")` never matches."""
from eAMFCalibrator import bet_moments as bm
rows = [("M", 1, 240, "FIRST_QUARTER_STARTED", None, None, None),
        ("M", 2, 200, None, "PLAY_STARTED", None, 1),
        ("M", 3, 196, None, "SAFETY_AWARDED_TEAM_B", None, None),
        ("M", 4, 196, None, "PLAY_OVER", None, None),
        ("M", 5, 196, None, "PLAY_STARTED", None, None),
        ("M", 6, 196, None, "FIELD_GOAL_GOOD_TEAM_A", None, None),
        ("M", 7, 196, None, "PLAY_OVER", None, None)]
for r, s in zip(rows, bm._states(rows)):
    print(r[4] or r[3], "->", s[0])
