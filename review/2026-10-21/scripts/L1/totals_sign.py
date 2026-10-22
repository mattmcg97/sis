"""indrive: the expected direction for a TOTAL on a defensive touchdown / safety (POINTS_AGAINST)."""
from eAMFCalibrator import indrive
from eAMFCalibrator.drives import PlayRow, ScoreRow
H, A = "Home Team", "Away Team"
s = indrive.OUTCOME_SIGN[indrive.POINTS_AGAINST]
print("points_against sign", s)
for mid, name in ((54, "Over prob"), (55, "Under prob")):
    print(f"  {name}: expected {indrive.expected_sign(mid, H, s, indrive.POINTS_AGAINST, in_drive=False):+d}")
print(f"  total line (54): expected {indrive.expected_line_sign(54, H, s, indrive.POINTS_AGAINST, in_drive=False):+d}")
# a pick six: Home has the ball, Away returns it for 7; the drive-ending transition
plays = [PlayRow(10, 2, H, 1, 10, 40), PlayRow(12, 2, H, 2, 6, 44), PlayRow(30, 2, H, 1, 10, 25)]
scores = [ScoreRow(20, 2, None, 6, 0, 6), ScoreRow(21, 2, None, 1, 0, 7)]
for t in indrive.transitions_for_match("M", plays, scores):
    print("  transition", t.from_message, "->", t.to_message, t.outcome, "points", t.points,
          "Over expected", indrive.expected_sign(54, t.offensive_team, t.sign, t.outcome, t.in_drive))
# both sides score in one window: Home TD+PAT then Away kick-return TD+PAT
plays = [PlayRow(10, 2, H, 1, 10, 80), PlayRow(40, 2, H, 1, 10, 25)]
scores = [ScoreRow(20, 2, 6, None, 6, 0), ScoreRow(21, 2, 1, None, 7, 0),
          ScoreRow(30, 2, None, 6, 7, 6), ScoreRow(31, 2, None, 1, 7, 7)]
for t in indrive.transitions_for_match("M", plays, scores):
    print("  14 points in one window:", t.outcome, "points", t.points, "Over expected",
          indrive.expected_sign(54, t.offensive_team, t.sign, t.outcome, t.in_drive))
print("-- with the next drive's rows")
plays = [PlayRow(10, 2, H, 1, 10, 40), PlayRow(12, 2, H, 2, 6, 44), PlayRow(30, 2, H, 1, 10, 25),
         PlayRow(32, 2, H, 2, 7, 28)]
scores = [ScoreRow(20, 2, None, 6, 0, 6), ScoreRow(21, 2, None, 1, 0, 7)]
for t in indrive.transitions_for_match("M", plays, scores):
    print("  transition", t.from_message, "->", t.to_message, t.outcome, "points", t.points, "in_drive", t.in_drive,
          "Over expected", indrive.expected_sign(54, t.offensive_team, t.sign, t.outcome, t.in_drive))
