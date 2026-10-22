"""indrive.reconcile cannot fail: the windows partition the scores whatever the drive boundaries are."""
import random
from eAMFCalibrator import indrive
from eAMFCalibrator.drives import PlayRow, ScoreRow
rng = random.Random(1)
bad = 0
for rep in range(2000):
    msgs = sorted(rng.sample(range(1, 400), 30))
    plays = [PlayRow(m, 1, rng.choice(["Home Team", "Away Team", None]), rng.choice([1, 2, 3, 4, None]),
                     rng.choice([10, 3, None]), rng.randint(1, 99), None) for m in msgs]
    scores = [ScoreRow(m, 1, rng.choice([0, 3, 6, 7, None]), rng.choice([0, 2, 7, None]), None, None)
              for m in sorted(rng.sample(range(0, 420), 6))]
    outcomes = indrive.drive_outcomes("M", plays, scores)
    if outcomes and not indrive.reconcile(outcomes, scores)["ok"]:
        bad += 1
print("random plays and scores, reconcile failures:", bad, "of 2000")
# and the run's checks() never sees the scores (__main__ passes four arguments)
f = indrive.checks({"streams": {}}, [], [], outcomes)
print("checks() without scores -> reconcile findings:", [x for x in f if x[1] == "reconcile"])
