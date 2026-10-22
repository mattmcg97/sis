"""indrive's P column: a row-level sign test over moves, where each transition gives mirrored moves on both
selections of a market (identical hits) -- under a coin-flip model, how often is p < 0.05?"""
import random
from eAMFCalibrator import indrive, directional
rng = random.Random(0)
t0 = indrive.Transition("M", 1, 1, "Home Team", 10, 11, 1, 10, 30, 45, 15, 0, indrive.FIRST_DOWN, True, 1.0)
import dataclasses
reject_p = reject_ci = 0
R = 300
for rep in range(R):
    moves = []
    for m in range(40):                       # 40 matches
        for k in range(30):                   # 30 transitions each
            t = dataclasses.replace(t0, match_code=f"M{m}", from_message=k, to_message=k + 1)
            for pair in ((50, 51), (52, 53), (54, 55)):
                up = rng.random() < 0.5        # a coin: the price goes up or down at random
                d = 0.01 if up else -0.01
                moves.append(indrive.Move(t, directional.PROD, pair[0], 0.5, 0.5 + d, None, None, +1))
                moves.append(indrive.Move(t, directional.PROD, pair[1], 0.5, 0.5 - d, None, None, -1))
    b = indrive.block(moves, n_bootstrap=200)
    reject_p += b["p_value"] < 0.05
    reject_ci += not (b["ci_low"] <= 0.5 <= b["ci_high"])
print(f"coin-flip model, {R} reps: sign-test p<0.05 in {reject_p / R:.1%}; clustered CI excludes 50% in {reject_ci / R:.1%}")
