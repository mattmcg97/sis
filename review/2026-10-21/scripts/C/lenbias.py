# Toy: is v21_endgame.states_lines' "each match counting the same" estimator unbiased for a calibrated sim?
import numpy as np
rng = np.random.default_rng(0)
q, K, N = 0.75, 6, 200000
per_match, per_state = [], []
for _ in range(N):
    sims, k = [], 0
    while True:
        sims.append(q ** (K - k))          # the calibrated win chance at this state (exact)
        if rng.random() < q:
            k += 1
            if k == K:
                real = 1.0
                break
        else:
            real = 0.0
            break
    per_match.append(real - np.mean(sims))
    per_state += [real - s for s in sims]
print("first-state win chance %.3f" % q ** K)
print("match-weighted real - sim: %+.4f  (states_lines' estimator)" % np.mean(per_match))
print("state-weighted real - sim: %+.4f" % np.mean(per_state))
