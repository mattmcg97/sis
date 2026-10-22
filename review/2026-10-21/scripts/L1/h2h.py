"""indrive.head_to_head: a candidate that stayed flat while prod moved the WRONG way is scored as a prod win."""
from eAMFCalibrator import indrive, directional
t = indrive.Transition("M", 1, 1, "Home Team", 10, 11, 1, 10, 30, 45, 15, 0, indrive.FIRST_DOWN, True, 1.0)
prod = indrive.Move(t, directional.PROD, 50, 0.60, 0.55, None, None, +1)        # prod moved the wrong way
cand = indrive.Move(t, directional.CANDIDATE, 50, 0.60, 0.60, None, None, +1)   # candidate flat
print("prod.hit", prod.hit, "cand.hit", cand.hit)
h = indrive.head_to_head([prod, cand], n_bootstrap=0)
print("candidate wins / decided:", h["hits"], "/", h["decided"], " ties", h["ties"])
# the mirror: prod flat, candidate wrong
prod2 = indrive.Move(t, directional.PROD, 51, 0.40, 0.40, None, None, -1)
cand2 = indrive.Move(t, directional.CANDIDATE, 51, 0.40, 0.45, None, None, -1)
h2 = indrive.head_to_head([prod2, cand2], n_bootstrap=0)
print("mirror (prod flat, candidate wrong): candidate wins / decided:", h2["hits"], "/", h2["decided"])
