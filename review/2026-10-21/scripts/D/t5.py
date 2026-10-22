import os, sys, numpy as np
sys.path.insert(0, "/home/user/sis")
D = os.path.dirname(os.path.abspath(__file__))
from eAMFModel import v21, v21_stream as vs, stream, markets
from eAMFModel.tests.test_v21 import MATCHES, HISTORY
codes = sorted(MATCHES)[:6]
lines = ((50, ""), (51, ""), (54, "Total points over 40.5"))
allq = []
for code in codes:
    rows = sorted(MATCHES[code], key=lambda r: int(r["message"]))
    info = [h for h in HISTORY if h["MATCH_CODE"] == code]
    prod = []
    for r in rows:
        for k, l in lines:
            prod.append((code, k, None, 50.0, 2.0, l, int(r["message"]), "OPEN", "true"))
    q = vs.quotes_for_matches({code: rows}, prod, os.path.join(D, "model"), n_paths=1000, workers=1,
                              match_info=info, lines=vs.PROD_LINES, nudge_share=0.0)
    allq += [x for x in q if x[1] in (50, 51)]
p = np.array([x[3] for x in allq])
print("ML quotes", len(p), "at 0.01 or 99.99:", int(((p <= 0.01) | (p >= 99.99)).sum()),
      "below 0.5%:", int(((p < 0.5) | (p > 99.5)).sum()))
