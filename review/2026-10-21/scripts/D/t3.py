import os, sys, numpy as np, bisect
sys.path.insert(0, "/home/user/sis")
D = os.path.dirname(os.path.abspath(__file__))
from eAMFModel import v21, v21_stream as vs, stream, markets
from eAMFModel.tests.test_v21 import MATCHES, HISTORY
code = sorted(MATCHES)[0]
rows = sorted(MATCHES[code], key=lambda r: int(r["message"]))
info = [h for h in HISTORY if h["MATCH_CODE"] == code]
lines = ((50, ""), (51, ""), (52, "PLAYER 1 to score over 3.5 points more than PLAYER 2"),
         (53, "PLAYER 2 to score over -3.5 points more than PLAYER 1"),
         (54, "Total points over 40.5"), (55, "Total points under 40.5"))
prod = [(code, k, f"2026-10-01 00:0{k - 50}:00", 50.0, 2.0, l, None, "OPEN", "true") for k, l in lines]
for r in rows:
    for k, l in lines:
        prod.append((code, k, None, 50.0, 2.0, l, int(r["message"]), "OPEN", "true"))
q = vs.quotes_for_matches({code: rows}, prod, os.path.join(D, "model"), n_paths=200, workers=1,
                          match_info=info, lines=vs.LINE_MODES, nudge_share=0.0)
for mode in vs.LINE_MODES:
    by = {(x[1], x[6]): (x[3], stream._parse_line(x[5])) for x in q[mode]}
    bad = []
    for (k, m), (p, l) in by.items():
        if k in (50, 52, 54):
            o = by.get((k + 1, m))
            if o is None: continue
            # line consistency
            if k == 52 and l is not None and o[1] is not None and abs(l + o[1]) > 1e-9: bad.append(("lines", k, m, l, o[1]))
            if k == 54 and l is not None and abs(l - o[1]) > 1e-9: bad.append(("lines", k, m, l, o[1]))
            if abs(p + o[0] - 100) > 0.05: bad.append(("sum", k, m, p, o[0], l, o[1]))
    print(mode, len(q[mode]), "bad", len(bad), bad[:4])
