"""Split the v21-drives same-bets change on the real drives run: moneyline price-only (and its
expected part were prod right), line bets at the same result, and line bets whose result flips."""
import csv, sys
from collections import defaultdict
path = sys.argv[1]
c = sys.argv[2] if len(sys.argv) > 2 else "v21-drives"
f = lambda v: float(v) if v not in ("", None) else None
agg = defaultdict(float)
stake_all = 0.0
for r in csv.DictReader(open(path, newline="", encoding="utf-8")):
    if r["excluded"] == "True" or r[f"candidate_ok_{c}"] != "True":
        continue
    s, rev, rc = f(r["stake"]), f(r["revenue"]), f(r[f"candidate_revenue_{c}"])
    if s is None or rev is None or rc is None:
        continue
    stake_all += s
    d = rc - rev
    agg["total"] += d
    m = r["market"]
    res, resc = r["result"], r[f"candidate_result_{c}"]
    if m == "moneyline":
        key = "ml"
    elif r[f"same_line_{c}"] == "True":
        key = "line_same"
    else:
        key = "line_moved_same_result" if res == resc else "line_moved_flipped"
    agg[key] += d
    agg[key + "_stake"] += s
    if m == "moneyline" and res in ("won", "lost"):
        o, oc, p = f(r["odds"]), f(r[f"candidate_odds_{c}"]), f(r["stream_prob"])
        if o and oc and p is not None:
            agg["ml_expected_if_prod_right"] += s * p * (o - oc)
            agg["ml_information"] += s * ((res == "won") - p) * (o - oc)
print(f"{c}: bets' stake {stake_all:,.0f}")
for k in ("total", "ml", "ml_expected_if_prod_right", "ml_information", "line_same", "line_moved_same_result", "line_moved_flipped"):
    v = agg[k]
    st = agg.get(k + "_stake")
    print(f"  {k:28s} {v:>12,.0f}  ({100 * v / stake_all:+.2f} pts of all stake)" + (f"  stake {st:,.0f}" if st else ""))
