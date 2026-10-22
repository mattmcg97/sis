import os, json, csv
import numpy as np
from eAMFModel import trader, game as gm, v21_stream
W = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/M/work"
table = os.path.join(W, "avg_form4_w05.csv")
t = trader.V21Trader(W, paths=600, prior=table)
print("version", t.version, "nudge_share", t.nudge_share)
pre = trader.respond(t, "/api/prematch", {"setup": {"match": "AF000"}})[1]
print("prematch", pre)
setup = dict(pre, match="AF000", opening_receiver="home", margin=0.05, paths=600)
# 1. blank margin as the page sends it (NaN -> null)
bad = dict(setup, margin=None)
try:
    print(trader.respond(t, "/api/start", {"setup": bad})[0])
except Exception as e:
    print("blank margin ->", type(e).__name__, e)
# 2. share 1.0 (trader default) vs 0.75 (betting sim default)
for share in (1.0, 0.75):
    tt = trader.V21Trader(W, paths=600, prior=table, nudge_share=share)
    p = tt.price(setup)
    print(f"share {share}: ML home {p['moneyline']['home']['p']:.4f}  exp {p['expected']}  nudge {p['v21']['nudge']}")
# 3. env pace table: the stream reads $EAMF_V21_DRIVES before the prior table; the trader doesn't
alt = os.path.join(W, "pace_alt.csv")
with open(alt, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["MATCH_CODE", "exp_drives", "league_drives"])
    for c, d in (("AF000", 8.0), ("AF001", 10.0), ("AF002", 12.0)):
        w.writerow([c, d, 10.0])
os.environ["EAMF_V21_DRIVES"] = alt
stream_tab = v21_stream.drives_table(None, table)
lvl = np.mean(list(stream_tab.values()))
t2 = trader.V21Trader(W, paths=600, prior=table)
print("stream pace AF000:", stream_tab["AF000"] / lvl, " trader pace AF000:", t2._pace(setup))
