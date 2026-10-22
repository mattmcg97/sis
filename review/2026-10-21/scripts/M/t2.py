import os, csv
import numpy as np
from eAMFModel import v21, v21_stream
from eAMFModel.tests.test_v21 import MATCHES, HISTORY
from eAMFCalibrator import replay_ui
W = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/M/work"
table = os.path.join(W, "avg_form4_w05.csv")
exp = os.path.join(W, "export.csv"); hist = os.path.join(W, "history.csv")
codes = sorted(MATCHES)[:3]
rows = [r for c in codes for r in MATCHES[c]]
keys = list(rows[0].keys())
with open(exp, "w", newline="") as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
with open(hist, "w", newline="") as fh:
    w = csv.DictWriter(fh, list(HISTORY[0].keys()) + ["STREAM_NUMBER", "SPORT_CODE"]); w.writeheader(); w.writerows([dict(r, STREAM_NUMBER="1", SPORT_CODE="AF") for r in HISTORY[:3]])
os.environ.pop("EAMF_V21_DRIVES", None)
rp = replay_ui.build(W, table, exp, hist, None, paths=1000)
code = codes[0]
v = rp.load(code)
print("prior", v["prior"], "pace", v["pace"], "n_priced", v["n_priced"])
p = rp.price(code, None, v["prior"], None, None, v["pace"]["exp"])
print("price(): alpha", p["nudge"]["alpha"], "tempo", p["nudge"]["pace"]["tempo"], "ML", p["moneyline"]["home"])
pa = rp.price_all(code, v["prior"], None, None, v["pace"]["exp"])
a, kick, own, nudged, tempo, pace = rp.nudge(code, v["prior"], rp.share, replay_ui.CHART_PATHS, v["pace"]["exp"])
print("price_all(): alpha", [round(float(x),3) for x in a], "tempo", round(tempo,3), "kick ML", pa["points"][0]["home"])
