import os, sys, numpy as np, bisect
sys.path.insert(0, "/home/user/sis")
D = os.path.dirname(os.path.abspath(__file__))
from eAMFModel import v21, v21_stream as vs, stream
from eAMFModel.tests.test_v21 import MATCHES, HISTORY
code = sorted(MATCHES)[0]
rows = sorted(MATCHES[code], key=lambda r: int(r["message"]))
info = [h for h in HISTORY if h["MATCH_CODE"] == code]
model = v21.Model.load(os.path.join(D, "model", v21.MODEL_FILE))
starts = vs.state_starts(model, [vs._as_text(r) for r in rows])
keys = [m for m, _ in starts]
kinds = {int(r["message"]): r["play_kind"] for r in rows}
print("rows", len(rows), "starts", len(starts))
print("first rows:", [(r["message"], r["play_kind"], r["period"], r["clock_seconds"]) for r in rows[:4]])
# which PLAY_OVER kinds are not keys
from collections import Counter
print("no book:", Counter(kinds[int(r["message"])] for r in rows if int(r["message"]) not in set(keys)))
# the opening: prod quotes between first PLAY_OVER and key[0]
print("keys[0]", keys[0], "first msg", rows[0]["message"])
# rows at clock 0
print("clock0 rows:", [(r["message"], r["play_kind"], r["period"]) for r in rows if r["clock_seconds"] in ("0",)][:10])
