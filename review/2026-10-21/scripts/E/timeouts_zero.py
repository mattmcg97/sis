import sys, os, csv, copy, tempfile; sys.path.insert(0, "/home/user/sis")
from eAMFModel import playover, feed
from eAMFModel.tests.fakes import _matches
m = _matches(n=2, seed=1)
codes = sorted(m)
# the export carries its own counts: pretend TEAM_A called two timeouts in Q4's last 2:00 of both matches
for code in codes:
    n = 0
    for r in m[code]:
        r["timeouts_used_a"], r["timeouts_used_b"] = "0", "0"
        if r["period"] == "4" and float(r["clock_seconds"]) <= 120:
            n = min(2, n + 1)
            r["timeouts_used_a"] = str(n)
before = {c: [(r["timeouts_used_a"], r["timeouts_used_b"]) for r in m[c] if r["period"] == "4"][-3:] for c in codes}
d = tempfile.mkdtemp(dir="/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/E")
path = os.path.join(d, "timeouts.csv")
with open(path, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["match_code", "message", "period", "caller", "prev_play"])
    late = [r for r in m[codes[0]] if r["period"] == "4" and float(r["clock_seconds"]) <= 120]
    w.writerow([codes[0], late[0]["message"], 4, "TEAM_A", ""])      # only the first match is in the file
found = playover.annotate_timeouts(m, path)
after = {c: [(r["timeouts_used_a"], r["timeouts_used_b"]) for r in m[c] if r["period"] == "4"][-3:] for c in codes}
print("calls found:", found)
for c in codes:
    print(c, "export's own counts, Q4 end:", before[c], " after annotate:", after[c])
rec = dict(period=4, offense="TEAM_A"); feed._timeouts_at(rec, [r for r in m[codes[1]] if r["period"] == "4"][-1])
print("match not in timeouts.csv, last Q4 snap: timeouts_left", rec["timeouts_left"])
