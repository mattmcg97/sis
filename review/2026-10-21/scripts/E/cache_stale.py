import sys, tempfile; sys.path.insert(0, "/home/user/sis")
from unittest import mock
from eAMFCalibrator import config, snowflake_io as io

class Cur:
    """EVENT's columns for the check, then the settled matches: one more each time it is asked."""
    def __init__(self, n): self.n = n
    def execute(self, sql, params=()):
        if "INFORMATION_SCHEMA" in sql:
            self.rows = [(c, "TEXT", "YES", k) for k, c in enumerate(io.MATCH_INFO_COLUMNS)]
        else:
            self.rows = [(f"AF{k:03d}011026", "AF", 1, f"2026-10-0{k+1} 10:00:00", "A", "T1", "B", "T2", 7, 3)
                         for k in range(self.n)]
        self.description = [("C%d" % k,) for k in range(len(self.rows[0]))]
    def fetchall(self): return list(self.rows)

d = tempfile.mkdtemp(dir="/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/E")
with mock.patch.object(config, "FETCH_DISK_CACHE", d), mock.patch.object(config, "FETCH_CACHE", False):
    first = io.fetch_history(Cur(3))          # the backtest suite's history step: fetch_history(cur), no until
    later = io.fetch_history(Cur(5))          # a rerun days later: the warehouse has 5 settled matches now
print("first run:", len(first), "matches; rerun (warehouse has 5):", len(later), "matches")
