"""expected_points.build_from_export closes a drive at a QUARTER change (Q1->Q2), scoring its states 0,
though the drive goes on (feed.half_left / v21_report.drive_starts: drives only end at the half)."""
import csv, os, sys, tempfile
from eAMFCalibrator import expected_points as ep

F = ["match_code", "message", "period", "play_kind", "offense", "down", "distance", "field_position", "play_messages"]
rows = [
    # a drive that starts late in Q1 and scores a TD in Q2
    ("M1", 1, 1, "KICKOFF", "TEAM_A", 1, 10, 25, ""),
    ("M1", 2, 1, "SCRIMMAGE", "TEAM_A", 1, 10, 35, ""),     # state 1&10 @35 (Q1)
    ("M1", 3, 1, "SCRIMMAGE", "TEAM_A", 2, 4, 41, ""),      # state 2&4 @41 (Q1)
    ("M1", 4, 2, "SCRIMMAGE", "TEAM_A", 1, 10, 55, ""),     # Q2: same drive goes on
    ("M1", 5, 2, "TOUCHDOWN", "TEAM_A", 1, 10, 85, "TOUCHDOWN_TEAM_A"),
]
d = tempfile.mkdtemp(dir=os.environ.get("SCR", "."))
p = os.path.join(d, "x.csv")
with open(p, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(F); w.writerows(rows)
orig = ep.MIN_STATES
ep.MIN_STATES = 1
table, counts = ep.build_from_export(p)
for k in sorted(counts):
    print(k, counts[k], table.get(k))
