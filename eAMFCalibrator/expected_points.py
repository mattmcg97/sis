"""Expected points on a drive, by down, distance and field position.

The in-drive analysis needs to know whether a play left the offence better
or worse off. Yardage alone does not say: five yards on 1st and 10 is
about an average Madden play, not a good one, and the same five yards on
3rd and 12 is a failure. Expected points do say. For every scrimmage state
in real games, the points the offence went on to score on that drive
(7 for a touchdown, 3 for a field goal, -7 for a defensive touchdown, 0 for
anything else), averaged by

    down (1-4) x distance (1-2, 3-5, 6-10, 11+) x field (tens of yards)

A play was good for the offence when it raised that number.

The table ships with the calibrator (expected_points.json, built off
SCOUTING_FULL's PLAY_OVER export) and can be rebuilt from a newer one:

    python -m eAMFCalibrator expected-points out/scouting_playover.csv
"""

import csv
import json
import os
from collections import defaultdict

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_points.json")
MIN_STATES = 20            # a cell needs this many states to be used
TOUCHDOWN_VALUE = 7.0
FIELD_GOAL_VALUE = 3.0
_TABLE = None


def distance_bucket(distance):
    return 0 if distance <= 2 else 1 if distance <= 5 else 2 if distance <= 10 else 3


def field_bucket(field):
    return min(9, max(0, (int(field) - 1) // 10))


def key(down, distance, field):
    return f"{int(down)}|{distance_bucket(int(distance))}|{field_bucket(field)}"


def _drive_points(end_row, offence):
    """What a drive's last event was worth to its offence, or None when the
    row does not end a drive."""
    kind = end_row["play_kind"]
    if kind == "TOUCHDOWN":
        return TOUCHDOWN_VALUE if end_row["offense"] == offence else -TOUCHDOWN_VALUE
    if kind == "FIELD_GOAL":
        return FIELD_GOAL_VALUE if "FIELD_GOAL_GOOD" in (end_row["play_messages"] or "") else 0.0
    if kind in ("PUNT", "TURNOVER_ON_DOWNS", "KICKOFF"):
        return 0.0
    return None


def build_from_export(path):
    """{key: expected points} and the counts behind them, off a
    scouting_playover.csv (PLAY_OVER rows in message order per match)."""
    by_match = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_match[row["match_code"]].append(row)
    sums, counts = defaultdict(float), defaultdict(int)
    for rows in by_match.values():
        rows.sort(key=lambda r: int(r["message"]))
        drive, offence, period = [], None, None

        def close(points):
            for state in drive:
                k = key(state["down"], state["distance"] or 1, state["field_position"])
                sums[k] += points
                counts[k] += 1

        for r in rows:
            if r["period"] != period and drive:
                close(0.0)                              # the half or quarter ran out on it
                drive = []
            period = r["period"]
            if r["play_kind"] == "SCRIMMAGE" and r["down"] and r["field_position"]:
                if drive and r["offense"] != offence:
                    close(0.0)                          # a turnover
                    drive = []
                offence = r["offense"]
                drive.append(r)
                continue
            if drive:
                points = _drive_points(r, offence)
                if points is not None:
                    close(points)
                    drive = []
    table = {k: sums[k] / counts[k] for k in counts if counts[k] >= MIN_STATES}
    return table, dict(counts)


def write(table, counts, source, path=FILE):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"source": source, "states": sum(counts.values()),
                   "table": {k: round(v, 4) for k, v in sorted(table.items())},
                   "counts": dict(sorted(counts.items()))}, fh, indent=0)
    global _TABLE
    _TABLE = None
    return path


def table():
    global _TABLE
    if _TABLE is None:
        try:
            with open(FILE, encoding="utf-8") as fh:
                _TABLE = json.load(fh)["table"]
        except (OSError, ValueError, KeyError):
            _TABLE = {}
    return _TABLE


def value(down, distance, field):
    """Expected points for the offence from this state, or None when the
    state is incomplete or the cell too thin."""
    if down is None or distance is None or field is None or not 1 <= down <= 4:
        return None
    return table().get(key(down, max(1, distance), field))


def change(before, after):
    """How much a play moved the offence's expected points, or None."""
    a = value(before.down_number, before.distance, before.field_position)
    b = value(after.down_number, after.distance, after.field_position)
    return None if a is None or b is None else b - a
