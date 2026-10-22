"""Leak check blind spot: a layer that reads results from LATER matches of the SAME day."""
import tempfile
from collections import defaultdict
from unittest import mock
from eAMFCalibrator import backtest_suite as bs, prior_backtest as pb, rematch_layer
from eAMFCalibrator.tests.test_backtest_suite import Fake, SINCE

real = rematch_layer.walk_forward

def sameday_peek(items, burn_in_days=7):
    rows, layer = real(items, burn_in_days)
    by_day = defaultdict(list)
    for it in items:
        if it["actual_total"] is not None:
            by_day[it["start"].date()].append(it["actual_total"])
    start = {it["match"]: it["start"].date() for it in items}
    own = {it["match"]: it["actual_margin"] for it in items}
    for r in rows:
        tot = by_day.get(start[r["match"]], [])
        if tot:
            # every move nudged by the mean total of the WHOLE day (later matches included)
            r["move_margin"] += 0.5 * (own[r["match"]] or 0)
    return rows, layer

with tempfile.TemporaryDirectory() as d, Fake(d) as f:
    ctx = pb.prepare(None, SINCE, features=f.stats)
    per_day = defaultdict(int)
    for c, m in ctx["window"].items():
        per_day[m["start"].date()] += 1
    audit = bs.Audit()
    with mock.patch.object(rematch_layer, "walk_forward", sameday_peek):
        bs.leak_check(audit, ctx, "model", f.arm)
lines = audit.lines
import sys
sys.__stdout__.write("matches/day: min %d max %d\n" % (min(per_day.values()), max(per_day.values())))
sys.__stdout__.write("\n".join(lines) + "\n")
