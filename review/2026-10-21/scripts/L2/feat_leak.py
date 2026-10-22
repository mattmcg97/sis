"""audit_features' scramble check against a daily reading that includes the match's own day."""
import csv, datetime as dt, os, sys, tempfile
from collections import defaultdict
from unittest import mock
from eAMFCalibrator import backtest_suite as bs, gamer_features as gf
from eAMFCalibrator.tests import test_prior_backtest as tpb

def sameday_daily(rows, half_life=gf.HALF_LIFE, prior_matches=gf.PRIOR_MATCHES, since=None):
    # the day's matches go into the book BEFORE they are read (own result included)
    by = defaultdict(dict)
    for r in rows:
        if r.get("match_code") and r.get("side") in ("home", "away"):
            by[r["match_code"]][r["side"]] = r
    days = defaultdict(list)
    for m, p in by.items():
        t = gf._start(p.get("home", p.get("away", {})).get("start"))
        if t is not None:
            days[t.date()].append((m, p))
    book, out = gf.Book(half_life, prior_matches), {}
    for day in sorted(days):
        for m, p in days[day]:
            h, a = p.get("home"), p.get("away")
            for r, o in ((h, a), (a, h)):
                if r and r.get("player"):
                    book.add(r["player"].strip().upper(), gf._reads(r, o))
        if since is None or day >= since:
            for m, p in days[day]:
                gh = p["home"]["player"].strip().upper(); ga = p["away"]["player"].strip().upper()
                out[m] = {"home": book.values(gh), "away": book.values(ga),
                          "n_home": book.matches.get(gh, 0.0), "n_away": book.matches.get(ga, 0.0)}
    return out

with tempfile.TemporaryDirectory() as d:
    hist, by, arm = tpb.fixture()
    path = os.path.join(d, "s.csv")
    tpb.write_stats(path, hist, arm)
    window = list(hist)
    T0 = tpb.T0
    first, last = T0.date(), T0.date() + dt.timedelta(days=23)
    audit = bs.Audit()
    with mock.patch.object(gf, "daily", sameday_daily), mock.patch("builtins.print"):
        bs.audit_features(audit, path, window, first, last)
sys.__stdout__.write("\n".join(l for l in audit.lines if "later day" in l) + "\n")
