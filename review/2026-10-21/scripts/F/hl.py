import datetime as dt, os, json, tempfile
from unittest import mock
from eAMFCalibrator import walkthrough as wt
from eAMFCalibrator.tests.test_nb2_history import write_rolling
from eAMFModel import rolling_prior, nb2_prior
d = tempfile.mkdtemp()
first, last = dt.date(2026, 9, 1), dt.date(2026, 9, 3)
write_rolling(os.path.join(d, "nb2_fits"), [(first + dt.timedelta(days=k)).isoformat() for k in range(3)])
prog = wt.Progress({}, out=lambda s: None, overlay=None)
called = []
with mock.patch.object(rolling_prior, "fit", lambda *a, **k: called.append(k)), \
     mock.patch.object(nb2_prior.Prematch, "exists", staticmethod(lambda p: os.path.exists(os.path.join(p, "level.json")))):
    wt.nb2_fits([{}] * 10, d, first, last, 4, prog, half_life=30)
print("rolling_prior.fit called:", called)
print("index options:", json.load(open(os.path.join(d, "nb2_fits", "rolling.json")))["options"])
stamp = dict(wt.history_stamp([{"SCHEDULED_START_TIME_UTC": "2026-09-03 10:00:00"}], first, last), draws=20000)
print("nb2_prices stamp keys (no half_life):", sorted(stamp))
