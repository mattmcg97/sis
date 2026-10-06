"""rolling_prior: one pre-match fit per match day, each match priced by its own day's fit."""

import datetime as dt
import json
import os
import tempfile
import unittest
from unittest import mock

from .. import rolling_prior, v12


class FakeFit:
    """A fitted pre-match model that prices every match at its own cut-off day's number."""

    FOLLOWS_RESULTS = True
    built = []

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, "fit.json")) as fh:
            self.day = json.load(fh)["day"]
        self.league = (17.0, 17.0)

    @classmethod
    def build(cls, history, directory, before):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "fit.json"), "w") as fh:
            json.dump({"day": before.date().isoformat(), "seen": len(history)}, fh)
        cls.built.append(before)
        return cls(directory)

    @staticmethod
    def exists(directory):
        return os.path.exists(os.path.join(directory, "fit.json"))

    def means(self, schedule, results=None, n_sims=None):
        return {r["MATCH_CODE"]: (float(self.day[-2:]), 0.0 if results is None else 1.0) for r in schedule}


def _row(code, when):
    return {"MATCH_CODE": code, "SCHEDULED_START_TIME_UTC": when}


class TestRollingPrior(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "daily")
        FakeFit.built = []
        self.patch = mock.patch.dict(rolling_prior.PRIORS, {"glmer": FakeFit})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_each_match_is_priced_by_the_fit_of_its_own_day(self):
        r = rolling_prior.fit([], "glmer", self.dir, dt.date(2026, 9, 10), dt.date(2026, 9, 12), verbose=False)
        self.assertEqual(sorted(FakeFit.built), [dt.datetime(2026, 9, d) for d in (10, 11, 12)])
        got = r.means([_row("a", "2026-09-10 00:30:00"), _row("b", "2026-09-11 23:59:00"),
                       _row("c", "2026-09-12 12:00:00"), _row("d", "2026-09-20 12:00:00"),
                       _row("e", "2026-09-01 12:00:00")], results=[])
        self.assertEqual({k: v[0] for k, v in got.items()}, {"a": 10, "b": 11, "c": 12, "d": 12, "e": 10})
        self.assertEqual(got["a"][1], 1.0)                        # the results reach the fit's form

    def test_days_already_fitted_are_kept(self):
        rolling_prior.fit([], "glmer", self.dir, dt.date(2026, 9, 10), dt.date(2026, 9, 11), verbose=False)
        FakeFit.built = []
        rolling_prior.fit([], "glmer", self.dir, dt.date(2026, 9, 10), dt.date(2026, 9, 12), verbose=False)
        self.assertEqual(FakeFit.built, [dt.datetime(2026, 9, 12)])

    def test_a_failed_day_is_said_and_skipped_and_its_matches_use_the_day_before(self):
        real = FakeFit.build.__func__

        def flaky(cls, history, directory, before):
            if before.day == 11:
                raise SystemExit("glmer/fit.R failed: boom")
            return real(cls, history, directory, before)

        with mock.patch.object(FakeFit, "build", classmethod(flaky)), \
                mock.patch("builtins.print") as said:
            r = rolling_prior.fit([], "glmer", self.dir, dt.date(2026, 9, 10), dt.date(2026, 9, 12),
                                  verbose=False)
        self.assertEqual(r.days, [dt.date(2026, 9, 10), dt.date(2026, 9, 12)])
        self.assertTrue(any("2026-09-11 FAILED" in str(c) for c in said.call_args_list))
        self.assertEqual(r.means([_row("b", "2026-09-11 12:00:00")], results=[])["b"][0], 10)

    def test_attach_points_a_build_at_the_fits_and_checks_the_kind(self):
        rolling_prior.fit([], "glmer", self.dir, dt.date(2026, 9, 10), dt.date(2026, 9, 10), verbose=False)
        build = os.path.join(self.tmp.name, "v12_model")
        os.makedirs(build)
        with open(os.path.join(build, v12.PRIOR_FILE), "w") as fh:
            json.dump({"prior": "nb2"}, fh)
        with self.assertRaises(SystemExit):
            rolling_prior.attach(build, self.dir)                 # an NB2 build's shrink is NB2's
        with open(os.path.join(build, v12.PRIOR_FILE), "w") as fh:
            json.dump({"prior": "glmer"}, fh)
        rolling_prior.attach(build, self.dir)
        pre = v12.prematch_model(build)
        self.assertIsInstance(pre, rolling_prior.Rolling)         # no shrink file: unwrapped
        self.assertEqual(pre.means([_row("a", "2026-09-10 08:00:00")])["a"][0], 10)
        rolling_prior.detach(build)
        with open(os.path.join(build, v12.PRIOR_FILE)) as fh:
            self.assertNotIn("rolling", json.load(fh))


if __name__ == "__main__":
    unittest.main()
