"""prior_variants: a v14 build copied to price off other priors, and chosen matches laid out by each."""

import csv
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from .. import prior_variants as pv, sim14, v14
from .fakes import _matches


class FakePrior:
    """A fitted pre-match model: every match at the same expected points."""

    FOLLOWS_RESULTS = False
    POINTS = (18.0, 14.0)

    def __init__(self, directory):
        self.directory = directory
        with open(os.path.join(directory, "fit.json")) as fh:
            self.meta = json.load(fh)
        self.league, self.scale = (16.0, 16.0), 1.0

    @classmethod
    def build(cls, history, directory, before):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "fit.json"), "w") as fh:
            json.dump({"before": before.isoformat()}, fh)
        return cls(directory)

    @staticmethod
    def exists(directory):
        return os.path.exists(os.path.join(directory, "fit.json"))

    def means(self, schedule, results=None, n_sims=None):
        return {r["MATCH_CODE"]: self.POINTS for r in schedule}


class FakeNB2(FakePrior):
    POINTS = (15.0, 17.0)


SHRINK = {"total_slope": 0.9, "margin_slope": 0.5, "total_centre": 32.0, "margin_centre": 0.0,
          "before": "2026-10-01T00:00:00"}


class TestPriorVariants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tables = sim14.Tables.build(_matches(), min_records=20)
        cls.grid = v14.PriorGrid.build(cls.tables, n_paths=60)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = os.path.join(self.tmp.name, "v14_model")
        os.makedirs(self.model)
        self.tables.save(os.path.join(self.model, "v14tables.npz"))
        self.grid.save(os.path.join(self.model, "v14grid.npz"))
        with open(os.path.join(self.model, v14.PRIOR_FILE), "w") as fh:
            json.dump({"prior": "glmer"}, fh)
        with open(os.path.join(self.model, v14.SHRINK_FILE), "w") as fh:
            json.dump(SHRINK, fh)
        FakePrior.build([], os.path.join(self.model, "glmer"), v14.dt.datetime(2026, 10, 1))
        self.patches = [mock.patch.dict(v14.PRIORS, {"glmer": FakePrior, "nb2": FakeNB2}),
                        mock.patch.object(pv.nb2_prior, "Prematch", FakeNB2),
                        mock.patch.object(v14, "fit_shrink", return_value=dict(SHRINK, total_slope=1.0)),
                        mock.patch("builtins.print")]
        for p in self.patches:
            p.start()
        self.history = [{"MATCH_CODE": code, "PLAYER_1_HANDLE": "ALPHA", "PLAYER_2_HANDLE": "BRAVO",
                         "PLAYER_1_TEAM": "Lions", "PLAYER_2_TEAM": "Bears", "STREAM_NUMBER": "1",
                         "SCHEDULED_START_TIME_UTC": "2026-10-03 12:00:00", "PLAYER_1_FINAL_SCORE": f1,
                         "PLAYER_2_FINAL_SCORE": f2} for code, f1, f2 in (("AF1", "24", "10"), ("AF2", "7", "14"))]

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_each_variant_shares_the_build_and_differs_only_in_its_prior(self):
        daily = mock.Mock()
        made = pv.make(self.model, self.history, daily=daily)
        self.assertEqual(set(made), {"raw", "layered", "nb2"})
        for d in made.values():
            self.assertTrue(os.path.exists(os.path.join(d, "v14tables.npz")))
        self.assertFalse(os.path.exists(os.path.join(made["raw"], v14.SHRINK_FILE)))
        with open(os.path.join(made["layered"], v14.PRIOR_FILE)) as fh:
            self.assertTrue(json.load(fh)["form_layer"])
        self.assertTrue(os.path.exists(os.path.join(made["layered"], v14.SHRINK_FILE)))
        with open(os.path.join(made["nb2"], v14.PRIOR_FILE)) as fh:
            self.assertEqual(json.load(fh)["prior"], "nb2")
        with open(os.path.join(made["nb2"], v14.SHRINK_FILE)) as fh:
            self.assertEqual(json.load(fh)["total_slope"], 1.0)               # its own shrink
        self.assertEqual(daily.call_args.args[2:4], ("nb2", v14.dt.date(2026, 10, 1)))
        # the build itself, the shrunk glmer, prices off the shrunk means; raw off the bare ones
        self.assertAlmostEqual(sum(v14.prematch_model(self.model).means(self.history)["AF1"]), 32.0 * 0.1 + 0.9 * 32.0)
        self.assertEqual(v14.prematch_model(made["raw"]).means(self.history)["AF1"], (18.0, 14.0))
        # NB2 off its own shrink: the total kept (slope 1), the margin pulled in by half
        self.assertEqual(v14.prematch_model(made["nb2"]).means(self.history)["AF1"], (15.5, 16.5))

    def test_the_matches_side_by_side(self):
        made = pv.make(self.model, self.history, variants=("raw", "nb2"), daily=None)
        snaps = os.path.join(self.tmp.name, "snaps.csv")
        with open(snaps, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["match_code", "prematch_prob_50", "prematch_line_52", "prematch_prob_52",
                        "prematch_line_54", "prematch_prob_54"])
            w.writerow(["AF1", "0.6", "2.5", "0.55", "33.5", "0.5"])
        bets = os.path.join(self.tmp.name, "bets_sim.csv")
        labels = ("v14", "v14-raw", "v14-nb2")
        with open(bets, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["match_code", "excluded", "result", "stake", "revenue"]
                       + [f"candidate_{k}_{lb}" for lb in labels for k in ("ok", "revenue")])
            w.writerow(["AF1", "False", "won", "100", "-80"] + ["True", "-70", "True", "-60", "True", "-90"])
            w.writerow(["AF1", "False", "lost", "50", "50"] + ["True", "50", "True", "50", "False", ""])
            w.writerow(["AF2", "False", "lost", "40", "40"] + ["True", "40", "True", "40", "True", "40"])
        models = {"v14": self.model, "v14-raw": made["raw"], "v14-nb2": made["nb2"]}
        got = pv.compare(models, self.history, ["AF1", "AF2"], snapshots=snaps, bets_csv=bets, n_paths=200)
        first = got[0]
        self.assertEqual(first["prod"]["spread"], (2.5, 0.55))
        self.assertEqual(first["models"]["v14-raw"]["prior"], (18.0, 14.0))
        self.assertEqual(first["models"]["v14"]["bare"], (18.0, 14.0))          # before the shrink
        self.assertNotEqual(first["models"]["v14"]["prior"], (18.0, 14.0))
        self.assertTrue(0 < first["models"]["v14-nb2"]["sim"]["p_home"] < 1)
        # only the bets every model re-priced: AF1's second was not re-priced by v14-nb2
        self.assertEqual(first["bets"], {"stake": 100.0, "prod": -80.0, "v14": -70.0, "v14-raw": -60.0,
                                         "v14-nb2": -90.0})
        text = "\n".join(pv.report(got, list(models)))
        for name in ("prod (implied)", "v14 bare prior", "v14 prior", "v14-raw sim", "v14-nb2 sim", "ML Brier"):
            self.assertIn(name, text)
        out = os.path.join(self.tmp.name, "v14_priors.csv")
        pv.write_csv(out, got, list(models))
        with open(out) as fh:
            self.assertEqual(len(list(csv.reader(fh))), 3)

    def test_the_implied_mean(self):
        self.assertAlmostEqual(pv.implied(2.5, 0.5, 9.0), 2.5)
        self.assertGreater(pv.implied(2.5, 0.6, 9.0), 2.5)
        self.assertIsNone(pv.implied(None, 0.5, 9.0))


if __name__ == "__main__":
    unittest.main()
