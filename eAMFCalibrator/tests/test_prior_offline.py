"""The priors graded against each other on the results alone: off a history CSV, with the gamers'
stats' feature layer and a blend, every variant against the first arm."""

import csv
import os
import tempfile
import unittest
from unittest import mock

from .. import feature_layer, prior_offline as po, snowflake_io
from . import test_prior_backtest as tpb


def write_history(path, hist):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, snowflake_io.HISTORY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(dict(h, SPORT_CODE="AF", STREAM_NUMBER="1") for h in hist)


class TestOffline(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.hist, _, arm = tpb.fixture()
        self.history = os.path.join(d, "history.csv")
        write_history(self.history, self.hist)
        self.arm = os.path.join(d, "arm.csv")
        with open(self.arm, "w", encoding="utf-8") as fh:
            fh.write("\n".join(arm) + "\n")
        # a second arm: the first's expected points pulled towards the league
        self.flat = os.path.join(d, "flat.csv")
        with open(self.flat, "w", encoding="utf-8") as fh:
            fh.write("MATCH_CODE,exp_p1,exp_p2\n")
            for line in arm[1:]:
                code, a, b = line.split(",")
                fh.write(f"{code},{(float(a) + 17) / 2:.3f},{(float(b) + 17) / 2:.3f}\n")
        self.stats = os.path.join(d, "stats.csv")
        tpb.write_stats(self.stats, self.hist, arm)

    def tearDown(self):
        self.tmp.cleanup()

    def run_offline(self, **kw):
        out = os.path.join(self.tmp.name, "out")
        with mock.patch.object(feature_layer, "MIN_TRAIN", 60), mock.patch("builtins.print"):
            path = po.run(out, [("model", self.arm), ("flat", self.flat)], self.history, "2026-09-01 00:00:00",
                          draws=2000, **kw)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        with open(os.path.join(out, "prior_offline.csv"), newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        return text, rows

    def test_every_variant_against_the_first_arm_with_the_stats_and_a_blend(self):
        text, rows = self.run_offline(features=self.stats, blends=["mix=model+both,flat+infer"])
        every = text.split("\n\n")[1]
        lines = [x for x in every.splitlines() if x.startswith("  ") and not x.startswith("  variant")][1:]
        self.assertTrue(lines[0].startswith("  model "))                     # the baseline first
        names = {x.split()[0] for x in lines}
        for v in ("model", "model+layer", "model+infer", "model+both", "model+feat", "model+both+feat",
                  "flat", "flat+both+feat", "mix"):
            self.assertIn(v, names)
        for part in ("Rematches", "First meetings", "Month: 2026-09", "feature layer, on the arm alone (+feat)",
                     "mix: the average of model+both, flat+infer"):
            self.assertIn(part, text)
        self.assertEqual(len(rows), 306)                                     # 432 matches less the burn-in's
        r = rows[100]
        self.assertAlmostEqual(float(r["mix_p_home"]), (float(r["model+both_p_home"]) + float(r["flat+infer_p_home"])) / 2,
                               places=3)
        self.assertEqual(int(r["margin"]), [int(h["PLAYER_1_FINAL_SCORE"]) - int(h["PLAYER_2_FINAL_SCORE"])
                                            for h in self.hist if h["MATCH_CODE"] == r["MATCH_CODE"]][0])

    def test_without_stats_no_feature_variants_and_bad_specs_are_refused(self):
        text, _ = self.run_offline()
        self.assertNotIn("+feat", text)
        with self.assertRaises(SystemExit):
            po.parse_blend("mix=model")
        with self.assertRaises(SystemExit), mock.patch("builtins.print"):
            po.run(self.tmp.name, [("model", self.arm)], self.history, "2026-09-01 00:00:00", draws=2000,
                   blends=["mix=model,nope"])
        with self.assertRaises(SystemExit), mock.patch("builtins.print"):
            po.run(self.tmp.name, [("model", self.arm)], self.history, "2027-01-01 00:00:00")
        with self.assertRaises(SystemExit):
            po.run(self.tmp.name, [], self.history, "2026-09-01 00:00:00")


if __name__ == "__main__":
    unittest.main()
