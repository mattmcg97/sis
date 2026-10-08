"""kickoff_check: one kick-off simulated many times against the real totals."""

import csv
import os
import tempfile
import unittest

import numpy as np

from .. import kickoff_check, sim12, v12
from .fakes import _matches


class TestKickoffCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        tables = sim12.Tables.build(_matches(), min_records=20)
        tables.save(os.path.join(cls.tmp.name, "v12tables.npz"))
        v12.PriorGrid.build(tables, n_paths=60).save(os.path.join(cls.tmp.name, "v12grid.npz"))
        cls.history = os.path.join(cls.tmp.name, "history.csv")
        with open(cls.history, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["MATCH_CODE", "SCHEDULED_START_TIME_UTC", "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"])
            w.writerows([("A", "2026-09-10 10:00:00", 24, 7), ("B", "2026-09-11 10:00:00", 17, 14),
                         ("C", "2026-09-12 10:00:00", 21, 17), ("D", "2026-08-01 10:00:00", 3, 0),
                         ("E", "2026-09-13 10:00:00", "", "")])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_totals_count_the_settled_matches_in_the_window(self):
        real, n = kickoff_check.real_totals(self.history, since="2026-09-01")
        self.assertEqual(n, 3)
        self.assertAlmostEqual(real[31], 2 / 3)
        self.assertAlmostEqual(real[38], 1 / 3)

    def test_the_sim_gives_a_distribution_and_the_report_reads_it(self):
        sim = kickoff_check.sim_totals(self.tmp.name, (15.5, 15.5), n_paths=400)
        self.assertAlmostEqual(float(sim.sum()), 1.0)
        real, n = kickoff_check.real_totals(self.history, since="2026-09-01")
        text = "\n".join(kickoff_check.report(sim, real, n, (15.5, 15.5)))
        self.assertIn("the share at 31 itself", text)
        self.assertRegex(text, r"\n\s+31\s+[\d.]+%\s+66\.7%")


if __name__ == "__main__":
    unittest.main()
