"""The totals lines, value by value, off a directional_pairs csv."""

import csv
import os
import tempfile
import unittest
from unittest import mock

from .. import report, totals_lines
from ..__main__ import main


def row(match, snap, prod_line, cand_line, realized, p1=7, p2=3, market=54, prod_p=0.5, cand_p=0.5,
        period=4, team="Home Team", live="1"):
    r = {k: "" for k in report.PAIR_FIELDS}
    r.update(match_code=match, drive_number=snap, period_number=period, score_p1=p1, score_p2=p2,
             offensive_team=team, market_id=market, prod_line=prod_line, candidate_line=cand_line,
             realized=realized, prod_probability=prod_p, candidate_probability=cand_p,
             prod_live=live, candidate_live=live)
    return r


class TestTotalsLines(unittest.TestCase):

    def write(self, rows):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "directional_pairs_v5.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, report.PAIR_FIELDS)
            w.writeheader()
            w.writerows(rows)
        return d, path

    def test_the_lines_are_read_as_points_needed_and_straddled_values(self):
        # 10 on the board: prod needs 6.5, the candidate 7.5; the game made exactly 7 more
        _, path = self.write([row("A", 1, 16.5, 17.5, 17, prod_p=0.6, cand_p=0.4),
                              row("A", 1, 16.5, 17.5, 17, market=55, prod_p=0.4, cand_p=0.6),
                              row("B", 1, 40.5, 40.5, 30, live="0")])
        (s,) = totals_lines.read(path)
        self.assertEqual((s.prod_need, s.cand_need, s.real, s.straddled()), (6.5, 7.5, 7, (7,)))
        self.assertAlmostEqual(s.cand_over, 0.4)
        self.assertEqual(len(totals_lines.read(path, live_only=False)), 2)

    def test_every_section_prints_and_the_csv_is_written(self):
        rows = [row("A", 1, 16.5, 17.5, 17), row("B", 2, 13.5, 13.5, 20), row("C", 3, 20.5, 23.5, 21),
                row("D", 4, 12.5, 11.5, 12, period=2, team="Away Team")]
        d, path = self.write(rows)
        with mock.patch("builtins.print") as printed:
            main(["totals-lines", d, "--out", d])
        text = "\n".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("==== v5 against prod: 4 total snapshots", text)
        self.assertIn("points still to come between them (1 of 4", text)
        self.assertIn("\n  7  ", text)
        self.assertIn("11,12,13", text)
        self.assertIn("at each points-needed value", text)
        self.assertIn("within 1 -> within 2", text)
        self.assertTrue(os.path.exists(os.path.join(d, "totals_lines_v5.csv")))


if __name__ == "__main__":
    unittest.main()
