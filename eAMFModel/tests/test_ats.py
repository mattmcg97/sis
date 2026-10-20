"""ats: prod's lines and moneyline against the real finals, the league and gamer by gamer."""

import contextlib
import copy
import csv
import io
import os
import tempfile
import unittest

import numpy as np

from .. import ats
from .test_v21 import HISTORY, MATCHES


def _pre(n=4000, seed=1, gamer_edge=None, line=1.5, p=0.516, short=0.05):
    """Matches with prod's favourite by `line` at moneyline `p`: it covers half the time and wins by less than
    the line another `short` of the time; gamer_edge[g] points more for a gamer's side, a gamer a side."""
    rng = np.random.default_rng(seed)
    G = 20
    home, away = rng.integers(0, G, n), rng.integers(0, G, n)
    away = np.where(away == home, (away + 1) % G, away)
    u = rng.random(n)
    margin = np.where(u < 0.5, rng.integers(2, 20, n), np.where(u < 0.5 + short, 1, -rng.integers(1, 20, n))).astype(float)
    if gamer_edge is not None:
        margin = margin + gamer_edge[home] - gamer_edge[away] + rng.normal(0, 0.01, n)
    return dict(match=np.arange(n, dtype=float), half=(np.arange(n) % 2).astype(float),
                home=np.array([f"G{g:02d}" for g in home], object), away=np.array([f"G{g:02d}" for g in away], object),
                line=np.full(n, line), pc=np.full(n, 0.5), p=np.full(n, p), margin=margin)


class TestLeague(unittest.TestCase):

    def test_a_moneyline_too_flat_for_its_line(self):
        out = ats.prematch_lines(_pre(n=12000), B=200)
        self.assertAlmostEqual(out["cover"][0], 0.5, delta=0.02)
        self.assertAlmostEqual(out["short"][0], 0.05, delta=0.01)        # won by exactly 1
        self.assertAlmostEqual(out["short_said"], 0.016, delta=1e-9)       # prod's moneyline less its cover chance
        self.assertAlmostEqual(out["won"][0], 0.55, delta=0.02)
        self.assertGreater(out["won_less"][1], 0)                          # the favourite wins more than prod says
        self.assertEqual(len(out["bands"]), 1)

    def test_in_play_rows_from_the_favourite_s_side(self):
        rng = np.random.default_rng(3)
        n = 6000
        p = rng.uniform(0.05, 0.95, n)
        home_won = rng.random(n) < p
        live = dict(match=(np.arange(n) // 30).astype(float), period=rng.integers(1, 5, n).astype(float), p=p,
                    line=np.zeros(n), pc=p, margin=np.where(home_won, 3.0, -3.0))
        rows = ats.live_lines(live, B=100)
        self.assertTrue(rows)
        for r in rows:
            self.assertLess(abs(r["won"][0] - r["said"]), 0.06)            # calibrated: won what it said
            self.assertAlmostEqual(r["cover"][0], r["won"][0])            # a line of 0: covering is winning


class TestGamers(unittest.TestCase):

    def test_a_gamer_prod_prices_short_every_time(self):
        edge = np.repeat([-2.0, 0.0, 2.0], [6, 8, 6])
        rows = ats.gamers(_pre(gamer_edge=edge), min_matches=40, min_half=15, B=100)
        by = {r["gamer"]: r for r in rows}
        est = np.array([by[f"G{g:02d}"]["ats"] for g in range(20)])
        self.assertGreater(np.corrcoef(est, edge)[0, 1], 0.9)
        from ..playstyle import split_half
        t = split_half([r["ats_a"] for r in rows], [r["ats_b"] for r in rows])
        self.assertGreater(t["lo"], 0.5)

    def test_no_edge_is_no_trait(self):
        from ..playstyle import split_half
        rows = ats.gamers(_pre(), min_matches=40, min_half=15, B=100)
        t = split_half([r["ats_a"] for r in rows], [r["ats_b"] for r in rows])
        self.assertLess(t["lo"], 0)


class TestCli(unittest.TestCase):

    def test_the_report_and_its_csv(self):
        from .. import __main__ as cli
        hist = copy.deepcopy(HISTORY)
        for i, h in enumerate(hist):
            h["SCHEDULED_START_TIME_UTC"] = f"2026-10-{1 + i % 10:02d} 12:00:00"
            h.setdefault("SPORT_CODE", "AF")
            h.setdefault("STREAM_NUMBER", "1")
        with tempfile.TemporaryDirectory() as d:
            snaps, history, out = (os.path.join(d, n) for n in ("snaps.csv", "history.csv", "out"))
            rows = [dict(r, live_50="1", prob_50="0.6", live_52="1", line_52="2.5", prob_52="0.5")
                    for c in sorted(MATCHES) for r in MATCHES[c]]
            with open(snaps, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, sorted({k for r in rows for k in r}))
                w.writeheader()
                w.writerows(rows)
            with open(history, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, sorted(hist[0]))
                w.writeheader()
                w.writerows(hist)
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["ats", "--snapshots", snaps, "--history", history, "--min-matches", "5", "--min-half", "2",
                          "--out", out])
            with open(os.path.join(out, "ats.txt"), encoding="utf-8") as fh:
                text = fh.read()
            with open(os.path.join(out, "ats_gamers.csv"), encoding="utf-8") as fh:
                table = list(csv.DictReader(fh))
        for part in ("PRE-MATCH", "IN PLAY", "AGAINST THE SPREAD", "THE MONEYLINE", "is it a trait?", "won without covering"):
            self.assertIn(part, text)
        self.assertIn("0.60-0.70", text)
        self.assertEqual(len(table), 7)
        self.assertIn("ats", table[0])


if __name__ == "__main__":
    unittest.main()
