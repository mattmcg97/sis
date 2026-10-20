"""playstyle: how each gamer plays by the score, off the export alone."""

import contextlib
import copy
import csv
import io
import os
import tempfile
import unittest

import numpy as np

from .. import playstyle as ps
from .test_v21 import HISTORY, MATCHES


def _planted(slopes, pace_extra=None, n_matches=2000, states=6, seed=1):
    """Observations shaped like read()'s: gamer g's own points still to come get slopes[g] x its lead x the
    share left, and its snaps ahead pace_extra[g] seconds more than everyone's 8."""
    rng = np.random.default_rng(seed)
    G = len(slopes)
    O = {k: [] for k in ("match", "half", "gamer", "opp", "lead", "left", "recv", "y", "mu")}
    S = {k: [] for k in ("match", "half", "gamer", "lead", "seconds")}
    for m in range(n_matches):
        a, b = rng.choice(G, 2, replace=False)
        for left in np.sort(rng.uniform(0.05, 1.0, states))[::-1]:
            L, recv = int(rng.integers(-14, 15)), int(rng.integers(0, 2))
            for sd, (g, o) in enumerate(((a, b), (b, a))):
                lead = L if sd == 0 else -L
                for k, v in (("match", m), ("half", m % 2), ("gamer", g), ("opp", o), ("lead", lead), ("left", left),
                             ("recv", float(recv == sd)), ("mu", 20.0)):
                    O[k].append(v)
                O["y"].append(20.0 * left + 2.0 * (recv == sd) + slopes[g] * lead * left
                              + rng.normal(0, 7.0 * np.sqrt(left)))
        for _ in range(20):
            sd, lead = int(rng.integers(0, 2)), int(rng.integers(-10, 11))
            g = (a, b)[sd]
            extra = 0.0 if pace_extra is None else pace_extra[g]
            for k, v in (("match", m), ("half", m % 2), ("gamer", g), ("lead", lead)):
                S[k].append(v)
            S["seconds"].append(16.0 + (8.0 + extra) * (lead > 0) + rng.normal(0, 6))
    floats = ("lead", "left", "recv", "y", "mu", "seconds")
    f = lambda d: {k: np.asarray(v, float if k in floats else np.int64) for k, v in d.items()}
    return dict(obs=f(O), snaps=f(S), gamers={g: f"G{g:02d}" for g in range(G)}, codes=list(range(n_matches)),
                since="2026-10-01", until="2026-10-18", expected="planted")


class TestRead(unittest.TestCase):

    def test_the_opening_and_every_score(self):
        rows = MATCHES[sorted(MATCHES)[0]]
        states = ps.kick_states(rows)
        self.assertEqual(states[0], (0, 0, 1.0, 0))                      # TEAM_A at home receives first
        for (h0, a0, l0, _), (h1, a1, l1, recv) in zip(states, states[1:]):
            self.assertNotEqual((h0, a0), (h1, a1))
            self.assertGreaterEqual(h1 + a1, h0 + a0)
            self.assertLessEqual(l1, l0)
            scorer = 0 if h1 > h0 else 1
            self.assertEqual(recv, 1 - scorer)                            # the side that conceded receives

    def test_each_state_from_both_sides(self):
        got = ps.read(MATCHES, HISTORY)
        o = got["obs"]
        i = np.arange(len(o["y"]))
        self.assertTrue(np.array_equal(o["lead"], -o["lead"][i ^ 1]))
        self.assertTrue(np.array_equal(o["gamer"], o["opp"][i ^ 1]))
        known = o["recv"] != 0.5
        self.assertTrue(np.allclose((o["recv"] + o["recv"][i ^ 1])[known], 1.0))
        self.assertTrue((o["y"] >= 0).all())
        self.assertEqual(len(got["codes"]), len(MATCHES))
        self.assertEqual(got["expected"], "prod's pre-match lines")

    def test_pace_reads_the_clock_a_snap_took(self):
        snaps = ps.read(MATCHES, HISTORY)["snaps"]
        p = ps.pace(snaps, None, B=50)
        self.assertAlmostEqual(p["ahead"], 28.0)                         # the fakes milk the clock ahead
        self.assertAlmostEqual(p["behind"], 16.0)


class TestTrait(unittest.TestCase):

    def test_a_planted_style_is_found_and_grouped(self):
        slopes = np.repeat([-0.3, 0.0, 0.3], 5)
        out = ps.analyse(_planted(slopes, np.repeat([-4.0, 0.0, 4.0], 5)), min_matches=40, min_half=15, B=100)
        self.assertTrue(out["passes"])
        self.assertGreater(out["trait"]["lo"], 0.5)
        self.assertGreater(out["tau"], 0.1)
        by = {s["gamer"]: s for s in out["gamers"]}
        est = np.array([by[f"G{g:02d}"]["slope"] for g in range(15)])
        self.assertGreater(np.corrcoef(est, slopes)[0, 1], 0.9)
        self.assertEqual({by[f"G{g:02d}"]["style"] for g in range(10, 15)}, {"front-runner"})
        self.assertEqual({by[f"G{g:02d}"]["style"] for g in range(5)}, {"closes up"})
        self.assertTrue(out["pace_passes"])
        self.assertEqual({by[f"G{g:02d}"]["tempo"] for g in range(10, 15)}, {"burns the clock ahead"})
        self.assertAlmostEqual(out["pace"]["diff"], 8.0, delta=0.5)

    def test_no_style_is_no_trait(self):
        out = ps.analyse(_planted(np.zeros(15)), min_matches=40, min_half=15, B=100)
        self.assertFalse(out["passes"])
        self.assertFalse(out["pace_passes"])
        self.assertEqual({s["style"] for s in out["gamers"]}, {""})
        text = "\n".join(ps.report(out))
        self.assertIn("-> not a trait", text)

    def test_the_league_s_lead_is_measured_apart(self):
        """Every gamer a front-runner alike: the league's slope, not any gamer's."""
        out = ps.analyse(_planted(np.full(15, 0.2)), min_matches=40, min_half=15, B=100)
        lg = out["league"]
        self.assertAlmostEqual(lg["slope"], 0.2, delta=0.03)
        self.assertLess(lg["lo"], 0.2)
        self.assertGreater(lg["hi"], 0.2)
        self.assertAlmostEqual(lg["possession"], 2.0, delta=0.3)
        self.assertLess(max(abs(s["slope"]) for s in out["gamers"]), 0.2)
        self.assertFalse(out["passes"])


class TestCli(unittest.TestCase):

    def test_the_report_and_its_csv(self):
        from .. import __main__ as cli
        hist = copy.deepcopy(HISTORY)
        for i, h in enumerate(hist):                                     # the matches over ten days
            h["SCHEDULED_START_TIME_UTC"] = f"2026-10-{1 + i % 10:02d} 12:00:00"
            h.setdefault("SPORT_CODE", "AF")
            h.setdefault("STREAM_NUMBER", "1")
        with tempfile.TemporaryDirectory() as d:
            snaps, history, out = (os.path.join(d, n) for n in ("snaps.csv", "history.csv", "out"))
            rows = [r for c in sorted(MATCHES) for r in MATCHES[c]]
            with open(snaps, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, sorted({k for r in rows for k in r}))
                w.writeheader()
                w.writerows(rows)
            with open(history, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, sorted(hist[0]))
                w.writeheader()
                w.writerows(hist)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                cli.main(["playstyle", "--snapshots", snaps, "--history", history, "--min-matches", "5",
                          "--min-half", "2", "--boot", "20", "--out", out])
            with open(os.path.join(out, "playstyle.txt"), encoding="utf-8") as fh:
                text = fh.read()
            with open(os.path.join(out, "playstyle_gamers.csv"), encoding="utf-8") as fh:
                table = list(csv.DictReader(fh))
        for part in ("PLAYSTYLE", "THE LEAGUE", "EACH GAMER", "is it a trait?", "PACE"):
            self.assertIn(part, text)
        self.assertIn("80 matches", printed.getvalue())
        self.assertEqual(len(table), 7)
        self.assertIn("slope", table[0])
        self.assertIn("pace_beyond", table[0])


if __name__ == "__main__":
    unittest.main()
