"""v2.1's endgame: the sim's census of how a level game was made level and its decisions' states, the real
games' read the same way, and the report from kick-off and from the same late states."""

import contextlib
import csv
import io
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from .. import v21, v21_endgame as ve
from .test_v21 import HISTORY, MATCHES, _row, built


def _game(tie_late=True, ot=True):
    """A real game: home 7 up, away scores a six and the kick to level it (at 2:00 of Q4, or Q2), then overtime."""
    p, c = (4, 120) if tie_late else (2, 120)
    rows = [_row(1, 1, 200, "TOUCHDOWN", "TEAM_A", 1, 10, 95, a=6),
            _row(2, 1, 200, "CONVERSION", "TEAM_A", 1, 10, 98, a=7),
            _row(3, p, c, "TOUCHDOWN", "TEAM_B", 1, 10, 95, a=7, b=6),
            _row(4, p, c, "CONVERSION", "TEAM_B", 1, 10, 98, a=7, b=7),
            _row(5, 4, 10, "RUSH", "TEAM_A", 1, 10, 50, a=7, b=7)]
    if ot:
        rows.append(_row(6, 5, 200, "FIELD_GOAL", "TEAM_A", 4, 5, 80, a=10, b=7))
    return rows


class TestCensus(unittest.TestCase):

    def test_the_census_leaves_the_draws_alone(self):
        m = built()
        start = v21.kickoff_start(m)
        h0, a0 = v21.simulate(m, start, 300, seed=4)
        st = {"_dec_fine": []}
        h1, a1 = v21.simulate(m, start, 300, seed=4, stats=st)
        self.assertTrue(np.array_equal(h0, h1) and np.array_equal(a0, a1))
        ot, ls, reg = st["went_ot"], st["last_score"], st["regulation"]
        self.assertTrue(np.array_equal(reg[:, 0] == reg[:, 1], ot))
        made = ot & (ls[:, 0] >= 0)
        self.assertTrue(made.any())
        self.assertTrue(np.all(ls[made, 2] + ls[made, 3] == 0))           # the last score made it level
        names = {x[0] for x in st["_dec_fine"]}
        self.assertTrue({"two", "fourth"} <= names)

    def test_the_ways_a_game_is_made_level(self):
        self.assertEqual(ve.tie_kind(v21.PAT, -1, 1), 0)
        self.assertEqual(ve.tie_kind(v21.SIX, -7, 7), 0)                  # the feed's six with its kick in it
        self.assertEqual(ve.tie_kind(v21.TWO, -2, 2), 1)
        self.assertEqual(ve.tie_kind(v21.KICK3, -3, 3), 2)
        self.assertEqual(ve.tie_kind(v21.SIX, -6, 6), 3)
        self.assertEqual(ve.tie_kind(v21.SAFETY, -2, 2), 4)
        self.assertEqual(ve.tie_kind(v21.KICK3, -1, 3), 5)                # not level after it


class TestReal(unittest.TestCase):

    def test_a_real_game_s_ending(self):
        self.assertEqual(ve.real_ending(_game()), (True, (0, True)))
        self.assertEqual(ve.real_ending(_game(tie_late=False)), (True, (0, False)))
        self.assertEqual(ve.real_ending(_game(ot=False)), (False, None))
        rows = _game()
        rows[4] = _row(5, 4, 10, "RUSH", "TEAM_A", 1, 10, 50, a=7, b=4)  # the feed's last row before the tie
        del rows[3]
        rows[2] = _row(3, 4, 120, "TOUCHDOWN", "TEAM_B", 1, 10, 95, a=7, b=4)
        self.assertEqual(ve.real_ending(rows)[1], (6, False))

    def test_the_forks_counted(self):
        names = np.array(["two", "two", "two", "fourth", "fourth", "early_fg"], dtype=object)
        choice = np.array([0, 1, 0, 1, 0, 1])
        period = np.array([4, 4, 2, 4, 4, 4])
        left = np.array([100, 170, 100, 90, 90, 30])
        lead = np.array([-1, -1, -1, -3, -2, -3])
        got = ve.fork_counts(names, choice, period, left, lead)
        self.assertEqual(list(got[0]), [1, 1])                           # 1 behind after the six, late: Q2's out
        self.assertEqual(list(got[4]), [0, 1, 0])                        # 3 behind on 4th down: the kick
        self.assertEqual(list(got[5]), [1, 0, 0])
        self.assertEqual(list(got[6]), [0, 1])


class TestReport(unittest.TestCase):

    def test_the_report(self):
        m = built()
        codes = sorted(MATCHES)[:10]
        means = {c: (10.0, 9.0) for c in codes}
        with tempfile.TemporaryDirectory() as d:
            m.save(os.path.join(d, v21.MODEL_FILE))
            out = ve.run(d, {c: MATCHES[c] for c in codes}, HISTORY, means, "2026-10-01", "2026-10-01", kick_paths=40,
                         state_paths=10)
            ve.write_states_csv(os.path.join(d, "s.csv"), out)
            with open(os.path.join(d, "s.csv"), newline="") as fh:
                states = list(csv.DictReader(fh))
        self.assertEqual(len(out["results"]), 10)
        self.assertEqual(out["results"][0]["kick_games"], 80)
        self.assertTrue(states)
        self.assertTrue(all(abs(int(s["lead"])) <= ve.LEAD_MAX for s in states))
        text = "\n".join(ve.report(out))
        for part in ("FROM KICK-OFF", "overtime: real", "FROM THE SAME LATE STATES", "THE FORK", "by 3"):
            self.assertIn(part, text)

    def test_the_command(self):
        from .. import __main__ as cli
        m = built()
        codes = sorted(MATCHES)[:4]
        means = {c: (9.0, 8.0) for c in codes}
        with tempfile.TemporaryDirectory() as d:
            m.save(os.path.join(d, v21.MODEL_FILE))
            snaps = os.path.join(d, "snaps.csv")
            rows = [r for c in codes for r in MATCHES[c]]
            with open(snaps, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]), extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            with mock.patch("eAMFModel.loading.history_rows", return_value=HISTORY[:4]), \
                    mock.patch("eAMFModel.v21_report.prior_inputs", return_value=(means, {})), \
                    contextlib.redirect_stdout(io.StringIO()):
                cli.main(["v21-endgame", "--model", d, "--prior-model", "v14_model", "--snapshots", snaps,
                          "--since", "2026-10-01", "--paths", "20", "--state-paths", "5", "--workers", "1",
                          "--out", os.path.join(d, "eg")])
            for f in ("v21_endgame.txt", "v21_endgame_states.csv"):
                self.assertTrue(os.path.exists(os.path.join(d, "eg", f)))


if __name__ == "__main__":
    unittest.main()
