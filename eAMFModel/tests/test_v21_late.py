"""v2.1's late Q4 correction: the cells, the rolling fit (a day never fitted on its own games), the book moved
to its shifted chance, the table beside the build and the stream pricing with it."""

import contextlib
import csv
import datetime as dt
import io
import os
import tempfile
import unittest

import numpy as np

from .. import v21, v21_late as vl, v21_stream as vs
from .test_v21 import HISTORY, MATCHES, built


def _states(days=20, change=10, seed=0):
    """Late states: 1-2 behind with 1:30 left, the sim at 0.40 throughout; the real side wins 40% of its games
    until day `change`, 70% from it. Elsewhere (level, 3:00 left) real and sim agree at 0.6."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(days):
        day = dt.date(2026, 9, 1) + dt.timedelta(days=d)
        for m in range(40):
            code = f"AF{m:03d}{day:%d%m%y}"
            p = 0.4 if d < change else 0.7
            out.append(dict(code=code, day=day, cell=vl.cell_of(-1, 90.0), lead=-1, clock=90.0, sim=0.4,
                            real=float(rng.random() < p)))
            out.append(dict(code=code, day=day, cell=vl.cell_of(0, 180.0), lead=0, clock=180.0, sim=0.6,
                            real=float(rng.random() < 0.6)))
    return out


class TestCells(unittest.TestCase):

    def test_a_match_s_day_and_a_state_s_cell(self):
        self.assertEqual(vl.day_of("AF001011026"), dt.date(2026, 10, 1))
        self.assertEqual(vl.day_of("AF000", "2026-10-03 05:00:00"), dt.date(2026, 10, 3))
        self.assertIsNone(vl.day_of("AF000"))
        self.assertEqual(vl.cell_name(vl.cell_of(-3, 200.0)), "3 behind, 4:00-2:01")
        self.assertEqual(vl.cell_name(vl.cell_of(0, 30.0)), "level, 1:00-0:00")
        self.assertIsNone(vl.cell_of(9, 100.0))
        self.assertIsNone(vl.cell_of(-2, 250.0))
        self.assertEqual(vl.shift_of({vl.cell_of(-2, 100.0): 0.5}, 4, 100.0, -2), 0.5)
        self.assertEqual(vl.shift_of({vl.cell_of(-2, 100.0): 0.5}, 3, 100.0, -2), 0.0)    # Q4 alone


class TestFit(unittest.TestCase):

    def test_the_shift_meets_the_real_wins_shrunk(self):
        rows = [dict(code=f"M{i}", sim=0.4, real=float(i % 10 < 6)) for i in range(300)]
        got, n = vl.fit_shift(rows, shrink=0.0)
        self.assertAlmostEqual(got, np.log(0.6 / 0.4) - np.log(0.4 / 0.6), places=3)
        self.assertEqual(n, 300)
        shrunk, _ = vl.fit_shift(rows, shrink=300.0)
        self.assertAlmostEqual(shrunk, got / 2, places=3)

    def test_it_rolls_and_never_reads_its_own_day(self):
        st = _states()
        days = [dt.date(2026, 9, 1) + dt.timedelta(days=d) for d in (10, 19)]
        table = vl.fit(st, days, window=7, shrink=10.0)
        behind, level = vl.cell_of(-1, 90.0), vl.cell_of(0, 180.0)
        self.assertLess(abs(table["2026-09-11"][behind]), 0.3)          # the change starts on its own day: unseen
        self.assertGreater(table["2026-09-20"][behind], 0.8)            # a week of it seen
        self.assertLess(abs(table["2026-09-20"][level]), 0.3)
        text = "\n".join(vl.report(st, table, 7, 10.0))
        self.assertIn("1-2 behind, 2:00-1:01", text)


class TestAdjust(unittest.TestCase):

    def test_the_side_with_the_ball_wins_at_its_shifted_chance(self):
        off = 10
        mp = np.zeros(2 * off + 1)
        mp[off + 3], mp[off - 3], mp[off] = 0.40, 0.55, 0.05
        p = 0.40 / 0.95
        z = np.log(p / (1 - p)) + 0.8
        got = vl.adjust(mp, True, 0.8, off)
        self.assertAlmostEqual(got.sum(), 1.0)
        self.assertAlmostEqual(got[off], 0.05)                          # a level final kept
        self.assertAlmostEqual(got[off + 1:].sum() / 0.95, 1 / (1 + np.exp(-z)))
        away = vl.adjust(mp, False, 0.8, off)                           # the away side has the ball: home down
        self.assertLess(away[off + 1:].sum(), mp[off + 1:].sum())
        self.assertTrue(np.array_equal(vl.adjust(mp, True, 0.0, off), mp))


class TestTableAndStream(unittest.TestCase):

    def test_the_fit_command_writes_the_table_the_stream_reads(self):
        from .. import __main__ as cli
        st = _states(days=16)
        with tempfile.TemporaryDirectory() as d:
            built().save(os.path.join(d, v21.MODEL_FILE))
            path = os.path.join(d, "states.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["code", "clock", "lead", "real_ot", "sim_ot", "real_win", "sim_win"])
                w.writeheader()
                for x in st:
                    w.writerow(dict(code=x["code"], clock=x["clock"], lead=x["lead"], real_ot=0, sim_ot=0,
                                    real_win=x["real"], sim_win=x["sim"]))
            with contextlib.redirect_stdout(io.StringIO()) as out:
                cli.main(["v21-late-fit", "--model", d, "--states", path, "--window", "7"])
            self.assertIn("v2.1-LATE", out.getvalue())
            table = vs.late_table(d)
            self.assertEqual(sorted(table)[0], "2026-09-08")              # the first day with a week before it
            self.assertEqual(sorted(table)[-1], "2026-09-17")             # the day after the last
            self.assertIn(vl.cell_of(-1, 90.0), table["2026-09-17"])
        self.assertTrue(vs.late_of("v21-late"))
        self.assertFalse(vs.late_of("v21") or vs.late_of("v21-pace"))
        with tempfile.TemporaryDirectory() as bare:
            built().save(os.path.join(bare, v21.MODEL_FILE))
            with self.assertRaises(SystemExit):
                vs.late_table(bare)

    def test_the_stream_moves_q4_moneylines_and_leaves_totals(self):
        model = built()
        hist = {h["MATCH_CODE"]: h for h in HISTORY}
        code = next(c for c in sorted(MATCHES)
                    if any(r.get("period") == "4" and abs(int(r["score_p1"]) - int(r["score_p2"])) <= 8 for r in MATCHES[c]))
        rows = sorted(MATCHES[code], key=lambda r: int(r["message"]))
        info = [hist[code]]
        lines = ((50, ""), (54, "Total points over 40.5"))
        prod = [(code, k, None, 50.0, 2.0, line, int(r["message"]), "OPEN", "true") for r in rows for k, line in lines]
        day = vl.day_of(code, hist[code]["SCHEDULED_START_TIME_UTC"]).isoformat()
        with tempfile.TemporaryDirectory() as d:
            model.save(os.path.join(d, v21.MODEL_FILE))
            vl.write(os.path.join(d, vl.LATE_FILE), {day: {c: 1.5 for c in range(len(vl.LEADS) * len(vl.TIMES))}})
            price = lambda late: vs.quotes_for_matches({code: rows}, prod, d, n_paths=60, workers=1, match_info=info,
                                                       lines=vs.PROD_LINES, nudge_share=0.0, late=late)
            off, on = price(False), price(True)
        self.assertEqual(vs.LAST_TIMING.get("late_matches"), 1)
        self.assertEqual(len(off), len(on))
        ml = lambda q: [x[3] for x in q if x[1] in (50, 51)]
        tot = lambda q: [x[3] for x in q if x[1] in (54, 55)]
        self.assertNotEqual(ml(off), ml(on))
        self.assertEqual(tot(off), tot(on))


if __name__ == "__main__":
    unittest.main()
