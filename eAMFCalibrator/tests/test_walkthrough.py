"""The pre-match walkthrough: its progress lines and finish estimates, every variant's table off the
two models and their layers, a rerun keeping what's made, and a sim's prior read off a table."""

import csv
import datetime as dt
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from .. import prior_backtest as pb, walkthrough as wt
from . import test_prior_backtest as tpb
from .test_nb2_history import write_rolling
from .test_prior_offline import write_history


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class TestProgress(unittest.TestCase):

    def test_lines_the_time_left_and_the_finish(self):
        clock, lines = Clock(), []
        start = dt.datetime(2026, 10, 16, 12, 0)
        prog = wt.Progress({"a": 100.0, "b": 50.0}, clock=clock,
                           now=lambda: start + dt.timedelta(seconds=clock.t - 1000), out=lines.append)
        prog.begin("a", "step a", 10, "days")
        self.assertIn("about 1m40s; the whole run about 2m30s more, finishing about 12:02", lines[-1])
        clock.t += 20
        prog.tick(force=True)                           # 1 of 10 in 20s: 3 minutes left, b's 50s after
        self.assertIn("1/10 days (10%), 20s in, ~3m00s left; the run ~3m50s left, finishing about 12:04", lines[-1])
        clock.t += 180
        prog.tick(9)
        prog.end()
        self.assertIn("step a: done in 3m20s", lines[-1])
        prog.begin("b", "step b", 5, "fits")            # a ran at 2x its estimate: b's 50s reads as 1m40s
        self.assertIn("about 1m40s", lines[-1])

    def test_side_by_side_the_plan_stands_until_the_first_round_is_done(self):
        clock, lines = Clock(), []
        prog = wt.Progress({"a": 400.0}, clock=clock, out=lines.append)
        prog.begin("a", "fits", 40, "days", parallel=4)
        clock.t += 50
        prog.tick(0, force=True)                        # none done yet: the plan, less the time gone
        self.assertIn("~5m50s left", lines[-1])
        clock.t += 50
        prog.tick(force=True)                           # 1 of 40 after 100s, four at a time: 39 x 100 / 4
        self.assertIn("~16m15s left", lines[-1])
        for _ in range(3):
            prog.tick(force=True)                       # the round of four done in 100s: 36 x 100 / 4
        self.assertIn("~15m00s left", lines[-1])


    def test_on_a_terminal_the_progress_line_is_drawn_over_itself(self):
        from eAMFModel import progress
        from eAMFModel.tests.test_progress import TTY, shown
        clock, lines, tty = Clock(), [], TTY()
        prog = wt.Progress({"a": 100.0}, clock=clock, out=lines.append, overlay=progress.Line(tty))
        prog.begin("a", "step a", 10, "days")
        clock.t += 20
        prog.tick(force=True)
        clock.t += 1
        prog.tick(note="day 2")
        self.assertEqual(len(lines), 1)                 # the step's own line alone printed
        self.assertIn("2/10 days (20%)", shown(tty.getvalue()))
        self.assertIn("[day 2]", shown(tty.getvalue()))
        prog.tick(8)
        prog.end()
        self.assertEqual(shown(tty.getvalue()), "")     # cleared before the step's last line
        self.assertIn("step a: done in", lines[-1])
        sent = wt.Progress({"a": 100.0}, clock=clock, out=lines.append, overlay=progress.Line(io.StringIO()))
        self.assertIsNone(sent.overlay)                 # not a terminal: lines, as before


def cadence_runner(arm_lines, hist):
    """A stand-in for glmer/cadence.R: progress lines, then cadence_matches.csv off the fixture's arm."""
    mus = {line.split(",")[0]: line.split(",")[1:] for line in arm_lines[1:]}

    def run(args, on_line):
        out = next(a.split("=", 1)[1] for a in args if a.startswith("--out="))
        on_line(f"[10:00:00] {wt.GLMER_ARM}: 24 fits in 2 stretch(es)")
        on_line("starting worker pid=1 on localhost")
        for part in (1, 2):
            for k in range(1, 12):
                on_line(f"[10:00:0{part}] {wt.GLMER_ARM} {part}/2: {k}/12 fits, 1.0 s a fit, ~0 min left")
            on_line(f"[10:00:0{part}] {wt.GLMER_ARM} {part}/2: done, 12 fits in 0.1 min (1.0 s a fit)")
        with open(os.path.join(out, "cadence_matches.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["MatchId", "Time", f"{wt.GLMER_ARM}|1", f"{wt.GLMER_ARM}|2"])
            for h in hist:
                a, b = mus[h["MATCH_CODE"]]
                w.writerow([h["MATCH_CODE"], h["SCHEDULED_START_TIME_UTC"], float(a) + 0.5, float(b) - 0.5])
        return 0
    return run


class TestRun(unittest.TestCase):

    def test_every_variant_a_table_and_a_rerun_keeps_them(self):
        hist, _, arm = tpb.fixture()
        mus = {line.split(",")[0]: tuple(map(float, line.split(",")[1:])) for line in arm[1:]}
        fitted_days = []

        def fit(history, prior, directory, first, last, workers=4, verbose=True, options=None, on_fit=None):
            days = [(first + dt.timedelta(days=k)).isoformat() for k in range((last - first).days + 1)]
            write_rolling(directory, days)
            for d in days:
                fitted_days.append(d)
                if on_fit:
                    on_fit(d, 1.0)

        def read_rolling(directory, rows, history, progress=None):
            if progress:
                progress(1, 1)
            codes = {r["MATCH_CODE"] for r in rows}
            return ({c: mus[c] for c in codes}, {c: ("nb2", mus[c] + (0.15, 0.15, 0.4)) for c in codes})

        from eAMFModel import nb2_prior, rolling_prior
        with tempfile.TemporaryDirectory() as d:
            history = os.path.join(d, "history.csv")
            write_history(history, hist)
            out = os.path.join(d, "wt")
            lines = []
            with mock.patch.object(rolling_prior, "fit", fit), mock.patch.object(pb, "read_rolling", read_rolling), \
                    mock.patch.object(nb2_prior.Prematch, "exists",
                                      staticmethod(lambda p: os.path.exists(os.path.join(p, "level.json")))), \
                    mock.patch("builtins.print", lambda *a, **k: lines.append(" ".join(map(str, a)))):
                path = wt.run(out, history, "2026-09-01 00:00:00", cores=2, draws=2000,
                              cadence_runner=cadence_runner(arm, hist))
                first_run = list(lines)
                lines.clear()
                wt.run(out, history, "2026-09-01 00:00:00", cores=2, draws=2000,
                       cadence_runner=lambda args, on_line: self.fail("glmer refitted"))
            with open(path) as fh:
                manifest = json.load(fh)
            tables = {}
            for name, _ in wt.VARIANTS:
                with open(manifest["variants"][name]["file"], newline="") as fh:
                    tables[name] = {r["MATCH_CODE"]: r for r in csv.DictReader(fh)}
            with open(os.path.join(out, "walkthrough.txt")) as fh:
                text = fh.read()
            from eAMFModel import table_prior, v21_stream
            pre = v21_stream.prematch_model(manifest["variants"]["avg"]["file"])
            self.assertIsInstance(pre, table_prior.TablePrior)
            self.assertEqual(v21_stream._prior_tables(manifest["variants"]["avg"]["file"]),
                             manifest["variants"]["avg"]["file"])
            got = pre.means([{"MATCH_CODE": "AF00100"}, {"MATCH_CODE": "nope"}])
        self.assertEqual(len(fitted_days), 24)                          # each day once: the rerun kept them
        self.assertEqual(set(tables), {n for n, _ in wt.VARIANTS})
        self.assertEqual(len(tables["avg"]), 432)
        for code in ("AF00000", "AF00300"):
            r = {n: t[code] for n, t in tables.items()}
            self.assertAlmostEqual(float(r["nb2"]["home_pts"]), mus[code][0], places=3)
            self.assertAlmostEqual(float(r["glmer"]["home_pts"]), mus[code][0] + 0.5, places=3)
            for side in ("home_pts", "away_pts"):                    # avg: the two variants' points, side by side
                self.assertAlmostEqual(float(r["avg"][side]),
                                       (float(r["nb2+infer"][side]) + float(r["glmer+both"][side])) / 2, places=3)
                self.assertAlmostEqual(float(r["avg-plain"][side]),
                                       (float(r["nb2"][side]) + float(r["glmer"][side])) / 2, places=3)
            lay = r["nb2+infer"]                                      # a layered variant: its model moved by the layer
            self.assertAlmostEqual(float(lay["exp_margin"]), mus[code][0] - mus[code][1] + float(lay["layer_margin"]),
                                   places=3)
            for t in r.values():
                self.assertTrue(0 < float(t["p_home"]) < 1)
        self.assertEqual(tables["nb2"]["AF00000"]["burn_in"], "True")
        self.assertEqual(tables["nb2"]["AF00300"]["burn_in"], "False")
        self.assertEqual(tables["nb2"]["AF00000"]["score_params"], "0.1500 0.1500 0.4000")
        self.assertEqual(got, {"AF00100": (float(tables["avg"]["AF00100"]["home_pts"]),
                                           float(tables["avg"]["AF00100"]["away_pts"]))})
        for part in ("avg ", "nb2+both", "Every match", "--v21-prior-model"):
            self.assertIn(part, text)
        joined = "\n".join(first_run)
        for part in ("first estimate", "== NB2 fits: 24 days to fit", "== glmer fits", "24/24 days (100%)",
                     "finishing about", "the walkthrough took"):
            self.assertIn(part, joined)
        self.assertIn("NB2 fits: kept from before", "\n".join(lines))
        self.assertIn("glmer fits: kept from before", "\n".join(lines))
        self.assertEqual(set(manifest["seconds"]), {"nb2_fits", "nb2_prices", "glmer", "layers", "variants", "report"})


class TestTablePrior(unittest.TestCase):

    def test_a_table_read_as_a_prior_and_refused_without_points(self):
        from eAMFModel import table_prior, v21_report
        with tempfile.TemporaryDirectory() as d:
            good, bad = os.path.join(d, "t.csv"), os.path.join(d, "bad.csv")
            with open(good, "w") as fh:
                fh.write("MATCH_CODE,start,home_pts,away_pts\nM1,2026-09-01 10:00:00,20.5,17.25\nM2,,x,1\n")
            with open(bad, "w") as fh:
                fh.write("MATCH_CODE,points\nM1,3\n")
            self.assertTrue(table_prior.is_table(good))
            self.assertFalse(table_prior.is_table(d))
            pre = table_prior.TablePrior(good)
            self.assertEqual(pre.means([{"MATCH_CODE": "M1"}, {"MATCH_CODE": "M2"}]), {"M1": (20.5, 17.25)})
            self.assertFalse(pre.FOLLOWS_RESULTS)
            with self.assertRaises(SystemExit):
                table_prior.TablePrior(bad)
            rows = [{"MATCH_CODE": "M1", "SCHEDULED_START_TIME_UTC": "2026-09-01 10:00:00"}]
            means, books = v21_report.prior_inputs(good, rows, [])
        self.assertEqual(means, {"M1": (20.5, 17.25)})
        self.assertIn("M1", books)


if __name__ == "__main__":
    unittest.main()
