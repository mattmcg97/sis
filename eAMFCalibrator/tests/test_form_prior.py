import contextlib
import csv
import datetime as dt
import importlib
import io
import os
import tempfile
import unittest
from unittest import mock

from eAMFCalibrator import config, form_prior as fp, score_model

T0 = dt.datetime(2026, 9, 1, 8, 0)
HISTORY_FIELDS = ["MATCH_CODE", "SPORT_CODE", "SCHEDULED_START_TIME_UTC", "PLAYER_1_HANDLE", "PLAYER_1_TEAM",
                  "PLAYER_2_HANDLE", "PLAYER_2_TEAM", "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"]
TABLE_FIELDS = ["MATCH_CODE", "start", "home", "away", "home_pts", "away_pts", "Pred_P1_Points", "Pred_P2_Points",
                "exp_margin", "p_home", "score_model", "score_params"]


def _history(n=60):
    """ACE and BOB level for a month, then ACE winning every match by 14; the last match unplayed."""
    rows = []
    for i in range(n):
        t = T0 + dt.timedelta(hours=12 * i)
        home, away = ("ACE", "BOB") if i % 2 == 0 else ("BOB", "ACE")
        if i == n - 1:
            hs = as_ = ""
        else:
            ace = 24 if i >= n // 2 else 20 + (3 if i % 4 < 2 else -3)
            bob = 10 if i >= n // 2 else 20
            hs, as_ = (ace, bob) if home == "ACE" else (bob, ace)
        rows.append(dict(MATCH_CODE=f"AF{i:03d}", SPORT_CODE="AF", SCHEDULED_START_TIME_UTC=f"{t:%Y-%m-%d %H:%M:%S}",
                         PLAYER_1_HANDLE=home, PLAYER_1_TEAM="Bears", PLAYER_2_HANDLE=away, PLAYER_2_TEAM="Bears",
                         PLAYER_1_FINAL_SCORE=hs, PLAYER_2_FINAL_SCORE=as_))
    return rows


def _table(history):
    """A table that has every match level, 20 points a side."""
    return [dict(MATCH_CODE=r["MATCH_CODE"], start=r["SCHEDULED_START_TIME_UTC"], home=r["PLAYER_1_HANDLE"],
                 away=r["PLAYER_2_HANDLE"], home_pts=20.0, away_pts=20.0, Pred_P1_Points=20.0, Pred_P2_Points=20.0,
                 exp_margin=0.0, p_home=0.5, score_model="normal", score_params="") for r in history]


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)


class TestFormPrior(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.history = os.path.join(self.dir.name, "history.csv")
        self.table = os.path.join(self.dir.name, "avg.csv")
        rows = _history()
        _write(self.history, rows, HISTORY_FIELDS)
        _write(self.table, _table(rows), TABLE_FIELDS)

    def tearDown(self):
        self.dir.cleanup()

    def ace_margin(self, view, code):
        """ACE's expected margin in a match, whichever side he is."""
        i = int(code[2:])
        return view[code][0] if i % 2 == 0 else -view[code][0]

    def test_each_match_is_priced_off_the_results_before_it(self):
        matches = fp.read_matches(self.history)
        self.assertEqual(len(matches), 60)
        self.assertIsNone(matches[-1][6])
        view = fp.rating(matches, 2.0)
        self.assertEqual(view["AF000"][0], 0.0)                     # nothing known yet
        self.assertLess(abs(self.ace_margin(view, "AF029")), 2.0)    # a level month
        self.assertGreater(self.ace_margin(view, "AF059"), 8.0)      # a month of 14-point wins
        self.assertIn("AF059", view)                                 # the unplayed match is priced too

    def test_a_faster_drift_follows_the_latest_results_further(self):
        matches = fp.read_matches(self.history)
        slow, fast = fp.rating(matches, 0.01), fp.rating(matches, 4.0)
        self.assertGreater(self.ace_margin(fast, "AF035"), self.ace_margin(slow, "AF035") + 1.0)

    def test_a_table_moves_part_of_the_way_and_keeps_its_total(self):
        rows, cols = fp.read_table(self.table)
        view = {"AF000": (8.0, 15.0)}
        moved, mean_move = fp.moved_rows(rows, cols, view, 0.5)
        r = moved[0]
        self.assertAlmostEqual(r["home_pts"] - r["away_pts"], 4.0)
        self.assertAlmostEqual(r["home_pts"] + r["away_pts"], 40.0)
        self.assertEqual((r["Pred_P1_Points"], r["Pred_P2_Points"], r["exp_margin"]), (22.0, 18.0, 4.0))
        self.assertAlmostEqual(r["p_home"], round(score_model.Normal(4.0, 40.0).p_home(), 5))
        self.assertEqual(moved[1], rows[1])                          # a match the rating has no view of
        self.assertAlmostEqual(mean_move, 4.0)

    def test_the_command_writes_a_table_for_every_drift_and_weight(self):
        from eAMFCalibrator import __main__ as cli
        out = os.path.join(self.dir.name, "out")
        bets = os.path.join(self.dir.name, "bets_sim.csv")
        _write(bets, [dict(match_code="AF001", stake=100), dict(match_code="AF059", stake=50),
                      dict(match_code="AF999", stake=25)], ["match_code", "stake"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(["form-prior", "--history", self.history, "--table", f"avg={self.table}",
                                       "--drift", "1,4", "--weight", "0.5,1", "--draws", "500", "--bets", bets,
                                       "--out", out]), 0)
        self.assertEqual(sorted(os.listdir(os.path.join(out, "priors"))),
                         ["avg_form1_w05.csv", "avg_form1_w10.csv", "avg_form4_w05.csv", "avg_form4_w10.csv"])
        from eAMFModel import table_prior
        prior = table_prior.TablePrior(os.path.join(out, "priors", "avg_form4_w10.csv"))
        h, a = prior.points["AF059"]
        self.assertGreater(a - h, 8.0)                               # ACE away in the last match
        self.assertAlmostEqual(h + a, 40.0, places=3)
        with open(os.path.join(out, "form_prior.txt"), encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("avg moved 0.5 of the way to the rating drifting 1/day", text)
        self.assertIn("the history has 1 of the 3 bet matches (100 of 175 stake); missing 2", text)   # AF059 unplayed
        self.assertIn("the avg table has 2 of the 3 bet matches (150 of 175 stake); missing 1", text)


class TestDefault(unittest.TestCase):
    def test_with_no_options_it_writes_v21_s_default_prior(self):
        """The history and the walkthrough's avg table from out/, drift 4, half way: the table v2.1 reads
        when no prior is named."""
        from eAMFCalibrator import __main__ as cli
        from eAMFModel import v21_stream
        with tempfile.TemporaryDirectory() as d:
            rows = _history()
            _write(os.path.join(d, "match_history.csv"), rows, HISTORY_FIELDS)
            os.makedirs(os.path.join(d, "walkthrough", "priors"))
            _write(os.path.join(d, "walkthrough", "priors", "avg.csv"), _table(rows), TABLE_FIELDS)
            with mock.patch.object(cli, "DEFAULT_OUT", d), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["form-prior", "--draws", "500"]), 0)
            self.assertEqual(os.listdir(os.path.join(d, "form_prior", "priors")), ["avg_form4_w05.csv"])
            self.assertTrue(v21_stream.DEFAULT_PRIOR_TABLE.endswith(
                os.path.join("out", "form_prior", "priors", "avg_form4_w05.csv")))
            with mock.patch.object(cli, "DEFAULT_OUT", os.path.join(d, "nowhere")), self.assertRaises(SystemExit):
                cli.main(["form-prior"])


class TestPriorFor(unittest.TestCase):
    """--prior-for: each v21 candidate of a betting-sim run nudged toward its own prior."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.prior = os.path.join(self.dir.name, "avg_form2_w05.csv")
        _write(self.prior, _table(_history(4)), TABLE_FIELDS)
        self.saved = (list(config.CANDIDATES), dict(config.STREAMS), dict(config.MODEL_PRIORS))

    def tearDown(self):
        config.CANDIDATES = self.saved[0]
        config.STREAMS.clear()
        config.STREAMS.update(self.saved[1])
        config.MODEL_PRIORS.clear()
        config.MODEL_PRIORS.update(self.saved[2])
        self.dir.cleanup()

    def parse(self, *argv):
        from eAMFCalibrator.__main__ import build_parser, apply_overrides
        with contextlib.redirect_stdout(io.StringIO()):
            apply_overrides(build_parser().parse_args(["bets", "run", *argv]))

    def test_each_candidate_reads_its_own_prior(self):
        from eAMFCalibrator import snowflake_io as sio, bets
        self.parse("--candidate", "v21,v21-avgf2", "--prior-for", f"v21-avgf2={self.prior}")
        self.assertEqual(config.CANDIDATES, ["MODEL:v21", "MODEL:v21-avgf2"])
        self.assertEqual(config.MODEL_PRIORS, {"v21-avgf2": self.prior})
        stream = importlib.import_module("eAMFModel.v21_stream")
        seen = []

        def fake_paths(model_dir, prior_dir=None):
            seen.append((model_dir, prior_dir))
            raise RuntimeError("stop")
        with mock.patch.object(config, "V21_PRIOR_MODEL_DIR", "v14_prior"), \
                mock.patch.object(config, "V21_MODEL_DIR", "v21_t10"), \
                mock.patch.object(sio, "_need_numpy"), \
                mock.patch.object(stream, "model_paths", side_effect=fake_paths), \
                contextlib.redirect_stdout(io.StringIO()):
            for name in ("MODEL:v21-avgf2", "MODEL:v21"):
                with self.assertRaises(RuntimeError):
                    sio._model_quotes(None, name, ["M1"])
            with self.assertRaises(RuntimeError):
                bets.check_models(["MODEL:v21-avgf2"])
        self.assertEqual(seen, [("v21_t10", self.prior), ("v21_t10", "v14_prior"), ("v21_t10", self.prior)])

    def test_a_prior_must_name_a_v21_candidate_and_exist(self):
        for argv in (["--candidate", "v21", "--prior-for", f"v21-avgf2={self.prior}"],
                     ["--candidate", "v21,v21-avgf2", "--prior-for", "v21-avgf2=no/such.csv"],
                     ["--candidate", "v9,v21-avgf2", "--prior-for", f"v9={self.prior}"],
                     ["--candidate", "v21,v21-avgf2", "--prior-for", "v21-avgf2"]):
            with self.assertRaises(SystemExit):
                self.parse(*argv)


if __name__ == "__main__":
    unittest.main()
