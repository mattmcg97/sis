"""The post-game form layer: each gamer's day and session form against the pre-match model's prices."""

import datetime as dt
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from .. import __main__ as cli
from .. import follow, form_layer, nb2_prior, v10, v11, v12

DAY = dt.datetime(2026, 9, 1)
NOISE, TAU_D, TAU_S = (9.0, 12.0), (1.5, 2.0), (1.0, 1.3)


def match(code, minutes, p1, p2, finals=None):
    t = DAY + dt.timedelta(minutes=minutes)
    row = {"MATCH_CODE": code, "SCHEDULED_START_TIME_UTC": f"{t:%Y-%m-%d %H:%M:%S}",
           "PLAYER_1_HANDLE": p1, "PLAYER_2_HANDLE": p2, "PLAYER_1_TEAM": "A", "PLAYER_2_TEAM": "B",
           "STREAM_NUMBER": "1", "PLAYER_1_FINAL_SCORE": "", "PLAYER_2_FINAL_SCORE": ""}
    if finals:
        row["PLAYER_1_FINAL_SCORE"], row["PLAYER_2_FINAL_SCORE"] = str(finals[0]), str(finals[1])
    return row


class Flat:
    """A pre-match model that prices every match 20-20, and records what it was asked."""
    league = (17.0, 17.0)
    meta = {"before": DAY.isoformat()}

    def __init__(self):
        self.asked = []

    def describe(self):
        return "flat"

    def means(self, schedule, n_sims=None, results=None):
        self.asked.append([r["MATCH_CODE"] for r in schedule])
        return {r["MATCH_CODE"]: (20.0, 20.0) for r in schedule}


def at(minutes):
    return DAY + dt.timedelta(minutes=minutes)


class TestFilter(unittest.TestCase):

    def filt(self, rho=0.0):
        return form_layer.Filter(noise=NOISE, tau_day=TAU_D, rho=rho, tau_session=TAU_S)

    def test_a_fresh_gamer_has_no_form(self):
        f = self.filt()
        self.assertEqual(f.adjustment("ann", "bob", at(0)), (0.0, 0.0))
        self.assertEqual(f.form("ann", at(0)), ((0.0, 0.0), (0.0, 0.0)))

    def test_one_result_is_shared_between_both_gamers_by_how_unsure_they_are(self):
        f = self.filt()
        f.update("ann", "bob", at(0), 10.0, 0.0)            # ann beat the spread by 10
        var = 2 * (1.5 ** 2 + 1.0 ** 2) + 9.0 ** 2
        (ann_m, _), (bob_m, _) = f.form("ann", at(30)), f.form("bob", at(30))
        self.assertAlmostEqual(ann_m[0], 10 * 1.5 ** 2 / var)          # day form
        self.assertAlmostEqual(ann_m[1], 10 * 1.0 ** 2 / var)          # session form
        self.assertAlmostEqual(bob_m[0], -ann_m[0])
        self.assertAlmostEqual(bob_m[1], -ann_m[1])
        # a rematch: ann's form less bob's
        self.assertAlmostEqual(f.adjustment("ann", "bob", at(30))[0], 2 * 10 * (1.5 ** 2 + 1.0) / var)
        self.assertAlmostEqual(f.adjustment("bob", "ann", at(30))[0], -2 * 10 * (1.5 ** 2 + 1.0) / var)

    def test_totals_move_both_gamers_the_same_way(self):
        f = self.filt()
        f.update("ann", "bob", at(0), 0.0, 20.0)
        dm, dt_ = f.adjustment("ann", "bob", at(30))
        self.assertAlmostEqual(dm, 0.0)
        var = 2 * (2.0 ** 2 + 1.3 ** 2) + 12.0 ** 2
        self.assertAlmostEqual(dt_, 2 * 20 * (2.0 ** 2 + 1.3 ** 2) / var)

    def test_a_losing_run_drags_the_price_down_less_and_less_per_match(self):
        f = self.filt()
        steps, last = [], 0.0
        for k in range(9):                                   # nine matches, 8 short each
            f.update("ann", f"o{k}", at(36 * k), -8.0, 0.0)
            now = f.adjustment("ann", "new", at(36 * k + 30))[0]
            steps.append(now - last)
            last = now
        self.assertLess(last, -1.0)
        self.assertTrue(all(s < 0 for s in steps))
        self.assertTrue(all(b > a for a, b in zip(steps, steps[1:])))   # each step smaller

    def test_a_new_session_starts_session_form_afresh_and_keeps_the_day_s(self):
        f = self.filt()
        f.update("ann", "bob", at(0), 10.0, 0.0)
        (d, s), _ = f.form("ann", at(100))
        self.assertGreater(s, 0)
        (d2, s2), _ = f.form("ann", at(200))                  # more than 2 hours on
        self.assertEqual(s2, 0.0)
        self.assertAlmostEqual(d2, d)

    def test_the_next_day_s_form_is_carried_by_rho_a_day(self):
        for rho, days, share in ((0.0, 1, 0.0), (0.5, 1, 0.5), (0.5, 2, 0.25)):
            f = self.filt(rho)
            f.update("ann", "bob", at(0), 10.0, 0.0)
            (d, _), _ = f.form("ann", at(30))
            (d2, _), _ = f.form("ann", at(30 + 1440 * days))
            self.assertAlmostEqual(d2, share * d, msg=(rho, days))

    def test_no_spread_of_form_means_no_layer(self):
        f = form_layer.Filter(noise=NOISE, tau_day=(0.0, 0.0), rho=0.0, tau_session=(0.0, 0.0))
        f.update("ann", "bob", at(0), 30.0, 30.0)
        self.assertEqual(f.adjustment("ann", "bob", at(30)), (0.0, 0.0))


class TestFormLayer(unittest.TestCase):

    def layer(self, since=DAY):
        return form_layer.FormLayer(Flat(), since, noise=NOISE, tau_day=TAU_D, rho=0.0, tau_session=TAU_S)

    def test_only_matches_that_started_before_count(self):
        results = [match(f"L{k}", 36 * k, "ann", f"o{k}", (10, 30)) for k in range(5)]   # ann -20 a match
        pre = self.layer()
        rows = [match("L2", 72, "ann", "o2"), match("NEXT", 400, "ann", "bob"), match("FIRST", -10, "ann", "bob")]
        got = pre.means(rows, results=results)
        self.assertEqual(got["FIRST"], (20.0, 20.0))                       # before any result
        self.assertLess(got["NEXT"][0] - got["NEXT"][1], got["L2"][0] - got["L2"][1])
        self.assertLess(got["L2"][0] - got["L2"][1], 0)                    # L2 sees L0 and L1
        f = form_layer.Filter(noise=NOISE, tau_day=TAU_D, tau_session=TAU_S)
        for k in range(2):
            f.update("ANN", f"O{k}", at(36 * k), -20.0, 0.0)
        dm, dt_ = f.adjustment("ANN", "O2", at(72))
        self.assertAlmostEqual(got["L2"][0], 20 + (dt_ + dm) / 2)
        self.assertAlmostEqual(got["L2"][1], 20 + (dt_ - dm) / 2)
        # the settled matches were priced alongside the schedule, once
        self.assertEqual(sorted(pre.pre.asked[0]), ["FIRST", "L0", "L1", "L2", "L3", "L4", "NEXT"])

    def test_a_match_does_not_see_one_settling_as_it_starts(self):
        results = [match("SAME", 60, "ann", "x", (40, 0))]
        got = self.layer().means([match("N", 60, "ann", "bob")], results=results)
        self.assertEqual(got["N"], (20.0, 20.0))

    def test_results_before_the_model_s_out_of_sample_start_are_not_read(self):
        results = [match("OLD", -30, "ann", "x", (40, 0))]                 # before the fit's cut-off
        got = self.layer().means([match("N", 60, "ann", "bob")], results=results)
        self.assertEqual(got["N"], (20.0, 20.0))
        got = self.layer(since=DAY - dt.timedelta(hours=1)).means([match("N", 60, "ann", "bob")],
                                                                  results=results)
        self.assertGreater(got["N"][0], got["N"][1])

    def test_no_results_is_the_model_as_it_was(self):
        pre = self.layer()
        self.assertEqual(pre.means([match("N", 10, "a", "b")]), {"N": (20.0, 20.0)})
        self.assertEqual(pre.league, (17.0, 17.0))                          # the rest is the model's own
        self.assertTrue(pre.FOLLOWS_RESULTS and pre.AS_OF)
        self.assertIn("post-game form layer from 2026-09-01", pre.describe())
        self.assertIn("fresh each day", pre.describe())

    def test_a_carried_day_form_reads_back_the_lookback(self):
        results = [match("Y", -1440 * 3, "ann", "x", (40, 0))]              # three days before
        carry = form_layer.FormLayer(Flat(), DAY - dt.timedelta(days=10), noise=NOISE, tau_day=TAU_D,
                                     rho=0.9, tau_session=TAU_S)
        got = carry.means([match("N", 60, "ann", "bob")], results=results)
        self.assertGreater(got["N"][0], got["N"][1])
        self.assertEqual(carry.lookback_days, form_layer.LOOKBACK_DAYS)


class TestSwitch(unittest.TestCase):

    def test_off_unless_switched_on(self):
        flat = Flat()
        self.assertIs(form_layer.wrap(flat, {}), flat)
        self.assertIsInstance(form_layer.wrap(flat, {"form_layer": True}), form_layer.FormLayer)
        with mock.patch.object(form_layer, "ON", True):
            self.assertIsInstance(form_layer.wrap(flat, {}), form_layer.FormLayer)
            self.assertIs(form_layer.wrap(flat, {"form_layer": False}), flat)
        self.assertIsNone(form_layer.wrap(None, {"form_layer": True}))

    def test_out_of_sample_from_the_first_daily_fit_or_the_cut_off(self):
        rolling = mock.Mock(spec=["days"], days=[dt.date(2026, 9, 3), dt.date(2026, 9, 2)])
        self.assertEqual(form_layer.out_of_sample_since(rolling), dt.datetime(2026, 9, 2))
        self.assertEqual(form_layer.out_of_sample_since(Flat()), DAY)
        self.assertEqual(form_layer.out_of_sample_since(follow.Following(Flat(), DAY + dt.timedelta(hours=5))),
                         DAY + dt.timedelta(hours=5))
        no_cut = Flat()
        no_cut.meta = {}
        self.assertIsNone(form_layer.out_of_sample_since(no_cut))
        self.assertIs(form_layer.wrap(no_cut, {"form_layer": True}), no_cut)

    def test_each_version_prices_with_it_when_the_build_says_so(self):
        for version in (v10, v11, v12):
            with tempfile.TemporaryDirectory() as d:
                os.makedirs(os.path.join(d, "nb2"))
                path = os.path.join(d, version.PRIOR_FILE)
                with open(path, "w") as fh:
                    json.dump({"prior": "nb2"}, fh)
                with mock.patch.object(nb2_prior.Prematch, "exists", return_value=True), \
                        mock.patch.object(nb2_prior.Prematch, "__init__", return_value=None), \
                        mock.patch.object(nb2_prior.Prematch, "meta", {"before": DAY.isoformat()}, create=True):
                    pre = version.prematch_model(d)
                    self.assertIsInstance(pre, follow.Following, version.__name__)    # as before
                    self.assertTrue(form_layer.switched_on(form_layer.switch(d, True)))
                    pre = version.prematch_model(d)
                    self.assertIsInstance(pre, form_layer.FormLayer, version.__name__)
                    self.assertIsInstance(pre.pre, follow.Following)                  # on top of it
                    self.assertIsInstance(form_layer.base_model(d), follow.Following)
                    with mock.patch("eAMFModel.rolling_prior.Rolling") as rolling:
                        rolling.return_value = mock.Mock(spec=["days", "means"], days=[dt.date(2026, 9, 2)])
                        with open(path) as fh:
                            meta = json.load(fh)
                        with open(path, "w") as fh:
                            json.dump(dict(meta, rolling=d), fh)
                        pre = version.prematch_model(d)
                        self.assertIsInstance(pre, form_layer.FormLayer, version.__name__)
                        self.assertIs(pre.pre, rolling.return_value)                # no follow layer
                        self.assertEqual(pre.since, dt.datetime(2026, 9, 2))
                    with open(path, "w") as fh:
                        json.dump(meta, fh)                                         # its own fit again
                    self.assertFalse(form_layer.switched_on(form_layer.switch(d, False)))
                    self.assertNotIsInstance(version.prematch_model(d), form_layer.FormLayer)
                    self.assertFalse(form_layer.switched_on(form_layer.switch(d, None)))
                    with open(path) as fh:
                        self.assertNotIn("form_layer", json.load(fh))

    def test_a_build_can_keep_settings_of_its_own(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "v12prior.json"), "w") as fh:
                json.dump({"prior": "nb2"}, fh)
            meta = form_layer.switch(d, True, tau_day=(2.0, 1.0), rho=0.5)
            self.assertEqual(meta["form_layer"], {"tau_day": [2.0, 1.0], "rho": 0.5})
            self.assertTrue(form_layer.switched_on(meta))
            self.assertEqual(form_layer.base_settings(d), {"tau_day": (2.0, 1.0), "rho": 0.5})
            layer = form_layer.wrap(Flat(), meta)
            self.assertEqual(layer.settings["tau_day"], (2.0, 1.0))
            self.assertEqual(layer.settings["rho"], 0.5)
            self.assertEqual(layer.settings["tau_session"], form_layer.TAU_SESSION)
            self.assertEqual(layer.lookback_days, form_layer.LOOKBACK_DAYS)
            with self.assertRaises(ValueError):
                form_layer.switch(d, False, rho=0.5)
            with self.assertRaises(ValueError):
                form_layer.switch(d, True, noise=(1, 1))
            out = io.StringIO()
            with redirect_stdout(out):
                cli.main(["form-layer", d, "--on", "--tau-session", "1.5,1", "--rho", "0.2"])
            self.assertIn("with tau_session (1.5, 1.0), rho 0.2", out.getvalue())
            with self.assertRaises(SystemExit):
                cli.main(["form-layer", d, "--off", "--rho", "0.2"])

    def test_the_command_switches_builds(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "v12prior.json"), "w") as fh:
                json.dump({"prior": "glmer"}, fh)
            out = io.StringIO()
            with redirect_stdout(out):
                cli.main(["form-layer", d, "--on"])
            self.assertIn("post-game form layer on", out.getvalue())
            with open(os.path.join(d, "v12prior.json")) as fh:
                self.assertEqual(json.load(fh), {"prior": "glmer", "form_layer": True})
            with redirect_stdout(io.StringIO()):
                cli.main(["form-layer", d, "--default"])
            with open(os.path.join(d, "v12prior.json")) as fh:
                self.assertEqual(json.load(fh), {"prior": "glmer"})
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            form_layer.switch(d, True)


class TestTrace(unittest.TestCase):

    def test_one_gamer_s_matches_from_their_side(self):
        history = [match("H0", 0, "ann", "bob", (10, 30)),       # ann at home, lost by 20
                   match("A1", 40, "cat", "ann", (35, 14)),      # ann away, lost by 21
                   match("H2", 80, "ann", "dan", (21, 20)),
                   match("X", 50, "bob", "cat", (20, 20))]
        layer = form_layer.FormLayer(Flat(), DAY, noise=NOISE, tau_day=TAU_D, tau_session=TAU_S)
        rows = form_layer.trace(layer, history, "Ann", DAY.date(), DAY.date())
        self.assertEqual([r["code"] for r in rows], ["H0", "A1", "H2"])
        self.assertEqual([r["side"] for r in rows], ["home", "away", "home"])
        self.assertEqual([(r["scored"], r["conceded"]) for r in rows], [(10, 30), (14, 35), (21, 20)])
        self.assertEqual(rows[0]["adjust"], 0.0)
        self.assertLess(rows[1]["adjust"], 0.0)                   # after a loss, from ann's side
        self.assertLess(rows[1]["day"], 0.0)
        self.assertEqual(rows[1]["opponent_form"], 0.0)           # cat's first match
        self.assertLess(rows[2]["adjust"], rows[1]["adjust"])
        self.assertAlmostEqual(rows[2]["margin"], rows[2]["base_margin"] + rows[2]["adjust"])
        lines = form_layer.report(rows, "ann", layer)
        self.assertIn("ANN: 3 matches", lines[0])
        self.assertTrue(any("2026-09-01 00:40 away CAT" in line for line in lines))
        self.assertIn("moneyline log loss", lines[-1])
        self.assertEqual(form_layer.trace(layer, history, "nobody", DAY.date(), DAY.date()), [])


if __name__ == "__main__":
    unittest.main()
