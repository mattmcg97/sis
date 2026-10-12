"""The margin uplift: the partner's dummy bets on every in-play tick, old feed against new."""

import datetime as dt
import os
import tempfile
import unittest
from unittest import mock

from .. import bets, config, margin_uplift as mu

KICK = dt.datetime(2026, 9, 1, 12)


def q(match, market, minutes, prob, desc="", active="true"):
    """A snowflake_io.fetch_quotes row `minutes` after kick-off (GAMEPLAI publishes 0-100)."""
    return (match, market, KICK + dt.timedelta(minutes=minutes), prob * 100, None, desc, 1, "OPEN", active)


STARTS = {"M1": KICK, "M2": KICK}
FINALS = {"M1": (24.0, 17.0), "M2": (20.0, 20.0)}
CHART = {k: 1.8 for k in range(1, 1000)}                    # every price 1.8: a stake of 125 wins 100


class TestChart(unittest.TestCase):

    def test_the_partners_chart_reads_numerator_over_denominator_plus_one(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chart.csv")
            with open(path, "w") as fh:
                fh.write("probability,numerator,denominator\n0.5,5,6\n0.545,8,13\nbad,1,2\n")
            chart = mu.load_chart(path)
        self.assertAlmostEqual(chart[500], 5 / 6 + 1)
        self.assertAlmostEqual(chart[545], 8 / 13 + 1)
        self.assertEqual(len(chart), 2)

    def test_without_a_chart_a_proportional_109_book(self):
        chart = mu.fallback_chart()
        self.assertAlmostEqual(chart[500], 1 / (0.5 * 1.09))
        self.assertEqual(chart[990], 1.01)


class TestTicks(unittest.TestCase):

    def test_only_live_in_play_moneyline_and_totals(self):
        rows = [q("M1", 50, -5, 0.6), q("M1", 50, 0.4, 0.6), q("M1", 50, 3, 0.6),
                q("M1", 51, 3, 0.4, active="false"), q("M1", 52, 3, 0.5, "PLAYER 1 -2.5"),
                q("M1", 54, 3, 0.5, "Over 40.5 points"), q("M3", 50, 3, 0.5)]
        t = mu.Ticks("x").add(rows, STARTS, FINALS)
        self.assertEqual(len(t), 2)
        self.assertEqual(list(t.array("group")), [0, 1])
        self.assertEqual(list(t.array("won")), [True, True])           # home won by 7; 41 is over 40.5

    def test_a_tie_loses_both_sides_and_a_push_both_totals(self):
        rows = [q("M2", 50, 3, 0.5), q("M2", 51, 3, 0.5), q("M2", 54, 3, 0.5, "Over 40"),
                q("M2", 55, 3, 0.5, "Under 40")]
        t = mu.Ticks("x").add(rows, STARTS, FINALS)
        self.assertEqual(list(t.array("won")), [False] * 4)

    def test_bands(self):
        self.assertEqual(mu._band(0, 0.1, 0), 0)                    # their pd.cut: (0, 0.1] is the first
        self.assertEqual(mu._band(0, 0.1001, 0), 1)
        self.assertEqual(mu._band(0, 1.0, 0), 9)
        self.assertEqual(mu.TOTAL_BANDS[mu._band(1, 0.5, 40.5)][1], "36-40.5")
        self.assertEqual(mu.TOTAL_BANDS[mu._band(1, 0.5, 70.5)][1], "56+")

    def test_the_clock_is_fitted_so_matches_end_at_sixty(self):
        rows = [q("M1", 50, 10, 0.6), q("M1", 50, 30, 0.6), q("M2", 50, 20, 0.5)]
        t = mu.Ticks("x").add(rows, STARTS, FINALS)
        self.assertAlmostEqual(t.mean_end(), 25.0)
        self.assertAlmostEqual(mu.factor_for(t), 60 / 25)


class TestGrade(unittest.TestCase):

    def test_the_partners_formulas(self):
        # minute 10 x 2 = 20 game minutes in: 40 left, Q2
        rows = [q("M1", 50, 10, 0.5), q("M1", 51, 10, 0.5)]
        cells = mu.grade(mu.Ticks("x").add(rows, STARTS, FINALS), 2.0, CHART)
        home, away = cells[("Moneyline", 40, "A", "0.4 - 0.5")], cells[("Moneyline", 40, "B", "0.4 - 0.5")]
        self.assertAlmostEqual(home["handle"], 125.0)                   # 100 / (1.8 - 1)
        self.assertAlmostEqual(home["expected"], 0.1 * 125)             # 1 - 1.8 / 2
        self.assertAlmostEqual(home["ggr"], 125 - 1.8 * 125)            # home won
        self.assertAlmostEqual(away["ggr"], 125.0)
        m, e = mu.margins(mu.overall(cells)["Moneyline"])
        self.assertAlmostEqual(m, 100 * (125 - 225 + 125) / 250)
        self.assertAlmostEqual(e, 10.0)

    def test_the_last_minutes_are_calibrated_but_not_bet(self):
        rows = [q("M1", 50, 28, 0.9), q("M1", 50, 10, 0.9995)]        # 4 left; fair 1.0005
        cells = mu.grade(mu.Ticks("x").add(rows, STARTS, FINALS), 2.0, CHART)
        late = cells[("Moneyline", 4, "A", "0.8 - 0.9")]
        self.assertEqual((late["n_cal"], late["n"], late["handle"]), (1, 0, 0))
        self.assertEqual(cells[("Moneyline", 40, "A", "0.9 - 1.0")]["n"], 0)   # fair price 1.001 or under

    def test_weighted_mae(self):
        rows = [q("M1", 50, 10, 0.6)] * 3 + [q("M2", 50, 10, 0.6)]     # 3 of 4 won at 0.6
        cells = mu.grade(mu.Ticks("x").add(rows, STARTS, FINALS), 2.0, CHART)
        w = mu.wmae(cells)
        self.assertEqual(w[("Moneyline", "A")][0], 4)
        self.assertAlmostEqual(w[("Moneyline", "A")][1], 15.0)
        self.assertAlmostEqual(w[("Moneyline", "A", 40)][1], 15.0)


class TestReport(unittest.TestCase):

    def data(self):
        old = mu.Ticks("old").add([q("M1", 50, 3, 0.5), q("M1", 50, 10, 0.5), q("M1", 50, 30, 0.5), q("M1", 54, 10, 0.5, "Over 40.5")],
                                  STARTS, FINALS)
        new = mu.Ticks("new").add([q("M1", 50, 3, 0.6), q("M1", 50, 10, 0.6), q("M1", 54, 10, 0.4, "Over 40.5")], STARTS, FINALS)
        return mu.compute(old, [("prod", old), ("v13", new)], CHART, chart_name="FD.csv",
                          window={"baseline": "2026-08-01 to 2026-09-01", "report": "2026-10-01 to 2026-10-12"})

    def test_one_clock_for_every_feed_and_the_uplift(self):
        data = self.data()
        self.assertAlmostEqual(data["factor"], 2.0)                    # the baseline's match ends at 30
        head = dict(mu.headline(data))
        self.assertEqual([round(m, 2) for m, _, _ in head["Moneyline"]], [-80.0, -80.0, -80.0])
        text = "\n".join(mu.text(data))
        self.assertIn("uplift on baseline", text)
        self.assertIn("FD.csv", text)
        self.assertIn("Q1, 54 game minutes left -- Moneyline", text)
        page = mu.html(data, str)
        self.assertIn("Uplift on baseline", page)
        self.assertIn("Q1, 54 game minutes left &middot; Moneyline", page)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "u.csv")
            mu.write_csv(path, data)
            with open(path) as fh:
                self.assertTrue(fh.readline().startswith("feed,market,remaining,quarter"))
                self.assertEqual(len(fh.readlines()), 9)

    def test_the_baseline_is_the_month_before(self):
        with mock.patch.object(config, "CUTOFF_START", "2026-10-01 00:00:00"):
            self.assertEqual(mu.baseline_window(None), (dt.datetime(2026, 9, 1), dt.datetime(2026, 10, 1)))
        self.assertEqual(mu.baseline_window("2026-07-01"), (dt.datetime(2026, 7, 1), dt.datetime(2026, 8, 1)))
        self.assertEqual(mu.baseline_window("2026-07-01", "2026-07-15")[1], dt.datetime(2026, 7, 15))

    def test_the_baseline_fetch_reads_prod_over_its_own_window(self):
        history = [{"MATCH_CODE": c, "SCHEDULED_START_TIME_UTC": f"{KICK:%Y-%m-%d %H:%M:%S}",
                    "PLAYER_1_FINAL_SCORE": "24", "PLAYER_2_FINAL_SCORE": "17"} for c in ("M1",)] + \
                  [{"MATCH_CODE": "LATE", "SCHEDULED_START_TIME_UTC": "2026-10-05 12:00:00",
                    "PLAYER_1_FINAL_SCORE": "1", "PLAYER_2_FINAL_SCORE": "2"}]
        seen = []

        def fetch(cur, stream, codes):
            seen.append((stream, list(codes), config.CUTOFF_START, config.CUTOFF_END))
            return [q("M1", 50, 10, 0.5)]
        with mock.patch.object(mu.snowflake_io, "fetch_quotes", side_effect=fetch):
            ticks, codes = mu.baseline_ticks(None, history, dt.datetime(2026, 9, 1), dt.datetime(2026, 10, 1))
        self.assertEqual(codes, ["M1"])
        self.assertEqual(seen[0][1:], (["M1"], "2026-09-01 00:00:00", "2026-10-01 01:00:00"))
        self.assertEqual(len(ticks), 1)
        self.assertNotEqual(config.CUTOFF_START, "2026-09-01 00:00:00")    # the window is put back


class TestPipeline(unittest.TestCase):

    def test_the_bets_run_hands_over_prod_and_each_candidates_ticks(self):
        from .. import snowflake_io
        from .test_bets import MONEYLINE, PRICES, _checks_for, at, odds_for
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER",
                "OPERATOR_NAME", "BET_IN_PLAY", "CUSTOMER_TEMPERATURE"]
        rows = [(i, "M1", at(c + 7), 1, 1, odds_for(p), 10, 10, None, 2, "FANDUEL", "Yes", "Standard")
                for i, (c, p) in enumerate(PRICES[1:])]
        history = [{"MATCH_CODE": "M1", "SCHEDULED_START_TIME_UTC": f"{at(0) - dt.timedelta(minutes=5):%Y-%m-%d %H:%M:%S}",
                    "PLAYER_1_FINAL_SCORE": "21", "PLAYER_2_FINAL_SCORE": "17"}]
        summary = {}
        with mock.patch.object(bets, "fetch_all", return_value=(cols, rows)), \
                mock.patch.object(snowflake_io, "fetch_quotes", return_value=MONEYLINE), \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 17)}), \
                mock.patch.object(snowflake_io, "fetch_history", return_value=history), \
                mock.patch.object(bets, "write_csv"), mock.patch.object(bets, "write_text"), \
                mock.patch("builtins.print"), mock.patch("os.makedirs"), \
                mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.object(config, "LAG_RANGE", (-20, 30)), mock.patch.object(config, "UPLIFT", True), \
                mock.patch.object(config, "CANDIDATES", ["CAND_TABLE"]), \
                mock.patch.object(mu, "run", return_value={"x": 1}) as run:
            bets.run(None, "out", summary=summary)
        feeds = run.call_args.args[2]
        self.assertEqual([name for name, _ in feeds], ["prod (report)", "CAND_TABLE"])
        self.assertGreater(len(feeds[0][1]), 0)
        self.assertIs(run.call_args.args[3], history)
        self.assertEqual(summary["uplift"], {"x": 1})


class TestSections(unittest.TestCase):

    def test_three_sections(self):
        data = TestReport().data()
        page = bets.html_section({"results": [("v13", [])], "uplift": data})
        for anchor in ('id="bets"', 'id="books"', 'id="uplift"'):
            self.assertIn(anchor, page)
        self.assertIn("Re-pricing", page)
        self.assertIn("Uplift on baseline", page)
        self.assertIn("--uplift was not run", bets.html_section({"results": [("v13", [])]}))

    def test_the_options(self):
        from .. import __main__ as cli
        args = cli.build_parser().parse_args(["report", "--bets", "--uplift-since", "2026-08-01",
                                              "--uplift-factor", "2.5"])
        with mock.patch.multiple(config, UPLIFT=False, UPLIFT_SINCE=None, UPLIFT_UNTIL=None,
                                 UPLIFT_CHART=None, UPLIFT_FACTOR=None):
            cli.apply_uplift(args, not args.no_uplift)
            self.assertEqual((config.UPLIFT, config.UPLIFT_SINCE, config.UPLIFT_FACTOR), (True, "2026-08-01", 2.5))
            args = cli.build_parser().parse_args(["bets", "--margin-chart", "/no/such.csv"])
            with self.assertRaises(SystemExit):
                cli.apply_uplift(args, args.uplift)


if __name__ == "__main__":
    unittest.main()
