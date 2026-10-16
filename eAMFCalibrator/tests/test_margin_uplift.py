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
        prod = mu.Ticks("prod").add([q("M1", 50, 3, 0.5), q("M1", 50, 10, 0.5), q("M1", 50, 30, 0.5),
                                     q("M1", 54, 10, 0.5, "Over 40.5"), q("M2", 50, 60, 0.5)], STARTS, FINALS)
        cand = mu.Ticks("v13").add([q("M1", 50, 3, 0.6), q("M1", 50, 10, 0.6), q("M1", 54, 10, 0.4, "Over 40.5")],
                                   STARTS, FINALS)
        return mu.compute([("prod", prod), ("v13", cand)], CHART, chart_name="FD.csv",
                          window="2026-10-01 to 2026-10-12")

    def test_the_same_matches_and_one_clock_for_every_feed(self):
        data = self.data()
        self.assertEqual(data["matches"], 1)                           # M2 has no candidate ticks
        self.assertEqual([f["ticks"] for f in data["feeds"]], [3, 3])  # prod's at 30: v13's last is 20 old
        self.assertEqual([f["of"] for f in data["feeds"]], [4, 3])
        self.assertAlmostEqual(data["factor"], 2.0)                    # prod's M1 ends at 30, M2 is out
        head = dict(mu.headline(data))
        self.assertEqual([round(m, 2) for m, _, _ in head["Moneyline"]], [-80.0, -80.0])
        text = "\n".join(mu.text(data))
        self.assertIn("uplift on prod", text)
        self.assertIn("FD.csv", text)
        self.assertIn("on 1 matches: prod's ticks", text)
        self.assertIn("Q1, 54 game minutes left -- Moneyline", text)
        page = mu.html(data, str)
        self.assertIn("Uplift on prod", page)
        self.assertIn("Q1, 54 game minutes left &middot; Moneyline", page)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "u.csv")
            mu.write_csv(path, data)
            with open(path) as fh:
                self.assertTrue(fh.readline().startswith("feed,market,remaining,quarter"))
                self.assertEqual(len(fh.readlines()), 6)

    def test_no_match_in_every_feed(self):
        prod = mu.Ticks("prod").add([q("M1", 50, 3, 0.5)], STARTS, FINALS)
        cand = mu.Ticks("v13").add([q("M2", 50, 3, 0.5)], STARTS, FINALS)
        self.assertIsNone(mu.compute([("prod", prod), ("v13", cand)], CHART))
        with mock.patch("builtins.print"):
            self.assertIsNone(mu.run("out", [("prod", prod), ("v13", cand)]))


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
                mock.patch.object(snowflake_io, "fetch_history", return_value=history) as fh, \
                mock.patch.object(bets, "write_csv"), mock.patch.object(bets, "write_text"), \
                mock.patch("builtins.print"), mock.patch("os.makedirs"), \
                mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.object(config, "LAG_RANGE", (-20, 30)), \
                mock.patch.object(config, "CANDIDATES", ["CAND_TABLE"]), \
                mock.patch.object(mu, "run", return_value={"x": 1}) as run:
            bets.run(None, "out", summary=summary)
        feeds = run.call_args.args[1]
        self.assertEqual(set(run.call_args.kwargs["names"]), {"M1"})
        self.assertEqual([name for name, _ in feeds], ["prod", "CAND_TABLE"])
        self.assertEqual([len(t) for _, t in feeds], [len(MONEYLINE)] * 2)
        self.assertEqual(fh.call_count, 1)                              # one history read serves both
        self.assertEqual(summary["uplift"], {"x": 1})


class TestUpliftFailure(unittest.TestCase):

    def test_the_bets_run_reports_without_it(self):
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
                mock.patch("builtins.print"), mock.patch("os.makedirs"), mock.patch("traceback.print_exc"), \
                mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.object(config, "LAG_RANGE", (-20, 30)), \
                mock.patch.object(config, "CANDIDATES", ["CAND_TABLE"]), \
                mock.patch.object(mu, "run", side_effect=ValueError("bad chart row")):
            bets.run(None, "out", summary=summary)
        self.assertIsNone(summary["uplift"])
        self.assertIn("results", summary)


class TestSections(unittest.TestCase):

    def test_three_sections(self):
        data = TestReport().data()
        page = bets.html_section({"results": [("v13", [])], "uplift": data})
        for anchor in ('id="bets"', 'id="books"', 'id="uplift"'):
            self.assertIn(anchor, page)
        self.assertIn("Re-pricing", page)
        self.assertIn("Uplift on prod", page)
        self.assertIn("No margin uplift", bets.html_section({"results": [("v13", [])]}))

    def test_the_options(self):
        from .. import __main__ as cli
        args = cli.build_parser().parse_args(["report", "--bets", "--uplift-factor", "2.5"])
        with mock.patch.multiple(config, UPLIFT_CHART=None, UPLIFT_FACTOR=None):
            cli.apply_uplift(args)
            self.assertEqual((config.UPLIFT_CHART, config.UPLIFT_FACTOR), (None, 2.5))
            args = cli.build_parser().parse_args(["bets", "--margin-chart", "/no/such.csv"])
            with self.assertRaises(SystemExit):
                cli.apply_uplift(args)
            with tempfile.TemporaryDirectory() as d:
                bad = os.path.join(d, "bad.csv")
                with open(bad, "w") as fh:
                    fh.write("probability,price\n0.5,1.9\n")
                with self.assertRaises(SystemExit) as caught:         # read at once, not after the run
                    cli.apply_uplift(cli.build_parser().parse_args(["bets", "--margin-chart", bad]))
                self.assertIn("numerator and denominator", str(caught.exception))
                good = os.path.join(d, "good.csv")
                with open(good, "w", encoding="cp1252") as fh:
                    fh.write("Probability,Numerator,Denominator,Note\n50%,10,11,\xe9\n")
                with mock.patch("builtins.print"):
                    cli.apply_uplift(cli.build_parser().parse_args(["bets", "--margin-chart", good]))
                self.assertEqual(mu.load_chart(good), {500: 1.0 + 10 / 11})
                bare = os.path.join(d, "FD_109_BALANCED.csv")          # theirs: no header
                with open(bare, "w") as fh:
                    fh.write("0.001,25000,100\n0.5,10,11\n")
                self.assertEqual(mu.load_chart(bare), {1: 251.0, 500: 1.0 + 10 / 11})


class TestMatchedTicks(unittest.TestCase):
    """Only prod's ticks every candidate has a fresh price for, with every probability 0.05-0.95."""

    def test_the_candidate_is_read_at_prods_moments(self):
        prod = mu.Ticks("prod").add([q("M1", 50, 2, 0.60), q("M1", 50, 3, 0.62), q("M1", 50, 5, 0.65),
                                     q("M1", 50, 9, 0.70), q("M1", 54, 4, 0.50, "Over 40.5 points"),
                                     q("M2", 50, 4, 0.50)], STARTS, FINALS)
        cand = mu.Ticks("v14").add([q("M1", 50, 2.5, 0.55), q("M1", 50, 4.5, 0.58),
                                    q("M1", 54, 3.5, 0.45, "Over 39.5 points")], STARTS, FINALS)
        (_, p), (_, c) = mu.matched([("prod", prod), ("v14", cand)])
        # 2.0: no candidate price yet; 9.0: the candidate's last is 4.5 minutes old; M2: not priced
        self.assertEqual(list(p.array("minutes")), [3.0, 5.0, 4.0])
        self.assertEqual(list(c.array("minutes")), [3.0, 5.0, 4.0])
        self.assertEqual([round(x, 2) for x in p.array("prob")], [0.62, 0.65, 0.50])
        self.assertEqual([round(x, 2) for x in c.array("prob")], [0.55, 0.58, 0.45])
        self.assertEqual(list(c.array("group")), [0, 0, 1])
        self.assertEqual(set(p.matches), {"M1"})

    def test_every_feeds_probability_within_the_range(self):
        prod = mu.Ticks("prod").add([q("M1", 50, 2, 0.04), q("M1", 50, 3, 0.50), q("M1", 50, 4, 0.95),
                                     q("M1", 50, 5, 0.50)], STARTS, FINALS)
        cand = mu.Ticks("v14").add([q("M1", 50, 2, 0.50), q("M1", 50, 3, 0.97), q("M1", 50, 4, 0.90),
                                    q("M1", 50, 5, 0.05)], STARTS, FINALS)
        (_, p), (_, c) = mu.matched([("prod", prod), ("v14", cand)])
        self.assertEqual(list(p.array("minutes")), [4.0, 5.0])      # 2: prod 0.04; 3: v14 0.97
        self.assertEqual([round(x, 2) for x in c.array("prob")], [0.90, 0.05])

    def test_every_candidate_needs_a_price(self):
        prod = mu.Ticks("prod").add([q("M1", 50, k, 0.6) for k in (2, 3, 4)], STARTS, FINALS)
        a = mu.Ticks("v13").add([q("M1", 50, 2, 0.7)], STARTS, FINALS)
        b = mu.Ticks("v14").add([q("M1", 50, 3, 0.7)], STARTS, FINALS)
        out = mu.matched([("prod", prod), ("v13", a), ("v14", b)])
        self.assertEqual([len(t) for _, t in out], [2, 2, 2])     # 2: no v14 price yet

    def test_compute_bets_every_feed_on_the_same_ticks(self):
        prod = mu.Ticks("prod").add([q("M1", 50, k, 0.6) for k in (2, 3, 4, 5)], STARTS, FINALS)
        cand = mu.Ticks("v14").add([q("M1", 50, 2, 0.7)], STARTS, FINALS)
        data = mu.compute([("prod", prod), ("v14", cand)], chart=CHART)
        self.assertEqual([(f["ticks"], f["of"]) for f in data["feeds"]], [(3, 4), (3, 1)])  # 5: over 2 old
        handles = [o["Moneyline"]["handle"] for o in data["overall"]]
        self.assertEqual(handles[0], handles[1])          # the same bets, at the same chart price
        self.assertIn("0.05-0.95", "\n".join(mu.text(data)))
        self.assertIn("matched", mu.html(data, lambda x: x))

    def test_nothing_matched(self):
        prod = mu.Ticks("prod").add([q("M1", 50, 2, 0.6)], STARTS, FINALS)
        cand = mu.Ticks("v14").add([q("M1", 50, 3, 0.6)], STARTS, FINALS)
        self.assertIsNone(mu.compute([("prod", prod), ("v14", cand)], chart=CHART))


class TestByMatch(unittest.TestCase):
    """Every match's dummy bets and its part of the uplift, with a proper score on the same ticks."""

    def data(self):
        starts = dict(STARTS, M3=KICK)
        finals = dict(FINALS, M3=(10.0, 13.0))                        # M1 home won, M3 away won
        prod = mu.Ticks("prod").add([q("M1", 50, 5, 0.80), q("M1", 51, 5, 0.20), q("M1", 50, 8, 0.90),
                                     q("M3", 50, 5, 0.85), q("M3", 51, 5, 0.15), q("M3", 50, 8, 0.92),
                                     q("M3", 54, 6, 0.55, "Over 20.5"), q("M1", 50, 29, 0.95)], starts, finals)
        cand = mu.Ticks("v14").add([q("M1", 50, 5, 0.75), q("M1", 51, 5, 0.25), q("M1", 50, 8, 0.85),
                                    q("M3", 50, 5, 0.70), q("M3", 51, 5, 0.30), q("M3", 50, 8, 0.80),
                                    q("M3", 54, 6, 0.50, "Over 20.5"), q("M1", 50, 29, 0.95)], starts, finals)
        return mu.compute([("prod", prod), ("v14", cand)], factor=2.0, window="2026-10-01 to 2026-10-12")

    def test_the_parts_add_up_to_the_uplift(self):
        data = self.data()
        rows = {r["code"]: r for r in data["by_match"]}
        self.assertEqual(set(rows), {"M1", "M3"})
        self.assertEqual(rows["M3"]["final"], (10.0, 13.0))
        m1 = rows["M1"]["feeds"][0]["Moneyline"]
        self.assertEqual((m1["n"], m1["scored"]), (3, 3))             # minute 29: 2 game minutes left
        self.assertAlmostEqual(m1["brier"], 0.2 ** 2 + 0.2 ** 2 + 0.1 ** 2)
        self.assertAlmostEqual(rows["M1"]["feeds"][0]["Moneyline"]["top_loser"], 0.20)
        self.assertAlmostEqual(rows["M3"]["feeds"][0]["Moneyline"]["top_loser"], 0.92)
        self.assertAlmostEqual(rows["M3"]["feeds"][1]["Moneyline"]["top_loser"], 0.80)
        for g in mu.GROUPS:
            ov = [o.get(g) for o in data["overall"]]
            if not ov[0]:
                continue
            uplift = mu.margins(ov[1])[0] - mu.margins(ov[0])[0]
            part = mu.parts(data["by_match"], 2)[(1, g)]
            self.assertAlmostEqual(sum(part.values()), uplift)
            tot = mu.match_totals(data["by_match"], 2)
            self.assertAlmostEqual(tot[0][g]["ggr"], ov[0]["ggr"])
            self.assertAlmostEqual(mu.uplift_of(tot, 1, g), uplift)
        # prod surer of M3's home side, which lost: a big stake kept, so prod's margin is higher
        part = mu.parts(data["by_match"], 2)[(1, "Moneyline")]
        self.assertLess(part["M3"], 0)
        b = data["scores"]
        self.assertLess(b[1]["Moneyline"][0], b[0]["Moneyline"][0])     # yet v14 scores better on Brier

    def test_the_reports(self):
        data = self.data()
        text = "\n".join(mu.text(data))
        for s in ("Brier, same ticks", "log loss, same ticks", "BY MATCH", "the 10 matches that cost v14 most"):
            self.assertIn(s, text)
        names = {"M1": ("09-01 12:00", "ALPHA v BRAVO")}
        lines = mu.match_text(data, names)
        self.assertTrue(any("ALPHA v BRAVO" in x and "24-17" in x for x in lines))
        first = next(i for i, x in enumerate(lines) if x.strip().startswith("match"))
        self.assertTrue(lines[first + 1].strip().startswith("M3"))      # the match that cost v14 most first
        self.assertIn("By match", mu.html(data, str))
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print"):
            mu.write_match_csv(os.path.join(d, "m.csv"), data, names)
            with open(os.path.join(d, "m.csv")) as fh:
                head, *body = fh.read().splitlines()
            self.assertIn("v14_moneyline_part", head)
            self.assertEqual(len(body), 2)

    def test_run_writes_every_match(self):
        prod = mu.Ticks("prod").add([q("M1", 50, 5, 0.8)], STARTS, FINALS)
        cand = mu.Ticks("v14").add([q("M1", 50, 5, 0.7)], STARTS, FINALS)
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print"):
            mu.run(d, [("prod", prod), ("v14", cand)], factor=2.0)
            self.assertTrue(os.path.exists(os.path.join(d, "margin_uplift_matches.txt")))
            self.assertTrue(os.path.exists(os.path.join(d, "margin_uplift_matches.csv")))

    def test_names(self):
        got = mu.names_of([{"MATCH_CODE": "M1", "SCHEDULED_START_TIME_UTC": "2026-10-03 12:05:00",
                            "PLAYER_1_HANDLE": "ALPHA", "PLAYER_2_HANDLE": "BRAVO"}])
        self.assertEqual(got, {"M1": ("10-03 12:05", "ALPHA v BRAVO")})


if __name__ == "__main__":
    unittest.main()
