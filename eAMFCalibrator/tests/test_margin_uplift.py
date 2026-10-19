"""The margin uplift: every in-play tick a dummy bet, prod and each candidate as two books."""

import datetime as dt
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

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

    def test_only_live_in_play_ticks_on_the_three_markets(self):
        rows = [q("M1", 50, -5, 0.6), q("M1", 50, 0.4, 0.6), q("M1", 50, 3, 0.6),
                q("M1", 51, 3, 0.4, active="false"),
                q("M1", 52, 3, 0.5, "PLAYER 1 to score over 2.5 points more than PLAYER 2"),
                q("M1", 53, 3, 0.5, "PLAYER 2 to score over -2.5 points more than PLAYER 1"),
                q("M1", 54, 3, 0.5, "Over 40.5 points"), q("M3", 50, 3, 0.5)]
        t = mu.Ticks("x").add(rows, STARTS, FINALS)
        self.assertEqual(list(t.array("group")), [0, 1, 1, 2])
        self.assertEqual(list(t.array("line")), [0.0, 2.5, -2.5, 40.5])
        self.assertEqual(list(t.array("won")), [True, True, False, True])   # home by 7; 41 over 40.5

    def test_a_push_is_left_out(self):
        rows = [q("M2", 50, 3, 0.5), q("M2", 51, 3, 0.5), q("M2", 54, 3, 0.5, "Over 40"),
                q("M2", 54, 3, 0.5, "Over 39.5")]                       # M2 20-20
        t = mu.Ticks("x").add(rows, STARTS, FINALS)
        self.assertEqual(list(t.array("line")), [39.5])
        self.assertEqual(list(t.array("won")), [True])

    def test_the_clock_is_fitted_so_matches_end_at_sixty(self):
        rows = [q("M1", 50, 10, 0.6), q("M1", 50, 30, 0.6), q("M2", 54, 20, 0.5, "Over 39.5")]
        t = mu.Ticks("x").add(rows, STARTS, FINALS)
        self.assertAlmostEqual(t.mean_end(), 25.0)
        self.assertAlmostEqual(mu.factor_for(t), 60 / 25)


def _pair(prod_rows, cand_rows, starts=STARTS, finals=FINALS, **kw):
    """compute() on prod and one candidate, the 109% book, minutes x 2 unless given."""
    kw.setdefault("factor", 2.0)
    kw.setdefault("boot", 200)
    return mu.compute([("prod", mu.Ticks("prod").add(prod_rows, starts, finals)),
                       ("v21", mu.Ticks("v21").add(cand_rows, starts, finals))], **kw)


class TestTwoBooks(unittest.TestCase):
    """Each dummy bet to the book that pays more, each book scored on what it is left with."""

    def test_the_longer_price_takes_the_bet(self):
        # M1, home won: prod 50-50; v21 60-40, so prod pays more on home and v21 on away
        data = _pair([q("M1", 50, 5, 0.5), q("M1", 51, 5, 0.5)], [q("M1", 50, 5, 0.6), q("M1", 51, 5, 0.4)])
        st = data["cands"][0]["groups"]["Moneyline"]["all"]
        price = lambda p: 1 / (p * 1.09)
        s_home, s_away = 100 / (price(0.5) - 1), 100 / (price(0.4) - 1)
        self.assertEqual((st["bets"], st["matches"]), (2, 1))
        self.assertAlmostEqual(st["to_cand"], 100 * s_away / (s_away + s_home))
        self.assertAlmostEqual(st["takes"], 100.0)                       # v21 takes the away, which lost
        self.assertAlmostEqual(st["keeps"], -100 * 100 / s_home)         # prod keeps the home, which won
        self.assertAlmostEqual(st["head"][0], st["takes"] - st["keeps"])
        self.assertAlmostEqual(st["both"], 100 * (s_away - 100) / (s_away + s_home))
        self.assertAlmostEqual(st["alone"], 100 * (s_home - 100) / (2 * s_home))   # prod takes both at evens
        self.assertAlmostEqual(st["gain"][0], st["both"] - st["alone"])
        self.assertAlmostEqual(st["ll_diff"][0], math.log(0.5 / 0.6))   # v21 nearer the result on both
        self.assertEqual(data["cands"][0]["groups"]["Moneyline"]["bands"][3]["bets"], 2)   # 0.50-0.65

    def test_an_overconfident_candidate_is_picked_off(self):
        """On simulated matches with the truth known, prod calibrated and v21 too sure (logit x 1.5):
        on one book alone v21 makes more, two books show it losing where it differs."""
        rng = np.random.default_rng(3)
        n = 4000
        truth = rng.uniform(0.2, 0.8, n)
        home_won = rng.random(n) < truth
        sure = 1 / (1 + np.exp(-1.5 * np.log(truth / (1 - truth))))
        codes = [f"S{i}" for i in range(n)]
        starts = {c: KICK for c in codes}
        finals = {c: (21.0, 14.0) if w else (14.0, 21.0) for c, w in zip(codes, home_won)}
        quote = lambda probs: [r for c, p in zip(codes, probs) for r in (q(c, 50, 5, p), q(c, 51, 5, 1 - p))]
        data = _pair(quote(truth), quote(sure), starts, finals)
        st = data["cands"][0]["groups"]["Moneyline"]["all"]
        self.assertLess(st["head"][2], 0)                              # the whole interval below zero
        self.assertGreater(st["ll_diff"][1], 0)                        # and the log loss agrees
        alone = _pair(quote(sure), quote(truth), starts, finals)["cands"][0]["groups"]["Moneyline"]["all"]
        self.assertGreater(alone["alone"], st["alone"])                # one book: the overconfident one "wins"

    def test_every_match_counts_the_same(self):
        # prod and v21 level at evens; M1 (home won) quoted ten times, M3 (away won) once
        finals = dict(FINALS, M3=(10.0, 13.0))
        starts = dict(STARTS, M3=KICK)
        m1 = lambda times, p: [q("M1", 50, 3 + k, p) for k in range(times)]
        ten = _pair(m1(10, 0.5) + [q("M3", 50, 3, 0.5)], m1(10, 0.5) + [q("M3", 50, 3, 0.5)], starts, finals)
        one = _pair(m1(1, 0.5) + [q("M3", 50, 3, 0.5)], m1(1, 0.5) + [q("M3", 50, 3, 0.5)], starts, finals)
        a, b = (d["cands"][0]["groups"]["Moneyline"]["all"] for d in (ten, one))
        self.assertEqual((a["bets"], b["bets"]), (11, 2))
        self.assertAlmostEqual(a["alone"], b["alone"])
        s = 100 / (1 / (0.5 * 1.09) - 1)
        self.assertAlmostEqual(a["alone"], 100 * (s - 100) / (2 * s))

    def test_lines_are_compared_only_at_prods(self):
        line = lambda m, k, p, x: q("M1", m, k, p, f"Over {x} points" if m == 54 else
                                     f"PLAYER 1 to score over {x} points more than PLAYER 2")
        data = _pair([q("M1", 50, 5, 0.5), line(54, 5, 0.5, 40.5), line(54, 6, 0.5, 40.5),
                      line(52, 5, 0.5, 2.5)],
                     [q("M1", 50, 5, 0.6), line(54, 5, 0.6, 41.5), line(54, 6, 0.6, 40.5),
                      line(52, 5, 0.4, 2.5)])
        groups = data["cands"][0]["groups"]
        self.assertEqual(list(groups), ["Moneyline", "Spread", "Totals"])
        self.assertEqual((groups["Totals"]["same_line"], groups["Totals"]["priced"]), (1, 2))
        self.assertEqual(groups["Totals"]["all"]["bets"], 1)
        self.assertAlmostEqual(groups["Spread"]["all"]["to_cand"], 100.0)   # v21 pays more on home -2.5
        self.assertAlmostEqual(groups["Spread"]["all"]["takes"], -100 * 100 / (100 / (1 / (0.4 * 1.09) - 1)))
        self.assertEqual(groups["Moneyline"]["bands"].get(0), None)
        self.assertEqual(groups["Spread"]["bands"], {})                  # the price cuts: moneyline only

    def test_the_last_minutes_are_left_out(self):
        # minute 28 x 2: 4 game minutes left, inside the last 6
        data = _pair([q("M1", 50, 5, 0.5), q("M1", 50, 28, 0.5)], [q("M1", 50, 5, 0.6), q("M1", 50, 28, 0.6)])
        gd = data["cands"][0]["groups"]["Moneyline"]
        self.assertEqual(gd["all"]["bets"], 1)
        self.assertEqual(gd["quarters"][1]["bets"], 1)                 # 50 left: Q1
        self.assertIsNone(gd["quarters"][4])


class TestByMatch(unittest.TestCase):
    """Every match's part of the headline: the parts add up to it."""

    def data(self):
        starts = dict(STARTS, M3=KICK)
        finals = dict(FINALS, M3=(10.0, 13.0))                        # M1 home won, M3 away won
        prod = [q("M1", 50, 5, 0.80), q("M1", 51, 5, 0.20), q("M1", 50, 8, 0.90),
                q("M3", 50, 5, 0.85), q("M3", 51, 5, 0.15), q("M3", 50, 8, 0.92),
                q("M3", 54, 6, 0.55, "Over 20.5"), q("M1", 54, 6, 0.40, "Over 40.5"), q("M1", 50, 29, 0.95)]
        cand = [q("M1", 50, 5, 0.75), q("M1", 51, 5, 0.25), q("M1", 50, 8, 0.85),
                q("M3", 50, 5, 0.70), q("M3", 51, 5, 0.30), q("M3", 50, 8, 0.80),
                q("M3", 54, 6, 0.50, "Over 20.5"), q("M1", 54, 6, 0.45, "Over 40.5"), q("M1", 50, 29, 0.95)]
        return _pair(prod, cand, starts, finals, window="2026-10-01 to 2026-10-12")

    def test_the_parts_add_up_to_the_headline(self):
        data = self.data()
        rows = {r["code"]: r for r in data["by_match"]}
        self.assertEqual(set(rows), {"M1", "M3"})
        self.assertEqual(rows["M3"]["final"], (10.0, 13.0))
        self.assertEqual(rows["M1"]["cands"][0]["Moneyline"]["ticks"], 3)   # minute 29: 2 game minutes left
        for g, gd in data["cands"][0]["groups"].items():
            parts = [r["cands"][0][g]["part"] for r in data["by_match"] if g in r["cands"][0]]
            self.assertAlmostEqual(sum(parts), gd["all"]["head"][0])
        # v21 less sure of M3's home side, which lost: it takes the home bets and keeps their stakes, while
        # prod keeps the away, which won; on M1 the home side won, so the same costs v21
        self.assertGreater(rows["M3"]["cands"][0]["Moneyline"]["part"], 0)
        self.assertLess(rows["M1"]["cands"][0]["Moneyline"]["part"], 0)

    def test_the_reports(self):
        data = self.data()
        text = "\n".join(mu.text(data))
        for s in ("MARGIN UPLIFT -- two books", "candidate - prod", "by quarter", "by prod's chance",
                  "v21 against prod, Totals", "BY MATCH", "the 10 matches that cost it most"):
            self.assertIn(s, text)
        names = {"M1": ("09-01 12:00", "ALPHA v BRAVO")}
        lines = mu.match_text(data, names)
        self.assertTrue(any("ALPHA v BRAVO" in x and "24-17" in x for x in lines))
        first = next(i for i, x in enumerate(lines) if x.strip().startswith("match"))
        self.assertTrue(lines[first + 1].strip().startswith("M1"))      # the match that cost v21 most first
        page = mu.html(data, str)
        for s in ("<th>Candidate &minus; prod</th>", "<th>Both &minus; prod alone</th>", "v21, Totals: by quarter",
                  "By match", "matched"):
            self.assertIn(s, page)
        with tempfile.TemporaryDirectory() as d:
            mu.write_match_csv(os.path.join(d, "m.csv"), data, names)
            with open(os.path.join(d, "m.csv")) as fh:
                head, *body = fh.read().splitlines()
            self.assertTrue(head.startswith("match_code,start,players"))
            self.assertIn("part", head)
            self.assertEqual(len(body), 4)                              # 2 matches x 2 markets
            mu.write_csv(os.path.join(d, "u.csv"), data)
            with open(os.path.join(d, "u.csv")) as fh:
                head, *body = fh.read().splitlines()
            self.assertTrue(head.startswith("candidate,market,cut,bets,matches,to_cand"))
            self.assertIn("v21,Moneyline,all,", "\n".join(body))

    def test_no_match_in_every_feed(self):
        prod = mu.Ticks("prod").add([q("M1", 50, 3, 0.5)], STARTS, FINALS)
        cand = mu.Ticks("v13").add([q("M2", 50, 3, 0.5)], STARTS, FINALS)
        self.assertIsNone(mu.compute([("prod", prod), ("v13", cand)], CHART))
        with mock.patch("builtins.print"):
            self.assertIsNone(mu.run("out", [("prod", prod), ("v13", cand)]))

    def test_run_writes_every_file(self):
        prod = mu.Ticks("prod").add([q("M1", 50, 5, 0.8)], STARTS, FINALS)
        cand = mu.Ticks("v14").add([q("M1", 50, 5, 0.7)], STARTS, FINALS)
        with tempfile.TemporaryDirectory() as d, mock.patch("builtins.print"):
            mu.run(d, [("prod", prod), ("v14", cand)], factor=2.0)
            self.assertEqual(sorted(os.listdir(d)), ["margin_uplift.csv", "margin_uplift.txt",
                                                     "margin_uplift_matches.csv", "margin_uplift_matches.txt"])

    def test_names(self):
        got = mu.names_of([{"MATCH_CODE": "M1", "SCHEDULED_START_TIME_UTC": "2026-10-03 12:05:00",
                            "PLAYER_1_HANDLE": "ALPHA", "PLAYER_2_HANDLE": "BRAVO"}])
        self.assertEqual(got, {"M1": ("10-03 12:05", "ALPHA v BRAVO")})


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


class TestBetMoments(unittest.TestCase):
    """The candidates are priced with the bets' moments in config.BET_MOMENTS (none with --all-moments),
    and a model priced at them alone is matched in the margin uplift at its own ticks only."""

    def run_bets(self, all_moments):
        from .. import snowflake_io
        from .test_bets import MONEYLINE, PRICES, _checks_for, at, odds_for
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER",
                "OPERATOR_NAME", "BET_IN_PLAY", "CUSTOMER_TEMPERATURE"]
        rows = [(i, "M1", at(c + 7), 1, 1, odds_for(p), 10, 10, None, 2, "FANDUEL", "Yes", "Standard")
                for i, (c, p) in enumerate(PRICES[1:])]
        history = [{"MATCH_CODE": "M1", "SCHEDULED_START_TIME_UTC": f"{at(0) - dt.timedelta(minutes=5):%Y-%m-%d %H:%M:%S}",
                    "PLAYER_1_FINAL_SCORE": "21", "PLAYER_2_FINAL_SCORE": "17"}]
        seen = []

        def quotes(cur, stream, batch):
            seen.append((stream, config.BET_MOMENTS))
            return MONEYLINE
        with mock.patch.object(bets, "fetch_all", return_value=(cols, rows)), \
                mock.patch.object(snowflake_io, "fetch_quotes", side_effect=quotes), \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 17)}), \
                mock.patch.object(snowflake_io, "fetch_history", return_value=history), \
                mock.patch.object(snowflake_io, "at_bet_moments", lambda s: s == "CAND_TABLE" and not all_moments), \
                mock.patch.object(bets, "write_csv"), mock.patch.object(bets, "write_text"), \
                mock.patch("builtins.print"), mock.patch("os.makedirs"), \
                mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.multiple(config, LAG_RANGE=(-20, 30), CANDIDATES=["CAND_TABLE"], ALL_MOMENTS=all_moments), \
                mock.patch.object(mu, "run", return_value=None) as run:
            bets.run(None, "out", summary={})
        return seen, run.call_args.args[1]

    def test_the_moments_reach_the_candidates_alone(self):
        seen, feeds = self.run_bets(False)
        self.assertEqual([m for s, m in seen if s == config.STREAMS["prod"]], [None])
        moments = dict(seen)["CAND_TABLE"]
        self.assertEqual(set(moments), {"M1"})
        self.assertTrue(moments["M1"] and {k for k, _ in moments["M1"]} == {50})
        self.assertLessEqual({m for _, m in moments["M1"]}, {0, 100, 200, 300, 400})   # prod's messages
        self.assertIsNone(config.BET_MOMENTS)                                     # put back after the run
        cand = dict(feeds)["CAND_TABLE"]
        self.assertEqual((cand.at_bets, cand.max_age), (True, 0.0))
        self.assertEqual((dict(feeds)["prod"].at_bets, dict(feeds)["prod"].max_age), (False, None))

    def test_all_moments(self):
        seen, feeds = self.run_bets(True)
        self.assertIsNone(dict(seen)["CAND_TABLE"])
        self.assertFalse(dict(feeds)["CAND_TABLE"].at_bets)

    def test_a_model_at_the_bets_moments_is_matched_at_its_own_ticks(self):
        prod = mu.Ticks("prod").add([q("M1", 50, k, 0.5) for k in (3, 4, 5, 6)], STARTS, FINALS)
        cand = mu.Ticks("v21").add([q("M1", 50, 4, 0.6)], STARTS, FINALS)    # the one moment a bet read
        self.assertEqual(len(mu.matched([("prod", prod), ("v21", cand)])[0][1]), 3)   # 4, 5, 6: its price held
        cand.at_bets, cand.max_age = True, 0.0
        (_, p), (_, c) = mu.matched([("prod", prod), ("v21", cand)])
        self.assertEqual(list(p.array("minutes")), [4.0])
        data = mu.compute([("prod", prod), ("v21", cand)], factor=2.0, boot=50)
        self.assertTrue(data["feeds"][1]["at_bets"])
        self.assertIn("v21 priced only at the moments the bets read", "\n".join(mu.text(data)))
        self.assertIn("v21 priced only at the moments the bets read", mu.html(data, str))


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

    def test_two_sections(self):
        data = TestByMatch().data()
        page = bets.html_section({"results": [("v13", [])], "uplift": data})
        for s in ('id="bets"', 'id="books"', 'id="uplift"', "<h2>Re-pricing</h2>", "<h2>Book comparison</h2>",
                  "<h3>Two books</h3>", "<h3 id=\"uplift\">Margin uplift</h3>", "<th>Candidate &minus; prod</th>"):
            self.assertIn(s, page)
        self.assertEqual(page.count("<h2>"), 2)
        self.assertNotIn("Betting simulation", page)
        self.assertLess(page.index("<h3>Two books</h3>"), page.index("Margin uplift</h3>"))
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
        self.assertEqual(list(c.array("group")), [0, 0, 2])
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
        data = mu.compute([("prod", prod), ("v14", cand)], chart=CHART, boot=50)
        self.assertEqual([(f["ticks"], f["of"]) for f in data["feeds"]], [(3, 4), (3, 1)])  # 5: over 2 old
        self.assertEqual(data["cands"][0]["groups"]["Moneyline"]["all"]["bets"], 3)
        self.assertIn("0.05-0.95", "\n".join(mu.text(data)))
        self.assertIn("matched", mu.html(data, lambda x: x))

    def test_nothing_matched(self):
        prod = mu.Ticks("prod").add([q("M1", 50, 2, 0.6)], STARTS, FINALS)
        cand = mu.Ticks("v14").add([q("M1", 50, 3, 0.6)], STARTS, FINALS)
        self.assertIsNone(mu.compute([("prod", prod), ("v14", cand)], chart=CHART))


if __name__ == "__main__":
    unittest.main()
