"""The betting simulation: a lag per operator off the odds, the join, and the re-pricing."""

import datetime as dt
import os
import unittest
from unittest import mock

from .. import bet_checks, bets, config

T0 = dt.datetime(2026, 9, 20, 12, 0, 0)


def at(seconds):
    return T0 + dt.timedelta(seconds=seconds)


def quote(match, market, seconds, prob, line=None, message=None, status="open", active="true"):
    """A snowflake_io.fetch_quotes row, published at T0 + seconds (GAMEPLAI publishes 0-100)."""
    desc = None if line is None else f"Line {line}"
    return (match, market, at(seconds), 100.0 * prob, None, desc,
            seconds if message is None else message, status, active)


def bet(match, seconds, market_type=1, selection=1, odds=1.9, stake=10.0, revenue=10.0,
        line=None, operator="FANDUEL", in_play="Yes", **extra):
    return bets.Bet(bet_id=f"{match}-{seconds}", match_code=match, time=at(seconds),
                    market_type=market_type, selection=selection, odds=odds, stake=stake,
                    revenue=revenue, line=line,
                    extra=dict({config.BET_GROUP_COLUMN: operator,
                                config.BET_IN_PLAY_COLUMN: in_play}, **extra))


def _checks_for(quotes):
    """Checks for M1 with every message in SCOUTING_FULL and nothing moving the game on."""
    rows = [("M1", q[6], None, None, None, None, None, None, None, None) for q in quotes]
    rows.append(("M1", 0, None, "FIRST_QUARTER_STARTED", None, None, None, None, None, None))
    return bet_checks.Checks(bet_checks.build_feeds(rows), bet_checks.message_times(quotes), {})
MARGIN = 1.05
# prod's home moneyline: changes every 100s
PRICES = [(0, 0.40), (100, 0.60), (200, 0.45), (300, 0.70), (400, 0.50)]
MONEYLINE = [quote("M1", 50, t, p) for t, p in PRICES]


def odds_for(p):
    return 1.0 / (p * MARGIN)


class TestLag(unittest.TestCase):

    def test_the_lag_is_where_the_odds_follow_prod(self):
        tl = bets.timeline(MONEYLINE)
        bs = []
        for (c, new), (_, old) in zip(PRICES[1:], PRICES):
            bs.append(bet("M1", c + 6, odds=odds_for(old)))   # still the old price: lag 7 or more
            bs.append(bet("M1", c + 7, odds=odds_for(new)))   # the new one: lag 7 or less
        lag = bets.fit_lags(bs, tl, lags=list(range(-20, 31)))["FANDUEL"]
        self.assertEqual(lag.seconds, 7)
        self.assertLess(lag.curve[7], lag.curve[0])

    def test_a_clock_behind_prods_gives_a_negative_lag(self):
        tl = bets.timeline(MONEYLINE)
        bs = []
        for (c, new), (_, old) in zip(PRICES[1:], PRICES):
            bs.append(bet("M1", c - 4, odds=odds_for(old)))
            bs.append(bet("M1", c - 3, odds=odds_for(new)))
        self.assertEqual(bets.fit_lags(bs, tl, lags=list(range(-20, 31)))["FANDUEL"].seconds, -3)

    def test_pre_match_bets_are_read_at_bet_time(self):
        lags = {"FANDUEL": bets.Lag(7.0)}
        self.assertEqual(bets.bet_lag(bet("M1", 50), lags), 7.0)
        self.assertEqual(bets.bet_lag(bet("M1", 50, in_play="No"), lags), 0.0)
        self.assertEqual(bets.bet_lag(bet("M1", 50, operator="OTHER"), lags), 0.0)

    def test_a_spread_quoted_from_the_other_side_gets_its_sign_turned(self):
        tl = bets.timeline([quote("M1", 52, 0, 0.5, -3.5), quote("M1", 53, 0, 0.5, -3.5),
                            quote("M1", 54, 0, 0.5, 44.5)])
        bs = [bet("M1", 10, 2, 1, line=-3.5), bet("M1", 11, 2, 2, line=3.5),
              bet("M1", 12, 2, 2, line=3.5), bet("M1", 13, 3, 1, line=44.5)]
        signs = bets.fit_line_signs(bs, tl, {})
        self.assertEqual(signs[("FANDUEL", 52)][0], 1)
        self.assertEqual(signs[("FANDUEL", 53)][0], -1)
        self.assertEqual(signs[("FANDUEL", 54)][0], 1)


class TestJoin(unittest.TestCase):

    def setUp(self):
        self.prod = bets.timeline([quote("M1", 54, 0, 0.50, 44.5, message=10),
                                   quote("M1", 54, 40, 0.40, 44.5, message=12)])
        cand = [quote("M1", 54, 1, 0.55, 44.5, message=10), quote("M1", 54, 41, 0.45, 44.5, message=12)]
        self.cand, self.cand_tl = bets.quote_index(cand), bets.timeline(cand)

    def join(self, bs, lag=0.0, signs=None):
        return bets.join(bs, {"FANDUEL": bets.Lag(lag)}, signs or {}, self.prod, self.cand,
                         self.cand_tl)

    def test_an_in_play_bet_is_priced_one_lag_earlier_and_a_pre_match_one_at_bet_time(self):
        live, pre = self.join([bet("M1", 50, 3, 1, line=44.5, odds=2.2),
                               bet("M1", 50, 3, 1, line=44.5, odds=2.2, in_play="No")], lag=15)
        self.assertEqual((live["message"], live["stream_prob"], live["candidate_prob"]), (10, 0.50, 0.55))
        self.assertEqual((pre["message"], pre["stream_prob"], pre["candidate_prob"]), (12, 0.40, 0.45))
        self.assertEqual(live["stream_prob_no_lag"], 0.40)

    def test_the_candidate_is_found_by_time_when_prods_quote_has_no_message(self):
        prod = bets.timeline([quote("M1", 50, 0, 0.5)])
        prod[("M1", 50)][1][0] = (None,) + prod[("M1", 50)][1][0][1:]
        cand_tl = bets.timeline([quote("M1", 50, 1, 0.6)])
        (r,) = bets.join([bet("M1", 5, in_play="No")], {}, {}, prod, {}, cand_tl)
        self.assertEqual((r["stream_prob"], r["candidate_prob"]), (0.5, 0.6))

    def test_a_loser_keeps_its_revenue_and_a_winner_is_repaid_at_the_candidates_odds(self):
        lost, won = self.join([bet("M1", 45, 3, 1, line=44.5, odds=2.4, stake=10, revenue=10),
                               bet("M1", 45, 3, 1, line=44.5, odds=2.4, stake=10, revenue=-14)])
        self.assertEqual((lost["result"], won["result"]), (bets.LOST, bets.WON))
        self.assertEqual(lost["candidate_revenue"], 10)
        odds_c = 2.4 * 0.40 / 0.45
        self.assertAlmostEqual(won["candidate_odds"], odds_c, places=4)
        self.assertAlmostEqual(won["candidate_revenue"], 10 - 10 * odds_c)

    def test_a_turned_line_is_matched_on_prods_side_and_another_line_is_not(self):
        turned, other = self.join([bet("M1", 45, 3, 1, line=-44.5), bet("M1", 45, 3, 1, line=47.5)],
                                  signs={("FANDUEL", 54): (-1, 0, 0, 0)})
        self.assertTrue(turned["line_match"] and turned["simulated"])
        self.assertFalse(other["line_match"] or other["simulated"])
        self.assertEqual(bets.why_not(other), "not on prod's line")

    def test_the_candidate_takes_the_bet_at_its_own_line_and_settles_it_there(self):
        # the game ends 24-21 (45): over 44.5 won with prod; the candidate's line was 45.5
        cand = [quote("M1", 54, 1, 0.40, 45.5, message=10)]
        (r,) = bets.join([bet("M1", 5, 3, 1, line=44.5, odds=2.0, stake=10, revenue=-10)], {}, {},
                         self.prod, bets.quote_index(cand), bets.timeline(cand), {"M1": (24, 21)})
        self.assertFalse(r["same_line"])
        self.assertEqual((r["result"], r["result_by_score"], r["candidate_result"]),
                         (bets.WON, bets.WON, bets.LOST))
        self.assertTrue(r["simulated"])
        self.assertAlmostEqual(r["candidate_odds"], 2.0 * 0.50 / 0.40)
        self.assertEqual(r["candidate_revenue"], 10)
        self.assertEqual(bets.outcomes([r])["total"][(bets.WON, bets.LOST)], (1, 10.0, -10.0, 10.0))
        self.assertEqual(bets.effects([r]), (0.0, 20.0))

    def test_without_a_final_score_a_bet_at_another_line_cannot_be_settled(self):
        cand = [quote("M1", 54, 1, 0.40, 45.5, message=10)]
        (r,) = bets.join([bet("M1", 5, 3, 1, line=44.5)], {}, {}, self.prod,
                         bets.quote_index(cand), bets.timeline(cand))
        self.assertFalse(r["simulated"])
        self.assertEqual(bets.why_not(r), "no final score")

    def test_settling_off_the_score(self):
        self.assertEqual(bets.settle(54, 44.5, (24, 21)), bets.WON)
        self.assertEqual(bets.settle(55, 44.5, (24, 21)), bets.LOST)
        self.assertEqual(bets.settle(54, 45.0, (24, 21)), bets.PUSH)
        self.assertEqual(bets.settle(52, 2.5, (24, 21)), bets.WON)       # home by 3, needs more than 2.5
        self.assertEqual(bets.settle(53, -2.5, (24, 21)), bets.LOST)     # away needs to lose by under 2.5
        self.assertEqual(bets.settle(50, None, (24, 21)), bets.WON)
        self.assertIsNone(bets.settle(54, None, (24, 21)))
        self.assertIsNone(bets.settle(54, 44.5, None))

    def test_the_settle_check_counts_where_the_score_agrees_with_the_operator(self):
        rows = self.join([bet("M1", 45, 3, 1, line=44.5, odds=2.0, stake=10, revenue=-10),
                          bet("M1", 45, 3, 1, line=44.5, odds=2.0, stake=10, revenue=10)])
        for r in rows:
            r["result_by_score"] = bets.settle(54, 44.5, (24, 21))
        self.assertEqual(bets.settle_check(rows), {"total": (2, 1)})

    def test_results_are_read_off_revenue_and_cash_outs_are_left_out(self):
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=0)), bets.PUSH)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=-10)), bets.WON)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=3)), bets.OTHER)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=10,
                                            BET_CASHED_OUT="Yes")), bets.CASHED)

    def test_the_summary_splits_pre_match_and_in_play_and_can_leave_out_the_vips(self):
        rows = self.join([bet("M1", 45, 3, 1, line=44.5, odds=2.4, stake=10, revenue=10,
                              CUSTOMER_TEMPERATURE="Standard"),
                          bet("M1", 45, 3, 1, line=44.5, odds=2.4, stake=10, revenue=-14,
                              CUSTOMER_TEMPERATURE="VIP", in_play="No")])
        n, stake, rev, margin, rev_c, margin_c = bets.summarise(rows)["all"]
        self.assertEqual((n, stake, rev), (2, 20.0, -4.0))
        self.assertGreater(margin_c, margin)
        split = bets.summarise(rows, "in_play")
        self.assertEqual((split[True][0], split[False][0]), (1, 1))
        no_vip = bets.summarise(rows, keep=lambda r: r["CUSTOMER_TEMPERATURE"] != "VIP")["all"]
        self.assertEqual(no_vip[:3], (1, 10.0, 10.0))

    def test_coverage_shows_where_the_money_is_and_how_much_is_re_priced(self):
        rows = self.join([bet("M1", 45, 3, 1, line=44.5, stake=10),
                          bet("M1", 45, 3, 1, line=47.5, stake=30)])
        self.assertEqual(bets.coverage(rows)[(True, "total")], (2, 40.0, 1, 10.0))


class TestCommand(unittest.TestCase):

    def test_bets_probe_runs_the_probe_and_bets_runs_the_pipeline(self):
        from .. import __main__ as cli, snowflake_io
        with mock.patch.object(snowflake_io, "get_connection", return_value=mock.MagicMock()), \
                mock.patch.object(bets, "probe", return_value=["probed"]) as probe, \
                mock.patch.object(bets, "run") as run, \
                mock.patch("builtins.print"), \
                mock.patch("builtins.open", mock.mock_open()):
            self.assertEqual(cli.main(["bets", "probe", "--since", "2026-09-18"]), 0)
            self.assertTrue(probe.called and not run.called)
            self.assertEqual(cli.main(["bets", "--since", "2026-09-18"]), 0)
            self.assertTrue(run.called)

    def test_the_probe_runs_on_fake_results(self):
        def fake_fetch(cur, sql, params=None):
            if "LIMIT 0" in sql:
                return [v for v in config.BET_COLUMNS.values() if v] + config.BET_EXTRA_COLUMNS, []
            if "GROUP BY OPERATOR_NAME" in sql:
                return ["OPERATOR_NAME", "BETS"], [("FANDUEL", 10)]
            if "GROUP BY 1, 2, 3, 4" in sql:
                return ["OPERATOR_NAME", "BET_TYPE", "ROWS_"], [("FANDUEL", "Multi", 70000)]
            return ["V", "N"], [("x", 3)]
        with mock.patch.object(bets, "fetch_all", side_effect=fake_fetch):
            text = "\n".join(bets.probe(None))
        self.assertIn("FANDUEL", text)
        self.assertIn("configured columns missing: none", text)
        self.assertIn("Multi", text)
        self.assertNotIn("failed", text)

    def test_the_whole_pipeline_runs_on_fake_results(self):
        from .. import snowflake_io
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER",
                "OPERATOR_NAME", "BET_IN_PLAY", "CUSTOMER_TEMPERATURE"]
        rows = []
        for (c, new), (_, old) in zip(PRICES[1:], PRICES):
            for t, p in ((c + 6, old), (c + 7, new)):
                rows.append((len(rows), "M1", at(t), 1, 1, odds_for(p), 10, 10, None, 2, "FANDUEL",
                             "Yes", "Standard"))
        with mock.patch.object(bets, "fetch_all", return_value=(cols, rows)), \
                mock.patch.object(snowflake_io, "fetch_quotes", return_value=MONEYLINE), \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 17)}), \
                mock.patch.object(bets, "write_csv"), mock.patch.object(bets, "write_text"), \
                mock.patch("builtins.print"), \
                mock.patch("os.makedirs"), mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.object(config, "LAG_RANGE", (-20, 30)):
            out = bets.run(None, "out")
        self.assertEqual(len(out), 8)
        self.assertTrue(all(r["latency_seconds"] == 7 for r in out))
        self.assertTrue(all(r["simulated"] for r in out))
        with mock.patch.object(bets, "fetch_all", return_value=(cols, rows)), \
                mock.patch.object(snowflake_io, "fetch_quotes", return_value=MONEYLINE) as fq, \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 17)}), \
                mock.patch.object(bets, "write_csv") as written, mock.patch.object(bets, "write_text"), \
                mock.patch("builtins.print"), \
                mock.patch("os.makedirs"), mock.patch.object(config, "LAG_RANGE", (-20, 30)), \
                mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.object(config, "CANDIDATES", ["MODEL:v8", "MODEL:v9", "MODEL:v10"]), \
                mock.patch.object(bets, "check_models"):
            results = bets.run(None, "out")
        self.assertEqual([n for n, _ in results], ["v8", "v9", "v10"])
        self.assertIn(mock.call(None, "MODEL:v10", ["M1"]), fq.call_args_list)
        self.assertIn("candidate_prob_v10", written.call_args_list[-1][0][1][0])


class TestReading(unittest.TestCase):

    def test_bets_are_read_through_the_configured_columns(self):
        cols = ["MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID", "ODDS", "STAKE_GBP",
                "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER", "OPERATOR_NAME", "BET_IN_PLAY"]
        row = ("M1", T0, 2, 2, 1.95, 5, -4.75, -3.5, 3, "HARDROCK", "No")
        (b,) = bets.to_bets(cols, [row])
        self.assertEqual((b.feed_market, b.line, b.operator, b.in_play), (53, -3.5, "HARDROCK", False))

    def test_the_query_reads_every_single_in_the_window(self):
        sql, params, wanted = bets.bets_sql()
        self.assertIn("MARKET_TYPE_ID IN (1, 2, 3)", sql)
        self.assertIn("UPPER(BET_TYPE) = 'SINGLE'", sql)
        self.assertNotIn("BET_IN_PLAY = 'Yes'", sql)
        self.assertIn(config.SPORT_CODE, params)

    def test_the_bet_source_can_live_in_another_database(self):
        saved = config.BET_TABLE
        try:
            config.BET_TABLE = "OTHER_DB.VIEWS.BETS"
            self.assertIn("FROM OTHER_DB.VIEWS.BETS", bets.bets_sql()[0])
            config.BET_TABLE = "CUSTOMER_REVENUE"
            self.assertEqual(bets.bet_table(), f"{config.DATABASE}.{config.SCHEMA}.CUSTOMER_REVENUE")
        finally:
            config.BET_TABLE = saved

    def test_models_are_candidates_at_their_own_lines(self):
        saved = config.STREAMS["candidate"], config.CANDIDATES
        try:
            config.CANDIDATES = []
            config.STREAMS["candidate"] = "GAMEPLAI_STREAM_CANDIDATE"
            self.assertEqual(bets.candidate_streams(), ["GAMEPLAI_STREAM_CANDIDATE"])
            config.CANDIDATES = ["MODEL:v8", "MODEL:v9", "MODEL:v10@prod"]
            self.assertEqual([bets.label(s) for s in bets.candidate_streams()],
                             ["v8", "v9", "v10@prod"])
        finally:
            config.STREAMS["candidate"], config.CANDIDATES = saved


class TestCandidates(unittest.TestCase):

    def test_each_model_is_read_at_prods_message_so_the_lag_is_the_same(self):
        # prod's total moves 44.5 -> 47.5 at message 3; the model priced at PLAY_OVERs 1 and 3
        # (its rows carry prod's messages and publish times), at its own lines
        prod = [quote("M1", 54, 0, 0.50, 44.5, message=1), quote("M1", 54, 30, 0.50, 47.5, message=3)]
        v5 = [quote("M1", 54, 0, 0.50, 45.5, message=1), quote("M1", 54, 30, 0.50, 48.5, message=3)]
        lags = {"FANDUEL": bets.Lag(10)}
        b = bet("M1", 35, market_type=3, selection=1, odds=1.9, line=44.5)    # seen at 25s: msg 1
        (row,) = bets.join([b], lags, {}, bets.timeline(prod), bets.quote_index(v5),
                           bets.timeline(v5), {"M1": (24, 22)})
        self.assertEqual((row["message"], row["stream_line"], row["candidate_line"]), (1, 44.5, 45.5))
        self.assertFalse(row["same_line"])
        self.assertEqual(row["candidate_result"], bets.WON)     # 46 over 45.5
        self.assertTrue(row["simulated"])

    def test_a_missing_model_stops_the_run_before_any_fetching(self):
        with mock.patch.object(config, "V9_MODEL_DIR", "no_such_v9_model"), \
                mock.patch.object(config, "CANDIDATES", ["GAMEPLAI_STREAM_CANDIDATE", "MODEL:v9"]), \
                mock.patch.object(bets, "fetch_all") as fetched:
            with self.assertRaises(SystemExit):
                bets.run(None, "out")
        fetched.assert_not_called()

    def test_candidates_are_compared_on_the_bets_they_all_priced(self):
        def row(stake, rev, rev_c, simulated=True, market="total"):
            return dict(stake=stake, revenue=rev, candidate_revenue=rev_c, simulated=simulated,
                        market=market, result=bets.WON, candidate_result=bets.WON, in_play=True)
        a = [row(10, 1, 2), row(10, 1, 3), row(10, 1, 4)]
        b = [row(10, 1, 1), row(10, 1, 0, simulated=False), row(10, 1, 5)]
        out = bets.compare([("v4", a), ("v5", b)])
        n, stake, m, margins, split = out["all"]
        self.assertEqual((n, stake, m), (2, 20, 10.0))
        self.assertEqual(margins, [30.0, 30.0])
        self.assertEqual(split, [(4.0, 0.0), (4.0, 0.0)])
        text = "\n".join(bets.compare_report([("v4", a), ("v5", b)]))
        self.assertIn("+20.00", text)
        w = bets.wide([("v4", a), ("v5", b)])
        self.assertEqual((w[1]["simulated_v4"], w[1]["simulated_v5"]), (True, False))
        self.assertNotIn("simulated", w[1])


class TestPrematch(unittest.TestCase):
    """`bets prematch`: the pre-match models' own test, on the bets placed before kick-off."""

    @staticmethod
    def results():
        def row(match, day, stake, rev, rev_c, market="spread", in_play=False):
            return dict(match_code=match, bet_time=dt.datetime(2026, 9, day, 12), stake=stake,
                        revenue=rev, candidate_revenue=rev_c, simulated=True, market=market,
                        in_play=in_play)
        # week 38 (M1): prod 10%, v10 20%, glmer 30%; week 39 (M2): -10%, -20%, -5%
        v10 = [row("M1", 15, 10, 1, 2), row("M1", 15, 10, 1, 2, market="total"),
               row("M2", 22, 20, -2, -4), row("M1", 15, 100, 50, 0, in_play=True)]
        glmer = [dict(r, candidate_revenue=c) for r, c in zip(v10, (3, 3, -1, 0))]
        return [("v10", v10), ("v10-glmer", glmer)]

    def test_the_report_compares_the_candidates_overall_by_market_and_week(self):
        lines = bets.prematch_report(self.results(), n_boot=200)
        text = "\n".join(lines)
        self.assertIn("3 bets placed before kick-off, re-priced by v10, v10-glmer", text)
        overall = next(l for l in lines if l.strip().startswith("all "))
        self.assertEqual(overall.split()[1:], ["3", "40", "0.00%", "+0.00", "+12.50"])
        week38 = next(l for l in lines if l.strip().startswith("2026-W38"))
        self.assertEqual(week38.split()[1:], ["2", "20", "10.00%", "+10.00", "+20.00"])
        week39 = next(l for l in lines if l.strip().startswith("2026-W39"))
        self.assertEqual(week39.split()[1:], ["1", "20", "-10.00%", "-10.00", "+5.00"])
        self.assertIn("total ", text)
        head2head = next(l for l in lines if l.strip().startswith("v10-glmer"))
        self.assertIn("+12.50", head2head)
        self.assertIn("ahead in 2 of 2 weeks", head2head)
        lo, hi = (float(x) for x in head2head.split("[")[1].split("]")[0].split(","))
        self.assertLessEqual(lo, 12.5)
        self.assertGreaterEqual(hi, 12.5)
        # one candidate: no head to head; no pre-match bets: said so
        self.assertNotIn("each candidate against", "\n".join(bets.prematch_report(self.results()[:1])))
        none = [("v10", [r for r in self.results()[0][1] if r["in_play"]])]
        self.assertIn("(none)", "\n".join(bets.prematch_report(none)))
        self.assertEqual(bets._iso_week(None), "unknown")
        self.assertEqual(bets._iso_week(dt.datetime(2027, 1, 1)), "2026-W53")

    def test_bets_prematch_prices_only_the_bets_before_kick_off(self):
        from .. import snowflake_io
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER",
                "OPERATOR_NAME", "BET_IN_PLAY", "CUSTOMER_TEMPERATURE"]
        rows = []
        for (c, new), (_, old) in zip(PRICES[1:], PRICES):
            for t, p in ((c + 6, old), (c + 7, new)):
                rows.append((len(rows), "M1", at(t), 1, 1, odds_for(p), 10, 10, None, 2, "FANDUEL",
                             "No" if len(rows) % 2 else "Yes", "Standard"))
        seen = []

        def quotes(cur, stream, matches):
            seen.append((stream, config.PREMATCH_ONLY))
            return MONEYLINE
        with mock.patch.object(bets, "fetch_all", return_value=(cols, rows)), \
                mock.patch.object(snowflake_io, "fetch_quotes", side_effect=quotes), \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 17)}), \
                mock.patch.object(bets, "write_csv") as written, \
                mock.patch.object(bets, "write_text") as text, mock.patch("builtins.print"), \
                mock.patch("os.makedirs"), mock.patch.object(config, "LAG_RANGE", (-20, 30)), \
                mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.object(config, "CANDIDATES", ["MODEL:v10", "MODEL:v10-glmer"]), \
                mock.patch.object(bets, "check_models"):
            results = bets.run(None, "out", prematch_only=True)
        self.assertEqual([n for n, _ in results], ["v10", "v10-glmer"])
        self.assertEqual(len(results[0][1]), 4)
        self.assertFalse(any(r["in_play"] for _, rs in results for r in rs))
        # the models priced pre-match only, and the switch is put back after
        self.assertEqual([on for s, on in seen if s.startswith("MODEL:")], [True, True])
        self.assertFalse(config.PREMATCH_ONLY)
        paths = [c[0][0] for c in written.call_args_list] + [c[0][0] for c in text.call_args_list]
        self.assertIn(os.path.join("out", "bets_prematch_sim.csv"), paths)
        self.assertIn(os.path.join("out", "bets_prematch.txt"), paths)
        self.assertFalse(any(os.path.basename(p) == "bets_sim.csv" for p in paths))


if __name__ == "__main__":
    unittest.main()


class TestHtmlSection(unittest.TestCase):

    def test_the_section_shows_each_candidate_and_the_side_by_side(self):
        def row(stake, rev, rev_c, simulated=True, market="total", op="FANDUEL_BET_BY_BET"):
            return {"stake": stake, "revenue": rev, "candidate_revenue": rev_c, "simulated": simulated,
                    "market": market, "result": bets.WON, "candidate_result": bets.WON, "in_play": True,
                    "period": 2, config.BET_GROUP_COLUMN: op, "stream_prob": 0.5, "candidate_prob": 0.5,
                    "on_prod_line": True}
        a = [row(10, 1, 2), row(10, 1, 3)]
        b = [row(10, 1, 1), row(10, 1, 0, simulated=False)]
        html = bets.html_section({"results": [("GAMEPLAI_STREAM_CANDIDATE", a), ("v6", b)],
                                  "lags": {"FANDUEL_BET_BY_BET": bets.Lag(1, 100)}, "bets": 2,
                                  "matches": 1})
        self.assertIn('id="bets"', html)
        self.assertIn("<h3>Side by side <span class=\"dim\">1 bets</span></h3>", html)
        self.assertNotIn("is re-priced with the candidate", html)
        self.assertIn("<th>v6</th>", html)
        self.assertIn("Fanduel +1s (100 bets)", html)
        self.assertIn('class="good">+10.00', html)
        self.assertEqual(bets.html_section({}), "")


class TestGamer(unittest.TestCase):
    """`bets --gamer NAME`: one gamer's matches, their closing prices both ways, and the bets."""

    INFO = {"M1": {"MATCH_CODE": "M1", "SCHEDULED_START_TIME_UTC": "2026-10-06 19:36:00",
                   "PLAYER_1_HANDLE": "NIGHTMARE", "PLAYER_1_TEAM": "Buffalo Bills",
                   "PLAYER_2_HANDLE": "FUSE", "PLAYER_2_TEAM": "Denver Broncos"},
            "M2": {"MATCH_CODE": "M2", "SCHEDULED_START_TIME_UTC": "2026-10-06 18:24:00",
                   "PLAYER_1_HANDLE": "KILLJOY", "PLAYER_1_TEAM": "Washington Commanders",
                   "PLAYER_2_HANDLE": "Nightmare", "PLAYER_2_TEAM": "Minnesota Vikings"}}

    def test_closing_prices_are_the_last_before_the_first_play(self):
        row = lambda market, seconds, prob, msg, desc=None: ("M1", market, at(seconds), 100.0 * prob, None,
                                                             desc, msg, "open", "true")
        rows = [row(50, 0, 0.40, None), row(50, 10, 0.45, None),
                row(50, 20, 0.47, 3),                          # a pre-game message, before play 5
                row(50, 30, 0.80, 7),                          # in play
                row(52, 15, 0.52, None, "Line -3.5")]
        got = bets.closing_prices(bets.timeline(rows), {"M1": 5}, ["M1", "M2"])
        self.assertEqual(got["M1"][50], (0.47, None))
        self.assertEqual(got["M1"][52], (0.52, -3.5))
        self.assertNotIn("M2", got)
        self.assertEqual(bets.closing_prices(bets.timeline(rows), {}, ["M1"])["M1"][50], (0.45, None))

    def test_the_report_walks_the_gamer_s_matches_both_ways(self):
        def row(match, stake, rev, rev_c):
            return dict(match_code=match, stake=stake, revenue=rev, candidate_revenue=rev_c, simulated=True)
        v10 = [row("M1", 10, 4, 1), row("M1", 10, -10, -5), row("M2", 20, 20, 15)]
        glmer = [dict(r, candidate_revenue=c) for r, c in zip(v10, (2, -2, 5))]
        closing = {"prod": {"M1": {50: (0.47, None), 52: (0.51, 1.5), 54: (0.5, 40.5)},
                            "M2": {51: (0.60, None), 53: (0.5, -2.5)}},
                   "v10": {"M1": {50: (0.42, None)}}, "v10-glmer": {"M1": {50: (0.38, None)}}}
        lines = bets.gamer_report(["nightmare"], self.INFO, [("v10", v10), ("v10-glmer", glmer)],
                                  closing, {"M1": (14, 27), "M2": (30, 10)})
        text = "\n".join(lines)
        self.assertIn("NIGHTMARE: 2 matches, won 0, lost 2; his margin -33 points against prod's closing"
                      " spread lines summing to -1", text)
        self.assertLess(text.index("v KILLJOY"), text.index("v FUSE"))          # kick-off order
        self.assertIn("lost 14-27", text)
        self.assertIn("lost 10-30", text)                                       # his side first
        self.assertIn("prod      47.0%", text)
        self.assertIn("v10-glmer      38.0%", text)
        self.assertIn("prod   +1.5 51%", text)
        self.assertIn("book revenue prod -6, v10 -4, v10-glmer +0", text)
        self.assertIn("all 2: 3 bets re-priced, stake 40; book revenue prod +14 (+35.0%), v10 +11 (+27.5%),"
                      " v10-glmer +5 (+12.5%)", text)

    def test_operator_margins_are_the_median_over_prod(self):
        rows = [dict(implied_prob=0.55, stream_prob=0.50, on_prod_line=True, market="moneyline", in_play=True,
                     **{config.BET_GROUP_COLUMN: "FANDUEL"}) for _ in range(30)]
        rows += [dict(implied_prob=0.9, stream_prob=0.5, on_prod_line=False, market="moneyline", in_play=True,
                      **{config.BET_GROUP_COLUMN: "FANDUEL"})]
        text = "\n".join(bets.operator_margins(rows))
        self.assertIn("FANDUEL", text)
        self.assertIn("+10.0%", text)
        self.assertIn("30 bets", text)

    def test_bets_gamer_prices_only_his_matches(self):
        from .. import snowflake_io
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER",
                "OPERATOR_NAME", "BET_IN_PLAY", "CUSTOMER_TEMPERATURE"]
        rows = []
        for match in ("M1", "M3"):
            for (c, new), (_, old) in zip(PRICES[1:], PRICES):
                rows.append((len(rows), match, at(c + 7), 1, 1, odds_for(new), 10, 10, None, 2, "FANDUEL",
                             "Yes", "Standard"))
        info = [self.INFO["M1"], dict(self.INFO["M2"], MATCH_CODE="M3", PLAYER_2_HANDLE="ACE")]
        asked = []

        def quotes(cur, stream, matches):
            asked.append(list(matches))
            return MONEYLINE
        with mock.patch.object(bets, "fetch_all", return_value=(cols, rows)), \
                mock.patch.object(snowflake_io, "fetch_match_info", return_value=info), \
                mock.patch.object(snowflake_io, "fetch_quotes", side_effect=quotes), \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 17)}), \
                mock.patch.object(bets, "write_csv"), mock.patch.object(bets, "write_text") as text, \
                mock.patch("builtins.print"), mock.patch("os.makedirs"), \
                mock.patch.object(config, "LAG_RANGE", (-20, 30)), \
                mock.patch.object(bets, "fetch_checks", return_value=_checks_for(MONEYLINE)), \
                mock.patch.object(config, "CANDIDATES", ["MODEL:v10"]), mock.patch.object(bets, "check_models"), \
                mock.patch.object(config, "GAMERS", ["Nightmare"]):
            out = bets.run(None, "out")
        self.assertEqual({r["match_code"] for r in out}, {"M1"})
        self.assertTrue(all(m == ["M1"] for m in asked))
        written = {c[0][0]: c[0][1] for c in text.call_args_list}
        report = "\n".join(written[os.path.join("out", "bets_gamers.txt")])
        self.assertIn("NIGHTMARE: 1 matches, won 1, lost 0", report)
