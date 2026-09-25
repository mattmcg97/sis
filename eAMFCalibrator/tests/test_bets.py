"""The betting simulation: a lag per operator off the odds, the join, and the re-pricing."""

import datetime as dt
import unittest
from unittest import mock

from .. import bets, config

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
        self.assertEqual(bets.why_not(other), "line differs")

    def test_results_are_read_off_revenue_and_cash_outs_are_left_out(self):
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=0)), bets.PUSH)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=-10)), bets.WON)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=3)), bets.OTHER)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=10,
                                            BET_CASHED_OUT="Yes")), bets.OTHER)

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
            return ["V", "N"], [("x", 3)]
        with mock.patch.object(bets, "fetch_all", side_effect=fake_fetch):
            text = "\n".join(bets.probe(None))
        self.assertIn("FANDUEL", text)
        self.assertIn("configured columns missing: none", text)
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
                mock.patch.object(bets, "write_csv"), mock.patch("builtins.print"), \
                mock.patch("os.makedirs"), \
                mock.patch.object(config, "LAG_RANGE", (-20, 30)):
            out = bets.run(None, "out")
        self.assertEqual(len(out), 8)
        self.assertTrue(all(r["latency_seconds"] == 7 for r in out))
        self.assertTrue(all(r["simulated"] for r in out))


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

    def test_a_model_candidate_is_read_at_prods_lines(self):
        saved = config.STREAMS["candidate"]
        try:
            config.STREAMS["candidate"] = "MODEL:v6"
            self.assertEqual(bets.candidate_stream(), "MODEL:v6@prod")
            config.STREAMS["candidate"] = "GAMEPLAI_STREAM_CANDIDATE"
            self.assertEqual(bets.candidate_stream(), "GAMEPLAI_STREAM_CANDIDATE")
        finally:
            config.STREAMS["candidate"] = saved


if __name__ == "__main__":
    unittest.main()
