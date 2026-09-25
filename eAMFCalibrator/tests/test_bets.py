"""The betting simulation: latency off the bets' lines, the join, and the re-pricing."""

import datetime as dt
import unittest
from unittest import mock

from .. import bets, config

T0 = dt.datetime(2026, 9, 20, 12, 0, 0)


def at(seconds):
    return T0 + dt.timedelta(seconds=seconds)


def quote(match, market, seconds, prob, line=None, message=None, status="open", active="true"):
    """A snowflake_io.fetch_quotes row, published at T0 + seconds (GAMEPLAI publishes 0-100)."""
    desc = None if line is None else f"Total {line}"
    return (match, market, at(seconds), 100.0 * prob, None, desc,
            seconds if message is None else message, status, active)


def bet(match, seconds, market_type=3, selection=1, odds=1.9, stake=10.0, revenue=10.0,
        line=None, operator="FANDUEL", **extra):
    return bets.Bet(bet_id=f"{match}-{seconds}", match_code=match, time=at(seconds),
                    market_type=market_type, selection=selection, odds=odds, stake=stake,
                    revenue=revenue, line=line,
                    extra=dict({config.BET_GROUP_COLUMN: operator}, **extra))


# prod's total: 44.5 until 100s, 47.5 until 200s, then 41.5
TOTALS = [quote("M1", 54, 0, 0.50, 44.5), quote("M1", 54, 100, 0.52, 47.5),
          quote("M1", 54, 200, 0.48, 41.5)]


class TestLatency(unittest.TestCase):

    def test_the_lag_is_where_the_bets_lines_agree_with_prods(self):
        tl = bets.timeline(TOTALS)
        # placed 8s behind prod: still on the old line just after each change (a quote published
        # exactly at bet time - lag is already showing, so 207s on 47.5 means a lag of 8 or more)
        bs = [bet("M1", 105, line=44.5), bet("M1", 107, line=44.5), bet("M1", 150, line=47.5),
              bet("M1", 207, line=47.5), bet("M1", 230, line=41.5)]
        lat = bets.fit_latency(bs, tl, max_lag=60, step=1, min_bets=2)[("M1", "FANDUEL")]
        self.assertEqual((lat.seconds, lat.source), (8, "lines"))
        self.assertEqual(lat.agree, 1.0)
        self.assertLess(lat.agree_no_lag, 1.0)

    def test_a_moneyline_group_is_fitted_on_the_odds(self):
        tl = bets.timeline([quote("M2", 50, 0, 0.50), quote("M2", 50, 100, 0.70)])
        margin = 1.05
        bs = [bet("M2", 102, 1, 1, odds=1 / (0.50 * margin)),
              bet("M2", 103, 1, 1, odds=1 / (0.50 * margin)),
              bet("M2", 110, 1, 1, odds=1 / (0.70 * margin)),
              bet("M2", 130, 1, 1, odds=1 / (0.70 * margin))]
        lat = bets.fit_latency(bs, tl, max_lag=30, step=1, min_bets=2)[("M2", "FANDUEL")]
        self.assertEqual((lat.seconds, lat.source), (4, "odds"))

    def test_a_thin_group_takes_its_operators_median(self):
        tl = bets.timeline(TOTALS)
        bs = [bet("M1", 105, line=44.5), bet("M1", 107, line=44.5), bet("M1", 150, line=47.5),
              bet("M1", 207, line=47.5), bet("M1", 230, line=41.5),
              bet("M9", 10, line=44.5)]
        lat = bets.fit_latency(bs, tl, max_lag=60, step=1, min_bets=2)
        self.assertEqual((lat[("M9", "FANDUEL")].seconds, lat[("M9", "FANDUEL")].source),
                         (8, "operator median"))


class TestJoin(unittest.TestCase):

    def setUp(self):
        self.prod = bets.timeline([quote("M1", 54, 0, 0.50, 44.5, message=10),
                                   quote("M1", 54, 40, 0.40, 44.5, message=12)])
        self.cand = bets.quote_index([quote("M1", 54, 1, 0.55, 44.5, message=10),
                                      quote("M1", 54, 41, 0.45, 44.5, message=12)])

    def test_a_bet_is_priced_at_prods_quote_one_latency_earlier(self):
        lat = {("M1", "FANDUEL"): bets.Latency(15.0, "lines")}
        (r,) = bets.join([bet("M1", 50, line=44.5, odds=2.2)], lat, self.prod, self.cand)
        self.assertEqual(r["message"], 10)
        self.assertEqual((r["stream_prob"], r["candidate_prob"]), (0.50, 0.55))
        self.assertEqual(r["stream_prob_no_lag"], 0.40)
        self.assertTrue(r["line_match"])

    def test_a_loser_keeps_its_revenue_and_a_winner_is_repaid_at_the_candidates_odds(self):
        lat = {("M1", "FANDUEL"): bets.Latency(0.0, "lines")}
        lost, won = bets.join([bet("M1", 45, line=44.5, odds=2.4, stake=10, revenue=10),
                               bet("M1", 45, line=44.5, odds=2.4, stake=10, revenue=-14)],
                              lat, self.prod, self.cand)
        self.assertEqual((lost["result"], won["result"]), (bets.LOST, bets.WON))
        self.assertEqual(lost["candidate_revenue"], 10)
        odds_c = 2.4 * 0.40 / 0.45
        self.assertAlmostEqual(won["candidate_odds"], odds_c, places=4)
        self.assertAlmostEqual(won["candidate_revenue"], 10 - 10 * odds_c)

    def test_a_bet_on_another_line_is_not_simulated(self):
        (r,) = bets.join([bet("M1", 45, line=47.5)], {}, self.prod, self.cand)
        self.assertFalse(r["line_match"] or r["simulated"])

    def test_results_are_read_off_revenue_and_cash_outs_are_left_out(self):
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=0)), bets.PUSH)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=-10)), bets.WON)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=3)), bets.OTHER)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=10,
                                            BET_CASHED_OUT="Yes")), bets.OTHER)

    def test_the_summary_is_margin_both_ways_and_can_leave_out_the_vips(self):
        lat = {("M1", "FANDUEL"): bets.Latency(0.0, "lines")}
        rows = bets.join([bet("M1", 45, line=44.5, odds=2.4, stake=10, revenue=10,
                              CUSTOMER_TEMPERATURE="Standard"),
                          bet("M1", 45, line=44.5, odds=2.4, stake=10, revenue=-14,
                              CUSTOMER_TEMPERATURE="VIP")],
                         lat, self.prod, self.cand)
        n, stake, rev, margin, rev_c, margin_c = bets.summarise(rows)["all"]
        self.assertEqual((n, stake, rev), (2, 20.0, -4.0))
        self.assertAlmostEqual(margin, -20.0)
        self.assertGreater(margin_c, margin)
        no_vip = bets.summarise(rows, keep=lambda r: r["CUSTOMER_TEMPERATURE"] != "VIP")["all"]
        self.assertEqual(no_vip[:3], (1, 10.0, 10.0))


class TestLines(unittest.TestCase):

    def test_the_line_report_tells_a_turned_spread_from_an_alternate_total(self):
        rows = [dict(market="spread", bet_line=3.5, stream_line=-3.5, OPERATOR_NAME="FD"),
                dict(market="spread", bet_line=-3.5, stream_line=-3.5, OPERATOR_NAME="FD"),
                dict(market="total", bet_line=45.5, stream_line=44.5, OPERATOR_NAME="FD"),
                dict(market="moneyline", bet_line=None, stream_line=None, OPERATOR_NAME="FD")]
        report = {(op, m): (n, same, turned, gaps) for op, m, n, same, turned, gaps in
                  bets.line_report(rows)}
        self.assertEqual(report[("FD", "spread")][:3], (2, 0.5, 0.5))
        self.assertEqual(report[("FD", "total")][3], [(1.0, 1)])
        self.assertNotIn(("FD", "moneyline"), report)

    def test_probabilities_are_read_as_0_to_1(self):
        tl = bets.timeline([quote("M1", 50, 0, 0.62)])
        self.assertAlmostEqual(bets.price_at_time(tl, "M1", 50, at(5))[1], 0.62)


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
                "OPERATOR_NAME", "CUSTOMER_TEMPERATURE"]
        rows = [(i, "M1", at(t), 3, 1, 1.9, 10, 10, line, 2, "FANDUEL", "Standard")
                for i, (t, line) in enumerate(((105, 44.5), (107, 44.5), (150, 47.5),
                                               (207, 47.5), (230, 41.5)))]
        with mock.patch.object(bets, "fetch_all", return_value=(cols, rows)), \
                mock.patch.object(snowflake_io, "fetch_quotes", return_value=TOTALS), \
                mock.patch.object(bets, "write_csv"), mock.patch("builtins.print"), \
                mock.patch("os.makedirs"):
            out = bets.run(None, "out")
        self.assertEqual(len(out), 5)
        self.assertTrue(all(r["latency_seconds"] == 8 for r in out))
        self.assertTrue(all(r["simulated"] for r in out))


class TestReading(unittest.TestCase):

    def test_bets_are_read_through_the_configured_columns(self):
        cols = ["MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID", "ODDS", "STAKE_GBP",
                "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER", "OPERATOR_NAME"]
        row = ("M1", T0, 2, 2, 1.95, 5, -4.75, -3.5, 3, "HARDROCK")
        (b,) = bets.to_bets(cols, [row])
        self.assertEqual((b.feed_market, b.line, b.group), (53, -3.5, ("M1", "HARDROCK")))

    def test_the_query_reads_in_play_singles_in_the_window(self):
        sql, params, wanted = bets.bets_sql()
        self.assertIn("MARKET_TYPE_ID IN (1, 2, 3)", sql)
        self.assertIn("UPPER(BET_TYPE) = 'SINGLE'", sql)
        self.assertIn("BET_IN_PLAY = 'Yes'", sql)
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
