"""The betting simulation: latency off the Q4 auto-suspend, the join, and the re-pricing."""

import datetime as dt
import unittest

from .. import bets, config

T0 = dt.datetime(2026, 9, 20, 12, 0, 0)


def at(seconds):
    return T0 + dt.timedelta(seconds=seconds)


def feed_row(match, message, seconds, status=None, clock=None):
    """A scouting.fetch_scouting row: MATCH_CODE, EVENT_MESSAGE_COUNT, CLOCK, STATUS, MESSAGE,
    TEAM, DOWN, DIST, FIELD, FILE_TIME."""
    return (match, message, clock, status, None, None, None, None, None, at(seconds))


def bet(match, seconds, market_type=3, selection=1, odds=1.9, stake=10.0, revenue=10.0,
        line=44.5, period=4):
    return bets.Bet(bet_id=f"{match}-{seconds}", match_code=match, time=at(seconds),
                    market_type=market_type, selection=selection, odds=odds, stake=stake,
                    revenue=revenue, line=line, period=period)


def quote(match, market, message, prob, desc="Total 44.5", status="open", active="true"):
    """A snowflake_io.fetch_quotes row."""
    return (match, market, at(message), prob, None, desc, message, status, active)


FEED = [
    feed_row("M1", 1, 0, status="FOURTH_QUARTER_STARTED", clock=240),
    feed_row("M1", 2, 50, status="BET_SUSPEND", clock=200),
    feed_row("M1", 3, 60, status="BET_UNSUSPEND", clock=199),
    feed_row("M1", 4, 180, status="BET_SUSPEND", clock=121),
    feed_row("M2", 1, 0, status="THIRD_QUARTER_STARTED", clock=240),
    feed_row("M2", 2, 100, status="BET_SUSPEND", clock=118),
]


class TestLatency(unittest.TestCase):

    def test_the_auto_suspend_is_the_first_q4_suspend_near_two_minutes(self):
        found = bets.auto_suspend_times(FEED, clock=120, slack=10)
        self.assertEqual(found, {"M1": at(180)})

    def test_latency_is_the_latest_bet_accepted_after_it(self):
        bs = [bet("M1", 170), bet("M1", 183), bet("M1", 187.5), bet("M1", 400), bet("M3", 10)]
        lat = bets.match_latency(bs, {"M1": at(180)}, max_lag=60)
        self.assertEqual(lat["M1"].seconds, 7.5)
        self.assertEqual((lat["M1"].source, lat["M1"].late_bets), ("suspend", 2))
        self.assertEqual((lat["M3"].seconds, lat["M3"].source), (7.5, "median"))

    def test_no_measured_match_means_no_lag(self):
        lat = bets.match_latency([bet("M1", 10)], {}, max_lag=60)
        self.assertEqual(lat["M1"].seconds, 0.0)


class TestJoin(unittest.TestCase):

    def setUp(self):
        self.times = bets.message_times([feed_row("M1", m, s) for m, s in
                                         ((10, 0), (11, 20), (12, 40), (13, 60))])
        self.prod = bets.quote_index([quote("M1", 54, 10, 0.50), quote("M1", 54, 12, 0.40)])
        self.cand = bets.quote_index([quote("M1", 54, 10, 0.55), quote("M1", 54, 12, 0.45)])

    def test_a_bet_is_priced_at_the_message_live_one_latency_earlier(self):
        lat = {"M1": bets.Latency(15.0, "suspend")}
        rows = bets.join([bet("M1", 50, odds=2.2, revenue=10.0)], lat, self.times, self.prod, self.cand)
        r = rows[0]
        self.assertEqual(r["message"], 11)
        self.assertEqual((r["stream_prob"], r["candidate_prob"]), (0.50, 0.55))
        self.assertEqual(r["stream_prob_no_lag"], 0.40)
        self.assertTrue(r["line_match"])

    def test_a_losing_bet_keeps_its_revenue_and_a_winner_is_repaid_at_the_candidates_odds(self):
        lat = {"M1": bets.Latency(0.0, "suspend")}
        lost, won = bets.join([bet("M1", 45, odds=2.4, stake=10, revenue=10),
                               bet("M1", 45, odds=2.4, stake=10, revenue=-14)],
                              lat, self.times, self.prod, self.cand)
        self.assertEqual((lost["result"], won["result"]), (bets.LOST, bets.WON))
        self.assertEqual(lost["candidate_revenue"], 10)
        odds_c = 2.4 * 0.40 / 0.45
        self.assertAlmostEqual(won["candidate_odds"], odds_c, places=4)
        self.assertAlmostEqual(won["candidate_revenue"], 10 - 10 * odds_c)

    def test_a_bet_on_another_line_is_not_simulated(self):
        rows = bets.join([bet("M1", 45, line=47.5)], {}, self.times, self.prod, self.cand)
        self.assertFalse(rows[0]["line_match"] or rows[0]["simulated"])

    def test_results_are_read_off_revenue(self):
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=0)), bets.PUSH)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=-10)), bets.WON)
        self.assertEqual(bets.result_of(bet("M1", 0, odds=2.0, stake=10, revenue=3)), bets.OTHER)

    def test_the_summary_is_margin_both_ways(self):
        lat = {"M1": bets.Latency(0.0, "suspend")}
        rows = bets.join([bet("M1", 45, odds=2.4, stake=10, revenue=10),
                          bet("M1", 45, odds=2.4, stake=10, revenue=-14)],
                         lat, self.times, self.prod, self.cand)
        n, stake, rev, margin, rev_c, margin_c = bets.summarise(rows)["all"]
        self.assertEqual((n, stake, rev), (2, 20.0, -4.0))
        self.assertAlmostEqual(margin, -20.0)
        self.assertGreater(margin_c, margin)


class TestReading(unittest.TestCase):

    def test_bets_are_read_through_the_configured_columns(self):
        cols = ["MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID", "ODDS", "STAKE_GBP",
                "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER", "BET_TYPE"]
        row = ("M1", T0, 2, 2, 1.95, 5, -4.75, -3.5, 3, "SINGLE")
        (b,) = bets.to_bets(cols, [row])
        self.assertEqual((b.feed_market, b.line, b.extra), (53, -3.5, {"BET_TYPE": "SINGLE"}))

    def test_the_query_reads_the_window_and_in_play_markets(self):
        sql, params, wanted = bets.bets_sql()
        self.assertIn(config.BET_TABLE, sql)
        self.assertIn("MARKET_TYPE_ID IN (1, 2, 3)", sql)
        self.assertIn(config.SPORT_CODE, params)

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
