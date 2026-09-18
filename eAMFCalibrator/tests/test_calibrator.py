"""Unit tests for everything in the suite that does not need Snowflake.

The database half cannot be exercised here, so the logic that decides what
a row MEANS -- line parsing, market resolution, drive cleaning, quote
matching, scoring -- is pinned down in full instead.

Run with:  py -m unittest discover eAMFCalibrator
"""

import datetime as dt
import unittest

from .. import buckets, clock, config, markets, metrics
from ..drives import PlayRow, ScoreRow, build_snapshots, clean_plays, score_at
from ..pipeline import nearest_quote, to_unit_probability


class TestParseLine(unittest.TestCase):
    def test_spread_negative_line_not_confused_with_player_number(self):
        # "PLAYER 2" would otherwise be picked up as the line.
        desc = "PLAYER 2 to score over -2.5 points more than PLAYER 1"
        self.assertEqual(markets.parse_line(desc), -2.5)

    def test_spread_positive_line(self):
        desc = "PLAYER 1 to score over 3.5 points more than PLAYER 2"
        self.assertEqual(markets.parse_line(desc), 3.5)

    def test_total_line(self):
        self.assertEqual(markets.parse_line("Total points over 46.5"), 46.5)

    def test_moneyline_has_no_line(self):
        self.assertIsNone(markets.parse_line("Away team (PLAYER 2) to win"))

    def test_empty(self):
        self.assertIsNone(markets.parse_line(None))
        self.assertIsNone(markets.parse_line(""))


class TestResolve(unittest.TestCase):
    def test_moneyline(self):
        self.assertTrue(markets.resolve(50, None, 24, 17))
        self.assertFalse(markets.resolve(51, None, 24, 17))
        self.assertFalse(markets.resolve(50, None, 17, 24))
        self.assertTrue(markets.resolve(51, None, 17, 24))

    def test_moneyline_tie_is_push(self):
        self.assertIsNone(markets.resolve(50, None, 21, 21))
        self.assertIsNone(markets.resolve(51, None, 21, 21))

    def test_spread_home(self):
        # Home by 7; needs to beat -2.5, so it wins.
        self.assertTrue(markets.resolve(52, -2.5, 24, 17))
        # Home by 7; needs to beat +10.5, so it loses.
        self.assertFalse(markets.resolve(52, 10.5, 24, 17))

    def test_spread_away(self):
        # Away lost by 7, i.e. margin -7; beats -10.5.
        self.assertTrue(markets.resolve(53, -10.5, 24, 17))
        self.assertFalse(markets.resolve(53, -2.5, 24, 17))

    def test_spread_exact_line_is_push(self):
        self.assertIsNone(markets.resolve(52, 7, 24, 17))

    def test_totals(self):
        self.assertTrue(markets.resolve(54, 40.5, 24, 17))   # 41 > 40.5
        self.assertFalse(markets.resolve(55, 40.5, 24, 17))
        self.assertFalse(markets.resolve(54, 44.5, 24, 17))
        self.assertTrue(markets.resolve(55, 44.5, 24, 17))

    def test_totals_exact_line_is_push(self):
        self.assertIsNone(markets.resolve(54, 41, 24, 17))

    def test_missing_inputs(self):
        self.assertIsNone(markets.resolve(50, None, None, 17))
        self.assertIsNone(markets.resolve(54, None, 24, 17))  # totals need a line
        self.assertIsNone(markets.resolve(99, None, 24, 17))  # unknown market


class TestScoreDiffBuckets(unittest.TestCase):
    def test_edges(self):
        cases = {
            -30: "<= -9", -9: "<= -9",
            -8: "-8..-3", -3: "-8..-3",
            -2: "-2..+2", 0: "-2..+2", 2: "-2..+2",
            3: "+3..+8", 8: "+3..+8",
            9: ">= +9", 40: ">= +9",
        }
        for diff, expected in cases.items():
            self.assertEqual(buckets.score_diff_bucket(diff), expected, f"diff={diff}")

    def test_every_integer_lands_somewhere(self):
        for diff in range(-60, 61):
            self.assertNotEqual(buckets.score_diff_bucket(diff), "unknown", f"diff={diff}")

    def test_none(self):
        self.assertEqual(buckets.score_diff_bucket(None), "unknown")


class TestPeriodAndPossession(unittest.TestCase):
    def test_periods(self):
        self.assertEqual(buckets.period_bucket(1), "Q1")
        self.assertEqual(buckets.period_bucket(4), "Q4")
        self.assertEqual(buckets.period_bucket(5), "OT")
        self.assertEqual(buckets.period_bucket(7), "OT")
        self.assertEqual(buckets.period_bucket(None), "unknown")

    def test_possession(self):
        self.assertEqual(buckets.possession_bucket("Home Team"), "Home")
        self.assertEqual(buckets.possession_bucket("Away Team"), "Away")
        self.assertEqual(buckets.possession_bucket("nonsense"), "unknown")
        self.assertEqual(buckets.possession_bucket(None), "unknown")


def play(msg, team, down=1, dist=10, pos=25, period=1, ts=None):
    return PlayRow(event_message_count=msg, period_number=period, offensive_team=team,
                   down_number=down, distance=dist, field_position=pos, play_time=ts)


def score(msg, p1_change=0, p2_change=0, p1_cum=0, p2_cum=0, period=1):
    return ScoreRow(event_message_count=msg, period_number=period, p1_change=p1_change,
                    p2_change=p2_change, p1_cumulative=p1_cum, p2_cumulative=p2_cum)


class TestDriveCleaning(unittest.TestCase):
    def test_stale_row_after_team_change_is_dropped(self):
        plays = [
            play(1, "Home Team", 1, 10, 25),
            play(2, "Home Team", 2, 5, 30),
            play(3, "Away Team", 1, 10, 20),  # stale duplicate at the handoff
            play(4, "Away Team", 1, 10, 20),
            play(5, "Away Team", 2, 7, 23),
        ]
        cleaned, dropped = clean_plays(plays, [])
        self.assertEqual(dropped, 1)
        self.assertEqual([p.event_message_count for p in cleaned], [1, 2, 4, 5])

    def test_touchdown_resume_row_is_kept(self):
        # Home scores a TD at msg 2; Away's fresh 1st-and-10 at msg 3 is
        # genuine, not the stale duplicate the general rule would drop.
        plays = [
            play(1, "Home Team", 1, 10, 25),
            play(2, "Home Team", 2, 5, 30),
            play(3, "Away Team", 1, 10, 20),
            play(4, "Away Team", 2, 7, 23),
        ]
        scores = [score(2, p1_change=6, p1_cum=6)]
        cleaned, dropped = clean_plays(plays, scores)
        self.assertEqual(dropped, 0)
        self.assertEqual([p.event_message_count for p in cleaned], [1, 2, 3, 4])

    def test_snapshots_one_per_drive_at_first_play(self):
        plays = [
            play(1, "Home Team", 1, 10, 25),
            play(2, "Home Team", 2, 5, 30),
            play(3, "Away Team", 1, 10, 20),
            play(4, "Away Team", 2, 7, 23),
        ]
        scores = [score(2, p1_change=6, p1_cum=6)]
        snaps = build_snapshots("AF1", plays, scores)
        self.assertEqual(len(snaps), 2)
        self.assertEqual(snaps[0].drive_number, 1)
        self.assertEqual(snaps[0].event_message_count, 1)
        self.assertEqual(snaps[0].offensive_team, "Home Team")
        self.assertEqual(snaps[0].n_plays, 2)
        self.assertEqual(snaps[1].drive_number, 2)
        self.assertEqual(snaps[1].event_message_count, 3)
        self.assertEqual(snaps[1].n_plays, 2)

    def test_snapshot_carries_score_as_of_its_own_message(self):
        plays = [
            play(1, "Home Team"),
            play(3, "Away Team"),
            play(4, "Away Team"),
        ]
        scores = [score(2, p1_change=6, p1_cum=6, p2_cum=0)]
        snaps = build_snapshots("AF1", plays, scores)
        # Drive 1 starts before the TD, drive 2 after it.
        self.assertEqual((snaps[0].score_p1, snaps[0].score_p2), (0, 0))
        self.assertEqual(snaps[0].score_diff, 0)
        self.assertEqual((snaps[1].score_p1, snaps[1].score_p2), (6, 0))
        self.assertEqual(snaps[1].score_diff, 6)

    def test_score_at_boundaries(self):
        scores = [score(5, p1_cum=7, p2_cum=0), score(9, p1_cum=7, p2_cum=3)]
        self.assertEqual(score_at(scores, 1), (0, 0))
        self.assertEqual(score_at(scores, 5), (7, 0))
        self.assertEqual(score_at(scores, 8), (7, 0))
        self.assertEqual(score_at(scores, 9), (7, 3))
        self.assertEqual(score_at(scores, 99), (7, 3))

    def test_empty_inputs(self):
        self.assertEqual(build_snapshots("AF1", [], []), [])


BASE = dt.datetime(2026, 9, 17, 12, 0, 0)


class TestNearestQuote(unittest.TestCase):
    def setUp(self):
        self.times = [BASE + dt.timedelta(seconds=s) for s in (-10, -2, 1, 6)]

    def test_nearest_picks_closest_either_side(self):
        hit = nearest_quote(self.times, BASE, 3.0, "nearest")
        self.assertIsNotNone(hit)
        idx, gap = hit
        self.assertEqual(idx, 2)          # +1s beats -2s
        self.assertAlmostEqual(gap, 1.0)

    def test_nearest_can_pick_a_quote_before_the_snapshot(self):
        target = BASE + dt.timedelta(seconds=-3)
        idx, gap = nearest_quote(self.times, target, 3.0, "nearest")
        self.assertEqual(idx, 1)          # -2s quote, 1s before target
        self.assertAlmostEqual(gap, 1.0)

    def test_forward_only_ignores_earlier_quotes(self):
        idx, gap = nearest_quote(self.times, BASE, 3.0, "forward")
        self.assertEqual(idx, 2)
        self.assertAlmostEqual(gap, 1.0)

    def test_forward_only_returns_none_when_next_quote_is_too_late(self):
        target = BASE + dt.timedelta(seconds=2)
        self.assertIsNone(nearest_quote(self.times, target, 3.0, "forward"))

    def test_tolerance_is_enforced(self):
        target = BASE + dt.timedelta(seconds=-6)
        self.assertIsNone(nearest_quote(self.times, target, 3.0, "nearest"))

    def test_tolerance_boundary_is_inclusive(self):
        target = BASE + dt.timedelta(seconds=-5)
        idx, gap = nearest_quote(self.times, target, 3.0, "nearest")
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(gap, 3.0)

    def test_empty_and_null(self):
        self.assertIsNone(nearest_quote([], BASE, 3.0, "nearest"))
        self.assertIsNone(nearest_quote(self.times, None, 3.0, "nearest"))


class TestProbabilityScale(unittest.TestCase):
    def test_percent_to_unit(self):
        self.assertAlmostEqual(to_unit_probability(85.26), 0.8526)
        self.assertAlmostEqual(to_unit_probability(0), 0.0)
        self.assertIsNone(to_unit_probability(None))


class TestMetrics(unittest.TestCase):
    def test_brier_perfect_and_worst(self):
        self.assertAlmostEqual(metrics.brier_score([(1.0, True), (0.0, False)]), 0.0)
        self.assertAlmostEqual(metrics.brier_score([(0.0, True), (1.0, False)]), 1.0)

    def test_brier_known_value(self):
        self.assertAlmostEqual(metrics.brier_score([(0.7, True), (0.3, False)]), 0.09)

    def test_log_loss_clips_instead_of_exploding(self):
        value = metrics.log_loss([(0.0, True)])
        self.assertTrue(value < float("inf"))
        self.assertGreater(value, 10)

    def test_ece_zero_when_perfectly_calibrated(self):
        # Ten rows at p=0.5, five of which win.
        pairs = [(0.5, True)] * 5 + [(0.5, False)] * 5
        self.assertAlmostEqual(metrics.expected_calibration_error(pairs, 10), 0.0)

    def test_ece_detects_miscalibration(self):
        pairs = [(0.9, False)] * 10
        self.assertAlmostEqual(metrics.expected_calibration_error(pairs, 10), 0.9)

    def test_summarize_gap_sign(self):
        # Model says 30%, reality is 50% -> underpriced, positive gap.
        pairs = [(0.3, True)] * 5 + [(0.3, False)] * 5
        s = metrics.summarize(pairs, 10)
        self.assertEqual(s["n"], 10)
        self.assertAlmostEqual(s["mean_predicted"], 0.3)
        self.assertAlmostEqual(s["realized"], 0.5)
        self.assertAlmostEqual(s["gap"], 0.2)

    def test_empty(self):
        self.assertIsNone(metrics.brier_score([]))
        self.assertEqual(metrics.summarize([], 10)["n"], 0)


class TestMatchClock(unittest.TestCase):
    def setUp(self):
        # Stream quoted messages 10, 12 and 30. Message 11 sits in a tight
        # bracket; 20 sits in a 18-wide one.
        self.clock = clock.MatchClock({
            10: BASE,
            12: BASE + dt.timedelta(seconds=4),
            30: BASE + dt.timedelta(seconds=40),
        })

    def test_exact_message_needs_no_estimation(self):
        when, provenance = self.clock.time_for(12)
        self.assertEqual(provenance, clock.EXACT)
        self.assertEqual(when, BASE + dt.timedelta(seconds=4))

    def test_interpolates_inside_a_tight_bracket(self):
        when, provenance = self.clock.time_for(11)
        self.assertEqual(provenance, clock.INTERPOLATED)
        self.assertEqual(when, BASE + dt.timedelta(seconds=2))

    def test_bracket_wider_than_limit_is_unresolved(self):
        when, provenance = self.clock.time_for(20, max_bracket=10)
        self.assertIsNone(when)
        self.assertEqual(provenance, clock.UNRESOLVED)

    def test_wide_bracket_allowed_when_limit_raised(self):
        when, provenance = self.clock.time_for(20, max_bracket=50)
        self.assertEqual(provenance, clock.INTERPOLATED)
        self.assertEqual(when, BASE + dt.timedelta(seconds=4 + 36 * (8 / 18)))

    def test_outside_the_streams_range_is_unresolved(self):
        for message in (5, 99):
            when, provenance = self.clock.time_for(message)
            self.assertIsNone(when, f"message={message}")
            self.assertEqual(provenance, clock.UNRESOLVED)

    def test_empty_clock_and_null_message(self):
        empty = clock.MatchClock({})
        self.assertEqual(empty.time_for(10), (None, clock.UNRESOLVED))
        self.assertEqual(self.clock.time_for(None), (None, clock.UNRESOLVED))

    def test_build_clocks_takes_earliest_time_per_message(self):
        # One message carries a row per market, published together.
        rows = [
            ("AF1", 7, BASE + dt.timedelta(seconds=1)),
            ("AF1", 7, BASE),
            ("AF1", 8, BASE + dt.timedelta(seconds=2)),
            ("AF2", 7, BASE + dt.timedelta(seconds=9)),
        ]
        clocks = clock.build_clocks(rows)
        self.assertEqual(set(clocks), {"AF1", "AF2"})
        self.assertEqual(clocks["AF1"].time_for(7), (BASE, clock.EXACT))
        self.assertEqual(len(clocks["AF1"]), 2)

    def test_build_clocks_skips_null_rows(self):
        clocks = clock.build_clocks([("AF1", None, BASE), ("AF1", 3, None), ("AF1", 4, BASE)])
        self.assertEqual(len(clocks["AF1"]), 1)


class TestConfigSanity(unittest.TestCase):
    def test_score_buckets_are_contiguous_and_ordered(self):
        edges = config.SCORE_DIFF_BUCKETS
        self.assertIsNone(edges[0][0])
        self.assertIsNone(edges[-1][1])
        for (_, high, _), (low_next, _, _) in zip(edges, edges[1:]):
            self.assertEqual(low_next, high + 1)

    def test_every_market_id_resolves_to_a_group(self):
        for market_id in markets.MARKET_IDS:
            self.assertIn(markets.market_group(market_id),
                          {markets.MONEYLINE, markets.SPREAD, markets.TOTAL})
            self.assertIsNotNone(markets.selection_label(market_id))


if __name__ == "__main__":
    unittest.main()
