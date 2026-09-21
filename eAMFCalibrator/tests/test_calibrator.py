"""Unit tests for everything in the suite that does not need Snowflake.

The database half cannot be exercised here, so the logic that decides what
a row MEANS -- line parsing, market resolution, drive cleaning, quote
matching, scoring -- is pinned down in full instead.

Run with:  py -m unittest discover eAMFCalibrator
"""

import collections
import contextlib
import datetime as dt
import io
import unittest

from .. import (buckets, clock, config, directional, handles, markets,
                metrics, report)
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


def clean_scan(n_matches):
    """A handle scan over `n_matches` matches that all look stable."""
    scan = handles.Scan()
    for i in range(n_matches):
        scan.add(f"AF{i}", [ScoreRow(1, 1, 7, 0, 7, 0)], (7, 0))
    return scan.summary()


class TestHandleCheck(unittest.TestCase):
    """PLAYER_1 / PLAYER_2 staying pinned to one team each."""

    @staticmethod
    def rows(*cumulatives):
        """ScoreRows from a sequence of (p1, p2) cumulative totals."""
        return [ScoreRow(i + 1, 1, None, None, p1, p2)
                for i, (p1, p2) in enumerate(cumulatives)]

    def kinds(self, *cumulatives, final=None):
        return [a.kind for a in
                handles.scan_match("AF1", self.rows(*cumulatives), final)]

    def test_a_normal_match_flags_nothing(self):
        self.assertEqual(self.kinds((7, 0), (7, 7), (14, 7), (14, 14)), [])

    def test_an_exact_swap_is_a_mirror(self):
        self.assertEqual(self.kinds((14, 7), (7, 14)), [handles.MIRROR])

    def test_a_swap_landing_on_a_score_is_a_regression(self):
        # 21-7 crossing over while the other side scores a touchdown: the
        # totals do not trade places exactly, but one still went down.
        self.assertEqual(self.kinds((21, 7), (7, 28)), [handles.REGRESSION])

    def test_a_rescinded_score_is_a_regression_not_a_mirror(self):
        # A touchdown overturned on review. Worth flagging, but it is not a
        # swap, and the kind has to say so.
        self.assertEqual(self.kinds((14, 7), (7, 7)), [handles.REGRESSION])

    def test_a_level_score_cannot_reveal_a_swap(self):
        # Both sides on 7: a swap here leaves the totals identical. This is
        # the documented blind spot, asserted so it stays documented.
        self.assertEqual(self.kinds((7, 7), (7, 7), (7, 14)), [])

    def test_sparse_cumulatives_carry_forward(self):
        # Only the scoring side reports a cumulative; the other arrives as
        # NULL and must hold its value rather than reading as a drop to 0.
        rows = [ScoreRow(1, 1, 7, None, 7, None),
                ScoreRow(2, 1, None, 7, None, 7),
                ScoreRow(3, 1, 7, None, 14, None)]
        self.assertEqual([a.kind for a in handles.scan_match("AF1", rows)], [])

    def test_rows_are_scanned_in_message_order(self):
        rows = list(reversed(self.rows((7, 0), (14, 0), (21, 0))))
        self.assertEqual([a.kind for a in handles.scan_match("AF1", rows)], [])

    def test_a_match_crossed_from_the_first_message_is_caught_at_the_final(self):
        # Nothing regresses -- the whole match is simply mirrored -- so the
        # only evidence is SCORE_ENDGAME disagreeing in mirror image.
        self.assertEqual(self.kinds((0, 7), (0, 14), final=(14, 0)),
                         [handles.FINAL_MIRRORED])

    def test_a_plain_final_disagreement_is_not_called_a_flip(self):
        # A missing late score explains this as well as a swap does.
        kinds = self.kinds((7, 0), final=(14, 0))
        self.assertEqual(kinds, [handles.FINAL_MISMATCH])
        self.assertNotIn(handles.FINAL_MISMATCH, handles.FLIP_KINDS)

    def test_a_level_final_is_never_read_as_mirrored(self):
        # 14-14 mirrored is still 14-14, so a disagreement at a level score
        # carries no directional evidence.
        self.assertEqual(self.kinds((14, 14), final=(21, 21)),
                         [handles.FINAL_MISMATCH])

    def test_an_unsettled_match_skips_the_final_check(self):
        self.assertEqual(self.kinds((7, 0), final=None), [])
        self.assertEqual(self.kinds((7, 0), final=(None, None)), [])

    def test_the_anomaly_names_the_message_and_both_scores(self):
        found = handles.scan_match("AF9", self.rows((14, 7), (7, 14)))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].match_code, "AF9")
        self.assertEqual(found[0].event_message_count, 2)
        self.assertEqual(found[0].describe(), "14-7 -> 7-14")
        self.assertTrue(found[0].is_flip)


class TestHandleScanSummary(unittest.TestCase):
    def setUp(self):
        self.scan = handles.Scan()
        self.scan.add("CLEAN1", TestHandleCheck.rows((7, 0), (7, 7)), (7, 7))
        self.scan.add("CLEAN2", TestHandleCheck.rows((3, 0)), (3, 0))
        self.scan.add("SWAP", TestHandleCheck.rows((14, 7), (7, 14)), (7, 14))
        self.scan.add("LATE", TestHandleCheck.rows((7, 0)), (10, 0))
        self.summary = self.scan.summary()

    def test_add_reports_whether_the_match_is_usable(self):
        scan = handles.Scan()
        self.assertTrue(scan.add("OK", TestHandleCheck.rows((7, 0))))
        self.assertFalse(scan.add("BAD", TestHandleCheck.rows((14, 7), (7, 14))))

    def test_counts_separate_flagged_from_flipped(self):
        # LATE is flagged (its final disagrees) but not flipped.
        self.assertEqual(self.summary["matches"], 4)
        self.assertEqual(self.summary["matches_flagged"], 2)
        self.assertEqual(self.summary["matches_flipped"], 1)
        self.assertEqual(self.summary["matches_clean"], 2)

    def test_only_flipped_matches_are_offered_for_exclusion(self):
        self.assertEqual(self.summary["flipped_matches"], ["SWAP"])
        self.assertEqual(self.scan.flipped_matches, frozenset({"SWAP"}))

    def test_counts_are_broken_out_by_kind(self):
        self.assertEqual(self.summary["counts"][handles.MIRROR], 1)
        self.assertEqual(self.summary["counts"][handles.FINAL_MISMATCH], 1)
        self.assertEqual(self.summary["counts"][handles.REGRESSION], 0)

    def test_the_scan_accumulates_across_chunks(self):
        # build_pairs is called once per chunk with one shared Scan, so a
        # second chunk must add to the first rather than replace it.
        self.scan.add("SWAP2", TestHandleCheck.rows((21, 0), (0, 21)))
        self.assertEqual(self.scan.summary()["matches"], 5)
        self.assertEqual(self.scan.summary()["matches_flipped"], 2)

    def test_where_distinguishes_the_three_cases(self):
        scores = [ScoreRow(1, 1, None, None, 14, 7),
                  ScoreRow(2, 1, None, None, 7, 14)]
        self.assertEqual(handles.scan_match("X", scores)[0].where, "2")
        late = handles.scan_match("X", [ScoreRow(1, 1, None, None, 7, 0)], (14, 0))
        self.assertEqual(late[0].where, "final")
        plays, clean = TestPossessionCrossCheck().feed([True, False, True, False])
        crossed = [TestPossessionCrossCheck.td(s.event_message_count,
                                               not bool(s.p1_change))
                   for s in clean]
        verdict, _ = handles.possession_verdict("X", plays, crossed)
        self.assertEqual(verdict.where, "match")
        self.assertFalse(verdict.is_final_check)

    def test_empty_summary_matches_the_real_shape(self):
        self.assertEqual(set(handles.empty_summary()), set(self.summary))


class TestFlippedMatchExclusion(unittest.TestCase):
    """build_pairs consulting the scan before it trusts a match.

    Patched at the fetch boundary rather than mocked at the cursor, so the
    real pairing code runs -- the exclusion branch is the point of the
    check and wants exercising, not stubbing.
    """

    MATCHES = ["CLEAN", "SWAP"]

    def feed(self):
        plays = []
        scores = []
        for match in self.MATCHES:
            for msg, team in ((1, "Home Team"), (2, "Away Team")):
                plays.append((match, msg, 1, team, 1, 10, 25, None))
        # CLEAN climbs; SWAP trades its totals on the second message. Both
        # finish decided, so the moneyline resolves rather than pushing.
        scores.extend([("CLEAN", 1, 1, 7, None, 7, 0),
                       ("CLEAN", 2, 1, 7, 7, 14, 7)])
        scores.extend([("SWAP", 1, 1, 14, None, 14, 7),
                       ("SWAP", 2, 1, None, None, 7, 14)])
        return plays, scores

    def quotes(self):
        rows = []
        for match in self.MATCHES:
            for msg in (1, 2):
                rows.append((match, 50, dt.datetime(2026, 9, 18, 12, msg),
                             60.0, 1.67, "PLAYER 1", msg))
        return rows

    @contextlib.contextmanager
    def patched(self):
        plays, scores = self.feed()
        quotes = self.quotes()
        io = directional.snowflake_io
        originals = {name: getattr(io, name) for name in
                     ("fetch_plays", "fetch_scores", "fetch_final_scores",
                      "fetch_quotes")}
        io.fetch_plays = lambda cur, m, t: plays
        io.fetch_scores = lambda cur, m: scores
        io.fetch_final_scores = lambda cur, m: {"CLEAN": (14, 7), "SWAP": (7, 14)}
        io.fetch_quotes = lambda cur, table, m: quotes
        try:
            yield
        finally:
            for name, function in originals.items():
                setattr(io, name, function)

    def build(self, exclude):
        scan = handles.Scan()
        stats = collections.defaultdict(int)
        previous = config.EXCLUDE_FLIPPED_MATCHES
        config.EXCLUDE_FLIPPED_MATCHES = exclude
        try:
            with self.patched():
                pairs = directional.build_pairs(
                    None, self.MATCHES, "PLAY_TIME", stats, scan)
        finally:
            config.EXCLUDE_FLIPPED_MATCHES = previous
        return pairs, stats, scan

    def test_the_flip_is_found_and_counted_either_way(self):
        for exclude in (False, True):
            _, stats, scan = self.build(exclude)
            self.assertEqual(scan.flipped_matches, frozenset({"SWAP"}),
                             f"exclude={exclude}")
            self.assertEqual(stats["matches_with_flipped_handles"], 1)
            # Both matches are scanned whichever way the flag is set, so the
            # report can state the size of the problem.
            self.assertEqual(scan.summary()["matches"], 2)

    def test_off_by_default_the_flipped_match_still_contributes(self):
        pairs, stats, _ = self.build(exclude=False)
        self.assertIn("SWAP", {p.match_code for p in pairs})
        self.assertEqual(stats["matches_excluded_for_flipped_handles"], 0)

    def test_on_the_flipped_match_is_dropped_and_the_clean_one_is_not(self):
        pairs, stats, _ = self.build(exclude=True)
        contributing = {p.match_code for p in pairs}
        self.assertNotIn("SWAP", contributing)
        self.assertIn("CLEAN", contributing)
        self.assertEqual(stats["matches_excluded_for_flipped_handles"], 1)


class TestDropFlippedFlag(unittest.TestCase):
    def setUp(self):
        self.previous = config.EXCLUDE_FLIPPED_MATCHES

    def tearDown(self):
        config.EXCLUDE_FLIPPED_MATCHES = self.previous

    def parse(self, argv):
        from .. import __main__ as cli
        return cli.build_parser().parse_args(argv)

    def test_reporting_is_the_default(self):
        config.EXCLUDE_FLIPPED_MATCHES = False
        from .. import __main__ as cli
        cli.apply_overrides(self.parse(["report"]))
        self.assertFalse(config.EXCLUDE_FLIPPED_MATCHES)

    def test_the_flag_turns_exclusion_on(self):
        config.EXCLUDE_FLIPPED_MATCHES = False
        from .. import __main__ as cli
        cli.apply_overrides(self.parse(["report", "--drop-flipped"]))
        self.assertTrue(config.EXCLUDE_FLIPPED_MATCHES)


class TestPossessionCrossCheck(unittest.TestCase):
    """The signal that survives a level score.

    Football's sequence after a touchdown is fixed, and the play feed names
    its teams in a different vocabulary from the score feed, so each
    touchdown independently tests the PLAYER_1 = Home assumption without
    looking at the totals at all.
    """

    HOME, AWAY = "Home Team", "Away Team"

    @staticmethod
    def td(msg, home_scored):
        return ScoreRow(msg, 1, 6 if home_scored else None,
                        None if home_scored else 6, None, None)

    @staticmethod
    def drive(start, team, n=3):
        return [PlayRow(start + i, 1, team,
                        1 if i == 0 else 2, 10 if i == 0 else 7, 40)
                for i in range(n)]

    def feed(self, scorers):
        """Plays where the right team always receives, plus `scorers`."""
        plays, scores = [], []
        for k, home_scored in enumerate(scorers):
            msg = 100 * k
            receiver = self.AWAY if home_scored else self.HOME
            scores.append(self.td(msg, home_scored))
            plays += self.drive(msg + 5, receiver)
        return plays, scores

    def test_a_consistent_match_says_nothing(self):
        plays, scores = self.feed([True, False, True, False])
        verdict, anchors = handles.possession_verdict("OK", plays, scores)
        self.assertIsNone(verdict)
        self.assertEqual(len(anchors), 4)
        self.assertTrue(all(a.agrees for a in anchors))

    def test_a_match_crossed_throughout_is_caught(self):
        # The exact case the score checks cannot see: nothing ever regresses
        # because the whole match is consistently mirrored.
        plays, scores = self.feed([True, False, True, False])
        crossed = [self.td(s.event_message_count, not bool(s.p1_change))
                   for s in scores]
        verdict, _ = handles.possession_verdict("X", plays, crossed)
        self.assertEqual(verdict.kind, handles.POSSESSION_INVERTED)
        self.assertEqual(verdict.describe(), "0/4 touchdowns agree")
        self.assertTrue(verdict.is_flip)

    def test_a_mid_match_flip_is_located(self):
        plays, scores = self.feed([True, False, True, False, True, False])
        mixed = scores[:3] + [self.td(s.event_message_count, not bool(s.p1_change))
                              for s in scores[3:]]
        verdict, _ = handles.possession_verdict("X", plays, mixed)
        self.assertEqual(verdict.kind, handles.POSSESSION_FLIP)
        # The flip is between the third anchor and the fourth, which is the
        # message reported.
        self.assertEqual(verdict.event_message_count, 300)

    def test_a_match_that_starts_crossed_and_corrects_is_also_a_flip(self):
        # Just as much of the match is read in the wrong frame either way.
        plays, scores = self.feed([True, False, True, False, True, False])
        mixed = [self.td(s.event_message_count, not bool(s.p1_change))
                 for s in scores[:3]] + scores[3:]
        verdict, _ = handles.possession_verdict("X", plays, mixed)
        self.assertEqual(verdict.kind, handles.POSSESSION_FLIP)
        self.assertEqual(verdict.event_message_count, 300)

    def test_a_flip_is_found_with_the_score_dead_level(self):
        # Both sides on 14 throughout the flip. The score checks have
        # nothing to work with here; this one is unaffected.
        plays, scores = self.feed([True, False, True, False, True, False])
        level = []
        for i, row in enumerate(scores):
            home_scored = bool(row.p1_change) if i < 3 else not bool(row.p1_change)
            level.append(ScoreRow(row.event_message_count, 1,
                                  6 if home_scored else None,
                                  None if home_scored else 6, 14, 14))
        self.assertEqual([a.kind for a in handles.scan_match("X", level)], [])
        verdict, _ = handles.possession_verdict("X", plays, level)
        self.assertEqual(verdict.kind, handles.POSSESSION_FLIP)

    def test_a_lagging_vision_row_does_not_invent_a_flip(self):
        # The row straight after a team-label change repeats the previous
        # team's down and distance, so a naive read has the scorer receiving
        # its own kickoff. It must be ignored, not counted.
        plays, scores = self.feed([True, False, True, False])
        noisy = []
        for k, row in enumerate(scores):
            stale = self.HOME if row.p1_change else self.AWAY
            noisy.append(PlayRow(row.event_message_count + 1, 1, stale, 1, 10, 40))
            noisy += self.drive(row.event_message_count + 5,
                                self.AWAY if row.p1_change else self.HOME)
        verdict, anchors = handles.possession_verdict("X", noisy, scores)
        self.assertIsNone(verdict)
        self.assertTrue(all(a.agrees for a in anchors))

    def test_too_few_touchdowns_to_say_anything(self):
        plays, scores = self.feed([True, False])
        verdict, anchors = handles.possession_verdict("X", plays, scores)
        self.assertIsNone(verdict)
        self.assertEqual(len(anchors), 2)

    def test_noise_without_a_clean_turnover_is_unstable_not_a_flip(self):
        # Alternating agreement is not a changepoint. Worth surfacing, but
        # calling the match crossed on it would be inventing a finding.
        plays, scores = self.feed([True, False, True, False, True, False])
        scrambled = list(scores)
        for i in (1, 3):
            row = scrambled[i]
            scrambled[i] = self.td(row.event_message_count, not bool(row.p1_change))
        verdict, _ = handles.possession_verdict("X", plays, scrambled)
        self.assertEqual(verdict.kind, handles.POSSESSION_UNSTABLE)
        self.assertNotIn(handles.POSSESSION_UNSTABLE, handles.FLIP_KINDS)

    def test_only_six_point_scores_anchor(self):
        # A field goal is not followed by the same fixed sequence in this
        # feed's scoring codes, so it is not used as an anchor.
        plays, _ = self.feed([True, False, True])
        field_goals = [ScoreRow(0, 1, 3, None, None, None),
                       ScoreRow(100, 1, None, 3, None, None)]
        self.assertEqual(handles.possession_anchors(plays, field_goals), [])

    def test_a_touchdown_with_no_readable_possession_after_it_is_skipped(self):
        scores = [self.td(0, True), self.td(100, False), self.td(200, True)]
        # Plays exist, but none is a sustained fresh 1st-and-10.
        plays = [PlayRow(5, 1, self.HOME, 3, 8, 40)]
        self.assertEqual(handles.possession_anchors(plays, scores), [])

    def test_the_check_reads_raw_plays_not_cleaned_ones(self):
        # clean_plays uses the very assumption under test, so cleaning first
        # would make the check confirm itself. Asserted by showing the
        # cleaned feed loses the rows the check depends on.
        plays, scores = self.feed([True, False, True, False])
        crossed = [self.td(s.event_message_count, not bool(s.p1_change))
                   for s in scores]
        cleaned, dropped = clean_plays(plays, crossed)
        self.assertGreater(dropped, 0)
        raw_verdict, _ = handles.possession_verdict("X", plays, crossed)
        self.assertEqual(raw_verdict.kind, handles.POSSESSION_INVERTED)


class TestScanUsesPossession(unittest.TestCase):
    def test_plays_are_optional_and_their_absence_skips_the_check(self):
        scan = handles.Scan()
        self.assertTrue(scan.add("NOPLAYS", [ScoreRow(1, 1, 7, 0, 7, 0)], (7, 0)))
        self.assertEqual(scan.summary()["anchors"], 0)

    def test_a_crossed_match_is_flipped_even_with_a_clean_score_series(self):
        plays, scores = TestPossessionCrossCheck().feed([True, False, True, False])
        crossed = [TestPossessionCrossCheck.td(s.event_message_count,
                                               not bool(s.p1_change))
                   for s in scores]
        scan = handles.Scan()
        # Nothing regresses; only the possession check sees this one.
        self.assertEqual(handles.scan_match("X", crossed), [])
        self.assertFalse(scan.add("X", crossed, None, plays))
        self.assertEqual(scan.flipped_matches, frozenset({"X"}))
        self.assertEqual(scan.summary()["anchors"], 4)


class TestScoreDiffBuckets(unittest.TestCase):
    def test_edges(self):
        # Boundaries are what matters here, not the wording, so the expected
        # labels come from the config the buckets are cut from.
        away2, away1, tight, home1, home2 = (
            label for _, _, label in config.SCORE_DIFF_BUCKETS)
        cases = {
            -30: away2, -9: away2,
            -8: away1, -3: away1,
            -2: tight, 0: tight, 2: tight,
            3: home1, 8: home1,
            9: home2, 40: home2,
        }
        for diff, expected in cases.items():
            self.assertEqual(buckets.score_diff_bucket(diff), expected, f"diff={diff}")

    def test_labels_name_the_game_state(self):
        labels = [label for _, _, label in config.SCORE_DIFF_BUCKETS]
        self.assertEqual(
            labels,
            ["Away 2 score", "Away 1 score", "Tight", "Home 1 score", "Home 2 score"])

    def test_every_integer_lands_somewhere(self):
        for diff in range(-60, 61):
            self.assertNotEqual(buckets.score_diff_bucket(diff), "unknown", f"diff={diff}")

    def test_none(self):
        self.assertEqual(buckets.score_diff_bucket(None), "unknown")


class TestTextTableWidths(unittest.TestCase):
    def test_score_column_fits_the_longest_label_with_a_gap(self):
        longest = max(len(label) for _, _, label in config.SCORE_DIFF_BUCKETS)
        self.assertGreater(report.SCORE_WIDTH, longest)
        self.assertGreater(report.CELL_WIDTH, longest + len(" unknown unknown") - 1)

    def test_printed_cells_stay_in_their_columns(self):
        rows = [{"score_diff": label, "time_bucket": "Q1", "possession": "Home",
                 "market": "moneyline", "selection": "Home", "n": 120,
                 "matches": 40, "mean_predicted": 0.52, "realized": 0.48,
                 "gap": -0.04, "brier": 0.2471}
                for _, _, label in config.SCORE_DIFF_BUCKETS]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report.print_cells(rows)
        lines = [line for line in buffer.getvalue().splitlines()
                 if "moneyline" in line]
        self.assertEqual(len(lines), len(rows))
        # Every row must put "moneyline" at the same offset, which only holds
        # if no bucket label has overflowed its column.
        offsets = {line.index("moneyline") for line in lines}
        self.assertEqual(len(offsets), 1, buffer.getvalue())


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


def pair(prod, candidate, outcome, match="AF1", market_id=50, drive=1,
         period=1, score_diff=0, team="Home Team", message=100, gap=0,
         home=None, away=None):
    """A moneyline pair: no line, so both streams answer the same question.

    Most tests care only about the difference, so `score_diff` still works
    and the totals are derived from it. Pass `home`/`away` where the actual
    totals matter.
    """
    if home is None and away is None:
        home, away = max(score_diff, 0), max(-score_diff, 0)
    return directional.PairedObservation(
        match_code=match, drive_number=drive, period_number=period,
        score_p1=home or 0, score_p2=away or 0,
        offensive_team=team, market_id=market_id,
        message_count=message, message_gap=gap,
        prod_probability=prod, candidate_probability=candidate,
        prod_line=None, candidate_line=None,
        prod_outcome=outcome, candidate_outcome=outcome, realized=None)


def line_pair(prod, candidate, prod_line, candidate_line, final_p1, final_p2,
              match="AF1", market_id=54):
    """A pair on a lined market, each side graded against ITS OWN line."""
    return directional.PairedObservation(
        match_code=match, drive_number=1, period_number=1,
        score_p1=0, score_p2=0,
        offensive_team="Home Team", market_id=market_id,
        message_count=100, message_gap=0,
        prod_probability=prod, candidate_probability=candidate,
        prod_line=prod_line, candidate_line=candidate_line,
        prod_outcome=markets.resolve(market_id, prod_line, final_p1, final_p2),
        candidate_outcome=markets.resolve(market_id, candidate_line, final_p1, final_p2),
        realized=markets.realized_value(market_id, final_p1, final_p2))


class TestOwnLineGrading(unittest.TestCase):
    """The bug this guards: grading both streams against prod's line.

    Total finishes at 45. Prod quoted over 44.5 (true), candidate quoted
    over 46.5 (false). Each must be scored against its own question.
    """

    def setUp(self):
        self.p = line_pair(prod=0.55, candidate=0.60,
                           prod_line=44.5, candidate_line=46.5,
                           final_p1=24, final_p2=21, market_id=54)

    def test_outcomes_differ_when_lines_differ(self):
        self.assertTrue(self.p.prod_outcome)
        self.assertFalse(self.p.candidate_outcome)

    def test_each_error_uses_its_own_outcome(self):
        self.assertAlmostEqual(self.p.prod_error, 1 - 0.55)
        self.assertAlmostEqual(self.p.candidate_error, 0.60)

    def test_grading_against_prods_line_would_have_flattered_prod(self):
        # Under the old behaviour both used prod's outcome (True), making the
        # candidate's error 0.40 and handing it the win it did not earn.
        wrong_candidate_error = abs(0.60 - 1.0)
        self.assertLess(wrong_candidate_error, self.p.prod_error)
        self.assertGreater(self.p.candidate_error, self.p.prod_error)
        self.assertEqual(self.p.winner(directional.PROBABILITY), directional.PROD)

    def test_line_closeness_is_available_when_lines_differ(self):
        self.assertAlmostEqual(self.p.realized, 45)
        self.assertAlmostEqual(self.p.prod_line_error, 0.5)
        self.assertAlmostEqual(self.p.candidate_line_error, 1.5)
        self.assertEqual(self.p.winner(directional.LINE), directional.PROD)

    def test_flagged_as_different_line(self):
        self.assertFalse(self.p.same_line)
        self.assertAlmostEqual(self.p.line_delta, 2.0)


class TestSameLineDetection(unittest.TestCase):
    def test_moneyline_has_no_line_and_counts_as_same(self):
        self.assertTrue(pair(0.6, 0.7, True).same_line)
        self.assertIsNone(pair(0.6, 0.7, True).line_delta)

    def test_identical_lines(self):
        p = line_pair(0.5, 0.5, 44.5, 44.5, 24, 21)
        self.assertTrue(p.same_line)
        self.assertAlmostEqual(p.line_delta, 0.0)
        self.assertEqual(p.prod_outcome, p.candidate_outcome)

    def test_whole_number_candidate_line_can_push(self):
        # Total lands exactly on a whole-number line: no 0/1 for that side.
        p = line_pair(0.5, 0.5, 44.5, 45.0, 24, 21, market_id=54)
        self.assertTrue(p.prod_outcome)
        self.assertIsNone(p.candidate_outcome)
        self.assertIsNone(p.candidate_error)
        self.assertFalse(p.comparable(directional.PROBABILITY))
        # Line closeness still works, and a line on the number is perfect.
        self.assertTrue(p.comparable(directional.LINE))
        self.assertAlmostEqual(p.candidate_line_error, 0.0)
        self.assertEqual(p.winner(directional.LINE), directional.CANDIDATE)

    def test_spread_line_closeness_uses_the_selections_own_margin(self):
        # Home wins 24-21, margin +3. Prod -2.5, candidate -6.5.
        p = line_pair(0.5, 0.5, -2.5, -6.5, 24, 21, market_id=52)
        self.assertAlmostEqual(p.realized, 3)
        self.assertAlmostEqual(p.prod_line_error, 5.5)
        self.assertAlmostEqual(p.candidate_line_error, 9.5)
        self.assertEqual(p.winner(directional.LINE), directional.PROD)


class TestLineAgreement(unittest.TestCase):
    def test_counts_same_and_different(self):
        pairs = [
            line_pair(0.5, 0.5, 44.5, 44.5, 24, 21),
            line_pair(0.5, 0.5, 44.5, 46.5, 24, 21),
            pair(0.6, 0.7, True),
        ]
        report = directional.line_agreement(pairs)
        self.assertEqual(report["n"], 3)
        self.assertEqual(report["same"], 2)   # identical line + moneyline
        self.assertEqual(report["different"], 1)
        self.assertAlmostEqual(report["delta_max"], 2.0)

    def test_whole_versus_half_lines_are_counted(self):
        pairs = [line_pair(0.5, 0.5, 44.5, 45.0, 24, 22)]
        report = directional.line_agreement(pairs)
        total = report["by_market"]["total"]
        self.assertEqual(total["prod_half"], 1)
        self.assertEqual(total["prod_whole"], 0)
        self.assertEqual(total["candidate_whole"], 1)
        self.assertEqual(total["candidate_half"], 0)


class TestLineGapDistribution(unittest.TestCase):
    """Same-line zeros must not be mixed into the "where they differ" stats."""

    def setUp(self):
        self.pairs = [
            line_pair(0.5, 0.5, 44.5, 44.5, 24, 21),   # same
            line_pair(0.5, 0.5, 44.5, 44.5, 24, 21),   # same
            line_pair(0.5, 0.5, 44.5, 46.5, 24, 21),   # differ by 2
            line_pair(0.5, 0.5, 44.5, 48.5, 24, 21),   # differ by 4
        ]
        self.report = directional.line_agreement(self.pairs)

    def test_differing_stats_exclude_the_same_line_zeros(self):
        self.assertEqual(self.report["differing_n"], 2)
        self.assertAlmostEqual(self.report["differing_mean"], 3.0)
        self.assertAlmostEqual(self.report["differing_max"], 4.0)

    def test_all_pairs_median_is_dragged_to_zero_by_the_same_line_pairs(self):
        # This is exactly why the two are reported separately.
        self.assertAlmostEqual(self.report["delta_median"], 2.0)
        self.assertAlmostEqual(self.report["differing_median"], 4.0)

    def test_no_differing_pairs(self):
        report = directional.line_agreement([line_pair(0.5, 0.5, 44.5, 44.5, 24, 21)])
        self.assertEqual(report["differing_n"], 0)
        self.assertIsNone(report["differing_median"])


class TestMarketBlocks(unittest.TestCase):
    def test_each_market_gets_its_own_clustered_test(self):
        pairs = ([pair(0.6, 0.8, True, match=f"AF{i}", market_id=50) for i in range(12)]
                 + [line_pair(0.5, 0.5, 44.5, 44.5, 24, 21, match=f"AF{i}", market_id=54)
                    for i in range(12)])
        blocks = directional.market_blocks(pairs, directional.PROBABILITY, n_bootstrap=100)
        self.assertIn(markets.MONEYLINE, blocks)
        self.assertIn(markets.TOTAL, blocks)
        for group in blocks.values():
            self.assertIn("brier", group)
            self.assertIn("votes", group)
            self.assertEqual(group["tally"]["n_matches"], 12)

    def test_a_flat_market_does_not_mask_a_moving_one(self):
        """The reason per-market clustering exists.

        Moneyline strongly favours the candidate; totals are dead level.
        Pooled, the effect is halved; split, it survives intact.
        """
        moving = [pair(0.2, 0.9, True, match=f"AF{i}", market_id=50) for i in range(20)]
        flat = [line_pair(0.5, 0.5, 44.5, 44.5, 24, 21, match=f"AF{i}", market_id=54)
                for i in range(20)]
        blocks = directional.market_blocks(moving + flat, directional.PROBABILITY,
                                           n_bootstrap=100)
        pooled = directional.paired_delta_summary(moving + flat, n_bootstrap=100)
        moneyline_mean = blocks[markets.MONEYLINE]["brier"]["mean"]
        self.assertGreater(moneyline_mean, pooled["mean"])
        self.assertAlmostEqual(blocks[markets.TOTAL]["brier"]["mean"], 0.0)


class TestSplitByLine(unittest.TestCase):
    def test_split(self):
        same, different = directional.split_by_line([
            line_pair(0.5, 0.5, 44.5, 44.5, 24, 21),
            line_pair(0.5, 0.5, 44.5, 46.5, 24, 21),
        ])
        self.assertEqual(len(same), 1)
        self.assertEqual(len(different), 1)


class TestPairedObservation(unittest.TestCase):
    def test_errors_against_a_win(self):
        p = pair(0.60, 0.80, True)
        self.assertAlmostEqual(p.prod_error, 0.40)
        self.assertAlmostEqual(p.candidate_error, 0.20)
        self.assertEqual(p.winner(), directional.CANDIDATE)

    def test_errors_against_a_loss(self):
        # The same probabilities, opposite outcome, flip the winner.
        p = pair(0.60, 0.80, False)
        self.assertAlmostEqual(p.prod_error, 0.60)
        self.assertAlmostEqual(p.candidate_error, 0.80)
        self.assertEqual(p.winner(), directional.PROD)

    def test_identical_probabilities_tie(self):
        self.assertEqual(pair(0.5, 0.5, True).winner(), directional.TIE)

    def test_disagreement_is_absolute(self):
        self.assertAlmostEqual(pair(0.8, 0.6, True).disagreement, 0.2)
        self.assertAlmostEqual(pair(0.6, 0.8, True).disagreement, 0.2)


class TestNearestMessage(unittest.TestCase):
    def test_prefers_an_exact_hit(self):
        self.assertEqual(directional.nearest_message([98, 100, 102], 100, 3), (100, 0))

    def test_takes_the_closer_side(self):
        self.assertEqual(directional.nearest_message([98, 103], 100, 3), (98, -2))

    def test_respects_the_gap_limit(self):
        self.assertIsNone(directional.nearest_message([90, 110], 100, 3))

    def test_boundary_is_inclusive(self):
        self.assertEqual(directional.nearest_message([97], 100, 3), (97, -3))

    def test_empty(self):
        self.assertIsNone(directional.nearest_message([], 100, 3))
        self.assertIsNone(directional.nearest_message([100], None, 3))


class TestPairingIndexes(unittest.TestCase):
    def test_index_by_message_keeps_probability_and_description(self):
        rows = [("AF1", 50, BASE, 85.26, 1.11, "Home team (PLAYER 1) to win", 300)]
        index = directional.index_by_message(rows)
        self.assertEqual(index[("AF1", 50)][300][0], 85.26)

    def test_index_by_message_skips_null_message_or_probability(self):
        rows = [("AF1", 50, BASE, 85.0, 1.1, "d", None),
                ("AF1", 50, BASE, None, 1.1, "d", 300)]
        self.assertEqual(directional.index_by_message(rows), {})

    def test_common_messages_is_the_intersection(self):
        prod = {("AF1", 50): {1: (10.0, "d"), 2: (20.0, "d"), 3: (30.0, "d")}}
        cand = {("AF1", 50): {2: (21.0, "d"), 3: (31.0, "d"), 9: (90.0, "d")}}
        self.assertEqual(directional.common_messages(prod, cand), {("AF1", 50): [2, 3]})

    def test_market_missing_from_one_stream_is_excluded(self):
        prod = {("AF1", 50): {1: (10.0, "d")}, ("AF1", 51): {1: (10.0, "d")}}
        cand = {("AF1", 50): {1: (11.0, "d")}}
        self.assertEqual(list(directional.common_messages(prod, cand)), [("AF1", 50)])


class TestTally(unittest.TestCase):
    def test_counts_and_deltas(self):
        pairs = [
            pair(0.60, 0.80, True),    # candidate closer
            pair(0.60, 0.80, True),    # candidate closer
            pair(0.60, 0.80, False),   # prod closer
            pair(0.50, 0.50, True),    # tie
        ]
        t = directional.tally(pairs)
        self.assertEqual(t["n"], 4)
        self.assertEqual(t[directional.CANDIDATE], 2)
        self.assertEqual(t[directional.PROD], 1)
        self.assertEqual(t[directional.TIE], 1)
        self.assertEqual(t["decisive"], 3)
        self.assertAlmostEqual(t["candidate_win_rate"], 2 / 3)
        # Positive delta means the candidate carries the smaller error.
        self.assertAlmostEqual(t["mae_delta"], (0.40 + 0.40 + 0.60 + 0.50) / 4
                                              - (0.20 + 0.20 + 0.80 + 0.50) / 4)

    def test_empty(self):
        t = directional.tally([])
        self.assertEqual(t["n"], 0)
        self.assertIsNone(t["candidate_win_rate"])
        self.assertIsNone(t["p_value"])


class TestMatchLevelVotes(unittest.TestCase):
    def test_each_match_votes_once(self):
        pairs = (
            # AF1: candidate takes 3 of 4 -> one vote for candidate
            [pair(0.6, 0.8, True, match="AF1")] * 3
            + [pair(0.6, 0.8, False, match="AF1")]
            # AF2: prod takes 2 of 2 -> one vote for prod
            + [pair(0.6, 0.8, False, match="AF2")] * 2
            # AF3: one each -> split
            + [pair(0.6, 0.8, True, match="AF3"), pair(0.6, 0.8, False, match="AF3")]
        )
        votes = directional.match_level_votes(pairs)
        self.assertEqual(votes["n_matches"], 3)
        self.assertEqual(votes[directional.CANDIDATE], 1)
        self.assertEqual(votes[directional.PROD], 1)
        self.assertEqual(votes[directional.TIE], 1)

    def test_one_lopsided_match_cannot_carry_the_vote(self):
        # 50 pairs in one match, all won by the candidate, against two
        # matches won by prod: row level says candidate, match level does not.
        pairs = ([pair(0.6, 0.8, True, match="AF1")] * 50
                 + [pair(0.6, 0.8, False, match="AF2")]
                 + [pair(0.6, 0.8, False, match="AF3")])
        row_level = directional.tally(pairs)
        votes = directional.match_level_votes(pairs)
        self.assertAlmostEqual(row_level["candidate_win_rate"], 50 / 52)
        self.assertEqual(votes[directional.CANDIDATE], 1)
        self.assertEqual(votes[directional.PROD], 2)
        self.assertAlmostEqual(votes["candidate_win_rate"], 1 / 3)


class TestPairedDelta(unittest.TestCase):
    def test_per_match_delta_positive_when_candidate_is_closer(self):
        pairs = [pair(0.60, 0.80, True, match="AF1"), pair(0.60, 0.80, True, match="AF1")]
        deltas = directional.per_match_deltas(pairs, directional.SQUARED)
        # prod error 0.40 -> 0.16, candidate 0.20 -> 0.04
        self.assertAlmostEqual(deltas["AF1"], 0.16 - 0.04)

    def test_per_match_delta_absolute_variant(self):
        pairs = [pair(0.60, 0.80, True, match="AF1")]
        deltas = directional.per_match_deltas(pairs, directional.ABSOLUTE)
        self.assertAlmostEqual(deltas["AF1"], 0.40 - 0.20)

    def test_identical_models_give_zero_delta(self):
        pairs = [pair(0.7, 0.7, True, match=f"AF{i}") for i in range(10)]
        summary = directional.paired_delta_summary(pairs, n_bootstrap=100)
        self.assertAlmostEqual(summary["mean"], 0.0)
        self.assertEqual(summary["matches_favouring_candidate"], 0)
        self.assertEqual(summary["matches_favouring_prod"], 0)

    def test_consistent_improvement_is_detected(self):
        # 30 matches, candidate closer in every one.
        pairs = [pair(0.60, 0.80, True, match=f"AF{i}") for i in range(30)]
        summary = directional.paired_delta_summary(pairs, n_bootstrap=200)
        self.assertEqual(summary["n_matches"], 30)
        self.assertGreater(summary["mean"], 0)
        self.assertEqual(summary["matches_favouring_candidate"], 30)

    def test_bootstrap_ci_brackets_the_mean(self):
        pairs = ([pair(0.60, 0.80, True, match=f"AF{i}") for i in range(20)]
                 + [pair(0.60, 0.80, False, match=f"AF{i}") for i in range(20, 25)])
        summary = directional.paired_delta_summary(pairs, n_bootstrap=500)
        self.assertLessEqual(summary["ci_low"], summary["mean"])
        self.assertGreaterEqual(summary["ci_high"], summary["mean"])

    def test_single_match_has_no_standard_error(self):
        summary = directional.paired_delta_summary([pair(0.6, 0.8, True, match="AF1")])
        self.assertEqual(summary["n_matches"], 1)
        self.assertIsNotNone(summary["mean"])
        self.assertIsNone(summary["se"])

    def test_empty(self):
        summary = directional.paired_delta_summary([])
        self.assertEqual(summary["n_matches"], 0)
        self.assertIsNone(summary["mean"])

    def test_win_rate_is_blind_to_a_better_model_that_paired_loss_sees(self):
        """The reason both views are reported.

        Candidate is closer on half the pairs by a wide margin and further on
        the other half by a narrow one: a 50% win rate, but clearly lower loss.
        """
        pairs = []
        for i in range(20):
            pairs.append(pair(0.10, 0.60, True, match=f"AF{i}"))   # cand much closer
            pairs.append(pair(0.50, 0.45, True, match=f"AF{i}"))   # prod barely closer
        tallied = directional.tally(pairs)
        self.assertAlmostEqual(tallied["candidate_win_rate"], 0.5)
        summary = directional.paired_delta_summary(pairs, n_bootstrap=200)
        self.assertGreater(summary["mean"], 0)
        self.assertGreater(summary["ci_low"], 0)


class TestSignTest(unittest.TestCase):
    def test_unanimous_ten(self):
        self.assertAlmostEqual(metrics.sign_test(10, 0), 2 / 1024)

    def test_even_split_is_capped_at_one(self):
        self.assertEqual(metrics.sign_test(5, 5), 1.0)

    def test_symmetric_in_its_arguments(self):
        self.assertEqual(metrics.sign_test(9, 3), metrics.sign_test(3, 9))

    def test_single_pair_is_uninformative(self):
        self.assertEqual(metrics.sign_test(1, 0), 1.0)

    def test_no_decisive_pairs(self):
        self.assertIsNone(metrics.sign_test(0, 0))

    def test_large_lopsided_split_is_significant(self):
        self.assertLess(metrics.sign_test(120, 80), 0.01)

    def test_thousands_of_pairs_do_not_overflow(self):
        # The exact form builds integers with hundreds of digits; scaling
        # one of those to float overflows, so large n takes the normal path.
        for wins, losses in [(2200, 2120), (30000, 29000), (5000, 5000)]:
            value = metrics.sign_test(wins, losses)
            self.assertIsNotNone(value)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_exact_and_approximate_agree_across_the_threshold(self):
        # Same win rate either side of the switch should give a similar p.
        n_exact = metrics.EXACT_SIGN_TEST_MAX_N
        exact = metrics.sign_test(int(n_exact * 0.55), n_exact - int(n_exact * 0.55))
        n_approx = n_exact + 2
        approx = metrics.sign_test(int(n_approx * 0.55), n_approx - int(n_approx * 0.55))
        self.assertAlmostEqual(exact, approx, places=3)

    def test_approximation_still_capped_at_one(self):
        self.assertEqual(metrics.sign_test(600, 600), 1.0)


class TestDisagreementBands(unittest.TestCase):
    def test_bands(self):
        cases = {0.000: "< 1pp", 0.005: "< 1pp", 0.02: "1-3pp",
                 0.04: "3-5pp", 0.07: "5-10pp", 0.5: "> 10pp"}
        for gap, expected in cases.items():
            p = pair(0.5, 0.5 + gap, True)
            self.assertEqual(directional.disagreement_band(p), expected, f"gap={gap}")


class TestCrossSectionalLineRule(unittest.TestCase):
    """The rule: a different line overrules the probability outright."""

    def setUp(self):
        # Two same-line pairs and two different-line pairs, same cell.
        self.same = [
            line_pair(0.55, 0.65, 44.5, 44.5, 24, 21, match="AF1"),
            line_pair(0.55, 0.65, 44.5, 44.5, 24, 21, match="AF2"),
        ]
        self.different = [
            line_pair(0.55, 0.65, 44.5, 46.5, 24, 21, match="AF3"),
            line_pair(0.55, 0.65, 44.5, 48.5, 24, 21, match="AF4"),
        ]
        self.pairs = self.same + self.different

    def test_probability_cells_use_only_same_line_pairs(self):
        cells = directional.calibration_cells(self.pairs, lambda p: "cell",
                                              n_bootstrap=50)
        self.assertEqual(cells[("cell", "total")]["n"], 2)
        self.assertEqual(cells[("cell", "total")]["matches"], 2)

    def test_same_line_means_one_shared_realized_rate(self):
        # Same line, same question, so the two streams resolve identically.
        cells = directional.calibration_cells(self.pairs, lambda p: "cell",
                                              n_bootstrap=50)
        row = cells[("cell", "total")]
        self.assertEqual(row["realized"], row["realized_check"])
        self.assertAlmostEqual(row["realized"], 1.0)   # total 45 is over 44.5

    def test_predicted_differs_while_realized_does_not(self):
        cells = directional.calibration_cells(self.pairs, lambda p: "cell",
                                              n_bootstrap=50)
        row = cells[("cell", "total")]
        self.assertAlmostEqual(row["prod_predicted"], 0.55)
        self.assertAlmostEqual(row["candidate_predicted"], 0.65)
        # Candidate's 0.65 is nearer a realized 1.0 than prod's 0.55.
        self.assertEqual(row["winner"], directional.CANDIDATE)

    def test_line_cells_use_only_different_line_pairs(self):
        cells = directional.line_cells(self.pairs, lambda p: "cell", n_bootstrap=50)
        self.assertEqual(cells["cell"]["n"], 2)
        self.assertEqual(cells["cell"]["matches"], 2)

    def test_line_cells_measure_distance_from_the_result(self):
        cells = directional.line_cells(self.pairs, lambda p: "cell", n_bootstrap=50)
        row = cells["cell"]
        # Total 45. Prod at 44.5 twice -> 0.5. Candidate at 46.5 and 48.5 -> 1.5, 3.5.
        self.assertAlmostEqual(row["prod_line_error"], 0.5)
        self.assertAlmostEqual(row["candidate_line_error"], 2.5)
        self.assertAlmostEqual(row["points_delta"], -2.0)   # prod's line closer

    def test_the_two_views_partition_the_pairs(self):
        prob = directional.calibration_cells(self.pairs, lambda p: "cell", n_bootstrap=50)
        line = directional.line_cells(self.pairs, lambda p: "cell", n_bootstrap=50)
        self.assertEqual(prob[("cell", "total")]["n"] + line["cell"]["n"], len(self.pairs))

    def test_moneyline_always_lands_in_the_probability_view(self):
        pairs = [pair(0.6, 0.7, True, match=f"AF{i}") for i in range(4)]
        prob = directional.calibration_cells(pairs, lambda p: "cell", n_bootstrap=50)
        line = directional.line_cells(pairs, lambda p: "cell", n_bootstrap=50)
        self.assertEqual(prob[("cell", "moneyline")]["n"], 4)
        self.assertEqual(line, {})

    def test_cross_axes_cover_the_three_requested_splits(self):
        names = [name for name, _, _ in directional.CROSS_AXES]
        self.assertEqual(names, ["Score difference", "Quarter", "Possession"])
        for _, key_function, order_factory in directional.CROSS_AXES:
            self.assertIn(key_function(self.same[0]), order_factory())


class TestMirrorCancellation(unittest.TestCase):
    """Pooling both sides of a market destroys the calibration measurement.

    The sides are complements, so every (p, y) comes with a mirror
    (1-p, 1-y): realized and mean prediction BOTH average to exactly 0.5
    whatever the model does. A realized rate of exactly 0.500 in every cell
    is the signature of this bug, not a property of the data.
    """

    def _both_sides(self, prod_probability, candidate_probability, outcome, match):
        """Home and Away rows for one moneyline market, as the feed supplies."""
        return [
            pair(prod_probability, candidate_probability, outcome,
                 match=match, market_id=50),
            pair(1 - prod_probability, 1 - candidate_probability, not outcome,
                 match=match, market_id=51),
        ]

    def setUp(self):
        self.pairs = []
        for i, (p, y) in enumerate([(0.85, True), (0.30, False), (0.62, True),
                                    (0.10, False), (0.55, True)]):
            self.pairs += self._both_sides(p, p + 0.02, y, f"AF{i}")

    def test_pooling_both_sides_would_pin_realized_to_one_half(self):
        pooled = metrics.summarize(
            [(p.prod_probability, p.prod_outcome) for p in self.pairs], 10)
        self.assertAlmostEqual(pooled["realized"], 0.5)
        self.assertAlmostEqual(pooled["mean_predicted"], 0.5)
        self.assertAlmostEqual(pooled["gap"], 0.0)

    def test_canonical_selection_lets_realized_move(self):
        cells = directional.calibration_cells(self.pairs, lambda p: "cell",
                                              n_bootstrap=50)
        row = cells[("cell", "moneyline")]
        # Only market 50 survives, so 5 rows not 10, and realized is the real
        # 3-of-5 rather than a forced 0.5.
        self.assertEqual(row["n"], 5)
        self.assertAlmostEqual(row["realized"], 0.6)
        self.assertNotAlmostEqual(row["realized"], 0.5)

    def test_only_canonical_market_ids_are_used(self):
        self.assertEqual(set(config.CANONICAL_SELECTIONS), {50, 52, 54})
        for market_id in config.CANONICAL_SELECTIONS:
            self.assertIn(market_id, markets.MARKET_IDS)

    def test_markets_are_kept_apart(self):
        # Different markets are different questions with different base
        # rates, so they must not share a cell.
        mixed = ([pair(0.9, 0.9, True, match=f"AF{i}", market_id=50) for i in range(4)]
                 + [line_pair(0.2, 0.2, 44.5, 44.5, 10, 10, match=f"AF{i}",
                              market_id=54) for i in range(4)])
        cells = directional.calibration_cells(mixed, lambda p: "cell", n_bootstrap=50)
        self.assertEqual(set(cells), {("cell", "moneyline"), ("cell", "total")})
        self.assertAlmostEqual(cells[("cell", "moneyline")]["realized"], 1.0)
        self.assertAlmostEqual(cells[("cell", "total")]["realized"], 0.0)


class TestComplementReport(unittest.TestCase):
    """Is the one-side-per-market shortcut sound for this feed?"""

    def _sides(self, p_home, outcome, match, message=100):
        return [
            pair(p_home, p_home, outcome, match=match, market_id=50, message=message),
            pair(1 - p_home, 1 - p_home, not outcome, match=match, market_id=51,
                 message=message),
        ]

    def test_fair_book_sums_to_one_and_partitions(self):
        pairs = self._sides(0.7, True, "AF1") + self._sides(0.4, False, "AF2")
        report = directional.complement_report(pairs)
        moneyline = report["moneyline"]
        self.assertEqual(moneyline["both_sides"], 2)
        self.assertAlmostEqual(moneyline["prob_sum_mean"], 1.0)
        self.assertEqual(moneyline["outcomes_partition"], 2)
        self.assertEqual(moneyline["both_won"], 0)
        self.assertEqual(moneyline["both_lost"], 0)

    def test_overround_shows_up_in_the_probability_sum(self):
        # Both sides quoted 3 points rich.
        pairs = [
            pair(0.73, 0.73, True, match="AF1", market_id=50),
            pair(0.30, 0.30, False, match="AF1", market_id=51),
        ]
        report = directional.complement_report(pairs)
        self.assertAlmostEqual(report["moneyline"]["prob_sum_mean"], 1.03)

    def test_overlapping_spread_sides_are_caught(self):
        """Both sides at the same line can both win, which breaks the shortcut.

        52 is "margin > L" and 53 is "-margin > L". At L = -2.5 both are true
        for any margin between -2.5 and +2.5.
        """
        overlapping = [
            directional.PairedObservation(
                match_code="AF1", drive_number=1, period_number=1,
                score_p1=0, score_p2=0,
                offensive_team="Home Team", market_id=market_id,
                message_count=100, message_gap=0,
                prod_probability=0.6, candidate_probability=0.6,
                prod_line=-2.5, candidate_line=-2.5,
                prod_outcome=markets.resolve(market_id, -2.5, 22, 21),
                candidate_outcome=markets.resolve(market_id, -2.5, 22, 21),
                realized=markets.realized_value(market_id, 22, 21))
            for market_id in (52, 53)
        ]
        # Margin is +1, so 52 (>-2.5) wins and 53 (-1 > -2.5) also wins.
        self.assertTrue(overlapping[0].prod_outcome)
        self.assertTrue(overlapping[1].prod_outcome)
        report = directional.complement_report(overlapping)
        self.assertEqual(report["spread"]["both_won"], 1)
        self.assertEqual(report["spread"]["outcomes_partition"], 0)

    def test_one_sided_markets_are_skipped(self):
        report = directional.complement_report([pair(0.6, 0.6, True, market_id=50)])
        self.assertEqual(report, {})


class TestBothSidesCalibration(unittest.TestCase):
    def test_complementary_sides_mirror(self):
        pairs = []
        for i, (p, y) in enumerate([(0.8, True), (0.3, False), (0.6, True)]):
            pairs.append(pair(p, p, y, match=f"AF{i}", market_id=50))
            pairs.append(pair(1 - p, 1 - p, not y, match=f"AF{i}", market_id=51))
        cells = directional.both_sides_calibration(pairs)
        home, away = cells[50], cells[51]
        self.assertAlmostEqual(home["realized"] + away["realized"], 1.0)
        self.assertAlmostEqual(home["prod_gap"] + away["prod_gap"], 0.0)

    def test_canonical_flag_marks_the_calibrated_side(self):
        pairs = [pair(0.6, 0.6, True, match="AF1", market_id=50),
                 pair(0.4, 0.4, False, match="AF1", market_id=51)]
        cells = directional.both_sides_calibration(pairs)
        self.assertTrue(cells[50]["canonical"])
        self.assertFalse(cells[51]["canonical"])

    def test_dropping_the_mirror_loses_no_information(self):
        """The reason one side is enough: the other is a sign flip."""
        pairs = []
        for i, (p, y) in enumerate([(0.75, True), (0.25, False)]):
            pairs.append(pair(p, p, y, match=f"AF{i}", market_id=50))
            pairs.append(pair(1 - p, 1 - p, not y, match=f"AF{i}", market_id=51))
        cells = directional.both_sides_calibration(pairs)
        self.assertAlmostEqual(cells[50]["prod_gap"], -cells[51]["prod_gap"])


def spread_sides(line_home, line_away, final_p1, final_p2, match="AF1",
                 p_home=0.5, p_away=0.5, message=100):
    """Both spread selections for one market, each with its own line."""
    out = []
    for market_id, line in ((52, line_home), (53, line_away)):
        out.append(directional.PairedObservation(
            match_code=match, drive_number=1, period_number=1,
            score_p1=0, score_p2=0,
            offensive_team="Home Team", market_id=market_id,
            message_count=message, message_gap=0,
            prod_probability=p_home if market_id == 52 else p_away,
            candidate_probability=p_home if market_id == 52 else p_away,
            prod_line=line, candidate_line=line,
            prod_outcome=markets.resolve(market_id, line, final_p1, final_p2),
            candidate_outcome=markets.resolve(market_id, line, final_p1, final_p2),
            realized=markets.realized_value(market_id, final_p1, final_p2)))
    return out


class TestSpreadResolutionSwitch(unittest.TestCase):
    def tearDown(self):
        config.SPREAD_RESOLUTION = "literal"

    def test_literal_lets_both_sides_win_inside_the_band(self):
        # Home by 1, both sides at -2.5: -2.5 < 1 < 2.5, so both clear.
        config.SPREAD_RESOLUTION = "literal"
        self.assertTrue(markets.resolve(52, -2.5, 22, 21))
        self.assertTrue(markets.resolve(53, -2.5, 22, 21))

    def test_complement_partitions(self):
        config.SPREAD_RESOLUTION = "complement"
        self.assertTrue(markets.resolve(52, -2.5, 22, 21))
        self.assertFalse(markets.resolve(53, -2.5, 22, 21))

    def test_complement_flips_when_the_home_margin_misses(self):
        config.SPREAD_RESOLUTION = "complement"
        # Home loses by 7, margin -7, which does not clear -2.5.
        self.assertFalse(markets.resolve(52, -2.5, 17, 24))
        self.assertTrue(markets.resolve(53, -2.5, 17, 24))

    def test_market_52_is_unaffected_by_the_switch(self):
        for mode in ("literal", "complement"):
            config.SPREAD_RESOLUTION = mode
            self.assertTrue(markets.resolve(52, -2.5, 22, 21), mode)
            self.assertFalse(markets.resolve(52, 10.5, 22, 21), mode)


class TestSpreadInterpretation(unittest.TestCase):
    def tearDown(self):
        config.SPREAD_RESOLUTION = "literal"

    def test_mirrored_lines_that_partition_confirm_the_literal_reading(self):
        # 52 at -2.5 and 53 at +2.5 partition on margin_1 vs -2.5.
        pairs = []
        for i, (p1, p2) in enumerate([(24, 21), (17, 24), (30, 10), (14, 20)]):
            pairs += spread_sides(-2.5, 2.5, p1, p2, match=f"AF{i}",
                                  p_home=0.55, p_away=0.45, message=100 + i)
        report = directional.spread_interpretation_report(pairs)
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["lines_mirrored"], 4)
        self.assertEqual(report["lines_equal"], 0)
        self.assertEqual(report["literal_partition"], 4)
        self.assertEqual(report["verdict"][0], "literal")

    def test_same_line_with_fair_probabilities_that_overlap_says_complement(self):
        # Both sides at -2.5 and probabilities summing to 1, but margins
        # inside the band make the literal reading double-count.
        pairs = []
        for i, (p1, p2) in enumerate([(22, 21), (21, 22), (23, 22), (20, 21),
                                      (24, 23), (19, 20)]):
            pairs += spread_sides(-2.5, -2.5, p1, p2, match=f"AF{i}",
                                  p_home=0.52, p_away=0.48, message=100 + i)
        report = directional.spread_interpretation_report(pairs)
        self.assertEqual(report["lines_equal"], 6)
        self.assertAlmostEqual(report["prob_sum_mean"], 1.0)
        self.assertEqual(report["literal_partition"], 0)
        self.assertEqual(report["literal_both_won"], 6)
        self.assertEqual(report["verdict"][0], "complement")

    def test_verdict_is_recomputed_from_lines_not_stored_outcomes(self):
        """Building the pairs under one reading must not bias the verdict."""
        config.SPREAD_RESOLUTION = "complement"
        pairs = []
        for i, (p1, p2) in enumerate([(24, 21), (17, 24), (30, 10), (14, 20)]):
            pairs += spread_sides(-2.5, 2.5, p1, p2, match=f"AF{i}", message=100 + i)
        report = directional.spread_interpretation_report(pairs)
        # Mirrored lines still read as literal despite the pairs being built
        # with the complement resolution active.
        self.assertEqual(report["verdict"][0], "literal")

    def test_no_spread_pairs(self):
        report = directional.spread_interpretation_report(
            [pair(0.6, 0.6, True, market_id=50)])
        self.assertEqual(report["n"], 0)
        self.assertEqual(report["verdict"][0], "unclear")


class TestFullReport(unittest.TestCase):
    def setUp(self):
        self.pairs = ([pair(0.6, 0.8, True, match=f"AF{i}", market_id=50)
                       for i in range(6)]
                      + [line_pair(0.5, 0.5, 44.5, 44.5, 24, 21, match=f"AF{i}",
                                   market_id=54) for i in range(6)]
                      + [line_pair(0.5, 0.6, 44.5, 46.5, 24, 21, match=f"AF{i}",
                                   market_id=54) for i in range(6, 10)])

    def test_report_carries_every_section(self):
        built = directional.build_full_report(self.pairs, n_bootstrap=50)
        for key in ("summary", "complement", "spread", "both_sides", "axes",
                    "full_cell", "full_cell_order"):
            self.assertIn(key, built)
        self.assertEqual([axis["name"] for axis in built["axes"]],
                         ["Score difference", "Quarter", "Possession"])

    def test_every_axis_carries_both_views(self):
        built = directional.build_full_report(self.pairs, n_bootstrap=50)
        for axis in built["axes"]:
            self.assertIn("probability", axis)
            self.assertIn("line", axis)
            self.assertTrue(axis["order"])

    def test_sorted_by_descending_disagreement(self):
        ordered = directional.sorted_pairs_by_disagreement(self.pairs)
        gaps = [p.disagreement for p in ordered]
        self.assertEqual(gaps, sorted(gaps, reverse=True))
        self.assertEqual(len(ordered), len(self.pairs))

    def test_sort_is_stable_on_ties(self):
        # Equal gaps fall back to match, drive, market so runs are reproducible.
        tied = [pair(0.5, 0.5, True, match="AF2", market_id=51),
                pair(0.5, 0.5, True, match="AF1", market_id=50),
                pair(0.5, 0.5, True, match="AF1", market_id=51)]
        ordered = directional.sorted_pairs_by_disagreement(tied)
        self.assertEqual([(p.match_code, p.market_id) for p in ordered],
                         [("AF1", 50), ("AF1", 51), ("AF2", 51)])

    def test_report_survives_no_pairs(self):
        built = directional.build_full_report([], n_bootstrap=10)
        self.assertEqual(built["summary"]["pairs"], 0)
        self.assertEqual(built["full_cell"], {})


class TestReportRendering(unittest.TestCase):
    """Guards two things a full audit of a real report turned up."""

    def setUp(self):
        from .. import html_full
        self.html_full = html_full
        self.pairs = [pair(0.6, 0.8, True, match=f"AF{i}", market_id=50,
                           message=100 + i) for i in range(5)]
        self.report = directional.build_full_report(self.pairs, n_bootstrap=50)
        # The universe deliberately exceeds the matches that produced pairs.
        self.header = {"paired_matches": 9}
        self.stats = {"snapshots": 12, "exact_message_pair": 4,
                      "offset_message_pair": 1}
        self.scan = clean_scan(5)

    def _render(self):
        return self.html_full.render(self.report, self.header, self.stats,
                                     self.pairs, self.scan)

    def test_header_separates_contributing_matches_from_the_universe(self):
        # 5 matches produced pairs; 9 were settled. Reporting only one
        # number implied the wrong thing.
        rendered = self._render()
        self.assertIn("<b>5</b> matches with pairs", rendered)
        self.assertIn("<b>9</b> settled", rendered)

    def test_pair_table_carries_the_message_columns(self):
        rendered = self._render()
        self.assertIn("<th>Msg</th>", rendered)
        self.assertIn("&plusmn;Msg</th>", rendered)

    def test_message_gap_is_flagged_when_not_exact(self):
        offset = [pair(0.6, 0.8, True, match="AF9", market_id=50, message=100, gap=2)]
        report = directional.build_full_report(offset, n_bootstrap=20)
        rendered = self.html_full.render(report, self.header, self.stats, offset,
                                         self.scan)
        self.assertIn('class="warn"', rendered)

    def test_every_pair_reaches_the_table(self):
        rendered = self._render()
        body = rendered[rendered.index('id="pairTable"'):]
        body = body[body.index("<tbody>"):body.index("</tbody>")]
        self.assertEqual(body.count("<tr "), len(self.pairs))

    def test_pair_rows_carry_both_totals_not_just_the_difference(self):
        # 7-7 and 21-21 are the same difference and very different games.
        pairs = [pair(0.6, 0.8, True, match="AF1", home=21, away=14)]
        report = directional.build_full_report(pairs, n_bootstrap=20)
        rendered = self.html_full.render(report, self.header, self.stats,
                                         pairs, self.scan)
        body = rendered[rendered.index('id="pairTable"'):]
        head = body[:body.index("</thead>")]
        for name in ("Home", "Away", "Diff"):
            self.assertIn(f">{name}</th>", head)
        row = body[body.index("<tbody>"):body.index("</tbody>")]
        self.assertIn('data-v="21">21</td>', row)
        self.assertIn('data-v="14">14</td>', row)
        self.assertIn('data-v="7">+7</td>', row)

    def test_score_diff_cannot_drift_from_the_totals(self):
        p = pair(0.6, 0.8, True, home=24, away=10)
        self.assertEqual(p.score_diff, 14)
        self.assertEqual((p.score_p1, p.score_p2), (24, 10))

    def test_every_row_carries_its_match_id_for_the_filter(self):
        pairs = [pair(0.6, 0.8, True, match="AF-Upper"),
                 pair(0.6, 0.8, True, match="af-lower")]
        report = directional.build_full_report(pairs, n_bootstrap=20)
        rendered = self.html_full.render(report, self.header, self.stats,
                                         pairs, self.scan)
        body = rendered[rendered.index('id="pairTable"'):]
        body = body[body.index("<tbody>"):body.index("</tbody>")]
        # Lower-cased on the row so the filter can compare without
        # re-casing twenty thousand strings per keystroke.
        self.assertIn('data-match="af-upper"', body)
        self.assertIn('data-match="af-lower"', body)

    def test_the_filter_box_is_wired_to_the_pair_table(self):
        rendered = self._render()
        self.assertIn('id="pairFilter"', rendered)
        self.assertIn('id="pairCount"', rendered)
        script = rendered.split("<script>")[1]
        self.assertIn("pairFilter", script)
        self.assertIn("dataset.match", script)
        # Filtering must not touch the sort handlers or vice versa.
        self.assertIn("pairTable", script)

    def test_rendered_page_has_no_external_fetches(self):
        rendered = self._render()
        self.assertNotIn('src="http', rendered)
        self.assertNotIn('href="http', rendered)
        self.assertNotIn("@import", rendered)


class TestRuntimeOverrides(unittest.TestCase):
    """The window and tuning must be settable without editing config.py."""

    def setUp(self):
        from ..__main__ import build_parser, apply_overrides
        self.parse = build_parser().parse_args
        self.apply = apply_overrides
        self.saved = {k: getattr(config, k) for k in
                      ("CUTOFF_START", "CUTOFF_END", "SPORT_CODE",
                       "MATCH_TOLERANCE_SECONDS", "MAX_PAIR_MESSAGE_GAP",
                       "SPREAD_RESOLUTION", "TIME_AXIS", "CLOCK_SOURCE",
                       "MATCH_CHUNK_SIZE")}

    def tearDown(self):
        for key, value in self.saved.items():
            setattr(config, key, value)

    def test_days_gives_a_rolling_window(self):
        self.apply(self.parse(["report", "--days", "3"]))
        start = dt.datetime.strptime(config.CUTOFF_START, "%Y-%m-%d %H:%M:%S")
        age = (dt.datetime.utcnow() - start).total_seconds() / 86400
        self.assertAlmostEqual(age, 3.0, places=2)

    def test_days_beats_since(self):
        self.apply(self.parse(["report", "--since", "2020-01-01 00:00:00",
                               "--days", "1"]))
        self.assertNotIn("2020", config.CUTOFF_START)

    def test_since_and_until(self):
        self.apply(self.parse(["report", "--since", "2026-09-17 10:00:00",
                               "--until", "2026-09-19 00:00:00"]))
        self.assertEqual(config.CUTOFF_START, "2026-09-17 10:00:00")
        self.assertEqual(config.CUTOFF_END, "2026-09-19 00:00:00")

    def test_tuning_flags(self):
        self.apply(self.parse(["report", "--sport", "NB", "--tolerance", "5",
                               "--message-gap", "7", "--time-axis", "drive",
                               "--spread-resolution", "complement",
                               "--chunk", "25"]))
        self.assertEqual(config.SPORT_CODE, "NB")
        self.assertEqual(config.MATCH_TOLERANCE_SECONDS, 5.0)
        self.assertEqual(config.MAX_PAIR_MESSAGE_GAP, 7)
        self.assertEqual(config.TIME_AXIS, "drive")
        self.assertEqual(config.SPREAD_RESOLUTION, "complement")
        self.assertEqual(config.MATCH_CHUNK_SIZE, 25)

    def test_clock_self_means_the_calibrated_stream(self):
        self.apply(self.parse(["report", "--clock", "self"]))
        self.assertIsNone(config.CLOCK_SOURCE)
        self.apply(self.parse(["report", "--clock", "candidate"]))
        self.assertEqual(config.CLOCK_SOURCE, "candidate")

    def test_defaults_are_left_alone(self):
        before = config.CUTOFF_START
        self.apply(self.parse(["report"]))
        self.assertEqual(config.CUTOFF_START, before)

    def test_every_command_accepts_the_window_flags(self):
        for command in ("preflight", "directional", "cross", "report"):
            argv = [command, "--days", "2"]
            if command == "run":
                argv.insert(1, "both")
            self.assertIsNotNone(self.parse(argv))


class TestDailyBreakdown(unittest.TestCase):
    def _at(self, day, **kw):
        p = pair(0.6, 0.8, True, **kw)
        return directional.PairedObservation(
            **{**p.__dict__, "publish_time": dt.datetime(2026, 9, day, 12, 0, 0)})

    def test_groups_by_calendar_day(self):
        pairs = ([self._at(17, match=f"AF{i}") for i in range(3)]
                 + [self._at(18, match=f"AF{i}") for i in range(3, 7)])
        daily = directional.daily_breakdown(pairs, n_bootstrap=20)
        self.assertEqual(sorted(daily), ["2026-09-17", "2026-09-18"])
        self.assertEqual(daily["2026-09-17"]["pairs"], 3)
        self.assertEqual(daily["2026-09-18"]["pairs"], 4)

    def test_day_property_reads_the_publish_time(self):
        self.assertEqual(self._at(19).day, "2026-09-19")

    def test_missing_timestamp_is_not_a_crash(self):
        daily = directional.daily_breakdown([pair(0.6, 0.8, True)], n_bootstrap=20)
        self.assertEqual(list(daily), ["unknown"])


class TestReportShape(unittest.TestCase):
    """The report leads with two views; diagnostics move behind a disclosure."""

    def setUp(self):
        from .. import html_full
        self.html_full = html_full
        self.pairs = []
        for i in range(12):
            self.pairs.append(pair(0.6, 0.8, True, match=f"AF{i}", market_id=50,
                                   period=1 + i % 4, score_diff=(i % 5) * 4 - 8,
                                   team="Home Team" if i % 2 else "Away Team"))
            self.pairs.append(line_pair(0.5, 0.5, 44.5, 44.5, 24, 21,
                                        match=f"AF{i}", market_id=54))
        self.report = directional.build_full_report(self.pairs, n_bootstrap=50)
        self.scan = clean_scan(12)
        self.rendered = self.html_full.render(
            self.report, {"paired_matches": 12},
            {"snapshots": 24, "exact_message_pair": 20, "offset_message_pair": 4},
            self.pairs, self.scan)

    def test_handle_check_reports_clean_when_nothing_flipped(self):
        block = self.rendered[self.rendered.index("Handle check"):]
        block = block[:block.index("</section>")]
        self.assertIn("clean", block)
        self.assertIn("12 matches", block)
        self.assertNotIn('class="bad"', block)

    def test_handle_check_shouts_when_a_match_flipped(self):
        scan = handles.Scan()
        scan.add("OK", [ScoreRow(1, 1, 7, 0, 7, 0)], (7, 0))
        scan.add("SWAP", [ScoreRow(1, 1, 14, 0, 14, 7),
                          ScoreRow(2, 1, None, None, 7, 14)], (7, 14))
        rendered = self.html_full.render(
            self.report, {"paired_matches": 12}, {}, self.pairs, scan.summary())
        block = rendered[rendered.index("Handle check"):]
        block = block[:block.index("</section>")]
        self.assertIn("1 of 2 matches", block)
        self.assertIn('class="bad"', block)
        self.assertIn("SWAP", block)
        self.assertIn("14&ndash;7 &rarr; 7&ndash;14", block)
        # The default is to report rather than drop, so the reader has to be
        # told the bad match is still in the numbers above.
        self.assertIn("STILL IN", block)

    def test_handle_block_shows_the_possession_evidence(self):
        plays, scores = TestPossessionCrossCheck().feed([True, False, True, False])
        crossed = [TestPossessionCrossCheck.td(s.event_message_count,
                                               not bool(s.p1_change))
                   for s in scores]
        scan = handles.Scan()
        scan.add("OK", scores, None, plays)
        scan.add("XED", crossed, None, plays)
        rendered = self.html_full.render(
            self.report, {"paired_matches": 12}, {}, self.pairs, scan.summary())
        block = rendered[rendered.index("Handle check"):]
        block = block[:block.index("</section>")]
        self.assertIn("Possession inverted", block)
        self.assertIn("0/4 touchdowns agree", block)
        self.assertIn("touchdown anchors", block)
        # A whole-match verdict is not a message count and must not be
        # labelled as the final cross-check either.
        self.assertIn("<td>match</td>", block)

    def test_a_flip_stops_the_checks_line_claiming_all_pass(self):
        scan = handles.Scan()
        scan.add("SWAP", [ScoreRow(1, 1, 14, 0, 14, 7),
                          ScoreRow(2, 1, None, None, 7, 14)], (7, 14))
        summary = self.html_full._checks_summary(self.report, scan.summary())
        self.assertIn("flipped handles", summary)
        self.assertIn("bad", summary)
        self.assertNotIn("all pass", summary)

    def test_cross_section_is_the_three_axes_together(self):
        self.assertIn("Cross-section calibration", self.rendered)
        self.assertIn("score diff &times; quarter &times; possession", self.rendered)

    def test_cross_section_covers_every_market_not_just_moneyline(self):
        cells = self.report["full_cell"]
        markets_present = {key[1] for key in cells}
        self.assertIn("moneyline", markets_present)
        self.assertIn("total", markets_present)

    def test_diagnostics_are_behind_a_disclosure(self):
        self.assertIn("<details", self.rendered)
        self.assertIn("<summary>Checks &mdash;", self.rendered)
        # And they are still present, not dropped.
        self.assertIn("Mirror check", self.rendered)
        self.assertIn("Integrity checks", self.rendered)

    def test_cross_section_axes_are_three_shaded_columns(self):
        head = self.rendered[self.rendered.index('id="crossTable"'):]
        head = head[:head.index("</thead>")]
        for name in ("Score diff", "Quarter", "Possession"):
            self.assertIn(f'class="ax"', head)
            self.assertIn(name, head)
        self.assertNotIn(">Cell<", head)
        # The shade is a token so it survives both themes.
        self.assertIn("td.ax,th.ax{background:var(--axis)", self.rendered)
        self.assertIn("--axis:", self.rendered)

    def test_axis_cells_sort_in_game_order_not_alphabetically(self):
        # "Tight" sorts last alphabetically but sits in the middle of the
        # game, so the axis cells carry their bucket index for the sorter.
        body = self.rendered[self.rendered.index('id="crossTable"'):]
        body = body[:body.index("</table>")]
        self.assertIn('class="ax" data-v=', body)

    def test_same_line_headline_drops_the_mae_row(self):
        head = self.rendered[self.rendered.index('id="directional"'):]
        head = head[:head.index("</section>")]
        self.assertNotIn("&Delta;MAE", head)
        self.assertIn("&Delta;Brier / match", head)

    def test_report_carries_no_explanatory_prose(self):
        # The report is a dashboard, not a write-up: column meanings live in
        # header tooltips so the tables stay readable.
        self.assertNotIn('class="note"', self.rendered)
        self.assertNotIn('class="sub"', self.rendered)
        self.assertGreater(self.rendered.count("<th title="), 20)

    def test_nav_names_the_two_headline_views(self):
        nav = self.rendered[self.rendered.index("<nav>"):self.rendered.index("</nav>")]
        self.assertIn("Directional calibration", nav)
        self.assertIn("Cross-section calibration", nav)

    def test_thin_cells_are_dimmed_not_dropped(self):
        # These cells sit well under MIN_CELL_MATCHES, so they must still be
        # there -- knowing a bucket is thin is part of the information.
        self.assertIn('class="thin"', self.rendered)

    def test_cross_table_is_sortable(self):
        self.assertIn('id="crossTable"', self.rendered)
        self.assertIn("crossTable", self.rendered.split("<script>")[1])

    def test_checks_summary_reports_clean_when_nothing_is_wrong(self):
        summary = self.html_full._checks_summary(self.report)
        self.assertIn("all pass", summary)

    def test_checks_summary_flags_a_partition_failure(self):
        report = dict(self.report)
        report["complement"] = {"spread": {"both_sides": 100,
                                           "outcomes_partition": 80,
                                           "prob_sum_mean": 1.0}}
        summary = self.html_full._checks_summary(report)
        self.assertIn("does not partition", summary)

    def test_axes_flag_defaults_off(self):
        from ..__main__ import build_parser
        args = build_parser().parse_args(["report"])
        self.assertFalse(args.axes)
        self.assertTrue(build_parser().parse_args(["report", "--axes"]).axes)


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
