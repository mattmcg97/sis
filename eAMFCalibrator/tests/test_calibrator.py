"""Unit tests for everything in the suite that does not need Snowflake.

The database half cannot be exercised here, so the logic that decides what
a row MEANS -- line parsing, market resolution, drive cleaning, quote
matching, scoring -- is pinned down in full instead.

Run with:  py -m unittest discover eAMFCalibrator
"""

import builtins
import collections
import contextlib
import dataclasses
import re
import datetime as dt
import io
import os
import unittest
from unittest import mock

from .. import (buckets, clock, config, directional, drives, handles,
                indrive, markets, metrics, report)
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

    def test_where_distinguishes_a_message_from_the_final_check(self):
        scores = [ScoreRow(1, 1, None, None, 14, 7),
                  ScoreRow(2, 1, None, None, 7, 14)]
        self.assertEqual(handles.scan_match("X", scores)[0].where, "2")
        late = handles.scan_match("X", [ScoreRow(1, 1, None, None, 7, 0)], (14, 0))
        self.assertEqual(late[0].where, "final")
        self.assertTrue(late[0].is_final_check)

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
                             60.0, 1.67, "PLAYER 1", msg, "open", "true"))
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


class TestTheLineOverrulesTheProbability(unittest.TestCase):
    """A different line settles the pair; the probability only breaks a tie."""

    def test_a_differing_line_decides_even_when_the_probability_disagrees(self):
        # Realized total 45. Prod's 44.5 is 0.5 away, the candidate's 60.5 is
        # 15.5 away. The candidate called its own under at 0.01 and was
        # right, so its probability error is tiny and prod's is 0.50 -- the
        # candidate looks far better on probability. The line must still win.
        p = line_pair(0.50, 0.01, 44.5, 60.5, 24, 21)
        self.assertFalse(p.same_line)
        self.assertEqual(p.decisive_mode, directional.LINE)
        self.assertEqual(p.winner(directional.PROBABILITY), "candidate")
        self.assertEqual(p.winner(directional.LINE), "prod")
        self.assertEqual(p.decisive_winner, "prod")

    def test_a_shared_line_is_decided_on_probability(self):
        p = line_pair(0.55, 0.62, 44.5, 44.5, 24, 21)
        self.assertTrue(p.same_line)
        self.assertEqual(p.decisive_mode, directional.PROBABILITY)
        self.assertEqual(p.decisive_winner, p.winner(directional.PROBABILITY))

    def test_probability_breaks_an_exact_tie_on_the_line(self):
        # Whole-number lines straddling a realized 45: both exactly 1.0 away,
        # so the line settles nothing -- but each stream was still graded
        # against its own outcome, so the probabilities are a real comparison.
        p = line_pair(0.90, 0.90, 44.0, 46.0, 24, 21)
        self.assertEqual(p.prod_line_error, p.candidate_line_error)
        self.assertEqual(p.winner(directional.LINE), "tie")
        self.assertEqual((p.prod_outcome, p.candidate_outcome), (True, False))
        self.assertEqual(p.decisive_winner, p.winner(directional.PROBABILITY))

    def test_a_tie_broken_pair_is_counted_as_settled_on_probability(self):
        # decisive_mode says LINE -- the lines do differ -- but the line
        # tied, so it was the probability that decided. The block counts
        # what actually decided, not what was tried first.
        tied = line_pair(0.90, 0.90, 44.0, 46.0, 24, 21)
        self.assertEqual(tied.decisive_mode, directional.LINE)
        self.assertEqual(tied.decided_by, directional.PROBABILITY)
        block = directional.decisive_block([tied])
        self.assertEqual(block["settled_on_line"], 0)
        self.assertEqual(block["settled_on_probability"], 1)

    def test_errors_under_decisive_follow_the_pair(self):
        same = line_pair(0.55, 0.62, 44.5, 44.5, 24, 21)
        diff = line_pair(0.55, 0.62, 44.5, 60.5, 24, 21)
        self.assertEqual(same.errors(directional.DECISIVE),
                         same.errors(directional.PROBABILITY))
        self.assertEqual(diff.errors(directional.DECISIVE),
                         diff.errors(directional.LINE))

    def test_tally_refuses_to_mix_units(self):
        # A line error is in points and a probability error is not, so the
        # sums tally keeps would be adding different units together.
        with self.assertRaises(ValueError):
            directional.tally([line_pair(0.5, 0.5, 44.5, 44.5, 24, 21)],
                              directional.DECISIVE)


class TestDecisiveBlock(unittest.TestCase):
    def setUp(self):
        # Ten matches. The candidate edges prod on one same-line pair each,
        # and is beaten on three different-line pairs each.
        self.pairs = [line_pair(0.50, 0.52, 44.5, 44.5, 24, 21, match=f"AF{i}")
                      for i in range(10)]
        # The candidate wins these on probability and loses them on the
        # line, so prod carrying the combined result is the line overruling.
        self.pairs += [line_pair(0.50, 0.01, 44.5, 60.5, 24, 21, match=f"AF{i}")
                       for i in range(10) for _ in range(3)]
        self.report = directional.build_full_report(self.pairs, n_bootstrap=100)
        self.block = self.report["summary"]["decisive"]

    def test_it_covers_every_pair_not_just_the_same_line_half(self):
        self.assertEqual(self.block["n"], len(self.pairs))
        self.assertEqual(self.block["settled_on_probability"], 10)
        self.assertEqual(self.block["settled_on_line"], 30)

    def test_the_lines_carry_the_combined_result(self):
        # The candidate wins on probability both in the same-line half and
        # among the different-line pairs, and still loses overall, because
        # the different-line pairs are decided on the line.
        self.assertGreater(self.report["summary"]["same_line"]["brier"]["mean"], 0)
        different = [p for p in self.pairs if not p.same_line]
        self.assertTrue(all(p.winner(directional.PROBABILITY) == "candidate"
                            for p in different))
        self.assertEqual(self.block["votes"]["prod"], 10)
        self.assertEqual(self.block["votes"]["candidate"], 0)

    def test_it_reports_no_mean_error(self):
        # There is no average of a points error and a probability error, so
        # the block must not offer one.
        for key in ("brier", "mae", "prod_error_sum", "candidate_error_sum"):
            self.assertNotIn(key, self.block)

    def test_the_verdict_lets_the_lines_overrule_the_same_line_brier(self):
        from .. import html_full
        cls, text = html_full._verdict(self.report)
        self.assertEqual(cls, "bad")
        self.assertIn("<b>PROD</b>", text)
        # Both readings stay on the line, so a disagreement between them
        # is visible in the numbers rather than narrated away.
        self.assertIn("same line", text)
        self.assertIn("&Delta;Brier", text)

    def test_the_verdict_still_uses_the_brier_when_the_combined_read_is_flat(self):
        from .. import html_full
        pairs = [pair(0.80, 0.20, True, match=f"AF{i}", message=100 + j)
                 for i in range(12) for j in range(3)]
        report = directional.build_full_report(pairs, n_bootstrap=100)
        cls, text = html_full._verdict(report)
        self.assertEqual(cls, "bad")
        # Both readings are on the line, so neither is hidden behind the
        # other: the same-line Brier and its interval are always shown.
        self.assertIn("&Delta;Brier", text)
        self.assertIn("CI excludes 0", text)


class TestSelectionBlocks(unittest.TestCase):
    """Both sides of every market, broken out rather than pooled."""

    def setUp(self):
        self.pairs = []
        for i in range(6):
            for market_id in (50, 51):
                self.pairs.append(pair(0.6, 0.7, True, match=f"AF{i}",
                                       market_id=market_id, message=100 + i))
            self.pairs.append(line_pair(0.5, 0.52, 44.5, 44.5, 24, 21,
                                        match=f"AF{i}", market_id=54))
            self.pairs.append(line_pair(0.5, 0.52, 44.5, 44.5, 24, 21,
                                        match=f"AF{i}", market_id=55))
        self.blocks = directional.selection_blocks(
            self.pairs, directional.PROBABILITY, n_bootstrap=50)

    def test_every_selection_gets_its_own_row(self):
        self.assertEqual(sorted(self.blocks), [50, 51, 54, 55])

    def test_each_row_names_its_market_and_side(self):
        self.assertEqual(self.blocks[50]["market"], "moneyline")
        self.assertEqual(self.blocks[50]["selection"], "Home")
        self.assertEqual(self.blocks[51]["selection"], "Away")
        self.assertEqual(self.blocks[55]["selection"], "Under")

    def test_it_marks_which_side_the_pooled_tables_read(self):
        self.assertTrue(self.blocks[50]["canonical"])
        self.assertFalse(self.blocks[51]["canonical"])

    def test_each_side_carries_its_own_clustered_test(self):
        for row in self.blocks.values():
            self.assertIn("mean", row["brier"])
            self.assertIn("p_value", row["brier"])
            self.assertEqual(row["tally"]["n_matches"], 6)

    def test_a_fault_on_one_side_only_survives_the_breakdown(self):
        # Pooled by market the two sides cancel to nothing; split by
        # selection the damage is visible on exactly one of them. This is
        # the whole reason the table exists.
        # Prod sits on 0.5 either side, so its Brier is 0.25 both times.
        # The candidate is 0.1 out on Home (+0.24) and 0.7 out on Away
        # (-0.24), which cancel exactly when the market is pooled.
        pairs = []
        for i in range(8):
            pairs.append(pair(0.5, 0.9, True, match=f"AF{i}", market_id=50))
            pairs.append(pair(0.5, 0.3, True, match=f"AF{i}", market_id=51))
        pooled = directional.market_blocks(
            pairs, directional.PROBABILITY, n_bootstrap=50)["moneyline"]
        split = directional.selection_blocks(
            pairs, directional.PROBABILITY, n_bootstrap=50)
        self.assertAlmostEqual(pooled["brier"]["mean"], 0.0, places=9)
        self.assertAlmostEqual(split[50]["brier"]["mean"], +0.24, places=9)
        self.assertAlmostEqual(split[51]["brier"]["mean"], -0.24, places=9)

    def test_the_summary_carries_them_for_both_views(self):
        summary = directional.build_summary(self.pairs, n_bootstrap=50)
        self.assertIn("selections", summary["same_line"])
        self.assertIn("selections", summary["different_line"])


class TestAxisBreakdownsShowEverySelection(unittest.TestCase):
    """The single-axis tables in the checks carry both sides."""

    def setUp(self):
        self.pairs = []
        for i in range(8):
            # Home wins six of the eight, so the two sides have genuinely
            # different realized rates and cancellation would be visible.
            home_won = i < 6
            for market_id in (50, 51):
                outcome = home_won if market_id == 50 else not home_won
                self.pairs.append(pair(0.5, 0.6, outcome, match=f"AF{i}",
                                       market_id=market_id, period=1 + i % 4))
        self.cells = directional.calibration_cells(
            self.pairs, directional.quarter_label, n_bootstrap=50,
            by_selection=True)

    def test_both_sides_of_a_market_get_their_own_cell(self):
        ids = {key[1] for key in self.cells}
        self.assertEqual(ids, {50, 51})

    def test_keying_by_selection_does_not_cancel_the_measurement(self):
        # Pooling the two sides forces EVERY cell's realized rate to exactly
        # 0.500 by construction. Keyed separately they are complements of
        # each other, and at least one cell escapes 0.500 -- which is the
        # whole property. (A cell can legitimately land on 0.500 when the
        # outcomes in it really did split evenly, so the test is that not
        # all of them do, not that none of them does.)
        realized = []
        for cell in {key[0] for key in self.cells}:
            home = self.cells.get((cell, 50))
            away = self.cells.get((cell, 51))
            if not home or not away:
                continue
            self.assertAlmostEqual(home["realized"] + away["realized"], 1.0,
                                   places=9, msg=cell)
            realized.append(home["realized"])
        self.assertTrue(any(abs(r - 0.5) > 1e-9 for r in realized), realized)

    def test_pooling_both_sides_would_cancel_it_which_is_why_it_is_keyed(self):
        # The counterfactual, asserted rather than asserted-about: feed the
        # same pairs through one shared bucket and everything collapses.
        pooled = directional.calibration_cells(
            self.pairs, lambda p: "all", n_bootstrap=20,
            market_ids=[50, 51])
        for row in pooled.values():
            self.assertAlmostEqual(row["realized"], 0.5, places=9)

    def test_each_cell_names_its_market_and_side(self):
        row = self.cells[("Q1", 51)]
        self.assertEqual(row["market"], "moneyline")
        self.assertEqual(row["selection"], "Away")
        self.assertFalse(row["canonical"])
        self.assertTrue(self.cells[("Q1", 50)]["canonical"])

    def test_the_pooled_cross_section_still_reads_one_side(self):
        # The headline view is unchanged: only the checks were widened.
        pooled = directional.calibration_cells(
            self.pairs, directional.quarter_label, n_bootstrap=50)
        self.assertEqual({key[1] for key in pooled}, {"moneyline"})


class TestPairStateColumns(unittest.TestCase):
    """Field position and down/distance ride on the pair."""

    def test_they_default_to_none_when_the_feed_has_none(self):
        p = pair(0.5, 0.6, True)
        self.assertIsNone(p.field_position)
        self.assertIsNone(p.down_number)
        self.assertIsNone(p.distance)

    def test_they_reach_the_csv(self):
        for field in ("field_position", "down_number", "distance"):
            self.assertIn(field, report.PAIR_FIELDS)


class TestMarketStateFlag(unittest.TestCase):
    """A non-live quote is carried, flagged, and scored by nothing."""

    def dead(self, prod_live=True, candidate_live=True, **kw):
        return dataclasses.replace(
            pair(0.9, 0.1, True, **kw),
            prod_live=prod_live, candidate_live=candidate_live)

    def test_is_active_alone_decides(self):
        self.assertTrue(directional.is_live("true"))
        self.assertTrue(directional.is_live("TRUE"))
        for active in ("false", "FALSE", "", None):
            self.assertFalse(directional.is_live(active), repr(active))

    def test_either_side_not_live_marks_the_pair(self):
        self.assertFalse(self.dead().not_live)
        self.assertTrue(self.dead(prod_live=False).not_live)
        self.assertTrue(self.dead(candidate_live=False).not_live)
        self.assertTrue(self.dead(False, False).not_live)

    def test_a_non_live_pair_is_scored_by_nothing(self):
        p = self.dead(prod_live=False)
        for mode in (directional.PROBABILITY, directional.LINE,
                     directional.DECISIVE):
            self.assertEqual(p.errors(mode), (None, None), mode)
            self.assertFalse(p.comparable(mode), mode)
            self.assertIsNone(p.winner(mode), mode)

    def test_it_is_excluded_from_every_aggregate_at_once(self):
        # errors() is what tally, the cells, the votes and the decisive
        # block are all built on, which is why refusing there is enough.
        live = [pair(0.9, 0.1, True, match=f"AF{i}") for i in range(6)]
        dead = [self.dead(prod_live=False, match=f"AF{i}") for i in range(6)]
        both = live + dead
        self.assertEqual(directional.tally(both, directional.PROBABILITY)["n"], 6)
        self.assertEqual(directional.decisive_block(both)["n"], 6)
        cells = directional.calibration_cells(
            both, directional.quarter_label, n_bootstrap=20)
        self.assertEqual(sum(c["n"] for c in cells.values()), 6)

    def test_turning_the_requirement_off_scores_them(self):
        previous = config.REQUIRE_LIVE_QUOTE
        config.REQUIRE_LIVE_QUOTE = False
        try:
            p = self.dead(prod_live=False)
            self.assertTrue(p.comparable(directional.PROBABILITY))
            self.assertIsNotNone(p.winner(directional.PROBABILITY))
        finally:
            config.REQUIRE_LIVE_QUOTE = previous

    def test_the_state_string_is_kept_verbatim(self):
        # STATUS no longer decides anything, but it is still recorded
        # beside IS_ACTIVE: side by side is how a disagreement between the
        # two columns stays visible.
        self.assertEqual(directional.state_label("UNDER SETTLEMENT", "false"),
                         "UNDER SETTLEMENT/false")
        self.assertEqual(directional.state_label("open", "true"), "open/true")

    def test_closed_but_active_is_live_now_that_status_does_not_vote(self):
        # The one combination the change actually moves. GAMEPLAI say the
        # status column is wrong, so a row it calls CLOSED while IS_ACTIVE
        # is true is a price somebody could have taken.
        self.assertTrue(directional.is_live("true"))

    def test_settlement_is_still_dead_because_is_active_says_so(self):
        # Nothing moves the other way: the 83% of rows that read
        # UNDER SETTLEMENT / false are still out, on IS_ACTIVE alone.
        self.assertFalse(directional.is_live("false"))

    def test_the_index_prefers_a_live_row_on_is_active_alone(self):
        # A message carrying a settled row and an active one must keep the
        # active one, and that choice is now made without reading STATUS.
        when = dt.datetime(2026, 9, 18, 12, 0)
        rows = [("AF1", 52, when, 61.0, 2.0, "PLAYER 1 -3.5", 6,
                 "UNDER SETTLEMENT", "false"),
                ("AF1", 52, when, 48.0, 2.1, "PLAYER 1 -6.5", 6,
                 "CLOSED", "true")]
        index = directional.index_by_message(rows)
        kept = index[("AF1", 52)][6]
        self.assertEqual(kept.probability, 48.0)
        self.assertTrue(kept.live)
        self.assertEqual(kept.state, "CLOSED/true")

    def test_the_report_breaks_down_by_state_and_by_stream(self):
        pairs = [
            dataclasses.replace(pair(0.9, 0.1, True, match="AF0"),
                                prod_live=False,
                                prod_state="UNDER SETTLEMENT/false"),
            dataclasses.replace(pair(0.9, 0.1, True, match="AF1"),
                                candidate_live=False,
                                candidate_state="CLOSED/false"),
        ]
        r = directional.market_state_report(pairs)
        self.assertEqual(r["by_state"][("prod", "UNDER SETTLEMENT/false")], 1)
        self.assertEqual(r["by_state"][("candidate", "CLOSED/false")], 1)

    def test_the_report_counts_which_side_went_non_live(self):
        pairs = ([pair(0.9, 0.1, True, match="AF0")]
                 + [self.dead(prod_live=False, match="AF1")] * 3
                 + [self.dead(candidate_live=False, match="AF2")]
                 + [self.dead(False, False, match="AF3")])
        r = directional.market_state_report(pairs)
        self.assertEqual((r["pairs"], r["not_live"]), (6, 5))
        self.assertEqual(r["prod_only"], 3)
        self.assertEqual(r["candidate_only"], 1)
        self.assertEqual(r["both"], 1)
        self.assertEqual(r["matches"], 3)

    def test_the_report_splits_by_quarter_so_lopsidedness_is_visible(self):
        pairs = [self.dead(prod_live=(q != 4), match=f"AF{q}", period=q)
                 for q in (1, 2, 3, 4)]
        r = directional.market_state_report(pairs)
        self.assertEqual(r["by_quarter"]["Q4"], (1, 1))
        self.assertEqual(r["by_quarter"]["Q1"], (1, 0))


class RecordingCursor:
    """A cursor that records what was asked and answers with nothing.

    Enough to exercise every query the pipeline builds without a database:
    the shape of the SQL and its bind parameters are what break, and both
    are visible here.
    """

    def __init__(self, rows=()):
        self.calls = []
        self._rows = list(rows)
        self.description = [("A",), ("B",), ("C",), ("D",), ("E",), ("F",),
                            ("G",), ("H",), ("I",)]

    def execute(self, sql, params=()):
        self.calls.append((sql, params))

    def fetchall(self):
        return self._rows


class TestGeneratedQueries(unittest.TestCase):
    """Every query binds as many parameters as it has placeholders.

    A mismatch is the failure mode that a dialect parse cannot catch and
    that only shows up against a live warehouse, which this suite has no
    access to.
    """

    def each_query(self):
        """(name, sql, params) for every query the io layer builds."""
        from .. import snowflake_io as io
        cases = [
            ("describe_columns", lambda c: io.describe_columns(c, "T")),
            ("match_universe", lambda c: io.match_universe(c, "S")),
            ("team_vocabulary", lambda c: io.team_vocabulary(c, "S")),
            ("market_descriptions", lambda c: io.market_descriptions(c, "S")),
            ("team_join_test", lambda c: io.team_join_test(c, "S")),
            ("fetch_plays", lambda c: io.fetch_plays(c, ["A", "B"], "FILE_TIME")),
            ("fetch_scores", lambda c: io.fetch_scores(c, ["A", "B"])),
            ("fetch_final_scores", lambda c: io.fetch_final_scores(c, ["A"])),
            ("fetch_quotes", lambda c: io.fetch_quotes(c, "S", ["A", "B"])),
            ("fetch_message_times", lambda c: io.fetch_message_times(c, "S", ["A"])),
            ("stream_window_summary", lambda c: io.stream_window_summary(c, "S")),
            ("status_profile", lambda c: io.status_profile(c, "S")),
            ("rows_per_message", lambda c: io.rows_per_message(c, "S")),
        ]
        for name, call in cases:
            cursor = RecordingCursor()
            try:
                call(cursor)
            except (IndexError, TypeError, ValueError):
                pass    # unpacking an empty result is not what is under test
            for sql, params in cursor.calls:
                yield name, sql, params

    def test_placeholders_match_the_bound_parameters(self):
        seen = 0
        for name, sql, params in self.each_query():
            seen += 1
            self.assertEqual(sql.count("%s"), len(params or ()), name)
        self.assertGreaterEqual(seen, 13)

    def test_every_query_is_fully_interpolated(self):
        # An unresolved brace means an f-string placeholder was left behind,
        # which a warehouse would reject and a dialect parse might not.
        for name, sql, _ in self.each_query():
            self.assertNotIn("{", sql, name)

    def test_the_new_team_queries_are_scoped_to_the_sport(self):
        from .. import snowflake_io as io
        for name, sql, params in self.each_query():
            if name in ("team_vocabulary", "team_join_test"):
                self.assertIn("SPORT_CODE", sql, name)
                self.assertEqual(list(params), [config.SPORT_CODE] * sql.count("%s"))

    def test_no_query_decides_liveness_on_status(self):
        # The quote STATUS column is wrong, per GAMEPLAI, so nothing may
        # test it. It is still SELECTed and still reported -- it just gets
        # no vote. INPLAY_EVENT_STATUS is a different column, the match's
        # own lifecycle, and picking settled matches still depends on it.
        quote_status = re.compile(r"(?<![A-Z_])STATUS\s*=")
        for name, sql, _ in self.each_query():
            self.assertIsNone(quote_status.search(sql), name)

    def test_the_live_count_is_read_from_is_active_alone(self):
        for name, sql, params in self.each_query():
            if name == "rows_per_message":
                self.assertIn("SUM(CASE WHEN IS_ACTIVE = %s", sql)
                self.assertEqual(params[0], config.LIVE_IS_ACTIVE)

    def test_quotes_are_no_longer_filtered_on_status_in_sql(self):
        # The filter moved onto the pair so what it drops can be counted.
        for name, sql, _ in self.each_query():
            if name == "fetch_quotes":
                self.assertNotIn("STATUS = ", sql)
                self.assertIn("STATUS", sql)      # still selected
                self.assertIn("IS_ACTIVE", sql)


class TestSnapshotAnchor(unittest.TestCase):
    """Where in a drive the snapshot is taken from."""

    HOME, AWAY = "Home Team", "Away Team"

    def snaps(self, plays, scores=()):
        return build_snapshots("AF1", plays, list(scores))

    def test_a_clean_drive_anchors_on_its_first_down(self):
        s = self.snaps([PlayRow(1, 1, self.HOME, 1, 10, 25),
                        PlayRow(2, 1, self.HOME, 2, 4, 31)])
        self.assertEqual(s[0].anchor, drives.FIRST_DOWN)
        self.assertEqual((s[0].down_number, s[0].distance), (1, 10))

    def test_it_walks_past_kick_mechanics_to_the_real_start(self):
        # The general cleaning rule drops only ONE row per team change, so
        # a transition carrying several kick rows leaves the rest behind.
        # Without the walk the snapshot sits on 4th-and-99 at field 35,
        # with a team label belonging to the kick.
        s = self.snaps([
            PlayRow(1, 1, self.HOME, 1, 10, 25), PlayRow(2, 1, self.HOME, 2, 4, 31),
            PlayRow(3, 1, self.AWAY, 2, 4, 31),     # stale duplicate, dropped
            PlayRow(4, 1, self.AWAY, 4, 99, 35),    # kick mechanic, survives
            PlayRow(5, 1, self.AWAY, 1, 10, 22),    # the real drive start
            PlayRow(6, 1, self.AWAY, 2, 6, 26)])
        self.assertEqual(s[1].event_message_count, 5)
        self.assertEqual((s[1].down_number, s[1].distance), (1, 10))
        self.assertEqual(s[1].field_position, 22)
        self.assertEqual(s[1].anchor, drives.FIRST_DOWN)

    def test_it_does_not_jump_to_a_first_down_conversion(self):
        # This run opens 2nd and 7: the drive's start was already lost.
        # The next 1st-and-10 in it is a CONVERSION -- a real state, but
        # not a drive start -- so the anchor stops at the first real snap
        # and says so rather than quietly relabelling a mid-drive play.
        s = self.snaps([
            PlayRow(1, 1, self.HOME, 1, 10, 25),
            PlayRow(2, 1, self.AWAY, 1, 10, 25),    # stale duplicate, dropped
            PlayRow(3, 1, self.AWAY, 2, 7, 28), PlayRow(4, 1, self.AWAY, 3, 2, 33),
            PlayRow(5, 1, self.AWAY, 4, 1, 34),
            PlayRow(6, 1, self.AWAY, 1, 10, 41)])   # the conversion
        self.assertEqual(s[1].event_message_count, 3)
        self.assertEqual(s[1].anchor, drives.MID_DRIVE)

    def test_a_run_with_no_real_snap_is_dropped_entirely(self):
        # Every row of the second team's run is unreadable, so there is no
        # drive to anchor rather than a drive anchored badly.
        s = self.snaps([
            PlayRow(1, 1, self.HOME, 1, 10, 25),
            PlayRow(2, 1, self.AWAY, None, None, None),
            PlayRow(3, 1, self.AWAY, 5, 99, 35)])
        self.assertEqual(len(s), 1)
        self.assertEqual(s[0].offensive_team, self.HOME)

    def test_what_counts_as_a_snap(self):
        def play(down, distance):
            return PlayRow(1, 1, self.HOME, down, distance, 25)
        for down, distance in ((1, 10), (4, 1), (1, 30), (2, 0)):
            self.assertTrue(drives.is_snap(play(down, distance)),
                            f"{down}&{distance}")
        for down, distance in ((5, 10), (0, 10), (None, 10), (1, None),
                               (1, 99), (1, -1)):
            self.assertFalse(drives.is_snap(play(down, distance)),
                             f"{down}&{distance}")


class TestQuotePreference(unittest.TestCase):
    """Which row wins when a message carries more than one for a market."""

    @staticmethod
    def row(message, probability, status, active, when=1):
        return ("AF1", 52, dt.datetime(2026, 9, 18, 12, when), probability,
                2.0, "PLAYER 1 -3.5", message, status, active)

    def test_a_live_row_beats_a_dead_one_published_earlier(self):
        # The old line settling is published before the new one opens, so
        # keeping the first row kept the dead one -- and the pair with it.
        index = directional.index_by_message([
            self.row(100, 55.0, "UNDER SETTLEMENT", "false", when=1),
            self.row(100, 48.0, "open", "true", when=2),
        ])
        quote = index[("AF1", 52)][100]
        self.assertTrue(quote.live)
        self.assertEqual(quote.probability, 48.0)

    def test_a_dead_row_never_displaces_a_live_one(self):
        index = directional.index_by_message([
            self.row(100, 48.0, "open", "true", when=1),
            self.row(100, 55.0, "UNDER SETTLEMENT", "false", when=2),
        ])
        self.assertEqual(index[("AF1", 52)][100].probability, 48.0)

    def test_among_equals_the_earliest_still_wins(self):
        # Two live rows: keep the one nearest the event, not a later
        # correction to it.
        index = directional.index_by_message([
            self.row(100, 48.0, "open", "true", when=1),
            self.row(100, 49.0, "open", "true", when=2),
        ])
        self.assertEqual(index[("AF1", 52)][100].probability, 48.0)
        # And the same when both are dead, so behaviour is unchanged there.
        index = directional.index_by_message([
            self.row(100, 55.0, "CLOSED", "false", when=1),
            self.row(100, 56.0, "UNDER SETTLEMENT", "false", when=2),
        ])
        self.assertEqual(index[("AF1", 52)][100].probability, 55.0)

    def test_it_counts_what_it_saw(self):
        stats = collections.defaultdict(int)
        directional.index_by_message([
            self.row(100, 55.0, "UNDER SETTLEMENT", "false", when=1),
            self.row(100, 48.0, "open", "true", when=2),
            self.row(101, 50.0, "open", "true", when=3),
        ], stats)
        self.assertEqual(stats["quote_rows_sharing_a_message"], 1)
        self.assertEqual(stats["quote_upgraded_to_live"], 1)

    def test_stats_are_optional(self):
        index = directional.index_by_message([self.row(100, 48.0, "open", "true")])
        self.assertEqual(index[("AF1", 52)][100].probability, 48.0)


class TestCleaningExplainsItself(unittest.TestCase):
    """classify_plays is the single source of truth for the cleaning."""

    HOME, AWAY = "Home Team", "Away Team"

    def feed(self):
        plays = [PlayRow(1, 1, self.HOME, 1, 10, 25),
                 PlayRow(2, 1, self.HOME, 2, 4, 31),
                 PlayRow(3, 1, self.HOME, 3, 1, 34),
                 PlayRow(4, 1, self.AWAY, 3, 1, 34),    # stale duplicate
                 PlayRow(5, 1, self.AWAY, 4, 99, 35),   # kick mechanic
                 PlayRow(6, 1, self.AWAY, 1, 10, 22),   # resume after the TD
                 PlayRow(7, 1, self.AWAY, 2, 6, 26)]
        return plays, [ScoreRow(3, 1, 6, None, 6, 0)]

    def test_every_row_gets_a_verdict(self):
        plays, scores = self.feed()
        reasons = drives.classify_plays(plays, scores)
        # msg 1 opens the match at a fresh 1st-and-10, so the opening
        # kickoff rule finds its drive immediately and strips nothing.
        self.assertEqual(reasons[1], drives.DRIVE_START)
        # Away's label on Home's 3rd-and-1 at the 34.
        self.assertEqual(reasons[4], drives.STALE_AFTER_CHANGE)
        # 4th-and-99 is no readable down, so it is not scrimmage.
        self.assertEqual(reasons[5], drives.KICKOFF)
        self.assertEqual(reasons[6], drives.DRIVE_START)
        self.assertEqual(reasons[7], drives.KEPT)

    def test_clean_plays_agrees_with_the_verdicts(self):
        # The two must not be able to drift: one is built from the other.
        plays, scores = self.feed()
        reasons = drives.classify_plays(plays, scores)
        cleaned, dropped = drives.clean_plays(plays, scores)
        kept = {p.event_message_count for p in cleaned}
        self.assertEqual(
            kept, {m for m, r in reasons.items() if not drives.was_dropped(r)})
        self.assertEqual(dropped, len(plays) - len(cleaned))

    def test_a_drive_start_counts_as_kept(self):
        self.assertFalse(drives.was_dropped(drives.DRIVE_START))
        self.assertFalse(drives.was_dropped(drives.KEPT))
        self.assertTrue(drives.was_dropped(drives.STALE_AFTER_CHANGE))
        self.assertTrue(drives.was_dropped(drives.KICKOFF))


class TestDump(unittest.TestCase):
    """The play-by-play CSV, built from the pipeline's own functions."""

    HOME, AWAY = "Home Team", "Away Team"
    T = dt.datetime(2026, 9, 18, 12, 0)

    def quote(self, message, market_id, probability, status, active,
              description):
        return ("AF1", market_id, self.T + dt.timedelta(seconds=message),
                probability, 2.0, description, message, status, active)

    def indexes(self):
        from .. import directional
        return {
            directional.PROD: directional.index_by_message([
                self.quote(1, 50, 55.0, "open", "true", "PLAYER 1 to win"),
                self.quote(1, 52, 61.0, "open", "true", "PLAYER 1 -3.5"),
                self.quote(6, 52, 48.0, "UNDER SETTLEMENT", "false",
                           "PLAYER 1 -6.5"),
            ]),
            directional.CANDIDATE: directional.index_by_message([
                self.quote(1, 50, 56.0, "open", "true", "PLAYER 1 to win"),
                self.quote(1, 52, 59.0, "open", "true", "PLAYER 1 -6.5"),
            ]),
        }

    def rows(self, indexes=None):
        from .. import dump
        plays = [("AF1", 1, 1, self.HOME, 1, 10, 25, None),
                 ("AF1", 2, 1, self.HOME, 2, 4, 31, None),
                 ("AF1", 3, 1, self.HOME, 3, 1, 34, None),
                 ("AF1", 4, 1, self.AWAY, 3, 1, 34, None),
                 ("AF1", 5, 1, self.AWAY, 4, 99, 35, None),
                 ("AF1", 6, 1, self.AWAY, 1, 10, 22, None),
                 ("AF1", 7, 1, self.AWAY, 2, 6, 26, None)]
        scores = [("AF1", 3, 1, 6, None, 6, 0)]
        return dump._rows_for_match("AF1", plays, scores, indexes)

    def test_every_play_row_survives_into_the_dump(self):
        plays, _ = self.rows()
        self.assertEqual(len(plays), 7)
        # Messages 4 and 5 are the kick between the touchdown and the
        # receiving team's first snap.
        self.assertEqual(sum(r["dropped"] for r in plays), 2)
        self.assertEqual({r["event_message_count"] for r in plays if r["dropped"]},
                         {4, 5})

    def test_dropped_rows_carry_the_rule_that_dropped_them(self):
        plays, _ = self.rows()
        by_message = {r["event_message_count"]: r for r in plays}
        self.assertEqual(by_message[4]["cleaning"], drives.STALE_AFTER_CHANGE)
        self.assertEqual(by_message[5]["cleaning"], drives.KICKOFF)
        # And a dropped row belongs to no drive, so the join stays honest.
        self.assertEqual(by_message[4]["drive_number"], "")

    def test_exactly_one_play_per_drive_is_the_anchor(self):
        plays, snapshots = self.rows()
        anchors = [r for r in plays if r["is_anchor"]]
        self.assertEqual(len(anchors), len(snapshots))
        self.assertEqual({r["event_message_count"] for r in anchors},
                         {s.event_message_count for s in snapshots})

    def test_the_score_travels_on_every_row(self):
        # The whole point of one table: no join to find out what the
        # scoreboard said at the row you are looking at.
        plays, _ = self.rows()
        by_message = {r["event_message_count"]: r for r in plays}
        self.assertEqual((by_message[2]["score_p1"], by_message[2]["score_p2"]),
                         (0, 0))
        self.assertEqual((by_message[6]["score_p1"], by_message[6]["score_p2"]),
                         (6, 0))
        self.assertEqual(by_message[6]["score_diff"], 6)
        # And the change itself is marked where it landed.
        self.assertEqual(by_message[3]["p1_change"], 6)
        self.assertEqual(by_message[2]["p1_change"], "")

    def test_a_score_with_no_play_row_still_gets_a_row(self):
        from .. import dump
        plays = [("AF1", 1, 1, self.HOME, 1, 10, 25, None),
                 ("AF1", 9, 1, self.AWAY, 1, 10, 25, None)]
        rows, _ = dump._rows_for_match("AF1", plays, [("AF1", 5, 1, 7, None, 7, 0)])
        by_message = {r["event_message_count"]: r for r in rows}
        self.assertIn(5, by_message)
        self.assertEqual(by_message[5]["cleaning"], "score")
        self.assertEqual(by_message[5]["p1_change"], 7)
        # It is not a play, so it belongs to no drive and anchors nothing.
        self.assertEqual(by_message[5]["drive_number"], "")
        self.assertEqual(by_message[5]["is_anchor"], 0)

    def test_the_rows_are_in_message_order(self):
        plays, _ = self.rows()
        messages = [r["event_message_count"] for r in plays]
        self.assertEqual(messages, sorted(messages))

    def test_both_streams_prices_sit_on_the_row(self):
        plays, _ = self.rows(self.indexes())
        by_message = {r["event_message_count"]: r for r in plays}
        self.assertEqual(by_message[1]["ml_home_prod"], 55.0)
        self.assertEqual(by_message[1]["ml_home_cand"], 56.0)
        # Lined markets carry each side's line, which is what the line rule
        # is decided on -- here the two streams disagree.
        self.assertEqual(by_message[1]["sp_home_line_prod"], -3.5)
        self.assertEqual(by_message[1]["sp_home_line_cand"], -6.5)

    def test_the_live_cell_names_which_side_was_dead(self):
        plays, _ = self.rows(self.indexes())
        by_message = {r["event_message_count"]: r for r in plays}
        self.assertEqual(by_message[1]["ml_home_live"], "live")
        # Prod settled the spread at 6, candidate never quoted it there.
        self.assertEqual(by_message[6]["sp_home_live"], "missing")
        # And a selection neither stream quoted stays blank rather than
        # claiming anything about it.
        self.assertEqual(by_message[1]["tot_over_live"], "")
        self.assertEqual(by_message[1]["tot_over_prod"], "")

    def test_the_market_columns_cover_all_six_selections(self):
        from .. import dump
        for name, _, lined in dump.MARKET_COLUMNS:
            self.assertIn(f"{name}_prod", dump.PLAY_FIELDS)
            self.assertIn(f"{name}_cand", dump.PLAY_FIELDS)
            self.assertIn(f"{name}_live", dump.PLAY_FIELDS)
            self.assertEqual(f"{name}_line_prod" in dump.PLAY_FIELDS, lined)

    def test_the_row_keys_are_exactly_the_csv_columns(self):
        from .. import dump
        plays, _ = self.rows(self.indexes())
        for row in plays:
            self.assertEqual(sorted(row), sorted(dump.PLAY_FIELDS))


class TestDumpDriveRows(unittest.TestCase):
    """The per-drive summary the console prints, built from the CSV rows."""

    HOME, AWAY = "Home Team", "Away Team"

    def rows(self):
        from .. import dump
        plays = [("AF1", 1, 1, self.HOME, 1, 10, 25, None),
                 ("AF1", 2, 1, self.HOME, 2, 4, 31, None),
                 ("AF1", 3, 1, self.HOME, 3, 1, 34, None),
                 ("AF1", 4, 1, self.AWAY, 3, 1, 34, None),
                 ("AF1", 5, 1, self.AWAY, 4, 99, 35, None),
                 ("AF1", 6, 1, self.AWAY, 1, 10, 22, None),
                 ("AF1", 7, 1, self.AWAY, 2, 6, 26, None)]
        scores = [("AF1", 3, 1, 6, None, 6, 0)]
        play_out, snapshots = dump._rows_for_match("AF1", plays, scores)
        return play_out, dump._drive_rows("AF1", play_out, snapshots)

    def test_one_row_per_drive(self):
        play_out, drive_out = self.rows()
        self.assertEqual(len(drive_out),
                         len({r["drive_number"] for r in play_out
                              if r["drive_number"] != ""}))

    def test_a_drive_counts_the_noise_inside_it(self):
        _, drive_out = self.rows()
        second = drive_out[1]
        self.assertEqual(second["offensive_team"], self.AWAY)
        self.assertEqual(second["first_message"], 6)
        self.assertEqual(second["n_plays"], 2)
        # Messages 4 and 5 were dropped before the drive opened, so they
        # are noise between drives rather than inside one.
        self.assertEqual(second["n_dropped_inside"], 0)

    def test_the_drive_carries_the_score_at_its_anchor(self):
        _, drive_out = self.rows()
        self.assertEqual((drive_out[0]["score_p1"], drive_out[0]["score_p2"]),
                         (0, 0))
        self.assertEqual((drive_out[1]["score_p1"], drive_out[1]["score_p2"]),
                         (6, 0))
        self.assertEqual(drive_out[1]["score_diff"], 6)

    def test_the_drive_rows_join_back_to_the_play_rows(self):
        play_out, drive_out = self.rows()
        keys = {(r["match_code"], r["event_message_count"]) for r in play_out}
        for row in drive_out:
            self.assertIn((row["match_code"], row["anchor_message"]), keys)


class TestPointsEndADrive(unittest.TestCase):
    """AF063170926 drive 10: one drive, 13 plays, a 15-point swing inside.

    The team label stayed on Away from message 310 to 407, across two Away
    scores. Team change alone therefore saw one possession where there
    were three, and the snapshot it anchored carried 21-7 while the drive
    after it opened at 21-22. Points cannot be argued with: nobody is on
    the same drive either side of one.
    """

    AWAY = "Away Team"

    def snapshots(self):
        plays = [drives.PlayRow(m, 4, self.AWAY, d, dist, f, None)
                 for m, d, dist, f in [
                     (310, 1, 10, 24), (315, 2, 6, 28), (320, 3, 2, 32),
                     (350, 2, 5, 30), (360, 1, 10, 45), (370, 3, 8, 47),
                     (407, 2, 3, 32)]]
        scores = [drives.ScoreRow(335, 4, 0, 7, 21, 14),
                  drives.ScoreRow(404, 4, 0, 8, 21, 22)]
        return drives.build_snapshots("AF1", plays, scores)

    def test_each_score_ends_the_drive_it_landed_in(self):
        snaps = self.snapshots()
        self.assertEqual(len(snaps), 3)
        self.assertEqual([s.event_message_count for s in snaps], [310, 350, 407])

    def test_no_drive_spans_a_score(self):
        # The bug read as a snapshot bucketed on a score that had already
        # changed by the time the drive ended, so this is the check that
        # matters: the score at the anchor is the score for the whole run.
        snaps = self.snapshots()
        self.assertEqual([(s.score_p1, s.score_p2) for s in snaps],
                         [(0, 0), (21, 14), (21, 22)])

    def test_a_score_on_a_kept_play_breaks_after_it_not_before(self):
        # The scoring play is the end of the old drive. Breaking at it
        # would strand the play that scored in the next possession.
        plays = [drives.PlayRow(m, 1, "Home Team", d, dist, f, None)
                 for m, d, dist, f in [
                     (1, 1, 10, 25), (2, 2, 4, 31), (3, 3, 1, 34),
                     (8, 2, 7, 28)]]
        scores = [drives.ScoreRow(3, 1, 6, None, 6, 0)]
        snaps = drives.build_snapshots("AF1", plays, scores)
        self.assertEqual([s.event_message_count for s in snaps], [1, 8])
        self.assertEqual(snaps[0].n_plays, 3)
        self.assertEqual((snaps[0].score_p1, snaps[0].score_p2), (0, 0))

    def test_a_scoreless_run_is_still_one_drive(self):
        plays = [drives.PlayRow(m, 1, "Home Team", d, dist, f, None)
                 for m, d, dist, f in [
                     (1, 1, 10, 25), (2, 2, 4, 31), (3, 3, 1, 34)]]
        snaps = drives.build_snapshots("AF1", plays, [])
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].n_plays, 3)


class TestALockedFileDoesNotLoseTheRun(unittest.TestCase):
    """Excel holds a CSV open; Windows refuses the write.

    The queries, the cleaning and the pairing are already paid for by
    then, so the other file still gets written and the command says which
    one it could not.
    """

    def test_a_refused_write_returns_none_rather_than_raising(self):
        from .. import dump
        import tempfile
        with tempfile.TemporaryDirectory() as out:
            path = os.path.join(out, "locked.csv")
            real_open = builtins.open

            def refuse(target, *args, **kwargs):
                if target == path:
                    raise PermissionError(13, "Permission denied")
                return real_open(target, *args, **kwargs)

            with mock.patch.object(builtins, "open", refuse):
                with contextlib.redirect_stdout(io.StringIO()) as out_text:
                    self.assertIsNone(dump._write(path, ["a"], [{"a": 1}]))
        self.assertIn("could NOT write", out_text.getvalue())
        self.assertIn("Excel", out_text.getvalue())

    def test_the_other_file_is_still_written(self):
        from .. import dump
        import tempfile
        with tempfile.TemporaryDirectory() as out:
            good = os.path.join(out, "good.csv")
            bad = os.path.join(out, "bad.csv")
            real_open = builtins.open

            def refuse(target, *args, **kwargs):
                if target == bad:
                    raise PermissionError(13, "Permission denied")
                return real_open(target, *args, **kwargs)

            with mock.patch.object(builtins, "open", refuse):
                with contextlib.redirect_stdout(io.StringIO()):
                    written = [p for p in (
                        dump._write(good, ["a"], [{"a": 1}]),
                        dump._write(bad, ["a"], [{"a": 1}])) if p]
            self.assertEqual(written, [good])
            self.assertTrue(os.path.exists(good))
        # And the caller can tell, because it knows how many it wanted.
        self.assertEqual(dump.FILES, 2)


class TestKickoffRows(unittest.TestCase):
    """Kickoffs as the feed really sends them.

    From AF063170926: the kick spot arrives as a full 1st-and-10, which is
    why down and distance alone cannot tell it from a drive. What gives it
    away is that the very next row is the same team's 1st-and-10 again,
    without the ten yards that would earn it -- or that the ball went
    backwards to reach it.
    """

    HOME, AWAY = "Home Team", "Away Team"

    def reasons(self, plays):
        return drives.classify_plays(plays)

    def test_the_kick_spot_is_a_full_first_and_ten(self):
        # 1&10 @35 then 1&10 @26: nine yards backwards, so the first row
        # is the spot the kick was taken from, not a drive.
        plays = [PlayRow(6, 1, self.HOME, 1, 10, 35),
                 PlayRow(10, 1, self.HOME, 1, 10, 26),
                 PlayRow(16, 1, self.HOME, 2, 11, 26)]
        reasons = self.reasons(plays)
        self.assertEqual(reasons[6], drives.KICKOFF)
        self.assertEqual(reasons[10], drives.DRIVE_START)
        snaps = drives.build_snapshots("AF1", plays, [])
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].event_message_count, 10)
        self.assertEqual(snaps[0].field_position, 26)

    def test_a_first_down_conversion_is_not_mistaken_for_a_kick(self):
        # Two 1st-and-10s in a row are both real when the ball advanced
        # the ten yards that earn the second.
        plays = [PlayRow(20, 1, self.HOME, 1, 10, 37),
                 PlayRow(24, 1, self.HOME, 1, 10, 50),
                 PlayRow(28, 1, self.HOME, 1, 10, 85)]
        reasons = self.reasons(plays)
        self.assertEqual(reasons[24], drives.KEPT)
        self.assertEqual(reasons[28], drives.KEPT)

    def test_a_first_and_ten_reached_backwards_is_a_kick(self):
        # After the touchdown the ball is on the 90; the next 1st-and-10
        # is on the 35. No first down travels backwards.
        plays = [PlayRow(36, 1, self.HOME, 3, 5, 90),
                 PlayRow(47, 1, self.HOME, 1, 10, 35),
                 PlayRow(52, 1, self.AWAY, 1, 10, 35),
                 PlayRow(54, 1, self.AWAY, 1, 10, 30)]
        reasons = self.reasons(plays)
        self.assertEqual(reasons[47], drives.KICKOFF)
        self.assertEqual(reasons[52], drives.STALE_AFTER_CHANGE)
        self.assertEqual(reasons[54], drives.DRIVE_START)

    def test_the_extra_point_keeps_the_touchdowns_down_and_distance(self):
        # 3&5 @90 then 3&5 @85: a scrimmage play always changes the down
        # or the distance, so this is the extra point, taken from the 85.
        plays = [PlayRow(36, 1, self.HOME, 3, 5, 90),
                 PlayRow(42, 1, self.HOME, 3, 5, 85)]
        self.assertEqual(self.reasons(plays)[42], drives.SPECIAL_TEAMS)

    def test_the_stale_row_rides_the_kick_spot_not_the_last_real_play(self):
        # The Away label lands on the kick spot's 1&10 @35, and the kick
        # spot was itself dropped. Comparing against the last SURVIVING
        # row would miss it and call it a drive.
        plays = [PlayRow(36, 1, self.HOME, 3, 5, 90),
                 PlayRow(47, 1, self.HOME, 1, 10, 35),
                 PlayRow(52, 1, self.AWAY, 1, 10, 35),
                 PlayRow(54, 1, self.AWAY, 1, 10, 30)]
        self.assertEqual(self.reasons(plays)[52], drives.STALE_AFTER_CHANGE)

    def test_a_row_with_no_readable_down_is_not_scrimmage(self):
        plays = [PlayRow(1, 1, self.HOME, 1, 10, 25),
                 PlayRow(2, 1, self.HOME, None, None, 35),
                 PlayRow(3, 1, self.AWAY, 1, 10, 22)]
        self.assertEqual(self.reasons(plays)[2], drives.KICKOFF)

    def test_a_punt_drops_only_its_stale_row(self):
        plays = [PlayRow(1, 1, self.HOME, 1, 10, 25),
                 PlayRow(2, 1, self.HOME, 2, 8, 27),
                 PlayRow(3, 1, self.AWAY, 2, 8, 27),    # stale duplicate
                 PlayRow(4, 1, self.AWAY, 1, 10, 40),
                 PlayRow(5, 1, self.AWAY, 2, 3, 47)]
        reasons = self.reasons(plays)
        self.assertEqual(reasons[3], drives.STALE_AFTER_CHANGE)
        self.assertEqual(reasons[4], drives.DRIVE_START)
        snaps = drives.build_snapshots("AF1", plays, [])
        self.assertEqual([s.event_message_count for s in snaps], [1, 4])

    def test_field_position_is_never_the_test(self):
        # A return finishing on the kick spot still starts a drive: the
        # row before it is what marks the kick, not the yard line.
        plays = [PlayRow(1, 1, self.HOME, 1, 10, 20),
                 PlayRow(2, 1, self.AWAY, 1, 10, 35),
                 PlayRow(3, 1, self.AWAY, 2, 6, 39)]
        reasons = self.reasons(plays)
        self.assertEqual(reasons[2], drives.DRIVE_START)
        snaps = drives.build_snapshots("AF1", plays, [])
        self.assertEqual(snaps[1].field_position, 35)


class TestFieldPositionDoesNotCarryAcrossAKick(unittest.TestCase):
    """AF063170926 messages 352-384: three real snaps called kickoffs.

    After the score at 359 the last surviving row was message 352, Away
    3rd-and-5 on the 91. The kick spot arrived three times (360, 362, 364,
    all on the 35) and the drive it produced ran 45 -> 63 -> 80. Every one
    of those was compared against the 91 and read as going backward, so
    the whole drive was dropped as kick rows and the snapshot fell on
    2nd-and-4 at the 87, six plays late.

    A kick resets field position. Nothing either side of it can be read
    against the other.
    """

    AWAY = "Away Team"

    def feed(self):
        plays = [drives.PlayRow(m, 4, self.AWAY, d, dist, f, None)
                 for m, d, dist, f in [
                     (352, 3, 5, 91),    # the drive that scored
                     (360, 1, 10, 35),   # the kick spot
                     (362, 3, 5, 35),    # the scoring play's d&d, restated
                     (364, 1, 10, 35),   # the kick spot again
                     (366, 1, 10, 45),   # the drive
                     (372, 1, 10, 63),
                     (377, 1, 10, 80),
                     (384, 2, 4, 87)]]
        scores = [drives.ScoreRow(359, 4, 0, 8, 21, 15)]
        return plays, scores

    def test_the_kick_spot_still_goes(self):
        reasons = drives.classify_plays(*self.feed())
        self.assertEqual(reasons[360], drives.KICKOFF)
        self.assertEqual(reasons[364], drives.KICKOFF)

    def test_the_drive_the_kick_produced_survives(self):
        reasons = drives.classify_plays(*self.feed())
        self.assertEqual(reasons[366], drives.DRIVE_START)
        self.assertEqual(reasons[372], drives.KEPT)
        self.assertEqual(reasons[377], drives.KEPT)
        self.assertEqual(reasons[384], drives.KEPT)

    def test_a_repeat_of_the_kick_spot_is_told_apart_by_its_yard_line(self):
        # 364 and 366 are both fresh 1st-and-10s after a dropped kick. The
        # only thing separating them is that 364 has not left the spot the
        # kick was taken from.
        reasons = drives.classify_plays(*self.feed())
        self.assertEqual((reasons[364], reasons[366]),
                         (drives.KICKOFF, drives.DRIVE_START))

    def test_the_snapshot_lands_on_the_drives_opening_first_down(self):
        plays, scores = self.feed()
        snaps = drives.build_snapshots("AF063170926", plays, scores)
        drive = [s for s in snaps if s.event_message_count >= 360][0]
        self.assertEqual(drive.event_message_count, 366)
        self.assertEqual(drive.anchor, drives.FIRST_DOWN)
        self.assertEqual(drive.field_position, 45)
        self.assertEqual(drive.n_plays, 4)

    def test_going_backward_is_still_evidence_inside_one_possession(self):
        # The rule it relaxes has to keep working where it applies: no
        # kick in between, so the ball cannot have gone back ten yards
        # and earned a fresh 1st-and-10.
        plays = [drives.PlayRow(m, 1, self.AWAY, d, dist, f, None)
                 for m, d, dist, f in [
                     (1, 1, 10, 40), (2, 2, 6, 44), (3, 1, 10, 30)]]
        self.assertEqual(drives.classify_plays(plays, [])[3], drives.KICKOFF)


class TestDuplicateRows(unittest.TestCase):
    """The feed republishes a play's state until it changes."""

    HOME = "Home Team"

    def test_consecutive_identical_rows_collapse(self):
        plays = [PlayRow(6, 1, self.HOME, 1, 10, 26),
                 PlayRow(9, 1, self.HOME, 1, 10, 26),
                 PlayRow(10, 1, self.HOME, 2, 4, 30),
                 PlayRow(15, 1, self.HOME, 2, 4, 30)]
        reasons = drives.classify_plays(plays)
        self.assertEqual(reasons[9], drives.DUPLICATE)
        self.assertEqual(reasons[15], drives.DUPLICATE)
        self.assertEqual(reasons[6], drives.DRIVE_START)
        self.assertEqual(reasons[10], drives.KEPT)

    def test_a_repeat_that_is_not_adjacent_is_not_a_duplicate(self):
        # Back to 1st-and-10 on the same yard line later in the drive is a
        # real play, not a republish.
        plays = [PlayRow(1, 1, self.HOME, 1, 10, 26),
                 PlayRow(2, 1, self.HOME, 2, 4, 30),
                 PlayRow(3, 1, self.HOME, 1, 10, 26)]
        reasons = drives.classify_plays(plays)
        self.assertNotEqual(reasons[3], drives.DUPLICATE)

    def test_dedupe_halves_the_real_sequence(self):
        # The 25 rows from AF063170926 collapse to 13.
        raw = [(6, 1, 10, 35), (9, 1, 10, 35), (10, 1, 10, 26), (15, 1, 10, 26),
               (16, 2, 11, 26), (19, 2, 11, 26), (20, 1, 10, 37), (23, 1, 10, 37)]
        plays = [PlayRow(m, 1, self.HOME, d, di, f) for m, d, di, f in raw]
        self.assertEqual([p.event_message_count for p in drives.dedupe(plays)],
                         [6, 10, 16, 20])


class TestHalfTimeKickoff(unittest.TestCase):
    """The second-half kick, from AF063170926 messages 176-214.

    Worse than the others because a row sits between the kick spot and the
    drive it produced: the down jumps 1 to 3 with neither the distance nor
    the ball moving. While that row is in the way the kick spot cannot see
    the drive, so the kick is read as a drive and the drive as the kick.
    """

    HOME, AWAY = "Home Team", "Away Team"

    def feed(self):
        raw = [(176, 2, self.HOME, 1, 10, 37), (178, 2, self.HOME, 1, 10, 37),
               (180, 2, self.HOME, 2, 10, 37), (184, 2, self.HOME, 2, 10, 37),
               (185, 2, self.HOME, 3, 10, 37), (193, 3, self.AWAY, 1, 10, 35),
               (196, 3, self.AWAY, 3, 10, 35), (197, 3, self.AWAY, 1, 10, 25),
               (202, 3, self.AWAY, 1, 10, 25), (203, 3, self.HOME, 1, 10, 25),
               (205, 3, self.HOME, 1, 10, 68), (209, 3, self.HOME, 1, 10, 68),
               (210, 3, self.HOME, 2, 8, 71), (213, 3, self.HOME, 2, 8, 71),
               (214, 3, self.HOME, 1, 1, 99)]
        return [PlayRow(m, p, t, d, di, f) for m, p, t, d, di, f in raw]

    def test_the_impossible_down_is_caught_first(self):
        # 1st-and-10 to 3rd-and-10 with the ball on the same yard line.
        reasons = drives.classify_plays(self.feed())
        self.assertEqual(reasons[196], drives.IMPOSSIBLE_DOWN)

    def test_the_kick_spot_is_not_the_drive(self):
        reasons = drives.classify_plays(self.feed())
        self.assertEqual(reasons[193], drives.KICKOFF)
        self.assertEqual(reasons[197], drives.DRIVE_START)

    def test_the_drive_anchors_on_the_real_first_down(self):
        snaps = drives.build_snapshots("AF063170926", self.feed(), [])
        self.assertEqual([s.event_message_count for s in snaps], [176, 197, 205])
        away = snaps[1]
        self.assertEqual(away.offensive_team, self.AWAY)
        self.assertEqual(away.field_position, 25)
        self.assertEqual((away.down_number, away.distance), (1, 10))
        self.assertTrue(all(s.anchor == drives.FIRST_DOWN for s in snaps))

    def test_an_ordinary_incomplete_pass_is_not_impossible(self):
        # 1st-and-10 to 2nd-and-10 on the same yard line happens every
        # game, and the rule must not touch it.
        reasons = drives.classify_plays(self.feed())
        self.assertEqual(reasons[180], drives.KEPT)
        self.assertEqual(reasons[185], drives.KEPT)

    def test_the_kick_spot_is_caught_across_a_team_change(self):
        # At half time the label changes onto the kick spot, so testing it
        # only where the team stayed the same missed it entirely.
        plays = [PlayRow(1, 2, self.HOME, 3, 4, 40),
                 PlayRow(2, 3, self.AWAY, 1, 10, 35),   # the kick
                 PlayRow(3, 3, self.AWAY, 1, 10, 25)]   # the drive
        reasons = drives.classify_plays(plays)
        self.assertEqual(reasons[2], drives.KICKOFF)
        self.assertEqual(reasons[3], drives.DRIVE_START)

    def test_a_gap_in_the_feed_is_not_an_impossible_down(self):
        # A missing row shows as a down jump too, but the distance moves
        # with it. Only a jump with nothing else changing is impossible.
        plays = [PlayRow(1, 1, self.HOME, 1, 10, 30),
                 PlayRow(2, 1, self.HOME, 3, 4, 36)]
        self.assertEqual(drives.classify_plays(plays)[2], drives.KEPT)


class TestAnchorFieldCheck(unittest.TestCase):
    """A spike on one yard line means kick spots, not football."""

    @staticmethod
    def rows(spike, spread):
        out = [{"field_position": 35, "n_plays": 3, "match_code": "AF1",
                "drive_number": i, "offensive_team": "Home Team",
                "anchor_kind": "first_down", "n_dropped_inside": 0}
               for i in range(spike)]
        out += [{"field_position": 20 + i % 40, "n_plays": 4,
                 "match_code": "AF1", "drive_number": 1000 + i,
                 "offensive_team": "Away Team", "anchor_kind": "first_down",
                 "n_dropped_inside": 0} for i in range(spread)]
        return out

    def capture(self, rows):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            report._print_anchor_field_check(rows)
        return buffer.getvalue()

    def test_it_calls_out_a_spike(self):
        # The pre-fix shape: 58% of anchors on one yard line.
        out = self.capture(self.rows(spike=58, spread=42))
        self.assertIn("of drives start on the 35", out)
        self.assertIn("kick spot", out)

    def test_a_healthy_spread_says_nothing(self):
        out = self.capture(self.rows(spike=2, spread=98))
        self.assertNotIn("kick spot", out)
        # But it still shows the distribution, which is the point.
        self.assertIn("FIELD", out)

    def test_it_survives_rows_with_no_field_position(self):
        out = self.capture([{"field_position": "", "n_plays": 1,
                             "match_code": "AF1", "drive_number": 1,
                             "offensive_team": "Home Team",
                             "anchor_kind": "first_down",
                             "n_dropped_inside": 0}])
        self.assertEqual(out, "")


class TestPairDump(unittest.TestCase):
    """directional_pairs.csv, widened with what the other files know."""

    def rows(self):
        from .. import dump
        pairs = [line_pair(0.50, 0.62, 44.5, 44.5, 24, 21, match="AF1",
                           market_id=54)]
        pairs = [dataclasses.replace(
            pairs[0], drive_number=2, period_number=3, score_p1=14,
            score_p2=7, field_position=26, down_number=1, distance=10,
            message_count=197, prod_state="open/true",
            candidate_state="UNDER SETTLEMENT/false", candidate_live=False,
            anchor=drives.FIRST_DOWN)]
        play_out = [{"match_code": "AF1", "event_message_count": 197,
                     "cleaning": drives.DRIVE_START}]
        drive_out = [{"match_code": "AF1", "drive_number": 2, "n_plays": 6,
                      "n_dropped_inside": 3}]
        rows_at = {("AF1", 197, "prod", 54): 2,
                   ("AF1", 197, "candidate", 54): 1}
        return dump._pair_dump_rows(pairs, play_out, drive_out, rows_at)

    def test_it_keeps_every_directional_pairs_column(self):
        from .. import dump
        row = self.rows()[0]
        for field in report.PAIR_FIELDS:
            self.assertIn(field, row, field)
        # And in the same order, so the file reads the same way.
        self.assertEqual(dump.PAIR_DUMP_FIELDS[:len(report.PAIR_FIELDS)],
                         report.PAIR_FIELDS)

    def test_the_shared_columns_come_from_one_place(self):
        # The dump widens report.pair_row rather than rebuilding it, so
        # the two files cannot describe the same pair differently.
        from .. import dump
        pair = line_pair(0.5, 0.6, 44.5, 44.5, 24, 21)
        widened = dump._pair_dump_rows([pair], [], [])[0]
        for key, value in report.pair_row(pair).items():
            self.assertEqual(widened[key], value, key)

    def test_it_carries_the_drive_detection_context(self):
        row = self.rows()[0]
        self.assertEqual(row["anchor_kind"], drives.FIRST_DOWN)
        self.assertEqual(row["anchor_cleaning"], drives.DRIVE_START)
        self.assertEqual(row["drive_n_plays"], 6)
        self.assertEqual(row["drive_dropped_inside"], 3)

    def test_it_carries_the_market_state(self):
        row = self.rows()[0]
        self.assertEqual(row["prod_state"], "open/true")
        self.assertEqual(row["candidate_state"], "UNDER SETTLEMENT/false")
        self.assertEqual((row["prod_live"], row["candidate_live"]), (1, 0))
        self.assertEqual(row["live"], "cand")

    def test_it_shows_how_many_rows_the_message_offered(self):
        # The case that cost the spread and total their pairs, on the row
        # it affected rather than in a separate file.
        row = self.rows()[0]
        self.assertEqual(row["prod_rows_at_message"], 2)
        self.assertEqual(row["candidate_rows_at_message"], 1)

    def test_it_carries_the_buckets_the_pair_lands_in(self):
        row = self.rows()[0]
        self.assertEqual(row["score_bucket"], "Home 1 score")
        self.assertEqual(row["time_bucket"], "Q3")
        self.assertEqual(row["possession_bucket"], "Home")

    def test_it_names_the_market_and_the_decision(self):
        row = self.rows()[0]
        self.assertEqual((row["market"], row["selection"]), ("total", "Over"))
        self.assertEqual(row["basis"], "prob")
        # Not live, so nothing decided it.
        self.assertIsNone(row["decisive_winner"])


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
            play(3, "Away Team", 2, 5, 30),   # stale: Home's state, Away's label
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
            play(1, "Home Team", 1, 10, 25),
            play(3, "Away Team", 1, 10, 40),
            play(4, "Away Team", 2, 6, 44),
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


class TestTransitionClassification(unittest.TestCase):
    """What the play feed says happened between two rows."""

    HOME, AWAY = "Home Team", "Away Team"

    def feed(self):
        plays = [PlayRow(m, 1, self.HOME, d, dist, f, None)
                 for m, d, dist, f in [
                     (10, 1, 10, 25),   # drive opens
                     (12, 2, 3, 32),    # +7, short of the line to gain
                     (14, 1, 10, 35),   # +3 and a fresh set: converted
                     (16, 2, 10, 35),   # 0
                     (18, 3, 14, 31),   # -4
                     (20, 4, 14, 31)]]  # 3rd down came and went
        plays.append(PlayRow(30, 1, self.AWAY, 1, 10, 25, None))
        return plays, []

    def outcomes(self):
        return {(t.from_message, t.to_message): t
                for t in indrive.transitions_for_match("AF1", *self.feed())}

    def test_each_class_comes_out_of_the_play_feed_alone(self):
        got = self.outcomes()
        self.assertEqual(got[(10, 12)].outcome, indrive.BIG_GAIN)
        self.assertEqual(got[(12, 14)].outcome, indrive.FIRST_DOWN)
        self.assertEqual(got[(14, 16)].outcome, indrive.NO_GAIN)
        self.assertEqual(got[(16, 18)].outcome, indrive.LOSS)
        self.assertEqual(got[(18, 20)].outcome, indrive.FAILED_CONVERSION)

    def test_the_drive_ending_transition_is_built_and_marked(self):
        got = self.outcomes()
        ending = got[(20, 30)]
        self.assertEqual(ending.outcome, indrive.POSSESSION_LOST)
        self.assertFalse(ending.in_drive)
        self.assertTrue(all(t.in_drive for k, t in got.items() if k != (20, 30)))

    def test_yards_are_only_claimed_inside_a_drive(self):
        # Across a kickoff the two field positions are in different frames,
        # so the difference between them is not a gain.
        got = self.outcomes()
        self.assertEqual(got[(10, 12)].yards, 7)
        self.assertIsNone(got[(20, 30)].yards)

    def test_a_short_gain_carries_no_direction(self):
        plays = [PlayRow(m, 1, self.HOME, d, dist, f, None)
                 for m, d, dist, f in [(1, 1, 10, 25), (2, 2, 7, 28)]]
        t = indrive.transitions_for_match("AF1", plays, [])[0]
        self.assertEqual(t.outcome, indrive.SHORT_GAIN)
        self.assertEqual(t.sign, 0)
        self.assertFalse(t.scorable)

    def test_points_settle_it_before_any_yardage(self):
        # Home scores; the next surviving row is Away receiving. The
        # transition between them is the touchdown, not a possession loss.
        plays = [PlayRow(1, 1, self.HOME, 3, 2, 95, None),
                 PlayRow(9, 1, self.AWAY, 1, 10, 25, None)]
        scores = [ScoreRow(5, 1, 6, None, 6, 0)]
        t = indrive.transitions_for_match("AF1", plays, scores)[0]
        self.assertEqual(t.outcome, indrive.TOUCHDOWN)
        self.assertEqual(t.points, 6)
        self.assertEqual(t.sign, +1)

    def test_a_field_goal_is_its_own_class(self):
        plays = [PlayRow(1, 1, self.HOME, 4, 5, 80, None),
                 PlayRow(9, 1, self.AWAY, 1, 10, 25, None)]
        scores = [ScoreRow(5, 1, 3, None, 3, 0)]
        t = indrive.transitions_for_match("AF1", plays, scores)[0]
        self.assertEqual(t.outcome, indrive.FIELD_GOAL)
        self.assertEqual(t.sign, +1)

    def test_a_conversion_is_told_apart_from_a_field_goal(self):
        # A PAT is their own point, but the touchdown it follows was
        # priced one transition ago, so there is far less news in it.
        plays = [PlayRow(1, 1, self.HOME, 1, 10, 95, None),
                 PlayRow(9, 1, self.AWAY, 1, 10, 25, None)]
        scores = [ScoreRow(5, 1, 1, None, 1, 0)]
        t = indrive.transitions_for_match("AF1", plays, scores)[0]
        self.assertEqual(t.outcome, indrive.EXTRA_POINT)
        self.assertEqual(t.sign, +1)

    def test_the_defence_scoring_is_bad_for_the_offence(self):
        # Points are signed to the side with the ball, so a pick six or a
        # safety comes back negative and cannot read as a good play.
        plays = [PlayRow(1, 1, self.HOME, 2, 8, 20, None),
                 PlayRow(9, 1, self.AWAY, 1, 10, 25, None)]
        scores = [ScoreRow(5, 1, None, 6, 0, 6)]
        t = indrive.transitions_for_match("AF1", plays, scores)[0]
        self.assertEqual(t.outcome, indrive.POINTS_AGAINST)
        self.assertEqual(t.sign, -1)
        self.assertEqual(t.points, -6)

    def test_every_class_has_a_sign_and_an_order(self):
        for outcome in indrive.OUTCOME_SIGN:
            self.assertIn(outcome, indrive.OUTCOME_ORDER, outcome)
        self.assertEqual(len(indrive.OUTCOME_ORDER),
                         len(indrive.OUTCOME_SIGN))


class TestExpectedDirection(unittest.TestCase):
    """Which way a selection should move when the offence does well."""

    HOME, AWAY = "Home Team", "Away Team"

    def test_the_offence_own_side_follows_the_play(self):
        # Home has the ball and does something good.
        self.assertEqual(indrive.expected_sign(50, self.HOME, +1), +1)
        self.assertEqual(indrive.expected_sign(52, self.HOME, +1), +1)
        # The opponent's side moves the other way.
        self.assertEqual(indrive.expected_sign(51, self.HOME, +1), -1)
        self.assertEqual(indrive.expected_sign(53, self.HOME, +1), -1)

    def test_it_flips_with_possession(self):
        for market_id in (50, 51, 52, 53):
            self.assertEqual(indrive.expected_sign(market_id, self.AWAY, +1),
                             -indrive.expected_sign(market_id, self.HOME, +1),
                             market_id)

    def test_totals_are_possession_blind(self):
        # Points are points whoever scores them, so Over follows a good
        # offensive play from either side.
        for team in (self.HOME, self.AWAY):
            self.assertEqual(indrive.expected_sign(54, team, +1), +1, team)
            self.assertEqual(indrive.expected_sign(55, team, +1), -1, team)

    def test_a_bad_play_reverses_every_expectation(self):
        for market_id in markets.MARKET_IDS:
            good = indrive.expected_sign(market_id, self.HOME, +1)
            bad = indrive.expected_sign(market_id, self.HOME, -1)
            self.assertEqual(good, -bad, market_id)

    def test_nothing_is_expected_without_a_direction_or_a_side(self):
        self.assertEqual(indrive.expected_sign(50, self.HOME, 0), 0)
        self.assertEqual(indrive.expected_sign(50, None, +1), 0)
        self.assertEqual(indrive.expected_sign(50, "Some Other Team", +1), 0)


class TestMoves(unittest.TestCase):
    """Reading the two prices, and refusing to where it would be wrong."""

    HOME = "Home Team"
    T = dt.datetime(2026, 9, 18, 12, 0)

    def quote(self, message, market_id, probability, description,
              status="open", active="true"):
        return ("AF1", market_id, self.T, probability, 2.0, description,
                message, status, active)

    def transition(self, outcome=None):
        plays = [PlayRow(m, 1, self.HOME, d, dist, f, None)
                 for m, d, dist, f in [(1, 2, 3, 32), (2, 1, 10, 40)]]
        return indrive.transitions_for_match("AF1", plays, [])[0]

    def index(self, rows):
        return {directional.PROD: directional.index_by_message(rows),
                directional.CANDIDATE: directional.index_by_message([])}

    def test_a_move_is_scored_on_its_sign(self):
        t = self.transition()
        self.assertEqual(t.outcome, indrive.FIRST_DOWN)
        rows = [self.quote(1, 50, 40.0, "PLAYER 1 to win"),
                self.quote(2, 50, 46.0, "PLAYER 1 to win")]
        move = indrive.moves_for_transitions([t], self.index(rows))[0]
        self.assertEqual(move.expected, +1)
        self.assertAlmostEqual(move.delta, 0.06)
        self.assertTrue(move.hit)
        self.assertFalse(move.flat)

    def test_the_wrong_way_is_a_miss(self):
        rows = [self.quote(1, 50, 46.0, "PLAYER 1 to win"),
                self.quote(2, 50, 40.0, "PLAYER 1 to win")]
        move = indrive.moves_for_transitions([self.transition()],
                                             self.index(rows))[0]
        self.assertFalse(move.hit)
        self.assertLess(move.signed, 0)

    def test_a_price_that_did_not_move_is_neither(self):
        rows = [self.quote(1, 50, 44.0, "PLAYER 1 to win"),
                self.quote(2, 50, 44.0, "PLAYER 1 to win")]
        move = indrive.moves_for_transitions([self.transition()],
                                             self.index(rows))[0]
        self.assertTrue(move.flat)
        self.assertIsNone(move.hit)

    def test_probabilities_are_read_on_the_suite_scale(self):
        # The feed publishes 0-100. A move has to read like a Brier delta,
        # not a hundred times larger than one.
        rows = [self.quote(1, 50, 40.0, "PLAYER 1 to win"),
                self.quote(2, 50, 46.0, "PLAYER 1 to win")]
        move = indrive.moves_for_transitions([self.transition()],
                                             self.index(rows))[0]
        self.assertAlmostEqual(move.before, 0.40)
        self.assertAlmostEqual(move.after, 0.46)

    def test_a_line_that_moved_is_scored_on_the_line(self):
        # The probability is answering a different question at each end,
        # but the LINE itself carries the news, so the move is not lost.
        rows = [self.quote(1, 52, 46.0, "PLAYER 1 -2.5"),
                self.quote(2, 52, 52.0, "PLAYER 1 -6.5")]
        stats = collections.defaultdict(int)
        move = indrive.moves_for_transitions([self.transition()],
                                             self.index(rows), stats)[0]
        self.assertEqual(move.basis, indrive.LINE)
        self.assertEqual((move.before, move.after), (-2.5, -6.5))
        self.assertEqual(stats["move_scored_on_line"], 1)
        # Home has the ball and converted, so its own spread line should
        # have gone UP. It went down, so this is a miss.
        self.assertEqual(move.expected, +1)
        self.assertFalse(move.hit)

    def test_the_spread_line_follows_the_side_it_names(self):
        rows = [self.quote(1, 52, 46.0, "PLAYER 1 2.5"),
                self.quote(2, 52, 46.0, "PLAYER 1 6.5")]
        move = indrive.moves_for_transitions([self.transition()],
                                             self.index(rows))[0]
        self.assertEqual(move.basis, indrive.LINE)
        self.assertTrue(move.hit)

    def test_the_total_line_rises_for_over_AND_under(self):
        # Over and Under share one number, so a good offensive play pushes
        # it up whichever selection is carrying it. Reading the line with
        # the probability's expectation would score every Under backwards.
        for market_id in (54, 55):
            rows = [self.quote(1, market_id, 46.0, "Over 44.5"),
                    self.quote(2, market_id, 46.0, "Over 47.5")]
            move = indrive.moves_for_transitions([self.transition()],
                                                 self.index(rows))[0]
            self.assertEqual(move.basis, indrive.LINE, market_id)
            self.assertEqual(move.expected, +1, market_id)
            self.assertTrue(move.hit, market_id)

    def test_the_under_probability_still_opposes_the_play(self):
        # The line expectation and the probability expectation differ for
        # Under, which is the whole reason they are separate functions.
        self.assertEqual(indrive.expected_sign(55, self.HOME, +1), -1)
        self.assertEqual(indrive.expected_line_sign(55, self.HOME, +1), +1)
        # For a spread they agree.
        self.assertEqual(indrive.expected_sign(52, self.HOME, +1),
                         indrive.expected_line_sign(52, self.HOME, +1))

    def test_a_moneyline_has_no_line_to_score(self):
        self.assertEqual(indrive.expected_line_sign(50, self.HOME, +1), 0)
        self.assertEqual(indrive.expected_line_sign(51, self.HOME, +1), 0)

    def test_an_unreadable_line_is_not_a_move(self):
        rows = [self.quote(1, 52, 46.0, "PLAYER 1 to cover"),
                self.quote(2, 52, 52.0, "PLAYER 1 -6.5")]
        stats = collections.defaultdict(int)
        self.assertEqual(
            indrive.moves_for_transitions([self.transition()],
                                          self.index(rows), stats), [])
        self.assertEqual(stats["move_line_unreadable"], 1)

    def test_magnitudes_are_never_pooled_across_bases(self):
        # Probability points and handicap points are different units.
        t = self.transition()
        prob = indrive.Move(transition=t, stream=directional.PROD,
                            market_id=50, before=0.50, after=0.54,
                            line_before=None, line_after=None, expected=+1)
        line = indrive.Move(transition=t, stream=directional.PROD,
                            market_id=52, before=2.5, after=6.5,
                            line_before=2.5, line_after=6.5, expected=+1,
                            basis=indrive.LINE)
        b = indrive.block([prob, line], n_bootstrap=20)
        self.assertEqual((b["n_prob"], b["n_line"]), (1, 1))
        self.assertAlmostEqual(b["mean_signed"], 0.04)
        self.assertAlmostEqual(b["mean_line_signed"], 4.0)
        # The RATE still pools: right is right, whichever moved.
        self.assertEqual(b["decided"], 2)
        self.assertEqual(b["rate"], 1.0)

    def test_the_same_line_is_fine(self):
        rows = [self.quote(1, 52, 46.0, "PLAYER 1 -2.5"),
                self.quote(2, 52, 52.0, "PLAYER 1 -2.5")]
        moves = indrive.moves_for_transitions([self.transition()],
                                              self.index(rows))
        self.assertEqual(len(moves), 1)
        self.assertTrue(moves[0].hit)

    def test_a_dead_endpoint_is_not_a_move(self):
        rows = [self.quote(1, 50, 40.0, "PLAYER 1 to win", active="false"),
                self.quote(2, 50, 46.0, "PLAYER 1 to win")]
        stats = collections.defaultdict(int)
        self.assertEqual(
            indrive.moves_for_transitions([self.transition()],
                                          self.index(rows), stats), [])
        self.assertEqual(stats["move_not_live"], 1)

    def test_a_missing_endpoint_is_not_a_move(self):
        # One market quoted at the first message and nowhere else. Every
        # (stream, selection) with nothing at both ends is counted, which
        # is six selections across two streams.
        rows = [self.quote(1, 50, 40.0, "PLAYER 1 to win")]
        stats = collections.defaultdict(int)
        self.assertEqual(
            indrive.moves_for_transitions([self.transition()],
                                          self.index(rows), stats), [])
        self.assertEqual(stats["move_missing_quote"],
                         len(markets.MARKET_IDS) * 2)

    def test_a_transition_with_no_direction_is_skipped_whole(self):
        plays = [PlayRow(m, 1, self.HOME, d, dist, f, None)
                 for m, d, dist, f in [(1, 1, 10, 25), (2, 2, 7, 28)]]
        t = indrive.transitions_for_match("AF1", plays, [])[0]
        rows = [self.quote(1, 50, 40.0, "PLAYER 1 to win"),
                self.quote(2, 50, 46.0, "PLAYER 1 to win")]
        stats = collections.defaultdict(int)
        self.assertEqual(
            indrive.moves_for_transitions([t], self.index(rows), stats), [])
        self.assertEqual(stats["transition_no_direction"], 1)


class TestIndriveAggregation(unittest.TestCase):
    """Hit rates, the flat case, and the head to head."""

    HOME = "Home Team"
    T = dt.datetime(2026, 9, 18, 12, 0)

    def moves(self, pattern, stream=None):
        """One move per character: + right, - wrong, 0 flat."""
        stream = stream or directional.PROD
        out = []
        for i, mark in enumerate(pattern):
            plays = [PlayRow(1, 1, self.HOME, 2, 3, 32, None),
                     PlayRow(2, 1, self.HOME, 1, 10, 40, None)]
            t = indrive.transitions_for_match(f"AF{i}", plays, [])[0]
            delta = {"+": 0.02, "-": -0.02, "0": 0.0}[mark]
            out.append(indrive.Move(transition=t, stream=stream, market_id=50,
                                    before=0.50, after=0.50 + delta,
                                    line_before=None, line_after=None,
                                    expected=+1))
        return out

    def test_the_rate_is_out_of_the_moves_that_moved(self):
        b = indrive.block(self.moves("+++--000"), n_bootstrap=50)
        self.assertEqual(b["n"], 8)
        self.assertEqual(b["decided"], 5)
        self.assertEqual(b["hits"], 3)
        self.assertAlmostEqual(b["rate"], 0.6)
        self.assertEqual(b["flat"], 3)
        self.assertAlmostEqual(b["flat_share"], 3 / 8)

    def test_a_flat_price_is_not_counted_as_wrong(self):
        # Silence is not a miss. It is its own finding, and pretending a
        # model that never moved got it wrong would overstate the result.
        self.assertEqual(indrive.block(self.moves("0000"), 50)["rate"], None)
        self.assertEqual(indrive.block(self.moves("0000"), 50)["flat"], 4)

    def test_the_interval_is_clustered_on_matches(self):
        b = indrive.block(self.moves("+-+-+-+-+-"), n_bootstrap=200)
        self.assertEqual(b["matches"], 10)
        self.assertIsNotNone(b["ci_low"])
        self.assertLessEqual(b["ci_low"], b["rate"])
        self.assertGreaterEqual(b["ci_high"], b["rate"])

    def test_head_to_head_pairs_the_two_streams_on_one_question(self):
        prod = self.moves("++--", directional.PROD)
        cand = self.moves("+-+-", directional.CANDIDATE)
        h = indrive.head_to_head(prod + cand, n_bootstrap=50)
        # Move 0: both right -> tie. Move 3: both wrong -> tie.
        self.assertEqual(h["ties"], 2)
        # Move 1: prod right, candidate wrong. Move 2: the other way.
        self.assertEqual(h["decided"], 2)
        self.assertAlmostEqual(h["rate"], 0.5)

    def test_both_flat_is_counted_apart_from_a_tie(self):
        h = indrive.head_to_head(self.moves("00", directional.PROD)
                                 + self.moves("00", directional.CANDIDATE),
                                 n_bootstrap=50)
        self.assertEqual(h["both_flat"], 2)
        self.assertEqual(h["ties"], 0)
        self.assertEqual(h["decided"], 0)

    def test_an_unpaired_move_decides_nothing(self):
        h = indrive.head_to_head(self.moves("++++", directional.PROD),
                                 n_bootstrap=50)
        self.assertEqual(h["decided"], 0)
        self.assertEqual(h["pairs"], 4)

    def test_the_report_splits_in_drive_from_the_drive_end(self):
        plays = [PlayRow(m, 1, self.HOME, d, dist, f, None)
                 for m, d, dist, f in [(1, 2, 3, 32), (2, 1, 10, 40)]]
        plays.append(PlayRow(9, 1, "Away Team", 1, 10, 25, None))
        transitions = indrive.transitions_for_match("AF1", plays, [])
        moves = []
        for t in transitions:
            moves.append(indrive.Move(transition=t, stream=directional.PROD,
                                      market_id=50, before=0.5, after=0.52,
                                      line_before=None, line_after=None,
                                      expected=indrive.expected_sign(
                                          50, self.HOME, t.sign)))
        result = indrive.report(moves, n_bootstrap=50)
        self.assertEqual(result["in_drive"], 1)
        self.assertEqual(result["ending"], 1)
        self.assertEqual(result["moves"], 2)


class TestIndriveCsvAndHtml(unittest.TestCase):
    """The files it writes."""

    HOME = "Home Team"

    def move(self):
        plays = [PlayRow(1, 1, self.HOME, 2, 3, 32, None),
                 PlayRow(2, 1, self.HOME, 1, 10, 40, None)]
        t = indrive.transitions_for_match("AF1", plays, [])[0]
        return indrive.Move(transition=t, stream=directional.PROD,
                            market_id=52, before=0.50, after=0.54,
                            line_before=-2.5, line_after=-2.5, expected=+1)

    def test_a_move_row_needs_no_join_to_be_read(self):
        row = indrive.move_row(self.move())
        self.assertEqual(sorted(row), sorted(indrive.MOVE_FIELDS))
        self.assertEqual(row["outcome"], indrive.FIRST_DOWN)
        self.assertEqual(row["market"], "spread")
        self.assertEqual(row["selection"], "Home")
        self.assertEqual(row["hit"], 1)
        self.assertEqual(row["basis"], indrive.PROBABILITY)
        self.assertEqual(row["line_before"], -2.5)

    def test_the_move_row_carries_its_whole_transition(self):
        row = indrive.move_row(self.move())
        for field in indrive.TRANSITION_FIELDS:
            self.assertIn(field, row, field)

    def test_a_transition_row_is_exactly_the_csv_columns(self):
        plays = [PlayRow(1, 1, self.HOME, 2, 3, 32, None),
                 PlayRow(2, 1, self.HOME, 1, 10, 40, None)]
        for t in indrive.transitions_for_match("AF1", plays, []):
            self.assertEqual(sorted(indrive.transition_row(t)),
                             sorted(indrive.TRANSITION_FIELDS))

    def test_the_html_renders_and_names_every_outcome_it_has(self):
        from .. import html_indrive
        moves = [self.move()]
        plays = [PlayRow(1, 1, self.HOME, 2, 3, 32, None),
                 PlayRow(2, 1, self.HOME, 1, 10, 40, None)]
        transitions = indrive.transitions_for_match("AF1", plays, [])
        page = html_indrive.render(indrive.report(moves, 50),
                                   indrive.outcome_census(transitions),
                                   transitions, {})
        self.assertIn("<!DOCTYPE html>", page)
        self.assertIn("in-drive reaction", page)
        self.assertIn("First down", page)
        self.assertIn("</html>", page)

    def test_every_outcome_class_has_a_title_in_the_html(self):
        from .. import html_indrive
        for outcome in indrive.OUTCOME_ORDER:
            self.assertIn(outcome, html_indrive.TITLES, outcome)

    def test_the_rate_ramp_is_centred_on_a_coin(self):
        from .. import html_indrive
        # Worse than random is the red end, however close to 50% it is.
        self.assertEqual(html_indrive._rate_class(0.49), "g4")
        self.assertEqual(html_indrive._rate_class(0.51), "g4")
        self.assertEqual(html_indrive._rate_class(0.95), "g0")
        # And it is monotone in between.
        steps = [html_indrive._rate_class(r)
                 for r in (0.50, 0.55, 0.60, 0.70, 0.90)]
        self.assertEqual(steps, sorted(steps, reverse=True))

    def test_the_ramp_is_spelled_out_in_a_key(self):
        from .. import html_indrive
        key = html_indrive._rate_key()
        for step in range(5):
            self.assertIn(f'class="key g{step}"', key)
        self.assertIn("50% is a coin", key)


class TestIndriveChecks(unittest.TestCase):
    """Findings about the analysis, not about the models."""

    def block(self, n=1000, decided=900, rate=0.8, lo=0.78, hi=0.82, flat=0.1):
        return {"n": n, "decided": decided, "rate": rate, "ci_low": lo,
                "ci_high": hi, "flat_share": flat, "hits": round(rate * decided),
                "matches": 100, "p_value": 0.0, "flat": round(flat * n),
                "n_prob": n, "n_line": 0, "mean_move": 0.02,
                "mean_signed": 0.01, "mean_line_move": None,
                "mean_line_signed": None}

    def result(self, prod_outcomes, cand_outcomes=None, selections=None):
        cand_outcomes = cand_outcomes or prod_outcomes
        def stream(outcomes):
            return {"overall": self.block(), "in_drive": self.block(),
                    "ending": self.block(), "by_outcome": outcomes,
                    "by_period": {}, "by_market": {},
                    "by_selection": selections or {}, "by_basis": {}}
        return {"streams": {directional.PROD: stream(prod_outcomes),
                            directional.CANDIDATE: stream(cand_outcomes)},
                "head_to_head": {}, "moves": 0, "matches": 0,
                "transitions": 0, "in_drive": 0, "ending": 0}

    def transition(self, outcome):
        plays = [PlayRow(1, 1, "Home Team", 2, 3, 32, None),
                 PlayRow(2, 1, "Home Team", 1, 10, 40, None)]
        t = indrive.transitions_for_match("AF1", plays, [])[0]
        return dataclasses.replace(t, outcome=outcome)

    def move(self, outcome, basis=None, delta=0.02):
        return indrive.Move(transition=self.transition(outcome),
                            stream=directional.PROD, market_id=50,
                            before=0.5, after=0.5 + delta, line_before=None,
                            line_after=None, expected=+1,
                            basis=basis or indrive.PROBABILITY)

    def findings(self, result, transitions=(), moves=()):
        return {(sev, subject): message
                for sev, subject, message in
                indrive.checks(result, list(transitions), list(moves))}

    def test_a_class_below_a_coin_on_both_streams_is_an_error(self):
        # This is the field_goal case. Two models built separately do not
        # agree with each other in the wrong direction on 1,800 plays, so
        # the expectation is what is backwards.
        below = self.block(rate=0.32, lo=0.29, hi=0.36)
        found = self.findings(self.result({indrive.FIELD_GOAL: below}))
        self.assertIn((indrive.ERROR, indrive.FIELD_GOAL), found)
        self.assertIn("backwards", found[(indrive.ERROR, indrive.FIELD_GOAL)])

    def test_one_stream_below_a_coin_is_not_enough(self):
        # One model getting it wrong is a finding about that model, which
        # is not what this check is for.
        found = self.findings(self.result(
            {indrive.BIG_GAIN: self.block(rate=0.32, lo=0.29, hi=0.36)},
            {indrive.BIG_GAIN: self.block(rate=0.80)}))
        self.assertNotIn((indrive.ERROR, indrive.BIG_GAIN), found)

    def test_an_interval_that_still_touches_a_coin_is_not_an_error(self):
        found = self.findings(self.result(
            {indrive.BIG_GAIN: self.block(rate=0.48, lo=0.44, hi=0.52)}))
        self.assertNotIn((indrive.ERROR, indrive.BIG_GAIN), found)

    def test_a_mostly_flat_class_is_flagged_as_a_minority_report(self):
        # failed_conversion: 96% flat, so 74.8% is computed on 4% of rows.
        found = self.findings(self.result(
            {indrive.FAILED_CONVERSION: self.block(n=5762, decided=222,
                                                   rate=0.748, flat=0.961)}))
        message = found[(indrive.WARN, indrive.FAILED_CONVERSION)]
        self.assertIn("FLAT", message)
        self.assertIn("does not move", message)

    def test_a_class_the_quotes_barely_cover_is_flagged(self):
        # extra_point: 0.4 moves per transition against a median near 5.
        transitions, moves = [], []
        for outcome, n_t, n_m in ((indrive.FIRST_DOWN, 100, 500),
                                  (indrive.NO_GAIN, 100, 500),
                                  (indrive.LOSS, 100, 500),
                                  (indrive.EXTRA_POINT, 100, 20)):
            transitions += [self.transition(outcome)] * n_t
            moves += [self.move(outcome)] * n_m
        found = self.findings(self.result({}), transitions, moves)
        self.assertIn((indrive.WARN, indrive.EXTRA_POINT), found)
        self.assertIn("per transition",
                      found[(indrive.WARN, indrive.EXTRA_POINT)])
        self.assertNotIn((indrive.WARN, indrive.FIRST_DOWN), found)

    def test_a_flat_line_move_is_impossible_and_says_so(self):
        # A move is scored on the line only where the line CHANGED, so a
        # flat one means the basis was set somewhere it should not be.
        moves = [self.move(indrive.FIRST_DOWN, indrive.LINE, delta=0.0)]
        found = self.findings(self.result({}), [], moves)
        self.assertIn((indrive.ERROR, "basis"), found)
        self.assertIn("cannot", found[(indrive.ERROR, "basis")])

    def test_a_real_line_move_does_not_trip_the_invariant(self):
        moves = [self.move(indrive.FIRST_DOWN, indrive.LINE, delta=1.0)]
        self.assertNotIn((indrive.ERROR, "basis"),
                         self.findings(self.result({}), [], moves))

    def test_two_sides_of_a_market_disagreeing_is_worth_saying(self):
        selections = {("moneyline", "Home"): self.block(rate=0.81),
                      ("moneyline", "Away"): self.block(rate=0.70)}
        found = self.findings(self.result({}, selections=selections))
        self.assertIn((indrive.NOTE, "moneyline"), found)
        self.assertIn("complements", found[(indrive.NOTE, "moneyline")])

    def test_matching_sides_say_nothing(self):
        selections = {("moneyline", "Home"): self.block(rate=0.811),
                      ("moneyline", "Away"): self.block(rate=0.812)}
        self.assertNotIn((indrive.NOTE, "moneyline"),
                         self.findings(self.result({}, selections=selections)))

    def test_a_clean_run_produces_nothing(self):
        found = self.findings(self.result(
            {indrive.FIRST_DOWN: self.block(n=20000, decided=18000)}))
        self.assertEqual(found, {})

    def test_the_html_renders_the_findings(self):
        from .. import html_indrive
        findings = [(indrive.ERROR, "field_goal", "below a coin everywhere"),
                    (indrive.NOTE, "total", "sides differ")]
        page = html_indrive.render(indrive.report([], 20), {}, [], {}, findings)
        self.assertIn("Checks", page)
        self.assertIn("below a coin everywhere", page)
        self.assertIn("Error", page)

    def test_the_html_says_so_when_everything_passes(self):
        from .. import html_indrive
        page = html_indrive.render(indrive.report([], 20), {}, [], {}, [])
        self.assertIn("all pass", page)


class TestDriveOutcomes(unittest.TestCase):
    """One row per drive: what it produced and what the scoreboard did."""

    HOME, AWAY = "Home Team", "Away Team"

    def feed(self):
        plays = [PlayRow(m, p, t, d, dist, f, None)
                 for m, p, t, d, dist, f in [
                     (10, 1, self.HOME, 1, 10, 25), (12, 1, self.HOME, 2, 4, 31),
                     (14, 1, self.HOME, 3, 1, 95),
                     (30, 1, self.AWAY, 1, 10, 25), (32, 1, self.AWAY, 2, 3, 32),
                     (50, 2, self.HOME, 1, 10, 40), (52, 2, self.HOME, 2, 6, 44)]]
        scores = [ScoreRow(16, 1, 6, None, 6, 0),     # the touchdown
                  ScoreRow(18, 1, 1, None, 7, 0),     # and its PAT
                  ScoreRow(40, 1, None, 3, 7, 3)]     # an away field goal
        return plays, scores

    def outcomes(self):
        return indrive.drive_outcomes("AF1", *self.feed())

    def test_every_drive_gets_a_row_including_the_last(self):
        # The transition view cannot see the last drive of a match at all:
        # a drive-ending transition needs a FOLLOWING drive to point at.
        outcomes = self.outcomes()
        self.assertEqual(len(outcomes), 3)
        self.assertEqual([o.drive_number for o in outcomes], [1, 2, 3])
        self.assertEqual(outcomes[-1].ended, indrive.MATCH_END)

    def test_a_drive_owns_the_score_that_ended_it(self):
        first = self.outcomes()[0]
        self.assertEqual(first.outcome, indrive.TOUCHDOWN)
        # Six for the touchdown plus one for the PAT: both land in the
        # window between this drive's last play and the next drive's first.
        self.assertEqual(first.points_for, 7)
        self.assertEqual((first.score_p1_before, first.score_p2_before), (0, 0))
        self.assertEqual((first.score_p1_after, first.score_p2_after), (7, 0))

    def test_points_are_signed_to_the_side_with_the_ball(self):
        second = self.outcomes()[1]
        self.assertEqual(second.offensive_team, self.AWAY)
        self.assertEqual(second.outcome, indrive.FIELD_GOAL)
        self.assertEqual(second.points_for, 3)
        self.assertEqual(second.points_against, 0)

    def test_a_scoreless_drive_says_so_rather_than_saying_nothing(self):
        last = self.outcomes()[-1]
        self.assertEqual(last.outcome, indrive.NO_POINTS)
        self.assertEqual(last.points_for, 0)
        self.assertEqual(last.score_diff_before, last.score_diff_after)

    def test_the_defence_scoring_lands_on_the_offence_row(self):
        plays = [PlayRow(10, 1, self.HOME, 1, 10, 25, None),
                 PlayRow(30, 1, self.AWAY, 1, 10, 41, None)]
        scores = [ScoreRow(20, 1, None, 6, 0, 6)]
        first = indrive.drive_outcomes("AF1", plays, scores)[0]
        self.assertEqual(first.offensive_team, self.HOME)
        self.assertEqual(first.outcome, indrive.POINTS_AGAINST)
        self.assertEqual((first.points_for, first.points_against), (0, 6))

    def test_how_it_ended_is_recorded(self):
        outcomes = self.outcomes()
        self.assertEqual(outcomes[0].ended, indrive.HANDOVER)
        self.assertEqual(outcomes[1].ended, indrive.HANDOVER)
        self.assertEqual(outcomes[2].ended, indrive.MATCH_END)

    def test_a_break_that_kept_the_team_label_is_marked(self):
        # Points end a drive even where the label never changed, which is
        # the case the feed never said possession turned over.
        plays = [PlayRow(10, 1, self.HOME, 1, 10, 25, None),
                 PlayRow(30, 1, self.HOME, 2, 7, 28, None)]
        scores = [ScoreRow(20, 1, 7, None, 7, 0)]
        outcomes = indrive.drive_outcomes("AF1", plays, scores)
        self.assertEqual(len(outcomes), 2)
        self.assertEqual(outcomes[0].ended, indrive.SAME_TEAM)


class TestDriveReconciliation(unittest.TestCase):
    """The points the drives claim against the points the feed sent."""

    HOME, AWAY = "Home Team", "Away Team"

    def test_the_windows_partition_the_match(self):
        # Every score belongs to exactly one drive, which is what makes
        # the check a test rather than a restatement of itself.
        plays = [PlayRow(m, 1, t, 1, 10, f, None) for m, t, f in
                 [(10, self.HOME, 25), (30, self.AWAY, 41), (50, self.HOME, 33)]]
        scores = [ScoreRow(5, 1, 3, None, 3, 0),      # before any drive ended
                  ScoreRow(20, 1, 7, None, 10, 0),
                  ScoreRow(40, 1, None, 7, 10, 7),
                  ScoreRow(90, 1, 6, None, 16, 7)]    # after the last play
        outcomes = indrive.drive_outcomes("AF1", plays, scores)
        claimed = sum(o.points_for + o.points_against for o in outcomes)
        self.assertEqual(claimed, 3 + 7 + 7 + 6)
        self.assertTrue(indrive.reconcile(outcomes, scores)["ok"])

    def test_a_score_after_the_last_play_still_belongs_to_a_drive(self):
        plays = [PlayRow(10, 1, self.HOME, 1, 10, 25, None)]
        scores = [ScoreRow(99, 1, 6, None, 6, 0)]
        outcomes = indrive.drive_outcomes("AF1", plays, scores)
        self.assertEqual(outcomes[0].points_for, 6)
        self.assertTrue(indrive.reconcile(outcomes, scores)["ok"])

    def test_a_score_before_the_first_play_still_belongs_to_a_drive(self):
        plays = [PlayRow(10, 1, self.HOME, 1, 10, 25, None)]
        scores = [ScoreRow(1, 1, 3, None, 3, 0)]
        outcomes = indrive.drive_outcomes("AF1", plays, scores)
        self.assertEqual(outcomes[0].points_for, 3)
        self.assertTrue(indrive.reconcile(outcomes, scores)["ok"])

    def test_a_mismatch_is_reported_as_an_error(self):
        plays = [PlayRow(10, 1, self.HOME, 1, 10, 25, None)]
        scores = [ScoreRow(20, 1, 7, None, 7, 0)]
        outcomes = indrive.drive_outcomes("AF1", plays, scores)
        # Hand the check a score the drives were never built from.
        extra = list(scores) + [ScoreRow(99, 1, 6, None, 13, 0)]
        findings = indrive.checks({"streams": {}}, [], [], outcomes, extra)
        self.assertIn((indrive.ERROR, "reconcile"),
                      {(f[0], f[1]) for f in findings})

    def test_a_clean_match_reports_nothing(self):
        plays = [PlayRow(10, 1, self.HOME, 1, 10, 25, None)]
        scores = [ScoreRow(20, 1, 7, None, 7, 0)]
        outcomes = indrive.drive_outcomes("AF1", plays, scores)
        findings = indrive.checks({"streams": {}}, [], [], outcomes, scores)
        self.assertNotIn((indrive.ERROR, "reconcile"),
                         {(f[0], f[1]) for f in findings})

    def test_no_drives_means_no_rows_rather_than_a_crash(self):
        self.assertEqual(indrive.drive_outcomes("AF1", [], []), [])


class TestDriveCensusAndCsv(unittest.TestCase):

    HOME, AWAY = "Home Team", "Away Team"

    def outcomes(self):
        plays = [PlayRow(m, 1, t, 1, 10, f, None) for m, t, f in
                 [(10, self.HOME, 25), (30, self.AWAY, 41), (50, self.HOME, 33)]]
        scores = [ScoreRow(20, 1, 7, None, 7, 0)]
        return indrive.drive_outcomes("AF1", plays, scores)

    def test_the_census_counts_points_as_well_as_drives(self):
        census = indrive.drive_census(self.outcomes())
        self.assertEqual(census[indrive.TOUCHDOWN]["n"], 1)
        self.assertEqual(census[indrive.TOUCHDOWN]["points"], 7)
        self.assertEqual(census[indrive.NO_POINTS]["n"], 2)

    def test_every_drive_class_is_in_the_census(self):
        census = indrive.drive_census([])
        self.assertEqual(list(census), indrive.DRIVE_OUTCOME_ORDER)

    def test_a_drive_row_is_exactly_the_csv_columns(self):
        for o in self.outcomes():
            self.assertEqual(sorted(indrive.drive_row(o)),
                             sorted(indrive.DRIVE_FIELDS))

    def test_the_row_carries_the_score_either_side(self):
        row = indrive.drive_row(self.outcomes()[0])
        self.assertEqual((row["score_p1_before"], row["score_p1_after"]), (0, 7))
        self.assertEqual(row["score_diff_after"], 7)
        self.assertEqual(row["outcome"], indrive.TOUCHDOWN)

    def test_the_html_renders_the_drive_panel(self):
        from .. import html_indrive
        outcomes = self.outcomes()
        page = html_indrive.render(indrive.report([], 20), {}, [], {}, [],
                                   indrive.drive_census(outcomes), outcomes)
        self.assertIn("What each drive produced", page)
        self.assertIn("Touchdown", page)
        self.assertIn("Last of the match", page)

    def test_every_drive_class_has_a_title_in_the_html(self):
        from .. import html_indrive
        for outcome in indrive.DRIVE_OUTCOME_ORDER:
            self.assertIn(outcome, html_indrive.DRIVE_TITLES, outcome)


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
        rows = [("AF1", 50, BASE, 85.26, 1.11, "Home team (PLAYER 1) to win",
                 300, "open", "true")]
        index = directional.index_by_message(rows)
        quote = index[("AF1", 50)][300]
        self.assertEqual(quote.probability, 85.26)
        self.assertEqual(quote.description, "Home team (PLAYER 1) to win")
        self.assertTrue(quote.live)

    def test_index_by_message_skips_null_message_or_probability(self):
        rows = [("AF1", 50, BASE, 85.0, 1.1, "d", None, "open", "true"),
                ("AF1", 50, BASE, None, 1.1, "d", 300, "open", "true")]
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


def _srgb_to_linear(channel):
    channel /= 255
    return (channel / 12.92 if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4)


def _channels(hex_colour):
    return tuple(int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))


def _oklab_l(hex_colour):
    """The perceptual lightness of a colour, 0-1."""
    r, g, b = (_srgb_to_linear(c) for c in _channels(hex_colour))
    long = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    medium = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    short = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return 0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short


def _contrast(one, other):
    """WCAG contrast ratio between two hex colours."""
    def luminance(hex_colour):
        r, g, b = (_srgb_to_linear(c) for c in _channels(hex_colour))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    high, low = sorted((luminance(one), luminance(other)), reverse=True)
    return (high + 0.05) / (low + 0.05)


class TestGapRamp(unittest.TestCase):
    """Green near zero, red far from it -- on magnitude, not sign."""

    def setUp(self):
        from .. import html_full
        self.h = html_full

    def test_the_steps_run_green_to_red_in_order(self):
        cuts = (0.02, 0.05, 0.10, 0.20)
        self.assertEqual(self.h._gap(0.000, cuts), "g0")
        self.assertEqual(self.h._gap(0.020, cuts), "g0")   # the cut is inclusive
        self.assertEqual(self.h._gap(0.021, cuts), "g1")
        self.assertEqual(self.h._gap(0.050, cuts), "g1")
        self.assertEqual(self.h._gap(0.099, cuts), "g2")
        self.assertEqual(self.h._gap(0.150, cuts), "g3")
        self.assertEqual(self.h._gap(0.900, cuts), "g4")

    def test_a_gap_is_a_distance_so_sign_does_not_matter(self):
        # This is the bug it replaces. _cls read the sign, so a calibration
        # gap of +0.40 -- badly over-predicted -- came out GREEN, while
        # -0.001, near perfect, came out red.
        cuts = self.h.PROB_GAP
        self.assertEqual(self.h._gap(+0.40, cuts), self.h._gap(-0.40, cuts))
        self.assertEqual(self.h._gap(-0.001, cuts), "g0")
        self.assertEqual(self.h._gap(+0.400, cuts), "g4")
        self.assertEqual(self.h._cls(+0.40), "good")   # the old reading
        self.assertEqual(self.h._cls(-0.001), "bad")

    def test_a_missing_value_gets_no_colour_at_all(self):
        # A dashed cell is "not comparable", which is not a small gap.
        self.assertEqual(self.h._gap(None, self.h.PROB_GAP), "")
        self.assertEqual(self.h._gap("", self.h.PROB_GAP), "")
        self.assertEqual(self.h._gap(None, self.h.PROB_GAP, "dim"), "dim")

    def test_it_rides_on_top_of_the_class_the_cell_already_had(self):
        self.assertEqual(self.h._gap(0.0, self.h.PROB_GAP, "dim"), "dim g0")

    def test_every_scale_has_four_rising_cut_points(self):
        for name in ("PROB_GAP", "PROB_DELTA", "LINE_GAP", "MESSAGE_GAP"):
            cuts = getattr(self.h, name)
            self.assertEqual(len(cuts), 4, name)
            self.assertEqual(list(cuts), sorted(cuts), name)

    def test_the_key_names_every_step_so_colour_is_never_the_only_cue(self):
        key = self.h._gap_key(self.h.PROB_GAP, "Gap", ".2f")
        for step in range(5):
            self.assertIn(f'class="key g{step}"', key)
        self.assertIn("&le;0.02", key)
        self.assertIn("&gt;0.20", key)

    def test_the_ramp_moves_in_lightness_as_well_as_hue(self):
        # Green and red are the one pair red-green colour blindness cannot
        # separate, so hue alone cannot carry the ordering. Lightness has
        # to move monotonically, by more than the 0.06 step floor, in BOTH
        # themes. The DIRECTION differs by theme on purpose: prominence
        # rises with severity either way, darkest on light and brightest
        # on dark, so the worst numbers shout loudest with no hue at all.
        import re
        from .. import html_style
        source = html_style.CSS
        found = re.findall(r"--g0:(#\w{6}); --g1:(#\w{6}); --g2:(#\w{6});"
                           r" --g3:(#\w{6}); --g4:(#\w{6});", source)
        self.assertEqual(len(found), 3)   # light, media-query dark, forced dark
        self.assertEqual(len(set(found[1:])), 1, "both dark rules must agree")
        for ramp in (found[0], found[1]):
            lightness = [_oklab_l(step) for step in ramp]
            ordered = (lightness == sorted(lightness)
                       or lightness == sorted(lightness, reverse=True))
            self.assertTrue(ordered, f"{ramp} is not monotone: {lightness}")
            gaps = [abs(lightness[i + 1] - lightness[i]) for i in range(4)]
            self.assertTrue(all(g >= 0.06 for g in gaps), f"{ramp}: {gaps}")

    def test_the_number_stays_readable_on_every_step(self):
        # The ramp IS the number now, not a wash behind it, so each step
        # has to clear AA against the PANEL it sits on rather than against
        # the ink -- and a colour light enough to be a nice fill is not
        # necessarily dark enough to be read as type.
        import re
        from .. import html_style
        source = html_style.CSS
        found = re.findall(r"--g0:(#\w{6}); --g1:(#\w{6}); --g2:(#\w{6});"
                           r" --g3:(#\w{6}); --g4:(#\w{6});", source)
        for ramp, panel in ((found[0], "#ffffff"), (found[1], "#1f1f23")):
            for step in ramp:
                self.assertGreaterEqual(_contrast(step, panel), 4.5,
                                        f"{step} on {panel}")

    def test_the_ramp_colours_the_type_not_the_cell(self):
        # A wash of filled cells reads as a heat map; the table wanted a
        # table. Nothing in the ramp may paint a background.
        from .. import html_style
        for step in range(5):
            self.assertIn(f"td.g{step}{{color:var(--g{step})", html_style.CSS)
            self.assertNotIn(f"td.g{step}{{background", html_style.CSS)


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

    def test_message_gap_is_shaded_by_how_far_off_it_is(self):
        offset = [pair(0.6, 0.8, True, match="AF9", market_id=50, message=100, gap=2)]
        report = directional.build_full_report(offset, n_bootstrap=20)
        rendered = self.html_full.render(report, self.header, self.stats, offset,
                                         self.scan)
        self.assertIn('data-v="2" class="g2"', rendered)

    def test_an_exact_message_match_sits_on_the_green_end(self):
        exact = [pair(0.6, 0.8, True, match="AF9", market_id=50, message=100, gap=0)]
        report = directional.build_full_report(exact, n_bootstrap=20)
        rendered = self.html_full.render(report, self.header, self.stats, exact,
                                         self.scan)
        self.assertIn('data-v="0" class="g0"', rendered)

    def test_the_ramp_is_spelled_out_rather_than_left_to_colour(self):
        rendered = self._render()
        self.assertIn("gapkey", rendered)
        for step in range(5):
            self.assertIn(f'class="key g{step}"', rendered)

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

    def _row_for(self, pairs):
        """The pair table's tbody for one set of pairs."""
        report = directional.build_full_report(pairs, n_bootstrap=20)
        rendered = self.html_full.render(report, self.header, self.stats,
                                         pairs, self.scan)
        body = rendered[rendered.index('id="pairTable"'):]
        return body[body.index("<tbody>"):body.index("</tbody>")]

    def _cells_for(self, pairs):
        """Header -> cell text for a single-row pair table."""
        report = directional.build_full_report(pairs, n_bootstrap=20)
        rendered = self.html_full.render(report, self.header, self.stats,
                                         pairs, self.scan)
        body = rendered[rendered.index('id="pairTable"'):]
        heads = [re.sub(r"<[^>]+>", "", h).strip() for h in
                 re.findall(r"<th[^>]*>(.*?)</th>",
                            body[:body.index("</thead>")], re.S)]
        row = body[body.index("<tbody>"):body.index("</tbody>")]
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in
                 re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.S)]
        return dict(zip(heads, cells))

    def test_probability_columns_are_blank_when_the_lines_differ(self):
        # A probability that refers to a different line is not comparable to
        # one that refers to another, and a line error is in points -- so a
        # number here would invite exactly the wrong comparison.
        cells = self._cells_for(
            [line_pair(0.50, 0.01, 44.5, 60.5, 24, 21, match="AF041170926")])
        for column in ("&Delta;prob", "Prod err", "Cand err"):
            self.assertEqual(cells[column], "&mdash;", column)
        # The two lines and each stream's own probability still show -- they
        # are facts about the row. It is only the comparisons between them
        # that are withheld.
        self.assertEqual(cells["Prod line"], "+44.5")
        self.assertEqual(cells["Cand line"], "+60.5")
        self.assertEqual(cells["Prod prob"], "0.5000")
        self.assertEqual(cells["Cand prob"], "0.0100")
        # And the line error in points never appears in an error column.
        self.assertNotIn("15.5000", self._row_for(
            [line_pair(0.50, 0.01, 44.5, 60.5, 24, 21)]))

    def test_the_closer_column_uses_the_line_when_the_lines_differ(self):
        pairs = [line_pair(0.50, 0.01, 44.5, 60.5, 24, 21)]
        row = self._row_for(pairs)
        # Prod's line is nearer, so prod is closer -- despite the candidate
        # having much the better probability against its own outcome.
        self.assertIn(">prod</span>", row)
        self.assertNotIn(">cand</span>", row)

    def test_same_line_rows_keep_their_probability_columns(self):
        row = self._row_for([line_pair(0.55, 0.62, 44.5, 44.5, 24, 21)])
        self.assertIn("0.0700", row)   # delta prob
        self.assertIn("0.4500", row)   # prod error
        self.assertIn("0.3800", row)   # candidate error

    def test_pair_rows_carry_the_game_state(self):
        p = directional.PairedObservation(
            match_code="AF1", drive_number=1, period_number=4,
            score_p1=21, score_p2=22, offensive_team="Away Team",
            market_id=50, message_count=428, message_gap=0,
            prod_probability=0.08, candidate_probability=0.664,
            prod_line=None, candidate_line=None, prod_outcome=False,
            candidate_outcome=False, realized=None,
            publish_time=dt.datetime(2026, 9, 17, 20, 5),
            field_position=55, down_number=3, distance=2)
        cells = self._cells_for([p])
        self.assertEqual(cells["Field"], "55")
        self.assertEqual(cells["D&amp;D"], "3&amp;2")

    def test_missing_state_shows_a_dash_not_a_zero(self):
        cells = self._cells_for([pair(0.5, 0.6, True)])
        self.assertEqual(cells["Field"], "&mdash;")
        self.assertEqual(cells["D&amp;D"], "&mdash;")

    def test_to_end_counts_down_to_the_match_last_quote(self):
        # No game clock exists in the feed, so this is the wall-clock proxy:
        # seconds from each snapshot to the last quote of its own match.
        base = dt.datetime(2026, 9, 17, 20, 0)
        early = pair(0.5, 0.6, True, match="AF1", message=100)
        late = pair(0.5, 0.6, True, match="AF1", message=428)
        early = dataclasses.replace(early, publish_time=base)
        late = dataclasses.replace(late, publish_time=base + dt.timedelta(seconds=300))
        row = self._row_for([early, late])
        self.assertIn(">300</td>", row)   # the early one is 300s from the end
        self.assertIn(">0</td>", row)     # the last one is the end

    def test_the_final_snapshot_of_a_match_is_flagged_as_near_the_end(self):
        base = dt.datetime(2026, 9, 17, 20, 0)
        p = dataclasses.replace(pair(0.5, 0.6, True, match="AF1"),
                                publish_time=base)
        self.assertIn('class="warn"', self._row_for([p]))

    def test_a_non_live_pair_shows_but_compares_nothing(self):
        p = dataclasses.replace(pair(0.9, 0.1, True), prod_live=False)
        cells = self._cells_for([p])
        self.assertEqual(cells["Live"], "prod")
        for column in ("&Delta;prob", "Prod err", "Cand err"):
            self.assertEqual(cells[column], "&mdash;", column)
        # The two probabilities are still facts about the row.
        self.assertEqual(cells["Prod prob"], "0.9000")
        self.assertIn('class="notlive"', self._row_for([p]))

    def test_a_live_pair_is_not_marked(self):
        cells = self._cells_for([pair(0.9, 0.1, True)])
        self.assertEqual(cells["Live"], "live")
        self.assertNotIn('class="notlive"', self._row_for([pair(0.9, 0.1, True)]))

    def test_off_anchor_rows_flag_their_down_and_distance(self):
        p = dataclasses.replace(
            pair(0.5, 0.6, True), down_number=2, distance=7,
            anchor=drives.MID_DRIVE)
        row = self._row_for([p])
        self.assertIn('class="warn">2&amp;7</td>', row)

    def test_clean_anchor_rows_are_not_flagged(self):
        p = dataclasses.replace(pair(0.5, 0.6, True), down_number=1,
                                distance=10, anchor=drives.FIRST_DOWN)
        self.assertIn('class="">1&amp;10</td>', self._row_for([p]))

    def test_the_anchor_panel_reports_the_split(self):
        pairs = [dataclasses.replace(pair(0.5, 0.6, True, match=f"AF{i}"),
                                     anchor=(drives.MID_DRIVE if i < 3
                                             else drives.FIRST_DOWN))
                 for i in range(10)]
        report = directional.build_full_report(pairs, n_bootstrap=20)
        rendered = self.html_full.render(report, self.header, self.stats,
                                         pairs, self.scan)
        block = rendered[rendered.index("Snapshot anchor"):]
        block = block[:block.index("</section>")]
        self.assertIn("Mid-drive snap", block)
        self.assertIn("70.0%", block)

    def test_a_low_clean_share_stops_the_checks_line_passing(self):
        pairs = [dataclasses.replace(pair(0.5, 0.6, True, match=f"AF{i}"),
                                     anchor=drives.MID_DRIVE)
                 for i in range(10)]
        report = directional.build_full_report(pairs, n_bootstrap=20)
        summary = self.html_full._checks_summary(report)
        self.assertIn("opening 1st", summary)
        self.assertNotIn("all pass", summary)

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

    def test_checks_carry_a_row_per_selection(self):
        pairs = []
        for i in range(6):
            for market_id in (50, 51, 54, 55):
                pairs.append(line_pair(0.5, 0.6, 44.5, 44.5, 24, 21,
                                       match=f"AF{i}", market_id=market_id))
        report = directional.build_full_report(pairs, n_bootstrap=50)
        rendered = self.html_full.render(report, {"paired_matches": 6}, {},
                                         pairs, self.scan)
        block = rendered[rendered.index("By selection"):]
        block = block[:block.index("</section>")]
        for label in ("Home", "Away", "Over", "Under"):
            self.assertIn(f">{label}</td>", block)
        # And it says which side the pooled tables read.
        self.assertIn("&check;", block)

    def test_rows_can_be_pinned_in_any_table(self):
        script = self.rendered.split("<script>")[1]
        self.assertIn("classList.toggle('picked')", script)
        self.assertIn("TBODY", script)
        self.assertIn("Escape", script)
        # Selecting text inside a row must not also pin it.
        self.assertIn("window.getSelection()", script)
        # Inset shadows, so pinning never reflows the table.
        self.assertIn("tbody tr.picked > *{background:var(--pick)", self.rendered)
        self.assertIn("box-shadow:inset", self.rendered)
        self.assertIn("--pick:", self.rendered)
        # The affordance is still named, just not explained at length.
        self.assertIn(">pin</span>", self.rendered)

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

    def test_the_directional_result_is_two_tables(self):
        head = self.rendered[self.rendered.index('id="directional"'):]
        head = head[:head.index("</section>")]
        self.assertEqual(head.count("<table>"), 2)
        self.assertEqual(head.count("<dl"), 0)     # was three stat lists
        self.assertNotIn("&Delta;MAE", head)
        # One row for the combined reading, then the two halves of it.
        overall, views = head.split("<table>")[1], head.split("<table>")[2]
        self.assertIn("<th>Overall</th>", overall)
        self.assertIn("<th>Same line</th>", views)
        self.assertIn("<th>Different line</th>", views)

    def test_the_directional_tables_keep_every_number_the_stat_lists_had(self):
        head = self.rendered[self.rendered.index('id="directional"'):]
        head = head[:head.index("</section>")]
        for column in ("Pairs", "Matches", "Cand win", "Match vote",
                       "On prob", "On line", "95% CI"):
            self.assertIn(f">{column}</th>", head, column)

    def test_report_carries_no_explanatory_prose(self):
        # The report is a dashboard, not a write-up: headings, tables and
        # numbers. Column meanings live in header tooltips, which cost no
        # space until asked for.
        self.assertNotIn('class="note"', self.rendered)
        self.assertNotIn('class="sub"', self.rendered)
        self.assertGreater(self.rendered.count("<th title="), 20)

    def test_no_heading_asks_itself_a_question(self):
        # Headings name the thing; they do not introduce it.
        import re
        for heading in re.findall(r"<h[123][^>]*>(.*?)</h[123]>", self.rendered,
                                  re.S):
            self.assertNotIn("?", heading, heading)

    def test_nothing_on_the_page_is_a_sentence(self):
        # Anything long enough to be prose, outside the tooltips that only
        # appear on hover, is the thing this strips.
        import re
        body = self.rendered[self.rendered.index("<body"):]
        body = re.sub(r"<script.*?</script>", "", body, flags=re.S)
        body = re.sub(r'title="[^"]*"', "", body)
        for line in re.sub(r"<[^>]+>", "\n", body).split("\n"):
            words = line.strip().split()
            self.assertLess(len(words), 12, line.strip())

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
