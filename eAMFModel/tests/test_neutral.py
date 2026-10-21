"""The shared modules (feed, markets, stream, books, loading): v2.1 imports none of v14's files, v14's files
still offer the shared code under their old names, and sim14's switches still reach what it shares."""

import subprocess
import sys
import textwrap
import unittest

from .. import books, feed, loading, markets, prior_check, sim14, stream, strength_check, v14, v14_stream
from .fakes import _matches

V14_FILES = ("eAMFModel.v14", "eAMFModel.sim14", "eAMFModel.v14_stream", "eAMFModel.prior_check",
             "eAMFModel.strength_check")
V21_FILES = ("v21", "v21_audit", "v21_coverage", "v21_endgame", "v21_gamers", "v21_react", "v21_react_fit",
             "v21_report", "v21_rules", "v21_stream", "v21_timing", "trader", "feed", "markets", "stream", "books",
             "loading")


class TestV21StandsAlone(unittest.TestCase):

    def test_v21_imports_none_of_v14s_files(self):
        """Every v2.1 module imports with v14's files unimportable, as if deleted."""
        code = textwrap.dedent(f"""
            import importlib, importlib.abc, sys
            BLOCKED = {V14_FILES!r}
            class Block(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path, target=None):
                    if name in BLOCKED:
                        raise ImportError(name + " is blocked")
            sys.meta_path.insert(0, Block())
            for m in {V21_FILES!r}:
                importlib.import_module("eAMFModel." + m)
            print("loaded:", [m for m in BLOCKED if m in sys.modules])
        """)
        got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        self.assertEqual(got.returncode, 0, got.stderr[-2000:])
        self.assertIn("loaded: []", got.stdout)


class TestV14KeepsItsNames(unittest.TestCase):

    def test_the_shared_code_under_v14s_names(self):
        for mod, shared, names in (
                (sim14, feed, ("_i", "_f", "_scorer", "_margin", "is_fresh", "_timeouts_at", "link_prev", "play_class",
                               "snap_class", "kick_decisions", "safety_kicks", "conversion_phase", "half_left",
                               "SNAP_KINDS", "SNAP_CLASSES", "N_CLASSES", "GAIN", "TURNOVER", "DEF_TD", "QUARTER",
                               "PLAY_CLOCK", "OT_TIMEOUTS", "OFFENCE", "DEFENCE", "KICK_TOUCHBACK",
                               "KICK_NO_LANDING", "CONV_MARGIN")),
                (v14, markets, ("MARGIN_MAX", "TOTAL_MAX", "_distributions", "even_line", "key_line", "held_line",
                                "anchored_line", "market_prob", "prob_above", "prob_below", "pool_context")),
                (v14_stream, stream, ("_quote", "_line", "_before_play", "paired_prod_rows", "LINE_MODES", "OWN",
                                      "EVEN", "PROD_LINES", "ANCHORED", "HELD", "description", "_parse_line")),
                (prior_check, books, ("base_prior", "sigmas", "raw_books", "nb2_books", "table_books", "paired",
                                      "_start", "_finals", "RAW_DRAWS")),
                (strength_check, loading, ("history_rows", "window_codes", "load_snapshots"))):
            for n in names:
                with self.subTest(module=mod.__name__, name=n):
                    self.assertIs(getattr(mod, n), getattr(shared, n))

    def test_sim14s_snaps_are_the_feeds_with_its_tables_keys(self):
        rows = sorted(next(iter(_matches(n=1, seed=2).values())), key=lambda r: int(r["message"]))
        mine, theirs = sim14.snap_records(rows), feed.snap_records(rows)
        self.assertEqual(len(mine), len(theirs))
        for a, b in zip(mine, theirs):
            self.assertEqual({k: v for k, v in a.items() if k not in ("mode", "key")}, b)
            self.assertEqual(a["key"], sim14.key_index(a["mode"], a["down"], a["distance"], a["field"]))

    def test_sim14s_switches_still_reach_the_shared_readers(self):
        rows = [dict(match_code="M", message="1", period="2", clock_seconds="100", play_kind="CONVERSION",
                     offense="TEAM_A", down="", distance="", field_position="35", play_messages="", score_p1="7",
                     score_p2="0", team_a_side="home"),
                dict(match_code="M", message="2", period="2", clock_seconds="90", play_kind="TOUCHDOWN",
                     offense="TEAM_B", down="", distance="", field_position="100",
                     play_messages="KICKOFF_TEAM_A|TOUCHDOWN_TEAM_B", score_p1="7", score_p2="6", team_a_side="home")]
        self.assertEqual(sim14.kick_records(rows), feed.kick_records(rows))
        saved = sim14.KICK_TDS, sim14.CONV_CLOCK
        try:
            sim14.KICK_TDS, sim14.CONV_CLOCK = False, False
            self.assertEqual(sim14.kick_records(rows), [])                     # off: the touchdown row skipped
            self.assertEqual(sim14.conversion_cell(4, 20.0), 3)                # off: no clock cells
        finally:
            sim14.KICK_TDS, sim14.CONV_CLOCK = saved
        self.assertEqual(feed.kick_records(rows), [(False, False, 100, 10.0)])  # the feed's own default, unmoved
        self.assertEqual((sim14.conversion_cell(4, 20.0), feed.conversion_cell(4, 20.0)), (4, 4))


if __name__ == "__main__":
    unittest.main()
