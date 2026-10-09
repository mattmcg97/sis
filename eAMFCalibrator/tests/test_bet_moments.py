"""Tests for bet_moments: the feed's state at each bet and the margin bucketed by it."""

import datetime as dt
import unittest
from types import SimpleNamespace

from .. import bet_checks, bet_moments as bm

T0 = dt.datetime(2026, 9, 20, 20, 0, 0)
FEED = [  # message, file second, clock, status, in-play message, down
    (1, 0, 0, "NOT_STARTED", None, None),
    (2, 1, 240, "FIRST_QUARTER_STARTED", None, None),
    (3, 2, 240, "BET_SUSPEND", None, None),
    (4, 3, 240, None, "KICKOFF_TEAM_B", None),
    (5, 4, 240, "BET_UNSUSPEND", None, None),
    (6, 5, 240, None, "POSSESSION_TEAM_A", 1),
    (7, 10, 239, "BET_SUSPEND", None, None),
    (8, 10.1, 239, None, "PLAY_STARTED", None),
    (9, 16, 233, None, "PLAY_OVER", 2),
    (10, 16.1, 233, "BET_UNSUSPEND", None, None),
    (11, 40, 210, "BET_SUSPEND", None, None),
    (12, 40.1, 210, None, "PLAY_STARTED", None),
    (13, 44, 207, None, "TOUCHDOWN_TEAM_A", None),
    (14, 60, 207, None, "PLAY_OVER", None),
    (15, 60.1, 207, "BET_UNSUSPEND", None, None),
    (16, 66, 207, "BET_SUSPEND", None, None),
    (17, 66.1, 207, None, "PLAY_STARTED", None),
    (18, 66.2, 207, None, "EXTRA_POINT_GOOD_TEAM_A", None),
    (19, 66.3, 207, None, "PLAY_OVER", None),
    (20, 66.4, 207, "BET_UNSUSPEND", None, None),
    (21, 80, 207, "BET_SUSPEND", None, None),
    (22, 80.1, 207, None, "PLAY_STARTED", None),
    (23, 80.2, 207, None, "KICKOFF_TEAM_A", None),
    (24, 87, 206, None, "POSSESSION_TEAM_B", 1),
    (25, 95, 203, None, "PLAY_OVER", 1),
    (26, 95.1, 203, "BET_UNSUSPEND", None, None),
    (27, 100, 200, None, "TIMEOUT_CALLED_TEAM_B", None),
    (28, 130, 200, None, "TIMEOUT_OVER_TEAM_B", None),
    (29, 200, 0, "FIRST_QUARTER_ENDED", None, None),
]
PROD_LAG = 2.0
PROD_MESSAGES = (9, 13, 14, 19, 25)


def _rows():
    return [("M", m, clock, status, kind, None, down, None, None, T0 + dt.timedelta(seconds=s))
            for m, s, clock, status, kind, down in FEED]


def _at(second):
    """A time on prod's clock: the feed's file second plus prod's lag."""
    return T0 + dt.timedelta(seconds=second + PROD_LAG)


def _checks():
    sec = {m: s for m, s, *_ in FEED}
    times = {"M": (list(PROD_MESSAGES), [_at(sec[m]) for m in PROD_MESSAGES])}
    checks = bet_checks.Checks({}, times, {"M": ([13, 18], [(6, 0), (7, 0)])})
    checks.timelines = bm.build(_rows(), times)
    return checks


class TestStates(unittest.TestCase):
    def test_every_part_of_a_play_and_the_breaks_get_their_moment(self):
        st = {m: s for (m, *_), s in zip(FEED, bm._states(_rows()))}
        want = {1: bm.BEFORE, 6: bm.BETWEEN, 8: bm.LIVE, 9: bm.BETWEEN, 13: bm.SCORED_LIVE,
                14: bm.CONV_NEXT, 17: bm.CONV_LIVE, 19: bm.KICK_NEXT, 22: bm.KICK_LIVE,
                25: bm.BETWEEN, 27: bm.TIMEOUT, 28: bm.BETWEEN, 29: bm.BREAK}
        self.assertEqual({m: st[m][0] for m in want}, want)
        self.assertTrue(st[8][1] and st[9][1] and not st[10][1])
        self.assertEqual((st[9][2], st[9][3], st[9][4]), (1, 233, 2))


class TestMoments(unittest.TestCase):
    def setUp(self):
        self.checks = _checks()
        self.tl = self.checks.timelines["M"]

    def test_feed_times_are_put_on_prods_clock(self):
        self.assertAlmostEqual(self.tl.times[8] - self.tl.times[0], 16.0)
        self.assertAlmostEqual(self.tl.times[0], _at(0).timestamp())

    def test_a_bet_mid_play_on_the_last_play_over_price(self):
        out = bm.moment_of(self.tl, self.checks, "M", _at(42), 9)
        self.assertEqual(out["moment"], bm.LIVE)
        self.assertEqual(out["feed_suspended"], "suspended")
        self.assertEqual(out["moved"], "play started")
        self.assertAlmostEqual(out["price_age_seconds"], 26.0)
        self.assertAlmostEqual(out["next_score_seconds"], 2.0)
        self.assertEqual(out["next_score"], "<10s")
        self.assertEqual((out["quarter"], out["clock_band"], out["score_margin"]),
                         ("Q1", "Q1 before 2:00", "level (0-2)"))

    def test_a_bet_after_the_touchdown_on_the_price_before_it(self):
        out = bm.moment_of(self.tl, self.checks, "M", _at(62), 9)
        self.assertEqual(out["moment"], bm.CONV_NEXT)
        self.assertEqual(out["moved"], "score")
        self.assertEqual(out["score_margin"], "1 score (3-8)")
        self.assertEqual(out["next_score"], "<10s")

    def test_a_current_price_moved_nothing_and_a_later_one_is_flagged(self):
        self.assertEqual(bm.moment_of(self.tl, self.checks, "M", _at(20), 9)["moved"], "nothing")
        self.assertEqual(bm.moment_of(self.tl, self.checks, "M", _at(20), 25)["moved"],
                         "price after the bet")
        self.assertEqual(bm.moment_of(self.tl, self.checks, "M", _at(150), 25)["next_score"],
                         "none after")

    def test_pre_match_bets_get_every_field(self):
        rows = [dict(message=None)]
        bm.annotate([("prod", rows)], [SimpleNamespace(in_play=False, match_code="M", time=None)],
                    self.checks, self.checks.timelines)
        self.assertEqual(rows[0]["moment"], "pre-match")
        self.assertIn("next_score", rows[0])


def _row(moment, revenue, stake=10.0, odds=2.0, prob=0.45, **kw):
    r = dict(in_play=True, excluded=False, result="lost" if revenue > 0 else "won", stake=stake,
             revenue=revenue, odds=odds, stream_prob=prob, moment=moment, simulated=True,
             candidate_revenue=revenue / 2, market="total", selection="over")
    r["candidate_result"] = r["result"]
    r["candidate_odds"] = odds
    r.update(kw)
    return r


class TestBuckets(unittest.TestCase):
    def rows(self):
        return ([_row(bm.LIVE, -10.0) for _ in range(150)] + [_row(bm.LIVE, 10.0) for _ in range(50)]
                + [_row(bm.BETWEEN, 10.0) for _ in range(110)] + [_row(bm.BETWEEN, -10.0) for _ in range(90)]
                + [_row(bm.LIVE, -5.0, in_play=False)])

    def test_margin_error_and_expected_margin_by_bucket(self):
        b = bm.buckets(self.rows(), "moment")
        n, stake, rev, m, se2, exp = b[bm.LIVE]
        self.assertEqual((n, stake, rev), (200, 2000.0, -1000.0))
        self.assertAlmostEqual(m, -50.0)
        self.assertAlmostEqual(exp, 10.0)
        self.assertGreater(se2, 0)
        self.assertAlmostEqual(b[bm.BETWEEN][3], 10.0)

    def test_the_report_puts_the_costliest_moment_first(self):
        rows = self.rows()
        for r in rows:
            r.update(moved="nothing", next_score="later")
        lines = bm.report([("prod", rows), ("v6", [dict(r) for r in rows])])
        text = "\n".join(lines)
        self.assertIn("by the feed at bet time", text)
        costly = text.split("the costliest moments")[1].splitlines()
        self.assertTrue(costly[2].strip().startswith(bm.LIVE))
        html = bm.html_tables([("prod", rows)], str, lambda c: f"<td>{c:+.2f}</td>")
        self.assertIn("<h3>In play <span class=\"dim\">400 bets</span></h3>", html)
        self.assertIn("<td class=\"bad\">-1,000</td><td class=\"bad\">-50.00%</td>", html)

    def test_a_written_csv_buckets_again_without_candidates(self):
        import os
        import tempfile
        from .. import bets
        rows = self.rows()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bets_sim.csv")
            bets.write_csv(path, rows)
            back = bm.from_csv(path)
        self.assertEqual(bm.buckets(back, "moment"), bm.buckets(rows, "moment"))
        text = "\n".join(bm.report([("prod", back)], candidates=False))
        self.assertIn("by the feed at bet time", text)

    def test_crossing_columns_puts_the_costliest_first(self):
        rows = self.rows()
        for r in rows:
            r["match_code"] = "AF001200926"
        lines = bm.cross(rows, ["moment", "day"])
        self.assertTrue(lines[2].strip().startswith(f"{bm.LIVE} / 2026-09-20"))
        self.assertIn("+60.00", lines[2])

    def test_bands_sort_by_their_seconds(self):
        bands = ["60s+", "10-20s", "<0s", "0-2s", "2-5s", "none after", "unknown"]
        self.assertEqual(sorted(bands, key=lambda g: bm._order("price_age", g)),
                         ["<0s", "0-2s", "2-5s", "10-20s", "60s+", "none after", "unknown"])


class TestCandidates(unittest.TestCase):
    def test_a_wide_csv_shows_each_candidates_change_by_bucket_and_line_gap(self):
        import os
        import tempfile
        from .. import bets
        rows = []
        for k in range(150):
            for sel, line_v6 in (("Over", 38.5), ("Under", 41.5)):
                r = _row(bm.BETWEEN, 10.0, selection=sel)
                r.update(bet_line_prod_side=40.5, simulated_v6=True, candidate_revenue_v6=4.0,
                         candidate_line_v6=line_v6, simulated_GAME=k % 2 == 0,
                         candidate_revenue_GAME=12.0)
                for key in ("simulated", "candidate_revenue", "candidate_result", "candidate_odds"):
                    r.pop(key, None)
                rows.append(r)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bets_sim.csv")
            bets.write_csv(path, rows)
            back = bm.from_csv(path)
        self.assertEqual(sorted(bm.candidate_names(back)), ["GAME", "v6"])
        lines = bm.cross(back, ["selection", "line_gap_v6"], min_bets=50)
        text = "\n".join(lines)
        self.assertIn("Over / -2", text)
        self.assertIn("Under / +1", text)
        row = next(l for l in lines if "Over / -2" in l)
        self.assertIn("-900 (   150)", row)          # v6 keeps 4 of every 10 the book kept
        self.assertIn("+150 (    75)", row)          # the other candidate, on the half it priced


class TestPlayers(unittest.TestCase):
    def rows(self):
        out = []
        for k in range(120):
            out.append(_row(bm.BETWEEN, -10.0, selection="Home", match_code="M1"))
            out.append(_row(bm.BETWEEN, 10.0, selection="Away", match_code="M1"))
            out.append(_row(bm.BETWEEN, 10.0, selection="Over", match_code="M2", in_play=False))
        return out

    def info(self):
        import os
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "history.csv")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("MATCH_CODE,PLAYER_1_HANDLE,PLAYER_1_TEAM,PLAYER_2_HANDLE,PLAYER_2_TEAM\n"
                     "M1,ACE,Bills,ZED,Jets\nM2,ZED,Lions,ACE,Bears\n")
        return bm.match_info(path)

    def test_each_bet_knows_the_side_it_backed(self):
        rows = bm.add_players(self.rows(), self.info())
        self.assertEqual((rows[0]["backed_player"], rows[0]["opposed_team"], rows[0]["matchup"]),
                         ("ACE", "Jets", "ACE v ZED"))
        self.assertEqual((rows[1]["backed_player"], rows[2]["backed_player"]), ("ZED", "total"))

    def test_a_gamer_gets_every_bet_of_its_matches_once(self):
        rows = bm.add_players(self.rows(), self.info())
        lines = bm.cross(rows, ["gamer", "gamer_role"], min_bets=50, keep=bm.settled)
        text = "\n".join(lines)
        self.assertIn("ACE / backed", text)
        self.assertIn("ZED / total", text)
        self.assertTrue(lines[2].strip().startswith(("ACE / backed", "ZED / backed")))
        in_play = bm.cross(rows, ["gamer", "gamer_role"], min_bets=50)
        self.assertNotIn("total", "\n".join(in_play[2:]))

    def test_when_splits_pre_match_from_in_play(self):
        rows = self.rows()
        for r in rows:
            r["clock_band"] = "Q1 before 2:00"
        text = "\n".join(bm.cross(rows, ["when"], min_bets=50, keep=bm.settled))
        self.assertIn("pre-match", text)
        self.assertIn("Q1 before 2:00", text)


if __name__ == "__main__":
    unittest.main()
