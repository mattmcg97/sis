"""v2.1: the draws (backoff groups, tilts, the censored time), the real snaps' features, the build in
both modes, the sim's bounds (quarter end, two-minute warning, end zones, overtime) and the report."""

import bisect
import contextlib
import copy
import csv
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from .. import prior_check, v21, v21_report, v21_rules
from .fakes import _matches


def _history(matches):
    out = []
    for i, c in enumerate(sorted(matches)):
        rows = matches[c]
        out.append(dict(MATCH_CODE=c, PLAYER_1_HANDLE=f"G{i % 7}", PLAYER_2_HANDLE=f"G{(i + 3) % 7}",
                        PLAYER_1_TEAM=f"T{i % 4}", PLAYER_2_TEAM=f"T{(i + 1) % 4}",
                        PLAYER_1_FINAL_SCORE=rows[-1]["score_p1"], PLAYER_2_FINAL_SCORE=rows[-1]["score_p2"],
                        SCHEDULED_START_TIME_UTC=f"2026-10-01 {i % 24:02d}:00:00"))
    return out


MATCHES = _matches(n=80, seed=3)
HISTORY = _history(MATCHES)
_BUILT = {}


def built(mode="full"):
    if mode not in _BUILT:
        _BUILT[mode] = v21.build(MATCHES, HISTORY, mode=mode, min_records=20, verbose=False, refine=False)
    return copy.deepcopy(_BUILT[mode])


def _row(msg, period, clock, kind, off, down, dist, y, a=0, b=0, messages=""):
    return dict(match_code="AF900", message=str(msg), period=str(period), clock_seconds=str(clock), play_kind=kind,
                offense=off, down=str(down), distance=str(dist), field_position=str(y), score_p1=str(a),
                score_p2=str(b), play_messages=messages, team_a_side="home", opening_offense="TEAM_A")


def _state(model, period=1, clock=240.0, team=0, down=1, dist=10, y=25, prev=v21.FRESH, home=0, away=0, gamers=None):
    return dict(period=np.array([period]), clock=np.array([clock]), team=np.array([team]),
                phase=np.array([v21.SNAP]), down=np.array([down]), dist=np.array([dist]), y=np.array([y]),
                home=np.array([home]), away=np.array([away]), prev=np.array([prev]), elapsed=np.array([0]),
                tos=np.array([[3, 3]]), kick2=np.array([1]),
                gamer=np.array([list(gamers) if gamers else [len(model.gamers)] * 2]),
                team_ix=np.array([[len(model.teams)] * 2]))


def _always(name, option, n_who):
    """A decision that always takes this option."""
    K = len(v21.DECISIONS[name]["features"])
    return v21.Choice.build(name, np.zeros((50, K), np.int64), np.full(50, option), np.full(50, n_who - 1), n_who)


def _fixed_time(model, seconds):
    """Every time draw the same."""
    for b in [model.time] + [v for k, v in model.own.items() if k[2] == "time"]:
        b.payload["seconds"][:] = seconds
        b.payload["cut"][:] = False


def _fixed_play(model, gain, td, made=3, rdist=0):
    """Every play the same; made 3: the down by the yards (0 short, 1 a new set, 2 played again: at
    the real snap's own distance rdist)."""
    for b in [model.dist] + [v for k, v in model.own.items() if k[2] == "dist"]:
        b.payload["kind"][:] = v21.GAIN
        b.payload["gain"][:] = gain
        b.payload["td"][:] = td
        b.payload["made"][:] = made
        b.payload["rdist"][:] = rdist


class TestBins(unittest.TestCase):

    def setUp(self):
        # feature a in 0..2, b in 0..1; (0, 0) has 50 records, (1, 0) 5, the rest none
        X = np.array([[0, 0]] * 50 + [[1, 0]] * 5)
        self.value = np.r_[np.arange(50)[::-1], [100, 101, 102, 103, 104]].astype(float)
        self.bins = v21.Bins.build(("a", "b"), (3, 2), (("a", "b"), ("a",), ()), X, self.value,
                                   {"v": self.value}, min_records=10)

    def test_a_full_group_draws_alone_and_the_rest_fall_back(self):
        b = self.bins.bin_of({"a": np.array([0, 1, 2]), "b": np.array([0, 0, 1])})
        self.assertEqual(self.bins.level_of[b[0]], 0)
        self.assertEqual(self.bins.count[b[0]], 50)
        self.assertEqual(self.bins.level_of[b[1]], 2)          # 5 records: under 10, to the whole pool
        self.assertEqual(self.bins.count[b[1]], 55)
        self.assertEqual(b[1], b[2])

    def test_groups_run_bottom_to_top_and_a_tilt_leans_up(self):
        b = self.bins.bin_of({"a": np.array([0, 0]), "b": np.array([0, 0])})
        v = self.bins.payload["v"]
        self.assertEqual(v[self.bins.draw(b[:1], np.array([0.0]))][0], 0)
        self.assertEqual(v[self.bins.draw(b[:1], np.array([0.999]))][0], 49)
        u = np.random.default_rng(0).random(4000)
        plain = v[self.bins.draw(np.repeat(b[0], 4000), u)].mean()
        up = v[self.bins.draw(np.repeat(b[0], 4000), u, np.full(4000, 2.0))].mean()
        self.assertGreater(up, plain + 5)

    def test_above_stays_in_the_group_and_at_or_over_the_record(self):
        b = self.bins.bin_of({"a": np.array([0]), "b": np.array([0])})[0]
        j = np.full(1000, self.bins.start[b] + 30)
        got = self.bins.above(j, np.random.default_rng(1).random(1000))
        self.assertTrue((got >= j).all())
        self.assertTrue((got < self.bins.start[b] + self.bins.count[b]).all())

    def test_no_tilt_gives_the_plain_mean(self):
        E = self.bins.tilted_means(self.bins.payload["v"], np.array([0.0, 0.5]))
        for g in range(len(self.bins.count)):
            s, n = self.bins.start[g], self.bins.count[g]
            self.assertAlmostEqual(E[g, 0], self.bins.payload["v"][s:s + n].mean())
            self.assertGreater(E[g, 1], E[g, 0])

    def test_a_gamer_s_own_draw_keeps_the_down(self):
        # gamer 0: 300 1st downs, 5 3rd downs of their own -- a 3rd down is the league's, not their 1st downs
        rows = [(0, 3, 0, 4, 2)] * 300 + [(0, 3, 2, 1, 2)] * 5
        f = {k: np.array([r[i] for r in rows]) for i, k in enumerate(("who", "tb", "down", "dist", "zone"))}
        f = v21.derive(dict(f, endgame=np.zeros(len(rows), np.int64)))
        X = np.stack([f[k] for k in v21.GAMER_DIST_FEATURES], axis=1)
        sizes = [3] + [v21.SIZES[k] for k in v21.GAMER_DIST_FEATURES[1:]]
        b = v21.Bins.build(v21.GAMER_DIST_FEATURES, sizes, v21.GAMER_DIST_LEVELS, X, np.zeros(len(X)),
                           {"v": np.zeros(len(X))}, 30, strict=True)
        q = v21.derive(dict(who=np.array([0, 0]), tb=np.array([3, 3]), down=np.array([0, 2]), dist=np.array([4, 1]),
                            zone=np.array([2, 2]), endgame=np.zeros(2, np.int64)))
        got = b.bin_of(q)
        self.assertGreaterEqual(got[0], 0)
        self.assertEqual(got[1], -1)

    def test_the_end_of_a_half(self):
        self.assertEqual(list(v21.late_of([1, 2, 2, 2, 3, 4, 4, 5], [20, 121, 120, 30, 10, 31, 30, 200])),
                         [0, 0, 1, 2, 0, 1, 2, 3])

    def test_blended_levels_draw_each_cell_by_its_records(self):
        # a 50-record cell and its parent of 55: drawn from n / (n + prior) of the time, else the parent
        b = self.bins
        C = b.chain(dict(a=np.array([0, 2]), b=np.array([0, 1])))
        W = b.blend_weights(C, 50.0)
        np.testing.assert_allclose(W.sum(1), 1.0)
        fine = int(np.flatnonzero(C[0] >= 0)[0])
        self.assertAlmostEqual(W[0, fine], 50 / 100)
        got = b.bin_blend(dict(a=np.zeros(20000, np.int64), b=np.zeros(20000, np.int64)), np.random.default_rng(0), 50.0)
        self.assertAlmostEqual(float(np.mean(got == C[0, fine])), 0.5, delta=0.02)
        self.assertTrue((got >= 0).all())

    def test_a_draw_near_its_field(self):
        # one group of 400: fields 1-99 four times over, the value its field
        fld = np.repeat(np.arange(1, 101), 4)[:400]
        b = v21.Bins.build(("a",), (1,), ((),), np.zeros((400, 1)), fld.astype(float),
                           {"field": fld, "v": fld.astype(float)}, 1)
        g = np.zeros(3000, np.int64)
        u = np.random.default_rng(0).random(3000)
        for y in (20, 60, 95):
            got = b.payload["field"][b.draw_near(g, np.full(3000, y), u, 0.2)]
            r = max(v21.FIELD_YARDS, round(0.2 * (100 - y)))
            r = max(r, (v21.FIELD_MIN // 4 + 1) // 2)              # 4 a yard: a thin window widens
            self.assertTrue((np.abs(got - y) <= r).all())
            self.assertAlmostEqual(got.mean(), y, delta=1.0)
        # a thin window widens the same either side; at the goal line it stays on the yard itself
        lo, hi = b.near(np.array([0]), np.array([95]), 0.0)
        got = b.payload["field"][b.by_field()[0][lo[0]:hi[0]]]
        self.assertGreaterEqual(int(hi[0] - lo[0]), v21.FIELD_MIN)
        self.assertEqual(float(got.mean()), 95.0)
        lo, hi = b.near(np.array([0]), np.array([99]), 0.0)
        self.assertEqual(set(b.payload["field"][b.by_field()[0][lo[0]:hi[0]]].tolist()), {99})
        # a group of 5 draws from all of it; a tilt still leans up
        small = v21.Bins.build(("a",), (1,), ((),), np.zeros((5, 1)), np.arange(5.0),
                               {"field": np.array([10, 30, 50, 70, 90]), "v": np.arange(5.0)}, 1)
        lo, hi = small.near(np.array([0]), np.array([50]), 0.2)
        self.assertEqual((int(lo[0]), int(hi[0])), (0, 5))
        # a tilt leans toward the top of the group by value (the yards), the field held
        rng = np.random.default_rng(1)
        val = rng.permutation(400).astype(float)
        mixed = v21.Bins.build(("a",), (1,), ((),), np.zeros((400, 1)), val, {"field": fld, "v": val}, 1)
        plain = mixed.payload["v"][mixed.draw_near(g, np.full(3000, 50), u, 0.2)].mean()
        up = mixed.payload["v"][mixed.draw_near(g, np.full(3000, 50), u, 0.2, np.full(3000, 3.0), rng)].mean()
        self.assertGreater(up, plain + 40)

    def test_strict_has_no_fallback(self):
        X = np.array([[0, 0]] * 50 + [[1, 0]] * 5)
        b = v21.Bins.build(("a", "b"), (3, 2), (("a", "b"),), X, self.value, {"v": self.value}, 10, strict=True)
        got = b.bin_of({"a": np.array([0, 1]), "b": np.array([0, 0])})
        self.assertGreaterEqual(got[0], 0)
        self.assertEqual(got[1], -1)


class TestRealSnaps(unittest.TestCase):

    def test_prev_remain_and_the_quarter_end(self):
        rows = MATCHES["AF000"]
        recs = v21.match_snaps(rows)
        self.assertEqual(recs[0]["prev_cls"], v21.FRESH)                   # after the opening kick-off
        firsts = [r for r in recs if r["period"] <= 2]
        self.assertEqual(firsts[-1]["remain"], 0)
        self.assertEqual(firsts[0]["remain"], len(firsts) - 1)
        self.assertTrue(any(r["cut"] for r in recs))                       # the quarter ran out on some
        for n, r in enumerate(recs):
            if r["clock"] - r["seconds"] <= 0.5:
                self.assertTrue(r["cut"])
            elif r["cut"]:                                                 # else only a running clock over 2:00
                self.assertIn(r["period"], (2, 4))
                self.assertGreater(r["clock"], v21.LATE)
            self.assertEqual(r["elapsed"], n)

    def test_a_running_clock_across_two_minutes_is_cut(self):
        rows = [_row(7, 2, 160, "SCRIMMAGE", "TEAM_A", 1, 10, 30), _row(14, 2, 135, "SCRIMMAGE", "TEAM_A", 2, 6, 34),
                _row(21, 2, 113, "SCRIMMAGE", "TEAM_A", 1, 10, 45), _row(28, 2, 106, "SCRIMMAGE", "TEAM_A", 2, 10, 45)]
        recs = v21.match_snaps(rows)
        self.assertEqual(recs[1]["prev_cls"], v21.SHORT)
        self.assertTrue(recs[1]["cut"])                                    # 2:15 to 1:53 on a running clock
        self.assertEqual(recs[1]["play_seconds"], 7.0)                     # its play: from 2:00
        self.assertFalse(recs[2]["cut"])                                   # 1:53 on, nothing crossed

    def test_starts_read_the_action_log(self):
        rows = MATCHES["AF001"]
        model = built()
        starts = v21.match_starts(model, rows)
        recs = v21.match_snaps(rows)
        self.assertTrue(starts)
        for i, s in starts:
            msg = int(rows[i]["message"])
            self.assertEqual(int(s["elapsed"][0]), sum(r["start"] < msg for r in recs))
            if rows[i]["play_kind"] == "KICKOFF":
                self.assertEqual(int(s["prev"][0]), v21.FRESH)
            self.assertGreater(float(s["clock"][0]), 0)
        self.assertEqual(len(v21.match_starts(model, rows, every=2)), (len(starts) + 1) // 2)

    def test_the_next_row_says_whether_the_chains_moved(self):
        rows = [_row(7, 1, 200, "SCRIMMAGE", "TEAM_A", 3, 1, 50),       # 3rd and inches: no yards, a 1st down
                _row(14, 1, 170, "SCRIMMAGE", "TEAM_A", 1, 10, 50),      # 10 yards, spotted short: 2nd down
                _row(21, 1, 140, "SCRIMMAGE", "TEAM_A", 2, 1, 60),       # a penalty: 2nd down again, 5 back
                _row(28, 1, 120, "SCRIMMAGE", "TEAM_A", 2, 6, 55),       # 4 yards: 3rd and 2
                _row(35, 1, 95, "SCRIMMAGE", "TEAM_A", 3, 2, 59),        # 1 yard: 4th and 1
                _row(42, 1, 70, "SCRIMMAGE", "TEAM_A", 4, 1, 60),        # 4th and inches: kept, a 1st down
                _row(49, 1, 45, "SCRIMMAGE", "TEAM_A", 1, 10, 60)]
        recs = v21.match_snaps(rows)
        got = [(r["down"], r["gain"], r["converted"], r["replay"]) for r in recs]
        self.assertEqual(got, [(3, 0, True, False), (1, 10, False, False), (2, -5, False, True),
                               (2, 4, False, False), (3, 1, False, False), (4, 0, True, False)])
        lines = "\n".join(v21.chains_lines(recs))
        self.assertIn("16.7% the same down played again", lines)
        self.assertIn("short of the yards yet moved the chains 2", lines)

    def test_points_share_runs_from_all_to_none(self):
        share = v21.points_share(MATCHES)
        self.assertAlmostEqual(share[0], 1.0, places=6)
        self.assertTrue((np.diff(share[:-1]) <= 1e-12).all())
        self.assertLess(share[-2], 0.2)


class TestBuild(unittest.TestCase):

    def test_full_keeps_every_side_s_own_draws(self):
        m = built("full")
        self.assertEqual(len(m.gamers), 7)
        self.assertEqual(len(m.teams), 4)
        for kind in ("gamer", "team"):
            for role in ("off", "def"):
                for draw in ("time", "dist"):
                    self.assertIn((kind, role, draw), m.own)
        self.assertEqual(m.tilt, {})
        self.assertTrue((m.remain[0, :-1] >= 0).all())

    def test_adjusted_fits_a_tilt_for_each_and_none_for_the_unknown(self):
        m = built("adjusted")
        self.assertEqual(m.own, {})
        for draw in ("time", "dist"):
            self.assertEqual(set(m.tilt[draw]), set(v21.EFFECTS))
            for v in m.tilt[draw].values():
                self.assertEqual(v[-1], 0.0)
                self.assertTrue((np.abs(v) <= 1.5).all())

    def test_save_and_load(self):
        m = built()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, v21.MODEL_FILE)
            m.save(p)
            back = v21.Model.load(p)
        self.assertEqual(back.gamers, m.gamers)
        np.testing.assert_array_equal(back.time.payload["seconds"], m.time.payload["seconds"])

    def test_refine_moves_the_remaining_table_to_the_sim_s_own(self):
        m = built()
        before = m.remain.copy()
        v21.refine_remaining(m, n_paths=400)
        self.assertEqual(m.remain.shape, before.shape)
        self.assertFalse(np.allclose(m.remain, before))

    def test_the_snaps_left_table_shrinks_toward_the_slice(self):
        cell = lambda h, sl, run, own, opp, l3: tuple(np.array([v]) for v in (h, sl, run, own, opp, l3))
        keys = [(1, 4, 0, 3, 3, 1)] * 200 + [(1, 4, 1, 0, 0, 1)] * 200 + [(1, 8, 0, 3, 3, 1)] * 100
        counts = np.r_[np.full(200, 6.0), np.full(200, 2.0), np.full(100, 10.0)]
        key = tuple(np.array(c) for c in zip(*keys))
        t = v21.remain_table(key, counts, prior=20.0)
        self.assertEqual(t.shape, v21.REMAIN_SHAPE)
        self.assertAlmostEqual(float(t[cell(1, 4, 0, 3, 3, 1)][0]), 6.0, delta=0.05)     # its own, many
        self.assertAlmostEqual(float(t[cell(1, 4, 1, 0, 0, 1)][0]), 2.0, delta=0.05)
        # none: its slice and clock's (6.0 over 200), shrunk toward the slice's (4.0 over 400)
        self.assertAlmostEqual(float(t[cell(1, 4, 0, 0, 0, 1)][0]), (200 * 6 + 20 * 4) / 220, places=6)
        self.assertAlmostEqual(float(t[cell(1, 6, 0, 3, 3, 1)][0]), 7.0, delta=0.05)     # no slice: between 4's and 8's
        m = built()
        self.assertEqual(m.remain.shape, v21.REMAIN_SHAPE)
        late_stopped = m.remain_at(4, 40.0, 0, 3, 3, 1)
        self.assertGreater(late_stopped, m.remain_at(4, 40.0, 1, 0, 0, 1) - 1e-9)
        self.assertLess(late_stopped, m.remain_at(4, 600.0, 0, 3, 3, 1))
        self.assertEqual(len(v21.remain_lines(m)), len(v21.REMAIN_SHOWN) + 1)

    def test_the_end_of_a_half_stands_apart_in_the_own_draws(self):
        # phase: 0 Q1, 2 Q2's last 2:00, 6 OT; remain: the snaps-left bucket (0-1, 2, 3-4, 5-7, 8-12, 13+);
        # margin: the lead's bucket (3 level, 4 1-3 ahead, 6 9+ ahead)
        # the second half's end apart from the first's (5: Q4's last 2:00)
        f = v21.derive(dict(phase=np.array([0, 0, 0, 2, 2, 2, 6, 0, 2, 5]),
                            remain=np.array([5, 2, 0, 5, 1, 0, 4, 3, 1, 1]),
                            margin=np.array([3, 3, 0, 6, 3, 3, 0, 6, 4, 3])))
        self.assertEqual(list(f["endgame"]), [0, 1 + 7 + 3, 1 + 0, 1 + 14 + 6, 1 + 7 + 3, 1 + 3, 43, 0, 1 + 7 + 4,
                                              1 + 21 + 7 + 3])
        self.assertEqual(list(f["endgame3"]), [0, 1 + 3 + 1, 1 + 0, 1 + 6 + 2, 1 + 3 + 1, 1 + 1, 19, 0, 1 + 3 + 2,
                                               1 + 9 + 3 + 1])
        self.assertEqual(list(f["part"]), [0, 0, 0, 1, 1, 1, 4, 0, 1, 3])
        self.assertLess(int(v21.endgame_of(6, 0, 6)), v21.SIZES["endgame"])
        self.assertLess(int(v21.endgame3_of(6, 0, 2)), v21.SIZES["endgame3"])
        self.assertEqual(list(v21.remain_b([1, 2, 4, 7, 12, 17, 18])), list(range(7)))
        for levels in (v21.GAMER_TIME_LEVELS, v21.GAMER_DIST_LEVELS):
            for lv in levels:
                self.assertIn("endgame", lv)
                self.assertIn("down", lv)
        self.assertEqual(list(v21.remain_b([0, 1, 2, 3, 4, 5, 7, 8, 12, 13])), [0, 0, 1, 2, 2, 3, 3, 4, 4, 5])

    def test_the_split_reads_the_half_or_the_seven_parts(self):
        snaps, time_ = v21.draw_spec("snaps"), v21.draw_spec("time")
        for k in snaps:
            self.assertNotIn("phase", snaps[k][0])
            self.assertEqual(tuple(x.replace("part", "phase") for x in snaps[k][0]), time_[k][0])
        m = built()
        self.assertEqual(m.split, "snaps")
        self.assertIn("part", m.time.features)
        old = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, split="time")
        self.assertIn("phase", old.time.features)
        h, a = v21.simulate(old, v21.kickoff_start(old), 20)
        self.assertEqual(h.shape, (2, 20))
        with self.assertRaises(ValueError):
            v21.build(MATCHES, HISTORY, verbose=False, split="other")

    def test_the_build_sets_late_q4_against_its_draw_groups(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            v21.build(MATCHES, HISTORY, min_records=20, verbose=True, refine=False)
        text = out.getvalue()
        self.assertIn("Q4's last 2:00 by the lead", text)
        self.assertIn("the end of a half", text)
        m = built()
        rates = v21.group_rates(m.dist, m.dist.payload["td"])
        self.assertEqual(len(rates), len(m.dist.count))
        self.assertTrue(((rates >= 0) & (rates <= 1)).all())

    def test_the_late_choices_by_fortnight(self):
        q4 = lambda code, lead, choice: dict(code=code, period=4, clock=60.0, lead=lead, choice=choice)
        dec = dict(fourth=[q4("AF001010926", -3, v21.FG), q4("AF001300926", -3, v21.GO), q4("AF002300926", 0, v21.FG)],
                   two=[dict(code="AF001300926", cell=4, lead=-1, choice=1)])
        lines = v21.trend_lines(dec, {"AF001010926": True, "AF001300926": False, "AF002300926": False})
        self.assertEqual(len(lines), 3)                       # the header, the latest fortnight, an older one
        self.assertIn("go 100.0% (1), kick 100.0% (1), two at -1 100.0% (1)", lines[-1])
        self.assertIn("overtime 100.0% (1)", lines[1])

    def test_a_blended_build(self):
        m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, blend=40.0)
        self.assertEqual(m.blend, 40.0)
        self.assertLessEqual(int(m.dist.count[m.dist.level_of < len(m.dist.levels) - 1].min()), 20)   # thin cells kept
        h, a = v21.simulate(m, v21.kickoff_start(m), 30)
        self.assertGreater((h + a).mean(), 10)
        from .. import v21_coverage as C
        with tempfile.TemporaryDirectory() as d:
            m.save(os.path.join(d, v21.MODEL_FILE))
            out = C.run(d, {c: MATCHES[c] for c in sorted(MATCHES)[:5]}, HISTORY)
        w = np.array([[x[f"dist_{k.replace(' ', '_')}"] for k in C.KINDS] for x in out["states"]])
        np.testing.assert_allclose(w.sum(1), 1.0, atol=1e-3)

    def test_the_lean_key(self):
        m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, key="lean")
        self.assertEqual(m.key, "lean")
        self.assertEqual(m.time.levels[0], v21.TIME_LEAN)
        self.assertEqual(m.dist.levels[0], v21.DIST_LEAN)
        for lv in m.time.levels[:-1]:
            self.assertIn("clock2", lv)                         # the clock into the snap, to the end
        h, a = v21.simulate(m, v21.kickoff_start(m), 30)
        self.assertEqual(h.shape, (2, 30))
        self.assertGreater((h + a).mean(), 10)
        f = v21.derive(dict(prev=np.array([v21.SHORT, v21.NOTHING, v21.TIMEOUT]),
                            phase=np.array([0, 2, 6]), remain=np.array([6, 1, 3])))
        self.assertEqual(list(f["clock2"]), [1, 0, 0])
        self.assertEqual(list(f["qtr"]), [0, 1, 4])
        self.assertEqual(list(f["rem4"]), [3, 1, 2])
        with self.assertRaises(ValueError):
            v21.build(MATCHES, HISTORY, verbose=False, key="other")

    def test_the_lean_key_keeps_the_lead_to_the_last(self):
        # all seven of the lead in every fallback that keeps the snaps left, before behind / level / ahead
        for levels in (v21.TIME_LEAN_LEVELS, v21.DIST_LEAN_LEVELS):
            seen3 = False
            for lv in levels:
                if "lead3" in lv:
                    seen3 = True
                if "margin" in lv:
                    self.assertFalse(seen3, lv)
            first3 = next(i for i, lv in enumerate(levels) if "lead3" in lv)
            self.assertTrue(all("zone" not in lv for lv in levels[first3 - 1:first3 + 1]))

    def test_the_lean_draws_keep_the_distance_to_the_end(self):
        # 2nd and 1 is played from 2nd and 1's snaps, its time too: the distance draw reads the time it was
        # given at the state's own distance, so neither draw pools the distances before the last two
        for levels in (v21.TIME_LEAN_LEVELS, v21.DIST_LEAN_LEVELS):
            for lv in levels:
                self.assertNotIn("dist3", lv)
            for lv in levels:
                if "dist" not in lv:
                    self.assertLessEqual(len(lv), 2, lv)
            self.assertTrue(any("phase" in lv and "dist" in lv and "margin" not in lv and "lead3" not in lv
                                for lv in levels))           # the period holds past the lead

    def test_the_distance_draws_never_let_the_down_go(self):
        # a blended draw passes weight down the chain: a level without the down would draw a 4th down
        # from every down's snaps
        for levels in (v21.DIST_LEVELS, v21.DIST_LEAN_LEVELS):
            for lv in levels[:-1]:
                self.assertIn("down", lv)
                if "tb" not in lv:                         # the time the play took, all six buckets, or
                    self.assertIn("tb2", lv)               # stopped / running where the lead and period hold
                    self.assertTrue({"phase", "lead3"} <= set(lv), lv)

    def test_a_blended_cell_with_none_near_the_field_passes_its_draws_on(self):
        # cell a=1: 27 snaps at the goal line, 3 at midfield; 400 others across midfield. A state in a=1
        # at the 70 has none of its cell near it: its draws all come from (); at midfield it weighs by all 30
        X = np.r_[np.ones(30, np.int64), np.zeros(400, np.int64)][:, None]
        rng = np.random.default_rng(0)
        field = np.r_[np.full(27, 95), np.full(3, 50), rng.integers(40, 61, 400)]
        b = v21.Bins.build(("a",), (2,), (("a",), ()), X, field.astype(float), {"field": field}, 5)
        C = b.chain(dict(a=np.array([1, 1, 1])))
        y = np.array([50, 95, 70])
        W_all = b.blend_weights(C, 40.0)
        W = b.blend_weights(C, 40.0, v21.near_weight_counts(b, C, y, 0.2))
        np.testing.assert_allclose(W_all[:, 0], 30 / 70)
        np.testing.assert_allclose(W[:, 0], [30 / 70, 30 / 70, 0.0])       # by all 30, or none near: none
        np.testing.assert_allclose(W.sum(1), 1.0)
        g, li = v21.Bins.pick(np.repeat(C[2:], 200, 0), np.repeat(W[2:], 200, 0), rng)
        self.assertTrue((li == 1).all())
        lo, hi = b.span(np.array([C[0, 0]]), np.array([50]), np.array([10]), "field")
        self.assertEqual(int((hi - lo)[0]), 3)                            # its 3 at midfield, not widened
        lo, hi = b.window(np.array([C[0, 0]]), np.array([50]), np.array([10]), "field", np.array([1]))
        self.assertEqual(int((hi - lo)[0]), 3)
        lo, hi = b.window(np.array([C[0, 0]]), np.array([50]), np.array([10]), "field")
        self.assertGreaterEqual(int((hi - lo)[0]), 20)             # widened, the old way, to the goal line

    def test_field_windows(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            m = v21.build(MATCHES, HISTORY, min_records=20, verbose=True, refine=False, mode="adjusted",
                          key="lean", blend=40.0, field=0.2)
        self.assertEqual(m.field, 0.2)
        self.assertEqual(m.meta["field"], 0.2)
        self.assertTrue(m.near_blend and m.meta["near_blend"])     # blended with windows: weighed near the field
        text = out.getvalue()
        self.assertIn("every snap by the field", text)
        self.assertIn("the sim draws from near it", text)
        self.assertIn("opp 19-11", text)
        st = {}
        h, a = v21.simulate(m, v21.kickoff_start(m), 30, stats=st)
        self.assertGreater((h + a).mean(), 10)
        last = st["drive_last"]                                       # the start's own drive at its end
        self.assertEqual(last.shape, (h.size, 4))
        done = st["drive_end"] >= 0
        self.assertTrue(((last[done, 0] >= 0) & (last[done, 0] <= 100)).all())
        # the sim's touchdown chance from near the field: none where the group's snaps there never reach
        recs = []
        for code in sorted(MATCHES)[:20]:
            recs += v21.match_snaps(v21._with_side(MATCHES[code]))
        feats = v21._features(recs, m)
        td, grp, whole, near = v21._td_states(m, recs, feats, np.arange(len(recs)))
        self.assertEqual(len(near), len(recs))
        self.assertTrue(((near >= 0) & (near <= 1)).all())
        full = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, field=0.2)
        h, a = v21.simulate(full, v21.kickoff_start(full), 30)
        self.assertGreater((h + a).mean(), 10)

    def test_exponential_tilts_recover_each_side_s_lean(self):
        # three bins (0, 1, 2 yards / 10), a flat base; gamer 0 lands high, gamer 1 as the base
        rng = np.random.default_rng(0)
        n = 3000
        who = {"gamer_off": np.repeat([0, 1], n)}
        obs = np.r_[rng.choice(3, n, p=[0.2, 0.3, 0.5]), rng.choice(3, n)]
        H = np.full((2 * n, 3), 1 / 3)
        phi_b = np.array([[0.0], [1.0], [2.0]])
        theta, gain = v21.fit_exp_tilts(H, obs, phi_b, who, {"gamer_off": 3}, prior=50.0)
        t = theta["gamer_off"][:, 0]
        self.assertGreater(t[0], 0.4)
        self.assertLess(abs(t[1]), 0.08)
        self.assertEqual(t[2], 0.0)                                 # the unknown gamer: none
        self.assertGreater(gain, 0)
        # a draw reweighted by a positive theta lands higher
        fld = np.repeat(np.arange(1, 101), 4)[:400]
        val = rng.permutation(400).astype(float)
        b = v21.Bins.build(("a",), (1,), ((),), np.zeros((400, 1)), val,
                           {"field": fld, "v": val, "phi": (val / 100.0)[:, None]}, 1)
        g = np.zeros(4000, np.int64)
        u = rng.random(4000)
        plain = b.payload["v"][b.draw(g, u)].mean()
        up = b.payload["v"][b.draw_tilted(g, u, np.full((4000, 1), 1.0), rng)].mean()
        near = b.payload["v"][b.draw_near(g, np.full(4000, 50), u, 0.2, theta=np.full((4000, 1), 1.0), rng=rng)].mean()
        self.assertGreater(up, plain + 40)
        self.assertGreater(near, plain + 30)

    def test_the_tilt_mode(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            m = v21.build(MATCHES, HISTORY, min_records=20, verbose=True, refine=False, mode="tilt", key="lean",
                          blend=40.0, field=0.2, clock=0.2)
        self.assertIn("tilts (exp(theta . phi)", out.getvalue())
        self.assertEqual(m.etilt["dist"]["gamer_off"].shape, (len(m.gamers) + 1, len(v21.PHI_DIST)))
        self.assertEqual(m.etilt["time"]["team_def"].shape, (len(m.teams) + 1, len(v21.PHI_TIME)))
        h, a = v21.simulate(m, v21.kickoff_start(m), 40)
        self.assertGreater((h + a).mean(), 10)
        # a side whose gamer gains more: more points in the sim
        hi = copy.deepcopy(m)
        hi.etilt["dist"]["gamer_off"][:, 0] = 0.8
        start = v21.kickoff_start(hi)
        self.assertGreater(v21.simulate(hi, start, 200, seed=1)[0].mean(), v21.simulate(m, start, 200, seed=1)[0].mean())
        with tempfile.TemporaryDirectory() as d:
            m.save(os.path.join(d, v21.MODEL_FILE))
            back = v21.Model.load(os.path.join(d, v21.MODEL_FILE))
        self.assertEqual(back.mode, "tilt")

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            v21.build(MATCHES, HISTORY, mode="other", verbose=False)


class TestSimulate(unittest.TestCase):

    def test_kickoff_games_finish_near_the_real_pace(self):
        for mode in ("full", "adjusted"):
            m = built(mode)
            st = {}
            h, a = v21.simulate(m, v21.kickoff_start(m, ("G1", "G2"), ("T1", "T2")), 300, seed=4, stats=st)
            self.assertEqual(h.shape, (2, 300))
            self.assertEqual(st["unfinished"], 0)
            self.assertTrue((h >= 0).all() and (a >= 0).all())
            real = np.mean([len(v21.match_snaps(r)) for r in MATCHES.values()])
            self.assertLess(abs(st["snaps"] / st["games"] / real - 1), 0.15)
            h2, a2 = v21.simulate(m, v21.kickoff_start(m, ("G1", "G2"), ("T1", "T2")), 300, seed=4)
            np.testing.assert_array_equal(h, h2)

    def test_a_running_clock_runs_the_quarter_out_before_the_snap(self):
        m = built()
        _fixed_time(m, 35.0)
        st = {}
        v21.simulate(m, _state(m, period=1, clock=20.0, prev=v21.SHORT, y=40, down=2, dist=7), 50, stats=st)
        self.assertGreaterEqual(st["clock_ran_out"], 50)
        log = v21.snap_log(st)
        first = log["path"] == 0
        self.assertEqual(log["period"][first][0], 2)                       # no snap in Q1
        self.assertEqual(log["clock"][first][0], v21.QUARTER)
        self.assertEqual(log["field"][first][0], 40)                       # the same ball carried over
        self.assertEqual(log["prev"][first][0], v21.FRESH)

    def test_the_two_minute_warning_stops_a_running_clock(self):
        m = built()
        _fixed_time(m, 35.0)
        st = {}
        v21.simulate(m, _state(m, period=2, clock=130.0, prev=v21.SHORT, y=40), 50, stats=st)
        self.assertGreaterEqual(st["two_minute"], 50)                       # Q2's, and Q4's later
        log = v21.snap_log(st)
        first = np.flatnonzero(log["path"] == 0)[0]
        self.assertEqual(log["clock"][first], 130.0)
        self.assertEqual(log["seconds"][first], 130.0 - (v21.LATE - 35.0))  # stopped at 2:00, then the snap

    def test_after_the_two_minute_warning_the_snap_takes_a_stopped_clock_s_time(self):
        m = built()
        for b in [m.time] + [v for k, v in m.own.items() if k[2] == "time"]:
            b.payload["cut"][:] = False
            b.payload["seconds"][:] = 35.0
            for g, (s0, n) in enumerate(zip(b.start, b.count)):
                b.payload["seconds"][s0] = 6.0                             # each group's one stopped snap
            b.__dict__.pop("_low", None)
        st = {}
        v21.simulate(m, _state(m, period=2, clock=130.0, prev=v21.SHORT, y=40), 50, stats=st)
        log = v21.snap_log(st)
        first = np.flatnonzero(log["path"] == 0)[0]
        self.assertEqual(log["seconds"][first], 130.0 - (v21.LATE - 6.0))
        self.assertEqual(log["drawn"][first], 6.0)

    def test_a_real_touchdown_from_further_back_runs_on(self):
        m = built()
        _fixed_time(m, 7.0)
        _fixed_play(m, 30, True)
        runs = lambda S: np.tile(S, (v21.SIZES["endgame3"], 1))
        m.gain_surv_end = runs(np.ones(101))                               # a long gain always goes on
        st = {}
        v21.simulate(m, _state(m, y=20), 20, stats=st)
        log = v21.snap_log(st)
        self.assertEqual(log["gain"][np.flatnonzero(log["path"] == 0)[0]], 80)
        m.gain_surv_end = runs(np.r_[np.ones(31), np.zeros(70)])           # nothing passes 30 yards
        st = {}
        v21.simulate(m, _state(m, y=20), 20, stats=st)
        log = v21.snap_log(st)
        self.assertEqual(log["gain"][np.flatnonzero(log["path"] == 0)[0]], 30)
        # the end of a half reads its own: Q4's last 0:20, level, nothing passes 30 yards there
        m.gain_surv_end = runs(np.ones(101))
        late = v21.endgame3_of(5, v21.remain_b(m.remain_at(4, 20.0, 1, 3, 3, 1)), 1)
        m.gain_surv_end[late] = np.r_[np.ones(31), np.zeros(70)]
        st = {}
        v21.simulate(m, _state(m, period=4, clock=20.0, y=20, prev=v21.SHORT), 20, stats=st)
        log = v21.snap_log(st)
        self.assertEqual(log["gain"][np.flatnonzero(log["path"] == 0)[0]], 30)
        b = built()
        self.assertEqual(b.gain_surv_end.shape, (v21.SIZES["endgame3"], 101))
        self.assertTrue((np.diff(b.gain_surv_end, axis=1) <= 1e-12).all())

    def test_near_its_field_a_snap_keeps_its_own_result(self):
        m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, field=0.2)
        _fixed_time(m, 7.0)

        def every(gain, td, field):
            _fixed_play(m, gain, td)
            for b in [m.dist] + [v for k, v in m.own.items() if k[2] == "dist"]:
                b.payload["field"][:] = field
                b.__dict__.pop("_by_field", None)

        first = lambda st: v21.snap_log(st)["gain"][np.flatnonzero(v21.snap_log(st)["path"] == 0)[0]]
        # 30 yards from its own 40, short of its goal: at the other side's 20 it stops as far of the way
        every(30, False, 40)
        st = {}
        v21.simulate(m, _state(m, y=80), 5, stats=st)
        self.assertEqual(first(st), 30 * 20 // 60)
        # a touchdown from the 5, drawn at midfield: a touchdown here, with no run-on chance
        every(5, True, 95)
        st = {}
        v21.simulate(m, _state(m, y=50), 5, stats=st)
        self.assertEqual(first(st), 50)
        # at another distance, its own result against the line: a 3rd-and-1 sneak for a yard moves the
        # chains at 3rd and 3 too, a 3rd-and-3 throw for 2 does not at 3rd and 1
        log1 = lambda st: {k: v21.snap_log(st)[k][np.flatnonzero(v21.snap_log(st)["path"] == 0)[0]]
                           for k in ("gain", "conv")}
        _fixed_play(m, 1, False, made=1, rdist=1)
        st = {}
        v21.simulate(m, _state(m, down=3, dist=3, y=50), 5, stats=st)
        self.assertEqual((log1(st)["gain"], bool(log1(st)["conv"])), (3, True))
        _fixed_play(m, 2, False, made=0, rdist=3)
        st = {}
        v21.simulate(m, _state(m, down=3, dist=1, y=50), 5, stats=st)
        self.assertEqual((log1(st)["gain"], bool(log1(st)["conv"])), (0, False))
        # a gain short of this goal plays as drawn
        every(8, False, 40)
        st = {}
        v21.simulate(m, _state(m, y=60), 5, stats=st)
        self.assertEqual(first(st), 8)

    def test_a_half_s_end_draws_its_time_from_snaps_with_as_long_left(self):
        def run(clock):
            m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, clock=clock)
            for b in [m.time] + [v for k, v in m.own.items() if k[2] == "time"]:
                n = len(b.payload["seconds"])
                b.payload["left"][:] = np.where(np.arange(n) % 2 == 0, 20, 45)      # half at 0:20, half at 0:45
                b.payload["seconds"][:] = np.where(b.payload["left"] == 20, 10.0, 35.0)
                b.payload["cut"][:] = False
                b.__dict__.pop("_by_left", None)
            st = {}
            v21.simulate(m, _state(m, period=4, clock=20.0, prev=v21.SHORT, y=40), 200, stats=st, max_steps=1)
            log = v21.snap_log(st)
            return log["seconds"][np.isin(log["path"], np.arange(200))], st.get("clock_ran_out", 0)
        secs, gone = run(0.2)
        self.assertTrue((secs == 10.0).all())
        self.assertEqual(gone, 0)
        secs, gone = run(0.0)
        self.assertGreater(gone, 40)                     # 35 seconds drawn at 0:20: the half runs out first

    def test_inches_move_the_chains_and_a_down_played_again_keeps_its_number(self):
        m = built()
        _fixed_time(m, 7.0)
        nxt = lambda log: (int(log["down"][1]), bool(log["conv"][0]))
        cases = ((1, 1, (1, True)),          # the real snap at 1 to go moved the chains for no yards
                 (1, 5, (3, False)),         # one at 5 to go: by the yards, 0 short of 1
                 (3, 0, (3, False)),         # by the yards
                 (2, 1, (2, False)))         # a down played again: still 2nd
        for made, rdist, want in cases:
            _fixed_play(m, 0, False, made=made, rdist=rdist)
            st = {}
            v21.simulate(m, _state(m, down=2, dist=1, y=50, prev=v21.SHORT), 1, stats=st, max_steps=3)
            log = v21.snap_log(st)
            self.assertEqual(nxt(log), want, (made, rdist))

    def test_a_side_in_range_calls_a_timeout_before_the_half_runs_out(self):
        # Q4 0:20 level, the clock running, at the other side's 20 with a timeout: every running snap takes
        # 30 seconds and every stopped one 5, every play a 1-yard gain; it never kicks early. The timeout
        # keeps the half alive; without one left the snap runs it out
        m = built()
        for b in [m.time] + [v for k, v in m.own.items() if k[2] == "time"]:
            b.payload["seconds"][:] = np.where(b.payload["seconds"] > v21.STOPPED_MAX, 30.0, 5.0)
            b.payload["cut"][:] = False
        _fixed_play(m, 1, False)
        n_who = len(m.gamers) + 1
        m.choices["early_fg"] = _always("early_fg", 0, n_who)
        m.choices["to_off"] = _always("to_off", 0, n_who)
        m.choices["to_def"] = _always("to_def", 0, n_who)
        s = _state(m, period=4, clock=20.0, team=0, down=1, dist=10, y=80, prev=v21.SHORT, home=7, away=7)
        st = {}
        v21.simulate(m, s, 20, stats=st, max_steps=1)
        self.assertGreater(st.get("late_timeouts", 0), 0)                   # each 30-second draw
        log = v21.snap_log(st)
        self.assertTrue((log["clock"] - log["seconds"] > 0).all())        # no snap ran the half out
        s["tos"] = np.array([[0, 3]])
        st = {}
        v21.simulate(m, s, 20, stats=st, max_steps=1)
        self.assertEqual(st.get("late_timeouts", 0), 0)
        s["tos"], s["home"] = np.array([[3, 3]]), np.array([14])            # ahead: lets it run
        st = {}
        v21.simulate(m, s, 20, stats=st, max_steps=1)
        self.assertEqual(st.get("late_timeouts", 0), 0)
        s["home"], s["y"] = np.array([7]), np.array([30])                    # level at its own 30: calls it too
        st = {}
        v21.simulate(m, s, 20, stats=st, max_steps=1)
        self.assertGreater(st.get("late_timeouts", 0), 0)
        s["period"], s["clock"] = np.array([2]), np.array([20.0])           # but not at Q2's end out of range
        st = {}
        v21.simulate(m, s, 20, stats=st, max_steps=1)
        self.assertEqual(st.get("late_timeouts", 0), 0)

    def test_a_first_half_start_can_play_to_the_half_alone(self):
        m = built()
        s = _state(m, period=2, clock=30.0, team=0, y=50)
        st = {}
        h, a = v21.simulate(m, s, 50, stats=st, until_half=True)
        self.assertEqual(st["unfinished"], 0)
        self.assertTrue((st["drive_end"] >= 0).all())                    # the start's drive ended by the half
        st2 = {}
        v21.simulate(m, s, 50, stats=st2)
        self.assertGreaterEqual(st2["snaps"], st["snaps"])                # the whole game plays on past it

    def test_overtime_ends_when_the_second_side_goes_ahead(self):
        m = built()
        _fixed_time(m, 7.0)
        _fixed_play(m, 99, True)                                           # every snap a touchdown
        m.gain_surv = np.ones(101)
        n_who = len(m.gamers) + 1
        m.choices["two"] = v21.Choice.build("two", np.zeros((0, 2)), [], [], n_who)         # always the kick
        m.choices["kickoff"] = v21.Choice.build("kickoff", np.zeros((0, 3)), [], [], n_who)  # always returned
        m.kick_pool = {k: (np.array([25]), np.array([4.0])) for k in m.kick_pool}
        m.two_good = m.kick_good = 1.0
        m.two_good_by = np.ones(3)
        st = {}
        h, a = v21.simulate(m, _state(m, period=5, clock=240.0, team=0, y=50, home=14, away=14), 20, stats=st)
        np.testing.assert_array_equal(h, 21)                               # 6 and the kick
        np.testing.assert_array_equal(a, 22)                               # 6 and the forced two
        self.assertEqual(st["unfinished"], 0)
        np.testing.assert_array_equal(st["q_points"][:, 4], 15)            # decided mid-period: still counted


def _r(msg, period, clock, kind, off, down=1, dist=10, y=25, a=0, b=0, messages=""):
    return _row(msg, period, clock, kind, off, down, dist, y, a, b, messages)


class TestDecisions(unittest.TestCase):

    def test_leans_recover_each_gamer_s_own_rate(self):
        rng = np.random.default_rng(0)
        n = 300
        who = np.repeat([0, 1, 2], n)
        rate = np.array([0.6, 0.1, 0.3])[who]
        chosen = np.where(rng.random(3 * n) < rate, v21.GO, v21.PUNT)
        K = len(v21.DECISIONS["fourth"]["features"])
        X = np.zeros((3 * n, K), np.int64)
        few = np.zeros((5, K), np.int64)
        X, who, chosen = np.r_[X, few], np.r_[who, np.full(5, 3)], np.r_[chosen, np.full(5, v21.GO)]
        ch = v21.Choice.build("fourth", X, chosen, who, 5)
        cols = {f: np.zeros(4, np.int64) for f in v21.DECISIONS["fourth"]["features"]}
        p = ch.probs(cols, np.array([0, 1, 2, 4]))[:, v21.GO]
        np.testing.assert_allclose(p[:3], [0.6, 0.1, 0.3], atol=0.07)
        self.assertAlmostEqual(p[3], ch.league[v21.GO], places=6)          # the unknown gamer: the league's
        few_go = ch.probs(cols, np.array([3]))[0, v21.GO]
        self.assertLess(few_go, 0.75)                                        # 5 of 5: shrunk well short of all
        self.assertGreater(few_go, ch.league[v21.GO])
        flat = v21.Choice.build("fourth", X, chosen, who, 5, gamers=False)
        np.testing.assert_allclose(flat.probs(cols, np.array([0, 1, 2, 3]))[:, v21.GO], ch.league[v21.GO], atol=1e-9)

    def test_blended_cells_weigh_a_thin_cell_by_its_decisions(self):
        # 10 real kick-offs at (desperate, late, lead) = (1, 1, 2), all touchbacks; 400 elsewhere returned
        X = np.r_[np.tile([1, 1, 2], (10, 1)), np.zeros((400, 3), np.int64)]
        chosen = np.r_[np.full(10, v21.TOUCHBACK), np.full(400, v21.RETURN)]
        who = np.zeros(410, np.int64)
        cols = dict(desperate=np.array([1]), late=np.array([1]), lead=np.array([2]))
        first = v21.Choice.build("kickoff", X, chosen, who, 1, gamers=False)
        mixed = v21.Choice.build("kickoff", X, chosen, who, 1, gamers=False, blend=40.0)
        self.assertLess(first.base(cols)[0, v21.TOUCHBACK], 0.1)             # under 25: the whole pool's
        p = mixed.base(cols)[0]
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertGreater(p[v21.TOUCHBACK], 0.3)                            # 10 of its own weigh in
        np.testing.assert_allclose(mixed.probs(cols, np.array([0]))[0], p, atol=1e-9)
        m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, decision_blend=25.0)
        self.assertEqual(m.choices["fourth"].blend, 25.0)
        self.assertEqual(m.choices["two"].blend, v21.CONV_BLEND)            # the lead after the six decides
        # two-point tries made by the part of the game, each shrunk toward the game's
        conv = [c for code in MATCHES for c in v21.decision_points(v21._with_side(MATCHES[code]))["conv"]]
        self.assertTrue(all(len(c) == 3 and c[2] in (0, 1, 2) for c in conv))
        self.assertEqual(m.two_good_by.shape, (3,))
        for k in range(3):
            n_k = sum(1 for t, g, l in conv if t and l == k)
            made = sum(g for t, g, l in conv if t and l == k)
            self.assertAlmostEqual(m.two_good_by[k], (made + v21.TWO_GOOD_PRIOR * m.two_good) / (n_k + v21.TWO_GOOD_PRIOR))
        self.assertLess(v21.CONV_BLEND, 25.0)
        self.assertEqual(v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False).choices["two"].blend, 0.0)
        h, a = v21.simulate(m, v21.kickoff_start(m), 30)
        self.assertGreater((h + a).mean(), 10)

    def test_the_early_kick_reads_the_clock_and_timeouts(self):
        spec = v21.DECISIONS["early_fg"]
        self.assertIn("clock2", spec["features"])
        self.assertIn("to3", spec["features"])
        for lv in spec["levels"][:-3]:
            self.assertIn("fg4", lv)                                     # what a kick does, to the end
        pts = [dict(left=20.0, togo=20, down=2, lead=0, period=4, prev=v21.SHORT, own_to=0),
               dict(left=20.0, togo=20, down=2, lead=0, period=4, prev=v21.NOTHING, own_to=3)]
        X = v21.decision_features("early_fg", pts)
        f = dict(zip(spec["features"], X.T))
        self.assertEqual(list(f["clock2"]), [1, 0])
        self.assertEqual(list(f["to3"]), [0, 2])
        t = v21.decision_features("to_off", [dict(period=4, clock=20.0, off_margin=0, left=20.0, togo=20),
                                             dict(period=4, clock=20.0, off_margin=0, left=20.0, togo=60)])
        self.assertEqual(list(t[:, -1]), [1, 0])                          # in range or not

    def test_match_decisions_reads_the_real_choices(self):
        rows = [_r(7, 1, 236, "KICKOFF", "TEAM_A", y=25),                            # the opening kick, returned
                _r(14, 1, 200, "SCRIMMAGE", "TEAM_A", 4, 2, 40),                    # 4th and 2: punted
                _r(21, 1, 190, "PUNT", "TEAM_B", 1, 10, 20),
                _r(28, 2, 20, "SCRIMMAGE", "TEAM_B", 2, 6, 75),                     # 0:20 left, in range: kicked
                _r(35, 2, 15, "FIELD_GOAL", "TEAM_B", b=3, messages="FIELD_GOAL_GOOD_TEAM_B"),
                _r(42, 2, 15, "KICKOFF", "TEAM_A", y=20, b=3),                      # a touchback
                _r(49, 2, 3, "SCRIMMAGE", "TEAM_A", 1, 10, 70, b=3),                # in range, the half runs out
                _r(56, 3, 236, "KICKOFF", "TEAM_B", y=30, b=3),
                _r(63, 4, 100, "TOUCHDOWN", "TEAM_A", a=6, b=3, messages="TOUCHDOWN_TEAM_A"),
                _r(70, 4, 100, "CONVERSION", "TEAM_A", a=8, b=3, messages="TWO_POINT_CONVERSION_SUCCESSFUL_TEAM_A"),
                _r(77, 4, 98, "KICKOFF", "TEAM_A", y=45, a=8, b=3)]                 # recovered by the kicker
        got = v21.match_decisions(rows)
        self.assertEqual([r[:2] for r in got["fourth"]], [(0, v21.PUNT)])
        self.assertEqual(sorted(r[:2] for r in got["early_fg"]), [(0, 0), (1, 1)])
        kicks = [(r[0], r[1], r[2], r[3]) for r in got["kickoff"]]
        self.assertEqual(kicks, [(1, v21.RETURN, 0, 0), (1, v21.TOUCHBACK, 0, 1), (0, v21.RETURN, 0, 0),
                                 (0, v21.KEPT, 0, 1)])
        self.assertEqual(got["two"][0][:2], (0, 1))
        self.assertEqual(got["two"][0][3], 3 + v21.CONV_MARGIN)                 # the lead with the six on the board
        self.assertEqual(got["fg_clock"], [(20.0, 5.0)])
        self.assertEqual(got["conv"], [(True, True, 1)])                 # made, in Q4's last 3:00

    def test_a_kick_the_feed_missed_at_the_half_s_end(self):
        # Q2: in range at 0:03, the next row is Q3's kick-off with 3 more to the side: it kicked
        rows = [_r(7, 1, 236, "KICKOFF", "TEAM_A", y=25),
                _r(14, 2, 3, "SCRIMMAGE", "TEAM_A", 2, 6, 75),
                _r(21, 3, 236, "KICKOFF", "TEAM_B", y=30, a=3),
                # Q4: 3 behind, in range at 0:04 on 4th down, the rows end there; the final has 3 more to it
                _r(28, 4, 4, "SCRIMMAGE", "TEAM_B", 4, 3, 80, a=3)]
        got = v21.decision_points(rows, final=(3.0, 3.0))
        self.assertEqual([q["choice"] for q in got["early_fg"]], [1])
        self.assertEqual([q["choice"] for q in got["fourth"]], [v21.FG])
        self.assertEqual(got["missed_kicks"], 2)
        # no final, or a final the rows already have: the half ran out on the side
        got = v21.decision_points(rows[:3] + [_r(28, 4, 4, "SCRIMMAGE", "TEAM_B", 1, 10, 80, a=3)], final=(3.0, 0.0))
        self.assertEqual([q["choice"] for q in got["early_fg"]], [1, 0])
        self.assertEqual(v21.end_points(rows[1], rows[2]), (3, 0))
        self.assertIsNone(v21.end_points(rows[-1], None))

    def test_scores_the_rows_do_not_mark(self):
        # a snap from the 12, then the conversion with 6 more to the side: its touchdown, no TOUCHDOWN row
        rows = [_r(7, 1, 200, "SCRIMMAGE", "TEAM_A", 1, 10, 88),
                _r(14, 1, 195, "CONVERSION", "TEAM_A", a=6),
                _r(21, 1, 195, "KICKOFF", "TEAM_A", y=35, a=7),
                _r(28, 1, 190, "SCRIMMAGE", "TEAM_B", 1, 10, 30, a=7),
                _r(35, 1, 185, "KICKOFF", "TEAM_B", y=35, a=13),               # a pick-six off TEAM_B
                _r(42, 1, 180, "SCRIMMAGE", "TEAM_B", 4, 8, 75, a=13),
                _r(49, 1, 175, "KICKOFF", "TEAM_B", y=35, a=13, b=3)]          # its kick, no FIELD_GOAL row
        recs = v21.snap_records(rows)
        first = next(r for r in recs if r["start"] == 7)
        self.assertEqual((first["kind"], first["gain"], first["converted"], first["read_off_score"]),
                         (v21.GAIN, 12, True, True))
        pick = next(r for r in recs if r["start"] == 28)
        self.assertEqual(pick["kind"], v21.DEF_TD)
        got = v21.decision_points(rows)
        self.assertEqual([q["choice"] for q in got["fourth"]], [v21.FG])

    def test_a_touchback_puts_the_ball_at_the_20_with_no_clock(self):
        m = built()
        m.choices["kickoff"] = _always("kickoff", v21.TOUCHBACK, len(m.gamers) + 1)
        st = {}
        v21.simulate(m, v21.kickoff_start(m), 20, stats=st)
        log = v21.snap_log(st)
        first = [np.flatnonzero(log["path"] == p)[0] for p in range(40)]
        self.assertTrue((log["field"][first] == v21.KICK_TOUCHBACK).all())
        self.assertTrue((log["clock"][first] == v21.QUARTER).all())

    def test_an_early_kick_comes_before_the_snap(self):
        m = built()
        m.choices["early_fg"] = _always("early_fg", 1, len(m.gamers) + 1)
        st = {}
        v21.simulate(m, _state(m, period=2, clock=30.0, down=1, y=80), 30, stats=st)
        self.assertGreaterEqual(st["early_field_goals"], 30)
        log = v21.snap_log(st)
        first = log["path"] == 0
        self.assertEqual(log["side"][first][0], 1)                          # the kick came first: the other side's ball

    def test_a_gamer_s_lean_carries_into_the_sim(self):
        m = built()
        n_who = len(m.gamers) + 1
        rng = np.random.default_rng(1)
        who = np.repeat([0, 1], 200)
        chosen = np.where(rng.random(400) < np.where(who == 0, 0.8, 0.05), v21.GO, v21.PUNT)
        X = rng.integers(0, 3, (400, len(v21.DECISIONS["fourth"]["features"])))
        m.choices["fourth"] = v21.Choice.build("fourth", X, chosen, who, n_who)
        m.rules = {}
        st = {}
        v21.simulate(m, _state(m, down=4, dist=5, y=50, gamers=(0, 1)), 200, stats=st)
        c = v21.decision_census(st, n_who)["fourth"]["who"]
        go = c[:, v21.GO] / np.maximum(1, c.sum(1))
        self.assertGreater(go[0], 0.6)
        self.assertLess(go[1], 0.2)


class TestRules(unittest.TestCase):

    def test_the_sweep_finds_where_an_option_never_happens(self):
        rng = np.random.default_rng(3)
        n = 4000
        left, lead = rng.integers(0, 240, n), rng.integers(-20, 21, n)
        never = (left <= 40) & (lead <= -4)                                   # planted: no kick here
        y = np.where(never, False, rng.random(n) < 0.4)
        y[np.flatnonzero(never)[:1]] = True                                    # one who did anyway
        regions = v21_rules.find_regions(np.stack([left, lead], axis=1), y, min_region=30, rare=0.01)
        self.assertTrue(regions)
        box, idx = max(regions, key=lambda r: len(r[1]))
        self.assertGreater(len(idx), 0.8 * never.sum())
        self.assertLessEqual(box[0][1], 45)                                     # left: up to about 0:40
        self.assertLessEqual(box[1][1], -3)                                     # lead: 4 or more behind
        self.assertLessEqual(y[idx].mean(), 0.01)
        self.assertEqual(v21_rules.find_regions(np.stack([left, lead], axis=1), rng.random(n) < 0.5), [])
        rare_opt = rng.random(n) < 0.03                                         # 3% anywhere: 0 of 40 is chance
        rare_opt[(left <= 40) & (lead <= -4)] = False
        small = np.flatnonzero((left <= 40) & (lead <= -4))[:40]
        F2 = np.stack([left, lead], axis=1)
        keep = np.r_[np.flatnonzero(~((left <= 40) & (lead <= -4))), small]
        self.assertEqual(v21_rules.find_regions(F2[keep], rare_opt[keep], min_region=30), [])
        self.assertAlmostEqual(v21_rules.binom_cdf(0, 100, 0.03), 0.97 ** 100)

    def test_learned_rules_strike_the_option_out(self):
        rules = {"fourth": [(v21.FG, {"period": (4, 4), "left": (None, 40), "lead": (None, -4)})]}
        cols = dict(period=np.array([4, 4, 3]), left=np.array([20, 20, 20]), lead=np.array([-7, -2, -7]))
        ok = v21_rules.mask(rules, "fourth", cols)
        self.assertEqual(ok[:, v21.FG].tolist(), [False, True, True])
        self.assertTrue(ok[:, v21.GO].all())
        self.assertEqual(v21_rules.describe(rules["fourth"][0][1]),
                         "Q4, 0:40 or less left in the half, behind by 4 or more")

    def test_the_sim_never_takes_a_struck_out_option(self):
        m = built()
        m.choices["fourth"] = _always("fourth", v21.FG, len(m.gamers) + 1)
        m.rules = {"fourth": [(v21.FG, {"period": (4, 4), "left": (None, 40), "lead": (None, -4)})]}
        st = {}
        v21.simulate(m, _state(m, period=4, clock=20.0, down=4, dist=3, y=80, home=10, away=17), 50, stats=st)
        name, who, k, sl = st["_dec"][0][:4]
        self.assertEqual(name, "fourth")
        self.assertFalse((k == v21.FG).any())
        m.use_rules = False
        st = {}
        v21.simulate(m, _state(m, period=4, clock=20.0, down=4, dist=3, y=80, home=10, away=17), 50, stats=st)
        self.assertTrue((st["_dec"][0][2] == v21.FG).all())

    def test_overtime_s_own_rules(self):
        ok = v21.ot_mask("fourth", 2, (np.array([True, False]), np.array([True, True])))
        self.assertEqual(ok[0].tolist(), [True, False, False])                  # behind 4+: no kick, no punt
        self.assertTrue(ok[1].all())
        self.assertFalse(v21.ot_mask("two", 1, other_done_close=np.array([True]))[0, 0])

    def test_the_game_ends_when_the_side_ahead_can_run_it_out(self):
        self.assertTrue(v21.auto_end(4, 100.0, 3, 2, 0))                        # 3 downs x 40 > 1:40
        self.assertFalse(v21.auto_end(4, 130.0, 3, 2, 0))
        self.assertFalse(v21.auto_end(4, 100.0, 3, 2, 1))                       # a timeout left
        self.assertFalse(v21.auto_end(4, 100.0, 0, 2, 0))                       # level
        m = built()
        start = _state(m, period=4, clock=100.0, down=2, y=40, prev=v21.SHORT, home=7, away=4)
        start["tos"] = np.array([[3, 0]])
        st = {}
        h, a = v21.simulate(m, start, 30, stats=st)
        np.testing.assert_array_equal(h, 7)
        np.testing.assert_array_equal(a, 4)
        self.assertEqual(st["auto_end"], 30)
        self.assertFalse(st.get("_snaps"))

    def test_a_build_learns_its_rules_or_goes_without(self):
        m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False)
        self.assertTrue(m.use_rules)
        self.assertIn("fourth", m.rules)                    # the fakes kick from 40 yards in and punt from further
        self.assertIn((v21.FG, {"togo": (41, None)}), m.rules["fourth"])
        self.assertIn((v21.PUNT, {"togo": (None, 40)}), m.rules["fourth"])
        m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, rules=False)
        self.assertEqual(m.rules, {})
        self.assertTrue(all(ch.ruled == 0 for ch in m.choices.values()))

    def test_the_sweep_report(self):
        found, auto, counts = v21_rules.sweep(MATCHES, HISTORY)
        lines = "\n".join(v21_rules.report(found, auto, counts))
        self.assertIn("4th down", lines)
        self.assertIn("field goal: never or all but never in", lines)
        self.assertIn("41 or more yards from the goal: 231 real, never", lines)
        self.assertIn("go for it: never or all but never in\n      anywhere", lines)   # the fakes never go for it
        with tempfile.TemporaryDirectory() as d:
            v21_rules.write_csv(os.path.join(d, "s.csv"), found)
            with open(os.path.join(d, "s.csv"), newline="") as fh:
                self.assertTrue(list(csv.DictReader(fh)))


class TestReport(unittest.TestCase):

    def run_report(self, workers=1):
        model = built()
        codes = sorted(MATCHES)[:12]
        means = {c: (9.0 + i % 3, 8.0) for i, c in enumerate(codes)}
        books = {c: prior_check.raw_books(means[c], 0.2, 0.1, n=2000) for c in codes}
        with tempfile.TemporaryDirectory() as d:
            model.save(os.path.join(d, v21.MODEL_FILE))
            out = v21_report.run(d, {c: MATCHES[c] for c in codes}, HISTORY, means, "2026-10-01", "2026-10-01",
                                 n_paths=60, state_paths=20, every=6, workers=workers, prior_books=books)
            v21_report.write_csv(os.path.join(d, "m.csv"), out)
            v21_report.write_states_csv(os.path.join(d, "s.csv"), out)
            with open(os.path.join(d, "s.csv"), newline="") as fh:
                states = list(csv.DictReader(fh))
        return out, states

    def test_the_nudge_meets_the_inputs(self):
        m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, mode="tilt", key="lean",
                      blend=40.0, field=0.2)
        d = np.asarray(m.nudge_dir)
        self.assertEqual(d.shape, (len(v21.PHI_DIST),))
        self.assertTrue((d * np.array([1.0, 1.0, -1.0, 1.0, -1.0]) >= 0).all())   # no component the wrong way
        self.assertGreater(m.nudge_slope, 0.4)                                 # a unit: about a point a side
        self.assertLess(m.nudge_slope, 2.5)
        start = v21.kickoff_start(m)
        h0, a0 = v21.simulate(m, start, 300, seed=3)
        target = (h0.mean() + 4.0, a0.mean() - 2.0)
        (h, a), alpha = v21.nudge_solve(m, start, target, 300, seed=3)
        self.assertGreater(alpha[0], 0)
        self.assertLess(alpha[1], 0)
        self.assertLess(abs(h.mean() - target[0]), 2.0)
        self.assertLess(abs(a.mean() - target[1]), 2.0)
        (hh, _), half = v21.nudge_solve(m, start, target, 300, seed=3, share=0.5)   # half the way there
        self.assertLess(abs(hh.mean() - (h0.mean() + 2.0)), 1.5)
        self.assertLess(half[0], alpha[0])
        codes = sorted(MATCHES)[:6]
        means = {c: (12.0, 8.0) for c in codes}
        with tempfile.TemporaryDirectory() as tmp:
            m.save(os.path.join(tmp, v21.MODEL_FILE))
            out = v21_report.run(tmp, {c: MATCHES[c] for c in codes}, HISTORY, means, "2026-10-01", "2026-10-01",
                                 n_paths=100, state_paths=10, every=8, nudge=True)
            plain = built()
            plain.save(os.path.join(tmp, v21.MODEL_FILE))
            with self.assertRaises(SystemExit):
                v21_report.run(tmp, {c: MATCHES[c] for c in codes}, HISTORY, means, "2026-10-01", "2026-10-01",
                               n_paths=20, state_paths=0, nudge=True)
        lines = "\n".join(v21_report.report(out))
        self.assertIn("nudged to the inputs", lines)
        self.assertIn("NUDGE:", lines)
        kick = np.array([r["kick"] for r in out["results"]])
        self.assertLess(abs(kick[:, 0].mean() - 12.0), 2.0)
        with tempfile.TemporaryDirectory() as tmp:                 # the gap shrink: the margin met is half the inputs'
            m.save(os.path.join(tmp, v21.MODEL_FILE))
            half = v21_report.run(tmp, {c: MATCHES[c] for c in codes}, HISTORY, means, "2026-10-01", "2026-10-01",
                                  n_paths=100, state_paths=0, nudge=True, gap=0.5)
        k2 = np.array([r["kick"] for r in half["results"]])
        self.assertLess((k2[:, 0] - k2[:, 1]).mean(), (kick[:, 0] - kick[:, 1]).mean() - 1.5)
        self.assertLess(abs((k2[:, 0] + k2[:, 1]).mean() - (kick[:, 0] + kick[:, 1]).mean()), 1.5)   # the total kept
        self.assertEqual(half["results"][0]["means"], (12.0, 8.0))            # set against the inputs as given
        self.assertIn("the inputs' margins x 0.5 (the gap shrink)", "\n".join(v21_report.report(half)))

    def test_report_sets_the_sim_against_the_inputs(self):
        out, states = self.run_report()
        self.assertEqual(len(out["results"]), 12)
        self.assertEqual(len(states), sum(len(x["states"]) for x in out["results"]))
        lines = "\n".join(v21_report.report(out))
        for part in ("KICK-OFF", "GRADED", "EACH STEP", "THE DRAWS", "by team", "THE DECISIONS", "4th down"):
            self.assertIn(part, lines)
        real = v21_report.real_census({c: MATCHES[c] for c in sorted(MATCHES)[:12]})
        self.assertEqual(real["events"]["snaps"], sum(len(v21.match_snaps(MATCHES[c])) for c in sorted(MATCHES)[:12]))
        self.assertEqual(int(real["time"].sum()), real["events"]["snaps"])

    def test_steps(self):
        self.assertEqual(list(v21_report.step_of([1, 1, 2, 4, 5], [200, 100, 121, 120, 50])), [0, 1, 2, 7, 8])

    def test_the_final_margin(self):
        """A sim sure of a 3-point home win, a prior spread over 1-14: the bands, the key margins, the spread and
        how the favourite wins, against finals of +3 and -10."""
        from .. import v13
        sure = v13._distributions(np.full(100, 24), np.full(100, 21))
        wide = v13._distributions(np.repeat([21, 21, 30, 35], 25), np.repeat([20, 14, 20, 21], 25))
        results = [dict(code=f"M{i}", finals=f, book=sure, prior_book=wide) for i, f in enumerate([(24, 21)] * 6 + [(14, 24)] * 2)]
        lines = v21_report.margin_lines(results)
        text = "\n".join(lines)
        self.assertIn("THE FINAL MARGIN from kick-off: 8 matches", text)
        row = lambda start: next(l for l in lines if l.strip().startswith(start)).split()
        self.assertEqual(row("v21 sim expects")[3:8], ["0.0%", "100.0%", "0.0%", "0.0%", "0.0%"])
        self.assertEqual(row("prior expects")[2:7], ["0.0%", "25.0%", "25.0%", "50.0%", "0.0%"])
        self.assertEqual(row("real")[1:6], ["0.0%", "75.0%", "0.0%", "25.0%", "0.0%"])
        keys = next(i for i, l in enumerate(lines) if "the key margins" in l)
        self.assertEqual(lines[keys + 1].split()[3:9], ["0.0%", "0.0%", "100.0%", "0.0%", "0.0%", "0.0%"])
        sim = next(l for l in lines if l.strip().startswith("v21 sim") and "its sd" in l)
        self.assertIn("its sd 0.00", sim)
        fav = next(l for l in lines if l.strip().startswith("fav by 2-4") and "(v21 sim)" in l).split()
        self.assertEqual(fav[4:10], ["100.0", "75.0", "0.0", "0.0", "0.0", "25.0"])   # wins 1-7, 8+, loses
        self.assertEqual(v21_report.margin_lines([]), [])


class TestAudit(unittest.TestCase):

    def test_slices_and_outcomes(self):
        from .. import v21_audit as A
        self.assertEqual(list(A.slice_of([1, 2, 2, 2, 2, 3, 4, 4, 5], [200, 200, 100, 45, 20, 10, 121, 30, 100])),
                         [0, 1, 2, 3, 4, 5, 6, 9, 10])
        self.assertEqual(list(A.outcome_of([v21.TURNOVER, v21.GAIN, v21.GAIN, v21.GAIN, v21.GAIN, v21.GAIN],
                                           [0, 30, 10, 3, 0, -2], [10, 10, 10, 10, 10, 10], [70, 70, 50, 50, 50, 50])),
                         [0, A.TD, A.FIRST, 3, 2, 1])

    def test_a_drive_the_rows_end_on_reads_the_final(self):
        from .. import v21_audit as A
        rows = [_r(7, 4, 20, "SCRIMMAGE", "TEAM_A", 1, 10, 70), _r(14, 4, 9, "SCRIMMAGE", "TEAM_A", 2, 4, 76)]
        self.assertEqual(A.real_drive_end(rows, 0), v21.DRIVE_ENDS.index("clock out"))
        self.assertEqual(A.real_drive_end(rows, 0, (3.0, 0.0)), v21.DRIVE_ENDS.index("field goal"))
        self.assertEqual(A.real_drive_end(rows, 0, (7.0, 0.0)), v21.DRIVE_ENDS.index("touchdown"))
        self.assertEqual(A.real_drive_end(rows, 0, (0.0, 0.0)), v21.DRIVE_ENDS.index("clock out"))

    def test_the_audit_sets_the_sim_against_the_real_downs(self):
        from .. import v21_audit as A
        model = built()
        codes = sorted(MATCHES)[:8]
        with tempfile.TemporaryDirectory() as d:
            model.save(os.path.join(d, v21.MODEL_FILE))
            out = A.run(d, {c: MATCHES[c] for c in codes}, HISTORY, n_paths=30, state_paths=5)
        real, flow, step = out["real"], out["flow"], out["step"]
        self.assertEqual(real["games"], 8)
        self.assertEqual(flow["games"], 8 * 2 * 30)
        n_real = sum(len([r for r in v21.match_snaps(MATCHES[c])]) for c in codes)
        self.assertEqual(int(real["flow"].sum()), n_real)
        self.assertGreater(step["step"].sum(), 0.5 * real["step"].sum())     # most states snap at once
        self.assertGreater(real["fourth"].sum(), 0)
        np.testing.assert_allclose(step["fourth"].sum(), real["fourth"].sum(), rtol=0.01)
        lines = "\n".join(A.report(out))
        self.assertGreater(real["ot_n"].sum(), 0)
        np.testing.assert_allclose(step["ot_n"], real["ot_n"])
        np.testing.assert_allclose(step["by_rem"].sum(), step["step"].sum())     # the same snaps, split another way
        np.testing.assert_allclose(real["by_rem"].sum(), real["step"].sum())
        # Q4's last 2:00 by the lead: the same snaps as the rest; regulation points from every Q4 state
        q4_real = real["step"][:, 6:10].sum() - real["step"][:, 6].sum()
        self.assertEqual(real["late"].sum(), q4_real)
        self.assertGreater(real["pts_ball"].sum() + real["pts_other"].sum(), 0)
        self.assertGreater(step["pts_ball"].sum() + step["pts_other"].sum(), 0)
        np.testing.assert_allclose(step["level"].sum(1), real["level"].sum(1))
        # how the drive with the ball ended: one real end per Q4 snap state, the sim's shares sum to one each
        self.assertGreater(real["drives"].sum(), 0)
        np.testing.assert_allclose(step["drives"].sum(), real["drives"].sum(), rtol=1e-6)
        # 2nd-4th downs by the time the play took: every real one in the distance rows once
        later = real["step"][1:].sum()
        self.assertEqual(real["short_tb"][:3, :, 0].sum(), later)
        self.assertTrue((real["short_tb"][..., 1] <= real["short_tb"][..., 0]).all())
        self.assertGreater(step["short_tb"][:3, :, 0].sum(), 0)
        # conversions by the part of the game, before Q4's last 3:00 (the last 3:00 has its own table)
        convs = [q for c in codes for q in v21.decision_points(v21._with_side(MATCHES[c]))["two"]]
        self.assertGreater(len(convs), 0)
        self.assertEqual(real["two_part"].sum() + real["two_late"].sum(),
                         sum(q["cell"] != v21.CONV_CELLS - 1 for q in convs))     # all but overtime's
        self.assertEqual(real["two_good"][:, 0].sum(), sum(q["choice"] for q in convs))
        self.assertEqual(flow["two_sim"][3], 1.0)
        for part in ("GAME FLOW", "ONE SNAP", "3rd down, by distance", "4th DOWN", "THE LARGEST GAPS", "OVERTIME",
                     "ONE SNAP by the snaps left", "ONE SNAP in Q4's last 2:00", "Q4 STATES", "Q4 STATES, level",
                     "by what a field goal does", "CONVERSIONS in Q4's last 3:00", "by the time its play took",
                     "the league's share in the sim's cells", "how the drive with the ball ended",
                     "ONE SNAP on 2nd-4th downs by the time its play took", "CONVERSIONS before Q4's last 3:00",
                     "two-point tries made"):
            self.assertIn(part, lines)
        self.assertEqual(step["clock_out"].shape, (2, 4, len(A.CLOCK_OUTS) + 1))
        # Q2's last 2:00 to the half's end: one real drive end per state, the sim's shares sum to one each
        self.assertGreater(real["n2"].sum(), 0)
        np.testing.assert_allclose(real["drives2"].sum(), real["n2"].sum())
        np.testing.assert_allclose(step["drives2"].sum(), real["n2"].sum(), rtol=1e-6)
        self.assertIn("Q2 STATES, the half's last 2:00", lines)
        self.assertTrue((step["clock_out"][..., :4].sum(-1) <= step["clock_out"][..., 5] + 1e-9).all())


class TestStream(unittest.TestCase):
    """v2.1 as the calibrator's price stream (v21_stream)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, mode="tilt",
                          key="lean", blend=40.0, field=0.2, decision_blend=25.0, clock=0.2)
        m.save(os.path.join(cls.tmp.name, v21.MODEL_FILE))
        cls.model = m
        cls.code = sorted(MATCHES)[0]
        cls.rows = sorted(MATCHES[cls.code], key=lambda r: int(r["message"]))
        cls.info = [h for h in HISTORY if h["MATCH_CODE"] == cls.code]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _prod(self):
        """Prod's quotes on every market at every PLAY_OVER, and three before the first play."""
        c = self.code
        lines = ((50, ""), (51, ""), (52, "PLAYER 1 to score over 3.5 points more than PLAYER 2"),
                 (54, "Total points over 40.5"))
        out = [(c, k, f"2026-10-01 00:0{k - 50}:00", 50.0, 2.0, line, None, "OPEN", "true")
               for k, line in lines if k != 51]
        for r in self.rows:
            for k, line in lines:
                out.append((c, k, None, 50.0, 2.0, line, int(r["message"]), "OPEN", "true"))
        return out

    def test_the_states_it_prices_from(self):
        from .. import v21_stream as vs
        starts = vs.state_starts(self.model, self.rows)
        msgs = [m for m, _ in starts]
        self.assertEqual(msgs, sorted(msgs))
        kinds = {int(r["message"]): r["play_kind"] for r in self.rows}
        phases = {m: int(s["phase"][0]) for m, s in starts}
        self.assertTrue(all(kinds[m] in ("SCRIMMAGE", "KICKOFF", "PUNT", "TURNOVER_ON_DOWNS")
                            for m, p in phases.items() if p == v21.SNAP))
        self.assertNotIn("TOUCHDOWN", {kinds[m] for m in phases})          # the conversion still to come
        kicks = [m for m, p in phases.items() if p == v21.KICK]
        self.assertTrue(kicks)                                              # after a score, the kick-off
        self.assertTrue(all(kinds[m] in ("CONVERSION", "FIELD_GOAL") for m in kicks))
        for m in kicks:
            s = dict(starts)[m]
            r = next(x for x in self.rows if int(x["message"]) == m)
            recv = 0 if r["play_kind"] == "CONVERSION" and "TEAM_B" in (r.get("play_messages") or "") else None
            if recv is not None:
                self.assertEqual(int(s["team"][0]), recv)                   # the scorer kicks to the other

    def test_quotes_from_kick_off_and_in_play_shaped_like_prods(self):
        from .. import v21_stream as vs
        prod = self._prod()
        q = vs.quotes_for_matches({self.code: self.rows}, prod, self.tmp.name, n_paths=40, workers=1,
                                  match_info=self.info, lines=vs.LINE_MODES, nudge_share=0.0)
        self.assertEqual(set(q), set(vs.LINE_MODES))
        own = q[vs.PROD_LINES]
        self.assertTrue(own)
        self.assertTrue(all(len(x) == 9 and 0 < x[3] < 100 and x[0] == self.code for x in own))
        pre = [x for x in own if x[6] is None]
        self.assertEqual(sorted(x[1] for x in pre), [50, 52, 54])           # the pre-match quotes
        live = {(x[1], x[6]) for x in own if x[6] is not None}
        kinds = {int(r["message"]): r["play_kind"] for r in self.rows}
        self.assertFalse({m for _, m in live if kinds.get(m) == "TOUCHDOWN"})
        self.assertTrue({m for _, m in live if kinds.get(m) == "CONVERSION"})
        # home and away moneylines at one state add to one
        by = {(x[1], x[6]): x[3] for x in own}
        both = [m for (k, m) in by if k == 50 and (51, m) in by]
        self.assertTrue(both)
        for m in both:
            self.assertAlmostEqual(by[(50, m)] + by[(51, m)], 100.0, delta=0.5)

    def test_the_books_the_bets_read(self):
        """A bet reads the latest quote of its market at or before prod's message: the book of the latest
        prod quote there that falls in a book's window, none before the first PLAY_OVER."""
        from .. import v21_stream as vs
        prod = [("M", k, None, 50.0, 2.0, "", m, "OPEN", "true") for k, ms in ((50, (5, 10, 12, 16, 20, 26, 31)),
                                                                              (54, (31,))) for m in ms]
        keys, windows = [10, 20, 30], {10: 15, 20: 25, 30: 99}
        self.assertEqual(vs._read(keys, prod, windows, [(50, 12)]), {10})
        self.assertEqual(vs._read(keys, prod, windows, [(50, 18)]), {10})      # 16 outside 10's window: 12's
        self.assertEqual(vs._read(keys, prod, windows, [(50, 27)]), {20})
        self.assertEqual(vs._read(keys, prod, windows, [(50, 7), (51, 20), (54, 29)]), set())   # kick-off, none
        self.assertEqual(vs._read(keys, prod, windows, [(50, 12), (54, 40), (50, 31)]), {10, 30})

    def test_only_the_moments_the_bets_read(self):
        from .. import v21_stream as vs
        prod = self._prod()
        price = lambda moments=None: vs.quotes_for_matches(
            {self.code: self.rows}, prod, self.tmp.name, n_paths=20, workers=1, match_info=self.info,
            lines=vs.PROD_LINES, nudge_share=0.0, moments=moments)
        full = price()
        every = vs.LAST_TIMING["states"]
        self.assertEqual(vs.LAST_TIMING["states_of"], every)
        none = price({})
        self.assertEqual(vs.LAST_TIMING["states"], 0)
        self.assertEqual(vs.LAST_TIMING["states_of"], every)
        kick = {(x[1], x[6]) for x in none}                                  # the kick-off's prices alone
        played = sorted({x[6] for x in full if x[1] == 50 and (50, x[6]) not in kick})
        read = played[len(played) // 2]
        fast = price({self.code: [(50, read)]})
        self.assertEqual((vs.LAST_TIMING["states"], vs.LAST_TIMING["states_of"]), (1, every))
        self.assertGreater(every, 1)
        live = {(x[1], x[6]) for x in fast} - kick
        self.assertIn((50, read), live)
        self.assertLess(live, {(x[1], x[6]) for x in full})                  # one book's quotes, of them all
        keys = [m for m, _ in vs.state_starts(self.model, [vs._as_text(r) for r in self.rows])]
        at = lambda m: bisect.bisect_right(keys, m)
        self.assertEqual({at(m) for _, m in live}, {at(read)})
        self.assertEqual(sorted(x for x in fast if (x[1], x[6]) in kick), sorted(none))   # the kick-off as before

    def test_the_gap_shrink(self):
        """v21-gapL: the prior's expected margin x L, its total kept, met by the nudge -- the prices exactly those of a
        prior with that margin."""
        from .. import v21_stream as vs
        self.assertEqual((vs.gap_of("v21-gap0.75"), vs.gap_of("v21-gap1"), vs.gap_of("v21"), vs.gap_of("v21-shape0.5")),
                         (0.75, None, None, None))
        self.assertEqual(vs.shrunk((30.0, 10.0), 0.5), (25.0, 15.0))
        self.assertEqual(vs.shrunk((30.0, 10.0), None), (30.0, 10.0))

        class Prior:
            league = (15.0, 15.0)

            def __init__(self, means):
                self._means = means

            def means(self, rows, results=None):
                return {r["MATCH_CODE"]: self._means for r in rows}

        prod = [p for p in self._prod() if p[6] is None]
        price = lambda means, gap: {x[1]: x[3] for x in self._priced(vs, prod, Prior, means, gap)}
        wide, half, level = price((14.0, 6.0), None), price((14.0, 6.0), 0.5), price((12.0, 8.0), None)
        self.assertEqual(half, level)
        self.assertNotEqual(half, wide)

    def _priced(self, vs, prod, prior, means, gap):
        with mock.patch.object(vs, "prematch_model", lambda _t: prior(means)), \
                mock.patch.object(vs, "_prior_tables", lambda _d=None: "prior"):
            return vs.quotes_for_matches({self.code: self.rows}, prod, self.tmp.name, n_paths=60, workers=1,
                                         match_info=self.info, lines=vs.PROD_LINES, nudge_share=1.0, gap=gap)

    def test_the_nudge_moves_the_kick_off_toward_the_prior(self):
        from .. import v21_stream as vs

        class Prior:
            league = (15.0, 15.0)

            def __init__(self, means):
                self._means = means

            def means(self, rows, results=None):
                return {r["MATCH_CODE"]: self._means for r in rows}

        prod = [p for p in self._prod() if p[6] is None]
        got = {}
        for name, means in (("home", (40.0, 5.0)), ("away", (5.0, 40.0))):
            with mock.patch.object(vs, "prematch_model", lambda _t, m=means: Prior(m)), \
                    mock.patch.object(vs, "_prior_tables", lambda _d=None: "prior"):
                q = vs.quotes_for_matches({self.code: self.rows}, prod, self.tmp.name, n_paths=60, workers=1,
                                          match_info=self.info, lines=vs.PROD_LINES, nudge_share=1.0)
            got[name] = {x[1]: x[3] for x in q}
        self.assertGreater(got["home"][50], got["away"][50] + 20)
        self.assertEqual((vs.LAST_TIMING["prior"], vs.LAST_TIMING["prior_matches"], vs.LAST_TIMING["matches"]),
                         ("prior", 1, 1))                         # which prior, and how many matches it has
        with mock.patch.object(vs, "_prior_tables", lambda _d=None: None), self.assertRaises(SystemExit):
            vs.quotes_for_matches({self.code: self.rows}, prod, self.tmp.name, n_paths=20, workers=1,
                                  match_info=self.info, nudge_share=0.75)

    def test_the_default_prior_is_the_form_table(self):
        """With no prior named: $EAMF_V21_PRIOR, then the form table form-prior writes, then $EAMF_V14_MODEL,
        then ./v14_model -- the first that resolves."""
        from .. import v21_stream as vs
        with mock.patch.dict(os.environ, {"EAMF_V21_PRIOR": "", "EAMF_V14_MODEL": ""}):
            c = vs._prior_candidates()
        self.assertEqual(c[1], os.path.join("eAMFCalibrator", "out", "form_prior", "priors", "avg_form4_w05.csv"))
        self.assertEqual(os.path.normpath(c[2]), os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(vs.__file__)), "..", vs.DEFAULT_PRIOR_TABLE)))
        self.assertEqual(c[4], "v14_model")
        with tempfile.TemporaryDirectory() as d:
            table = os.path.join(d, "avg_form4_w05.csv")
            with open(table, "w", encoding="utf-8") as fh:
                fh.write("MATCH_CODE,home_pts,away_pts\nAF1,20,15\n")
            build = os.path.join(d, "v14_model")
            os.makedirs(build)
            with mock.patch.object(vs, "_prior_candidates", lambda: [None, os.path.join(d, "none.csv"), table, build]):
                self.assertEqual(vs.default_prior(), table)
                self.assertEqual(vs._prior_tables(None), table)
            with mock.patch.object(vs, "_prior_candidates", lambda: [os.path.join(d, "none.csv"), build]):
                self.assertIsNone(vs.default_prior())               # an empty directory is no build
            with mock.patch.dict(os.environ, {"EAMF_V21_PRIOR": table}):
                self.assertEqual(vs._prior_candidates()[0], table)
                self.assertEqual(vs._prior_tables(), table)

    def test_the_prior_is_the_fit_under_the_build_s_layers(self):
        from .. import prior_check, v21_stream as vs

        class Fit:
            FOLLOWS_RESULTS = False

        class Layer:
            FOLLOWS_RESULTS = True             # v14's follow layer over an NB2 fit

            def __init__(self, pre):
                self.pre = pre

        build = type("Build", (), {"prematch_model": staticmethod(lambda _t: Layer(Layer(Fit())))})
        with mock.patch.object(prior_check, "version_of", lambda _d: "v14"), \
                mock.patch.object(prior_check, "_modules", lambda _v: (build, None)):
            pre = vs.prematch_model(os.path.join("v14_model", "v14tables.npz"))
        self.assertIsInstance(pre, Fit)        # the calibrator reads no history for it
        self.assertIsNone(vs.prematch_model(None))


class TestShape(unittest.TestCase):
    """The second strength lever (v21.SHAPE_TRAIL): the favourite's strength gap played by the score, so it
    wins more often without winning by more; off, the draws exactly as before."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, mode="tilt",
                          key="lean", blend=40.0, field=0.2)
        m.save(os.path.join(cls.tmp.name, v21.MODEL_FILE))
        cls.model = m
        cls.code = sorted(MATCHES)[0]
        cls.rows = sorted(MATCHES[cls.code], key=lambda r: int(r["message"]))
        cls.info = [h for h in HISTORY if h["MATCH_CODE"] == cls.code]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_name_picks_the_lever(self):
        from .. import v21_react as vr, v21_stream as vs
        self.assertEqual(vs.shape_of("v21-shape0.25"), 0.25)
        self.assertEqual(vs.shape_of("V21-SHAPE1"), 1.0)
        for name in ("v21", "v21-shape0", "v21-shape", "v21-shapex", "v21-snap4", "v21-avgf4"):
            self.assertIsNone(vs.shape_of(name), name)
        self.assertIsNone(vr.variant_of("v21-shape0.5"))              # no in-play update by that name

    def test_off_the_draws_are_as_before(self):
        start = v21.kickoff_start(self.model)
        n = len(start["period"])
        plain = v21.simulate(self.model, dict(start, nudge=np.full((n, 2), 0.5)), 200, seed=5)
        off = v21.simulate(self.model, dict(start, nudge=np.full((n, 2), 0.5), edge=np.full(n, 6.0), shape=np.zeros(n)),
                           200, seed=5)
        self.assertTrue(np.array_equal(plain[0], off[0]) and np.array_equal(plain[1], off[1]))
        a = v21.nudge_solve(self.model, start, (12.0, 9.0), 200, seed=5)
        b = v21.nudge_solve(self.model, start, (12.0, 9.0), 200, seed=5, shape=0.0)
        self.assertTrue(np.array_equal(a[0][0], b[0][0]) and np.array_equal(a[1], b[1]))

    def test_the_favourite_wins_more_often_by_no_more(self):
        start = v21.kickoff_start(self.model)
        h0, a0 = v21.simulate(self.model, start, 2000, seed=3)
        target = (h0.mean() + 3.0, a0.mean() - 3.0)
        got = {}
        for k in (0.0, 1.0):
            (h, a), _ = v21.nudge_solve(self.model, start, target, 3000, seed=3, shape=k)
            mg = (h - a).ravel()
            got[k] = (mg.mean(), mg.std(), np.mean(mg > 0), np.mean((mg >= 1) & (mg <= 7)))
        self.assertLess(abs(got[1.0][0] - got[0.0][0]), 1.0)           # the nudge still meets the expected points
        self.assertGreater(got[1.0][2], got[0.0][2] + 0.04)            # the favourite wins more often
        self.assertLess(got[1.0][1], got[0.0][1] - 1.0)                # by less: the margin tighter
        self.assertGreater(got[1.0][3], got[0.0][3] + 0.05)            # more close wins

    def test_the_stream_carries_it_in_play(self):
        from .. import v21_stream as vs

        class Prior:
            league = (15.0, 15.0)

            def means(self, rows, results=None):
                return {r["MATCH_CODE"]: (16.0, 10.0) for r in rows}

        prod = [(self.code, k, None, 50.0, 2.0, "", int(r["message"]), "OPEN", "true")
                for r in self.rows for k in (50, 51)]
        prod.append((self.code, 50, "2026-10-01 00:00:00", 50.0, 2.0, "", None, "OPEN", "true"))
        seen, real = [], v21.simulate

        def spy(model, start, n_paths, **kw):
            seen.append((len(start["period"]), "shape" in start, float(np.max(start.get("shape", [0.0])))))
            return real(model, start, n_paths, **kw)
        got = {}
        for shape in (None, 1.0):
            seen.clear()
            with mock.patch.object(vs, "prematch_model", lambda _t: Prior()), \
                    mock.patch.object(vs, "_prior_tables", lambda _d=None: "prior"), \
                    mock.patch.object(v21, "simulate", spy):
                got[shape] = vs.quotes_for_matches({self.code: self.rows}, prod, self.tmp.name, n_paths=60, workers=1,
                                                   match_info=self.info, lines=vs.PROD_LINES, nudge_share=1.0, shape=shape)
            calls = list(seen)
            if shape:
                self.assertTrue(all(on and k == 1.0 for _, on, k in calls))   # the kick-off solve and the states
                self.assertTrue(any(n > 1 for n, _, _ in calls))
            else:
                self.assertFalse(any(on for _, on, _ in calls))
        live = lambda q: {(x[1], x[6]): x[3] for x in q if x[6] is not None}
        self.assertNotEqual(live(got[None]), live(got[1.0]))


class TestPace(unittest.TestCase):
    """Pace when ahead (v21_pace): each gamer's own time tilt while its side is ahead."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.hist = copy.deepcopy(HISTORY)
        for i, h in enumerate(cls.hist):                                 # the matches over ten days
            h["SCHEDULED_START_TIME_UTC"] = f"2026-10-{1 + i % 10:02d} 12:00:00"
        with contextlib.redirect_stdout(io.StringIO()):
            m = v21.build(MATCHES, cls.hist, min_records=20, verbose=False, refine=False, mode="tilt", key="lean",
                          blend=40.0, field=0.2, decision_blend=25.0, clock=0.2)
        m.save(os.path.join(cls.tmp.name, v21.MODEL_FILE))
        cls.model = m

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _seconds(self, pace, n=2000):
        start = v21.kickoff_start(self.model, ("G0", "G1"), ("T0", "T1"))
        st = dict(start, pace=np.tile(np.asarray(pace, float), (len(start["period"]), 1)))
        stats = {}
        got = v21.simulate(self.model, st, n, seed=5, stats=stats)
        S = stats["_snaps"]
        cat = lambda k: np.concatenate([x[k] for x in S])
        return got, cat(4), cat(10), cat(12)

    def test_off_the_draws_are_as_before(self):
        start = v21.kickoff_start(self.model, ("G0", "G1"), ("T0", "T1"))
        a = v21.simulate(self.model, start, 300, seed=4)
        b = v21.simulate(self.model, dict(start, pace=np.zeros((len(start["period"]), 2))), 300, seed=4)
        for x, y in zip(a, b):
            self.assertTrue(np.array_equal(x, y))

    def test_a_side_takes_its_own_time_only_while_ahead(self):
        _, sec0, side0, lead0 = self._seconds((0.0, 0.0))
        _, sec1, side1, lead1 = self._seconds((3.0, 0.0))
        mean = lambda sec, side, lead, sd, up: sec[(side == sd) & ((lead >= 1) if up else (lead <= 0))].mean()
        self.assertGreater(mean(sec1, side1, lead1, 0, True), mean(sec0, side0, lead0, 0, True) + 1.0)
        self.assertLess(abs(mean(sec1, side1, lead1, 0, False) - mean(sec0, side0, lead0, 0, False)), 0.5)
        self.assertLess(abs(mean(sec1, side1, lead1, 1, True) - mean(sec0, side0, lead0, 1, True)), 0.5)

    def test_the_fit_finds_a_planted_pace(self):
        from .. import v21_pace as vp
        rng = np.random.default_rng(2)
        B, n, G = 12, 12000, 6
        phi = np.linspace(0.3, 4.0, B)[:, None]
        planted = np.array([-0.4, -0.2, 0.0, 0.0, 0.2, 0.4])
        gamer = rng.integers(0, G, n)
        lead = rng.integers(-10, 11, n).astype(float)
        H = np.full((n, B), 1.0 / B)
        d = np.where(lead >= vp.PACE_LEAD, 0.3 + planted[gamer], 0.0)[:, None]
        p = vp._probs(H, phi, d)
        obs = np.array([rng.choice(B, p=row) for row in p])
        frame = dict(H=H, obs=obs, phi=phi, gamer=gamer, lead=lead, code=np.arange(n) // 20, weights=np.ones(n))
        stub = type("Build", (), {"gamers": {f"G{i}": i for i in range(G)}, "tilt_prior": 200.0})()
        delta, gain, n_up, league = vp.fit(stub, frame)
        self.assertAlmostEqual(league, 0.3, delta=0.1)
        self.assertGreater(np.corrcoef(delta[:G, 0], planted)[0, 1], 0.95)
        self.assertEqual(delta[G, 0], league)                            # a gamer it has not met: the league's
        self.assertGreater(gain, 0)
        got = vp.scored(frame, delta)
        self.assertEqual(len(got["ll"]), int((lead >= vp.PACE_LEAD).sum()))

    def test_the_fit_command_writes_the_table_the_stream_reads(self):
        from .. import __main__ as cli, v21_pace as vp, v21_stream as vs
        d = self.tmp.name
        snaps, history = os.path.join(d, "snaps.csv"), os.path.join(d, "history.csv")
        rows = [r for c in sorted(MATCHES) for r in MATCHES[c]]
        with open(snaps, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        hist = [dict(h, SPORT_CODE="AF", STREAM_NUMBER="1") for h in self.hist]
        with open(history, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, sorted(hist[0]))
            w.writeheader()
            w.writerows(hist)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            cli.main(["v21-pace-fit", "--model", d, "--snapshots", snaps, "--history", history, "--until", "2026-10-07"])
        text = out.getvalue()
        for part in ("pace when ahead", "fitted on", "on the 32 matches after", "log-likelihood gained", "pace_ahead.json"):
            self.assertIn(part, text)
        table = vp.load(d)
        self.assertIn("G0", table)
        self.assertEqual(vp.side_pace(table, ("G0", "NOBODY"))[1], table[""])      # the league's part
        self.assertTrue(vs.pace_of("v21-pace"))
        self.assertFalse(vs.pace_of("v21") or vs.pace_of("v21-shape0.5"))
        self.assertEqual(vs.pace_table(d), table)
        with tempfile.TemporaryDirectory() as bare:
            self.model.save(os.path.join(bare, v21.MODEL_FILE))
            with self.assertRaises(SystemExit):
                vs.pace_table(bare)

    def test_the_stream_prices_with_it(self):
        from .. import v21_pace as vp, v21_stream as vs
        code = sorted(MATCHES)[0]
        rows = sorted(MATCHES[code], key=lambda r: int(r["message"]))
        info = [h for h in self.hist if h["MATCH_CODE"] == code]
        lines = ((50, ""), (54, "Total points over 40.5"))
        prod = [(code, k, None, 50.0, 2.0, line, int(r["message"]), "OPEN", "true") for r in rows for k, line in lines]
        with tempfile.TemporaryDirectory() as d:
            self.model.save(os.path.join(d, v21.MODEL_FILE))
            pair = v21.match_people(code, rows, {h["MATCH_CODE"]: h for h in self.hist})[0]
            vp.write(os.path.join(d, vp.PACE_FILE), dict(gamers={pair[0]: 3.0, pair[1]: -3.0}, league=0.0))
            price = lambda pace: vs.quotes_for_matches({code: rows}, prod, d, n_paths=60, workers=1, match_info=info,
                                                       lines=vs.PROD_LINES, nudge_share=0.0, pace=pace)
            off, on = price(False), price(True)
        self.assertEqual(len(off), len(on))
        self.assertNotEqual([x[3] for x in off], [x[3] for x in on])


class TestReact(unittest.TestCase):
    """v2.1's in-play strength update (v21_react): the variants by name, each side's nudge moved the way
    its play says, the stream's prices untouched without one, and v21-react-fit's table."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with contextlib.redirect_stdout(io.StringIO()):
            m = v21.build(MATCHES, HISTORY, min_records=20, verbose=False, refine=False, mode="tilt",
                          key="lean", blend=40.0, field=0.2, decision_blend=25.0, clock=0.2)
        m.save(os.path.join(cls.tmp.name, v21.MODEL_FILE))
        cls.model = m
        cls.code = sorted(MATCHES)[0]
        cls.rows = sorted(MATCHES[cls.code], key=lambda r: int(r["message"]))
        cls.info = [h for h in HISTORY if h["MATCH_CODE"] == cls.code]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    COEF = (0.5, 0.3, 1.0, 1.5, -0.1, -0.3, -0.6, 0.0)          # a win-probability table of the right shape

    def _prod(self):
        c = self.code
        out = [(c, k, "2026-10-01 00:00:00", 50.0, 2.0, line, None, "OPEN", "true")
               for k, line in ((50, ""), (52, "PLAYER 1 to score over 3.5 points more than PLAYER 2"),
                               (54, "Total points over 40.5"))]
        for r in self.rows:
            for k in (50, 51):
                out.append((c, k, None, 50.0, 2.0, "", int(r["message"]), "OPEN", "true"))
        return out

    def _quotes(self, react):
        from .. import v21_stream as vs
        return vs.quotes_for_matches({self.code: self.rows}, self._prod(), self.tmp.name, n_paths=40, workers=1,
                                     match_info=self.info, lines=vs.PROD_LINES, nudge_share=0.0, react=react)

    def test_the_variants_by_name(self):
        from .. import v21_react as vr, v21_stream as vs
        self.assertEqual(vr.variant_of("v21-snap4"), ("snap", 4.0))
        self.assertEqual(vr.variant_of("v21-pts2.5"), ("pts", 2.5))
        self.assertEqual(vr.variant_of("v21-wpa10"), ("wpa", 10.0))
        for name in ("v21", "v21-glmer", "v21-snap0", "v21-snap", "v21-snapx"):
            self.assertIsNone(vr.variant_of(name), name)
        self.assertEqual(vs.react_of("v21-snap8"), ("snap", 8.0))
        self.assertTrue(all(vr.spec_of(v) for v in vr.VARIANTS))

    def test_no_update_prices_as_before(self):
        from .. import v21_react as vr
        before = self._quotes(None)
        with mock.patch.object(vr, "updates", side_effect=AssertionError("no update without a variant")):
            again = self._quotes(None)
        self.assertEqual(before, again)

    def test_an_update_moves_the_in_play_prices_not_the_pre_match(self):
        base, moved = self._quotes(None), self._quotes(("snap", 8.0))
        pre = lambda q: sorted(x for x in q if x[6] is None)
        live = lambda q: {(x[1], x[6]): x[3] for x in q if x[6] is not None}
        self.assertEqual(pre(base), pre(moved))
        a, b = live(base), live(moved)
        self.assertEqual(set(a), set(b))
        self.assertTrue(any(abs(a[k] - b[k]) > 1e-9 for k in a))

    def test_the_stronger_side_is_pushed_up(self):
        import random
        from .. import v21_react as vr, v21_stream as vs
        from .fakes import _fake_match
        rng = random.Random(11)
        got = {}
        for strength in ((0.6, 0.0), (0.0, 0.6)):
            ends = []
            for k in range(10):
                rows = [vs._as_text(r) for r in _fake_match(f"AFX{k}", rng, strength=strength)]
                starts = vs.state_starts(self.model, rows)
                ends.append(vr.updates(self.model, rows, starts, None, ("snap", 4.0))[-1])
            got[strength] = np.mean(ends, axis=0)
        self.assertGreater(got[(0.6, 0.0)][0], got[(0.6, 0.0)][1] + 1.0)
        self.assertGreater(got[(0.0, 0.6)][1], got[(0.0, 0.6)][0] + 1.0)

    def test_a_tighter_prior_moves_it_less(self):
        from .. import v21_react as vr, v21_stream as vs
        rows = [vs._as_text(r) for r in self.rows]
        starts = vs.state_starts(self.model, rows)
        tight, loose = (np.abs(vr.updates(self.model, rows, starts, None, ("snap", t))).sum() for t in (1.0, 8.0))
        self.assertLess(tight, loose)
        self.assertTrue(np.all(np.abs(vr.updates(self.model, rows, starts, np.full(2, 9.5), ("snap", 50.0))
                                      + 9.5) <= v21.NUDGE_MAX + 1e-9))

    def test_the_points_update(self):
        from .. import v21_react as vr
        st = dict(period=np.array([3]), clock=np.array([240.0]), home=np.array([20]), away=np.array([7]))
        kick = dict(mean=(14.0, 14.0), var=(50.0, 50.0))
        f = vr.elapsed_share(self.model, 3, 240.0)
        self.assertTrue(0.3 < f < 0.8)
        d = vr.points_updates(self.model, [st], None, 4.0, kick)[0]
        self.assertAlmostEqual(d[0], 16 * (20 - 14 * f) / (50 + 16 * f))
        self.assertAlmostEqual(d[1], 16 * (7 - 14 * f) / (50 + 16 * f))
        self.assertGreater(d[0], 0)
        self.assertLess(d[1], 0)

    def test_the_win_chance_a_snap_added(self):
        from .. import v21_react as vr, v21_stream as vs
        rows = [vs._as_text(r) for r in self.rows]
        s = vr.wp_state(rows[5], 0.0)
        self.assertAlmostEqual(vr.wp_of(self.COEF, s, 0) + vr.wp_of(self.COEF, s, 1), 1.0)
        got = vr.snap_wpa(rows, self.COEF)
        self.assertTrue(got)
        recs = {r["message"]: r for r in vr._snaps(rows, ("", ""), ("", ""))}
        tds = [(side, add) for side, msg, add in got
               if recs[msg]["kind"] == v21.GAIN and recs[msg]["gain"] >= 100 - recs[msg]["field"]]
        self.assertTrue(tds)
        self.assertTrue(all(add > 0 for _, add in tds))              # a touchdown adds to its side's chance
        W = vr.wpa_sums(rows, [int(rows[-1]["message"])], self.COEF)
        self.assertEqual(W.shape, (1, 2))

    def test_wpa_needs_its_table(self):
        from .. import v21_react as vr
        with self.assertRaises(SystemExit):
            self._quotes(("wpa", 10.0))
        path = os.path.join(self.tmp.name, vr.WP_FILE)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"coef": list(self.COEF)}, fh)
        try:
            self.assertEqual(vr.load_wp(self.tmp.name), self.COEF)
            q = self._quotes(("wpa", 10.0))
            self.assertTrue(q)
        finally:
            os.remove(path)

    def test_gamer_by_gamer(self):
        from .. import v21_react as vr, v21_stream as vs
        self.assertEqual(vr.variant_of("v21-gsnap2"), ("gsnap", 2.0))
        self.assertEqual(vr.variant_of("v21-gwpa1"), ("gwpa", 1.0))
        table = {"snap": {"setting": 4.0, "k": 0.5, "gamers": {"G1": 2.0, "G2": 0.0}}}
        self.assertEqual(list(vr.gamer_scale(table, "snap", ("G1", "nobody"))), [2.0, 0.5])
        rows = [vs._as_text(r) for r in self.rows]
        starts = vs.state_starts(self.model, rows)
        base = vr.updates(self.model, rows, starts, None, ("snap", 4.0), ("G1", "G2"))
        own = vr.updates(self.model, rows, starts, None, ("gsnap", 4.0), ("G1", "G2"), gamers=table)
        self.assertTrue(np.allclose(own[:, 0], np.clip(2.0 * base[:, 0], -v21.NUDGE_MAX, v21.NUDGE_MAX)))
        self.assertTrue(np.allclose(own[:, 1], 0.0))
        with self.assertRaises(ValueError):
            vr.updates(self.model, rows, starts, None, ("gsnap", 4.0), ("G1", "G2"))
        with self.assertRaises(SystemExit):                        # priced without the gamers' table
            self._quotes(("gsnap", 4.0))
        path = os.path.join(self.tmp.name, vr.GAMERS_FILE)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(table, fh)
        try:
            self.assertEqual(vr.load_gamers(self.tmp.name), table)
            self.assertTrue(self._quotes(("gsnap", 4.0)))
        finally:
            os.remove(path)

    def test_the_fit_writes_the_table_and_scores_the_updates(self):
        import datetime as dt
        import random
        from .. import v21_react as vr, v21_react_fit as fit
        from .fakes import _fake_match
        rng = random.Random(4)
        data, hist = {}, []
        for i in range(30):
            code = f"AF{i:03d}{(dt.date(2026, 9, 1) + dt.timedelta(days=i // 3)):%d%m%y}"
            rows = _fake_match(code, rng, strength=(0.3 * (i % 3 - 1), 0.0))
            data[code] = rows
            hist.append(dict(MATCH_CODE=code, PLAYER_1_HANDLE="", PLAYER_2_HANDLE="", PLAYER_1_TEAM="",
                             PLAYER_2_TEAM="", PLAYER_1_FINAL_SCORE=rows[-1]["final_p1"],
                             PLAYER_2_FINAL_SCORE=rows[-1]["final_p2"]))
        lines, got = fit.run(data, hist, os.path.join(self.tmp.name, v21.MODEL_FILE), None, dt.date(2026, 10, 1),
                             test_share=0.4, n_paths=20, workers=1)
        self.assertEqual(len(got["coef"]), len(vr.WP_FEATURES))
        self.assertGreater(got["coef"][1], 0)                          # ahead, likelier to win
        for v in ("snap4", "pts4", "wpa10", "csnap4", "cpts4", "cwpa10"):
            self.assertIn(v, got["scores"])
            self.assertIn("open", got["scores"][v])
            self.assertTrue(any(line.lstrip().startswith(v + " ") for line in lines), v)
        out = fit.write(os.path.join(self.tmp.name, "fit", vr.WP_FILE), got)
        self.assertEqual(vr.load_wp(os.path.dirname(out)), tuple(got["coef"]))
        self.assertTrue(any("gamer by gamer" in line for line in lines))

    def test_the_sessions_and_the_carry(self):
        from .. import v21_react_fit as fit
        hist = [dict(MATCH_CODE=f"M{i}", PLAYER_1_HANDLE="A", PLAYER_2_HANDLE=f"B{i}",
                     SCHEDULED_START_TIME_UTC=t) for i, t in enumerate(
                         ("2026-09-01 10:00:00", "2026-09-01 11:00:00", "2026-09-01 12:00:00", "2026-09-01 20:00:00"))]
        sess = fit.sessions(hist)
        self.assertEqual(sess[("M2", "A")], ["M1", "M0"])               # latest first
        self.assertEqual(sess[("M3", "A")], [])                          # eight hours on: a new session
        c = fit.carried([[1.0, 2.0, 3.0, 0.1], [1.0, 2.0, 3.0, 0.1]])
        self.assertTrue(np.allclose(c, np.array([1, 2, 3, 1, 0.1]) * (1 + fit.CARRY_DECAY)))


def _with_timeouts(matches, directory):
    """The matches with timeout calls placed (playover.annotate_timeouts), so the timeout decisions
    have real ones: every third snap of Q2's and Q4's last 2:00, the sides in turn."""
    from .. import playover
    out = copy.deepcopy(matches)
    path = os.path.join(directory, "timeouts.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_code", "message", "period", "caller", "prev_play"])
        for code, rows in sorted(out.items()):
            late = [r for r in rows if r["play_kind"] == "SCRIMMAGE" and r["period"] in ("2", "4")
                    and float(r["clock_seconds"] or 0) <= 120]
            for k, r in enumerate(late[::3][:4]):
                w.writerow([code, r["message"], r["period"], ("TEAM_A", "TEAM_B")[k % 2], ""])
    playover.annotate_timeouts(out, path)
    return out


class TestSpeedups(unittest.TestCase):
    """The exact speedups (TILT_CACHE, DECISION_TABLE): the same draws bit for bit."""

    def test_on_and_off_the_sim_gives_the_same_games(self):
        with tempfile.TemporaryDirectory() as d:
            matches = _with_timeouts(MATCHES, d)
        hist = _history(matches)
        with contextlib.redirect_stdout(io.StringIO()):
            m = v21.build(matches, hist, mode="tilt", key="lean", blend=40.0, field=0.2, decision_blend=25.0,
                          clock=0.2, min_records=20, verbose=False, refine=False)
        self.assertGreater(m.choices["to_def"].n, 0)                     # the timeout decisions have data
        code = sorted(matches)[0]
        pair, teams = v21.match_people(code, matches[code], {r["MATCH_CODE"]: r for r in hist})
        from .. import v21_stream as vs
        starts = vs.state_starts(m, [vs._as_text(r) for r in matches[code]], pair, teams)
        S = v21.concat_starts([s for _, s in starts])
        S["nudge"] = np.tile([0.5, -0.3], (len(S["period"]), 1))

        def runs():
            (h, a), alpha = v21.nudge_solve(m, v21.kickoff_start(m, pair, teams), (24.0, 18.0), 60, seed=4, share=0.75)
            return [h, a, alpha, *v21.simulate(m, S, 40, seed=5), *v21.simulate(m, dict(starts[0][1]), 80, seed=6)]

        with mock.patch.multiple(v21, TILT_CACHE=False, DECISION_TABLE=False):
            off = runs()
        with mock.patch.object(v21, "TILT_CACHE_PATHS", 0):                # the tilt table in every run
            on = runs()
            again = runs()                                                 # off the tables kept
        for x, y, z in zip(off, on, again):
            self.assertTrue(np.array_equal(x, y) and np.array_equal(x, z))
        self.assertTrue(m.dist.__dict__.get("_tiltz") and m.time.__dict__.get("_tiltz"))
        for name in ("to_def", "to_off", "fourth", "kickoff", "two"):
            T = m.choices[name].__dict__.get("_table")
            self.assertTrue(T is not None and (~np.isnan(T[:, 0])).any(), name)
        with tempfile.TemporaryDirectory() as d:                           # the tables stay out of the build
            m.save(os.path.join(d, v21.MODEL_FILE))
            back = v21.Model.load(os.path.join(d, v21.MODEL_FILE))
        self.assertFalse(any(k.startswith("_tilt") for k in back.dist.__dict__))
        self.assertFalse(any("_table" in ch.__dict__ for ch in back.choices.values()))


class TestTiming(unittest.TestCase):

    def test_a_sample_priced_as_the_stream_prices_it_timed_and_projected(self):
        from .. import v21_timing as T
        model = built()
        codes = sorted(MATCHES)[:4]
        with tempfile.TemporaryDirectory() as d:
            model.save(os.path.join(d, v21.MODEL_FILE))
            out = T.run(d, {c: MATCHES[c] for c in codes}, HISTORY, n_matches=2, n_paths=20, workers=3)
        self.assertEqual(len(out["per"]), 2)
        self.assertEqual(out["window"], 4)
        for _, t in out["per"]:
            self.assertGreater(t["states"], 0)
            self.assertEqual(t["play_games"], 20 * t["states"])
            self.assertEqual(t["kick_games"], 2 * 20)                # not a tilt build: one kick-off run
            self.assertGreater(t["play_snaps"], 0)
        text = "\n".join(T.report(out))
        for part in ("a match on average", "on 3 workers", "one live price update", "simulate's parts",
                     "the distance draw"):
            self.assertIn(part, text)


class TestCoverage(unittest.TestCase):

    def test_every_real_state_s_draws_and_the_thinnest_situations(self):
        from .. import v21_coverage as C
        model = built()
        codes = sorted(MATCHES)[:10]
        with tempfile.TemporaryDirectory() as d:
            model.save(os.path.join(d, v21.MODEL_FILE))
            out = C.run(d, {c: MATCHES[c] for c in codes}, HISTORY)
            lines = C.write(out, os.path.join(d, "cov"))
            self.assertEqual(sorted(os.listdir(os.path.join(d, "cov"))),
                             ["coverage.txt", "coverage_decisions.csv", "coverage_states.csv"])
        n = sum(len([r for r in v21.match_snaps(MATCHES[c]) if r["period"] <= 4 + v21.MAX_OT]) for c in codes)
        self.assertEqual(len(out["states"]), n)
        self.assertTrue(all(x["dist_level"] >= 0 and x["dist_n"] >= 20 for x in out["states"]))
        text = "\n".join(lines)
        for part in ("by the part of the game and the lead", "by the snaps left in the half", "the thinnest situations",
                     "the decisions"):
            self.assertIn(part, text)
        kind = C.level_kind(model.dist.levels, np.array([False, True]))
        self.assertEqual(int(kind(0, False)), 0)
        k_end = [lv for lv, f in enumerate(model.dist.levels) if f == ("tb", "down", "dist", "zone", "endgame3")][0]
        self.assertEqual([int(kind(k_end, False)), int(kind(k_end, True))], [2, 1])   # the lead only at the end


class TestCli(unittest.TestCase):

    def test_build_and_report(self):
        from .. import __main__ as cli
        with tempfile.TemporaryDirectory() as d:
            snaps = os.path.join(d, "snaps.csv")
            rows = [r for c in sorted(MATCHES) for r in MATCHES[c]]
            keys = list(rows[0])
            with open(snaps, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            out = os.path.join(d, "v21")
            quiet = contextlib.redirect_stdout(io.StringIO())
            with quiet:
                cli.main(["v21-build", snaps, "--out", out, "--min-records", "20", "--no-refine",
                          "--history", os.path.join(d, "none.csv"), "--no-timeouts"])
            self.assertTrue(os.path.exists(os.path.join(out, v21.MODEL_FILE)))
            codes = sorted(MATCHES)[:6]
            means = {c: (9.0, 8.0) for c in codes}
            with mock.patch("eAMFModel.strength_check.history_rows", return_value=HISTORY[:6]), \
                    mock.patch("eAMFModel.v21_report.prior_inputs", return_value=(means, {})), \
                    contextlib.redirect_stdout(io.StringIO()):
                cli.main(["v21-report", "--model", out, "--prior-model", "v14_model", "--snapshots", snaps, "--since", "2026-10-01",
                          "--paths", "40", "--state-paths", "10", "--every", "8", "--workers", "1",
                          "--out", os.path.join(d, "rep")])
            for f in ("v21_report.txt", "v21_matches.csv", "v21_states.csv"):
                self.assertTrue(os.path.exists(os.path.join(d, "rep", f)))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["v21-sweep", snaps, "--history", os.path.join(d, "none.csv"), "--out", os.path.join(d, "sw")])
            self.assertTrue(os.path.exists(os.path.join(d, "sw", "v21_sweep.txt")))


if __name__ == "__main__":
    unittest.main()
