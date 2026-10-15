"""v2.1: the draws (backoff groups, tilts, the censored time), the real snaps' features, the build in
both modes, the sim's bounds (quarter end, two-minute warning, end zones, overtime) and the report."""

import contextlib
import copy
import csv
import io
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
        # a thin window widens to the FIELD_MIN nearest
        lo, hi = b.near(np.array([0]), np.array([99]), 0.0)
        self.assertEqual(int(hi[0] - lo[0]), v21.FIELD_MIN)
        self.assertGreaterEqual(int(b.payload["field"][b.by_field()[0][lo[0]]]), 90)
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

    def test_the_distance_draws_never_let_the_down_go(self):
        # a blended draw passes weight down the chain: a level without the down would draw a 4th down
        # from every down's snaps
        for levels in (v21.DIST_LEVELS, v21.DIST_LEAN_LEVELS):
            for lv in levels[:-1]:
                self.assertIn("down", lv)
                self.assertIn("tb", lv)                    # the time the play took, all six buckets

    def test_field_windows(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            m = v21.build(MATCHES, HISTORY, min_records=20, verbose=True, refine=False, mode="adjusted",
                          key="lean", blend=40.0, field=0.2)
        self.assertEqual(m.field, 0.2)
        self.assertEqual(m.meta["field"], 0.2)
        text = out.getvalue()
        self.assertIn("every snap by the field", text)
        self.assertIn("the sim draws from near it", text)
        self.assertIn("opp 19-11", text)
        h, a = v21.simulate(m, v21.kickoff_start(m), 30)
        self.assertGreater((h + a).mean(), 10)
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
        st = {}
        h, a = v21.simulate(m, _state(m, period=5, clock=240.0, team=0, y=50, home=14, away=14), 20, stats=st)
        np.testing.assert_array_equal(h, 21)                               # 6 and the kick
        np.testing.assert_array_equal(a, 22)                               # 6 and the forced two
        self.assertEqual(st["unfinished"], 0)


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
        self.assertEqual(got["conv"], [(True, True)])

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
        for part in ("GAME FLOW", "ONE SNAP", "3rd down, by distance", "4th DOWN", "THE LARGEST GAPS", "OVERTIME",
                     "ONE SNAP by the snaps left", "ONE SNAP in Q4's last 2:00", "Q4 STATES", "Q4 STATES, level",
                     "by what a field goal does", "CONVERSIONS in Q4's last 3:00", "by the time its play took",
                     "the league's share in the sim's cells", "how the drive with the ball ended"):
            self.assertIn(part, lines)


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
                cli.main(["v21-report", "--model", out, "--snapshots", snaps, "--since", "2026-10-01",
                          "--paths", "40", "--state-paths", "10", "--every", "8", "--workers", "1",
                          "--out", os.path.join(d, "rep")])
            for f in ("v21_report.txt", "v21_matches.csv", "v21_states.csv"):
                self.assertTrue(os.path.exists(os.path.join(d, "rep", f)))
            with contextlib.redirect_stdout(io.StringIO()):
                cli.main(["v21-sweep", snaps, "--history", os.path.join(d, "none.csv"), "--out", os.path.join(d, "sw")])
            self.assertTrue(os.path.exists(os.path.join(d, "sw", "v21_sweep.txt")))


if __name__ == "__main__":
    unittest.main()
