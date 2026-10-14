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

from .. import prior_check, v21, v21_report
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


def _state(model, period=1, clock=240.0, team=0, down=1, dist=10, y=25, prev=v21.FRESH, home=0, away=0):
    return dict(period=np.array([period]), clock=np.array([clock]), team=np.array([team]),
                phase=np.array([v21.SNAP]), down=np.array([down]), dist=np.array([dist]), y=np.array([y]),
                home=np.array([home]), away=np.array([away]), prev=np.array([prev]), elapsed=np.array([0]),
                tos=np.array([[3, 3]]), kick2=np.array([1]), gamer=np.array([[len(model.gamers)] * 2]),
                team_ix=np.array([[len(model.teams)] * 2]))


def _fixed_time(model, seconds):
    """Every time draw the same."""
    for b in [model.time] + [v for k, v in model.own.items() if k[2] == "time"]:
        b.payload["seconds"][:] = seconds
        b.payload["cut"][:] = False


def _fixed_play(model, gain, td):
    for b in [model.dist] + [v for k, v in model.own.items() if k[2] == "dist"]:
        b.payload["kind"][:] = v21.GAIN
        b.payload["gain"][:] = gain
        b.payload["td"][:] = td


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
        m.gain_surv = np.ones(101)                                         # a long gain always goes on
        st = {}
        v21.simulate(m, _state(m, y=20), 20, stats=st)
        log = v21.snap_log(st)
        self.assertEqual(log["gain"][np.flatnonzero(log["path"] == 0)[0]], 80)
        m.gain_surv = np.r_[np.ones(31), np.zeros(70)]                     # nothing passes 30 yards
        st = {}
        v21.simulate(m, _state(m, y=20), 20, stats=st)
        log = v21.snap_log(st)
        self.assertEqual(log["gain"][np.flatnonzero(log["path"] == 0)[0]], 30)

    def test_overtime_ends_when_the_second_side_goes_ahead(self):
        m = built()
        _fixed_time(m, 7.0)
        _fixed_play(m, 99, True)                                           # every snap a touchdown
        m.gain_surv = np.ones(101)
        m.two_p[:] = 0.0
        m.two_good = m.kick_good = 1.0
        m.kicks = {False: (np.array([False]), np.array([25]), np.array([4.0]))}
        st = {}
        h, a = v21.simulate(m, _state(m, period=5, clock=240.0, team=0, y=50, home=14, away=14), 20, stats=st)
        np.testing.assert_array_equal(h, 21)                               # 6 and the kick
        np.testing.assert_array_equal(a, 22)                               # 6 and the forced two
        self.assertEqual(st["unfinished"], 0)


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
        for part in ("KICK-OFF", "GRADED", "EACH STEP", "THE DRAWS", "by team"):
            self.assertIn(part, lines)
        real = v21_report.real_census({c: MATCHES[c] for c in sorted(MATCHES)[:12]})
        self.assertEqual(real["events"]["snaps"], sum(len(v21.match_snaps(MATCHES[c])) for c in sorted(MATCHES)[:12]))
        self.assertEqual(int(real["time"].sum()), real["events"]["snaps"])

    def test_steps(self):
        self.assertEqual(list(v21_report.step_of([1, 1, 2, 4, 5], [200, 100, 121, 120, 50])), [0, 1, 2, 7, 8])


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


if __name__ == "__main__":
    unittest.main()
