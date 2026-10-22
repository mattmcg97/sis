"""inplay_wp.py's reading of the export and its walk-forward, on hand-made rows and the repo's own fixtures.

    PYTHONDONTWRITEBYTECODE=1 python -m unittest test_inplay_wp      (from this folder; reads /home/user/sis)
"""

import os
import random
import sys
import unittest

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/user/sis"
sys.path.insert(0, HERE)
sys.path.insert(1, REPO)
sys.dont_write_bytecode = True

import inplay_wp as W  # noqa: E402


def frame(rows):
    ex = pd.DataFrame(rows)
    for c in W.WANT:
        if c not in ex:
            ex[c] = np.nan
    for c in ("match_code", "file_time", "play_kind", "team_a_side", "offense", "play_messages", "opening_offense"):
        ex[c] = ex[c].astype(object).where(ex[c].notna(), None)
        ex[c] = ex[c].map(lambda v: None if v is None or v == "" else str(v))
    ex["message"] = pd.to_numeric(ex.message)
    return ex.sort_values(["match_code", "message"]).reset_index(drop=True)


def row(code, msg, kind, offense, down="", dist="", field="", clock=200, period=1, messages="", board=(0, 0),
        start=None, final=(7, 3), side="home", used=(0, 0)):
    start = board if start is None else start
    return dict(match_code=code, message=msg, period=period, clock_seconds=clock, play_kind=kind, team_a_side=side,
                offense=offense, down=down, distance=dist, field_position=field, play_messages=messages,
                final_p1=final[0], final_p2=final[1], score_p1=board[0], score_p2=board[1],
                score_p1_at_start=start[0], score_p2_at_start=start[1], opening_offense="TEAM_A",
                timeouts_used_a=used[0], timeouts_used_b=used[1], prob_50=0.6, live_50=1,
                file_time=f"2026-10-01 00:{msg:02d}:00")


class TestStates(unittest.TestCase):

    def states(self, rows, pre=0.6):
        ex = frame(rows)
        st, _ = W.build_states(ex, {c: pre for c in ex.match_code.unique()}, {})
        return st.set_index("message")

    def test_scoring_rows(self):
        # TEAM_A is home: a touchdown whose board lags, the kick after it, the kick-off, a field goal by TEAM_B
        rows = [row("M", 1, "KICKOFF", "TEAM_A", 1, 10, 25, 236),
                row("M", 2, "SCRIMMAGE", "TEAM_A", 1, 10, 60, 220),
                row("M", 3, "TOUCHDOWN", "TEAM_A", 1, 10, 97, 210, messages="TOUCHDOWN_TEAM_A", board=(0, 0)),
                row("M", 4, "CONVERSION", "TEAM_A", clock=210, messages="EXTRA_POINT_GOOD_TEAM_A", board=(6, 0)),
                row("M", 5, "KICKOFF", "TEAM_B", 1, 10, 30, 205, messages="KICKOFF_TEAM_A|POSSESSION_TEAM_B", board=(7, 0)),
                row("M", 6, "FIELD_GOAL", "TEAM_B", 1, 10, 35, 150, messages="FIELD_GOAL_GOOD_TEAM_B", board=(7, 0)),
                row("M", 7, "KICKOFF", "TEAM_A", 1, 10, 22, 146, board=(7, 3)),
                row("M", 8, "SAFETY", "TEAM_A", clock=120, messages="SAFETY_AWARDED_TEAM_B", board=(7, 3)),
                row("M", 9, "FIELD_GOAL", "TEAM_A", 1, 10, 70, 100, messages="FIELD_GOAL_MISSED_TEAM_B", board=(7, 5)),
                row("M", 10, "SCRIMMAGE", "TEAM_A", 2, 4, 76, 90, board=(7, 5), used=(1, 2))]
        s = self.states(rows)
        self.assertEqual((s.loc[3, "kind"], s.loc[3, "home_score"], s.loc[3, "pos_home"]), ("conv", 6, -1))
        self.assertEqual((s.loc[4, "kind"], s.loc[4, "home_score"], s.loc[4, "pos_home"]), ("kick", 7, -1))
        self.assertEqual((s.loc[5, "kind"], s.loc[5, "pos_home"], s.loc[5, "ytg"]), ("snap", -1, 70))
        self.assertEqual((s.loc[6, "kind"], s.loc[6, "away_score"], s.loc[6, "pos_home"]), ("kick", 3, 1))
        self.assertEqual((s.loc[8, "kind"], s.loc[8, "away_score"], s.loc[8, "pos_home"]), ("kick", 5, -1))
        # a missed kick by TEAM_B with TEAM_A's snap on the row: TEAM_A's snap from its 70
        self.assertEqual((s.loc[9, "kind"], s.loc[9, "pos_home"], s.loc[9, "ytg"]), ("snap", 1, 30))
        self.assertEqual((s.loc[10, "to_home"], s.loc[10, "to_away"]), (2, 1))
        self.assertEqual(s.loc[10, "lead_home"], 2)
        self.assertEqual(s.loc[2, "secs_left"], 3 * 240 + 220)
        self.assertTrue((s.y == 1).all())

    def test_team_a_away(self):
        # the same game with TEAM_A the away side: home's view must not change
        home = [row("M", 1, "SCRIMMAGE", "TEAM_A", 1, 10, 40, 200, board=(3, 0), used=(1, 0)),
                row("M", 2, "TOUCHDOWN", "TEAM_B", 1, 10, 97, 190, messages="TOUCHDOWN_TEAM_B", board=(3, 0))]
        away = [dict(r, team_a_side="away",
                     offense={"TEAM_A": "TEAM_B", "TEAM_B": "TEAM_A"}[r["offense"]],
                     play_messages=r["play_messages"].replace("TEAM_B", "TEAM_A"),
                     timeouts_used_a=r["timeouts_used_b"], timeouts_used_b=r["timeouts_used_a"],
                     opening_offense="TEAM_B") for r in home]
        a, b = self.states(home), self.states(away)
        for c in ("lead_home", "pos_home", "home_score", "away_score", "to_home", "to_away", "kind", "ytg"):
            self.assertEqual(list(a[c]), list(b[c]), c)
        self.assertEqual(a.loc[2, "away_score"], 6)            # the lagging board, the away side's six

    def test_lost_quarter_dropped(self):
        rows = [row("L", 1, "SCRIMMAGE", "TEAM_A", 1, 10, 30, 100, period=2),
                row("L", 2, "SCRIMMAGE", "TEAM_A", 2, 5, 35, 230, period=2)]
        ex = frame(rows + [row("M", 1, "SCRIMMAGE", "TEAM_A", 1, 10, 30, 100)])
        st, _ = W.build_states(ex, {"L": 0.5, "M": 0.5}, {})
        self.assertEqual(set(st.match_code), {"M"})

    def test_repo_fixtures_parse(self):
        """The repo's hand-made export rows (xgb/tests/test_drives.py) read without loss."""
        from xgb.tests import test_drives as T
        ex = frame([dict(r, opening_offense="TEAM_A") for r in T.M1 + T.M2])
        st, _ = W.build_states(ex, {"M1": 0.55, "M2": 0.55}, {})
        self.assertEqual(len(st), len(T.M1) + len(T.M2))
        m2 = st[st.match_code == "M2"].set_index("message")
        self.assertEqual(m2.loc[4, "home_score"], 7)            # the kick after a six the board had not caught up to
        m1 = st[st.match_code == "M1"].set_index("message")
        self.assertEqual((m1.loc[16, "home_score"], m1.loc[16, "away_score"]), (7, 3))
        self.assertEqual((m1.loc[15, "away_score"], m1.loc[15, "pos_home"]), (3, 1))

    def test_repo_fakes_parse(self):
        """eAMFModel's synthetic exports (tests/fakes.py) read whole, the board matching the final."""
        from eAMFModel.tests import fakes
        matches = fakes._matches(n=6, seed=5)
        ex = frame([dict(r, file_time=None) for rows in matches.values() for r in rows])
        st, _ = W.build_states(ex, {c: 0.5 for c in matches}, {c: pd.Timestamp("2026-10-01") for c in matches})
        self.assertEqual(len(st), sum(len(v) for v in matches.values()))
        last = st.groupby("match_code").last()
        self.assertTrue(((last.home_score == last.final_p1) & (last.away_score == last.final_p2)).all())


class TestWalkForward(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        import make_synthetic as S
        cls.dir = tempfile.mkdtemp()
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            S.main(["--out", cls.dir, "--days", "24", "--per-day", "20", "--seed", "3"])
        ex = W.read_export(os.path.join(cls.dir, "scouting_playover.csv"))
        cls.st, _ = W.build_states(ex, W.read_prematch(os.path.join(cls.dir, "prematch.csv")),
                                   W.read_starts(os.path.join(cls.dir, "history.csv")))

    def test_no_state_after_the_cut(self):
        seen = []
        real = W.fit_all

        def spy(tr, params, rounds):
            seen.append(tr)
            return real(tr, params, rounds)

        W.fit_all = spy
        try:
            out = W.walk_forward(self.st, "2026-07-10", "2026-07-25", 5, 60, dict(W.PARAMS, max_depth=3), 50,
                                 log=lambda *_: None)
        finally:
            W.fit_all = real
        self.assertEqual(len(seen), 3)
        days = sorted(self.st.loc[(self.st.day >= "2026-07-10") & (self.st.day < "2026-07-25"), "day"].unique())
        for tr, first in zip(seen, days[::5]):
            self.assertTrue((tr.settled <= pd.Timestamp(first)).all())
            self.assertTrue((tr.day < first).all())
        w = (self.st.day >= "2026-07-10") & (self.st.day < "2026-07-25")
        self.assertTrue(out["wp"][w].notna().all() and out["wp"][~w].isna().all())

    def test_monotone_and_better_than_time_alone(self):
        tr = self.st[(self.st.day < "2026-07-18") & self.st.y.notna()]
        te = self.st[(self.st.day >= "2026-07-18") & self.st.y.notna()]
        m = W.fit_all(tr, dict(W.PARAMS, max_depth=3), 150)
        bl, bp, _ = W.mono_check(m["wp"], te, n=500)
        self.assertEqual((bl, bp), (0, 0))
        p = W.predict_all(m, te)
        ll = {k: W.logloss_each(te.y.to_numpy(), v).mean() for k, v in p.items()}
        pre = W.logloss_each(te.y.to_numpy(), te.pre_p_home.to_numpy()).mean()
        self.assertLess(ll["wp"], pre)
        self.assertLess(ll["stern"], pre)
        a, b, c, _ = m["stern"]
        self.assertGreater(b, 0)
        self.assertGreater(c, 0)


if __name__ == "__main__":
    unittest.main()
