"""The gamers' daily stats (nothing from the match's own day, shrunk to the league, recent matches
counting more) and the feature layer fitted on a prior's errors: a planted skill found and used."""

import datetime as dt
import random
import unittest

from .. import feature_layer as fl, gamer_features as gf, rematch_layer as rl

T0 = dt.datetime(2026, 9, 1, 18, 0)


def side_row(code, start, side, player, opp, ypp, opp_ypp, points, against, plays=40):
    """A player_match_stats.csv row with every count the features read."""
    return dict(match_code=code, start=str(start), side=side, player=player, opponent=opp, points=str(points),
                points_against=str(against), scouting="1", snaps=str(plays), detail_snaps=str(plays),
                plays=str(plays), yards=str(round(ypp * plays)), yards_known=str(plays),
                opp_yards=str(round(opp_ypp * plays)), opp_yards_known=str(plays), third_conv="4", third_att="8",
                rz_td="2", rz_trips="3", turnovers="1", tod="0", top_s="480")


def match_rows(code, start, home, away, h_ypp, a_ypp, s1, s2):
    return [side_row(code, start, "home", home, away, h_ypp, a_ypp, s1, s2),
            side_row(code, start, "away", away, home, a_ypp, h_ypp, s2, s1)]


class TestDaily(unittest.TestCase):

    def test_nothing_from_the_matchs_own_day(self):
        day1 = T0
        day2 = T0 + dt.timedelta(days=1)
        rows = (match_rows("M1", day1, "A", "B", 10, 6, 21, 7) + match_rows("M2", day2, "A", "C", 9, 7, 14, 10)
                + match_rows("M3", day2 + dt.timedelta(hours=1), "A", "B", 4, 12, 0, 35))
        f = gf.daily(rows)
        self.assertEqual(f["M2"]["home"], f["M3"]["home"])            # M2 is the same day: not read for M3
        changed = [dict(r, yards="0") if r["match_code"] == "M2" else r for r in rows]
        self.assertEqual(gf.daily(changed)["M3"], f["M3"])
        self.assertNotEqual(f["M1"]["home"]["ypp_for"], f["M2"]["home"]["ypp_for"])   # day 1 is read on day 2

    def test_a_new_gamer_reads_as_the_league_and_shrinkage_pulls_towards_it(self):
        rows = []
        for k in range(30):
            rows += match_rows(f"M{k}", T0 + dt.timedelta(days=k), "A", f"X{k}", 12, 6, 28, 10)
        rows += match_rows("N", T0 + dt.timedelta(days=40), "A", "NEW", 8, 8, 14, 14)
        f = gf.daily(rows)["N"]
        league = (12 * 30 + 6 * 30) / 60
        self.assertAlmostEqual(f["away"]["ypp_for"], league, places=1)      # nothing on NEW: the league's
        self.assertGreater(f["home"]["ypp_for"], league)
        self.assertLess(f["home"]["ypp_for"], 12)                            # shrunk, not all the way
        self.assertEqual(f["n_away"], 0.0)

    def test_recent_matches_count_more(self):
        rows = []
        for k in range(100):
            ypp = 12 if k < 50 else 6                                        # A was good, now poor
            rows += match_rows(f"M{k}", T0 + dt.timedelta(days=k), "A", f"X{k % 7}", ypp, 8, 20, 20)
        rows += match_rows("N", T0 + dt.timedelta(days=200), "A", "B", 8, 8, 14, 14)
        a = gf.daily(rows, half_life=10, prior_matches=1)["N"]["home"]["ypp_for"]
        self.assertLess(a, 7.0)                                              # nearer the recent 6 than 9
        self.assertEqual(len(gf.margin_x(gf.daily(rows)["N"])), len(gf.NAMES))
        self.assertIsNone(gf.margin_x(None))


def planted(days=40, per_day=24, seed=11):
    """Gamers with a lasting skill the prior (a margin of 0, a total of 40) doesn't see: it shows in
    their yards a play and in their margins."""
    rnd = random.Random(seed)
    gamers = [f"G{i}" for i in range(12)]
    skill = {g: rnd.gauss(0, 1) for g in gamers}
    rows, items = [], []
    k = 0
    for d in range(days):
        for j in range(per_day):
            h, a = rnd.sample(gamers, 2)
            start = T0 + dt.timedelta(days=d, minutes=20 * j)
            margin = 5 * (skill[h] - skill[a]) + rnd.gauss(0, 8)
            total = 40 + rnd.gauss(0, 10)
            s1, s2 = max(0, round((total + margin) / 2)), max(0, round((total - margin) / 2))
            code = f"M{k:05d}"
            rows += match_rows(code, start, h, a, 8 + 1.2 * skill[h] + rnd.gauss(0, 1.5),
                               8 + 1.2 * skill[a] + rnd.gauss(0, 1.5), s1, s2)
            items.append(dict(match=code, start=start, margin=0.0, total=40.0,
                              actual_margin=s1 - s2, actual_total=s1 + s2))
            k += 1
    return rows, items


class TestLayer(unittest.TestCase):

    def test_a_planted_skill_is_found_and_used(self):
        rows, items = planted()
        feats = gf.daily(rows)
        moves, fm, ft = fl.walk_forward(items, feats)
        names = list(gf.NAMES)
        self.assertGreater(fm.beta[names.index("ypp_for")] + fm.beta[names.index("points_for")], 0.5)
        late = [r for r in items if r["start"] >= T0 + dt.timedelta(days=25)]
        for r in late:
            r["m_adj"] = r["margin"] + moves[r["match"]][0]
        base, adj = rl.score(late), rl.score(late, "m_adj", "total")
        self.assertLess(adj["log_loss"], base["log_loss"] - 0.02)
        self.assertTrue(any("ypp_for" in line for line in fl.describe(fm, ft)))

    def test_no_move_until_enough_matches_and_none_without_features(self):
        rows, items = planted(days=10)
        feats = gf.daily(rows)
        first_day = [r for r in items if r["start"].date() == T0.date()]
        moves, _, _ = fl.walk_forward(items, feats, min_train=len(items) + 1)
        self.assertTrue(all(m == (0.0, 0.0) for m in moves.values()))
        moves, _, _ = fl.walk_forward(items, {}, min_train=10)
        self.assertTrue(all(m == (0.0, 0.0) for m in moves.values()))
        moves, _, _ = fl.walk_forward(items, feats, min_train=10)
        self.assertTrue(all(moves[r["match"]] == (0.0, 0.0) for r in first_day))   # nothing before day one

    def test_on_a_base_layer_it_fits_what_that_leaves(self):
        rows, items = planted(days=30)
        feats = gf.daily(rows)
        for r in items:                                                  # a base that already takes the margin
            r["base_m"], r["base_t"] = r["actual_margin"], 0.0
        moves, fm, _ = fl.walk_forward(items, feats, base=("base_m", "base_t"))
        self.assertLess(max(abs(m[0]) for m in moves.values()), 1e-6)   # nothing left to fit on the margin
        self.assertIsNotNone(fm.beta)


if __name__ == "__main__":
    unittest.main()
