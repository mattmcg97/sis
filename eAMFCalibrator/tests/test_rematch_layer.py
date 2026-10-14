"""The rematch layer: the pair-day effect's variance off the errors of two meetings the same day,
the move off the earlier meetings, and the walk-forward fit recovering a planted effect."""

import datetime as dt
import math
import random
import unittest

from .. import rematch_layer as rl

T0 = dt.datetime(2026, 9, 1, 18, 0)


def plant(days=40, pairs_a_day=30, meetings=3, tau=(3.0, 4.0), sigma=(9.0, 12.0), seed=1):
    """Items with a planted pair-day effect: each pair-day's margin and total errors share u."""
    rnd = random.Random(seed)
    out = []
    for d in range(days):
        for p in range(pairs_a_day):
            um, ut = rnd.gauss(0, tau[0]), rnd.gauss(0, tau[1])
            a, b = f"A{p}", f"B{p}"
            for k in range(meetings):
                home, away = (a, b) if k % 2 == 0 else (b, a)
                sign = 1 if home == min(a, b) else -1
                start = T0 + dt.timedelta(days=d, minutes=40 * k + p)
                margin, total = rnd.gauss(0, 4), 35 + rnd.gauss(0, 3)
                team = {a: "Ravens", b: "Lions"}             # each gamer keeps his team all day
                out.append(rl.item(f"M{d}_{p}_{k}", start, (a, b, d), home, away, team[home], team[away],
                                   margin, total, margin + sign * um + rnd.gauss(0, sigma[0]),
                                   total + ut + rnd.gauss(0, sigma[1])))
    return out


class TestLayer(unittest.TestCase):

    def test_the_margin_and_the_moneyline_read_each_other(self):
        self.assertAlmostEqual(rl.win_chance(rl.implied_margin(0.62)), 0.62)
        self.assertAlmostEqual(rl.implied_margin(0.5), 0.0)

    def test_the_fit_finds_the_planted_effect(self):
        layer = rl.Layer.fit(plant())
        tm, tt = (math.sqrt(layer.tau2["same teams"][q]) for q in (0, 1))
        self.assertAlmostEqual(tm, 3.0, delta=0.8)
        self.assertAlmostEqual(tt, 4.0, delta=1.0)
        self.assertAlmostEqual(math.sqrt(layer.sigma2[0]), 9.0, delta=0.5)
        self.assertEqual(layer.pairs["same teams"], 40 * 30 * 3)

    def test_one_earlier_meeting_moves_by_its_share(self):
        layer = rl.Layer({r: [9.0, 16.0] for r in rl.RELATIONS}, [81.0, 144.0])
        a = rl.item("M1", T0, "k", "ANN", "BOB", "Ravens", "Lions", 0.0, 35.0, 10.0, 45.0)
        b = rl.item("M2", T0, "k", "BOB", "ANN", "Lions", "Ravens", 0.0, 35.0)
        dm, dt_ = layer.move(b, [a])
        self.assertAlmostEqual(dm, -10.0 * 9 / 90)       # ANN won by 10: BOB, at home now, moves down
        self.assertAlmostEqual(dt_, 10.0 * 16 / 160)
        self.assertEqual(layer.move(b, []), (0.0, 0.0))

    def test_teams_changed_carry_their_own_share(self):
        layer = rl.Layer({"same teams": [9.0, 0.0], "teams changed": [0.0, 0.0]}, [81.0, 144.0])
        a = rl.item("M1", T0, "k", "ANN", "BOB", "Ravens", "Lions", 0.0, 35.0, 10.0, 35.0)
        same = rl.item("M2", T0, "k", "ANN", "BOB", "Ravens", "Lions", 0.0, 35.0)
        changed = rl.item("M3", T0, "k", "ANN", "BOB", "Bills", "Lions", 0.0, 35.0)
        self.assertGreater(layer.move(same, [a])[0], 0.9)
        self.assertEqual(layer.move(changed, [a])[0], 0.0)

    def test_walk_forward_helps_where_the_effect_is_real(self):
        rows, layer = rl.walk_forward(plant(days=30, tau=(4.0, 6.0)), min_days=7)
        late = [r for r in rows if r["n_earlier"] and (r["start"] - T0).days >= 7]
        self.assertTrue(all(r["move_margin"] == 0 for r in rows if (r["start"] - T0).days < 7))
        a, b = rl.score(late), rl.score(late, "margin_adj", "total_adj")
        self.assertLess(b["total_rmse"], a["total_rmse"])
        self.assertLess(b["log_loss"], a["log_loss"])
        d, t = rl.paired(a["te"], b["te"])
        self.assertLess(t, -3)

    def test_no_effect_moves_next_to_nothing(self):
        rows, _ = rl.walk_forward(plant(days=30, tau=(0.0, 0.0)), min_days=7)
        late = [r for r in rows if r["n_earlier"] and (r["start"] - T0).days >= 7]
        self.assertLess(max(abs(r["move_total"]) for r in late), 2.0)


if __name__ == "__main__":
    unittest.main()
