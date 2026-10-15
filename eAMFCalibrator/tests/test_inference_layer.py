"""The inference layer: its covariances by hand, the A-beat-B, C-beat-D read at A v D, and a planted
form per gamer and session recovered and used walk-forward."""

import datetime as dt
import random
import unittest

from .. import inference_layer as il, rematch_layer as rl

T0 = dt.datetime(2026, 9, 1, 18, 0)


def match(code, minutes, home, away, margin_err=0.0, total_err=0.0, teams=("Ravens", "Lions"), session=0,
          day=None, prior=(0.0, 40.0), played=True):
    """An item: the prior's margin and total, the result off them by the errors."""
    a, b = sorted((home, away))
    m, t = prior
    return rl.item(code, T0 + dt.timedelta(minutes=minutes), day or (a, b, session), home, away, teams[0], teams[1],
                   m, t, (m + margin_err) if played else None, (t + total_err) if played else None,
                   ((home, session), (away, session)))


class TestCovariances(unittest.TestCase):

    def test_what_two_matches_share(self):
        layer = il.InferenceLayer({"same team": [4.0, 9.0], "team changed": [1.0, 2.0]},
                                  {"same teams": [2.0, 3.0], "teams changed": [0.5, 1.0]}, [50.0, 100.0])
        ab = match("1", 0, "A", "B")
        ca = match("2", 40, "C", "A", teams=("Bills", "Ravens"))        # A away, the same team
        ab2 = match("3", 80, "A", "B")                                   # a rematch, the same teams
        cd = match("4", 80, "C", "D", teams=("Bills", "Chiefs"))
        (m, t) = layer.covariances([ab, ca, ab2, cd])
        self.assertAlmostEqual(m[0, 1], -4.0)          # A: home in one, away in the other
        self.assertAlmostEqual(t[0, 1], 9.0)
        self.assertAlmostEqual(m[0, 2], 2 * 4.0 + 2.0)  # both gamers' form and the pair's
        self.assertAlmostEqual(t[0, 2], 2 * 9.0 + 3.0)
        self.assertAlmostEqual(m[0, 3], 0.0)           # nothing shared
        self.assertAlmostEqual(m[1, 3], 4.0)           # C home in both, the same team
        self.assertAlmostEqual(m[0, 0], 2 * 4.0 + 2.0 + 50.0)
        ab_swapped = match("5", 120, "B", "A")
        m2, _ = layer.covariances([ab, ab_swapped])
        self.assertAlmostEqual(m2[0, 1], -(2 * 1.0 + 0.5))   # sides swapped: each gamer on the other team

    def test_negative_estimates_are_used_as_zero(self):
        layer = il.InferenceLayer({"same team": [-1.0, 4.0], "team changed": [3.0, 9.0]})
        self.assertEqual(layer.use_gamer["same team"], [0.0, 4.0])
        self.assertEqual(layer.use_gamer["team changed"], [3.0, 9.0])
        self.assertIn("-1.00", "\n".join(layer.describe()))

    def test_a_rematch_reads_as_the_rematch_layer_does(self):
        # prod's fits: the rematch layer's same-teams 2.97^2 and changed 2.83^2 against the inference
        # layer's form (same team 1.63^2, changed 0.78^2) and pair (same teams 1.87^2, changed 2.65^2)
        layer = il.InferenceLayer({"same team": [1.63 ** 2, 0.0], "team changed": [0.78 ** 2, 0.0]},
                                  {"same teams": [1.87 ** 2, 0.0], "teams changed": [2.65 ** 2, 0.0]}, [9.26 ** 2, 144])
        ab = match("1", 0, "A", "B")
        same = match("2", 40, "A", "B")
        changed = match("3", 80, "A", "B", teams=("Bills", "Chiefs"))
        m, _ = layer.covariances([ab, same, changed])
        self.assertAlmostEqual(m[0, 1], 2.97 ** 2, delta=0.3)
        self.assertAlmostEqual(m[0, 2], 2.83 ** 2, delta=0.3)
        won = match("4", 0, "A", "B", margin_err=10)            # A beat the price by 10 the first time
        self.assertGreater(layer.move(changed, [won])[0], 0.5)


class TestInference(unittest.TestCase):

    def test_a_beat_b_and_c_beat_d_moves_a_v_d(self):
        layer = il.InferenceLayer({"same team": [9.0, 9.0], "team changed": [9.0, 9.0]}, None, [64.0, 100.0])
        ab = match("1", 0, "A", "B", margin_err=14)      # A beat B by 14 more than the prior said
        cd = match("2", 0, "C", "D", margin_err=14)      # C beat D likewise
        by_key = {}
        for it in (ab, cd):
            for k in il._keys(it):
                by_key.setdefault(k, []).append(it)
        ad = match("3", 40, "A", "D")
        earlier, near = il.earlier_for(ad, by_key)
        self.assertEqual((len(earlier), near), (2, 2))
        dm, _ = layer.move(ad, earlier)
        self.assertGreater(dm, 2.0)                      # towards A
        cb = match("4", 40, "C", "B")
        dm2, _ = layer.move(cb, il.earlier_for(cb, by_key)[0])
        self.assertAlmostEqual(dm, dm2)                  # and towards C
        da = match("5", 40, "D", "A")
        self.assertAlmostEqual(layer.move(da, il.earlier_for(da, by_key)[0])[0], -dm)

    def test_an_opponents_other_result_counts_only_once_it_is_in(self):
        ab = match("1", 0, "A", "B", margin_err=10)
        bc = match("2", 20, "B", "C", margin_err=-10)    # B lost again: B is off form, A's win says less
        by_key = {}
        for it in (ab, bc):
            for k in il._keys(it):
                by_key.setdefault(k, []).append(it)
        ad = match("3", 50, "A", "D")
        self.assertEqual([e["match"] for e in il.earlier_for(ad, by_key)[0]], ["1"])     # B v C not in yet
        ad_later = match("4", 60, "A", "D")
        self.assertEqual([e["match"] for e in il.earlier_for(ad_later, by_key)[0]], ["1", "2"])
        layer = il.InferenceLayer({"same team": [9.0, 9.0], "team changed": [9.0, 9.0]}, None, [64.0, 100.0])
        self.assertLess(layer.move(ad_later, il.earlier_for(ad_later, by_key)[0])[0],
                        layer.move(ad, il.earlier_for(ad, by_key)[0])[0])

    def test_planted_form_is_found_and_used(self):
        rnd = random.Random(7)
        items, gamers = [], [f"G{i}" for i in range(8)]
        k = 0
        for day in range(40):
            form = {g: (rnd.gauss(0, 3.5), rnd.gauss(0, 3.0)) for g in gamers}
            for game in range(10):
                order = gamers[:]
                rnd.shuffle(order)
                for i in range(0, 8, 2):
                    h, a = order[i], order[i + 1]
                    me = form[h][0] - form[a][0] + rnd.gauss(0, 8)
                    te = form[h][1] + form[a][1] + rnd.gauss(0, 10)
                    it = match(f"M{k}", day * 1440 + 40 * game + i, h, a, me, te, session=day,
                               day=tuple(sorted((h, a))) + (day,))
                    items.append(it)
                    k += 1
        fit = il.InferenceLayer.fit(items)
        self.assertAlmostEqual(fit.gamer["same team"][0] ** 0.5, 3.5, delta=0.6)
        self.assertAlmostEqual(fit.gamer["same team"][1] ** 0.5, 3.0, delta=0.8)
        self.assertLess(abs(fit.pair["same teams"][0]), 4.0)              # no pair effect planted
        rows, _ = il.walk_forward(items, min_days=7)
        live = [r for r in rows if r["start"] >= T0 + dt.timedelta(days=7) and r["n_others"]]
        base = rl.score(live)
        for r in live:
            r["m_adj"], r["t_adj"] = r["margin"] + r["infer_margin"], r["total"] + r["infer_total"]
        adj = rl.score(live, "m_adj", "t_adj")
        self.assertLess(adj["log_loss"], base["log_loss"] - 0.01)
        self.assertLess(adj["total_rmse"], base["total_rmse"])
        burn = [r for r in rows if (r["start"].date() - T0.date()).days < 7]      # UTC days, as walk_forward
        self.assertTrue(burn and all(r["infer_margin"] == 0.0 and r["infer_total"] == 0.0 for r in burn))


if __name__ == "__main__":
    unittest.main()
