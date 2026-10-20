"""Where the money is, as prices: the money's price fitted and scored, and what the money knows."""

import datetime as dt
import os
import tempfile
import unittest

import numpy as np

from .. import bet_money as bm, bets, config

T0 = dt.datetime(2026, 10, 1, 12, 0, 0)


def _world(n=600, seed=1, informed=0.6):
    """Matches priced level by prod throughout; each has a truth s (+1 / -1): home wins 0.5 + 0.2 s of the time,
    and each bettor backs home 0.5 + informed x 0.25 s of the time (informed 0: the money knows nothing)."""
    rng = np.random.default_rng(seed)
    quotes, all_bets, finals = [], [], {}
    for m in range(n):
        code = f"M{m:04d}"
        start = T0 + dt.timedelta(days=m * 18 // n, minutes=m)
        truth = 1 if rng.random() < 0.5 else -1
        home_won = rng.random() < 0.5 + 0.2 * truth
        finals[code] = (21, 14) if home_won else (14, 21)
        quotes.append((code, 50, start - dt.timedelta(minutes=10), 50.0, 2.0, "", None, "open", "true"))
        quotes.append((code, 51, start - dt.timedelta(minutes=10), 50.0, 2.0, "", None, "open", "true"))
        for k in range(1, 41):
            t = start + dt.timedelta(seconds=30 * k)
            quotes.append((code, 50, t, 50.0, 2.0, "", k, "open", "true"))
            quotes.append((code, 51, t, 50.0, 2.0, "", k, "open", "true"))
        for j in range(30):
            t = start + dt.timedelta(seconds=float(rng.uniform(40, 1200)))
            home = rng.random() < 0.5 + informed * 0.25 * truth
            won = home == home_won
            stake = float(rng.integers(5, 100))
            all_bets.append(bets.Bet(bet_id=f"{code}-{j}", match_code=code, time=t, market_type=1, selection=1 if home else 2,
                                     odds=1.9, stake=stake, revenue=-stake * 0.9 if won else stake,
                                     extra={config.BET_GROUP_COLUMN: "FANDUEL", config.BET_IN_PLAY_COLUMN: "Yes",
                                            config.BET_VIP_COLUMN: "Standard" if j % 3 else "VIP"}))
    return quotes, all_bets, finals


def _run(informed):
    quotes, all_bets, finals = _world(informed=informed)
    tl = bets.timeline(quotes)
    money = bm.Money(all_bets, {}, 120)
    joined = bets.join(all_bets, {}, {}, tl, {}, None, finals)
    points = bm.moment_points(tl, money, None, finals, {})
    bpts = bm.bet_points(all_bets, joined, tl, money, None, finals)
    split = sorted({x["t"].date() for x in points})[9]
    return bm.layer1(points, bpts, split, B=60), bm.layer2(bpts, B=60), points, bpts


class TestMoney(unittest.TestCase):

    def test_the_money_before_a_moment(self):
        t = T0 + dt.timedelta(minutes=5)
        b = lambda s, sel, stake, temp: bets.Bet(bet_id=s, match_code="M1", time=t + dt.timedelta(seconds=s), market_type=1,
                                                 selection=sel, odds=2.0, stake=stake, revenue=0.0,
                                                 extra={config.BET_IN_PLAY_COLUMN: "Yes", config.BET_VIP_COLUMN: temp})
        money = bm.Money([b(-200, 1, 50, "VIP"), b(-60, 1, 30, "Standard"), b(-30, 2, 10, "Standard"), b(0, 2, 100, "VIP")],
                         {}, 120)
        f, stake = money.at("M1", "moneyline", t.timestamp(), True)
        self.assertAlmostEqual(f["all"], (30 - 10) / 40)              # the last two minutes, the bet at t itself out
        self.assertAlmostEqual(f["Standard"], 0.5)
        self.assertEqual(f["VIP"], 0.0)
        self.assertEqual(stake, 40)


class TestLayers(unittest.TestCase):

    def test_informed_money_moves_the_price_and_scores_better(self):
        l1, l2, points, bpts = _run(1.0)
        r = next(x for x in l1 if x["group"] == "moneyline" and x["in_play"] and x["label"] == "all money")
        self.assertGreater(r["beta_ci"][0][0], 0)                       # the money's tilt above 0
        self.assertLess(r["ll_ci"][1], 0)                               # its price beats prod's on the later matches
        self.assertGreater(r["change"], 0)                              # and re-priced at it, the book keeps more
        self.assertTrue(any(x["label"] == "alone" for x in l2[0]["fits"]))
        text = "\n".join(bm.report(l1, l2, "2026-10-10", 120))
        for part in ("LAYER 1, THE MONEY'S PRICE", "LAYER 2, WHAT THE MONEY KNOWS", "moneyline", "what the money follows"):
            self.assertIn(part, text)

    def test_money_that_knows_nothing(self):
        l1, _, _, _ = _run(0.0)
        r = next(x for x in l1 if x["group"] == "moneyline" and x["in_play"] and x["label"] == "all money")
        self.assertLess(r["beta_ci"][0][0], 0)
        self.assertGreater(r["beta_ci"][0][1], 0)                       # the tilt's interval holds 0

    def test_v21_s_price_off_a_sim_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bets_sim.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("bet_id,candidate_prob_v21,candidate_line_v21,candidate_prob_v9\nA,0.6,,0.5\nB,,,0.4\n")
            got = bm.read_sim(path)
        self.assertEqual(got["A"], (0.6, None))
        self.assertEqual(got["B"], (None, None))


if __name__ == "__main__":
    unittest.main()
