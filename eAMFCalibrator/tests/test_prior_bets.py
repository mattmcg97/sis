import csv
import datetime as dt
import os
import random
import tempfile
import unittest

from eAMFCalibrator import prior_bets as pb, score_model

T0 = dt.datetime(2026, 10, 1, 8, 0)
TABLE_FIELDS = ["MATCH_CODE", "start", "home", "away", "home_pts", "away_pts", "p_home", "score_model", "score_params"]
BET_FIELDS = ["bet_id", "match_code", "bet_time", "in_play", "market", "selection", "feed_market", "odds", "stake",
              "revenue", "result", "stream_prob", "stream_line", "on_prod_line", "final_p1", "final_p2", "quarter",
              "clock_left", "candidate_prob", "candidate_odds", "candidate_revenue", "simulated"]


def _world(n=240, seed=4):
    """Matches whose true home chance the 'good' table knows and prod doesn't (prod says 0.5); one pre-match
    and two in-play moneyline bets on the winner's side's... on a random side each, at prod's price."""
    rng = random.Random(seed)
    tables = {"good": [], "flat": []}
    bets = []
    for i in range(n):
        t = T0 + dt.timedelta(minutes=30 * i)
        code = f"AF{i % 100:03d}{t:%d%m%y}"
        margin = rng.choice([-8.0, -4.0, 4.0, 8.0])
        h, a = 20 + margin / 2, 20 - margin / 2
        truth = score_model.Normal(margin, 40.0).p_home()
        won = rng.random() < truth
        fin = (27, 13) if won else (13, 27)
        home, away = rng.sample(["A", "B", "C", "D", "E", "F"], 2)
        for name, (hp, ap) in (("good", (h, a)), ("flat", (20.0, 20.0))):
            tables[name].append(dict(MATCH_CODE=code, start=f"{t:%Y-%m-%d %H:%M:%S}", home=home, away=away, home_pts=hp,
                                     away_pts=ap, p_home=round(score_model.Normal(hp - ap, hp + ap).p_home(), 5),
                                     score_model="normal", score_params=""))
        for k, (in_play, quarter, clock) in enumerate(((False, "", ""), (True, "Q1", 236), (True, "Q1", 100))):
            sel = rng.choice(["Home", "Away"])
            bet_won = won == (sel == "Home")
            when = t + dt.timedelta(minutes=-10 + 2 * k)
            bets.append(dict(bet_id=f"{i}-{k}", match_code=code, bet_time=f"{when:%Y-%m-%d %H:%M:%S}",
                             in_play=str(in_play), market="moneyline", selection=sel, feed_market=50 if sel == "Home" else 51,
                             odds=1.85, stake=100, revenue=100 - 185 if bet_won else 100, result="won" if bet_won else "lost",
                             stream_prob=0.5 if k < 2 else 0.6, stream_line="", on_prod_line="True", final_p1=fin[0],
                             final_p2=fin[1], quarter=quarter, clock_left=clock, candidate_prob=0.5, candidate_odds=1.85,
                             candidate_revenue=100 - 185 if bet_won else 100, simulated="True"))
    return tables, bets


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)


class TestPriorBets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables, cls.bets = _world()
        cls.dir = tempfile.TemporaryDirectory()
        cls.paths = {}
        for name, rows in cls.tables.items():
            cls.paths[name] = os.path.join(cls.dir.name, f"{name}.csv")
            _write(cls.paths[name], rows, TABLE_FIELDS)
        cls.bets_path = os.path.join(cls.dir.name, "bets_sim.csv")
        _write(cls.bets_path, cls.bets, BET_FIELDS)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_a_table_prices_off_its_own_score_model(self):
        t = pb.Table(self.paths["good"], draws=2000)
        code = self.tables["good"][0]["MATCH_CODE"]
        self.assertAlmostEqual(t.p_home(code), float(self.tables["good"][0]["p_home"]))
        self.assertAlmostEqual(t.chance(code, 50, None), t.book(code).p_margin(0.0))
        self.assertAlmostEqual(t.chance(code, 51, None), 1 - t.book(code).p_margin(0.0))
        self.assertIsNone(t.chance("NOPE", 50, None))

    def test_the_kick_off_money_is_kept_and_the_moved_price_left_out(self):
        rows = [dict(r, stake=str(r["stake"])) for r in self.bets]
        res = pb.results(rows)
        codes = set(res)
        prod = pb.prematch_prices(rows, "stream_prob")
        book = pb.bet_book(rows, codes, {"good": pb.Table(self.paths["good"], draws=2000)}, [], prod, {})
        groups = {b["group"] for b in book}
        self.assertEqual(groups, {"pre-match, clean", "kick-off, first 15s"})     # the third bet's price had moved

    def test_a_previous_result_not_yet_in(self):
        prev = pb.results_in({"good": pb.Table(self.paths["good"], draws=2000)})
        self.assertTrue(prev)
        code, when = next(iter(prev.items()))
        self.assertGreater(when, pb._time(next(r for r in self.tables["good"] if r["MATCH_CODE"] == code)["start"]) - dt.timedelta(days=30))

    def test_the_table_that_knows_beats_prod(self):
        out = os.path.join(self.dir.name, "out")
        path = pb.run(out, [self.bets_path], [("good", self.paths["good"]), ("flat", self.paths["flat"])], draws=2000)
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("the moneyline on", text)
        self.assertIn("pre-match, clean: every market", text)
        self.assertIn("kick-off, first 15s", text)
        good = next(l for l in text.splitlines() if l.strip().startswith("good") and "[" in l)
        self.assertLess(float(good.split()[2]), 0)                                # its Brier less prod's
        with open(os.path.join(out, "prior_bets.csv"), encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(set(rows[0]) - {"match_code", "home_won", "moneyline_stake"},
                         {"p_home_prod", "p_home_sim", "p_home_good", "p_home_flat"})

    def test_the_disagreement_weight(self):
        rng = random.Random(1)
        res, prod, price = {}, {}, {}
        for i in range(3000):
            p = rng.choice([0.3, 0.5, 0.7])
            res[i] = (1.0 if rng.random() < p else 0.0, "d", 1.0)
            prod[i], price[i] = rng.choice([0.45, 0.5, 0.55]), p
        fit = pb.disagreement(list(res), res, prod, price)
        self.assertAlmostEqual(fit[2], 1.0, delta=3 * fit[3])


if __name__ == "__main__":
    unittest.main()
