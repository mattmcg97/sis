import csv
import datetime as dt
import os
import random
import tempfile
import unittest

from eAMFCalibrator import config, money_layer as ml

T0 = dt.datetime(2026, 10, 1, 8, 0)


def _world(n_days=12, per_day=30, seed=3):
    """Matches among eight gamers priced even, ACE winning three in four; his backers stake big and win."""
    rng = random.Random(seed)
    gamers = ["ACE"] + [f"G{k}" for k in range(7)]
    rows, hist, bets = [], {}, []
    for d in range(n_days):
        for k in range(per_day):
            t = T0 + dt.timedelta(days=d, minutes=40 * k)
            code = f"AF{k:03d}{t:%d%m%y}"
            home, away = rng.sample(gamers, 2)
            p_home = 0.75 if home == "ACE" else 0.25 if away == "ACE" else 0.5
            won = rng.random() < p_home
            fin = (28, 14) if won else (14, 28)
            rows.append(dict(MATCH_CODE=code, start=f"{t:%Y-%m-%d %H:%M:%S}", home=home, away=away,
                             home_score=fin[0], away_score=fin[1], home_pts=20.0, away_pts=20.0, p_home=0.5))
            hist[code] = (t, home, away, fin)
            for side, g in ((0, home), (1, away)):
                stake = 900.0 if g == "ACE" else 30.0
                bets.append(ml.Backed(code, side, stake, won == (side == 0), 0.5, True))
    return rows, list(rows[0]), hist, bets


class TestMoneyLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.cols, cls.hist, cls.bets = _world()
        cls.matches = ml.matches_of(cls.rows, cls.cols)

    def test_a_match_reads_only_the_bets_settled_before_its_cut(self):
        codes = sorted(self.matches, key=lambda c: self.matches[c]["start"])
        first, second = codes[0], codes[1]                     # 40 minutes apart: the first's bets settle in 36
        sig = ml.signals(self.bets, self.matches, shrink=1000.0, cut="kickoff")
        self.assertEqual(sig[first], (0.0, 0.0))
        m1, m2 = self.matches[first], self.matches[second]
        g1 = {m1["home"], m1["away"]}
        if g1 & {m2["home"], m2["away"]}:
            self.assertNotEqual(sig[second], (0.0, 0.0))
        daily = ml.signals(self.bets, self.matches, shrink=1000.0, cut="daily")
        self.assertEqual(daily[second], (0.0, 0.0))            # nothing settled before the window's first day

    def test_the_money_finds_the_strong_gamer(self):
        sig = ml.signals(self.bets, self.matches)
        late = [c for c, m in self.matches.items() if m["start"].date() >= dt.date(2026, 10, 8)]
        ace = [s[0] if self.matches[c]["home"] == "ACE" else s[1] for c, s in sig.items() if c in late
               and "ACE" in (self.matches[c]["home"], self.matches[c]["away"])]
        self.assertTrue(ace and min(ace) > 0.05)
        top = ml.gamer_rows(self.bets, self.matches)[0]
        self.assertEqual(top["gamer"], "ACE")

    def test_the_slope_sees_only_the_days_before_and_the_money_beats_the_even_price(self):
        sig = ml.signals(self.bets, self.matches)
        moves, path = ml.walk(self.matches, sig)
        self.assertEqual(path[0][1], 0.0)                      # the first day has nothing to fit on
        self.assertGreater(path[-1][1], 0.3)
        g = ml.grade(self.matches, moves, ml.tested(self.matches, moves, dt.date(2026, 10, 4)))
        self.assertLess(g["brier"][2], 0)
        self.assertLess(g["brier"][4], 0)                      # the interval clear of zero
        fixed, _ = ml.walk(self.matches, sig, slope=0.5)
        self.assertTrue(all(b == 0.5 for b, _ in fixed.values()))

    def test_the_table_moves_the_price_and_the_margin_over_the_share(self):
        sig = {c: (0.1, 0.0) for c in self.matches}
        moves = {c: (0.5, 0.05) for c in self.matches}
        out = ml.money_table(self.rows, self.cols, self.matches, moves, sig, sigma=13.0, share=0.75)
        r = out[0]
        self.assertAlmostEqual(float(r["p_home"]), 0.55)
        dm = 13.0 * (ml._probit(0.55) - ml._probit(0.5))
        self.assertAlmostEqual(r["money_margin"], round(dm, 4), places=4)
        self.assertAlmostEqual(r["home_pts"] - r["away_pts"], dm / 0.75, places=3)
        self.assertAlmostEqual(r["home_pts"] + r["away_pts"], 40.0, places=3)

    def test_points_per_probit_off_the_table(self):
        rows = [dict(MATCH_CODE=str(i), p_home=p, home_pts=20 + 6.5 * ml._probit(p), away_pts=20 - 6.5 * ml._probit(p))
                for i, p in enumerate([0.1 + 0.8 * k / 99 for k in range(100)])]
        self.assertAlmostEqual(ml.points_per_probit(rows, ["MATCH_CODE", "p_home", "home_pts", "away_pts"]), 13.0, places=6)
        self.assertEqual(ml.points_per_probit(rows[:10], ["MATCH_CODE", "p_home", "home_pts", "away_pts"]), ml.SIGMA)

    def test_the_bets_sim_rows_are_read(self):
        rows = [dict(market="moneyline", selection="Home", result="won", stake="10", stream_prob="0.4", in_play="True",
                     match_code="M1"),
                dict(market="moneyline", selection="Away", result="lost", stake="10", stream_prob="", implied_prob="0.546",
                     in_play="False", match_code="M1"),
                dict(market="moneyline", selection="Home", result="push", stake="10", stream_prob="0.5", match_code="M1"),
                dict(market="spread", selection="Home", result="won", stake="10", stream_prob="0.5", match_code="M1")]
        got = ml.backed_from_sim(rows)
        self.assertEqual(len(got), 2)
        self.assertEqual((got[0].side, got[0].won, got[0].price), (0, True, 0.4))
        self.assertAlmostEqual(got[1].price, 0.546 / (1 + ml.BET_MARGIN[False]))

    def test_the_money_so_far_in_the_match(self):
        """Bets on the side the money is already on win more than the price says: beta above zero, and the
        bets re-priced with it keep more for the book."""
        rng = random.Random(7)
        rows = []
        for d in range(6):
            for k in range(60):
                code = f"AF{k:03d}{d + 1:02d}1026"
                strong = rng.choice(["Home", "Away"])
                won_home = rng.random() < (0.7 if strong == "Home" else 0.3)
                for j in range(8):
                    sel = strong if j % 4 else ("Away" if strong == "Home" else "Home")
                    won = won_home == (sel == "Home")
                    rows.append(dict(match_code=code, market="moneyline", selection=sel, stake="100", odds="1.85",
                                     stream_prob="0.5", candidate_prob="0.5", result="won" if won else "lost",
                                     revenue=str(100 - 185 if won else 100),
                                     bet_time=f"2026-10-{d + 1:02d} {8 + j // 2:02d}:{k % 60:02d}:{j:02d}"))
        bets = ml.flow_bets(rows, "stream_prob")
        self.assertEqual(bets[0]["x"], 0.0)                    # the first bet of a match: no money yet
        moved, path = ml.flow_walk(bets)
        self.assertEqual(path[0][0], "2026-10-02")             # the first day is only fitted on
        self.assertGreater(path[-1][1], 0)
        stats = ml.flow_stats(list(ml.flow_sums(moved).values()))
        self.assertGreater(stats[3], 0)                        # moved less as it is
        text = "\n".join(ml.flow_test(rows, ["stream_prob", "candidate_prob"], n_boot=50))
        self.assertIn("prod (stream_prob)", text)
        self.assertIn("candidate (candidate_prob)", text)

    def test_the_run_writes_the_table(self):
        old = config.CUTOFF_START, config.CUTOFF_END
        config.CUTOFF_START, config.CUTOFF_END = "2026-10-01 00:00:00", "2026-10-13 00:00:00"
        try:
            with tempfile.TemporaryDirectory() as d:
                base = os.path.join(d, "avg.csv")
                with open(base, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, self.cols)
                    w.writeheader()
                    w.writerows(self.rows)
                bets = os.path.join(d, "bets_sim.csv")
                with open(bets, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, ["bet_id", "match_code", "bet_time", "market", "selection", "stake", "odds",
                                            "revenue", "result", "stream_prob", "in_play"])
                    w.writeheader()
                    for i, b in enumerate(self.bets):
                        t = self.hist[b.match][0] + dt.timedelta(minutes=5)
                        w.writerow(dict(bet_id=i, match_code=b.match, bet_time=f"{t:%Y-%m-%d %H:%M:%S}",
                                        market="moneyline", selection=("Home", "Away")[b.side], stake=b.stake,
                                        odds=1.85, revenue=b.stake - b.stake * 1.85 if b.won else b.stake,
                                        result="won" if b.won else "lost", stream_prob=0.5, in_play="True"))
                ml.N_FLOW_BOOT, keep = 50, ml.N_FLOW_BOOT
                try:
                    path = ml.run(os.path.join(d, "out"), base, bets=[bets], grid=False)
                finally:
                    ml.N_FLOW_BOOT = keep
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                with open(os.path.join(d, "out", "money.csv"), newline="", encoding="utf-8") as fh:
                    table = list(csv.DictReader(fh))
        finally:
            config.CUTOFF_START, config.CUTOFF_END = old
        self.assertIn("the test:", text)
        self.assertIn("the match so far", text)
        self.assertEqual(len(table), len(self.rows))
        ace = [r for r in table if r["home"] == "ACE" and r["start"] >= "2026-10-08"]
        self.assertTrue(ace and all(float(r["p_home"]) > 0.5 and float(r["home_pts"]) > 20 for r in ace))


if __name__ == "__main__":
    unittest.main()
