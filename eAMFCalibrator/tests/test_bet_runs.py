"""Who starts a run of bets: by customer id, whether the other customers' money follows a customer's bets."""

import datetime as dt
import unittest

import numpy as np

from .. import bet_money as bm, bet_runs as br, bets, config

T0 = dt.datetime(2026, 10, 19, 12, 0, 0)
COL = "CUSTOMER_ID"


def _world(n=300, leaders=6, seed=3, copy=0.9):
    """In play, each match a crowd of 40 customers betting sides at random; on most matches a leader bets and,
    copy of the time, its two followers bet its side 5-60s after (copy 0: they bet at random then)."""
    rng = np.random.default_rng(seed)
    all_bets, finals = [], {}
    crowd = [f"C{k:03d}" for k in range(40)]
    for m in range(n):
        code = f"M{m:04d}"
        start = T0 + dt.timedelta(minutes=30 * m)
        finals[code] = (21, 14) if rng.random() < 0.5 else (14, 21)
        rows = []
        for j in range(25):
            rows.append((rng.choice(crowd), float(rng.uniform(0, 1500)), int(rng.integers(1, 3)), float(rng.integers(5, 50))))
        for k in range(leaders):
            if rng.random() < 0.6:
                t, sel = float(rng.uniform(0, 1400)), int(rng.integers(1, 3))
                rows.append((f"L{k}", t, sel, 50.0))
                for f in (f"L{k}F1", f"L{k}F2"):
                    s2 = sel if rng.random() < copy else int(rng.integers(1, 3))
                    rows.append((f, t + float(rng.uniform(5, 60)), s2, 40.0))
        for i, (c, secs, sel, stake) in enumerate(rows):
            all_bets.append(bets.Bet(bet_id=f"{code}-{i}", match_code=code, time=start + dt.timedelta(seconds=secs),
                                     market_type=1, selection=sel, odds=1.9, stake=stake, revenue=0.0,
                                     extra={config.BET_GROUP_COLUMN: "FANDUEL", config.BET_IN_PLAY_COLUMN: "Yes",
                                            config.BET_VIP_COLUMN: "Standard", COL: c}))
    joined = [dict(result=None) for _ in all_bets]
    return all_bets, joined


class TestRuns(unittest.TestCase):

    def test_the_money_after_leaves_the_customer_out(self):
        t = T0
        b = lambda i, c, s, sel, stake: bets.Bet(bet_id=i, match_code="M1", time=t + dt.timedelta(seconds=s), market_type=1,
                                                 selection=sel, odds=2.0, stake=stake, revenue=0.0,
                                                 extra={config.BET_IN_PLAY_COLUMN: "Yes", COL: c})
        every = [b("a", "X", 0, 1, 10), b("b", "X", 5, 1, 100), b("c", "Y", 10, 1, 30), b("d", "Z", 20, 2, 10),
                 b("e", "Y", 500, 1, 10)]
        recs = br.follow(br.records(every, [{}] * 5, {}, COL), after=120)
        a = recs[0]
        self.assertAlmostEqual(a["after"], (30 - 10) / 40)              # X's own 100 out, Y's 500s too late
        self.assertEqual(a["after_stake"], 40)
        self.assertTrue(a["first"])
        self.assertEqual(a["followers"], 1)                             # Y on its side
        self.assertEqual([recs[j]["customer"] for j in a["follow_ix"]], ["Y"])
        self.assertFalse(recs[2]["first"])                              # X was on home before Y

    def test_planted_leaders_are_found(self):
        every, joined = _world()
        out = br.analyse(every, joined, {}, COL, min_matches=10)
        top = {r["customer"] for r in sorted(out["customers"], key=lambda r: -r["shrunk"])[:6]}
        self.assertEqual(top, {f"L{k}" for k in range(6)})
        self.assertGreater(out["trait"]["r"], 0.3)
        pair = {(p["leader"], p["follower"]): p for p in out["pairs"]}
        self.assertGreater(pair[("L0", "L0F1")]["followed"], 50)
        self.assertLess(pair.get(("L0F1", "L0"), {"followed": 0})["followed"], 10)
        text = "\n".join(br.report(out))
        for part in ("WHO STARTS A RUN", "is it a trait?", "pairs:", "L0"):
            self.assertIn(part, text)

    def test_no_leader_is_no_trait(self):
        every, joined = _world(copy=0.0)
        out = br.analyse(every, joined, {}, COL, min_matches=10)
        self.assertLess(out["trait"]["lo"], 0)
        self.assertGreater(out["trait"]["hi"], 0)

    def test_no_ids_says_so(self):
        every, joined = _world(n=5)
        for b in every:
            b.extra[COL] = " "
        out = br.analyse(every, joined, {}, COL)
        self.assertEqual(out["ids"], 0)
        self.assertIn("nothing to probe", "\n".join(br.report(out)))

    def test_the_column_looked_for(self):
        self.assertEqual(br.find_column(["A", COL], COL), (COL, []))
        self.assertEqual(br.find_column(["A", "CUSTOMER_KEY", "ACCOUNT_REF"], COL), (None, ["CUSTOMER_KEY", "ACCOUNT_REF"]))

        class Cur:
            description = [("BET_ID",), ("CUSTOMER_KEY",)]

            def execute(self, sql, params):
                self.sql = sql

            def fetchall(self):
                return []
        column, note = bm._customer_column(Cur(), COL)
        self.assertIsNone(column)
        self.assertIn("CUSTOMER_KEY", note)


if __name__ == "__main__":
    unittest.main()
