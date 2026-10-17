"""gamer-dig: the targeted gamers, the earlier-matches test, the match-so-far test and the snap a bet
followed, on toy games of gamers who stay as strong as they are."""

import csv
import datetime as dt
import os
import random
import tempfile
import unittest

import numpy as np

from .. import gamer_dig as gd
from .fakes import _fake_match


def _world(n=160, seed=5):
    rng = random.Random(seed)
    g = np.random.default_rng(seed)
    gamers = {f"G{k}": float(s) for k, s in enumerate(g.normal(0, 0.35, 8))}
    gamers["ACE"] = 0.6                                   # one the money will find
    names = sorted(gamers)
    data, hist = {}, []
    t0 = dt.datetime(2026, 9, 1, 8, 0)
    for i in range(n):
        t = t0 + dt.timedelta(minutes=45 * i)
        code = f"AF{i % 1000:03d}{t:%d%m%y}"
        a, b = rng.sample(names, 2)
        rows = _fake_match(code, rng, strength=(gamers[a], gamers[b]))
        for r in rows:
            r["prematch_prob_50"] = "0.5"
        data[code] = rows
        hist.append(dict(MATCH_CODE=code, PLAYER_1_HANDLE=a, PLAYER_2_HANDLE=b, PLAYER_1_TEAM="", PLAYER_2_TEAM="",
                         SCHEDULED_START_TIME_UTC=f"{t:%Y-%m-%d %H:%M:%S}", PLAYER_1_FINAL_SCORE=rows[-1]["final_p1"],
                         PLAYER_2_FINAL_SCORE=rows[-1]["final_p2"]))
    return data, hist


def _bets(data, hist, since):
    """A bet on each side of every window match at a few PLAY_OVERs, priced at even money."""
    rng = random.Random(1)
    out = []
    for h in hist:
        c = h["MATCH_CODE"]
        if gd.day(c) < since:
            continue
        fin = (int(h["PLAYER_1_FINAL_SCORE"]), int(h["PLAYER_2_FINAL_SCORE"]))
        if fin[0] == fin[1]:
            continue
        rows = data[c]
        for r in rng.sample(rows, 4):
            for home in (True, False):
                won = (fin[0] > fin[1]) == home
                stake = 400.0 if "ACE" in (h["PLAYER_1_HANDLE"] if home else h["PLAYER_2_HANDLE"]) else 40.0
                out.append(dict(match=c, home=home, in_play=True, stake=stake, revenue=-stake if won else stake,
                                won=won, p=0.5, message=int(r["message"]), time=""))
    return out


class TestGamerDig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.hist = _world()
        cls.since = dt.date(2026, 9, 3)
        cls.bets = _bets(cls.data, cls.hist, cls.since)

    def test_the_profile_of_a_side(self):
        snaps = gd.side_snaps(next(iter(self.data.values())))
        p = gd.profile(snaps[0])
        self.assertTrue(set(gd.PROFILE) <= set(p))
        self.assertGreater(p["n"], 5)
        self.assertTrue(0 <= p["explosive"] <= 1 and 0 <= p["turnover"] <= 1)
        self.assertIsNone(gd.profile([]))

    def test_the_money_finds_the_strong_gamer(self):
        dig = gd.Dig(self.data, self.hist, self.bets, self.since)
        top = gd.targeted(dig)
        self.assertEqual(top[0][0], "ACE")                    # the big winning stakes back him
        self.assertGreater(top[0][1], 0)
        named = gd.targeted(dig, ["ACE", "NOBODY"])
        self.assertEqual([r[0] for r in named], ["ACE", "NOBODY"])

    def test_a_run_and_the_session_before(self):
        gm = gd.gamer_matches(self.hist)
        g = "ACE"
        ms = gm[g]
        code = ms[5][1]
        earlier = gd.session_before(gm, g, code)
        self.assertTrue(all(c in {m for _, m, _ in ms[:5]} for c, _ in earlier))
        results = {h["MATCH_CODE"]: (int(h["PLAYER_1_FINAL_SCORE"]), int(h["PLAYER_2_FINAL_SCORE"])) for h in self.hist}
        n = gd.run_of(results, earlier)
        self.assertLessEqual(abs(n), len(earlier))

    def test_the_report_runs(self):
        lines, rows = gd.run(self.data, self.hist, self.bets, self.since)
        text = "\n".join(lines)
        self.assertIn("the gamers the money wins on", text)
        self.assertIn("earlier matches of the session", text)
        self.assertIn("every in-play bet", text)
        self.assertIn("the snap the bet followed", text)
        self.assertTrue(rows)
        self.assertTrue({"gamer", "run", "last", "mine_ypp", "theirs_ypp"} <= set(rows[0]))

    def test_the_bets_file_is_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bets_sim.csv")
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, ["match_code", "market", "selection", "in_play", "stake", "revenue", "result",
                                        "stream_prob", "message", "bet_time"])
                w.writeheader()
                w.writerow(dict(match_code="AF001010926", market="moneyline", selection="Home", in_play="True",
                                stake="10", revenue="-9", result="won", stream_prob="0.5", message="70", bet_time=""))
                w.writerow(dict(match_code="AF001010926", market="total", selection="Over", in_play="True",
                                stake="10", revenue="10", result="lost", stream_prob="0.5", message="70", bet_time=""))
                w.writerow(dict(match_code="AF001010926", market="moneyline", selection="Away", in_play="False",
                                stake="10", revenue="10", result="cashed out", stream_prob="0.5", message="", bet_time=""))
            got = gd.read_bets(path)
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0]["home"] and got[0]["in_play"] and got[0]["won"])
        self.assertEqual(got[0]["message"], 70)


if __name__ == "__main__":
    unittest.main()
