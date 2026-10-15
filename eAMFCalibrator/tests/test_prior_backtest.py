"""The prior backtest: reading each arm's expected points, prod's pre-match total, and a run graded
against prod with and without the rematch layer."""

import datetime as dt
import os
import random
import tempfile
import unittest
from unittest import mock

from .. import bets, config, prior_backtest as pb, snowflake_io

T0 = dt.datetime(2026, 9, 1, 18, 0)


class TestRead(unittest.TestCase):

    def write(self, d, name, text):
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_every_source_shape(self):
        with tempfile.TemporaryDirectory() as d:
            cad = self.write(d, "cadence_matches.csv",
                             "MatchId,Time,P1,P2,S1,S2,GameOfDay,trim:hl60:game|1,trim:hl60:game|2\n"
                             "AF1,2026-09-01 18:00:00,A,B,21,14,1,17.5,15.25\n")
            nb2 = self.write(d, "nb2.csv", "MATCH_CODE,Pred_P1_Points,Pred_P2_Points\nAF1,16,18\n")
            pvs = self.write(d, "pvs.csv", "MATCH_CODE,raw_home,raw_away,means_home\nAF1,15,15.5,9\n")
            bad = self.write(d, "bad.csv", "MATCH_CODE,x\nAF1,1\n")
            self.assertEqual(pb.read_points(f"{cad}@trim:hl60:game", [], []), {"AF1": (17.5, 15.25)})
            self.assertEqual(pb.read_points(nb2, [], []), {"AF1": (16.0, 18.0)})
            self.assertEqual(pb.read_points(pvs, [], []), {"AF1": (15.0, 15.5)})
            with self.assertRaises(SystemExit):
                pb.read_points(bad, [], [])

    def test_an_nb2_csv_carries_its_own_score_model(self):
        with tempfile.TemporaryDirectory() as d:
            nb2 = self.write(d, "nb2.csv", "Match Id,Pred_P1_Points,Pred_P2_Points,P1_Effective_Alpha,"
                                           "P2_Effective_Alpha,Shared_Gamma_Fraction\nAF1,16,18,0.12,0.1,0.45\n")
            plain = self.write(d, "plain.csv", "MATCH_CODE,exp_p1,exp_p2\nAF1,16,18\n")
            points, own = pb.read_arm(nb2, [], [])
            self.assertEqual(points, {"AF1": (16.0, 18.0)})
            self.assertEqual(own, {"AF1": ("nb2", (16.0, 18.0, 0.12, 0.1, 0.45))})
            self.assertEqual(pb.read_arm(plain, [], [])[1], {})

    def test_an_arms_mode(self):
        self.assertEqual(pb.parse_arm("x.csv@trim:hl60:game#fitted"), ("x.csv@trim:hl60:game", "fitted"))
        self.assertEqual(pb.parse_arm("rolling:d"), ("rolling:d", None))
        with self.assertRaises(SystemExit):
            pb.parse_arm("x.csv#poisson")

    def test_each_mode_prices_a_match(self):
        r = dict(match="AF1", start=T0, margin=3.0, total=36.0, move_margin=1.0, move_total=-2.0,
                 margin_adj=4.0, total_adj=34.0, actual_margin=7, actual_total=35)
        prod = {"AF1": {"p1": 0.6, "line": 35.5, "over": 0.5}}
        normal, _ = pb.readings("m", [r], prod, "normal")
        own, _ = pb.readings("m", [r], prod, "own", {"AF1": ("nb2", (19.5, 16.5, 0.1, 0.1, 0.4))}, draws=20000)
        for got in (normal, own):
            p, t, po = got["AF1"]["m"]
            p2, t2, po2 = got["AF1"]["m+layer"]
            self.assertTrue(0.5 < p < p2 < 1)                  # the layer adds a point to the margin
            self.assertEqual((t, t2), (36.0, 34.0))
            self.assertGreater(po, po2)                         # and takes two off the total
        self.assertNotAlmostEqual(normal["AF1"]["m"][0], own["AF1"]["m"][0], places=3)
        p_prod = pb.readings(pb.PROD, [r], prod)[0]["AF1"]
        self.assertEqual(p_prod[pb.PROD], (0.6, 35.5, 0.5))

    def test_prods_closing_quote_is_the_last_before_the_first_play(self):
        def q(minutes, p, msg, line=None, market=50):
            return ("AF1", market, T0 + dt.timedelta(minutes=minutes), 100.0 * p, None,
                    None if line is None else f"Line {line}", msg, "open", "true")
        tl = bets.timeline([q(-60, 0.5, None), q(-5, 0.6, 1), q(-1, 0.62, 2), q(2, 0.9, 4),
                            q(-60, 0.5, None, 40.5, 54), q(-5, 0.55, 1, 38.5, 54), q(2, 0.7, 4, 38.5, 54)])
        self.assertEqual(pb.prematch_p1(tl, "AF1"), 0.5)                                 # as read before
        self.assertEqual(pb.prematch_p1(tl, "AF1", first_play=3), 0.62)                  # below the first play
        self.assertEqual(pb.prematch_p1(tl, "AF1", first_play=2), 0.6)
        self.assertEqual(pb.prematch_p1(tl, "AF1", kickoff=T0), 0.62)                    # no play feed: kick-off
        self.assertEqual(pb.prematch_total(tl, "AF1", 3, T0), (38.5, 0.55))
        self.assertEqual(pb.prematch_total(tl, "AF1"), (40.5, 0.5))
        pre = bets.closing_timeline(tl, {"AF1": 3}, {})
        self.assertEqual(pb.prematch_p1(pre, "AF1"), 0.62)                               # messages dropped
        self.assertEqual(pre[("AF1", 50)][1][-1], (None, 0.62, None, True))
        self.assertEqual(len(pre[("AF1", 50)][0]), 3)                                    # the in-play one out

    def test_prods_closing_total_and_its_over(self):
        def q(line, p, msg=None):
            return ("AF1", 54, T0, 100.0 * p, None, f"Line {line}", msg, "open", "true")
        tl = bets.timeline([q(36.5, 0.48), q(37.5, 0.51), q(40.5, 0.6, msg=9)])
        self.assertEqual(pb.prematch_total(tl, "AF1"), (37.5, 0.51))
        self.assertEqual(pb.prematch_total(tl, "AF2"), (None, None))


class TestRun(unittest.TestCase):

    def test_prod_and_an_arm_each_with_the_layer(self):
        rnd = random.Random(4)
        hist, quotes, arm = [], [], ["MATCH_CODE,exp_p1,exp_p2"]
        k = 0
        for d in range(24):
            for pair in range(6):
                a, b = f"A{pair}", f"B{pair}"
                um, ut = rnd.gauss(0, 4), rnd.gauss(0, 5)             # each pair's day
                for j in range(3):
                    code = f"AF{k:05d}"
                    start = T0 + dt.timedelta(days=d, minutes=40 * j + pair)
                    mu1, mu2 = 17 + rnd.gauss(0, 2), 17 + rnd.gauss(0, 2)
                    s1 = max(0, round(mu1 + um / 2 + ut / 2 + rnd.gauss(0, 7)))
                    s2 = max(0, round(mu2 - um / 2 + ut / 2 + rnd.gauss(0, 7)))
                    hist.append(dict(MATCH_CODE=code, SCHEDULED_START_TIME_UTC=str(start), PLAYER_1_HANDLE=a,
                                     PLAYER_2_HANDLE=b, PLAYER_1_TEAM="Ravens", PLAYER_2_TEAM="Lions",
                                     PLAYER_1_FINAL_SCORE=str(s1), PLAYER_2_FINAL_SCORE=str(s2)))
                    p = pb.rematch_layer.win_chance(mu1 - mu2)
                    opening, closing = start - dt.timedelta(minutes=60), start - dt.timedelta(minutes=5)
                    line = round(mu1 + mu2) + 0.5
                    # prod opens at evens with no message, closes on a pre-game message, then goes in play
                    quotes += [(code, 50, opening, 50.0, None, None, None, "open", "true"),
                               (code, 54, opening, 50.0, None, f"Line {line + 5}", None, "open", "true"),
                               (code, 50, closing, 100 * p, None, None, 1, "open", "true"),
                               (code, 54, closing, 50.0, None, f"Line {line}", 1, "open", "true"),
                               (code, 50, start + dt.timedelta(minutes=2), 90.0, None, None, 5, "open", "true")]
                    arm.append(f"{code},{mu1:.3f},{mu2:.3f}")
                    k += 1
        by = defaultdict_list(quotes)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "arm.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(arm) + "\n")
            with mock.patch.object(snowflake_io, "fetch_history", lambda cur: hist), \
                    mock.patch.object(snowflake_io, "fetch_quotes",
                                      lambda cur, s, codes: [q for c in codes for q in by.get(c, [])]), \
                    mock.patch.object(snowflake_io, "fetch_first_plays",     # half by the play feed, half by kick-off
                                      lambda cur, codes: {c: 3 for c in codes if int(c[2:]) % 2 == 0}), \
                    mock.patch.object(config, "MATCH_CHUNK_SIZE", 200):
                out = pb.run(None, d, [("model", path)], "2026-09-01 00:00:00", None)
                with open(out, encoding="utf-8") as fh:
                    text = fh.read()
                self.assertTrue(os.path.exists(os.path.join(d, "prior_backtest.csv")))
        for arm_name in ("prod ", "prod+layer", "prod+infer", "prod-early", "model ", "model+layer", "model+infer"):
            self.assertIn(arm_name, text)
        rows = {line.split()[0]: line.split() for line in text.splitlines()[:20] if line.strip().startswith("prod")}
        self.assertLess(float(rows["prod"][1]), float(rows["prod-early"][1]))     # the closing price is the sharper
        self.assertIn("Rematches", text)
        self.assertIn("The layers' last fits", text)
        self.assertIn("NB2's construction fitted to its errors", text)


def defaultdict_list(quotes):
    out = {}
    for q in quotes:
        out.setdefault(q[0], []).append(q)
    return out


if __name__ == "__main__":
    unittest.main()
