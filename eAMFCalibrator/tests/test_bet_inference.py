"""Inference: each gamer's other games that session read at a match (A beat B, C beat D, then A v D),
the form side and a common opponent, bets read against them, and a report and run over planted form."""

import datetime as dt
import os
import random
import tempfile
import unittest
from unittest import mock

from .. import bet_inference as bi, bet_rematches as br, bets, config, snowflake_io
from .test_bets import bet

T0 = dt.datetime(2026, 10, 1, 18, 0)


def game(code, minutes, home, away, s1, s2, t1="Ravens", t2="Lions"):
    return dict(MATCH_CODE=code, SCHEDULED_START_TIME_UTC=str(T0 + dt.timedelta(minutes=minutes)),
                PLAYER_1_HANDLE=home, PLAYER_2_HANDLE=away, PLAYER_1_TEAM=t1, PLAYER_2_TEAM=t2,
                PLAYER_1_FINAL_SCORE=str(s1), PLAYER_2_FINAL_SCORE=str(s2))


def quote(match, market, p, line=None, t=T0):
    return (match, market, t, 100.0 * p, None, None if line is None else f"Line {line}", None, "open", "true")


# A beats B by 14 and C beats D by 7; then A v D and C v B; ZED beats D and loses to A first
HISTORY = [game("G1", 0, "ANN", "BOB", 24, 10),
           game("G2", 0, "CAT", "DAN", 17, 10),
           game("G3", 40, "ZED", "DAN", 21, 3),
           game("G4", 40, "ANN", "ZED", 14, 13, t1="Bills"),
           game("G5", 80, "ANN", "DAN", 20, 17, t1="Bills"),
           game("G6", 80, "CAT", "BOB", 3, 10),
           game("G7", 120, "ANN", "BOB", 7, 0)]


class TestReading(unittest.TestCase):

    def setUp(self):
        self.meets = br.day_meetings(HISTORY)
        self.games = bi.session_games(self.meets)
        before = dt.datetime(2026, 9, 1)                  # before test_bets' bet times
        self.tl = bets.timeline([q for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7")
                                 for q in (quote(g, 50, 0.5, t=before), quote(g, 51, 0.5, t=before),
                                           quote(g, 54, 0.5, 30.5, t=before))])

    def test_a_beat_b_and_c_beat_d_then_a_v_d(self):
        r = bi.match_row(self.meets["G5"], [], self.tl, self.games)
        self.assertEqual((r["home_last"], r["away_last"]), (1, -18))     # ANN beat ZED by 1, DAN lost to ZED by 18
        self.assertEqual((r["state"], r["form"], r["gap"], r["gap_band"]), ("won v lost", "ANN", 19, "15-21"))
        self.assertEqual(r["played"], "first, after others")
        self.assertAlmostEqual(r["home_last_surprise"], 1.0)              # prod had it even: the margin itself
        self.assertEqual(r["form_same_team"], "same team as his last other")
        self.assertEqual((r["form_p"], r["form_won"], r["form_res"]), (0.5, 1.0, 0.5))
        self.assertEqual((r["home_last_miss"], r["away_last_miss"]), (27 - 30.5, 24 - 30.5))
        # ZED: ANN beat him and DAN lost to him -- a common opponent
        self.assertEqual((r["common"], r["common_side"], r["common_chain"], r["common_opponent"]),
                         ("beat v lost to", "ANN", 19, "ZED"))

    def test_c_v_b_and_a_rematch(self):
        r = bi.match_row(self.meets["G6"], [], self.tl, self.games)
        self.assertEqual((r["state"], r["form"]), ("won v lost", "CAT"))   # CAT beat DAN, BOB lost to ANN
        self.assertEqual(r["form_won"], 0.0)
        r = bi.match_row(self.meets["G7"], [], self.tl, self.games)
        self.assertEqual(r["played"], "rematch")
        # their other games: ANN beat ZED and DAN, BOB beat CAT -- the last of each
        self.assertEqual((r["home_last"], r["away_last"], r["state"]), (3, 7, "won v won"))
        self.assertEqual(r["form"], "BOB")
        first = bi.match_row(self.meets["G1"], [], self.tl, self.games)
        self.assertEqual((first["state"], first["form"], first["played"]), ("fresh v fresh", None, "first, no others"))

    def test_a_bet_reads_against_the_form_side(self):
        m = bi.match_row(self.meets["G5"], [], self.tl, self.games)
        on_d = bet("G5", 0, 1, 2, stake=10.0, revenue=10.0, odds=2.0)       # the moneyline on DAN (away)
        (j,) = bets.join([on_d], {}, {}, self.tl, {}, None, {})
        r = bi.bet_row(on_d, j, m)
        self.assertEqual((r["side"], r["co_side"], r["state"], r["res"]), (bi.OTHER, bi.OTHER, "won v lost", -0.5))
        over = bet("G5", 0, 3, 1, stake=10.0, revenue=10.0, odds=1.9, line=30.5)
        (j,) = bets.join([over], {}, {}, self.tl, {}, None, {})
        self.assertEqual(bi.bet_row(over, j, m)["side"], "Over")


def planted(n_sessions=40, seed=3):
    """Sessions of six gamers, each with a form that night prod's even prices don't see."""
    rnd = random.Random(seed)
    hist, quotes, k = [], [], 0
    names = ["A", "B", "C", "D", "E", "F"]
    for s in range(n_sessions):
        form = {g: rnd.gauss(0, 6) for g in names}
        for game_no in range(12):
            order = names[:]
            rnd.shuffle(order)
            for i in range(0, 6, 2):
                h, a = order[i], order[i + 1]
                d = form[h] - form[a] + rnd.gauss(0, 8)
                t = 40 + rnd.gauss(0, 10)
                s1, s2 = max(0, round((t + d) / 2)), max(0, round((t - d) / 2))
                code = f"M{k:05d}"
                minute = s * 1440 + 40 * game_no + i
                hist.append(game(code, minute, h, a, s1, s2, t1=rnd.choice(("Ravens", "Lions")), t2="Bills"))
                start = T0 + dt.timedelta(minutes=minute - 5)
                quotes += [quote(code, 50, 0.5, t=start), quote(code, 51, 0.5, t=start),
                           quote(code, 54, 0.5, 40.5, t=start), quote(code, 55, 0.5, 40.5, t=start)]
                k += 1
    return hist, quotes


class TestReport(unittest.TestCase):

    def test_planted_form_shows_and_the_layer_reads_it(self):
        hist, quotes = planted()
        meets = br.day_meetings(hist)
        games = bi.session_games(meets)
        tl = bets.timeline(quotes)
        rnd = random.Random(9)
        brows, mrows = [], []
        for code, m in sorted(meets.items(), key=lambda kv: kv[1]["start"]):
            m0 = bi.match_row(m, [], tl, games)
            rows = []
            for j in range(3):
                b = bet(code, j, rnd.choice((1, 2, 3)), rnd.choice((1, 2)), stake=20.0, revenue=rnd.choice((20.0, -18.0)),
                        odds=1.9, line=40.5, in_play=rnd.choice(("Yes", "No")),
                        **{config.BET_VIP_COLUMN: rnd.choice(("VIP", "Restricted", "Standard"))})
                (jr,) = bets.join([b], {}, {}, tl, {}, None, {})
                rows.append(bi.bet_row(b, jr, m0))
            brows += rows
            mrows.append(bi.match_row(m, rows, tl, games))
        wl = [r for r in mrows if r["state"] == "won v lost" and r["form_won"] is not None]
        self.assertGreater(sum(r["form_won"] for r in wl) / len(wl), 0.6)      # form carries over; prod's evens don't
        text = "\n".join(bi.report(brows, mrows))
        for heading in ("Every match by the state", "First meetings of the pair that session, by the state",
                        "Won v lost, by the gap", "A common opponent", "Handicap: the money on the form side",
                        "Totals against prod's line", "How far to move a price", "The 25 matches"):
            self.assertIn(heading, text)
        info = {r["match_code"]: r for r in mrows}
        rows, layers = bi.prod_layers(meets, sorted(meets), tl, info, min_days=7)
        self.assertGreater(layers["inference"].gamer["same team"][0], 15.0)        # 2 x 6^2 apart: one gamer's 36
        live = [r for r in rows if r["active"] and r["played"] == "first, after others"]
        from .. import rematch_layer as rl
        base, inf = rl.score(live), rl.score(live, "infer_margin_adj", "infer_total_adj")
        self.assertLess(inf["log_loss"], base["log_loss"] - 0.02)
        text = "\n".join(bi.layer_lines(rows, layers, 7))
        self.assertIn("first, won v lost", text)

    def test_the_run(self):
        hist, quotes = planted(n_sessions=10)
        by = {}
        for q in quotes:
            by.setdefault(q[0], []).append(q)
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID", "ODDS",
                "STAKE_GBP", "REVENUE_GBP", "MARKET_LINE", "BET_PLACED_PERIOD_NUMBER", "OPERATOR_NAME",
                "BET_IN_PLAY", "CUSTOMER_TEMPERATURE"]
        rows = [(i, h["MATCH_CODE"], dt.datetime.fromisoformat(h["SCHEDULED_START_TIME_UTC"]) - dt.timedelta(minutes=2),
                 1, 1 + i % 2, 1.9, 10.0, 10.0 if i % 3 else -9.0, None, 1, "OPX", "No", "Standard")
                for i, h in enumerate(hist)]
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(snowflake_io, "fetch_history", lambda cur: hist), \
                    mock.patch.object(snowflake_io, "fetch_quotes",
                                      lambda cur, s, codes: [q for c in codes for q in by.get(c, [])]), \
                    mock.patch.object(bets, "fetch_all", lambda cur, sql, params: (cols, rows)), \
                    mock.patch.object(snowflake_io, "fetch_first_plays", lambda cur, codes: {}), \
                    mock.patch.object(config, "CUTOFF_START", str(T0)), mock.patch.object(config, "CUTOFF_END", None), \
                    mock.patch("builtins.print"):
                path = bi.run(None, d)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertTrue(os.path.exists(os.path.join(d, "bets_inference.csv")))
        self.assertIn("The layers on prod's prices", text)
        self.assertIn("won v lost", text)


if __name__ == "__main__":
    unittest.main()
