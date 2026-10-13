"""The moneyline sweep: each bet against prod's price and the game at the bet, the matches against
their results, runs and pile-ons, and the report over every match."""

import random
import unittest

from .. import bet_checks, bet_losers as bl, bet_moments, bets, config
from .test_bets import at, bet

# M1: player 1 (ANN, home) drives into the red zone and scores at message 5; 14-7 at the end
SROWS = [("M1", 1, 240, "FIRST_QUARTER_STARTED", None, None, None, None, None, at(10)),
         ("M1", 2, 240, None, "PLAY_STARTED", "TEAM_A", 1, 10, 75, at(20)),
         ("M1", 3, 236, None, "PLAY_OVER", "TEAM_A", 1, 10, 75, at(30)),
         ("M1", 4, 230, None, "PLAY_STARTED", "TEAM_A", 2, 2, 83, at(40)),
         ("M1", 5, 226, None, "TOUCHDOWN_TEAM_A", "TEAM_A", None, None, None, at(50)),
         ("M1", 6, 226, None, "PLAY_OVER", "TEAM_A", None, None, None, at(60))]
SCORES = [("M1", 5, None, None, None, 6, 0)]


def q(market, seconds, prob, message, line=None):
    return ("M1", market, at(seconds), 100.0 * prob, None, None if line is None else f"Line {line}",
            message, "open", "true")


PROD = [q(50, 0, 0.60, None), q(51, 0, 0.40, None), q(54, 0, 0.50, None, 44.5), q(55, 0, 0.50, None, 44.5)]
for _m, _p in ((1, 0.60), (3, 0.62), (4, 0.70), (5, 0.80), (6, 0.80)):
    PROD += [q(50, 10 * _m, _p, _m), q(51, 10 * _m, 1 - _p, _m)]
SNAPS = {"M1": [{"message": 3, "offense": "TEAM_A", "next_offense": "TEAM_A", "next_field_position": 83,
                 "next_down": 2, "next_distance": 2, "team_a_side": "home"}]}
INFO = {"M1": {"MATCH_CODE": "M1", "PLAYER_1_HANDLE": "ANN", "PLAYER_2_HANDLE": "BOB",
               "PLAYER_1_TEAM": "Chiefs", "PLAYER_2_TEAM": "Bills",
               "SCHEDULED_START_TIME_UTC": "2026-09-20 12:00:00", "STREAM_NUMBER": "1"}}
POS = {("ANN", "M1"): (3, 1, 3), ("BOB", "M1"): (1, 4, 4)}
FINALS = {"M1": (14, 7)}


def context():
    times = bet_checks.message_times(PROD)
    checks = bet_checks.Checks(bet_checks.build_feeds(SROWS), times, bet_checks.score_index(SCORES))
    checks.timelines = bet_moments.build(SROWS, times)
    return bl.Context(bets.timeline(PROD), checks, SNAPS, SROWS, INFO, POS,
                      {"M1": {"slow": 0.55, "fast": 0.70}}), checks


def ml(seconds, selection=1, in_play="Yes", customer="C1", stake=10.0, revenue=-9.0, odds=1.9):
    return bet("M1", seconds, 1, selection, odds=odds, stake=stake, revenue=revenue, in_play=in_play,
               **{config.BET_VIP_COLUMN: "Restricted", "CUSTOMER_NAME_HASH": customer})


def rows_for(bs):
    ctx, checks = context()
    joined = bets.join(bs, {}, {}, ctx.tl, {}, None, FINALS, checks)
    return [bl.bet_row(b, r, ctx) for b, r in zip(bs, joined)], ctx


class TestBet(unittest.TestCase):

    def test_an_in_play_bet_reads_the_game_on_the_screen(self):
        (r,), _ = rows_for([ml(35)])
        self.assertEqual((r["side"], r["backed"], r["opposed"], r["backed_team"]), ("home", "ANN", "BOB", "Chiefs"))
        self.assertAlmostEqual(r["p"], 0.62)
        self.assertEqual((r["won"], r["fav"], r["lead"], r["lead_band"]), (1.0, "favourite", 0, "level"))
        self.assertAlmostEqual(r["res"], 0.38)
        # between plays after message 3: ANN on the opponent's 17, 2nd and 2
        self.assertEqual(r["moment"], bet_moments.BETWEEN)
        self.assertEqual((r["ball"], r["zone"], r["down_dist"], r["territory"]),
                         (bl.BACKED, "backed: red zone", "2nd & short", 83))
        # ANN scored 15s later, and prod went 0.62 -> 0.80 their way
        self.assertEqual((r["next_scorer"], r["next_score"]), (bl.BACKED, "backed within 30s"))
        self.assertAlmostEqual(r["after_p60"], 0.18)
        self.assertEqual(r["after"], "their way")
        self.assertEqual(r["session"], "backed on his last")
        self.assertAlmostEqual(r["elo_fast_gap"], 0.10)
        # 4 seconds of 960 played, nothing on the board against a 44.5 line
        self.assertAlmostEqual(r["running"], -44.5 * 4 / 960)
        self.assertEqual(r["running_band"], "on pace")

    def test_backing_the_other_side_turns_everything_round(self):
        (r,), _ = rows_for([ml(35, selection=2, revenue=10.0)])
        self.assertEqual((r["side"], r["backed"], r["fav"], r["won"]), ("away", "BOB", "underdog", 0.0))
        self.assertEqual((r["ball"], r["zone"], r["territory"]), (bl.OPPONENT, "opp: red zone", 17))
        self.assertEqual(r["next_scorer"], bl.OPPONENT)
        self.assertEqual(r["session"], "opponent on his last")
        self.assertAlmostEqual(r["elo_fast_gap"], -0.10)

    def test_a_pre_match_bet_reads_prods_move_to_the_kick_off(self):
        (r,), _ = rows_for([ml(5, in_play="No")])
        self.assertEqual(r["phase"], "pre-match")
        self.assertAlmostEqual(r["to_close"], 0.0)
        self.assertEqual(r["closing"], "held")
        self.assertNotIn("ball", r)

    def test_the_match(self):
        rows, ctx = rows_for([ml(35, stake=100.0, revenue=-90.0), ml(36, selection=2, stake=50.0, revenue=50.0)])
        m = bl.match_row("M1", FINALS["M1"], ctx, rows, [])
        self.assertEqual((m["fav_won"], m["dog_won"], m["margin"], m["final_total"]), (1, 0, 7, 21))
        self.assertAlmostEqual(m["total_vs_line"], 21 - 44.5)
        self.assertEqual((m["ml_bets"], m["ml_stake"], m["ml_revenue"]), (2, 150.0, -40.0))
        self.assertAlmostEqual(m["winner_share"], 100 / 150)
        self.assertAlmostEqual(m["winner_bet_p"], 0.62)
        self.assertAlmostEqual(m["winner_min_p"], 0.60)
        self.assertEqual((m["session_match"], m["last_any"], m["last_both"]), (3, 1, 0))
        self.assertAlmostEqual(m["elo_fast_vs_fav"], 0.70 - 0.60)


class TestHelpers(unittest.TestCase):

    def test_elo_is_walked_forward(self):
        hist = [dict(MATCH_CODE=f"G{k}", SCHEDULED_START_TIME_UTC=f"2026-09-0{k + 1} 12:00:00",
                     PLAYER_1_HANDLE="A", PLAYER_2_HANDLE="B", PLAYER_1_FINAL_SCORE="21",
                     PLAYER_2_FINAL_SCORE="7") for k in range(3)]
        e = bl.elo_by_match(hist)
        self.assertEqual(e["G0"], {"slow": 0.5, "fast": 0.5})           # nothing read before it
        self.assertGreater(e["G1"]["fast"], e["G1"]["slow"])           # fast moves further
        self.assertGreater(e["G2"]["slow"], e["G1"]["slow"])

    def test_score_changes_either_way_and_a_correction_has_no_scorer(self):
        ch = bet_checks.Checks({}, {}, bet_checks.score_index(
            [("M", 5, 0, 0, 0, 7, 0), ("M", 6, 0, 0, 0, 7, 0), ("M", 9, 0, 0, 0, 7, 3), ("M", 12, 0, 0, 0, 7, 0)]))
        self.assertEqual(bl.score_change(ch, "M", 6), ("away", 9))
        self.assertEqual(bl.score_change(ch, "M", 4), ("home", 5))
        self.assertEqual(bl.score_change(ch, "M", 7, forward=False), ("home", 5))
        self.assertEqual(bl.score_change(ch, "M", 10), (None, 12))      # 3 taken off the board
        self.assertEqual(bl.score_change(ch, "M", 12), (None, None))
        self.assertEqual(bl.score_change(ch, "M", 2, forward=False), (None, None))

    def test_the_ball_off_the_snapshot(self):
        snap = {"next_offense": "TEAM_B", "next_field_position": 30, "next_down": 3,
                "next_distance": 9, "team_a_side": "away"}
        out = bl.ball(snap, "home", bet_moments.BETWEEN)          # TEAM_B is home here
        self.assertEqual((out["ball"], out["zone"], out["down_dist"], out["has_ball"]),
                         (bl.BACKED, "backed: own 20-49", "3rd & long", 1))
        self.assertEqual(bl.ball(snap, "home", bet_moments.KICK_NEXT)["ball"], "score, kick-off or conversion")
        self.assertEqual(bl.ball(dict(snap, team_a_side=None), "home", bet_moments.LIVE)["ball"], "unknown")

    def test_runs_split_on_the_gap_and_the_biggest_are_pile_ons(self):
        def r(sec, stake, side="home", match="M"):
            return dict(match_code=match, side=side, in_play=True, bet_time=at(sec), stake=stake)
        rs = [r(0, 10), r(30, 10), r(60, 10), r(200, 5), r(10, 10, "away")] + \
             [r(1000 + 100 * k, 1, match="N") for k in range(40)]
        runs = bl.runs(rs)
        self.assertEqual(sorted(len(x) for x in runs if x[0]["match_code"] == "M"), [1, 1, 3])
        piles, cut = bl.pile_ons(runs)
        self.assertEqual(cut, 5)            # the 95th percentile of 43 runs: 40 of 1, then 5, 10, 30
        self.assertEqual([[x["bet_time"] for x in p] for p in piles], [[at(0), at(30), at(60)]])

    def test_lead_changes_ignore_ties(self):
        self.assertEqual(bl.lead_changes([(7, 0), (7, 7), (7, 10), (14, 10), (14, 17)]), 3)
        self.assertEqual(bl.lead_changes([(0, 3), (3, 3), (3, 10)]), 0)

    def test_bands(self):
        self.assertEqual(bl.elapsed("Q3", 120), 0.625)
        self.assertEqual(bl.elapsed("OT", 100), 1.0)
        self.assertIsNone(bl.elapsed("unknown", 100))
        self.assertEqual(bl.p_band(0.33), "25-40%")
        self.assertEqual(bl.running_band(-7.5), "under 7+")
        self.assertEqual(bl.band(5, bl.STAKES), "<10")

    def test_the_worst_tenth_are_losing_matches(self):
        ms = [dict(ml_stake=10, ml_revenue=float(v)) for v in range(-5, 15)] + [dict(ml_stake=0, ml_revenue=0.0)]
        bl.classify(ms)
        self.assertEqual([m["class"] for m in ms[:6]], ["worst", "worst", "losing", "losing", "losing", "winning"])
        self.assertEqual(ms[-1]["class"], "no bets")


class TestReport(unittest.TestCase):

    def test_the_report_over_many_bets(self):
        rnd = random.Random(3)
        rows, mrows = [], []
        for k in range(40):
            match = f"M{k}"
            n = 0
            for j in range(30):
                p = rnd.uniform(0.05, 0.95)
                won = float(rnd.random() < p)
                stake = rnd.choice((5.0, 20.0, 100.0, 800.0))
                odds = 0.95 / p
                inp = j > 4
                rows.append(dict(
                    match_code=match, bet_time=at(60 * j), side=rnd.choice(("home", "away")),
                    group=rnd.choice(("Restricted", "VIP", "Standard", "none")), operator="OP",
                    customer=f"C{rnd.randrange(30)}", stake=stake,
                    revenue=stake - (stake * odds if won else 0.0), odds=odds, result="won" if won else "lost",
                    won=won, in_play=inp, phase=rnd.choice(("Q1", "Q2", "Q3", "Q4")) if inp else "pre-match",
                    match_phase="in play" if inp else "pre-match", excluded=False, p=p, res=won - p,
                    expected=1 - odds * p, backed=f"G{rnd.randrange(6)}", opposed=f"G{rnd.randrange(6)}",
                    backed_team="T", opposed_team="U", stake_band=bl.band(stake, bl.STAKES),
                    log_stake=1.0, pre_p=0.5, fav=rnd.choice(("favourite", "underdog")), drift=0.0,
                    elo_slow_gap=rnd.uniform(-0.1, 0.1), elo_fast_gap=rnd.uniform(-0.2, 0.2),
                    backed_session_match=rnd.randrange(1, 9), opp_session_match=rnd.randrange(1, 9),
                    session="neither on his last", hour_utc=rnd.randrange(24),
                    into_m60=0.0, into_m30=0.0, after_p30=0.0, after_p60=rnd.uniform(-0.1, 0.1),
                    after_p300=0.0, before="held", after="held", to_close=0.0, closing="held",
                    moment=bet_moments.BETWEEN, quarter="Q2", clock_left=100.0, price_age_seconds=3.0,
                    lead=rnd.randrange(-14, 15), lead_band="level", board=10, elapsed=0.4,
                    running=rnd.uniform(-10, 10), running_band="on pace", ball=bl.BACKED,
                    zone="backed: red zone", territory=80, has_ball=1, down=rnd.randrange(1, 5),
                    down_dist="3rd & long", next_score_s=20.0, next_scorer=bl.BACKED,
                    next_score="backed within 30s", since_score_s=100.0, plays_so_far=20,
                    clock_per_play=20.0, presnap_s=15.0))
                n += 1
        piles, cut = bl.pile_ons(bl.runs(rows), quantile=0.8, min_bets=1)
        for k, p in enumerate(piles):
            for r in p:
                r["pile_on"] = k
        for k in range(45):
            mine = [r for r in rows if r["match_code"] == f"M{k}"]
            m = dict(match_code=f"M{k}", ml_stake=sum(r["stake"] for r in mine),
                     ml_revenue=sum(r["revenue"] for r in mine), fav_p=0.6, total_vs_line=-4.0,
                     dog_won=k % 2, comeback=k % 3 == 0, session_match=k % 7 + 1, hour_utc=k % 24,
                     stream="1")
            m.update({key: rnd.random() for key, _ in bl.MATCH_FEATURES if key not in m})
            mrows.append(m)
        bl.classify(mrows)
        lines = bl.report(rows, mrows, piles, cut)
        text = "\n".join(lines)
        for section in ("1. Where", "2. The losing", "3. In play", "4. In play", "5. Pre-match",
                        "6. Pile-ons", "7. Gamers", "Customers"):
            self.assertIn(section, text)
        self.assertIn("most-bet", text)


if __name__ == "__main__":
    unittest.main()
