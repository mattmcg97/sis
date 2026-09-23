"""Unit tests for the eAMF model. Pure Python, no data files.

Run with:  python -m unittest eAMFModel.tests.test_model
"""

import csv
import math
import os
import re
import tempfile
import unittest

from .. import backtest, clock, dist, drive, feed, players, playover, strength, stream
from ..params import Params, version
from ..pricer import (AWAY, HOME, ML_AWAY, ML_HOME, SPREAD_AWAY, SPREAD_HOME,
                      TOTAL_OVER, TOTAL_UNDER, GameState, Model)

_MODEL = None


def model():
    """One solved model for the whole run: the chain grid takes a moment."""
    global _MODEL
    if _MODEL is None:
        _MODEL = Model(Params(prior_strength=1e9))
    return _MODEL


PRIOR = strength.Prior.from_lines(-2.5, 38.5)


def state(**kw):
    base = dict(period=2, elapsed_in_period=30, home_score=7, away_score=7,
                offense=HOME, down=1, field_position=25, distance=10,
                opening_receiver=HOME)
    base.update(kw)
    return GameState(**base)


class TestDist(unittest.TestCase):
    def test_convolve_sums(self):
        a = [0.5, 0.0, 0.0, 0.5]
        b = [0.25, 0.75]
        out = dist.convolve(a, b)
        self.assertAlmostEqual(sum(out), 1.0)
        self.assertAlmostEqual(out[0], 0.125)
        self.assertAlmostEqual(out[4], 0.375)

    def test_poisson_normalised_and_mean(self):
        p = dist.poisson(4.2)
        self.assertAlmostEqual(sum(p), 1.0)
        self.assertAlmostEqual(dist.mean(p), 4.2, places=6)

    def test_signed_tails(self):
        s = dist.Signed.empty(-3, 3)
        s.add(-2, 0.25)
        s.add(0, 0.25)
        s.add(3, 0.5)
        self.assertAlmostEqual(s.prob_above(0.5), 0.5)
        self.assertAlmostEqual(s.prob_below(-0.5), 0.25)
        self.assertAlmostEqual(s.prob_at(0), 0.25)


class TestDriveChain(unittest.TestCase):
    def setUp(self):
        self.chain = model().drives.chain(1.0)

    def ep(self, d, y, t):
        td, fg = self.chain.value(d, y, t)
        return drive.expected_points(td, fg, 0.05, 0.043)

    def test_field_position_helps(self):
        values = [self.ep(1, y, 10) for y in (5, 25, 50, 75, 90)]
        self.assertEqual(values, sorted(values))

    def test_first_down_beats_later_downs(self):
        self.assertGreater(self.ep(1, 40, 10), self.ep(2, 40, 10))
        self.assertGreater(self.ep(2, 40, 10), self.ep(3, 40, 10))
        self.assertGreater(self.ep(3, 40, 10), self.ep(4, 40, 10))

    def test_shorter_distance_helps(self):
        self.assertGreater(self.ep(3, 40, 2), self.ep(3, 40, 12))

    def test_fourth_down_deep_in_own_half_mostly_punts(self):
        td, fg = self.chain.value(4, 25, 8)
        self.assertLess(td, 0.15)                 # goes for it about a third of the time
        self.assertLess(fg, 0.05)
        self.assertLess(td, self.chain.value(3, 25, 8)[0])

    def test_fourth_down_in_range_is_mostly_a_kick(self):
        td, fg = self.chain.value(4, 80, 8)
        self.assertGreater(fg, td)
        self.assertGreater(fg, 0.4)

    def test_go_probability_follows_the_data(self):
        p = self.chain.params
        self.assertGreater(drive.go_probability(p, 35, 1), 0.8)     # 4th-and-1, own 35
        self.assertLess(drive.go_probability(p, 35, 15), 0.35)
        self.assertGreater(drive.go_probability(p, 35, 3, 1.0), drive.go_probability(p, 35, 3))
        self.assertAlmostEqual(drive.fg_make(p, 72), 0.949, places=2)   # a 45-yarder

    def test_aggression_changes_4th_down_value(self):
        dm = model().drives
        bold = dm.value(1.0, 4, 30, 3, drive.NORMAL, 1.5)[0]
        timid = dm.value(1.0, 4, 30, 3, drive.NORMAL, -1.5)[0]
        self.assertGreater(bold, timid)

    def test_goal_line_mostly_touchdowns(self):
        td, fg = self.chain.value(1, 99, 1)
        self.assertGreater(td, 0.8)

    def test_probabilities_bounded(self):
        for d in (1, 2, 3, 4):
            for y in (1, 30, 60, 95):
                td, fg = self.chain.value(d, y, 10)
                self.assertGreaterEqual(td, 0.0)
                self.assertGreaterEqual(fg, 0.0)
                self.assertLessEqual(td + fg, 1.0 + 1e-9)

    def test_quality_lifts_ep_and_inverts(self):
        dm = model().drives
        curve = dm.fresh_ep_curve()
        eps = [e for _, e in curve]
        self.assertEqual(eps, sorted(eps))
        for target in (1.5, 3.2, 5.0):
            q = dm.quality_for(target)
            td, fg = dm.value(q, 1, dm.params.start_field, 10)
            self.assertAlmostEqual(drive.expected_points(td, fg, dm.q6, dm.q8), target, delta=0.15)

    def test_outcome_pmf(self):
        pmf = drive.outcome_pmf(0.4, 0.1, 0.05, 0.05)
        self.assertAlmostEqual(sum(pmf), 1.0)
        self.assertAlmostEqual(pmf[0], 0.5)
        self.assertAlmostEqual(pmf[3], 0.1)
        self.assertAlmostEqual(pmf[7], 0.4 * 0.9)


class TestClock(unittest.TestCase):
    def setUp(self):
        self.p = clock.ClockParams()

    def test_shares_sum_to_one(self):
        self.assertAlmostEqual(sum(self.p.shares), 1.0)

    def test_kickoff_is_whole_game(self):
        self.assertAlmostEqual(clock.fraction_remaining(self.p, 1, 0), 1.0, places=2)
        self.assertAlmostEqual(clock.fraction_remaining(self.p, None, 0), 1.0, places=2)

    def test_monotone_through_the_game(self):
        points = [(1, 0), (1, 50), (2, 0), (2, 100), (3, 0), (4, 0), (4, 100), (4, 250)]
        values = [clock.fraction_remaining(self.p, p, e) for p, e in points]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[-1], 0.0)

    def test_overtime_has_no_regulation_left(self):
        self.assertEqual(clock.fraction_remaining(self.p, 5, 0), 0.0)

    def test_halves_split(self):
        first = clock.remaining(self.p, 1, 0)
        self.assertAlmostEqual(first.this_half, self.p.shares[0] + self.p.shares[1], places=2)
        self.assertAlmostEqual(first.next_half, self.p.shares[2] + self.p.shares[3])
        second = clock.remaining(self.p, 3, 0)
        self.assertEqual(second.next_half, 0.0)

    def test_residual_never_negative(self):
        for e in (0, 50, 120, 400, 2000):
            self.assertGreater(clock.residual(100, 25, e), 0.0)
            self.assertGreaterEqual(clock.residual_variance(100, 25, e), 0.0)


class TestHalfClock(unittest.TestCase):
    def setUp(self):
        self.p = clock.ClockParams()

    def test_default_is_the_half_clock(self):
        self.assertEqual(self.p.mode, "half")
        self.assertAlmostEqual(sum(self.p.half_shares), 1.0)

    def test_quarter_break_is_the_middle_of_the_half(self):
        a1, a2 = self.p.half_slopes
        s1, s2 = self.p.half_shares
        self.assertAlmostEqual(clock.remaining(self.p, 2, 0).this_half, s1 * 0.5 ** (1 + a1))
        self.assertAlmostEqual(clock.remaining(self.p, 4, 0).this_half, s2 * 0.5 ** (1 + a2))

    def test_no_step_at_the_quarter_break(self):
        # The end of the 3rd, run long, meets the start of the 4th.
        end_q3 = clock.remaining(self.p, 3, 400).this_half
        start_q4 = clock.remaining(self.p, 4, 0).this_half
        self.assertAlmostEqual(end_q3, start_q4, delta=0.01)

    def test_second_half_slopes_down(self):
        # Equal game time, less scoring left for it late than early.
        early = clock.half_left(self.p, 3, 0)[0] - clock.remaining(self.p, 4, 0).this_half / self.p.half_shares[1]
        late = clock.remaining(self.p, 4, 0).this_half / self.p.half_shares[1]
        self.assertGreater(early, late)

    def test_uncertainty_only_once_the_quarter_is_running(self):
        self.assertEqual(clock.remaining(self.p, 1, 0).this_half_var, 0.0)
        self.assertGreater(clock.remaining(self.p, 3, 60).this_half_var, 0.0)


class TestStrength(unittest.TestCase):
    def test_prior_from_lines(self):
        p = strength.Prior.from_lines(3.5, 40.5)
        self.assertAlmostEqual(p.home_points, 22.0)
        self.assertAlmostEqual(p.away_points, 18.5)

    def test_posterior_mean(self):
        [(w, theta)] = strength.multipliers(4.0, scored=14, expected_by_now=7,
                                            points_per_score=7.0, n_slices=1)
        self.assertAlmostEqual(theta, (4 + 2) / (4 + 1))

    def test_infinite_k_holds_the_prior(self):
        self.assertEqual(strength.multipliers(math.inf, 50, 1, 6.2), [(1.0, 1.0)])

    def test_slices_keep_the_mean(self):
        nodes = strength.multipliers(3.0, 7, 10, 6.2, n_slices=5)
        mean = sum(w * t for w, t in nodes)
        self.assertAlmostEqual(mean, (3 + 7 / 6.2) / (3 + 10 / 6.2))


class TestPricer(unittest.TestCase):
    def book(self, **kw):
        return model().book(PRIOR, state(**kw))

    def test_distributions_are_whole(self):
        b = self.book()
        self.assertAlmostEqual(b.margin.total(), 1.0, places=6)
        self.assertAlmostEqual(sum(b.total), 1.0, places=6)

    def test_selections_complement(self):
        b = self.book()
        self.assertAlmostEqual(b.prob(ML_HOME) + b.prob(ML_AWAY), 1.0)
        self.assertAlmostEqual(b.prob(SPREAD_HOME, -2.5) + b.prob(SPREAD_AWAY, 2.5), 1.0)
        self.assertAlmostEqual(b.prob(TOTAL_OVER, 38.5) + b.prob(TOTAL_UNDER, 38.5), 1.0)

    def test_push_is_graded_out(self):
        b = self.book()
        p = b.prob(SPREAD_HOME, 3)
        above = b.margin.prob_above(3)
        at = b.margin.prob_at(3)
        self.assertAlmostEqual(p, above / (1 - at))

    def test_fair_lines_near_evens(self):
        b = self.book()
        for market in (SPREAD_HOME, TOTAL_OVER):
            line = b.fair_line(market)
            self.assertLess(abs(b.prob(market, line) - 0.5), 0.12)

    def test_kickoff_reproduces_the_lines(self):
        b = model().book(PRIOR, GameState(1, 0, 0, 0))
        self.assertAlmostEqual(b.margin.mean(), -2.5, delta=0.3)
        self.assertAlmostEqual(dist.mean(b.total), 38.5, delta=0.6)

    def test_good_play_helps_the_offense(self):
        base = self.book(down=2, field_position=40, distance=8).p_home
        first_down = self.book(down=1, field_position=52, distance=10).p_home
        loss = self.book(down=3, field_position=35, distance=13).p_home
        self.assertGreater(first_down, base)
        self.assertLess(loss, base)

    def test_good_play_lifts_the_total(self):
        base = dist.mean(self.book(down=2, field_position=40, distance=8).total)
        gain = dist.mean(self.book(down=1, field_position=70, distance=10).total)
        self.assertGreater(gain, base)

    def test_away_offense_mirrors(self):
        home_ball = self.book(offense=HOME, field_position=70).p_home
        away_ball = self.book(offense=AWAY, field_position=70).p_home
        self.assertGreater(home_ball, away_ball)

    def test_touchdown_moves_the_moneyline(self):
        before = self.book().p_home
        after = self.book(home_score=14, offense=AWAY).p_home
        self.assertGreater(after, before)

    def test_big_late_lead_is_nearly_settled(self):
        b = self.book(period=4, elapsed_in_period=150, home_score=35, away_score=7,
                      offense=AWAY)
        self.assertGreater(b.p_home, 0.995)

    def test_end_of_half_cuts_the_current_drive(self):
        s = state(period=2, elapsed_in_period=150, field_position=60)
        on = model().book(PRIOR, s)
        off = Model(Params(prior_strength=1e9, drive_time=0.0)).book(PRIOR, s)
        self.assertLess(dist.mean(on.total), dist.mean(off.total))
        self.assertLess(on.p_home, off.p_home)
        # Early in the half the cut barely matters.
        s = state(period=1, elapsed_in_period=5, field_position=60)
        on = model().book(PRIOR, s)
        off = Model(Params(prior_strength=1e9, drive_time=0.0)).book(PRIOR, s)
        self.assertAlmostEqual(on.p_home, off.p_home, places=2)

    def test_scoreboard_moves_strength_when_allowed(self):
        reactive = Model(Params(prior_strength=2.0))
        stubborn = Model(Params(prior_strength=1e9))
        # Away was the underdog and is running away with it.
        s = state(period=2, elapsed_in_period=60, home_score=0, away_score=28, offense=HOME)
        prior = strength.Prior.from_lines(10.5, 38.5)
        self.assertLess(reactive.book(prior, s).p_home, stubborn.book(prior, s).p_home)
        self.assertLess(dist.mean(stubborn.book(prior, s).total),
                        dist.mean(reactive.book(prior, s).total))

    def test_overtime(self):
        b = self.book(period=5, elapsed_in_period=3, home_score=21, away_score=21,
                      offense=HOME, field_position=70)
        self.assertGreater(b.p_home, 0.5)
        done = self.book(period=5, elapsed_in_period=3, home_score=24, away_score=21)
        self.assertEqual(done.p_home, 1.0)

    def test_fit_prior_hits_targets(self):
        m = model()
        prior = m.fit_prior(-2.5, 38.5, ml_home=0.42, over=0.47)
        b = m.book(prior, GameState(1, 0, 0, 0))
        self.assertAlmostEqual(b.p_home, 0.42, places=2)
        self.assertAlmostEqual(b.prob(TOTAL_OVER, 38.5), 0.47, places=2)


class TestEndGame(unittest.TestCase):
    def late(self, **kw):
        base = dict(period=4, elapsed_in_period=100, home_score=14, away_score=19,
                    offense=HOME, down=1, field_position=40, distance=10,
                    opening_receiver=HOME)
        base.update(kw)
        return GameState(**base)

    def test_policies_follow_the_deficit(self):
        m = model()
        left = m._drives_of_clock(self.late())
        policies, kill = m._endgame(self.late(), left)
        self.assertEqual(policies[HOME], drive.NEED_TD)     # down 5
        self.assertEqual(kill[AWAY], "run")
        policies, _ = m._endgame(self.late(away_score=16), left)
        self.assertEqual(policies[HOME], drive.MUST_SCORE)  # down 2
        policies, kill = m._endgame(self.late(home_score=30, away_score=7), left)
        self.assertEqual(kill[HOME], "kneel")

    def test_early_or_first_half_is_normal(self):
        m = model()
        for s in (self.late(period=3, elapsed_in_period=0), self.late(period=2)):
            policies, kill = m._endgame(s, m._drives_of_clock(s))
            self.assertEqual(set(policies.values()), {drive.NORMAL})
            self.assertEqual(kill, {})

    def test_need_td_never_kicks_and_must_score_never_punts(self):
        dm = model().drives
        self.assertEqual(dm.value(1.0, 4, 80, 6, drive.NEED_TD)[1], 0.0)
        self.assertGreater(dm.value(1.0, 4, 20, 2, drive.MUST_SCORE)[0], 0.0)
        self.assertGreater(dm.value(1.0, 4, 20, 2, drive.MUST_SCORE)[0],
                           dm.value(1.0, 4, 20, 2, drive.NORMAL)[0])

    def test_kneeling_takes_points_off_the_total(self):
        s = self.late(home_score=24, away_score=14, offense=HOME, field_position=50,
                      elapsed_in_period=40)
        on = model().book(PRIOR, s)
        off = Model(Params(prior_strength=1e9, late_drives=0.0)).book(PRIOR, s)
        self.assertLess(dist.mean(on.total), dist.mean(off.total))

    def test_needing_a_touchdown_on_4th_goes_for_it(self):
        # Down 5 late, 4th-and-6 in field-goal range: the kick is worthless.
        s = self.late(down=4, distance=6, field_position=75)
        on = model().book(PRIOR, s)
        off = Model(Params(prior_strength=1e9, late_drives=0.0)).book(PRIOR, s)
        self.assertGreater(on.p_home, off.p_home)


class TestRealClock(unittest.TestCase):
    def test_kickoff_fit_on_the_clock_hits_the_targets(self):
        m = model()
        prior = m.fit_prior(-2.5, 38.5, ml_home=0.42, over=0.5, on_clock=True)
        b = m.book(prior, GameState(1, 0, 0, 0, clock_seconds=240))
        self.assertAlmostEqual(b.p_home, 0.42, places=2)
        self.assertAlmostEqual(b.prob(TOTAL_OVER, 38.5), 0.5, places=2)

    def test_the_clock_settles_a_late_lead_gradually(self):
        m = model()
        prices = [m.book(PRIOR, GameState(4, 0, 21, 17, HOME, 1, 40, 10, clock_seconds=c,
                                          opening_receiver=HOME)).p_home
                  for c in (240, 120, 60, 10)]
        self.assertEqual(prices, sorted(prices))
        self.assertLess(prices[1], 0.95)       # two minutes is time for a drive back
        self.assertGreater(prices[-1], 0.9)

    def test_the_scoring_curve(self):
        c = clock.ClockParams()
        self.assertEqual(c.scoring_curve[0], 0.0)
        self.assertAlmostEqual(c.scoring_curve[-1], 1.0)
        self.assertEqual(list(c.scoring_curve), sorted(c.scoring_curve))
        self.assertAlmostEqual(clock.remaining_on_clock(c, 1, 240).fraction, 1.0)
        self.assertAlmostEqual(clock.remaining_on_clock(c, 4, 0).fraction, 0.0)
        # The end of the 2nd quarter carries more than the start of the 1st.
        start_q1 = 1.0 - clock.remaining_on_clock(c, 1, 210).fraction
        end_q2 = clock.remaining_on_clock(c, 2, 30).this_half
        self.assertGreater(end_q2, 2 * start_q1)

    def test_overtime_is_a_timed_period(self):
        c = clock.ClockParams()
        self.assertAlmostEqual(clock.remaining_on_clock(c, 5, 240).fraction, c.ot_share)
        self.assertAlmostEqual(clock.remaining_on_clock(c, 5, 0).fraction, 0.0)
        # Up six in overtime is not a win yet: a touchdown answers it.
        b = model().book(PRIOR, GameState(5, 0, 23, 17, AWAY, 1, 25, 10, clock_seconds=150,
                                          snap_confirmed=True))
        self.assertLess(b.p_home, 0.9)

    def test_pending_conversion_adds_about_a_point(self):
        m = model()
        done = m.book(PRIOR, GameState(2, 0, 7, 0, AWAY, clock_seconds=100))
        pending = m.book(PRIOR, GameState(2, 0, 6, 0, AWAY, clock_seconds=100,
                                          pending_conversion=HOME))
        self.assertAlmostEqual(dist.mean(pending.total), dist.mean(done.total), delta=0.5)


class TestPlayOver(unittest.TestCase):
    def snap(self, **kw):
        r = {f: "" for f in ("match_code", "message", "period", "clock_seconds", "play_kind",
                              "team_a_side", "offense", "down", "distance", "field_position",
                              "next_offense", "next_down", "next_distance",
                              "next_field_position", "score_p1", "score_p2",
                              "score_p1_at_start", "score_p2_at_start", "play_messages",
                              "opening_offense")}
        r.update(match_code="AF1", message="20", period="2", clock_seconds="150",
                 play_kind="SCRIMMAGE", team_a_side="away", offense="TEAM_B", down="1",
                 distance="10", field_position="40", next_offense="TEAM_B", next_down="2",
                 next_distance="4", next_field_position="46", score_p1="7", score_p2="0",
                 score_p1_at_start="7", score_p2_at_start="0", opening_offense="TEAM_A")
        r.update({k: str(v) for k, v in kw.items()})
        return r

    def test_scrimmage_takes_the_row(self):
        state, _ = playover.state_for(self.snap())
        self.assertEqual((state.offense, state.down, state.distance, state.field_position),
                         (HOME, 1, 10, 40))       # TEAM_A is away, so TEAM_B is home
        self.assertTrue(state.snap_confirmed)
        self.assertEqual(state.clock_seconds, 150)
        self.assertEqual(state.opening_receiver, AWAY)
        state, _ = playover.state_for(self.snap(), playover.NEXT)
        self.assertEqual((state.down, state.field_position), (2, 46))

    def test_a_turnover_row_is_the_new_side(self):
        state, _ = playover.state_for(self.snap(offense="TEAM_A", field_position="55"))
        self.assertEqual((state.offense, state.field_position), (AWAY, 55))

    def test_touchdown_scorer_from_the_message_and_points_added(self):
        # A pick six: the touchdown message names the side that scored.
        state, _ = playover.state_for(self.snap(play_kind="TOUCHDOWN",
                                                play_messages="POSSESSION_TEAM_A|TOUCHDOWN_TEAM_A"))
        self.assertEqual(state.pending_conversion, AWAY)
        self.assertEqual((state.home_score, state.away_score), (7, 6))
        self.assertEqual(state.offense, HOME)
        state, _ = playover.state_for(self.snap(play_kind="TOUCHDOWN", score_p1="13",
                                                play_messages="TOUCHDOWN_TEAM_B"))
        self.assertEqual(state.home_score, 13)   # already there: not added twice

    def test_conversion_gives_the_other_side_the_ball(self):
        state, _ = playover.state_for(self.snap(play_kind="CONVERSION", score_p1="13",
                                                score_p1_at_start="13",
                                                play_messages="EXTRA_POINT_GOOD_TEAM_B"))
        self.assertEqual((state.home_score, state.offense), (14, AWAY))
        self.assertIsNone(state.down)

    def test_field_goals(self):
        state, _ = playover.state_for(self.snap(play_kind="FIELD_GOAL",
                                                play_messages="FIELD_GOAL_GOOD_TEAM_B"))
        self.assertEqual((state.home_score, state.offense), (10, AWAY))
        state, _ = playover.state_for(self.snap(play_kind="FIELD_GOAL", offense="TEAM_A",
                                                field_position="30",
                                                play_messages="FIELD_GOAL_MISSED_TEAM_B"))
        self.assertEqual((state.offense, state.field_position, state.home_score), (AWAY, 30, 7))

    def test_unresolved_team(self):
        state, why = playover.state_for(self.snap(team_a_side=""))
        self.assertIsNone(state)
        self.assertEqual(why, "team_a_unresolved")

    def test_runs_on_an_export(self):
        fields = list(self.snap()) + ["prematch_line_52", "prematch_prob_52", "prematch_line_54",
                                      "prematch_prob_54", "prematch_prob_50"]
        for m in (50, 51, 52, 53, 54, 55):
            fields += [f"line_{m}", f"prob_{m}", f"live_{m}", f"outcome_{m}"]
        rows = []
        for msg, clock_s, kind in ((20, 150, "SCRIMMAGE"), (30, 100, "TOUCHDOWN"),
                                   (40, 90, "KICKOFF")):
            r = self.snap(message=msg, clock_seconds=clock_s, play_kind=kind,
                          prematch_line_52=-2.5, prematch_prob_52=0.47, prematch_line_54=38.5,
                          prematch_prob_54=0.5, prematch_prob_50=0.44)
            r.update(line_50="", prob_50="0.6", live_50="1", outcome_50="1",
                     line_52="3.5", prob_52="0.55", live_52="1", outcome_52="1",
                     line_54="40.5", prob_54="0.5", live_54="0", outcome_54="0")
            rows.append(r)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "po.csv")
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            graded, skipped = playover.run(path, {"v1": version("v1")})
        self.assertEqual(len(graded), 6)                  # 3 snapshots x 2 live markets
        self.assertEqual(skipped["prod_not_live"], 3)
        self.assertTrue(all(0.0 <= g[2]["v1"] <= 1.0 for g in graded))


def _row(msg, clock_s, kind="SCRIMMAGE", offense="TEAM_A", down="1", distance="10",
         field="30", period="1", p1="0", p2="0"):
    return {"message": str(msg), "period": period, "clock_seconds": str(clock_s),
            "play_kind": kind, "offense": offense, "down": down, "distance": distance,
            "field_position": field, "score_p1": p1, "score_p2": p2}


class TestPlayers(unittest.TestCase):
    def match(self, gap, offense="TEAM_A"):
        rows, clock_s = [], 240
        for i in range(12):
            rows.append(_row(10 + i, clock_s, offense=offense))
            clock_s -= gap
        return rows

    def test_slow_and_fast_players(self):
        matches = {"AF1": self.match(30), "AF2": self.match(15)}
        handles = {"AF1": ("SLOW", "X"), "AF2": ("FAST", "Y")}
        book = players.build(matches, handles)
        self.assertGreater(book.profile("SLOW").pace, 1.0)
        self.assertLess(book.profile("FAST").pace, 1.0)
        self.assertEqual(book.profile("NOBODY").pace, 1.0)

    def test_aggression_reads_4th_down_choices(self):
        def fourth(went):
            a = _row(10, 200, down="4", distance="3", field="40")
            b = _row(11, 190, kind="SCRIMMAGE" if went else "PUNT")
            return [a, b]
        matches = {f"AF{i}": fourth(i % 5 != 0) for i in range(40)}
        handles = {code: ("BOLD", "X") for code in matches}
        book = players.build(matches, handles)
        self.assertGreater(book.profile("BOLD").aggression, 0.0)

    def test_pace_ratio_moves_with_the_game(self):
        book = players.Book(seconds={"0|tied": 24.0}, players={})
        pace = players.Pace(book, "A", "B", prior_seconds=100)
        self.assertEqual(pace.ratio(), 1.0)
        for _ in range(20):
            pace.add("TEAM_A", (False, "tied"), 36.0)
        self.assertGreater(pace.ratio(), 1.3)

    def test_a_leader_on_the_ball_late_plays_safe(self):
        s = GameState(4, 0, 21, 17, HOME, 1, 40, 10, clock_seconds=90, snap_confirmed=True)
        safe = Model(Params(prior_strength=1e9, run_score=0.6)).book(PRIOR, s)
        plain = Model(Params(prior_strength=1e9, run_score=0.0)).book(PRIOR, s)
        self.assertLess(dist.mean(safe.total), dist.mean(plain.total))


def _play(m, team, d, t, y, period=1):
    return feed.Play(m, period, team, d, t, y)


class TestFeed(unittest.TestCase):
    def verdicts(self, plays, scores=()):
        return {t.message: (t.verdict, t.reason) for t in feed.replay(plays, list(scores))}

    def test_opening_is_suspended_until_a_snap(self):
        v = self.verdicts([_play(6, HOME, 1, 10, 35), _play(10, HOME, 1, 10, 26),
                           _play(16, HOME, 2, 11, 26)])
        self.assertEqual(v[6][0], feed.SUSPENDED)
        self.assertEqual(v[10][0], feed.LIVE)
        self.assertEqual(v[16][0], feed.LIVE)

    def test_score_to_next_drive_is_suspended(self):
        plays = [_play(10, HOME, 1, 10, 26), _play(36, HOME, 3, 5, 90),
                 _play(42, HOME, 3, 5, 85),          # the extra point
                 _play(47, HOME, 1, 10, 35),         # the kick spot
                 _play(52, AWAY, 1, 10, 35),         # label flip, stale spot
                 _play(54, AWAY, 1, 10, 30)]         # the return: new drive
        scores = [feed.Score(40, 1, 6, 0), feed.Score(46, 1, 7, 0)]
        v = self.verdicts(plays, scores)
        for m in (40, 42, 46, 47, 52):
            self.assertEqual(v[m][0], feed.SUSPENDED, m)
        self.assertEqual(v[54][0], feed.LIVE)

    def test_state_carries_score_and_drive(self):
        plays = [_play(10, HOME, 1, 10, 26), _play(54, AWAY, 1, 10, 30),
                 _play(60, AWAY, 2, 4, 36)]
        ticks = feed.replay(plays, [feed.Score(40, 1, 7, 0)])
        last = ticks[-1].state
        self.assertEqual((last.home_score, last.away_score), (7, 0))
        self.assertEqual(last.offense, AWAY)
        self.assertEqual(last.drive_age, 6)
        self.assertEqual(last.opening_receiver, HOME)

    def test_stale_label_is_suspended(self):
        v = self.verdicts([_play(10, HOME, 1, 10, 26), _play(20, HOME, 4, 10, 33),
                           _play(22, AWAY, 4, 10, 33), _play(24, AWAY, 1, 10, 3)])
        self.assertEqual(v[22], (feed.SUSPENDED, feed.STALE))
        self.assertEqual(v[24][0], feed.LIVE)

    def test_impossible_down_is_suspended(self):
        v = self.verdicts([_play(10, AWAY, 1, 10, 25), _play(12, AWAY, 3, 10, 25)])
        self.assertEqual(v[12], (feed.SUSPENDED, feed.IMPOSSIBLE_DOWN))

    def test_new_half_waits_for_a_snap(self):
        plays = [_play(10, HOME, 1, 10, 26), _play(193, AWAY, 1, 10, 35, period=3),
                 _play(197, AWAY, 1, 10, 25, period=3)]
        v = self.verdicts(plays)
        self.assertEqual(v[193][0], feed.SUSPENDED)
        self.assertEqual(v[197][0], feed.LIVE)

    def test_onside_recovery_eventually_goes_live(self):
        plays = [_play(10, AWAY, 1, 10, 26), _play(350, AWAY, 3, 5, 91, 4),
                 _play(360, AWAY, 1, 10, 35, 4), _play(366, AWAY, 1, 10, 45, 4),
                 _play(384, AWAY, 2, 4, 87, 4)]
        v = self.verdicts(plays, [feed.Score(353, 4, 0, 7)])
        self.assertEqual(v[366][0], feed.SUSPENDED)
        self.assertEqual(v[384][0], feed.LIVE)

    def test_republish_keeps_the_verdict(self):
        v = self.verdicts([_play(10, HOME, 1, 10, 26), _play(15, HOME, 1, 10, 26)])
        self.assertEqual(v[15][0], feed.LIVE)


def _parse_line(text):
    cleaned = re.sub(r"player\s*\d+", " ", text, flags=re.IGNORECASE)
    found = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(found.group()) if found else None


class TestStream(unittest.TestCase):
    def test_descriptions_round_trip(self):
        for market, line in ((SPREAD_HOME, -2.5), (SPREAD_AWAY, 3.5), (TOTAL_OVER, 38.5),
                             (TOTAL_UNDER, 41.0)):
            self.assertEqual(_parse_line(stream.description(market, line)), line)
        self.assertIsNone(_parse_line(stream.description(ML_HOME, None)))

    def _feed(self):
        mc = "AF1"
        plays = [(mc, 6, 1, "Home Team", 1, 10, 35, None), (mc, 10, 1, "Home Team", 1, 10, 26, None),
                 (mc, 16, 1, "Home Team", 2, 11, 26, None), (mc, 20, 1, "Home Team", 1, 10, 37, None)]
        scores = []
        prod = []
        for m in (4, 10, 16, 20):
            for mid, line, p in ((50, None, 42.0), (51, None, 58.0), (52, -2.5, 46.0),
                                 (53, 2.5, 54.0), (54, 38.5, 47.0), (55, 38.5, 53.0)):
                prod.append((mc, mid, m, p, 100 / p, stream.description(mid, line),
                             None if m == 4 else m, "OPEN", "true"))
        return mc, plays, scores, prod

    def test_rows_shaped_like_gameplai(self):
        mc, plays, scores, prod = self._feed()
        rows = stream.quotes_for_matches(model(), [mc], plays, scores, prod)
        self.assertTrue(rows)
        self.assertTrue(all(len(r) == 9 for r in rows))
        by_message = {}
        for r in rows:
            by_message.setdefault(r[6], {})[r[1]] = r
        self.assertEqual(by_message[6][50][8], "false")       # the kick: suspended
        self.assertEqual(by_message[10][50][8], "true")
        self.assertEqual(_parse_line(by_message[10][52][5]), -2.5)   # prod's line
        self.assertAlmostEqual(by_message[10][50][3] + by_message[10][51][3], 100.0, places=1)

    def test_own_lines(self):
        mc, plays, scores, prod = self._feed()
        rows = stream.quotes_for_matches(model(), [mc], plays, scores, prod,
                                         line_mode=stream.OWN)
        spread = [r for r in rows if r[1] == SPREAD_HOME]
        self.assertTrue(all(_parse_line(r[5]) % 1 == 0.5 for r in spread))


class TestBacktest(unittest.TestCase):
    def test_runs_on_a_pairs_file(self):
        fields = ["match_code", "message_count", "period_number", "score_p1", "score_p2",
                  "offensive_team", "field_position", "down_number", "distance",
                  "market_id", "prod_line", "candidate_line", "prod_probability",
                  "candidate_probability", "prod_outcome", "candidate_outcome"]
        rows = []
        for m, period, s1, s2, team, y in ((6, 1, 0, 0, "Home Team", 35),
                                           (60, 2, 7, 0, "Away Team", 25)):
            for mid, line, p, out in ((50, "", 0.45, 1), (52, -2.5, 0.47, 1),
                                      (54, 38.5, 0.5, 0)):
                rows.append(dict(match_code="AF1", message_count=m, period_number=period,
                                 score_p1=s1, score_p2=s2, offensive_team=team,
                                 field_position=y, down_number=1, distance=10,
                                 market_id=mid, prod_line=line, candidate_line=line,
                                 prod_probability=p, candidate_probability=p,
                                 prod_outcome=out, candidate_outcome=out))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.csv")
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            graded, skipped = backtest.run(path, {"v1": version("v1")})
        self.assertEqual(skipped, 0)
        self.assertEqual(len(graded), 6)
        summary = backtest.summarise(graded, ["v1"], n_boot=20)
        self.assertEqual(summary["all"]["pairs"], 6)
        self.assertIn("v1_delta", summary["all"])


class TestVersions(unittest.TestCase):
    def test_known_versions(self):
        self.assertIn("v1", {"v1": version("v1")})
        with self.assertRaises(KeyError):
            version("v99")


if __name__ == "__main__":
    unittest.main()
