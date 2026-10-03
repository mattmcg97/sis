import unittest

from eAMFModel import game as gm


def run(g, *plays):
    for p in plays:
        g, _ = gm.apply(g, p)
    return g


def snap(**kw):
    return gm.Game(**{**dict(phase=gm.SCRIMMAGE, side=gm.HOME, field=25, down=1, distance=10,
                             fresh=False), **kw})


class TestPlays(unittest.TestCase):

    def test_the_opening_kick_goes_to_the_receiver_and_a_touchback_comes_out_at_the_25(self):
        g = gm.Game.start(gm.AWAY)
        self.assertEqual((g.phase, g.side), (gm.KICKOFF, gm.AWAY))
        g, what = gm.apply(g, {"type": "kickoff"})
        self.assertEqual((g.phase, g.side, g.field, g.down, g.distance, g.fresh),
                         (gm.SCRIMMAGE, gm.AWAY, 25, 1, 10, True))
        self.assertEqual(what, "touchback")
        self.assertEqual(g.clock, gm.QUARTER)

    def test_downs_first_downs_and_a_turnover_on_downs(self):
        g = run(snap(), {"type": "play", "yards": 4, "seconds": 20})
        self.assertEqual((g.down, g.distance, g.field, g.clock), (2, 6, 29, 220))
        g = run(g, {"type": "play", "yards": 7})
        self.assertEqual((g.down, g.distance, g.field), (1, 10, 36))
        g = run(g, {"type": "incomplete"}, {"type": "play", "yards": -2}, {"type": "incomplete"})
        self.assertEqual((g.down, g.distance), (4, 12))
        g, what = gm.apply(g, {"type": "play", "yards": 5})
        self.assertIn("turnover on downs", what)
        self.assertEqual((g.side, g.field, g.down, g.fresh), (gm.AWAY, 61, 1, True))   # 100 - 39

    def test_goal_to_go_and_a_touchdown_then_the_conversion_then_the_kick_to_the_other_side(self):
        g = run(snap(field=95, distance=5), {"type": "play", "yards": 3})
        self.assertEqual((g.down, g.distance), (2, 2))
        g = run(g, {"type": "play", "yards": 2})
        self.assertEqual((g.phase, g.side, g.home), (gm.CONVERSION, gm.HOME, 6))
        g = run(g, {"type": "conversion", "kind": "two"})
        self.assertEqual((g.home, g.phase, g.side), (8, gm.KICKOFF, gm.AWAY))

    def test_field_goals_punts_turnovers_and_safeties(self):
        made = run(snap(field=70), {"type": "field_goal", "made": True})
        self.assertEqual((made.home, made.phase, made.side), (3, gm.KICKOFF, gm.AWAY))
        missed = run(snap(field=70), {"type": "field_goal", "made": False})
        self.assertEqual((missed.side, missed.field), (gm.AWAY, 37))              # the spot of the kick
        short = run(snap(field=90), {"type": "field_goal", "made": False})
        self.assertEqual(short.field, 20)
        punt = run(snap(field=30), {"type": "punt", "yards": 45})
        self.assertEqual((punt.side, punt.field), (gm.AWAY, 25))
        deep = run(snap(field=70), {"type": "punt", "yards": 45})
        self.assertEqual(deep.field, 20)                                           # touchback
        pick = run(snap(field=40), {"type": "turnover"})
        self.assertEqual((pick.side, pick.field), (gm.AWAY, 60))
        six = run(snap(field=40), {"type": "turnover", "touchdown": True})
        self.assertEqual((six.away, six.phase, six.side), (6, gm.CONVERSION, gm.AWAY))
        safety = run(snap(field=2), {"type": "play", "yards": -3})
        self.assertEqual((safety.away, safety.phase, safety.side), (2, gm.KICKOFF, gm.AWAY))

    def test_quarters_half_time_and_the_end(self):
        g = run(snap(clock=10), {"type": "play", "yards": 3, "seconds": 15})
        self.assertEqual((g.period, g.clock, g.phase, g.down), (2, gm.QUARTER, gm.SCRIMMAGE, 2))
        g = run(snap(period=2, clock=10, timeouts_home=0, opening_receiver=gm.HOME),
                {"type": "play", "yards": 3, "seconds": 15})
        self.assertEqual((g.period, g.phase, g.side, g.timeouts_home), (3, gm.KICKOFF, gm.AWAY, 3))
        g = run(snap(period=4, clock=5, home=10), {"type": "kneel"})
        self.assertEqual(g.phase, gm.FINAL)
        self.assertIsNone(g.model_state())
        with self.assertRaises(ValueError):
            gm.apply(g, {"type": "play", "yards": 1})

    def test_a_touchdown_as_time_runs_out_still_gets_its_conversion(self):
        g = run(snap(period=2, clock=5, field=90), {"type": "touchdown", "seconds": 8})
        self.assertEqual((g.phase, g.period, g.clock), (gm.CONVERSION, 2, 0))
        g = run(g, {"type": "conversion", "kind": "xp"})
        self.assertEqual((g.home, g.period, g.phase, g.side), (7, 3, gm.KICKOFF, gm.AWAY))

    def test_level_after_four_goes_to_overtime_which_ends_once_the_side_behind_has_had_the_ball(self):
        g = run(snap(period=4, clock=5, home=10, away=10), {"type": "kneel"})
        self.assertEqual((g.period, g.phase, g.timeouts_home), (5, gm.KICKOFF, gm.OT_TIMEOUTS))
        g = run(g, {"type": "kickoff"}, {"type": "field_goal", "made": True})        # home 13-10
        g = run(g, {"type": "kickoff"}, {"type": "play", "yards": 2, "seconds": 300})
        self.assertEqual(g.phase, gm.FINAL)                                         # away had its go
        g = run(snap(period=5, clock=5, home=13, away=10, ot_had_ball=[gm.HOME]),
                {"type": "play", "yards": 2, "seconds": 10})
        self.assertEqual((g.period, g.phase), (6, gm.SCRIMMAGE))                    # away has not

    def test_timeouts_and_setting_a_state(self):
        g = run(snap(), {"type": "timeout", "side": "away"})
        self.assertEqual(g.timeouts_away, 2)
        with self.assertRaises(ValueError):
            run(snap(timeouts_home=0), {"type": "timeout", "side": "home"})
        g = run(snap(), {"type": "set", "state": {"period": 3, "clock": 61, "away": 14}})
        self.assertEqual((g.period, g.clock, g.away, g.field), (3, 61, 14, 25))
        with self.assertRaises(ValueError):
            run(snap(), {"type": "set", "state": {"down": 5}})
        with self.assertRaises(ValueError):
            gm.apply(gm.Game.start(), {"type": "play", "yards": 3})                 # a kick-off first

    def test_the_model_reads_each_phase(self):
        s = snap(side=gm.AWAY, field=80, distance=10, down=3).model_state()
        self.assertEqual((s.offense, s.down, s.field_position, s.distance), (gm.AWAY, 3, 80, 10))
        self.assertEqual(snap(field=95, distance=10).model_state().distance, 5)     # goal to go
        k = gm.Game.start(gm.AWAY).model_state()
        self.assertEqual((k.offense, k.down, k.pending_conversion), (gm.AWAY, None, None))
        c = gm.Game(phase=gm.CONVERSION, side=gm.HOME).model_state()
        self.assertEqual((c.pending_conversion, c.offense), (gm.HOME, gm.AWAY))
        self.assertEqual(snap(timeouts_home=1).model_state().timeouts, (1, 3))

    def test_a_state_goes_to_the_page_and_back(self):
        g = snap(period=2, home=7, ot_had_ball=[])
        self.assertEqual(gm.Game.from_dict(g.to_dict()), g)
        self.assertEqual(g.describe(), "Home 1st & 10 at own 25")
        self.assertEqual(snap(field=92, distance=8).describe(), "Home 1st & goal at opp 8")


if __name__ == "__main__":
    unittest.main()
