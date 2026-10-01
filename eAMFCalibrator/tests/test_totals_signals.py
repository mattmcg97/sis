"""How a match was played, off SCOUTING_FULL rows: the plays, the segment features, and the money."""

import datetime as dt
import unittest

from .. import bet_checks, bets, config, totals_signals as ts

T0 = dt.datetime(2026, 9, 20, 12, 0, 0)


def row(msg, seconds, status=None, kind=None, clock=None, team=None, field=None):
    """A fetch_scouting row: match, message, clock, status, message, team, down, distance, field,
    FILE_TIME."""
    return ("M1", msg, clock, status, kind, team, None, None, field, T0 + dt.timedelta(seconds=seconds))


def snap(msg, t, clock, field, team="TEAM_A", run=6, extra=()):
    """One play: PLAY_STARTED at t, any extra messages, PLAY_OVER `run` seconds later."""
    out = [row(msg, t, kind="PLAY_STARTED", clock=clock, team=team, field=field)]
    out += [row(msg + 1 + i, t + 1, kind=k) for i, k in enumerate(extra)]
    out.append(row(msg + 1 + len(extra), t + run, kind="PLAY_OVER"))
    return out


# Q1: three scrimmage plays 20s of huddle apart burning 30s of clock each, the third a touchdown,
# then the conversion and the kick-off; a timeout between plays; Q2 starts at message 100
ROWS = ([row(1, 0, status="FIRST_QUARTER_STARTED")]
        + snap(10, 10, 240, 25) + snap(20, 36, 210, 35)
        + [row(29, 40, kind="TIMEOUT_CALLED_TEAM_A")]
        + snap(30, 62, 180, 50, extra=("TOUCHDOWN_TEAM_A",))
        + snap(40, 90, 150, 98, extra=("EXTRA_POINT_GOOD_TEAM_A",))
        + snap(50, 110, 150, 35, extra=("KICKOFF_TEAM_A",))
        + [row(100, 200, status="SECOND_QUARTER_STARTED")])


class TestPlays(unittest.TestCase):

    def test_the_play_after_a_touchdown_is_its_conversion_and_then_the_kick_off(self):
        kinds = [p.kind for p in ts.plays(ROWS)]
        self.assertEqual(kinds, ["scrimmage", "scrimmage", "scrimmage", "conversion", "kickoff"])

    def test_the_segment_reads_pace_clock_yards_and_drives(self):
        f = ts.segment_features(ts.plays(ROWS), (1,), ts.timeouts(ROWS))
        self.assertEqual(f["plays"], 3)
        self.assertEqual(f["presnap_s"], 20.0)          # 16->36 and 42->62
        self.assertEqual(f["play_s"], 6.0)
        self.assertEqual(f["clock_per_play"], 30.0)
        self.assertEqual(f["yards_per_play"], 12.5)     # 25->35->50
        self.assertEqual((f["tds"], f["drives"], f["timeouts"]), (1, 1, 1))

    def test_bets_are_read_in_the_next_segment_by_temperature(self):
        def b(seconds, sel, temp, stake=10.0, revenue=10.0):
            return bets.Bet("x", "M1", T0 + dt.timedelta(seconds=seconds), 3, sel, 1.9, stake, revenue,
                            extra={config.BET_VIP_COLUMN: temp, config.BET_IN_PLAY_COLUMN: "Yes"})
        m = ts.money([b(150, 2, "VIP"), b(250, 2, "restricted", revenue=-9.0), b(250, 1, None),
                      b(400, 2, "Standard")], T0 + dt.timedelta(seconds=200), T0 + dt.timedelta(seconds=300))
        self.assertEqual(m["All"][:3], [2, 20.0, 10.0])
        self.assertEqual(m["Restricted"][:4], [1, 10.0, 10.0, -9.0])
        self.assertEqual(m["none"][:3], [1, 10.0, 0.0])
        self.assertEqual(m["VIP"][0], 0)

    def test_a_count_is_binned_by_value_and_a_measure_by_quintile(self):
        self.assertEqual(ts.edges_for([0, 1, 1, 2, 0, 3]), [1, 2, 3])
        self.assertEqual(len(ts.edges_for([x / 10 for x in range(100)])), 4)


class TestMatch(unittest.TestCase):

    def test_over_is_the_final_less_prods_line_at_the_next_quarters_start(self):
        prod = [("M1", 54, T0 + dt.timedelta(seconds=s), 50.0, None, f"Line {line}", msg, "open", "true")
                for s, line, msg in ((0, 44.5, 1), (195, 40.5, 99))]
        tl = bet_checks.message_times(prod)
        from .. import bet_moments
        tls = bet_moments.build(ROWS, tl)
        scores = bet_checks.Checks({}, {}, bet_checks.score_index([("M1", 36, 1, 7, 0, 7, 0)]))
        r = ts.match_row("M1", ROWS, tls["M1"], bets.timeline(prod), scores, (24, 21), [])
        self.assertEqual((r["q1_line"], r["q1_points"], r["q1_needed"]), (40.5, 7, 33.5))
        self.assertEqual(r["q1_over"], 45 - 40.5)
        lines = ts.report([r] * 30)
        self.assertTrue(any("end of Q1" in x for x in lines))


class TestForm(unittest.TestCase):

    def test_form_reads_only_the_matches_before(self):
        g = lambda m, t, a, b, tot: dict(MATCH_CODE=m, SCHEDULED_START_TIME_UTC=t, PLAYER_1_HANDLE=a,
                                        PLAYER_2_HANDLE=b, PLAYER_1_FINAL_SCORE=str(tot), PLAYER_2_FINAL_SCORE="0")
        h = [g("A", "2026-09-01 10:00:00", "ann", "bob", 40), g("B", "2026-09-01 11:00:00", "ann", "cat", 20),
             g("C", "2026-09-01 12:00:00", "ann", "bob", 30)]
        form = ts.form_by_match(h, weight=0.5, shrink=0)
        self.assertEqual(form["A"], (0.0, 0.0))                 # nothing before it
        # B: league mean 40 after A, so A's residual was 0; C: B's residual 20 - 40 = -20,
        # ann's half -10 at weight 0.5 -> -5, bob unchanged at 0
        self.assertEqual(form["C"][0], -5.0)


if __name__ == "__main__":
    unittest.main()
