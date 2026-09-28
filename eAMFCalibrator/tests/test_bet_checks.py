"""The checks: when in the match each bet was placed, and a model's state against prod's."""

import datetime as dt
import unittest
from unittest import mock

from .. import bet_checks, bets, config

T0 = dt.datetime(2026, 9, 27, 12, 0, 0)


def at(seconds):
    return T0 + dt.timedelta(seconds=seconds)


def srow(msg, status=None, kind=None, clock=None, match="M1"):
    """A scouting.fetch_scouting row."""
    return (match, msg, clock, status, kind, None, None, None, None, None)


# kickoff at 2, plays over at 5, 9, 14 and 25, two minutes left at 14, match over at 30;
# SCOUTING_FULL lacks messages 21-23
SCOUTING = [srow(1, "FIRST_QUARTER_STARTED"), srow(2, kind="PLAY_STARTED", clock=240),
            srow(5, kind="PLAY_OVER", clock=230), srow(6, kind="PLAY_STARTED", clock=225),
            srow(7), srow(9, kind="PLAY_OVER", clock=200),
            srow(10, "FOURTH_QUARTER_STARTED", clock=240), srow(11, kind="PLAY_STARTED", clock=130),
            srow(12), srow(13), srow(14, kind="PLAY_OVER", clock=110), srow(20, kind="PLAY_STARTED"),
            srow(24), srow(25, kind="PLAY_OVER"), srow(30, "MATCH_OVER")]
SCOUTING += [srow(m) for m in (3, 4, 8, 15, 16, 17, 18, 19, 26, 27, 28, 29)]
# prod publishes message m at 10 * m seconds
PROD = [("M1", 50, at(10 * m), 50.0, None, None, m, "open", "true") for m in range(1, 33)]
SCORES = [("M1", 1, 1, 0, 0, 0, 0), ("M1", 7, 1, 7, 0, 7, 0)]


def checks():
    return bet_checks.Checks(bet_checks.build_feeds(SCOUTING), bet_checks.message_times(PROD),
                             bet_checks.score_index(SCORES))


def bet(seconds, in_play=True, **extra):
    return bets.Bet(bet_id=seconds, match_code="M1", time=at(seconds), market_type=1, selection=1,
                    odds=1.9, stake=10, revenue=10,
                    extra=dict({config.BET_GROUP_COLUMN: "FANDUEL",
                                config.BET_IN_PLAY_COLUMN: "Yes" if in_play else "No"}, **extra))


class TestFeed(unittest.TestCase):

    def test_kickoff_two_minutes_and_match_over_are_read_off_the_feed(self):
        f = bet_checks.build_feeds(SCOUTING)["M1"]
        self.assertEqual((f.kickoff, f.two_minutes, f.over, f.over_status), (2, 14, 30, "MATCH_OVER"))
        self.assertEqual(f.play_overs, [5, 9, 14, 25])


class TestPhase(unittest.TestCase):

    def test_each_bet_is_placed_against_the_match(self):
        c = checks()
        cases = [(bet(5, in_play=False), bet_checks.PRE_MATCH),
                 (bet(100, in_play=False), bet_checks.PRE_AFTER_KICKOFF),
                 (bet(0), bet_checks.BEFORE_KICKOFF),
                 (bet(100), bet_checks.IN_PLAY),
                 (bet(150), bet_checks.AFTER_TWO),
                 (bet(305), bet_checks.AFTER_TWO),
                 (bet(400), bet_checks.AFTER_OVER)]
        for b, phase in cases:
            self.assertEqual(c.phase(b)[0], phase, b.time)
        self.assertEqual(c.phase(bet(150))[1], 10.0)
        self.assertEqual(c.phase(bet(400))[1], 100.0)
        self.assertEqual(c.phase(bet(0))[1], 20.0)

    def test_a_match_without_scouting_is_not_placed(self):
        b = bet(100)
        b.match_code = "M2"
        self.assertEqual(checks().phase(b)[0], bet_checks.NO_FEED)


class TestState(unittest.TestCase):

    def test_the_model_matches_prod_only_with_the_same_information(self):
        c = checks()
        ok = c.state("M1", 12)
        self.assertEqual((ok["snapshot_message"], ok["scouting_missing"], ok["score_changed"],
                          ok["state_ok"], ok["snapshot_age_seconds"]), (9, 0, False, True, 30.0))
        scored = c.state("M1", 8)
        self.assertEqual((scored["snapshot_message"], scored["score_changed"], scored["state_ok"]),
                         (5, True, False))
        gap = c.state("M1", 24)
        self.assertEqual((gap["snapshot_message"], gap["scouting_missing"], gap["state_ok"]),
                         (14, 3, False))
        self.assertIsNone(c.state("M1", 3)["snapshot_message"])

    def test_the_allowed_gap_is_configurable(self):
        saved = config.MAX_SCOUTING_GAP
        try:
            config.MAX_SCOUTING_GAP = 3
            self.assertTrue(checks().state("M1", 24)["state_ok"])
        finally:
            config.MAX_SCOUTING_GAP = saved


class TestJoin(unittest.TestCase):

    def join(self, bs, same_state=True):
        prod = bets.timeline(PROD)
        return bets.join(bs, {}, {}, prod, bets.quote_index(PROD), prod, {"M1": (21, 17)},
                         checks(), same_state=same_state)

    def test_a_model_is_read_only_at_prods_information_and_stragglers_are_left_out(self):
        rows = self.join([bet(120), bet(80), bet(240), bet(400), bet(30),
                          bet(120, BET_CASHED_OUT="Yes")])
        self.assertEqual([bets.why_not(r) for r in rows],
                         ["simulated", "score changed since the model's PLAY_OVER",
                          "SCOUTING_FULL missing prod's messages", "placed after match over",
                          "before the model's first PLAY_OVER", "cashed out"])
        table = self.join([bet(80), bet(240)], same_state=False)
        self.assertEqual([bets.why_not(r) for r in table], ["simulated", "simulated"])

    def test_the_report_names_what_it_found(self):
        rows = self.join([bet(5, in_play=False), bet(120), bet(80), bet(400),
                          bet(120, BET_CASHED_OUT="Yes")])
        text = "\n".join(bet_checks.report(rows, checks()))
        self.assertIn("after match over (left out)", text)
        self.assertIn("match-over message: MATCH_OVER 1", text)
        self.assertIn("score changed since the PLAY_OVER 1", text)
        self.assertIn("M1", text)


class TestCommand(unittest.TestCase):

    def test_the_check_runs_without_pricing_any_candidate(self):
        from .. import snowflake_io
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "OPERATOR_NAME", "BET_IN_PLAY", "BET_CASHED_OUT"]
        raw = [(1, "M1", at(120), 1, 1, 1.9, 10, 10, "FANDUEL", "Yes", "No"),
               (2, "M1", at(400), 1, 1, 1.9, 10, 10, "FANDUEL", "Yes", "No"),
               (3, "M1", at(130), 1, 1, 1.9, 10, 10, "FANDUEL", "Yes", "Yes")]
        with mock.patch.object(bets, "fetch_all", return_value=(cols, raw)), \
                mock.patch.object(snowflake_io, "fetch_quotes", return_value=PROD) as fq, \
                mock.patch.object(snowflake_io, "fetch_final_scores", return_value={"M1": (21, 17)}), \
                mock.patch.object(bets, "fetch_checks", return_value=checks()), \
                mock.patch.object(bets, "write_csv"), mock.patch("builtins.print"), \
                mock.patch("os.makedirs"), mock.patch.object(bets, "check_models") as models:
            rows = bets.run(None, "out", only_checks=True)
        models.assert_not_called()
        self.assertEqual(fq.call_count, 1)
        self.assertEqual([r["match_phase"] for r in rows],
                         [bet_checks.IN_PLAY, bet_checks.AFTER_OVER, bet_checks.IN_PLAY])
        self.assertEqual(rows[2]["result"], bets.CASHED)


if __name__ == "__main__":
    unittest.main()
