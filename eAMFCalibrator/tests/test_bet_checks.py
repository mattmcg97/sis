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


# the first quarter starts at 1, plays over at 5, 9, 12, 17, 24 and 28, two minutes left at 28,
# match over at 30; SCOUTING_FULL lacks messages 19-21
SCOUTING = [srow(1, "FIRST_QUARTER_STARTED"), srow(2, kind="PLAY_STARTED", clock=240),
            srow(5, kind="PLAY_OVER", clock=230), srow(6, kind="PLAY_STARTED"),
            srow(9, kind="PLAY_OVER"), srow(10, kind="PLAY_STARTED"), srow(12, kind="PLAY_OVER"),
            srow(13, kind="PLAY_STARTED"), srow(17, kind="PLAY_OVER"), srow(18, kind="PLAY_STARTED"),
            srow(24, kind="PLAY_OVER"), srow(25, "FOURTH_QUARTER_STARTED", clock=240),
            srow(26, kind="PLAY_STARTED", clock=130), srow(28, kind="PLAY_OVER", clock=110),
            srow(29, "PERMANENT_BET_SUSPEND"), srow(30, "MATCH_OVER")]
SCOUTING += [srow(m) for m in (3, 4, 7, 8, 11, 14, 15, 16, 22, 23, 27, 31, 32)]
# prod publishes message m at 10 * m seconds
PROD = [("M1", 50, at(10 * m), 50.0, None, None, m, "open", "true") for m in range(1, 33)]
SCORES = [("M1", 1, 1, 0, 0, 0, 0), ("M1", 7, 1, 7, 0, 7, 0)]
# the model cannot read the PLAY_OVER at 12
BOOKS = {"M1": [5, 9, 17, 24, 28]}


def checks(books=BOOKS, guessed=()):
    return bet_checks.Checks(bet_checks.build_feeds(SCOUTING), bet_checks.message_times(PROD),
                             bet_checks.score_index(SCORES), books, guessed)


def bet(seconds, in_play=True, **extra):
    return bets.Bet(bet_id=seconds, match_code="M1", time=at(seconds), market_type=1, selection=1,
                    odds=1.9, stake=10, revenue=10,
                    extra=dict({config.BET_GROUP_COLUMN: "FANDUEL",
                                config.BET_IN_PLAY_COLUMN: "Yes" if in_play else "No"}, **extra))


class TestFeed(unittest.TestCase):

    def test_the_start_two_minutes_and_match_over_are_read_off_the_feed(self):
        f = bet_checks.build_feeds(SCOUTING)["M1"]
        self.assertEqual((f.start, f.two_minutes, f.over, f.over_status), (1, 28, 30, "MATCH_OVER"))
        self.assertEqual(f.play_overs, [5, 9, 12, 17, 24, 28])
        self.assertEqual(f.late, {"PERMANENT_BET_SUSPEND": 29, "MATCH_OVER": 30})


class TestPhase(unittest.TestCase):

    def test_each_bet_is_placed_against_the_match(self):
        c = checks()
        cases = [(bet(5, in_play=False), bet_checks.PRE_MATCH, None),
                 (bet(100, in_play=False), bet_checks.PRE_AFTER_START, 90.0),
                 (bet(-5), bet_checks.BEFORE_START, 15.0),
                 (bet(0), bet_checks.IN_PLAY, None),
                 (bet(285), bet_checks.IN_PLAY, None),
                 (bet(305), bet_checks.AFTER_TWO, 25.0),
                 (bet(400), bet_checks.AFTER_OVER, 100.0)]
        for b, phase, secs in cases:
            self.assertEqual(c.phase(b), (phase, secs), b.time)

    def test_a_match_without_scouting_is_not_placed(self):
        b = bet(100)
        b.match_code = "M2"
        self.assertEqual(checks().phase(b)[0], bet_checks.NO_FEED)


class TestState(unittest.TestCase):

    def test_the_model_matches_prod_only_with_the_same_information(self):
        c = checks()
        ok = c.state("M1", 11)
        self.assertEqual((ok["snapshot_message"], ok["skipped_play_overs"], ok["score_changed"],
                          ok["state_ok"], ok["snapshot_age_seconds"]), (9, 0, False, True, 20.0))
        scored = c.state("M1", 8)
        self.assertEqual((scored["snapshot_message"], scored["score_changed"], scored["state_ok"]),
                         (5, True, False))
        older = c.state("M1", 14)
        self.assertEqual((older["snapshot_message"], older["skipped_play_overs"], older["state_ok"]),
                         (9, 1, False))
        gap = c.state("M1", 23)
        self.assertEqual((gap["snapshot_message"], gap["scouting_missing"], gap["state_ok"]),
                         (17, 3, False))
        self.assertIsNone(c.state("M1", 3)["snapshot_message"])
        self.assertFalse(checks(guessed={"M1"}).state("M1", 11)["state_ok"])

    def test_the_allowed_gap_is_configurable(self):
        saved = config.MAX_SCOUTING_GAP
        try:
            config.MAX_SCOUTING_GAP = 3
            self.assertTrue(checks().state("M1", 23)["state_ok"])
        finally:
            config.MAX_SCOUTING_GAP = saved

    def test_the_models_books_are_the_play_overs_it_can_read(self):
        snap = dict(team_a_side="home", period="1", clock_seconds="200", play_kind="PASS",
                    score_p1="0", score_p2="0", offense="TEAM_A", down="1", distance="10",
                    field_position="25", play_messages="", opening_offense="TEAM_A")
        snaps = {"M1": [dict(snap, message=5), dict(snap, message=9, period=""),
                        dict(snap, message=12)],
                 "M2": [dict(snap, message=5, team_a_side=None)]}
        books, guessed, reasons = bet_checks.model_books(snaps)
        self.assertEqual(books, {"M1": [5, 12], "M2": [5]})
        self.assertEqual(guessed, {"M2"})
        self.assertEqual(reasons["no_clock"], 1)


class TestJoin(unittest.TestCase):

    def join(self, bs, same_state=True):
        prod = bets.timeline(PROD)
        return bets.join(bs, {}, {}, prod, bets.quote_index(PROD), prod, {"M1": (21, 17)},
                         checks(), same_state=same_state)

    def test_a_model_is_read_only_at_prods_information_and_stragglers_are_left_out(self):
        rows = self.join([bet(110), bet(80), bet(140), bet(230), bet(400), bet(305), bet(30),
                          bet(110, BET_CASHED_OUT="Yes")])
        self.assertEqual([bets.why_not(r) for r in rows],
                         ["simulated", "score changed since the model's PLAY_OVER",
                          "model on an older PLAY_OVER (the latest unpriceable)",
                          "SCOUTING_FULL missing prod's messages", "placed after match over",
                          "placed after two minutes", "before the model's first PLAY_OVER",
                          "cashed out"])
        table = self.join([bet(80), bet(230)], same_state=False)
        self.assertEqual([bets.why_not(r) for r in table], ["simulated", "simulated"])

    def test_the_report_names_what_it_found(self):
        c = checks()
        c.reasons.update({"priced": 5, "no_clock": 1})
        rows = self.join([bet(5, in_play=False), bet(110), bet(80), bet(400), bet(305),
                          bet(110, BET_CASHED_OUT="Yes")])
        text = "\n".join(bet_checks.report(rows, c))
        self.assertIn("after match over (left out)", text)
        self.assertIn("match-over message: MATCH_OVER 1", text)
        self.assertIn("PERMANENT_BET_SUSPEND", text)
        self.assertIn("score changed since the model's PLAY_OVER 1", text)
        self.assertIn("priced 5 (83.3%); not: no_clock 1", text)
        self.assertIn("M1", text)


class TestCommand(unittest.TestCase):

    def test_the_check_runs_without_pricing_any_candidate(self):
        from .. import snowflake_io
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "OPERATOR_NAME", "BET_IN_PLAY", "BET_CASHED_OUT"]
        raw = [(1, "M1", at(110), 1, 1, 1.9, 10, 10, "FANDUEL", "Yes", "No"),
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

    def test_the_checks_are_fetched_with_the_models_snapshots(self):
        from .. import snowflake_io
        snap = dict(team_a_side="home", period="1", clock_seconds="200", play_kind="PASS",
                    score_p1="0", score_p2="0", offense="TEAM_A", down="1", distance="10",
                    field_position="25", play_messages="", opening_offense="TEAM_A", message=5)

        def fake(cur, matches, prod_by_match=None, keep=None):
            keep["scouting"], keep["scores"] = SCOUTING, SCORES
            self.assertEqual(len(prod_by_match["M1"]), len(PROD))
            return {"M1": [snap]}, []
        with mock.patch.object(snowflake_io, "_play_over_snapshots", side_effect=fake):
            c = bets.fetch_checks(None, ["M1"], PROD)
        self.assertEqual((c.books, c.feeds["M1"].over), ({"M1": [5]}, 30))


if __name__ == "__main__":
    unittest.main()
