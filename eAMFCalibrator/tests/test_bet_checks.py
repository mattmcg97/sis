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


# the first quarter starts at 1; plays start at 2, 9, 15, 20, 27 and are over at 5, 11, 17, 22, 28;
# a score at 7; a neutral BET_SUSPEND at 13; two minutes left at 28, match over at 30;
# SCOUTING_FULL lacks messages 23-24
SCOUTING = [srow(1, "FIRST_QUARTER_STARTED"), srow(2, kind="PLAY_STARTED", clock=240),
            srow(5, kind="PLAY_OVER", clock=230), srow(9, kind="PLAY_STARTED"),
            srow(11, kind="PLAY_OVER"), srow(13, "BET_SUSPEND"), srow(15, kind="PLAY_STARTED"),
            srow(17, kind="PLAY_OVER"), srow(20, kind="PLAY_STARTED"), srow(22, kind="PLAY_OVER"),
            srow(26, "FOURTH_QUARTER_STARTED", clock=240), srow(27, kind="PLAY_STARTED", clock=130),
            srow(28, kind="PLAY_OVER", clock=110), srow(29, "PERMANENT_BET_SUSPEND"),
            srow(30, "MATCH_OVER")]
SCOUTING += [srow(m) for m in (3, 4, 6, 7, 8, 10, 12, 14, 16, 18, 19, 21, 25, 31, 32)]
# prod publishes message m at 10 * m seconds
PROD = [("M1", 50, at(10 * m), 50.0, None, None, m, "open", "true") for m in range(1, 33)]
SCORES = [("M1", 1, 1, 0, 0, 0, 0), ("M1", 7, 1, 7, 0, 7, 0)]
# the model cannot read the PLAY_OVER at 17
BOOKS = {"M1": [5, 11, 22, 28]}


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
        self.assertEqual(f.play_overs, [5, 11, 17, 22, 28])
        self.assertEqual(f.late, {"PERMANENT_BET_SUSPEND": 29, "MATCH_OVER": 30})
        self.assertNotIn(13, f.moves)


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
        ok = c.state("M1", 14)
        self.assertEqual((ok["feed_from"], ok["feed_to"], ok["state_ok"], ok["snapshot_age_seconds"]),
                         (11, 14, True, 30.0))
        reasons = {m: c.state("M1", m)["state_reason"] for m in (8, 10, 18, 25, 3)}
        self.assertEqual(reasons, {8: "score changed", 10: "feed moved on: PLAY_STARTED",
                                   18: "the latest PLAY_OVER cannot be read",
                                   25: "SCOUTING_FULL missing prod's messages",
                                   3: "before the first PLAY_OVER"})
        self.assertEqual(checks(guessed={"M1"}).state("M1", 14)["state_reason"],
                         "TEAM_A's side not known")

    def test_a_candidate_is_held_to_the_feed_between_its_quote_and_prods(self):
        c = checks()
        self.assertTrue(c.candidate("M1", 11, 14)["candidate_ok"])
        self.assertEqual(c.candidate("M1", 5, 10)["candidate_reason"], "feed moved on: PLAY_STARTED")
        self.assertEqual(c.candidate("M1", 15, 14)["candidate_reason"], "candidate's quote after prod's")
        self.assertIsNone(c.candidate("M1", None, 14)["candidate_ok"])

    def test_the_allowed_gap_is_configurable(self):
        saved = config.MAX_SCOUTING_GAP
        try:
            config.MAX_SCOUTING_GAP = 2
            self.assertTrue(checks().state("M1", 25)["state_ok"])
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
        rows = self.join([bet(140), bet(80), bet(100), bet(180), bet(250), bet(400), bet(305),
                          bet(30), bet(140, BET_CASHED_OUT="Yes")])
        self.assertEqual([bets.why_not(r) for r in rows],
                         ["simulated", "model: score changed", "model: feed moved on: PLAY_STARTED",
                          "model: the latest PLAY_OVER cannot be read",
                          "model: SCOUTING_FULL missing prod's messages", "placed after match over",
                          "placed after two minutes", "model: before the first PLAY_OVER",
                          "cashed out"])
        self.assertEqual((rows[0]["feed_from"], rows[0]["feed_to"], rows[0]["candidate_message"]),
                         (11, 14, 14))
        table = self.join([bet(80), bet(250)], same_state=False)
        self.assertEqual([bets.why_not(r) for r in table], ["simulated", "simulated"])

    def test_a_candidate_quote_from_before_the_feed_moved_on_is_not_used(self):
        cand = [r for r in PROD if r[6] in (5, 11)]
        prod = bets.timeline(PROD)
        rows = bets.join([bet(100), bet(140)], {}, {}, prod, bets.quote_index(cand),
                         bets.timeline(cand), {"M1": (21, 17)}, checks())
        self.assertEqual([bets.why_not(r) for r in rows],
                         ["candidate: feed moved on: PLAY_STARTED", "simulated"])

    def test_the_report_names_what_it_found(self):
        c = checks()
        c.reasons.update({"priced": 5, "no_clock": 1})
        rows = self.join([bet(5, in_play=False), bet(140), bet(80), bet(400), bet(305),
                          bet(140, BET_CASHED_OUT="Yes")])
        text = "\n".join(bet_checks.report(rows, c))
        self.assertIn("after match over (left out)", text)
        self.assertIn("match-over message: MATCH_OVER 1", text)
        self.assertIn("PERMANENT_BET_SUSPEND", text)
        self.assertIn("same information 2 (66.7%)", text)
        self.assertIn("score changed", text)
        self.assertIn("priced 5 (83.3%); not: no_clock 1", text)
        self.assertIn("M1", text)


class TestCommand(unittest.TestCase):

    def test_the_check_runs_without_pricing_any_candidate(self):
        from .. import snowflake_io
        cols = ["OPERATOR_UNIQUE_ID", "MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID",
                "ODDS", "STAKE_GBP", "REVENUE_GBP", "OPERATOR_NAME", "BET_IN_PLAY", "BET_CASHED_OUT"]
        raw = [(1, "M1", at(140), 1, 1, 1.9, 10, 10, "FANDUEL", "Yes", "No"),
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
