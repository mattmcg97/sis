"""A pre-match model whose form follows results prices each pre-match quote off the results known
when it was published (stream.known_states), and glmer reads every final known when pricing."""

import csv
import datetime as dt
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from .. import glmer_prior, stream, v8_stream, v9_stream, v10_stream

T0 = dt.datetime(2026, 9, 1, 12, 0)


def at(minutes):
    return T0 + dt.timedelta(minutes=minutes)


def match(code, minutes, p1, p2, finals=("", "")):
    return {"MATCH_CODE": code, "SPORT_CODE": "AF", "STREAM_NUMBER": "1",
            "SCHEDULED_START_TIME_UTC": at(minutes).strftime("%Y-%m-%d %H:%M:%S"),
            "PLAYER_1_HANDLE": p1, "PLAYER_1_TEAM": "T1", "PLAYER_2_HANDLE": p2,
            "PLAYER_2_TEAM": "T2", "PLAYER_1_FINAL_SCORE": finals[0],
            "PLAYER_2_FINAL_SCORE": finals[1]}


class TestKnownStates(unittest.TestCase):

    def setUp(self):
        # X: ann v bob at 12:00. Ann played at 10:48 and 11:24, bob at 11:00.
        self.x = match("X", 0, "ann", "bob")
        self.history = [match("A1", -72, "ANN", "cat", (20, 14)),
                        match("A2", -36, "ann", "dan", (17, 21)),
                        match("B1", -60, "bob", "eve", (24, 10)),
                        match("Z", -40, "fay", "gus", (3, 3))]        # neither gamer: never cuts

    def test_each_publish_time_is_cut_at_the_earliest_unfinished_match(self):
        published = [at(-80), at(-50), at(-30), at(-10), at(-1)]
        rows, code_at = stream.known_states([self.x], {"X": published}, self.history)
        # results come in 36 minutes after kick-off: A1 at 11:24, B1 at 11:36, A2 at 12:00
        self.assertEqual(code_at[("X", at(-80))], "X@1")               # A1 not yet started
        self.assertEqual(code_at[("X", at(-50))], "X@1")               # A1 still on
        self.assertEqual(code_at[("X", at(-30))], "X@2")               # A1 in, B1 still on
        self.assertEqual(code_at[("X", at(-10))], "X@3")               # only A2 still on
        self.assertEqual(code_at[("X", at(-1))], "X@3")
        cuts = {r["MATCH_CODE"]: r for r in rows}
        self.assertEqual(cuts["X"], self.x)                              # its own row comes first
        self.assertEqual(rows[0], self.x)
        self.assertEqual(cuts["X@1"]["SCHEDULED_START_TIME_UTC"], "2026-09-01 10:47:59")
        self.assertEqual(cuts["X@2"]["SCHEDULED_START_TIME_UTC"], "2026-09-01 10:59:59")
        self.assertEqual(cuts["X@3"]["SCHEDULED_START_TIME_UTC"], "2026-09-01 11:23:59")
        for code in ("X@1", "X@2", "X@3"):
            self.assertEqual((cuts[code]["PLAYER_1_FINAL_SCORE"], cuts[code]["PLAYER_2_FINAL_SCORE"]),
                             ("", ""))
            self.assertEqual(cuts[code]["PLAYER_1_HANDLE"], "ann")

    def test_the_feed_s_end_counts_when_it_is_believable(self):
        ends = {"A1": at(-71), "A2": at(-8), "B1": at(240)}
        rows, code_at = stream.known_states([self.x], {"X": [at(-60), at(-10), at(-5)]},
                                            self.history, ends=ends)
        cut = {r["MATCH_CODE"]: r["SCHEDULED_START_TIME_UTC"] for r in rows}
        # A1's feed end (10:49) is read: at 11:00 B1 is the earliest still on
        self.assertEqual(cut[code_at[("X", at(-60))]], "2026-09-01 10:59:59")
        # B1's (five hours on) is not: it ends 36 minutes after kick-off, so at 11:50 only A2 is on
        self.assertEqual(cut[code_at[("X", at(-10))]], "2026-09-01 11:23:59")
        self.assertEqual(code_at[("X", at(-5))], "X")                  # A2 over at 11:52

    def test_nothing_unfinished_prices_off_the_match_s_own_row(self):
        rows, code_at = stream.known_states([self.x], {"X": [at(-1)]}, self.history,
                                            ends={"A2": at(-5)})
        self.assertEqual(rows, [self.x])
        self.assertEqual(code_at, {("X", at(-1)): "X"})
        rows, code_at = stream.known_states([self.x], {}, self.history)
        self.assertEqual((rows, code_at), ([self.x], {}))

    def test_times_are_read_as_naive_utc(self):
        self.assertEqual(stream.when("2026-09-01T12:00:00.000+01:00"), at(-60))
        self.assertEqual(stream.when("2026-09-01 12:00:00.123 junk"), T0)
        self.assertEqual(stream.when(T0.replace(tzinfo=dt.timezone.utc)), T0)
        self.assertIsNone(stream.when(""))
        self.assertIsNone(stream.when("not a time"))
        ends = stream.match_ends({"X": [{"file_time": "2026-09-01 12:30:00"},
                                        {"file_time": at(28)}, {"file_time": None}], "Y": []})
        self.assertEqual(ends, {"X": at(30)})

    def test_prematch_publish_times_are_before_the_first_play(self):
        rows = [("X", 54, at(-5), 50.0, 2.0, "", None, "OPEN", "true"),
                ("X", 54, at(-4), None, None, "", None, "OPEN", "true"),     # no price
                ("X", 54, at(-3), 50.0, 2.0, "", 4, "OPEN", "true"),
                ("X", 54, at(1), 50.0, 2.0, "", 9, "OPEN", "true"),
                ("X", 54, None, 50.0, 2.0, "", None, "OPEN", "true")]
        self.assertEqual(stream.prematch_publish_times(rows, 5), {at(-5), at(-3)})
        self.assertEqual(stream.prematch_publish_times(rows, None), {at(-5)})


class TestKickoffPerQuote(unittest.TestCase):

    def test_each_prematch_quote_takes_the_kickoff_book_of_its_publish_time(self):
        def book(total):
            margin = np.zeros(201)
            margin[100] = 1.0
            tp = np.zeros(161)
            tp[total] = 1.0
            return margin, tp
        early, late = book(20), book(50)
        prod = [("M", 54, at(-30), 50.0, 2.0, "Total points over 30.5", None, "OPEN", "true"),
                ("M", 54, at(-5), 50.0, 2.0, "Total points over 30.5", None, "OPEN", "true"),
                ("M", 54, at(-2), 50.0, 2.0, "Total points over 30.5", 3, "OPEN", "true")]
        kickoff = lambda row: early if row[2] < at(-10) else late
        for module in (v8_stream, v9_stream, v10_stream):
            rows = module.quote_rows("M", [], prod, first_play_message=5,
                                     lines=module.PROD_LINES, kickoff=kickoff)
            by = {(r[2], r[6]): r[3] for r in rows}
            self.assertEqual(by, {(at(-30), None): 0.01, (at(-5), None): 99.99, (at(-2), 3): 99.99},
                             module.__name__)
            # a single book still prices them all alike
            same = module.quote_rows("M", [], prod, 5, module.PROD_LINES, kickoff=early)
            self.assertEqual({r[3] for r in same}, {0.01})


class TestGlmerReadsTheFinals(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name
        with open(os.path.join(self.dir, glmer_prior.HISTORY), "w", newline="") as fh:
            w = csv.DictWriter(fh, list(match("A", 0, "a", "b")))
            w.writeheader()
            w.writerow(match("OLD", -999, "ann", "cat", (21, 7)))
        self.written = {}

    def tearDown(self):
        self.tmp.cleanup()

    def _fake_run(self, script, args):
        opts = dict(a[2:].split("=", 1) for a in args)
        for key in ("schedule", "history"):
            with open(opts[key], newline="") as fh:
                self.written[key] = {r["MATCH_CODE"]: r for r in csv.DictReader(fh)}
        with open(opts["out"], "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["MATCH_CODE", "Prediction_Status", "Pred_P1_Points", "Pred_P2_Points"])
            for code in self.written["schedule"]:
                w.writerow([code, "OK", 20, 18])

    def test_a_scheduled_match_carries_its_known_final_and_a_cut_none(self):
        schedule = [match("X", 0, "ann", "bob"), match("X@1", -40, "ann", "bob"),
                    match("Y", 36, "ann", "cat")]
        results = [match("X", 0, "ann", "bob", (24, 17)), match("W", -36, "bob", "dan", (10, ""))]
        with mock.patch.object(glmer_prior, "_run", side_effect=self._fake_run):
            got = glmer_prior.predict(self.dir, schedule, results)
        self.assertEqual(set(got), {"X", "X@1", "Y"})
        sched = self.written["schedule"]
        self.assertEqual((sched["X"]["PLAYER_1_FINAL_SCORE"], sched["X"]["PLAYER_2_FINAL_SCORE"]),
                         ("24", "17"))
        self.assertEqual(sched["X@1"]["PLAYER_1_FINAL_SCORE"], "")
        self.assertEqual(sched["Y"]["PLAYER_1_FINAL_SCORE"], "")
        # the history read: the model's own and the settled results (W has half a final: left out)
        self.assertEqual(set(self.written["history"]), {"OLD", "X"})
        with mock.patch.object(glmer_prior, "_run", side_effect=self._fake_run):
            glmer_prior.predict(self.dir, schedule)
        self.assertEqual(set(self.written["history"]), {"OLD"})
        self.assertEqual(self.written["schedule"]["X"]["PLAYER_1_FINAL_SCORE"], "")

    def test_a_model_that_reads_the_clock_is_not_priced_as_of_earlier(self):
        def prematch(model):
            with open(os.path.join(self.dir, glmer_prior.META), "w") as fh:
                json.dump({"scale": 1.0, "league_average": [17, 17], "model": model}, fh)
            return glmer_prior.Prematch(self.dir)
        self.assertTrue(prematch({"feature_set": "form"}).AS_OF)
        self.assertFalse(prematch({"feature_set": "form_session"}).AS_OF)
        self.assertFalse(prematch({"feature_set": "x", "formula": "Score ~ RestLog + (1|Player)"}).AS_OF)
        # form_clock's late-in-session and time-of-day terms are read near enough at an earlier time
        clock = "Score ~ FormFor + SessLate + OppSessLate + (1 | HourBlock) + (1 | Player)"
        self.assertTrue(prematch({"feature_set": "form_clock", "formula": clock}).AS_OF)
        self.assertFalse(prematch({"feature_set": "x", "formula": clock + " + SessFormFor"}).AS_OF)
        self.assertTrue(prematch({"feature_set": "context",
                                  "formula": "Score ~ FormFor + ExpLog + (1|Player)"}).AS_OF)
        self.assertTrue(glmer_prior.Prematch.FOLLOWS_RESULTS)


if __name__ == "__main__":
    unittest.main()
