"""The second layer on the pre-match model: each gamer's offset from their results since its fit."""

import datetime as dt
import json
import os
import tempfile
import unittest
from unittest import mock

from .. import follow, nb2_prior, v10, v11, v12

CUT = dt.datetime(2026, 9, 1)


def match(code, minutes, p1, p2, finals=None):
    t = CUT + dt.timedelta(minutes=minutes)
    row = {"MATCH_CODE": code, "SCHEDULED_START_TIME_UTC": f"{t:%Y-%m-%d %H:%M:%S}",
           "PLAYER_1_HANDLE": p1, "PLAYER_2_HANDLE": p2, "PLAYER_1_TEAM": "A", "PLAYER_2_TEAM": "B",
           "STREAM_NUMBER": "1", "PLAYER_1_FINAL_SCORE": "", "PLAYER_2_FINAL_SCORE": ""}
    if finals:
        row["PLAYER_1_FINAL_SCORE"], row["PLAYER_2_FINAL_SCORE"] = str(finals[0]), str(finals[1])
    return row


class Flat:
    """A pre-match model that prices every match 20-20, and records what it was asked."""
    league = (17.0, 17.0)
    meta = {"before": CUT.isoformat()}

    def __init__(self):
        self.asked = []

    def describe(self):
        return "flat"

    def means(self, schedule, n_sims=None, results=None):
        self.asked.append([r["MATCH_CODE"] for r in schedule])
        return {r["MATCH_CODE"]: (20.0, 20.0) for r in schedule}


class TestFollowing(unittest.TestCase):

    def test_a_gamer_beating_their_price_is_priced_up_shrunk_by_the_prior(self):
        # ann wins 4 matches by 10 after the cut-off (errors +10 each); one before it doesn't count
        results = [match("OLD", -60, "ann", "x", (30, 20))] + \
                  [match(f"W{k}", 36 * k, "ann", f"o{k}", (25, 15)) for k in range(4)]
        later = match("NEXT", 36 * 10, "ann", "bob")
        pre = follow.Following(Flat(), CUT, prior=80)
        got = pre.means([later], results=results)["NEXT"]
        offset = 4 * 10 / (4 + 80)
        self.assertAlmostEqual(got[0] - got[1], offset)              # margin moves by ann's offset
        self.assertAlmostEqual(got[0] + got[1], 40 + (0 + 0) / 2)    # totals were as priced: none
        # the settled matches were priced alongside, once
        self.assertEqual(sorted(pre.pre.asked[0]), ["NEXT", "W0", "W1", "W2", "W3"])

    def test_only_matches_before_this_one_count_and_totals_follow_too(self):
        results = [match(f"H{k}", 36 * k, "ann", "bob", (30, 30)) for k in range(3)]   # totals +20
        pre = follow.Following(Flat(), CUT, prior=80)
        rows = [match("H1", 36, "ann", "bob"), match("LATER", 36 * 5, "ann", "bob"),
                match("EARLY", -30, "ann", "bob")]
        got = pre.means(rows, results=results)
        # H1 sees only H0: margin level (both gamers' margin errors 0), total +20 / 81 each side
        self.assertAlmostEqual(sum(got["H1"]), 40 + 20 / 81)
        self.assertAlmostEqual(sum(got["LATER"]), 40 + 3 * 20 / 83)
        self.assertEqual(got["EARLY"], (20.0, 20.0))                 # before any result
        self.assertAlmostEqual(got["LATER"][0], got["LATER"][1])

    def test_an_opponent_s_offset_counts_against(self):
        results = [match(f"L{k}", 36 * k, "cat", f"o{k}", (10, 30)) for k in range(5)]   # cat -20
        got = follow.Following(Flat(), CUT, prior=80).means([match("N", 400, "ann", "cat")], results=results)["N"]
        self.assertAlmostEqual(got[0] - got[1], 5 * 20 / 85)          # ann, the home side, gains

    def test_no_results_since_the_fit_is_the_model_as_it_was(self):
        flat = Flat()
        pre = follow.Following(flat, CUT)
        self.assertEqual(pre.means([match("N", 10, "a", "b")]), {"N": (20.0, 20.0)})
        self.assertEqual(pre.means([match("N", 10, "a", "b")], results=[match("OLD", -5, "a", "b", (9, 1))]),
                         {"N": (20.0, 20.0)})
        self.assertEqual(pre.league, (17.0, 17.0))                    # the rest is the model's own
        self.assertTrue(pre.FOLLOWS_RESULTS and pre.AS_OF)
        self.assertIn("following each gamer's results since 2026-09-01", pre.describe())

    def test_wrap_needs_the_cut_off_and_leaves_a_rolling_prior_alone(self):
        flat = Flat()
        self.assertIsInstance(follow.wrap(flat), follow.Following)
        self.assertEqual(follow.wrap(flat).since, CUT)
        self.assertIs(follow.wrap(flat, rolling=True), flat)
        no_cut = Flat()
        no_cut.meta = {}
        self.assertIs(follow.wrap(no_cut), no_cut)
        self.assertIsNone(follow.wrap(None))
        with mock.patch.object(follow, "ON", False):
            self.assertIs(follow.wrap(flat), flat)


class TestVersionsFollow(unittest.TestCase):

    def test_each_version_s_prior_follows_the_results_since_its_fit(self):
        for version in (v10, v11, v12):
            with tempfile.TemporaryDirectory() as d:
                os.makedirs(os.path.join(d, "nb2"))
                with open(os.path.join(d, version.PRIOR_FILE), "w") as fh:
                    json.dump({"prior": "nb2"}, fh)
                with mock.patch.object(nb2_prior.Prematch, "exists", return_value=True), \
                        mock.patch.object(nb2_prior.Prematch, "__init__", return_value=None):
                    with mock.patch.object(nb2_prior.Prematch, "meta", {"before": CUT.isoformat()}, create=True):
                        pre = version.prematch_model(d)
                        self.assertIsInstance(pre, follow.Following, version.__name__)
                        with open(os.path.join(d, version.PRIOR_FILE), "w") as fh:
                            json.dump({"prior": "nb2", "rolling": d}, fh)
                        with mock.patch("eAMFModel.rolling_prior.Rolling") as rolling:
                            self.assertIs(version.prematch_model(d), rolling.return_value)


if __name__ == "__main__":
    unittest.main()
