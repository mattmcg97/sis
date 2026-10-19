import datetime as dt
import unittest

from eAMFCalibrator import pair_check

T0 = dt.datetime(2026, 10, 1, 12, 0, 0)


def quote(market, seconds, prob, msg, active="true"):
    return ("M1", market, T0 + dt.timedelta(seconds=seconds), prob, None, None, msg, "OPEN", active)


def snap(message, next_start):
    return {"message": message, "next_start_message": next_start, "team_a_side": "home",
            "offense": "TEAM_A", "period": 1, "field_position": 25, "down": 1, "distance": 10,
            "file_time": None, "score_p1": 0, "score_p2": 0, "clock_seconds": 600}


class TestPairCheck(unittest.TestCase):
    def setUp(self):
        # home won 21-14; PLAY_OVERs at 10 (until 15) and 20 (until 25)
        self.finals = {"M1": (21, 14)}
        self.snaps = {"M1": [snap(10, 15), snap(20, 25)]}
        self.quotes = [
            quote(50, 0, 40.0, 11), quote(50, 2, 55.0, 11),       # message 11: a first row, corrected
            quote(50, 3, 60.0, 12),
            quote(50, 9, 70.0, 21),                               # message 21: one row
            quote(50, 1, 99.0, 11, active="false"),               # dead: never read
            quote(51, 0, 60.0, 11), quote(51, 2, 45.0, 11),
        ]

    def test_first_and_last_row_on_the_message_each_play_over_pairs_on(self):
        rows = pair_check.at_play_overs(self.quotes, self.snaps, self.finals)
        home = [r for r in rows if r["market_id"] == 50]
        self.assertEqual([(r["play_over_message"], r["message"], r["rows"]) for r in home], [(10, 11, 2), (20, 21, 1)])
        self.assertEqual((home[0]["first_prob"], home[0]["last_prob"], home[0]["outcome"]), (0.40, 0.55, 1))
        self.assertEqual(home[0]["seconds_first_to_last"], 2.0)
        away = [r for r in rows if r["market_id"] == 51]
        self.assertEqual((away[0]["first_prob"], away[0]["last_prob"], away[0]["outcome"]), (0.60, 0.45, 0))

    def test_no_message_in_the_window_no_row(self):
        snaps = {"M1": [snap(13, 14)]}                     # nothing quoted in [13, 14)
        self.assertEqual(pair_check.at_play_overs(self.quotes, snaps, self.finals), [])

    def test_a_staler_first_row_shows_as_a_positive_gap(self):
        rows = pair_check.at_play_overs(self.quotes, self.snaps, self.finals)
        mean, lo, hi, m = pair_check.brier_gap(rows, n_boot=50)
        # home at 11: (0.40-1)^2 - (0.55-1)^2 = 0.1575; at 21: 0; away at 11: 0.36 - 0.2025 = 0.1575
        self.assertAlmostEqual(mean, (0.1575 + 0 + 0.1575) / 3)
        self.assertEqual(m, 1)
        self.assertTrue(any("first and last differ" in line for line in pair_check.summary(rows)))

    def test_a_push_or_no_final_is_left_out(self):
        self.assertEqual(pair_check.at_play_overs(self.quotes, self.snaps, {"M1": (14, 14)}), [])
        self.assertEqual(pair_check.at_play_overs(self.quotes, self.snaps, {}), [])


if __name__ == "__main__":
    unittest.main()
