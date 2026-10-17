import csv
import datetime as dt
import os
import tempfile
import unittest

from eAMFCalibrator import ml_states as ms

T0 = dt.datetime(2026, 10, 1, 8, 0)
PAIR_FIELDS = ["publish_time", "match_code", "drive_number", "period_number", "score_p1", "score_p2", "offensive_team",
               "market_id", "prod_probability", "candidate_probability", "prod_outcome", "prod_live", "candidate_live"]
BET_FIELDS = ["match_code", "bet_time", "in_play", "market", "selection", "stake", "odds", "revenue", "result",
              "stream_prob", "candidate_prob", "candidate_revenue", "simulated"]


def _match(code, home_fav, dog_scores_first, home_won, start):
    """Snapshots of one match: the opening drive by the side that scores first (a TD and the try), the other
    side's drive after it, and a second-quarter snapshot."""
    fav_p = 0.56 if home_fav else 0.44
    first = "Away Team" if (dog_scores_first == home_fav) else "Home Team"
    other = "Home Team" if first == "Away Team" else "Away Team"
    p1, p2 = (7, 0) if first == "Home Team" else (0, 7)
    after = 0.40 if dog_scores_first else 0.66                     # prod's price on the favourite after it
    home_after = after if home_fav else 1 - after
    snaps = [(0, 1, 0, 0, first, fav_p), (20, 1, 6 if p1 else 0, 6 if p2 else 0, first, fav_p),
             (30, 1, p1, p2, first, fav_p), (40, 1, p1, p2, other, home_after), (300, 2, p1, p2, other, home_after)]
    return [dict(publish_time=f"{start + dt.timedelta(seconds=s):%Y-%m-%d %H:%M:%S}", match_code=code, drive_number=k + 1,
                 period_number=per, score_p1=a, score_p2=b, offensive_team=off, market_id=50, prod_probability=p,
                 candidate_probability=p, prod_outcome=int(home_won), prod_live=1, candidate_live=1)
            for k, (s, per, a, b, off, p) in enumerate(snaps)]


class TestMlStates(unittest.TestCase):
    def setUp(self):
        self.rows, self.bets = [], []
        for i in range(40):
            start = T0 + dt.timedelta(hours=i)
            home_fav, dog_first = i % 2 == 0, i % 4 < 2
            fav_won = i % 5 != 0                                    # the favourite wins four in five
            home_won = fav_won == home_fav
            code = f"AF{i:03d}011026"
            self.rows += _match(code, home_fav, dog_first, home_won, start)
            sel = "Home" if home_fav else "Away"                    # a bet on the favourite after the first score
            won = fav_won
            self.bets.append(dict(match_code=code, bet_time=f"{start + dt.timedelta(seconds=60):%Y-%m-%d %H:%M:%S}",
                                  in_play="True", market="moneyline", selection=sel, stake=100, odds=2.0,
                                  revenue=-100 if won else 100, result="won" if won else "lost", stream_prob=0.5,
                                  candidate_prob=0.6, candidate_revenue=-66.7 if won else 100, simulated="True"))

    def _write(self, d):
        pairs, bets = os.path.join(d, "pairs.csv"), os.path.join(d, "bets.csv")
        for path, rows, fields in ((pairs, self.rows, PAIR_FIELDS), (bets, self.bets, BET_FIELDS)):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fields)
                w.writeheader()
                w.writerows(rows)
        return pairs, bets

    def test_the_first_score_is_read_once_the_other_side_has_the_ball(self):
        with tempfile.TemporaryDirectory() as d:
            pairs, _ = self._write(d)
            by, won = ms.read_pairs([pairs])
        s = ms.match_states(by["AF000011026"], won["AF000011026"])   # home favourite, the underdog scored first
        self.assertEqual(s["fav"], 1)
        self.assertEqual(s["first"]["situation"], "the underdog TD, opening drive")
        self.assertAlmostEqual(s["first"]["prod"], 0.40)
        self.assertEqual(s["first"]["row"], 3)
        self.assertEqual(s["quarters"][0][:2], (2, "1 score behind"))
        s = ms.match_states(by["AF002011026"], won["AF002011026"])   # the favourite scored first
        self.assertEqual(s["first"]["who"], "the favourite")

    def test_the_report(self):
        with tempfile.TemporaryDirectory() as d:
            pairs, bets = self._write(d)
            path = ms.run(os.path.join(d, "out"), [pairs], [bets], label="v21")
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        self.assertIn("the underdog TD, opening drive", text)
        self.assertIn("the favourite TD, opening drive", text)
        self.assertIn("the underdog's touchdown on the opening drive, by how strong a favourite", text)
        self.assertIn("40 settled in-play moneyline bets", text)
        line = next(l for l in text.splitlines() if l.strip().startswith("the underdog TD, opening drive") and "%" not in l)
        self.assertIn("0.400", line)                                 # prod's price on the favourite after it

    def test_by_gamer(self):
        with tempfile.TemporaryDirectory() as d:
            pairs, bets = self._write(d)
            hist = os.path.join(d, "history.csv")
            with open(hist, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["MATCH_CODE", "SCHEDULED_START_TIME_UTC", "PLAYER_1_HANDLE", "PLAYER_2_HANDLE",
                            "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE"])
                for i in range(40):
                    home_won = (i % 5 != 0) == (i % 2 == 0)
                    w.writerow([f"AF{i:03d}011026", f"{T0:%Y-%m-%d %H:%M:%S}", "ACE" if i % 2 == 0 else "BOB",
                                "BOB" if i % 2 == 0 else "ACE", 21 if home_won else 20, 20 if home_won else 21])
            by, won = ms.read_pairs([pairs])
            states = {c: s for c, rows in by.items() if (s := ms.match_states(rows, won.get(c)))}
            placed = ms.place_bets(ms.read_bets([bets]), states)
            numbers = ms.gamer_numbers(states, ms.read_history(hist), placed)
            path = ms.run(os.path.join(d, "out"), [pairs], [bets], label="v21", history=hist)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertTrue(os.path.isfile(os.path.join(d, "out", "ml_states_gamers.csv")))
        ace = numbers["ACE"]                                          # always the favourite, wins four in five
        self.assertEqual(ace["matches"], 40)
        self.assertAlmostEqual(ace["as_fav"], 1.0)
        self.assertAlmostEqual(ace["won"], 0.8)
        self.assertAlmostEqual(ace["kick_prod"], 0.56)
        self.assertEqual(ace["conceded_first"] + ace["scored_first"], 40)
        self.assertAlmostEqual(ace["close_won"], 0.8)                 # every game decided by one point
        self.assertEqual(ace["bets"]["bets"], 40)
        self.assertIsNone(numbers["BOB"]["bets"])
        self.assertIn("by gamer (2 with 12+ matches", text)

    def test_lead_bands(self):
        self.assertEqual(ms.lead_band(-10), "2+ scores behind")
        self.assertEqual(ms.lead_band(0), "level (within 2)")
        self.assertEqual(ms.lead_band(7), "1 score ahead")


if __name__ == "__main__":
    unittest.main()
