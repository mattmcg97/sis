"""Bets and stake by each gamer's place in his session."""

import datetime as dt
import unittest

from .. import bet_sessions as bs, bets, config
from ..bet_totals import sessions


def game(code, minutes, p1, p2):
    t = dt.datetime(2026, 9, 15, 10) + dt.timedelta(minutes=minutes)
    return dict(MATCH_CODE=code, SCHEDULED_START_TIME_UTC=t.strftime("%Y-%m-%d %H:%M:%S"),
                PLAYER_1_HANDLE=p1, PLAYER_2_HANDLE=p2)


def bet(code, selection=1, market=3, stake=10.0, revenue=10.0, temp="Standard"):
    return bets.Bet("x", code, dt.datetime(2026, 9, 15, 12), market, selection, 1.9, stake, revenue,
                    extra={config.BET_VIP_COLUMN: temp})


class TestSessions(unittest.TestCase):

    # ann plays three in a row; bob joins her first and third; cat her second
    HISTORY = [game("M1", 0, "ann", "bob"), game("M2", 40, "ann", "cat"), game("M3", 80, "ann", "bob")]

    def setUp(self):
        self.pos = sessions(self.HISTORY)

    def test_each_match_appears_once_for_each_gamer_at_his_place(self):
        apps = bs.appearances(self.HISTORY, self.pos, None, None)
        self.assertEqual(apps[("match", "1")], 3)       # ann M1, bob M1, cat M2
        self.assertEqual(apps[("match", "2")], 2)       # ann M2, bob M3
        self.assertEqual(apps[("match", "3")], 1)       # ann M3
        self.assertEqual(apps[("left", "last")], 3)     # cat M2, ann M3, bob M3

    def test_a_bet_counts_for_both_gamers_by_group_and_market(self):
        t = bs.tally([bet("M3", selection=2, stake=20.0, revenue=-18.0, temp="Restricted"),
                      bet("M1", market=1, temp="VIP")], self.HISTORY, self.pos)
        self.assertEqual(t[("match", "3", "Restricted", "all")][:2], [1, 20.0])     # ann's 3rd
        self.assertEqual(t[("match", "2", "All", markets_total())][4], 20.0)        # bob's 2nd, under
        self.assertEqual(t[("match", "1", "VIP", "moneyline")][0], 2)               # ann and bob, M1
        lines = bs.report(t, bs.appearances(self.HISTORY, self.pos, None, None))
        self.assertTrue(any("stake/match" in x for x in lines))


def markets_total():
    from .. import markets
    return markets.TOTAL


if __name__ == "__main__":
    unittest.main()
