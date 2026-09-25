"""Prod's lines through a match against the lines bet."""

import datetime as dt
import unittest
from unittest import mock

from .. import bet_lines, bets, config

T0 = dt.datetime(2026, 9, 20, 12, 0, 0)


def at(seconds):
    return T0 + dt.timedelta(seconds=seconds)


def row(market, seconds, line, active="true", prob=50.0, match="M1"):
    """A snowflake_io.fetch_quotes row."""
    return (match, market, at(seconds), prob, None, f"Line {line}", seconds, "open", active)


def bet(seconds, line, market_type=3, selection=1, operator="FANDUEL", match="M1"):
    return bets.Bet(bet_id=seconds, match_code=match, time=at(seconds), market_type=market_type,
                    selection=selection, odds=1.9, stake=10, revenue=10, line=line,
                    extra={config.BET_GROUP_COLUMN: operator, config.BET_IN_PLAY_COLUMN: "Yes"})


# the total: 44.5 live from 0s, suspended at 60s, 47.5 live from 70s, 41.5 live from 200s
ROWS = [row(54, 0, 44.5), row(54, 30, 44.5), row(54, 60, 44.5, active="false"),
        row(54, 70, 47.5), row(54, 200, 41.5), row(54, 260, 41.5)]


class TestSpans(unittest.TestCase):

    def test_runs_of_one_line_are_merged_and_last_until_the_next_change(self):
        sp = bet_lines.spans(ROWS)[("M1", 54)]
        self.assertEqual([(s[2], s[3]) for s in sp],
                         [(44.5, True), (44.5, False), (47.5, True), (41.5, True)])
        self.assertEqual((sp[0][0], sp[0][1]), (at(0), at(60)))
        self.assertEqual(sp[-1][1], at(260))


class TestClassify(unittest.TestCase):

    def setUp(self):
        self.sp = bet_lines.spans(ROWS)[("M1", 54)]

    def test_a_bet_on_prods_live_line_is_current(self):
        self.assertEqual(bet_lines.classify(at(100), 47.5, self.sp), (bet_lines.CURRENT, 0.0))

    def test_a_bet_on_the_line_prod_just_left_is_past_by_how_long_ago(self):
        self.assertEqual(bet_lines.classify(at(210), 47.5, self.sp), (bet_lines.PAST, 10.0))

    def test_a_bet_on_the_line_prod_is_about_to_show_is_future(self):
        self.assertEqual(bet_lines.classify(at(190), 41.5, self.sp), (bet_lines.FUTURE, 10.0))

    def test_a_line_prod_never_showed_nearby_is_never(self):
        self.assertEqual(bet_lines.classify(at(100), 50.5, self.sp), (bet_lines.NEVER, None))

    def test_while_suspended_the_last_live_line_is_current(self):
        self.assertEqual(bet_lines.classify(at(65), 44.5, self.sp)[0], bet_lines.CURRENT)


class TestReport(unittest.TestCase):

    def test_the_90th_percentile_is_never_below_the_median(self):
        sp = bet_lines.spans(ROWS)
        items = [(bet(210, 47.5), 47.5, bet_lines.PAST, 10.0),
                 (bet(211, 47.5), 47.5, bet_lines.PAST, 40.0)]
        text = "\n".join(bet_lines.summary(items))
        self.assertIn("25.0   40.0", text)

    def test_bets_are_read_on_prods_side_and_summarised(self):
        sp = bet_lines.spans(ROWS + [row(52, 0, 3.5)])
        signs = {("FANDUEL", 52): (-1, 1, 0, 1)}
        classified = bet_lines.classify_bets(
            [bet(100, 47.5), bet(210, 47.5), bet(50, -3.5, 2, 1), bet(20, None, 1, 1)], sp, signs)
        self.assertEqual([c[2] for c in classified],
                         [bet_lines.CURRENT, bet_lines.PAST, bet_lines.CURRENT])
        text = "\n".join(bet_lines.summary(classified))
        self.assertIn("FANDUEL · total", text)
        html = bet_lines.page(["M1"], sp, classified, ["summary"])
        self.assertIn("<svg", html)
        self.assertIn("<circle", html)

    def test_the_command_writes_the_page(self):
        from .. import snowflake_io
        cols = ["MATCH_CODE", "BET_DATE_UTC", "MARKET_TYPE_ID", "SELECTION_ID", "ODDS", "STAKE_GBP",
                "REVENUE_GBP", "MARKET_LINE", "OPERATOR_NAME", "BET_IN_PLAY"]
        raw = [("M1", at(100), 3, 1, 1.9, 10, 10, 47.5, "FANDUEL", "Yes"),
               ("M1", at(210), 3, 1, 1.9, 10, 10, 47.5, "FANDUEL", "Yes")]
        opened = mock.mock_open()
        with mock.patch.object(bets, "fetch_all", return_value=(cols, raw)), \
                mock.patch.object(snowflake_io, "fetch_quotes", return_value=ROWS), \
                mock.patch("builtins.open", opened), mock.patch("builtins.print"), \
                mock.patch("os.makedirs"):
            path, classified = bet_lines.run(None, "out")
        self.assertTrue(path.endswith("bets_lines.html"))
        self.assertEqual([c[2] for c in classified], [bet_lines.CURRENT, bet_lines.PAST])


if __name__ == "__main__":
    unittest.main()
