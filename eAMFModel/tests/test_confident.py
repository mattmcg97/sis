"""v4-v6 price only where they are sure of the state: the latest PLAY_OVER, read, until the next
play starts, and TEAM_A's side known."""

import unittest

import numpy as np

from .. import stream, v4_stream, v5_stream, v6_stream


def snap(message, next_start, side="home"):
    return dict(message=message, next_start_message=next_start, team_a_side=side)


SNAPS = [snap(5, 9), snap(11, 15), snap(17, 20)]


class TestConfident(unittest.TestCase):

    def test_each_read_play_over_lasts_until_the_next_play_starts(self):
        self.assertEqual(stream.confident_windows(SNAPS, [5, 11]), {5: 9, 11: 15})
        self.assertEqual(stream.confident_windows([snap(5, None), snap(8, 12)], [5, 8]),
                         {5: 8, 8: 12})
        self.assertEqual(stream.confident_windows([snap(5, 9, side=None)], [5]), {})
        self.assertEqual(stream.confident_windows([snap(5, "")], [5]), {5: float("inf")})

    def test_the_streams_quote_only_inside_the_windows(self):
        margin = np.zeros(201)
        margin[103] = 1.0
        total = np.zeros(161)
        total[45] = 1.0
        books = [(5, margin, total), (11, margin, total)]
        prod = [("M1", 50, None, 50.0, 2.0, None, m, "open", "true") for m in range(4, 21)]
        windows = stream.confident_windows(SNAPS, [5, 11])
        for module in (v4_stream, v5_stream, v6_stream):
            rows = module.quote_rows("M1", books, prod, windows=windows)
            self.assertEqual([r[6] for r in rows], [5, 6, 7, 8, 11, 12, 13, 14], module.__name__)
            every = module.quote_rows("M1", books, prod)
            self.assertEqual(len(every), 16)


if __name__ == "__main__":
    unittest.main()
