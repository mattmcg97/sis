"""The progress bar: drawn over itself on a terminal, nothing anywhere else, never wider than the
terminal."""

import io
import os
import unittest
from unittest import mock

from .. import progress


class TTY(io.StringIO):
    def isatty(self):
        return True


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def shown(written):
    """What a terminal shows after `written`: each carriage return goes back to the line's start."""
    line = []
    for seg in written.split("\r"):
        for i, ch in enumerate(seg):
            if i < len(line):
                line[i] = ch
            else:
                line.append(ch)
    return "".join(line).rstrip()


class TestBar(unittest.TestCase):

    def test_on_a_terminal_it_draws_over_itself(self):
        tty, clock = TTY(), Clock()
        bar = progress.Bar("v21: simulating", 10, "matches", stream=tty, clock=clock)
        self.assertIn("0/10 matches 0%", shown(tty.getvalue()))
        clock.t += 60
        bar.update(4)
        self.assertNotIn("\n", tty.getvalue())
        self.assertEqual(shown(tty.getvalue()),
                         "  v21: simulating: [##########..............] 4/10 matches 40%, 1m00s in, ~1m30s left")
        drawn = tty.getvalue().count("\r")
        clock.t += 0.1
        bar.update(1)                                   # inside EVERY: not redrawn
        self.assertEqual(tty.getvalue().count("\r"), drawn)
        clock.t += 0.1
        bar.update(5, note="the last")                  # done: always drawn
        self.assertEqual(shown(tty.getvalue()),
                         "  v21: simulating: [########################] 10/10 matches 100%, 1m00s in  the last")
        bar.close()
        self.assertEqual(shown(tty.getvalue()), "")
        self.assertNotIn("\n", tty.getvalue())

    def test_a_shorter_line_covers_the_longer_one_before(self):
        tty = TTY()
        line = progress.Line(tty)
        line.draw("a long line of progress")
        line.draw("short")
        self.assertEqual(shown(tty.getvalue()), "short")

    def test_sent_to_a_file_it_draws_nothing(self):
        out = io.StringIO()
        bar = progress.Bar("reading", 3, "matches", stream=out)
        bar.update(3)
        bar.close()
        self.assertEqual(out.getvalue(), "")
        self.assertFalse(progress.live(out))

    def test_it_never_wraps(self):
        tty = TTY()
        with mock.patch.object(progress.shutil, "get_terminal_size", return_value=os.terminal_size((40, 24))):
            bar = progress.Bar("reading the play feed from Snowflake", 1000, "matches", stream=tty)
            bar.update(10)
        self.assertLessEqual(max(len(seg) for seg in tty.getvalue().split("\r")), 39)

    def test_times(self):
        self.assertEqual([progress.fmt(s) for s in (42, 185, 3720)], ["42s", "3m05s", "1h02m"])


if __name__ == "__main__":
    unittest.main()
