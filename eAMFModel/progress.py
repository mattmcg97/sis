"""A progress bar that draws over itself on one line, for the long runs (the betting sim's Snowflake
reads and simulation, the walkthrough).

On a terminal it redraws in place at most every EVERY seconds: done of total, a bar, the share, the
time in and the time left at the pace so far. Anywhere else (output sent to a file) it draws nothing,
so a log keeps only the lines the run prints itself. It writes to stderr, so with the output sent to a
file the bar still shows on the screen. It pads over the line before rather than using terminal codes,
so a Windows console draws it too, and it is cut to the terminal's width so it never wraps.
"""

import shutil
import sys
import time

EVERY = 0.5                     # seconds between redraws
WIDTH = 24                      # the bar's characters


def live(stream=None):
    """Whether `stream` (stderr) is a terminal a bar can redraw on."""
    s = stream if stream is not None else sys.stderr
    try:
        return bool(s.isatty())
    except (AttributeError, ValueError):
        return False


def fmt(seconds):
    """Seconds as '42s', '3m05s' or '1h02m'."""
    seconds = max(0, int(round(seconds)))
    h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


class Line:
    """One line of a terminal, drawn over itself: draw(text) replaces what's there, clear() leaves it
    empty for the next print. Silent where the stream isn't a terminal (live False)."""

    def __init__(self, stream=None, is_live=None):
        self.stream = stream if stream is not None else sys.stderr
        self.live = live(self.stream) if is_live is None else is_live
        self.shown = 0

    def draw(self, text):
        if not self.live:
            return
        cols = shutil.get_terminal_size((100, 24)).columns - 1
        text = text[:max(10, cols)]
        self.stream.write("\r" + text + " " * max(0, self.shown - len(text)))
        self.stream.flush()
        self.shown = len(text)

    def clear(self):
        if self.live and self.shown:
            self.stream.write("\r" + " " * self.shown + "\r")
            self.stream.flush()
        self.shown = 0


class Bar:
    """`total` units of `unit`, counted with update(n): '  label: [#####.....] 412/1,230 matches 33%,
    4m10s in, ~8m30s left'. close() clears it (the run prints its own line after)."""

    def __init__(self, label, total, unit="", stream=None, is_live=None, clock=time.monotonic, every=EVERY):
        self.label, self.total, self.unit = label, max(0, int(total)), unit
        self.line = Line(stream, is_live)
        self.clock, self.every = clock, every
        self.t0 = clock()
        self.done, self.last, self.note = 0, None, ""
        self.draw(force=True)

    def text(self, now=None):
        now = self.clock() if now is None else now
        elapsed = now - self.t0
        share = self.done / self.total if self.total else 1.0
        k = int(round(WIDTH * min(1.0, share)))
        left = elapsed / self.done * (self.total - self.done) if self.done else None
        return (f"  {self.label}: [{'#' * k}{'.' * (WIDTH - k)}] {self.done:,}/{self.total:,}"
                + (f" {self.unit}" if self.unit else "") + f" {100 * share:.0f}%, {fmt(elapsed)} in"
                + (f", ~{fmt(left)} left" if left is not None and self.done < self.total else "")
                + (f"  {self.note}" if self.note else ""))

    def draw(self, force=False):
        now = self.clock()
        if force or self.last is None or now - self.last >= self.every or self.done >= self.total:
            self.last = now
            self.line.draw(self.text(now))

    def update(self, n=1, note=None):
        self.done += n
        if note is not None:
            self.note = note
        self.draw()

    def close(self):
        self.line.clear()
