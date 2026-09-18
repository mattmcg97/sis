"""Recover a wall-clock time for a play, from the stream's own message clock.

INPLAY_FIELD_POSITION_PERIOD has no timestamp -- it carries MATCH_CODE,
EVENT_MESSAGE_COUNT, PERIOD_NUMBER, OFFENSIVE_TEAM, DOWN_NUMBER, DISTANCE
and FIELD_POSITION, and nothing else. But EVENT_MESSAGE_COUNT is the same
feed sequence the GAMEPLAI streams are keyed on, and those DO carry
PUBLISH_TIME. So the play feed's clock is reconstructed from the stream:

  exact         the stream quoted that same message -- its PUBLISH_TIME is
                the message's time, no estimation involved
  interpolated  the stream never quoted that message, so the time is placed
                linearly between the bracketing quoted messages
  unresolved    the message sits outside the stream's range for that match,
                or the bracket is wider than config.MAX_BRACKET_MESSAGES,
                which would make the interpolation guesswork

Provenance is carried on every snapshot and counted in the run header, so
a run that leans heavily on interpolation says so rather than quietly
reporting a 3-second match that was never measured against a real clock.
"""

from bisect import bisect_left

from . import config

EXACT = "exact"
INTERPOLATED = "interpolated"
UNRESOLVED = "unresolved"


class MatchClock:
    """Message count -> time, for one match."""

    def __init__(self, message_times):
        # message_times: {event_message_count: datetime}
        self.messages = sorted(message_times)
        self.times = [message_times[m] for m in self.messages]

    def __len__(self):
        return len(self.messages)

    def time_for(self, message_count, max_bracket=None):
        """Returns (time, provenance). time is None when unresolved."""
        if not self.messages or message_count is None:
            return None, UNRESOLVED

        max_bracket = config.MAX_BRACKET_MESSAGES if max_bracket is None else max_bracket

        idx = bisect_left(self.messages, message_count)
        if idx < len(self.messages) and self.messages[idx] == message_count:
            return self.times[idx], EXACT

        # Needs a quoted message on each side to interpolate between.
        if idx == 0 or idx >= len(self.messages):
            return None, UNRESOLVED

        lo_msg, hi_msg = self.messages[idx - 1], self.messages[idx]
        span = hi_msg - lo_msg
        if span <= 0 or span > max_bracket:
            return None, UNRESOLVED

        lo_time, hi_time = self.times[idx - 1], self.times[idx]
        fraction = (message_count - lo_msg) / span
        return lo_time + (hi_time - lo_time) * fraction, INTERPOLATED


def build_clocks(clock_rows):
    """Rows of (match_code, event_message_count, publish_time) -> clocks.

    A message carries one row per market, all published together, so the
    earliest PUBLISH_TIME for a message is taken as its time.
    """
    per_match = {}
    for match_code, message_count, publish_time in clock_rows:
        if message_count is None or publish_time is None:
            continue
        bucket = per_match.setdefault(match_code, {})
        current = bucket.get(message_count)
        if current is None or publish_time < current:
            bucket[message_count] = publish_time
    return {match_code: MatchClock(times) for match_code, times in per_match.items()}
