"""Follow a match through the raw feed, one message at a time, causally.

The calibrator's cleaning (eAMFCalibrator/drives.py) reads a row against
the row AFTER it -- the right call for auditing history, and not available
to a price that has to go out now. This tracker only ever looks back.

Every message ends in one of two verdicts:

  LIVE       the state is a real snap the model can price
  SUSPENDED  the model will not quote, because the state is garbage or
             about to change under it

Suspended:
  - from the opening whistle to the first real snap (the kick is not a snap)
  - from any score change until the next drive's first real snap: the
    extra point, the kickoff and the return all sit in that window, and
    the score is liable to move again (the PAT) while it is open. Rows of
    the team that scored stay suspended in it -- they are the conversion
    and the kick -- unless that team is still on the ball ONSIDE_MESSAGES
    later, off the conversion spots: an onside recovery
  - from the start of each half (and overtime) to its first real snap
  - a row at the kickoff spot (1st-and-10 on the 35) outside a drive
  - a label flip onto the previous row's field: the possession has
    changed but the spot has not been re-read from the new side yet (a
    real change of possession flips the field, 100 - y)
  - a down that skips ahead with nothing else moving (1st to 3rd, same
    spot): the feed's vision layer misread it; held for that row

An exact republish of the last row changes nothing and keeps the last
verdict. Everything else is a snap: priced, live.

A drive starts on a live row whose team differs from the last live row's,
or on the first live row after a suspension window, and its age (messages
since then) is carried so the pricer knows how much of it is used up.
"""

from dataclasses import dataclass, replace
from typing import Optional

from .pricer import AWAY, HOME, GameState

LIVE = "live"
SUSPENDED = "suspended"

# Why a message was suspended, for the trace.
OPENING = "opening"
AFTER_SCORE = "after_score"
NEW_HALF = "new_half"
KICK_SPOT = "kick_spot"
STALE = "stale_label"
IMPOSSIBLE_DOWN = "impossible_down"
NOT_A_SNAP = "not_a_snap"

KICKOFF_FIELD = 35
MAX_DISTANCE = 40
# Where a conversion is tried from: the extra point (85) and the two (98).
CONVERSION_SPOTS = (85, 98)
# A scoring team still on the ball this long after its score recovered an
# onside kick; before that its rows are the conversion and the kick.
ONSIDE_MESSAGES = 20


def side(label):
    if not label:
        return None
    text = str(label).lower()
    if text.startswith("home") or text in ("player 1", "p1", "1"):
        return HOME
    if text.startswith("away") or text in ("player 2", "p2", "2"):
        return AWAY
    return None


@dataclass(frozen=True)
class Play:
    message: int
    period: Optional[int]
    offense: Optional[str]
    down: Optional[int]
    distance: Optional[int]
    field_position: Optional[int]

    @property
    def key(self):
        return (self.offense, self.down, self.distance, self.field_position)

    @property
    def plausible(self):
        return (self.offense in (HOME, AWAY) and self.down in (1, 2, 3, 4)
                and self.distance is not None and 0 < self.distance <= MAX_DISTANCE
                and self.field_position is not None and 0 < self.field_position < 100)

    @property
    def at_kick_spot(self):
        return self.down == 1 and self.distance == 10 and self.field_position == KICKOFF_FIELD


@dataclass(frozen=True)
class Score:
    message: int
    period: Optional[int]
    home: Optional[int]
    away: Optional[int]


@dataclass(frozen=True)
class Tick:
    """The tracker's view after one message."""
    message: int
    verdict: str
    reason: Optional[str]
    state: Optional[GameState]     # None until the first period is known


class Tracker:
    def __init__(self):
        self.period = None
        self.period_starts = {}
        self.home = 0
        self.away = 0
        self.last_play = None           # last play row seen, any verdict
        self.last_live = None           # last play row priced live
        self.drive_start = None         # message the current drive began
        self.opening_receiver = None
        self.window = OPENING           # open suspension window, or None
        self.window_opened = None       # message the window opened on
        self.scorer = None
        self.verdict = SUSPENDED
        self.reason = OPENING

    # -- events --------------------------------------------------------

    def _enter_period(self, period, message):
        if period is None or period == self.period:
            return
        new_half = self.period is not None and (period == 3 or period >= 5)
        self.period = period
        self.period_starts.setdefault(period, message)
        if new_half:
            self.window = NEW_HALF
            self.window_opened = message
            self.last_live = None

    def on_score(self, score):
        self._enter_period(score.period, score.message)
        home = self.home if score.home is None else score.home
        away = self.away if score.away is None else score.away
        if (home, away) != (self.home, self.away):
            self.scorer = HOME if home > self.home else AWAY if away > self.away else None
            self.home, self.away = home, away
            if self.window != AFTER_SCORE:
                self.window_opened = score.message
            self.window = AFTER_SCORE
        return self._tick(score.message, SUSPENDED, self.window)

    def on_play(self, play):
        self._enter_period(play.period, play.message)
        previous = self.last_play
        self.last_play = play

        # A republish: nothing new, same verdict as before.
        if previous is not None and play.key == previous.key:
            return self._tick(play.message, self.verdict, self.reason)

        if not play.plausible:
            return self._tick(play.message, SUSPENDED, self.window or NOT_A_SNAP)

        if previous is not None and self._impossible_down(play, previous):
            return self._tick(play.message, SUSPENDED, IMPOSSIBLE_DOWN)
        if any(row is not None and self._stale(play, row)
               for row in (previous, self.last_live)):
            return self._tick(play.message, SUSPENDED, STALE)

        if self.window is not None:
            if play.at_kick_spot:
                return self._tick(play.message, SUSPENDED, self.window)
            if self.window == AFTER_SCORE and play.offense == self.scorer:
                # The scoring team straight after its own score: the extra
                # point (the TD play's down and distance, left on the 85 or
                # the 98) and the kick. Only a scorer's drive that outlasts
                # the window -- an onside recovery -- is real.
                opened = self.window_opened if self.window_opened is not None else play.message
                if (play.message - opened < ONSIDE_MESSAGES
                        or play.field_position in CONVERSION_SPOTS):
                    return self._tick(play.message, SUSPENDED, self.window)
            return self._start_drive(play)

        # Inside a live drive.
        if play.at_kick_spot and (self.last_live is None
                                  or play.offense != self.last_live.offense):
            return self._tick(play.message, SUSPENDED, KICK_SPOT)
        if self.last_live is None or play.offense != self.last_live.offense:
            return self._start_drive(play)
        self.last_live = play
        return self._tick(play.message, LIVE, None)

    @staticmethod
    def _stale(play, previous):
        """A new team label on the old spot: the possession changed hands
        but the field has not been re-read from the new side yet."""
        return (play.offense != previous.offense
                and play.field_position == previous.field_position)

    @staticmethod
    def _impossible_down(play, previous):
        """The down jumped by two or more with the ball and distance still."""
        return (play.offense == previous.offense
                and play.field_position == previous.field_position
                and play.distance == previous.distance
                and play.down is not None and previous.down is not None
                and play.down > previous.down + 1)

    def _start_drive(self, play):
        if self.opening_receiver is None:
            self.opening_receiver = play.offense
        self.window = None
        self.scorer = None
        self.drive_start = play.message
        self.last_live = play
        return self._tick(play.message, LIVE, None)

    # -- state -------------------------------------------------------------

    def _tick(self, message, verdict, reason):
        self.verdict, self.reason = verdict, reason
        return Tick(message, verdict, reason, self.state(message))

    def state(self, message):
        if self.period is None:
            return None
        start = self.period_starts.get(self.period, message)
        half_first = 1 if self.period <= 2 else 3 if self.period <= 4 else self.period
        half_start = self.period_starts.get(half_first, start)
        snap = self.last_live if self.window is None else None
        return GameState(
            period=self.period,
            elapsed_in_period=max(0, message - start),
            elapsed_in_half=max(0, message - half_start),
            home_score=self.home,
            away_score=self.away,
            offense=snap.offense if snap else None,
            down=snap.down if snap else None,
            field_position=snap.field_position if snap else None,
            distance=snap.distance if snap else None,
            drive_age=(message - self.drive_start) if snap and self.drive_start is not None else None,
            opening_receiver=self.opening_receiver,
        )


def replay(plays, scores):
    """[Tick] for every message in the feed, in order.

    A score and a play on the same message: the score goes first, so the
    play is judged knowing the scoreboard just moved.
    """
    events = [(s.message, 0, s) for s in scores] + [(p.message, 1, p) for p in plays]
    events.sort(key=lambda e: (e[0], e[1]))
    tracker = Tracker()
    ticks = []
    for _, kind, event in events:
        tick = tracker.on_score(event) if kind == 0 else tracker.on_play(event)
        if ticks and ticks[-1].message == tick.message:
            # One verdict per message: a play after a score on the same
            # message decides it.
            ticks[-1] = tick
        else:
            ticks.append(tick)
    return ticks


def with_state(tick, **changes):
    return replace(tick, state=replace(tick.state, **changes))
