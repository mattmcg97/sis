"""A game to click through, play by play, for the trader: where it stands and what each play does
to it, under the rules the simulation plays (4-minute quarters, three timeouts a side each half,
the second half kicked to the side that kicked the first, overtime off a kick-off with two
timeouts a side each period)."""

from dataclasses import asdict, dataclass, field as _default, fields
from types import SimpleNamespace

QUARTER = 240.0
KICKOFF, SCRIMMAGE, CONVERSION, FINAL = "kickoff", "scrimmage", "conversion", "final"
PHASES = (KICKOFF, SCRIMMAGE, CONVERSION, FINAL)
HOME, AWAY = "home", "away"
TOUCHBACK = 25                  # a kick-off's touchback
PUNT_TOUCHBACK = 20
MISSED_FG_MIN = 20              # a missed field goal comes back to the spot of the kick, or the 20
KICK_BEHIND = 7                 # the kick is taken this far behind the line of scrimmage
TIMEOUTS = 3
OT_TIMEOUTS = 2
MAX_OT = 3


def other(side):
    return AWAY if side == HOME else HOME


@dataclass
class Game:
    """Where a game stands. `side` is the side with the ball at a scrimmage snap, the side
    receiving at a kick-off, and the side that scored at a conversion; `field` is yards from that
    side's own goal line."""
    period: int = 1
    clock: float = QUARTER
    home: int = 0
    away: int = 0
    phase: str = KICKOFF
    side: str = HOME
    down: int = 1
    distance: int = 10
    field: int = TOUCHBACK
    timeouts_home: int = TIMEOUTS
    timeouts_away: int = TIMEOUTS
    opening_receiver: str = HOME
    fresh: bool = True              # the first snap of a possession
    ot_had_ball: list = _default(default_factory=list)

    @classmethod
    def start(cls, opening_receiver=HOME):
        """The opening kick-off, to `opening_receiver`."""
        return cls(side=opening_receiver, opening_receiver=opening_receiver)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        g = cls(**{k: v for k, v in (d or {}).items() if k in names})
        g.check()
        return g

    def to_dict(self):
        return asdict(self)

    def check(self):
        """Stop on a state the rules cannot reach."""
        if self.phase not in PHASES or self.side not in (HOME, AWAY) \
                or self.opening_receiver not in (HOME, AWAY):
            raise ValueError(f"no such phase or side: {self.phase!r}, {self.side!r}")
        if not 1 <= self.period <= 4 + MAX_OT or not 0 <= self.clock <= QUARTER:
            raise ValueError("the period runs 1 to 4 and overtime; the clock 0:00 to 4:00")
        if self.phase == SCRIMMAGE and not (1 <= self.down <= 4 and 0 < self.field < 100
                                            and self.distance > 0):
            raise ValueError("a snap needs a down of 1 to 4, a distance and the ball inside the field")
        if min(self.home, self.away, self.timeouts_home, self.timeouts_away) < 0:
            raise ValueError("scores and timeouts cannot go below 0")

    def model_state(self):
        """The state as v10.start_from reads it (None when the game is over)."""
        if self.phase == FINAL:
            return None
        base = dict(period=self.period, clock_seconds=float(self.clock), home_score=self.home,
                    away_score=self.away, opening_receiver=self.opening_receiver,
                    timeouts=(self.timeouts_home, self.timeouts_away), pending_conversion=None,
                    down=None, distance=None, field_position=None, fresh=False)
        if self.phase == SCRIMMAGE:
            base.update(offense=self.side, down=self.down, field_position=self.field,
                        distance=max(1, min(self.distance, 100 - self.field)), fresh=self.fresh)
        elif self.phase == KICKOFF:
            base.update(offense=self.side)                 # the receiver; the other side kicks
        else:
            base.update(offense=other(self.side), pending_conversion=self.side)
        return SimpleNamespace(**base)

    def describe(self):
        """The scoreboard line: what is next."""
        if self.phase == FINAL:
            return "Final"
        if self.phase == KICKOFF:
            return f"Kick-off to {self.side}"
        if self.phase == CONVERSION:
            return f"{self.side.capitalize()} conversion"
        to_go = "goal" if self.field + self.distance >= 100 else str(self.distance)
        spot = f"own {self.field}" if self.field < 50 else ("50" if self.field == 50
                                                              else f"opp {100 - self.field}")
        return f"{self.side.capitalize()} {_ordinal(self.down)} & {to_go} at {spot}"


def _ordinal(n):
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(n, f"{n}th")


def _num(play, key, default):
    v = play.get(key)
    return default if v in (None, "") else float(v)


def _score(g, side, points):
    if side == HOME:
        g.home += points
    else:
        g.away += points


def _ball(g, side, at):
    """`side` takes over at `at` yards from its own goal: 1st and 10 (or goal)."""
    g.phase, g.side = SCRIMMAGE, side
    g.field = min(99, max(1, int(round(at))))
    g.down, g.distance = 1, min(10, 100 - g.field)
    g.fresh = True


def _kick_to(g, receiver):
    g.phase, g.side, g.fresh = KICKOFF, receiver, True
    g.down, g.distance, g.field = 1, 10, TOUCHBACK


def _touchdown(g, side):
    _score(g, side, 6)
    g.phase, g.side = CONVERSION, side


def _advance(g, yards):
    """A scrimmage play gaining `yards`: a first down, the next down, a turnover on downs, a
    touchdown or a safety."""
    at = g.field + int(round(yards))
    if at >= 100:
        _touchdown(g, g.side)
        return "touchdown"
    if at <= 0:
        _score(g, other(g.side), 2)
        _kick_to(g, other(g.side))               # the side that gave up the safety kicks
        return "safety"
    if yards >= g.distance:
        g.field, g.down, g.distance = at, 1, min(10, 100 - at)
        g.fresh = False
        return "first down"
    if g.down == 4:
        _ball(g, other(g.side), 100 - at)
        return "turnover on downs"
    g.field, g.down, g.distance = at, g.down + 1, g.distance - int(round(yards))
    g.fresh = False
    return None


def _end_period(g):
    """The clock has run out: the next quarter, half time, overtime or the end."""
    if g.period in (1, 3):
        g.period, g.clock = g.period + 1, QUARTER
        return f"end of Q{g.period - 1}"
    if g.period == 2:
        g.period, g.clock = 3, QUARTER
        g.timeouts_home = g.timeouts_away = TIMEOUTS
        _kick_to(g, other(g.opening_receiver))
        return "half time"
    level = g.home == g.away
    if g.period == 4:
        if not level:
            g.phase = FINAL
            return "final"
        g.period, g.clock = 5, QUARTER
        g.timeouts_home = g.timeouts_away = OT_TIMEOUTS
        g.ot_had_ball = []
        _kick_to(g, HOME)
        return "overtime"
    behind = HOME if g.home < g.away else AWAY
    if (not level and behind in g.ot_had_ball) or g.period >= 4 + MAX_OT:
        g.phase = FINAL
        return "final"
    g.period, g.clock = g.period + 1, QUARTER
    g.timeouts_home = g.timeouts_away = OT_TIMEOUTS
    return f"end of OT{g.period - 5}"


def _tick(g, seconds):
    """Run the clock; a period that runs out ends once any conversion is taken."""
    g.clock = max(0.0, g.clock - max(0.0, seconds))
    if g.clock <= 0 and g.phase not in (CONVERSION, FINAL):
        return _end_period(g)
    return None


SCRIMMAGE_PLAYS = ("play", "incomplete", "touchdown", "field_goal", "punt", "turnover", "safety",
                   "kneel", "spike")
DEFAULT_SECONDS = {"play": 25, "incomplete": 6, "touchdown": 8, "field_goal": 5, "punt": 10,
                   "turnover": 8, "safety": 6, "kneel": 40, "spike": 3, "kickoff": 0}


def apply(game, play):
    """(the game after `play`, what happened). A play is a dict with a "type" and, as it needs
    them, "yards", "seconds", "at" (yards from the new side's own goal), "made", "touchdown",
    "kind" (a conversion's: xp, xp_miss, two, two_fail), "side" (a timeout's) or "state" (set)."""
    g = Game.from_dict(game.to_dict() if isinstance(game, Game) else game)
    kind = play.get("type")
    if kind == "set":
        g = Game.from_dict({**g.to_dict(), **(play.get("state") or {})})
        return g, "state set"
    if kind == "timeout":
        side = play.get("side")
        attr = f"timeouts_{side}"
        if side not in (HOME, AWAY) or getattr(g, attr) <= 0:
            raise ValueError(f"{side} has no timeouts left")
        setattr(g, attr, getattr(g, attr) - 1)
        return g, f"timeout {side}"
    if g.phase == FINAL:
        raise ValueError("the game is over")
    seconds = _num(play, "seconds", DEFAULT_SECONDS.get(kind, 0))
    if g.phase == KICKOFF:
        if kind != "kickoff":
            raise ValueError("a kick-off comes next")
        receiver = g.side
        if play.get("touchdown"):
            _touchdown(g, receiver)
            what = f"kick-off returned for a touchdown by {receiver}"
        else:
            at = _num(play, "at", TOUCHBACK)
            _ball(g, receiver, at)
            what = "touchback" if at == TOUCHBACK and not seconds else f"kick-off to the {int(at)}"
        return g, _join(what, _tick(g, seconds))
    if g.phase == CONVERSION:
        if kind != "conversion":
            raise ValueError("the conversion comes next")
        points = {"xp": 1, "xp_miss": 0, "two": 2, "two_fail": 0}.get(play.get("kind"))
        if points is None:
            raise ValueError("a conversion is xp, xp_miss, two or two_fail")
        scorer = g.side
        _score(g, scorer, points)
        _kick_to(g, other(scorer))
        what = f"{scorer} conversion {'good' if points else 'no good'}"
        end = _end_period(g) if g.clock <= 0 else None
        return g, _join(what, end)
    if kind not in SCRIMMAGE_PLAYS:
        raise ValueError(f"{kind!r} is not a scrimmage play")
    side = g.side
    if g.period >= 5 and side not in g.ot_had_ball:
        g.ot_had_ball = g.ot_had_ball + [side]
    if kind in ("play", "kneel"):
        yards = -1 if kind == "kneel" else _num(play, "yards", 0)
        result = _advance(g, yards)
        what = f"{side} {kind} {int(round(yards)):+d}" + (f", {result}" if result else "")
    elif kind in ("incomplete", "spike"):
        result = _advance(g, 0) if g.down == 4 else None
        if result is None:
            g.down += 1
            g.fresh = False
        what = f"{side} {kind}" + (f", {result}" if result else "")
    elif kind == "touchdown":
        _touchdown(g, side)
        what = f"{side} touchdown"
    elif kind == "safety":
        _advance(g, -g.field)
        what = f"safety, 2 to {other(side)}"
    elif kind == "field_goal":
        if play.get("made", True):
            _score(g, side, 3)
            _kick_to(g, other(side))
            what = f"{side} field goal good"
        else:
            _ball(g, other(side), max(MISSED_FG_MIN, 100 - (g.field - KICK_BEHIND)))
            what = f"{side} field goal missed"
    elif kind == "punt":
        lands = g.field + _num(play, "yards", 40)
        _ball(g, other(side), PUNT_TOUCHBACK if lands >= 100 else 100 - lands)
        what = f"{side} punt"
    else:                                                     # turnover
        if play.get("touchdown"):
            _touchdown(g, other(side))
            what = f"{side} turnover, returned for a touchdown"
        else:
            _ball(g, other(side), _num(play, "at", 100 - g.field))
            what = f"{side} turnover"
    return g, _join(what, _tick(g, seconds))


def _join(what, end):
    return what if not end else f"{what}; {end}"
