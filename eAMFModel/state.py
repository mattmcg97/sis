"""The game state a version prices from, and the market ids GAMEPLAI publishes."""

from dataclasses import dataclass
from typing import Optional

HOME = "home"
AWAY = "away"

ML_HOME, ML_AWAY, SPREAD_HOME, SPREAD_AWAY, TOTAL_OVER, TOTAL_UNDER = 50, 51, 52, 53, 54, 55
MARKET_IDS = (ML_HOME, ML_AWAY, SPREAD_HOME, SPREAD_AWAY, TOTAL_OVER, TOTAL_UNDER)


@dataclass(frozen=True)
class GameState:
    """Where a game stands at a PLAY_OVER snapshot."""
    period: Optional[int]
    home_score: int
    away_score: int
    offense: Optional[str] = None            # HOME, AWAY, or None when not known
    down: Optional[int] = None
    field_position: Optional[int] = None     # yards from the offense's own goal line
    distance: Optional[int] = None
    clock_seconds: Optional[float] = None    # game clock left in the quarter
    pending_conversion: Optional[str] = None # scored a TD, the conversion still to come
    snap_confirmed: bool = False             # the feed says this is a snap
    opening_receiver: Optional[str] = None   # who had the ball first

    @property
    def has_snap(self):
        """Whether the state is a snap: a side on the ball with its down, distance and field."""
        return (self.offense in (HOME, AWAY) and self.down is not None
                and self.field_position is not None and self.distance is not None
                and 1 <= self.down <= 4 and 0 < self.field_position < 100
                and self.distance > 0)
