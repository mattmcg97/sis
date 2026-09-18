"""The axes a calibration run is split along.

Three dimensions, each independent of the others:

  score difference   PLAYER_1 minus PLAYER_2 at the snapshot, bucketed
  time axis          quarter today; drive number once drives reconcile
  possession         which side has the ball at the snapshot

The time axis is switchable in config rather than hardcoded, so moving
from quarter to drive number later is a one-line change, not a rewrite.
"""

from . import config

HOME = "Home"
AWAY = "Away"

# INPLAY_FIELD_POSITION_PERIOD labels offense this way; normalised here so
# nothing downstream has to know the feed's spelling.
OFFENSIVE_TEAM_TO_SIDE = {
    "Home Team": HOME,
    "Away Team": AWAY,
}


def score_diff_bucket(diff):
    """Bucket a home-minus-away score difference."""
    if diff is None:
        return "unknown"
    for low, high, label in config.SCORE_DIFF_BUCKETS:
        if (low is None or diff >= low) and (high is None or diff <= high):
            return label
    return "unknown"


def period_bucket(period):
    """Quarters 1-4 named; anything beyond pooled as OT."""
    if period is None:
        return "unknown"
    if 1 <= period <= 4:
        return f"Q{period}"
    return "OT"


def drive_bucket(drive_number):
    if drive_number is None:
        return "unknown"
    for low, high, label in config.DRIVE_BUCKETS:
        if (low is None or drive_number >= low) and (high is None or drive_number <= high):
            return label
    return "unknown"


def time_bucket(period, drive_number):
    """Whichever time axis config selects."""
    if config.TIME_AXIS == "drive":
        return drive_bucket(drive_number)
    return period_bucket(period)


def time_axis_name():
    return "DRIVE" if config.TIME_AXIS == "drive" else "QUARTER"


def possession_bucket(offensive_team):
    """Which side has the ball, as a plain Home/Away flag."""
    if offensive_team is None:
        return "unknown"
    return OFFENSIVE_TEAM_TO_SIDE.get(offensive_team, "unknown")


def bucket_key(snapshot):
    """The full cell a snapshot lands in, before market/selection split."""
    return (
        score_diff_bucket(snapshot.score_diff),
        time_bucket(snapshot.period_number, snapshot.drive_number),
        possession_bucket(snapshot.offensive_team),
    )


def sort_key(cell):
    """Stable, readable ordering for printed tables."""
    score_order = [label for _, _, label in config.SCORE_DIFF_BUCKETS] + ["unknown"]
    if config.TIME_AXIS == "drive":
        time_order = [label for _, _, label in config.DRIVE_BUCKETS] + ["unknown"]
    else:
        time_order = ["Q1", "Q2", "Q3", "Q4", "OT", "unknown"]
    poss_order = [HOME, AWAY, "unknown"]

    def index(seq, value):
        return seq.index(value) if value in seq else len(seq)

    score, time_b, poss = cell[0], cell[1], cell[2]
    return (index(score_order, score), index(time_order, time_b), index(poss_order, poss)) + tuple(cell[3:])
