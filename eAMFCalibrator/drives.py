"""Turn the vision-derived play feed into drives, and each drive into a
snapshot of the game state at its first play.

The cleaning is carried over unchanged from
analysis/clean_possession_sequence.py, which was validated by inspection
against sample matches. Two rules:

  1. Around a touchdown, football's fixed sequence (TD -> PAT -> kickoff ->
     the other team's offense) means the next team with the ball is always
     the opponent of whoever scored. Everything between the TD and that
     team's next fresh 1st-and-10 is vision noise and gets dropped.
  2. The first play row after ANY team-label change duplicates the previous
     team's down/distance/field position, so it is dropped too -- except a
     TD-anchored resume row, which rule 1 already verified as genuine.

Drive numbering is 1-based within the match and counts cleaned drives
only, so it stays stable against the noise rows. Reconciling that number
against the real drive count is still outstanding, which is why
config.TIME_AXIS defaults to quarter rather than drive.

One trap worth naming, because it looks like a clean bill of health. A
drive here is DEFINED as a maximal run of the same offensive team, so
consecutive drives always belong to opposite teams and possession
"alternates" 100% of the time. That is arithmetic, not a measurement --
it would read 100% on a feed of pure noise. The real number is the count:
about 10 drives per match against a realistic 22ish, so the team label
changes roughly half as often as possession actually does, and every
missed change silently merges two possessions into one. Anything that
reasons from "the next drive belongs to the other team" inherits that
error, which is what sank the possession cross-check.
"""

from dataclasses import dataclass
from typing import Optional

from . import config

TOUCHDOWN_POINTS = 6
HOME_TEAM = "Home Team"
AWAY_TEAM = "Away Team"

# Which play in a drive the snapshot was taken from.
FIRST_DOWN = "first_down"   # the drive's opening 1st-and-10, as intended
MID_DRIVE = "mid_drive"     # a real snap, but not 1st-and-10: start lost
NO_SNAP = "no_snap"         # no plausible snap in the run at all


@dataclass(frozen=True)
class PlayRow:
    event_message_count: int
    period_number: Optional[int]
    offensive_team: Optional[str]
    down_number: Optional[int]
    distance: Optional[int]
    field_position: Optional[int]
    play_time: object = None  # datetime, or None when the feed has no clock


@dataclass(frozen=True)
class ScoreRow:
    event_message_count: int
    period_number: Optional[int]
    p1_change: Optional[int]
    p2_change: Optional[int]
    p1_cumulative: Optional[int]
    p2_cumulative: Optional[int]


@dataclass(frozen=True)
class Snapshot:
    match_code: str
    drive_number: int
    event_message_count: int
    period_number: Optional[int]
    offensive_team: Optional[str]
    field_position: Optional[int]
    down_number: Optional[int]
    distance: Optional[int]
    play_time: object
    score_p1: int
    score_p2: int
    n_plays: int
    # How the snapshot's play was chosen. FIRST_DOWN means the drive's
    # opening 1st-and-10, which is what a drive start actually is.
    anchor: str = FIRST_DOWN

    @property
    def score_diff(self):
        """Home minus away, the frame every score bucket is read in."""
        return self.score_p1 - self.score_p2


def _touchdown_scorers(scores):
    """message count -> which side scored a 6-point touchdown."""
    out = {}
    for s in scores:
        delta = s.p1_change if s.p1_change else s.p2_change
        if delta == TOUCHDOWN_POINTS:
            out[s.event_message_count] = HOME_TEAM if s.p1_change else AWAY_TEAM
    return out


# Why a play row was kept or dropped. The reason is as much of the output
# as the decision: reconciling drive detection by eye means seeing which
# rule fired on which row, not just which rows survived.
KEPT = "kept"
DRIVE_START = "drive_start"    # kept, and the kickoff rule vouched for it
KICKOFF = "kickoff"            # a kick row: between a kickoff and the
                               # receiving team's first fresh 1st-and-10
STALE_AFTER_CHANGE = "stale_after_change"   # repeats the previous team's
                                            # down, distance and field


def _kickoff_triggers(plays, scores):
    """Message counts at or after which a kickoff happens.

    Three of them, and only the last was handled before:

      the opening kickoff, which is simply the start of the match
      the second-half kickoff, and any other period that starts with one
      every score -- not only the 6-point touchdowns

    Field goals and safeties are followed by a kick too, so restricting
    this to touchdowns left most kicks in the feed.
    """
    triggers = set()
    if plays:
        triggers.add(plays[0].event_message_count - 1)

    previous_period = None
    for play in plays:
        period = play.period_number
        if (period != previous_period and period in config.KICKOFF_PERIODS):
            triggers.add(play.event_message_count - 1)
        previous_period = period

    for row in scores:
        if row.p1_change or row.p2_change:
            triggers.add(row.event_message_count)
    return sorted(triggers)


def _is_stale_duplicate(play, previous):
    """The documented noise row: a new team label on the old team's state.

    It repeats the previous row's down, distance and field position
    exactly, which is what distinguishes it from a genuine snap -- and it
    has to be told apart, because it looks like a legitimate play and
    would otherwise stop a kickoff walk dead.
    """
    if previous is None or play.offensive_team == previous.offensive_team:
        return False
    return (play.down_number == previous.down_number
            and play.distance == previous.distance
            and play.field_position == previous.field_position)


def _drive_start_after(plays, trigger):
    """The first real 1st-and-10 after the kick that follows `trigger`.

    A kickoff shows the KICKING team first and the receiving team second,
    both sitting at the kick spot, so neither the team label changing nor
    the ball being on a particular yard line marks the drive. The kick spot
    is a fixed line but a return can legitimately finish on it, so field
    position both misses real drives and invents others.

    Down and distance are what separate a kick from a snap. The walk skips
    rows that are not snaps, skips the stale duplicate that rides every
    team change, and stops at the first fresh 1st-and-10. That works
    whether or not the feed shows the kick rows at all, which matters:
    keying on the team change would lose the drive entirely on a
    transition where they are absent.

    Hitting a real snap that is neither stale nor a fresh 1st-and-10 means
    play was already under way and there is no kick here to strip, so the
    walk gives up rather than delete a genuine drive.
    """
    indexed = list(enumerate(plays))
    window = [(i, p) for i, p in indexed
              if p.event_message_count > trigger][:config.KICKOFF_SEARCH_LIMIT]
    for i, play in window:
        previous = plays[i - 1] if i else None
        if _is_stale_duplicate(play, previous):
            continue
        if not is_snap(play):
            continue
        if play.down_number == 1 and play.distance == 10:
            return play.event_message_count
        return None
    return None


def classify_plays(plays, scores):
    """message -> why that row was kept or dropped.

    The single source of truth for the cleaning, so the dump and the
    pipeline can never disagree about what happened to a row.
    """
    plays = sorted(plays, key=lambda p: p.event_message_count)
    scores = sorted(scores, key=lambda s: s.event_message_count)
    reasons = {p.event_message_count: KEPT for p in plays}
    starts = set()

    for trigger in _kickoff_triggers(plays, scores):
        start = _drive_start_after(plays, trigger)
        if start is None:
            continue
        starts.add(start)
        reasons[start] = DRIVE_START
        for play in plays:
            if trigger < play.event_message_count < start:
                reasons[play.event_message_count] = KICKOFF

    # What is left are real possession changes -- a punt, a turnover, a
    # turnover on downs -- where the first row after the change repeats the
    # previous team's down, distance and field position.
    previous_team = None
    for play in plays:
        already = reasons[play.event_message_count]
        if (previous_team is not None
                and play.offensive_team != previous_team
                and already == KEPT):
            # Only a row the kickoff rule had no opinion about. Inside a
            # kickoff the team label changes from the kicking side to the
            # receiving side, and calling that row stale would both hide
            # what it is and claim a possession change that never happened.
            reasons[play.event_message_count] = STALE_AFTER_CHANGE
        previous_team = play.offensive_team
    return reasons


def was_dropped(reason):
    return reason in (KICKOFF, STALE_AFTER_CHANGE)


def clean_plays(plays, scores):
    """Drop vision-recognition noise. Returns (cleaned_plays, n_dropped)."""
    reasons = classify_plays(plays, scores)
    cleaned = [p for p in plays
               if not was_dropped(reasons[p.event_message_count])]
    return cleaned, len(plays) - len(cleaned)


def is_snap(play):
    """Whether a row is a real scrimmage play.

    Kick mechanics and stale duplicates carry a down outside 1-4 or a
    distance no offence ever faces, which is what separates them from a
    snap without having to know what the feed calls them.
    """
    if play.down_number not in (1, 2, 3, 4):
        return False
    return (play.distance is not None
            and 0 <= play.distance <= config.MAX_PLAUSIBLE_DISTANCE)


def score_at(scores, event_message_count):
    """Cumulative score as of a message count: the last score change at or
    before it, or 0-0 if the match has not scored yet."""
    p1, p2 = 0, 0
    for s in scores:
        if s.event_message_count > event_message_count:
            break
        if s.p1_cumulative is not None:
            p1 = s.p1_cumulative
        if s.p2_cumulative is not None:
            p2 = s.p2_cumulative
    return p1, p2


def build_snapshots(match_code, plays, scores):
    """One snapshot per cleaned drive, taken at the drive's first play."""
    plays = sorted(plays, key=lambda p: p.event_message_count)
    scores = sorted(scores, key=lambda s: s.event_message_count)

    reasons = classify_plays(plays, scores)
    cleaned = [p for p in plays
               if not was_dropped(reasons[p.event_message_count])]
    if not cleaned:
        return []

    snapshots = []
    drive_number = 0
    current_run = []

    def flush():
        if not current_run:
            return
        # A drive starts at 1st and 10. Taking the run's first ROW instead
        # lands on whatever survived cleaning: the general rule drops only
        # ONE row per team change, so a transition carrying several
        # kickoff-mechanic rows leaves the rest behind, and the snapshot
        # sits on a row whose down, distance and team label all belong to
        # the kick rather than the drive.
        #
        # So the anchor walks forward to the first row that is a plausible
        # SNAP, and then asks whether that snap is 1st and 10. Walking to
        # the first 1st-and-10 instead would be wrong: a drive whose run
        # opens 2nd and 7 has already lost its start, and the next
        # 1st-and-10 in it is a first-down CONVERSION -- a real game state,
        # but not this drive's. Stopping at the first real snap keeps that
        # case visible as MID_DRIVE rather than silently relabelling it.
        anchor = next((p for p in current_run if is_snap(p)), None)
        if anchor is None:
            anchor, kind = current_run[0], NO_SNAP
        elif anchor.down_number == 1 and anchor.distance == 10:
            kind = FIRST_DOWN
        else:
            kind = MID_DRIVE
        p1, p2 = score_at(scores, anchor.event_message_count)
        snapshots.append(Snapshot(
            match_code=match_code,
            drive_number=drive_number,
            event_message_count=anchor.event_message_count,
            period_number=anchor.period_number,
            offensive_team=anchor.offensive_team,
            field_position=anchor.field_position,
            down_number=anchor.down_number,
            distance=anchor.distance,
            play_time=anchor.play_time,
            score_p1=p1,
            score_p2=p2,
            n_plays=len(current_run),
            anchor=kind,
        ))

    for p in cleaned:
        # A team change ends a drive, and so does a row the kickoff rule
        # positively identified as a drive start -- otherwise the side that
        # had the ball before half time and receives after it would have
        # both possessions merged into one, the boundary invisible because
        # the label never changed.
        starts_drive = (not current_run
                        or p.offensive_team != current_run[0].offensive_team
                        or reasons[p.event_message_count] == DRIVE_START)
        if starts_drive:
            flush()
            drive_number += 1
            current_run = []
        current_run.append(p)
    flush()

    return snapshots
