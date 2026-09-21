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
DRIVE_START = "drive_start"      # a fresh 1st-and-10 that begins a possession
DUPLICATE = "duplicate"          # an exact republish of the row before it
KICKOFF = "kickoff"              # a 1st-and-10 that cannot be a real snap
SPECIAL_TEAMS = "special_teams"  # same down and distance, ball moved: a PAT
                                 # or a kick, not a scrimmage play
STALE_AFTER_CHANGE = "stale_after_change"   # a new team label on the old
                                            # team's down, distance and field

FIRST_DOWN_YARDS = 10

_DROPPED = (DUPLICATE, KICKOFF, SPECIAL_TEAMS, STALE_AFTER_CHANGE)


def _state(play):
    return (play.offensive_team, play.down_number, play.distance,
            play.field_position)


def _is_fresh_first_down(play):
    return play.down_number == 1 and play.distance == FIRST_DOWN_YARDS


def dedupe(plays):
    """Collapse runs of identical rows, keeping the first of each.

    The feed republishes a play's state until it changes, so almost every
    row arrives twice. They carry no information the first did not, and
    leaving them in doubles every play count and hides the structure.
    """
    out = []
    for play in plays:
        if out and _state(play) == _state(out[-1]):
            continue
        out.append(play)
    return out


def classify_plays(plays, scores=()):
    """message -> why that row was kept or dropped.

    A single forward pass over the deduplicated rows, each judged against
    the last row that survived. The rules come from what the feed actually
    does between one scrimmage play and the next.

      DUPLICATE       an exact republish.
      STALE           the team label changed onto the previous team's
                      down, distance and field, unchanged.
      SPECIAL_TEAMS   down and distance unchanged while the ball moved. A
                      scrimmage play always changes one or the other, so
                      this is a PAT or a kick -- the extra point taken
                      from the 85 with the touchdown's 3rd-and-5 still on
                      it, in the sequence that prompted this.
      KICKOFF         a 1st-and-10 that cannot be a snap: either the ball
                      went BACKWARD to reach it, which no first down does,
                      or the very next row is another 1st-and-10 for the
                      same team without the ten yards that would earn it.
                      That is the kick spot followed by the real start.

    Everything else is play. A fresh 1st-and-10 whose predecessor was a
    different team, or was dropped as a kick, begins a drive.

    `scores` is accepted and unused: the rules above read the play feed
    alone, which keeps drive detection independent of the PLAYER_1 frame
    rather than resting on it.
    """
    ordered = sorted(plays, key=lambda p: p.event_message_count)
    kept_rows = dedupe(ordered)
    keep_msgs = {p.event_message_count for p in kept_rows}

    reasons = {p.event_message_count:
               (KEPT if p.event_message_count in keep_msgs else DUPLICATE)
               for p in ordered}

    previous = None          # the last row that survived
    for i, play in enumerate(kept_rows):
        # Two predecessors matter and they are not the same row. The stale
        # duplicate mirrors whatever came immediately before it, kept or
        # not -- it rides the kick spot as readily as a real play -- while
        # every other rule compares against the last row that survived.
        before = kept_rows[i - 1] if i else None
        following = kept_rows[i + 1] if i + 1 < len(kept_rows) else None
        reasons[play.event_message_count] = _judge(play, previous, before,
                                                   following)
        if reasons[play.event_message_count] not in _DROPPED:
            previous = play
    return reasons


def _judge(play, previous, before, following):
    """Which rule this row falls under, given its neighbours."""
    fresh = _is_fresh_first_down(play)

    if not is_snap(play):
        # No readable down or distance at all: a kick mechanic rather than
        # a play. Rarer than the 1st-and-10-at-the-kick-spot shape, but it
        # occurs, and it is not scrimmage either way.
        return KICKOFF

    if (before is not None
            and play.offensive_team != before.offensive_team
            and _state(play)[1:] == _state(before)[1:]):
        return STALE_AFTER_CHANGE

    if previous is None:
        # The first row of a match is the opening kick whenever the row
        # after it is the same team's real 1st-and-10.
        if fresh and _looks_like_a_kick_spot(play, following):
            return KICKOFF
        return DRIVE_START if fresh else KEPT

    if play.offensive_team != previous.offensive_team:
        return DRIVE_START if fresh else KEPT

    known = (play.field_position is not None
             and previous.field_position is not None)
    moved = known and play.field_position != previous.field_position
    advance = (play.field_position - previous.field_position) if known else 0
    if (moved and play.down_number == previous.down_number
            and play.distance == previous.distance
            and not (fresh and advance >= FIRST_DOWN_YARDS)):
        # Down and distance unchanged while the ball moved. A scrimmage
        # play always changes one or the other -- the exception being a
        # 1st-and-10 converted into another, which needs the ten yards.
        return SPECIAL_TEAMS

    if fresh:
        went_backward = moved and advance < 0
        if went_backward or _looks_like_a_kick_spot(play, following):
            return KICKOFF
        # A fresh 1st-and-10 for the same team after a kick was dropped is
        # the drive that kick produced.
        return KEPT
    return KEPT


def _looks_like_a_kick_spot(play, following):
    """Is the row after this one the same team's real 1st-and-10?

    Two fresh 1st-and-10s in a row for one team are only both real if the
    ball advanced the ten yards that earn the second. Without them, the
    first row is the spot the kick was taken from and the second is where
    the drive actually starts.
    """
    if following is None or not _is_fresh_first_down(following):
        return False
    if following.offensive_team != play.offensive_team:
        return False
    if play.field_position is None or following.field_position is None:
        return True
    return (following.field_position - play.field_position) < FIRST_DOWN_YARDS


def was_dropped(reason):
    return reason in _DROPPED


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
