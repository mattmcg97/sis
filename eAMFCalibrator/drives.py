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
RESUME = "resume_after_td"          # kept, and the TD rule vouched for it
NOISE_AFTER_TD = "noise_after_td"   # between a TD and the receiving team's
                                    # first fresh 1st-and-10
STALE_AFTER_CHANGE = "stale_after_change"   # repeats the previous team's
                                            # down, distance and field


def classify_plays(plays, scores):
    """message -> why that row was kept or dropped.

    The single source of truth for the cleaning, so the dump and the
    pipeline can never disagree about what happened to a row.
    """
    reasons = {p.event_message_count: KEPT for p in plays}
    td_scorer_team = _touchdown_scorers(scores)
    resume_msgs = set()

    for td_msg, scoring_team in sorted(td_scorer_team.items()):
        receiving_team = AWAY_TEAM if scoring_team == HOME_TEAM else HOME_TEAM
        resume_msg = None
        for p in plays:
            if p.event_message_count <= td_msg:
                continue
            if p.offensive_team == receiving_team and p.down_number == 1 and p.distance == 10:
                resume_msg = p.event_message_count
                break
        if resume_msg is None:
            continue
        resume_msgs.add(resume_msg)
        reasons[resume_msg] = RESUME
        for p in plays:
            if td_msg < p.event_message_count < resume_msg:
                reasons[p.event_message_count] = NOISE_AFTER_TD

    prev_team = None
    for p in plays:
        if (prev_team is not None
                and p.offensive_team != prev_team
                and p.event_message_count not in resume_msgs):
            reasons[p.event_message_count] = STALE_AFTER_CHANGE
        prev_team = p.offensive_team
    return reasons


def was_dropped(reason):
    return reason in (NOISE_AFTER_TD, STALE_AFTER_CHANGE)


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

    cleaned, _ = clean_plays(plays, scores)
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
        if not current_run or p.offensive_team != current_run[0].offensive_team:
            flush()
            drive_number += 1
            current_run = []
        current_run.append(p)
    flush()

    return snapshots
