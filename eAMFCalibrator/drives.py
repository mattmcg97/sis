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
"""

from dataclasses import dataclass
from typing import Optional

TOUCHDOWN_POINTS = 6
HOME_TEAM = "Home Team"
AWAY_TEAM = "Away Team"


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


def clean_plays(plays, scores):
    """Drop vision-recognition noise. Returns (cleaned_plays, n_dropped)."""
    td_scorer_team = _touchdown_scorers(scores)

    noise_msgs = set()
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
        for p in plays:
            if td_msg < p.event_message_count < resume_msg:
                noise_msgs.add(p.event_message_count)

    prev_team = None
    for p in plays:
        if (prev_team is not None
                and p.offensive_team != prev_team
                and p.event_message_count not in resume_msgs):
            noise_msgs.add(p.event_message_count)
        prev_team = p.offensive_team

    cleaned = [p for p in plays if p.event_message_count not in noise_msgs]
    return cleaned, len(noise_msgs)


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
    current_first = None
    current_count = 0

    def flush():
        if current_first is None:
            return
        p1, p2 = score_at(scores, current_first.event_message_count)
        snapshots.append(Snapshot(
            match_code=match_code,
            drive_number=drive_number,
            event_message_count=current_first.event_message_count,
            period_number=current_first.period_number,
            offensive_team=current_first.offensive_team,
            field_position=current_first.field_position,
            down_number=current_first.down_number,
            distance=current_first.distance,
            play_time=current_first.play_time,
            score_p1=p1,
            score_p2=p2,
            n_plays=current_count,
        ))

    for p in cleaned:
        if current_first is None or p.offensive_team != current_first.offensive_team:
            flush()
            drive_number += 1
            current_first = p
            current_count = 0
        current_count += 1
    flush()

    return snapshots
