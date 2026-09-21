"""Write the drive-detection working out to CSV, one match at a time.

Reconciling drives means answering "why did this snapshot land here?", and
that is not a question a summary can answer -- it needs the row-by-row
working: every play the feed sent, which cleaning rule fired on it, which
drive it ended up in, and which single row became the snapshot.

Four files, joinable on (match_code, event_message_count):

  plays       every raw play row, with the cleaning verdict and the drive
              it was assigned to
  scores      every score change, so the play rows can be read against
              what the scoreboard did
  drives      one row per detected drive, with its anchor and buckets
  snapshots   what the calibrator actually pairs on

Deliberately the pipeline's own functions rather than a private copy, so
what is inspected here is what the report is built from.
"""

import csv
import os

from . import buckets, config, drives, markets, snowflake_io
from .drives import PlayRow, ScoreRow, build_snapshots, classify_plays, score_at

PLAY_FIELDS = [
    "match_code", "event_message_count", "period_number", "offensive_team",
    "down_number", "distance", "field_position", "is_snap",
    "cleaning", "dropped", "drive_number", "is_anchor",
]

SCORE_FIELDS = [
    "match_code", "event_message_count", "period_number",
    "player_1_change", "player_2_change",
    "player_1_cumulative", "player_2_cumulative",
    "is_touchdown", "scorer",
]

DRIVE_FIELDS = [
    "match_code", "drive_number", "offensive_team",
    "first_message", "last_message", "n_plays", "n_dropped_inside",
    "anchor_message", "anchor_kind", "down_number", "distance",
    "field_position", "score_p1", "score_p2", "score_diff",
    "score_bucket", "time_bucket", "possession_bucket",
]


def _rows_for_match(match_code, play_rows, score_rows):
    """Everything one match contributes to the four files."""
    plays = [
        PlayRow(event_message_count=r[1], period_number=r[2], offensive_team=r[3],
                down_number=r[4], distance=r[5], field_position=r[6], play_time=r[7])
        for r in play_rows
    ]
    scores = [
        ScoreRow(event_message_count=r[1], period_number=r[2], p1_change=r[3],
                 p2_change=r[4], p1_cumulative=r[5], p2_cumulative=r[6])
        for r in score_rows
    ]
    plays.sort(key=lambda p: p.event_message_count)
    scores.sort(key=lambda s: s.event_message_count)

    reasons = classify_plays(plays, scores)
    snapshots = build_snapshots(match_code, plays, scores)
    anchors = {s.event_message_count for s in snapshots}

    # Re-derive the drive each surviving play landed in, the same way
    # build_snapshots groups them: a maximal run of one offensive team.
    drive_of = {}
    drive_number = 0
    previous_team = None
    for play in plays:
        if drives.was_dropped(reasons[play.event_message_count]):
            continue
        if play.offensive_team != previous_team:
            drive_number += 1
            previous_team = play.offensive_team
        drive_of[play.event_message_count] = drive_number

    play_out = []
    for play in plays:
        reason = reasons[play.event_message_count]
        play_out.append({
            "match_code": match_code,
            "event_message_count": play.event_message_count,
            "period_number": play.period_number,
            "offensive_team": play.offensive_team,
            "down_number": play.down_number,
            "distance": play.distance,
            "field_position": play.field_position,
            "is_snap": int(drives.is_snap(play)),
            "cleaning": reason,
            "dropped": int(drives.was_dropped(reason)),
            "drive_number": drive_of.get(play.event_message_count, ""),
            "is_anchor": int(play.event_message_count in anchors),
        })

    touchdowns = {msg: team for msg, team in
                  drives._touchdown_scorers(scores).items()}
    score_out = []
    for row in scores:
        score_out.append({
            "match_code": match_code,
            "event_message_count": row.event_message_count,
            "period_number": row.period_number,
            "player_1_change": row.p1_change,
            "player_2_change": row.p2_change,
            "player_1_cumulative": row.p1_cumulative,
            "player_2_cumulative": row.p2_cumulative,
            "is_touchdown": int(row.event_message_count in touchdowns),
            "scorer": touchdowns.get(row.event_message_count, ""),
        })

    by_drive = {}
    for play in plays:
        number = drive_of.get(play.event_message_count)
        if number is not None:
            by_drive.setdefault(number, []).append(play)

    drive_out = []
    for snapshot in snapshots:
        run = by_drive.get(snapshot.drive_number, [])
        first = run[0].event_message_count if run else snapshot.event_message_count
        last = run[-1].event_message_count if run else snapshot.event_message_count
        inside = sum(1 for p in plays
                     if first <= p.event_message_count <= last
                     and drives.was_dropped(reasons[p.event_message_count]))
        drive_out.append({
            "match_code": match_code,
            "drive_number": snapshot.drive_number,
            "offensive_team": snapshot.offensive_team,
            "first_message": first,
            "last_message": last,
            "n_plays": snapshot.n_plays,
            "n_dropped_inside": inside,
            "anchor_message": snapshot.event_message_count,
            "anchor_kind": snapshot.anchor,
            "down_number": snapshot.down_number,
            "distance": snapshot.distance,
            "field_position": snapshot.field_position,
            "score_p1": snapshot.score_p1,
            "score_p2": snapshot.score_p2,
            "score_diff": snapshot.score_diff,
            "score_bucket": buckets.score_diff_bucket(snapshot.score_diff),
            "time_bucket": buckets.time_bucket(snapshot.period_number,
                                               snapshot.drive_number),
            "possession_bucket": buckets.possession_bucket(
                snapshot.offensive_team),
        })
    return play_out, score_out, drive_out


def _write(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(cur, match_codes, out_dir, time_column, verbose=True):
    """Dump the drive working for these matches. Returns the paths written."""
    play_rows = snowflake_io.fetch_plays(cur, match_codes, time_column)
    score_rows = snowflake_io.fetch_scores(cur, match_codes)

    plays_by_match, scores_by_match = {}, {}
    for row in play_rows:
        plays_by_match.setdefault(row[0], []).append(row)
    for row in score_rows:
        scores_by_match.setdefault(row[0], []).append(row)

    all_plays, all_scores, all_drives = [], [], []
    for match_code in match_codes:
        p, s, d = _rows_for_match(match_code,
                                  plays_by_match.get(match_code, []),
                                  scores_by_match.get(match_code, []))
        all_plays.extend(p)
        all_scores.extend(s)
        all_drives.extend(d)
        if verbose:
            kept = sum(1 for row in p if not row["dropped"])
            off = sum(1 for row in d if row["anchor_kind"] != drives.FIRST_DOWN)
            print(f"  {match_code}: {len(p):,} plays ({kept:,} kept), "
                  f"{len(d)} drives, {off} off anchor, {len(s)} score changes")

    written = [
        _write(os.path.join(out_dir, "dump_plays.csv"), PLAY_FIELDS, all_plays),
        _write(os.path.join(out_dir, "dump_scores.csv"), SCORE_FIELDS, all_scores),
        _write(os.path.join(out_dir, "dump_drives.csv"), DRIVE_FIELDS, all_drives),
    ]
    return written, all_plays, all_scores, all_drives
