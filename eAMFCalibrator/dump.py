"""Write the drive-detection working out to CSV, two files.

Reconciling drives means answering "why did this snapshot land here?",
and that is not a question a summary can answer -- it needs the row-by-row
working. It also does not want answering across five files.

  play_by_play   every raw play row, with the cleaning verdict, the drive
                 it landed in, whether it became the snapshot, the score
                 as of that message, and what both streams were quoting
                 on all six selections at the time
  pairs          directional_pairs.csv for these matches, widened with
                 what the play-by-play knows about each row

Both are built from the pipeline's own functions, so what is inspected
here is what the report is made of.
"""

import collections
import csv
import os

from . import (buckets, config, directional, drives, markets, report,
               snowflake_io)
from .drives import PlayRow, ScoreRow, build_snapshots, classify_plays, score_at

# Market columns, three groups of two selections. Named for what they are
# rather than by ID, since the point of one wide table is to be read.
MARKET_COLUMNS = [
    ("ml_home", 50, False), ("ml_away", 51, False),
    ("sp_home", 52, True), ("sp_away", 53, True),
    ("tot_over", 54, True), ("tot_under", 55, True),
]


def _market_fields():
    out = []
    for name, _, lined in MARKET_COLUMNS:
        out += [f"{name}_prod", f"{name}_cand"]
        if lined:
            out += [f"{name}_line_prod", f"{name}_line_cand"]
        out.append(f"{name}_live")
    return out


PLAY_FIELDS = [
    "match_code", "event_message_count", "period_number", "offensive_team",
    "down_number", "distance", "field_position", "is_snap",
    "cleaning", "dropped", "drive_number", "is_anchor",
    "score_p1", "score_p2", "score_diff", "p1_change", "p2_change",
] + _market_fields()


def _market_cells(match_code, message, indexes):
    """What both streams were quoting on every selection at this message."""
    cells = {}
    for name, market_id, lined in MARKET_COLUMNS:
        quotes = {}
        for stream, index in indexes.items():
            quotes[stream] = index.get((match_code, market_id), {}).get(message)
        prod, candidate = quotes[directional.PROD], quotes[directional.CANDIDATE]
        cells[f"{name}_prod"] = "" if prod is None else prod.probability
        cells[f"{name}_cand"] = "" if candidate is None else candidate.probability
        if lined:
            cells[f"{name}_line_prod"] = (
                "" if prod is None else markets.parse_line(prod.description))
            cells[f"{name}_line_cand"] = (
                "" if candidate is None
                else markets.parse_line(candidate.description))
        if prod is None and candidate is None:
            cells[f"{name}_live"] = ""
        elif prod is not None and candidate is not None and prod.live and candidate.live:
            cells[f"{name}_live"] = "live"
        elif prod is None or candidate is None:
            cells[f"{name}_live"] = "missing"
        elif not prod.live and not candidate.live:
            cells[f"{name}_live"] = "both"
        else:
            cells[f"{name}_live"] = "prod" if not prod.live else "cand"
    return cells


def _rows_for_match(match_code, play_rows, score_rows, indexes=None):
    """One match's play-by-play, with its score and market context."""
    indexes = indexes or {directional.PROD: {}, directional.CANDIDATE: {}}
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
    changes = {row.event_message_count: row for row in scores}

    # Assign each surviving play to a drive by walking the snapshots the
    # pipeline produced, rather than re-deriving the grouping here.
    drive_of = {}
    boundaries = sorted((s.event_message_count, s.drive_number)
                        for s in snapshots)
    for play in plays:
        if drives.was_dropped(reasons[play.event_message_count]):
            continue
        current = None
        for message, number in boundaries:
            if play.event_message_count >= message:
                current = number
            else:
                break
        if current is not None:
            drive_of[play.event_message_count] = current

    out = []
    for play in plays:
        reason = reasons[play.event_message_count]
        p1, p2 = score_at(scores, play.event_message_count)
        change = changes.get(play.event_message_count)
        row = {
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
            "score_p1": p1,
            "score_p2": p2,
            "score_diff": p1 - p2,
            "p1_change": "" if change is None else (change.p1_change or ""),
            "p2_change": "" if change is None else (change.p2_change or ""),
        }
        row.update(_market_cells(match_code, play.event_message_count, indexes))
        out.append(row)

    # A score can land on a message with no play row of its own. Those are
    # what separate two drives the feed never relabelled, so they cannot be
    # left out of a play-by-play that is meant to explain the drives.
    play_msgs = {p.event_message_count for p in plays}
    for row in scores:
        if row.event_message_count in play_msgs:
            continue
        if not (row.p1_change or row.p2_change):
            continue
        p1, p2 = score_at(scores, row.event_message_count)
        extra = {
            "match_code": match_code,
            "event_message_count": row.event_message_count,
            "period_number": row.period_number,
            "offensive_team": "", "down_number": "", "distance": "",
            "field_position": "", "is_snap": "", "cleaning": "score",
            "dropped": "", "drive_number": "", "is_anchor": 0,
            "score_p1": p1, "score_p2": p2, "score_diff": p1 - p2,
            "p1_change": row.p1_change or "", "p2_change": row.p2_change or "",
        }
        extra.update(_market_cells(match_code, row.event_message_count, indexes))
        out.append(extra)

    out.sort(key=lambda r: r["event_message_count"])
    return out, snapshots


def _drive_rows(match_code, play_out, snapshots):
    """One row per drive, for the console summary.

    Not written out any more -- the play-by-play carries the drive number
    on every row, so a separate drives file was one more table to flip
    through. The summary still wants the per-drive shape, so it is built
    here from the same rows the CSV holds.
    """
    out = []
    by_number = {}
    for row in play_out:
        if row["drive_number"] == "":
            continue
        by_number.setdefault(row["drive_number"], []).append(row)

    for snapshot in snapshots:
        kept = by_number.get(snapshot.drive_number, [])
        if not kept:
            continue
        first = kept[0]["event_message_count"]
        last = kept[-1]["event_message_count"]
        dropped = sum(1 for row in play_out
                      if row["dropped"] == 1
                      and first <= row["event_message_count"] <= last)
        out.append({
            "match_code": match_code,
            "drive_number": snapshot.drive_number,
            "offensive_team": snapshot.offensive_team,
            "first_message": first,
            "last_message": last,
            "n_plays": len(kept),
            "n_dropped_inside": dropped,
            "anchor_message": snapshot.event_message_count,
            "anchor_kind": snapshot.anchor,
            "down_number": snapshot.down_number,
            "distance": snapshot.distance,
            "field_position": snapshot.field_position,
            "score_p1": snapshot.score_p1,
            "score_p2": snapshot.score_p2,
            "score_diff": snapshot.score_diff,
        })
    return out


def _rows_at_message(quotes_by_stream):
    """(stream, market, message) -> how many raw rows the index chose from.

    A message carrying several rows for one market is where the index had
    to pick, and picking a dead row over a live one is what made spread
    liveness collapse. The count travels on the pair so the choice is
    visible without a second file.
    """
    counts = collections.Counter()
    for stream, rows in quotes_by_stream.items():
        for row in rows:
            code, market_id, message = row[0], row[1], row[6]
            if message is None:
                continue
            counts[(code, message, stream, market_id)] += 1
    return counts


# directional_pairs.csv, plus every column the play-by-play can add.
# Same shape, so it reads the same way, with the drive-detection and
# market-state context that otherwise needs a four-way join.
PAIR_DUMP_FIELDS = report.PAIR_FIELDS + [
    "market", "selection", "prod_decimal", "candidate_decimal",
    "decisive_winner", "decided_by", "basis",
    "prod_state", "candidate_state", "prod_live", "candidate_live", "live",
    "anchor_kind", "anchor_cleaning", "drive_n_plays", "drive_dropped_inside",
    "prod_rows_at_message", "candidate_rows_at_message",
    "score_bucket", "time_bucket", "possession_bucket",
]


def _pair_dump_rows(pairs, play_out, drive_out, rows_at=None):
    """One row per pair, carrying the context the play-by-play holds.

    Built from report.write_pairs_csv's own row for the shared columns, so
    the two cannot describe the same pair differently.
    """
    rows_at = rows_at or {}
    by_drive = {(r["match_code"], r["drive_number"]): r for r in drive_out}
    cleaning = {(r["match_code"], r["event_message_count"]): r["cleaning"]
                for r in play_out}

    out = []
    for pair in pairs:
        row = dict(report.pair_row(pair))
        drive = by_drive.get((pair.match_code, pair.drive_number), {})
        row.update({
            "market": markets.market_group(pair.market_id),
            "selection": markets.selection_label(pair.market_id),
            "prod_decimal": pair.prod_decimal,
            "candidate_decimal": pair.candidate_decimal,
            "decisive_winner": pair.decisive_winner,
            "decided_by": pair.decided_by,
            "basis": "line" if not pair.same_line else "prob",
            "prod_state": pair.prod_state,
            "candidate_state": pair.candidate_state,
            "prod_live": int(pair.prod_live),
            "candidate_live": int(pair.candidate_live),
            "live": _live_label(pair),
            "anchor_kind": pair.anchor,
            "anchor_cleaning": cleaning.get(
                (pair.match_code, pair.message_count), ""),
            "drive_n_plays": drive.get("n_plays", ""),
            "drive_dropped_inside": drive.get("n_dropped_inside", ""),
            "prod_rows_at_message": rows_at.get(
                (pair.match_code, pair.message_count,
                 directional.PROD, pair.market_id), ""),
            "candidate_rows_at_message": rows_at.get(
                (pair.match_code, pair.message_count,
                 directional.CANDIDATE, pair.market_id), ""),
            "score_bucket": buckets.score_diff_bucket(pair.score_diff),
            "time_bucket": buckets.time_bucket(pair.period_number,
                                               pair.drive_number),
            "possession_bucket": buckets.possession_bucket(pair.offensive_team),
        })
        out.append(row)
    out.sort(key=lambda r: (r["match_code"], r["message_count"], r["market_id"]))
    return out


def _live_label(pair):
    """Which side was not tradeable, in the report's own vocabulary."""
    if not pair.not_live:
        return "live"
    if not pair.prod_live and not pair.candidate_live:
        return "both"
    return "prod" if not pair.prod_live else "cand"


def _write(path, fields, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def run(cur, match_codes, out_dir, time_column, verbose=True):
    """Dump the working out for these matches. Returns the paths written."""
    from . import handles
    play_rows = snowflake_io.fetch_plays(cur, match_codes, time_column)
    score_rows = snowflake_io.fetch_scores(cur, match_codes)
    quote_rows = {
        stream: snowflake_io.fetch_quotes(cur, config.STREAMS[stream], match_codes)
        for stream in (directional.PROD, directional.CANDIDATE)
    }

    plays_by_match, scores_by_match = {}, {}
    for row in play_rows:
        plays_by_match.setdefault(row[0], []).append(row)
    for row in score_rows:
        scores_by_match.setdefault(row[0], []).append(row)

    # The same index the calibrator reads its prices from, so a cell in the
    # play-by-play is the price the pair was built on and not a second
    # reading of the feed.
    indexes = {stream: directional.index_by_message(rows)
               for stream, rows in quote_rows.items()}
    rows_at = _rows_at_message(quote_rows)

    all_plays, all_drives = [], []
    for match_code in match_codes:
        rows, snapshots = _rows_for_match(match_code,
                                          plays_by_match.get(match_code, []),
                                          scores_by_match.get(match_code, []),
                                          indexes)
        drive_rows = _drive_rows(match_code, rows, snapshots)
        all_plays.extend(rows)
        all_drives.extend(drive_rows)
        if verbose:
            dropped = sum(1 for row in rows if row["dropped"] == 1)
            off = sum(1 for row in drive_rows
                      if row["anchor_kind"] != drives.FIRST_DOWN)
            live = sum(1 for row in rows
                       if any(row[f"{name}_live"] == "live"
                              for name, _, _ in MARKET_COLUMNS))
            print(f"  {match_code}: {len(rows):,} rows ({dropped:,} dropped), "
                  f"{len(drive_rows)} drives, {off} off anchor, "
                  f"{live:,} messages with a live market in both streams")

    # The pairs the calibrator would build from these same matches, so the
    # anchor rows above can be read against what they became.
    stats = collections.defaultdict(int)
    pairs = directional.build_pairs(cur, match_codes, time_column, stats,
                                    handles.Scan())
    pair_rows = _pair_dump_rows(pairs, all_plays, all_drives, rows_at)
    if verbose and pairs:
        live = sum(1 for r in pair_rows if r["live"] == "live")
        print(f"  {len(pair_rows):,} pairs ({live:,} live) from "
              f"{len({r['message_count'] for r in pair_rows}):,} snapshots")

    written = [
        _write(os.path.join(out_dir, "dump_play_by_play.csv"), PLAY_FIELDS,
               all_plays),
        _write(os.path.join(out_dir, "dump_pairs.csv"), PAIR_DUMP_FIELDS,
               pair_rows),
    ]
    return written, all_plays, all_drives, pair_rows
