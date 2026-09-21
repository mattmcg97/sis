"""Snapshot -> nearest quote -> realized outcome, aggregated into cells.

One observation is produced per (drive-start snapshot x market selection)
where a quote from the chosen stream lands within
config.MATCH_TOLERANCE_SECONDS of the snapshot. The quote supplies the
predicted probability and, for spread and totals, the line that was live
at that moment; the match's final score supplies the realized outcome.

Snapshots with no quote inside the tolerance are counted, not silently
dropped -- a run whose coverage collapses is telling you something about
the feed, and the run header says so.

Sample-size caveat carried over from analysis/unconditional_calibration.py:
drives inside one match share a game and are not independent, so both row
count and distinct match count are reported per cell. Read the match count.
"""

from collections import defaultdict
from dataclasses import dataclass
from bisect import bisect_left
from typing import Optional

from . import (buckets, clock, config, handles, markets, metrics,
               snowflake_io)
from .drives import PlayRow, ScoreRow, build_snapshots


@dataclass(frozen=True)
class Observation:
    match_code: str
    drive_number: int
    period_number: Optional[int]
    score_diff: int
    offensive_team: Optional[str]
    market_id: int
    line: Optional[float]
    probability: float
    outcome: bool
    gap_seconds: float


def to_unit_probability(raw):
    """GAMEPLAI publishes 0-100; everything downstream works in 0-1."""
    if raw is None:
        return None
    return float(raw) / 100.0


def nearest_quote(times, target, tolerance, direction):
    """Index of the quote closest to target within tolerance, or None.

    times must be sorted. Returns (index, signed_gap_seconds) where a
    positive gap means the quote was published after the snapshot.
    """
    if not times or target is None:
        return None

    idx = bisect_left(times, target)
    candidates = []
    if direction == "forward":
        if idx < len(times):
            candidates.append(idx)
    else:
        if idx < len(times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)

    best = None
    for i in candidates:
        gap = (times[i] - target).total_seconds()
        if abs(gap) <= tolerance and (best is None or abs(gap) < abs(best[1])):
            best = (i, gap)
    return best


def index_quotes(quote_rows):
    """(match_code, market_id) -> (sorted times, parallel quote records)."""
    grouped = defaultdict(list)
    for match_code, market_id, publish_time, probability, decimal_odd, description, msg in quote_rows:
        grouped[(match_code, market_id)].append((publish_time, probability, description))
    indexed = {}
    for key, rows in grouped.items():
        rows.sort(key=lambda r: r[0])
        indexed[key] = ([r[0] for r in rows], rows)
    return indexed


def group_rows(rows, key_index):
    out = defaultdict(list)
    for row in rows:
        out[row[key_index]].append(row)
    return out


def snapshot_time(snap, match_clock, stats):
    """A snapshot's wall-clock time, and where it came from.

    The play feed's own column wins if it ever gains one; otherwise the
    time is reconstructed from the stream's message clock.
    """
    if snap.play_time is not None:
        stats["clock_from_play_feed"] += 1
        return snap.play_time

    if match_clock is None:
        stats["clock_unresolved"] += 1
        return None

    when, provenance = match_clock.time_for(snap.event_message_count)
    stats[f"clock_{provenance}"] += 1
    return when


def observations_for_chunk(cur, stream_table, match_codes, time_column, stats,
                           clock_table=None, scan=None):
    """Build every observation for one chunk of matches.

    `scan` is the handle check, accumulating across chunks. Every match is
    scanned whether or not the result is acted on, so the report can say
    how big the problem is even when nothing is being excluded.
    """
    scan = scan if scan is not None else handles.Scan()
    play_rows = snowflake_io.fetch_plays(cur, match_codes, time_column)
    score_rows = snowflake_io.fetch_scores(cur, match_codes)
    finals = snowflake_io.fetch_final_scores(cur, match_codes)
    quote_rows = snowflake_io.fetch_quotes(cur, stream_table, match_codes)

    clocks = {}
    if time_column is None:
        clock_rows = snowflake_io.fetch_message_times(
            cur, clock_table or stream_table, match_codes)
        clocks = clock.build_clocks(clock_rows)

    plays_by_match = group_rows(play_rows, 0)
    scores_by_match = group_rows(score_rows, 0)
    quote_index = index_quotes(quote_rows)

    observations = []
    for match_code in match_codes:
        plays = [
            PlayRow(event_message_count=r[1], period_number=r[2], offensive_team=r[3],
                    down_number=r[4], distance=r[5], field_position=r[6], play_time=r[7])
            for r in plays_by_match.get(match_code, [])
        ]
        scores = [
            ScoreRow(event_message_count=r[1], period_number=r[2], p1_change=r[3],
                     p2_change=r[4], p1_cumulative=r[5], p2_cumulative=r[6])
            for r in scores_by_match.get(match_code, [])
        ]
        final = finals.get(match_code)
        if not scan.add(match_code, scores, final):
            stats["matches_with_flipped_handles"] += 1
            if config.EXCLUDE_FLIPPED_MATCHES:
                stats["matches_excluded_for_flipped_handles"] += 1
                continue

        if not plays:
            stats["matches_without_plays"] += 1
            continue

        if final is None or final[0] is None or final[1] is None:
            stats["matches_without_final"] += 1
            continue
        final_p1, final_p2 = final

        snapshots = build_snapshots(match_code, plays, scores)
        stats["snapshots"] += len(snapshots)
        match_clock = clocks.get(match_code)

        for snap in snapshots:
            snap_time = snapshot_time(snap, match_clock, stats)
            if snap_time is None:
                stats["snapshots_without_time"] += 1
                continue
            matched_any = False
            for market_id in markets.MARKET_IDS:
                entry = quote_index.get((match_code, market_id))
                if not entry:
                    stats["no_quote_for_market"] += 1
                    continue
                times, rows = entry
                hit = nearest_quote(times, snap_time,
                                    config.MATCH_TOLERANCE_SECONDS, config.QUOTE_DIRECTION)
                if hit is None:
                    stats["quote_outside_tolerance"] += 1
                    continue
                idx, gap = hit
                _, raw_probability, description = rows[idx]
                probability = to_unit_probability(raw_probability)
                if probability is None:
                    stats["null_probability"] += 1
                    continue

                line = markets.parse_line(description) if markets.needs_line(market_id) else None
                if markets.needs_line(market_id) and line is None:
                    stats["unparsed_line"] += 1
                    continue

                outcome = markets.resolve(market_id, line, final_p1, final_p2)
                if outcome is None:
                    stats["pushes_or_unresolved"] += 1
                    continue

                matched_any = True
                observations.append(Observation(
                    match_code=match_code,
                    drive_number=snap.drive_number,
                    period_number=snap.period_number,
                    score_diff=snap.score_diff,
                    offensive_team=snap.offensive_team,
                    market_id=market_id,
                    line=line,
                    probability=probability,
                    outcome=outcome,
                    gap_seconds=gap,
                ))
            if matched_any:
                stats["snapshots_matched"] += 1

    return observations


def cell_for(observation):
    """Full cell key: the three split axes, then market group and selection."""
    return (
        buckets.score_diff_bucket(observation.score_diff),
        buckets.time_bucket(observation.period_number, observation.drive_number),
        buckets.possession_bucket(observation.offensive_team),
        markets.market_group(observation.market_id),
        markets.selection_label(observation.market_id),
    )


def aggregate(observations):
    """Cell -> pairs, plus the distinct matches behind each cell."""
    cells = defaultdict(list)
    cell_matches = defaultdict(set)
    for obs in observations:
        key = cell_for(obs)
        cells[key].append((obs.probability, obs.outcome))
        cell_matches[key].add(obs.match_code)
    return cells, cell_matches


def new_stats():
    return defaultdict(int)


def run(stream_key, verbose=True):
    """Calibrate one stream.

    Returns (observations, stats, header, handle_scan).
    """
    if stream_key not in config.STREAMS:
        raise ValueError(f"unknown stream {stream_key!r}; expected one of {sorted(config.STREAMS)}")
    stream_table = config.STREAMS[stream_key]

    stats = new_stats()
    observations = []
    header = {}
    scan = handles.Scan()

    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            time_column, play_columns = snowflake_io.detect_play_time_column(cur)
            header["play_time_column"] = time_column
            header["play_columns"] = play_columns

            clock_table = None
            if time_column is None:
                clock_key = config.CLOCK_SOURCE or stream_key
                clock_table = config.STREAMS.get(clock_key, stream_table)
                header["clock_mode"] = f"reconstructed from {clock_key} message clock"
            else:
                header["clock_mode"] = f"{snowflake_io.PLAY_TABLE}.{time_column}"

            n_rows, n_matches, first, last = snowflake_io.stream_window_summary(cur, stream_table)
            header.update({"stream": stream_key, "table": stream_table,
                           "window_rows": n_rows, "window_matches": n_matches,
                           "window_first": first, "window_last": last})

            match_codes = snowflake_io.match_universe(cur, stream_table)
            header["universe_matches"] = len(match_codes)
            if verbose:
                print(f"  {len(match_codes)} settled {config.SPORT_CODE} matches with quotes in window")

            chunk = config.MATCH_CHUNK_SIZE
            for start in range(0, len(match_codes), chunk):
                batch = match_codes[start:start + chunk]
                if verbose:
                    print(f"  chunk {start // chunk + 1}: {len(batch)} matches", flush=True)
                observations.extend(
                    observations_for_chunk(cur, stream_table, batch, time_column,
                                           stats, clock_table, scan)
                )
    finally:
        conn.close()

    return observations, stats, header, scan.summary()


def overall_summary(observations):
    """Headline metrics for a whole run, and per market group."""
    by_group = defaultdict(list)
    everything = []
    for obs in observations:
        pair = (obs.probability, obs.outcome)
        everything.append(pair)
        by_group[markets.market_group(obs.market_id)].append(pair)

    summary = {"ALL": metrics.summarize(everything, config.N_RELIABILITY_BINS)}
    for group, pairs in by_group.items():
        summary[group] = metrics.summarize(pairs, config.N_RELIABILITY_BINS)
    return summary
