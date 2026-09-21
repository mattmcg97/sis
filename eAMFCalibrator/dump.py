"""Write the drive-detection working out to CSV, one match at a time.

Reconciling drives means answering "why did this snapshot land here?", and
that is not a question a summary can answer -- it needs the row-by-row
working: every play the feed sent, which cleaning rule fired on it, which
drive it ended up in, and which single row became the snapshot.

Five files, joinable on (match_code, event_message_count):

  plays       every raw play row, with the cleaning verdict and the drive
              it was assigned to
  scores      every score change, so the play rows can be read against
              what the scoreboard did
  drives      one row per detected drive, with its anchor and buckets
  quotes      every quote row from both streams, undeduplicated, with
              which one the index chose and why
  timeline    one row per message: what the play feed, the scoreboard and
              each stream had at that point in the sequence
  pairs       directional_pairs.csv for these matches, widened with
              everything the other four files know about each row

The three sources share one EVENT_MESSAGE_COUNT sequence -- that is what
makes pairing possible at all -- but they do not line up one to one. A
message can carry six markets times two streams of quotes and no play at
all, or a play and no quote. The timeline is where that is visible.

Deliberately the pipeline's own functions rather than a private copy, so
what is inspected here is what the report is built from.
"""

import collections
import csv
import os

from . import (buckets, config, directional, drives, markets, report,
               snowflake_io)
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

QUOTE_FIELDS = [
    "match_code", "event_message_count", "stream", "market_id", "market",
    "selection", "publish_time", "probability", "decimal_odd", "line",
    "market_description", "status", "is_active", "live",
    "rows_at_this_message", "chosen",
]

TIMELINE_FIELDS = [
    "match_code", "event_message_count", "has_play", "cleaning",
    "drive_number", "is_anchor", "has_score", "score_p1", "score_p2",
    "prod_markets", "prod_live_markets", "candidate_markets",
    "candidate_live_markets", "markets_both_live",
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

    # Assign each surviving play to a drive by walking the snapshots the
    # pipeline produced, rather than re-deriving the grouping here. The
    # first version segmented by team change alone and so disagreed with
    # build_snapshots wherever a drive start did not coincide with one --
    # which is exactly the case this file exists to make visible.
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


def _quote_rows(match_code, quotes_by_stream):
    """Every quote row, with which one the index kept and how many it beat.

    Undeduplicated on purpose. The point is to show the case the pipeline
    had to make a choice about: a message carrying several rows for one
    market, some tradeable and some not.
    """
    out = []
    for stream, rows in quotes_by_stream.items():
        index = directional.index_by_message(rows)
        seen = collections.Counter(
            (r[1], r[6]) for r in rows if r[6] is not None)
        for (code, market_id, publish_time, probability, decimal_odd,
             description, message, status, is_active) in rows:
            if message is None or probability is None:
                continue
            quote = directional.Quote(
                probability=float(probability), description=description,
                decimal=float(decimal_odd) if decimal_odd is not None else None,
                publish_time=publish_time,
                live=directional.is_live(status, is_active),
                state=directional.state_label(status, is_active))
            kept = index.get((code, market_id), {}).get(message)
            out.append({
                "match_code": code,
                "event_message_count": message,
                "stream": stream,
                "market_id": market_id,
                "market": markets.market_group(market_id),
                "selection": markets.selection_label(market_id),
                "publish_time": publish_time,
                "probability": probability,
                "decimal_odd": decimal_odd,
                "line": (markets.parse_line(description)
                         if markets.needs_line(market_id) else ""),
                "market_description": description,
                "status": status,
                "is_active": is_active,
                "live": int(quote.live),
                "rows_at_this_message": seen[(market_id, message)],
                "chosen": int(kept == quote),
            })
    out.sort(key=lambda r: (r["event_message_count"], r["stream"],
                            r["market_id"], str(r["publish_time"])))
    return out


def _timeline_rows(match_code, play_out, score_out, quote_out):
    """One row per message: what each source had at that point.

    Answers the question the other files cannot: whether the play feed,
    the scoreboard and the two streams are talking about the same moments.
    """
    messages = sorted({r["event_message_count"] for r in play_out}
                      | {r["event_message_count"] for r in score_out}
                      | {r["event_message_count"] for r in quote_out})
    plays = {r["event_message_count"]: r for r in play_out}
    scores = {r["event_message_count"]: r for r in score_out}

    per_stream = collections.defaultdict(lambda: collections.defaultdict(set))
    per_stream_live = collections.defaultdict(lambda: collections.defaultdict(set))
    for row in quote_out:
        if not row["chosen"]:
            continue
        per_stream[row["stream"]][row["event_message_count"]].add(row["market_id"])
        if row["live"]:
            per_stream_live[row["stream"]][row["event_message_count"]].add(
                row["market_id"])

    out = []
    for message in messages:
        play = plays.get(message)
        score = scores.get(message)
        prod_live = per_stream_live["prod"][message]
        cand_live = per_stream_live["candidate"][message]
        out.append({
            "match_code": match_code,
            "event_message_count": message,
            "has_play": int(play is not None),
            "cleaning": play["cleaning"] if play else "",
            "drive_number": play["drive_number"] if play else "",
            "is_anchor": play["is_anchor"] if play else 0,
            "has_score": int(score is not None),
            "score_p1": score["player_1_cumulative"] if score else "",
            "score_p2": score["player_2_cumulative"] if score else "",
            "prod_markets": len(per_stream["prod"][message]),
            "prod_live_markets": len(prod_live),
            "candidate_markets": len(per_stream["candidate"][message]),
            "candidate_live_markets": len(cand_live),
            "markets_both_live": len(prod_live & cand_live),
        })
    return out


# directional_pairs.csv, plus every column the rest of the dump can add.
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


def _pair_dump_rows(pairs, play_out, drive_out, quote_out):
    """One row per pair, carrying the context the other files hold.

    Built from report.write_pairs_csv's own row for the shared columns, so
    the two cannot describe the same pair differently.
    """
    by_drive = {(r["match_code"], r["drive_number"]): r for r in drive_out}
    cleaning = {(r["match_code"], r["event_message_count"]): r["cleaning"]
                for r in play_out}
    rows_at = collections.defaultdict(int)
    for row in quote_out:
        key = (row["match_code"], row["event_message_count"],
               row["stream"], row["market_id"])
        rows_at[key] = max(rows_at[key], row["rows_at_this_message"])

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
    quotes_by_match = collections.defaultdict(
        lambda: {directional.PROD: [], directional.CANDIDATE: []})
    for stream, rows in quote_rows.items():
        for row in rows:
            quotes_by_match[row[0]][stream].append(row)

    all_plays, all_scores, all_drives = [], [], []
    all_quotes, all_timeline = [], []
    for match_code in match_codes:
        p, s, d = _rows_for_match(match_code,
                                  plays_by_match.get(match_code, []),
                                  scores_by_match.get(match_code, []))
        q = _quote_rows(match_code, quotes_by_match[match_code])
        t = _timeline_rows(match_code, p, s, q)
        all_plays.extend(p)
        all_scores.extend(s)
        all_drives.extend(d)
        all_quotes.extend(q)
        all_timeline.extend(t)
        if verbose:
            kept = sum(1 for row in p if not row["dropped"])
            off = sum(1 for row in d if row["anchor_kind"] != drives.FIRST_DOWN)
            shared = sum(1 for row in t if row["markets_both_live"])
            print(f"  {match_code}: {len(p):,} plays ({kept:,} kept), "
                  f"{len(d)} drives, {off} off anchor, {len(s)} score changes, "
                  f"{len(q):,} quote rows, {len(t):,} messages "
                  f"({shared:,} with a live market in both streams)")

    # The pairs the calibrator would build from these same matches, so the
    # snapshot rows above can be read against what they became.
    stats = collections.defaultdict(int)
    pairs = directional.build_pairs(cur, match_codes, time_column, stats,
                                    handles.Scan())
    pair_rows = _pair_dump_rows(pairs, all_plays, all_drives, all_quotes)
    if verbose and pairs:
        live = sum(1 for r in pair_rows if r["live"] == "live")
        print(f"  {len(pair_rows):,} pairs ({live:,} live) from "
              f"{len({r['message_count'] for r in pair_rows}):,} snapshots")

    written = [
        _write(os.path.join(out_dir, "dump_pairs.csv"), PAIR_DUMP_FIELDS,
               pair_rows),
        _write(os.path.join(out_dir, "dump_plays.csv"), PLAY_FIELDS, all_plays),
        _write(os.path.join(out_dir, "dump_scores.csv"), SCORE_FIELDS, all_scores),
        _write(os.path.join(out_dir, "dump_drives.csv"), DRIVE_FIELDS, all_drives),
        _write(os.path.join(out_dir, "dump_quotes.csv"), QUOTE_FIELDS, all_quotes),
        _write(os.path.join(out_dir, "dump_timeline.csv"), TIMELINE_FIELDS,
               all_timeline),
    ]
    return (written, all_plays, all_scores, all_drives, all_quotes,
            all_timeline, pair_rows)
