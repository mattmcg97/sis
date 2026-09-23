"""Console tables and CSV output for a calibration run.

The CSVs are the point of the suite being re-runnable: each run writes a
cell-level summary keyed identically every time, so two runs (prod vs
candidate, or the same stream a week apart) diff cleanly with `compare`.
"""

import collections
import csv
import os

from . import buckets, config, handles, metrics

CELL_FIELDS = [
    "stream", "score_diff", "time_bucket", "possession", "market", "selection",
    "n", "matches", "mean_predicted", "realized", "gap", "brier", "log_loss", "ece",
]

OBS_FIELDS = [
    "match_code", "drive_number", "period_number", "score_diff", "offensive_team",
    "market_id", "line", "probability", "outcome", "gap_seconds",
]

# Size the bucket columns from the labels themselves, so renaming a bucket
# cannot knock the text tables out of alignment. CELL_WIDTH is for the full
# cross-section label, which is the three axes joined by spaces.
SCORE_WIDTH = max(len(label) for _, _, label in config.SCORE_DIFF_BUCKETS) + 2
CELL_WIDTH = SCORE_WIDTH + len(" unknown unknown")


def _fmt(value, spec=".4f"):
    return "n/a" if value is None else format(value, spec)


def print_header(header, stats):
    print(f"\n{'=' * 78}")
    print(f"Stream: {header.get('stream')}  ({header.get('table')})")
    print(f"{'=' * 78}")
    print(f"Window start        : {config.CUTOFF_START}"
          f"{'' if config.CUTOFF_END is None else ' -> ' + str(config.CUTOFF_END)}")
    print(f"Quotes in window    : {header.get('window_rows', 0):,} rows, "
          f"{header.get('window_matches', 0):,} matches")
    print(f"Actual quote span   : {header.get('window_first')}  ->  {header.get('window_last')}")
    print(f"{'Settled ' + config.SPORT_CODE + ' universe':<20}: {header.get('universe_matches', 0):,} matches")
    print(f"Snapshot clock      : {header.get('clock_mode')}")
    print(f"Match tolerance     : {config.MATCH_TOLERANCE_SECONDS}s ({config.QUOTE_DIRECTION})")
    print(f"Time axis           : {buckets.time_axis_name()}")

    print("\nCoverage:")
    print(f"  drive snapshots built            : {stats.get('snapshots', 0):,}")
    print(f"  snapshots with >=1 matched quote : {stats.get('snapshots_matched', 0):,}")

    exact = stats.get("clock_exact", 0)
    interpolated = stats.get("clock_interpolated", 0)
    unresolved = stats.get("clock_unresolved", 0)
    from_feed = stats.get("clock_from_play_feed", 0)
    timed = exact + interpolated + unresolved + from_feed
    if timed:
        print("\n  Snapshot clock provenance:")
        if from_feed:
            print(f"    from the play feed's own column: {from_feed:,}")
        if exact or interpolated or unresolved:
            print(f"    exact message hit              : {exact:,} ({100 * exact / timed:.1f}%)")
            print(f"    interpolated between messages  : {interpolated:,} ({100 * interpolated / timed:.1f}%)")
            print(f"    unresolved (dropped)           : {unresolved:,} ({100 * unresolved / timed:.1f}%)")
        if exact and 100 * exact / timed < 50:
            print("    ^ under half the play feed's message counts appear in the stream.")
            print("      Check that both are numbering the same sequence before trusting")
            print("      the sub-3-second gaps.")
    for label, key in [
        ("snapshots with no clock", "snapshots_without_time"),
        ("no quote at all for market", "no_quote_for_market"),
        ("nearest quote outside tolerance", "quote_outside_tolerance"),
        ("null probability", "null_probability"),
        ("line could not be parsed", "unparsed_line"),
        ("push / unresolved", "pushes_or_unresolved"),
        ("messages offering >1 row for a market", "quote_rows_sharing_a_message"),
        ("...where a live row replaced a dead one", "quote_upgraded_to_live"),
        ("matches with no play rows", "matches_without_plays"),
        ("matches with no final score", "matches_without_final"),
    ]:
        if stats.get(key):
            print(f"  {label:<33}: {stats[key]:,}")


def print_overall(summary):
    print(f"\n{'-' * 78}\nOverall calibration\n{'-' * 78}")
    print(f"  {'GROUP':<12}{'N':>9}{'PRED':>9}{'REAL':>9}{'GAP':>9}"
          f"{'BRIER':>10}{'LOGLOSS':>10}{'ECE':>9}")
    for group in ["ALL"] + sorted(k for k in summary if k != "ALL"):
        s = summary[group]
        if not s["n"]:
            continue
        print(f"  {group:<12}{s['n']:>9,}{_fmt(s['mean_predicted'], '.3f'):>9}"
              f"{_fmt(s['realized'], '.3f'):>9}{_fmt(s['gap'], '+.3f'):>9}"
              f"{_fmt(s['brier']):>10}{_fmt(s['log_loss']):>10}{_fmt(s['ece']):>9}")
    print("\n  GAP is realized minus predicted. Positive means the model is")
    print("  underpricing the selection; negative means overpricing.")


def print_reliability(observations):
    pairs = [(o.probability, o.outcome) for o in observations]
    if not pairs:
        return
    print(f"\n{'-' * 78}\nReliability curve (all markets pooled)\n{'-' * 78}")
    print(f"  {'BIN':<14}{'N':>9}{'PRED':>9}{'REAL':>9}{'GAP':>9}")
    for b in metrics.reliability_bins(pairs, config.N_RELIABILITY_BINS):
        if not b["n"]:
            continue
        label = f"{b['lo']:.1f}-{b['hi']:.1f}"
        print(f"  {label:<14}{b['n']:>9,}{_fmt(b['predicted'], '.3f'):>9}"
              f"{_fmt(b['realized'], '.3f'):>9}{_fmt(b['gap'], '+.3f'):>9}")


def cell_rows(stream, cells, cell_matches):
    rows = []
    for key in sorted(cells, key=buckets.sort_key):
        pairs = cells[key]
        s = metrics.summarize(pairs, config.N_RELIABILITY_BINS)
        score_b, time_b, poss, market, selection = key
        rows.append({
            "stream": stream,
            "score_diff": score_b,
            "time_bucket": time_b,
            "possession": poss,
            "market": market,
            "selection": selection,
            "n": s["n"],
            "matches": len(cell_matches[key]),
            "mean_predicted": s["mean_predicted"],
            "realized": s["realized"],
            "gap": s["gap"],
            "brier": s["brier"],
            "log_loss": s["log_loss"],
            "ece": s["ece"],
        })
    return rows


def print_cells(rows):
    print(f"\n{'-' * 100}\nCalibration by cell "
          f"(score diff x {buckets.time_axis_name().lower()} x possession x selection)\n{'-' * 100}")
    print(f"  {'SCORE':<{SCORE_WIDTH}}{buckets.time_axis_name():<12}{'POSS':<7}"
          f"{'MARKET':<11}{'SEL':<7}"
          f"{'N':>8}{'MATCH':>7}{'PRED':>8}{'REAL':>8}{'GAP':>8}{'BRIER':>9}")
    for r in rows:
        print(f"  {r['score_diff']:<{SCORE_WIDTH}}{r['time_bucket']:<12}"
              f"{r['possession']:<7}"
              f"{r['market']:<11}{r['selection']:<7}{r['n']:>8,}{r['matches']:>7,}"
              f"{_fmt(r['mean_predicted'], '.3f'):>8}{_fmt(r['realized'], '.3f'):>8}"
              f"{_fmt(r['gap'], '+.3f'):>8}{_fmt(r['brier'], '.4f'):>9}")


def print_handle_check(scan, limit=20):
    """Whether PLAYER_1 / PLAYER_2 stayed pinned to the same team."""
    print(f"\n{'=' * 104}\nHANDLE CHECK -- do PLAYER_1 / PLAYER_2 stay on the "
          f"same team?\n{'=' * 104}")
    if not scan["matches"]:
        print("  No matches scanned.")
        return

    flipped = scan["matches_flipped"]
    share = flipped / scan["matches"]
    verdict = "CLEAN" if not flipped else "FLIPPED HANDLES FOUND"
    print(f"  {scan['matches']:,} matches scanned   {scan['matches_clean']:,} with "
          f"nothing to flag   {flipped:,} with flipped handles ({share:.1%})")
    print(f"  Verdict: {verdict}")
    if flipped and not config.EXCLUDE_FLIPPED_MATCHES:
        print("  These matches are STILL IN the numbers above. Their score")
        print("  difference, possession flag and outcomes invert at the flip,")
        print("  so their snapshots sit in the wrong buckets. Set")
        print("  config.EXCLUDE_FLIPPED_MATCHES = True to drop them.")
    elif flipped:
        print("  Excluded from the calibration (EXCLUDE_FLIPPED_MATCHES is on).")

    width = handles.KIND_WIDTH
    print(f"\n  {'KIND':<{width}}{'EVENTS':>8}{'MATCHES':>9}{'FLIP':>6}")
    for kind in handles.KIND_ORDER:
        events = scan["counts"][kind]
        if not events:
            continue
        flips = "yes" if kind in handles.FLIP_KINDS else "no"
        print(f"  {handles.KIND_TITLES[kind]:<{width}}{events:>8,}"
              f"{scan['matches_by_kind'][kind]:>9,}{flips:>6}")
    if not scan["anomalies"]:
        print("  (nothing flagged)")
        return

    print(f"\n  {'MATCH':<16}{'WHERE':>9}  {'KIND':<{width}}EVIDENCE")
    for anomaly in scan["anomalies"][:limit]:
        print(f"  {anomaly.match_code:<16}{anomaly.where:>9}  "
              f"{handles.KIND_TITLES[anomaly.kind]:<{width}}{anomaly.describe()}")
    if len(scan["anomalies"]) > limit:
        print(f"  ... {len(scan['anomalies']) - limit:,} more")

    print("\n  MIRRORED is a swap and nothing else: the two totals traded")
    print("  places exactly. REGRESSION is a total going down some other way,")
    print("  which a swap landing on a scoring message also looks like, as")
    print("  does a rescinded score. FINAL MISMATCH is the running total")
    print("  disagreeing with SCORE_ENDGAME, which a missing late score")
    print("  explains as well as a swap, so it is not counted as a flip.")
    print("  Blind spot: a swap while the score is level leaves no trace.")


def print_worst(rows):
    eligible = [r for r in rows
                if r["n"] >= config.MIN_CELL_OBSERVATIONS
                and r["matches"] >= config.MIN_CELL_MATCHES
                and r["gap"] is not None]
    if not eligible:
        print(f"\n  No cell reaches {config.MIN_CELL_OBSERVATIONS} observations across "
              f"{config.MIN_CELL_MATCHES} matches yet -- too little data to call anything mispriced.")
        return
    worst = sorted(eligible, key=lambda r: abs(r["gap"]), reverse=True)[:15]
    print(f"\n{'-' * 78}\nWorst cells by |gap| "
          f"(min {config.MIN_CELL_OBSERVATIONS} obs, {config.MIN_CELL_MATCHES} matches)\n{'-' * 78}")
    for r in worst:
        print(f"  {r['score_diff']:<9}{r['time_bucket']:<10}{r['possession']:<6}"
              f"{r['market']:<10}{r['selection']:<6} n={r['n']:<6} matches={r['matches']:<5} "
              f"pred={_fmt(r['mean_predicted'], '.3f')} real={_fmt(r['realized'], '.3f')} "
              f"gap={_fmt(r['gap'], '+.3f')}")


def write_cells_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CELL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  cell summary -> {path}")


def write_observations_csv(path, observations):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OBS_FIELDS)
        writer.writeheader()
        for o in observations:
            writer.writerow({
                "match_code": o.match_code,
                "drive_number": o.drive_number,
                "period_number": o.period_number,
                "score_diff": o.score_diff,
                "offensive_team": o.offensive_team,
                "market_id": o.market_id,
                "line": o.line,
                "probability": o.probability,
                "outcome": int(o.outcome),
                "gap_seconds": o.gap_seconds,
            })
    print(f"  observations -> {path}")


def read_cells_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def print_comparison(rows_a, rows_b, label_a, label_b, limit=40):
    """Diff two cell summaries on the cells they share.

    Ordered by how much the two disagree, capped at `limit` rows -- the
    full grid runs to hundreds of cells and the ones worth reading are
    always at the top.
    """
    key = lambda r: (r["score_diff"], r["time_bucket"], r["possession"], r["market"], r["selection"])
    index_b = {key(r): r for r in rows_b}

    shared = [(a, index_b[key(a)]) for a in rows_a if key(a) in index_b]
    print(f"\n{'=' * 104}\n{label_a} vs {label_b}: {len(shared)} shared cells\n{'=' * 104}")
    if not shared:
        print("  No cells in common -- were both runs built with the same buckets and time axis?")
        return

    print(f"  {'SCORE':<9}{'TIME':<10}{'POSS':<6}{'MARKET':<10}{'SEL':<6}"
          f"{'N_A':>7}{'N_B':>7}{'GAP_A':>9}{'GAP_B':>9}{'|A|-|B|':>10}{'BRIER_A':>10}{'BRIER_B':>10}")
    scored = []
    for a, b in shared:
        gap_a, gap_b = _float(a["gap"]), _float(b["gap"])
        if gap_a is None or gap_b is None:
            continue
        improvement = abs(gap_a) - abs(gap_b)
        scored.append((improvement, a, b, gap_a, gap_b))

    ranked = sorted(scored, key=lambda t: abs(t[0]), reverse=True)
    for improvement, a, b, gap_a, gap_b in ranked[:limit]:
        print(f"  {a['score_diff']:<9}{a['time_bucket']:<10}{a['possession']:<6}"
              f"{a['market']:<10}{a['selection']:<6}{int(a['n']):>7,}{int(b['n']):>7,}"
              f"{gap_a:>+9.3f}{gap_b:>+9.3f}{improvement:>+10.3f}"
              f"{_fmt(_float(a['brier'])):>10}{_fmt(_float(b['brier'])):>10}")

    if len(ranked) > limit:
        print(f"  ... {len(ranked) - limit} further cells not shown (smallest disagreement).")

    better = sum(1 for imp, *_ in scored if imp > 0)
    worse = sum(1 for imp, *_ in scored if imp < 0)
    print(f"\n  GAP_A is {label_a}, GAP_B is {label_b}. |A|-|B| positive means {label_b}")
    print(f"  sits closer to realized in that cell.")
    print(f"  {label_b} closer in {better} cells, {label_a} closer in {worse}, "
          f"tied in {len(scored) - better - worse}.")


# ---------------------------------------------------------------------------
# Directional (paired) comparison
# ---------------------------------------------------------------------------

PAIR_FIELDS = [
    "publish_time", "match_code", "drive_number", "period_number",
    "score_p1", "score_p2", "score_diff", "offensive_team",
    "field_position", "down_number", "distance",
    "market_id", "message_count", "message_gap",
    "prod_line", "candidate_line", "line_delta", "same_line",
    "prod_probability", "candidate_probability",
    "prod_outcome", "candidate_outcome", "realized",
    "prod_error", "candidate_error", "prod_line_error", "candidate_line_error",
    "disagreement", "probability_winner", "line_winner",
    "prod_live", "candidate_live", "prod_state", "candidate_state",
]


def _pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _p(value):
    if value is None:
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def print_directional_header(header, stats):
    print(f"\n{'=' * 78}")
    print("Directional comparison: prod vs candidate, paired at each snapshot")
    print(f"{'=' * 78}")
    print(f"Window start        : {config.CUTOFF_START}")
    for key in ("prod", "candidate"):
        info = header.get(key, {})
        print(f"{key:<20}: {info.get('rows', 0):,} rows, {info.get('matches', 0):,} matches"
              f"   {info.get('first')} -> {info.get('last')}")
    print(f"Matches in both     : {header.get('paired_matches', 0):,}"
          f"  (prod-only {header.get('prod_only_matches', 0)},"
          f" candidate-only {header.get('candidate_only_matches', 0)})")
    print(f"Paired on           : EVENT_MESSAGE_COUNT, max gap "
          f"{config.MAX_PAIR_MESSAGE_GAP} messages")

    print("\nCoverage:")
    print(f"  drive snapshots built     : {stats.get('snapshots', 0):,}")
    print(f"  snapshots with >=1 pair   : {stats.get('snapshots_paired', 0):,}")
    exact = stats.get("exact_message_pair", 0)
    offset = stats.get("offset_message_pair", 0)
    if exact + offset:
        share = 100 * exact / (exact + offset)
        print(f"  pairs on the exact message: {exact:,} ({share:.1f}%)")
        print(f"  pairs on a nearby message : {offset:,}")
    for label, key in [
        ("messages offering >1 row for a market", "quote_rows_sharing_a_message"),
        ("...where a live row replaced a dead one", "quote_upgraded_to_live"),
        ("market never quoted by both", "no_common_message_for_market"),
        ("nearest common message too far", "outside_message_gap"),
        ("conversion/kick/score quote passed over", "garbage_quote_message_avoided"),
        ("line could not be parsed", "unparsed_line"),
        ("push / unresolved", "pushes_or_unresolved"),
        ("matches with no play rows", "matches_without_plays"),
        ("matches with no final score", "matches_without_final"),
    ]:
        if stats.get(key):
            print(f"  {label:<39}: {stats[key]:,}")


def print_directional_headline(overall, votes):
    print(f"\n{'-' * 78}\nHeadline\n{'-' * 78}")
    print(f"  pairs                     : {overall['n']:,} "
          f"across {overall['n_matches']:,} matches")
    print(f"  candidate closer          : {overall['candidate']:,}")
    print(f"  prod closer               : {overall['prod']:,}")
    print(f"  identical probability     : {overall['tie']:,}")
    print(f"  candidate win rate        : {_pct(overall['candidate_win_rate'])}"
          f"  (of {overall['decisive']:,} decisive pairs)")
    print(f"  sign test p               : {_p(overall['p_value'])}   [row level]")
    print()
    print(f"  mean |error|  prod        : {_fmt(overall['prod_mae'])}")
    print(f"  mean |error|  candidate   : {_fmt(overall['candidate_mae'])}")
    print(f"  improvement (prod - cand) : {_fmt(overall['mae_delta'], '+.4f')}")
    print(f"  Brier         prod        : {_fmt(overall['prod_brier'])}")
    print(f"  Brier         candidate   : {_fmt(overall['candidate_brier'])}")
    print(f"  improvement (prod - cand) : {_fmt(overall['brier_delta'], '+.4f')}")

    print(f"\n{'-' * 78}\nMatch-level vote (clustering-robust)\n{'-' * 78}")
    print("  Each match votes once, by which model won more of its pairs.")
    print("  Pairs inside a match are not independent; matches are.")
    print(f"  matches                   : {votes['n_matches']:,}")
    print(f"  candidate won the match   : {votes['candidate']:,}")
    print(f"  prod won the match        : {votes['prod']:,}")
    print(f"  split evenly              : {votes['tie']:,}")
    print(f"  candidate win rate        : {_pct(votes['candidate_win_rate'])}")
    print(f"  sign test p               : {_p(votes['p_value'])}   [match level]")
    print("\n  If the two levels disagree, the match level is the one to trust.")


def _print_market_coverage(plays):
    """What the two streams had quoted at the messages the plays sit on.

    The play-by-play carries a live flag per selection, so the ceiling on
    pairs -- how many snapshots had a market both streams were actually
    quoting -- reads straight off it, without a second file to join.
    """
    from . import dump
    names = [name for name, _, _ in dump.MARKET_COLUMNS]

    def live_count(row):
        return sum(1 for name in names if row[f"{name}_live"] == "live")

    def quoted(row):
        return any(row[f"{name}_live"] for name in names)

    with_quote = sum(1 for row in plays if quoted(row))
    print(f"\n  {len(plays):,} rows in the play-by-play, "
          f"{with_quote:,} carrying a quote from either stream")

    print(f"\n  {'SELECTION':<12}{'QUOTED':>9}{'LIVE BOTH':>11}{'SHARE':>8}"
          f"   {'MISSING':>8}")
    for name in names:
        counted = collections.Counter(row[f"{name}_live"] for row in plays)
        total = sum(n for key, n in counted.items() if key)
        if not total:
            continue
        print(f"  {name:<12}{total:>9,}{counted['live']:>11,}"
              f"{_pct(counted['live'] / total):>8}   {counted['missing']:>8,}")

    anchors = [row for row in plays if row["is_anchor"]]
    if anchors:
        full = sum(1 for row in anchors if live_count(row) == len(names))
        none = sum(1 for row in anchors if not live_count(row))
        print(f"\n  At the {len(anchors):,} rows a snapshot anchors on:")
        print(f"    all six selections live in both streams : {full:,}")
        print(f"    no selection live in both               : {none:,}")
        print("  A snapshot can only be paired on a selection both streams")
        print("  had live at that message, so this is the ceiling on pairs")
        print("  before any tolerance or line rule is applied.")


def _print_anchor_field_check(drive_rows):
    """Are the anchors piling up on one yard line?

    Drives start all over the field, so a spike is not a fact about
    football -- it is the kick spot being read as a drive. Worth checking
    automatically because it took reading a CSV by eye to notice that 58%
    of snapshots sat on the same yard line.
    """
    positions = collections.Counter(
        row["field_position"] for row in drive_rows
        if row["field_position"] not in (None, ""))
    if not positions:
        return
    total = sum(positions.values())
    print(f"\n  {'FIELD':<8}{'DRIVES':>8}{'SHARE':>8}   anchor field position")
    for value, n in positions.most_common(5):
        print(f"  {str(value):<8}{n:>8,}{_pct(n / total):>8}")
    top_value, top_n = positions.most_common(1)[0]
    share = top_n / total
    if share > 0.2:
        print(f"\n  {_pct(share)} of drives start on the {top_value}. Drives")
        print("  start all over the field, so this is the kick spot being")
        print("  read as a drive rather than anything about the football.")


def print_dump_summary(plays, drive_rows, pair_rows=()):
    """What the dumped rows say about drive detection, before opening them."""
    print(f"\n{'=' * 78}\nDRIVE DETECTION -- what the CSVs contain\n{'=' * 78}")
    if not plays:
        print("  No play rows.")
        return
    matches = len({row["match_code"] for row in plays})
    snaps = [row for row in plays if row["cleaning"] != "score"]
    dropped = sum(1 for row in snaps if row["dropped"] == 1)
    scores = sum(1 for row in plays if row["p1_change"] or row["p2_change"])
    print(f"  {len(snaps):,} play rows across {matches} matches, "
          f"{dropped:,} dropped by cleaning ({100 * dropped / len(snaps):.1f}%)")
    print(f"  {scores:,} score changes, carried on the same rows")
    print(f"  {len(drive_rows):,} drives detected "
          f"({len(drive_rows) / matches:.1f} per match, against a realistic 22ish)")

    by_reason = collections.Counter(row["cleaning"] for row in snaps)
    print(f"\n  {'CLEANING VERDICT':<24}{'ROWS':>8}{'SHARE':>8}")
    for reason, n in by_reason.most_common():
        print(f"  {reason:<24}{n:>8,}{_pct(n / len(snaps)):>8}")

    if drive_rows:
        by_anchor = collections.Counter(row["anchor_kind"] for row in drive_rows)
        print(f"\n  {'ANCHOR':<24}{'DRIVES':>8}{'SHARE':>8}")
        for kind, n in by_anchor.most_common():
            print(f"  {kind:<24}{n:>8,}{_pct(n / len(drive_rows)):>8}")

        # A drive carrying dropped rows inside it is where two possessions
        # were most likely merged into one.
        merged = [row for row in drive_rows if row["n_dropped_inside"] > 1]
        print(f"\n  drives with >1 dropped row inside them: {len(merged):,}"
              f"  ({_pct(len(merged) / len(drive_rows))})")
        print("  Those are the ones to open first: a drive is a maximal run of")
        print("  one offensive team, so a possession change the feed never")
        print("  labelled is invisible except as cleaning noise inside a drive.")

    _print_market_coverage(plays)
    _print_anchor_field_check(drive_rows)

    if pair_rows:
        live = sum(1 for row in pair_rows if row["live"] == "live")
        print(f"\n  {len(pair_rows):,} pairs in dump_pairs.csv, {live:,} live "
              f"({_pct(live / len(pair_rows))})")

    if not drive_rows:
        return
    longest = sorted(drive_rows, key=lambda r: -r["n_plays"])[:5]
    print(f"\n  Longest drives (a merge shows up as an implausible play count):")
    print(f"  {'MATCH':<16}{'DRIVE':>6}{'PLAYS':>7}  {'TEAM':<12}{'ANCHOR':<12}")
    for row in longest:
        print(f"  {row['match_code']:<16}{row['drive_number']:>6}"
              f"{row['n_plays']:>7}  {str(row['offensive_team']):<12}"
              f"{row['anchor_kind']:<12}")


def print_anchor(report):
    """Where in its drive each snapshot landed."""
    print(f"\n{'=' * 78}\nSNAPSHOT ANCHOR -- did it land on the drive's "
          f"start?\n{'=' * 78}")
    if not report or not report["pairs"]:
        print("  No pairs.")
        return
    from . import drives
    titles = {drives.FIRST_DOWN: "opening 1st and 10",
              drives.MID_DRIVE: "mid-drive snap",
              drives.NO_SNAP: "no real snap"}
    print(f"  {_pct(report['share_clean'])} of {report['pairs']:,} pairs sat "
          f"on a drive's opening 1st and 10")
    print(f"  {report['off_anchor']:,} elsewhere, across "
          f"{report['matches']:,} matches")
    print(f"\n  {'ANCHOR':<22}{'PAIRS':>9}{'SHARE':>8}")
    for kind in (drives.FIRST_DOWN, drives.MID_DRIVE, drives.NO_SNAP):
        n = report["counts"].get(kind, 0)
        if not n:
            continue
        print(f"  {titles[kind]:<22}{n:>9,}{_pct(n / report['pairs']):>8}")
    print(f"\n  {'QUARTER':<10}{'PAIRS':>9}{'OFF ANCHOR':>12}{'SHARE':>8}")
    for key in sorted(report["by_quarter"]):
        total, off = report["by_quarter"][key]
        if not total:
            continue
        print(f"  {str(key):<10}{total:>9,}{off:>12,}{_pct(off / total):>8}")
    print("\n  A snapshot is meant to be a drive's opening 1st and 10. The")
    print("  general cleaning rule drops only ONE row per team change, so a")
    print("  transition carrying several kick rows leaves the rest behind.")
    print("  Anything off anchor means the drive's start was never found,")
    print("  and the score, possession and field position on that row")
    print("  describe a different moment -- on a row that still looks")
    print("  perfectly well-formed, which is why it is counted here.")


def print_market_state(report):
    """What non-live quotes cost, and which state they were in."""
    print(f"\n{'=' * 78}\nMARKET STATE -- were both quotes tradeable?\n{'=' * 78}")
    if not report or not report["pairs"]:
        print("  No pairs.")
        return
    if not report["not_live"]:
        print(f"  All {report['pairs']:,} pairs had both markets open and active.")
        return
    print(f"  {report['not_live']:,} of {report['pairs']:,} pairs "
          f"({_pct(report['share'])}) across {report['matches']:,} matches")
    print(f"    prod not live only      : {report['prod_only']:,}")
    print(f"    candidate not live only : {report['candidate_only']:,}")
    print(f"    both                    : {report['both']:,}")

    if report["by_state"]:
        print(f"\n  {'STREAM':<11}{'STATUS/ACTIVE':<26}{'PAIRS':>8}")
        for (stream, state), n in sorted(report["by_state"].items(),
                                         key=lambda kv: -kv[1]):
            print(f"  {stream:<11}{state:<26}{n:>8,}")

    print(f"\n  {'SPLIT':<9}{'BUCKET':<14}{'PAIRS':>9}{'DEAD':>8}{'SHARE':>8}")
    for label, table in (("quarter", report["by_quarter"]),
                         ("market", report["by_market"])):
        for key in sorted(table):
            total, dead = table[key]
            if not total:
                continue
            print(f"  {label:<9}{str(key):<14}{total:>9,}{dead:>8,}"
                  f"{_pct(dead / total):>8}")
    print("\n  These pairs are shown in the report and scored by nothing: a")
    print("  price nobody could have taken is not a price. Liveness is read")
    print("  from IS_ACTIVE alone -- GAMEPLAI say STATUS is wrong, so it is")
    print("  reported in the table beside it but gets no vote. Most non-live")
    print("  rows are post-match settlement that a drive-start snapshot")
    print("  should never land on. Anything here that is NOT post-match is")
    print("  the interesting case, and a lopsided split between the streams")
    print("  is the one to chase: those pairs cluster around scores, which")
    print("  is where the two models differ most.")


def print_selections(summary):
    """Every selection, both sides, with its own clustered test.

    The pooled per-market tables read one side and treat the other as its
    mirror, which is right for reading a result and wrong for checking
    one: a fault confined to one side averages away into a flat market
    row. This is where it would show.
    """
    print(f"\n{'=' * 104}\nBY SELECTION -- both sides of every market"
          f"\n{'=' * 104}")
    printed = False
    for key, label, metric, spec in (("same_line", "same line", "brier", "+.4f"),
                                     ("different_line", "diff line", "mae", "+.3f")):
        selections = summary.get(key, {}).get("selections", {})
        if not selections:
            continue
        if printed:
            print()
        printed = True
        delta = "BRIER_D" if metric == "brier" else "POINTS_D"
        print(f"  {'VIEW':<11}{'MARKET':<11}{'SEL':<7}{'USED':<6}{'ID':>4}"
              f"{'PAIRS':>8}{'MATCH':>7}{'WIN%':>8}{delta:>10}{'P':>8}")
        for market_id in sorted(selections, key=lambda m: (
                MARKET_PRINT_ORDER.index(selections[m]["market"]), m)):
            row = selections[market_id]
            tallied = row["tally"]
            if not tallied["n"]:
                continue
            clustered = row.get(metric, {})
            used = "yes" if row["canonical"] else ""
            print(f"  {label:<11}{row['market']:<11}{row['selection']:<7}"
                  f"{used:<6}{market_id:>4}{tallied['n']:>8,}"
                  f"{tallied['n_matches']:>7,}"
                  f"{_pct(tallied['candidate_win_rate']):>8}"
                  f"{_fmt(clustered.get('mean'), spec):>10}"
                  f"{_p(clustered.get('p_value')):>8}")
    if not printed:
        print("  No pairs.")
        return
    print("\n  USED marks the side the pooled tables read; the other is its")
    print("  mirror. Both are shown here because a fault on one side only --")
    print("  a line parsed for Over and not for Under, outcomes resolved the")
    print("  wrong way round -- averages away into a flat market row.")


def print_decisive(decisive):
    """The combined verdict: every pair judged on its own question."""
    print(f"\n{'=' * 78}\nOVERALL -- every pair on the question it actually "
          f"asked\n{'=' * 78}")
    if not decisive["n"]:
        print("  No pairs could be decided either way.")
        return
    votes = decisive["votes"]
    print(f"  pairs decided             : {decisive['n']:,} of "
          f"{decisive['n_offered']:,}")
    print(f"    settled on probability  : {decisive['settled_on_probability']:,}"
          f"   (both streams quoted the same line)")
    print(f"    settled on the line     : {decisive['settled_on_line']:,}"
          f"   (lines differ, so the line decides)")
    print(f"  pair wins cand / prod     : {decisive['candidate']:,} / "
          f"{decisive['prod']:,}   ({decisive['tie']:,} level)")
    print(f"  candidate win rate        : {_pct(decisive['candidate_win_rate'])}")
    print(f"  match vote cand / prod    : {votes['candidate']} / {votes['prod']}"
          f"   ({votes['tie']} level)   [{votes['n_matches']} matches]")
    print(f"  sign test p               : {_p(votes['p_value'])}   [match level]")
    print("\n  A different line overrules the probability, so a pair whose")
    print("  lines differ is decided on whose line landed nearer the result")
    print("  and its probabilities are not compared at all. Where the lines")
    print("  match, the probabilities answer the same question and decide.")
    print("\n  No mean error is shown here on purpose: a line error is in")
    print("  points and a probability error is not, so there is no average")
    print("  of the two to take. Only the per-pair decision survives the")
    print("  mix. That also makes this the WEAKER test -- it throws away how")
    print("  much closer each was. The two halves below keep that, so read")
    print("  this for direction and them for strength.")


def print_directional_breakdown(title, grouped, order=None,
                                label_width=SCORE_WIDTH,
                                mode="probability"):
    # In line mode the errors are points, so squaring them gives squared
    # points -- a number with no readable meaning. Show mean points instead.
    line_mode = mode == "line"
    second_label = "" if line_mode else "BRIER_D"
    first_label = "POINTS_D" if line_mode else "MAE_D"
    print(f"\n{'-' * 92}\n{title}\n{'-' * 92}")
    print(f"  {'GROUP':<{label_width}}{'PAIRS':>8}{'MATCH':>7}{'CAND':>7}{'PROD':>7}"
          f"{'TIE':>6}{'WIN%':>8}{first_label:>10}{second_label:>9}{'P':>9}")
    keys = order if order is not None else sorted(grouped)
    for key in keys:
        row = grouped.get(key)
        if not row or not row["n"]:
            continue
        label = key if isinstance(key, str) else " ".join(str(k) for k in key)
        first = _fmt(row['mae_delta'], '+.3f' if line_mode else '+.4f')
        second = "" if line_mode else _fmt(row['brier_delta'], '+.4f')
        print(f"  {label:<{label_width}}{row['n']:>8,}{row['n_matches']:>7,}"
              f"{row['candidate']:>7,}{row['prod']:>7,}{row['tie']:>6,}"
              f"{_pct(row['candidate_win_rate']):>8}"
              f"{first:>10}{second:>9}{_p(row['p_value']):>9}")
    unit = "points" if line_mode else "probability"
    print(f"\n  WIN% is the candidate's share of decisive pairs. Deltas are prod minus")
    print(f"  candidate in {unit}, so positive means the candidate is better.")


def pair_row(p):
    """One paired observation as a flat row.

    Shared with the dump so the two files cannot describe the same pair
    differently -- the dump widens this row rather than rebuilding it.
    """
    return {
        "publish_time": p.publish_time,
        "match_code": p.match_code,
        "drive_number": p.drive_number,
        "period_number": p.period_number,
        "score_p1": p.score_p1,
        "score_p2": p.score_p2,
        "score_diff": p.score_diff,
        "offensive_team": p.offensive_team,
        "field_position": p.field_position,
        "down_number": p.down_number,
        "distance": p.distance,
        "market_id": p.market_id,
        "message_count": p.message_count,
        "message_gap": p.message_gap,
        "prod_line": p.prod_line,
        "candidate_line": p.candidate_line,
        "line_delta": p.line_delta,
        "same_line": int(p.same_line),
        "prod_probability": p.prod_probability,
        "candidate_probability": p.candidate_probability,
        "prod_outcome": "" if p.prod_outcome is None else int(p.prod_outcome),
        "candidate_outcome": ("" if p.candidate_outcome is None
                              else int(p.candidate_outcome)),
        "realized": p.realized,
        "prod_error": p.prod_error,
        "candidate_error": p.candidate_error,
        "prod_line_error": p.prod_line_error,
        "candidate_line_error": p.candidate_line_error,
        "disagreement": p.disagreement,
        "probability_winner": p.winner("probability"),
        "line_winner": p.winner("line"),
        "prod_live": int(p.prod_live),
        "candidate_live": int(p.candidate_live),
        "prod_state": p.prod_state,
        "candidate_state": p.candidate_state,
    }


def write_pairs_csv(path, pairs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for p in pairs:
            writer.writerow(pair_row(p))
    print(f"\n  paired observations -> {path}")


def print_paired_delta(brier, mae):
    """The powerful half of the comparison: paired loss difference."""
    print(f"\n{'-' * 78}\nPaired error difference, averaged per match\n{'-' * 78}")
    print("  The win rate above discards magnitude. If both models are unbiased")
    print("  around the same truth and differ only in noise, whichever lands")
    print("  closer to the realized 0/1 is a coin flip regardless of which is")
    print("  actually better -- so a real improvement can sit at a ~50% win rate.")
    print("  This difference does see it, and clusters by match.\n")
    print(f"  {'LOSS':<10}{'MATCHES':>9}{'MEAN_DELTA':>12}{'SE':>10}"
          f"{'95% CI':>22}{'T':>8}{'P':>9}")
    for name, summary in (("Brier", brier), ("MAE", mae)):
        if summary["mean"] is None:
            continue
        ci = ("n/a" if summary["ci_low"] is None
              else f"[{summary['ci_low']:+.4f}, {summary['ci_high']:+.4f}]")
        print(f"  {name:<10}{summary['n_matches']:>9,}"
              f"{summary['mean']:>+12.4f}{_fmt(summary['se']):>10}"
              f"{ci:>22}{_fmt(summary['t'], '+.2f'):>8}{_p(summary['p_value']):>9}")

    print("\n  MEAN_DELTA is prod minus candidate, so positive favours the candidate.")
    print(f"  Matches favouring candidate: {brier['matches_favouring_candidate']:,}"
          f"   favouring prod: {brier['matches_favouring_prod']:,}"
          f"   (sign test p {_p(brier.get('sign_p_value'))})")
    print("  CI is a bootstrap over matches, so it carries the same clustering")
    print("  assumption as the point estimate. If it straddles zero, one day of")
    print("  data has not separated the two models.")


def print_line_agreement(lines):
    """Whether the two streams are even quoting the same question."""
    print(f"\n{'-' * 78}\nLine agreement\n{'-' * 78}")
    print("  If the candidate quotes a different line, its probability answers a")
    print("  different question, and only the line-closeness view compares fairly.")
    print(f"\n  pairs on the same line : {lines['same']:,} / {lines['n']:,} "
          f"({_pct(lines['same_rate'])})")
    if lines.get("differing_n"):
        print(f"  gap on the {lines['differing_n']:,} pairs that DIFFER: "
              f"median {lines['differing_median']:.2f}, "
              f"mean {lines['differing_mean']:.2f}, "
              f"max {lines['differing_max']:.2f}")
    print(f"\n  {'MARKET':<12}{'PAIRS':>8}{'SAME':>8}"
          f"{'PROD half/whole':>20}{'CAND half/whole':>20}")
    for group in ("moneyline", "spread", "total"):
        bucket = lines["by_market"].get(group)
        if not bucket or not bucket["n"]:
            continue
        same_rate = bucket["same"] / bucket["n"]
        print(f"  {group:<12}{bucket['n']:>8,}{100 * same_rate:>7.1f}%"
              f"{bucket['prod_half']:>12,} /{bucket['prod_whole']:>6,}"
              f"{bucket['candidate_half']:>12,} /{bucket['candidate_whole']:>6,}")
    print("\n  'whole' counts lines on an integer, which can push. A stream that")
    print("  only quotes whole numbers will rarely share a line with a .5 book.")


def print_block(title, subtitle, block):
    overall = block["overall"]
    brier = block["brier"]
    mae = block["mae"]
    votes = block["votes"]

    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"  {subtitle}")
    if not overall["n"]:
        print("\n  No comparable pairs here.")
        return

    print(f"\n  pairs {overall['n']:,} of {overall['n_offered']:,} offered"
          f"   across {overall['n_matches']:,} matches")
    print(f"  candidate closer {overall['candidate']:,}"
          f"   prod closer {overall['prod']:,}"
          f"   level {overall['tie']:,}"
          f"   win rate {_pct(overall['candidate_win_rate'])}")
    print(f"  match vote: candidate {votes['candidate']}, prod {votes['prod']}, "
          f"level {votes['tie']}  (p {_p(votes['p_value'])})")
    line_mode = overall.get("mode") == "line"
    if line_mode:
        # Errors here are in points, so squaring them gives squared points --
        # not a unit anyone can read. Mean absolute points leads instead.
        print(f"  paired delta per match: points {_fmt(mae['mean'], '+.3f')} "
              f"{_ci_text(mae, '+.3f')} p {_p(mae['p_value'])}")
        print("                          (positive = candidate's line landed closer,"
              " in points)")
        metric_label, metric_key, metric_spec = "POINTS_D", "mae_delta", "+.3f"
    else:
        print(f"  paired delta per match: Brier {_fmt(brier['mean'], '+.4f')} "
              f"{_ci_text(brier)} p {_p(brier['p_value'])}")
        print(f"                          MAE   {_fmt(mae['mean'], '+.4f')} "
              f"{_ci_text(mae)} p {_p(mae['p_value'])}")
        metric_label, metric_key, metric_spec = "BRIER_D", "brier_delta", "+.4f"

    print(f"\n  {'MARKET':<11}{'PAIRS':>7}{'MATCH':>6}{'WIN%':>7}{'VOTE':>9}"
          f"{metric_label:>10}{'95% CI':>22}{'P_CLUST':>9}{'P_ROW':>8}")
    for group in ("moneyline", "spread", "total"):
        row = block["by_market"].get(group)
        if not row or not row["n"]:
            continue
        detail = block.get("markets", {}).get(group, {})
        clustered = detail.get("mae" if line_mode else "brier", {})
        votes_m = detail.get("votes", {})
        vote_text = (f"{votes_m.get('candidate', 0)}-{votes_m.get('prod', 0)}"
                     if votes_m else "")
        ci = _ci_text(clustered, metric_spec) or ""
        print(f"  {group:<11}{row['n']:>7,}{row['n_matches']:>6,}"
              f"{_pct(row['candidate_win_rate']):>7}{vote_text:>9}"
              f"{_fmt(clustered.get('mean'), metric_spec):>10}{ci:>22}"
              f"{_p(clustered.get('p_value')):>9}{_p(row['p_value']):>8}")
    print("\n  P_CLUST is the match-clustered paired test -- the one to read.")
    print("  P_ROW is the row-level sign test, which treats every pair as")
    print("  independent and so overstates significance.")


def _ci_text(summary, spec="+.4f"):
    if summary.get("ci_low") is None:
        return ""
    return f"[{format(summary['ci_low'], spec)}, {format(summary['ci_high'], spec)}]"


# ---------------------------------------------------------------------------
# Cross-sectional cells, under the line rule
# ---------------------------------------------------------------------------

CROSS_FIELDS = [
    "axis", "cell", "market", "selection", "n", "matches", "realized",
    "prod_predicted", "prod_gap", "prod_brier",
    "candidate_predicted", "candidate_gap", "candidate_brier",
    "brier_delta", "ci_low", "ci_high", "p_value", "closer",
]


MARKET_PRINT_ORDER = ["moneyline", "spread", "total"]


def print_calibration_cells(title, cells, order=None, label_width=SCORE_WIDTH,
                            markets_shown=None):
    """Same-line cells, one selection per market so REALIZED can vary."""
    print(f"\n{'=' * 110}\n{title}\n{'=' * 110}")
    if not cells:
        print("  No same-line pairs in this cut.")
        return
    print("  One selection per market (moneyline Home, spread Home, total Over).")
    print("  Pooling both sides would force REAL and the predictions to 0.500 by")
    print("  construction, since the sides are complements.")
    print(f"\n  {'CELL':<{label_width}}{'MARKET':<11}{'SEL':<6}{'N':>6}{'MATCH':>6}"
          f"{'REAL':>8}{'PROD':>8}{'P_GAP':>8}{'CAND':>8}{'C_GAP':>8}"
          f"{'ΔBRIER':>9}{'P':>8}{'CLOSER':>7}")

    wanted = markets_shown or MARKET_PRINT_ORDER
    cell_labels = order if order is not None else sorted({k[0] for k in cells}, key=str)
    for cell_label in cell_labels:
        for market in wanted:
            row = cells.get((cell_label, market))
            if not row or not row["n"]:
                continue
            label = cell_label if isinstance(cell_label, str) else " ".join(
                str(k) for k in cell_label)
            closer = "cand" if row["winner"] == "candidate" else "prod"
            print(f"  {label:<{label_width}}{market:<11}{row['selection']:<6}"
                  f"{row['n']:>6,}{row['matches']:>6,}"
                  f"{_fmt(row['realized'], '.3f'):>8}"
                  f"{_fmt(row['prod_predicted'], '.3f'):>8}{_fmt(row['prod_gap'], '+.3f'):>8}"
                  f"{_fmt(row['candidate_predicted'], '.3f'):>8}"
                  f"{_fmt(row['candidate_gap'], '+.3f'):>8}"
                  f"{_fmt(row['brier_delta'], '+.4f'):>9}{_p(row['p_value']):>8}{closer:>7}")
    print("\n  P_GAP / C_GAP are realized minus predicted: positive means that stream")
    print("  underpriced the selection. CLOSER is whichever gap is smaller in size.")
    print("  ΔBRIER is prod minus candidate, match-clustered; positive favours the")
    print("  candidate. P is the clustered test.")


def print_line_cells(title, cells, order=None, label_width=SCORE_WIDTH):
    """Different-line cells: the line overrules the probability."""
    print(f"\n{'=' * 104}\n{title}\n{'=' * 104}")
    if not cells:
        print("  No different-line pairs in this cut.")
        return
    print("  Lines differ here, so the probabilities are not comparable and the")
    print("  line decides. Errors are in points from the actual margin or total.")
    print(f"\n  {'CELL':<{label_width}}{'N':>7}{'MATCH':>6}{'GAP':>7}"
          f"{'PROD_ERR':>10}{'CAND_ERR':>10}{'ΔPOINTS':>9}"
          f"{'95% CI':>20}{'P':>8}")
    keys = order if order is not None else sorted(cells, key=str)
    for key in keys:
        row = cells.get(key)
        if not row or not row["n"]:
            continue
        label = key if isinstance(key, str) else " ".join(str(k) for k in key)
        ci = ("" if row["ci_low"] is None
              else f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]")
        print(f"  {label:<{label_width}}{row['n']:>7,}{row['matches']:>6,}"
              f"{row['mean_line_gap']:>7.2f}"
              f"{row['prod_line_error']:>10.3f}{row['candidate_line_error']:>10.3f}"
              f"{_fmt(row['points_delta'], '+.3f'):>9}{ci:>20}{_p(row['p_value']):>8}")
    print("\n  GAP is the mean distance between the two lines. ΔPOINTS is prod minus")
    print("  candidate line error, so positive means the candidate's line was closer.")


def write_cross_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CROSS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  cross-sectional cells -> {path}")


def cross_rows(axis_name, cells):
    rows = []
    for key, row in cells.items():
        cell_label, market = key
        label = cell_label if isinstance(cell_label, str) else " ".join(
            str(k) for k in cell_label)
        rows.append({
            "axis": axis_name,
            "cell": label,
            "market": market,
            "selection": row["selection"],
            "n": row["n"],
            "matches": row["matches"],
            "realized": row["realized"],
            "prod_predicted": row["prod_predicted"],
            "prod_gap": row["prod_gap"],
            "prod_brier": row["prod_brier"],
            "candidate_predicted": row["candidate_predicted"],
            "candidate_gap": row["candidate_gap"],
            "candidate_brier": row["candidate_brier"],
            "brier_delta": row["brier_delta"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
            "p_value": row["p_value"],
            "closer": row["winner"],
        })
    return rows


def print_complement_report(report):
    """Are the two sides of each market really complements?"""
    print(f"\n{'=' * 96}\nAre the two sides of each market complements?\n{'=' * 96}")
    print("  Calibrating one side per market assumes the other adds nothing: if Over")
    print("  is overpriced by 11 points then Under is underpriced by 11. That holds")
    print("  only if the sides really do partition. This tests it.")
    print(f"\n  {'MARKET':<12}{'BOTH SIDES':>11}{'P_SUM MEAN':>12}{'MIN':>8}{'MAX':>8}"
          f"{'SAME LINE':>11}{'PARTITION':>11}{'BOTH WON':>10}{'BOTH LOST':>11}")
    for group in ("moneyline", "spread", "total"):
        row = report.get(group)
        if not row or not row["both_sides"]:
            continue
        n = row["both_sides"]
        print(f"  {group:<12}{n:>11,}"
              f"{_fmt(row['prob_sum_mean'], '.4f'):>12}"
              f"{_fmt(row['prob_sum_min'], '.3f'):>8}{_fmt(row['prob_sum_max'], '.3f'):>8}"
              f"{100 * row['line_equal'] / n:>10.1f}%"
              f"{100 * row['outcomes_partition'] / n:>10.1f}%"
              f"{row['both_won']:>10,}{row['both_lost']:>11,}")
    print("\n  P_SUM is the two sides' probabilities added. 1.0000 means a fair book")
    print("  with no overround, so each side is exactly the other's complement and")
    print("  one of them can be dropped without losing anything.")
    print("  PARTITION should be 100%: exactly one side wins. Anything less means")
    print("  the two are not opposite sides of one market, and BOTH WON / BOTH LOST")
    print("  count the cases -- which would invalidate the one-side shortcut.")


def print_both_sides(cells):
    """Every selection's calibration, as a mirroring check."""
    print(f"\n{'=' * 96}\nEvery selection, as a mirroring check\n{'=' * 96}")
    print("  Not an extra finding. If the sides are complements, each pair of rows")
    print("  should show REAL summing to 1.000 and gaps that are equal and opposite.")
    print(f"\n  {'MARKET':<11}{'SEL':<7}{'USED':>6}{'N':>7}{'MATCH':>6}"
          f"{'REAL':>8}{'PROD':>8}{'P_GAP':>8}{'CAND':>8}{'C_GAP':>8}")
    by_market = {}
    for market_id, row in sorted(cells.items()):
        by_market.setdefault(row["market"], []).append((market_id, row))
    for group in ("moneyline", "spread", "total"):
        rows = by_market.get(group, [])
        for _, row in rows:
            used = "yes" if row["canonical"] else "-"
            print(f"  {row['market']:<11}{row['selection']:<7}{used:>6}"
                  f"{row['n']:>7,}{row['matches']:>6,}"
                  f"{_fmt(row['realized'], '.3f'):>8}"
                  f"{_fmt(row['prod_predicted'], '.3f'):>8}{_fmt(row['prod_gap'], '+.3f'):>8}"
                  f"{_fmt(row['candidate_predicted'], '.3f'):>8}"
                  f"{_fmt(row['candidate_gap'], '+.3f'):>8}")
        if len(rows) == 2:
            realized_sum = sum(r["realized"] for _, r in rows if r["realized"] is not None)
            gap_sum = sum(r["prod_gap"] for _, r in rows if r["prod_gap"] is not None)
            verdict = "mirror ok" if abs(realized_sum - 1.0) < 1e-9 else "NOT MIRRORED"
            print(f"  {'':<11}{'sum':<7}{'':>6}{'':>7}{'':>6}"
                  f"{realized_sum:>8.3f}{'':>8}{gap_sum:>+8.3f}   <- {verdict}")
    print("\n  USED marks the selection the cross-sectional tables calibrate.")
    print("  A realized sum of 1.000 and gaps cancelling is the check passing.")


def print_spread_interpretation(report):
    """Which reading of the spread's second selection the data supports."""
    print(f"\n{'=' * 96}\nSpread: are the two selections yes/no on one proposition?\n{'=' * 96}")
    print('  52 reads "PLAYER 1 to score over L more than PLAYER 2".')
    print('  53 reads "PLAYER 2 to score over L more than PLAYER 1".')
    print("  Literal    = two propositions. Same L means they OVERLAP: both win on")
    print("               any margin between -L and +L.")
    print("  Complement = one proposition, 53 the NO of 52. Partitions by design.")

    n = report["n"]
    if not n:
        print("\n  No spread pairs carrying both sides.")
        return

    print(f"\n  spread markets with both sides : {n:,}")
    print(f"  probabilities summed           : mean {_fmt(report['prob_sum_mean'], '.4f')}"
          f"  min {_fmt(report['prob_sum_min'], '.3f')}"
          f"  max {_fmt(report['prob_sum_max'], '.3f')}")
    print(f"  the two sides' lines           : equal {100 * report['lines_equal'] / n:.1f}%"
          f"   mirrored {100 * report['lines_mirrored'] / n:.1f}%"
          f"   neither {100 * report['lines_other'] / n:.1f}%")
    print(f"  literal reading partitions     : {100 * report['literal_partition'] / n:.1f}%"
          f"   (both won {report['literal_both_won']:,},"
          f" both lost {report['literal_both_lost']:,})")
    print(f"  the two readings agree on 53   : {100 * report['readings_agree'] / n:.1f}%")

    verdict, explanation = report["verdict"]
    print(f"\n  VERDICT: {verdict.upper()}")
    for line in explanation.split(" -- "):
        print(f"    {line}")
    print(f"\n  Currently resolving as: {config.SPREAD_RESOLUTION}")
    if verdict != "unclear" and verdict != config.SPREAD_RESOLUTION:
        print(f"  !! That does not match. Set SPREAD_RESOLUTION = \"{verdict}\" in")
        print("     config.py and re-run; every spread number above is affected.")


def print_daily(daily):
    """Per-day view, for watching a growing window settle or drift."""
    print(f"\n{'=' * 96}\nBy day (same-line pairs)\n{'=' * 96}")
    print("  The window only grows, so the question as games accumulate is whether")
    print("  the answer is stable. A day out of step with its neighbours is worth a")
    print("  look before it gets averaged away.")
    if not daily:
        print("\n  No same-line pairs.")
        return
    print(f"\n  {'DAY':<12}{'PAIRS':>8}{'MATCH':>7}{'WIN%':>8}"
          f"{'ΔBRIER':>10}{'95% CI':>22}{'P':>8}")
    for day, row in sorted(daily.items()):
        brier = row["brier"]
        print(f"  {day:<12}{row['pairs']:>8,}{row['matches']:>7,}"
              f"{_pct(row['win_rate']):>8}{_fmt(brier.get('mean'), '+.4f'):>10}"
              f"{_ci_text(brier) or '':>22}{_p(brier.get('p_value')):>8}")
    means = [r["brier"].get("mean") for r in daily.values()
             if r["brier"].get("mean") is not None]
    if len(means) > 1:
        print(f"\n  spread of daily ΔBrier: {min(means):+.4f} to {max(means):+.4f}")
        signs = {m > 0 for m in means}
        print("  all days agree on direction." if len(signs) == 1
              else "  days disagree on direction -- the effect is not stable yet.")


def print_checks_summary(full):
    """One block saying whether the diagnostics passed, instead of four tables.

    The full tables are still a flag away; what matters on a routine run is
    whether anything needs looking at.
    """
    print(f"\n{'-' * 78}\nChecks\n{'-' * 78}")
    lines = full["summary"]["lines"]
    print(f"  same line            : {_pct(lines['same_rate'])} of pairs")

    issues = []
    for market in ("moneyline", "spread", "total"):
        row = full["complement"].get(market)
        if not row or not row["both_sides"]:
            continue
        both = row["both_sides"]
        partition = row["outcomes_partition"] / both
        state = "ok" if partition > 0.999 else f"ONLY {100*partition:.1f}%"
        if partition <= 0.999:
            issues.append(f"{market} sides do not partition ({100*partition:.1f}%)")
        print(f"  {market:<21}: P(both sides) {row['prob_sum_mean']:.4f}"
              f"   partition {state}")

    verdict = full["spread"]["verdict"][0]
    flag = "" if verdict in ("unclear", config.SPREAD_RESOLUTION) else "  <-- MISMATCH"
    print(f"  spread reading       : {verdict}"
          f" (configured {config.SPREAD_RESOLUTION}){flag}")
    if flag:
        issues.append(f"spread reads {verdict}, configured {config.SPREAD_RESOLUTION}")

    mirrors = {}
    for market_id, row in full["both_sides"].items():
        mirrors.setdefault(row["market"], []).append(row["realized"])
    for market, values in mirrors.items():
        if len(values) == 2 and abs(sum(values) - 1.0) > 1e-9:
            issues.append(f"{market} realized does not mirror")
    print(f"  mirror check         : "
          f"{'ok' if not any('mirror' in i for i in issues) else 'BROKEN'}")

    print(f"\n  {'ALL PASS' if not issues else 'NEEDS ATTENTION:'}")
    for issue in issues:
        print(f"    - {issue}")
    if not issues:
        print("  Run with --axes for the full integrity tables and single-axis views.")


# ---------------------------------------------------------------------------
# In-drive reaction
# ---------------------------------------------------------------------------

def _rate_row(label, b, width=22):
    ci = ("" if b["ci_low"] is None
          else f"[{_pct(b['ci_low'])}, {_pct(b['ci_high'])}]")
    prob = "" if b["mean_signed"] is None else f"{b['mean_signed']:+.4f}"
    line = ("" if b["mean_line_signed"] is None
            else f"{b['mean_line_signed']:+.2f}")
    print(f"  {label:<{width}}{b['n']:>8,}{b['decided']:>9,}"
          f"{_pct(b['rate']):>8}{ci:>18}{_pct(b['flat_share']):>8}"
          f"{b['n_prob']:>8,}{prob:>10}{b['n_line']:>8,}{line:>8}"
          f"{_p(b['p_value']):>9}")


def _rate_header(label, width=22):
    print(f"\n  {label:<{width}}{'MOVES':>8}{'DECIDED':>9}{'RIGHT':>8}"
          f"{'95% CI':>18}{'FLAT':>8}{'ON PROB':>8}{'SIGNED':>10}"
          f"{'ON LINE':>8}{'SIGNED':>8}{'P':>9}")


def print_indrive_census(census, transitions):
    """What the play feed actually produced, before any price is read."""
    print(f"\n{'=' * 78}\nIN-DRIVE REACTION -- what the plays were\n{'=' * 78}")
    total = len(transitions)
    if not total:
        print("  No transitions.")
        return
    in_drive = sum(1 for t in transitions if t.in_drive)
    scorable = sum(1 for t in transitions if t.scorable)
    print(f"  {total:,} transitions across "
          f"{len({t.match_code for t in transitions}):,} matches")
    print(f"    inside a drive        : {in_drive:,}")
    print(f"    at a drive's end      : {total - in_drive:,}")
    print(f"    carrying a direction  : {scorable:,} "
          f"({_pct(scorable / total)})")

    print(f"\n  {'OUTCOME':<22}{'SIGN':>6}{'N':>9}{'IN DRIVE':>10}"
          f"{'MATCHES':>9}{'SHARE':>8}")
    for outcome, row in census.items():
        if not row["n"]:
            continue
        sign = {1: "good", -1: "bad", 0: "--"}[row["sign"]]
        print(f"  {outcome:<22}{sign:>6}{row['n']:>9,}{row['in_drive']:>10,}"
              f"{row['matches']:>9,}{_pct(row['n'] / total):>8}")
    print("\n  A failed third down usually ENDS the drive, so it is not an")
    print("  in-drive transition. Restricted to plays inside one possession")
    print("  the sample leans towards things going well, which is why the")
    print("  drive-ending rows are built and reported apart.")


def print_drive_outcomes(census, outcomes, stats=None):
    """What each drive produced, and whether the points add up."""
    from . import indrive
    print(f"\n{'=' * 78}\nIN-DRIVE REACTION -- what each drive produced"
          f"\n{'=' * 78}")
    if not outcomes:
        print("  No drives.")
        return
    matches = len({o.match_code for o in outcomes})
    points = sum(o.points_for + o.points_against for o in outcomes)
    print(f"  {len(outcomes):,} drives across {matches:,} matches "
          f"({len(outcomes) / matches:.1f} per match) carrying {points:,} points")

    print(f"\n  {'OUTCOME':<18}{'DRIVES':>8}{'SHARE':>8}{'POINTS':>9}"
          f"{'PTS/DRIVE':>11}{'PLAYS':>8}{'MATCHES':>9}")
    for outcome, row in census.items():
        if not row["n"]:
            continue
        plays = "" if row["mean_plays"] is None else f"{row['mean_plays']:.1f}"
        print(f"  {outcome:<18}{row['n']:>8,}{_pct(row['n'] / len(outcomes)):>8}"
              f"{row['points']:>9,}{row['points'] / row['n']:>11.2f}"
              f"{plays:>8}{row['matches']:>9,}")

    ended = collections.Counter(o.ended for o in outcomes)
    print(f"\n  {'HOW IT ENDED':<18}{'DRIVES':>8}{'SHARE':>8}")
    for kind in (indrive.HANDOVER, indrive.SAME_TEAM, indrive.MATCH_END):
        n = ended.get(kind, 0)
        if n:
            print(f"  {kind:<18}{n:>8,}{_pct(n / len(outcomes)):>8}")

    scored = [o for o in outcomes if o.points_for or o.points_against]
    print(f"\n  drives that produced points: {len(scored):,} "
          f"({_pct(len(scored) / len(outcomes))})")
    print(f"  points per drive overall   : {points / len(outcomes):.2f}")
    if stats and stats.get("match_points_do_not_reconcile"):
        print(f"  matches whose points do NOT reconcile: "
              f"{stats['match_points_do_not_reconcile']:,}")
    else:
        print("  every match's drive points reconcile with the feed")


def print_indrive(result, stats=None):
    """Did the price move the way the play says it should?"""
    from . import indrive
    print(f"\n{'=' * 78}\nIN-DRIVE REACTION -- did the price follow the "
          f"play?\n{'=' * 78}")
    if not result["moves"]:
        print("  No scorable moves.")
        return
    print(f"  {result['moves']:,} moves from {result['transitions']:,} "
          f"transitions across {result['matches']:,} matches")
    print(f"  RIGHT is the share of moves that went the expected way, out of")
    print(f"  the ones that moved at all. FLAT is the share that did not move.")
    print(f"  A move is scored ON PROB where the line held and ON LINE where")
    print(f"  it moved; the two SIGNED columns are in their own units and are")
    print(f"  never pooled. Probability points on the left, line points right.")

    for stream in (directional_prod(), directional_candidate()):
        s = result["streams"][stream]
        _rate_header(stream.upper())
        for label, key in (("overall", "overall"), ("inside a drive", "in_drive"),
                           ("at a drive's end", "ending")):
            if s[key]["n"]:
                _rate_row(label, s[key])

    for stream in (directional_prod(), directional_candidate()):
        s = result["streams"][stream]
        _rate_header(f"{stream.upper()} BY OUTCOME")
        for outcome in indrive.OUTCOME_ORDER:
            b = s["by_outcome"].get(outcome)
            if b and b["n"]:
                _rate_row(outcome, b)

        _rate_header(f"{stream.upper()} BY PERIOD")
        for period in sorted(s["by_period"]):
            _rate_row(str(period), s["by_period"][period])

        _rate_header(f"{stream.upper()} BY BASIS")
        for basis in sorted(s["by_basis"], key=str):
            _rate_row(str(basis), s["by_basis"][basis])

        _rate_header(f"{stream.upper()} BY MARKET")
        for market in sorted(s["by_market"], key=str):
            _rate_row(str(market), s["by_market"][market])

        _rate_header(f"{stream.upper()} BY SELECTION")
        for key in sorted(s["by_selection"], key=lambda k: (str(k[0]), str(k[1]))):
            _rate_row(f"{key[0]} {key[1]}", s["by_selection"][key])
        print("  Two sides of one market read the same wherever the feed")
        print("  publishes exact complements: one goes up, the other down,")
        print("  and the expectations mirror. A row that does NOT match its")
        print("  partner is the interesting one.")

    print(f"\n  {'HEAD TO HEAD':<22}{'PAIRS':>8}{'DECIDED':>9}{'CAND':>8}"
          f"{'95% CI':>18}{'TIES':>8}{'BOTH FLAT':>11}{'P':>9}")
    for label, key in (("overall", "all"), ("inside a drive", "in_drive"),
                       ("at a drive's end", "ending")):
        h = result["head_to_head"][key]
        if not h["pairs"]:
            continue
        ci = ("" if h["ci_low"] is None
              else f"[{_pct(h['ci_low'])}, {_pct(h['ci_high'])}]")
        print(f"  {label:<22}{h['pairs']:>8,}{h['decided']:>9,}"
              f"{_pct(h['rate']):>8}{ci:>18}{h['ties']:>8,}"
              f"{h['both_flat']:>11,}{_p(h['p_value']):>9}")
    print("\n  Head to head is paired on (transition, selection), so both")
    print("  streams answer the same question about the same play. CAND is")
    print("  the candidate's share of the moves the two disagreed on.")

    if stats:
        print(f"\n  {'WHERE THE MOVES WENT':<28}{'N':>10}")
        for key in ("transition_no_direction", "move_missing_quote",
                    "move_not_live", "move_line_unreadable",
                    "move_scored_on_prob", "move_scored_on_line"):
            if stats.get(key):
                print(f"  {key:<28}{stats[key]:>10,}")


def print_indrive_checks(findings):
    """What the run says about itself, before anything is read off it."""
    from . import indrive
    print(f"\n{'=' * 78}\nIN-DRIVE REACTION -- checks\n{'=' * 78}")
    if not findings:
        print("  All pass.")
        return
    order = {indrive.ERROR: 0, indrive.WARN: 1, indrive.NOTE: 2}
    labels = {indrive.ERROR: "ERROR", indrive.WARN: "WARN", indrive.NOTE: "note"}
    import textwrap
    for severity, subject, message in sorted(
            findings, key=lambda f: (order[f[0]], f[1])):
        lines = textwrap.wrap(message, 50) or [""]
        print(f"  {labels[severity]:<6}{subject:<20}{lines[0]}")
        for line in lines[1:]:
            print(f"  {'':<26}{line}")
    print("\n  An ERROR here is about the CHECK, not the models: two models")
    print("  built separately do not agree with each other in the wrong")
    print("  direction across thousands of plays. Read it as the expected")
    print("  direction being backwards for that class.")


def directional_prod():
    from . import directional
    return directional.PROD


def directional_candidate():
    from . import directional
    return directional.CANDIDATE


# ---------------------------------------------------------------------------
# Pre-match calibration
# ---------------------------------------------------------------------------

def _prematch_cell_header(label, width=26):
    print(f"\n  {label:<{width}}{'N':>7}{'MATCH':>7}{'REAL':>8}{'PROD':>8}"
          f"{'P_GAP':>8}{'CAND':>8}{'C_GAP':>8}{'DBRIER':>9}"
          f"{'P':>8}{'CLOSER':>8}")


def _prematch_cell_row(label, c, width=26):
    print(f"  {str(label):<{width}}{c['n']:>7,}{c['matches']:>7,}"
          f"{_fmt(c['realized'], '.3f'):>8}{_fmt(c['prod_predicted'], '.3f'):>8}"
          f"{_fmt(c['prod_gap'], '+.3f'):>8}"
          f"{_fmt(c['candidate_predicted'], '.3f'):>8}"
          f"{_fmt(c['candidate_gap'], '+.3f'):>8}"
          f"{_fmt(c['brier_delta'], '+.4f'):>9}{_p(c['p_value']):>8}"
          f"{str(c['closer'] or ''):>8}")


def print_prematch(summary, stats=None):
    """Were the closing prices calibrated against what happened?"""
    print(f"\n{'=' * 104}\nPRE-MATCH CALIBRATION -- the closing price"
          f"\n{'=' * 104}")
    if not summary:
        print("  No pre-match quotes resolved.")
        return
    print(f"  {summary['n']:,} closing prices across {summary['matches']:,} "
          f"matches -> {summary['pairs']:,} pairs both streams closed on "
          f"the same line")
    if summary["line_differs"]:
        print(f"  {summary['line_differs']:,} dropped: the two streams closed "
              f"on DIFFERENT lines, so they resolved against different "
              f"outcomes")
    if summary["one_stream_only"]:
        print(f"  {summary['one_stream_only']:,} dropped: only one stream "
              f"closed on them")

    # Before any calibration number: is there a price here at all?
    spread = summary["spread"]
    print(f"\n  HOW FAR FROM EVEN MONEY THE CLOSING PRICES SIT")
    print(f"  {'|p - 0.500|':<16}{'N':>9}{'SHARE':>8}")
    for band in spread["bands"]:
        label = f"{band['low']:.2f} to {band['high']:.2f}"
        print(f"  {label:<16}{band['n']:>9,}{_pct(band['share']):>8}")
    print(f"  {'mean':<16}{'':>9}{spread['mean_distance']:>8.3f}")
    print(f"  {'max':<16}{'':>9}{spread['max_distance']:>8.3f}")

    print(f"\n  {'MARKET':<12}{'N':>9}{'MEAN |p-.5|':>13}{'MAX':>8}"
          f"{'WITHIN .02':>12}")
    for group, s in summary["spread_by_market"].items():
        within = s["bands"][0]["share"] if s["bands"] else None
        print(f"  {str(group):<12}{s['n']:>9,}{s['mean_distance']:>13.3f}"
              f"{s['max_distance']:>8.3f}{_pct(within):>12}")

    _prematch_cell_header("SELECTION")
    for key in sorted(summary["by_selection"], key=lambda k: (str(k[0]), str(k[1]))):
        _prematch_cell_row(f"{key[0]} {key[1]}", summary["by_selection"][key])

    if summary["by_line"]:
        _prematch_cell_header("SELECTION x LINE")
        for key in sorted(summary["by_line"],
                          key=lambda k: (str(k[0]), str(k[1]),
                                         k[2] if k[2] is not None else 0)):
            label = f"{key[0]} {key[1]} {key[2]:+g}"
            _prematch_cell_row(label, summary["by_line"][key])

    h = summary["head_to_head"]
    print(f"\n  {'HEAD TO HEAD':<18}{'PAIRS':>8}{'MATCHES':>9}{'DBRIER':>10}"
          f"{'95% CI':>22}{'CAND':>7}{'PROD':>7}{'P':>9}")
    ci = ("" if h["ci_low"] is None
          else f"[{h['ci_low']:+.4f}, {h['ci_high']:+.4f}]")
    print(f"  {'closing price':<18}{h['pairs']:>8,}{h['matches']:>9,}"
          f"{_fmt(h['mean'], '+.4f'):>10}{ci:>22}"
          f"{h['matches_favouring_candidate']:>7,}"
          f"{h['matches_favouring_prod']:>7,}{_p(h['p_value']):>9}")
    print("\n  P_GAP / C_GAP are realized minus predicted: positive means that")
    print("  stream underpriced the selection. DBRIER is prod minus candidate,")
    print("  match-clustered; positive favours the candidate. Read the bands")
    print("  above the tables first: a book that never leaves 50/50 has no")
    print("  view to be calibrated, and its Brier sits at 0.25 regardless.")

    if stats:
        print(f"\n  {'DROPPED':<26}{'N':>10}")
        for key in ("prematch_no_final", "prematch_no_quotes",
                    "prematch_push", "prematch_not_live",
                    "prematch_line_differs", "prematch_one_stream_only"):
            if stats.get(key):
                print(f"  {key:<26}{stats[key]:>10,}")
