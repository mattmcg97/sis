"""CLI for the eAMF calibration suite.

    py -m eAMFCalibrator preflight
    py -m eAMFCalibrator directional
    py -m eAMFCalibrator indrive
    py -m eAMFCalibrator run prod
    py -m eAMFCalibrator run candidate
    py -m eAMFCalibrator run both
    py -m eAMFCalibrator compare out/prod_cells.csv out/candidate_cells.csv

Start with preflight. It confirms the tables this suite depends on and,
critically, whether the play feed carries a usable clock -- the 3-second
snapshot-to-quote match cannot be done without one, and no existing script
in the repo has ever needed that column.
"""

import argparse
import datetime as dt
import os
import sys

from . import (buckets, config, directional, dump, html_full, html_indrive,
               html_report, indrive, pipeline,
               report, snowflake_io)


DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "out")

PREFLIGHT_TABLES = [
    snowflake_io.PLAY_TABLE,
    snowflake_io.SCORE_TABLE,
    snowflake_io.FINAL_TABLE,
    snowflake_io.EVENT_TABLE,
]


def cmd_preflight(args):
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            print(f"Database: {config.DATABASE}.{config.SCHEMA}")
            for table in PREFLIGHT_TABLES:
                columns = snowflake_io.describe_columns(cur, table)
                print(f"\n{'=' * 70}\n{table}  ({len(columns)} columns)\n{'=' * 70}")
                if not columns:
                    print("  !! NOT FOUND")
                    continue
                for name, dtype, nullable, pos in columns:
                    print(f"  {pos:>3}  {name:<38} {dtype:<18} nullable={nullable}")

            # Who is PLAYER_1? Three columns claim to name the two sides,
            # in what may or may not be the same vocabulary.
            print(f"\n{'=' * 70}\nWho is PLAYER_1?\n{'=' * 70}")
            print(f"  {'SOURCE':<22}{'VALUE':<26}{'ROWS':>12}{'MATCHES':>9}")
            for src, value, rows, matches in snowflake_io.team_vocabulary(
                    cur, config.STREAMS[directional.PROD]):
                print(f"  {src:<22}{str(value)[:25]:<26}{rows:>12,}{matches:>9,}")

            print("\n  Rows per (match, market, message) -- the pipeline"
                  " assumed one:")
            print(f"  {'ID':>4}{'MESSAGES':>11}{'MEAN':>7}{'MAX':>5}"
                  f"{'MULTI-ROW':>11}{'MIXED LIVE/DEAD':>17}")
            for market_id, messages, mean_rows, max_rows, multi, mixed in \
                    snowflake_io.rows_per_message(
                        cur, config.STREAMS[directional.PROD]):
                print(f"  {market_id:>4}{messages:>11,}{float(mean_rows):>7.2f}"
                      f"{max_rows:>5}{multi:>11,}{mixed:>17,}")
            print("  MIXED is the case that costs pairs: the message offered")
            print("  a tradeable quote AND a dead one, so which row the index")
            print("  kept decided whether the pair could be scored at all.")

            print("\n  Market descriptions, one sample per market:")
            print(f"  {'ID':>4}  {'FORMS':>6}  {'ROWS':>10}  SAMPLE")
            for market_id, forms, sample, n in snowflake_io.market_descriptions(
                    cur, config.STREAMS[directional.PROD]):
                print(f"  {market_id:>4}  {forms:>6,}  {n:>10,}  {str(sample)[:60]}")

            print("\n  Do a match's play-feed teams match its EVENT teams?")
            print(f"  {'PLAY TEAMS':<28}{'P1_TEAM':<18}{'P2_TEAM':<18}{'MATCHES':>8}")
            for play_teams, p1, p2, matches in snowflake_io.team_join_test(
                    cur, config.STREAMS[directional.PROD]):
                print(f"  {str(play_teams)[:27]:<28}{str(p1)[:17]:<18}"
                      f"{str(p2)[:17]:<18}{matches:>8,}")
            print("\n  If the play feed's vocabulary and EVENT's overlap, the")
            print("  PLAYER_1 = Home Team mapping can be read per match instead")
            print("  of assumed. If they do not, it stays positional and the")
            print("  market descriptions above are the evidence for it.")

            time_column, _ = snowflake_io.detect_play_time_column(cur)
            print(f"\n{'=' * 70}\nPlay clock\n{'=' * 70}")
            if time_column:
                print(f"  Using {snowflake_io.PLAY_TABLE}.{time_column} as the snapshot timestamp.")
            else:
                print(f"  {snowflake_io.PLAY_TABLE} carries no TIMESTAMP column, as expected.")
                clock_key = config.CLOCK_SOURCE or "the calibrated stream"
                print(f"  Snapshot times will be reconstructed from the {clock_key} stream's")
                print("  EVENT_MESSAGE_COUNT -> PUBLISH_TIME map: exact where the stream quoted")
                print(f"  that message, interpolated across gaps up to {config.MAX_BRACKET_MESSAGES}")
                print("  messages wide, dropped beyond that. Every run reports the split.")

            for key, table in sorted(config.STREAMS.items()):
                n_rows, n_matches, first, last = snowflake_io.stream_window_summary(cur, table)
                print(f"\n{key:<10} {table}")
                print(f"  in window from {config.CUTOFF_START}: {n_rows:,} rows, {n_matches:,} matches")
                print(f"  span: {first}  ->  {last}")

                # What STATUS and IS_ACTIVE really contain. Only
                # IS_ACTIVE decides -- STATUS is reported beside it so the
                # two can be checked against each other.
                profile = snowflake_io.status_profile(cur, table)
                print(f"  {'STATUS':<14}{'ACTIVE':<9}{'ROWS':>12}{'MATCHES':>9}"
                      f"{'NULL P':>8}{'ZERO P':>8}  live?")
                for status, active, rows, matches, null_p, zero_p in profile:
                    live = "live" if directional.is_live(active) else "SUSPENDED"
                    print(f"  {str(status):<14}{str(active):<9}{rows:>12,}"
                          f"{matches:>9,}{null_p or 0:>8.1%}{zero_p or 0:>8.1%}  {live}")
    finally:
        conn.close()
    return 0


def cmd_dump(args):
    """Write the drive-detection working out to CSV for inspection.

    Reconciling drives is a row-by-row job, so this writes the rows rather
    than a summary, in two files. One play-by-play carrying every play the
    feed sent, the cleaning rule that fired on it, the drive it landed in,
    the score at that message and what both streams were quoting on all
    six selections; and the pairs those snapshots became.
    """
    out_dir = args.out or DEFAULT_OUT
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            time_column, _ = snowflake_io.detect_play_time_column(cur)
            if args.match:
                match_codes = list(args.match)
            else:
                prod = set(snowflake_io.match_universe(cur, config.STREAMS[directional.PROD]))
                candidate = set(snowflake_io.match_universe(
                    cur, config.STREAMS[directional.CANDIDATE]))
                match_codes = sorted(prod & candidate)[-args.matches:]
            print(f"\nDumping drive detection for {len(match_codes)} matches")
            written, plays, drive_rows, pair_rows = dump.run(
                cur, match_codes, out_dir, time_column)
    finally:
        conn.close()

    report.print_dump_summary(plays, drive_rows, pair_rows)
    print()
    for path in written:
        size = os.path.getsize(path) / 1024
        print(f"  wrote {path}  ({size:,.0f} KB)")
    if len(written) < dump.FILES:
        print(f"  {dump.FILES - len(written)} file(s) could not be written")
        return 1
    return 0


def run_one(stream_key, out_dir):
    print(f"\nRunning calibration for stream: {stream_key}")
    observations, stats, header, handle_scan = pipeline.run(stream_key)

    report.print_header(header, stats)
    report.print_handle_check(handle_scan)
    if not observations:
        print("\n  No observations produced. Nothing matched inside the tolerance.")
        return None

    report.print_overall(pipeline.overall_summary(observations))
    report.print_reliability(observations)

    cells, cell_matches = pipeline.aggregate(observations)
    rows = report.cell_rows(stream_key, cells, cell_matches)
    report.print_cells(rows)
    report.print_worst(rows)

    report.write_cells_csv(os.path.join(out_dir, f"{stream_key}_cells.csv"), rows)
    report.write_observations_csv(os.path.join(out_dir, f"{stream_key}_observations.csv"), observations)
    return rows


def cmd_run(args):
    out_dir = args.out or DEFAULT_OUT
    targets = sorted(config.STREAMS) if args.stream == "both" else [args.stream]

    results = {}
    for stream_key in targets:
        rows = run_one(stream_key, out_dir)
        if rows is None:
            return 1
        results[stream_key] = rows

    if len(results) == 2:
        # prod is the baseline and candidate the challenger, so a positive
        # |A|-|B| reads as "the candidate improved this cell".
        order = ["prod", "candidate"]
        a, b = [k for k in order if k in results] or sorted(results)
        report.print_comparison(results[a], results[b], a, b)
        print("\n  !! This comparison does NOT apply the line rule: each stream was")
        print("     calibrated on its own quotes, so a cell can mix pairs where the")
        print("     two streams priced different lines. Each stream's OWN calibration")
        print("     above is sound; it is the side-by-side that is loose.")
        print("     Use `cross` for the comparison restricted to same-line pairs.")
    return 0


def cmd_compare(args):
    rows_a = report.read_cells_csv(args.file_a)
    rows_b = report.read_cells_csv(args.file_b)
    label_a = rows_a[0]["stream"] if rows_a else os.path.basename(args.file_a)
    label_b = rows_b[0]["stream"] if rows_b else os.path.basename(args.file_b)
    report.print_comparison(rows_a, rows_b, label_a, label_b)
    return 0


def cmd_directional(args):
    """Paired head-to-head, which is what a single day of data can answer."""
    out_dir = args.out or DEFAULT_OUT
    print("\nPairing prod against candidate at each drive-start snapshot")
    pairs, stats, header, handle_scan = directional.run()

    report.print_directional_header(header, stats)
    # Before any comparison: is the PLAYER_1 frame the buckets are read in
    # the same frame all the way through each match?
    report.print_handle_check(handle_scan)
    if not pairs:
        print("\n  No paired observations. Nothing to compare.")
        return 1

    summary = directional.build_summary(pairs)

    report.print_line_agreement(summary["lines"])
    report.print_decisive(summary["decisive"])
    report.print_anchor(directional.anchor_report(pairs))
    report.print_market_state(directional.market_state_report(pairs))
    report.print_selections(summary)

    report.print_block(
        "SAME LINE -- whose probability was closer to its own 0/1",
        "Both streams quoted the same line, so the probabilities answer the "
        "same question. This is the clean comparison.",
        summary["same_line"])

    report.print_block(
        "DIFFERENT LINE -- whose line was closer to what happened",
        "Lines differ, so the probabilities are not comparable. Scored on "
        "which line landed nearer the actual margin or total.",
        summary["different_line"])

    report.print_block(
        "DIFFERENT LINE -- each probability against its own line (secondary)",
        "Fair, but it measures line choice and probability together, so read "
        "the line-closeness view above first.",
        summary["different_line_probability"])

    report.print_directional_breakdown(
        "By line gap (line-closeness mode)",
        summary["by_line_delta"],
        order=directional.LINE_DELTA_ORDER,
        mode=directional.LINE,
    )

    same_pairs, _ = directional.split_by_line(pairs)
    if same_pairs:
        report.print_directional_breakdown(
            "Same-line pairs by quarter",
            directional.group_by(same_pairs, _time_label),
            order=["Q1", "Q2", "Q3", "Q4", "OT", "unknown"],
        )
        report.print_directional_breakdown(
            "Same-line pairs by score difference",
            directional.group_by(same_pairs, _score_label),
            order=[label for _, _, label in config.SCORE_DIFF_BUCKETS],
        )

    report.write_pairs_csv(os.path.join(out_dir, "directional_pairs.csv"), pairs)

    html_path = args.html or os.path.join(out_dir, "directional.html")
    html_report.write(html_path, summary, header, stats)
    print(f"  one-screen report  -> {html_path}")
    return 0


def markets_label(pair):
    from . import markets as m
    return f"{m.market_group(pair.market_id)} {m.selection_label(pair.market_id)}"


def _score_label(pair):
    from . import buckets
    return buckets.score_diff_bucket(pair.score_diff)


def _time_label(pair):
    from . import buckets
    return buckets.time_bucket(pair.period_number, pair.drive_number)


def _possession_label(pair):
    from . import buckets
    return buckets.possession_bucket(pair.offensive_team)


def cmd_indrive(args):
    """Does the price move the right way when a play goes well?

    Every pair of consecutive cleaned play rows is classified from the
    play feed alone -- first down, touchdown, five yards, a stop -- and
    each class carries an expected direction for the side in possession.
    Both streams' probabilities are then read at the two messages and
    scored on their SIGN.
    """
    out_dir = args.out or DEFAULT_OUT
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            time_column, _ = snowflake_io.detect_play_time_column(cur)
            if args.match:
                match_codes = list(args.match)
            else:
                prod = set(snowflake_io.match_universe(
                    cur, config.STREAMS[directional.PROD]))
                candidate = set(snowflake_io.match_universe(
                    cur, config.STREAMS[directional.CANDIDATE]))
                match_codes = sorted(prod & candidate)
                if args.matches:
                    match_codes = match_codes[-args.matches:]
            print(f"\nIn-drive reaction across {len(match_codes)} matches")
            transitions, moves, stats = indrive.run(
                cur, match_codes, time_column, n_bootstrap=args.bootstrap)
    finally:
        conn.close()

    report.print_indrive_census(indrive.outcome_census(transitions), transitions)
    if not moves:
        print("\n  No scorable price moves.")
        return 1

    result = indrive.report(moves, n_bootstrap=args.bootstrap)
    report.print_indrive(result, stats)

    written = [
        _write_csv(os.path.join(out_dir, "indrive_moves.csv"),
                   indrive.MOVE_FIELDS,
                   [indrive.move_row(m) for m in moves]),
        _write_csv(os.path.join(out_dir, "indrive_transitions.csv"),
                   indrive.TRANSITION_FIELDS,
                   [indrive.transition_row(t) for t in transitions]),
    ]
    path = args.html or os.path.join(out_dir, "indrive.html")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html_indrive.render(
            result, indrive.outcome_census(transitions), transitions, stats))
    written.append(path)

    print()
    for item in written:
        print(f"  wrote {item}  ({os.path.getsize(item) / 1024:,.0f} KB)")
    return 0


def _write_csv(path, fields, rows):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def cmd_cross(args):
    """Cross-sectional calibration by cell, under the line rule.

    Same pairing as `directional`, but the output is the cell view: for each
    score-difference / quarter / possession bucket, both streams' predicted
    against the shared realized rate. Probability is only used where the two
    streams quoted the SAME line; where lines differ the line decides, and
    those pairs get their own table.
    """
    out_dir = args.out or DEFAULT_OUT
    print("\nPairing prod against candidate at each drive-start snapshot")
    pairs, stats, header, handle_scan = directional.run()

    report.print_directional_header(header, stats)
    # Before any comparison: is the PLAYER_1 frame the buckets are read in
    # the same frame all the way through each match?
    report.print_handle_check(handle_scan)
    if not pairs:
        print("\n  No paired observations. Nothing to compare.")
        return 1

    lines = directional.line_agreement(pairs)
    report.print_line_agreement(lines)

    same, different = directional.split_by_line(pairs)
    print(f"\n  Probability comparison uses the {len(same):,} same-line pairs.")
    print(f"  The {len(different):,} different-line pairs are judged on the line instead.")

    # Before any cell table: is the one-side-per-market shortcut sound?
    report.print_complement_report(directional.complement_report(pairs))
    report.print_spread_interpretation(directional.spread_interpretation_report(pairs))
    report.print_both_sides(directional.both_sides_calibration(pairs))

    csv_rows = []
    for axis_name, key_function, order_factory in directional.CROSS_AXES:
        cells = directional.calibration_cells(pairs, key_function)
        report.print_calibration_cells(
            f"SAME LINE by {axis_name.lower()} -- predicted vs realized",
            cells, order=order_factory())
        csv_rows.extend(report.cross_rows(axis_name, cells))

        line_view = directional.line_cells(pairs, key_function)
        report.print_line_cells(
            f"DIFFERENT LINE by {axis_name.lower()} -- whose line was closer",
            line_view, order=order_factory())

    # The three axes crossed, which is the cell the calibrator was built for.
    # The three axes crossed. Moneyline only on screen -- three markets per
    # cell would be 130+ rows, and moneyline is the market with no line to
    # complicate it. The CSV carries all three.
    full = directional.calibration_cells(pairs, directional.cell_key)
    cell_order = sorted({k[0] for k in full}, key=buckets.sort_key)
    report.print_calibration_cells(
        "SAME LINE by full cell (score x quarter x possession) -- moneyline only",
        full, order=cell_order, label_width=report.CELL_WIDTH,
        markets_shown=["moneyline"])
    csv_rows.extend(report.cross_rows("full cell", full))

    report.write_cross_csv(os.path.join(out_dir, "cross_cells.csv"), csv_rows)
    return 0


def cmd_report(args):
    """One pairing pass, both console views, and one combined HTML file.

    Everything is built from directional.build_full_report() so the
    directional numbers, the cross-sectional cells and the per-pair table at
    the bottom cannot disagree with each other.
    """
    out_dir = args.out or DEFAULT_OUT
    print("\nPairing prod against candidate at each drive-start snapshot")
    pairs, stats, header, handle_scan = directional.run()

    report.print_directional_header(header, stats)
    # Before any comparison: is the PLAYER_1 frame the buckets are read in
    # the same frame all the way through each match?
    report.print_handle_check(handle_scan)
    if not pairs:
        print("\n  No paired observations. Nothing to compare.")
        return 1

    full = directional.build_full_report(pairs)
    summary = full["summary"]

    # --- 0. the combined verdict, before either half of it ---
    report.print_decisive(summary["decisive"])

    # --- 1. directional calibration ---
    report.print_block(
        "DIRECTIONAL -- same line, whose probability was closer to its own 0/1",
        "Both streams quoted the same line, so the probabilities answer the "
        "same question. This is the clean comparison.",
        summary["same_line"])
    report.print_block(
        "DIRECTIONAL -- different line, whose line was closer",
        "Lines differ, so the probabilities are not comparable. Scored on "
        "which line landed nearer the actual margin or total.",
        summary["different_line"])

    # --- 2. cross-section calibration: the three axes as one bucket ---
    report.print_calibration_cells(
        "CROSS-SECTION -- score difference x quarter x possession",
        full["full_cell"], order=full["full_cell_order"],
        label_width=report.CELL_WIDTH)

    csv_rows = report.cross_rows("full cell", full["full_cell"])
    for axis in full["axes"]:
        csv_rows.extend(report.cross_rows(axis["name"], axis["probability"]))

    # --- 3. checks, compact unless asked for in full ---
    report.print_checks_summary(full)
    report.print_daily(full["daily"])
    if args.axes:
        report.print_line_agreement(summary["lines"])
        report.print_anchor(full["anchor"])
        report.print_market_state(full["market_state"])
        report.print_selections(summary)
        report.print_complement_report(full["complement"])
        report.print_spread_interpretation(full["spread"])
        report.print_both_sides(full["both_sides"])
        for axis in full["axes"]:
            report.print_calibration_cells(
                f"{axis['name']} -- predicted vs realized",
                axis["probability"], order=axis["order"])
            report.print_line_cells(
                f"{axis['name']} -- whose line was closer",
                axis["line"], order=axis["order"])

    # --- outputs ---
    report.write_cross_csv(os.path.join(out_dir, "cross_cells.csv"), csv_rows)
    report.write_pairs_csv(os.path.join(out_dir, "directional_pairs.csv"), pairs)

    ordered = directional.sorted_pairs_by_disagreement(pairs)
    html_path = args.html or os.path.join(out_dir, "eamf_report.html")
    html_full.write(html_path, full, header, stats, ordered, handle_scan)
    size = os.path.getsize(html_path) / 1024 ** 2
    print(f"  combined report    -> {html_path}  ({size:.1f} MB, "
          f"{len(ordered):,} pair rows)")
    return 0


def common_options():
    """Options every command shares, so nothing needs a config.py edit.

    The window default stays anchored to when the candidate changed, because
    quotes before that instant came out of the OLD candidate and would
    pollute the comparison. --days is for slicing a rolling window on top of
    that; it does not replace the anchor's purpose.
    """
    parent = argparse.ArgumentParser(add_help=False)
    window = parent.add_argument_group("window")
    window.add_argument("--since", metavar="TS",
                        help="window start, 'YYYY-MM-DD HH:MM:SS' "
                             f"(default {config.CUTOFF_START})")
    window.add_argument("--until", metavar="TS",
                        help="window end (default: latest data available)")
    window.add_argument("--days", type=float, metavar="N",
                        help="rolling window of the last N days, overrides --since")
    tuning = parent.add_argument_group("tuning")
    tuning.add_argument("--sport", help=f"sport code (default {config.SPORT_CODE})")
    tuning.add_argument("--tolerance", type=float, metavar="SEC",
                        help="snapshot-to-quote match tolerance in seconds "
                             f"(default {config.MATCH_TOLERANCE_SECONDS})")
    tuning.add_argument("--message-gap", type=int, metavar="N",
                        help="widest message offset allowed when pairing "
                             f"(default {config.MAX_PAIR_MESSAGE_GAP})")
    tuning.add_argument("--clock", choices=sorted(config.STREAMS) + ["self"],
                        help=f"stream supplying the snapshot clock "
                             f"(default {config.CLOCK_SOURCE})")
    tuning.add_argument("--spread-resolution", choices=["literal", "complement"],
                        help=f"how to resolve the spread's second selection "
                             f"(default {config.SPREAD_RESOLUTION})")
    tuning.add_argument("--chunk", type=int, metavar="N",
                        help="matches fetched per batch, lower it if memory is "
                             f"tight on a long window (default {config.MATCH_CHUNK_SIZE})")
    tuning.add_argument("--time-axis", choices=["period", "drive"],
                        help=f"time axis for the cells (default {config.TIME_AXIS})")
    tuning.add_argument("--drop-flipped", action="store_true",
                        help="drop matches whose PLAYER_1 / PLAYER_2 handles "
                             "swap sides; the default reports them instead")
    return parent


def apply_overrides(args):
    """Push CLI options into config, and report the window actually used."""
    if getattr(args, "days", None):
        start = dt.datetime.utcnow() - dt.timedelta(days=args.days)
        config.CUTOFF_START = start.strftime("%Y-%m-%d %H:%M:%S")
    elif getattr(args, "since", None):
        config.CUTOFF_START = args.since
    if getattr(args, "until", None):
        config.CUTOFF_END = args.until

    for attribute, key in (("sport", "SPORT_CODE"),
                           ("tolerance", "MATCH_TOLERANCE_SECONDS"),
                           ("message_gap", "MAX_PAIR_MESSAGE_GAP"),
                           ("spread_resolution", "SPREAD_RESOLUTION"),
                           ("time_axis", "TIME_AXIS"),
                           ("chunk", "MATCH_CHUNK_SIZE")):
        value = getattr(args, attribute, None)
        if value is not None:
            setattr(config, key, value)

    if getattr(args, "drop_flipped", False):
        config.EXCLUDE_FLIPPED_MATCHES = True

    clock = getattr(args, "clock", None)
    if clock is not None:
        config.CLOCK_SOURCE = None if clock == "self" else clock

    if getattr(args, "command", None) == "compare":
        return
    end = config.CUTOFF_END or "latest available"
    print(f"\nWindow: {config.CUTOFF_START}  ->  {end}"
          f"   sport {config.SPORT_CODE}"
          f"   spread {config.SPREAD_RESOLUTION}")


def build_parser():
    parser = argparse.ArgumentParser(prog="eAMFCalibrator", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    shared = common_options()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", parents=[shared], help="check tables, clock column and window volume")

    run_parser = sub.add_parser("run", parents=[shared], help="calibrate one stream, or both")
    run_parser.add_argument("stream", choices=sorted(config.STREAMS) + ["both"])
    run_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    dir_parser = sub.add_parser(
        "directional", parents=[shared],
        help="paired head-to-head: which model is closer to the result at each snapshot")
    dir_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    dir_parser.add_argument("--html", help="path for the one-screen HTML report")

    report_parser = sub.add_parser(
        "report", parents=[shared],
        help="everything in one go: directional + cross-sectional + one HTML "
             "with every pair listed")
    report_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    report_parser.add_argument("--html", help="path for the combined HTML report")
    report_parser.add_argument("--axes", action="store_true",
                               help="also print the integrity tables and the "
                                    "single-axis breakdowns")

    cross_parser = sub.add_parser(
        "cross", parents=[shared],
        help="cross-sectional calibration by score diff / quarter / possession, "
             "under the line rule")
    cross_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    dump_parser = sub.add_parser(
        "dump", parents=[shared],
        help="write the drive-detection working out to CSV for inspection")
    dump_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    dump_parser.add_argument("--match", action="append", metavar="CODE",
                             help="dump this match; repeatable. Without it, "
                                  "the most recent --matches are used")
    dump_parser.add_argument("--matches", type=int, default=3, metavar="N",
                             help="how many recent matches to dump (default 3)")

    indrive_parser = sub.add_parser(
        "indrive", parents=[shared],
        help="does the price move the right way when a play goes well for "
             "the offence?")
    indrive_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    indrive_parser.add_argument("--html", help="path for the HTML report")
    indrive_parser.add_argument("--match", action="append", metavar="CODE",
                                help="analyse this match; repeatable. Without "
                                     "it, every match in both streams is used")
    indrive_parser.add_argument("--matches", type=int, default=0, metavar="N",
                                help="limit to the most recent N matches "
                                     "(default 0, meaning all of them)")
    indrive_parser.add_argument("--bootstrap", type=int, default=1000,
                                metavar="N",
                                help="bootstrap resamples for the intervals "
                                     "(default 1000)")

    cmp_parser = sub.add_parser("compare", parents=[shared], help="diff two cell-summary CSVs")
    cmp_parser.add_argument("file_a")
    cmp_parser.add_argument("file_b")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    apply_overrides(args)
    if args.command == "preflight":
        return cmd_preflight(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "directional":
        return cmd_directional(args)
    if args.command == "cross":
        return cmd_cross(args)
    if args.command == "dump":
        return cmd_dump(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "indrive":
        return cmd_indrive(args)
    if args.command == "compare":
        return cmd_compare(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
