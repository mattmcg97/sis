"""CLI for the eAMF calibration suite.

    py -m eAMFCalibrator preflight
    py -m eAMFCalibrator directional
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

from . import (buckets, config, directional, html_full, html_report, pipeline,
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
    finally:
        conn.close()
    return 0


def run_one(stream_key, out_dir):
    print(f"\nRunning calibration for stream: {stream_key}")
    observations, stats, header = pipeline.run(stream_key)

    report.print_header(header, stats)
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
    pairs, stats, header = directional.run()

    report.print_directional_header(header, stats)
    if not pairs:
        print("\n  No paired observations. Nothing to compare.")
        return 1

    summary = directional.build_summary(pairs)

    report.print_line_agreement(summary["lines"])

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
    pairs, stats, header = directional.run()

    report.print_directional_header(header, stats)
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
        full, order=cell_order, label_width=26, markets_shown=["moneyline"])
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
    pairs, stats, header = directional.run()

    report.print_directional_header(header, stats)
    if not pairs:
        print("\n  No paired observations. Nothing to compare.")
        return 1

    full = directional.build_full_report(pairs)
    summary = full["summary"]

    # --- directional half ---
    report.print_line_agreement(summary["lines"])
    report.print_complement_report(full["complement"])
    report.print_spread_interpretation(full["spread"])
    report.print_both_sides(full["both_sides"])
    report.print_daily(full["daily"])

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

    # --- cross-sectional half ---
    csv_rows = []
    for axis in full["axes"]:
        report.print_calibration_cells(
            f"SAME LINE by {axis['name'].lower()} -- predicted vs realized",
            axis["probability"], order=axis["order"])
        report.print_line_cells(
            f"DIFFERENT LINE by {axis['name'].lower()} -- whose line was closer",
            axis["line"], order=axis["order"])
        csv_rows.extend(report.cross_rows(axis["name"], axis["probability"]))

    report.print_calibration_cells(
        "SAME LINE by full cell (score x quarter x possession) -- moneyline only",
        full["full_cell"], order=full["full_cell_order"], label_width=26,
        markets_shown=["moneyline"])
    csv_rows.extend(report.cross_rows("full cell", full["full_cell"]))

    # --- outputs ---
    report.write_cross_csv(os.path.join(out_dir, "cross_cells.csv"), csv_rows)
    report.write_pairs_csv(os.path.join(out_dir, "directional_pairs.csv"), pairs)

    ordered = directional.sorted_pairs_by_disagreement(pairs)
    html_path = args.html or os.path.join(out_dir, "eamf_report.html")
    html_full.write(html_path, full, header, stats, ordered)
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

    cross_parser = sub.add_parser(
        "cross", parents=[shared],
        help="cross-sectional calibration by score diff / quarter / possession, "
             "under the line rule")
    cross_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

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
    if args.command == "report":
        return cmd_report(args)
    if args.command == "compare":
        return cmd_compare(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
