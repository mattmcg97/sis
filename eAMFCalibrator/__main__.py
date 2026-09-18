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
import os
import sys

from . import config, directional, html_report, pipeline, report, snowflake_io


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


def build_parser():
    parser = argparse.ArgumentParser(prog="eAMFCalibrator", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="check tables, clock column and window volume")

    run_parser = sub.add_parser("run", help="calibrate one stream, or both")
    run_parser.add_argument("stream", choices=sorted(config.STREAMS) + ["both"])
    run_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    dir_parser = sub.add_parser(
        "directional",
        help="paired head-to-head: which model is closer to the result at each snapshot")
    dir_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    dir_parser.add_argument("--html", help="path for the one-screen HTML report")

    cmp_parser = sub.add_parser("compare", help="diff two cell-summary CSVs")
    cmp_parser.add_argument("file_a")
    cmp_parser.add_argument("file_b")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        return cmd_preflight(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "directional":
        return cmd_directional(args)
    if args.command == "compare":
        return cmd_compare(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
