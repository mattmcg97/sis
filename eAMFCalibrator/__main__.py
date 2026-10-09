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

from . import (buckets, config, directional, drives, dump, html_full, html_indrive,
               html_report, indrive, labels, pipeline, prematch,
               report, scouting, snowflake_io)


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


def cmd_scouting(args):
    """Investigate SCOUTING_FULL against GAMEPLAI_STREAM, then export the
    PLAY_OVER snapshots GAMEPLAI quoted. Last 30 days unless told otherwise."""
    out_dir = args.out or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            table = scouting.locate(cur, args.scouting_table)
            if not args.no_probe:
                lines = scouting.probe(cur, table, [])
                path = os.path.join(out_dir, "scouting_probe.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
                print(f"\n  probe -> {path}")
            if args.sample:
                recent = scouting.scouting_matches(cur, table)[-args.sample:]
                path = os.path.join(out_dir, "scouting_sample.csv")
                n = scouting.write_sample(cur, table, recent, path)
                print(f"  sample: {n:,} rows of {len(recent)} matches -> {path}")
            if not args.no_export:
                scouting.export(cur, table, os.path.join(out_dir, "scouting_playover.csv"),
                                limit=args.limit)
    finally:
        conn.close()
    return 0


def cmd_scouting_check(args):
    """Is SCOUTING_FULL complete? Each day and match from --since to --until (default today),
    against a baseline of the days before --break."""
    from . import scouting_check
    brk = dt.date.fromisoformat(args.break_day)
    until = (dt.date.fromisoformat(args.until[:10]) if args.until
             else dt.datetime.utcnow().date())
    if args.days:
        since = until - dt.timedelta(days=int(args.days))
    elif args.since:
        since = dt.date.fromisoformat(args.since[:10])
    else:
        since = brk - dt.timedelta(days=scouting_check.BASELINE_DAYS)
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            table = scouting.locate(cur, args.scouting_table)
            scouting_check.run(cur, table, since, until, brk, args.out or DEFAULT_OUT)
    finally:
        conn.close()
    return 0


def cmd_timeouts(args):
    """Every timeout in SCOUTING_FULL over the window: who called it, when, at what score, and
    how many each side had left at the end of each half -- what the model's timeout windows are
    set from."""
    from . import timeouts
    out_dir = args.out or DEFAULT_OUT
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            table = scouting.locate(cur, args.scouting_table)
            timeouts.run(cur, table, out_dir)
    finally:
        conn.close()
    return 0


def cmd_drive_audit(args):
    """Every drive the play feed yields in the window, checked, and each
    one's end compared with SCOUTING_FULL where it can be reached."""
    import csv
    from collections import defaultdict
    from . import drive_audit
    out_dir = args.out or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    rows, issues = [], {}
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            time_column, _ = snowflake_io.detect_play_time_column(cur)
            codes = sorted(snowflake_io.match_universe(cur, config.STREAMS["prod"]))
            if args.limit:
                codes = codes[-args.limit:]
            print(f"  {len(codes):,} matches", flush=True)
            scouting_by = {}
            if not args.no_scouting:
                try:
                    scouting_by, _ = snowflake_io._play_over_snapshots(cur, codes)
                except SystemExit as exc:
                    print(f"  SCOUTING_FULL not used: {exc}")
            chunk = config.MATCH_CHUNK_SIZE
            for start in range(0, len(codes), chunk):
                batch = codes[start:start + chunk]
                plays_by, scores_by = defaultdict(list), defaultdict(list)
                for r in snowflake_io.fetch_plays(cur, batch, time_column):
                    plays_by[r[0]].append(drives.PlayRow(*r[1:8]))
                for r in snowflake_io.fetch_scores(cur, batch):
                    scores_by[r[0]].append(drives.ScoreRow(*r[1:7]))
                for code in batch:
                    got, problems = drive_audit.audit_match(
                        code, plays_by.get(code, []), scores_by.get(code, []),
                        scouting_by.get(code))
                    rows.extend(got)
                    issues[code] = problems
    finally:
        conn.close()
    path = os.path.join(out_dir, "drive_audit.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, drive_audit.FIELDS)
        w.writeheader()
        w.writerows(rows)
    drive_audit.print_summary(drive_audit.summarise(rows, issues, len(issues)))
    print(f"\n  every drive, with its flags -> {path}")
    return 0


def cmd_expected_points(args):
    """Rebuild the in-drive analysis's expected-points table off a
    scouting_playover.csv export."""
    from . import expected_points
    table, counts = expected_points.build_from_export(args.export)
    path = expected_points.write(table, counts, f"SCOUTING_FULL PLAY_OVER export {args.export}")
    print(f"  {len(table)} cells from {sum(counts.values()):,} scrimmage states -> {path}")
    return 0


def cmd_bets(args):
    """The betting simulation: every in-play bet in the window with prod's and the candidate's
    probability at the moment it was priced (`probe` first checks the names it reads)."""
    from . import bets
    out_dir = args.out or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    if args.action == "moments":
        from . import bet_moments
        path = args.csv or os.path.join(out_dir, "bets_sim.csv")
        rows = bet_moments.from_csv(path)
        print(f"  {len(rows):,} bets from {path}")
        if args.players:
            info = bet_moments.match_info(args.players)
            bet_moments.add_players(rows, info)
            print(f"  gamers and teams for {sum('home_player' in r for r in rows):,} bets from {args.players}")
        keep = {"in-play": None, "pre-match": lambda r: bet_moments.settled(r) and not r.get("in_play"),
                "all": bet_moments.settled}[args.scope]
        if args.by:
            print(f"  bets: {args.scope}")
            for spec in args.by:
                cols = [c.strip() for c in spec.split(",") if c.strip()]
                print("\n".join(bet_moments.cross(rows, cols, args.min_bets, keep)))
        else:
            print("\n".join(bet_moments.report([("prod", rows)], args.min_bets, candidates=False)))
        return 0
    if getattr(args, "sharp", None):
        config.SIM_SHARP_GROUPS = tuple(g.strip() for g in args.sharp.split(",") if g.strip())
    for attribute, key in (("elasticity", "SIM_ELASTICITY"), ("max_scale", "SIM_MAX_SCALE"),
                           ("boot", "SIM_BOOT")):
        if getattr(args, attribute, None) is not None:
            setattr(config, key, getattr(args, attribute))
    if args.bet_table:
        config.BET_TABLE = args.bet_table
    if getattr(args, "gamer", None):
        config.GAMERS = [g.strip() for g in args.gamer if g.strip()]
    if args.gap is not None:
        config.MAX_SCOUTING_GAP = args.gap
    if args.max_lag is not None:
        config.LAG_RANGE = (config.LAG_RANGE[0], args.max_lag)
    if args.extra_columns:
        config.BET_EXTRA_COLUMNS = [c.strip().upper() for c in args.extra_columns.split(",") if c.strip()]
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            if args.action == "lag":
                from . import bet_lag
                paths = bet_lag.run(cur, out_dir)
                print(f"\n  -> {os.path.join(out_dir, 'bets_lag.csv')}, {paths[0]} and one page a day")
            elif args.action == "lines":
                from . import bet_lines
                path, _ = bet_lines.run(cur, out_dir, n_matches=args.matches, only=args.match)
                print(f"\n  -> {path}")
            elif args.action == "sessions":
                from . import bet_sessions
                print(f"\n  -> {bet_sessions.run(cur, out_dir)}")
            elif args.action == "favourite":
                from . import bet_favourite
                path = bet_favourite.run(cur, out_dir, min_bets=args.min_bets)
                if path:
                    print(f"\n  -> {path} and bets_favourite.txt")
            elif args.action == "totals-signals":
                from . import totals_signals
                path = totals_signals.run(cur, out_dir, react=args.react)
                print(f"\n  -> {path} and totals_signals.txt")
            elif args.action == "totals-moves":
                from . import bet_totals
                paths = bet_totals.run(cur, out_dir, min_bets=args.min_bets,
                                       all_operators=args.all_operators)
                if paths:
                    print(f"\n  -> {paths[0]} and {paths[1]}")
            elif args.action == "probe":
                lines = bets.probe(cur)
                path = os.path.join(out_dir, "bets_probe.txt")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + "\n")
                print("\n".join(lines))
                print(f"\n  probe -> {path}")
            elif args.action == "prematch":
                bets.run(cur, out_dir, prematch_only=True)
                print(f"\n  -> {os.path.join(out_dir, 'bets_prematch_sim.csv')} and bets_prematch.txt")
            elif args.action == "check":
                bets.run(cur, out_dir, only_checks=True)
                print(f"\n  -> {os.path.join(out_dir, 'bets_checks.csv')}")
            else:
                bets.run(cur, out_dir)
                print(f"\n  -> {os.path.join(out_dir, 'bets_sim.csv')} and bets_latency.csv")
    finally:
        conn.close()
    return 0


def cmd_history(args):
    """Every settled match before --until (default: now), shaped like
    nb2/AMFELO.csv: what v4's own pre-match model (NB2) is fitted on."""
    import csv
    out_dir = args.out or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            rows = snowflake_io.fetch_history(cur, until=config.CUTOFF_END)
    finally:
        conn.close()
    path = os.path.join(out_dir, "match_history.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, snowflake_io.HISTORY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    first = rows[0]["SCHEDULED_START_TIME_UTC"] if rows else "-"
    last = rows[-1]["SCHEDULED_START_TIME_UTC"] if rows else "-"
    print(f"\n  {len(rows):,} settled {config.SPORT_CODE} matches, {first} -> {last}: {path}")
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

    html_path = args.html or os.path.join(out_dir, labels.default_report_name("directional.html"))
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
            transitions, moves, outcomes, stats = indrive.run(
                cur, match_codes, time_column, n_bootstrap=args.bootstrap)
    finally:
        conn.close()

    report.print_indrive_census(indrive.outcome_census(transitions), transitions)
    report.print_drive_outcomes(indrive.drive_census(outcomes), outcomes, stats)
    if not moves:
        print("\n  No scorable price moves.")
        return 1

    result = indrive.report(moves, n_bootstrap=args.bootstrap)
    report.print_indrive(result, stats)
    findings = indrive.checks(result, transitions, moves, outcomes)
    report.print_indrive_checks(findings)

    written = [
        _write_csv(os.path.join(out_dir, "indrive_moves.csv"),
                   indrive.MOVE_FIELDS,
                   [indrive.move_row(m) for m in moves]),
        _write_csv(os.path.join(out_dir, "indrive_transitions.csv"),
                   indrive.TRANSITION_FIELDS,
                   [indrive.transition_row(t) for t in transitions]),
        _write_csv(os.path.join(out_dir, "indrive_drives.csv"),
                   indrive.DRIVE_FIELDS,
                   [indrive.drive_row(o) for o in outcomes]),
    ]
    path = args.html or os.path.join(out_dir, labels.default_report_name("indrive.html"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(labels.relabel(html_indrive.render(
            result, indrive.outcome_census(transitions), transitions, stats,
            findings, indrive.drive_census(outcomes), outcomes)))
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
    """One pairing pass per candidate, the console view of each, and one
    combined HTML file with every candidate beside prod.

    Every table is built from directional.build_full_report() on a common
    population (multi.build), so the sections, the candidates and the
    per-pair table cannot disagree with each other.
    """
    from . import multi
    out_dir = args.out or DEFAULT_OUT
    candidates = config.CANDIDATES or [config.STREAMS["candidate"]]
    sides = multi.run(candidates)
    first = sides[0]["line"]

    report.print_directional_header(first["header"], first["stats"])
    # Before any comparison: is the PLAYER_1 frame the buckets are read in
    # the same frame all the way through each match?
    report.print_handle_check(first["scan"])
    if not any(s["line"]["pairs"] for s in sides):
        print("\n  No paired observations. Nothing to compare.")
        return 1

    dropped = multi.build(sides)
    for side in sides:
        full, prob = side["line_full"], side["prob_full"]
        if len(sides) > 1:
            print(f"\n{'=' * 78}\n{side['name']} against prod\n{'=' * 78}")
        report.print_decisive(full["summary"]["decisive"])
        report.print_prematch(side["line"]["prematch"], side["line"]["stats"])
        report.print_block(
            "DIRECTIONAL -- at prod's line, whose probability was closer to its own 0/1",
            "", prob["summary"]["same_line"])
        report.print_block(
            "DIRECTIONAL -- different line, whose line was closer",
            "", full["summary"]["different_line"])
        report.print_calibration_cells(
            "CROSS-SECTION -- score difference x quarter x possession",
            prob["full_cell"], order=prob["full_cell_order"],
            label_width=report.CELL_WIDTH)
        report.print_checks_summary(full)
        report.print_daily(prob["daily"])
        if args.axes:
            report.print_line_agreement(full["summary"]["lines"])
            report.print_anchor(full["anchor"])
            report.print_market_state(full["market_state"])
            report.print_selections(prob["summary"])
            report.print_complement_report(full["complement"])
            report.print_spread_interpretation(full["spread"])
            report.print_both_sides(prob["both_sides"])
            for axis in prob["axes"]:
                report.print_calibration_cells(
                    f"{axis['name']} -- predicted vs realized",
                    axis["probability"], order=axis["order"])
            for axis in full["axes"]:
                report.print_line_cells(
                    f"{axis['name']} -- whose line was closer",
                    axis["line"], order=axis["order"])

        # --- outputs, one set per candidate ---
        suffix = "" if len(sides) == 1 else f"_{_safe(side['name'])}"
        csv_rows = report.cross_rows("full cell", prob["full_cell"])
        for axis in prob["axes"]:
            csv_rows.extend(report.cross_rows(axis["name"], axis["probability"]))
        report.write_cross_csv(os.path.join(out_dir, f"cross_cells{suffix}.csv"), csv_rows)
        report.write_pairs_csv(os.path.join(out_dir, f"directional_pairs{suffix}.csv"),
                               side["line_pairs"])
        _write_csv(os.path.join(out_dir, f"prematch_closing{suffix}.csv"), prematch.FIELDS,
                   side["line"]["prematch_rows"])
    extra = ""
    if getattr(args, "bets", False):
        from . import bets
        print(f"\n{'=' * 78}\nBetting simulation\n{'=' * 78}")
        summary = {}
        conn = snowflake_io.get_connection()
        try:
            with conn.cursor() as cur:
                bets.run(cur, out_dir, summary=summary)
        finally:
            conn.close()
        extra = bets.html_section(summary)
    html_path = args.html or os.path.join(out_dir, _report_name(sides))
    html_full.write_sides(html_path, sides, dropped, extra)
    size = os.path.getsize(html_path) / 1024 ** 2
    print(f"  combined report    -> {html_path}  ({size:.1f} MB, "
          f"{len(sides[0]['line_pairs']):,} pair rows)")
    return 0


def _safe(name):
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name)


def _report_name(sides):
    """eamf_report.html, or eamf_report_v4_v5.html when named models stand in."""
    if len(sides) == 1:
        return labels.default_report_name("eamf_report.html")
    return "eamf_report_" + "_".join(_safe(s["name"]) for s in sides) + ".html"


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
    tuning.add_argument("--snapshots", choices=["drive", "play_over"],
                        help="what prod and the candidate are paired at: one snapshot per drive "
                             "start (play table), or every SCOUTING_FULL PLAY_OVER, before the "
                             f"next play starts (default {config.SNAPSHOTS})")
    tuning.add_argument("--time-axis", choices=["period", "drive"],
                        help=f"time axis for the cells (default {config.TIME_AXIS})")
    tuning.add_argument("--candidate", metavar="STREAM",
                        help="what stands in the candidate's place: a table name, "
                             "or a model version (v8 to v13 -- see eAMFModel) priced "
                             "live off SCOUTING_FULL's PLAY_OVER snapshots and its build; "
                             "several, comma-separated (v8,v9), set each against prod "
                             "side by side in the report; a second build of a version is "
                             "NAME=DIR, e.g. v9,v9-glmer=v9_glmer_917 "
                             f"(default {config.STREAMS['candidate']})")
    tuning.add_argument("--candidate-label", metavar="NAME",
                        help="what the HTML reports call the candidate (default: the "
                             "model version when one stands in, e.g. v9; else 'candidate')")
    tuning.add_argument("--v8-model", metavar="DIR",
                        help="eAMFModel v8-build's output, for --candidate v8 "
                             "(default $EAMF_V8_MODEL, then ./v8_model)")
    tuning.add_argument("--v8-paths", type=int, metavar="N",
                        help=f"games simulated per snapshot for --candidate v8 "
                             f"(default {config.V8_PATHS})")
    tuning.add_argument("--v8-lines", choices=["own", "even", "prod"],
                        help=f"--candidate v8: its own lines (in the gap between the key numbers), the"
                             f" even line or prod's (default {config.V8_LINES})")
    tuning.add_argument("--v9-model", metavar="DIR",
                        help="eAMFModel v9-build's output, for --candidate v9 "
                             "(default $EAMF_V9_MODEL, then ./v9_model)")
    tuning.add_argument("--v9-paths", type=int, metavar="N",
                        help=f"games simulated per snapshot for --candidate v9 "
                             f"(default {config.V9_PATHS})")
    tuning.add_argument("--v9-lines", choices=["own", "even", "prod"],
                        help=f"--candidate v9: its own lines (in the gap between the key numbers), the"
                             f" even line or prod's (default {config.V9_LINES})")
    tuning.add_argument("--v10-model", metavar="DIR",
                        help="eAMFModel v10-build's output, for --candidate v10 "
                             "(default $EAMF_V10_MODEL, then ./v10_model)")
    tuning.add_argument("--v10-paths", type=int, metavar="N",
                        help=f"games simulated per snapshot for --candidate v10 "
                             f"(default {config.V10_PATHS})")
    tuning.add_argument("--v10-lines", choices=["own", "even", "prod", "anchored", "hyst"],
                        help="--candidate v10: its own lines (in the gap between the key numbers), the"
                             " even line, the even line held until far off (hyst), prod's, or prod's"
                             f" moved only as far as it must (anchored) (default {config.V10_LINES})")
    tuning.add_argument("--v11-model", metavar="DIR",
                        help="eAMFModel v11-build's output, for --candidate v11 "
                             "(default $EAMF_V11_MODEL, then ./v11_model)")
    tuning.add_argument("--v11-paths", type=int, metavar="N",
                        help=f"games simulated per snapshot for --candidate v11 "
                             f"(default {config.V11_PATHS})")
    tuning.add_argument("--v11-lines", choices=["own", "even", "prod", "anchored", "hyst"],
                        help="--candidate v11: as --v10-lines "
                             f"(default {config.V11_LINES})")
    tuning.add_argument("--v12-model", metavar="DIR",
                        help="eAMFModel v12-build's output, for --candidate v12 "
                             "(default $EAMF_V12_MODEL, then ./v12_model)")
    tuning.add_argument("--v12-paths", type=int, metavar="N",
                        help=f"games simulated per snapshot for --candidate v12 "
                             f"(default {config.V12_PATHS})")
    tuning.add_argument("--v12-lines", choices=["own", "even", "prod", "anchored", "hyst"],
                        help="--candidate v12: as --v10-lines "
                             f"(default {config.V12_LINES})")
    tuning.add_argument("--v13-model", metavar="DIR",
                        help="eAMFModel v13-build's output, for --candidate v13 "
                             "(default $EAMF_V13_MODEL, then ./v13_model)")
    tuning.add_argument("--v13-paths", type=int, metavar="N",
                        help=f"games simulated per snapshot for --candidate v13 "
                             f"(default {config.V13_PATHS})")
    tuning.add_argument("--v13-lines", choices=["own", "even", "prod", "anchored", "hyst"],
                        help="--candidate v13: as --v10-lines "
                             f"(default {config.V13_LINES})")
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
                           ("snapshots", "SNAPSHOTS"),
                           ("chunk", "MATCH_CHUNK_SIZE"),
                           ("candidate_label", "CANDIDATE_LABEL"),
                           ("v8_model", "V8_MODEL_DIR"),
                           ("v8_paths", "V8_PATHS"),
                           ("v8_lines", "V8_LINES"),
                           ("v9_model", "V9_MODEL_DIR"),
                           ("v9_paths", "V9_PATHS"),
                           ("v9_lines", "V9_LINES"),
                           ("v10_model", "V10_MODEL_DIR"),
                           ("v10_paths", "V10_PATHS"),
                           ("v10_lines", "V10_LINES"),
                           ("v11_model", "V11_MODEL_DIR"),
                           ("v11_paths", "V11_PATHS"),
                           ("v11_lines", "V11_LINES"),
                           ("v12_model", "V12_MODEL_DIR"),
                           ("v12_paths", "V12_PATHS"),
                           ("v12_lines", "V12_LINES"),
                           ("v13_model", "V13_MODEL_DIR"),
                           ("v13_paths", "V13_PATHS"),
                           ("v13_lines", "V13_LINES")):
        value = getattr(args, attribute, None)
        if value is not None:
            setattr(config, key, value)

    if getattr(args, "drop_flipped", False):
        config.EXCLUDE_FLIPPED_MATCHES = True

    candidate = getattr(args, "candidate", None)
    if candidate:
        try:
            names = [snowflake_io.stream_name(c.strip()) for c in candidate.split(",") if c.strip()]
        except ValueError as error:
            raise SystemExit(f"--candidate {error}")
        config.CANDIDATES = names
        config.STREAMS["candidate"] = names[0]

    clock = getattr(args, "clock", None)
    if clock is not None:
        config.CLOCK_SOURCE = None if clock == "self" else clock

    if getattr(args, "command", None) in ("compare", "totals-lines", "totals-calibrate", "halves", "kickoffs"):
        return
    end = config.CUTOFF_END or "latest available"
    print(f"\nWindow: {config.CUTOFF_START}  ->  {end}"
          f"   sport {config.SPORT_CODE}"
          f"   spread {config.SPREAD_RESOLUTION}"
          f"   candidate {', '.join(config.CANDIDATES or [config.STREAMS['candidate']])}")


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
    report_parser.add_argument("--bets", action="store_true",
                               help="also run the betting simulation for the same window and "
                                    "candidates, and add it as a section at the bottom of the page")
    report_parser.add_argument("--axes", action="store_true",
                               help="also print the integrity tables and the "
                                    "single-axis breakdowns")

    cross_parser = sub.add_parser(
        "cross", parents=[shared],
        help="cross-sectional calibration by score diff / quarter / possession, "
             "under the line rule")
    cross_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    ep_parser = sub.add_parser(
        "expected-points",
        help="rebuild the expected-points table the in-drive analysis judges plays by")
    ep_parser.add_argument("export", help="scouting_playover.csv (the `scouting` export)")

    da_parser = sub.add_parser(
        "drive-audit", parents=[shared],
        help="check every drive the play feed yields against SCOUTING_FULL")
    da_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    da_parser.add_argument("--limit", type=int, metavar="N", help="the N most recent matches")
    da_parser.add_argument("--no-scouting", action="store_true",
                           help="skip the SCOUTING_FULL cross-check")

    hi_parser = sub.add_parser(
        "history", parents=[shared],
        help="every settled match before --until, shaped like nb2/AMFELO.csv (for v9-build --history)")
    hi_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    bets_parser = sub.add_parser(
        "bets", parents=[shared],
        help="betting simulation: every single bet, pre-match and in play, with prod's and the "
             "candidate's probability at the moment it was priced (a lag per operator off its odds); "
             "--candidate v4,v5,v6 re-prices with each model at its own lines, side by side")
    bets_parser.add_argument("action", nargs="?",
                             choices=["run", "prematch", "probe", "lines", "check", "moments", "lag",
                                      "totals-moves", "totals-signals", "sessions", "favourite"],
                             default="run",
                             help="prematch: the pre-match models' own test -- only the bets "
                                  "placed before kick-off, each model version priced off its kick-off "
                                  "alone (cheap over many weeks), margins week by week and each "
                                  "candidate against the first with a bootstrap interval "
                                  "(out/bets_prematch.txt); "
                                  "favourite: every bet split by the pre-match favourite, the score "
                                  "when it was struck and the side it backs, by temperature: where "
                                  "the stake goes and the book's margin against what prod's price "
                                  "leaves it (out/bets_favourite.txt); "
                                  "sessions: bets and stake by each gamer's match of the session "
                                  "(1st .. 10th+) and matches left in it, against how many matches sit "
                                  "there, by temperature and market (out/bets_sessions.txt); "
                                  "totals-signals: how every match was played in Q1 and the first "
                                  "half (pace, time between plays, clock per play, drives) against prod's "
                                  "total there and where the totals money went next, by temperature "
                                  "(out/totals_signals.csv); "
                                  "totals-moves: prod's totals line before and after every totals bet, "
                                  "restricted accounts against everyone else, and whether each "
                                  "candidate already leaned their way (out/bets_totals_moves.html); "
                                  "lag: the lag match by match off the lines and the odds, every match of "
                                  "the window drawn (out/bets_lag.html); moments: where in the game the book loses, bucketed again off a "
                                  "bets_sim.csv already written (no Snowflake); check: the checks alone (when in the match each bet was placed, "
                                  "cash-outs, SCOUTING_FULL against prod), quick, no candidate "
                                  "priced; probe: print the columns it reads, to check the names; lines: "
                                  "prod's spread and total lines through each match against the "
                                  "lines bet (out/bets_lines.html)")
    bets_parser.add_argument("--sharp", metavar="GROUPS",
                             help="book comparison: the customer temperatures that bet for an edge and "
                                  f"shop on price, comma-separated (default {','.join(config.SIM_SHARP_GROUPS)})")
    bets_parser.add_argument("--elasticity", type=float, metavar="E",
                             help="book comparison: everyone else's stake x (candidate odds / odds) ^ E "
                                  f"(default {config.SIM_ELASTICITY:g}: bets as placed)")
    bets_parser.add_argument("--max-scale", type=float, metavar="X",
                             help=f"book comparison: the most a sharp stake may grow (default {config.SIM_MAX_SCALE:g})")
    bets_parser.add_argument("--boot", type=int, metavar="N",
                             help=f"book comparison and `bets prematch`: bootstrap resamples over matches (default {config.SIM_BOOT})")
    bets_parser.add_argument("--gamer", action="append", metavar="HANDLE",
                             help="run, prematch: only the matches this gamer played (repeatable), "
                                  "with each one's closing pre-match prices -- prod against each "
                                  "candidate -- the result, and what its bets returned the book both "
                                  "ways (out/bets_gamers.txt); one night of one gamer is quick, in play too")
    bets_parser.add_argument("--matches", type=int, default=12, metavar="N",
                             help="lines: draw the N matches with the most spread and total bets")
    bets_parser.add_argument("--match", action="append", metavar="CODE",
                             help="lines: draw this match (repeatable)")
    bets_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    bets_parser.add_argument("--bet-table", metavar="NAME",
                             help=f"the bet-by-bet table or view, DATABASE.SCHEMA.NAME for one "
                                  f"outside {config.DATABASE}.{config.SCHEMA} "
                                  f"(default {config.BET_TABLE})")
    bets_parser.add_argument("--max-lag", type=float, metavar="SECONDS",
                             help=f"the longest lag tried "
                                  f"(default {config.LAG_RANGE[1]})")
    bets_parser.add_argument("--gap", type=int, metavar="N",
                             help="prod messages since a model's PLAY_OVER that SCOUTING_FULL may "
                                  f"lack and the bet still be re-priced (default {config.MAX_SCOUTING_GAP})")
    bets_parser.add_argument("--csv", metavar="PATH",
                             help="moments: the bets CSV to read (default <out>/bets_sim.csv)")
    bets_parser.add_argument("--by", action="append", metavar="COLS",
                             help="moments: cross these bets_sim.csv columns, comma-separated (or "
                                  "day), costliest first; repeatable, e.g. --by clock_band,market,selection")
    bets_parser.add_argument("--players", metavar="CSV",
                             help="moments: a match history CSV (`history`'s, or nb2/AMFELO.csv) to cut by "
                                  "gamer and NFL team: backed_player, opposed_player, backed_team, matchup, "
                                  "team_matchup, home_player, away_team, ..., or gamer / team (each bet "
                                  "once for each side) with gamer_role (backed, opposed, total)")
    bets_parser.add_argument("--scope", choices=["all", "pre-match", "in-play"], default="all",
                             help="moments --by: which bets (default all; `when` splits pre-match from "
                                  "in play by quarter and clock)")
    bets_parser.add_argument("--min-bets", type=int, default=100, metavar="N",
                             help="moments: hide buckets with fewer bets (default 100)")
    bets_parser.add_argument("--react", action="store_true",
                             help="totals-signals: also price each model with its in-game efficiency "
                                  "update on (the streams run it off), labelled <model>-react")
    bets_parser.add_argument("--all-operators", action="store_true",
                             help="totals-moves: keep every operator in the baseline (default: only the "
                                  "operators that send restricted accounts)")
    bets_parser.add_argument("--extra-columns", metavar="COLS",
                             help="bet-table columns to carry into the output, comma-separated "
                                  "(e.g. the customer and VIP columns)")
    bets_parser.set_defaults(func=cmd_bets)

    sc_parser = sub.add_parser(
        "scouting", parents=[shared],
        help="SCOUTING_FULL: probe it against GAMEPLAI_STREAM and export every "
             "PLAY_OVER snapshot GAMEPLAI quoted (AF, last 30 days by default)")
    sc_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    sc_parser.add_argument("--scouting-table", default=scouting.DEFAULT_TABLE, metavar="NAME",
                           help="table name, or DATABASE.SCHEMA.TABLE "
                                f"(default: search {config.DATABASE} for {scouting.DEFAULT_TABLE})")
    sc_parser.add_argument("--sample", type=int, default=2, metavar="N",
                           help="write every column for the N most recent matches (default 2)")
    sc_parser.add_argument("--limit", type=int, metavar="N",
                           help="export only the N most recent matches")
    sc_parser.add_argument("--no-probe", action="store_true")
    sc_parser.add_argument("--no-export", action="store_true")

    ck_parser = sub.add_parser(
        "scouting-check", parents=[shared],
        help="is SCOUTING_FULL complete? every match day from --since (default two weeks before "
             "--break) to --until (default today): matches scheduled against those in scouting, "
             "and each one's messages, sequence, quarters, ENDED, plays, detail, clock and score "
             "against a baseline of the days before --break (-> scouting_check*.txt/csv)")
    ck_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    ck_parser.add_argument("--scouting-table", default=scouting.DEFAULT_TABLE, metavar="NAME",
                           help="table name, or DATABASE.SCHEMA.TABLE")
    ck_parser.add_argument("--break", dest="break_day", default="2026-09-23", metavar="DATE",
                           help="the first day after the baseline (default 2026-09-23)")

    to_parser = sub.add_parser(
        "timeouts", parents=[shared],
        help="every timeout in SCOUTING_FULL: who called it (with or without the ball), when and "
             "at what score, and how many each side had left late in each half "
             "(-> timeouts.csv, timeouts_summary.txt; last 30 days by default)")
    to_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    to_parser.add_argument("--scouting-table", default=scouting.DEFAULT_TABLE, metavar="NAME",
                           help="table name, or DATABASE.SCHEMA.TABLE")

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

    tl_parser = sub.add_parser(
        "totals-lines",
        help="the totals lines value by value, off the report's directional_pairs*.csv (no Snowflake): "
             "how far apart prod's and the candidate's lines sit, the points still to come between "
             "them, and each line's P(over) against how often the game went over")
    tl_parser.add_argument("paths", nargs="*", default=[DEFAULT_OUT],
                           help=f"directional_pairs*.csv files, or folders holding them "
                                f"(default {DEFAULT_OUT})")
    tl_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    tl_parser.add_argument("--all-quotes", action="store_true",
                           help="keep pairs where either quote was not live")

    tc_parser = sub.add_parser(
        "totals-calibrate",
        help="the points still to come, calibrated: fit the rest of the match on a line's points to "
             "come, the board, the plays and the pre-match line at the end of Q1, the half and the end of "
             "Q3, off the `scouting` export (no Snowflake), and test it on the latest matches; "
             "--signals reads totals_signals.csv to fit the same on v8/v9's own points to come")
    tc_parser.add_argument("export", nargs="?", default=os.path.join(DEFAULT_OUT, "scouting_playover.csv"),
                           help="scouting_playover.csv (default <out>/scouting_playover.csv)")
    tc_parser.add_argument("--history", metavar="CSV",
                           help="a match history (nb2/AMFELO.csv, or `history`'s) for the gamers' recent form")
    tc_parser.add_argument("--signals", metavar="CSV",
                           help="a totals_signals.csv (run with --candidate v8,v9) to correct the models too")
    tc_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    hv_parser = sub.add_parser(
        "halves",
        help="off the `scouting` export (no Snowflake): how scoring splits between the halves, the "
             "second half from each half-time margin, touchdown lengths -- for the league and each gamer")
    hv_parser.add_argument("export", nargs="?", default=os.path.join(DEFAULT_OUT, "scouting_playover.csv"),
                           help="scouting_playover.csv (default <out>/scouting_playover.csv)")
    hv_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    hv_parser.add_argument("--min-games", type=int, default=30,
                           help="gamers with at least this many games get a row (default 30)")
    hv_parser.add_argument("--history",
                           help="match_history.csv to take each side's NFL team from (PLAYER_1_TEAM / "
                                "PLAYER_2_TEAM): adds the team and gamer / team splits (default: "
                                "match_history.csv beside the export, when there is one)")
    hv_parser.add_argument("--min-pair-games", type=int, default=20,
                           help="gamer / team pairs with at least this many games get a row (default 20)")

    ko_parser = sub.add_parser(
        "kickoffs",
        help="off the `scouting` export (no Snowflake): how kick-offs land (touchback, no landing zone, "
             "out of bounds, returned) and who kicks which, the half-time double and Q2's pace, and "
             "how prod's total moves on a kick")
    ko_parser.add_argument("export", nargs="?", default=os.path.join(DEFAULT_OUT, "scouting_playover.csv"),
                           help="scouting_playover.csv (default <out>/scouting_playover.csv)")
    ko_parser.add_argument("--history", help="match_history.csv for the gamers (default: match_history.csv "
                                             "beside the export, when there is one)")
    ko_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    ko_parser.add_argument("--min-games", type=int, default=20,
                           help="gamers with at least this many kicks / games get a row (default 20)")

    cmp_parser = sub.add_parser("compare", parents=[shared], help="diff two cell-summary CSVs")
    cmp_parser.add_argument("file_a")
    cmp_parser.add_argument("file_b")

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command in ("scouting", "timeouts") and not (args.days or args.since):
        args.days = 30          # the scouting export defaults to the last month
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
    if args.command == "scouting":
        return cmd_scouting(args)
    if args.command == "timeouts":
        return cmd_timeouts(args)
    if args.command == "scouting-check":
        return cmd_scouting_check(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "indrive":
        return cmd_indrive(args)
    if args.command == "compare":
        return cmd_compare(args)
    if args.command == "history":
        return cmd_history(args)
    if args.command == "drive-audit":
        return cmd_drive_audit(args)
    if args.command == "expected-points":
        return cmd_expected_points(args)
    if args.command == "bets":
        return cmd_bets(args)
    if args.command == "totals-calibrate":
        from . import totals_calibrate
        path = totals_calibrate.run(args.export, args.out or DEFAULT_OUT, args.history, args.signals)
        print(f"\n  -> {path} and totals_calibration.txt")
        return 0
    if args.command == "halves":
        from . import halves
        history = args.history
        if history is None:
            beside = os.path.join(os.path.dirname(args.export) or ".", "match_history.csv")
            history = beside if os.path.exists(beside) else None
        halves.run(args.export, args.out or DEFAULT_OUT, min_games=args.min_games, history_path=history,
                   min_pair_games=args.min_pair_games)
        return 0
    if args.command == "kickoffs":
        from . import kickoffs
        history = args.history
        if history is None:
            beside = os.path.join(os.path.dirname(args.export) or ".", "match_history.csv")
            history = beside if os.path.exists(beside) else None
        kickoffs.run(args.export, args.out or DEFAULT_OUT, history_path=history, min_games=args.min_games)
        return 0
    if args.command == "totals-lines":
        from . import totals_lines
        totals_lines.run(args.paths, args.out or DEFAULT_OUT, live_only=not args.all_quotes)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
