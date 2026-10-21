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


def _arms(args):
    """[(name, source)] off the --arm NAME=SOURCE options."""
    arms = []
    for spec in args.arm or []:
        name, sep, source = spec.partition("=")
        if not sep or not name or not source:
            raise SystemExit(f"--arm {spec!r}: give it as NAME=SOURCE")
        arms.append((name.strip(), source.strip()))
    return arms


def _features(args):
    """The gamers' per-match stats for the feature layer: --features, else player-stats' cache in the
    output directory; "none" turns the layer off."""
    from . import player_stats
    if getattr(args, "features", None) == "none":
        return None
    return args.features or os.path.join(args.out or DEFAULT_OUT, player_stats.CACHE)


def cmd_prior_backtest(args):
    """Pre-match priors against prod, each with and without the rematch layer."""
    from . import prior_backtest
    arms = _arms(args)
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            path = prior_backtest.run(cur, args.out or DEFAULT_OUT, arms, config.CUTOFF_START, config.CUTOFF_END,
                                      draws=args.draws, features=_features(args))
    finally:
        conn.close()
    print(f"\n  -> {path} and prior_backtest.csv")
    return 0


def cmd_replay_ui(args):
    """The v21 replay page: a historic match a PLAY_OVER at a time, v21 against prod, the nudge by hand."""
    from . import replay_ui
    cur = None
    if not args.export:
        config.FETCH_DISK_CACHE = os.path.join(args.out or DEFAULT_OUT, "replay_cache")
        cur = _LazyCursor()
    replay = replay_ui.build(args.v21_model, args.prior_model, args.export, args.history, cur, args.paths, args.share)
    try:
        replay_ui.serve(replay, args.port, browser=not args.no_browser)
    finally:
        if cur is not None:
            cur.close()
    return 0


def cmd_walkthrough(args):
    """The best pre-match variants priced for every match, walk-forward, one table each for the sims."""
    from . import walkthrough
    out = args.out or os.path.join(DEFAULT_OUT, "walkthrough")
    path = walkthrough.run(out, args.history, args.since or "2026-01-01 00:00:00", args.until, workers=args.workers,
                           cores=args.cores, draws=args.draws, half_life=args.half_life)
    print(f"\n  -> {path}; the tables in {os.path.join(out, 'priors')}")
    return 0


def cmd_prior_offline(args):
    """The priors and their layers graded against each other on the results: no prod, no Snowflake."""
    from . import prior_offline
    path = prior_offline.run(args.out or DEFAULT_OUT, _arms(args), args.history, args.since, args.until,
                             _features(args), args.blend or (), args.draws)
    print(f"\n  -> {path} and prior_offline.csv")
    return 0


def cmd_nb2_history(args):
    """NB2 refitted daily over a span, every match priced by its own day's fit, into one CSV."""
    from . import nb2_history
    out = args.out or DEFAULT_OUT
    cur = None if args.history else _LazyCursor()
    try:
        path = nb2_history.run(out, args.since or "2026-01-01 00:00:00", config.CUTOFF_END, args.history, cur,
                               args.half_life, args.workers)
    finally:
        if cur is not None:
            cur.close()
    print(f"\n  -> {path} and {nb2_history.TXT_NAME}")
    return 0


def cmd_backtest_suite(args):
    """Every pre-match prior variant over one window: fitted, audited, graded, replayed, ranked."""
    from . import backtest_suite, player_stats
    cfg = backtest_suite.load_config(args.config)
    if cfg.get("features") is None:
        cfg["features"] = os.path.join(DEFAULT_OUT, player_stats.CACHE)
    out = args.out or os.path.join(DEFAULT_OUT, "suite")
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    cur = _LazyCursor()                     # connects on the first query the suite's cache can't answer
    try:
        path = backtest_suite.run(cur, out, cfg, config.CUTOFF_START, config.CUTOFF_END, steps, args.keep_going,
                                  history=args.history, fresh=args.fresh)
    finally:
        cur.close()
    print(f"\n  -> {path} (the suite: {out})" + ("" if cur.opened else "; no Snowflake connection needed"))
    return 0


class _LazyCursor:
    """A Snowflake cursor opened on the first query that runs (none at all when every answer is
    already in a query cache, so no warehouse is woken)."""

    def __init__(self):
        self.conn = self.cur = None
        self.opened = False

    def execute(self, *args, **kwargs):
        if self.cur is None:
            self.conn = snowflake_io.get_connection()
            self.cur = self.conn.cursor()
            self.opened = True
        return self.cur.execute(*args, **kwargs)

    @property
    def description(self):
        return self.cur.description

    def fetchall(self):
        return self.cur.fetchall()

    def close(self):
        if self.conn is not None:
            self.cur.close()
            self.conn.close()


def cmd_prior_replay(args):
    """The pre-match bets replayed at each prior's prices, with and without the rematch layer."""
    from . import prior_replay
    arms = _arms(args)
    if not arms:
        raise SystemExit("prior-replay: give at least one --arm NAME=SOURCE")
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            path = prior_replay.run(cur, args.out or DEFAULT_OUT, arms, config.CUTOFF_START, config.CUTOFF_END,
                                    draws=args.draws, features=_features(args))
    finally:
        conn.close()
    print(f"\n  -> {path} and prior_replay.csv")
    return 0


def cmd_player_stats(args):
    """Every gamer's h2hggl stats off our history and SCOUTING_FULL, and the check against the site."""
    from . import player_stats
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            path = player_stats.run(cur, args.out or DEFAULT_OUT, since=args.since, check=args.check,
                                    refresh=args.refresh)
    finally:
        conn.close()
    print(f"\n  -> {path}, player_stats.csv, {player_stats.CACHE}"
          + (" and player_stats_check.csv" if args.check else ""))
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


def prior_spec(value, candidates):
    """'v21-avgf2=<prior>' -> ('v21-avgf2', '<prior>'): the name must be a v21 candidate of this run
    and the prior a table or build directory that exists."""
    name, eq, prior = value.partition("=")
    name, prior = name.strip().lower(), prior.strip()
    if not eq or not name or not prior:
        raise ValueError(f"{value!r}: give NAME=PRIOR, e.g. v21-avgf2=priors/avg_form2_w05.csv")
    names = {snowflake_io.model_version(c)[0] for c in candidates if snowflake_io.is_model(c)}
    if name not in names:
        raise ValueError(f"{value!r}: {name} is not a --candidate of this run"
                         + (f" ({', '.join(sorted(names))})" if names else ""))
    if snowflake_io.model_base(name) != "v21":
        raise ValueError(f"{value!r}: only v21 takes a prior")
    if not os.path.exists(prior):
        raise ValueError(f"{value!r}: no such table or build directory, {prior}")
    return name, prior


def read_match_list(value):
    """Match codes from a file (one a line, the first word; # starts a comment) or a comma-separated
    list."""
    if os.path.exists(value):
        with open(value, encoding="utf-8") as fh:
            codes = [line.split("#", 1)[0].split() for line in fh]
        return [c[0] for c in codes if c]
    return [c.strip() for c in value.split(",") if c.strip()]


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
    if getattr(args, "only_matches", None):
        config.ONLY_MATCHES = read_match_list(args.only_matches)
    if args.gap is not None:
        config.MAX_SCOUTING_GAP = args.gap
    if args.max_lag is not None:
        config.LAG_RANGE = (config.LAG_RANGE[0], args.max_lag)
    apply_uplift(args)
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
            elif args.action == "rematches":
                from . import bet_rematches
                path = bet_rematches.run(cur, out_dir)
                print(f"\n  -> {path} and bets_rematches.csv")
            elif args.action == "inference":
                from . import bet_inference
                path = bet_inference.run(cur, out_dir)
                print(f"\n  -> {path} and bets_inference.csv")
            elif args.action == "ml-losers":
                from . import bet_losers
                path = bet_losers.run(cur, out_dir)
                print(f"\n  -> {path}, bets_ml_losers.csv, bets_ml_matches.csv and bets_ml_pileons.csv")
            elif args.action == "favourite":
                from . import bet_favourite
                path = bet_favourite.run(cur, out_dir, min_bets=args.min_bets)
                if path:
                    print(f"\n  -> {path} and bets_favourite.txt")
            elif args.action == "totals-signals":
                from . import totals_signals
                path = totals_signals.run(cur, out_dir, react=args.react)
                print(f"\n  -> {path} and totals_signals.txt")
            elif args.action == "money":
                from . import bet_money
                paths = bet_money.run(cur, out_dir, sim_file=args.sim_file, sim_candidate=args.sim_candidate,
                                      window=args.money_window,
                                      split=dt.date.fromisoformat(args.money_split) if args.money_split else None,
                                      customer_column=args.customer_column, run_after=args.run_window)
                if paths:
                    print(f"\n  -> {paths[0]} and {paths[1]}")
            elif args.action == "ml-moves":
                from . import bet_ml_moves
                paths = bet_ml_moves.run(cur, out_dir, min_bets=args.min_bets)
                if paths:
                    print(f"\n  -> {paths[0]} and {paths[1]}")
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
                print(f"\n  -> {os.path.join(out_dir, 'bets_sim.csv')}, bets_latency.csv and margin_uplift.txt")
    finally:
        conn.close()
    return 0


def cmd_history(args):
    """Every settled match before --until (default: now), shaped like
    nb2/AMFELO.csv: what v4's own pre-match model (NB2) is fitted on."""
    out_dir = args.out or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            _write_history(cur, out_dir)
    finally:
        conn.close()
    return 0


def cmd_refresh(args):
    """The data every build and report reads, in one go: SCOUTING_FULL's PLAY_OVER snapshots and
    every timeout from --since (default config.REFRESH_SINCE) to --until, and every settled match
    before --until -- scouting_playover.csv, timeouts.csv and match_history.csv in --out."""
    from . import timeouts
    out_dir = args.out or DEFAULT_OUT
    os.makedirs(out_dir, exist_ok=True)
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            table = scouting.locate(cur, args.scouting_table)
            scouting.export(cur, table, os.path.join(out_dir, "scouting_playover.csv"))
            timeouts.run(cur, table, out_dir)
            _write_history(cur, out_dir)
    finally:
        conn.close()
    print(f"\n  refreshed {out_dir}: scouting_playover.csv, timeouts.csv, match_history.csv")
    return 0


def _write_history(cur, out_dir):
    """match_history.csv: every settled match before config.CUTOFF_END."""
    import csv
    rows = snowflake_io.fetch_history(cur, until=config.CUTOFF_END)
    path = os.path.join(out_dir, "match_history.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, snowflake_io.HISTORY_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    first = rows[0]["SCHEDULED_START_TIME_UTC"] if rows else "-"
    last = rows[-1]["SCHEDULED_START_TIME_UTC"] if rows else "-"
    print(f"\n  {len(rows):,} settled {config.SPORT_CODE} matches, {first} -> {last}: {path}")
    return path


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


def cmd_pair_check(args):
    """Prod's first live row on each PLAY_OVER's message against its last (pair_check.py)."""
    from . import pair_check
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            pair_check.run(cur, args.out or DEFAULT_OUT)
    finally:
        conn.close()
    return 0


def cmd_money_layer(args):
    """What the money already bet says about who wins (money_layer.py): the days before, as a prior table,
    and the match so far."""
    from . import money_layer

    def go(cur):
        return money_layer.run(args.out or os.path.join(DEFAULT_OUT, "money"), args.base, history=args.history,
                               bets=args.bets, cur=cur, shrink=args.shrink, half_life=args.half_life, cut=args.cut,
                               slope=args.slope, share=args.share, test_since=args.test_since, grid=not args.no_grid)
    if args.bets:
        go(None)
        return 0
    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            go(cur)
    finally:
        conn.close()
    return 0


def table_specs(specs):
    """[(name, path)] off --table NAME=CSV options."""
    tables = []
    for spec in specs or []:
        name, eq, path = spec.partition("=")
        if not eq or not name.strip() or not path.strip():
            raise SystemExit(f"--table {spec!r}: give it as NAME=CSV, e.g. nb2=eAMFCalibrator/out/walkthrough/priors/nb2.csv")
        if not os.path.isfile(path.strip()):
            raise SystemExit(f"--table {spec!r}: no such file")
        tables.append((name.strip(), path.strip()))
    return tables


def cmd_prior_bets(args):
    """The pre-match report: prod against the priors' own prices and a sim's (prior_bets.py)."""
    from . import prior_bets
    prior_bets.run(args.out or os.path.join(DEFAULT_OUT, "prior_bets"), args.bets, table_specs(args.table),
                   sim_name=args.sim_name, draws=args.draws)
    return 0


def cmd_form_prior(args):
    """Walkthrough tables moved toward a fast walk-forward form rating (form_prior.py)."""
    from . import form_prior
    tables = table_specs(args.table or [f"avg={os.path.join(DEFAULT_OUT, 'walkthrough', 'priors', 'avg.csv')}"])
    history = args.history or os.path.join(DEFAULT_OUT, "match_history.csv")
    if not os.path.isfile(history):
        raise SystemExit(f"form-prior: no history at {history} (python -m eAMFCalibrator history, or --history)")
    numbers = lambda text: tuple(float(x) for x in text.split(",") if x.strip())
    form_prior.run(args.out or os.path.join(DEFAULT_OUT, "form_prior"), history, tables,
                   drifts=numbers(args.drift), weights=numbers(args.weight), draws=args.draws, bets=args.bets)
    return 0


def cmd_ml_states(args):
    """The moneyline by game situation, prod against a candidate (ml_states.py)."""
    from . import ml_states
    ml_states.run(args.out or os.path.join(DEFAULT_OUT, "ml_states"), args.pairs, args.bets, label=args.label,
                  history=args.history)
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
        apply_uplift(args)
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
                             "or a model version (v14, v21 -- see eAMFModel) priced "
                             "live off SCOUTING_FULL's PLAY_OVER snapshots and its build; "
                             "several, comma-separated (v14,v21), set each against prod "
                             "side by side in the report; a second build of a version is "
                             "NAME=DIR, e.g. v21,v21-t919=v21_t919; v21 with an in-play "
                             "strength update is v21-snapT, v21-ptsT or v21-wpaC, e.g. "
                             "v21,v21-snap4,v21-wpa10 (eAMFModel's v21_react); with the pace nudge "
                             "v21-drives (--v21-drives) "
                             f"(default {config.STREAMS['candidate']})")
    tuning.add_argument("--candidate-label", metavar="NAME",
                        help="what the HTML reports call the candidate (default: the "
                             "model version when one stands in, e.g. v21; else 'candidate')")
    tuning.add_argument("--v14-model", metavar="DIR",
                        help="eAMFModel v14-build's output, for --candidate v14 "
                             "(default $EAMF_V14_MODEL, then ./v14_model)")
    tuning.add_argument("--v14-paths", type=int, metavar="N",
                        help=f"games simulated per snapshot for --candidate v14 "
                             f"(default {config.V14_PATHS})")
    tuning.add_argument("--v14-lines", choices=["own", "even", "prod", "anchored", "hyst"],
                        help="--candidate v14: its own lines (in the gap between the key numbers), the"
                             " even line, prod's, prod's line moved only as far as it must (anchored) or the"
                             f" even line held until far off (hyst) (default {config.V14_LINES})")
    tuning.add_argument("--v21-model", metavar="DIR",
                        help="eAMFModel v21-build's output, for --candidate v21 "
                             "(default $EAMF_V21_MODEL, then ./v21_model)")
    tuning.add_argument("--v21-paths", type=int, metavar="N",
                        help=f"games simulated per snapshot for --candidate v21 "
                             f"(default {config.V21_PATHS}; the kick-off twice that, each side receiving)")
    tuning.add_argument("--v21-lines", choices=["own", "even", "prod", "anchored", "hyst"],
                        help="--candidate v21: as --v14-lines "
                             f"(default {config.V21_LINES})")
    tuning.add_argument("--v21-prior-model", metavar="DIR|CSV",
                        help="--candidate v21: a walkthrough variant's table (walkthrough's priors/VARIANT.csv), "
                             "or the v14 build (made with --history) whose pre-match "
                             "expected points v21 nudges toward (default $EAMF_V21_PRIOR, then the form table "
                             "form-prior writes, out/form_prior/priors/avg_form4_w05.csv, then $EAMF_V14_MODEL, "
                             "then ./v14_model)")
    tuning.add_argument("--v21-nudge-share", type=float, metavar="S",
                        help="--candidate v21: how far toward the prior's expected points each side's "
                             f"offense is nudged from the sim's own (default {config.V21_NUDGE_SHARE}; "
                             "0: the sim's own means, no prior)")
    tuning.add_argument("--v21-drives", metavar="CSV",
                        help="--candidate v21-drives: each match's expected drives the pace nudge meets "
                             "(xgb/pace_eff.py's pace_eff_prior.csv; default $EAMF_V21_DRIVES, then "
                             "xgb/out/pace_eff/pace_eff_prior.csv)")
    tuning.add_argument("--prior-for", action="append", metavar="NAME=PRIOR",
                        help="a v21 candidate's own prior, as --v21-prior-model takes (a walkthrough table or "
                             "a build), so v21 candidates nudged toward different priors are priced in one run "
                             "on the same bets, e.g. --candidate v21,v21-avgf2 --prior-for "
                             "v21-avgf2=eAMFCalibrator/out/form_prior/priors/avg_form2_w05.csv (repeat it)")
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
                           ("v14_model", "V14_MODEL_DIR"),
                           ("v14_paths", "V14_PATHS"),
                           ("v14_lines", "V14_LINES"),
                           ("v21_model", "V21_MODEL_DIR"),
                           ("v21_paths", "V21_PATHS"),
                           ("v21_lines", "V21_LINES"),
                           ("v21_prior_model", "V21_PRIOR_MODEL_DIR"),
                           ("v21_nudge_share", "V21_NUDGE_SHARE"),
                           ("v21_drives", "V21_DRIVES_TABLE")):
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
    for spec in getattr(args, "prior_for", None) or ():
        try:
            config.MODEL_PRIORS.update([prior_spec(spec, config.CANDIDATES)])
        except ValueError as error:
            raise SystemExit(f"--prior-for {error}")

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


def uplift_options(parser):
    """The margin uplift's options (margin_uplift.py: prod against each candidate, every in-play
    tick a dummy bet between the two books), for report --bets and bets."""
    group = parser.add_argument_group("margin uplift")
    group.add_argument("--margin-chart", metavar="CSV",
                       help="the margin chart the dummy bets are priced at (probability, numerator, "
                            "denominator, e.g. FD_109_BALANCED.csv); default a proportional 109%% book")
    group.add_argument("--uplift-factor", type=float, metavar="X",
                       help="minutes from the scheduled start x X = game minutes (default: fitted so "
                            "prod's matches end at 60)")
    group.add_argument("--all-moments", action="store_true",
                       help="price v21 at every moment prod's quotes read, not only the ones the bets read "
                            "(the default, several times quicker): slower, and the margin uplift then bets "
                            "every in-play tick rather than the bets' moments alone")


def apply_uplift(args):
    config.UPLIFT_CHART = getattr(args, "margin_chart", None)
    config.UPLIFT_FACTOR = getattr(args, "uplift_factor", None)
    config.ALL_MOMENTS = bool(getattr(args, "all_moments", False))
    if config.UPLIFT_CHART and not os.path.exists(config.UPLIFT_CHART):
        raise SystemExit(f"--margin-chart {config.UPLIFT_CHART}: no such file")
    if config.UPLIFT_CHART:                     # read it now: a bad chart stops the run before the long part
        from . import margin_uplift
        chart = margin_uplift.load_chart(config.UPLIFT_CHART)
        print(f"  margin chart {config.UPLIFT_CHART}: {len(chart):,} prices, probability"
              f" {min(chart) / 1000:.3f} to {max(chart) / 1000:.3f}")


def build_parser():
    parser = argparse.ArgumentParser(prog="eAMFCalibrator", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    shared = common_options()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", parents=[shared], help="check tables, clock column and window volume")

    pc_parser = sub.add_parser(
        "pair-check", parents=[shared],
        help="prod's moneyline at each PLAY_OVER: its first live row on the message the report pairs "
             "on against its last, graded on the result -- whether the report's pairing flatters a "
             "candidate (-> out/pair_check.csv)")
    pc_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    mo_parser = sub.add_parser(
        "money-layer", parents=[shared],
        help="what the money already bet says about who wins: each gamer's pre-match price moved by their backers' "
             "edge on the matches before (walk-forward, a prior table for --v21-prior-model, tested on the results), "
             "and with --bets each moneyline price moved toward the stake already on its side of the match "
             "(re-priced bets and two books) (-> out/money/)")
    mo_parser.add_argument("--base", required=True, metavar="CSV|prod",
                           help="the prior table it moves (a walkthrough's priors/VARIANT.csv), or prod: prod's "
                                "pre-match price off --bets, tested only")
    mo_parser.add_argument("--history", metavar="CSV",
                           help="a match_history.csv: the kick-offs, gamers and results a table lacks (--base prod needs it)")
    mo_parser.add_argument("--bets", nargs="+", metavar="CSV",
                           help="betting-sim bets_sim.csv files: each bet at prod's price, and the match-so-far test "
                                "(default: config.BET_TABLE over the window, at the operator's odds less its margin)")
    mo_parser.add_argument("--shrink", type=float, metavar="STAKE",
                           help="the stake a gamer's backers' edge is shrunk by (default 50,000)")
    mo_parser.add_argument("--half-life", type=float, metavar="DAYS", help="older bets count for less (default: none)")
    mo_parser.add_argument("--cut", choices=["kickoff", "daily"], default="kickoff",
                           help="the bets a match is priced off: settled by its kick-off, or by the start of its day "
                                "(default kickoff)")
    mo_parser.add_argument("--slope", type=float, help="a fixed slope (default: fitted day by day on the days before)")
    mo_parser.add_argument("--share", type=float,
                           help="the share of the way a sim goes to the table's points (default v21's nudge share, 0.75)")
    mo_parser.add_argument("--test-since", metavar="DATE", help="the test's first day (default: a week into the window)")
    mo_parser.add_argument("--no-grid", action="store_true", help="skip the settings side by side")
    mo_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT}/money)")

    pb_parser = sub.add_parser(
        "prior-bets",
        help="the pre-match report, no simulation: prod against each walkthrough table's own price (its score "
             "distribution) and a sim's pre-match price from a betting-sim run -- the moneyline graded on the "
             "results by how heavily each match was bet, and every pre-match bet (and the kick-off moneyline money) "
             "re-priced, same bets and two books (-> out/prior_bets/)")
    pb_parser.add_argument("--bets", nargs="+", required=True, metavar="CSV",
                           help="betting-sim bets_sim.csv files: prod's price at each bet, and the sim's (its candidate)")
    pb_parser.add_argument("--table", action="append", metavar="NAME=CSV",
                           help="a walkthrough table (priors/VARIANT.csv) to price with, as NAME=CSV; repeat it")
    pb_parser.add_argument("--sim-name", default="sim", help="what to call the bets' candidate (default sim)")
    pb_parser.add_argument("--draws", type=int, default=40000, metavar="N",
                           help="simulated scorelines per match for a table's score model (default 40000)")
    pb_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT}/prior_bets)")

    fp_parser = sub.add_parser(
        "form-prior",
        help="walkthrough tables moved toward a fast walk-forward rating (each gamer's strength as his latest "
             "results have it, drifting DRIFT points^2 a day), WEIGHT of the way; with no options, v2.1's default "
             "prior: the walkthrough's avg table, drift 4, half way (-> out/form_prior/priors/avg_form4_w05.csv)")
    fp_parser.add_argument("--history", metavar="CSV",
                           help=f"a match_history.csv (`history`): every result, kick-off, gamer and team "
                                f"(default {os.path.join(DEFAULT_OUT, 'match_history.csv')})")
    fp_parser.add_argument("--table", action="append", metavar="NAME=CSV",
                           help="a walkthrough table (priors/VARIANT.csv) to move, as NAME=CSV; repeat it (default "
                                f"avg={os.path.join(DEFAULT_OUT, 'walkthrough', 'priors', 'avg.csv')})")
    fp_parser.add_argument("--drift", default="4", metavar="LIST",
                           help="the rating's drift, points^2 a day, comma-separated (default 4)")
    fp_parser.add_argument("--weight", default="0.5", metavar="LIST",
                           help="how far toward the rating each table's margin moves, comma-separated (default 0.5)")
    fp_parser.add_argument("--draws", type=int, default=20000, metavar="N",
                           help="simulated scorelines per match for a table's score model, for p_home (default 20000)")
    fp_parser.add_argument("--bets", nargs="+", metavar="CSV",
                           help="betting-sim bets_sim.csv files: how many of their matches (and stake) the history "
                                "and each table have")
    fp_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT}/form_prior)")

    ms_parser = sub.add_parser(
        "ml-states",
        help="the moneyline by game situation off the report's pairs, no Snowflake: what prod and the candidate price "
             "the pre-match favourite at after the first score (who, TD or FG, the opening drive or later) and at each "
             "quarter's start by its lead, against how often it won; with --bets, what the bets struck there did "
             "(-> out/ml_states/)")
    ms_parser.add_argument("--pairs", nargs="+", required=True, metavar="CSV",
                           help="the report's directional_pairs.csv (prod's and the candidate's prices at every snapshot)")
    ms_parser.add_argument("--bets", nargs="+", metavar="CSV", help="betting-sim bets_sim.csv files on the same matches")
    ms_parser.add_argument("--label", default="candidate", help="what to call the candidate (default candidate)")
    ms_parser.add_argument("--history", metavar="CSV",
                           help="a match_history.csv: the gamers' handles, for every gamer's numbers")
    ms_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT}/ml_states)")

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
    uplift_options(report_parser)
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
                                      "totals-moves", "ml-moves", "money", "totals-signals", "sessions", "favourite", "ml-losers",
                                      "rematches", "inference"],
                             default="run",
                             help="prematch: the pre-match models' own test -- only the bets "
                                  "placed before kick-off, each model version priced off its kick-off "
                                  "alone (cheap over many weeks), margins week by week and each "
                                  "candidate against the first with a bootstrap interval "
                                  "(out/bets_prematch.txt); "
                                  "rematches: two gamers meeting again the same day (their session): the "
                                  "money by meeting (1st, 2nd, 3rd, 4th+) and teams (the same as the first "
                                  "meeting, swapped, changed), and where the moneyline, handicap and total "
                                  "money goes against the first meeting's winner and margin, with prod's "
                                  "price of him then and now (out/bets_rematches.txt); "
                                  "inference: two gamers meeting after either played someone else that "
                                  "session (A beat B, C beat D, then A v D): the money and results by each "
                                  "one's last other game (won v lost, both won, ...), the gap between them "
                                  "and a common opponent, against prod's price of the form side, the "
                                  "totals carry-over, fits, and prod with the rematch and inference layers "
                                  "(out/bets_inference.txt); "
                                  "ml-losers: the matches the book loses on the moneyline against "
                                  "the rest (what is targeted, how they played out), every moneyline "
                                  "bet's residual (won less prod's p) by the game at the bet (lead, ball, "
                                  "down and distance, pace, session, form), the features ranked, the "
                                  "pile-ons, gamers, teams and the customers who win; --candidate adds "
                                  "each model's residual (out/bets_ml_losers.txt); "
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
                                  "money: where the money is, as prices -- every market's price moved by the money "
                                  "toward a side (fitted on the window's earlier matches, scored on the later), and "
                                  "what the money knows beside prod's move, the game script, a recent score and "
                                  "v2.1's price (--sim-file) (out/bets_money.txt and bets_money_points.csv); "
                                  "ml-moves: prod's moneyline price either side of every moneyline bet -- where "
                                  "the money came, at what prices, and whether prod moved the bettors' way after, "
                                  "a score in between or not, and pre-match the closing-line value "
                                  "(out/bets_ml_moves.txt and .csv; no candidate priced); "
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
    bets_parser.add_argument("--sim-file", metavar="CSV",
                             help="money: a bets run's bets_sim.csv (or .zip) for v2.1's price at each bet")
    bets_parser.add_argument("--sim-candidate", metavar="NAME",
                             help="money: the candidate in --sim-file (default: its first v21 one)")
    bets_parser.add_argument("--money-window", type=float, default=120.0, metavar="SECONDS",
                             help="money: in play, the money over the last this many seconds (default 120)")
    bets_parser.add_argument("--customer-column", metavar="COLUMN",
                             help=f"money: the bet table's customer id column (default {config.BET_CUSTOMER_COLUMN})")
    bets_parser.add_argument("--run-window", type=float, default=120.0, metavar="SECONDS",
                             help="money: in play, the seconds after a customer's bet the other customers' money is read"
                                  " over (default 120; pre-match 1800)")
    bets_parser.add_argument("--money-split", metavar="YYYY-MM-DD",
                             help="money: fit on the matches before this day, score on the rest (default: the "
                                  "window's middle day)")
    bets_parser.add_argument("--boot", type=int, metavar="N",
                             help=f"book comparison and `bets prematch`: bootstrap resamples over matches (default {config.SIM_BOOT})")
    bets_parser.add_argument("--gamer", action="append", metavar="HANDLE",
                             help="run, prematch: only the matches this gamer played (repeatable), "
                                  "with each one's closing pre-match prices -- prod against each "
                                  "candidate -- the result, and what its bets returned the book both "
                                  "ways (out/bets_gamers.txt); one night of one gamer is quick, in play too")
    bets_parser.add_argument("--only-matches", metavar="FILE|CODES",
                             help="run, prematch: only these matches' bets -- a file of match codes (one a "
                                  "line, # comments), or codes comma-separated; a quick test on a chosen few")
    bets_parser.add_argument("--matches", type=int, default=12, metavar="N",
                             help="lines: draw the N matches with the most spread and total bets")
    bets_parser.add_argument("--match", action="append", metavar="CODE",
                             help="lines: draw this match (repeatable)")
    uplift_options(bets_parser)
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

    rf_parser = sub.add_parser(
        "refresh", parents=[shared],
        help="the data every build and report reads, in one go: the PLAY_OVER snapshots and timeouts "
             f"from --since (default {config.REFRESH_SINCE}) and every settled match to --until "
             "(-> scouting_playover.csv, timeouts.csv, match_history.csv)")
    rf_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")
    rf_parser.add_argument("--scouting-table", default=scouting.DEFAULT_TABLE, metavar="NAME",
                           help="table name, or DATABASE.SCHEMA.TABLE")

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

    pb_parser = sub.add_parser(
        "prior-backtest", parents=[shared],
        help="pre-match priors against prod, each as it is, with the rematch layer (+layer, rematch_layer.py) "
             "and with the inference layer (+infer, inference_layer.py: each gamer's form that session from "
             "every match he played), both fitted walk-forward on its own errors: the moneyline's log loss "
             "and Brier, the total's RMSE and the over at prod's line, match by match against prod, for "
             "every match, the rematches and the first meetings (-> prior_backtest.txt/csv)")
    pb_parser.add_argument("--arm", action="append", metavar="NAME=SOURCE[#MODE]",
                           help="a prior to test (repeatable): NAME=PATH@PREFIX (glmer/cadence.R's "
                                "cadence_matches.csv, columns PREFIX|1 and |2), NAME=PATH (a CSV of "
                                "MATCH_CODE and Pred_P1_Points / Pred_P2_Points, raw_home / raw_away or "
                                "exp_p1 / exp_p2), or NAME=rolling:DIR (eAMFModel prior-daily's fits). "
                                "#MODE reads its moneyline and over: own (NB2's or glmer's own joint "
                                "score model; the default where there is one), fitted (NB2's construction "
                                "fitted to its errors; the default otherwise) or normal")
    pb_parser.add_argument("--draws", type=int, default=40000, metavar="N",
                           help="simulated scorelines per match (default 40000)")
    pb_parser.add_argument("--features", metavar="CSV",
                           help="the gamers' per-match stats for the feature layer (+feat, +both+feat): "
                                "player-stats' player_match_stats.csv (default: the one in --out; "
                                "'none' for no feature layer)")
    pb_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    pr_parser = sub.add_parser(
        "prior-replay", parents=[shared],
        help="the window's pre-match bets replayed at each prior's prices (prior-backtest's arms, each as it "
             "is and with each layer, and prod with each layer): the same bets and stakes at the arm's "
             "chance at each bet's line, the operator's margin kept, the book's margin against prod's "
             "(-> prior_replay.txt/csv)")
    pr_parser.add_argument("--arm", action="append", metavar="NAME=SOURCE[#MODE]",
                           help="a prior to replay (repeatable), as prior-backtest's --arm")
    pr_parser.add_argument("--draws", type=int, default=40000, metavar="N",
                           help="simulated scorelines per match (default 40000)")
    pr_parser.add_argument("--features", metavar="CSV",
                           help="as prior-backtest's --features")
    pr_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    ru_parser = sub.add_parser(
        "replay-ui",
        help="a local page to step through a historic match a PLAY_OVER at a time: v21 prices every state "
             "beside prod's quote at that message, with the nudge (the expected points v21 is pulled toward) "
             "set by hand")
    ru_parser.add_argument("--v21-model", metavar="DIR", help="eAMFModel v21-build's output (default $EAMF_V21_MODEL, "
                                                             "then ./v21_model)")
    ru_parser.add_argument("--prior-model", "--v21-prior-model", dest="prior_model", metavar="DIR|CSV",
                           help="the expected points' source: a v14 build or a walkthrough table "
                                "(priors/avg.csv); default $EAMF_V21_PRIOR, then the form table form-prior writes "
                                "(out/form_prior/priors/avg_form4_w05.csv), then $EAMF_V14_MODEL, then ./v14_model")
    ru_parser.add_argument("--export", metavar="CSV",
                           help="read the matches off a `scouting` export (out/scouting_playover.csv) instead of "
                                "Snowflake")
    ru_parser.add_argument("--history", metavar="CSV",
                           help="a match_history.csv (or AMFELO file) for the day's matches, gamers and teams "
                                "(default: Snowflake's history)")
    ru_parser.add_argument("--paths", type=int, default=1000, help="games simulated a price (default 1000)")
    ru_parser.add_argument("--share", type=float, default=0.75,
                           help="how far the kick-off means move toward the expected points (default 0.75)")
    ru_parser.add_argument("--port", type=int, default=8766, help="the local port (default 8766)")
    ru_parser.add_argument("--no-browser", action="store_true", help="don't open the page")
    ru_parser.add_argument("--out", help=f"where the query cache goes (default: {DEFAULT_OUT}/replay_cache)")

    wt_parser = sub.add_parser(
        "walkthrough",
        help="the best pre-match variants priced for every match of a span, walk-forward (NB2 and glmer refitted "
             "daily, the rematch and inference layers, their averages), one table of expected points each for a "
             "sim to read (--v21-prior-model TABLE), with progress and an estimated finish (-> out/walkthrough/)")
    wt_parser.add_argument("--history", required=True, metavar="CSV",
                           help="a match_history.csv (eAMFCalibrator history's, or nb2/AMFELO.csv): no Snowflake")
    wt_parser.add_argument("--since", metavar="DATE", help="the first day priced (default 2026-01-01)")
    wt_parser.add_argument("--until", metavar="DATE", help="the day after the last priced (default: the history's end)")
    wt_parser.add_argument("--workers", type=int, default=4, help="NB2 fits side by side (default 4)")
    wt_parser.add_argument("--cores", type=int, help="glmer's cores (default: all but one)")
    wt_parser.add_argument("--draws", type=int, default=20000, metavar="N",
                           help="simulated scorelines per match for the moneylines (default 20000)")
    wt_parser.add_argument("--half-life", type=float, metavar="DAYS",
                           help="NB2's half-life (default its own, 60); a new value needs a new --out")
    wt_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT}/walkthrough)")

    po_parser = sub.add_parser(
        "prior-offline",
        help="the pre-match priors graded against each other on the results alone, no prod and no Snowflake: "
             "prior-backtest's arms off a match history CSV, each as it is and with each layer (+feat and "
             "+both+feat with the gamers' stats), and blends, against the first arm (-> prior_offline.txt/csv)")
    po_parser.add_argument("--history", required=True, metavar="CSV",
                           help="a match_history.csv (eAMFCalibrator history's, or nb2/AMFELO.csv)")
    po_parser.add_argument("--since", required=True, metavar="TS", help="the window's start, 'YYYY-MM-DD HH:MM:SS'")
    po_parser.add_argument("--until", metavar="TS", help="the window's end (default: the history's)")
    po_parser.add_argument("--arm", action="append", metavar="NAME=SOURCE[#MODE]",
                           help="a prior (repeatable), as prior-backtest's; the first is the baseline. "
                                "nb2-history's nb2_history.csv is one")
    po_parser.add_argument("--blend", action="append", metavar="NAME=VARIANT,VARIANT",
                           help="a blend of variants (repeatable): their moneylines and totals averaged, "
                                "e.g. avg=nb2+infer,glmer+both")
    po_parser.add_argument("--features", metavar="CSV",
                           help="the gamers' per-match stats for the feature layer (default: player-stats' "
                                "player_match_stats.csv in --out; 'none' for none)")
    po_parser.add_argument("--draws", type=int, default=40000, metavar="N",
                           help="simulated scorelines per match (default 40000)")
    po_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    nh_parser = sub.add_parser(
        "nb2-history",
        help="NB2's historical pass: refitted every day from --since (default 2026-01-01) as live, no layers; "
             "every match priced by its own day's fit (expected points, margin, total, moneyline) beside its "
             "metadata (-> nb2_history.csv, graded month by month in nb2_history.txt)")
    nh_parser.add_argument("--since", metavar="DATE", help="the first day priced (default 2026-01-01)")
    nh_parser.add_argument("--until", metavar="DATE", help="the day after the last priced (default: the history's end)")
    nh_parser.add_argument("--history", metavar="CSV",
                           help="a match_history.csv (eAMFCalibrator history's) to fit on: no Snowflake. "
                                "Without it the history is fetched")
    nh_parser.add_argument("--half-life", type=float, metavar="DAYS",
                           help="NB2's half-life (NBRatingTrial --half-life-days; default its own, 60)")
    nh_parser.add_argument("--workers", type=int, default=4, help="fits run side by side (default 4)")
    nh_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

    bs_parser = sub.add_parser(
        "backtest-suite", parents=[shared],
        help="every pre-match prior variant over one window: NB2 refitted daily at each half-life, glmer, each "
             "with the rematch, inference and feature layers; the inputs audited for leaks, a scramble and "
             "determinism check on the layers, echo and shuffled controls, prior-backtest and prior-replay off "
             "one read, and one ranked summary (-> out/suite/)")
    bs_parser.add_argument("--config", metavar="JSON",
                           help="the suite's settings (backtest_suite.DEFAULTS: nb2_half_lives, glmer, arms, "
                                "features, periods, ...)")
    bs_parser.add_argument("--steps", default="history,fits,audit,backtest,replay,summary",
                           help="the steps to run, comma-separated (default all: history,fits,audit,backtest,"
                                "replay,summary)")
    bs_parser.add_argument("--keep-going", action="store_true", help="carry on past a failed audit check")
    bs_parser.add_argument("--history", metavar="CSV",
                           help="the history step takes this match_history.csv (eAMFCalibrator history's) "
                                "instead of fetching one: no Snowflake")
    bs_parser.add_argument("--fresh", action="store_true",
                           help="empty the suite's query cache first (every query asked of Snowflake again)")
    bs_parser.add_argument("--out", help=f"the suite's directory (default: {DEFAULT_OUT}/suite)")

    ps_parser = sub.add_parser(
        "player-stats", parents=[shared],
        help="every gamer's stats as h2hggl.com shows them, rebuilt from the match history (played, "
             "won, lost, form, points) and SCOUTING_FULL (touchdowns, red zone, 3rd and 4th downs, "
             "yards, time of possession, two-point tries, punts, fouls, turnovers on downs), one row "
             "per match and side kept in out/player_match_stats.csv (a rerun reads only new matches); "
             "--check sets them against the site's (-> player_stats.txt/csv, player_stats_check.csv). "
             "--since limits the matches read (default: every one)")
    ps_parser.add_argument("--check", metavar="CSV",
                           help="the site's players.csv (analysis/h2hggl_api.py) to check against")
    ps_parser.add_argument("--refresh", action="store_true", help="read every match again")
    ps_parser.add_argument("--out", help=f"output directory (default: {DEFAULT_OUT})")

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


def safe_console():
    """Printed through a pipe on Windows (Tee-Object), the output is in the locale's code page
    (cp1252), which has no Δ or ±: a character it can't take prints as ? rather than stopping the
    report halfway."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None):
    safe_console()
    args = build_parser().parse_args(argv)
    if args.command in ("scouting", "timeouts") and not (args.days or args.since):
        args.days = 30          # the scouting export defaults to the last month
    if args.command == "refresh" and not (args.days or args.since):
        args.since = config.REFRESH_SINCE
    apply_overrides(args)
    if args.command == "refresh":
        return cmd_refresh(args)
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
    if args.command == "player-stats":
        return cmd_player_stats(args)
    if args.command == "prior-backtest":
        return cmd_prior_backtest(args)
    if args.command == "prior-replay":
        return cmd_prior_replay(args)
    if args.command == "backtest-suite":
        return cmd_backtest_suite(args)
    if args.command == "nb2-history":
        return cmd_nb2_history(args)
    if args.command == "prior-offline":
        return cmd_prior_offline(args)
    if args.command == "walkthrough":
        return cmd_walkthrough(args)
    if args.command == "replay-ui":
        return cmd_replay_ui(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "pair-check":
        return cmd_pair_check(args)
    if args.command == "money-layer":
        return cmd_money_layer(args)
    if args.command == "prior-bets":
        return cmd_prior_bets(args)
    if args.command == "ml-states":
        return cmd_ml_states(args)
    if args.command == "form-prior":
        return cmd_form_prior(args)
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
