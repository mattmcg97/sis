"""eAMFModel: eAMF moneyline, spread and total priced by simulating the rest of each game play
by play, off PLAY_OVER snapshots on the real game clock.

  profiles SNAPS.csv     player profiles off the snapshots: pace, 4th-down
                         aggression, clock milking
  v14-build / v14        v14's play tables, pre-match grid (glmer or NB2, --history) and
                         player profiles off PLAY_OVER snapshots, built in one joint fit
                         with a calibration report, and v14 scored against prod
  v21-build              v2.1: each snap's time, then its distance, from real snaps (the
                         live model; v21-report, v21-endgame and the rest read its builds)
  prior-daily            refit the pre-match model every day of a window, as live, and point
                         builds at the daily fits
  form-layer DIR         switch the post-game form layer on or off for builds (form_layer.py)
  form-trace             one gamer's matches as the pre-match model priced them and as the
                         post-game form layer moves them
  trader                 a local page to click through a game and see a version's prices
  v14-variants DIR       copies of a v14 build pricing off other priors: without the shrink,
                         with the form layer, NB2 (refitted daily) -- each a calibrator candidate
  v14-priors             chosen matches as prod and each v14 variant price them: prior and
                         kick-off expected points, prices at prod's lines, what the bets kept
  prior-vs-sim           a window's matches as the pre-match prior prices them and as v14's
                         kick-off does off it, both graded on the results (calibration)
  remaining SNAPS.csv    a version's points still to come against what really came

The calibrator runs a version as a stream in its own right:
  python -m eAMFCalibrator report --candidate v21 --v21-model v21_model
"""

import argparse
import json
import os
import sys

from . import grading, playover, players


def cmd_profiles(args):
    data = playover.load(args.snapshots)
    handles = players.load_handles(args.handles or args.snapshots)
    if args.half:
        codes = sorted(data)[0::2] if args.half == "train" else sorted(data)[1::2]
        data = {c: data[c] for c in codes}
    book = players.build(data, handles)
    book.save(args.out)
    profiled = [p for p in book.players.values() if p.plays]
    print(f"\n  {len(book.players)} players from {len(data):,} matches -> {args.out}")
    for name, p in sorted(book.players.items(), key=lambda kv: -kv[1].plays)[:args.show]:
        print(f"  {name:<16} plays {p.plays:>5}  pace {p.pace:5.3f}   4th downs {p.fourth_downs:>4}"
              f"  aggression {p.aggression:+5.2f}   late-lead plays {p.milk_plays:>4}  milk {p.milk:5.3f}")
    return profiled


def _half(path, half):
    codes = sorted(playover.load(path))
    return codes[0::2] if half == "train" else codes[1::2] if half == "test" else codes


def _sim_module(name):
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit(f"{name} needs numpy:  py -m pip install numpy   (or python -m pip ...)")
    import importlib
    return importlib.import_module(f".{name}", __package__)


def cmd_sim_build(args):
    name = args.version
    model = _sim_module(name)
    import datetime as dt
    if name in OPTIMUM:
        _optimum_inputs(args)            # before the timeouts are read: it finds timeouts.csv
    data = playover.load(args.snapshots)
    if getattr(args, "timeouts", None):
        found = playover.annotate_timeouts(data, args.timeouts)
        print(f"  timeouts: {found:,} real calls placed before their snaps ({args.timeouts})")
    keep = set(_half(args.snapshots, args.half))
    if args.until:
        from .remaining import day
        cut = dt.date.fromisoformat(args.until)
        keep = {c for c in keep if day(c) < cut}
        print(f"  building on {len(keep):,} matches before {cut}")
    handles = players.load_handles(args.handles) if args.handles else None
    history = before = None
    if args.history:
        from . import nb2_prior
        history = nb2_prior.load_history(args.history)
        if args.before or args.until:
            before = dt.datetime.fromisoformat(args.before or args.until)
    if history is None:
        raise SystemExit(f"{name}-build needs --history: {name} takes every match's prior from its own "
                         "NB2 pre-match model, never from GAMEPLAI's prices")
    if getattr(args, "in_play", False):
        model.IN_PLAY_FIT = True
    if getattr(args, "sim_workers", None):
        model.BUILD_WORKERS = max(1, args.sim_workers)
    if getattr(args, "red_zone", None) == "off":
        model.sim.RED_ZONE_FIT = False
    elif getattr(args, "red_zone", None):
        model.sim.RED_ZONE_FIT, model.sim.RED_ZONE_MODE = True, args.red_zone
    prior = getattr(args, "prior", None)            # which pre-match model --history fits
    model.build({c: rows for c, rows in data.items() if c in keep}, args.out, handles=handles,
                history=history, before=before, **({"prior": prior} if prior else {}))
    players_file = f"{name}players.json"
    print(f"  wrote {args.out}/{name}tables.npz, {args.out}/{name}grid.npz"
          f"{' and ' + players_file if os.path.exists(os.path.join(args.out, players_file)) else ''}")
    if name in OPTIMUM and not args.no_daily:
        if before is None:
            from .remaining import day
            before = dt.datetime.combine(max(day(c) for c in keep) + dt.timedelta(days=1), dt.time())
        _daily(args.out, history, prior or "nb2", before.date(), workers=args.workers)


# v14's build is the optimum build by default: every input from the calibrator's `refresh`
# (eAMFCalibrator/out), every snapshot before --until, the glmer pre-match model (glmer/fit.R's
# defaults) and that model refitted for every day from the build's cut-off to the history's last
# day, each fit on every result settled before its day began -- as a nightly refit would live.
OPTIMUM = ("v14",)
DATA = {"snapshots": os.path.join("eAMFCalibrator", "out", "scouting_playover.csv"),
        "history": os.path.join("eAMFCalibrator", "out", "match_history.csv"),
        "timeouts": os.path.join("eAMFCalibrator", "out", "timeouts.csv")}


def _optimum_inputs(args):
    """The optimum build's inputs: the history (and its handles) and the timeouts from the
    calibrator's out directory when not given."""
    if not args.history and os.path.exists(DATA["history"]):
        args.history = DATA["history"]
    if not args.handles and args.history:
        args.handles = args.history
    if not getattr(args, "timeouts", None) and os.path.exists(DATA["timeouts"]):
        args.timeouts = DATA["timeouts"]


def _daily(model_dir, history, prior, since, until=None, workers=2):
    """The build's pre-match model refitted for every day from `since` (the build's cut-off) to
    `until` (default: the history's last match day), into model_dir/<prior>_daily, and the build
    attached to it. The cut-off day's fit is the build's own, copied rather than refitted; days
    already fitted are kept."""
    import shutil
    from . import nb2_prior, rolling_prior, v14
    days = [s.date() for s in (nb2_prior._start(r) for r in history) if s is not None]
    until = until or (max(days) if days else since)
    if until < since:
        print(f"  daily pre-match fits: none (the history ends {until}, before the cut-off {since})")
        return None
    out = os.path.join(model_dir, f"{prior}_daily")
    own, first = os.path.join(model_dir, prior), os.path.join(out, since.isoformat())
    if v14.PRIORS[prior].exists(own) and not os.path.exists(first) \
            and v14.PRIORS[prior](own).meta["before"][:10] == since.isoformat():
        os.makedirs(out, exist_ok=True)
        shutil.copytree(own, first)
    print(f"  daily pre-match fits: {prior} for every day from {since} to {until}")
    rolling = rolling_prior.fit(history, prior, out, since, until, workers=workers)
    rolling_prior.attach(model_dir, out)
    print(f"  {rolling.describe()}; {model_dir} prices each match from its own day's fit")
    return rolling


def cmd_daily(args):
    """The optimum build's daily pre-match fits brought up to the refreshed history's last day."""
    import datetime as dt
    from . import nb2_prior
    name = args.version
    with open(os.path.join(args.model, f"{name}prior.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    prior = meta.get("prior", "nb2")
    model = _sim_module(name)
    own = os.path.join(args.model, prior)
    if not model.PRIORS[prior].exists(own):
        raise SystemExit(f"{args.model}: no {prior} fit of its own to start the days from")
    since = dt.datetime.fromisoformat(model.PRIORS[prior](own).meta["before"][:19]).date()
    history = nb2_prior.load_history(args.history or DATA["history"])
    until = dt.date.fromisoformat(args.until) if args.until else None
    _daily(args.model, history, prior, since, until, workers=args.workers)
    return 0


def cmd_sim(args):
    name = args.version
    model = _sim_module(name)
    variants = [model.Variant(name)]
    if args.without_profiles:
        variants.append(model.Variant(f"{name}_no_profiles", profiles=False, pace=False))
    handles = players.load_handles(args.handles) if args.handles else None
    matches = _half(args.snapshots, args.half)
    if args.limit:
        matches = matches[:args.limit]
    history = None
    if args.history:
        from . import nb2_prior
        history = nb2_prior.load_history(args.history)
    graded, skipped = model.run(args.snapshots, os.path.join(args.model, f"{name}tables.npz"),
                                os.path.join(args.model, f"{name}grid.npz"), variants, matches=matches,
                                n_paths=args.paths, workers=args.workers, handles=handles,
                                history=history)
    names = [v.name for v in variants]
    print(f"\n  {len(graded):,} graded PLAY_OVER quotes across {len({g[0] for g in graded}):,} matches")
    for reason, n in skipped.most_common():
        print(f"  skipped, {reason.replace('_', ' ')}: {n:,}")
    keys = {
        "market": lambda r: grading.GROUPS[r.market_id],
        "period-market": lambda r: (r.period if r.period and r.period <= 4 else "OT",
                                    grading.GROUPS[r.market_id]),
        "kind": lambda r: r.kind,
    }
    for by in args.by.split(","):
        summary = grading.summarise(graded, names, key=keys[by], n_boot=args.boot)
        grading.print_summary(summary, names, f"Brier by {by} ({name}, PLAY_OVER snapshots)")


def cmd_remaining(args):
    import datetime as dt
    from . import remaining
    history = None
    if args.history:
        from . import nb2_prior
        history = nb2_prior.load_history(args.history)
    handles = players.load_handles(args.handles) if args.handles else None
    rows = remaining.price(args.snapshots, args.version, args.model or f"{args.version}_model",
                           since=dt.date.fromisoformat(args.since) if args.since else None,
                           until=dt.date.fromisoformat(args.until) if args.until else None,
                           n_paths=args.paths, workers=args.workers, history=history,
                           handles=handles, limit=args.limit, drive=args.drive,
                           timeouts=getattr(args, "timeouts", None))
    print(f"\n  {args.version}: {len(rows):,} PLAY_OVER snapshots across "
          f"{len({r[0] for r in rows}):,} matches")
    if args.drive:
        print("\n".join(remaining.summary(rows, remaining.DRIVE_VALUES, "the rest of the drive")))
    else:
        print("\n".join(remaining.summary(rows) + remaining.over_calibration(rows)))
    path = args.out or f"remaining_{args.version}.csv"
    remaining.write_csv(path, rows)
    print(f"\n  -> {path}")


def cmd_trader(args):
    from . import trader
    trader.serve(args.model, port=args.port, paths=args.paths, margin=args.margin,
                 browser=not args.no_browser, version=args.version, prior=args.prior,
                 nudge_share=args.nudge_share, drives_share=args.drives_share, pace=not args.no_pace)


def cmd_prior_vs_sim(args):
    import datetime as dt
    import os
    from . import nb2_prior, prior_check
    since = dt.date.fromisoformat(args.since)
    until = dt.date.fromisoformat(args.until) if args.until else since
    history = nb2_prior.load_history(args.history)
    results = prior_check.run(args.model, history, since, until, n_paths=args.paths,
                              workers=args.workers or max(1, (os.cpu_count() or 2) - 1), seed=args.seed)
    lines = prior_check.report(results, since, until, prior_check.version_of(args.model))
    print("\n".join(lines))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "prior_vs_sim.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        prior_check.write_csv(os.path.join(args.out, "prior_vs_sim.csv"), results)
        print(f"\n  -> {args.out}/prior_vs_sim.txt and prior_vs_sim.csv")
    return 0


def cmd_strength_check(args):
    import datetime as dt
    from . import strength_check as sc
    since = dt.date.fromisoformat(args.since)
    until = dt.date.fromisoformat(args.until) if args.until else since
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print("  A: the sweep from kick-off", flush=True)
    rows = sc.sweep(args.model, n_paths=args.sweep_paths, workers=workers)
    results = []
    if not args.no_matches:
        history = sc.history_rows(args.history or DATA["history"])
        codes = sc.window_codes(history, since, until)
        print(f"  B and C: reading the snapshots of {len(codes):,} matches", flush=True)
        snaps = sc.load_snapshots(args.snapshots or DATA["snapshots"], codes)
        print("  B and C: pricing the window's matches, with the day's draw and with fixed strengths", flush=True)
        results = sc.run(args.model, history, snaps, since, until, n_paths=args.paths, workers=workers)
    lines = sc.report(rows, results, since, until)
    print("\n".join(lines))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "strength_check.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        if results:
            sc.write_csv(os.path.join(args.out, "strength_check.csv"), results)
        print(f"\n  -> {args.out}/strength_check.txt" + (" and strength_check.csv" if results else ""))
    return 0


def cmd_v21_build(args):
    import datetime as dt
    import time
    from . import nb2_prior, v21
    from .remaining import day
    t0 = time.time()
    data = playover.load(args.snapshots)
    timeouts = args.timeouts or (DATA["timeouts"] if os.path.exists(DATA["timeouts"]) else None)
    if timeouts and not args.no_timeouts:
        found = playover.annotate_timeouts(data, timeouts)
        print(f"  timeouts: {found:,} real calls placed before their snaps ({timeouts})")
    if args.until:
        cut = dt.date.fromisoformat(args.until)
        data = {c: rows for c, rows in data.items() if day(c) < cut}
        print(f"  building on {len(data):,} matches before {cut}")
    history = nb2_prior.load_history(args.history) if args.history and os.path.exists(args.history) else None
    model = v21.build(data, history, mode=args.mode, min_records=args.min_records, refine=not args.no_refine,
                      gamer_decisions=not args.no_gamer_decisions, rules=not args.no_rules, split=args.split,
                      key=args.key, blend=args.blend, field=args.field, decision_blend=args.decision_blend,
                      clock=args.clock, tilt_prior=args.tilt_prior, tilt_scale=args.tilt_scale,
                      tilt_prior_time=args.tilt_prior_time)
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, v21.MODEL_FILE))
    print(f"  wrote {args.out}/{v21.MODEL_FILE} ({time.time() - t0:.0f}s)")
    return 0


def cmd_v21_tilt_tune(args):
    import datetime as dt
    from . import nb2_prior, v21
    from .remaining import day
    model = v21.Model.load(os.path.join(args.model, v21.MODEL_FILE))
    if getattr(model, "mode", None) != "tilt":
        raise SystemExit(f"{args.model} is a {getattr(model, 'mode', '?')} build: the tilts are a tilt build's "
                         "(v21-build --mode tilt)")
    data = playover.load(args.snapshots)
    timeouts = args.timeouts or (DATA["timeouts"] if os.path.exists(DATA["timeouts"]) else None)
    if timeouts:
        playover.annotate_timeouts(data, timeouts)
    cut = dt.date.fromisoformat(args.test_from)
    end = dt.date.fromisoformat(args.test_until) if args.test_until else None
    train = {c: rows for c, rows in data.items() if day(c) < cut}
    test = {c: rows for c, rows in data.items() if day(c) >= cut and (end is None or day(c) < end)}
    if not train or not test:
        raise SystemExit(f"no matches {'before' if not train else 'from'} {cut} in {args.snapshots}")
    history = nb2_prior.load_history(args.history) if args.history and os.path.exists(args.history) else None
    priors = [float(x) for x in args.priors.split(",")]
    scales = [s.strip() for s in args.scales.split(",")]
    print(f"  {args.model}: tilts refitted on {len(train):,} matches before {cut}, scored on {len(test):,}"
          f" from it{f' to {end}' if end else ''}; priors {', '.join(f'{p:g}' for p in priors)}", flush=True)
    rows = v21.tune_tilts(model, train, test, history, priors, scales)
    n_train = len(v21.real_snaps(train, history))
    n_test = len(v21.real_snaps(test, history))
    lines = v21.tune_lines(rows, n_train, n_test)
    print("\n".join(lines))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return 0


def cmd_v21_sweep(args):
    import datetime as dt
    from . import nb2_prior, v21_rules
    from .remaining import day
    data = playover.load(args.snapshots)
    timeouts = args.timeouts or (DATA["timeouts"] if os.path.exists(DATA["timeouts"]) else None)
    if timeouts:
        found = playover.annotate_timeouts(data, timeouts)
        print(f"  timeouts: {found:,} real calls placed before their snaps ({timeouts})")
    else:
        print("  no timeouts file: the timeouts' rules can't be swept, and the others don't see them")
    if args.since:
        data = {c: rows for c, rows in data.items() if day(c) >= dt.date.fromisoformat(args.since)}
    if args.until:
        data = {c: rows for c, rows in data.items() if day(c) < dt.date.fromisoformat(args.until)}
    history = nb2_prior.load_history(args.history) if args.history and os.path.exists(args.history) else None
    print(f"  sweeping {len(data):,} matches", flush=True)
    found, auto, counts = v21_rules.sweep(data, history, min_region=args.min_region, rare=args.rare)
    lines = v21_rules.report(found, auto, counts, args.min_region, args.rare)
    print("\n".join(lines))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "v21_sweep.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        v21_rules.write_csv(os.path.join(args.out, "v21_sweep.csv"), found)
        print(f"\n  -> {args.out}/v21_sweep.txt and v21_sweep.csv")
    return 0


def cmd_v21_audit(args):
    import datetime as dt
    import time
    from . import loading as sc, v21_audit
    t0 = time.time()
    since = dt.date.fromisoformat(args.since)
    until = dt.date.fromisoformat(args.until) if args.until else since
    history = sc.history_rows(args.history)
    codes = sc.window_codes(history, since, until)
    print(f"  reading the snapshots of {len(codes):,} matches", flush=True)
    snaps = sc.load_snapshots(args.snapshots, codes)
    timeouts = args.timeouts or (DATA["timeouts"] if os.path.exists(DATA["timeouts"]) else None)
    if timeouts:
        playover.annotate_timeouts(snaps, timeouts)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"  auditing {len(snaps):,} matches ({workers} processes)", flush=True)
    out = v21_audit.run(args.model, snaps, history, n_paths=args.paths, state_paths=args.state_paths,
                        every=args.every, workers=workers, since=since, until=until)
    lines = v21_audit.report(out)
    print("\n".join(lines))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "v21_audit.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n  -> {args.out}/v21_audit.txt")
    print(f"  ({time.time() - t0:.0f}s)")
    return 0


def cmd_v21_coverage(args):
    import datetime as dt
    import time
    from . import loading as sc, v21_coverage
    t0 = time.time()
    since = dt.date.fromisoformat(args.since)
    until = dt.date.fromisoformat(args.until) if args.until else since
    history = sc.history_rows(args.history)
    codes = sc.window_codes(history, since, until)
    print(f"  reading the snapshots of {len(codes):,} matches", flush=True)
    snaps = sc.load_snapshots(args.snapshots, codes)
    timeouts = args.timeouts or (DATA["timeouts"] if os.path.exists(DATA["timeouts"]) else None)
    if timeouts:
        playover.annotate_timeouts(snaps, timeouts)
    out = v21_coverage.run(args.model, snaps, history)
    lines = v21_coverage.write(out, args.out) if args.out else v21_coverage.report(out)
    print("\n".join(lines))
    if args.out:
        print(f"\n  -> {args.out}/coverage.txt, coverage_states.csv, coverage_decisions.csv")
    print(f"  ({time.time() - t0:.0f}s)")
    return 0


def cmd_v21_report(args):
    import datetime as dt
    import time
    from . import loading as sc, v21_report
    t0 = time.time()
    since = dt.date.fromisoformat(args.since)
    until = dt.date.fromisoformat(args.until) if args.until else since
    history = sc.history_rows(args.history)
    codes = sc.window_codes(history, since, until)
    print(f"  reading the snapshots of {len(codes):,} matches", flush=True)
    snaps = sc.load_snapshots(args.snapshots, codes)
    timeouts = args.timeouts or (DATA["timeouts"] if os.path.exists(DATA["timeouts"]) else None)
    if timeouts:
        playover.annotate_timeouts(snaps, timeouts)
    rows = [r for r in history if r["MATCH_CODE"] in snaps]
    from . import v21_stream
    prior = args.prior_model or v21_stream.default_prior()
    if prior is None:
        raise SystemExit("no prior: build the form table (python -m eAMFCalibrator form-prior) or give --prior-model")
    print(f"  the inputs: {prior}'s pre-match expected points ({args.inputs})", flush=True)
    means, books = v21_report.prior_inputs(prior, rows, history, which=args.inputs)
    drives = None
    if args.drives is not None:
        drives = v21_stream.drives_table(args.drives or None, prior)
        have = sum(1 for r in rows if r["MATCH_CODE"] in drives)
        print(f"  the expected drives: {args.drives or prior}, {have:,} of {len(rows):,} matches"
              + (", the pace nudge meeting them" if args.nudge else " (set against the sim's; --nudge: the pace nudge)"),
              flush=True)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"  simulating {len(rows):,} matches from kick-off and from every {args.every} real state"
          f" ({workers} processes)", flush=True)
    out = v21_report.run(args.model, snaps, history, means, since, until, n_paths=args.paths,
                         state_paths=args.state_paths, every=args.every, workers=workers, prior_books=books,
                         nudge=args.nudge, nudge_share=args.nudge_share, drives=drives, drives_share=args.drives_share)
    lines = v21_report.report(out)
    print("\n".join(lines))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "v21_report.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        v21_report.write_csv(os.path.join(args.out, "v21_matches.csv"), out)
        v21_report.write_states_csv(os.path.join(args.out, "v21_states.csv"), out)
        print(f"\n  -> {args.out}/v21_report.txt, v21_matches.csv and v21_states.csv")
    print(f"  ({time.time() - t0:.0f}s)")
    return 0


def cmd_v21_endgame(args):
    import datetime as dt
    import time
    from . import loading as sc, v21_endgame, v21_report, v21_stream
    t0 = time.time()
    since = dt.date.fromisoformat(args.since)
    until = dt.date.fromisoformat(args.until) if args.until else since
    history = sc.history_rows(args.history)
    codes = sc.window_codes(history, since, until)
    print(f"  reading the snapshots of {len(codes):,} matches", flush=True)
    snaps = sc.load_snapshots(args.snapshots, codes)
    timeouts = args.timeouts or (DATA["timeouts"] if os.path.exists(DATA["timeouts"]) else None)
    if timeouts:
        playover.annotate_timeouts(snaps, timeouts)
    rows = [r for r in history if r["MATCH_CODE"] in snaps]
    prior = args.prior_model or v21_stream.default_prior()
    if prior is None:
        raise SystemExit("no prior: build the form table (python -m eAMFCalibrator form-prior) or give --prior-model")
    print(f"  the inputs: {prior}'s pre-match expected points ({args.inputs})", flush=True)
    means, _ = v21_report.prior_inputs(prior, rows, history, which=args.inputs)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"  simulating {len(rows):,} matches from kick-off and from their late states ({workers} processes)", flush=True)
    out = v21_endgame.run(args.model, snaps, history, means, since, until, kick_paths=args.paths,
                          state_paths=args.state_paths, workers=workers, nudge=args.nudge, nudge_share=args.nudge_share)
    lines = v21_endgame.report(out)
    print("\n".join(lines))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "v21_endgame.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        v21_endgame.write_states_csv(os.path.join(args.out, "v21_endgame_states.csv"), out)
        print(f"\n  -> {args.out}/v21_endgame.txt and v21_endgame_states.csv")
    print(f"  ({time.time() - t0:.0f}s)")
    return 0


def cmd_gamer_dig(args):
    import csv as csv_
    import datetime as dt
    from . import gamer_dig, nb2_prior
    data = playover.load(args.snapshots)
    history = nb2_prior.load_history(args.history)
    bets = gamer_dig.read_bets(args.bets) if args.bets else None
    since = dt.date.fromisoformat(args.since) if args.since else None
    until = dt.date.fromisoformat(args.until) if args.until else None
    lines, rows = gamer_dig.run(data, history, bets, since, until, args.gamers.split(",") if args.gamers else None)
    print("\n".join(lines))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "gamer_dig.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        if rows:
            with open(os.path.join(args.out, "gamer_dig_bets.csv"), "w", newline="", encoding="utf-8") as fh:
                w = csv_.DictWriter(fh, list(rows[0]))
                w.writeheader()
                w.writerows(rows)
        print(f"\n  -> {args.out}/gamer_dig.txt" + (" and gamer_dig_bets.csv" if rows else ""))
    return 0


def cmd_v21_gamers(args):
    from . import v21, v21_gamers, v21_stream
    model = v21.Model.load(v21_stream._model_file(args.model))
    rows, summary = v21_gamers.run(model, args.paths)
    print("\n".join(summary))
    print("\n".join(v21_gamers.lines_for(rows, args.gamers.split(",") if args.gamers else None)))
    if args.out:
        print(f"\n  -> {v21_gamers.write(args.out, rows)}")
    return 0


def cmd_v21_react_fit(args):
    import datetime as dt
    from . import nb2_prior, v21_react, v21_react_fit, v21_stream
    data = playover.load(args.snapshots)
    history = nb2_prior.load_history(args.history)
    model_file = v21_stream._model_file(args.model) if args.model else None
    since = dt.date.fromisoformat(args.since) if args.since else None
    until = dt.date.fromisoformat(args.until)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    lines, got = v21_react_fit.run(data, history, model_file, since, until, args.test_share, args.matches,
                                   args.paths, workers)
    print("\n".join(lines))
    out = args.out or (os.path.join(args.model, v21_react.WP_FILE) if args.model else v21_react.WP_FILE)
    print(f"\n  -> {v21_react_fit.write(out, got)}")
    if got.get("gamers"):
        print(f"  -> {v21_react_fit.write(os.path.join(os.path.dirname(out) or '.', v21_react.GAMERS_FILE), got['gamers'])}"
              "  (each gamer's own scale, for v21-gsnapT, v21-gptsT, v21-gwpaC)")
    return 0


def cmd_ats(args):
    import datetime as dt
    from . import ats, nb2_prior
    print(f"  reading {args.snapshots}", flush=True)
    data = playover.load(args.snapshots)
    history = nb2_prior.load_history(args.history)
    since = dt.date.fromisoformat(args.since) if args.since else None
    until = dt.date.fromisoformat(args.until) if args.until else None
    out = ats.run(data, history, since, until, args.min_matches, args.min_half)
    lines = ats.write(args.out, out)
    print("\n".join(lines))
    print(f"\n  -> {os.path.join(args.out, 'ats.txt')} and ats_gamers.csv")
    return 0


def cmd_playstyle(args):
    import datetime as dt
    from . import nb2_prior, playstyle, table_prior
    print(f"  reading {args.snapshots}", flush=True)
    data = playover.load(args.snapshots)
    history = nb2_prior.load_history(args.history)
    prior = table_prior.TablePrior(args.prior).points if args.prior else None
    since = dt.date.fromisoformat(args.since) if args.since else None
    until = dt.date.fromisoformat(args.until) if args.until else None
    out = playstyle.run(data, history, prior, since, until, args.min_matches, args.min_half, args.boot)
    lines = playstyle.write(args.out, out)
    print("\n".join(lines))
    print(f"\n  -> {os.path.join(args.out, 'playstyle.txt')} and playstyle_gamers.csv")
    return 0


def cmd_v21_timing(args):
    import datetime as dt
    from . import loading as sc, v21_timing
    since = dt.date.fromisoformat(args.since)
    until = dt.date.fromisoformat(args.until) if args.until else since
    history = sc.history_rows(args.history)
    codes = sc.window_codes(history, since, until)
    print(f"  reading the snapshots of {len(codes):,} matches", flush=True)
    snaps = sc.load_snapshots(args.snapshots, codes)
    timeouts = args.timeouts or (DATA["timeouts"] if os.path.exists(DATA["timeouts"]) else None)
    if timeouts:
        playover.annotate_timeouts(snaps, timeouts)
    print(f"  pricing {min(args.matches, len(snaps))} of them as the calibrator's v21 stream does", flush=True)
    out = v21_timing.run(args.model, snaps, history, n_matches=args.matches, n_paths=args.paths,
                         workers=args.workers, profile=not args.no_profile, compare=args.compare)
    print("\n".join(v21_timing.report(out)))
    return 0


def cmd_v14_variants(args):
    import datetime as dt
    from . import nb2_prior, prior_variants
    history = nb2_prior.load_history(args.history or DATA["history"])
    until = dt.date.fromisoformat(args.until) if args.until else None
    made = prior_variants.make(args.model, history, variants=[v.strip() for v in args.variants.split(",") if v.strip()],
                               workers=args.workers, until=until, daily=None if args.no_daily else _daily)
    for name, d in made.items():
        print(f"  {name}: {d}")
    tags = ",".join(f"v14-{name}={d}" for name, d in made.items())
    print(f"\n  as calibrator candidates: --candidate v14,{tags} --v14-model {args.model}")
    return 0


def cmd_v14_priors(args):
    import os
    from . import nb2_prior, prior_variants
    history = nb2_prior.load_history(args.history or DATA["history"])
    codes = _match_list(args.matches)
    models = {}
    for item in args.models.split(","):
        label, eq, directory = item.strip().partition("=")
        models[label] = directory if eq else label
    results = prior_variants.compare(models, history, codes, snapshots=args.snapshots or DATA["snapshots"],
                                     bets_csv=args.bets, n_paths=args.paths, seed=args.seed)
    lines = prior_variants.report(results, list(models))
    print("\n".join(lines))
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "v14_priors.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        prior_variants.write_csv(os.path.join(args.out, "v14_priors.csv"), results, list(models))
        print(f"\n  -> {args.out}/v14_priors.txt and v14_priors.csv")
    return 0


def _match_list(value):
    """Match codes from a file (one a line, the first word; # starts a comment) or a comma-separated
    list."""
    if os.path.exists(value):
        with open(value, encoding="utf-8") as fh:
            return [w[0] for w in (line.split("#", 1)[0].split() for line in fh) if w]
    return [c.strip() for c in value.split(",") if c.strip()]


def cmd_prior_daily(args):
    import datetime as dt
    from . import nb2_prior, rolling_prior
    if args.detach:
        for m in args.detach.split(","):
            rolling_prior.detach(m)
            print(f"  {m}: prices from its own single fit again")
        return 0
    if not (args.history and args.since and args.until and args.out):
        raise SystemExit("prior-daily needs --history, --since, --until and --out (or --detach)")
    history = nb2_prior.load_history(args.history)
    since, until = dt.date.fromisoformat(args.since), dt.date.fromisoformat(args.until)
    options = {}
    if args.half_life is not None:
        if args.prior != "nb2":
            raise SystemExit("--half-life is NB2's (glmer's weighting is glmer/config.R's)")
        options["half_life"] = args.half_life
    rolling = rolling_prior.fit(history, args.prior, args.out, since, until, workers=args.workers,
                                options=options)
    print(f"  {rolling.describe()}")
    for m in (args.attach or "").split(","):
        if m:
            rolling_prior.attach(m, args.out)
            print(f"  {m}: now prices each match from its own day's fit")
    return 0


def _pair(text):
    """'1.5' -> (1.5, 1.5); '1.5,1' -> (1.5, 1.0): a layer setting for the margin and the total."""
    parts = [float(x) for x in text.split(",")]
    if len(parts) not in (1, 2):
        raise SystemExit(f"{text!r}: give one number, or two (margin,total)")
    return (parts[0], parts[-1])


def _layer_settings(args):
    own = {}
    if args.tau_day is not None:
        own["tau_day"] = _pair(args.tau_day)
    if args.tau_session is not None:
        own["tau_session"] = _pair(args.tau_session)
    if args.rho is not None:
        own["rho"] = args.rho
    if args.trigger is not None:
        own["trigger"] = args.trigger
    if args.either_way:
        own["same_way"] = False
    if args.s_curve:
        own["s_curve"] = True
    return own


def cmd_form_layer(args):
    from . import form_layer
    value = True if args.on else False if args.off else None
    own = _layer_settings(args)
    if own and not args.on:
        raise SystemExit("--tau-day, --tau-session, --rho, --trigger, --s-curve and --either-way come with --on")
    for d in args.builds:
        meta = form_layer.switch(d, value, **own)
        on = form_layer.switched_on(meta)
        kept = form_layer.own_settings(meta)
        print(f"  {d}: post-game form layer {'on' if on else 'off'}"
              f"{'' if value is not None else ' (the default, form_layer.ON)'}"
              + (f", with {', '.join(f'{k} {v}' for k, v in kept.items())}" if kept else ""))
    return 0


def cmd_form_trace(args):
    import datetime as dt
    from . import form_layer, nb2_prior
    if bool(args.model) == bool(args.rolling):
        raise SystemExit("form-trace needs one of --model (a build) or --rolling (prior-daily's fits)")
    own = {}
    if args.rolling:
        from . import rolling_prior
        base = rolling_prior.Rolling(args.rolling)
    else:
        base = form_layer.base_model(args.model)
        own = form_layer.base_settings(args.model)
    since = form_layer.out_of_sample_since(base)
    if since is None:
        raise SystemExit("the pre-match model records no cut-off, so its out-of-sample prices are unknown")
    layer = form_layer.FormLayer(base, since, **dict(own, **_layer_settings(args)))
    first = dt.date.fromisoformat(args.since)
    last = dt.date.fromisoformat(args.until) if args.until else first
    history = nb2_prior.load_history(args.history)
    rows = []
    for gamer in args.gamer:
        got = form_layer.trace(layer, history, gamer, first, last)
        print("\n".join(form_layer.report(got, gamer, layer)) + "\n")
        rows += [dict(r, gamer=gamer.upper()) for r in got]
    if args.csv and rows:
        import csv
        keys = ["gamer", "code", "start", "side", "opponent", "scored", "conceded", "run", "opponent_run",
                "base_margin", "day", "session", "opponent_form", "adjust", "margin", "total_adjust"]
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"  {len(rows)} rows -> {args.csv}")
    return 0


def _layer_options(p, what):
    p.add_argument("--tau-day", metavar="M[,T]", help=f"sd of a fresh day form, margin[,total], {what}")
    p.add_argument("--tau-session", metavar="M[,T]", help=f"sd of a fresh session form, {what}")
    p.add_argument("--rho", type=float, help=f"share of a day's form carried to the next day, {what}")
    p.add_argument("--trigger", type=int, help="move the margin only once a gamer has lost (won) this "
                                               f"many in a row this session (0: always), {what}")
    p.add_argument("--s-curve", action="store_true", help="move the margin by the S-curve (a gamer's "
                   f"chance of being off or on today, form_layer.S_CURVE) in place of day and session form, {what}")
    p.add_argument("--either-way", action="store_true", help="let the margin move a gamer against the way "
                                                             f"his day is going (default: never), {what}")


def build_parser():
    """The command line's parser."""
    parser = argparse.ArgumentParser(prog="eAMFModel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("profiles", help="build player profiles (pace, 4th-down aggression, "
                                        "clock milking) from PLAY_OVER snapshots")
    p.add_argument("snapshots", help="scouting_playover.csv")
    p.add_argument("--handles", help="CSV of MATCH_CODE, PLAYER_1_HANDLE, PLAYER_2_HANDLE "
                                     "(default: the snapshot file's own handle columns)")
    p.add_argument("--out", default="player_profiles.json")
    p.add_argument("--half", choices=["train", "test"])
    p.add_argument("--show", type=int, default=20)
    p.set_defaults(func=cmd_profiles)

    for name, what in (("v14", "v14: the play-by-play simulation played close to the real game -- strength as "
                               "boom or bust, the clock as played -- built in one joint fit (see README)"),):
        p = sub.add_parser(f"{name}-build", help=what)
        if name in OPTIMUM:
            p.add_argument("snapshots", nargs="?", default=DATA["snapshots"],
                           help=f"scouting_playover.csv (default {DATA['snapshots']})")
        else:
            p.add_argument("snapshots", help="scouting_playover.csv")
        p.add_argument("--half", choices=["train", "test", "all"],
                       default="all" if name in OPTIMUM else "train")
        p.add_argument("--out", default=f"{name}_model")
        p.add_argument("--handles", help="CSV of MATCH_CODE, PLAYER_1_HANDLE, PLAYER_2_HANDLE "
                                         "(default: the export's own handle columns)")
        p.add_argument("--history", help="match history (eAMFCalibrator history's CSV, shaped like "
                                         "nb2/AMFELO.csv): fits NB2 as the pre-match model instead "
                                         "of reading prod's pre-match quotes")
        p.add_argument("--before", help="fit NB2 on history before this date (default: the day "
                                        "after the last match built on)")
        p.add_argument("--until", help="build everything -- play tables, profiles and NB2 -- on the "
                                       "matches before this date (YYYY-MM-DD), so a test after it is "
                                       "out of sample")
        p.add_argument("--in-play", action="store_true",
                       help="also fit the in-play total shift by segment of the game "
                            "(see README: off by default, it has not held up out of sample)")
        p.add_argument("--timeouts", help="the calibrator's timeouts.csv (`python -m eAMFCalibrator "
                                          "timeouts`): every real timeout, to fit when sides call them")
        p.add_argument("--prior", choices=["nb2", "glmer"], default="glmer" if name in OPTIMUM else "nb2",
                       help="the pre-match model --history fits: nb2 (nb2/) or glmer (glmer/ in R, "
                            "needs R and lme4; see glmer/README.md); default "
                            + ("glmer" if name in OPTIMUM else "nb2"))
        if name in OPTIMUM:
            p.add_argument("--no-daily", action="store_true",
                           help="price from the build's single pre-match fit, without the daily refits "
                                "from the cut-off to the history's last day (see v14-daily)")
            p.add_argument("--workers", type=int, default=2,
                           help="daily pre-match fits run side by side (default 2)")
            p.add_argument("--sim-workers", type=int,
                           help="processes for the build's long simulations (default 4); the results"
                                " are the same for any number, only the time changes")
        p.add_argument("--red-zone", choices=["hold", "tilt", "off"],
                       help="how drives finish inside the 30 by quarter and lead: hold (the "
                            "default), tilt, or off (see README)")
        p.set_defaults(func=cmd_sim_build, version=name)

        p = sub.add_parser(name, help=f"score {name} against prod on PLAY_OVER snapshots")
        p.add_argument("snapshots", help="scouting_playover.csv")
        p.add_argument("--model", default=f"{name}_model", help=f"{name}-build's output directory")
        p.add_argument("--half", choices=["train", "test", "all"], default="test")
        p.add_argument("--paths", type=int, default=2000, help="simulated games per snapshot")
        p.add_argument("--without-profiles", action="store_true",
                       help="also a variant without the player profiles")
        p.add_argument("--handles", help="CSV of MATCH_CODE, PLAYER_1_HANDLE, PLAYER_2_HANDLE")
        p.add_argument("--history", help="the graded matches' players, teams and streams (eAMFCalibrator"
                                         " history's CSV): needed when the model has NB2's pre-match")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument("--limit", type=int, help="first N matches only")
        p.add_argument("--boot", type=int, default=300)
        p.add_argument("--by", default="market,period-market,kind")
        p.set_defaults(func=cmd_sim, version=name)

    p = sub.add_parser("remaining", help="a version's points still to come against what the rest "
                                         "of each game really made, value by value, by quarter and "
                                         "game state (see remaining.py)")
    p.add_argument("snapshots", help="scouting_playover.csv")
    p.add_argument("--version", choices=["v14"], default="v14")
    p.add_argument("--model", help="the version's build directory (default <version>_model)")
    p.add_argument("--since", help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD")
    p.add_argument("--paths", type=int, default=500, help="simulated games per snapshot")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--history", help="the matches' players, teams and streams (eAMFCalibrator history's "
                                     "CSV): needed when the model has NB2's pre-match")
    p.add_argument("--handles", help="CSV of MATCH_CODE, PLAYER_1_HANDLE, PLAYER_2_HANDLE")
    p.add_argument("--limit", type=int, help="first N matches only")
    p.add_argument("--out", help="per-snapshot CSV (default remaining_<version>.csv)")
    p.add_argument("--timeouts", help="the calibrator's timeouts.csv, so each state knows the "
                                      "timeouts each side has left")
    p.add_argument("--drive", action="store_true",
                   help="the points on the rest of the drive under way at each scrimmage PLAY_OVER "
                        "(0, safety, 3, 6, 7, 8), instead of the rest of the game")
    p.set_defaults(func=cmd_remaining)

    p = sub.add_parser("v14-daily", help="bring a v14 build's daily pre-match fits up to the refreshed "
                                         "history's last day (run after eAMFCalibrator refresh)")
    p.add_argument("--model", default="v14_model", help="the v14 build (default v14_model)")
    p.add_argument("--history", help=f"the match history (default {DATA['history']})")
    p.add_argument("--until", help="last day to fit (default: the history's last match day)")
    p.add_argument("--workers", type=int, default=2, help="fits run side by side (default 2)")
    p.set_defaults(func=cmd_daily, version="v14")

    p = sub.add_parser("prior-daily", help="refit the pre-match model every day of a window, as it "
                                           "would be live, and point builds at it (rolling_prior.py)")
    p.add_argument("--history", help="eAMFCalibrator history's CSV (every match's players and finals)")
    p.add_argument("--prior", choices=["nb2", "glmer"], default="glmer")
    p.add_argument("--since", help="first match day, YYYY-MM-DD (its fit sees results before it)")
    p.add_argument("--until", help="last match day, YYYY-MM-DD")
    p.add_argument("--out", help="directory for the daily fits (days already fitted are kept)")
    p.add_argument("--attach", help="builds to price from these fits, comma-separated (each built "
                                    "with the same --prior)")
    p.add_argument("--detach", help="builds to return to their own single fit, comma-separated")
    p.add_argument("--workers", type=int, default=2, help="fits run side by side (default 2)")
    p.add_argument("--half-life", type=float, metavar="DAYS",
                   help="NB2 only: each fit's results weighted 2^(-days old / DAYS) (NBRatingTrial's "
                        "--half-life-days; default its own, 60). A directory keeps one setting")
    p.set_defaults(func=cmd_prior_daily)

    p = sub.add_parser("prior-vs-sim", help="calibration: a window's matches priced by the pre-match prior "
                                            "itself and by the build's (v14) kick-off off it, graded on the results")
    p.add_argument("--model", required=True, help="a v14 build (its prior: glmer or NB2, daily if attached)")
    p.add_argument("--history", required=True, help="the match history CSV (eAMFCalibrator history)")
    p.add_argument("--since", required=True, help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: --since)")
    p.add_argument("--paths", type=int, default=4000, help="simulated games per kick-off (default 4000)")
    p.add_argument("--workers", type=int, help="processes for the kick-offs (default: all cores but one)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="also write prior_vs_sim.txt and a per-match prior_vs_sim.csv here")
    p.set_defaults(func=cmd_prior_vs_sim)

    p = sub.add_parser("v14-variants", help="copies of a v14 build pricing off other priors (no shrink, the "
                                            "form layer, NB2 refitted daily), for the calibrator side by side")
    p.add_argument("model", nargs="?", default="v14_model", help="the v14 build (default v14_model)")
    p.add_argument("--variants", default="raw,layered,nb2", help="which: raw, layered, nb2 (default all)")
    p.add_argument("--history", help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--until", help="NB2's daily fits up to this day (default: the history's last)")
    p.add_argument("--no-daily", action="store_true", help="NB2 from its single fit at the cut-off")
    p.add_argument("--workers", type=int, default=2, help="NB2's daily fits run side by side (default 2)")
    p.set_defaults(func=cmd_v14_variants)

    p = sub.add_parser("v14-priors", help="chosen matches as prod and each v14 variant price them before "
                                          "kick-off, and what each kept of their bets")
    p.add_argument("--matches", required=True, help="a file of match codes (one a line) or codes, comma-separated")
    p.add_argument("--models", default="v14=v14_model,v14-raw=v14_model_raw,v14-layered=v14_model_layered,"
                                       "v14-nb2=v14_model_nb2",
                   help="label=build pairs, comma-separated; the labels the calibrator's candidates had")
    p.add_argument("--history", help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", help=f"the snapshot export, for prod's prices (default {DATA['snapshots']})")
    p.add_argument("--bets", help="the calibrator's bets_sim.csv with every model a candidate: each one's kept")
    p.add_argument("--paths", type=int, default=4000, help="simulated games per kick-off (default 4000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="also write v14_priors.txt and v14_priors.csv here")
    p.set_defaults(func=cmd_v14_priors)

    p = sub.add_parser("strength-check", help="a v14 build's strengths snap by snap: which kinds of play "
                                              "they move, the sim's snaps against the real ones, and the day's "
                                              "strength draw against fixed strengths, graded on the results")
    p.add_argument("--model", default="v14_model", help="the v14 build (default v14_model)")
    p.add_argument("--history", help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--since", required=True, help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: --since)")
    p.add_argument("--paths", type=int, default=2000, help="simulated games per kick-off (default 2000)")
    p.add_argument("--sweep-paths", type=int, default=4000, help="games per step of the sweep (default 4000)")
    p.add_argument("--no-matches", action="store_true", help="the sweep only (part A)")
    p.add_argument("--workers", type=int, help="processes (default: all cores but one)")
    p.add_argument("--out", help="also write strength_check.txt and a per-side strength_check.csv here")
    p.set_defaults(func=cmd_strength_check)

    p = sub.add_parser("v21-build", help="v2.1: a snap sim drawing each snap's time, then its distance given "
                                         "the time, from real snaps alone (no strength; see v21.py)")
    p.add_argument("snapshots", nargs="?", default=DATA["snapshots"],
                   help=f"scouting_playover.csv (default {DATA['snapshots']})")
    p.add_argument("--until", help="build on the matches before this date (YYYY-MM-DD)")
    p.add_argument("--out", default="v21_model", help="the build directory (default v21_model)")
    p.add_argument("--mode", choices=["full", "adjusted", "tilt"], default="full",
                   help="full: the gamers' and teams' own real snaps as draws; adjusted: the league's draws "
                        "leaned by a fitted tilt for each; tilt: the league's draws reweighted by exp(theta . "
                        "phi) on yards, touchdowns, turnovers and seconds, theta per gamer and team, offense and "
                        "defense, fitted by maximum likelihood (default full)")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV, for the gamers and "
                                                               f"teams (default {DATA['history']})")
    p.add_argument("--timeouts", help=f"the calibrator's timeouts.csv (default {DATA['timeouts']} if there)")
    p.add_argument("--no-timeouts", action="store_true", help="build without the timeouts")
    p.add_argument("--min-records", type=int, default=40, help="real snaps a draw's group needs (default 40)")
    p.add_argument("--no-refine", action="store_true",
                   help="keep the real games' snaps left in the half (no simulated pass)")
    p.add_argument("--key", choices=["full", "lean"], default="full",
                   help="the league's draws' key: full -- the play before in six classes, the part of the game, the "
                        "snaps played so far; lean -- the clock running or stopped (or the play's time), the period, "
                        "the lead, down, distance, field and snaps left only (default full)")
    p.add_argument("--blend", type=float, default=0.0,
                   help="blend the league's draws' levels: a cell of n real snaps is drawn from n / (n + BLEND) of "
                        "the time, else its parent's (cells of 5+ kept); 0 (default): the first level with "
                        "--min-records")
    p.add_argument("--field", type=float, default=0.0,
                   help="field windows: the distance draw takes its real snap from near the state's own field, "
                        "within FIELD x the yards to go (2 yards at least; the 20 nearest where fewer), so a gain "
                        "made at midfield is not played out at the 12; 0 (default): anywhere in its group "
                        "(try 0.2)")
    p.add_argument("--clock", type=float, default=0.0,
                   help="clock windows: in a half's last 2:00 the time draw takes its real snap from those with as "
                        "long left in the half, within CLOCK x the seconds left (2 at least; the 20 nearest where "
                        "fewer); 0 (default): anywhere in its group (try 0.2)")
    p.add_argument("--tilt-prior", type=float, default=200.0,
                   help="--mode tilt: each tilt shrunk toward none as if it had this many snaps (default 200; "
                        "v21-tilt-tune picks it)")
    p.add_argument("--tilt-prior-time", type=float,
                   help="--mode tilt: the time tilts' own prior, in snaps (default --tilt-prior's)")
    p.add_argument("--tilt-scale", choices=("state", "league"), default="state",
                   help="--mode tilt: what a prior snap is worth -- state: what a snap tells within its own "
                        "state, so a gamer with --tilt-prior snaps moves half way (default); league: the spread "
                        "over every snap, as builds before 22 Oct had it, several times stronger for the time")
    p.add_argument("--decision-blend", type=float, default=0.0,
                   help="blend the decisions' cells: a cell of n real decisions weighs n / (n + K) of its state's "
                        "shares, the rest its parents' (cells of 5+ kept); 0 (default): the first level with 25")
    p.add_argument("--split", choices=["snaps", "time"], default="snaps",
                   help="the draws' part of the game: snaps -- the half and the two-minute warning, the snaps "
                        "left in the half the rest; time -- the seven parts of the game (default snaps)")
    p.add_argument("--no-gamer-decisions", action="store_true",
                   help="every gamer decides (4th downs, early kicks, kick-offs, conversions, timeouts) as the "
                        "league does in the same state: no gamer's own lean")
    p.add_argument("--no-rules", action="store_true",
                   help="without the rules learned off the real decisions (options never or all but never taken "
                        "in a situation, see v21-sweep): the real shares alone")
    p.set_defaults(func=cmd_v21_build)

    p = sub.add_parser("v21-tilt-tune", help="a tilt build's tilts refitted at several priors and scales on the "
                                             "matches before a day, each scored on the matches from it")
    p.add_argument("snapshots", nargs="?", default=DATA["snapshots"],
                   help=f"scouting_playover.csv (default {DATA['snapshots']})")
    p.add_argument("--model", default="v21_model", help="a v21-build --mode tilt directory (default v21_model)")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV, for the gamers "
                                                               f"(default {DATA['history']})")
    p.add_argument("--timeouts", help=f"the calibrator's timeouts.csv (default {DATA['timeouts']} if there)")
    p.add_argument("--test-from", required=True,
                   help="the held-out matches' first day, YYYY-MM-DD: the build's --until, so the draws never saw "
                        "them; the tilts are refitted on the matches before it")
    p.add_argument("--test-until", help="held-out matches before this day only, YYYY-MM-DD")
    p.add_argument("--priors", default=",".join(str(p) for p in (25, 50, 100, 200, 400, 800)),
                   help="the priors to try, in snaps (default 25,50,100,200,400,800)")
    p.add_argument("--scales", default="state,league", help="the scales to try (default state,league)")
    p.add_argument("--out", help="also write the table here")
    p.set_defaults(func=cmd_v21_tilt_tune)

    p = sub.add_parser("v21-sweep", help="every real decision in the export swept for the situations where an "
                                         "option never happens, or all but never (and who did it anyway)")
    p.add_argument("snapshots", nargs="?", default=DATA["snapshots"],
                   help=f"scouting_playover.csv (default {DATA['snapshots']})")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV, for the gamers "
                                                               f"(default {DATA['history']})")
    p.add_argument("--timeouts", help=f"the calibrator's timeouts.csv (default {DATA['timeouts']} if there)")
    p.add_argument("--since", help="first match day, YYYY-MM-DD (default: every match)")
    p.add_argument("--until", help="matches before this day only, YYYY-MM-DD")
    p.add_argument("--min-region", type=int, default=30, help="real decisions a situation needs (default 30)")
    p.add_argument("--rare", type=float, default=0.01,
                   help="the most an option may be taken in a situation to count as all but never (default 0.01)")
    p.add_argument("--out", help="also write v21_sweep.txt and v21_sweep.csv here")
    p.set_defaults(func=cmd_v21_sweep)

    p = sub.add_parser("v21-audit", help="v2.1 down by down and by the time left against the real games: its own "
                                         "games from kick-off, and the next snap from every real state")
    p.add_argument("--model", default="v21_model", help="a v21-build directory (default v21_model)")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--timeouts", help=f"the calibrator's timeouts.csv (default {DATA['timeouts']} if there)")
    p.add_argument("--since", required=True, help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: --since)")
    p.add_argument("--paths", type=int, default=500, help="kick-off games per receiving side (default 500)")
    p.add_argument("--state-paths", type=int, default=50, help="next snaps drawn from each real state (default 50)")
    p.add_argument("--every", type=int, default=1, help="every Nth real state (default 1: all)")
    p.add_argument("--workers", type=int, help="processes (default: all cores but one)")
    p.add_argument("--out", help="also write v21_audit.txt here")
    p.set_defaults(func=cmd_v21_audit)

    p = sub.add_parser("v21-coverage", help="v2.1's coverage: for every real state, the fallback level each draw "
                                            "reaches and the real snaps behind it; the thinnest situations")
    p.add_argument("--model", default="v21_model", help="a v21-build directory (default v21_model)")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--timeouts", help=f"the calibrator's timeouts.csv (default {DATA['timeouts']} if there)")
    p.add_argument("--since", required=True, help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: --since)")
    p.add_argument("--out", help="write coverage.txt and the per-state CSVs here")
    p.set_defaults(func=cmd_v21_coverage)

    p = sub.add_parser("v21-report", help="v2.1 against the pre-match expected points, from kick-off and from "
                                          "every step of the real games, and its draws against the real snaps")
    p.add_argument("--model", default="v21_model", help="a v21-build directory (default v21_model)")
    p.add_argument("--prior-model",
                   help="a v14 build whose pre-match model gives the expected points, or a walkthrough "
                        "variant's table (eAMFCalibrator walkthrough's priors/VARIANT.csv) (default: v2.1's own --"
                        " $EAMF_V21_PRIOR, then the form table eAMFCalibrator form-prior writes, then v14_model)")
    p.add_argument("--inputs", choices=["raw", "priced"], default="raw",
                   help="raw: the pre-match model's own expected points; priced: with the build's layers")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--timeouts", help=f"the calibrator's timeouts.csv (default {DATA['timeouts']} if there)")
    p.add_argument("--since", required=True, help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: --since)")
    p.add_argument("--paths", type=int, default=1000, help="kick-off games per receiving side (default 1000)")
    p.add_argument("--state-paths", type=int, default=200, help="games from each real state (default 200; 0: none)")
    p.add_argument("--nudge", action="store_true",
                   help="a tilt build: each side's offense nudged along the strength direction until the sim's "
                        "kick-off mean meets its expected points (the switch for pricing), the same nudge in its states")
    p.add_argument("--nudge-share", type=float, default=1.0,
                   help="with --nudge: go this share of the way from the sim's own kick-off means to the inputs "
                        "(default 1: all of it)")
    p.add_argument("--drives", nargs="?", const="", metavar="CSV",
                   help="each match's expected drives (alone: the prior table's exp_drives, the walkthrough's pace; "
                        "or a table of them, MATCH_CODE and exp_drives), set against the sim's "
                        "and the real drives; with --nudge, the pace nudge: one tempo on both sides' time tilts so the "
                        "kick-off games' drives meet them, carried into the states (the betting sim's v21-drives)")
    p.add_argument("--drives-share", type=float, default=1.0,
                   help="with --drives and --nudge: go this share of the way from the sim's own drives to the "
                        "expected (default 1: all of it; the betting sim's v21-drivesS)")
    p.add_argument("--every", type=int, default=4, help="every Nth real state of a match (default 4)")
    p.add_argument("--workers", type=int, help="processes (default: all cores but one)")
    p.add_argument("--out", help="also write v21_report.txt, v21_matches.csv and v21_states.csv here")
    p.set_defaults(func=cmd_v21_report)

    p = sub.add_parser("v21-endgame", help="v2.1's endgame against the real games: overtime and how a level game "
                                           "was made level, from kick-off and from the same late states, and the "
                                           "choices between a tie and a win")
    p.add_argument("--model", default="v21_model", help="a v21-build directory (default v21_model)")
    p.add_argument("--prior-model", help="the expected points, as v21-report's --prior-model")
    p.add_argument("--inputs", choices=["raw", "priced"], default="raw", help="as v21-report's --inputs")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--timeouts", help=f"the calibrator's timeouts.csv (default {DATA['timeouts']} if there)")
    p.add_argument("--since", required=True, help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: --since)")
    p.add_argument("--paths", type=int, default=250, help="kick-off games per receiving side (default 250)")
    p.add_argument("--state-paths", type=int, default=200,
                   help="games from each real snap of Q4 within 8 points (default 200; 0: none)")
    p.add_argument("--nudge", action="store_true", help="a tilt build: nudged to the expected points, as the stream prices")
    p.add_argument("--nudge-share", type=float, default=1.0,
                   help="with --nudge: this share of the way to the inputs (default 1; the stream's 0.75)")
    p.add_argument("--workers", type=int, help="processes (default: all cores but one)")
    p.add_argument("--out", help="also write v21_endgame.txt and v21_endgame_states.csv here")
    p.set_defaults(func=cmd_v21_endgame)

    p = sub.add_parser("gamer-dig",
                       help="what the money backing particular gamers sees in their play: the targeted gamers' "
                            "profiles, whether earlier matches of the session or the match so far tell the result "
                            "beyond prod's price, and the snap each bet followed (hot-streak bets apart)")
    p.add_argument("--bets", help="a betting simulation's bets_sim.csv (the window and the money); without it, the "
                                  "earlier-matches test alone over --since to --until")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--since", help="first match day, YYYY-MM-DD (default: the bets' first)")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: the bets' last)")
    p.add_argument("--gamers", help="these gamers as the targeted ones, comma-separated (default: the money's top)")
    p.add_argument("--out", help="the report (gamer_dig.txt) and every in-play bet's row (gamer_dig_bets.csv) here")
    p.set_defaults(func=cmd_gamer_dig)

    p = sub.add_parser("v21-gamers",
                       help="how far v2.1's own draws set each gamer apart: every gamer against an average opponent "
                            "from kick-off (no nudge), the sim's points for and against and win chance, beside the "
                            "gamer's real points and tilts")
    p.add_argument("--model", default="v21_model", help="a v21-build directory (default v21_model)")
    p.add_argument("--gamers", help="these gamers' rows only, comma-separated (the summary is every gamer's)")
    p.add_argument("--paths", type=int, default=1000, help="games each gamer plays each way (default 1000)")
    p.add_argument("--out", help="every gamer's row as a CSV here")
    p.set_defaults(func=cmd_v21_gamers)

    p = sub.add_parser("v21-react-fit",
                       help="v2.1's in-play strength updates on real games: the win-probability table the wpa update "
                            "reads (react_wp.json beside the build), and every update (snapT, ptsT, wpaC) scored on "
                            "who won, before any is priced")
    p.add_argument("--model", help="a v21-build directory: the table goes beside it, and the snap update is scored "
                                   "(it reads the build's draws)")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--since", help="first match day, YYYY-MM-DD (default: the export's first)")
    p.add_argument("--until", required=True, help="the day the priced window starts, YYYY-MM-DD: only matches before it")
    p.add_argument("--test-share", type=float, default=0.4,
                   help="the later share of the matches the updates are scored on, the rest fitting the table "
                        "(default 0.4)")
    p.add_argument("--matches", type=int, help="the last N matches before --until only (default: all)")
    p.add_argument("--paths", type=int, default=500,
                   help="games the sim plays from each kick-off the updates are scored at (default 500)")
    p.add_argument("--workers", type=int, help="matches scored at once (default: all cores but one)")
    p.add_argument("--out", help="where the table goes (default: react_wp.json in --model, else here)")
    p.set_defaults(func=cmd_v21_react_fit)

    p = sub.add_parser("ats", help="the spread against the moneyline: prod's pre-match and in-play lines and moneyline "
                                   "against the real finals, the league and each gamer (against the spread, wins over "
                                   "the moneyline), each tested for a trait on alternate days")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--since", help="first match day, YYYY-MM-DD (default: the export's first)")
    p.add_argument("--until", help="the day after the last, YYYY-MM-DD (default: the export's last)")
    p.add_argument("--min-matches", type=int, default=40, help="a gamer's matches to be shown (default 40)")
    p.add_argument("--min-half", type=int, default=15, help="and in each half of its matches for the trait test (default 15)")
    p.add_argument("--out", default=os.path.join("eAMFCalibrator", "out", "ats"),
                   help="where ats.txt and ats_gamers.csv go (default eAMFCalibrator/out/ats)")
    p.set_defaults(func=cmd_ats)

    p = sub.add_parser("playstyle", help="how each gamer plays by the score, on real games: the league's points still to "
                                         "come by the lead, each gamer's slope beyond it (front-runner or closes up) "
                                         "and pace ahead against behind, each tested for a trait on alternate days")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--prior", help="a prior table (form-prior's or the walkthrough's CSV) for the expected points "
                                   "(default: prod's pre-match spread and total in the export)")
    p.add_argument("--since", help="first match day, YYYY-MM-DD (default: the export's first)")
    p.add_argument("--until", help="the day after the last, YYYY-MM-DD (default: the export's last)")
    p.add_argument("--min-matches", type=int, default=40, help="a gamer's matches for a slope (default 40)")
    p.add_argument("--min-half", type=int, default=15,
                   help="and in each half of its matches (alternate days) for the trait test (default 15)")
    p.add_argument("--boot", type=int, default=400, help="bootstrap resamples over matches (default 400)")
    p.add_argument("--out", default=os.path.join("eAMFCalibrator", "out", "playstyle"),
                   help="where playstyle.txt and playstyle_gamers.csv go (default eAMFCalibrator/out/playstyle)")
    p.set_defaults(func=cmd_playstyle)

    p = sub.add_parser("v21-timing", help="v2.1's pricing time as the calibrator's v21 stream spends it: a sample of "
                                          "matches timed part by part, the sim profiled, the window's run projected")
    p.add_argument("--model", default="v21_model", help="a v21-build directory (default v21_model)")
    p.add_argument("--history", default=DATA["history"], help=f"the match history CSV (default {DATA['history']})")
    p.add_argument("--snapshots", default=DATA["snapshots"], help=f"the snapshot export (default {DATA['snapshots']})")
    p.add_argument("--timeouts", help=f"the calibrator's timeouts.csv (default {DATA['timeouts']} if there)")
    p.add_argument("--since", required=True, help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: --since)")
    p.add_argument("--matches", type=int, default=6, help="matches timed, a seeded sample of the window's (default 6)")
    p.add_argument("--paths", type=int, default=1000, help="games a state, as --v21-paths (default 1000)")
    p.add_argument("--workers", type=int, help="the calibrator's workers, for the projection (default: all cores but one)")
    p.add_argument("--no-profile", action="store_true", help="skip the profile of the first match's in-play run")
    p.add_argument("--compare", action="store_true",
                   help="also time the first match with v2.1's exact speedups off and on, its outputs set against each "
                        "other bit for bit")
    p.set_defaults(func=cmd_v21_timing)

    p = sub.add_parser("form-layer", help="switch the post-game form layer on or off for builds "
                                          "(form_layer.py; the follow layer stays as it is)")
    p.add_argument("builds", nargs="+", metavar="DIR", help="build directories")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--on", action="store_true")
    which.add_argument("--off", action="store_true")
    which.add_argument("--default", action="store_true", help="back to form_layer.ON")
    _layer_options(p, "the build keeps for itself, with --on")
    p.set_defaults(func=cmd_form_layer)

    p = sub.add_parser("form-trace", help="one gamer's matches as the pre-match model priced them, "
                                          "and as the post-game form layer moves them")
    p.add_argument("--gamer", action="append", required=True, help="a gamer's handle (repeat for more)")
    p.add_argument("--history", required=True, help="eAMFCalibrator history's CSV (every match's "
                                                    "players and finals)")
    p.add_argument("--model", help="a build: its pre-match model as it prices (a daily prior if one is "
                                   "attached, its shrink)")
    p.add_argument("--rolling", help="or prior-daily's directory of daily fits, as they are")
    p.add_argument("--since", required=True, help="first match day, YYYY-MM-DD")
    p.add_argument("--until", help="last match day, YYYY-MM-DD (default: --since)")
    p.add_argument("--csv", help="also write the rows to this CSV")
    _layer_options(p, "for this trace (default: the build's own, else form_layer's)")
    p.set_defaults(func=cmd_form_trace)

    p = sub.add_parser("trader", help="a local web page to test a model by hand: set up a match, click "
                                      "through it play by play, and see its prices after each play")
    p.add_argument("--model", help="the build directory; its version is read off it (default: the "
                                   "newest version's $EAMF_VN_MODEL, then ./vN_model)")
    p.add_argument("--version", help="the version to price with, e.g. v14, v21 or v21-drives (default: the build's "
                                     "own; a v2.1 build is v21-drives, the pace nudge on where the build can pace)")
    p.add_argument("--prior", help="v2.1: the prior it nudges toward, a walkthrough table or a build (default the "
                                   "form table, as the betting sim's)")
    p.add_argument("--nudge-share", type=float, default=1.0,
                   help="v2.1: how far toward the prior's expected points (default 1, as the betting sim was run)")
    p.add_argument("--drives-share", type=float, default=1.0,
                   help="v2.1: how far toward the match's expected drives (default 1)")
    p.add_argument("--no-pace", action="store_true", help="v2.1: no pace nudge (plain v21)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--paths", type=int, default=4000, help="simulated games for each price")
    p.add_argument("--margin", type=float, default=0.05, help="the book's margin on the odds shown")
    p.add_argument("--no-browser", action="store_true", help="do not open the page")
    p.set_defaults(func=cmd_trader)

    return parser


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):          # a pipe's code page lacking a character prints ?
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
