"""eAMFModel: eAMF moneyline, spread and total priced by simulating the rest of each game play
by play, off PLAY_OVER snapshots on the real game clock.

  profiles SNAPS.csv     player profiles off the snapshots: pace, 4th-down
                         aggression, clock milking
  v8-build / v8          v8's play tables, pre-match grid (NB2 or glmer, --history) and
                         player profiles off PLAY_OVER snapshots, and v8 scored against
                         prod: the play-by-play simulation with the end-of-half clock,
                         timeouts and kneels
  v9-build / v9          the same for v9 (v8 plus recent weeks weighing more and the
                         late timeout plays' real time)
  v10-build / v10        the same for v10 (v9 with the pre-match prior's pace counted once
                         and its spread fitted out of sample)
  v11-build / v11        the same for v11 (v10 with close endings, timeouts, overtime and
                         return touchdowns played as real games play them)
  v12-build / v12        the same for v12 (v11 with level sides' late drives run down to
                         the kick as real ones are)
  v13-build / v13        the same for v13 (v12 with each side's big-play rate: its gamer's,
                         its team's and what the other side allows)
  prior-daily            refit the pre-match model every day of a window, as live, and point
                         builds at the daily fits
  form-layer DIR         switch the post-game form layer on or off for builds (form_layer.py)
  form-trace             one gamer's matches as the pre-match model priced them and as the
                         post-game form layer moves them
  trader                 a local page to click through a game and see a version's prices
  kickoff-dist           a v12 kick-off simulated at given expected points: its final totals
                         against the real ones
  remaining SNAPS.csv    a version's points still to come against what really came

The calibrator runs a version as a stream in its own right:
  python -m eAMFCalibrator report --candidate v9 --v9-model v9_model
"""

import argparse
import os

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
                 browser=not args.no_browser, version=args.version)


def cmd_kickoff_dist(args):
    from . import kickoff_check
    try:
        means = tuple(float(x) for x in args.means.split(","))
        assert len(means) == 2
    except (ValueError, AssertionError):
        raise SystemExit("--means takes two expected points, home,away (e.g. 15.5,15.5)")
    kickoff_check.run(args.model, means, args.history, since=args.since or kickoff_check.default_since(),
                      until=args.until, n_paths=args.paths, seed=args.seed)


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
    rolling = rolling_prior.fit(history, args.prior, args.out, since, until, workers=args.workers)
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


def main(argv=None):
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

    for name, what in (("v8", "v8: the play-by-play simulation with the end-of-half clock, timeouts "
                               "and kneels (see README)"),
                       ("v9", "v9: v8 plus recent weeks weighing more and the late timeout plays' "
                              "real time (see README)"),
                       ("v10", "v10: v9 with the pre-match prior's pace counted once and its spread "
                               "fitted out of sample (see README)"),
                       ("v11", "v11: v10 with close endings, timeouts, overtime and return touchdowns "
                               "played as real games play them (see README)"),
                       ("v12", "v12: v11 with level sides' late drives run down to the kick as real "
                               "ones are (see README)"),
                       ("v13", "v13: v12 with each side's big-play rate -- its gamer's, its team's and "
                               "what the other side allows (see README)")):
        p = sub.add_parser(f"{name}-build", help=what)
        p.add_argument("snapshots", help="scouting_playover.csv")
        p.add_argument("--half", choices=["train", "test", "all"], default="train")
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
        p.add_argument("--prior", choices=["nb2", "glmer"], default="nb2",
                       help="the pre-match model --history fits: nb2 (nb2/, the default) or "
                            "glmer (glmer/ in R, needs R and lme4; see glmer/README.md)")
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
    p.add_argument("--version", choices=["v8", "v9", "v10", "v11", "v12", "v13"], default="v9")
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
    p.set_defaults(func=cmd_prior_daily)

    p = sub.add_parser("kickoff-dist", help="a v12 kick-off simulated at given expected points: its "
                                            "final totals against the real ones over a window")
    p.add_argument("--model", required=True, help="a v12 build directory, e.g. v12_1001")
    p.add_argument("--means", default="15.5,15.5", help="home,away expected points (default 15.5,15.5)")
    p.add_argument("--history", required=True, help="the match history CSV (eAMFCalibrator history)")
    p.add_argument("--since", help="first real match day, YYYY-MM-DD (default: 30 days ago)")
    p.add_argument("--until", help="day after the last real match day, YYYY-MM-DD")
    p.add_argument("--paths", type=int, default=20000, help="simulated games (default 20000)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_kickoff_dist)

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
    p.add_argument("--version", help="the version to price with, e.g. v9 (default: the build's own)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--paths", type=int, default=4000, help="simulated games for each price")
    p.add_argument("--margin", type=float, default=0.05, help="the book's margin on the odds shown")
    p.add_argument("--no-browser", action="store_true", help="do not open the page")
    p.set_defaults(func=cmd_trader)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
