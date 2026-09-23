"""eAMFModel: a top-down, non-ML pricer for eAMF moneyline, spread and total.

  backtest PAIRS.csv     re-price a calibrator directional_pairs.csv at prod's
                         lines and score prod, candidate and each version
  trace DUMP.csv         walk a dump_play_by_play.csv message by message:
                         the tracker's verdict and the model's prices next
                         to prod's
  price                  one state, by hand
  playover SNAPS.csv     the same on PLAY_OVER snapshots off SCOUTING_FULL, on the
                         real game clock (eAMFCalibrator scouting writes the file)
  profiles SNAPS.csv     player profiles off the same snapshots: pace, 4th-down
                         aggression, clock milking
  fit-finals             refit the possession structure to nb2/AMFELO.csv
  v3-build SNAPS.csv     v3's play tables and pre-match grid off PLAY_OVER snapshots
  v3 SNAPS.csv           score v3 -- the play-by-play simulation -- against prod

The calibrator runs a version as a stream in its own right:
  python -m eAMFCalibrator report --candidate v1
"""

import argparse
import csv
import os
import sys

from . import backtest, dist, feed, fit, playover, players
from .params import VERSIONS, version
from .pricer import (AWAY, HOME, ML_HOME, SPREAD_HOME, TOTAL_OVER, GameState, Model)
from .strength import Prior


def _versions(text):
    names = [n.strip().lower() for n in text.split(",") if n.strip()]
    return {name: version(name) for name in names}


def cmd_backtest(args):
    versions = _versions(args.versions)
    names = list(versions)
    live_known = backtest.has_liveness(args.pairs)
    graded, skipped = backtest.run(args.pairs, versions, limit=args.limit,
                                   fit_prior=not args.lines_only,
                                   require_live=not args.all_rows)
    if args.moneyline_only:
        graded = [g for g in graded if g[1].market_id in (50, 51)]
    print(f"\n  {len(graded):,} graded pairs, {skipped} matches skipped (no kickoff lines)")
    if not live_known:
        print("  NOTE: this file has no liveness columns. Late spread and total quotes from")
        print("  prod include suspended ones (near 50% whatever the line), which flatters")
        print("  any rival; read the moneyline, or re-export with the current calibrator.")
    keys = {
        "market": lambda r: backtest.GROUPS[r.market_id],
        "period": lambda r: r.period if r.period and r.period <= 4 else "OT",
        "period-market": lambda r: (r.period if r.period and r.period <= 4 else "OT",
                                    backtest.GROUPS[r.market_id]),
    }
    for by in args.by.split(","):
        summary = backtest.summarise(graded, names, key=keys[by], n_boot=args.boot)
        backtest.print_summary(summary, names, f"Brier by {by}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["match_code", "message", "period", "market_id", "prod_line",
                        "prod_probability", "candidate_probability", "prod_outcome"] + names)
            for match_code, r, probs in graded:
                w.writerow([match_code, r.message, r.period, r.market_id, r.prod_line,
                            r.prod_probability, r.candidate_probability, r.prod_outcome]
                           + [round(probs[n], 5) for n in names])
        print(f"\n  per-pair prices -> {args.out}")


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


def cmd_playover(args):
    versions = _versions(args.versions)
    names = list(versions)
    book = players.Book.load(args.profiles) if args.profiles else None
    handles = (players.load_handles(args.handles or args.snapshots) if book else None)
    effects = {name: {"aggression": True, "milk": True, "pace": args.pace} for name in names}
    matches = None
    if args.half:
        codes = sorted(playover.load(args.snapshots))
        matches = codes[0::2] if args.half == "train" else codes[1::2]
    graded, skipped = playover.run(args.snapshots, versions, state_mode=args.state,
                                   scrimmage_only=args.scrimmage_only, limit=args.limit,
                                   require_live=not args.all_rows, workers=args.workers,
                                   matches=matches, book=book, handles=handles,
                                   effects=effects)
    matches = len({g[0] for g in graded})
    print(f"\n  {len(graded):,} graded PLAY_OVER quotes across {matches:,} matches")
    for reason, n in skipped.most_common():
        print(f"  skipped, {reason.replace('_', ' ')}: {n:,}")
    keys = {
        "market": lambda r: backtest.GROUPS[r.market_id],
        "period": lambda r: r.period if r.period and r.period <= 4 else "OT",
        "period-market": lambda r: (r.period if r.period and r.period <= 4 else "OT",
                                    backtest.GROUPS[r.market_id]),
        "kind": lambda r: r.kind,
    }
    for by in args.by.split(","):
        summary = backtest.summarise(graded, names, key=keys[by], n_boot=args.boot)
        backtest.print_summary(summary, names, f"Brier by {by} (PLAY_OVER snapshots)")


def _half(path, half):
    codes = sorted(playover.load(path))
    return codes[0::2] if half == "train" else codes[1::2] if half == "test" else codes


def _v3_module():
    """v3 needs numpy; nothing else in the package does, so it loads here."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit("v3 needs numpy:  py -m pip install numpy   (or python -m pip ...)")
    from . import v3
    return v3


def cmd_v3_build(args):
    v3 = _v3_module()
    data = playover.load(args.snapshots)
    keep = set(_half(args.snapshots, args.half))
    v3.build({c: rows for c, rows in data.items() if c in keep}, args.out)
    print(f"  wrote {args.out}/v3tables.npz and {args.out}/v3grid.npz")


def cmd_v3(args):
    v3 = _v3_module()
    variants = [v3.Variant("v3", react=False)]
    if args.react:
        variants.append(v3.Variant("v3_react", kappa=args.kappa))
    book = players.Book.load(args.profiles) if args.profiles else None
    handles = players.load_handles(args.handles or args.snapshots) if book else None
    if book:
        variants.append(v3.Variant("v3_profiles", react=False, profiles=True, pace=True))
    matches = _half(args.snapshots, args.half)
    if args.limit:
        matches = matches[:args.limit]
    graded, skipped = v3.run(args.snapshots, os.path.join(args.model, "v3tables.npz"),
                             os.path.join(args.model, "v3grid.npz"), variants, matches=matches,
                             n_paths=args.paths, workers=args.workers, book=book, handles=handles)
    names = [v.name for v in variants]
    print(f"\n  {len(graded):,} graded PLAY_OVER quotes across {len({g[0] for g in graded}):,} matches")
    for reason, n in skipped.most_common():
        print(f"  skipped, {reason.replace('_', ' ')}: {n:,}")
    keys = {
        "market": lambda r: backtest.GROUPS[r.market_id],
        "period-market": lambda r: (r.period if r.period and r.period <= 4 else "OT",
                                    backtest.GROUPS[r.market_id]),
        "kind": lambda r: r.kind,
    }
    for by in args.by.split(","):
        summary = backtest.summarise(graded, names, key=keys[by], n_boot=args.boot)
        backtest.print_summary(summary, names, f"Brier by {by} (v3, PLAY_OVER snapshots)")


def _num(value):
    return int(float(value)) if value not in ("", None) else None


def cmd_trace(args):
    rows = list(csv.DictReader(open(args.dump, newline="")))
    if args.match:
        rows = [r for r in rows if r["match_code"] == args.match]
    match_codes = sorted({r["match_code"] for r in rows})
    if len(match_codes) != 1:
        sys.exit(f"trace needs one match; the file has {len(match_codes)}: pass --match")
    plays, scores = [], []
    for r in rows:
        m = _num(r["event_message_count"])
        if r["cleaning"] == "score":
            scores.append(feed.Score(m, _num(r["period_number"]), _num(r["score_p1"]),
                                     _num(r["score_p2"])))
        else:
            plays.append(feed.Play(m, _num(r["period_number"]), feed.side(r["offensive_team"]),
                                   _num(r["down_number"]), _num(r["distance"]),
                                   _num(r["field_position"])))
    first = min(rows, key=lambda r: _num(r["event_message_count"]))
    model = Model(version(args.version))
    if args.spread is not None and args.total is not None:
        prior = Prior.from_lines(args.spread, args.total)
    else:
        prior = model.fit_prior(float(first["sp_home_line_prod"]), float(first["tot_over_line_prod"]),
                                ml_home=float(first["ml_home_prod"]) / 100,
                                spread_home=float(first["sp_home_prod"]) / 100,
                                over=float(first["tot_over_prod"]) / 100)
    print(f"\n  {match_codes[0]}  prior: home {prior.home_points:.1f}  away {prior.away_points:.1f}"
          f"   ({args.version})")
    print(f"  {'msg':>4} {'P':>1} {'team':<4} {'state':<10} {'feed says':<13} {'model':<5} "
          f"{'why':<14} {'score':<6}  {'ML home':>15}  {'spread line':>15}  {'total line':>15}")
    by_message = {_num(r["event_message_count"]): r for r in rows}
    for tick in feed.replay(plays, scores):
        r = by_message[tick.message]
        s = tick.state
        situation = (f"{r['down_number']}&{r['distance']}@{r['field_position']}"
                     if r["down_number"] else "")
        line = (f"  {tick.message:>4} {r['period_number']:>1} {r['offensive_team'][:4]:<4} "
                f"{situation:<10} {r['cleaning'][:13]:<13} {tick.verdict[:4]:<5} "
                f"{(tick.reason or ''):<14} {s.home_score}-{s.away_score:<4}")
        if tick.verdict == feed.LIVE:
            b = model.book(prior, s)
            line += (f"  {100 * b.p_home:5.1f} v {r['ml_home_prod'] or '-':>6}"
                     f"  {b.fair_line(SPREAD_HOME):+5.1f} v {r['sp_home_line_prod'] or '-':>5}"
                     f"  {b.fair_line(TOTAL_OVER):5.1f} v {r['tot_over_line_prod'] or '-':>5}")
        print(line)
    print("\n  model v prod: moneyline home %, the model's own fair spread (home margin) and"
          "\n  total lines against prod's. Suspended rows are not priced.")


def cmd_price(args):
    model = Model(version(args.version))
    prior = Prior.from_lines(args.spread, args.total)
    home, away = (int(x) for x in args.score.split("-"))
    offense = {"home": HOME, "away": AWAY}.get((args.offense or "").lower())
    state = GameState(args.period, args.elapsed, home, away, offense, args.down, args.field,
                      args.distance, opening_receiver={"home": HOME, "away": AWAY}.get(
                          (args.opening or "").lower()))
    b = model.book(prior, state)
    print(f"\n  P(home wins) {b.p_home:.4f}   share of game left {b.fraction_left:.3f}")
    print(f"  final margin mean {b.margin.mean():+.2f}   total mean {dist.mean(b.total):.2f}")
    print(f"  fair spread (home) {b.fair_line(SPREAD_HOME):+.1f}   fair total {b.fair_line(TOTAL_OVER):.1f}")
    for line in (args.spread - 3, args.spread, args.spread + 3):
        print(f"  home margin > {line:+.1f}: {b.prob(SPREAD_HOME, line):.4f}")
    for line in (args.total - 7, args.total, args.total + 7):
        print(f"  total > {line:.1f}: {b.prob(TOTAL_OVER, line):.4f}")


def cmd_fit_finals(args):
    theta, nll, n = fit.fit_finals(args.finals, args.iterations)
    print(f"\n  {n:,} finals, negative log-likelihood {nll:,.1f}")
    for name, value in theta.items():
        print(f"  {name:<16}{value:.4f}")
    mom = fit.moments(theta)
    print(f"  implied total {mom['total'][0]:.2f} sd {mom['total'][1]:.2f}   "
          f"margin sd {mom['margin'][1]:.2f}   score correlation {mom['corr']:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="eAMFModel", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("backtest", help="score versions against prod on a pairs CSV")
    p.add_argument("pairs")
    p.add_argument("--versions", default=",".join(sorted(VERSIONS)))
    p.add_argument("--by", default="market,period-market",
                   help="comma list of: market, period, period-market")
    p.add_argument("--moneyline-only", action="store_true")
    p.add_argument("--all-rows", action="store_true",
                   help="keep pairs where a stream was suspended")
    p.add_argument("--lines-only", action="store_true",
                   help="prior from the kickoff lines alone, not fitted to the prices")
    p.add_argument("--limit", type=int, help="first N matches only")
    p.add_argument("--boot", type=int, default=300)
    p.add_argument("--out", help="write per-pair prices here")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("playover", help="score versions against prod on PLAY_OVER snapshots "
                                        "(eAMFCalibrator scouting's export)")
    p.add_argument("snapshots", help="scouting_playover.csv")
    p.add_argument("--versions", default=",".join(sorted(VERSIONS)))
    p.add_argument("--by", default="market,period-market,kind",
                   help="comma list of: market, period, period-market, kind")
    p.add_argument("--state", choices=[playover.NEXT, playover.OVER], default=playover.OVER,
                   help="scrimmage state: the PLAY_OVER row's (default) or the next PLAY_STARTED's")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="processes to price with (default: all cores but one)")
    p.add_argument("--scrimmage-only", action="store_true",
                   help="leave out kickoffs, conversions, punts, kicks and scores")
    p.add_argument("--all-rows", action="store_true", help="keep prod quotes that were not live")
    p.add_argument("--limit", type=int, help="first N matches only")
    p.add_argument("--boot", type=int, default=300)
    p.add_argument("--profiles", help="player profiles (eAMFModel profiles) to apply")
    p.add_argument("--handles", help="CSV of MATCH_CODE, PLAYER_1_HANDLE, PLAYER_2_HANDLE "
                                     "(default: the snapshot file's own handle columns)")
    p.add_argument("--pace", choices=[playover.PACE_OFF, playover.PACE_NEWS, playover.PACE_FULL],
                   default=playover.PACE_OFF, help="in-game pace effect (default off)")
    p.add_argument("--half", choices=["train", "test"],
                   help="every other match: build profiles on train, score on test")
    p.set_defaults(func=cmd_playover)

    p = sub.add_parser("profiles", help="build player profiles (pace, 4th-down aggression, "
                                        "clock milking) from PLAY_OVER snapshots")
    p.add_argument("snapshots", help="scouting_playover.csv")
    p.add_argument("--handles", help="CSV of MATCH_CODE, PLAYER_1_HANDLE, PLAYER_2_HANDLE "
                                     "(default: the snapshot file's own handle columns)")
    p.add_argument("--out", default="player_profiles.json")
    p.add_argument("--half", choices=["train", "test"])
    p.add_argument("--show", type=int, default=20)
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("trace", help="message-by-message walk of one match")
    p.add_argument("dump")
    p.add_argument("--match")
    p.add_argument("--version", default="v1")
    p.add_argument("--spread", type=float, help="prior spread; default prod's kickoff")
    p.add_argument("--total", type=float, help="prior total; default prod's kickoff")
    p.set_defaults(func=cmd_trace)

    p = sub.add_parser("price", help="price one state")
    p.add_argument("--version", default="v1")
    p.add_argument("--spread", type=float, required=True, help="pre-match expected home margin")
    p.add_argument("--total", type=float, required=True)
    p.add_argument("--period", type=int, default=1)
    p.add_argument("--elapsed", type=float, default=0, help="messages into the period")
    p.add_argument("--score", default="0-0", help="home-away")
    p.add_argument("--offense", choices=["home", "away"])
    p.add_argument("--opening", choices=["home", "away"], help="who received first")
    p.add_argument("--down", type=int)
    p.add_argument("--distance", type=int)
    p.add_argument("--field", type=int, help="yards from the offense's own goal")
    p.set_defaults(func=cmd_price)

    p = sub.add_parser("fit-finals", help="refit the possession structure")
    p.add_argument("--finals", default=fit.DEFAULT_FINALS)
    p.add_argument("--iterations", type=int, default=400)
    p.set_defaults(func=cmd_fit_finals)

    p = sub.add_parser("v3-build", help="v3: play tables and pre-match grid off PLAY_OVER "
                                        "snapshots")
    p.add_argument("snapshots", help="scouting_playover.csv")
    p.add_argument("--half", choices=["train", "test", "all"], default="train")
    p.add_argument("--out", default="v3_model")
    p.set_defaults(func=cmd_v3_build)

    p = sub.add_parser("v3", help="score v3, the play-by-play simulation, against prod on "
                                  "PLAY_OVER snapshots")
    p.add_argument("snapshots", help="scouting_playover.csv")
    p.add_argument("--model", default="v3_model", help="v3-build's output directory")
    p.add_argument("--half", choices=["train", "test", "all"], default="test")
    p.add_argument("--paths", type=int, default=2000, help="simulated games per snapshot")
    p.add_argument("--react", action="store_true",
                   help="also a variant whose efficiencies move with first-down success")
    p.add_argument("--kappa", type=float, default=200.0, help="shrinkage for --react")
    p.add_argument("--profiles", help="player profiles (eAMFModel profiles): adds a variant "
                                      "with each player's 4th-down aggression and pace")
    p.add_argument("--handles", help="CSV of MATCH_CODE, PLAYER_1_HANDLE, PLAYER_2_HANDLE")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, help="first N matches only")
    p.add_argument("--boot", type=int, default=300)
    p.add_argument("--by", default="market,period-market,kind")
    p.set_defaults(func=cmd_v3)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
