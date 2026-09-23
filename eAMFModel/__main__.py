"""eAMFModel: a top-down, non-ML pricer for eAMF moneyline, spread and total.

  backtest PAIRS.csv     re-price a calibrator directional_pairs.csv at prod's
                         lines and score prod, candidate and each version
  trace DUMP.csv         walk a dump_play_by_play.csv message by message:
                         the tracker's verdict and the model's prices next
                         to prod's
  price                  one state, by hand
  fit-finals             refit the possession structure to nb2/AMFELO.csv

The calibrator runs a version as a stream in its own right:
  python -m eAMFCalibrator report --candidate v1
"""

import argparse
import csv
import os
import sys

from . import backtest, dist, feed, fit
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
