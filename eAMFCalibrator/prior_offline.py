"""The pre-match priors graded against each other on the results alone -- no prod, no bets, no
Snowflake: prior-backtest's arms and layers, read off a match history CSV.

    python -m eAMFCalibrator prior-offline --history eAMFCalibrator/out/match_history.csv \\
        --since "2026-08-15 00:00:00" --arm "nb2=eAMFCalibrator/out/nb2_history.csv" \\
        --arm "glmer=glmer/out/cadence/cadence_matches.csv@trim:hl60_x0.5:daily" \\
        --blend "avg=nb2+infer,glmer+both"

The arms are prior-backtest's (NAME=SOURCE[#MODE]: nb2-history's nb2_history.csv, cadence.R's
PATH@PREFIX, a rolling:DIR of prior-daily's fits, ...), each read, layered and scored the same way:
as it is, +layer, +infer, +both and, with the gamers' stats (--features, player-stats'
player_match_stats.csv), +feat and +both+feat, every layer fitted walk-forward on the arm's own
errors. A blend (--blend NAME=VARIANT,VARIANT,...) averages its variants' moneylines and expected
totals, match by match.

Every variant is graded against the first arm as it is (the baseline), match by match, on the
matches every variant priced after the layers' burn-in: the moneyline's log loss and Brier (draws
out), the expected total's RMSE and bias; for every match, the rematches, the first meetings, the
first meetings after either gamer had played someone else that session, each period
(backtest_suite's: either side of the play-detail gap) and each month. With no prod there is no
over at a line and no bets: prior-backtest and prior-replay are the tests against prod.

Writes prior_offline.txt and prior_offline.csv (every match, every variant's moneyline and total).
"""

import csv
import datetime as dt
import math
import os

from . import prior_backtest as pb, score_model
from .backtest_suite import DEFAULTS as SUITE
from .bet_rematches import day_meetings
from .bet_totals import _start

PERIODS = SUITE["periods"]


def prepare(history_path, since, until=None, features=None):
    """prior_backtest.prepare's context off a match history CSV, with no prod."""
    from eAMFModel import nb2_prior
    history = nb2_prior.load_history(history_path)
    meets = day_meetings(history)
    start, end = _start(since), _start(until) if until else None
    window = {c: m for c, m in meets.items() if m["start"] >= start and (end is None or m["start"] < end)
              and m.get("home_score") is not None and m.get("away_score") is not None}
    codes = sorted(window)
    if not codes:
        raise SystemExit(f"no settled matches in {history_path} from {since} to {until or 'its end'}")
    last = max(m["start"] for m in window.values())
    print(f"  {len(codes):,} settled matches from {since} to {last:%Y-%m-%d %H:%M} ({history_path})", flush=True)
    prod = {c: dict(p1=None, line=None, over=None) for c in codes}
    return pb.context(history, window, codes, prod, features, start, tl={}, early={}, first_plays={})


def parse_blend(spec):
    """NAME=VARIANT,VARIANT,... -> (name, [variants])."""
    name, sep, parts = spec.partition("=")
    variants = [p.strip() for p in parts.split(",") if p.strip()]
    if not sep or not name.strip() or len(variants) < 2:
        raise SystemExit(f"--blend {spec!r}: give it as NAME=VARIANT,VARIANT[,...] (e.g. avg=nb2+infer,glmer+both)")
    return name.strip(), variants


def blend(prices, variants):
    """{match: (p home, expected total, None)}: the variants' moneylines and totals averaged."""
    missing = [v for v in variants if v not in prices]
    if missing:
        raise SystemExit(f"--blend: no variant {', '.join(missing)}; the variants are {', '.join(prices)}")
    codes = set.intersection(*(set(prices[v]) for v in variants))
    return {c: (sum(prices[v][c][0] for v in variants) / len(variants),
                sum(prices[v][c][1] for v in variants) / len(variants), None) for c in codes}


def table(label, codes, prices, results, starts, base):
    """Lines: one subset's grades, every variant against the baseline, best moneyline first."""
    g = pb.grade(codes, prices, results)
    first = min(starts).date()
    rows = []
    for v, m in g.items():
        dl, tl = pb._paired(g[base]["ll"], m["ll"])
        bt, n = pb._fortnights(g[base]["ll"], m["ll"], starts, first)
        de, te = pb._paired(g[base]["te"], m["te"])
        rmse = pb._mean(m["te"])
        rows.append((pb._mean(m["ll"]), v, m, dl, tl, bt, n, de, te, rmse))
    rows.sort(key=lambda x: (x[1] != base, x[0]))
    L = [f"\n  {label}: {len(codes):,} matches",
         f"  {'variant':28s} {'log loss':>9s} {'vs base':>8s} {'t':>5s} {'fn':>5s} {'Brier':>7s} "
         f"{'tot RMSE':>8s} {'bias':>6s} {'vs base':>8s} {'t':>5s}"]
    for ll, v, m, dl, tl, bt, n, de, te, rmse in rows:
        own = v == base
        L.append(f"  {v[:28]:28s} {pb._fmt(ll, '.4f', 9)} {pb._fmt(None if own else dl, '+.4f', 8)} "
                 f"{pb._fmt(None if own else tl, '+.1f', 5)} {('-' if own or not n else f'{bt}/{n}'):>5s} "
                 f"{pb._fmt(pb._mean(m['br']), '.4f', 7)} {pb._fmt(math.sqrt(rmse) if rmse is not None else None, '.2f', 8)} "
                 f"{pb._fmt(pb._mean(m['tb']), '+.2f', 6)} {pb._fmt(None if own else de, '+.2f', 8)} "
                 f"{pb._fmt(None if own else te, '+.1f', 5)}")
    return L


def _in(t, lo, hi):
    d = t.date()
    return (lo is None or d >= dt.date.fromisoformat(lo)) and (hi is None or d < dt.date.fromisoformat(hi))


def report(prices, results, meets, burn_from, base, others, models, layers, periods=PERIODS):
    """Lines: every variant against the baseline on the matches all of them priced after the burn-in."""
    variants = list(prices)
    codes = sorted((c for c in results if all(c in prices[v] for v in variants) and meets[c]["start"] >= burn_from),
                   key=lambda c: meets[c]["start"])
    if not codes:
        raise SystemExit("no match every variant priced after the burn-in")
    L = [f"  {len(codes):,} matches priced by every variant, from {burn_from:%Y-%m-%d} (after the layers' "
         f"{pb.BURN_IN_DAYS}-day burn-in); the baseline: {base}",
         "  log loss / Brier: the moneyline (draws out); tot RMSE / bias: the expected total against the final "
         "(bias = final less it); vs base: the variant less the baseline, match by match (negative is better; "
         "the total's in squared points); t on it; fn: fortnights it beat the baseline on log loss"]
    subsets = [("Every match", lambda c: True), ("Rematches", lambda c: meets[c]["k"] > 1),
               ("First meetings", lambda c: meets[c]["k"] == 1),
               ("First meetings after either gamer played someone else that session",
                lambda c: meets[c]["k"] == 1 and c in others)]
    subsets += [(f"Period: {label}", lambda c, lo=lo, hi=hi: _in(meets[c]["start"], lo, hi)) for label, lo, hi in periods]
    months = sorted({meets[c]["start"].strftime("%Y-%m") for c in codes})
    subsets += [(f"Month: {m}", lambda c, m=m: meets[c]["start"].strftime("%Y-%m") == m) for m in months]
    for label, keep in subsets:
        sub = [c for c in codes if keep(c)]
        if len(sub) >= 30:
            L += table(label, sub, prices, results, [meets[c]["start"] for c in sub], base)
    L.append("\n  How each arm's moneyline is read:")
    L += [f"  {name}: {text}" for name, text in models.items()]
    L += pb.layer_lines(layers)
    return L, codes


def run(out_dir, arms, history_path, since, until=None, features=None, blends=(), draws=score_model.DRAWS,
        periods=PERIODS):
    """Grade every arm, as it is and with each layer, and every blend, against the first arm.
    Returns the report's path."""
    if not arms:
        raise SystemExit("give at least one --arm")
    ctx = prepare(history_path, since, until, features)
    prices, layers, models, others = {}, {}, {}, set()
    for name, spec in arms:
        mode, own, rows, layer = pb.fit_arm(name, spec, ctx)
        layers[name] = layer
        others |= {r["match"] for r in rows if r.get("n_others")}
        got, fits = pb.readings(name, rows, ctx["prod"], mode or "normal", own, draws, pb.active_layers(ctx))
        models[name] = pb.describe_model(mode, own, fits)
        for code, v in got.items():
            for variant, val in v.items():
                prices.setdefault(variant, {})[code] = val
    for spec in blends:
        name, variants = parse_blend(spec)
        if name in prices:
            raise SystemExit(f"--blend {name}: that name is a variant already")
        prices[name] = blend(prices, variants)
        models[name] = "the average of " + ", ".join(variants)
    window = ctx["window"]
    results = {c: (m["home_score"] - m["away_score"], m["home_score"] + m["away_score"], None) for c, m in window.items()}
    L, codes = report(prices, results, window, ctx["burn_from"], arms[0][0], others, models, layers, periods)
    print("\n".join(L))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prior_offline.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "start", "meeting", "home", "away", "margin", "total"]
                   + [f"{v}_{x}" for v in prices for x in ("p_home", "total")])
        for c in codes:
            m = window[c]
            w.writerow([c, m["start"], m["meeting"], m["home"], m["away"], *results[c][:2]]
                       + [round(prices[v][c][i], 4) for v in prices for i in range(2)])
    path = os.path.join(out_dir, "prior_offline.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path
