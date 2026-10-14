"""Pre-match priors against prod, each with and without the rematch layer, graded on the results.

    python -m eAMFCalibrator prior-backtest --since 2026-09-01 --until 2026-10-14 \\
        --arm "glmer-trim=glmer/out/cadence/cadence_matches.csv@trim:hl60:game" \\
        --arm "nb2=rolling:nb2_daily"

An arm is NAME=SOURCE, its expected points per side for every match as of the match's kick-off:
  PATH@PREFIX   glmer/cadence.R's cadence_matches.csv: the columns PREFIX|1 and PREFIX|2
  PATH          a CSV with MATCH_CODE (or MatchId) and the two sides' expected points:
                Pred_P1_Points / Pred_P2_Points (NB2_schedule_predict.py, glmer/predict.R),
                raw_home / raw_away (eAMFModel prior-vs-sim), or exp_p1 / exp_p2
  rolling:DIR   eAMFModel prior-daily's directory of daily fits (NB2 or glmer): each match priced by
                its own day's fit, as live
prod is always there: its last pre-match moneyline and total (line and the over's price), from
GAMEPLAI_STREAM.

Each arm, prod among them, is graded as it is and with the rematch layer (rematch_layer.py) on top,
fitted walk-forward on that arm's own errors (each UTC day on the days before it). An arm's
moneyline is the normal about its expected margin (rematch_layer.MARGIN_SD), its total's over the
normal about its expected total (TOTAL_SD) at prod's line; prod's own are its prices, and prod with
the layer moves the margin its moneyline implies and the mean its total implies.

Graded on the matches every arm priced, after the layer's burn-in, by:
  moneyline   log loss and Brier (draws left out)
  total       RMSE and bias of the expected total (prod: its line), and the log loss of the over at
              prod's line (pushes left out)
each against prod match by match (the difference, its t, and the fortnights it was better), for every
match, the rematches, the 2nd meetings and the 3rd+.
"""

import csv
import datetime as dt
import math
import os
from collections import defaultdict
from statistics import NormalDist

from . import bets, config, rematch_layer, snowflake_io
from .bet_losers import prematch_p1
from .bet_rematches import day_meetings
from .bet_totals import _start

TOTAL_SD = 12.0
BURN_IN_DAYS = 7
PROD = "prod"
POINT_COLUMNS = (("Pred_P1_Points", "Pred_P2_Points"), ("raw_home", "raw_away"), ("exp_p1", "exp_p2"))
_N = NormalDist()


def _f(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def prematch_total(tl, match):
    """(prod's last pre-match total line, its over's price), or (None, None)."""
    entry = tl.get((match, 54))
    if not entry or not entry[1]:
        return None, None
    pre = [q for q in entry[1] if q[0] is None and q[2] is not None]
    q = pre[-1] if pre else next((q for q in entry[1] if q[2] is not None), None)
    return (q[2], q[1]) if q else (None, None)


def read_points(spec, rows, history):
    """{match: (home, away) expected points} for one arm's source."""
    if spec.startswith("rolling:"):
        from eAMFModel import rolling_prior
        prior = rolling_prior.Rolling(spec[len("rolling:"):])
        got = prior.means(rows, results=history)
        return {k: (float(v[0]), float(v[1])) for k, v in got.items()}
    path, _, prefix = spec.partition("@")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        key = "MATCH_CODE" if "MATCH_CODE" in cols else "MatchId"
        if prefix:
            pair = (f"{prefix}|1", f"{prefix}|2")
        else:
            pair = next((p for p in POINT_COLUMNS if p[0] in cols and p[1] in cols), None)
        if pair is None or pair[0] not in cols or pair[1] not in cols or key not in cols:
            raise SystemExit(f"{spec}: no {' / '.join(p[0] for p in POINT_COLUMNS)} (or PREFIX|1) columns "
                             f"and match code there; its columns are {', '.join(cols)}")
        out = {}
        for r in reader:
            a, b = _f(r.get(pair[0])), _f(r.get(pair[1]))
            if a is not None and b is not None:
                out[r[key]] = (a, b)
        return out


def items_for(name, points, prod, meets):
    """rematch_layer items for one arm: its expected margin and total (prod: the margin its moneyline
    implies, its line), with the result."""
    out = []
    for code, m in meets.items():
        if name == PROD:
            p = prod.get(code)
            if not p or p["p1"] is None or p["line"] is None:
                continue
            margin, total = rematch_layer.implied_margin(p["p1"]), p["line"]
        else:
            if code not in points:
                continue
            a, b = points[code]
            margin, total = a - b, a + b
        s1, s2 = m.get("home_score"), m.get("away_score")
        it = rematch_layer.item(code, m["start"], m["day_key"], m["home"], m["away"], m["team"][m["home"]],
                                m["team"][m["away"]], margin, total,
                                None if s1 is None or s2 is None else s1 - s2,
                                None if s1 is None or s2 is None else s1 + s2)
        out.append(it)
    return out


def over_chance(total, line, sd=TOTAL_SD):
    return 1 - _N.cdf((line - total) / sd)


def readings(name, rows, prod):
    """{match: {variant: (p home, expected total, p over prod's line)}} for an arm and its +layer."""
    out = {}
    for r in rows:
        p = prod.get(r["match"]) or {}
        line, over = p.get("line"), p.get("over")
        v = {}
        for tag, mk, tk in (("", "margin", "total"), ("+layer", "margin_adj", "total_adj")):
            m, t = r[mk], r[tk]
            if name == PROD and tag == "":
                v[name] = (p.get("p1"), line, over)
                continue
            if name == PROD:                       # prod moved: its total's implied mean moves
                base = line + TOTAL_SD * _N.inv_cdf(min(max(over, 1e-6), 1 - 1e-6)) if over is not None else line
                t = base + r["move_total"]
            v[name + tag] = (rematch_layer.win_chance(m), t, None if line is None else over_chance(t, line))
        out[r["match"]] = v
    return out


def _ll(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def grade(codes, prices, results):
    """{arm: {metric: [per-match value or None]}} over the codes (one list order for every arm)."""
    out = {}
    for arm, by in prices.items():
        g = defaultdict(list)
        for c in codes:
            p, t, po = by[c]
            margin, total, line = results[c]
            g["ll"].append(None if margin == 0 or p is None else _ll(p, 1.0 if margin > 0 else 0.0))
            g["br"].append(None if margin == 0 or p is None else (p - (1.0 if margin > 0 else 0.0)) ** 2)
            g["te"].append(None if t is None else (total - t) ** 2)
            g["tb"].append(None if t is None else total - t)
            g["ou"].append(None if po is None or line is None or total == line else _ll(po, 1.0 if total > line else 0.0))
        out[arm] = g
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _paired(a, b):
    d = [y - x for x, y in zip(a, b) if x is not None and y is not None]
    if len(d) < 3:
        return None, None
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m, (m / (sd / math.sqrt(len(d))) if sd else None)


def _fortnights(a, b, starts, first):
    """(fortnights b beat a in mean, fortnights)."""
    by = defaultdict(lambda: [[], []])
    for x, y, s in zip(a, b, starts):
        if x is not None and y is not None:
            k = (s.date() - first).days // 14
            by[k][0].append(x)
            by[k][1].append(y)
    better = sum(1 for xs, ys in by.values() if sum(ys) / len(ys) < sum(xs) / len(xs))
    return better, len(by)


def _fmt(x, fmt, w):
    return f"{'-':>{w}s}" if x is None else f"{x:{fmt}}".rjust(w)


def table(label, codes, prices, results, starts):
    """Lines: one subset's grades, each arm against prod."""
    g = grade(codes, prices, results)
    first = min(starts).date() if starts else None
    L = [f"\n  {label}: {len(codes):,} matches",
         f"  {'arm':22s} {'log loss':>9s} {'vs prod':>8s} {'t':>5s} {'fn':>5s} {'Brier':>7s} {'tot RMSE':>8s} "
         f"{'bias':>6s} {'vs prod':>8s} {'t':>5s} {'o/u ll':>7s} {'vs prod':>8s} {'t':>5s}"]
    base = g[PROD]
    for arm, m in g.items():
        dl, tl = _paired(base["ll"], m["ll"])
        bt, n = _fortnights(base["ll"], m["ll"], starts, first) if arm != PROD else (None, None)
        de, te = _paired(base["te"], m["te"])
        do, to = _paired(base["ou"], m["ou"])
        rmse = _mean(m["te"])
        L.append(f"  {arm[:22]:22s} {_fmt(_mean(m['ll']), '.4f', 9)} "
                 f"{_fmt(dl if arm != PROD else None, '+.4f', 8)} {_fmt(tl if arm != PROD else None, '+.1f', 5)} "
                 f"{(f'{bt}/{n}' if n else '-'):>5s} {_fmt(_mean(m['br']), '.4f', 7)} "
                 f"{_fmt(math.sqrt(rmse) if rmse is not None else None, '.2f', 8)} {_fmt(_mean(m['tb']), '+.2f', 6)} "
                 f"{_fmt(de if arm != PROD else None, '+.2f', 8)} {_fmt(te if arm != PROD else None, '+.1f', 5)} "
                 f"{_fmt(_mean(m['ou']), '.4f', 7)} {_fmt(do if arm != PROD else None, '+.4f', 8)} "
                 f"{_fmt(to if arm != PROD else None, '+.1f', 5)}")
    return L


def report(prices, results, meets, layers, burn_from):
    """Lines of text: every arm against prod on the matches all of them priced, after the burn-in."""
    arms = list(prices)
    codes = sorted((c for c in results if all(c in prices[a] for a in arms)
                    and meets[c]["start"] >= burn_from), key=lambda c: meets[c]["start"])
    L = [f"  {len(codes):,} matches priced by every arm ({', '.join(arms)}), from {burn_from:%Y-%m-%d} "
         f"(after the layer's {BURN_IN_DAYS}-day burn-in)",
         "  log loss / Brier: the moneyline (draws out); tot RMSE / bias: the expected total (prod: its line) "
         "against the final, bias = final less it; o/u ll: the over at prod's line (pushes out)",
         "  vs prod: the arm less prod, match by match (negative is better; the total's in squared points); "
         "t on it; fn: fortnights the arm beat prod on log loss"]
    subsets = (("Every match", lambda c: True), ("Rematches", lambda c: meets[c]["k"] > 1),
               ("2nd meetings", lambda c: meets[c]["k"] == 2), ("3rd+ meetings", lambda c: meets[c]["k"] > 2),
               ("First meetings", lambda c: meets[c]["k"] == 1))
    for label, keep in subsets:
        sub = [c for c in codes if keep(c)]
        if sub:
            L += table(label, sub, {a: prices[a] for a in arms}, results, [meets[c]["start"] for c in sub])
    L.append("\n  The rematch layers' last fits (each arm's own errors):")
    for name, layer in layers.items():
        L.append(f"  {name}:")
        L += ["  " + x for x in layer.describe()]
    return L


def run(cur, out_dir, arms, since, until=None):
    """Grade prod and every arm, each with and without the rematch layer, over [since, until]."""
    history = snowflake_io.fetch_history(cur)
    meets = day_meetings(history)
    start, end = _start(since), _start(until) if until else None
    window = {c: m for c, m in meets.items() if m["start"] >= start and (end is None or m["start"] < end)}
    codes = sorted(window)
    print(f"  {len(codes):,} settled matches from {since} to {until or 'now'}; prod's pre-match prices", flush=True)
    prod_rows = []
    for k in range(0, len(codes), config.MATCH_CHUNK_SIZE):
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], codes[k:k + config.MATCH_CHUNK_SIZE])
    tl = bets.timeline(prod_rows)
    del prod_rows
    prod = {}
    for c in codes:
        line, over = prematch_total(tl, c)
        prod[c] = dict(p1=prematch_p1(tl, c), line=line, over=over)
    rows_by_code = {r["MATCH_CODE"]: r for r in history if r["MATCH_CODE"] in window}
    sched = [rows_by_code[c] for c in codes if c in rows_by_code]
    prices, layers = {}, {}
    for name, spec in [(PROD, None)] + list(arms):
        points = {} if spec is None else read_points(spec, sched, history)
        if spec is not None:
            print(f"  {name}: {sum(1 for c in codes if c in points):,} of {len(codes):,} matches priced ({spec})",
                  flush=True)
        rows, layer = rematch_layer.walk_forward(items_for(name, points, prod, window), BURN_IN_DAYS)
        layers[name] = layer
        for code, v in readings(name, rows, prod).items():
            for arm, val in v.items():
                prices.setdefault(arm, {})[code] = val
    results = {}
    for c, m in window.items():
        s1, s2 = m.get("home_score"), m.get("away_score")
        if s1 is not None and s2 is not None and prod[c]["line"] is not None:
            results[c] = (s1 - s2, s1 + s2, prod[c]["line"])
    first = min((m["start"] for m in window.values()), default=start)
    burn_from = first.replace(hour=0, minute=0, second=0, microsecond=0) + dt.timedelta(days=BURN_IN_DAYS)
    lines = report(prices, results, window, layers, burn_from)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prior_backtest.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        arm_names = list(prices)
        w.writerow(["MATCH_CODE", "start", "meeting", "home", "away", "margin", "total", "prod_line"]
                   + [f"{a}_{x}" for a in arm_names for x in ("p_home", "total", "p_over")])
        for c in sorted(results, key=lambda c: window[c]["start"]):
            m = window[c]
            w.writerow([c, m["start"], m["meeting"], m["home"], m["away"], *results[c]]
                       + [("" if prices[a].get(c, (None,) * 3)[i] is None else round(prices[a][c][i], 4))
                          for a in arm_names for i in range(3)])
    path = os.path.join(out_dir, "prior_backtest.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
