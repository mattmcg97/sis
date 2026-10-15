"""Pre-match priors against prod, each with and without the rematch layer, graded on the results.

    python -m eAMFCalibrator prior-backtest --since 2026-09-01 --until 2026-10-14 \\
        --arm "glmer-trim=glmer/out/cadence/cadence_matches.csv@trim:hl60:game" \\
        --arm "nb2=rolling:nb2_daily"

An arm is NAME=SOURCE[#MODE], its expected points per side for every match as of the match's
kick-off:
  PATH@PREFIX   glmer/cadence.R's cadence_matches.csv: the columns PREFIX|1 and PREFIX|2
  PATH          a CSV with MATCH_CODE (or MatchId) and the two sides' expected points:
                Pred_P1_Points / Pred_P2_Points (NB2_schedule_predict.py, glmer/predict.R),
                raw_home / raw_away (eAMFModel prior-vs-sim), or exp_p1 / exp_p2
  rolling:DIR   eAMFModel prior-daily's directory of daily fits (NB2 or glmer): each match priced by
                its own day's fit, as live
prod is always there: its last pre-match moneyline and total (line and the over's price), from
GAMEPLAI_STREAM.

MODE is how an arm's moneyline and over are read off its expected points (score_model.py), each
match's two scores simulated together:
  own      the prior's own joint score model: NB2's (each side negative binomial with its gamers'
           dispersions, a shared gamma pace factor: covariance) for an NB2 source -- a rolling NB2
           directory, or a CSV with NB2's P1_Effective_Alpha / P2_Effective_Alpha /
           Shared_Gamma_Fraction -- and glmer's (Poisson about a log-normal mean, a shared match
           noise) for a rolling glmer directory. The default where there is one
  fitted   NB2's construction with one dispersion and shared fraction fitted to the arm's own errors
           by the method of moments, walk-forward (each UTC day on the days before it): for a prior
           with no score model of its own, as glmer trimmed of its noise terms. The default otherwise
  normal   the normal about the expected margin (sd 9.35) and total (sd 12): for comparison
A moneyline splits ties half and half (NB2's way); the over leaves out pushes.

Each arm, prod among them, is graded as it is and with the rematch layer (rematch_layer.py) on top,
fitted walk-forward on that arm's own errors (each UTC day on the days before it). The layer shifts
the arm's simulated margin and total, so it moves the model's own price. prod's own are its prices;
prod with the layer moves the margin its moneyline implies and the mean its total implies.

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

from . import bets, config, rematch_layer, score_model, snowflake_io
from .bet_losers import prematch_p1
from .bet_rematches import day_meetings
from .bet_totals import _start

TOTAL_SD = 12.0
BURN_IN_DAYS = 7
PROD = "prod"
POINT_COLUMNS = (("Pred_P1_Points", "Pred_P2_Points"), ("raw_home", "raw_away"), ("exp_p1", "exp_p2"))
NB2_JOINT = ("P1_Effective_Alpha", "P2_Effective_Alpha", "Shared_Gamma_Fraction")
MODES = ("own", "fitted", "normal")
NB2_PREDICT_SIMS = 2000          # NB2's own simulation is not read here (score_model draws again)
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


def read_arm(spec, rows, history):
    """({match: (home, away) expected points}, {match: (kind, params)} of the prior's own score model
    where it has one) for one arm's source."""
    if spec.startswith("rolling:"):
        return read_rolling(spec[len("rolling:"):], rows, history)
    points = read_points(spec, rows, history)
    own = {}
    path = spec.partition("@")[0]
    if "@" not in spec:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            if all(c in cols for c in NB2_JOINT):
                key = "MATCH_CODE" if "MATCH_CODE" in cols else "Match Id" if "Match Id" in cols else "MatchId"
                for r in reader:
                    a1, a2, sh = (_f(r.get(c)) for c in NB2_JOINT)
                    if r.get(key) in points and None not in (a1, a2, sh):
                        own[r[key]] = ("nb2", points[r[key]] + (a1, a2, sh))
    return points, own


def read_rolling(directory, rows, history):
    """A rolling prior's expected points and its own score model, each match by its own day's fit:
    NB2's dispersions and shared fraction (its means times the fit's level scale), or glmer's noise
    sds."""
    from eAMFModel import nb2_prior, rolling_prior
    prior = rolling_prior.Rolling(directory)
    if prior.kind != "nb2":
        means = prior.means(rows, results=history)
        points = {k: (float(v[0]), float(v[1])) for k, v in means.items()}
        own = {}
        for r in rows:
            code = r["MATCH_CODE"]
            if code not in points:
                continue
            start = _start(r.get("SCHEDULED_START_TIME_UTC"))
            fit = prior.fit_for(start.date() if start else None)
            m = (getattr(fit, "meta", None) or {}).get("model", {}) or {}
            own[code] = ("lognormal", points[code] + (float(m.get("sigma_obs") or 0.0),
                                                       float(m.get("sigma_match") or 0.0)))
        return points, own
    by_fit = defaultdict(list)
    for r in rows:
        start = _start(r.get("SCHEDULED_START_TIME_UTC"))
        by_fit[id(fit := prior.fit_for(start.date() if start else None))].append((fit, r))
    points, own = {}, {}
    for group in by_fit.values():
        fit = group[0][0]
        got = nb2_prior.predict(fit.directory, [r for _, r in group], NB2_PREDICT_SIMS, full=True)
        for code, row in got.items():
            mu1, mu2 = float(row["Pred_P1_Points"]) * fit.scale, float(row["Pred_P2_Points"]) * fit.scale
            a1, a2, sh = (_f(row.get(c)) for c in NB2_JOINT)
            points[code] = (mu1, mu2)
            if None not in (a1, a2, sh):
                own[code] = ("nb2", (mu1, mu2, a1, a2, sh))
    return points, own


def read_points(spec, rows, history):
    """{match: (home, away) expected points} for one arm's CSV source."""
    path, _, prefix = spec.partition("@")
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols = reader.fieldnames or []
        key = "MATCH_CODE" if "MATCH_CODE" in cols else "Match Id" if "Match Id" in cols else "MatchId"
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


def fitted_by_day(rows):
    """{UTC day: (alpha, shared)}: NB2's dispersion and shared fraction fitted to an arm's errors on
    every match before that day (score_model.fit_nb2); (None, None) until there are enough."""
    out, train, day = {}, [], None
    for r in sorted(rows, key=lambda r: r["start"]):
        d = r["start"].date()
        if d != day:
            day = d
            out[d] = score_model.fit_nb2(train)
        if r["actual_margin"] is not None and r["margin"] is not None:
            mu1, mu2 = (r["total"] + r["margin"]) / 2, (r["total"] - r["margin"]) / 2
            y1, y2 = (r["actual_total"] + r["actual_margin"]) / 2, (r["actual_total"] - r["actual_margin"]) / 2
            train.append((mu1, mu2, y1, y2))
    return out


def score_book(r, mode, own, fits, draws):
    """The pricing object of one arm's match: its own score model, NB2's fitted one, or the normal."""
    mu1, mu2 = (r["total"] + r["margin"]) / 2, (r["total"] - r["margin"]) / 2
    if mode == "own" and r["match"] in own:
        kind, params = own[r["match"]]
        return score_model.book(kind, params, r["match"], draws)
    if mode == "fitted":
        alpha, shared = fits.get(r["start"].date(), (None, None))
        if alpha is not None:
            return score_model.book("nb2", (max(mu1, 0.1), max(mu2, 0.1), alpha, alpha, shared), r["match"], draws)
    return score_model.Normal(r["margin"], r["total"])


def readings(name, rows, prod, mode="normal", own=None, draws=score_model.DRAWS):
    """{match: {variant: (p home, expected total, p over prod's line)}} for an arm and its +layer."""
    own = own or {}
    fits = fitted_by_day(rows) if mode == "fitted" else {}
    out = {}
    for r in rows:
        p = prod.get(r["match"]) or {}
        line, over = p.get("line"), p.get("over")
        v = {}
        if name == PROD:
            v[name] = (p.get("p1"), line, over)
            # prod moved: the margin its moneyline implies, and the mean its total implies
            base = line + TOTAL_SD * _N.inv_cdf(min(max(over, 1e-6), 1 - 1e-6)) if over is not None else line
            t = None if base is None else base + r["move_total"]
            v[name + "+layer"] = (rematch_layer.win_chance(r["margin_adj"]), t,
                                  None if line is None or t is None else over_chance(t, line))
        else:
            bk = score_book(r, mode, own, fits, draws)
            for tag, dm, dt_ in (("", 0.0, 0.0), ("+layer", r["move_margin"], r["move_total"])):
                v[name + tag] = (bk.p_home(dm), r["total"] + dt_, None if line is None else bk.p_over(line, dt_))
        out[r["match"]] = v
    return out, fits


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


def describe_model(mode, own, fits):
    """One line: how an arm's moneyline and over were read."""
    if mode is None:
        return "its own prices"
    if mode == "own":
        kinds = defaultdict(list)
        for kind, params in own.values():
            kinds[kind].append(params)
        parts = []
        for kind, ps in kinds.items():
            if kind == "nb2":
                parts.append(f"NB2's own: dispersion {sum(p[2] + p[3] for p in ps) / (2 * len(ps)):.3f} on average, "
                             f"shared fraction {sum(p[4] for p in ps) / len(ps):.3f}")
            else:
                parts.append(f"glmer's own: s_obs {sum(p[2] for p in ps) / len(ps):.3f}, "
                             f"s_match {sum(p[3] for p in ps) / len(ps):.3f}")
        return "; ".join(parts) + " (the normal for a match without them)"
    if mode == "fitted":
        last = [v for _, v in sorted(fits.items()) if v[0] is not None]
        return ("NB2's construction fitted to its errors, walk-forward" +
                (f"; the last fit: dispersion {last[-1][0]:.3f}, shared fraction {last[-1][1]:.3f}" if last else
                 " (never enough matches: the normal)"))
    return "the normal about its expected margin and total"


def report(prices, results, meets, layers, burn_from, models=None):
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
    if models:
        L.append("\n  How each arm's moneyline and over are read:")
        L += [f"  {name}: {text}" for name, text in models.items()]
    L.append("\n  The rematch layers' last fits (each arm's own errors):")
    for name, layer in layers.items():
        L.append(f"  {name}:")
        L += ["  " + x for x in layer.describe()]
    return L


def parse_arm(spec):
    """(source, mode) off SOURCE[#MODE]."""
    source, _, mode = spec.partition("#")
    if mode and mode not in MODES:
        raise SystemExit(f"{spec!r}: the mode after # is one of {', '.join(MODES)}")
    return source, mode or None


def prepare(cur, since, until=None):
    """What every arm is read against: the history, the window's matches with their day's meetings,
    prod's quote timeline and its pre-match prices, and the window's history rows."""
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
    first = min((m["start"] for m in window.values()), default=start)
    return dict(history=history, window=window, codes=codes, tl=tl, prod=prod,
                sched=[rows_by_code[c] for c in codes if c in rows_by_code],
                burn_from=first.replace(hour=0, minute=0, second=0, microsecond=0)
                + dt.timedelta(days=BURN_IN_DAYS))


def fit_arm(name, spec, ctx):
    """(mode, own score models, layer rows, the last layer) for one arm (prod: spec None): its expected
    points read, the rematch layer fitted walk-forward on its own errors."""
    points, own, mode = {}, {}, None
    codes = ctx["codes"]
    if spec is not None:
        source, mode = parse_arm(spec)
        points, own = read_arm(source, ctx["sched"], ctx["history"])
        if mode == "own" and not own:
            raise SystemExit(f"--arm {name}: {source} has no score model of its own (only NB2 and a rolling "
                             "glmer do); use #fitted or #normal")
        mode = mode or ("own" if own else "fitted")
        print(f"  {name}: {sum(1 for c in codes if c in points):,} of {len(codes):,} matches priced ({source}), "
              f"scores {mode}", flush=True)
    rows, layer = rematch_layer.walk_forward(items_for(name, points, ctx["prod"], ctx["window"]), BURN_IN_DAYS)
    return mode, own, rows, layer


def run(cur, out_dir, arms, since, until=None, draws=score_model.DRAWS):
    """Grade prod and every arm, each with and without the rematch layer, over [since, until]."""
    ctx = prepare(cur, since, until)
    window, prod = ctx["window"], ctx["prod"]
    prices, layers, models = {}, {}, {}
    for name, spec in [(PROD, None)] + list(arms):
        mode, own, rows, layer = fit_arm(name, spec, ctx)
        layers[name] = layer
        got, fits = readings(name, rows, prod, mode or "normal", own, draws)
        models[name] = describe_model(mode, own, fits)
        for code, v in got.items():
            for arm, val in v.items():
                prices.setdefault(arm, {})[code] = val
    results = {}
    for c, m in window.items():
        s1, s2 = m.get("home_score"), m.get("away_score")
        if s1 is not None and s2 is not None and prod[c]["line"] is not None:
            results[c] = (s1 - s2, s1 + s2, prod[c]["line"])
    burn_from = ctx["burn_from"]
    lines = report(prices, results, window, layers, burn_from, models)
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
