"""v21-react-fit: the win-probability table v2.1's wpa update reads, and every in-play strength update
(v21_react) scored on real games before any is priced.

On the matches before --until, ordered by day:
- the first part fits the table: who won, against each PLAY_OVER's state (v21_react.wp_features:
  the margin, the field and the down over the root of the time left, prod's pre-match price), a
  logistic fit, written to react_wp.json beside the build;
- the rest scores the updates as the sim would use them. At every kick-off of regulation (a fresh
  possession at the receiver's end, so what the state says of the points to come is the time left
  alone) each side's points still to come are set against the prior's -- prod's pre-match spread
  and total, x the share of the game's points left -- and against that plus each variant's nudge x
  the share left (snapT, ptsT, wpaC at a spread of settings). The squared error change, per match,
  averaged with a 95% bootstrap interval over matches: below zero, the update knew something of the
  rest of the game the prior did not. The slope of what came on what the update said says how far
  to trust it: 1 right, under 1 too far, over 1 not far enough.

A win-probability table fitted on real games has learned already that a side ahead is likelier the
better one, so it is not the test: v2.1's sim prices the state off the prior's strength alone, and
an update is worth what it adds to that.

    python -m eAMFModel v21-react-fit --model v21_t10 --until 2026-10-01

It reads the export alone: no Snowflake, no simulation."""

import json
import math
import os
import random
from collections import defaultdict

import numpy as np

from . import v21, v21_react as vr
from .remaining import day

SNAP_TAUS = (1, 2, 3, 4, 6, 8)
PTS_TAUS = (2, 4, 6, 8)
WPA_SCALES = (2.5, 5, 10, 20, 40)
BOOT = 1000


def fit_logistic(X, y, ridge=1e-3, iters=50):
    """Coefficients of a logistic fit (Newton steps, a light ridge)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ b)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * b)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def _loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def match_states(rows, final):
    """[(PLAY_OVER message, wp_state, did the side with the ball win)] of a match, ties left out."""
    prior = vr.prior_logit(rows)
    out = []
    for r in rows:
        s = vr.wp_state(r, prior)
        if s is None:
            continue
        on = s[0]
        margin = (final[0] - final[1]) * (1 if on == 0 else -1)
        if margin:
            out.append((v21.s14._i(r.get("message")), s, float(margin > 0)))
    return out


def _arrays(states):
    st = [s for _, s, _ in states]
    return (np.array([s[0] for s in st]), np.array([s[1] for s in st]), np.array([s[2] for s in st], float),
            np.array([s[3] for s in st]), np.array([s[4] for s in st]), np.array([s[5] for s in st]),
            np.array([y for _, _, y in states]))


def fit_table(per_match):
    """(coefficients, states, log loss) of the win-probability table off [(code, states)]."""
    X, Y = [], []
    for _, states in per_match:
        on, t, m, pr, f, d, y = _arrays(states)
        X.append(vr.wp_features(t, m, pr, f, d))
        Y.append(y)
    X, Y = np.concatenate(X), np.concatenate(Y)
    b = fit_logistic(X, Y)
    return b, len(Y), float(_loss(Y, 1 / (1 + np.exp(-(X @ b)))).mean())


def _boot(deltas, seed=0):
    d = np.asarray(deltas, float)
    if not len(d):
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in range(len(d))])) for _ in range(BOOT))
    return float(d.mean()), means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1]


def expected_points(rows, league):
    """(home, away) points a game the match was priced for before kick-off: prod's pre-match spread
    line (the home margin it is even money on) and total, else the league's."""
    try:
        M, T = float(rows[0].get("prematch_line_52")), float(rows[0].get("prematch_line_54"))
    except (TypeError, ValueError, IndexError):
        return np.asarray(league["mean"], float)
    if not (10 <= T <= 120 and abs(M) <= 60):
        return np.asarray(league["mean"], float)
    return np.array([(T + M) / 2.0, (T - M) / 2.0])


_MODEL = {}


def _model(model_file):
    if model_file not in _MODEL:
        _MODEL.clear()
        _MODEL[model_file] = v21.Model.load(model_file)
    return _MODEL[model_file]


def score_match(job):
    """One test match through v2.1: the kick-off nudged toward the prior's points (the stream's share),
    the sim run from every kick-off after a score with that nudge, each side's points still to come
    beyond the sim's, and each variant's nudge change there. (code, quarter, share of the game left,
    (states, 2) residual, {(variant, setting): (states, 2) nudge change}, (states, 2) each side's lead),
    or None."""
    from . import v21_stream as vs
    code, rows, final, pair, teams, target, model_file, n_paths, coef = job
    model = _model(model_file)
    start = v21.kickoff_start(model, pair, teams)
    if getattr(model, "nudge_dir", None) is not None:
        (h, a), alpha = v21.nudge_solve(model, start, target, n_paths, seed=vs._seed(code), share=vs.NUDGE_SHARE)
    else:
        (h, a), alpha = v21.simulate(model, start, n_paths, seed=vs._seed(code)), np.zeros(2)
    h, a = h.ravel(), a.ravel()
    kick = dict(mean=(h.mean(), a.mean()), var=(h.var(), a.var()))
    starts = [(m, st) for m, st in vs.state_starts(model, rows, pair, teams)
              if int(st["phase"][0]) == v21.KICK and int(st["period"][0]) <= 4]
    if not starts:
        return None
    S = v21.concat_starts([st for _, st in starts])
    S["nudge"] = np.tile(np.asarray(alpha, float), (len(starts), 1))
    sh, sa = v21.simulate(model, S, n_paths, seed=vs._seed(code) + 2)
    now = np.array([[float(st["home"][0]), float(st["away"][0])] for _, st in starts])
    resid = np.asarray(final, float)[None, :] - now - (np.stack([sh.mean(1), sa.mean(1)], axis=1) - now)
    period = np.array([int(st["period"][0]) for _, st in starts])
    left = np.array([1.0 - vr.elapsed_share(model, int(st["period"][0]), float(st["clock"][0])) for _, st in starts])
    msgs = [m for m, _ in starts]
    a0 = np.asarray(alpha, float)
    deltas = {}
    if getattr(model, "nudge_dir", None) is not None:
        got = vr.snap_sums(model, rows, msgs, a0, pair, teams)
        if got is not None:
            U, I = got
            for tau in SNAP_TAUS:
                deltas[("snap", tau)] = vr._clip(a0, U / (I + 1.0 / tau ** 2))
        for tau in PTS_TAUS:
            deltas[("pts", tau)] = vr.points_updates(model, [st for _, st in starts], a0, tau, kick)
        W = vr.wpa_sums(rows, msgs, coef, pair, teams)
        for c in WPA_SCALES:
            deltas[("wpa", c)] = vr._clip(a0, c * W)
    margin = (now[:, 0] - now[:, 1])[:, None] * np.array([[1.0, -1.0]])      # each side's lead
    return code, period, left, resid, deltas, margin


def score_variants(model_file, test, coef, hist, league, n_paths=500, workers=1):
    """{(variant, setting): per match the squared error change of each side's points still to come at
    its kick-offs, the sim's own against the sim's plus the update's nudge x the share left}, the same
    by quarter, and the sums for the slope of the residual on the update's say."""
    jobs = []
    for code, rows, _ in test:
        pair, teams = v21.match_people(code, rows, hist)
        jobs.append((code, rows, v21.match_final(hist[code]), pair, teams,
                     tuple(float(x) for x in expected_points(rows, league)), model_file, n_paths, tuple(coef)))
    if workers > 1:
        from . import v14
        with v14.pool_context().Pool(workers) as pool:
            got = pool.map(score_match, jobs, chunksize=1)
    else:
        got = [score_match(j) for j in jobs]
    out = defaultdict(list)
    by_q = defaultdict(lambda: defaultdict(list))
    slope = defaultdict(lambda: np.zeros(2))
    lead = []
    for g in got:
        if g is None:
            continue
        _, period, left, resid, deltas, margin = g
        x = margin * left[:, None]
        lead.append((float((resid * x).sum()), float((x * x).sum())))
        for key, delta in deltas.items():
            pred = delta * left[:, None]
            change = (resid - pred) ** 2 - resid ** 2
            out[key].append(float(change.mean()))
            slope[key] += (float((resid * pred).sum()), float((pred * pred).sum()))
            for k in range(1, 5):
                if np.any(period == k):
                    by_q[key][k].append(float(change[period == k].mean()))
    slope["_lead"] = np.array(lead) if lead else np.zeros((0, 2))
    return out, by_q, slope


def lead_slope(per_match, seed=0):
    """(slope, 95% interval over matches) of the points still to come beyond the sim's on the side's
    lead x the share left: below zero, the sim gives a side ahead more of the rest of the game than it
    gets (its game script by the lead runs the wrong way), which any update that follows the lead
    inherits."""
    a = np.asarray(per_match, float)
    if not len(a) or a[:, 1].sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    draws = []
    for _ in range(BOOT):
        idx = [rng.randrange(len(a)) for _ in range(len(a))]
        xy, xx = a[idx, 0].sum(), a[idx, 1].sum()
        if xx > 0:
            draws.append(xy / xx)
    draws.sort()
    return float(a[:, 0].sum() / a[:, 1].sum()), draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]


def run(data, history, model_file=None, since=None, until=None, test_share=0.4, max_matches=None,
        n_paths=500, workers=1):
    """(the report's lines, the table and the scores as a dict) off the export's matches before until."""
    hist = {h["MATCH_CODE"]: h for h in (history or [])}
    codes = sorted((c for c in data if (since is None or day(c) >= since) and (until is None or day(c) < until)
                    and c in hist and v21.match_final(hist[c]) is not None), key=lambda c: (day(c), c))
    if max_matches:
        codes = codes[-max_matches:]
    per = []
    for c in codes:
        rows = sorted(data[c], key=lambda r: v21.s14._i(r.get("message")) or 0)
        states = match_states(rows, v21.match_final(hist[c]))
        if states:
            per.append((c, rows, states))
    if len(per) < 20:
        raise SystemExit(f"v21-react-fit: {len(per)} matches with a final and PLAY_OVERs before {until}; too few to fit")
    cut = int(len(per) * (1 - test_share))
    train, test = per[:cut], per[cut:]
    coef, n_states, ll = fit_table([(c, s) for c, _, s in train])
    finals = np.array([v21.match_final(hist[c]) for c, _, _ in train], float)
    league = dict(mean=(float(finals[:, 0].mean()), float(finals[:, 1].mean())),
                  var=(float(finals[:, 0].var()), float(finals[:, 1].var())))
    scores, by_q, slope = (score_variants(model_file, test, coef, hist, league, n_paths, workers)
                           if model_file else ({}, {}, {}))
    L = [f"  the win-probability table (the wpa update's): {len(train):,} matches ({day(train[0][0])} to "
         f"{day(train[-1][0])}), {n_states:,} PLAY_OVERs, log loss {ll:.4f}",
         "    " + ", ".join(f"{n} {b:+.3f}" for n, b in zip(vr.WP_FEATURES, coef)),
         f"  the updates on the {len(test):,} matches after ({day(test[0][0])} to {day(test[-1][0])}), at every"
         f" kick-off after a score: each side's points still to come against v2.1's own from there ({n_paths:,}"
         " games, the kick-off nudged toward prod's pre-match spread and total), and against that plus the"
         " update's nudge x the share of the game left",
         "  squared error change a side a kick-off (below zero: the update knew something the sim did not), 95%"
         " interval over matches; slope: what came against what the update said (1: right, under 1: too much)",
         f"  {'':10s} {'all':>26s} {'slope':>7s} {'Q1':>8s} {'Q2':>8s} {'Q3':>8s} {'Q4':>8s}"]
    table = {}
    lead = slope.pop("_lead", None) if isinstance(slope, dict) else None
    for (name, setting), d in sorted(scores.items()):
        mean, lo, hi = _boot(d)
        xy, xx = slope[(name, setting)]
        b_ = xy / xx if xx > 0 else float("nan")
        qs = [float(np.mean(by_q[(name, setting)][k])) if by_q[(name, setting)][k] else float("nan")
              for k in (1, 2, 3, 4)]
        label = f"{name}{setting:g}"
        table[label] = dict(mean=mean, lo=lo, hi=hi, slope=b_, by_quarter=qs, matches=len(d))
        L.append(f"  {label:10s} {mean:+8.3f} [{lo:+7.3f}, {hi:+7.3f}] {b_:7.2f} " + " ".join(f"{x:+8.3f}" for x in qs))
    best = {k: min((v for v in table.items() if v[0].startswith(k)), key=lambda kv: kv[1]["mean"], default=None)
            for k in ("snap", "pts", "wpa")}
    if lead is not None and len(lead):
        b_, lo_, hi_ = lead_slope(lead)
        L.append(f"  the sim's own game script by the lead: points still to come beyond its, on the side's lead x"
                 f" the share left, slope {b_:+.3f} [{lo_:+.3f}, {hi_:+.3f}] (0: right; below 0: it gives the side"
                 " ahead too much of the rest, and an update that follows the lead looks the worse for it)")
        table["_lead"] = dict(slope=b_, lo=lo_, hi=hi_)
    L.append("  best of each: " + ", ".join(f"{b[0]} {b[1]['mean']:+.3f} (slope {b[1]['slope']:.2f})"
                                           for b in best.values() if b))
    if not model_file:
        L.append("  (the updates are scored through the build: --model)")
    return L, dict(coef=[float(x) for x in coef], features=list(vr.WP_FEATURES), eps=vr.WP_EPS,
                   fit_matches=len(train), fit_states=n_states, log_loss=ll,
                   until=str(until) if until else None, scores=table)


def write(path, got):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(got, fh, indent=1)
    return path
