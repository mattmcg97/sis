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


CARRY_DECAY = 0.7              # an earlier match of the session counts this much less for each match since
SESSION_GAP = 240              # minutes between a gamer's matches that end a session


def sessions(history):
    """{(match, gamer): [earlier matches of the gamer's session, latest first]} off the history's
    scheduled starts (a run with no gap over SESSION_GAP minutes)."""
    import datetime as dt
    by = defaultdict(list)
    for h in history or []:
        try:
            t = dt.datetime.fromisoformat(str(h.get("SCHEDULED_START_TIME_UTC"))[:19])
        except ValueError:
            continue
        for side in ("PLAYER_1_HANDLE", "PLAYER_2_HANDLE"):
            g = (h.get(side) or "").strip().upper()
            if g:
                by[g].append((t, h["MATCH_CODE"]))
    out = {}
    for g, ms in by.items():
        ms.sort()
        run = []
        for k, (t, c) in enumerate(ms):
            if k and (t - ms[k - 1][0]).total_seconds() > SESSION_GAP * 60:
                run = []
            out[(c, g)] = [x for _, x in reversed(run)]
            run.append((t, c))
    return out


def match_evidence(model, rows, pair, teams, coef, mu):
    """{side: (snap score, snap information, points over the prior's, win chance added)} over a whole
    match: what each side's play showed, for a later match of its gamer's session to carry."""
    out = {0: [0.0, 0.0, 0.0, 0.0], 1: [0.0, 0.0, 0.0, 0.0]}
    last = [max(v21.s14._i(r.get("message")) or 0 for r in rows)]
    if model is not None and getattr(model, "nudge_dir", None) is not None:
        got = vr.snap_sums(model, rows, last, None, pair, teams)
        if got is not None:
            for sd in (0, 1):
                out[sd][0], out[sd][1] = float(got[0][0, sd]), float(got[1][0, sd])
    W = vr.wpa_sums(rows, last, coef, pair, teams)
    for sd in (0, 1):
        out[sd][3] = float(W[0, sd])
    try:
        fin = (float(rows[-1].get("score_p1")), float(rows[-1].get("score_p2")))
    except (TypeError, ValueError):
        fin = None
    if fin is not None:
        for sd in (0, 1):
            out[sd][2] = fin[sd] - float(mu[sd])
    return out


def carried(earlier):
    """(snap score, snap information, points over, games, win chance added) a gamer brings to a match
    from the earlier matches of the session ([evidence of the gamer's side in each, latest first]),
    each match further back worth CARRY_DECAY less."""
    c = np.zeros(5)
    for k, e in enumerate(earlier):
        w = CARRY_DECAY ** k
        c += w * np.array([e[0], e[1], e[2], 1.0, e[3]])
    return c


def score_match(job):
    """One test match through v2.1: the kick-off nudged toward the prior's points (the stream's share),
    the sim run from the kick-off and every kick-off after a score with that nudge, each side's points
    still to come beyond the sim's, and each variant's nudge change there -- the in-play updates, and
    the same carried from the earlier matches of each gamer's session (c-). A dict, or None."""
    from . import v21_stream as vs
    code, rows, final, pair, teams, target, model_file, n_paths, coef, carry = job
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
    exp_rest, now, period, left, msgs = [np.array(kick["mean"])], [np.zeros(2)], [1], [1.0], [-1]
    if starts:                                    # the opening kick-off first: nothing of the match known yet
        S = v21.concat_starts([st for _, st in starts])
        S["nudge"] = np.tile(np.asarray(alpha, float), (len(starts), 1))
        sh, sa = v21.simulate(model, S, n_paths, seed=vs._seed(code) + 2)
        for k, (m, st) in enumerate(starts):
            n = np.array([float(st["home"][0]), float(st["away"][0])])
            now.append(n)
            exp_rest.append(np.array([sh[k].mean(), sa[k].mean()]) - n)
            period.append(int(st["period"][0]))
            left.append(1.0 - vr.elapsed_share(model, int(st["period"][0]), float(st["clock"][0])))
            msgs.append(m)
    now, exp_rest, period, left = np.array(now), np.array(exp_rest), np.array(period), np.array(left)
    resid = np.asarray(final, float)[None, :] - now - exp_rest
    a0 = np.asarray(alpha, float)
    deltas = {}
    C = np.array([carry.get(0, np.zeros(5)), carry.get(1, np.zeros(5))]) if carry else np.zeros((2, 5))
    if getattr(model, "nudge_dir", None) is not None:
        got = vr.snap_sums(model, rows, msgs, a0, pair, teams)
        U, I = got if got is not None else (np.zeros((len(msgs), 2)), np.zeros((len(msgs), 2)))
        for tau in SNAP_TAUS:
            deltas[("snap", tau)] = vr._clip(a0, U / (I + 1.0 / tau ** 2))
            deltas[("csnap", tau)] = vr._clip(a0, (U + C[None, :, 0]) / (I + C[None, :, 1] + 1.0 / tau ** 2))
        sofar = now - np.asarray(kick["mean"])[None, :] * (1.0 - left)[:, None]
        var = np.maximum(np.asarray(kick["var"], float), 1.0)
        for tau in PTS_TAUS:
            f = (1.0 - left)[:, None]
            deltas[("pts", tau)] = vr._clip(a0, tau ** 2 * sofar / (var[None] + tau ** 2 * f))
            deltas[("cpts", tau)] = vr._clip(a0, tau ** 2 * (sofar + C[None, :, 2])
                                             / (var[None] + tau ** 2 * (f + C[None, :, 3])))
        W = vr.wpa_sums(rows, msgs, coef, pair, teams)
        for c in WPA_SCALES:
            deltas[("wpa", c)] = vr._clip(a0, c * W)
            deltas[("cwpa", c)] = vr._clip(a0, c * (W + C[None, :, 4]))
    margin = (now[:, 0] - now[:, 1])[:, None] * np.array([[1.0, -1.0]])      # each side's lead
    return dict(code=code, pair=pair, period=period, left=left, resid=resid, deltas=deltas, margin=margin)


def score_variants(model_file, test, coef, hist, league, n_paths=500, workers=1, evidence=None, sess=None):
    """The test matches through score_match: [its dict] in the matches' order."""
    jobs = []
    for code, rows, _ in test:
        pair, teams = v21.match_people(code, rows, hist)
        carry = {}
        if evidence is not None and sess is not None:
            for sd in (0, 1):
                earlier = [evidence[c][s_] for c in sess.get((code, pair[sd]), []) if c in evidence
                           for s_ in (0, 1) if evidence[c].get("pair", ("", ""))[s_] == pair[sd]]
                carry[sd] = carried(earlier)
        jobs.append((code, rows, v21.match_final(hist[code]), pair, teams,
                     tuple(float(x) for x in expected_points(rows, league)), model_file, n_paths, tuple(coef), carry))
    if workers > 1:
        from . import v14
        with v14.pool_context().Pool(workers) as pool:
            got = pool.map(score_match, jobs, chunksize=1)
    else:
        got = [score_match(j) for j in jobs]
    return [g for g in got if g is not None]


def tally(scored, key, scale=None):
    """(per match the squared error change, by quarter, (xy, xx)) of a variant over scored matches;
    scale: {match: (states, 2) multiplier} on its nudge (the gamer-specific ones)."""
    out, by_q, xy, xx = [], defaultdict(list), 0.0, 0.0
    for g in scored:
        if key not in g["deltas"]:
            continue
        pred = g["deltas"][key] * g["left"][:, None]
        if scale is not None:
            pred = pred * scale[g["code"]]
        change = (g["resid"] - pred) ** 2 - g["resid"] ** 2
        out.append(float(change.mean()))
        xy += float((g["resid"] * pred).sum())
        xx += float((pred * pred).sum())
        by_q["open"].append(float(change[0].mean()))
        for k in range(1, 5):
            if np.any(g["period"] == k):
                by_q[k].append(float(change[g["period"] == k].mean()))
    return out, by_q, (xy, xx)


def gamer_scales(scored, key):
    """(pooled slope, {gamer: own slope}) of the residual on a variant's say, each gamer's shrunk toward
    the pooled by an average gamer's evidence (none below zero)."""
    xy, xx = defaultdict(float), defaultdict(float)
    for g in scored:
        if key not in g["deltas"]:
            continue
        pred = g["deltas"][key] * g["left"][:, None]
        for sd in (0, 1):
            name = g["pair"][sd]
            if name:
                xy[name] += float((g["resid"][:, sd] * pred[:, sd]).sum())
                xx[name] += float((pred[:, sd] ** 2).sum())
    tot_xx = sum(xx.values())
    if tot_xx <= 0:
        return 1.0, {}
    k = sum(xy.values()) / tot_xx
    lam = tot_xx / max(1, sum(1 for v in xx.values() if v > 0))
    return k, {name: max(0.0, (xy[name] + lam * k) / (xx[name] + lam)) for name in xx}


def scale_of(scored, k, ks):
    """{match: (states, 2)} each side's multiplier: its gamer's own slope, else the pooled."""
    return {g["code"]: np.tile([ks.get(g["pair"][0], k), ks.get(g["pair"][1], k)], (len(g["left"]), 1))
            for g in scored}


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


def _scored_line(label, d, by_q, xy_xx):
    mean, lo, hi = _boot(d)
    xy, xx = xy_xx
    b_ = xy / xx if xx > 0 else float("nan")
    qs = [float(np.mean(by_q[k])) if by_q.get(k) else float("nan") for k in ("open", 1, 2, 3, 4)]
    return (f"  {label:10s} {mean:+8.3f} [{lo:+7.3f}, {hi:+7.3f}] {b_:7.2f} " + " ".join(f"{x:+8.3f}" for x in qs),
            dict(mean=mean, lo=lo, hi=hi, slope=b_, by_quarter=qs[1:], open=qs[0], matches=len(d)))


def _paired(a, b, seed=0):
    """(mean of b - a, 95% bootstrap interval) over matches scored by both."""
    d = np.asarray(b, float) - np.asarray(a, float)
    return _boot(d, seed)


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
    coef, n_states, ll = fit_table([(c, s_) for c, _, s_ in train])
    finals = np.array([v21.match_final(hist[c]) for c, _, _ in train], float)
    league = dict(mean=(float(finals[:, 0].mean()), float(finals[:, 1].mean())),
                  var=(float(finals[:, 0].var()), float(finals[:, 1].var())))
    L = [f"  the win-probability table (the wpa update's): {len(train):,} matches ({day(train[0][0])} to "
         f"{day(train[-1][0])}), {n_states:,} PLAY_OVERs, log loss {ll:.4f}",
         "    " + ", ".join(f"{n} {b:+.3f}" for n, b in zip(vr.WP_FEATURES, coef))]
    got = dict(coef=[float(x) for x in coef], features=list(vr.WP_FEATURES), eps=vr.WP_EPS,
               fit_matches=len(train), fit_states=n_states, log_loss=ll,
               until=str(until) if until else None, scores={}, gamers={})
    if not model_file:
        L.append("  (the updates are scored through the build: --model)")
        return L, got
    model = _model(model_file)
    evidence = {}
    for c, rows, _ in per:                        # what every match showed of each side, for the carry
        pair, teams = v21.match_people(c, rows, hist)
        ev = match_evidence(model, rows, pair, teams, coef, expected_points(rows, league))
        ev["pair"] = pair
        evidence[c] = ev
    sess = sessions(history)
    scored = score_variants(model_file, test, coef, hist, league, n_paths, workers, evidence, sess)
    L += [f"  the updates on the {len(test):,} matches after ({day(test[0][0])} to {day(test[-1][0])}), at the"
          f" kick-off and every kick-off after a score: each side's points still to come against v2.1's own from"
          f" there ({n_paths:,} games, the kick-off nudged toward prod's pre-match spread and total), and against"
          " that plus the update's nudge x the share of the game left. c-: the same, with the gamer's earlier"
          f" matches of the session carried in (each one back worth x{CARRY_DECAY:g})",
          "  squared error change a side a kick-off (below zero: the update knew something the sim did not), 95%"
          " interval over matches; slope: what came against what the update said (1: right, under 1: too much)",
          f"  {'':10s} {'all':>26s} {'slope':>7s} {'opening':>8s} {'Q1':>8s} {'Q2':>8s} {'Q3':>8s} {'Q4':>8s}"]
    keys = sorted({k for g in scored for k in g["deltas"]})
    table = {}
    for key in keys:
        d, by_q, xyxx = tally(scored, key)
        line, row = _scored_line(f"{key[0]}{key[1]:g}", d, by_q, xyxx)
        L.append(line)
        table[f"{key[0]}{key[1]:g}"] = row
    lead = []
    for g in scored:
        x = g["margin"] * g["left"][:, None]
        lead.append((float((g["resid"] * x).sum()), float((x * x).sum())))
    b_, lo_, hi_ = lead_slope(lead)
    L.append(f"  the sim's own game script by the lead: points still to come beyond its, on the side's lead x the share"
             f" left, slope {b_:+.3f} [{lo_:+.3f}, {hi_:+.3f}] (0: right; below 0: it gives the side ahead too much of"
             " the rest, and an update that follows the lead looks the worse for it)")
    table["_lead"] = dict(slope=b_, lo=lo_, hi=hi_)
    # gamer by gamer: each approach at its best-scoring setting, every gamer's own slope fitted on the earlier
    # part of the test matches and scored on the later against the pooled slope
    split = int(len(scored) * 0.6)
    fit_part, hold = scored[:split], scored[split:]
    L.append(f"  gamer by gamer: each approach at its best-scoring setting (of those with a slope above 0), every gamer's own slope (shrunk"
             f" toward the pooled by an average gamer's evidence) fitted on the first {len(fit_part):,} of these matches"
             f" and scored on the last {len(hold):,}, against the pooled slope there")
    gamers = {}
    for kind_ in ("snap", "pts", "wpa"):
        cands = [(row["mean"], k) for k, row in table.items()
                 if k.startswith(kind_) and k[len(kind_):].replace(".", "").isdigit() and row["slope"] > 0]
        if not cands:
            continue
        label = min(cands)[1]
        key = (kind_, float(label[len(kind_):]))
        if key not in {k for g in scored for k in g["deltas"]}:
            key = (kind_, int(label[len(kind_):]))
        k, ks = gamer_scales(fit_part, key)
        raw, _, _ = tally(hold, key)
        pooled, _, _ = tally(hold, key, scale_of(hold, k, {}))
        own, _, _ = tally(hold, key, scale_of(hold, k, ks))
        m_p, lo_p, hi_p = _boot(pooled)
        m_o, lo_o, hi_o = _boot(own)
        dm, dlo, dhi = _paired(pooled, own)
        spread = np.array(list(ks.values())) if ks else np.zeros(1)
        L.append(f"  {label:8s}: pooled slope {k:.2f}, gamers' own {np.percentile(spread, 10):.2f} to"
                 f" {np.percentile(spread, 90):.2f} (10th-90th); on the later matches, as it is {np.mean(raw):+.3f},"
                 f" scaled by the pooled slope {m_p:+.3f} [{lo_p:+.3f}, {hi_p:+.3f}], by each gamer's own {m_o:+.3f}"
                 f" [{lo_o:+.3f}, {hi_o:+.3f}]; gamer's own less pooled {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
        k_all, ks_all = gamer_scales(scored, key)
        gamers[kind_] = dict(setting=float(key[1]), k=k_all, gamers={g: round(v, 4) for g, v in sorted(ks_all.items())})
    best = {k: min((v for v in table.items() if v[0].startswith(k) and not v[0].startswith("_")),
                   key=lambda kv: kv[1]["mean"], default=None)
            for k in ("snap", "pts", "wpa", "csnap", "cpts", "cwpa")}
    L.append("  best of each: " + ", ".join(f"{b[0]} {b[1]['mean']:+.3f} (slope {b[1]['slope']:.2f})"
                                           for b in best.values() if b))
    got.update(scores=table, gamers=gamers)
    return L, got


def write(path, got):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(got, fh, indent=1)
    return path
