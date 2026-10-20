"""playstyle: how each gamer plays by the score, measured on real games rather than guessed.

Strength by the score. At the opening kick-off and the kick-off after every score in regulation (a
conversion's, a made field goal's or a safety's PLAY_OVER: the score and the time left are the whole
state, the receiver's ball at its 25), each side's points still to come are set against its expected
points (prod's pre-match spread and total in the export, or --prior's table) x the league's share of a
game's points still to come at that time left. What is left over is fitted three ways:

  the league   on the side's possession (the side that conceded receives, so a trailing side has the
               ball more) and its lead x the share of the game left: how the league's games run by the
               score. v2.1's sim plays real snaps for the situation, so this part is in it already
               (v21-react-fit found its game script right, -0.028 [-0.070, +0.013]).
  each gamer   beyond the league, the margin still to come (own points less the other side's) on the
               share left and the lead x the share left: the first the gamer's own over- or
               under-performance of its price, kept apart so a gamer prod prices short does not pass
               for a front-runner; the second the slope. Above 0, a front-runner (a lead grows, a
               deficit too); below 0, a gamer whose leads shrink and deficits close.
  a trait?     every gamer's matches split by alternate days, the slope fitted on each half, and the
               two halves correlated across gamers. A playstyle shows up in both halves; noise does
               not. Only when the correlation's 95% interval is above 0 are the gamers put in groups.

The spread of the true slopes across gamers (tau, DerSimonian-Laird on the bootstrap errors) says how
far apart they are beyond noise; each gamer's slope is shrunk toward the others by it.

Pace by the score. Every snap of the second half but its last two minutes (the hurry-up everyone runs)
and those a quarter's end cut short: the seconds it took off the clock, ahead against behind, for the
league and, beyond the league's, for each gamer, the same split-half test on top.

    python -m eAMFModel playstyle --snapshots eAMFCalibrator/out/scouting_playover.csv \\
        --history eAMFCalibrator/out/match_history.csv --out eAMFCalibrator/out/playstyle

It reads the export alone: no Snowflake, no simulation."""

import csv
import math
import os
from collections import defaultdict

import numpy as np

from . import playover, v21, v21_react as vr
from .remaining import day
from .state import AWAY, HOME
from .v21_react_fit import expected_points

MIN_MATCHES = 40          # a gamer's matches for a slope
MIN_HALF = 15             # and in each half of the split, for the trait test
MIN_SNAPS = 30            # pace: a gamer's snaps ahead, and behind
BOOT = 400                # bootstrap resamples over matches
F_BINS = 12               # the share of the game left, cut for the league's share of points still to come
LATE = 120                # pace: the last two minutes of a half left out
SCORING = ("CONVERSION", "FIELD_GOAL", "SAFETY")
LEAD_BANDS = ((-999, -9, "9+ behind"), (-8, -1, "1-8 behind"), (0, 0, "level"), (1, 8, "1-8 ahead"),
              (9, 999, "9+ ahead"))


# ---------------------------------------------------------------- what the export says

def _side(s):
    return 0 if s == HOME else 1 if s == AWAY else None


def kick_states(rows):
    """[(home, away, share of regulation left, receiver 0/1 or None)]: the opening kick-off and the
    kick-off after every score in regulation, a score once."""
    if not rows:
        return []
    r0 = rows[0]
    out = [(0, 0, 1.0, _side(playover.side_of(r0.get("opening_offense"), r0.get("team_a_side"))))]
    for r in rows:
        if r.get("play_kind") not in SCORING:
            continue
        st, _ = playover.state_for(r)
        if st is None or st.pending_conversion is not None or st.down is not None or _side(st.offense) is None:
            continue
        if st.period is None or st.period > 4 or st.clock_seconds is None or st.clock_seconds <= 0:
            continue
        score = (int(st.home_score), int(st.away_score))
        if score == out[-1][:2]:
            continue
        out.append(score + (vr.time_left(st.period, st.clock_seconds), _side(st.offense)))
    return out


def _date(code, hist_row):
    """The match's day: its scheduled start in the history, else its code's."""
    import datetime as dt
    try:
        return dt.date.fromisoformat(str(hist_row.get("SCHEDULED_START_TIME_UTC"))[:10])
    except (TypeError, ValueError):
        try:
            return day(code)
        except ValueError:
            return None


def read(data, history, prior=None, since=None, until=None):
    """The observations: one a side a kick-off state (`obs`, arrays; the two sides of a state side by
    side, so an observation's other side is its index ^ 1) and one a snap of the second half (`snaps`),
    with the gamers (`gamers`, index -> handle) and the matches. `prior` {code: (home, away)} the
    expected points, else prod's pre-match lines in the export."""
    hist = {h["MATCH_CODE"]: h for h in (history or [])}
    when = {c: _date(c, hist[c]) for c in data if c in hist}
    codes = sorted((c for c, d in when.items() if d is not None and v21.match_final(hist[c]) is not None
                    and (since is None or d >= since) and (until is None or d < until)), key=lambda c: (when[c], c))
    finals = np.array([v21.match_final(hist[c]) for c in codes], float).reshape(-1, 2)
    league = dict(mean=tuple(finals.mean(0)) if len(finals) else (17.0, 17.0))
    gid, O, S = {}, defaultdict(list), defaultdict(list)
    ix = lambda g: gid.setdefault(g, len(gid))
    for m, c in enumerate(codes):
        rows = data[c]
        pair, teams = v21.match_people(c, rows, hist)
        if not all(pair):
            continue
        final = finals[m]
        mu = np.asarray(prior[c], float) if prior and c in prior else expected_points(rows, league)
        parity = when[c].toordinal() % 2
        g = (ix(pair[0]), ix(pair[1]))
        for home, away, left, recv in kick_states(rows):
            now = (home, away)
            for sd in (0, 1):
                O["match"].append(m)
                O["half"].append(parity)
                O["gamer"].append(g[sd])
                O["opp"].append(g[1 - sd])
                O["lead"].append((home - away) * (1 if sd == 0 else -1))
                O["left"].append(left)
                O["recv"].append(0.5 if recv is None else float(recv == sd))
                O["y"].append(final[sd] - now[sd])
                O["mu"].append(mu[sd])
        for r in v21.match_snaps(rows, pair, teams):
            if r["period"] not in (3, 4) or r.get("cut") or (r["period"] == 4 and r["clock"] <= LATE):
                continue
            S["match"].append(m)
            S["half"].append(parity)
            S["gamer"].append(g[r["side"]])
            S["lead"].append(r["margin"])
            S["seconds"].append(r["seconds"])
    arr = lambda d: {k: np.asarray(v, float if k in ("lead", "left", "recv", "y", "mu", "seconds") else np.int64)
                     for k, v in d.items()}
    return dict(obs=arr(O), snaps=arr(S), gamers={i: g for g, i in gid.items()}, codes=codes,
                since=when[codes[0]] if codes else None, until=when[codes[-1]] if codes else None,
                expected="the prior table" if prior else "prod's pre-match lines")


# ---------------------------------------------------------------- the league

def share_left(obs):
    """(bin edges, the league's share of a game's points still to come in each bin of the share of the
    game left): points to come over expected points, every side pooled."""
    edges = np.linspace(0.0, 1.0, F_BINS + 1)
    b = np.clip(np.searchsorted(edges, obs["left"], side="right") - 1, 0, F_BINS - 1)
    num = np.bincount(b, weights=obs["y"], minlength=F_BINS)
    den = np.bincount(b, weights=obs["mu"], minlength=F_BINS)
    f = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return edges, f, b


def _league_X(obs):
    p = obs["recv"] - 0.5
    return np.stack([p, p * obs["left"], obs["lead"] * obs["left"]], axis=1)


def residuals(obs):
    """Each observation's points still to come beyond its expected points x the league's share, and
    beyond the league's fit on possession and the lead (`e`); with the fit's coefficients."""
    _, f, b = share_left(obs)
    r = obs["y"] - obs["mu"] * f[b]
    X = _league_X(obs)
    coef = np.linalg.lstsq(X, r, rcond=None)[0] if len(r) else np.zeros(3)
    return r, r - X @ coef, coef


def _weights(n_matches, B, seed):
    """B resamples of the matches: (B, matches) how often each is drawn."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_matches, [1.0 / n_matches] * n_matches, size=B).astype(float)


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (float("nan"), float("nan"))


def _per_match(match, X, y):
    """(matches, k, k) and (matches, k): each match's X'X and X'y, the sums a least-squares fit over
    resampled matches is made of."""
    u, mi = np.unique(match, return_inverse=True)
    k = X.shape[1]
    XtX = np.zeros((len(u), k, k))
    Xty = np.zeros((len(u), k))
    np.add.at(XtX, mi, X[:, :, None] * X[:, None, :])
    np.add.at(Xty, mi, X * y[:, None])
    return XtX, Xty


def _solve(A, b):
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _fit_boot(match, X, y, B, seed, coef=1):
    """(the least-squares coefficients, the bootstrap draws of coefficient `coef` over matches)."""
    XtX, Xty = _per_match(match, X, y)
    b = _solve(XtX.sum(0), Xty.sum(0))
    if not B:
        return b, np.array([])
    W = _weights(len(XtX), B, seed)
    A = np.einsum("bm,mij->bij", W, XtX)
    v = W @ Xty
    draws = np.array([_solve(A[j], v[j])[coef] for j in range(B)])
    return b, draws


def league(obs, n_matches, B=BOOT, seed=0):
    """The league's lead slope (own points still to come per point of lead x share left, and the
    margin's, twice it) with its interval, and the receiver's margin still to come by its lead and half."""
    r, _, coef = residuals(obs)
    _, draws = _fit_boot(obs["match"], _league_X(obs), r, B, seed, coef=2)
    lo, hi = _ci(draws)
    recv = obs["recv"] == 1.0
    margin = r - r[np.arange(len(r)) ^ 1]                 # the side's residual less the other side's
    u, mi = np.unique(obs["match"], return_inverse=True)
    W = _weights(len(u), B, seed)
    bands = {}
    for lo_, hi_, name in LEAD_BANDS:
        for half, keep in (("1st half", obs["left"] > 0.5), ("2nd half", obs["left"] <= 0.5)):
            k = recv & keep & (obs["lead"] >= lo_) & (obs["lead"] <= hi_)
            if k.sum() < 20:
                continue
            tot = np.bincount(mi[k], weights=margin[k], minlength=len(u))
            cnt = np.bincount(mi[k], minlength=len(u)).astype(float)
            means = (W @ tot) / np.maximum(W @ cnt, 1e-9)
            bands[(name, half)] = (float(margin[k].mean()), *_ci(means), int(k.sum()))
    return dict(coef=coef, slope=float(coef[2]), lo=lo, hi=hi, bands=bands, possession=float(coef[0]))


# ---------------------------------------------------------------- each gamer

def gamer_slopes(obs, e, keep, B=BOOT, seed=0):
    """A gamer's slopes beyond the league on its observations `keep`: the margin still to come (own
    less the other side's) on the share left and on the lead x the share left -- the net slope, its
    interval over the gamer's matches, and the same for its own points (for) and the other side's
    (against), and with ahead and behind apart. B 0: the point estimates alone."""
    i = np.flatnonzero(keep)
    if len(i) < 3:
        return None
    own, other = e[i], e[i ^ 1]
    s, L = obs["left"][i], obs["lead"][i]
    X = np.stack([s, L * s], axis=1)
    X2 = np.stack([s, np.maximum(L, 0) * s, np.minimum(L, 0) * s], axis=1)
    m = obs["match"][i]
    b, draws = _fit_boot(m, X, own - other, B, seed)
    lo, hi = _ci(draws) if B else (float("nan"), float("nan"))
    split = _fit_boot(m, X2, own - other, 0, seed)[0]
    return dict(matches=len(np.unique(m)), states=len(i), slope=float(b[1]), lo=lo, hi=hi,
                se=float(np.std(draws)) if B else float("nan"), overall=float(b[0]),
                slope_for=float(_fit_boot(m, X, own, 0, seed)[0][1]),
                slope_against=float(_fit_boot(m, X, other, 0, seed)[0][1]),
                ahead=float(split[1]), behind=float(split[2]))


def tau2(est, se):
    """DerSimonian-Laird: the variance of the true values across gamers beyond their own noise."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    ok = np.isfinite(est) & np.isfinite(se) & (se > 0)
    est, se = est[ok], se[ok]
    if len(est) < 3:
        return 0.0, float("nan")
    w = 1.0 / se ** 2
    mean = float((w * est).sum() / w.sum())
    Q = float((w * (est - mean) ** 2).sum())
    den = float(w.sum() - (w ** 2).sum() / w.sum())
    return max(0.0, (Q - (len(est) - 1)) / den) if den > 0 else 0.0, mean


def shrunk(est, se, t2, mean):
    """Each gamer's value drawn toward the gamers' mean by its noise against the spread, and its sd."""
    est, se = np.asarray(est, float), np.asarray(se, float)
    if not t2:
        return np.full(len(est), mean), np.zeros(len(est))
    k = t2 / (t2 + se ** 2)
    return mean + k * (est - mean), np.sqrt(k * se ** 2)


def split_half(a, b, B=2000, seed=0):
    """The correlation across gamers of a value fitted on each half of their matches, with a 95%
    interval over gamers."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 5:
        return dict(n=len(a), r=float("nan"), lo=float("nan"), hi=float("nan"))
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        k = rng.integers(0, len(a), len(a))
        if np.std(a[k]) > 0 and np.std(b[k]) > 0:
            draws.append(np.corrcoef(a[k], b[k])[0, 1])
    lo, hi = _ci(draws)
    return dict(n=len(a), r=r, lo=lo, hi=hi)


def pace(snaps, gamer, keep=None, B=BOOT, seed=0):
    """(seconds a snap ahead, behind, ahead less behind, its interval over matches) on the snaps
    `keep` (all with None) of a gamer (every one with None). B 0: the point estimates alone."""
    k = np.ones(len(snaps["seconds"]), bool) if keep is None else keep
    if gamer is not None:
        k = k & (snaps["gamer"] == gamer)
    ahead, behind = k & (snaps["lead"] >= 1), k & (snaps["lead"] <= -1)
    if ahead.sum() < MIN_SNAPS or behind.sum() < MIN_SNAPS:
        return None
    sec = snaps["seconds"]
    A, Bh = float(sec[ahead].mean()), float(sec[behind].mean())
    lo = hi = se = float("nan")
    if B:
        i = np.flatnonzero(ahead | behind)
        u, mi = np.unique(snaps["match"][i], return_inverse=True)
        W = _weights(len(u), B, seed)
        sums = lambda sel: (W @ np.bincount(mi[sel], weights=sec[i][sel], minlength=len(u)),
                            W @ np.bincount(mi[sel], minlength=len(u)).astype(float))
        (sa, ca), (sb, cb) = sums(ahead[i]), sums(behind[i])
        d = sa / np.maximum(ca, 1e-9) - sb / np.maximum(cb, 1e-9)
        lo, hi = _ci(d)
        se = float(np.std(d))
    return dict(ahead=A, behind=Bh, diff=A - Bh, lo=lo, hi=hi, se=se, snaps=int(k.sum()))


# ---------------------------------------------------------------- the whole

def run(data, history, prior=None, since=None, until=None, min_matches=MIN_MATCHES, min_half=MIN_HALF, B=BOOT):
    """Everything the report shows, as a dict: read() then analyse()."""
    return analyse(read(data, history, prior, since, until), min_matches, min_half, B)


def analyse(got, min_matches=MIN_MATCHES, min_half=MIN_HALF, B=BOOT):
    """The league, each gamer and the trait tests off read()'s observations."""
    obs, snaps = got["obs"], got["snaps"]
    if not len(obs.get("y", [])):
        raise SystemExit("playstyle: no settled matches with gamers and PLAY_OVERs in the window")
    n_matches = len(got["codes"])
    lg = league(obs, n_matches, B)
    _, e, _ = residuals(obs)
    rows = []
    by_gamer = defaultdict(set)
    for m, g in zip(obs["match"], obs["gamer"]):
        by_gamer[int(g)].add(int(m))
    for g, ms in sorted(by_gamer.items(), key=lambda kv: got["gamers"][kv[0]]):
        if len(ms) < min_matches:
            continue
        mine = obs["gamer"] == g
        s = gamer_slopes(obs, e, mine, B, seed=g)
        if s is None:
            continue
        s["gamer"] = got["gamers"][g]
        halves = [gamer_slopes(obs, e, mine & (obs["half"] == h), 0) for h in (0, 1)]
        s["half_a"], s["half_b"] = [(h["slope"] if h and h["matches"] >= min_half else float("nan")) for h in halves]
        p = pace(snaps, g, B=B, seed=g)
        s["pace"] = p
        ph = [pace(snaps, g, snaps["half"] == h, 0) for h in (0, 1)]
        s["pace_a"], s["pace_b"] = [(x["diff"] if x else float("nan")) for x in ph]
        rows.append(s)
    lp = pace(snaps, None, B=B)
    for s in rows:                                      # pace beyond the league's own ahead less behind
        for k in ("pace_a", "pace_b"):
            s[k] = s[k] - lp["diff"] if lp else float("nan")
    t2, mean = tau2([s["slope"] for s in rows], [s["se"] for s in rows])
    sh, sd = shrunk([s["slope"] for s in rows], [s["se"] for s in rows], t2, mean)
    for s, a, b in zip(rows, sh, sd):
        s["shrunk"], s["shrunk_sd"] = float(a), float(b)
    trait = split_half([s["half_a"] for s in rows], [s["half_b"] for s in rows])
    pr = [s for s in rows if s["pace"]]
    pt2, pmean = tau2([s["pace"]["diff"] - lp["diff"] for s in pr], [s["pace"]["se"] for s in pr]) if lp and pr \
        else (0.0, float("nan"))
    psh, psd = shrunk([s["pace"]["diff"] - lp["diff"] for s in pr], [s["pace"]["se"] for s in pr], pt2, pmean) \
        if pr else ([], [])
    for s, a, b in zip(pr, psh, psd):
        s["pace"]["beyond"] = s["pace"]["diff"] - lp["diff"]
        s["pace"]["shrunk"], s["pace"]["shrunk_sd"] = float(a), float(b)
    pace_trait = split_half([s["pace_a"] for s in rows], [s["pace_b"] for s in rows])
    passes = lambda t: bool(np.isfinite(t["lo"]) and t["lo"] > 0)
    for s in rows:
        s["style"] = _group(s["shrunk"], s["shrunk_sd"], mean, "front-runner", "closes up") if passes(trait) else ""
        p = s["pace"]
        s["tempo"] = (_group(p["shrunk"], p["shrunk_sd"], pmean, "burns the clock ahead", "keeps the tempo ahead")
                      if p and passes(pace_trait) else "")
    return dict(league=lg, gamers=rows, tau=math.sqrt(t2), mean=mean, trait=trait, passes=passes(trait),
                pace=lp, pace_tau=math.sqrt(pt2), pace_mean=pmean, pace_trait=pace_trait, pace_passes=passes(pace_trait),
                matches=n_matches, states=len(obs["y"]) // 2, snaps=len(snaps.get("seconds", [])),
                since=got["since"], until=got["until"], expected=got["expected"], min_matches=min_matches,
                min_half=min_half)


def _group(value, sd, mean, above, below):
    """A group where the shrunk value's 80% interval leaves the gamers' mean, else steady."""
    if not sd or not np.isfinite(value):
        return "steady"
    z = (value - mean) / sd
    return above if z > 1.2816 else below if z < -1.2816 else "steady"


# ---------------------------------------------------------------- the report

def _n(v, p=3, width=0):
    """A signed number, or '-' where there is none."""
    t = f"{v:+.{p}f}" if v is not None and np.isfinite(v) else "-"
    return f"{t:>{width}s}" if width else t


def _iv(v, lo, hi, p=3):
    return f"{_n(v, p)} [{_n(lo, p)}, {_n(hi, p)}]"


def report(out):
    lg = out["league"]
    L = [f"  PLAYSTYLE: how each gamer plays by the score -- real games {out['since']} to {out['until']}, "
         f"{out['matches']:,} matches, {out['states']:,} kick-off states (the opening one and every one after a score"
         f" in regulation), {out['snaps']:,} second-half snaps",
         f"  each side's points still to come against its expected points ({out['expected']}) x the league's share of"
         " a game's points still to come; 95% intervals over matches", "",
         "  THE LEAGUE: the receiver's margin still to come beyond expected, by its lead (the side that conceded"
         " receives, so the lead is the receiver's after the other side scored)",
         f"    {'lead':12s} {'1st half':>30s} {'2nd half':>30s}"]
    for _, _, name in LEAD_BANDS:
        cells = []
        for half in ("1st half", "2nd half"):
            b = lg["bands"].get((name, half))
            cells.append(f"{_iv(b[0], b[1], b[2], 2)} ({b[3]:,})" if b else "-")
        L.append(f"    {name:12s} {cells[0]:>30s} {cells[1]:>30s}")
    L += [f"  the league's lead slope: a side's own points still to come per point of lead x the share of the game"
          f" left {_iv(lg['slope'], lg['lo'], lg['hi'])} (the margin's twice that); having the ball {lg['possession']:+.2f}"
          " points",
          "  (v2.1's sim plays real snaps for the situation, so the league's part is in it: v21-react-fit found its game"
          " script right, -0.028 [-0.070, +0.013])", ""]
    t = out["trait"]
    rows = sorted(out["gamers"], key=lambda s: -s["shrunk"])
    L += ["  EACH GAMER beyond the league: the margin still to come per point of lead x the share left (above 0 a"
          " front-runner: leads grow, deficits too; below 0 leads shrink and deficits close); the gamer's own over- or"
          " under-performance of its price fitted apart",
          f"    is it a trait? the slope fitted on alternate days apart, {t['n']} gamers with {out['min_half']}+ matches"
          f" in each half: the halves correlate r = {_iv(t['r'], t['lo'], t['hi'], 2)} (0: noise)",
          f"    the spread of the true slopes across gamers (tau) {out['tau']:.3f}; a gamer's own noise (median se)"
          f" {np.median([s['se'] for s in rows]) if rows else float('nan'):.3f}",
          ("    -> a trait: the gamers grouped below (80% sure off the gamers' mean, else steady)" if out["passes"] else
           "    -> not a trait: the gamers' slopes don't repeat from one half of their matches to the other, so no"
           " groups; the numbers are kept for the record"),
          "    a slope of +0.10: a 7-point lead at half-time grows 0.35 points more than the league's over the rest",
          f"    {'gamer':14s} {'matches':>7s} {'states':>7s} {'slope [95%]':>26s} {'for':>7s} {'against':>8s}"
          f" {'ahead':>7s} {'behind':>7s} {'shrunk':>7s} {'half A':>7s} {'half B':>7s}  group"]
    for s in rows:
        L.append(f"    {s['gamer'][:14]:14s} {s['matches']:7,d} {s['states']:7,d} {_iv(s['slope'], s['lo'], s['hi']):>26s}"
                 f" {_n(s['slope_for'], 3, 7)} {_n(s['slope_against'], 3, 8)} {_n(s['ahead'], 3, 7)}"
                 f" {_n(s['behind'], 3, 7)} {_n(s['shrunk'], 3, 7)} {_n(s['half_a'], 3, 7)} {_n(s['half_b'], 3, 7)}"
                 f"  {s['style']}")
    p, pt = out["pace"], out["pace_trait"]
    L += ["", "  PACE: seconds a snap took off the clock in the second half (its last two minutes left out), the side"
          " with the ball ahead against behind"]
    if p:
        L += [f"    the league: ahead {p['ahead']:.1f}s, behind {p['behind']:.1f}s, ahead less behind"
              f" {_iv(p['diff'], p['lo'], p['hi'], 2)}s",
              f"    each gamer's ahead less behind beyond the league's: the halves correlate r ="
              f" {_iv(pt['r'], pt['lo'], pt['hi'], 2)} ({pt['n']} gamers); tau {out['pace_tau']:.2f}s",
              ("    -> a trait: grouped below" if out["pace_passes"] else
               "    -> not a trait: no groups"),
              f"    {'gamer':14s} {'snaps':>7s} {'ahead':>7s} {'behind':>7s} {'beyond league [95%]':>26s} {'shrunk':>7s}"
              f" {'half A':>7s} {'half B':>7s}  group"]
        for s in sorted((s for s in rows if s["pace"]), key=lambda s: -s["pace"]["shrunk"]):
            q = s["pace"]
            L.append(f"    {s['gamer'][:14]:14s} {q['snaps']:7,d} {q['ahead']:7.1f} {q['behind']:7.1f}"
                     f" {_iv(q['beyond'], q['lo'] - p['diff'], q['hi'] - p['diff'], 2):>26s} {_n(q['shrunk'], 2, 7)}"
                     f" {_n(s['pace_a'], 2, 7)} {_n(s['pace_b'], 2, 7)}  {s['tempo']}")
    else:
        L.append("    too few snaps ahead and behind")
    return L


def write_csv(path, out):
    """One row per gamer: every number the report shows."""
    keys = ("gamer", "matches", "states", "slope", "lo", "hi", "se", "slope_for", "slope_against", "ahead", "behind",
            "overall", "shrunk", "shrunk_sd", "half_a", "half_b", "style")
    pkeys = ("snaps", "ahead", "behind", "diff", "beyond", "lo", "hi", "shrunk")
    rnd = lambda v: round(v, 5) if isinstance(v, float) and np.isfinite(v) else ("" if isinstance(v, float) else v)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(keys) + [f"pace_{k}" for k in pkeys] + ["pace_half_a", "pace_half_b", "tempo"])
        for s in out["gamers"]:
            p = s["pace"] or {}
            w.writerow([rnd(s[k]) for k in keys] + [rnd(p.get(k, float("nan"))) for k in pkeys]
                       + [rnd(s["pace_a"]), rnd(s["pace_b"]), s["tempo"]])
    return path


def write(out_dir, out):
    os.makedirs(out_dir, exist_ok=True)
    lines = report(out)
    with open(os.path.join(out_dir, "playstyle.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    write_csv(os.path.join(out_dir, "playstyle_gamers.csv"), out)
    return lines
