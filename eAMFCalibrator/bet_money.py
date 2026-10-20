"""Where the money is, as prices -- and what it knows.

Layer 1, the money's price. Every bet is money on a side: home or away on the moneyline and the spread,
over or under on the total. At any moment, the money toward the home (over) side is the stake on it less
the stake on the other, over both, -1 to +1, in play over the last WINDOW seconds and pre-match over every
pre-match bet before then, by customer temperature. The money's price is prod's, its log-odds moved by the
money:

    logit q = logit p_prod + sum over temperatures of beta x the money

beta fitted on the results by market, pre-match and in play apart, on the earlier half of the window's
matches, and scored on the later half: at every PLAY_OVER prod priced the market (pre-match, its last price
before the kick-off against every pre-match bet's money), each match counting the same, the log loss of the money's price less prod's; and
every later bet re-priced at it, the operator's margin kept (bets.reprice's way), the book's margin change.

Layer 2, what the money knows. On the bets: the money's tilt alone, then with each thing a punter could have
seen at that moment beside it -- prod's own move over the last minute, the score against the time left (the
game script), a score in the last 30 seconds, and v2.1's price against prod's (--sim-file, a bets_sim.csv
with a v21 candidate) -- and all of them at once. Where the tilt falls when a thing comes in, the money was
that thing; what is left is what it knows that none of them say. And what the money follows: the money
regressed on the same things.

And who starts a run (bet_runs.py): with a customer id on the bets, whether the other customers' money
follows a customer's bets, and which customers follow which.

Every interval is 95% over matches. Each bet sees only the money before it (bet time less the operator's
lag, on prod's clock).

    python -m eAMFCalibrator bets money --since 2026-10-01 --until 2026-10-19 --sim-file eAMFCalibrator/out_bets_gap/bets_sim.zip"""

import csv
import datetime as dt
import io
import os
import zipfile
from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np

from . import bet_runs, bets, config, snowflake_io
from .bet_totals import kickoff_time, message_at

PLUS = {"moneyline": 50, "spread": 52, "total": 54}
GROUP = {50: "moneyline", 51: "moneyline", 52: "spread", 53: "spread", 54: "total", 55: "total"}
SIGN = {50: 1, 51: -1, 52: 1, 53: -1, 54: 1, 55: -1}
SEGMENTS = ("Restricted", "VIP", "Standard", "none")
WINDOW = 120.0
MOMENTUM = 60.0
RECENT = 30.0
BOOT = 300
P_RANGE = (0.03, 0.97)
QUARTER = 240.0


def _segment(b):
    t = str(b.extra.get(config.BET_VIP_COLUMN) or "").strip()
    return t if t in SEGMENTS else "none"


def _ts(t):
    return None if t is None else t.timestamp()


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------- the money

class Money:
    """Every bet as money on the home (over) side of its market: its time on prod's clock, the signed stake
    and the stake, by temperature, pre-match and in play apart."""

    def __init__(self, all_bets, lags, window=WINDOW):
        self.window = float(window)
        by = defaultdict(list)
        for b in all_bets:
            g = GROUP.get(b.feed_market)
            if g is None or b.time is None or not b.stake:
                continue
            seen = b.time - dt.timedelta(seconds=bets.bet_lag(b, lags))
            by[(b.match_code, g, bool(b.in_play))].append((_ts(seen), SIGN[b.feed_market] * b.stake, b.stake, _segment(b)))
        self.cells = {}
        for key, rows in by.items():
            rows.sort(key=lambda r: r[0])
            t = np.array([r[0] for r in rows])
            seg = np.array([r[3] for r in rows])
            cum = {}
            for s in ("all",) + SEGMENTS:
                k = np.ones(len(rows), bool) if s == "all" else seg == s
                cum[s] = (np.r_[0.0, np.cumsum(np.where(k, [r[1] for r in rows], 0.0))],
                          np.r_[0.0, np.cumsum(np.where(k, [r[2] for r in rows], 0.0))])
            self.cells[key] = (t, cum)

    def at(self, match, group, t, in_play, window=None):
        """{temperature and 'all': the money toward the home (over) side before t} and the stake behind it:
        in play the last `window` seconds (the money's own with None), pre-match every pre-match bet before t."""
        window = self.window if window is None else window
        cell = self.cells.get((match, group, bool(in_play)))
        out = {s: 0.0 for s in ("all",) + SEGMENTS}
        if cell is None or t is None:
            return out, 0.0
        times, cum = cell
        hi = bisect_left(times, t)
        lo = bisect_left(times, t - window) if in_play else 0
        stake = 0.0
        for s, (sig, tot) in cum.items():
            den = tot[hi] - tot[lo]
            out[s] = (sig[hi] - sig[lo]) / den if den > 0 else 0.0
            if s == "all":
                stake = den
        return out, stake


# ---------------------------------------------------------------- the state at a moment

def _outcome(group, line, final):
    if final is None or None in final:
        return None
    h, a = final
    if group == "moneyline":
        return None if h == a else float(h > a)
    if line is None:
        return None
    x = (h - a) if group == "spread" else (h + a)
    return None if x == line else float(x > line)


def state_at(checks, match, t):
    """(home score, away score, share of regulation left) at prod's time t (a datetime), or Nones."""
    if checks is None or t is None:
        return None, None, None
    m = message_at(checks, match, t)
    h, a = checks.score_at(match, m) if m is not None else (0, 0)
    tl = (getattr(checks, "timelines", None) or {}).get(match)
    left = None
    if tl is not None:
        j = bisect_right(tl.times, t.timestamp()) - 1
        if 0 <= j < len(tl.period) and tl.period[j] is not None and tl.clock[j] is not None:
            period, clock = int(tl.period[j]), float(tl.clock[j])
            left = 0.0 if period >= 5 else max(0.0, ((4 - min(period, 4)) * QUARTER + clock) / (4 * QUARTER))
    return h, a, left


def since_score(checks, match, t):
    """Seconds from the latest score before prod's time t, or None (no score yet)."""
    entry = None if checks is None else checks.scores.get(match)
    m = None if checks is None else message_at(checks, match, t)
    if not entry or m is None:
        return None
    msgs, vals = entry
    i = bisect_right(msgs, m) - 1
    while i > 0 and vals[i] == vals[i - 1]:
        i -= 1
    if i < 0 or (i == 0 and vals[0] == (0, 0)):
        return None
    ts = checks.time_of(match, msgs[i])
    return None if ts is None else (t - ts).total_seconds()


def features(tl, checks, match, group, t, p, line, in_play):
    """What a punter could have seen at t: prod's move on the home (over) side over the last MOMENTUM seconds
    (log-odds), the game script (home's lead, or the points the total still needs, x the share left), and a
    score in the last RECENT seconds."""
    q = bets.price_at_time(tl, match, PLUS[group], t - dt.timedelta(seconds=MOMENTUM))
    momentum = float(_logit(p) - _logit(q[1])) if q is not None and q[1] is not None and in_play else 0.0
    h, a, left = state_at(checks, match, t) if in_play else (0, 0, 1.0)
    script = 0.0
    if in_play and h is not None and left is not None:
        script = ((h - a) if group != "total" else ((line - h - a) if line is not None else 0.0)) * left
    s = since_score(checks, match, t) if in_play else None
    return dict(momentum=momentum, script=script, recent=float(s is not None and s <= RECENT))


# ---------------------------------------------------------------- the points

def moment_points(tl, money, checks, finals, kickoffs):
    """Every PLAY_OVER prod priced the home (over) side of each market, live, and each match's last
    pre-match price, against the money of all its pre-match bets: (match, group, in play, t, p, line,
    outcome, the money, its stake)."""
    out = []
    for (match, market), (times, quotes) in tl.items():
        group = GROUP.get(market)
        if group is None or PLUS[group] != market:
            continue
        final = finals.get(match)
        ko = kickoffs.get(match)
        last_pre, seen_msg = None, {}
        for t, (msg, p, line, live) in zip(times, quotes):
            if not live or p is None or not P_RANGE[0] <= p <= P_RANGE[1]:
                continue
            if msg is None or (ko is not None and t < ko):
                last_pre = (t, p, line)
            else:
                seen_msg[msg] = (t, p, line)                       # the latest row of each message
        if last_pre is not None:
            y = _outcome(group, last_pre[2], final)
            if y is not None:
                # prod's pre-match price is posted once, before most of the betting, and stands till the
                # kick-off: against it, the money of every pre-match bet
                f, stake = money.at(match, group, float("inf"), False)
                out.append(dict(match=match, group=group, in_play=False, t=last_pre[0], p=last_pre[1], line=last_pre[2],
                                y=y, money=f, stake=stake))
        for msg, (t, p, line) in seen_msg.items():
            y = _outcome(group, line, final)
            if y is None:
                continue
            f, stake = money.at(match, group, _ts(t), True)
            out.append(dict(match=match, group=group, in_play=True, t=t, p=p, line=line, y=y, money=f, stake=stake,
                            **features(tl, checks, match, group, t, p, line, True)))
    return out


def bet_points(ml_bets, joined, tl, money, checks, finals, sim=None):
    """Every settled bet: prod's price of the home (over) side and its line at the moment the bet was priced,
    the money before it, what a punter could have seen, v2.1's price (sim: {bet id: (prob of the bet's side,
    line)}) and the bet itself (its side, prod's price of it, the odds, stake, result and revenue)."""
    out = []
    for b, r in zip(ml_bets, joined):
        group = GROUP.get(b.feed_market)
        if group is None or r.get("result") not in (bets.WON, bets.LOST, bets.PUSH) or r.get("excluded"):
            continue
        seen = None if b.time is None else b.time - dt.timedelta(seconds=r.get("latency_seconds") or 0)
        q = bets.price_at_time(tl, b.match_code, PLUS[group], seen)
        if q is None or q[1] is None or not P_RANGE[0] <= q[1] <= P_RANGE[1] or r.get("stream_prob") is None:
            continue
        y = _outcome(group, q[2], finals.get(b.match_code))
        if y is None:
            continue
        f, stake = money.at(b.match_code, group, _ts(seen), b.in_play)
        row = dict(match=b.match_code, group=group, in_play=bool(b.in_play), t=seen, p=q[1], line=q[2], y=y, money=f,
                   stake=stake, bet_id=b.bet_id, sign=SIGN[b.feed_market], p_sel=r["stream_prob"], odds=r["odds"],
                   bet_stake=b.stake, result=r["result"], revenue=r["revenue"], temperature=_segment(b),
                   on_line=bool(r.get("on_prod_line")),
                   **features(tl, checks, b.match_code, group, seen, q[1], q[2], b.in_play))
        v = (sim or {}).get(str(b.bet_id))
        if v is not None and v[0] is not None and P_RANGE[0] <= v[0] <= P_RANGE[1]:
            same = group == "moneyline" or (v[1] is not None and r.get("stream_line") is not None
                                            and abs(v[1] - r["stream_line"]) < 1e-6)
            if same:
                vp = v[0] if row["sign"] > 0 else 1 - v[0]
                row["v21"] = float(_logit(vp) - _logit(q[1]))
        out.append(row)
    return out


def read_sim(path, name=None):
    """{bet id: (the candidate's chance of the bet's side, its line)} off a bets_sim.csv (or a .zip of one):
    the candidate `name`, else the first v21 one."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist() if n.endswith(".csv"))
            fh = io.TextIOWrapper(z.open(member), encoding="utf-8")
            return _sim_rows(csv.DictReader(fh), name)
    with open(path, newline="", encoding="utf-8") as fh:
        return _sim_rows(csv.DictReader(fh), name)


def _sim_rows(reader, name):
    cols = reader.fieldnames or []
    probs = [c[len("candidate_prob_"):] for c in cols if c.startswith("candidate_prob_")]
    pick = name if name else next((c for c in probs if c.startswith("v21")), probs[0] if probs else None)
    if pick is None:
        raise SystemExit("--sim-file: no candidate_prob_ column (a bets run's bets_sim.csv)")
    f = lambda v: float(v) if v not in ("", None) else None
    return {r["bet_id"]: (f(r.get(f"candidate_prob_{pick}")), f(r.get(f"candidate_line_{pick}"))) for r in reader}


# ---------------------------------------------------------------- the fits

def fit_logit(X, y, offset, w, ridge=1e-4, iters=12):
    """Logistic regression with an offset (prod's log-odds) and weights: the coefficients."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        z = offset + X @ b
        p = 1 / (1 + np.exp(-z))
        H = (X * (w * p * (1 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        b = b + np.linalg.solve(H, X.T @ (w * (y - p)) - ridge * b)
    return b


def _weights(match_ix, n):
    """Each point 1 / its match's points: every match counting the same."""
    c = np.bincount(match_ix, minlength=n).astype(float)
    return 1.0 / c[match_ix]


def _boot_fit(X, y, off, mi, n, B, seed=0):
    rng = np.random.default_rng(seed)
    base = _weights(mi, n)
    draws = []
    for _ in range(B):
        k = rng.multinomial(n, [1.0 / n] * n).astype(float)
        draws.append(fit_logit(X, y, off, base * k[mi]))
    return np.array(draws)


def _ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (float("nan"), float("nan"))


def _ll(y, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def layer1(points, bets_pts, split, B=BOOT):
    """By market, pre-match and in play: the money's tilts fitted on the matches before `split` (a day),
    scored on the rest -- the log loss of its price less prod's at every point, and every later bet re-priced
    at it."""
    out = []
    for group in PLUS:
        for in_play in (False, True):
            pts = [x for x in points if x["group"] == group and x["in_play"] == in_play]
            if len(pts) < 100:
                continue
            match = np.array([x["match"] for x in pts])
            day = np.array([x["t"].date() for x in pts])
            train, test = day < split, day >= split
            if train.sum() < 50 or test.sum() < 50:
                continue
            y = np.array([x["y"] for x in pts])
            off = _logit([x["p"] for x in pts])
            for label, cols in (("all money", ("all",)), ("by temperature", SEGMENTS)):
                X = np.array([[x["money"][c] for c in cols] for x in pts])
                u, mi = np.unique(match[train], return_inverse=True)
                b = fit_logit(X[train], y[train], off[train], _weights(mi, len(u)))
                draws = _boot_fit(X[train], y[train], off[train], mi, len(u), B)
                ut, ti = np.unique(match[test], return_inverse=True)
                wt = _weights(ti, len(ut))
                q = 1 / (1 + np.exp(-(off[test] + X[test] @ b)))
                d = _ll(y[test], q) - _ll(y[test], 1 / (1 + np.exp(-off[test])))
                per = np.bincount(ti, weights=wt * d, minlength=len(ut))
                rng = np.random.default_rng(1)
                W = rng.multinomial(len(ut), [1.0 / len(ut)] * len(ut), size=1000).astype(float)
                lo, hi = _ci(W @ per / len(ut))
                row = dict(group=group, in_play=in_play, label=label, cols=cols, beta=b,
                           beta_ci=[_ci(draws[:, j]) for j in range(len(cols))], train=int(train.sum()),
                           test=int(test.sum()), train_matches=len(u), test_matches=len(ut),
                           with_money=int((X[train] != 0).any(axis=1).sum()),
                           ll=float(per.sum() / len(ut)), ll_ci=(lo, hi))
                row.update(reprice(bets_pts, group, in_play, cols, b, split))
                out.append(row)
    return out


def reprice(bets_pts, group, in_play, cols, beta, split):
    """Every bet of the later matches re-priced at the money's price (prod's price of its side moved by the
    money's tilt, the operator's margin kept): the book's margin as bet and re-priced, and the change with its
    interval over matches."""
    rows = [x for x in bets_pts if x["group"] == group and x["in_play"] == in_play and x["t"].date() >= split
            and (group == "moneyline" or x["on_line"])]
    if len(rows) < 30:
        return dict(bets=len(rows))
    stake = np.array([x["bet_stake"] for x in rows])
    rev = np.array([x["revenue"] for x in rows])
    shift = np.array([x["sign"] * sum(beta[j] * x["money"][c] for j, c in enumerate(cols)) for x in rows])
    p = np.clip(np.array([x["p_sel"] for x in rows]), 1e-4, 1 - 1e-4)
    q = 1 / (1 + np.exp(-(_logit(p) + shift)))
    odds_c = np.array([x["odds"] for x in rows]) * p / q
    res = np.array([x["result"] for x in rows])
    rev_c = np.where(res == bets.WON, -stake * (odds_c - 1), np.where(res == bets.LOST, stake, 0.0))
    u, mi = np.unique([x["match"] for x in rows], return_inverse=True)
    s, r0, r1 = (np.bincount(mi, weights=v, minlength=len(u)) for v in (stake, rev, rev_c))
    W = np.random.default_rng(2).multinomial(len(u), [1.0 / len(u)] * len(u), size=1000).astype(float)
    ch = 100 * (W @ r1 - W @ r0) / (W @ s)
    lo, hi = _ci(ch)
    return dict(bets=len(rows), stake=float(stake.sum()), as_bet=float(100 * rev.sum() / stake.sum()),
                at_money=float(100 * rev_c.sum() / stake.sum()), change=float(100 * (rev_c.sum() - rev.sum()) / stake.sum()),
                change_ci=(lo, hi))


LAYER2 = ("momentum", "script", "recent", "v21")
NAMES = {"momentum": "prod's move, the last minute", "script": "the game script (lead x time left)",
         "recent": "a score in the last 30s", "v21": "v2.1's price against prod's"}


def layer2(bets_pts, B=BOOT):
    """On the bets, by market, pre-match and in play: the money's tilt alone, beside each thing a punter could
    have seen, and beside all of them; and the money regressed on the same things."""
    out = []
    for group in PLUS:
        for in_play in (False, True):
            rows = [x for x in bets_pts if x["group"] == group and x["in_play"] == in_play]
            have = [f for f in LAYER2 if (f != "v21" or sum("v21" in x for x in rows) >= 0.5 * len(rows))
                    and (in_play or f in ("v21",))]
            if have and "v21" in have:
                rows = [x for x in rows if "v21" in x]
            if len(rows) < 200:
                continue
            y = np.array([x["y"] for x in rows])
            off = _logit([x["p"] for x in rows])
            F = np.array([x["money"]["all"] for x in rows])
            u, mi = np.unique([x["match"] for x in rows], return_inverse=True)
            w = _weights(mi, len(u))
            cols = {f: np.array([x.get(f, 0.0) for x in rows], float) for f in have}
            fits = []
            for label, extra in [("alone", [])] + [(f, [f]) for f in have] + ([("all of them", list(have))] if len(have) > 1 else []):
                X = np.c_[F, *[cols[f] for f in extra]] if extra else F[:, None]
                b = fit_logit(X, y, off, w)
                d = _boot_fit(X, y, off, mi, len(u), max(50, B // 3))
                fits.append(dict(label=label, extra=extra, beta=float(b[0]), ci=_ci(d[:, 0]),
                                 own={f: (float(b[1 + j]), _ci(d[:, 1 + j])) for j, f in enumerate(extra)}))
            # what the money follows: the money on the things, bets with money before them
            k = F != 0
            follows = None
            if k.sum() >= 100 and have:
                X = np.c_[np.ones(k.sum()), off[k], *[cols[f][k] for f in have]]
                coef, *_ = np.linalg.lstsq(X, F[k], rcond=None)
                resid = F[k] - X @ coef
                r2 = 1 - resid.var() / F[k].var() if F[k].var() > 0 else float("nan")
                sd = {f: float(cols[f][k].std()) for f in have}
                follows = dict(n=int(k.sum()), r2=float(r2), coef={f: float(coef[2 + j]) for j, f in enumerate(have)},
                               prod=float(coef[1]), sd=sd)
            out.append(dict(group=group, in_play=in_play, bets=len(rows), matches=len(u), have=have, fits=fits,
                            follows=follows))
    return out


# ---------------------------------------------------------------- the report

def _iv(v, ci, p=3):
    return f"{v:+.{p}f} [{ci[0]:+.{p}f}, {ci[1]:+.{p}f}]"


def report(l1, l2, split, window):
    L = [f"  WHERE THE MONEY IS, AS PRICES -- the money toward the home (over) side: stake on it less on the other, over"
         f" both, -1 to +1; in play over the last {window:g}s, pre-match every pre-match bet before; each bet sees only"
         " the money before it; 95% intervals over matches", "",
         f"  LAYER 1, THE MONEY'S PRICE: prod's log-odds moved by beta x the money, fitted on the matches before {split},"
         " scored on the rest (each match counting the same)",
         f"    {'market':10s} {'':9s} {'model':15s} {'money':11s} {'beta [95%]':>26s} {'fitted on':>18s} {'points':>8s}"
         f" {'log loss less prod [95%]':>30s}"
         f" {'bets':>7s} {'as bet':>7s} {'at money':>9s} {'change [95%]':>24s}"]
    for r in l1:
        for j, c in enumerate(r["cols"]):
            first = j == 0
            ph = "in play" if r["in_play"] else "pre-match"
            name = "all" if c == "all" else c
            tail = ""
            if first:
                fitted = f"{r['with_money']:,} of {r['train']:,}"
                tail = (f" {fitted:>18s} {r['test']:8,d}"
                        f" {_iv(r['ll'] * 1000, (r['ll_ci'][0] * 1000, r['ll_ci'][1] * 1000), 2):>30s}")
                if r.get("stake"):
                    tail += (f" {r['bets']:7,d} {r['as_bet']:+6.2f}% {r['at_money']:+8.2f}%"
                             f" {_iv(r['change'], r['change_ci'], 2):>24s}")
            model = ("one tilt" if r["label"] == "all money" else "by temperature") if first else ""
            L.append(f"    {(r['group'] if first and r['label'] == 'all money' else ''):10s}"
                     f" {(ph if first and r['label'] == 'all money' else ''):9s} {model:15s} {name:11s}"
                     f" {_iv(r['beta'][j], r['beta_ci'][j]):>26s}{tail}")
    L += ["    beta: how far a side's log-odds move with all the money on it (at an even price +0.40 is about 10 points of"
          " chance); fitted on: the earlier points with some money, of all of them; points: the later ones it is scored"
          " on; log loss less prod in thousandths, below 0 the money's price the better; change: the book's margin"
          " re-priced at the money's price less as bet, in points", ""]
    L += ["  LAYER 2, WHAT THE MONEY KNOWS: the money's tilt on the bets alone, beside each thing a punter could have seen"
          " then, and beside all of them -- where it falls, the money was that thing; what is left, it knows that none of"
          " them say"]
    for r in l2:
        ph = "in play" if r["in_play"] else "pre-match"
        L.append(f"    {r['group']} {ph}: {r['bets']:,} bets on {r['matches']:,} matches")
        for f in r["fits"]:
            own = "; ".join(f"{NAMES[k]} {_iv(v[0], v[1])}" for k, v in f["own"].items())
            L.append(f"      {'beside ' + f['label'] if f['extra'] else 'alone':40s} the money {_iv(f['beta'], f['ci'])}"
                     + (f"   ({own})" if own else ""))
        fo = r["follows"]
        if fo:
            L.append(f"      what the money follows ({fo['n']:,} bets with money before them, R2 {fo['r2']:.3f}): one sd more of "
                     + ", ".join(f"{NAMES[k]} {fo['coef'][k] * fo['sd'][k]:+.3f}" for k in fo["coef"])
                     + f" of money toward the side; prod's log-odds {fo['prod']:+.3f} a unit")
    return L


def _customer_column(cur, column):
    """The customer id column if the bet table has it, and a line on it if not."""
    try:
        cols, _ = bets.fetch_all(cur, f"SELECT * FROM {bets.bet_table()} LIMIT 0")
    except (Exception, SystemExit) as exc:                      # noqa: BLE001 -- the probe is an extra
        return None, f"    (reading {bets.bet_table()}'s columns failed: {exc}; no customer probe)"
    found, like = bet_runs.find_column(cols, column)
    if found is None:
        return None, (f"    {bets.bet_table()} has no {column} (--customer-column); the columns that look like a customer's: "
                      + (", ".join(like) or "none"))
    return found, None


def run(cur, out_dir, sim_file=None, sim_candidate=None, window=WINDOW, split=None, customer_column=None,
        run_after=bet_runs.AFTER):
    """Fetch every bet of the window and prod's quotes, build the money, fit both layers and the customer
    probe; print, and write the text and the bets' points."""
    column, note = _customer_column(cur, customer_column or config.BET_CUSTOMER_COLUMN)
    extras = config.BET_EXTRA_COLUMNS
    if column is not None and column not in extras:
        config.BET_EXTRA_COLUMNS = extras + [column]
    try:
        sql, params, _ = bets.bets_sql()
        cols, raw = bets.fetch_all(cur, sql, tuple(params))
    finally:
        config.BET_EXTRA_COLUMNS = extras
    every = [b for b in bets.to_bets(cols, raw) if b.feed_market in GROUP]
    matches = sorted({b.match_code for b in every})
    print(f"  {len(every):,} bets on {len(matches):,} matches ({sum(b.in_play for b in every):,} in play)")
    if not every:
        return None
    prod_rows, finals = [], {}
    for start in range(0, len(matches), config.MATCH_CHUNK_SIZE):
        batch = matches[start:start + config.MATCH_CHUNK_SIZE]
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        finals.update(snowflake_io.fetch_final_scores(cur, batch))
    checks = bets.fetch_checks(cur, matches, prod_rows)
    tl = bets.timeline(prod_rows)
    del prod_rows
    lags = bets.fit_lags(every, tl)
    signs = bets.fit_line_signs(every, tl, lags)
    joined = bets.join(every, lags, signs, tl, {}, None, finals, checks)
    money = Money(every, lags, window)
    sim = read_sim(sim_file, sim_candidate) if sim_file else None
    if sim is not None:
        print(f"  {sim_file}: v2.1's price for {sum(v[0] is not None for v in sim.values()):,} bets")
    kickoffs = {m: kickoff_time(checks, m) for m in matches}
    points = moment_points(tl, money, checks, finals, kickoffs)
    bpts = bet_points(every, joined, tl, money, checks, finals, sim)
    days = sorted({x["t"].date() for x in points})
    split = split or (days[len(days) // 2] if days else None)
    print(f"  {len(points):,} priced moments, {len(bpts):,} bets; fitted before {split}, scored from it", flush=True)
    l1 = layer1(points, bpts, split)
    l2 = layer2(bpts)
    lines = report(l1, l2, split, window)
    runs = None
    if column is not None:
        runs = bet_runs.analyse(every, joined, lags, column, checks, after=run_after)
        lines += [""] + bet_runs.report(runs)
    else:
        lines += ["", "  WHO STARTS A RUN, by customer id", note]
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    path_txt = os.path.join(out_dir, "bets_money.txt")
    bets.write_text(path_txt, lines)
    path_csv = os.path.join(out_dir, "bets_money_points.csv")
    flat = [dict({k: v for k, v in x.items() if k != "money"}, **{f"money_{s}": v for s, v in x["money"].items()})
            for x in bpts]
    bets.write_csv(path_csv, flat, list(dict.fromkeys(k for r in flat for k in r)))
    if runs is not None and runs.get("customers") is not None:
        rows = bet_runs.customer_rows(runs)
        bets.write_csv(os.path.join(out_dir, "bets_money_customers.csv"), rows)
        bets.write_csv(os.path.join(out_dir, "bets_money_pairs.csv"), runs["pairs"],
                       ["leader", "follower", "followed", "together", "back"])
        print(f"  -> {out_dir}/bets_money_customers.csv ({len(rows):,} customers) and bets_money_pairs.csv"
              f" ({len(runs['pairs']):,} pairs)")
    return path_txt, path_csv
