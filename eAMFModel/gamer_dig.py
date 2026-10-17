"""gamer-dig: what the money backing particular gamers sees in their play -- the backed gamer's and
the opponent's, in the match before the bet and in the matches before it.

Every match's snaps (v21.match_snaps) give each side a play profile (PROFILE): yards a snap, the
share of explosive gains (15+), of stuffed snaps (0 or less), of first downs, the 3rd downs made,
turnovers, touchdowns and the seconds a snap. Each gamer's baseline is the profile of their matches in
the BASELINE_DAYS before the window. Then:

  targeted     the gamers the money wins on (their backers' edge over prod's price, shrunk by
               SHRINK_STAKE of stake), or those named: their record against prod's pre-match price,
               their baseline profile against the league's, their play in the window against their own
  earlier      does the play of a gamer's earlier matches in the session (against their baseline)
               say anything of their result beyond prod's pre-match price? every match with prod's
               pre-match moneyline in the export, a logistic fit a feature at a time
  this match   for every in-play moneyline bet, the backed gamer's and the opponent's play so far in
               the match (against their baselines): which tells the bet's result beyond prod's price
               at the bet, with match-clustered intervals
  the snap     the snap the bet followed (the last one known at prod's message): touchdown, turnover,
  before       explosive, first down, stuffed, other -- how often, and the backers' edge after each
  hot streak   the same for bets backing a gamer on a run of HOT wins this session

    python -m eAMFModel gamer-dig --bets eAMFCalibrator/out_bets_react/bets_sim.csv

Nothing here is priced: it reads the export, the history and a betting simulation's bets_sim.csv."""

import csv
import datetime as dt
import math
import os
from collections import defaultdict

import numpy as np

from . import v21
from .remaining import day

PROFILE = ("ypp", "explosive", "stuffed", "first", "third", "turnover", "touchdown", "pace")
EXPLOSIVE = 15
BASELINE_DAYS = 30
SHRINK_STAKE = 50000.0
TOP = 10
HOT = 3
SESSION_GAP = 240


def _f(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def profile(recs):
    """{feature: value} of a side's snaps, and the snaps (None where it had none)."""
    if not recs:
        return None
    kind = np.array([r["kind"] for r in recs])
    gain = np.array([r["gain"] for r in recs], float)
    field = np.array([r["field"] for r in recs], float)
    is_gain = kind == v21.GAIN
    td = is_gain & (gain >= 100 - field)
    conv = np.array([bool(r.get("converted")) and not r.get("replay") for r in recs]) & is_gain
    down = np.array([r["down"] for r in recs])
    third = down == 3
    secs = np.array([r.get("play_seconds", r.get("seconds", 0.0)) for r in recs], float)
    g = gain[is_gain]
    return dict(n=len(recs), ypp=float(g.mean()) if len(g) else 0.0,
                explosive=float(np.mean(is_gain & (gain >= EXPLOSIVE))), stuffed=float(np.mean(is_gain & (gain <= 0))),
                first=float(conv.mean()), third=float(conv[third].mean()) if third.any() else float("nan"),
                turnover=float(np.mean(~is_gain)), touchdown=float(td.mean()), pace=float(secs.mean()))


def side_snaps(rows, until=None, snaps=None):
    """{side: [snaps]} of a match's real snaps, those known by message `until` (all without); snaps: the
    match's v21.match_snaps already read."""
    out = {0: [], 1: []}
    for r in (v21.match_snaps(rows) if snaps is None else snaps):
        if r["period"] > 4 + v21.MAX_OT or (until is not None and r["message"] > until):
            continue
        out[r["side"]].append(r)
    return out


def _start(h):
    try:
        return dt.datetime.fromisoformat(str(h.get("SCHEDULED_START_TIME_UTC"))[:19])
    except ValueError:
        return None


def gamer_matches(history):
    """{gamer: [(start, match, side)]} in start order."""
    by = defaultdict(list)
    for h in history:
        t = _start(h)
        if t is None:
            continue
        for side, col in ((0, "PLAYER_1_HANDLE"), (1, "PLAYER_2_HANDLE")):
            g = (h.get(col) or "").strip().upper()
            if g:
                by[g].append((t, h["MATCH_CODE"], side))
    for v in by.values():
        v.sort()
    return by


def session_before(gm, gamer, code):
    """The gamer's earlier matches of the session ([(match, side)], latest first) and its run of wins
    or losses into this one (+3: won the last three; the history's finals)."""
    ms = gm.get(gamer, [])
    k = next((i for i, (_, c, _) in enumerate(ms) if c == code), None)
    if k is None:
        return []
    out = []
    for i in range(k - 1, -1, -1):
        if (ms[i + 1][0] - ms[i][0]).total_seconds() > SESSION_GAP * 60:
            break
        out.append((ms[i][1], ms[i][2]))
    return out


def run_of(results, earlier):
    """+n: won the n before in a row, -n: lost them (results {match: (home, away)})."""
    n = 0
    for code, side in earlier:
        fin = results.get(code)
        if fin is None or fin[0] == fin[1]:
            break
        won = (fin[0] > fin[1]) == (side == 0)
        if n == 0:
            n = 1 if won else -1
        elif (n > 0) == won:
            n += 1 if won else -1
        else:
            break
    return n


def _logit(p):
    p = min(0.98, max(0.02, float(p)))
    return math.log(p / (1 - p))


def logistic(X, y, groups=None, iters=60):
    """(coefficients, standard errors -- match-clustered where groups are given)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-(X @ b)))
    Hi = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(X.shape[1]))
    if groups is None:
        return b, np.sqrt(np.diag(Hi))
    sc = X * (y - p)[:, None]
    G = defaultdict(lambda: np.zeros(X.shape[1]))
    for gi, s in zip(groups, sc):
        G[gi] += s
    meat = sum(np.outer(v, v) for v in G.values())
    return b, np.sqrt(np.diag(Hi @ meat @ Hi))


def read_bets(path):
    """The moneyline bets of a betting simulation's bets_sim.csv, settled won or lost."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("market") != "moneyline" or r.get("result") not in ("won", "lost"):
                continue
            p, stake = _f(r.get("stream_prob")), _f(r.get("stake"))
            if p is None or not stake:
                continue
            out.append(dict(match=r["match_code"], home=r.get("selection") == "Home", in_play=r.get("in_play") == "True",
                            stake=stake, revenue=_f(r.get("revenue")) or 0.0, won=r["result"] == "won", p=p,
                            message=int(float(r["message"])) if _f(r.get("message")) is not None else None,
                            time=r.get("bet_time", "")))
    return out


class Dig:
    """The window's matches and bets, with every side's profile and every gamer's baseline."""

    def __init__(self, data, history, bets=None, since=None, until=None):
        self.data, self.bets = data, bets or []
        self.hist = {h["MATCH_CODE"]: h for h in history}
        self.gm = gamer_matches(history)
        self.results = {c: v21.match_final(h) for c, h in self.hist.items() if v21.match_final(h) is not None}
        if since is None:
            since = min((day(b["match"]) for b in self.bets), default=None) or min(day(c) for c in data)
        self.since = since
        self.until = until or (max((day(b["match"]) for b in self.bets), default=None) or max(day(c) for c in data))
        self._prof = {}
        self._snaps = {}

    def snaps(self, code):
        if code not in self._snaps:
            rows = self.data.get(code)
            self._snaps[code] = v21.match_snaps(rows) if rows else []
        return self._snaps[code]

    def gamer(self, code, side):
        h = self.hist.get(code) or {}
        return (h.get("PLAYER_1_HANDLE" if side == 0 else "PLAYER_2_HANDLE") or "").strip().upper()

    def match_profile(self, code, until=None):
        key = (code, until)
        if key not in self._prof:
            rows = self.data.get(code)
            self._prof[key] = {} if not rows else {s: profile(v) for s, v in
                                                   side_snaps(rows, until, self.snaps(code)).items()}
        return self._prof[key]

    def baseline(self, gamer):
        """The gamer's profile over the BASELINE_DAYS before the window (snap-weighted), or None."""
        lo = self.since - dt.timedelta(days=BASELINE_DAYS)
        acc, n = defaultdict(float), 0
        for t, code, side in self.gm.get(gamer, []):
            if not lo <= t.date() < self.since or code not in self.data:
                continue
            pr = self.match_profile(code).get(side)
            if not pr:
                continue
            for f in PROFILE:
                if not math.isnan(pr[f]):
                    acc[f] += pr[f] * pr["n"]
            n += pr["n"]
        return {f: acc[f] / n for f in PROFILE} | {"n": n} if n else None

    def league(self):
        """The league's profile over the same days."""
        acc, n = defaultdict(float), 0
        lo = self.since - dt.timedelta(days=BASELINE_DAYS)
        for code in self.data:
            if not lo <= day(code) < self.since:
                continue
            for pr in self.match_profile(code).values():
                if pr:
                    for f in PROFILE:
                        if not math.isnan(pr[f]):
                            acc[f] += pr[f] * pr["n"]
                    n += pr["n"]
        return {f: acc[f] / n for f in PROFILE} if n else {f: float("nan") for f in PROFILE}

    def prematch(self, code):
        """Prod's pre-match home win price off the export (None where it has none)."""
        rows = self.data.get(code) or []
        return _f(rows[0].get("prematch_prob_50")) if rows else None


def targeted(dig, gamers=None):
    """The gamers the money wins on, most first: [(gamer, backers' edge shrunk, stake, matches)]."""
    acc = defaultdict(lambda: [0.0, 0.0, set()])
    for b in dig.bets:
        g = dig.gamer(b["match"], 0 if b["home"] else 1)
        if not g:
            continue
        a = acc[g]
        a[0] += b["stake"] * ((1.0 if b["won"] else 0.0) - b["p"])
        a[1] += b["stake"]
        a[2].add(b["match"])
    rows = [(g, a[0] / (a[1] + SHRINK_STAKE), a[1], len(a[2])) for g, a in acc.items()]
    rows.sort(key=lambda r: -r[1])
    if gamers:
        want = {g.strip().upper() for g in gamers}
        return [r for r in rows if r[0] in want] + [(g, float("nan"), 0.0, 0) for g in sorted(want - {r[0] for r in rows})]
    return rows[:TOP]


def section_targeted(dig, top):
    """Their record against prod, baseline against the league's, window play against their own."""
    lg = dig.league()
    L = [f"  the gamers the money wins on (backers' edge over prod's price, shrunk by {SHRINK_STAKE:,.0f} stake):",
         f"  {'gamer':12s} {'edge':>6s} {'stake':>9s} {'matches':>7s} {'won':>6s} {'prod':>6s}   "
         + " ".join(f"{f:>9s}" for f in PROFILE) + "   (baseline less the league's; then the window's less the baseline)"]
    for g, edge, stake, n in top:
        won, pr, cnt = 0.0, 0.0, 0
        for t, code, side in dig.gm.get(g, []):
            if not dig.since <= t.date() <= dig.until or code not in dig.results:
                continue
            p = dig.prematch(code)
            if p is None:
                continue
            fin = dig.results[code]
            won += 0.5 if fin[0] == fin[1] else float((fin[0] > fin[1]) == (side == 0))
            pr += p if side == 0 else 1 - p
            cnt += 1
        base = dig.baseline(g)
        acc, m = defaultdict(float), 0
        for t, code, side in dig.gm.get(g, []):
            if dig.since <= t.date() <= dig.until and code in dig.data:
                pf = dig.match_profile(code).get(side)
                if pf:
                    for f in PROFILE:
                        if not math.isnan(pf[f]):
                            acc[f] += pf[f] * pf["n"]
                    m += pf["n"]
        win_p = {f: acc[f] / m for f in PROFILE} if m else None
        rec = f"{won / cnt:6.3f} {pr / cnt:6.3f}" if cnt else f"{'':6s} {'':6s}"
        L.append(f"  {g:12s} {edge:+6.3f} {stake:9,.0f} {n:7d} {rec}   "
                 + " ".join(f"{(base[f] - lg[f]) if base else float('nan'):+9.3f}" for f in PROFILE))
        if base and win_p:
            L.append(f"  {'':12s} {'':6s} {'':9s} {'':7s} {'':6s} {'':6s}   "
                     + " ".join(f"{win_p[f] - base[f]:+9.3f}" for f in PROFILE))
    L.append("  the league's: " + ", ".join(f"{f} {lg[f]:.3f}" for f in PROFILE))
    return L


def section_earlier(dig):
    """Does a gamer's play in the session's earlier matches, against their baseline, tell their result
    beyond prod's pre-match price? Every window match with a pre-match price, both gamers."""
    rows = []
    bases = {}
    for code, h in dig.hist.items():
        if code not in dig.data or code not in dig.results or not dig.since <= day(code) <= dig.until:
            continue
        p = dig.prematch(code)
        if p is None:
            continue
        fin = dig.results[code]
        if fin[0] == fin[1]:
            continue
        for side in (0, 1):
            g = dig.gamer(code, side)
            if g not in bases:
                bases[g] = dig.baseline(g)
            base = bases[g]
            earlier = session_before(dig.gm, g, code)
            if not base or not earlier:
                continue
            acc, n = defaultdict(float), 0
            for c, s_ in earlier[:3]:
                pf = dig.match_profile(c).get(s_) if c in dig.data else None
                if pf:
                    for f in PROFILE:
                        if not math.isnan(pf[f]):
                            acc[f] += pf[f] * pf["n"]
                    n += pf["n"]
            if not n:
                continue
            rows.append(dict(code=code, y=float((fin[0] > fin[1]) == (side == 0)),
                             price=_logit(p if side == 0 else 1 - p), run=run_of(dig.results, earlier),
                             **{f: acc[f] / n - base[f] for f in PROFILE}))
    L = [f"  earlier matches of the session (the last three, against the gamer's own baseline): {len(rows):,}"
         " gamer-matches; won ~ logit(prod's pre-match price) + the feature, match-clustered"]
    if len(rows) < 30:
        return L + ["  (too few)"]
    y = np.array([r["y"] for r in rows])
    grp = [r["code"] for r in rows]
    for f in PROFILE + ("run",):
        x = np.array([r[f] for r in rows], float)
        ok = ~np.isnan(x)
        if ok.sum() < 30 or x[ok].std() == 0:
            continue
        z = (x[ok] - x[ok].mean()) / x[ok].std()
        X = np.c_[np.ones(ok.sum()), [r["price"] for r, k in zip(rows, ok) if k], z]
        b, se = logistic(X, y[ok], [g for g, k in zip(grp, ok) if k])
        L.append(f"    {f:10s} a sd of it: {b[2]:+.3f} (se {se[2]:.3f}, t {b[2] / se[2]:+.2f})   price {b[1]:.2f}")
    return L


def last_snap(recs):
    """The kind of the snap before a bet: touchdown, turnover, explosive, first down, stuffed, other."""
    if not recs:
        return "none yet"
    r = max(recs, key=lambda x: x["message"])
    if r["kind"] != v21.GAIN:
        return "turnover"
    if r["gain"] >= 100 - r["field"]:
        return "touchdown"
    if r["gain"] >= EXPLOSIVE:
        return "explosive"
    if r.get("converted") and not r.get("replay"):
        return "first down"
    if r["gain"] <= 0:
        return "stuffed"
    return "other"


def bet_rows(dig, bets):
    """Each in-play bet with the backed side's and the opponent's play so far (against their baselines)
    and the snap it followed."""
    out = []
    bases = {}
    for b in bets:
        if not b["in_play"] or b["message"] is None or b["match"] not in dig.data:
            continue
        side = 0 if b["home"] else 1
        so_far = side_snaps(dig.data[b["match"]], b["message"], dig.snaps(b["match"]))
        backed = dig.gamer(b["match"], side)
        row = dict(code=b["match"], gamer=backed, run=run_of(dig.results, session_before(dig.gm, backed, b["match"])),
                   y=float(b["won"]), price=_logit(b["p"]), stake=b["stake"], edge=float(b["won"]) - b["p"],
                   last_mine=last_snap(so_far[side]), last_theirs=last_snap(so_far[1 - side]),
                   last=last_snap(so_far[0] + so_far[1]),
                   last_by="backed" if so_far[side] and (not so_far[1 - side] or max(r["message"] for r in so_far[side])
                                                        > max(r["message"] for r in so_far[1 - side])) else "opponent")
        for who, s_ in (("mine", side), ("theirs", 1 - side)):
            g = dig.gamer(b["match"], s_)
            if g not in bases:
                bases[g] = dig.baseline(g)
            pf = profile(so_far[s_])
            base = bases[g]
            for f in PROFILE:
                row[f"{who}_{f}"] = (pf[f] - base[f]) if (pf and base and not math.isnan(pf[f])) else float("nan")
            row[f"{who}_n"] = pf["n"] if pf else 0
        out.append(row)
    return out


def section_this_match(rows, label):
    L = [f"  {label}: {len(rows):,} in-play moneyline bets; won ~ logit(prod's price at the bet) + the feature so far"
         " this match (against the gamer's baseline), match-clustered"]
    if len(rows) < 50:
        return L + ["  (too few)"]
    y = np.array([r["y"] for r in rows])
    grp = [r["code"] for r in rows]
    for who in ("mine", "theirs"):
        for f in PROFILE:
            x = np.array([r[f"{who}_{f}"] for r in rows], float)
            n = np.array([r[f"{who}_n"] for r in rows])
            ok = ~np.isnan(x) & (n >= 5)
            if ok.sum() < 50 or x[ok].std() == 0:
                continue
            z = (x[ok] - x[ok].mean()) / x[ok].std()
            X = np.c_[np.ones(ok.sum()), [r["price"] for r, k in zip(rows, ok) if k], z]
            b, se = logistic(X, y[ok], [g for g, k in zip(grp, ok) if k])
            L.append(f"    {'backed' if who == 'mine' else 'opponent':8s} {f:10s} a sd of it: {b[2]:+.3f}"
                     f" (se {se[2]:.3f}, t {b[2] / se[2]:+.2f})")
    return L


def section_last(rows, label):
    """How often a bet followed each kind of snap, and the backers' edge after it (stake-weighted)."""
    by = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        k = (r["last_by"], r["last"])
        by[k][0] += 1
        by[k][1] += r["stake"]
        by[k][2] += r["stake"] * r["edge"]
    tot = sum(v[1] for v in by.values()) or 1.0
    L = [f"  {label}: the snap the bet followed (whose, what), the share of the stake and the backers' edge after it"]
    for (whose, what), (n, stake, se) in sorted(by.items(), key=lambda kv: -kv[1][1]):
        L.append(f"    {whose:8s} {what:10s} bets {n:6,d}  stake {100 * stake / tot:5.1f}%  edge {se / stake:+.3f}")
    return L


def run(data, history, bets=None, since=None, until=None, gamers=None):
    """The report's lines (and the in-play bet rows)."""
    dig = Dig(data, history, bets, since, until)
    L = [f"  window {dig.since} to {dig.until}; baselines over the {BASELINE_DAYS} days before; {len(dig.bets):,}"
         " moneyline bets"]
    top = targeted(dig, gamers) if dig.bets else []
    if top:
        L += section_targeted(dig, top)
    L += section_earlier(dig)
    rows = bet_rows(dig, dig.bets) if dig.bets else []
    if rows:
        names = {g for g, *_ in top}
        on_top = [r for r in rows if r["gamer"] in names]
        hot = [r for r in rows if r["run"] >= HOT]
        L += section_this_match(rows, "every in-play bet")
        L += section_last(rows, "every in-play bet")
        if on_top:
            L += section_this_match(on_top, "bets backing the targeted gamers")
            L += section_last(on_top, "bets backing the targeted gamers")
        if hot:
            L += section_this_match(hot, f"bets backing a gamer on {HOT}+ wins this session")
            L += section_last(hot, f"bets backing a gamer on {HOT}+ wins this session")
    return L, rows
