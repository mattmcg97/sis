"""The money layer: what the money already bet says about who wins, two ways.

  the days before   each gamer's pre-match price moved by their backers' edge over the price each bet
                    was laid at, on the gamer's matches settled before this one, shrunk by the stake
                    behind it: a prior table any sim can price from, walk-forward
  the match so far  each moneyline bet's price moved toward the stake already placed on its side of the
                    match against the other side's (bets_sim.csv only): the same bets re-priced, and two
                    books

Bettors beat both prod and v2.1 by about 9.5 points of the operator's margin on the moneyline, and the
gap is all informed money: bets under 10 lose at the margin prod's prices say they should (10.4% in play
against 10.5%), bets of 200 and up carry nearly two-thirds of the shortfall, and most of it is on a
handful of gamers whose level jumped beyond chance from September to October, with the money on them
from their first days. On Oct 1-12 (the first pass): the days before moved the Brier by -0.0003
[-0.0035, +0.0023] (kick-off cut; -0.0018 at the best slope picked in hindsight) -- no material gain;
the match so far, its slope fitted day by day on the days before, re-priced prod's bets by +2.3 points
[+1.4, +3.1] and v2.1's by +1.7 [+1.0, +2.3] (95% over matches) -- a quarter of the gap, from following
the money, not from the model.

    python -m eAMFCalibrator money-layer --since "2026-09-01 00:00:00" --until "2026-10-13 00:00:00" \\
        --base eAMFCalibrator/out/walkthrough/priors/avg.csv --out eAMFCalibrator/out_money_avg
    python -m eAMFCalibrator money-layer --since "2026-10-01 00:00:00" --until "2026-10-13 00:00:00" \\
        --base prod --history eAMFCalibrator/out/match_history.csv --bets eAMFCalibrator/out_bets_v21_4k/bets_sim.csv

The days before:
  the edge     each settled moneyline bet backs a gamer: won (1 or 0) less the price it was laid at --
               prod's probability at the bet when the bets come from a betting sim's bets_sim.csv
               (--bets), else the operator's own odds with its margin taken out (BET_MARGIN), off
               config.BET_TABLE over the window. On Oct 1-12 the two give the same signal (correlation
               1.000), so the bet table alone, with no quotes fetched, is enough
  the signal   a gamer's, as of a match: the stake-weighted edge over the bets on the gamer's matches
               settled by its kick-off (--cut daily: by the start of its day), over that stake plus
               SHRINK, so a gamer with little money behind them stays near none (--half-life: older bets
               count for less)
  the slope    the base price's miss (won - p_home, a tie half) on the home gamer's signal less the away
               gamer's, fitted on every earlier day's matches and shrunk toward none by SLOPE_PRIOR
               matches' worth (--slope: one fixed throughout), so no match is priced off its own day
  the table    the base table (a walkthrough's priors/VARIANT.csv, or any table with MATCH_CODE and the
               two sides' points) with each match's p_home moved by slope x that difference, and its
               expected margin moved with it at the table's own points per probit. The points move that
               margin over `share` (half to each side, the total kept), so a sim that goes `share` of
               the way to the table -- v21's kick-off nudge, NUDGE_SHARE -- lands on the whole move
               (--share 1: a reader that takes the table as it is)
  the test     base and money on the results from --test-since: Brier and log loss and the change, a 95%
               interval over days; the slope on the test days alone; the base's miss by fifth of the
               signal; each week; prod's own pre-match price on the same matches when --bets carry it;
               and the settings (shrink, half-life, cut) side by side
  --base prod tests it on prod's own pre-match price (the latest pre-match moneyline bet's prod
               probability in --bets), the kick-offs, gamers and results off --history: no table

The match so far (with --bets): each settled moneyline bet's money so far, (the stake already on its
side - on the other) / (both + FLOW_SHRINK), every earlier bet placed counting; the price (prod's, and
each candidate's in the file) moved by beta x that in logit, beta fitted (stake-weighted, the price as
offset) on the bets of the days before; each bet re-priced as the betting sim does (odds x prod /
price, the operator's margin kept); the margin as it is and moved, the change, and two books against
prod as placed, 95% intervals over matches.

Writes (to --out): money.csv (every base row, those in the window moved, the money's columns added),
money_gamers.csv (each gamer's backers over the window: bets, stake, edge, signal at the end) and
money_layer.txt.
"""

import csv
import datetime as dt
import math
import os
import random
from collections import defaultdict, namedtuple
from statistics import NormalDist

from . import config, markets

SHRINK = 50000.0                           # stake: a signal is the backers' edge x stake / (stake + SHRINK)
SLOPE_PRIOR = 200                          # matches' worth of no effect each day's slope is shrunk toward
BET_MARGIN = {False: 0.082, True: 0.092}   # the operators' margin over prod's probability, pre-match / in
#                                            play (bets run's operator check, Oct 1-12: 8.1-8.3% / 8.8-9.5%)
RESULT_IN = dt.timedelta(minutes=36)       # a match's bets are settled this long after its kick-off
P_FLOOR = 0.02                             # the money's price is kept to [P_FLOOR, 1 - P_FLOOR]
SIGMA = 13.0                               # points per probit, where a table is too small to fit its own
TEST_AFTER = 7                             # days: the test starts this long after the window does
FLOW_SHRINK = 500.0                        # stake: the money so far is (this side - the other) / (both + it)
N_FLOW_BOOT = 1000
CUTS = ("daily", "kickoff")
GRID = dict(shrink=(25000.0, 50000.0, 100000.0), half_life=(None, 14.0, 7.0), cut=CUTS)
N_BOOT = 2000
SIDE = {"Home": 0, "Away": 1}
FEED_SIDE = {50: 0, 51: 1}                 # the moneyline's feed markets: the side each backs
MONEY_COLUMNS = ["money_home", "money_away", "money_slope", "money_dp", "money_margin", "base_p_home",
                 "base_home_pts", "base_away_pts"]
GAMER_FIELDS = ["gamer", "matches", "bets", "stake", "edge", "signal"]
POINT_PAIRS = (("home_pts", "away_pts"), ("Pred_P1_Points", "Pred_P2_Points"), ("exp_p1", "exp_p2"))

Backed = namedtuple("Backed", "match side stake won price in_play")


def _time(text):
    text = str(text or "")[:19].replace("T", " ")
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def _int(v):
    x = _num(v)
    return None if x is None else int(round(x))


def _handle(v):
    return str(v or "").strip().upper()


def _probit(p):
    return NormalDist().inv_cdf(min(max(p, 1e-6), 1 - 1e-6))


def _clamp(p):
    return min(max(p, P_FLOOR), 1 - P_FLOOR)


def outcome(m):
    """The home side's result: 1 won, 0 lost, a tie half; None without a final."""
    f = m.get("final")
    if not f:
        return None
    return 1.0 if f[0] > f[1] else 0.0 if f[0] < f[1] else 0.5


# ---------------------------------------------------------------------------------------- reading

def read_table(path):
    """(rows, columns) of a prior table as written."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def point_columns(cols):
    """The table's (home, away) points columns, or None."""
    return next((p for p in POINT_PAIRS if p[0] in cols and p[1] in cols), None)


def read_history(path):
    """code -> (kick-off, home gamer, away gamer, (home score, away score) or None) off a match history."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("SPORT_CODE", "AF") not in ("AF", "", None):
                continue
            t = _time(r.get("SCHEDULED_START_TIME_UTC"))
            if t is None or not r.get("MATCH_CODE"):
                continue
            a, b = _int(r.get("PLAYER_1_FINAL_SCORE")), _int(r.get("PLAYER_2_FINAL_SCORE"))
            out[r["MATCH_CODE"].strip()] = (t, _handle(r.get("PLAYER_1_HANDLE")), _handle(r.get("PLAYER_2_HANDLE")),
                                            None if a is None or b is None else (a, b))
    return out


def points_per_probit(rows, cols):
    """The table's expected margin per unit of probit of its home price (least squares through the origin,
    prices 0.05-0.95): how far its book moves the margin for a move in the price. SIGMA without one."""
    pair = point_columns(cols)
    if pair is None or "p_home" not in cols:
        return SIGMA
    sxy = sxx = 0.0
    n = 0
    for r in rows:
        p, h, a = _num(r.get("p_home")), _num(r.get(pair[0])), _num(r.get(pair[1]))
        if p is None or h is None or a is None or not 0.05 < p < 0.95:
            continue
        z = _probit(p)
        sxy += z * (h - a)
        sxx += z * z
        n += 1
    return sxy / sxx if n >= 50 and sxx > 0 and sxy > 0 else SIGMA


def matches_of(rows, cols, history=None):
    """code -> dict(start, home, away, final, p): the table's matches, the kick-off, gamers and result
    filled from the history where the table lacks them; the history's other matches carry no price (their
    bets still count toward the gamers' signals). A table without p_home is priced off its points."""
    history = history or {}
    pair = point_columns(cols)
    out = {}
    for code, (t, home, away, final) in history.items():
        out[code] = dict(start=t, home=home, away=away, final=final, p=None)
    for r in rows:
        code = str(r.get("MATCH_CODE") or "").strip()
        if not code:
            continue
        h = history.get(code)
        t = _time(r.get("start")) or (h[0] if h else None)
        if t is None:
            continue
        hs, as_ = _int(r.get("home_score")), _int(r.get("away_score"))
        final = (hs, as_) if hs is not None and as_ is not None else (h[3] if h else None)
        p = _num(r.get("p_home"))
        if p is None and pair:
            hp, ap = _num(r.get(pair[0])), _num(r.get(pair[1]))
            p = None if hp is None or ap is None else NormalDist().cdf((hp - ap) / SIGMA)
        out[code] = dict(start=t, home=_handle(r.get("home")) or (h[1] if h else ""),
                         away=_handle(r.get("away")) or (h[2] if h else ""), final=final, p=p)
    return out


def read_sim_bets(paths):
    """A betting sim's bets (bets_sim.csv), each once (by bet_id) across the files."""
    seen, out = set(), []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = r.get("bet_id")
                if key and key in seen:
                    continue
                seen.add(key)
                out.append(r)
    return out


def sim_columns(paths):
    """Every column of the bets_sim.csv files, in order."""
    cols = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            cols += [c for c in (next(csv.reader(fh), None) or []) if c not in cols]
    return cols


def backed_from_sim(rows):
    """The settled moneyline bets of bets_sim.csv rows, each at prod's probability at the bet (else the
    operator's odds less its margin)."""
    out = []
    for r in rows:
        if r.get("market") != markets.MONEYLINE or r.get("result") not in ("won", "lost"):
            continue
        side, stake = SIDE.get(r.get("selection")), _num(r.get("stake"))
        if side is None or not stake or stake <= 0:
            continue
        in_play = str(r.get("in_play")).strip().lower() == "true"
        price = _num(r.get("stream_prob"))
        if price is None:
            implied = _num(r.get("implied_prob"))
            if not implied:
                continue
            price = implied / (1 + BET_MARGIN[in_play])
        out.append(Backed(str(r.get("match_code")).strip(), side, stake, r["result"] == "won", price, in_play))
    return out


def backed_from_table(bets):
    """The settled moneyline bets of bets.Bet objects, each at the operator's odds less its margin."""
    from .bets import LOST, WON, result_of
    out = []
    for b in bets:
        side, res = FEED_SIDE.get(b.feed_market), result_of(b)
        if side is None or res not in (WON, LOST) or not b.odds or b.odds <= 1 or not b.stake or b.stake <= 0:
            continue
        out.append(Backed(str(b.match_code).strip(), side, b.stake, res == WON,
                          1.0 / b.odds / (1 + BET_MARGIN[b.in_play]), b.in_play))
    return out


def fetch_backed(cur):
    """The window's settled moneyline bets off config.BET_TABLE."""
    from . import bets as bets_mod
    from .snowflake_io import fetch_all
    sql, params, _ = bets_mod.bets_sql()
    cols, raw = fetch_all(cur, sql, tuple(params))
    return backed_from_table(bets_mod.to_bets(cols, raw))


def prod_prematch(rows):
    """code -> prod's pre-match home probability: its price at the match's last settled pre-match
    moneyline bet of bets_sim.csv rows."""
    last = {}
    for r in rows:
        if (r.get("market") != markets.MONEYLINE or str(r.get("in_play")).strip().lower() == "true"
                or r.get("result") not in ("won", "lost", "push")):
            continue
        p, side = _num(r.get("stream_prob")), SIDE.get(r.get("selection"))
        if p is None or side is None:
            continue
        key = str(r.get("bet_time") or "")
        code = str(r.get("match_code")).strip()
        if code not in last or key >= last[code][0]:
            last[code] = (key, p if side == 0 else 1 - p)
    return {c: v[1] for c, v in last.items()}


# ---------------------------------------------------------------------------------------- the layer

def cut_of(start, cut):
    """When a match's signal is read: the start of its day (daily) or its kick-off."""
    return dt.datetime(start.year, start.month, start.day) if cut == "daily" else start


def signals(backed, matches, shrink=SHRINK, half_life=None, cut="kickoff"):
    """code -> (home gamer's signal, away gamer's) as of each match's cut, off the bets settled by then."""
    events = []
    for b in backed:
        m = matches.get(b.match)
        g = m and (m["home"], m["away"])[b.side]
        if g:
            events.append((m["start"] + RESULT_IN, g, b.stake, b.stake * ((1.0 if b.won else 0.0) - b.price)))
    events.sort(key=lambda e: e[0])

    def at(a, t):
        if not half_life:
            return a[0], a[1]
        f = 0.5 ** ((t - a[2]).total_seconds() / 86400.0 / half_life)
        return a[0] * f, a[1] * f

    acc, out, i = {}, {}, 0
    for c in sorted(matches, key=lambda c: (cut_of(matches[c]["start"], cut), c)):
        t = cut_of(matches[c]["start"], cut)
        while i < len(events) and events[i][0] <= t:
            when, g, stake, edge = events[i]
            se, s = at(acc[g], when) if g in acc else (0.0, 0.0)
            acc[g] = (se + edge, s + stake, when)
            i += 1
        sig = []
        for g in (matches[c]["home"], matches[c]["away"]):
            se, s = at(acc[g], t) if g in acc else (0.0, 0.0)
            sig.append(se / (s + shrink))
        out[c] = tuple(sig)
    return out


def walk(matches, sig, slope=None, prior=SLOPE_PRIOR):
    """code -> (slope, the home price's move) for every match with a base price, the slope fitted on the
    matches of the days before (the base's miss on the signals' difference, shrunk toward none by `prior`
    matches' worth), or `slope` throughout; and [(day, slope)]."""
    by_day = defaultdict(list)
    for c, m in matches.items():
        if m.get("p") is not None and c in sig:
            by_day[m["start"].date()].append(c)
    sxy = sxx = 0.0
    n = 0
    out, path = {}, []
    for d in sorted(by_day):
        b = slope if slope is not None else (sxy / (sxx * (1 + prior / n)) if n and sxx > 0 else 0.0)
        path.append((d, b))
        for c in by_day[d]:
            out[c] = (b, b * (sig[c][0] - sig[c][1]))
        for c in by_day[d]:
            y, x = outcome(matches[c]), sig[c][0] - sig[c][1]
            if y is None or x == 0:
                continue
            sxy += x * (y - matches[c]["p"])
            sxx += x * x
            n += 1
    return out, path


def money_price(m, move):
    return _clamp(m["p"] + move[1])


# ---------------------------------------------------------------------------------------- the test

def _scores(p, y):
    q = min(max(p, 1e-6), 1 - 1e-6)
    return (p - y) ** 2, -(y * math.log(q) + (1 - y) * math.log(1 - q))


def _boot(by_day, n_boot=N_BOOT, seed=0):
    """The mean of the per-match values and its 95% interval over days: by_day {day: [values]}."""
    days = [v for v in by_day.values() if v]
    n = sum(len(v) for v in days)
    if not n:
        return None, None, None
    mean = sum(sum(v) for v in days) / n
    rng = random.Random(seed)
    sums = [(sum(v), len(v)) for v in days]
    draws = []
    for _ in range(n_boot):
        s = c = 0
        for _ in range(len(sums)):
            a, b = sums[rng.randrange(len(sums))]
            s += a
            c += b
        draws.append(s / c if c else 0.0)
    draws.sort()
    return mean, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot) - 1]


def tested(matches, moves, since, until=None):
    """The test's matches: a base price, a move and a result, kicked off from `since` (a date) to `until`."""
    return [c for c, m in matches.items() if c in moves and outcome(m) is not None and m["start"].date() >= since
            and (until is None or m["start"] < until)]


def grade(matches, moves, codes, other=None):
    """dict: n, base and money Brier and log loss, each change with its 95% interval over days (money
    less base: below zero, the money closer). `other`: code -> another price (prod's) for the same matches."""
    out = dict(n=len(codes))
    for k, (name, i) in enumerate((("brier", 0), ("log_loss", 1))):
        base, money, change, vs = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for c in codes:
            m, y = matches[c], outcome(matches[c])
            d = m["start"].date()
            b, s = _scores(m["p"], y)[i], _scores(money_price(m, moves[c]), y)[i]
            base[d].append(b)
            money[d].append(s)
            change[d].append(s - b)
            if other and c in other:
                vs[d].append(s - _scores(other[c], y)[i])
        out[name] = (_boot(base)[0], _boot(money)[0]) + _boot(change, seed=k)
        if other:
            out[name + "_vs_other"] = _boot(vs, seed=k + 2)
    return out


def test_slope(matches, sig, codes):
    """The base's miss on the signals' difference over the test matches alone (through the origin), with a
    95% interval over days."""
    by = defaultdict(list)
    for c in codes:
        x = sig[c][0] - sig[c][1]
        if x:
            by[matches[c]["start"].date()].append((x * (outcome(matches[c]) - matches[c]["p"]), x * x))
    days = list(by.values())
    if not days:
        return None, None, None
    sxy, sxx = sum(a for v in days for a, _ in v), sum(b for v in days for _, b in v)
    rng = random.Random(1)
    draws = []
    for _ in range(N_BOOT):
        a = b = 0.0
        for _ in range(len(days)):
            v = days[rng.randrange(len(days))]
            a += sum(x for x, _ in v)
            b += sum(x for _, x in v)
        draws.append(a / b if b else 0.0)
    draws.sort()
    return sxy / sxx, draws[int(0.025 * N_BOOT)], draws[int(0.975 * N_BOOT) - 1]


def fifths(matches, sig, moves, codes):
    """Lines: the test's matches with any signal, in fifths by the signals' difference: the base's miss
    and the money's move."""
    xs = sorted((sig[c][0] - sig[c][1], c) for c in codes if sig[c][0] - sig[c][1])
    lines = []
    if len(xs) < 25:
        return lines
    lines.append("  fifth   matches   signal (home - away)   base's miss (won - p)   money's move")
    for k in range(5):
        part = xs[k * len(xs) // 5:(k + 1) * len(xs) // 5]
        miss = [outcome(matches[c]) - matches[c]["p"] for _, c in part]
        se = (sum((v - sum(miss) / len(miss)) ** 2 for v in miss) / max(1, len(miss) - 1)) ** 0.5 / len(miss) ** 0.5
        lines.append(f"  {k + 1:>5}   {len(part):>7,}   {sum(x for x, _ in part) / len(part):+20.4f}"
                     f"   {sum(miss) / len(miss):+12.4f} (se {se:.3f})   {sum(moves[c][1] for _, c in part) / len(part):+12.4f}")
    return lines


def gamer_rows(backed, matches, shrink=SHRINK):
    """Each gamer's backers over the window: matches, bets, stake, edge (per unit staked) and the signal
    at the end (no half-life)."""
    acc = defaultdict(lambda: [set(), 0, 0.0, 0.0])
    for b in backed:
        m = matches.get(b.match)
        g = m and (m["home"], m["away"])[b.side]
        if not g:
            continue
        a = acc[g]
        a[0].add(b.match)
        a[1] += 1
        a[2] += b.stake
        a[3] += b.stake * ((1.0 if b.won else 0.0) - b.price)
    rows = [dict(gamer=g, matches=len(a[0]), bets=a[1], stake=round(a[2], 2), edge=round(a[3] / a[2], 4),
                 signal=round(a[3] / (a[2] + shrink), 5)) for g, a in acc.items() if a[2] > 0]
    return sorted(rows, key=lambda r: -r["signal"])


def _f(x, fmt="+.4f"):
    return "    -   " if x is None else format(x, fmt)


def _change(g, key):
    base, money, mean, lo, hi = g[key]
    return (f"{_f(base, '.4f')} -> {_f(money, '.4f')}  change {_f(mean)} [{_f(lo)}, {_f(hi)}]")


# ---------------------------------------------------------------------------------------- the match so far

def price_columns(cols):
    """The prices a bets_sim.csv carries: prod's, then each candidate's (candidate_prob or candidate_prob_<name>)."""
    return [("prod", "stream_prob")] + [(c[len("candidate_prob_"):] if c != "candidate_prob" else "candidate", c)
                                        for c in cols if c == "candidate_prob" or c.startswith("candidate_prob_")]


def flow_bets(rows, price):
    """The settled moneyline bets of bets_sim.csv rows priced by prod and by the column `price`, each with
    the money already in its match: (the stake on its side - on the other) / (both + FLOW_SHRINK), every
    earlier bet placed (settled or not) counting."""
    ml = [r for r in rows if r.get("market") == markets.MONEYLINE and SIDE.get(r.get("selection")) is not None]
    ml.sort(key=lambda r: (str(r.get("match_code")), str(r.get("bet_time") or "")))
    on = defaultdict(lambda: [0.0, 0.0])
    out = []
    for r in ml:
        side, stake = SIDE[r["selection"]], _num(r.get("stake")) or 0.0
        match = str(r.get("match_code")).strip()
        a = on[match]
        p, q, odds, rev = _num(r.get("stream_prob")), _num(r.get(price)), _num(r.get("odds")), _num(r.get("revenue"))
        if r.get("result") in ("won", "lost") and p and q and odds and rev is not None and stake > 0:
            out.append(dict(match=match, day=str(r.get("bet_time") or "")[:10],
                            x=(a[side] - a[1 - side]) / (a[side] + a[1 - side] + FLOW_SHRINK),
                            y=1.0 if r["result"] == "won" else 0.0, stake=stake, odds=odds,
                            p=min(max(p, 0.005), 0.995), q=min(max(q, 0.005), 0.995), rev=rev))
        a[side] += stake
    return out


def _logit(p):
    return math.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z > -30 else 0.0


def flow_beta(bets):
    """The money so far's effect in logit on the price (as offset), stake-weighted, by Newton's method."""
    b = 0.0
    for _ in range(25):
        g = h = 0.0
        for r in bets:
            c = _sigmoid(_logit(r["q"]) + b * r["x"])
            g += r["stake"] * (r["y"] - c) * r["x"]
            h += r["stake"] * c * (1 - c) * r["x"] * r["x"]
        if h <= 0:
            return 0.0
        step = g / h
        b += step
        if abs(step) < 1e-6:
            break
    return b


def flow_walk(bets):
    """The bets from the second day on, each with the beta fitted on the days before; [(day, beta)]."""
    days = sorted({r["day"] for r in bets})
    by = defaultdict(list)
    for r in bets:
        by[r["day"]].append(r)
    out, path, seen = [], [], []
    for d in days:
        if seen:
            b = flow_beta(seen)
            path.append((d, b))
            out += [dict(r, beta=b) for r in by[d]]
        seen += by[d]
    return out, path


def flow_sums(bets):
    """match -> [stake, prod's revenue as placed, the price's, the price moved's, stake it takes in two
    books, its revenue there, stake prod keeps, prod's revenue there], each bet re-priced as the betting
    sim does (odds x prod / price, the operator's margin kept)."""
    out = defaultdict(lambda: [0.0] * 8)
    for r in bets:
        c = min(max(_sigmoid(_logit(r["q"]) + r["beta"] * r["x"]), 0.005), 0.995)
        s, won = r["stake"], r["y"] == 1.0
        o_q, o_c = r["odds"] * r["p"] / r["q"], r["odds"] * r["p"] / c
        a = out[r["match"]]
        a[0] += s
        a[1] += r["rev"]
        a[2] += s - s * o_q if won else s
        rc = s - s * o_c if won else s
        a[3] += rc
        if o_c > r["odds"]:                      # the bettor takes the better price
            a[4] += s
            a[5] += rc
        else:
            a[6] += s
            a[7] += r["rev"]
    return out


def flow_stats(sums):
    """[prod as placed, the price, moved, moved less the price, two books: it takes, prod keeps, it less prod]
    margins (%) over these matches' sums."""
    t = [sum(v[i] for v in sums) for i in range(8)]
    m = [100 * t[1] / t[0], 100 * t[2] / t[0], 100 * t[3] / t[0]]
    takes = 100 * t[5] / t[4] if t[4] else 0.0
    keeps = 100 * t[7] / t[6] if t[6] else 0.0
    return m + [m[2] - m[1], takes, keeps, takes - keeps]


def flow_test(rows, cols, n_boot=None, seed=0):
    """Lines: the match so far, for prod's price and each candidate's in the bets."""
    n_boot = n_boot or N_FLOW_BOOT
    L = [f"\n  the match so far: each moneyline price moved by beta x the money already on its side, (this side"
         f" - the other) / (both + {FLOW_SHRINK:,.0f}), beta fitted on the days before; the bets re-priced (odds x"
         f" prod / price), margins with 95% intervals over matches"]
    names = ("prod as placed", "the price as it is", "the price moved", "moved less as it is",
             "two books: moved takes", "two books: prod keeps", "two books: takes less keeps")
    for label, col in price_columns(cols):
        bets, path = flow_walk(flow_bets(rows, col))
        if not bets:
            continue
        sums = flow_sums(bets)
        keys = sorted(sums)
        full = flow_stats([sums[k] for k in keys])
        rng = random.Random(seed)
        draws = [flow_stats([sums[keys[rng.randrange(len(keys))]] for _ in keys]) for _ in range(n_boot)]
        taken = sum(v[4] for v in sums.values()) / sum(v[0] for v in sums.values())
        betas = [b for _, b in path]
        L.append(f"\n  {label} ({col}): {len(bets):,} bets on {len(keys):,} matches from {path[0][0]}; beta {min(betas):+.2f}"
                 f" to {max(betas):+.2f}, the last {betas[-1]:+.2f}; in two books the moved price takes {100 * taken:.0f}% of the stake")
        for i, n in enumerate(names):
            v = sorted(d[i] for d in draws)
            L.append(f"    {n:28s} {full[i]:+7.2f}%  [{v[int(0.025 * n_boot)]:+.2f}, {v[int(0.975 * n_boot) - 1]:+.2f}]")
    return L


# ---------------------------------------------------------------------------------------- the table

def money_table(rows, cols, matches, moves, sig, sigma, share):
    """The base table's rows with those priced here moved: p_home to the money's price, the expected margin
    by sigma x the change in probit over share (half to each side's points, the total kept)."""
    pair = point_columns(cols)
    out = []
    for r in rows:
        r = dict(r)
        code = str(r.get("MATCH_CODE") or "").strip()
        m = matches.get(code)
        if m is None or code not in moves or m.get("p") is None or pair is None:
            out.append(r)
            continue
        h, a = _num(r.get(pair[0])), _num(r.get(pair[1]))
        if h is None or a is None:
            out.append(r)
            continue
        p0, p1 = m["p"], money_price(m, moves[code])
        dm = sigma * (_probit(p1) - _probit(p0))
        half = dm / share / 2
        r.update(money_home=round(sig[code][0], 5), money_away=round(sig[code][1], 5),
                 money_slope=round(moves[code][0], 4), money_dp=round(p1 - p0, 5), money_margin=round(dm, 4),
                 base_p_home=round(p0, 5), base_home_pts=h, base_away_pts=a)
        for k, v in zip(pair, (h + half, a - half)):
            r[k] = round(v, 4)
        for k, v in zip(("Pred_P1_Points", "Pred_P2_Points"), (h + half, a - half)):
            if k in cols and pair[0] != k:
                r[k] = round(v, 4)
        if "exp_margin" in cols and _num(r.get("exp_margin")) is not None:
            r["exp_margin"] = round(_num(r["exp_margin"]) + 2 * half, 4)
        if "p_home" in cols:
            r["p_home"] = round(p1, 5)
        out.append(r)
    return out


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------------------- the run

def _window():
    since = _time(config.CUTOFF_START)
    until = _time(config.CUTOFF_END) if config.CUTOFF_END else None
    return since, until


def run(out_dir, base, history=None, bets=None, cur=None, shrink=None, half_life=None, cut="kickoff", slope=None,
        share=None, test_since=None, grid=True):
    """The whole layer. `base`: a prior table's path, or 'prod' (prod's pre-match price off `bets`, tested
    only); `bets`: bets_sim.csv paths, else the bet table through `cur`. Returns money_layer.txt's path."""
    from eAMFModel.v21_stream import NUDGE_SHARE
    shrink = SHRINK if shrink is None else float(shrink)
    share = NUDGE_SHARE if share is None else float(share)
    if not share > 0:
        raise SystemExit("--share must be above 0")
    since, until = _window()
    os.makedirs(out_dir, exist_ok=True)
    hist = read_history(history) if history else {}
    sim_rows = read_sim_bets(bets) if bets else None
    sim_cols = sim_columns(bets) if bets else []
    if str(base).lower() == "prod":
        if not sim_rows or not hist:
            raise SystemExit("--base prod reads prod's pre-match price off --bets and the matches off --history")
        rows, cols = [], []
        matches = matches_of([], [], hist)
        for c, p in prod_prematch(sim_rows).items():
            if c in matches:
                matches[c]["p"] = p
        sigma = SIGMA
    else:
        if not os.path.isfile(base):
            raise SystemExit(f"--base {base}: no such table")
        rows, cols = read_table(base)
        if "MATCH_CODE" not in cols or (point_columns(cols) is None and "p_home" not in cols):
            raise SystemExit(f"{base}: a prior table needs MATCH_CODE and its points (home_pts, away_pts) or p_home")
        matches = matches_of(rows, cols, hist)
        sigma = points_per_probit(rows, cols)
    matches = {c: m for c, m in matches.items() if m["start"] >= since and (until is None or m["start"] < until)}
    backed = backed_from_sim(sim_rows) if sim_rows is not None else fetch_backed(cur)
    known = [b for b in backed if b.match in matches]
    test_from = (_time(test_since) if test_since else None) or since + dt.timedelta(days=TEST_AFTER)
    test_from = test_from.date()
    source = "prod's probability at each bet (bets_sim.csv)" if sim_rows is not None else \
        f"the operators' odds less their margin ({100 * BET_MARGIN[False]:.1f}% pre-match, {100 * BET_MARGIN[True]:.1f}% in play)"
    L = [f"\nThe money layer: {since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M}" if until else
         f"\nThe money layer: from {since:%Y-%m-%d %H:%M}",
         f"  base: {'prod pre-match (off the bets)' if str(base).lower() == 'prod' else base}",
         f"  {sum(1 for m in matches.values() if m['p'] is not None):,} matches priced in the window,"
         f" {len(known):,} settled moneyline bets on them ({sum(b.in_play for b in known):,} in play,"
         f" {sum(b.stake for b in known):,.0f} staked), each at {source}"]
    if len(known) < len(backed):
        L.append(f"  ({len(backed) - len(known):,} bets on matches outside the window or the table left out)")
    sig = signals(known, matches, shrink, half_life, cut)
    moves, path = walk(matches, sig, slope)
    codes = tested(matches, moves, test_from, until)
    other = prod_prematch(sim_rows) if sim_rows is not None and str(base).lower() != "prod" else None
    g = grade(matches, moves, codes, other)
    L.append(f"\n  settings: shrink {shrink:,.0f} stake, half-life {f'{half_life:g} days' if half_life else 'none'},"
             f" cut {cut}, slope {'fitted day by day' if slope is None else f'{slope:g} fixed'}"
             f"; the table's points per probit {sigma:.2f}, its points moved over share {share:g}")
    L.append(f"\n  the test: {g['n']:,} matches from {test_from} (base and money on the result; change: money less"
             f" base, below zero the money closer; 95% over days)")
    L.append(f"    Brier     {_change(g, 'brier')}")
    L.append(f"    log loss  {_change(g, 'log_loss')}")
    if other:
        both = [c for c in codes if c in other]
        if both:
            gp = grade(matches, moves, both, other)
            prod_b = sum(_scores(other[c], outcome(matches[c]))[0] for c in both) / len(both)
            L.append(f"    on the {len(both):,} of them with prod's pre-match price in the bets: prod's Brier {prod_b:.4f},"
                     f" base {gp['brier'][0]:.4f}, money {gp['brier'][1]:.4f}; money less prod"
                     f" {_f(gp['brier_vs_other'][0])} [{_f(gp['brier_vs_other'][1])}, {_f(gp['brier_vs_other'][2])}]")
    b, lo, hi = test_slope(matches, sig, codes)
    if b is not None:
        L.append(f"    the base's miss on the signals' difference over the test days alone: slope {b:+.2f}"
                 f" [{lo:+.2f}, {hi:+.2f}] (the day-by-day fit's last: {path[-1][1]:+.2f})")
    moved = [abs(moves[c][1]) for c in codes]
    if moved:
        moved.sort()
        L.append(f"    the money's move on the home price: median {moved[len(moved) // 2]:.4f}, 90th percentile"
                 f" {moved[int(0.9 * (len(moved) - 1))]:.4f}, largest {moved[-1]:.4f}")
    L += ["\n  by fifth of the signal (the test's matches with any):"] + fifths(matches, sig, moves, codes)
    weeks = defaultdict(list)
    for c in codes:
        d = matches[c]["start"].date()
        weeks[d - dt.timedelta(days=d.weekday())].append(c)
    slope_on = dict(path)
    if weeks:
        L.append("\n  week of     matches   base Brier   money Brier   change    slope at its start")
        for w in sorted(weeks):
            gw = grade(matches, moves, weeks[w])
            first = min(matches[c]["start"].date() for c in weeks[w])
            L.append(f"  {w}  {gw['n']:>7,}   {gw['brier'][0]:.4f}       {gw['brier'][1]:.4f}       "
                     f"{gw['brier'][1] - gw['brier'][0]:+.4f}   {slope_on.get(first, 0.0):+.2f}")
    gamers = gamer_rows(known, matches, shrink)
    if gamers:
        b_last = path[-1][1] if path else 0.0
        L.append(f"\n  the gamers the money backs best and worst over the window (edge: per unit staked;"
                 f" signal at the end, its move on a price at the last slope {b_last:+.2f}):")
        L.append("  gamer             matches     bets        stake     edge    signal    move")
        top = gamers[:10] + [None] + gamers[-10:] if len(gamers) > 20 else gamers
        for r in top:
            if r is None:
                L.append("  ...")
                continue
            L.append(f"  {r['gamer'][:16]:16s} {r['matches']:>8,} {r['bets']:>8,} {r['stake']:>12,.0f}  {r['edge']:+.3f}"
                     f"  {r['signal']:+.4f}  {b_last * r['signal']:+.4f}")
    if grid:
        L.append("\n  the settings side by side (Brier change on the same test, 95% over days; the slope fitted"
                 " day by day each time):")
        L.append("  shrink      half-life   cut        Brier change                  log loss change   last slope")
        for k in GRID["shrink"]:
            for hl in GRID["half_life"]:
                for ct in GRID["cut"]:
                    s2 = signals(known, matches, k, hl, ct)
                    mv, pth = walk(matches, s2, slope)
                    g2 = grade(matches, mv, tested(matches, mv, test_from, until))
                    mark = "  <- this run" if (k, hl, ct) == (shrink, half_life, cut) else ""
                    L.append(f"  {k:>9,.0f}   {f'{hl:g} days' if hl else 'none':9s}   {ct:8s}   {_f(g2['brier'][2])}"
                             f" [{_f(g2['brier'][3])}, {_f(g2['brier'][4])}]   {_f(g2['log_loss'][2])}"
                             f"          {pth[-1][1] if pth else 0.0:+.2f}{mark}")
    if sim_rows is not None:
        L += flow_test(sim_rows, sim_cols)
    else:
        L.append("\n  the match so far: needs a betting sim's bets_sim.csv (--bets), each bet with the prices beside it")
    _write(os.path.join(out_dir, "money_gamers.csv"), gamers, GAMER_FIELDS)
    written = ["money_gamers.csv"]
    if rows:
        table = money_table(rows, cols, matches, moves, sig, sigma, share)
        _write(os.path.join(out_dir, "money.csv"), table, cols + [c for c in MONEY_COLUMNS if c not in cols])
        written.insert(0, "money.csv")
        L.append(f"\n  money.csv: the base's {len(rows):,} rows, {sum(1 for r in table if r.get('money_dp') not in (None, '')):,}"
                 f" moved; point a sim at it with --v21-prior-model {os.path.join(out_dir, 'money.csv')}")
    path_txt = os.path.join(out_dir, "money_layer.txt")
    with open(path_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n  -> {', '.join(os.path.join(out_dir, w) for w in written)}")
    return path_txt
