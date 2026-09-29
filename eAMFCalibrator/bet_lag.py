"""The lag, match by match: at what delay behind prod each operator's bets were struck, read off the
lines and off the odds, for every match of a window, with a plot of each.

  lines   a spread or total bet is on the line the operator showed, which is prod's live line a lag
          earlier. So each bet allows the lags at which prod's line (its latest live quote) equals
          the bet's line; a bet struck near a line move pins the lag down. Per match: the share of
          its line bets that fit each lag, the best lag, and the plateau of lags that fit as many.
  odds    each in-play bet's odds against prod's probability a lag earlier (moneyline always;
          spread and total where the line is prod's then), the operator's margin taken out as the
          median: the lag with the smallest misfit.
  bounds  every line bet that fits some lags and not others says which: struck on the old line
          after prod moved (the lag is at least the time since the move), or on the new line soon
          after (at most). Their spread says whether one lag fits every bet.
  held out  per-match lags fitted on half of each match's line bets, scored on the other half,
          against the operator's one lag on the same half: does a lag per match fit better?

A positive lag reads prod from before the bet (the operator runs behind); a negative one from after.
"""

import html
import math
import os
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import bets, config, markets

MIN_INFORMATIVE = 3


def lag_grid():
    lo, hi = config.LAG_RANGE
    return list(range(int(lo), int(hi) + 1, max(1, config.LAG_STEP_SECONDS)))


def line_steps(tl):
    """(match, market) -> (times, lines): prod's latest live line as a step, one entry per change."""
    out = {}
    for key, (times, quotes) in tl.items():
        ts, ls = [], []
        for t, q in zip(times, quotes):
            line = q[2]
            if ls and ls[-1] is not None and line is not None and abs(ls[-1] - line) < 1e-6:
                continue
            if ts and ts[-1] == t:
                ls[-1] = line
                continue
            ts.append(t)
            ls.append(line)
        out[key] = (ts, ls)
    return out


def line_at(step, s):
    """Prod's latest live line at time s, or None before its first quote."""
    ts, ls = step
    i = bisect_right(ts, s)
    return ls[i - 1] if i else None


@dataclass
class BetLags:
    """One in-play bet, lag by lag: whether its line is prod's (None: no line or no prod quote) and
    the log of its implied odds over prod's probability (None where it cannot be read)."""
    bet: object
    line: float
    fits: list
    logs: list


@dataclass
class MatchLag:
    """One match and operator: the lag read off the lines and off the odds."""
    match: str
    operator: str
    period_bets: Counter = field(default_factory=Counter)
    line_bets: int = 0
    informative: int = 0
    agree: list = field(default_factory=list)
    best: float = None
    plateau: tuple = (None, None)
    odds: list = field(default_factory=list)
    odds_best: float = None
    lower: list = field(default_factory=list)
    upper: list = field(default_factory=list)


def per_bet(in_play, steps, tl, signs, grid):
    """BetLags for every in-play bet with a time."""
    out = []
    for b in in_play:
        market = b.feed_market
        has_line = b.market_type in (2, 3)
        sign = signs.get((b.operator, market), (1,))[0]
        line = sign * b.line if (has_line and b.line is not None) else None
        step = steps.get((b.match_code, market))
        fits, logs = [], []
        for lag in grid:
            s = b.time - _delta(lag)
            if has_line and line is not None and step:
                prod_line = line_at(step, s)
                fits.append(None if prod_line is None else abs(prod_line - line) < 1e-6)
            else:
                fits.append(None)
            q = bets.price_at_time(tl, b.match_code, market, s)
            ok = (q is not None and b.odds and q[1] and 0 < q[1] < 1
                  and (not has_line or (line is not None and q[2] is not None
                                        and abs(q[2] - line) < 1e-6)))
            logs.append(math.log(1.0 / b.odds / q[1]) if ok else None)
        out.append(BetLags(b, line, fits, logs))
    return out


_DELTAS = {}


def _delta(lag):
    import datetime as dt
    d = _DELTAS.get(lag)
    if d is None:
        d = _DELTAS[lag] = dt.timedelta(seconds=lag)
    return d


def _bounds(bl, grid):
    """(at least, at most) the lag this bet allows, None where it allows every lag or none."""
    ok = [lag for lag, f in zip(grid, bl.fits) if f]
    if not ok or len(ok) == sum(f is not None for f in bl.fits):
        return None, None
    lo, hi = min(ok), max(ok)
    return (lo if lo > grid[0] else None), (hi if hi < grid[-1] else None)


def _agreement(items, grid):
    """Per lag, the share of these bets whose line is prod's (None with no line bet)."""
    n = [0] * len(grid)
    k = [0] * len(grid)
    for bl in items:
        for i, f in enumerate(bl.fits):
            if f is not None:
                n[i] += 1
                k[i] += f
    return [k[i] / n[i] if n[i] else None for i in range(len(grid))]


def _odds_curve(items, grid):
    """Per lag, the mean absolute log odds ratio less its median (the margin)."""
    out = []
    for i in range(len(grid)):
        xs = [bl.logs[i] for bl in items if bl.logs[i] is not None]
        if len(xs) < 5:
            out.append(None)
            continue
        mid = statistics.median(xs)
        out.append(sum(abs(x - mid) for x in xs) / len(xs))
    return out


def _best(curve, grid, higher=True, anchor=0.0):
    """The lag with the best value (ties to the one nearest anchor) and the plateau of lags within
    a hair of it."""
    pts = [(v, lag) for v, lag in zip(curve, grid) if v is not None]
    if not pts:
        return None, (None, None)
    top = max(v for v, _ in pts) if higher else min(v for v, _ in pts)
    tied = [lag for v, lag in pts if abs(v - top) < 1e-9]
    best = min(tied, key=lambda lag: abs(lag - anchor))
    return best, (min(tied), max(tied))


def per_match(bet_lags, grid, anchors):
    """(match, operator) -> MatchLag."""
    groups = defaultdict(list)
    for bl in bet_lags:
        groups[(bl.bet.match_code, bl.bet.operator)].append(bl)
    out = {}
    for (match, op), items in groups.items():
        m = MatchLag(match, op)
        lined = [bl for bl in items if any(f is not None for f in bl.fits)]
        m.line_bets = len(lined)
        for bl in lined:
            lo, hi = _bounds(bl, grid)
            if lo is not None:
                m.lower.append(lo)
            if hi is not None:
                m.upper.append(hi)
            if lo is not None or hi is not None:
                m.informative += 1
        m.agree = _agreement(lined, grid)
        anchor = anchors.get(op, 0.0)
        if m.informative:
            m.best, m.plateau = _best(m.agree, grid, True, anchor)
        m.odds = _odds_curve(items, grid)
        m.odds_best, _ = _best(m.odds, grid, False, anchor)
        m.period_bets = Counter(str(bl.bet.period) for bl in items)
        out[(match, op)] = m
    return out


def held_out(bet_lags, grid, anchors):
    """operator -> (bets scored, share on prod's line at the operator's lag, at each match's own lag
    fitted on the other half): the line bets of each match split alternately in time."""
    groups = defaultdict(list)
    for bl in bet_lags:
        if any(f is not None for f in bl.fits):
            groups[(bl.bet.match_code, bl.bet.operator)].append(bl)
    tally = defaultdict(lambda: [0, 0, 0])
    for (match, op), items in groups.items():
        items = sorted(items, key=lambda bl: bl.bet.time)
        anchor = anchors.get(op, 0.0)
        gi = grid.index(min(grid, key=lambda g: abs(g - anchor)))
        for fit_half, test_half in ((items[0::2], items[1::2]), (items[1::2], items[0::2])):
            if not any(_bounds(bl, grid) != (None, None) for bl in fit_half):
                lag_i = gi
            else:
                best, _ = _best(_agreement(fit_half, grid), grid, True, anchor)
                lag_i = grid.index(best)
            for bl in test_half:
                if bl.fits[gi] is None or bl.fits[lag_i] is None:
                    continue
                t = tally[op]
                t[0] += 1
                t[1] += bl.fits[gi]
                t[2] += bl.fits[lag_i]
    return {op: (n, a / n if n else None, b / n if n else None) for op, (n, a, b) in tally.items()}


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(q * (len(xs) - 1))))] if xs else None


def _gi(grid, lag):
    return grid.index(min(grid, key=lambda g: abs(g - lag)))


def summary(matches, bet_lags, grid, lags, held):
    """Lines of text: the per-match lags against the operator's one lag."""
    lines = []
    anchors = {op: lag.seconds for op, lag in lags.items()}
    ops = sorted({m.operator for m in matches.values()}, key=str)
    lined = defaultdict(list)
    for bl in bet_lags:
        if any(f is not None for f in bl.fits):
            lined[(bl.bet.match_code, bl.bet.operator)].append(bl)
    fmt = lambda xs, qs: "  ".join(f"{_pct(xs, q):+5.0f}" if xs else "    -" for q in qs)

    lines.append("\n  the lag match by match (seconds prod's quote is read back from each bet; + = the "
                 "operator runs behind; the best lag ties to the one nearest the operator's)")
    lines.append(f"  {'':22s} {'its lag':>7s} {'matches':>7s} {'line lag read':>13s}   "
                 f"line lag: 10th  25th median  75th  90th   odds lag: 10th median  90th")
    for op in ops:
        ms = [m for m in matches.values() if m.operator == op]
        line_best = [m.best for m in ms if m.best is not None and m.informative >= MIN_INFORMATIVE]
        odds_best = [m.odds_best for m in ms if m.odds_best is not None]
        lines.append(f"  {str(op)[:22]:22s} {anchors.get(op, 0):+7.0f} {len(ms):7,d} {len(line_best):13,d}   "
                     f"          {fmt(line_best, (0.1, 0.25, 0.5, 0.75, 0.9))}"
                     f"             {fmt(odds_best, (0.1, 0.5, 0.9))}")

    lines.append("\n  one lag per match? each bet that pins the lag says 'at least' or 'at most'; a match's bets "
                 f"agree when its largest 'at least' is under its smallest 'at most' ({MIN_INFORMATIVE}+ pinning bets)")
    lines.append(f"  {'':22s} {'matches':>7s} {'agree':>7s} {'hold its lag':>12s}   "
                 "lag range where they agree: from (median)  to (median)   disagree by (median 90th)")
    for op in ops:
        ms = [m for m in matches.values() if m.operator == op and m.informative >= MIN_INFORMATIVE]
        agree, holds, los, his, gaps = 0, 0, [], [], []
        for m in ms:
            lo = max(m.lower) if m.lower else grid[0]
            hi = min(m.upper) if m.upper else grid[-1]
            if lo <= hi:
                agree += 1
                los.append(lo)
                his.append(hi)
                holds += lo <= anchors.get(op, 0) <= hi
            else:
                gaps.append(lo - hi)
        n = len(ms)
        lines.append(f"  {str(op)[:22]:22s} {n:7,d} {100 * agree / n if n else 0:6.1f}% "
                     f"{100 * holds / n if n else 0:11.1f}%   "
                     f"{_pct(los, 0.5) if los else 0:+25.0f} {_pct(his, 0.5) if his else 0:+12.0f}"
                     f"   {_pct(gaps, 0.5) if gaps else 0:+13.0f} {_pct(gaps, 0.9) if gaps else 0:+4.0f}")

    lines.append("\n  line bets that pin the lag: struck on prod's old line after it moved (the lag is at least "
                 "...) or on the new line soon after (at most ...)")
    lines.append(f"  {'':22s} {'line bets':>9s} {'pinning':>8s}   at least: median  90th  >5s  >10s  >20s"
                 f"   at most: median  10th   <0s")
    for op in ops:
        items = [bl for key, bls in lined.items() if key[1] == op for bl in bls]
        lo, hi, pinning = [], [], 0
        for bl in items:
            a, b = _bounds(bl, grid)
            pinning += a is not None or b is not None
            if a is not None:
                lo.append(a)
            if b is not None:
                hi.append(b)
        share = lambda xs, cut: 100 * sum(x > cut for x in xs) / len(xs) if xs else 0.0
        lines.append(
            f"  {str(op)[:22]:22s} {len(items):9,d} {pinning:8,d}   "
            f"{_pct(lo, 0.5) if lo else 0:+15.0f} {_pct(lo, 0.9) if lo else 0:+5.0f} {share(lo, 5):4.0f}% "
            f"{share(lo, 10):4.0f}% {share(lo, 20):4.0f}%   {_pct(hi, 0.5) if hi else 0:+14.0f} "
            f"{_pct(hi, 0.1) if hi else 0:+5.0f} {100 * sum(x < 0 for x in hi) / len(hi) if hi else 0:5.0f}%")

    lines.append("\n  on prod's line: at 0s, at the operator's one lag, at each match's own (in sample), and held "
                 "out (per-match lag fitted on alternate bets, scored on the others)")
    lines.append(f"  {'':22s} {'line bets':>9s} {'at 0s':>7s} {'its lag':>8s} {'per match':>10s}   "
                 f"held out: {'bets':>7s} {'its lag':>8s} {'per match':>10s}")
    zi = _gi(grid, 0)
    for op in ops:
        items = [bl for key, bls in lined.items() if key[1] == op for bl in bls]
        gi = _gi(grid, anchors.get(op, 0))
        agree = _agreement(items, grid)
        own_n = own_k = 0
        for (match, o), bls in lined.items():
            if o != op:
                continue
            m = matches.get((match, op))
            i = grid.index(m.best) if (m is not None and m.best is not None) else gi
            for bl in bls:
                if bl.fits[i] is not None:
                    own_n += 1
                    own_k += bl.fits[i]
        h = held.get(op, (0, None, None))
        lines.append(f"  {str(op)[:22]:22s} {len(items):9,d} {100 * (agree[zi] or 0):6.1f}% "
                     f"{100 * (agree[gi] or 0):7.1f}% {100 * own_k / own_n if own_n else 0:9.1f}%   "
                     f"{h[0]:>17,d} {100 * (h[1] or 0):7.1f}% {100 * (h[2] or 0):9.1f}%")

    lines.append("\n  does the lag move through a match? line bets pooled by period")
    lines.append(f"  {'':22s} {'period':>7s} {'line bets':>9s} {'best lag':>9s} {'plateau':>11s} {'at its lag':>10s}")
    for op in ops:
        anchor = anchors.get(op, 0)
        gi = _gi(grid, anchor)
        by_p = defaultdict(list)
        for key, bls in lined.items():
            if key[1] == op:
                for bl in bls:
                    by_p[str(bl.bet.period)].append(bl)
        for p in sorted(by_p):
            ag = _agreement(by_p[p], grid)
            best, (lo, hi) = _best(ag, grid, True, anchor)
            plateau = "" if lo is None else f"{lo:+.0f}..{hi:+.0f}"
            lines.append(f"  {str(op)[:22]:22s} {p:>7s} {len(by_p[p]):9,d} "
                         f"{best if best is not None else 0:+9.0f} {plateau:>11s} {100 * (ag[gi] or 0):9.1f}%")
    return lines


def write_csv(path, matches, grid, anchors):
    fields = ["match_code", "operator", "line_bets", "informative", "line_lag", "plateau_lo",
              "plateau_hi", "agree_at_line_lag", "agree_at_operator_lag", "agree_at_0", "odds_lag",
              "at_least_max", "at_most_min"]
    rows = []
    for m in sorted(matches.values(), key=lambda m: (m.match, str(m.operator))):
        gi, zi = _gi(grid, anchors.get(m.operator, 0.0)), _gi(grid, 0)
        bi = grid.index(m.best) if m.best is not None else None
        rows.append(dict(match_code=m.match, operator=m.operator, line_bets=m.line_bets,
                         informative=m.informative, line_lag=m.best, plateau_lo=m.plateau[0],
                         plateau_hi=m.plateau[1],
                         agree_at_line_lag=None if bi is None else m.agree[bi],
                         agree_at_operator_lag=m.agree[gi] if m.agree else None,
                         agree_at_0=m.agree[zi] if m.agree else None, odds_lag=m.odds_best,
                         at_least_max=max(m.lower) if m.lower else None,
                         at_most_min=min(m.upper) if m.upper else None))
    bets.write_csv(path, rows, fields)


COLOURS = {"fits": "#2e7d32", "other lag": "#e65100", "never": "#c62828"}
PANELS = ((54, "total (over; unders on the same line)", (54, 55)), (52, "spread, home", (52,)),
          (53, "spread, away", (53,)))


def _dot_class(bl, gi):
    if bl.fits[gi]:
        return "fits"
    return "other lag" if any(bl.fits) else "never"


def _line_svg(match, title, ids, tl, bet_lags, gis, width=960, height=170):
    """Prod's live line as a step, each bet a dot at its time: green on prod's line at its
    operator's lag, orange only at another lag, red at none."""
    entry = tl.get((match, ids[0]))
    pts = [bl for bl in bet_lags if bl.bet.match_code == match and bl.bet.feed_market in ids
           and bl.line is not None]
    if not entry and not pts:
        return ""
    times, quotes = entry if entry else ([], [])
    all_t = list(times) + [bl.bet.time for bl in pts]
    vals = [q[2] for q in quotes if q[2] is not None] + [bl.line for bl in pts]
    if not all_t or not vals:
        return ""
    t0, t1 = min(all_t), max(all_t)
    span = max(1.0, (t1 - t0).total_seconds())
    lo, hi = min(vals) - 1, max(vals) + 1
    left, right, top, bottom = 46, 8, 20, 20
    x = lambda t: left + (width - left - right) * (t - t0).total_seconds() / span
    y = lambda v: top + (height - top - bottom) * (hi - v) / max(1e-9, hi - lo)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{left}" y="13" class="t">{html.escape(title)}</text>']
    step_v = max(1, int((hi - lo) / 5))
    v = math.floor(lo)
    while v <= hi:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>'
                     f'<text x="{left - 5}" y="{y(v) + 3:.1f}" class="a" text-anchor="end">{v:g}</text>')
        v += step_v
    for minute in range(0, int(span // 60) + 1, 5):
        xx = left + (width - left - right) * minute * 60 / span
        parts.append(f'<text x="{xx:.1f}" y="{height - 5}" class="a" text-anchor="middle">{minute}m</text>')
    prev = None
    for t, q in zip(times, quotes):
        if q[2] is None:
            continue
        if prev is not None:
            parts.append(f'<line x1="{x(prev[0]):.1f}" x2="{x(t):.1f}" y1="{y(prev[1]):.1f}" '
                         f'y2="{y(prev[1]):.1f}" class="s"/>')
            if prev[1] != q[2]:
                parts.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{y(prev[1]):.1f}" '
                             f'y2="{y(q[2]):.1f}" class="s"/>')
        prev = (t, q[2])
    if prev is not None:
        parts.append(f'<line x1="{x(prev[0]):.1f}" x2="{x(t1):.1f}" y1="{y(prev[1]):.1f}" '
                     f'y2="{y(prev[1]):.1f}" class="s"/>')
    for bl in pts:
        gi = gis.get(bl.bet.operator, 0)
        cls = _dot_class(bl, gi)
        ok = [g for g, f in zip(_GRID, bl.fits) if f]
        tip = (f"{bl.bet.time:%H:%M:%S} {markets.selection_label(bl.bet.feed_market)} {bl.line:g} "
               f"{bl.bet.operator}: {cls}" + (f" (fits {min(ok):+d}..{max(ok):+d}s)" if ok else ""))
        parts.append(f'<circle cx="{x(bl.bet.time):.1f}" cy="{y(bl.line):.1f}" r="3" fill="{COLOURS[cls]}" '
                     f'fill-opacity="0.75"><title>{html.escape(tip)}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


_GRID = []


def _curve_svg(ms, grid, anchors, width=960, height=120):
    """Per operator: the share of line bets on prod's line at each lag (solid) and the odds misfit,
    scaled to its own range (dashed), the operator's lag and the match's best marked."""
    left, right, top, bottom = 46, 8, 16, 18
    x = lambda lag: left + (width - left - right) * (lag - grid[0]) / max(1, grid[-1] - grid[0])
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="lag curves">',
             '<text x="46" y="11" class="t">share of line bets on prod\'s line by lag (solid), odds misfit '
             '(dashed, low is good); | the operator\'s lag, ▲ this match\'s best</text>']
    for lag in range(grid[0] - grid[0] % 10, grid[-1] + 1, 10):
        if lag < grid[0]:
            continue
        parts.append(f'<line x1="{x(lag):.1f}" x2="{x(lag):.1f}" y1="{top}" y2="{height - bottom}" class="g"/>'
                     f'<text x="{x(lag):.1f}" y="{height - 5}" class="a" text-anchor="middle">{lag:+d}s</text>')
    y = lambda v: top + (height - top - bottom) * (1 - v)
    for k, m in enumerate(ms):
        colour = ("#1565c0", "#6a1b9a", "#00838f")[k % 3]
        pts = [(x(lag), y(v)) for lag, v in zip(grid, m.agree) if v is not None]
        if pts:
            parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="'
                         + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        odds = [(lag, v) for lag, v in zip(grid, m.odds) if v is not None]
        if odds:
            lo_v, hi_v = min(v for _, v in odds), max(v for _, v in odds)
            pts = [(x(lag), y(1 - (v - lo_v) / max(1e-9, hi_v - lo_v))) for lag, v in odds]
            parts.append(f'<polyline fill="none" stroke="{colour}" stroke-width="1.5" stroke-dasharray="4 3" '
                         f'points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in pts) + '"/>')
        a = anchors.get(m.operator, 0.0)
        parts.append(f'<line x1="{x(a):.1f}" x2="{x(a):.1f}" y1="{top}" y2="{height - bottom}" '
                     f'stroke="{colour}" stroke-width="1"/>')
        if m.best is not None:
            parts.append(f'<text x="{x(m.best):.1f}" y="{top + 10 + 10 * k}" fill="{colour}" '
                         f'text-anchor="middle" font-size="11">▲</text>')
        parts.append(f'<text x="{width - right}" y="{top + 10 + 12 * k}" fill="{colour}" font-size="11" '
                     f'text-anchor="end">{html.escape(str(m.operator).replace("_BET_BY_BET", ""))}: line '
                     f'{"-" if m.best is None else f"{m.best:+.0f}s"}, odds '
                     f'{"-" if m.odds_best is None else f"{m.odds_best:+.0f}s"}</text>')
    parts.append("</svg>")
    return "".join(parts)


STYLE = """
:root { --bg:#fff; --ink:#1a1a1a; --muted:#666; --grid:#e6e6e6; --step:#1a1a1a; }
@media (prefers-color-scheme: dark) { :root { color-scheme:dark; --bg:#141414; --ink:#e8e8e8;
  --muted:#999; --grid:#2a2a2a; --step:#e8e8e8; } }
body { background:var(--bg); color:var(--ink); font-family:Helvetica,Arial,sans-serif; margin:0;
  padding:20px 16px; }
main { max-width:1000px; margin:0 auto; } pre { font-size:12px; overflow-x:auto; }
h2 { font-size:15px; margin:26px 0 4px; } a { color:inherit; }
.t { font-size:12px; fill:var(--ink); } .a { font-size:10px; fill:var(--muted); }
.g { stroke:var(--grid); } .s { stroke:var(--step); stroke-width:2; }
.key span { display:inline-block; margin-right:14px; font-size:13px; }
.key i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
"""


def _page(title, body):
    key = "".join(f'<span><i style="background:{c}"></i>{k}</span>' for k, c in COLOURS.items())
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{STYLE}</style></head><body><main><h1>{html.escape(title)}</h1>"
            f"<p class='key'>{key} &nbsp; bets on prod's line at their operator's lag / only at another "
            f"lag / at none; prod's live line: dark step</p>{body}</main></body></html>")


def write_pages(out_dir, matches, bet_lags, tl, grid, anchors, summary_lines):
    """bets_lag.html (the summary and a table of every match) and bets_lag_<day>.html (every match of
    that day drawn)."""
    global _GRID
    _GRID = grid
    gis = {op: grid.index(min(grid, key=lambda g: abs(g - a))) for op, a in anchors.items()}
    by_match = defaultdict(list)
    for m in matches.values():
        by_match[m.match].append(m)
    by_day = defaultdict(list)
    from .bet_checks import _day
    for match in sorted(by_match):
        by_day[_day(match)].append(match)
    bets_by_match = defaultdict(list)
    for bl in bet_lags:
        bets_by_match[bl.bet.match_code].append(bl)
    paths = []
    rows = []
    for day, codes in sorted(by_day.items()):
        body = []
        for match in codes:
            ms = sorted(by_match[match], key=lambda m: str(m.operator))
            body.append(f"<h2 id='{html.escape(match)}'>{html.escape(match)}</h2>")
            body.append(_curve_svg(ms, grid, anchors))
            for market, title, ids in PANELS:
                body.append(_line_svg(match, title, ids, tl, bets_by_match[match], gis))
            for m in ms:
                rows.append(f"<tr><td><a href='bets_lag_{day}.html#{html.escape(match)}'>{html.escape(match)}</a></td>"
                            f"<td>{html.escape(str(m.operator).replace('_BET_BY_BET', ''))}</td>"
                            f"<td>{m.line_bets}</td><td>{m.informative}</td>"
                            f"<td>{'' if m.best is None else f'{m.best:+.0f}'}</td>"
                            f"<td>{'' if m.plateau[0] is None else f'{m.plateau[0]:+.0f}..{m.plateau[1]:+.0f}'}</td>"
                            f"<td>{'' if m.odds_best is None else f'{m.odds_best:+.0f}'}</td></tr>")
        path = os.path.join(out_dir, f"bets_lag_{day}.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(_page(f"Bet lag {day}", "\n".join(body)))
        paths.append(path)
    index = ("<pre>" + html.escape("\n".join(summary_lines)) + "</pre>"
             "<p>" + " · ".join(f"<a href='bets_lag_{d}.html'>{d}</a>" for d in sorted(by_day)) + "</p>"
             "<table><thead><tr><th>match</th><th>operator</th><th>line bets</th><th>pinning</th>"
             "<th>line lag</th><th>plateau</th><th>odds lag</th></tr></thead><tbody>"
             + "".join(rows) + "</tbody></table>")
    path = os.path.join(out_dir, "bets_lag.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_page("Bet lag, match by match", index))
    return [path] + paths


def run(cur, out_dir):
    """Fetch the window's bets and prod's quotes, read the lag match by match, print the summary and
    write bets_lag.csv and the pages."""
    from . import snowflake_io
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    all_bets = bets.to_bets(cols, raw)
    in_play = [b for b in all_bets if b.in_play and b.time is not None and b.feed_market]
    matches = sorted({b.match_code for b in in_play})
    print(f"  {len(all_bets):,} bets, {len(in_play):,} in play on {len(matches):,} matches")
    prod_rows = []
    for start in range(0, len(matches), config.MATCH_CHUNK_SIZE):
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"],
                                               matches[start:start + config.MATCH_CHUNK_SIZE])
    tl = bets.timeline(prod_rows)
    del prod_rows
    lags = bets.fit_lags(all_bets, tl)
    signs = bets.fit_line_signs(all_bets, tl, lags)
    anchors = {op: lag.seconds for op, lag in lags.items()}
    grid = lag_grid()
    print(f"  reading every in-play bet at {len(grid)} lags, {grid[0]:+d}s to {grid[-1]:+d}s", flush=True)
    bet_lags = per_bet(in_play, line_steps(tl), tl, signs, grid)
    ms = per_match(bet_lags, grid, anchors)
    held = held_out(bet_lags, grid, anchors)
    lines = summary(ms, bet_lags, grid, lags, held)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "bets_lag.csv"), ms, grid, anchors)
    paths = write_pages(out_dir, ms, bet_lags, tl, grid, anchors, lines)
    return paths
