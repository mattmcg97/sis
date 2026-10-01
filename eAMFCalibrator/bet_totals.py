"""Restricted accounts on totals: how prod's line moved either side of each bet, and whether the
models already sat where the line went.

  path     every totals bet against prod's live line and price at fixed offsets either side of
           the moment it was priced (bet time less the operator's lag), signed with the bet
  before   did the bet follow a move (prod had moved its way into the bet), fade one, or come
           on a still line
  after    did prod move the bettor's way after, and was there a score in between: a move with
           no score is the price drifting, one across a score is the game
  model    each candidate's line and price at the message the bet saw: did it already lean the
           bettor's way, and did its line sit nearer where prod's went than prod's own did

The sign everywhere: level = side x (line then - line at the bet), side +1 for Over and -1 for
Under. Positive means the market rated the bettor's side higher then than at the bet: after
the bet that is a move their way (they beat it); before it, a move away (they faded it). Where
the line held, the bet selection's probability carries the move instead, never pooled with it.

Restricted is CUSTOMER_TEMPERATURE = config.BET_RESTRICTED_VALUE. Everyone else on the same
matches is the baseline, so a pattern is only one where it differs from theirs.
"""

import datetime as dt
import html
import math
import os
from collections import Counter, defaultdict

from . import bet_moments, bets, config, markets, snowflake_io

SIDE = {54: 1, 55: -1}
OFFSETS = (-120, -60, -30, -10, 10, 30, 60, 120, 300)
PROB_FLAT = 0.01
MIN_BETS = 30
RESTRICTED, OTHERS = "Restricted", "Everyone else"
GROUPS = (RESTRICTED, OTHERS)
FOR, AGAINST, HELD = "their way", "against them", "held"
FOLLOWED, FADED, STILL = "followed a move", "faded a move", "still line"
NEEDED_EDGES = (7, 14, 21, 28)


def tag(k):
    """-60 -> 'm60', 60 -> 'p60': an offset as a column suffix."""
    return f"m{-k}" if k < 0 else f"p{k}"


def group_of(row):
    t = str(row.get(config.BET_VIP_COLUMN) or "").strip().lower()
    return RESTRICTED if t == config.BET_RESTRICTED_VALUE.lower() else OTHERS


def _quote(tl, match, market, t):
    """(line, probability) of the latest live quote at or before t, or None."""
    q = bets.price_at_time(tl, match, market, t)
    return None if q is None or q[2] is None else (q[2], q[1])


def level(side, ref, q):
    """(line level, probability level) of quote q against the reference: side x the line's change,
    and the selection's probability change where the line held (None where it did not)."""
    if ref is None or q is None:
        return None, None
    lv = side * (q[0] - ref[0])
    same = abs(q[0] - ref[0]) < 1e-6
    return lv, (q[1] - ref[1] if same and q[1] is not None and ref[1] is not None else None)


def direction(lv, pv):
    """+1 / -1 / 0 off a level: the line where it moved, else the probability past PROB_FLAT."""
    if lv is None:
        return None
    if abs(lv) > 1e-6:
        return 1 if lv > 0 else -1
    if pv is not None and abs(pv) >= PROB_FLAT:
        return 1 if pv > 0 else -1
    return 0


def after_label(d):
    return None if d is None else {1: FOR, -1: AGAINST, 0: HELD}[d]


def before_label(d):
    return None if d is None else {1: FADED, -1: FOLLOWED, 0: STILL}[d]


def message_at(checks, match, t):
    """prod's latest message published at or before t, or None."""
    from bisect import bisect_right
    entry = checks.times.get(match)
    if entry is None or t is None:
        return None
    msgs, ts = entry
    i = bisect_right(ts, t)
    return msgs[i - 1] if i else None


def points(checks, match, message):
    return sum(checks.score_at(match, message)) if message is not None else 0


def next_score_time(checks, match, message):
    """prod's time of the first score after this message, or None."""
    from bisect import bisect_right
    entry = checks.scores.get(match)
    if entry is None or message is None:
        return None
    smsgs, vals = entry
    now = checks.score_at(match, message)
    k = bisect_right(smsgs, message)
    while k < len(smsgs) and vals[k] == now:
        k += 1
    return checks.time_of(match, smsgs[k]) if k < len(smsgs) else None


def kickoff_time(checks, match):
    f = checks.feeds.get(match)
    return checks.time_of(match, f.start) if f is not None and f.start is not None else None


def needed_band(n):
    if n is None:
        return "unknown"
    lo = None
    for e in NEEDED_EDGES:
        if n < e:
            return f"<{e}" if lo is None else f"{lo}-{e}"
        lo = e
    return f"{lo}+"


def path(b, r, tl, checks):
    """The prod line path around one totals bet (r: its bets.join row)."""
    side, market, match = SIDE[b.feed_market], b.feed_market, b.match_code
    seen = None if b.time is None else b.time - dt.timedelta(seconds=r["latency_seconds"] or 0)
    ref = None if r["stream_line"] is None else (r["stream_line"], r["stream_prob"])
    msg = r["message"]
    board = points(checks, match, msg)
    out = dict(group=group_of(r), side=markets.selection_label(market), seen_time=seen,
               on_board=board if ref else None,
               points_needed=None if ref is None else ref[0] - board)
    out["needed_band"] = needed_band(out["points_needed"])
    final = (r.get("final_p1"), r.get("final_p2"))
    out["final_total"] = None if None in final else final[0] + final[1]
    out["landed"] = (None if out["final_total"] is None or b.line is None or r["bet_line_prod_side"] is None
                     else side * (out["final_total"] - r["bet_line_prod_side"]))
    for k in OFFSETS:
        t = None if seen is None else seen + dt.timedelta(seconds=k)
        lv, pv = level(side, ref, _quote(tl, match, market, t))
        d = direction(lv, pv)
        m = message_at(checks, match, t)
        scored = None if m is None or msg is None else abs(points(checks, match, m) - board)
        out.update({f"level_{tag(k)}": lv, f"plevel_{tag(k)}": pv, f"scored_{tag(k)}": scored,
                    f"way_{tag(k)}": before_label(d) if k < 0 else after_label(d)})
    lv, pv = level(side, ref, _quote(tl, match, market, b.time))
    out.update(level_bet=lv, plevel_bet=pv, way_bet=after_label(direction(lv, pv)))
    eps = dt.timedelta(milliseconds=1)
    score_t = next_score_time(checks, match, msg) if b.in_play else None
    out["next_score_seconds"] = (None if score_t is None or seen is None
                                 else (score_t - seen).total_seconds())
    lv, pv = level(side, ref, _quote(tl, match, market, score_t - eps)) if score_t else (None, None)
    out.update(level_pre_score=lv, way_pre_score=after_label(direction(lv, pv)))
    ko = None if b.in_play else kickoff_time(checks, match)
    lv, pv = level(side, ref, _quote(tl, match, market, ko - eps)) if ko and seen and ko > seen \
        else (None, None)
    out.update(level_kickoff=lv, plevel_kickoff=pv, way_kickoff=after_label(direction(lv, pv)))
    return out


def comparable(r):
    """A candidate quote that carries the same information as prod's at the bet's message."""
    return (r.get("candidate_prob") is not None and r.get("candidate_line") is not None
            and r.get("candidate_ok") is not False
            and not (r.get("state_required") and not r.get("state_ok")))


def model_fields(name, b, r, base, cand_tl):
    """One candidate against prod at the bet: its lean, and whether its line sat where prod's
    went."""
    side, match, market = SIDE[b.feed_market], b.match_code, b.feed_market
    keys = ("ok", "line", "prob", "line_edge", "prob_edge", "lean", "nearer_p60", "nearer_p120",
            "level_p60", "change")
    out = {f"{name}_{k}": None for k in keys}
    out[f"{name}_ok"] = comparable(r) and base["stream_line"] is not None
    if not out[f"{name}_ok"]:
        return out
    ref = (base["stream_line"], base["stream_prob"])
    c = (r["candidate_line"], r["candidate_prob"])
    lv, pv = level(side, ref, c)
    out.update({f"{name}_line": c[0], f"{name}_prob": c[1], f"{name}_line_edge": lv,
                f"{name}_prob_edge": pv, f"{name}_lean": after_label(direction(lv, pv))})
    seen = base["seen_time"]
    for k in (60, 120):
        later = base.get(f"level_{tag(k)}")
        if later is None or abs(later) < 1e-6:
            continue
        prod_gap, cand_gap = abs(later), abs(later - lv)
        out[f"{name}_nearer_p{k}"] = ("nearer" if cand_gap < prod_gap - 1e-6
                                      else "further" if cand_gap > prod_gap + 1e-6 else "as far")
    c0 = _quote(cand_tl, match, market, seen)
    c1 = _quote(cand_tl, match, market, seen + dt.timedelta(seconds=60)) if seen else None
    out[f"{name}_level_p60"] = level(side, c0, c1)[0]
    if r.get("simulated") and r.get("candidate_revenue") is not None:
        out[f"{name}_change"] = r["candidate_revenue"] - r["revenue"]
    return out


def build(totals, base_rows, results, prod_tl, checks):
    """One row per totals bet: the bets.join row, the path, the moment and every candidate."""
    rows = []
    for k, b in enumerate(totals):
        r = dict(base_rows[k])
        for f in bets.CANDIDATE_FIELDS:
            r.pop(f, None)
        r.update(path(b, base_rows[k], prod_tl, checks))
        if b.in_play:
            r.update(bet_moments.moment_of(checks.timelines.get(b.match_code), checks, b.match_code,
                                           b.time, base_rows[k]["message"]))
        else:
            r["moment"] = "pre-match"
        for name, joined, cand_tl in results:
            r.update(model_fields(name, b, joined[k], r, cand_tl))
        rows.append(r)
    return rows


# --- summaries -------------------------------------------------------------------------------

def in_scope(r):
    """A bet the path is read on: settled, the checks keeping it, on prod's live line."""
    return (not r.get("excluded") and r.get("result") in (bets.WON, bets.LOST, bets.PUSH)
            and r.get("stake") and r.get("on_prod_line") and r.get("stream_line") is not None)


def mean_se(rows, key, cluster="match_code"):
    """(n, mean, standard error clustered by match) of key over the rows where it is not None."""
    xs = [(r[cluster], r[key]) for r in rows if r.get(key) is not None]
    n = len(xs)
    if not n:
        return 0, None, None
    m = sum(x for _, x in xs) / n
    sums = defaultdict(float)
    for g, x in xs:
        sums[g] += x - m
    g = len(sums)
    se = math.sqrt(sum(s * s for s in sums.values()) * g / (g - 1)) / n if g > 1 else None
    return n, m, se


def shares(rows, key, labels):
    c = Counter(r.get(key) for r in rows if r.get(key) is not None)
    n = sum(c.values())
    return n, [100 * c[x] / n if n else None for x in labels]


def money(rows):
    """(bets, stake, margin %, expected margin %): what the book kept, and what prod's probability
    says it should have."""
    stake = sum(r["stake"] for r in rows)
    rev = sum(r["revenue"] for r in rows)
    priced = [r for r in rows if r.get("stream_prob") and r.get("odds")]
    ps = sum(r["stake"] for r in priced)
    exp = sum(r["stake"] * (1 - r["odds"] * r["stream_prob"]) for r in priced) / ps if ps else None
    return len(rows), stake, 100 * rev / stake if stake else None, None if exp is None else 100 * exp


def _f(x, fmt="+.2f", width=7):
    return f"{'-':>{width}s}" if x is None else f"{x:{fmt}}".rjust(width)


def population(rows):
    """{group: (bets, customers, stake, margin, expected, over %, in play %, on prod's line %)}."""
    out = {}
    for g in GROUPS:
        rs = [r for r in rows if r["group"] == g and not r.get("excluded")
              and r.get("result") in (bets.WON, bets.LOST, bets.PUSH)]
        if not rs:
            continue
        n, stake, m, exp = money(rs)
        out[g] = (n, len({r.get("CUSTOMER_NAME_HASH") for r in rs}), stake, m, exp,
                  100 * sum(r["side"] == "Over" for r in rs) / n,
                  100 * sum(bool(r["in_play"]) for r in rs) / n,
                  100 * sum(bool(r.get("on_prod_line")) for r in rs) / n)
    return out


def event_study(rows):
    """[(offset label, mean line level, 2se, % their way, % against)] over the rows, one per offset
    and the moment the bet was struck."""
    cols = [(tag(k), f"{k:+d}s") for k in OFFSETS if k < 0] + [("bet", "at bet")] + \
           [(tag(k), f"{k:+d}s") for k in OFFSETS if k > 0]
    out = []
    for t, text in cols:
        n, m, se = mean_se(rows, f"level_{t}")
        lab = (FADED, FOLLOWED) if t.startswith("m") else (FOR, AGAINST)
        _, (up, down) = shares(rows, f"way_{t}", lab)
        out.append((text, n, m, None if se is None else 2 * se, up, down))
    return out


def cut(rows, key, min_bets):
    """{bucket: (money, n, mean level +60s, 2se, % their way +60s, % against +60s, mean level +120s)}."""
    groups = defaultdict(list)
    for r in rows:
        g = key(r) if callable(key) else r.get(key)
        groups["unknown" if g is None else g].append(r)
    out = {}
    for g, rs in groups.items():
        if len(rs) < min_bets:
            continue
        n, m, se = mean_se(rs, "level_p60")
        _, (up, down, _) = shares(rs, "way_p60", (FOR, AGAINST, HELD))
        out[g] = (money(rs), n, m, None if se is None else 2 * se, up, down,
                  mean_se(rs, "level_p120")[1])
    return out


def scored_split(r, k=60):
    s = r.get(f"scored_{tag(k)}")
    return None if s is None else "no score in between" if s == 0 else "a score in between"


def candidate_names(rows):
    return [k[:-len("_lean")] for k in (rows[0] if rows else {}) if k.endswith("_lean")]


def model_table(rows, name):
    """{(group, prod after +60s): (n, lean their way %, same %, against %, mean line edge, 2se,
    mean prob edge at the same line, nearer prod's +120s line %, its own level +60s, margin
    change)} over the bets where the candidate carries the same information."""
    out = {}
    for g in GROUPS:
        base = [r for r in rows if r["group"] == g and r.get(f"{name}_ok")]
        for after in (None, FOR, AGAINST, HELD):
            rs = [r for r in base if after is None or r.get("way_p60") == after]
            if not rs:
                continue
            n, (up, same, down) = shares(rs, f"{name}_lean", (FOR, HELD, AGAINST))
            _, m, se = mean_se(rs, f"{name}_line_edge")
            _, pm, _ = mean_se(rs, f"{name}_prob_edge")
            _, (nearer,) = shares(rs, f"{name}_nearer_p120", ("nearer",))
            _, own, _ = mean_se(rs, f"{name}_level_p60")
            changed = [r for r in rs if r.get(f"{name}_change") is not None]
            stake = sum(r["stake"] for r in changed)
            change = 100 * sum(r[f"{name}_change"] for r in changed) / stake if stake else None
            out[(g, after or "all")] = (n, up, same, down, m, None if se is None else 2 * se, pm,
                                        nearer, own, change)
    return out


def customers(rows, n=15):
    """The restricted customers with the most stake: (customer, bets, stake, margin, over %, mean
    level +60s, % their way +60s)."""
    by = defaultdict(list)
    for r in rows:
        if r["group"] == RESTRICTED:
            by[r.get("CUSTOMER_NAME_HASH")].append(r)
    out = []
    for c, rs in sorted(by.items(), key=lambda kv: -sum(r["stake"] for r in kv[1]))[:n]:
        k, stake, m, _ = money(rs)
        _, (up,) = shares(rs, "way_p60", (FOR,))
        out.append((str(c)[:12], k, stake, m, 100 * sum(r["side"] == "Over" for r in rs) / k,
                    mean_se(rs, "level_p60")[1], up))
    return out


CUTS = (("side", "side"), ("in play", lambda r: "in play" if r["in_play"] else "pre-match"),
        ("quarter", lambda r: r.get("quarter") or ("pre-match" if not r["in_play"] else "unknown")),
        ("points still needed", "needed_band"), ("feed at bet time", "moment"),
        ("price age", "price_age"), ("before the bet (-60s)", "way_m60"),
        ("score between bet and +60s", scored_split), ("operator", config.BET_GROUP_COLUMN))


def report(rows, min_bets=MIN_BETS):
    """Lines of text: every table, restricted against everyone else."""
    scope = [r for r in rows if in_scope(r)]
    L = ["\n  totals bets by customer temperature (settled, the checks keeping them)",
         f"  {'':16s} {'bets':>7s} {'custs':>6s} {'stake':>11s} {'margin':>8s} {'expected':>9s} "
         f"{'over':>6s} {'in play':>8s} {'on line':>8s}"]
    for g, (n, c, stake, m, exp, over, ip, online) in population(rows).items():
        L.append(f"  {g:16s} {n:7,d} {c:6,d} {stake:11,.0f} {_f(m, '.2f', 7)}% {_f(exp, '.2f', 8)}% "
                 f"{over:5.1f}% {ip:7.1f}% {online:7.1f}%")
    L.append("\n  prod's line either side of the bet, signed with the bet (+: the market rates the "
             "bettor's side higher then than at the bet; before the bet that is a move they faded, "
             "after it a move their way). Points, mean +-2se clustered by match; the shares are "
             "the line, else the probability past 1 point")
    for g in GROUPS:
        for side in (None, "Over", "Under"):
            rs = [r for r in scope if r["group"] == g and (side is None or r["side"] == side)]
            if len(rs) < min_bets:
                continue
            L.append(f"\n  {g}{'' if side is None else ' ' + side} ({len(rs):,} bets)")
            L.append(f"  {'':8s} {'bets':>7s} {'level':>7s} {'+-2se':>6s} {'up':>6s} {'down':>6s}"
                     "   (up/down: faded/followed before, their way/against after)")
            for text, n, m, se2, up, down in event_study(rs):
                L.append(f"  {text:8s} {n:7,d} {_f(m)} {_f(se2, '.2f', 6)} {_f(up, '.1f', 5)}% "
                         f"{_f(down, '.1f', 5)}%")
    L.append("\n  pre-match bets: prod's closing line (the last before kick-off) against the bet's")
    for g in GROUPS:
        rs = [r for r in scope if r["group"] == g and not r["in_play"]]
        n, m, se = mean_se(rs, "level_kickoff")
        if n:
            _, (up, down) = shares(rs, "way_kickoff", (FOR, AGAINST))
            L.append(f"  {g:16s} {n:7,d} bets  level {_f(m)} +-{_f(None if se is None else 2 * se, '.2f', 5)}"
                     f"  their way {_f(up, '.1f', 5)}%  against {_f(down, '.1f', 5)}%")
    L.append("\n  before (-60s) against after (+60s): share of each group's bets, and the book's "
             "margin on them")
    for g in GROUPS:
        rs = [r for r in scope if r["group"] == g and r.get("way_m60") and r.get("way_p60")]
        if not rs:
            continue
        L.append(f"  {g} ({len(rs):,} bets), after +60s:")
        L.append(f"  {'before -60s':24s} " + "  ".join(f"{a:>13s}" for a in (FOR, AGAINST, HELD)))
        for b_ in (FOLLOWED, FADED, STILL):
            cells = []
            for a in (FOR, AGAINST, HELD):
                sub = [r for r in rs if r["way_m60"] == b_ and r["way_p60"] == a]
                m = money(sub)[2] if sub else None
                cells.append(f"{100 * len(sub) / len(rs):5.1f}% {_f(m, '+.0f', 5)}%")
            L.append(f"  {b_:24s} " + "  ".join(cells))
    for title, key in CUTS:
        L.append(f"\n  by {title}: the book's margin and prod's line after the bet "
                 "(level +60s, +-2se, their way / against, level +120s)")
        L.append(f"  {'':16s} {'':22s} {'bets':>6s} {'stake':>10s} {'margin':>8s} {'expect':>7s} "
                 f"{'+60s':>6s} {'+-2se':>6s} {'their':>6s} {'agnst':>6s} {'+120s':>6s}")
        for g in GROUPS:
            table = cut([r for r in scope if r["group"] == g], key, min_bets)
            for bucket in sorted(table, key=str):
                (n, stake, m, exp), _, lv, se2, up, down, lv2 = table[bucket]
                L.append(f"  {g:16s} {str(bucket)[:22]:22s} {n:6,d} {stake:10,.0f} {_f(m, '.2f', 7)}% "
                         f"{_f(exp, '.2f', 6)}% {_f(lv, '+.2f', 6)} {_f(se2, '.2f', 6)} "
                         f"{_f(up, '.1f', 5)}% {_f(down, '.1f', 5)}% {_f(lv2, '+.2f', 6)}")
    L.append("\n  the restricted customers with the most stake")
    L.append(f"  {'customer':12s} {'bets':>6s} {'stake':>10s} {'margin':>8s} {'over':>6s} "
             f"{'+60s':>6s} {'their':>6s}")
    for c, n, stake, m, over, lv, up in customers(scope):
        L.append(f"  {c:12s} {n:6,d} {stake:10,.0f} {_f(m, '.2f', 7)}% {over:5.1f}% {_f(lv, '+.2f', 6)} "
                 f"{_f(up, '.1f', 5)}%")
    for name in candidate_names(rows):
        L.append(f"\n  {name} at the message each bet saw, where it carries the same information as "
                 "prod's: does it lean the bettor's way (its line, else its probability at the same "
                 "line), and where prod moved by +120s, was its line nearer where prod went?")
        L.append(f"  {'':16s} {'prod +60s':12s} {'bets':>6s} {'lean':>6s} {'same':>6s} {'agnst':>6s} "
                 f"{'edge':>6s} {'+-2se':>6s} {'p edge':>7s} {'nearer':>7s} {'own+60':>7s} {'margin chg':>10s}")
        for (g, after), (n, up, same, down, m, se2, pm, nearer, own, change) in \
                model_table([r for r in rows if in_scope(r)], name).items():
            L.append(f"  {g:16s} {after:12s} {n:6,d} {_f(up, '.1f', 5)}% {_f(same, '.1f', 5)}% "
                     f"{_f(down, '.1f', 5)}% {_f(m, '+.2f', 6)} {_f(se2, '.2f', 6)} "
                     f"{_f(None if pm is None else 100 * pm, '+.2f', 7)} {_f(nearer, '.1f', 6)}% "
                     f"{_f(own, '+.2f', 7)} {_f(change, '+.2f', 9)}")
    return L


# --- HTML ------------------------------------------------------------------------------------

def _esc(x):
    return html.escape(str(x))


def chart(scope):
    """The mean signed line level at each offset, restricted against everyone else, as SVG."""
    series = []
    for g, colour in ((RESTRICTED, "var(--bad)"), (OTHERS, "var(--accent)")):
        rs = [r for r in scope if r["group"] == g]
        if rs:
            series.append((g, colour, event_study(rs)))
    if not series:
        return ""
    labels = [p[0] for p in series[0][2]]
    vals = [v for _, _, pts in series for _, _, m, se2, _, _ in pts if m is not None
            for v in (m - (se2 or 0), m + (se2 or 0))] + [0.0]
    lo, hi = min(vals), max(vals)
    pad = max(0.05, 0.1 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    w, h, left, right, top, bottom = 760, 260, 54, 110, 14, 30
    x = lambda i: left + (w - left - right) * i / (len(labels) - 1)
    y = lambda v: top + (h - top - bottom) * (hi - v) / (hi - lo)
    p = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Mean signed line level '
         'around the bet">']
    for i, text in enumerate(labels):
        p.append(f'<text x="{x(i):.1f}" y="{h - 10}" class="ax" text-anchor="middle">{_esc(text)}</text>')
    step = 10 ** math.floor(math.log10(max(hi - lo, 1e-6) / 2))
    v = math.ceil(lo / step) * step
    while v <= hi:
        p.append(f'<line x1="{left}" x2="{w - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" '
                 f'class="{"zero" if abs(v) < step / 2 else "grid"}"/>'
                 f'<text x="{left - 6}" y="{y(v) + 4:.1f}" class="ax" text-anchor="end">{v:+.2g}</text>')
        v += step
    for g, colour, pts in series:
        xy = [(x(i), y(m), m, se2, n) for i, (_, n, m, se2, _, _) in enumerate(pts) if m is not None]
        p.append(f'<polyline fill="none" stroke="{colour}" stroke-width="2" points="'
                 + " ".join(f"{a:.1f},{b:.1f}" for a, b, *_ in xy) + '"/>')
        for a, b, m, se2, n in xy:
            if se2:
                p.append(f'<line x1="{a:.1f}" x2="{a:.1f}" y1="{y(m - se2):.1f}" y2="{y(m + se2):.1f}" '
                         f'stroke="{colour}" stroke-width="1.5" stroke-opacity="0.5"/>')
            p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{colour}" stroke="var(--panel)" '
                     f'stroke-width="2"><title>{_esc(g)}: {m:+.3f} pts '
                     f'(&#177;{se2 or 0:.3f}, {n:,} bets)</title></circle>')
        if xy:
            p.append(f'<text x="{xy[-1][0] + 8:.1f}" y="{xy[-1][1] + 4:.1f}" class="lab">{_esc(g)}</text>')
    p.append("</svg>")
    return "".join(p)


def page(lines, scope, window):
    from .html_style import CSS
    style = ("<style>.ax{font-size:10px;fill:var(--dim)}.lab{font-size:11px;fill:var(--ink)}"
             ".grid{stroke:var(--line)}.zero{stroke:var(--dim)}"
             "pre{font-size:11.5px;overflow-x:auto}</style>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Restricted Totals Moves</title>{CSS}{style}</head><body><div class='wrap'>"
            f"<header><h1>Totals bets: prod's line around the bet</h1>"
            f"<span class='meta'>{_esc(window)}</span></header>"
            "<section class='panel'><h2>Mean signed line level, points</h2>"
            "<p class='meta'>+ is the market rating the bettor's side higher than at the bet. "
            "Left of the bet: a move they faded; right: a move their way. Bars &#177;2se by match.</p>"
            f"{chart(scope)}</section>"
            f"<section class='panel'><h2>Tables</h2><pre>{_esc(chr(10).join(lines))}</pre></section>"
            "</div></body></html>")


# --- run -------------------------------------------------------------------------------------

def run(cur, out_dir, min_bets=MIN_BETS):
    """Fetch every totals bet of the window and prod's quotes, read prod's line around each, price
    the candidates at the message each bet saw, then print, and write the CSV and the page."""
    bets.check_models(bets.candidate_streams())
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    every = bets.to_bets(cols, raw)
    totals = [b for b in every if b.feed_market in SIDE]
    matches = sorted({b.match_code for b in totals})
    restricted = sum(group_of(b.extra) == RESTRICTED for b in totals)
    print(f"  {len(totals):,} totals bets on {len(matches):,} matches, {restricted:,} from "
          f"{config.BET_VIP_COLUMN} = {config.BET_RESTRICTED_VALUE}")
    if not totals:
        return None
    prod_rows, finals = [], {}
    for start in range(0, len(matches), config.MATCH_CHUNK_SIZE):
        batch = matches[start:start + config.MATCH_CHUNK_SIZE]
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        finals.update(snowflake_io.fetch_final_scores(cur, batch))
    checks = bets.fetch_checks(cur, matches, prod_rows)
    prod_tl = bets.timeline(prod_rows)
    del prod_rows
    wanted = set(matches)
    lags = bets.fit_lags([b for b in every if b.match_code in wanted], prod_tl)
    signs = bets.fit_line_signs(totals, prod_tl, lags)
    results = []
    for stream in bets.candidate_streams():
        print(f"  pricing the totals bets with {bets.label(stream)}", flush=True)
        cand_rows = bets.candidate_quotes(cur, stream, matches)
        cand_tl = bets.timeline(cand_rows)
        joined = bets.join(totals, lags, signs, prod_tl, bets.quote_index(cand_rows), cand_tl,
                           finals, checks, same_state=snowflake_io.is_model(stream))
        results.append((bets.label(stream).replace(".", "_"), joined, cand_tl))
    rows = build(totals, results[0][1], results, prod_tl, checks)
    lines = report(rows, min_bets)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    path_csv = os.path.join(out_dir, "bets_totals_moves.csv")
    bets.write_csv(path_csv, rows, list(dict.fromkeys(k for r in rows for k in r)))
    window = f"{config.CUTOFF_START} to {config.CUTOFF_END or 'latest'}"
    path_html = os.path.join(out_dir, "bets_totals_moves.html")
    with open(path_html, "w", encoding="utf-8") as fh:
        fh.write(page(lines, [r for r in rows if in_scope(r)], window))
    return path_csv, path_html
