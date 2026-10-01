"""Where the money goes through a gamer's session: bets and stake by the gamer's match of the
session (1st, 2nd ... 10th+) and by matches left in it, set against how many of the window's
matches sit there.

A gamer's session is a run of matches with no gap over config.SESSION_BREAK_MINUTES. Every bet
counts once for each of the two gamers in its match, at that gamer's place in his own session,
so a match where one gamer is on his 2nd and the other on his 9th is read at both. The
denominator is every settled match of the window at that place (gamer appearances), bet on or
not, so a place the money is lumped on shows a stake share above its share of appearances.

Bets are every single the checks read (`bets_sql`: AF, moneyline, handicap and total, pre-match
and in play), split by customer temperature (Restricted, VIP, Standard, none) and market.
"""

import os
from collections import defaultdict

from . import bets, config, markets, snowflake_io
from .bet_totals import _start, sessions

GROUPS = ("All", "Restricted", "VIP", "Standard", "none")
MARKETS = (markets.MONEYLINE, markets.SPREAD, markets.TOTAL)
MAX_MATCH = 11


def group_of(b):
    t = str(b.extra.get(config.BET_VIP_COLUMN) or "").strip().lower()
    return next((g for g in GROUPS[1:4] if g.lower() == t), "none")


def match_label(n):
    return f"{n}" if n < MAX_MATCH else f"{MAX_MATCH}+"


def left_label(n):
    return {1: "last", 2: "2nd last", 3: "3rd last"}.get(n, "4-5 left" if n <= 5 else "6+ left")


MATCH_ORDER = [match_label(n) for n in range(1, MAX_MATCH + 1)]
LEFT_ORDER = ["6+ left", "4-5 left", "3rd last", "2nd last", "last"]


def appearances(history, pos, since, until):
    """{(axis, place): gamer appearances} over the settled matches started in [since, until): each
    match twice, once for each gamer at his place."""
    out = defaultdict(int)
    for r in history:
        t = _start(r.get("SCHEDULED_START_TIME_UTC"))
        if t is None or (since and t < since) or (until and t >= until):
            continue
        for side in ("PLAYER_1_HANDLE", "PLAYER_2_HANDLE"):
            p = pos.get((r.get(side), r["MATCH_CODE"]))
            if p:
                out[("match", match_label(min(p[0], MAX_MATCH)))] += 1
                out[("left", left_label(p[1]))] += 1
    return out


def tally(all_bets, history, pos):
    """{(axis, place, group, market): [bets, stake, revenue, settled stake, under stake on totals]}
    with every bet counted once for each gamer of its match; market "all" sums the three."""
    gamers = {r["MATCH_CODE"]: (r.get("PLAYER_1_HANDLE"), r.get("PLAYER_2_HANDLE")) for r in history}
    out = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for b in all_bets:
        pair = gamers.get(b.match_code)
        market = markets.market_group(b.feed_market)
        if not pair or market is None:
            continue
        settled = bets.result_of(b) in (bets.WON, bets.LOST, bets.PUSH)
        under = b.feed_market == 55
        for g in pair:
            p = pos.get((g, b.match_code))
            if not p:
                continue
            for axis, place in (("match", match_label(min(p[0], MAX_MATCH))), ("left", left_label(p[1]))):
                for grp in ("All", group_of(b)):
                    for mk in ("all", market):
                        c = out[(axis, place, grp, mk)]
                        c[0] += 1
                        c[1] += b.stake
                        if settled:
                            c[2] += b.revenue
                            c[3] += b.stake
                        c[4] += b.stake if under else 0.0
    return out


def _pct(a, b):
    return f"{100 * a / b:6.1f}%" if b else f"{'-':>7s}"


def table(t, apps, axis, order, grp, market="all"):
    """Lines of text: one group and market by place."""
    tot_apps = sum(apps.get((axis, p), 0) for p in order)
    tot_b = sum(t[(axis, p, grp, market)][0] for p in order if (axis, p, grp, market) in t)
    tot_s = sum(t[(axis, p, grp, market)][1] for p in order if (axis, p, grp, market) in t)
    if not tot_b:
        return []
    L = [f"  {'':10s} {'gamer-matches':>13s} {'share':>7s} {'bets':>8s} {'share':>7s} {'stake':>12s} "
         f"{'share':>7s} {'stake/match':>11s} {'vs avg':>7s} {'margin':>8s}"
         + (f" {'under':>7s}" if market == markets.TOTAL else "")]
    avg = tot_s / tot_apps if tot_apps else None
    for p in order:
        n = apps.get((axis, p), 0)
        c = t.get((axis, p, grp, market))
        if not n and not c:
            continue
        nb, stake, rev, settled, under = c or (0, 0.0, 0.0, 0.0, 0.0)
        per = stake / n if n else None
        L.append(f"  {p:10s} {n:13,d} {_pct(n, tot_apps)} {nb:8,d} {_pct(nb, tot_b)} {stake:12,.0f} "
                 f"{_pct(stake, tot_s)} {('-' if per is None else f'{per:,.0f}'):>11s} "
                 f"{('-' if per is None or not avg else f'{per / avg:.2f}x'):>7s} "
                 f"{_pct(rev, settled) if settled else '      -'}"
                 + (f" {_pct(under, stake)}" if market == markets.TOTAL else ""))
    return L


def report(t, apps):
    """Lines of text: by match of the session and by matches left, for every group, then each
    market for all customers and for restricted accounts."""
    L = []
    for axis, order, title in (("match", MATCH_ORDER, "the gamer's match of the session"),
                               ("left", LEFT_ORDER, "matches left in the gamer's session")):
        for grp in GROUPS:
            lines = table(t, apps, axis, order, grp)
            if lines:
                L.append(f"\n  {grp}: all markets, by {title} (each bet once for each gamer; "
                         "share of gamer-matches against share of bets and stake; vs avg: stake per "
                         "gamer-match against the group's average; margin: the book's)")
                L += lines
        for grp in ("All", "Restricted", "VIP"):
            for mk in MARKETS:
                lines = table(t, apps, axis, order, grp, mk)
                if lines:
                    L.append(f"\n  {grp}: {mk}, by {title}")
                    L += lines
    return L


def run(cur, out_dir):
    """Fetch the window's bets and the match history, print the tables, write bets_sessions.txt."""
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    all_bets = bets.to_bets(cols, raw)
    history = snowflake_io.fetch_history(cur)
    pos = sessions(history)
    since = _start(config.CUTOFF_START)
    until = _start(config.CUTOFF_END) if config.CUTOFF_END else None
    apps = appearances(history, pos, since, until)
    t = tally(all_bets, history, pos)
    print(f"  {len(all_bets):,} bets; {sum(apps[k] for k in apps if k[0] == 'match') // 2:,} settled matches "
          f"in the window; sessions: no gap over {config.SESSION_BREAK_MINUTES} minutes")
    lines = report(t, apps)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bets_sessions.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
