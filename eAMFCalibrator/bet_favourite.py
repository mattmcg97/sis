"""Bets split by the pre-match favourite: does the money back the favourite or the underdog, at
what score, and does the book keep the margin prod's price says it should there?

  favourite   the side prod's last pre-match moneyline rated above 50% (a match within
              FAV_MIN of evens has none, and is left out)
  phase       pre-match, or the quarter the bet was placed in (the operator's period)
  state       the favourite's lead at the prod message the bet was priced at: down 9+, down
              1-8, level, up 1-8, up 9+ (0-0 pre-match)
  side        moneyline and handicap: favourite or underdog; total: over or under

  price       for in-play bets, whether the price was current: "current" where the feed did
              nothing that moves a price between the price's message and bet time (or only a
              play started or a play over), "score since" where the board moved in between
              (a price the game had overtaken), "other" for a kick-off, timeout or possession

Per cell: bets, stake, the share of the phase's stake, the book's margin (revenue over settled
stake) and the margin prod's price says it keeps (1 - odds x prod's probability of the selection,
on bets struck at prod's line), and their gap: positive where bettors took more than the price
allowed. Prod's probability against how often the selection won says the same per bet. Groups are
the customer temperature (Restricted, VIP, Standard, none).

    python -m eAMFCalibrator bets favourite --since 2026-09-10 --until 2026-09-23
"""

import os
from bisect import bisect_right
from collections import defaultdict

from . import bet_moments, bets, config, markets, snowflake_io
from .bet_sessions import GROUPS, group_of

FAV_MIN = 0.02
PHASES = ("pre-match", "Q1", "Q2", "Q3", "Q4", "OT")
STATES = ("fav down 9+", "fav down 1-8", "level", "fav up 1-8", "fav up 9+")
FAV, DOG, OVER, UNDER = "favourite", "underdog", "over", "under"
SIDES = {markets.MONEYLINE: (FAV, DOG), markets.SPREAD: (FAV, DOG), markets.TOTAL: (OVER, UNDER)}
PLAYER_1 = {50: True, 51: False, 52: True, 53: False}
MIN_BETS = 30
CURRENT, SCORED, OTHER_MOVE = "current", "score since", "other"


def price_state(moved):
    """current / score since / other off bet_moments' `moved` (None pre-match or unknown)."""
    if moved in ("nothing", "play started", "play over", "price after the bet"):
        return CURRENT
    if moved == "score":
        return SCORED
    return None if moved is None else OTHER_MOVE


def prematch_favourite(tl, match):
    """(P1 is the favourite, the favourite's probability) off prod's last pre-match moneyline (its
    first in-play one where it never quoted before the kick-off), or None within FAV_MIN of evens
    or without a quote."""
    entry = tl.get((match, 50))
    if not entry:
        return None
    quotes = entry[1]
    pre = [q for q in quotes if q[0] is None]
    q = pre[-1] if pre else quotes[0]
    p = q[1]
    if p is None or abs(p - 0.5) < FAV_MIN:
        return None
    return p > 0.5, max(p, 1 - p)


def phase_of(b):
    """pre-match, Q1 .. Q4, OT (any period past 4), or None for an in-play bet with no period."""
    if not b.in_play:
        return "pre-match"
    try:
        p = int(float(b.period))
    except (TypeError, ValueError):
        return None
    return None if p < 1 else f"Q{p}" if p <= 4 else "OT"


def state_of(lead):
    """The favourite's lead as a state label."""
    if lead <= -9:
        return STATES[0]
    if lead < 0:
        return STATES[1]
    if lead == 0:
        return STATES[2]
    return STATES[3] if lead <= 8 else STATES[4]


def side_of(feed_market, p1_fav):
    """favourite / underdog for a moneyline or handicap selection, over / under for a total."""
    if feed_market in PLAYER_1:
        return FAV if PLAYER_1[feed_market] == p1_fav else DOG
    return OVER if feed_market == 54 else UNDER if feed_market == 55 else None


def tag(rows, prod_tl, scores):
    """bets.join rows with the favourite, phase, state, side and group added; rows of matches with
    no favourite, or bets with no phase, are dropped. `scores` is bet_checks.score_index's."""
    out = []
    for r in rows:
        fav = prematch_favourite(prod_tl, r["match_code"])
        b = r["_bet"]
        phase = phase_of(b)
        group = markets.market_group(r["feed_market"])
        if fav is None or phase is None or group not in SIDES:
            continue
        p1_fav, fav_p = fav
        p1, p2 = (0, 0)
        if r["message"] is not None and r["match_code"] in scores:
            msgs, vals = scores[r["match_code"]]
            i = bisect_right(msgs, int(r["message"]))
            p1, p2 = vals[i - 1] if i else (0, 0)
        lead = (p1 - p2) if p1_fav else (p2 - p1)
        won = {bets.WON: 1.0, bets.LOST: 0.0}.get(r["result"])
        priced = r["stream_prob"] is not None and r["on_prod_line"] and r["odds"]
        out.append(dict(r, fav_p=fav_p, phase=phase, fav_lead=lead, state=state_of(lead),
                        price=price_state(r.get("moved")) if b.in_play else "pre-match",
                        side=side_of(r["feed_market"], p1_fav), group=group_of(b),
                        won=won, priced=bool(priced),
                        expected_revenue=(r["stake"] * (1 - r["odds"] * r["stream_prob"])
                                          if priced else None)))
    return out


def cell(rows):
    """[bets, stake, settled stake, revenue, priced stake, expected revenue, prod p x stake on
    settled priced, won x stake on settled priced, settled priced stake]"""
    c = [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for r in rows:
        c[0] += 1
        c[1] += r["stake"]
        settled = r["result"] in (bets.WON, bets.LOST, bets.PUSH)
        if settled:
            c[2] += r["stake"]
            c[3] += r["revenue"]
        if r["priced"]:
            c[4] += r["stake"]
            c[5] += r["expected_revenue"]
            if r["won"] is not None:
                c[6] += r["stream_prob"] * r["stake"]
                c[7] += r["won"] * r["stake"]
                c[8] += r["stake"]
    return c


def _pct(a, b, width=7):
    return f"{100 * a / b:{width}.1f}%" if b else f"{'-':>{width + 1}s}"


HEAD = (f"  {'':26s} {'bets':>7s} {'stake':>11s} {'share':>8s} {'margin':>8s} {'expected':>9s} "
        f"{'gap':>7s} {'prod p':>7s} {'won':>7s}")


def line(label, c, total_stake):
    """One row: margin is the book's, expected what prod's price leaves it, gap = expected -
    margin (positive: bettors beat the price); prod p and won are stake-weighted on priced,
    settled bets."""
    margin = c[3] / c[2] if c[2] else None
    expected = c[5] / c[4] if c[4] else None
    gap = (expected - margin) if margin is not None and expected is not None else None
    f = lambda x: f"{100 * x:7.1f}%" if x is not None else f"{'-':>8s}"
    return (f"  {label:26s} {c[0]:7,d} {c[1]:11,.0f} {_pct(c[1], total_stake)} {f(margin)} "
            f"{f(expected):>9s} {('-' if gap is None else f'{100 * gap:+.1f}'):>7s} "
            f"{_pct(c[6], c[8], 6)} {_pct(c[7], c[8], 6)}")


def table(rows, keys, title, min_bets=MIN_BETS, share_of=None):
    """Lines: rows grouped by the tuple of `keys`, each with its share of `share_of`'s stake
    (the same grouping without the last key, by default)."""
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    parent = defaultdict(float)
    for k, rs in groups.items():
        parent[k[:-1] if share_of is None else share_of(k)] += sum(r["stake"] for r in rs)
    order = {"phase": PHASES, "state": STATES, "side": (FAV, DOG, OVER, UNDER),
             "market": (markets.MONEYLINE, markets.SPREAD, markets.TOTAL), "group": GROUPS}
    rank = lambda k: tuple(order.get(name, ()).index(v) if v in order.get(name, ()) else 99
                           for name, v in zip(keys, k))
    L = [f"\n  {title}", HEAD]
    for k in sorted(groups, key=rank):
        rs = groups[k]
        if len(rs) < min_bets:
            continue
        L.append(line(" / ".join(str(x) for x in k), cell(rs),
                      parent[k[:-1] if share_of is None else share_of(k)]))
    return L


def report(rows, min_bets=MIN_BETS):
    """Lines of text: where the stake goes against the favourite's lead, then the favourite ahead
    early by market, side and customer group."""
    L = [f"  {len(rows):,} bets on matches with a pre-match favourite "
         f"({sum(1 for r in rows if r['group'] == 'Restricted'):,} restricted)",
         "  margin: the book's; expected: what prod's price leaves it (1 - odds x prod's probability,"
         " bets at prod's line); gap = expected - margin, positive where bettors beat the price;"
         " prod p / won: prod's probability of the selection against how often it won (stake-weighted)"]
    early = [r for r in rows if r["phase"] in ("Q1", "Q2")]
    L += table(rows, ["phase", "state"], "All bets by phase and the favourite's lead (share: of the "
               "phase's stake)", min_bets)
    for mk in (markets.MONEYLINE, markets.SPREAD, markets.TOTAL):
        sub = [r for r in early if r["market"] == mk]
        L += table(sub, ["state", "side"], f"Q1 and Q2, {mk}: which side the money backs at each "
                   "lead (share: of the state's stake)", min_bets)
    for grp in GROUPS[1:]:
        sub = [r for r in early if r["group"] == grp]
        for mk in (markets.MONEYLINE, markets.SPREAD, markets.TOTAL):
            g = [r for r in sub if r["market"] == mk]
            if len(g) >= min_bets:
                L += table(g, ["state", "side"], f"{grp}: Q1 and Q2, {mk}", min_bets)
    L += table(rows, ["group", "phase", "side"], "Every group by phase and side, all states "
               "(share: of the group's phase)", min_bets)
    for name, keep in ((CURRENT, lambda r: r["price"] == CURRENT),
                       (SCORED, lambda r: r["price"] == SCORED)):
        sub = [r for r in early if keep(r)]
        for mk in (markets.MONEYLINE, markets.SPREAD, markets.TOTAL):
            g = [r for r in sub if r["market"] == mk]
            L += table(g, ["state", "side"], f"Q1 and Q2, {mk}, price {name} (share: of the state's "
                       "stake at that price)", min_bets)
    L += table([r for r in rows if r["phase"] != "pre-match"], ["phase", "price"],
               "In play by phase and whether the price was current (share: of the phase's stake)",
               min_bets)
    fav_up = [r for r in early if r["state"] in ("fav up 1-8", "fav up 9+")]
    L += table(fav_up, ["group", "market", "side"], "The favourite ahead in Q1 and Q2, by group, "
               "market and side (share: of the group's market)", max(10, min_bets // 3))
    return L


def run(cur, out_dir, min_bets=MIN_BETS):
    """Fetch the window's bets, prod's quotes, the scores and finals; print the tables and write
    bets_favourite.csv and bets_favourite.txt."""
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    every = bets.to_bets(cols, raw)
    every = [b for b in every if b.feed_market is not None]
    matches = sorted({b.match_code for b in every})
    print(f"  {len(every):,} bets on {len(matches):,} matches")
    if not every:
        return None
    prod_rows, finals = [], {}
    for start in range(0, len(matches), config.MATCH_CHUNK_SIZE):
        batch = matches[start:start + config.MATCH_CHUNK_SIZE]
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        finals.update(snowflake_io.fetch_final_scores(cur, batch))
    checks = bets.fetch_checks(cur, matches, prod_rows)
    prod_tl = bets.timeline(prod_rows)
    del prod_rows
    lags = bets.fit_lags(every, prod_tl)
    signs = bets.fit_line_signs(every, prod_tl, lags)
    joined = bets.join(every, lags, signs, prod_tl, {}, None, finals, checks)
    bet_moments.annotate([("prod", joined)], every, checks, checks.timelines)
    for r, b in zip(joined, every):
        r["_bet"] = b
    rows = tag(joined, prod_tl, checks.scores)
    lines = report(rows, min_bets)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    keep = [{k: v for k, v in r.items() if k != "_bet"} for r in rows]
    path = os.path.join(out_dir, "bets_favourite.csv")
    bets.write_csv(path, keep, list(dict.fromkeys(k for r in keep for k in r)))
    with open(os.path.join(out_dir, "bets_favourite.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
