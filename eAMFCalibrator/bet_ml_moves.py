"""The moneyline movement audit: where the money comes on prod's moneyline, at what prices, and how prod's
price moved either side of each bet.

  where    every moneyline bet by prod's chance of the side backed (the price it read: bet time less
           the operator's lag), favourite or underdog, pre-match or in play, the customer temperature
           and the stake: bets, stake and the book's margin
  path     prod's chance of the side backed at fixed offsets either side of the moment the bet was
           priced, less the chance it was struck at: after the bet, above 0 prod shortened the side
           (moved the bettor's way: the bet beat the price); before it, above 0 the side had drifted
           into the bet (the bettor faded a move), below 0 it had shortened (the bettor followed one)
  scores   in play, a move across a score is the game; one with no score between is prod re-pricing
           the same state -- the drift the bettor was ahead of, the part a model could have had
  close    pre-match, the side's chance at the kick-off less at the bet (closing-line value)

Every mean is stake-weighted where it says so, else a bet each; its 2 standard errors are clustered by
match. A move of MOVE_FLAT (one point of chance) or more is a move; less, the price held.

    python -m eAMFCalibrator bets ml-moves --since 2026-10-01 --until 2026-10-19

No candidate is priced and nothing is simulated: the bets and prod's quotes alone."""

import datetime as dt
import math
import os
from collections import Counter, defaultdict

from . import bet_moments, bets, config, markets, snowflake_io
from .bet_totals import kickoff_time, mean_se, message_at, points, tag

ML = (50, 51)
OFFSETS = (-300, -120, -60, -30, -10, 10, 30, 60, 120, 300)
MOVE_FLAT = 0.01
PRICE_BANDS = ((0.0, 0.2), (0.2, 0.35), (0.35, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01))
STAKE_BANDS = ((0, 25), (25, 100), (100, 500), (500, 2000), (2000, math.inf))
FOR, AGAINST, HELD = "their way", "against them", "held"
FOLLOWED, FADED, STILL = "followed a move", "faded a move", "still price"
MIN_BETS = 30


def _prob(tl, match, market, t):
    """prod's chance of the selection in the latest live quote at or before t, or None."""
    q = bets.price_at_time(tl, match, market, t)
    return None if q is None else q[1]


def direction(v):
    if v is None:
        return None
    return 1 if v >= MOVE_FLAT else -1 if v <= -MOVE_FLAT else 0


def band(v, bands):
    if v is None:
        return None
    for lo, hi in bands:
        if lo <= v < hi:
            return f"{lo:g}-{hi:g}" if math.isfinite(hi) else f"{lo:g}+"
    return None


def path(b, r, tl, checks=None):
    """prod's chance of the backed side around one moneyline bet (r: its bets.join row)."""
    match, market = b.match_code, b.feed_market
    seen = None if b.time is None else b.time - dt.timedelta(seconds=r.get("latency_seconds") or 0)
    p0 = r.get("stream_prob")
    msg = r.get("message")
    board = points(checks, match, msg) if checks is not None and msg is not None else None
    out = dict(seen_time=seen, side=markets.selection_label(market), price=p0,
               price_band=band(p0, PRICE_BANDS), stake_band=band(b.stake, STAKE_BANDS),
               backs=None if p0 is None else ("favourite" if p0 >= 0.5 else "underdog"),
               temperature=str(b.extra.get(config.BET_VIP_COLUMN) or "none"))
    for k in OFFSETS:
        t = None if seen is None else seen + dt.timedelta(seconds=k)
        p = _prob(tl, match, market, t)
        mv = None if p is None or p0 is None else p - p0          # above 0: prod rated the side higher then
        d = direction(mv)
        out[f"move_{tag(k)}"] = mv
        out[f"way_{tag(k)}"] = None if d is None else ({1: FOR, -1: AGAINST, 0: HELD}[d] if k > 0
                                                        else {1: FADED, -1: FOLLOWED, 0: STILL}[d])
        if k > 0 and checks is not None and b.in_play and board is not None:
            m = message_at(checks, match, t)
            out[f"scored_{tag(k)}"] = None if m is None else int(points(checks, match, m) != board)
    if not b.in_play and checks is not None and seen is not None:
        ko = kickoff_time(checks, match)
        if ko is not None and ko > seen:
            p = _prob(tl, match, market, ko - dt.timedelta(milliseconds=1))
            out["close"] = None if p is None or p0 is None else p - p0
    return out


def build(ml_bets, joined, tl, checks):
    """One row per moneyline bet: its bets.join row, the path and, in play, its moment."""
    rows = []
    for b, r in zip(ml_bets, joined):
        row = dict(r)
        for f in bets.CANDIDATE_FIELDS:
            row.pop(f, None)
        row.update(path(b, r, tl, checks))
        if b.in_play and checks is not None and getattr(checks, "timelines", None) is not None:
            row.update(bet_moments.moment_of(checks.timelines.get(b.match_code), checks, b.match_code, b.time,
                                             r.get("message")))
        elif not b.in_play:
            row["moment"] = "pre-match"
        rows.append(row)
    return rows


def in_scope(r):
    return (not r.get("excluded") and r.get("result") in (bets.WON, bets.LOST, bets.PUSH) and r.get("stake")
            and r.get("price") is not None)


def money(rows):
    """(bets, stake, the book's margin %) of the rows."""
    stake = sum(r["stake"] for r in rows)
    return len(rows), stake, (100 * sum(r["revenue"] for r in rows) / stake) if stake else None


def weighted(rows, key):
    """The stake-weighted mean of key, in points of chance (x100)."""
    xs = [(r["stake"], r[key]) for r in rows if r.get(key) is not None]
    w = sum(s for s, _ in xs)
    return 100 * sum(s * x for s, x in xs) / w if w else None


def _f(x, fmt="+.2f", width=7):
    return f"{'-':>{width}s}" if x is None else f"{x:{fmt}}".rjust(width)


def where(rows, key, order=None, min_bets=MIN_BETS):
    """Lines: the money by a key -- bets, stake, its share, the margin, the mean +60s move (a bet each,
    and stake-weighted) and the share prod moved the bettors' way by +60s."""
    by = defaultdict(list)
    for r in rows:
        by[r.get(key)].append(r)
    total = sum(r["stake"] for r in rows) or 1.0
    keys = order or sorted(by, key=lambda k: (k is None, str(k)))
    out = []
    for k in keys:
        rs = by.get(k, [])
        if len(rs) < min_bets:
            continue
        n, stake, m = money(rs)
        _, mv, se = mean_se(rs, "move_p60")
        ways = Counter(r.get("way_p60") for r in rs if r.get("way_p60"))
        nw = sum(ways.values())
        out.append(f"    {str(k)[:20]:20s} {n:8,d} {stake:12,.0f} {100 * stake / total:6.1f}% {_f(m, '+.2f', 8)}%"
                   f" {_f(None if mv is None else 100 * mv)} {_f(None if se is None else 200 * se, '.2f', 6)}"
                   f" {_f(weighted(rs, 'move_p60'))} {_f(100 * ways[FOR] / nw if nw else None, '.1f', 7)}%"
                   f" {_f(100 * ways[AGAINST] / nw if nw else None, '.1f', 7)}%")
    return out


WHERE_HEAD = (f"    {'':20s} {'bets':>8s} {'stake':>12s} {'share':>7s} {'margin':>9s} {'+60s':>7s} {'2se':>6s}"
              f" {'by £':>7s} {'their way':>9s} {'against':>9s}")


def event_study(rows, offsets=OFFSETS):
    """Lines: the mean move at each offset (a bet each, with 2se, and stake-weighted), and the shares."""
    out = [f"    {'':10s} {'bets':>8s} {'mean':>7s} {'2se':>6s} {'by £':>7s}   shares"]
    for k in offsets:
        t = tag(k)
        n, m, se = mean_se(rows, f"move_{t}")
        c = Counter(r.get(f"way_{t}") for r in rows if r.get(f"way_{t}"))
        nw = sum(c.values()) or 1
        labels = (FOLLOWED, FADED, STILL) if k < 0 else (FOR, AGAINST, HELD)
        out.append(f"    {k:+5d}s     {n:8,d} {_f(None if m is None else 100 * m)} {_f(None if se is None else 200 * se, '.2f', 6)}"
                   f" {_f(weighted(rows, f'move_{t}'))}   " + ", ".join(f"{lab} {100 * c[lab] / nw:.0f}%" for lab in labels))
    return out


def margin_by_move(rows, key="way_p60", labels=(FOR, HELD, AGAINST)):
    """Lines: the book's margin on the bets prod moved their way, held, or moved against, by +60s."""
    out = []
    for lab in labels:
        rs = [r for r in rows if r.get(key) == lab]
        if rs:
            n, stake, m = money(rs)
            out.append(f"    {lab:16s} {n:8,d} bets {stake:12,.0f} stake  margin {_f(m, '+.2f', 7)}%")
    return out


def customers(rows, n=15):
    """Lines: the customers with the most moneyline stake -- bets, stake, margin, the stake-weighted +60s
    move and, pre-match, the closing-line value."""
    by = defaultdict(list)
    for r in rows:
        by[r.get("CUSTOMER_NAME_HASH")].append(r)
    out = [f"    {'customer':12s} {'temperature':12s} {'bets':>6s} {'stake':>10s} {'margin':>8s} {'+60s by £':>9s}"
           f" {'close by £':>10s}"]
    for c, rs in sorted(by.items(), key=lambda kv: -sum(r["stake"] for r in kv[1]))[:n]:
        k, stake, m = money(rs)
        temp = Counter(r.get("temperature") for r in rs).most_common(1)[0][0]
        out.append(f"    {str(c)[:12]:12s} {str(temp)[:12]:12s} {k:6,d} {stake:10,.0f} {_f(m, '+.2f', 7)}%"
                   f" {_f(weighted(rs, 'move_p60'), '+.2f', 9)} {_f(weighted(rs, 'close'), '+.2f', 10)}")
    return out


def report(rows, min_bets=MIN_BETS):
    rs = [r for r in rows if in_scope(r)]
    pre = [r for r in rs if not r.get("in_play")]
    live = [r for r in rs if r.get("in_play")]
    L = [f"  MONEYLINE MOVES: {len(rs):,} moneyline bets on {len({r['match_code'] for r in rs}):,} matches, prod's chance of"
         " the side backed either side of the moment each was priced (bet time less the operator's lag)",
         "  move: points of chance; after the bet, above 0 prod shortened the side (their way); before it, above 0 the side"
         " had drifted into the bet (they faded a move), below 0 it had shortened (they followed one); 2se clustered by"
         f" match; a move is {100 * MOVE_FLAT:g} point or more", ""]
    for title, sub in (("PRE-MATCH", pre), ("IN PLAY", live)):
        if not sub:
            continue
        n, stake, m = money(sub)
        L += [f"  {title}: {n:,} bets, {stake:,.0f} stake, the book's margin {_f(m, '+.2f', 0)}%",
              "    where the money came, by prod's chance of the side backed:", WHERE_HEAD]
        L += where(sub, "price_band", [band(lo + 1e-9, PRICE_BANDS) for lo, _ in PRICE_BANDS], min_bets)
        L += ["    by the customer temperature:", WHERE_HEAD] + where(sub, "temperature", None, min_bets)
        L += ["    by the stake:", WHERE_HEAD] + where(sub, "stake_band", [band(lo, STAKE_BANDS) for lo, _ in STAKE_BANDS],
                                                       min_bets)
        L += ["    by the operator:", WHERE_HEAD] + where(sub, config.BET_GROUP_COLUMN, None, min_bets)
        if title == "IN PLAY":
            L += ["    by the moment:", WHERE_HEAD] + where(sub, "moment", None, min_bets)
        L += ["    prod's price either side of the bet:"] + event_study(sub)
        L += ["    the book's margin by what prod did in the 60s after:"] + margin_by_move(sub)
        L += ["    and in the 60s before:"] + margin_by_move(sub, "way_m60", (FOLLOWED, STILL, FADED))
        if title == "IN PLAY":
            quiet = [r for r in sub if r.get("scored_p60") == 0]
            if quiet:
                L += [f"    with no score in the 60s after ({len(quiet):,} bets): prod re-pricing the same state"]
                L += event_study(quiet, [k for k in OFFSETS if k > 0])
                L += margin_by_move(quiet)
        else:
            n_c, m_c, se_c = mean_se(sub, "close")
            if n_c:
                L.append(f"    closing-line value: the side's chance at the kick-off less at the bet, {n_c:,} bets: mean"
                         f" {_f(100 * m_c)} points (2se {_f(None if se_c is None else 200 * se_c, '.2f', 0)}),"
                         f" stake-weighted {_f(weighted(sub, 'close'))}")
                for lab, test in (("beat the close", lambda v: v >= MOVE_FLAT), ("held", lambda v: abs(v) < MOVE_FLAT),
                                  ("the close beat them", lambda v: v <= -MOVE_FLAT)):
                    sel = [r for r in sub if r.get("close") is not None and test(r["close"])]
                    if sel:
                        k, s, mm = money(sel)
                        L.append(f"      {lab:20s} {k:8,d} bets {s:12,.0f} stake  margin {_f(mm, '+.2f', 7)}%")
        L.append("")
    L += ["  THE CUSTOMERS WITH THE MOST MONEYLINE STAKE"] + customers(rs)
    return L


def run(cur, out_dir, min_bets=MIN_BETS):
    """Fetch every moneyline bet of the window and prod's quotes, read prod's price around each; print,
    and write the text and a CSV a bet."""
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    every = bets.to_bets(cols, raw)
    ml = [b for b in every if b.feed_market in ML]
    matches = sorted({b.match_code for b in ml})
    print(f"  {len(ml):,} moneyline bets on {len(matches):,} matches ({sum(b.in_play for b in ml):,} in play)")
    if not ml:
        return None
    prod_rows, finals = [], {}
    for start in range(0, len(matches), config.MATCH_CHUNK_SIZE):
        batch = matches[start:start + config.MATCH_CHUNK_SIZE]
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        finals.update(snowflake_io.fetch_final_scores(cur, batch))
    checks = bets.fetch_checks(cur, matches, prod_rows)
    tl = bets.timeline(prod_rows)
    del prod_rows
    lags = bets.fit_lags([b for b in every if b.match_code in set(matches)], tl)
    print("  lag by operator: " + ", ".join(f"{op} {lag.seconds:+.0f}s" for op, lag in sorted(lags.items(), key=lambda kv: str(kv[0]))))
    joined = bets.join(ml, lags, {}, tl, {}, None, finals, checks)
    rows = build(ml, joined, tl, checks)
    lines = report(rows, min_bets)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    path_txt = os.path.join(out_dir, "bets_ml_moves.txt")
    bets.write_text(path_txt, lines)
    path_csv = os.path.join(out_dir, "bets_ml_moves.csv")
    bets.write_csv(path_csv, rows, list(dict.fromkeys(k for r in rows for k in r)))
    return path_txt, path_csv
