"""Bets against the prices: a betting simulation of prod against the candidate.

bets -> latency -> join -> analysis

  bets      every single AF bet in the window, bet by bet, from
            config.BET_TABLE (the CUSTOMER_REVENUE view), pre-match and in play
  latency   one lag per operator: the one, config.LAG_RANGE seconds, at
            which its in-play moneyline odds follow prod's probability
            closest (its margin taken out); pre-match bets are priced at
            bet time
  join      each bet against prod's quote as published at (bet time - lag),
            and the candidate's at the same message; a spread line is read
            from prod's side, the sign fitted per operator and side
  analysis  the bet re-priced at the candidate's probability with the
            operator's own margin kept, and the margin both ways

The feeds carry no two-minute auto-suspend to measure the lag against, and
per-match lags fitted off the lines (which agree for only a fifth of bets)
made the odds follow prod worse than no lag at all; the odds are the signal.
"""

import csv
import datetime as dt
import math
import os
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import config, markets, snowflake_io
from .pipeline import to_unit_probability
from .snowflake_io import fetch_all

FEED_MARKET = {(1, 1): 50, (1, 2): 51, (2, 1): 52, (2, 2): 53, (3, 1): 54, (3, 2): 55}
WON, LOST, PUSH, OTHER = "won", "lost", "push", "other"


@dataclass
class Bet:
    """One bet as the operator recorded it."""
    bet_id: object
    match_code: str
    time: object
    market_type: int
    selection: int
    odds: float
    stake: float
    revenue: float
    line: float = None
    period: int = None
    extra: dict = field(default_factory=dict)

    @property
    def feed_market(self):
        """The GAMEPLAI market id this bet was placed on."""
        return FEED_MARKET.get((self.market_type, self.selection))

    @property
    def operator(self):
        """The operator that took the bet."""
        return self.extra.get(config.BET_GROUP_COLUMN)

    @property
    def in_play(self):
        """Placed in play, by the operator's own flag."""
        return str(self.extra.get(config.BET_IN_PLAY_COLUMN, "")).strip().lower() == "yes"


@dataclass
class Lag:
    """One operator's lag behind prod: the seconds, and the odds misfit at each lag tried."""
    seconds: float
    bets: int = 0
    curve: dict = field(default_factory=dict)


def _naive(t):
    """A datetime in UTC without tzinfo, so feed and bet times compare."""
    if t is None or getattr(t, "tzinfo", None) is None:
        return t
    return t.astimezone(dt.timezone.utc).replace(tzinfo=None)


def bet_table():
    """config.BET_TABLE, qualified: taken as it is when it names its database and schema."""
    name = config.BET_TABLE
    return name if name.count(".") == 2 else snowflake_io.qualified(name)


def bets_sql():
    """The query for every AF bet in the window with a bet time, from config.BET_TABLE."""
    c = config.BET_COLUMNS
    wanted = [c[k] for k in ("id", "match", "time", "market_type", "selection", "odds", "stake",
                             "revenue", "line", "period") if c.get(k)]
    wanted += [x for x in config.BET_EXTRA_COLUMNS if x not in wanted]
    predicate, params = snowflake_io.window_predicate(c["time"])
    where = [predicate, f"{c['market_type']} IN (1, 2, 3)"]
    if c.get("sport"):
        where.append(f"{c['sport']} = %s")
        params.append(config.SPORT_CODE)
    where += list(config.BET_FILTERS)
    sql = (f"SELECT {', '.join(wanted)} FROM {bet_table()} "
           f"WHERE {' AND '.join(where)} ORDER BY {c['match']}, {c['time']}")
    return sql, params, wanted


def _float(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def to_bets(cols, rows):
    """Bet objects from the query's rows."""
    c = config.BET_COLUMNS
    at = {name: i for i, name in enumerate(cols)}

    def get(r, key):
        name = c.get(key)
        return r[at[name]] if name and name in at else None
    out = []
    for n, r in enumerate(rows):
        mt, sel = get(r, "market_type"), get(r, "selection")
        if mt is None or sel is None:
            continue
        out.append(Bet(bet_id=get(r, "id") if c.get("id") else n, match_code=get(r, "match"),
                       time=_naive(get(r, "time")), market_type=int(mt), selection=int(sel),
                       odds=_float(get(r, "odds")), stake=_float(get(r, "stake")) or 0.0,
                       revenue=_float(get(r, "revenue")) or 0.0, line=_float(get(r, "line")),
                       period=get(r, "period"),
                       extra={x: r[at[x]] for x in config.BET_EXTRA_COLUMNS if x in at}))
    return out


def timeline(quote_rows):
    """(match, market) -> (sorted publish times, [(message, probability, line, live)]), off
    snowflake_io.fetch_quotes rows (published 0-100, kept 0-1); the latest row per publish time
    wins."""
    latest = {}
    for r in quote_rows:
        match, market, publish, prob, _, desc, msg, status, active = r[:9]
        if publish is None or prob is None:
            continue
        live = str(status).lower() == "open" and str(active).lower() == "true"
        latest[(match, int(market), _naive(publish))] = (msg, to_unit_probability(prob),
                                                         markets.parse_line(desc), live)
    grouped = defaultdict(list)
    for (match, market, publish), entry in latest.items():
        grouped[(match, market)].append((publish, entry))
    out = {}
    for key, qs in grouped.items():
        qs.sort(key=lambda q: q[0])
        out[key] = ([q[0] for q in qs], [q[1] for q in qs])
    return out


def price_at_time(tl, match, market, t):
    """(message, probability, line, live) of the latest quote published at or before t, or None."""
    entry = tl.get((match, market))
    if entry is None or t is None:
        return None
    times, quotes = entry
    i = bisect_right(times, t)
    return quotes[i - 1] if i else None


def quote_index(quote_rows):
    """(match, market) -> (sorted message counts, [(probability, line, live)]), off
    snowflake_io.fetch_quotes rows; the latest row per message wins."""
    latest = {}
    for r in quote_rows:
        match, market, publish, prob, _, desc, msg, status, active = r[:9]
        if msg is None or prob is None:
            continue
        key = (match, int(market), int(msg))
        if key in latest and latest[key][0] > publish:
            continue
        live = str(status).lower() == "open" and str(active).lower() == "true"
        latest[key] = (publish, to_unit_probability(prob), markets.parse_line(desc), live)
    grouped = defaultdict(list)
    for (match, market, msg), (_, prob, line, live) in latest.items():
        grouped[(match, market)].append((msg, prob, line, live))
    out = {}
    for key, qs in grouped.items():
        qs.sort(key=lambda q: q[0])
        out[key] = ([q[0] for q in qs], [q[1:] for q in qs])
    return out


def price_at(index, match, market, message):
    """(probability, line, live) of the latest quote at or before this message, or None."""
    entry = index.get((match, market))
    if entry is None or message is None:
        return None
    msgs, quotes = entry
    i = bisect_right(msgs, int(message))
    return quotes[i - 1] if i else None


def _same_line(a, b):
    return a is None or b is None or abs(a - b) < 1e-6


def odds_misfit(bets, tl, lag):
    """(bets priced, misfit) at this lag: the mean absolute log of implied over prod, less its median
    (the operator's margin). A negative lag reads prod's quote from after the bet (a clock that
    runs behind GAMEPLAI's)."""
    delta = dt.timedelta(seconds=lag)
    logs = []
    for b in bets:
        q = price_at_time(tl, b.match_code, b.feed_market, b.time - delta)
        if q is not None and b.odds and q[1] and 0 < q[1] < 1:
            logs.append(math.log(1.0 / b.odds / q[1]))
    if not logs:
        return 0, float("inf")
    mid = statistics.median(logs)
    return len(logs), sum(abs(x - mid) for x in logs) / len(logs)


def fit_lags(bets, tl, lags=None):
    """operator -> Lag: over its in-play moneyline bets, the lag at which the odds follow prod
    closest (ties to the lag nearest 0)."""
    lo, hi = config.LAG_RANGE if lags is None else (min(lags), max(lags))
    lags = list(range(int(lo), int(hi) + 1, config.LAG_STEP_SECONDS)) if lags is None else lags
    by_op = defaultdict(list)
    for b in bets:
        if b.in_play and b.market_type == 1 and b.time is not None and b.odds:
            by_op[b.operator].append(b)
    out = {}
    for op, bs in by_op.items():
        curve = {lag: odds_misfit(bs, tl, lag) for lag in lags}
        best = min(lags, key=lambda lag: (curve[lag][1], abs(lag)))
        out[op] = Lag(best, curve[best][0], {lag: m for lag, (_, m) in curve.items()})
    return out


def bet_lag(b, lags):
    """The seconds a bet is read back from: its operator's lag in play, none pre-match."""
    lag = lags.get(b.operator)
    return lag.seconds if (lag is not None and b.in_play) else 0.0


def fit_line_signs(bets, tl, lags):
    """(operator, market) -> (sign, bets, same, turned): whether the operator records a spread's line
    from prod's side (+1) or the other (-1), by which agrees with prod's line more often."""
    counts = defaultdict(lambda: [0, 0, 0])
    for b in bets:
        if b.line is None or b.market_type not in (2, 3):
            continue
        q = price_at_time(tl, b.match_code, b.feed_market,
                          b.time - dt.timedelta(seconds=bet_lag(b, lags)))
        if q is None or q[2] is None:
            continue
        c = counts[(b.operator, b.feed_market)]
        c[0] += 1
        c[1] += abs(b.line - q[2]) < 1e-6
        c[2] += abs(b.line + q[2]) < 1e-6
    return {k: (-1 if turned > same else 1, n, same / n, turned / n)
            for k, (n, same, turned) in counts.items()}


def result_of(bet):
    """won / lost / push off the bet's revenue, stake and odds; other for anything else (cash out,
    part settled)."""
    if not bet.stake or not bet.odds:
        return OTHER
    if str(bet.extra.get("BET_CASHED_OUT", "")).strip().lower() == "yes":
        return OTHER
    payout = (bet.stake - bet.revenue) / bet.stake
    if abs(payout) < 0.01:
        return LOST
    if abs(payout - 1.0) < 0.01:
        return PUSH
    if abs(payout - bet.odds) < 0.01 * bet.odds:
        return WON
    return OTHER


def join(bets, lags, signs, prod_tl, cand, cand_tl=None):
    """One row per bet: prod's quote as published one lag before the bet (at bet time pre-match), the
    candidate's probability at the same message (at the same time when prod's quote has none), and
    the bet re-priced at the candidate's probability with the operator's margin kept."""
    rows = []
    for b in bets:
        lag = bet_lag(b, lags)
        market = b.feed_market
        seen = None if b.time is None else b.time - dt.timedelta(seconds=lag)
        p = price_at_time(prod_tl, b.match_code, market, seen)
        p0 = price_at_time(prod_tl, b.match_code, market, b.time)
        msg = p[0] if p else None
        if msg is not None:
            c = price_at(cand, b.match_code, market, msg)
        else:
            q = price_at_time(cand_tl or {}, b.match_code, market, seen)
            c = q[1:] if q else None
        sign = signs.get((b.operator, market), (1,))[0]
        line = None if b.line is None else sign * b.line
        row = dict(bet_id=b.bet_id, match_code=b.match_code, bet_time=b.time, in_play=b.in_play,
                   market=markets.market_group(market), selection=markets.selection_label(market),
                   feed_market=market, period=b.period, bet_line=b.line, bet_line_prod_side=line,
                   odds=b.odds, stake=b.stake, revenue=b.revenue, result=result_of(b),
                   latency_seconds=lag, message=msg,
                   implied_prob=(1.0 / b.odds) if b.odds else None,
                   stream_prob=p[1] if p else None, stream_line=p[2] if p else None,
                   stream_live=p[3] if p else None,
                   candidate_prob=c[0] if c else None, candidate_line=c[1] if c else None,
                   candidate_live=c[2] if c else None,
                   stream_prob_no_lag=p0[1] if p0 else None)
        row["line_match"] = bool(p and c and _same_line(line, p[2]) and _same_line(line, c[1]))
        row.update(reprice(row))
        row.update(b.extra)
        rows.append(row)
    return rows


def reprice(row):
    """The bet at the candidate's probability: the operator's margin over prod (implied / prod)
    kept, so the candidate's odds are odds * prod / candidate. Revenue as the book sees it."""
    p, c, odds, stake, result = (row["stream_prob"], row["candidate_prob"], row["odds"],
                                 row["stake"], row["result"])
    if not (p and c and odds and row["line_match"]) or result == OTHER:
        return dict(candidate_odds=None, candidate_revenue=None, simulated=False)
    odds_c = odds * p / c
    revenue_c = stake - (stake * odds_c if result == WON else stake if result == PUSH else 0.0)
    return dict(candidate_odds=round(odds_c, 4), candidate_revenue=revenue_c, simulated=True)


def summarise(rows, by=None, keep=None):
    """{group: (bets, stake, revenue, margin %, candidate revenue, candidate margin %)} over the
    simulated bets that `keep` allows; by is a row key (None: all)."""
    groups = defaultdict(list)
    for r in rows:
        if r["simulated"] and (keep is None or keep(r)):
            groups["all" if by is None else r.get(by)].append(r)
    out = {}
    for g, rs in groups.items():
        stake = sum(r["stake"] for r in rs)
        rev = sum(r["revenue"] for r in rs)
        rev_c = sum(r["candidate_revenue"] for r in rs)
        out[g] = (len(rs), stake, rev, 100 * rev / stake if stake else None, rev_c,
                  100 * rev_c / stake if stake else None)
    return out


def coverage(rows):
    """{(in play, market): (bets, stake, bets re-priced, stake re-priced)}: where the money is and how
    much of it the simulation reaches."""
    out = defaultdict(lambda: [0, 0.0, 0, 0.0])
    for r in rows:
        c = out[(r["in_play"], r["market"])]
        c[0] += 1
        c[1] += r["stake"]
        if r["simulated"]:
            c[2] += 1
            c[3] += r["stake"]
    return {k: tuple(v) for k, v in out.items()}


def why_not(r):
    """Why a bet was not re-priced."""
    if r["simulated"]:
        return "simulated"
    if r["stream_prob"] is None:
        return "no prod price"
    if r["candidate_prob"] is None:
        return "no candidate price"
    if not r["line_match"]:
        return "line differs"
    return f"result {r['result']}"


def lag_check(rows):
    """How closely the operator's odds follow prod's probability in play, with the lag and without:
    the mean absolute log of implied / prod, less each operator's median (its margin)."""
    by_op = defaultdict(lambda: ([], []))
    for r in rows:
        if r["in_play"] and r["implied_prob"] and r["stream_prob"] and r["stream_prob_no_lag"]:
            a, b = by_op[r.get(config.BET_GROUP_COLUMN)]
            a.append(math.log(r["implied_prob"] / r["stream_prob"]))
            b.append(math.log(r["implied_prob"] / r["stream_prob_no_lag"]))
    if not by_op:
        return None
    def dev(xs):
        mid = statistics.median(xs)
        return [abs(x - mid) for x in xs]
    with_lag = [d for a, _ in by_op.values() for d in dev(a)]
    without = [d for _, b in by_op.values() for d in dev(b)]
    return len(with_lag), sum(with_lag) / len(with_lag), sum(without) / len(without)


def line_report(rows):
    """For spread and total bets priced by prod, by operator and market: how often the bet's line (read
    from prod's side) is prod's, and the commonest gaps (bet line - prod's line)."""
    groups = defaultdict(list)
    for r in rows:
        if r["market"] in (markets.SPREAD, markets.TOTAL) and r["bet_line_prod_side"] is not None \
                and r["stream_line"] is not None:
            groups[(r.get(config.BET_GROUP_COLUMN), r["market"])].append(r)
    out = []
    for (op, market), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        same = sum(abs(r["bet_line_prod_side"] - r["stream_line"]) < 1e-6 for r in rs)
        gaps = Counter(round(r["bet_line_prod_side"] - r["stream_line"], 1) for r in rs).most_common(6)
        out.append((op, market, len(rs), same / len(rs), gaps))
    return out


def write_csv(path, rows, fields=None):
    """Write rows (dicts) to a CSV."""
    fields = fields or (list(rows[0]) if rows else [])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def candidate_stream():
    """The candidate stream, a model that quotes its own lines read at prod's (bets were placed at
    prod's line)."""
    name = config.STREAMS["candidate"]
    if snowflake_io.is_model(name) and "@" not in name:
        version, _ = snowflake_io.model_version(name)
        if version in snowflake_io.LINE_MODELS:
            name = f"{name}@prod"
    return name


def run(cur, out_dir):
    """The whole pipeline: fetch, lag, join, write the CSVs and print the summary."""
    sql, params, _ = bets_sql()
    cols, raw = fetch_all(cur, sql, tuple(params))
    bets = to_bets(cols, raw)
    matches = sorted({b.match_code for b in bets})
    print(f"  {len(bets):,} bets on {len(matches):,} matches from {bet_table()} "
          f"({sum(b.in_play for b in bets):,} in play)")
    if not bets:
        return None
    prod_rows, cand_rows = [], []
    cand_name = candidate_stream()
    chunk = config.MATCH_CHUNK_SIZE
    for start in range(0, len(matches), chunk):
        batch = matches[start:start + chunk]
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        cand_rows += snowflake_io.fetch_quotes(cur, cand_name, batch)
    prod_tl = timeline(prod_rows)
    lags = fit_lags(bets, prod_tl)
    signs = fit_line_signs(bets, prod_tl, lags)
    rows = join(bets, lags, signs, prod_tl, quote_index(cand_rows), timeline(cand_rows))
    os.makedirs(out_dir, exist_ok=True)
    lag_rows = [dict(operator=op, lag_seconds=lag.seconds, bets=lag.bets,
                     **{f"misfit_{k}s": round(v, 5) for k, v in sorted(lag.curve.items())})
                for op, lag in sorted(lags.items(), key=lambda kv: str(kv[0]))]
    write_csv(os.path.join(out_dir, "bets_latency.csv"), lag_rows)
    write_csv(os.path.join(out_dir, "bets_sim.csv"), rows)
    report(rows, lags, signs, cand_name)
    return rows


def report(rows, lags, signs, cand_name):
    """Print the lag, the lines, where the money is and the margin both ways."""
    print("\n  lag by operator: in-play moneyline odds against prod's probability, misfit at each lag "
          "(lower follows prod closer)")
    for op, lag in sorted(lags.items(), key=lambda kv: str(kv[0])):
        shown = [x for x in (-20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 45, 60) if x in lag.curve]
        print(f"  {str(op)[:22]:22s} lag {lag.seconds:+.0f}s ({lag.bets:,} bets)  "
              + "  ".join(f"{x:+d}s {lag.curve[x]:.4f}" for x in shown))
    check = lag_check(rows)
    if check:
        n, a, b = check
        print(f"  in play, odds against prod (margin taken out): {a:.4f} with the lag, {b:.4f} "
              f"without ({n:,} bets)")
    print("\n  spread and total lines against prod's (sign: +1 as prod, -1 from the other side):")
    for (op, market), (sign, n, same, turned) in sorted(signs.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        print(f"  {str(op)[:22]:22s} {markets.market_group(market):7s} {markets.selection_label(market):5s} "
              f"{n:6,d} bets  as prod {100 * same:5.1f}%  turned {100 * turned:5.1f}%  -> sign {sign:+d}")
    for op, market, n, same, gaps in line_report(rows):
        print(f"  {str(op)[:22]:22s} {market:7s} {n:6,d} bets  on prod's line {100 * same:5.1f}%  "
              "commonest gaps " + ", ".join(f"{g:+g} ({c:,})" for g, c in gaps))
    print("\n  where the money is, and how much the simulation re-prices:")
    print(f"  {'':22s} {'bets':>8s} {'stake':>12s} {'re-priced':>10s} {'of stake':>9s}")
    cov = coverage(rows)
    for (in_play, market), (n, stake, n_s, stake_s) in sorted(cov.items(), key=lambda kv: (not kv[0][0], str(kv[0][1]))):
        label = f"{'in play' if in_play else 'pre-match'} {market}"
        print(f"  {label:22s} {n:8,d} {stake:12,.0f} {n_s:10,d} {100 * stake_s / stake if stake else 0:8.1f}%")
    joined = Counter(why_not(r) for r in rows)
    print("  bets: " + ", ".join(f"{k} {v:,}" for k, v in joined.most_common()))
    vip = config.BET_VIP_VALUE.lower()
    not_vip = lambda r: str(r.get(config.BET_VIP_COLUMN, "")).strip().lower() != vip
    print(f"\n  margin, prod as priced against {cand_name} re-priced (operator margin kept)")
    print(f"  {'':24s} {'bets':>7s} {'stake':>12s} {'prod':>8s} {'candidate':>10s} {'change':>8s}")
    cuts = [(None, None, ""), (None, not_vip, f"all but {config.BET_VIP_VALUE}"),
            ("in_play", None, ""), (config.BET_GROUP_COLUMN, None, ""),
            (config.BET_VIP_COLUMN, None, ""), ("market", None, ""), ("period", None, "")]
    for by, keep, name in cuts:
        for g, (n, stake, rev, m, rev_c, m_c) in sorted(summarise(rows, by, keep).items(),
                                                          key=lambda kv: str(kv[0])):
            if by == "in_play":
                label = "in play" if g else "pre-match"
            else:
                label = name or (str(g) if by is None else f"{by.lower()} {g}")
            print(f"  {label[:24]:24s} {n:7,d} {stake:12,.0f} {m:7.2f}% {m_c:9.2f}% {m_c - m:+7.2f}")


def probe(cur):
    """What the pipeline reads, for checking the configured names: the bet source's columns and its
    coverage by operator in the window."""
    lines = []
    try:
        cols, _ = fetch_all(cur, f"SELECT * FROM {bet_table()} LIMIT 0")
    except Exception as exc:
        cols = []
        lines.append(f"   (reading {bet_table()} failed: {exc})")
    lines.append(f"\n== {bet_table()} (the configured bet source): {len(cols)} columns")
    lines.append("   " + ", ".join(cols))
    missing = [v for v in list(config.BET_COLUMNS.values()) + config.BET_EXTRA_COLUMNS
               if v and v not in set(cols)]
    lines.append(f"   configured columns missing: {', '.join(missing) or 'none'}")
    try:
        lines += _coverage(cur)
    except Exception as exc:
        lines.append(f"   (failed: {exc})")
    return lines


def _coverage(cur):
    """Bets in the window by operator: how many, with a bet time, single, in play, and the customer
    temperatures (the VIP flag to filter on)."""
    c = config.BET_COLUMNS
    lines = []
    since, until = config.CUTOFF_START[:10], (config.CUTOFF_END or "2100-01-01")[:10]
    names, rows = fetch_all(cur, f"""
        SELECT OPERATOR_NAME, COUNT(*) AS BETS,
               COUNT({c['time']}) AS WITH_TIME,
               SUM(CASE WHEN UPPER(BET_TYPE) = 'SINGLE' THEN 1 ELSE 0 END) AS SINGLES,
               SUM(CASE WHEN UPPER(BET_TYPE) = 'SINGLE' AND {c['time']} IS NOT NULL
                   THEN 1 ELSE 0 END) AS USABLE,
               MIN({c['time']}) AS FIRST_TIME, MAX({c['time']}) AS LAST_TIME
        FROM {bet_table()}
        WHERE {c['sport']} = %s AND REVENUE_DATE BETWEEN TO_DATE(%s) AND TO_DATE(%s)
          AND {c['market_type']} IN (1, 2, 3)
        GROUP BY OPERATOR_NAME ORDER BY BETS DESC""", (config.SPORT_CODE, since, until))
    lines.append(f"\n== {bet_table()}, {since} to {until}, by operator "
                 "(usable = single with a bet time)")
    lines += ["   " + "; ".join(f"{n}={v}" for n, v in zip(names, r)) for r in rows]
    for col in ("BET_IN_PLAY", "CUSTOMER_TEMPERATURE", "BET_CASHED_OUT", "CUSTOMER_WIN_LOSS"):
        _, vals = fetch_all(cur, f"""
            SELECT {col}, COUNT(*) FROM {bet_table()}
            WHERE {c['sport']} = %s AND REVENUE_DATE BETWEEN TO_DATE(%s) AND TO_DATE(%s)
              AND {c['time']} IS NOT NULL
            GROUP BY {col} ORDER BY 2 DESC LIMIT 20""", (config.SPORT_CODE, since, until))
        lines.append(f"   {col}: " + ", ".join(f"{v}={n:,}" for v, n in vals))
    return lines
