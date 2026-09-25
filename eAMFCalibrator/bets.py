"""Bets against the prices: a betting simulation of prod against the candidate.

bets -> latency -> join -> analysis

  bets      every in-play single AF bet in the window, bet by bet, from
            config.BET_TABLE (the CUSTOMER_REVENUE view)
  latency   for each match and operator, the lag at which the bets' lines
            agree best with the line prod was quoting at (bet time - lag);
            ties go to the lag at which the odds follow prod's probability
            closest. A group with too few bets takes its operator's median.
  join      each bet against prod's quote as published at (bet time -
            latency), and the candidate's probability at the same message
  analysis  the bet re-priced at the candidate's probability with the
            operator's own margin kept, and the margin both ways

The feeds carry no two-minute auto-suspend to measure the lag against: the
scouting feed's Q4 suspends and unsuspends balance to 0:00 and prod quotes to
the end, so the operator's suspend is its own, and HudStats is not in
Snowflake. The lines are the next best clock.
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
    def group(self):
        """The (match, operator) the latency is fitted for."""
        return self.match_code, self.operator


@dataclass
class Latency:
    """How far one match's bets at one operator ran behind prod's quotes."""
    seconds: float
    source: str
    bets: int = 0
    line_bets: int = 0
    agree: float = None
    agree_no_lag: float = None


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


def _lag_score(bets, tl, lag):
    """(lines agreeing, line bets priced, odds misfit) for these bets at this lag: the misfit is the
    mean absolute log of implied over prod, less its median (the operator's margin)."""
    agree = n_line = 0
    logs = []
    delta = dt.timedelta(seconds=lag)
    for b in bets:
        q = price_at_time(tl, b.match_code, b.feed_market, b.time - delta)
        if q is None:
            continue
        _, prob, line, _ = q
        if b.line is not None and line is not None:
            n_line += 1
            if abs(b.line - line) < 1e-6:
                agree += 1
            else:
                continue
        if b.odds and prob and 0 < prob < 1:
            logs.append(math.log(1.0 / b.odds / prob))
    if not logs:
        return agree, n_line, float("inf")
    mid = statistics.median(logs)
    return agree, n_line, sum(abs(x - mid) for x in logs) / len(logs)


def fit_latency(bets, tl, max_lag=None, step=None, min_bets=None):
    """(match, operator) -> Latency: the lag, 0 to max_lag seconds, at which the most bets' lines
    agree with prod's line at (bet time - lag), ties to the best odds fit; odds fit alone for a
    moneyline-only group. A group with fewer than min_bets priced bets takes its operator's
    median ("operator median")."""
    max_lag = config.MAX_LAG_SECONDS if max_lag is None else max_lag
    step = config.LAG_STEP_SECONDS if step is None else step
    min_bets = config.MIN_LAG_BETS if min_bets is None else min_bets
    lags = [i * step for i in range(int(max_lag / step) + 1)]
    groups = defaultdict(list)
    for b in bets:
        if b.time is not None and b.feed_market is not None:
            groups[b.group].append(b)
    out = {}
    for key, bs in groups.items():
        scores = {lag: _lag_score(bs, tl, lag) for lag in lags}
        n_line = max(s[1] for s in scores.values())
        priced = sum(1 for b in bs if price_at_time(tl, b.match_code, b.feed_market, b.time))
        if priced < min_bets:
            continue
        best = max(lags, key=lambda lag: (scores[lag][0], -scores[lag][2], -lag))
        source = "lines" if n_line else "odds"
        rate = lambda lag: scores[lag][0] / scores[lag][1] if scores[lag][1] else None
        out[key] = Latency(best, source, len(bs), scores[best][1], rate(best), rate(0))
    by_operator = defaultdict(list)
    for (match, op), lat in out.items():
        by_operator[op].append(lat.seconds)
    everyone = [x.seconds for x in out.values()]
    for key, bs in groups.items():
        if key not in out:
            pool = by_operator.get(key[1]) or everyone
            out[key] = Latency(statistics.median(pool) if pool else 0.0, "operator median", len(bs))
    return out


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


def join(bets, latency, prod_tl, cand):
    """One row per bet: prod's quote as published one latency before the bet, the candidate's
    probability at the same message, and the bet re-priced at the candidate's probability with the
    operator's margin kept."""
    rows = []
    for b in bets:
        lat = latency.get(b.group)
        lag = lat.seconds if lat else 0.0
        market = b.feed_market
        seen = None if b.time is None else b.time - dt.timedelta(seconds=lag)
        p = price_at_time(prod_tl, b.match_code, market, seen)
        p0 = price_at_time(prod_tl, b.match_code, market, b.time)
        msg = p[0] if p else None
        c = price_at(cand, b.match_code, market, msg)
        row = dict(bet_id=b.bet_id, match_code=b.match_code, bet_time=b.time,
                   market=markets.market_group(market), selection=markets.selection_label(market),
                   feed_market=market, period=b.period, bet_line=b.line, odds=b.odds,
                   stake=b.stake, revenue=b.revenue, result=result_of(b),
                   latency_seconds=lag, latency_source=lat.source if lat else None,
                   message=msg, implied_prob=(1.0 / b.odds) if b.odds else None,
                   stream_prob=p[1] if p else None, stream_line=p[2] if p else None,
                   stream_live=p[3] if p else None,
                   candidate_prob=c[0] if c else None, candidate_line=c[1] if c else None,
                   candidate_live=c[2] if c else None,
                   stream_prob_no_lag=p0[1] if p0 else None)
        row["line_match"] = bool(p and c and _same_line(b.line, p[2]) and _same_line(b.line, c[1]))
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


def lag_check(rows):
    """How closely the operator's odds follow prod's probability, with the latency and without:
    the mean absolute log of implied / prod. Smaller with the latency means it lines bets up."""
    with_lag, without = [], []
    for r in rows:
        if r["implied_prob"] and r["stream_prob"] and r["stream_prob_no_lag"]:
            with_lag.append(abs(math.log(r["implied_prob"] / r["stream_prob"])))
            without.append(abs(math.log(r["implied_prob"] / r["stream_prob_no_lag"])))
    if not with_lag:
        return None
    return len(with_lag), sum(with_lag) / len(with_lag), sum(without) / len(without)


def line_report(rows):
    """For spread and total bets priced by prod: how often the bet's line is prod's, how often it is
    prod's with the sign turned (a spread quoted from the other side), and the commonest gaps
    (bet line - prod's line), by operator and market."""
    groups = defaultdict(list)
    for r in rows:
        if r["market"] in (markets.SPREAD, markets.TOTAL) and r["bet_line"] is not None \
                and r["stream_line"] is not None:
            groups[(r.get(config.BET_GROUP_COLUMN), r["market"])].append(r)
    out = []
    for (op, market), rs in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        same = sum(abs(r["bet_line"] - r["stream_line"]) < 1e-6 for r in rs)
        flipped = sum(abs(r["bet_line"] + r["stream_line"]) < 1e-6 for r in rs)
        gaps = Counter(round(r["bet_line"] - r["stream_line"], 1) for r in rs).most_common(6)
        out.append((op, market, len(rs), same / len(rs), flipped / len(rs), gaps))
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
    """The whole pipeline: fetch, latency, join, write the CSVs and print the summary."""
    sql, params, _ = bets_sql()
    cols, raw = fetch_all(cur, sql, tuple(params))
    bets = to_bets(cols, raw)
    matches = sorted({b.match_code for b in bets})
    print(f"  {len(bets):,} bets on {len(matches):,} matches from {bet_table()}")
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
    latency = fit_latency(bets, prod_tl)
    rows = join(bets, latency, prod_tl, quote_index(cand_rows))
    os.makedirs(out_dir, exist_ok=True)
    lat_rows = [dict(match_code=m, operator=op, latency_seconds=x.seconds, source=x.source,
                     bets=x.bets, line_bets=x.line_bets, lines_agree=x.agree,
                     lines_agree_no_lag=x.agree_no_lag)
                for (m, op), x in sorted(latency.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))]
    write_csv(os.path.join(out_dir, "bets_latency.csv"), lat_rows)
    write_csv(os.path.join(out_dir, "bets_sim.csv"), rows)
    report(rows, latency, cand_name)
    return rows


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def report(rows, latency, cand_name):
    """Print the latency, the join and the margin both ways."""
    print("\n  latency by operator (fitted per match off the bets' lines against prod's):")
    by_op = defaultdict(list)
    for (_, op), lat in latency.items():
        by_op[op].append(lat)
    for op, lats in sorted(by_op.items(), key=lambda kv: str(kv[0])):
        fitted = [x for x in lats if x.source != "operator median"]
        secs = [x.seconds for x in fitted]
        agree = [x for x in fitted if x.agree is not None]
        line = f"  {str(op):24s} {len(lats):4d} matches, fitted {len(fitted):4d}"
        if secs:
            line += (f", median {statistics.median(secs):.0f}s (10th-90th "
                     f"{_pct(secs, 0.1):.0f}-{_pct(secs, 0.9):.0f}s)")
        if agree:
            n = sum(x.line_bets for x in agree)
            a = sum(x.agree * x.line_bets for x in agree) / n
            a0 = sum((x.agree_no_lag or 0) * x.line_bets for x in agree) / n
            line += f"; lines agree {100 * a:.1f}% at the fitted lag, {100 * a0:.1f}% at none"
        print(line)
    check = lag_check(rows)
    if check:
        n, a, b = check
        print(f"  operator odds against prod: mean |log(implied / prod)| {a:.4f} with the latency, "
              f"{b:.4f} without ({n:,} bets)")
    print("\n  lines of spread and total bets against prod's at the fitted lag "
          "(same / sign turned / commonest gaps, bet - prod):")
    for op, market, n, same, flipped, gaps in line_report(rows):
        print(f"  {str(op)[:22]:22s} {market:7s} {n:6,d} bets  same {100 * same:5.1f}%  "
              f"turned {100 * flipped:5.1f}%  gaps " + ", ".join(f"{g:+g} ({c:,})" for g, c in gaps))
    joined = Counter("simulated" if r["simulated"] else
                     "no prod price" if r["stream_prob"] is None else
                     "no candidate price" if r["candidate_prob"] is None else
                     "line differs" if not r["line_match"] else
                     f"result {r['result']}" for r in rows)
    print("  bets: " + ", ".join(f"{k} {v:,}" for k, v in joined.most_common()))
    vip = config.BET_VIP_VALUE.lower()
    not_vip = lambda r: str(r.get(config.BET_VIP_COLUMN, "")).strip().lower() != vip
    print(f"\n  margin, prod as priced against {cand_name} re-priced (operator margin kept)")
    print(f"  {'':24s} {'bets':>7s} {'stake':>12s} {'prod':>8s} {'candidate':>10s} {'change':>8s}")
    cuts = [(None, None, ""), (None, not_vip, f"all but {config.BET_VIP_VALUE}"),
            (config.BET_GROUP_COLUMN, None, ""), (config.BET_VIP_COLUMN, None, ""),
            ("market", None, ""), ("period", None, "")]
    for by, keep, name in cuts:
        for g, (n, stake, rev, m, rev_c, m_c) in sorted(summarise(rows, by, keep).items(),
                                                          key=lambda kv: str(kv[0])):
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
                         AND BET_IN_PLAY = 'Yes' THEN 1 ELSE 0 END) AS USABLE,
               MIN({c['time']}) AS FIRST_TIME, MAX({c['time']}) AS LAST_TIME
        FROM {bet_table()}
        WHERE {c['sport']} = %s AND REVENUE_DATE BETWEEN TO_DATE(%s) AND TO_DATE(%s)
          AND {c['market_type']} IN (1, 2, 3)
        GROUP BY OPERATOR_NAME ORDER BY BETS DESC""", (config.SPORT_CODE, since, until))
    lines.append(f"\n== {bet_table()}, {since} to {until}, by operator "
                 "(usable = single, in play, with a bet time)")
    lines += ["   " + "; ".join(f"{n}={v}" for n, v in zip(names, r)) for r in rows]
    for col in ("BET_IN_PLAY", "CUSTOMER_TEMPERATURE", "BET_CASHED_OUT", "CUSTOMER_WIN_LOSS"):
        _, vals = fetch_all(cur, f"""
            SELECT {col}, COUNT(*) FROM {bet_table()}
            WHERE {c['sport']} = %s AND REVENUE_DATE BETWEEN TO_DATE(%s) AND TO_DATE(%s)
              AND {c['time']} IS NOT NULL
            GROUP BY {col} ORDER BY 2 DESC LIMIT 20""", (config.SPORT_CODE, since, until))
        lines.append(f"   {col}: " + ", ".join(f"{v}={n:,}" for v, n in vals))
    return lines
