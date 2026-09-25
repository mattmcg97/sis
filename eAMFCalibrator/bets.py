"""Bets against the prices: a betting simulation of prod against the candidate.

bets -> latency -> join -> analysis

  bets      every in-play AF bet in the window from config.BET_TABLE
  latency   for each match, how long after the Q4 two-minute auto-suspend
            the operator still accepted bets: the latest such bet, capped at
            config.MAX_LAG_SECONDS; matches without one take the median
  join      each bet against the feed message live at (bet time - latency),
            and prod's and the candidate's probability at that message
  analysis  the bet re-priced at the candidate's probability with the
            operator's own margin kept, and the margin both ways
"""

import csv
import datetime as dt
import math
import os
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import config, markets, scouting, snowflake_io
from .snowflake_io import fetch_all

FEED_MARKET = {(1, 1): 50, (1, 2): 51, (2, 1): 52, (2, 2): 53, (3, 1): 54, (3, 2): 55}
SUSPENDS = ("BET_SUSPEND", "PERMANENT_BET_SUSPEND")
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


@dataclass
class Latency:
    """How late one match's bets ran behind the feed."""
    seconds: float
    source: str
    suspend_time: object = None
    late_bets: int = 0


def _naive(t):
    """A datetime in UTC without tzinfo, so feed and bet times compare."""
    if t is None or getattr(t, "tzinfo", None) is None:
        return t
    return t.astimezone(dt.timezone.utc).replace(tzinfo=None)


def _seconds(a, b):
    """a - b in seconds."""
    return (_naive(a) - _naive(b)).total_seconds()


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


def auto_suspend_times(scouting_rows, clock=None, slack=None):
    """match -> the time of the Q4 two-minute auto-suspend. The feed suspends around every play,
    so it is the fourth quarter's last suspend that is never lifted in the quarter, when it came
    with no more than clock + slack seconds left. Rows are scouting.fetch_scouting's."""
    clock = config.AUTO_SUSPEND_CLOCK if clock is None else clock
    slack = config.AUTO_SUSPEND_SLACK if slack is None else slack
    period, last = {}, {}
    for r in scouting_rows:
        match, status = r[0], scouting._text(r[3])
        if status in scouting.PERIOD_START:
            period[match] = scouting.PERIOD_START[status]
        if period.get(match) != 4:
            continue
        if status in SUSPENDS:
            last[match] = (_float(r[2]), _naive(r[9]))
        elif status == "BET_UNSUSPEND":
            last[match] = None
    return {m: t for m, v in last.items() if v is not None
            for left, t in [v] if left is not None and left <= clock + slack and t is not None}


def kickoff_times(scouting_rows):
    """match -> the time of its first play: a bet before it is pre-match."""
    out = {}
    for r in scouting_rows:
        if r[0] not in out and scouting._text(r[4]) == "PLAY_STARTED" and r[9] is not None:
            out[r[0]] = _naive(r[9])
    return out


def match_latency(bets, suspend_times, max_lag=None):
    """match -> Latency: the latest bet accepted after the Q4 auto-suspend, up to max_lag seconds
    after it. A match with a suspend but no such bet, or with no suspend, takes the median of the
    rest ("median")."""
    max_lag = config.MAX_LAG_SECONDS if max_lag is None else max_lag
    by_match = defaultdict(list)
    for b in bets:
        by_match[b.match_code].append(b)
    out = {}
    for match, suspend in suspend_times.items():
        late = [_seconds(b.time, suspend) for b in by_match.get(match, [])]
        late = [x for x in late if 0 < x <= max_lag]
        if late:
            out[match] = Latency(max(late), "suspend", suspend, len(late))
    fallback = statistics.median([x.seconds for x in out.values()]) if out else 0.0
    for match in by_match:
        if match not in out:
            out[match] = Latency(fallback, "median", suspend_times.get(match), 0)
    return out


def message_times(scouting_rows):
    """match -> (sorted feed times, their message counts), off scouting.fetch_scouting's rows."""
    pairs = defaultdict(list)
    for r in scouting_rows:
        if r[9] is not None and r[1] is not None:
            pairs[r[0]].append((_naive(r[9]), int(r[1])))
    out = {}
    for match, ps in pairs.items():
        ps.sort()
        out[match] = ([t for t, _ in ps], [m for _, m in ps])
    return out


def message_at(times, match, t):
    """The last feed message at or before time t, or None."""
    if match not in times or t is None:
        return None
    ts, msgs = times[match]
    i = bisect_right(ts, t)
    return msgs[i - 1] if i else None


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
        latest[key] = (publish, float(prob), markets.parse_line(desc), live)
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
    i = bisect_right(msgs, message)
    return quotes[i - 1] if i else None


def result_of(bet):
    """won / lost / push off the bet's revenue, stake and odds; other for anything else (cash out,
    part settled)."""
    if not bet.stake or not bet.odds:
        return OTHER
    payout = (bet.stake - bet.revenue) / bet.stake
    if abs(payout) < 0.01:
        return LOST
    if abs(payout - 1.0) < 0.01:
        return PUSH
    if abs(payout - bet.odds) < 0.01 * bet.odds:
        return WON
    return OTHER


def _same_line(a, b):
    return a is None or b is None or abs(a - b) < 1e-6


def join(bets, latency, times, prod, cand, kickoff=None):
    """One row per bet: the message it was priced at, prod's and the candidate's probability there,
    and the bet re-priced at the candidate's probability with the operator's margin kept. A bet
    before the match's first play (kickoff) is pre-match and not re-priced."""
    rows = []
    for b in bets:
        lat = latency.get(b.match_code)
        lag = lat.seconds if lat else 0.0
        t0 = b.time
        seen = None if t0 is None else t0 - dt.timedelta(seconds=lag)
        msg = message_at(times, b.match_code, seen)
        msg0 = message_at(times, b.match_code, t0)
        market = b.feed_market
        p = price_at(prod, b.match_code, market, msg)
        c = price_at(cand, b.match_code, market, msg)
        p0 = price_at(prod, b.match_code, market, msg0)
        result = result_of(b)
        row = dict(bet_id=b.bet_id, match_code=b.match_code, bet_time=b.time,
                   market=markets.market_group(market), selection=markets.selection_label(market),
                   feed_market=market, period=b.period, bet_line=b.line, odds=b.odds,
                   stake=b.stake, revenue=b.revenue, result=result,
                   latency_seconds=round(lag, 3), latency_source=lat.source if lat else None,
                   message=msg, implied_prob=(1.0 / b.odds) if b.odds else None,
                   stream_prob=p[0] if p else None, stream_line=p[1] if p else None,
                   stream_live=p[2] if p else None,
                   candidate_prob=c[0] if c else None, candidate_line=c[1] if c else None,
                   candidate_live=c[2] if c else None,
                   stream_prob_no_lag=p0[0] if p0 else None)
        row["in_play"] = bool(kickoff is None or (b.match_code in kickoff and t0 is not None
                                                    and t0 >= kickoff[b.match_code]))
        row["line_match"] = bool(p and c and _same_line(b.line, p[1]) and _same_line(b.line, c[1]))
        row.update(reprice(row))
        row.update({k: v for k, v in b.extra.items()})
        rows.append(row)
    return rows


def reprice(row):
    """The bet at the candidate's probability: the operator's margin over prod (implied / prod)
    kept, so the candidate's odds are odds * prod / candidate. Revenue as the book sees it."""
    p, c, odds, stake, result = (row["stream_prob"], row["candidate_prob"], row["odds"],
                                 row["stake"], row["result"])
    if not (p and c and odds and row["line_match"] and row.get("in_play", True)) or result == OTHER:
        return dict(candidate_odds=None, candidate_revenue=None, simulated=False)
    odds_c = odds * p / c
    revenue_c = stake - (stake * odds_c if result == WON else stake if result == PUSH else 0.0)
    return dict(candidate_odds=round(odds_c, 4), candidate_revenue=revenue_c, simulated=True)


def summarise(rows, by=None):
    """{group: (bets, stake, revenue, margin %, candidate revenue, candidate margin %)} over the
    simulated bets; by is a row key (None: all)."""
    groups = defaultdict(list)
    for r in rows:
        if r["simulated"]:
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
    print(f"  {len(bets):,} bets on {len(matches):,} matches from {config.BET_TABLE}")
    if not bets:
        return None
    table = scouting.locate(cur, config.SCOUTING_TABLE)
    feed, prod_rows, cand_rows = [], [], []
    cand_name = candidate_stream()
    chunk = config.MATCH_CHUNK_SIZE
    for start in range(0, len(matches), chunk):
        batch = matches[start:start + chunk]
        feed += scouting.fetch_scouting(cur, table, batch, windowed=False)
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        cand_rows += snowflake_io.fetch_quotes(cur, cand_name, batch)
    suspend = auto_suspend_times(feed)
    latency = match_latency(bets, suspend)
    rows = join(bets, latency, message_times(feed), quote_index(prod_rows), quote_index(cand_rows),
                kickoff_times(feed))
    os.makedirs(out_dir, exist_ok=True)
    lat_rows = [dict(match_code=m, latency_seconds=round(x.seconds, 3), source=x.source,
                     suspend_time=x.suspend_time, late_bets=x.late_bets)
                for m, x in sorted(latency.items())]
    write_csv(os.path.join(out_dir, "bets_latency.csv"), lat_rows)
    write_csv(os.path.join(out_dir, "bets_sim.csv"), rows)
    report(rows, latency, cand_name)
    return rows


def report(rows, latency, cand_name):
    """Print the latency, the join and the margin both ways."""
    measured = [x.seconds for x in latency.values() if x.source == "suspend"]
    print(f"\n  latency: measured on {len(measured):,} of {len(latency):,} matches"
          + (f", median {statistics.median(measured):.1f}s, 10th-90th "
             f"{sorted(measured)[len(measured) // 10]:.1f}-{sorted(measured)[(9 * len(measured)) // 10]:.1f}s"
             if measured else "; none measured, every match at 0s"))
    check = lag_check(rows)
    if check:
        n, a, b = check
        print(f"  operator odds against prod: mean |log(implied / prod)| {a:.4f} with the latency, "
              f"{b:.4f} without ({n:,} bets)")
    joined = Counter("simulated" if r["simulated"] else
                     "pre-match" if not r.get("in_play", True) else
                     "no prod price" if r["stream_prob"] is None else
                     "no candidate price" if r["candidate_prob"] is None else
                     "line differs" if not r["line_match"] else
                     f"result {r['result']}" for r in rows)
    print("  bets: " + ", ".join(f"{k} {v:,}" for k, v in joined.most_common()))
    print(f"\n  margin, prod as priced against {cand_name} re-priced (operator margin kept)")
    print(f"  {'':12s} {'bets':>7s} {'stake':>12s} {'prod margin':>12s} {'candidate':>10s} {'change':>8s}")
    for by in (None, "market", "period"):
        for g, (n, stake, rev, m, rev_c, m_c) in sorted(summarise(rows, by).items(), key=lambda kv: str(kv[0])):
            label = str(g) if by is None else f"{by} {g}"
            print(f"  {label:12s} {n:7,d} {stake:12,.0f} {m:11.2f}% {m_c:9.2f}% {m_c - m:+7.2f}")


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
               SUM(CASE WHEN UPPER(BET_TYPE) = 'SINGLE' AND {c['time']} IS NOT NULL THEN 1 ELSE 0 END)
                   AS USABLE,
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


def _revenue_views(cur):
    """Views named like CUSTOMER_REVENUE in any database this login can see, with their columns and
    two rows: the bet-level source may be a view rather than the SHARED table."""
    lines = []
    try:
        names, rows = fetch_all(cur, "SHOW VIEWS LIKE 'CUSTOMER_REVENUE%' IN ACCOUNT")
    except Exception as exc:
        return [f"\n== views like CUSTOMER_REVENUE in the account: search failed ({exc})"]
    at = {n.lower(): i for i, n in enumerate(names)}
    found = [f"{r[at['database_name']]}.{r[at['schema_name']]}.{r[at['name']]}" for r in rows]
    lines.append(f"\n== views like CUSTOMER_REVENUE in the account: {len(found)}")
    for v in found:
        try:
            cols, sample = fetch_all(cur, f"SELECT * FROM {v} WHERE SPORT_CODE = %s LIMIT 2",
                                     (config.SPORT_CODE,))
        except Exception:
            cols, sample = fetch_all(cur, f"SELECT * FROM {v} LIMIT 2")
        lines.append(f"   {v}: {len(cols)} columns: " + ", ".join(cols))
        lines += ["   --- " + "; ".join(f"{n}={x}" for n, x in zip(cols, r)) for r in sample]
    return lines


def _hud_tables(cur):
    """Tables named like HUD in any database this login can see."""
    lines = []
    try:
        names, rows = fetch_all(cur, "SHOW TABLES LIKE '%HUD%' IN ACCOUNT")
        at = {n.lower(): i for i, n in enumerate(names)}
        found = [f"{r[at['database_name']]}.{r[at['schema_name']]}.{r[at['name']]}" for r in rows]
        lines.append(f"\n== tables like HUD in the account: {len(found)}")
        lines += [f"   {t}" for t in found]
        for t in found[:5]:
            cols, sample = fetch_all(cur, f"SELECT * FROM {t} LIMIT 2")
            lines.append(f"   {t}: " + ", ".join(cols))
            lines += ["   --- " + "; ".join(f"{n}={v}" for n, v in zip(cols, r)) for r in sample]
    except Exception as exc:                      # no rights to search the account
        lines.append(f"\n== tables like HUD in the account: search failed ({exc})")
    return lines


def _suspends(cur, sample_matches):
    """The fourth quarter's suspends and unsuspends by game clock, the last never-lifted suspend,
    and where prod's last live quote of each match was."""
    lines = []
    table = scouting.locate(cur, config.SCOUTING_TABLE)
    recent = scouting.scouting_matches(cur, table)[-sample_matches:]
    feed = scouting.fetch_scouting(cur, table, recent, windowed=False)
    period, seen = {}, Counter()
    clock_at = {}
    for r in feed:
        status = scouting._text(r[3])
        if status in scouting.PERIOD_START:
            period[r[0]] = scouting.PERIOD_START[status]
        left = _float(r[2])
        if period.get(r[0]) == 4 and left is not None:
            clock_at[(r[0], r[1])] = left
        if period.get(r[0]) == 4 and status in SUSPENDS + ("BET_UNSUSPEND",):
            seen[(None if left is None else int(left // 20) * 20, status)] += 1
    lines.append(f"\n== fourth quarter in {table.qualified}, last {len(recent)} matches: "
                 "suspends / unsuspends by game clock left (20s bins)")
    for b in sorted({k[0] for k in seen}, key=lambda x: -1 if x is None else x):
        lines.append(f"   {'-' if b is None else b:>4}s  suspend {seen[(b, 'BET_SUSPEND')]:3d}  "
                     f"unsuspend {seen[(b, 'BET_UNSUSPEND')]:3d}  "
                     f"permanent {seen[(b, 'PERMANENT_BET_SUSPEND')]:3d}")
    last = auto_suspend_times(feed, clock=240, slack=0)
    lefts = []
    for r in feed:
        if r[0] in last and _naive(r[9]) == last[r[0]] and scouting._text(r[3]) in SUSPENDS:
            lefts.append(_float(r[2]))
    lines.append(f"   last never-lifted Q4 suspend, game clock left: "
                 + ", ".join(f"{x:.0f}" for x in sorted(x for x in lefts if x is not None))
                 + f"  ({len(last)} of {len(recent)} matches)")
    found = auto_suspend_times(feed)
    lines.append(f"   taken as the two-minute auto-suspend (clock <= {config.AUTO_SUSPEND_CLOCK}"
                 f"+{config.AUTO_SUSPEND_SLACK}s): {len(found)} of {len(recent)} matches")
    prod = snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], recent)
    last_live = {}
    for q in prod:
        if str(q[7]).lower() == "open" and str(q[8]).lower() == "true" and q[6] is not None:
            last_live[q[0]] = max(last_live.get(q[0], 0), int(q[6]))
    lefts = sorted(clock_at[(m, msg)] for m, msg in last_live.items() if (m, msg) in clock_at)
    lines.append("   prod's last live quote, Q4 game clock left at that message: "
                 + ", ".join(f"{x:.0f}" for x in lefts)
                 + f"  ({len(lefts)} of {len(last_live)} matches in Q4)")
    return lines


def probe(cur, sample_matches=20):
    """What the pipeline reads, for checking the configured names: the bet table's columns, its
    coverage by operator in the window, any HUD-like tables, and the fourth quarter's suspends."""
    lines = _revenue_views(cur)
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
    for section in (_coverage, _hud_tables, lambda c: _suspends(c, sample_matches)):
        try:
            lines += section(cur)
        except Exception as exc:              # one failing query should not hide the rest
            lines.append(f"   (failed: {exc})")
    return lines
