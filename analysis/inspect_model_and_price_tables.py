"""First look at the three tables behind the GAMEPLAI model and latency audit.

Two separate questions share these tables:

  * prod model      SHARED.GAMEPLAI_STREAM            what GAMEPLAI send us
  * candidate model SHARED.GAMEPLAI_STREAM_CANDIDATE  same, their candidate
  * price changes   SHARED.PRICE_CHANGES              what we send customers

Candidate vs prod is a model comparison. Prod vs price changes is a
pipeline audit: PRICE_CHANGES is our own outbound feed, so in theory it
should trail GAMEPLAI_STREAM by only a fraction of a second, and the
point of the audit is to find out whether it really does.

For each table this prints size and freshness, the full column layout, a
few sample rows, the span of every time column, and a profile of the
price/probability columns. It then diffs the prod and candidate schemas
(what is comparable at all), measures where the three tables overlap in
matches and in time -- the candidate stream is likely to cover only a
short recent window, and that window, not the prod history, is what caps
any comparison -- and finally sizes up the inbound/outbound relationship
per match, both in timing offset and in message volume.

PRICE_CHANGES is a guess from the SCORE_CHANGES naming convention. If it
does not resolve, the script lists the real tables whose names look like
price changes and exits; set PRICE_CHANGES below to the right one and
re-run. You can also override any table on the command line:

    python analysis/inspect_model_and_price_tables.py SHARED.PRICE_CHANGE_LOG
"""

import sys

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"

PROD_MODEL = ("SHARED", "GAMEPLAI_STREAM")
CANDIDATE_MODEL = ("SHARED", "GAMEPLAI_STREAM_CANDIDATE")
PRICE_CHANGES = ("SHARED", "PRICE_CHANGES")

SAMPLE_ROWS = 3
LOOKBACK_DAYS = 30
SPORT_CODE = "AF"
MAX_VALUE_CHARS = 200

# Time column to anchor windowed queries on, most preferred first.
#
# PRICE_ISSUE_TIME_UTC beats FILE_TIME for PRICE_CHANGES: FILE_TIME is when
# the outbound file was written, which sampling shows runs 180-240ms behind
# the price issue itself. Anchoring on it would charge that file-write lag
# to pipeline latency. FILE_LOADED is not a load timestamp at all -- in
# GAMEPLAI_STREAM it lands ~80s BEFORE PUBLISH_TIME -- so it stays out of
# this list entirely.
TIME_COLUMN_PREFERENCE = [
    "PUBLISH_TIME",
    "PRICE_ISSUE_TIME_UTC",
    "CHANGE_TIME",
    "FILE_TIME",
    "EVENT_TIME",
    "UPDATED_TIME_UTC",
    "CREATED_TIME_UTC",
    "UPDATED_AT",
    "CREATED_AT",
]

# Numeric columns worth a min/mean/max profile.
VALUE_COLUMN_HINTS = ["PRICE", "ODDS", "PROBABILITY", "MARGIN", "LINE", "HANDICAP"]


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def table_metadata(cur, schema, table):
    _, rows = fetch_all(
        cur,
        f"""
        SELECT table_type, row_count, bytes, created, last_altered
        FROM {DATABASE}.INFORMATION_SCHEMA.TABLES
        WHERE table_schema = %s AND table_name = %s
        """,
        (schema, table),
    )
    return rows[0] if rows else None


def describe_columns(cur, schema, table):
    _, rows = fetch_all(
        cur,
        f"""
        SELECT column_name, data_type, is_nullable, ordinal_position
        FROM {DATABASE}.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return rows


def find_similar_tables(cur, *patterns):
    clauses = " OR ".join(["table_name ILIKE %s"] * len(patterns))
    _, rows = fetch_all(
        cur,
        f"""
        SELECT table_schema, table_name, row_count, last_altered
        FROM {DATABASE}.INFORMATION_SCHEMA.TABLES
        WHERE {clauses}
        ORDER BY table_schema, table_name
        """,
        tuple(patterns),
    )
    return rows


def time_columns(columns):
    return [name for name, dtype, _, _ in columns if dtype.startswith(("TIMESTAMP", "DATE"))]


def primary_time_column(columns):
    available = time_columns(columns)
    for preferred in TIME_COLUMN_PREFERENCE:
        if preferred in available:
            return preferred
    return available[0] if available else None


def value_columns(columns):
    numeric = {"NUMBER", "FLOAT", "DECIMAL", "INT", "INTEGER", "BIGINT", "DOUBLE", "REAL"}
    return [
        name
        for name, dtype, _, _ in columns
        if dtype.upper() in numeric and any(hint in name.upper() for hint in VALUE_COLUMN_HINTS)
    ]


def truncate(val):
    text = str(val)
    return text[:MAX_VALUE_CHARS] + "...(truncated)" if len(text) > MAX_VALUE_CHARS else text


def print_sample(cur, schema, table):
    cols, rows = fetch_all(cur, f'SELECT * FROM "{DATABASE}"."{schema}"."{table}" LIMIT {SAMPLE_ROWS}')
    if not rows:
        print("  (no rows)")
        return
    for row in rows:
        for col_name, val in zip(cols, row):
            print(f"  {col_name}: {truncate(val)}")
        print("  ---")


def inspect_table(cur, label, schema, table, columns, meta):
    print(f"\n{'=' * 80}\n{label}: {schema}.{table}\n{'=' * 80}")

    table_type, row_count, size_bytes, created, last_altered = meta
    gib = size_bytes / 1024 ** 3 if size_bytes else 0
    rows_text = f"{row_count:,}" if row_count is not None else "n/a (view)"
    print(f"Type: {table_type}   rows: {rows_text}   size: {gib:.2f} GiB")
    print(f"Created: {created}   last altered: {last_altered}")

    print(f"\nColumns ({len(columns)}):")
    for name, dtype, nullable, pos in columns:
        print(f"  {pos:>3}  {name:<40} {dtype:<20} nullable={nullable}")

    print(f"\nSample ({SAMPLE_ROWS} rows):")
    print_sample(cur, schema, table)

    times = time_columns(columns)
    if times:
        spans = ", ".join(f"MIN({c}) AS MIN_{c}, MAX({c}) AS MAX_{c}" for c in times)
        cols, rows = fetch_all(cur, f"SELECT {spans} FROM {DATABASE}.{schema}.{table}")
        print("\nTime coverage:")
        for col_name, val in zip(cols, rows[0]):
            print(f"  {col_name}: {val}")

    # Aggregations below are scoped to the recent window: a full scan of a
    # stream table this size is slow and costs warehouse credits, and a
    # first look only needs the shape of current data.
    time_col = primary_time_column(columns)
    if time_col:
        where = f"WHERE {time_col} >= DATEADD(day, -{LOOKBACK_DAYS}, CURRENT_TIMESTAMP())"
        scope = f"last {LOOKBACK_DAYS} days by {time_col}"
    else:
        where = ""
        scope = "whole table -- no time column to scope by"

    values = value_columns(columns)
    if values:
        stats = ", ".join(
            f"MIN({c}) AS MIN_{c}, AVG({c}) AS AVG_{c}, MAX({c}) AS MAX_{c}, "
            f"COUNT_IF({c} IS NULL) AS NULLS_{c}"
            for c in values
        )
        cols, rows = fetch_all(cur, f"SELECT {stats} FROM {DATABASE}.{schema}.{table} {where}")
        print(f"\nPrice / probability columns ({scope}):")
        for col_name, val in zip(cols, rows[0]):
            print(f"  {col_name}: {val}")

    names = {c[0] for c in columns}
    grain = ["COUNT(*) AS N_ROWS"]
    for key in ("MATCH_CODE", "MARKET_ID", "SELECTION_ID", "MARKET_TYPE", "SPORT_CODE"):
        if key in names:
            grain.append(f"APPROX_COUNT_DISTINCT({key}) AS DISTINCT_{key}")
    cols, rows = fetch_all(cur, f"SELECT {', '.join(grain)} FROM {DATABASE}.{schema}.{table} {where}")
    print(f"\nGrain, approximate distinct counts ({scope}):")
    for col_name, val in zip(cols, rows[0]):
        print(f"  {col_name}: {val:,}" if isinstance(val, int) else f"  {col_name}: {val}")


def compare_schemas(prod_columns, candidate_columns):
    print(f"\n{'=' * 80}\nProd vs candidate schema diff\n{'=' * 80}")

    prod = {name: dtype for name, dtype, _, _ in prod_columns}
    cand = {name: dtype for name, dtype, _, _ in candidate_columns}

    shared = sorted(set(prod) & set(cand))
    print(f"\nShared columns ({len(shared)}) -- these are what a comparison can use:")
    for name in shared:
        flag = "" if prod[name] == cand[name] else f"   <-- TYPE DIFFERS: prod={prod[name]} candidate={cand[name]}"
        print(f"  {name:<40} {prod[name]}{flag}")

    only_prod = sorted(set(prod) - set(cand))
    only_cand = sorted(set(cand) - set(prod))
    print(f"\nProd only ({len(only_prod)}):")
    for name in only_prod:
        print(f"  {name:<40} {prod[name]}")
    print(f"\nCandidate only ({len(only_cand)}):")
    for name in only_cand:
        print(f"  {name:<40} {cand[name]}")


def compare_coverage(cur, targets):
    """Where do the three tables actually overlap, in matches and in time?"""
    print(f"\n{'=' * 80}\nOverlap across the three tables (last {LOOKBACK_DAYS} days)\n{'=' * 80}")

    usable = []
    for label, (schema, table), columns in targets:
        names = {c[0] for c in columns}
        time_col = primary_time_column(columns)
        if "MATCH_CODE" not in names:
            print(f"\n  {label} ({schema}.{table}): no MATCH_CODE column, skipping overlap.")
            key_like = sorted(n for n in names if n.endswith(("_CODE", "_ID")))
            print(f"    candidate join keys: {', '.join(key_like) or '(none found)'}")
            continue
        if time_col is None:
            print(f"\n  {label} ({schema}.{table}): no time column, skipping overlap.")
            continue
        usable.append((label, schema, table, time_col))

    if not usable:
        print("\n  No table pair shares MATCH_CODE plus a time column; join key needs a manual look.")
        return

    print(f"\nPer-table volume in the window (anchored on each table's own time column):")
    for label, schema, table, time_col in usable:
        _, rows = fetch_all(
            cur,
            f"""
            SELECT COUNT(*), APPROX_COUNT_DISTINCT(MATCH_CODE),
                   MIN({time_col}), MAX({time_col})
            FROM {DATABASE}.{schema}.{table}
            WHERE {time_col} >= DATEADD(day, -{LOOKBACK_DAYS}, CURRENT_TIMESTAMP())
            """,
        )
        n, matches, first, last = rows[0]
        print(f"  {label:<16} rows={n:>12,}  matches={matches:>7,}  {first}  ->  {last}")
        print(f"                   (time column: {time_col})")

    if len(usable) < 2:
        return

    ctes = []
    for i, (_, schema, table, time_col) in enumerate(usable):
        ctes.append(
            f"""t{i} AS (
                SELECT DISTINCT MATCH_CODE
                FROM {DATABASE}.{schema}.{table}
                WHERE {time_col} >= DATEADD(day, -{LOOKBACK_DAYS}, CURRENT_TIMESTAMP())
            )"""
        )

    print(f"\nShared matches in the window (all {SPORT_CODE} matches, and pairwise):")
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            _, rows = fetch_all(
                cur,
                f"""
                WITH {ctes[i]}, {ctes[j]}
                SELECT COUNT(*) FROM t{i} JOIN t{j} USING (MATCH_CODE)
                """,
            )
            print(f"  {usable[i][0]} & {usable[j][0]}: {rows[0][0]:,} matches")

    if len(usable) == 3:
        _, rows = fetch_all(
            cur,
            f"""
            WITH {ctes[0]}, {ctes[1]}, {ctes[2]}
            SELECT COUNT(*) FROM t0 JOIN t1 USING (MATCH_CODE) JOIN t2 USING (MATCH_CODE)
            """,
        )
        print(f"  all three: {rows[0][0]:,} matches  <-- the population any three-way comparison runs on")

    joined = " UNION ALL ".join(
        f"SELECT '{label}' AS SRC, MATCH_CODE FROM t{i}" for i, (label, _, _, _) in enumerate(usable)
    )
    _, rows = fetch_all(
        cur,
        f"""
        WITH {', '.join(ctes)}, all_matches AS ({joined})
        SELECT e.SPORT_CODE, a.SRC, COUNT(DISTINCT a.MATCH_CODE)
        FROM all_matches a
        JOIN {DATABASE}.SHARED.EVENT e ON e.MATCH_CODE = a.MATCH_CODE
        GROUP BY 1, 2
        ORDER BY 1, 2
        """,
    )
    print("\nMatches by sport and source (is the candidate stream {} only?):".format(SPORT_CODE))
    for sport, src, n in rows:
        print(f"  {str(sport):<8} {src:<16} {n:,}")


def latency_readiness(cur, resolved):
    """Size up the inbound -> outbound relationship before measuring latency.

    PRICE_CHANGES is our outbound feed, GAMEPLAI_STREAM the inbound one,
    so the gap between them should be fractional. Two things have to hold
    before any such number means anything, and both are checked here:

      1. The timestamps must share a clock. If the inbound time is
         stamped by GAMEPLAI and the outbound by us, the per-match offset
         below is latency PLUS clock skew, and a negative offset (prices
         out before quotes in) is the tell that they do not share one.
      2. The message counts must be comparable. If we publish far fewer
         rows per match than we receive, prices are being throttled or
         filtered, and a per-tick latency figure would be measuring only
         the subset that survived.

    This is a per-match sketch, not the audit itself -- matching a price
    change to the quote that caused it needs the market/selection key,
    which the schema dump above is there to establish.
    """
    by_label = {label: (schema, table, columns) for label, (schema, table), columns in resolved}
    pairs = [("prod model", "price changes"), ("candidate model", "price changes")]

    print(f"\n{'=' * 80}\nInbound vs outbound, per match (last {LOOKBACK_DAYS} days)\n{'=' * 80}")

    for inbound_label, outbound_label in pairs:
        if inbound_label not in by_label or outbound_label not in by_label:
            continue
        in_schema, in_table, in_cols = by_label[inbound_label]
        out_schema, out_table, out_cols = by_label[outbound_label]

        in_time = primary_time_column(in_cols)
        out_time = primary_time_column(out_cols)
        in_names = {c[0] for c in in_cols}
        out_names = {c[0] for c in out_cols}

        print(f"\n--- {inbound_label} -> {outbound_label} ---")
        if "MATCH_CODE" not in in_names or "MATCH_CODE" not in out_names:
            print("  No shared MATCH_CODE; cannot pair these two without a manual key.")
            continue
        if in_time is None or out_time is None:
            print("  One side has no time column; nothing to measure.")
            continue

        print(f"  inbound time: {in_time}    outbound time: {out_time}")
        window = f"DATEADD(day, -{LOOKBACK_DAYS}, CURRENT_TIMESTAMP())"
        _, rows = fetch_all(
            cur,
            f"""
            WITH inbound AS (
                SELECT MATCH_CODE, MIN({in_time}) AS FIRST_T, MAX({in_time}) AS LAST_T,
                       COUNT(*) AS N
                FROM {DATABASE}.{in_schema}.{in_table}
                WHERE {in_time} >= {window}
                GROUP BY MATCH_CODE
            ), outbound AS (
                SELECT MATCH_CODE, MIN({out_time}) AS FIRST_T, MAX({out_time}) AS LAST_T,
                       COUNT(*) AS N
                FROM {DATABASE}.{out_schema}.{out_table}
                WHERE {out_time} >= {window}
                GROUP BY MATCH_CODE
            ), paired AS (
                SELECT DATEDIFF(millisecond, i.FIRST_T, o.FIRST_T) AS FIRST_GAP_MS,
                       DATEDIFF(millisecond, i.LAST_T, o.LAST_T) AS LAST_GAP_MS,
                       i.N AS IN_N, o.N AS OUT_N
                FROM inbound i JOIN outbound o USING (MATCH_CODE)
            )
            SELECT COUNT(*) AS MATCHES,
                   MIN(FIRST_GAP_MS), PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY FIRST_GAP_MS),
                   PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY FIRST_GAP_MS),
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY FIRST_GAP_MS),
                   MAX(FIRST_GAP_MS),
                   PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY LAST_GAP_MS),
                   COUNT_IF(FIRST_GAP_MS < 0) AS NEGATIVE_GAPS,
                   SUM(IN_N), SUM(OUT_N),
                   PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY OUT_N / NULLIF(IN_N, 0))
            FROM paired
            """,
        )
        (matches, gap_min, gap_p05, gap_p50, gap_p95, gap_max,
         last_gap_p50, negative, in_total, out_total, ratio_p50) = rows[0]

        if not matches:
            print("  No matches present in both tables in this window.")
            continue

        print(f"  matches paired: {matches:,}")
        print("  first-message offset, outbound minus inbound (ms):")
        print(f"    min={gap_min}  p05={gap_p05}  median={gap_p50}  p95={gap_p95}  max={gap_max}")
        print(f"  last-message offset, median (ms): {last_gap_p50}")
        print(f"  matches where outbound STARTS BEFORE inbound: {negative:,} / {matches:,}")
        if negative:
            print("    ^ impossible as pure latency -- these two timestamps are on different")
            print("      clocks, or the time columns do not mean what their names suggest.")
        print(f"  rows: inbound={in_total:,}  outbound={out_total:,}")
        print(f"  median per-match outbound/inbound row ratio: {ratio_p50}")
        if ratio_p50 is not None and ratio_p50 < 0.9:
            print("    ^ we publish materially fewer rows than we receive: throttled,")
            print("      filtered, or deduplicated. Establish which before reading latency.")


def resolve_tables(cur, requested):
    """Confirm each table exists; if not, show the closest real names."""
    resolved = []
    for label, (schema, table) in requested:
        meta = table_metadata(cur, schema, table)
        if meta is None:
            print(f"\n!! {label}: {DATABASE}.{schema}.{table} not found.")
            similar = find_similar_tables(cur, f"%{table.split('_')[0]}%", "%PRICE%", "%QUOTE%")
            if similar:
                print("   Tables with similar names:")
                for s, t, n, altered in similar:
                    rows = f"{n:,}" if n is not None else "?"
                    print(f"     {s}.{t:<45} rows={rows:>14}  last_altered={altered}")
            else:
                print("   No similarly named tables found.")
            return None
        resolved.append((label, (schema, table), describe_columns(cur, schema, table), meta))
    return resolved


def parse_overrides(argv):
    tables = {"prod model": PROD_MODEL, "candidate model": CANDIDATE_MODEL, "price changes": PRICE_CHANGES}
    for arg in argv:
        if "." not in arg:
            sys.exit(f"Table override must be SCHEMA.TABLE, got: {arg}")
        schema, table = arg.split(".", 1)
        name = table.upper()
        if "CANDIDATE" in name:
            tables["candidate model"] = (schema.upper(), name)
        elif "PRICE" in name:
            tables["price changes"] = (schema.upper(), name)
        else:
            tables["prod model"] = (schema.upper(), name)
    return list(tables.items())


def main():
    requested = parse_overrides(sys.argv[1:])
    print(f"Database: {DATABASE}")
    for label, (schema, table) in requested:
        print(f"  {label:<16} {schema}.{table}")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            resolved = resolve_tables(cur, requested)
            if resolved is None:
                sys.exit("\nFix the table name above (edit the constants or pass SCHEMA.TABLE) and re-run.")

            for label, (schema, table), columns, meta in resolved:
                inspect_table(cur, label, schema, table, columns, meta)

            by_label = {label: columns for label, _, columns, _ in resolved}
            compare_schemas(by_label["prod model"], by_label["candidate model"])
            trimmed = [(label, t, columns) for label, t, columns, _ in resolved]
            compare_coverage(cur, trimmed)
            latency_readiness(cur, trimmed)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
