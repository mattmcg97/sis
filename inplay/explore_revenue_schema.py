"""Discover column layout and sample rows for the customer/revenue
tables, ahead of a P&L investigation: where is the book losing money on
AF moneyline/spread/total markets, segmented by game state (score,
time/period remaining), and are there session effects across a gamer's
consecutive matches?
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"

TABLES = [
    ("SHARED", "CUSTOMER_REVENUE"),
    ("SHARED", "CUSTOMER_REVENUE_EVENT"),
    ("SHARED", "CUSTOMER_REVENUE_EVENT_SELECTION"),
    ("SHARED", "CUSTOMER_MARKET_EXCEPTIONS"),
    ("SHARED", "MARKET_CUSTOMER"),
    ("SHARED", "SELECTION_CUSTOMER"),
]

SAMPLE_ROWS = 3


def describe_columns(cur, schema, table):
    cur.execute(
        f"""
        SELECT column_name, data_type, is_nullable, ordinal_position
        FROM {DATABASE}.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    )
    return cur.fetchall()


def sample_rows(cur, schema, table):
    cur.execute(f'SELECT * FROM "{DATABASE}"."{schema}"."{table}" LIMIT {SAMPLE_ROWS}')
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return cols, rows


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for schema, table in TABLES:
                print(f"\n{'=' * 80}\n{schema}.{table}\n{'=' * 80}")

                columns = describe_columns(cur, schema, table)
                if not columns:
                    print("  (table not found)")
                    continue

                print("Columns:")
                for name, dtype, nullable, pos in columns:
                    print(f"  {pos:>3}  {name:<40} {dtype:<20} nullable={nullable}")

                print(f"\nSample ({SAMPLE_ROWS} rows):")
                cols, rows = sample_rows(cur, schema, table)
                if not rows:
                    print("  (no rows)")
                    continue
                for row in rows:
                    for col_name, val in zip(cols, row):
                        text = str(val)
                        if len(text) > 200:
                            text = text[:200] + "...(truncated)"
                        print(f"  {col_name}: {text}")
                    print("  ---")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
