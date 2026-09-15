"""Look for possession/drive-level data for AF that could support a
possessions x points-per-possession decomposition of expected points,
instead of (or alongside) Adrian's direct attack-minus-defense fit.

Table names from the earlier full inventory that sound promising for
American football specifically (field position / ball position / drive
sequence are gridiron concepts) and haven't been explored yet this
session: INPLAY_FIELD_POSITION, INPLAY_BALL_POSITION,
INPLAY_FIELD_POSITION_PERIOD, INPLAY_FIELD_POSITION_ENDPERIOD,
PERIOD_INPLAY_SEQUENCE. Row counts per AF match (from the original
inventory, ~24,080 AF matches total) are suggestive:
  INPLAY_FIELD_POSITION:  3.3M rows  / 24,080 matches =~ 137/match  (play-level?)
  INPLAY_BALL_POSITION:  43.2M rows  / 24,080 matches =~ 1795/match (continuous tracking?)
  PERIOD_INPLAY_SEQUENCE: 679K rows / 24,080 matches =~ 28/match   (possession/drive-level?)
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"

TABLES = [
    ("SHARED", "PERIOD_INPLAY_SEQUENCE"),
    ("SHARED", "INPLAY_FIELD_POSITION"),
    ("SHARED", "INPLAY_FIELD_POSITION_PERIOD"),
    ("SHARED", "INPLAY_FIELD_POSITION_ENDPERIOD"),
    ("SHARED", "INPLAY_BALL_POSITION"),
]

SAMPLE_ROWS = 5


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


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


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("=== Picking a sample settled AF match with a moderate message count ===")
            _, rows = fetch_all(cur, f"""
                SELECT MATCH_CODE FROM {DATABASE}.SHARED.EVENT
                WHERE SPORT_CODE = 'AF' AND INPLAY_EVENT_STATUS = 'SETTLED'
                ORDER BY SCHEDULED_START_TIME_UTC DESC
                LIMIT 1
            """)
            sample_match = rows[0][0]
            print(f"  Using MATCH_CODE = {sample_match}")

            for schema, table in TABLES:
                print(f"\n{'=' * 80}\n{schema}.{table}\n{'=' * 80}")
                columns = describe_columns(cur, schema, table)
                if not columns:
                    print("  (table not found)")
                    continue
                print("Columns:")
                for name, dtype, nullable, pos in columns:
                    print(f"  {pos:>3}  {name:<40} {dtype:<20} nullable={nullable}")

                has_match_code = any(c[0] == "MATCH_CODE" for c in columns)
                if has_match_code:
                    print(f"\nRow count for {sample_match}:")
                    _, cnt = fetch_all(cur, f"""
                        SELECT COUNT(*) FROM {DATABASE}.{schema}.{table} WHERE MATCH_CODE = %s
                    """, (sample_match,))
                    print(f"  {cnt[0][0]} rows")

                    print(f"\nSample ({SAMPLE_ROWS} rows) for {sample_match}:")
                    _, rows = fetch_all(cur, f"""
                        SELECT * FROM {DATABASE}.{schema}.{table}
                        WHERE MATCH_CODE = %s
                        LIMIT {SAMPLE_ROWS}
                    """, (sample_match,))
                else:
                    print(f"\nSample ({SAMPLE_ROWS} rows, no MATCH_CODE column to filter by):")
                    _, rows = fetch_all(cur, f"""
                        SELECT * FROM {DATABASE}.{schema}.{table} LIMIT {SAMPLE_ROWS}
                    """)

                cols = [c[0] for c in columns]
                for row in rows:
                    for col_name, val in zip(cols, row):
                        text = str(val)
                        if len(text) > 150:
                            text = text[:150] + "...(truncated)"
                        print(f"  {col_name}: {text}")
                    print("  ---")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
