"""List every table in a Snowflake database with basic stats.

Uses the same connection setup as snowflake_connect.py (SSO by default).
Reads the database to inspect from SNOWFLAKE_DATABASE in .env, or pass one
as a command-line argument.
"""

import sys

from snowflake_connect import get_connection

DEFAULT_DATABASE = "SIS_PROD_CG_CURATED"


def list_tables(database: str) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT table_schema, table_name, table_type, row_count,
                       bytes, created, last_altered
                FROM {database}.INFORMATION_SCHEMA.TABLES
                ORDER BY table_schema, table_name
                """
            )
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description]
    finally:
        conn.close()

    if not rows:
        print(f"No tables found in {database}.")
        return

    widths = [
        max(len(str(row[i])) for row in rows + [columns]) for i in range(len(columns))
    ]
    header = "  ".join(col.ljust(w) for col, w in zip(columns, widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(val).ljust(w) for val, w in zip(row, widths)))

    print(f"\n{len(rows)} tables in {database}")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATABASE
    list_tables(db)
