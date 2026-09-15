"""Simple Snowflake connection example.

Credentials are read from environment variables so nothing sensitive is
hardcoded or committed. Copy .env.example to .env and fill in your values,
or export the variables in your shell before running.
"""

import os
import sys

from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

REQUIRED_VARS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PASSWORD",
]


def get_connection() -> snowflake.connector.SnowflakeConnection:
    missing = [var for var in REQUIRED_VARS if not os.environ.get(var)]
    if missing:
        sys.exit(f"Missing required environment variables: {', '.join(missing)}")

    return snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"],
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=os.environ.get("SNOWFLAKE_DATABASE"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
    )


def main() -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT CURRENT_VERSION()")
            print("Connected to Snowflake, version:", cur.fetchone()[0])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
