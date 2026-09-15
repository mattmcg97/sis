"""Simple Snowflake connection example.

Credentials are read from environment variables so nothing sensitive is
hardcoded or committed. Copy .env.example to .env and fill in your values,
or export the variables in your shell before running.

Defaults to SSO login (SNOWFLAKE_AUTHENTICATOR=externalbrowser): this opens
your default browser to authenticate through your org's identity provider
and needs no password. Set SNOWFLAKE_AUTHENTICATOR=snowflake and provide
SNOWFLAKE_PASSWORD to use password auth instead.
"""

import os
import sys

from dotenv import load_dotenv
import snowflake.connector

load_dotenv()

REQUIRED_VARS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
]


def get_connection() -> snowflake.connector.SnowflakeConnection:
    missing = [var for var in REQUIRED_VARS if not os.environ.get(var)]
    if missing:
        sys.exit(f"Missing required environment variables: {', '.join(missing)}")

    authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR", "externalbrowser")

    connect_kwargs = dict(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ["SNOWFLAKE_USER"],
        authenticator=authenticator,
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=os.environ.get("SNOWFLAKE_DATABASE"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
    )

    if authenticator == "snowflake":
        password = os.environ.get("SNOWFLAKE_PASSWORD")
        if not password:
            sys.exit("SNOWFLAKE_PASSWORD is required when SNOWFLAKE_AUTHENTICATOR=snowflake")
        connect_kwargs["password"] = password

    return snowflake.connector.connect(**connect_kwargs)


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
