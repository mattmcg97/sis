# sis

Personal workspace for work-related scripts and utilities.

## Snowflake connection

`inplay/snowflake_connect.py` opens a connection to Snowflake using
credentials from environment variables (never hardcoded).

By default it authenticates via SSO (`SNOWFLAKE_AUTHENTICATOR=externalbrowser`):
running the script opens your default browser, you log in through your org's
identity provider, and no password is needed. To use a password instead, set
`SNOWFLAKE_AUTHENTICATOR=snowflake` and provide `SNOWFLAKE_PASSWORD`.

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your account and user
```

### Run

```bash
python inplay/snowflake_connect.py
```

A browser window opens for SSO login; it prints the Snowflake version once
connected. `.env` is gitignored so credentials never get committed.

## Folders

- `inplay/` — American football in-play model investigation: raw feed
  exploration, book P&L analysis, and GAMEPLAI calibration checks against
  realized outcomes. All scripts here share `inplay/snowflake_connect.py`
  for the connection, so run them as `python inplay/<script>.py` from the
  repo root.
