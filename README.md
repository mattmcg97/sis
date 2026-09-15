# sis

Personal workspace for work-related scripts and utilities.

## Snowflake connection

`snowflake_connect.py` opens a connection to Snowflake using credentials from
environment variables (never hardcoded).

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your account, user, password, etc.
```

### Run

```bash
python snowflake_connect.py
```

It prints the Snowflake version if the connection succeeds. `.env` is
gitignored so credentials never get committed.
