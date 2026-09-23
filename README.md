# sis

Personal workspace for work-related scripts and utilities.

## Snowflake connection

`analysis/snowflake_connect.py` opens a connection to Snowflake using
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
python analysis/snowflake_connect.py
```

A browser window opens for SSO login; it prints the Snowflake version once
connected. `.env` is gitignored so credentials never get committed.

## Folders

- `analysis/` — American football in-play model investigation: raw feed
  exploration, book P&L analysis, and GAMEPLAI calibration checks against
  realized outcomes. All scripts here share `analysis/snowflake_connect.py`
  for the connection, so run them as `python analysis/<script>.py` from the
  repo root. `inspect_model_and_price_tables.py` is the starting point for
  both open GAMEPLAI questions -- candidate vs prod (a model comparison) and
  GAMEPLAI_STREAM vs PRICE_CHANGES (a latency audit of our own outbound
  feed). It profiles all three tables, diffs the two model schemas, reports
  where they overlap in matches and in time, and sketches the per-match
  timing offset and message-volume ratio between inbound quotes and
  outbound prices.
- `eAMFCalibrator/` — Calibration suite for the GAMEPLAI in-play models on
  eAMF. Takes either stream (prod or candidate) as input, samples one
  snapshot per drive, pairs it with the model quote within 3 seconds, and
  scores predicted against realized across score-difference / quarter /
  possession cells. Re-runnable: each run writes CSVs that `compare` diffs,
  so prod vs candidate (or the same stream week on week) is one command.
  See `eAMFCalibrator/README.md`.
- `eAMFModel/` — A rival in-play pricer for moneyline, spread and total. It
  is top-down with no machine learning: a pre-match anchor, a scoreboard
  update, and exact Markov chains over possessions and over the current
  drive. It suspends around feed garbage. The calibrator runs it in the
  candidate's place with `--candidate v1`. See `eAMFModel/README.md`.
- `nb2/` — Pre-match NB2 rating model (Adrian's): fitting (`NBRatingTrial.py`),
  schedule pricing (`NB2_schedule_predict.py`), and out-of-sample calibration
  backtests in both Python and R (`backtest_nb2_calibration.py`,
  `backtest_nb2_halflife.R`). `AMFELO.csv` is the full historical match
  dataset both the Python and R fits are trained and tested on.
