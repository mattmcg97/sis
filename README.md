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
- `eAMFModel/` — A rival in-play pricer for moneyline, spread and total. From
  each `PLAY_OVER` snapshot it simulates the rest of the game play by play on
  the real clock, starting from our own pre-match model (NB2 or glmer). The
  versions are v8, v9 and v10. The calibrator runs one in the candidate's
  place with `--candidate v9`. See `eAMFModel/README.md`.
- `nb2/` — Pre-match NB2 rating model (Adrian's): fitting (`NBRatingTrial.py`),
  schedule pricing (`NB2_schedule_predict.py`), and out-of-sample calibration
  backtests in both Python and R (`backtest_nb2_calibration.py`,
  `backtest_nb2_halflife.R`). `AMFELO.csv` is the full historical match
  dataset both the Python and R fits are trained and tested on.
- `glmer/` — An alternative pre-match model in R (`lme4::glmer`).
  - **The model.** A global mixed-effects model with each player's attack
    and defence, teams, stream and both sides' recent form. It beat NB2
    out of sample. Separate per-player models are there to backtest; they
    lost to the global one.
  - **The scripts.** `backtest.R` grids feature sets × row weightings out of
    sample, `fit.R` fits the chosen one, and `predict.R` prices a schedule
    into the same expected-points columns NB2 gives the eAMFModel versions.
  - **In v8–v10.** `python -m eAMFModel v9-build ... --prior glmer` prices the
    version's pre-match prior off it in place of NB2.

  See `glmer/README.md`.
