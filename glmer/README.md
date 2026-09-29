# glmer — per-player mixed-effects pre-match model

An alternative to the NB2 pre-match model (`nb2/`) that eAMFModel's v4, v5
and v6 use for each match's prior. It is written in R with `lme4::glmer`
and has two parts:

- **A global model.** One Poisson GLMM over every side of every match, with
  random effects for the gamer, the opponent, both NFL teams and the stream.
  It prices every player who is new or has too little data.
- **Per-player models.** Each player with enough matches
  (`MIN_PLAYER_MATCHES`, default 60) gets two small GLMMs of their own:
  - an **attack** model, fitted on the points they score;
  - a **defence** model, fitted on the points they concede.

For a side's expected points there are four **modes**:

| mode      | the side's points come from                                        |
|-----------|---------------------------------------------------------------------|
| `global`  | the global model only                                               |
| `attack`  | the scorer's attack model                                           |
| `defence` | the opponent's defence model                                        |
| `pair`    | both of those, averaged on the log scale                            |
| `blend`   | `pair` shrunk toward global: `BLEND_WEIGHT` (default 0.5) on the per-player side |

In every mode, a side with no per-player model falls back to the global
model, and so does a row that a per-player model can't price.

The per-player models see each row from the modelled player's side, so
one formula means the same thing in both models. `(1|Rival)` is who they
were playing, `OwnHome` is whether they were at home, and `OwnFormFor` is
their own scoring form. An attack model reads those columns against the
points the player scored; a defence model reads them against the points
they conceded.

`backtest.R` scores every mode. The output has the same shape as NB2's:
`Pred_P1_Points` and `Pred_P2_Points` are each side's expected points,
the numbers the versions take from `nb2_prior.Prematch.means()`.

## The model

For each side of a match (the scorer against the opponent):

```
Score ~ Poisson(exp(eta + m + e))
  eta   fixed + random effects from the feature set (config.R)
  m     (1|MatchId)  shared by both sides of the match: pace/tempo, so the scores correlate
  e     (1|ObsID)    one per row: NB-like overdispersion
```

- **Expected points** are `exp(eta + (sd_m² + sd_e²)/2)`.
- **The moneyline** is simulated. Each draw takes `m` once per match for
  both sides; ties split half/half, as in the NB2 backtests.
- **Overdispersion.** Per-player models have no match term, so their row sd
  also carries the match swing. The global model's match sd is taken out of
  it before pricing.

This is the same Poisson-plus-observation-level-effect approach as
`nb2/glmer_poisson_settings.R`, extended to per-player models.

## Setup

1. Install R (4.x). RStudio is optional.
2. Install the packages from the repo root: `Rscript glmer/install_packages.R`.
   That is just lme4 and nloptr, and CRAN has binaries for both.
3. Get a match history. Either option works:
   - `python -m eAMFCalibrator history --until YYYY-MM-DD` writes
     `eAMFCalibrator/out/match_history.csv`, which is picked up automatically;
   - or pass any CSV shaped like `nb2/AMFELO.csv` with `--history`.
   If neither exists, the scripts fall back to `nb2/AMFELO.csv`
   (Dec 2025 to Sep 2026, about 24k matches).

## Running

All three scripts take `--key=value` options. In RStudio, open a script,
edit its `DEFAULTS` list at the top, and Source it. The working directory
doesn't matter.

### 1. Backtest the grid: `backtest.R`

```bash
Rscript glmer/backtest.R                                              # every feature set x weighting
Rscript glmer/backtest.R --feature-sets=home,form --weightings=hl30,hl60 --cores=6
Rscript glmer/backtest.R --test-from=2026-07-01 --refit-days=14       # walk forward, refit every 14 days
Rscript glmer/backtest.R --nagq=0                                     # quicker first sweep
```

For each feature set × weighting, the backtest does three things:
1. fits the global model and every player's models on the matches before
   the test period;
2. prices the test matches in each mode;
3. scores them against the results.

The default split trains on the first 75% of matches and tests on the
rest. That is the same split as `nb2/backtest_nb2_calibration.py`, so the
Brier, log loss and totals numbers compare directly with NB2's.
`--refit-days` walks forward instead, which is slower and closer to how the
model would run in production.

It writes these files to `glmer/out/backtest/`:

| file              | what                                                                         |
|-------------------|------------------------------------------------------------------------------|
| `summary.csv`     | one row per feature set × weighting × mode, for every test match (`all`) and for only matches whose players and teams were all in training (`seen`, NB2's subset) |
| `predictions.csv` | every test match priced by every configuration                               |
| `by_player.csv`   | per player, errors on points scored and conceded in each mode, and the change against global (negative means their own model helps) |
| `fits.csv`        | for each fit: rows, variance components, how many per-player models, warnings, seconds taken |

### 2. Fit the chosen configuration: `fit.R`

```bash
Rscript glmer/fit.R --feature-set=home --weighting=hl60 --mode=pair
Rscript glmer/fit.R --before=2026-09-24 --out=glmer/out/model_0924   # fit only on matches before a date
```

`fit.R` writes these files to `glmer/out/model/`:

| file                 | what                                                                     |
|----------------------|--------------------------------------------------------------------------|
| `model.rds`          | everything `predict.R` needs                                             |
| `player_ratings.csv` | each player's global attack and defence effects, plus a summary of their own models |
| `effects.csv`        | every random effect in the global model (players, teams, streams, …)     |
| `fit_summary.txt`    | formulas, variance components, fixed effects and warnings                |

### 3. Price matches: `predict.R`

```bash
Rscript glmer/predict.R --schedule=schedule.csv
Rscript glmer/predict.R --model=glmer/out/model_0924/model.rds       # no schedule: every history match after the cut-off
```

The schedule is a CSV shaped like the history; final scores are optional.
The history is also read, so the form, rest and session features can look
back at each player's earlier matches.

The output is `glmer/out/predictions.csv`. Its columns `Pred_P1_Points`,
`Pred_P2_Points` and `Prediction_Status` match
`NB2_schedule_predict.py`'s output. It also has the moneyline, total,
spread, which model priced each side, and whether each player is known to
the model. If the schedule includes results, `predict.R` scores itself
against them.

## Feature sets and weightings (`config.R`)

### Feature sets

A feature set is two formula right-hand sides in lme4 syntax: one for the
global model and one for the per-player models. Add your own by copying an
entry.

| set           | adds                                                                                     |
|---------------|------------------------------------------------------------------------------------------|
| `nb2`         | NB2's own structure: player attack and defence, team attack and defence, stream           |
| `home`        | + home/away side (`IsHome`)                                                               |
| `context`     | + rest since the last match, matches this session, time of day                           |
| `form`        | + both sides' recency-weighted points scored and conceded, and experience                 |
| `matchup`     | + each player's own team preference, and team-vs-team matchups                            |
| `home_offset` | `home`, but the per-player models only learn corrections to the global model's prediction (`offset(GlobalEta)`) |

Every feature uses only matches that started before the one being priced;
a match's own result never leaks into its features. The available columns
are listed at the top of the `FEATURE_SETS` section in `config.R`.

### Weightings

Each weighting sets how the fit's rows are weighted:

| type          | weight                                                                      |
|---------------|-----------------------------------------------------------------------------|
| `none`        | all 1                                                                       |
| `exp_days`    | `2^(-age_days / half_life)`; NB2 uses 60                                    |
| `exp_matches` | `2^(-k / half_life)`, where `k` counts that player's more recent matches, so heavy players decay faster in calendar time |
| `window`      | 1 inside the last `days`; older rows are dropped                            |
| `step`        | 1 inside the last `days`, `older` otherwise                                 |

Two options work with any type:
- **`scalar`** multiplies every weight after they're normalised to mean 1.
  In a mixed model it moves the variance components, so it changes how hard
  every rating is shrunk.
- **`drop_before`** drops the early rows, such as the launch period where
  scoring ran wild.

## Speed

The global model is the slow part: two very large random effects,
`MatchId` and `ObsID`. These timings are one global `home` fit on the
first ~35k rows of `AMFELO.csv`, one core each:

| optimiser   | nAGQ | time   | log-likelihood |
|-------------|------|--------|----------------|
| `nloptwrap` | 0    | 149 s  | −119996.47     |
| `nloptwrap` | 1    | 197 s  | −119996.10     |
| `bobyqa`    | 0    | 544 s  | −119996.47     |

- **Optimiser.** `GLMER_OPTIMIZER = "nloptwrap"` (lme4's default) is the
  default here. It is about 3.6× quicker than `"bobyqa"` (minqa's, which the
  older `nb2/` R scripts use) and gives the same fit. Switch with
  `--optimizer=bobyqa`.
- **Precision.** `--nagq=0` is quicker still and makes little difference
  here. It's handy for a first wide sweep.
- **Per-player fits.** These are small: all ~180 take a minute or two
  together.
- **A backtest job.** One (feature set, weighting, fold) job, global plus
  per-player, is ~5–10 min on the full history. `backtest.R` runs one job
  per core, each using ~300–500 MB. The full default grid is 6 feature sets
  × 8 weightings = 48 jobs, a few hours on a 4-core laptop. Narrow it with
  `--feature-sets` / `--weightings` first.

## Not wired in yet

The versions still take their prior from `eAMFModel/nb2_prior.py`. The
predictions CSV carries the same columns `nb2_prior.predict()` reads from
NB2's output. Plugging this model in means a prior class that runs
`fit.R` / `predict.R` in place of `NBRatingTrial.py` /
`NB2_schedule_predict.py`, plus a flag on the `v4-build`, `v5-build` and
`v6-build` commands to choose between them.
