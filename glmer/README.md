# glmer — mixed-effects pre-match model

An alternative to the NB2 pre-match model (`nb2/`) that eAMFModel's
versions use for each match's prior. It is written in R with `lme4::glmer`.
v8 to v10 can price off it with `v9-build --prior glmer` (see
[In eAMFModel's versions](#in-eamfmodels-versions)).

**The chosen model is the global one** with the `form` features and a
60-day row half-life: `fit.R`'s defaults. It is a single Poisson GLMM over
every side of every match, with:
- random effects for the gamer's attack, the opponent's defence, both NFL
  teams and the stream;
- home/away;
- both sides' recent form.

Every player's rating is shrunk toward the league by how much data they
have, so a new player starts as an average one. The per-player models
below are still here to backtest, but they lost to the global model (see
[First findings](#first-findings-29-sep-2026-nb2amfelocsv)).

The two parts:

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
Rscript glmer/backtest.R --feature-sets=form --weightings=hl30,hl60,hl120 --form-half-lives=5,10,20 \
    --players=FALSE --refit-days=14                                   # the half-life sweep
```

`--form-half-lives` adds the form features' own half-life (in matches) to
the grid; the outputs then call the feature set `form_f5`, `form_f10`, and
so on. `--scalars=0.25,0.5` does the same for the weight scalar: each scalar
is applied to every weighting, named `hl60_x0.25` and so on. `--players=FALSE`
fits the global model only, which scores mode `global` and saves the
per-player fits' time.

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
| `predictions.csv.gz` | every test match priced by every configuration                            |
| `calibration.csv` | moneyline calibration in probability deciles, per configuration              |
| `by_player.csv`   | per player, errors on points scored and conceded in each mode, and the change against global (negative means their own model helps) |
| `fits.csv`        | for each fit: rows, variance components, how many per-player models, warnings, seconds taken |

### 2. Fit the chosen configuration: `fit.R`

```bash
Rscript glmer/fit.R                                                  # form / hl60 / global: best so far
Rscript glmer/fit.R --feature-set=home --weighting=hl60 --mode=blend
Rscript glmer/fit.R --before=2026-09-24 --out=glmer/out/model_0924   # fit only on matches before a date
Rscript glmer/fit.R --form-half-life=20 --weighting=hl30             # other half-lives
Rscript glmer/fit.R --scalar=0.5                                     # another weight scalar
```

Per-player models are fitted only when `--mode` needs them, unless
`--players=TRUE` or `--players=FALSE` says otherwise. The default
global mode fits the global model alone.

`fit.R` writes these files to `glmer/out/model/`:

| file                 | what                                                                     |
|----------------------|--------------------------------------------------------------------------|
| `model.rds`          | everything `predict.R` needs                                             |
| `player_ratings.csv` | each player's global attack and defence effects, plus a summary of their own models |
| `effects.csv`        | every random effect in the global model (players, teams, streams, …)     |
| `fit_summary.txt`    | formulas, variance components, fixed effects and warnings                |
| `model_info.json`    | the configuration and cut-off, read by `eAMFModel/glmer_prior.py`        |

### 3. Price matches: `predict.R`

```bash
Rscript glmer/predict.R --schedule=schedule.csv
Rscript glmer/predict.R --model=glmer/out/model_0924/model.rds       # no schedule: every history match after the cut-off
Rscript glmer/predict.R --schedule=schedule.csv --n-sims=0           # expected points only, no moneyline
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
| `form_noexp`  | `form` without the experience terms (tested against newcomers' pricing; worse overall)    |
| `form_session`| `form` + how both sides are doing this session, and where in it they are                  |
| `form_clock`  | `form` + late in a session and time of day (helps totals a little)                       |
| `form_shift`  | `form_clock` + each gamer's own start to a session                                       |
| `form_pair`   | `form` + gamer-v-gamer matchups, `(1\|Player:OpponentPlayer)` (tested; about level)      |
| `matchup`     | + each player's own team preference, and team-vs-team matchups                            |
| `home_offset` | `home`, but the per-player models only learn corrections to the global model's prediction (`offset(GlobalEta)`) |

Every feature uses only matches that started before the one being priced;
a match's own result never leaks into its features.

**Experience.** `ExpLog` is the log of a gamer's matches so far.
- `EXP_CAP_MATCHES` (`fit.R --exp-cap`) caps it in the fit. The default is
  no cap.
- `EXP_FLOOR_MATCHES` (`predict.R --exp-floor`) prices a gamer as if they
  had played at least that many matches. The default is 30. It applies only
  to the rows being priced (`predict.R`, and `backtest.R`'s test rows).
- Without the floor, glmer extrapolates its learning curve for a newcomer
  joining a settled league. It predicts them about half a veteran's points,
  and out of sample they fall far less behind. See "glmer and new gamers" in
  `eAMFModel/README.md` for the walk-forward check behind the floor. The available columns
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

## First findings (29 Sep 2026, `nb2/AMFELO.csv`)

These are out of sample on NB2's own split: train on the first 75% of
matches (to 6 Jul 2026), test on the 5,988 after it (6 Jul – 10 Sep).
Settings were `--nagq=0` and 5,000 draws per match. "Totals bias" is actual
minus predicted.

**One fit at the split**, as `nb2/backtest_nb2_calibration.py` does it.
Scored on the 5,919 test matches whose players and teams were all in
training, which is the subset NB2 prices:

| model                     | Brier      | log loss   | totals bias | totals RMSE | totals corr |
|---------------------------|------------|------------|-------------|-------------|-------------|
| NB2 (`NBRatingTrial.py`)  | 0.2422     | 0.6782     | +1.68       | 12.72       | 0.269       |
| glmer `nb2` / hl60 global | 0.2427     | 0.6793     | +0.70       | 12.63       | 0.270       |
| glmer `nb2` / hl60 blend  | 0.2411     | 0.6751     | +0.91       | 12.64       | 0.264       |
| glmer `form` / hl60 global| **0.2371** | **0.6669** | +0.52       | **12.40**   | **0.316**   |
| glmer `form` / hl60 blend | 0.2384     | 0.6694     | +0.35       | 12.41       | 0.313       |

**Refit every 14 days** (`--refit-days=14`), scored on all 5,988 test
matches:

| model                     | Brier      | log loss   | totals bias | totals RMSE | totals corr |
|---------------------------|------------|------------|-------------|-------------|-------------|
| glmer `nb2` / hl60 global | 0.2384     | 0.6695     | +0.50       | 12.48       | 0.294       |
| glmer `nb2` / hl60 blend  | 0.2386     | 0.6700     | +0.64       | 12.48       | 0.291       |
| glmer `form` / hl60 global| **0.2357** | **0.6638** | +0.22       | **12.32**   | **0.327**   |
| glmer `form` / hl60 blend | 0.2373     | 0.6672     | +0.07       | 12.32       | 0.325       |
| glmer `form` / hl60 pair  | 0.2423     | 0.6790     | −0.16       | 12.46       | 0.297       |

**Half-life sweep** of the global `form` model. Refit every 14 days, all
5,988 test matches; each cell is log loss / totals RMSE:

| form half-life ↓ · row half-life → | 30 days         | 60 days             | 120 days        |
|------------------------------------|-----------------|---------------------|-----------------|
| 5 matches                          | 0.6637 / 12.35  | 0.6633 / 12.34      | 0.6639 / 12.34  |
| 10 matches                         | 0.6640 / 12.33  | **0.6634 / 12.32**  | 0.6635 / 12.32  |
| 20 matches                         | 0.6639 / 12.32  | 0.6637 / 12.32      | 0.6642 / 12.33  |

Paired match by match against the defaults (10 matches, 60 days), nothing
does better:
- **5 matches / 60 days** is level on log loss: −0.0001, t −0.3, better in
  2 of 5 fortnights. It is worse on totals (t +3.5).
- **Every other cell** is worse on log loss, by up to +0.0008 (t +2.0 at 20
  matches / 120 days).

So the defaults stay. The whole grid spans less than 0.001 in log loss.
That is about the size of the difference between two machines fitting the
same model: fits are exact on one machine, but the optimiser stops a hair
apart on another.

**Weight scalar.** Same model and refits: form half-life 10 matches, rows
60 days. The scalar multiplies every row weight after the weights are
normalised to mean 1:

| scalar | Brier  | log loss | vs scalar 1 (t, fortnights better) | totals RMSE | totals bias | sd of win prob |
|--------|--------|----------|------------------------------------|-------------|-------------|----------------|
| 0.5    | 0.2352 | 0.6628   | −0.0006 (−1.1, 4 of 5)             | 12.31       | +0.15       | 0.121          |
| 1      | 0.2355 | 0.6634   | n/a                                | 12.32       | +0.22       | 0.107          |
| 2      | 0.2360 | 0.6643   | +0.0009 (+2.5, 0 of 5)             | 12.34       | −0.06       | 0.100          |
| 4      | 0.2364 | 0.6653   | +0.0019 (+3.7, 0 of 5)             | 12.40       | −0.83       | 0.096          |

- **What a bigger scalar does.** It tells the model it has more data than it
  really does. From scalar 0.5 to 4, the player ratings spread further
  (player sd 0.160 → 0.204, opponent 0.125 → 0.155). The per-row
  overdispersion grows much more (0.25 → 0.55). Our pricing simulates with
  that overdispersion, so moneylines drift toward 50% and expected totals
  rise.
- **The data's real weight.** The 60-day weights' effective sample size
  (Kish) is only 62–71% of the rows. A scalar of about 0.65 would count the
  data at its true weight.
- **So 0.5 edges 1,** though not significantly on log loss. It is
  significantly better on totals (t −2.4).
- **Stream is noise here.** Its variance component isn't identified with
  only three streams: it jumps between 0 and 2.5 across refits, while the
  stream effects stay near 0.

**Scalar × half-life sweep.** Scalars 0.25 / 0.5 / 0.75 crossed with form
half-life 5 / 10 / 20 matches and row half-life 30 / 60 / 120 days: 135
fits, same refits as above. The scalar-1 rows come from the half-life
sweep. Log loss, averaged over the three form half-lives:

| scalar | rows 30 days | rows 60 days | rows 120 days |
|--------|--------------|--------------|---------------|
| 0.25   | 0.6642       | 0.6669       | 0.6674        |
| 0.5    | **0.6633**   | **0.6630**   | **0.6637**    |
| 0.75   | 0.6638       | 0.6633       | 0.6638        |
| 1      | 0.6639       | 0.6635       | 0.6639        |
| ESS / rows | 0.39     | 0.67         | 0.88          |

- **Best single setting: form 10 matches / rows 60 days / scalar 0.5.**
  Log loss 0.6628, −0.0006 against the defaults (t −1.1, better in 4 of 5
  fortnights); totals RMSE 12.31 (t −2.4).
- **Scalar 0.25 is too low** unless the rows decay fast. With 60- or
  120-day rows it costs 0.003–0.004 (t ≈ +2).
- **The pattern follows the effective sample size.** The faster the rows
  decay, the less real data there is, and the lower the scalar the model
  wants. At 120 days, 0.5–1 are level.
- **The half-lives stay flat at every scalar.** Form 5–20 matches and rows
  30–120 days all land within about 0.001 once the scalar is 0.5 or more.

What the numbers say:

- **The form features are the gain.** These are both sides'
  recency-weighted points scored and conceded, updated after every result.
  - With one fit, form keeps updating through the test window while static
    ratings can't, and it beats NB2 by 0.011 in log loss.
  - Refitting every 14 days narrows the gap, but form still wins
    (0.6638 against 0.6695).
  - It is also the best calibrated: each probability decile is within ~4
    points. NB2's outer deciles are off by +9.4 and −6.5, overconfident at
    both ends.
- **Per-player models don't beat the global model.**
  - On their own (`attack`, `defence`, `pair`) they are worse in every
    feature set. Each sees only its own player's matches, so it judges
    opponents from a handful of meetings, and its intercept and slopes
    aren't shrunk.
  - Blended with global, they help only where the global model has no form
    and isn't refitted (one-fit `nb2`: 0.6751 against 0.6793), which means
    what they add is recent form. The `form` global model captures that
    better.
  - `backtest.R`'s per-player table still shows which individual players'
    own models beat the global one.
- **The other features add nothing measurable.** Home/away, rest, session,
  time of day and team matchups make no difference. A 60-day half-life and a
  150-match one are within noise of each other.
- **Guard rails.** SHARK had just passed 60 matches and had a conceding
  form that had barely moved. Their defence model fitted a slope of −9.5 on
  it, then extrapolated to 90+ expected points against them. Every model's
  numeric covariates are now clamped to its own training range, and a
  player's own prediction is capped within `PLAYER_MAX_SHIFT` of the global
  one. In the refit run the cap bound on 2–79 per-player predictions per
  fold, out of ~5,000. The per-player rows in the one-fit table predate the
  guards; the global rows don't depend on them.

## Team ratings and in-session form (7 Oct 2026)

**A colleague's Madden team ratings** gave each team an overall, offence, pass and run offence,
deep threat and defence score, with notes. They were tested against `nb2/AMFELO.csv` and glmer's
out-of-sample errors (12 fortnights of 14-day refits, 15,418 matches).
- **The tiers are the league's pools.** Tier 1 (PHI, DET, BAL, KC, SF) and Tier 2 (BUF, DEN, CIN,
  WAS, MIN) only ever play within their own tier.
- **Raw results run backwards.** PHI, rated best, wins 43% of its games, and WAS, rated worst in
  its tier, wins 54%. Weaker gamers are given the stronger teams. A gamer's strength correlates
  about −0.25 with their team's in every month since January.
- **Net of the gamer, the ratings are right.** Least squares on margins, with both gamers and both
  teams, gives team strengths that correlate 0.93 with Overall within the tiers. That is about 0.5
  points of margin per rating point, and it is steady from December to September.
- **glmer already has this.** Gamers rotate teams (a median of 9 of the 10 in their last 6,000
  matches), so its crossed player and team effects separate the two. Its fitted team effects follow
  the ratings, and a gamer's rating is net of the teams they were given.
- **Nothing left over in glmer's errors.**
  - Overall: +0.07 points per rating point (t 2.7), 0.02% of the error variance.
  - Pass or run offence against the opponent's defence, offensive style, and deep threats: all
    under t 1.
  - The 40 team pairings spread exactly as chance would (sd of pair z-scores 0.99).
  - CIN, Tier 2's only deep threat, scored 1 point under its price against DEN (z −3.7), as the
    note on DEN's corner says. With 40 pairings tested it is borderline.

So no rating feature was added.

**In-session form.** A gamer's errors against glmer correlate about as much across sessions as
within one: +0.013 for matches under 2 hours apart, +0.018 for matches 4 to 16 days apart. Losing
streaks run a little worse than glmer says: 38.4% won against 41.5% priced after losing 6 of 8.
That is the gamer's level drifting from the fitted ratings over days, not a same-night effect. A
second layer that follows each gamer's results since the fit catches it, for NB2 as well as glmer.
See "Following the results since the fit" in `eAMFModel/README.md`.

## Gamer-v-gamer and team-v-team matchups (7 Oct 2026)

Neither is in the production `form` model. Its player, opponent and team effects price each side's
own strength, but no pairing.
- **Team v team: nothing.** A team pairing's errors against glmer don't repeat from one half of the
  season to the other (20 pairings with enough matches in both). `matchup`'s `(1|TeamPair)` made no
  difference either.
- **A gamer with or against a particular team: nothing worth having.**
- **Gamer v gamer: small, and not steady.**
  - A pair's errors repeat a little across the season (split-half correlation +0.13 over 252
    pairs, with each gamer's own offset taken out in each half).
  - As an add-on on top of glmer and the follow layer, each pair's average leftover error over
    its earlier meetings, shrunk by 80 meetings, took 0.0013 off moneyline log loss on fortnights
    6–11, in all 6. But the shrinkage was chosen on those same fortnights.
  - Fitted inside glmer, `form_pair` adds `(1|Player:OpponentPlayer)` (sd 0.06 on the log scale)
    and was refitted on the same six fortnights. Log loss went 0.6632 → 0.6630 alone and
    0.6616 → 0.6612 with the follow layer. Margin RMSE fell 0.006 and total RMSE 0.009. It was
    better in only 2 of the 6 fortnights alone, and 3 with the follow layer.

So `form` stays the default. `form_pair` is there to try (`fit.R --feature-set=form_pair`).

## A gamer's day and session (8 Oct 2026)

Does a gamer have good and bad days on top of their level and form? Measured instead of guessed:
- **As variance components.** `form` plus `(1|DayP) + (1|DayO) + (1|SessP) + (1|SessO)`, each a
  gamer's scoring or conceding deviation that day or session. It was fitted by ML on Jun–Aug 2026
  (8,920 matches, hl60). Points scored vary a little by day: sd 0.043 on the log scale, about 0.9
  points. Conceding, and both session terms, come out at or near zero. The log likelihood gains
  1.55 for 4 more parameters, which is not significant. After six games each 7 points short, the
  implied update moves a gamer about half a point.
- **As an S-curve.** A day is normal, off or on, with the state hidden and fitted by EM on the daily
  glmer's out-of-sample errors. The chance a gamer is off rises as their errors pile up, so the
  shift starts small and climbs toward a cap.
  - Fitted on 16 Jul – 9 Aug: off on 5% of gamer-days, at −3.9 points. Twelve points short each
    game moves a gamer −0.2, −0.4, −0.7, −1.0, −1.5, −2.0 over games 1–6.
  - Fitted on 10 Aug – 3 Sep: no off or on days at all.
  - Scored on the other half: moneyline log loss +0.0003 and Brier +0.0002 against the daily glmer
    alone.

So the daily refit plus `form` already holds what there is. No day or session term was added.

## Shifts, time of day and weekdays (9 Oct 2026)

Out-of-sample errors (Mar–Sep, 15,418 matches) show no day-of-week effect for anyone, and late
fades don't repeat from odd weeks to even weeks. Two patterns do:
- **Totals by time of day, and late in a session.** Totals run lower than priced at 04–08h and
  16–20h UTC, and from a side's 8th game of a session on.
- **Slow and fast starters.** A gamer's first two games of a session repeat from odd weeks to even
  weeks (r +0.37 on points scored). About a third of a gamer's measured slow start carries forward.

Two feature sets test them, refitted on fortnights 6–11 (7,409 matches) and compared with `form`,
each with the follow layer:

| set | ML log loss | Brier | margin RMSE | total RMSE | games 1–2: ML | 8th+: total RMSE |
|---|---|---|---|---|---|---|
| `form_clock`: + `SessLate`, `(1\|HourBlock)` | −0.00006 | −0.00002 | +0.001 | −0.010 | −0.00009 | −0.029 |
| `form_shift`: + each gamer's own start, `(0 + SessEarly\|Player)` | −0.00009 | −0.00003 | +0.002 | −0.009 | −0.00041 | −0.027 |

`form_clock` improves totals in 5 of 6 fortnights. Its fitted effects are small: a gamer's own
scoring late in a session falls by about 1% (0.25 points), and time of day varies by sd 0.02 (0.4
points a side). `form_shift`'s per-gamer start (sd 0.027 on the log scale, 0.5 points) helps a
little in games 1–2 and hurts a little from the 8th on; it was better in only 3 of 6 fortnights.
Both read the clock, so a build with either prices pre-match quotes off the kick-off
(`glmer_prior.CLOCK_TERMS`). `form` stays the default.

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

## In eAMFModel's versions

`python -m eAMFModel v9-build ... --history <csv> --prior glmer` (or v8, v10) prices the version's
pre-match prior off this model in place of NB2. `eAMFModel/glmer_prior.py`
runs `fit.R` at build time and `predict.R --n-sims=0` whenever the version
needs expected points. It uses whatever `fit.R`'s defaults and `config.R`
say, so changing the chosen half-lives here changes v7's prior at its next
build. See the v7 section of `eAMFModel/README.md`.

Its form follows the results wherever it prices, the calibrator included.
- The calibrator fetches every match settled by the window's end, and
  `predict.R` reads them alongside the model's own history.
- Each pre-match quote reads only the results in when it was published, so
  a gamer's back-to-back match still being played is never seen early.
- `model_info.json` records the global formula. A build whose features read
  the clock (rest, session, hour) prices every pre-match quote off the
  kick-off instead.

See "Form follows results, as it would live" in `eAMFModel/README.md`.
`python -m eAMFCalibrator bets prematch` sets NB2 and glmer builds side
by side on every pre-match bet over many weeks (see
`eAMFCalibrator/README.md`).
