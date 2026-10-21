# xgb — an XGBoost pre-match model, the trader deck's way

A trader's deck (`New_Pre_Preds_User_Case.pptx`) built an eBasketball pre-match model, "CV3", from
XGBoost with 184 features:
- **Features:** each player's all-time and 5/15/50-game points per possession, defence, pace,
  scores and shot types; stats with each team; head to head; Elo; momentum; rest; target
  encodings; differences between the two players; and its old formula model's predictions.
- **Models:** four regressors on a rolling 180 days: each side's score, the margin and the total.
  They're blended as 0.3 × the side models + 0.7 × (total ± margin).
- **Results:** against the opening line over 19,276 games, total error fell 9.51 → 9.31 points
  and spread error 7.43 → 7.34.

This folder builds the same thing for AF. Step 1, here, uses the results alone. Step 2 would add
play-by-play efficiency stats (points per drive, touchdown and field-goal rates, yards per play,
explosive plays) from `SCOUTING_FULL`.

```bash
pip install -r xgb/requirements.txt
# the walk-forward glmer to stack on (the walkthrough's own: trim:hl60_x0.5:daily)
Rscript glmer/cadence.R --history=eAMFCalibrator/out/match_history.csv --test-from=2026-01-15 \
    --test-until=2026-10-19 --arms=trim:hl60_x0.5:daily --cores=4 --out=xgb/out/glmer
python -m xgb.features --history eAMFCalibrator/out/match_history.csv \
    --glmer xgb/out/glmer/cadence_matches.csv --out xgb/out/features.csv
python -m xgb.backtest --features xgb/out/features.csv --since 2026-07-01 \
    --table nb2=eAMFCalibrator/out/walkthrough/priors/nb2.csv \
    --table avg=eAMFCalibrator/out/walkthrough/priors/avg.csv \
    --table v21=eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv
```

The glmer fits take about 10 minutes on 4 cores, the features 30 seconds, and the backtest about 5
minutes.

## The features (`features.py`)

Every feature is computed as of the match's kick-off from the results in by then. A result counts
36 minutes after its match's kick-off, as eAMFModel's stream has it. The schedule (who plays when,
with which team, on which stream) is known in advance.

| group | per side |
|---|---|
| all time | matches, points scored and conceded, win rate |
| rolling 5/15/50 | points scored, conceded, margin and total |
| form | points scored and conceded as an EWMA (half-life 10 matches, glmer's) |
| spread | sd of points scored and of the margin over the last 15; max and min scored over the last 10 |
| streaks | win rate over the last 3 and 10 |
| team | matches with this team; points scored and conceded with it, shrunk toward the gamer's last 50 |
| rest and session | hours since the last kick-off, game of the session (gaps under 2 hours) and of the day, and the session's settled results so far |
| rating | v2.1's Kalman form rating (`eAMFCalibrator/form_prior.py`, drifting 4 points² a day), updated as results come in |

And per match:
- **Head to head:** meetings, PLAYER_1's mean margin (shrunk by 5 meetings), the mean total and
  PLAYER_1's wins in the last 5.
- **Each team's league form:** points scored and conceded over the last 60 days.
- **Context:** hour, weekday and stream.
- **Simple expectations:** each side's points as the mean of its scoring and the other's
  conceding, over 5, 15, 50 matches and all time.
- **The walk-forward glmer's expected points.**

## The backtest (`backtest.py`)

The models are XGBoost regressors on a rolling 180 days, retrained every 7 days:
- a side's points (both sides of every match, from the side's view);
- the margin;
- the total.

They're blended as the deck does: margin = 0.3 × (home − away points) + 0.7 × the margin model,
and the same for the total. The number of trees is found by early stopping on the window's last 14
days, then the whole window is refitted. The settings are fixed, not tuned on the test.

There are two feature sets:
- **`scores`:** the results alone. This is the deck's set without its old model.
- **`stack`:** `scores` plus glmer's expected points and the form rating, the equivalent of the
  deck's meta-predictions. XGBoost learns corrections to the models we have.

Every method's expected margin becomes a moneyline the same way: a normal with sd 9.35, ties left
out.

### Results (20 Oct 2026, `eAMFCalibrator/out/match_history.csv`)

**1 Jul – 18 Oct, 9,224 matches:**

| | log loss | Brier | margin RMSE | margin MAE | total RMSE | total MAE |
|---|---|---|---|---|---|---|
| glmer | 0.6688 | 0.2381 | 9.314 | 7.276 | 12.352 | 9.733 |
| XGBoost, results only | 0.6745 | 0.2409 | 9.363 | 7.334 | 12.388 | 9.768 |
| XGBoost, stacked | 0.6691 | 0.2383 | 9.307 | 7.271 | 12.329 | 9.712 |
| glmer and XGBoost (results only) averaged | 0.6696 | 0.2385 | 9.307 | 7.281 | 12.330 | 9.718 |

Against glmer, match by match:
- **XGBoost on the results only: worse everywhere.** Log loss +0.0057 (t +4.3, worse in all 8
  fortnights); margin and total worse.
- **Stacked: level on the moneyline,** +0.0003 (t +0.5). It's a little better on the total
  (squared error t −1.7) and level on the margin (t −0.8).
- **Averaged with glmer: level on the moneyline,** better on the total (t −2.1).

**1 Sep – 18 Oct, against the walkthrough tables (3,463 matches):**

| | log loss | margin RMSE | total RMSE |
|---|---|---|---|
| glmer | 0.6772 | 9.544 | 12.493 |
| XGBoost, stacked | 0.6780 | 9.552 | 12.439 |
| NB2 | 0.6787 | 9.570 | 12.601 |
| avg | 0.6770 | 9.540 | 12.499 |
| v2.1's default (`avg_form4_w05`) | 0.6823 | 9.623 | 12.499 |

The stacked model has the best totals. It's better than the v2.1 default on the moneyline
(−0.0042, t −2.0) and the total (t −2.6). glmer and avg are as good on the moneyline. The v2.1
default trails on log loss: its form rating moves margins toward who is playing well now, which
`form-prior` chose for the betting sim, not for log loss.

**What the margin model leans on (share of its gain):**
- **Results only:** head-to-head margin 23%, then the simple expected margins (all-time 14%, and
  about 4% each for 5, 15 and 50 matches).
- **Stacked:** glmer's margin 44%, the form rating 10%, head to head 4%, then nothing above 3.5%.

**So the deck's approach doesn't find more in the results than glmer already has.** On its own,
XGBoost on 87 score features falls well short of glmer. Stacked on glmer, it adds a little on
totals and nothing on the moneyline. That matches glmer's residual tests: gamer level plus recent
form carry nearly all the signal in the scores, and teams, pairs and sessions add little.

The deck's gains were measured against prod's opening line, not against a model like glmer. Any
further edge has to come from information the scores don't carry, which is step 2: drive-level
efficiency from the play-by-play.

## Step 2: the play-by-play (`drives.py`)

The scouting export (one row per play, a gigabyte or more) is reduced locally to one row per match:
each side's drive and snap counts, plus prod's last pre-match quotes.

```bash
py -m eAMFCalibrator scouting --since "2026-01-01 00:00:00" --until "2026-10-19 00:00:00" --no-probe --sample 0 --out eAMFCalibrator/out_scouting_xgb
py -m xgb.drives --export eAMFCalibrator/out_scouting_xgb/scouting_playover.csv --out xgb/data/drive_stats.csv
python -m xgb.features --history eAMFCalibrator/out/match_history.csv \
    --glmer xgb/out/glmer/cadence_matches.csv --drives xgb/data/drive_stats.csv --out xgb/out/features.csv
python -m xgb.backtest --features xgb/out/features.csv --since 2026-07-01 \
    --table avg=eAMFCalibrator/out/walkthrough/priors/avg.csv \
    --table v21=eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv
```

- **What `drives.py` reads.** Every scrimmage snap (its gain, success, 20+ and 40+ yard plays,
  losses, turnovers, 3rd and 4th downs, the clock it used) and every drive (touchdown, field goal
  made or missed, punt, turnover, turnover on downs or the half; red-zone trips and touchdowns
  from them; plays; where it started).
- **What it writes.** Counts only, in `drive_stats.csv`, plus a league summary in
  `drive_stats.txt` to sanity-check against the game.
- **What `features.py --drives` adds.** For each gamer, over his last 15 and 50 matches with
  play-by-play, his offence's rates and the offences he faced (`drv_off_`, `drv_def_`): points,
  touchdowns, field goals, punts, turnovers and downs per drive, red zone, plays per drive, start,
  yards and success a snap, explosive plays, losses, 3rd downs converted, drives a match and
  seconds a snap.
- **What `backtest.py` adds.** Two more variants, `drives` and `stack_drives`. Prod's quotes come
  through as `prod_` columns, never features. The report sets every method against prod's
  pre-match total and spread lines as the deck does (bias, MAE, within 5 and 10 points).
- **v21 tables.** Every variant is written as a prior table
  (`priors/xgb_<set>.csv`) that v21 reads with `--v21-prior-model`.

### Step 2 results (20 Oct 2026; play-by-play from 27 Apr, 97% of matches from June)

**1 Aug – 18 Oct, 6,313 matches:**

| | log loss | margin RMSE | total RMSE |
|---|---|---|---|
| glmer | 0.6727 | 9.447 | 12.475 |
| XGBoost, results only | 0.6789 | 9.513 | 12.477 |
| XGBoost, results + drives | 0.6792 | 9.515 | 12.483 |
| XGBoost, stacked | 0.6739 | 9.456 | 12.442 |
| XGBoost, stacked + drives | 0.6741 | 9.461 | 12.444 |

- **The drive and snap rates add nothing.** With them, the results-only model is no better
  (+0.0003 log loss) and neither is the stacked one (+0.0002). In the stacked model no drive rate
  carries more than 0.6% of the margin model's gain. Over these weeks the stacked models trail
  glmer on the moneyline (+0.0012, t +1.8) and beat it on totals (t −2.2).
- **On the deck's own measure, glmer already beats prod's pre-match lines** by what CV3 claimed.
  On 6,162 matches prod quoted before kick-off:

  | | total MAE | spread MAE | moneyline log loss |
  |---|---|---|---|
  | prod | 10.041 | 7.570 | 0.6847 (its own price) |
  | glmer | 9.810 (−2.3%) | 7.423 (−1.9%) | 0.6738 |
  | XGBoost, stacked | 9.799 | 7.436 | 0.6751 |

  The deck's CV3 took total MAE −2.2% and spread −1.2% against its opening line. Whatever prod
  has on us, it isn't in the pre-match expectation.

### Pace (`pace.py`)

```bash
python -m xgb.pace --features xgb/out/features.csv --drives xgb/data/drive_stats.csv --since 2026-08-01
```

Should a pre-match model put out a pace beside expected points? Three checks, 1 Aug – 18 Oct:

1. **Pace is a gamer's trait.** Split-half (odd v even ISO weeks, 74 gamers with 30+ matches in
   each), correlations are:
   - drives in his matches: 0.86;
   - snaps: 0.86;
   - seconds a snap: 0.89.

   That's about as steady as points per drive (0.84). The gamers spread by 0.7 drives a match
   about a mean of 10.1.
2. **It's mostly in the expected points already.** A match's foreseen pace correlates 0.62 with
   glmer's expected total. The two gamers' drives over their last 50 matches call the match's
   drives with r 0.26 (one match's drives vary a lot: sd 2.7). It's calibrated: real drives run
   9.4 → 11.3 across the foreseen fifths, 9.6 → 11.3.
3. **It widens the margin.** Given glmer's expected points, the margin's sd about glmer runs 8.6
   (slowest fifth) to 10.3 (fastest), t +5.4 for the squared miss net of the expected total. The
   total's sd follows the expected total (t +5.1), not pace (t +1.5). On the moneyline, scaling
   the margin's sd by the square root of the foreseen drives is worth −0.0001 (t −1.4). It matters
   more for spreads and lines away from the middle, where the margin's spread decides the price.

So pace belongs in the pre-match output as the margin's spread, not the mean. The sim makes its
margin's spread from each gamer's pace profile. Before adding a pace output, check whether v21's
kick-off distributions already widen 8.6 → 10.3 across these fifths. If they do, the profiles
carry it.

### Pace and efficiency as the pre-match model's outputs (`pace_eff.py`)

```bash
python -m xgb.pace_eff --features xgb/out/features.csv --drives xgb/data/drive_stats.csv \
    --since 2026-07-01 --score-from 2026-08-01
```

This pre-match model puts out pace and efficiency, then prices off them. It's walk-forward,
refitted weekly on 180 days, with rows weighted as glmer's (60-day half-life):
- **Pace:** the match's expected drives. It's a ridge regression of each match's drives on each
  gamer's and team's pace effects.
- **Efficiency:** each side's touchdowns and field goals a drive. These are Poisson regressions
  with the side's drives as exposure, on the side's attack and the opponent's defence (ridge-shrunk
  as random effects), both teams, home, and glmer's form and experience covariates. The gamer
  effects' sd (0.25) is chosen on July, the calibration span, out of 0.15, 0.25 and 0.35.
- **Points:** drives / 2 × (7 × touchdowns a drive + 3 × field goals a drive).
- **Spread:** a side's points sum its drives' 7, 3 or 0, so each side's variance is drives / 2 ×
  (49 p_td + 9 p_fg − ppd²). One scale factor, fitted on July, sets the margin's and the total's sd.

It's set against glmer and against a points-only model fitted the same way. Touchdowns come from
the final score less the field goals the feed saw (the drive table pushed on 20 Oct undercounts
them).

**Scored 1 Aug – 18 Oct (6,313 matches), each against glmer, match by match:**

| | log loss | margin sq err | total sq err | margin RPS | total RPS |
|---|---|---|---|---|---|
| points only (same fitting as pace × efficiency) | +0.0002 (t +0.5) | +0.01 | +0.20 | +0.0017 | +0.0060 |
| points only, the matches with play-by-play | +0.0017 (t +2.4) | +0.33 | +0.69 | +0.0124 | +0.0191 |
| pace × efficiency | +0.0073 (t +5.4) | +1.62 (t +5.6) | +2.83 (t +5.4) | +0.0548 | +0.0450 |
| glmer's points, sd by pace | −0.0000 | same | same | **−0.0055 (t −4.4)** | **−0.0078 (t −4.8)** |
| glmer's points, sd by pace × efficiency | −0.0001 (t −1.0) | same | same | **−0.0060 (t −4.5)** | **−0.0090 (t −5.2)** |

- **Pace × efficiency as the mean loses to points.**
  - Its expected margins are squeezed (sd 1.5 against glmer's 2.2) and track the results less
    well (r 0.19 against 0.23).
  - Less data explains only a sixth of it: points alone on the play-by-play matches cost +0.0017.
  - The rest is the split itself. A match's drive count and how its drives go pull against each
    other, and a side's touchdowns run tighter than Poisson (variance / mean 0.59). So a
    predicted pace times a predicted rate adds noise that the points already avoid.
- **Pace and the touchdown / field-goal mix are worth having for the spread.** With glmer's points
  as the means, a spread from pace and efficiency improves the whole distribution:
  - margin RPS −0.0060 (t −4.5);
  - total RPS −0.0090 (t −5.2);
  - the moneyline level (only the means set the favourite).

  Across the fifths of foreseen pace (9.0 to 11.3 drives), the real margin sd about the means runs
  8.75 → 10.28, and this model's sd 8.70 → 9.81.

So the pre-match model should put out three things:
- expected points for the means (glmer's, as now);
- expected drives;
- each side's touchdown and field-goal rate a drive, for the spread.

`pace_eff_prior.csv` holds all of them per match:
- `Pred_P1_Points` / `Pred_P2_Points` (glmer's);
- `exp_drives`;
- `P1/P2_td_per_drive` and `P1/P2_fg_per_drive`;
- `margin_sd` and `total_sd`;
- `p_home`.

v21 reads the points (`--v21-prior-model`). Its spread comes from its own simulation and player
profiles, so to use these it would take the match's expected drives as its pace in place of each
gamer's long-run profile.

### Does v21 already widen with pace? (`v21_pace.py`)

`v21-report` now writes each match's kick-off spread to `v21_matches.csv`:
- `sim_margin_sd` and `sim_total_sd`, from v21's own kick-off distributions;
- `prior_margin_sd` and `prior_total_sd`, from the prior's score model, where it has one.

`v21_pace.py` joins that with `pace_eff_prior.csv`. By fifth of expected drives, it sets the real
misses' spread against v21's own spread. It also regresses each match's squared miss about v21's
mean on v21's own variance and the expected drives. A drives term clear of zero is spread v21's
snap draws are missing.

```bash
python -m eAMFModel v21-report --since 2026-09-01 --until 2026-10-18 --nudge --nudge-share 0.75 \
    --state-paths 0 --out eAMFCalibrator/out_v21_pace
python -m xgb.v21_pace --v21 eAMFCalibrator/out_v21_pace/v21_matches.csv --pace xgb/out/pace_eff/pace_eff_prior.csv
```

**v21 (`v21_t10`, nudged 75% to the form table) on 1 Sep – 18 Oct (3,444 matches, 2,000 kick-off
games each; 21 Oct 2026):**

| fifth of expected drives | drives | margin: real about v21 | v21's own sd | total: real about v21 | v21's own sd |
|---|---|---|---|---|---|
| 1 | 9.23 | 8.77 | 9.81 | 10.85 | 10.89 |
| 2 | 9.90 | 9.38 | 9.90 | 12.44 | 11.33 |
| 3 | 10.30 | 9.36 | 9.92 | 12.55 | 11.67 |
| 4 | 10.71 | 9.83 | 9.95 | 12.49 | 12.01 |
| 5 | 11.35 | 10.52 | 10.09 | 13.52 | 12.40 |

- **v21's margin spread barely moves with pace.** It runs 9.81 → 10.09 while the real one runs
  8.77 → 10.52. So v21 is 12% too wide in the slowest fifth and 4% too narrow in the fastest.
  Each match's squared miss about v21's mean rises 10.3 a sd of expected drives (t +4.2) beyond
  v21's own variance.
- **Its total's spread already follows pace** (10.89 → 12.40). Its drives term isn't clear of
  zero (t +0.9).
- **The moneyline.** On the same matches, v21's own kick-off has log loss 0.6795. glmer's points
  with the pace × efficiency spread have 0.6769, and the form table that v21 nudges toward has
  0.6830.

So v21's snap draws carry pace into the total but not into the margin.

### The pace nudge in v21 (`v21-report --drives`, `--candidate v21-drives`)

v21 now takes the expected drives the way it takes the expected points. One tempo a match, on both
sides' time tilts, is solved with the points nudge so the kick-off games' drives meet the match's
goal. The goal is `exp_drives` over the table's mean, times the sim's own drives a match, so the
nudge moves pace from match to match and leaves v21's league level alone (`eAMFModel/README.md`, "The
pace nudge").

```bash
python -m eAMFModel v21-report --model v21_t10 --since 2026-09-01 --until 2026-10-18 --nudge --nudge-share 0.75 \
    --drives --state-paths 0 --out eAMFCalibrator/out_v21_drives
```

The pace now comes in the prior table (`exp_drives`, the walkthrough's `--scouting` step or
`eAMFCalibrator pace-prior`). `--drives` alone reads it from there; `--drives
xgb/out/pace_eff/pace_eff_prior.csv` still takes this pipeline's table.

If the nudge works, v21's own margin sd should rise across the fifths of expected drives as the real
one does (8.77 → 10.52), not sit at 9.81 → 10.09. The report's drives table shows, by fifth, the sim's
own drives before the nudge, its goal, its drives after and the real drives.

### Which model foresees drives best? (`pace_models.py`)

```bash
python -m xgb.pace_models --history eAMFCalibrator/out/match_history.csv --drives xgb/data/drive_stats.csv \
    --score-from 2026-08-01 --until 2026-10-19
```

Every model is refitted every midnight on the results in by then (180 days, rows halving every 60
days) and prices that day's matches. Contenders:

- **league**: the weighted mean.
- **gamers**: the two gamers' own recent drives a match (an EWMA).
- **ridge**: pace_eff's (above), its shrinkage set by hand.
- **ridge+stream**: the same with the stream.
- **mixed**: the model glmer would fit, gamer and team pace as random effects, every variance found by
  REML at each refit. lme4 can't put a match's two gamers on one effect, so it's fitted in Python.
  On a model lmer can fit (one gamer and one team a match) it matches lmer to four figures: shrinkage
  15.01 v 15.01 and 299.3 v 299.2, the fixed effects the same. `tests/test_pace_models.py` holds
  that check.
- **xgb**: XGBoost on each gamer's and team's drives over their last 15 and 50 matches, an EWMA,
  matches played and the stream.

Results, 6,285 matches, 1 Aug – 18 Oct, real drives 10.51 a match (sd 2.73):

| model | RMSE | r | slope | sd | sq. error v ridge |
|---|---|---|---|---|---|
| league | 2.762 | 0.01 | – | 0.12 | +0.629 (t +12.5) |
| gamers | 2.643 | 0.260 | 1.16 | 0.61 | −0.011 (t −0.4) |
| ridge | 2.645 | **0.279** | **1.02** | 0.75 | |
| ridge+stream | 2.645 | 0.279 | 1.01 | 0.75 | −0.004 (t −0.9) |
| mixed | 2.648 | 0.277 | 0.83 | 0.92 | +0.014 (t +1.1) |
| xgb | 2.650 | 0.248 | 0.98 | 0.69 | +0.023 (t +0.6) |

Slope is real drives on foreseen: 1 is calibrated, below 1 means the forecast spreads too wide.

- **No model beats the ridge.** It ranks matches best in every month (r 0.31, 0.28, 0.21 for Aug,
  Sep, Oct, against xgb's 0.29, 0.23, 0.15), and its spread is calibrated.
- **What v21 uses.** v21 takes `exp_drives` over its table's mean, so the level doesn't matter. The
  ridge's −0.3 bias (the league's drives rising through the autumn) drops out. The ranking (r) and
  the spread (slope) are what count.
- **REML shrinks the gamers far less than the hand-set ridge.** It asks for λ 15 against 90 (team
  650 against 360). The mixed model's forecasts then spread too wide (slope 0.83). REML measures
  how far gamers' pace differs within the 180 days. Part of that difference doesn't last into the
  next week, so the heavier shrinkage foresees better.
- **XGBoost on rolling form ranks worse** than the ridge's effects.
- **A Poisson glmer would be the wrong family.** Drives are under-dispersed: variance 7.5 on a mean
  of 10.5.

So the walkthrough's pace (`exp_drives`, `eAMFCalibrator/pace_prior.py`) is the ridge, refitted daily.
It counts drives off the scouting export the sim's way, not off `drive_stats.csv`.

### How much of a match's drives is its turnovers? (`turnovers.py`)

```bash
python -m xgb.turnovers --scouting eAMFCalibrator/out_scouting_xgb/scouting_playover.csv \
    --history eAMFCalibrator/out/match_history.csv --since 2026-09-01
```

A match's drives vary a lot (sd 2.4 on the sim's count), and a first look at `drive_stats.csv` put most
of it on turnovers. On that table's counts, a match's drives correlate 0.75 with its turnovers, and the
pace model's miss 0.73. That table's turnover counts are off where the feed filed a touchdown late, so
this script counts again, off the export, the way v21 reads its snaps (`sim14.snap_records`, the
records its turnover draws are fitted on).

It reports four things:

1. **Drives against turnovers.** The correlation, the drives a turnover adds, and how much of the
   drive count's variance is the snaps a game holds against the snaps a drive lasts.
2. **A gamer's turnovers as a trait.** Odd against even weeks: the turnovers a drive each gamer gives
   up and forces.
3. **Before the match.** Expected drives and expected turnovers, each refitted daily with the pace
   prior's ridge. How well each foresees its own, how much of the pace's miss is the match's
   turnovers, and whether expected turnovers add anything to the pace.
4. **The margin.** How much of the points margin the turnover margin explains.

It also counts `suspect` turnovers: ones whose next row carries a touchdown message for the side that
lost the ball. Few of them means the sim's turnover count is clean.

**Results** (21 Oct, the export of 20 Oct, 3,444 matches, 1 Sep – 18 Oct). There are no suspect
turnovers, so the sim's count is clean. `drive_stats.csv` had overstated turnovers (2.66 a match
against 1.96), and with them how much they explain.

A match has 9.87 drives (sd 2.40). Most end in a touchdown (3.99 a match) or a turnover (1.96); there
are few punts (0.73) or turnovers on downs (0.44).

1. **Turnovers explain under a third of the drive count's variance.** r 0.54, R² 0.30, a turnover
   +0.92 drives. With punts and turnovers on downs added, R² is 0.41. Of the drive count's
   variance (in logs), the snaps a drive lasts are 64%, the snaps a game holds 27%, the two together
   +9%.
2. **A gamer's turnover rate is a steady trait, but a small one.** Odd against even weeks, the
   turnovers a drive given up correlate 0.73 and those forced 0.71. Gamers' rates spread only 0.033
   about 0.201 a drive: one sd is 0.16 turnovers a match.
3. **Before the match they barely help.** Expected turnovers (the pace ridge, refitted daily)
   correlate 0.21 with the real ones. Half the pace's miss is the match's own turnovers (r 0.52), but
   adding expected turnovers to the pace moves nothing (R² 0.0896 → 0.0902, t +1.5). Turnovers are
   in-match luck, and no turnover prior is needed for pace.
4. **They decide the margin.** The turnover margin explains 55% of the points margin (r 0.74), +4.7
   points a turnover. Some of that runs backwards, with a trailing side taking risks, so not all of
   it is the turnovers' doing. v21 already gives each gamer and team, on offense and on defense, its
   own turnover rate (the `turnover` component of the tilts, `PHI_DIST`).

### Does a turnover prior add to the margin? (`turnover_margin.py`)

```bash
python -m xgb.turnover_margin --counts xgb/out/turnovers/turnovers_matches.csv \
    --prior eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv
```

A side's expected turnovers come from a home term, the offense gamer's turnovers given up, the defense
gamer's forced and both teams'. It's a ridge refitted every midnight, its shrinkage chosen on August
walk-forward (gamer 5, team 640). Their difference, the expected turnover margin, is then set against
the form table's expected margin and moneyline, 1 Sep – 18 Oct (3,444 matches).

- **It foresees the turnover margin only weakly** (r 0.15; spread sd 0.26 against the real 1.55),
  and it shares a quarter of its variance with the prior's margin (r 0.49).
- **Against the form table alone it looks useful.** The margin's coefficient is +2.1 (t +2.9); out
  of sample (fitted to 23 Sep, scored after), squared error −0.27 a match (t −1.6) and moneyline log
  loss −0.0010 (t −1.6).
- **That is strength the form table lost, not turnovers.** With the walkthrough's `avg` margin (the
  table form-prior moves) in the regression, the turnover margin drops to t +0.8; with `glmer` and
  `nb2` too, t +0.4. The gamers' turnover traits are already in the points models, so no turnover
  prior is needed.

On the way: **the form table's own pre-match price is over-confident against results.** Real margins
move 0.70 points a point of its margin, its moneyline's logit slope is 0.72 (`avg`'s 1.11), and its
margin is spread wider (sd 2.64 against 1.94). Against `avg` on the same 3,461 matches, its log loss
is +0.0051 a match (t +2.6) and its margin's squared error +1.59 (t +3.6), in both months.

That fits form-prior's own finding ("its Brier score was no better than prod's"): it was chosen for
the betting sim's returns re-pricing prod's bets, not for accuracy against results. v21 nudges only
75% of the way toward it and makes its own spread, so the price the sim makes needn't be as
over-confident. The test that settles it is v21 nudged toward `avg` against v21 nudged toward the
form table, on the same bets (`--prior-for`).

## Files

- `features.py`: the as-of feature table (`python -m xgb.features`).
- `backtest.py`: the walk-forward models and the comparison (`python -m xgb.backtest`). It writes
  `backtest_report.txt`, `backtest_matches.csv` (every match, every method's expected margin and
  total) and `priors/xgb_<set>.csv`.
- `drives.py`: the scouting export reduced to drive and snap stats per match
  (`python -m xgb.drives`). Scores come off the scoreboard: the feed's touchdown message can land
  after its play's PLAY_OVER and file the play as a scrimmage. `data/drive_stats.csv` (pushed 20
  Oct) was made before that fix. Its touchdown, turnover and half-over counts are off; its drives,
  snaps, yards, pace and points per drive (off the final score) are not.
- `pace.py`: the three pace checks.
- `pace_eff.py`: the pace-and-efficiency pre-match model and its comparison
  (`python -m xgb.pace_eff`), writing `pace_eff_prior.csv`.
- `v21_pace.py`: v21's kick-off spread by pace against the real one (`python -m xgb.v21_pace`).
- `pace_models.py`: the drive forecasters compared walk-forward (`python -m xgb.pace_models`), writing
  `pace_models.txt` and `pace_models.csv` (every match, every model).
- `turnovers.py`: turnovers and drives counted the sim's way, and how they relate
  (`python -m xgb.turnovers`), writing `turnovers.txt` and `turnovers_matches.csv`.
- `turnover_margin.py`: a walk-forward turnover prior against a prior table's margin and moneyline
  (`python -m xgb.turnover_margin`).
- `tests/`: `python -m unittest xgb.tests.test_drives xgb.tests.test_pace_models xgb.tests.test_turnovers`.
