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

So v21's snap draws carry pace into the total but not into the margin. The next step is a pace
nudge in v21: solve each side's time tilt so the kick-off games' drives meet the match's expected
drives, the way the points nudge meets the expected points.

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
- `tests/`: `python -m unittest xgb.tests.test_drives`.
