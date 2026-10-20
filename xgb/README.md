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

## Files

- `features.py`: the as-of feature table (`python -m xgb.features`).
- `backtest.py`: the walk-forward models and the comparison (`python -m xgb.backtest`). It writes
  `backtest_report.txt` and `backtest_matches.csv` (every match, every method's expected margin
  and total).
