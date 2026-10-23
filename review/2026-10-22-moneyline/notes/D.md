# Agent D: XGBoost for the pre-match moneyline

Scripts are in `scripts/D/`, outputs in `out/D/`, and predictions in `preds/D_*.csv` (2026-06-01..10-18, 12,119
matches each). Every number below is walk-forward. Settings were tuned on TUNE only and frozen before TEST was
looked at. Comparisons use common matches with the day-resampled 95% interval (`harness.report`).

## Summary

- **XGBoost beats prod and the simple link on `rating_margin`, but it does not beat a 4-feature logistic.**
  - On TEST, the model chosen on TUNE (`xgb_margin_nodrv`, a margin regressor plus a link) scores:
    - log loss 0.6792 against prod's 0.6848: **-0.0056 (-0.0104, -0.0007)**;
    - against the link on `rating_margin`: **-0.0038 (-0.0066, -0.0009)**;
    - calibration slope 0.92.
  - The best plain classifier (`xgb_paired`) scores 0.6783 on TEST: -0.0064 (-0.0111, -0.0021) against prod
    and -0.0046 (-0.0072, -0.0019) against the link.
  - A walk-forward logistic on the XGBoost model's top three features plus a ridge strength (`lr4`) matches
    XGBoost on TUNE and beats every XGBoost model on TEST (+0.0022 to +0.0030 for XGBoost, intervals at or
    above 0). It scores 0.6762 on TEST: -0.0086 (-0.0130, -0.0044) against prod.
- **Corrections on a strong base are worth nothing.** XGBoost corrections help a weak base, but on top of
  `lr4` they add -0.0005 on TUNE and +0.0003 on TEST.
- **The drive stats don't matter.** Dropping all 152 drive features changes the paired model by -0.0008 on
  TUNE and +0.0004 on TEST, both inside the noise.
- **XGBoost is not over-confident here.** Calibration slopes run 0.88-1.00 on TEST and 0.86-1.03 on SEP.
  Walk-forward Platt and temperature scaling never help (+0.0000 to +0.0003).
- **The paired design helps a little.** It beats the plain design by -0.0018 on TUNE and -0.0013 on TEST, both
  intervals touching 0.
- **Recommendation: don't productionise a standalone XGBoost model** (details in the last section). Other
  agents' strength models (A_best, C_stats_strength, B_mov_kal) sit about 0.004 better on TEST.

## Methods

**Data.**
- Loaded with `harness.load()`.
- Features: every features.csv column except the identity columns (code, start, p1, p2, t1, t2, s1, s2) and
  `prod_*`. That leaves 244 columns: 116 per side plus 12 per match, 152 of them drive stats.
- Never features: prod, the prior tables and ELO.
- Target: y = home win, with ties left out.

**Walk-forward.**
- Everything runs through `harness.walk_forward`: fit on the matches settled (36 minutes after kick-off) by the
  day's midnight UTC.
- XGBoost is **refitted every 3 days on a rolling 180-day window**.
- At each refit:
  - the number of trees comes from early stopping (patience 100, at most 3,000) on the window's last 14 days;
  - the whole window is then refitted with that many trees.
- XGBoost runs with `hist` on **1 thread**. Two OpenMP threads ran 15 times slower than one on this shared,
  oversubscribed box (151 s against 9 s for the same fit), because of spin-waits.
- The full runs start on 2026-04-01. That way the links and recalibrations, which are fitted on earlier
  out-of-sample predictions, have 60 days of history by 06-01.

**Encodings.**
- hour (0-23), weekday (0-6) and stream (1-3) stay ordinal. Trees can cut them anywhere, and stream has only 3
  levels.
- Gamer identity was tested once, as a pandas categorical (`enable_categorical`): p1 and p2 in the plain
  design, gamer and opponent in the paired one.

**Designs.**
- *Plain*: one row a match, from the home side's view.
- *Paired*: each match appears twice.
  - The home view has own = p1, `home` = 1 and label y. The away view has own = p2, `home` = 0 and label 1 - y.
  - Match features are oriented to the own side: `rating_margin`, `h2h_margin` and `exp_margin_*` change sign,
    and the away twin of `h2h_wins_5` is min(h2h_n, 5) - h2h_wins_5.
  - The output is p(home) = (p_home_view + 1 - p_away_view) / 2. The model is antisymmetric by construction and
    sees twice as many rows.
- *Diffs*: own_x - opp_x (or p1_x - p2_x) added for every side feature.

**Simple models.** These are the baselines and the offset bases (`s01_baselines.py`, `s08_offset_lr4.py`).
- `link_rm` is the required simple reference: p = sigmoid(a + b · `rating_margin`), refitted daily on the last
  60 days. Windows of 60, 120, 240 days and all data tie on TUNE (0.66985 / 0.66988 / 0.66994 / 0.66994).
- `ridge_p` is a time-decayed ridge strength:
  - margin = h + s_p1 - s_p2, with rows weighted 0.5^(age / 45 days) and L2 = 2 on the strengths;
  - refitted daily, then passed through a logistic link fitted on its previous 60 days of out-of-sample margins;
  - the grid on TUNE was half-life 10/20/45/90 × lambda 2/5/20, and 45 / 2 scored best (0.66706).
- `lr4` is a walk-forward logistic, refitted daily on 180 days, on four inputs: `rating_margin`, `h2h_margin`,
  `exp_margin_all` (the XGBoost model's top three by gain) and the ridge margin.
  - The top three were checked on TUNE-era refits alone: 50%, 17% and 7% of gain, the same order.
  - `lr3` is the same without the ridge margin.

**Margin regressor (`xgb_margin*`).**
- An XGBoost regressor on the margin, in the paired design: target = own margin, and margin = (m_home - m_away) / 2.
- p = sigmoid(a + b · margin), fitted each day on the previous **30 days** of out-of-sample predicted margins
  only. The window was chosen on TUNE: 30 days 0.66229, 60 days 0.66233, 120 days 0.66272.

**Offset / stacked XGBoost (`xgb_offset*`).**
- On every row, base_margin = logit(base p). The away view gets the negative.
- So the trees learn only corrections. Early stopping may keep zero trees, and the base then stands.
- The parameters are stronger-regularised: depth 3, eta 0.02, min_child_weight 100, lambda 20, colsample 0.5.

**Recalibration.**
- Platt (a + b · logit p) and temperature (b · logit p), fitted each day on the previous N days of
  out-of-sample predictions only.
- N was chosen on TUNE from 30, 60 and 120 (60 or 120, depending on the model).

## The search (TUNE only; `s02_tune.py`, `s04_tune_other.py`; `out/D/tune_results.csv`, `tune_summary.txt`)

The stages ran in this order. Log loss is on all 5,802 TUNE matches. "vs anchor" is against B_paired_w180,
with the day-resampled interval.

**A. Design** (default parameters: depth 3, eta 0.03, min_child_weight 50, subsample 0.8, colsample 0.5,
lambda 10; window 120; refit every 3 days):

| design | TUNE ll |
|---|---|
| plain | 0.66647 |
| plain + diffs | 0.66526 |
| plain + gamer identity | 0.66578 |
| **paired** | **0.66393** |
| paired + diffs | 0.66465 |
| paired + identity | 0.66300 |

Paired beats plain by -0.0025 (-0.0042, -0.0009). The diffs don't help the paired design.

**B. Window** (paired, refit every 7 days):

| window | TUNE ll |
|---|---|
| 60 days | 0.66630 |
| 120 days | 0.66454 |
| **180 days** | **0.66340** |
| all | 0.66340 |

180 days and all data tie, so I chose the bounded window. Refitting every 3 days rather than 7 was 0.0006
better at 120 days.

**C. Parameters.** A random search of 16 configurations on paired, window 180, refit every 7 days, over:
- depth 2/3/4/6;
- eta 0.02/0.05/0.1;
- min_child_weight 10/50/200;
- subsample 0.6/0.8/1;
- colsample 0.3/0.6/0.9;
- lambda 1/10/50.

The range was 0.66330 to 0.66547: flat. Half the configurations sat within 0.0006 of the anchor (0.66340).
- **Chosen (best TUNE): C03 = depth 2, eta 0.1, min_child_weight 200, subsample 0.8, colsample 0.9,
  lambda 50**, at 0.66330.
- Deep, unregularised settings were worst: depth 6 with min_child_weight 10 and lambda 1 scored 0.66453, and
  depth 6 with eta 0.02 scored 0.66525.

**D. Design re-check with C03.**

| variant | TUNE ll |
|---|---|
| paired | 0.66330 |
| paired + identity | 0.66352 (identity no longer helps) |
| plain | 0.66518 (used for `xgb_cls`) |
| plain + diffs | 0.66543 |

**M. Margin regressor** (C03 parameters, paired, run from 04-01 so the link has history):

| objective | min_child_weight 50 | 200 | 800 |
|---|---|---|---|
| squared error | 0.66322 | **0.66196** | 0.66267 |
| pseudo-Huber (slope 10) | 0.66260 | 0.66318 | 0.66385 |

Chosen: squared error with min_child_weight 200.

**O. Offset.** Bases: `link_rm` (TUNE 0.66985) and `ridge_p` (0.66706). Each base was tried with three
parameter sets.

| base | r1 (depth 2, eta 0.02, min_child_weight 200, lambda 50) | r2 (depth 3, eta 0.02, min_child_weight 100, lambda 20) | C03 |
|---|---|---|---|
| link_rm | 0.66407 | 0.66405 | 0.66399 |
| ridge_p | 0.66395 | **0.66307** | 0.66365 |

Chosen: r2 on `ridge_p` (`xgb_offset`). r2 also ran on `link_rm` for the corrections question.

**Frozen final runs** (`configs_frozen.json`; `s03_runs.py`, refit every 3 days, window 180, from 04-01):

| run | TUNE ll |
|---|---|
| xgb_cls (plain, C03) | 0.66482 |
| xgb_paired (C03) | 0.66305 |
| xgb_paired_ident | 0.66274 |
| xgb_paired_nodrv | 0.66238 |
| xgb_margin (link 30 days) | 0.66229 |
| **xgb_margin_nodrv** | **0.66174** (best on TUNE, so this is "the" TUNE-selected D model) |
| xgb_offset | 0.66265 |
| xgb_offset_nodrv | 0.66279 |
| xgb_offset_rm | 0.66327 |
| ablations, from 06-01: nodrv | 0.66222 |
| ablations: ratings only | 0.66935 |
| ablations: form only | 0.67009 |

The `_nodrv` variants were added after the ablation: dropping the drive stats was better on TUNE, so TUNE
chose them. The `lr4` check and the offset on `lr4` (`s07`, `s08`) came after the importance analysis. They
are a check of what XGBoost adds, not a selection made on TEST.

## Results (`out/D/s05_report.txt`, `s09_final_compare.txt`)

Log loss on common matches (every column and prod priced). The intervals are the day-resampled 95%.

| model | TUNE (5,462) | TEST (6,170) | SEP (3,426) | 09-22..10-18 (1,840) |
|---|---|---|---|---|
| prod | 0.6764 | 0.6848 | 0.6894 | 0.6880 |
| link_rm (simple link on rating_margin) | 0.6704 | 0.6829 | 0.6856 | 0.6866 |
| ridge_p | 0.6689 | 0.6794 | 0.6836 | 0.6852 |
| lr4 (4-feature logistic) | 0.6639 | **0.6762** | **0.6801** | **0.6815** |
| xgb_cls (plain) | 0.6657 | 0.6794 | 0.6828 | 0.6832 |
| xgb_paired | 0.6642 | 0.6783 | 0.6820 | 0.6827 |
| xgb_paired_nodrv | 0.6637 | 0.6785 | 0.6819 | 0.6818 |
| xgb_margin (+ link) | 0.6632 | 0.6797 | 0.6832 | 0.6844 |
| **xgb_margin_nodrv (TUNE pick)** | **0.6626** | 0.6792 | 0.6821 | 0.6827 |
| xgb_offset (base ridge_p) | 0.6647 | 0.6779 | 0.6824 | 0.6834 |
| xgb_offset_nodrv | 0.6649 | 0.6774 | 0.6824 | 0.6835 |
| xgb_offset_rm (base link_rm) | 0.6646 | 0.6798 | 0.6831 | 0.6838 |
| xgb_offset_lr4 (base lr4) | 0.6634 | 0.6765 | 0.6806 | 0.6818 |

Differences with intervals:

| model | window | vs prod | vs link_rm | vs lr4 |
|---|---|---|---|---|
| xgb_margin_nodrv | TUNE | -0.0138 (-0.0198, -0.0081) | -0.0078 (-0.0114, -0.0043) | -0.0013 (-0.0040, +0.0013) |
| | TEST | -0.0056 (-0.0104, -0.0007) | -0.0038 (-0.0066, -0.0009) | +0.0030 (+0.0010, +0.0051) |
| | SEP | -0.0073 (-0.0138, -0.0010) | -0.0036 (-0.0069, -0.0002) | +0.0020 (-0.0005, +0.0045) |
| | 09-22.. | -0.0052 (-0.0117, +0.0011) | -0.0038 (-0.0084, +0.0010) | |
| xgb_paired | TEST | -0.0064 (-0.0111, -0.0021) | -0.0046 (-0.0072, -0.0019) | +0.0022 (-0.0000, +0.0042) |
| | SEP | -0.0074 (-0.0142, -0.0010) | -0.0036 (-0.0067, -0.0004) | +0.0019 (-0.0007, +0.0044) |
| | 09-22.. | -0.0053 (-0.0127, +0.0018) | -0.0039 (-0.0077, +0.0000) | |
| xgb_offset_nodrv | TEST | -0.0074 (-0.0116, -0.0035) | -0.0056 (-0.0095, -0.0014) | +0.0012 (-0.0005, +0.0031) |
| lr4 | TEST | -0.0086 (-0.0130, -0.0044) | -0.0068 (-0.0097, -0.0039) | |
| | SEP | -0.0093 (-0.0157, -0.0036) | -0.0055 (-0.0089, -0.0018) | |
| link_rm | TEST | -0.0018 (-0.0071, +0.0032) | | |
| | SEP | -0.0037 (-0.0111, +0.0033) | | |

**Paired against plain** (same parameters), xgb_paired less xgb_cls:

| window | difference |
|---|---|
| TUNE | -0.0018 (-0.0036, +0.0001) |
| TEST | -0.0013 (-0.0030, +0.0003) |
| SEP | -0.0008 (-0.0024, +0.0010) |
| 09-22.. | -0.0005 |

- The direction is consistent and the size is small. With default parameters on TUNE the gap was -0.0025
  (-0.0042, -0.0009).
- Gamer identity on top of paired: -0.0003 on TUNE and 0.0000 on TEST. It is not worth it.

**Margin regressor and link against the classifier.**
- On TUNE it is the best (-0.0013 to -0.0005 against the paired classifier).
- On TEST it is a little worse: xgb_margin_nodrv 0.6792 against xgb_paired_nodrv 0.6785.
- It is coarse: the predicted margin's sd is 2.6 points on TUNE and 2.0 on TEST, against a realised sd of 9.7.

**Offset: do corrections beat the base on held-out log loss?**

| offset model | TUNE | TEST | SEP | 09-22..10-18 |
|---|---|---|---|---|
| on link_rm (weak base) | -0.0066 (-0.0095, -0.0038) | -0.0033 (-0.0057, -0.0006) | -0.0026 (-0.0055, +0.0003) | |
| on ridge_p | -0.0044 (-0.0066, -0.0022) | -0.0018 (-0.0035, +0.0001) | -0.0011 (-0.0028, +0.0006) | -0.0018 (-0.0033, -0.0004) |
| on ridge_p, no drives | -0.0043 | -0.0022 (-0.0040, -0.0005) | -0.0011 | |
| on lr4 (strong base) | -0.0005 (-0.0021, +0.0012) | +0.0003 (-0.0007, +0.0013) | +0.0005 (-0.0003, +0.0014) | +0.0003 |

- Corrections beat a weak base clearly, and the gain shrinks as the base gets stronger.
- On `lr4` the corrections are worth nothing. Early stopping kept zero trees at 5 of 72 refits.
- The corrections on `ridge_p` are mostly the information `lr4` adds explicitly: rating_margin, exp_margin_5,
  h2h. Their largest SHAP share is spread over drive stats (43%) but with little effect.

## Calibration

Slopes from `harness.calib`: 1 is calibrated, below 1 means the prices are too spread out.

| model | TUNE | TEST | SEP | 09-22.. |
|---|---|---|---|---|
| prod | 1.13 | 0.93 | 0.68 | 0.79 |
| link_rm | 1.06 | 0.79 | 0.76 | 0.79 |
| xgb_paired | 1.04 | 0.98 | 1.00 | 1.19 |
| xgb_margin_nodrv | 1.05 | 0.92 | 1.03 | 1.24 |
| xgb_offset | 0.95 | 0.88 | 0.87 | 0.94 |
| lr4 | 1.05 | 0.99 | 0.94 | 0.94 |

- **XGBoost here is not over-confident.** Early stopping and the regularised C03 settings keep it slightly timid
  on TUNE (1.04-1.05) and about calibrated on TEST and SEP. The offset models are the most spread out (0.86-0.89
  on TEST and SEP).
- Monthly slopes swing for every model, prod included. In Jun / Jul / Aug / Sep / Oct, xgb_paired runs 1.00,
  1.12, 0.96, 0.92, 1.22, and prod runs 1.01, 1.22, 1.25, 0.51, 1.04. That is regime noise, not a fixed bias.
- **Walk-forward Platt and temperature recalibration** (fitted on the previous 60 or 120 days of out-of-sample
  predictions, window chosen on TUNE) is +0.0000 to +0.0003 worse for every model and window. For xgb_paired,
  Platt costs:

  | window | Platt less raw |
  |---|---|
  | TUNE | +0.0002 |
  | TEST | +0.0003 (-0.0001, +0.0006) |
  | SEP | +0.0002 |

  The fitted slope trails the month-to-month swings, so it adds noise. Don't recalibrate. `D_xgb_paired_platt`
  is written only for completeness.

## Importance (`out/D/s06_importance.txt`; gain summed over every refit, mean |SHAP| over TEST)

SHAP comes from `pred_contribs`. In the paired design it is (home view - away view) / 2, the contribution to the
antisymmetric logit.

**xgb_paired by group** (share of total gain / share of mean |SHAP| on TEST / on SEP):

| group | features | gain | SHAP TEST | SHAP SEP |
|---|---|---|---|---|
| rating (rating_margin, p1/p2_rating, _var) | 5 | 52% | 28% | 39% |
| h2h (h2h_margin, h2h_wins_5, h2h_n, h2h_total) | 4 | 19% | 17% | 24% |
| form (rolling scores, exp_margin_*) | 56 | 17% | 19% | 13% |
| drives | 152 | 7% | 28% | 17% |
| team | 10 | 4% | 6% | 4% |
| rest / session | 14 | 1% | 2% | 1% |
| context (hour, weekday, stream, home) | 4 | 1% | 2% | 2% |

- **Top features by mean |SHAP| on TEST:**
  1. `rating_margin`: 26% (gain 50%)
  2. `h2h_margin`: 14% (gain 18%)
  3. `exp_margin_all`: 5% (gain 6%)
  4. `h2h_wins_5`: 2%
  5. `home`: 1.7%

  No other feature reaches 1.1%.
- The plain and margin models have the same order. In the margin model `rating_margin` is 49% of gain,
  `h2h_margin` 17% and `exp_margin_all` 10%, and `home` is worth about 0.13-0.17 points of margin on average.
- **Drive stats** (the h2h-site style features) take 28% of the SHAP mass on TEST but only 7% of the gain. That
  is 152 features each contributing at most 0.7%: the top ones are `drv_def_start_15`, `drv_def_loss_50` and
  `drv_off_third_50`. Diffuse, low-gain contributions like this are what fitted noise looks like, and the
  ablation agrees.

**Ablations** (paired, C03, same windows), each against xgb_paired:

| ablation | TUNE | TEST | SEP | 09-22.. |
|---|---|---|---|---|
| no drive stats | -0.0008 (-0.0021, +0.0004) | +0.0004 (-0.0011, +0.0019) | -0.0001 | -0.0010 |
| ratings only (5 features) | +0.0063 | +0.0041 | +0.0026 | +0.0031 |
| form only (56 rolling score features) | +0.0070 | +0.0037 | +0.0044 | +0.0032 |

- **The drive stats don't matter.** Without them the paired model is level on held-out data.
- Ratings-only XGBoost is just `link_rm` (-0.0005 TUNE, -0.0007 TEST against it).
- Form-only is about `link_rm` too.
- XGBoost's whole gain over the simple link comes from **combining** the rating with `h2h_margin` and
  `exp_margin_all`. A 3-4 input logistic does that as well (TUNE) or better (TEST).

## How D sits among the other agents (informational, `out/D/s07_extras.txt`)

TEST log loss on the matches each model and prod price (prod 0.6848):

| agent | model | TEST ll |
|---|---|---|
| A | A_best | 0.6731 |
| C | C_stats_strength_newgamer | 0.6737 |
| B | B_mov_kal | 0.6739 |
| D | lr4 | about 0.6749-0.6762 (depending on the match set) |
| D | best XGBoost: xgb_offset_nodrv | 0.6774 |
| D | xgb_paired | 0.6783 |
| D | TUNE pick: xgb_margin_nodrv | 0.6792 |

On SEP the gap between XGBoost and A_best is about 0.005.

## Conclusions and what I would productionise

1. **Not a standalone XGBoost moneyline.**
   - It beats prod on TEST by about 0.005-0.007 and the `rating_margin` link by about 0.004-0.005, all with
     intervals clear of 0.
   - But a walk-forward logistic on four numbers we already compute does better, by +0.002 to +0.003 on TEST.
   - The strength models from agents A, B and C do better still.
   - On top of that, XGBoost costs 244 features, refits with early stopping, and a design-dependent output
     (paired averaging).
2. **The useful finding is `h2h_margin` and `exp_margin_all`.** These two match-level features from
   xgb/features.py carry real information beyond the Kalman rating. The rating alone sits at the `link_rm`
   level. On TEST, rating + `h2h_margin` + `exp_margin_all` (`lr3`) gains about -0.0047 over `link_rm`, and
   the ridge strength margin (`lr4`) another -0.0024.
   - Whatever strength model is productionised (A's link, the stacker) should be offered `h2h_margin` and
     `exp_margin_all` as covariates.
   - `D_lr4` is written for the stacker.
3. **Drop the drive stats from any pre-match moneyline.** They add nothing held-out, in XGBoost or as
   corrections.
4. **If XGBoost is used at all, use it as an offset on the best strength model, with a hold-out gate.**
   - Paired design, no drive stats, depth 2-3, eta 0.02, min_child_weight 100+, early stopping that may keep zero
     trees.
   - Keep it only if its rolling held-out log loss beats the base.
   - On today's evidence, with a strong base it would mostly sit at zero trees or add noise (+0.0003).
5. **No Platt or temperature layer** on top of any of these.

## Files

**Scripts:**

| script | what it does |
|---|---|
| `common.py` | designs, the early-stopped fit, the walk-forward driver, links and ridge |
| `s01_baselines.py` | link_rm and ridge_p |
| `s02_tune.py` | search stages A, B, C and D |
| `s04_tune_other.py` | search stages M and O |
| `s03_runs.py` | the frozen full-span runs |
| `s05_report.py` | the report and the preds files |
| `s06_importance.py` | gain and SHAP importance |
| `s07_extras.py` | lr3 / lr4 and the cross-agent table |
| `s08_offset_lr4.py` | the offset on lr4 |
| `s09_final_compare.py` | the final head-to-head; writes D_lr4 and D_xgb_offset_lr4 |

The `q*.sh` files are the queues the runs went through.

**Outputs:**
- `out/D/series_<name>.csv`: walk-forward series from 04-01, keyed by code; `_m` holds the predicted margins.
- `out/D/run_<name>.json`: trees per refit.
- `out/D/gain_*.csv` and `out/D/shap_*.csv`: importance.
- `out/D/all_preds_wide.csv`: all predictions side by side.

**Predictions:**

| preds file | model |
|---|---|
| `D_xgb_cls` | plain classifier |
| `D_xgb_paired` | paired classifier |
| `D_xgb_paired_nodrv` | paired classifier, no drive stats |
| `D_xgb_paired_ident` | paired classifier with gamer identity |
| `D_xgb_paired_platt` | paired classifier, Platt-recalibrated |
| `D_xgb_margin_link` | margin regressor + link |
| `D_xgb_margin_link_nodrv` | margin regressor + link, no drive stats (TUNE pick) |
| `D_xgb_offset` | offset on ridge_p |
| `D_xgb_offset_nodrv` | offset on ridge_p, no drive stats |
| `D_xgb_offset_rm` | offset on link_rm |
| `D_xgb_offset_lr4` | offset on lr4 |
| `D_link_rm` | simple link on rating_margin |
| `D_ridge` | ridge strength + link |
| `D_lr4` | 4-feature logistic |
