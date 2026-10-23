# Agent F: the total (over/under) -- the distribution of the total, drives x points a drive

The idea under test: model the distribution of the total and read any line off it (rather than one O/U
classifier per line); split the total into drives x points a drive; let the spread depend on pace.
Same discipline as PROTOCOL.md: walk-forward only (daily refits at midnight on matches settled 36 min after
kick-off, or online filters that take each result 36 min after its kick-off), settings chosen on TUNE
(06-01..07-31) and frozen, reported on TEST (08-01..10-18), SEP (09-01..10-18) and LATE (09-22..10-18), prod's
quotes only as the benchmark, intervals resample whole days (2,000 draws). No network; XGBoost nthread 2, the rest
single-threaded.

Code `scripts/F/`, outputs `out/F/`, predictions `preds_total/F_*.csv` (12,119 matches each, 06-01..10-18, every
match; 11,625 with a prod line). Columns: code, exp_total (the mean model), dist_mean, sd_total, p_over
(P(total > prod's line)), exp_drives, mean_model, dist.

## Headline

* **Our O/U beats prod's at prod's own line, in every window.** Best TUNE-chosen model (`F_blend_k4_lumpynb`: the
  blend of the expected totals, a negative binomial whose variance grows with the expected total, times
  per-integer "football" weights):
  | window | n | ours | prod | ours less prod (95%, days) |
  |---|---|---|---|---|
  | TUNE | 5,463 | 0.6763 | 0.6917 | -0.0155 (-0.0207, -0.0103) |
  | TEST | 6,162 | **0.6747** | 0.6926 | **-0.0179 (-0.0233, -0.0124)** |
  | SEP | 3,427 | 0.6790 | 0.6925 | -0.0135 (-0.0211, -0.0062) |
  | LATE | 1,841 | 0.6801 | 0.6919 | -0.0118 (-0.0214, -0.0025) |

  Prod's O/U is barely off a coin flip (0.6917-0.6926 against 0.6931): it sets its line near its median and quotes
  ~0.50 (sd 0.023), and it leans to the over (mean P(over) 0.500; overs land 48.8% TUNE, 48.4% TEST, 47.5% SEP).
  Ours averages 0.488 / 0.482 / 0.467 there; a normal about our mean would say 0.522 / 0.519 / 0.504.
  The walkthrough tables' own score models (NB2 shared gamma) already beat prod at its line in SEP
  (glmer_both -0.0095, avg -0.0110); ours is better again: **-0.0040 (-0.0074, -0.0004) vs glmer_both's own O/U
  in SEP, -0.0050 (-0.0095, -0.0001) in LATE**.
* **Expected total: RMSE 12.440 on TEST** (blend_k4) against prod's line 12.844 (on prod's matches 12.445 v 12.844,
  squared error -10.1 (-12.9, -7.4)); **SEP 12.449 against the tables' best, glmer_both, 12.491**: squared error
  -1.04 (-2.12, +0.08); LATE 12.554 v 12.630, -1.91 (-3.21, -0.54). Prod's line adds nothing to our mean (OLS of
  the total on both, day-clustered, TEST: blend_k4 0.93 (t 17.4), prod's line 0.07 (t 0.8); SEP 0.86 (t 11.0) and
  0.03 (t 0.2)).
* **Drives x points a drive is as good as a direct total model, not better**: ridge 12.535 v 12.543 (TEST, -0.19
  (-0.91, +0.51) in squared error), Kalman 12.504 v 12.469 (+0.88 (+0.28, +1.46), worse). It earns a small weight in
  the blend.
* **The spread depends on the expected total, not on pace.** Var ∝ mu^1.4-1.7 (sd 11.0 at an expected 30, 12.3 at
  35, 13.6 at 40); that is worth -0.011 nats of integer log score and -0.0010 of O/U log loss over a constant sd.
  Fast games are more variable only because they score more: given the expected total, expected drives add
  -0.0004 (-0.0008, +0.0001) of log score on TEST, nothing on the O/U.
* **Lumpiness is the biggest thing in the distribution**: per-integer weights (31, 24, 17, 38, 45 = 7k+3 come up
  1.3-2x as often as a smooth curve says; 30, 32, 29, 33 0.2-0.4x) take -0.26 to -0.30 nats off the integer log
  score. At prod's x.5 lines the skew (NB rather than normal) is what pays (-0.0025); the lumps add nothing more
  there.

## 0. Data facts

* **prod_total_prob is P(over).** Two independent checks: eAMFCalibrator/markets.py defines market 54 as
  (TOTAL, "Over") and 55 as "Under", and prematch_prob_54 is market 54's last pre-match quote; in the data, corr(prob,
  over) = +0.050, the over rate rises 0.43 -> 0.55 from the lowest to the highest decile of the prob, and the log loss
  of the prob read as P(over) is 0.6919 against 0.6964 read as P(under) (13,452 matches).
* **Every prod total line is x.5** (20.5 .. 56.5; 30.5, 31.5, 34.5 hold half of them), so there are no pushes: the
  "ties at the line" rule never bites (none counted half, none left out).
* The total is right-skewed (mean 33.7, median 31, skew 0.59) and very lumpy: since June a total of 31 is 10.6% of
  matches, 24 6.4%, 38 6.1%, 45 5.2%, 17 5.1%, but 30 2.2%, 32 1.6%. A side's points are 7s and 3s (try misses 6-7%,
  2-pointers 7-8% of one-TD sides); a match has 4.4 TDs (var 3.6, under-dispersed), 1.1 FGs (49% exactly one).
* The league total drifts (TUNE 33.5, TEST 35.3): every TUNE-best model tracks the level fast (ridge intercept
  re-levelled at a 3-day half-life; Kalman mu drift).

## 1. The expected total (`tune_ridge.py`, `means.py`, `mixed_total.py`, `kalman.py`, `tune_kalman.py`, `xgb_total.py`, `compare_means.py`)

| model | what | TUNE-frozen settings |
|---|---|---|
| b_tot | ridge: total = mu + stream + a[p1] + a[p2] + t[t1] + t[t2], rows 2^(-age/hl), daily refit | hl 30 d, 365 d window, lam_gamer 64, lam_team 80, intercept re-levelled at hl 3 d |
| b_side | ridge on each side's points, off/def gamer and team, home | identical to b_tot for the total (the sum/difference split decouples it exactly; TUNE RMSE equal to 1e-8); used for the margin |
| b_mixed | b_tot as a mixed model, REML variances each day (pace_models.mixed_fit) | REML asks lam_gamer ~15 (13.8-17.5), team ~220: under-shrinks, as for pace |
| c_prod (c) | drives ridge x points-a-drive ridge (side-level, drive-weighted, opponent-adjusted) / 2 x (ppd_h + ppd_a) | drives: hl 15, 120 d, lam 20 / 1440, re-level 3 d; ppd: hl 60, 180 d, lam 32 / 80, stream, re-level 7 d |
| k_tot | **online Kalman** on the total (gamer and team total effects random walks, mu drifting; full covariance; every result in at +36 min) | V_g 0.046, q_g 0.0005/d, V_t 10.5, q_t 1.6e-5, q_mu 2.3e-4 (ratios to the noise) |
| k_pts | the same per side (off/def) | gives the margin as well |
| k_c (c) | Kalman drives x Kalman points a drive | k_drv V_g 0.016, q_g 5.7e-4; k_ppd V_g 0.0044, q_g 3e-5 |
| xgb_base (d) | XGBoost on features.csv (p1/p2 pairs as sum and abs difference + match-level), refit weekly, 180 d, trees by early stopping on the last 14 d | depth 4, eta 0.03, mcw 30 (backtest.py's), 60-500 trees |
| xgb_stack / xgb_stack_k | + the ridge / Kalman walk-forward outputs as features | same |
| C_kal | agent C's points Kalman (k_ptsall_pred_h + _a), for reference | C's |
| blend_* (e) | each day an OLS of the total on members' walk-forward predictions, last W days | W 56 chosen on TUNE for every blend; **blend_k4 = k_tot, k_c, b_tot, xgb_base** best on TUNE (11.978) |

RMSE (common matches), bias = mean(pred - total), slope = total on pred (1 calibrated), and squared error against
b_tot (95%, days). Full tables: `out/F/means_report.txt`.

| model | TUNE RMSE | TEST RMSE | TEST bias / slope / r | SEP RMSE | LATE RMSE | TEST sq err vs b_tot |
|---|---|---|---|---|---|---|
| naive (features' exp_total_50) | 12.145 | 12.579 | -0.08 / 1.14 / .287 | 12.550 | 12.602 | +0.91 (-0.81, +2.57) |
| b_tot (ridge) | 12.033 | 12.543 | -0.15 / 0.95 / .294 | 12.537 | 12.676 | |
| b_mixed (REML) | 12.063 | 12.558 | -0.80 / 0.91 / .298 | 12.544 | 12.694 | +0.38 (-0.14, +0.90) |
| c_prod (ridge drives x ppd) | 12.026 | 12.535 | -0.13 / 0.94 / .296 | 12.548 | 12.717 | -0.19 (-0.91, +0.51) |
| k_tot (Kalman) | 11.994 | 12.469 | -0.12 / 0.90 / .313 | 12.498 | 12.627 | -1.85 (-2.78, -0.97) |
| k_c (Kalman drives x ppd) | 11.992 | 12.504 | -0.06 / 0.92 / .304 | 12.516 | 12.657 | -0.97 (-1.75, -0.21) |
| xgb_base | 12.092 | 12.501 | +0.31 / 1.06 / .305 | 12.490 | 12.553 | -1.05 (-2.67, +0.46) |
| xgb_stack_k | 12.013 | 12.442 | +0.03 / 0.99 / .318 | 12.436 | 12.538 | -2.53 (-3.62, -1.50) |
| C_kal (agent C) | 11.995 | 12.478 | -0.12 / 0.92 / .310 | 12.498 | 12.637 | -1.63 (-2.36, -0.95) |
| **blend_k4** | **11.980** | **12.440** | +0.01 / 0.97 / .318 | **12.449** | **12.554** | **-2.56 (-3.75, -1.47)** |
| prod's line (its matches) | 12.397 | 12.844 | -0.85 / 1.16 / .215 | 12.673 | 12.781 | |

SEP, the tables (3,463 matches): glmer_both 12.491, glmer 12.493, avg 12.499, form 12.499, avg_plain 12.524,
nb2_infer 12.541, nb2_both 12.563, nb2 12.601 (all biased low by 0.32-0.43 and overspread, slope 0.78-0.84).
blend_k4 12.449 (-1.04 (-2.12, +0.08) in squared error vs glmer_both), xgb_stack_k 12.436, k_tot 12.498 (level
with glmer_both), b_tot 12.537. LATE: glmer_both 12.630, blend_k4 12.554 (-1.91 (-3.21, -0.54)).

Reading: the online Kalman beats the midnight ridge (probably because it takes the day's earlier results in, and a
random walk decays better than one half-life); XGBoost on the rolling features adds something different from both (blend weight
~0.38); the drives x ppd split is no better than modelling the total directly -- a match's drive count and how
its drives go pull against each other (realised drives vs points a drive: r -0.31; the total's elasticity to the
realised drives about their forecast is 0.52, not 1), so the product adds noise the direct total avoids. All our
means are less biased than the tables (+-0.1 against -0.4); the blends are a little better calibrated in SEP (slope
0.88 against 0.83; k_tot alone 0.81). SEP is a hard window: every model's slope drops there.

**Drives (the sim's pace input)**, RMSE / r / slope:
| | TUNE | TEST | SEP |
|---|---|---|---|
| pace_models ridge (hl 60, 180 d, lam 90/360: the walkthrough's pace_prior) | 2.527 / .365 / 1.50 | 2.645 / .279 / 1.02 | 2.593 / .257 / 1.00 |
| ridge tuned on TUNE (hl 15, lam 20/1440, re-level 3 d) | 2.489 / .376 / 1.01 | 2.626 / .284 / 0.84 | 2.593 / .249 / 0.81 |
| REML mixed | 2.500 / .369 / 0.95 | 2.635 / .278 / 0.81 | 2.598 / .243 / 0.80 |
| k_drv (Kalman) | 2.480 / .384 / 0.98 | 2.621 / .291 / 0.83 | 2.588 / .257 / 0.81 |
The RMSE gains are mostly the level (pace_models' -0.3 bias); in ranking (r) and spread (slope) -- what the sim's
pace nudge reads, as it takes exp_drives over the table's mean -- the walkthrough's ridge is as good as anything
here and better calibrated after TUNE. No change recommended for the drives figure.

## 2. The distribution (`dist.py`, `dist_eval.py`, `dist_score.py`)

Every 7 days at midnight, on the matches settled in the last 120 days (W chosen on TUNE from 30/60/120 with the blend3 mean; the
residuals are the walk-forward means' own, so out of sample):
* **variance function**: log V = a0 + a1 log(mu/35) + a2 log(muD/10.3) by Gaussian ML on the residuals; forms
  const, mu, D (expected drives), mu+D.
* **normal** on whole points; **NB** (mean mu, that variance); **lumpy**: normal or NB x per-integer weights w_k
  (iterative proportional fitting on the window, 5 pseudo-counts, alpha chosen on TUNE from 1/5/25), the mean put
  back to mu by shifting the base.
* **compound (drives x points a drive, the idea as a distribution)**: D ~ discretised normal(muD, sigD); each drive
  scores 7 (6 / 8 after the try) w.p. p_t, 3 w.p. p_f, else 0; the rates scale with (D/muD)^beta and a match-level
  lognormal factor (sd tau); the rates' level set so the mean is mu (the TD/FG mix from the TD- and FG-a-drive
  ridges); built through the characteristic function (FFT). Fitted each refit: sigD 2.40-2.62 (the drives' miss),
  **beta -0.46 to -0.49** (a game with 10% more drives than expected scores ~5% more, not 10%), q6 0.06-0.075,
  q8 0.07-0.08, tau 0.02-0.12 (ML). Variants: D fixed at muD; x per-integer weights.

Scores on the blend_k4 mean (`out/F/dist_final_nc_score.txt`, `dist_final_score.txt`): the integer total's log
score and RPS, and the O/U log loss at prod's line (5,463 / 6,162 / 3,427 / 1,841 matches), O/U calibration slope:

| distribution | TUNE logS | TEST logS | TEST RPS | TEST O/U (slope) | SEP O/U (slope) | LATE O/U |
|---|---|---|---|---|---|---|
| normal, constant sd | 3.9022 | 3.9403 | 6.9418 | 0.6790 (0.89) | 0.6813 (0.83) | 0.6829 |
| normal, sd(mu) | 3.8917 | 3.9337 | 6.9305 | 0.6772 (0.94) | 0.6801 (0.85) | 0.6813 |
| NB, var(mu) | 3.8786 | 3.9130 | 6.9103 | 0.6747 (0.93) | 0.6786 (0.85) | 0.6807 |
| lumpy normal, const | 3.6054 | 3.6637 | 6.9070 | 0.6754 (0.87) | 0.6794 (0.78) | 0.6802 |
| **lumpy NB, var(mu)** | **3.5998** | **3.6568** | **6.9015** | **0.6747 (0.91)** | **0.6790 (0.82)** | **0.6801** |
| lumpy NB, var(mu, D) | 3.5995 | 3.6564 | 6.9008 | 0.6747 (0.92) | 0.6789 (0.83) | 0.6801 |
| compound: drives x 7 / 3 / 0 a drive, D random | 3.6815 | 3.7369 | 6.9158 | 0.6757 (0.88) | 0.6791 (0.81) | 0.6803 |
| compound, D fixed at its forecast | 3.7621 | 3.8294 | 6.9403 | 0.6782 (0.83) | 0.6812 (0.76) | 0.6819 |
| compound x per-integer weights | 3.6048 | 3.6638 | 6.9013 | 0.6746 (0.92) | 0.6790 (0.82) | 0.6802 |
| prod's P(over) | | | | 0.6926 (0.97) | 0.6925 (1.03) | 0.6919 |

* **The drives x points-a-drive compound, as a distribution**: from first principles (7s and 3s a drive, random
  drives) it recovers two thirds of the lumpiness (TEST log score 3.737 against the NB's 3.913 and the lumpy NB's
  3.657: +0.080 (+0.069, +0.092) behind the lumpy NB); letting the drives vary is essential (D fixed at its forecast:
  +0.17 log score, +0.0035 (+0.0015, +0.0056) O/U); with the per-integer weights on top it is level with the lumpy NB
  on the O/U (-0.0001 (-0.0004, +0.0002)) and a little behind on the integer log score (+0.0070 (+0.0030, +0.0110)).
  The lumpy NB is the TUNE pick (3.5998 against 3.6048) and is far simpler: the structure is right, but empirical
  lumps capture what independent per-drive rates miss (one field goal a match is far commoner than they allow: 49%
  of matches have exactly one against ~38% under a per-drive binomial; TDs and FGs trade off within a match).
* Lumpiness is worth -0.26 to -0.30 nats of integer log score in every window (intervals ~+-0.02). The O/U at x.5
  lines barely needs it: what the O/U needs is the skew -- the normal puts the median at the mean and so prices
  the over too high (calibration intercept -0.14 to -0.17); the NB (or the lumps) fix it, -0.0025 of O/U log loss.
* The lumpy NB with var(mu) is the TUNE pick (var(mu, D) is level: -0.0003 on TUNE log score, inside noise; it is
  the simpler one). The fitted weights (last refit, `out/F/dist_final_nc_lumps_last.csv`): 3: 17, 7: 1.9, 10: 2.1,
  17: 2.0, 24: 1.3, 31: 1.9, 38: 1.4, 45: 1.75, 52: 1.05, 59: 1.4; 21 0.88, 28 0.51, 35 0.86, 42 0.53; 8, 11, 12,
  15, 18 under 0.1.
* The O/U slopes are below 1 in SEP/LATE (0.82-0.85): the expected totals there are a little overspread (the means'
  own slope 0.88 in SEP); TUNE showed none of it (0.96), so nothing was shrunk.
* Other means through the same lumpy NB (TEST / SEP / LATE O/U vs prod): k_tot alone -0.0163 / -0.0109 / -0.0086;
  xgb_stack_k -0.0186 / -0.0144 / -0.0123 (not TUNE-chosen); the tables' means with our shape (SEP, variance and
  lumps fitted on our blend's residuals): glmer_both -0.0105, avg -0.0107 -- about their own NB2 O/U (-0.0095,
  -0.0110): in SEP the tables' gap to us is the mean, not the shape.
* Integer log score in SEP, each mean through the lumpy NB: blend_k4 3.6668 v glmer_both's mean 3.6701 (-0.0033
  (-0.0070, +0.0004)); LATE -0.0062 (-0.0105, -0.0015).

## 3. Does the spread depend on pace? (`pace_spread.py`, `out/F/pace_spread_final_nc.txt`)

Fitted variance function (blend_k4, refit path 06-01 .. 10-12): mu-only log V = 5.02-5.09 + **1.44-1.71** log(mu/35):
**sd = 12.3 (mu/35)^0.72** at the last refit (exponent 0.72-0.85 over the refits; sd 9.7 / 11.0 / 12.3 / 13.6 /
14.8 at an expected 25 / 30 / 35 / 40 / 45).
With both: a1 1.16-1.82, a2 (drives) -0.36 .. +0.63, last +0.51: sd 12.0 at 9.3 drives against 12.6 at 11.3, mu 35.

Each pace-aware form less its pace-free twin (95% days), TEST / SEP:
| | log score | RPS | O/U log loss |
|---|---|---|---|
| normal sd(mu) - constant | -0.0065 (-0.0106,-0.0023) / -0.0051 | -0.011 / -0.008 | -0.0018 (-0.0027,-0.0010) / -0.0012 |
| normal sd(D) - constant | -0.0054 / -0.0056 | -0.009 / -0.009 | -0.0010 / -0.0008 |
| lumpy NB var(mu) - constant | -0.0109 (-0.0154,-0.0067) / -0.0108 | -0.019 / -0.017 | -0.0010 (-0.0020,+0.0000) / -0.0004 |
| **NB var(mu, D) - var(mu)** | **-0.0005 (-0.0010,-0.0000) / -0.0009 (-0.0017,-0.0001)** | -0.0006 / -0.0010 | +0.0000 / -0.0000 |
| lumpy NB var(mu, D) - var(mu) | -0.0004 (-0.0008,+0.0001) / -0.0007 (-0.0015,+0.0001) | -0.0007 / -0.0011 | -0.0000 / -0.0001 |

Binned (TEST): the residual sd by fifth of expected drives runs 10.9 -> 13.4 (9.2 -> 11.7 drives), but the
expected total runs 31.2 -> 39.4 across the same fifths; by fifth of the expected total the sd runs 11.1 -> 13.7.
So: **the spread grows with the expected total (variance ~ mu^1.5, Poisson-like with lumps of 7); fast games are
more variable because they score more; given the expected total, pace adds a trace (sd +4% from slowest to fastest
fifth, a twentieth of the mu effect, and nothing on the O/U).** The compound model agrees: at a fixed mean, more
drives means lower rates a drive, so its variance barely moves with pace.

## 4. The margin jointly with the total? (`margin_joint.py`, `out/F/margin_joint.txt`)

Margin from the per-side Kalman (k_pts_margin), total from blend_k4:
* corr(margin error, total error) **-0.03** (TUNE -0.028, TEST -0.034, SEP -0.035); corr(|margin error|, total
  error) -0.01 .. 0.00; corr(favourite's margin error, total error) -0.04 .. +0.01: the two errors are close to
  independent.
* The margin's spread grows with **pace**, not with the expected total: margin sd by fifth of expected drives 8.5 ->
  10.1 (TEST), by fifth of the expected total 9.05 -> 9.80; fitted log Vm = 4.47 - 0.45 log(mu/35) + 1.85
  log(muD/10.3). By fifth of the *realised* total the margin sd is flat (9.2 -> 9.9).
* The moneyline from a pace / total dependent margin sd (normal about the margin): +-0.0002 against a constant sd in
  every window (intervals +-0.0005). Nothing for the moneyline; worth it only for spreads and alternative lines,
  which the sim's pace nudge already carries (README: v21's margin sd should follow pace).
So no joint margin-total model is needed: modelling the total's distribution separately loses nothing measurable.
(By-product: k_pts_margin as a moneyline, sd fitted walk-forward: TEST 0.6748, -0.0100 (-0.0141,-0.0059) vs prod;
SEP 0.6788, close to agent C's C_strength.)

## 5. Recommendation for the sim's targets

1. **Expected total**: the online Kalman of the total (k_tot: gamer and team total effects as random walks, all
   results in 36 min after kick-off, settings above) is the one model to productionise if only one: TEST RMSE
   12.469 (-1.85 sq err vs the ridge), level with glmer_both in SEP, with the day's own results in. If a blend can
   be run, blend_k4 (k_tot, k_c, b_tot, xgb_base; daily OLS on the last 56 days) adds -0.7 squared error more and
   is the best here (TEST 12.440, SEP 12.449 v glmer_both 12.491). Feed it as the total target (split by the
   table's margin into each side's points).
2. **sd of the total**: **sd = 12.3 (mu/35)^0.75** (variance ∝ mu^1.5), refitted weekly on the last 120 days'
   out-of-sample residuals; do not add pace to it (a 4% effect worth nothing on O/U). For prices at a line use the
   lumpy NB (NB with that variance x the per-integer weights) -- the normal misprices the over by the skew
   (+3.5 points of P(over) at prod's lines on TEST: 0.519 against 0.484 overs; the lumpy NB 0.482).
3. **Drives**: keep the walkthrough's pace ridge (pace_prior) for exp_drives; the tuned ridge / Kalman do no better in
   ranking or spread. The drives x ppd decomposition is not worth adding for the mean; its place is the shape
   (the compound reproduces the lumps but the plain lumpy NB is as good and far simpler).
4. The v2.1 sim's own total spread should be checked against sd(mu) above: at a fixed expected total it should not
   widen with pace; its margin spread should (README).

## Files

scripts/F: flib.py (panel, designs, walk-forward ridge, scoring), tune_ridge.py, means.py, mixed_total.py,
kalman.py, tune_kalman.py, xgb_total.py, compare_means.py, dist.py, dist_eval.py, dist_score.py, pace_spread.py,
margin_joint.py, tables_ou.py, write_preds.py.
out/F: means_report.txt, tuned.json, kalman_tuned.json, tune_*.csv, means_all.pkl (every expected total),
dist_<tag>.pkl (+ _params.csv, _lumps_last.csv, _score.txt; tags final / final_nc = blend_k4, ktot, xgbk,
tbl_glmer_both, tbl_avg, blend3_W60 and the TUNE grid b3_*), pmf_final_nc_lumpy_nb_mu.pkl (every match's PMF on
0..159), pace_spread_*.txt, margin_joint.txt, tables_ou.pkl, logs.
preds_total: F_blend_k4_lumpynb (the pick), F_blend_k4_normal (normal, sd(mu)), F_ktot_lumpynb (one-model
production candidate), F_xgbk_lumpynb, F_blend_k4_compound_lumpy (the compound x weights).
