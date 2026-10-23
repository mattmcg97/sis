# Agent C: a moneyline model on the h2h site's stats (red zone, 3rd down, yards per play ...)

All scripts in `scripts/C/`, outputs in `out/C/`, predictions in `preds/C_*.csv`. No outside calls: the
h2h site's stats are rebuilt from our own play-by-play (`xgb/data/drive_stats.csv`) and match history.

## 1. What the site shows and what we can rebuild

| site stat | ours | source |
|---|---|---|
| matches, wins, win %, streaks | win % (all time, last 10), current streak (signed) | match history (features.csv; streak rebuilt in `eb.py`) |
| points for / against | points a match (and a drive) | final scores |
| touchdowns | TDs a drive, TDs a match | drive_stats `td` |
| red zone | trips a drive, TD % of trips | `rz_trips`, `rz_td` |
| third / fourth down | conversion % | `third_*`, `fourth_*` |
| yards per play | yards a gained snap | `yards / gains` |
| total offence (match) | yards a match | `yards` |
| first downs (match) | **approximation**: snaps that reached the line to gain or scored (`success`) a match | `success` |
| rushing / passing (match) | **not available** | -- |

Plus the rest of drive_stats: field goals, punts, turnovers and turnovers on downs a drive, plays a drive,
drive start, success rate, 20+ / 40+ yard plays, losses, drives a match, clock a snap, turnovers a snap.

**Not rebuildable from our data.** SCOUTING_FULL's message vocabulary (PLAY_STARTED / PLAY_OVER, TOUCHDOWN_,
KICKOFF_, PUNT_, FIELD_GOAL_, EXTRA_POINT_, TWO_POINT_, SAFETY_, TURNOVER_ON_DOWNS_, POSSESSION_, FOUL_) has
no play type, so rushing vs passing yards, completions, sacks, and interceptions vs fumbles cannot be told
apart from it: only the site's match/stats has rush/pass splits. **First downs** could be rebuilt exactly
from the scouting export (scouting_playover.csv): a snap whose next snap is the same offence's 1st down (or
a TD) is a first down; `xgb/drives.py`'s `success` already counts that plus the offensive TDs, so a
`first_downs = success - scrimmage TDs` counter would be a two-line addition to `drives.snaps`.
`eAMFCalibrator/player_stats.py` also rebuilds, off SCOUTING_FULL, time of possession, punts' net yards,
red-zone field goals, red zone per set of downs, two-point tries and fouls (its `player_match_stats.csv`
is not in the repo, so those are not used here). Note: features.csv's `drv_*_dppd_*` are all 0 (this
drive_stats.csv has no `drive_pts` column); I use the final score for points.

Coverage of drive_stats: 94% of 04-27..30, 50% of May, 91% of June, 99%+ from July.

## 2. Feature construction (all as of kick-off; a result counts 36 min after its kick-off)

**Observations** (`common.py`): per match and side, 24 stats as (numerator, denominator) off drive_stats,
joined to the history by match code (drive_stats' home = PLAYER_1; finals agree on all 13,635):
`pts` points a match, `ppd` points a drive, `td` TDs a drive, `fg`, `punt`, `turnover`, `downs` a drive, `rz`
red-zone trips a drive, `rz_td` TDs a trip, `plays` a drive, `start` drive start (yards from own goal),
`ypp` yards a gained snap, `success` (line to gain or TD) a snap, `big20`, `big40`, `loss` a snap, `third`,
`fourth` conversion %, `drives` a match, `pace` clock seconds a snap, `tds_pm` TDs a match, `yds_pm` total
offence a match, `fd_pm` successful snaps a match (~ first downs), `to_snap` turnovers a snap.

**Variance components** (`kalman.variance_components`, on 05-01..07-31 only, so nothing from TEST):
noise about a side's true rate var(y|n) = b/n for rates (b = mean of (y-league)^2 x n over matches with
n >= its 25th percentile, less the true spread; the low-n tail is inflated, an LS fit of A + b/n was pulled
by it and over-shrank), a constant for per-match counts; the gamers' true spread off / def = split-half
covariance of gamer-month pooled rates; teams' by pooled team rates less sampling variance. The implied
EB prior weight m = noise / between-variance, in the stat's own units: points 15 matches; success 370
snaps (~19 matches); ypp 416 gained snaps (~23 matches); TD a drive 135 drives (~28 matches); turnover a drive
150 drives (~31 matches); 3rd down 243 attempts (~70 matches); **red-zone TD% 197 trips (~100 matches)**;
4th down 93-554 attempts.

**Kalman, opponent-adjusted** (`kalman.py`, the `kal_` features): per stat a full-covariance Kalman filter on
y = mu + home + off[offence gamer] + def[defence gamer] + offT[team] + defT[team] + e, gamers' effects
N(0, V_off / V_def) at debut, random-walk drift q x V a day once active, teams drifting 0.002 V a day, mu
diffuse; every result updates the joint posterior, so a gamer's numbers are net of whom he played (a
time-decayed joint ridge refreshed on every result) and shrunk by his sample (a 3-trip red zone barely
moves him). q per stat from {0, .002, .005, .01, .02, .05} by the one-step-ahead predictive log likelihood
of the stat itself on side-matches 05-15..07-31 (no win/loss): 0.01 for most, 0.005 for pts, downs, fg,
loss, rz, tds_pm, yds_pm, ypp, drives; 0.002 rz_td; 0 fourth (static). The same filter on points over
the whole history (2025-12-10 on, every match) is **`s_kal`, the strength margin**: (home's expected points
less away's), q = 0.005 V a day.
Matchup feature: `kal_<s>` = expected home offence rate less expected away offence rate
= (off[p1] + def[p2] + teams + home) - (off[p2] + def[p1] + teams).

**EB-shrunk raw** (`eb.py`, `eb<hl>_`): per gamer recency-weighted (2^(-k/hl) for his k-th last pbp match;
hl 15/30/60) sums of numerator and denominator, offence and offences faced, shrunk to the league
(30-day half-life) by the m above; matchup off[p1] + def[p2] - off[p2] - def[p1]. Not opponent-adjusted.
**Raw windows** (`raw15_`, `raw50_`): features.csv's last-15 / 50 rates, same combination (no 4th down there).
**Site non-pbp**: `win_all`, `win_10` (home less away), `streak` (signed current streak, capped at 6, home
less away, rebuilt in eb.py). `s_rating` = features' rating_margin.

## 3. Repeatability: skill or noise? (`repeat.py`, out/C/repeatability.txt)

260 gamer-months with 2 x 25 pbp matches, alternating into halves (each less its month's league mean);
r = split-half correlation = reliability of a 25-match window; n50 = matches for reliability 0.5
(Spearman-Brown); r_all = the same over the whole span (71 gamers with 100+ matches, ~150 a half);
r_marg = corr(net stat in one half, scoring margin a match in the other); partial t = the net stat's t in
margin_other ~ points margin_same + stat_same (signal beyond points; ~70 tests, so |t| < 2.5 is noise).
"net" = the gamer's offence less what his opponents' offences did against him.

| stat | league | r off | r def | r net | n50 net | r_all net | r_marg net | partial t |
|---|---|---|---|---|---|---|---|---|
| pts (points a match) | 17.1 | 0.65 | 0.57 | 0.42 | 34 | 0.69 | +0.42 | - |
| ppd | 3.37 | 0.50 | 0.38 | 0.35 | 46 | 0.67 | +0.37 | -0.1 |
| td a drive | 0.364 | 0.51 | 0.35 | 0.38 | 42 | 0.63 | +0.33 | +1.2 |
| tds_pm | 1.85 | 0.61 | 0.41 | 0.39 | 39 | 0.65 | +0.37 | +1.2 |
| fg a drive | 0.094 | 0.38 | 0.13 | 0.31 | 57 | 0.69 | +0.04 | -1.3 |
| punt a drive | 0.066 | 0.37 | 0.44 | 0.19 | 104 | 0.61 | +0.07 | -0.1 |
| turnover a drive | 0.262 | 0.43 | 0.32 | 0.35 | 46 | 0.71 | -0.34 | +0.4 |
| to_snap | 0.067 | 0.43 | 0.31 | 0.26 | 73 | 0.61 | -0.28 | +2.8 |
| downs a drive | 0.045 | 0.31 | 0.47 | 0.38 | 41 | 0.71 | +0.20 | +0.9 |
| rz trips a drive | 0.364 | 0.39 | 0.49 | 0.43 | 33 | 0.80 | +0.16 | -0.1 |
| **rz_td %** | 0.630 | **0.20** | **0.19** | **0.23** | **85** | 0.45 | +0.15 | +1.5 |
| **third %** | 0.528 | **0.29** | **0.26** | **0.23** | **83** | 0.60 | +0.13 | -1.4 |
| **fourth %** | 0.514 | **0.18** | **0.16** | **0.20** | **102** | 0.31 | -0.02 | +0.4 |
| success | 0.423 | 0.54 | 0.48 | 0.46 | 29 | 0.82 | +0.29 | +2.5 |
| ypp | 7.57 | 0.53 | 0.50 | 0.44 | 32 | 0.70 | +0.12 | +1.7 |
| big20 | 0.136 | 0.69 | 0.65 | 0.64 | 14 | 0.86 | +0.13 | +1.4 |
| big40 | 0.038 | 0.58 | 0.63 | 0.48 | 27 | 0.82 | +0.00 | -0.2 |
| loss | 0.118 | 0.68 | 0.46 | 0.50 | 25 | 0.79 | -0.08 | -1.8 |
| start (field pos.) | 38.7 | 0.42 | 0.57 | 0.51 | 24 | 0.80 | +0.25 | +1.8 |
| plays a drive | 4.23 | 0.63 | 0.58 | 0.60 | 17 | 0.82 | -0.17 | -2.8 |
| drives a match | 5.07 | 0.60 | 0.62 | 0.21 | 93 | 0.57 | +0.23 | +0.6 |
| pace (s a snap) | 20.8 | 0.71 | 0.54 | 0.53 | 22 | 0.84 | +0.01 | -0.8 |
| yds_pm (total offence) | 144 | 0.63 | 0.44 | 0.38 | 41 | 0.75 | +0.09 | -0.7 |
| fd_pm (~first downs) | 8.34 | 0.65 | 0.51 | 0.55 | 21 | 0.86 | +0.10 | -0.9 |
| win % | - | 0.31 | - | 0.31 | 56 | 0.63 | +0.35 | - |
| scoring margin | - | 0.42 | - | 0.42 | 34 | 0.69 | +0.42 | - |

Reading: **red-zone TD %, 3rd and 4th down % are mostly noise** at the windows the site shows (25 matches:
r 0.18-0.29; ~85-100 matches for half signal); style stats (pace, plays a drive, big plays, losses, first
downs, total yards) are very repeatable but barely foretell the margin; what foretells the next matches'
margin is the scoring margin itself (0.42), then points/TDs a drive (0.33-0.37), turnovers (-0.34), success
rate (0.29) and field position (0.25). Beyond the points margin nothing is clearly significant (success
+2.5, plays -2.8, turnovers a snap +2.8 out of ~70 tests).

## 4. Models (`models.py`, walk-forward with `harness.walk_forward`)

Daily refits on every match settled by the day's midnight from 2026-05-04 (expanding; a rolling 60-day
window was tried and won only for the stats-only model), 06-01..10-18 priced. Features standardised on each fit's training rows (NaN -> the
training mean). sklearn logistic, L2 (lbfgs) or L1 (liblinear); the strength inputs' z-scores are multiplied
by 5 so their L2 penalty is 1/25 of a stat's. Every choice below by TUNE walk-forward log loss only
(`tune_grid.csv`, ~170 configurations); TEST/SEP never looked at for a choice.

| name (preds/) | what | TUNE-chosen setting |
|---|---|---|
| C_strength | logistic on s_rating (features' rating) + s_kal (my points Kalman margin) | (strength only, C=1) |
| C_stats | **(a)** stats only: all 24 Kalman matchups + win %, last-10, streak | L2, C=0.003, 60-day window |
| C_stats_site | (a) on the site's own list: pts, tds_pm, rz, rz_td, third, fourth, ypp, fd_pm, yds_pm + win %, last-10, streak | L2, C=0.01 |
| C_stats_strength | **(b)** strength (both) + all stats | L2, C=0.0003 (the grid's smallest: stats squeezed out) |
| C_four_factors | **(c)** success, big20, ypp, rz_td, turnover, third, start (Kalman matchups) | L2, C=0.1 |
| C_four_factors_strength | (c) + s_kal | raw50 version, C=0.01 |
| C_stats_mlink | **(d)** ridge of the margin on the site stats, logistic on the predicted margin | alpha=100 |
| C_stats_strength_mlink | (d) with both strengths among the margin's inputs | alpha=1e4 |
| C_stats_strength_newgamer | strength + stats + (stats, strength) x g, g = 1/(1+min games/50) | L2, C=0.003 |

Strength choice on TUNE (5,462 matches with prod): s_rating alone 0.6704, s_kal alone 0.6639, both 0.6635
(TEST: 0.6833, 0.6742, 0.6743).

## 5. Results (harness.report, common matches, day-resampled 95% intervals; `final.txt`, `more.txt`, `newgamer.txt`)

| window | model | log loss | Brier | AUC | slope | vs prod | vs C_strength |
|---|---|---|---|---|---|---|---|
| TUNE (5,462) | prod | 0.6764 | 0.2418 | 0.605 | 1.13 | | +0.0128 |
| | C_strength | 0.6635 | 0.2355 | 0.639 | 1.01 | -0.0128 (-0.0193,-0.0068) | |
| | C_stats | 0.6720 | 0.2396 | 0.616 | 1.07 | -0.0044 (-0.0095,+0.0002) | +0.0084 (+0.0036,+0.0132) |
| | C_stats_strength | 0.6628 | 0.2351 | 0.642 | 1.10 | -0.0136 | -0.0007 (-0.0017,+0.0002) |
| | C_four_factors | 0.6784 | 0.2427 | 0.599 | 1.26 | +0.0020 | +0.0148 |
| | C_stats_strength_mlink | 0.6627 | 0.2351 | 0.641 | 1.00 | -0.0137 | -0.0008 (-0.0016,-0.0002) |
| TEST (6,170) | prod | 0.6848 | 0.2460 | 0.569 | 0.93 | | +0.0105 |
| | **C_strength** | **0.6743** | 0.2408 | 0.606 | 0.97 | **-0.0105 (-0.0151,-0.0061)** | |
| | C_stats | 0.6786 | 0.2428 | 0.594 | 0.97 | -0.0062 (-0.0106,-0.0020) | +0.0043 (+0.0017,+0.0069) |
| | C_stats_site | 0.6788 | 0.2429 | 0.593 | 1.06 | -0.0060 (-0.0102,-0.0019) | +0.0045 (+0.0016,+0.0072) |
| | C_stats_strength | 0.6743 | 0.2408 | 0.606 | 1.01 | -0.0105 (-0.0149,-0.0062) | **+0.0000 (-0.0007,+0.0007)** |
| | C_four_factors | 0.6852 | 0.2460 | 0.572 | 0.87 | +0.0004 (-0.0035,+0.0041) | +0.0109 |
| | C_four_factors_strength | 0.6746 | 0.2409 | 0.606 | 0.97 | -0.0102 | +0.0003 (-0.0006,+0.0011) |
| | C_stats_mlink | 0.6789 | 0.2430 | 0.593 | 1.00 | -0.0058 | +0.0047 |
| | C_stats_strength_mlink | 0.6744 | 0.2408 | 0.606 | 0.96 | -0.0104 | +0.0001 (-0.0004,+0.0005) |
| | C_stats_strength_newgamer | 0.6737 | 0.2405 | 0.608 | 1.00 | -0.0110 (-0.0154,-0.0067) | -0.0005 (-0.0016,+0.0005) |
| SEP (3,426) | prod | 0.6894 | 0.2482 | 0.550 | 0.68 | | |
| | tbl_glmer_both_p | 0.6774 | 0.2423 | 0.595 | 1.10 | -0.0120 | |
| | C_strength | 0.6781 | 0.2427 | 0.593 | 0.97 | -0.0113 (-0.0175,-0.0053) | (vs glmer_both +0.0007 (-0.0016,+0.0031)) |
| | C_stats | 0.6813 | 0.2442 | 0.582 | 0.97 | -0.0081 | +0.0032 (-0.0000,+0.0065) |
| | C_stats_strength | 0.6783 | 0.2427 | 0.593 | 1.00 | -0.0111 | +0.0002 (-0.0007,+0.0011) |
| | C_stats_strength_newgamer | 0.6778 | 0.2425 | 0.594 | 1.00 | -0.0116 | -0.0003 (-0.0018,+0.0012) |
| 09-22..10-18 (1,840) | prod | 0.6880 | 0.2475 | 0.557 | 0.79 | | |
| | C_strength | 0.6793 | 0.2432 | 0.589 | 0.98 | -0.0086 (-0.0164,-0.0012) | (vs glmer_both +0.0016 (-0.0007,+0.0040)) |
| | C_stats | 0.6847 | 0.2458 | 0.572 | 0.90 | -0.0032 | +0.0054 (+0.0009,+0.0099) |
| | C_stats_strength | 0.6799 | 0.2435 | 0.588 | 1.00 | -0.0080 | +0.0006 (-0.0005,+0.0019) |

**The nested question, strength + stats over strength alone (same strength inputs and training rows):**
TEST +0.0000 (-0.0007, +0.0007) for the logistic, +0.0001 (-0.0004, +0.0005) for the margin-link, +0.0003
(-0.0006, +0.0011) for the four factors; SEP +0.0002 (-0.0007, +0.0011); 09-22..10-18 +0.0006 (-0.0005,
+0.0019). TUNE (where they were chosen) -0.0007 / -0.0008. **The stats add nothing measurable to a good
points-based strength.** They do add to a weak one: on top of features' rating_margin alone the stats gain
-0.0040 (-0.0063, -0.0018) on TEST, but that is the stats (and their opponent-adjusted points, kal_pts)
standing in for a better strength, not information beyond points: on top of s_kal the gain vanishes.

**Where they help: new gamers.** TEST, C_stats_strength less strength alone by the lesser gamer's matches
played: <100 matches (310) -0.0047 (-0.0086, -0.0009); 100-200 -0.0011; 200-500 -0.0001; 500+ +0.0005. The
same on TUNE: <100 (925) -0.0037 (-0.0070, -0.0006). A model letting the stats fade with experience
(C_stats_strength_newgamer, g = 1/(1+min games/50), chosen on TUNE) gets -0.0005 (-0.0016, +0.0005) overall on
TEST and -0.012 (-0.022, -0.001) on the 310 new-gamer matches: a real but small (5% of matches) effect.

**Feature construction matters for stats-only models** (TEST, vs prod, each at its TUNE-best C): Kalman
opponent-adjusted -0.0057 (all) / -0.0060 (site list); EB-shrunk raw (hl 30) -0.0010; raw last-15 -0.0007;
raw last-50 -0.0006. With s_kal underneath all of them are level with strength (+0.0001..+0.0005).
**Pure play-by-play efficiency without the points**: stats-only without pts / win % (`pbp_nopts`) TEST
0.6799, -0.0049 (-0.0086, -0.0015) vs prod, +0.0056 vs C_strength; without any scoring stat (no pts, ppd,
td, tds_pm, fg: yards, success, red zone trips and %, turnovers, downs, field position, pace) 0.6831,
-0.0017 (-0.0052, +0.0015) vs prod: the box score without the scoreboard prices about as well as prod.

### Which stats matter

Add one stat to strength (both) alone, logistic C=1, walk-forward (`addone.csv`): no stat's gain is
distinguishable from zero on TEST; the TUNE leaders (win % -0.0005, red-zone trips a drive -0.0004, success
-0.0003, first downs -0.0003) are -0.0002, -0.0001, +0.0003, +0.0001 on TEST. Red-zone TD %, third and fourth
down, ypp, big plays: +0.0000..+0.0003 on both.

Four-factors (stats only, Kalman matchups, coefficients per sd of the matchup, fitted on all settled by
10-19; by 08-01 in brackets): **turnovers a drive -0.204 (-0.184), success rate +0.140 (+0.171)**, field
position +0.060 (+0.070), 20+ yard plays -0.055 (-0.063; given success, explosiveness adds nothing),
3rd down -0.030 (-0.016), ypp -0.011 (-0.008), red-zone TD % -0.004 (-0.006). Per unit: +1 point of success
rate (0.01) = +0.040 logit; +0.01 turnovers a drive = -0.030 logit; +1 yard of drive start = +0.016 logit.
The margin-link model with strength puts its stat weight on red-zone trips a drive, ypp and win % (points of
margin per sd: +0.09, +0.12, +0.16 against s_kal's +1.94), which still does not move the price.
In the all-stats stats-only model (refitted on all rows since 05-04 to 10-19 for the table) the biggest weights are win % (+0.15 per sd), kal_pts (+0.13), first downs
(+0.07), streak (+0.06), third down (-0.05: the noise stat gets a negative, i.e. regressed, weight),
red-zone TD % (-0.05), success (+0.05), points a drive (+0.05).

## 6. Conclusions

1. The h2h site's stats, rebuilt from our play-by-play, carry real but redundant information: a stats-only
   model beats prod (TEST -0.006, interval clear of 0) but loses to a points-margin strength by 0.004.
2. **Strength + stats is no better than strength alone** (TEST +0.0000, -0.0007..+0.0007; every window and
   variant within +-0.001). Opponent adjustment and EB shrinkage don't change that; the margin-link and
   four-factor forms don't either.
3. The one place stats help is matches with a gamer under ~100 matches (5% of TEST): ~-0.004 to -0.012
   there, ~-0.0005 overall, not significant overall.
4. Red-zone TD %, 3rd-down % and 4th-down % are mostly noise at the site's windows (split-half r 0.2-0.3 at
   25 matches); turnovers and success rate are the efficiency stats that track winning, and both are
   already in the points.
5. By-product: my strength-only logistic on (rating_margin, a full-covariance Kalman of points with gamer
   off/def and team effects, `s_kal`) is the best thing here: TEST 0.6743, -0.0105 (-0.0151, -0.0061) vs
   prod, slope 0.97; SEP 0.6781, level with walkthrough glmer_both (+0.0007, -0.0016..+0.0031). s_kal alone
   (TUNE 0.6639, TEST 0.6742) beats rating_margin alone (0.6704, 0.6833) by a lot: rating_margin is overspread (slope 0.77 on
   TEST alone), s_kal drifts far less (q = 0.005 V a day vs features' 4 points^2 a day).

Caveats: TUNE starts 5 weeks after play-by-play does (May only half covered), so the earliest stat states
are thin; Kalman/EB variance components and q were set on 05-01..07-31 (TUNE included, TEST not). The
new-gamer finding was spotted on TEST first, then confirmed on TUNE, and the newgamer model's settings
were chosen on TUNE; still, treat it as exploratory.

Files: scripts/C/{common,repeat,kalman,eb,models,more,newgamer}.py; out/C/{repeatability,final,more,
newgamer}.txt, tune_grid.csv (TUNE only), addone.csv, picks.json, kalman_{vc,q}.json, kalman_feats.pkl
(per-match Kalman states; `k_ptsall_pred_h - k_ptsall_pred_a` = s_kal), eb_feats_hl*.pkl;
preds/C_{strength,stats,stats_site,stats_strength,four_factors,four_factors_strength,stats_mlink,
stats_strength_mlink,stats_strength_newgamer}.csv (12,119 matches, 06-01..10-18 each).
