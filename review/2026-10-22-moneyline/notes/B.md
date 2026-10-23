# Agent B: win/loss-only baselines (the Bradley-Terry family)

The user's check: *"Compare it with a plain Bradley-Terry model as a check. If the margin-plus-link model
doesn't beat it on held-out log loss, the margin isn't adding much."* B built the strongest honest
results-only models A's margin-plus-link model has to beat, plus margin-weighted versions as a bridge.

Everything follows PROTOCOL.md: walk-forward only (static fits through `harness.walk_forward`, daily refits on
matches settled by the day's midnight; online ratings take a result in 36 minutes after its kick-off), every
hyper-parameter chosen on TUNE (06-01..07-31) log loss and frozen, reported on TEST / SEP / 09-22..10-18,
`prod` only as a benchmark, ties out, comparisons on common matches with the day-resampled 95% interval.
No network, BLAS single-threaded, at most 2-3 single-threaded jobs, R single-threaded.

Code: `scripts/B/` (`btlib.py` static BT, `dyn.py` online ratings, `glmm_wf.R` lme4, `amfelo.py`,
`elo_leak.py`, `eval_B.py`). Outputs: `out/B/` (`eval_B.txt` is the full table set, `variants.txt` every
variant, `tune_*.csv|json` the tuning). Preds (06-01..10-18): `preds/B_*.csv`.

## Headline

* **The plain BT the user asked for** (`B_bt`: home + gamer, L2, 30-day half-life, daily refit) beats prod on
  TEST by 0.0049 (95% -0.0088,-0.0011), log loss 0.6791 vs 0.6840 (5,805 common matches), slope 0.98.
  Adding team strength + stream home (+ pair; `B_bt_team`) takes it to 0.6763 (-0.0076, -0.0121,-0.0033).
* **The strongest results-only model is dynamic**: a Glicko-style logistic Kalman filter that takes each
  result in 36 min after kick-off (`B_glicko`, gamers + fast intraday form + teams + stream home + pairs):
  TEST 0.6740, -0.0099 vs prod (-0.0153,-0.0049), slope 0.89; SEP -0.0084 (-0.0169,-0.0008);
  09-22..10-18 -0.0043 (-0.0137,+0.0049). A tuned Elo with a new-gamer K and team ratings
  (`B_elo_tuned`) is level with it (TEST -0.0102 vs prod; less B_glicko -0.0003 (-0.0021,+0.0016)).
  The dynamic models beat the daily-refit static BT mainly through same-day updating: the same Kalman fed
  only results settled by midnight loses 0.0032 (+0.0008,+0.0056) on TEST and 0.009 on TUNE.
* **So A's margin-plus-link model should be judged against B_bt (the literal check, TEST 0.6791) and,
  more demandingly, B_glicko / B_elo_tuned (TEST ~0.6738).** A model priced from a daily refit should be
  compared first with B_bt_team / B_glmm (same information cadence), then with B_glicko.
* **The margin does add something inside each results-only family** (bridge, reported separately): weighting
  results by the margin improves log loss by 0.0004-0.0017 on TEST and 0.0014-0.0028 on SEP (static BT
  -0.0017 (-0.0030,-0.0004) TEST; Kalman -0.0011 (-0.0025,+0.0001) TEST, -0.0021 (-0.0040,-0.0003) SEP;
  `out/B/mov_pairs.txt`). Best bridge: `B_mov_kal` TEST 0.6731 (-0.0109 vs prod), `B_mov_elo` 0.6735
  (slope 1.03).
* The binomial GLMM (lme4, difference design, variances by Laplace each day) is no better than the grid-tuned
  BT (TEST 0.6776, -0.0064 vs prod) and falls apart after 09-22 (0.6920, AUC 0.53): its marginal likelihood
  shrinks the gamer sd from ~0.75 to ~0.1 in October (verified by profiling the deviance; the spread of
  gamer win rates beyond binomial noise falls from 0.06-0.10 in Jun-Jul to 0.03-0.05 in Sep-Oct).
* AMFELO is not leaking (sequential, never contains its own result, no previous match within 36 min), but
  it is a weak Elo: raw +0.0056 vs prod on TEST (slope 0.53), with a fitted link -0.0041 (-0.0084,+0.0006).
* The online models tuned on Jun-Jul are a little too spread out from September (slopes 0.82-0.87 on SEP
  and 09-22..10-18; the static BT 0.86-0.97; prod 0.73 on SEP, 0.92 late): the league's skill spread
  compressed. In October alone prod (0.6824) is ahead of every win/loss-only model (B_glicko 0.6839,
  1,095 matches); only the MOV versions stay ahead (0.6793-0.6804).

## 1. Models and frozen settings

All on the logit scale: logit P(home wins) = home + s[home gamer] - s[away gamer] (+ team, pair ...).

### Static penalised BT (`B_bt`, `B_bt_team`), daily refits with `harness.walk_forward`
Weighted logistic likelihood, weights 0.5^(age/half-life), Gaussian (L2) penalty per block, fitted by Newton
on a sparse design (the dense Hessian is at most ~1,000 x 1,000; the pair model uses a sparse solve), warm
started from the previous day (scipy L-BFGS with the analytic gradient gives the same fit to 1e-3 but was
~15x slower on the shared cores). Grid on TUNE (61 daily refits per setting, ~80 settings).

| variant | TUNE ll | setting |
|---|---|---|
| gamer only, one home (**B_bt**, "plain BT") | 0.6688 | half-life 30 d, lambda_gamer 0.75 |
| + stream-specific home | 0.6681 | hl 30, lambda 0.5 |
| + team strength | 0.6649 | hl 25, lambda_g 0.5, lambda_team 3 (team lambda 1-10 all the same) |
| + gamer x team (shrunk toward gamer) | 0.6650 | lambda_gt 100 (smaller lambda hurts: 10 -> 0.6668, 1 -> 0.6841) |
| + gamer-specific home | 0.6649 | lambda 30 |
| + head-to-head pair effect (**B_bt_team**) | 0.6642 | hl 25, lambda_pair 10 |
| MOV-weighted, team (**B_bt_mov**, bridge) | 0.6633 | weight x (abs(margin)+1)^0.5, hl 30 |

Half-life is the main knob: no decay 0.6755, 270 d 0.6735, 90 d 0.6710, 30 d 0.6688, 15 d 0.6719 (gamer
only). Gamer strength drifts a lot month to month.

### Online ratings (dynamic BT): every result in 36 min after its kick-off
Tuned by Nelder-Mead on TUNE log loss (the filter runs from 2025-12-10; only Jun-Jul is scored).

* **Elo** (logit scale, `B_elo_tuned` = `elo_team_sh`): K(n) = K + Knew/(1 + n/nc) for a gamer's n-th game,
  team ratings (Kt), stream-specific home learned online (Kh). K 0.046, Knew 1.75, nc 1.8 (a new gamer's
  first results move him ~1.7 logits, by game 20 K ~ 0.2), Kt 0.0036, Kh 0.003. TUNE 0.6605.
  Plain Elo (one K, fixed home) is far worse: K 0.069, TUNE 0.6727; the new-gamer K alone takes it to 0.6658.
* **Glicko-style logistic Kalman filter** (EKF / one-step Laplace update, `B_glicko` = `kal3`): per gamer a
  slow strength (random walk, q 0.0022/day, new-gamer variance 3.4) plus a fast "form" state
  (Ornstein-Uhlenbeck, stationary sd 0.26, e-folding 0.30 days ~ 7 h), 2x2 covariance per gamer; team states
  (near-static), stream-specific home states, head-to-head pair states (sd 0.19, slow drift). Prices
  integrate the uncertainty, p = sigmoid(z / sqrt(1 + pi s / 8)). TUNE 0.6592.
  The ladder on TUNE: gamers only 0.6659 -> + teams 0.6605 -> + stream home 0.6600 / + pair 0.6596 /
  + fast form 0.6600 -> all (kal3) 0.6592. A full-covariance EKF over gamers + teams (0.6603) is no better
  than the diagonal one; not integrating (0.6608) or a probit-moment update (0.6609) slightly worse.
* MOV bridges (report separately; they use the margin): **B_mov_elo** = Elo with update x
  ((abs(m)+0.13)/(7.13))^0.68 / (1 + 0.55 * winner's pre-match edge) (538-style autocorrelation damping):
  TUNE 0.6583. **B_mov_kal** = kal3 with the observation information scaled by ((abs(m)+1.6)/8.6)^0.52:
  TUNE 0.6574.

### Binomial GLMM, difference design (`B_glmm`), lme4, **daily** refits
`glmer` through the modular functions (`glFormula` -> replace `Zt` by the +1 home / -1 away difference
matrix -> `mkGlmerDevfun` -> `optimizeGlmer` (nAGQ=0 then Laplace) -> `mkMerMod`); logit = stream home
(fixed) + u[g1] - u[g2] [+ t[k1] - t[k2]] [+ v[g1,k1] - v[g2,k2]], variances by Laplace marginal likelihood
each day (so the penalty is chosen by the data, not the grid). Time decay through prior weights
0.5^(age/hl) scaled to sum to Kish's effective sample size. Same cut as the harness (settled by midnight).

Daily refits over 06-01..10-18 (one R process, 2-18 s per fit), chosen by TUNE log loss:

| GLMM | TUNE ll | mean daily sd (gamer, team) |
|---|---|---|
| gamer + team, hl 30 (**B_glmm**) | 0.6652 | 0.77, 0.09 |
| gamer only, hl 30 | 0.6681 | 0.64 |
| gamer only, hl 60 | 0.6692 | 0.84 |
| gamer only, last 90 days unweighted | 0.6697 | 0.84 |

Marginal-likelihood shrinkage matches the hand-tuned grid on TUNE (gamer-only 0.6681 = BT 0.6681 with
stream home; gamer + team 0.6652 vs BT 0.6649). A first pass with weekly refits (and mean-1 weights)
scored 0.6759-0.6818 on TUNE for gamer-only, so the cadence matters: daily it is. Gamer x team (`gtx`, a third difference-coded random effect
v[g,k]): its variance is estimated at exactly 0 on every day checked (06-01..06-05, 07-15, 07-31), so the
model collapses to gamer + team and was not run further (the static BT agrees: lambda_gt 100 best, i.e. no
interaction).
The gamer sd collapses to ~0.1 from late September (deviance profile at 10-09: theta 0.1 -> 7630.8,
0.45 -> 7662.7, 0.8 -> 7689.8), which is why B_glmm loses its signal after 09-22.

### AMFELO (`B_amfelo_raw`, `B_amfelo_link`)
Raw: p = 1/(1 + 10^(-(elo1 - elo2)/400)). Link: logistic a + b * diff, refit daily on the last 60 days
(window chosen on TUNE from all/14/30/60/120). The link is a ~0.03-0.06, b ~0.55-0.60: the file's
ratings are ~1.7x too spread and carry no home edge.

## 2. AMFELO leakage audit (`out/B/elo_leak.txt`, `out/B/amfelo_b2b.txt`)
* Sequential: a gamer's starting rating equals his previous ending in 98.1% of 44,359 gamer-matches (99%+
  from June). The 854 exceptions are jumps (Jan 28-29, early March and May bursts; median 0, sd 29) that look
  like matches missing from our data (AMFELO has 26,264 matches, 400 of ours are not in it) and 234 cases
  of the previous update applied twice. Jumps do not predict a gamer's next 30 results (corr -0.04), so no
  hindsight re-rating.
* Never contains its own result: only 8 starts equal the same match's ending (4 matches with a zero update,
  both sides); starts equal to "previous end + this match's change" are rare (16 of the 854 jumps, about
  what coincidence of small integer updates gives) and the jumps are uncorrelated with the match's own
  result (corr 0.04) and its own update (corr 0.005).
* Timing: the gap between a gamer's consecutive kick-offs is never below 36 minutes (the schedule's own
  minimum; 25% are exactly 36), so every starting rating uses only results settled 36+ min before the
  kick-off: no match has a "previous match < 36 min before". On back-to-back matches (previous kick-off
  exactly 36 min before) AMFELO is not relatively better: AMFELO link less B_glicko is +0.0088
  (+0.0054,+0.0123) on back-to-back vs +0.0064 (+0.0027,+0.0103) otherwise (Jun-Oct).
* One caveat: new gamers' first ratings are seeded (825 to 1643), and among the 20 gamers who arrived after
  mid-January the seed correlates 0.55 with the eventual win rate. Either outside history (another league /
  the h2h site) or hindsight; the file cannot say. It helps AMFELO raw a little on <30-game matches but our own
  Kalman (seeded at 0) still does better there.
* K ~ 16-17 per 400 (median), rising with the margin (16.7 at |m|<=3 to 20 at |m|>28): a mild MOV Elo,
  no home term. As a predictor it is simply a weaker Elo (too spread, no home, no teams).

## 3. Results

Common-match tables from `harness.report` (refs prod and B_glicko), then log loss by month and by experience
bucket (fewer games of the two gamers before the match). TUNE rows are in-sample for the hyper-parameters
(shown for transparency only). Full set: `out/B/eval_B.txt`; every variant: `out/B/variants.txt`.

```
### TUNE (in-sample for the hyper-parameters)
2026-06-01 to 2026-08-01: 5,456 matches, 60 days (priced by every column)
  model                         logloss   brier    auc  slope    int   vs       prod (95%)          vs   B_glicko (95%)       
  B_glicko                       0.6620  0.2347  0.644   0.89  -0.01   -0.0143 (-0.0216,-0.0079)                              
  B_amfelo_link                  0.6697  0.2385  0.623   0.99  -0.03   -0.0066 (-0.0119,-0.0020)   +0.0077 (+0.0039,+0.0116)
  B_amfelo_raw                   0.6814  0.2428  0.623   0.57  +0.00   +0.0050 (-0.0036,+0.0131)   +0.0194 (+0.0141,+0.0245)
  B_bt                           0.6713  0.2392  0.619   0.91  -0.03   -0.0051 (-0.0103,-0.0002)   +0.0092 (+0.0045,+0.0139)
  B_bt_mov                       0.6660  0.2365  0.636   0.86  -0.05   -0.0104 (-0.0169,-0.0043)   +0.0039 (-0.0002,+0.0082)
  B_bt_team                      0.6668  0.2370  0.632   0.94  -0.03   -0.0095 (-0.0154,-0.0042)   +0.0048 (+0.0010,+0.0088)
  B_elo_tuned                    0.6632  0.2352  0.642   0.90  -0.02   -0.0132 (-0.0196,-0.0073)   +0.0012 (-0.0009,+0.0032)
  B_glmm                         0.6678  0.2375  0.629   0.94  -0.02   -0.0085 (-0.0143,-0.0033)   +0.0058 (+0.0016,+0.0101)
  B_mov_elo                      0.6604  0.2340  0.647   0.98  -0.01   -0.0160 (-0.0220,-0.0107)   -0.0017 (-0.0042,+0.0010)
  B_mov_kal                      0.6599  0.2337  0.648   0.88  -0.01   -0.0165 (-0.0237,-0.0102)   -0.0022 (-0.0034,-0.0010)
  prod                           0.6764  0.2418  0.605   1.13  -0.00                                 +0.0143 (+0.0079,+0.0216)

### TEST
2026-08-01 to 2026-10-19: 5,805 matches, 79 days (priced by every column)
  model                         logloss   brier    auc  slope    int   vs       prod (95%)          vs   B_glicko (95%)       
  B_glicko                       0.6740  0.2407  0.607   0.89  +0.04   -0.0099 (-0.0153,-0.0049)                              
  B_amfelo_link                  0.6799  0.2435  0.585   0.90  +0.03   -0.0041 (-0.0084,+0.0006)   +0.0059 (+0.0022,+0.0093)
  B_amfelo_raw                   0.6895  0.2477  0.586   0.53  +0.06   +0.0056 (-0.0015,+0.0129)   +0.0155 (+0.0104,+0.0206)
  B_bt                           0.6791  0.2431  0.588   0.98  +0.03   -0.0049 (-0.0088,-0.0011)   +0.0050 (+0.0013,+0.0086)
  B_bt_mov                       0.6737  0.2405  0.607   0.95  +0.01   -0.0103 (-0.0147,-0.0060)   -0.0003 (-0.0032,+0.0026)
  B_bt_team                      0.6763  0.2418  0.599   0.98  +0.02   -0.0076 (-0.0121,-0.0033)   +0.0023 (-0.0004,+0.0049)
  B_elo_tuned                    0.6737  0.2406  0.605   0.93  +0.02   -0.0102 (-0.0148,-0.0055)   -0.0003 (-0.0021,+0.0016)
  B_glmm                         0.6776  0.2424  0.591   1.03  +0.02   -0.0064 (-0.0108,-0.0021)   +0.0036 (+0.0001,+0.0071)
  B_mov_elo                      0.6735  0.2405  0.607   1.03  +0.04   -0.0105 (-0.0147,-0.0062)   -0.0005 (-0.0031,+0.0020)
  B_mov_kal                      0.6731  0.2403  0.610   0.85  +0.03   -0.0109 (-0.0162,-0.0058)   -0.0010 (-0.0024,+0.0004)
  prod                           0.6840  0.2456  0.573   0.97  +0.06                                 +0.0099 (+0.0049,+0.0153)

### SEP
2026-09-01 to 2026-10-19: 3,061 matches, 48 days (priced by every column)
  model                         logloss   brier    auc  slope    int   vs       prod (95%)          vs   B_glicko (95%)       
  B_glicko                       0.6800  0.2436  0.587   0.83  +0.02   -0.0084 (-0.0169,-0.0008)                              
  B_amfelo_link                  0.6844  0.2458  0.565   0.83  +0.00   -0.0040 (-0.0105,+0.0023)   +0.0044 (-0.0004,+0.0091)
  B_amfelo_raw                   0.6939  0.2501  0.565   0.48  +0.04   +0.0055 (-0.0042,+0.0147)   +0.0139 (+0.0076,+0.0198)
  B_bt                           0.6846  0.2458  0.567   0.88  +0.01   -0.0039 (-0.0094,+0.0011)   +0.0045 (-0.0006,+0.0096)
  B_bt_mov                       0.6793  0.2433  0.588   0.92  -0.02   -0.0091 (-0.0154,-0.0031)   -0.0007 (-0.0049,+0.0038)
  B_bt_team                      0.6823  0.2447  0.578   0.90  +0.00   -0.0061 (-0.0130,+0.0001)   +0.0023 (-0.0013,+0.0059)
  B_elo_tuned                    0.6799  0.2436  0.585   0.87  -0.00   -0.0085 (-0.0158,-0.0018)   -0.0001 (-0.0028,+0.0026)
  B_glmm                         0.6862  0.2466  0.558   0.88  -0.00   -0.0022 (-0.0085,+0.0039)   +0.0062 (+0.0006,+0.0117)
  B_mov_elo                      0.6777  0.2425  0.592   1.04  +0.03   -0.0107 (-0.0172,-0.0049)   -0.0023 (-0.0061,+0.0014)
  B_mov_kal                      0.6783  0.2428  0.594   0.82  +0.01   -0.0101 (-0.0183,-0.0025)   -0.0017 (-0.0037,+0.0003)
  prod                           0.6884  0.2477  0.554   0.73  +0.04                                 +0.0084 (+0.0008,+0.0169)

### LATE 09-22..10-18
2026-09-22 to 2026-10-19: 1,475 matches, 27 days (priced by every column)
  model                         logloss   brier    auc  slope    int   vs       prod (95%)          vs   B_glicko (95%)       
  B_glicko                       0.6813  0.2442  0.585   0.82  +0.01   -0.0043 (-0.0137,+0.0049)                              
  B_amfelo_link                  0.6872  0.2471  0.559   0.74  +0.00   +0.0016 (-0.0061,+0.0096)   +0.0059 (-0.0019,+0.0130)
  B_amfelo_raw                   0.6998  0.2528  0.558   0.40  +0.04   +0.0142 (+0.0029,+0.0265)   +0.0185 (+0.0088,+0.0277)
  B_bt                           0.6860  0.2465  0.567   0.86  +0.01   +0.0004 (-0.0049,+0.0056)   +0.0047 (-0.0024,+0.0120)
  B_bt_mov                       0.6807  0.2439  0.583   0.97  -0.01   -0.0049 (-0.0104,+0.0007)   -0.0006 (-0.0065,+0.0055)
  B_bt_team                      0.6826  0.2449  0.576   0.97  +0.00   -0.0030 (-0.0099,+0.0042)   +0.0013 (-0.0034,+0.0064)
  B_elo_tuned                    0.6816  0.2444  0.581   0.85  -0.00   -0.0040 (-0.0121,+0.0045)   +0.0003 (-0.0020,+0.0028)
  B_glmm                         0.6920  0.2494  0.525   0.65  +0.01   +0.0064 (+0.0007,+0.0116)   +0.0107 (+0.0025,+0.0188)
  B_mov_elo                      0.6784  0.2428  0.592   1.06  +0.03   -0.0071 (-0.0150,+0.0004)   -0.0029 (-0.0082,+0.0021)
  B_mov_kal                      0.6795  0.2433  0.591   0.82  +0.01   -0.0061 (-0.0147,+0.0029)   -0.0018 (-0.0053,+0.0016)
  prod                           0.6856  0.2463  0.566   0.92  +0.04                                 +0.0043 (-0.0049,+0.0137)

### TEST, each column on its own priced matches (not common)
2026-08-01 to 2026-10-19: 6,311 matches, 79 days
  model                         logloss   brier    auc  slope    int   vs       prod (95%)       
  B_glicko                       0.6739  0.2406  0.607   0.88  +0.03   -0.0096 (-0.0148,-0.0045)
  B_amfelo_link                  0.6788  0.2430  0.589   0.93  +0.03   -0.0041 (-0.0084,+0.0006)
  B_amfelo_raw                   0.6878  0.2469  0.589   0.55  +0.06   +0.0056 (-0.0015,+0.0129)
  B_bt                           0.6786  0.2429  0.589   0.97  +0.02   -0.0044 (-0.0082,-0.0008)
  B_bt_mov                       0.6735  0.2405  0.607   0.93  +0.00   -0.0099 (-0.0141,-0.0059)
  B_bt_team                      0.6760  0.2416  0.599   0.96  +0.02   -0.0074 (-0.0118,-0.0032)
  B_elo_tuned                    0.6737  0.2406  0.605   0.93  +0.02   -0.0097 (-0.0143,-0.0051)
  B_glmm                         0.6772  0.2422  0.592   1.02  +0.01   -0.0061 (-0.0104,-0.0020)
  B_mov_elo                      0.6730  0.2402  0.608   1.03  +0.04   -0.0104 (-0.0147,-0.0063)
  B_mov_kal                      0.6727  0.2401  0.611   0.85  +0.03   -0.0108 (-0.0161,-0.0057)
  prod                           0.6848  0.2460  0.569   0.93  +0.05                              

### log loss by month, 2026-06-01..10-18 (common matches with prod)
month            2026-06    2026-07    2026-08    2026-09    2026-10
B_glicko          0.6629     0.6612     0.6674     0.6778     0.6839
B_amfelo_link     0.6735     0.6663     0.6749     0.6834     0.6862
B_amfelo_raw      0.6848     0.6783     0.6847     0.6903     0.7004
B_bt              0.6729     0.6698     0.6729     0.6845     0.6847
B_bt_mov          0.6701     0.6622     0.6675     0.6790     0.6799
B_bt_team         0.6698     0.6641     0.6697     0.6820     0.6828
B_elo_tuned       0.6635     0.6629     0.6669     0.6781     0.6832
B_glmm            0.6706     0.6653     0.6680     0.6830     0.6920
B_mov_elo         0.6617     0.6591     0.6689     0.6767     0.6793
B_mov_kal         0.6619     0.6579     0.6673     0.6771     0.6804
prod              0.6805     0.6726     0.6790     0.6918     0.6824
n              2621.0000  2835.0000  2744.0000  1966.0000  1095.0000

### log loss by experience bucket (fewer games of the two gamers), Jun-Oct
exp               30-200       <30       >200
B_glicko          0.6579    0.6452     0.6708
B_amfelo_link     0.6635    0.6531     0.6777
B_amfelo_raw      0.6695    0.6503     0.6896
B_bt              0.6665    0.6746     0.6768
B_bt_mov          0.6588    0.7013     0.6707
B_bt_team         0.6606    0.6916     0.6729
B_elo_tuned       0.6595    0.6534     0.6707
B_glmm            0.6614    0.6875     0.6743
B_mov_elo         0.6559    0.6383     0.6701
B_mov_kal         0.6573    0.6388     0.6693
prod              0.6747    0.6638     0.6818
n              1573.0000  339.0000  9349.0000

### log loss by experience bucket (fewer games of the two gamers), TUNE
exp              30-200       <30       >200
B_glicko         0.6422    0.6452     0.6678
B_amfelo_link    0.6534    0.6531     0.6747
B_amfelo_raw     0.6559    0.6503     0.6895
B_bt             0.6567    0.6746     0.6742
B_bt_mov         0.6458    0.7013     0.6676
B_bt_team        0.6460    0.6916     0.6695
B_elo_tuned      0.6446    0.6534     0.6681
B_glmm           0.6484    0.6875     0.6705
B_mov_elo        0.6439    0.6383     0.6658
B_mov_kal        0.6407    0.6388     0.6658
prod             0.6683    0.6638     0.6791
n              927.0000  339.0000  4190.0000

### log loss by experience bucket (fewer games of the two gamers), TEST
exp              30-200       >200
B_glicko         0.6805     0.6732
B_amfelo_link    0.6781     0.6801
B_amfelo_raw     0.6891     0.6896
B_bt             0.6805     0.6789
B_bt_mov         0.6776     0.6732
B_bt_team        0.6815     0.6757
B_elo_tuned      0.6810     0.6728
B_glmm           0.6800     0.6773
B_mov_elo        0.6731     0.6736
B_mov_kal        0.6810     0.6721
prod             0.6838     0.6840
n              646.0000  5159.0000
```

### What the margin adds within each family (bridge; `out/B/mov_pairs.txt`)
```
MOV version less win/loss twin, log loss (day-resampled 95%); negative = margin helps
TUNE:
  static BT (+team, stream home), MOV weights          -0.0016 (-0.0031,-0.0002)
  Elo (+team, new-gamer K), MOV multiplier             -0.0015 (-0.0030,-0.0002)
  Elo, MOV multiplier + autocorrelation damping        -0.0029 (-0.0053,-0.0007)
  Kalman (+team), MOV-scaled information               -0.0016 (-0.0030,-0.0004)
  Kalman kal3, MOV-scaled information                  -0.0018 (-0.0031,-0.0007)
TEST:
  static BT (+team, stream home), MOV weights          -0.0017 (-0.0030,-0.0004)
  Elo (+team, new-gamer K), MOV multiplier             -0.0004 (-0.0017,+0.0008)
  Elo, MOV multiplier + autocorrelation damping        -0.0014 (-0.0032,+0.0005)
  Kalman (+team), MOV-scaled information               -0.0007 (-0.0021,+0.0005)
  Kalman kal3, MOV-scaled information                  -0.0011 (-0.0025,+0.0001)
SEP:
  static BT (+team, stream home), MOV weights          -0.0024 (-0.0041,-0.0006)
  Elo (+team, new-gamer K), MOV multiplier             -0.0014 (-0.0031,+0.0004)
  Elo, MOV multiplier + autocorrelation damping        -0.0028 (-0.0054,-0.0001)
  Kalman (+team), MOV-scaled information               -0.0018 (-0.0035,+0.0001)
  Kalman kal3, MOV-scaled information                  -0.0021 (-0.0040,-0.0003)
LATE:
  static BT (+team, stream home), MOV weights          -0.0024 (-0.0049,+0.0000)
  Elo (+team, new-gamer K), MOV multiplier             -0.0013 (-0.0042,+0.0014)
  Elo, MOV multiplier + autocorrelation damping        -0.0034 (-0.0075,+0.0003)
  Kalman (+team), MOV-scaled information               -0.0020 (-0.0048,+0.0005)
  Kalman kal3, MOV-scaled information                  -0.0025 (-0.0055,+0.0003)
```

### Every variant on TEST (6,170 common matches; `__midnight` = same frozen model fed only results settled
by the day's midnight, i.e. the static fits' information)
```
2026-08-01 to 2026-10-19: 6,170 matches, 79 days (priced by every column)
  model                         logloss   brier    auc  slope    int   vs       prod (95%)          vs       kal3 (95%)       
  kal3                           0.6751  0.2412  0.604   0.87  +0.03   -0.0096 (-0.0148,-0.0047)                              
  bt                             0.6803  0.2437  0.584   0.95  +0.02   -0.0044 (-0.0079,-0.0009)   +0.0052 (+0.0015,+0.0090)
  bt_stream                      0.6795  0.2433  0.587   0.90  +0.02   -0.0052 (-0.0092,-0.0011)   +0.0044 (+0.0007,+0.0082)
  bt_team                        0.6766  0.2419  0.597   1.00  +0.02   -0.0081 (-0.0120,-0.0042)   +0.0015 (-0.0013,+0.0043)
  bt_team_gt                     0.6757  0.2415  0.600   1.03  +0.01   -0.0091 (-0.0129,-0.0053)   +0.0005 (-0.0023,+0.0034)
  bt_team_ghome                  0.6759  0.2416  0.599   1.02  +0.01   -0.0088 (-0.0128,-0.0049)   +0.0008 (-0.0019,+0.0036)
  bt_team_pair                   0.6774  0.2423  0.596   0.96  +0.02   -0.0074 (-0.0115,-0.0033)   +0.0022 (-0.0004,+0.0049)
  bt_mov                         0.6749  0.2411  0.604   0.93  +0.00   -0.0099 (-0.0138,-0.0058)   -0.0003 (-0.0032,+0.0026)
  mov_kal3                       0.6739  0.2407  0.608   0.85  +0.03   -0.0108 (-0.0158,-0.0058)   -0.0012 (-0.0025,+0.0002)
  mov_kal3__midnight             0.6775  0.2424  0.601   0.73  +0.03   -0.0072 (-0.0126,-0.0015)   +0.0024 (+0.0001,+0.0047)
  elo                            0.6819  0.2445  0.582   0.80  +0.04   -0.0029 (-0.0073,+0.0017)   +0.0068 (+0.0031,+0.0101)
  elo_new                        0.6800  0.2436  0.587   0.84  +0.04   -0.0048 (-0.0088,-0.0004)   +0.0048 (+0.0012,+0.0083)
  elo_team                       0.6758  0.2416  0.600   0.92  +0.04   -0.0090 (-0.0133,-0.0045)   +0.0007 (-0.0012,+0.0022)
  elo_team_gt                    0.6758  0.2416  0.601   0.92  +0.04   -0.0090 (-0.0134,-0.0045)   +0.0006 (-0.0012,+0.0023)
  elo_team_sh                    0.6751  0.2412  0.602   0.91  +0.02   -0.0097 (-0.0142,-0.0051)   -0.0001 (-0.0018,+0.0018)
  elo_team_sh__midnight          0.6784  0.2428  0.596   0.77  +0.02   -0.0064 (-0.0114,-0.0012)   +0.0033 (+0.0006,+0.0061)
  elo_team_tau                   0.6758  0.2416  0.600   0.94  +0.04   -0.0090 (-0.0132,-0.0046)   +0.0007 (-0.0012,+0.0022)
  kal                            0.6799  0.2435  0.588   0.82  +0.04   -0.0049 (-0.0090,-0.0005)   +0.0047 (+0.0014,+0.0080)
  kal_r0                         0.6799  0.2435  0.588   0.82  +0.04   -0.0049 (-0.0089,-0.0005)   +0.0047 (+0.0014,+0.0080)
  kal_team                       0.6757  0.2415  0.601   0.90  +0.04   -0.0091 (-0.0136,-0.0047)   +0.0005 (-0.0010,+0.0021)
  kal_team__midnight             0.6782  0.2428  0.595   0.78  +0.04   -0.0065 (-0.0117,-0.0014)   +0.0031 (+0.0004,+0.0057)
  kal_team_gt                    0.6756  0.2415  0.602   0.89  +0.04   -0.0091 (-0.0137,-0.0046)   +0.0005 (-0.0012,+0.0021)
  kal_team_sh                    0.6749  0.2411  0.603   0.90  +0.02   -0.0099 (-0.0144,-0.0053)   -0.0002 (-0.0016,+0.0013)
  kal_team_noint                 0.6756  0.2415  0.601   0.89  +0.04   -0.0092 (-0.0137,-0.0048)   +0.0004 (-0.0011,+0.0020)
  kal_team_probit                0.6757  0.2415  0.601   0.88  +0.04   -0.0091 (-0.0137,-0.0047)   +0.0006 (-0.0010,+0.0021)
  kal_team_pair                  0.6757  0.2415  0.602   0.88  +0.04   -0.0091 (-0.0142,-0.0043)   +0.0006 (-0.0003,+0.0014)
  kal2                           0.6753  0.2413  0.603   0.90  +0.04   -0.0094 (-0.0140,-0.0049)   +0.0002 (-0.0012,+0.0014)
  kal3__midnight                 0.6783  0.2428  0.595   0.77  +0.03   -0.0065 (-0.0119,-0.0007)   +0.0032 (+0.0008,+0.0056)
  kfull                          0.6750  0.2412  0.603   0.91  +0.04   -0.0098 (-0.0143,-0.0054)   -0.0002 (-0.0018,+0.0014)
  mov_elo                        0.6753  0.2413  0.602   0.87  +0.04   -0.0094 (-0.0139,-0.0052)   +0.0002 (-0.0018,+0.0021)
  mov_elo_ac                     0.6744  0.2409  0.605   1.02  +0.04   -0.0104 (-0.0144,-0.0063)   -0.0008 (-0.0033,+0.0015)
  mov_kal_team                   0.6749  0.2411  0.605   0.85  +0.04   -0.0099 (-0.0146,-0.0054)   -0.0003 (-0.0022,+0.0015)
  mov_kal_team__midnight         0.6781  0.2426  0.599   0.74  +0.04   -0.0067 (-0.0115,-0.0017)   +0.0029 (+0.0005,+0.0054)
  glmm_g_td0_hl60                0.6783  0.2428  0.592   0.93  +0.02   -0.0064 (-0.0101,-0.0021)   +0.0032 (-0.0005,+0.0070)
  glmm_gt_td0_hl30               0.6786  0.2429  0.588   1.01  +0.01   -0.0061 (-0.0101,-0.0019)   +0.0035 (+0.0001,+0.0070)
  glmm_g_td90_hl0                0.6795  0.2433  0.588   0.88  +0.02   -0.0052 (-0.0089,-0.0013)   +0.0044 (+0.0005,+0.0082)
  glmm_g_td0_hl30                0.6810  0.2441  0.578   0.96  +0.01   -0.0037 (-0.0076,+0.0003)   +0.0059 (+0.0020,+0.0098)
  prod                           0.6848  0.2460  0.569   0.93  +0.05                                 +0.0096 (+0.0047,+0.0148)
```

## 4. Reading the results

* Ladder on TEST (log loss less prod): plain BT -0.0044/-0.0049, + stream home -0.0052, + teams -0.0081,
  + gamer x team -0.0091 / + gamer home -0.0088 / + pairs -0.0074 (pairs won TUNE by 0.0007 and lost TEST by
  0.0008: noise level); GLMM -0.0061..-0.0064; online Elo / Kalman with teams -0.0090..-0.0099; kal3 -0.0096.
  Teams are the biggest single structural gain (~0.003 on both TUNE and TEST): gamers' results depend on the
  team picked, and a gamer-only BT mixes the two.
* Static BT vs dynamic: on matching information (midnight) the Kalman and the daily BT are close (TEST
  kal3__midnight 0.6783 vs bt_team 0.6766 -- the static fit is even a little better); the dynamic models win
  by using today's earlier results (gamers play ~7 a day, ~72 min apart). The Kalman's fast form state
  (sd 0.26, ~7 h e-folding) says same-session streaks are real.
* New gamers: BT struggles as expected (<30-game bucket, TUNE: B_bt 0.675, B_bt_team 0.692, B_glmm 0.688
  vs B_glicko 0.645, prod 0.664); the online models' large initial variance / new-gamer K fixes it.
  The bucket is almost empty after July (14 matches in TEST).
* Calibration: static BT and GLMM slopes ~0.95-1.03 on TEST; online models 0.85-0.93 (a little too spread)
  and 0.82-0.87 on SEP / late; mov_elo_ac is the exception at 1.02-1.06. Prod is 0.97 on TEST but 0.73 on SEP.
* By month all models degrade from August; prod degrades less in October (0.6824) and is level with or
  ahead of everything but the MOV models and B_bt_mov there.

## 5. Preds written (`preds/`, code, p_home; 2026-06-01..10-18, ~12,100 rows each)

| file | model | TUNE ll (selection) |
|---|---|---|
| B_bt | static BT, home + gamer, hl 30, lambda 0.75 | 0.6688 |
| B_bt_team | static BT + teams + stream home + pairs, hl 25 | 0.6642 |
| B_bt_mov | static BT + teams + stream home, MOV weights (bridge) | 0.6633 |
| B_glmm | lme4 GLMM gamer + team, hl 30, daily | 0.6652 |
| B_elo_tuned | Elo + teams + new-gamer K + stream home online | 0.6605 |
| B_glicko | logistic Kalman kal3 (best results-only) | 0.6592 |
| B_mov_elo | MOV Elo with autocorrelation damping (bridge) | 0.6583 |
| B_mov_kal | kal3 with MOV information (bridge) | 0.6574 |
| B_amfelo_raw | AMFELO starting ratings, raw Elo logistic | -- |
| B_amfelo_link | AMFELO diff with a 60-day walk-forward link | 0.6686 |

## 6. Caveats
* All online models run from 2025-12-10; their TUNE score is out-of-sample in time but the 6-13
  hyper-parameters were fit to it (Nelder-Mead), so TUNE is optimistic; TEST is the honest number.
* The static BT and GLMM refit once a day (harness cadence) and so cannot use same-day results; that is a
  real handicap here, not a modelling choice of BT itself.
* The new-gamer seeding in AMFELO is unexplained (could be outside history).


## 7. Cross-check against A (A_best.csv as written at 14:25; A may still change it)
Common matches, harness.report: TEST (6,311) A_best 0.6719 (slope 1.01) vs B_bt -0.0067 (-0.0104,-0.0031),
vs B_glicko -0.0019 (-0.0052,+0.0011); SEP (3,461) vs B_bt -0.0091 (-0.0137,-0.0044), vs B_glicko -0.0043
(-0.0086,-0.0000). So it passes the user's literal check (beats plain BT) clearly, and is ahead of the best
win/loss-only model by an amount the interval does not yet separate from zero on TEST.
