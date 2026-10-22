# Agent A: a margin model with a fitted link

The idea tested: keep gamer and team strengths fitted on margin, then fit P(home) = logistic(a + b·m + extras)
walk-forward on actual results, so the link learns the shrinkage that today sits in the means (the drift-4 form
rating's margins are too spread out; calibration slope about 0.4 on SEP).

## The short answer

1. **The link works as advertised, but what it fixes is small next to the choice of margin.**
   - Drift-4 rating: the link takes TEST log loss from 0.6963 to 0.6829 (-0.0134). That still trails a slow
     margin by +0.0086 (+0.0052,+0.0118).
   - Slow margins (my ridge, a Kalman at drift 0.05, the walkthrough's avg): these are already roughly
     calibrated. The link changes their log loss by 0.0002 to 0.0008, within noise.
   - The information that is lost goes in the means, not in the map. The drift-4 rating is about 16 times too
     fast: on TUNE the linked log loss falls steadily as the drift drops from 16 to 0.05 points²/day.
2. **The drift-4 rating's shrinkage is not stable.** The fitted b·9.35 (1 means the normal sd 9.35 is right)
   is 0.58 in June, 0.64-0.66 in August and 0.45 by mid-October, so the effective sd goes from 14-16 to 21
   points.
   - A 60-day link lags that drift. The linked rating's TEST slope is 0.79.
   - The slow margins' fitted b is steady: ridge effective sd 7.3 to 8.9, avg table 8.3 to 9.2.
3. **The form table (v2.1's prior today) is the problem the user describes, and a link does not cure it.**
   - On SEP the form table's own p is worse than avg's own p by +0.0050 (+0.0012,+0.0085).
   - Linking the form margin (on the extended tables) still leaves +0.0036 (+0.0002,+0.0068).
   - The results' own weighting of avg and the drift-4 rating, logistic(a + b1·avg + b2·rating), gives the
     rating 3-8% of avg's weight. form_prior gives it 50%.
4. **The extras add nothing once the margin is good.** On the ridge link, none of rematch, session form, form
   gap, stream, home, uncertainty or total clears zero on TUNE, and none on TEST either. Most estimates are
   within ±0.0004.
   - Two extras help only weak margins: the stream intercepts and the head-to-head margin help the
     Kalman / drift-4 margins, because those have no stream home edge and no pair memory.
5. **Best long-history model: the ridge link.** This is a decayed ridge on margin (gamer + team + home edge
   per stream, half-life 30 days, updated as results arrive) with a plain 60-day link.
   - TEST 0.6744 against prod 0.6848: -0.0104 (-0.0147,-0.0061).
   - SEP: -0.0105 against prod; +0.0012 (-0.0016,+0.0038) against tbl_avg_p.
6. **Best overall (A_best): logistic(a + b1·avg margin + b2·ridge margin)** where the avg table prices, else
   the ridge link. The two weights come out about equal (0.10 each).
   - SEP 0.6772: against prod -0.0122 (-0.0182,-0.0066); against tbl_avg_p -0.0005 (-0.0020,+0.0009).
   - TEST with the extended tables 0.6731: against prod -0.0117 (-0.0161,-0.0073); against tbl_avg_p
     -0.0003 (-0.0014,+0.0009).
   - It is level with the avg table's own moneyline, not better than it.

## Data, rules, coverage
- harness.load(). Every fit uses harness.walk_forward: a daily refit on matches settled 36 minutes after
  kick-off by the day's midnight UTC.
- The margin sources are online. Each match's margin uses results settled by its kick-off, exactly as
  xgb/features.py does. The re-implemented rating reproduces features.csv's rating_margin and its variances to
  1e-14.
- prod is never a feature.
- Settings chosen on TUNE (06-01..07-31) are frozen for TEST (08-01..10-18). Table-specific choices (the
  A_best stack) were made on 09-01..09-21 at most.
- Every comparison uses common matches and harness.report's day-resampled intervals.
- **Walkthrough extension.** The timing test (2 days, 1 worker / 1 core, idle machine) ran at about 33 s a
  day, so 99 days would have taken about 55 minutes.
  - The full run from 05-25 slowed to 70-150 s a day once the 4 shared cores reached load 15-25. I aborted it
    after 8 days (kept in out/A/walkthrough_may25_aborted).
  - I re-ran only **2026-07-25..08-31** (7 days of layer burn-in, then August): `python -m eAMFCalibrator
    walkthrough --since 2026-07-25 --until 2026-09-01 --workers 1 --cores 1 --no-pace`, BLAS at 2 threads, into
    out/A/walkthrough. It took 1h25m under the load.
  - These August tables are placed in front of the repo's Sep-Oct tables (scripts/A/tables.py). The form
    table's August margin is rebuilt exactly as form_prior builds it: avg + 0.5·(rating − avg), which matches
    the repo's Sep-Oct form table to within 0.04 points. Its own p is left blank there.
  - In August, tbl_avg_p beats prod by -0.0111 (-0.0175,-0.0051): the same picture as Sep-Oct.
  - The table links are trained from 07-25. Without the extension (training from 09-01, so the link has
    almost no data for its first days) the table links lose +0.0026 against tbl_avg_p on 09-01..09-21.

## 1. Margin sources (TUNE: plain link logistic(a+b·m), 120-day window, all 5,802 TUNE results)

**Kalman form rating**: features.py's filter re-run (the RATING_DRIFT / SIGMA / V0 / TEAM_RATE logic), TUNE log
loss after the link:

| drift \ sigma | 6 | 7 | 8 | 9 | 10 | 12 |
|---|---|---|---|---|---|---|
| 0.02 | | 0.6605 | 0.6611 | | | |
| 0.05 | 0.6599 | **0.6601** | 0.6605 | 0.6610 | 0.6615 | |
| 0.1 | | 0.6607 | 0.6608 | 0.6611 | 0.6615 | |
| 0.25 | | | 0.6622 | 0.6622 | 0.6623 | 0.6628 |
| 0.5 | | | | | 0.6636 | 0.6637 |
| 1 | | | | | 0.6655 | 0.6652 |
| 2 | | | | | 0.6680 | 0.6672 |
| 4 (features.csv) | | | | | 0.6710 | **0.6699** |
| 8 | | | | | 0.6744 | 0.6730 |
| 16 | | | | | 0.6776 | 0.6762 |

The optimum is flat (drift 0.02-0.1, sigma 6-8, all within 0.0006). I chose kal_d0.05_s7 at the middle of it.
An online home edge (home_rate 0.002) changed nothing (±0.0001).

**Decayed ridge on margin** (common.ridge_live):
- Model: margin = home + home_stream + g_home − g_away + t_home − t_away. Each settled result is weighted
  2^(−age/half-life). Penalties are in matches of full weight; there are 89 gamers and 10 teams.
- The system is re-solved whenever a result settles (36 minutes after kick-off). A once-a-day (midnight)
  version is consistently 0.003-0.004 worse: ridge_hl30_g2 0.6639 daily against 0.6601 live (TUNE).

Top of the grid (half-life 7-240 days, gamer penalty 1-40, team penalty 10-200):

| ridge | TUNE ll |
|---|---|
| hl30, g2, t50, **home edge per stream** (chosen) | **0.6591** |
| hl20, g1, t50 | 0.6597 |
| hl30, g2, t50, home edge per gamer | 0.6599 |
| hl30, g2, t50 | 0.6601 |
| hl45, g2, t50 | 0.6611 |
| hl15, g5 / hl60, g5 / hl120, g5 | 0.6653 / 0.6626 / 0.6650 |
| hl7, g5 | 0.6720 |

The penalties barely matter (gamers play hundreds of matches). The half-life matters.

The ridge's fitted home edge (points), as of 10-18: stream 1 +0.79, stream 2 −0.27, stream 3 −0.04. That is
the only home effect that matters. Gamer strength sd is about 4 points; team strengths run −0.8..+1.1.

**Rolling-form expected margins** from features.csv (plain link, TUNE): exp_margin_all 0.6767, ewma 0.6788,
exp_margin_50 0.6823, exp_margin_15 0.6823, exp_margin_5 0.6833. All are well behind.

The ridge's expected total (same design, sum coded; hl30) has TUNE RMSE 12.04 against exp_total_50's 12.14. I
used it for the "total" extra.

## 2. The link: window, form, against fixed maps (TUNE)

| source | 60 days | 90 | 120 | 180 | expanding | hl 30 | hl 60 | probit (120) | fixed normal 9.35 |
|---|---|---|---|---|---|---|---|---|---|
| ridge (chosen) | **0.6590** | 0.6590 | 0.6591 | 0.6592 | 0.6592 | 0.6591 | 0.6591 | 0.6592 | 0.6599 |
| kal d0.05 s7 | 0.6600 | 0.6600 | 0.6601 | 0.6602 | 0.6602 | 0.6601 | 0.6601 | 0.6602 | 0.6599 |
| rating (drift 4) | 0.6699 | 0.6698 | 0.6699 | 0.6699 | 0.6699 | 0.6699 | 0.6698 | 0.6699 | 0.6766 |

- The window hardly matters. I froze the link at **the last 60 days, refitted daily**.
- Logit and probit are the same.
- The fixed normal 9.35 is as good as the fitted link for the slow Kalman, slightly worse for the ridge
  (+0.0008 TUNE, +0.0003 TEST, both within noise), and far worse for the drift-4 rating.
- For the tables (Sep-Oct, trained from 07-25):
  - The link on avg's margin equals its own NB2 moneyline: SEP −0.0002 (−0.0010,+0.0008), TEST +0.0002.
  - The link on logit(own p) is the same: SEP +0.0001.
  - The probit is the same: −0.0002.
  - The link only moves the calibration slope (1.11 → 1.01).

**Coefficients over time** (daily refits; means by fortnight in out/A/s7_*.log). The shrink is the probit's
b·9.35 (1 means the normal sd 9.35 is right); sd_eff is the probit's 1/b.

| source | b (logit) Jun → Oct | shrink Jun / Aug / mid-Sep / mid-Oct | sd_eff | stable? |
|---|---|---|---|---|
| rating, drift 4 | 0.099 → 0.077 | 0.58 / 0.66 / 0.59 / **0.45** | 14-16 → 21 | no: drifting down since late Sep |
| kal d0.05 | 0.167 → 0.160 | 0.97 / 1.06 / 1.03 / 0.93 | 8.8-10.0 | yes |
| ridge (chosen) | 0.182 → 0.209 (peak 0.223 Sep) | 1.05 / 1.22 / 1.28 / 1.21 | 7.3-8.9 | fairly (its margins are a little compressed) |
| tbl avg (from Aug) | 0.177 → 0.187 | – / 1.12 / 1.10 / 1.09 | 8.3-9.2 | yes |
| tbl glmer_both | 0.175 → 0.187 | – / 1.13 / 1.10 / 1.09 | 8.3-9.3 | yes |
| tbl form | 0.142 → 0.129 | – / 0.92 / 0.84 / **0.75** | 10.1 → 12.4 | no: same drift as the rating |

- The link's intercept a (logit) on the ridge is about −0.04 in Jun-Aug and about 0 in Sep-Oct; the ridge has
  its own home edge. On the Kalman (no home term) a is +0.02..+0.05, about +1% home.
- Margin spread on TEST: rating sd 3.78, kal 2.40, form table 2.79, avg table 2.20, ridge 1.97.

## 3. The extras, one at a time (log loss less the plain link's; day-resampled 95%)

The extras:
- **rematch**: mean residual (margin − the source's margin) of the pair's earlier meetings in the same
  bet_rematches pair-session (240-minute session gap), settled by kick-off.
- **rematch_full**: rematch plus the earlier results ±1 and a has-a-settled-meeting indicator.
- **meeting**: meeting 2, 3+ dummies.
- **sess_gap**: each gamer's mean residual against the source in his earlier matches this session (against
  anyone), home minus away. This is the inference layer's idea.
- **sess_win / sess_margin**: features.csv's session results, as a difference.
- **fast_gap**: the form gap. For slow sources it is drift-4 rating minus the source; for the rating it is
  ridge minus rating.
- **ewma_gap**: the EWMA form margin less the source.
- **stream**: stream 2 and 3 intercepts. **stream_x**: those plus margin × stream.
- **h2h**: features.csv's shrunk head-to-head margin.
- **new_x**: margin × (fewer than 30 games).
- **tot**: margin·sqrt(34/expected total). **tot_x**: margin × (total − 34)/34.
- **unc_c**: m/sqrt(1 + c·var), with var the source's own variance: ridge x'A⁻¹x·81, Kalman var1+var2.
- **unc_g_c**: m/sqrt(1 + c/min games).

**Ridge link** (base: TUNE 0.6590, TEST 0.6732):

| extra | TUNE | TEST |
|---|---|---|
| rematch | −0.0003 (−0.0009,+0.0002) | −0.0004 (−0.0012,+0.0005) |
| rematch_full | −0.0001 (−0.0009,+0.0006) | −0.0002 (−0.0011,+0.0008) |
| meeting | +0.0001 (−0.0001,+0.0004) | +0.0000 (−0.0004,+0.0005) |
| sess_gap | −0.0004 (−0.0009,+0.0001) | −0.0002 (−0.0010,+0.0006) |
| sess_win | −0.0003 (−0.0011,+0.0005) | −0.0000 (−0.0009,+0.0008) |
| sess_margin | −0.0003 (−0.0007,+0.0002) | +0.0001 (−0.0005,+0.0008) |
| fast_gap (drift-4 rating − ridge) | −0.0004 (−0.0010,+0.0002) | −0.0001 (−0.0009,+0.0006) |
| ewma_gap | +0.0001 (−0.0000,+0.0002) | +0.0001 (−0.0003,+0.0004) |
| stream | +0.0000 (−0.0004,+0.0004) | −0.0000 (−0.0004,+0.0004) |
| stream_x | +0.0004 (−0.0001,+0.0009) | +0.0001 (−0.0004,+0.0006) |
| h2h | −0.0004 (−0.0011,+0.0003) | −0.0004 (−0.0012,+0.0004) |
| new_x | +0.0003 (−0.0003,+0.0010) | +0.0000 (−0.0001,+0.0001) |
| tot | −0.0003 (−0.0008,+0.0002) | +0.0001 (−0.0003,+0.0004) |
| tot_x | −0.0003 (−0.0011,+0.0006) | +0.0001 (−0.0003,+0.0005) |
| unc_0.03 (best c) | −0.0002 (−0.0005,+0.0001) | −0.0000 (−0.0001,+0.0001) |
| unc_g_5 (best c) | −0.0004 (−0.0009,+0.0002) | +0.0000 (−0.0001,+0.0001) |

**Kalman d0.05 link** (base: TUNE 0.6600, TEST 0.6750). The only clear TUNE gain is **stream** −0.0010
(−0.0018,−0.0002); on TEST it is −0.0007 (−0.0019,+0.0006). h2h is −0.0009 (−0.0019,+0.0002) on TUNE and −0.0011
(−0.0022,+0.0001) on TEST. Everything else is within ±0.0004.

**Drift-4 rating link** (base: TUNE 0.6699, TEST 0.6820):
- fast_gap (the ridge brought in): −0.0113 (−0.0148,−0.0080) TUNE, −0.0089 (−0.0116,−0.0059) TEST.
- h2h: −0.0041 (−0.0063,−0.0019) TUNE, −0.0038 (−0.0060,−0.0015) TEST.
- stream: −0.0012 (−0.0021,−0.0003) TUNE, −0.0007 TEST.
- uncertainty unc_0.03..0.3: about −0.0003 on TEST (n.s.).
- The rest are within ±0.0002.

**On the avg table** (links trained from 07-25; SEP and TEST against tbl_avg_p, in out/A/s4_tables.csv): every
extra is within ±0.0006 on SEP and TEST, and none is clear of zero.
- rematch: SEP −0.0001 (−0.0011,+0.0010), TEST +0.0005.
- h2h: SEP −0.0005 (−0.0016,+0.0006).
- stream: SEP +0.0000 (−0.0020,+0.0020).
- tot: SEP −0.0002.
- The table already carries the rematch and inference layers.

**Selection.** A first greedy forward selection added anything with a TUNE gain over 0.0002:
- ridge picked unc_g_5 + sess_gap + tot: TUNE −0.0011, TEST −0.0001 (−0.0010,+0.0007). It overfit.

I replaced it with a strict rule (scripts/A/s3b_select.py): a candidate must be clear of zero on TUNE by
itself, and each added step must be clear of zero on TUNE against the model so far.
- **ridge**: none.
- **kal**: + stream. TEST −0.0007 (−0.0020,+0.0006).
- **drift-4 rating**: + (ridge − rating). TEST −0.0089 (−0.0115,−0.0060). After that, h2h's step was
  −0.0004 (−0.0010,+0.0004) and was not added.

**Rematch in particular.** The fitted coefficient on the earlier meetings' residual is +0.007..+0.014 logit per
point, against 0.18-0.22 per point of margin. That is the same as moving the margin by about 5% of the earlier
meeting's miss. It is stable over time and not worth anything (above). It agrees with the codebase's own
rematch layer on the walkthrough's errors: "one earlier meeting moves it by 0.012 of its error" (same teams),
0.000 with teams changed. The meeting number adds nothing either.

**Home.**
- On the ridge, one home edge (no stream split) against the chosen home edge per stream: +0.0010
  (+0.0003,+0.0018) TUNE, +0.0006 (−0.0006,+0.0020) TEST.
- A home edge per gamer instead (one home + a per-gamer home edge, penalty 200): against the single-home ridge
  −0.0003 TUNE, +0.0001 TEST. Against the per-stream ridge +0.0008 (+0.0001,+0.0015) TUNE, +0.0007 TEST.
- The link's own stream intercepts on top of the stream-home ridge: +0.0000 / −0.0000. So **home differs by stream** (stream 1 about +0.8 points; 2 and 3 about 0). It is best put
in the margin model, not the link. A per-gamer home edge does not pay.

## 4. Several margins at once (a small stack)

**Long-history** (60-day link; against the single ridge link):

| stack | TUNE | TEST |
|---|---|---|
| ridge + rating (drift 4) | −0.0004 (−0.0010,+0.0002) | −0.0001 (−0.0009,+0.0006) |
| ridge + kal d0.05 | −0.0002 (−0.0009,+0.0004) | −0.0000 (−0.0007,+0.0006) |
| ridge hl30 + ridge hl7 + ridge hl120 | +0.0002 (−0.0005,+0.0008) | −0.0002 (−0.0007,+0.0003) |
| ridge + kal + rating | −0.0004 (−0.0010,+0.0003) | +0.0000 (−0.0008,+0.0008) |
| ridge + rating + h2h | −0.0008 (−0.0018,+0.0003) | −0.0005 (−0.0016,+0.0006) |
| kal + rating + exp_margin_all | +0.0012 (−0.0006,+0.0031) | +0.0020 (−0.0001,+0.0042) |
| ridge + ridge hl7 + kal + rating + exp_margin_all | −0.0003 (−0.0015,+0.0009) | −0.0002 (−0.0012,+0.0010) |
| ridge live + ridge daily | −0.0001 (−0.0004,+0.0002) | +0.0001 (−0.0002,+0.0005) |

No stack beats the single ridge.

**With the tables** (SEP / TEST against tbl_avg_p):

| stack | 09-01..09-21 | SEP | TEST |
|---|---|---|---|
| avg + ridge | −0.0014 | −0.0005 (−0.0020,+0.0008) | −0.0003 (−0.0014,+0.0009) |
| avg + ridge + rating | −0.0013 | −0.0004 | −0.0001 |
| avg + rating | −0.0000 | −0.0001 | +0.0004 |
| nb2_infer + glmer_both | −0.0002 | −0.0004 | +0.0002 |

The first two rows are A_best's choice on 09-01..09-21. avg + rating is the results' version of the form table:
the rating's weight is 0.005-0.015 against avg's 0.18.

The avg+ridge weights are about 0.10 / 0.10 from August, so it is roughly the plain average of the two margins.
It is the best on 09-01..09-21 (the only window allowed for this choice), so it became A_best's table part. It
is level with tbl_avg_p.

## 5. Final reports (harness.report, common matches, prod and reference columns)

**TUNE 2026-06-01..07-31** (5,462 matches priced by every column)

| model | log loss | Brier | AUC | slope | vs prod | vs ridge link |
|---|---|---|---|---|---|---|
| rating, normal 9.35 (raw) | 0.6773 | 0.2413 | 0.623 | 0.63 | +0.0009 (−0.0069,+0.0086) | +0.0165 |
| rating linked | 0.6704 | 0.2388 | 0.622 | 1.06 | −0.0060 (−0.0111,−0.0006) | +0.0096 |
| rating + (ridge − rating) linked | 0.6602 | 0.2340 | 0.646 | 1.04 | −0.0161 | −0.0006 |
| kal d0.05 raw | 0.6613 | 0.2344 | 0.644 | 1.03 | −0.0151 | +0.0005 |
| kal linked | 0.6615 | 0.2345 | 0.643 | 1.00 | −0.0149 | +0.0007 |
| kal + stream linked | 0.6604 | 0.2340 | 0.646 | 1.01 | −0.0160 | −0.0004 |
| ridge raw | 0.6616 | 0.2346 | 0.645 | 1.19 | −0.0147 | +0.0008 (−0.0001,+0.0018) |
| **ridge linked** | **0.6608** | 0.2342 | 0.645 | 1.04 | **−0.0156 (−0.0211,−0.0104)** | |
| prod | 0.6764 | 0.2418 | 0.605 | 1.13 | | |

**TEST 2026-08-01..10-18** (6,170)

| model | log loss | Brier | AUC | slope | vs prod | vs ridge link |
|---|---|---|---|---|---|---|
| rating raw | 0.6963 | 0.2506 | 0.580 | 0.46 | +0.0115 (+0.0037,+0.0187) | +0.0219 |
| rating linked | 0.6829 | 0.2450 | 0.580 | 0.79 | −0.0018 (−0.0071,+0.0032) | +0.0086 (+0.0052,+0.0118) |
| rating + gap linked | 0.6742 | 0.2408 | 0.606 | 0.96 | −0.0106 | −0.0001 |
| kal raw | 0.6762 | 0.2417 | 0.601 | 0.94 | −0.0086 | +0.0018 |
| kal linked | 0.6761 | 0.2417 | 0.601 | 0.92 | −0.0087 | +0.0017 (−0.0004,+0.0040) |
| kal + stream linked | 0.6753 | 0.2413 | 0.602 | 0.90 | −0.0094 | +0.0010 |
| ridge raw | 0.6747 | 0.2409 | 0.607 | 1.22 | −0.0101 | +0.0003 (−0.0006,+0.0013) |
| **ridge linked** | **0.6744** | 0.2408 | 0.606 | 0.98 | **−0.0104 (−0.0147,−0.0061)** | |
| *with the August tables:* tbl_avg_p | 0.6734 | 0.2404 | 0.608 | 1.12 | −0.0114 (−0.0156,−0.0072) | −0.0010 (−0.0029,+0.0009) |
| avg linked | 0.6736 | 0.2405 | 0.607 | 0.99 | −0.0112 | −0.0008 |
| glmer_both linked | 0.6735 | 0.2404 | 0.607 | 1.00 | −0.0113 | −0.0008 |
| form linked | 0.6776 | 0.2424 | 0.595 | 0.90 | −0.0072 | +0.0033 (+0.0006,+0.0059) |
| **A_best: avg + ridge linked** | **0.6731** | 0.2402 | 0.609 | 1.01 | **−0.0117 (−0.0161,−0.0073)** | −0.0012 (−0.0023,−0.0001); vs tbl_avg_p −0.0003 (−0.0014,+0.0009) |
| prod | 0.6848 | 0.2460 | 0.569 | 0.93 | | |

**SEP 2026-09-01..10-18** (3,426; repo tables, links trained from 07-25)

| model | log loss | slope | vs prod | vs ridge link | vs tbl_avg_p |
|---|---|---|---|---|---|
| tbl_avg_p (its own NB2) | 0.6777 | 1.11 | −0.0116 (−0.0179,−0.0058) | −0.0012 (−0.0038,+0.0016) | |
| tbl_glmer_both_p | 0.6774 | 1.10 | −0.0120 | −0.0015 | −0.0003 (−0.0017,+0.0010) |
| tbl_form_p (v2.1's prior) | 0.6828 | 0.71 | −0.0066 (−0.0150,+0.0013) | +0.0038 | **+0.0050 (+0.0012,+0.0085)** |
| avg linked | 0.6776 | 1.01 | −0.0118 | −0.0013 | −0.0002 (−0.0010,+0.0008) |
| glmer_both linked | 0.6772 | 1.01 | −0.0122 | −0.0017 | −0.0005 (−0.0018,+0.0009) |
| form linked | 0.6813 | 0.87 | −0.0080 | +0.0024 | +0.0036 (+0.0002,+0.0068) |
| **A_best (avg + ridge linked)** | **0.6772** | 1.02 | **−0.0122 (−0.0182,−0.0066)** | −0.0017 (−0.0032,−0.0002) | −0.0005 (−0.0020,+0.0009) |
| ridge linked | 0.6789 | 0.96 | −0.0105 (−0.0163,−0.0048) | | +0.0012 (−0.0016,+0.0038) |
| kal linked | 0.6803 | 0.90 | −0.0091 | +0.0013 | +0.0025 |
| rating linked | 0.6856 | 0.76 | −0.0037 | +0.0067 | +0.0079 |
| rating raw | 0.7013 | 0.40 | +0.0119 | +0.0223 | +0.0235 |
| prod | 0.6894 | 0.68 | | +0.0105 | +0.0116 |

**09-22..10-18** (1,840)
- A_best: 0.6788. Against prod −0.0091 (−0.0156,−0.0028); against tbl_avg_p +0.0003 (−0.0019,+0.0023);
  against the ridge link −0.0024 (−0.0046,−0.0002).
- Ridge link: 0.6813. Against prod −0.0067 (−0.0129,−0.0008); against tbl_avg_p +0.0027 (−0.0015,+0.0064).
- tbl_glmer_both_p: 0.6777, the best single here. Against tbl_avg_p −0.0009 (−0.0024,+0.0008).
- tbl_form_p: +0.0057 (+0.0012,+0.0099) against tbl_avg_p.
- prod: 0.6880.

Full text: out/A/s6_final.txt.

**Calibration by decile** (each model on its own deciles, common matches):
- **TEST, ridge link**: mean p and win rate in each decile are 0.336/0.323, 0.407/0.408, 0.444/0.473,
  0.472/0.470, 0.496/0.524, 0.520/0.525, 0.546/0.574, 0.572/0.551, 0.608/0.603, 0.679/0.676. The largest gap
  is 0.029, which is within noise (±0.02 a decile).
- **TEST, prod**: 0.378/0.380 … 0.508/**0.566**, 0.542/0.572, 0.620/**0.595**. Gaps reach 0.059.
- **SEP, A_best**: gaps within about 0.03, except the top and bottom deciles at 0.66/0.65 and 0.36/0.31.
- **SEP, prod**: gaps up to ±0.06 in the middle deciles, and its top decile is 0.618 against 0.560.

## 6. Predictions written (preds/, code + p_home; 2026-06-01..10-18 wherever priced)

| file | what | matches |
|---|---|---|
| A_best | avg+ridge stack link where the avg table prices (from 07-28, after the link's 300-match start), else the ridge link | 12,119 |
| A_link_ridge | ridge hl30 g2 t50 + stream home edge, 60-day link, no extras | 12,119 |
| A_link_kal / A_link_kal_stream | Kalman d0.05 s7, linked / + stream intercepts | 12,119 |
| A_link_rating / A_link_rating_gap | drift-4 rating linked / + (ridge − rating) | 12,119 |
| A_link_tbl_avg, A_link_tbl_glmer_both, A_link_tbl_form | table margins linked (trained from 07-25) | 6,596 |
| A_raw_rating, A_raw_kal, A_raw_ridge | margin read as normal sd 9.35 | 12,119 |
| A_raw_tbl_avg, A_raw_tbl_glmer_both | the tables' own p (Jul 25-Aug 31 from my walkthrough run, of which Jul 25-31 is the layers' burn-in; Sep-Oct the repo's) | 6,981 |
| A_raw_tbl_form | the form table's own p (Sep-Oct only) | 3,463 |

## Conclusions

- The link learns the shrinkage. For the drift-4 rating it multiplies the margin by about 0.6 (effective sd 16
  points, against the 9.35 the codebase assumes) and gains 0.007-0.013 log loss. The user's diagnosis is right.
- But the shrinkage it learns for the drift-4 rating is not stable: 0.58 → 0.45 over Jun-Oct.
- The linked rating still loses to a slower margin by 0.009 on TEST. Most of the problem is the drift-4 rating
  carrying too much noise, not too wide a map. A link on top cannot undo noise in the means; it can only shrink
  everything toward evens. That is why the linked form table is still 0.0036 worse than avg.
- Slow margins (ridge half-life 30 days, Kalman drift 0.05, the NB2/glmer tables) are already near-calibrated
  under a normal sd 8-9.5. For them the fitted link is a cheap safety net (slope 1.0 instead of 1.1-1.2), worth
  only 0.0002-0.0008.
- None of the extras earn a place on a good margin. The rematch residual, session form, form gap, stream,
  uncertainty and total are all within ±0.0004 with intervals across zero on TUNE and TEST. The home edge
  belongs in the margin model, per stream.
- Gamer×team strengths were not tried. Teams are worth only ±1 point and a live 930-parameter ridge was too
  slow for the shared cores.

## Recommendation for productionising

1. **Margin.** The decayed ridge on margin:
   - Terms: gamer + team + a home edge per stream (+0.8 points on stream 1, about 0 on 2 and 3).
   - Half-life 30 days; penalties 2 (gamers) and 50 (teams), in matches of full weight.
   - Re-solved each time a result is in (36 minutes after kick-off). That is a 103-parameter solve, about 1 ms.
     Re-solving only at midnight cost 0.003-0.004 log loss on TUNE.
   - Or keep the walkthrough's avg table and average its margin with the ridge's. That is A_best: equal to
     tbl_avg_p, 0.001-0.002 better than the ridge alone, but it needs the daily NB2 + glmer fits.
2. **Link.** P(home) = logistic(a + b·m), refitted once a day on the last 60 days of settled results (about
   5,500 matches; the window length is immaterial between 30 and 180 days or a 30-120 day half-life).
   - Expect b ≈ 0.20-0.22 per point (effective sd about 7.5-8.5) and a ≈ 0.
   - **No extras.**
   - Monitor b·9.35. A drift like the drift-4 rating's (0.6 → 0.45) means the margin model, not the link, needs
     fixing.
3. **For v2.1's prior means.**
   - Stop moving avg half way to the drift-4 rating (avg_form4_w05). It costs +0.0050 (+0.0012,+0.0085) on SEP
     against avg itself, and the results give the rating 3-8% of avg's weight.
   - If a form move is wanted, move toward a drift-0.05 rating or the ridge.
   - If the sims need a wider or narrower margin, use the fitted effective sd (8.3-8.6 for avg) rather than
     9.35.
4. **Rematches.** No pre-match move for the margin: the link and the codebase's own rematch layer both find
   about 1-5% of the earlier miss. The totals might be different; I did not test them.

## Caveats
- **Clearing zero.** All margins of 0.001-0.002 between the good models (A_best, tbl_avg_p, tbl_glmer_both_p,
  ridge link) are inside the day-resampled intervals. Only "beats prod by about 0.010-0.012" and "drift-4 /
  form is worse by 0.004-0.009" clear zero.
- **August table rows** come from my own walkthrough run (same code, `--since 2026-07-25`), not the repo's
  tables. Jul 25-31 is the layers' burn-in.
- **A_best's table part** was chosen on 09-01..09-21 among the table stacks. Its edge there (−0.0014 against
  tbl_avg_p) shrank to −0.0005 (SEP) and +0.0003 (09-22..10-18).

## Files
- Scripts, in order: scripts/A/common.py (data, Kalman, ridge, rematch/session features, logistic, link
  walk-forward), extras.py, tables.py, s1_sources.py, s1b_refine.py, s2_window.py, s3_extras.py,
  s3b_select.py, s4_tables.py, s5_stack.py, s6_final.py, s7_coefs.py.
- Outputs: out/A/sources.pkl (every margin source for every match), s1_tune.csv, s2_window.csv,
  s3_*.csv/.log, s3b_*.txt, s4_tables.csv, s5_stack.csv, s6_final.txt, s7_coefs.csv, the logs, and
  walkthrough/ (my tables, 07-25..08-31).
