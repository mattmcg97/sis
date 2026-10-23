# Agent G: an in-play win-probability benchmark (NFL-style), and a check on prod's and v2.1's in-play moneyline

Scripts `scripts/G/`, outputs `out/G/`. Nothing here is a pre-match model, so nothing went to `preds/`.

## The short answer

1. **On the bet moments, a direct win-probability model on the state the bets carry loses to both in-play prices by
   about 0.025-0.035 of log loss.** REPORT window (match days 10-11..10-18, frozen settings, 6,110 moments, 526
   matches):
   - The Stern logistic is 0.6115. Against prod (0.5862) it is +0.0253: days (+0.0157,+0.0360), matches
     (+0.0099,+0.0396). Against v2.1 (0.5798) it is +0.0317 (+0.0236,+0.0421; +0.0188,+0.0441).
   - Monotone XGBoost is 0.6151: +0.0289 against prod and +0.0353 against v2.1.
   - XGBoost boosted from the Stern logit is 0.6112: +0.0249 and +0.0313.
   - This is not because the model is poorly fitted: its calibration slope is 0.98 and its intercept +0.01. It loses
     because the bets carry no possession, field position, distance or timeouts. Prod and v2.1 price with all of
     them, so the gap is the value of that missing state. In a walk-forward stack the WP model gets weight 0.10
     beside prod's logit and -0.02 beside v2.1's: it adds nothing either price lacks.
2. **So, as a check on v2.1, it says nothing bad about v2.1.** On the same moments:
   - v2.1 against prod: -0.0064. The day-resampled interval (-0.0123,-0.0009) excludes zero; the match-resampled
     one (-0.0169,+0.0042) does not. On TUNE (10-04..10-10) v2.1 was +0.0021 (-0.0064,+0.0103) against prod.
   - The late-Q4 variant (v21-late, since removed) is -0.0013 against v2.1 on REPORT.
   - Both prices are well calibrated on REPORT (slopes 1.05 and 1.07). Both were too spread out on TUNE (0.84, 0.89).
   - A walk-forward stack of prod and v2.1 gives v2.1 the larger weight (0.61 against 0.37). The stack is not better
     than v2.1 alone (+0.0028, interval includes 0).
3. **The full benchmark needs the PLAY_OVER export, and `scripts/G/inplay_wp.py` runs it.** It reads the export's
   full state (score, clock, period/overtime, possession, yard line, down, distance, both sides' timeouts, the
   pending conversion, who gets the second-half kick-off) and joins a pre-match table. It fits monotone XGBoost
   from the side with the ball's view, walk-forward. It scores the fit against prod's live in-play price at every
   PLAY_OVER prod quoted (and against v2.1's, given its prices). Tested on synthetic exports and on the repo's own
   fixtures. Not run on real data: the export is not in this container.
4. **The late-game check on v2.1 (endgame states, no betting selection) is in section 4.**

## 1. What exists in this container

- **`eAMFCalibrator/out_bets_*/bets_sim.csv` (and `.zip`):** every bet, 2026-10-01..10-18 at most. These are the only
  in-play states with prices.
  - The big files cover the same bets: `late`, `gap` and `pace` (18 days, 1,214 matches with in-play moneyline bets),
    and `shape`. `react` and `v21_4k` cover 10-01..10-12; `drives` and `avg` cover 10-12..10-18.
  - **Their base v2.1 runs differ.** `late` = `gap` = `pace`: the form-prior build at 2,000 games a snapshot (to
    1e-4). `shape`, `react`, `v21_4k` and `drives` are other builds (sd of the difference 0.014-0.04).
  - I use `out_bets_late`'s `candidate_prob_v21` as "v2.1" and its `candidate_prob_v21-late` as the late-Q4 variant.
- **No full PLAY_OVER play-by-play** (scouting_playover.csv) and no per-snapshot pair rows. The calibrator HTML
  reports hold only aggregates.
- **`eAMFCalibrator/out_v21_endgame{,_sep}/v21_endgame_states.csv`:** 27,892 real Q4 snaps with |lead| <= 8,
  09-01..10-17. Each has the clock, the side with the ball's lead, real_win and v2.1's sim_win.
- **`eAMFCalibrator/out_money/bets_money_points.zip`:** the same bets with `script` = home's lead x the share of
  regulation left, at the bet's seen time. **This is the only signed score in the container**; see section 2.
- **`eAMFCalibrator/out_pair_check/pair_check.csv`:** prod's market-50 prices at PLAY_OVER messages (101k), with no
  state. Not used.

## 2. The state fields, exactly

bets_sim's in-play state, read off SCOUTING_FULL at **bet time** (bet_moments.moment_of):

| column | what it is |
|---|---|
| `quarter` | Q1-Q4, OT, unknown |
| `clock_left` | the game clock left in the quarter (s, of 240) |
| `clock_band` | quarter x (last 2:00 or before) |
| `down` | 1-4. **Whose down is not given.** |
| `score_margin` | **absolute band only**: level (0-2), 1 score (3-8), 2+ scores (9+). No sign, no points. |
| `moment` | the feed's state: between plays, play live, timeout, kick-off to come / live, conversion to come, quarter break, interruption |
| `feed_suspended` | the feed's own suspension (open / suspended) |
| `moved` | what the feed did between prod's price message and the bet: nothing, play started, timeout, possession, score, kick-off, play over, price after the bet |
| `price_age_seconds` | how old prod's price was at the bet (typically 10-60 s) |
| `match_phase` | in play, in play before the start, after two minutes |
| `next_score(_seconds)` | **the future** (seconds to the board's next move): never used |

**Missing:** possession (the offense), field position, distance, timeouts, the signed score and the total.

**The signed lead.** It is recovered as `lead = script / share_left`, where share_left = ((4 - q) x 240 + clock) / 960
from bets_sim's quarter and clock.
- 99.9% of the recovered leads fall inside the bet's own `score_margin` band.
- 88% of the in-play moneyline and spread bets get a clean whole number. The others either have no money row (10%)
  or a clock that moved between the seen time and the bet time.
- Overtime has share 0, so it has no lead and is left out.

**Side convention.**
- `stream_prob` is prod's quote for the bet's own selection: Home = market 50 = P(home wins), Away = market 51.
- The book is fair: for every bet, P(home) = stream_prob (Home) or 1 - stream_prob (Away) equals the money file's
  market-50 price (median |difference| 0.0000).
- The same holds for `candidate_prob_v21`, which is identical across a message's bets.
- Only `market == "moneyline"` is used for scoring (spread and total are left out). Moneyline and spread bets' states
  are used for training.

## 3. The bet-moment check (scripts/G/moments.py, wp_bets.py)

**Moments.** One row per (match, prod message) of the in-play moneyline bets: 24,923. The state comes from the bet
nearest the message (least price age). The **clean set (14,902 moments, 1,197 matches)** keeps a moment when:
- it is in Q1-Q4 and its lead was read cleanly;
- `moved == "nothing"`: no score, possession, kick, play or timeout between prod's price and the bet, so the bet's
  state is the priced state. Only the clock can have run on a few seconds;
- `candidate_ok_v21`: v2.1 priced the same state;
- it is not excluded, all three pre-match probabilities exist, and the match is not a tie.

Almost all clean moments are "between plays". The **broad set** (7,917 REPORT moments) adds moments where a play had
started or a timeout was called after the price. It gives the same answer.

**These are where people chose to bet: a biased sample of game states.** Every model is scored on the same moments,
so the comparison across models is fair, but the levels are not those of all states.

**Training states.** Every in-play moneyline and spread bet's state (any moment), one per match and message: 30,109.

**Walk-forward.** Each match day is priced by a fit on the states of matches settled (kick-off + 36 min) by that day's
midnight UTC. Only 18 days exist, so instead of the protocol's ~5 weeks:
- burn-in 10-01..10-03;
- **TUNE** = match days 10-04..10-10 (6,199 moments, 463 matches);
- **REPORT** = 10-11..10-18 (6,110 moments, 526 matches, 8 days).

Day-resampled intervals over 8 days are coarse, so match-resampled ones are shown beside them.

**Models.** s = share of regulation left; pre = the pre-match logit (A_best).
- `stern`: logit p = a + b·lead/√(s+k) + c·pre·√(s) [+ d·pre]. This is the final margin as a Brownian motion
  drifting at the pre-match edge.
- `xgb`: binary:logistic on lead, lead/√(s+k), pre, pre·√s, seconds left, period, down and moment, with monotone +
  on the two lead terms and the two pre terms (each rises with the lead / the pre-match logit, so the whole does).
  Time and the rest are free.
- `xgb_on_stern`: the same trees boosted from stern's logit.
- `stern_ls`: adds a possession proxy built from the bets' own past states: who scored last, and how long ago.

**Tuning (TUNE only; out/G/wp_tune.txt, wp_tune2.txt).** 192 + 28 configurations.
- Stern: k is flat over 0.005-0.04 (0.6511 at all of them). A free pre-match term does not help.
- Pre-match input: A_best 0.6511, prod's pre-match 0.6512, C_strength 0.6513, v2.1's own kick-off 0.6516. A_best
  was kept.
- The last-scorer proxy hurt on TUNE (0.6527), so it was not used.
- XGBoost: depth 2, min_child_weight 20, per-match weights, 50 rounds (0.6522); 100+ rounds overfit.
- Boosted-from-stern: depth 2, 10 rounds (0.6526).
- On TUNE every model, prod and v2.1 included, was too spread out (slopes 0.7-0.9): those days had more comebacks.

**REPORT (frozen; out/G/wp_report.txt).** 6,110 moments, 526 matches, 8 days. Each delta is the model's log loss less
the reference's, with 95% intervals (days; matches).

| model | log loss | Brier | AUC | slope | vs prod | vs v2.1 |
|---|---|---|---|---|---|---|
| prod in-play | 0.5862 | 0.2021 | 0.748 | 1.07 | | +0.0064 (+0.0009,+0.0123; -0.0042,+0.0169) |
| **v2.1 in-play** | **0.5798** | 0.1994 | 0.755 | 1.05 | -0.0064 (-0.0123,-0.0009; -0.0169,+0.0042) | |
| v2.1-late (removed) | 0.5785 | 0.1988 | 0.757 | 1.05 | -0.0077 (-0.0138,-0.0024; -0.0182,+0.0028) | -0.0013 (-0.0023,-0.0005; -0.0027,+0.0000) |
| stern (A_best) | 0.6115 | 0.2128 | 0.717 | 0.98 | +0.0253 (+0.0157,+0.0360; +0.0099,+0.0396) | +0.0317 (+0.0236,+0.0421; +0.0188,+0.0441) |
| xgb monotone | 0.6151 | 0.2140 | 0.710 | 1.19 | +0.0289 (+0.0185,+0.0414; +0.0138,+0.0430) | +0.0353 (+0.0218,+0.0501; +0.0204,+0.0493) |
| xgb from stern | 0.6112 | 0.2127 | 0.717 | 0.96 | +0.0249 (+0.0143,+0.0374; +0.0088,+0.0402) | +0.0313 (+0.0218,+0.0431; +0.0177,+0.0442) |
| stern, prod's pre-match | 0.6185 | 0.2159 | 0.707 | 0.97 | +0.0322 | +0.0386 |
| stern, v2.1's kick-off | 0.6128 | 0.2134 | 0.716 | 0.97 | +0.0266 | +0.0330 |
| stern + last scorer (not chosen) | 0.6079 | 0.2113 | 0.721 | 1.00 | +0.0216 (+0.0119,+0.0323) | +0.0280 |
| pre-match A_best alone | 0.6832 | 0.2454 | 0.558 | 0.84 | +0.0970 | +0.1034 |

- **Weighting each match equally:** prod 0.5634, v2.1 0.5529, stern 0.5921, xgb 0.5996. Same order.
- **The same pre-match information:**
  - With prod's own pre-match quote, the stern is +0.0322 behind prod in play.
  - With v2.1's own kick-off price, it is +0.0330 behind v2.1.
  - A_best's better pre-match is worth about 0.007 to the WP model (0.6115 against 0.6185). That is less than the
    in-play gap.
- **By quarter.** The gap grows through the game: Q1 +0.005, Q2 +0.022, Q3 +0.028, Q4 +0.074 (stern against prod).
  Late in the game possession and field position are worth most, and the bet moments carry neither.
- **By |lead|.** The biggest gap is at 1-3 points: stern 0.690 against prod 0.634. A field goal's lead is worth
  little without knowing who has the ball.
- **Stacks** (walk-forward logistic on the two logits, fitted on clean moments from 10-04):
  - prod + stern: weights 0.91 / 0.10; log loss +0.0016 against prod.
  - v2.1 + stern: 0.97 / -0.02.
  - prod + v2.1: 0.37 / 0.61; -0.0036 against prod, +0.0028 against v2.1.
- **Broad set** (out/G/wp_broad.txt): 7,917 moments. Stern +0.0250 against prod and +0.0314 against v2.1; v2.1
  -0.0064 against prod.
- **The ceiling with these features** (descriptive, all days, in-sample): a stern fitted on all the clean moments
  scores 0.622 against prod's 0.597. The gap is the missing state, not the fitting.

## 4. The late-game check of v2.1 (scripts/G/endgame_check.py)

**The data.** eAMFModel `v21-endgame`'s late states: every real Q4 snap with the sides 8 or fewer apart, 09-01..10-17.
That is 27,878 states on 2,593 matches, with ties left out. For each state, v2.1 played the real state out 200 times
(`sim_win`, nudged 75%), against the real result (`real_win` = the side with the ball won). These states are not
selected by betting.
- The file does not say which side had the ball. The final score settles it exactly: the side with the ball is home
  when (it won) == (home won). That recovers a true fact, not the label, so the pre-match P for that side is the
  true feature.
- Possession is implicit (the side with the ball). There is no field position, down or timeouts.

**Models.** Walk-forward, refitted daily.
- `logit`: a + b·lead/√(s+k) + c·pre·√s + d·pre + e·lead.
- `xgb`: monotone + in lead and the pre-match terms.
- `v21_platt`: v2.1's own sim_win, recalibrated walk-forward.

TUNE = 09-08..09-30; REPORT = 10-01..10-17, 9,240 states on 868 matches (out/G/endgame_check.txt).

| model | log loss | slope | less v2.1 (days 95%; matches 95%) |
|---|---|---|---|
| **v2.1 sim_win** | **0.4519** | 1.02 | |
| logit (k 0.005, A_best) | 0.4852 | 0.93 | +0.0332 (+0.0220,+0.0448; +0.0210,+0.0457) |
| xgb (depth 3, 100 rounds) | 0.4784 | 0.94 | +0.0265 (+0.0186,+0.0350; +0.0168,+0.0365) |
| v2.1, Platt-recalibrated | 0.4509 | 0.90 | -0.0010 (-0.0048,+0.0028; -0.0042,+0.0025) |
| logit, prod's pre-match | 0.4867 | 0.93 | +0.0348 |

- **Overall, v2.1's late game is sound.** It beats a direct WP on lead, clock, possession and pre-match by about
  0.03, and recalibration does not help it.
- **In the cells, though, v2.1 underrates the side with the ball that trails by 1-3 points late in Q4.** This
  matches the repo's own out_v21_endgame/v21_endgame.txt, October, each match counted equally.

  | time left | side with the ball | real win % | v2.1 win % |
  |---|---|---|---|
  | 1:00-0:00 | 3 behind | 57.0 | 43.2 |
  | 2:00-1:01 | 3 behind | 66.6 | 47.8 |
  | 4:00-2:01 | 3 behind | 56.6 | 43.6 |
  | 1:00-0:00 | 1-2 behind | 66.1 | 51.1 |
  | 2:00-1:01 | 1-2 behind | 74.5 | 55.4 |

- In those cells the simple logit lands closer for 1-2 behind (64-66%).
- The reverse bias shows when 1-3 ahead with 4:00-2:01 left: real 75.2%, v2.1 81.2%.
- In September the same cells agreed (v21_endgame_sep.txt: 3 behind, last minute 35.4 / 34.8).
- **So it is a recent drift worth a look.** Each state is counted once, so a long successful drive weighs more. The
  same weighting applies to every model.
- v2.1's sim_win is a 200-game mean, which costs it a little log loss by noise.

## 5. The reusable script: scripts/G/inplay_wp.py (not in the repo)

```
python inplay_wp.py --export eAMFCalibrator/out_scouting_xgb/scouting_playover.csv \
    --prematch <ML>/preds/A_best.csv --history eAMFCalibrator/out/match_history.csv \
    --since 2026-06-01 --tune-until 2026-08-01 --until 2026-10-19 --out out_inplay_wp \
    [--compare v21=v21_ml_quotes.csv] [--refit-days 7] [--train-days 120] [--train-every 1] [--tune-every 2] [--nthread 2]
```

**What it reads.**
- **The export:** the columns of eAMFCalibrator/scouting.py's EXPORT_FIELDS that it needs. Missing optional columns
  are tolerated: timeouts_used_a/b, file_time, opening_offense, prob_50/live_50 and prematch_prob_50.
- **The pre-match table:** MATCH_CODE (or code) and p_home. It must be walk-forward, as A_best is. A match the table
  lacks falls back to prod's pre-match quote from the export, counted in the report; `--no-prod-fallback` drops
  those matches instead.
- **The history:** kick-off times. Without it, a match's first file_time is used, else the date in its code.

**The state at each PLAY_OVER.** It follows eAMFModel/playover.state_for and v21.match_starts. The row's play kind
is the play that just ended; its offense, down, distance and field position are the state for the next snap.
- **TOUCHDOWN:** the scorer's six goes on the board (score_p*_at_start + 6 when the board lags), and the conversion is
  pending. The side to receive after it "has the ball", with the opponent's conversion flagged.
- **CONVERSION:** its 1 or 2 points go on the board, and the other side receives.
- **FIELD_GOAL:** a good kick adds 3 and the other side receives. A miss gives the other side its snap off the row,
  else its ball with the spot unknown.
- **SAFETY:** adds 2, and the scorer receives the free kick.
- **Anything else:** the row's own snap state.

A kick to come is set at 1st and 10 from the receiver's 25 (v21's kick-off start) and flagged.
- TEAM_A/TEAM_B are mapped to home/away by team_a_side. A match whose side is unresolved is read as home, as v21
  does, and counted.
- Timeouts left = 3 a half (2 in an overtime period) less timeouts_used.
- Who gets the second-half kick-off comes from opening_offense, as v21 reads it.
- Matches whose feed lost a quarter's start (the clock back up 60 s or more within a period) are dropped, as
  feed.lost_quarter flags them.

**Models.** Each is refitted every `--refit-days` on the matches settled (their last file_time) by the block's first
midnight, over the last `--train-days`.
- **`wp`:** XGBoost from the side with the ball's view, as nflfastR does. P(home) is p or 1 - p.
  - Features: lead, lead/√(s+0.02), pre-match logit, pre·√s, seconds left, seconds left in the half, period,
    overtime, down, distance, yards to goal, own and opponent's timeouts, the opponent's conversion to come, gets
    the second-half kick, is home, and the kind of state.
  - Monotone + on: the lead terms, the pre-match terms and own timeouts.
  - Monotone - on: down, distance, yards to go, the opponent's timeouts and its pending conversion.
  - Time and the rest are free.
- **`wp_lite`:** home's view with only what the bet moments carry. Comparing it with `wp` shows what possession,
  field position and timeouts add.
- **`stern`:** the logistic baseline.
- **`pre`:** the pre-match probability alone.
- **`--tune-until`:** chooses depth, min_child_weight and rounds from a small grid by log loss on the earlier test
  days, then freezes them for the later days, which the report leads with.

**What it writes (to `--out`).**
- **`states.csv`:** every state with its features, prod's live price and each model's P(home).
- **`report.txt`:** for TUNE and REPORT separately, on every state and on the PLAY_OVERs prod quoted live, each on
  common rows:
  - log loss, Brier, AUC and calibration slope/intercept;
  - the difference against prod's in-play price (and any `--compare` column) with 95% intervals resampling days and
    matches;
  - log loss by quarter and calibration by decile;
  - a monotonicity check on the last fit (0 violations expected), the trees' gain shares and the stern
    coefficients.
- **`settings.json`.**

**v2.1's prices for `--compare`.** A CSV of match_code, message, p_home. eAMFModel/v21_stream.quotes_for_matches gives
market 50's quotes per message, the same thing the calibrator's `--candidate v21` reads.

**Tests (scripts/G/test_inplay_wp.py, 7 tests, all pass).** Run from scripts/G with
`PYTHONDONTWRITEBYTECODE=1 python -m unittest test_inplay_wp`. They check:
- each scoring row's board, possession and kind, including a lagging board, a safety and a missed kick;
- that the same game with TEAM_A as the away side gives identical home-view states;
- that a lost quarter start drops the match;
- that the repo's own fixtures read whole: xgb/tests/test_drives.py's M1/M2 (the late kick-after-six board), and
  eAMFModel/tests/fakes.py's synthetic matches, whose last board equals the final;
- that the walk-forward never trains on a match settled after the block's cut;
- that the trees are monotone (0 violations) and beat the pre-match alone.

**Synthetic run (scripts/G/make_synthetic.py; out/G/synth/run2).** 1,800 toy matches with the export's columns and
quirks: TEAM_A away in a third of them, a lagging board on a fifth, timeouts and overtime, and a noisy "prod".
The run tuned on 07-15..08-04, which chose depth 4, min_child_weight 50 and 300 rounds, and reported on
08-05..08-29 (48,406 states, 722 matches):

| model | log loss |
|---|---|
| wp | 0.3357 |
| wp_lite | 0.3578 |
| stern | 0.3605 |
| pre-match alone | 0.5408 |

- wp's calibration slope is 0.99.
- In this toy, possession, field position and timeouts are worth 0.022 (wp_lite less wp).
- Against the synthetic "prod": -0.0757, with 95% intervals of (-0.0945,-0.0563) over days and (-0.0916,-0.0579)
  over matches.
- 0 monotonicity violations.
- The run took 6 minutes for 121k rows.
- These numbers only show that the plumbing works: the toy's "prod" is a made-up price.

**Runtime.** The real export has about 26k matches and roughly 3-4M PLAY_OVERs. Expect a few minutes to read it and
about 1-2 minutes a fit on a 120-day window at nthread 2. The default grid is 18 settings x one fit per tuning block,
on every second state. Run it in the background; `--train-every 2` halves the time.

## 6. Caveats

- **The time window.** Only 18 days of bet moments exist, so REPORT is 8 days and 526 matches, not 5 weeks. The
  day-resampled intervals rest on 8 days.
- **The bet moments are a betting-selected sample.** Almost all clean moments are between plays.
- **The bet-moment state lacks possession, field position, distance and timeouts.** The bet-moment WP model is
  therefore a floor for the benchmark, not the benchmark the design notes describe. That one needs the export
  (inplay_wp.py).
- **The lead comes from the money file's `script`, not the feed's score.** Where the two clocks differ the lead does
  not round cleanly, and those moments are dropped. 10% of bets have no money row. No overtime.
- **The clean filter keeps the state at bet time.** Only the game clock can have run on since prod's message, a slight
  edge to the WP model, and it still loses.
- **v2.1's price is one 2,000-game run of one build.** Other runs in the folders differ by sd 0.014-0.04 per moment.
- **The decision not to use the last-scorer proxy was taken on TUNE.** It would have helped on REPORT (0.6079 against
  0.6115), and that REPORT number was not used for any choice.
