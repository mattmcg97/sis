# eAMFModel

A rival pricer for eAMF moneyline, spread and total, built top-down with no
machine learning, so the calibration suite can put it next to GAMEPLAI's prod
and candidate.

## How it works

It works top-down: the whole game first, then drives inside it.

1. **Pre-match anchor** (`strength.py`, `pricer.Model.fit_prior`). The
   pre-match spread and total give each side its expected points. When the
   pre-match probabilities are available, the anchor is fitted to them too,
   so at kickoff the model quotes the pre-match moneyline and total
   probability back exactly.
2. **Moving off the anchor as the game goes** (`strength.py`). Each side's
   scoring rate has a Gamma prior with mean 1 and shape k. The scoreboard
   updates it: points scored against points expected by now, counted in
   scores of about 6.2 points. k is the setting that separates the
   versions: a large k trusts the pre-match number, a small k trusts the
   game. Only the scoreboard feeds it, never drive counts, so a drive the
   feed mis-splits can't move a strength.
3. **The rest of the game as an exact possession Markov chain**
   (`pricer.py`). The remaining drives alternate: the current drive, then
   the defense, then the offense, and so on. Possession resets at halftime
   to whoever didn't receive the opening kickoff. The number of drives
   still to start follows a discretised normal whose variance combines
   the game's own spread (`count_dispersion`) and the uncertainty from
   having no clock. Everything is combined by convolution into the full
   final-score distribution, and all six markets are read off that. There
   is no simulation, so the same state always gets the same price, and a
   price moves only when the state does.
4. **The current drive as a play-by-play Markov chain** (`drive.py`). The
   state is down, distance and field position, and each snap is a
   turnover, a loss, no gain, a gain or a big play. On 4th down the chain
   kicks a field goal in range, goes for it on 4th and short, and punts
   otherwise. It is solved backwards, exactly, into P(TD) and P(FG) from
   every state, on a grid of offensive quality. Quality is set so a fresh
   drive is worth the side's expected points per drive. The chain supplies
   the shape of in-drive moves; the game level sets the height. A first
   down, a big gain or a short 4th down lifts the offense's price. A sack,
   a stall or a long 3rd down drops it. The total moves with the drive.
5. **The clock** (`clock.py`). The feed has no game clock, only period and
   message count. Each period gets its own length in messages (mean and
   spread) and its own share of the game's scoring drives, because the 4th
   quarter scores far less than the others. Time left in the current
   period is the conditional expectation given how long it has run. The
   current drive can also be cut off by the end of the half (`drive_time`).
6. **Suspension** (`feed.py`). A causal tracker suspends:
   - from the opening kick to the first snap
   - from any score to the next drive's first snap (the extra point, the
     kickoff and the return; an onside recovery is picked up)
   - from each new half to its first snap
   - kick-spot rows
   - stale label flips (the possession changed but the spot wasn't re-read)
   - impossible down jumps

   Everything else is priced live. The calibrator's own cleaning looks
   one row ahead, so it can't be used for a live price.

Monte Carlo would add noise that moves prices without the game moving, so
the model uses exact chains instead of simulation.

## Where the numbers come from

| constant | value | source |
|---|---|---|
| drives per team | 4.80, count sd 2.14 | max likelihood on 23,952 AF finals (`nb2/AMFELO.csv`), `fit.py` |
| TD / FG per drive | 45.1% / 10.3% | same fit; directional_pairs drives read 46% / 11% directly |
| conversions | 6 only 5.1%, 8 4.3% | same fit |
| overtime | won by a field goal | same fit (ot_fg at its bound) |
| messages per period | 71, 122, 81, 120 | period boundaries in 262 matches of directional_pairs |
| period scoring shares | .27 .30 .25 .18 | points per quarter in the same matches; Q4 scores about half |
| play kernel | mean gain 10, 18% big plays | calibrated so a fresh drive from the 25 scores ~46% TD / 11% FG |
| drive_time | 1.0 | swept on a train half of the matches, checked on the other half |

The normal drive count replaced a negative binomial, which can't go below
Poisson dispersion. That overstated the total's spread (sd 14.8 against
12.9 in the data) and cost 563 log-likelihood points.

## Versions

| | prior strength k | reading |
|---|---|---|
| v1 | 16 | anchored: four TDs above expectation move a side about 25% |
| v2 | 4 | reactive: the game takes over about four times as fast |

Add a version by naming what differs in `params.VERSIONS`.

## Backtest (262 matches, directional_pairs, 17–20 Sep)

Brier score, lower is better. d is prod's Brier minus the version's, so
positive means the version beat prod. The 95% interval is a match-level
bootstrap. The model prices at prod's line, from the same state.

| market | prod | candidate | v1 | v2 | d v1 [95% CI] | d v2 [95% CI] |
|---|---|---|---|---|---|---|
| moneyline | 0.1595 | 0.1629 | 0.1601 | 0.1638 | −0.0006 [−0.0030, +0.0023] | −0.0043 [−0.0086, +0.0003] |
| spread | 0.2484 | 0.2476 | 0.2314 | 0.2396 | +0.0170 [+0.0097, +0.0247] | +0.0088 [−0.0018, +0.0199] |
| total | 0.2473 | 0.2479 | 0.2357 | 0.2374 | +0.0117 [+0.0064, +0.0168] | +0.0099 [+0.0023, +0.0156] |
| all | 0.2184 | 0.2195 | 0.2090 | 0.2136 | +0.0094 [+0.0052, +0.0126] | +0.0048 [−0.0001, +0.0097] |

Read it with care:

- **Moneyline is the clean comparison.** It is live throughout. v1 is
  level with prod (−0.0006, the interval straddles zero) and ahead of the
  candidate (0.1629). Q1 is slightly behind prod (−0.0046); Q4 and
  overtime are ahead.
- **The spread and total wins are overstated.** This file predates the
  liveness columns, and late in the game prod's spread and total quotes
  are mostly suspended rows that sit near 50% whatever the line (home up
  25, line −2.5, prod 44.5%). Nearly all of the gain is in Q4 (spread
  +0.071, total +0.045). In Q1–Q3, spread and total are level with prod
  within noise. Rerun the calibrator: its pairs file now carries
  `prod_live` / `candidate_live`, the backtest drops suspended pairs
  automatically, and `report --candidate v1` does the same.
- **Trusting the scoreboard more cost Brier.** At kickoff-fitted priors:
  k = ∞ scored 0.1593 on the moneyline (prod 0.1593), k = 16 scored
  0.1601, k = 4 scored 0.1638, and k = 2 was worse again. The score
  itself already moves the price (a lead is a lead). What didn't help was
  also re-rating each side's scoring rate off the points. Over 262
  matches, how a gamer has played so far says little about how they will
  play the rest of the game, beyond the score. That is why v1 is the
  anchored one.
- **Two changes carried the model from 0.2289 to 0.2090 overall.** The
  first was per-period scoring shares (Q4 scores about half). The second
  was letting the end of a half cut the current drive short. The shares
  come from points per quarter; fitting them to outcomes overfitted on a
  held-out half, so the plain ones stayed. `drive_time` was swept on half
  the matches, and its gain held on the other half (0.2186 → 0.2115,
  against prod's 0.2186).

## Run it

```bash
# score versions against prod and candidate on a calibrator pairs file
python -m eAMFModel backtest eAMFCalibrator/out/directional_pairs.csv --versions v1,v2

# walk one match message by message: the tracker's verdict, the model's prices, prod's
python -m eAMFModel trace eAMFCalibrator/out/dump_play_by_play.csv

# one state by hand
python -m eAMFModel price --spread -2.5 --total 38.5 --period 2 --elapsed 30 \
    --score 7-7 --offense home --down 1 --distance 10 --field 60 --opening home

# the whole calibration report with a version standing in for the candidate
python -m eAMFCalibrator report --candidate v1

python -m unittest eAMFModel.tests.test_model
```

`--candidate v1` makes the calibrator price every match with the model, off
prod's pre-match quotes, the play feed and the scores. It prices at prod's
lines, so every section compares like with like. Pure Python, no new
dependencies.

## Known gaps

- **No game clock.** The late 4th quarter is where prod clearly knows how
  long is left and the model doesn't. Messages per period vary a lot
  between matches, and nothing in the feed says why.
- **The play kernel is calibrated, not estimated.** It is tuned to drive
  outcomes because the local data has one snapshot per drive. The full
  play feed in Snowflake would let the gain distribution be read off the
  snaps directly.
- **No end-game behaviour.** Nothing models kneel-downs, hurry-up or going
  for two when chasing, beyond the scoring shares and the half-end cut.
