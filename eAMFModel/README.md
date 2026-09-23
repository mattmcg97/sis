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
5. **The clock, by halves** (`clock.py`). The feed has no game clock, only
   period and message count. Q2 continues Q1 and Q4 continues Q3, so time
   is counted through each half. Scoring falls smoothly across a half
   instead of stepping down at the quarter break: the share of a half's
   scoring still to come at game-time fraction u is (1 − u)^(1 + a). The
   first half is near flat; the second falls (a = 0.5). The quarter break
   anchors game time at u = 0.5, because it's a real game-time point even
   though Q2 and Q4 run far more messages. Within a quarter, u moves with
   messages over the quarter's uncertain length. The drive in progress
   follows the second half's falling rate too, not just the count of
   drives left.
6. **The end of each half** (`pricer.py`). The current drive can be cut
   off by the clock. The clock left is measured in messages (time, not
   scoring share), and the time a drive needs scales with the field it
   still has to cover. A drive on the 1 needs a snap or two; one from its
   own 20 needs a full drive's worth.
7. **End of the game** (`pricer.Model._endgame`). With under 3 drives of
   clock left in the second half:
   - behind by 1–3 or by 9+, the trailing side never punts (it still
     kicks when a field goal helps);
   - behind by 4–8, it never punts and never kicks, because three points
     don't catch up;
   - ahead by 9+ with the ball, the leader kneels, scoring 80% less.

   The drive chain is solved for each 4th-down policy.
8. **Suspension** (`feed.py`). A causal tracker suspends:
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
| messages per quarter | 71, 122, 81, 120 | quarter boundaries in 262 matches of directional_pairs |
| half shares, slopes | 57% / 43%; a = −0.2, 0.5 | fitted on half the matches, checked on the other |
| play kernel | mean gain 10, 18% big plays | calibrated so a fresh drive from the 25 scores ~46% TD / 11% FG |
| drive cut-off | needs 1.2 drives of clock, spread 0.8 | fitted on half the matches, checked on the other |
| end game | 3 drives of clock left; kneel 80% less | same |

The normal drive count replaced a negative binomial, which can't go below
Poisson dispersion. That overstated the total's spread (sd 14.8 against
12.9 in the data) and cost 563 log-likelihood points.

## Versions

| | prior strength k | reading |
|---|---|---|
| v1 | 16 | anchored: four TDs above expectation move a side about 25% |
| v2 | 4 | reactive: the game takes over about four times as fast |

Add a version by naming what differs in `params.VERSIONS`. v3 is a separate model (below).

## Backtest (262 matches, directional_pairs, 17–20 Sep)

Brier score, lower is better. d is prod's Brier minus the version's, so
positive means the version beat prod. The 95% interval is a match-level
bootstrap. The model prices at prod's line, from the same state.

| market | prod | candidate | v1 | v2 | d v1 [95% CI] | d v2 [95% CI] |
|---|---|---|---|---|---|---|
| moneyline | 0.1595 | 0.1629 | 0.1602 | 0.1639 | −0.0007 [−0.0034, +0.0020] | −0.0044 [−0.0083, −0.0001] |
| spread | 0.2484 | 0.2476 | 0.2322 | 0.2400 | +0.0162 [+0.0099, +0.0235] | +0.0084 [−0.0024, +0.0189] |
| total | 0.2473 | 0.2479 | 0.2371 | 0.2387 | +0.0103 [+0.0032, +0.0174] | +0.0086 [−0.0005, +0.0155] |
| all | 0.2184 | 0.2195 | 0.2098 | 0.2142 | +0.0086 [+0.0037, +0.0121] | +0.0042 [−0.0014, +0.0097] |

Read it with care:

- **Moneyline is the clean comparison.** It is live throughout. v1 is
  level with prod (the interval straddles zero) and ahead of the
  candidate. Q1 is slightly behind (−0.0045); Q4 and overtime are ahead.
- **The spread and total wins are overstated.** This file predates the
  liveness columns, and late prod spread and total quotes are mostly
  suspended rows near 50% whatever the line. Nearly all of the gain is
  in Q4. Q1–Q3 are level with prod within noise.
- **Most of this file's snapshots sit on garbage rows.** 1,378 of 2,615
  are a 1st & 10 on the 35: the opening kick, or the receiving side's
  label flipped onto the kick spot straight after a score. In the full
  play-by-play the real drive start follows on the 20–30. The current
  calibrator drops those rows, and now also refuses any quote sitting on
  a conversion, kick or score message. A fresh export will be a cleaner
  test.
- **Halves and end-game rules versus the earlier quarter clock.** On
  held-out matches it's a tie (Brier 0.2123 vs 0.2120):
  better Q4 totals (0.197 vs 0.206), slightly worse Q4 moneyline. A
  clock counting only messages since the half began lost to both
  (0.2140). The quarter break carries real game-time information.
- **Trusting the scoreboard more cost Brier.** On the moneyline: k = ∞
  0.1593, k = 16 0.1601, k = 4 0.1638. The score itself already moves
  every price. Re-rating each side's scoring rate off its points on top
  of that didn't help. That is why v1 is the anchored one.

## PLAY_OVER snapshots and the real game clock

`SCOUTING_FULL` has the game clock (`IN_PLAY_CLOCK_SECONDS`, 240 a
quarter), and `PLAY_OVER` is the message snapshots are keyed on. With the
clock known (`GameState.clock_seconds`):
- The drives left are a matter of time: this half's seconds in drives'
  worth, plus a whole half more in the first half.
- The second half's lower scoring comes off what each drive is worth. Its
  drives score at `half_shares[1] / half_shares[0]` of the first half's,
  and fall through the half at w(u).
- The spread of the drive count carries the current drive's own length:
  a quick turnover hands the ball back with time on the clock.
- The prior is fitted on this path (`fit_prior(on_clock=True)`).

A touchdown's `PLAY_OVER` is priced with the conversion still to come
(`pending_conversion`) and a fresh drive for the other side. Its points
(or a good field goal's) are added if the scoreboard hasn't caught up with
them yet.

```bash
python -m eAMFCalibrator scouting                            # writes scouting_playover.csv
python -m eAMFModel playover eAMFCalibrator/out/scouting_playover.csv --versions v1,v2
python -m eAMFModel playover ... --scrimmage-only            # leave out kicks, conversions, scores
python -m eAMFModel playover ... --state over                # the PLAY_OVER row's own state
```

Only prod quotes that were live are compared. The kind of play a snapshot
closed is its own breakdown (`--by kind`).

The clock-path settings (half shares and slopes, drive cut-off, end-game)
are carried over from the message-clock fit. They should be refitted on
play-over data once there is an export to fit them on.

## v3: a play-by-play simulation

v3 is its own model (`sim.py`, `v3.py`), not a setting of v1/v2. From the
snapshot's state it plays the rest of the game snap by snap on the real
clock, 2,000 times, and prices the markets off the final scores.

**Every snap is drawn from a real one** in the same situation: down,
distance bucket, field zone and the offense's game situation. The
situations are:
- Q1;
- Q2;
- Q2's last two minutes;
- second half, ahead;
- second half, level or behind;
- the last two minutes, split into ahead, behind and level.

A drawn snap brings its yards, turnover or touchdown, and the clock it
used through to the next snap. So a leader milks the clock because
leaders do: about 19 s a snap in the third quarter, against 15.5 s level
or behind. A trailer's last-minute incompletions stop it. A touchdown's
gain is cut off at the goal line, so from further back the same play
still scores with probability exp(−extra yards / 12).

**Efficiency, not score.** Each offense has an efficiency θ that tilts its
draws toward the good end: u → 1 − (1 − u)^e^θ. That turns the league's
first-down rate f into f^e^−θ.
- The **prior** θ for each side is set so that the simulation from kickoff
  reproduces prod's pre-match spread, total and moneyline
  (`PriorGrid.fit`, from a 17 × 17 grid of simulated games).
- **In-game** (`--react`), θ moves with each snap's first-down success
  against the league's rate in that situation. That is one Newton step,
  shrunk by κ. Points never move θ.

**Decisions**, all fitted to real snaps:
- 4th down: the league's go / field-goal / punt curves, shifted by quarter
  phase × score margin. Leaders go less; trailers go for it late.
- Going for two: by phase × margin. Trailing by 7, a player who scores late
  goes for two to win 65% of the time.
- Early-down field goals as a half runs out, by clock and kick distance.
- Level or 1–2 behind late and in range: run the clock down and kick as
  time expires.
- A leader with the downs to cover the clock kneels.
- Clock management leaves time for the kick.
- Onside kicks when chasing.
- Overtime is a timed period, replayed while level.
- The league's efficiency by quarter is fitted so the simulation scores
  what the league scores in each quarter (5.5 / 13.1 / 6.7 / 9.2).

**Held-out test** (the 1,160 matches not used to build the tables, 410,602
live prod quotes; Brier, prod − v3, 95% CI clustered by match):

| | prod | v1 | v3 | v3 − prod |
|---|---|---|---|---|
| moneyline | 0.1656 | 0.1647 | 0.1629 | +0.0026 [+0.0017, +0.0036] |
| spread | 0.2474 | 0.2490 | 0.2345 | +0.0128 [+0.0110, +0.0146] |
| total | 0.2475 | 0.2431 | 0.2367 | +0.0109 [+0.0092, +0.0124] |
| all | 0.2184 | 0.2172 | 0.2098 | +0.0086 [+0.0076, +0.0096] |

v3 is ahead of prod in every quarter × market. The gain is largest in Q4:
spread +0.043 and total +0.034. Q1–Q2 gain +0.001 to +0.002, and the
overtime moneyline (23 matches) is level. Adding each player's 4th-down
aggression and pace (`--profiles`, where handles are known) takes the
total to +0.0119.

**What did not help.** The in-game efficiency update (`--react`) matches
static v3: first-down success predicts later first-down success only
weakly (best κ ≈ 80–200 snaps' worth of information), and not later
points. Drawing θ with its uncertainty didn't help either.

```bash
python -m eAMFModel v3-build eAMFCalibrator/out/scouting_playover.csv --half train --out v3_model
python -m eAMFModel v3 eAMFCalibrator/out/scouting_playover.csv --model v3_model --half test
python -m eAMFModel v3 ... --react --profiles player_profiles.json --handles nb2/AMFELO.csv
python -m unittest eAMFModel.tests.test_v3
```

In the calibrator, v3 stands in for the candidate with
`python -m eAMFCalibrator report --candidate v3 --v3-model v3_model` (see
the calibrator README): `v3_stream.py` turns its prices into
GAMEPLAI-shaped quote rows.

v3 needs numpy. The build takes about a minute; scoring the 1,160-match
test half takes about 18 minutes on four cores at 2,000 paths.

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
- **End-game behaviour is rule-based.** Policies switch on a threshold of
  clock left, and the kneel is a flat reduction. There is no hurry-up
  clock management, no timeouts and no going for two when chasing. The
  local data (about 2.5 Q4 snapshots per match) can't separate finer
  rules; the full play feed could.
