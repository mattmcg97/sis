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

v3 needs numpy (`py -m pip install numpy` on Windows); v1 and v2 do not. The build takes about a minute; scoring the 1,160-match
test half takes about 18 minutes on four cores at 2,000 paths.

## v4: v3 plus a pre-match total correction and smoother prices

v4 is a copy of v3 (`sim4.py`, `v4.py`, `v4_stream.py`) with v3's files left
as they are. It came out of re-checking the first v3 calibration report
(17–23 Sep). All of that report's figures reproduce from its own pair rows,
and its weak spot was clear: totals leaned over before Q4, most in Q3
(realised 0.438, prod 0.489, v3 0.531).

**Where the lean comes from.** It isn't the simulation. v3 takes each
match's scoring level from prod's pre-match total: it fits the efficiencies
so that its own P(over) at prod's line equals prod's. Through September
prod's pre-match total fell behind the scoring:

| matches | over rate at prod's pre-match line | prod's P(over) |
|---|---|---|
| before 10 Sep | 0.497 | 0.500 |
| 10–16 Sep | 0.466 | 0.501 |
| 17–23 Sep | 0.448 | 0.503 |

v3 inherits that miss for the whole game.

**What v4 changes:**
- **Pre-match total correction** (`recent_total_shade`). At build time,
  v4 measures how far games in the last 7 days went over prod's pre-match
  line against the P(over) prod priced them at. It shrinks that toward 0 by
  200 matches' worth of evidence, and adds it to prod's P(over) before
  fitting each match's efficiencies. The build prints it: for a model built
  to 16 Sep, 476 matches went over 46.6% of the time at a priced 50.1%,
  giving a correction of −0.024.
- **Common random numbers** (`sim4.py`). Each simulated path draws from its
  own stream, keyed on the path and the number of events since the
  snapshot. Snapshots priced together play out the same luck, so a price
  moves because the game moved, not because of Monte Carlo noise. The
  noise in the difference between two neighbouring states falls from
  0.018 to 0.006.
- **Player profiles** (pace and 4th-down aggression, `players.py`). They're
  built with the model from the export's player handles and applied when a
  match's handles are known. On the Sep 3–9 check they came out neutral
  (totals +0.0002, the rest −0.0001).
- **In-game check.** The build prints the points real games scored from
  each quarter's first `PLAY_OVER`, against what the simulation expects.

**Tried and dropped: fitting the quarters to real in-game states.** From
real states, v3 expected 0.3–0.6 points too many at every quarter. Fitting
the league's efficiency by quarter to those states helped once, but
alternating it with the prior fit never settled. Each pass lowered the
quarters and the pre-match fit raised the efficiencies back to match
prod's total. The in-game excess comes from the pre-match level, which the
correction above addresses.

**Results.** v4 is built on everything before a week and scored on every
`PLAY_OVER` of that week; Brier, prod − version:

| week | market | v3 | v4 |
|---|---|---|---|
| 10–16 Sep (correction −0.004) | all | +0.0087 | +0.0091 |
| 17–23 Sep (correction −0.024) | all | +0.0073 | +0.0078 |
| | total | +0.0102 | +0.0117 |
| | Q3 total | −0.0012 | +0.0014 |
| | moneyline | +0.0015 | +0.0016 |
| | spread | +0.0107 | +0.0107 |

In the 17–23 Sep week, v4's P(over) is 0.474 / 0.480 / 0.505 / 0.401 by
quarter (v3 0.490 / 0.493 / 0.515 / 0.405; realised 0.443 / 0.439 / 0.412 /
0.348). The correction closes about half of that week's miss, as far as
the week before could support.

```bash
python -m eAMFModel v4-build eAMFCalibrator/out/scouting_playover.csv --half all --out v4_model
python -m eAMFModel v4 eAMFCalibrator/out/scouting_playover.csv --model v4_model --half test
python -m eAMFCalibrator report --candidate v4 --v4-model v4_model --out eAMFCalibrator/out_v4
python -m unittest eAMFModel.tests.test_v4
```

The correction is read from the last 7 days of whatever the model is
built on, so rebuild weekly for it to track. Build on an export that ends
where the report window starts.

### Our own pre-match: NB2 (`--history`)

Built with `--history`, v4 takes each match's scoring level from our own
pre-match model, the NB2 player + team model in `nb2/` (`nb2_prior.py`),
and never reads GAMEPLAI's pre-match quotes. NB2 prices the kickoff and
the simulation takes over from the first `PLAY_OVER`.

- **Fit.** `nb2/NBRatingTrial.py` runs on the match history before the
  build's cut-off: every settled match up to the day after the last one
  the tables are built on, or `--before`. It fits recency-weighted player
  and team attack/defence, stream effects and the score correlation, with
  a 60-day half-life.
- **Level.** NB2's totals ran about 3% under the real ones, week after
  week. Each week was fitted only on what came before: ratios 1.027,
  1.028, 1.035. The build measures that on the last 7 days, walk-forward,
  shrinks it toward 1 by 200 matches, and scales both sides' expected
  points by it. The build prints the figure.
- **Prior.** `nb2/NB2_schedule_predict.py` gives each side's expected
  points. v4 picks the home and away efficiencies whose simulated games
  average those points (`fit_means`). A match NB2 can't price (an unknown
  player or team) gets the league's average of the last 30 days.

Everything lands in `v4_model/nb2/`, and `nb2/` itself is never written to.
The two scripts need pandas and scipy (`py -m pip install pandas scipy`).

**The seed.** Each match simulates off a fixed seed, a hash of its match
code, on top of the common random numbers above. Re-pricing a match gives
the same prices every time, and the same state always gets the same price.
Two neighbouring states differ only by what happened between them. A fixed
seed alone would not do this: without common random numbers, one extra
play would shift every path's draws, and prices would jump just the same.

**What still touches GAMEPLAI.** For an NB2 model, nothing in the prices:
- the simulation tables come from `SCOUTING_FULL` alone;
- the prior comes from NB2;
- the pre-match correction above is set to 0.
Only the comparison reads prod. The calibrator quotes v4 at the line and
message of the prod quote it pairs with, and grades on the same matches, so
both answer the same question. A model built without `--history` still
falls back to prod's pre-match quotes, as before.

**Results** (walk-forward: built on everything before 3 Sep, scored on
every `PLAY_OVER` of 3–9 Sep; Brier, prod − version):

| market | v4, prod's pre-match | v4, NB2 |
|---|---|---|
| moneyline | +0.0027 | +0.0056 |
| spread | +0.0114 | +0.0148 |
| total | +0.0125 | +0.0103 |
| all (202,520 quotes, 583 matches) | +0.0087 | +0.0101 |

NB2's level was scaled 1.021 for that week. It gains on moneyline and
spread, where prod's pre-match had no edge to give. It gives back a little
on totals, where the prod version carried the pre-match correction.

```bash
python -m eAMFCalibrator history --until 2026-09-24          # out/match_history.csv
python -m eAMFModel v4-build eAMFCalibrator/out/scouting_playover.csv --half all --out v4_model --history eAMFCalibrator/out/match_history.csv
python -m eAMFModel v4 eAMFCalibrator/out/scouting_playover.csv --model v4_model --half test --history eAMFCalibrator/out/match_history.csv
```

`--history` takes any CSV shaped like `nb2/AMFELO.csv`. For `v4`, it needs
only the priced matches' players, teams and stream; the finals are not read.

### Its own lines

As a stream (`--candidate v4`), v4 quotes its own line on the spread and
the total, not prod's. At every snapshot it takes the half-point line
nearest even money off its own distribution for the state: the margin's
for the spread, the total's for the total. The line moves when the game
moves it. The calibrator then pairs the two streams as it pairs any two:
where the lines are the same, it compares the probabilities; where they
differ, it scores whose line landed nearer the result.
`--v4-lines prod` goes back to reading v4's price at prod's line.

On 3–9 Sep (built before 3 Sep, NB2 prior, 583 matches, live quotes):

| | spread | total |
|---|---|---|
| v4's line same as prod's | 38% | 23% |
| mean distance from the final result, v4 / prod | 5.09 / 5.22 | 6.46 / 6.49 |
| different-line pairs won by v4 | 53.7% | 49.9% |
| Brier at each stream's own line, v4 / prod | 0.2386 / 0.2473 | 0.2384 / 0.2474 |
| Brier at prod's line, prod − v4 | +0.0144 | +0.0108 |

v4's lines are nearer the result on the spread in every quarter, and most
of all in Q4 (2.83 against 3.09). Totals are level with prod's: slightly
better in Q1, Q3 and Q4, slightly worse in Q2 (7.60 against 7.52). The
last row is the old same-line comparison, unchanged by the goal-line fixes
below.

### Backed up on the goal line

A leader late with the ball on their own 1–3 won 88.9% of the time in real
games (27 cases), but v4 gave them 98.3%. Two things were missing:
- **The kneel.** v4 knelt out the clock from anywhere. A kneel loses a
  yard, so on the goal line it's a safety. v4 now only kneels with the room
  to take the kneels it needs; backed up, it runs real plays.
- **The free kick.** After a safety, the kick comes from the 20. Real
  receivers start about their 42, against a kickoff's 26. v4 now draws
  from real free kicks, or a kickoff moved on 16 yards when fewer than 20
  have been seen.

Now v4 gives the leader 94.4%. That's within the noise of 27 cases.
Everywhere else on the field it's unchanged, and matches what happened:

| ball on | cases | leader wins, v4 | real |
|---|---|---|---|
| own 1–3 | 27 | 0.944 (was 0.983) | 0.889 |
| own 4–10 | 55 | 0.957 | 0.964 |
| own 11–30 | 305 | 0.938 | 0.934 |
| beyond | 1,175 | 0.949 | 0.941 |

Rebuild the model (`v4-build`) to pick up the free-kick table.

## v5: v4 plus play calling by game state

v5 is a copy of v4 (`sim5.py`, `v5.py`, `v5_stream.py`; v4 untouched). It
tests three ideas about how players really play:
- run and pass;
- clock bleed;
- the rubber band.

**What the games say.** These are real games only, from Aug 24 to Sep 22.
In the last 1:20–2:40 of Q4, a leader by 1–8 with the ball leaves 4.2
points to be scored; with the trailer on the ball it's 7.2. In Q3 the
leader with the ball scores less in the rest of the quarter, but the
rest-of-game totals are almost the same (15.8 against 16.0 early in Q3).
Leaders use 24–26 s a snap in Q3, and teams trailing by 9+ use 18 s.

**Run or pass.** The feed doesn't say which, but it does say what a play
did to the clock. From one snap to the next, a play that stopped the clock
(an incompletion, out of bounds: mostly passes) takes about 4–10 s. One
that kept it running (runs, completions in bounds) takes about 30–40 s,
the play plus the time to pick the next one. v5 splits every bin's plays
that way and decides the call from the game state: the quarter, which
40-second slice of it, and the offense's lead. It then draws the yards
from real plays of that kind.

**Clock bleed.** Within v4's bins, real snaps used 2–4 s more or less
clock than the bin by state. Examples: Q3 mid-quarter level or ahead by
1–8, +2.6 to +3.9 s; the last minute of Q2, about −2 s. v5 shifts each
kind of play's clock by state. The shifts are fitted on snaps with at
least 60 s left in the quarter. Nearer the buzzer, the data only keeps
plays whose next snap came before it, so their clock looks short.

**The rubber band.** v5 gives each state an efficiency shift from real
first-down success. It also fits a pull per half, so that from real
in-game states the simulated share of a lead that comes back by the end
matches the real one. v4 brought back 12.2% of each point of second-half
lead against a real 14.8% (held out, Sep 10–22); v5 matches it. The
pull came out at −0.09 per score in the first half, where the state
shifts already pulled too hard, and +0.09 in the second.

**The fourth quarter** is left to v4's end-game tables. On held-out games
every shift cost a little there, and all three together cost
significantly: spread −0.0017 and total −0.0024 Brier.

**Results.** v5 was built on games before Sep 3 and scored on held-out
games; Brier, v5 − v4:

| held-out games | moneyline | spread | total |
|---|---|---|---|
| Sep 3–9, 583 matches, NB2 pre-match | −0.0001 | +0.0000 | +0.0003 |
| Sep 10–22, 453 matches, prod's pre-match | −0.0006 | −0.0005 | +0.0001 |

All within the noise (95% intervals by match span about ±0.001). By
quarter, nothing is significant beyond Q4 spread on Sep 3–9 (+0.0005).
Across the states, v5 has:
- a 15.4% / 12.7% rubber band, matching real games in both halves;
- Q3 totals from level scores closer to real (+1.04 against v4's +1.37
  points too many);
- a big lead's movement in Q3 fixed (trailing by 17+: −1.21 to −0.01);
- but it over-reverts some middle buckets (Q3 leading by 9–16: −0.85).

v5 does what it was built to do in the states it targets. It's level with
v4 overall: no quarter or market moves beyond the noise. v4's bins
already carry most of this behaviour, because they split plays by
situation (the leader milking, the trailer hurrying, the last two
minutes).

### Backed up: safeties, turnovers and touchdowns from the own 1–10

A snap's yards come from its bin, and the first field zone runs from the
1 to the 39. So a one-yard loss seen at the 30, drawn at the 1, became a
safety. Also, the snaps that really went for a safety never reached the
bins, because the feed shows them as the free kick that follows. Per snap,
on real states from SCOUTING_FULL (Aug 24 to Sep 22):

| ball on own | snaps | safety real / v5 before / now | defensive TD real / before / now | TD real / before / now |
|---|---|---|---|---|
| 1 | 136 | 5.1% / 12.4% / 5.1% | 0.7% / 0.1% / 1.1% | 3.0% / 1.2% / 3.2% |
| 2 | 84 | 4.8% / 7.4% / 4.5% | 1.2% / 0.2% / 1.4% | 3.7% / 1.1% / 3.7% |
| 3 | 108 | 3.7% / 5.2% / 3.5% | 2.8% / 0.2% / 2.4% | 3.7% / 1.3% / 3.7% |
| 4 | 99 | 1.0% / 3.5% / 1.6% | 2.0% / 0.3% / 1.9% | 3.2% / 1.3% / 3.3% |
| 5 | 130 | 1.5% / 2.9% / 1.7% | 2.3% / 0.2% / 2.1% | 4.0% / 1.3% / 3.9% |
| 6–10 | 667 | 0.9% / 2.1% / 1.0% | 0.9% / 0.3% / 1.0% | 3.4% / 1.3% / 3.3% |

Before this, v5 made twice the real safeties on the 1 and 2, a tenth of the
defensive touchdowns (a fumble in the end zone), and under half the
long touchdowns. Turnovers that stay in the field of play were about right
(3.5% real against 3.8% simulated on the 1–5). The "now" rates are the
fitted ones, which the simulation draws exactly. They're fitted on the
same games, so they're in-sample.

Now, inside the 10 each snap first draws one of these outcomes from that
yard line's own rate:
- a safety;
- a defensive touchdown;
- a turnover;
- a touchdown;
- none of these.

Only a snap that comes to none of them takes its yards from the bin, and
it stays in the field of play. Each yard line's rate is pulled toward a
logistic curve in the yard line by 60 snaps' worth, since a yard line sees
about 100 snaps a month. A turnover's new spot comes from real backed-up
turnovers. `v5-build` prints the rates on the 1 to 5.

After a safety, v5 adds 2 points to the defence. The conceding side then
free-kicks from its 20, and the side that scored the safety starts from
real post-safety spots, about its own 43. That's the chance of a second
score: in the 31 real safeties, the receiving side scored a touchdown on
the next drive in about 3 in 10.

Held out on Sep 3–9 (583 matches, NB2 pre-match), overall Brier moved by
less than 0.0002 in every market. On snapshots with the ball on the own
6–10, spread improved by 0.0034 (95% interval +0.0008 to +0.0059).
Moneyline improved by 0.0011 and total by 0.0014, both within the noise.
On the own 1–5 (about 120–160 snapshots per market) every change is within
the noise. A leader in the last three minutes on their own 1–5 now wins
94.2% of the time (was 94.8%, real 93.2%, 44 cases).

**The second-half kick.** The opening receiver kicks off the second half.
That holds in 2,294 of the 2,320 matches in SCOUTING_FULL, and v5 already
did it. Where a snapshot doesn't know who received the opening kick, v5
used to have home kick. Now each path tosses a coin.

### The rubber band by quarter

**What the report showed.** In the report week (Sep 17–24), v4 and v5
lines through Q3 went over only 41–44% of the time at about 50% priced.
That held in every game state and at every level of points already
scored. Q4 was close to priced, and Q1 was fine.

**Where it came from.** On held-out games, by leader and trailer on the
ball:

| state | lead change to the end, real / v5 | points still to come, real / v5 |
|---|---|---|
| Q3, trailer has ball | −3.16 / −3.62 | 13.1 / 13.9 |
| Q3, leader has ball | +0.65 / +0.24 | |
| Q4, trailer has ball | −3.24 / −2.97 | 5.9 / 5.4 |
| Q4, leader has ball | +0.53 / +0.63 | |

(Sep 3–9; Sep 10–22 shows the same.) v5's third-quarter trailers came
back too much and scored too much, and its fourth-quarter trailers too
little. The band was fitted as one pull per half, but applied only in
Q1–Q3. So the whole second half's comeback was pushed into the third
quarter.

**Now** the band has three pulls: the first half, Q3 and Q4, the fourth
included. They're fitted one at a time from the last, each on its own
states with the later pulls held. Solved together, a Q1 lead's fate runs
through every later pull, and the pulls chased each other. A separate
pull for Q1 came out wild off its few real leads. Each state plays its
own luck, the same for every trial pull (`simulate(..., distinct=True)`).
Before, the fit shared one set of draws across all states. In the build
built on games before Sep 3, the pulls came out +0.05 / −0.05 / +0.05.
The old ones were −0.11 for the first half and +0.12 for the second.

**Held out**, Brier against the same build with the old band:

| | moneyline | spread | total |
|---|---|---|---|
| Sep 3–9 | +0.0002 | −0.0004 | +0.0004 (Q3 **+0.0024**, CI +0.0008 to +0.0044) |
| Sep 10–22 | **+0.0003** (Q1 +0.0013) | +0.0003 | 0.0000 |

On Sep 3–9, Q3 points still to come dropped from 0.36 too many to 0.16.
With the trailer on the ball two scores down, the miss went from +1.15
to +0.28. No market got significantly worse in either week.

**Tried and left off: the quarters' levels.** v5's kickoff fit reads each
quarter's points off its last row. That misses the quarter's last score,
which the feed posts on the next quarter's first row: Q1 comes out 0.65
short and Q4 0.5 long. Counted properly, and with Q2's two-minute drill
given a level of its own (it scores about 9 of the quarter's 13 points),
the quarter-start fit (`QUARTER_START_FIT`, `fit_quarter_levels`)
matched every quarter. But held out it cost moneyline and spread about
0.001 in both weeks, with no gain on the total. The late-game margin
swings lean on those extra fourth-quarter points. So the kickoff fit
keeps its old reading, and the fit is left in the code, off.

### Form on the day

**What was wrong.** v5 played every simulated game with both offenses
fixed at the match's prior, so all of a snapshot's spread came from the
plays. Held out, the points still to come were too narrow:

| | real results in the bottom / top tenth of v5 | spread, real / v5 |
|---|---|---|
| Sep 3–9 | 10.5% / 10.9% | 8.52 / 8.13 |
| Sep 10–22 | 13.0% / 10.8% | 9.26 / 8.23 |

**Where the missing spread is.** Over a year of finished matches, with
NB2's expected points for each side, next to the simulation with the
strengths fixed:

| | real | v5, strengths fixed |
|---|---|---|
| each side's variance | ~59 | ~53 |
| home/away covariance | ~15 | ~9.5 |
| margin variance | 88 | 87 |
| total variance | 148 | 125 |

The margin is already as wide as real games; the total is not. So what
is missing is mostly a swing that both sides share, where a game scores
more or less as a whole. Some players also swing more than the
simulation gives them on their own.

**Now** each simulated game draws two things on the log scale of
points, both on common random numbers (path k plays the same draw at
every snapshot of a match):
- **the game:** one normal draw that moves both offenses together;
- **each player's own form:** a normal draw for each side, with a
  standard deviation for each player.

Each draw is converted to theta by how many log points one unit of theta
is worth at the side's strength (`tables.strength_slope`). The start is
moved so that the draws leave each side's expected points where they
were (`sim5.strength_draw`).

**The fit (`fit_form`).** It uses every finished match in the year
before the build, weighted by NB2's own 60-day half-life, with NB2's
expected points for each side:
- the simulation's own variance and covariance with the strengths fixed
  are measured from kickoff (`fixed_strength_spread`) and taken off;
- what's left over in the covariance is the game's share;
- what's left over in each side's variance, beyond the game's, is the
  player's own. It is shrunk toward the league by how much of it is
  noise (empirical Bayes; volatility splits half against half at 0.27).

Nothing is fitted week by week. Built before Sep 3 and before Sep 10,
the game came out 0.154 and 0.147. Players' own form came out 0 to 0.19
and 0 to 0.21; most players are 0, and a player with no history gets 0.
It's stored in the tables (the game) and in `v5players.json` (each
player's `form`).

**Held out**, against the same build with the strengths fixed. Brier,
where positive is better:

| | moneyline | spread | total |
|---|---|---|---|
| Sep 3–9 | −0.0001 | −0.0001 | +0.0004 (Q1 +0.0011, Q3 +0.0006; overtime all three **+0.002 to +0.004**) |
| Sep 10–22 | −0.0001 | +0.0001 | **+0.0012** (Q1 **+0.0025**, Q3 **+0.0010**, Q2 +0.0011) |

Spread of the points still to come, real / v5:

| | before | after | bottom / top tenth after |
|---|---|---|---|
| Sep 3–9 | 8.52 / 8.13 | 8.52 / 8.79 | 9.1% / 9.8% |
| Sep 10–22 | 9.26 / 8.23 | 9.26 / 8.84 | 11.5% / 10.0% |

An earlier version drew the two offenses independently, with one sd for
everyone refitted on the last weeks built on. It swung from week to week
(0.19 against 0.13) and cost spread (−0.0006, and −0.0005 in Q4 on
Sep 10–22), because an independent draw widens the margin, which was
already right.

Q3's narrowness remains (13.4% of real results in the bottom tenth on
Sep 10–22), and so does Q4's. That is where the line sits, not how wide
the distribution is.

### v5 reads nothing of GAMEPLAI's

v5's prices are only ever compared with GAMEPLAI's, never trained on them.
Every match's prior comes from our own pre-match model, NB2, which is
fitted on match results. That covers pricing, the rubber-band fit, the
in-play fit and the build's checks. `v5-build` refuses to run without
`--history`. A model without NB2 (only in tests) takes league-average
offenses. Removed from v5:
- the prior read off prod's pre-match lines (`prior_lines`,
  `PriorGrid.fit`);
- the correction to prod's pre-match total (`recent_total_shade`).

A test moves every prod price column in the export and checks that
nothing in v5 changes.

### Overtime as it is played

The feed shows overtime played like the NFL's current rule: each side
has the ball once, and after that the game ends the moment one side
leads. A side that scores a touchdown to trail by one then goes for two,
to win or lose: finals like 24-23 and 32-31 are common. v5 used to play
overtime as a full timed quarter in which both sides kept scoring. From
an overtime kickoff it averaged 10.0 points, with 20+ in 8.7% of games;
real overtimes never went past 15. Now it averages 7.8. Real overtimes
averaged about 8.9 over 29 matches. The second side answers a touchdown
with one of its own more often in real games (about 65% of 23 cases,
against v5's 36%).

Held out on Sep 3–9, overtime snapshots now expect 5.9 points still to
come, against 7.2 before and a real 5.3. Over at v5's own line went from
40.7% to 45.3%. Nothing else moves.

### A dead game's line

When v5 is certain no more points will come (a leader kneeling it out),
the half-point lines either side of the score were equally near even
money. v5 took the lower one, quoting an Over that had already won; that
was 2.8% of snapshots. It now never picks a line that everything goes
over (v4 too).

### The shape of the points still to come

The real distribution of points still to come is skewed right, not left.
Its median is below its mean. v5's is skewed almost exactly as much in
every cross-section, and its medians match. Held out on Sep 3–9:

| cross-section | skew, real / v5 | median, real / v5 |
|---|---|---|
| Q1, level | +0.45 / +0.44 | 32 / 31 |
| Q3, 1 score, leader has ball | +0.69 / +0.81 | 14 / 14 |
| Q4, 1 score, leader has ball | +1.58 / +1.52 | 0 / 0 |
| Q4, level | +1.98 / +1.86 | 3 / 3 |

So a middle line doesn't sit too high from skew. Late in a game the
points still to come are lumpy (0, 3 or 7), so the line nearest even
money isn't 50/50. The test is each stream's own P(over) at its line
against how often the game went over it. For v5 (Sep 3–9, P(over) /
real):

| cross-section | v5 | real |
|---|---|---|
| Q4, leader has ball, 1 score | 55.5% | 54.1% |
| Q4, leader has ball, 2+ scores | 57.5% | 55.7% |
| Q4, level | 47.0% | 40.3% |
| Q4, trailer has ball, 2+ scores | 45.4% | 51.0% |
| Q4, trailer has ball, 1 score | 44.7% | 48.6% |
| Q3, leader has ball, 2+ scores | 50.1% | 55.4% |
| Q3, trailer has ball, 2+ scores | 49.7% | 46.0% |

The real misses are:
- level Q4, where v5 gives too many points;
- trailing teams late, where it gives too few;
- two-score Q3 games.

The report's totals breakdown shows the same for prod and every
candidate.

### Totals from inside a game

**Lines against results.** A report can show that the rest of the game
really produced one score or less more often than the lines sat within
one score (31.8% real against about 22% for prod, v4 and v5, Sep 17–24).
Most of that is the line being a middle, not a miss. A line sits in the
middle of the points still to come, so it is more than a score out more
often than the result is. On held-out games (Sep 3–9):
- 29.4% of snapshots really had one score or less still to come;
- v5 put its line within one score on 22.0% of them;
- yet v5's own distributions gave one score or less a 30.1% chance.

The fair test is how often the game went over the line, by where the
line sat; a middle line goes over half the time wherever it sits. On
Sep 3–9:

| line above the score | prod | v5 |
|---|---|---|
| within 1 score | 45.1% | 52.3% |
| 1–2 scores | 43.9% | 48.0% |
| more than 2 scores | 47.0% | 50.5% |
| 1–2 scores, Q4 | 34.3% | 41.1% |
| more than 2, Q3 | 41.4% | 45.2% |

Prod holds its lines about a score too far out. v5 is near a half apart
from Q3, where it leaves 0.3–0.8 points too many, and Q4 with the line
1–2 scores out. The report now carries this table (Additional checks).

**By state, held out (Sep 3–9).**

| state | chance of | real | v5 |
|---|---|---|---|
| Q4, within 3 | 7+ more | 39.0% | 42.6% |
| Q3, within 3 | 21+ more | 18.9% | 23.0% |
| Q3, 17+ margin | no more points | 14.1% | 6.1% |
| Q4, 17+ margin | 14+ more | 12.1% | 6.8% |

The last two minutes of the first half score more than v5 gives them,
and Q3 less.

**The in-play shift (`v5-build --in-play`, off by default).** It fits a
scoring shift, `inplay_theta`, for each segment of the game (Q1; Q2; Q2's
last 2:00; Q3; Q4; Q4's last 2:00; OT). Optionally it also fits each
margin band within a segment (0–3, 4–8, 9–16, 17+). The fit uses states
from inside the build's games, with each match's prior as pricing sets
it, so that from each the simulation leaves the points really still to
come. It goes from the end of the game backwards. Only the total is
priced off the shifted run; moneyline and spread keep the plain one. The
pre-match isn't touched.

The mean moved the right way in every test, and the shape got closer.
On Sep 3–9, close Q4 games went from 42.6% to 39.7% for 7+ more (real
39.0%), and Q3 at v5's own line went from 48.6% to 51.3% over. But total
Brier at prod's line didn't hold up. Positive is better:

| shift (total only) | Sep 3–9, NB2 pre-match | Sep 10–22, prod's pre-match |
|---|---|---|
| by segment | +0.0001 | −0.0012 (significant) |
| by segment and margin | +0.0004 | −0.0010 |
| Q3 and overtime only | −0.0005 | +0.0012 |

Shifting scoring for every market cost moneyline 0.0011 and spread 0.0018
in Q1 (Sep 3–9, significant); that's why only the total uses it. How much
of the in-game miss there is depends on whether the pre-match level is
right, and that changes from week to week. Q4 by margin even flipped
between weeks: blowouts scored in garbage time in one week and not the
next. So the shift stays off until a longer run of weeks shows it pays.
With the flag, the build prints real against simulated for each segment,
and pricing runs two simulations per snapshot.

**Common random numbers.** Averages over thousands of states from many
matches need independent paths (`common=False`). On common random numbers
they get only `n_paths` games' worth of luck between them, and that had
hidden the Q3 miss: the build's in-game check read Q3 as right. The
in-play fit and that check now use independent paths. The rubber band's
fit still uses common ones, and is worth refitting the same way.

```bash
python -m eAMFModel v5-build eAMFCalibrator/out/scouting_playover.csv --half all --out v5_model --history eAMFCalibrator/out/match_history.csv
python -m eAMFCalibrator report --candidate v5 --v5-model v5_model --since 2026-09-17 --until 2026-09-24 --out eAMFCalibrator/out_v5
python -m unittest eAMFModel.tests.test_v5
```

## v6: v5 plus late-game decisions from real play

v6 (`sim6.py`, `v6.py`, `v6_stream.py`) is a copy of v5 with two changes. Both
came from testing three ideas against held-out games (Sep 3–9 built before
Sep 3, Sep 10–22 built before Sep 10).

### The ideas tested

**1. The line and the chunks.** Taking the line under the nearest score (the
likeliest points-still-to-come near the middle, minus a half) rather than the
even line puts v5's P(over) at 0.56–0.57 instead of 0.47. The miss against
real stays where it was under every line rule: +0.015 to +0.019 on Sep 3–9
and −0.022 to −0.030 on Sep 10–22. So the line rule doesn't fix the
calibration. What the test did find is that v5 puts the wrong weight on
exact scores:

| points still to come | v5 / real, Sep 3–9 | v5 / real, Sep 10–22 |
|---|---|---|
| exactly 3, Q4 | 9.0% / 11.7% | 9.0% / 11.4% |
| exactly 7, Q4 | 17.4% / 15.6% | 17.8% / 16.2% |
| exactly 10, Q3 | 8.0% / 10.4% | 7.6% / 10.1% |

From kickoff the touchdown and field-goal counts are right (4.36 / 1.12 a
match, real 4.39 / 1.04). The miss is late and depends on the game state.
The biggest is in Q4 with the sides two scores apart: 6–8 more points 44–45%
against a real 33–36%, and a single field goal 4–5% against 5–8%.

**2. What has happened in the game so far.** Points scored so far against
the pre-match expectation, and pace so far, don't predict v5's miss on the
points still to come. The effects are near zero, well inside the noise, and
change sign between weeks. That matches the in-game strength update,
which measured as noise too. No change.

**3. How many points are still available.** v5's upper tail matches real
games at every time left. For example, with 2–4 minutes left the chance of
14+ more points is 20.4% against a real 20.0% (Sep 3–9). A cap would only
cut real outcomes. No change.

### What v6 changes

- **Late 4th downs when behind.** v5 sent a side four or more behind in the
  last three minutes (or overtime) for it on every 4th down. Real players
  kick: 28% of the time when 9–11 behind in field-goal range, making it a
  one-score game. v6 takes go / field goal / punt from a table fitted on
  real 4th downs by deficit (−4..−8, −9..−11, −12..−16, −17 or more) and
  kick range (up to 45, 46–55, longer), shrunk toward the kick range's
  rate over all deficits (`late_fourth_choices`, `fit_late_fourths`). A side
  1–3 behind keeps v5's fitted kick-to-tie rates.
- **Big second-half leads as situations of their own.** "Second half,
  ahead" and "last two minutes, ahead" are split at a lead of 9
  (`BIG_LEAD`), so a two-score leader draws from its own real plays. The
  setting is stored in the tables.

**Held out**, against v5. Brier, where positive is better:

| | moneyline | spread | total |
|---|---|---|---|
| Sep 3–9 | +0.0001 (Q3 **+0.0005**) | −0.0002 | −0.0002 |
| Sep 10–22 | +0.0002 (Q3 **+0.0005**) | 0.0000 | −0.0001 |

That's level with v5; rebuilding the same model moves Brier by about
0.0005. The late 4th-down table moves the chunk it was aimed at: in Q4 with
the sides two scores apart, exactly 3 more points went from 4.3% to 5.8%
against a real 8.2%.

**Tried and dropped: kneels that are not certain.** When the leader has the
downs to kneel it out, v5 ends the game there (99.9% no more points; real
89–92%). A per-snap kneel chance fitted to real kneel-it-out states fixed
those states (0.45 more points against a real 0.46). But it cost the total
0.0015 on Sep 10–22 (Q4 0.0045): the extra points spilled into every late
state the leader would later kneel from.

**Still open.** In Q4 with the leader on the ball and ahead by 9 or more,
v5 and v6 give 1–2 points too many. For example, with 80–120 seconds left
and a lead of 9–16: 3.6 more against a real 2.5, and 54% no more points
against 67%. It isn't in the play tables (the lead split didn't move it) or
in the 4th-down choices.

**4th downs, fitted jointly (Sep 2026).** v6 went for it on 4th down too
often inside field-goal range: 44% against a real 33% from 6–10 yards out,
and 51% against 34% from 11–20. So it turned field goals into touchdown
tries: too few games finished with exactly 3 or 10 more points, and too
many with 7 or 14 (`remaining`). The league curve was a fixed constant.
Each player's aggression and the part-of-game shifts were each measured
against it separately, so the league's own difference from the curve was
counted twice. Now each build fits the curve, the part-of-game and margin
shifts, and every player's own go shift together (`sim6.fit_fourth_downs`).
It does the same for a kick being a field goal rather than a punt, with
each player's own kick shift (`Profile.kick`). The fitted curves are saved
in the tables. Fitted before Sep 10 and checked on Sep 10–22 (players left
out):
- log loss per decision went from 0.534 to 0.529;
- the go rate from 6–10 yards out went from 44% to 38%, and from 11–20
  yards out from 51% to 40%.

The overpriced overs at 3.5–13.5 points still to come shrank. A lone field
goal in a level Q3 (too rare) and a lone touchdown when trailing by two
scores in Q4 (too common) are still open.

**Overtime (`OT_RULES`).** A touchdown that wins overtime ends the game
with no conversion, which is 6, not 7. A side trailing once the other has
had the ball never punts or kicks short of a tie. A side level or ahead
before the other has had the ball goes for it in field-goal range at the
rate real overtimes show (`ot_go`, about two in three). Only about 50
overtimes a month are played, so the rest of the overtime gap to reality
is within noise.

**Q4 drives and the red zone (Sep 2026).** Per drive (`remaining
--drive`, built before Sep 10, scored on Sep 10–22), v6 turns Q4 drives
into touchdowns where real ones settle for field goals. It's worst when the
score is level (field goals 11.8 points too rare, touchdowns 6.6 too common)
or a side leads with the ball.
- **Q4 play calling** (`PLAY_CALLING_Q4`). The clock-stop, seconds and
  efficiency shifts, fitted in Q4 too, moved these rows by 1.5 points at
  most and left the log score where it was.
- **Where it comes from.** Real Q4 drives run more snaps and first downs
  than v6's: level, with over 2:40 left, 4.2 snaps against 3.6. They then
  stall and kick. Before the last two minutes, Q4 plays come from the same
  bins as Q3. But inside the 30, a real Q4 leader scores a touchdown on
  6.8% of snaps, where a Q3 leader scores on 13.3%. Per snap, v6 scores
  12.8% inside the 30 in Q4 against a real 8.8%, and 43% inside the 10
  against 37%. The bins' fallback order isn't the cause: keeping the field
  zone before the game situation changed less than 1 point.
- **The red-zone fit** (`RED_ZONE_FIT`, `fit_red_zone`). Inside the 30,
  by quarter, lead and inside the 10 or not, a share of the would-be
  touchdowns is redrawn from the same bin (`rz_hold`: 20–34% in Q4, 0–16%
  in Q1–Q3). A tilt mode (`RED_ZONE_MODE = "tilt"`) does it by shifting
  the draw instead. That also cut first downs. Per drive, Q4 level went to
  −10.1 field goals and +4.8 touchdowns, and Q3 two scores up with the ball
  to +4.3 touchdowns from +8.2. The Q4 log score went from 1.095 to 1.092.
- **Why it's only a small gain.** The quarter-points fit then raises Q4's level
  (0.014 to 0.075), to keep kickoff-to-Q4 points on a target about 0.5
  points too high (see v5's quarters' levels). Over the whole
  rest of the game, the log score went from 2.966 to 2.965. It was better
  in Q1 and in Q4 for leaders and level scores, and worse in Q2, in Q3 and
  for Q4 trailers. Pinning Q4's level to the old value cost more (2.990).
  It's on in v6 (`--red-zone off` builds without it): it moves the Q4
  leader and level states toward real play at no cost overall. A model built before it has no `rz_hold` and plays as before
  until it's rebuilt.

```bash
python -m eAMFModel v6-build eAMFCalibrator/out/scouting_playover.csv --half all --out v6_model --history eAMFCalibrator/out/match_history.csv --handles eAMFCalibrator/out/match_history.csv
python -m eAMFCalibrator report --candidate v5,v6 --v5-model v5_model --v6-model v6_model --since 2026-09-18 --until 2026-09-25 --out eAMFCalibrator/out_v5_v6
python -m unittest eAMFModel.tests.test_v6
```

## v7: v6 plus fresh possessions, Q4's own play, leaders settling and the go-ahead check

v7 (`sim7.py`, `v7.py`, `v7_stream.py`) is a copy of v6 with four changes. Each is
behind a switch in `sim7` (`FRESH_CLOCK`, `Q4_MODES`, `SETTLE_FIT`, `GO_AHEAD`); with
all four off, v7 plays exactly as v6 (`test_with_its_own_switches_off_v7_plays_as_v6`).
They came from setting v6's play against real games from the same states, one snap at
a time and one drive at a time (built before Sep 1, played on Sep 1–10).

### What v6 got wrong in Q3 and Q4

- **The clock mid-drive.** A snap takes the clock of real snaps in the same
  situation. After a kick-off, a punt, a turnover on downs, a missed field goal or an
  interception, the next snap starts with the clock stopped: it takes 7–9 seconds.
  After a scrimmage play a snap takes about 23 (26.8 after a first down, 21.4
  otherwise). v6's tables mixed the two, and in the own half 40% of the snaps are fresh
  possessions. So v6 played the snaps of a drive under way 1.5–2.5 seconds too fast in
  every quarter, and the first snap after a kick too slow. In-play prices start
  mid-drive.

  | seconds a snap, real / v6 | Q1 level | Q3 level | Q3 1 score ahead | Q4 level |
  |---|---|---|---|---|
  | Sep 1–10 | 28.4 / 27.0 | 26.6 / 24.6 | 28.7 / 26.7 | 24.0 / 21.5 |

- **Q4 played as Q3.** Before the last two minutes, the second half's situations
  (ahead, level, behind) pooled Q3's plays with Q4's. Real Q4 leaders score and turn the
  ball over about half as often as Q3 leaders. Two scores up: 0.057 touchdowns and 0.011
  turnovers a snap in Q4, against 0.093 and 0.023 in Q3. v6's Q4 leader scored on 0.099
  a snap. The clock-stop and seconds shifts by game state were fitted in Q1–Q3 only, so
  a Q4 side down 9+ played at the pace of one down 1.
- **The red-zone hold saw part of the picture.** It was fitted against the tables'
  own chance of a touchdown, without the team's strength or the quarter's scoring level,
  and only inside the 30. Two scores up in Q4, v6 scored on 34% of snaps inside the 10
  against a real 18%, with the hold on.
- **Late 4th downs.** The table for a trailing side's late 4th down pooled the whole of
  the last three minutes. In field-goal range, real players down 4–8 went for it on all
  86 such 4th downs with a minute or less left, kicking on none. With 1–2 minutes left 6%
  kicked, and with 2–3 minutes 16%. v6 kicked far more often.
- **The rubber band's Q3 pull** could stop on a step that was worse than one before
  it. One build ended with a Q3 pull that boosted leaders: Q3 comebacks of 0.096 of a
  point per point of lead, against a real 0.146.

### What v7 changes

- **Fresh possessions (`FRESH_CLOCK`).** Every snap record and every priced state
  knows whether its snap starts a possession with the clock stopped
  (`sim7.is_fresh`). That's a KICKOFF, PUNT, TURNOVER_ON_DOWNS or FIELD_GOAL
  PLAY_OVER, or a scrimmage one carrying the feed's POSSESSION message.
  - A fresh snap draws a clock-stopped play at the rate fresh snaps do (95%).
  - Every other snap draws from the bins with the fresh snaps left out.
  - The play-calling fit (clock stops and seconds by game state) uses only snaps that
    follow a scrimmage play.
  - The simulation marks the next snap fresh after a kick-off, a punt, a turnover and a
    missed field goal, and at the start of Q2 and Q4.
- **Q4's own situations (`Q4_MODES`).** Before Q4's last two minutes, ahead 1–8,
  ahead 9+, level and behind are situations of their own, drawn from real Q4 plays. A
  sparse bin falls back to the second half's pooled plays, as in v6. The clock-stop and
  seconds shifts by game state now run in Q4 too (`PLAY_CALLING_Q4`), so a side 9+
  behind hurries and one 1–8 behind less.
- **Leaders settle (`SETTLE_FIT`, `v7.fit_settle`).** The red-zone hold becomes a
  hold on would-be touchdowns anywhere on the field (`td_hold`). It's kept by part of
  the game (Q1, Q2, Q2's last two minutes, Q3, Q4, Q4's last two minutes), the
  offense's lead and the field zone.
  - The build simulates one snap from every real snap. Each start carries the match's
    NB2 strengths, its fresh flag and every shift the simulation applies.
  - It moves each cell's hold until the touchdown rate matches that of the real next
    play. A small cell is shrunk toward no change.
  - It alternates twice with the quarter-level fit, so kick-off totals stay on the
    league's.
- **The go-ahead check (`GO_AHEAD`).** The late 4th-down table is split into the last
  minute and the two before it. In Q4, a side behind by 4–8 with a minute or less left,
  or by more with 30 seconds or less, never kicks a field goal that can't tie: it goes
  for it.
- **The rubber band keeps its nearest step** (`v7.fit_rubber_band`).
- **A side 9+ behind in Q4 is a situation of its own**, before and in the last two
  minutes. Real sides that far behind turn the ball over 0.065–0.073 a snap, against
  0.037–0.050 when pooled with one-score trailers.

### Held out

Points still to come (`remaining`), every PLAY_OVER snapshot, built before Sep 17 and
played on Sep 17–22 (26,734 snapshots). "Mean" is the version's mean minus the real
one; "rps" is the ranked probability score over 0–35 points (lower is better).

| | v6 mean | v7 mean | v6 rps | v7 rps |
|---|---|---|---|---|
| all | +0.85 | +0.66 | 3.937 | 3.920 |
| Q2 | +1.10 | +0.80 | 4.867 | 4.847 |
| Q3 | +1.29 | +1.15 | 4.417 | 4.385 |
| Q3 level | +1.98 | +1.26 | 4.254 | 4.158 |
| Q3 1 score, trailer has ball | +1.65 | +1.17 | 4.290 | 4.223 |
| Q4 | +0.32 | +0.14 | 2.393 | 2.385 |
| Q4 level | +1.03 | +0.47 | 2.056 | 1.957 |
| Q3 2+ scores, leader has ball | +1.72 | +2.13 | 4.248 | 4.301 |
| Q4 1 score, trailer has ball | −0.07 | −0.54 | 2.999 | 3.007 |
| Q4 2+ scores, trailer has ball | +0.29 | +0.54 | 2.480 | 2.507 |

Q4 level, P(more than N still to come) minus reality moved from +6 to +7 points of %
to +2 to +3. On Sep 1–10 (built before Sep 1), v7 without the 9+-behind split took the
rps from 3.901 to 3.889, and the Q4 level mean from +0.57 to −0.15.

Per snap on Sep 17–22 (real / v6 / v7):

| | seconds a snap | touchdowns a snap |
|---|---|---|
| Q3 level | 26.8 / 25.2 / 26.7 | |
| Q3 2+ ahead | 29.1 / 26.9 / 28.1 | 0.064 / 0.071 / 0.073 |
| Q4 level | 24.5 / 21.6 / 24.2 | 0.061 / 0.091 / 0.081 |
| Q4 2+ ahead | 22.0 / 23.8 / 22.7 | 0.060 / 0.079 / 0.062 |
| Q4 2+ behind | 15.3 / 21.2 / 16.7 | 0.106 / 0.120 / 0.111 |

A quarter's mean miss moves about ±0.3 points between halves of the same week's matches
(snapshots in a match move together), so read single cells with care.

### Still open

- **Two scores apart.** With the leader on the ball, both v6 and v7 give too many
  points still to come (Q3 +1.7 / +2.1, Q4 +1.2), and in Q4 "no more points" is 9–10
  points of % too rare. The leader's own snaps are now about right, so what's left is in
  the sequence: more trailer possessions, and trailers scoring too easily late.
- **A one-score trailer in the last two minutes** converts fewer first downs in the
  simulation than real ones do (0.27 against 0.31 a snap on Sep 1–10). Since the
  go-ahead check correctly sends it for it, "no more points" comes out too likely.
- **The league's level moves between weeks.** From real Q4 starts on Sep 1–10, real Q4s
  made 8.84 points against 9.43 in the weeks before. No in-play fit follows that.

```bash
python -m eAMFModel v7-build eAMFCalibrator/out/scouting_playover.csv --half all --until 2026-09-17 --out v7_model --history eAMFCalibrator/out/match_history.csv --handles eAMFCalibrator/out/match_history.csv
python -m eAMFModel remaining eAMFCalibrator/out/scouting_playover.csv --version v7 --model v7_model --history eAMFCalibrator/out/match_history.csv --handles eAMFCalibrator/out/match_history.csv --since 2026-09-17 --until 2026-09-22
python -m eAMFCalibrator report --since 2026-09-17 --until 2026-09-23 --snapshots play_over --candidate GAMEPLAI_STREAM_CANDIDATE,v6,v7 --v6-model v6_model --v7-model v7_model
python -m unittest eAMFModel.tests.test_v7
```

## Pricing only what the model is sure of (v4–v7 streams)

A version quotes a prod message only where its state is the game's at that
message:
- its latest `PLAY_OVER` is one it can read (`playover.state_for`), not an
  older one standing in for a play it can't;
- the message is before the next `PLAY_STARTED` (a play under way is new
  information);
- TEAM_A's side is tied to the scoreboard. A match where it isn't isn't
  priced at all; the side used to be guessed as home.

`stream.confident_windows` gives each read `PLAY_OVER` the messages it's
the state for, and `quote_rows(..., windows=...)` quotes only inside them.
Coverage drops, but no price is stale. `eAMFCalibrator bets` checks the
same against SCOUTING_FULL message by message.

**v6 before the first play (`v6_stream.PREMATCH`, on).** Before a snap
nothing is known of the game but who is playing. So v6 prices the kick-off
(`v6.price_kickoff`) and quotes it on every prod row published before the
first play started: prod's pre-match rows (no message) and in-play rows
below the first `PLAY_STARTED`.
- The kick-off is simulated off the NB2 prior with the players' profiles.
  Each side receives the opening kick in half the paths.
- It needs no play state, so a match whose TEAM_A side isn't known still
  gets its pre-match price.
- Between the first `PLAY_STARTED` and the first `PLAY_OVER` nothing is
  quoted.

In the bet sim these are the same information as prod's: pre-match, and
before the first play with the score 0–0. So v6 is now tested on the
pre-match bets and on the bets struck before the first snap. The
calibration report's pre-match section pairs v6's closing pre-match price
with prod's on the same line (v6@prod).

## Points still to come against reality: `remaining`

Comparing a version with prod only says which is nearer. To find where a
version itself is wrong, `remaining` sets it against what actually happened.
At every `PLAY_OVER` in an export, the version's distribution of the final
total, less the points on the board, is its distribution of the points
still to come. The export's final gives what really came.

```bash
python -m eAMFModel remaining eAMFCalibrator/out/scouting_playover.csv --version v6 --model v6_model \
    --history eAMFCalibrator/out/match_history.csv --since 2026-09-10 --until 2026-09-22
```

Build the model on matches before `--since`, or the comparison is in-sample:
`vN-build --until DATE` builds everything (play tables, profiles and NB2) on
the matches before DATE:

```bash
python -m eAMFModel v6-build eAMFCalibrator/out/scouting_playover.csv --half all --until 2026-09-10 \
    --out v6_model_pre10 --history eAMFCalibrator/out/match_history.csv
```

Snapshots in one match share one final, so a cell's real sample is its
matches, not its snapshots. Read cells with a few hundred matches behind
them.
By quarter and game state (level, one score or two+ apart, and whether the
leader or the trailer has the ball), it prints:
- **real:** the share of snapshots where the rest of the game made exactly
  0, 3, 6, 7, 8, 10, 13, 14, 17 or 21 points, more than 21, and the mean;
- **the version minus real** at each of those values, the mean, and the log
  score of what really came (lower is sharper and right). For example, a
  +8 at =3 in a level Q4 means a lone field goal is simulated 8 points of
  probability more often than it happens;
- **P(more than N):** at N = 0.5, 2.5, 3.5, 6.5 ... 20.5, the version's mean
  against reality, by quarter, over every snapshot. Positive means the over
  is priced too high there.

**By drive: `--drive` (v6).** The rest-of-game view mixes up who scores and
when. `--drive` narrows it to the drive under way. At each scrimmage
`PLAY_OVER`, it compares the points scored on the rest of that drive (0, a
safety, 3, 6, 7, 8) with what the version simulates to the end of the same
drive. The drive ends when the ball changes hands, at the kick-off after a
score, or when the half turns. Read by quarter and game state from the
offense's side, it separates a leader that scores too often from a trailer
that scores too rarely, and a touchdown drive from a field-goal one.

`remaining_<version>.csv` has one row per snapshot: the state, the points
on the board and still to come, and the version's probability of each
value 0 to 36+. Matches whose TEAM_A side isn't known are left out, as the
streams leave them.

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
