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

```bash
python -m eAMFModel v5-build eAMFCalibrator/out/scouting_playover.csv --half all --out v5_model --history eAMFCalibrator/out/match_history.csv
python -m eAMFCalibrator report --candidate v5 --v5-model v5_model --since 2026-09-17 --until 2026-09-24 --out eAMFCalibrator/out_v5
python -m unittest eAMFModel.tests.test_v5
```

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
