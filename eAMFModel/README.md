# eAMFModel

A rival pricer for eAMF moneyline, spread and total, built top-down with no
machine learning, so the calibration suite can put it next to GAMEPLAI's prod
and candidate.

## How it works

From a `PLAY_OVER` snapshot's state -- score, quarter and clock, who has the ball, down,
distance and field position, timeouts left -- a version plays the rest of the game snap by snap
on the real game clock, 2,000 times, and prices moneyline, spread and total off the final scores.
Every snap is drawn from real snaps in the same situation; the clock, 4th downs, kneels,
timeouts and overtime follow what real players do (each version's section says how).

Each match starts from its pre-match prior: the expected points of our own pre-match model (NB2,
`nb2/`, or glmer with `--prior glmer`, fitted on the match history), turned into the two sides'
starting strengths. Player profiles (pace, 4th-down aggression, form on the day) come from the
same export.

The code that runs:

| file | what it is |
|---|---|
| `sim8.py` … `sim12.py` | each version's simulation: play tables and the snap-by-snap game |
| `v8.py` … `v12.py` | each version's build (fits, pre-match grid, profiles) and pricing |
| `v8_stream.py` … `v12_stream.py` | each version as a GAMEPLAI-shaped price stream for the calibrator |
| `v21.py`, `v21_stream.py` | v2.1's snap sim (build, draws, simulate) and its price stream for the calibrator |
| `playover.py`, `state.py` | reading a snapshot into a game state; market ids |
| `stream.py`, `grading.py` | what every stream shares; Brier against prod |
| `players.py`, `drive.py` | player profiles; the league's 4th-down and field-goal curves |
| `nb2_prior.py`, `glmer_prior.py` | the pre-match models |
| `remaining.py` | points still to come against reality |
| `playstyle.py` | how each gamer plays by the score on real games, and whether it is a trait |
| `ats.py` | prod's lines and moneyline against the real finals, the league and gamer by gamer |

## Versions

| | what it adds |
|---|---|
| v8 | the clock to the last second, timeouts, kneels and overtime as played |
| v9 | v8 with recent weeks weighing more, and the late timeout plays' real time |
| v10 | v9 with the pre-match prior's pace counted once, and its totals' spread fitted out of sample (held out RPS 3.837 → 3.825) |
| v11 | v10 with close endings, late conversions by the clock, late-half timeouts, overtime and kick-off touchdowns as played (overtime 5.4% → 3.2% of games, real 2.7%; held out RPS level) |
| v12 | v11 with level sides' late drives run down to the kick (Q4 level RPS 2.090 → 2.042; held out RPS 3.829 → 3.825) |
| v13 | v12 with each side's big-play rate: its gamer's, its team's and what the other side allows, boom or bust (held out in-play RPS 3.821 → 3.811; totals' spread follows the big-play sides) |
| v14 | v13 played closer to the real game -- strength as boom or bust, the clock as played, long plays, losses and kick-offs by side, the go curve's field shape -- and built in one joint fit with a calibration report (first downs 37.6% → 41.5% of snaps, real 41.0%; field goals 1.51 → 1.22 a game, real 1.15; held out RPS level) |
| v2.1 | apart from v8–v14: each snap's time drawn from real snaps, then its distance given that time, with no strength -- the pre-match expected points are only what its report measures it against (`v21-build`, `v21-report`) |

The analytic pricer that came before (v1/v2) and the simulation versions v3 to v7 have been
removed. The v3–v7 sections below are kept as the record of how the simulation was built: v8
onward still carries everything they introduced. Their own commands no longer run, so use v8–v14
in their place.

## v3: a play-by-play simulation

> v3's code has been removed (as have v4–v7's). This section and the next four describe what
> they introduced, which v8–v13 still run.

v3 was the first simulation (`sim.py`, `v3.py`). From the
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

### Its pre-match prior: NB2 or glmer (`--prior`)

`v7-build --prior glmer` fits the glmer model on `--history` in place of
NB2. That's the global mixed model in `glmer/`, run in R with lme4.
`--prior nb2` is the default and is unchanged.

The build saves the model in `v7_model/glmer/` and records the choice in
`v7prior.json`. From then on, `v7`, `remaining`, the stream and the
calibrator's `--candidate v7` all price off whichever prior the model was
built with. A model built before `--prior` existed reads as NB2.

- **The model.** Its terms are:
  - player attack and defence;
  - both NFL teams;
  - the stream;
  - home/away;
  - both sides' recent form: points scored and conceded, recency-weighted
    over their last matches.

  Out of sample on NB2's own split it beat NB2 on the moneyline and on
  totals (log loss 0.6669 against 0.6782). With a refit every 14 days it
  also beat the NB2-structured glmer, 0.6638 against 0.6695. See
  `glmer/README.md`.
- **Its settings.** The feature set, row weighting and form half-life are
  `glmer/fit.R`'s defaults (`glmer/config.R`). `glmer_prior.py` only runs
  `glmer/fit.R` and `glmer/predict.R`.
- **No level scale.** The expected points aren't rescaled. Out of sample
  its totals ran 0.2 points under the real ones. NB2's level scale corrects
  a 1.7-point gap that this model doesn't have.
- **Form follows results, as it would live.** Each match's form counts
  only results from matches that started before it, never its own.
  - Priced with `--history` (`v7`, `remaining`), the rows carry finals, so
    the form reads every earlier result.
  - In the calibrator, the run fetches every match settled by `--until`,
    and the stream passes it in (`means(..., results=...)`). A scheduled
    match takes its final from there, so later matches' form reads it.
  - **Each pre-match quote reads only the results in when it was
    published.** The schedule runs a gamer's matches back to back, 36
    minutes apart, so the previous match is often still on while the next
    one's pre-match prices go up. Pricing those quotes off the kick-off
    would hand glmer a result no bettor had yet.
    - So `stream.known_states` cuts each quote at the earliest of the two
      gamers' earlier matches still unfinished at its publish time. The
      match is then priced as if it started a second before that cut.
    - A result counts from the match's last play in the feed, or 36
      minutes after kick-off (`RESULT_MINUTES`) where the feed has none.
    - In-play prices and quotes published after every earlier match
      ended read every earlier result.
    - A glmer build whose features read the clock (rest, session, hour:
      `context`, `form_session`) prices every pre-match quote off the
      kick-off instead. Its `AS_OF` is false.
  - NB2's ratings stay as they were at the build, live and here.
- **It needs R.**
  - Install R 4.x, then `Rscript glmer/install_packages.R`.
  - `Rscript` is found on the PATH or through `RSCRIPT`. On Windows it is
    also found under Program Files.
  - The fit takes a few minutes on the full history; pricing a batch of
    matches takes seconds.

```bash
python -m eAMFModel v7-build eAMFCalibrator/out/scouting_playover.csv --half all --until 2026-09-17 --out v7_glmer --history eAMFCalibrator/out/match_history.csv --handles eAMFCalibrator/out/match_history.csv --prior glmer
python -m eAMFModel remaining eAMFCalibrator/out/scouting_playover.csv --version v7 --model v7_glmer --history eAMFCalibrator/out/match_history.csv --handles eAMFCalibrator/out/match_history.csv --since 2026-09-17 --until 2026-09-22
python -m eAMFCalibrator report --since 2026-09-17 --until 2026-09-23 --snapshots play_over --candidate v7,v7-glmer=v7_glmer --v7-model v7_model
```

The report line sets the NB2 build (`--v7-model`) and the glmer build side by
side; both need building on the same `--until`. The report prices with R as
well, because the glmer prior predicts each match through `glmer/predict.R`.

## v8: v7 plus the clock to the last second, timeouts, kneels and overtime

v8 (`sim8.py`, `v8.py`, `v8_stream.py`) is a copy of v7 with five changes, each behind a
switch in `sim8` (`LATE_CLOCK`, `TIMEOUTS`, `KNEELS`, `OT_CARRY`, and overtime's own clock
shift under `TIMEOUTS`). With all of them off, v8 plays exactly as v7
(`test_with_its_own_switches_off_v8_plays_as_v7`).

```bash
python -m eAMFCalibrator timeouts --since 2026-08-24          # every real timeout -> out/timeouts.csv
python -m eAMFModel v8-build eAMFCalibrator/out/scouting_playover.csv --half all --until 2026-09-17 \
    --out v8_model --history eAMFCalibrator/out/match_history.csv \
    --handles eAMFCalibrator/out/match_history.csv --timeouts eAMFCalibrator/out/timeouts.csv
python -m eAMFCalibrator report --candidate v7,v8 --v7-model v7_model --v8-model v8_model
```

### What v7 got wrong

Set against real snaps from the same states (built before Sep 10, played Sep 10–22, 44,002
snaps), v7's clock was within 0.3 seconds a snap everywhere except:

- **The last 40 seconds of each half.** Real sides stop the clock on 86–88% of snaps; v7 did on
  74%, and each snap took 2 seconds too long. The clock-stopped share was fitted only on snaps
  with a minute or more left: a snap that ran the quarter out was dropped. The last 40-second
  slice of every quarter was never fitted.
- **Kneels.** v7 ended the game once the leader had the ball with 20 seconds or less a down.
  582 of 905 matches reached that, and 0.47 points a match still came after it.
- **Timeouts** were not there at all. The feed has them (`TIMEOUT_CALLED_TEAM_A/B`), but
  the PLAY_OVER export dropped them.
- **Overtime.** When a period's clock ran out, a lead ended the game even if the side behind
  hadn't had the ball, and a level score kicked off again. In real games play carries on into
  the next period with the same ball, down and spot, until a side leads and the other has had
  its possession. 39% of real overtimes reach a second period; none ends level.

### What v8 changes

- **The clock to the last second** (`LATE_CLOCK`). A snap that ran its quarter out is kept as a
  censored observation: it used at least the time that was left. The clock-stopped share is
  fitted by maximum likelihood on every snap, so the last 40 seconds have their own share and
  stopped-play time.
- **Timeouts** (`TIMEOUTS`). Three a side each half, two each overtime period. Before a snap,
  with the clock running after the last play, a side stops it at the rate real sides do:
  - The rate is split by quarter (2 or 4), 40-second slice, with or without the ball, and
    the side's own score, while it has a timeout left.
  - It is fitted on the real calls (`v8-build --timeouts timeouts.csv`), else read from
    `sim8.DEFAULT_CALL_P` (12,000 calls, 24 Aug – 22 Sep).
  - The snap then takes only its own play's time.
  - A real play that followed a timeout goes back in the tables with a clock-running play's
    time, so the calls are not counted twice.
  - Overtime reads the fourth quarter's rates scaled by one fitted factor (0.23; 35 real calls
    in 75 periods).
  - States know each side's timeouts left where the export carries them
    (`timeouts_used_a/b`).
- **Kneels** (`KNEELS`). A fourth-quarter leader's kneel is a play of its own:
  - It is called as often as real leaders call it, by down, 40-second slice and one score or
    more.
  - It takes a 36-second play clock, unless the trailing side calls a timeout.
  - The leader kneels the game out only when its downs left, less the trailing side's
    timeouts (a 40-second play clock each), outlast the clock.
- **Overtime carries on** (`OT_CARRY`). An expiring period ends the game only when a side leads
  and the side behind has had its possession; otherwise play goes on into the next period.
  Overtime also gets its own clock-stopped shift, since real sides hardly call timeouts there.

What the real timeouts showed (`python -m eAMFCalibrator timeouts`; see its README section):
- Nearly all come in the last four minutes of Q2 and Q4.
- Q2: from 3:00 to 1:00 the side without the ball calls them. In the last 30 seconds the side
  with the ball calls 69% of them, at any score.
- Q4: sides without the ball call them from about 3:00, whether down one score or more, and so
  do level sides and sides a score ahead. Trailing sides with the ball call them in the last
  minute.

### Held out

Points still to come (`remaining`), every PLAY_OVER snapshot, built before Sep 10 and played
on Sep 10–22 (56,890 snapshots, 905 matches). "Mean" is the version's mean minus the real one;
"rps" is the ranked probability score over 0–35 points (lower is better).

| | v7 mean | v8 mean | v7 rps | v8 rps |
|---|---|---|---|---|
| all | +0.09 | +0.14 | 3.851 | 3.841 |
| Q2 | +0.04 | +0.12 | 4.759 | 4.757 |
| Q3 | +0.44 | +0.43 | 4.418 | 4.413 |
| Q4 | −0.11 | +0.01 | 2.416 | 2.384 |
| Q4 level | −0.02 | +0.14 | 2.102 | 2.080 |
| Q4 1 score, trailer has ball | −0.74 | −0.46 | 2.911 | 2.877 |
| Q4 2+ scores, leader has ball | +0.65 | +0.55 | 1.725 | 1.656 |
| Q4 last 2:00, 1 score, leader has ball | −0.13 | −0.03 | 1.333 | 1.257 |
| Q4 last 2:00, 2+ scores, leader has ball | +0.23 | +0.23 | 1.015 | 0.961 |
| OT | −1.14 | +0.45 | 2.182 | 2.282 |

The log loss over all snapshots goes from 2.551 to 2.538, and in Q4 from 1.806 to 1.747.
Overtime is 24 matches: its mean miss is gone, but its rps is still above v7's.

Per snap (real / v7 / v8, seconds), the last 40 seconds of Q2: 6.5 / 8.4 / 6.8.

### Still open

- The trailer one score down with the ball in the last two minutes of Q4 goes on to make
  5.0 points; v8 gives 4.6 (v7 4.3).
- Q3 and Q2 two scores apart still run a point high, as in v7.

## v9: v8 with recent weeks weighing more, and the late timeout plays' real time

v9 (`sim9.py`, `v9.py`, `v9_stream.py`) is a copy of v8. Its changes are behind `sim9`'s
`NATURAL_SHIFT`, `TO_PLAY_SECONDS` and `CLOCK_HALF_LIFE`, and `v9`'s `SETTLE_HALF_LIFE`; with
the first two off and no half-lives it plays as v8 (`test_with_its_own_switches_off_v9_plays_as_v8`).
Build and report it as v8 (`v9-build ... --timeouts timeouts.csv`, `--candidate v8,v9`).

### What v8 got wrong, and why

v8's totals ran over in Q3, and in Q4 two scores apart. Per snap against real games
(built before Sep 10, played Sep 10–22):

- **The trailing side two scores down scored too often.** Touchdowns a snap, Q4's last two
  minutes, behind 9+: 0.141 against 0.112 real. On the build's own weeks v8 matched (0.121
  against 0.121). Real sides two scores down have scored less week on week (Q4 behind 9+:
  0.107, 0.104, 0.099, 0.096, 0.097 a snap from the week of 24 Aug); close games held still.
- **Leaders bled less late than they now do.** A leader's clock-running play in Q4's last two
  minutes took 31.6–31.9 seconds in v8 against 32.8–34.0 real. That has crept up too: leading
  by 1–8, 32.0 seconds the week of 24 Aug, 33.2 three weeks on; leading by 9+, 32.2 to 38.1.
- **Timeout plays took a stopped play's time.** A play the trailing side stopped from defence
  in Q4 took 10.7 seconds on average; v8 gave it an incompletion's 6.3. Real sides let the clock
  run on a while before calling.
- **The clock-running time by state was diluted.** v8 fitted it on every running play,
  timeout-stopped ones included with a time borrowed from their bin.

The per-snap clock everywhere else in Q3 and Q4 was right to within a second. The late level
drain-and-kick, which looked heavy snap by snap, makes the right points: from real level
in-range states in Q4's last two minutes, the offense's 3.97 against 3.90 real (3.90 → 4.75
without it).

### What v9 changes

- **Recent weeks weigh more** (`SETTLE_HALF_LIFE`, `CLOCK_HALF_LIFE`: 7 days). The touchdown
  (settle) fit and the clock fits weight each match by its age at the cut-off, halving every
  week. The clock fits are the clock-stopped share, the play time by state, the kneels and
  the timeout calls.
- **The play time by state is fitted on plays nobody stopped** (`NATURAL_SHIFT`).
- **A play stopped by a timeout takes the time real ones did** (`TO_PLAY_SECONDS`), by quarter,
  who called it and whether the caller is behind, level or ahead.

### Held out

Points still to come, built before Sep 10, played on Sep 10–22 (56,890 snapshots).

| | v8 mean | v9 mean | v8 rps | v9 rps |
|---|---|---|---|---|
| all | +0.14 | +0.14 | 3.841 | 3.837 |
| Q3 2+ scores, trailer has ball | +0.94 | +0.77 | 4.587 | 4.576 |
| Q3 2+ scores, leader has ball | +1.20 | +1.14 | 4.115 | 4.114 |
| Q4 | +0.01 | +0.01 | 2.384 | 2.382 |
| Q4 2+ scores, leader has ball | +0.55 | +0.46 | 1.656 | 1.647 |
| Q4 1 score, trailer has ball | −0.46 | −0.39 | 2.877 | 2.874 |
| Q3 level | +0.49 | +0.66 | 4.401 | 4.412 |
| Q4 level | +0.14 | +0.23 | 2.080 | 2.085 |

The two-score states come down; level states rise a little. A half-life can only follow a trend
as far as the build's own weeks show it: the held-out week's leaders up 9+ bled far more (38.1
seconds a running play) than any week before.

## v14: v13 played closer to the real game, and built in one joint fit

v14 (`sim14.py`, `v14.py`, `v14_stream.py`) started as v13 copied exactly. Each of its changes has a
switch in `sim14` or `v14`; with them off it plays as v13. Its build is `python -m eAMFModel v14-build`
and its candidate `--candidate v14` (`--v14-model`, `--v14-paths`, `--v14-lines`).

### What v13 got wrong

An audit (the calibrator's snapshots, every match with both expectations within 3 of 15.5: 1,178
real games, each simulated at its own expectations) set v13's play-by-play against the real games:

- **Strength worked through success, not variance.** In real matches about level on paper the share
  of snaps that make the line hardly moves with the expected total (43% at 26 points a match, 45% at
  47) and neither does the share that gain nothing; the 40+ yard plays run from 1.6% of snaps to
  5.6%. v13 moved a side's whole draw with its strength, so a low-scoring match made the line on 36%
  of snaps and gained nothing on 22%, and its points were held up by too many long plays. It reached
  4th and long a quarter more often than real sides, and kicked more: 1.51 field goals tried a game
  (real 1.15).
- **The clock.** A snap whose play could not start before the quarter's end was played anyway; the
  two-minute warning did not stop the clock; the stopped-clock share took no account of what the
  play before did (a real snap after nothing gained follows a stopped clock 76% of the time, after a
  gain 10%). Snaps per quarter ran off by up to a snap.
- **Kick-offs and losses.** Every kicker kicked alike; every side lost yards alike.

### What v14 changes

- **Strength as boom or bust** (`sim14.STRENGTH_BOOM`, `BOOM`, `TILT_SCALE`). A side's strength also
  acts as its big-play rate does: exp(BOOM x theta) times its bin's big plays and the same chance
  more of its failed plays, with only TILT_SCALE of it tilting the whole draw. The prior grid
  (now -1.1 to 0.8) fits the strengths to the points. The big-play ratings' elasticity is held at
  0.5 or more (`BIG_ELASTICITY_MIN`).
- **The clock as played.** A snap with no time to start is not played (`CLOCK_EXPIRES`); the
  two-minute warning stops the clock (`TWO_MINUTE`); the stopped-clock share follows the play before
  (`STOP_LINK`, `play_class`: nothing gained, a short gain, a first down, a loss); each quarter's
  pace is fitted by 20-second slice to the real seconds a snap and snaps a quarter
  (`QUARTER_PACE`); the snaps from the two-minute mark and the end-of-half field goals there are
  fitted to the real ones (`MANAGE_FIT`, `EARLY_FG_HALF_END`).
- **Long plays run on** (`LONG_TAIL`): a touchdown play stretched to a longer field runs on as often
  as real gains that long do (Kaplan-Meier, `gain_survival`), else stops short.
- **Each side's lost-yardage rate and loss size** (`LOSS_PLAYS`, `zone_warp`), rated as the big plays
  are, and **each kicker's touchback and no-landing-zone rates** by the part of the half
  (`KICK_STYLES`).
- **The go curve's field shape** (`GO_ZONES`): a term and a distance slope for midfield (own 40-54)
  and the goal line, where real sides go for it far more than the smooth curve gave (in-sample
  misfit over 29 field x distance cells: chi-square 88 -> 33).
- **The prior's points held as pricing plays the match** (`HOLD_AS_PRICED`, `HOLD_STEPS`): the held
  kick-off is simulated with the day's strength spread and the total's in-play run, and stepped until
  it lands within `HOLD_TOL` points of the prior's total.

### The build

`v14.build` runs in three parts:

1. **The fits on real data alone** -- the tables (play bins, the clock's stopped shares and seconds,
   kicks, punts, kneels, the 4th-down curves and every decision table), the players' profiles,
   kick-off styles and big-play and lost-yardage ratings. Nothing later moves them.
2. **One joint fit** of everything calibrated through the simulation -- the quarters' pace, the
   timeouts, the two-minute snaps and field goals, the rubber band, settle, close endings and the
   quarters' scoring levels, with the grid, its responses and the form spread -- in `JOINT_ROUNDS`
   (2) rounds, each stage starting where the last round left it (`JOINT_WARM`: fewer steps after
   the first) and every state played as pricing plays its match (held strengths, own profiles). Fitted
   one after another, each stage moved what the ones before had fitted.
3. **The calibration report** (`calibration_report`): every target measured once more on the final
   tables, real against simulated, with misses beyond `REPORT_TOL` (5%) marked. Each round also
   prints its seconds by stage.

The long simulations (the grid, the responses, the held strengths, the two-minute fits and the rubber
band) run on `--sim-workers` processes (default 4; `BUILD_WORKERS`). Each start's paths draw the same
random numbers wherever they run, so any number of processes builds the same model; on a 16-core
machine `--sim-workers 12` leaves room for the daily glmer fits.

On the held-out build (before 10 Sep) the build takes about 100 minutes. The report's points by
quarter, from real quarter starts and from the two-minute marks, are within 0.6%. Still marked: the
rubber band (+8% to +20%), the settle cells' touchdowns (+12%) and Q4's seconds a snap (-8%) move
after they are fitted; the level side's late field goals (50% of drives, real 68%) and Q4's field
goals on downs 1-3 from 2:00 (0.076, real 0.094) sit at the edge of what their fits may move.

### The strength refit (13 Oct)

Three changes, all in the build. The next `v14-build` makes all three, and a build from before plays
as it did.

- **The lead with TEAM_A on either side** (`sim14._margin`).
  - The export's scoreboard is home's (`score_p1`) and away's. TEAM_A is whichever side
    `team_a_side` says.
  - Every lead the tables read off the snaps assumed TEAM_A was home. So on a match where it is
    away, the lead had the wrong sign in the modes, settle, kneels, 4th downs, kick-offs,
    conversions and the close endings.
  - The rest of v14 (the 4th-down players, kick styles, big-play records, pricing) already read
    `team_a_side`. The build now prints how many matches have TEAM_A on each side.
- **How strength reaches the snaps, fitted** (`v14.STRENGTH_FIT`, `fit_strength`).
  - **Before:** with `BOOM` 3 and `TILT_SCALE` 0.5, strength worked almost all through the 40+
    yard plays. From kick-off, a side expected at 10 points drew them on 0.2% of its snaps and one
    at 25 on 19%. Its first downs under 20 yards fell from 25% to 14.5%, and its snaps from 25 a
    game to 16.
  - **Real sides** (`strength-check`, 1-12 Oct, 1,494 sides), from under 13 expected points to
    over 21:
    - the 40+ plays only double (2.3% to 4.8%; the sim 0.6% to 8.1%);
    - the plays short of the line fall (27.9% to 23.7%);
    - the 20-39 yard plays rise (9.1% to 11.7%);
    - the first downs under 20 hold at 28-29%;
    - the snaps rise (20 to 24 a game).
  - **The fit.** The build tries each boom (`STRENGTH_BOOMS`) with each tilt (`STRENGTH_TILTS`),
    then the steps around the best. For each:
    - it finds the strengths on a coarse grid of the candidate's own;
    - it plays its newest 1,500 matches from kick-off at them, with the players' own profiles;
    - it scores their snaps by class against the real ones in each band of the side's expected
      points (`STRENGTH_BANDS`) by the multinomial deviance.
  - **Kept with the tables.** The best boom and tilt are kept (`Tables.boom`, `tilt_scale`), and
    the build prints the deviance of each candidate and the real, fitted and old snap mix by band.
    The prior grid then spans the strengths that give 6 to 34 points against a league side
    (`Tables.theta_grid`), and the day's spread is read at those for 10 to 26 (`Tables.form_grid`).
  - **Order.** It runs before the joint fit, so everything after it fits on the new strength.
- **The day's spread, through the priced sim** (`v14.SPREAD_FIT`, `fit_spread`).
  - **Before:** priced, the draw made the total too wide (central 50% holding 53.4% of results, 80%
    holding 83.8%) and the margin a touch narrow.
  - **The fit.** After the first round's form fit, the build prices its newest 600 matches as the
    stream does: held strengths, own profiles, and the total off its in-play run, read as if
    centred on the prior's.
    - It scales the shared swing (x0, 0.25, 0.5 or 1) to the best margin RPS plus total RPS.
    - Then it scales each player's own form (x1 to x4) the same way.
  - **Kept with the tables.** The scales are kept (`Tables.spread_scale`) and applied to every
    later form fit, and the build prints each candidate's scores and coverage.
  - **Caveat.** The prior's means on those matches are in-sample, so if anything the fit leans a
    little narrow; `strength-check` part C and `prior-vs-sim` grade it out of sample.

The fits add about 15-25 minutes to the build. `strength-check` reads a build's own boom and tilt;
its sweep's "tilt alone" and "boom alone" switch the build's own off.

**The first refit, out of sample** (1-12 Oct, `v14_fit`):
- **What it chose:** boom 0 and tilt 0.75 (deviance 409, against 892 for the old setting), and the
  shared swing went to 0. TEAM_A was home in all 2,960 build matches, so the lead fix changed
  nothing there.
- **What improved:** the averages landed on the prior (margin 0.00, total +0.01). The margin and
  total covered close to 50% and 80% (51.4% / 80.2%, 51.8% / 81.3%). The moneyline was unchanged
  (log loss 0.6817).
- **What didn't:** priced, the strongest sides (21+ points) still drew too many 40+ yard plays
  (7.2%, real 4.8%) and too few first downs under 20 (23.2%, real 28.8%). Their share short of the
  line stayed at 27.6% (real 23.7%).

So the fit was extended (13 Oct, the second refit):
- **A conversion** (`sim14.CONVERT`, `Tables.convert`, `ZERO_SPLIT`).
  - Each bin keeps its no-gain plays below its short gains (`Tables.n_zero_stop/run`).
  - A side's strength multiplies the odds of a first down against a short gain by
    exp(convert x theta). Turnovers, losses, no gains and big plays stay where the tilt and boom
    put them; real strong sides convert, and those hardly move.
  - `zone_warp` takes six parts. With no conversion it plays exactly as before.
- **The fit reads the ratings as pricing does, and holds the points.**
  - It runs in the joint fit's first round, once the grid's responses have given the big-play and
    lost-yardage ratings their elasticity. Then the grid and its responses are made again on the
    fitted strength.
  - Each candidate's sides are held to their priors by one step along its strength curve before
    their snaps are counted. In the first refit the strongest band ran 2.4 points over.
- **One knob at a time** (`STRENGTH_START`, `STRENGTH_AXES`, `STRENGTH_CYCLES`).
  - The search starts from the first refit's boom 0, tilt 0.75 and convert 0, and tries the boom
    from -1 to 1, the tilt from 0.25 to 1.25 and the conversion from 0 to 3, two cycles round.
  - The build prints every candidate's deviance and, by band, the real snap mix against the
    fitted one, the first refit's and the old one.

### Fixed in the sweep (12 Oct)

- **End-of-half kicks** (`sim14.early_kicks`). A state at 0:00 is the half's end, not a choice, but
  `EARLY_FG_HALF_END` counted it as a side that chose not to kick: 264 at Q2's end, in range on downs
  1-3. Left out, Q2's last-five-seconds kick rates by range go from 36/44/31% to 47/58/47%.
- **Held touchdowns' redraws** (`sim14.simulate`). A would-be touchdown held back (`td_hold`) was
  redrawn without the boom warp where neither side had a big-play or lost-yardage rate of its own, so
  the redraw ignored the strengths' boom or bust.
- **Q4's early-kick target** (`v14.manage_states`, `MANAGE_Q4_CLOSE`). The sim's fourth-quarter early
  kicks are the close sides' alone (level or 1-3 behind), but the target counted every side's: 26 of
  the 207 real kicks were by sides ahead or 4+ behind, which no stage makes, and `fit_manage` held the
  close sides' clock at its limit chasing them. That was the report's Q4 miss (15.5% short).
- **Distinct paths split across processes** (`par_simulate`). Each block now numbers its paths where
  its rows start (`path_offset`), so the held strengths no longer change with the number of processes.
- **The default build never read the timeouts** (13 Oct). `v14-build` with no `--timeouts` finds the
  calibrator's `timeouts.csv` itself, but looked for it after placing the calls, so none were placed:
  the export says how many timeouts each side has used, not who called one, and every call rate was
  fitted at nothing (the report: 2.26 timeouts from Q2's two-minute mark, simulated 0.006). It reads
  them first now, and a build with timeouts used but no calls placed warns and takes the real rates in
  `DEFAULT_CALL_P`. Builds since 12 Oct, the 1-12 Oct report's among them, played without timeouts.
- **Faster**: the rubber band (about half of each round), the timeout calls' fit and the quarters'
  scoring levels run on the build's processes, and the close-level fit no longer plays out every grid
  point's games for an overtime share that weighs nothing: a round took 30 and 21 minutes on 12
  processes, against 64 and 53 on one.

### Held out

The audit's 1,178 games (real in brackets):

| | v13 | v14 | real |
|---|---|---|---|
| first downs a snap | 37.6% | 41.5% | 41.0% |
| 40+ yard gains a snap | 3.01% | 2.70% | 2.80% |
| turnovers a snap | 4.08% | 3.52% | 3.59% |
| 4th downs a game | 3.53 | 2.87 | 2.96 |
| field goals tried a game | 1.51 | 1.22 | 1.15 |
| interceptions and fumbles a game | 2.05 | 1.66 | 1.64 |

Pre-match, the 906 matches of 10-22 Sep (12,000 paths each), v14 minus v13 [95% over matches]: total
RPS +0.004 [-0.011, +0.021], margin RPS -0.002 [-0.022, +0.016], moneyline +0.0007 [-0.0006, +0.0021].
In play, points still to come at 56,890 snapshots (`remaining`, 500 paths), RPS v14 minus v13: all
+0.004 [-0.007, +0.016], Q1 +0.009 [-0.006, +0.022], Q2 +0.007 [-0.011, +0.026], Q3 -0.009
[-0.029, +0.013], Q4 +0.008 [-0.000, +0.016]. A run of the same model on other seeds moves these by
up to 0.007: v14 prices level with v13 while playing much closer to the real game.

### Still open

- Too few blowouts in matches level on paper (margins 9-13: 14.6% of games, real 17.0%).
- Too few 4th and 1 (33% of 4th downs, real 40%): too few short gains on 2nd down.
- Punts 0.61 a game (real 0.71); overtime still runs high.
- Points still to come early in play run 0.5 low, from a heavy right tail; correcting the mean
  (by level, by points scored or by margin) made the RPS worse in every quarter, cross-validated.

## v13: v12 with each side's big-play rate

v13 (`sim13.py`, `v13.py`, `v13_stream.py`) started as v12 copied exactly. With its own switch off
(`sim13.BIG_PLAYS`) it plays as v12 (`TestV13IsV12`).

### What v12 got wrong

Each side has one strength. It moves every play toward the top of its bin at once, so a strong side
gets more first downs and more long plays together, in a fixed proportion. Real sides don't play that
way (the calibrator's `halves`, all history):

- Touchdown length depends on the gamer and on the NFL team, each with the other held level (gamers
  spread their games across many teams). Lions and Eagles touchdowns run 7-8 yards longer than
  Vikings and Ravens ones, yet those teams score no more points and win less often.
- The teams differ on the long plays, not the medium ones. 40+ yard plays are 4.5% of the Lions'
  snaps and 2.9% of the Vikings'. 20+ yard plays are 13-14% for every team.
- Rated only on earlier matches, a side's share of long touchdowns carries over (predicted 30-58%
  across fifths, real 31-56%). Matches between big-play sides spread their totals wider at the same
  expected total: sd about expectation 10.6 in the lowest fifth, 13.5 in the highest. This held in
  every month from June to October.

### What v13 changes

- **Each side carries a big-play rate** (`Start.big`, `Profile.big`). A snap's chance of drawing one
  of its bin's 40+ yard plays (`BIG_GAIN`) is multiplied by it. The same chance moves onto the bin's
  failed plays (no first down) and out of the plays between (`big_warp`). Inside each of the three
  parts, the strength's tilt toward the top still applies. Each bin's plays are ordered with the big
  ones on top, and the tables count both parts (`n_big_*`, `n_fail_*`).
  A first try took the extra big plays evenly from the whole bin. The strength that then held the
  side's points took most of them back: a rate of 1.8 kept a quarter of its big plays, and the totals
  barely moved. Boom or bust keeps them.
- **The rate comes from four effects** (`fit_big_ratings`). These are the log odds of a big play on
  every scrimmage snap built on, against its bin's league rate: the gamer with the ball, its team,
  what the other gamer allows and what the other team allows. Each is shrunk by how far its kind
  really spreads beyond noise, and recent weeks weigh more (`BIG_HALF_LIFE`). The teams come from
  `--history` (`PLAYER_1_TEAM` is home). On the held-out build (47,723 snaps) the spreads are: gamer
  0.31, team 0.11, what a gamer allows 0.23, what a team allows 0.07. The team effects run from Lions
  +0.15 and Eagles +0.08 to 49ers -0.11.
- **The prior's points are held** (`big_response`, `fit_big_elasticity`). As with pace, each side's
  points from kick-off are simulated at pairs of rates, and a match's expected points are divided by
  its response before the strengths are fitted. The build also measures how far the simulated big
  plays move with the rate once the points are held (0.52 on the held-out build). A rating is then
  played at its odds to the power 1 / 0.52, so the sim's big plays move as the rating says. Rates
  are kept within 0.37-2.72 (`BIG_POINTS`).
- **Where the rate is applied:** the build writes the ratings to `v13big.json`, and every pricing
  path applies them: `v13`, the stream (`--candidate v13`; teams from the match info), `remaining`
  and the trader.

At league strengths with the points held, moving both sides' rate from 0.55 to 1.82 takes the
touchdowns from 18.6 to 26.7 yards and the 40+ yard touchdowns from 14% to 29%. The totals' sd goes
from 11.3 to 12.4.

### Held out

Built before Sep 10, played on Sep 10-22. Pre-match, from kick-off: 906 matches, both versions
priced off the same NB2 expected points, v13 minus v12 [95% over matches]:

| | v12 | v13 | v13 - v12 |
|---|---|---|---|
| total, log loss | 3.6569 | 3.6453 | -0.0116 [-0.0254, +0.0030] |
| total, RPS | 6.8855 | 6.8803 | -0.0052 [-0.0239, +0.0131] |
| margin, log loss | 3.3676 | 3.3691 | +0.0016 [-0.0101, +0.0132] |
| margin, RPS | 5.1424 | 5.1309 | -0.0115 [-0.0239, +0.0004] |
| moneyline, Brier | 0.2459 | 0.2457 | -0.0002 |

By fifths of the match's big-play rate (the two sides' geometric mean), each version's predicted sd
of the total against the real miss (rms):

| fifth | rate | v12 sd | v13 sd | real |
|---|---|---|---|---|
| 1 | 0.51 | 11.66 | 11.28 | 10.98 |
| 2 | 0.67 | 12.03 | 11.81 | 11.07 |
| 3 | 0.87 | 12.45 | 12.42 | 12.88 |
| 4 | 1.10 | 12.97 | 13.15 | 13.84 |
| 5 | 1.60 | 13.37 | 13.93 | 13.71 |

The real misses widen 2.7 points from the first fifth to the last. v12 widens 1.7, all of it from
the expected totals, which run with the rate. v13 widens 2.65.

In play: points still to come at 56,890 snapshots (`remaining`, 500 paths), RPS, v13 minus v12 [95%
over matches]. Both builds were run with today's code (v12's pre-match now follows the results, so its
numbers are a little better than in the v12 section):

| | v12 | v13 | v13 - v12 | mean error v12 / v13 |
|---|---|---|---|---|
| all | 3.821 | 3.811 | -0.010 [-0.020, -0.001] | +0.01 / -0.12 |
| Q1 | 3.782 | 3.775 | -0.007 [-0.020, +0.007] | -0.10 / -0.34 |
| Q2 | 4.739 | 4.724 | -0.015 [-0.028, -0.002] | +0.03 / -0.14 |
| Q3 | 4.392 | 4.378 | -0.014 [-0.028, -0.001] | +0.20 / +0.10 |
| Q4 | 2.364 | 2.359 | -0.005 [-0.013, +0.001] | -0.07 / -0.12 |
| Q2 2+ scores, trailer has ball | 5.152 | 5.110 | -0.042 [-0.077, -0.009] | +0.74 / +0.56 |
| Q3 2+ scores, leader has ball | 4.106 | 4.060 | -0.046 [-0.081, -0.008] | +1.04 / +0.95 |
| Q4 2+ scores, trailer has ball | 2.595 | 2.580 | -0.015 [-0.028, -0.001] | +0.24 / +0.18 |

v13 is better overall and in Q2 and Q3. It gains most where a side two scores behind needs big plays,
or a side two scores ahead can hit one. No state is clearly worse. v13 runs 0.13 points a game lower
in play than v12 (mean error -0.12, v12 +0.01): the points are held at kick-off, not from in-game
states.

### Still open

- **The sandbox build rates gamers on 9 days of plays.** A build on the whole history rates them
  far more surely; rebuild before reading the report.
- **v13 runs 0.13 points a game low in play** (above). Holding the points at the in-game states as
  well as at kick-off would take it back.
- **Form on the day is fitted as v12 fits it**, on results against the pre-match model with the
  sim's spread at league rates. A gamer whose big plays the rate now spreads may still carry some of
  that spread in its form (the two correlate +0.26 across gamers). If the most explosive matches
  price too wide, form should be refitted with each side's rate.

## v12: v11 with level sides' late drives run down to the kick

v12 (`sim12.py`, `v12.py`, `v12_stream.py`) started as v11 copied exactly. With its own switches off
it plays as v11 (`TestV12IsV11`).

### What v11 got wrong

From level states in Q4's last two minutes, v11 went to overtime about twice as often as real games.
Its side with the ball kicked the go-ahead field goal about as often as real sides do (59% of these
drives, real 64%), but too early:

| level, 2:00-1:01, the side with the ball kicks the go-ahead field goal | real | v11 |
|---|---|---|
| clock left at the kick, median / mean | 0s / 14s | 43s / 40s |
| the other side's points after it, to the end of regulation | 0.60 | 1.88 |
| overtime after it | 4% | 14% |

v11's close-endings fit reached the real field-goal share by raising the level side's kick odds +4.0
everywhere, the top of its grid. So it kicked on early downs, and on 4th down 98% of the time with a
minute or more left, where real level sides in range go for it about half the time and keep the
clock running. And the kick itself took the whole game's kick seconds (median 4). Real close sides
let the play clock run first: from 0:40 or less, 65% of their kicks leave nothing on the clock.

### What v12 changes

- **Close 4th downs by the clock** (`FOURTH_CLOCK`). The close 4th-down kick table has five clock
  slices (0:10, 0:30, 1:00, 2:00, 3:00) for each class, not two, each shrunk toward the v11 cell it
  falls in. Level sides in range kick 46% with 1:00-3:00 left, 72% to 0:31 and 91-94% after.
- **Late kicks take the clock as real ones do** (`KICK_CLOCK`). A side level or 1-3 behind kicking in
  Q4's last 2:00 runs the clock out from 0:40 or less as often as real kicks did (69%), and from
  further out takes the seconds a real kick took (a third of them 25s or more).
- **The level kick shift only in the last 10 seconds** (`CLOSE_LEVEL_LATE`). The build fits it, and
  the efficiency shift (either way now), to the real field-goal and touchdown shares of the drive
  under way. On the held-out build: kick +5.5, efficiency -0.10.

Fitting overtime from these states directly was tried. With a 4th-down shift free as well, the fit
took both kick shifts to the edges of their grids for under a point of overtime, and kicked early
again. The overtime left is in the reply (below), not the level side's choices.

### Held out

Built before Sep 10, played on Sep 10-22.

| from real level states in Q4 | v11 | v12 | real [95%] |
|---|---|---|---|
| 1:00-0:00: other side's points | 1.30 | 0.99 | 0.87 [0.48, 1.34] |
| 1:00-0:00: field goal only | 77.6% | 72.9% | 73.3% [64.5, 81.7] |
| 1:00-0:00: overtime | 21.4% | 19.6% | 11.5% [5.0, 18.7] |
| 2:00-1:01: other side's points | 2.36 | 2.00 | 1.31 [0.85, 1.83] |
| 2:00-1:01: overtime | 12.6% | 12.8% | 6.1% [1.9, 11.2] |

Points still to come (56,890 snapshots), RPS, v12 minus each [95% over matches]:

| | v10 | v11 | v12 | v12 - v11 | v12 - v10 |
|---|---|---|---|---|---|
| all | 3.825 | 3.829 | 3.825 | -0.004 [-0.008, +0.001] | +0.000 [-0.005, +0.006] |
| Q1 | 3.785 | 3.796 | 3.790 | -0.006 [-0.013, +0.002] | +0.005 [-0.002, +0.012] |
| Q2 | 4.738 | 4.746 | 4.743 | -0.004 [-0.010, +0.003] | +0.005 [-0.003, +0.013] |
| Q3 | 4.397 | 4.397 | 4.396 | -0.002 [-0.009, +0.006] | -0.001 [-0.007, +0.006] |
| Q4 | 2.374 | 2.372 | 2.366 | -0.006 [-0.013, +0.002] | -0.008 [-0.019, +0.002] |
| Q4 level | 2.068 | 2.090 | 2.042 | -0.048 [-0.090, -0.008] | |

Q4 level states gain clearly, and their mean error falls from +0.50 points to +0.12. Every quarter
leans better than v11; against v10, v12 is level overall and leans better in Q4. From kick-off,
games go to overtime 3.6% of the time (v11 3.2%, real 2.7% [1.8, 3.8]).

### Still open

- **Overtime from level states is still about twice real.** The reply of the side behind is the
  rest: after a go-ahead score from 2:00-1:01 it still scores 2.00 points (real 1.31), and from 1:00
  1-3 behind it takes the field goal and no more 18% of the time (real 13.5%).
- **Level drives end in a field goal 55% of the time** (real 68%), more often in nothing: the level
  side in range gives the ball up on downs or a turnover more often than real ones do. The turnover
  excess is the whole game's (1.9 a game, real 1.45).

## v11: v10 with close endings, late-half timeouts, overtime and kick-off touchdowns as played

v11 (`sim11.py`, `v11.py`, `v11_stream.py`) started as v10 copied exactly. With its switches off
it still plays as v10 (`TestV11IsV9`, `v11_off`). The changes come from a census of the simulation
against the real feed (Sep 10–22, held out): every event the feed records, counted per game
from kick-off, and the last two minutes of each half played forward from the real states.

### What v10 got wrong

- **Close endings.** A side level or 1–3 behind late in Q4 drained the clock to 0:00 and kicked
  from up to 55 yards whenever its downs could run the clock out. Real sides almost never kick
  early with more than 45 seconds left. One 3 behind hardly ever kicks to tie before the last 10
  seconds. In the last minute, sides 1–3 behind took a field goal and nothing more 25.6% of the time
  (real 13.5%) and went to overtime 28.5% of the time (real 15.1%).
- **Going for two late.** A side 7 behind that scores a touchdown late goes for two to win, and
  how often depends on the clock: 94% of the time in the last 30 seconds, 70% to 1:00, 37% to 2:00
  and 12% to 3:00 (2,320 matches). v10 had one rate, 66%, for the whole last three minutes. So with
  seconds left it kicked to tie, and went to overtime, a third of the time.
- **Overtime.** A game level after regulation was sent into overtime and, in the same step, counted
  among the periods just ended, so every overtime started in its second. Games went to overtime
  5.4% of the time (real 2.7%) and played 2.44 periods there (real 1.33).
- **Late-half timeouts.** Called at the fitted rate on every running clock, the simulation called
  1.8 timeouts in Q2 and 1.7 in Q4 (real 2.5 and 2.7).
- **Kick-off return touchdowns** never happened (real 0.042 a game, about one game in 25). The
  row after one is the touchdown, not a kick-off, and was skipped.

### What v11 changes

- **Close kicks by how far behind, the down, the clock and the length** (`CLOSE_FG`). In Q4 a side
  level, 1–2 behind or 3 behind, in range on downs 1–3, kicks at its own fitted rate for each
  down × clock-left slice (to 2:00) × kick length. The rate shrinks toward the clock slice, then
  the class. On 4th down in the last three minutes each class kicks at its own rate, not the
  whole-game 4th-down curve. The drain stays for overtime alone.
- **Burning the clock before the kick.** Real go-ahead field goals leave a median of 0 seconds. A
  close side that commits to the kick with the clock inside what its downs can run off kicks
  with none left.
- **Playing for the kick** (`SETTLE_PLAYS`). A side level or 1–2 behind and in range in Q4's last
  two minutes draws its plays from the real plays in that spot (about 3 yards and 18 seconds a
  play, a touchdown on one in ten, almost never a turnover). Before, it fell back to bins of every
  late play, most of them going for the end zone.
- **Level sides kick sooner and get there more often** (`CLOSE_LEVEL`). Real drives from level
  snaps in the last two minutes ended in a field goal 67.7% of the time and a touchdown 16.1%. The
  build fits a kick-odds shift (+4.0 in log odds) and an efficiency shift (+0.20) for level sides
  there. Together they bring the simulated drives from 51.5% / 20.8% to 67.2% / 15.9%. The scoring
  level by quarter is refitted after.
- **The conversion by the clock** (`CONV_CLOCK`). The go-for-two table's last-three-minutes cell
  is split at 0:30, 1:00 and 2:00, with overtime its own cell. Each shrinks toward the old single
  cell. A model built before this change loads with the single cell repeated.
- **Timeouts fitted to real counts** (`CALL_FIT`). From the first snap at or inside 2:00 of every
  real half, the build scales each half's call rates until the simulation calls as many timeouts
  as the real sides did (Q2 ×1.24, Q4 ×1.68 on the held-out build).
- **Overtime from its first period** (`OT_FIRST_PERIOD`). `MAX_OT` stays 3: at 2, 6.75% of games
  ended level.
- **Kick-off touchdowns** (`KICK_TDS`). A kick-off returned for a touchdown, or recovered and run in
  by the kicking side, is kept in the kick-off table at field 100: the side with the ball scores.

### Held out

Built before Sep 10 with the same export, history, handles and timeouts as v10, played on Sep
10–22.

Per game from kick-off (905 matches, 543,000 simulated games each):

| | v10 | v11 | real [95%] |
|---|---|---|---|
| games to overtime | 5.4% | 3.2% | 2.7% [1.8, 3.8] |
| overtime periods | 0.131 | 0.047 | 0.035 [0.022, 0.051] |
| kick-off return touchdowns | 0 | 0.042 | 0.042 [0.029, 0.056] |
| two-point tries | 0.365 | 0.355 | 0.351 [0.312, 0.392] |
| timeouts Q2 / Q4 | 1.81 / 1.73 | 2.28 / 2.55 | 2.53 / 2.71 |
| final margin 1 / 2 / 3 (%) | 8.3 / 5.5 / 20.4 | 7.9 / 4.8 / 20.6 | 8.4 / 5.0 / 22.2 |
| points | 35.33 | 35.58 | 35.15 [34.33, 35.99] |

From real states in Q4's last two minutes, played to the end:

| | v10 | v11 | real [95%] |
|---|---|---|---|
| trail 1–3, 1:00–0:00: field goal only | 25.6% | 18.8% | 13.5% [7.7, 19.7] |
| trail 1–3, 1:00–0:00: overtime | 28.5% | 17.6% | 15.1% [8.2, 22.6] |
| trail 1–3, 2:00–1:01: overtime | 15.3% | 7.5% | 6.1% [1.8, 11.7] |
| trail 4–8, 2:00–1:01: overtime | 10.6% | 6.9% | 4.5% [2.0, 7.6] |
| level, 2:00–1:01: overtime | 16.7% | 12.6% | 6.1% [1.9, 11.2] |
| level, 1:00–0:00: overtime | 21.9% | 21.4% | 11.5% [5.0, 18.7] |

Points still to come, held out (56,890 snapshots); the difference is v11 minus v10 with a 95%
interval from resampling matches:

| | v10 rps | v11 rps | v11 − v10 [95%] |
|---|---|---|---|
| all | 3.825 | 3.829 | +0.004 [−0.002, +0.010] |
| Q1 | 3.785 | 3.796 | +0.010 [+0.003, +0.018] |
| Q2 | 4.738 | 4.746 | +0.009 [−0.001, +0.018] |
| Q3 | 4.397 | 4.397 | +0.001 [−0.008, +0.010] |
| Q4 | 2.374 | 2.372 | −0.003 [−0.015, +0.009] |
| OT | 2.270 | 2.248 | −0.022 [−0.057, +0.014] |

Overall the two are level. v11's changes are about how games end, so the gain shows late. Q1
loses a little. That isn't any one of the simulation's switches: turned off one at a time on the
same build, none moves Q1. It comes with the rebuild, most likely the scoring level refitted
around kick-off touchdowns and the level sides' shifts. Switched off one at a time, the
conversion by the clock is the largest gain (−0.008 [−0.014, −0.003] overall, −0.019 in Q4), and
playing for the kick next (−0.002 [−0.003, −0.000]). The close-kick table is neutral on RPS; its
gain is in the endings above.

### Still open

- **Level in the last two minutes.** These states still go to overtime about twice as often as real
  ones. The trouble is the reply: after a late go-ahead score, the other side scores back 2.36
  points from 2:00–1:01 (real 1.31) and 1.30 from the last minute (real 0.87).
- **Snaps late in halves.** The extra timeouts buy more plays than real ones do: 15.7 snaps in Q2
  (real 14.6) and 13.3 in Q4 (real 12.9).
- **Q1 held out** is +0.010 RPS on v10 (above).
- **Turnovers and field goals.** These were already high in v10 and are unchanged: interceptions
  and fumbles 1.9 a game (real 1.45), field goals 1.29 (real 1.08).

## v10: v9 with the prior's pace counted once, and its totals' spread fitted out of sample

v10 (`sim10.py`, `v10.py`, `v10_stream.py`) started as v9 copied exactly. Its simulation is still
v9's (`test_v10_plays_as_v9`). What changes is how each match's pre-match prior becomes its
starting strengths. An earlier v10, which learned the day's scoring from the game so far, showed
no gain held out and was replaced.

### What v9 got wrong, and why

v9's totals ran over in Q3: points still to come +0.34 too high there, against +0.01 to +0.05
in the other quarters. By the pre-match total, held out (Sep 10–22), the games NB2 expected to be
high-scoring (around 42) came in 1.6–1.9 points under v9 in each of Q1–Q3. At half time, real
points to come moved only 0.74 for each point of v9's.

- **Pace counted twice.** From kickoff, v9's total moved 1.15 points for each point of NB2's
  total. Real games moved 0.89. The pre-match grid maps the prior one to one (slope 1.00). The
  extra 0.15 is the players' pace: fast players score more, so NB2's expected points already
  carry their pace, and the sim's clock then sped those players up again. Pace runs −0.54 with the
  prior total, and each 0.1 of the two paces moves the sim 1.7 points.
- **NB2's totals spread a little wider than real ones.** Out of sample, real totals move 0.89–0.92
  points for each point of NB2's.

How far the game so far runs above or below its pre-match pace carries little extra: 0.07 points
of the rest of the game for each point of surprise. That is why learning from it didn't help.

### What v10 changes

- **The prior's pace is counted once** (`PACE_NEUTRAL`). At build, each side's points from
  kickoff are simulated over a grid of the two players' paces (0.85–1.15, `fit_pace_response`,
  kept in the grid file). A match's expected points are divided by its pace response before its
  starting strengths are fitted. With both paces applied, the sim then averages the prior's
  points.
- **The prior's totals and margins are pulled in by their out-of-sample slopes**
  (`PRIOR_SHRINK`, `MARGIN_SHRINK`). Both priors predict margins and totals spread wider than
  real games turn out.
  - **How the slopes are read.** At build, the pre-match model (NB2 or glmer) is refitted at
    the start of each of the 4 fortnights before the cut-off (`SHRINK_WINDOWS`,
    `SHRINK_DAYS`). Each refit predicts its fortnight, out of sample and as it would live:
    NB2's ratings as fitted, glmer's form following the results.
  - **Pooled, on established gamers.** The real totals and margins are regressed on the
    predictions, pooled within the fortnights. Only matches whose two gamers had both played
    30 earlier matches count (`SHRINK_MIN_EXPERIENCE`); see below.
  - **Only as far as the data are sure of.** Each slope is moved toward 1 by its standard
    error and kept within 0.5–1.2.
  - **Applied.** Every match's expected total is pulled toward the build's average by its
    slope. Its expected margin is pulled by half of its slope's pull (`MARGIN_WEIGHT`).
    `v10shrink.json` keeps them all, with each fortnight's own.
  - **Why four fortnights, pooled.** One fortnight measures a margin slope only to about
    ±0.08. On AMFELO the single-fortnight slopes ran from 0.63 to 1.13 for NB2. Four pooled
    measure it to about ±0.04.
  - **Why established gamers only.** In one fortnight in June, a cohort of new gamers dragged
    glmer's margin slope from 0.78 to 0.27. NB2 can't price a newcomer at all, and glmer
    extrapolates their learning curve (see "glmer and new gamers" below).
  - **Why half the margin's pull.** The best slope for the mean margin (about 0.77) is
    stronger than the moneyline's (about 0.87), because big predicted margins come back
    further than small ones. The full pull is best for the spread's mean but costs the
    moneyline. Half keeps most of the spread's gain and leaves the moneyline as it was.
    Set `MARGIN_WEIGHT = 1` for the full pull.

  Out of sample on AMFELO, each prior was refitted every 14 days. The shrink was read off the 4
  fortnights before each, then applied to the next one (8 fortnights, 25 May – 10 Sep), over the
  matches each prior priced:

  | | pull (margin; total) | margin RMSE | total RMSE | moneyline log loss* |
  |---|---|---|---|---|
  | NB2 (9,808 matches) | 0.89–0.93; 0.83–0.96 | 9.260 → 9.246 | 12.263 → 12.248 | 0.6688 → 0.6687 |
  | glmer (10,116) | 0.88–0.93; 0.86–0.99 | 9.344 → 9.318 | 12.151 → 12.140 | 0.6692 → 0.6686 |

  \* A normal around the predicted margin (sd 9.35), as a stand-in for the sim's own moneyline.

  The full margin pull (`MARGIN_WEIGHT = 1`, slopes 0.76–0.86) takes margin RMSE to 9.241 for
  NB2 and 9.303 for glmer. It costs NB2's moneyline 0.0003.

  Head to head, on the 9,808 matches both priced, glmer (floored, below) beats NB2 with or
  without the shrink:

  | | margin RMSE | total RMSE | moneyline log loss |
  |---|---|---|---|
  | NB2 | 9.260 → 9.246 | 12.263 → 12.248 | 0.6687 → 0.6686 |
  | glmer | 9.227 → 9.213 | 12.138 → 12.127 | 0.6645 → 0.6647 |

  The build refits the pre-match model 4 more times for this, 2 at a time (`SHRINK_WORKERS`).
  That takes about a minute for NB2 and about 10 for glmer.

### glmer and new gamers

In glmer's `form` model, a gamer's experience term (`ExpLog`, the log of their matches so far)
extrapolates a learning curve. Against a veteran, a newcomer is predicted to score about half
the points. Out of sample they don't fall that far behind. Over the 12 fortnights above (14-day
refits, 15,418 matches), split by the less experienced gamer's earlier matches:

| earlier matches | matches | sd of predicted margin | real margin per predicted point |
|---|---|---|---|
| under 10 | 146 | 7.6 | 0.39 |
| 10–29 | 264 | 6.9 | 0.21 |
| 30–59 | 394 | 4.8 | 0.52 |
| 60 or more | 14,614 | 3.1 | 0.79 |

A cohort joined in June. In that fortnight 13% of matches had a gamer with under 30 matches
behind them, and glmer's moneyline log loss for the fortnight was 0.710 against about 0.66
elsewhere. NB2 doesn't price newcomers at all, so it falls back to the league average.

**Fix: price newcomers as if they had 30 matches** (`EXP_FLOOR_MATCHES` in `glmer/config.R`,
`predict.R --exp-floor`). The floor applies only to the matches being priced. The fit never
floors, so veterans' prices don't move. Over the same 12 fortnights:

| floor | all: ML log loss | margin RMSE | under 10 earlier: ML | 10–29: ML | fortnights worse |
|---|---|---|---|---|---|
| none | 0.66855 | 9.425 | 0.708 | 0.752 | – |
| 10 | 0.66831 | 9.420 | 0.683 | 0.752 | 1 (+0.0001) |
| 20 | 0.66825 | 9.417 | 0.682 | 0.749 | 4 (≤ +0.0006) |
| **30** | **0.66818** | **9.415** | 0.685 | 0.743 | 4 (≤ +0.0009) |

The cohort fortnight gains most: 0.7099 → 0.7050.

Tried and rejected, because each refit cost veterans more than it gained on newcomers:
- **Capping experience in the fit** (`fit.R --exp-cap 30` / `100`). It wins only in the
  cohort fortnight and is worse for veterans (+0.0017 to +0.0019 log loss). Over fortnights
  4–7 it came out at +0.0004 (cap 30) and −0.0005 (cap 100) overall.
- **Dropping the experience terms** (`form_noexp`). Worse in 4 of 5 fortnights, +0.0029
  overall. Experience carries real signal among veterans.
- **A separate, stronger shrink for newcomers' matches.** The June cohort's slopes (0.2–0.3)
  didn't carry over to later newcomers (0.68), and it worsened their moneyline (0.697 →
  0.725).

Gamers with 10–29 matches stay overdispersed even with the floor (real margin 0.21 per
predicted point). The general shrink pulls them in with everyone else.

### Held out

Built before Sep 10 with the same export, history, handles and timeouts as v9, played on Sep
10–22 (56,890 snapshots). "Pace" is v10 with only the pace change.

| | v9 mean | pace mean | v10 mean | v9 rps | pace rps | v10 rps |
|---|---|---|---|---|---|---|
| all | +0.09 | +0.04 | +0.03 | 3.837 | 3.828 | 3.825 |
| Q1 | +0.01 | −0.09 | −0.09 | 3.795 | 3.789 | 3.785 |
| Q2 | +0.05 | −0.02 | −0.04 | 4.753 | 4.742 | 4.738 |
| Q3 | +0.34 | +0.28 | +0.27 | 4.408 | 4.398 | 4.397 |
| Q4 | +0.00 | −0.01 | −0.01 | 2.382 | 2.375 | 2.374 |

By the pre-match total, the highest fifth (around 42) goes from +1.85 to +0.56 in Q1, from +1.62
to +0.61 in Q2, and from +1.85 to +1.23 in Q3. At half time, real points to come move 0.88 for
each point of v10's (v9 0.74).

### Still open

Q3 is still +0.27 overall. It sits in the two-score states (+1.08 with the leader on the ball,
+0.52 with the trailer) and late level ones. Those are game-state effects, not the prior.

### Line rules (own, `@even`, `@prod`, `v10@anchored`, `v10@hyst`)

v8 to v13 quote their own lines in the gap between the key numbers (below). `@even` reads
them at the half-point line nearest 50% instead, the old own line, and `@prod` at prod's. v10 has
two more rules. All of them come off one simulation:

| stream | line |
|---|---|
| `v10` | its own line, in the gap between the key numbers: of the half-point lines with P(over) in 40–60% (`KEY_BAND`), the one with the least chance on the two whole numbers either side, kept while it stays in the band within 0.02 of the best (`KEY_HOLD`) |
| `v10@even` | the even line: the half-point line nearest 50%, at every snapshot |
| `v10@prod` | prod's line |
| `v10@anchored` | prod's line, moved a point at a time toward even only until P(over), or P(home covers), is inside 40–60% (`ANCHOR_BAND`) |
| `v10@hyst` | its own even line, kept until the even line is 2 points away (`HOLD_MOVE`) or P(over) at the kept line leaves 35–65% (`HOLD_BAND`) |

Offline, on the held-out build's remaining totals (Sep 10–22, 52k snapshots, 905 matches, 5%
margin), a bettor who learns where each rule misprices on half the matches and bets those spots on
the other half:

| rule | book per 100 of their bets | line moves a match | from prod's line |
|---|---|---|---|
| prod | −6.7 | 29.4 | 0 |
| v10 at prod's line | +0.4 | 27.8 | 0 |
| v10 even line | −0.1 | 30.5 | 1.7 |
| anchored 40–60% | +2.2 | 27.4 | 0.6 |
| even line + hysteresis | +1.5 | 16.4 | 1.7 |

The even line is the best calibrated, but it moves most and crosses key numbers as the
distribution shifts. Those moves are what the bettor picks off. The rules are within a couple of
points of each other here, so the book comparison against real bets decides.

#### Key numbers

Points come in lumps. A touchdown is 6 and its kick 1, and a field goal 3. 21% of final margins
are exactly 3, 11% are 7 and 8% are 10. 37% of final totals are 3 more than a multiple of 7. v10's
distributions carry the same lumps: at every value its share is within about a point of reality,
pre-match and in play. So the distributions are spiky, and where the line sits on them matters.
A line next to a spike prices far from the lines either side of it. A small error in the spike
moves its price a lot, and the even line flips across the spike as the game moves. A line in a gap
between spikes prices about the same as its neighbours, so it can stay put.

The own line (`key_line`, in v8 to v13) takes the line in the gap. On the held-out build's full distributions (Sep 10–22, 57,795
rows including kick-offs, 905 matches), each rule against a bettor who learns where it misprices
(by phase, score, price, side of prod's line and the line's key-number position) on half the
matches and bets the other half, five splits shared by every rule. The figures are book per 100
of their bets, the difference from the even line [95% over matches], and line moves a match in play:

| | total | vs even | moves | handicap | vs even | moves |
|---|---|---|---|---|---|---|
| even line | −1.07 | | 31.6 | +3.88 | | 28.1 |
| hyst | −0.78 | +0.30 [−0.75, +1.48] | 17.7 | +4.42 | +0.54 [−0.57, +1.75] | 14.3 |
| own: key 40–60%, hold 0.02 | −0.42 | +0.65 [−0.53, +1.83] | 18.7 | +4.43 | +0.55 [−0.66, +1.84] | 21.4 |

Every key setting tried beat the even line in both markets (bands 42–58% to 38–62%, holds 0.005
to 0.02: 18 of 18), by +0.3 to +1.2. Each one alone is inside the noise. The shipped setting is
the middle of that grid, not its best. Its lines have about half the chance next to them that the
even line's do.

## v2.1: each snap's time, then its distance, from real snaps (`v21-build`, `v21-report`)

v2.1 is a separate model, not a v14 variant. It is still a snap simulation, but every snap is two
draws from real snaps, and it has no strength. The pre-match expected points are not an input to
any draw. The report sets the sim's output against them, so the gaps show where the draws are
missing something. Nothing is ever fitted to close those gaps.

- **Inputs:**
  - the fixture: the gamers and teams;
  - the state: clock, down, distance, field, score and timeouts;
  - the action log: what the play before did to the clock, and the snaps so far.
- **The time draw.** The clock from the end of the play before to the end of this one, covering the
  wait for the snap and the play itself. It is drawn from real snaps alike in:
  - the play before: nothing gained, a short gain, a first down, a loss, a fresh possession or a
    timeout;
  - the half (first, second, overtime) and whether the two-minute warning has passed;
  - the offense's lead, down, distance and field zone (8 zones: the own 10, to the 30, to midfield,
    to the other 30, the other 29–20 and 19–11 apart as field-goal range, the 10 to the 5, inside
    the 5);
  - the snaps played so far and the snaps left in the half (below).

  Snaps left stands in for the time left (`--split snaps`, the default): it tells Q1 from Q2 and the
  start of a half from its middle. Two things it can't tell stay in. The half: at the end of Q2 a
  level side attacks, at the end of Q4 it plays for the kick. The audit showed this when a fallback
  mixed the two: level sides in Q4's last 2:00 scored a touchdown on 15% of the sim's snaps against
  5% real. The two-minute warning: the clock's rules change there. `--split time` keeps the seven
  parts of the game (Q1, Q2 and Q4 each side of 2:00, Q3, overtime) for comparison.

  This draw is where the bimodal split lives. A stopped clock gives a play's own 5–8 seconds, a
  running one 25–40, and the play before (from the action log) sets the mix. So from any point in
  play the first snap can come off a running or a stopped clock in the real proportions for that
  state.
- **The distance draw.** What came of the snap (the yards, a turnover or a defensive touchdown),
  drawn from real snaps alike in the time just drawn (6 buckets) and the same state. A snap over in
  5 seconds can only have gone so far.
- **Backoff for thin data.** Where too few real snaps share every feature (40, `--min-records`), a
  draw falls back level by level to fewer features. The build prints how deep the real states fall.
  The fallbacks keep the situation to the end: the down, the distance, the field and the snaps left
  in the half, plus coarsely the half and the lead. So a 3rd down is never drawn from
  1st downs, and a side with time for one snap never from one with time for ten, while there are
  real snaps alike enough to draw from. (Before the down audit, the part of the game went first.)
  At the end of a half (its last 2:00, or under 5 snaps left) the distance draw coarsens the
  distance and the time, then the field, before it lets the lead go. It keeps the half, the snaps
  left (0–1, 2–4, 5+) and behind / level / ahead, down to the down and the field's three parts
  (own half, the other half, inside the 10) before it gives the situation up. In September 30% of
  level sides' Q4-last-2:00 states had drawn from every game's snaps at that spot. The field keeps
  its zones as long as it can:
  with the other half as one zone, a real touchdown from the 12 drawn at the 45 ran on to a long
  one. The audit showed why: level sides in Q4's last 2:00 scored a
  touchdown on 15% of the sim's snaps against 5% real. Real sides there play for the kick, and
  sides ahead run the clock out, while the sim had drawn from every lead at that spot. The build
  prints how deep real states fall, for the rest of the game and the end of a half apart. It
  also sets Q4's last 2:00, by the lead, against the league groups those real snaps draw from:
  touchdowns, losses and seconds. For touchdowns it also gives the sim's own chance at each state's
  field: the group's gains that reach the goal from there, plus its touchdowns that run on. It
  gives it twice, drawing from the whole group and from near the state's field (`--field`, below).
  A group that mixes situations shows up as a gap there, apart from the game flow and the
  decisions. The same four figures follow for every snap of the game by field zone, on a sample
  of 8,000 real snaps. Last, the late choices that settle a game in regulation
  (going for it 3 behind, kicking level, going for two at −1 and −2) and the overtime rate, by
  fortnight of the build's data: a trend says the players are still learning, and the oldest
  matches weigh too much.
- **The lean key (`--key lean`).** The full key has close to a million cells, so only about half
  of real states draw on all of it. The lean key keeps only:
  - the clock into the snap, running or stopped (the time draw), or the time the play took (the
    distance draw);
  - the period, with the two-minute warning apart (Q1, Q2, Q2's last 2:00, Q3, Q4, Q4's last 2:00,
    overtime): a level side with three snaps left at 2:30 attacks, at 1:30 it sets up its kick. With
    the period alone, the tilt build's level sides in Q4's last 2:00 scored touchdowns on 9.2% of
    snaps near their field against 6.2% real;
  - the lead (7 groups);
  - down, distance and field;
  - snaps left in the half.

  That is 78,400 cells for the time draw. On 172,751 real states, 57% sit in a cell with 40 or
  more real snaps: 82% in Q1, 51% in Q2, 64% in Q3, 40% in Q4. The six classes of the play before
  collapse to running or stopped, since their times split that way (26–29 s against 8–10 s on
  average). The fallbacks coarsen the field and the snaps left before the lead gives way to
  behind, level or ahead. A side 9+ ahead in Q4 runs it out, where one 1–3
  ahead still plays for a first down. The first lean build collapsed the lead to those three at its
  first fallback. Its sides 9+ ahead in Q4's 4:00–2:00 then scored a touchdown on 36% of their
  drives, against 23% real, the same as sides 1–3 ahead. `--key full` (the default) keeps the full
  key, for comparison.

  The lean key's distance fallbacks keep the distance too, all five buckets (1, 2–3, 4–6, 7–10,
  11+). Pooling 1 to go with 2–3, a 3rd and 1 drew 3rd-and-3 snaps that kept their own result, a
  failure, and at 1 to go a failure is no gain: 25% of the sim's 3rd-and-1 snaps against 17% real.

  The field windows keep level with the goal line. Where it is nearer than the window's reach, the
  window stops as far short on the other side, and a thin one widens the same either side as far
  as the goal allows. Before, a state at the 1 drew from the 1 to the 5 when the cell had few snaps
  at the 1. From the real states at the opposing 4–1 the sim scored 40.9% against 43.4% real, and
  on 1st and goal at the 1, 44.6% against 51.4%. The clock windows keep level with the half's end
  the same way.

  Every distance fallback but the last also keeps the time the play took, all six buckets, in
  both keys. Most had kept only stopped or running (14 seconds or under, or more), which pools an
  incomplete pass (6 seconds or less) with a 40-yard gain (10–14 seconds): the time drawn and the
  distance drawn came apart. In September's real games 10.8% of the 10–14 second snaps gained 40
  yards or more, in the sim 5.6%; of those of 6 seconds or less, 0.8% real and 2.4% in the sim. The
  sim made 4% more snaps a game than the real games and 1.6 points fewer (33.2 against 34.8),
  short in every quarter from Q2.

  The lean key's distance fallbacks hold the period to the last two. Blended, a thin late cell
  passes its draws down the chain: a level side in Q4's last 2:00 drew 27% of them from levels
  without it, every snap of the game at its down and distance, where the touchdown chance near its
  field was 16–17% against 4–9% in the levels that kept the situation. Its snaps scored 9.2% near
  the field against 6.4% real. They hold the distance, all five of it, as far: a build whose last
  levels kept the period and lead but let the distance go played 2nd and 1 off every 2nd down's own
  results, 44% moving the chains against 51% real (3rd and 1, 47% against 55%).

  The lean time draw keeps the down and all five distances as far too. The distance draw reads the
  time it was given at the state's own distance, so a time drawn from 1 to 3 to go pooled (or from
  every distance) gave 2nd and 1 the quick snaps of 2nd and 3 -- its incompletions -- and played
  them out as 2nd and 1's own quick snaps. Its fallbacks coarsen the field and the snaps left, then
  the lead to behind, level or ahead, and the period, before the distance.

  Before the distance draw lets the lead go, the time the play took coarsens to stopped or running,
  the period and the lead held. A level side in Q4's last 2:00 drew 25% of its snaps from levels of
  every lead or the whole game, which scored 13–18% near its field against its own 4–8%, and its
  drives with 1:00–0:31 left scored a touchdown 19.6% of the time against 10.3% real (a field goal
  58% against 78%): late, a level side plays for its kick.

  Every distance fallback but the last keeps the down, in both keys. Two did not: the lean key's
  period, lead, field zone and snaps left, and the full key's field zone and end of the half. Blended,
  a thin 4th-down cell passed some of its draws down the chain to them, so 4th downs were drawn
  partly from 1st and 2nd downs. From the real states the sim converted fewer at every distance: 2–3
  to go, 43% against 50% real; 7–10, 35% against 40%. The extra turnovers on downs fell late, where a
  level side drives for its kick.
- **Blended levels (`--blend K`).** Without it, a draw takes the first level whose cell holds 40
  real snaps (`--min-records`): a cell of 39 is dropped whole for its parent, one of 40 used whole.
  With it, the league's draws keep every cell of 5 or more, and a draw takes its finest cell
  n / (n + K) of the time, else the next level's for the state, and so on, the last level taking
  the rest. With K = 40, a cell of 10 real snaps is drawn from 20% of the time, one of 40 half the
  time, one of 360 nine times in ten. Every draw is still a real snap, a thin cell weighs as much as
  its snaps say, and nothing jumps at a threshold. The build's level shares, its late-Q4 check and
  `v21-coverage` read the blend: each level's expected share of the draws, and the real snaps
  behind them weighted the same way. `--blend 0` (the default) keeps the first-level rule.

  Blended with field windows, the distance draw never widens a window but the last level's, and a
  cell with none of its snaps near the state's own field passes all its weight on (`NEAR_BLEND`,
  the model's `near_blend`). A thin cell had widened its window to the 20 nearest: a late cell of
  2nd and 1 at midfield reached its snaps at the goal line, a touchdown kept there and a stuffed
  run played as no gain. From September's real states at the other side's 49–30 the sim's draws
  scored 6.0% against 4.3% real (drawing from the whole group, 5.1%). A cell still weighs by all
  its snaps: weighed by those near the field alone, the thin late cells passed their draws to the
  levels without the lead or the period. A level side in Q4's last 2:00 then drew 36% of them from
  levels with both, against 61%, and its snaps scored 9.4% against 6.1% real, while 1 to go moved
  the chains 63.2% against 67.2%. The tilts' bases and the build's lines weigh the cells the same way.
- **Field windows (`--field F`).** A group's real snaps come from across its field zone (20 yards
  at midfield), or wider in the coarse fallbacks. A gain is played out at the state's own field,
  so a 35-yard gain made from midfield becomes a touchdown at the opposing 30. And a touchdown
  from the 12, drawn at midfield, runs on with the league's chance of a gain that long. Near the
  goal the field compresses, so this mixing makes touchdowns: for a level side in Q4's last 2:00,
  the real snaps scored 6.2%, their groups' real snaps 7.0%, the sim's draws 9.1%. With `--field`,
  the distance draw takes its real snap from the group's snaps near the state's own field: within
  F x the yards to go (1 yard at least), or the 20 nearest where fewer are there, or the whole
  group where it holds fewer than 20. With F = 0.2, that is ±10 yards at the opposing 50 and ±2 at
  the 10. A gamer's or team's own draws do the same. The drawn snap keeps its own result: its
  touchdown is one here, with no run-on, and a gain that fell short of its own goal falls short of
  this one, as far of the way there (the chains then move by the yards). The line too: a snap
  drawn from another distance keeps its own result against the line. A new set of downs stays one,
  past the line by as far; a snap short of it stays short, as far of the way there (rounded, and a
  gain stays at least a yard where the line leaves room for one). Before, the
  yards were replayed: a 4th-and-1 sneak that gained a yard failed at 4th and 3, and a 3rd-and-3
  throw for 2 moved the chains at 3rd and 1. Short yardage isn't played like a longer distance, so
  from the real states the sim converted fewer 4th downs at every distance (2–3 to go, 44% against
  50% real). With the field windows doing the field's work, the lean key's distance fallbacks now
  let the field go before the distance. Windows alone, with the
  gains replayed at the new field, took the level late-Q4 figure only from 9.1% to 8.3%. Near the
  goal, any short gain from a yard or two further back still scored: from the opposing 4–1, the
  real snaps scored 43%, the sim 51%. An adjusted build's tilt still leans each
  draw toward the top of its whole group, by resampling six near snaps on where they sit in the
  group. `--field 0` (the default) draws from anywhere in the group.
- **Clock windows (`--clock C`).** The time draw's cell knows the snaps left, not the clock. In a
  half's last 2:00, a side at 0:20 on a running clock could draw a 30-second snap a real side made
  at 0:45, and the half ran out before it could kick. After the buzzer-kick fix, the clock ran out
  on level sides in Q4's 1:00–0:31 on 15.5% of their drives in the sim, against 0.3% real. With
  `--clock`, the time draw there takes its real snap from the cell's snaps with as long left in
  the half: within C x the seconds left (2 at least), or the 20 nearest where fewer. With C = 0.2,
  that is ±4 seconds at 0:20 and ±24 at 2:00. A thin cell's window widens to its 5 nearest only:
  its 20 nearest at 0:12 could reach back to 0:40 and the 30-second snaps made there. A gamer's or
  team's own draws do the same.
  `--clock 0` (the default) draws from anywhere in the cell.
- **Gamers and teams, two ways (`--mode`):**
  - `full`, straight from what was seen. Half of each draw takes the offense's own real snaps: its
    gamer's where it has 30 alike, else its team's, else the league's. The other half takes what the
    defending gamer, else its team, allowed. "Alike" always means the same down. For the time it
    also means the same play before, part of the game and (at least coarsely) lead; for the
    distance, the same time drawn, distance and field, at least coarsely. The end of a half stands
    apart in both: the half's last 2:00, or under 5 snaps left, split by the half, snaps left
    (0–1, 2–4, 5+) and lead (the 7 lead buckets, so 1–3 ahead apart from 9+ ahead), with
    overtime by lead. There a side's own snaps come from its own ends
    of halves, else the league's draw, which keeps the full situation. The build prints how often
    real snaps find enough of their own, for the rest of the game and the end of a half. A
    gamer's own draw gives way to the league's rather than drop the situation. Before the down
    audit, it fell back to all the gamer's snaps with that time, whatever the down, so 3rd downs
    played like 1st downs.
  - `adjusted`, the league's draws for everyone, then leaned. Each draw is pushed toward its longer
    or shorter end by a tilt for each of: the offense's gamer, the defense's gamer and both teams.
    The tilts are fitted on their real snaps against what the league's draws gave the same states,
    and shrunk toward none by 300 snaps. This one is built for thin data.
  - `tilt`, the league's draws for everyone, reweighted per matchup (step 2 of the plan). Every real
    snap a draw can land on is weighted by exp(theta . phi). For the distance draw, phi is its
    yards / 10 (cut to −10..50), whether it was a touchdown, a turnover, a new set of downs short of a
    touchdown, and a loss. For the time draw, phi is its seconds / 10. Theta is the sum of four tilts: the offense's gamer and team,
    and the defense's gamer and team (what each allows). The draw keeps its group's shape, field
    window and own result, and leans only as far as the real snaps say. The tilts are fitted by
    maximum likelihood. Each real snap's own outcome is scored against what the league's draw gave
    its state: blended over the levels, near the field, as a histogram by yards, touchdown and
    turnover (the time's cut snaps left out). Each tilt is fitted in turn by Newton steps, the others
    held, over four rounds. Each is shrunk toward none by a penalty of 200 snaps' worth of the
    league's variance, so a gamer with 200 snaps moves half way to their own. In the sim, a draw
    takes 16 real snaps evenly from its group or window and keeps one on its weight. The build
    prints each tilt's spread on each feature, the gamers furthest each way and the log-likelihood
    gained a snap. `--tilt-prior N` sets the shrinkage (snaps; default 200). Unlike `adjusted`, the
    tilt can move touchdowns and turnovers on their own, not
    only the yards, and is fitted on the whole outcome, not its mean. The fit weighs recent form: a
    real snap's weight halves every 30 days before the build's last match day.

    The first tilt build (yards, touchdowns, turnovers only) on held-out October: a point more
    expected moved the sim 0.45 (0.27 with `adjusted`; real 0.76), and its moneyline log loss came
    within 0.006 ± 0.006 of the prior's (0.017 with `adjusted`).
- **The points nudge (step 3, `v21-report --nudge`, a tilt build).** A switch for pricing: each
  side's offense tilt is moved along the strength direction until the sim's kick-off mean meets
  the side's expected points (the input). The strength direction is the gamers' offense tilts
  regressed on the points their side really scored a match, with no component the wrong way (more
  turnovers or losses, fewer yards, touchdowns or first downs). It is scaled so a unit is a point a
  side in the sim, and the build prints it and the slope it then has. Per match, secant steps start
  from that slope (three steps, each on the same random numbers, capped at ±10 points), and the
  last run prices the match. The first nudged run, on held-out October, priced moneylines at the
  prior's log loss (+0.001 ± 0.003) and margins and totals better (RPS 5.53 against 5.55, 6.89
  against 6.92). Its direction moved a side only half a point a unit, so 43% of the nudges stopped
  at the cap short of the inputs. With the unit a point and the cap at 10, the next run met the
  inputs (within 0.1 a match, none at the cap): margin RPS 5.53 against the prior's 5.55, total
  RPS level (6.92), moneyline log loss 0.004 ± 0.002 worse. Following the inputs all the way carries
  their errors too: in October a point more expected was 0.76 more in the real games. `--nudge-share
  S` goes only S of the way from the sim's own kick-off means to the inputs. On held-out October:

  | October | prior | nudge 50% | nudge 75% | nudge 100% |
  |---|---|---|---|---|
  | moneyline log loss | 0.6804 | 0.6819 | 0.6806 | 0.6842 |
  | margin RPS | 5.550 | 5.527 | 5.511 | 5.527 |
  | total RPS | 6.922 | 6.935 | 6.902 | 6.915 |

  So three quarters of the way prices moneylines at the prior's level and margins and totals
  better. The default stays the whole way: what the trader sets is what the sim meets. The same
  nudge carries into the match's real states. The report prints the nudges taken: their mean,
  spread, home and away, and the share at the cap. The base model never sees the inputs. The
  nudge is a separate layer, on only when asked.
- **Bounds:**
  - The time never passes the play clock, since no real draw does.
  - A running clock that runs out before the snap ends the quarter, and the same ball carries over.
  - A running clock reaching 2:00 of Q2 or Q4 stops there. The snap then takes a time from the
    stopped-clock part of the same draw.
  - A play under way when time runs out is finished.
  - Some real snaps were cut short: the quarter ran out on them, or a running clock crossed 2:00.
    These are kept as "at least this long". Where one is drawn with more time to go than it took,
    the draw moves to a real snap at least as long.
  - The distance stops at the end zones: past the goal line is a touchdown, behind its own a
    safety.
  - A real touchdown drawn from further back runs on with the real chance that a gain that long
    goes further, in the same part of the game: at the end of a half by the half, snaps left and
    lead, each shrunk toward the league's by 200 snaps. A level side late plays for the kick, and
    its gains go less far than the league's.
- **Snaps left in the half** is a core split of both draws, in place of a cut at 0:30. It is
  the expected count, never the count that actually followed, read off a table by:
  - the half and its 10-second slice of time left;
  - the clock running into the snap (a short gain, a first down, a loss) or stopped (nothing gained,
    a fresh possession, a timeout);
  - the timeouts of the side with the ball and of the other side (0–3);
  - the lead (behind, level, ahead).

  Each cell is shrunk toward its slice's figure for that clock, and that toward the slice's alone,
  by 20 snaps. Real snaps and sim states read the table the same way. The real games' counts come
  first. Then, 3 times over, the build plays 6,000 games from kick-off, replaces the table with the
  sim's own counts and rebuilds the draws on it. The draws bucket it as 0–1, 2, 3–4, 5–7, 8–12,
  13–17 and 18+. The build prints the table at a few late states.
- **First downs come from the next row's down.** Spots are whole yards and a distance of inches
  reads as 1, so a sneak for inches shows as a 0-yard gain that moves the chains, and a gain of the
  yards to go can be spotted short. The build prints how often each happens by distance. The sim
  takes the drawn real snap's own result where it was at the same distance, and otherwise goes by
  the yards. A down played again (a penalty: the same down, short of the line) keeps its number.
  4th downs short of the line that kept the ball are no longer left out.
- **Scores the rows do not mark.** In September the real games' touchdown, field-goal and
  conversion rows account for about 33 points a match, and their scores for 34.8: some scores come
  with no TOUCHDOWN or FIELD_GOAL row. A snap whose next row has 6 or more to one side and nothing
  to the other is read as that side's touchdown (the offense's own, or a defensive one), and a
  4th down whose next row in the quarter has 3 more to the side with the ball as its kick. Before,
  such a touchdown snap was dropped from the draws and the sim scored too little. The build
  prints how many it read.
- **The decisions**, each made by the gamer whose call it is:

  | decision | options | drawn from real decisions alike in |
  |---|---|---|
  | 4th down (the offense) | go for it, field goal, punt | yards to the goal, distance, lead, part of the game, time left in the half |
  | downs 1-3 in the half's last minute, in range (the offense) | play on, kick now | time left, the clock running or stopped, its own timeouts (0, 1, 2+), the kick's length, down, what a field goal does, which half |
  | kick-off (the kicker) | returned, touchback, no landing zone, kept (onside) | behind late in Q4, the half's last 2:00, lead |
  | conversion (the scorer) | kick, go for two | v14's cells by the clock, lead after the six; then Q4's last 3:00 as one, before the whole game |
  | timeouts before a running-clock snap (the defense, then the offense) | call one or not | part of the game, lead, time left in the half; the offense's also whether a field goal is in range |

  Each is drawn in three steps:
  - **The rules first**, the same for every gamer. They aren't written by hand: they are the
    situations where real sides never take an option, or all but never, learned from the real
    decisions (see `v21-sweep` below). In those situations the option is struck out. Real decisions
    that took a struck-out option anyway are left out of the gamers' leans. `--no-rules` builds
    without them.

    Overtime's own logic stays as a rule. Once the other side has had the ball, a side behind can't
    punt or kick a field goal 4+ behind, and after a touchdown 1 or 2 behind it goes for two.
  - **The league's share of each option** among real decisions in similar states. Like the snap
    draws, it falls back to fewer features where fewer than 25 real decisions share all of them.
    The 4th down and the early kick coarsen the distance, and the 4th down the field too (inside
    the 20, an easy kick, a long one, out of range), before they let the lead go. The lead they
    keep is what a field goal does: not enough (4+ behind), a tie (3 behind), the lead (1–2
    behind or level), or more of it (ahead). Real sides avoid overtime: 3 behind late they go for
    the touchdown, and level they kick to win. In Q4's last 2:00 a level side kicked on 57% of real 4th downs against 34% in
    the sim, and a side behind went for it on 90% against 73%, because thin late cells had fallen
    back to every lead.

    The early kick later fell short the same way. From the real states, level or 1–2 behind in Q4,
    real sides kicked on 23% of in-range snaps in the last minute, and the sim's cells on 17%. Its
    thin cells fell back to levels without what a field goal does, and none knew the clock or the
    timeouts. A level side at 0:20 on a running clock with no timeouts kicks now; with two, it can
    run a play first. So level sides kicked 42% of their drives from 1:00–0:31 against 68% real,
    and the clock ran out on them twice as often. The early kick now reads the clock running or
    stopped and its own timeouts, and keeps what a field goal does to the end of its fallbacks. The
    offense's timeout reads whether a field goal is in range, the timeout a level side calls to
    kick.

    And a side that a kick puts level or ahead in Q4 or overtime, wherever it is (at Q2's end, any
    lead, in field-goal range), with a timeout left, calls it where a running clock would run the
    half out before or during its snap (`LATE_TIMEOUT`); the snap then comes off a stopped clock, and the kick, or not, is the next
    snap's choice. The timeout before a running-clock snap is drawn from the real ones, which don't
    know the time the snap is about to take. Real sides level with the ball with 1:00–0:31 left ran
    out of clock on 0.3% of their drives, the sim's on 9.8%, and of those 62% were in range with more
    than 0:05 at their last snap and 88% still had a timeout. With it (the same model), 5.0%; their
    field goals 58% from 54% (real 78%), overtime from those states 18% from 22% (real 5%), from
    kick-off 3.9% from 4.2% (real 2.1%), timeouts 4.17 a game from 4.06 (real 3.98). In range
    alone, 5.3% still ran out (t10), 41% of them out of range at their last snap and 74% with a
    timeout left; real level sides almost never let the clock run out wherever they are.

    The 4th down's fallbacks now keep what a field goal does and the time left to the end, and
    coarsen the field and distance first. From the real states, level or 1–2 behind in Q4's last
    2:00, real sides kicked on 54% of 4th downs and the sim's cells on 49%.

    The feed can also miss a half's last play, a kick at the buzzer, though never the score it
    left: the next half's rows have it, and at the game's end the history's final score. In the
    September audit, level sides with the ball in Q4's last 0:30 went on to win 81% of the time,
    yet their drives read as only 68% scored and 24% run out of clock. So the build reads a kick
    off the score after a half's last row: 3 more to the side with the ball and none to the other,
    with a minute or less left, is a kick it took and made. Before, that state counted as a side
    that let the half run out. It prints how many such kicks it found. The audit reads the real
    drive endings and choices the same way.

    `--decision-blend K` blends the cells, as `--blend` does the draws. Every cell of 5 real
    decisions or more is kept, and a state's shares are its cells' over the levels: a cell of n
    weighs n / (n + K), the rest passes to its parent. `--decision-blend 0` (the default) takes
    the first level with 25. Conversions blend lighter, n / (n + 5) (`CONV_BLEND`): the lead after
    the six all but makes the choice, and a thin late cell passing 25 tries' worth down the chain
    reached the whole game's at its lead, where sides kick. After a six in Q4's last 3:00, 2 behind
    went for two 77% of the time in the sim's cells against 87% real, 5 behind 79% against 86%, 10
    behind 72% against all 16 real ones, 1 behind 59% against 62%.

    A two-point try is made with its own chance by the part of the game: before Q4's last 3:00,
    in it, and in overtime, each shrunk toward the whole game's by 20 tries (`TWO_GOOD_PRIOR`).
    September's real tries were made 64–71% of the time before the last 3:00 (239 of them),
    52.9% in it (493) and 36% in overtime (11). The sim had made 56.6% of all of them, so early
    tries, the first of two touchdowns back among them, failed too often and left more games level.
  - **The deciding gamer's lean on top:** a log-odds shift for each option, fitted on their own
    real decisions against what the league chose in the same states. It is shrunk toward none, so a
    gamer with a handful of decisions decides almost as the league does. This is how earlier
    versions handled 4th-down aggression and kickers' touchback rates. A gamer's lean applies in
    every state, even ones where they have no decisions of their own.

  `--no-gamer-decisions` builds without the leans, so every gamer decides as the league does, for
  comparison.

  The build prints each decision's real shares and the gamers who lean furthest each way.

  What follows a decision is real data too:
  - a returned or kept kick-off's field and clock, from real kicks in the same situation;
  - touchbacks at the 20 and no landing zone at the 35, with no clock;
  - field goals made by distance, and the clock they take. A kick with 40 seconds or less left in the
    half takes it as the real ones did, so one that ran the clock out runs it out;
  - punts' net yards;
  - safety free kicks.

  A timeout moves only the time draw.

  Overtime is played as in v14.
- **The game's own end.** In Q4 the side ahead with the ball wins there and then when two things
  hold: its downs left, each taking the 40-second play clock, outlast the clock, and the defense
  has no timeouts. The clock goes to 0; there are no kneel-downs to play out.

### In the calibrator (`--candidate v21`, `v21_stream.py`)

v2.1 prices as v21-report does, and the calibrator pairs it with prod as it pairs any version. From
kick-off the sim plays the match's gamers and teams, each side's offense nudged `--v21-nudge-share`
(0.75) of the way from the sim's own means to the prior's expected points. The prior is a
walkthrough table or a v13/v14 build made with `--history`, given as `--v21-prior-model`. By default
it is the form table `python -m eAMFCalibrator form-prior` writes
(`eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv`: the walkthrough's avg table moved half way
to a rating drifting 4 points² a day), then `./v14_model`. `v21-report` reads the same default. Every
PLAY_OVER the sim can start from is priced from there with the same nudge: a snap, or the kick-off
after a conversion or a field goal.

A touchdown's PLAY_OVER, with its conversion still to come, has no book, so prod's quotes in that
window go unpaired. Only the PLAY_OVERs prod's quotes read are simulated, at `--v21-paths` (1000)
games each. The lines are v14's rules (`--v21-lines own|even|prod|anchored|hyst`).

A betting simulation (`eAMFCalibrator bets`) hands over `moments`: each bet's market and the prod
message it reads. Then only the books those bets read are simulated, and the kick-off as ever, which
is several times quicker. Its README's "Only the moments the bets read" says what that changes, and
`--all-moments` turns it off.

    python -m eAMFCalibrator report --since 2026-10-01 --until 2026-10-14 --candidate v21 \
        --v21-model v21_t10 --v21-prior-model v14_model

Every match's sim is seeded by the match (`MATCH_SEEDED`), so its quotes are the same whichever
matches it is priced with. Two things follow from that:
- **Several chunks at once:** the calibrator prices `MODEL_CHUNK_SIZE` (300) matches of its 50-match
  pairing chunks in one pool. Each chunk's quotes are kept for its turn, in the order its own call
  would have given them. A pre-match quote's earlier states read its own chunk's ends of play, as
  they did. The older versions (v8–v14) draw a worker's matches from one random stream, so they
  still price a chunk at a time.
- **One match at a time:** within a pool, a match goes to whichever worker is free; a fixed share of
  matches per worker left the rest waiting on its longest ones.

### The second strength lever (`--candidate v21-shape0.25`, `v21-report --shape`)

A big gap between gamers makes the favourite win more often without winning by much more. Over Dec
10 – Sep 10 the favourite by 250–300 Elo won 74.4% by 4.8 points on average. A model that prices the
win off the margin, with the league's spread, gives 68.6% at that margin. The margin's spread narrows
as the gap grows (9.85 at under 50 Elo, 9.46 at 250+), and close wins rise from 31% of games to about
40%. Measured by the pre-match points expectancy instead of Elo, the favourite wins 0.3–2.1 points
more often than its real margin implies (Feb–Oct, 22,370 matches). The nudge alone can't do this:
v2.1 turns its margin into a win chance with a fixed spread (chance = Φ(margin / 9.87) on Oct 1–17).

The lever plays the strength gap by the score. Each side's distance tilt is kept three ways: as it is,
with the favourite behind by `SHAPE_TRAIL` (1) or more, and with it `SHAPE_LEAD` (9) or more ahead. A
behind favourite plays the gap κ × |edge| / 2 nudge units a side wider (it raises its game); a
favourite two scores up plays it that much narrower (it eases off). The edge is the kick-off's
expected home margin, and the favourite is the side it favours. The nudge is still solved so the
kick-off means meet the expected points. The result is more comebacks and fewer blowouts: the
favourite wins more often, by the same mean margin.

On the tests' build, a favourite at +5.8 points expected:

| κ | mean margin | margin sd | wins | wins by 1–7 | by 8+ |
|---|---|---|---|---|---|
| 0 (off) | +5.84 | 8.74 | 76.3% | 37.6% | 38.7% |
| 0.5 | +5.80 | 7.14 | 80.7% | 45.6% | 35.0% |
| 1 | +6.04 | 5.88 | 86.0% | 52.0% | 34.0% |

The real games show about +2 points of win chance with the spread barely moving, so κ is small,
fitted on the real build: `v21-report --nudge --nudge-share 0.75 --shape K` (the final-margin
section shows how the favourite wins, the sim's against the real), then the betting sim's
`--candidate v21,v21-shape0.1,v21-shape0.25` on the same bets. Off (κ 0, any name without
`-shapeK`), the draws are exactly as before. The three tilts a side share the sim's tilt cache (six a
match, under `TILT_CACHE_MAX`), so the lever costs little time.

**Result: it doesn't beat v21.** The betting sim ran Oct 1–18 on 129,849 bets over 1,230 matches.
Against plain v21:
- On one book it's level at every κ (κ 0.5: +0.20 [−0.18, +0.59]).
- On two books, with each bet going to the better of prod and the candidate, κ 0.5 is worse:
  −0.40 [−0.68, −0.15], and on the moneyline −0.45 [−0.75, −0.17].
- In practice it raised the favourite's chance by 0.7–1.1 points pre-match, level and behind, and
  barely moved it ahead (the ease-off at 9+ seldom fires).
- The longer underdog prices that follow got taken, and the underdogs won too often for them:
  pre-match −2.12, favourite behind 3–8 −1.46, level −1.17 (two books, κ 0.5, each interval below
  zero).
- The kick-off check at κ 0.25 agrees: favourites by 2–4 lost 41.6% of real games, against 38.5%
  expected.

So the favourite isn't stronger than v21 already says. The lever stays off.

### The gap shrink (`--candidate v21-gap0.75`, `v21-report --gap`)

The prior's big gaps come out smaller in the real finals. Taking the form table's expected margin
(`avg_form4_w05.csv`) against the real margin, burn-in left out:

| Months | Matches | Real margin per point of the prior's | Prior 3+ points: prior, real |
|---|---|---|---|
| September | 1,611 | 0.45 (se 0.10) | +4.15, +2.11 |
| October 1–18 | 1,220 | 0.77 (se 0.11) | +4.48, +1.85 |

v2.1's sim already turns a point of the prior's margin into about 0.86 of a point (the kick-off
check). That leaves about 0.9 (October) to 0.55 (September) still to take off.

The lever takes the prior's expected margin L times over and keeps its total, before the nudge
meets it (`v21_stream.shrunk`). That holds at kick-off and in every earlier pre-match state. The
in-play states carry the nudge as ever.
- A candidate named `v21-gapL` asks for it. `v21-gap1`, or no `-gapL`, is the prior as it is.
- `v21-report --nudge --gap L` runs the same kick-off check. The report still sets the sim against
  the inputs as given, so its "by the expected home margin" table shows what the shrink does
  against the real finals.

L = 0.9, 0.75 and 0.6 bracket the two months. They were chosen off September's fit and the sim's
own slope, before looking at the October bets:

    python -m eAMFCalibrator bets run --since "2026-10-01 00:00:00" --until "2026-10-19 00:00:00" \
        --candidate v21,v21-gap0.9,v21-gap0.75,v21-gap0.6 --v21-model v21_t10 --v21-paths 2000

### Playstyle by the score (`playstyle`)

Does a gamer play differently ahead from behind, in a way that repeats? The second strength lever
guessed one answer for every gamer. This report measures it on real games instead, gamer by gamer,
and tests whether it's a trait before anything is built on it.

    python -m eAMFModel playstyle --snapshots eAMFCalibrator/out/scouting_playover.csv \
        --history eAMFCalibrator/out/match_history.csv --out eAMFCalibrator/out/playstyle

It reads the export alone: no Snowflake and no simulation, minutes on a full export.

- **The states.** The opening kick-off, and the kick-off after every score in regulation (a
  conversion's, a made field goal's or a safety's PLAY_OVER). There the score and the time left
  are the whole state: the receiver's ball at its 25.
- **What's measured.** Each side's points still to come, against its expected points times the
  league's share of a game's points still to come at that time left. The expected points come from
  prod's pre-match spread and total in the export, or `--prior`'s table.
- **The league.**
  - The leftover is fitted on possession (the side that conceded receives, so a trailing side has
    the ball more) and on the lead × the share of the game left.
  - The report shows the receiver's margin still to come by its lead and half.
  - v2.1's sim plays real snaps for the situation, so this part is in it already. `v21-react-fit`
    found its game script right: −0.028 [−0.070, +0.013].
- **Each gamer, beyond the league.** The margin still to come (own points less the other side's) is
  fitted on the share left and on the lead × the share left.
  - The first term is the gamer's own over- or under-performance of its price, kept apart so a
    gamer prod prices short doesn't pass for a front-runner.
  - The second is the slope. Above 0 is a front-runner: leads grow, and deficits too. Below 0, leads
    shrink and deficits close.
  - The slope is also shown for own points and the other side's apart, and with ahead and behind
    apart.
  - A slope of +0.10 means a 7-point lead at half-time grows 0.35 points more than the league's
    over the rest of the game.
- **Is it a trait?**
  - Every gamer's matches are split by alternate days, the slope fitted on each half, and the two
    halves correlated across gamers. A playstyle shows in both halves; noise doesn't.
  - Only when that correlation's 95% interval is above 0 are the gamers grouped: front-runner,
    closes up, or steady. A group needs the shrunk slope 80% sure away from the gamers' mean.
  - tau is the spread of the true slopes beyond noise (DerSimonian–Laird on the bootstrap errors).
    Each slope is shrunk toward the others by it.
- **Pace.** Every snap of the second half, except its last two minutes and those a quarter's end cut
  short, gives the seconds it took off the clock, ahead against behind. The league's pace is shown,
  each gamer's beyond the league's, and the same split-half test, with groups "burns the clock
  ahead" and "keeps the tempo ahead".
- **What it gives.** Intervals are 95% over matches. `playstyle.txt` has the report and
  `playstyle_gamers.csv` every number, a row a gamer.
- **The checks.** On planted data a ±0.3 slope across 15 gamers is found (the halves correlate
  above 0.9) and grouped right. With nothing planted it says "not a trait".

### The spread against the moneyline (`ats`)

Punters have an edge on prod's moneyline and none on its handicap. This report sets both of prod's
prices against the real finals.

    python -m eAMFModel ats --snapshots eAMFCalibrator/out/scouting_playover.csv \
        --history eAMFCalibrator/out/match_history.csv --out eAMFCalibrator/out/ats

It reads the export alone: no Snowflake, no simulation.

- **Pre-match.** Each match's prod quotes before the first play: `prematch_line_52` (the home margin
  the spread is set at), `prematch_prob_52` and `prematch_prob_50`.
  - **The line.** How often prod's favourite covers its line, which is 50% when the line is the
    margin's median. Also the real margin less the line, its mean against its median.
  - **The moneyline.** How often the favourite wins, against prod's moneyline.
  - **The two against each other.** A favourite wins whenever it covers, and also when it wins by less
    than the line: a 1.5-point favourite by exactly 1. So prod's moneyline less its cover chance
    should be how often that happens. If it's less, the moneyline is too flat for the line it sits
    on.
- **In play.** The favourite at every PLAY_OVER of regulation with both prices live, against prod's
  moneyline and its line, by price and half.
- **Gamer by gamer.**
  - From the gamer's side: its margin less prod's line (against the spread), how often it covered,
    and its wins less prod's moneyline chance.
  - Each is tested for a trait as playstyle does: alternate days apart, the halves correlated across
    gamers, and grouped (beats prod, short of prod, as priced) only when that holds.
- **What it gives.** Intervals are 95% over matches. `ats.txt` has the report and `ats_gamers.csv` a
  row a gamer.

**A first look off the bets file (Oct 1–18, 1,013 matches with both of prod's pre-match prices).**
- Prod's favourite covered exactly 50.0% of the time, and the real margin less the line had mean
  −0.66 and median −0.50. So the line sits on the margin's middle, and mean against median isn't
  the story.
- The favourite won 56.4% against prod's moneyline of 54.9%. Almost all of the gap is in 1–2.5 point
  favourites: prod 51.6%, real 54.2%.
- With a 1.5-point line the favourite also wins by exactly 1, but prod's moneyline leaves it little
  room above the cover chance. It's as if the margin were spread over about 22 points, where the real
  misses have an sd of about 10.
- This fits the bets: backing prod's pre-match favourites, the book lost 3.4%; backing its underdogs,
  it made 5.4%.
- v2.1's kick-off moneyline comes off its own simulated scores, wins by 1 included: its favourite
  won 57.1% against the 57.2% it said.

### Pace when ahead (`v21-pace-fit`, `--candidate v21-pace`)

The playstyle report found one trait: how much a gamer slows the clock when ahead. Beyond the league,
this repeats from one half of a gamer's matches to the other: r = +0.58 [+0.40, +0.73], Aug 23 –
Oct 17, 65 gamers. OFFICER takes about 5 s a snap longer than the league when ahead. STRIDER is about
6 s quicker, and in fact quicker ahead than behind.

**What the build has.** A tilt build's time draws are the league's real snaps for the state, the
lead among it, so the league's slowing down when ahead is in them. They're reweighted by one time
tilt a gamer (and team), the same whether the gamer is ahead or not.

**What this adds.** A second time tilt a gamer, used on its side's snaps while the side is
`PACE_LEAD` (1) or more points ahead.
- It's fitted as the build fits its own: Newton steps on the penalized log-likelihood of the gamer's
  real snaps ahead, against the league's draws for those states with the build's tilts on them.
- A league part is fitted with it, for whatever the build's draws miss of every side's clock ahead.
  Each gamer's own part sits beyond that, shrunk by the build's `TILT_PRIOR`.
- A gamer the table hasn't met takes the league part alone.

```
python -m eAMFModel v21-pace-fit --model v21_t10 --until 2026-10-01
```

**The fit command.**
- It fits on the matches before `--until`.
- It scores on the matches after (to `--test-until`): each ahead snap's log-likelihood with the
  tilts against without, with a 95% interval over matches. It also checks whether the seconds a
  gamer's tilt says it gains ahead are the ones it gained on those later matches (r across gamers,
  and the slope: 1 right).
- It writes `pace_ahead.json` beside the build and reads the export alone: no Snowflake, no
  simulation.

**Pricing with it.**
- The calibrator's candidate `v21-pace` prices the same build with the table: `--candidate
  v21,v21-pace --v21-model v21_t10`. It stops before any fetching if the table is missing.
- In the sim, each side's time tilt is kept two ways, as it is and with its gamer's extra. Each
  snap's time is drawn on its score's: the second lever's pattern, with the tilt cache covering both.
- Without the table (`v21` itself), the draws are exactly as before.
- On the tests' build, +3 on the home side's tilt takes its snaps ahead from 20.7 s to 23.0 s, and
  leaves its snaps behind and the other side's as they were.

**The betting sim (Oct 1–18, 144,043 bets).** No gain. On one book v21-pace was −0.12 [−0.44,
+0.18] against v21 (moneyline +0.08 [−0.19, +0.34], Q4 moneyline −0.16 [−0.53, +0.20]), and on two
books its head-to-head was −1.90 [−6.41, +2.37] against v21's. It stays off.

### The endgame (`v21-endgame`)

```
python -m eAMFModel v21-endgame --model v21_t10 --since 2026-10-01 --until 2026-10-18 --nudge --nudge-share 0.75 --out eAMFModel/out/endgame
```

The real gamers seem to do what they can to keep a game out of overtime. The sim sends about twice as
many there: v21-report's census has 0.04 a game against the real 0.02 (Oct 1–17). Its final margins
are off around the key numbers too: by 1, 9.1% against 7.4% real; by 2, 3.7% against 5.1%; by 6, 6.6%
against 4.7%; and by 7, 12.2% against 10.7%. This report looks at where that comes from, three ways.
It reads the export alone, with no Snowflake. Nothing in it feeds back into the model.

- **From kick-off.** The sim's games from kick-off, with the match's gamers and teams, nudged to its
  expected points as the stream prices (`--nudge --nudge-share 0.75`), against the real games:
  - overtime a game;
  - games level at the end of regulation, per 100 games, by the score that made them level, and
    whether it came in Q4's last 3:00. The scores are the kick after a six from 7 behind, two after a
    six from 8 behind, a field goal from 3 behind, a six from 6 behind, and a safety;
  - the final margin's lumps (by 1, 2, 3, 6, 7 and 8 either way), and how many of them came in
    overtime.
- **From the same late states.** Every real snap of Q4 with the sides 8 or fewer apart is played out
  from the real state with the kick-off's nudge (`--state-paths`, 200). The report sets overtime and the
  side with the ball winning against what really followed, by the time left and that side's lead,
  each match counting the same. Each time band also gives overtime sim less real, with a 95% interval
  over matches. The sim starts from the same states the real game reached, so what differs here is
  the endgame itself, not how the game got there.
- **The fork.** Where a side picks between a tie and a win, the real choices against the sim's (from
  kick-off and from the late states):
  - the conversion 1 behind after the six (the kick ties) and 2 behind (two ties), in Q4's last 3:00
    and in overtime;
  - 4th down, and a kick on downs 1–3 in the last minute, 3 behind (the field goal ties) and 1–2 behind
    or level (it wins), in Q4's last 2:00.

To do it, the sim's census (`stats`) now also keeps each path's last score of regulation
(`last_score`: its kind, the clock, the scorer's lead before it and its points) and, when asked
(`_dec_fine`), each decision's state. Neither touches the random numbers: the draws are the same with
the census as without it.

It writes `v21_endgame.txt`, and `v21_endgame_states.csv` with a row for each late state.
(`v21-audit` has an overtime-from-Q4-states table too, without the nudge. This report adds the nudge,
the score that made a game level and the fork.)

**Oct 1–18 (`v21_t10`, the form prior, nudged 75%; 1,154 matches):**
- **Overtime:** real 1.6% of games (19), the sim 3.7%. The extra level games come mostly from the
  kick after a six from 7 behind (0.35 per 100 games real, 1.53 sim) and a field goal from 3 behind
  (1.13 against 1.84).
- **From the same late states the sim goes to overtime more:** +2.5 to +3.0 points in every Q4 time
  band, and +7.2 [+4.3, +9.6] from level.
- **The bigger finding is who wins.** A side 1–3 behind with the ball in Q4 really wins 51.0% of the
  time, but the sim gives it 9.6 points less [−14.1, −5.0]. A side 1–3 ahead with the ball gets
  +4.7 [+1.0, +8.8] too many, and one 4–8 behind +5.1 [+1.9, +8.4]. It isn't the nudge's strength:
  the 1–3-behind gap holds when the side with the ball is the prior's favourite (−11.8), its
  underdog (−6.1) and near even (−8.4).
- **The fork:**
  - On 4th down in Q4's last 2:00, 1–2 behind or level, real sides kick 60.9% (n 115). The sim
    kicks 45–47%, goes for it more and punts more (10% against 4%).
  - 3 behind, real sides go for it 83% (n 47) against the sim's 75%. On downs 1–3 in the last
    minute they kick to tie 3.7% (n 270) against the sim's 8%.
  - Conversions 1 behind after a late six match from the late states (two 70.3% sim, 69.6% real).
- **The bets agree** (the Oct 1–18 betting sim, Q4 moneyline, which closes with about 1:53 left).
  The book lost about 30% on bets backing a side 1–3 behind:

  | Backed side | prod | v2.1 | won |
  |---|---|---|---|
  | 3 behind (1,630 bets) | 0.377 | 0.393 | 0.569 |
  | 1–2 behind (591 bets) | 0.447 | 0.458 | 0.662 |

  Backing a side 3 ahead, v2.1 said 0.649 and it won 0.510. The punters pick their moments, so the
  bets' gap is about twice the late states' one.

**September (2,226 matches) says the 1–3-behind gap is new.** The side with the ball 1–3 behind won
39.5% in September, and the sim had it right (−0.2 [−3.5, +3.1]). Week by week the real share went 41,
34, 39, 40, 48, 50 and 57%, from Aug 31 to Oct 12, while the sim stayed at 38–44%. The 1–3 ahead side
went the other way: 89–94% a week in September, 81–85% in October. Level sides did too: the sim was
6–9 points short of them in September and about right in October. The overtime excess held in both
months (+2.3 to +2.5 points from the late states). The late game changed from the last week of
September, so a correction fitted on September alone would miss it.

### The late Q4 correction (`v21-late-fit`, `--candidate v21-late`)

```
python -m eAMFModel v21-late-fit --model v21_t10 --states eAMFCalibrator/out_v21_endgame_sep/v21_endgame_states.csv eAMFCalibrator/out_v21_endgame/v21_endgame_states.csv
```

A rolling correction for the drift above, while the sim itself plays the endgame as it always has.
- **The cells.** The lead of the side with the ball (4–8 behind, 3 behind, 1–2 behind, level, 1–3
  ahead, 4–8 ahead) times Q4's time left (4:00–2:01, 2:00–1:01, 1:00–0:00).
- **The fit.** For each match day, a shift on the log-odds of the side with the ball winning in each
  cell. It's fitted on v21-endgame's late states of the 14 days before (`--window`): the sim's chance
  against what really followed, each match counting the same. It's shrunk toward none by 30 matches
  (`--shrink`). A day's own games never fit its own shift.
- **The days.** By default the fit covers every day from the states' first plus the window, to the
  day after their last; `--since` and `--until` set them. It writes `late_q4.json` beside the build.
- **Pricing.** The calibrator's candidate `v21-late` prices off the same build with the table:
  `--candidate v21,v21-late --v21-model v21_t10`. It stops before any fetching if the table is
  missing.
  - Each in-play book of Q4 with the sides 8 or fewer apart has its margin's mass either side of a
    level game scaled, so the side with the ball wins at its shifted chance. The spreads move with it.
  - The kick-off, the earlier quarters, overtime and the totals are as v21 prices them.
  - A match whose day isn't in the table is priced as v21; the run says how many.

On October's own late states, the rolling shifts take the sim's chance for the side with the ball:

| Lead | Real | Sim | Corrected |
|---|---|---|---|
| 1–2 behind | 54.2% | 43.4% | 48.7% |
| 3 behind | 49.6% | 40.3% | 46.3% |
| 4–8 behind | 20.5% | 25.6% | 21.9% |
| Level | 73.4% | 71.1% | 77.2% |
| 1–3 ahead | 84.0% | 88.7% | 87.5% |

Level overshoots: September's level gap had gone by October, and the correction lags drift in both
directions. The betting sim is the judge.

**The betting sim (Oct 1–18, 144,051 bets) gives no gain overall.**
- One book, moneyline: +0.02 [−0.39, +0.38] against v21.
- Two books, moneyline: v21-late's head-to-head is −0.51 [−4.94, +3.52] against v21's.
- Margin uplift, moneyline: +2.99 against v21's +3.11.

It worked where it was aimed and gave it back elsewhere. Q4 moneyline bets, by the backed side's lead,
the book's margin re-priced at each candidate:

| Backed side | Stake | v21's chance | v21-late's | Won | v21 margin | v21-late margin | Late − v21 [95%] |
|---|---|---|---|---|---|---|---|
| 3 behind | 88,752 | 0.393 | 0.446 | 0.569 | −24.6% | −9.8% | +14.8 [+11.4, +18.1] |
| 1–2 behind | 60,300 | 0.458 | 0.506 | 0.662 | −24.2% | −14.5% | +9.7 [+6.6, +13.7] |
| 4–8 behind | 89,191 | 0.272 | 0.229 | 0.239 | +5.6% | −15.3% | −20.9 [−32.0, −10.9] |
| 1–2 ahead | 19,116 | 0.594 | 0.544 | 0.525 | +6.8% | −3.7% | −10.6 [−20.4, −3.3] |
| 3 ahead | 62,928 | 0.649 | 0.603 | 0.510 | +2.2% | −3.1% | −5.2 [−8.7, −2.9] |
| All Q4 | 567,679 | 0.450 | 0.453 | 0.496 | −3.75% | −3.92% | −0.17 [−2.68, +2.21] |

The 4–8-behind and leaders' moves were right too: closer to how often those sides won. But on the
same bets, a price made more accurate on a side the punters back without an edge lengthens the odds on
bets that were losing at v21's. The book was earning on that lean. In the bets, accuracy pays only
where the money is informed, and late in Q4 that's the side 1–3 behind. v21-late stays off.

### The pace nudge (`--candidate v21-drives`, `v21-report --drives`)

v21's kick-off margin spread hardly moves with a match's pace. In the slowest fifth of expected drives
it's 9.81 against a real 8.77; in the fastest, 10.09 against 10.52 (`xgb/README.md`, 1 Sep – 18 Oct).
Its gamers' time tilts are long-run profiles. The pre-match pace model (`xgb/pace_eff.py`) foresees
each match's drives, and it's calibrated: real drives run 9.4 → 11.3 across its fifths.

**What it adds.** One tempo a match, added to both sides' time tilts (`PHI_TIME`: seconds / 10; below
0, quicker snaps and more drives). It's solved with the points nudge (`v21.drives_solve`), so the
kick-off games' drives meet the match's goal and the points still meet theirs.
- **The goal.** The match's expected drives over their table's mean, times the sim's own drives a
  match at kick-off (`v21.drive_level`). The nudge moves pace from match to match, not the league's
  level, so the table's drive count and the sim's needn't agree on what a drive is to the last snap.
- **The steps.** Secant steps on the same random numbers, from the build's slope: the drives a unit
  of tempo adds, measured at ±`TEMPO_PROBE`. A new build keeps it; an older one works it out once
  per process. The points nudge steps on the points the sim would score at the drives the tempo is
  moving to, so the two levers don't chase each other. With the pace nudge on, a secant step needs
  a real move (`SECANT_MIN`) and a slope within 0.3–3× the build's, else it takes the build's slope.
- **The points stay put.** With `--nudge-share` below 1, the points target is taken from the sim's
  own means as they are, not as they'd be at the new pace, so the pace nudge moves no mean: pace
  goes in the spread. At a share of 1 the target is the prior's points either way.
- **The limit.** The tempo stays within ±`TEMPO_MAX` (3). The report says how many matches hit it.
- **In play.** The kick-off's tempo is carried into every state, as the nudge is.
- **A drive.** A side's possession from its first snap. The sim counts one at each possession's
  first snap. v21-report counts the real ones the same way, off the snap records: a new drive at a
  fresh possession, a change of side or a new half.
- Without it (`v21` itself), the draws are exactly as before.

**v21-report.** `--drives TABLE` (`pace_eff_prior.csv`) adds each match's drives to the report:
- the sim's and the real drives a match;
- with `--nudge`, the pace nudge itself (`--drives-share S` goes that share of the way from the sim's
  own drives);
- by fifth of the expected drives: the expected, the sim's own, the goal, the sim's, the real, the
  sim's margin sd and the real margins' sd about the sim's means.

`v21_matches.csv` gains `exp_drives`, `sim_own_drives`, `goal_drives`, `sim_drives`, `real_drives` and
`tempo`.

```
python -m eAMFModel v21-report --model v21_t10 --since 2026-09-01 --until 2026-10-18 --nudge --nudge-share 0.75 \
    --drives xgb/out/pace_eff/pace_eff_prior.csv --state-paths 0 --out eAMFCalibrator/out_v21_drives
python -m xgb.v21_pace --v21 eAMFCalibrator/out_v21_drives/v21_matches.csv --pace xgb/out/pace_eff/pace_eff_prior.csv
```

**Pricing.** The calibrator's candidate `v21-drives` (or `v21-drivesS`, S of the way) prices off the
same build with the pace table: `--candidate v21,v21-drives --v21-drives <pace_eff_prior.csv>` (default
`$EAMF_V21_DRIVES`, then `xgb/out/pace_eff/pace_eff_prior.csv`). It stops before any fetching if the
table is missing. A match the table lacks is priced as v21, and the run says how many.

On the tests' build (its snap times spread out), a tempo of −1 plays 11.6 drives a match, 0 plays
8.0 and +1 plays 6.6. The solve meets a goal a drive either side within 0.15, with the points within
half a point.

### In-play strength update (`v21_react.py`, `--candidate v21-snap4`)

Without one, every in-play state carries the kick-off's nudge. With one, each state gets its own:
the kick-off's, moved by what the match has shown of each side by then. The nudge is in points a
game for the side's offense, so a side outplaying its price is priced up for the rest of the game.
The calibrator names a variant as a candidate, priced off the same `--v21-model` build:

| Name | What moves the nudge | Its setting |
|---|---|---|
| `v21-snapT` | every real snap scored against the draw the sim makes in its place (the league's snaps for the situation, its blended levels and field window, tilted by the side's strength): the snap's features less the draw's mean along the strength direction, over the draw's variance. One Newton step from the kick-off, shrunk by a prior of 1/T² (v14's react, in v2.1's terms) | T: the prior sd, points a game |
| `v21-ptsT` | points so far against the kick-off mean × the share of a game's points that has come (`points_share`); the kick-off's simulated spread of points is the noise | T: the prior sd, points a game |
| `v21-wpaC` | every snap looked up in a win-probability table: the side's chance after it less before it. The nudge is C × the chance its snaps added. A snap counts for how far it moved the match, not how far it was from its situation's usual | C: points a game for a whole win chance |

**Gamer by gamer** (`v21-gsnapT`, `v21-gptsT`, `v21-gwpaC`): the approach's nudge × the gamer's own
scale. That scale is the slope of the gamer's rest-of-game points beyond the sim's, against what the
approach said, shrunk toward the pooled slope by an average gamer's evidence. `v21-react-fit` writes
it to `react_gamers.json` beside the table. A gamer it has none for takes the pooled slope.

The snap update moves most early: a snap's score is divided by the information so far plus 1/T², so
one 20-yard gain on the first snap moves a side further than the same gain an hour in. `wpaC` reads
its table from `react_wp.json` beside the build, written by `v21-react-fit`; without it the run stops
before pricing. No variant (`v21` itself) leaves every row as it was: the stream's quotes match the
previous code row for row.

    python -m eAMFCalibrator bets run --since "2026-10-01 00:00:00" --until "2026-10-13 00:00:00" \
        --candidate v21,v21-snap2,v21-snap4,v21-pts4,v21-wpa10 --v21-model v21_t10 --v21-prior-model v14_model

Each candidate is a pricing of its own, so each one adds a full v2.1 run's time.

**Choosing the settings (`v21-react-fit`).** On the matches before the window, by day:
- The first 60% fit the win-probability table `wpaC` reads. It's a logistic fit of who won on each
  PLAY_OVER's state: the margin, the field and the down over the root of the time left, and prod's
  pre-match price times that root.
- The last 40% score every update at a spread of settings, through the build:
  - **Baseline:** at each kick-off after a score, the sim plays the rest of the game with the kick-off
    nudge (toward prod's pre-match spread and total, at the stream's share).
  - **Score:** each side's points still to come beyond the sim's expectation, against the update's
    nudge × the share of the game left. This is the squared-error change, with a 95% interval over
    matches.
  - **Slope:** of what came on what the update said. 1 means right, under 1 means it moves too far.

It scores three more things on the same matches:
- **Earlier matches (`c-`):** each update with the gamer's earlier matches of the session carried in.
  Each one back is worth ×0.7, and it is scored from the opening kick-off. This is fit-only for now:
  it gets priced if it scores.
- **Gamer by gamer:** each approach's gamer scales fitted on the first 60% of the scored matches,
  then set against the pooled slope on the last 40%.
- **The sim's own game script by the lead:** the slope of points still to come beyond the
sim's on the side's lead × the share left. Below zero, the sim gives the side ahead too much of the
rest of the game. That is a sim fix of its own, and any update that follows the lead (`pts`, `wpa`)
looks the worse for it.

A table fitted on real games has already learned that the side ahead is likelier the better one, so
it isn't the test: v2.1's sim prices a state off the prior's strength alone, and an update is worth
what it adds to that.

    python -m eAMFModel v21-react-fit --model v21_t10 --until 2026-10-01

It reads the export alone (no Snowflake), at `--paths` (500) games a kick-off.

### How far the draws set each gamer apart (`v21-gamers`)

Every gamer in the build plays an average opponent from kick-off, home and away, with no nudge. It
prints the sim's points for and against, the margin and the win chance. Beside them are the gamer's
real points a match and the gamer's offense and defense tilts read along the strength direction.
The summary gives the spread of the sim's margins across gamers, and how much of a point of real
scoring the sim follows. Pre-match the kick-off nudge then moves each side most of the way to the
prior's expected points, so the rest of a gamer's pre-match strength is the prior's.

    python -m eAMFModel v21-gamers --model v21_t10 --gamers KILLJOY,MERLIN,HERMES --out v21_gamers.csv

### What the money backing a gamer sees (`gamer-dig`)

Off the export, the history and a betting simulation's `bets_sim.csv`, each side of every match gets
a play profile: yards a snap, explosive gains (15+), stuffed snaps, first downs, 3rd downs made,
turnovers, touchdowns and seconds a snap. Each gamer has a baseline over the 30 days before the window.
- **Targeted gamers:** the gamers the money wins on (their backers' edge over prod's price), or those
  `--gamers` names. It gives their record against prod's pre-match price, their baseline against the
  league's, and their play in the window against their own baseline.
- **Earlier matches:** whether the gamer's play in the last three matches of the session, against
  their baseline, tells their result beyond prod's pre-match price (every match the export prices).
- **This match:** whether the backed gamer's or the opponent's play so far tells the bet's result
  beyond prod's price at the bet, with match-clustered intervals.
- **The snap before:** the snap each bet followed (touchdown, turnover, explosive, first down,
  stuffed), how often, and the edge after it.

Each section is repeated for bets backing a gamer on 3+ wins this session.

    python -m eAMFModel gamer-dig --bets eAMFCalibrator/out_bets_react/bets_sim.csv --out gamer_dig

### Timing (`v21-timing`)

```powershell
python -m eAMFModel v21-timing --model v21_t10 --since 2026-10-01 --until 2026-10-14
```

How long the calibrator's v21 stream takes to price the window, measured off the local snapshot
export with no Snowflake connection. It takes a seeded sample of `--matches` (6) matches and prices
each as one worker does: the kick-off with its nudge solved (four runs of both sides receiving),
then every in-play PLAY_OVER state at `--paths` (1000) games in one run. For each match it gives
the seconds per part, the games and snaps simulated, and the microseconds per snap. It then
projects the window's run over the workers. One live price update comes next: a single state
simulated alone at points through the game, as a live pricer would run it with the model loaded
and the nudge solved before kick-off. A profile of the first match's in-play run follows:
simulate's parts and the functions it spends its own time in. The projection covers simulation
only; reading Snowflake and the prior's predictions come on top.

`--compare` also runs the first match with v2.1's two exact speedups off and on, in turn three
times, keeping the best of each so other work on the machine weighs on both alike. It times
the kick-off, the in-play run and a live update, and sets every output against the run with them
off, bit for bit.

### The exact speedups (`TILT_CACHE`, `DECISION_TABLE`)

Two switches in `v21.py`, both on. Each gives the same draws, bit for bit, in less time; with
either off, the sim runs the code it ran before.
- **`TILT_CACHE`:** a run's tilts are one a side for a match. Each real record's theta . phi is
  worked out once per run, with the same einsum the draws ran on their candidates
  (`Bins.tilt_z`). The tilted draws then read their 16 candidates' values off that table instead of
  working each out again. A run of fewer than `TILT_CACHE_PATHS` (5000) games, such as the
  kick-off's nudge solve, works them out as before, unless an earlier run of the same tilts left
  its table.
- **`DECISION_TABLE`:** a decision's league shares in a cell of its features are drawn up level by
  level the first time the cell comes up, and read off a table after (`Choice.base`). Most of
  this saving is the defense's and the offense's timeout choices before every running-clock snap.
  The rules' strike-outs and the gamers' leans still run on every decision's full state.

Neither table is saved with the build.

### Coverage (`v21-coverage`)

```powershell
python -m eAMFModel v21-coverage --model v21_dec --since 2026-10-01 --until 2026-10-12 --out eAMFCalibrator/out_v21_coverage
```

For every real state in the window, how much real data the sim's draws there stand on. It covers
the time draw at the state and the distance draw at the state with the time its real play took.
For each draw it gives the fallback level reached, the real snaps in that group and, in full
mode, whether the gamer's and team's own draws had enough alike. It does the same for every real
4th down, early kick and conversion and its cell. Each draw is one of:
- **full:** the whole key;
- **situation:** the lead and the time of the game kept, but coarser;
- **no lead:** the time of the game kept, not the lead;
- **bare:** neither.

`coverage.txt` sums them by the part of the game and the lead, and by the snaps left and the lead,
with the median group size. It also lists the thinnest situations (by half, lead, down and snaps
left), where the most states' distance draws lose the lead or the time of the game, for the end of
a half and the rest of the game apart. `coverage_states.csv` and `coverage_decisions.csv` have one
row per real state and decision, to look any situation up.

### Down by down, by the time left (`v21-audit`)

```powershell
python -m eAMFModel v21-audit --model v21_dec --since 2026-10-01 --until 2026-10-12 --out eAMFCalibrator/out_v21_audit
```

The audit sets the sim against the window's real games, down by down. The parts of the game are
Q1, Q3, overtime, and Q2 and Q4 each cut at 2:00, 1:00 and 0:30. It compares three ways:
- **Game flow.** The sim's own games from kick-off (`--paths` per receiving side, each match with
  its gamers and teams) against the real games. For each down and part of the game: snaps a game,
  the share ending in a first down or touchdown, turnovers, yards and clock used. Where the sim
  reaches a down more often than the real games, the drives part company there, compounding
  included.
- **One snap.** The next snap alone, drawn `--state-paths` times from every real state before a snap,
  against what really came of it. This shows the draws with the game flow taken out. For each down,
  by part of the game and by distance, it gives the share of each outcome: turnover, loss, no gain,
  short of the line, first down, touchdown. It also gives the yards, the clock, and the 4th-down and
  early-kick choices from the same states. A further table splits each down by the snaps left in
  the half, as the draws read them, and another splits Q4's last 2:00 by the lead. That one shows
  the real states the game's own end covers on their own row, since the sim ends the game there
  with no snap. The 4th-down and early-kick choices in Q4's last 2:00 are split by what a field
  goal does for the side with the ball, and conversions in Q4's last 3:00 by the lead after the
  six (−1: a kick ties, two wins), the sim's own games against the real ones. Each choice also
  shows the league's share in the sim's own cells, before the gamers' leans and the rules: where
  the sim parts from the real choice, that says whether the cells or the leans move it. Q4's last
  2:00 is split by the time each play took, too, and the sim's distance draws by where they came
  from (the league's, a team's own, a gamer's own). Conversions before Q4's last 3:00 are split
  the same way, by the part of the game (8 behind after the six: 14 behind, the first of two
  touchdowns back), with the real two-point tries made by part against the sim's. 2nd–4th
  downs are split by the time each play took, by the distance (1, 2–3, 4+ to go) and in the halves'
  last 2:00: the share of the snaps in each bucket and the share that moved the chains, real and
  sim, and the sim's own rates on the real snaps' times. Where that last is near the real, the
  time draw is what's off; near the sim's, the draw given the time. And from every real Q4 state,
  1–3 behind or level, it gives how often the sim's drive ran out of clock and where it was: in
  field-goal range (a kick of 52 yards or less) or not, with 0:05 or less at its last snap's start
  or more, and with a timeout left. From every real state in Q2's last 2:00 it plays the sim to the
  half's end (`simulate(..., until_half=True)`): each side's points still to come in the half and
  how the drive with the ball ended, by the lead and the time left, against the real half (the
  score at the second half's first row, a kick at the buzzer the feed missed included).
- **Q4 states.** From every real Q4 state to the end of the game: how often it went to overtime,
  each side's points still to come in regulation, and, from level, how often the side with the
  ball won, went to overtime or lost. It also gives how the drive with the ball ended: a
  touchdown, a field goal made or missed, a punt, a turnover, on downs, or the clock running
  out. Real states of one game share its result, so the real figures move in steps of whole
  games.

It opens with the largest gaps: the cells where the sim parts most from the real figure, measured in
standard errors of the real one. It ends with overtime: how often a game goes there, from kick-off
and from every real Q4 state, by the lead of the side with the ball and the time left, against
whether the real game did.

### Where options never happen (`v21-sweep`)

```powershell
python -m eAMFModel v21-sweep --out eAMFCalibrator/out_v21_sweep
```

The sweep goes through every real decision in the export, by default all of it (`--since` and
`--until` narrow it). For each decision and each option, a tree splits the decisions on their full
state:
- the period and the seconds left in the half;
- the deciding side's exact lead;
- yards to the goal and yards to go;
- the down;
- both sides' timeouts left;
- the play before.

It splits so as to separate where the option was taken from where it wasn't. A tree searches many
splits and reports the cleanest, so pure chance can look like "never" (0 of 74 happens 10% of the
time for something taken 3% of the time). So the search is honest:
- The matches are split at random in two, and a tree grows on each half.
- On its half, a situation needs 20 or more real decisions with the option taken by 1% of them or
  fewer, and by a tenth or less of its overall rate, every part of the situation alike.
- It is kept only where the other half bears it out: as rare there, and so few that chance at the
  option's overall rate would give that with p ≤ 0.005.
- Both halves together need 30 or more decisions (`--min-region`).

On pure-noise data this finds no situations, and it finds a planted late-game never-situation
(Q4, the last 0:45, 4+ behind, in range) most of the time. A small one inside lots of noise can
still be missed.

For example: "Q4, 0:41 or less left in the half, behind by 4 or more, 8-35 yards from the goal:
233 real, never".

Where a few real sides took the option anyway, it lists who: the match, the gamer, the quarter,
the clock and the lead, so each can be judged as a real choice or a mistake. It also checks the
game's own end: what came after every real state it covers. There are five outcomes:
- no more snaps;
- only the same side's kneel-like losses;
- other snaps;
- the other side got the ball;
- points scored.

These are split by whether the two-minute warning was still to come.

It writes `v21_sweep.txt` and `v21_sweep.csv`. `v21-build` learns the same situations from its own
matches. For early kicks it learns them only from the half's last minute, where its sim decides.

```powershell
python -m eAMFModel v21-build --until 2026-10-01 --out v21_full --mode full
python -m eAMFModel v21-build --until 2026-10-01 --out v21_adj --mode adjusted
python -m eAMFModel v21-report --model v21_full --prior-model v14_model --since 2026-10-01 --until 2026-10-12 --out eAMFCalibrator/out_v21_full
```

The build reads the calibrator's snapshots, history and timeouts from `eAMFCalibrator/out` by
default and writes `v21model.pkl`. The report reads the expected points from a v13/v14 build's
pre-match model: its own means by default, or those it prices with under `--inputs priced`. Its
sections:

- **Kick-off.** Each side's expected points against the sim's mean (both sides receiving,
  `--paths` games each) and the real points:
  - overall, with how many of the sim's points a point more expected brings;
  - by band of the side's expected points and of the expected margin;
  - by gamer and by team, for points for and against;
  - graded against the prior's own score model: moneyline log loss and Brier, margin and total RPS,
    and the paired log-loss difference.
- **The final margin.** Is the sim's margin too wide, with too few close games? The sim's and the
  prior's own score model against the real finals (overtime in both the sim and the real finals):
  - how often the margin lands in each band, either way: a tie, 1–3, 4–7, 8–14, 15+;
  - the key margins: 1, 2, 3, 6, 7, 8;
  - each match's own margin sd against the real misses' sd, and how often the real final lands
    inside its central 50% and 80%;
  - where each real final falls in its distribution, by tenth. When the distribution is right, 10%
    land in each tenth. Too wide piles them in the middle; too narrow, in the ends;
  - how the favourite wins (by 1–7, by 8+) or loses, expected against real, by its expected margin.
- **Each step.** From every `--every`-th real state (`--state-paths` games each), each side's points
  still to come:
  - the input's, which is its expected points times the league's share of a game's points still to
    come at that clock;
  - the sim's;
  - the real ones.

  These are shown by part of the game, for the favourite and the underdog, and by the side's lead at
  the time.
- **The draws.** The kick-off games against the real games of the window:
  - events per game and points by quarter;
  - the time draw, as time buckets after each kind of play;
  - the distance draw, as snap classes within each time bucket.
- **The decisions.** Each decision's choices, real against the sim, per game and by slice of the
  state. For example, 4th downs are sliced by how far out and how far to go, and early kicks by the
  time left. For the gamers with enough real decisions, it shows each one's real and simulated share
  of the option that shows their lean (going for it, touchbacks, kicking early, going for two,
  calling timeouts), and how well the two correlate.

`--out` also writes `v21_matches.csv` (a row per match) and `v21_states.csv` (a row per real state).

## Which prior on a few matches (`v14-variants`, `v14-priors`)

`v14-variants v14_model` makes three copies of a v14 build that share its tables, grid, profiles and
ratings and differ only in the prior they price with:
- `v14_model_raw`: the same daily glmer fits without the prior shrink;
- `v14_model_layered`: the same fits and shrink, with the post-game form layer on top;
- `v14_model_nb2`: NB2, fitted at the build's cut-off, refitted every day after it and shrunk by its
  own `fit_shrink` (the glmer shrink's centre), as the glmer prior is.

The build itself is the shrunk glmer. Each copy then runs in the calibrator as its own candidate, so
the same bets are re-priced by every prior; `bets --only-matches` keeps it to a chosen list.
`v14-priors` lays the list out match by match: prod's closing pre-match prices (and the expected
margin and total they imply, off a normal with the fitted sds), each prior's expected points as
priced and bare (before the shrink and layers), v14's kick-off simulation off them (expected points,
P home win, and its prices at prod's spread and total lines), the result, and with the calibrator's
bets_sim.csv what each kept of the match's bets. A summary grades the list on the moneyline (Brier),
the margin and the total, and adds up what each kept.

```powershell
python -m eAMFModel v14-variants v14_model
python -m eAMFCalibrator bets --since 2026-10-01 --until 2026-10-12 --only-matches eAMFCalibrator/match_lists/v14_worst15_oct.txt --candidate v14,v14-raw=v14_model_raw,v14-layered=v14_model_layered,v14-nb2=v14_model_nb2 --v14-model v14_model --out eAMFCalibrator/out_worst15
python -m eAMFModel v14-priors --matches eAMFCalibrator/match_lists/v14_worst15_oct.txt --bets eAMFCalibrator/out_worst15/bets_sim.csv --out eAMFCalibrator/out_worst15
```

## The prior against the sim: how much calibration the sim keeps (`prior-vs-sim`, v13 and v14)

For every settled match in a window, priced as of its kick-off, this grades three prices of it on the
real results:
- **glmer raw**: the prior's own expected points and its own score model. That is Poisson with
  glmer's log-normal noise for each side (`sigma_obs`) and one shared by the match (`sigma_match`),
  as `glmer/predict.R` prices the moneyline. A prior with no sigmas, NB2, gets plain Poisson.
- **glmer + layers**: the same score model off the means the build prices with, after the prior
  shrink and any follow or form layer. Against raw it shows what those do.
- **v13 sim** (or **v14 sim**, for a v14 build): the build's kick-off price off those means, with each
  side's profile and big-play rates. That is what the stream quotes before the first play. Against
  the column before, it shows what the simulation alone gains or loses. v14 holds the prior's total
  (within `HOLD_TOL`) but lets each side's rates move the margin, so its expected margin differs from
  the prior's by design; this grades whether that move helps.

Each is graded on the moneyline (log loss, Brier, a decile calibration table, pushes left out) and on
the margin and the total. For those it reports the bias and RMSE of the expected value, the ranked
probability score over the whole distribution, and how often the result lands inside the central 50%
and 80%.

```powershell
python -m eAMFModel prior-vs-sim --model v13_glmer_1001 --history eAMFCalibrator/out/match_history.csv --since 2026-10-01 --until 2026-10-12 --out eAMFCalibrator/out_prior_vs_sim
```

`--paths` sets the simulated games per kick-off (default 4000), and `--workers` the processes. `--out`
also writes a per-match CSV. Under the table, the moneyline log loss differences are given match by
match with their paired standard error. A difference inside about two of them is within the noise of
the window.

## A side's strength, snap by snap (`strength-check`, v14)

A side's strength (theta) reaches its snaps two ways in `sim14.simulate`:
- **The tilt**, exp(tilt x theta). It leans every draw toward the better end of its bin: fewer
  turnovers and losses, and longer gains, all at once.
- **The boom**, exp(boom x theta). It multiplies the bin's 40+ yard plays and moves the same share
  onto its failed plays (turnovers, losses and the rest short of the line), out of the plays
  between. A side's own big-play rate works the same way.

A build from before the strength refit plays `TILT_SCALE` 0.5 and `BOOM` 3. One with it plays its
own (`Tables.tilt_scale`, `Tables.boom`; see "The strength refit").

Nothing else reads the strength. The play calling, the clock, 4th-down choices, kicking and red-zone
holds are the same for a strong side as for a weak one; they differ only by the states the plays
lead to. The prior grid finds the strengths that give each side its expected points. Each simulated
game also draws the day's strengths around them (`sim14.strength_draw`): one swing both sides share,
and each player's own form.

`strength-check` measures all of this on a build, in three parts:
- **A. From kick-off.**
  - One side's expected points are stepped from 10 to 25, with the other side at 16, league
    profiles and no draw.
  - It shows the side's points, snaps a game and how its snaps split: turnover, loss, no gain, short
    of the line, a first down under 20 yards, 20-39 yards, 40+.
  - Each step is run as built and with each of the tilt, the boom and the conversion alone (the
    other two off), at the same strengths. So it shows which kinds of play each part of the strength
    moves, and how many of the points each carries.
- **B. Every side, real against the sim.** The window's settled matches are priced as the stream
  prices their kick-off: the held strengths, the players' profiles and the day's draw. Every side's
  points and snap split are set against its real ones. They're grouped by the side's expected points
  and by its expected margin, with the sim less real on each class.
  - If strong sides really get their points with fewer failed plays while the sim gets them from big
    plays, it shows here.
  - The margin grouping shows how favourites and outsiders differ.
- **C. The day's draw against fixed strengths.**
  - The same matches are priced again with the draw off. The hold is run without the draw too, so
    the prior's total is still met.
  - It compares, for both: the mean total and margin against the prior's, the margin and total
    spread (against the real results' RMS about the prior), the moneyline log loss and Brier, and
    the margin and total RPS and coverage, all on the results.
  - It ends with the log loss difference and its paired standard error.

```powershell
python -m eAMFModel strength-check --model v14_model --since 2026-10-01 --until 2026-10-12 --out eAMFCalibrator/out_strength
```

- `--no-matches` runs part A alone, in a few minutes.
- `--paths` sets the games per kick-off for B and C (default 2000); every match is priced twice, each
  with its own hold.
- `--out` also writes a per-side CSV with each side's prior, real and simulated points, strengths
  and snap counts.
- Sim snaps are the real plays the draws landed on, with the gain cut at the goal line. The
  touchdown run-on and backed-up rules that can change a play afterwards are not applied, so the
  shares are what each side draws. Kneels and overtime are left out of both.

## Refitting the prior every day (`prior-daily`, v10–v13)

A build fits its pre-match model (NB2 or glmer) once, at its cut-off. Over a test window of weeks
its ratings go stale where a live model's would not: glmer's form features follow the results as
they come in, but its coefficients and player effects do not move, and NB2 does not move at all.
`prior-daily` refits the pre-match model for every day of a window, each fit on every result
settled before that day began, as a nightly refit would live. Every match is priced from its own
day's fit (`rolling_prior.py`); the build's prior shrink is applied on top, as before.

```powershell
python -m eAMFModel prior-daily --prior glmer --history eAMFCalibrator/out/match_history.csv --since 2026-08-23 --until 2026-09-23 --out glmer_daily --attach v10_0823,v11_0823,v12_0823
```

- Each build attached must have been built with the same `--prior`, since its shrink was fitted for
  that model. `--detach` returns builds to their own single fit.
- The days are fitted in `--workers` runs of consecutive days side by side (default 2). Within a run
  each glmer fit warm-starts from the day before's (`glmer/fit.R --start`), as `backtest.R
  --warm-start` does. Days already fitted are kept, so a window can be extended later.
- One directory of fits serves every version attached to it.
- Only the prior is refitted. The simulation's tables, the shrink and the form spread stay as the
  build fitted them.
- `--half-life DAYS` (NB2 only) weights each fit's results by 2^(-days old / DAYS), NBRatingTrial's
  `--half-life-days` (its default is 60). It reaches both of a day's fits (the day's own and its
  level scale's). A directory keeps one setting: rerunning it with another is refused, so one
  directory never mixes two models.
- A match before the first fitted day is priced by the first fit, which has seen that match's
  result. `prior-backtest` and `prior-replay` leave such matches out.

The trader prices a match as of today, so with a rolling prior attached it uses the latest day's
fit. To give it an up-to-date pre-match model without rebuilding, refresh the history and fit
today's day onto the build it runs (the header shows which fit it is using):

```powershell
python -m eAMFCalibrator history
python -m eAMFModel prior-daily --prior nb2 --history eAMFCalibrator/out/match_history.csv --since 2026-10-06 --until 2026-10-06 --out nb2_live --attach v12_model
```

## A walkthrough variant as v21's prior (`table_prior.py`)

`eAMFCalibrator walkthrough` prices every match with the best pre-match variants and writes one
table of expected points per variant (`priors/<variant>.csv`). Wherever v21 takes a prior build, it
also takes one of those tables:

```bash
python -m eAMFCalibrator report --candidate v21 --v21-prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv
python -m eAMFModel v21-report ... --prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv
```

Each match's expected points are read as the table has them. They were priced as of its kick-off,
so the table doesn't follow results itself, and a match it doesn't have gets no nudge.

Each NB2 fit now keeps only its training rows' codes and kick-offs once fitted, and drops
NB2_schedule_predict.py's market tables after each prediction. A day of fits (the day's and its
level scale's) was about 19 MB, and is now about 2 MB: a year of daily fits about 0.6 GB, not 5.

## Following the results since the fit (`follow.py`, v10–v13)

A build's pre-match model (NB2 or glmer) is fitted once. Between fits its ratings stay put while
gamers' real levels move. NB2 doesn't move at all. glmer's form features follow the results, but its
player effects don't. `follow.py` adds a second layer on top:
- Each gamer has an offset: the sum of their errors against the model's own prices over their
  matches settled since the fit, divided by that many matches plus 80 (`PRIOR`).
- That offset is added to their later matches: the margin errors to the expected margin, the total
  errors to the expected total. A gamer 2 points worse than rated over 40 matches is priced
  40 × 2 / 120 = 0.67 points worse.
- A match's offset reads only the gamer's matches that started before it. A pre-match quote priced
  as of an earlier time (`stream.known_states`) reads only what was known then.
- NB2 now follows the results too, so the calibrator fetches the settled matches for an NB2 build as
  it does for glmer.

`prematch_model()` puts it on top of the prior shrink, for both priors (`follow.ON`). A rolling
prior refitted every day (`prior-daily`) is left as it is: the layer gains nothing in the first days
after a fit.

**What it catches: drift, not a bad night.** The errors are glmer's (floored) and NB2's out of
sample, 12 fortnights of 14-day refits on AMFELO.
- Two of a gamer's errors correlate only weakly, and about as much in the same session as across
  sessions the same number of matches apart: +0.037 against +0.019 three matches apart, +0.015
  against +0.010 five apart.
- By time apart the correlation is +0.013 within 2 hours and +0.018 at 4 to 16 days.
- So a gamer's level drifts away from the fitted ratings over days. There is no same-night tilt on
  top for the model to catch. The earlier session-form features (`form_session`) found nothing for
  the same reason.

**What it does.** Out of sample, errors restarting at each fit:

| | moneyline log loss | margin RMSE | total RMSE | fortnights better |
|---|---|---|---|---|
| glmer | 0.6682 → 0.6656 | 9.415 → 9.378 | 12.068 → 12.028 | 9 of 12 |
| NB2 | 0.6692 → 0.6653 | 9.375 → 9.323 | 12.151 → 12.067 | 12 of 12 |

- **The older the fit, the more it gains.** For glmer: nothing in the first 3 days after a fit,
  then −0.0013 log loss on days 4–7 and −0.0046 from day 12 on.
- **With the shrink.** On top of the prior shrink (fortnights 4–11), moneyline log loss goes
  0.6686 → 0.6655 for glmer and 0.6686 → 0.6652 for NB2.
- **Head to head.** On the 9,808 matches both priced, glmer stays ahead of NB2 (0.6636 against
  0.6652; totals RMSE 12.106 against 12.171). Its lead shrinks from 0.0042: much of it came from
  following the results, which the layer now gives NB2.
- **Bad and good nights.** The layer moves the next match toward what really happened:

  | session so far | matches | real win rate | glmer | glmer + layer |
  |---|---|---|---|---|
  | lost 3+ of 4+ | 2,596 | 0.421 | 0.439 | 0.429 |
  | lost 6+ of 8+ | 581 | 0.384 | 0.415 | 0.404 |
  | lost 8+ of 10+ | 56 | 0.321 | 0.383 | 0.367 |
  | won 3+ of 4+ | 2,824 | 0.588 | 0.579 | 0.588 |
  | won 6+ of 8+ | 688 | 0.610 | 0.593 | 0.606 |

- **The prior.** 40 chased the noise in some fortnights; 160 was steadier but gained less.

## Pricing only what the model is sure of (v8–v13 streams)
## The post-game form layer (`form_layer.py`, v10–v13; off unless switched on)

A second model on top of the daily glmer (`prior-daily`), for in-session hot and cold streaks.
The daily glmer gives each side's expected points; the layer adds a ± from how the two gamers have
done against those prices today and this session. It is built and tested, but **off**: the test so
far says it doesn't pay on the margin (below).

- Each gamer carries a **day form** and a **session form**, for the margin and for the total. A
  Kalman filter updates them after every match the gamer plays, from the match's error against the
  model's price. The opponent's form counts against them.
- **Day form** starts fresh each day at 00:00 UTC, when the daily prior refits (`--rho` carries a
  share into the next day). **Session form** starts fresh after a gap of more than 2 hours, and at
  midnight. So only the day's own games count.
- **Never against the way a gamer's day is going** (on by default; `--either-way` turns it off). A
  gamer with more losses than wins today is never priced above the daily glmer, and one with more
  wins never below it. When both gamers are down on the day, or both up, the match stays at the
  glmer's price. Without this rule, a gamer who lost by about what the price expected could be
  lifted by an opponent having a worse day. That happened to NIGHTMARE on 6 Oct: four straight
  losses and still priced above the glmer. On 41 days of daily fits the rule cut the default
  layer's cost from +0.0005 to +0.0004 log loss, and the trend layer's (trigger 3, sd 2) from
  +0.0018 to +0.0012. Neither beats the daily glmer alone.
- A match's margin moves by the home gamer's day + session form less the away gamer's; its total by
  both gamers' forms added.
- **The error is shared.** Each match's error is split between the two gamers by how unsure each
  one's form is, so a gamer whose form has settled moves less than a fresh one.
- **Only results known at the time count.** A match reads only matches that started before it, as
  `stream.known_states` needs. A settled match's error is taken against the price the model made
  out of sample, so the layer reads results from the daily prior's first fit on (a single fit: from
  its cut-off).
- **Where it sits.** It goes on top of whatever the build prices with: the prior shrink, and the
  follow layer on a single fit. A daily prior has no follow layer.

**Switching it on**, for one build, with or without settings of its own (margin[,total]):

```powershell
python -m eAMFModel form-layer v12_daily_form --on
python -m eAMFModel form-layer v12_daily_form --on --tau-day 2 --tau-session 2
python -m eAMFModel form-layer v12_daily_form --off        # or --default: back to form_layer.ON
```

The defaults are a day-form sd of 1 point, a session-form sd of 1 point, and a fresh start each
day. They're the design's starting point, not a setting the test chose.

**One gamer's matches, model against layer** (`form-trace`). Each match shows the daily glmer's
expected margin and win chance, the gamer's day and session form, the opponent's form, the ±, and
the layer's margin and win chance. `--model` prices as the build does; `--rolling` takes
`prior-daily`'s fits as they are.

```powershell
python -m eAMFModel form-trace --model v12_daily --history eAMFCalibrator/out/match_history.csv --gamer NIGHTMARE --since 2026-10-06
```

**What the test says so far.** The early read covers 27 days of daily glmer fits on AMFELO (16–25
Jul, 4–13 Aug, 23–29 Aug; 2,523 matches). Each match was priced by its own day's fit, with glmer's
form following the results. Settings were chosen on the first half of the days and scored on the
second.
- **Nothing is left on the margin.** On these daily fits, the next match's error doesn't follow the
  gamer's errors so far: −0.002 a point this session (t −0.2), −0.0004 earlier today (t 0.0),
  −0.011 yesterday (t −1.3). On 14-day fits this session's errors did carry on: +0.0135 a point (t 2.8).
  A daily refit, plus glmer's own form, takes in what was there.
- **Every margin setting is worse out of sample.** The more the layer reacts, the worse it gets:

  | margin sd: day, session | NIGHTMARE's own form after 8 matches, 22 Jul / 24 Aug | moneyline log loss | margin RMSE |
  |---|---|---|---|
  | 1, 1 (the default) | −0.43 / −0.21 | +0.0015 | +0.025 |
  | 2, 2 | −1.16 / −0.55 | +0.0088 | +0.141 |
  | 3, 3 | −1.74 / −0.73 | +0.0213 | +0.331 |
  | 0, 3 (session only) | −1.24 / −0.17 | +0.0123 | +0.177 |

- **The total gains a little.** A day form of sd 1.5 cut the total's RMSE by 0.018 on the second
  half, and 1 + 1 by 0.020.
- **NIGHTMARE's 0–9 on 22 Jul and 1–8 on 24 Aug.** The daily glmer already had him as the underdog
  in every match of both nights (31–49% to win). He fell short of his price by 3.1 points a match on
  22 Jul and 2.7 on 24 Aug. With the default settings the layer moved his price by at most 1.3
  points, mostly through his opponents' form; his own form got no lower than −0.4 points. On 25 Aug
  he went 4–5, 0.2 points a match short of the daily glmer's prices.

The full test (56 days, 16 Jul – 9 Sep) is the one that sets the defaults or keeps the layer off.

**The S-curve (`--s-curve`).** In place of day and session form, the margin moves by the chance a
gamer is having an off (or on) day, given his errors against the pre-match price so far today. The
default curve is fitted by EM on Apr–Sep (4,444 gamer-days, 14-day fits with the follow layer):
off on 1.65% of gamer-days by −5.5 points, on on 7.5% by +1.3. The shift starts small and steepens
from about the third poor game. It levels off at the cap. Only the day's own games count, and the
same-way rule applies; the total still follows the day form.
- **Off days come and go by month.** Fitted month by month: −5.1 points on 2.4% of days in April,
  −7.7 on 1.5% in June, −4.4 on 3.8% in July, none in May or August. They don't belong to
  particular gamers: a gamer's share of off days in odd months correlates +0.12 with even months
  (63 gamers).
- **Forward test.** Fitted on Apr–Jul (1.25% off by −6.25), it was level on Aug–Sep: log loss
  −0.00007 and Brier −0.00002 on 14-day fits, +0.00025 and +0.00013 on the daily glmer. It moved
  only 2% of matches by half a point or more, and on those it lost (+0.010 to +0.018 log loss),
  because August had no off days. Judge it in the bet sim.

```powershell
python -m eAMFModel form-layer v12_daily_s --on --s-curve
```

**Only once a trend shows (`--trigger K`).** Punters pile on once a run shows, two or three games
in. With a trigger, a gamer's margin form counts only once they have lost (won) at least K in a row
tonight, and only the way the run goes. A run is the night's: kick-offs no more than 6 hours apart,
so a break of 2–3 hours doesn't end it. The total is not gated.

```powershell
python -m eAMFModel form-layer v12_daily_trend --on --trigger 3
python -m eAMFModel form-trace --model v12_daily --history eAMFCalibrator/out/match_history.csv --gamer NIGHTMARE --since 2026-10-06 --trigger 3 --tau-day 2 --tau-session 2
```

**What happens after a run, against the daily glmer's price.** These are the matches right after
a run, from 41 days of daily fits (3,791 matches), with a run counted over the whole night:

| run tonight | next matches | margin vs price (se) | won | priced |
|---|---|---|---|---|
| lost the last 5+ | 159 | −0.54 (0.67) | 34.6% | 38.7% |
| lost 4 | 172 | −0.30 (0.74) | 40.7% | 43.7% |
| lost 3 | 389 | +0.26 (0.48) | 44.5% | 44.8% |
| lost 2 | 783 | +0.01 (0.34) | 46.0% | 47.6% |
| won 2 | 803 | −0.30 (0.33) | 52.2% | 52.1% |
| won 3 | 367 | +0.36 (0.50) | 57.5% | 54.8% |
| won 4+ | 366 | −0.53 (0.48) | 57.9% | 58.8% |

- **Two or three games in, when punters pile on, the next match is priced right.** That holds both
  ways.
- **Four or more losses** win 3–4 points less often than priced. The 14-day fits, with 15,418
  matches, show the same (−4.4 points after 4, −1.8 after 5+). But the margin is within its noise
  every time, and winning runs don't carry on at all.
- **Nothing tuned on it holds up.** Several kinds of shift were chosen on the first half of the
  days and scored on the second: a flat or per-game shift after 2–5 straight losses (or wins), the
  layer gated as above, and a shift once the night's error passes a threshold. On the daily glmer,
  all 67 settings made the second half worse, by +0.0001 to +0.03 moneyline log loss. On 14-day
  fits, the best took off 0.0003: shading a gamer 0.5 points after 4 straight losses.
- **NIGHTMARE.** With `--trigger 3` and form sd 2, the layer would have moved him 0.2–2.1 points
  from his fifth game on 22 Jul. On 24 Aug its biggest move, −3.7 points, came on his one win of the
  night (45–38).

The bets say more than the prices here. Every `bets` run now shows the book's margin on the bets
that pile onto a run, against those that fade it (`bets_trends.txt`, in eAMFCalibrator's README).
If piling on beats the book, a trigger is worth turning on.

## Pricing only what the model is sure of (v8–v12 streams)

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
python -m eAMFModel remaining eAMFCalibrator/out/scouting_playover.csv --version v9 --model v9_model \
    --history eAMFCalibrator/out/match_history.csv --since 2026-09-10 --until 2026-09-22
```

Build the model on matches before `--since`, or the comparison is in-sample:
`vN-build --until DATE` builds everything (play tables, profiles and NB2) on
the matches before DATE:

```bash
python -m eAMFModel v9-build eAMFCalibrator/out/scouting_playover.csv --half all --until 2026-09-10 \
    --out v9_model_pre10 --history eAMFCalibrator/out/match_history.csv
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

**By drive: `--drive`.** The rest-of-game view mixes up who scores and
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

## Testing a model by hand: `trader`

```
python -m eAMFModel trader --model v10_model
```

This opens a local page (http://127.0.0.1:8765) for clicking through a game and seeing v10's
prices after every play. It needs only the v10 build; nothing else is installed. The pre-match
model's expected points need pandas and scipy, like the build.

For v2.1, and for a historic match against prod's quotes, use eAMFCalibrator's `replay-ui`
(a match replayed a PLAY_OVER at a time, with the nudge set by hand).

- **Match:** pick the two players, their teams and the stream. *Get expected points* asks the
  build's pre-match model (NB2 or glmer, with its shrink) for each side's expected points. The
  points can be typed over before *Start match*, after which the pane shows the pre-match
  prices.
- **Away @ home:** the page lays a match out the American way: the away side on the left of the
  scoreboard, the field and the match pane, the home side on the right, an @ between them, and
  in the prices the away side's column (with the over) first.
- **Possession:** a football beside a side's name on the scoreboard marks the side with the ball
  (the receiver at a kick-off, the scorer at a conversion).
- **Team colours:** each side is drawn in its NFL team's main colour (all 32 teams, in
  `trader.html`'s `TEAM_COLOURS`), lightened or darkened to stand out on the page's theme. When
  the two are too alike (CIE76 delta E under 35, or the same team) the away side takes its next
  colour -- road kits are white, so that is the kit's trim (49ers gold against the Chiefs' red,
  Seahawks green against the Patriots' navy) -- and the match pane says so.
- **Plays:** each play button moves the game by the rules the sim plays: 4-minute quarters,
  downs and distance, first downs, touchdowns and conversions, field goals, punts, turnovers
  and safeties. Three timeouts a side each half, called with the *Timeout* button under each
  side's timeouts on the scoreboard; the second half is kicked to
  the side that kicked first. A level game goes to overtime off a kick-off, two timeouts each.
  *Ball now at* is where the ball is after the play: a half (home's or away's) and a yard line
  1-50, set to the ball's spot before it, with the gain shown beside it. Scrolling up over the
  yard line (or ↑) moves the ball forward for the side with it, through the 50 into the other
  half; down (or ↓) moves it back; shift moves 5 yards at a time. *Play*, *Punt*,
  *Turnover* and *Kick-off* take it from there. *Edit state* sets anything directly, and *Undo*
  (in red, under the first play button) steps back once you confirm it.
- **Clock:** it runs on the scoreboard. ▶ starts it (or the space bar) and ⏸ stops it; ⏩ runs it
  forward the seconds beside it (20 by default). Each play is recorded at the clock showing when
  it is entered. Plays that stop a real clock (incompletions, scores, kicks, turnovers, timeouts)
  stop it; a run or a completion leaves it running. Stopping it and fast-forwarding
  reprice; at 0:00 the period ends by itself.
- **Prices:** after every play v10 prices the state it leaves: moneyline, handicap and total at
  its own key-number lines (held from play to play), with ladders either side, the expected
  score, and the final margin and total distributions split at the quoted lines. Odds carry the
  margin set at the top. The charts follow home's win probability against pre-match, and the
  expected total against the quoted line. Hover any chart for the play behind it.
- **Prices are repeatable:** every state of a match is simulated from the same random paths, so
  a play's price change is the play's alone. `--paths` (or the page's *Games a price*) trades
  speed for noise: about 0.3 s a price at 4,000.

`game.py` is the play engine and `trader.py` the server; the page holds the game and sends the
state with every request.

**Any version.** The trader prices with the version whose build it is given (`v10tables.npz`
means v10; `--version` overrides it), so `--model v9_model` runs v9 and a v11 build runs v11 with
nothing changed. A version needs what `trader.INTERFACE` lists: `price_states`, `price_kickoff`,
`prior_theta`, `key_line` and the rest, plus its stream's `model_paths`. Every version copied from
the last keeps them, and `test_trader.TestEveryVersion` checks each `vN.py` there is, so a new
version that drops one fails the tests before it reaches the page.

## Run it

Three steps: refresh the data, build, report. v14's build is the optimum build with nothing to set:
every input from the refresh, glmer as the pre-match model (`glmer/fit.R`'s defaults), and that model
refitted for every day from the build's cut-off to the history's last day, each on every result
settled before its day began (the cut-off day's fit is the build's own).

```bash
# 1. the data every build and report reads: snapshots and timeouts from 24 Aug, every settled match
python -m eAMFCalibrator refresh --until 2026-10-12

# 2. the build on everything before the cut-off (about 100 minutes, then the daily fits)
python -m eAMFModel v14-build --until 2026-10-01

# 3. the report against prod over the days after the cut-off, with the bets and the margin uplift
python -m eAMFCalibrator report --since 2026-10-01 --until 2026-10-12 --candidate v14 \
    --bets --margin-chart FD_109_BALANCED.csv
```

- **Live:** build without `--until`, then each day `eAMFCalibrator refresh` and `eAMFModel
  v14-daily` (fits the days since the last one and keeps the build attached).
- `v14-build --no-daily` prices from the build's single fit; `--prior nb2` builds on NB2 instead.
- `v14-build --sim-workers 12` runs the build's simulations on 12 processes (default 4): the same
  model, sooner.
- v8-v13 keep their own commands and defaults (`--prior nb2`, `--half train`; `prior-daily` to
  attach daily fits).

```bash
python -m unittest eAMFModel.tests.test_base eAMFModel.tests.test_remaining eAMFModel.tests.test_v14
```
