# eAMFCalibrator

Calibration suite for the GAMEPLAI in-play models on eAMF (Competitive
Gaming American Football). Takes either stream as input, splits the data
into cells, and asks the only question that matters for a probability:
**when the model said 30%, did it happen 30% of the time?**

```
py -m eAMFCalibrator refresh --until 2026-10-12   # the data the builds and reports read, in one go
py -m eAMFCalibrator preflight              # check tables + the play clock
py -m eAMFCalibrator report                 # everything, one HTML (start here)
py -m eAMFCalibrator report --days 7        # rolling window, no config edit
py -m eAMFCalibrator report --since "2026-09-17 10:00:00" --until "2026-09-19 00:00:00"
py -m eAMFCalibrator directional            # paired head-to-head
py -m eAMFCalibrator cross                  # cell view, under the line rule
py -m eAMFCalibrator run prod
py -m eAMFCalibrator run candidate
py -m eAMFCalibrator run both               # runs both, then diffs them
py -m eAMFCalibrator compare out/prod_cells.csv out/candidate_cells.csv
```

`refresh` writes `scouting_playover.csv` and `timeouts.csv` (the snapshots and timeouts from
`--since`, default `REFRESH_SINCE`, 24 Aug) and `match_history.csv` (every settled match before
`--until`) into `out/` over one connection: what `scouting`, `timeouts` and `history` write one at a
time, and what `eAMFModel v14-build` reads with no options (see eAMFModel's README, "Run it").

Four views:

- **`report`** — one pairing pass, one HTML, two headline views:

  1. **Directional calibration** — paired at each snapshot, with per-market
     clustered tests.
  2. **Cross-section calibration** — one bucket is score difference ×
     quarter × possession, *all three together*, for every market. Sparse by
     design and denser as matches accumulate; cells under
     `MIN_CELL_MATCHES` are dimmed rather than dropped, since knowing a
     bucket is thin is part of the information. Sortable.

  The handle check, the per-selection breakdown, integrity checks, the
  mirror check, the by-day view and the single-axis
  breakdowns sit behind a collapsed **Checks** disclosure whose summary line
  says whether anything failed. Console prints a compact version; `--axes`
  prints them in full.

  Clicking any row in any table on the page pins it, so a row stays legible
  while you scroll a wide table or compare it against another. Click again
  to unpin, Escape clears them all.

  Then **every paired observation**, with the score at the snapshot (home,
  away and the difference), the game state (field position, down and
  distance, and seconds to the match's last quote), both streams' line,
  price, probability, outcome and error, sorted by widest probability
  disagreement. A search box above the table filters it to one match id.
  Around 600 bytes per pair row, so a three-day window lands near 5 MB.


- **`directional`** — at the same snapshot, on the same selection, which
  model was closer to the result? Paired, so the between-snapshot variance
  cancels. This is the one a day of data can answer.
- **`cross`** — the same pairing, shown as cells: score difference, quarter
  and possession, with both streams' predicted against the shared realized
  rate. Applies the line rule (below).
- **`run`** — single-stream calibration: is this model's 30% really 30%?
  Needs many snapshots per cell before a realized rate means anything. Its
  per-stream numbers are sound, but `run both`'s side-by-side does **not**
  apply the line rule — use `cross` for a rule-respecting comparison.

## Window and tuning, at runtime

Nothing needs a `config.py` edit. Every command takes the same flags:

| flag | what it does |
| --- | --- |
| `--days N` | rolling window of the last N days (overrides `--since`) |
| `--since` / `--until` | explicit window bounds |
| `--sport` | sport code |
| `--tolerance` | snapshot-to-quote match tolerance, seconds |
| `--message-gap` | widest message offset allowed when pairing |
| `--clock` | which stream supplies the snapshot clock, or `self` |
| `--spread-resolution` | `literal` or `complement` |
| `--time-axis` | `period` or `drive` |
| `--chunk` | matches per batch, if memory gets tight on a long window |
| `--candidate` | what stands in for the candidate: a table, or an eAMFModel version (`v8` to `v14`, `v21`) |
| `--candidate-label NAME` | what the HTML reports call the candidate (default: the model version when one stands in, e.g. `v9`) |
| `--v9-model DIR` | `eAMFModel v9-build` output for `--candidate v9` (default `$EAMF_V9_MODEL`, then `./v9_model`) |
| `--v9-paths N` | games simulated per snapshot for `--candidate v9` (default 2000) |
| `--v9-lines own\|even\|prod` | `--candidate v9`: quote v9's own line, in the gap between the key numbers and moved as the game moves (`own`, the default), at the half-point line nearest 50% (`even`), or read v9's price at prod's line (`prod`) |
| `--v8-model`, `--v8-paths`, `--v8-lines`; `--v10-model`, `--v10-paths`, `--v10-lines`; `--v11-model`, `--v11-paths`, `--v11-lines`; `--v12-model`, `--v12-paths`, `--v12-lines`; `--v13-model`, `--v13-paths`, `--v13-lines`; `--v14-model`, `--v14-paths`, `--v14-lines` | the same for `--candidate v8`, `--candidate v10`, `--candidate v11`, `--candidate v12`, `--candidate v13` and `--candidate v14` |
| `--v21-model`, `--v21-paths`, `--v21-lines` | the same for `--candidate v21` (default `$EAMF_V21_MODEL`, then `./v21_model`; 1000 games a snapshot, the kick-off twice that) |
| `--v21-prior-model DIR` | `--candidate v21`: the v13/v14 build (made with `--history`) whose pre-match expected points v2.1 nudges toward (default `$EAMF_V21_PRIOR`, then `$EAMF_V14_MODEL`, then `./v14_model`) |
| `--v21-nudge-share S` | `--candidate v21`: how far toward the prior's expected points each side's offense is nudged from the sim's own means (default 0.75; 0: the sim's own, no prior) |
| `--candidate A,B` | several candidates side by side in one report (see below) |

Every run prints the window it actually used.

**The default start stays anchored to when the candidate changed**, and should:
quotes before that instant came out of the *old* candidate and would pollute
the comparison. `--days` is for slicing on top of that, not for replacing the
anchor's purpose. The window grows on its own as matches settle — that is the
intended behaviour, not drift.

## By day

Because the window only grows, the question as games accumulate is whether the
answer is **stable**. The per-day table gives each day's pairs, win rate,
clustered ΔBrier and CI, and says whether the days agree on direction. A day
out of step with its neighbours is worth a look before it gets averaged away.

## The line rule

A different line overrules the probability. Probabilities are only compared
where both streams quoted the **same** line; where the lines differ, the
comparison is on which line landed closer to the result.

Restricting to same-line pairs buys a property that makes the cell tables
readable: the same line is the same question, so both streams resolve to
**one shared realized outcome** per cell. The realized column is common and
only the predictions differ, which reduces each cell to "whose number was
nearer the truth". `TestCrossSectionalLineRule` pins that invariant, along
with the two views partitioning the pairs exactly.

### Where the rule bites

**Per pair.** `decisive_winner` settles each pair on the question it
actually asked: the closer line where the lines differ, the closer
probability where they match. Probability breaks a tie only when the two
lines are exactly equidistant from the result — which is reachable with
whole-number lines straddling it, and where each stream was still graded
against its own outcome, so the probabilities are a real comparison.

**In the pair table.** A different-line row shows a dash for &Delta;prob
and for both error columns. Each stream's own line and probability still
show — those are facts about the row — but the *comparisons* between them
are withheld, because a probability quoted against 44.5 is not comparable
to one quoted against 60.5, and a line error is in points.

**In the overall verdict.** The two halves answer different questions in
different units, so there is no average of them to take. What combines is
the per-pair *decision*, which is unit-free, so `decisive_block` reports
counts and the match-clustered vote and deliberately reports no mean
error. Where that separates the two models it **overrules** the same-line
Brier, exactly as a different line overrules a probability on one pair —
and where the two readings point opposite ways, the headline says so
rather than picking one silently.

That combined reading is the *weaker* test: a win rate throws away how
much closer each was, which is what the per-half paired deltas keep. Read
it for direction and the halves for strength.

## Both sides, when checking rather than reading

The pooled tables read **one** selection per market (`CANONICAL_SELECTIONS`)
because the two sides are complements — one carries the information and the
other is its mirror, and pooling them forces realized and predicted to 0.500
by construction.

That is right for reading a result and wrong for checking one. A fault
confined to one side — a line parsed for Over and not for Under, outcomes
resolved the wrong way round — averages away into a flat market row. So the
checks carry a **by selection** table: every side of every market with its
own clustered test, under both the same-line and different-line views, with
the side the pooled tables read marked.

`TestSelectionBlocks` pins the point directly: a candidate 0.1 out on Home
and 0.7 out on Away cancels to exactly 0.0000 pooled, and splits to +0.2400
and -0.2400 per selection.

## One selection per market, always

A market's two sides are complements, so every `(p, y)` arrives with a mirror
`(1-p, 1-y)`. Pool them and the realized rate **and** the mean prediction
both average to exactly 0.500 whatever the model does — the calibration gap
is cancelled out of existence before you can measure it. A realized rate of
0.500 in every cell is the signature of this bug, not a fact about the data.

So the cross-sectional view takes one side per market
(`config.CANONICAL_SELECTIONS`: moneyline Home, spread Home, total Over) and
never mixes markets in a cell — moneyline, spread and total are different
questions with different base rates, and their average describes none of
them. `TestMirrorCancellation` pins both halves of this.

Nothing is lost by taking one side — **provided the sides really are
complements**, which `cross` now tests rather than assumes:

- **P_SUM** — the two sides' probabilities added. 1.0000 means a fair book
  with no overround, so each side is exactly the other's complement and one
  can be dropped for free. Above 1 means the discarded side holds a little
  information the kept one does not.
- **PARTITION** — should be 100%: exactly one side wins. Spread is the one to
  watch, because market 52 reads "PLAYER 1 over L" and 53 reads "PLAYER 2
  over L". If both carry the *same* L they overlap rather than partition: at
  L = -2.5 both win for any margin between -2.5 and +2.5. `BOTH WON` counts
  those cases.

## Spread: two propositions, or one with a yes and a no?

Market 52 reads "PLAYER 1 to score over L more than PLAYER 2"; 53 reads
"PLAYER 2 to score over L more than PLAYER 1". Two readings are possible and
they give **different outcomes**:

| Reading | 53 wins when | Behaviour at the same L |
| --- | --- | --- |
| `literal` | `margin_2 > L` | overlaps 52 — both win on any margin in (−L, +L) |
| `complement` | `margin_1 <= L` | partitions by construction |

`config.SPREAD_RESOLUTION` switches between them, and `cross` prints a report
that decides from the data rather than from a reading of the text. Both
readings are recomputed there from the line and the realized margin, so the
verdict does not depend on whichever resolution was active when the pairs were
built.

The decisive combination is **probabilities summing to 1 while the two sides
carry the same L** — complementary probabilities require complementary events,
so that pairing proves the literal reading wrong. Mirrored lines (−2.5 against
+2.5) that partition cleanly prove it right. The report names the reading it
supports and flags a mismatch with the configured one.

A **both-sides table** prints every selection's calibration next to the one
actually used. That is a consistency check, not an extra finding: if the
sides are complements, each pair of rows has realized summing to 1.000 and
gaps that are equal and opposite. If they do not, the pipeline is measuring
something other than what it claims.

`preflight` confirms the tables and reports how the snapshot clock is being
reconstructed — worth a look after any feed change.

## How an observation is built

1. **Snapshot** — one per drive, taken at the drive's first play. Drives come
   from `INPLAY_FIELD_POSITION_PERIOD`, cleaned with the score-anchored rules
   carried over unchanged from `analysis/clean_possession_sequence.py`.
2. **Quote** — for each of the six selections, the model quote nearest in
   time to the snapshot, within **3 seconds**. Snapshots with no quote in
   range are counted in the coverage block, not silently dropped.

   The play feed has no clock of its own (see below), so a snapshot's time
   comes from the stream's `EVENT_MESSAGE_COUNT` → `PUBLISH_TIME` map:
   exact where the stream quoted that same message, interpolated across
   short gaps, dropped where the gap is too wide to trust. Every run prints
   the split.
3. **Outcome** — did that selection result true, from `SCORE_ENDGAME`.
   Pushes (final lands exactly on the line) are counted and excluded, since
   there is no 0/1 outcome to score a probability against.

Observations are then aggregated into cells and scored with Brier, log loss
and ECE. Each run writes `<stream>_cells.csv` and `<stream>_observations.csv`
to `out/`, so runs are comparable over time as data accumulates.

### Every PLAY_OVER instead of one per drive: `--snapshots play_over`

```bash
python -m eAMFCalibrator report --since 2026-09-17 --until 2026-09-23 --snapshots play_over
```

The drive-start snapshots give about 10 a match, off the play table's noisy
drives. `--snapshots play_over` takes every `PLAY_OVER` in SCOUTING_FULL
instead, about 60 a match: the same snapshots v4–v6 price off and the bet
simulation checks against. It carries the period, score, side on the ball,
down, distance and field position. Each is paired at the first message
both streams quoted from the `PLAY_OVER` until the next play starts, so
both prices were made on the same feed. The pair's `anchor` is
`play_over`.

Matches with no SCORE_CHANGES rows (but a non-zero final) are left out
(`matches_without_score_rows`), since every score bucket would read 0–0.
So are matches with no `PLAY_OVER`s (`matches_without_play_overs`).
SCOUTING_FULL's play detail has been empty since 23 Sep, which only leaves
the side on the ball unknown; the score and period still bucket.

## Directional comparison

```
py -m eAMFCalibrator directional            # console + one-screen HTML
py -m eAMFCalibrator directional --html report.html
```

Pairs prod against candidate at each drive-start snapshot and reports both
halves of the comparison, because they do not always agree.

The HTML panel is **two tables**: the combined reading over every pair,
then the two halves it is made of.

| Reading | Pairs | Matches | On prob | On line | Cand win | Match vote | Level | p |
|---|---|---|---|---|---|---|---|---|
| Overall | every decided pair, under the rule that applies to each |

| Reading | Pairs | Matches | Cand win | Match vote | Level | Δ | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| Same line | Δ is ΔBrier per match |
| Different line | Δ is Δpoints per match |

### Which prod row a pair reads: `pair-check`

A pair reads prod's **first** live row on the first message after a
PLAY_OVER; the betting simulation reads prod's **latest** row at the bet.
Where prod publishes a quick price and corrects it on the same message,
the pairing grades a candidate against the quick one. `pair-check` reads
prod's moneyline alone (no simulation) and sets the first row against the
last at every PLAY_OVER, graded on the result:

```
py -m eAMFCalibrator pair-check --since "2026-10-01 00:00:00" --until "2026-10-13 00:00:00"
```

Brier first − last above zero means the last row was the closer one. Set it
beside a candidate's moneyline edge to see how much of that edge the
pairing alone could account for (`out/pair_check.csv`, a row per quote).

### The report is a dashboard, not a write-up

Headings, tables and numbers. No paragraph explains what a table is for,
no heading asks itself a question, and the verdict is a line of figures
rather than a sentence — where the two readings disagree, both are on
the page and the reader can see it without being told. Column meanings
live in header tooltips, which cost nothing until hovered. A test keeps
it that way: nothing rendered outside a tooltip may run to twelve words.

### Each stream is graded against its own line

The streams do not always quote the same line, so the run splits on that
before comparing anything:

- **Same line** — both quoted the same number, so their probabilities answer
  the same question. Compared directly on which sat closer to its 0/1. This
  is the clean comparison.
- **Different line** — the probabilities are answering different questions,
  so comparing them head to head would score two questions against one
  outcome. Compared instead on **which line landed closer** to the actual
  margin or total, in points. A secondary view still scores each probability
  against its own line, which is fair but measures line choice and
  probability together.

Grading both streams against *one* stream's line is a bug, not a
simplification: a total of 45 is over 44.5 but under 46.5, so the other
stream gets marked on a question it never asked. `TestOwnLineGrading` pins
this, including the direction of the error it used to cause.

A stream quoting whole-number lines against a half-point book will rarely
share a line, and whole numbers can push (the result lands exactly on the
line, so there is no 0/1 for that side). Pushes are counted per stream, and
line closeness still works for them — a line on the number is a perfect
line. The run header reports same-line rate and the half/whole split per
market, so this is visible before any verdict.

**Win rate** — how often each model was closer to the realized 0/1. This is
the intuitive reading, and it is a weak test. If both models are unbiased
around the same truth and differ only in noise, then whichever lands closer
to the realization is decided by which noise draw happened to point at the
outcome — a coin flip, regardless of which model is actually better.
Simulated with the candidate at half prod's noise:

| truth | candidate win rate | Brier improvement |
| --- | --- | --- |
| 0.50 | 49.9% | +0.0129 |
| 0.65 | 49.9% | +0.0133 |
| 0.80 | 49.9% | +0.0102 |
| 0.95 | 52.5% | +0.0069 |

A strictly better model barely wins more often. The win rate throws away
magnitude, so it cannot see an improvement that consists of being less wrong.

**Paired error difference** — mean per-match `prod_loss - candidate_loss`,
for Brier and for absolute error, with a bootstrap CI over matches. This
does see it, and it clusters correctly: matches are independent, pairs
inside them are not. If the CI straddles zero, the data has not separated
the two models yet.

Read the win rate as a direction check and the paired difference as the
verdict. Both are broken down by market, by how much the two models
disagree, by prod's probability, and by the three split axes below.

Pairing is on `EVENT_MESSAGE_COUNT`, not on each stream's nearest quote in
time: both streams carry the same feed sequence, so the same message is the
same event for both. Matching independently on time would let one stream
land 0.1s from the snapshot and the other 2.5s away, scoring two different
game states against one outcome.

## Pre-match calibration

```
py -m eAMFCalibrator report      # the section, and prematch_closing.csv
```

Everything else in this suite reads prices taken DURING a match, at a
drive start or between two plays, where being right and being *fast* are
tangled together. This reads the last price published **before** the
match got under way and asks the oldest question there is: when the
model said 62%, did it happen 62% of the time? There is no play feed to
react to, so nothing to be quick or slow about — only whether the number
was right.

### What counts as pre-match

Taken from the play feed rather than from a schedule. The play rows
begin when the match does, so a quote is pre-match when it carries **no
message count at all** or one **below the first play row's**. That needs
no column the rest of the suite does not already read, and it cannot
drift out of step with a kickoff time nobody publishes.

Those rows were being fetched and thrown away: `index_by_message` drops
a quote with no message count, which is exactly what a pre-match quote
has.

### One observation per (match, selection, stream)

The **closing** quote — the last one before kickoff, the most informed
price the model ever published without seeing a snap. Taking the first
instead would grade it on how it opened, which is a different and much
easier question. The count of pre-match quotes travels with it, so a
market quoted once months out is not read as the same kind of thing as
one quoted two hundred times up to kickoff.

A selection the final score cannot settle — a push, or a match with no
final — is counted and dropped. It has no realized 0/1 to compare a
probability against.

### Read the price spread before the calibration

The panel opens with how far the closing prices sit from even money,
because everything under it depends on the answer. **A book that never
leaves 50/50 has no view to be calibrated, and its Brier sits at 0.25
whatever else is true.** Bands rather than a standard deviation, since
the question is whether any price is ever confident, not what the
average one looks like.

For scale: a realistic NFL moneyline book has a standard deviation of
about 0.24 and a Brier near 0.19.

### Per selection, and per line, against ONE realized rate

Laid out the way the cross-section is, and for the same reason. **Pooling
the two sides of a market forces the realized rate and both predictions
to exactly 0.500 whatever the model does** — Home and Away are
complements, so the mean of a price and one minus it is 0.5 by
arithmetic. The Brier survives that pooling; the gap does not, and a
table of `0.500 / 0.500 / +0.000` says nothing at all.

So every row is one selection, with the realized rate once and each
stream's prediction and gap beside it. Spread and total also break down
**by line**, which is where the pooled view hides the most: the feed
uses a handful of distinct lines, and a model quoting 0.500 on all of
them is badly wrong on the outer ones and right in the middle, which
averages to looking fine.

### The line rule applies here too

A pair only exists where both streams closed on the **same line**. Where
the lines differ they resolved against different outcomes, so neither
the shared realized rate nor the Brier difference means anything. Those
are counted and dropped, as they are in play.

Like the in-drive view it **costs no extra queries**: `build_pairs`
already fetches every quote for the chunk, so a `prematch.Sink` rides
along on that same fetch.

## The in-drive section on the main report

`report` carries a high-level in-drive panel: the hit rate per stream
overall and split in-drive against drive-end, the drive outcomes and
what they were worth, and the head to head. Everything else — by
outcome, by period, by market, by selection, the checks, the CSVs —
lives in `indrive`, and the panel names that command rather than trying
to be it.

It costs **no extra queries**. `directional.build_pairs` already holds
the plays, the scores and both quote indexes for each chunk of matches,
which is everything the in-drive analysis reads, so an `indrive.Sink`
rides along on that same fetch. A second pass would have doubled the
query load to read the same rows twice.

## Reading the gap columns

Every column headed a **gap** — the calibration gap (realized minus
predicted), the line gap, `±Msg`, `ΔLine`, `Δprob` — asks the same
question: how far apart are two things that were meant to agree. They
share one green-to-red ramp and differ only in where its steps fall.

| Column | ≤ | ≤ | ≤ | ≤ | above |
|---|---|---|---|---|---|
| calibration gap, `Δprob` between streams | 0.02 | 0.05 / 0.04 | 0.10 / 0.06 | 0.20 / 0.10 | — |
| line gap, `ΔLine` (points) | 0.5 | 1.0 | 3.0 | 6.0 | — |
| `±Msg` (feed messages) | 0 | 1 | 2 | 3 | — |

The steps are **fixed**, not quantiles of each run, so two reports can be
read against each other. They come from the observed distributions:
calibration gaps run p25 0.02, p50 0.07, p75 0.23; line gaps p75 1, p90
3, p95 6; `Δprob` p50 0.018, p90 0.061, p95 0.080.

**It is magnitude, not sign.** A gap is a distance from agreement, so
+0.40 is no better than −0.40. This replaced a sign-based colouring that
painted a badly over-predicted `+0.40` *green* and a near-perfect
`−0.001` *red*. The columns where the sign genuinely is the point —
ΔBrier, Δpoints, candidate win rate — still read by sign.

A cell with nothing to compare (a dashed `Δprob` on a different-line
pair) gets no colour at all. "Not comparable" is not a small gap.

### The colour is on the number, and it is hue only

A wash of filled cells reads as a heat map when what is wanted is a
table, so the ramp colours the type.

It is **hue only**. Every step sits at the same OKLCH lightness as
`--bad` in its own theme — 0.501 on light, 0.693 on dark — and differs
from its neighbours by hue alone, so the ramp sweeps green → amber →
orange → red without ever getting brighter, and its last step *is*
`--bad`: `#b3261e` and `#ef6f66`, the same red the ΔBrier columns
already use.

The cost is named rather than hidden. Green and red at equal lightness
are the one pair red-green colour blindness cannot separate, and a
constant-lightness ramp gives up the second channel that would otherwise
carry the ordering. What carries it instead is the **number**, which is
in the cell and is the real answer, and the **key** above each table,
which names every band. Every step still clears 4.5:1 against its panel
— 5.6 at worst — so it is readable type first and coloured second.

### Every panel folds

The heading is the handle: click it, or tab to it and press Enter. The
nav carries a **collapse all**, which closes the native `<details>`
groups as well as the sections, so it does not collapse most of the page
and quietly leave the rest open.

Done in script rather than in the markup, so a panel does not have to
know it is foldable — the heading becomes the handle and everything
after it becomes the body, whichever function built it.

## In-drive reaction: `indrive`

```
py -m eAMFCalibrator indrive                    # every match in both streams
py -m eAMFCalibrator indrive --match AF063170926
py -m eAMFCalibrator indrive --matches 50
```

Every other view asks whether a probability was **right**. This one asks
something cheaper to be sure about and harder to fake: does the model
**react**? A first down is good for the team with the ball. Its moneyline
should go up. If it does not, the number is not reading the game, whatever
its Brier score says.

### The unit is a transition

Two consecutive cleaned play rows, and what happened between them. Inside a
drive that is one snap to the next; at a drive's end it is the last row of
one possession to the first row of the next. Each is classified from the
play feed alone — down, distance, field position, and any points on the
scoreboard — and each class carries an expected direction for the side in
possession.

| Outcome | For the offence | How it is read |
|---|---|---|
| `touchdown` | good | six or more points to the side with the ball |
| `field_goal` | good | exactly three |
| `extra_point` | good | one or two: a PAT or a two-point conversion |
| `score` | good | any other points short of a touchdown |
| `first_down` | good | the down reset to 1 with the ball forward |
| `big_gain` | good | 5+ yards, short of the line to gain |
| `short_gain` | — | 1–4 yards, short of it |
| `no_gain` | bad | the ball did not move |
| `loss` | bad | the ball went backwards |
| `failed_conversion` | bad | 3rd or 4th down came and went without a first |
| `possession_lost` | bad | the drive ended and nobody scored |
| `points_against` | bad | a safety, or the defence scored |

`short_gain` carries **no** expectation. Three yards on 3rd and 8 helps
nobody, so it is counted and shown and scored by nothing. Points settle a
transition before any yardage does, and they are signed to the side with
the ball — a pick six comes back negative and cannot read as a good play.

**Gains are judged by expected points, not yards.** Five yards on 1st and
10 is about an average Madden play, not a good one, and nine yards on 3rd
and 12 is a failure. So `big_gain`, `short_gain`, `no_gain` and `loss` take
their direction from what the play did to the offence's expected points on
the drive (`expected_points.py`: the points real drives went on to score
from each down, distance and field position, 95,000 states off
SCOUTING_FULL). A play that raised them is good, one that lowered them is
bad, and within 0.15 points of no change it sets no direction. The class
names still describe the yardage. Rebuild the table from a newer export
with `py -m eAMFCalibrator expected-points out/scouting_playover.csv`.

**A gain inside a drive sets no direction for a total.** It makes points on
this drive likelier, but it uses clock the rest of the game needed. A
30-second play is worth about 1–1.5 points of total, so the total can
rightly go either way. Totals are scored on what ends drives: points, stops
and turnovers.

**Moves of a point or more.** How often a move goes the right way depends
on its size. On Sep 3–9, v5's moneyline moves went the right way:

| Size of move | Right |
|---|---|
| under 0.2 points of probability | 55% |
| 0.5–1 point | 69% |
| 2–5 points | 92% |
| 5+ points | 98% |

Below a point, the play barely changed anything and the direction is close
to a coin flip for any model. So the report shows moves of a point or more
(or any line move) as their own row. Under these rules, on those plays v5
went the right way 89.5% of the time and prod 84.9%.

### Which way a price should move

A team-sided selection follows the side it names: on a good play the
offence's own moneyline and spread rise, its opponent's fall. Totals are
possession-blind — points are points whoever scores them — so Over follows
the play and Under opposes it from either side.

Scored on the **sign**, not the size. `|Move|` and `Signed` report the size
beside it, so a model that gets the direction right on a touchdown and a
five-yard gain by the same amount is visible as such.

### When the line moves, the line is what is scored

A line move is not a failure to react — it *is* the reaction, and on a live
feed it is about a third of everything. So the basis switches rather than
the move being dropped:

| The line | What is scored | Why |
|---|---|---|
| held | the **probability** | nothing else changed, so the price carries the news |
| moved | the **line** | the probability answers a different question at each end and cannot be differenced |

The direction expected of a line is **not** the direction expected of the
probability, and getting that wrong would score a third of the sample
backwards:

- A **spread** line follows the side it names, exactly as that side's
  probability would. Checked against the data rather than assumed: market
  52's line tracks the home lead almost one for one (−14.9 at a 15–21
  point deficit, +14.5 at a 21–27 point lead) and 53's is its mirror.
- A **total** line does not. Over and Under share one number — 94% of
  snapshots quote the identical value on both — so it goes **up** on a good
  offensive play whichever selection is carrying it. Under's probability
  should fall while Under's line rises.

`Signed` is reported twice, once per basis, in probability points and in
line points. They are different units and are **never** pooled; the hit
rate does pool them, because right is right whichever moved.

### What is still refused

- A transition where **either endpoint was not live**. A price nobody could
  have taken did not move.
- A line that moved but had **no readable number** at one end. That is a
  difference in the description text, not a move.
- A **flat** price is neither right nor wrong. It is reported as its own
  column rather than counted as a miss — silence is a finding, not an
  error, and on some classes it is the whole finding.

### The bias to know before reading any number

A failed third down usually **ends the drive**, so it is not an in-drive
transition at all. Restricted to plays inside one possession the sample
leans towards things going well. That is why the drive-ending transition is
built too, and why in-drive and drive-ending are reported apart rather than
pooled. The drive-ending rows are also the noisier half: their two messages
sit either side of a kickoff, so more than one play's worth of news is in
the price difference.

### Two sides of one market read the same

Wherever the feed publishes exact complements, one side goes up as the
other goes down and the expectations mirror, so the per-selection rates
come in identical pairs. That is arithmetic, not a measurement. A row that
does **not** match its partner is the interesting one — which is why the
split is shown at all.

### What the first full run said

326 matches, 18,321 transitions, and the answer to the headline question is
yes: both streams move the right way about three quarters of the time,
overall and inside a drive alike, on every split with enough rows to say so.
Four things in it are worth more than the headline.

**The model does not react to a failed conversion at all.** `RIGHT` reads
73.4%, on 158 decided moves out of 5,698 — because **97.2% were flat**, with
a mean absolute move of 0.0009. That is not a hit rate, it is a frozen
price, and the flat column is the only honest way to read that row.

**Prices freeze in Q4.** Flat runs 12.8% in Q1 and **56.2%** in Q4. Any Q4
number is computed on less than half the moves the count column implies.

**A 5+ yard gain barely registers** — 56.2% against 81.3% for no gain and
84.7% for a loss. Both streams read bad plays much better than good ones
short of a first down. `BIG_GAIN_YARDS` is a constant, and five yards on
1st and 10 leaving 2nd and 5 is close to neutral, so the threshold is worth
moving before concluding anything about the models.

**A made field goal moves the price AGAINST the kicking team** — 32.0% and
31.7%, both intervals entirely below a coin, on 1,854 and 1,887 decided
moves. Splitting `score` isolated it: `extra_point` is fine at 77.5%, and
`touchdown` at 97.7%, so this is not the drive-ending structure. It is the
three points. The check reports it as an `ERROR` because the expectation
is the likely fault, not the models: a made field goal ends the drive,
hands over possession, and banks three where the market had priced some
chance of seven, so calling it unambiguously good for the offence is a
claim the data does not support. Whether to zero its sign — as
`short_gain` already is — is a football judgement, not a coding one, and
`OUTCOME_SIGN` is where it lives.

**`extra_point` is barely measured.** 512 transitions produced 0.39 moves
each against a median of 5.28, so the feed quotes those messages far less
often than any other class. Its 77.5% rests on 178 decided moves.

### What each drive produced

The transition view answers "did the price follow the play". It cannot
answer "what did this drive do", and two gaps say why:

- a drive-ending transition needs a **following** drive to point at, so
  the last drive of every match has none at all — 326 of 4,468 on the full
  run;
- another 110 drive-ending rows carry a yardage class rather than an
  outcome, because the team label did not change across the break.

So `indrive_drives.csv` is one row per drive, every drive, carrying what
it produced and the scoreboard either side of it: `points_for`,
`points_against`, `score_p1_before` through `score_diff_after`, the
`outcome` (`touchdown`, `field_goal`, `extra_point`, `score`,
`points_against`, `no_points`) and how it `ended` — `handover`,
`same_team` where the feed never said possession turned over, or
`match_end`.

#### The windows partition the match

Drive N owns every score from where drive N−1's window closed up to the
message drive N+1 opens on; the first drive owns everything before that
and the last owns everything after. No score belongs to two drives and
none belongs to none.

That is what makes the reconciliation a real test rather than a
restatement of itself: **the points the drives claim must equal the
points the feed sent.** A mismatch means a drive boundary is in the wrong
place or a score is being counted twice, and it is reported per match and
as an `ERROR`.

A drive owns the score that *ended* it, which is why a touchdown drive
reads 7 rather than 6 — the PAT lands in the same window.

### Checks

The run checks itself, on the console and at the top of the HTML. The
findings are about the **analysis**, not the models, and the distinction
is the point: two independently built models agreeing with each other in
the *wrong* direction across thousands of plays is not two broken models,
it is one miscoded expectation.

| Severity | What it means |
|---|---|
| `ERROR` | a class whose interval sits entirely below a coin on **both** streams — the expected direction for it is backwards; or the points the drives claim not matching the points the feed sent |
| `WARN` | a class that is mostly flat, so its rate is a minority report; or one the feed quotes far less often than the rest, so it is not measured the same way |
| `note` | a thin row; or two sides of a market that do not match, meaning the feed is not publishing exact complements there |

One `ERROR` is also an invariant: a line-basis move cannot be flat, since
a move is scored on the line only where the line changed. If that ever
fires, the basis is being set somewhere it should not be.

### Output

`indrive_moves.csv` is one row per (transition × selection × stream),
carrying the whole transition so it needs no join. `indrive_transitions.csv`
is one row per transition, and `indrive_drives.csv` one row per drive.
`indrive.html` is the same tables on a page,
with the hit-rate ramp centred on a **coin**: 50% is a model reacting at
random, and a rate barely above it lands on the red end alongside one below
it.

## The split axes

| Axis | Buckets |
| --- | --- |
| Score difference | `Away 2 score`, `Away 1 score`, `Tight`, `Home 1 score`, `Home 2 score` |
| Time | `Q1`–`Q4`, `OT` (switchable to drive number) |
| Possession | `Home`, `Away` |

Crossed with market (moneyline / spread / total) and selection.

Flipping the time axis to drive number is `TIME_AXIS = "drive"` in
`config.py` — the drive number is already computed and carried on every
snapshot, it just isn't the default until drives reconcile.

## Assumptions worth checking

**The cutoff is read as a start, not an end.** `CUTOFF_START =
"2026-09-17 10:00:00"` keeps data **from** that instant onward, on the
reading that quotes before the change came out of the *old* candidate. Both
streams are cut identically so the two calibrations run on the same
population.

**The cutoff is read as UTC.** `PUBLISH_TIME` is `TIMESTAMP_NTZ` and this
feed is stamped UTC elsewhere (`PRICE_ISSUE_TIME_UTC`). 2026-09-17 was BST,
so if "10am" meant UK local time, set it to `09:00:00`.

**Score difference is `PLAYER_1 − PLAYER_2`** (home minus away), so
`Home 2 score` means the home side leads by 9 or more regardless of who has
the ball — a score here is 6–8 points, so 9+ is two of them. Possession is a
separate axis, so the home-leading-while-defending case is still visible as
its own cell.

**Market IDs** are `50`/`51` moneyline, `52`/`53` spread, `54`/`55` totals.
50/51/54/55 are carried over from `analysis/unconditional_calibration.py`;
52/53 come from the `MARKET_DESCRIPTION` text. Lines are parsed from that
description rather than from `FULFILLMENT_SCORE`, which sampling showed
holds the *achieved* margin, not the line.

**Probabilities are rescaled.** GAMEPLAI publishes `PROBABILITY` on 0–100;
everything here works in 0–1.

**`QUOTE_DIRECTION = "nearest"`** allows a quote from just before the
snapshot, as asked. Set it to `"forward"` to take only quotes at or after
the snapshot — stricter about a price predating its own game state, at the
cost of dropped snapshots. The signed gap is on every observation row, so
the bias is measurable either way.

## The handle check

Everything the calibrator buckets on is read in the `PLAYER_1` frame: the
score difference is p1 minus p2, the possession flag is Home or Away, and
each market resolves by asking which of the two won. If the handles swap
sides part-way through a match, all three invert from that point and the
match still looks perfectly well-formed — every row parses, every bucket
fills, and the numbers land in the **wrong** cells. That is worse than
missing data, because nothing downstream can tell.

Scoring only goes up in this sport, so a swap leaves a mark: one side's
cumulative total goes **down**. That is the whole check.

| Kind | What it is | Counts as a flip |
|---|---|---|
| `Mirrored` | `(p1, p2)` became exactly `(p2, p1)`. Nothing but a swap does that. | yes |
| `Regression` | A total went down some other way — a swap landing on the same message as a score, a rescinded score after review, or a feed glitch. | yes |
| `Final mirrored` | The last running total is the mirror of `SCORE_ENDGAME`. | yes |
| `Final mismatch` | The two tables disagree, but not in mirror image. | no |

`Final mismatch` is deliberately not a flip: a missing late score explains
it as well as a swap does.

### What was tried and removed

A possession cross-check used to cover both blind spots, reading football's
fixed post-touchdown sequence (TD, PAT, kickoff, the *other* team's
offence) against the play feed's Home/Away labels. On 263 matches it
flagged **84% of them** with near-zero agreement window-wide.

That is not 221 corrupt matches. Near-zero agreement *everywhere* is one
wrong assumption or a play feed too noisy to track possession — and drive
detection currently reconstructs about 9.5 drives per match against a
realistic ~22, so the second is the likely one. It has been removed rather
than left to produce false positives; it is worth rebuilding once drive
reconciliation lands, and the implementation is in git history.

The score-based checks below are unaffected — they read `SCORE_CHANGES`
and `SCORE_ENDGAME` and never touch the play feed. On that same run they
found **nothing**: no mirrored totals, no regressions, no final
disagreements across 263 matches.

What is **not** available: `EVENT.PLAYER_1_HANDLE` / `PLAYER_2_HANDLE` hold
the gamers' names, but `EVENT` is one row per match, so they say who the
slots are meant to be and nothing about whether the per-message feed
honoured that. No table carries a per-message identity.

The default **reports** flipped matches without dropping them, so the size
of the problem is visible before any data is thrown away — the console and
the report both say the flagged matches are still in the numbers. Pass
`--drop-flipped` (or set `EXCLUDE_FLIPPED_MATCHES = True`) to exclude them.

## Market state

`STATUS` and `IS_ACTIVE` live on both stream tables. They used to be a
`WHERE` clause — `STATUS = 'open' AND IS_ACTIVE = 'true'` — which meant a
quote in any other state produced no row and the snapshot simply vanished,
indistinguishable from a feed gap.

**`IS_ACTIVE` alone decides.** GAMEPLAI say the `STATUS` column is wrong,
so it gets no vote. It is still selected, still carried onto every pair
and still broken out in the report — side by side is how a disagreement
between the two columns stays visible, and the breakdown is now the
evidence for the claim rather than a decoration.

`preflight` prints what those columns really contain. On the
2026-09-17 → 09-21 window, per stream:

| STATUS | IS_ACTIVE | Rows | Share | Read as |
|---|---|---:|---:|---|
| `UNDER SETTLEMENT` | false | ~985k | 83% | not live |
| `open` | true | ~178k | 15% | **live** |
| `CLOSED` | false | ~14.5k | 1.2% | not live |
| `CLOSED` | true | ~1.7k | 0.1% | **live** |

Dropping `STATUS` from the test moves exactly one bucket, and it is worth
being explicit that this is not a no-op: `CLOSED`/`true` — about 0.1% of
rows — goes from dead to live. Under the old rule those prices were
thrown away on the strength of a column now known to be wrong. Nothing
moves the other way: `UNDER SETTLEMENT`/`false` is still out, on
`IS_ACTIVE` alone.

The state is still reported verbatim rather than mapped onto a tidy
vocabulary. Markets run `open` → `UNDER SETTLEMENT` → `CLOSED`, and
**there is no suspension value in this feed** — inventing a word for one
would be guessing where reporting the value is not.

Most non-live rows are almost certainly post-match settlement churn that a
drive-start snapshot would never land on — but "would never" is an
assumption, and the point of moving the filter out of SQL is to turn it
into a count. What the pipeline does now:

- Every quote is fetched with its state.
- A pair is not live if **either** side was: a price nobody could have
  taken is not a quote, whichever side it is.
- `errors()` refuses a non-live pair, and since `comparable`, `winner`,
  the tallies, the cells, the votes and the decisive block are all built
  on `errors()`, that one refusal keeps it out of every metric at once —
  while the pair stays visible in the table, marked in the `Live` column
  and dimmed.

The **Market state** panel reports the share, the breakdown by the raw
state per stream, the split by quarter and market, and — the number that
matters — how lopsided it is between the two streams. Pairs lost on one
side only are not lost at random: they cluster around scores, which is
where two models differ most. `REQUIRE_LIVE_QUOTE = False` scores them
anyway, so the cost of excluding them stays measurable.

## One row per message was not true

`index_by_message` kept the **first** row per `(match, market, message)`,
on the stated assumption that a message carries one row per market. For
the moneyline that holds. For the spread and total it does not: those
lines move, so a message can carry the settlement of the old line beside
the open quote for the new one, and rows arrive ordered by publish time
rather than by usefulness.

The cost showed up as markets that looked permanently dead:

| Market | Live, drives 1–3 | Live, drives 4+ |
|---|---:|---:|
| Moneyline | 100% | 100% |
| Spread | 24.7% | **2.2%** |
| Total | 16.3% | **1.3%** |

The moneyline has no line to move and never degrades. The spread and total
collapse as a match accumulates settled lines — which is the shape you get
when the row kept is chosen arbitrarily and the pool of dead rows grows.

A live row now replaces a dead one for the same message. Among rows of
equal liveness the earliest still wins, which keeps the quote nearest the
event rather than a later correction to it. The run header counts both how
many messages offered more than one row and how many pairs that rescued,
and `preflight` reports rows per message per market — including the
**mixed** case, where a message offered a tradeable quote *and* a dead one
and the choice decided whether the pair could be scored at all.

### It is not the drive-start anchor

Anchoring mid-drive would not help, and the data already says so: the
snapshots that landed off-anchor are *less* live than the ones on a
drive's opening 1st and 10 (spread 1.5% against 9.3%, total 2.5% against
5.9%). Liveness tracks the market and the match clock, not where in a
drive the snapshot sits.

## Checking every drive: `drive-audit`

```
py -m eAMFCalibrator drive-audit --since 2026-09-17 --until 2026-09-24
```

Goes through every drive the play feed yields in the window and writes
`out/drive_audit.csv`, one row per drive with its flags, plus a summary:

| Flag | Meaning |
|---|---|
| `extra_point_alone` | the drive's only points are a PAT or a two: the conversion was split off its touchdown |
| `odd_points` | points that are no drive's score |
| `fragment` | one row, no points, not the match's last drive |
| `same_side_next` | the next drive is the same side's, with no score or turnover between: one possession cut in two |
| `no_first_down_start` | the first row is not 1st and 10 |
| `defence_scored` | the other side scored while this side had the ball |
| `scouting_disagrees` | SCOUTING_FULL ends the drive differently (a touchdown, field goal, punt, turnover on downs, turnover or the end of a half) |

A match whose drives' points don't add up to the feed's total is counted
too. `--no-scouting` skips the SCOUTING_FULL cross-check, and `--limit N`
takes the N most recent matches.

**The `extra_point` drives.** A touchdown's conversion rows were dropped
only when they sat on the 85 or 98 within 10 messages of the touchdown. A
PAT retaken after a penalty, or one the feed posted late, stayed in the
feed as a "drive" of its own that owned the extra point. Now every row
between a touchdown and its conversion's own score row is the conversion,
so the point stays with the touchdown.

## Inspecting drive detection: `dump`

There is no separate reconciliation script — `analysis/` has three
possession scripts that print to stdout, and none of them runs the code
the snapshots actually come from. `dump` does:

```
py -m eAMFCalibrator dump --match AF063170926
py -m eAMFCalibrator dump --matches 5
```

**Two** CSVs in the output directory, joinable on
`(match_code, event_message_count)`:

| File | One row per | What it carries |
|---|---|---|
| `dump_play_by_play.csv` | message | the play, the cleaning verdict, the drive, the score, and both streams' prices on all six selections |
| `dump_pairs.csv` | **pair** | `directional_pairs.csv` for these matches, widened with everything the play-by-play knows |

It used to be five files. Reconciling a drive meant opening a play row,
carrying its message number to the scores file, then the drives file,
then the quotes file, and holding four tabs in your head to answer one
question. The play-by-play now carries all of it on the row.

### `dump_play_by_play.csv`

| Columns | What they tell you |
|---|---|
| `period_number`, `offensive_team`, `down_number`, `distance`, `field_position` | the play as the feed sent it |
| `is_snap`, `cleaning`, `dropped` | whether it was a scrimmage play at all, which rule fired on it, and whether that rule dropped it |
| `drive_number`, `is_anchor` | the drive it landed in, and whether it became that drive's snapshot |
| `score_p1`, `score_p2`, `score_diff` | the scoreboard as of this message |
| `p1_change`, `p2_change` | the points scored **on** this message, blank otherwise |
| `ml_home_*`, `ml_away_*`, `sp_home_*`, `sp_away_*`, `tot_over_*`, `tot_under_*` | per selection: `_prod` and `_cand` probability, `_line_prod`/`_line_cand` where the market has a line, and `_live` |

`_live` reads `live` (both tradeable), `prod` or `cand` (that side was
not), `both` (neither), `missing` (only one stream quoted it), or blank
(neither did).

A score landing on a message with **no** play row still gets a row, with
`cleaning = score`. Those are the boundaries the feed never labelled, so
leaving them out of a table meant to explain the drives would hide the
thing being looked for.

`cleaning` names the rule rather than just the outcome — `kept`,
`drive_start`, `kickoff`, `special_teams`, `duplicate`,
`stale_after_change`, `impossible_down`, `score` — because reconciling by
eye means seeing *why* a row went, not just that it did.
`classify_plays` is the single source of truth for it, and `clean_plays`
is built from it, so the dump and the pipeline cannot disagree.

### `dump_pairs.csv`

Every column `directional_pairs.csv` has, in the same order, plus the
context that otherwise needs a four-way join:

| Added | What it tells you |
|---|---|
| `market`, `selection` | the market by name, not just its ID |
| `prod_decimal`, `candidate_decimal` | the prices |
| `decisive_winner`, `decided_by`, `basis` | which reading settled the pair |
| `prod_state`, `candidate_state`, `live` | `STATUS`/`IS_ACTIVE` verbatim, and which side was untradeable |
| `anchor_kind`, `anchor_cleaning` | where in its drive the snapshot landed, and which cleaning rule vouched for that row |
| `drive_n_plays`, `drive_dropped_inside` | the drive it came from, and how much noise sat inside it |
| `prod_rows_at_message`, `candidate_rows_at_message` | how many rows that message offered for this market — above 1 is the case that cost the spread and total their pairs |
| `score_bucket`, `time_bucket`, `possession_bucket` | the cells it lands in |

The shared columns come from `report.pair_row`, which
`write_pairs_csv` also uses, so the two files cannot describe the same
pair differently.

### Do the quotes interleave with the plays?

They share **one** `EVENT_MESSAGE_COUNT` sequence — that is what makes
pairing possible — but they are not one to one. A message can carry six
markets across two streams and no play at all; another can carry a play
and no quote. In the play-by-play that reads as a row with a `cleaning`
verdict and blank market cells, or the reverse.

The `_live` columns at the anchor rows are the ceiling on pairs before
any tolerance or line rule applies: a snapshot can only pair on a
selection **both** streams had live at that message.

**Where to look first:** a run of `dropped` rows sitting between two
kept rows with the **same** `drive_number`. A drive is a maximal run of
one offensive team, so a possession change the feed never labelled is
invisible — except as cleaning noise sitting inside a drive that should
have been two. A `drive_number` that spans an implausible number of rows
is the same signal from the other end, and the console summary prints
both.

### Points end a drive

Team change alone merged possessions. `AF063170926` drive 10 ran from
message 310 to 407 — thirteen plays, twenty-six dropped rows inside it,
the snapshot bucketed at 21-7 while the drive after it opened at 21-22.
The label never left Away across two Away scores, so nothing in the play
feed said the possession had changed.

Points are the one boundary the feed cannot argue with: nobody is on the
same drive either side of a score. A drive now also ends at any scoring
message, with the break falling **after** the scoring play rather than on
it — the play that scored is the end of the old drive, not the start of
the new one.

### Field position does not carry across a kick

The backward test — a fresh 1st-and-10 the ball moved *back* to reach is
not a first down — compares against the last row that survived. Across a
kick there is no such row to compare against. From `AF063170926`:

```
 msg  team   d&d   field   verdict
 352  Away   3&5     91    kept           the drive that scored
 359                       score, 21-15
 360  Away   1&10    35    kickoff        the kick spot
 362  Away   3&5     35    special_teams  the scoring play's d&d, restated
 364  Away   1&10    35    kickoff        the kick spot again
 366  Away   1&10    45    was kickoff <- the drive
 372  Away   1&10    63    was kickoff
 377  Away   1&10    80    was kickoff
 384  Away   2&4     87    kept           the snapshot landed here
```

45, 63 and 80 were each read against the **91** on the far side of the
kick, so all three "went backward" and the whole drive went with them.
The snapshot fell on 2nd-and-4 at the 87, six plays late.

A kick resets field position, so once a row has been dropped as one the
backward test stops applying. What tells the kick spot from the drive
after that is the **yard line**: 364 has not left the spot the kick was
taken from, 366 has.

## Cleaning the play feed

From `AF063170926`, twenty-five raw rows covering one drive, a touchdown
and the kickoff after it. Every row arrives **twice**, and the kick spot
arrives as a full `1&10`:

```
 msg  team   d&d   field   verdict
   6  Home   1&10    35    kickoff             the kick spot
   9  Home   1&10    35    duplicate
  10  Home   1&10    26    drive_start   <--   the drive
  16  Home   2&11    26    kept
  20  Home   1&10    37    kept                +11, a first down earned
  24  Home   1&10    50    kept                +13
  28  Home   1&10    85    kept                +35
  32  Home    2&7    88    kept
  36  Home    3&5    90    kept                the touchdown
  42  Home    3&5    85    special_teams       the extra point, from the 85
  47  Home   1&10    35    kickoff             90 -> 35: backwards
  52  Away   1&10    35    stale_after_change  Away's label on that row
  54  Away   1&10    30    drive_start   <--   the drive
  60  Away   2&11    30    kept
```

Five rules, applied in one forward pass over the deduplicated rows, each
judged against the last row that survived:

| Verdict | Rule |
|---|---|
| `duplicate` | An exact republish of the row before it. The feed repeats a play's state until it changes, so almost every row arrives twice — 25 rows here collapse to 13. |
| `stale_after_change` | The team label changed onto the previous row's down, distance and field, unchanged. |
| `special_teams` | Down and distance unchanged while the ball moved. A scrimmage play always changes one or the other, so this is a PAT or a kick — the extra point taken from the 85 with the touchdown's 3rd-and-5 still on it. |
| `kickoff` | A `1&10` that cannot be a snap: either the ball went **backwards** to reach it, which no first down does, or the very next row is the same team's `1&10` again without the ten yards that would earn it. Also any row with no readable down. |
| `impossible_down` | The down skipped ahead by two or more while neither the distance nor the ball moved. No play does that. |
| `drive_start` | A fresh `1&10` whose predecessor was a different team, or was dropped as a kick. |

### Why not simpler tests

**Down and distance alone cannot find the kick spot** — it arrives as a
genuine-looking `1&10`.

**Field position cannot either.** The kick spot is a fixed yard line, but a
return can legitimately finish on it, so `field == 35` both misses real
drives and invents others. There is a test for that case.

**Two `1&10`s in a row are not automatically wrong.** `37 -> 50 -> 85` is
three consecutive first downs, each earned. What marks the kick is the ten
yards *not* being there.

**The stale row compares against the row immediately before it**, not the
last surviving one. It rides the kick spot as readily as a real play, and
at message 52 the row it mirrors had itself just been dropped.

### Half time is the hard one

`AF063170926`, messages 193–197:

```
 193  P3  Away  1&10 @35   the kick spot
 196  P3  Away  3&10 @35   down 1 -> 3, distance and ball unmoved
 197  P3  Away  1&10 @25   the drive
```

Message 196 sits *between* the kick spot and the drive it produced. While
it is in the way, 193 looks at its neighbour, sees no second `1&10`, and
is read as a drive — and 197, arriving on a nearer yard line, is read as
the kick. Exactly inverted.

So impossible downs are dropped in a **first pass**, before anything else
runs. The test is a jump of two or more with neither the distance nor the
field moving: `1&10 → 2&10` on the same yard line is an incomplete pass
and ordinary, while `1&10 → 3&10` is not a play at all. A row genuinely
missing from the feed also shows as a down jump, but the distance moves
with it.

The kick spot is also tested **on both sides of a team change**. At the
opening kick nothing precedes it; at half time the label has just changed.
Testing it only where the team stayed the same missed every second-half
kick.

The scale this was at, before the fix: **58% of all snapshots sat on the
35** — the kick spot — and the first snapshot of every one of 262 matches
was on it. In Q3, 12.3% of snapshots were not even on a `1&10`, against
0.7% in Q1. `dump` now checks for that spike automatically, since drives
start all over the field and a pile-up on one yard line is not football.

The rules read the play feed alone. Drive detection no longer depends on
the score feed, and so no longer rests on the `PLAYER_1 = Home` mapping.

## Where the snapshot lands

A snapshot is meant to be a drive's **opening 1st and 10**. It was taken
from the drive's first surviving row instead, which is not the same thing.

`clean_plays` drops exactly **one** row per team change — the stale
duplicate. Only a TD-anchored transition gets the full walk-forward
cleanup. So a punt, turnover or turnover-on-downs carrying several
kickoff-mechanic rows leaves the rest behind, and the snapshot sat on a
row whose down, distance, field position **and team label** all belong to
the kick rather than to the drive.

The anchor now walks forward to the first row that is a plausible **snap**
(`down_number` in 1–4, `distance` within `MAX_PLAUSIBLE_DISTANCE`), and
then asks whether that snap is 1st and 10:

| Anchor | Meaning |
|---|---|
| `first_down` | The drive's opening 1st and 10, as intended. |
| `mid_drive` | A real snap, but not 1st and 10 — the drive's start was never found. |
| `no_snap` | No plausible snap in the run at all. |

Walking to the first *1st-and-10* instead would be wrong, and a test pins
why: a run that opens 2nd and 7 has already lost its start, and the next
1st and 10 in it is a first-down **conversion** — a real game state, but
not this drive's. Stopping at the first real snap keeps that case visible
as `mid_drive` rather than silently relabelling a mid-drive play as a
drive start.

The **Snapshot anchor** panel reports the split and the share by quarter,
off-anchor rows have their `D&D` cell flagged in the pair table, and a
clean share under 90% stops the Checks line reading "all pass".

## Who is PLAYER_1?

Everything the calibrator buckets on is read in the `PLAYER_1` frame, and
the claim that `PLAYER_1 = Home Team` came from a comment in
`analysis/clean_possession_sequence.py` citing market-description text.
`preflight` now puts that evidence on screen instead:

- **`team_vocabulary`** — what `EVENT.PLAYER_1_TEAM`, `EVENT.PLAYER_2_TEAM`
  and the play feed's `OFFENSIVE_TEAM` actually contain.
- **`market_descriptions`** — one sample per market ID, with how many
  distinct forms exist.
- **`team_join_test`** — whether a match's play-feed team names match its
  `EVENT` team names, i.e. whether the mapping can be *read* per match
  rather than assumed.

What the last full run already settles: `OFFENSIVE_TEAM` is purely
positional — only `Home Team` and `Away Team`, zero unknowns across 15,028
pairs. So if `PLAYER_n_TEAM` holds real team names, the two vocabularies
never meet and the mapping stays positional.

### The drive-count trap

Possession "alternates" between consecutive drives **100% of the time**.
That is not a clean bill of health: a drive is *defined* as a maximal run
of the same offensive team, so it is arithmetic, and it would read 100% on
a feed of pure noise.

The number that means something is **10 drives per match against a
realistic ~22**. The team label changes roughly half as often as
possession actually does, so every missed change silently merges two
possessions into one — and anything reasoning from "the next drive belongs
to the other team" inherits that error. That is a sufficient explanation
for the removed possession check's near-zero agreement without the
`PLAYER_1 = Home` mapping being wrong at all.

## There is no game clock

Worth stating because its absence shapes what the report can answer.
`INPLAY_FIELD_POSITION_PERIOD` has seven columns — `MATCH_CODE`,
`EVENT_MESSAGE_COUNT`, `PERIOD_NUMBER`, `OFFENSIVE_TEAM`, `DOWN_NUMBER`,
`DISTANCE`, `FIELD_POSITION` — and none of them is a clock. Every
timestamp the pipeline can reach (`FILE_TIME`, `PUBLISH_TIME`,
`EVENT_TIME`, `FILE_LOADED`) is an ingest or publish time, not time
remaining.

So "how close to the end was this?" is answered two ways, both proxies and
both labelled as such in the pair table:

- **To end** — seconds from the snapshot to the last quote of its own
  match, off the wall clock. Highlighted under two minutes.
- **Field / D&D** — where the ball is and what it needs. In a one-score
  game these separate a live drive from a dead one far better than the
  quarter does, which is why they are on the row.

Both are on every pair and in `directional_pairs.csv`.

## The snapshot clock

`INPLAY_FIELD_POSITION_PERIOD` carries no timestamp — it has seven columns
(`MATCH_CODE`, `EVENT_MESSAGE_COUNT`, `PERIOD_NUMBER`, `OFFENSIVE_TEAM`,
`DOWN_NUMBER`, `DISTANCE`, `FIELD_POSITION`) and none of them is a clock.
That is why every existing script in `analysis/` orders it by
`EVENT_MESSAGE_COUNT`.

`EVENT_MESSAGE_COUNT` is the same feed sequence the GAMEPLAI streams are
keyed on, and those do carry `PUBLISH_TIME`, so the clock is reconstructed
from the stream (`clock.py`):

| Provenance | Meaning |
| --- | --- |
| `exact` | the stream quoted that same message; its `PUBLISH_TIME` *is* the message's time, nothing estimated |
| `interpolated` | the stream never quoted that message, so the time is placed linearly between the bracketing quoted messages |
| `unresolved` | outside the stream's range, or the bracket is wider than `MAX_BRACKET_MESSAGES` — dropped rather than guessed |

The clock is taken from **one** stream for every run (`CLOCK_SOURCE`,
default `prod`), so prod and candidate are calibrated against an identical
set of snapshot times rather than each stamping the same message a few
hundred milliseconds apart.

A low exact-hit rate would mean the play feed and the stream are not
numbering the same sequence, which would undermine the whole pairing. The
run header calls it out below 50%.

**Sample size is bounded by matches, not rows.** Drives within a match share
a game and are not independent. Every cell reports both counts; read the
match column. `MIN_CELL_OBSERVATIONS` / `MIN_CELL_MATCHES` gate the "worst
cells" summary for the same reason.

## Tests

```
py -m unittest discover eAMFCalibrator
```

301 tests covering line parsing, market resolution, bucket edges, drive
cleaning, clock reconstruction, quote matching, message pairing, the handle
check, the sign test and the paired-delta machinery. No Snowflake needed —
the database half is exercised separately against a mock shaped like the
real schema, play feed with no clock included, and the flipped-match
exclusion runs the real pairing code with the fetch calls patched out.

Two tests pin known traps directly: a candidate much closer on half the
pairs and barely further on the rest sits at a 50% win rate while the paired
loss difference is clearly positive; and grading both streams against one
stream's line inverts the winner on a total that falls between the two
lines.

## SCOUTING_FULL and PLAY_OVER snapshots: `scouting`

```bash
python -m eAMFCalibrator scouting                 # AF, the last 30 days
python -m eAMFCalibrator scouting --days 7 --limit 200
python -m eAMFCalibrator scouting --scouting-table DB.SCHEMA.SCOUTING_FULL
```

`SCOUTING_FULL` is the Madden scouting feed the play table is cut from,
with more in it. It has explicit play bookends (`PLAY_STARTED` /
`PLAY_OVER`), the events inside a play (touchdowns, conversions, kickoffs,
punts, field goals), game status (quarters, `BET_SUSPEND` /
`BET_UNSUSPEND`), and the game clock (`IN_PLAY_CLOCK_SECONDS`, 240 a
quarter, counting down). The column names and vocabulary follow
`MaddenScoutingAudit.py`. Duplicate (match, message) rows are collapsed to
the latest loaded, as that script does. Only match codes starting `AF` are
read. It compares against `GAMEPLAI_STREAM` alone.

It writes three files to `--out`:

| file | what it is |
|---|---|
| `scouting_probe.txt` | the investigation: columns (and any that look like markets or prices), message and status counts, how often GAMEPLAI quotes on the same message as each kind of scouting message (any market / at least one live / all six live), `PLAY_OVER` coverage by market and at nearby offsets, the clock at `PLAY_OVER`, the scouting state against the play table on the same message (team labels, same and mirrored field), and `TEAM_A` / `TEAM_B` against the scoreboard's `PLAYER_1` / `PLAYER_2` |
| `scouting_sample.csv` | every column for the two most recent matches, with prod's line, probability and liveness on all six markets beside each message |
| `scouting_playover.csv` | one row per `PLAY_OVER`; `quoted` is 1 where GAMEPLAI quoted it on at least one market (see below) |

Each `scouting_playover.csv` row carries:
- the clock and quarter, and the betting state;
- the kind of play it closed (scrimmage, touchdown, field goal, punt,
  kickoff, conversion, and so on) and its messages;
- the state on the `PLAY_OVER` row and on the next `PLAY_STARTED`;
- the score then and when the play started, and the final;
- which scoreboard side `TEAM_A` is in that match, read off its scoring
  messages;
- prod's pre-match quotes;
- on each market, prod's line, probability, liveness and outcome.

`PLAY_OVER`s that GAMEPLAI never quoted (about 1.7%) are kept with
`quoted = 0` and no market columns. The play-by-play models read each
play against the one before it, so dropping one would join two plays into
a single wrong one. Everything that scores prices skips them.

It is the input to `python -m eAMFModel v8-build` to `v14-build`, `remaining` and `profiles`.

Each row also carries `timeouts_used_a` / `timeouts_used_b`: how many timeouts
each side had called in the half at that `PLAY_OVER`, off the feed's
`TIMEOUT_CALLED_TEAM_A/B` messages. v8 reads them into the game state, so it
knows each side's timeouts left. Exports made before this have no such columns;
v8 then assumes three each.

## Is SCOUTING_FULL complete? `scouting-check`

```
python -m eAMFCalibrator scouting-check                          # 9 Sep (baseline) to today, break 23 Sep
python -m eAMFCalibrator scouting-check --since 2026-09-01 --until 2026-10-06 --break 2026-09-23
```

Checks every match day from `--since` (default two weeks before `--break`) to `--until` (default
today, UTC days). The expected matches are those EVENT schedules for the sport, with the result off
SCORE_ENDGAME where settled. Each match SCOUTING_FULL has is read whole (a day either side of the
window, deduplicated to the latest load) and judged against a **baseline**: the matches on the days
before `--break`. Matches that started in the last 4 hours are `LIVE` and not judged.

| Flag | Meaning |
|---|---|
| `MISSING` | scheduled and started, but no scouting rows at all |
| `NO_START` / `NO_END` | no `FIRST_QUARTER_STARTED` / no `ENDED` |
| `QUARTERS` | fewer than four quarter starts |
| `GAPS` | holes in `EVENT_MESSAGE_COUNT` between the first and the last |
| `HEAD_MISSING` | the first message count is later than the baseline's usual first |
| `FEW_MESSAGES` / `FEW_PLAYS` | under half the baseline's median messages / `PLAY_OVER`s a match |
| `NO_DETAIL` / `NO_CLOCK` | field position on under half the `PLAY_OVER`s / the clock on under half the messages, where the baseline had them |
| `NO_DOWN` | a usable down and distance (down 1–4, distance above 0) on under half the `PLAY_OVER`s, where the baseline had them |
| `SCORE` | the scoring messages (TD 6, XP 1, two-point 2, FG 3, safety 2) do not add up to the final |
| `NOT_SCHEDULED`, `NO_FINAL`, `DUPLICATES` | information only: not in EVENT, not settled yet, a message loaded more than once |

| File | What |
|---|---|
| `scouting_check.txt` | the summary: the baseline, a line a day (scheduled, settled, in scouting, complete, incomplete, missing, median messages and plays, detail and clock fill, scores agreeing, the problems), the days with a hole since the break, the columns whose daily fill moved 20+ points off the baseline, and the message kinds whose rate a match halved, doubled or appeared |
| `scouting_check_days.csv` | one row a day, with a count of each flag and the latest `FILE_LOADED` (when a backfill landed) |
| `scouting_check_matches.csv` | one row a match, with its flags and the numbers behind them |
| `scouting_check_columns.csv` | every column's fill a day, over all rows and over the `PLAY_OVER` rows |
| `scouting_check_messages.csv` | every message kind's count a day, and per match |
| `scouting_check_detail_days.csv` / `_hours.csv` | down and distance a day / an hour: the messages and `PLAY_OVER`s with a usable down and distance (and with a field position, an offensive team), the matches with any, the first and last such message, the latest `FILE_LOADED` |
| `scouting_check_detail_values.csv` | the commonest down, distance, field position and offensive team values on each day's `PLAY_OVER`s, so a placeholder shows |

The summary ends with the **down & distance** section: a line a day, the last message in the
window with a down and distance (and how many `PLAY_OVER`s have come since without), every hour
where the `PLAY_OVER`s' down-and-distance share jumped more than 30 points from the hour before
(the hour a change landed), and each day's downs. A message kind is only listed as changed when it
was off on 3+ days since the break, each with 5+ of it expected or seen.

Rerun it after a backfill: a day is fully complete when every scheduled match that started is in
scouting with no flag.

### Finals when SCORE_ENDGAME is missing them

From 23 Sep 2026 SCORE_ENDGAME has no final for about half the matches. With
`config.FINALS_FROM_SCORES` (on), a match EVENT has `SETTLED` that SCORE_ENDGAME is missing takes
the last cumulative score in SCORE_CHANGES (`snowflake_io.finals_source`). That covers the history
the pre-match models fit on (`history`, `prior-daily`), the matches a report pairs and the finals
it grades against. `scouting-check`'s **Finals** section shows, a day, how many matches it recovers
and, where both exist, how often SCORE_CHANGES' last score is SCORE_ENDGAME's final -- check that
is ~100% before trusting the recovered ones.

SCORE_CHANGES lost the same matches, so they had no score rows either: no board score in their
snapshots and no TEAM_A side, and a report left them out (`matches_without_score_rows`). With
`config.FINALS_FROM_SCOUTING` (on) a settled match in neither table takes SCOUTING_FULL's scoring
messages added up, TEAM_A as PLAYER_1 (the Finals section's `scouting=endgame` checks the
orientation). With `config.SCORES_FROM_SCOUTING` (on) a match SCORE_CHANGES has no rows for gets
them rebuilt from the same messages (`snowflake_io.scouting_score_rows`): one row on each scoring
message, with its period and the running score, so it is priced in play like any other.

## Halves, half-time margins, touchdown lengths, by gamer and team: `halves`

```powershell
python -m eAMFCalibrator scouting --since 2026-01-01 --no-probe     # every PLAY_OVER, all history
python -m eAMFCalibrator halves                                      # off out/scouting_playover.csv
python -m eAMFCalibrator halves --min-games 50
python -m eAMFCalibrator halves --history out/match_history.csv --min-pair-games 20
```

Off a `scouting` export, no Snowflake. A match's half-time score is the board after its last play
of the first half. A touchdown's length is the yards to the goal line when its play started; kick
and punt returns and turnovers run back are counted apart. Its drive runs from the first play its
side had the ball for (afresh after a score, kick or punt and at half time).

`halves.txt` gives, for the league:
- how often the bigger half holds 60/70/80/90% of a match's points, against the same matches with
  their halves paired at random (what independent halves would give), and the correlation of the
  two halves' points;
- the second half from each half-time margin: points, lead changes (any, 2+), how often the side
  behind at half time won;
- touchdown lengths (mean, 20+ and 40+ yards, the drive's plays and seconds) by half, and in the
  second half by the half-time margin;

then a line for each gamer with `--min-games` or more (default 30): their points a half, their
games' half correlation and lopsided share against chance, their games 3-4 apart at half time (lead
changes, won from up and from down), and their touchdowns' lengths, overall and in the second half
of games 1-4 apart at half time, and their opponents' touchdowns' length (`allowed`).
`halves_matches.csv`, `halves_tds.csv` and `halves_gamers.csv` carry every match, touchdown and
gamer.

With `--history` (default: `match_history.csv` beside the export, when there is one) each side
gets its NFL team (`PLAYER_1_TEAM` is home; a row with the handles the other way round is turned
round), and the same lines follow for each team (`halves_teams.csv`) and each gamer / team pair
with `--min-pair-games` or more (`halves_gamer_teams.csv`). Touchdown length is then fitted as
gamer + team + opponent + opponent's team (least squares over every offensive touchdown), so:
- a team's `td_team_effect` is its yards with the gamers who picked it and whom they played held
  level, `td_allowed_team_effect` the same for the touchdowns scored against it;
- a gamer's `td_gamer_effect` is theirs with the teams they picked held level;
- a pair's `td_expected_yards` is what its gamer, its team and their opponents add up to, and
  `td_vs_expected_z` how far the pair's own mean sits from that. `halves.txt` gives the spread of
  the pairs beyond their gamer plus their team (a chi-squared against noise alone) and the pairs
  furthest each way.

## Kick-offs, the half-time double and prod's line on a kick: `kickoffs`

```powershell
python -m eAMFCalibrator kickoffs                                   # off out/scouting_playover.csv
python -m eAMFCalibrator kickoffs eAMFCalibrator\out_all\scouting_playover.csv --history eAMFCalibrator\out_oct\match_history.csv --out eAMFCalibrator\out_all
```

Off a `scouting` export, no Snowflake. The gamers come from `--history` (default:
`match_history.csv` beside the export), PLAYER_1 as home. It writes `kickoffs.txt` and three CSVs
to `--out`: `kickoffs_kicks.csv` (one row a kick), `kickoffs_matches.csv` (one row a match) and
`kickoffs_gamers.csv` (one row a gamer).

- **How kicks land.** Each kick-off is classed as one of:
  - a touchback: the 20, no clock;
  - no landing zone: the 35, no clock;
  - out of bounds: the 40, no clock;
  - returned, onside, or run back for a touchdown.

  For each class: its share by quarter, the start and seconds, and the receiver's points on the
  drive it starts (to the change of possession or half time). Kicks in Q2's and Q4's last two
  minutes are compared with the rest.
- **Kickers.** Each gamer with `--min-games` kicks gets their kinds, and a chi-squared on them
  against the league's (3 df: about 3 by chance, 11.3 at 1%).
- **The half-time double.** R is the side receiving the second half's kick, the one that kicked
  off the game; K is the other. Per match it counts:
  - R's and K's points in Q2's last two minutes;
  - R's points on its first Q3 drive, and whether R scored both (the double);
  - both sides' Q2 scrimmage snaps (pace) and points.

  It also reads prod's total and spread at Q2's two-minute mark and at the second half's kick
  against the result. Each gamer gets their double rate as R against the league, with a z score.
- **Prod's total on a kick.** Where nothing was scored on the kick, prod's line on the kick is set
  against its quote just before: the conversion, Q2's last quote for the second half, or the
  pre-match quote for the opening kick. The move is shown by who receives (favourite or underdog,
  home or away) and by where the receiver starts. Each line is then checked against the result:
  over at the line before against its price, and over at the kick's line against its price.

On September (2,320 matches, 16,880 kicks):
- **Late in a half, kickers kick touchbacks.** In Q2's and Q4's last two minutes, 13.7% of kicks
  are touchbacks and 5.8% land in no landing zone; elsewhere the figures are 2.8% and 2.4%. A late
  touchback's drive makes 1.28 points against 2.10 after a return. The sim draws every kick from
  one pool, whatever the clock, so it gives late-half receivers too good a start.
- **A few kickers do it far more:** SHROUD into no landing zone 32% of the time; FUSE, MERLIN and
  FENRIR touchbacks 16-17%. The league is 6% and 3.5%.
- **No double effect to speak of.**
  - R scores in Q2's last two minutes 66.7% of the time and on its first Q3 drive 63.6%. It does
    both 43.5% of the time, a little above independence: correlation +0.05.
  - Across gamers the double rate spreads only a little beyond chance (z variance 1.27); LUNA,
    68% over 56 games, stands out.
  - R runs 0.32 more Q2 snaps than K (se 0.13) and scores 0.22 more Q2 points (se 0.14).
- **Prod's total is high at Q2's two-minute mark:** over 46.0% against 49.8% priced.
- **Prod's total moves on a kick by where the receiver starts, not by who receives.**
  - The line moves on 38% of kicks. Favourite or underdog receiving, home or away, the shares are
    the same.
  - A return to the 46 or beyond lifts it 1.0 point on average, up 51% of the time; a start inside
    the 20 lowers it.
  - The moves are right. Where it went up, the game went over the old line 55.5% of the time
    against 50.2% priced.
  - At the opening kick prod's first in-play total sits about 0.3 points under its pre-match line,
    a point down in 34% of matches, whoever receives. The pre-match line runs a little high: over
    47.7% against 50.1% priced.

## Every timeout: `timeouts`

```bash
python -m eAMFCalibrator timeouts --since 2026-08-24
```

The `PLAY_OVER` export keeps only the messages inside a play, so the timeouts,
called between plays, never reached it. This reads them straight off
`SCOUTING_FULL` and writes two files to `--out`:

| file | what it is |
|---|---|
| `timeouts.csv` | one row per `TIMEOUT_CALLED`: quarter and clock, the side that called it, whether it had the ball (`role`: the offense of the next snap), its lead, how many it had called before in the half, and the play before it (`incomplete`, `first down`, `in play`, `change of possession`) |
| `timeouts_summary.txt` | the calls by quarter, clock, role and score, and how many each side had used by 2:00 of Q2 and 3:00, 2:00 and 1:00 of Q4 |

`python -m eAMFModel v8-build ... --timeouts timeouts.csv` fits v8's timeout
calls on it (see eAMFModel's README).

What 15,612 calls over 2,977 matches (24 Aug – 22 Sep) showed:
- Almost all come in the last four minutes of Q2 (7,387) and Q4 (7,942).
  Q1 and Q3 have about 250 between them.
- Q2: the side without the ball calls most of them from 3:00 to 1:00 (67–79%).
  In the last 30 seconds the side with the ball calls 69%, at any score.
- Q4: sides without the ball call them from about 3:00, whether down by one
  score or more. Level sides and sides a score ahead call them too, to keep
  time to answer. Trailing sides with the ball call them from about 1:30,
  most in the last minute.
- Only about 6% follow an incompletion. The rest follow a play in bounds
  (about 45%) or a first down (about 30%).
- A side still has all three at Q2's 2:00 94% of the time, at Q4's 3:00 95%
  of the time, and at Q4's 2:00 85% of the time.

## eAMFModel as the candidate: `--candidate v9`

The simulation versions v8, v9 and v10 stand in for the candidate. (v3–v7 have been removed;
use v8–v10 in their place in the commands that follow.)

```bash
# once: export PLAY_OVER snapshots, the match history and every timeout, and build the model
python -m eAMFCalibrator scouting --since 2026-08-24 --until 2026-09-23
python -m eAMFCalibrator history --until 2026-09-23
python -m eAMFCalibrator timeouts --since 2026-08-24 --until 2026-09-23
python -m eAMFModel v9-build eAMFCalibrator/out/scouting_playover.csv --half all --until 2026-09-23 \
    --out v9_model --history eAMFCalibrator/out/match_history.csv --timeouts eAMFCalibrator/out/timeouts.csv

# then any report, with v9 in the candidate's place
python -m eAMFCalibrator report --candidate v9 --v9-model v9_model
```

A version prices on the game clock, which only `SCOUTING_FULL` has. So for each
match in the window, the calibrator builds the same `PLAY_OVER` snapshots
the `scouting` export writes (`snowflake_io._sim_quotes`), with each match's
player handles for the profiles, and the model prices each one by simulation
(`eAMFModel.v9_stream`).
- **Prices between snapshots:** like any stream, the model's quote stands until
  its next one. At every message and market prod quoted after a
  `PLAY_OVER`, it quotes that snapshot's book.
- **Lines:** it quotes its own line on the spread and the total, so pairs
  split between the same line (compared on probability) and a different line
  (compared on whose line landed nearer the result). `--v9-lines prod` reads its
  price at the line of the exact prod row the pairing uses there: the first live
  row on the message, otherwise the first row (the rule in
  `directional.index_by_message`). Then every pair is on the same line, including
  on messages where prod moved its line and carries both the old and the new one.
- **Whole matches:** it reads every `SCOUTING_FULL` row of each match,
  including rows before the window opened, so a match already under way
  at the window start still knows who received the opening kickoff.
- **Liveness:** its quotes are always live. The pairing's own liveness
  rule decides what counts, and prod's suspensions still apply to prod's
  side.

`history` writes `out/match_history.csv`, shaped like `nb2/AMFELO.csv`: every
settled match of the sport before `--until`, with players, teams, stream
and finals, off `EVENT` and `SCORE_ENDGAME`. The model's own pre-match
model (NB2, or glmer with `--prior glmer`) is fitted on it. The model prices
pre-match from that, never from GAMEPLAI, so the report also fetches each priced
match's players, teams and stream from `EVENT`. That needs pandas and scipy
as well as numpy.

**The reports say "v9".** With a model standing in, every "candidate" a
reader sees in the HTML reports (headers, tooltips, verdicts, the title)
becomes its name, and the report is written as `eamf_report_v9.html`, so
it never overwrites a prod-vs-candidate report. `--candidate-label NAME`
picks another name; `--html PATH` another file. The CSVs keep their
column names and file names, so give a model run its own folder with
`--out` if you want to keep both sets.

**The model must be built from matches before the window.** It is fitted
to play-by-play outcomes and final scores, so building it on the matches
being scored makes the comparison in-sample. Build it with `--until` where
the window starts. When the model directory is missing, the run stops and
prints the build commands.

Cost: about a second of simulation per match on each core at 2,000 paths
(`--v9-paths`). The work is spread over all cores but one
(`config.MODEL_WORKERS`).

`--candidate v8`, `--candidate v10`, `--candidate v11`, `--candidate v12`, `--candidate v13` and `--candidate v14` work the same way off their own builds
(`--v8-model`, `--v10-model`). What each version changes is in the eAMFModel README.

## Several candidates in one report: `--candidate v8,v9`

```bash
python -m eAMFCalibrator report --candidate v8,v9 --v8-model v8_model --v9-model v9_model --since 2026-09-17 --until 2026-09-23 --out eAMFCalibrator/out_v8_v9
```

Name any number of candidates, comma-separated, and the report sets each one
beside prod: every table reads real, prod, then a group of columns per
candidate. The file is `eamf_report_v8_v9.html`. To move on to a new
version, change the names on the command line.

- **One population.** Every candidate is paired with prod in its own pass,
  then all of them are cut to the snapshots every candidate paired. So
  prod's figures and the real outcomes are the same beside every
  candidate. The Run table counts what was dropped for want of one.
- **Two readings of a model.** A model that quotes its own lines (v8 to v10)
  is read at prod's line for the calibration tables, where its probability
  answers prod's question. It's read at its own line for the line tables,
  where the question is whose line landed nearer the result. One simulation
  gives both, so the second costs a pairing pass and no simulation, and
  repeated Snowflake queries are answered from memory.
- **Sections.**
  - Directional calibration: Brier at prod's line, line error at each
    side's own line, how often the lines matched, and the match votes.
  - By market.
  - Cross-section calibration.
  - Score difference, Quarter and Possession, each its own section with the
    calibration at prod's line and each side's line error.
  - Pre-match.
  - In-drive.
  - **Line moves** (below).
  - Additional checks.
  - Every pair.
- **Every pair** carries prod's line, probability, result and error, and
  for each candidate its own line and probability, its probability at
  prod's line, the difference from prod, its result, its error and who
  was closer. The rows are drawn in the browser from the embedded data:
  the first 2,000 of whatever is sorted or filtered are shown, and the
  match filter searches all of them.

With one candidate the page is the same with one group of columns.

### Two builds of the same version: `v9,v9-glmer=DIR`

```bash
python -m eAMFCalibrator report --since 2026-09-17 --until 2026-09-23 --snapshots play_over --candidate v9,v9-glmer=v9_glmer_917 --v9-model v9_model_917 --bets
```

- `v9` is priced off `--v9-model` as usual.
- `v9-glmer=v9_glmer_917` is a second candidate. It runs v9's code off the
  build in `v9_glmer_917`, and its columns are headed `v9-glmer`.
- The name is the version, a dash, then any tag (`v9-glmer`, `v9-old`).
  Paths and lines follow the version (`--v9-paths`, `--v9-lines`).
- Use this to compare two builds that differ only in what they were built
  with, e.g. `--prior nb2` against `--prior glmer`.

### Line moves

This replaced "Totals line within 1 and 2 scores". It asks whether a stream reacts too far to a
score. At each snapshot (`line_moves.py`), it sets each stream's total and spread against that
stream's previous quote in the match. The line went up, went down, or held. Then it counts how
often the game finished over the line: for the spread, how often the home side beat the home
margin the line asked for. The page shows the numbers only; this is what they mean.

- **Each cell** shows three things:
  - that share;
  - the share less the stream's own priced chance at its line, in points, coloured on the
    probability-gap ramp;
  - the count.
- **What a fair line looks like.** It sits at its price: about 50-50 in the middle of a game, and
  less evenly late on, when the points still to come are lumpy (0, 3 or 7).
- **What over-reacting looks like.** A minus after the line goes up means games fall short of a
  line lifted too far. A plus after it goes down means a total dropped too fast as the clock ran.
- **Totals and Spread** each have two tables:
  - one row per stream, with columns for the move: up (after a score, no score), down (the same),
    held, all. For the spread, up is toward the home side.
  - quarter by score difference (level 0-2, one score 3-8, two or more 9+), with columns for line
    up, line down and all. Each quarter's all row is shaded above its states.
- **First touchdown** takes every quote from each match's first touchdown until the next score
  (its conversion stays in). Two splits:
  - when it came: the game's first minute, 1:00-2:00, the rest of Q1, Q2, the second half;
  - who scored it: the favourite or the underdog by prod's first moneyline quote, or pick'em
    within 5 points of 50%.
  Over is the total; Scorer covers is the spread read from the scoring side. A minus under Over
  after an early touchdown means the total went up too far. The clock comes from the PLAY_OVER
  snapshots (`--snapshots play_over`); other runs show "clock unknown" for Q1.
- Pushes are left out. A spread quote on the away side is read as the home margin it implies.

On August-September v12 against prod, after a favourite's touchdown in the game's first minute
(60 matches), totals finished 13 points under prod's price and 12 under v12's. The favourite then
covered 9 points over prod's price and 13 over v12's. Each cell holds a few hundred quotes from a
few dozen matches, so read a month or more.

On the month of v12 against prod (23 Aug - 23 Sep), prod's spread went toward the side that had
just scored 9,723 times. The home side then covered 39% of the time, priced 50%. After a score
moved prod's line the other way, they covered 62%, priced 50%. Prod's total, held or lifted with
no score, finished over 44%, priced 49%. v12 sat within about 2 points of its price in every row.

## The totals lines, value by value: `totals-lines`

```bash
python -m eAMFCalibrator report --since 2026-09-17 --until 2026-09-23 --snapshots play_over --candidate GAMEPLAI_STREAM_CANDIDATE,v8,v9 --v8-model v8_model --v9-model v9_model
python -m eAMFCalibrator totals-lines                      # every out/directional_pairs*.csv
python -m eAMFCalibrator totals-lines out/directional_pairs_v9.csv
```

This reads the report's `directional_pairs*.csv` (no Snowflake, seconds to
run). At a snapshot with some points on the board, a total line asks for
N = line − on board more. What the rest of the game produces, R, is lumpy
(0, 3, 7, 10, 14 ...). So two lines a point apart can sit either side of a
value R often takes: prod needs 6.5 (a touchdown goes over), the candidate
7.5 (a touchdown and its kick stays under). For each candidate against prod,
with both quotes live (`--all-quotes` keeps the rest):
- **gaps:** the candidate's N minus prod's, overall and by quarter;
- **straddles:** the lines grouped by the whole values between them (7;
  8,9; 14 ...). For each group it gives prices, real over rates and each
  stream's Brier at its own line. It also gives how often the game landed
  exactly on a value between the lines, where the two settle differently,
  and each stream's Brier there;
- **by N:** at each points-needed value, each stream's number of lines,
  mean P(over) and real over rate, and the gap between them;
- **real:** the points the rest of the game made, share at each value, by
  quarter and game state, beside the median N of each stream;
- **one score:** each line's reach (within one score, two, beyond;
  `totals_reach`) against the other's, and where the game really landed.

`out/totals_lines_<name>.csv` has one row per snapshot: the score, one
score's worth, each N, the straddled values, R and each P(over).

## Betting simulation: `bets`

What would the book's margin have been had the candidate's prices been
used instead of prod's? `bets` runs four steps: bets, then latency, then the
join, then the analysis.

```bash
python -m eAMFCalibrator bets probe --since 2026-09-18 --until 2026-09-25
python -m eAMFCalibrator bets --since 2026-09-18 --until 2026-09-25
python -m eAMFCalibrator bets --since 2026-09-17 --until 2026-09-23 --candidate v9 --v9-model v9_model
```

- **Bets.** Every single bet on an AF moneyline, handicap or total in the
  window, pre-match and in play, from the `CUSTOMER_REVENUE` view (bet by
  bet: bet time, odds, line, operator, customer).
  `SHARED.CUSTOMER_REVENUE_EVENT` is one row per operator, match and day,
  so it can't be used. The operator, `BET_IN_PLAY`, customer hash,
  `CUSTOMER_TEMPERATURE` (Standard / VIP / Restricted; Hard Rock sends
  none) and cash-out flag come through to the output.
- **Latency, one lag per operator.** It's the lag, −30 to +60s, at which
  the operator's in-play moneyline odds follow prod's probability most
  closely, once its margin is taken out. A negative lag means the
  operator's clock runs behind GAMEPLAI's. The report prints the misfit at
  each lag, so the minimum and its sharpness can be seen. Pre-match bets
  are read at bet time.

  Why not per match: the feeds we can reach have no two-minute
  auto-suspend to measure against, and per-match lags fitted off the lines
  (which agree for only a fifth of bets) made the odds follow prod *worse*
  than no lag at all.
- **The join.** Each bet takes prod's quote as published one lag before it
  (`stream_prob`), and the candidate's quote at the same feed message
  (`candidate_prob`, `candidate_line`), or at the same time when prod's
  quote carries no message. A spread line is read from prod's side. Whether
  an operator records a side's line turned is fitted from the data, per
  operator and side (`bet_line_prod_side`). A bet on another line than
  prod's (an alternate line) can't be re-priced, since prod's price for it
  isn't in the stream.
- **Analysis: the bet placed with the candidate.** It's the same selection
  at the candidate's line and probability. The operator's margin over prod
  (implied / prod) is kept, so its odds are odds × prod / candidate. It's
  settled at the candidate's line against the final score (`SCORE_ENDGAME`):
  over 44.5 and over 45.5 can settle differently on the same game. Where
  the candidate's line is the bet's own (and on moneyline) it keeps the
  operator's result. Cash-outs are left out (see the checks below). A check reads each bet on
  prod's line off the final score and sets it against the operator's
  settlement, by market; that confirms the line conventions.

The report prints:
- the lag by operator, and the odds misfit with and without it;
- the spread and total line checks: the sign fitted per operator and side,
  how often the bet is on prod's line, and the commonest gaps;
- where the money is: bets and stake by pre-match or in play and market,
  and how much of each the simulation re-prices;
- how each re-priced bet settles with prod and with the candidate (won→won,
  won→lost, lost→won, ...), by market, and the change in revenue split into
  the candidate's odds on bets settled the same and bets its line settles
  differently;
- the margin both ways: overall, all but VIPs, and by pre-match or in play,
  operator, customer temperature, market and period.

### A chosen few matches: `bets --only-matches`

```bash
python -m eAMFCalibrator bets --since 2026-10-01 --until 2026-10-12 --only-matches eAMFCalibrator/match_lists/v14_worst15_oct.txt --candidate v14,v14-raw=v14_model_raw --v14-model v14_model --out eAMFCalibrator/out_worst15
```

`--only-matches` takes a file of match codes (one a line, the first word; `#` starts a comment) or
codes comma-separated, and keeps only those matches' bets, so a handful is priced in minutes, in play
included. `match_lists/` holds the lists: `v14_worst15_oct.txt`, the 15 matches where v14's re-priced
bets fell furthest short of its markup over 1-11 Oct (96% of its shortfall on 4.5% of the stake), and
`v14_control15_oct.txt`, 15 others with as much money on them, drawn at random. A list chosen for
losing flatters any model that disagrees with the one it was chosen on, so read the two together.
`eAMFModel v14-priors` lays the same matches out by each prior (see its README).

### One gamer's night: `bets --gamer`

```bash
python -m eAMFCalibrator bets --since 2026-10-06 --until 2026-10-08 --gamer NIGHTMARE --candidate v10,v10-glmer=v10_glmer_1006 --v10-model v10_nb2_1006 --out eAMFCalibrator/out_nightmare
```

`--gamer HANDLE` (repeatable) keeps only the matches that gamer played. Handles are matched in
capitals, against `EVENT`. Only those matches are priced, so one gamer's night is quick, in play
included. `bets prematch --gamer` works the same on the pre-match bets.

The usual report is printed for those bets. On top of it, `bets_gamers.txt` takes each of the
gamer's matches in kick-off order and shows:
- **The closing pre-match prices**, prod against each candidate: his moneyline, the margin his
  spread needs him to beat with its chance, and the total. The closing price is the last live quote
  before the first play.
- **The result**, his score first.
- **The bets on the match.** Book revenue as placed (prod), and re-priced by each candidate on the
  bets every candidate re-priced.

A summary line gives his record for the night, his margin against prod's closing spread lines, and
the night's book revenue each way.

### Punters piling onto a trend: `bets_trends.txt`

Every `bets` run (and `bets prematch`) breaks the moneyline and spread bets down by how each one
stood to a gamer's run this session when it was placed:
- **Piles on** backs a gamer who had won the last 2+ (`TREND_RUN`), or opposes one who had lost
  them. **Fades** goes the other way. **Mixed** does both, and **none** has no run of 2+ on either
  side.
- **A session** is a run of kick-offs no more than 2 hours apart. **A result counts** once it's in,
  36 minutes after kick-off, at the bet's own time. A bet placed before the last match finished
  doesn't see it.
- For each group, by run length (2, 3, 4+) and pre-match apart from in play, it shows the bets, the
  stake, and the book's margin (revenue / stake). The margin is shown as placed with prod and
  re-priced by each candidate, on the bets every candidate re-priced.

If punters make money by piling onto trends that carry on, the "piles on" rows show a lower margin
than "none". A model that moves on trends (`eAMFModel form-layer --trigger`) should then raise it.

### Margin uplift, the partner's way: `margin_uplift.txt`

Every `bets` run (and so `report --bets`) also runs the dummy-bet method from the partner's
eBasketball notebook. It's applied to AF as they applied it, weaknesses and all, but as a direct
comparison: prod against each candidate over the report window, on matched ticks only (below). It
reuses the quotes the simulation has already fetched, so it costs nothing
extra.

```
python -m eAMFCalibrator report --since 2026-10-01 --until 2026-10-12 --candidate v13 --v13-model v13_model --bets --margin-chart FD_109_BALANCED.csv
```

- **Dummy bets.** Every live in-play tick on the moneyline (50/51, home is their side A) and
  totals (54/55) is a bet to win 100. It's priced at the margin chart's price for the probability
  rounded to 3 places: their `FD_109_BALANCED.csv`, probability then `numerator`, `denominator`,
  so the price is numerator / denominator + 1. The file can have a header naming those columns, or,
  as theirs does, no header with the three in that order. The chart is read when the run starts, so
  a chart it can't use stops the run at once. Without `--margin-chart`, a proportional 109% book
  stands in.
- **Per bet.**
  - Stake = 100 / (price − 1).
  - Expected margin = 1 − price × probability.
  - GGR = stake − price × stake if it won, else the stake. A moneyline tie loses both sides, and
    a totals push loses both.
- **The clock.** Minutes run from the scheduled start, scaled so prod's matches end at 60 game
  minutes on average (their 0.146 cut made basketball's 20). The same factor applies to every
  feed (`--uplift-factor` sets it). The money leaves out the last 6 minutes (their last 2 of 20),
  and anything with a fair price of 1.001 or less.
- **Headline.** Margin % (GGR / handle) and expected margin % for each feed and market, and the
  uplift: each candidate's margin less prod's.
- **Calibration.** Their weighted MAE: |mean probability − win rate| for each game minute left,
  selection and band, weighted by ticks.
  - Moneyline bands are tenths of probability.
  - Totals bands are AF's (≤35.5, 36–40.5, 41–45.5, 46–50.5, 51–55.5, 56+). Theirs were
    basketball's.
- **Checkpoints.** At 54, 45, 36, 30, 21 and 15 minutes left (their 18, 15, 12, 10, 7 and 5),
  each band's calibration and margins are shown. A band's calibration needs 25 ticks in every
  feed; its margin needs more than 50.
- **Matched ticks only.** The feeds tick at different rates, so every table bets at prod's ticks
  alone. At each one, each candidate's latest live price for the same match, market and selection
  is taken (no more than `PAIR_MAX_AGE`, 2 minutes, old). A prod tick without one in every
  candidate is dropped from every feed, and so is one where any feed's probability is outside
  `PROB_RANGE` (0.05 to 0.95). The moments and bets are the same for all, only the probabilities
  differ, so the uplift is the model's alone. Each feed's matched ticks are shown against all its
  ticks. The clock factor is still fitted on all of prod's ticks. The matches are the ones
  the simulation covers: those with at least one bet.

The page's betting simulation has three panels:
- **Re-pricing:** the real bets at each candidate's probability.
- **Book comparison.**
- **Margin uplift.**

`margin_uplift.csv` holds the sums behind every table: one row per feed, market, minute,
selection and band.

- **Brier and log loss, same ticks.** The headline also scores every feed on exactly the matched
  ticks with more than 6 game minutes left, every tick alike (lower is better). It's a proper
  score: it's best when the probabilities are right. The dummy-bet margin isn't:
  - A bet to win 100 at a short price carries a big stake. At a 0.92 favourite, the 1.01 floor of
    the proportional book makes it a stake of 10,000.
  - So a book keeps a lot when it was sure of a side that lost, and the margin rewards confidence.
  - At a true 75% for one side, pricing it 80% scores about 12.9%, pricing it 75% scores 8.3%, and
    pricing it 70% scores 4.6%.
  - Read the uplift next to the Brier score: a candidate behind on the uplift but ahead on Brier
    is the less confident book, not the worse one.
- **By match: `margin_uplift_matches.txt` and `.csv`.**
  - **Contents.** Every match with its start, players, final score and dummy bets (ticks) per
    market. Each feed has its share of the handle (wt), GGR, margin, Brier and "sure": the highest
    probability it gave a selection that lost.
  - **Part.** Each candidate's part of the uplift: the match's GGR over the candidate's whole
    handle, less prod's over prod's. The parts add up to the uplift exactly.
  - **Order.** Matches are sorted so the ones that cost the candidate most come first, with a
    running sum.
  - **Summary.** `margin_uplift.txt` (and the page's "By match" fold) sums it up for each market:
    - the 10 matches that cost the candidate most and the 10 that gain it most;
    - the uplift without them;
    - how many matches hold half of prod's handle.

### The operators' margin

Every `bets` run prints each operator's margin over prod's probability: the median of
implied / prod − 1 on the bets at prod's line, by market, pre-match and in play. The models' own
quotes carry no margin. A re-priced bet keeps the operator's margin over the model's probability
(its odds are odds × prod / candidate), so a candidate is priced with exactly the margin the
operator charged on that bet.

### The pre-match models' own test: `bets prematch`

A pre-match model (NB2 or glmer) is judged on the bets placed before
kick-off. Those bets don't depend on the in-play simulation, so this test
prices each model version off its kick-off alone. That makes a window of
many weeks cheap enough to run, and a week of bets says little.

```bash
# two v10 builds on the same cut-off, differing only in their prior
python -m eAMFModel v10-build eAMFCalibrator/out/scouting_playover.csv --half all --until 2026-08-01 --out v10_nb2_0801 --history eAMFCalibrator/out/match_history.csv --handles eAMFCalibrator/out/match_history.csv
python -m eAMFModel v10-build eAMFCalibrator/out/scouting_playover.csv --half all --until 2026-08-01 --out v10_glmer_0801 --history eAMFCalibrator/out/match_history.csv --handles eAMFCalibrator/out/match_history.csv --prior glmer
# every pre-match bet since then, re-priced by each
python -m eAMFCalibrator bets prematch --since 2026-08-01 --until 2026-09-30 --candidate v10,v10-glmer=v10_glmer_0801 --v10-model v10_nb2_0801
```

- **The bets.** Only bets the operator flags as not in play
  (`BET_IN_PLAY`). They're joined and re-priced as in `bets`: each prod
  quote published before the first play gets the model's kick-off price at
  the same publish time.
- **No look-ahead.** glmer's form follows the results.
  - The run fetches every match settled by `--until`.
  - Each pre-match quote reads only the results in when it was published:
    a match counts from its last play in the feed, else 36 minutes after
    kick-off. See eAMFModel's README, "Form follows results, as it would
    live".
  - NB2's ratings stay as they were at its build, as they do live.
- **Output.**
  - `bets_prematch_sim.csv` has one row per bet, with each candidate's
    re-priced revenue.
  - `bets_prematch.txt` is the report, also printed:
    - the book's margin with prod, and each candidate's change against it,
      in points: overall, by market and week by week (the week the bet was
      placed);
    - then each candidate against the first: the change in margin, its 95%
      interval over `--boot` resamples of the matches, and in how many
      weeks it came out ahead.
- **Builds age.** Both builds are fitted once, on the same cut-off, so
  both age over a long window. NB2's ratings and glmer's player effects
  stay fixed, while glmer's form keeps up. To test how they'd do rebuilt
  every week or two, run the window in pieces, with fresh builds before
  each piece.

### The comparison tables

Every comparison table reads the same way. It opens with N, then grouped columns:
- **Mean:** Real, Prod, then each stream (Cand, v6, v7 ...);
- **Gap:** Prod, then each stream (real minus the stream's mean);
- **Brier:** Prod, then each stream, green where the stream beats prod.

Line-error tables use Line error (Prod, then each stream) and Same line as prod.
The top table puts both readings side by side, all markets then each market: Brier
at prod's line, and line error at each stream's own line. Pre-match reads a model
at prod's closing line (its prod-line pass), so every match pairs, not only those
where its own line happened to be prod's. The page no longer carries the pair by
pair table; the pairs CSVs still have every pair. The betting simulation is added
at the bottom with `report --bets`.
There are no confidence intervals or p-values. `GAMEPLAI_STREAM_CANDIDATE` is
shown as Cand.

Each stream is scored on the rows it can be compared on:
- its own live rows;
- at prod's line, only where its line is prod's.

The GAMEPLAI candidate quotes prod's line on about 56% of pairs, while a model
is read at prod's line on all of them. So streams can sit on different rows,
with a different N, real rate and prod figure. Streams whose N, Real and Prod
agree on every row of a table share one N, Real and Prod column. Where they
differ, those columns repeat once per set of streams that shares them, labelled
with the streams (for example "Real · Cand" and "Real · v6 v7"). Every gap sits
beside the real rate it was taken from.

### In the calibration report: `report --bets`

```bash
python -m eAMFCalibrator report --since 2026-09-10 --until 2026-09-23 --snapshots play_over --candidate GAMEPLAI_STREAM_CANDIDATE,v9 --v9-model v9_model --bets
```

`--bets` runs the betting simulation for the same window and candidates, and
adds a "Betting simulation" section at the bottom of the report's HTML. The
section has:
- each candidate on the bets it re-priced: prod's margin, the candidate's,
  the change, and the change split into odds and line;
- every candidate side by side on the bets they all re-priced, by the same
  cuts as the console;
- why bets were not re-priced;
- the lag by operator.

The CSVs (`bets_sim.csv`, `bets_latency.csv`) are written to the report's
`--out` folder. Models are priced again for the bets' matches, so the run
takes longer.

### Book comparison: who would still bet, and which book they would take

Re-pricing the same bets assumes every bet would still be placed, at the same stake, whatever the
candidate quoted. But bettors chose those bets against prod's price. So every `bets` run, and
`report --bets`, also sets each candidate against prod as a book. The tables are written to
`out/bets_book_comparison.txt` and shown on the report page under **Book comparison**. Each
candidate keeps the operator's margin over its own price, bet by bet, as the same-bets re-sim
does, so every reading is a like-for-like comparison with prod.

- **Would they still bet?**
  - Sharp accounts (`--sharp`, Restricted by default) bet for an edge. Each is given its
    segment's measured edge over prod's price, by market: the bettors' return on those bets,
    shrunk toward the segment's own over 200 bets.
  - At the candidate's odds the edge left is belief × candidate odds − 1, with belief = (1 +
    edge) / odds. With none left, the bet is not placed. Otherwise the stake moves with the edge,
    capped at 3× (`--max-scale`).
  - Everyone else (VIP, Standard, untagged) has little edge and doesn't shop on price, so the bet
    stands at stake × (candidate odds / odds) ^ `--elasticity` (default 0: as placed).
  - Kept is the stake the candidate would have taken against the real stake.
- **Two books.**
  - Prod and the candidate both quote. A bettor takes whichever pays more on the bet's
    selection, where both quote the same line: the moneyline always, and a line market only where
    the candidate's line is the bet's (run a candidate `@prod` to compare every line market).
  - Each book is scored on what it is left with. Candidate − prod is the head-to-head: positive
    where the candidate's prices hold up better where the two differ.
  - Both − alone is the two books together against prod taking every bet alone. Any second book
    lowers it, as the bettor always gets the better price.
- **Intervals.** Each margin, and each change, comes with a 95% bootstrap interval over matches
  (`--boot` resamples, 1000 by default). A change is coloured only where its whole interval is one
  side of zero.
- **On the page** (headed Acceptance and Two books): each margin is green where the book is ahead,
  red behind; its interval sits underneath.

```bash
python -m eAMFCalibrator bets --since 2026-09-17 --until 2026-09-23 --candidate v9,v10 --v9-model v9_model --v10-model v10_model
python -m eAMFCalibrator bets ... --sharp Restricted,VIP --elasticity 1    # other assumptions
```

### The lag, match by match: `bets lag`

```bash
python -m eAMFCalibrator bets lag --since 2026-09-17 --until 2026-09-24
```

This checks the bet-to-price matching for every match of the window
(`--until` is a timestamp, so `2026-09-24` takes in all of the 23rd). It
reads each operator's lag behind prod two ways, match by match.
- **Lines.** A spread or total bet sits on the line the operator showed,
  which is prod's live line one lag earlier. So each bet allows the lags at
  which prod's line equals the bet's. A bet struck near a line move pins the
  lag: on the old line after prod moved means the lag is *at least* the
  time since the move; on the new line soon after means *at most*. Per
  match: the share of line bets on prod's line at each lag, the best lag,
  and whether one lag fits all its pinning bets (the largest "at least"
  under the smallest "at most").
- **Odds.** Each in-play bet's odds against prod's probability a lag
  earlier, with the operator's margin taken out: the lag with the smallest
  misfit.
- **Held out.** A lag per match is fitted on alternate bets and scored on
  the others, against the operator's one lag on the same bets. It shows
  whether a lag per match beats one per operator on bets it wasn't fitted
  to.
- **By period.** Whether the lag drifts through a match.

It writes:
- `bets_lag.csv`: one row per match and operator;
- `bets_lag.html`: the summary and a table linking every match;
- `bets_lag_<day>.html`: every match drawn, with prod's line as a step and
  each bet a dot. Green means on prod's line at the operator's lag, orange
  only at another lag, red at none. Above each match, its lag curves.

The per-match lags fitted off the lines were dropped once, when only a
fifth of spread bets agreed with prod. That was before the spread line's
sign was fitted per operator; now 99.8% of bets sit on prod's line, so the
lines can be read again.

### Where in the game the book loses

Every run (`bets`, `bets check`, `report --bets`) tags each in-play bet
with what the scouting feed was doing when the bet was struck. It then
buckets the book's margin by each tag, on the console and in the HTML
section. The same tags are added as columns to `bets_sim.csv`.

| column | what it is |
|---|---|
| `moment` | the feed at bet time: play live, between plays, score in with the play not over, conversion to come or live, kick-off to come or live, timeout, quarter break |
| `feed_suspended` | the feed's own BET_SUSPEND .. BET_UNSUSPEND was on (it suspends for every play) |
| `moved` | what the feed did between the message the price came from and bet time: score, kick-off, play started, play over, timeout, possession, nothing |
| `price_age` | seconds from the price's message to the bet |
| `next_score` | seconds from the bet to the board's next move |
| `clock_band`, `score_margin`, `down` | quarter and clock, the score, the down |

On the page (headed In play), each margin is red where it sits more than two standard errors below
what prod's own probability says the book keeps, green more than two above. Revenue is green or red
by its sign. The candidate columns are the change in margin on the bets every candidate re-priced.

Bet time is read on the feed's own clock. Prod only publishes on some
messages; the feed carries every message's `FILE_TIME`. Each match's feed
is moved onto prod's publishing clock by the median gap between the two.
Each bucket shows:
- its margin with ±2 standard errors;
- the expected margin: what the book keeps if prod's probability were right
  (1 − odds × prod's probability);
- each candidate's change in margin on the bets they all re-priced.

A bucket well below its expected margin is where bettors know more than the
price. That can be a stale price (the feed moved on) or a bet struck while
a play is live. The list of the costliest moments crosses the feed at bet
time with what moved since the price and the next score.

To bucket a CSV already written again, with no Snowflake:

```bash
python -m eAMFCalibrator bets moments --csv eAMFCalibrator/out/bets_sim.csv
python -m eAMFCalibrator bets moments --by clock_band,market,selection --by score_margin,selection --by day,market
```

`--by` crosses any `bets_sim.csv` columns (or `day`), costliest first, with
the edge bettors had over prod's probability (the expected margin less the
margin).
- `--scope all|pre-match|in-play` picks the bets (default all). `when`
  splits pre-match from in play by quarter and clock.
- `--players CSV` takes the match history (`history`'s, shaped like
  `nb2/AMFELO.csv`; PLAYER_1 is home). It adds columns to cut by:
  - the gamer and NFL team backed and opposed (`backed_player`,
    `opposed_player`, `backed_team`, `opposed_team`; `total` for totals);
  - `matchup`, `team_matchup`, and the home and away gamers and teams;
  - `gamer` and `team`, which count each bet once for each side of its
    match, with `gamer_role` (backed, opposed or total).

```bash
python -m eAMFCalibrator bets moments --players eAMFCalibrator/out/match_history.csv --by when,market,selection --by backed_player --by gamer,gamer_role --by backed_team --by matchup
```

When the CSV carries candidates, each `--by` row also shows each
candidate's change in revenue against prod's on the bets of that bucket it
re-priced, with how many there were: a bucket where prod loses and v6 still loses is
somewhere v6 needs fixing too. The per-candidate columns are `simulated_v6`
and `candidate_revenue_v6`. `line_gap_v6` buckets the bets by v6's line
less the bet's (whole points, capped at ±3), to separate what moving the
line did from what the probability did. Running `--candidate v6,v6@prod`
prices both from one simulation. A model's own line sits in the gap between
the key numbers; `v10@even` reads it at the half-point line nearest 50%
instead. v10 also has `v10@anchored` (prod's line, moved only as far as it
must) and `v10@hyst` (the even line, held until far off). See eAMFModel's
README. `--candidate v10,v10@even` prices both from one simulation.

**Prod's pre-match by gamer (Aug 25–Sep 10, 1,424 matches).** Prod's
pre-match misses by gamer are larger than chance. The mean z² runs
1.8–2.5 against 1.0 for no signal, largest on totals. They also persist: a
gamer's miss in the first half of the window carries into the second
(correlation +0.2 to +0.36). By stream, there's nothing. The home side
wins 51.1% of 24k matches, +0.3 points, and prod prices that right.
- **NB2 fitted before each half beats prod on the second half:** margin
  RMSE 9.46 against 9.76, total RMSE 11.86 against 11.97.
- **Correcting by each gamer's first-half miss:** it helps prod a little
  (margin 9.72, total 11.91) and NB2 barely (9.45, 11.85). So NB2 already
  carries most of the gamer signal prod misses.

**Sep 17–28, first run.** The in-play book kept −1.9% against an expected
+7.6%, so bettors beat prod's probability by about 9.5 points.
- **It isn't timing.** The edge is about the same between plays and during
  them. Bets struck under 10 s before a score made the book money (+10.6%).
  Bets across a score, or with a conversion to come, number fewer than 100
  (the operators suspend there).
- **Small leaks:**
  - prices over 60 s old (−48%, 177 bets);
  - a price from before a kick-off (−30%, 308 bets);
  - timeouts (−8.7%, 2.3% of stake).
- **Where the money goes.** Level scores (−6.1%, 54% of stake, −204k), the
  first two minutes of Q1 (−7.8%, −104k), unders (−8.5%, −130k) and the home
  side (−4.5%).
- A third of in-play stake is struck while the feed has betting suspended
  (a play is live).

### Several candidates: the model versions

```bash
python -m eAMFCalibrator bets --since 2026-09-17 --until 2026-09-23 --candidate v8,v9,v10 \
    --v8-model v8_model --v9-model v9_model --v10-model v10_model
```

Every candidate shares the bets, the lags, the line signs and prod's quotes.
Each is read at the prod message the bet saw (one lag before it), so the lag
is the same for all of them. A model version prices every prod message off
its latest PLAY_OVER at or before it, carrying prod's message and publish
time. It quotes its own lines (in the gap between the key numbers) and
probabilities, and the bet is settled at that line. `v5@prod` reads it at prod's lines instead. A table
(`GAMEPLAI_STREAM_CANDIDATE`) can sit in the list too.

A model has no pre-match price (its first quote is the first PLAY_OVER), so
its pre-match bets aren't re-priced. Each model simulates every match in the
window (`--vN-paths` sets how many games per snapshot).

The report prints the lag and line checks once, then each candidate's own
sections as above. It ends side by side: the margin change of each candidate
on the bets every one of them re-priced (the same money), by the same cuts,
and each one's price and result effects.

`bets lines` draws prod's spread and total lines through each match against
the lines customers bet. If an operator only offers the lines GAMEPLAI
sends, a bet on another line than prod's live one was placed on a line prod
showed a little earlier (the operator runs behind) or a little later (its
clock runs behind GAMEPLAI's).

```bash
python -m eAMFCalibrator bets lines --since 2026-09-18 --until 2026-09-25
python -m eAMFCalibrator bets lines --since 2026-09-18 --until 2026-09-25 --match AF012200926
```

Every spread and total bet's line, read from prod's side, is classed
against prod's LIVE rows:
- **current:** prod's live line at the bet;
- **past:** a line prod last showed live that many seconds before;
- **future:** one prod showed live that many seconds after;
- **never:** not within two minutes either way.

The summary gives the shares and the seconds (median, 90th), by operator
and market, by pre-match or in play, and in play by period (does the gap
move through the match?). `out/bets_lines.html` draws the busiest matches
(`--matches N`, or `--match CODE`): prod's live line as a step, inactive
rows grey, each bet a dot coloured by its class, with the details on hover.

Prices and lines are read off prod's live rows only (`IS_ACTIVE`, as
everywhere in the calibrator; `STATUS` isn't trusted). One moment can carry
the settlement of the old line beside the open quote for the new one, and
only the live row is a price anyone could take.

### Checks: when each bet was placed, cash-outs, and the same information

```bash
python -m eAMFCalibrator bets check --since 2026-09-25 --until 2026-09-28
```

`bets check` runs the checks alone (no candidate is priced, so it's quick)
and writes `out/bets_checks.csv`. Every `bets` run prints them too.

- **Sport check.** Only AF may reach the join. The bets are filtered to
  `SPORT_CODE = 'AF'`, and everything else is read by those bets' match
  codes (SCOUTING_FULL also by `MATCH_CODE LIKE 'AF%'`). The check confirms
  it from the data: EVENT's `SPORT_CODE` for the matches, any SCOUTING_FULL
  match code not starting AF, messages that look like another sport
  (free throws, rebounds, ...), and every SCOUTING_FULL message with its
  rows and matches.
- **When in the match.** The start (the first quarter's start message, or
  the first `PLAY_STARTED` without one), two minutes left in Q4 (the first
  row in the fourth quarter with the clock at 120s or under) and the
  match-over message are read off SCOUTING_FULL. They're put on prod's
  clock through the message count (prod's first row on or after that
  message). Each bet is placed pre-match, in play, after two minutes left,
  or after match over. Bets marked pre-match after the start, marked in play
  before it, accepted after two minutes left (the operators suspend there;
  `EXCLUDE_AFTER_TWO_MINUTES`) or after the match-over message are left out
  (resettlements, trader reviews). `BET_PHASE_TOLERANCE` (10s) is allowed.
  The match-over message is `ENDED`.
  The report gives bets, stake, margin and seconds past the bound for each,
  by operator. It also lists the statuses seen from two minutes left
  (seconds after the mark) and on the matches' last rows. That's how the
  match-over message is found; where none is, the match's last feed message
  stands in.
- **Cash-outs.** `BET_CASHED_OUT = Yes` is its own result (`cashed out`).
  They're settled at the operator's cash-out offer, and there's no candidate
  offer to set against it, so they aren't re-priced. The report shows them
  by operator and pre-match or in play: bets, stake, share of stake and
  margin. Payouts that are none of won, lost, push or cashed out are
  `payout not won, lost or push`.
- **The same information: the exact feed messages behind every price.**
  Prod's price the bet saw was made at feed message m
  (`EVENT_MESSAGE_COUNT`, the SCOUTING_FULL message count; `message`).
  - **The models** (v4–v6) price off their latest `PLAY_OVER` s (`feed_from`
    to `feed_to` = m). They now only price where they're sure of the state
    (see eAMFModel's README): the latest `PLAY_OVER`, one they can read, with
    TEAM_A's side known, until the next play starts. The check reads the
    same snapshots the same way without simulating. It counts which
    `PLAY_OVER`s can't be read and why, and the matches with TEAM_A's side
    not known.
    The report breaks the unreadable ones down by play kind, reason and the
    fields missing, and gives both by match day. For the matches with
    TEAM_A's side not known it gives why (no scoring message by a score
    change, or split) and the feed messages found at their score changes.
  - **A candidate table's** quote at message c (`candidate_message`) is held
    to the same rule.
  - A price is only compared where the feed didn't move on between it and
    m: no play started or other message in SCOUTING_FULL beyond
    `NEUTRAL_FEED_MESSAGES` (`BET_SUSPEND`, `BET_UNSUSPEND`), no score
    change, and no prod message SCOUTING_FULL lacks (at most
    `MAX_SCOUTING_GAP`, `--gap N`, default 0).

  Anything else isn't re-priced, with the reason (`model: feed moved on:
  PLAY_STARTED`, `model: the latest PLAY_OVER cannot be read`, `candidate:
  score changed`, ...). The report gives the share with the same
  information and the commonest reasons: a message that moved the feed on
  but shouldn't have is one to add to `NEUTRAL_FEED_MESSAGES`. It also
  gives, by period, the seconds from s to m, and the matches where
  SCOUTING_FULL holds under 99% of prod's messages. The side-by-side table
  is on bets every candidate re-priced, so every comparison there is on the
  same information.

`bets probe` prints the bet source's columns (and any configured ones it
lacks), the window's bets by operator, and the values of `BET_IN_PLAY`,
`CUSTOMER_TEMPERATURE`, `BET_CASHED_OUT` and `CUSTOMER_WIN_LOSS`.

Output:
- `out/bets_sim.csv`, one row per bet: `in_play`, `stream_prob`,
  `candidate_prob`, `candidate_line`, `implied_prob`, `latency_seconds`,
  `result`, `candidate_result`, `candidate_odds`, `candidate_revenue`, the
  final score, plus the bet source's own columns. With several candidates,
  each candidate's columns are suffixed by its name (`candidate_prob_v5`,
  `candidate_line_v5`, `candidate_revenue_v5`, `simulated_v5`, ...);
- `out/bets_latency.csv`, one row per operator, with the misfit at every
  lag tried.

### Restricted accounts on totals: `bets totals-moves`

```bash
python -m eAMFCalibrator bets totals-moves --since 2026-09-17 --until 2026-09-30 --min-bets 30
python -m eAMFCalibrator bets totals-moves --since 2026-09-17 --until 2026-09-30 --candidate v8,v9 --v8-model v8_model --v9-model v9_model --min-bets 30
```

**Same book only.** The baseline keeps only the operators that send
restricted accounts (FanDuel: Hard Rock sends no temperature, so its restricted
accounts would sit in the baseline). It says how many bets of other operators
it left out. `--all-operators` keeps every operator.

This reads prod's totals line before and after every totals bet, and compares
restricted accounts (`CUSTOMER_TEMPERATURE = BET_RESTRICTED_VALUE`) with
everyone else on the same matches. A pattern only counts if it's different from
the baseline. The bets, lags, line signs, checks and candidate pricing are the
`bets` pipeline's, unchanged.

**The sign.** level = side × (prod's line then − prod's line at the bet), with
side +1 for Over and −1 for Under. + means the market rated the bettor's side
higher then than at the bet:
- after the bet, + is a move **their way** (they beat the move);
- before it, + is a move they **faded**, and − is a move they **followed**.

The bet is read at the price it saw (bet time less the operator's lag). Where
the line held, the bet selection's probability carries the move, if it changed
by a point or more. Line and probability are never averaged together.

It prints, and writes to `out/bets_totals_moves.html`:
- totals bets by temperature: stake, margin, expected margin, over share;
- the **event study**: mean level at −120, −60, −30, −10s, at bet time (the
  move inside the operator's lag), +10, +30, +60, +120, +300s, ±2se clustered by
  match, with the shares each way. Shown for all bets, Over and Under;
- pre-match bets against prod's closing line (the last before kick-off);
- **before × after**: followed / faded / still line at −60s against their way
  / against / held at +60s, with the book's margin in each cell;
- the restricted bets and the baseline cut by side, in play, quarter, points
  still needed (line − on board), the feed at bet time, price age, the move
  before, **whether a score fell between the bet and +60s** (a move with no
  score is the price drifting; a move across a score is the game), and operator;
- the restricted customers with the most stake: is it a few accounts or all of them?
- **by gamer and by matchup** (gamers off `EVENT`, PLAYER_1 home), most
  restricted stake first. Each gamer gets every bet of his matches. Columns:
  - restricted bets, stake, margin and under share, and the baseline's bets
    and margin;
  - **prod miss**: prod's line at the bet less the final total (+ is a line
    set too high);
  - for each candidate, on the bets it carries (how many): its own line less
    the final, prod's on those same bets (the like-for-like comparison: the
    first prod miss is over every restricted bet), and its change in margin.

  A gamer whose matches prod sets too high, which restricted accounts bet
  under, and which a model already sets lower, is information the model
  carries and prod lacks.
- **through a session.** A gamer's session is a run of matches with no gap
  over `SESSION_BREAK_MINUTES` (240). The schedule runs a match every 35–110
  minutes, then breaks for three hours or more, and most sessions are 8–10
  matches. Each match is labelled by its place: both gamers on their last
  match, one of them on it, 2nd last, 3rd last, or earlier. Two tables:
  - **scoring through a session** over every settled match (`history`): each
    total less its two gamers' own mean totals, ±2se;
  - **the window's matches and money by place in the session**: how many
    matches, their share, their totals against the gamers' means, and the
    totals bets and stake on them (per match, and the share on the under).
    Then, for all customers, restricted accounts and everyone else: each
    place's share of matches beside its share of bets and stake (money lumped
    on a place shows as a stake share above its match share), stake per
    match, the share on the under by bets and by stake, and the book's margin
    on unders and overs. Every settled bet the checks keep, priced or not;
  - **the bets by place in the session**, laid out like the gamer table:
    restricted and baseline margins, prod's miss, and each model's miss
    against prod's on the same bets.

  On `nb2/AMFELO.csv` (23,952 matches, Dec 2025 – Sep 2026), a match where
  both gamers are on their last of the session scores **2.39 ± 0.52 points
  under** their means, against +0.24 for the rest. It's the last match
  whatever the session's length (8 to 11), so it's the end of the session,
  not the match count. With only one gamer on his last, there's no effect.

  `bets_totals_moves.csv` carries each gamer's `*_session_match`,
  `*_session_left` (1 = last) and `*_session_length`, and the match's
  `session_end`.
- **each candidate**, at the message the bet saw, only where it carries the same
  information as prod's:
  - whether it already leaned the bettor's way (its line, else its probability
    at the same line);
  - its line edge over prod's;
  - whether its line sat nearer where prod went by +120s than prod's own did;
  - its own move over the next 60s;
  - the change in margin had it priced the bet.

  It's split by what prod did next. On the bets prod moved their way, a model
  that leaned their way already had the information. One that didn't has the
  same leak.

`out/bets_totals_moves.csv` has one row per totals bet: every `bets_sim.csv`
column, plus `level_*`, `plevel_*`, `way_*` and `scored_*` at each offset
(`m60` is −60s, `p60` is +60s), `level_bet`, `level_pre_score`,
`level_kickoff`, and each candidate's `<name>_lean`, `_line_edge`,
`_prob_edge`, `_nearer_p60`/`_p120`, `_level_p60` and `_change`.
`--min-bets` (default 100) hides thin buckets. Restricted samples are small, so
30 is a better starting point.

### What separates the totals that go under: `bets totals-signals`

```bash
python -m eAMFCalibrator bets totals-signals --since 2026-09-17 --until 2026-09-30
```

This reads every settled match in the window, not only the ones with bets. At
three checkpoints it sets how the match was shaping up against prod's total
there, and against where the totals money went next.

| checkpoint | prod's line | features | money |
|---|---|---|---|
| pre-match | the closing line (the last before Q1 starts) | match of the session, both gamers on their last, both gamers' recent form and long-run level (walk-forward off `history`: nothing after the match is read) | pre-match bets |
| end of Q1 | at the message Q2 starts on | how Q1 was played (below) | bets struck in Q2 |
| half time | at the message Q3 starts on | how the first half was played | bets struck in the second half |

How a segment was played comes off SCOUTING_FULL, play by play:
- scrimmage plays;
- **real seconds between plays**: the median from one `PLAY_OVER` to the next
  snap, which is how long the gamers take;
- real seconds a play runs;
- **game-clock seconds per play**: how much of the 240-second quarter each
  play burns;
- the real minutes the segment took;
- yards per play;
- drives ended (punt, field goal, touchdown, turnover on downs), and punts,
  touchdowns and field goals on their own;
- timeouts called;
- points on the board, and prod's line less them.

The play after a touchdown is its conversion, and the one after a conversion,
field goal or safety is the kick-off. Neither counts as a scrimmage play.

**over** = final total less prod's line at the checkpoint. A feature prod
already prices sits flat around 0 whatever its value.

For each checkpoint the report:
- **ranks the features** by how far over moves from the lowest bin to the
  highest (prod missing it), with the correlation, and the under share of the
  next segment's totals stake in those bins (the money reading it);
- shows **each feature by quintile**: over ±2se, prod's line less the board,
  stake per match, the under share of the stake for all customers, Restricted,
  VIP and Standard, and the book's margin on unders. A count with six values or
  fewer gets one bin per value.

A feature where over moves and the VIP or restricted under share moves with it
is a signal the money reads and prod doesn't. One where over moves and the
money doesn't is one nobody is using yet.

Writes `out/totals_signals.csv` (one row per match, every feature, line, over
and the money by temperature) and `out/totals_signals.txt`.

On `nb2/AMFELO.csv`, out of sample: knowing the two gamers takes the RMSE of the
total from 13.06 to 12.61, and their recent form (an EWMA of about the last 20
matches) takes it to 12.30. The long-run level adds little once form is in.
Their teams, gamer × team and the gamer pair add nothing.

#### The models at the same checkpoints: `--candidate v8,v9`

```bash
python -m eAMFCalibrator bets totals-signals --since 2026-09-17 --until 2026-09-30 --candidate v8,v9 --v8-model v8_model --v9-model v9_model
python -m eAMFCalibrator bets totals-signals ... --candidate v8,v9 ... --react
```

Each model version in `--candidate` is priced only where it's read:
- at the kick-off, from its pre-match model;
- at the latest priceable `PLAY_OVER` before Q2 starts, and before Q3 starts.

Every earlier snap is still read into the game, as the stream reads them.
That's three simulations a match rather than one for every `PLAY_OVER`. Each
checkpoint gets the model's own line, its mean total, P(over prod's line),
final less its line, and its points to come (mean less the board).

The report adds, per checkpoint, on the matches every line priced:
- **over, MAE and Brier** at prod's line, prod against each model;
- **points to come, by points on the board** (by form, pre-match): what
  really came, against prod's line less the board and each model's. A line
  that anchors on the pre-match total gives a slow start more to come, when it
  really produces less;
- over **with and without play detail**: the models read the game from the
  down and field position, which SCOUTING_FULL has carried empty since 23 Sep;
- **a regression** of the points the rest of the match made on each line's
  points to come and how the match was played (points, plays, clock per play,
  seconds between plays, drives). A line that priced everything takes a
  coefficient of 1 and leaves the rest at 0.

**`react`.** The streams run v8 and v9 with `react=False`, so each side's
strength stays at its pre-match value all match. The score, clock and field
move the price; how well the sides are playing doesn't. `--react` prices each
model again with its in-game efficiency update on (labelled `v8-react`). That
update moves strength on first-down success, never on points. It was dropped
when first-down success predicted later points only weakly (eAMFModel README).
In the first run here (17–30 Sep), the scoring so far predicted the rest of the
match beyond prod's line. After a scoreless Q1 the match finished 3.1 points
under prod's line at Q2. So an update on points and drives is the untried one.

### The points still to come, calibrated: `totals-calibrate`

```bash
python -m eAMFCalibrator scouting --since 2026-06-01 --until 2026-09-30     # the export, months of it
python -m eAMFCalibrator totals-calibrate                                    # out/scouting_playover.csv
python -m eAMFCalibrator totals-calibrate --history nb2/AMFELO.csv --signals eAMFCalibrator/out/totals_signals.csv
```

Every line read so far (prod's, v8's, v9's) spreads its points to come wider
than the rest of the match does. At half time each line's own points to come
took a coefficient of about 0.35, where a line that needs no correction takes 1.
It also gives a slow start more to come when it really produces less.

This fits the correction at the end of Q1, half time and the end of Q3:

    rest of the match = a + b × prod's points to come + c × points on the board
                        + d × prod's closing pre-match line + plays + margin + clock left (+ form)

It reads the `scouting` export only, so it needs no Snowflake. It doesn't
need the play detail either: only the period, clock, score, scrimmage flag and
prod's total. The fit is on the earlier two-thirds of the matches; the latest
third is held out. For each checkpoint it prints:
- the **simple** fit (points to come alone) and the **full** one, with each
  coefficient ±2se. A feature that duplicates others is left out;
- on the held-out matches, prod, simple and full side by side: over (the rest
  of the match less each line's points to come), MAE, and the Brier of
  P(over prod's line). The calibrated P(over) reads the fit's own residuals,
  so a lumpy total keeps its shape;
- points to come by points on the board, real against each line's.

`--history` adds the gamers' recent form (walk-forward). `--signals` reads a
`totals_signals.csv` run with `--candidate v8,v9`. At the start of Q2 and Q3 it
fits, five-fold cross-validated, rest = a + b × the line's points to come + c
× points on the board for prod and each model, and sets each line as it stands
against it corrected (MAE, Brier, and the mean b).

It writes `out/totals_calibration.json` (per checkpoint, each fit's
coefficients and standard errors, and its residual quantiles) and
`out/totals_calibration.txt`.

### The pre-match favourite: `bets favourite`

```bash
python -m eAMFCalibrator bets favourite --since 2026-09-10 --until 2026-09-23 --min-bets 50
```

Every single (moneyline, handicap and total, pre-match and in play) on a match with a pre-match
favourite, split by:
- **the favourite:** the side prod's last pre-match moneyline rated above 50%. A match within 2%
  of evens has none and is left out;
- **phase:** pre-match, or the quarter the bet was placed in (the operator's period);
- **state:** the favourite's lead at the prod message the bet was priced at (down 9+, down 1–8,
  level, up 1–8, up 9+);
- **side:** favourite or underdog on the moneyline and handicap, over or under on totals;
- **customer temperature:** Restricted, VIP, Standard, none.

Each cell gives:
- bets and stake, and the share of the state's (or phase's) stake;
- the book's margin, and the margin prod's price leaves it (1 − odds × prod's probability of the
  selection, on bets struck at prod's line);
- the gap between them. Positive means bettors took more than the price allowed there;
- prod's probability against how often the selection won (stake-weighted).

The tables are, in order:
1. every bet by phase and the favourite's lead;
2. Q1 and Q2 by lead and side, for each market;
3. the same for each temperature;
4. every temperature by phase and side;
5. the favourite ahead in Q1 and Q2, by temperature, market and side.

Writes `out/bets_favourite.csv` (one row per bet, with its prod price, favourite, state and side)
and `out/bets_favourite.txt`.

### Every gamer's h2hggl stats from our own data: `player-stats`

```bash
python -m eAMFCalibrator player-stats --since 2026-08-01 --check scrapes/h2hggl_api/<time>/players.csv
python -m eAMFCalibrator player-stats                       # every match, all of SCOUTING_FULL
```

This rebuilds the player stats h2hggl.com shows (see `analysis/h2hggl_api.py`)
from our own data. They are meant as inputs to a pre-match model, and are
checked against the site's own numbers.

- **From the match history, all time:** matches played, won, lost and drawn,
  win %, the longest and current winning streaks, the last ten results, and
  points for and against.
- **From SCOUTING_FULL, match by match:**
  - **How plays are read:** the PLAY_OVER snapshots are taken in pairs, as
    the models read them. A snapshot's offence, down, distance and field
    position are the snap. The next snapshot's play kind and spot are what
    that snap became.
  - **The stats:** scrimmage plays, yards and yards per play, 3rd and 4th
    downs converted, red-zone trips and how they ended (touchdown or field
    goal), turnovers on downs, time of possession (game clock), punts and
    their net yards, touchdowns, two-point tries, and fouls. A side's yards
    conceded are its opponent's yards.
  - **The detail gap:** field position, down and distance were missing for a
    while (about 23 Sep – 9 Oct 2026). Those matches count only towards the
    stats that need no spot.

One row per match and side goes to `out/player_match_stats.csv`. It is kept
between runs, so a rerun reads only the matches it lacks, and it is the input
for pre-match features. The per-player totals go to `out/player_stats.csv`.

With `--check`, each stat is set against the site's figures (the site's are
career totals):
- **Exact matches:** played, won, lost, win % and the form.
- **Level and rank across the players:** the rest.
- **Both readings, where a stat has two:** red zone as touchdowns only or
  any score; fouls by the team on the message or the other side; yards with
  or without plays that had a foul. The comparison shows which reading the
  site uses.

Writes `out/player_stats.txt` and `out/player_stats_check.csv`.

### Pre-match priors against prod, with and without the rematch layer: `prior-backtest`

```bash
python -m eAMFCalibrator prior-backtest --since 2026-09-01 --until 2026-10-14 \
    --arm "glmer-trim=glmer/out/cadence/cadence_matches.csv@trim:hl60:game" \
    --arm "nb2=rolling:nb2_daily"
```

An arm is `NAME=SOURCE`: its expected points per side for every match, as of
the match's kick-off. The sources are:

| source | what it reads |
|---|---|
| `PATH@PREFIX` | `glmer/cadence.R`'s `cadence_matches.csv`, columns `PREFIX\|1` and `PREFIX\|2` (e.g. the trimmed model refitted after every result, `trim:hl60:game`) |
| `PATH` | a CSV of `MATCH_CODE` (or `MatchId`) and `Pred_P1_Points` / `Pred_P2_Points` (`NB2_schedule_predict.py`, `glmer/predict.R`), `raw_home` / `raw_away` (eAMFModel `prior-vs-sim`'s CSV) or `exp_p1` / `exp_p2` |
| `rolling:DIR` | eAMFModel `prior-daily`'s directory of daily fits (NB2 or glmer): each match priced by its own day's fit, as live |

Prod is always included: its closing pre-match moneyline and total (the line
and the over's price). That is its last live quote before the match got under
way, which `prematch.py` defines as no message count, or a message count below
the match's first `PLAY_STARTED` in SCOUTING_FULL. Prod's pre-game quotes carry
a message count. Where the play feed has no first play, the quote must be
published before the scheduled kick-off.

Prod is a third party's price, so it is the benchmark only and gets no
layers. The run prints how far its closing moneyline is from the price these
tools read before, which was its last quote with no message, else its first.
The same closing prices are used by `prior-replay`, `bets rematches` and
`bets inference`.

**How each arm is graded:** each arm is graded as it is and with each layer
on top. Each layer is fitted walk-forward on that arm's own errors.

| suffix | layer |
|---|---|
| `+layer` | the rematch layer: the pair's earlier meetings that day |
| `+infer` | the inference layer (`bets inference`, below): each gamer's form that session, from every match he played against anyone, beside the pair's own effect |
| `+both` | the two as one layer: a rematch moves by the pair's earlier meetings alone, and a first meeting by what the gamers did against others |
| `+feat` | the feature layer (below): what the two gamers' stats say beyond the arm's price |
| `+both+feat` | the feature layer fitted on the errors `+both` leaves, on top of `+both` |

The feature layers need the gamers' stats: `player-stats`'
`out/player_match_stats.csv` by default, another with `--features CSV`, none
with `--features none`. Without the file, the run says so and leaves them out.

An arm's moneyline and over come from its two scores simulated together
(`score_model.py`), so the covariance between them is in the price. Choose
how with `#MODE` on the arm:

| mode | the score model |
|---|---|
| `own` | the prior's own (the default where it has one). For NB2, each side is negative binomial with its gamers' dispersions, and the two share a gamma pace factor, which gives the covariance. This is read off NB2's prediction rows: a `rolling:` NB2 directory, or a CSV with `P1_Effective_Alpha`, `P2_Effective_Alpha` and `Shared_Gamma_Fraction`. For glmer, each side is Poisson about a log-normal mean with a shared match noise, read from a `rolling:` glmer directory. |
| `fitted` | NB2's construction with one dispersion and shared fraction fitted to the arm's own errors by the method of moments, walk-forward. This is the default for a prior with no score model of its own, e.g. the trimmed glmer. |
| `normal` | the normal about the expected margin (sd 9.35) and total (sd 12): no covariance, shape or ties. For comparison. |

- **Ties:** a moneyline splits them half and half, as NB2 does; the over
  leaves out pushes.
- **The layer:** shifts the simulated margin and total, so it moves the
  model's own price.
- **Randomness:** every match draws from its own seed, so every arm and
  variant of a match sees the same randomness. `--draws` sets the scorelines
  per match (default 40,000).
For example, to see what the score model alone is worth:
`--arm "nb2=rolling:nb2_daily" --arm "nb2-normal=rolling:nb2_daily#normal"`.

**What it reports**, on the matches every arm priced, after the layer's
7-day burn-in:
- **moneyline:** log loss and Brier;
- **total:** the expected total's RMSE and bias, and the log loss of the
  over at prod's line;
- **each against prod, match by match:** the difference, its t, and the
  fortnights the arm beat prod.

These are given for every match, the rematches, the 2nd meetings, the 3rd+
and the first meetings. The first meetings are also split into those after
either gamer had played someone else that session (only the inference layer
moves these) and those where neither had. Each arm's last fit of every layer
follows. Writes
`out/prior_backtest.txt` and `out/prior_backtest.csv` (every match, every
arm's moneyline, total and over).

#### The gamers' stats as features: the feature layer

`gamer_features.py` turns `player-stats`' per-match rows into each gamer's
stats as of each day. `feature_layer.py` moves an arm's price by what those
stats say beyond it.

**The stats**, each gamer's, recalculated every UTC day from the matches
before that day only:

| stat | read as |
|---|---|
| `ypp_for`, `ypp_against` | yards a play made and conceded |
| `third_pct` | 3rd downs converted over run |
| `rz_td_pct` | red-zone touchdowns over trips |
| `giveaways`, `takeaways` | turnovers and turnovers on downs a match, the gamer's own and the opponent's |
| `top_min` | time of possession a match, minutes |
| `plays` | scrimmage plays a match (pace) |
| `points_for`, `points_against` | points scored and conceded a match |

- **Recency:** a gamer's matches are weighted by how recent they are among
  the gamer's own, 2^(-k/50) for the k-th most recent. Every match the gamer
  plays ages the rest.
- **Shrinkage:** towards the league as it stood before the day, by 20
  matches' worth of its average. A new gamer reads as the league.
- **The detail gap:** a stat that needs a field position reads only matches
  with one on 80%+ of the snaps (`player-stats`' rule). Through the 23 Sep – 9
  Oct gap such a stat stays where the gamer's earlier matches left it.
- **Nothing from the match's own day:** a match is read with its gamers'
  stats as of the start of its UTC day.

**The layer:** each UTC day, on every earlier match of the window with a
result, a ridge regression of the arm's errors on the stats:
- the margin's error on the home gamer's stats less the away's;
- the total's error on the two gamers' stats added.

The stats are standardised and the errors centred, so the layer moves a price
by what the stats say about it, not by the arm's overall bias (as the
rematch layer is centred). The ridge (50) pulls every coefficient towards 0.
No move until 400 matches have been seen, and none for a match missing a
stat. `+both+feat` fits on what `+both` leaves, so the two don't count the
same thing twice. The report gives each arm's last fit: every stat's
coefficient in points per standard deviation, with its se and t.

### The pre-match bets at each prior's prices: `prior-replay`

```bash
python -m eAMFCalibrator prior-replay --since "2026-09-01 00:00:00" --until "2026-10-15 00:00:00" \
    --arm "glmer-daily=glmer/out/cadence/cadence_matches.csv@trim:hl60_x0.5:daily" \
    --arm "nb2=rolling:eAMFModel/out/nb2_daily"
```

`prior-backtest` grades the prices. This is what the book would have made
from them. The arms are `prior-backtest`'s (`NAME=SOURCE[#MODE]`) and are read
the same way, `--features` too. Each arm runs as it is and with each layer
(`+layer`, `+infer`, `+both`, and with the gamers' stats `+feat` and
`+both+feat`). Prod, a third party's price, is the benchmark: the bets' own
prices.

**How a bet is re-priced:** the same way `bets prematch` re-prices a
candidate stream. Every pre-match bet on the matches every arm priced, after
the layer's burn-in, keeps its selection, stake and line:
- prod's probability at bet time (its quote then, whatever its message
  count) is swapped for the arm's at that line;
- the operator's margin is kept, so the odds become odds × prod / arm;
- the bet settles as it settled.

**An arm's price at a line:**
- moneyline and spread: its home margin over the line, pushes left out;
- total: its total over the line, pushes left out.

**Clean bets:** an arm prices a match as of kick-off, from every result
before it. A bet placed before either gamer's previous result was in (36
minutes after that match's kick-off) would be re-priced with results neither
the bettor nor prod had at the time. Those bets are reported separately, and
the headline is the clean bets.

**The bets stay as placed:** nobody bets differently at the arm's prices.
Only winning bets move the book: a shorter price on one is revenue kept, a
longer one revenue given away. A losing bet loses its stake at any price.
Bets that wouldn't have been placed at the arm's price, or that would have
been placed instead, aren't modelled.

**What it reports:** prod's margin on the bets (revenue / stake) and each
variant's change against it, in points of stake. Each change comes with a 95%
interval (resampling matches) and the weeks it came out ahead. The cuts are:
- the clean bets, overall and by market;
- the clean bets by meeting, market and meeting, operator, VIP and the rest,
  and week;
- every pre-match bet;
- stake by minutes before kick-off.

Writes `out/prior_replay.txt` and `out/prior_replay.csv` (a row per bet with
each variant's probability, odds and revenue).

### NB2 refitted daily over the year, every match priced: `nb2-history`

```bash
python -m eAMFCalibrator nb2-history --since 2026-01-01 --history eAMFCalibrator/out/match_history.csv
```

NB2 is refitted every day from `--since`, each fit on every result settled before that day began,
as it would run live, with no layers. Every match is priced by its own day's fit. It needs no
Snowflake given `--history`. The fits are kept in `out/nb2_history/` (`nb2_history_hl<D>/` with
`--half-life`), so a rerun fits only the days it lacks. Only the days the history covers are fitted.
About 20 s a fit, two a day (the day's and its level scale's): `--workers 4` (the default) does the
year in under an hour.

`out/nb2_history.csv` has a row per match:
- **The match:** code, kick-off, stream, gamers, teams, the pair's meeting number that day, each
  gamer's settled matches before the day, the score.
- **NB2's price:** expected points (`Pred_P1_Points`, `Pred_P2_Points`), margin and total, and the
  moneyline from NB2's own joint score model (`p_home` splits a tie half and half; the home win,
  draw and away win are given apart). NB2 gives ties 3–4%; the games have almost none (6 in 24,000:
  overtime settles them).
- **The fit:** its day, the matches it was fitted on, its level scale, and the gamers' dispersions
  and shared fraction. These let `prior-backtest` read the CSV as an arm with NB2's own score model.

Every price was made before its match's day, so a model fitted on the CSV's earlier rows and
priced on a later day's sees nothing it couldn't have live. `glmer/cadence.R --nb2` fits glmer on it
(glmer/README.md). `out/nb2_history.txt` grades NB2 month by month (the moneyline's log loss and
Brier, the margin's and total's RMSE and bias) with the moneyline's calibration.

### The pre-match variants for the sims: `walkthrough`

```bash
python -m eAMFCalibrator walkthrough --history eAMFCalibrator/out/match_history.csv --since 2026-01-01
```

Prices every match of a span with the best pre-match variants, walk-forward, and writes one table
of expected points per variant. A sim can then be pointed at any variant without pricing the
pre-match again. No Snowflake is needed: the history comes from a CSV.

**Progress:** before it starts, it prints a first estimate of the whole run. Each step then prints
a line as it goes: units done, its pace, the time it and the run have left, and the run's estimated
finish. For example:
`NB2 fits: 37/253 days to fit (15%), 9m12s in, ~53m45s left; the run ~1h12m left, finishing about 15:46`.
A step's own pace rescales the steps still to come, and the estimate settles after the first few
fits.

| step | what it does |
|---|---|
| NB2 fits | NB2 refitted every day, each on the results before its day (`nb2_fits/`). The days already fitted are kept |
| NB2 prices | every match priced by its own day's fit (`nb2.csv`) |
| glmer fits | glmer refitted every day (`glmer/cadence.R`, `trim:hl60_x0.5:daily`, `glmer/`). Needs `Rscript` |
| layers | the rematch and inference layers fitted walk-forward on each model's own errors |
| variants | each variant's expected points and moneyline for every match (`priors/`) |
| report | every variant graded against NB2 on the results (`walkthrough.txt`) |

A step whose output is already there for the same history and span is kept, so a rerun after a
stop picks up where it was. From scratch, a year takes about an hour and a half on 4 cores, most of
it the NB2 fits (`--workers`, default 4).

**The variants**, each a table `priors/<variant>.csv`:

| variant | expected points |
|---|---|
| `nb2` | NB2 as it is |
| `nb2+infer` | NB2 moved by the inference layer |
| `nb2+both` | NB2 moved by the rematch layer at a rematch, else the inference layer |
| `glmer` | glmer as it is |
| `glmer+both` | glmer moved as `nb2+both` |
| `avg` | `nb2+infer` and `glmer+both` averaged side by side: the best pre-match model in `prior-offline`'s tests |
| `avg-plain` | `nb2` and `glmer` averaged |

A layer moves the expected margin and total; a side's points are (total ± margin) / 2. Each table
also has:
- the match, its score and its meeting number;
- the moneyline off the variant's points: NB2's own joint score model for `nb2`, its layers and
  the averages (the averages' points with NB2's dispersions), and NB2's construction fitted to
  glmer's errors for glmer;
- the score model's parameters, the layer's move, and whether the match is in the layers' 7-day
  burn-in (no move there).

Every value was priced before its match, as it would have been live.

**Pointing a sim at a variant:** give its table where a prior build would go:

```bash
python -m eAMFCalibrator report --candidate v21 --v21-prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv
python -m eAMFModel v21-report ... --prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv
```

v21 then nudges toward that variant's expected points (eAMFModel's `table_prior.py`). A match the
table doesn't have gets no nudge. `walkthrough.json` records the span, the history, each variant's
file and each step's time.

**On the repo's history** (`nb2/AMFELO.csv`, 1 Jan – 10 Sep, 21,638 matches after the burn-in).
Moneyline log loss, NB2 alone 0.6581; the other variants against it:

| variant | moneyline | t | fortnights better | total sq err (t) |
|---|---|---|---|---|
| `avg` | −0.0028 | −5.4 | 17/18 | −1.90 (−8.8) |
| `glmer+both` | −0.0020 | −2.8 | 11/18 | −1.86 (−7.2) |
| `nb2+infer` | −0.0018 | −3.2 | 13/18 | −1.33 (−6.5) |
| `avg-plain` | −0.0018 | −6.7 | 17/18 | −1.26 (−12.0) |
| `glmer` | −0.0016 | −2.9 | 15/18 | −1.74 (−8.3) |
| `nb2+both` | −0.0012 | −2.6 | 13/18 | −0.79 (−5.0) |

`avg` leads at rematches (−0.0032) and first meetings (−0.0019) alike. On 4 cores, with the NB2 fits
already made, the run took 19 minutes:
- NB2 prices 5 min;
- glmer fits 5 min;
- layers 3 min;
- variants 6 min.

### A historic match replayed, v21 against prod: `replay-ui`

```bash
# matches off Snowflake (every answer cached), the expected points off the walkthrough's best variant
python -m eAMFCalibrator replay-ui --v21-model v21_model \
    --v21-prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv --history AMFELO_20261013.csv
# no Snowflake: the matches off a `scouting` export
python -m eAMFCalibrator replay-ui --v21-model v21_model --export eAMFCalibrator/out/scouting_playover.csv \
    --history AMFELO_20261013.csv --v21-prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv
```

Opens a local page (`http://127.0.0.1:8766/`) to step through a historic match a PLAY_OVER at a
time. v2.1 prices each state, and prod's quote at that message sits beside it. It is the v21
counterpart of eAMFModel's `trader` page, which sets up made-up states and doesn't load a v21
build.

**Picking a match:** choose a day and one of its matches (from `--history`, or Snowflake's history
without it), or type a match code. With Snowflake, the match's PLAY_OVERs and prod's quotes are
read as `scouting` reads them, and every answer is kept in `out/replay_cache`. A match read once
costs nothing again. With `--export`, nothing touches Snowflake. A match with no play feed can't be
loaded: it has no PLAY_OVERs to replay.

**The page:**

| part | what it shows |
|---|---|
| timeline | every PLAY_OVER: quarter, clock, score, the play, and prod's and v21's home win chance. Click one to jump to it |
| scoreboard | the score, who has the ball, down, distance and field position |
| controls | first / back / next / last, play at a chosen speed, "only states v21 prices", and "Price the whole match" |
| prices | moneyline, spread and total, v21 and prod side by side with the difference. The spread and total are shown at prod's line and at v21's own key-number line, with v21's ladder around it. Marked suspended where prod wasn't live |
| expected | v21's expected score, margin and total from here |
| distributions | v21's final margin and total, with prod's line marked |
| nudge | the expected points, the share, and what v21 makes of them (below) |
| pre-match | prod's closing quote against v21's kick-off price |
| chart | after "Price the whole match": the home win chance through the match, v21 against prod, with quarter lines. Hover for the numbers, click to jump |

Keys: ← / → step, Home / End jump to the ends, space plays and pauses.

**The nudge:** v21 starts from its own kick-off means (the gamers' and teams' effects in the
build). The nudge moves each side's offense along the build's strength direction until the
kick-off means sit `share` of the way from v21's own to the expected points (`v21.nudge_solve`, as
the price stream does). The same nudge then applies to every later state.

The expected points start as the prior's (`--v21-prior-model`: a walkthrough table or a v14 build),
read with only the results before the match. They can be changed by hand:
- "Apply" solves the nudge again and every price follows;
- "Back to the prior" undoes the change;
- the readout shows the target, v21's own means, the nudged means and the nudge itself.

Share 0, or no expected points, prices with v21's own means. A nudge needs a tilt build
(`v21-build --mode tilt`). With any other build, v21 prices off its own means and the page says so.

**What v21 can't price:** a PLAY_OVER with a conversion still to come has no v21 state, as in the
price stream. The page shows prod's quote there and says why v21 has none.

**Speed:** a price is `--paths` games (default 1000, about half a second). The whole-match chart
plays 400 games for each state in one run, a few seconds a match. Each state is played from a fixed
seed, so the same state prices the same each time.

### The priors against each other, offline: `prior-offline`

```bash
python -m eAMFCalibrator prior-offline --history eAMFCalibrator/out/match_history.csv \
    --since "2026-05-01 00:00:00" --until "2026-09-10 00:00:00" \
    --arm "nb2=eAMFCalibrator/out/nb2_history.csv" \
    --arm "glmer=glmer/out/cadence/cadence_matches.csv@trim:hl60_x0.5:daily" \
    --blend "avg=nb2+infer,glmer+both"
```

`prior-backtest` without prod, the bets or Snowflake: the same arms (`NAME=SOURCE[#MODE]`), read,
layered and scored the same way, graded against each other on the results alone. The match
history comes from a CSV (`--history`). Each arm runs as it is and with each layer (`+layer`,
`+infer`, `+both`, and `+feat`, `+both+feat` with the gamers' stats, `--features`, by default
player-stats' `out/player_match_stats.csv`). A blend (`--blend NAME=VARIANT,VARIANT`) averages its
variants' moneylines and expected totals.

Every variant is graded against the first arm as it is, match by match, on the matches every
variant priced after the layers' 7-day burn-in:
- **moneyline:** log loss and Brier (draws out), and the fortnights it beat the baseline;
- **total:** the expected total's RMSE and bias.

These are given for every match, the rematches, the first meetings, the first meetings after
either gamer had played someone else, each period (either side of the play-detail gap) and each
month. With no prod there's no over at a line and no bets: `prior-backtest` and `prior-replay` are
the tests against prod. Writes `out/prior_offline.txt` and `out/prior_offline.csv`.

### Every prior variant, audited, graded and replayed: `backtest-suite`

```bash
python -m eAMFCalibrator backtest-suite --since "2026-08-15 00:00:00" --until "2026-10-15 00:00:00"
python -m eAMFCalibrator backtest-suite --config suite.json --steps audit,backtest,replay,summary
```

One command runs the whole pre-match backtest. It makes the daily NB2 fits, checks every input for
anything a match couldn't have known, runs `prior-backtest` and `prior-replay` on every variant from
one read of the data, and ranks the variants. Everything goes to `out/suite/`. Each step's output
is kept, so a step can be rerun on its own.

| step | what it does |
|---|---|
| `history` | every settled match, for the fits: `match_history.csv`. `--history CSV` copies one you already have instead, with no Snowflake |
| `fits` | NB2 refitted every day of the window (`eAMFModel prior-daily`), one directory per half-life: `nb2_hl<D>/`. Days already fitted are kept. Only the days the history covers are fitted; a later day waits for a fresher history rather than being fitted short |
| `audit` | the inputs checked (below): `suite_audit.txt`. A FAIL stops the run; `--keep-going` carries on |
| `backtest` | the layers' leak check, then `prior-backtest` on every arm and the controls: `prior_backtest.txt/csv` |
| `replay` | `prior-replay` on the same arms, from the same read and fits: `prior_replay.txt/csv` |
| `summary` | every variant ranked: `suite_summary.txt/csv` |

**Warehouse credits:** every Snowflake query's answer is kept on disk in `out/suite/cache/`. A
later run asking the same query reads it from there, so rerunning a step or changing a setting
spends no credits. When every answer is already there, no connection is made at all. `--fresh`
empties the cache. The fits and the audit never touch Snowflake, so they can run while the
warehouse is unavailable:

```bash
python -m eAMFCalibrator backtest-suite --since "2026-08-15 00:00:00" --until "2026-10-15 00:00:00" \
    --steps history,fits --history eAMFCalibrator/out/match_history.csv
```

**The arms:** an NB2 arm per half-life (`nb2-hl30`, `nb2-hl60`, `nb2-hl120` by default), the glmer
arm (`glmer/cadence.R`'s daily trimmed model by default; cadence.R must have been run over the
window), and any others in the settings. Each comes as it is and with every layer: `+layer`,
`+infer`, `+both`, and with the gamers' stats `+feat` and `+both+feat`.

**What the audit checks:**
- **The history:** every match appears once, and the history reaches the window's end.
- **Each NB2 directory:** fitted at its half-life, with a fit for every day. Every day's training
  file (and its level scale's) holds only results from before that day. Every fit saw at least 99%
  of the results the current history has before its day; a fit made on an older history fails, and
  its day's directory should be deleted and refitted.
- **Each CSV arm:** the file is there and how much of the window it prices (warns below 95%: the
  backtest grades only the matches every arm priced). cadence.R's daily fits are cut at midnight.
- **The gamers' stats:** there, fresh (else rerun `player-stats`), and their coverage of the window.
  How many matches in each period had play detail.
- **The scramble check on the stats:** the stats from the window's middle day on are shuffled
  between matches and scaled at random. Every earlier match must read exactly as before.

**The layers' leak check** (the first arm): the results and stats from the window's middle day on
are scrambled the same way, and every layer is refitted. Every earlier match must move exactly as
it did, and the fitted score model's fits before that day must be unchanged. A layer reading a later
result, or a total or average that includes one, would differ. The same refit without the scramble
must reproduce every move (determinism).

**The controls** are checks on the pipeline, not candidates, and are listed apart:

| control | what it is | what it must show |
|---|---|---|
| `echo` | prod's closing moneyline and line read back as an arm (its implied margin and total, the normal reading) | grades as prod does. If not, the matches, sides or grading are out of line |
| `shuffled` | the first NB2 arm's prices shuffled between matches of the same day | grades well below its source. If not, the grading isn't reading the prices |

**The summary:** every variant is ranked by its moneyline log loss against prod, match by match.
Each line also gives:
- the over at prod's line, and the expected total's RMSE;
- the book's margin on the clean bets at the variant's prices less at prod's, with its 95% interval
  (resampling matches);
- the moneyline against prod at first meetings, at rematches, and in each period.

The periods default to before, during and after the play-detail gap (23 Sep – 9 Oct). The clean
bets follow, by market, then the controls' checks and the audit's tally. The grading is done again
from `prior_backtest.csv`, apart from `prior-backtest`'s own, and the two agree.

**The settings** (`--config`, a JSON object; anything left out keeps its default,
`backtest_suite.DEFAULTS`):

| setting | default |
|---|---|
| `nb2_half_lives` | `[30, 60, 120]` (days) |
| `nb2_mode`, `nb2_workers` | the NB2 arms' `#MODE` (`""`: their own score model); fits side by side (2) |
| `glmer` | `glmer/out/cadence/cadence_matches.csv@trim:hl60_x0.5:daily` (`""`: none) |
| `arms` | `{}`: more arms, `{"NAME": "SOURCE[#MODE]"}` |
| `features` | `out/player_match_stats.csv` (`"none"`: no feature layer) |
| `draws`, `boot` | 40,000 scorelines a match; the replay's resamples (config's) |
| `controls`, `leak_check` | on |
| `periods` | `[[label, from, until], ...]`: the detail gap's three |

`suite.json` records the settings, the window, the arms, each step's time and git commit, and the
audit's tally.
### Rematches the same day: `bets rematches`

```bash
python -m eAMFCalibrator bets rematches --since "2026-10-01 00:00:00"
```

This looks at what happens to the money when two gamers meet again the same
day. A day is the gamers' session (no gap over `SESSION_BREAK_MINUTES`), so a
night that runs past midnight UTC counts as one day. Each pair's meetings that
day are numbered 1st, 2nd, 3rd, 4th+.

Each rematch's teams are read against the day's first meeting:
- **same teams:** each gamer on the team he had then (the exact matchup);
- **swapped teams:** each gamer on the other's;
- **teams changed:** anything else.

The report has these parts:
1. **Money per match by meeting, teams and the first margin:** stake per
   match on the moneyline, handicap and total, and the book's margin. The
   first meeting's margin of victory is banded draw, 1–3, 4–7, 8–13, 14–20
   and 21+.
2. **The moneyline in rematches, against the first meeting's winner:**
   - prod's pre-match price for him at the first meeting and now (whether
     prod moved after the result), and how often he won again;
   - the share of the stake on him;
   - the book's margin and the residual (won less prod's p at the bet) on
     him and on his opponent.

   This is shown by the first margin, by teams, by meeting, pre-match
   against in play, and for restricted and VIP accounts.
3. **The handicap and the total:** the handicap money on the first winner,
   and the over and under money by the first meeting's total and margin.
4. **Prod's total line in rematches:** the line at the first meeting and at
   the rematch, and how far each total landed from it (its miss). This is
   shown by the first meeting's total and by the first meeting's miss.
5. **The fits: how far to move a rematch.** Each slope is the points per
   point of the earlier miss:
   - the rematch's total miss on the first meeting's (and, from the 3rd
     meeting on, the previous meeting's);
   - the first winner's margin miss on the first meeting's margin surprise.
     Prod's moneyline is read as a margin, with `MARGIN_SD` 9.35;
   - a logistic of the first winner winning again on prod's price (as an
     offset) and the first meeting's margin surprise.
6. **The rematch layer on prod's prices, walk-forward:** what
   `rematch_layer.py` moves, and the log loss, Brier and total RMSE of prod
   with and without it. These are compared match by match on rematches, by
   meeting and by teams.
7. **The 25 rematches with the most moneyline stake.**

**The rematch layer** (`rematch_layer.py`) works on any pre-match price:
expected points per side, or a moneyline and a total line.
- **The model:** each meeting's error against the price (margin and total)
  is the pair's effect that day plus noise. Two meetings the same day share
  the effect, so the covariance of their errors is its variance.
- **What is estimated:** that variance separately for meetings with the
  same teams and with teams changed. The rest of an error's variance is the
  noise.
- **What it does at a rematch:** the margin and the total move by the
  effect's best guess from the day's earlier errors (the posterior mean).
- **How it is fitted:** walk-forward, refitted each UTC day on the days
  before it, with every error centred on the price's own mean error. Nothing
  a match is priced with comes after its kick-off.

Writes `out/bets_rematches.txt` and `out/bets_rematches.csv` (one row per
match).

### What the gamers' other games that session say: `bets inference`

```bash
python -m eAMFCalibrator bets inference --since "2026-09-01 00:00:00"
```

Punters watch A beat B and C beat D, then back A against D (and C against
B). This is the same analysis as `bets rematches`, for what each gamer had
played that session against anyone else.

**What each match is read against:**
- **Others:** a gamer's earlier matches that session against anyone but
  this opponent. The last of them is his last other game: won, lost or drew,
  and by how much, or none (fresh).
- **The state:** both gamers' last other games, the form side first. "won v
  lost" is A v D above; "won v won" is both won theirs; "fresh v lost" is one
  who hadn't played anyone else against one who had lost.
- **The form side:** the gamer whose last other margin is bigger (0 when
  fresh). The gap is the difference (1-7, 8-14, 15-21, 22+). Equal margins
  are "level", with no form side.
- **A common opponent:** both gamers played the same third gamer earlier
  that session (the latest such). "beat v lost to" is one beat him and the
  other lost to him; the side is the gamer who did better against him.
- **The surprise:** a gamer's last other margin less the margin prod's
  moneyline implied for him then, so what that game said beyond what prod
  already knew.

**What it reports:**
1. **How much is bet,** by what the gamers had played (a rematch, a first
   meeting after others, or neither had), by the state, and won v lost by
   the gap.
2. **Where the moneyline money goes:** prod's price for the form side, how
   often it won (the residual, won less prod's p, 2se by match), and the
   share of stake and the book's margin on it and against it. This is cut by
   state for every match, first meetings and rematches; by the gap; by
   whether the form side kept the team from his last other game;
   pre-match, in play, Restricted and VIP.
3. **A common opponent,** the same way, by how each did against him.
4. **The handicap and the total:** the handicap money on the form side, and
   over and under by the gamers' last other games' total less prod's line
   then. The match's total against prod's line is also given by that.
5. **Fits on first meetings of the pair:**
   - the home margin's miss against prod on each gamer's last other
     surprise;
   - the total's miss on each gamer's last other total miss;
   - a logistic of the form side winning on prod's price (as an offset) and
     the surprise gap.
6. **The layers on prod's prices, walk-forward:** prod alone, with the
   rematch layer and with the inference layer. These give log loss and total
   RMSE match by match, by what the gamers had played, by state (first
   meetings) and by common opponent.
7. **The 25 matches with a form side and the most moneyline stake.**

**The inference layer** (`inference_layer.py`) is the rematch layer widened
to every match the two gamers played that session, against anyone.
- **The model:** a match's margin error is the home gamer's form that
  session, less the away gamer's, plus the pair's own effect that day, plus
  noise. Its total error is both gamers' pace that session plus the pair's
  plus noise.
- **What is estimated:** the form's variance comes from pairs of matches
  the same session sharing one gamer (the margins from his side), split by
  whether he kept his team. The pair's variance is what two meetings of a
  pair share beyond both gamers' form. The noise is the rest. A raw estimate
  below zero is reported, and used as zero.
- **What it does at a match:** the margin and total move by the best guess
  of the match's error from the earlier errors that session. These are both
  gamers' own matches, plus their opponents' other matches whose results were
  in, which separate a gamer's form from his opponent's. A beating B and D
  losing to C both move A v D.
- **How it is fitted:** walk-forward, each UTC day on the days before it, as
  the rematch layer is. With no form it gives the rematch layer's moves.

Writes `out/bets_inference.txt` and `out/bets_inference.csv` (one row per
match).

### The matches the book loses on the moneyline: `bets ml-losers`

```bash
python -m eAMFCalibrator bets ml-losers --since "2026-10-01 00:00:00"
python -m eAMFCalibrator bets ml-losers --since "2026-10-01 00:00:00" --candidate v9
```

A sweep of the moneyline over every settled match of the window. It asks
what kind of match is targeted, how the losing ones played out, and whether
there is any situation in which the money piles on. It reads every moneyline
bet against prod's probability of its selection where it was priced (bet time
less the operator's lag) and the game at that moment off SCOUTING_FULL.

**The residual** is won less prod's probability, bet by bet. It averages 0
wherever prod's price is right. A situation where it sits away from 0 is one
prod misprices. Its 2se is clustered by match, since one result settles every
bet of a match. With `--candidate`, each model's own residual sits beside
prod's. A model that already prices the situation brings it back to 0.

Picking out the losing matches picks on the result: the money was on the
winner. So the losing against the rest table says what those matches looked
like. The residual tables, read over every match, say whether a situation is
one to target.

The sections, in order:
1. **Where the money goes:** every bet by when it was struck against the feed
   (the phases the checks leave out are starred), then by phase, temperature,
   operator, and phase with the side's pre-match standing.
2. **The losing matches against the rest:** the worst 10% by the book's
   moneyline result, every losing match, and the winning ones, feature by
   feature (z: losing less winning over its se). The features are:
   - the money: stake, customers, in play, restricted, VIP, the biggest
     customer's share, pile-ons, the share on the eventual winner and the
     price it was backed at;
   - prod's prices: pre-match favourite, the winner's probability through
     the match;
   - the result: upset, comeback, lead changes, overtime, close or blowout,
     the final total against prod's closing line;
   - how Q1 and the first half were played: plays, real seconds between
     plays, clock and yards per play;
   - the gamers: a walk-forward Elo of every gamer's wins (slow: the long
     run; fast: recent form) against prod, and their places in their
     sessions.

   The same table then compares the most-bet fifth of matches with the rest
   (what the money targets, the result aside). Then the matches by class,
   stake, favourite, total against the line, winner, comeback, match of the
   session, kick-off hour and stream.
3. **In play, by situation:**
   - quarter and the backed side's lead, and prod's probability;
   - who had the ball, where, and the down and distance (off the latest
     PLAY_OVER at bet time);
   - the feed's moment, and the points against prod's pre-match total so far
     (is a low scorer read early?);
   - prod's move over the 60s before and after the bet, and who scored next
     and how soon;
   - the gamers' sessions, temperature, stake and the price's age;
   - restricted and VIP alone.
4. **In play, the features ranked:** every numeric feature cut in quintiles,
   ranked by how far the residual moves from the bottom bin to the top, with
   the top eight shown bin by bin.
5. **Pre-match**, the same.
6. **Pile-ons:** a run is the moneyline bets on one side of a match, none
   more than 45s after the last. A pile-on is a run with 3+ bets and its stake
   in the top 5% of runs. This section gives:
   - the situation at a pile-on's first bet against all the in-play money
     (lift);
   - what came next;
   - the 40 biggest, one by one.
7. **Gamers and teams:** backed and opposed, costliest to the book first.
   Then the 20 customers who took the most, and how they bet.

Writes `out/bets_ml_losers.txt`, `out/bets_ml_losers.csv` (one row per bet),
`out/bets_ml_matches.csv` (one row per match) and `out/bets_ml_pileons.csv`.

### Where the money goes through a session: `bets sessions`

```bash
python -m eAMFCalibrator bets sessions --since 2026-09-10 --until 2026-09-23
```

This answers whether the bets and stake are flat across a gamer's session, or
lumped on particular matches of it. It reads every single the bet source
gives (moneyline, handicap and total, pre-match and in play) and the match
history. It needs no prices or play detail. Each bet counts once for each of
the two gamers in its match, at that gamer's place in his own session. Two
places are used:
- his match of the session: 1st, 2nd … 10th, 11+;
- matches left in it: 6+, 4–5, 3rd last, 2nd last, last.

The denominator is every settled match of the window at that place, bet on or
not. Per group (All, Restricted, VIP, Standard, none), and per market for All,
Restricted and VIP, each place gets:
- gamer-matches and their share;
- bets and stake, and their shares;
- stake per gamer-match, and that against the group's average (`vs avg`, 1.00x
  is flat);
- the book's margin, and the share on the under for totals.

A place where the stake share runs above its share of gamer-matches is where
the money is lumped. Writes `out/bets_sessions.txt`.

### What the money already bet says: `money-layer`

```bash
python -m eAMFCalibrator money-layer --since "2026-10-01 00:00:00" --until "2026-10-13 00:00:00" --base prod --history eAMFCalibrator/out/match_history.csv --bets eAMFCalibrator/out_bets_v21_4k/bets_sim.csv
python -m eAMFCalibrator money-layer --since "2026-09-01 00:00:00" --until "2026-10-13 00:00:00" --base eAMFCalibrator/out/walkthrough/priors/avg.csv
```

On the moneyline the bettors beat prod by about 9.5 points of the operator's
margin. All of the gap is informed money:
- bets under 10 lose at the margin prod's prices say they should (10.4% in
  play against 10.5%);
- bets of 200 and up carry nearly two-thirds of the shortfall;
- most of it is on a handful of gamers whose level jumped beyond chance from
  September to October, with the money on them from their first days.

This tests two ways of using that money.

**The days before.** Each gamer's pre-match price is moved by their backers'
edge on the gamer's matches settled before this one, walk-forward:
- **the edge:** each bet's result less the price it was laid at, which is
  prod's probability at the bet with `--bets`, or else the operator's odds with
  its margin taken out (the bet table alone; the two agree to three places);
- **the signal:** stake-weighted, over the stake plus `--shrink` (default
  50,000), settled by the match's kick-off (`--cut daily`: by the start of its
  day);
- **the slope:** fitted on the days before;
- **the price move:** goes into a copy of the base table as `p_home` and the
  points (over the nudge share, so v21's kick-off lands on the whole move).
  `money.csv` reads like any other table with `--v21-prior-model`.

The test grades base and money on the results: Brier and log loss with 95%
intervals over days, the slope on the test days, fifths of the signal, each
week, and the settings side by side. On Oct 1–12 the Brier moved by −0.0003
[−0.0035, +0.0023]; even the best slope picked in hindsight only reached
−0.0018. That is no material gain.

**The match so far** (with `--bets`). Each moneyline price is moved in logit
by beta × the money already on its side of the match:
- **the money:** (this side − the other) / (both + 500);
- **beta:** fitted on the bets of the days before;
- **the outputs:** the same bets re-priced as the betting sim does, and two
  books, with 95% intervals over matches, for prod and each candidate in the
  file.

On Oct 1–12 this re-priced prod's bets by +2.3 points [+1.5, +3.2] and v21's
by +1.7 [+1.1, +2.4]. That is the only first-pass gain with an interval clear
of zero, and it comes from following the money, not from the model: about a
quarter of the gap. Pricing that way live needs the bets as they are placed.

Writes `money_layer.txt`, `money_gamers.csv` and (for a table base)
`money.csv` to `--out` (default `out/money/`).

### The pre-match report, no simulation: `prior-bets`

```bash
python -m eAMFCalibrator prior-bets --bets eAMFCalibrator/out_bets_v21_4k/bets_sim.csv --sim-name glmer-sim --table nb2=eAMFCalibrator/out/walkthrough/priors/nb2.csv --table glmer=eAMFCalibrator/out/walkthrough/priors/glmer.csv --table avg=eAMFCalibrator/out/walkthrough/priors/avg.csv
```

This sets prod against each walkthrough table's own price and against a
sim's pre-match price, all on the same bets, with no simulation run here.
- **A table's price:** each row's score model rebuilt about its `home_pts` /
  `away_pts`, read at the bet's line as `prior-replay` reads an arm. That is
  NB2's shared-gamma joint model for `nb2` and the averages, and NB2's
  construction fitted to glmer's errors for `glmer`.
- **The sim's price:** the bets file's candidate, at its kick-off price
  pre-match. Its bets are re-priced as the betting sim made them. Point
  `--bets` at a run of v21 on a glmer build for the glmer-driven sim.

**The grades.** These are for the moneyline, on every match with a result:
- Brier and log loss, each less prod's, with 95% intervals over days;
- the same by how heavily the match was bet (quarters of its moneyline
  stake), and weighted by stake;
- each price's disagreement with prod as a second term on the result (1:
  take its whole view, 0: none of it).

The spread and the total are graded at prod's pre-match line, using each
match's last pre-match bet in the market.

**The bets.** Each bet is re-priced as the betting sim does (odds × prod /
price, the operator's margin kept), on the same bets and in two books, with
95% intervals over matches. The groups are:
- **every pre-match bet, by market:** the clean ones first, placed once both
  gamers' previous results were in;
- **the kick-off money:** in-play moneyline bets struck while prod's price was
  still within 2 points of its pre-match one, the game's first 15 seconds
  apart. Here a table's price is moved by prod's small move since kick-off.

Writes `prior_bets.txt` and `prior_bets.csv` (a row per match: the result, the
moneyline stake and every price) to `--out` (default `out/prior_bets/`).
