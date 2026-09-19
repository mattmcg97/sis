# eAMFCalibrator

Calibration suite for the GAMEPLAI in-play models on eAMF (Competitive
Gaming American Football). Takes either stream as input, splits the data
into cells, and asks the only question that matters for a probability:
**when the model said 30%, did it happen 30% of the time?**

```
py -m eAMFCalibrator preflight              # check tables + the play clock
py -m eAMFCalibrator report                 # everything, one HTML (start here)
py -m eAMFCalibrator directional            # paired head-to-head
py -m eAMFCalibrator cross                  # cell view, under the line rule
py -m eAMFCalibrator run prod
py -m eAMFCalibrator run candidate
py -m eAMFCalibrator run both               # runs both, then diffs them
py -m eAMFCalibrator compare out/prod_cells.csv out/candidate_cells.csv
```

Four views:

- **`report`** — one pairing pass, both console views, and a single combined
  HTML: directional headline, per-market clustered tests, the integrity checks,
  the cross-sectional cells, and **every paired observation** at the bottom
  with both streams' line, price, probability, outcome and error, sorted by
  widest probability disagreement. Column headers sort. Around 600 bytes per
  pair row, so a two-day window lands near 4 MB.


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

## Directional comparison

```
py -m eAMFCalibrator directional            # console + one-screen HTML
py -m eAMFCalibrator directional --html report.html
```

Pairs prod against candidate at each drive-start snapshot and reports both
halves of the comparison, because they do not always agree.

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

## The split axes

| Axis | Buckets |
| --- | --- |
| Score difference | `<= -9`, `-8..-3`, `-2..+2`, `+3..+8`, `>= +9` |
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

**Score difference is `PLAYER_1 − PLAYER_2`** (home minus away), so `>= +9`
means the home side leads by 9+ regardless of who has the ball. Possession
is a separate axis, so the home-leading-while-defending case is still
visible as its own cell.

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

133 tests covering line parsing, market resolution, bucket edges, drive
cleaning, clock reconstruction, quote matching, message pairing, the sign
test and the paired-delta machinery. No Snowflake needed — the database
half is exercised separately against a mock shaped like the real schema,
play feed with no clock included.

Two tests pin known traps directly: a candidate much closer on half the
pairs and barely further on the rest sits at a 50% win rate while the paired
loss difference is clearly positive; and grading both streams against one
stream's line inverts the winner on a total that falls between the two
lines.
