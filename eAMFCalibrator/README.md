# eAMFCalibrator

Calibration suite for the GAMEPLAI in-play models on eAMF (Competitive
Gaming American Football). Takes either stream as input, splits the data
into cells, and asks the only question that matters for a probability:
**when the model said 30%, did it happen 30% of the time?**

```
py -m eAMFCalibrator preflight              # check tables + the play clock
py -m eAMFCalibrator directional            # paired head-to-head (start here)
py -m eAMFCalibrator run prod
py -m eAMFCalibrator run candidate
py -m eAMFCalibrator run both               # runs both, then diffs them
py -m eAMFCalibrator compare out/prod_cells.csv out/candidate_cells.csv
```

Two different questions:

- **`directional`** — at the same snapshot, on the same selection, which
  model was closer to the result? Paired, so the between-snapshot variance
  cancels. This is the one a day of data can answer.
- **`run`** — is this model's 30% really 30%? Needs many snapshots per cell
  before a realized rate means anything.

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

Pairs prod against candidate at each drive-start snapshot and reports both
halves of the comparison, because they do not always agree.

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

84 tests covering line parsing, market resolution, bucket edges, drive
cleaning, clock reconstruction, quote matching, message pairing, the sign
test and the paired-delta machinery. No Snowflake needed — the database
half is exercised separately against a mock shaped like the real schema,
play feed with no clock included.

One test pins the win-rate blind spot directly: a candidate that is much
closer on half the pairs and barely further on the other half sits at a 50%
win rate while the paired loss difference is clearly positive.
