# eAMFCalibrator

Calibration suite for the GAMEPLAI in-play models on eAMF (Competitive
Gaming American Football). Takes either stream as input, splits the data
into cells, and asks the only question that matters for a probability:
**when the model said 30%, did it happen 30% of the time?**

```
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

`preflight` now prints what those columns really contain. On the
2026-09-17 → 09-21 window, per stream:

| STATUS | IS_ACTIVE | Rows | Share | Read as |
|---|---|---:|---:|---|
| `UNDER SETTLEMENT` | false | ~985k | 83% | not live |
| `open` | true | ~178k | 15% | **live** |
| `CLOSED` | false | ~14.5k | 1.2% | not live |
| `CLOSED` | true | ~1.7k | 0.1% | not live |

**There is no suspension in this feed.** Markets run `open` →
`UNDER SETTLEMENT` → `CLOSED`, and only ~15% of published rows are open.
So the state is reported verbatim rather than mapped onto a vocabulary the
feed does not use. `CLOSED`/`true` is an inconsistent combination and is
read as not live: it is not open, whatever `IS_ACTIVE` claims.

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

258 tests covering line parsing, market resolution, bucket edges, drive
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
