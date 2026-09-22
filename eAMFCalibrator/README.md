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

The HTML panel is **two tables**: the combined reading over every pair,
then the two halves it is made of.

| Reading | Pairs | Matches | On prob | On line | Cand win | Match vote | Level | p |
|---|---|---|---|---|---|---|---|---|
| Overall | every decided pair, under the rule that applies to each |

| Reading | Pairs | Matches | Cand win | Match vote | Level | Δ | 95% CI | p |
|---|---|---|---|---|---|---|---|---|
| Same line | Δ is ΔBrier per match |
| Different line | Δ is Δpoints per match |

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

### Why the ramp darkens as well as reddening

Green and red are the one pair red-green colour blindness cannot
separate, so hue alone cannot carry the ordering. The five steps also
fall monotonically in OKLCH lightness — 0.93 → 0.67 on the light theme,
0.27 → 0.53 on the dark one — with every adjacent gap clearing the 0.06
floor. A reader who sees no hue at all still sees the cell darken as the
gap grows.

Each theme's steps are picked against its own surface rather than
flipped from the other's, and every step clears 4.7:1 against the ink
sitting on it. The number is the real answer; the colour only says which
band it is in, and a key above each table names every band, so nothing
is ever read by colour alone.

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
| `score` | good | one to five points: a field goal or a conversion |
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

### Which way a price should move

A team-sided selection follows the side it names: on a good play the
offence's own moneyline and spread rise, its opponent's fall. Totals are
possession-blind — points are points whoever scores them — so Over follows
the play and Under opposes it from either side.

Scored on the **sign**, not the size. `|Move|` and `Signed` report the size
beside it, so a model that gets the direction right on a touchdown and a
five-yard gain by the same amount is visible as such.

### What is refused

- A transition whose **line moved** between the two messages. A different
  line is a different question, exactly as it is on a pair.
- A transition where **either endpoint was not live**. A price nobody could
  have taken did not move.
- A **flat** price is neither right nor wrong. It is reported as its own
  column rather than counted as a miss — silence is a finding, not an error.

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

### Output

`indrive_moves.csv` is one row per (transition × selection × stream),
carrying the whole transition so it needs no join. `indrive_transitions.csv`
is one row per transition. `indrive.html` is the same tables on a page,
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
