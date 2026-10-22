# Code sweep, 21 Oct 2026

A review of the live pricing code on main at a7c4639: eAMFModel (v2.1, its stream, the trader and the
shared modules), eAMFCalibrator, glmer, nb2 and xgb. The legacy v14/sim14 files and `analysis/` were
left out. Sixteen reviewers took one area each, looking for bugs and methodology problems, and tested
what they suspected with short scripts on synthetic matches (`eAMFModel/tests/fakes.py`) or on the run
outputs in the repo (`eAMFCalibrator/out_bets_*/bets_sim.csv`, the walkthrough priors, the AMFELO
history). The high findings were then re-checked against the code.

- `report.html`: the ranked report, with what to fix first. Findings carry IDs (A1, B3, ...).
- `notes/`: each area's findings in full, with file and line, evidence, impact and fix.
- `scripts/`: the scripts behind the evidence, one folder per area (`verify/` holds the re-checks).
  They were run from the repo root with `PYTHONPATH=.`; some write to, or read from, the session's
  scratch folder named at their top, so point that at a folder of your own to re-run them.

| notes | area |
|---|---|
| A | v21's build and fitting (`v21.py` up to the solves) |
| B | v21's simulation, `v21_rules.py`, `game.py` |
| C | v21's fitted layers: react, react-fit, timing, endgame, gamers, coverage |
| D | the pricing stream and the market maths |
| E | the feed and data loading |
| F | the priors in Python (walkthrough, form, pace, layers) |
| G | glmer and nb2 |
| H | the xgb folder |
| I | the evaluation statistics (bets, prior-bets, margin uplift, book comparison) |
| J1, J2 | the bet diagnostics |
| K | the calibrator's adjustment layers |
| L1, L2 | the in-play analyses and the report |
| M | the trader and replay pages, and the command lines |
| N | look-ahead and evaluation integrity across the pipeline |

Severity: High changes a price or a headline result materially, or crashes a normal run; Medium is
wrong in a realistic but narrower case, or biases a result modestly; Low is an edge case. VERIFIED means
a run showed it, REASONED that it was read from the code.

## Fixed since

- D2, bet times read in the machine's local time: every conversion now goes through `bets.epoch`, which
  reads a naive time as UTC (also the form rating's and xgb features' day counts).
- A1, A5, and A3 labelled: the bets report leads with the change at the bets' own lines (odds alone), the
  bets moved to a candidate's line apart, each with a 95% interval over days, and "leaning with the money"
  beside it; the book comparison, the uplift, the pre-match test and prior-bets resample days. The uplift
  at the bets' moments is labelled as favouring the candidate (`--all-moments` stays opt-in, as it is
  several times slower). A2 isn't fixed by code: the same-bets change still credits any model that errs
  differently from prod, and the report says so; the fair tests are the two books and the uplift with
  `--all-moments`.
- B2, B3, B4: auto_end counts the play clock only while the clock runs, and the two-minute warning once;
  every start carries whose overtime possessions are done (`v21.ot_history`, the trader's
  `game.ot_had_ball`); the stream quotes a selection that won k of n games at (k + 1/2) / (n + 1).
