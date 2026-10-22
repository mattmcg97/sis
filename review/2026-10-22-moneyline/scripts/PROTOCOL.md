# Pre-match moneyline study: shared protocol

Goal: a pre-match P(home wins) more accurate than what we have. v2.1 will then be nudged at kick-off
so its own moneyline meets it (the "ML nudge"). Score: held-out log loss first, then Brier and the
calibration slope (logit of y on logit p: 1 calibrated, below 1 too spread out).

## Layout (ML = this folder)
- `harness.py`  load(), walk_forward(), metrics(), report(), boot_diff(), write_preds(), read_preds(). Read it first.
- `features.csv`  xgb/features.py with the drive stats (built from the repo's own data; see harness docstring).
- `preds/<agent>_<model>.csv`  every serious model's walk-forward p_home (code, p_home), 2026-06-01 to 2026-10-18
  wherever the model can price it. The stacking agent and the final report read these.
- `scripts/<agent>/`, `out/<agent>/`, `notes/<agent>.md`  each agent's own.

## Rules
1. Nothing after kick-off. Features in features.csv are as of kick-off (results count 36 minutes after their
   kick-off). Model fits: `harness.walk_forward` (fit on matches settled by the day's midnight UTC, refit daily
   or every few days). Anything you build yourself must respect the same cut.
2. Tune on TUNE (2026-06-01 to 07-31) only; freeze the settings; report on TEST (2026-08-01 to 10-18). The
   walkthrough / form prior tables exist only from 2026-09-01: models that need them tune on 09-01 to 09-21 at
   most and report on SEP (09-01 to 10-18) with the burn-in said plainly; better, report them on 09-22 to 10-18
   too. Never choose anything by its TEST score.
3. `prod` (prod_p_home: prod's last pre-match moneyline quote) is a benchmark, never a feature.
4. Ties (15 matches) are left out of every score (y is NaN).
5. Compare on common matches (harness.report(common=True)), with the day-resampled 95% interval.
6. The repo is read-only for the modelling agents. Do not call any outside API or website (the h2h stats site
   blocked the user's office last time); everything needed is local.
7. Four cores are shared: XGBoost nthread<=2, joblib n_jobs<=2, R single-threaded. A run over ~10 minutes goes
   in the background with its progress to a log you check.

## Baselines (harness, common matches)
| window | model | log loss | Brier | AUC | slope |
|---|---|---|---|---|---|
| TUNE Jun-Jul (5,462) | prod | 0.6764 | 0.2418 | 0.605 | 1.13 |
| | form rating, normal sd 9.35 | 0.6773 | 0.2413 | 0.623 | 0.63 |
| TEST Aug-Oct (6,170) | prod | 0.6848 | 0.2460 | 0.569 | 0.93 |
| | form rating, normal sd 9.35 | 0.6963 | 0.2506 | 0.580 | 0.46 |
| SEP Sep-Oct (3,426) | prod | 0.6894 | 0.2482 | 0.550 | 0.68 |
| | walkthrough avg (its own NB2 joint ML) | 0.6777 | 0.2425 | 0.593 | 1.11 |
| | walkthrough glmer_both | 0.6774 | 0.2423 | 0.595 | 1.10 |
| | form table avg_form4_w05 (v2.1's prior today) | 0.6828 | 0.2449 | 0.584 | 0.71 |
| | form rating | 0.7013 | 0.2529 | 0.569 | 0.40 |
A coin flip scores 0.6931. The skill is small, so differences of 0.001-0.003 matter; a paired interval over
days is about +-0.003 on TEST. ELO (AMFELO's starting ratings) looks sequential (97% of starts equal the
gamer's previous end) and doesn't beat prod in any window, so it is not obviously leaking; treat it as a
feature to test, not a baseline to trust.
