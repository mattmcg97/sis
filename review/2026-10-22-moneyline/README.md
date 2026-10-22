# Pre-match moneyline study, 22 Oct 2026

The question: can a model that predicts the moneyline directly beat the one we read off expected points, and
can v2.1 use it as a calibration? Four ideas were tested side by side, then two from the earlier design:
- a margin model with a fitted link;
- a plain Bradley–Terry check;
- the h2h site's stats (red zone and the rest);
- XGBoost;
- totals;
- an in-play win-probability benchmark.

Two pieces of code came out of it:
- **the moneyline nudge** (`v21-ml`): the sim's kick-off moneyline meets a model's;
- **`ml-prior`**: writes the model's table.

## The answers

1. **Margin plus a fitted link: the link works, but the margin matters far more.** On the drift-4 form
   rating, the link takes Aug–Oct log loss from 0.6961 to 0.6826.
   - That still trails a slower margin by 0.009, because the rating's margins are too noisy, not only too
     spread.
   - Its fitted shrink keeps drifting: 0.58 in June, 0.66 in August, 0.45 by mid-October. A link always lags
     that.
   - On TUNE, log loss improves steadily as the rating's drift falls from 16 to 0.05 points²/day. Drift 4 is
     about 16 times too fast.
   - On a good margin the link is worth only 0.0002–0.0008, because such a margin is already calibrated. The
     best margin is a ridge on margin:
     - gamer and team strengths, and a home edge per stream;
     - results weighted by a 30-day half-life;
     - solved again as each result comes in.
   - None of the extras earned a place on it: rematch, form gap, stream, uncertainty, total. Each was within
     ±0.0004 on TUNE and on TEST.
   - The home edge belongs in the margin, per stream: stream 1 is about +0.8 points, streams 2 and 3 about 0.
2. **Bradley–Terry check: the margin adds information.**
   - Plain BT (home and gamers, decayed, refit daily) scores 0.6791 on Aug–Oct.
   - The ridge link scores 0.6735: −0.0055 (−0.0091, −0.0019).
   - The best model scores 0.6722: −0.0068 (−0.0105, −0.0031).
   - Win/loss models catch up only by borrowing what the margin gives: teams, same-day updating (Glicko-style,
     0.6740), or margin-of-victory weights (0.6731).
   - A binomial GLMM (lme4) is weaker (0.6776). In October its gamer spread collapses.
   - Your AMFELO file shows no leakage, but it is weak even with a fitted link (0.6799).
3. **The h2h site's stats add nothing beyond a points-based strength.** They were rebuilt from our own
   play-by-play: red zone, 3rd and 4th down, yards per play, success, big plays, turnovers, field position.
   - Each was opponent-adjusted and shrunk by its own Kalman filter.
   - On their own they beat prod (0.6776).
   - On top of strength they change log loss by +0.0000 (−0.0007, +0.0007) on Aug–Oct.
   - The repeatable ones are turnovers per drive, success rate and field position.
   - Red-zone TD %, 3rd-down and 4th-down rates are mostly noise: split-half r is 0.18–0.29 over 25 matches.
   - They help only gamers with fewer than 100 matches, about 5% of matches. That was spotted on TEST and
     confirmed on TUNE, so treat it as exploratory.
   - Rushing/passing splits aren't in our data. First downs could be rebuilt from the scouting export.
4. **XGBoost doesn't make the cut.** A walk-forward 4-input logistic beats every XGBoost variant: classifier,
   paired, margin plus link, and corrections on a strong base.
   - The best XGBoost is 0.6778, against 0.6722 for the best model.
   - The drive stats carry 7% of its gain and add nothing.
5. **Combining doesn't help.** Walk-forward logistic stacks of the strongest models are level with the best
   single model, within 0.001 either way on every window (`results/stack.txt`).
6. **Totals: our over/under beats prod's at prod's own line by 0.018 log loss on Aug–Oct (−0.0233, −0.0124).**
   - It does so in every window. Prod's O/U is barely better than a coin flip, and its lines sit slightly low.
   - The best expected total is an online Kalman filter on the total: RMSE 12.47 against prod's line's 12.84.
   - Drives × points per drive is no better than modelling the total directly.
   - The total's spread grows with the expected total, not with pace.
   - Margin and total errors are almost uncorrelated (−0.03), so the moneyline needs no joint model.
7. **In play (a rough check from here):** the bets here carry no possession, field position, distance or
   timeouts. So a win-probability model on what they do carry loses to both live prices by 0.025–0.035.
   - Two things came out of it:
     - On Oct 11–18, at the moments bets were placed, v2.1's in-play moneyline beat prod's by 0.0064 (−0.0123,
       −0.0009) with intervals over days; over matches the interval includes zero.
     - In October, v2.1 underrates the side with the ball trailing by 1–3 in the last two minutes. They won
       57–75%; v2.1 gave them 43–55%. September agreed, and `out_v21_endgame` shows the same.
   - `scripts/G/inplay_wp.py` builds the full model off the scouting export on your machine.

## The leaderboard

Every model was walk-forward: fitted on results in by each day's midnight, its inputs as of kick-off. Settings
were tuned on Jun–Jul and frozen. Scores are on the matches every column prices; intervals resample whole days.
Full tables are in `results/leaderboard.txt`.

| Aug 1 – Oct 18, 5,805 matches | log loss | vs prod (95%) | calibration slope |
|---|---|---|---|
| avg table + ridge, linked (`A_best`, what `ml-prior --table` writes) | **0.6722** | −0.0117 (−0.0161, −0.0073) | 1.02 |
| walkthrough avg's own moneyline (extended to August) | 0.6725 | −0.0115 (−0.0158, −0.0073) | 1.13 |
| ridge margin, linked (what `ml-prior` writes without a table) | 0.6735 | −0.0104 (−0.0148, −0.0061) | 0.99 |
| Kalman points strength (C) | 0.6736 | −0.0104 (−0.0150, −0.0058) | 0.98 |
| margin-of-victory Kalman (B) | 0.6731 | −0.0109 (−0.0162, −0.0058) | 0.85 |
| Glicko-style win/loss (B) | 0.6740 | −0.0099 (−0.0153, −0.0049) | 0.89 |
| plain Bradley–Terry (B) | 0.6791 | −0.0049 (−0.0088, −0.0011) | 0.98 |
| h2h stats alone (C) | 0.6776 | −0.0063 (−0.0108, −0.0020) | 0.99 |
| best XGBoost (paired, D) | 0.6778 | −0.0061 (−0.0111, −0.0015) | 0.97 |
| drift-4 form rating, linked | 0.6826 | −0.0013 (−0.0068, +0.0038) | 0.79 |
| drift-4 form rating, normal sd 9.35 | 0.6961 | +0.0121 (+0.0040, +0.0196) | 0.46 |
| prod's last pre-match quote | 0.6840 | | 0.97 |

On Sep 1 – Oct 18, the form table v2.1 nudges toward today (`avg_form4_w05`) scores 0.6822. The best model is
0.0062 better (−0.0101, −0.0020), and the walkthrough avg table's own 0.0057 better. A coin flip scores 0.6931.

## Using it

1. Write the table. Every match of the history is priced walk-forward, including scheduled ones, in about 10
   seconds:

       python -m eAMFCalibrator ml-prior                                  # the ridge and link alone
       python -m eAMFCalibrator ml-prior --table eAMFCalibrator/out/walkthrough/priors/avg.csv   # stacked with avg

   `out/ml_prior/ml_prior.csv` carries:
   - `p_home`, for the moneyline nudge;
   - `exp_margin`, `exp_total` and `Pred_P1_Points` / `Pred_P2_Points`, for the points nudge.

   So one file serves both switches.

   The stack needs at least 300 matches the table priced in the 60 days before a day. With the repo's avg
   table, which starts Sep 1, it starts Sep 5. A walkthrough reaching back further gives it a running start.
2. Price v2.1 with its kick-off calibrated to it:

       python -m eAMFCalibrator bets run --candidate v21,v21-ml --v21-ml eAMFCalibrator/out/ml_prior/ml_prior.csv

   - `v21-ml0.5` goes half way from the sim's own moneyline.
   - `v21-drives-ml` adds the pace nudge.
   - The replay page and the trader take `--v21-ml` / `--ml` and an ML share.
   - The trader also takes a typed home-win chance.

   Judge it with the two-book comparison and `--all-moments` against prod, not the same-bets reprice (see the
   21 Oct sweep).

What the nudge does: the points nudge's target keeps its total. Its margin is moved until the sim's kick-off
P(home wins), with a tie counted as half, meets `p_home`. It takes four steps on the same random numbers.

On the synthetic build at the stream's 1,000 paths:
- The kick-off costs 32% more CPU, or 17% with the pace nudge on.
- It meets the target within 0.005 on average, 0.021 at worst (the noise at 2,000 games is ±0.011).
- The total stays within 0.06 points, or 0.2 with pace on.
- In-play states cost nothing extra: they carry the kick-off's nudge.

## Still to do on your side

1. A bets run with `v21-ml` (or `v21-drives-ml`) against `v21` and prod, judged by the two books and
   `--all-moments`. The study says the pre-match moneyline should improve by about 0.006 against the form
   table. Only a run shows what the nudge does to the in-play prices that follow.
2. Consider pointing v2.1's points at `ml_prior.csv` (`--v21-prior-model`) or at the walkthrough avg table
   instead of `avg_form4_w05`. The form table is the problem the link was meant to fix. Its margins come from
   the drift-4 rating, which the results weight at 3–8% of avg's; form_prior gives it 50%.
3. Run `scripts/G/inplay_wp.py` on the scouting export, for a proper in-play benchmark with possession, field
   position and timeouts. Look at the late-game finding (v2.1 underrating a side with the ball down 1–3) with
   v21-endgame on October.
4. The over/under: the sim makes its own total distribution. Check whether its O/U at prod's lines is as good
   as the study's lumpy NB (`notes/F.md`).

## What's here

- `notes/`: each agent's findings in full, with method, tuned settings and tables.
  - A: margin and link.
  - B: Bradley–Terry, Elo, Glicko, GLMM, AMFELO.
  - C: h2h stats.
  - D: XGBoost.
  - F: totals.
  - G: in play.
- `results/`: the reports behind the notes, `leaderboard.txt`, `stack.txt` and `ml_prior.txt`. The last is the
  command's own report on the full history: Jan–Oct log loss 0.6598, calibration slope 1.03, the link month by
  month.
- `scripts/`:
  - `harness.py`: the shared loader, walk-forward loop and scorer.
  - `PROTOCOL.md`: the rules every agent followed.
  - `compare.py` and `stack.py`.
  - each agent's scripts.

  They read the session's scratch folder named at their top (features.csv was built with `python -m
  xgb.features --history eAMFCalibrator/out/match_history.csv --drives xgb/data/drive_stats.csv`). Point that
  at a folder of your own to re-run them.

No outside site was called: the h2h site's stats were rebuilt from `xgb/data/drive_stats.csv`. Agent E (the
nudge) and agent H (`ml-prior`) wrote code, not notes; their commits say what they did.
