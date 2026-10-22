"""bet_lag.held_out sets each match's line-fitted lag against the operator's ODDS-fitted lag. When the
operator's line lag differs from its odds lag but is the same in every match, 'a lag per match' still
wins held out: the baseline is the wrong one-lag model, not one lag per operator off the lines."""
import datetime as dt

from eAMFCalibrator import bet_lag, bets
from eAMFCalibrator.tests.test_bet_lag import T0, GRID, _timeline, _bet

bets.config.BET_GROUP_COLUMN = "OPERATOR_NAME"
tl, all_bets = {}, []
for k in range(6):
    match = f"AF00{k}200926"
    tl.update(_timeline(match, (100 + 7 * k, 250 + 5 * k, 400 - 3 * k)))
    step = bet_lag.line_steps(tl)[(match, 54)]
    for second in range(60, 560, 2):
        line = bet_lag.line_at(step, T0 + dt.timedelta(seconds=second - 9))    # every match 9s behind
        all_bets.append(_bet(match, second, line))
bl = bet_lag.per_bet(all_bets, bet_lag.line_steps(tl), tl, {}, GRID)
for anchor in (5, 9):
    n, one, own = bet_lag.held_out(bl, GRID, {"OP": anchor})["OP"]
    ms = bet_lag.per_match(bl, GRID, {"OP": anchor})
    print(f"operator's (odds) lag {anchor}s: held out on {n} bets  one lag {100 * one:.1f}%  per match {100 * own:.1f}%"
          f"   per-match lags {sorted(m.best for m in ms.values())}")
