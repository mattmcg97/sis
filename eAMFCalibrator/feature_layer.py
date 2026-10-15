"""The feature layer: a pre-match price's margin and total moved by what the two gamers' stats say
beyond it (gamer_features.py: yards a play, 3rd downs, the red zone, turnovers, possession, pace,
points), fitted on the price's own errors.

Each UTC day, on every earlier match of the window with a result:

    margin error = b' (home's stats less away's) + noise        total error = c' (the two added) + noise

a ridge regression on the stats standardised (each to its mean and sd on the training matches), the
errors centred on their mean -- so the layer moves a price by what the stats say about it, not by
the prior's overall bias, as the rematch layer is centred. RIDGE pulls every coefficient towards 0
(in standardised units, a ridge of RIDGE matches' worth). No move until MIN_TRAIN matches have been
seen, and none for a match without features. `base` fits the layer on the errors left after another
layer's moves (the rematch and inference layers' +both), so the two stack without counting the same
thing twice.

Every match is priced off a fit made on the days before its own, with features read before its day.
"""

from collections import defaultdict

import numpy as np

from . import gamer_features

RIDGE = 50.0
MIN_TRAIN = 400


class Fit:
    """One ridge fit: coefficients on the standardised stats, the standardisation, their standard
    errors and the matches it was fitted on."""

    def __init__(self, beta=None, mean=None, sd=None, se=None, n=0):
        self.beta, self.mean, self.sd, self.se, self.n = beta, mean, sd, se, n

    @classmethod
    def fit(cls, X, y, ridge=RIDGE):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n, p = X.shape
        if n < p + 2:
            return cls()
        mean, sd = X.mean(axis=0), X.std(axis=0)
        const = sd < 1e-9                             # a stat that never varied: no coefficient
        sd[const] = 1.0
        Z = (X - mean) / sd
        Z[:, const] = 0.0
        yc = y - y.mean()
        A = Z.T @ Z + ridge * np.eye(p)
        Ainv = np.linalg.inv(A)
        beta = Ainv @ Z.T @ yc
        resid = yc - Z @ beta
        s2 = float(resid @ resid) / max(n - p, 1)
        cov = s2 * Ainv @ (Z.T @ Z) @ Ainv
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        beta[const], se[const] = 0.0, 0.0
        return cls(beta, mean, sd, se, n)

    def move(self, x):
        if self.beta is None or x is None:
            return 0.0
        return float(((np.asarray(x, dtype=float) - self.mean) / self.sd) @ self.beta)

    def lines(self, names, label):
        """Lines: each stat's coefficient, in points per sd of the stat, with its se and t."""
        if self.beta is None:
            return [f"    {label}: not fitted (fewer than {MIN_TRAIN} matches)"]
        parts = [f"{n} {b:+.2f}±{s:.2f} (t {b / s:+.1f})" if s > 0 else f"{n} - (never varied)"
                 for n, b, s in zip(names, self.beta, self.se)]
        return [f"    {label} ({self.n:,} matches; points per sd of each stat):", "      " + "; ".join(parts)]


def walk_forward(rows, features, base=None, ridge=None, min_train=None):
    """Each row (prior_backtest.fit_arm's: match, start, margin, total, actual_*) with the layer's
    moves, fitted each UTC day on the days before it: ({match: (margin move, total move)}, the last
    margin fit, the last total fit). `base`: (margin key, total key) of moves already on the price
    (the layer is fitted on what they leave). ridge and min_train: RIDGE and MIN_TRAIN by default."""
    ridge = RIDGE if ridge is None else ridge
    min_train = MIN_TRAIN if min_train is None else min_train
    rows = sorted(rows, key=lambda r: r["start"])
    days = defaultdict(list)
    for r in rows:
        days[r["start"].date()].append(r)
    train_m, train_t = ([], []), ([], [])
    fit_m, fit_t = Fit(), Fit()
    moves = {}
    for day in sorted(days):
        if len(train_m[1]) >= min_train:
            fit_m = Fit.fit(train_m[0], train_m[1], ridge)
        if len(train_t[1]) >= min_train:
            fit_t = Fit.fit(train_t[0], train_t[1], ridge)
        for r in days[day]:
            f = features.get(r["match"])
            xm, xt = gamer_features.margin_x(f), gamer_features.total_x(f)
            moves[r["match"]] = (fit_m.move(xm) if len(train_m[1]) >= min_train else 0.0,
                                 fit_t.move(xt) if len(train_t[1]) >= min_train else 0.0)
        for r in days[day]:                           # then the day's results go into the training
            if r.get("actual_margin") is None or r.get("margin") is None or r.get("total") is None:
                continue
            f = features.get(r["match"])
            xm, xt = gamer_features.margin_x(f), gamer_features.total_x(f)
            bm, bt = (r.get(base[0], 0.0), r.get(base[1], 0.0)) if base else (0.0, 0.0)
            if xm is not None:
                train_m[0].append(xm)
                train_m[1].append(r["actual_margin"] - r["margin"] - bm)
            if xt is not None:
                train_t[0].append(xt)
                train_t[1].append(r["actual_total"] - r["total"] - bt)
    return moves, fit_m, fit_t


def describe(fit_m, fit_t):
    """Lines: the last fits."""
    return (fit_m.lines(gamer_features.NAMES, "margin (home's stats less away's)")
            + fit_t.lines(gamer_features.NAMES, "total (the two added)"))
