"""Which model foresees a match's drives best: the pace output v21-drives reads.

    python -m xgb.pace_models --history eAMFCalibrator/out/match_history.csv --drives xgb/data/drive_stats.csv \
        --score-from 2026-08-01 --until 2026-10-19

Every model is refitted every day at midnight on the matches of the TRAIN_DAYS before (the results in
by then, RESULT_LAG after kick-off), rows weighted 2^(-days ago / HALF_LIFE), and prices that day's
matches. A match's drives are both sides' (the drive table's h_drives + a_drives).

  league        the weighted mean
  gamers        the two gamers' own recent drives a match (EWMA_MATCHES), averaged; the league where new
  ridge         pace_eff's pace model: a ridge regression on the two gamers' and the two teams' pace
                effects (each match's gamers and teams added), its shrinkage set by hand (PACE_LAMBDA)
  ridge+stream  the same with the stream as a fixed effect
  mixed         the mixed model glmer would fit: gamer and team pace as random effects (each match's two
                gamers on one effect each, multi-membership, as the ridge), the stream fixed, every
                variance estimated by REML at each refit -- the shrinkage the data asks for (lme4 can't
                take multi-membership directly; mixed_fit is checked against lmer on a single-membership
                fit in the tests)
  xgb           XGBoost on each gamer's and team's drives over their last 15 and 50 matches, an EWMA,
                their matches played, and the stream, all as of the fit; its rounds set by early stopping
                on the train window's last 14 days, then refitted on the whole window

Scored from --score-from on every match with drives: RMSE, MAE, the correlation with the real drives,
the slope of real on foreseen (1: calibrated), the skill against the league mean, and each model's
squared error against the ridge's, match by match (t). Writes pace_models.txt and pace_models.csv
(every match, every model) to --out.
"""

import argparse
import datetime as dt
import math
import os

import numpy as np
import pandas as pd
from scipy import optimize, sparse

RESULT_LAG = dt.timedelta(minutes=36)
TRAIN_DAYS = 180
HALF_LIFE = 60.0
PACE_LAMBDA = {"gamer": 45.0, "team": 180.0, "fixed": 1e-4}      # pace_eff's, on its weights of mean
PACE_SCALAR = 0.5                # SCALAR: so 90 and 360 on weights of mean 1, the units mixed's are given in
EWMA_MATCHES = 20.0              # the gamers baseline's and xgb's EWMA: a match's weight halves every 20 matches
WINDOWS = (15, 50)
XGB_PARAMS = dict(max_depth=3, eta=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                  objective="reg:squarederror", verbosity=0, seed=0)
XGB_ROUNDS, XGB_STOP = 600, 50
EARLY_DAYS = 14
MODELS = ("league", "gamers", "ridge", "ridge+stream", "mixed", "xgb")


def load(history, drives):
    """Every settled match with its drives: code, start, p1, p2, t1, t2, stream, drives."""
    h = pd.read_csv(history, parse_dates=["SCHEDULED_START_TIME_UTC"])
    d = pd.read_csv(drives)
    d["drives"] = d.h_drives + d.a_drives
    m = h.merge(d[["match_code", "drives"]], left_on="MATCH_CODE", right_on="match_code")
    m = m.rename(columns={"MATCH_CODE": "code", "SCHEDULED_START_TIME_UTC": "start", "PLAYER_1_HANDLE": "p1",
                          "PLAYER_2_HANDLE": "p2", "PLAYER_1_TEAM": "t1", "PLAYER_2_TEAM": "t2",
                          "STREAM_NUMBER": "stream"})
    m = m[["code", "start", "p1", "p2", "t1", "t2", "stream", "drives"]].dropna(subset=["drives", "p1", "p2"])
    m["stream"] = m.stream.astype(str)
    return m.sort_values("start").reset_index(drop=True)


def weights(start, cut):
    age = (pd.Timestamp(cut) - start).dt.total_seconds().values / 86400.0
    w = np.power(2.0, -age / HALF_LIFE)
    return w / w.mean()


# ---------------------------------------------------------------- the design shared by ridge and mixed

class Design:
    """[fixed | gamers | teams]: the intercept (and the streams, but the first), then one column a gamer and
    one a team, each match's two added (multi-membership; one side alone, gamer_sides=("p1",), is the
    single-membership model lmer fits, for the tests)."""

    def __init__(self, train, stream=True, gamer_sides=("p1", "p2"), team_sides=("t1", "t2")):
        self.streams = sorted(train.stream.unique())[1:] if stream else []
        self.sides = (gamer_sides, team_sides)
        self.gamers = {g: i for i, g in enumerate(sorted(set().union(*(train[c] for c in gamer_sides))))}
        self.teams = {t: i for i, t in enumerate(sorted(set().union(*(train[c] for c in team_sides))))}
        self.p = 1 + len(self.streams)
        self.width = self.p + len(self.gamers) + len(self.teams)

    def matrix(self, df):
        n = len(df)
        rows, cols = [np.arange(n)], [np.zeros(n, int)]
        for k, s in enumerate(self.streams):
            on = np.flatnonzero(df.stream.values == s)
            rows.append(on)
            cols.append(np.full(len(on), 1 + k))
        for key, base, lookup in ([(c, self.p, self.gamers) for c in self.sides[0]]
                                  + [(c, self.p + len(self.gamers), self.teams) for c in self.sides[1]]):
            idx = df[key].map(lookup)
            ok = idx.notna().values
            rows.append(np.flatnonzero(ok))
            cols.append(base + idx[ok].astype(int).values)
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, self.width))

    def penalty(self, lam_g, lam_t, lam_fixed=PACE_LAMBDA["fixed"]):
        return np.concatenate([np.full(self.p, lam_fixed), np.full(len(self.gamers), lam_g),
                               np.full(len(self.teams), lam_t)])


def ridge_fit(design, X, y, w, lam_g, lam_t):
    A = (X.T @ sparse.diags(w) @ X).toarray() + np.diag(design.penalty(lam_g, lam_t))
    return np.linalg.solve(A, X.T @ (w * y))


def reml(design, X, y, w, log_lam):
    """-2 x the REML log-likelihood (up to a constant) of y = X b + Z u + e, u_gamer ~ N(0, s2 / lam_g),
    u_team ~ N(0, s2 / lam_t), e ~ N(0, s2 / w), at log_lam = (log lam_g, log lam_t): Henderson's equations,
    the residual variance profiled out."""
    lam_g, lam_t = np.exp(log_lam)
    p = design.p
    pen = design.penalty(lam_g, lam_t, lam_fixed=0.0)
    C = (X.T @ sparse.diags(w) @ X).toarray() + np.diag(pen)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        return np.inf
    sol = np.linalg.solve(C, X.T @ (w * y))
    res = y - X @ sol
    rss = float(res @ (w * res) + sol[p:] @ (pen[p:] * sol[p:]))
    n = len(y)
    logdet_c = 2.0 * np.log(np.diag(L)).sum()
    logdet_lam = float(np.log(pen[p:]).sum())
    return (n - p) * math.log(rss / (n - p)) + logdet_c - logdet_lam


def mixed_fit(design, X, y, w, start=(math.log(90.0), math.log(360.0))):
    """The REML variances (as lam = s2 / s2_effect) and the coefficients at them (the BLUPs)."""
    opt = optimize.minimize(lambda v: reml(design, X, y, w, v), np.asarray(start, float), method="Nelder-Mead",
                            options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
    lam_g, lam_t = np.exp(opt.x)
    pen = design.penalty(lam_g, lam_t, lam_fixed=0.0)
    beta = np.linalg.solve((X.T @ sparse.diags(w) @ X).toarray() + np.diag(pen), X.T @ (w * y))
    return beta, opt.x


# ---------------------------------------------------------------- as-of rolling stats (gamers, xgb)

class Rolling:
    """Each key's (gamer's, team's) matches in time order, and its stats as of any moment: matches played,
    the mean of its last k, an EWMA -- off the matches settled RESULT_LAG before."""

    def __init__(self, m, keys):
        by = {}
        for side in keys:
            for k, t, v in zip(m[side].values, m.start.values, m.drives.values):
                by.setdefault(k, []).append((t, v))
        self.t, self.cum, self.ewma = {}, {}, {}
        a = 1.0 - 2.0 ** (-1.0 / EWMA_MATCHES)
        for k, rows in by.items():
            rows.sort()
            t = np.array([r[0] for r in rows], dtype="datetime64[ns]")
            v = np.array([r[1] for r in rows], float)
            e = np.empty(len(v) + 1)
            e[0] = np.nan
            for i, x in enumerate(v):
                e[i + 1] = x if i == 0 else e[i] + a * (x - e[i])
            self.t[k], self.cum[k], self.ewma[k] = t, np.concatenate([[0.0], np.cumsum(v)]), e

    def at(self, k, when):
        """(matches settled by `when`, mean of the last 15, of the last 50, the EWMA); NaN where none."""
        if k not in self.t:
            return 0, np.nan, np.nan, np.nan
        i = int(np.searchsorted(self.t[k], np.datetime64(pd.Timestamp(when) - RESULT_LAG), side="right"))
        if i == 0:
            return 0, np.nan, np.nan, np.nan
        c = self.cum[k]
        means = [(c[i] - c[max(0, i - w)]) / min(i, w) for w in WINDOWS]
        return i, means[0], means[1], self.ewma[k][i]


def roll_features(m, gamers, teams, when=None):
    """xgb's features for each row of m, as of its own start (training) or `when` (a day's prices)."""
    out = []
    for r in m.itertuples(index=False):
        at = r.start if when is None else when
        f = []
        for side in ("p1", "p2"):
            f.extend(gamers.at(getattr(r, side), at))
        for side in ("t1", "t2"):
            f.extend(teams.at(getattr(r, side), at)[1:])
        f.append(float(r.stream))
        out.append(f)
    X = np.array(out, float)
    # the two gamers' and teams' stats, symmetrised: their sum and absolute difference
    g1, g2, t1, t2 = X[:, 0:4], X[:, 4:8], X[:, 8:11], X[:, 11:14]
    return np.column_stack([g1 + g2, np.abs(g1 - g2), t1 + t2, np.abs(t1 - t2), X[:, 14:]])


# ---------------------------------------------------------------- the walk

def walk(m, score_from, until, log=print, models=MODELS):
    out = []
    cut = pd.Timestamp(score_from)
    end = pd.Timestamp(until)
    mixed_start = (math.log(90.0), math.log(360.0))
    gamers, teams = Rolling(m, ("p1", "p2")), Rolling(m, ("t1", "t2"))
    own = roll_features(m, gamers, teams) if "xgb" in models else None     # each row's as of its own start
    lams = []
    while cut < end:
        nxt = cut + pd.Timedelta(days=1)
        test = m[(m.start >= cut) & (m.start < nxt)]
        train = m[(m.start >= cut - pd.Timedelta(days=TRAIN_DAYS)) & (m.start + RESULT_LAG <= cut)]
        cut, prev = nxt, cut
        if not len(test) or len(train) < 200:
            continue
        w = weights(train.start, prev)
        y = train.drives.values.astype(float)
        got = pd.DataFrame({"code": test.code.values, "start": test.start.values, "real": test.drives.values})
        got["league"] = np.average(y, weights=w)
        if "gamers" in models:
            g = np.array([[gamers.at(r.p1, prev)[3], gamers.at(r.p2, prev)[3]] for r in test.itertuples()])
            got["gamers"] = np.where(np.isnan(g), got.league.values[:, None], g).mean(axis=1)
        for name, stream in (("ridge", False), ("ridge+stream", True)):
            if name in models:
                d = Design(train, stream=stream)
                beta = ridge_fit(d, d.matrix(train), y, PACE_SCALAR * w, PACE_LAMBDA["gamer"], PACE_LAMBDA["team"])
                got[name] = d.matrix(test) @ beta
        if "mixed" in models:
            d = Design(train, stream=True)
            beta, mixed_start = mixed_fit(d, d.matrix(train), y, w, mixed_start)
            got["mixed"] = d.matrix(test) @ beta
            lams.append(np.exp(mixed_start))
        if "xgb" in models:
            import xgboost
            Xt = own[train.index.values]
            Xe = roll_features(test, gamers, teams, when=prev)
            early = (train.start >= prev - pd.Timedelta(days=EARLY_DAYS)).values
            probe = xgboost.train(XGB_PARAMS, xgboost.DMatrix(Xt[~early], label=y[~early], weight=w[~early]),
                                  XGB_ROUNDS, evals=[(xgboost.DMatrix(Xt[early], label=y[early], weight=w[early]),
                                                      "early")], early_stopping_rounds=XGB_STOP, verbose_eval=False)
            full = xgboost.train(XGB_PARAMS, xgboost.DMatrix(Xt, label=y, weight=w), probe.best_iteration + 1)
            got["xgb"] = full.predict(xgboost.DMatrix(Xe))
        out.append(got)
        if prev.day == 1 or prev == pd.Timestamp(score_from):
            extra = (f"; mixed's REML shrinkage: gamer {lams[-1][0]:.0f}, team {lams[-1][1]:.0f} (the ridge's 90, 360)"
                     if lams else "")
            log(f"  {prev:%Y-%m-%d}: trained on {len(train):,} matches, {np.average(y, weights=w):.2f} drives a match{extra}")
    return pd.concat(out, ignore_index=True), np.array(lams)


def score_lines(f, models):
    rows = f.dropna(subset=["real"])
    y = rows.real.values
    base = ((y - rows.league.values) ** 2)
    ref = ((y - rows["ridge"].values) ** 2) if "ridge" in rows else None
    lines = [f"  {len(rows):,} matches scored, {rows.start.min():%Y-%m-%d} to {rows.start.max():%Y-%m-%d}; real drives "
             f"{y.mean():.2f} a match (sd {y.std():.2f})", "",
             f"  {'model':14s}{'RMSE':>8s}{'MAE':>8s}{'r':>8s}{'slope':>8s}{'bias':>8s}{'skill':>9s}"
             f"{'sd':>7s}{'vs ridge: sq err':>20s}"]
    for name in models:
        if name not in rows:
            continue
        p = rows[name].values
        e2 = (y - p) ** 2
        r = np.corrcoef(p, y)[0, 1] if p.std() > 0 else float("nan")
        slope = np.polyfit(p, y, 1)[0] if p.std() > 0 else float("nan")
        skill = 1 - e2.mean() / base.mean()
        vs = ""
        if ref is not None and name != "ridge":
            d = e2 - ref
            vs = f"{d.mean():+.3f} (t {d.mean() / (d.std(ddof=1) / math.sqrt(len(d))):+.1f})"
        lines.append(f"  {name:14s}{math.sqrt(e2.mean()):8.3f}{np.abs(y - p).mean():8.3f}{r:8.3f}{slope:8.2f}"
                     f"{(p - y).mean():+8.2f}{100 * skill:8.2f}%{p.std():7.2f}{vs:>20s}")
    lines += ["", "  skill: the squared error saved against the league mean; slope: real on foreseen (1 calibrated, "
                  "below 1 spread too wide); sd: the spread of the foreseen drives across matches"]
    return lines


def fifth_lines(f, name):
    rows = f.dropna(subset=["real", name]).copy()
    rows["q"] = pd.qcut(rows[name], 5, labels=False)
    lines = [f"  by fifth of {name}'s foreseen drives:  " + "  ".join(
        f"{g[name].mean():.2f} -> {g.real.mean():.2f}" for _, g in rows.groupby("q"))]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default="eAMFCalibrator/out/match_history.csv")
    ap.add_argument("--drives", default="xgb/data/drive_stats.csv")
    ap.add_argument("--score-from", default="2026-08-01")
    ap.add_argument("--until", default="2026-10-19", help="the day after the last scored")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--out", default="xgb/out/pace_models")
    a = ap.parse_args(argv)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    models = tuple(x.strip() for x in a.models.split(",") if x.strip())
    m = load(a.history, a.drives)
    log(f"  {len(m):,} matches with drives, {m.start.min():%Y-%m-%d} to {m.start.max():%Y-%m-%d}; refitted daily "
        f"on {TRAIN_DAYS} days, half-life {HALF_LIFE:g} days")
    f, lams = walk(m, a.score_from, a.until, log, models)
    log("")
    for s in score_lines(f, models):
        log(s)
    log("")
    for name in models:
        if name in f and name != "league":
            for s in fifth_lines(f, name):
                log(s)
    if len(lams):
        log("")
        log(f"  mixed's REML shrinkage over the refits (lam = residual variance / effect variance): gamer "
            f"{np.median(lams[:, 0]):.0f} ({lams[:, 0].min():.0f}-{lams[:, 0].max():.0f}), team "
            f"{np.median(lams[:, 1]):.0f} ({lams[:, 1].min():.0f}-{lams[:, 1].max():.0f}); the ridge's 90 and 360 (pace_eff's 45 and 180 on weights "
            f"of mean {PACE_SCALAR:g})")
    os.makedirs(a.out, exist_ok=True)
    f.to_csv(os.path.join(a.out, "pace_models.csv"), index=False)
    with open(os.path.join(a.out, "pace_models.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  -> {a.out}/pace_models.txt, pace_models.csv")


if __name__ == "__main__":
    main()
