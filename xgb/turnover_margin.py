"""Does a turnover prior add to the pre-match margin? Each side's expected turnovers, walk-forward, against
the form table's expected margin and moneyline.

    python -m xgb.turnover_margin --counts xgb/out/turnovers/turnovers_matches.csv \
        --history eAMFCalibrator/out/match_history.csv \
        --prior eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv

The turnover prior: each side's turnovers (xgb/turnovers.py's counts, as the sim reads its snaps) on a home
term, the offense gamer's turnovers given up, the defense gamer's forced, and both teams' (a ridge, every
effect shrunk toward none), refitted every midnight on the results in by then (RESULT_LAG) over the
TRAIN_DAYS before, rows weighted 2^(-days ago / HALF_LIFE). Its shrinkage is chosen walk-forward on the
TUNE_DAYS before the prior's first match, off the per-side squared error, so the scored span plays no part.
The expected turnover margin (away's expected turnovers less home's: what the home side gains) is then set
against the prior on the prior's matches:
  1. how well it foresees the real turnover margin, and how much of it the prior's margin already carries
  2. the margin: the real margin on the prior's expected margin and the expected turnover margin (the
     turnovers' coefficient and t), and out of sample: the blend fitted on the first half of the matches
     and scored on the second, against the prior alone (refitted the same way) -- squared error, paired t
  3. the moneyline: a home win (ties out) on the prior's logit(p_home) and the expected turnover margin,
     in sample and fitted on the first half, scored on the second (log loss, paired t)
Writes turnover_margin.txt and turnover_margin.csv (each match: the prior, the expected turnovers, the
real) to --out.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy import optimize, sparse

RESULT_LAG = pd.Timedelta(minutes=36)
TRAIN_DAYS = 180
HALF_LIFE = 60.0
TUNE_DAYS = 31
MIN_TRAIN = 400                  # side rows
GRID_GAMER = (1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0)
GRID_TEAM = (160.0, 640.0, 2560.0, 1e4)
FIXED = 1e-4


def sides(counts, history):
    """One row a side a match: its turnovers, home or not, the offense's gamer and team, the defense's."""
    h = pd.read_csv(history, usecols=["MATCH_CODE", "PLAYER_1_TEAM", "PLAYER_2_TEAM"])
    m = counts.merge(h, on="MATCH_CODE", how="left")
    home = pd.DataFrame({"code": m.MATCH_CODE, "start": m.start, "y": m.h_turnovers, "home": 1.0,
                         "og": m.home, "dg": m.away, "ot": m.PLAYER_1_TEAM, "dt": m.PLAYER_2_TEAM})
    away = pd.DataFrame({"code": m.MATCH_CODE, "start": m.start, "y": m.a_turnovers, "home": 0.0,
                         "og": m.away, "dg": m.home, "ot": m.PLAYER_2_TEAM, "dt": m.PLAYER_1_TEAM})
    s = pd.concat([home, away], ignore_index=True).fillna({"ot": "", "dt": ""})
    return s.sort_values(["start", "code", "home"], ascending=[True, True, False]).reset_index(drop=True)


class Design:
    """[intercept, home | gamers given up | gamers forced | teams given up | teams forced], every side row."""

    def __init__(self, s):
        gamers = sorted(set(s.og) | set(s.dg))
        teams = sorted((set(s.ot) | set(s.dt)) - {""})
        self.ng, self.nt = len(gamers), len(teams)
        g, t = {x: i for i, x in enumerate(gamers)}, {x: i for i, x in enumerate(teams)}
        n = len(s)
        base = (2, 2 + self.ng, 2 + 2 * self.ng, 2 + 2 * self.ng + self.nt)
        rows = [np.arange(n), np.arange(n)]
        cols = [np.zeros(n, int), np.ones(n, int)]
        vals = [np.ones(n), s.home.values]
        for key, lookup, b in (("og", g, base[0]), ("dg", g, base[1]), ("ot", t, base[2]), ("dt", t, base[3])):
            idx = s[key].map(lookup)
            ok = idx.notna().values
            rows.append(np.flatnonzero(ok))
            cols.append(b + idx[ok].astype(int).values)
            vals.append(np.ones(int(ok.sum())))
        self.width = 2 + 2 * self.ng + 2 * self.nt
        self.X = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(n, self.width))

    def penalty(self, lam_g, lam_t):
        return np.concatenate([[FIXED, FIXED], np.full(2 * self.ng, lam_g), np.full(2 * self.nt, lam_t)])


def walk(s, design, first, last, lams):
    """{lams: each side row's expected turnovers (NaN where not priced)} for the days first..last, each day's
    rows priced by the fit on the results in by its midnight."""
    out = {lam: np.full(len(s), np.nan) for lam in lams}
    start = s.start.values
    y = s.y.values.astype(float)
    day = pd.Timestamp(first)
    while day <= pd.Timestamp(last):
        nxt = day + pd.Timedelta(days=1)
        today = np.flatnonzero((start >= day.to_datetime64()) & (start < nxt.to_datetime64()))
        train = np.flatnonzero((start >= (day - pd.Timedelta(days=TRAIN_DAYS)).to_datetime64())
                               & (start + RESULT_LAG.to_timedelta64() <= day.to_datetime64()))
        if len(today) and len(train) >= MIN_TRAIN:
            age = (day.to_datetime64() - start[train]) / np.timedelta64(1, "D")
            w = np.power(2.0, -age / HALF_LIFE)
            w /= w.mean()
            Xt = design.X[train]
            A = (Xt.T @ sparse.diags(w) @ Xt).toarray()
            b = Xt.T @ (w * y[train])
            Xe = design.X[today]
            for lam in lams:
                out[lam][today] = Xe @ np.linalg.solve(A + np.diag(design.penalty(*lam)), b)
        day = nxt
    return out


def tune(s, design, first, lines):
    """The shrinkage with the least squared error on the TUNE_DAYS before `first`."""
    lams = [(g, t) for g in GRID_GAMER for t in GRID_TEAM]
    lo = (pd.Timestamp(first) - pd.Timedelta(days=TUNE_DAYS)).date()
    hi = (pd.Timestamp(first) - pd.Timedelta(days=1)).date()
    got = walk(s, design, lo, hi, lams)
    ok = ~np.isnan(got[lams[0]])
    y = s.y.values[ok]
    err = {lam: float(np.mean((y - got[lam][ok]) ** 2)) for lam in lams}
    base = float(np.mean((y - y.mean()) ** 2))
    best = min(err, key=err.get)
    lines.append(f"  shrinkage chosen on {lo} to {hi} ({ok.sum():,} side rows, walk-forward): gamer {best[0]:g}, "
                 f"team {best[1]:g}; squared error {err[best]:.4f} against {base:.4f} for the span's own mean")
    lines.append("    gamer \\ team " + "".join(f"{t:>9g}" for t in GRID_TEAM))
    for g in GRID_GAMER:
        lines.append(f"    {g:>13g} " + "".join(f"{err[(g, t)]:9.4f}" for t in GRID_TEAM))
    return best


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    se = np.sqrt(np.diag(res @ res / (len(y) - X.shape[1]) * np.linalg.pinv(X.T @ X)))
    return b, se, X @ b


def logit_fit(y, X):
    """Logistic regression (with an intercept): (coefficients, standard errors)."""
    X = np.column_stack([np.ones(len(y)), X])

    def nll(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z))

    def grad(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        return X.T @ (p - y)

    b = optimize.minimize(nll, np.zeros(X.shape[1]), jac=grad, method="BFGS").x
    p = 1 / (1 + np.exp(-(X @ b)))
    H = X.T @ (X * (p * (1 - p))[:, None])
    return b, np.sqrt(np.diag(np.linalg.pinv(H)))


def logloss(y, p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired(a, b):
    """mean(a - b) and its t."""
    d = a - b
    return d.mean(), d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))


def report(t, lines):
    """The tests on the prior's matches (t: one row a match with exp_margin, p_home, xtm, the real)."""
    M, TM, E, X = t.margin.values, t.to_margin.values, t.exp_margin.values, t.xtm.values
    lines += ["", f"  {len(t):,} matches, {t.start.min():%Y-%m-%d} to {t.start.max():%Y-%m-%d}",
              "", "1. The expected turnover margin (away's expected turnovers less home's)",
              f"  spread sd {X.std():.3f} against the real turnover margin's {TM.std():.2f}; r {corr(X, TM):.3f} with it",
              f"  r {corr(X, E):.3f} with the prior's expected margin: the share of it the prior already carries "
              f"{corr(X, E) ** 2:.0%}",
              f"  the prior's miss (real margin - expected, sd {(M - E).std():.2f}) against it: r {corr(M - E, X):.3f}"]
    b, se, _ = ols(M, np.column_stack([E, X]))
    b0, _, _ = ols(M, E)
    lines += ["", "2. The margin",
              f"  in sample: margin = {b[0]:+.2f} + {b[1]:.3f} x the prior's + {b[2]:+.2f} (t {b[2] / se[2]:+.1f}) x the "
              f"expected turnover margin; the prior alone {b0[1]:.3f}"]
    half = len(t) // 2
    fit, test = slice(0, half), slice(half, None)
    pa = ols(M[fit], E[fit])[0]
    pb = ols(M[fit], np.column_stack([E[fit], X[fit]]))[0]
    ea = (M[test] - (pa[0] + pa[1] * E[test])) ** 2
    eb = (M[test] - (pb[0] + pb[1] * E[test] + pb[2] * X[test])) ** 2
    e0 = (M[test] - E[test]) ** 2
    d, tt = paired(eb, ea)
    lines += [f"  out of sample (fitted on {t.start.iloc[0]:%d %b} - {t.start.iloc[half - 1]:%d %b}, scored on "
              f"{t.start.iloc[half]:%d %b} - {t.start.iloc[-1]:%d %b}, {len(e0):,} matches): RMSE the prior as it is "
              f"{math.sqrt(e0.mean()):.3f}, refitted {math.sqrt(ea.mean()):.3f}, with the turnovers "
              f"{math.sqrt(eb.mean()):.3f} (turnover weight {pb[2]:+.2f}); squared error {d:+.3f} a match (t {tt:+.1f})"]
    ml = t[t.margin != 0]
    y = (ml.margin.values > 0).astype(float)
    p = np.clip(ml.p_home.values, 1e-6, 1 - 1e-6)
    L, Xm = np.log(p / (1 - p)), ml.xtm.values
    bl, sel = logit_fit(y, np.column_stack([L, Xm]))
    half = len(ml) // 2
    fa = logit_fit(y[:half], L[:half])[0]
    fb = logit_fit(y[:half], np.column_stack([L[:half], Xm[:half]]))[0]
    sig = lambda z: 1 / (1 + np.exp(-z))
    la = logloss(y[half:], sig(fa[0] + fa[1] * L[half:]))
    lb = logloss(y[half:], sig(fb[0] + fb[1] * L[half:] + fb[2] * Xm[half:]))
    l0 = logloss(y[half:], p[half:])
    d, tt = paired(lb, la)
    lines += ["", f"3. The moneyline ({len(ml):,} matches, ties out)",
              f"  in sample: logit P(home) = {bl[0]:+.3f} + {bl[1]:.3f} x the prior's logit + {bl[2]:+.3f} (t "
              f"{bl[2] / sel[2]:+.1f}) x the expected turnover margin",
              f"  out of sample ({len(l0):,} matches): log loss the prior as it is {l0.mean():.4f}, refitted "
              f"{la.mean():.4f}, with the turnovers {lb.mean():.4f}; {d:+.4f} a match (t {tt:+.1f})"]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", default=os.path.join("xgb", "out", "turnovers", "turnovers_matches.csv"))
    ap.add_argument("--history", default=os.path.join("eAMFCalibrator", "out", "match_history.csv"))
    ap.add_argument("--prior", default=os.path.join("eAMFCalibrator", "out", "form_prior", "priors", "avg_form4_w05.csv"))
    ap.add_argument("--out", default=os.path.join("xgb", "out", "turnover_margin"))
    a = ap.parse_args(argv)
    counts = pd.read_csv(a.counts, parse_dates=["start"])
    prior = pd.read_csv(a.prior, parse_dates=["start"])
    prior = prior[prior.p_home.notna() & prior.exp_margin.notna()]
    s = sides(counts, a.history)
    design = Design(s)
    first, last = prior.start.min().date(), prior.start.max().date()
    lines = [f"Turnovers before the match, against {a.prior}: {len(counts):,} matches counted, {len(prior):,} in the prior",
             f"  the turnover prior: a side's turnovers on home, the offense gamer's given up, the defense gamer's "
             f"forced and both teams', refitted daily ({TRAIN_DAYS} days, half-life {HALF_LIFE:g})"]
    lam = tune(s, design, first, lines)
    s["exp"] = walk(s, design, first, last, [lam])[lam]
    per = s.pivot_table(index="code", columns="home", values=["exp", "y"])
    per.columns = [f"{v}_{'home' if h else 'away'}" for v, h in per.columns]
    t = prior.merge(per, left_on="MATCH_CODE", right_index=True).dropna(subset=["exp_home", "exp_away"])
    t = t.merge(counts[["MATCH_CODE", "home_final", "away_final"]], on="MATCH_CODE").dropna(subset=["home_final"])
    t["margin"] = t.home_final - t.away_final
    t["to_margin"] = t.y_away - t.y_home
    t["xtm"] = t.exp_away - t.exp_home
    t = t.sort_values("start").reset_index(drop=True)
    report(t, lines)
    print("\n".join(lines))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "turnover_margin.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    t[["MATCH_CODE", "start", "home", "away", "exp_margin", "p_home", "exp_home", "exp_away", "xtm", "y_home",
       "y_away", "margin"]].to_csv(os.path.join(a.out, "turnover_margin.csv"), index=False)
    print(f"\n  -> {a.out}/turnover_margin.txt, turnover_margin.csv")


if __name__ == "__main__":
    main()
