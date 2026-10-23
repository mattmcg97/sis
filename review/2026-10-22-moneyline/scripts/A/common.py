"""Agent A's shared pieces: data, walk-forward margin sources, rematch / session features, the link fitter.

Every margin source here is an expected home margin known at kick-off: a result enters RESULT_LAG
(36 minutes) after its own kick-off, exactly as xgb/features.py does.
"""
import math
import os
import sys
from collections import defaultdict

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ML = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml"
sys.path.insert(0, ML)
import harness as H  # noqa: E402

OUT = os.path.join(ML, "out", "A")
os.makedirs(OUT, exist_ok=True)
LAG_DAYS = 36.0 / 1440.0
REMATCH_GAP_MIN = 240          # eAMFCalibrator config.SESSION_BREAK_MINUTES (bet_rematches.day_meetings)


# --------------------------------------------------------------------------- data

def load(tables=True):
    df = H.load(tables=tables, elo=False)
    df["tday"] = (df.start - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
    df["stream"] = df["stream"].fillna(1).astype(int)
    add_sessions(df)
    return df


def add_sessions(df):
    """Each gamer's session (a new one after a gap over REMATCH_GAP_MIN between his kick-offs), the pair's
    day key (sorted pair + the first gamer's session, as bet_rematches.day_meetings) and the meeting number."""
    last, sess = {}, {}
    s1 = np.zeros(len(df), int)
    s2 = np.zeros(len(df), int)
    gap = REMATCH_GAP_MIN / 1440.0
    for i, (t, a, b) in enumerate(zip(df.tday.to_numpy(), df.p1.to_numpy(), df.p2.to_numpy())):
        for g, arr in ((a, s1), (b, s2)):
            if g in last:
                n = sess[g] + ((t - last[g]) > gap)
            else:
                n = 0
            sess[g] = n
            last[g] = t
            arr[i] = n
    df["sess1"], df["sess2"] = s1, s2
    first_is_p1 = df.p1.to_numpy() < df.p2.to_numpy()
    a = np.where(first_is_p1, df.p1, df.p2)
    b = np.where(first_is_p1, df.p2, df.p1)
    sa = np.where(first_is_p1, s1, s2)
    df["pair_key"] = pd.Series(a, index=df.index) + "|" + pd.Series(b, index=df.index) + "|" + pd.Series(sa, index=df.index).astype(str)
    df["meeting"] = df.groupby("pair_key").cumcount() + 1
    df["pair_sign"] = np.where(first_is_p1, 1.0, -1.0)    # +1 if home is the alphabetically first gamer
    return df


# --------------------------------------------------------------------------- the Kalman form rating

def kalman_rating(df, drift=4.0, sigma=12.0, v0=16.0, team_rate=0.002, home_rate=0.0, var_cap=None):
    """xgb/features.py's rating (form_prior.rating), re-run with these settings, results entering 36 minutes
    after kick-off. Returns (margin, var1, var2) arrays aligned to df (df in kick-off order).
    home_rate: a home edge learned online like the teams (0: none, as in features.py)."""
    t = df.tday.to_numpy()
    p1, p2 = df.p1.to_numpy(), df.p2.to_numpy()
    t1, t2 = df.t1.fillna("").to_numpy(), df.t2.fillna("").to_numpy()
    s1, s2 = df.s1.to_numpy(), df.s2.to_numpy()
    n = len(df)
    settled = [i for i in range(n) if not (np.isnan(s1[i]) or np.isnan(s2[i]))]
    settled.sort(key=lambda i: (t[i], i))
    mu, var, last = {}, {}, {}
    team = defaultdict(float)
    home = 0.0
    out_m = np.zeros(n)
    out_v1 = np.zeros(n)
    out_v2 = np.zeros(n)
    j = 0

    def settle(i):
        nonlocal home
        day = t[i]
        for g in (p1[i], p2[i]):
            if g not in mu:
                mu[g], var[g] = 0.0, v0
            elif g in last:
                var[g] += drift * max(day - last[g], 0.0)
                if var_cap is not None:
                    var[g] = min(var[g], var_cap)
            last[g] = day
        exp = mu[p1[i]] - mu[p2[i]] + team[t1[i]] - team[t2[i]] + home
        s2_ = sigma ** 2 + var[p1[i]] + var[p2[i]]
        miss = (s1[i] - s2[i]) - exp
        v1, v2 = var[p1[i]], var[p2[i]]
        mu[p1[i]] += v1 / s2_ * miss
        mu[p2[i]] -= v2 / s2_ * miss
        var[p1[i]], var[p2[i]] = v1 - v1 * v1 / s2_, v2 - v2 * v2 / s2_
        if t1[i] and t2[i]:
            team[t1[i]] += team_rate * miss
            team[t2[i]] -= team_rate * miss
            mean = sum(team.values()) / len(team)
            for k in list(team):
                team[k] -= mean
        home += home_rate * miss

    def now(g, tt):
        if g not in mu:
            return 0.0, v0
        v = var[g] + drift * max(tt - last.get(g, tt), 0.0)
        if var_cap is not None:
            v = min(v, var_cap)
        return mu[g], v

    for i in range(n):
        while j < len(settled) and t[settled[j]] + LAG_DAYS <= t[i] + 1e-9:
            settle(settled[j])
            j += 1
        m1, v1 = now(p1[i], t[i])
        m2, v2 = now(p2[i], t[i])
        out_m[i] = m1 - m2 + team[t1[i]] - team[t2[i]] + home
        out_v1[i], out_v2[i] = v1, v2
    return out_m, out_v1, out_v2


# --------------------------------------------------------------------------- the decayed ridge

class Design:
    """Columns: 0 home intercept, then gamers, then teams, then (optional) per-stream home, per-gamer home."""

    def __init__(self, df, stream_home=False, gamer_home=False):
        gamers = sorted(set(df.p1) | set(df.p2))
        teams = sorted(set(df.t1.fillna("")) | set(df.t2.fillna("")))
        self.gi = {g: k for k, g in enumerate(gamers)}
        self.ti = {t: k for k, t in enumerate(teams)}
        self.ng, self.nt = len(gamers), len(teams)
        self.off_g = 1
        self.off_t = 1 + self.ng
        self.off_s = self.off_t + self.nt
        self.stream_home = stream_home
        self.gamer_home = gamer_home
        self.off_gh = self.off_s + (3 if stream_home else 0)
        self.p = self.off_gh + (self.ng if gamer_home else 0)

    def rows(self, df):
        """(idx list, val list) per match: the sparse margin row."""
        out = []
        for a, b, ta, tb, s in zip(df.p1, df.p2, df.t1.fillna(""), df.t2.fillna(""), df.stream):
            idx = [0, self.off_g + self.gi[a], self.off_g + self.gi[b], self.off_t + self.ti[ta], self.off_t + self.ti[tb]]
            val = [1.0, 1.0, -1.0, 1.0, -1.0]
            if self.stream_home:
                idx.append(self.off_s + int(s) - 1)
                val.append(1.0)
            if self.gamer_home:
                idx.append(self.off_gh + self.gi[a])
                val.append(1.0)
            out.append((np.array(idx), np.array(val)))
        return out

    def penalty(self, lam_g, lam_t, lam_h=1e-3, lam_s=50.0, lam_gh=200.0):
        P = np.zeros(self.p)
        P[0] = lam_h
        P[self.off_g:self.off_t] = lam_g
        P[self.off_t:self.off_s] = lam_t
        if self.stream_home:
            P[self.off_s:self.off_s + 3] = lam_s
        if self.gamer_home:
            P[self.off_gh:] = lam_gh
        return P


def ridge_live(df, half_life=30.0, lam_g=5.0, lam_t=50.0, target="margin", stream_home=False, gamer_home=False,
               lam_s=50.0, lam_gh=200.0, daily=False, return_beta_days=None, return_var=False,
               refresh_every=0, refine_iters=3):
    """Expected home margin (target 'margin') at every kick-off from a ridge on every result settled by then
    (36 minutes after its kick-off; with daily=True only those settled by the day's midnight UTC), each
    weighted 2^(-age/half_life days). The penalties are on the weighted sum of squares (so lam is in units
    of matches of full weight). Kick-off order assumed."""
    des = Design(df, stream_home, gamer_home)
    rows = des.rows(df)
    P = des.penalty(lam_g, lam_t, lam_s=lam_s, lam_gh=lam_gh)
    y_all = (df.s1 - df.s2).to_numpy() if target == "margin" else (df.s1 + df.s2).to_numpy()
    t = df.tday.to_numpy()
    n = len(df)
    settle_t = t + LAG_DAYS
    order = [i for i in np.argsort(settle_t, kind="stable") if not np.isnan(y_all[i])]
    S = np.zeros((des.p, des.p))
    r = np.zeros(des.p)
    sw = 0.0                     # decayed weight sum (for diagnostics)
    t_ref = None
    lam_decay = math.log(2) / half_life if half_life else 0.0
    out = np.full(n, np.nan)
    pvar = np.full(n, np.nan)
    Ainv = np.diag(1.0 / P)
    j = 0
    beta = np.zeros(des.p)
    last_solved_j = -1
    stale = None
    if target == "total":
        sgn_rows = [(idx, np.abs(val)) for idx, val in rows]   # sum coding for totals
        rows = sgn_rows
    days_beta = {}
    for i in range(n):
        cut = (math.floor(t[i]) if daily else t[i]) + 1e-9
        moved = False
        while j < len(order) and settle_t[order[j]] <= cut + (0.0 if daily else 0.0):
            k = order[j]
            # decay everything to the settle time of k
            tk = settle_t[k]
            if t_ref is None:
                t_ref = tk
            if lam_decay and tk > t_ref:
                f = math.exp(-lam_decay * (tk - t_ref))
                S *= f
                r *= f
                sw *= f
                t_ref = tk
            idx, val = rows[k]
            w = math.exp(-lam_decay * (tk - t[k])) if lam_decay else 1.0   # its kick-off age at settle (36 min)
            S[np.ix_(idx, idx)] += w * np.outer(val, val)
            r[idx] += w * val * y_all[k]
            sw += w
            j += 1
            moved = True
        if moved or last_solved_j < 0:
            if j == 0:
                beta = np.zeros(des.p)
            else:
                # bring the system to now: A = f*S + P where f decays from t_ref to the kick-off
                t_now = cut if daily else t[i]
                f = math.exp(-lam_decay * max(t_now - t_ref, 0.0)) if lam_decay else 1.0
                A = f * S + np.diag(P)
                rhs = f * r
                if stale is None or stale >= refresh_every:
                    Ainv = np.linalg.inv(A)
                    beta = Ainv @ rhs
                    stale = 0
                else:
                    b_try = beta
                    for _ in range(refine_iters):        # iterative refinement off the recent inverse
                        b_try = b_try + Ainv @ (rhs - A @ b_try)
                    if np.linalg.norm(rhs - A @ b_try) <= 1e-7 * (np.linalg.norm(rhs) + 1e-12):
                        beta = b_try
                        stale += 1
                    else:
                        Ainv = np.linalg.inv(A)
                        beta = Ainv @ rhs
                        stale = 0
            last_solved_j = j
        idx, val = rows[i]
        out[i] = float(val @ beta[idx])
        if return_var:
            pvar[i] = float(val @ Ainv[np.ix_(idx, idx)] @ val)
        if return_beta_days is not None:
            d = df.day.iat[i]
            if d in return_beta_days and d not in days_beta:
                days_beta[d] = beta.copy()
    if return_beta_days is not None:
        return out, days_beta, des
    if return_var:
        return out, pvar
    return out


# --------------------------------------------------------------------------- rematch / session features

def rematch_features(df, mcol, prefix="rm"):
    """For each match, its pair's earlier meetings in the same pair-session (bet_rematches' day key) that were
    settled by kick-off: the count, the mean margin residual against the source mcol (from this match's home
    side), the mean result (+1 home gamer won, -1 lost, 0 tie), and the meeting number."""
    t = df.tday.to_numpy()
    key = df.pair_key.to_numpy()
    p1 = df.p1.to_numpy()
    mar = (df.s1 - df.s2).to_numpy()
    m = df[mcol].to_numpy()
    n = len(df)
    cnt = np.zeros(n)
    res = np.zeros(n)
    win = np.zeros(n)
    last_res = np.zeros(n)
    hist = defaultdict(list)     # key -> [(t, home gamer, margin, source margin)]
    for i in range(n):
        lst = hist[key[i]]
        rs, ws = [], []
        for (tj, hj, mj, sj) in lst:
            if tj + LAG_DAYS <= t[i] + 1e-9 and not np.isnan(mj) and not np.isnan(sj):
                sg = 1.0 if hj == p1[i] else -1.0
                rs.append(sg * (mj - sj))
                ws.append(sg * np.sign(mj))
        if rs:
            cnt[i] = len(rs)
            res[i] = np.mean(rs)
            win[i] = np.mean(ws)
            last_res[i] = rs[-1]
        lst.append((t[i], p1[i], mar[i], m[i]))
    return pd.DataFrame({f"{prefix}_n": cnt, f"{prefix}_res": res, f"{prefix}_win": win, f"{prefix}_last": last_res},
                        index=df.index)


def session_features(df, mcol, prefix="sf"):
    """Each gamer's earlier matches this session (his own 240-minute session) against anyone, settled by
    kick-off: the mean margin residual against mcol from his side, and the count. Returns home minus away."""
    t = df.tday.to_numpy()
    p1, p2 = df.p1.to_numpy(), df.p2.to_numpy()
    se1, se2 = df.sess1.to_numpy(), df.sess2.to_numpy()
    mar = (df.s1 - df.s2).to_numpy()
    m = df[mcol].to_numpy()
    n = len(df)
    hist = defaultdict(list)     # (gamer, session) -> [(t, residual from his side)]
    r1 = np.zeros(n)
    r2 = np.zeros(n)
    c1 = np.zeros(n)
    c2 = np.zeros(n)
    for i in range(n):
        for g, s, rr, cc in ((p1[i], se1[i], r1, c1), (p2[i], se2[i], r2, c2)):
            xs = [x for (tj, x) in hist[(g, s)] if tj + LAG_DAYS <= t[i] + 1e-9]
            if xs:
                rr[i] = np.mean(xs)
                cc[i] = len(xs)
        if not (np.isnan(mar[i]) or np.isnan(m[i])):
            e = mar[i] - m[i]
            hist[(p1[i], se1[i])].append((t[i], e))
            hist[(p2[i], se2[i])].append((t[i], -e))
    # shrink each mean by its count (k/(k+c)) is left to the link; give the sums too
    return pd.DataFrame({f"{prefix}_gap": r1 - r2, f"{prefix}_sum": r1 * c1 - r2 * c2, f"{prefix}_n1": c1,
                         f"{prefix}_n2": c2}, index=df.index)


# --------------------------------------------------------------------------- the link

def fit_logistic(X, y, w=None, l2=None, beta0=None, iters=50, tol=1e-9):
    """Weighted logistic regression by Newton steps; l2: per-coefficient ridge (array) or None.
    X includes the intercept column if wanted."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    L = np.zeros(p) if l2 is None else np.asarray(l2, float) * np.ones(p)
    b = np.zeros(p) if beta0 is None else np.asarray(beta0, float).copy()
    for _ in range(iters):
        z = X @ b
        pr = 1.0 / (1.0 + np.exp(-z))
        g = X.T @ (w * (pr - y)) + L * b
        Hm = (X * (w * pr * (1 - pr))[:, None]).T @ X + np.diag(L)
        step = np.linalg.solve(Hm + 1e-10 * np.eye(p), g)
        b -= step
        if np.max(np.abs(step)) < tol:
            break
    return b


def fit_probit(X, y, w=None):
    from scipy.optimize import minimize
    from scipy.stats import norm
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else w

    def nll(b):
        z = X @ b
        return -np.sum(w * (y * norm.logcdf(z) + (1 - y) * norm.logcdf(-z)))

    b0 = fit_logistic(X, y, w) / 1.7
    r = minimize(nll, b0, method="BFGS")
    return r.x


def decay_weights(train_start, cut, half_life):
    if not half_life:
        return None
    age = (cut - train_start).dt.total_seconds().to_numpy() / 86400.0
    return np.power(0.5, age / half_life)


def link_walk_forward(df, terms, start, end, train_days=None, half_life=None, min_train=300, refit_days=1,
                      l2=None, probit=False, keep_coefs=False, train_from=None, train_mask=None):
    """P(home) = logistic(a + sum b_k * term_k), each term a column (or a function of df -> array), fitted on
    the matches settled by each day's midnight (harness.walk_forward), optionally the last train_days or
    decayed by half_life days. Rows with any term NaN are left out of the fit and the predictions."""
    cols = {}
    for k, tm in enumerate(terms):
        cols[f"x{k}"] = tm(df) if callable(tm) else df[tm].to_numpy(float)
    Xall = np.column_stack([np.ones(len(df))] + [cols[f"x{k}"] for k in range(len(terms))])
    ok = np.isfinite(Xall).all(1)
    if train_mask is not None:
        okfit = ok & train_mask
    else:
        okfit = ok
    if train_from is not None:
        okfit = okfit & (df.day.to_numpy() >= train_from)
    coefs = {}

    def fit(tr):
        idx = tr.index[okfit[tr.index]]
        if len(idx) < min_train:
            return None
        X = Xall[idx]
        y = df.y.to_numpy()[idx]
        cut = tr.start.max()
        w = decay_weights(df.start[idx], cut, half_life)
        if probit:
            b = fit_probit(X, y, w)
        else:
            b = fit_logistic(X, y, w, l2=l2)
        return b

    def predict(b, dd):
        if b is None:
            return np.full(len(dd), np.nan)
        X = Xall[dd.index]
        z = X @ b
        if keep_coefs:
            coefs[dd.day.iat[0]] = b.copy()
        if probit:
            from scipy.stats import norm
            p = norm.cdf(z)
        else:
            p = 1 / (1 + np.exp(-z))
        p[~ok[dd.index]] = np.nan
        return p

    pred = H.walk_forward(df, fit, predict, start, end, train_days=train_days, min_train=min_train,
                          refit_days=refit_days)
    if keep_coefs:
        return pred, coefs
    return pred


def ll(df, col, window, mask=None):
    w = H.in_window(df, window) & df.y.notna() & df[col].notna()
    if mask is not None:
        w &= mask
    return float(H.logloss_each(df.y[w].to_numpy(), df[col][w].to_numpy()).mean()), int(w.sum())
