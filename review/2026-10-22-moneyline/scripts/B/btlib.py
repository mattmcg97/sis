import common  # noqa: F401,E402  (sets single-thread BLAS before numpy loads)
"""Static penalised Bradley-Terry (logistic on win/loss) with time-decayed weights, sparse design, L-BFGS."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve
from common import H, ngamers, nteams

NG, NT, NS = 89, 10, 3


def design(df, spec):
    """Sparse design for rows of df. spec: dict with keys
    home: 'one' | 'stream'; team: bool; gt: bool (gamer x team, +1 home pair, -1 away pair);
    ghome: bool (gamer-specific home advantage, only the home gamer's column);
    Returns (X csr, blocks list of (name, slice))."""
    n = len(df)
    rows, cols, vals = [], [], []
    blocks = []
    off = 0
    r = np.arange(n)
    if spec.get("home", "one") == "stream":
        rows += [r]; cols += [off + df.st.to_numpy()]; vals += [np.ones(n)]
        blocks.append(("home", slice(off, off + NS))); off += NS
    else:
        rows += [r]; cols += [np.full(n, off)]; vals += [np.ones(n)]
        blocks.append(("home", slice(off, off + 1))); off += 1
    g1, g2 = df.g1.to_numpy(), df.g2.to_numpy()
    rows += [r, r]; cols += [off + g1, off + g2]; vals += [np.ones(n), -np.ones(n)]
    blocks.append(("gamer", slice(off, off + NG))); off += NG
    if spec.get("team"):
        k1, k2 = df.k1.to_numpy(), df.k2.to_numpy()
        rows += [r, r]; cols += [off + k1, off + k2]; vals += [np.ones(n), -np.ones(n)]
        blocks.append(("team", slice(off, off + NT))); off += NT
    if spec.get("gt"):
        k1, k2 = df.k1.to_numpy(), df.k2.to_numpy()
        rows += [r, r]; cols += [off + g1 * NT + k1, off + g2 * NT + k2]; vals += [np.ones(n), -np.ones(n)]
        blocks.append(("gt", slice(off, off + NG * NT))); off += NG * NT
    if spec.get("ghome"):
        rows += [r]; cols += [off + g1]; vals += [np.ones(n)]
        blocks.append(("ghome", slice(off, off + NG))); off += NG
    if spec.get("pair"):
        # antisymmetric pair effect: index min,max; sign +1 if g1<g2
        a, b = np.minimum(g1, g2), np.maximum(g1, g2)
        sgn = np.where(g1 < g2, 1.0, -1.0)
        rows += [r]; cols += [off + a * NG + b]; vals += [sgn]
        blocks.append(("pair", slice(off, off + NG * NG))); off += NG * NG
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, off))
    return X, blocks


def penalty_vector(blocks, lam, p):
    pen = np.zeros(p)
    for name, s in blocks:
        pen[s] = lam.get(name, 0.0)
    return pen


def fit_bt(X, y, w, pen, theta0=None, maxiter=50, tol=1e-8):
    """Penalised weighted logistic by Newton (dense Hessian from a sparse design), step-halving."""
    p = X.shape[1]
    th = np.zeros(p) if theta0 is None or len(theta0) != p else theta0.copy()
    XT = X.T.tocsr()
    ridge = np.where(pen > 0, 0.0, 1e-6)

    def obj(t):
        z = X @ t
        return float(w @ (np.logaddexp(0, z) - y * z) + 0.5 * pen @ (t * t))

    f0 = obj(th)
    for it in range(maxiter):
        z = X @ th
        mu = 1 / (1 + np.exp(-z))
        g = XT @ ((mu - y) * w) + pen * th
        Hsp = XT @ sp.diags(w * mu * (1 - mu)) @ X + sp.diags(pen + ridge)
        if p > 300:
            step = spsolve(Hsp.tocsc(), g)
        else:
            Hs = Hsp.toarray()
            try:
                step = np.linalg.solve(Hs, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(Hs, g, rcond=None)[0]
        a = 1.0
        while True:
            tn = th - a * step
            fn = obj(tn)
            if fn <= f0 + 1e-12 or a < 1e-4:
                break
            a *= 0.5
        dec = f0 - fn
        th, f0 = tn, fn
        if dec < tol * max(1.0, abs(f0)) and it > 0:
            break
    return th


def fit_bt_lbfgs(X, y, w, pen, theta0=None, maxiter=500):
    p = X.shape[1]
    th = np.zeros(p) if theta0 is None else theta0.copy()
    XT = X.T.tocsr()

    def f(t):
        z = X @ t
        ll = np.logaddexp(0, z) - y * z
        r = (1 / (1 + np.exp(-z)) - y) * w
        return float(w @ ll + 0.5 * pen @ (t * t)), XT @ r + pen * t

    res = minimize(f, th, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter, gtol=1e-6))
    return res.x


def make_bt(spec, lam, half_life, mov=None):
    """fit/predict closures for H.walk_forward. half_life in days (None = no decay).
    mov: optional (a, c): weight multiplier (|margin|+c)^a / mean -- MOV-weighted variant."""
    state = {"theta": None}

    def fit(tr):
        X, blocks = design(tr, spec)
        cut = tr.attrs.get("cut")
        tnow = tr.tday.max() + 1.0  # reference; decay relative to last training day (constant shift cancels in ratios)
        w = np.ones(len(tr)) if not half_life else 0.5 ** ((tnow - tr.tday.to_numpy()) / half_life)
        if mov is not None:
            a, c = mov
            m = (np.abs(tr.margin.to_numpy()) + c) ** a
            w = w * m / m.mean()
        pen = penalty_vector(blocks, lam, X.shape[1])
        th = fit_bt(X, tr.y.to_numpy(), w, pen, state["theta"])
        state["theta"] = th
        return th

    def predict(th, day):
        X, _ = design(day, spec)
        return H.sigmoid(X @ th)

    return fit, predict
