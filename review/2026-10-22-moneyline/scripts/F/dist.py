"""Total distributions on the integers 0..K-1 (a matrix of PMFs, one row a match) and their scores.

normal_pmf     a normal about mu, sd, on whole points (k-.5, k+.5]
nb_pmf         a negative binomial with mean mu and variance var
lumpy          any base PMF x per-integer weights w_k (learned), its mean put back to mu
compound_pmf   the drives x points-a-drive model: D ~ discretised normal(muD, sigD); each drive scores 7 (or 6 / 8
               after the try) with prob p_t, 3 with p_f, else 0; the rates scale with (D / muD)^beta (a game with
               more drives than expected scores less a drive) and a match-level lognormal factor (sd tau); the
               rates' level set so the total's mean is mu. Built through the characteristic function.
"""
import numpy as np
from scipy import special, stats

K = 160
L = 256


def _tail_fix(pmf):
    pmf = np.clip(pmf, 1e-12, None)
    return pmf / pmf.sum(1, keepdims=True)


def normal_pmf(mu, sd, K=K):
    mu, sd = np.asarray(mu, float)[:, None], np.asarray(sd, float)[:, None]
    k = np.arange(K)[None, :]
    cdf = special.ndtr((k + 0.5 - mu) / sd)
    cdf[:, -1] = 1.0
    pmf = np.diff(np.concatenate([np.zeros((cdf.shape[0], 1)), cdf], 1), axis=1)
    return _tail_fix(pmf)


def nb_pmf(mu, var, K=K):
    mu = np.asarray(mu, float)
    var = np.maximum(np.asarray(var, float), mu * 1.02)
    r = mu ** 2 / (var - mu)
    p = r / (r + mu)
    k = np.arange(K)
    pmf = stats.nbinom.pmf(k[None, :], r[:, None], p[:, None])
    pmf[:, -1] += np.clip(1 - pmf.sum(1), 0, None)
    return _tail_fix(pmf)


def pmf_mean(pmf):
    return pmf @ np.arange(pmf.shape[1])


def pmf_sd(pmf):
    k = np.arange(pmf.shape[1])
    m = pmf @ k
    return np.sqrt(np.clip(pmf @ k ** 2 - m ** 2, 0, None))


def lumpy(base_fn, mu, sdish, w, iters=3):
    """base_fn(mu', sdish) -> PMF; multiply by w, renormalise, shift mu' so the mean is mu again."""
    m = np.asarray(mu, float).copy()
    for _ in range(iters):
        pmf = base_fn(m, sdish) * w[None, :]
        pmf /= pmf.sum(1, keepdims=True)
        m = m + (mu - pmf_mean(pmf))
    pmf = base_fn(m, sdish) * w[None, :]
    return _tail_fix(pmf / pmf.sum(1, keepdims=True))


def fit_lumps(pmfs, totals, alpha=5.0, iters=8, base_fn=None):
    """Per-integer weights w_k by iterative proportional fitting: observed counts / expected counts (pmfs already
    with the current w; a fixed-base approximation), shrunk toward 1 by alpha pseudo-counts."""
    Kk = pmfs.shape[1]
    obs = np.bincount(np.clip(totals.astype(int), 0, Kk - 1), minlength=Kk).astype(float)
    w = np.ones(Kk)
    for _ in range(iters):
        q = pmfs * w[None, :]
        q /= q.sum(1, keepdims=True)
        exp = q.sum(0)
        w = w * (obs + alpha) / (exp + alpha)
        w /= np.average(w, weights=obs + 1)
    return w


def gh_nodes(n):
    x, wt = np.polynomial.hermite_e.hermegauss(n)
    return x, wt / wt.sum()


def compound_pmf(mu, muD, sigD, beta, tau, pt, pf, q6, q8, nz=7, K=K, chunk=200, dmax=34):
    """See the module doc. mu, muD, pt, pf: arrays (pt, pf: the match's touchdown / field goal rates a drive;
    only their mix matters, the level is scaled to meet mu). sigD: scalar or array. Returns PMFs (n x K) and the
    scale k each match needed."""
    mu = np.asarray(mu, float)
    n = len(mu)
    muD = np.asarray(muD, float)
    sigD = np.broadcast_to(np.asarray(sigD, float), (n,))
    pt, pf = np.asarray(pt, float), np.asarray(pf, float)
    q7 = 1.0 - q6 - q8
    Dg = np.arange(1, dmax + 1, dtype=float)
    om = np.arange(L // 2 + 1)
    e = lambda j: np.exp(-2j * np.pi * om * j / L)
    e3 = e(3)
    Q = q6 * e(6) + q7 * e(7) + q8 * e(8)
    tdpts = 6 * q6 + 7 * q7 + 8 * q8
    if tau > 0:
        zx, zw = gh_nodes(nz)
        z = np.exp(tau * zx - tau ** 2 / 2)
    else:
        z, zw = np.array([1.0]), np.array([1.0])
    out = np.empty((n, K))
    ks = np.empty(n)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        m = b - a
        # P(D)
        if np.all(sigD[a:b] > 0):
            lo = special.ndtr((Dg[None, :] - 0.5 - muD[a:b, None]) / sigD[a:b, None])
            hi = special.ndtr((Dg[None, :] + 0.5 - muD[a:b, None]) / sigD[a:b, None])
            pD = hi - lo
        else:
            pD = np.zeros((m, len(Dg)))
            pD[np.arange(m), np.clip(np.rint(muD[a:b]).astype(int) - 1, 0, len(Dg) - 1)] = 1.0
        pD /= pD.sum(1, keepdims=True)
        rel = (Dg[None, :] / muD[a:b, None]) ** beta                     # m x D
        eper = tdpts * pt[a:b] + 3 * pf[a:b]                               # points a drive at scale 1
        expo = (pD * Dg[None, :] * rel).sum(1) * eper                      # E[total] at scale 1
        k = mu[a:b] / expo
        ks[a:b] = k
        cf = np.zeros((m, len(om)), complex)
        for zi, wz in zip(z, zw):
            s = (k * zi)[:, None] * rel                                    # m x D
            ptd = np.clip(s * pt[a:b, None], 0, 0.9)
            pfg = np.clip(s * pf[a:b, None], 0, 0.97 - ptd)
            phi = 1.0 + pfg[:, :, None] * (e3[None, None, :] - 1.0) + ptd[:, :, None] * (Q[None, None, :] - 1.0)
            # phi^D, by logs (phi never crosses 0 here: |phi - 1| < 1 needs care) -> use repeated powers
            powD = np.power(phi, Dg[None, :, None])
            cf += wz * (pD[:, :, None] * powD).sum(1)
        pmf = np.fft.irfft(cf, n=L, axis=1)[:, :K]
        out[a:b] = pmf
    return _tail_fix(out), ks


def scores(pmf, total, line=None):
    """Per match: log score of the whole total, RPS over the integers, and P(total > line) for x.5 lines."""
    total = np.asarray(total, float)
    n, Kk = pmf.shape
    idx = np.clip(total.astype(int), 0, Kk - 1)
    logs = -np.log(pmf[np.arange(n), idx])
    cdf = np.cumsum(pmf, 1)
    step = (np.arange(Kk)[None, :] >= total[:, None]).astype(float)
    rps = ((cdf - step) ** 2).sum(1)
    p_over = None
    if line is not None:
        line = np.asarray(line, float)
        ok = ~np.isnan(line)
        p_over = np.full(n, np.nan)
        fl = np.clip(np.floor(line[ok]).astype(int), 0, Kk - 1)
        p_over[ok] = 1.0 - cdf[np.flatnonzero(ok), fl]
    return logs, rps, p_over


def p_over_at(pmf, line):
    cdf = np.cumsum(pmf, 1)
    line = np.asarray(line, float)
    out = np.full(len(line), np.nan)
    ok = ~np.isnan(line)
    fl = np.clip(np.floor(line[ok]).astype(int), 0, pmf.shape[1] - 1)
    out[ok] = 1.0 - cdf[np.flatnonzero(ok), fl]
    return out
