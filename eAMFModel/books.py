"""A pre-match prior's own price and its grading: the margin and total distributions its expected points
imply under its score model (glmer's log-normal Poisson, NB2's construction, a walkthrough table's own),
and those books graded on the results (moneyline log loss and Brier, the margin's and total's bias, RMSE,
RPS and coverage). Shared by every version's checks (prior_check for v14, v21-report for v2.1)."""

import math

import numpy as np

from . import markets, nb2_prior

RAW_DRAWS = 20000


def _start(r):
    return nb2_prior._start(r)


def _finals(r):
    try:
        return float(r["PLAYER_1_FINAL_SCORE"]), float(r["PLAYER_2_FINAL_SCORE"])
    except (KeyError, TypeError, ValueError):
        return None


def base_prior(pre):
    """The prior under the build's layers (shrink, follow, form): the model itself."""
    while hasattr(pre, "pre"):
        pre = pre.pre
    return pre


def sigmas(base, day):
    """(sigma_obs, sigma_match) of the fit that prices a match on `day`: a daily prior's fit for
    that day, else the single fit; (0, 0) when the fit saves none (NB2: plain Poisson)."""
    fit = base.fit_for(day) if hasattr(base, "fit_for") else base
    m = (getattr(fit, "meta", None) or {}).get("model", {}) or {}
    return float(m.get("sigma_obs") or 0.0), float(m.get("sigma_match") or 0.0)


def raw_books(means, s_obs, s_match, n=RAW_DRAWS, seed=0):
    """Margin and total distributions (v14's shapes) from expected points under glmer's score model:
    each side Poisson around exp(eta + noise), eta set so its mean is the expected points."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0.0, s_match, n) if s_match else np.zeros(n)
    sides = []
    for mu in means:
        eta = math.log(max(mu, 0.1)) - (s_obs ** 2 + s_match ** 2) / 2
        noise = rng.normal(0.0, s_obs, n) if s_obs else np.zeros(n)
        sides.append(rng.poisson(np.exp(np.clip(eta + shared + noise, -15, 15))))
    return markets._distributions(sides[0].astype(int), sides[1].astype(int))


def nb2_books(means, a1, a2, shared, n=RAW_DRAWS, seed=0):
    """Margin and total distributions (v14's shapes) under NB2's own construction: each side negative
    binomial (variance mu + alpha mu^2), the two sharing a gamma pace factor (eAMFCalibrator's
    score_model.nb2_draws)."""
    rng = np.random.default_rng(seed)
    r1, r2 = 1.0 / max(a1, 1e-4), 1.0 / max(a2, 1e-4)
    k = min(max(shared, 0.0), 0.999) * min(r1, r2)
    u = rng.gamma(k, 1.0, n) if k > 1e-12 else np.zeros(n)
    v1, v2 = rng.gamma(max(r1 - k, 1e-12), 1.0, n), rng.gamma(max(r2 - k, 1e-12), 1.0, n)
    mu1, mu2 = means
    return markets._distributions(rng.poisson((mu1 / r1) * (u + v1)).astype(int), rng.poisson((mu2 / r2) * (u + v2)).astype(int))


def table_books(means, score, n=RAW_DRAWS, seed=0):
    """A walkthrough table's own score model on its means: (kind, params) nb2 (alpha1, alpha2, shared)
    or lognormal (sigma_obs, sigma_match); plain Poisson for anything else."""
    kind, params = score
    if kind == "nb2" and len(params) >= 3:
        return nb2_books(means, *params[:3], n=n, seed=seed)
    if kind == "lognormal" and len(params) >= 2:
        return raw_books(means, *params[:2], n=n, seed=seed)
    return raw_books(means, 0.0, 0.0, n=n, seed=seed)


def _cdf_scores(pmf, offset, actual):
    """(expected value, RPS, inside the central 50%, inside the central 80%) for one distribution
    over values k - offset."""
    k = np.arange(len(pmf)) - offset
    cdf = np.cumsum(pmf)
    rps = float(((cdf - (k >= actual)) ** 2).sum())
    lo50, hi50 = k[np.searchsorted(cdf, 0.25)], k[min(len(k) - 1, np.searchsorted(cdf, 0.75))]
    lo80, hi80 = k[np.searchsorted(cdf, 0.10)], k[min(len(k) - 1, np.searchsorted(cdf, 0.90))]
    return float((k * pmf).sum()), rps, lo50 <= actual <= hi50, lo80 <= actual <= hi80


def grade(results, buckets=10, methods=None):
    """{method: summary dict} and the calibration tables, for `methods` (default: the first result's):
    the keys of every result's books."""
    summary, tables = {}, {}
    for m in methods or list(results[0]["books"]):
        p, won, mstats, tstats = [], [], [], []
        for x in results:
            mp_, tp = x["books"][m]
            s1, s2 = x["finals"]
            margin, total = s1 - s2, s1 + s2
            mid = markets.MARGIN_MAX
            if margin != 0:
                push = mp_[mid]
                p.append(float(mp_[mid + 1:].sum() / max(1e-12, 1 - push)))
                won.append(1.0 if margin > 0 else 0.0)
            mstats.append(_cdf_scores(mp_, mid, margin) + (margin,))
            tstats.append(_cdf_scores(tp, 0, total) + (total,))
        p, won = np.clip(np.array(p), 1e-9, 1 - 1e-9), np.array(won)
        ll = -(won * np.log(p) + (1 - won) * np.log(1 - p))
        out = dict(n=len(results), ml_n=len(p), log_loss=float(ll.mean()),
                   brier=float(((p - won) ** 2).mean()), ll=ll)
        for name, st in (("margin", mstats), ("total", tstats)):
            a = np.array([s[:4] + (s[4],) for s in st], dtype=float)
            err = a[:, 4] - a[:, 0]
            out[name] = dict(bias=float(err.mean()), rmse=float(math.sqrt((err ** 2).mean())),
                             rps=float(a[:, 1].mean()), in50=float(a[:, 2].mean()), in80=float(a[:, 3].mean()))
        order = np.argsort(p)
        rows = []
        for chunk in np.array_split(order, min(buckets, len(p))):
            if len(chunk):
                rows.append((float(p[chunk].mean()), float(won[chunk].mean()), len(chunk)))
        summary[m], tables[m] = out, rows
    return summary, tables


def paired(summary, a, b):
    """(mean, standard error) of method a's moneyline log loss less b's, match by match."""
    d = summary[a]["ll"] - summary[b]["ll"]
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
