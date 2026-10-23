"""Online Kalman filters (the online equivalent of the time-decayed ridges): every result enters at kick-off + 36 min,
every match is priced at its kick-off off the state then (so a late match uses the day's earlier results).

  match-level  y = mu + stream + a[p1] + a[p2] + t[t1] + t[t2] + e                       (total, drives)
  side-level   y = mu + home + stream + off[g] + def[opp] + offT[team] + defT[opp team] + e   (points a drive: R / drives)

States: mu drifts q_mu a day (diffuse start); gamers' effects start N(0, V_g) when first met and drift q_g a day
from then; teams N(0, V_t), drift q_t; streams N(0, V_s) static. Variances relative to the noise R (only the ratios
move the mean). Full covariance, so every result moves the whole design (opponent-adjusted).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402


def _events(df, y_ok):
    """(time in days, type, row): updates (type 0) at settled for rows with a result, prices (type 1) at kick-off;
    an update at the same moment as a price goes first (a result 36 min old counts)."""
    t0 = np.datetime64("2025-12-01")
    ks = (df.start.values - t0).astype("timedelta64[s]").astype(float) / 86400.0
    st = (df.settled.values - t0).astype("timedelta64[s]").astype(float) / 86400.0
    n = len(df)
    ev_t = np.concatenate([st[y_ok], ks])
    ev_k = np.concatenate([np.zeros(int(y_ok.sum()), int), np.ones(n, int)])
    ev_r = np.concatenate([np.flatnonzero(y_ok), np.arange(n)])
    o = np.lexsort((ev_k, ev_t))
    return ev_t[o], ev_k[o], ev_r[o]


def run(df, kind, y, p, w=None, obs_scale=None):
    """kind 'match' or 'side'. y: (n,) for match, (n, 2) home/away for side. p: dict with V_g, q_g, V_t, q_t, q_mu,
    V_s (ratios to R=1) [, V_home]. w: observation weights (R / w), same shape as y. obs_scale: per-row multiplier
    of R (e.g. a variance function of the prediction) - callable(pred) or None.
    Returns (pred, pred_var) at kick-off: (n,) or (n, 2) for side."""
    ix = F.Index(df)
    ng, nt, ns = ix.ng, ix.nt, ix.ns
    side = kind == "side"
    # state layout
    o_mu = 0
    o_home = 1
    o_s = 2                                   # streams 1..ns-1
    o_g = o_s + ns - 1
    if side:
        o_d = o_g + ng
        o_t = o_d + ng
        o_td = o_t + nt
        nstate = o_td + nt
    else:
        o_t = o_g + ng
        nstate = o_t + nt
    x = np.zeros(nstate)
    P = np.zeros((nstate, nstate))
    P[o_mu, o_mu] = 1e3
    P[o_home, o_home] = p.get("V_home", 0.05) if side else 0.0
    for k in range(ns - 1):
        P[o_s + k, o_s + k] = p["V_s"]
    q = np.zeros(nstate)
    q[o_mu] = p["q_mu"]
    active = np.zeros(nstate, bool)
    active[o_mu] = True
    g1 = df.p1.map(ix.g).to_numpy()
    g2 = df.p2.map(ix.g).to_numpy()
    t1 = df.t1.map(ix.t).to_numpy()
    t2 = df.t2.map(ix.t).to_numpy()
    s_ = df.stream.map(ix.s).to_numpy()
    yv = np.asarray(y, float)
    wv = np.ones_like(yv) if w is None else np.asarray(w, float)
    ok = ~np.isnan(yv) & ~np.isnan(wv) & (wv > 0)
    y_ok = ok.any(1) if side else ok

    def meet(idx, V, qq):
        if not active[idx]:
            active[idx] = True
            P[idx, idx] = V
            q[idx] = qq

    def h_of(r, home):
        h = np.zeros(nstate)
        h[o_mu] = 1
        if s_[r] > 0:
            h[o_s + s_[r] - 1] = 1
        if side:
            go, gd, to, td = (g1[r], g2[r], t1[r], t2[r]) if home else (g2[r], g1[r], t2[r], t1[r])
            if home:
                h[o_home] = 1
            for idx, V, qq in ((o_g + go, p["V_g"], p["q_g"]), (o_d + gd, p["V_g"], p["q_g"]),
                               (o_t + to, p["V_t"], p["q_t"]), (o_td + td, p["V_t"], p["q_t"])):
                meet(idx, V, qq)
                h[idx] += 1
        else:
            for idx, V, qq in ((o_g + g1[r], p["V_g"], p["q_g"]), (o_g + g2[r], p["V_g"], p["q_g"]),
                               (o_t + t1[r], p["V_t"], p["q_t"]), (o_t + t2[r], p["V_t"], p["q_t"])):
                meet(idx, V, qq)
                h[idx] += 1
        return h

    n = len(df)
    pred = np.full((n, 2) if side else n, np.nan)
    pvar = np.full((n, 2) if side else n, np.nan)
    ev_t, ev_k, ev_r = _events(df, y_ok)
    last = ev_t[0]
    di = np.diag_indices(nstate)
    for t, kk, r in zip(ev_t, ev_k, ev_r):
        dt = t - last
        if dt > 0:
            P[di] += q * active * dt
            last = t
        if kk == 1:
            for j, home in (((0, True), (1, False)) if side else ((None, True),)):
                h = h_of(r, home)
                if side:
                    pred[r, j] = h @ x
                    pvar[r, j] = h @ P @ h
                else:
                    pred[r] = h @ x
                    pvar[r] = h @ P @ h
        else:
            for j, home in (((0, True), (1, False)) if side else ((None, True),)):
                yy = yv[r, j] if side else yv[r]
                good = ok[r, j] if side else ok[r]
                if not good:
                    continue
                h = h_of(r, home)
                R = 1.0 / (wv[r, j] if side else wv[r])
                if obs_scale is not None:
                    R *= obs_scale(h @ x)
                Ph = P @ h
                s = h @ Ph + R
                K = Ph / s
                x += K * (yy - h @ x)
                P -= np.outer(K, Ph)
    return pred, pvar
