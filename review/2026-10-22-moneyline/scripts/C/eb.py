"""Empirical-Bayes shrunk, recency-weighted raw rates (not opponent-adjusted), as of each kick-off, and each
side's current streak.

Per gamer and stat, his offence's and the offences he faced (defence): sum(w num) / sum(w den) over his
play-by-play matches settled by kick-off (RESULT_LAG), w = 2^(-k/HL) for his k-th most recent, shrunk to the
league (recency-weighted, 30-day half-life) by m = noise per unit of denominator / between-gamer variance
(kalman.variance_components on 05-01..07-31): m units of denominator at the league rate.
Writes out/C/eb_feats_hl<HL>.pkl: per match eb_<stat>_{offh,defh,offa,defa,lg}, plus streak_p1/p2.
"""

import json
import sys

import numpy as np
import pandas as pd

from common import H, OUT, STATS, side_obs

LEAGUE_HL_DAYS = 30.0


def build(hl):
    df = H.load(tables=False, elo=False)
    obs, has = side_obs(df)
    vc = json.load(open(f"{OUT}/kalman_vc.json"))["stats"]
    keys = list(STATS)
    K = len(keys)
    gamers = sorted(set(df.p1) | set(df.p2))
    gi = {g: i for i, g in enumerate(gamers)}
    G = len(gamers)
    m_off = np.array([(vc[k]["b"] if vc[k]["b"] > 1e-9 else vc[k]["R_A"]) / vc[k]["V_off"] for k in keys])
    m_def = np.array([(vc[k]["b"] if vc[k]["b"] > 1e-9 else vc[k]["R_A"]) / vc[k]["V_def"] for k in keys])
    # numerators/denominators per side-match
    Yh = np.stack([obs[k][0] for k in keys], 1)
    Nh = np.stack([obs[k][1] for k in keys], 1)
    Ya = np.stack([obs[k][2] for k in keys], 1)
    Na = np.stack([obs[k][3] for k in keys], 1)
    NUMh, NUMa = np.nan_to_num(Yh * Nh), np.nan_to_num(Ya * Na)
    DENh, DENa = np.nan_to_num(Nh), np.nan_to_num(Na)
    decay = 2.0 ** (-1.0 / hl)
    on, od = np.zeros((G, K)), np.zeros((G, K))      # offence num, den
    dn, dd = np.zeros((G, K)), np.zeros((G, K))      # defence (offences faced)
    ln, ld = np.zeros(K), np.zeros(K)
    lt = None
    n = len(df)
    starts = df.start.to_numpy()
    settle = starts + np.timedelta64(int(H.RESULT_LAG.total_seconds()), "s")
    order = np.argsort(settle, kind="stable")
    p1 = df.p1.map(gi).to_numpy()
    p2 = df.p2.map(gi).to_numpy()
    y = df.y.to_numpy()
    streak = np.zeros(G)
    out = np.full((n, K, 5), np.nan)
    st = np.zeros((n, 2))
    j = 0
    for i in range(n):
        t = starts[i]
        while j < n and settle[order[j]] <= t:
            mi = order[j]
            j += 1
            a, b = p1[mi], p2[mi]
            # streaks off every settled result (ties end a streak)
            s1, s2 = df.s1.iat[mi], df.s2.iat[mi]
            if np.isfinite(s1) and np.isfinite(s2):
                for g, won in ((a, s1 > s2), (b, s2 > s1)):
                    if s1 == s2:
                        streak[g] = 0
                    elif won:
                        streak[g] = streak[g] + 1 if streak[g] > 0 else 1
                    else:
                        streak[g] = streak[g] - 1 if streak[g] < 0 else -1
            if not has[mi]:
                continue
            tt = settle[mi]
            if lt is not None:
                f = 2.0 ** (-((tt - lt) / np.timedelta64(1, "D")) / LEAGUE_HL_DAYS)
                ln *= f
                ld *= f
            lt = tt
            ln += NUMh[mi] + NUMa[mi]
            ld += DENh[mi] + DENa[mi]
            for g, num_o, den_o, num_d, den_d in ((a, NUMh[mi], DENh[mi], NUMa[mi], DENa[mi]),
                                                  (b, NUMa[mi], DENa[mi], NUMh[mi], DENh[mi])):
                on[g] = on[g] * decay + num_o
                od[g] = od[g] * decay + den_o
                dn[g] = dn[g] * decay + num_d
                dd[g] = dd[g] * decay + den_d
        if ld.sum() == 0:
            continue
        lg = ln / np.maximum(ld, 1e-12)
        a, b = p1[i], p2[i]
        out[i, :, 0] = (on[a] + m_off * lg) / (od[a] + m_off)
        out[i, :, 1] = (dn[a] + m_def * lg) / (dd[a] + m_def)
        out[i, :, 2] = (on[b] + m_off * lg) / (od[b] + m_off)
        out[i, :, 3] = (dn[b] + m_def * lg) / (dd[b] + m_def)
        out[i, :, 4] = lg
        st[i] = streak[a], streak[b]
    res = pd.DataFrame({"code": df.code})
    for c, k in enumerate(keys):
        for j2, nm in enumerate(("offh", "defh", "offa", "defa", "lg")):
            res[f"eb_{k}_{nm}"] = out[:, c, j2]
    res["streak_p1"], res["streak_p2"] = st[:, 0], st[:, 1]
    res.to_pickle(f"{OUT}/eb_feats_hl{hl}.pkl")
    print("wrote", f"{OUT}/eb_feats_hl{hl}.pkl", res.shape, flush=True)


if __name__ == "__main__":
    for hl in [int(x) for x in sys.argv[1:]] or [30]:
        build(hl)
