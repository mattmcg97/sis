"""Opponent-adjusted, shrunk, time-varying stat ratings: one full-covariance Kalman filter per stat.

For stat k, a side's observed rate in a match (its offence's: e.g. red-zone TDs over its trips) is
    y = mu + home*[offence at home] + off[offence gamer] + def[defence gamer] + offT[offence's team]
          + defT[defence's team] + e,      var(e) = a + b / n   (n the rate's denominator)
mu and home drift slowly; every gamer's off and def effect starts at N(0, V_off / V_def) (the between-gamer
spread of true rates, by split-half covariance on 05-01..07-31) and drifts q a day once he has played; teams
N(0, V_team) drifting slowly. The full covariance is kept, so a gamer's numbers are read net of whom he has
played (a joint ridge, time-decayed, refreshed on every result) and shrunk toward the league by the size of
his sample (a 3-trip red zone moves little). Results enter RESULT_LAG after kick-off; each match's features
are the state at its kick-off.

  --tune   picks q per stat from a grid by the one-step-ahead predictive log likelihood of the stat itself
           (side-matches 05-15..07-31): no win/loss, nothing after 07-31
  default  runs the chosen settings over all matches and writes out/C/kalman_feats.pkl
Also the strength margin: the same filter on points a match over the whole history (2025-12-10 on).
"""

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from common import H, OUT, STATS, side_obs

LAG = H.RESULT_LAG
Q_GRID = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05)      # gamer drift a day, as a share of V
TEAM_Q = 0.002
EST = ("2026-05-01", "2026-08-01")                   # variance components estimated here (pre-TEST)
TUNE_LL = ("2026-05-15", "2026-08-01")


def variance_components(df, obs, has, keys):
    """Per stat: league mean, R's a and b, V_off, V_def (gamer-month split-half covariance), V_team_off/def."""
    w = H.in_window(df, EST).to_numpy() & has
    out = {}
    sub = df.loc[w, ["p1", "p2", "t1", "t2", "day"]].reset_index(drop=True)
    for k in keys:
        yh, nh, ya, na = (o[w] for o in obs[k])
        y = np.r_[yh, ya]
        n = np.r_[nh, na]
        off = np.r_[sub.p1, sub.p2]
        dfn = np.r_[sub.p2, sub.p1]
        toff = np.r_[sub.t1, sub.t2]
        tdef = np.r_[sub.t2, sub.t1]
        mon = np.r_[sub.day.str[:7], sub.day.str[:7]]
        f = np.isfinite(y) & np.isfinite(n)
        y, n, off, dfn, toff, tdef, mon = y[f], n[f], off[f], dfn[f], toff[f], tdef[f], mon[f]
        lg = np.sum(y * n) / np.sum(n)
        r2 = (y - lg) ** 2
        per_match = STATS[k][1] is None
        # first a rough noise level for the team step's sampling variance
        if per_match:
            A, b = float(r2.mean()), 0.0
        else:
            big = n >= np.quantile(n, 0.25)
            A, b = 0.0, float(np.mean(r2[big] * n[big]))
        # gamer-month split-half covariance of pooled rates
        T = pd.DataFrame({"y": y, "n": n, "off": off, "def": dfn, "mon": mon, "toff": toff, "tdef": tdef})
        V = {}
        for role in ("off", "def"):
            T["k"] = T.groupby([role, "mon"]).cumcount() % 2
            g = T.assign(yn=T.y * T.n).groupby([role, "mon", "k"])[["yn", "n"]].sum()
            c = T.groupby([role, "mon", "k"]).size()
            rate = (g.yn / g.n).unstack("k")
            cc = c.unstack("k")
            ok = (cc[0] >= 20) & (cc[1] >= 20)
            rate = rate[ok]
            rate = rate - rate.groupby(level="mon").transform("mean")
            V[role] = max(np.cov(rate[0], rate[1])[0, 1], 1e-4 * (A + b))
            # team: pooled over the window, less sampling variance
            g2 = T.assign(yn=T.y * T.n).groupby("t" + role)[["yn", "n"]].sum()
            c2 = T.groupby("t" + role).size()
            tr = (g2.yn / g2.n)[c2 >= 200]
            samp = ((A + b / T.n.mean()) / c2[c2 >= 200]).mean()
            V["team_" + role] = max(tr.var() - samp, 1e-4 * (A + b))
        Vtot = V["off"] + V["def"] + V["team_off"] + V["team_def"]
        # the noise about a side's true rate: var(y | n) = R_A + b / n, the true spread taken out
        if per_match:
            R_A, b = max(A - Vtot, 0.2 * A), 0.0
        else:
            R_A, b = 0.0, max(float(np.mean((r2[big] - Vtot) * n[big])), 0.2 * b)
        out[k] = dict(league=lg, A=A, b=b, R_A=R_A,
                      V_off=V["off"], V_def=V["def"], V_team_off=V["team_off"], V_team_def=V["team_def"],
                      nbar=float(n.mean()))
    return out


class Filter:
    def __init__(self, gamers, teams, params, q_frac):
        """params: list of per-channel dicts (variance components); q_frac: per-channel gamer drift."""
        self.G, self.T = len(gamers), len(teams)
        self.gi = {g: i for i, g in enumerate(gamers)}
        self.ti = {t: i for i, t in enumerate(teams)}
        G, T = self.G, self.T
        self.d = d = 2 + 2 * G + 2 * T
        C = self.C = len(params)
        self.x = np.zeros((C, d))
        self.P = np.zeros((C, d, d))
        self.RA = np.array([p["R_A"] for p in params])
        self.Rb = np.array([p["b"] for p in params])
        self.qg = np.zeros((C, d))
        for c, p in enumerate(params):
            tot = p["A"] + p["b"] / max(p["nbar"], 1e-9)
            self.x[c, 0] = 0.0
            self.P[c, 0, 0] = 25.0 * tot            # mu: diffuse, learnt from the first results
            self.P[c, 1, 1] = 0.01 * tot             # home
            self.P[c, 2:2 + G, 2:2 + G][np.diag_indices(G)] = p["V_off"]
            self.P[c, 2 + G:2 + 2 * G, 2 + G:2 + 2 * G][np.diag_indices(G)] = p["V_def"]
            o = 2 + 2 * G
            self.P[c, o:o + T, o:o + T][np.diag_indices(T)] = p["V_team_off"]
            self.P[c, o + T:o + 2 * T, o + T:o + 2 * T][np.diag_indices(T)] = p["V_team_def"]
            self.qg[c, 0] = 1e-4 * tot
            self.qg[c, 1] = 1e-5 * tot
            self.qg[c, 2:2 + G] = q_frac[c] * p["V_off"]
            self.qg[c, 2 + G:2 + 2 * G] = q_frac[c] * p["V_def"]
            self.qg[c, o:o + T] = TEAM_Q * p["V_team_off"]
            self.qg[c, o + T:o + 2 * T] = TEAM_Q * p["V_team_def"]
        self.active = np.zeros(d, bool)
        self.active[[0, 1]] = True
        self.active[2 + 2 * G:] = True
        self.t = None
        self.diag = np.arange(d) * (d + 1)

    def advance(self, t):
        if self.t is not None and t > self.t:
            dt = (t - self.t).total_seconds() / 86400.0
            idx = self.diag[self.active]
            Pf = self.P.reshape(self.C, -1)
            Pf[:, idx] += self.qg[:, self.active] * dt
        self.t = t if self.t is None else max(self.t, t)

    def idx(self, off_g, def_g, off_t, def_t, home):
        G, T = self.G, self.T
        ix = [0]
        if home:
            ix.append(1)
        ix += [2 + self.gi[off_g], 2 + G + self.gi[def_g]]
        if off_t in self.ti and def_t in self.ti:
            ix += [2 + 2 * G + self.ti[off_t], 2 + 2 * G + T + self.ti[def_t]]
        return ix

    def activate(self, *gamers):
        for g in gamers:
            i = self.gi[g]
            self.active[2 + i] = True
            self.active[2 + self.G + i] = True

    def update(self, ix, y, n):
        """y, n: (C,) observations (NaN = no observation for that channel)."""
        ok = np.isfinite(y) & np.isfinite(n) & (n > 0)
        if not ok.any():
            return None
        Ph = self.P[:, :, ix].sum(-1)                    # (C, d)
        R = self.RA + self.Rb / np.where(ok, n, 1.0)
        S = Ph[:, ix].sum(-1) + R
        pred = self.x[:, ix].sum(-1)
        innov = np.where(ok, y - pred, 0.0)
        K = np.where(ok[:, None], Ph / S[:, None], 0.0)
        self.x += K * innov[:, None]
        self.P -= K[:, :, None] * Ph[:, None, :]
        return ok, innov, S

    def read(self, ix):
        return self.x[:, ix].sum(-1)


def run(df, obs_list, params, q_frac, record=True, ll_window=None, verbose=True):
    """obs_list: per channel (y_home, n_home, y_away, n_away) arrays on df's rows. Returns the per-match
    features (C, n_matches, 8): pred home offence, pred away offence, off_h, def_h, off_a, def_a, mu, home;
    and the per-channel predictive log likelihood over ll_window."""
    gamers = sorted(set(df.p1) | set(df.p2))
    teams = sorted(set(df.t1.dropna()) | set(df.t2.dropna()))
    F = Filter(gamers, teams, params, q_frac)
    C = F.C
    Yh = np.stack([o[0] for o in obs_list], 1)
    Nh = np.stack([o[1] for o in obs_list], 1)
    Ya = np.stack([o[2] for o in obs_list], 1)
    Na = np.stack([o[3] for o in obs_list], 1)
    starts = df.start.to_numpy()
    settle = starts + np.timedelta64(int(LAG.total_seconds()), "s")
    order_s = np.argsort(settle, kind="stable")
    n = len(df)
    feats = np.full((C, n, 8), np.nan) if record else None
    ll = np.zeros(C)
    lln = np.zeros(C)
    if ll_window:
        lw = H.in_window(df, ll_window).to_numpy()
    p1, p2 = df.p1.to_numpy(), df.p2.to_numpy()
    t1, t2 = df.t1.fillna("").to_numpy(), df.t2.fillna("").to_numpy()
    has_y = (np.isfinite(Yh).any(1) | np.isfinite(Ya).any(1))
    j = 0
    t0 = time.time()
    G = F.G
    for i in range(n):
        t = starts[i]
        while j < n and settle[order_s[j]] <= t:
            m = order_s[j]
            j += 1
            if not has_y[m]:
                continue
            F.advance(pd.Timestamp(settle[m]))
            F.activate(p1[m], p2[m])
            for (og, dg, ot, dt_, home, y, nn) in ((p1[m], p2[m], t1[m], t2[m], True, Yh[m], Nh[m]),
                                                   (p2[m], p1[m], t2[m], t1[m], False, Ya[m], Na[m])):
                ix = F.idx(og, dg, ot, dt_, home)
                r = F.update(ix, y, nn)
                if r is not None and ll_window and lw[m]:
                    ok, innov, S = r
                    ll += np.where(ok, -0.5 * (np.log(S) + innov ** 2 / S), 0.0)
                    lln += ok
        if record:
            F.advance(pd.Timestamp(t))
            a, b = F.gi[p1[i]], F.gi[p2[i]]
            ixh = F.idx(p1[i], p2[i], t1[i], t2[i], True)
            ixa = F.idx(p2[i], p1[i], t2[i], t1[i], False)
            x = F.x
            feats[:, i, 0] = F.read(ixh)
            feats[:, i, 1] = F.read(ixa)
            feats[:, i, 2] = x[:, 2 + a]
            feats[:, i, 3] = x[:, 2 + G + a]
            feats[:, i, 4] = x[:, 2 + b]
            feats[:, i, 5] = x[:, 2 + G + b]
            feats[:, i, 6] = x[:, 0]
            feats[:, i, 7] = x[:, 1]
        if verbose and i % 5000 == 0:
            print(f"  {i:,}/{n:,} {time.time() - t0:.0f}s", flush=True)
    return feats, ll / np.maximum(lln, 1), lln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()
    df = H.load(tables=False, elo=False)
    obs, has = side_obs(df)
    keys = list(STATS)
    vc = variance_components(df, obs, has, keys)
    # points over the whole history (strength): every settled match
    pts_all = (df.s1.to_numpy(float), np.where(df.s1.notna(), 1.0, np.nan),
               df.s2.to_numpy(float), np.where(df.s2.notna(), 1.0, np.nan))
    allw = H.in_window(df, ("2026-01-01", "2026-08-01")).to_numpy() & df.s1.notna().to_numpy()
    s = np.r_[df.s1[allw], df.s2[allw]]
    vc_all = dict(vc["pts"])
    vc_all["league"] = float(s.mean())
    with open(f"{OUT}/kalman_vc.json", "w") as fh:
        json.dump({"stats": vc, "pts_all": vc_all}, fh, indent=1, default=float)
    for k in keys:
        v = vc[k]
        print(f"{k:9s} lg {v['league']:8.3f} A {v['A']:.4g} b {v['b']:.4g} R_A {v['R_A']:.4g} "
              f"sd_off {np.sqrt(v['V_off']):.4g} sd_def {np.sqrt(v['V_def']):.4g} "
              f"sd_toff {np.sqrt(v['V_team_off']):.4g} sd_tdef {np.sqrt(v['V_team_def']):.4g}")
    if a.tune:
        # stats: drive-stat matches only (start the run 04-20 to skip the 4 months without them)
        sub = df.day >= "2026-04-20"
        d2 = df[sub].reset_index(drop=True)
        o2 = {k: tuple(x[sub.to_numpy()] for x in obs[k]) for k in keys}
        params, ol, qf, lab = [], [], [], []
        for k in keys:
            for q in Q_GRID:
                params.append(vc[k])
                ol.append(o2[k])
                qf.append(q)
                lab.append((k, q))
        _, ll, lln = run(d2, ol, params, qf, record=False, ll_window=TUNE_LL)
        res = pd.DataFrame(lab, columns=["stat", "q"]).assign(ll=ll, n=lln)
        # points over the whole history too
        params2 = [vc_all] * len(Q_GRID)
        _, ll2, lln2 = run(df, [pts_all] * len(Q_GRID), params2, list(Q_GRID), record=False, ll_window=TUNE_LL)
        res = pd.concat([res, pd.DataFrame({"stat": "pts_all", "q": Q_GRID, "ll": ll2, "n": lln2})])
        res.to_csv(f"{OUT}/kalman_tune.csv", index=False)
        piv = res.pivot(index="stat", columns="q", values="ll")
        piv = piv.sub(piv.max(1), axis=0)
        print(piv.round(5).to_string())
        best = res.loc[res.groupby("stat").ll.idxmax()].set_index("stat").q.to_dict()
        json.dump(best, open(f"{OUT}/kalman_q.json", "w"), indent=1)
        print(best)
        return
    best = json.load(open(f"{OUT}/kalman_q.json"))
    sub = df.day >= "2026-04-20"
    d2 = df[sub].reset_index(drop=True)
    o2 = {k: tuple(x[sub.to_numpy()] for x in obs[k]) for k in keys}
    feats, _, _ = run(d2, [o2[k] for k in keys], [vc[k] for k in keys], [best[k] for k in keys])
    featsP, _, _ = run(df, [pts_all], [vc_all], [best["pts_all"]])
    names = ["pred_h", "pred_a", "off_h", "def_h", "off_a", "def_a", "mu", "home"]
    out = pd.DataFrame({"code": df.code})
    tmp = pd.DataFrame({"code": d2.code})
    for c, k in enumerate(keys):
        for j, nm in enumerate(names):
            tmp[f"k_{k}_{nm}"] = feats[c, :, j]
    out = out.merge(tmp, on="code", how="left")
    for j, nm in enumerate(names):
        out[f"k_ptsall_{nm}"] = featsP[0, :, j]
    out.to_pickle(f"{OUT}/kalman_feats.pkl")
    print("wrote", f"{OUT}/kalman_feats.pkl", out.shape)


if __name__ == "__main__":
    main()
