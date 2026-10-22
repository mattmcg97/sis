"""The total's distribution about a walk-forward expected total, walk-forward, scored.

  python dist_eval.py MEANCOL [--W 60] [--alpha 5] [--refit 7] [--start 2026-06-01] [--tag name]

Every REFIT days at a day's midnight, on the matches settled by then within the last W days (their walk-forward
expected totals and drives, so the residuals are out of sample):
  var function   log V = a0 + a1 log(mu/35) + a2 log(muD/10.3), Gaussian ML on the residuals; forms const, mu, D, mu+D
  normal / NB    on whole points, sd from the var function
  lumpy          normal / NB x per-integer weights (IPF on the window, alpha pseudo-counts), mean put back
  compound       drives x points a drive (dist.compound_pmf): sigD = sd of D about its forecast, beta = elasticity of
                 the total to D/muD (Poisson GLM, offset log mu), q6/q8 the try's misses and 2-pointers (one-TD
                 sides), tau (match-level scoring factor) by ML on the window; td/fg mix from the td/fg-a-drive ridges
  compound+lumpy the compound x per-integer weights
Scores per match: log score of the integer total, RPS, P(over prod's line) (all prod lines are x.5: no pushes).
Writes out/F/dist_<tag>.pkl (every distribution's p_over, logs, rps, mean, sd per match) and params.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import optimize, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402
import dist as DS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mean")
ap.add_argument("--means", default="means_all.pkl")
ap.add_argument("--W", type=int, default=60)
ap.add_argument("--alpha", type=float, default=5.0)
ap.add_argument("--refit", type=int, default=7)
ap.add_argument("--start", default="2026-06-01")
ap.add_argument("--end", default="2026-10-19")
ap.add_argument("--tag", default="")
ap.add_argument("--drives", default="drives")
ap.add_argument("--no-compound", action="store_true")
ap.add_argument("--tau-sub", type=int, default=1500)
ap.add_argument("--fit-mean", default="", help="fit the variance function / lumps / compound on this mean's residuals, "
                                              "price with MEAN (for the tables, which have no history before 09-01)")
ap.add_argument("--save-pmf", default="lumpy_nb_mu,lumpy_nb_muD")
a = ap.parse_args()
tag = a.tag or f"{a.mean}_W{a.W}"

df = F.panel()
mm = pd.read_pickle(os.path.join(F.OUT, a.means))
cols = list(dict.fromkeys([a.mean, a.drives, "td_h", "td_a", "fg_h", "fg_a"] + ([a.fit_mean] if a.fit_mean else [])))
d = df[["code", "day", "start", "settled", "total", "D", "s1", "s2", "h_fgx", "a_fgx", "prod_total_line",
        "prod_total_prob", "over"]].merge(mm[["code"] + cols], on="code", how="left")
d["mu_price"] = d[a.mean]
d["mu_fit"] = d[a.fit_mean] if a.fit_mean else d[a.mean]
d = d.drop(columns=[c for c in set([a.mean, a.fit_mean]) if c]).rename(columns={a.drives: "muD"})
d["mu"] = d.mu_fit
d["pt"] = (d.td_h + d.td_a) / 2
d["pf"] = (d.fg_h + d.fg_a) / 2
REF_MU, REF_D = 35.0, 10.3
FORMS = {"const": [], "mu": ["lmu"], "D": ["lD"], "muD": ["lmu", "lD"]}
d["lmu"] = np.log(d.mu / REF_MU)
d["lD"] = np.log(d.muD / REF_D)


def fit_var(tr, form):
    Z = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in FORMS[form]])
    r2 = (tr.total - tr.mu).values ** 2

    def nll(b):
        lv = Z @ b
        return 0.5 * np.sum(lv + r2 * np.exp(-lv))

    def grad(b):
        lv = Z @ b
        return 0.5 * Z.T @ (1 - r2 * np.exp(-lv))

    b0 = np.zeros(Z.shape[1])
    b0[0] = np.log(r2.mean())
    return optimize.minimize(nll, b0, jac=grad, method="BFGS").x


def var_of(rows, form, b):
    Z = np.column_stack([np.ones(len(rows))] + [rows[c].values for c in FORMS[form]])
    return np.exp(Z @ b)


def fit_beta(tr):
    """total ~ Poisson(mu * exp(c) * (D/muD)^(1+beta)): (c, beta)."""
    x = np.log(tr.D.values / tr.muD.values)
    off = np.log(tr.mu.values)
    y = tr.total.values

    def nll(p):
        eta = off + p[0] + p[1] * x
        return np.sum(np.exp(eta) - y * eta)

    r = optimize.minimize(nll, np.array([0.0, 0.6]), method="BFGS")
    return r.x[0], r.x[1] - 1.0


def fit_q(tr):
    out = []
    for s, f in (("h", "s1"), ("a", "s2")):
        rest = tr[f] - 3 * tr[f"{s}_fgx"]
        T = np.round(rest / 7)
        j = rest - 7 * T
        one = (T == 1) & j.between(-1, 1)
        out.append(j[one])
    j = pd.concat(out)
    return float((j == -1).mean()), float((j == 1).mean())


cmp_params = {}


def compound_for(rows, P, tau):
    return DS.compound_pmf(rows.mu.values, rows.muD.values, P["sigD"], P["beta"], tau, rows.pt.values,
                           rows.pf.values, P["q6"], P["q8"], nz=5)[0]


def fit_tau(tr, P, rng):
    sub = tr if len(tr) <= a.tau_sub else tr.iloc[np.sort(rng.choice(len(tr), a.tau_sub, replace=False))]
    tot = sub.total.values.astype(int)

    def nll(tau):
        pmf = compound_for(sub, P, tau)
        return -np.log(pmf[np.arange(len(sub)), np.clip(tot, 0, DS.K - 1)]).mean()

    r = optimize.minimize_scalar(nll, bounds=(0.0, 0.6), method="bounded", options=dict(xatol=0.005))
    return float(r.x), float(r.fun)


res = {k: np.full((len(d),), np.nan) for k in []}
store = {}


def put(name, idx, p_over, logs, rps, mean, sd):
    if name not in store:
        store[name] = {k: np.full(len(d), np.nan) for k in ("p_over", "logs", "rps", "mean", "sd")}
    for k, v in (("p_over", p_over), ("logs", logs), ("rps", rps), ("mean", mean), ("sd", sd)):
        store[name][k][idx] = v


rng = np.random.default_rng(0)
store_pmf = {}
days = F.days_between(d, a.start, a.end)
params = []
P = None
for k, day in enumerate(days):
    cut = pd.Timestamp(day)
    if P is None or k % a.refit == 0:
        t0 = time.time()
        tr = d[(d.settled <= cut) & (d.start >= cut - pd.Timedelta(days=a.W)) & d.total.notna() & d.mu.notna()
               & d.muD.notna()]
        P = {"n": len(tr)}
        for form in FORMS:
            P["var_" + form] = fit_var(tr, form)
        # lumpiness weights on the window, normal and NB bases (mu+D variance)
        sd_tr = np.sqrt(var_of(tr, "muD", P["var_muD"]))
        tot_tr = tr.total.values.astype(int)
        P["w_norm"] = DS.fit_lumps(DS.normal_pmf(tr.mu.values, sd_tr), tot_tr, alpha=a.alpha)
        P["w_nb"] = DS.fit_lumps(DS.nb_pmf(tr.mu.values, sd_tr ** 2), tot_tr, alpha=a.alpha)
        if not a.no_compound:
            trd = tr[tr.D.notna() & tr.pt.notna()]
            P["sigD"] = float(np.sqrt(np.mean((trd.D - trd.muD) ** 2)))
            P["c"], P["beta"] = fit_beta(trd)
            P["q6"], P["q8"] = fit_q(trd)
            P["tau"], P["tau_nll"] = fit_tau(trd, P, rng)
            sub = trd if len(trd) <= 4000 else trd.iloc[np.sort(rng.choice(len(trd), 4000, replace=False))]
            pm_tr = compound_for(sub, P, P["tau"])
            P["w_cmp"] = DS.fit_lumps(pm_tr, sub.total.values.astype(int), alpha=a.alpha * len(sub) / len(trd))
        params.append(dict(day=day, **{kk: (vv.tolist() if isinstance(vv, np.ndarray) and kk.startswith("var")
                                            else vv) for kk, vv in P.items() if not kk.startswith("w_")}))
        print(f"  {day}: {len(tr):,} rows; var muD {np.round(P['var_muD'], 3)}"
              + (f"; sigD {P['sigD']:.2f} beta {P['beta']:+.2f} q6 {P['q6']:.3f} q8 {P['q8']:.3f} tau {P['tau']:.3f}"
                 if not a.no_compound else "") + f" ({time.time() - t0:.0f}s)", flush=True)
    rows = d[(d.day == day) & d.mu_price.notna() & d.muD.notna()].copy()
    rows["mu"] = rows.mu_price
    rows["lmu"] = np.log(rows.mu / REF_MU)
    if not len(rows):
        continue
    idx = rows.index.values
    tot = rows.total.values
    line = rows.prod_total_line.values
    for form in FORMS:
        sd = np.sqrt(var_of(rows, form, P["var_" + form]))
        pmf = DS.normal_pmf(rows.mu.values, sd)
        lg, rp, po = DS.scores(pmf, tot, line)
        put(f"normal_{form}", idx, po, lg, rp, rows.mu.values, sd)
        pmf = DS.nb_pmf(rows.mu.values, sd ** 2)
        lg, rp, po = DS.scores(pmf, tot, line)
        put(f"nb_{form}", idx, po, lg, rp, DS.pmf_mean(pmf), DS.pmf_sd(pmf))
        for base, wk, fn in (("normal", "w_norm", DS.normal_pmf), ("nb", "w_nb", lambda m_, s_: DS.nb_pmf(m_, s_ ** 2))):
            if base == "normal" and form not in ("const", "muD"):
                continue
            pmf = DS.lumpy(fn, rows.mu.values, sd, P[wk])
            lg, rp, po = DS.scores(pmf, tot, line)
            put(f"lumpy_{base}_{form}", idx, po, lg, rp, DS.pmf_mean(pmf), DS.pmf_sd(pmf))
            if f"lumpy_{base}_{form}" in a.save_pmf.split(","):
                store_pmf.setdefault(f"lumpy_{base}_{form}", []).append((idx, pmf))
    if not a.no_compound:
        ok = rows.pt.notna().values
        r2 = rows[ok]
        if len(r2):
            pmf = compound_for(r2, P, P["tau"])
            lg, rp, po = DS.scores(pmf, r2.total.values, r2.prod_total_line.values)
            put("compound", r2.index.values, po, lg, rp, DS.pmf_mean(pmf), DS.pmf_sd(pmf))
            pmf0 = DS.compound_pmf(r2.mu.values, r2.muD.values, 0.0, P["beta"], P["tau"], r2.pt.values,
                                   r2.pf.values, P["q6"], P["q8"], nz=5)[0]
            lg, rp, po = DS.scores(pmf0, r2.total.values, r2.prod_total_line.values)
            put("compound_fixedD", r2.index.values, po, lg, rp, DS.pmf_mean(pmf0), DS.pmf_sd(pmf0))
            pmfl = pmf * P["w_cmp"][None, :]
            pmfl /= pmfl.sum(1, keepdims=True)
            # put the mean back (one shift of mu through the compound)
            shift = r2.mu.values - DS.pmf_mean(pmfl)
            pm2 = DS.compound_pmf(r2.mu.values + shift, r2.muD.values, P["sigD"], P["beta"], P["tau"], r2.pt.values,
                                  r2.pf.values, P["q6"], P["q8"], nz=5)[0] * P["w_cmp"][None, :]
            pm2 /= pm2.sum(1, keepdims=True)
            lg, rp, po = DS.scores(pm2, r2.total.values, r2.prod_total_line.values)
            put("compound_lumpy", r2.index.values, po, lg, rp, DS.pmf_mean(pm2), DS.pmf_sd(pm2))

res = pd.DataFrame({"code": d.code})
for name, v in store.items():
    for kk, arr in v.items():
        res[f"{name}__{kk}"] = arr
res.to_pickle(os.path.join(F.OUT, f"dist_{tag}.pkl"))
for nm, lst in store_pmf.items():
    ii = np.concatenate([i for i, _ in lst])
    pm = np.vstack([p_ for _, p_ in lst]).astype(np.float32)
    pd.to_pickle({"code": d.code.values[ii], "pmf": pm}, os.path.join(F.OUT, f"pmf_{tag}_{nm}.pkl"))
pd.DataFrame(params).to_csv(os.path.join(F.OUT, f"dist_{tag}_params.csv"), index=False)
pd.DataFrame({k_: P[k_] for k_ in ("w_norm", "w_nb", "w_cmp") if k_ in P}).rename_axis("total").to_csv(
    os.path.join(F.OUT, f"dist_{tag}_lumps_last.csv"))
print("saved", tag)
