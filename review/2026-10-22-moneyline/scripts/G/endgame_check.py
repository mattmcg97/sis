"""v2.1's late game against a direct win probability: eAMFModel v21-endgame's late states (every real Q4 snap
with the sides 8 or fewer apart, Sep 1 - Oct 17), where v2.1 played each state out 200 times (sim_win: the side
with the ball wins) and the real game says what happened (real_win).

    python scripts/G/endgame_check.py      -> out/G/endgame_check.txt, endgame_preds.csv

The states carry the clock, the side with the ball's lead and the result, not which side had the ball; with
the final score that is exact: the side with the ball is home when (it won) == (home won). Matches that ended
level are left out (no side to read). The pre-match P(the side with the ball wins) is then A_best's (or
C_strength's / prod's) from that side.

Models, refitted daily on the states of the matches settled by the day's midnight (harness's rule):
  logit   logit p = a + b * lead / sqrt(clock/960 + k) + c * pre * sqrt(clock/960) + d * pre + e * lead
  xgb     binary:logistic on lead, lead / sqrt(...), clock, pre, pre * sqrt(...); monotone + in the lead and the
          pre-match terms
TUNE on the test days 09-08..09-30 (settings: k, the XGBoost depth / rounds), REPORT 10-01..10-17 (frozen), the
same moments for v2.1 and the models, day- and match-resampled intervals. v2.1's sim_win is a 200-game mean
(sd about 0.035 at p = 0.6): its sampling noise costs it roughly 0.002-0.003 of log loss; shown as is and,
for a fairer read, shrunk to its own walk-forward calibration (Platt on logit sim_win).
"""

import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ML)
import harness as H  # noqa: E402

ROOT = "/home/user/sis/eAMFCalibrator"
OUT = os.path.join(ML, "out", "G")
FILES = ("out_v21_endgame_sep/v21_endgame_states.csv", "out_v21_endgame/v21_endgame_states.csv")
TUNE = ("2026-09-08", "2026-10-01")
REPORT = ("2026-10-01", "2026-10-19")


def load():
    e = pd.concat([pd.read_csv(os.path.join(ROOT, f)).assign(src=f.split("/")[0]) for f in FILES], ignore_index=True)
    f = pd.read_csv(H.FEATURES, usecols=["code", "start", "s1", "s2", "prod_p_home"])
    for n in ("A_best", "C_strength"):
        f = f.merge(pd.read_csv(os.path.join(ML, "preds", f"{n}.csv")).rename(columns={"p_home": f"pre_{n}"}),
                    on="code", how="left")
    e = e.merge(f, on="code", how="inner")
    e = e[e.s1 != e.s2].copy()
    home_won = e.s1 > e.s2
    e["ball_home"] = (e.real_win == 1) == home_won
    for n in ("pre_A_best", "pre_C_strength", "prod_p_home"):
        e[n + "_side"] = np.where(e.ball_home, e[n], 1 - e[n])
    e["start"] = pd.to_datetime(e.start)
    e["day"] = e.start.dt.strftime("%Y-%m-%d")
    e["y"] = e.real_win
    e["v21"] = e.sim_win.clip(0.0025, 0.9975)          # a 200-game mean: 0 and 1 are a half-game away
    return e.reset_index(drop=True)


def feats(d, pre="pre_A_best_side", k=0.02):
    s = d.clock.to_numpy(float) / 960.0
    p = H.logit(d[pre].to_numpy(float))
    lead = d.lead.to_numpy(float)
    return pd.DataFrame({"lead": lead, "lead_z": lead / np.sqrt(s + k), "clock": d.clock.to_numpy(float),
                         "pre": p, "pre_s": p * np.sqrt(s)})


def logit_fit(X, y):
    Z = np.column_stack([np.ones(len(X)), X.lead_z, X.pre_s, X.pre, X.lead])

    def nll(b):
        z = Z @ b
        return np.sum(np.logaddexp(0, z) - y * z)

    def grad(b):
        return Z.T @ (H.sigmoid(Z @ b) - y)

    return minimize(nll, np.zeros(Z.shape[1]), jac=grad, method="BFGS").x


def logit_pred(b, X):
    return H.sigmoid(np.column_stack([np.ones(len(X)), X.lead_z, X.pre_s, X.pre, X.lead]) @ b)


def platt_fit(x, y):
    Z = np.column_stack([np.ones(len(x)), x])

    def nll(b):
        z = Z @ b
        return np.sum(np.logaddexp(0, z) - y * z)

    return minimize(nll, np.array([0.0, 1.0]), method="BFGS").x


def walk(e, days, cfgs):
    out = {c["name"]: pd.Series(np.nan, index=e.index) for c in cfgs}
    settled = e.start + H.RESULT_LAG
    for d in sorted(e.loc[(e.day >= days[0]) & (e.day < days[1]), "day"].unique()):
        tr0 = e[settled <= pd.Timestamp(d)]
        for c in cfgs:
            pre = c.get("pre", "pre_A_best_side")
            tr = tr0[tr0[pre].notna()]
            te = e.index[(e.day == d) & e[pre].notna()]
            Xtr, Xte = feats(tr, pre, c["k"]), feats(e.loc[te], pre, c["k"])
            if c["kind"] == "logit":
                out[c["name"]].loc[te] = logit_pred(logit_fit(Xtr, tr.y.to_numpy()), Xte)
            elif c["kind"] == "platt":
                b = platt_fit(H.logit(tr.v21.to_numpy()), tr.y.to_numpy())
                out[c["name"]].loc[te] = H.sigmoid(b[0] + b[1] * H.logit(e.loc[te, "v21"].to_numpy()))
            else:
                mono = "(1,1,0,1,1)"
                bst = xgb.train(dict(objective="binary:logistic", max_depth=c["depth"], eta=0.05, min_child_weight=50,
                                     subsample=0.8, reg_lambda=5.0, monotone_constraints=mono, nthread=2, seed=0),
                                xgb.DMatrix(Xtr, label=tr.y.to_numpy()), num_boost_round=c["rounds"])
                out[c["name"]].loc[te] = bst.predict(xgb.DMatrix(Xte))
    return out


def main():
    e = load()
    lines = [f"late states: {len(e):,} on {e.code.nunique():,} matches ({e.day.min()} to {e.day.max()}), ties left out; "
             f"the side with the ball is home on {e.ball_home.mean():.1%}"]
    grid = [dict(kind="logit", k=k, name=f"logit k{k}") for k in (0.005, 0.01, 0.02, 0.05)]
    grid += [dict(kind="xgb", k=0.02, depth=dd, rounds=r, name=f"xgb d{dd} r{r}") for dd in (2, 3) for r in (100, 200, 400)]
    grid += [dict(kind="platt", k=0.02, name="v21 platt")]
    pt = walk(e, TUNE, grid)
    w = e[(e.day >= TUNE[0]) & (e.day < TUNE[1])]
    lines.append(f"\nTUNE {TUNE[0]}..{TUNE[1]}: {len(w):,} states, {w.code.nunique():,} matches")
    res = []
    for c in grid:
        m = H.metrics(w.y, pt[c["name"]].loc[w.index])
        res.append((c["name"], m["logloss"], m["cal_slope"]))
        lines.append(f"  {c['name']:14s} {m['logloss']:.4f}  slope {m['cal_slope']:.2f}")
    m = H.metrics(w.y, w.v21)
    lines.append(f"  {'v21 sim_win':14s} {m['logloss']:.4f}  slope {m['cal_slope']:.2f}")
    best_logit = min((r for r in res if r[0].startswith("logit")), key=lambda r: r[1])[0]
    best_xgb = min((r for r in res if r[0].startswith("xgb")), key=lambda r: r[1])[0]
    lines.append(f"  frozen: {best_logit}, {best_xgb}")
    frozen = [c for c in grid if c["name"] in (best_logit, best_xgb, "v21 platt")]
    frozen += [dict(c, pre=p, name=f"{c['name']} pre={p[:-5]}") for c in grid if c["name"] == best_logit
               for p in ("pre_C_strength_side", "prod_p_home_side")]
    pr = walk(e, REPORT, frozen)
    r = e[(e.day >= REPORT[0]) & (e.day < REPORT[1])].copy()
    names = {best_logit: "logit", best_xgb: "xgb", "v21 platt": "v21_platt"}
    for c in frozen:
        r[names.get(c["name"], c["name"])] = pr[c["name"]].loc[r.index]
    cols = ["logit", "xgb", "v21_platt"] + [c["name"] for c in frozen if "pre=" in c["name"]]
    r = r.dropna(subset=cols)
    lines.append(f"\nREPORT {REPORT[0]}..{REPORT[1]} (frozen): {len(r):,} states, {r.code.nunique():,} matches, "
                 f"{r.day.nunique()} days")
    lines.append(f"  {'model':30s} {'logloss':>8s} {'brier':>7s} {'slope':>6s}   less v21 sim_win (days 95%; matches 95%)")
    from wp_bets import boot_match
    for c in ["v21"] + cols:
        m = H.metrics(r.y, r[c])
        s = f"  {c:30s} {m['logloss']:8.4f} {m['brier']:7.4f} {m['cal_slope']:6.2f}"
        if c != "v21":
            mu, lo, hi = H.boot_diff(r.assign(code_=r.code), c, "v21")
            _, mlo, mhi = boot_match(r.rename(columns={"code": "match_code"}), c, "v21")
            s += f"   {mu:+.4f} ({lo:+.4f},{hi:+.4f}; {mlo:+.4f},{mhi:+.4f})"
        lines.append(s)
    lines.append("\n  by time left and the side with the ball's lead (REPORT; real / v21 / logit, % the side with the ball wins)")
    r["tb"] = pd.cut(r.clock, [-1, 60, 120, 240], labels=["Q4 1:00-0:00", "Q4 2:00-1:01", "Q4 4:00-2:01"])
    r["lb"] = pd.cut(r.lead, [-9, -4, -3, -1, 0, 3, 8], labels=["4-8 behind", "3 behind", "1-2 behind", "level", "1-3 ahead", "4-8 ahead"])
    g = r.groupby(["tb", "lb"], observed=True).agg(n=("y", "size"), real=("y", "mean"), v21=("v21", "mean"), logit=("logit", "mean"),
                                                   ll_v21=("v21", lambda x: 0), )
    for (tb, lb), row in g.iterrows():
        x = r[(r.tb == tb) & (r.lb == lb)]
        llv = H.logloss_each(x.y.to_numpy(), x.v21.to_numpy()).mean()
        lll = H.logloss_each(x.y.to_numpy(), x.logit.to_numpy()).mean()
        lines.append(f"  {tb:14s} {lb:11s} {int(row.n):5d}   {100 * row.real:5.1f} / {100 * row.v21:5.1f} / {100 * row.logit:5.1f}"
                     f"   log loss v21 {llv:.3f}, logit {lll:.3f}")
    r[["code", "day", "clock", "lead", "ball_home", "y", "v21", "logit", "xgb", "v21_platt"]].to_csv(
        os.path.join(OUT, "endgame_preds.csv"), index=False)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "endgame_check.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
