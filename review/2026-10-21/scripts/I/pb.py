import sys; sys.path.insert(0, ".")
import pandas as pd, numpy as np
from eAMFCalibrator import prior_bets as pbm
p = pd.read_csv("eAMFCalibrator/out_prior_bets_full/prior_bets.csv")
h = pd.read_csv("eAMFCalibrator/out/match_history.csv"); h["t"] = pd.to_datetime(h.SCHEDULED_START_TIME_UTC)
rows = [h[["MATCH_CODE","t",s]].rename(columns={s:"g"}) for s in ("PLAYER_1_HANDLE","PLAYER_2_HANDLE")]
g = pd.concat(rows).dropna(); g["g"]=g.g.str.strip().str.upper(); g=g.sort_values(["g","t"]); g["prev_t"]=g.groupby("g").t.shift(1)
prev = g.groupby("MATCH_CODE").prev_t.max() + pd.Timedelta(minutes=36)
b = pd.read_csv("eAMFCalibrator/out_bets_v21_4k/bets_sim.csv", low_memory=False, usecols=["match_code","bet_time","market","in_play","selection"])
b = b[(b.market=="moneyline") & (b.in_play==False)]
b["bt"] = pd.to_datetime(b.bet_time, format="mixed")
last = b.groupby("match_code").bt.max()
p["last_bet"] = p.match_code.map(last); p["prev_in"] = p.match_code.map(prev)
p["clean"] = p.prev_in.isna() | (p.last_bet >= p.prev_in)
print("matches", len(p), "clean", p.clean.sum(), "unclean", (~p.clean).sum(), "(no last bet:", p.last_bet.isna().sum(), ")")
y = p.home_won.values
def brier(q): return (q - y)**2
for col in [c for c in p.columns if c.startswith("p_home_") and c != "p_home_prod"]:
    d = brier(p[col].values) - brier(p.p_home_prod.values)
    print(f"{col:18s} Brier less prod: clean {d[p.clean].mean():+.4f}  unclean {d[~p.clean].mean():+.4f}  all {d.mean():+.4f}")
# disagreement with and without intercept
import math
def fit(X, y):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        q = 1/(1+np.exp(-X@b)); W = q*(1-q)
        H = X.T@(X*W[:,None]); g = X.T@(y-q)
        b = b + np.linalg.solve(H, g)
    cov = np.linalg.inv(H)
    return b, np.sqrt(np.diag(cov))
clip = lambda q: np.clip(q, pbm.P_CLIP, 1-pbm.P_CLIP)
lp = np.log(clip(p.p_home_prod)/(1-clip(p.p_home_prod)))
for col in [c for c in p.columns if c.startswith("p_home_") and c != "p_home_prod"]:
    lq = np.log(clip(p[col])/(1-clip(p[col])))
    X0 = np.c_[lp, lq-lp]; X1 = np.c_[np.ones(len(p)), lp, lq-lp]
    b0, s0 = fit(X0, y); b1, s1 = fit(X1, y)
    codes = list(p.match_code); res = {c:(float(v),None,None) for c,v in zip(codes, y)}
    own = pbm.disagreement(codes, res, dict(zip(codes, p.p_home_prod)), dict(zip(codes, p[col])))
    print(f"{col:18s} module: prod {own[0]:+.2f} ({own[1]:.2f}) dis {own[2]:+.2f} ({own[3]:.2f}) | no-int prod {b0[0]:+.2f} dis {b0[1]:+.2f} | with intercept {b1[0]:+.2f} ({s1[0]:.2f}) prod {b1[1]:+.2f} ({s1[1]:.2f}) dis {b1[2]:+.2f} ({s1[2]:.2f})")
print("home win rate", y.mean(), "prod mean p_home", p.p_home_prod.mean())
from collections import defaultdict
print("--- intervals over days, as grade_lines")
for col in [c for c in p.columns if c.startswith("p_home_") and c != "p_home_prod"]:
    for name, mask in (("all", np.ones(len(p), bool)), ("clean", p.clean.values), ("unclean", ~p.clean.values)):
        dd = defaultdict(list); dl = defaultdict(list)
        for code, yy, q, pr in zip(p.match_code[mask], y[mask], p[col][mask], p.p_home_prod[mask]):
            sb, sl = pbm._scores(q, yy); pb_, pl = pbm._scores(pr, yy)
            dd[code[-6:]].append(sb - pb_); dl[code[-6:]].append(sl - pl)
        a = pbm._interval(dd); l = pbm._interval(dl, seed=1)
        print(f"{col:18s} {name:8s} n={mask.sum():4d} Brier {a[0]:+.4f} [{a[1]:+.4f}, {a[2]:+.4f}]  logloss {l[0]:+.4f} [{l[1]:+.4f}, {l[2]:+.4f}]")
