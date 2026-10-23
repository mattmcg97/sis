"""TUNE-only searches for the margin regressor + link (stage M) and the offset classifier (stage O).
M: walk-forward from 2026-04-01 so the link (fitted on the previous 60 days' out-of-sample margins) has
   history by 06-01; scored on TUNE after the link. O: base_margin = logit of a walk-forward base
   (link_rm: logistic link on rating_margin; ridge_p: time-decayed ridge strength through its OOS link).
    ONLY=<substr> python s04_tune_other.py M|O <design> <diffs 0|1> <window|all> '<json params of the classifier>'
"""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

stage, design, diffs, wdw, cls_params = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4], json.loads(sys.argv[5])
wdw = None if wdw == "all" else int(wdw)
only = os.environ.get("ONLY")
REFIT = int(os.environ.get("REFIT", "3"))
SUF = f"_r{REFIT}" if REFIT != 3 else ""
log = Log(os.path.join(OUT, f"s04_tune_{stage}{only or ''}.log"))
df = load()
feats = feature_set(df, "all")
RES = os.path.join(OUT, "tune_results.csv")
PRED = os.path.join(OUT, "tune")
os.makedirs(PRED, exist_ok=True)
done = set(pd.read_csv(RES).name) if os.path.exists(RES) else set()
if design == "plain":
    Xh, Xa = plain_matrix(df, feats, diffs=diffs), None
else:
    Xh, Xa = paired_matrix(df, feats, diffs=diffs)


def record(name, p, params, info, extra):
    df["_p"] = p
    l, n = ll(df, "_p", H.TUNE)
    w = df[H.in_window(df, H.TUNE) & df.y.notna() & df._p.notna()]
    a, b = H.calib(w.y, w._p)
    row = dict(name=name, stage=stage, design=design, diffs=diffs, ident=False, window=wdw or 0, refit=REFIT,
               params=json.dumps(dict(params, **extra)), tune_ll=l, n=n, slope=b,
               rounds_med=float(np.median(info["rounds"])), secs=info["secs"])
    pd.DataFrame([row]).to_csv(RES, mode="a", header=not os.path.exists(RES), index=False)
    pd.DataFrame({"code": df.code, "p": p}).dropna().to_csv(os.path.join(PRED, f"{name}.csv"), index=False)
    log(f"  {name}: TUNE ll {l:.5f} (n {n}) slope {b:.2f} rounds med {row['rounds_med']:.0f} ({info['secs']:.0f}s)")


if stage == "M":
    grid = []
    for obj in ("reg:squarederror", "reg:pseudohubererror"):
        for mcw in (50, 200, 800):
            grid.append((obj, mcw))
    for obj, mcw in grid:
        name = f"M_{'ph' if 'pseudohuber' in obj else 'se'}_mcw{mcw}_{design}{'_diffs' if diffs else ''}{SUF}"
        if (only and only not in name) or name in done:
            continue
        params = dict(cls_params, min_child_weight=float(mcw))
        if obj == "reg:pseudohubererror":
            params["huber_slope"] = 10.0
        m, info = wf_xgb(df, Xh, params, "2026-04-01", H.TUNE[1], design=design, Xa=Xa, objective=obj,
                         target="margin", refit_days=REFIT, train_days=wdw)
        prob = wf_on_oos(df, m, "2026-05-15", H.TUNE[1], kind="link", hist_days=60)
        record(name, prob, params, info, {"objective": obj})
elif stage == "O":
    regs = {"r1": dict(max_depth=2, eta=0.02, min_child_weight=200.0, subsample=0.8, colsample_bytree=0.5,
                       reg_lambda=50.0, reg_alpha=0.0, gamma=0.0),
            "r2": dict(max_depth=3, eta=0.02, min_child_weight=100.0, subsample=0.8, colsample_bytree=0.5,
                       reg_lambda=20.0, reg_alpha=0.0, gamma=0.0),
            "tuned": dict(cls_params)}
    for base in ("link_rm", "ridge_p"):
        bp = load_series(df, base)
        bh = pd.Series(H.logit(bp.fillna(0.5).to_numpy()), index=df.index)
        df["_b"] = bp
        lb, nb = ll(df, "_b", H.TUNE)
        log(f"  base {base}: TUNE ll {lb:.5f} (n {nb})")
        for rk, params in regs.items():
            name = f"O_{base}_{rk}_{design}{'_diffs' if diffs else ''}{SUF}"
            if (only and only not in name) or name in done:
                continue
            p, info = wf_xgb(df, Xh, params, *H.TUNE, design=design, Xa=Xa, refit_days=REFIT, train_days=wdw,
                             base_h=bh, base_a=-bh)
            record(name, p, params, info, {"base": base})
