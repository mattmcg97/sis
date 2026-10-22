"""TUNE-only search for the XGB classifier (2026-06-01..07-31, refit every 3 days, early stopping on the
window's last 14 days, then refit on the whole window). Stages:
  A  design: plain, plain+diffs, paired, paired+diffs, plain+gamer identity (default params, 120 days)
  B  window: 60 / 120 / 180 / all days for the stage-A design(s)
  C  params: a 24-config random search (depth, eta, min_child_weight, subsample, colsample, lambda)
Results append to out/D/tune_results.csv; TUNE predictions to out/D/tune_preds.csv (code + one column each).
    python s02_tune.py A|B|C [design ...]
"""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

stage = sys.argv[1]
log = Log(os.path.join(OUT, f"s02_tune_{stage}{os.environ.get('ONLY', '')}.log"))
df = load()
feats = feature_set(df, "all")
RES = os.path.join(OUT, "tune_results.csv")
PRED = os.path.join(OUT, "tune")      # one file of TUNE predictions per config: tune/<name>.csv
os.makedirs(PRED, exist_ok=True)
done = set(pd.read_csv(RES).name) if os.path.exists(RES) else set()
only = os.environ.get("ONLY")         # run only configs whose name contains this (to split across 2 processes)
mats = {}


def matrices(design, diffs, ident):
    k = (design, diffs, ident)
    if k not in mats:
        if design == "plain":
            mats[k] = (plain_matrix(df, feats, diffs=diffs, ident=ident), None)
        else:
            mats[k] = paired_matrix(df, feats, diffs=diffs, ident=ident)
    return mats[k]


REFIT = int(os.environ.get("REFIT", "3"))


def run(name, design, diffs=False, ident=False, window=120, params=None, refit=None):
    refit = refit or REFIT
    if refit != 3:
        name = f"{name}_r{refit}"
    if only and only not in name:
        return
    if name in done:
        log(f"  {name}: done already")
        return
    params = dict(DEFAULT if params is None else params)
    Xh, Xa = matrices(design, diffs, ident)
    p, info = wf_xgb(df, Xh, params, *H.TUNE, design=design, Xa=Xa, refit_days=refit, train_days=window)
    df["_p"] = p
    l, n = ll(df, "_p", H.TUNE)
    w = df[H.in_window(df, H.TUNE) & df.y.notna()]
    a, b = H.calib(w.y, w._p)
    row = dict(name=name, stage=stage, design=design, diffs=diffs, ident=ident, window=window or 0, refit=refit,
               params=json.dumps(params), tune_ll=l, n=n, slope=b, rounds_med=float(np.median(info["rounds"])),
               secs=info["secs"])
    pd.DataFrame([row]).to_csv(RES, mode="a", header=not os.path.exists(RES), index=False)
    pd.DataFrame({"code": df.code, "p": p}).dropna().to_csv(os.path.join(PRED, f"{name}.csv"), index=False)
    done.add(name)
    log(f"  {name}: TUNE ll {l:.5f} slope {b:.2f} rounds med {row['rounds_med']:.0f} ({info['secs']:.0f}s)")


if stage == "A":
    run("A_plain", "plain")
    run("A_plain_diffs", "plain", diffs=True)
    run("A_paired", "paired")
    run("A_paired_diffs", "paired", diffs=True)
    run("A_plain_ident", "plain", ident=True)
    run("A_paired_ident", "paired", ident=True)
elif stage == "B":
    for spec in sys.argv[2:]:
        design, diffs = spec.split(":")
        for wdw in (60, 120, 180, None):
            run(f"B_{design}{'_diffs' if diffs == '1' else ''}_w{wdw or 'all'}", design, diffs == "1", window=wdw)
elif stage == "D":
    # design re-check with the stage-C winner's params (window 180)
    import json as _j
    c = _j.loads(sys.argv[2])
    run("D_paired_ident", "paired", ident=True, window=180, params=c)
    run("D_plain", "plain", window=180, params=c)
    run("D_plain_diffs", "plain", diffs=True, window=180, params=c)
elif stage == "C":
    # spec design:diffs:window
    rng = np.random.default_rng(20260601)
    grid = dict(max_depth=[2, 3, 4, 6], eta=[0.02, 0.05, 0.1], min_child_weight=[10, 50, 200],
                subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.3, 0.6, 0.9], reg_lambda=[1.0, 10.0, 50.0])
    cfgs = []
    for i in range(24):
        cfgs.append({k: (v[rng.integers(len(v))]) for k, v in grid.items()})
    cfgs = [{k: (int(v) if k == "max_depth" else float(v)) for k, v in c.items()} for c in cfgs]
    for spec in sys.argv[2:]:
        design, diffs, wdw = spec.split(":")
        wdw = None if wdw == "all" else int(wdw)
        ncfg = int(os.environ.get("NCFG", "24"))
        part = os.environ.get("PART")
        for i, c in enumerate(cfgs[:ncfg]):
            if part is not None and i % 2 != int(part):
                continue
            c = dict(c, reg_alpha=0.0, gamma=0.0)
            run(f"C{i:02d}_{design}{'_diffs' if diffs == '1' else ''}", design, diffs == "1", window=wdw, params=c)
