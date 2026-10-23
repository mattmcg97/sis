"""Full-span walk-forward runs (RUN_FROM..RUN_TO) of the frozen configs in out/D/configs_frozen.json.
Each run writes out/D/series_<name>.csv (p_home; the margin regressors also series_<name>_m.csv),
out/D/run_<name>.json (rounds per refit), out/D/gain_<name>.csv and out/D/shap_<name>.csv (mean |SHAP|
over the TEST window, and over SEP, of the logit contributions).
    python s03_runs.py name [name ...]
"""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

CFG = json.load(open(os.path.join(OUT, "configs_frozen.json")))
log = Log(os.path.join(OUT, "s03_runs.log"))
df = load()


def run(name):
    c = CFG[name]
    feats = feature_set(df, c.get("feats", "all"))
    if c.get("extra"):
        feats = feats + [x for x in c["extra"] if x not in feats]
    if c["design"] == "plain":
        Xh, Xa = plain_matrix(df, feats, diffs=c.get("diffs", False), ident=c.get("ident", False)), None
    else:
        Xh, Xa = paired_matrix(df, feats, diffs=c.get("diffs", False), ident=c.get("ident", False))
    kind = c["kind"]
    obj = c.get("objective", "binary:logistic")
    params = dict(c["params"])
    base_h = base_a = None
    if kind == "offset":
        bp = load_series(df, c["base"])
        base_h = pd.Series(H.logit(bp.fillna(0.5).to_numpy()), index=df.index)
        base_a = -base_h
    target = "margin" if kind == "margin" else "y"
    p, info = wf_xgb(df, Xh, params, c.get("from", RUN_FROM), RUN_TO, design=c["design"], Xa=Xa, objective=obj,
                     target=target, refit_days=c.get("refit", 3), train_days=c.get("window"),
                     valid_days=c.get("valid_days", 14), base_h=base_h, base_a=base_a, contribs_from=H.TEST[0])
    if kind == "margin":
        save_series(df, p, name + "_m")
        prob = wf_on_oos(df, p, "2026-05-01", RUN_TO, kind="link", hist_days=c.get("link_days", 60))
        save_series(df, prob, name)
    else:
        save_series(df, p, name)
    json.dump({"rounds": info["rounds"], "fit_days": info["fit_days"], "secs": info["secs"], "cfg": c},
              open(os.path.join(OUT, f"run_{name}.json"), "w"))
    g = pd.Series(info["gain"]).sort_values(ascending=False)
    (g / g.sum()).rename("share").to_csv(os.path.join(OUT, f"gain_{name}.csv"))
    if info["contribs"]:
        C = pd.concat(info["contribs"])
        sep = df.loc[C.index, "day"] >= H.SEP[0]
        pd.DataFrame({"test": C.abs().mean(), "sep": C[sep.to_numpy()].abs().mean()}).sort_values(
            "test", ascending=False).to_csv(os.path.join(OUT, f"shap_{name}.csv"))
    df["_p"] = load_series(df, name)
    l, n = ll(df, "_p", H.TUNE)
    log(f"{name}: TUNE ll {l:.5f} (n {n}); rounds med {np.median(info['rounds']):.0f}; "
        f"{info['secs']:.0f}s")


for nm in sys.argv[1:]:
    run(nm)
