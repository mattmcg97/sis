"""Extras: (1) how much of XGB a 3-4 feature walk-forward logistic gets (the top SHAP features: rating_margin,
h2h_margin, exp_margin_all, + the ridge strength margin); (2) key pairwise comparisons; (3) D's models next to
the other agents' preds (informational, TEST/SEP, common matches)."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

open(os.path.join(OUT, "s07_extras.txt"), "w").close()
log = Log(os.path.join(OUT, "s07_extras.txt"))
df = load()
for n in ["link_rm", "ridge_p", "ridge_margin", "xgb_paired", "xgb_paired_nodrv", "xgb_offset", "xgb_offset_nodrv"]:
    df[n] = load_series(df, n)
w = pd.read_csv(os.path.join(OUT, "all_preds_wide.csv"))
df["xgb_margin_nodrv"] = w.set_index("code").xgb_margin_nodrv.reindex(df.code).to_numpy()
df["xgb_margin"] = w.set_index("code").xgb_margin.reindex(df.code).to_numpy()


def wf_logit(cols, td=180):
    def fit(tr):
        x = tr[cols].fillna(0).to_numpy()
        return fit_logit(x, tr.y.to_numpy())

    def predict(b, d):
        return apply_logit(b, d[cols].fillna(0).to_numpy())
    return H.walk_forward(df, fit, predict, "2026-06-01", RUN_TO, refit_days=1, train_days=td)


df["lr3"] = wf_logit(["rating_margin", "h2h_margin", "exp_margin_all"])
df["lr4"] = wf_logit(["rating_margin", "h2h_margin", "exp_margin_all", "ridge_margin"])
for c in ("lr3", "lr4"):
    log(f"{c}: TUNE ll {ll(df, c, H.TUNE)[0]:.5f}")
save_series(df, df.lr3, "lr3")
save_series(df, df.lr4, "lr4")
for wdw in (H.TUNE, H.TEST, H.SEP, ("2026-09-22", "2026-10-19")):
    log(H.report(df, ["lr3", "lr4", "ridge_p", "xgb_paired", "xgb_paired_nodrv", "xgb_margin_nodrv", "xgb_offset_nodrv"],
                 window=wdw, refs=("lr3", "ridge_p")))
    log("")
# D vs the other agents (informational only; nothing chosen here)
o = H.read_preds(df[["code", "day", "y", "prod"]].copy())
o["D_best_tune"] = df["xgb_margin_nodrv"].to_numpy()
pc = [c for c in o.columns if c.startswith("pred_")]
log("Other agents' preds on TEST and SEP, each on the matches it and prod price (log loss; vs prod)")
for wdw in (H.TEST, H.SEP):
    x = o[H.in_window(o, wdw) & o.y.notna() & o["prod"].notna()]
    rows = []
    for c in pc + ["D_best_tune"]:
        z = x[x[c].notna()]
        if len(z) < 0.9 * len(x):
            continue
        rows.append((c, len(z), H.logloss_each(z.y.to_numpy(), z[c].to_numpy()).mean(),
                     H.logloss_each(z.y.to_numpy(), z["prod"].to_numpy()).mean()))
    t = pd.DataFrame(rows, columns=["model", "n", "ll", "ll_prod"]).assign(d=lambda t: t.ll - t.ll_prod)
    log(f"{wdw}:\n" + t.sort_values("ll").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
