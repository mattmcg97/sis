"""(1) Check that the top gain features hold on TUNE-era fits alone (refits up to 07-31 of the paired model):
the 4-feature logistic lr4 uses them. (2) Offset XGB on a stronger simple base: lr4 = walk-forward logistic
on rating_margin, h2h_margin, exp_margin_all and the ridge strength margin (180 days, daily), from 2025-12-20;
XGB (paired, no drive stats, the offset's R2 params, refit 3, window 180) learns corrections on logit(lr4)."""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

open(os.path.join(OUT, "s08_offset_lr4.txt"), "w").close()
log = Log(os.path.join(OUT, "s08_offset_lr4.txt"))
df = load()
cfg = json.load(open(os.path.join(OUT, "configs_frozen.json")))
# (1) gains of the paired model's TUNE-era refits only
feats = feature_set(df, "all")
Xh, Xa = paired_matrix(df, feats)
_, info = wf_xgb(df, Xh, cfg["xgb_paired"]["params"], "2026-06-01", "2026-08-01", design="paired", Xa=Xa,
                 refit_days=3, train_days=180)
g = pd.Series(info["gain"]).sort_values(ascending=False)
log("paired model, refits 06-01..07-31 only: top gain shares\n" + (g / g.sum()).head(8).to_string())
# (2) lr4 from the start, then the offset
df["ridge_margin"] = load_series(df, "ridge_margin")
cols = ["rating_margin", "h2h_margin", "exp_margin_all", "ridge_margin"]


def fit(tr):
    return fit_logit(tr[cols].fillna(0).to_numpy(), tr.y.to_numpy())


def predict(b, d):
    return apply_logit(b, d[cols].fillna(0).to_numpy())


df["lr4"] = H.walk_forward(df, fit, predict, "2025-12-20", RUN_TO, refit_days=1, train_days=180)
save_series(df, df.lr4, "lr4_full")
log(f"lr4 coverage from {df.loc[df.lr4.notna(), 'day'].min()}; TUNE ll {ll(df, 'lr4', H.TUNE)[0]:.5f}")
c = cfg["xgb_offset_nodrv"]
f2 = feature_set(df, "nodrv")
Xh, Xa = paired_matrix(df, f2)
bh = pd.Series(H.logit(df.lr4.fillna(0.5).to_numpy()), index=df.index)
p, info = wf_xgb(df, Xh, c["params"], RUN_FROM, RUN_TO, design="paired", Xa=Xa, refit_days=3, train_days=180,
                 base_h=bh, base_a=-bh)
df["xgb_offset_lr4"] = p
save_series(df, p, "xgb_offset_lr4")
log(f"xgb_offset_lr4: TUNE ll {ll(df, 'xgb_offset_lr4', H.TUNE)[0]:.5f}; rounds med {np.median(info['rounds']):.0f}, "
    f"refits with 0 trees: {sum(r == 0 for r in info['rounds'])} of {len(info['rounds'])}")
for wdw in (H.TUNE, H.TEST, H.SEP, ("2026-09-22", "2026-10-19")):
    log(H.report(df, ["xgb_offset_lr4", "lr4"], window=wdw, refs=("lr4", "prod")))
    log("")
