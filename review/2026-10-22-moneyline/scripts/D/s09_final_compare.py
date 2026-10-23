"""The TUNE-selected XGB (xgb_margin_nodrv), the best design of each kind, and the 4-feature logistic lr4
against prod, link_rm and lr4 on every window; writes preds/D_lr4.csv and preds/D_xgb_offset_lr4.csv."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

open(os.path.join(OUT, "s09_final_compare.txt"), "w").close()
log = Log(os.path.join(OUT, "s09_final_compare.txt"))
df = load()
w = pd.read_csv(os.path.join(OUT, "all_preds_wide.csv")).set_index("code")
for c in ["link_rm", "xgb_paired", "xgb_paired_nodrv", "xgb_margin_nodrv", "xgb_offset_nodrv"]:
    df[c] = w[c].reindex(df.code).to_numpy()
df["lr4"] = load_series(df, "lr4_full")
df["xgb_offset_lr4"] = load_series(df, "xgb_offset_lr4")
cols = ["link_rm", "lr4", "xgb_paired", "xgb_paired_nodrv", "xgb_margin_nodrv", "xgb_offset_nodrv", "xgb_offset_lr4"]
for wdw in (H.TUNE, H.TEST, H.SEP, ("2026-09-22", "2026-10-19")):
    log(H.report(df, cols, window=wdw, refs=("prod", "link_rm", "lr4")))
    log("")
span = (df.day >= "2026-06-01") & (df.day < "2026-10-19")
for f, c in (("D_lr4", "lr4"), ("D_xgb_offset_lr4", "xgb_offset_lr4")):
    log(f"wrote {H.write_preds(df.loc[span], c, f)}")
