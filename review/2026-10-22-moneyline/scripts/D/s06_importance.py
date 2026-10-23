"""Feature importance of the frozen runs: total gain summed over every refit (share), and mean |SHAP| of the
logit contributions (xgb pred_contribs; for the paired design, (home view - away view) / 2, the
contribution to the antisymmetric logit) over TEST and SEP, by feature and by group.
    python s06_importance.py name [name ...]"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import OUT, group_of, Log
import os
import pandas as pd

log = Log(os.path.join(OUT, "s06_importance.txt"))
for name in sys.argv[1:]:
    g = pd.read_csv(os.path.join(OUT, f"gain_{name}.csv"), index_col=0)["share"]
    s = pd.read_csv(os.path.join(OUT, f"shap_{name}.csv"), index_col=0)
    s = s.drop(index="BIAS", errors="ignore")
    t = pd.DataFrame({"gain": g, "shap_test": s.test / s.test.sum(), "shap_sep": s.sep / s.sep.sum(),
                      "mean_abs_shap_test": s.test}).fillna(0)
    t["group"] = [group_of(c) for c in t.index]
    log(f"\n=== {name}: top 25 features by mean |SHAP| on TEST (shares of the total; gain = share of total gain)")
    log(t.sort_values("shap_test", ascending=False).head(25).to_string(float_format=lambda x: f"{x:.4f}"))
    gr = t.groupby("group")[["gain", "shap_test", "shap_sep"]].sum().sort_values("shap_test", ascending=False)
    gr["n_features"] = t.groupby("group").size()
    log(f"--- {name}: by group")
    log(gr.to_string(float_format=lambda x: f"{x:.3f}"))
    drv = t[t.group == "drives"].sort_values("shap_test", ascending=False).head(10)
    log(f"--- {name}: top drive features")
    log(drv.to_string(float_format=lambda x: f"{x:.4f}"))
