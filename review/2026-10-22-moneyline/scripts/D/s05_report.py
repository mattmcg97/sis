"""Agent D's final tables, every frozen run (out/D/series_<name>.csv) through harness.report on TUNE, TEST,
SEP and 09-22..10-18 (common matches, day-resampled 95% intervals):
  1 main: prod, the simple link on rating_margin (link_rm), the ridge base, and the XGB models, vs prod and link_rm
  2 offsets against their own bases; paired against plain
  3 ablations (paired design, same params) against the full paired model
  4 walk-forward recalibration of each XGB model: Platt (a + b logit p) and temperature (b logit p) fitted on
    the previous HIST days' out-of-sample predictions only (HIST chosen on TUNE from 30/60/120)
  5 margin link window (30/60/120 days of previous OOS margins), chosen on TUNE
Then preds/D_<model>.csv for 2026-06-01..10-18.
    python s05_report.py
"""
import sys, json
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

open(os.path.join(OUT, "s05_report.txt"), "w").close()
log = Log(os.path.join(OUT, "s05_report.txt"))
df = load()
LATE = ("2026-09-22", "2026-10-19")
WINDOWS = (H.TUNE, H.TEST, H.SEP, LATE)
have = lambda n: os.path.exists(os.path.join(OUT, f"series_{n}.csv"))
names = ["link_rm", "ridge_p", "xgb_cls", "xgb_paired", "xgb_paired_ident", "xgb_paired_nodrv", "xgb_margin",
         "xgb_margin_nodrv", "xgb_offset", "xgb_offset_nodrv", "xgb_offset_rm", "abl_nodrv", "abl_ratings", "abl_form"]
names = [n for n in names if have(n)]
for n in names:
    df[n] = load_series(df, n)
xgbs = [n for n in names if n.startswith("xgb")]


def tbl(cols, refs, title, windows=WINDOWS):
    log("=" * 130)
    log(title)
    for w in windows:
        log(H.report(df, [c for c in cols if c in df], window=w, refs=refs))
        log("")


# 5: margin link window, on TUNE only
for mm in ("xgb_margin", "xgb_margin_nodrv"):
    if not have(mm + "_m"):
        continue
    m = load_series(df, mm + "_m")
    res = {}
    for hd in (30, 60, 120):
        k = f"_ml{hd}"
        df[k] = wf_on_oos(df, m, "2026-05-01", RUN_TO, kind="link", hist_days=hd)
        res[hd] = ll(df, k, H.TUNE)[0]
    log(f"{mm}: link window (days of previous OOS margins) -> TUNE log loss: " +
        ", ".join(f"{k}: {v:.5f}" for k, v in res.items()))
    best = min(res, key=res.get)
    log(f"  chosen: {best} days ({mm} = this link on the walk-forward XGB margin)")
    df[mm] = df[f"_ml{best}"]
    log(f"  sd of the predicted margin: TUNE {m[H.in_window(df, H.TUNE)].std():.2f}, "
        f"TEST {m[H.in_window(df, H.TEST)].std():.2f}; realised margin sd TEST "
        f"{df.margin[H.in_window(df, H.TEST)].std():.2f}")

# 4: recalibration window on TUNE (Platt), then both kinds at that window
rc = {}
for c in xgbs:
    best, bl = None, 9
    for hd in (30, 60, 120):
        x = wf_on_oos(df, df[c], "2026-05-15", RUN_TO, kind="platt", hist_days=hd)
        df["_x"] = x
        l = ll(df, "_x", H.TUNE)[0]
        if l < bl:
            best, bl = hd, l
            df[f"{c}_platt"] = x
    df[f"{c}_temp"] = wf_on_oos(df, df[c], "2026-05-15", RUN_TO, kind="temp", hist_days=best)
    rc[c] = best
log(f"recalibration history window chosen on TUNE (Platt): {rc}")

main = ["link_rm", "ridge_p"] + [c for c in xgbs]
# TUNE ranking of every model (the selection: best TUNE log loss, raw or recalibrated)
log("=" * 130)
log("TUNE log loss, every model on all TUNE matches it prices (selection by TUNE only)")
for c in sorted([c for c in df.columns if c in names or c.endswith(("_platt", "_temp"))], key=lambda c: ll(df, c, H.TUNE)[0]):
    l, n = ll(df, c, H.TUNE)
    log(f"  {c:28s} {l:.5f}  (n {n})")
tbl(main, ("prod", "link_rm"), "1. Main table (each vs prod and vs the simple link on rating_margin)")
tbl(["xgb_offset", "ridge_p", "xgb_offset_rm", "link_rm"], ("ridge_p", "link_rm"),
    "2a. Offset models vs their bases (xgb_offset: base ridge_p; xgb_offset_rm: base link_rm)")
tbl(["xgb_offset_nodrv", "ridge_p"], ("ridge_p",), "2a'. No-drive offset vs its base")
tbl(["xgb_cls", "xgb_paired", "xgb_paired_ident"], ("xgb_cls",), "2b. Paired vs plain design (same params)")
tbl(["xgb_paired", "abl_nodrv", "abl_form", "abl_ratings"], ("xgb_paired", "link_rm"),
    "3. Ablations (paired design, same params, refit 3, window 180)")
for c in xgbs:
    tbl([c, f"{c}_platt", f"{c}_temp"], (c, "prod"), f"4. Recalibration of {c} (window {rc[c]} days)")

# calibration slopes by month (all matches each model prices)
log("=" * 130)
log("Calibration slope by month (each column on the matches it prices; prod on its own)")
rows = []
for mon in ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]:
    w = df[(df.day.str[:7] == mon) & df.y.notna()]
    r = {"month": mon}
    for c in ["prod", "link_rm", "ridge_p"] + xgbs + [f"{c}_platt" for c in xgbs]:
        x = w[w[c].notna()]
        r[c] = round(H.calib(x.y, x[c])[1], 2) if len(x) > 100 else np.nan
    rows.append(r)
log(pd.DataFrame(rows).set_index("month").T.to_string())

json.dump({"recal_hist": rc}, open(os.path.join(OUT, "s05_choices.json"), "w"))
keep = ["code", "day"] + [c for c in df.columns if c in names or c.endswith(("_platt", "_temp"))] + ["prod"]
df[keep].to_csv(os.path.join(OUT, "all_preds_wide.csv"), index=False)

# preds files: 2026-06-01..10-18
PRED = {"D_link_rm": "link_rm", "D_ridge": "ridge_p", "D_xgb_cls": "xgb_cls", "D_xgb_paired": "xgb_paired",
        "D_xgb_paired_platt": "xgb_paired_platt", "D_xgb_margin_link": "xgb_margin", "D_xgb_offset": "xgb_offset",
        "D_xgb_offset_rm": "xgb_offset_rm", "D_xgb_paired_ident": "xgb_paired_ident",
        "D_xgb_paired_nodrv": "xgb_paired_nodrv", "D_xgb_margin_link_nodrv": "xgb_margin_nodrv",
        "D_xgb_offset_nodrv": "xgb_offset_nodrv"}
span = (df.day >= "2026-06-01") & (df.day < "2026-10-19")
for f, c in PRED.items():
    if c in df:
        x = df.loc[span].copy()
        log(f"  wrote {H.write_preds(x, c, f)} ({x[c].notna().sum():,} matches)")
