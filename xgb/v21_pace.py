"""Does v21's kick-off spread already follow pace? v21-report's matches table against pace_eff's.

    python -m xgb.v21_pace --v21 eAMFCalibrator/out_v21_pace/v21_matches.csv \
        --pace xgb/out/pace_eff/pace_eff_prior.csv

By fifth of the match's expected drives (pace_eff's exp_drives, as of kick-off), on the matches both
priced:
  real    the margin's and total's real spread about each model's mean (the misses' sd)
  v21     the sd of v21's own kick-off distributions (sim_margin_sd, sim_total_sd)
  pace    the sd pace and efficiency give (margin_sd, total_sd) about glmer's points

If v21's own sd rises with pace as the real misses do, its snap draws carry pace already. If it stays
flat, the expected drives are something v21 should take. A regression says the same in one number:
each match's squared miss about v21's mean against v21's own variance and the expected drives (in sd
units); a drives term clear of zero is spread v21 is missing.
"""

import argparse
import math

import numpy as np
import pandas as pd


def norm_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v21", required=True, help="v21-report's v21_matches.csv (with sim_margin_sd)")
    ap.add_argument("--pace", default="xgb/out/pace_eff/pace_eff_prior.csv")
    ap.add_argument("--out", default="xgb/out/v21_pace.txt")
    a = ap.parse_args(argv)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    v = pd.read_csv(a.v21)
    if "sim_margin_sd" not in v:
        raise SystemExit(f"{a.v21}: no sim_margin_sd -- rerun v21-report from this branch")
    p = pd.read_csv(a.pace)
    t = v.merge(p[["MATCH_CODE", "exp_drives", "Pred_P1_Points", "Pred_P2_Points", "margin_sd", "total_sd",
                   "p_home"]], on="MATCH_CODE")
    t = t.dropna(subset=["exp_drives", "sim_margin_sd", "real_home", "real_away"]).copy()
    t["margin"], t["total"] = t.real_home - t.real_away, t.real_home + t.real_away
    t["v21_m"], t["v21_t"] = t.sim_home - t.sim_away, t.sim_home + t.sim_away
    t["g_m"], t["g_t"] = t.Pred_P1_Points - t.Pred_P2_Points, t.Pred_P1_Points + t.Pred_P2_Points
    say(f"  {len(t):,} matches in both, {t.start.min()[:10]} to {t.start.max()[:10]}")
    t["q"] = pd.qcut(t.exp_drives, 5, labels=False)
    for what, real, mean_v, sd_v, mean_p, sd_p in (
            ("margin", "margin", "v21_m", "sim_margin_sd", "g_m", "margin_sd"),
            ("total", "total", "v21_t", "sim_total_sd", "g_t", "total_sd")):
        say()
        say(f"  the {what}'s spread by fifth of expected drives")
        say(f"  {'fifth':>6s}{'drives':>8s}{'matches':>9s}{'real about v21':>16s}{'v21 own':>9s}"
            f"{'real about glmer':>18s}{'pace sd':>9s}")
        for q, g in t.groupby("q"):
            say(f"  {q + 1:>6d}{g.exp_drives.mean():8.2f}{len(g):9d}{(g[real] - g[mean_v]).std():16.2f}"
                f"{g[sd_v].mean():9.2f}{(g[real] - g[mean_p]).std():18.2f}{g[sd_p].mean():9.2f}")
        miss2 = (t[real] - t[mean_v]) ** 2
        z = (t.exp_drives - t.exp_drives.mean()) / t.exp_drives.std()
        varies = t[sd_v].std() > 1e-9
        X = np.column_stack([np.ones(len(t))] + ([t[sd_v] ** 2] if varies else []) + [z])
        coef, *_ = np.linalg.lstsq(X, miss2, rcond=None)
        res = miss2 - X @ coef
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (res @ res) / (len(t) - X.shape[1])))
        own = f" + {coef[1]:.2f} x v21's own variance (t {coef[1] / se[1]:+.1f})" if varies else \
            " (v21's own sd is the same for every match)"
        say(f"  squared miss about v21 = {coef[0]:+.1f}{own} {coef[-1]:+.2f} a sd of expected drives "
            f"(t {coef[-1] / se[-1]:+.1f})")
    dec = t.margin != 0
    won = (t.margin > 0).astype(float)
    say()
    for name, prob in (("v21 (its own kick-off)", t.sim_p_home), ("glmer, sd by pace x efficiency", t.p_home)):
        q = np.clip(prob.values, 1e-9, 1 - 1e-9)
        ll = -(won * np.log(q) + (1 - won) * np.log(1 - q))[dec].mean()
        say(f"  moneyline log loss, {name}: {ll:.4f}")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
