"""Does a pre-match model need a pace output beside expected points? Three checks off the drive stats.

    python -m xgb.pace --features xgb/out/features.csv --drives xgb/data/drive_stats.csv --since 2026-08-01

  1. a trait   do gamers keep their pace? Each gamer's drives, snaps and seconds a snap a match in odd
               ISO weeks against even ones (gamers with MIN_HALF matches in each), beside the same for
               points per drive (efficiency) for scale
  2. foreseen  how much of a match's pace is known before kick-off: the two gamers' drives a match over
               their last 50 matches with play-by-play (xgb/features.py's drv_ rates, as of kick-off)
               against the drives the match had; and how much of it the expected total already holds
  3. needed    given glmer's expected points, does the pace foreseen widen the result? The total's and
               the margin's squared misses against glmer, by fifth of foreseen pace and in a
               regression; and the moneyline with the margin's sd scaled by the square root of the
               foreseen drives (the same sd on average) against one sd for every match
"""

import argparse
import math

import numpy as np
import pandas as pd

MIN_HALF = 30
MARGIN_SD = 9.35


def _erf_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def trait(dr, hist):
    """Split-half correlations of each gamer's per-match pace and efficiency."""
    m = dr.merge(hist, on="match_code")
    rows = []
    for side, opp in (("h", "a"), ("a", "h")):
        rows.append(pd.DataFrame({
            "gamer": m["home"] if side == "h" else m["away"], "week": m.start.dt.isocalendar().week.values,
            "drives": m[f"{side}_drives"] + m[f"{opp}_drives"], "own_drives": m[f"{side}_drives"],
            "snaps": m[f"{side}_snaps"], "sec_snap": m[f"{side}_seconds"] / m[f"{side}_snaps"].replace(0, np.nan),
            "ppd": m[f"final_p{1 if side == 'h' else 2}"] / m[f"{side}_drives"].replace(0, np.nan)}))
    long = pd.concat(rows, ignore_index=True)
    long["half"] = long.week % 2
    out = []
    for k in ("drives", "own_drives", "snaps", "sec_snap", "ppd"):
        g = long.groupby(["gamer", "half"])[k].agg(["mean", "count"]).unstack("half")
        ok = (g["count"] >= MIN_HALF).all(axis=1)
        r = np.corrcoef(g["mean"][0][ok], g["mean"][1][ok])[0, 1]
        sd = long.groupby("gamer")[k].mean()[ok.index[ok]].std()
        out.append((k, int(ok.sum()), r, sd, long[k].mean()))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="xgb/out/features.csv")
    ap.add_argument("--drives", default="xgb/data/drive_stats.csv")
    ap.add_argument("--since", default="2026-08-01")
    ap.add_argument("--out", default="xgb/out/pace.txt")
    a = ap.parse_args(argv)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    f = pd.read_csv(a.features, parse_dates=["start"])
    dr = pd.read_csv(a.drives)
    hist = f[["code", "start"]].rename(columns={"code": "match_code"})

    say("1. Do gamers keep their pace? Split-half (odd v even ISO weeks), gamers with "
        f"{MIN_HALF}+ matches in each half")
    say(f"   {'per match':34s}{'gamers':>8s}{'r':>8s}{'sd across gamers':>18s}{'league mean':>13s}")
    names = {"drives": "drives in his matches (both sides)", "own_drives": "his own drives",
             "snaps": "his own snaps", "sec_snap": "seconds a snap, his offence",
             "ppd": "points per drive (efficiency)"}
    for k, n, r, sd, mean in trait(dr, hist):
        say(f"   {names[k]:34s}{n:8d}{r:8.2f}{sd:18.3f}{mean:13.3f}")

    t = f[(f.start >= a.since) & f.s1.notna() & f.glmer_p1.notna()].merge(
        dr[["match_code", "h_drives", "a_drives", "h_snaps", "a_snaps"]], left_on="code", right_on="match_code")
    need = ["p1_drv_off_drives_50", "p1_drv_def_drives_50", "p2_drv_off_drives_50", "p2_drv_def_drives_50"]
    t = t.dropna(subset=need).copy()
    t["drives"] = t.h_drives + t.a_drives
    t["pace_hat"] = t[need].sum(axis=1) / 2           # each side's own drives and those against it, averaged
    t["exp_total"] = t.glmer_p1 + t.glmer_p2
    t["exp_margin"] = t.glmer_p1 - t.glmer_p2
    t["total"] = t.s1 + t.s2
    t["margin"] = t.s1 - t.s2
    say()
    say(f"2. Is a match's pace known before kick-off? {len(t):,} matches from {a.since}")
    r = np.corrcoef(t.pace_hat, t.drives)[0, 1]
    b = np.polyfit(t.pace_hat, t.drives, 1)
    say(f"   drives a match: mean {t.drives.mean():.2f}, sd {t.drives.std():.2f}; foreseen sd {t.pace_hat.std():.2f}")
    say(f"   foreseen against real: r {r:.3f} (R^2 {r * r:.3f}), slope {b[0]:.2f}")
    say(f"   foreseen pace against glmer's expected total: r {np.corrcoef(t.pace_hat, t.exp_total)[0, 1]:.3f}; "
        f"real drives against real total: r {np.corrcoef(t.drives, t.total)[0, 1]:.3f}")

    say()
    say("3. Given glmer's expected points, does the foreseen pace widen the result?")
    t["q"] = pd.qcut(t.pace_hat, 5, labels=False)
    say(f"   {'fifth of foreseen pace':24s}{'pace':>7s}{'drives':>8s}{'total sd':>10s}{'margin sd':>11s}"
        f"{'total bias':>12s}{'matches':>9s}")
    for q, g in t.groupby("q"):
        say(f"   {q + 1:<24d}{g.pace_hat.mean():7.2f}{g.drives.mean():8.2f}{(g.total - g.exp_total).std():10.2f}"
            f"{(g.margin - g.exp_margin).std():11.2f}{(g.total - g.exp_total).mean():+12.2f}{len(g):9d}")
    z = (t.pace_hat - t.pace_hat.mean()) / t.pace_hat.std()
    for name, miss in (("total", t.total - t.exp_total), ("margin", t.margin - t.exp_margin)):
        y = miss ** 2
        X = np.column_stack([np.ones(len(t)), z, (t.exp_total - t.exp_total.mean()) / t.exp_total.std()])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ coef
        cov = np.linalg.inv(X.T @ X) * (res @ res) / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(cov))
        say(f"   {name} squared miss: {coef[1]:+.2f} a sd of foreseen pace (t {coef[1] / se[1]:+.1f}), "
            f"{coef[2]:+.2f} a sd of expected total (t {coef[2] / se[2]:+.1f}); mean {y.mean():.1f}")
    dec = t.margin != 0
    won = (t.margin > 0).astype(float)
    scale = np.sqrt(t.pace_hat / t.pace_hat.mean())
    scale = scale / scale.mean()
    for name, sd in (("one sd", MARGIN_SD), ("sd x sqrt(foreseen drives)", MARGIN_SD * scale)):
        p = np.clip(1 - _erf_cdf(-t.exp_margin / sd), 1e-9, 1 - 1e-9)
        ll = -(won * np.log(p) + (1 - won) * np.log(1 - p))
        t[f"ll_{name}"] = ll
        say(f"   moneyline log loss, {name}: {ll[dec].mean():.5f}")
    d = (t["ll_sd x sqrt(foreseen drives)"] - t["ll_one sd"])[dec]
    say(f"   difference {d.mean():+.5f} (t {d.mean() / (d.std() / math.sqrt(len(d))):+.1f})")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
