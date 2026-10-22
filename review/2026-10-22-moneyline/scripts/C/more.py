"""Follow-ups: (1) how much of strength pure play-by-play efficiency recovers (stats-only without points / any
scoring stat), C chosen on TUNE; (2) the strength models against the walkthrough tables on SEP; (3) the stats'
gain over strength by how much history the gamers have. Writes out/C/more.txt."""

import json

import numpy as np
import pandas as pd

import models as M
from common import H, OUT

SCORING = {"pts", "ppd", "td", "tds_pm", "fg"}


def main():
    df = M.load()
    df = M.read_grid(df)
    pick = json.load(open(f"{OUT}/picks.json"))
    for k, v in pick.items():
        df[k] = df[v]
    lines = []
    nopts = [f"kal_{s}" for s in M.STAT_KEYS if s != "pts"]
    noscore = [f"kal_{s}" for s in M.STAT_KEYS if s not in SCORING]
    runs = {}
    for lab, cols in (("pbp_nopts", nopts), ("pbp_noscore", noscore)):
        best = None
        for C in (0.003, 0.01, 0.03):
            s = M.Spec(f"stats[{lab}]_C{C}", cols, None, C=C)
            df[s.name] = s.run(df).to_numpy()
            w = df[H.in_window(df, H.TUNE) & df.y.notna() & df[s.name].notna()]
            ll = H.metrics(w.y, w[s.name])["logloss"]
            lines.append(f"  TUNE {s.name}: {ll:.4f}")
            if best is None or ll < best[1]:
                best = (s.name, ll)
        runs[lab] = best[0]
        df[lab] = df[best[0]]
    lines.append("chosen on TUNE: " + json.dumps(runs))
    cols = ["C_strength", "C_stats", "pbp_nopts", "pbp_noscore", "C_four_factors"]
    for w, lab in ((H.TUNE, "TUNE"), (H.TEST, "TEST")):
        lines += [f"== stats-only without points, {lab}", H.report(df, cols, window=w, refs=("prod", "C_strength"))]
    # against the walkthrough tables on SEP
    tb = ["tbl_glmer_both_p", "tbl_avg_p", "tbl_form_p"]
    for w, lab in ((H.SEP, "SEP"), (M.LATE, "LATE")):
        lines += [f"== {lab}: strength and stats models against the walkthrough tables",
                  H.report(df, ["C_strength", "C_stats_strength", "C_stats_strength_mlink"] + tb, window=w,
                           refs=("prod", "tbl_glmer_both_p"))]
    # the stats' gain by the gamers' play-by-play history (fewer matches: strength less sure)
    x = df[H.in_window(df, H.TEST) & df.y.notna()].dropna(subset=["C_stats_strength", "C_strength_nested"])
    n = np.minimum(x.p1_games, x.p2_games)
    lines.append("== TEST: C_stats_strength less C_strength_nested by the lesser gamer's matches played")
    for a, b in ((0, 200), (200, 500), (500, 1e9)):
        sub = x[(n >= a) & (n < b)]
        if len(sub) > 200:
            mu, lo, hi = H.boot_diff(sub, "C_stats_strength", "C_strength_nested")
            mu2, lo2, hi2 = H.boot_diff(sub, "C_stats_strength_mlink", "C_strength_nested")
            lines.append(f"  {int(a)}-{b:g} matches: {len(sub):,} matches, logit {mu:+.4f} ({lo:+.4f},{hi:+.4f}), "
                         f"mlink {mu2:+.4f} ({lo2:+.4f},{hi2:+.4f})")
    txt = "\n".join(lines)
    open(f"{OUT}/more.txt", "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
