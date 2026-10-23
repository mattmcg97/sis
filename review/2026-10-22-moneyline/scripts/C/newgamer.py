"""The stats where strength is unsure: strength (s_rating, s_kal) plus the site stats' matchups weighted by
g = 1 / (1 + min(p1_games, p2_games) / G0), so they count for matches with a new gamer and fade with
experience (and strength x g, so strength's own weight can move with experience). Variants:
  inter   stats only through stats x g
  both    stats and stats x g
G0 and C chosen on TUNE (walk-forward log loss), reported on TEST/SEP/LATE. Writes out/C/newgamer.txt and,
for the TUNE-best, preds/C_stats_strength_newgamer.csv.

    python newgamer.py run --part 0/2 ; python newgamer.py report
"""

import argparse
import os

import numpy as np
import pandas as pd

import models as M
from common import H, OUT

G0S = (50, 100, 200)
CS = (0.003, 0.01, 0.03, 0.1)


def prep(df):
    base = M.cols_of("kal:site")
    mg = np.minimum(df.p1_games, df.p2_games).astype(float)
    add = {}
    for G0 in G0S:
        g = 1.0 / (1.0 + mg / G0)
        for c in base + ["s_rating", "s_kal"]:
            add[f"{c}_x{G0}"] = df[c] * g
    return pd.concat([df, pd.DataFrame(add)], axis=1), base


def specs(base):
    out = []
    for G0 in G0S:
        inter = [f"{c}_x{G0}" for c in base]
        strength = ["s_rating", "s_kal", f"s_rating_x{G0}", f"s_kal_x{G0}"]
        out.append(M.Spec(f"newg[str]_G{G0}", None, strength, C=1.0))
        for C in CS:
            out.append(M.Spec(f"newg[inter]_G{G0}_C{C}", inter, strength, C=C))
            out.append(M.Spec(f"newg[both]_G{G0}_C{C}", base + inter, strength, C=C))
    return out


def run(part):
    i, n = map(int, part.split("/"))
    df, base = prep(M.load())
    path = f"{OUT}/wf_newg_{i}of{n}.pkl"
    res = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame({"code": df.code})
    for s in specs(base)[i::n]:
        if s.name not in res:
            res[s.name] = s.run(df).to_numpy()
            res.to_pickle(path)
            print("  ", s.name, flush=True)
    print("done", flush=True)


def report():
    import glob
    import json
    df = M.load()
    df = M.read_grid(df)
    for f in sorted(glob.glob(f"{OUT}/wf_newg_*.pkl")):
        df = df.merge(pd.read_pickle(f), on="code", how="left")
    pick = json.load(open(f"{OUT}/picks.json"))
    for k, v in pick.items():
        df[k] = df[v]
    names = [c for c in df.columns if c.startswith("newg[")]
    t, n = M.tune_table(df, names + ["C_strength"])
    lines = [f"TUNE ({n:,} matches):", t.to_string(index=False, float_format="{:.4f}".format)]
    best = t[t.model.str.startswith("newg[inter]") | t.model.str.startswith("newg[both]")].iloc[0].model
    beststr = t[t.model.str.startswith("newg[str]")].iloc[0].model
    lines.append(f"chosen: {best}; strength x g alone: {beststr}")
    df["C_stats_strength_newgamer"] = df[best]
    df["C_strength_newgamer"] = df[beststr]
    cols = ["C_stats_strength_newgamer", "C_strength_newgamer", "C_stats_strength", "C_strength"]
    for w, lab in ((H.TUNE, "TUNE"), (H.TEST, "TEST"), (H.SEP, "SEP"), (M.LATE, "LATE")):
        lines += [f"== {lab}", H.report(df, cols, window=w, refs=("prod", "C_strength", "C_strength_newgamer"))]
    x = df[H.in_window(df, H.TEST) & df.y.notna()].dropna(subset=cols)
    mg = np.minimum(x.p1_games, x.p2_games)
    lines.append("== TEST by the lesser gamer's matches: C_stats_strength_newgamer less C_strength_newgamer / C_strength")
    for a, b in ((0, 100), (100, 200), (200, 500), (500, 1e9)):
        sub = x[(mg >= a) & (mg < b)]
        r1 = H.boot_diff(sub, "C_stats_strength_newgamer", "C_strength_newgamer")
        r2 = H.boot_diff(sub, "C_stats_strength_newgamer", "C_strength")
        lines.append(f"  {a}-{b:g}: {len(sub):,}  vs str x g {r1[0]:+.4f} ({r1[1]:+.4f},{r1[2]:+.4f})  "
                     f"vs C_strength {r2[0]:+.4f} ({r2[1]:+.4f},{r2[2]:+.4f})")
    txt = "\n".join(lines)
    open(f"{OUT}/newgamer.txt", "w").write(txt + "\n")
    print(txt)
    H.write_preds(df, "C_stats_strength_newgamer", "C_stats_strength_newgamer")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    run(a.part) if a.cmd == "run" else report()
