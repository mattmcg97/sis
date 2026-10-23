"""AMFELO vs our own clean online ratings on back-to-back matches (a side's previous kick-off exactly 36 min
earlier, so its previous result enters right at kick-off) vs the rest; and on new gamers (seeded ratings)."""
import common as C
import numpy as np, pandas as pd
H = C.H
d = C.slim()
L = pd.read_pickle(f"{C.OUT}/elo_long.pkl")
gap = L.groupby("code").gap_min.min()
d = d.merge(gap.rename("gap").reset_index(), on="code", how="left")
d["b2b"] = np.where(d.gap <= 36, "b2b (36 min)", "gap > 36 min")
d["mg"] = np.minimum(d.p1_games, d.p2_games)
d["exp"] = pd.cut(d.mg, [-1, 29, 200, 1e9], labels=["<30", "30-200", ">200"]).astype(str)
cols = ["B_amfelo_link", "B_amfelo_raw", "B_elo_tuned", "B_glicko", "B_bt_team"]
for c in cols:
    try:
        d = d.merge(pd.read_csv(f"{C.ML}/preds/{c}.csv").rename(columns={"p_home": c}), on="code", how="left")
    except FileNotFoundError:
        cols.remove(c)
lines = []
for win in [C.FULLWIN, H.TUNE, H.TEST]:
    w = d[H.in_window(d, win) & d.y.notna()].dropna(subset=cols)
    for by in ["b2b", "exp"]:
        t = {}
        for c in cols:
            t[c] = w.groupby(by).apply(lambda g: H.logloss_each(g.y.to_numpy(), g[c].to_numpy()).mean())
        t = pd.DataFrame(t).T
        t.loc["n"] = w.groupby(by).size()
        lines.append(f"{win} by {by}\n" + t.round(4).to_string())
        if "B_glicko" in cols:
            for gname, g in w.groupby(by):
                if len(g) > 50:
                    mu, lo, hi = H.boot_diff(g, "B_amfelo_link", "B_glicko")
                    lines.append(f"   AMFELO link - glicko, {gname}: {mu:+.4f} ({lo:+.4f},{hi:+.4f})  n={len(g)}")
print("\n".join(lines))
open(f"{C.OUT}/amfelo_b2b.txt", "w").write("\n".join(lines))
