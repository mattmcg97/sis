"""Step 3: is each stat a gamer's skill or noise? Split-half repeatability of every stat, offence, defence
and net (offence less defence), and whether a gamer's stat in one half foretells his scoring margin in the
other beyond his points margin.

Blocks: each gamer's play-by-play matches in a calendar month (gamer-months with >= 2 * HALF matches);
within a block the matches alternate into halves A and B, each pooled as sum(num)/sum(den) over its first
HALF matches. Across blocks (each stat less its month's league mean):
  r_half      the correlation of A with B: the reliability of a HALF-match window
  sd_between  sqrt(cov(A, B)): the spread of the gamers' true rates
  n50         the matches for a reliability of 0.5 (Spearman-Brown off r_half)
  r_win       corr(stat in A, win rate in B); r_marg corr(stat in A, margin a match in B)
  partial     the stat's t in margin_B ~ points margin_A + stat_A (the signal beyond points)
Also over the whole span (05-01 to 10-18, all of each gamer's matches alternating, gamers with >= 100).
Writes out/C/repeatability.csv and .txt.
"""

import numpy as np
import pandas as pd

from common import H, OUT, STATS, side_obs, load_drives

HALF = 25


def main():
    df = H.load(tables=False, elo=False)
    obs, has = side_obs(df)
    # long table: one row per side-match with play-by-play
    rows = []
    base = df[["code", "start", "day", "p1", "p2", "s1", "s2"]].copy()
    for tag, me, op, pf, pa in (("h", "p1", "p2", "s1", "s2"), ("a", "p2", "p1", "s2", "s1")):
        t = pd.DataFrame({"code": base.code, "day": base.day, "start": base.start, "gamer": base[me],
                          "opp": base[op], "pf": base[pf], "pa": base[pa]})
        j = 0 if tag == "h" else 2
        jd = 2 if tag == "h" else 0         # the opponent's offence: what this gamer's defence allowed
        for k in STATS:
            t[f"off_{k}_y"], t[f"off_{k}_n"] = obs[k][j], obs[k][j + 1]
            t[f"def_{k}_y"], t[f"def_{k}_n"] = obs[k][jd], obs[k][jd + 1]
        rows.append(t[has])
    L = pd.concat(rows).sort_values(["start", "code"]).reset_index(drop=True)
    L = L[(L.day >= "2026-05-01") & (L.day < "2026-10-19") & L.pf.notna()]
    L["month"] = L.day.str[:7]
    L["marg"] = L.pf - L.pa
    L["win"] = np.where(L.pf > L.pa, 1.0, np.where(L.pf < L.pa, 0.0, 0.5))
    L["k"] = L.groupby(["gamer", "month"]).cumcount()
    L["half"] = L.k % 2
    L["idx"] = L.k // 2
    Lb = L[L.idx < HALF]
    cnt = Lb.groupby(["gamer", "month", "half"]).size().unstack()
    ok = cnt[(cnt[0] >= HALF) & (cnt[1] >= HALF)].index
    Lb = Lb.set_index(["gamer", "month"]).loc[ok].reset_index()
    print(f"{len(ok)} gamer-months with 2 x {HALF} matches", flush=True)

    # whole-span version
    L["kk"] = L.groupby("gamer").cumcount()
    L["half2"] = L.kk % 2
    big = L.groupby("gamer").size()
    Lw = L[L.gamer.isin(big[big >= 100].index)]

    def pooled(T, keys, col):
        num = (T[f"{col}_y"] * T[f"{col}_n"]).groupby([T[k] for k in keys]).sum()
        den = T[f"{col}_n"].groupby([T[k] for k in keys]).sum()
        return num / den

    out = []
    stats = list(STATS) + ["win", "marg"]
    for k in stats:
        for role in ("off", "def", "net"):
            if k in ("win", "marg") and role != "off":
                continue
            res = {"stat": k, "role": role}
            for name, T, keys, hk in (("blk", Lb, ["gamer", "month", "half"], "half"),
                                      ("all", Lw, ["gamer", "half2"], "half2")):
                if k in ("win", "marg"):
                    v = T.groupby(keys)[k].mean()
                elif role == "net":
                    v = pooled(T, keys, f"off_{k}") - pooled(T, keys, f"def_{k}")
                else:
                    v = pooled(T, keys, f"{role}_{k}")
                v = v.unstack(hk)
                if name == "blk":
                    # less the month's league mean (gamer-months share months)
                    mm = v.groupby(level="month").transform("mean")
                    v = v - mm
                a, b = v[0].to_numpy(), v[1].to_numpy()
                okk = np.isfinite(a) & np.isfinite(b)
                a, b = a[okk], b[okk]
                r = np.corrcoef(a, b)[0, 1]
                cov = np.cov(a, b)[0, 1]
                res[f"r_{name}"] = r
                if name == "blk":
                    res["sd_between"] = np.sqrt(max(cov, 0))
                    res["n50"] = HALF * (1 - r) / r if r > 0 else np.inf
                    # cross-half prediction of results
                    mg = T.groupby(keys)["marg"].mean().unstack(hk)
                    wn = T.groupby(keys)["win"].mean().unstack(hk)
                    mg = mg - mg.groupby(level="month").transform("mean")
                    wn = wn - wn.groupby(level="month").transform("mean")
                    # both directions (A->B and B->A) stacked
                    xs = np.r_[v[0], v[1]]
                    pm = np.r_[mg[0], mg[1]]          # points margin in the same half as the stat
                    ym = np.r_[mg[1], mg[0]]          # margin in the other half
                    yw = np.r_[wn[1], wn[0]]
                    f = np.isfinite(xs) & np.isfinite(pm) & np.isfinite(ym)
                    res["r_marg"] = np.corrcoef(xs[f], ym[f])[0, 1]
                    res["r_win"] = np.corrcoef(xs[f], yw[f])[0, 1]
                    X = np.c_[np.ones(f.sum()), pm[f], (xs[f] - xs[f].mean()) / xs[f].std()]
                    beta, *_ = np.linalg.lstsq(X, ym[f], rcond=None)
                    e = ym[f] - X @ beta
                    s2 = e @ e / (len(e) - 3)
                    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
                    res["partial_t"] = beta[2] / se[2] if k not in ("pts", "marg") else np.nan
                    res["partial_b"] = beta[2] if k not in ("pts", "marg") else np.nan
            # league level
            if k not in ("win", "marg"):
                col = "off_" + k
                res["league"] = float((L[col + "_y"] * L[col + "_n"]).sum() / L[col + "_n"].sum())
                res["sd_match"] = float(L[col + "_y"].std())
                res["den_match"] = float(L[col + "_n"].mean())
            out.append(res)
    R = pd.DataFrame(out)
    R.to_csv(f"{OUT}/repeatability.csv", index=False)
    cols = ["stat", "role", "league", "sd_match", "sd_between", "r_blk", "n50", "r_all", "r_marg", "r_win",
            "partial_t"]
    with pd.option_context("display.width", 200, "display.max_rows", 200, "display.float_format", "{:.3f}".format):
        txt = R[cols].to_string(index=False)
    hdr = (f"Split-half repeatability: {len(ok)} gamer-months with 2 x {HALF} matches (alternating), and the "
           f"whole span 05-01..10-18 (r_all, {Lw.gamer.nunique()} gamers with >= 100 matches)\n")
    open(f"{OUT}/repeatability.txt", "w").write(hdr + txt + "\n")
    print(hdr + txt)


if __name__ == "__main__":
    main()
