"""AMFELO leakage audit: is each match's starting rating the gamer's previous ending, settled 36+ min before
kick-off? Does any starting rating already contain this match's result?"""
import numpy as np, pandas as pd
from common import H, slim, OUT

d = slim()
e = pd.read_csv(H.ELO)
e["start"] = pd.to_datetime(e.SCHEDULED_START_TIME_UTC)
print("AMFELO rows", len(e), "unique codes", e.MATCH_CODE.nunique(), "with start elo", e.PLAYER_1_ELO_STARTING.notna().sum())
e = e.drop_duplicates("MATCH_CODE")
m = d[["code", "start", "p1", "p2", "s1", "s2", "y"]].merge(
    e[["MATCH_CODE", "start", "PLAYER_1_HANDLE", "PLAYER_2_HANDLE", "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE",
       "PLAYER_1_ELO_STARTING", "PLAYER_1_ELO_ENDING", "PLAYER_2_ELO_STARTING", "PLAYER_2_ELO_ENDING"]]
    .rename(columns={"MATCH_CODE": "code", "start": "estart"}), on="code", how="left")
print("features matches", len(d), "in AMFELO", m.estart.notna().sum(), "start mismatch",
      (m.estart.notna() & (m.estart != m.start)).sum(),
      "p1 mismatch", (m.PLAYER_1_HANDLE.notna() & (m.PLAYER_1_HANDLE != m.p1)).sum(),
      "score mismatch", (m.PLAYER_1_FINAL_SCORE.notna() & ((m.PLAYER_1_FINAL_SCORE != m.s1) | (m.PLAYER_2_FINAL_SCORE != m.s2))).sum())

# long format: one row per gamer-match
L = pd.concat([
    pd.DataFrame({"code": m.code, "start": m.start, "g": m.p1, "opp": m.p2, "side": 1, "res": m.y,
                  "es": m.PLAYER_1_ELO_STARTING, "ee": m.PLAYER_1_ELO_ENDING, "os": m.PLAYER_2_ELO_STARTING}),
    pd.DataFrame({"code": m.code, "start": m.start, "g": m.p2, "opp": m.p1, "side": 2, "res": 1 - m.y,
                  "es": m.PLAYER_2_ELO_STARTING, "ee": m.PLAYER_2_ELO_ENDING, "os": m.PLAYER_1_ELO_STARTING})])
L = L.sort_values(["g", "start", "code"]).reset_index(drop=True)
grp = L.groupby("g")
L["prev_ee"] = grp.ee.shift(1)
L["prev_es"] = grp.es.shift(1)
L["prev_start"] = grp.start.shift(1)
L["next_ee"] = grp.ee.shift(-1)
L["gap_min"] = (L.start - L.prev_start).dt.total_seconds() / 60
L["nth"] = grp.cumcount()
has = L.es.notna() & L.prev_ee.notna()
eq = has & (L.es == L.prev_ee)
print(f"\nstart == previous ending: {eq.sum():,} of {has.sum():,} ({eq.sum()/has.sum():.1%})")
print("start == this match's ending:", (L.es.notna() & (L.es == L.ee)).sum(),
      " (of which ending==starting also for opp side?)")
neq = has & ~eq
print("non-equal: diff start-prev_end summary\n", (L.es - L.prev_ee)[neq].describe())
# does a non-equal start contain this match's result? delta = es - prev_ee vs this match's change ee-es
L["delta_this"] = L.ee - L.es
L["jump"] = L.es - L.prev_ee
x = L[neq & L.delta_this.notna()]
print("corr(jump, this match's change) on non-equal starts:", np.corrcoef(x.jump, x.delta_this)[0, 1].round(3),
      " corr(jump, result-0.5):", np.corrcoef(x.jump, x.res.fillna(0.5) - 0.5)[0, 1].round(3))
print("jump == this match's change:", (x.jump == x.delta_this).sum())
# start equals the ending of the previous match two back (skip) etc
L["prev2_ee"] = grp.ee.shift(2)
print("non-equal starts that equal the ending two matches back:", (neq & (L.es == L.prev2_ee)).sum())
# by month: share equal
L["month"] = L.start.dt.to_period("M")
print(L[has].assign(eq=eq[has]).groupby("month").eq.mean().round(3).to_string())

# gap between this kick-off and previous kick-off of the same gamer
print("\ngap (min) to the gamer's previous kick-off, matches with a start rating:")
print(L.loc[has, "gap_min"].describe(percentiles=[.01, .05, .1, .25, .5]).round(1).to_string())
short = has & (L.gap_min < 36)
print(f"previous kick-off < 36 min before: {short.sum():,} gamer-matches; of these start==prev end: {(short & eq).sum():,}")
# in those, is the starting rating using a result that would not have been settled by the protocol?
# K / form of the update: ee - es = K*(res - E)
x = L[L.es.notna() & L.ee.notna() & L.os.notna() & L.res.notna()].copy()
x["E"] = 1 / (1 + 10 ** (-(x.es - x.os) / 400))
x["chg"] = x.ee - x.es
x["sres"] = x.res - x.E
b = np.linalg.lstsq(np.c_[x.sres], x.chg, rcond=None)[0]
print("\nfitted K (change = K*(res-E)):", b.round(2), " resid sd", np.std(x.chg - b[0] * x.sres).round(2))
print("chg/(res-E) quantiles:", (x.chg / x.sres).replace([np.inf, -np.inf], np.nan).quantile([.05, .25, .5, .75, .95]).round(2).to_dict())
L.to_pickle(f"{OUT}/elo_long.pkl")

# match level: flag matches where either gamer's previous kick-off was < 36 min before (and start == that end)
mm = L.assign(short=(L.gap_min < 36) & eq).groupby("code").short.any()
d2 = d.merge(mm.rename("elo_short").reset_index(), on="code", how="left")
d2["p_elo"] = 1 / (1 + 10 ** (-(d2.elo_p1 - d2.elo_p2) / 400))
ok = d2.y.notna() & d2.p_elo.notna()
print("\nAMFELO raw logistic, log loss by whether a side's starting rating came from a match < 36 min earlier:")
for win in [("2025-12-01", "2026-10-19"), H.TUNE, H.TEST]:
    w = ok & H.in_window(d2, win)
    for f in [False, True]:
        s = w & (d2.elo_short.fillna(False) == f)
        if s.sum() < 30:
            continue
        mt = H.metrics(d2.y[s], d2.p_elo[s])
        mp = H.metrics(d2.y[s & d2["prod"].notna()], d2["prod"][s & d2["prod"].notna()]) if (s & d2["prod"].notna()).sum() > 30 else None
        print(f"  {win} short={f}: n={s.sum():5d} elo ll={mt['logloss']:.4f} auc={mt['auc']:.3f} slope={mt['cal_slope']:.2f}"
              + (f" | prod n={(s & d2['prod'].notna()).sum()} ll={mp['logloss']:.4f} auc={mp['auc']:.3f}" if mp else ""))
d2[["code", "elo_short"]].to_pickle(f"{OUT}/elo_short.pkl")
