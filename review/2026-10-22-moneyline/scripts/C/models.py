"""Agent C's models on the h2h-site stats (rebuilt from our play-by-play), walk-forward through harness.

Features (each a home-less-away matchup, so the win probability is antisymmetric up to the intercept):
  kal_<s>    the Kalman filter's (kalman.py) expected home offence rate less the expected away offence rate:
             (mu + home + off[p1] + def[p2] + teams) - (mu + off[p2] + def[p1] + teams), opponent-adjusted
             and shrunk
  eb<hl>_<s> EB-shrunk recency-weighted raw rates (eb.py): off[p1] + def[p2] - off[p2] - def[p1]
  raw<w>_<s> features.csv's raw last-w rates, the same combination
  win_all, win_10, streak  the site's win %, last-10 and the current streak, home less away
Strength: s_rating (features' rating_margin) or s_kal (the Kalman filter on points over the whole history).

    python models.py grid --part 0/2        # the tuning grid, a slice (both halves in parallel)
    python models.py tune                   # TUNE-only table, the choices
    python models.py final                  # reports, preds, coefficients
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from common import H, ML, OUT, STATS

warnings.filterwarnings("ignore")
WIN = ("2026-06-01", "2026-10-19")
TRAIN_FROM = "2026-05-04"
LATE = ("2026-09-22", "2026-10-19")
STAT_KEYS = list(STATS)
RAW_RATES = ["ppd", "td", "fg", "punt", "turnover", "downs", "rz", "rz_td", "plays", "start", "ypp", "success",
             "big20", "big40", "loss", "third", "drives", "pace"]
# the site's stats: points for/against, touchdowns, red zone, 3rd and 4th down, yards per play; per match
# first downs (~ snaps reaching the line to gain) and total offence; plus win %, form and the streak
SITE = ["pts", "tds_pm", "rz", "rz_td", "third", "fourth", "ypp", "fd_pm", "yds_pm"]
NONPBP = ["win_all", "win_10", "streak"]
FOUR = ["success", "big20", "ypp", "rz_td", "turnover", "third", "start"]
STRENGTH_SCALE = 5.0           # strength's z-score times this: its L2 penalty a 25th of a stat's


def load():
    df = H.load()
    k = pd.read_pickle(f"{OUT}/kalman_feats.pkl")
    df = df.merge(k, on="code", how="left")
    for hl in (15, 30, 60):
        e = pd.read_pickle(f"{OUT}/eb_feats_hl{hl}.pkl")
        e = e.rename(columns={c: c.replace("eb_", f"eb{hl}_") for c in e.columns if c.startswith("eb_")})
        if hl != 30:
            e = e.drop(columns=["streak_p1", "streak_p2"])
        df = df.merge(e, on="code", how="left")
    new = {}
    for s in STAT_KEYS:
        new[f"kal_{s}"] = df[f"k_{s}_pred_h"] - df[f"k_{s}_pred_a"]
        for hl in (15, 30, 60):
            p = f"eb{hl}_{s}"
            new[p] = df[f"{p}_offh"] + df[f"{p}_defa"] - df[f"{p}_offa"] - df[f"{p}_defh"]
    for w in (15, 50):
        for s in RAW_RATES:
            new[f"raw{w}_{s}"] = (df[f"p1_drv_off_{s}_{w}"] + df[f"p2_drv_def_{s}_{w}"]
                                  - df[f"p2_drv_off_{s}_{w}"] - df[f"p1_drv_def_{s}_{w}"])
        new[f"raw{w}_pts"] = df[f"p1_pf_{w}"] + df[f"p2_pa_{w}"] - df[f"p2_pf_{w}"] - df[f"p1_pa_{w}"]
    new["win_all"] = df.p1_win_all - df.p2_win_all
    new["win_10"] = df.p1_win_10 - df.p2_win_10
    new["streak"] = (df.streak_p1.clip(-6, 6) - df.streak_p2.clip(-6, 6))
    new["s_rating"] = df.rating_margin
    new["s_kal"] = df.k_ptsall_pred_h - df.k_ptsall_pred_a
    df = pd.concat([df, pd.DataFrame(new)], axis=1)
    return df


def cols_of(fs):
    """'kal:all' -> the columns."""
    src, sub = fs.split(":")
    if src.startswith("raw"):
        stats = {"all": RAW_RATES + ["pts"], "site": [s for s in SITE if s in RAW_RATES + ["pts"]],
                 "four": [s for s in FOUR if s in RAW_RATES]}[sub]
    else:
        stats = {"all": STAT_KEYS, "site": SITE, "four": FOUR, "nopts": [s for s in STAT_KEYS if s != "pts"]}.get(sub)
        if stats is None:
            stats = sub.split("+")
    cols = [f"{src}_{s}" for s in stats]
    if sub in ("all", "site"):
        cols += NONPBP
    return cols


class Spec:
    def __init__(self, name, feats=None, strength=None, kind="logit", C=1.0, penalty="l2", train_days=None,
                 alpha=10.0, refit_days=1):
        self.name, self.kind, self.C, self.penalty = name, kind, C, penalty
        self.train_days, self.alpha, self.refit_days = train_days, alpha, refit_days
        self.cols = (cols_of(feats) if isinstance(feats, str) else list(feats or []))
        self.strength = [strength] if isinstance(strength, str) else list(strength or [])

    def fit(self, tr):
        tr = tr[tr.day >= TRAIN_FROM]
        Xs = tr[self.cols].to_numpy(float) if self.cols else np.zeros((len(tr), 0))
        Xg = tr[self.strength].to_numpy(float) if self.strength else np.zeros((len(tr), 0))
        X = np.c_[Xs, Xg]
        X = np.where(np.isfinite(X), X, np.nan)
        mu = np.nanmean(X, 0)
        X = np.where(np.isnan(X), mu, X)
        sd = X.std(0) + 1e-12
        Z = (X - mu) / sd
        sc = np.r_[np.ones(Xs.shape[1]), np.full(Xg.shape[1], STRENGTH_SCALE)]
        Z = Z * sc
        y = tr.y.to_numpy()
        if self.kind == "logit":
            if self.penalty == "l1":
                m = LogisticRegression(C=self.C, penalty="l1", solver="liblinear", intercept_scaling=10.0, max_iter=500)
            else:
                m = LogisticRegression(C=self.C, max_iter=1000)
            m.fit(Z, y)
            return ("logit", mu, sd, sc, m)
        # stats -> margin (ridge), then a logistic link on the predicted margin
        r = Ridge(alpha=self.alpha).fit(Z, tr.margin.to_numpy())
        mh = r.predict(Z)
        lk = LogisticRegression(C=1e6, max_iter=1000).fit(mh[:, None], y)
        return ("mlink", mu, sd, sc, (r, lk))

    def predict(self, model, d):
        kind, mu, sd, sc, m = model
        X = np.c_[d[self.cols].to_numpy(float) if self.cols else np.zeros((len(d), 0)),
                  d[self.strength].to_numpy(float) if self.strength else np.zeros((len(d), 0))]
        X = np.where(np.isfinite(X), X, mu)
        Z = (X - mu) / sd * sc
        if kind == "logit":
            return m.predict_proba(Z)[:, 1]
        r, lk = m
        return lk.predict_proba(r.predict(Z)[:, None])[:, 1]

    def run(self, df, window=WIN):
        # a slim frame: the harness copies the training rows every day
        keep = list(dict.fromkeys(["code", "day", "start", "y", "margin"] + self.cols + self.strength))
        sub = df[keep]
        return H.walk_forward(sub, self.fit, self.predict, *window, refit_days=self.refit_days,
                              train_days=self.train_days)


def base_specs():
    specs = []
    Cs = (0.003, 0.01, 0.03, 0.1, 1.0)
    for s in (["s_rating"], ["s_kal"], ["s_rating", "s_kal"]):
        for td in (None, 60):
            t = "" if td is None else f"_td{td}"
            specs.append(Spec(f"str[{'+'.join(s)}]{t}", None, s, C=1.0, train_days=td))
    for fs in ("kal:all", "kal:site", "eb30:all", "raw15:all", "raw50:all"):
        for C in Cs:
            specs.append(Spec(f"stats[{fs}]_C{C}", fs, None, C=C))
            for st in ("s_rating", "s_kal"):
                specs.append(Spec(f"{st}+stats[{fs}]_C{C}", fs, st, C=C))
    for C in Cs:
        specs.append(Spec(f"s_kal+stats[kal:nopts]_C{C}", "kal:nopts", "s_kal", C=C))
        specs.append(Spec(f"stats[kal:all]_C{C}_td60", "kal:all", None, C=C, train_days=60))
        specs.append(Spec(f"s_kal+stats[kal:all]_C{C}_td60", "kal:all", "s_kal", C=C, train_days=60))
    for fs in ("eb15:all", "eb60:all"):
        for C in (0.01, 0.03):
            specs.append(Spec(f"s_kal+stats[{fs}]_C{C}", fs, "s_kal", C=C))
    for fs in ("kal:four", "eb30:four", "raw50:four"):
        for C in (0.01, 0.1, 1.0):
            specs.append(Spec(f"four[{fs}]_C{C}", fs, None, C=C))
            specs.append(Spec(f"s_kal+four[{fs}]_C{C}", fs, "s_kal", C=C))
    for fs in ("kal:all", "kal:site"):
        for al in (100.0, 1000.0, 1e4):
            specs.append(Spec(f"mlink[{fs}]_a{al:g}", fs, None, kind="mlink", alpha=al))
            specs.append(Spec(f"s_kal+mlink[{fs}]_a{al:g}", fs, "s_kal", kind="mlink", alpha=al))
    for C in (0.01, 0.03, 0.1):
        specs.append(Spec(f"stats[kal:all]_L1_C{C}", "kal:all", None, C=C, penalty="l1"))
        specs.append(Spec(f"s_kal+stats[kal:all]_L1_C{C}", "kal:all", "s_kal", C=C, penalty="l1"))
    return specs


def extra_specs():
    """Added after a first look at TUNE (the best C sat at the grid's low end; the best strength was both)."""
    specs = []
    both = ["s_rating", "s_kal"]
    for C in (0.0003, 0.001):
        for fs in ("kal:all", "kal:site"):
            specs.append(Spec(f"stats[{fs}]_C{C}", fs, None, C=C))
            specs.append(Spec(f"s_kal+stats[{fs}]_C{C}", fs, "s_kal", C=C))
    for C in (0.0003, 0.001, 0.003, 0.01, 0.03):
        for fs in ("kal:all", "kal:site"):
            specs.append(Spec(f"s_both+stats[{fs}]_C{C}", fs, both, C=C))
    for C in (0.01, 0.1, 1.0):
        specs.append(Spec(f"s_both+four[kal:four]_C{C}", "kal:four", both, C=C))
    for al in (1000.0, 1e4, 1e5):
        specs.append(Spec(f"s_both+mlink[kal:site]_a{al:g}", "kal:site", both, kind="mlink", alpha=al))
    return specs


def grid_specs():
    return base_specs() + extra_specs()


def cmd_grid(part, which="base"):
    i, n = map(int, part.split("/"))
    df = load()
    specs = (base_specs() if which == "base" else extra_specs())[i::n]
    path = f"{OUT}/wf_grid_{i}of{n}.pkl" if which == "base" else f"{OUT}/wf_grid_extra_{i}of{n}.pkl"
    res = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame({"code": df.code})
    t0 = time.time()
    for k, s in enumerate(specs):
        if s.name in res:
            continue
        res[s.name] = s.run(df).to_numpy()
        if k % 5 == 0:
            res.to_pickle(path)
            print(f"  {k + 1}/{len(specs)} {s.name} {time.time() - t0:.0f}s", flush=True)
    res.to_pickle(path)
    print("done", path, flush=True)


def tune_table(df, cols, window=H.TUNE):
    w = df[H.in_window(df, window) & df.y.notna()].dropna(subset=cols)
    rows = []
    for c in cols:
        m = H.metrics(w.y, w[c])
        rows.append(dict(model=c, logloss=m["logloss"], brier=m["brier"], auc=m["auc"], slope=m["cal_slope"]))
    return pd.DataFrame(rows).sort_values("logloss"), len(w)


def read_grid(df):
    for f in sorted(glob.glob(f"{OUT}/wf_grid_*.pkl")):
        g = pd.read_pickle(f)
        df = df.merge(g, on="code", how="left")
    return df


def cmd_tune():
    df = H.load(tables=False, elo=False)
    df = read_grid(df)
    names = [s.name for s in grid_specs() if s.name in df]
    # every model on TUNE's matches it prices (all from 06-01): common rows
    t, n = tune_table(df, names + ["prod"])
    t.to_csv(f"{OUT}/tune_grid.csv", index=False)
    print(f"TUNE {n} matches (common)")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(t.to_string(index=False, float_format="{:.4f}".format))


def addone_specs(strength):
    """strength alone, and strength plus each single stat (Kalman matchup) or site non-pbp stat."""
    lab = "+".join(strength) if isinstance(strength, list) else strength
    specs = [Spec(f"one[{lab}]", None, strength, C=1.0)]
    for s in STAT_KEYS:
        specs.append(Spec(f"one[{lab}]+kal_{s}", [f"kal_{s}"], strength, C=1.0))
    for s in NONPBP:
        specs.append(Spec(f"one[{lab}]+{s}", [s], strength, C=1.0))
    return specs


def cmd_addone(part, strength):
    i, n = map(int, part.split("/"))
    df = load()
    specs = addone_specs(strength)[i::n]
    path = f"{OUT}/wf_addone_{i}of{n}.pkl"
    res = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame({"code": df.code})
    for s in specs:
        if s.name not in res:
            res[s.name] = s.run(df).to_numpy()
            res.to_pickle(path)
            print("  ", s.name, flush=True)
    print("done", path, flush=True)


def choose(t, pred):
    """The TUNE-best model among names satisfying pred."""
    c = t[t.model.map(pred)]
    return c.iloc[0].model


def spec_by_name(name):
    for s in grid_specs():
        if s.name == name:
            return s
    raise KeyError(name)


def coef_table(df, spec, cut):
    tr = df[(df.start + H.RESULT_LAG <= pd.Timestamp(cut)) & df.y.notna()]
    kind, mu, sd, sc, m = spec.fit(tr)
    names = spec.cols + spec.strength
    if kind == "logit":
        b = m.coef_[0] * sc                     # per sd of the feature
        icpt = m.intercept_[0]
    else:
        r, lk = m
        b = r.coef_ * sc                        # points of margin per sd
        icpt = r.intercept_
    return pd.DataFrame({"feature": names, "coef_per_sd": b, "sd": sd, "coef_per_unit": b / sd}), icpt


def cmd_final():
    df = load()
    df = read_grid(df)
    for f in sorted(glob.glob(f"{OUT}/wf_addone_*.pkl")):
        df = df.merge(pd.read_pickle(f), on="code", how="left")
    names = [s.name for s in grid_specs() if s.name in df]
    t, n = tune_table(df, names)
    is_str = lambda m: m.startswith("str[")
    has_s = lambda m: m.startswith(("s_kal+", "s_rating+", "s_both+"))
    pick = {
        "C_strength": choose(t, is_str),
        "C_stats": choose(t, lambda m: m.startswith("stats[")),
        "C_stats_site": choose(t, lambda m: m.startswith("stats[kal:site]")),
        "C_stats_strength": choose(t, lambda m: has_s(m) and "+stats[" in m),
        "C_four_factors": choose(t, lambda m: m.startswith("four[")),
        "C_four_factors_strength": choose(t, lambda m: has_s(m) and "+four[" in m),
        "C_stats_mlink": choose(t, lambda m: m.startswith("mlink[")),
        "C_stats_strength_mlink": choose(t, lambda m: has_s(m) and "+mlink[" in m),
    }
    # the nested comparison's own strength: the same strength input and training window, alone
    sname = pick["C_stats_strength"].split("+")[0]
    sname = {"s_both": "s_rating+s_kal"}.get(sname, sname)
    td = "_td60" if pick["C_stats_strength"].endswith("_td60") else ""
    pick["C_strength_nested"] = f"str[{sname}]{td}"
    json.dump(pick, open(f"{OUT}/picks.json", "w"), indent=1)
    lines = ["Choices (TUNE log loss, walk-forward):"] + [f"  {k:26s} {v}" for k, v in pick.items()]
    for k, v in pick.items():
        df[k] = df[v]
    cols = list(pick)
    for w, lab in ((H.TUNE, "TUNE"), (H.TEST, "TEST"), (H.SEP, "SEP"), (LATE, "LATE 09-22..10-18")):
        lines += ["", f"== {lab}", H.report(df, cols, window=w, refs=("prod", "C_strength"))]
        lines += [H.report(df, ["C_stats_strength", "C_four_factors_strength", "C_stats_strength_mlink"],
                           window=w, refs=("C_strength_nested",))]
    # how the stats are built: raw windows vs EB-shrunk vs opponent-adjusted Kalman (each at its TUNE-best C)
    src = {}
    for fs in ("kal:all", "kal:site", "kal:nopts", "eb15:all", "eb30:all", "eb60:all", "raw15:all", "raw50:all"):
        for pre in ("stats", "s_kal+stats", "s_rating+stats", "s_both+stats"):
            cand = [m for m in t.model if m.startswith(f"{pre}[{fs}]_C") and not m.endswith("_td60") and "_L1_" not in m]
            if cand:
                best = choose(t, lambda m, c=set(cand): m in c)
                src[f"{pre}[{fs}]"] = best
                df[f"{pre}[{fs}]"] = df[best]
    lines += ["", "== Feature construction (each at its TUNE-best C): " + json.dumps(src)]
    nest = {"stats": None, "s_kal+stats": "str[s_kal]", "s_rating+stats": "str[s_rating]",
            "s_both+stats": "str[s_rating+s_kal]"}
    for pre, ref in nest.items():
        cs = [k for k in src if k.startswith(pre + "[")]
        refs = ("prod",) + ((ref,) if ref else ())
        for w, lab in ((H.TUNE, "TUNE"), (H.TEST, "TEST")):
            lines += [f"-- {pre}, {lab}", H.report(df, cs, window=w, refs=refs)]
    # add-one: strength + each stat
    one = [c for c in df.columns if c.startswith("one[")]
    if one:
        base = [c for c in one if c.endswith("]")][0]
        rows = []
        for c in one:
            if c == base:
                continue
            r = {"stat": c.split("]+", 1)[1]}
            for w, lab in ((H.TUNE, "tune"), (H.TEST, "test")):
                x = df[H.in_window(df, w)]
                mu, lo, hi = H.boot_diff(x, c, base)
                r[f"{lab}_d"], r[f"{lab}_lo"], r[f"{lab}_hi"] = mu, lo, hi
            rows.append(r)
        A = pd.DataFrame(rows).sort_values("tune_d")
        A.to_csv(f"{OUT}/addone.csv", index=False)
        lines += ["", f"== Add one stat to {base} (log loss less strength alone; 95% day-resampled)",
                  A.to_string(index=False, float_format="{:+.4f}".format)]
    # coefficients
    for k in ("C_stats", "C_stats_strength", "C_four_factors", "C_four_factors_strength", "C_stats_strength_mlink"):
        s = spec_by_name(pick[k])
        for cut in ("2026-08-01", "2026-10-19"):
            ct, icpt = coef_table(df, s, cut)
            lines += ["", f"== {k} = {s.name}: fitted on everything settled by {cut} (intercept {icpt:+.4f})",
                      ct.to_string(index=False, float_format="{:+.4g}".format)]
    txt = "\n".join(lines)
    open(f"{OUT}/final.txt", "w").write(txt + "\n")
    print(txt)
    for k in pick:
        H.write_preds(df, k, k)
    with pd.option_context("display.width", 200):
        print(t.head(40).to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--strength", default="s_kal")
    a = ap.parse_args()
    if a.cmd == "grid":
        cmd_grid(a.part)
    elif a.cmd == "extra":
        cmd_grid(a.part, "extra")
    elif a.cmd == "tune":
        cmd_tune()
    elif a.cmd == "addone":
        cmd_addone(a.part, a.strength.split(",") if "," in a.strength else a.strength)
    elif a.cmd == "final":
        cmd_final()
