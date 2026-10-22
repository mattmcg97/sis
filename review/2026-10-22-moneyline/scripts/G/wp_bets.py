"""In-play win probability on the bet moments: a direct P(home wins) on the game state, walk-forward by
day, against prod's and v2.1's in-play moneyline at the same moments (out/G/moments_ml.csv, clean set).

    python scripts/G/wp_bets.py tune      -> out/G/wp_tune.csv, wp_tune.txt   (test days 10-04..10-10)
    python scripts/G/wp_bets.py tune2     -> out/G/wp_tune2.csv, wp_tune2.txt (the possession proxy, the pre-match source)
    python scripts/G/wp_bets.py report    -> out/G/wp_report.txt, wp_preds.csv (frozen settings, 10-11..10-18)
    python scripts/G/wp_bets.py broad     -> out/G/wp_broad.txt (the same on the broad set of moments)

Models (each refit every day on the states of matches settled by that day's midnight UTC, harness's rule):
  stern   logit p = a + b * lead / sqrt(s + k) + c * pre * sqrt(s) [+ d * pre]; s = share of regulation
          left, pre = the pre-match logit. Final margin as a Brownian motion: drift from the pre-match
          view, spread growing with the time left (Stern 1994). b, c > 0 in every fit (checked).
  xgb     binary:logistic on lead, lead / sqrt(s + k), pre, pre * sqrt(s), seconds left, period, down,
          moment; monotone + on the two lead terms and the two pre terms (both increase in the lead / the
          pre-match logit, so the whole is monotone in each); time and the rest free.
  xgb_on  the same trees boosted from stern's logit (base_margin): stern's shape plus the corrections.
The states trained on: every in-play moneyline and spread bet's state (states_train.csv, any moment, one per
match and message); scored on the clean moneyline moments only.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

ML = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ML)
import harness as H  # noqa: E402

OUT = os.path.join(ML, "out", "G")
TUNE_DAYS = ("2026-10-04", "2026-10-11")
REPORT_DAYS = ("2026-10-11", "2026-10-19")
REG = 960.0
PHASES = ("between plays", "play live", "timeout", "kick-off live", "kick-off to come", "conversion to come",
          "quarter break", "interruption", "score in, play not over")
NTHREAD = 2


BROAD_MOVED = ("nothing", "play started", "timeout", "price after the bet")   # no score, possession or kick between


def load(which="clean"):
    tr = pd.read_csv(os.path.join(OUT, "states_train.csv"))
    mo = pd.read_csv(os.path.join(OUT, "moments_ml.csv"))
    mo = mo[mo.clean].copy() if which == "clean" else mo[mo.broad & mo.moved.isin(BROAD_MOVED)].copy()
    for d in (tr, mo):
        d["start"] = pd.to_datetime(d.start)
        d["day"] = d.start.dt.strftime("%Y-%m-%d")
    tr = tr[tr.y.notna()].copy()
    tr["n_match"] = tr.groupby("match_code").match_code.transform("size")
    for d in (tr, mo):
        d["pre_v21f"] = d.pre_v21.fillna(d.pre_prod)       # v2.1's kick-off price, prod's where it has none
    return tr, mo.reset_index(drop=True)


def features(d, pre_col, k):
    s = np.clip(d.secs_left.to_numpy(float) / REG, 0, 1)
    pre = H.logit(d[pre_col].to_numpy(float))
    lead = d.lead.to_numpy(float)
    phase = d.moment.map({p: i for i, p in enumerate(PHASES)}).fillna(len(PHASES)).to_numpy(float)
    ls = d.last_scorer.to_numpy(float)
    since = d.since_score.to_numpy(float)
    return pd.DataFrame({"lead": lead, "lead_z": lead / np.sqrt(s + k), "pre": pre, "pre_s": pre * np.sqrt(s),
                         "secs_left": d.secs_left.to_numpy(float), "period": d.period.to_numpy(float),
                         "down": d.down.to_numpy(float), "phase": phase, "ls": ls,
                         "ls_recent": ls * np.exp(-np.nan_to_num(since, nan=240.0) / 60.0), "since": since})


BASE = ["lead", "lead_z", "pre", "pre_s", "secs_left", "period", "down", "phase"]
LS = ["ls", "ls_recent", "since"]


MONO = {"lead": 1, "lead_z": 1, "pre": 1, "pre_s": 1}


def stern_fit(X, y, w, free_pre, ls=False):
    cols = ["lead_z", "pre_s"] + (["pre"] if free_pre else []) + (["ls", "ls_recent"] if ls else [])
    m = LogisticRegression(C=1e6, max_iter=1000)
    m.fit(X[cols].to_numpy(), y, sample_weight=w)
    return m, cols


def stern_margin(model, X):
    m, cols = model
    return m.decision_function(X[cols].to_numpy())


def weights(tr, scheme):
    if scheme == "state":
        return np.ones(len(tr))
    if scheme == "match":
        return (1.0 / tr.n_match.to_numpy()) * len(tr) / tr.match_code.nunique()
    raise ValueError(scheme)


def xgb_fit(X, y, w, depth, mcw, rounds, base=None, eta=0.05, seed=0):
    cols = list(X.columns)
    mono = "(" + ",".join(str(MONO.get(c, 0)) for c in cols) + ")"
    dm = xgb.DMatrix(X, label=y, weight=w, base_margin=base)
    params = dict(objective="binary:logistic", eval_metric="logloss", max_depth=depth, eta=eta,
                  min_child_weight=mcw, subsample=0.8, colsample_bytree=1.0, reg_lambda=5.0,
                  monotone_constraints=mono, nthread=NTHREAD, seed=seed, tree_method="hist")
    return xgb.train(params, dm, num_boost_round=rounds)


def xgb_predict(bst, X, rounds, base=None):
    dm = xgb.DMatrix(X, base_margin=base)
    return bst.predict(dm, iteration_range=(0, rounds))


def train_mask(tr, day):
    cut = pd.Timestamp(day)
    return (tr.start + H.RESULT_LAG) <= cut


def walk(tr, mo, days, cfgs, pre_col="pre_A_best", verbose=False):
    """{name: Series of P(home) on mo's index} for each config over the test days; cfgs is a list of dicts
    (kind stern / xgb / xgb_on with its settings)."""
    out = {c["name"]: pd.Series(np.nan, index=mo.index) for c in cfgs}
    test_days = sorted(mo.loc[(mo.day >= days[0]) & (mo.day < days[1]), "day"].unique())
    for d in test_days:
        trd = tr[train_mask(tr, d)]
        te = mo.index[mo.day == d]
        ytr = trd.y.to_numpy()
        cache = {}
        for c in cfgs:
            k, scheme, ls = c["k"], c["w"], c.get("ls", False)
            pc = c.get("pre", pre_col)
            Xtr = features(trd, pc, k)
            Xte = features(mo.loc[te], pc, k)
            cols = BASE + (LS if ls else [])
            w = weights(trd, scheme)
            if c["kind"] in ("stern", "xgb_on"):
                key = ("stern", k, scheme, c.get("free_pre", True), ls, pc)
                if key not in cache:
                    cache[key] = stern_fit(Xtr, ytr, w, c.get("free_pre", True), ls)
                sm = cache[key]
                if c["kind"] == "stern":
                    out[c["name"]].loc[te] = H.sigmoid(stern_margin(sm, Xte))
                    continue
                btr, bte = stern_margin(sm, Xtr), stern_margin(sm, Xte)
            else:
                btr = bte = None
            key = (c["kind"], k, scheme, c["depth"], c["mcw"], c.get("free_pre", True), c.get("max_rounds", 400), ls, pc)
            if key not in cache:
                cache[key] = xgb_fit(Xtr[cols], ytr, w, c["depth"], c["mcw"], c.get("max_rounds", 400), base=btr)
            out[c["name"]].loc[te] = xgb_predict(cache[key], Xte[cols], c["rounds"], base=bte)
        if verbose:
            print(f"  {d}: trained on {len(trd):,} states of {trd.match_code.nunique():,} matches, "
                  f"priced {len(te):,}", flush=True)
    return out


def tune_grid():
    cfgs = []
    for k, free in itertools.product((0.005, 0.01, 0.02, 0.04, 0.08, 0.15), (False, True)):
        for w in ("state", "match"):
            cfgs.append(dict(kind="stern", k=k, w=w, free_pre=free, name=f"stern k{k} {'free' if free else 'tied'} {w}"))
    for kind, k, w, depth, mcw in itertools.product(("xgb", "xgb_on"), (0.02, 0.05), ("state", "match"),
                                                     (2, 3, 4), (20, 100)):
        rounds_set = (100, 200, 400) if kind == "xgb" else (25, 50, 100, 200)
        for r in rounds_set:
            cfgs.append(dict(kind=kind, k=k, w=w, depth=depth, mcw=mcw, rounds=r, free_pre=True,
                             max_rounds=max(rounds_set), name=f"{kind} k{k} {w} d{depth} m{mcw} r{r}"))
    return cfgs


def score_table(d, cols, refs=("prod_home", "v21_home"), n_boot=2000):
    """Text lines: metrics on the rows every column prices, with day- and match-resampled intervals."""
    d = d.dropna(subset=list(cols) + list(refs) + ["y"])
    lines = [f"  {len(d):,} moments, {d.match_code.nunique():,} matches, {d.day.nunique()} days"]
    hdr = f"  {'model':34s} {'logloss':>8s} {'brier':>7s} {'auc':>6s} {'slope':>6s} {'int':>6s}"
    for r in refs:
        hdr += f"  {'vs ' + r + ' (days 95%; matches 95%)':>52s}"
    lines.append(hdr)
    for c in list(dict.fromkeys(list(cols) + list(refs))):
        m = H.metrics(d.y, d[c])
        s = f"  {c:34s} {m['logloss']:8.4f} {m['brier']:7.4f} {m['auc']:6.3f} {m['cal_slope']:6.2f} {m['cal_int']:+6.2f}"
        for r in refs:
            if r == c:
                s += " " * 54
                continue
            mu, lo, hi = H.boot_diff(d, c, r, n_boot=n_boot)
            _, mlo, mhi = boot_match(d, c, r, n_boot=n_boot)
            s += f"  {mu:+.4f} ({lo:+.4f},{hi:+.4f}; {mlo:+.4f},{mhi:+.4f})"
        lines.append(s)
    return lines


def boot_match(d, a, b, n_boot=2000, seed=0):
    """As harness.boot_diff, resampling whole matches."""
    diff = H.logloss_each(d.y.to_numpy(), d[a].to_numpy()) - H.logloss_each(d.y.to_numpy(), d[b].to_numpy())
    g = pd.DataFrame({"m": d.match_code.to_numpy(), "x": diff}).groupby("m")["x"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    k = rng.integers(0, len(s), size=(n_boot, len(s)))
    draws = s[k].sum(1) / c[k].sum(1)
    return float(diff.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def tune():
    tr, mo = load()
    cfgs = tune_grid()
    print(f"tuning {len(cfgs)} configs on test days {TUNE_DAYS[0]}..{TUNE_DAYS[1]}", flush=True)
    preds = walk(tr, mo, TUNE_DAYS, cfgs, verbose=True)
    w = mo[(mo.day >= TUNE_DAYS[0]) & (mo.day < TUNE_DAYS[1])].copy()
    rows = []
    for c in cfgs:
        p = preds[c["name"]].loc[w.index]
        m = H.metrics(w.y, p)
        rows.append(dict(name=c["name"], kind=c["kind"], **{k: m[k] for k in ("logloss", "brier", "auc", "cal_slope")}))
    t = pd.DataFrame(rows).sort_values("logloss")
    for r in ("prod_home", "v21_home", "v21late_home"):
        m = H.metrics(w.y, w[r])
        t = pd.concat([t, pd.DataFrame([dict(name=r, kind="benchmark", **{k: m[k] for k in ("logloss", "brier", "auc", "cal_slope")})])])
    t.to_csv(os.path.join(OUT, "wp_tune.csv"), index=False)
    lines = [f"TUNE test days {TUNE_DAYS[0]}..{TUNE_DAYS[1]} (exclusive): {len(w):,} moments, {w.match_code.nunique():,} matches"]
    for kind in ("stern", "xgb", "xgb_on", "benchmark"):
        lines.append(f"\n  best {kind}:")
        for _, r in t[t.kind == kind].sort_values("logloss").head(8).iterrows():
            lines.append(f"    {r['name']:44s} {r.logloss:.4f} {r.brier:.4f} {r.auc:.3f} {r.cal_slope:.2f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "wp_tune.txt"), "w") as fh:
        fh.write(text + "\n")


def tune2_grid():
    """The second round, on the same TUNE days: the last-scorer proxy, the pre-match source, fewer trees."""
    cfgs = []
    for pre, ls in itertools.product(("pre_A_best", "pre_C_strength", "pre_prod", "pre_v21f"), (False, True)):
        for free in (False, True):
            cfgs.append(dict(kind="stern", k=0.02, w="state", free_pre=free, ls=ls, pre=pre,
                             name=f"stern {pre} {'free' if free else 'tied'}{' ls' if ls else ''}"))
    for ls in (False, True):
        for r in (25, 50, 100):
            cfgs.append(dict(kind="xgb", k=0.05, w="match", depth=2, mcw=20, rounds=r, max_rounds=100, ls=ls,
                             name=f"xgb k0.05 match d2 m20 r{r}{' ls' if ls else ''}"))
        for r in (10, 25, 50):
            cfgs.append(dict(kind="xgb_on", k=0.05, w="state", depth=2, mcw=100, rounds=r, max_rounds=50, ls=ls,
                             name=f"xgb_on k0.05 state d2 m100 r{r}{' ls' if ls else ''}"))
    return cfgs


def tune2():
    tr, mo = load()
    cfgs = tune2_grid()
    print(f"tuning round 2: {len(cfgs)} configs on test days {TUNE_DAYS[0]}..{TUNE_DAYS[1]}", flush=True)
    preds = walk(tr, mo, TUNE_DAYS, cfgs, verbose=True)
    w = mo[(mo.day >= TUNE_DAYS[0]) & (mo.day < TUNE_DAYS[1])].copy()
    rows = []
    for c in cfgs:
        m = H.metrics(w.y, preds[c["name"]].loc[w.index])
        rows.append(dict(name=c["name"], kind=c["kind"], **{k: m[k] for k in ("logloss", "brier", "auc", "cal_slope")}))
    t = pd.DataFrame(rows).sort_values("logloss")
    t.to_csv(os.path.join(OUT, "wp_tune2.csv"), index=False)
    lines = [f"TUNE round 2, test days {TUNE_DAYS[0]}..{TUNE_DAYS[1]} (exclusive): {len(w):,} moments"]
    for _, r in t.iterrows():
        lines.append(f"  {r['name']:44s} {r.logloss:.4f} {r.brier:.4f} {r.auc:.3f} {r.cal_slope:.2f}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "wp_tune2.txt"), "w") as fh:
        fh.write(text + "\n")


# Frozen on TUNE (wp_tune.txt, wp_tune2.txt): k is flat over 0.005-0.04; the last-scorer proxy and a free pre-match
# term did not help there; A_best was the best pre-match input. The pre-match variants are for the decomposition.
FROZEN = [
    dict(kind="stern", k=0.02, w="state", free_pre=False, name="stern"),
    dict(kind="xgb", k=0.05, w="match", depth=2, mcw=20, rounds=50, max_rounds=50, name="xgb"),
    dict(kind="xgb_on", k=0.05, w="state", depth=2, mcw=100, rounds=10, max_rounds=10, name="xgb_on_stern"),
    dict(kind="stern", k=0.02, w="state", free_pre=False, pre="pre_prod", name="stern_preprod"),
    dict(kind="stern", k=0.02, w="state", free_pre=False, pre="pre_v21f", name="stern_prev21"),
    dict(kind="stern", k=0.02, w="state", free_pre=False, ls=True, name="stern_ls"),
]


def stack(mo, days, a, b, start=TUNE_DAYS[0]):
    """Walk-forward logistic of y on logit a and logit b: each day fitted on the clean moments of the days
    from `start` before it. Returns (P, the last fit's coefficients)."""
    out = pd.Series(np.nan, index=mo.index)
    coef = None
    for d in sorted(mo.loc[(mo.day >= days[0]) & (mo.day < days[1]), "day"].unique()):
        tr = mo[(mo.day >= start) & (mo.start + H.RESULT_LAG <= pd.Timestamp(d))].dropna(subset=[a, b, "y"])
        te = mo.index[(mo.day == d) & mo[a].notna() & mo[b].notna()]
        Z = np.column_stack([H.logit(tr[a]), H.logit(tr[b])])
        m = LogisticRegression(C=1e6, max_iter=1000).fit(Z, tr.y)
        out.loc[te] = m.predict_proba(np.column_stack([H.logit(mo.loc[te, a]), H.logit(mo.loc[te, b])]))[:, 1]
        coef = (float(m.intercept_[0]), *m.coef_[0].round(3))
    return out, coef


def breakdown(d, by, cols):
    lines = [f"  {'':16s} {'moments':>8s} " + " ".join(f"{c[:12]:>12s}" for c in cols)]
    for g, x in d.groupby(by, observed=True):
        lines.append(f"  {str(g):16s} {len(x):8,d} " + " ".join(
            f"{H.logloss_each(x.y.to_numpy(), x[c].to_numpy()).mean():12.4f}" for c in cols))
    return lines


def report():
    tr, mo = load()
    days = (TUNE_DAYS[0], REPORT_DAYS[1])
    preds = walk(tr, mo, days, FROZEN, verbose=True)
    for c in FROZEN:
        mo[c["name"]] = preds[c["name"]]
    mo["pre_only"] = mo.pre_A_best
    names = [c["name"] for c in FROZEN]
    lines = ["IN-PLAY WIN PROBABILITY ON THE BET MOMENTS (moneyline, clean set: Q1-Q4, the feed unmoved between "
             "prod's price and the bet, v2.1 on the same state; one row per match and prod message)",
             "frozen on TUNE: " + "; ".join(f"{c['name']} = {c['kind']} k{c['k']} {c['w']}"
                                            + (f" d{c['depth']} mcw{c['mcw']} r{c['rounds']}" if 'depth' in c else '')
                                            + (f" pre={c['pre']}" if 'pre' in c else ' pre=A_best')
                                            + (' +last scorer' if c.get('ls') else '') for c in FROZEN)]
    for title, (lo, hi) in (("REPORT", REPORT_DAYS), ("TUNE (settings chosen here; for reference)", TUNE_DAYS)):
        w = mo[(mo.day >= lo) & (mo.day < hi)].copy()
        lines.append(f"\n==== {title}: match days {lo}..{hi} (exclusive) ====")
        lines += score_table(w, names + ["pre_only", "v21late_home"], refs=("prod_home", "v21_home"))
    w = mo[(mo.day >= REPORT_DAYS[0]) & (mo.day < REPORT_DAYS[1])].copy()
    cols = ["prod_home", "v21_home", "stern", "xgb", "stern_preprod"]
    lines.append("\n  REPORT by quarter (log loss)")
    lines += breakdown(w, "period", cols)
    w["lead_band"] = pd.cut(w.lead.abs(), [-1, 0, 3, 8, 99], labels=["level", "1-3", "4-8", "9+"])
    lines.append("\n  REPORT by |home lead| (log loss)")
    lines += breakdown(w, "lead_band", cols)
    w["ls_band"] = w.last_scorer.map({1.0: "home scored last", -1.0: "away scored last", 0.0: "none / new half"})
    lines.append("\n  REPORT by the last scorer (a possession proxy; log loss)")
    lines += breakdown(w, "ls_band", cols)
    for c in ("prod_home", "v21_home", "stern", "xgb"):
        x = w.dropna(subset=[c])
        edges = np.quantile(x[c], np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(edges, x[c], side="right") - 1, 0, 9)
        g = x.groupby(b).agg(p=(c, "mean"), y=("y", "mean"))
        lines.append(f"  calibration {c:10s} by decile: " + "; ".join(f"{r.p:.2f}->{r.y:.2f}" for r in g.itertuples()))
    # does each carry what the others lack? walk-forward stacks on the clean moments from TUNE's start
    lines.append("\n  REPORT stacks (walk-forward logistic on the two logits, fitted on the clean moments from "
                 f"{TUNE_DAYS[0]} settled before each day; last fit's intercept, weights)")
    for a, b in (("prod_home", "v21_home"), ("prod_home", "stern"), ("v21_home", "stern"), ("prod_home", "xgb")):
        mo[f"stack_{a}_{b}"], coef = stack(mo, REPORT_DAYS, a, b)
        w[f"stack_{a}_{b}"] = mo[f"stack_{a}_{b}"]
        lines.append(f"    {a} + {b}: weights {coef}")
    st_cols = [c for c in w if c.startswith("stack_")]
    lines += score_table(w, st_cols, refs=("prod_home", "v21_home"))
    # the broad set: every moment with the state read at the bet, the feed moved or not
    mo.to_csv(os.path.join(OUT, "wp_preds.csv"), index=False)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "wp_report.txt"), "w") as fh:
        fh.write(text + "\n")


def broad():
    """Robustness: the frozen stern and xgb on the broad set (the state read at the bet, the feed moved by a
    play start or a timeout at most since prod's price), REPORT days."""
    tr, mo = load("broad")
    cfgs = [c for c in FROZEN if c["name"] in ("stern", "xgb")]
    preds = walk(tr, mo, REPORT_DAYS, cfgs, verbose=True)
    for c in cfgs:
        mo[c["name"]] = preds[c["name"]]
    w = mo[(mo.day >= REPORT_DAYS[0]) & (mo.day < REPORT_DAYS[1])]
    lines = [f"BROAD SET (moved in {BROAD_MOVED}), REPORT {REPORT_DAYS[0]}..{REPORT_DAYS[1]}"]
    lines += score_table(w, ["stern", "xgb"], refs=("prod_home", "v21_home"))
    lines.append("  by what moved (log loss)")
    lines += breakdown(w, "moved", ["prod_home", "v21_home", "stern", "xgb"])
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUT, "wp_broad.txt"), "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    {"tune": tune, "tune2": tune2, "report": report, "broad": broad}[sys.argv[1]]()
