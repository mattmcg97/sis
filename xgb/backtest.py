"""The trader deck's XGBoost pre-match model, for AF, walk-forward against glmer and v2.1's prior.

Like the deck's eBasketball model: three XGBoost regressors trained on a rolling TRAIN_DAYS window --
a side's points (both sides of every match, from the side's own view), the margin and the total --
blended as the deck does: margin = BLEND x (home points - away points) + (1 - BLEND) x the margin
model, and the same for the total. Each side's expected points are then (total +/- margin) / 2.

Retrained every --refit-days on the matches settled by then (RESULT_LAG after kick-off); the number
of trees is found by early stopping on the window's last VALID_DAYS, then the window is refitted
whole with that many. Two feature sets (xgb/features.py):
  scores   the results alone: rolling form, spreads, streaks, teams, rest, sessions, head to head
  stack    scores + the walk-forward glmer's expected points and the Kalman form rating -- the deck's
           "meta-predictions": XGBoost learns corrections to the models we have

With xgb/drives.py's play-by-play in the features, two more: drives (scores + each gamer's drive and
snap rates) and stack_drives (stack + the same).

Every method's expected margin is turned into a moneyline the same way (a normal, sd MARGIN_SD, ties
left out), so the comparison is of the expectations themselves. Baseline tables (--table name=path,
eAMFCalibrator's walkthrough or form-prior tables) are scored on the matches they cover. Where the
features carry prod's pre-match quotes, every method is also set against prod's lines as the trader
deck does: the total line and the spread line against the results (bias, MAE, within 5 and 10).

Each XGBoost variant's prices are written as a prior table (priors/xgb_<set>.csv: MATCH_CODE,
Pred_P1_Points, Pred_P2_Points, p_home), which v21 reads with --v21-prior-model.

    python -m xgb.backtest --features xgb/out/features.csv --since 2026-07-01 \
        --table v21=eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv
"""

import argparse
import datetime as dt
import math
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

RESULT_LAG = dt.timedelta(minutes=36)
TRAIN_DAYS = 180
VALID_DAYS = 14
BLEND = 0.3
MARGIN_SD = 9.35
PARAMS = dict(objective="reg:squarederror", eta=0.03, max_depth=4, min_child_weight=30, subsample=0.8,
              colsample_bytree=0.7, reg_lambda=5.0, nthread=4, verbosity=0)
MAX_ROUNDS = 3000
EARLY_STOP = 150

RATING = ["p1_rating", "p2_rating", "p1_rating_var", "p2_rating_var", "rating_margin"]
GLMER = ["glmer_p1", "glmer_p2", "glmer_margin", "glmer_total"]


def feature_sets(cols):
    ident = {"code", "start", "p1", "p2", "t1", "t2", "s1", "s2"}
    usable = [c for c in cols if c not in ident and not c.startswith("prod_")]
    drv = [c for c in usable if "_drv_" in c]
    scores = [c for c in usable if c not in RATING and c not in GLMER and c not in drv]
    stack = scores + [c for c in RATING + GLMER if c in cols]
    sets = {"scores": scores, "stack": stack}
    if drv:
        sets["drives"] = scores + drv
        sets["stack_drives"] = stack + drv
    return sets


def side_frame(d, feats):
    """Both sides of every match as rows from the side's own view: own_ / opp_ features, the match's
    oriented to the side, and is_home."""
    side_cols = sorted({c[3:] for c in feats if c.startswith("p1_")})
    match_cols = [c for c in feats if not c.startswith(("p1_", "p2_")) and not c.startswith("glmer_")]
    flip = {"h2h_margin", "rating_margin"} | {c for c in match_cols if c.startswith("exp_margin_")}
    out = []
    for own, opp, home in (("p1", "p2", 1), ("p2", "p1", 0)):
        x = pd.DataFrame({"is_home": home}, index=d.index)
        for c in side_cols:
            x[f"own_{c}"] = d[f"{own}_{c}"]
            x[f"opp_{c}"] = d[f"{opp}_{c}"]
        for c in match_cols:
            if c == "h2h_wins_5" and not home:
                x[c] = np.nan                      # the deck's count is PLAYER_1's; no away twin
            else:
                x[c] = -d[c] if (c in flip and not home) else d[c]
        if "glmer_p1" in feats:
            x["glmer_own"], x["glmer_opp"] = d[f"glmer_{own}"], d[f"glmer_{opp}"]
        x["y"] = d["s1"] if home else d["s2"]
        x["code"], x["home"] = d["code"], home
        out.append(x)
    return pd.concat(out, ignore_index=True)


def fit_one(x, y, x_valid, y_valid):
    """Early-stopped on the validation days, then refitted on everything with that many trees."""
    dtrain = xgb.DMatrix(x, label=y)
    dvalid = xgb.DMatrix(x_valid, label=y_valid)
    probe = xgb.train(PARAMS, dtrain, MAX_ROUNDS, evals=[(dvalid, "valid")], early_stopping_rounds=EARLY_STOP,
                      verbose_eval=False)
    rounds = max(50, probe.best_iteration + 1)
    full = xgb.train(PARAMS, xgb.DMatrix(pd.concat([x, x_valid]), label=np.concatenate([y, y_valid])), rounds)
    return full, rounds


def walk(d, feats, since, until, refit_days, log):
    """Expected points for every match from `since` to `until`, each from a model fitted before its day."""
    preds = []
    gains = {}
    cut = since
    while cut < until:
        nxt = min(until, cut + dt.timedelta(days=refit_days))
        test = d[(d.start >= cut) & (d.start < nxt) & d.s1.notna()]
        train = d[(d.start >= cut - dt.timedelta(days=TRAIN_DAYS)) & (d.start + RESULT_LAG <= cut) & d.s1.notna()]
        if not len(test):
            cut = nxt
            continue
        valid_from = cut - dt.timedelta(days=VALID_DAYS)
        tr, va = train[train.start < valid_from], train[train.start >= valid_from]
        t0 = time.time()
        # the side model, both sides of every match
        str_, sva, ste = side_frame(tr, feats), side_frame(va, feats), side_frame(test, feats)
        xcols = [c for c in str_.columns if c not in ("y", "code", "home")]
        side, r_side = fit_one(str_[xcols], str_.y, sva[xcols], sva.y)
        ste["mu"] = side.predict(xgb.DMatrix(ste[xcols]))
        # the margin and total models, one row a match
        mcols = list(feats)
        marg, r_m = fit_one(tr[mcols], tr.s1 - tr.s2, va[mcols], va.s1 - va.s2)
        tot, r_t = fit_one(tr[mcols], tr.s1 + tr.s2, va[mcols], va.s1 + va.s2)
        dm = xgb.DMatrix(test[mcols])
        p = pd.DataFrame({"code": test.code.values, "m_direct": marg.predict(dm), "t_direct": tot.predict(dm)})
        h = ste[ste.home == 1].set_index("code").mu
        a = ste[ste.home == 0].set_index("code").mu
        p["h_side"], p["a_side"] = h.reindex(p.code).values, a.reindex(p.code).values
        preds.append(p)
        for name, model in (("margin", marg), ("total", tot)):
            for k, v in model.get_score(importance_type="total_gain").items():
                gains[(name, k)] = gains.get((name, k), 0.0) + v
        log(f"  {cut:%Y-%m-%d}: trained on {len(train):,} matches, priced {len(test):,} "
            f"(trees: side {r_side}, margin {r_m}, total {r_t}; {time.time() - t0:.0f}s)")
        cut = nxt
    p = pd.concat(preds, ignore_index=True)
    p["margin"] = BLEND * (p.h_side - p.a_side) + (1 - BLEND) * p.m_direct
    p["total"] = BLEND * (p.h_side + p.a_side) + (1 - BLEND) * p.t_direct
    p["p1"], p["p2"] = (p.total + p.margin) / 2, (p.total - p.margin) / 2
    return p, gains


def read_table(path):
    """A walkthrough or form-prior table: code -> (home points, away points, its own p_home)."""
    t = pd.read_csv(path)
    return t.set_index("MATCH_CODE")[["Pred_P1_Points", "Pred_P2_Points", "p_home"]]


def norm_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))


def scores(margin_pred, total_pred, s1, s2, p_own=None):
    """Per-match losses: log loss and Brier (normal about the margin; ties out), squared and absolute errors."""
    margin, total = s1 - s2, s1 + s2
    p = np.clip(1 - norm_cdf(-margin_pred / MARGIN_SD), 1e-9, 1 - 1e-9)
    won = (margin > 0).astype(float)
    dec = margin != 0
    out = pd.DataFrame({
        "ll": np.where(dec, -(won * np.log(p) + (1 - won) * np.log(1 - p)), np.nan),
        "brier": np.where(dec, (p - won) ** 2, np.nan),
        "m_se": (margin - margin_pred) ** 2, "m_ae": np.abs(margin - margin_pred),
        "t_se": (total - total_pred) ** 2, "t_ae": np.abs(total - total_pred)})
    if p_own is not None:
        q = np.clip(p_own, 1e-9, 1 - 1e-9)
        out["ll_own"] = np.where(dec, -(won * np.log(q) + (1 - won) * np.log(1 - q)), np.nan)
    return out


def prod_lines(frame, methods):
    """The trader deck's view: prod's pre-match total and spread lines, and each method's, against the
    results -- bias (result less line), MAE, the share within 5 and 10 points -- on the matches prod
    quoted both before kick-off. Prod's spread line is read in whichever sign runs with its moneyline."""
    f = frame[frame.prod_total_line.notna() & frame.prod_spread_line.notna() & frame.prod_p_home.notna()]
    f = f[f[[f"{m}_margin" for m in methods]].notna().all(axis=1)]
    if len(f) < 50:
        return []
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(f.prod_spread_line, f.prod_p_home)[0, 1]
    sign = 1.0 if np.isfinite(corr) and corr > 0 else -1.0     # a home favourite's handicap is negative
    margin, total = f.s1 - f.s2, f.s1 + f.s2
    est = {"prod": (sign * f.prod_spread_line, f.prod_total_line)}
    for m in methods:
        est[m] = (f[f"{m}_margin"], f[f"{m}_total"])
    w = max(12, max(len(m) for m in est) + 2)
    lines = [f"  against prod's pre-match lines: {len(f):,} matches prod quoted before kick-off "
             f"(its spread line read as {'+' if sign > 0 else '-'}line = the home margin)", "",
             f"  {'':{w}s}{'total bias':>11s}{'MAE':>8s}{'<=5':>7s}{'<=10':>7s}{'spread bias':>13s}{'MAE':>8s}"
             f"{'<=5':>7s}{'<=10':>7s}{'ML ll':>8s}"]
    won = (margin > 0).astype(float)
    dec = margin != 0
    for name, (m_est, t_est) in est.items():
        te, me = total - t_est, margin - m_est
        if name == "prod":
            p = np.clip(f.prod_p_home.values, 1e-9, 1 - 1e-9)
        else:
            p = np.clip(1 - norm_cdf(-m_est.values / MARGIN_SD), 1e-9, 1 - 1e-9)
        ll = -(won * np.log(p) + (1 - won) * np.log(1 - p))[dec].mean()
        lines.append(f"  {name:{w}s}{te.mean():+11.2f}{te.abs().mean():8.3f}{(te.abs() <= 5).mean():7.1%}"
                     f"{(te.abs() <= 10).mean():7.1%}{me.mean():+13.2f}{me.abs().mean():8.3f}"
                     f"{(me.abs() <= 5).mean():7.1%}{(me.abs() <= 10).mean():7.1%}{ll:8.4f}")
    lines.append("  (prod's moneyline is its own probability; every model's, a normal about its margin)")
    return lines


def write_prior(frame, name, path):
    """A method's prices as a prior table v21 reads (--v21-prior-model)."""
    f = frame[frame[f"{name}_margin"].notna()]
    t = pd.DataFrame({"MATCH_CODE": f.code, "start": f.start.dt.strftime("%Y-%m-%d %H:%M:%S"), "home": f.p1,
                      "away": f.p2, "home_score": f.s1, "away_score": f.s2,
                      "Pred_P1_Points": ((f[f"{name}_total"] + f[f"{name}_margin"]) / 2).round(4),
                      "Pred_P2_Points": ((f[f"{name}_total"] - f[f"{name}_margin"]) / 2).round(4),
                      "exp_margin": f[f"{name}_margin"].round(4), "exp_total": f[f"{name}_total"].round(4),
                      "p_home": (1 - norm_cdf(-f[f"{name}_margin"].values / MARGIN_SD)).round(5)})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t.to_csv(path, index=False)
    return len(t)


def report(frame, methods, ref, title):
    """Lines: each method's scores on the matches every method priced, then each against `ref`."""
    keep = frame[[f"{m}_margin" for m in methods] + [f"{m}_total" for m in methods]].notna().all(axis=1)
    f = frame[keep]
    fortnight = ((f.start - f.start.min()).dt.days // 14).values
    sc = {m: scores(f[f"{m}_margin"].values, f[f"{m}_total"].values, f.s1.values, f.s2.values,
                    f[f"{m}_p_own"].values if f"{m}_p_own" in f else None) for m in methods}
    w = max(12, max(len(m) for m in methods) + 2)
    lines = [f"  {title}: {len(f):,} matches, {f.start.min():%Y-%m-%d} to {f.start.max():%Y-%m-%d}", "",
             f"  {'':{w}s}{'log loss':>10s}{'Brier':>9s}{'mrg RMSE':>10s}{'mrg MAE':>9s}{'tot RMSE':>10s}"
             f"{'tot MAE':>9s}{'own ll*':>9s}"]
    for m in methods:
        s = sc[m]
        own = f"{s.ll_own.mean():9.4f}" if "ll_own" in s else f"{'':9s}"
        lines.append(f"  {m:{w}s}{s.ll.mean():10.4f}{s.brier.mean():9.4f}{math.sqrt(s.m_se.mean()):10.3f}"
                     f"{s.m_ae.mean():9.3f}{math.sqrt(s.t_se.mean()):10.3f}{s.t_ae.mean():9.3f}{own}")
    lines += ["", f"  each less {ref}, match by match (t; fortnights better on log loss)",
              f"  {'':{w}s}{'log loss':>22s}{'Brier':>18s}{'margin sq err':>20s}{'total sq err':>20s}"]
    for m in methods:
        if m == ref:
            continue
        cells = []
        for k in ("ll", "brier", "m_se", "t_se"):
            dlt = (sc[m][k] - sc[ref][k]).values
            ok = ~np.isnan(dlt)
            t = dlt[ok].mean() / (dlt[ok].std(ddof=1) / math.sqrt(ok.sum()))
            if k == "ll":
                by = pd.Series(dlt[ok]).groupby(fortnight[ok]).mean()
                cells.append(f"{dlt[ok].mean():+9.5f} ({t:+4.1f}; {int((by < 0).sum())}/{len(by)})")
            else:
                cells.append(f"{dlt[ok].mean():+9.4f} ({t:+4.1f})" if k == "brier" else f"{dlt[ok].mean():+11.3f} ({t:+4.1f})")
        lines.append(f"  {m:{w}s}{cells[0]:>22s}{cells[1]:>18s}{cells[2]:>20s}{cells[3]:>20s}")
    lines.append("  (* own ll: a table's own moneyline, off its score model, where it has one)")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="xgb/out/features.csv")
    ap.add_argument("--since", default="2026-07-01")
    ap.add_argument("--until", default="", help="exclusive; default after the last match")
    ap.add_argument("--refit-days", type=int, default=7)
    ap.add_argument("--table", action="append", default=[], metavar="NAME=CSV",
                    help="a baseline table (eAMFCalibrator walkthrough / form-prior), scored where it has the matches")
    ap.add_argument("--out", default="xgb/out")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    d = pd.read_csv(a.features, parse_dates=["start"])
    since = pd.Timestamp(a.since)
    until = pd.Timestamp(a.until) if a.until else d.start.max() + pd.Timedelta(minutes=1)
    sets = feature_sets(d.columns)
    if "glmer_p1" not in d:
        log("  (no glmer in the features: the stack set has the rating only)")
    test = d[(d.start >= since) & (d.start < until) & d.s1.notna()].copy()
    keep = ["code", "start", "p1", "p2", "s1", "s2"] + [c for c in d.columns if c.startswith("prod_")]
    frame = test[keep].reset_index(drop=True)
    gains_all = {}
    for name, feats in sets.items():
        log(f"\n  XGBoost on the {name} features ({len(feats)}), refitted every {a.refit_days} days on {TRAIN_DAYS}:")
        p, gains = walk(d, feats, since.to_pydatetime(), until.to_pydatetime(), a.refit_days, log)
        p = p.set_index("code")
        frame[f"xgb_{name}_margin"] = p.margin.reindex(frame.code).values
        frame[f"xgb_{name}_total"] = p.total.reindex(frame.code).values
        frame[f"xgb_{name}_direct_margin"] = p.m_direct.reindex(frame.code).values
        frame[f"xgb_{name}_direct_total"] = p.t_direct.reindex(frame.code).values
        gains_all[name] = gains
    if "glmer_p1" in d:
        g = test.set_index("code")
        frame["glmer_margin"] = (g.glmer_p1 - g.glmer_p2).reindex(frame.code).values
        frame["glmer_total"] = (g.glmer_p1 + g.glmer_p2).reindex(frame.code).values
        frame["glmer+xgb_margin"] = (frame.glmer_margin + frame.xgb_scores_margin) / 2
        frame["glmer+xgb_total"] = (frame.glmer_total + frame.xgb_scores_total) / 2
    tables = []
    for spec in a.table:
        name, path = spec.split("=", 1)
        t = read_table(path)
        frame[f"{name}_margin"] = (t.Pred_P1_Points - t.Pred_P2_Points).reindex(frame.code).values
        frame[f"{name}_total"] = (t.Pred_P1_Points + t.Pred_P2_Points).reindex(frame.code).values
        frame[f"{name}_p_own"] = t.p_home.reindex(frame.code).values
        tables.append(name)

    base = [m for m in ("glmer",) if f"{m}_margin" in frame]
    xgbs = [f"xgb_{n}" for n in sets] + ["xgb_scores_direct", "xgb_stack_direct"]
    extra = ["glmer+xgb"] if "glmer+xgb_margin" in frame else []
    log("")
    lines_all = report(frame, base + xgbs + extra, base[0] if base else "xgb_scores", "the whole span")
    for s in lines_all:
        log(s)
    if tables:
        log("")
        for s in report(frame, base + [f"xgb_{n}" for n in sets] + extra + tables, tables[-1],
                        "where the tables have the matches"):
            log(s)
    if "prod_total_line" in frame:
        for group in ([base + [f"xgb_{n}" for n in sets] + extra]
                      + ([base + [f"xgb_{n}" for n in sets] + tables] if tables else [])):
            log("")
            for s in prod_lines(frame, group):
                log(s)
    for n in sets:
        k = write_prior(frame, f"xgb_{n}", os.path.join(a.out, "priors", f"xgb_{n}.csv"))
        log(f"  prior table: {a.out}/priors/xgb_{n}.csv ({k:,} matches; v21 --v21-prior-model)")
    for name, gains in gains_all.items():
        tot = sum(v for (m, _), v in gains.items() if m == "margin")
        top = sorted(((v, k) for (m, k), v in gains.items() if m == "margin"), reverse=True)[:15]
        log(f"\n  {name}: the margin model's top features (share of the gain)")
        for v, k in top:
            log(f"    {k:28s}{100 * v / tot:6.1f}%")
    frame.to_csv(os.path.join(a.out, "backtest_matches.csv"), index=False)
    with open(os.path.join(a.out, "backtest_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  -> {a.out}/backtest_report.txt and backtest_matches.csv")


if __name__ == "__main__":
    main()
