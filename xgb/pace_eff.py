"""A pre-match model whose outputs are pace and efficiency, not just expected points.

    python -m xgb.pace_eff --features xgb/out/features.csv --drives xgb/data/drive_stats.csv \
        --since 2026-07-01 --score-from 2026-08-01

For every match, priced off the results before it (walk-forward, refitted every --refit-days on a
rolling TRAIN_DAYS, rows weighted 2^(-days ago / 60) as glmer's):

  pace         the match's expected drives: a ridge regression of each match's drives on each
               gamer's and each team's pace (the two gamers' and teams' effects added)
  efficiency   each side's touchdowns and field goals a drive: Poisson regressions with the side's
               drives as exposure, on the side's attack and the opponent's defence (each gamer's own,
               ridge-shrunk), both teams, home, and glmer's form and experience covariates
  points       a side's expected points = its drives (half the match's) x (7 x touchdowns a drive +
               3 x field goals a drive)

Fitted the same way on points alone (a Poisson regression of a side's points on the same terms, no
pace), the points-only model is the like-for-like baseline; glmer (the walkthrough's) is the one we
run.

The spread of each result: a side's points are a sum over its drives of 7, 3 or 0, so given pace and
efficiency each side's variance is drives x (49 p_td + 9 p_fg - (7 p_td + 3 p_fg)^2) and the margin's
and total's variances follow (with a covariance through the shared drives). Each is scaled to the
results by one factor fitted on the calibration span (--since to --score-from), as is every
constant-sd model's sd. Scored from --score-from: the moneyline (a normal about the expected margin,
ties out), the margin's and total's errors and RPS (the normal on whole points), and against prod's
pre-match lines.

Touchdowns per side: the drive table's field goals as the feed saw them, and the rest of the final
score read as touchdowns (7 each; a remainder of 5-6 is one more, 3-4 one more field goal), since
tables made before xgb/drives.py read scores off the scoreboard undercount touchdowns.
"""

import argparse
import datetime as dt
import math
import os

import numpy as np
import pandas as pd
from scipy import sparse

RESULT_LAG = dt.timedelta(minutes=36)
TRAIN_DAYS = 180
HALF_LIFE = 60.0
SCALAR = 0.5                 # the points model's: data counted at half, as the trimmed glmer (points run
                             # far wider than Poisson)
EFF_SCALAR = 1.0             # touchdowns and field goals a side are about Poisson: counted whole
EFF_SD = 0.15                # the efficiency models' gamer effects (log scale); --eff-sd, chosen on calibration
EXP_FLOOR = 30
LAMBDA = {"gamer": 1 / 0.15 ** 2, "team": 1 / 0.04 ** 2, "fixed": 1e-4}
PACE_LAMBDA = {"gamer": 45.0, "team": 180.0, "fixed": 1e-4}
TD_POINTS, FG_POINTS = 7.0, 3.0


def td_fg(final, fg_seen):
    """(touchdowns, field goals) a side scored: the field goals seen and the rest as touchdowns."""
    if final is None or np.isnan(final):
        return np.nan, np.nan
    fg = fg_seen if not np.isnan(fg_seen) else 0.0
    rest = max(final - FG_POINTS * fg, 0.0)
    td, r = divmod(rest, TD_POINTS)
    if r >= 5:
        td += 1
    elif r >= 3:
        fg += 1
    return td, fg


def load(features, drives):
    """One row per settled match: identity, scores, form covariates and its drives, touchdowns and field goals."""
    f = pd.read_csv(features, parse_dates=["start"])
    keep = ["code", "start", "p1", "p2", "t1", "t2", "s1", "s2", "p1_ewma_pf", "p1_ewma_pa", "p2_ewma_pf",
            "p2_ewma_pa", "p1_games", "p2_games", "glmer_p1", "glmer_p2"] + \
           [c for c in f.columns if c.startswith("prod_")]
    f = f[[c for c in keep if c in f.columns]]
    d = pd.read_csv(drives)[["match_code", "h_drives", "a_drives", "h_fg", "a_fg"]]
    m = f.merge(d, left_on="code", right_on="match_code", how="left").drop(columns="match_code")
    m = m[m.s1.notna()].reset_index(drop=True)
    for side, s in (("h", "s1"), ("a", "s2")):
        tf = [td_fg(x, y) for x, y in zip(m[s].astype(float), m[f"{side}_fg"].astype(float))]
        m[f"{side}_td"] = [t for t, _ in tf]
        m[f"{side}_fgx"] = [g for _, g in tf]
    m["drives"] = m.h_drives + m.a_drives
    return m


def weights(start, cut):
    age = (cut - start).dt.total_seconds() / 86400.0
    w = np.power(2.0, -age / HALF_LIFE)
    return SCALAR * w / w.mean()


class Design:
    """Sparse design for one side's row: fixed covariates, then one-hot attack, defence and teams."""

    def __init__(self, train_long):
        self.gamers = {g: i for i, g in enumerate(sorted(set(train_long.off) | set(train_long.dfn)))}
        self.teams = {t: i for i, t in enumerate(sorted(set(train_long.off_team) | set(train_long.dfn_team)))}
        self.fixed = ["const", "home", "form_for", "form_against", "opp_form_for", "opp_form_against",
                      "exp_own", "exp_opp"]
        g, t = len(self.gamers), len(self.teams)
        self.blocks = [("fixed", len(self.fixed)), ("att", g), ("def", g), ("team_off", t), ("team_def", t)]
        self.width = sum(n for _, n in self.blocks)
        self.penalty = np.concatenate([np.full(n, LAMBDA["fixed" if b == "fixed" else
                                                          "gamer" if b in ("att", "def") else "team"])
                                       for b, n in self.blocks])

    def matrix(self, long):
        n = len(long)
        fx = np.column_stack([long[c].values if c != "const" else np.ones(n) for c in self.fixed])
        cols, rows, vals = [], [], []
        off = len(self.fixed)
        g, t = len(self.gamers), len(self.teams)
        for key, base, lookup in (("off", off, self.gamers), ("dfn", off + g, self.gamers),
                                  ("off_team", off + 2 * g, self.teams), ("dfn_team", off + 2 * g + t, self.teams)):
            idx = long[key].map(lookup)
            ok = idx.notna().values
            rows.append(np.flatnonzero(ok))
            cols.append(base + idx[ok].astype(int).values)
            vals.append(np.ones(ok.sum()))
        onehot = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                   shape=(n, self.width))
        dense = sparse.csr_matrix((fx.ravel(), (np.repeat(np.arange(n), len(self.fixed)),
                                                np.tile(np.arange(len(self.fixed)), n))), shape=(n, self.width))
        return (onehot + dense).tocsr()


def poisson_ridge(X, y, w, offset, penalty, iters=30):
    """A ridge-penalised Poisson regression (Newton steps): the coefficients."""
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(np.average(y / np.exp(offset), weights=w), 1e-6))
    for _ in range(iters):
        mu = np.exp(np.clip(X @ beta + offset, -20, 20))
        grad = X.T @ (w * (y - mu)) - penalty * beta
        H = (X.T @ sparse.diags(w * mu) @ X).toarray() + np.diag(penalty)
        step = np.linalg.solve(H, grad)
        beta += step
        if np.abs(step).max() < 1e-7:
            break
    return beta


def long_frame(m, league):
    """Both sides of every match from the scoring side's view."""
    out = []
    for side, own, opp, home in (("h", "p1", "p2", 1.0), ("a", "p2", "p1", 0.0)):
        lf = lambda x: np.log((x.fillna(league) + 1.0) / (league + 1.0))
        out.append(pd.DataFrame({
            "code": m.code, "side": side, "off": m[own], "dfn": m[opp], "off_team": m["t1" if home else "t2"],
            "dfn_team": m["t2" if home else "t1"], "home": home,
            "form_for": lf(m[f"{own}_ewma_pf"]), "form_against": lf(m[f"{own}_ewma_pa"]),
            "opp_form_for": lf(m[f"{opp}_ewma_pf"]), "opp_form_against": lf(m[f"{opp}_ewma_pa"]),
            "exp_own": np.log1p(np.maximum(m[f"{own}_games"].fillna(0), EXP_FLOOR)),
            "exp_opp": np.log1p(np.maximum(m[f"{opp}_games"].fillna(0), EXP_FLOOR)),
            "points": m["s1" if home else "s2"], "drives": m[f"{side}_drives"], "td": m[f"{side}_td"],
            "fg": m[f"{side}_fgx"], "start": m.start}))
    return pd.concat(out, ignore_index=True)


def pace_fit(train, w):
    """Ridge regression of a match's drives on the two gamers' and teams' pace: (lookups, coefficients)."""
    gamers = {g: i for i, g in enumerate(sorted(set(train.p1) | set(train.p2)))}
    teams = {t: i for i, t in enumerate(sorted(set(train.t1) | set(train.t2)))}
    width = 1 + len(gamers) + len(teams)

    def matrix(df):
        n = len(df)
        rows, cols = [np.arange(n)], [np.zeros(n, int)]
        for key, base, lookup in (("p1", 1, gamers), ("p2", 1, gamers), ("t1", 1 + len(gamers), teams),
                                  ("t2", 1 + len(gamers), teams)):
            idx = df[key].map(lookup)
            ok = idx.notna().values
            rows.append(np.flatnonzero(ok))
            cols.append(base + idx[ok].astype(int).values)
        r, c = np.concatenate(rows), np.concatenate(cols)
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, width))

    X = matrix(train)
    pen = np.concatenate([[PACE_LAMBDA["fixed"]], np.full(len(gamers), PACE_LAMBDA["gamer"]),
                          np.full(len(teams), PACE_LAMBDA["team"])])
    A = (X.T @ sparse.diags(w) @ X).toarray() + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ (w * train.drives.values))
    return matrix, beta


def walk(m, since, until, refit_days, log, eff_scalar=EFF_SCALAR, eff_sd=EFF_SD):
    """Every match from `since` priced by a fit before its week: pace, efficiency and points-only outputs."""
    out = []
    cut = since
    while cut < until:
        nxt = min(until, cut + dt.timedelta(days=refit_days))
        test = m[(m.start >= cut) & (m.start < nxt)]
        train = m[(m.start >= cut - dt.timedelta(days=TRAIN_DAYS)) & (m.start + RESULT_LAG <= cut)]
        if not len(test):
            cut = nxt
            continue
        league = float(pd.concat([train.s1, train.s2]).mean())
        lt, le = long_frame(train, league), long_frame(test, league)
        design = Design(lt)
        Xt, Xe = design.matrix(lt), design.matrix(le)
        w = weights(lt.start, pd.Timestamp(cut)).values
        # points alone
        b_pts = poisson_ridge(Xt, lt.points.values, w, np.zeros(len(lt)), design.penalty)
        # efficiency: touchdowns and field goals a drive, drives as exposure
        has = lt.drives.notna() & (lt.drives > 0) & lt.td.notna()
        Xd, wd, off = Xt[has.values], w[has.values], np.log(lt.drives[has].values)
        we = wd * eff_scalar / SCALAR
        pen_e = design.penalty.copy()
        g0, g1 = len(design.fixed), len(design.fixed) + 2 * len(design.gamers)
        pen_e[g0:g1] = 1 / eff_sd ** 2
        b_td = poisson_ridge(Xd, lt.td[has].values, we, off, pen_e)
        b_fg = poisson_ridge(Xd, lt.fg[has].values, we, off, pen_e)
        # points alone on just the matches the efficiency models see: what the data alone costs
        b_pts_d = poisson_ridge(Xd, lt.points[has].values, wd, np.zeros(int(has.sum())), design.penalty)
        # pace: the match's drives
        tm = train[train.drives.notna()]
        matrix, b_pace = pace_fit(tm, weights(tm.start, pd.Timestamp(cut)).values / SCALAR)
        pace = matrix(test) @ b_pace
        e = pd.DataFrame({"code": le.code.values, "side": le.side.values,
                          "pts": np.exp(Xe @ b_pts), "ptsd": np.exp(Xe @ b_pts_d), "p_td": np.exp(Xe @ b_td),
                          "p_fg": np.exp(Xe @ b_fg)})
        e = e.pivot(index="code", columns="side")
        e.columns = [f"{a}_{b}" for a, b in e.columns]
        e["pace"] = pd.Series(pace, index=test.code.values).reindex(e.index)
        out.append(e)
        log(f"  {cut:%Y-%m-%d}: trained on {len(train):,} matches ({int(has.sum() / 2):,} with drives), "
            f"priced {len(test):,}; league {league:.1f} a side, {tm.drives.mean():.2f} drives a match")
        cut = nxt
    e = pd.concat(out)
    for s in ("h", "a"):
        e[f"p_td_{s}"] = e[f"p_td_{s}"].clip(1e-4, 0.95)
        e[f"p_fg_{s}"] = np.minimum(e[f"p_fg_{s}"], 0.98 - e[f"p_td_{s}"]).clip(1e-4)
        e[f"ppd_{s}"] = TD_POINTS * e[f"p_td_{s}"] + FG_POINTS * e[f"p_fg_{s}"]
        e[f"eff_pts_{s}"] = e.pace / 2 * e[f"ppd_{s}"]
        # a side's points over its drives: each drive 7, 3 or 0
        e[f"var_{s}"] = e.pace / 2 * (TD_POINTS ** 2 * e[f"p_td_{s}"] + FG_POINTS ** 2 * e[f"p_fg_{s}"] - e[f"ppd_{s}"] ** 2)
    return e


def norm_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x, float) / math.sqrt(2)))


def rps(mean, sd, actual, lo, hi):
    """The ranked probability score of a normal on whole points from lo to hi."""
    k = np.arange(lo, hi + 1)
    cdf = norm_cdf((k[None, :] + 0.5 - mean[:, None]) / sd[:, None])
    step = (k[None, :] >= actual[:, None]).astype(float)
    return ((cdf - step) ** 2).sum(axis=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="xgb/out/features.csv")
    ap.add_argument("--drives", default="xgb/data/drive_stats.csv")
    ap.add_argument("--since", default="2026-07-01", help="the walk starts here; until --score-from it calibrates")
    ap.add_argument("--score-from", default="2026-08-01")
    ap.add_argument("--refit-days", type=int, default=7)
    ap.add_argument("--eff-scalar", type=float, default=EFF_SCALAR, help="the efficiency models' data weight")
    ap.add_argument("--eff-sd", default="0.15,0.25,0.35",
                    help="gamer-effect sds to try for the efficiency models; the best on the calibration "
                         "span's margin is kept")
    ap.add_argument("--out", default="xgb/out/pace_eff")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    m = load(a.features, a.drives)
    since, score_from = pd.Timestamp(a.since), pd.Timestamp(a.score_from)
    until = m.start.max() + pd.Timedelta(minutes=1)
    log(f"  pace and efficiency, walk-forward from {since:%Y-%m-%d}, scored from {score_from:%Y-%m-%d}:")
    sds = [float(x) for x in str(a.eff_sd).split(",")]
    if len(sds) > 1:
        cal_until = min(until, score_from).to_pydatetime()
        best = None
        for sd_ in sds:
            ec = walk(m, since.to_pydatetime(), cal_until, a.refit_days, lambda s: None, a.eff_scalar, sd_)
            tc = m.set_index("code").join(ec, how="inner")
            err = float((((tc.s1 - tc.s2) - (tc.eff_pts_h - tc.eff_pts_a)) ** 2).mean())
            log(f"  efficiency gamer sd {sd_:.2f}: calibration margin RMSE {math.sqrt(err):.3f}")
            if best is None or err < best[0]:
                best = (err, sd_)
        sds = [best[1]]
        log(f"  kept {best[1]:.2f}")
    e = walk(m, since.to_pydatetime(), until.to_pydatetime(), a.refit_days, log, a.eff_scalar, sds[0])
    t = m.set_index("code").join(e, how="inner")
    t = t[t.glmer_p1.notna()].copy()
    t["margin"], t["total"] = t.s1 - t.s2, t.s1 + t.s2
    t["drives_real"] = t.drives
    methods = {
        "glmer": (t.glmer_p1 - t.glmer_p2, t.glmer_p1 + t.glmer_p2, None),
        "points only": (t.pts_h - t.pts_a, t.pts_h + t.pts_a, None),
        "points, sd by pace": (t.pts_h - t.pts_a, t.pts_h + t.pts_a, "pace"),
        "points only, drive matches": (t.ptsd_h - t.ptsd_a, t.ptsd_h + t.ptsd_a, None),
        "glmer, sd by pace": (t.glmer_p1 - t.glmer_p2, t.glmer_p1 + t.glmer_p2, "pace"),
        "glmer, sd by pace x efficiency": (t.glmer_p1 - t.glmer_p2, t.glmer_p1 + t.glmer_p2, "own"),
        "pace x efficiency": (t.eff_pts_h - t.eff_pts_a, t.eff_pts_h + t.eff_pts_a, None),
        "pace x efficiency, own spread": (t.eff_pts_h - t.eff_pts_a, t.eff_pts_h + t.eff_pts_a, "own"),
    }
    cal = t.start < score_from
    sc = ~cal
    # each method's spread: one sd, or the shape its outputs give, scaled once on the calibration span
    shape_m = {"pace": np.sqrt(t.pace / t.pace[cal].mean()), "own": np.sqrt(t.var_h + t.var_a)}
    shape_t = {"pace": np.sqrt(t.pace / t.pace[cal].mean()), "own": np.sqrt(t.var_h + t.var_a)}
    res = {}
    for name, (mg, tt, shape) in methods.items():
        if shape is None:
            sm = pd.Series(np.sqrt(((t.margin - mg)[cal] ** 2).mean()), index=t.index)
            st = pd.Series(np.sqrt(((t.total - tt)[cal] ** 2).mean()), index=t.index)
        else:
            km = np.sqrt(((t.margin - mg)[cal] ** 2).mean() / (shape_m[shape][cal] ** 2).mean())
            kt = np.sqrt(((t.total - tt)[cal] ** 2).mean() / (shape_t[shape][cal] ** 2).mean())
            sm, st = km * shape_m[shape], kt * shape_t[shape]
        res[name] = (mg, tt, sm, st)
    f = t[sc]
    dec = (f.margin != 0).values
    won = (f.margin > 0).astype(float).values
    log()
    log(f"  scored: {len(f):,} matches, {f.start.min():%Y-%m-%d} to {f.start.max():%Y-%m-%d} "
        f"(calibrated on {int(cal.sum()):,} before)")
    log(f"  {'':32s}{'log loss':>10s}{'Brier':>9s}{'mrg RMSE':>10s}{'tot RMSE':>10s}{'mrg RPS':>9s}{'tot RPS':>9s}"
        f"{'mrg sd':>8s}")
    per = {}
    for name, (mg, tt, sm, st) in res.items():
        mg, tt, sm, st = mg[sc].values, tt[sc].values, sm[sc].values, st[sc].values
        p = np.clip(1 - norm_cdf(-mg / sm), 1e-9, 1 - 1e-9)
        ll = np.where(dec, -(won * np.log(p) + (1 - won) * np.log(1 - p)), np.nan)
        br = np.where(dec, (p - won) ** 2, np.nan)
        rm = rps(mg, sm, f.margin.values, -60, 60)
        rt = rps(tt, st, f.total.values, 0, 110)
        per[name] = dict(ll=ll, br=br, m=(f.margin.values - mg) ** 2, t=(f.total.values - tt) ** 2, rm=rm, rt=rt)
        log(f"  {name:32s}{np.nanmean(ll):10.4f}{np.nanmean(br):9.4f}{math.sqrt(per[name]['m'].mean()):10.3f}"
            f"{math.sqrt(per[name]['t'].mean()):10.3f}{rm.mean():9.4f}{rt.mean():9.4f}{sm.mean():8.2f}")
    log()
    log("  each less glmer, match by match (t)")
    log(f"  {'':32s}{'log loss':>18s}{'margin sq err':>18s}{'total sq err':>18s}{'margin RPS':>16s}{'total RPS':>16s}")
    for name in res:
        if name == "glmer":
            continue
        cells = []
        for k, fmt in (("ll", "{:+.5f}"), ("m", "{:+.3f}"), ("t", "{:+.3f}"), ("rm", "{:+.4f}"), ("rt", "{:+.4f}")):
            dlt = per[name][k] - per["glmer"][k]
            ok = ~np.isnan(dlt)
            sd_ = dlt[ok].std(ddof=1)
            cells.append(f"{fmt.format(dlt[ok].mean())} ({dlt[ok].mean() / (sd_ / math.sqrt(ok.sum())):+.1f})"
                         if sd_ > 1e-12 else "same")
        log(f"  {name:32s}{cells[0]:>18s}{cells[1]:>18s}{cells[2]:>18s}{cells[3]:>16s}{cells[4]:>16s}")
    # what the outputs look like
    log()
    log(f"  the outputs, scored span: drives a match {f.pace.mean():.2f} (sd across matches {f.pace.std():.2f}; "
        f"real {f.drives_real.mean():.2f}), touchdowns a drive {pd.concat([f.p_td_h, f.p_td_a]).mean():.3f}, "
        f"field goals a drive {pd.concat([f.p_fg_h, f.p_fg_a]).mean():.3f}")
    rr = f.dropna(subset=["drives_real"])
    log(f"  foreseen drives against real: r {np.corrcoef(rr.pace, rr.drives_real)[0, 1]:.3f}; points a drive "
        f"foreseen against real: r {np.corrcoef(pd.concat([rr.ppd_h, rr.ppd_a]), pd.concat([rr.s1 / rr.h_drives, rr.s2 / rr.a_drives]).replace(np.inf, np.nan).fillna(0))[0, 1]:.3f}")
    _, _, sm_own, _ = res["pace x efficiency, own spread"]
    q = pd.qcut(f.pace, 5, labels=False)
    mg = res["pace x efficiency"][0][sc]
    log("  by fifth of foreseen pace: the margin's real sd about pace x efficiency, and its own sd")
    for k in range(5):
        g = q == k
        log(f"    {k + 1}: pace {f.pace[g].mean():5.2f}  real sd {(f.margin[g] - mg[g]).std():5.2f}  "
            f"own sd {sm_own[sc][g].mean():5.2f}")
    # prod's lines, the deck's view
    if "prod_total_line" in f:
        g = f[f.prod_total_line.notna() & f.prod_spread_line.notna() & f.prod_p_home.notna()]
        sign = 1.0 if np.corrcoef(g.prod_spread_line, g.prod_p_home)[0, 1] > 0 else -1.0
        log()
        log(f"  against prod's pre-match lines ({len(g):,} matches): total MAE, spread MAE, moneyline log loss")
        won_g = (g.margin > 0).astype(float)
        dg = g.margin != 0
        pp = np.clip(g.prod_p_home.values, 1e-9, 1 - 1e-9)
        llp = -(won_g * np.log(pp) + (1 - won_g) * np.log(1 - pp))[dg].mean()
        log(f"    {'prod':30s}{(g.total - g.prod_total_line).abs().mean():8.3f}"
            f"{(g.margin - sign * g.prod_spread_line).abs().mean():8.3f}{llp:9.4f}")
        for name, (mg_, tt_, sm_, _) in res.items():
            mgg, ttg, smg = mg_.reindex(g.index), tt_.reindex(g.index), sm_.reindex(g.index)
            p = np.clip(1 - norm_cdf(-mgg.values / smg.values), 1e-9, 1 - 1e-9)
            ll = -(won_g * np.log(p) + (1 - won_g) * np.log(1 - p))[dg].mean()
            log(f"    {name:30s}{(g.total - ttg).abs().mean():8.3f}{(g.margin - mgg).abs().mean():8.3f}{ll:9.4f}")
    # the outputs as a table: glmer's points for the means (v21 reads Pred_P1/P2_Points with
    # --v21-prior-model), pace and the touchdown / field-goal mix beside them, and the spread they give
    _, _, sm_h, st_h = res["glmer, sd by pace x efficiency"]
    tab = t[["start", "p1", "p2", "s1", "s2"]].copy()
    tab.insert(0, "MATCH_CODE", tab.index)
    tab["Pred_P1_Points"], tab["Pred_P2_Points"] = t.glmer_p1.round(4), t.glmer_p2.round(4)
    tab["exp_drives"] = t.pace.round(3)
    for s_, n in (("h", "P1"), ("a", "P2")):
        tab[f"{n}_td_per_drive"], tab[f"{n}_fg_per_drive"] = t[f"p_td_{s_}"].round(4), t[f"p_fg_{s_}"].round(4)
        tab[f"{n}_eff_points"] = t[f"eff_pts_{s_}"].round(4)
    tab["margin_sd"], tab["total_sd"] = sm_h.round(3), st_h.round(3)
    tab["p_home"] = (1 - norm_cdf(-(t.glmer_p1 - t.glmer_p2).values / sm_h.values)).round(5)
    tab.to_csv(os.path.join(a.out, "pace_eff_prior.csv"), index=False)
    log(f"\n  -> {a.out}/pace_eff_prior.csv: glmer's points (Pred_P1/P2_Points), exp_drives, touchdowns and "
        "field goals a drive, margin_sd and total_sd off them, p_home")
    with open(os.path.join(a.out, "pace_eff.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
