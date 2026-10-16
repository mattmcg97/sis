"""NB2's historical pass: every match since a day priced by NB2 refitted the night before, as live,
with no layers -- its expected points, margin, total and moneyline beside the match's metadata in
one CSV, to fit later models on (glmer/cadence.R --nb2) and to grade NB2 itself over the year.

    python -m eAMFCalibrator nb2-history --since 2026-01-01 --history eAMFCalibrator/out/match_history.csv

The fits are eAMFModel prior-daily's (rolling_prior.py): one a day, each on every result settled
before that day began (NBRatingTrial at --half-life, its own default 60 days, and the level scale
fitted a week earlier), kept in nb2_history[_hl<D>]/ so a rerun fits only the days it lacks. Only
the days the history covers are fitted: a later day's fit would miss results. Each match is priced
by its own day's fit and NB2's own joint score model (each side negative binomial with its gamers'
dispersions, a shared gamma pace factor; score_model.py): the moneyline splits ties half and half,
and the home win, draw and away win are given apart too.

Every NB2 price in the CSV was made before its match's day began, so a model fitted on the CSV's
earlier rows and priced on a later day's sees nothing it couldn't have live.

Writes (to --out, default out/):
  nb2_history.csv   a row a match: the match (code, kick-off, stream, gamers, teams, the day's
                    meeting number, each gamer's settled matches before the day, the score), NB2's
                    expected points (Pred_P1_Points / Pred_P2_Points, with P1_Effective_Alpha /
                    P2_Effective_Alpha / Shared_Gamma_Fraction, so prior-backtest reads its own score
                    model), margin, total, moneyline, and the fit that priced it (its day, matches
                    fitted on, level scale)
  nb2_history.txt   NB2 graded month by month: the moneyline's log loss and Brier, the margin's and
                    total's RMSE and bias, and its calibration
"""

import csv
import datetime as dt
import math
import os
from collections import defaultdict

from . import prior_backtest as pb, score_model, snowflake_io
from .bet_rematches import day_meetings
from .bet_totals import _start

CSV_NAME = "nb2_history.csv"
TXT_NAME = "nb2_history.txt"
DRAWS = 20000
COLUMNS = ["MATCH_CODE", "start", "day", "stream", "home", "home_team", "away", "away_team", "meeting",
           "home_matches_before", "away_matches_before", "home_score", "away_score", "margin", "total",
           "Pred_P1_Points", "Pred_P2_Points", "exp_margin", "exp_total", "p_home", "p_home_win", "p_draw",
           "p_away_win", "P1_Effective_Alpha", "P2_Effective_Alpha", "Shared_Gamma_Fraction", "fit_day",
           "fit_matches", "level_scale"]


def fits_dir(out_dir, half_life=None):
    return os.path.join(out_dir, "nb2_history" if half_life is None else f"nb2_history_hl{float(half_life):g}")


def _i(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def experience(history):
    """{(gamer, day): the gamer's settled matches before that day} for every day the gamer played."""
    played = defaultdict(list)
    for r in history:
        t = _start(r.get("SCHEDULED_START_TIME_UTC"))
        if t is None or _i(r.get("PLAYER_1_FINAL_SCORE")) is None:
            continue
        for k in ("PLAYER_1_HANDLE", "PLAYER_2_HANDLE"):
            played[(r.get(k) or "").strip().upper()].append(t.date())
    out = {}
    for g, days in played.items():
        days.sort()
        seen = 0
        for k, d in enumerate(days):
            if k == 0 or d != days[k - 1]:
                seen = k
            out[(g, d)] = seen
    return out


def price_rows(fit_dir, rows, history, draws=DRAWS):
    """[CSV row] for the rows (history rows) each priced by its own day's fit."""
    from eAMFModel import rolling_prior
    prior = rolling_prior.Rolling(fit_dir)
    points, own = pb.read_rolling(fit_dir, rows, history)
    meets = day_meetings(history)
    seen = experience(history)
    out = []
    for r in rows:
        code = r["MATCH_CODE"]
        if code not in points:
            continue
        t = _start(r["SCHEDULED_START_TIME_UTC"])
        fit = prior.fit_for(t.date())
        mu1, mu2 = points[code]
        kind, params = own.get(code, ("normal", (mu1 - mu2, mu1 + mu2)))
        bk = score_model.book(kind, params, code, draws)
        if kind == "nb2":
            m = bk.margin
            p_win, p_draw = float((m > 0).mean()), float((m == 0).mean())
        else:
            p_win, p_draw = bk.p_home(), 0.0
        s1, s2 = _i(r.get("PLAYER_1_FINAL_SCORE")), _i(r.get("PLAYER_2_FINAL_SCORE"))
        home, away = r["PLAYER_1_HANDLE"].strip().upper(), r["PLAYER_2_HANDLE"].strip().upper()
        meta = getattr(fit, "meta", {}) or {}
        out.append(dict(
            MATCH_CODE=code, start=t.strftime("%Y-%m-%d %H:%M:%S"), day=t.date().isoformat(),
            stream=r.get("STREAM_NUMBER"), home=home, home_team=r.get("PLAYER_1_TEAM"), away=away,
            away_team=r.get("PLAYER_2_TEAM"), meeting=(meets.get(code) or {}).get("k"),
            home_matches_before=seen.get((home, t.date()), 0), away_matches_before=seen.get((away, t.date()), 0),
            home_score=s1, away_score=s2, margin=None if s1 is None else s1 - s2,
            total=None if s1 is None else s1 + s2, Pred_P1_Points=round(mu1, 4), Pred_P2_Points=round(mu2, 4),
            exp_margin=round(mu1 - mu2, 4), exp_total=round(mu1 + mu2, 4), p_home=round(bk.p_home(), 5),
            p_home_win=round(p_win, 5), p_draw=round(p_draw, 5), p_away_win=round(1 - p_win - p_draw, 5),
            P1_Effective_Alpha=params[2] if kind == "nb2" else None,
            P2_Effective_Alpha=params[3] if kind == "nb2" else None,
            Shared_Gamma_Fraction=params[4] if kind == "nb2" else None,
            fit_day=meta.get("before", "")[:10] or None, fit_matches=meta.get("fitted_on"),
            level_scale=meta.get("scale")))
    out.sort(key=lambda x: (x["start"], x["MATCH_CODE"]))
    return out


def _ll(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def grade(rows):
    """Lines: NB2 month by month, then its moneyline's calibration."""
    by = defaultdict(list)
    for r in rows:
        if r["margin"] is not None:
            by[r["day"][:7]].append(r)
            by["all"].append(r)
    L = ["  NB2 refitted daily, no layers: each match by its own day's fit and NB2's joint score model",
         "  ML: the moneyline (ties half each way) graded on the decided matches; RMSE and bias (final less "
         "expected) of the margin and total", "",
         f"  {'month':8s} {'matches':>8s} {'ML ll':>7s} {'Brier':>7s} {'mrg RMSE':>9s} {'mrg bias':>9s} "
         f"{'tot RMSE':>9s} {'tot bias':>9s} {'draws':>6s} {'p draw':>7s}"]
    for month in sorted(k for k in by if k != "all") + ["all"]:
        rs = by[month]
        dec = [r for r in rs if r["margin"] != 0]
        ll = sum(_ll(r["p_home"], 1.0 if r["margin"] > 0 else 0.0) for r in dec) / len(dec) if dec else math.nan
        br = sum((r["p_home"] - (1.0 if r["margin"] > 0 else 0.0)) ** 2 for r in dec) / len(dec) if dec else math.nan
        me = [r["margin"] - r["exp_margin"] for r in rs]
        te = [r["total"] - r["exp_total"] for r in rs]
        L.append(f"  {month:8s} {len(rs):8,d} {ll:7.4f} {br:7.4f} {math.sqrt(sum(x * x for x in me) / len(me)):9.2f} "
                 f"{sum(me) / len(me):+9.2f} {math.sqrt(sum(x * x for x in te) / len(te)):9.2f} "
                 f"{sum(te) / len(te):+9.2f} {(len(rs) - len(dec)) / len(rs):6.1%} "
                 f"{sum(r['p_draw'] for r in rs) / len(rs):7.1%}")
    dec = [r for r in by["all"] if r["margin"] != 0]
    if dec:
        L += ["", "  the moneyline's calibration (decided matches): NB2's chance of a home win against how "
                  "often the home side won",
              f"  {'NB2':>11s} {'matches':>8s} {'NB2 mean':>9s} {'won':>7s}"]
        bins = defaultdict(list)
        for r in dec:
            bins[min(9, int(r["p_home"] * 10))].append(r)
        for b in sorted(bins):
            rs = bins[b]
            L.append(f"  {b / 10:4.1f}-{(b + 1) / 10:3.1f}  {len(rs):8,d} {sum(r['p_home'] for r in rs) / len(rs):9.3f} "
                     f"{sum(1 for r in rs if r['margin'] > 0) / len(rs):7.3f}")
    return L


def load_history(path=None, cur=None):
    """The history rows: from a match_history.csv, else Snowflake's."""
    if path:
        from eAMFModel import nb2_prior
        return nb2_prior.load_history(path)
    return snowflake_io.fetch_history(cur)


def run(out_dir, since, until=None, history_path=None, cur=None, half_life=None, workers=4, draws=DRAWS):
    """Fit, price and grade. Returns the CSV's path."""
    from eAMFModel import rolling_prior
    os.makedirs(out_dir, exist_ok=True)
    history = load_history(history_path, cur)
    first = _start(since).date()
    end = _start(until) if until else None
    newest = max((t for t in (_start(r.get("SCHEDULED_START_TIME_UTC")) for r in history
                              if _i(r.get("PLAYER_1_FINAL_SCORE")) is not None) if t), default=None)
    if newest is None:
        raise SystemExit("no settled matches in the history")
    last = newest.date()
    if end is not None:
        last = min(last, (end - dt.timedelta(seconds=1)).date())
    if last < first:
        raise SystemExit(f"the history ends {newest:%Y-%m-%d %H:%M}, before {first}")
    print(f"  {len(history):,} matches in the history, to {newest:%Y-%m-%d %H:%M}; NB2 fitted each day "
          f"{first} to {last}" + (f" (half-life {float(half_life):g} days)" if half_life else ""), flush=True)
    directory = fits_dir(out_dir, half_life)
    rolling_prior.fit(history, "nb2", directory, first, last, workers=workers,
                      options={} if half_life is None else {"half_life": half_life})
    rows = [r for r in history if (t := _start(r.get("SCHEDULED_START_TIME_UTC")))
            and first <= t.date() <= last and _i(r.get("PLAYER_1_FINAL_SCORE")) is not None]
    print(f"  pricing {len(rows):,} settled matches, each by its own day's fit", flush=True)
    out = price_rows(directory, rows, history, draws)
    path = os.path.join(out_dir, CSV_NAME)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        w.writerows(out)
    L = [f"  {len(out):,} matches from {first} to {last} ({directory})"] + grade(out)
    missing = len(rows) - len(out)
    if missing:
        L.append(f"\n  {missing:,} settled matches NB2 couldn't price (a gamer or team its fit hadn't seen)")
    print("\n".join(L))
    with open(os.path.join(out_dir, TXT_NAME), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path
