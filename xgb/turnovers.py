"""How much of a match's drive count is its turnovers, counted off the scouting export as the sim counts them.

    python -m xgb.turnovers --scouting eAMFCalibrator/out_scouting_xgb/scouting_playover.csv \
        --history eAMFCalibrator/out/match_history.csv --since 2026-09-01

Each match's snaps are read as v21 reads them (eAMFModel/feed.snap_records: the records its turnover draws
are fitted on) and its drives as v21 counts them (v21_report.drive_starts). Per side:
  drives, snaps
  turnovers   interceptions and fumbles: a snap after which the other side has the ball, not on downs
              (TURNOVER), and the defence scoring (DEF_TD)
  downs       turnovers on downs
  punts
  tds         touchdowns off the side's own snaps
and per match `suspect`: turnovers whose next row has a touchdown message for the side that lost the ball
(a touchdown the feed filed late: what made xgb/drives.py's turnover counts unreliable).

On the matches from --since to --until (the gamers' trait on every match counted):
  1. drives against turnovers: the correlation, the drives a turnover adds (alone, and with punts and
     turnovers on downs), and the drive count's variance split between the snaps a game holds and the
     snaps a drive lasts
  2. a gamer's turnovers as a trait: odd v even ISO weeks (MIN_HALF matches in each), the turnovers a drive
     the gamer gives up and forces, and the spread in turnovers a match that implies
  3. before the match: the expected drives and the expected turnovers, each the pace prior's ridge
     (eAMFCalibrator/pace_prior.py) refitted every day on the results in by then; how well each foresees
     its own, how much of the pace's miss is the match's turnovers, and whether the expected turnovers add
     to the pace in foreseeing drives
  4. the margin: how much of the points margin the turnover margin explains
Writes turnovers.txt and turnovers_matches.csv (every match counted) to --out; the counts are kept there
while the export is the same.
"""

import argparse
import csv
import datetime as dt
import json
import os

import numpy as np

from eAMFCalibrator import pace_prior
from eAMFCalibrator.bet_totals import _start
from eAMFModel import feed, nb2_prior, progress
from eAMFModel.v21_report import drive_starts

COUNTS = ("drives", "snaps", "turnovers", "downs", "punts", "tds")
FIELDS = [f"{s}_{k}" for s in ("h", "a") for k in COUNTS] + ["suspect"]
MIN_HALF = 30


def count(rows):
    """A match's counts per side (h_: home's, a_: away's offense) and its suspect turnovers."""
    rows = sorted(rows, key=lambda r: feed._i(r.get("message")) or 0)
    side = next((r["team_a_side"] for r in rows if r.get("team_a_side") in ("home", "away")), "home")
    home, away = ("TEAM_A", "TEAM_B") if side == "home" else ("TEAM_B", "TEAM_A")
    by_msg = {feed._i(r.get("message")): r for r in rows}
    n = {t: dict.fromkeys(COUNTS, 0) for t in (home, away)}
    suspect = 0
    recs = feed.snap_records(rows)
    for r in drive_starts(recs):
        if r["offense"] in n:
            n[r["offense"]]["drives"] += 1
    for r in recs:
        o = r["offense"]
        if o not in n:
            continue
        b = by_msg.get(r["message"], {})
        n[o]["snaps"] += 1
        if r["kind"] in (feed.TURNOVER, feed.DEF_TD):
            n[o]["turnovers"] += 1
            suspect += r["kind"] == feed.TURNOVER and feed._scorer(b, "TOUCHDOWN_TEAM") == o
        elif b.get("play_kind") == "TOUCHDOWN":
            n[o]["tds"] += 1
        elif b.get("offense") in n and b["offense"] != o:      # a gain record handing the ball over: on downs
            n[o]["downs"] += 1
    for a, b in zip(rows, rows[1:]):
        if b.get("play_kind") == "PUNT" and a.get("offense") in n:
            n[a["offense"]]["punts"] += 1
    out = {f"{s}_{k}": n[t][k] for s, t in (("h", home), ("a", away)) for k in COUNTS}
    out["suspect"] = int(suspect)
    return out


def counts(export, out_dir, log=print):
    """{match: count(...)} off the export, kept in out_dir while the export is the same."""
    path, stamp_path = os.path.join(out_dir, "counts.csv"), os.path.join(out_dir, "counts.json")
    stamp = pace_prior.export_stamp(export)
    if os.path.exists(path) and os.path.exists(stamp_path):
        with open(stamp_path, encoding="utf-8") as fh:
            if json.load(fh) == stamp:
                with open(path, newline="", encoding="utf-8") as fh:
                    got = {r["MATCH_CODE"]: {k: int(r[k]) for k in FIELDS} for r in csv.DictReader(fh)}
                log(f"  {len(got):,} matches counted before off {export}, kept")
                return got
    line = progress.Line()
    got = pace_prior.each_match(export, count, warn=log,
                                on_match=lambda k: k % 250 == 0 and line.draw(f"  counting: {k:,} matches"))
    line.clear()
    got = {c: v for c, v in got.items() if v["h_drives"] + v["a_drives"] > 0}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE"] + FIELDS)
        for c, v in sorted(got.items()):
            w.writerow([c] + [v[k] for k in FIELDS])
    with open(stamp_path, "w", encoding="utf-8") as fh:
        json.dump(stamp, fh, indent=1)
    log(f"  {len(got):,} matches counted off {export}")
    return got


# ---------------------------------------------------------------- the numbers

def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def ols(y, *xs):
    """(coefficients, their standard errors, R2) of y on an intercept and xs."""
    X = np.column_stack([np.ones(len(y))] + list(xs))
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    s2 = res @ res / max(1, len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    return b, se, 1.0 - res.var() / y.var()


def drive_lines(m):
    D, S, T = m["D"], m["S"], m["T"]
    DN, P, TD = m["DN"], m["P"], m["TD"]
    L = [f"  {len(D):,} matches. A match (both sides): drives {D.mean():.2f} (sd {D.std():.2f}), snaps {S.mean():.1f} "
         f"(sd {S.std():.1f}), turnovers {T.mean():.2f} (sd {T.std():.2f}), turnovers on downs {DN.mean():.2f}, "
         f"punts {P.mean():.2f}, touchdowns {TD.mean():.2f}",
         f"  turnovers whose next row has a touchdown message for the side that lost the ball: {m['suspect'].mean():.3f} "
         f"a match ({m['suspect'].sum() / max(1, T.sum()):.1%} of turnovers)", ""]
    r = corr(D, T)
    b1, se1, _ = ols(D, T)
    b3, se3, r2 = ols(D, T, DN, P)
    L += [f"  drives against turnovers: r {r:.2f} (R2 {r * r:.2f}); a turnover {b1[1]:+.2f} drives (se {se1[1]:.2f}) on its own",
          f"  with turnovers on downs and punts (R2 {r2:.2f}): a turnover {b3[1]:+.2f}, a turnover on downs {b3[2]:+.2f}, "
          f"a punt {b3[3]:+.2f} drives"]
    ok = (D > 0) & (S > 0)
    lD, lS, lR = np.log(D[ok]), np.log(S[ok]), np.log(S[ok] / D[ok])
    v = lD.var()
    L += [f"  the drive count's variance (logs): the snaps a game {lS.var() / v:.0%}, the snaps a drive {lR.var() / v:.0%}, "
          f"together {-2 * np.cov(lS, lR)[0, 1] / v:+.0%}; drives against snaps a drive r {corr(D[ok], S[ok] / D[ok]):.2f}, "
          f"against turnovers a drive r {corr(D[ok], T[ok] / D[ok]):.2f}"]
    return L


def trait_lines(rows):
    """Odd v even ISO weeks: each gamer's turnovers a drive given up and forced, and the drives in their matches."""
    acc = {}
    for r in rows:
        half = r["start"].isocalendar()[1] % 2
        for g, own, opp in ((r["g1"], "h", "a"), (r["g2"], "a", "h")):
            a = acc.setdefault((g, half), [0, 0, 0, 0, 0, 0])
            a[0] += 1
            a[1] += r[f"{own}_turnovers"]
            a[2] += r[f"{own}_drives"]
            a[3] += r[f"{opp}_turnovers"]
            a[4] += r[f"{opp}_drives"]
            a[5] += r[f"{own}_drives"] + r[f"{opp}_drives"]
    both = [g for g in {g for g, _ in acc} if all(acc.get((g, h), [0])[0] >= MIN_HALF for h in (0, 1))]
    if len(both) < 5:
        return [f"  {len(both)} gamers with {MIN_HALF}+ matches in both halves: too few"]
    L = [f"  {len(both)} gamers with {MIN_HALF}+ matches in both halves (odd v even ISO weeks):"]
    for name, f in (("turnovers a drive given up", lambda a: a[1] / max(1, a[2])),
                    ("turnovers a drive forced", lambda a: a[3] / max(1, a[4])),
                    ("drives in their matches", lambda a: a[5] / a[0])):
        x = np.array([f(acc[(g, 0)]) for g in both])
        y = np.array([f(acc[(g, 1)]) for g in both])
        L.append(f"    {name:28s} r {corr(x, y):.2f}; mean {np.r_[x, y].mean():.3f}, spread between gamers sd {np.r_[x, y].std():.3f}")
    rate = np.array([acc[(g, h)][1] / max(1, acc[(g, h)][2]) for g in both for h in (0, 1)])
    drives = np.mean([acc[(g, h)][2] / acc[(g, h)][0] for g in both for h in (0, 1)])
    L.append(f"    a gamer's rate a sd from the middle: {rate.std() * drives:.2f} turnovers a match of their own "
             f"({drives:.1f} drives a side)")
    return L


def prematch_lines(m):
    D, T = m["D"], m["T"]
    eD, eT = m["eD"] / m["lD"], m["eT"] / m["lT"]            # each against its day's league
    missD, missT = D - m["eD"], T - m["eT"]
    L = [f"  {len(D):,} matches priced before kick-off (each fit on the results in by its midnight)",
         f"  expected drives:    r {corr(eD, D):.3f} with the real; spread sd {m['eD'].std():.2f} against the real {D.std():.2f}",
         f"  expected turnovers: r {corr(eT, T):.3f} with the real; spread sd {m['eT'].std():.2f} against the real {T.std():.2f}",
         f"  the pace's miss (real - expected drives, sd {missD.std():.2f}) against the match's turnovers: r "
         f"{corr(missD, T):.2f} (R2 {corr(missD, T) ** 2:.2f}); against the turnovers' own miss r {corr(missD, missT):.2f}"]
    _, _, r2a = ols(D, eD)
    b, se, r2b = ols(D, eD, eT)
    L.append(f"  drives on the pace and the expected turnovers: the turnovers' coefficient {b[2]:+.2f} (t {b[2] / se[2]:+.1f}); "
             f"R2 {r2a:.4f} with the pace alone, {r2b:.4f} with both")
    return L


def margin_lines(m):
    M, TM = m["margin"], m["to_margin"]
    b, se, r2 = ols(M, TM)
    return [f"  the points margin (home - away, sd {M.std():.1f}) against the turnover margin (away's - home's, sd "
            f"{TM.std():.2f}): r {corr(M, TM):.2f} (R2 {r2:.2f}); a turnover {b[1]:+.1f} points (se {se[1]:.1f})"]


def frame(history, got, pace, tos, first, last):
    """The matches counted, with their gamers, finals and expected drives and turnovers, as columns."""
    rows = []
    for h in history:
        c = (h.get("MATCH_CODE") or "").strip()
        t = _start(h.get("SCHEDULED_START_TIME_UTC"))
        if c not in got or not t:
            continue
        f1, f2 = h.get("PLAYER_1_FINAL_SCORE"), h.get("PLAYER_2_FINAL_SCORE")
        try:
            f1, f2 = float(f1), float(f2)
        except (TypeError, ValueError):
            f1 = f2 = None
        r = dict(got[c], code=c, start=t, g1=(h.get("PLAYER_1_HANDLE") or "").strip().upper(),
                 g2=(h.get("PLAYER_2_HANDLE") or "").strip().upper(), f1=f1, f2=f2,
                 scored=first <= t.date() <= last)
        if c in pace and c in tos:
            r.update(eD=pace[c][0], lD=pace[c][1], eT=tos[c][0], lT=tos[c][1])
        rows.append(r)
    return rows


def columns(rows):
    g = lambda k: np.array([r[k] for r in rows], float)
    m = {"D": g("h_drives") + g("a_drives"), "S": g("h_snaps") + g("a_snaps"), "T": g("h_turnovers") + g("a_turnovers"),
         "DN": g("h_downs") + g("a_downs"), "P": g("h_punts") + g("a_punts"), "TD": g("h_tds") + g("a_tds"),
         "suspect": g("suspect")}
    if rows and all("eD" in r for r in rows):
        m.update(eD=g("eD"), lD=g("lD"), eT=g("eT"), lT=g("lT"))
    if rows and all(r["f1"] is not None for r in rows):
        m.update(margin=g("f1") - g("f2"), to_margin=g("a_turnovers") - g("h_turnovers"))
    return m


def run(export, history_path, since, until, out_dir, log=print):
    os.makedirs(out_dir, exist_ok=True)
    history = nb2_prior.load_history(history_path)
    got = counts(export, out_dir, log)
    first = _start(since).date()
    last = (_start(until) - dt.timedelta(seconds=1)).date() if until else max(
        t.date() for t in (_start(h.get("SCHEDULED_START_TIME_UTC")) for h in history) if t)
    drives = {c: v["h_drives"] + v["a_drives"] for c, v in got.items()}
    turnovers = {c: v["h_turnovers"] + v["a_turnovers"] for c, v in got.items()}
    pace = pace_prior.walk(history, drives, first, last)
    tos = pace_prior.walk(history, turnovers, first, last)
    rows = frame(history, got, pace, tos, first, last)
    scored = [r for r in rows if r["scored"]]
    L = [f"Turnovers and drives, counted as the sim counts them, off {export}", ""]
    if not scored:
        raise SystemExit(f"no counted matches from {first} to {last}")
    L += ["1. A match's drives against its turnovers", f"  {first} to {last}:"] + drive_lines(columns(scored)) + [""]
    L += ["2. A gamer's turnovers as a trait (every match counted)"] + trait_lines(rows) + [""]
    priced = [r for r in scored if "eD" in r]
    L += ["3. Before the match: expected drives and expected turnovers, the pace prior's ridge refitted daily"]
    L += prematch_lines(columns(priced)) if len(priced) >= 50 else ["  too few matches priced"]
    L += [""]
    finals = [r for r in scored if r["f1"] is not None]
    L += ["4. The margin"] + (margin_lines(columns(finals)) if len(finals) >= 50 else ["  too few finals"])
    log("\n".join(L))
    with open(os.path.join(out_dir, "turnovers.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(out_dir, "turnovers_matches.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["MATCH_CODE", "start", "home", "away", "home_final", "away_final"] + FIELDS
                   + ["exp_drives", "league_drives", "exp_turnovers", "league_turnovers"])
        for r in rows:
            w.writerow([r["code"], r["start"].strftime("%Y-%m-%d %H:%M:%S"), r["g1"], r["g2"], r["f1"], r["f2"]]
                       + [r[k] for k in FIELDS]
                       + [round(r[k], 3) if k in r else "" for k in ("eD", "lD", "eT", "lT")])
    log(f"\n  -> {out_dir}: turnovers.txt, turnovers_matches.csv")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scouting", required=True, help="the scouting export (scouting_playover.csv)")
    ap.add_argument("--history", default=os.path.join("eAMFCalibrator", "out", "match_history.csv"))
    ap.add_argument("--since", default="2026-09-01", help="the first day scored (default 2026-09-01)")
    ap.add_argument("--until", help="the day after the last scored (default: the history's end)")
    ap.add_argument("--out", default=os.path.join("xgb", "out", "turnovers"))
    a = ap.parse_args(argv)
    for p in (a.scouting, a.history):
        if not os.path.isfile(p):
            raise SystemExit(f"{p}: no such file")
    run(a.scouting, a.history, a.since, a.until, a.out)


if __name__ == "__main__":
    main()
