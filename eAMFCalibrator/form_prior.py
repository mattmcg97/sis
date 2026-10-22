"""Form priors: each walkthrough table's expected margin moved toward a fast walk-forward rating -- every
gamer's strength as his latest results have it -- so a sim prices the gamer who is playing well now.

On the Oct 1-12 bets a rating like this one, drifting 1-4 points^2 a day, re-priced prod's moneyline by
+1.2 to +1.9 points on the same bets (95% intervals clear of zero, in play as well as at kick-off) while
its Brier was no better than prod's or worse; the slow ratings (NB2's and glmer's 60-day half-lives, a
rating drifting 0.01 a day) gained nothing there. These tables put that view into a prior any sim reads.

    python -m eAMFCalibrator form-prior          # v2.1's default prior, from out/walkthrough/priors/avg.csv
    python -m eAMFCalibrator form-prior --history eAMFCalibrator/out/match_history.csv \\
        --table avg=eAMFCalibrator/out/walkthrough_oct/priors/avg.csv --drift 1,2,4 --weight 0.5,1

By default it moves the walkthrough's avg table half way to the rating drifting 4 points^2 a day and
writes priors/avg_form4_w05.csv to out/form_prior/: v2.1's default prior (eAMFModel's v21_stream
DEFAULT_PRIOR_TABLE), the one that beat the v14 glmer build on the Oct 1-12 bets and on Oct 13-17 held
out (the moneyline in play +1.1 and +3.7 points against it, on the same bets).

  the rating   every gamer's strength in points of margin, a random walk drifting DRIFT points^2 a day,
               each match's margin a draw about the two strengths and the teams' (SIGMA a match), the
               gamers updated after every result in kick-off order (a Kalman filter) and the teams by
               TEAM_RATE of the miss; each match is priced off the results before it
  the tables   each base table's rows with its expected margin moved WEIGHT of the way to the rating's,
               the total kept (half the move to each side's points), p_home read off the row's own score
               model at the new points; a match the rating has no view of is left as it is

Writes (to --out): priors/<base>_form<drift>_w<weight>.csv for every base, drift and weight, and
form_prior.txt; with --bets (bets_sim.csv files), how many of their matches the history and each table
have, first.
"""

import csv
import datetime as dt
import math
import os

from . import score_model
from .money_layer import _num, _time, read_table

SIGMA = 12.0               # points: a match's margin about the two strengths (fitted Mar-Sep)
V0 = 16.0                  # points^2: a new gamer's uncertainty
TEAM_RATE = 0.002          # a team's rating moves this share of each miss
DRIFTS = (4.0,)                # v2.1's default prior: drift 4, half way
WEIGHTS = (0.5,)
DRAWS = 20000


def read_matches(path):
    """[(kick-off, code, home, away, home team, away team, home score, away score or None)] in kick-off order."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("SPORT_CODE", "AF") not in ("AF", "", None):
                continue
            t = _time(r.get("SCHEDULED_START_TIME_UTC"))
            code = (r.get("MATCH_CODE") or "").strip()
            if t is None or not code:
                continue
            a, b = _num(r.get("PLAYER_1_FINAL_SCORE")), _num(r.get("PLAYER_2_FINAL_SCORE"))
            out.append((t, code, (r.get("PLAYER_1_HANDLE") or "").strip().upper(), (r.get("PLAYER_2_HANDLE") or "").strip().upper(),
                        (r.get("PLAYER_1_TEAM") or "").strip(), (r.get("PLAYER_2_TEAM") or "").strip(),
                        a, None if a is None or b is None else b))
    out.sort(key=lambda m: (m[0], m[1]))
    return out


def rating(matches, drift, sigma=SIGMA, v0=V0, team_rate=TEAM_RATE):
    """code -> (expected home margin, its sd) for every match, each off the results before it."""
    mu, var, last, team = {}, {}, {}, {}
    out = {}
    for t, code, home, away, th, ta, hs, as_ in matches:
        day = t.replace(tzinfo=dt.timezone.utc).timestamp() / 86400.0     # UTC, not the machine's local time
        for g in (home, away):
            if g not in mu:
                mu[g], var[g] = 0.0, v0
            elif g in last:
                var[g] += drift * max(day - last[g], 0.0)
            last[g] = day
        m = mu[home] - mu[away] + team.get(th, 0.0) - team.get(ta, 0.0)
        s2 = sigma ** 2 + var[home] + var[away]
        out[code] = (m, math.sqrt(s2))
        if hs is None or as_ is None:
            continue
        miss = (hs - as_) - m
        vh, va = var[home], var[away]
        mu[home] += vh / s2 * miss
        mu[away] -= va / s2 * miss
        var[home], var[away] = vh - vh * vh / s2, va - va * va / s2
        if th and ta:
            team[th] = team.get(th, 0.0) + team_rate * miss
            team[ta] = team.get(ta, 0.0) - team_rate * miss
            mean = sum(team.values()) / len(team)
            for k in team:
                team[k] -= mean
    return out


def moved_rows(rows, cols, view, weight, draws=DRAWS):
    """The table's rows, each margin moved `weight` of the way to the rating's (the total kept), p_home off
    the row's own score model at the new points; (rows, the mean absolute move in points)."""
    out, moves = [], []
    for r in rows:
        r = dict(r)
        code = (r.get("MATCH_CODE") or "").strip()
        h, a = _num(r.get("home_pts")), _num(r.get("away_pts"))
        if code not in view or h is None or a is None:
            out.append(r)
            continue
        move = weight * (view[code][0] - (h - a))
        h2, a2 = max(0.1, h + move / 2), max(0.1, a - move / 2)
        moves.append(abs(move))
        kind = (r.get("score_model") or "normal").strip() or "normal"
        params = [_num(x) for x in str(r.get("score_params") or "").split()]
        if kind == "normal" or any(x is None for x in params):
            bk = score_model.Normal(h2 - a2, h2 + a2)
        else:
            bk = score_model.book(kind, (h2, a2) + tuple(params), code, draws)
        r.update(home_pts=round(h2, 4), away_pts=round(a2, 4))
        for k, v in (("Pred_P1_Points", h2), ("Pred_P2_Points", a2), ("exp_margin", h2 - a2)):
            if k in cols:
                r[k] = round(v, 4)
        if "p_home" in cols:
            r["p_home"] = round(bk.p_home(), 5)
        out.append(r)
    return out, (sum(moves) / len(moves) if moves else 0.0)


def bet_matches(paths):
    """{match: stake} off betting-sim bets_sim.csv files."""
    out = {}
    for path in paths or ():
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                code = (r.get("match_code") or "").strip()
                if code:
                    out[code] = out.get(code, 0.0) + (_num(r.get("stake")) or 0.0)
    return out


def coverage(name, have, bets):
    """'NAME has n of the N bet matches (s of S stake)'."""
    n = sum(1 for c in bets if c in have)
    s = sum(v for c, v in bets.items() if c in have)
    return (f"  {name} has {n:,} of the {len(bets):,} bet matches ({s:,.0f} of {sum(bets.values()):,.0f} stake)"
            + ("" if n == len(bets) else f"; missing {len(bets) - n:,}"))


def file_name(base, drift, weight):
    return f"{base}_form{drift:g}_w{str(weight).replace('.', '')}.csv"


def run(out_dir, history, tables, drifts=DRIFTS, weights=WEIGHTS, draws=DRAWS, bets=None):
    """Every base table moved toward every drift's rating at every weight; with `bets` (bets_sim.csv
    files), how many of their matches the history and each table have. Returns form_prior.txt's path."""
    matches = read_matches(history)
    if not matches:
        raise SystemExit(f"{history}: no matches")
    d = os.path.join(out_dir, "priors")
    os.makedirs(d, exist_ok=True)
    L = [f"\nForm priors: {len(matches):,} matches in {history} ({matches[0][0]:%Y-%m-%d} to {matches[-1][0]:%Y-%m-%d})",
         f"  the rating: margin sd {SIGMA:g} a match, a new gamer {V0:g} points^2, teams {TEAM_RATE:g} of each miss"]
    staked = bet_matches(bets)
    if staked:
        L.append(coverage("the history", {m[1] for m in matches if m[6] is not None}, staked))
        for name, path in tables:
            L.append(coverage(f"the {name} table", {(r.get("MATCH_CODE") or "").strip() for r in read_table(path)[0]},
                              staked))
    for drift in drifts:
        view = rating(matches, drift)
        for name, path in tables:
            rows, cols = read_table(path)
            have = sum(1 for r in rows if (r.get("MATCH_CODE") or "").strip() in view)
            for w in weights:
                moved, mean_move = moved_rows(rows, cols, view, w, draws)
                out = os.path.join(d, file_name(name, drift, w))
                with open(out, "w", newline="", encoding="utf-8") as fh:
                    wr = csv.DictWriter(fh, cols, extrasaction="ignore")
                    wr.writeheader()
                    wr.writerows(moved)
                L.append(f"  {os.path.basename(out):32s} {name} moved {w:g} of the way to the rating drifting {drift:g}/day:"
                         f" {have:,} of {len(rows):,} matches, the margin moved {mean_move:.2f} points on average")
    path = os.path.join(out_dir, "form_prior.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n  -> {d}")
    return path
