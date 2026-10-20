"""Pre-match features for every match, each computed only from what was known at its kick-off.

A result counts once it is in, RESULT_LAG after its match's kick-off (eAMFModel's stream), so a
gamer's previous game is only in his features if it finished before this one starts. The schedule
itself (who plays when, for which team, on which stream) is known in advance.

Per side (p1_ = PLAYER_1, the home side; p2_ = PLAYER_2), as the trader deck builds them for
eBasketball, with AF's scores in place of its shot types:
  games, all-time      matches settled, points scored / conceded, win rate
  rolling 5/15/50      points scored, conceded, margin and total over the last N settled matches
  ewma                 points scored / conceded, half-life FORM_HALF_LIFE matches (glmer's form)
  spread               sd of points scored and of the margin over the last 15; max and min scored
                       over the last 10
  streaks              win rate over the last 3 and 10
  team                 matches with this team, and points scored / conceded with it, shrunk toward
                       the gamer's last 50 by TEAM_SHRINK matches
  rest and session     hours since the last kick-off, game of the session (kick-offs under
                       SESSION_GAP apart) and of the day, and the session's settled results so far
  rating               a Kalman rating of every gamer's margin strength (eAMFCalibrator's
                       form_prior: drifting RATING_DRIFT points^2 a day), updated as results come in
Per match:
  head to head         meetings, PLAYER_1's mean margin (shrunk by H2H_SHRINK), the mean total,
                       PLAYER_1's wins in the last 5
  teams                each team's points scored / conceded over the league's last TEAM_DAYS days
  context              hour, weekday, stream
  derived              simple expectations: each side's points as the mean of its scoring and the
                       other's conceding (5, 15, 50 and all-time), their margin and total
  glmer (optional)     a walk-forward glmer's expected points (glmer/cadence.R's matches table)
  drives (optional)    xgb/drives.py's table: each gamer's offence and the offences he has faced over
                       his last 15 and 50 matches with play-by-play (drv_off_ / drv_def_): points,
                       touchdowns, field goals, punts, turnovers and turnovers on downs per drive,
                       red-zone trips and touchdowns from them, plays per drive and where drives
                       start, yards and success per snap, 20+ and 40+ yard plays, losses, 3rd downs
                       converted, drives a match and seconds a snap
  prod (optional)      prod's last pre-match quotes from the same table (prod_): for scoring against,
                       never a feature

    python -m xgb.features --history eAMFCalibrator/out/match_history.csv --glmer cadence_matches.csv \
        --out xgb/out/features.csv
"""

import argparse
import csv
import datetime as dt
import math
import os
from bisect import bisect_left
from collections import defaultdict, deque

import numpy as np

from .drives import STATS

RESULT_LAG = dt.timedelta(minutes=36)
DRIVE_WINDOWS = (15, 50)
DRIVE_STATS = STATS                      # xgb/drives.py's counts, in its order
# rate name -> (numerator, denominator); "points" and "matches" are kept beside the drive stats
DRIVE_RATES = {"ppd": ("points", "drives"), "dppd": ("drive_pts", "drives"), "td": ("td", "drives"), "fg": ("fg", "drives"),
               "punt": ("punt", "drives"), "turnover": ("turnover", "drives"), "downs": ("downs", "drives"),
               "rz": ("rz_trips", "drives"), "rz_td": ("rz_td", "rz_trips"), "plays": ("drive_plays", "drives"),
               "start": ("start_field", "drives"), "ypp": ("yards", "gains"), "success": ("success", "snaps"),
               "big20": ("big20", "snaps"), "big40": ("big40", "snaps"), "loss": ("loss", "snaps"),
               "third": ("third_conv", "third_n"), "drives": ("drives", "matches"), "pace": ("seconds", "snaps")}
DRIVE_KEYS = DRIVE_STATS + ("points", "matches")
SESSION_GAP = dt.timedelta(hours=2)
FORM_HALF_LIFE = 10
TEAM_SHRINK = 10
H2H_SHRINK = 5
TEAM_DAYS = 60
RATING_DRIFT = 4.0
RATING_SIGMA = 12.0
RATING_V0 = 16.0
RATING_TEAM_RATE = 0.002
WINDOWS = (5, 15, 50)


def _time(text):
    text = (text or "").strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return dt.datetime.strptime(text[:26], fmt)
        except ValueError:
            pass
    return None


def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_history(path, sport="AF"):
    """Matches in kick-off order: dicts with code, t, stream, p1, p2, t1, t2, s1, s2 (None if unsettled)."""
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r.get("SPORT_CODE") or sport) != sport:
                continue
            t = _time(r.get("SCHEDULED_START_TIME_UTC"))
            if t is None:
                continue
            s1, s2 = _num(r.get("PLAYER_1_FINAL_SCORE")), _num(r.get("PLAYER_2_FINAL_SCORE"))
            out.append(dict(code=r["MATCH_CODE"].strip(), t=t, stream=(r.get("STREAM_NUMBER") or "").strip(),
                            p1=r["PLAYER_1_HANDLE"].strip().upper(), p2=r["PLAYER_2_HANDLE"].strip().upper(),
                            t1=(r.get("PLAYER_1_TEAM") or "").strip(), t2=(r.get("PLAYER_2_TEAM") or "").strip(),
                            s1=s1 if s2 is not None else None, s2=s2 if s1 is not None else None))
    out.sort(key=lambda m: (m["t"], m["code"]))
    return out


def read_drives(path):
    """{match: {"h": [stats...], "a": [...], prod quotes}} off xgb/drives.py's table."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f1, f2 = _num(r.get("final_p1")), _num(r.get("final_p2"))
            row = {}
            for tag, pts in (("h", f1), ("a", f2)):
                row[tag] = np.array([_num(r.get(f"{tag}_{k}")) or 0.0 for k in DRIVE_STATS]
                                    + [pts if pts is not None else math.nan, 1.0])
            row["prod"] = {"prod_p_home": _num(r.get("prematch_prob_50")),
                           "prod_spread_line": _num(r.get("prematch_line_52")),
                           "prod_spread_prob": _num(r.get("prematch_prob_52")),
                           "prod_total_line": _num(r.get("prematch_line_54")),
                           "prod_total_prob": _num(r.get("prematch_prob_54"))}
            out[r["match_code"]] = row
    return out


def drive_features(g):
    """A side's drive rates, its offence and the offences it faced, over its last DRIVE_WINDOWS
    matches with play-by-play."""
    f = {}
    for role, rows in (("off", g.drv_off), ("def", g.drv_def)):
        for w in DRIVE_WINDOWS:
            if rows:
                tot = np.nansum(np.asarray(rows[-w:]), axis=0)
                val = dict(zip(DRIVE_KEYS, tot))
            for name, (num, den) in DRIVE_RATES.items():
                f[f"drv_{role}_{name}_{w}"] = (val[num] / val[den] if rows and val[den] else math.nan)
    return f


def read_glmer(path):
    """{match: (home points, away points)} off a glmer/cadence.R matches table (its single arm)."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        c1 = [c for c in reader.fieldnames if c.endswith("|1")]
        c2 = [c for c in reader.fieldnames if c.endswith("|2")]
        if len(c1) != 1 or len(c2) != 1:
            raise SystemExit(f"{path}: expected one arm's |1 and |2 columns, found {c1 + c2}")
        for r in reader:
            out[r["MatchId"]] = (float(r[c1[0]]), float(r[c2[0]]))
    return out


class Gamer:
    """One gamer's settled results, newest last, and his schedule."""

    def __init__(self):
        self.pf, self.pa, self.won = [], [], []
        self.by_team = defaultdict(lambda: [0.0, 0.0, 0])   # team -> [scored, conceded, matches]
        self.ewma_f = self.ewma_a = None
        self.drv_off, self.drv_def = [], []   # per match with play-by-play: his offence, the one he faced
        self.kickoffs = []            # every kick-off, settled or not
        self.settled_at = []          # kick-off of each settled result, matching pf


def _mean(x):
    return float(np.mean(x)) if len(x) else math.nan


def _std(x):
    return float(np.std(x)) if len(x) > 1 else math.nan


def side_features(g, team, t):
    """One side's features from its settled results before t."""
    f = {}
    n = len(g.pf)
    pf, pa = np.asarray(g.pf, float), np.asarray(g.pa, float)
    f["games"] = n
    f["pf_all"], f["pa_all"] = _mean(pf), _mean(pa)
    f["win_all"] = _mean(g.won)
    for w in WINDOWS:
        a, b = pf[-w:], pa[-w:]
        f[f"pf_{w}"], f[f"pa_{w}"] = _mean(a), _mean(b)
        f[f"margin_{w}"], f[f"total_{w}"] = _mean(a - b), _mean(a + b)
    f["ewma_pf"] = g.ewma_f if g.ewma_f is not None else math.nan
    f["ewma_pa"] = g.ewma_a if g.ewma_a is not None else math.nan
    f["pf_sd_15"] = _std(pf[-15:])
    f["margin_sd_15"] = _std((pf - pa)[-15:])
    f["pf_max_10"] = float(pf[-10:].max()) if n else math.nan
    f["pf_min_10"] = float(pf[-10:].min()) if n else math.nan
    f["win_3"], f["win_10"] = _mean(g.won[-3:]), _mean(g.won[-10:])
    sf, sa, k = g.by_team[team] if team in g.by_team else (0.0, 0.0, 0)
    f["team_games"] = k
    base_f, base_a = f["pf_50"], f["pa_50"]
    if k and not math.isnan(base_f):
        f["team_pf"] = (sf + TEAM_SHRINK * base_f) / (k + TEAM_SHRINK)
        f["team_pa"] = (sa + TEAM_SHRINK * base_a) / (k + TEAM_SHRINK)
    else:
        f["team_pf"], f["team_pa"] = base_f, base_a
    # rest and session, off the schedule (every kick-off before this one)
    nb = bisect_left(g.kickoffs, t)
    f["rest_hours"] = min(168.0, (t - g.kickoffs[nb - 1]).total_seconds() / 3600) if nb else 168.0
    i = nb
    prev = t
    while i > 0 and prev - g.kickoffs[i - 1] <= SESSION_GAP:
        i -= 1
        prev = g.kickoffs[i]
    f["game_of_session"] = nb - i + 1
    day_start = dt.datetime.combine(t.date(), dt.time())
    f["game_of_day"] = 1 + nb - bisect_left(g.kickoffs, day_start)
    start = prev                                   # the session's first kick-off
    idx = list(range(bisect_left(g.settled_at, start), len(g.settled_at)))
    f["session_settled"] = len(idx)
    f["session_pf"] = _mean(pf[idx]) if idx else math.nan
    f["session_pa"] = _mean(pa[idx]) if idx else math.nan
    f["session_win"] = _mean([g.won[i] for i in idx]) if idx else math.nan
    return f


def build(matches, glmer=None, drive_table=None):
    """[row dict] for every match: identity, result and features."""
    alpha = 1 - 2 ** (-1 / FORM_HALF_LIFE)
    gamers = defaultdict(Gamer)
    h2h = defaultdict(list)                   # (a, b) sorted -> [(a's margin, total)]
    team_log = deque()                        # (kick-off, team, scored, conceded) for TEAM_DAYS
    team_sum = defaultdict(lambda: [0.0, 0.0, 0])
    r_mu, r_var, r_last, r_team = {}, {}, {}, defaultdict(float)
    for m in matches:
        gamers[m["p1"]].kickoffs.append(m["t"])
        gamers[m["p2"]].kickoffs.append(m["t"])
    for g in gamers.values():
        g.kickoffs.sort()

    def settle(m):
        s1, s2 = m["s1"], m["s2"]
        for me, op, team, a, b in ((m["p1"], m["p2"], m["t1"], s1, s2), (m["p2"], m["p1"], m["t2"], s2, s1)):
            g = gamers[me]
            g.pf.append(a)
            g.pa.append(b)
            g.won.append(1.0 if a > b else 0.0 if a < b else 0.5)
            bt = g.by_team[team]
            bt[0] += a
            bt[1] += b
            bt[2] += 1
            g.settled_at.append(m["t"])
            g.ewma_f = a if g.ewma_f is None else g.ewma_f + alpha * (a - g.ewma_f)
            g.ewma_a = b if g.ewma_a is None else g.ewma_a + alpha * (b - g.ewma_a)
        if drive_table is not None and m["code"] in drive_table:
            dr = drive_table[m["code"]]
            gamers[m["p1"]].drv_off.append(dr["h"])
            gamers[m["p1"]].drv_def.append(dr["a"])
            gamers[m["p2"]].drv_off.append(dr["a"])
            gamers[m["p2"]].drv_def.append(dr["h"])
        key = tuple(sorted((m["p1"], m["p2"])))
        h2h[key].append(((s1 - s2) if key[0] == m["p1"] else (s2 - s1), s1 + s2))
        for team, a, b in ((m["t1"], s1, s2), (m["t2"], s2, s1)):
            team_log.append((m["t"], team, a, b))
            ts = team_sum[team]
            ts[0] += a
            ts[1] += b
            ts[2] += 1
        # the rating, as form_prior.rating updates it
        day = m["t"].timestamp() / 86400.0
        for g in (m["p1"], m["p2"]):
            if g not in r_mu:
                r_mu[g], r_var[g] = 0.0, RATING_V0
            elif g in r_last:
                r_var[g] += RATING_DRIFT * max(day - r_last[g], 0.0)
            r_last[g] = day
        exp = r_mu[m["p1"]] - r_mu[m["p2"]] + r_team[m["t1"]] - r_team[m["t2"]]
        s2_ = RATING_SIGMA ** 2 + r_var[m["p1"]] + r_var[m["p2"]]
        miss = (s1 - s2) - exp
        v1, v2 = r_var[m["p1"]], r_var[m["p2"]]
        r_mu[m["p1"]] += v1 / s2_ * miss
        r_mu[m["p2"]] -= v2 / s2_ * miss
        r_var[m["p1"]], r_var[m["p2"]] = v1 - v1 * v1 / s2_, v2 - v2 * v2 / s2_
        if m["t1"] and m["t2"]:
            r_team[m["t1"]] += RATING_TEAM_RATE * miss
            r_team[m["t2"]] -= RATING_TEAM_RATE * miss
            mean = sum(r_team.values()) / len(r_team)
            for k in list(r_team):
                r_team[k] -= mean

    def rating_now(g, t):
        if g not in r_mu:
            return 0.0, RATING_V0
        day = t.timestamp() / 86400.0
        return r_mu[g], r_var[g] + RATING_DRIFT * max(day - r_last.get(g, day), 0.0)

    rows = []
    j = 0
    settled = [m for m in matches if m["s1"] is not None]
    for m in matches:
        t = m["t"]
        while j < len(settled) and settled[j]["t"] + RESULT_LAG <= t:
            settle(settled[j])
            j += 1
        while team_log and team_log[0][0] < t - dt.timedelta(days=TEAM_DAYS):
            _, team, a, b = team_log.popleft()
            ts = team_sum[team]
            ts[0] -= a
            ts[1] -= b
            ts[2] -= 1
        row = dict(code=m["code"], start=t, p1=m["p1"], p2=m["p2"], t1=m["t1"], t2=m["t2"], s1=m["s1"], s2=m["s2"])
        f1 = side_features(gamers[m["p1"]], m["t1"], t)
        f2 = side_features(gamers[m["p2"]], m["t2"], t)
        row.update({f"p1_{k}": v for k, v in f1.items()})
        row.update({f"p2_{k}": v for k, v in f2.items()})
        mu1, v1 = rating_now(m["p1"], t)
        mu2, v2 = rating_now(m["p2"], t)
        row["p1_rating"], row["p2_rating"] = mu1, mu2
        row["p1_rating_var"], row["p2_rating_var"] = v1, v2
        row["rating_margin"] = mu1 - mu2 + r_team[m["t1"]] - r_team[m["t2"]]
        for side, team in (("p1", m["t1"]), ("p2", m["t2"])):
            ts = team_sum[team]
            row[f"{side}_teamlg_pf"] = ts[0] / ts[2] if ts[2] else math.nan
            row[f"{side}_teamlg_pa"] = ts[1] / ts[2] if ts[2] else math.nan
        key = tuple(sorted((m["p1"], m["p2"])))
        meets = h2h[key]
        sign = 1 if key[0] == m["p1"] else -1
        row["h2h_n"] = len(meets)
        row["h2h_margin"] = sign * sum(x[0] for x in meets) / (len(meets) + H2H_SHRINK) if meets else 0.0
        row["h2h_total"] = _mean([x[1] for x in meets])
        row["h2h_wins_5"] = sum(1 for x in meets[-5:] if sign * x[0] > 0) if meets else math.nan
        row["hour"] = t.hour
        row["weekday"] = t.weekday()
        row["stream"] = _num(m["stream"])
        for w in ("5", "15", "50", "all"):
            e1 = (row[f"p1_pf_{w}"] + row[f"p2_pa_{w}"]) / 2
            e2 = (row[f"p2_pf_{w}"] + row[f"p1_pa_{w}"]) / 2
            row[f"exp_margin_{w}"], row[f"exp_total_{w}"] = e1 - e2, e1 + e2
        if glmer is not None:
            g = glmer.get(m["code"])
            row["glmer_p1"], row["glmer_p2"] = g if g else (math.nan, math.nan)
            row["glmer_margin"] = row["glmer_p1"] - row["glmer_p2"]
            row["glmer_total"] = row["glmer_p1"] + row["glmer_p2"]
        if drive_table is not None:
            row.update({f"p1_{k}": v for k, v in drive_features(gamers[m["p1"]]).items()})
            row.update({f"p2_{k}": v for k, v in drive_features(gamers[m["p2"]]).items()})
            prod = drive_table.get(m["code"], {}).get("prod", {})
            for k in ("prod_p_home", "prod_spread_line", "prod_spread_prob", "prod_total_line", "prod_total_prob"):
                v = prod.get(k)
                row[k] = math.nan if v is None else v
        rows.append(row)
    return rows


def write(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["start"] = r["start"].strftime("%Y-%m-%d %H:%M:%S")
            w.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in out.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default="eAMFCalibrator/out/match_history.csv")
    ap.add_argument("--glmer", help="glmer/cadence.R's cadence_matches.csv for one walk-forward arm")
    ap.add_argument("--drives", help="xgb/drives.py's table (drive and snap stats, prod's pre-match quotes)")
    ap.add_argument("--out", default="xgb/out/features.csv")
    a = ap.parse_args(argv)
    matches = read_history(a.history)
    rows = build(matches, read_glmer(a.glmer) if a.glmer else None, read_drives(a.drives) if a.drives else None)
    write(rows, a.out)
    print(f"{len(rows):,} matches, {len(rows[0]) - 8} features -> {a.out}")


if __name__ == "__main__":
    main()
