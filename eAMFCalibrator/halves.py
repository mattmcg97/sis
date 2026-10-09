"""How a match's scoring splits between its halves, how the second half plays out from each
half-time margin, and how long the touchdowns are -- for the league, each gamer, each NFL team and
each gamer / team pair. Read off a `scouting` export (no Snowflake), the teams off a match history:

    python -m eAMFCalibrator scouting --since 2026-01-01 --no-probe     # every PLAY_OVER, all history
    python -m eAMFCalibrator halves                                      # out/scouting_playover.csv
    python -m eAMFCalibrator halves --history out/match_history.csv     # and the teams

A match's half-time score is the board after its last play of the first half; its second half is
every play after. A touchdown's length is the yards to the goal line when its play started (the
row before it says where the next play starts, `next_field_position`, in yards from the offense's
own goal line); a kick or punt returned, or a turnover run back, is counted apart. Its drive runs
from the first play its side had the ball for.

Writes to --out:

    halves.txt           the league's tables, then each gamer with --min-games or more
    halves_matches.csv   one row a match: each half's points, the half-time margin, the second half's
                         lead changes, both gamers
    halves_tds.csv       one row a touchdown: scorer, half, length, drive plays and seconds, the
                         half-time margin of its match
    halves_gamers.csv    one row a gamer
    halves_teams.csv     one row a team (with --history)
    halves_gamer_teams.csv  one row a gamer / team pair with --min-pair-games or more (with --history)

With the teams, touchdown length is also fitted as gamer + team + opponent + opponent's team, so a
team's effect is its own with the gamers who pick it held level, and a pair is measured against what
its gamer and its team add up to.
"""

import csv
import os
from collections import defaultdict

import numpy as np

QUARTER = 240
MARGIN_BANDS = [("level", 0, 0), ("1-2", 1, 2), ("3-4", 3, 4), ("5-7", 5, 7), ("8-10", 8, 10),
                ("11-14", 11, 14), ("15+", 15, 999)]
LOPSIDED = 0.7                 # the bigger half holds at least this share of the match's points
LONG = (20, 40)                # touchdown plays of at least these yards
PERMUTATIONS = 200
RESTARTS = ("KICKOFF", "CONVERSION", "PUNT", "FIELD_GOAL", "TOUCHDOWN", "SAFETY", "TURNOVER_ON_DOWNS")


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _band(margin):
    m = abs(margin)
    return next(name for name, lo, hi in MARGIN_BANDS if lo <= m <= hi)


def _matches(path):
    """Each match's export rows, in message order, a match at a time."""
    rows, code = [], None
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["match_code"] != code and rows:
                yield code, sorted(rows, key=lambda x: _int(x["message"]) or 0)
                rows = []
            code = r["match_code"]
            rows.append(r)
    if rows:
        yield code, sorted(rows, key=lambda x: _int(x["message"]) or 0)


def _elapsed(period_from, clock_from, period_to, clock_to):
    """Game seconds between two (period, clock left) points."""
    if None in (period_from, clock_from, period_to, clock_to):
        return None
    return (period_to - period_from) * QUARTER + (clock_from - clock_to)


def read_match(code, rows):
    """(match dict, [touchdown dicts]) for one match, or (None, []) when it can't be read."""
    first = rows[0]
    a_side = first.get("team_a_side") or None
    f1, f2 = _int(first.get("final_p1")), _int(first.get("final_p2"))
    home, away = (first.get("home_handle") or "").strip().upper(), (first.get("away_handle") or "").strip().upper()
    if a_side not in ("home", "away") or f1 is None or f2 is None or not home or not away:
        return None, []
    gamer = {"TEAM_A": home if a_side == "home" else away, "TEAM_B": away if a_side == "home" else home}
    side = {"TEAM_A": "home" if a_side == "home" else "away", "TEAM_B": "away" if a_side == "home" else "home"}

    ht = None
    for r in rows:
        p = _int(r["period"])
        if p is not None and p <= 2:
            ht = (_int(r["score_p1"]) or 0, _int(r["score_p2"]) or 0)
    if ht is None or not any((_int(r["period"]) or 0) >= 3 for r in rows) or f1 < ht[0] or f2 < ht[1]:
        return None, []

    # the second half's board, play by play: lead changes and who answered whom
    board = [ht] + [(_int(r["score_p1"]) or 0, _int(r["score_p2"]) or 0)
                    for r in rows if (_int(r["period"]) or 0) >= 3] + [(f1, f2)]
    states = [board[0]]
    for s in board[1:]:
        if s != states[-1] and s[0] >= states[-1][0] and s[1] >= states[-1][1]:
            states.append(s)
    leads = [x for x in (np.sign(a - b) for a, b in states) if x != 0]
    swaps = sum(1 for x, y in zip(leads, leads[1:]) if x != y)

    match = {"match_code": code, "home": home, "away": away, "h1_home": ht[0], "h1_away": ht[1],
             "h2_home": f1 - ht[0], "h2_away": f2 - ht[1], "final_home": f1, "final_away": f2,
             "ht_margin": ht[0] - ht[1], "ht_band": _band(ht[0] - ht[1]), "lead_changes_h2": swaps}

    tds, drive = [], None
    for i, r in enumerate(rows):
        prev = rows[i - 1] if i else None
        start_off = (prev or {}).get("next_offense") or None
        # a drive starts afresh after a score, kick or punt and at half time, not only when the ball
        # changes hands (an onside kick recovered, the side that ended the half receiving the second)
        fresh = (prev is None or prev["play_kind"] in RESTARTS
                 or ((_int(prev["period"]) or 0) <= 2 < (_int(r["period"]) or 0)))
        if start_off and (drive is None or fresh or drive["offense"] != start_off):
            drive = {"offense": start_off, "field": _int(prev.get("next_field_position")),
                     "period": _int(r["period"]), "clock": _int(prev.get("next_start_clock")), "plays": 0}
        if drive is not None and r.get("scrimmage") == "1":
            drive["plays"] += 1
        msgs = (r.get("play_messages") or "").split("|")
        team = next((m[-6:] for m in msgs if m.startswith("TOUCHDOWN_TEAM_")), None)
        if team not in gamer:
            continue
        kind = r["play_kind"]
        start_field = _int((prev or {}).get("next_field_position"))
        if kind in ("KICKOFF", "PUNT"):
            how, length = "return", None
        elif start_off and start_off != team:
            how, length = "defence", None
        else:
            how = "offence"
            length = (100 - start_field) if start_field is not None and 0 < start_field < 100 else None
        period, clock = _int(r["period"]), _int(r["clock_seconds"])
        same_drive = how == "offence" and drive is not None and drive["offense"] == team
        tds.append({
            "match_code": code, "gamer": gamer[team], "opponent": gamer["TEAM_B" if team == "TEAM_A" else "TEAM_A"],
            "side": side[team], "period": period, "half": 1 if (period or 0) <= 2 else 2,
            "how": how, "length": length,
            "drive_yards": (100 - drive["field"]) if same_drive and drive["field"] is not None else None,
            "drive_plays": drive["plays"] if same_drive else None,
            "drive_seconds": _elapsed(drive["period"], drive["clock"], period, clock) if same_drive else None,
            "ht_band": match["ht_band"],
        })
    return match, tds


def _lopsided(h1, h2):
    t = h1 + h2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(t > 0, np.maximum(h1, h2) / np.where(t > 0, t, 1), np.nan)


def halves_table(matches, rng):
    """League lines: how lopsided the halves are against independent halves."""
    h1 = np.array([m["h1_home"] + m["h1_away"] for m in matches], float)
    h2 = np.array([m["h2_home"] + m["h2_away"] for m in matches], float)
    lines = [f"{len(matches):,} matches; first half {h1.mean():.1f} points, second half {h2.mean():.1f}; "
             f"corr(first half, second half) {np.corrcoef(h1, h2)[0, 1]:+.3f}", "",
             "share of matches whose bigger half holds at least ...   real   if the halves were independent"]
    obs = _lopsided(h1, h2)
    nulls = [_lopsided(h1, rng.permutation(h2)) for _ in range(PERMUTATIONS)]
    for c in (0.6, 0.7, 0.8, 0.9):
        nv = [np.nanmean(n >= c) for n in nulls]
        lines.append(f"  {int(100 * c)}% of the points: {100 * np.nanmean(obs >= c):6.1f}%  {100 * np.mean(nv):6.1f}%"
                     f"  ({100 * np.percentile(nv, 2.5):.1f}-{100 * np.percentile(nv, 97.5):.1f})")
    return lines


def margin_table(matches):
    """League lines: the second half from each half-time margin."""
    lines = [f"{'half-time margin':<17}{'matches':>8}{'2nd-half pts':>13}{'lead changes':>13}{'any':>6}{'2+':>6}"
             f"{'trailer won':>12}"]
    for name, _, _ in MARGIN_BANDS:
        ms = [m for m in matches if m["ht_band"] == name]
        if not ms:
            continue
        lc = np.array([m["lead_changes_h2"] for m in ms])
        pts = np.mean([m["h2_home"] + m["h2_away"] for m in ms])
        tr = [np.sign(m["final_home"] - m["final_away"]) == -np.sign(m["ht_margin"]) for m in ms if m["ht_margin"]]
        lines.append(f"{name:<17}{len(ms):>8}{pts:>13.1f}{lc.mean():>13.2f}{100 * (lc > 0).mean():>5.0f}%"
                     f"{100 * (lc >= 2).mean():>5.0f}%{(f'{100 * np.mean(tr):.0f}%' if tr else '-'):>12}")
    return lines


def td_table(tds):
    """League lines: touchdown lengths, by half-time margin of the match and half."""
    lines = [f"{'':<26}{'TDs':>7}{'mean yds':>9}{f'{LONG[0]}+':>6}{f'{LONG[1]}+':>6}{'drive plays':>12}"
             f"{'drive secs':>11}{'returns+def':>12}"]

    def row(name, ts):
        off = [t for t in ts if t["how"] == "offence" and t["length"] is not None]
        if not ts:
            return
        ln = np.array([t["length"] for t in off]) if off else np.array([np.nan])
        dp = [t["drive_plays"] for t in off if t["drive_plays"] is not None]
        ds = [t["drive_seconds"] for t in off if t["drive_seconds"] is not None]
        other = sum(1 for t in ts if t["how"] != "offence")
        lines.append(f"{name:<26}{len(ts):>7}{np.nanmean(ln):>9.1f}{100 * np.nanmean(ln >= LONG[0]):>5.0f}%"
                     f"{100 * np.nanmean(ln >= LONG[1]):>5.0f}%{(np.median(dp) if dp else np.nan):>12.0f}"
                     f"{(np.median(ds) if ds else np.nan):>11.0f}{100 * other / len(ts):>11.0f}%")

    row("all", tds)
    row("first half", [t for t in tds if t["half"] == 1])
    row("second half", [t for t in tds if t["half"] == 2])
    for name, _, _ in MARGIN_BANDS:
        row(f"2nd half, half-time {name}", [t for t in tds if t["half"] == 2 and t["ht_band"] == name])
    return lines


def _side_key(by, m, own):
    """The row a match side counts towards: its gamer, its team, or both."""
    if by == "gamer":
        return m[own]
    team = m.get(f"{own}_team")
    if not team:
        return None
    return team if by == "team" else (m[own], team)


def _td_keys(by, t):
    """(the scorer's row, the conceder's row) for a touchdown."""
    if by == "gamer":
        return t["gamer"], t["opponent"]
    if not t.get("team"):
        return None, None
    if by == "team":
        return t["team"], t["opp_team"]
    return (t["gamer"], t["team"]), (t["opponent"], t["opp_team"])


def side_rows(matches, tds, rng, min_games, by="gamer"):
    """One dict a gamer, team or gamer-and-team (`by`) with min_games or more."""
    games, scored, allowed = defaultdict(list), defaultdict(list), defaultdict(list)
    for m in matches:
        for own in ("home", "away"):
            k = _side_key(by, m, own)
            if k is not None:
                games[k].append((m, own))
    for t in tds:
        k, opp = _td_keys(by, t)
        if k is None:
            continue
        scored[k].append(t)
        if t["how"] == "offence" and t["length"] is not None:
            allowed[opp].append(t["length"])
    out = []
    for g, ms in games.items():
        if len(ms) < min_games:
            continue
        own_h1 = np.array([m[f"h1_{o}"] for m, o in ms], float)
        own_h2 = np.array([m[f"h2_{o}"] for m, o in ms], float)
        opp = {"home": "away", "away": "home"}
        opp_h1 = np.array([m[f"h1_{opp[o]}"] for m, o in ms], float)
        opp_h2 = np.array([m[f"h2_{opp[o]}"] for m, o in ms], float)
        h1, h2 = own_h1 + opp_h1, own_h2 + opp_h2
        lop = _lopsided(h1, h2)
        null = np.mean([np.nanmean(_lopsided(h1, rng.permutation(h2)) >= LOPSIDED) for _ in range(50)])
        # half-time margin 3-4, from this side: + it led
        close = [(m, o) for m, o in ms if 3 <= abs(m["ht_margin"]) <= 4]
        led = [(m, o) for m, o in close if (m["ht_margin"] > 0) == (o == "home")]
        trailed = [(m, o) for m, o in close if (m["ht_margin"] > 0) != (o == "home")]

        def won(group):
            return np.mean([(m["final_home"] - m["final_away"]) * (1 if o == "home" else -1) > 0
                            for m, o in group]) if group else None

        ts = scored.get(g, [])
        off = [t for t in ts if t["how"] == "offence" and t["length"] is not None]
        ln = np.array([t["length"] for t in off], float) if off else np.array([], float)
        late_close = [t["length"] for t in off if t["half"] == 2 and t["ht_band"] in ("1-2", "3-4")]
        against = allowed.get(g, [])
        key = {"gamer": g} if by == "gamer" else {"team": g} if by == "team" else {"gamer": g[0], "team": g[1]}
        out.append({
            **key, "games": len(ms),
            "won": round(float(np.mean(own_h1 + own_h2 > opp_h1 + opp_h2)), 3),
            "own_h1": round(own_h1.mean(), 2), "own_h2": round(own_h2.mean(), 2),
            "opp_h1": round(opp_h1.mean(), 2), "opp_h2": round(opp_h2.mean(), 2),
            "corr_halves": round(float(np.corrcoef(h1, h2)[0, 1]), 3) if h1.std() and h2.std() else None,
            "lopsided": round(float(np.nanmean(lop >= LOPSIDED)), 3), "lopsided_if_independent": round(float(null), 3),
            "ht_close_games": len(close),
            "ht_close_lead_changes": round(float(np.mean([m["lead_changes_h2"] for m, _ in close])), 2) if close else None,
            "won_leading_3_4": None if won(led) is None else round(float(won(led)), 3), "n_leading_3_4": len(led),
            "won_trailing_3_4": None if won(trailed) is None else round(float(won(trailed)), 3),
            "n_trailing_3_4": len(trailed),
            "tds": len(ts), "td_mean_yards": round(float(ln.mean()), 1) if len(ln) else None,
            f"td_{LONG[0]}_plus": round(float((ln >= LONG[0]).mean()), 3) if len(ln) else None,
            f"td_{LONG[1]}_plus": round(float((ln >= LONG[1]).mean()), 3) if len(ln) else None,
            "td_close_2h_mean_yards": round(float(np.mean(late_close)), 1) if late_close else None,
            "n_td_close_2h": len(late_close),
            "td_drive_plays_median": float(np.median([t["drive_plays"] for t in off if t["drive_plays"] is not None]))
            if off else None,
            "returns_and_defence": sum(1 for t in ts if t["how"] != "offence"),
            "td_allowed_mean_yards": round(float(np.mean(against)), 1) if against else None,
            "n_td_allowed": len(against),
        })
    return sorted(out, key=lambda r: -r["games"])


def gamer_rows(matches, tds, rng, min_games):
    """One dict a gamer with min_games or more."""
    return side_rows(matches, tds, rng, min_games, "gamer")


def read_teams(history_path):
    """{match code: (home team, away team)} off a match history (PLAYER_1 is home), with the handles
    to check them against."""
    teams = {}
    with open(history_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            t1, t2 = (r.get("PLAYER_1_TEAM") or "").strip(), (r.get("PLAYER_2_TEAM") or "").strip()
            if t1 and t2:
                teams[r["MATCH_CODE"]] = (t1, t2, (r.get("PLAYER_1_HANDLE") or "").strip().upper(),
                                          (r.get("PLAYER_2_HANDLE") or "").strip().upper())
    return teams


def attach_teams(matches, tds, teams):
    """Give each match its two teams and each touchdown its side's and the other's; returns how many
    matches got them. A match the history lacks, or whose handles it has the other way round, gets ''."""
    found = {}
    for m in matches:
        t = teams.get(m["match_code"])
        if t and (t[2], t[3]) == (m["home"], m["away"]):
            home_team, away_team = t[0], t[1]
        elif t and (t[3], t[2]) == (m["home"], m["away"]):
            home_team, away_team = t[1], t[0]
        else:
            home_team = away_team = ""
        m["home_team"], m["away_team"] = home_team, away_team
        found[m["match_code"]] = (home_team, away_team)
    for t in tds:
        home_team, away_team = found.get(t["match_code"], ("", ""))
        t["team"], t["opp_team"] = (home_team, away_team) if t["side"] == "home" else (away_team, home_team)
    return sum(1 for v in found.values() if v[0])


def td_effects(tds, iterations=200):
    """Offensive touchdown length as league mean + gamer + team + opponent gamer + opponent team,
    fitted by least squares (backfitting). Returns ({factor: {level: yards}}, residual sd, fitted
    length a touchdown); each factor's effects average 0 over the touchdowns."""
    off = [t for t in tds if t["how"] == "offence" and t["length"] is not None and t.get("team")]
    if not off:
        return {}, None, []
    y = np.array([t["length"] for t in off], float)
    factors = {"gamer": "gamer", "team": "team", "opponent": "opponent", "opp_team": "opp_team"}
    codes = {}
    for f, col in factors.items():
        levels = sorted({t[col] for t in off})
        idx = {v: i for i, v in enumerate(levels)}
        codes[f] = (levels, np.array([idx[t[col]] for t in off]))
    eff = {f: np.zeros(len(lv)) for f, (lv, _) in codes.items()}
    mu = y.mean()
    for _ in range(iterations):
        biggest = 0.0
        for f, (lv, ix) in codes.items():
            part = mu + sum(eff[g][codes[g][1]] for g in codes if g != f)
            new = np.bincount(ix, y - part, len(lv)) / np.bincount(ix, minlength=len(lv))
            new -= np.average(new[ix])
            biggest = max(biggest, float(np.abs(new - eff[f]).max()))
            eff[f] = new
        if biggest < 1e-6:
            break
    fitted = mu + sum(eff[f][codes[f][1]] for f in codes)
    k = 1 + sum(len(lv) - 1 for lv, _ in codes.values())
    sd = float(np.sqrt(((y - fitted) ** 2).sum() / max(len(y) - k, 1)))
    out = {f: dict(zip(codes[f][0], eff[f])) for f in codes}
    return out, sd, list(zip(off, fitted))


def team_effect_lines(effects, sd, fitted, cell_rows):
    """Lines: how far a team's touchdowns run with who played it and against whom held level, and how
    much gamer-and-team pairs differ beyond their gamer and their team."""
    if not effects or not sd:
        return []
    t, a, g = effects["team"], effects["opp_team"], effects["gamer"]
    lines = ["touchdown yards against the average, with the gamer, the opponent and the opponent's team "
             "held level (an additive fit over every offensive touchdown):", "",
             f"{'team':<24}{'scoring':>9}{'allowing':>10}"]
    for name in sorted(t, key=lambda x: -t[x]):
        lines.append(f"{name[:23]:<24}{t[name]:>+9.1f}{a.get(name, np.nan):>+10.1f}")
    gv = np.array(list(g.values()))
    lines += ["", f"spread of the effects (sd): team {np.std(list(t.values())):.1f} yds, gamer {gv.std():.1f}, "
              f"a touchdown about its fit {sd:.1f}"]
    # what gamer-and-team cells add: their residuals' spread against what noise alone gives
    cells = defaultdict(list)
    for td, fit in fitted:
        cells[(td["gamer"], td["team"])].append(td["length"] - fit)
    n = np.array([len(v) for v in cells.values()], float)
    ss = sum(len(v) * np.mean(v) ** 2 for v in cells.values())
    df = len(cells) - len(g) - len(t) + 1
    tau2 = max(0.0, (ss - df * sd ** 2) / n.sum())
    lines.append(f"gamer-and-team pairs beyond their gamer plus their team: {len(cells)} pairs, sd {np.sqrt(tau2):.1f} "
                 f"yds (chi2 {ss / sd ** 2:.0f} on {df} df, where noise alone gives about {df})")
    picked = [r for r in cell_rows if r.get("td_vs_expected_z") is not None]
    by_z = sorted(picked, key=lambda r: -r["td_vs_expected_z"])
    for title, rows in (("pairs running longest against their fit", by_z),
                        ("pairs running shortest against their fit", by_z[::-1])):
        lines += ["", f"{title} ({len(picked)} pairs with the games):",
                  f"  {'gamer':<16}{'team':<24}{'TDs':>5}{'TD yds':>8}{'fit':>7}{'z':>7}"]
        for r in rows[:10]:
            lines.append(f"  {r['gamer'][:15]:<16}{r['team'][:23]:<24}{r['n_td_fit']:>5}{r['td_mean_yards']:>8.1f}"
                         f"{r['td_expected_yards']:>7.1f}{r['td_vs_expected_z']:>+7.1f}")
    lines.append(f"  (with {len(picked)} pairs, |z| of 2.5 or more by chance alone: about {0.0124 * len(picked):.0f})")
    return lines


def add_fit_columns(rows, by, effects, sd, fitted):
    """Each row's effect off the additive fit; a gamer-and-team row also its touchdowns' fitted mean."""
    if not effects:
        return
    if by == "gamer":
        for r in rows:
            r["td_gamer_effect"] = _round(effects["gamer"].get(r["gamer"]), 2)
            r["td_allowed_gamer_effect"] = _round(effects["opponent"].get(r["gamer"]), 2)
    elif by == "team":
        for r in rows:
            r["td_team_effect"] = _round(effects["team"].get(r["team"]), 2)
            r["td_allowed_team_effect"] = _round(effects["opp_team"].get(r["team"]), 2)
    else:
        cells = defaultdict(list)
        for td, fit in fitted:
            cells[(td["gamer"], td["team"])].append((td["length"], fit))
        for r in rows:
            c = cells.get((r["gamer"], r["team"]), [])
            r["n_td_fit"] = len(c)
            if c:
                y, f = np.array(c, float).T
                r["td_expected_yards"] = round(float(f.mean()), 1)
                r["td_vs_expected"] = round(float((y - f).mean()), 1)
                r["td_vs_expected_z"] = (round(float((y - f).mean() / (sd / np.sqrt(len(c)))), 2)
                                         if sd > 0 else None)


def _round(v, n):
    return None if v is None else round(float(v), n)


SIDE_FIELDS = ["games", "won", "own_h1", "own_h2", "opp_h1", "opp_h2", "corr_halves", "lopsided",
               "lopsided_if_independent", "ht_close_games", "ht_close_lead_changes", "won_leading_3_4",
               "n_leading_3_4", "won_trailing_3_4", "n_trailing_3_4", "tds", "td_mean_yards",
               f"td_{LONG[0]}_plus", f"td_{LONG[1]}_plus", "td_close_2h_mean_yards", "n_td_close_2h",
               "td_drive_plays_median", "returns_and_defence", "td_allowed_mean_yards", "n_td_allowed"]
GAMER_FIELDS = ["gamer"] + SIDE_FIELDS + ["td_gamer_effect", "td_allowed_gamer_effect"]
TEAM_FIELDS = ["team"] + SIDE_FIELDS + ["td_team_effect", "td_allowed_team_effect"]
GAMER_TEAM_FIELDS = ["gamer", "team"] + SIDE_FIELDS + ["n_td_fit", "td_expected_yards", "td_vs_expected",
                                                       "td_vs_expected_z"]


def side_table(rows, label="gamer"):
    """Lines: each row, the columns that answer the questions. `label`: gamer, team or both."""
    def f(v, fmt, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return format(100 * v, fmt) + "%" if pct else format(v, fmt)
    width = {"gamer": 18, "team": 24, "gamer_team": 38}[label]

    def name(r):
        return r["gamer"] if label == "gamer" else r["team"] if label == "team" else f"{r['gamer']} / {r['team']}"
    head = (f"{label.replace('_', ' / '):<{width}}{'games':>6}{'won':>5}{'own 1H':>7}{'own 2H':>7}{'corr':>6}"
            f"{'lopsided':>9}{'(chance)':>9}{'HT 3-4':>7}{'swaps':>6}{'win up':>7}{'win dn':>7}{'TDs':>5}"
            f"{'TD yds':>7}{'20+':>6}{'40+':>6}{'close 2H yds':>13}{'allowed':>8}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{name(r)[:width - 1]:<{width}}{r['games']:>6}{f(r['won'], '.0f', True):>5}"
                     f"{f(r['own_h1'], '.1f'):>7}{f(r['own_h2'], '.1f'):>7}"
                     f"{f(r['corr_halves'], '+.2f'):>6}{f(r['lopsided'], '.0f', True):>9}"
                     f"{f(r['lopsided_if_independent'], '.0f', True):>9}{r['ht_close_games']:>7}"
                     f"{f(r['ht_close_lead_changes'], '.2f'):>6}{f(r['won_leading_3_4'], '.0f', True):>7}"
                     f"{f(r['won_trailing_3_4'], '.0f', True):>7}{r['tds']:>5}{f(r['td_mean_yards'], '.1f'):>7}"
                     f"{f(r[f'td_{LONG[0]}_plus'], '.0f', True):>6}{f(r[f'td_{LONG[1]}_plus'], '.0f', True):>6}"
                     f"{f(r['td_close_2h_mean_yards'], '.1f'):>13}{f(r['td_allowed_mean_yards'], '.1f'):>8}")
    return lines


SIDE_KEY = ("won: share of games won; own 1H / 2H: points a half; corr: the games' first-half against "
            "second-half points; lopsided: games whose bigger half held 70%+ of the points, and (chance) what "
            "independent halves would give the same games; HT 3-4: games 3-4 apart at half time, swaps: their "
            "second halves' average lead changes, win up / dn: how often won from 3-4 up / down; TD yds: "
            "offensive touchdowns' average length, 20+ / 40+: the share that long; close 2H yds: second-half "
            "touchdowns' average length when it was 1-4 at half time; allowed: the opponents' touchdowns' "
            "average length")


def gamer_table(gamers):
    """Lines: each gamer, the columns that answer the questions."""
    return side_table(gamers, "gamer") + ["", SIDE_KEY]


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(export_path, out_dir, min_games=30, seed=0, history_path=None, min_pair_games=20):
    """Read the export (and the match history's teams, when given), write the files, print the
    summary; return its lines."""
    rng = np.random.default_rng(seed)
    matches, tds = [], []
    for code, rows in _matches(export_path):
        m, t = read_match(code, rows)
        if m is not None:
            matches.append(m)
            tds.extend(t)
    if not matches:
        raise SystemExit(f"{export_path}: no match could be read (it needs team_a_side, the finals and "
                         "both handles: a `scouting` export)")
    with_teams = attach_teams(matches, tds, read_teams(history_path)) if history_path else 0
    effects, sd, fitted = td_effects(tds) if with_teams else ({}, None, [])
    gamers = gamer_rows(matches, tds, rng, min_games)
    add_fit_columns(gamers, "gamer", effects, sd, fitted)
    lines = (["HALVES", ""] + halves_table(matches, rng)
             + ["", "THE SECOND HALF BY HALF-TIME MARGIN", ""] + margin_table(matches)
             + ["", "TOUCHDOWNS (yards: the length of the scoring play)", ""] + td_table(tds)
             + ["", f"GAMERS with {min_games}+ games", ""] + gamer_table(gamers))
    files = ["halves.txt", "halves_matches.csv", "halves_tds.csv", "halves_gamers.csv"]
    teams = pairs = []
    if with_teams:
        teams = side_rows(matches, tds, rng, 1, "team")
        add_fit_columns(teams, "team", effects, sd, fitted)
        pairs = side_rows(matches, tds, rng, min_pair_games, "gamer_team")
        add_fit_columns(pairs, "gamer_team", effects, sd, fitted)
        pairs.sort(key=lambda r: (r["gamer"], -r["games"]))
        lines += (["", f"TEAMS ({with_teams:,} of {len(matches):,} matches have them in {history_path})", ""]
                  + side_table(teams, "team")
                  + ["", "TOUCHDOWN LENGTH: GAMER, TEAM, OR THE PAIR", ""]
                  + team_effect_lines(effects, sd, fitted, pairs)
                  + ["", f"GAMER / TEAM pairs with {min_pair_games}+ games", ""] + side_table(pairs, "gamer_team")
                  + ["", SIDE_KEY + "; fit: the touchdowns' length from the gamer + team + opponent + opponent's "
                     "team fit, z: how far the pair's mean sits from it in standard errors"])
        files += ["halves_teams.csv", "halves_gamer_teams.csv"]
    elif history_path:
        lines += ["", f"TEAMS: no match in {history_path} lined up with the export (MATCH_CODE, PLAYER_1/2_TEAM "
                  "and the handles)"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "halves.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _write(os.path.join(out_dir, "halves_matches.csv"), matches, list(matches[0]))
    if tds:
        _write(os.path.join(out_dir, "halves_tds.csv"), tds, list(tds[0]))
    _write(os.path.join(out_dir, "halves_gamers.csv"), gamers, GAMER_FIELDS)
    if with_teams:
        _write(os.path.join(out_dir, "halves_teams.csv"), teams, TEAM_FIELDS)
        _write(os.path.join(out_dir, "halves_gamer_teams.csv"), pairs, GAMER_TEAM_FIELDS)
    print("\n".join(lines))
    print(f"\n  -> {out_dir}: {', '.join(files)}")
    return lines
