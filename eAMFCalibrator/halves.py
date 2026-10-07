"""How a match's scoring splits between its halves, how the second half plays out from each
half-time margin, and how long the touchdowns are -- for the league and for each gamer. Read off
a `scouting` export (no Snowflake):

    python -m eAMFCalibrator scouting --since 2026-01-01 --no-probe     # every PLAY_OVER, all history
    python -m eAMFCalibrator halves                                      # out/scouting_playover.csv

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


def gamer_rows(matches, tds, rng, min_games):
    """One dict a gamer with min_games or more."""
    games, scored = defaultdict(list), defaultdict(list)
    for m in matches:
        for g, own in ((m["home"], "home"), (m["away"], "away")):
            games[g].append((m, own))
    for t in tds:
        scored[t["gamer"]].append(t)
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
        # half-time margin 3-4, from the gamer's side: + they led
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
        out.append({
            "gamer": g, "games": len(ms),
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
        })
    return sorted(out, key=lambda r: -r["games"])


GAMER_FIELDS = ["gamer", "games", "own_h1", "own_h2", "opp_h1", "opp_h2", "corr_halves", "lopsided",
                "lopsided_if_independent", "ht_close_games", "ht_close_lead_changes", "won_leading_3_4",
                "n_leading_3_4", "won_trailing_3_4", "n_trailing_3_4", "tds", "td_mean_yards",
                f"td_{LONG[0]}_plus", f"td_{LONG[1]}_plus", "td_close_2h_mean_yards", "n_td_close_2h",
                "td_drive_plays_median", "returns_and_defence"]


def gamer_table(gamers):
    """Lines: each gamer, the columns that answer the questions."""
    def f(v, fmt, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "-"
        return format(100 * v, fmt) + "%" if pct else format(v, fmt)
    head = (f"{'gamer':<18}{'games':>6}{'own 1H':>7}{'own 2H':>7}{'corr':>6}{'lopsided':>9}{'(chance)':>9}"
            f"{'HT 3-4':>7}{'swaps':>6}{'win up':>7}{'win dn':>7}{'TDs':>5}{'TD yds':>7}{'20+':>6}{'40+':>6}"
            f"{'close 2H yds':>13}")
    lines = [head, "-" * len(head)]
    for r in gamers:
        lines.append(f"{r['gamer'][:17]:<18}{r['games']:>6}{f(r['own_h1'], '.1f'):>7}{f(r['own_h2'], '.1f'):>7}"
                     f"{f(r['corr_halves'], '+.2f'):>6}{f(r['lopsided'], '.0f', True):>9}"
                     f"{f(r['lopsided_if_independent'], '.0f', True):>9}{r['ht_close_games']:>7}"
                     f"{f(r['ht_close_lead_changes'], '.2f'):>6}{f(r['won_leading_3_4'], '.0f', True):>7}"
                     f"{f(r['won_trailing_3_4'], '.0f', True):>7}{r['tds']:>5}{f(r['td_mean_yards'], '.1f'):>7}"
                     f"{f(r[f'td_{LONG[0]}_plus'], '.0f', True):>6}{f(r[f'td_{LONG[1]}_plus'], '.0f', True):>6}"
                     f"{f(r['td_close_2h_mean_yards'], '.1f'):>13}")
    lines += ["", "own 1H / 2H: the gamer's points a half; corr: their games' first-half against second-half "
              "points; lopsided: games whose bigger half held 70%+ of the points, and (chance) what independent "
              "halves would give their games; HT 3-4: games 3-4 apart at half time, swaps: their second halves' "
              "average lead changes, win up / dn: how often they won from 3-4 up / down; TD yds: their "
              "offensive touchdowns' average length, 20+ / 40+: the share that long; close 2H yds: their "
              "second-half touchdowns' average length when it was 1-4 at half time"]
    return lines


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(export_path, out_dir, min_games=30, seed=0):
    """Read the export, write the files, print the summary; return its lines."""
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
    gamers = gamer_rows(matches, tds, rng, min_games)
    lines = (["HALVES", ""] + halves_table(matches, rng)
             + ["", "THE SECOND HALF BY HALF-TIME MARGIN", ""] + margin_table(matches)
             + ["", "TOUCHDOWNS (yards: the length of the scoring play)", ""] + td_table(tds)
             + ["", f"GAMERS with {min_games}+ games", ""] + gamer_table(gamers))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "halves.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _write(os.path.join(out_dir, "halves_matches.csv"), matches, list(matches[0]))
    if tds:
        _write(os.path.join(out_dir, "halves_tds.csv"), tds, list(tds[0]))
    _write(os.path.join(out_dir, "halves_gamers.csv"), gamers, GAMER_FIELDS)
    print("\n".join(lines))
    print(f"\n  -> {out_dir}: halves.txt, halves_matches.csv, halves_tds.csv, halves_gamers.csv")
    return lines
