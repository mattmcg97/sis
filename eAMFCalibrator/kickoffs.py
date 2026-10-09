"""Kick-offs: how they land, the half-time double (score before the break, then again on the second
half's opening drive), the second quarter's pace, and how prod's lines move on a kick. Read off a
`scouting` export (no Snowflake), the gamers and favourites off a match history:

    python -m eAMFCalibrator kickoffs out/scouting_playover.csv --history out/match_history.csv

The side that kicks off the game receives the second half's kick: it can score late in Q2 and again
straight after the break. For each match this finds that side, its points in Q2's last two minutes
and on its first drive of Q3, and each side's pace in Q2 (scrimmage snaps). Then prod's total and
spread, read at Q2's two-minute mark and at the second half's kick, against what the game made.

Every kick-off is classed by where the receiver starts and the clock it took: a touchback (the 20,
no clock), no landing zone (the 35, no clock), out of bounds (the 40, no clock), returned, onside,
or run back for a touchdown. And where nothing was scored on the kick, prod's total line just before
it (the conversion's quote) against its line on the kick itself: does the line move with who
receives?

Writes to --out: kickoffs.txt, kickoffs_kicks.csv (one row a kick), kickoffs_matches.csv (one row a
match), kickoffs_gamers.csv (one row a gamer).
"""

import csv
import os
from collections import Counter, defaultdict

import numpy as np

QUARTER = 240
LATE = 120                       # Q2's last two minutes
TOUCHBACK, NO_LANDING, OUT, RETURNED, ONSIDE, RETURN_TD = (
    "touchback (20)", "no landing zone (35)", "out of bounds (40)", "returned", "onside", "return TD")
KINDS = (TOUCHBACK, NO_LANDING, OUT, RETURNED, ONSIDE, RETURN_TD)
STARTS = {20: TOUCHBACK, 35: NO_LANDING, 40: OUT}


def _i(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _matches(path):
    """Each match's export rows, in message order."""
    rows = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["match_code"]].append(r)
    for code, rs in rows.items():
        rs.sort(key=lambda r: _i(r["message"]) or 0)
        yield code, rs


def read_history(path):
    """{match: (home handle, home team, away handle, away team)} (PLAYER_1 is home)."""
    out = {}
    if not path:
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["MATCH_CODE"]] = ((r.get("PLAYER_1_HANDLE") or "").strip().upper(),
                                    (r.get("PLAYER_1_TEAM") or "").strip(),
                                    (r.get("PLAYER_2_HANDLE") or "").strip().upper(),
                                    (r.get("PLAYER_2_TEAM") or "").strip())
    return out


class Match:
    """One match's rows read as home / away."""

    def __init__(self, code, rows, history=None):
        self.code, self.rows = code, rows
        a_side = rows[0].get("team_a_side")
        self.ok = a_side in ("home", "away")
        self.home_team = "TEAM_A" if a_side == "home" else "TEAM_B"
        self.final = (_i(rows[0].get("final_p1")), _i(rows[0].get("final_p2")))
        h = (history or {}).get(code)
        self.handles = (h[0], h[2]) if h else ((rows[0].get("home_handle") or "").upper(),
                                              (rows[0].get("away_handle") or "").upper())
        self.teams = (h[1], h[3]) if h else ("", "")
        p50 = _f(rows[0].get("prematch_prob_50"))
        self.favourite = None if p50 is None or abs(p50 - 0.5) < 0.05 else ("home" if p50 > 0.5 else "away")

    def side(self, team):
        """TEAM_A / TEAM_B as home / away."""
        if team not in ("TEAM_A", "TEAM_B"):
            return None
        return "home" if team == self.home_team else "away"

    @staticmethod
    def board(r, start=False):
        sfx = "_at_start" if start else ""
        return (_i(r.get(f"score_p1{sfx}")) or 0, _i(r.get(f"score_p2{sfx}")) or 0)


def _half(period):
    """1 or 2 for the halves, the period itself in overtime: a drive ends at half time, not at Q1's end."""
    return 1 if period <= 2 else 2 if period <= 4 else period


def _pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def _over_prob(r):
    """prod's total line and P(over) at a row, where live."""
    line, p = _f(r.get("line_54")), _f(r.get("prob_54"))
    if line is None or p is None or str(r.get("live_54")) not in ("1", "True", "true"):
        return None
    return line, p


def _spread(r):
    """prod's home spread line (the home margin it asks for) and P(home covers), where live."""
    line, p = _f(r.get("line_52")), _f(r.get("prob_52"))
    if line is None or p is None or str(r.get("live_52")) not in ("1", "True", "true"):
        return None
    return line, p


def kicks(m):
    """Every kick-off of a match as a dict: quarter, clock, kicking and receiving side, start, the
    seconds it took, its kind, the receiver's points on the drive after it, and prod's total line
    on the quote before and on the kick."""
    out, rows = [], m.rows
    for i, r in enumerate(rows):
        if r.get("play_kind") != "KICKOFF":
            continue
        period, clock = _i(r["period"]), _i(r["clock_seconds"])
        receiver = m.side(r.get("next_offense") or r.get("offense"))
        if receiver is None or period is None or clock is None:
            continue
        prev = rows[i - 1] if i else None
        same = prev is not None and _i(prev["period"]) == period
        secs = (_i(prev["clock_seconds"]) - clock) if same and _i(prev["clock_seconds"]) is not None \
            else (QUARTER - clock if period in (1, 3) else None)
        start = _i(r.get("next_field_position"))
        msgs = r.get("play_messages") or ""
        if "TOUCHDOWN_TEAM" in msgs and "UNDO_TOUCHDOWN" not in msgs:
            kind = RETURN_TD
        elif "ONSIDE_KICK" in msgs:
            kind = ONSIDE
        elif secs == 0 and start in STARTS:
            kind = STARTS[start]
        else:
            kind = RETURNED
        # the receiver's points on the drive the kick starts: until the other side has the ball
        before = m.board(r, start=True)
        k = 0 if receiver == "home" else 1
        end = None
        for row in rows[i + 1:]:
            if m.side(row.get("offense")) != receiver or _half(_i(row["period"]) or 0) != _half(period):
                end = m.board(row, start=True)
                break
        if end is None:
            end = m.final if None not in m.final else m.board(rows[-1])
        q_before = _over_prob(prev) if prev is not None and (same or period == 3) else None   # Q3: Q2's last
        if period == 1 and not same:                          # the opening kick: the pre-match quote
            line, p = _f(r.get("prematch_line_54")), _f(r.get("prematch_prob_54"))
            q_before = (line, p) if line is not None and p is not None else None
        q_kick = _over_prob(r)
        out.append({"match": m.code, "period": period, "clock": clock, "receiver": receiver,
                    "kicker_handle": m.handles[1 - k], "receiver_handle": m.handles[k],
                    "start": start, "seconds": secs, "kind": kind,
                    "drive_points": end[k] - before[k],
                    "scored_on_kick": m.board(r) != before,
                    "line_before": q_before[0] if q_before else None,
                    "line_kick": q_kick[0] if q_kick else None,
                    "p_before": q_before[1] if q_before else None,
                    "p_kick": q_kick[1] if q_kick else None,
                    "receiver_favourite": None if m.favourite is None else m.favourite == receiver,
                    "final_total": sum(m.final) if None not in m.final else None})
    return out


def half_time(m, ks):
    """The half-time double for one match: the side receiving the second half's kick (R), its points
    in Q2's last two minutes and on its first drive of Q3, both sides' Q2 snaps and points, and prod's
    total and spread at Q2's two-minute mark and on the second half's kick."""
    second = next((k for k in ks if k["period"] == 3), None)
    if second is None or None in m.final:
        return None
    rows, r_side = m.rows, second["receiver"]
    k = 0 if r_side == "home" else 1
    q2 = [r for r in rows if _i(r["period"]) == 2]
    if not q2:
        return None
    first_q2 = m.board(q2[0], start=True)
    end_q2 = m.board(q2[-1])
    late = next((r for r in q2 if (_i(r["clock_seconds"]) or 0) <= LATE), None)
    at_two = m.board(late, start=True) if late is not None else end_q2
    snaps = Counter(m.side(r.get("offense")) for r in q2 if r.get("scrimmage") == "1")
    late_points = (end_q2[k] - at_two[k], end_q2[1 - k] - at_two[1 - k])
    margin = m.final[0] - m.final[1]
    out = {"match": m.code, "r": r_side, "r_handle": m.handles[k], "k_handle": m.handles[1 - k],
           "r_team": m.teams[k], "r_favourite": None if m.favourite is None else m.favourite == r_side,
           "q2_points_r": end_q2[k] - first_q2[k], "q2_points_k": end_q2[1 - k] - first_q2[1 - k],
           "q2_snaps_r": snaps.get(r_side, 0), "q2_snaps_k": snaps.get("away" if r_side == "home" else "home", 0),
           "late_q2_r": late_points[0], "late_q2_k": late_points[1],
           "q3_first_drive_r": second["drive_points"],
           "double": late_points[0] > 0 and second["drive_points"] > 0,
           "ht_total": sum(end_q2), "final_total": sum(m.final), "final_margin": margin}
    for name, row in (("two_min", late), ("second_kick", next(r for r in rows if _i(r["period"]) == 3))):
        t = _over_prob(row) if row is not None else None
        s = _spread(row) if row is not None else None
        out[f"{name}_line"], out[f"{name}_p_over"] = (t if t else (None, None))
        out[f"{name}_spread"], out[f"{name}_p_home"] = (s if s else (None, None))
    return out


def _rate(xs):
    return (sum(xs) / len(xs)) if xs else None


def _cal(rows, line, prob, real):
    """(n, share over the line, mean priced) over rows with a line, pushes left out."""
    xs = [(r[real] > r[line], r[prob]) for r in rows
          if r.get(line) is not None and r.get(prob) is not None and r[real] != r[line]]
    if not xs:
        return 0, None, None
    return len(xs), sum(o for o, _ in xs) / len(xs), sum(p for _, p in xs) / len(xs)


def _fmt_cal(c):
    n, o, p = c
    return f"{100 * o:5.1f}% vs {100 * p:5.1f}% ({100 * (o - p):+5.1f}, n {n:,})" if n else "-"


def kick_table(all_kicks):
    """Lines: kick kinds, the clock they take and the drive after them, overall and by period."""
    n = len(all_kicks)
    lines = [f"{n:,} kick-offs", "", f"{'kind':<22}{'share':>7}{'n':>8}{'start':>7}{'secs':>6}"
             f"{'drive pts':>10}{'TD drive':>9}   share by quarter Q1 / Q2 / Q3 / Q4"]
    for kind in KINDS:
        ks = [k for k in all_kicks if k["kind"] == kind]
        if not ks:
            continue
        by_q = []
        for q in (1, 2, 3, 4):
            inq = [k for k in all_kicks if k["period"] == q]
            by_q.append(f"{100 * sum(k['kind'] == kind for k in inq) / max(1, len(inq)):4.1f}")
        secs = [k["seconds"] for k in ks if k["seconds"] is not None]
        lines.append(f"{kind:<22}{100 * len(ks) / n:6.1f}%{len(ks):8,}"
                     f"{np.mean([k['start'] for k in ks if k['start'] is not None]):7.1f}"
                     f"{(np.mean(secs) if secs else float('nan')):6.1f}"
                     f"{np.mean([k['drive_points'] for k in ks]):10.2f}"
                     f"{100 * np.mean([k['drive_points'] >= 6 for k in ks]):8.0f}%   " + " / ".join(by_q))
    return lines


def late_table(all_kicks):
    """Lines: kicks in Q2's and Q4's last two minutes against the rest -- their kinds, the clock they
    take and the drive after each kind."""
    late = lambda k: k["period"] in (2, 4) and k["clock"] <= LATE
    lines = [f"{'':<26}{'kicks':>7}{'touchback':>10}{'no landing':>11}{'returned':>9}{'secs':>6}   drive pts:"
             " touchback / no landing / returned"]
    for name, g in (("Q2 and Q4 last two minutes", [k for k in all_kicks if late(k)]),
                    ("the rest", [k for k in all_kicks if not late(k)])):
        if not g:
            continue
        share = lambda kd: 100 * sum(k["kind"] == kd for k in g) / len(g)
        pts = lambda kd: np.mean([k["drive_points"] for k in g if k["kind"] == kd]) if any(
            k["kind"] == kd for k in g) else float("nan")
        secs = [k["seconds"] for k in g if k["seconds"] is not None]
        lines.append(f"{name:<26}{len(g):7,}{share(TOUCHBACK):9.1f}%{share(NO_LANDING):10.1f}%{share(RETURNED):8.1f}%"
                     f"{np.mean(secs):6.2f}   {pts(TOUCHBACK):.2f} / {pts(NO_LANDING):.2f} / {pts(RETURNED):.2f}")
    return lines


def kicker_rows(all_kicks, min_kicks):
    """One dict a kicking gamer: their kicks' kinds, against the league's (a chi-squared on the
    kinds, and each kind's share)."""
    league = Counter(k["kind"] for k in all_kicks)
    total = sum(league.values())
    by = defaultdict(list)
    for k in all_kicks:
        if k["kicker_handle"] and k["kind"] not in (ONSIDE, RETURN_TD):
            by[k["kicker_handle"]].append(k)
    main = (TOUCHBACK, NO_LANDING, OUT, RETURNED)
    base = {kd: league[kd] / sum(league[x] for x in main) for kd in main}
    out = []
    for g, ks in by.items():
        if len(ks) < min_kicks:
            continue
        c = Counter(k["kind"] for k in ks)
        chi2 = sum((c[kd] - len(ks) * base[kd]) ** 2 / (len(ks) * base[kd]) for kd in main if base[kd])
        out.append({"gamer": g, "kicks": len(ks), "chi2": round(chi2, 1),
                    **{kd: round(c[kd] / len(ks), 3) for kd in main},
                    "drive_points_allowed": round(float(np.mean([k["drive_points"] for k in ks])), 2)})
    return sorted(out, key=lambda r: -r["chi2"]), base, total


def gamer_doubles(halves, min_games):
    """One dict a gamer: as the second half's receiver, how often they doubled up and their late-Q2
    and first-Q3-drive points, against when they kicked the second half (their Q2 pace both ways)."""
    as_r, as_k = defaultdict(list), defaultdict(list)
    for h in halves:
        if h["r_handle"]:
            as_r[h["r_handle"]].append(h)
        if h["k_handle"]:
            as_k[h["k_handle"]].append(h)
    league = _rate([h["double"] for h in halves])
    out = []
    for g in set(as_r) | set(as_k):
        r, k = as_r.get(g, []), as_k.get(g, [])
        if len(r) < min_games:
            continue
        rate = _rate([h["double"] for h in r])
        se = np.sqrt(league * (1 - league) / len(r))
        out.append({"gamer": g, "games_r": len(r), "games_k": len(k), "double_rate": round(rate, 3),
                    "z": round((rate - league) / se, 2) if se else None,
                    "late_q2_pts_r": round(float(np.mean([h["late_q2_r"] for h in r])), 2),
                    "late_q2_pts_k": round(float(np.mean([h["late_q2_k"] for h in k])), 2) if k else None,
                    "q3_drive_pts_r": round(float(np.mean([h["q3_first_drive_r"] for h in r])), 2),
                    "q2_snaps_r": round(float(np.mean([h["q2_snaps_r"] for h in r])), 1),
                    "q2_snaps_k": round(float(np.mean([h["q2_snaps_k"] for h in k])), 1) if k else None})
    return sorted(out, key=lambda r: -r["z"] if r["z"] is not None else 0), league


def line_moves(all_kicks):
    """Lines: prod's total line on the kick against its quote just before, where nothing was scored
    on the kick: how often it moves, which way, by who receives, and whether the move helped."""
    ks = [k for k in all_kicks if not k["scored_on_kick"] and k["line_before"] is not None
          and k["line_kick"] is not None and k["final_total"] is not None]
    lines = [f"{len(ks):,} kick-offs with prod's total live just before and on the kick, nothing scored on it"]
    moves = Counter(round(k["line_kick"] - k["line_before"], 1) for k in ks)
    lines.append("  line on the kick less the line before: " + ", ".join(
        f"{d:+g} {100 * c / len(ks):.1f}%" for d, c in sorted(moves.items()) if c / len(ks) >= 0.005))
    groups = [("all", ks),
              ("opening kick (pre-match)", [k for k in ks if k["period"] == 1 and k["clock"] >= 230]),
              ("second half's kick", [k for k in ks if k["period"] == 3 and k["clock"] >= 230]),
              ("after a score", [k for k in ks if not (k["period"] in (1, 3) and k["clock"] >= 230)]),
              ("favourite receives", [k for k in ks if k["receiver_favourite"] is True]),
              ("underdog receives", [k for k in ks if k["receiver_favourite"] is False]),
              ("home receives", [k for k in ks if k["receiver"] == "home"]),
              ("away receives", [k for k in ks if k["receiver"] == "away"])]
    lines.append(f"\n  {'':<24}{'n':>7}{'moved':>7}{'up':>6}{'down':>6}{'mean':>7}   over at the line before"
                 f" / priced  ->  over at the kick's line / priced")
    for name, g in groups:
        if not g:
            continue
        d = np.array([k["line_kick"] - k["line_before"] for k in g])
        before = _cal(g, "line_before", "p_before", "final_total")
        after = _cal(g, "line_kick", "p_kick", "final_total")
        lines.append(f"  {name:<24}{len(g):7,}{100 * np.mean(d != 0):6.1f}%{100 * np.mean(d > 0):5.1f}%"
                     f"{100 * np.mean(d < 0):5.1f}%{d.mean():+7.2f}   {_fmt_cal(before)}  ->  {_fmt_cal(after)}")
    lines.append("\n  " + "by the receiver's start".ljust(24) + f"{'n':>7}{'mean':>7}{'up':>6}{'down':>6}")
    for lo, hi, name in ((0, 19, "inside the 20"), (20, 20, "the 20"), (21, 25, "21-25"), (26, 30, "26-30"),
                         (31, 34, "31-34"), (35, 35, "the 35"), (36, 45, "36-45"), (46, 99, "46 or more")):
        g = [k for k in ks if k["start"] is not None and lo <= k["start"] <= hi]
        if g:
            d = np.array([k["line_kick"] - k["line_before"] for k in g])
            lines.append(f"  {name:<24}{len(g):7,}{d.mean():+7.2f}{100 * np.mean(d > 0):5.1f}%{100 * np.mean(d < 0):5.1f}%")
    moved = [k for k in ks if k["line_kick"] != k["line_before"]]
    if moved:
        before = _cal(moved, "line_before", "p_before", "final_total")
        after = _cal(moved, "line_kick", "p_kick", "final_total")
        lines.append(f"\n  where it moved ({len(moved):,}): over at the old line {_fmt_cal(before)}; at the new "
                     f"{_fmt_cal(after)}")
        for name, sel in (("moved up", [k for k in moved if k["line_kick"] > k["line_before"]]),
                          ("moved down", [k for k in moved if k["line_kick"] < k["line_before"]])):
            if sel:
                lines.append(f"    {name:<10} {len(sel):6,}: old line {_fmt_cal(_cal(sel, 'line_before', 'p_before', 'final_total'))};"
                             f" new {_fmt_cal(_cal(sel, 'line_kick', 'p_kick', 'final_total'))}")
    return lines


def half_table(halves):
    """Lines: the half-time double, Q2's pace and prod's prices around it."""
    n = len(halves)
    lines = [f"{n:,} matches with a second-half kick (R: the side receiving it, which kicked off the game;"
             " K: the other)"]
    dbl = _rate([h["double"] for h in halves])
    lines.append(f"  R scored in Q2's last two minutes {_pct(_rate([h['late_q2_r'] > 0 for h in halves]))}"
                 f" (K {_pct(_rate([h['late_q2_k'] > 0 for h in halves]))}); R scored on its first Q3 drive "
                 f"{_pct(_rate([h['q3_first_drive_r'] > 0 for h in halves]))}; both (the double) {_pct(dbl)}")
    late_r = [h for h in halves if h["late_q2_r"] > 0]
    late_n = [h for h in halves if h["late_q2_r"] == 0]
    lines.append(f"  R's first Q3 drive scores {_pct(_rate([h['q3_first_drive_r'] > 0 for h in late_r]))} after "
                 f"R scored late in Q2, {_pct(_rate([h['q3_first_drive_r'] > 0 for h in late_n]))} after it didn't")
    d = np.array([h["q2_snaps_r"] - h["q2_snaps_k"] for h in halves], float)
    p = np.array([h["q2_points_r"] - h["q2_points_k"] for h in halves], float)
    a = np.array([h["late_q2_r"] > 0 for h in halves], float)
    b = np.array([h["q3_first_drive_r"] > 0 for h in halves], float)
    corr = float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() and b.std() else float("nan")
    lines.append(f"  corr(R scored late in Q2, R scored on its first Q3 drive) {corr:+.3f};"
                 f" R less K in Q2: snaps {d.mean():+.2f} (se {d.std() / np.sqrt(len(d)):.2f}), points {p.mean():+.2f}"
                 f" (se {p.std() / np.sqrt(len(p)):.2f})")
    lines.append(f"  Q2 scrimmage snaps: R {np.mean([h['q2_snaps_r'] for h in halves]):.1f}, K "
                 f"{np.mean([h['q2_snaps_k'] for h in halves]):.1f}; Q2 points: R "
                 f"{np.mean([h['q2_points_r'] for h in halves]):.2f}, K {np.mean([h['q2_points_k'] for h in halves]):.2f};"
                 f" late Q2 points: R {np.mean([h['late_q2_r'] for h in halves]):.2f}, K "
                 f"{np.mean([h['late_q2_k'] for h in halves]):.2f}")
    lines += ["", "  prod's total at Q2's two-minute mark and on the second half's kick: over / priced"]
    for name, g in (("all", halves), ("R the favourite", [h for h in halves if h["r_favourite"] is True]),
                    ("R the underdog", [h for h in halves if h["r_favourite"] is False]),
                    ("pick'em", [h for h in halves if h["r_favourite"] is None])):
        lines.append(f"  {name:<18} two-minute mark {_fmt_cal(_cal(g, 'two_min_line', 'two_min_p_over', 'final_total'))}"
                     f"   second-half kick {_fmt_cal(_cal(g, 'second_kick_line', 'second_kick_p_over', 'final_total'))}")
    lines += ["", "  prod's spread there (home covers / priced)"]
    for name, g in (("all", halves), ("R at home", [h for h in halves if h["r"] == "home"]),
                    ("R away", [h for h in halves if h["r"] == "away"])):
        lines.append(f"  {name:<18} two-minute mark {_fmt_cal(_cal(g, 'two_min_spread', 'two_min_p_home', 'final_margin'))}"
                     f"   second-half kick {_fmt_cal(_cal(g, 'second_kick_spread', 'second_kick_p_home', 'final_margin'))}")
    return lines


def _write(path, rows, fields=None):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields or list(rows[0]), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(export_path, out_dir, history_path=None, min_games=20):
    """Read the export, write the files, print the summary; return its lines."""
    history = read_history(history_path)
    all_kicks, halves = [], []
    for code, rows in _matches(export_path):
        m = Match(code, rows, history)
        if not m.ok:
            continue
        ks = kicks(m)
        all_kicks += ks
        h = half_time(m, ks)
        if h is not None:
            halves.append(h)
    if not all_kicks:
        raise SystemExit(f"{export_path}: no kick-off could be read (a `scouting` export, with team_a_side)")
    kickers, base, _ = kicker_rows(all_kicks, min_games)
    doubles, league = gamer_doubles(halves, min_games)
    lines = (["KICK-OFFS", ""] + kick_table(all_kicks) + [""] + late_table(all_kicks)
             + ["", f"KICKERS with {min_games}+ kicks (chi2 on their kinds against the league's, 3 df: about 3"
                " by chance, 11.3 at 1%)", "",
                f"{'gamer':<16}{'kicks':>6}{'chi2':>7}" + "".join(f"{k.split(' (')[0][:11]:>12}" for k in base)
                + f"{'drive pts':>10}"]
             + [f"{r['gamer'][:15]:<16}{r['kicks']:>6}{r['chi2']:>7.1f}"
                + "".join(f"{100 * r[k]:11.1f}%" for k in base) + f"{r['drive_points_allowed']:>10.2f}"
                for r in kickers]
             + [f"{'league':<16}{'':>6}{'':>7}" + "".join(f"{100 * v:11.1f}%" for v in base.values())]
             + ["", "THE HALF-TIME DOUBLE", ""] + half_table(halves)
             + ["", f"GAMERS receiving the second half, {min_games}+ games (league double {100 * league:.1f}%)", "",
                f"{'gamer':<16}{'as R':>6}{'double':>8}{'z':>6}{'late Q2 pts R / K':>19}{'Q3 drive pts':>13}"
                f"{'Q2 snaps R / K':>16}"]
             + [f"{r['gamer'][:15]:<16}{r['games_r']:>6}{100 * r['double_rate']:7.1f}%{(r['z'] if r['z'] is not None else float('nan')):>+6.1f}"
                f"{r['late_q2_pts_r']:>10.2f} / {r['late_q2_pts_k'] if r['late_q2_pts_k'] is not None else float('nan'):<6.2f}"
                f"{r['q3_drive_pts_r']:>13.2f}{r['q2_snaps_r']:>9.1f} / {r['q2_snaps_k'] if r['q2_snaps_k'] is not None else float('nan'):<5.1f}"
                for r in doubles]
             + ["", "PROD'S TOTAL ON A KICK-OFF", ""] + line_moves(all_kicks))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "kickoffs.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    _write(os.path.join(out_dir, "kickoffs_kicks.csv"), all_kicks)
    _write(os.path.join(out_dir, "kickoffs_matches.csv"), halves)
    _write(os.path.join(out_dir, "kickoffs_gamers.csv"), doubles)
    print("\n".join(lines))
    print(f"\n  -> {out_dir}: kickoffs.txt, kickoffs_kicks.csv, kickoffs_matches.csv, kickoffs_gamers.csv")
    return lines
