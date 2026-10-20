"""Each match's drive and snap stats per side, off the scouting export: a few MB from a file of a
gigabyte or more, for xgb/features.py to read.

    python -m xgb.drives --export eAMFCalibrator/out_scouting_xgb/scouting_playover.csv \
        --out xgb/data/drive_stats.csv

The export has one row per PLAY_OVER, in message order, each with the state the play was snapped
from (offence, down, distance, field position: yards from the offence's own goal line) and the kind
of play it was (eAMFCalibrator/scouting.py's classify_play). Read here:

  snaps    every scrimmage snap (SCRIMMAGE, or a TURNOVER_ON_DOWNS on 4th down), against the next
           play: its gain (a touchdown gains the rest of the field), whether it succeeded (the line
           to gain or a touchdown), 20+ and 40+ yard gains, losses, turnovers (the other side has
           the next snap, not off a punt, a field goal or a 4th down; or a defensive touchdown), on
           3rd and 4th down, and the clock it used
  drives   each side's runs of plays with the ball, ended by a touchdown, a field goal (made or
           missed), a punt, a turnover on downs, a turnover, or the half: their outcomes, red-zone
           trips (a snap from the opponent's 20 or closer) and touchdowns from them, the plays
           they took, where they started and the points they scored

Scores are read off the scoreboard, not the play's messages: the feed's touchdown message can land
after the play's PLAY_OVER, which files the play as a scrimmage one. A snap is a touchdown when its
side's score has risen 6 or more by the next snap; a drive scored what its side's score rose from
its first snap to the next possession (a conversion included).
  prod     prod's last pre-match quotes (moneyline, spread and total), as the export carries them

Counts and sums only, so features.py can turn any window of matches into rates.
"""

import argparse
import csv
import os
import sys
from collections import Counter

KEEP = ("match_code", "message", "period", "clock_seconds", "play_kind", "team_a_side", "offense", "down",
        "distance", "field_position", "play_messages", "final_p1", "final_p2", "home_handle", "away_handle",
        "score_p1", "score_p2", "score_p1_at_start", "score_p2_at_start",
        "prematch_line_52", "prematch_prob_52", "prematch_line_54", "prematch_prob_54", "prematch_prob_50")
POSSESSION_KINDS = ("SCRIMMAGE", "TURNOVER_ON_DOWNS", "FIELD_GOAL", "PUNT", "TOUCHDOWN")
SNAP_KINDS = ("SCRIMMAGE", "TURNOVER_ON_DOWNS")
RED_ZONE = 80
STATS = ("drives", "td", "fg", "fg_miss", "punt", "downs", "turnover", "half_end", "rz_trips", "rz_td",
         "drive_plays", "start_field", "snaps", "gains", "yards", "success", "big20", "big40", "loss",
         "third_n", "third_conv", "fourth_n", "fourth_conv", "seconds", "drive_pts")


def _i(v):
    try:
        return None if v in ("", None) else int(float(v))
    except ValueError:
        return None


def _f(v):
    try:
        return None if v in ("", None) else float(v)
    except ValueError:
        return None


def _half(period):
    return 1 if period <= 2 else 2 if period <= 4 else period


def _scorer(row, prefix):
    for m in reversed((row.get("play_messages") or "").split("|")):
        if m.startswith(prefix) and m[-6:] in ("TEAM_A", "TEAM_B"):
            return m[-6:]
    return None


def _board(r, a_side, at_start=True):
    """{TEAM_A: score, TEAM_B: score} at a row's PLAY_STARTED (or its PLAY_OVER), or None."""
    p1 = _i(r["score_p1_at_start" if at_start else "score_p1"])
    p2 = _i(r["score_p2_at_start" if at_start else "score_p2"])
    if p1 is None or p2 is None:
        return None
    return {"TEAM_A": p1, "TEAM_B": p2} if a_side == "home" else {"TEAM_A": p2, "TEAM_B": p1}


def _other(team):
    return "TEAM_B" if team == "TEAM_A" else "TEAM_A"


def snaps(rows, out, a_side):
    """Add every scrimmage snap's outcome to out[team]."""
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in SNAP_KINDS or a["offense"] not in out:
            continue
        down, dist, y = _i(a["down"]), _i(a["distance"]), _i(a["field_position"])
        pa, pb = _i(a["period"]), _i(b["period"])
        if down is None or not dist or y is None or pa is None or pb is None or _half(pa) != _half(pb):
            continue
        off, dfn = a["offense"], _other(a["offense"])
        s = out[off]
        sa, sb = _board(a, a_side), _board(b, a_side)
        scored = sb[off] - sa[off] if sa and sb else 0
        conceded = sb[dfn] - sa[dfn] if sa and sb else 0
        bk = b["play_kind"]
        turnover = False
        if scored >= 6 or (bk == "TOUCHDOWN" and (_scorer(b, "TOUCHDOWN_TEAM") or b["offense"]) == off):
            gain = 100 - y
        elif conceded >= 6 or bk == "TOUCHDOWN":
            turnover, gain = True, None    # a defensive touchdown
        elif bk in POSSESSION_KINDS and pa == pb:
            by = _i(b["field_position"])
            if by is None:
                continue
            if b["offense"] == off:
                gain = by - y
            elif a["play_kind"] == "TURNOVER_ON_DOWNS" or down == 4:
                gain = 100 - by - y        # a failed 4th down: the other side takes over where it ended
            else:
                turnover, gain = True, None
        else:
            continue                       # a kick-off or conversion next with no score: the half
        ca, cb = _f(a["clock_seconds"]), _f(b["clock_seconds"])
        s["snaps"] += 1
        if pa == pb and ca is not None and cb is not None and 0 <= ca - cb <= 60:
            s["seconds"] += ca - cb
        success = False
        if not turnover:
            s["gains"] += 1
            s["yards"] += gain
            success = gain >= dist or gain >= 100 - y
            s["success"] += success
            s["big20"] += gain >= 20
            s["big40"] += gain >= 40
            s["loss"] += gain < 0
        if down == 3:
            s["third_n"] += 1
            s["third_conv"] += success
        elif down == 4:
            s["fourth_n"] += 1
            s["fourth_conv"] += success


def _team_of(r):
    """The side a possession row belongs to: a kick's kicker off its message, else the offence."""
    if r["play_kind"] == "FIELD_GOAL":
        return _scorer(r, "FIELD_GOAL_") or r["offense"]
    if r["play_kind"] == "PUNT":
        return _scorer(r, "PUNT_TEAM") or r["offense"]
    return r["offense"]


def _close(run, s, turnover_next, rows, a_side, finals):
    """Add one drive's outcome to its side's counts. `run` is [(row index, row, team)]."""
    team = run[0][2]
    j, last, _ = run[-1]
    kind = last["play_kind"]
    s["drives"] += 1
    plays = [r for _, r, _ in run if r["play_kind"] in SNAP_KINDS]
    s["drive_plays"] += len(plays)
    start = _i(run[0][1]["field_position"])
    s["start_field"] += start if start is not None else 25
    fields = [_i(r["field_position"]) for r in plays if _i(r["field_position"]) is not None]
    red = bool(fields) and max(fields) >= RED_ZONE
    s["rz_trips"] += red
    # the scoreboard from the drive's first snap to the next play that is not its conversion
    before = _board(run[0][1], a_side)
    k = j + 1
    while k < len(rows) and rows[k]["play_kind"] == "CONVERSION":
        k += 1
    after = _board(rows[k], a_side) if k < len(rows) else finals
    pts = after[team] - before[team] if before and after else 0
    against = after[_other(team)] - before[_other(team)] if before and after else 0
    if not before or not after:            # no scoreboard: the play kinds alone
        if kind == "TOUCHDOWN" and (_scorer(last, "TOUCHDOWN_TEAM") or team) == team:
            pts = 7
        elif kind == "FIELD_GOAL" and "FIELD_GOAL_GOOD" in (last.get("play_messages") or ""):
            pts = 3
        elif kind == "TOUCHDOWN":
            against = 7
    s["drive_pts"] += max(pts, 0)
    if pts >= 6:
        s["td"] += 1
        s["rz_td"] += red
    elif pts >= 3:
        s["fg"] += 1
    elif against >= 6:
        s["turnover"] += 1                 # a defensive touchdown
    elif kind == "FIELD_GOAL":
        s["fg_miss"] += 1
    elif kind == "PUNT":
        s["punt"] += 1
    elif kind == "TURNOVER_ON_DOWNS" or (turnover_next and _i(last["down"]) == 4):
        s["downs"] += 1                    # flagged, or a 4th down the other side took over from
    elif turnover_next:
        s["turnover"] += 1                 # the other side had the next snap of the half
    else:
        s["half_end"] += 1                 # the half ran out (or a safety)


def drives(rows, out, a_side, finals):
    """Add each side's drives to out[team]: runs of its possession rows, each ended by a touchdown, a
    kick, a turnover on downs, the other side's next snap of the half, a kick-off or the half."""
    run, half = [], None

    def close(turnover_next):
        _close(run, out[run[0][2]], turnover_next, rows, a_side, finals)

    for i, r in enumerate(rows):
        kind, period = r["play_kind"], _i(r["period"])
        if kind == "KICKOFF" or period is None:
            if run:
                close(False)
                run = []
            continue
        if kind not in POSSESSION_KINDS:
            continue                       # conversions, safeties
        h = _half(period)
        if kind == "TOUCHDOWN":            # ends the drive in progress, whoever scored
            if run and h == half:
                run.append((i, r, run[0][2]))
                close(False)
            run = []
            continue
        team = _team_of(r)
        if team not in out:
            continue
        if run and (team != run[0][2] or h != half):
            close(team != run[0][2] and h == half)
            run = []
        if not run:
            half = h
        run.append((i, r, team))
        if kind in ("FIELD_GOAL", "PUNT", "TURNOVER_ON_DOWNS"):
            close(False)
            run = []
    if run:
        close(False)


def match_stats(code, rows):
    """One output row for a match, or None when its sides can't be told apart."""
    rows.sort(key=lambda r: _i(r["message"]) or 0)
    a_side = next((r["team_a_side"] for r in rows if r["team_a_side"] in ("home", "away")), None)
    if a_side is None:
        return None
    out = {"TEAM_A": Counter(dict.fromkeys(STATS, 0)), "TEAM_B": Counter(dict.fromkeys(STATS, 0))}
    home, away = ("TEAM_A", "TEAM_B") if a_side == "home" else ("TEAM_B", "TEAM_A")
    first = rows[0]
    f1, f2 = _i(first.get("final_p1")), _i(first.get("final_p2"))
    finals = {home: f1, away: f2} if f1 is not None and f2 is not None else None
    snaps(rows, out, a_side)
    drives(rows, out, a_side, finals)
    row = {"match_code": code, "home": first.get("home_handle", ""), "away": first.get("away_handle", ""),
           "final_p1": first.get("final_p1", ""), "final_p2": first.get("final_p2", "")}
    for k in ("prematch_line_52", "prematch_prob_52", "prematch_line_54", "prematch_prob_54", "prematch_prob_50"):
        row[k] = next((r[k] for r in rows if r.get(k) not in ("", None)), "")
    for tag, team in (("h", home), ("a", away)):
        for k in STATS:
            row[f"{tag}_{k}"] = out[team][k]
    return row


def run(export, out_path, verbose=True):
    """Stream the export match by match (it is written a match at a time) and write the table."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fields = ["match_code", "home", "away", "final_p1", "final_p2", "prematch_line_52", "prematch_prob_52",
              "prematch_line_54", "prematch_prob_54", "prematch_prob_50"] + \
             [f"{t}_{k}" for t in ("h", "a") for k in STATS]
    league = Counter()
    n = skipped = 0
    seen = set()
    with open(export, newline="", encoding="utf-8") as fh, open(out_path, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        cur_code, cur = None, []

        def flush(code, rows):
            nonlocal n, skipped
            if code in seen:
                print(f"  {code}: rows not together in the export; later ones dropped", file=sys.stderr)
                return
            seen.add(code)
            row = match_stats(code, rows)
            if row is None:
                skipped += 1
                return
            w.writerow(row)
            n += 1
            for t in ("h", "a"):
                for k in STATS:
                    league[k] += row[f"{t}_{k}"]
            league["final"] += (_f(row["final_p1"]) or 0) + (_f(row["final_p2"]) or 0)
            if verbose and n % 2000 == 0:
                print(f"  {n:,} matches", flush=True)

        for r in csv.DictReader(fh):
            code = r["match_code"]
            if code != cur_code:
                if cur_code is not None:
                    flush(cur_code, cur)
                cur_code, cur = code, []
            cur.append({k: r.get(k, "") for k in KEEP})
        if cur_code is not None:
            flush(cur_code, cur)
    games = 2 * max(n, 1)
    d = max(league["drives"], 1)
    sn = max(league["snaps"], 1)
    g = max(league["gains"], 1)
    summary = [f"  {n:,} matches -> {out_path} ({skipped:,} skipped: TEAM_A's side unknown)"]
    summary.append(f"  per side a match: {league['drives'] / games:.2f} drives, {league['snaps'] / games:.1f} snaps")
    summary.append(f"  per drive: touchdown {league['td'] / d:.1%}, field goal {league['fg'] / d:.1%} (missed "
          f"{league['fg_miss'] / d:.1%}), punt {league['punt'] / d:.1%}, turnover {league['turnover'] / d:.1%}, "
          f"downs {league['downs'] / d:.1%}, half over {league['half_end'] / d:.1%}; {league['drive_plays'] / d:.1f} "
          f"plays, starting at the own {league['start_field'] / d:.0f}")
    summary.append(f"  per snap: {league['yards'] / g:.2f} yards, success {league['success'] / sn:.1%}, 20+ yards "
          f"{league['big20'] / sn:.1%}, 40+ {league['big40'] / sn:.1%}, loss {league['loss'] / sn:.1%}; "
          f"3rd down converted {league['third_conv'] / max(league['third_n'], 1):.1%}, 4th "
          f"{league['fourth_conv'] / max(league['fourth_n'], 1):.1%} of {league['fourth_n'] / games:.2f} a side")
    summary.append(f"  red zone: {league['rz_trips'] / d:.1%} of drives, touchdown on "
                   f"{league['rz_td'] / max(league['rz_trips'], 1):.1%}")
    summary.append(f"  points: {league['drive_pts'] / d:.2f} a drive, {league['drive_pts'] / games:.1f} a side "
                   f"from drives, of {league['final'] / games:.1f} on the scoreboard")
    print("\n" + "\n".join(summary))
    with open(os.path.splitext(out_path)[0] + ".txt", "w", encoding="utf-8") as fh:
        fh.write(f"xgb.drives off {export}\n" + "\n".join(summary) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True, help="the scouting export (scouting_playover.csv)")
    ap.add_argument("--out", default="xgb/data/drive_stats.csv")
    a = ap.parse_args(argv)
    run(a.export, a.out)


if __name__ == "__main__":
    main()
