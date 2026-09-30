"""Every timeout in SCOUTING_FULL, with who called it and the game around it.

The PLAY_OVER export keeps only the messages inside a play, so the feed's TIMEOUT_CALLED_TEAM_A/B
messages -- called between plays -- never reached the model. This reads them straight off the
feed: for each call, the quarter and clock, the side that called it and whether that side had the
ball (the offense of the next snap), its lead, how many it had called before in the half, and
what the play before it was -- an incompletion (no gain, the next down), a first down, a gain or
loss in play, or a change of possession -- which says whether the clock was already stopped.

    python -m eAMFCalibrator timeouts --since 2026-08-24     -> out/timeouts.csv, timeouts_summary.txt

The summary is what the model's timeout windows are set from: who calls them (the side with the
ball or without), when (quarter and clock), at what score, and how many a side has left when the
end of each half comes.
"""

import csv
import os
from collections import Counter, defaultdict

from . import scouting, snowflake_io

FIELDS = ["match_code", "message", "period", "clock", "caller", "role", "caller_lead",
          "called_before", "prev_play", "clock_running", "run_since_play", "down", "distance",
          "field_position",
          "last_play_over_clock", "next_snap_clock", "team_a_side"]
CALLED = "TIMEOUT_CALLED_"
RUNNING_SECONDS = 2            # the clock moved at least this much since the play ended


def _lead(team, a_side, p1, p2):
    """A team's lead on the scoreboard, or None when TEAM_A's side is not known."""
    if a_side not in ("home", "away") or team not in ("TEAM_A", "TEAM_B"):
        return None
    home = (team == "TEAM_A") == (a_side == "home")
    return (p1 - p2) if home else (p2 - p1)


def prev_play(started, over):
    """What the last play was, off its PLAY_STARTED and PLAY_OVER rows: "incomplete" (no gain and
    the next down), "first down", "in play" (a gain or loss), "change of possession", or None."""
    if started is None or over is None:
        return None
    o1, o2 = scouting._text(started[5]), scouting._text(over[5])
    d1, d2 = scouting._int(started[6]), scouting._int(over[6])
    f1, f2 = scouting._int(started[8]), scouting._int(over[8])
    if o1 is None or o2 is None or d1 is None or d2 is None or f1 is None or f2 is None:
        return None
    if o1 != o2:
        return "change of possession"
    if d2 == 1 and (d1 != 1 or f2 - f1 >= (scouting._int(started[7]) or 10)):
        return "first down"
    if f2 == f1 and d2 == d1 + 1:
        return "incomplete"
    return "in play"


def timeouts_for_match(match_code, rows, scores):
    """[dict] one per timeout called in the match (rows as scouting.fetch_scouting gives them)."""
    a_side = scouting.team_a_side(rows, scores)
    changes = sorted((s[1], s[5], s[6]) for s in scores if s[1] is not None)
    period = None
    used = Counter()                        # (half, team) -> calls so far
    last_over = last_started = None
    out = []
    for i, r in enumerate(rows):
        status, kind = scouting._text(r[3]), scouting._text(r[4])
        if status in scouting.PERIOD_START:
            period = scouting.PERIOD_START[status]
        if kind == "PLAY_STARTED":
            last_started = r
            continue
        if kind == "PLAY_OVER":
            last_over = r
            continue
        if not kind or not kind.startswith(CALLED):
            continue
        team = kind[len(CALLED):]
        if team not in ("TEAM_A", "TEAM_B") or period is None:
            continue
        half = 1 if period <= 2 else 2 if period <= 4 else period
        nxt = next((x for x in rows[i + 1:] if scouting._text(x[4]) == "PLAY_STARTED"), None)
        offense = scouting._text(nxt[5]) if nxt else (scouting._text(last_over[5]) if last_over else None)
        clock = scouting._int(r[2])
        over_clock = scouting._int(last_over[2]) if last_over is not None else None
        p1, p2 = scouting._score_at(changes, r[1])
        run = (over_clock - clock) if over_clock is not None and clock is not None else None
        out.append({
            "match_code": match_code, "message": r[1], "period": period, "clock": clock,
            "caller": team, "role": None if offense is None else ("offense" if offense == team else "defense"),
            "caller_lead": _lead(team, a_side, p1, p2), "called_before": used[(half, team)],
            "prev_play": prev_play(last_started, last_over),
            "clock_running": None if run is None else int(run >= RUNNING_SECONDS),
            "run_since_play": run,
            "down": scouting._int(last_over[6]) if last_over is not None else None,
            "distance": scouting._int(last_over[7]) if last_over is not None else None,
            "field_position": scouting._int(last_over[8]) if last_over is not None else None,
            "last_play_over_clock": over_clock,
            "next_snap_clock": scouting._int(nxt[2]) if nxt else None, "team_a_side": a_side,
        })
        used[(half, team)] += 1
    return out


def half_counts(match_codes, timeouts, cutoffs=((2, 120), (4, 180), (4, 120), (4, 60))):
    """Per match and side, how many timeouts it had called in the half by each cut-off (quarter,
    clock): {(quarter, clock): Counter(calls -> sides)}."""
    by = defaultdict(list)
    for t in timeouts:
        by[t["match_code"]].append(t)
    out = {c: Counter() for c in cutoffs}
    for code in match_codes:
        for team in ("TEAM_A", "TEAM_B"):
            for q, clock in cutoffs:
                half = 1 if q <= 2 else 2
                n = sum(1 for t in by[code] if t["caller"] == team
                        and (1 if t["period"] <= 2 else 2) == half
                        and (t["period"] < q or (t["period"] == q and (t["clock"] or 0) > clock)))
                out[(q, clock)][min(n, 3)] += 1
    return out


def _state(lead):
    if lead is None:
        return "unknown"
    return ("down 9+" if lead <= -9 else "down 1-8" if lead < 0 else "level" if lead == 0
            else "up 1-8" if lead <= 8 else "up 9+")


def _slice(clock):
    if clock is None:
        return "?"
    for edge, name in ((30, "0:00-0:30"), (60, "0:30-1:00"), (90, "1:00-1:30"), (120, "1:30-2:00"),
                       (180, "2:00-3:00"), (240, "3:00-4:00")):
        if clock <= edge:
            return name
    return "4:00+"


SLICES = ["3:00-4:00", "2:00-3:00", "1:30-2:00", "1:00-1:30", "0:30-1:00", "0:00-0:30"]
STATES = ["down 9+", "down 1-8", "level", "up 1-8", "up 9+"]


def summary(match_codes, timeouts):
    """The tables the windows are set from, as lines of text."""
    lines = []
    say = lines.append
    n = len(match_codes)
    say(f"{len(timeouts):,} timeouts in {n:,} matches ({len(timeouts) / max(1, n):.2f} a match)")
    by_q = Counter(t["period"] for t in timeouts)
    say("by quarter: " + ", ".join(f"Q{q} {by_q[q]:,}" if q <= 4 else f"OT{q - 4} {by_q[q]:,}"
                                  for q in sorted(by_q)))
    for q in (1, 2, 3, 4):
        sub = [t for t in timeouts if t["period"] == q]
        if not sub:
            continue
        say("")
        say(f"Q{q}: by clock and by who called it (share of the quarter's calls), and the play "
            "before it (incomplete: the clock had stopped already)")
        say(f"  {'clock':12s} {'calls':>6s} {'offense':>8s} {'defense':>8s} {'in play':>8s} "
            f"{'1st down':>8s} {'incompl':>8s} {'poss chg':>8s}")
        for sl in SLICES:
            x = [t for t in sub if _slice(t["clock"]) == sl]
            if not x:
                continue
            share = lambda key, val: 100 * sum(t[key] == val for t in x) / len(x)
            say(f"  {sl:12s} {len(x):6d} {share('role', 'offense'):7.0f}% {share('role', 'defense'):7.0f}% "
                f"{share('prev_play', 'in play'):7.0f}% {share('prev_play', 'first down'):7.0f}% "
                f"{share('prev_play', 'incomplete'):7.0f}% {share('prev_play', 'change of possession'):7.0f}%")
        early = [t for t in sub if (t["clock"] or 0) > 240]
        say(f"  (earlier than 4:00: {len(early):,})")
        say(f"  by the caller's score, last 4:00 -- offense / defense calls")
        for st in STATES:
            x = [t for t in sub if _state(t["caller_lead"]) == st and (t["clock"] or 0) <= 240]
            off = sum(t["role"] == "offense" for t in x)
            dfn = sum(t["role"] == "defense" for t in x)
            say(f"    {st:9s} {off:6d} / {dfn:6d}")
    for q in (2, 4):
        say("")
        say(f"Q{q}, last 3:00, calls by role x caller's score x clock (offense | defense)")
        say(f"  {'':9s} " + " ".join(f"{sl:>13s}" for sl in SLICES[1:]))
        for st in STATES:
            cells = []
            for sl in SLICES[1:]:
                x = [t for t in timeouts if t["period"] == q and _state(t["caller_lead"]) == st
                     and _slice(t["clock"]) == sl]
                off = sum(t["role"] == "offense" for t in x)
                dfn = sum(t["role"] == "defense" for t in x)
                cells.append(f"{off:5d} | {dfn:5d}")
            say(f"  {st:9s} " + " ".join(f"{c:>13s}" for c in cells))
    say("")
    say("timeouts a side had already called in the half, at each cut-off (share of sides)")
    for (q, clock), c in half_counts(match_codes, timeouts).items():
        tot = sum(c.values())
        say(f"  Q{q} {clock // 60}:{clock % 60:02d}  " + "  ".join(
            f"{k}{'+' if k == 3 else ''} called {100 * c[k] / max(1, tot):4.1f}%" for k in range(4)))
    return lines


def run(cur, table, out_dir, verbose=True):
    """Fetch every timeout in the window, write timeouts.csv and the summary."""
    match_codes = scouting.scouting_matches(cur, table)
    if verbose:
        print(f"\nTimeouts in {len(match_codes):,} matches")
    timeouts = []
    chunk = 200
    for start in range(0, len(match_codes), chunk):
        batch = match_codes[start:start + chunk]
        if verbose:
            print(f"  chunk {start // chunk + 1}: {len(batch)} matches", flush=True)
        rows = scouting.fetch_scouting(cur, table, batch, windowed=False)
        scores = snowflake_io.fetch_scores(cur, batch)
        by_match, scores_by = defaultdict(list), defaultdict(list)
        for r in rows:
            by_match[r[0]].append(r)
        for s in scores:
            scores_by[s[0]].append(s)
        for code in batch:
            timeouts += timeouts_for_match(code, by_match.get(code, []), scores_by.get(code, []))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "timeouts.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(timeouts)
    lines = summary(match_codes, timeouts)
    spath = os.path.join(out_dir, "timeouts_summary.txt")
    with open(spath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if verbose:
        print("\n" + "\n".join(lines))
        print(f"\n  {len(timeouts):,} timeouts -> {path}\n  summary -> {spath}")
    return timeouts
