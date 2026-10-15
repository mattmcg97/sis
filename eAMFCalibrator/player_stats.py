"""Every gamer's stats, as h2hggl.com shows them, rebuilt from our own data, and checked against the
site's (analysis/h2hggl_api.py's players.csv).

  results   off the match history (EVENT and the finals), all time: matches played, won, lost,
            drawn, win %, the longest and current winning streaks, the last ten results, points
            for and against
  play      off SCOUTING_FULL, match by match: the PLAY_OVER snapshots (scouting.py, as the models
            read them) taken in pairs -- a snapshot's offence, down, distance and field position
            are the snap, the next snapshot's play kind is what that snap became, and its state is
            where it left the ball (eAMFModel/sim12.snap_records reads them the same way):

              plays            scrimmage snaps (a run or pass: SCRIMMAGE, TOUCHDOWN, TURNOVER_ON_DOWNS,
                               SAFETY)
              yards            the ball's move on each: next spot less this one with the same
                               offence, 100 less the spot on a touchdown, past the turnover on downs;
                               a turnover's play counts with no yards (yards_known: the plays with a
                               yardage). yards_clean leaves out plays with a foul in them
              third / fourth   3rd and 4th downs run (not punted or kicked) and how many made it: a
                               1st down next with the ball kept, or a touchdown
              red zone         drives with a snap inside the opponent's 20 (field position 80+), and
                               how many ended in a touchdown, and in a field goal
              turnover on downs  the snaps that became one, and the TURNOVER_ON_DOWNS_TEAM_X messages
              time of possession  the game clock run while each side had the ball (a kick-off's
                               run is the receiver's)
              punts            punts and their net yards (the receiver's spot)
              touchdowns, two-point tries, fouls  their TEAM_X messages
            A side's yards conceded are its opponent's yards. Field position, down and distance were
            missing from SCOUTING_FULL for a while (about 23 Sep - 9 Oct 2026): those matches add to
            the counts that need no spot (touchdowns, time of possession, two-point tries, fouls)
            and are left out of the rest (detail: the share of snaps with a spot)

Second readings, to say which definition the site uses: a red zone per set of downs inside the 20
(rz_series: a first down starts a new attempt) beside ours per drive, and all turnovers (an
interception or fumble -- the ball changing hands on a scrimmage play other than on downs -- plus the
turnovers on downs) beside the turnovers on downs alone.

Each match is also checked against itself where two sources say the same thing (reconcile_lines):
the points its scoring messages add up to (td 6, extra point 1, two-point try 2, field goal 3,
safety 2, less the undone) against the final score, touchdown messages against touchdown plays,
turnovers on downs as plays against messages, and both sides' possession against the 4 x 240 s,
week by week with the share of snaps carrying a spot. A cached match read before these counts
existed is read again.

Per player the averages are per match, over the matches each stat could be read on. The site's
are career figures; ours cover the window read, so the check is: played, won, lost and the form
exactly (all time, off the history), and the rest by level and rank across the players. Each
stat with two readings (a touchdown-only red zone or any score; fouls by the TEAM_X on the
message or the other side) is checked both ways, to say which the site uses.

One row per match and side goes to player_match_stats.csv, kept between runs: a rerun reads
only the matches it lacks. That table is the input for pre-match features.

    python -m eAMFCalibrator player-stats --check scrapes/h2hggl_api/20261014_132135/players.csv
    python -m eAMFCalibrator player-stats --since 2026-09-01
"""

import csv
import datetime as dt
import json
import math
import os
import statistics
from collections import Counter, defaultdict

from . import config, scouting, snowflake_io
from .bet_totals import _start

SCRIM = ("SCRIMMAGE", "TOUCHDOWN", "TURNOVER_ON_DOWNS", "SAFETY")
STATE = ("SCRIMMAGE", "KICKOFF", "PUNT", "TURNOVER_ON_DOWNS", "FIELD_GOAL")
RED_ZONE = 80
RECHECK_DAYS = 3             # a cached match without SCOUTING_FULL is read again while this recent
SAVE_EVERY = 20              # chunks between saves of the cache
QUARTER = 240.0
SIDES = ("home", "away")
COUNTS = ("snaps", "detail_snaps", "plays", "yards_known", "yards", "clean_plays", "yards_clean",
          "third_att", "third_conv", "fourth_att", "fourth_conv", "tod", "tod_msgs", "rz_trips",
          "rz_td", "rz_fg", "top_s", "punts", "punt_yards", "punts_spotted", "tds", "twopt_att",
          "twopt_good", "fouls", "fouls_other", "turnovers", "td_plays", "points_msgs", "rz_series",
          "rz_series_td")
# counts a cached row must carry: one without them (read before they were added) is read again
REREAD_IF_MISSING = ("points_msgs", "rz_series")
# points off the scoring messages, by message (without its team), for the check against the final
POINTS = (("TOUCHDOWN", 6), ("EXTRA_POINT_GOOD", 1), ("TWO_POINT_CONVERSION_SUCCESSFUL", 2),
          ("FIELD_GOAL_GOOD", 3), ("SAFETY_AWARDED", 2), ("UNDO_TOUCHDOWN", -6), ("UNDO_EXTRA_POINT", -1),
          ("UNDO_TWO_POINT_CONVERSION", -2), ("UNDO_FIELD_GOAL", -3))
GAME_S = 4 * 240.0
CACHE = "player_match_stats.csv"
FIELDS = ("match_code", "start", "side", "player", "opponent", "team", "opp_team", "points",
          "points_against", "result", "scouting") + COUNTS + ("opp_yards", "opp_yards_known")


def _i(v):
    try:
        return None if v is None or v == "" else int(float(v))
    except (TypeError, ValueError):
        return None


def _f(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def other(side):
    return "away" if side == "home" else "home"


def side_of(team, a_side):
    """TEAM_A / TEAM_B as home / away, given TEAM_A's side."""
    if team not in ("TEAM_A", "TEAM_B") or a_side not in SIDES:
        return None
    return a_side if team == "TEAM_A" else other(a_side)


def messages_of(snap):
    return [m for m in (snap.get("play_messages") or "").split("|") if m]


def team_in(messages, prefix):
    """The TEAM_X of the first message starting with prefix, or None."""
    for m in messages:
        if m.startswith(prefix) and m[-6:] in ("TEAM_A", "TEAM_B"):
            return m[-6:]
    return None


def message_counts(rows):
    """{(kind without its team, TEAM_X): count} of a match's in-play messages, and the kinds seen."""
    out, kinds = Counter(), Counter()
    for r in rows:
        m = scouting._text(r[4])
        if not m:
            continue
        team = m[-6:] if m[-6:] in ("TEAM_A", "TEAM_B") else None
        base = m[:-7] if team else m
        kinds[base] += 1
        if team:
            out[(base, team)] += 1
    return out, kinds


def play_stats(snaps, a_side):
    """{side: Counter} of one match off its PLAY_OVER snapshots (in message order)."""
    c = {s: Counter() for s in SIDES}
    drive = None                                      # [side, reached the red zone]
    series = None                                     # the set of downs under way: [side, in the red zone]

    def end_series(outcome=None):
        nonlocal series
        if series and series[1]:
            c[series[0]]["rz_series"] += 1
            c[series[0]]["rz_series_td"] += outcome == "td"
        series = None

    def close(outcome=None):
        nonlocal drive
        end_series(outcome)
        if drive and drive[1]:
            c[drive[0]]["rz_trips"] += 1
            c[drive[0]]["rz_td"] += outcome == "td"
            c[drive[0]]["rz_fg"] += outcome == "fg"
        drive = None

    for a, b in zip(snaps, snaps[1:]):
        kind, mine = b.get("play_kind"), side_of(a.get("offense"), a_side)
        if a.get("play_kind") not in STATE or mine is None:
            if kind in ("KICKOFF", "CONVERSION"):
                close()
            continue
        y, down = _i(a.get("field_position")), _i(a.get("down"))
        by, b_side = _i(b.get("field_position")), side_of(b.get("offense"), a_side)
        msgs = messages_of(b)
        if _i(a.get("period")) != _i(b.get("period")) and _i(b.get("period")) in (3, 5):
            close()
        if kind in SCRIM:
            k = c[mine]
            k["snaps"] += 1
            if kind == "TOUCHDOWN":                   # to the scorer: a defensive score is the other side's
                c[side_of(team_in(msgs, "TOUCHDOWN_TEAM"), a_side) or mine]["td_plays"] += 1
            if y is None or down is None:
                continue
            k["detail_snaps"] += 1
            k["plays"] += 1
            if drive is None or drive[0] != mine:
                close()
                drive = [mine, False]
            drive[1] = drive[1] or y >= RED_ZONE
            if series is None or series[0] != mine or down == 1:     # a first down starts a new set
                end_series()
                series = [mine, False]
            series[1] = series[1] or y >= RED_ZONE
            scorer = side_of(team_in(msgs, "TOUCHDOWN_TEAM"), a_side) if kind == "TOUCHDOWN" else None
            kept = b_side == mine and by is not None and kind != "TOUCHDOWN"
            if kind == "TOUCHDOWN" and scorer in (mine, None):
                gain = 100 - y
            elif kind == "TURNOVER_ON_DOWNS" and by is not None:
                gain = 100 - by - y
            elif kind == "SAFETY":
                gain = -y
            elif kept:
                gain = by - y
            else:
                gain = None                           # an interception, a fumble, a defensive score
                k["turnovers"] += kind == "TOUCHDOWN" or b_side not in (mine, None)
            if gain is not None and -100 < gain <= 100:
                k["yards_known"] += 1
                k["yards"] += gain
                if not any(m.startswith("FOUL") for m in msgs):
                    k["clean_plays"] += 1
                    k["yards_clean"] += gain
            made = (kind == "TOUCHDOWN" and scorer in (mine, None)) or (kept and _i(b.get("down")) == 1)
            if down == 3:
                k["third_att"] += 1
                k["third_conv"] += made
            elif down == 4:
                k["fourth_att"] += 1
                k["fourth_conv"] += made
            k["tod"] += kind == "TURNOVER_ON_DOWNS"
            if kind == "TOUCHDOWN":
                close("td" if scorer in (mine, None) else None)
            elif not kept:
                close()
        elif kind == "FIELD_GOAL":
            if drive is None or drive[0] != mine:
                close()
                drive = [mine, False]
            drive[1] = drive[1] or (y is not None and y >= RED_ZONE)
            close("fg" if any(m.startswith("FIELD_GOAL_GOOD") for m in msgs) else None)
        elif kind == "PUNT":
            close()
            c[mine]["punts"] += 1
            if y is not None and by is not None and b_side == other(mine):
                c[mine]["punts_spotted"] += 1
                c[mine]["punt_yards"] += 100 - by - y
        elif kind in ("KICKOFF", "CONVERSION"):
            close()
    close()
    for side, seconds in possession(snaps, a_side).items():
        c[side]["top_s"] += seconds
    return c


def possession(snaps, a_side):
    """{side: game-clock seconds with the ball}: the clock each play ran, to the side that had it
    (the offence at the snap; a kick-off's receiver), and what was left of a quarter after its last
    play to the side with the ball then."""
    out = Counter()
    period = clock = holder = None
    for s in snaps:
        p, t = _i(s.get("period")), _f(s.get("clock_seconds"))
        if p is None or t is None:
            continue
        if p != period:
            if holder and clock is not None and period is not None and period <= 4:
                out[holder] += max(0.0, min(QUARTER, clock))
            period, clock = p, QUARTER
        side = side_of(s.get("offense"), a_side)
        who = side if s.get("play_kind") == "KICKOFF" else (holder or side)
        if who and p <= 4 and 0 <= clock - t <= QUARTER:
            out[who] += clock - t
        clock, holder = t, side or holder
    if holder and clock is not None and period is not None and period <= 4:
        out[holder] += max(0.0, min(QUARTER, clock))
    return out


def match_rows(match, info, rows, scores, final):
    """The match's two rows (home, away): who, the result, and every count."""
    f1, f2 = _i(final[0]) if final else None, _i(final[1]) if final else None
    a_side = scouting.team_a_side(rows, scores) if rows else None
    base = {s: dict(match_code=match, start=info.get("SCHEDULED_START_TIME_UTC"), side=s) for s in SIDES}
    for s, n, o in (("home", "1", "2"), ("away", "2", "1")):
        mine, theirs = (f1, f2) if s == "home" else (f2, f1)
        base[s].update(player=info.get(f"PLAYER_{n}_HANDLE"), opponent=info.get(f"PLAYER_{o}_HANDLE"),
                       team=info.get(f"PLAYER_{n}_TEAM"), opp_team=info.get(f"PLAYER_{o}_TEAM"),
                       points=mine, points_against=theirs,
                       result=None if mine is None or theirs is None
                       else "w" if mine > theirs else "l" if mine < theirs else "d",
                       scouting=int(bool(rows) and a_side in SIDES))
    if not rows or a_side not in SIDES:
        return [base[s] for s in SIDES]
    snaps, _ = scouting.snapshots_for_match(match, rows, scores, final, {}, {}, require_quote=False)
    counts = play_stats(snaps, a_side)
    msgs, _ = message_counts(rows)
    for team in ("TEAM_A", "TEAM_B"):
        s = side_of(team, a_side)
        k = counts[s]
        k["tds"] += msgs[("TOUCHDOWN", team)] - msgs[("UNDO_TOUCHDOWN", team)]
        k["points_msgs"] += sum(v * msgs[(kind, team)] for kind, v in POINTS)
        k["twopt_good"] += msgs[("TWO_POINT_CONVERSION_SUCCESSFUL", team)]
        k["twopt_att"] += msgs[("TWO_POINT_CONVERSION_SUCCESSFUL", team)] + msgs[("TWO_POINT_CONVERSION_FAILED", team)]
        k["tod_msgs"] += msgs[("TURNOVER_ON_DOWNS", team)]
        fouls = sum(n for (kind, t), n in msgs.items() if kind.startswith("FOUL") and t == team)
        k["fouls"] += fouls
        counts[other(s)]["fouls_other"] += fouls
    for s in SIDES:
        base[s].update({k: counts[s][k] for k in COUNTS})
        base[s]["opp_yards"] = counts[other(s)]["yards"]
        base[s]["opp_yards_known"] = counts[other(s)]["yards_known"]
    return [base[s] for s in SIDES]


# --- the players -----------------------------------------------------------------------------

def results(history):
    """{player: dict} off the match history, all time: played, won, lost, drawn, win %, streaks,
    points for and against, the last ten results (newest first)."""
    by = defaultdict(list)
    for r in history:
        t = _start(r.get("SCHEDULED_START_TIME_UTC"))
        f1, f2 = _i(r.get("PLAYER_1_FINAL_SCORE")), _i(r.get("PLAYER_2_FINAL_SCORE"))
        if t is None or f1 is None or f2 is None:
            continue
        for n, mine, theirs in (("1", f1, f2), ("2", f2, f1)):
            g = r.get(f"PLAYER_{n}_HANDLE")
            if g:
                by[g.strip().upper()].append((t, mine, theirs))
    out = {}
    for g, ms in by.items():
        ms.sort()
        res = ["w" if a > b else "l" if a < b else "d" for _, a, b in ms]
        longest = run = 0
        for x in res:
            run = run + 1 if x == "w" else 0
            longest = max(longest, run)
        n = len(ms)
        won, lost = res.count("w"), res.count("l")
        out[g] = dict(matchesPlayed=n, matchesWon=won, matchesLost=lost, matchesDrawn=n - won - lost,
                      matchesWinPct=round(100 * won / n), matchLongestWinningStreak=longest,
                      matchWinningStreak=run, avgPoints=sum(a for _, a, _ in ms) / n,
                      avgPointsAgainst=sum(b for _, _, b in ms) / n, form=res[::-1][:10],
                      first=ms[0][0], last=ms[-1][0])
    return out


def _ratio(a, b, scale=1.0):
    return scale * a / b if b else None


def player_stats(rows):
    """{player: dict} of the site's stats off the match rows, and the second readings."""
    by = defaultdict(list)
    for r in rows:
        if r.get("player") and _i(r.get("scouting")):
            by[r["player"].strip().upper()].append(r)
    out = {}
    for g, rs in by.items():
        tot = Counter()
        for r in rs:
            for k in COUNTS + ("opp_yards", "opp_yards_known", "points", "points_against"):
                tot[k] += _f(r.get(k)) or 0
        n = len(rs)
        detail = [r for r in rs if (_f(r.get("snaps")) or 0) and (_f(r.get("detail_snaps")) or 0) >= 0.8 * _f(r["snaps"])]
        nd = len(detail)
        dt_ = Counter()
        for r in detail:
            for k in COUNTS + ("opp_yards", "opp_yards_known"):
                dt_[k] += _f(r.get(k)) or 0
        out[g] = dict(
            matches_scouted=n, matches_detail=nd,
            avgPoints_scouted=_ratio(tot["points"], n), avgTouchdowns=_ratio(tot["tds"], n),
            redZonEfficiencyPct=_ratio(dt_["rz_td"], dt_["rz_trips"], 100),
            redZone_any_score=_ratio(dt_["rz_td"] + dt_["rz_fg"], dt_["rz_trips"], 100),
            thirdDownConversionPct=_ratio(dt_["third_conv"], dt_["third_att"], 100),
            fourthDownConversionPct=_ratio(dt_["fourth_conv"], dt_["fourth_att"], 100),
            avgYardsPlay=_ratio(dt_["yards"], dt_["yards_known"]),
            yardsPlay_all_plays=_ratio(dt_["yards"], dt_["plays"]),
            yardsPlay_clean=_ratio(dt_["yards_clean"], dt_["clean_plays"]),
            avgOffensiveYards=_ratio(dt_["yards"], nd), offensiveYards_clean=_ratio(dt_["yards_clean"], nd),
            avgTimeOfPossession=_ratio(tot["top_s"], n * 60),
            twoPtConversionPct=_ratio(tot["twopt_good"], tot["twopt_att"], 100),
            avgPuntYards=_ratio(dt_["punt_yards"], nd), puntYards_per_punt=_ratio(dt_["punt_yards"], dt_["punts_spotted"]),
            avgYardsConceded=_ratio(dt_["opp_yards"], nd),
            avgPenaltiesConceded=_ratio(tot["fouls"], n), penalties_other_side=_ratio(tot["fouls_other"], n),
            avgTurnoverOnDowns=_ratio(dt_["tod"], nd), turnoverOnDowns_msgs=_ratio(tot["tod_msgs"], n),
            turnovers_all=_ratio(dt_["turnovers"] + dt_["tod"], nd),
            redZone_per_series=_ratio(dt_["rz_series_td"], dt_["rz_series"], 100),
            avgPlays=_ratio(dt_["plays"], nd), avgTurnovers=_ratio(dt_["turnovers"], nd))
    return out


# the site's field -> our readings of it (the first ours; the rest other definitions, to say which the
# site uses)
CHECKS = (
    ("matchesPlayed", ("matchesPlayed",)), ("matchesWon", ("matchesWon",)),
    ("matchesLost", ("matchesLost",)), ("matchesWinPct", ("matchesWinPct",)),
    ("avgPoints", ("avgPoints", "avgPoints_scouted")), ("avgPointsAgainst", ("avgPointsAgainst",)),
    ("avgTouchdowns", ("avgTouchdowns",)),
    ("redZonEfficiencyPct", ("redZonEfficiencyPct", "redZone_any_score", "redZone_per_series")),
    ("thirdDownConversionPct", ("thirdDownConversionPct",)),
    ("fourthDownConversionPct", ("fourthDownConversionPct",)),
    ("avgYardsPlay", ("avgYardsPlay", "yardsPlay_clean")),
    ("avgOffensiveYards", ("avgOffensiveYards", "offensiveYards_clean")),
    ("avgTimeOfPossession", ("avgTimeOfPossession",)),
    ("twoPtConversionPct", ("twoPtConversionPct",)),
    ("avgPuntYards", ("avgPuntYards", "puntYards_per_punt")),
    ("avgYardsConceded", ("avgYardsConceded",)),
    ("avgPenaltiesConceded", ("avgPenaltiesConceded", "penalties_other_side")),
    ("avgTurnoverOnDowns", ("avgTurnoverOnDowns", "turnoverOnDowns_msgs", "turnovers_all")),
)


def _corr(pairs):
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy) if sx and sy else None


def read_site(path):
    """{player: the site's row} off h2hggl_api.py's players.csv."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["participantName"].strip().upper(): r for r in csv.DictReader(fh) if r.get("participantName")}


def _fmt(x, w=9):
    if x is None:
        return f"{'-':>{w}s}"
    return f"{x:>{w},.2f}" if isinstance(x, float) else f"{x:>{w},}"


def check_lines(site, ours):
    """Lines: field by field, ours against the site's across the players."""
    both = sorted(set(site) & set(ours))
    missing = sorted(set(site) - set(ours))
    L = [f"\n  ==== Against the site: {len(both)} of {len(site)} of its players found here"
         + (f" (missing: {', '.join(missing[:15])}{' ...' if len(missing) > 15 else ''})" if missing else "")
         + " ====",
         "  exact: players where ours equals the site's (to its rounding); corr: across the players; "
         "site / ours: the mean over the players; |diff|: the median gap",
         f"  {'field':26s} {'reading':22s} {'players':>7s} {'exact':>6s} {'corr':>6s} {'site':>9s} "
         f"{'ours':>9s} {'|diff|':>8s}"]
    for field, readings in CHECKS:
        for name in readings:
            pairs = [(_f(site[g].get(field)), ours[g].get(name)) for g in both]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            if not pairs:
                L.append(f"  {field[:26]:26s} {name[:22]:22s} {0:7d}")
                continue
            dec = max((len(str(site[g].get(field)).split(".")[1]) if "." in str(site[g].get(field)) else 0)
                      for g in both)
            exact = sum(1 for a, b in pairs if round(b, dec) == round(a, dec))
            diffs = sorted(abs(a - b) for a, b in pairs)
            L.append(f"  {field[:26]:26s} {name[:22]:22s} {len(pairs):7d} {100 * exact / len(pairs):5.0f}% "
                     f"{_fmt(_corr(pairs), 6)} {_fmt(sum(a for a, _ in pairs) / len(pairs))} "
                     f"{_fmt(sum(b for _, b in pairs) / len(pairs))} {_fmt(statistics.median(diffs), 8)}")
    forms = [(json.loads(site[g]["matchForm"]) if site[g].get("matchForm") else None, ours[g].get("form")) for g in both]
    forms = [(a, b) for a, b in forms if a and b]
    newest = sum(1 for a, b in forms if a == b[:len(a)])
    oldest = sum(1 for a, b in forms if a == b[:len(a)][::-1])
    L.append(f"  {'matchForm':26s} {'last ten, newest first':22s} {len(forms):7d} "
             f"{100 * newest / len(forms) if forms else 0:5.0f}%   (oldest first: "
             f"{100 * oldest / len(forms) if forms else 0:.0f}%)")
    L.append("\n  Player by player (site / ours): played, won, lost, the form, points, and the stats "
             "that need SCOUTING_FULL over the matches read here")
    L.append(f"  {'player':12s} {'played':>11s} {'won':>9s} {'lost':>9s} {'form':>4s} {'points':>11s} "
             f"{'TDs':>9s} {'red zone':>11s} {'3rd down':>11s} {'4th down':>11s} {'yds/play':>9s} "
             f"{'off yds':>13s} {'scouted':>7s}")
    for g in sorted(both, key=lambda g: -(_f(site[g].get("matchesPlayed")) or 0)):
        s, o = site[g], ours[g]
        form = json.loads(s["matchForm"]) if s.get("matchForm") else None
        pair = lambda f, k, d=1: (f"{_f(s.get(f)) or 0:.{d}f}/" + ("-" if o.get(k) is None else f"{o[k]:.{d}f}"))
        L.append(f"  {g[:12]:12s} {s.get('matchesPlayed')}/{o.get('matchesPlayed', '-'):<6} "
                 f"{s.get('matchesWon')}/{o.get('matchesWon', '-'):<4} {s.get('matchesLost')}/{o.get('matchesLost', '-'):<4} "
                 f"{'Y' if form and o.get('form') and form == o['form'][:len(form)] else 'N':>4s} "
                 f"{pair('avgPoints', 'avgPoints'):>11s} {pair('avgTouchdowns', 'avgTouchdowns', 2):>9s} "
                 f"{pair('redZonEfficiencyPct', 'redZonEfficiencyPct'):>11s} "
                 f"{pair('thirdDownConversionPct', 'thirdDownConversionPct'):>11s} "
                 f"{pair('fourthDownConversionPct', 'fourthDownConversionPct'):>11s} "
                 f"{pair('avgYardsPlay', 'avgYardsPlay'):>9s} {pair('avgOffensiveYards', 'avgOffensiveYards'):>13s} "
                 f"{o.get('matches_scouted', 0):>7}")
    return L


# --- our reading against itself ---------------------------------------------------------------

def _week(text):
    t = _start(text)
    if t is None:
        return "unknown"
    y, w, _ = t.isocalendar()
    return f"{y}-W{w:02d}"


def reconcile_lines(rows):
    """Lines: each match read against itself, where two sources say the same thing -- the points the
    scoring messages add up to against the final score, touchdown messages against touchdown plays,
    turnovers on downs as plays against messages, both sides' possession against the game's
    regulation clock -- overall and week by week, with the share of snaps carrying a spot."""
    matches = defaultdict(dict)
    for r in rows:
        if _i(r.get("scouting")) and r.get("points_msgs") not in (None, ""):
            matches[r["match_code"]][r["side"]] = r
    pairs = [m for m in matches.values() if len(m) == 2]
    L = [f"\n  ==== Our reading against itself: {len(pairs):,} matches with SCOUTING_FULL ====",
         "  points: the scoring messages (TD 6, extra point 1, two-point try 2, field goal 3, safety 2, less "
         "the undone) against the final score; TDs: TOUCHDOWN messages against touchdown plays; ToD: "
         "turnovers on downs, plays against messages (matches with a spot on 80%+ of snaps); clock: both "
         "sides' possession within 15 s of the 4 x 240 s; spot: the snaps carrying a field position"]
    if not pairs:
        return L + ["  (none: rerun player-stats to read the matches again with these counts)"]

    def summary(ms):
        sides = [r for m in ms for r in m.values()]
        pts = [(_f(r["points_msgs"]), _f(r.get("points"))) for r in sides if _f(r.get("points")) is not None]
        tds = [(_f(r["tds"]), _f(r["td_plays"])) for r in sides]
        det = [r for r in sides if (_f(r.get("snaps")) or 0) and (_f(r.get("detail_snaps")) or 0) >= 0.8 * _f(r["snaps"])]
        tod = [(_f(r["tod"]), _f(r["tod_msgs"])) for r in det]
        clock = [abs(sum(_f(r["top_s"]) or 0 for r in m.values()) - GAME_S) for m in ms]
        snaps = sum(_f(r.get("snaps")) or 0 for r in sides)
        pct = lambda xs: None if not xs else 100 * sum(1 for a, b in xs if a == b) / len(xs)
        return dict(n=len(ms), pts=pct(pts), tds=pct(tds), tod=pct(tod),
                    clock=100 * sum(1 for x in clock if x <= 15) / len(clock) if clock else None,
                    spot=100 * sum(_f(r.get("detail_snaps")) or 0 for r in sides) / snaps if snaps else None,
                    gaps=Counter(int(a - b) for a, b in pts if a != b))
    p = lambda x: f"{'-':>7s}" if x is None else f"{x:6.1f}%"
    head = f"  {'':12s} {'matches':>8s} {'points':>7s} {'TDs':>7s} {'ToD':>7s} {'clock':>7s} {'spot':>7s}"
    tot = summary(pairs)
    L += [head, f"  {'all':12s} {tot['n']:8,d} {p(tot['pts'])} {p(tot['tds'])} {p(tot['tod'])} {p(tot['clock'])} "
                f"{p(tot['spot'])}"]
    if tot["gaps"]:
        L.append("  points off by (messages less the final): " +
                 ", ".join(f"{g:+d} ({n:,})" for g, n in tot["gaps"].most_common(8)))
    weeks = defaultdict(list)
    for m in pairs:
        weeks[_week(next(iter(m.values())).get("start"))].append(m)
    for w in sorted(weeks):
        x = summary(weeks[w])
        L.append(f"  {w:12s} {x['n']:8,d} {p(x['pts'])} {p(x['tds'])} {p(x['tod'])} {p(x['clock'])} {p(x['spot'])}")
    return L


# --- run -------------------------------------------------------------------------------------

def read_cache(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(fields), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def run(cur, out_dir, since=None, check=None, refresh=False):
    """Read every settled match since `since` (all, by default) that player_match_stats.csv lacks
    off SCOUTING_FULL, add it, then the players' stats and the check against the site. A cached
    match SCOUTING_FULL did not have is read again while it is under RECHECK_DAYS old."""
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, CACHE)
    cached = [] if refresh else read_cache(cache_path)
    recent = dt.datetime.utcnow() - dt.timedelta(days=RECHECK_DAYS)
    done = {r["match_code"] for r in cached
            if (_i(r.get("scouting")) and all(r.get(k) not in (None, "") for k in REREAD_IF_MISSING))
            or (not _i(r.get("scouting")) and (_start(r.get("start")) or recent) < recent)}
    history = snowflake_io.fetch_history(cur)
    info = {r["MATCH_CODE"]: r for r in history}
    start = _start(since) if since else None
    todo = [m for m, r in info.items()
            if m not in done and (start is None or (_start(r.get("SCHEDULED_START_TIME_UTC")) or start) >= start)]
    todo.sort(key=lambda m: str(info[m].get("SCHEDULED_START_TIME_UTC")))
    print(f"  {len(history):,} settled matches in the history; {len(done):,} already in {CACHE}; "
          f"{len(todo):,} to read" + (f" (since {since})" if since else ""), flush=True)
    table = snowflake_io.scouting_table(cur)
    todo_set = set(todo)
    kept = [r for r in cached if r["match_code"] not in todo_set]
    new, kinds = [], Counter()
    chunk = config.MATCH_CHUNK_SIZE
    try:
        for k in range(0, len(todo), chunk):
            batch = todo[k:k + chunk]
            srows = scouting.fetch_scouting(cur, table, batch, windowed=False)
            scores = snowflake_io.fetch_scores(cur, batch)
            by_rows, by_scores = defaultdict(list), defaultdict(list)
            for r in srows:
                if r[1] is not None:
                    by_rows[r[0]].append(r)
            for s in scores:
                by_scores[s[0]].append(s)
            for m in batch:
                rows = sorted(by_rows.get(m, []), key=lambda r: int(r[1]))
                i = info[m]
                final = (i.get("PLAYER_1_FINAL_SCORE"), i.get("PLAYER_2_FINAL_SCORE"))
                new += match_rows(m, i, rows, by_scores.get(m, []), final)
                kinds.update(message_counts(rows)[1])
            print(f"  {min(k + chunk, len(todo)):,} of {len(todo):,} matches read "
                  f"({sum(_i(r['scouting']) or 0 for r in new) // 2:,} with SCOUTING_FULL)", flush=True)
            if (k // chunk + 1) % SAVE_EVERY == 0:
                write_csv(cache_path, kept + new, FIELDS)
    finally:                                          # keep what was read, even on Ctrl+C
        rows = kept + new
        write_csv(cache_path, rows, FIELDS)
    window = [r for r in rows if since is None or (_start(r.get("start")) or start) >= start]
    res = results(history)
    ours = player_stats(window)
    for g, r in res.items():
        ours.setdefault(g, {}).update(r)
    scouted = sum(_i(r.get("scouting")) or 0 for r in window) // 2
    first = min((str(r["start"])[:10] for r in window if _i(r.get("scouting"))), default="-")
    L = [f"  {len(window) // 2:,} matches in the window, {scouted:,} with SCOUTING_FULL (from {first}); "
         f"{len(ours):,} players; results all time off the history"]
    if kinds:
        L.append("  in-play messages read (without their team): " +
                 ", ".join(f"{k} {n:,}" for k, n in kinds.most_common(40)))
    L += reconcile_lines(window)
    names = sorted(ours, key=lambda g: -(ours[g].get("matchesPlayed") or 0))
    fields = list(dict.fromkeys(["player"] + [k for g in names for k in ours[g] if k != "form"] + ["form"]))
    write_csv(os.path.join(out_dir, "player_stats.csv"),
              [dict(ours[g], player=g, form="".join(ours[g].get("form") or [])) for g in names], fields)
    if check:
        site = read_site(check)
        L += check_lines(site, ours)
        side = []
        for g in sorted(set(site) & set(ours)):
            row = {"player": g}
            for field, readings in CHECKS:
                row[f"site_{field}"] = site[g].get(field)
                for name in readings:
                    row[f"ours_{name}"] = ours[g].get(name)
            row["site_matchForm"] = "".join(json.loads(site[g]["matchForm"])) if site[g].get("matchForm") else ""
            row["ours_form"] = "".join(ours[g].get("form") or [])
            row["matches_scouted"] = ours[g].get("matches_scouted")
            row["matches_detail"] = ours[g].get("matches_detail")
            side.append(row)
        write_csv(os.path.join(out_dir, "player_stats_check.csv"), side,
                  list(dict.fromkeys(k for r in side for k in r)))
    print("\n".join(L))
    path = os.path.join(out_dir, "player_stats.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    return path
