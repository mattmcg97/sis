"""Every drive, checked.

The in-drive analysis and the drive census rest on drives.build_drives
reading the play feed. This goes through every drive it builds and flags
the ones that do not make sense, and -- where SCOUTING_FULL is there to
ask -- checks how each drive ended against the play kinds SCOUTING_FULL
records (TOUCHDOWN, FIELD_GOAL, PUNT, TURNOVER_ON_DOWNS, a turnover, the
end of a half), which the play feed does not carry.

    python -m eAMFCalibrator drive-audit          # out/drive_audit.csv

Flags, per drive:
  extra_point_alone   the drive's only points are a PAT or a two: the
                      conversion was split off its touchdown
  odd_points          points that are no drive's score (not 3, 6, 7 or 8)
  fragment            one row, no points, not the match's last drive
  same_side_next      the next drive is the same side's with no score or
                      turnover between -- one possession cut in two
  no_first_down_start the first row is not 1st and 10
  defence_scored      the other side scored while this side had the ball
  scouting_disagrees  SCOUTING_FULL ends the drive differently
and per match:
  points_do_not_add_up  the drives' points differ from the feed's total
"""

import collections

from . import drives, indrive

FLAGS = ["extra_point_alone", "odd_points", "fragment", "same_side_next",
         "no_first_down_start", "defence_scored", "scouting_disagrees"]

FIELDS = ["match_code", "drive_number", "period", "offense", "first_message",
          "last_message", "n_rows", "start", "end_field", "points_for",
          "points_against", "outcome", "ended", "scouting_end", "flags"]

DRIVE_SCORES = (3, 6, 7, 8)


def _side(row):
    """SCOUTING_FULL's offence as the play feed's team label."""
    if row.get("offense") not in ("TEAM_A", "TEAM_B") or row.get("team_a_side") not in ("home", "away"):
        return None
    a_home = row["team_a_side"] == "home"
    return drives.HOME_TEAM if (row["offense"] == "TEAM_A") == a_home else drives.AWAY_TEAM


def scouting_end(rows, offence, after, until):
    """How SCOUTING_FULL says a drive ended: the first ending event after the
    drive's last row (message `after`) and before the next drive's first
    (`until`, None: open). Returns (label, expected outcome) or (None, None)."""
    for r in rows:
        m = int(r["message"])
        if m <= after:
            continue
        if until is not None and m >= until:
            break
        kind = r.get("play_kind")
        side = _side(r)
        if kind == "TOUCHDOWN":
            scorer = side
            # the touchdown row carries the scorer as the offence
            return ("touchdown", indrive.TOUCHDOWN) if scorer == offence \
                else ("defensive touchdown", indrive.POINTS_AGAINST)
        if kind == "FIELD_GOAL":
            good = "FIELD_GOAL_GOOD" in (r.get("play_messages") or "")
            return ("field goal", indrive.FIELD_GOAL) if good else ("missed field goal", indrive.NO_POINTS)
        if kind == "PUNT":
            if "SAFETY" in (r.get("play_messages") or ""):
                return "safety", indrive.POINTS_AGAINST
            return "punt", indrive.NO_POINTS
        if kind == "TURNOVER_ON_DOWNS":
            return "turnover on downs", indrive.NO_POINTS
        if kind == "SCRIMMAGE" and side is not None and side != offence:
            return "turnover", indrive.NO_POINTS
        if kind == "KICKOFF":
            return "end of half", indrive.NO_POINTS
    return None, None


def audit_match(match_code, plays, scores, scouting_rows=None):
    """([row per drive], [match-level issue]) for one match."""
    built = drives.build_drives(match_code, plays, scores)
    outcomes = indrive.drive_outcomes(match_code, plays, scores)
    rows = sorted(scouting_rows or [], key=lambda r: int(r["message"]))
    out = []
    for i, (drive, o) in enumerate(zip(built, outcomes)):
        flags = []
        if o.outcome == indrive.EXTRA_POINT:
            flags.append("extra_point_alone")
        if o.points_for and o.points_for not in DRIVE_SCORES and o.outcome != indrive.EXTRA_POINT:
            flags.append("odd_points")
        if len(drive.plays) == 1 and o.outcome == indrive.NO_POINTS and o.ended != indrive.MATCH_END:
            flags.append("fragment")
        if o.ended == indrive.SAME_TEAM and not o.points_for and not o.points_against:
            flags.append("same_side_next")
        first = drive.first
        if not (first.down_number == 1 and first.distance == drives.FIRST_DOWN_YARDS):
            flags.append("no_first_down_start")
        if o.points_against:
            flags.append("defence_scored")
        label = None
        if rows:
            until = built[i + 1].first.event_message_count if i + 1 < len(built) else None
            label, expected = scouting_end(rows, drive.offensive_team,
                                           drive.last.event_message_count, until)
            if expected is not None and expected != o.outcome and not (
                    expected == indrive.TOUCHDOWN and o.outcome == indrive.TOUCHDOWN):
                flags.append("scouting_disagrees")
        out.append({
            "match_code": match_code, "drive_number": drive.drive_number,
            "period": drive.period_number, "offense": drive.offensive_team,
            "first_message": o.first_message, "last_message": o.last_message,
            "n_rows": o.n_plays,
            "start": f"{first.down_number}&{first.distance} @{first.field_position}",
            "end_field": o.end_field, "points_for": o.points_for,
            "points_against": o.points_against, "outcome": o.outcome, "ended": o.ended,
            "scouting_end": label or "", "flags": "|".join(flags),
        })
    issues = []
    if not indrive.reconcile(outcomes, scores)["ok"]:
        issues.append("points_do_not_add_up")
    return out, issues


def summarise(rows, match_issues, n_matches):
    """Counts by flag and by outcome, for the console."""
    flags = collections.Counter()
    for r in rows:
        for f in filter(None, r["flags"].split("|")):
            flags[f] += 1
    outcomes = collections.Counter(r["outcome"] for r in rows)
    disagree = collections.Counter((r["outcome"], r["scouting_end"]) for r in rows
                                   if "scouting_disagrees" in r["flags"])
    return {"drives": len(rows), "matches": n_matches, "flags": flags,
            "outcomes": outcomes, "disagreements": disagree,
            "match_issues": collections.Counter(i for issues in match_issues.values() for i in issues),
            "flagged_drives": sum(1 for r in rows if r["flags"]),
            "clean_share": (1 - sum(1 for r in rows if r["flags"]) / len(rows)) if rows else None}


def print_summary(summary):
    print(f"\n{'=' * 78}\nDRIVE AUDIT\n{'=' * 78}")
    print(f"  {summary['drives']:,} drives across {summary['matches']:,} matches; "
          f"{summary['flagged_drives']:,} flagged "
          f"({0 if summary['clean_share'] is None else 100 * (1 - summary['clean_share']):.1f}%)")
    print("\n  outcome            drives")
    for outcome, n in summary["outcomes"].most_common():
        print(f"  {outcome:18s} {n:7,}")
    print("\n  flag                   drives")
    for flag in FLAGS:
        print(f"  {flag:22s} {summary['flags'].get(flag, 0):7,}")
    for issue, n in summary["match_issues"].items():
        print(f"  {issue:22s} {n:7,} matches")
    if summary["disagreements"]:
        print("\n  play feed says       SCOUTING_FULL says       drives")
        for (outcome, label), n in summary["disagreements"].most_common(15):
            print(f"  {outcome:20s} {label or '-':24s} {n:6,}")
