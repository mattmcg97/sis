"""A field goal at Q2's buzzer that the feed gave no row in Q2 (v21's build counts 305 such half-end kicks in
2,131 matches: 'kicks at a half's end the feed missed, read off the score after it'): halves and kickoffs
book its 3 points to the second half."""
import csv, os, tempfile
from eAMFCalibrator import halves, kickoffs, scouting

def row(msg, period, clock, kind, offense, score, start=None, nxt_off=None, nxt_field="", messages=""):
    r = {f: "" for f in scouting.EXPORT_FIELDS}
    start = start or score
    r.update(match_code="AF1", message=msg, period=period, clock_seconds=clock, play_kind=kind, offense=offense,
             scrimmage="0" if kind in ("KICKOFF", "CONVERSION") else "1", team_a_side="home",
             score_p1=score[0], score_p2=score[1], score_p1_at_start=start[0], score_p2_at_start=start[1],
             next_offense=nxt_off or offense, next_field_position=nxt_field, play_messages=messages,
             final_p1=10, final_p2=7, home_handle="ALPHA", away_handle="BETA")
    return r

rows = [
    row(10, 1, 236, "KICKOFF", "TEAM_A", (0, 0), nxt_field=25),        # A (home) receives; B kicked off -> B receives Q3
    row(20, 1, 200, "TOUCHDOWN", "TEAM_A", (6, 0), (0, 0), messages="TOUCHDOWN_TEAM_A", nxt_field=98),
    row(21, 1, 200, "CONVERSION", "TEAM_A", (7, 0), (6, 0)),
    row(30, 1, 196, "KICKOFF", "TEAM_B", (7, 0), nxt_field=25),
    row(40, 2, 30, "SCRIMMAGE", "TEAM_B", (7, 0), nxt_field=70),        # B drives late in Q2 ...
    row(50, 2, 5, "SCRIMMAGE", "TEAM_B", (7, 0), nxt_field=75),         # ... and kicks a FG at the buzzer: no Q2 row
    row(60, 3, 236, "KICKOFF", "TEAM_B", (7, 3), nxt_field=25),         # Q3's kick: the board already 7-3
    row(70, 3, 150, "SCRIMMAGE", "TEAM_B", (7, 3), nxt_field=30),
    row(80, 3, 100, "PUNT", "TEAM_A", (7, 3), nxt_field=20),
    row(90, 4, 20, "FIELD_GOAL", "TEAM_A", (10, 3), messages="FIELD_GOAL_GOOD_TEAM_A"),
    row(95, 4, 0, "TOUCHDOWN", "TEAM_B", (10, 9), messages="TOUCHDOWN_TEAM_B"),
    row(96, 4, 0, "CONVERSION", "TEAM_B", (10, 10)),
]
m, tds = halves.read_match("AF1", rows)
print("halves: half time", (m["h1_home"], m["h1_away"]), "second half", (m["h2_home"], m["h2_away"]),
      "ht band", m["ht_band"], " (truth: half time 7-3, band 3-4)")
km = kickoffs.Match("AF1", rows)
h = kickoffs.half_time(km, kickoffs.kicks(km))
print("kickoffs: R", h["r"], "late Q2 points R", h["late_q2_r"], "Q2 points R", h["q2_points_r"],
      "ht_total", h["ht_total"], " (truth: R=away kicked the buzzer FG: 3, 3, 10)")
