"""v2.1: a snap simulation that draws each snap's time, then its distance given that time.

    python -m eAMFModel v21-build --until 2026-10-01 --out v21_full --mode full
    python -m eAMFModel v21-report --model v21_full --prior-model v14_model --since 2026-10-01 --until 2026-10-12

v2.1 stands apart from v8-v14. It reads the same snapshot export (PLAY_OVER rows) and nothing of their
fitted tables, and it has no strength: the pre-match expected points are not an input to the draws. The
sim plays from the real data alone, and v21_report sets its output against the expected points, so
where the two part shows where the draws are missing something. Nothing is ever fitted to make them
agree.

Each scrimmage snap is two draws, both from real snaps:

  time       the clock from the end of the play before to the end of this one: the wait for the snap
             (a running clock) and the play itself. Drawn from the real times of snaps alike in
             TIME_FEATURES: what the play before did to the clock (PREV: nothing gained -- mostly an
             incomplete pass --, a short gain, a first down, a loss, a fresh possession, a timeout),
             the part of the game (SPLIT: the half and the two-minute warning), the offense's lead,
             down, distance, field zone, the snaps played so far and the snaps left in the half.
             Bimodal by nature: a stopped clock gives a play's own 5-8 seconds, a running one 25-40,
             and the play before sets the split.
  distance   what came of the snap -- yards gained, a turnover, a defensive touchdown -- drawn from
             the real snaps alike in DIST_FEATURES, the first of them the time drawn (TIME_BUCKETS):
             a snap over in 5 seconds can only have gone so far.

Where too few real snaps share every feature (MIN_RECORDS), a draw falls back level by level to
fewer (TIME_LEVELS, DIST_LEVELS). The gamers and their teams come in one of two ways (the build's
`mode`):
  full       straight from what was seen: each draw takes, half the time, the offense's own real snaps
             (its gamer's where it has MIN_GAMER_RECORDS alike in GAMER_*_LEVELS, else its team's,
             else the league's), and half the time what the defense allowed (the defending gamer's,
             else its team's, else the league's).
  adjusted   the league's draws for every side, each leaned toward the longer or shorter end
             (u -> 1 - (1 - u) ** k, k = exp of the summed tilts) by a tilt for the offense's gamer,
             the defense's gamer and both teams, fitted on their real snaps against what the league's
             draws gave the same states, shrunk toward none by ADJUST_PRIOR snaps.

Bounds: a running clock that runs out before the snap ends the quarter; one that reaches 2:00 of Q2
or Q4 stops there and the snap takes a time from the stopped-clock part of the same draw (its real
snaps of STOPPED_MAX seconds or less); a play under way when time runs out is
finished. Real snaps cut short by either (the quarter ran out on them, or a running clock crossed
2:00) are kept in the time draw as "at least this long": drawn where more time is left than they
took, the draw moves on to a real snap at least as long. The distance stops at either end zone: past
the goal line is a touchdown, behind its own a safety; a real touchdown drawn from further back runs on
with the real chance a gain that long goes further (gain_surv). With field windows (the build's
`field`), the distance draw takes its real snap from its group's snaps near the state's own field
(Bins.near), so a gain made at midfield is not played out near the goal, where the field compresses.

Snaps left in the half (REMAIN) is a core split of both draws: the expected count from a table by
the half's 10-second slice, the clock running or stopped into the snap, both sides' timeouts and the
lead (remain_key), each cell shrunk toward its slice and clock's, that toward its slice's
(remain_table). Real snaps and the sim's read it the same way, off their own state -- never the count
that actually followed. The real games' counts first, then the sim's own from kick-off
(refine_remaining), the draws rebuilt on it each round (REFINE_ROUNDS).

A new set of downs is the next row's down (snap_records): spots are whole yards, so a sneak for inches
shows as a 0-yard gain that moves the chains, and a gain of the yards to go can be spotted short. The
sim takes the drawn real snap's own result where it was at the same distance, else the yards; a down
played again (a penalty) keeps its number.

The decisions (DECISIONS) are each the deciding gamer's: 4th downs (go, field goal, punt), a kick on
downs 1-3 in the half's last minute, kick-offs (returned, touchback, no landing zone, kept), going for
two and the timeouts before a running-clock snap (the defense's, then the offense's). The rules come
first, the same for every gamer: the options real sides never or all but never take in a situation,
learned off the real decisions in their full state (v21_rules, the sweep), are struck out there; with
overtime's own (ot_mask). The rest is drawn from the league's shares among real decisions alike in its
features, leaned by the gamer's own log-odds shift for each option (Choice, fit_leans): fitted on their
real decisions against the league's in the same states (those that took a struck-out option left out)
and shrunk toward none, so it carries into every state. What follows is real data too: the kicks'
fields and clock, field goals made by distance, punts' net yards and safety free kicks; a timeout moves
only the time draw. The game's own end (auto_end): in Q4, the side ahead with the ball whose downs left
at the 40-second play clock outlast the clock, the defense out of timeouts, wins there and then.
Overtime is v14's: over once both sides have had the
ball and one leads, a touchdown that puts the second side ahead ends it at once, and a period's end
decides it only when the side behind has had the ball.
"""

import datetime as dt
import math
import pickle
from bisect import bisect_left
from collections import defaultdict

import numpy as np

from . import sim14 as s14

QUARTER = 240.0
LATE = 120.0
PLAY_SECONDS = 7.0           # how long a play takes once snapped (the feed's stopped-clock median)
STOPPED_MAX = 15.0           # a snap's time no longer than this came off a stopped clock
MAX_SECONDS = 60.0
MAX_OT = 3
TIMEOUTS, OT_TIMEOUTS = 3, 2
MAX_STEPS = 900

KICK, SNAP, DONE = 0, 1, 2
GAIN, TURNOVER, DEF_TD = s14.GAIN, s14.TURNOVER, s14.DEF_TD

# the features, each a small integer
NOTHING, SHORT, FIRST, LOSS, FRESH, TIMEOUT = range(6)
PREV_NAMES = ("nothing gained", "short gain", "first down", "loss", "fresh possession", "timeout")
PHASES = ("Q1", "Q2", "Q2 2:00", "Q3", "Q4", "Q4 2:00", "OT")
MARGIN_EDGES = np.array([-8.5, -3.5, -0.5, 0.5, 3.5, 8.5])
DIST_EDGES = np.array([1.5, 3.5, 6.5, 10.5])
ZONE_EDGES = np.array([10.5, 30.5, 50.5, 70.5, 80.5, 89.5, 95.5])   # the other side's 29-20 and 19-11 apart
                                                                     # (field-goal range); the 10 to the 5, inside the 5
ELAPSED_EDGES = np.array([30.5, 60.5, 90.5])
REMAIN_EDGES = np.array([1.5, 2.5, 4.5, 7.5, 12.5, 17.5])   # snaps left in the half: 0-1, 2, 3-4, 5-7, 8-12, 13-17, 18+
REMAIN_NAMES = ("0-1", "2", "3-4", "5-7", "8-12", "13-17", "18+")
TIME_EDGES = np.array([6.5, 9.5, 14.5, 24.5, 34.5])
TIME_BUCKETS = ("<=6s", "7-9s", "10-14s", "15-24s", "25-34s", "35s+")

SIZES = dict(prev=6, clock2=2, qtr=5, rem4=4, phase=7, part=5, half=3, margin=7, down=4, dist=5, zone=8, elapsed=4, remain=7, tb=6,
             lead3=3, tb2=2, dist3=3, zone3=3, endgame=50, endgame3=22)  # part and the last six: derive's
# The draws' part of the game (SPLIT): "snaps" -- the half and the two-minute warning (part: the first
# half before its 2:00, its last 2:00, the same of the second, overtime), the snaps left in the half
# telling the rest (Q1 from Q2, the start of a half from its middle); "time" -- the seven parts (phase).
# The half stays: Q2's end is not the game's (a level side attacks before the half, plays for the kick
# at the end of the game), and so does the warning: the clock's rules change there.
SPLIT = "snaps"
SPLITS = ("snaps", "time")
# The draws' key (KEY): "full" -- TIME_BASE / DIST_BASE (the play before in its six classes, the part of
# the game, the snaps played so far); "lean" -- fewer cells, more real snaps in each: the clock into the
# snap running or stopped (clock2) or the time the play took (tb), the period with the two-minute
# warning apart (phase: a level side with three snaps left at 2:30 attacks, at 1:30 sets up its kick), the lead, down,
# distance, field and snaps left in the half, and nothing else. Its fallbacks coarsen the field and
# the snaps left before the lead (all seven of margin_b's) gives way to behind, level or ahead: a side
# 9+ ahead in Q4 runs it out where one 1-3 ahead still plays for a first down.
# The distance draw lets the field go before the distance: the field windows find the snaps near the
# state's own field in a group of every field. It keeps the distance (all five of dist: 3rd and 1 is not
# played from 3rd and 3's snaps), the time the play took (all six of TIME_BUCKETS) and the down to the end: a 40-yard gain takes 10-14
# seconds and an incomplete pass 6 or less, and a draw that pools the two (stopped or running alone)
# gives a 6-second snap the long gains and takes them from the 12-second ones. The period holds to the
# last two: blended, a thin late cell passes its draws down the chain, and a level side in Q4's last
# 2:00 drew 27% of them from every snap of the game at its down and distance, a touchdown near the
# field 16-17% of the time against its own 4-9%. The distance holds too: levels without it played 2nd
# and 1 off every 2nd down's own results, 44% moving the chains against 51% real. The time draw keeps
# the down and all five distances as far: the distance draw reads the time it was given at the state's
# own distance, so a time pooled over 1 to 3 to go (or every distance) gives 2nd and 1 the quick
# incompletions of 2nd and 3, played out as 2nd and 1's own quick snaps. Before the lead goes, the time
# the play took coarsens to stopped or running (tb2), the period and lead held: a level side in Q4's
# last 2:00 drew 25% of its snaps from levels of every lead or the whole game, which scored 13-18% near
# its field against its own 4-8%, and its drives with 1:00-0:31 left scored a touchdown 19.6% of the
# time against 10.3% real (a field goal 58% against 78%).
KEY = "full"
KEYS = ("full", "lean")
# Blended levels (build's blend, 0 off): the league's draws store every cell of BLEND_MIN real snaps or
# more, and a draw takes its finest cell n / (n + blend) of the time, else the next level's (Bins.
# bin_blend), in place of the first level with MIN_RECORDS.
BLEND_MIN = 5
# Field windows (build's field, 0 off): the distance draw takes a real snap of its group from near the
# state's own field -- within `field` x the yards to go of it (FIELD_YARDS at least), or the FIELD_MIN
# nearest where fewer are there (Bins.near) -- and keeps its own result: its touchdown is one, a gain
# short of its own goal stays short of this one (draw_play), so a gain made at midfield is not played
# out at the 12, nor a touchdown from the 12 run on from midfield. A tilted draw (adjusted) resamples
# FIELD_CANDIDATES near ones on their place in the whole group.
FIELD_YARDS = 1
FIELD_MIN = 20
FIELD_CANDIDATES = 6
# Blended with field windows (NEAR_BLEND, a build's near_blend): a cell's window is never widened -- only
# the last level's, to FIELD_MIN -- and a cell with none of its snaps near the state's own field passes
# all its weight on. A thin cell widened its window to 20 of its snaps: a late cell of 2nd and 1 at
# midfield reached its snaps at the goal line, a touchdown kept there and a stuffed run played as no
# gain. From the real states at the other side's 49-30 the sim's draws scored 6.0% against 4.3% real
# (from the whole group, 5.1%). A cell still weighs by all its snaps, n / (n + blend): weighed by those
# near the field alone (t8), the thin late cells passed their draws to levels without the lead or the
# period -- a level side in Q4's last 2:00 drew 36% of them from levels with both, against 61% -- and
# its snaps scored 9.4% against 6.1% real, 1 to go moved the chains 63.2% against 67.2%.
NEAR_BLEND = True
# Clock windows (build's clock, 0 off): in a half's last CLOCK_LATE seconds the time draw takes a real snap
# of its group from those with as long left in the half -- within `clock` x the seconds left (CLOCK_SECONDS
# at least), or the CLOCK_MIN nearest -- so a side at 0:20 on a running clock draws what real sides did at
# 0:20, not a 30-second snap made at 0:45 that runs the half out before it can kick.
WINDOW_SPAN = 1024
WINDOW_ENDS = {"field": (1, 99), "left": (0, WINDOW_SPAN - 1)}     # the scale's ends a window keeps level with
CLOCK_LATE = 120.0
CLOCK_SECONDS = 2
CLOCK_MIN = 5                 # a thin cell's window widens to its 5 nearest only: its 20 nearest at 0:12
                              # could reach back to 0:40 and its 30-second snaps
QTR_OF = np.array([0, 1, 1, 2, 3, 3, 4])                   # phase -> period (Q1-Q4, overtime)
TIME_LEAN = ("clock2", "phase", "margin", "down", "dist", "zone", "remain")
DIST_LEAN = ("tb", "phase", "margin", "down", "dist", "zone", "remain")
TIME_LEAN_FEATURES = TIME_LEAN + ("lead3", "zone3", "rem4")
DIST_LEAN_FEATURES = DIST_LEAN + ("lead3", "tb2", "dist3", "zone3", "rem4")
TIME_LEAN_LEVELS = (TIME_LEAN,
                    ("clock2", "phase", "margin", "down", "dist", "zone3", "remain"),
                    ("clock2", "phase", "margin", "down", "dist", "remain"),
                    ("clock2", "phase", "margin", "down", "dist", "rem4"),
                    ("clock2", "phase", "lead3", "down", "dist", "rem4"),
                    ("clock2", "phase", "lead3", "down", "dist"),
                    ("clock2", "phase", "down", "dist"),
                    ("clock2", "down", "dist"),
                    ("clock2", "down"),
                    ("clock2",),
                    ())
DIST_LEAN_LEVELS = (DIST_LEAN,
                    ("tb", "phase", "margin", "down", "dist", "zone3", "remain"),
                    ("tb", "phase", "margin", "down", "dist", "remain"),
                    ("tb", "phase", "lead3", "down", "dist", "remain"),
                    ("tb", "phase", "lead3", "down", "dist", "rem4"),
                    ("tb", "phase", "lead3", "down", "dist"),
                    ("tb2", "phase", "lead3", "down", "dist"),
                    ("tb", "phase", "down", "dist"),
                    ("tb", "down", "dist"),
                    ("tb", "down"),
                    ())
PART_OF = np.array([0, 0, 1, 2, 2, 3, 4])                  # phase -> part
TIME_BASE = ("prev", "phase", "margin", "down", "dist", "zone", "elapsed", "remain")
DIST_BASE = ("tb", "down", "dist", "zone", "margin", "phase", "elapsed", "remain")
TIME_FEATURES = TIME_BASE + ("lead3", "endgame3")       # the fine features, then the coarse ones the fallbacks read
DIST_FEATURES = DIST_BASE + ("lead3", "zone3", "endgame3", "dist3", "tb2", "half")


def _drop(features, *names):
    return tuple(f for f in features if f not in names)


# The fallbacks keep the situation -- the down, the distance, the field, the snaps left in the half (the
# sim's own expected count for the clock, whether it's running, both sides' timeouts and the lead) and
# coarsely the part of the game and the lead -- to the end: a 3rd down is never drawn from 1st downs,
# a side with time for one snap never from one with time for ten, while there are real snaps alike.
# At the end of a half (endgame3: the snaps left and the lead) the distance and time coarsen before
# the lead goes, and the field (its 7 zones) last: a level side in field-goal range late plays for the
# kick, a side ahead runs it out, and neither is drawn from the snaps of every lead at that spot; nor
# from the whole of the other half, where a real touchdown from the 12 drawn at the 45 runs on.
TIME_LEVELS = (TIME_BASE,
               _drop(TIME_BASE, "elapsed"),
               _drop(TIME_BASE, "elapsed", "zone"),
               ("prev", "phase", "margin", "down", "remain"),
               ("prev", "phase", "lead3", "down", "remain"),
               ("prev", "phase", "lead3", "remain"),
               ("prev", "phase", "endgame3"),
               ("prev", "phase", "remain"),
               ("prev", "remain"),
               ("prev",),
               ())
DIST_LEVELS = (DIST_BASE,
               _drop(DIST_BASE, "elapsed"),
               ("tb", "down", "dist", "zone", "lead3", "phase", "remain"),
               ("tb", "down", "dist", "zone", "lead3", "half", "remain"),
               ("tb", "down", "dist", "zone", "endgame3"),
               ("tb", "down", "dist3", "zone", "endgame3"),
               ("tb", "down", "dist3", "zone3", "endgame3"),
               ("tb", "down", "zone3", "endgame3"),
               ("tb", "down", "endgame3"),
               ("tb", "down", "dist", "zone"),
               ("tb", "down", "dist", "zone3"),
               ("tb", "down", "dist"),
               ("tb", "down"),
               ())
# a gamer's or team's own snaps: alike in the situation, or the league's draw -- never their snaps of
# every down and distance, nor of every part of the game: the end of a half (endgame) apart, by the
# snaps left and the lead, so a side's own snaps there come from its own ends of halves or the league's
GAMER_TIME_FEATURES = ("who", "prev", "phase", "margin", "down", "endgame", "lead3")
GAMER_DIST_FEATURES = ("who", "tb", "down", "dist", "zone", "endgame", "tb2", "dist3", "zone3")
GAMER_TIME_LEVELS = (("who", "prev", "phase", "margin", "down", "endgame"), ("who", "prev", "endgame", "lead3", "down"))
GAMER_DIST_LEVELS = (("who", "tb", "down", "dist", "zone", "endgame"), ("who", "tb2", "down", "dist3", "zone3", "endgame"))
MIN_RECORDS = 40
MIN_GAMER_RECORDS = 30
DEF_SHARE = 0.5                                           # full: the share of draws from what the defense allowed
MODES = ("full", "adjusted", "tilt")
# tilt: the league's draws for every side, each reweighted by exp(theta . phi) of the real snap drawn --
# phi the distance's (yards / 10 cut to -10..50, a touchdown, a turnover: PHI_DIST) and the time's
# (seconds / 10: PHI_TIME) -- theta the summed tilts of the offense's gamer and team and the defense's,
# fitted by maximum likelihood on their real snaps against what the league's draws (blended, near the
# field) gave the same states, each shrunk toward none as if it had TILT_PRIOR league-average snaps.
# A draw resamples TILT_CANDIDATES real snaps of its group (or window) on their weights.
PHI_DIST = ("yards", "touchdown", "turnover", "first down", "loss")
PHI_TIME = ("seconds",)
TILT_PRIOR = 200.0
TILT_ROUNDS = 4
TILT_CANDIDATES = 16
TILT_HALF_LIFE = 30.0         # days: a real snap's weight in the fit halves every so many days before the last
# The points nudge (step 3, a switch for pricing: nudge_solve): each side's offense tilt moved along the
# strength direction (nudge_direction: the gamers' offense tilts regressed on the points their side really
# scores a match, scaled so a unit is a point a side in the sim) until the sim's kick-off mean meets the
# side's expected points,
# by secant steps from the build's slope (calibrate_nudge), the same random numbers every step.
NUDGE_ITERS = 3
NUDGE_MAX = 10.0              # points a side
NUDGE_PATHS = 3000
YARD_BIN_EDGES = np.array([-5, 0, 1, 3, 5, 8, 12, 16, 20, 30, 40, 50])     # the fit's histograms
SECOND_BIN_EDGES = np.arange(2, 61, 2)
ADJUST_GRID = np.round(np.linspace(-1.5, 1.5, 31), 3)     # log tilts tried
ADJUST_PRIOR = 300.0
ADJUST_SWEEPS = 3
TO_VALUE, DEF_TD_VALUE = -20.0, -40.0                     # a turnover's place among the gains, for the order
SMOOTH_YARDS = 0                                          # +-yards a drawn gain is spread over (0: as drawn)
EFFECTS = ("gamer_off", "gamer_def", "team_off", "team_def")

GO, FG, PUNT = 0, 1, 2
FG_MAX = 65                   # longest kick tried (yards)
CONV_CELLS, CONV_MARGIN, CONV_CLOCK = s14.CONV_CELLS, s14.CONV_MARGIN, s14.CONV_CLOCK
RETURN, TOUCHBACK, NO_LANDING, KEPT = range(4)
KICK_TOUCHBACK, KICK_NO_LANDING = s14.KICK_TOUCHBACK, s14.KICK_NO_LANDING
LEFT_EDGES = np.array([10.5, 30.5, 60.5, 120.5, 240.5])     # seconds left in the half
EARLY_LEFT = 60.0                                            # early kicks: downs 1-3 in the half's last minute
EARLY_EDGES = np.array([5.5, 10.5, 20.5, 30.5, 45.5])
RANGE_EDGES = np.array([35.5, 45.5, 55.5])                   # the kick's length
FG_NEAR = 40.0                # a kick with this little left in the half takes its clock as the real ones did
KICK_POOL_MIN = 20

# The decisions. Each is drawn from the real ones in states alike (Bins over its features, falling back
# level by level below DECISION_MIN), and leaned by the deciding gamer's own log-odds shift for each
# option against the first (fitted on their real decisions, shrunk by DECISION_PRIOR).
DECISION_MIN = 25
DECISION_PRIOR = 3.0
# A conversion's cells blend lighter (a spec's own "blend", with the build's decision blend on): the
# lead after the six all but makes the choice, and a thin late cell passing 25 tries' worth down the
# chain reached the whole game's at that lead, where sides kick -- 2 behind after a late six went for
# two 77% of the time in the sim's cells against 87% real, 5 behind 79% against 86%, 10 behind 72%
# against all 16 real ones.
CONV_BLEND = 5.0
# A two-point try's chance by the part of the game (conv_late), each shrunk toward the whole game's by
# TWO_GOOD_PRIOR tries: September's real ones were made 64-71% of the time before Q4's last 3:00 (239
# of them), 52.9% in it (493) and 36% in overtime (11), where the sim made 56.6% of all of them.
TWO_GOOD_PRIOR = 20.0
# The half's end (LATE_TIMEOUT): a side that a kick puts level or ahead in Q4 or overtime, wherever it
# is (at Q2's end, any lead, in field-goal range), with a timeout left, calls it where a running clock
# would run the half out before or during its snap. Real sides level with the ball with 1:00-0:31 left
# ran out of clock on 0.3% of their drives (2:00-1:01, 0.4%); the sim's 9.8% (4.2%), and of those 62%
# (77%) were in range, their last snap with more than 0:05 to go, and 88% (95%) had a timeout left. In
# range alone (t9b, t10), 5.3%: 41% of them out of range at their last snap, 74% with a timeout left.
LATE_TIMEOUT = True
DECISIONS = {
    "fourth": dict(label="4th down", options=("go for it", "field goal", "punt"), who="offense",
                   features=("togo10", "dist", "margin", "phase", "left", "fg4", "dist3", "togo4"),
                   sizes=(10, 5, 7, 7, 6, 4, 3, 4),
                   levels=(("togo10", "dist", "margin", "fg4", "phase", "left"), ("togo10", "dist", "fg4", "phase", "left"),
                           ("togo10", "dist3", "fg4", "phase", "left"), ("togo4", "dist3", "fg4", "phase", "left"),
                           ("togo4", "fg4", "phase", "left"), ("togo4", "dist3", "fg4", "phase"),
                           ("togo4", "fg4", "phase"), ("fg4", "phase"), ("togo10", "dist"), ("togo10",), ())),
    "early_fg": dict(label="field goal on downs 1-3 in the half's last minute", options=("play on", "kick"),
                     who="offense", features=("left", "range", "down", "margin", "half", "fg4", "clock2", "to3"),
                     sizes=(6, 4, 3, 7, 3, 4, 2, 3),
                     levels=(("left", "clock2", "to3", "range", "down", "fg4", "half"),
                             ("left", "clock2", "to3", "range", "fg4", "half"), ("left", "clock2", "to3", "fg4", "half"),
                             ("left", "clock2", "fg4", "half"), ("left", "range", "fg4", "half"), ("left", "fg4", "half"),
                             ("fg4", "half"), ("left", "half"), ("left",), ())),
    "kickoff": dict(label="kick-off", options=("return", "touchback", "no landing zone", "kept"), who="kicker",
                    features=("desperate", "late", "lead"), sizes=(2, 2, 3),
                    levels=(("desperate", "late", "lead"), ("desperate", "late"), ("desperate",), ())),
    "two": dict(label="conversion", options=("kick", "go for two"), who="scorer", features=("cell", "lead", "late"),
                sizes=(CONV_CELLS, 2 * CONV_MARGIN + 1, 3),
                levels=(("cell", "lead"), ("late", "lead"), ("lead",), ()), blend=CONV_BLEND),
    "to_def": dict(label="defense's timeout before a running-clock snap", options=("none", "timeout"),
                   who="defense", features=("phase", "margin", "left"), sizes=(7, 7, 6),
                   levels=(("phase", "margin", "left"), ("phase", "margin"), ("phase",), ())),
    "to_off": dict(label="offense's timeout before a running-clock snap", options=("none", "timeout"),
                   who="offense", features=("phase", "margin", "left", "range2"), sizes=(7, 7, 6, 2),
                   levels=(("phase", "margin", "left", "range2"), ("phase", "margin", "left"), ("phase", "margin"),
                           ("phase",), ())),
}
MARGIN_NAMES = ("9+ behind", "4-8 behind", "1-3 behind", "level", "1-3 ahead", "4-8 ahead", "9+ ahead")
SLICE_NAMES = {
    "fourth": ("inside the 20, 1-3 to go", "inside the 20, 4+ to go", "20-39 out, 1-3 to go", "20-39 out, 4+ to go",
               "40-59 out, 1-3 to go", "40-59 out, 4+ to go", "60+ out, 1-3 to go", "60+ out, 4+ to go"),
    "early_fg": ("0:05 or less left", "0:06-0:10", "0:11-0:20", "0:21-0:30", "0:31-0:45", "0:46-1:00"),
    "kickoff": ("ordinary", "the half's last 2:00", "behind late in Q4"),
    "two": tuple(f"{m} after the six" for m in MARGIN_NAMES),
    "to_def": PHASES, "to_off": PHASES,
}
# the option each gamer's lean is shown for, and the real decisions a gamer needs to be shown
LEAN_SHOWN = {"fourth": (GO, 20), "early_fg": (1, 10), "kickoff": (TOUCHBACK, 20), "two": (1, 10),
              "to_def": (1, 40), "to_off": (1, 40)}

# The rules: options never or all but never taken in a situation, learned off the real decisions
# (v21_rules: the sweep), struck out for every gamer before the league's shares and any gamer's lean;
# and overtime's own (ot_mask). RULES off: the real shares alone.
RULES = True
# The game's own end: in Q4, the side ahead with the ball whose downs left, each taking the play clock,
# outlast the clock while the defense has no timeouts -- the game is over (auto_end).
AUTO_END = True
REMAIN_SLICE = 10.0           # snaps left in the half, by 10-second slice of the time left in it,
REMAIN_SLICES = int(2 * QUARTER // REMAIN_SLICE) + 1      # the clock running or not, both sides' timeouts, the lead
REMAIN_SHAPE = (3, REMAIN_SLICES, 2, 4, 4, 3)
REMAIN_PRIOR = 20.0           # a cell's own count shrunk toward its slice's (and clock's) by this many snaps
SURV_PRIOR = 200.0            # gains' survival at the end of a half: shrunk toward the league's by this many snaps
REFINE_REMAINING = True
REFINE_ROUNDS = 3             # the sim's own counts, the draws rebuilt on them, so many times over
REFINE_PATHS = 6000
SHARE_SLICE = 20.0            # points still to come, by 20-second slice of regulation (v21_report)
SHARE_SLICES = int(4 * QUARTER // SHARE_SLICE)


# ---------------------------------------------------------------- features

def phase_of(period, clock):
    period, clock = np.asarray(period), np.asarray(clock, float)
    late = clock <= LATE
    return np.select([period == 1, (period == 2) & ~late, period == 2, period == 3, (period == 4) & ~late,
                      period == 4], [0, 1, 2, 3, 4, 5], 6).astype(np.int64)


def _cut(v, edges):
    return np.searchsorted(edges, np.asarray(v, float), side="right").astype(np.int64)


def margin_b(m):
    return _cut(m, MARGIN_EDGES)


def dist_b(t):
    return _cut(t, DIST_EDGES)


def zone_b(y):
    return _cut(y, ZONE_EDGES)


def elapsed_b(n):
    return _cut(n, ELAPSED_EDGES)


def remain_b(r):
    return _cut(r, REMAIN_EDGES)


def time_b(s):
    return _cut(s, TIME_EDGES)


def field_radius(share, y, ymin=FIELD_YARDS):
    """The field window's reach at field y: share x the yards to go, ymin at least."""
    return np.maximum(ymin, np.round(share * (100 - np.asarray(y)))).astype(np.int64)


def near_blended(m):
    """Whether the model's blended distance draws weigh their cells by the snaps near the field."""
    return bool(getattr(m, "near_blend", False) and getattr(m, "field", 0.0) and getattr(m, "blend", 0.0))


def near_weight_counts(b, C, field, share):
    """(states, levels) the records a blended distance draw counts each level's group for (NEAR_BLEND):
    all of them where any is near the state's field, none where none is (the group passes all on)."""
    full = np.where(C >= 0, b.count[np.maximum(C, 0)], 0)
    return np.where(b.near_counts(C, field, field_radius(share, field), "field") > 0, full, 0)


def dist_weights(m, C, field):
    """(states, levels) the distance draw's blended weights for states at `field` (chain C): where the
    model blends near (near_blended), a group with none of its snaps near the field passes all on;
    else by all of them."""
    b = m.dist
    if near_blended(m):
        return b.blend_weights(C, m.blend, near_weight_counts(b, C, field, m.field))
    return b.blend_weights(C, m.blend)


def field_nmin(m, last):
    """The fewest snaps a level's field window takes before it widens: FIELD_MIN at the last level or
    where the model does not blend near, else 1 (its cell's weight counted only those near)."""
    return np.where(np.asarray(last), FIELD_MIN, 1) if near_blended(m) else np.full(np.shape(last), FIELD_MIN)


def left_b(s):
    return _cut(s, LEFT_EDGES)


def early_b(s):
    return _cut(s, EARLY_EDGES)


def range_b(k):
    return _cut(k, RANGE_EDGES)


def in_range(togo):
    """1 where a field goal is in reach (the kick FG_MAX yards or less), off the yards to the goal."""
    return (np.asarray(togo) + 17 <= FG_MAX).astype(np.int64)


def lead3(m):
    m = np.asarray(m)
    return np.where(m < 0, 0, np.where(m == 0, 1, 2)).astype(np.int64)


FG4_NAMES = ("4+ behind", "3 behind", "1-2 behind or level", "ahead")
DRIVE_ENDS = ("touchdown", "field goal", "missed kick", "punt", "turnover", "downs", "clock out")


def fg4_of(lead):
    """The lead as a field goal reads it: not enough (4+ behind), a tie (3 behind: overtime, which real
    sides go for a touchdown rather than settle for), the lead (1-2 behind or level), ahead."""
    lead = np.asarray(lead)
    return np.select([lead <= -4, lead == -3, lead <= 0], [0, 1, 2], 3).astype(np.int64)


def conv_late(cell):
    """A conversion's part of the game as its fallback reads it: 0 before Q4's last 3:00, 1 in it, 2
    overtime. Where a clock cell has too few real tries for its lead, the try reads Q4's last 3:00 as one
    before it reads the whole game: 1 behind after a late six goes for two; earlier it kicks."""
    cell = np.asarray(cell)
    return np.where(cell == CONV_CELLS - 1, 2, np.where(cell >= 3, 1, 0)).astype(np.int64)


def togo4(togo):
    """The yards to the goal as a kicker reads them: inside the 20, 20-34 (an easy kick), 35-51 (a long
    one) and further (no real side kicks from 52 or more, v21_rules)."""
    return np.select([np.asarray(togo) <= 19, np.asarray(togo) <= 34, np.asarray(togo) <= 51], [0, 1, 2], 3).astype(np.int64)


def togo10(y):
    return np.clip((100 - np.asarray(y)) // 10, 0, 9).astype(np.int64)


def decision_slice(name, c):
    """The report's slice (SLICE_NAMES) of each decision, off its features."""
    if name == "fourth":
        band = np.select([c["togo10"] <= 1, c["togo10"] <= 3, c["togo10"] <= 5], [0, 1, 2], 3)
        return (band * 2 + (np.asarray(c["dist"]) >= 2)).astype(np.int64)
    if name == "early_fg":
        return np.asarray(c["left"], np.int64)
    if name == "kickoff":
        return np.where(np.asarray(c["desperate"]) == 1, 2, np.asarray(c["late"])).astype(np.int64)
    if name == "two":
        return margin_b(np.asarray(c["lead"]) - CONV_MARGIN)
    return np.asarray(c["phase"], np.int64)


def down_b(d):
    return np.clip(np.asarray(d, np.int64) - 1, 0, 3)


def play_class(gain, dist):
    gain = np.asarray(gain)
    return np.where(gain == 0, NOTHING, np.where(gain < 0, LOSS, np.where(gain >= np.asarray(dist), FIRST, SHORT)))


def half_left(period, clock):
    """Seconds left in the half (regulation), in the overtime period for overtime."""
    period, clock = np.asarray(period), np.asarray(clock, float)
    return np.where((period == 1) | (period == 3), clock + QUARTER, clock)


def remain_slice(period, clock):
    """(half: 0 first, 1 second, 2 overtime, 10-second slice of the time left in it)."""
    period = np.asarray(period)
    half = np.where(period <= 2, 0, np.where(period <= 4, 1, 2))
    sl = np.clip((half_left(period, clock) // REMAIN_SLICE).astype(np.int64), 0, REMAIN_SLICES - 1)
    return half, sl


def running_of(prev):
    """Whether the clock runs into the snap off the play before (a gain short of the line, a first
    down, a loss); nothing gained (mostly an incomplete pass), a fresh possession and a timeout stop it."""
    return np.isin(np.asarray(prev), (SHORT, FIRST, LOSS)).astype(np.int64)


def remain_key(period, clock, running, own_to, opp_to, lead3):
    """The snaps-left table's index of a state: the half, its 10-second slice, the clock running or
    not, the side with the ball's timeouts and the other side's (0-3) and the lead (behind, level,
    ahead)."""
    half, sl = remain_slice(period, clock)
    to = lambda v: np.clip(np.asarray(v, np.int64), 0, 3)
    return half, sl, np.asarray(running, np.int64), to(own_to), to(opp_to), np.asarray(lead3, np.int64)


def remain_table(key, counts, prior=REMAIN_PRIOR, base=None):
    """The expected snaps left in the half for every cell of REMAIN_SHAPE off states (key) and the
    snaps that followed each (counts): each cell's own average shrunk toward its slice and clock's,
    that toward its slice's; slices with none read off their neighbours (or `base`'s)."""
    num, den = np.zeros(REMAIN_SHAPE), np.zeros(REMAIN_SHAPE)
    np.add.at(num, key, counts)
    np.add.at(den, key, 1)
    n2, s2 = den.sum((2, 3, 4, 5)), num.sum((2, 3, 4, 5))
    m2 = np.zeros(n2.shape)
    for h in range(n2.shape[0]):
        have = np.flatnonzero(n2[h] > 0)
        if len(have):
            m2[h] = np.interp(np.arange(n2.shape[1]), have, s2[h, have] / n2[h, have])
        elif base is not None:
            m2[h] = base[h].mean((1, 2, 3, 4))
        else:
            m2[h] = np.arange(n2.shape[1]) * REMAIN_SLICE / 20.0
    n3, s3 = den.sum((3, 4, 5)), num.sum((3, 4, 5))
    m3 = (s3 + prior * m2[..., None]) / (n3 + prior)
    return (num + prior * m3[..., None, None, None]) / (den + prior)


def share_slice(period, clock):
    """The 20-second slice of regulation a clock is in (0 the first; overtime the last)."""
    period, clock = np.asarray(period), np.asarray(clock, float)
    played = (np.minimum(period, 4) - 1) * QUARTER + (QUARTER - clock)
    return np.where(period >= 5, SHARE_SLICES - 1,
                    np.clip((played // SHARE_SLICE).astype(np.int64), 0, SHARE_SLICES - 1))


def ot_mask(name, n, must_score=None, other_done_close=None):
    """(decisions, options) what overtime's own rules leave, or None: a side behind once the other side
    has had the ball (must_score) can't punt, nor kick a field goal 4+ behind (lead4: the lead in
    must_score's place); after a touchdown 1 or 2 behind once the other side has had it
    (other_done_close) it goes for two."""
    if name in ("fourth", "early_fg") and must_score is not None:
        must, behind4 = must_score
        if not must.any():
            return None
        ok = np.ones((n, len(DECISIONS[name]["options"])), bool)
        if name == "fourth":
            ok[must, PUNT] = False
            ok[must & behind4, FG] = False
        else:
            ok[must & behind4, 1] = False
        return ok
    if name == "two" and other_done_close is not None and np.any(other_done_close):
        ok = np.ones((n, 2), bool)
        ok[np.asarray(other_done_close, bool), 0] = False
        return ok
    return None


def auto_end(period, clock, lead, down, def_timeouts):
    """Whether the game is over by its own end: Q4, the side with the ball ahead, its downs left at the
    play clock each outlasting the clock, the defense out of timeouts."""
    period, clock, lead = np.asarray(period), np.asarray(clock, float), np.asarray(lead)
    return (period == 4) & (lead > 0) & (np.asarray(def_timeouts) == 0) \
        & ((5 - np.asarray(down)) * s14.PLAY_CLOCK > clock)


def combine(*masks):
    """The options every mask leaves (None: no mask)."""
    got = [m for m in masks if m is not None]
    if not got:
        return None
    ok = np.logical_and.reduce(got)
    ok[~ok.any(1)] = True
    return ok


# ---------------------------------------------------------------- the draws

def phi_dist_of(kind, gain, td, first):
    """(records, 5) a real snap's distance features the tilts weigh (PHI_DIST): its yards / 10 (cut to
    -10..50; none for a turnover), a touchdown, a turnover (or a defensive touchdown), a new set of
    downs short of a touchdown (first), a loss."""
    kind, gain = np.asarray(kind), np.asarray(gain, float)
    td, first = np.asarray(td, bool), np.asarray(first, bool)
    is_gain = kind == GAIN
    g = np.where(is_gain, np.clip(gain, -10, 50), 0.0)
    return np.stack([g / 10.0, td.astype(float), (~is_gain).astype(float),
                     (is_gain & first & ~td).astype(float), (is_gain & (gain < 0)).astype(float)], axis=1)


def dist_bin(kind, gain, td, first):
    """Each record's histogram bin for the fit: by yards (YARD_BIN_EDGES) and whether it moved the
    chains for a gain, by yards for a touchdown, one bin for a turnover or a defensive touchdown."""
    kind, gain = np.asarray(kind), np.asarray(gain, float)
    td, first = np.asarray(td, bool), np.asarray(first, bool)
    yb = np.digitize(np.clip(gain, -10, 50), YARD_BIN_EDGES)
    nb = len(YARD_BIN_EDGES) + 1
    return np.where(kind != GAIN, 3 * nb, np.where(td, 2 * nb + yb, yb + nb * first)).astype(np.int64)


DIST_BINS = 3 * (len(YARD_BIN_EDGES) + 1) + 1
TIME_BINS = len(SECOND_BIN_EDGES) + 1


def time_bin(seconds):
    return np.digitize(np.asarray(seconds, float), SECOND_BIN_EDGES).astype(np.int64)


class Bins:
    """Real records grouped for drawing: a group for every combination of each level's features that
    holds `min_records` (every level but the last, (); with `strict` that one too), each group's records
    sorted by their value, so a draw (and a tilt) runs from the bottom of the group to the top. A draw's
    group is the first level, finest first, whose group for its features is there (strict: none if no
    level's is)."""

    def __init__(self, features, sizes, levels, codes, start, count, level_of, payload):
        self.features, self.sizes, self.levels = tuple(features), tuple(sizes), tuple(levels)
        self.codes = codes                 # per level: (its groups' codes, sorted; the first group's index)
        self.start, self.count, self.level_of, self.payload = start, count, level_of, payload

    def _dims(self, lv):
        pos = {f: i for i, f in enumerate(self.features)}
        return tuple(self.sizes[pos[f]] for f in lv)

    @classmethod
    def build(cls, features, sizes, levels, X, value, payload, min_records, strict=False):
        """X (records, features) ints; value the sort key; payload {name: array} kept with them."""
        X = np.asarray(X, np.int64).reshape(-1, len(features))
        value = np.asarray(value, float)
        n = len(X)
        sizes = tuple(int(x) for x in sizes)
        pos = {f: i for i, f in enumerate(features)}
        codes, start, count, level_of, take = [], [], [], [], []
        total = 0
        for li, lv in enumerate(levels):
            cols = [pos[f] for f in lv]
            dims = tuple(sizes[c] for c in cols)
            rc = np.ravel_multi_index(tuple(np.clip(X[:, c], 0, sizes[c] - 1) for c in cols), dims) if cols \
                else np.zeros(n, np.int64)
            need = min_records if (strict or lv) else 1
            o = np.lexsort((value, rc))
            sc = rc[o]
            uniq, first, cnt = np.unique(sc, return_index=True, return_counts=True)
            ok = cnt >= need
            codes.append((uniq[ok], len(start)))
            for u0, f0, c0 in zip(uniq[ok], first[ok], cnt[ok]):
                start.append(total)
                count.append(int(c0))
                level_of.append(li)
                take.append(o[f0:f0 + c0])                # sorted by value within the group
                total += int(c0)
        idx = np.concatenate(take) if take else np.zeros(0, np.int64)
        return cls(features, sizes, levels, codes, np.array(start, np.int64), np.array(count, np.int64),
                   np.array(level_of, np.int64), {k: np.asarray(v)[idx] for k, v in payload.items()})

    def bin_of(self, cols):
        """The group each row of features (a dict of arrays) draws from (-1: none)."""
        n = len(np.asarray(cols[self.features[0]]))
        out = np.full(n, -1, np.int64)
        for li, lv in enumerate(self.levels):
            elig, first = self.codes[li]
            todo = np.flatnonzero(out < 0)
            if not len(todo):
                break
            if not len(elig):
                continue
            if lv:
                dims = self._dims(lv)
                rc = np.ravel_multi_index(tuple(np.asarray(cols[f], np.int64)[todo] for f in lv), dims, mode="clip")
            else:
                rc = np.zeros(len(todo), np.int64)
            k = np.minimum(np.searchsorted(elig, rc), len(elig) - 1)
            hit = elig[k] == rc
            out[todo[hit]] = first + k[hit]
        return out

    def chain(self, cols):
        """(states, levels): each state's group at every level (-1: none there)."""
        n = len(np.asarray(cols[self.features[0]]))
        out = np.full((n, len(self.levels)), -1, np.int64)
        for li, lv in enumerate(self.levels):
            elig, first = self.codes[li]
            if not len(elig) or not n:
                continue
            if lv:
                rc = np.ravel_multi_index(tuple(np.asarray(cols[f], np.int64) for f in lv), self._dims(lv), mode="clip")
            else:
                rc = np.zeros(n, np.int64)
            k = np.minimum(np.searchsorted(elig, rc), len(elig) - 1)
            hit = elig[k] == rc
            out[hit, li] = first + k[hit]
        return out

    def blend_weights(self, chain, prior, n=None):
        """(states, levels) the chance each level's group is drawn from when the levels are blended: the
        finest first, each taken with its real records' share n / (n + prior), the rest passed on to the
        next level there; the last level takes what is left. n: (states, levels) the records each group
        counts for the state, else all of them (0: the group passes all on)."""
        n = (np.where(chain >= 0, self.count[np.maximum(chain, 0)], 0) if n is None else n).astype(float)
        a = np.where((chain >= 0) & (n > 0), n / (n + prior), 0.0)
        a[:, -1] = np.where(chain[:, -1] >= 0, 1.0, 0.0)
        left = np.cumprod(np.c_[np.ones(len(a)), 1.0 - a[:, :-1]], axis=1)
        return left * a

    @staticmethod
    def pick(chain, W, rng):
        """(groups, levels) a level drawn for each state on its weights W (blend_weights), and its group."""
        cw = np.cumsum(W, 1)
        u = rng.random(len(W)) * cw[:, -1]
        li = np.minimum((cw <= u[:, None]).sum(1), W.shape[1] - 1)
        g = chain[np.arange(len(W)), li]
        bad = g < 0                                 # a state with no weight anywhere: its last level
        li[bad] = W.shape[1] - 1
        g[bad] = chain[bad, -1]
        return g, li

    def span(self, b, v, r, name):
        """(lo, hi) each draw's records of its group (b) within r of the state's own value v, in
        by_value(name) order: the same either side, stopping as far short of a nearer end of the scale on
        the other side (WINDOW_ENDS), never widened."""
        _, key = self.by_value(name)
        b, v = np.asarray(b, np.int64), np.asarray(v, np.int64)
        lo_end, hi_end = WINDOW_ENDS.get(name, (0, WINDOW_SPAN - 1))
        rr = np.minimum(np.asarray(r, np.int64), np.maximum(0, np.minimum(v - lo_end, hi_end - v)))
        base = b * WINDOW_SPAN
        return (np.searchsorted(key, base + np.clip(v - rr, 0, WINDOW_SPAN - 1), "left"),
                np.searchsorted(key, base + np.clip(v + rr, 0, WINDOW_SPAN - 1), "right"))

    def near_counts(self, chain, v, r, name):
        """(states, levels) each level's group's real records within r of the state's own value (span;
        0 where the level has no group)."""
        out = np.zeros(chain.shape)
        v, r = np.asarray(v, np.int64), np.broadcast_to(np.asarray(r, np.int64), np.shape(v))
        for li in range(chain.shape[1]):
            ok = np.flatnonzero(chain[:, li] >= 0)
            if len(ok):
                lo, hi = self.span(chain[ok, li], v[ok], r[ok], name)
                out[ok, li] = hi - lo
        return out

    def bin_blend(self, cols, rng, prior):
        """The group each draw takes with the levels blended (blend_weights): a cell of n real records
        is drawn from n / (n + prior) of the time, else the next level's for the state, and so on, so a
        thin cell weighs as much as its records say and nothing jumps at a threshold."""
        n = len(np.asarray(cols[self.features[0]]))
        out = np.full(n, -1, np.int64)
        todo = np.arange(n)
        last = len(self.levels) - 1
        for li, lv in enumerate(self.levels):
            if not len(todo):
                break
            elig, first = self.codes[li]
            if not len(elig):
                continue
            if lv:
                rc = np.ravel_multi_index(tuple(np.asarray(cols[f], np.int64)[todo] for f in lv), self._dims(lv),
                                          mode="clip")
            else:
                rc = np.zeros(len(todo), np.int64)
            k = np.minimum(np.searchsorted(elig, rc), len(elig) - 1)
            hit = np.flatnonzero(elig[k] == rc)
            g = first + k[hit]
            take = np.ones(len(hit), bool) if li == last else rng.random(len(hit)) < self.count[g] / (self.count[g] + prior)
            out[todo[hit[take]]] = g[take]
            keep = np.ones(len(todo), bool)
            keep[hit[take]] = False
            todo = todo[keep]
        return out

    def draw(self, b, u, k=None):
        """The record each draw lands on: u uniform, leaned toward the top of its group by k."""
        if k is not None:
            u = 1.0 - np.power(np.clip(1.0 - u, 0.0, 1.0), k)
        n = self.count[b]
        return self.start[b] + np.minimum(n - 1, (u * n).astype(np.int64))

    def draw_low(self, b, u, value, limit):
        """A draw from the part of each group at or under `limit` (value: the payload the group is
        sorted by); a group with none there draws as usual."""
        cache = self.__dict__.setdefault("_low", {})
        if limit not in cache:
            cache[limit] = np.add.reduceat((self.payload[value] <= limit).astype(np.int64), self.start) \
                if len(self.start) else np.zeros(0, np.int64)
        n = cache[limit][b]
        full = self.count[b]
        n = np.where(n > 0, n, full)
        return self.start[b] + np.minimum(n - 1, (u * n).astype(np.int64))

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_by_") and k != "_identity"}

    def by_value(self, name):
        """(order, key): every group's records by a payload value (whole numbers, 0 to WINDOW_SPAN - 1)
        -- order the payload positions so sorted, key each one's group * WINDOW_SPAN + value -- each
        group in its own place."""
        got = self.__dict__.get("_by_" + name)
        if got is None:
            g_of = np.repeat(np.arange(len(self.count)), self.count)
            v = np.clip(np.asarray(self.payload[name], np.int64), 0, WINDOW_SPAN - 1)
            o = np.lexsort((v, g_of))
            got = (o, g_of[o] * WINDOW_SPAN + v[o])
            self.__dict__["_by_" + name] = got
        return got

    def by_field(self):
        return self.by_value("field")

    def window(self, b, v, r, name, nmin=None):
        """(lo, hi) each draw's window in by_value(name) order: its group's (b) real records within r of
        the state's own value v, the same either side: where an end of the scale is nearer than r
        (WINDOW_ENDS: the goal line, the half's end), the window stops as far short of it on the other
        side, so a state at the 1 is not drawn from the 1 to the 5 alone. Where fewer than nmin are in
        it, it widens the same either side, as far as that end allows; only where none are left there,
        to the nearest nmin (nmin: FIELD_MIN, CLOCK_MIN for the time left)."""
        nmin = np.asarray(nmin if nmin is not None else CLOCK_MIN if name == "left" else FIELD_MIN)
        _, key = self.by_value(name)
        b, v = np.asarray(b, np.int64), np.asarray(v, np.int64)
        lo_end, hi_end = WINDOW_ENDS.get(name, (0, WINDOW_SPAN - 1))
        cap = np.maximum(0, np.minimum(v - lo_end, hi_end - v))
        rr = np.minimum(np.asarray(r, np.int64), cap)
        base = b * WINDOW_SPAN
        find = lambda i, rad: (np.searchsorted(key, base[i] + np.clip(v[i] - rad, 0, WINDOW_SPAN - 1), "left"),
                               np.searchsorted(key, base[i] + np.clip(v[i] + rad, 0, WINDOW_SPAN - 1), "right"))
        everyone = np.arange(len(b))
        lo, hi = find(everyone, rr)
        s, n = self.start[b], self.count[b]
        w = np.minimum(nmin, n)
        thin = np.flatnonzero((hi - lo < w) & (rr < cap))
        while len(thin):
            rr[thin] = np.minimum(cap[thin], np.maximum(rr[thin] + 1, 2 * rr[thin]))
            lo[thin], hi[thin] = find(thin, rr[thin])
            thin = thin[(hi[thin] - lo[thin] < w[thin]) & (rr[thin] < cap[thin])]
        empty = np.flatnonzero(hi <= lo)
        if len(empty):
            p = np.searchsorted(key, base[empty] + v[empty], "left")
            lo[empty] = np.clip(p - w[empty] // 2, s[empty], s[empty] + n[empty] - w[empty])
            hi[empty] = lo[empty] + w[empty]
        return lo, hi

    def near(self, b, y, share, ymin=FIELD_YARDS, nmin=FIELD_MIN):
        """window on the field: within share x the yards to go (ymin at least) of field y."""
        y = np.asarray(y, np.int64)
        return self.window(b, y, field_radius(share, y, ymin), "field", nmin)

    def tilted(self, order, lo, hi, u, theta, rng, m=TILT_CANDIDATES):
        """A record from each draw's span lo..hi of `order` reweighted by exp(theta . phi) (payload
        "phi"): m drawn evenly from the span, one of them taken on its weight (u uniform)."""
        width = hi - lo
        U = rng.random((len(lo), m))
        cand = order[lo[:, None] + np.minimum(width[:, None] - 1, (U * width[:, None]).astype(np.int64))]
        z = np.einsum("nmd,nd->nm", self.payload["phi"][cand].astype(float), np.asarray(theta, float))
        w = np.exp(z - z.max(1, keepdims=True))
        c = np.cumsum(w, 1)
        pick = np.minimum(m - 1, (c < np.asarray(u)[:, None] * c[:, -1:]).sum(1))
        return cand[np.arange(len(lo)), pick]

    def draw_tilted(self, b, u, theta, rng):
        """draw from the whole group, reweighted by theta (tilted)."""
        b = np.asarray(b, np.int64)
        order = self.__dict__.get("_identity")
        if order is None or len(order) != len(self.payload["phi"]):
            order = self.__dict__["_identity"] = np.arange(len(self.payload["phi"]))
        return self.tilted(order, self.start[b], self.start[b] + self.count[b], u, theta, rng)

    def draw_window(self, b, v, r, u, name, k=None, rng=None, m=FIELD_CANDIDATES, theta=None, nmin=None):
        """The record each draw lands on from its window (window, nmin its fewest), u uniform; k: leaned
        toward the top of its group as draw's is, by resampling m of the window on their place in the
        group; theta: reweighted by it instead (tilted)."""
        order, _ = self.by_value(name)
        lo, hi = self.window(b, v, r, name, nmin)
        width = hi - lo
        if theta is not None:
            return self.tilted(order, lo, hi, u, theta, rng)
        if k is None:
            return order[lo + np.minimum(width - 1, (u * width).astype(np.int64))]
        U = rng.random((len(lo), m))
        cand = order[lo[:, None] + np.minimum(width[:, None] - 1, (U * width[:, None]).astype(np.int64))]
        b = np.asarray(b, np.int64)
        rank = (cand - self.start[b][:, None] + 0.5) / self.count[b][:, None]
        w = np.power(np.clip(1.0 - rank, 1e-9, 1.0), 1.0 / np.asarray(k, float)[:, None] - 1.0)
        c = np.cumsum(w, 1)
        pick = np.minimum(m - 1, (c < np.asarray(u)[:, None] * c[:, -1:]).sum(1))
        return cand[np.arange(len(lo)), pick]

    def draw_near(self, b, y, u, share, k=None, rng=None, m=FIELD_CANDIDATES, theta=None, nmin=None):
        """draw_window on the field (near)."""
        y = np.asarray(y, np.int64)
        return self.draw_window(b, y, field_radius(share, y), u, "field", k, rng, m, theta, nmin)

    def above(self, j, u):
        """A record at or above record j in its group (j's group read off its position), u uniform."""
        g = np.searchsorted(self.start, j, side="right") - 1
        end = self.start[g] + self.count[g]
        return j + np.minimum(end - j - 1, (u * (end - j)).astype(np.int64))

    def tilted_means(self, values, grid):
        """(bins, len(grid)) each group's mean value under each log tilt of the grid."""
        n = np.repeat(self.count, self.count)
        i = np.arange(len(values)) - np.repeat(self.start, self.count)
        out = np.zeros((len(self.count), len(grid)))
        for g, lk in enumerate(grid):
            inv = math.exp(-lk)
            G = lambda x: 1.0 - np.power(np.clip(1.0 - x, 0.0, 1.0), inv)
            w = G((i + 1) / n) - G(i / n)
            out[:, g] = np.add.reduceat(w * values, self.start) if len(values) else 0.0
        return out

    def level_shares(self, b):
        """The share of draws (their groups b) at each level."""
        lv = np.bincount(self.level_of[b[b >= 0]], minlength=len(self.levels))
        return lv / max(1, lv.sum())


def fit_leans(base, chosen, who, n_who, prior=DECISION_PRIOR, steps=8):
    """(n_who, options) each gamer's log-odds shift of each option against the first: Newton steps on
    their real decisions' likelihood around the league's shares (base: each decision's log shares),
    with a normal prior of precision `prior` pulling every shift toward none."""
    K = base.shape[1]
    S = np.zeros((n_who, K))
    Y = np.eye(K)[chosen]
    for _ in range(steps):
        z = base + S[who]
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(1, keepdims=True)
        for k in range(1, K):
            g = np.bincount(who, Y[:, k] - p[:, k], minlength=n_who) - prior * S[:, k]
            h = np.bincount(who, p[:, k] * (1 - p[:, k]), minlength=n_who) + prior
            S[:, k] += g / h
        S[-1] = 0.0                                       # the unknown gamer: none
    return S


class Choice:
    """A decision drawn from real ones: the league's share of each option among the real decisions in
    states alike (Bins over its features), leaned by the deciding gamer's log-odds shift (lean). Blended
    (blend > 0): every cell of BLEND_MIN real decisions or more kept, a state's shares its cells' over
    the levels, each weighted as Bins.blend_weights has it (a cell of n weighs n / (n + blend))."""

    def __init__(self, name, bins, shares, lean, league, n, counts, ruled=0, broke=0):
        self.name, self.bins, self.shares, self.lean, self.league, self.n = name, bins, shares, lean, league, n
        self.counts = counts                              # real decisions by gamer and option
        self.ruled, self.broke = ruled, broke             # real decisions under a global rule, and how many broke it

    @staticmethod
    def _masked(base, allowed):
        """Shares with the options a rule strikes out taken away, the rest scaled up (all of them
        equal where none of the allowed ones was ever chosen)."""
        if allowed is None:
            return base
        m = base * allowed
        tot = m.sum(1, keepdims=True)
        return np.where(tot > 0, m / np.where(tot > 0, tot, 1), allowed / allowed.sum(1, keepdims=True))

    def base(self, cols):
        """(states, options) the league's shares in each state's cells (no lean, no rule)."""
        n = len(np.asarray(next(iter(cols.values()))))
        if self.bins is None:
            return np.tile(self.shares[0], (n, 1))
        blend = getattr(self, "blend", 0.0)
        if not blend:
            return self.shares[self.bins.bin_of(cols)]
        C = self.bins.chain(cols)
        W = self.bins.blend_weights(C, blend)
        out = np.zeros((n, self.shares.shape[1]))
        for li in range(C.shape[1]):
            ok = C[:, li] >= 0
            if ok.any():
                out[ok] += W[ok, li, None] * self.shares[C[ok, li]]
        return out

    @classmethod
    def build(cls, name, X, chosen, who, n_who, min_records=DECISION_MIN, prior=DECISION_PRIOR, gamers=True,
              allowed=None, blend=0.0):
        """allowed: (decisions, options) what the rules left each real decision (None: no rules); blend: the
        cells blended with this prior (0: the first level with min_records)."""
        spec = DECISIONS[name]
        K = len(spec["options"])
        chosen, who = np.asarray(chosen, np.int64), np.asarray(who, np.int64)
        counts = np.zeros((n_who, K))
        np.add.at(counts, (who, chosen), 1)
        if not len(chosen):
            league = np.eye(K)[0]
            return cls(name, None, league[None], np.zeros((n_who, K)), league, 0, counts)
        league = np.bincount(chosen, minlength=K) / len(chosen)
        X = np.asarray(X, np.int64).reshape(len(chosen), -1)
        bins = Bins.build(spec["features"], spec["sizes"], spec["levels"], X, chosen.astype(float),
                          {"choice": chosen}, BLEND_MIN if blend else min_records)
        group = np.repeat(np.arange(len(bins.count)), bins.count)
        got = np.zeros((len(bins.count), K))
        np.add.at(got, (group, bins.payload["choice"]), 1)
        shares = (got + league) / (bins.count[:, None] + 1.0)
        bins.payload = {}
        cols = {f: X[:, i] for i, f in enumerate(spec["features"])}
        ruled = broke = 0
        keep = np.ones(len(chosen), bool)
        if allowed is not None and not allowed.all():
            under = ~allowed.all(1)
            keep = allowed[np.arange(len(chosen)), chosen]
            ruled, broke = int(under.sum()), int((~keep).sum())
        lean = np.zeros((n_who, K))
        out = cls(name, bins, shares, lean, league, len(chosen), counts, ruled, broke)
        out.blend = float(blend or 0.0)
        mine = out.base(cols)
        if gamers and keep.any():
            base = cls._masked(mine, None if allowed is None or allowed.all() else allowed)[keep]
            out.lean = fit_leans(np.log(np.maximum(base, 1e-9)), chosen[keep], who[keep], n_who, prior)
        out.real_base = mine.astype(np.float32)            # the real decisions' shares, for gamer_rate
        return out

    def gamer_rate(self, g, opt):
        """How often gamer g would take the option over every real decision's state (their lean on the
        league's shares there)."""
        base = getattr(self, "real_base", None)
        if base is None and getattr(self, "real_bins", None) is not None:
            base = self.shares[self.real_bins]
        if base is None or self.bins is None:
            return float(self.league[opt])
        z = np.log(np.maximum(base, 1e-12)) + self.lean[g]
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return float((p[:, opt] / p.sum(1)).mean())

    def probs(self, cols, who, allowed=None):
        base = self._masked(self.base(cols), allowed)
        z = np.log(np.maximum(base, 1e-12)) + self.lean[who]
        if allowed is not None:
            z = np.where(allowed, z, -np.inf)
        z -= z.max(1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(1, keepdims=True)

    def draw(self, cols, who, u, allowed=None):
        c = np.cumsum(self.probs(cols, who, allowed), 1)
        return np.minimum((u[:, None] > c).sum(1), c.shape[1] - 1)


# ---------------------------------------------------------------- the real snaps

def _side_of(team, a_home):
    return 0 if (team == "TEAM_A") == a_home else 1


def _with_side(rows):
    return [r if r.get("team_a_side") in ("home", "away") else dict(r, team_a_side="home") for r in rows]


def snap_records(rows):
    """Every scrimmage snap of a match, as sim14.snap_records reads them, with what the next row says
    the down became: `converted` (a new set of downs, or a touchdown) and `replay` (the same down again,
    the gain short of the line: a penalty). Spots are whole yards, so a gain of 0 can move the chains (a
    sneak for inches) and one of the yards to go can fall short; sim14 left out the 4th downs that kept
    the ball short of the line, which these are kept."""
    _i, _f = s14._i, s14._f
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] not in s14.SNAP_KINDS or not a["down"] or not a["field_position"]:
            continue
        if not a["period"] or a["period"] != b["period"] or not a["clock_seconds"] \
                or not b["clock_seconds"] or not a["distance"]:
            continue
        seconds = _f(a["clock_seconds"]) - _f(b["clock_seconds"])
        if not 0 <= seconds <= 60:
            continue
        down, t, y = _i(a["down"]), max(1, _i(a["distance"])), _i(a["field_position"])
        rec = dict(offense=a["offense"], period=_i(a["period"]), clock=_f(a["clock_seconds"]),
                   margin=s14._margin(a, a["offense"]), down=down, distance=t, field=y, seconds=seconds,
                   message=_i(b["message"]), kind=GAIN, gain=0, new_field=0, replay=False, converted=False,
                   fresh=s14.is_fresh(a), start=_i(a["message"]))
        if rec["margin"] is None or rec["clock"] is None or y is None or down is None:
            continue
        bk = b["play_kind"]
        if bk == "TOUCHDOWN":
            scorer = s14._scorer(b, "TOUCHDOWN_TEAM") or b["offense"]
            if scorer == a["offense"]:
                rec["gain"], rec["converted"] = 100 - y, True
            else:
                rec["kind"] = DEF_TD
        elif bk in ("SCRIMMAGE", "TURNOVER_ON_DOWNS") and not _scored(a, b):
            if not b["field_position"]:
                continue
            by = _i(b["field_position"])
            if b["offense"] == a["offense"]:
                rec["gain"] = by - y
                nd = _i(b["down"])
                rec["replay"] = nd == down and by - y < t
                rec["converted"] = nd == 1
            elif bk == "TURNOVER_ON_DOWNS" or (down == 4 and bk == "SCRIMMAGE"
                                               and "TURNOVER_ON_DOWNS" in (b["play_messages"] or "")):
                rec["gain"] = 100 - by - y
            else:
                rec["kind"] = TURNOVER
                rec["new_field"] = by
        else:
            # no TOUCHDOWN row, but the score after the snap says one: 6 or more to the side with the ball
            # (its own touchdown) or to the other (a defensive one), none to the one that did not score
            pts = _scored(a, b)
            if not pts:
                continue
            if pts[0] >= 6:
                rec["gain"], rec["converted"] = 100 - y, True
            else:
                rec["kind"] = DEF_TD
            rec["read_off_score"] = True
        s14._timeouts_at(rec, a)
        out.append(rec)
    return out


def _scored(a, b):
    """(the offense's points, the other's) from row a to row b where one side scored a touchdown's 6 or
    more and the other nothing (a scoring snap whatever row the feed gave it), else None."""
    pts = end_points(a, b)
    if pts is None:
        return None
    return pts if (pts[0] >= 6 and pts[1] == 0) or (pts[1] >= 6 and pts[0] == 0) else None


def match_snaps(rows, pair=None, teams=None):
    """A match's scrimmage snaps with the features both draws read and who had the ball: a list of
    dicts (sim14.snap_records plus side, prev_cls, elapsed, remain, cut, play_seconds, the gamers and
    teams)."""
    rows = _with_side(rows)
    if not rows:
        return []
    a_home = rows[0]["team_a_side"] != "away"
    recs = s14.link_prev(snap_records(rows))
    for n, r in enumerate(recs):
        r["side"] = _side_of(r["offense"], a_home)
        r["elapsed"] = n
        r["prev_raw"] = FRESH if (r.get("fresh") or r["prev"] < 0) else int(r["prev"])
        r["prev_cls"] = TIMEOUT if r.get("timeout_role") is not None else r["prev_raw"]
        c, s = r["clock"], r["seconds"]
        running = r["prev_cls"] <= LOSS
        warned = running and r["period"] in (2, 4) and c > LATE and c - s < LATE
        r["cut"] = (c - s <= 0.5) or warned                   # the quarter's end or 2:00 cut it short
        r["play_seconds"] = max(1.0, s - (c - LATE)) if warned else s
        r["gamer"] = (pair[r["side"]], pair[1 - r["side"]]) if pair else ("", "")
        r["team"] = (teams[r["side"]], teams[1 - r["side"]]) if teams else ("", "")
    halves = defaultdict(list)
    for r in recs:
        halves[0 if r["period"] <= 2 else 1 if r["period"] <= 4 else 2].append(r)
    for h in halves.values():
        for k, r in enumerate(h):
            r["remain"] = len(h) - k - 1
    return recs


TEAMS = ("TEAM_A", "TEAM_B")


def _tos_left(row, team, period):
    """(the side's timeouts left, the other side's) at a row, or (-1, -1) where the row doesn't carry
    them (playover.annotate_timeouts)."""
    a, b = s14._i(row.get("timeouts_used_a")), s14._i(row.get("timeouts_used_b"))
    if a is None or b is None or team not in TEAMS:
        return -1, -1
    have = TIMEOUTS if period <= 4 else OT_TIMEOUTS
    mine, theirs = (a, b) if team == "TEAM_A" else (b, a)
    return max(0, have - mine), max(0, have - theirs)


def _prev_at(row, by_end, period):
    """The play before a PLAY_OVER as the next snap's clock sees it (PREV), off the match's snaps."""
    last = by_end.get(s14._i(row["message"]))
    if s14.is_fresh(row) or last is None or last["offense"] != row.get("offense") or last["period"] != period \
            or last["kind"] != GAIN or last["gain"] >= 100 - last["field"]:
        return FRESH
    return int(play_class(last["gain"], last["distance"]))


def end_points(row, after, final=None):
    """(the offense's points, the other side's) from a half's last row to the first row after it (after),
    or to the game's final score (final: (home, away)) where it was regulation's last; None where
    neither is known. The feed can miss the half's last play -- a kick at the buzzer -- but not the score
    it left (the next half's rows, the history's final)."""
    if after is not None:
        h1, w1 = s14._i(after.get("score_p1")), s14._i(after.get("score_p2"))
    elif final is not None:
        h1, w1 = int(final[0]), int(final[1])
    else:
        return None
    h0, w0 = s14._i(row.get("score_p1")), s14._i(row.get("score_p2"))
    if None in (h0, w0, h1, w1) or row.get("offense") not in TEAMS:
        return None
    mine_home = (row["offense"] == "TEAM_A") == ((row.get("team_a_side") or "home") != "away")
    return (h1 - h0, w1 - w0) if mine_home else (w1 - w0, h1 - h0)


def match_final(hist_row):
    """The history's (home, away) final score of a match, or None."""
    try:
        return float(hist_row["PLAYER_1_FINAL_SCORE"]), float(hist_row["PLAYER_2_FINAL_SCORE"])
    except (KeyError, TypeError, ValueError):
        return None


def decision_points(rows, recs=None, final=None):
    """A match's real decisions in their full state: {name: [dict]} for every name in DECISIONS, each
    with the deciding side (0 home, 1 away), the choice, the period, clock, seconds left in the half
    (left), the deciding side's lead, yards to the goal (togo), to go (dist), the down, both sides'
    timeouts left (own_to, opp_to; -1 unknown), the play before (prev) and the row (msg); and
    "kick_pool" (choice, desperate, late, field, seconds) for every kick-off, "fg_clock" (seconds left in
    the half, seconds used) for every field goal and "conv" (two, good) for every conversion. `recs`:
    the match's match_snaps; final: the game's (home, away) final score, for the end of regulation.
    At the end of a half the score after it says what the feed missed (end_points): 3 more to the side
    with the ball and none to the other is its kick at the buzzer."""
    rows = _with_side(rows)
    out = {k: [] for k in DECISIONS}
    out.update(kick_pool=[], fg_clock=[], conv=[])
    if not rows:
        return out
    recs = recs if recs is not None else match_snaps(rows)
    by_end = {r["message"]: r for r in recs}
    a_home = rows[0]["team_a_side"] != "away"
    _i, _f = s14._i, s14._f
    regulation = final is not None and not any((_i(r.get("period")) or 0) >= 5 for r in rows)
    out["missed_kicks"] = 0
    for a, b in zip(rows, rows[1:] + [None]):
        if a["play_kind"] not in s14.SNAP_KINDS or a.get("offense") not in TEAMS or not a.get("field_position") \
                or not a.get("down") or not a.get("period") or not a.get("clock_seconds"):
            continue
        p, c, y, d = _i(a["period"]), _f(a["clock_seconds"]), _i(a["field_position"]), _i(a["down"])
        m = s14._margin(a, a["offense"])
        if c <= 0 or m is None:
            continue
        if b is None and not (regulation and p == 4):
            continue
        same = b is not None and a["period"] == b["period"]
        # the half's end after this row: what the score did from here says whether the side kicked
        pts = end_points(a, b, final if b is None else None) if not same and p in (2, 4) else None
        kicked = pts is not None and pts == (3, 0) and float(half_left(p, c)) <= EARLY_LEFT
        # in the quarter: a kick the feed gave no FIELD_GOAL row, read off the score after it
        if same and b["play_kind"] not in ("FIELD_GOAL", "PUNT") and end_points(a, b) == (3, 0):
            kicked = True
        b = b if b is not None else dict(play_kind="", period=None)
        side, left = _side_of(a["offense"], a_home), float(half_left(p, c))
        if b["play_kind"] == "FIELD_GOAL" and same and b.get("clock_seconds"):
            out["fg_clock"].append((left, c - _f(b["clock_seconds"])))
        own, opp = _tos_left(a, a["offense"], p)
        point = dict(side=side, period=p, clock=c, left=left, lead=m, togo=100 - y,
                     dist=max(1, _i(a["distance"]) or 1), down=d, own_to=own, opp_to=opp,
                     prev=_prev_at(a, by_end, p), msg=_i(a["message"]))
        if d == 4 and (same or kicked) and a.get("distance"):
            point["choice"] = FG if kicked else PUNT if b["play_kind"] == "PUNT" \
                else FG if b["play_kind"] == "FIELD_GOAL" else GO
            out["fourth"].append(point)
            out["missed_kicks"] += int(kicked)
        elif d in (1, 2, 3) and 100 - y + 17 <= FG_MAX and (same or p in (2, 4)):
            # a half that ran out before the next row is a side that did not kick, unless the score
            # after it says it did
            point["choice"] = int((b["play_kind"] == "FIELD_GOAL" and same) or kicked)
            out["early_fg"].append(point)
            out["missed_kicks"] += int(kicked)
    # kick-offs: after a conversion or a field goal (a touchdown straight after: returned or run in), and
    # each half's opening kick
    prev = None
    for r in rows:
        kind_r, got = r["play_kind"], None
        after = prev is not None and prev["period"] == r["period"] and prev["play_kind"] in ("CONVERSION", "FIELD_GOAL") \
            and bool(prev.get("clock_seconds")) and prev.get("offense") in TEAMS
        if r.get("period") and r.get("clock_seconds"):
            if after and kind_r == "TOUCHDOWN":
                kicker = prev["offense"]
                scorer = s14._scorer(r, "TOUCHDOWN_TEAM") or r["offense"]
                got = (kicker, _f(prev["clock_seconds"]), s14._margin(prev, kicker), 100,
                       KEPT if scorer == kicker else RETURN, prev)
            elif kind_r == "KICKOFF" and r.get("field_position") and r.get("offense") in TEAMS:
                if after:
                    kicker, start, at = prev["offense"], _f(prev["clock_seconds"]), prev
                    lead = s14._margin(prev, kicker)
                elif prev is None or prev["period"] != r["period"]:
                    kicker, start, at = ("TEAM_B" if r["offense"] == "TEAM_A" else "TEAM_A"), QUARTER, r
                    lead = s14._margin(r, kicker)
                else:
                    kicker = None
                if kicker:
                    field, secs = _i(r["field_position"]), start - _f(r["clock_seconds"])
                    choice = KEPT if r["offense"] == kicker else TOUCHBACK if (field == KICK_TOUCHBACK and secs == 0) \
                        else NO_LANDING if (field == KICK_NO_LANDING and secs == 0) else RETURN
                    if s14._scorer(r, "TOUCHDOWN_TEAM") is not None:
                        field = 100                       # run in from the kick
                    got = (kicker, start, lead, field, choice, at)
        if got is not None and got[2] is not None:
            kicker, start, lead, field, choice, at = got
            p, secs = _i(r["period"]), start - _f(r["clock_seconds"])
            if 0 <= secs <= 60:
                left = float(half_left(p, start))
                desperate = int(after and p >= 4 and lead < 0 and left <= 180)
                late = int(p in (2, 4) and start <= LATE)
                own, opp = _tos_left(at, kicker, p)
                out["kickoff"].append(dict(side=_side_of(kicker, a_home), choice=choice, period=p, clock=start,
                                           left=left, lead=lead, own_to=own, opp_to=opp, desperate=desperate,
                                           late=late, msg=_i(r["message"])))
                out["kick_pool"].append((choice, desperate, late, field, secs))
        prev = r
    # conversions: who scored, the lead after the six, whether they went for two and made it
    for a, b in zip(rows, rows[1:]):
        if a["play_kind"] != "TOUCHDOWN" or b["play_kind"] != "CONVERSION" or not a.get("period"):
            continue
        msgs = [x for x in (b.get("play_messages") or "").split("|") if "POINT" in x and x[-6:] in TEAMS]
        if not msgs:
            continue
        scorer = msgs[-1][-6:]
        two = any("TWO_POINT" in x for x in msgs)
        good = msgs[-1].startswith(("EXTRA_POINT_GOOD", "TWO_POINT_CONVERSION_SUCCESSFUL"))
        m = s14._margin(a, scorer)
        if m is None:
            continue
        p, c = _i(a["period"]), _f(a.get("clock_seconds")) or 0.0
        own, opp = _tos_left(a, scorer, p)
        out["two"].append(dict(side=_side_of(scorer, a_home), choice=int(two), period=p, clock=c,
                               left=float(half_left(p, c)), lead=m, own_to=own, opp_to=opp,
                               cell=s14.conversion_cell(p, c), good=int(good), msg=_i(a["message"])))
        out["conv"].append((two, good, int(conv_late(s14.conversion_cell(p, c)))))
    # timeouts before a running-clock snap, with a timeout left to call: the defense's first, then
    # (the defense not calling) the offense's
    for r in recs:
        if r["prev_raw"] > LOSS or not r.get("timeouts_left"):
            continue
        base = dict(period=r["period"], clock=r["clock"], left=float(half_left(r["period"], r["clock"])),
                    togo=100 - r["field"], dist=r["distance"], down=r["down"], prev=r["prev_raw"],
                    off_margin=r["margin"], msg=r["start"])
        role = r.get("timeout_role")
        off_left, def_left = r["timeouts_left"]
        if def_left > 0:
            out["to_def"].append(dict(base, side=1 - r["side"], choice=int(role == s14.DEFENCE), lead=-r["margin"],
                                      own_to=def_left, opp_to=off_left))
        if off_left > 0 and role != s14.DEFENCE:
            out["to_off"].append(dict(base, side=r["side"], choice=int(role == s14.OFFENCE), lead=r["margin"],
                                      own_to=off_left, opp_to=def_left))
    return out


def decision_features(name, pts):
    """The draws' features (DECISIONS[name]["features"]) of real decision points, as an int array."""
    if not pts:
        return np.zeros((0, len(DECISIONS[name]["features"])), np.int64)
    g = lambda k: np.array([q[k] for q in pts])
    if name == "fourth":
        mb = margin_b(g("lead"))
        cols = [togo10(100 - g("togo")), dist_b(g("dist")), mb, phase_of(g("period"), g("clock")),
                left_b(g("left")), fg4_of(g("lead")), derive(dict(dist=dist_b(g("dist"))))["dist3"],
                togo4(g("togo"))]
    elif name == "early_fg":
        p = g("period")
        mb = margin_b(g("lead"))
        cols = [early_b(g("left")), range_b(g("togo") + 17), g("down") - 1, mb,
                np.where(p == 2, 0, np.where(p == 4, 1, 2)), fg4_of(g("lead")), running_of(g("prev")),
                np.where(g("own_to") < 0, 2, np.minimum(2, g("own_to")))]          # not known: three, as remain
    elif name == "kickoff":
        cols = [g("desperate"), g("late"), lead3(g("lead"))]
    elif name == "two":
        cols = [g("cell"), np.clip(g("lead"), -CONV_MARGIN, CONV_MARGIN) + CONV_MARGIN, conv_late(g("cell"))]
    else:
        cols = [phase_of(g("period"), g("clock")), margin_b(g("off_margin")), left_b(g("left"))]
        if name == "to_off":
            cols.append(in_range(g("togo")))
    return np.stack([np.asarray(c, np.int64) for c in cols], axis=1)


def in_draw_window(name, pts):
    """Which real decision points v2.1's sim decides at (early kicks: downs 1-3 in the half's last
    minute of Q2, Q4 and overtime; the rest: all)."""
    if name != "early_fg":
        return np.ones(len(pts), bool)
    return np.array([(q["period"] == 2 or q["period"] >= 4) and q["left"] <= EARLY_LEFT for q in pts], bool)


def match_decisions(rows, recs=None):
    """A match's real decisions as the draws read them: {name: [(side, choice, *features)]} (only those
    the sim decides at), with decision_points' kick_pool, fg_clock and conv."""
    pts = decision_points(rows, recs)
    out = {k: pts[k] for k in ("kick_pool", "fg_clock", "conv")}
    for name in DECISIONS:
        keep = in_draw_window(name, pts[name])
        got = [q for q, k in zip(pts[name], keep) if k]
        X = decision_features(name, got)
        out[name] = [(q["side"], q["choice"]) + tuple(int(v) for v in x) for q, x in zip(got, X)]
    return out


def _turnover_shift(rec):
    """A real turnover's move of the spot, toward the defense's goal, from where it was snapped."""
    return (100 - int(rec["new_field"])) - int(rec["field"])


def _score_path(rows):
    """[(slice of regulation, home + away)] at every row with a score and a clock."""
    out = []
    for r in rows:
        p, c, h, a = s14._i(r.get("period")), s14._f(r.get("clock_seconds")), s14._i(r.get("score_p1")), \
            s14._i(r.get("score_p2"))
        if None not in (p, c, h, a):
            out.append((int(share_slice(p, c)), h + a))
    return out


def points_share(matches):
    """The share of a game's points still to come at the start of each 20-second slice of regulation
    (the last slice: overtime's), the real games' points pooled."""
    left, final = np.zeros(SHARE_SLICES), 0.0
    for rows in matches.values():
        path = _score_path(rows)
        if not path:
            continue
        end = path[-1][1]
        final += end
        at, k = 0, 0
        for sl in range(SHARE_SLICES):
            while k < len(path) and path[k][0] < sl:
                at = path[k][1]
                k += 1
            left[sl] += end - at
    return left / final if final else np.linspace(1, 0, SHARE_SLICES)


# ---------------------------------------------------------------- the model

class Model:
    """Everything v2.1 draws from."""

    def __init__(self):
        self.mode = "full"
        self.split = SPLIT
        self.key = KEY
        self.blend = 0.0
        self.field = 0.0                   # the distance draws' field windows (0: off)
        self.near_blend = False            # blended, the distance draws' cells weighed by their snaps near the field
        self.decision_blend = 0.0          # the decisions' cells blended (0: off)
        self.clock = 0.0                   # the time draws' clock windows at a half's end (0: off)
        self.time = self.dist = None
        self.own = {}                      # full: {(who, role, "time"|"dist"): Bins}
        self.gamers, self.teams = {}, {}
        self.tilt = {}                     # adjusted: {"time"|"dist": {effect: arr}}
        self.gain_surv = np.ones(101)      # P(a positive gain reaches x yards)
        self.gain_surv_end = None          # the same by endgame3 (the end of a half by snaps left and lead), shrunk to it
        self.choices = {}                  # the decisions: {name in DECISIONS: Choice}
        self.fg_make = np.array([0.8])     # P(make) by 5-yard kick distance from 15
        self.fg_near = np.array([[5.0, 5.0]])   # (seconds left, seconds used) of real kicks FG_NEAR or less from the end
        self.fg_far = np.array([5.0])      # seconds used by the rest
        self.punts = None                  # Bins over (ybin,) payload net, seconds
        self.kick_pool = {}                # (RETURN | KEPT, desperate, late) -> (field, seconds)
        self.safety_fields = np.array([35])
        self.two_good, self.kick_good = 0.5, 0.97
        self.two_good_by = None            # a two-point try made, by conv_late (before Q4's last 3:00, in it, overtime)
        self.rules = {}                    # learned (v21_rules): {decision: [(option, {feature: (lo, hi)})]}
        self.use_rules = True
        self.remain = np.zeros(REMAIN_SHAPE)
        self.share = np.linspace(1, 0, SHARE_SLICES)
        self.meta = {}

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self.__dict__, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        m = cls()
        with open(path, "rb") as fh:
            got = pickle.load(fh)
        if "choices" not in got:
            raise SystemExit(f"{path} was built before v2.1's decisions: build it again with v21-build")
        m.__dict__.update(got)
        return m

    def gamer_index(self, handle):
        return self.gamers.get((handle or "").strip().upper(), len(self.gamers))

    def team_index(self, team):
        return self.teams.get((team or "").strip(), len(self.teams))

    def remain_at(self, period, clock, running, own_to, opp_to, lead3):
        """The expected snaps left in the half (remain_key)."""
        return self.remain[remain_key(period, clock, running, own_to, opp_to, lead3)]


MODEL_FILE = "v21model.pkl"


ZONE3 = np.array([0, 0, 0, 1, 1, 1, 2, 2])               # own half, the other half outside the 10, inside the 10


def late_of(period, clock):
    """The end of a half: 0 the rest, 1 Q2's or Q4's 2:00 to 0:31, 2 their last 0:30, 3 overtime."""
    period, clock = np.asarray(period), np.asarray(clock, float)
    end = (period == 2) | (period == 4)
    return np.select([period >= 5, end & (clock <= 30), end & (clock <= LATE)], [3, 2, 1], 0).astype(np.int64)


def endgame_of(phase, remain, margin, k=7):
    """The end of a half apart from the rest, and the first half's from the second's: 0 the bulk of the
    game (before the half's 2:00, 5+ snaps left); then the half's last 2:00 or under 5 snaps left, by
    the half, the snaps left (0-1, 2-4, 5+) and the lead (k of them: margin_b's 7, so a side 1-3 ahead
    plays it out apart from one 9+ ahead, or lead3's 3); then overtime by the lead. `remain` is the
    snaps-left bucket (remain_b). A level side at the end of Q2 attacks; at the end of Q4 it plays for
    the kick."""
    phase, remain, margin = np.asarray(phase), np.asarray(remain), np.asarray(margin)
    r3 = np.select([remain == 0, remain <= 2], [0, 1], 2)
    end = (phase == 2) | (phase == 5) | (remain <= 2)
    h = (phase >= 3).astype(np.int64)
    return np.where(phase == 6, 1 + 6 * k + margin, np.where(end, 1 + 3 * k * h + k * r3 + margin, 0)).astype(np.int64)


def endgame3_of(phase, remain, lead3):
    """endgame_of with the lead behind / level / ahead (the league draws' fallbacks)."""
    return endgame_of(phase, remain, lead3, k=3)


def derive(f):
    """The coarse features off the fine ones: the lead (lead3: behind, level, ahead), the clock (tb2:
    stopped, running), the distance (dist3: 1-3, 4-6, 7+), the field (zone3), the part of the game
    (part: the half and the two-minute warning; half) and the end of a half (endgame_of)."""
    if "margin" in f:
        f["lead3"] = np.select([f["margin"] <= 2, f["margin"] == 3], [0, 1], 2).astype(np.int64)
    if "prev" in f:
        f["clock2"] = running_of(f["prev"])
    if "phase" in f:
        f["part"] = PART_OF[np.asarray(f["phase"])]
        f["qtr"] = QTR_OF[np.asarray(f["phase"])]
    if "remain" in f:
        f["rem4"] = np.select([np.asarray(f["remain"]) == 0, np.asarray(f["remain"]) <= 2,
                               np.asarray(f["remain"]) <= 4], [0, 1, 2], 3).astype(np.int64)
        f["half"] = np.select([np.asarray(f["phase"]) <= 2, np.asarray(f["phase"]) <= 5], [0, 1], 2).astype(np.int64)
    if "remain" in f and "phase" in f and "margin" in f:
        f["endgame"] = endgame_of(f["phase"], f["remain"], f["margin"])
        f["endgame3"] = endgame3_of(f["phase"], f["remain"], f["lead3"])
    if "tb" in f:
        f["tb2"] = (np.asarray(f["tb"]) >= 3).astype(np.int64)
    if "dist" in f:
        f["dist3"] = np.minimum(2, np.maximum(0, np.asarray(f["dist"]) - 1)).astype(np.int64)
    if "zone" in f:
        f["zone3"] = ZONE3[np.asarray(f["zone"])]
    return f


def remain_inputs(recs):
    """(period, clock, running, own timeouts, the other side's, lead3) of real snaps, as the sim reads
    its own (timeouts not known: three each)."""
    get = lambda k: np.array([r[k] for r in recs])
    tos = np.array([r.get("timeouts_left") or (TIMEOUTS, TIMEOUTS) for r in recs], np.int64).reshape(-1, 2)
    mb = margin_b(get("margin"))
    return (get("period"), get("clock").astype(float), running_of(get("prev_cls")), tos[:, 0], tos[:, 1],
            derive(dict(margin=mb))["lead3"])


def _features(recs, model):
    """{feature: array} for real snaps (snaps left in the half off the model's table, for each snap's
    own state)."""
    get = lambda k: np.array([r[k] for r in recs])
    period, clock = get("period"), get("clock").astype(float)
    return derive(dict(prev=get("prev_cls"), phase=phase_of(period, clock),
                       margin=margin_b(get("margin")),
                       down=down_b(get("down")), dist=dist_b(get("distance")), zone=zone_b(get("field")),
                       elapsed=elapsed_b(get("elapsed")), remain=remain_b(model.remain_at(*remain_inputs(recs))),
                       tb=time_b(get("play_seconds"))))


def _cols(feats, names):
    return np.stack([feats[f] for f in names], axis=1)


def match_people(code, rows, hist):
    """((home gamer, away gamer), (home team, away team)) off the history, else the export."""
    h = hist.get(code, {})
    r0 = rows[0] if rows else {}
    pair = ((h.get("PLAYER_1_HANDLE") or r0.get("home_handle") or "").strip().upper(),
            (h.get("PLAYER_2_HANDLE") or r0.get("away_handle") or "").strip().upper())
    teams = ((h.get("PLAYER_1_TEAM") or "").strip(), (h.get("PLAYER_2_TEAM") or "").strip())
    return pair, teams


def build(matches, history=None, mode="full", min_records=MIN_RECORDS, verbose=True, refine=REFINE_REMAINING,
          seed=0, gamer_decisions=True, rules=True, split=SPLIT, key=KEY, blend=0.0, field=0.0, decision_blend=0.0,
          clock=0.0, tilt_prior=TILT_PRIOR):
    """The model off the export's matches ({code: rows}) and the history (gamers and teams).
    gamer_decisions=False: every gamer decides as the league does in the same state; rules=False: no
    global rules, the real shares alone; split: the draws' part of the game (SPLIT); key: the league's
    draws' key (KEY); blend: the league's draws' levels blended with this prior (0: off, BLEND_MIN);
    field: the distance draws' field windows, this share of the yards to go (0: off, FIELD_YARDS);
    decision_blend: the decisions' cells blended with this prior (0: off, Choice); clock: the time draws'
    clock windows in a half's last CLOCK_LATE seconds, this share of the seconds left (0: off); tilt_prior:
    the tilts' shrinkage, in snaps (TILT_PRIOR)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    draw_spec(split, key)
    m = Model()
    m.mode = mode
    m.split = split
    m.key = key
    m.blend = float(blend or 0.0)
    m.field = float(field or 0.0)
    m.near_blend = bool(NEAR_BLEND and m.field and m.blend)
    m.decision_blend = float(decision_blend or 0.0)
    m.clock = float(clock or 0.0)
    m.tilt_prior = float(tilt_prior)
    m.use_rules = bool(rules and RULES)
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    recs, kick_dec, safeties = [], [], []
    missed_kicks = 0
    gamer_points = defaultdict(list)                 # each gamer's side's real points a match (the nudge)
    dec = {k: [] for k in DECISIONS}
    dec.update(kick_pool=[], fg_clock=[], conv=[])
    ot_games = {}
    for code, rows in matches.items():
        rows = _with_side(rows)
        ot_games[code] = any((s14._i(r.get("period")) or 0) >= 5 for r in rows)
        pair, teams = match_people(code, rows, hist)
        fin = match_final(hist[code]) if code in hist else None
        if fin is not None:
            for side in (0, 1):
                if pair[side]:
                    gamer_points[pair[side]].append(fin[side])
        mine = match_snaps(rows, pair, teams)
        for r in mine:
            r["code"] = code
        recs += mine
        got = decision_points(rows, mine, match_final(hist[code]) if code in hist else None)
        missed_kicks += got["missed_kicks"]
        for name in DECISIONS:
            dec[name] += [dict(q, gamer=pair[q["side"]], code=code)
                          for q, k in zip(got[name], in_draw_window(name, got[name])) if k]
        for name in ("kick_pool", "fg_clock", "conv"):
            dec[name] += got[name]
        kick_dec += s14.kick_decisions(rows)
        safeties += s14.safety_kicks(rows)
    recs = [r for r in recs if r["period"] <= 4 + MAX_OT]
    if not recs:
        raise ValueError("no real snaps to build on")
    if verbose:
        print(f"  v2.1: {len(recs):,} real snaps from {len(matches):,} matches ({mode})", flush=True)
        print(f"  kicks at a half's end the feed missed, read off the score after it (3 more to the side with"
              f" the ball, none to the other): {missed_kicks:,}", flush=True)
        print(f"  scoring snaps with no TOUCHDOWN row, read off the score after them (6 or more to one side, none"
              f" to the other): {sum(1 for r in recs if r.get('read_off_score')):,}", flush=True)
        try:
            m._trend_lines = trend_lines(dec, ot_games)
        except (ValueError, IndexError):          # codes without a date
            m._trend_lines = []
    for r in recs:
        for g in r["gamer"]:
            if g and g not in m.gamers:
                m.gamers[g] = len(m.gamers)
        for t in r["team"]:
            if t and t not in m.teams:
                m.teams[t] = len(m.teams)

    # snaps left in the half, from the real games first
    m.remain = remain_table(remain_key(*remain_inputs(recs)), np.array([r["remain"] for r in recs], float))
    m.share = points_share(matches)

    # positive gains' survival, touchdowns counted as at least the yards the field allowed
    g = np.array([r["gain"] for r in recs if r["kind"] == GAIN and r["gain"] > 0], np.int64)
    td = np.array([r["gain"] >= 100 - r["field"] for r in recs if r["kind"] == GAIN and r["gain"] > 0], bool)
    if len(g):
        at_risk = np.bincount(np.minimum(g, 100), minlength=101)[::-1].cumsum()[::-1]
        ended = np.bincount(np.minimum(g[~td], 100), minlength=101)
        hazard = np.where(at_risk > 0, ended / np.maximum(at_risk, 1), 0.0)
        m.gain_surv = np.concatenate([[1.0], np.cumprod(1 - hazard[:-1])])
        m.gain_surv[:2] = 1.0
        # and by the end of a half (endgame3): a level side late in field-goal range plays for the kick,
        # its gains go less far than the league's; each shrunk toward the league's by SURV_PRIOR snaps
        pos = [r for r in recs if r["kind"] == GAIN and r["gain"] > 0]
        f = derive(dict(phase=phase_of([r["period"] for r in pos], np.array([r["clock"] for r in pos], float)),
                        remain=remain_b(m.remain_at(*remain_inputs(pos))), margin=margin_b([r["margin"] for r in pos])))
        cls = f["endgame3"]
        m.gain_surv_end = np.tile(m.gain_surv, (SIZES["endgame3"], 1))
        for c in np.unique(cls):
            w = cls == c
            ar = np.bincount(np.minimum(g[w], 100), minlength=101)[::-1].cumsum()[::-1]
            en = np.bincount(np.minimum(g[w & ~td], 100), minlength=101)
            hz = (en + SURV_PRIOR * hazard) / (ar + SURV_PRIOR)
            sv = np.concatenate([[1.0], np.cumprod(1 - hz[:-1])])
            sv[:2] = 1.0
            m.gain_surv_end[c] = sv

    _build_draws(m, recs, min_records, verbose)

    # the decisions: the rules learned off the real ones (where an option never or all but never happens),
    # then each gamer's lean on top of the league's choices in the same states
    from . import v21_rules
    m.rules = v21_rules.learn(dec) if m.use_rules else {}
    n_who = len(m.gamers) + 1
    for name, spec in DECISIONS.items():
        got = dec[name]
        who = np.array([m.gamer_index(q["gamer"]) for q in got], np.int64)
        chosen = np.array([q["choice"] for q in got], np.int64)
        X = decision_features(name, got)
        allowed = v21_rules.mask(m.rules, name, v21_rules.fine(name, got)) if got else None
        m.choices[name] = Choice.build(name, X, chosen, who, n_who, gamers=gamer_decisions, allowed=allowed,
                                       blend=spec.get("blend", m.decision_blend) if m.decision_blend else 0.0)
    # kick-offs' outcomes: the field and clock of the returned and the kept ones, by desperate and late
    pool = np.array(dec["kick_pool"], float).reshape(-1, 5)
    for kind in (RETURN, KEPT):
        for desperate in (0, 1):
            for late in (0, 1):
                wants = [(pool[:, 0] == kind) & (pool[:, 1] == desperate) & (pool[:, 2] == late),
                         (pool[:, 0] == kind) & (pool[:, 1] == desperate), pool[:, 0] == kind]
                pick = next((w for w in wants if w.sum() >= KICK_POOL_MIN), None)
                if pick is None:
                    pick = next((w for w in wants if w.any()), None)
                m.kick_pool[(kind, desperate, late)] = (pool[pick, 3].astype(np.int64), pool[pick, 4]) \
                    if pick is not None else (np.array([25 if kind == RETURN else 45]), np.array([5.0]))
    # field goals: made by distance, and the clock they take
    fgs = [d for d in kick_dec if d[0] == "fg"]
    dist_fg = np.arange(15, FG_MAX + 5, 5)
    made, tried = np.zeros(len(dist_fg)), np.zeros(len(dist_fg))
    for _, y, ok, secs, _ in fgs:
        k = min(len(dist_fg) - 1, max(0, int((100 - y + 17 - 15) // 5)))
        made[k] += bool(ok)
        tried[k] += 1
    league = made.sum() / max(1, tried.sum())
    m.fg_make = (made + 10 * league) / (tried + 10)
    clock_used = np.array(dec["fg_clock"], float).reshape(-1, 2)
    near = clock_used[:, 0] <= FG_NEAR
    if near.any():
        m.fg_near = clock_used[near]
    if (~near).any():
        m.fg_far = clock_used[~near, 1]
    punts = [d for d in kick_dec if d[0] == "punt"]
    if punts:
        P = np.array([(d[1], d[2], d[3]) for d in punts], float)
        m.punts = Bins.build(("ybin",), (5,), (("ybin",), ()), np.clip((P[:, 0] // 20).astype(int), 0, 4)[:, None],
                             P[:, 1], {"net": P[:, 1].astype(np.int64), "seconds": P[:, 2]}, 20)
    if safeties:
        m.safety_fields = np.array(safeties, np.int64)
    conv = dec["conv"]
    if conv:
        two = [g for t, g, _ in conv if t]
        kick = [g for t, g, _ in conv if not t]
        m.two_good = (sum(two) + 5) / (len(two) + 10)
        m.kick_good = (sum(kick) + 9) / (len(kick) + 10)
        # by the part of the game, each shrunk toward the whole game's by TWO_GOOD_PRIOR tries
        m.two_good_n = np.array([sum(1 for t, g, l in conv if t and l == k) for k in range(3)])
        m.two_good_by = np.array([(sum(g for t, g, l in conv if t and l == k) + TWO_GOOD_PRIOR * m.two_good)
                                  / (m.two_good_n[k] + TWO_GOOD_PRIOR) for k in range(3)])
    run = dec["to_def"]
    m.meta = dict(matches=len(matches), snaps=len(recs), mode=mode, split=split, key=key, blend=m.blend,
                  field=m.field, near_blend=m.near_blend, decision_blend=m.decision_blend, clock=m.clock,
                  tilt_prior=m.tilt_prior,
                  timeouts=bool(run),
                  built=dt.datetime.now().isoformat(timespec="seconds"))
    if refine:
        for r_ in range(REFINE_ROUNDS):
            before = m.remain.copy()
            refine_remaining(m, seed=seed + r_)
            _build_draws(m, recs, min_records, verbose=False)
            if verbose:
                print(f"  snaps left in the half, round {r_ + 1}: the sim's own counts, mean change"
                      f" {np.abs(m.remain - before).mean():.2f} snaps; the draws rebuilt on them", flush=True)
    if m.mode == "tilt":
        fit_model_tilts(m, recs, verbose)
        m._gamer_points = {m.gamers[g]: v for g, v in gamer_points.items() if g in m.gamers}
        m.nudge_dir = nudge_direction(m)
        slope = calibrate_nudge(m, seed=seed)
        m.nudge_dir = m.nudge_dir / slope                   # a unit of nudge: a point a side
        m.nudge_slope = calibrate_nudge(m, seed=seed)
    if verbose:
        report_build(m, recs)
    return m


def draw_spec(split=SPLIT, key=KEY):
    """The draws' features and fallbacks for a split (SPLIT: "snaps" reads part where "time" reads
    phase) and a key (KEY: the league's draws on the full key or the lean one; the gamers' and teams'
    own as the split has them either way)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    if key not in KEYS:
        raise ValueError(f"key must be one of {', '.join(KEYS)}")
    sub = (lambda t: tuple("part" if x == "phase" else x for x in t)) if split == "snaps" else tuple
    subs = lambda levels: tuple(sub(lv) for lv in levels)
    out = dict(time=(sub(TIME_FEATURES), subs(TIME_LEVELS)), dist=(sub(DIST_FEATURES), subs(DIST_LEVELS)),
               gamer_time=(sub(GAMER_TIME_FEATURES), subs(GAMER_TIME_LEVELS)),
               gamer_dist=(sub(GAMER_DIST_FEATURES), subs(GAMER_DIST_LEVELS)))
    if key == "lean":
        out.update(time=(TIME_LEAN_FEATURES, TIME_LEAN_LEVELS), dist=(DIST_LEAN_FEATURES, DIST_LEAN_LEVELS))
    return out


def _build_draws(m, recs, min_records=MIN_RECORDS, verbose=True):
    """The time and distance draws (and the gamers' and teams' own or their tilts) on the model's
    current table of snaps left in the half."""
    spec = draw_spec(getattr(m, "split", "time"), getattr(m, "key", "full"))
    (time_f, time_l), (dist_f, dist_l) = spec["time"], spec["dist"]
    feats = _features(recs, m)
    seconds = np.array([min(MAX_SECONDS, r["seconds"]) for r in recs], float)
    cut = np.array([r["cut"] for r in recs], bool)
    left = np.array([half_left(r["period"], r["clock"]) for r in recs], float)
    time_pay = {"seconds": seconds.astype(np.float32), "cut": cut,        # compact: a record sits in a group
                "left": np.clip(np.round(left), 0, WINDOW_SPAN - 1).astype(np.int16)}   # at every level
    if m.mode == "tilt":
        time_pay["phi"] = (seconds / 10.0).astype(np.float32)[:, None]
    league_min = BLEND_MIN if getattr(m, "blend", 0) else min_records
    m.time = Bins.build(time_f, [SIZES[f] for f in time_f], time_l, _cols(feats, time_f), seconds, time_pay,
                        league_min)
    kind = np.array([r["kind"] for r in recs])
    gain = np.array([r["gain"] for r in recs], np.int64)
    field = np.array([r["field"] for r in recs], np.int64)
    shift = np.array([_turnover_shift(r) if r["kind"] == TURNOVER else 0 for r in recs], np.int64)
    dvalue = np.where(kind == DEF_TD, DEF_TD_VALUE, np.where(kind == TURNOVER, TO_VALUE, gain)).astype(float)
    made = np.array([2 if r["replay"] else int(r["converted"]) if r["kind"] == GAIN else 3 for r in recs], np.int8)
    dist_pay = {"made": made, "rdist": np.array([r["distance"] for r in recs], np.int16),
                "kind": kind.astype(np.int8), "gain": gain.astype(np.int16), "shift": shift.astype(np.int16),
                "td": (kind == GAIN) & (gain >= 100 - field), "value": dvalue.astype(np.float32),
                "field": field.astype(np.int16)}
    if m.mode == "tilt":
        dist_pay["phi"] = phi_dist_of(kind, gain, dist_pay["td"], made == 1).astype(np.float32)
    m.dist = Bins.build(dist_f, [SIZES[f] for f in dist_f], dist_l, _cols(feats, dist_f), dvalue, dist_pay,
                        league_min)

    who = dict(gamer_off=np.array([m.gamer_index(r["gamer"][0]) for r in recs]),
               gamer_def=np.array([m.gamer_index(r["gamer"][1]) for r in recs]),
               team_off=np.array([m.team_index(r["team"][0]) for r in recs]),
               team_def=np.array([m.team_index(r["team"][1]) for r in recs]))
    sizes = dict(gamer_off=len(m.gamers) + 1, gamer_def=len(m.gamers) + 1,
                 team_off=len(m.teams) + 1, team_def=len(m.teams) + 1)
    m.own, m.tilt = {}, {}
    if m.mode == "full":
        for name, ids in who.items():
            kind_, role = name.split("_")
            f = dict(feats, who=ids)
            for draw, features, levels, value, pay in (
                    ("time", *spec["gamer_time"], seconds, time_pay),
                    ("dist", *spec["gamer_dist"], dvalue, dist_pay)):
                m.own[(kind_, role, draw)] = Bins.build(
                    features, [sizes[name]] + [SIZES[x] for x in features[1:]], levels, _cols(f, features),
                    value, pay, MIN_GAMER_RECORDS, strict=True)
    elif m.mode == "adjusted":
        timed = ~cut
        b_time = m.time.bin_of({f: feats[f][timed] for f in time_f})
        m.tilt["time"] = fit_tilts(m.time, m.time.payload["seconds"], b_time, seconds[timed],
                                   {k: v[timed] for k, v in who.items()}, sizes)
        m.tilt["dist"] = fit_tilts(m.dist, m.dist.payload["value"], m.dist.bin_of(feats), dvalue, who, sizes)
    m._real_bins = (m.time.bin_of(feats), m.dist.bin_of(feats))


def fit_tilts(bins, values, bin_ids, observed, who, sizes, grid=ADJUST_GRID, prior=ADJUST_PRIOR,
              sweeps=ADJUST_SWEEPS):
    """{effect: log tilt per entity}: each gamer's and team's lean of the league's draws, fitted in
    turn (each with the others held) so the snaps' mean under the summed tilt meets the real one,
    shrunk toward none by `prior` snaps."""
    E = bins.tilted_means(values, grid)
    keep = bin_ids >= 0
    bin_ids, observed = bin_ids[keep], observed[keep]
    who = {k: v[keep] for k, v in who.items()}
    step = grid[1] - grid[0]
    effects = {k: np.zeros(sizes[k]) for k in who}
    for _ in range(sweeps):
        for name, ent in who.items():
            others = sum(effects[k][who[k]] for k in who if k != name)
            loss = np.zeros((sizes[name], len(grid)))
            for g, a in enumerate(grid):
                gi = np.clip(np.round((others + a - grid[0]) / step).astype(int), 0, len(grid) - 1)
                loss[:, g] = np.bincount(ent, (observed - E[bin_ids, gi]) ** 2, minlength=sizes[name])
            n = np.bincount(ent, minlength=sizes[name])
            best = grid[np.argmin(loss, axis=1)]
            effects[name] = np.where(n > 0, best * n / (n + prior), 0.0)
            effects[name][-1] = 0.0                       # the unknown gamer or team: none
    return effects


def _tilt_who(m, recs):
    """{effect: entity per real snap} and {effect: entities} (the unknown last) for the tilts."""
    who = dict(gamer_off=np.array([m.gamer_index(r["gamer"][0]) for r in recs], np.int64),
               gamer_def=np.array([m.gamer_index(r["gamer"][1]) for r in recs], np.int64),
               team_off=np.array([m.team_index(r["team"][0]) for r in recs], np.int64),
               team_def=np.array([m.team_index(r["team"][1]) for r in recs], np.int64))
    sizes = dict(gamer_off=len(m.gamers) + 1, gamer_def=len(m.gamers) + 1,
                 team_off=len(m.teams) + 1, team_def=len(m.teams) + 1)
    return who, sizes


def tilt_bases(b, feats, payload_bin, n_bins, blend, near=None, near_blend=False):
    """(states, n_bins) what the league's draw gives each state, by histogram bin: its groups' real
    records over the levels as bin_blend weighs them (blend 0: its first group), each group's records
    in the state's window where near is (values, radii, payload name: Bins.window) or all of them;
    records of bin -1 left out (the time's cut ones: only "at least this long"). near_blend: a group with
    none of its records in the window passes all on, no window widened but the last level's (dist_weights)."""
    n = len(np.asarray(feats[b.features[0]]))
    near_blend = bool(near_blend and blend and near is not None)
    if blend:
        C = b.chain(feats)
        W = b.blend_weights(C, blend, np.where(b.near_counts(C, near[0], near[1], near[2]) > 0,
                                               np.where(C >= 0, b.count[np.maximum(C, 0)], 0), 0)
                            if near_blend else None)
    else:
        g = b.bin_of(feats)
        C, W = g[:, None], (g >= 0).astype(float)[:, None]
    H = np.zeros((n, n_bins))
    if near is not None:
        order = b.by_value(near[2])[0]
        pb = np.asarray(payload_bin)[order]
    else:
        pb = np.asarray(payload_bin)
    pos = [np.flatnonzero(pb == k) for k in range(n_bins)]
    for li in range(C.shape[1]):
        ok = np.flatnonzero((C[:, li] >= 0) & (W[:, li] > 0))
        if not len(ok):
            continue
        g = C[ok, li]
        if near is not None:
            lo, hi = b.window(g, near[0][ok], near[1][ok], near[2],
                              1 if near_blend and li < C.shape[1] - 1 else None)
        else:
            lo, hi = b.start[g], b.start[g] + b.count[g]
        cnt = np.zeros((len(ok), n_bins))
        for k in range(n_bins):
            if len(pos[k]):
                cnt[:, k] = np.searchsorted(pos[k], hi) - np.searchsorted(pos[k], lo)
        tot = cnt.sum(1, keepdims=True)
        H[ok] += W[ok, li, None] * np.where(tot > 0, cnt / np.maximum(tot, 1), 0.0)
    tot = H.sum(1, keepdims=True)
    return np.where(tot > 0, H / np.where(tot > 0, tot, 1), 1.0 / n_bins)


def recency_weights(recs, half_life=TILT_HALF_LIFE):
    """Each real snap's weight in the tilts' fit: halving every half_life days before the build's last
    match day (1 for all where the codes carry no date, or half_life is 0)."""
    if not half_life:
        return np.ones(len(recs))
    from .remaining import day
    try:
        days = np.array([day(r.get("code", "")).toordinal() for r in recs], float)
    except (ValueError, IndexError, TypeError):
        return np.ones(len(recs))
    return np.power(0.5, (days.max() - days) / half_life)


def fit_exp_tilts(H, obs, phi_b, who, sizes, prior=TILT_PRIOR, rounds=TILT_ROUNDS, weights=None):
    """({effect: (entities, dims) tilt}, the in-sample log-likelihood gained a snap): each entity's
    theta, fitted in turn by Newton steps on the penalized log-likelihood of its real snaps' own bins
    (obs) under their bases H reweighted by exp(theta . phi_b), theta the summed tilts of the snap's
    entities, each snap counted at its weight (recency_weights); the penalty prior x the league's
    variance of phi, so an entity with prior snaps' worth of evidence moves half way. The unknown entity
    (the last) stays at none."""
    S, B = H.shape
    D = phi_b.shape[1]
    w8 = np.ones(S) if weights is None else np.asarray(weights, float)
    logH = np.log(np.maximum(H, 1e-300))
    phi_o = phi_b[obs]
    lam = prior * np.maximum(phi_o.var(0), 1e-6)
    theta = {k: np.zeros((sizes[k], D)) for k in who}
    outer = phi_b[:, :, None] * phi_b[:, None, :]
    def moments(th):
        z = logH + th @ phi_b.T
        zm = z.max(1, keepdims=True)
        p = np.exp(z - zm)
        Z = p.sum(1, keepdims=True)
        p /= Z
        return p, (zm + np.log(Z))[:, 0]
    for _ in range(rounds):
        for name, ent in who.items():
            th = sum(theta[k][who[k]] for k in who)
            p, _ = moments(th)
            mu = p @ phi_b
            cov = np.einsum("sb,bij->sij", p, outer) - mu[:, :, None] * mu[:, None, :]
            n_e = sizes[name]
            g = np.zeros((n_e, D))
            np.add.at(g, ent, w8[:, None] * (phi_o - mu))
            g -= lam * theta[name]
            Hs = np.zeros((n_e, D, D))
            np.add.at(Hs, ent, w8[:, None, None] * cov)
            Hs += np.diag(lam)[None]
            theta[name] = theta[name] + np.linalg.solve(Hs, g[:, :, None])[:, :, 0]
            theta[name][-1] = 0.0
    th = sum(theta[k][who[k]] for k in who)
    _, logZ = moments(th)
    gain = float(np.average((th * phi_o).sum(1) - logZ, weights=w8))
    return theta, gain


def fit_model_tilts(m, recs, verbose=True):
    """The tilt mode's thetas for the distance and the time draws (m.etilt) off the real snaps, against
    the model's own league draws as the sim takes them."""
    feats = _features(recs, m)
    who, sizes = _tilt_who(m, recs)
    blend = getattr(m, "blend", 0.0)
    g = lambda k: np.array([r[k] for r in recs])
    kind, gain, field = g("kind"), g("gain"), g("field")
    td = (kind == GAIN) & (gain >= 100 - field)
    first = np.array([bool(r["converted"]) and not r["replay"] for r in recs])
    pd_ = m.dist.payload
    pbin = dist_bin(pd_["kind"], pd_["gain"], pd_["td"], pd_["made"] == 1)
    share = getattr(m, "field", 0.0)
    near = None
    if share:
        r = np.maximum(FIELD_YARDS, np.round(share * (100 - field))).astype(np.int64)
        near = (field.astype(np.int64), r, "field")
    fd = {k: v for k, v in feats.items()}
    H = tilt_bases(m.dist, fd, pbin, DIST_BINS, blend, near, near_blended(m))
    obs = dist_bin(kind, gain, td, first)
    wts = recency_weights(recs)
    phi_all = pd_["phi"].astype(float)
    phi_b = np.zeros((DIST_BINS, len(PHI_DIST)))
    cnt = np.bincount(pbin, minlength=DIST_BINS)
    for d in range(len(PHI_DIST)):
        phi_b[:, d] = np.bincount(pbin, phi_all[:, d], minlength=DIST_BINS) / np.maximum(cnt, 1)
    m.etilt = {}
    prior = getattr(m, "tilt_prior", TILT_PRIOR)
    m.etilt["dist"], gain_d = fit_exp_tilts(H, obs, phi_b, who, sizes, weights=wts, prior=prior)
    # the time: the uncut real snaps, against their whole groups (the clock windows' late states aside)
    seconds = np.array([min(MAX_SECONDS, r["seconds"]) for r in recs], float)
    timed = ~np.array([r["cut"] for r in recs], bool)
    pt = m.time.payload
    tbin = np.where(pt["cut"], -1, time_bin(pt["seconds"]))
    Ht = tilt_bases(m.time, {k: v[timed] for k, v in feats.items()}, tbin, TIME_BINS, blend)
    keep = tbin >= 0
    tb_cnt = np.bincount(tbin[keep], minlength=TIME_BINS)
    phi_t = (np.bincount(tbin[keep], pt["seconds"][keep].astype(float), minlength=TIME_BINS)
             / np.maximum(tb_cnt, 1) / 10.0)[:, None]
    m.etilt["time"], gain_t = fit_exp_tilts(Ht, time_bin(seconds[timed]), phi_t,
                                            {k: v[timed] for k, v in who.items()}, sizes, weights=wts[timed],
                                            prior=prior)
    m._etilt_gain = (gain_d, gain_t)
    return m.etilt


def nudge_direction(m):
    """The strength direction in the distance tilts' space: the gamers' offense tilts regressed on the
    points their side really scored a match (against the league's mean, matches weighted), so a unit is
    the tilt of a gamer who scores a point a match more; where too few gamers have points, the tilts'
    first principal axis turned toward more yards, touchdowns and first downs, fewer turnovers and
    losses (PHI_DIST's order), one gamer sd a unit; a fixed lean that way where none are fitted."""
    sign = np.array([1.0, 1.0, -1.0, 1.0, -1.0])
    th = getattr(m, "etilt", {}).get("dist", {}).get("gamer_off")
    if th is None or len(th) < 4 or not np.any(th[:-1]):
        return 0.05 * sign
    clip = lambda d: np.where(d * sign > 0, d, 0.0)        # no component the wrong way (more turnovers)
    pts = getattr(m, "_gamer_points", {}) or {}
    ids = np.array([g for g in pts if g < len(th) - 1 and len(pts[g])], np.int64)
    if len(ids) >= 8:
        n = np.array([len(pts[g]) for g in ids], float)
        y = np.array([np.mean(pts[g]) for g in ids], float)
        dy = y - np.average(y, weights=n)
        x = th[ids] - np.average(th[ids], axis=0, weights=n)
        if np.sum(n * dy * dy) > 0:
            d = clip((n * dy) @ x / np.sum(n * dy * dy))
            if np.any(d):
                return d
    x = th[:-1] - th[:-1].mean(0)
    _, sv, vt = np.linalg.svd(x, full_matrices=False)
    d = vt[0] * (sv[0] / np.sqrt(len(x)))
    d = clip(d if d @ sign >= 0 else -d)
    return d if np.any(d) else 0.05 * sign


def calibrate_nudge(m, n_paths=NUDGE_PATHS, seed=0):
    """Points a unit of nudge adds to a side's kick-off mean (both sides nudged alike, unknown gamers)."""
    start = kickoff_start(m)
    got = []
    for a in (-1.0, 1.0):
        st = dict(start, nudge=np.full((2, 2), a))
        h, w = simulate(m, st, n_paths // 2, seed=seed + 7)
        got.append(0.5 * (h.mean() + w.mean()))
    return max(0.05, (got[1] - got[0]) / 2.0)


def nudge_solve(model, start, target, n_paths, seed=0, iters=NUDGE_ITERS, stats=None, share=1.0):
    """((home, away) final scores, alpha): the sim from `start` with each side's offense nudged along
    the strength direction by alpha (2,) so its kick-off means meet `target` (home, away) -- or, share
    below 1, that share of the way there from the sim's own means: secant steps from the build's slope,
    each on the same random numbers; the last run's scores (and its stats)."""
    alpha, slope = np.zeros(2), getattr(model, "nudge_slope", 1.0) or 1.0
    prev = None
    target = np.asarray(target, float)
    for it in range(iters + 1):
        st = dict(start, nudge=np.tile(alpha, (len(start["period"]), 1)))
        last = it == iters
        h, a = simulate(model, st, n_paths, seed=seed, stats=stats if last else None)
        if last:
            return (h, a), alpha
        x = np.array([h.mean(), a.mean()])
        if it == 0 and share != 1.0:
            target = x + share * (target - x)
        step = np.full(2, slope)
        if prev is not None:
            da, dx = alpha - prev[0], x - prev[1]
            ok = np.abs(da) > 1e-6
            step = np.where(ok, np.clip(dx / np.where(ok, da, 1.0), 0.3 * slope, 3.0 * slope), slope)
        prev = (alpha.copy(), x)
        alpha = np.clip(alpha + (target - x) / step, -NUDGE_MAX, NUDGE_MAX)
    return (h, a), alpha


def etilt_lines(m):
    """The tilt mode's lines: each effect's spread on each feature, and the log-likelihood gained."""
    t = getattr(m, "etilt", None)
    if not t:
        return []
    names = {i: g for g, i in m.gamers.items()}
    gd, gt = getattr(m, "_etilt_gain", (float("nan"), float("nan")))
    out = [f"  tilts (exp(theta . phi) on the league's draws, shrunk by {getattr(m, 'tilt_prior', TILT_PRIOR):g} snaps):"
           f" in-sample log-likelihood"
           f" gained a snap, distance {gd:+.4f}, time {gt:+.4f}"]
    for kind_, feats_ in (("dist", PHI_DIST), ("time", PHI_TIME)):
        for eff, th in t[kind_].items():
            if len(th) < 2:
                continue
            core = th[:-1]
            out.append(f"    {kind_} {eff.replace('_', ' ')}: " + ", ".join(
                f"{f} sd {core[:, d].std():.3f} ({core[:, d].min():+.2f} to {core[:, d].max():+.2f})"
                for d, f in enumerate(feats_)))
    d = getattr(m, "nudge_dir", None)
    if d is not None:
        out.append("    the strength direction (a unit: a point a side in the sim): " + ", ".join(
            f"{f} {v:+.3f}" for f, v in zip(PHI_DIST, d))
                   + f"; a unit of nudge {getattr(m, 'nudge_slope', float('nan')):+.2f} points a side")
    th = t["dist"]["gamer_off"][:-1]
    if len(th):
        top = np.argsort(-th[:, 0])[:3]
        low = np.argsort(th[:, 0])[:3]
        out.append("    gamers furthest each way on the offense's yards: "
                   + ", ".join(f"{names.get(i, i)} {th[i, 0]:+.2f}" for i in top) + "; "
                   + ", ".join(f"{names.get(i, i)} {th[i, 0]:+.2f}" for i in low))
    return out


def report_build(m, recs=None):
    """The build's lines: the draws' groups and how deep the real snaps' own states fall back."""
    real = getattr(m, "_real_bins", None)
    feats = _features(recs, m) if recs else None
    end = feats["endgame"] > 0 if feats is not None else None
    for i, (name, b) in enumerate((("time", m.time), ("distance", m.dist))):
        if real is None or end is None or len(real[i]) != len(end):
            share = np.bincount(b.level_of, minlength=len(b.levels)) / max(1, len(b.level_of))
            print(f"  {name} draw: {len(b.count):,} groups; by the features they draw on: "
                  + ", ".join(f"{len(b.levels[k])} {100 * share[k]:.0f}%" for k in range(len(share)) if share[k] >= 0.005))
            continue
        if getattr(m, "blend", 0):                     # blended: each level's expected share of the draws
            C = b.chain(feats)
            W = dist_weights(m, C, np.array([r["field"] for r in recs], np.int64)) if b is m.dist \
                else b.blend_weights(C, m.blend)
            rest, late = W[~end].mean(0), W[end].mean(0)
        else:
            rest, late = b.level_shares(real[i][~end]), b.level_shares(real[i][end])
        print(f"  {name} draw: {len(b.count):,} groups; real snaps' states by the features they draw on"
              f" (the rest of the game / the end of a half):")
        for k, lv in enumerate(b.levels):
            if rest[k] >= 0.005 or late[k] >= 0.005:
                print(f"    {'+'.join(lv) or 'all':52s}{100 * rest[k]:4.0f}% /{100 * late[k]:4.0f}%")
    if m.mode == "full":
        for kind_ in ("gamer", "team"):
            for role in ("off", "def"):
                t, d = m.own.get((kind_, role, "time")), m.own.get((kind_, role, "dist"))
                if t is None:
                    continue
                line = (f"  {kind_}s' own ({'offense' if role == 'off' else 'allowed'}): {len(t.count):,} time groups,"
                        f" {len(d.count):,} distance groups")
                if feats is not None:
                    k = 0 if role == "off" else 1
                    ids = np.array([m.gamer_index(r["gamer"][k]) if kind_ == "gamer" else m.team_index(r["team"][k])
                                    for r in recs])
                    f = dict(feats, who=ids)
                    ht, hd = t.bin_of(f) >= 0, d.bin_of(f) >= 0
                    end = feats["endgame"] > 0
                    pc = lambda h, w: f"{100 * h[w].mean():.0f}%" if w.any() else "-"
                    line += (f"; real snaps alike enough to draw from their own (time / distance): the rest of the"
                             f" game {pc(ht, ~end)} / {pc(hd, ~end)}, the end of a half {pc(ht, end)} / {pc(hd, end)}")
                print(line)
    elif m.mode == "adjusted":
        for kind_ in ("time", "dist"):
            for k, v in m.tilt[kind_].items():
                if len(v) > 1:
                    print(f"  {kind_} tilt by {k.replace('_', ' ')}: sd {np.std(v[:-1]):.3f}, from {v[:-1].min():+.2f}"
                          f" to {v[:-1].max():+.2f}")
    for line in etilt_lines(m):
        print(line)
    print(f"  timeouts: {'fitted from the export' if m.meta.get('timeouts') else 'none (no timeout columns)'}")
    print(f"  the draws' part of the game: " + ("the half and the two-minute warning, the snaps left the rest"
                                               if getattr(m, "split", "time") == "snaps" else "the seven parts")
          + ("; the league's draws on the lean key (the clock or the play's time, period, lead, down, distance,"
             " field, snaps left)" if getattr(m, "key", "full") == "lean" else "")
          + (f"; distance draws from near the state's own field (within {m.field:g} x the yards to go)"
             if getattr(m, "field", 0.0) else "")
          + (f"; time draws in a half's last {CLOCK_LATE:g}s from real snaps with as long left (within"
             f" {m.clock:g} x the seconds left)" if getattr(m, "clock", 0.0) else ""))
    for line in remain_lines(m):
        print(line)
    if recs is not None:
        for line in chains_lines(recs):
            print(line)
        for line in late_lines(m, recs, real, feats) + late_lines(m, recs, real, feats, period=2) \
                + field_lines(m, recs, feats):
            print(line)
    for line in getattr(m, "_trend_lines", []):
        print(line)
    names = {i: g for g, i in m.gamers.items()}
    if m.use_rules:
        print(f"  rules learned off the real decisions (v21_rules: never or all but never taken): "
              + ", ".join(f"{DECISIONS[k]['label'].split(' (')[0]} {len(v)}" for k, v in m.rules.items()))
    else:
        print("  no rules: the real shares alone")
    by = getattr(m, "two_good_by", None)
    if by is not None:
        print("  two-point tries made (shrunk by %g tries toward the game's %.1f%%): " % (TWO_GOOD_PRIOR, 100 * m.two_good)
              + ", ".join(f"{name} {100 * by[k]:.1f}% ({int(m.two_good_n[k])})" for k, name in
                          enumerate(("before Q4's last 3:00", "in it", "overtime"))))
    if getattr(m, "decision_blend", 0.0):
        own = ", ".join(f"{DECISIONS[k]['label']}s n / (n + {getattr(ch, 'blend', 0.0):g})"
                        for k, ch in m.choices.items() if getattr(ch, "blend", 0.0) != m.decision_blend)
        print(f"  the decisions' cells blended: a cell of n real decisions weighs n / (n + {m.decision_blend:g}),"
              f" the rest its parents' (cells of {BLEND_MIN}+ kept)" + (f"; {own}" if own else ""))
    for name, ch in m.choices.items():
        spec = DECISIONS[name]
        opt, need = LEAN_SHOWN[name]
        print(f"  {spec['label']}: {ch.n:,} real, "
              + ", ".join(f"{o} {100 * p:.1f}%" for o, p in zip(spec["options"], ch.league)))
        if ch.ruled:
            print(f"    {ch.ruled:,} real ones under a rule, {ch.broke:,} took a struck-out option anyway"
                  f" (left out of the gamers' leans)")
        n_g = ch.counts.sum(1)
        ok = np.flatnonzero(n_g[:-1] >= need)
        if len(ok) and ch.lean[ok].any() and ch.league[opt] > 0:
            rate = {i: ch.gamer_rate(i, opt) for i in ok}
            order = sorted(ok, key=lambda i: rate[i])
            show = lambda ids: ", ".join(f"{names[i]} {100 * rate[i]:.1f}% ({int(n_g[i])})" for i in ids)
            print(f"    {spec['options'][opt]}: each gamer's rate over every real state (the league's"
                  f" {100 * ch.gamer_rate(len(m.gamers), opt):.1f}%), {len(ok)} gamers with {need}+:"
                  f" least {show(order[:3])}; most {show(order[::-1][:3])}")


# the snaps-left table at a few states: (label, period, clock, running, own timeouts, the other side's)
REMAIN_SHOWN = (("Q2 2:00, clock stopped, 3 timeouts each", 2, 120.0, 0, 3, 3),
                ("Q2 0:40, clock stopped, 3 timeouts each", 2, 40.0, 0, 3, 3),
                ("Q2 0:40, clock running, no timeouts", 2, 40.0, 1, 0, 0),
                ("Q4 2:00, clock stopped, 3 timeouts each", 4, 120.0, 0, 3, 3),
                ("Q4 0:40, clock stopped, 3 timeouts each", 4, 40.0, 0, 3, 3),
                ("Q4 0:40, clock running, none for the ball, 3 the other side", 4, 40.0, 1, 0, 3),
                ("Q4 0:40, clock running, no timeouts", 4, 40.0, 1, 0, 0))


def remain_lines(m):
    """The snaps-left table at REMAIN_SHOWN's states, behind / level / ahead."""
    out = ["  snaps left in the half at a few states (behind / level / ahead; buckets "
           + ", ".join(REMAIN_NAMES) + "):"]
    for label, p, c, run, own_, opp_ in REMAIN_SHOWN:
        v = [float(m.remain_at(p, c, run, own_, opp_, l3)) for l3 in range(3)]
        out.append(f"    {label}: " + " / ".join(f"{x:.1f}" for x in v))
    return out


def group_rates(b, flag):
    """Each group's share of its real records with `flag` (a payload-ordered bool array)."""
    if not len(b.start):
        return np.zeros(0)
    return np.add.reduceat(np.asarray(flag, float), b.start) / np.maximum(1, b.count)


def td_chance(m, groups, need, cls, share=None, nmin=FIELD_MIN):
    """The chance the sim's distance draw ends in a touchdown from each state: its group's real snaps,
    each a touchdown where its gain reaches the state's goal, or where it was one itself and runs on
    there (gain_surv_end of the state's endgame3 class); with field windows (share: Bins.near's; None
    the model's own, 0 the whole group), the share of the snaps near the state's field that were one."""
    b, pay = m.dist, m.dist.payload
    share = (getattr(m, "field", 0.0) or 0.0) if share is None else share
    S = getattr(m, "gain_surv_end", None)
    S = S if S is not None else np.tile(m.gain_surv, (SIZES["endgame3"], 1))
    if not len(groups):
        return np.zeros(0)
    key = np.stack([groups, need, cls], 1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    vals = np.full(len(uniq), np.nan)
    ok = np.flatnonzero(uniq[:, 0] >= 0)
    if not share:                     # the whole group: every need at once, per group and class
        pairs, pinv = np.unique(uniq[ok][:, [0, 2]], axis=0, return_inverse=True)
        pinv = pinv.ravel()
        by = np.argsort(pinv, kind="stable")
        cut = np.searchsorted(pinv[by], np.arange(len(pairs) + 1))
        for pi, (g_, c) in enumerate(pairs):
            sl = slice(b.start[g_], b.start[g_] + b.count[g_])
            gain, td = pay["gain"][sl].astype(np.int64), pay["td"][sl]
            pos = (pay["kind"][sl] == GAIN) & (gain > 0)
            gp = np.minimum(gain[pos], 100)
            reach = np.bincount(gp, minlength=101)[::-1].cumsum()[::-1]             # gains >= each need
            ran = np.bincount(gp[td[pos]], 1.0 / np.maximum(S[c, gp[td[pos]]], 1e-12), minlength=101)
            ran = np.r_[0.0, np.cumsum(ran)[:-1]]                                   # touchdowns short of it
            curve = (reach + S[c] * ran) / max(1, len(gain))
            w = ok[by[cut[pi]:cut[pi + 1]]]
            vals[w] = curve[np.clip(uniq[w, 1], 0, 100)]
        return vals[inv.ravel()]
    if len(ok):                       # near the field: the drawn snap's own touchdown (draw_play)
        order = b.by_field()[0]
        lo, hi = b.near(uniq[ok, 0], 100 - uniq[ok, 1], share, nmin=nmin)
        hit = np.r_[0, np.cumsum(pay["td"][order].astype(np.int64))]
        vals[ok] = (hit[hi] - hit[lo]) / np.maximum(1, hi - lo)
    return vals[inv.ravel()]


def _mix(m, b, feats, sub, rates=None, per_group=None, field=None):
    """Each state's (sub: rows of feats) expected value over the groups the league's draw takes it from:
    blended, every level's weighted as bin_blend draws them (blend_weights; the distance draw's at the
    states' own field: dist_weights); else its first level's. rates: per group; per_group(groups, rows,
    last): the value for some of the states (rows of sub), last whether at the chain's last level."""
    f = {k: v[sub] for k, v in feats.items()}
    blend = getattr(m, "blend", 0)
    if not blend:
        g = b.bin_of(f)
        V = np.full(len(g), np.nan)
        okg = g >= 0
        if okg.any():
            V[okg] = rates[g[okg]] if per_group is None else per_group(g[okg], okg, True)
        return V
    C = b.chain(f)
    W = dist_weights(m, C, field) if field is not None and b is m.dist else b.blend_weights(C, blend)
    V = np.zeros(W.shape)
    for li in range(C.shape[1]):
        okl = C[:, li] >= 0
        if okl.any():
            V[okl, li] = rates[C[okl, li]] if per_group is None else \
                per_group(C[okl, li], okl, li == C.shape[1] - 1)
    return (W * V).sum(1)


FIELD_SHOWN = 0.2             # the field windows shown in the build's lines where the model has none


def _td_states(m, recs, feats, sub):
    """Each real snap's (sub) touchdown, its groups' real snaps' share of them and the sim's chance of one
    at its own field drawing from the whole group and from near it (the model's field windows, else
    FIELD_SHOWN's): four arrays."""
    g = lambda k: np.array([recs[i][k] for i in sub])
    kind, gain, field = g("kind"), g("gain"), g("field")
    td = (kind == GAIN) & (gain >= 100 - field)
    need, cls = (100 - field).astype(np.int64), feats["endgame3"][sub]
    share = getattr(m, "field", 0.0) or FIELD_SHOWN
    d_td = group_rates(m.dist, m.dist.payload["td"])
    fld = field.astype(np.int64)
    grp = _mix(m, m.dist, feats, sub, d_td, field=fld)
    whole = _mix(m, m.dist, feats, sub, field=fld,
                 per_group=lambda gs, okl, last: td_chance(m, gs, need[okl], cls[okl], 0.0))
    near = _mix(m, m.dist, feats, sub, field=fld, per_group=lambda gs, okl, last: td_chance(
        m, gs, need[okl], cls[okl], share, int(field_nmin(m, last))))
    return td, grp, whole, near


def _pct(parts, w=None):
    """The four of _td_states as % (w: some of the states)."""
    return tuple(100 * float(np.nanmean(v if w is None else v[w])) for v in parts)


def late_lines(m, recs, real=None, feats=None, period=4):
    """Q4's last 2:00, by the lead: the real snaps' touchdowns, losses and seconds against what the
    league's draws give the same states -- their groups' real snaps, and the sim's touchdown chance at
    the states' own field (td_chance), drawing from the whole group and from near the state's field
    (Bins.near): a group that mixes the situation, or gains that reach further from where the state
    is, part from the real snaps it stands for."""
    real = real if real is not None else getattr(m, "_real_bins", None)
    if real is None or len(real[1]) != len(recs):
        return []
    g = lambda k: np.array([r[k] for r in recs])
    on = (g("period") == period) & (g("clock") <= LATE)
    if not on.any():
        return []
    lead = np.sign(g("margin")).astype(np.int64) + 1
    kind, gain = g("kind"), g("gain")
    loss = (kind == GAIN) & (gain < 0)
    d_loss = group_rates(m.dist, (m.dist.payload["kind"] == GAIN) & (m.dist.payload["gain"] < 0))
    t_mean = np.add.reduceat(m.time.payload["seconds"].astype(float), m.time.start) / np.maximum(1, m.time.count) \
        if len(m.time.start) else np.zeros(0)
    feats = feats if feats is not None else _features(recs, m)
    drawn = "near it" if getattr(m, "field", 0.0) else "the whole group"
    share = getattr(m, "field", 0.0) or FIELD_SHOWN
    out = [f"  Q{period}'s last 2:00 by the lead: the real snaps against their league groups (touchdown % real / group /"
           f" the sim's chance at the state's own field from the whole group / from near it, field {share:g}, its"
           " own touchdown;"
           f" the sim draws from {drawn}; loss %, seconds: real / group)"]
    for k, name in enumerate(("behind", "level", "ahead")):
        sub = np.flatnonzero(on & (lead == k))
        if not len(sub) or (real[1][sub] < 0).all():
            continue
        r_td, g_td, whole, near = _pct(_td_states(m, recs, feats, sub))
        g_loss = _mix(m, m.dist, feats, sub, d_loss, field=g("field")[sub].astype(np.int64))
        g_sec = _mix(m, m.time, feats, sub, t_mean)
        out.append(f"    {name}: {len(sub):,} snaps, touchdown {r_td:.1f} / {g_td:.1f} / {whole:.1f} / {near:.1f},"
                   f" loss {100 * loss[sub].mean():.1f} / {100 * np.nanmean(g_loss):.1f}, seconds"
                   f" {g('seconds')[sub].mean():.1f} / {np.nanmean(g_sec):.1f}")
        if name == "level":
            out += level_lines(m, recs, feats, sub)
    return out


def level_lines(m, recs, feats, sub):
    """Where the league's draws take some real states (sub) from: each level's share of the draws (as
    blended) and its groups' touchdown chance near the states' own field, against the real snaps'."""
    f = {k: v[sub] for k, v in feats.items()}
    b = m.dist
    blend = getattr(m, "blend", 0.0)
    C = b.chain(f)
    field = np.array([recs[i]["field"] for i in sub], np.int64)
    W = dist_weights(m, C, field) if blend else None
    if W is None:
        g = b.bin_of(f)
        W = np.zeros(C.shape)
        for li in range(C.shape[1]):
            W[(g >= 0) & (C[:, li] == g), li] = 1.0
    td = np.array([recs[i]["kind"] == GAIN and recs[i]["gain"] >= 100 - recs[i]["field"] for i in sub])
    share = getattr(m, "field", 0.0) or FIELD_SHOWN
    cls = f["endgame3"]
    parts = []
    for li, lv in enumerate(b.levels):
        w = W[:, li]
        if w.sum() < 0.005 * len(sub):
            continue
        ok = C[:, li] >= 0
        ch = np.zeros(len(sub))
        ch[ok] = td_chance(m, C[ok, li], (100 - field[ok]), cls[ok], share,
                           int(field_nmin(m, li == len(b.levels) - 1)))
        parts.append(f"{'+'.join(lv) or 'all'} {100 * w.sum() / len(sub):.0f}% at {100 * np.average(ch, weights=w):.1f}")
    return [f"      by the draw's level (share of the draws at touchdown % near the field; real {100 * td.mean():.1f}): "
            + "; ".join(parts)]


ZONE_NAMES = ("own 1-10", "own 11-30", "own 31-50", "opp 49-30", "opp 29-20", "opp 19-11", "opp 10-5", "opp 4-1")
FIELD_SAMPLE = 8000           # real snaps the build's field lines read (a seeded sample)


def field_lines(m, recs, feats=None, n_max=FIELD_SAMPLE, seed=0):
    """Every part of the game, by the field: the real snaps' touchdowns against what the league's draws
    give the same states (late_lines' figures, on a sample of n_max real snaps)."""
    if not recs:
        return []
    feats = feats if feats is not None else _features(recs, m)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(len(recs), n_max, replace=False)) if len(recs) > n_max else np.arange(len(recs))
    zone = zone_b(np.array([recs[i]["field"] for i in sub]))
    drawn = "near it" if getattr(m, "field", 0.0) else "the whole group"
    share = getattr(m, "field", 0.0) or FIELD_SHOWN
    out = [f"  every snap by the field ({len(sub):,} real ones): touchdown % real / group / the sim's chance at the"
           f" state's own field from the whole group / from near it, field {share:g}, its own touchdown (the sim"
           f" draws from {drawn})"]
    got = _td_states(m, recs, feats, sub)
    for name, w in [("all", np.ones(len(sub), bool))] + [(nm, zone == z) for z, nm in enumerate(ZONE_NAMES)]:
        if w.any():
            r_td, g_td, whole, near = _pct(got, w)
            out.append(f"    {name:<10} {int(w.sum()):>6,}  {r_td:5.1f} / {g_td:5.1f} / {whole:5.1f} / {near:5.1f}")
    return out


def trend_lines(dec, ot_games):
    """The late choices that settle a game in regulation, by fortnight of the build's data (the
    latest last): a trend says the players are still learning, and the oldest weigh too much.
    dec: build's decision points ({name: [dict with code]}); ot_games: {code: went to overtime}."""
    from .remaining import day
    codes = list(ot_games)
    if not codes:
        return []
    last = max(day(c) for c in codes)
    span = lambda c: min(5, (last - day(c)).days // 14)
    names = [f"{14 * k}-{14 * k + 13} days before the last" if k < 5 else "older" for k in range(6)]
    rows = {k: dict(three=[0, 0], kick=[0, 0], minus1=[0, 0], minus2=[0, 0], ot=[0, 0]) for k in range(6)}
    late = lambda q: q["period"] == 4 and q["clock"] <= LATE
    for q in dec.get("fourth", []):
        if late(q):
            f = int(fg4_of(q["lead"]))
            if f == 1:
                rows[span(q["code"])]["three"][0] += q["choice"] == GO
                rows[span(q["code"])]["three"][1] += 1
            elif f == 2:
                rows[span(q["code"])]["kick"][0] += q["choice"] == FG
                rows[span(q["code"])]["kick"][1] += 1
    for q in dec.get("two", []):
        if 3 <= q["cell"] <= CONV_CELLS - 2 and q["lead"] in (-1, -2):
            key = "minus1" if q["lead"] == -1 else "minus2"
            rows[span(q["code"])][key][0] += q["choice"] == 1
            rows[span(q["code"])][key][1] += 1
    for c, went in ot_games.items():
        rows[span(c)]["ot"][0] += went
        rows[span(c)]["ot"][1] += 1
    pc = lambda a: f"{100 * a[0] / a[1]:5.1f}% ({a[1]})" if a[1] else "    -"
    out = ["  the late choices that settle a game in regulation, by fortnight of the build's data (go for it at"
           " 3 behind, kick at 1-2 behind or level -- Q4's last 2:00; go for two at -1 and -2 after the six --"
           " Q4's last 3:00; overtime a game):"]
    for k in range(5, -1, -1):
        r = rows[k]
        if r["ot"][1]:
            out.append(f"    {names[k]:28s} go {pc(r['three'])}, kick {pc(r['kick'])}, two at -1 {pc(r['minus1'])},"
                       f" at -2 {pc(r['minus2'])}, overtime {pc(r['ot'])}")
    return out


def chains_lines(recs):
    """How the real snaps' new sets of downs line up with the yards: spots are whole yards, so a gain
    short of the distance can move the chains (inches) and one of the distance can fall short; and the
    downs played again (a penalty, the gain short of the line)."""
    g = [r for r in recs if r["kind"] == GAIN]
    if not g:
        return []
    out = [f"  first downs off the next row's down: {len(g):,} real gains, "
           f"{100 * np.mean([r['replay'] for r in g]):.1f}% the same down played again"]
    for lo, hi, label in ((1, 1, "1 to go"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 99, "7+")):
        s = [r for r in g if lo <= r["distance"] <= hi and not r["replay"] and r["gain"] < 100 - r["field"]]
        if not s:
            continue
        yards = np.array([r["gain"] >= r["distance"] for r in s])
        conv = np.array([r["converted"] for r in s])
        out.append(f"    {label}: {len(s):,} snaps, first down {100 * conv.mean():.1f}% (by the yards"
                   f" {100 * yards.mean():.1f}%); short of the yards yet moved the chains {(conv & ~yards).sum():,},"
                   f" the yards yet not {(~conv & yards).sum():,}")
    return out


# ---------------------------------------------------------------- the starts

def _start(model, n, pair, teams):
    return dict(gamer=np.tile([model.gamer_index(pair[0]), model.gamer_index(pair[1])], (n, 1)),
                team_ix=np.tile([model.team_index(teams[0]), model.team_index(teams[1])], (n, 1)))


def kickoff_start(model, pair=("", ""), teams=("", "")):
    """The two kick-off states (each side receiving), with the gamers and teams."""
    return dict(period=np.array([1, 1]), clock=np.array([QUARTER, QUARTER]), team=np.array([0, 1]),
                phase=np.array([KICK, KICK]), down=np.array([1, 1]), dist=np.array([10, 10]),
                y=np.array([25, 25]), home=np.zeros(2, np.int64), away=np.zeros(2, np.int64),
                prev=np.array([FRESH, FRESH]), elapsed=np.zeros(2, np.int64),
                tos=np.full((2, 2), TIMEOUTS), kick2=np.array([1, 0]), **_start(model, 2, pair, teams))


def match_starts(model, rows, pair=("", ""), teams=("", ""), every=1):
    """[(row index, state)] at every `every`-th PLAY_OVER before a scrimmage snap, with what the action
    log says up to it: the clock's state off the play before (a fresh possession, a new quarter: stopped;
    otherwise the play's class), the snaps so far and the timeouts used. The state is a start dict of
    one."""
    rows = _with_side(rows)
    if not rows:
        return []
    a_home = rows[0]["team_a_side"] != "away"
    recs = match_snaps(rows)
    by_end = {r["message"]: r for r in recs}
    starts = sorted(r["start"] for r in recs)
    opening = rows[0].get("opening_offense")
    first = _side_of(opening, a_home) if opening in ("TEAM_A", "TEAM_B") else 0
    out, seen = [], 0
    for i, r in enumerate(rows):
        if r["play_kind"] not in s14.SNAP_KINDS or not r.get("down") or not r.get("field_position") \
                or not r.get("period") or not r.get("clock_seconds") or not r.get("distance") \
                or r.get("offense") not in ("TEAM_A", "TEAM_B"):
            continue
        period, clock = s14._i(r["period"]), s14._f(r["clock_seconds"])
        if period is None or clock is None or clock <= 0 or period > 4 + MAX_OT:
            continue
        seen += 1
        if (seen - 1) % every:
            continue
        msg = s14._i(r["message"])
        last = by_end.get(msg)
        if s14.is_fresh(r) or last is None or last["offense"] != r["offense"] or last["period"] != period \
                or last["kind"] != GAIN or last["gain"] >= 100 - last["field"]:
            prev = FRESH
        else:
            prev = int(play_class(last["gain"], last["distance"]))
        used = (s14._i(r.get("timeouts_used_a")), s14._i(r.get("timeouts_used_b")))
        have = TIMEOUTS if period <= 4 else OT_TIMEOUTS
        tos_a = [max(0, have - u) if u is not None else have for u in used]
        tos = tos_a if a_home else tos_a[::-1]
        home, away = s14._i(r["score_p1"]) or 0, s14._i(r["score_p2"]) or 0
        out.append((i, dict(period=np.array([period]), clock=np.array([clock]),
                            team=np.array([_side_of(r["offense"], a_home)]), phase=np.array([SNAP]),
                            down=np.array([s14._i(r["down"])]), dist=np.array([max(1, s14._i(r["distance"]) or 10)]),
                            y=np.array([s14._i(r["field_position"])]), home=np.array([home]),
                            away=np.array([away]), prev=np.array([prev]),
                            elapsed=np.array([bisect_left(starts, msg)]), tos=np.array([tos]),
                            kick2=np.array([1 - first]), **_start(model, 1, pair, teams))))
    return out


def concat_starts(starts):
    return {k: np.concatenate([s[k] for s in starts]) for k in starts[0]}


def _etilts(model, kind, gamer, team_ix):
    """(paths, 2, dims) each side's theta with the ball (tilt builds; None otherwise): its gamer's and
    team's offense tilts and the other side's defense ones."""
    t = getattr(model, "etilt", None)
    if model.mode != "tilt" or not t or kind not in t:
        return None
    t = t[kind]
    D = t["gamer_off"].shape[1]
    out = np.zeros(gamer.shape + (D,))
    for s_ in (0, 1):
        out[:, s_] = (t["gamer_off"][gamer[:, s_]] + t["gamer_def"][gamer[:, 1 - s_]]
                      + t["team_off"][team_ix[:, s_]] + t["team_def"][team_ix[:, 1 - s_]])
    return out


def _tilts(model, kind, gamer, team_ix):
    """(paths, 2) each side's tilt with the ball (adjusted builds; None otherwise)."""
    if model.mode != "adjusted" or kind not in model.tilt:
        return None
    t = model.tilt[kind]
    out = np.zeros(gamer.shape)
    for s in (0, 1):
        out[:, s] = (t["gamer_off"][gamer[:, s]] + t["gamer_def"][gamer[:, 1 - s]]
                     + t["team_off"][team_ix[:, s]] + t["team_def"][team_ix[:, 1 - s]])
    return np.exp(out)


# ---------------------------------------------------------------- the simulation

# a snap as the census reads it: its clock and the play before as at its PLAY_OVER, the clock it used to
# the next (seconds) and the time its distance was drawn on (drawn)
SNAP_LOG = ("path", "period", "clock", "prev", "seconds", "drawn", "kind", "gain", "dist", "field", "side", "down",
            "lead", "conv", "rem", "to_own", "to_opp", "src", "tb")     # src: the distance draw's (0 league, 1 team, 2 gamer)


def simulate(model, start, n_paths, seed=0, stats=None, max_steps=MAX_STEPS):
    """Play every start n_paths times to the end: (home, away) final scores shaped (starts, paths).
    `stats` (a dict) gathers the census: every snap (SNAP_LOG), counts and points by quarter."""
    rng = np.random.default_rng(seed)
    S = len(start["period"])
    P = S * n_paths
    rep = lambda a, t: np.repeat(np.asarray(a), n_paths, axis=0).astype(t)
    period, clock, team = rep(start["period"], np.int64), rep(start["clock"], float), rep(start["team"], np.int64)
    phase, down, dist, y = (rep(start[k], np.int64) for k in ("phase", "down", "dist", "y"))
    score = np.stack([rep(start["home"], np.int64), rep(start["away"], np.int64)], axis=1)
    prev, elapsed, tos = rep(start["prev"], np.int64), rep(start["elapsed"], np.int64), rep(start["tos"], np.int64)
    kick2, gamer, team_ix = rep(start["kick2"], np.int64), rep(start["gamer"], np.int64), rep(start["team_ix"], np.int64)
    k_time, k_dist = _tilts(model, "time", gamer, team_ix), _tilts(model, "dist", gamer, team_ix)
    th_time, th_dist = _etilts(model, "time", gamer, team_ix), _etilts(model, "dist", gamer, team_ix)
    if "nudge" in start and th_dist is not None and getattr(model, "nudge_dir", None) is not None:
        th_dist = th_dist + rep(start["nudge"], float)[:, :, None] * np.asarray(model.nudge_dir)[None, None, :]
    ot_cur = np.full(P, -1, np.int64)
    ot_done = np.zeros((P, 2), bool)
    ahead = (period >= 5) & (score[:, 0] != score[:, 1])
    ot_done[ahead, np.where(score[ahead, 0] > score[ahead, 1], 0, 1)] = True
    st = stats
    tally = (lambda k, v: st.__setitem__(k, st.get(k, 0) + int(v))) if st is not None else (lambda k, v: None)
    if st is not None:
        st.setdefault("_snaps", [])
        st.setdefault("_dec", [])
        q_points = np.zeros((P, 5), np.int64)
    q_score = score.copy()
    reg_score = score.copy()                  # the score when regulation ended (a start in overtime: its own)
    # how the start's own drive ended (DRIVE_ENDS; -1 not yet): the audit's late drives
    drive_end = np.full(P, -1, np.int64)
    start_team = team.copy()
    last_clk = clock.copy()                   # the clock at the latest snap's start
    # at the drive's end: the yards to the goal, the down, the side's timeouts, the clock at its last snap
    drive_last = np.full((P, 4), -1.0)

    def mark(ix, code):
        ix = np.asarray(ix, np.int64)
        if len(ix):
            new = ix[drive_end[ix] < 0]
            drive_end[new] = code
            drive_last[new] = np.stack([100 - y[new], down[new], tos[new, team[new]], last_clk[new]], 1)

    from . import v21_rules
    use_rules = RULES and getattr(model, "use_rules", True)
    field_share = getattr(model, "field", 0.0) or 0.0
    clock_share = getattr(model, "clock", 0.0) or 0.0

    def state(ix, side):
        """The full state of a decision by `side` on each path (v21_rules.SWEEP_FEATURES)."""
        return dict(period=np.minimum(period[ix], 5), left=np.round(half_left(period[ix], clock[ix])).astype(np.int64),
                    lead=score[ix, side] - score[ix, 1 - side], togo=100 - y[ix], dist=dist[ix], down=down[ix],
                    own_to=tos[ix, side], opp_to=tos[ix, 1 - side], prev=prev[ix])

    def decide(name, cols, who, fine=None, ot=None, ix=None):
        """A decision for each path: the options the rules strike out in its full state (`fine`) and
        overtime's own (`ot`) taken away, the rest drawn from the real ones in the state, leaned by the
        gamer deciding."""
        learned = v21_rules.mask(model.rules, name, fine) if use_rules and fine is not None else None
        k = model.choices[name].draw(cols, who, rng.random(len(who)), combine(learned, ot))
        if st is not None:
            st["_dec"].append((name, who.copy(), k.copy(), decision_slice(name, cols), None if ix is None else ix.copy()))
            ch = model.choices[name]          # the league's shares in each decision's cell (no lean, no rule)
            st.setdefault("_dec_base", []).append(ch.base(cols))
            if name == "two":                 # the audit's conversions: the clock's cell, the lead after the six
                st.setdefault("_two", []).append((np.asarray(cols["cell"]).copy(), np.asarray(cols["lead"]) - CONV_MARGIN,
                                                  k.copy(), st["_dec_base"][-1][:, 1].copy()))
        return k

    def kickoff(ix):
        recv = team[ix]
        kicker = 1 - recv
        lead = score[ix, kicker] - score[ix, recv]
        desperate = ((period[ix] >= 4) & (lead < 0) & (half_left(period[ix], clock[ix]) <= 180)).astype(np.int64)
        late = (((period[ix] == 2) | (period[ix] == 4)) & (clock[ix] <= LATE)).astype(np.int64)
        kind = decide("kickoff", dict(desperate=desperate, late=late, lead=lead3(lead)), gamer[ix, kicker],
                      fine=state(ix, kicker), ix=ix)
        tally("kickoffs", len(ix))
        for k_, spot in ((TOUCHBACK, KICK_TOUCHBACK), (NO_LANDING, KICK_NO_LANDING)):
            sel = kind == k_
            if sel.any():
                set_ball(ix[sel], recv[sel], np.full(int(sel.sum()), spot))
        for k_ in (RETURN, KEPT):
            s_ = np.flatnonzero(kind == k_)
            if not len(s_):
                continue
            field, secs = np.empty(len(s_), np.int64), np.empty(len(s_))
            for d in (0, 1):
                for l_ in (0, 1):
                    q = (desperate[s_] == d) & (late[s_] == l_)
                    if q.any():
                        pf, ps = model.kick_pool[(k_, d, l_)]
                        j = rng.integers(0, len(pf), int(q.sum()))
                        field[q], secs[q] = pf[j], ps[j]
            sel = ix[s_]
            clock[sel] = np.maximum(0.0, clock[sel] - secs)
            side = kicker[s_] if k_ == KEPT else recv[s_]
            if k_ == KEPT:
                tally("onside_kept", len(s_))
            td = field >= 100
            if td.any():
                tally("kick_tds", td.sum())
                touchdown(sel[td], side[td])
            set_ball(sel[~td], side[~td], field[~td])

    def set_ball(ix, side, field):
        team[ix] = side
        y[ix] = np.clip(field, 1, 99)
        down[ix] = 1
        dist[ix] = np.minimum(10, 100 - y[ix])
        prev[ix] = FRESH
        phase[ix] = SNAP

    def touchdown(ix, side):
        own = np.asarray(side) == start_team[ix]
        mark(ix[own], 0)
        mark(ix[~own], 4)                     # the other side's: a defensive touchdown
        score[ix, side] += 6
        tally("touchdowns", len(ix))
        lead = score[ix, side] - score[ix, 1 - side]
        ot = period[ix] >= 5
        walk = ot & ot_done[ix, 1 - side] & (lead > 0)            # the second side goes ahead: over
        if walk.any():
            phase[ix[walk]] = DONE
            tally("ot_walk_off", walk.sum())
        ix, side, lead, ot = ix[~walk], side[~walk], lead[~walk], ot[~walk]
        if not len(ix):
            return
        p, c = period[ix], clock[ix]
        cell = np.where(p <= 2, 0, np.where(p == 3, 1, np.where(ot | (c <= 180), 3, 2)))
        if CONV_CLOCK:
            cell = np.where(ot, CONV_CELLS - 1, cell)
            for i, edge in reversed(list(enumerate(CONV_CLOCK))):
                cell = np.where((p == 4) & (c <= edge), 4 + i, cell)
        close = ot & ot_done[ix, 1 - side] & ((lead == -1) | (lead == -2))
        two = decide("two", dict(cell=cell, lead=np.clip(lead, -CONV_MARGIN, CONV_MARGIN) + CONV_MARGIN,
                                 late=conv_late(cell)),
                     gamer[ix, side], fine=state(ix, side), ot=ot_mask("two", len(ix), other_done_close=close),
                     ix=ix) == 1
        by = getattr(model, "two_good_by", None)
        two_good = by[conv_late(cell)] if by is not None else model.two_good
        good = rng.random(len(ix)) < np.where(two, two_good, model.kick_good)
        score[ix, side] += np.where(good, np.where(two, 2, 1), 0)
        tally("two_point_tries", two.sum())
        kick_to(ix, 1 - side)

    def kick_to(ix, receiver):
        team[ix] = receiver
        phase[ix] = KICK

    def field_goal(ix):
        """A field goal try: made by its length; the clock it takes as the real kicks' did (a kick with
        FG_NEAR or less left in the half from those, one that ran the clock out running this one out)."""
        tally("field_goal_tries", len(ix))
        kd = 100 - y[ix] + 17
        made = rng.random(len(ix)) < model.fg_make[np.clip((kd - 15) // 5, 0, len(model.fg_make) - 1)]
        near = half_left(period[ix], clock[ix]) <= FG_NEAR
        used = np.empty(len(ix))
        if near.any():
            j = rng.integers(0, len(model.fg_near), int(near.sum()))
            left0, used0 = model.fg_near[j, 0], model.fg_near[j, 1]
            used[near] = np.where(used0 >= left0 - 0.5, np.inf, used0)
        if (~near).any():
            used[~near] = model.fg_far[rng.integers(0, len(model.fg_far), int((~near).sum()))]
        clock[ix] = np.maximum(0.0, clock[ix] - used)
        tally("field_goals", made.sum())
        mk, ms = ix[made], ix[~made]
        mark(mk, 1)
        mark(ms, 2)
        score[mk, team[mk]] += 3
        kick_to(mk, 1 - team[mk])
        set_ball(ms, 1 - team[ms], np.maximum(20, 100 - y[ms]))

    def fourth(ix):
        off = team[ix]
        cols = derive(dict(togo10=togo10(y[ix]), dist=dist_b(dist[ix]), margin=margin_b(score[ix, off] - score[ix, 1 - off]),
                           phase=phase_of(period[ix], clock[ix]), left=left_b(half_left(period[ix], clock[ix])),
                           togo4=togo4(100 - y[ix]), fg4=fg4_of(score[ix, off] - score[ix, 1 - off])))
        cols = {k: v for k, v in cols.items() if k in DECISIONS["fourth"]["features"]}
        lead = score[ix, off] - score[ix, 1 - off]
        must = (period[ix] >= 5) & ot_done[ix, 1 - off] & (lead < 0)
        choice = decide("fourth", cols, gamer[ix, off], fine=state(ix, off),
                        ot=ot_mask("fourth", len(ix), (must, lead <= -4)), ix=ix)
        choice = np.where((choice == FG) & (100 - y[ix] + 17 > FG_MAX), np.where(dist[ix] <= 2, GO, PUNT), choice)
        pu, fg = ix[choice == PUNT], ix[choice == FG]
        if len(pu):
            tally("punts", len(pu))
            mark(pu, 3)
            if model.punts is not None:
                j = model.punts.draw(model.punts.bin_of(dict(ybin=np.clip(y[pu] // 20, 0, 4))), rng.random(len(pu)))
                net, secs = model.punts.payload["net"][j], model.punts.payload["seconds"][j]
            else:
                net, secs = np.full(len(pu), 40), np.full(len(pu), 8.0)
            clock[pu] = np.maximum(0.0, clock[pu] - secs)
            land = y[pu] + net
            set_ball(pu, 1 - team[pu], np.where(land >= 100, 20, 100 - land))
        if len(fg):
            field_goal(fg)
        return ix[choice == GO]

    layers = [None] + sorted(model.own) if model.mode == "full" else [None]

    def own(draw, f, ix, off, u, fields, out, stopped=False, near=None):
        """full: half the draws from the offense's own real snaps, half from what the defense
        allowed, each its gamer's where it has them, else its team's (else the league's, as drawn).
        out["_from"] and out["_j"] keep which draw (0 the league's) and record each came from; near:
        (values, radii, payload name), each record taken from its window (Bins.window)."""
        if model.mode != "full" or not model.own:
            return
        allowed = rng.random(len(ix)) < DEF_SHARE
        for role, rows in (("off", ~allowed), ("def", allowed)):
            sel = np.flatnonzero(rows)
            if not len(sel):
                continue
            side = off[sel] if role == "off" else 1 - off[sel]
            for kind_, who in (("team", team_ix), ("gamer", gamer)):
                layer = model.own.get((kind_, role, draw))
                if layer is None:
                    continue
                lb = layer.bin_of(dict({k: v[sel] for k, v in f.items()}, who=who[ix[sel], side]))
                ok = lb >= 0
                if ok.any():
                    jj = layer.draw_low(lb[ok], u[sel[ok]], "seconds", STOPPED_MAX) if stopped \
                        else layer.draw_window(lb[ok], near[0][sel[ok]], near[1][sel[ok]], u[sel[ok]], near[2]) \
                        if near is not None \
                        else layer.draw(lb[ok], u[sel[ok]])
                    for name in fields:
                        out[name][sel[ok]] = layer.payload[name][jj]
                    out["_from"][sel[ok]] = layers.index((kind_, role, draw))
                    out["_j"][sel[ok]] = jj

    def league_bins(bins, f):
        """The league's group for each draw: blended (Bins.bin_blend) or the first level with enough."""
        blend = getattr(model, "blend", 0.0)
        return bins.bin_blend(f, rng, blend) if blend else bins.bin_of(f)

    def remain_of(ix, prv, tos0):
        """The snaps-left bucket of each path's snap: its clock, running or not (prv), the timeouts
        each side had before any was called for it (tos0: own, other's) and the lead."""
        off = team[ix]
        lead3 = derive(dict(margin=margin_b(score[ix, off] - score[ix, 1 - off])))["lead3"]
        return remain_b(model.remain_at(period[ix], clock[ix], running_of(prv), tos0[:, 0], tos0[:, 1], lead3))

    def draw_time(ix, prv, rem, stopped=False):
        """Each snap's time; `stopped`: from the stopped-clock part of the same draw (the snap after
        the two-minute warning)."""
        off = team[ix]
        f = derive(dict(prev=prv, phase=phase_of(period[ix], clock[ix]),
                        margin=margin_b(score[ix, off] - score[ix, 1 - off]), down=down_b(down[ix]),
                        dist=dist_b(dist[ix]), zone=zone_b(y[ix]), elapsed=elapsed_b(elapsed[ix]), remain=rem))
        u = rng.random(len(ix))
        k = k_time[ix, off] if k_time is not None else None
        b = league_bins(model.time, f)
        near = None
        if clock_share and not stopped:
            # a half's end: from the real snaps with as long left in the half (the rest: their whole group)
            left = half_left(period[ix], clock[ix])
            late = (left <= CLOCK_LATE) & (period[ix] <= 4)
            r_c = np.where(late, np.maximum(CLOCK_SECONDS, np.round(clock_share * left)), 4 * WINDOW_SPAN)
            near = (np.round(left).astype(np.int64), r_c.astype(np.int64), "left")
        th = th_time[ix, off] if th_time is not None else None
        j = model.time.draw_low(b, u, "seconds", STOPPED_MAX) if stopped \
            else model.time.draw_window(b, near[0], near[1], u, "left", k, rng, theta=th) if near is not None \
            else model.time.draw_tilted(b, u, th, rng) if th is not None \
            else model.time.draw(b, u, k)
        out = {"seconds": model.time.payload["seconds"][j].copy(), "cut": model.time.payload["cut"][j].copy(),
               "_from": np.zeros(len(ix), np.int64), "_j": j.copy()}
        own("time", f, ix, off, u, ("seconds", "cut"), out, stopped, near)
        secs, cut = out["seconds"], out["cut"]
        if stopped:
            return secs
        # a real snap the quarter's end or 2:00 cut short, drawn with more time to go than it took:
        # the draw moves on to a real snap at least as long in the same group
        limit = np.where(((period[ix] == 2) | (period[ix] == 4)) & (clock[ix] > LATE),
                         clock[ix] - LATE + PLAY_SECONDS, clock[ix])
        again = np.flatnonzero(cut & (secs < limit))
        if len(again):
            u2 = rng.random(len(again))
            for li in np.unique(out["_from"][again]):
                a = out["_from"][again] == li
                b = model.time if li == 0 else model.own[layers[li]]
                secs[again[a]] = b.payload["seconds"][b.above(out["_j"][again[a]], u2[a])]
        return secs

    def draw_play(ix, secs, clk, rem):
        """What came of each snap, off the state before it (clk: the clock at the play before's end):
        (kind, gain, shift, a new set of downs, the down played again)."""
        off = team[ix]
        f = derive(dict(tb=time_b(secs), down=down_b(down[ix]), dist=dist_b(dist[ix]), zone=zone_b(y[ix]),
                        margin=margin_b(score[ix, off] - score[ix, 1 - off]), phase=phase_of(period[ix], clk),
                        elapsed=elapsed_b(elapsed[ix]), remain=rem))
        u = rng.random(len(ix))
        k = k_dist[ix, off] if k_dist is not None else None
        nmin = None
        if near_blended(model):
            # each cell weighed by its snaps near the field, its window never widened (the last level's aside)
            C = model.dist.chain(f)
            b, li = Bins.pick(C, dist_weights(model, C, y[ix]), rng)
            nmin = field_nmin(model, li == C.shape[1] - 1)
        else:
            b = league_bins(model.dist, f)
        th = th_dist[ix, off] if th_dist is not None else None
        j = model.dist.draw_near(b, y[ix], u, field_share, k, rng, theta=th, nmin=nmin) if field_share \
            else model.dist.draw_tilted(b, u, th, rng) if th is not None else model.dist.draw(b, u, k)
        names = ("kind", "gain", "shift", "td", "made", "rdist") + (("field",) if field_share else ())
        out = {x: model.dist.payload[x][j].copy() for x in names}
        out["_from"], out["_j"] = np.zeros(len(ix), np.int64), j.copy()
        r_f = np.maximum(FIELD_YARDS, np.round(field_share * (100 - y[ix]))).astype(np.int64)
        own("dist", f, ix, off, u, names, out, near=(y[ix], r_f, "field") if field_share else None)
        kind, gain, shift, td, made, rdist = (out[x] for x in names[:6])
        gain = gain.astype(np.int64)
        if SMOOTH_YARDS:
            spread = (kind == GAIN) & (gain != 0) & ~td
            gain = np.where(spread, gain + rng.integers(-SMOOTH_YARDS, SMOOTH_YARDS + 1, len(ix)), gain)
        need = 100 - y[ix]
        if field_share:
            # drawn at another distance, the real snap's own result against the line: a new set of downs
            # past it by as far, or short of it as far of the way (a 4th-and-1 sneak is not replayed at 4th
            # and 3 as a yard short, nor a 3rd-and-3 incompletion at 3rd and 1)
            other = (kind == GAIN) & ~td & (made <= 1) & (rdist != dist[ix]) & (rdist > 0)
            if other.any():
                d_, rd = dist[ix].astype(np.int64), np.maximum(1, rdist.astype(np.int64))
                moved = np.maximum(gain - rd + d_, d_)
                # a gain short of the line stays a gain where there is room for one (rounded, not floored:
                # a 2-yard gain at 3rd and 10 is a yard at 3rd and 4, not nothing)
                scaled = np.maximum(1, np.round(gain * d_ / rd)).astype(np.int64)
                short = np.minimum(np.where(gain > 0, scaled, gain), d_ - 1)
                gain = np.where(other, np.where(made == 1, moved, short), gain)
            # drawn near the state's own field, the real snap's own result: its touchdown is one here, and
            # a gain that fell short of its own goal falls short of this one, as far of the way there
            gain = np.where(td, np.maximum(gain, need), gain)
            far = (kind == GAIN) & ~td & (gain >= need)
            r_need = np.maximum(1, 100 - out["field"].astype(np.int64))
            gain = np.where(far, np.minimum(need - 1, gain * need // r_need), gain)
            on = np.zeros(0, np.int64)
        else:
            far = np.zeros(len(ix), bool)
            # a real touchdown drawn from further back runs on with the real chance a gain that long goes
            # on, in the same part of the game (endgame3: at the end of a half by the snaps left and the lead)
            on = np.flatnonzero(td & (gain > 0) & (gain < need))
        if len(on):
            S_ = model.gain_surv_end[f["endgame3"][on]] if getattr(model, "gain_surv_end", None) is not None \
                else np.tile(model.gain_surv, (len(on), 1))
            g0 = np.clip(gain[on], 0, 100)
            t = rng.random(len(on)) * S_[np.arange(len(on)), g0]
            reach = (S_ >= t[:, None]).sum(1) - 1                         # the longest gain with S >= t
            gain[on] = np.maximum(g0, reach)
        # a new set of downs: the real snap's own, where it was to go as far (spots are whole yards: a
        # sneak for inches shows as 0 and moves the chains); else by the yards
        same = (made <= 1) & (rdist == dist[ix]) & ~far
        conv = np.where(same, made == 1, gain >= dist[ix])
        src = np.array([0] + [1 if lay[0] == "team" else 2 for lay in layers[1:]], np.int64)[out["_from"]]
        return kind, gain, shift, conv, made == 2, src

    def play(ix):
        off = team[ix]
        # timeouts: only before a snap off a running clock, and they move only the time draw
        running = prev[ix] <= LOSS
        cols = dict(phase=phase_of(period[ix], clock[ix]), margin=margin_b(score[ix, off] - score[ix, 1 - off]),
                    left=left_b(half_left(period[ix], clock[ix])))
        call_def, call_off = np.zeros(len(ix), bool), np.zeros(len(ix), bool)
        d_ = np.flatnonzero(running & (tos[ix, 1 - off] > 0))
        if len(d_):
            call_def[d_] = decide("to_def", {k: v[d_] for k, v in cols.items()}, gamer[ix[d_], 1 - off[d_]],
                                  fine=state(ix[d_], 1 - off[d_]), ix=ix[d_]) == 1
        o_ = np.flatnonzero(running & ~call_def & (tos[ix, off] > 0))
        if len(o_):
            c_off = dict({k: v[o_] for k, v in cols.items()}, range2=in_range(100 - y[ix[o_]]))
            call_off[o_] = decide("to_off", c_off, gamer[ix[o_], off[o_]], fine=state(ix[o_], off[o_]), ix=ix[o_]) == 1
        prv = prev[ix].copy()
        tos0 = np.stack([tos[ix, off], tos[ix, 1 - off]], axis=1)      # before any is called for this snap
        tos[ix[call_def], 1 - off[call_def]] -= 1
        tos[ix[call_off], off[call_off]] -= 1
        prv[call_def | call_off] = TIMEOUT
        tally("timeouts", (call_def | call_off).sum())
        rem = remain_of(ix, prv, tos0)
        secs = draw_time(ix, prv, rem)
        if LATE_TIMEOUT:
            # a side that a kick puts level or ahead in Q4 or overtime (in range at Q2's end, any lead), with a
            # timeout left, does not let a running clock run the half out before or during its snap: it calls
            # one, and the snap comes off a stopped clock (its kick, or not, is the next snap's choice)
            lead_o = score[ix, off] - score[ix, 1 - off]
            save = np.flatnonzero((prv <= LOSS) & (secs >= clock[ix]) & (tos[ix, off] > 0)
                                  & (((period[ix] == 2) & (100 - y[ix] + 17 <= FG_MAX))
                                     | ((period[ix] >= 4) & (lead_o >= -3) & (lead_o <= 0))))
            if len(save):
                tos[ix[save], off[save]] -= 1
                prv[save] = TIMEOUT
                rem[save] = remain_of(ix[save], prv[save], tos0[save])
                secs[save] = draw_time(ix[save], prv[save], rem[save], stopped=True)
                tally("timeouts", len(save))
                tally("late_timeouts", len(save))
        ran = secs > STOPPED_MAX
        # a running clock that runs out before the snap: the period is over, no snap
        gone = ran & (secs - PLAY_SECONDS >= clock[ix])
        if gone.any():
            clock[ix[gone]] = 0.0
            tally("clock_ran_out", gone.sum())
        keep = ~gone
        ix, secs, ran, prv, tos0, rem = ix[keep], secs[keep], ran[keep], prv[keep], tos0[keep], rem[keep]
        if not len(ix):
            return
        start_clock, start_prv = clock[ix].copy(), prv.copy()
        last_clk[ix] = start_clock
        # the two-minute warning stops a running clock at 2:00; the snap then comes off a stopped clock
        warn = ran & ((period[ix] == 2) | (period[ix] == 4)) & (clock[ix] > LATE) \
            & (clock[ix] - (secs - PLAY_SECONDS) <= LATE)
        if warn.any():
            w = ix[warn]
            clock[w] = LATE
            # the play from 2:00: the stopped part of the draw at 2:00, the clock stopped
            rem_w = remain_of(w, np.full(len(w), TIMEOUT), tos0[warn])
            secs[warn] = draw_time(w, prv[warn], rem_w, stopped=True)
            tally("two_minute", len(w))
        clock[ix] = np.maximum(0.0, clock[ix] - secs)
        # the rest as a real snap reads: off the state at the play before's end (a snap after the
        # two-minute warning is filed under its clock before it), the time its play took
        kind, gain, shift, conv, replay, src = draw_play(ix, secs, start_clock, rem)
        off = team[ix]
        elapsed[ix] += 1
        if st is not None:                    # logged as a real snap reads: PLAY_OVER to PLAY_OVER
            got = (kind == GAIN) & (conv | (gain >= 100 - y[ix])) & ~replay
            st["_snaps"].append((ix.copy(), period[ix].copy(), start_clock, start_prv, start_clock - clock[ix],
                                 secs.copy(), kind.copy(), np.minimum(gain, 100 - y[ix]), dist[ix].copy(),
                                 y[ix].copy(), off.copy(), down[ix].copy(), score[ix, off] - score[ix, 1 - off],
                                 got, rem.copy(), tos0[:, 0].copy(), tos0[:, 1].copy(), src, time_b(secs)))
        tally("snaps", len(ix))
        to = kind == TURNOVER
        if to.any():
            t = ix[to]
            new = 100 - (y[t] + shift[to])
            turnover(t, 1 - team[t], np.where(new < 1, 20, new))
        dt_ = kind == DEF_TD
        if dt_.any():
            d = ix[dt_]
            tally("defensive_tds", len(d))
            touchdown(d, 1 - team[d])
        g = kind == GAIN
        gx, gg, gconv, grep = ix[g], gain[g], conv[g], replay[g]
        if not len(gx):
            return
        y2 = y[gx] + gg
        td = y2 >= 100
        if td.any():
            touchdown(gx[td], team[gx[td]])
        sf = y2 <= 0
        if sf.any():
            s = gx[sf]
            tally("safeties", len(s))
            score[s, 1 - team[s]] += 2
            mark(s, 4)
            set_ball(s, 1 - team[s], model.safety_fields[rng.integers(0, len(model.safety_fields), len(s))])
        on = ~td & ~sf
        ox, og, oy, oc, orp = gx[on], gg[on], y2[on], gconv[on], grep[on]
        first = oc & ~orp
        prev[ox] = np.where(first, FIRST, play_class(og, dist[ox]))
        y[ox] = oy
        f, nf = ox[first], ox[~first]
        down[f] = 1
        dist[f] = np.minimum(10, 100 - y[f])
        down[nf] += np.where(orp[~first], 0, 1)                    # a down played again keeps its number
        dist[nf] = np.maximum(1, dist[nf] - og[~first])
        failed = nf[down[nf] > 4]
        if len(failed):
            tally("turnovers_on_downs", len(failed))
            mark(failed, 5)
            set_ball(failed, 1 - team[failed], 100 - y[failed])

    def turnover(ix, side, field):
        tally("turnovers", len(ix))
        mark(ix, 4)
        set_ball(ix, side, field)

    def end_period(ix):
        p = period[ix]
        mark(ix[(p == 2) | (p >= 4)], 6)       # the half's (the game's) end with the ball
        if st is not None:
            q_points[ix, np.minimum(p, 5) - 1] += (score[ix] - q_score[ix]).sum(1)
        q_score[ix] = score[ix]
        carry = (p == 1) | (p == 3)
        c = ix[carry]
        period[c] += 1
        clock[c] = QUARTER
        prev[c] = FRESH
        h = ix[p == 2]
        period[h] = 3
        clock[h] = QUARTER
        tos[h] = TIMEOUTS
        kick_to(h, kick2[h])
        reg = ix[p == 4]
        reg_score[reg] = score[reg]
        level = score[reg, 0] == score[reg, 1]
        phase[reg[~level]] = DONE
        go = reg[level]
        tally("overtime", len(go))
        period[go] = 5
        clock[go] = QUARTER
        tos[go] = OT_TIMEOUTS
        ot_done[go], ot_cur[go] = False, -1
        kick_to(go, rng.integers(0, 2, len(go)))
        ot = ix[p >= 5]
        behind = np.where(score[ot, 0] < score[ot, 1], 0, 1)
        decided = (score[ot, 0] != score[ot, 1]) & ot_done[ot, behind]
        stop = decided | (period[ot] >= 4 + MAX_OT)
        phase[ot[stop]] = DONE
        on = ot[~stop]
        period[on] += 1
        clock[on] = QUARTER
        tos[on] = OT_TIMEOUTS
        prev[on] = FRESH

    for _ in range(max_steps):
        live = np.flatnonzero(phase != DONE)
        if not len(live):
            break
        ot = live[period[live] >= 5]
        if len(ot):                                   # overtime: whose possessions are done
            sn_ = ot[phase[ot] == SNAP]
            ended = sn_[(ot_cur[sn_] >= 0) & (ot_cur[sn_] != team[sn_])]
            ot_done[ended, ot_cur[ended]] = True
            ot_cur[sn_] = team[sn_]
            kk = ot[phase[ot] == KICK]
            ended = kk[ot_cur[kk] >= 0]
            ot_done[ended, ot_cur[ended]] = True
            ot_cur[kk] = -1
            over = ot[ot_done[ot, 0] & ot_done[ot, 1] & (score[ot, 0] != score[ot, 1])]
            phase[over] = DONE
            tally("ot_decided", len(over))
            live = np.flatnonzero(phase != DONE)
        k = live[phase[live] == KICK]
        if len(k):
            kickoff(k)
        sn = live[phase[live] == SNAP]
        sn = sn[~np.isin(sn, k)]
        if AUTO_END and len(sn):
            # the game's own end: the side ahead in Q4 can run the clock out, the defense has no timeouts
            off = team[sn]
            over = auto_end(period[sn], clock[sn], score[sn, off] - score[sn, 1 - off], down[sn], tos[sn, 1 - off])
            if over.any():
                clock[sn[over]] = 0.0
                tally("auto_end", over.sum())
                mark(sn[over], 6)
                sn = sn[~over]
        if len(sn):
            # downs 1-3 in range in the half's last minute: kick now or play on
            left = half_left(period[sn], clock[sn])
            kd = 100 - y[sn] + 17
            can = (down[sn] < 4) & (kd <= FG_MAX) & (left <= EARLY_LEFT) & ((period[sn] == 2) | (period[sn] >= 4))
            e = sn[can]
            if len(e):
                off = team[e]
                mb = margin_b(score[e, off] - score[e, 1 - off])
                cols = dict(left=early_b(left[can]), range=range_b(kd[can]), down=down[e] - 1, margin=mb,
                            half=np.where(period[e] == 2, 0, np.where(period[e] == 4, 1, 2)),
                            fg4=fg4_of(score[e, off] - score[e, 1 - off]), clock2=running_of(prev[e]),
                            to3=np.minimum(2, tos[e, off]))
                lead = score[e, off] - score[e, 1 - off]
                must = (period[e] >= 5) & ot_done[e, 1 - off] & (lead < 0)
                now = e[decide("early_fg", cols, gamer[e, off], fine=state(e, off),
                               ot=ot_mask("early_fg", len(e), (must, lead <= -4)), ix=e) == 1]
                if len(now):
                    tally("early_field_goals", len(now))
                    field_goal(now)
                    sn = sn[~np.isin(sn, now)]
            on4 = sn[down[sn] >= 4]
            go = fourth(on4) if len(on4) else on4
            plays = np.concatenate([sn[down[sn] < 4], go])
            if len(plays):
                play(plays)
        ended = np.flatnonzero((phase != DONE) & (clock <= 0.0))
        if len(ended):
            end_period(ended)
    if st is not None:
        st["went_ot"] = (period >= 5)
        st["games"] = st.get("games", 0) + P
        st["final"] = score.copy()
        st["regulation"] = reg_score.copy()
        st["drive_end"] = drive_end.copy()
        st["drive_last"] = drive_last.copy()
        st["q_points"] = q_points
        st["unfinished"] = st.get("unfinished", 0) + int((phase != DONE).sum())
    return score[:, 0].reshape(S, n_paths), score[:, 1].reshape(S, n_paths)


def decision_census(stats, n_who):
    """{name: {"slices": (slices, options), "who": (gamers + 1, options)}} counts of a sim's decisions."""
    out = {}
    for name, who, k, sl, *_ in stats.get("_dec") or []:
        K, S_ = len(DECISIONS[name]["options"]), len(SLICE_NAMES[name])
        c = out.setdefault(name, {"slices": np.zeros((S_, K)), "who": np.zeros((n_who, K))})
        np.add.at(c["slices"], (sl, k), 1)
        np.add.at(c["who"], (who, k), 1)
    return out


def real_decision_census(matches, model, history=None):
    """decision_census's counts for the real decisions of these matches ({code: rows})."""
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    n_who = len(model.gamers) + 1
    out = {name: {"slices": np.zeros((len(SLICE_NAMES[name]), len(spec["options"]))),
                  "who": np.zeros((n_who, len(spec["options"])))} for name, spec in DECISIONS.items()}
    for code, rows in matches.items():
        pair, _ = match_people(code, rows, hist)
        got = match_decisions(rows)
        for name, spec in DECISIONS.items():
            if not got[name]:
                continue
            a = np.array([r[1:] for r in got[name]], np.int64)
            who = np.array([model.gamer_index(pair[r[0]]) for r in got[name]], np.int64)
            sl = decision_slice(name, {f: a[:, 1 + i] for i, f in enumerate(spec["features"])})
            np.add.at(out[name]["slices"], (sl, a[:, 0]), 1)
            np.add.at(out[name]["who"], (who, a[:, 0]), 1)
    return out


def snap_log(stats):
    """The census's snaps as {SNAP_LOG name: array}."""
    log = stats.get("_snaps") or []
    if not log:
        return {k: np.zeros(0) for k in SNAP_LOG}
    return {k: np.concatenate([x[i] for x in log]) for i, k in enumerate(SNAP_LOG)}


def refine_remaining(model, n_paths=REFINE_PATHS, seed=0):
    """Snaps left in the half from the sim's own games: kick-offs at league, each snap's count of the
    snaps after it in its half, by remain_key (the slice, the clock running or not, the timeouts, the
    lead), shrunk as remain_table shrinks; slices the sim never reached read off the table before."""
    stats = {}
    simulate(model, kickoff_start(model), max(1, n_paths // 2), seed=seed, stats=stats)
    log = snap_log(stats)
    if not len(log["path"]):
        return model.remain
    half, sl = remain_slice(log["period"], log["clock"])
    key = log["path"].astype(np.int64) * 3 + half
    order = np.argsort(key, kind="stable")                    # snaps are logged in order: stable keeps it
    k = key[order]
    first = np.r_[True, k[1:] != k[:-1]]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(k)), 0))
    ordinal = np.arange(len(k)) - group_start
    size = np.bincount(k, minlength=int(k.max()) + 1)[k]
    rem = np.empty(len(k), float)
    rem[order] = size - ordinal - 1
    lead3 = derive(dict(margin=margin_b(log["lead"])))["lead3"]
    cells = remain_key(log["period"], log["clock"], running_of(log["prev"]), log["to_own"], log["to_opp"], lead3)
    model.remain = remain_table(cells, rem, base=model.remain)
    return model.remain
