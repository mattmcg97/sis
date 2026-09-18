"""Tunables for the eAMF calibration suite.

Everything here is meant to be edited between runs; nothing else in the
package hardcodes a table name, a cutoff or a bucket edge.
"""

DATABASE = "SIS_PROD_CG_CURATED"
SCHEMA = "SHARED"

# The two streams the suite can calibrate. Same schema, same feed --
# GAMEPLAI's live model and their candidate, respectively.
STREAMS = {
    "prod": "GAMEPLAI_STREAM",
    "candidate": "GAMEPLAI_STREAM_CANDIDATE",
}

# Candidate model was changed on the morning of 2026-09-17, so quotes from
# before this instant came out of the OLD candidate and would pollute the
# comparison. Both streams are cut to the same instant to keep the two
# calibrations on an identical population.
#
# NOTE ON TIMEZONE: PUBLISH_TIME is TIMESTAMP_NTZ and the rest of this feed
# is stamped UTC (see PRICE_ISSUE_TIME_UTC). This value is therefore read as
# UTC. 2026-09-17 was BST (UTC+1) in the UK, so if "10am" meant UK local
# time, set this to 09:00:00 instead.
CUTOFF_START = "2026-09-17 10:00:00"
CUTOFF_END = None  # None = up to the latest data available

SPORT_CODE = "AF"

# A model quote is only paired with a snapshot if it lands within this many
# seconds of it. The nearest surviving quote wins.
MATCH_TOLERANCE_SECONDS = 3.0

# INPLAY_FIELD_POSITION_PERIOD carries no timestamp -- confirmed by
# preflight, it has 7 columns and none is a clock. What it does carry is
# EVENT_MESSAGE_COUNT, the same feed sequence the GAMEPLAI streams are keyed
# on, so a snapshot's wall-clock time is recovered from the stream itself:
# the PUBLISH_TIME of the quotes published for that same message.
#
# Exact hits need no estimation at all. Where the play feed has a message
# the stream never quoted, the time is interpolated between the bracketing
# quoted messages, and only if that bracket is tight enough to be worth
# trusting (see MAX_BRACKET_MESSAGES).
#
# The clock comes from ONE stream for every run, so prod and candidate are
# calibrated against an identical set of snapshot times and the comparison
# is not confounded by the two feeds stamping the same message a few hundred
# milliseconds apart. Set to a key of STREAMS, or None to use whichever
# stream is being calibrated.
CLOCK_SOURCE = "prod"

# Widest message-count bracket an interpolated snapshot time may sit in.
# Beyond this the stream went quiet either side of the play and the
# interpolation is guesswork, so the snapshot is dropped instead.
MAX_BRACKET_MESSAGES = 10

# "nearest"  -- closest quote either side of the snapshot (what was asked for)
# "forward"  -- only quotes at or after the snapshot; avoids pairing a price
#               that predates the snapshot's own game state, at the cost of
#               dropping snapshots whose next quote is late
QUOTE_DIRECTION = "nearest"

# GAMEPLAI publishes PROBABILITY = 0.00 on markets that are effectively
# dead. analysis/unconditional_calibration.py drops those, and this keeps
# the two consistent; set False to let them through.
EXCLUDE_ZERO_PROBABILITY = True

# Process the match universe in chunks so memory stays flat as the window
# grows. Each chunk pulls its own plays, scores and quotes.
MATCH_CHUNK_SIZE = 50

# Score-difference buckets, as (low, high, label) with both ends inclusive.
# None means unbounded. Read in the PLAYER_1 (home) frame: positive means
# home leads.
SCORE_DIFF_BUCKETS = [
    (None, -9, "<= -9"),
    (-8, -3, "-8..-3"),
    (-2, 2, "-2..+2"),
    (3, 8, "+3..+8"),
    (9, None, ">= +9"),
]

# Which time-axis to split on: "period" (quarter, available now) or
# "drive" (drive number within the match, pending drive reconciliation).
TIME_AXIS = "period"

# Drive-number buckets, used only when TIME_AXIS == "drive".
DRIVE_BUCKETS = [
    (1, 4, "drives 1-4"),
    (5, 8, "drives 5-8"),
    (9, 12, "drives 9-12"),
    (13, None, "drives 13+"),
]

# Directional (paired) comparison: prod and candidate quotes are paired on
# the SAME EVENT_MESSAGE_COUNT, not matched to each stream's own nearest
# quote in time. Pairing on time independently would let one stream land a
# quote 0.1s from the snapshot while the other lands 2.5s away, comparing
# two different game states and flattering whichever got the closer quote.
# Same message means same feed event for both, which is what makes the
# comparison paired at all.
#
# Widest departure from the snapshot's own message allowed when the streams
# never both quoted it.
MAX_PAIR_MESSAGE_GAP = 3

# Disagreement bands for the directional breakdown, as (low, high, label) on
# |prod - candidate| in probability points. Where the two models agree the
# comparison carries almost no information, so it is worth seeing the win
# rate separately at each level of disagreement.
DISAGREEMENT_BANDS = [
    (0.0, 0.01, "< 1pp"),
    (0.01, 0.03, "1-3pp"),
    (0.03, 0.05, "3-5pp"),
    (0.05, 0.10, "5-10pp"),
    (0.10, None, "> 10pp"),
]

# Cells thinner than this are printed but excluded from the "worst cells"
# summary, where noise would otherwise dominate.
MIN_CELL_OBSERVATIONS = 30
MIN_CELL_MATCHES = 10

# Reliability-curve bins for the ECE calculation.
N_RELIABILITY_BINS = 10

# Drive-cleaning thresholds, carried over from
# analysis/classify_possession_outcomes.py.
TOD_GAP_TOLERANCE = 5
PUNT_GAP_THRESHOLD = 25
