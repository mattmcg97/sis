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

# The report can set several candidates against prod at once
# (`--candidate v8,v9`): each is paired with prod in turn and every table
# shows them side by side. Empty: just STREAMS["candidate"].
CANDIDATES = []
# A second build of a model version, by name: --candidate v9-glmer=<dir> -> {"v9-glmer": "<dir>"}
MODEL_DIRS = {}
# A model version prices only the prod rows published before each match's first play, off its
# kick-off (no in-play simulation): set by `bets prematch`, the pre-match models' own test.
PREMATCH_ONLY = False
# Answer repeated identical queries from memory (set by a report that runs
# more than one pairing pass over the same window).
FETCH_CACHE = False

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

# An eAMFModel version standing in for the candidate (--candidate v9) prices
# PLAY_OVER snapshots off SCOUTING_FULL (the game clock lives only there) in
# MODEL_WORKERS processes (None: all cores but one).
SCOUTING_TABLE = "SCOUTING_FULL"
# What the HTML reports call the candidate (None: the model version when one
# stands in, e.g. "v9"; otherwise "candidate"). --candidate-label sets it.
CANDIDATE_LABEL = None
MODEL_WORKERS = None
# --candidate v8 (its model: `python -m eAMFModel v8-build`; None:
# $EAMF_V8_MODEL, then ./v8_model). "own" lines: v8 quotes its own line, in
# the gap between the key numbers, moved as the game moves; "even": the
# half-point line nearest 50%; "prod": its book read at the line prod quoted,
# so every pair answers one question.
V8_MODEL_DIR = None
V8_PATHS = 2000
V8_LINES = "own"
# The same for --candidate v9 (`python -m eAMFModel v9-build`; None:
# $EAMF_V9_MODEL, then ./v9_model).
V9_MODEL_DIR = None
V9_PATHS = 2000
V9_LINES = "own"
# The same for --candidate v10 (`python -m eAMFModel v10-build`; None:
# $EAMF_V10_MODEL, then ./v10_model).
V10_MODEL_DIR = None
V10_PATHS = 2000
V10_LINES = "own"
# The same for --candidate v11 (`python -m eAMFModel v11-build`; None:
# $EAMF_V11_MODEL, then ./v11_model).
V11_MODEL_DIR = None
V11_PATHS = 2000
V11_LINES = "own"
# The same for --candidate v12 (`python -m eAMFModel v12-build`; None:
# $EAMF_V12_MODEL, then ./v12_model).
V12_MODEL_DIR = None
V12_PATHS = 2000
V12_LINES = "own"

# The betting simulation (`python -m eAMFCalibrator bets`): every in-play
# single bet on an AF moneyline, handicap or total in the window, bet by bet,
# from BET_TABLE (a view; DATABASE.SCHEMA.NAME for one elsewhere), read
# through BET_COLUMNS (logical name -> column; None: not read) and cut by
# BET_FILTERS (SQL). BET_EXTRA_COLUMNS are carried through to the output as
# they are. (SHARED.CUSTOMER_REVENUE_EVENT is one row per operator, match and
# day -- no bet times or odds.)
BET_TABLE = "CUSTOMER_REVENUE"
BET_COLUMNS = {
    "id": "OPERATOR_UNIQUE_ID",
    "match": "MATCH_CODE",
    "time": "BET_DATE_UTC",
    "market_type": "MARKET_TYPE_ID",
    "selection": "SELECTION_ID",
    "odds": "ODDS",
    "stake": "STAKE_GBP",
    "revenue": "REVENUE_GBP",
    "line": "MARKET_LINE",
    "period": "BET_PLACED_PERIOD_NUMBER",
    "sport": "SPORT_CODE",
}
BET_FILTERS = ["UPPER(BET_TYPE) = 'SINGLE'"]
BET_EXTRA_COLUMNS = ["OPERATOR_NAME", "CUSTOMER_NAME_HASH", "CUSTOMER_TEMPERATURE", "BET_TYPE",
                     "BET_IN_PLAY", "BET_CASHED_OUT", "CUSTOMER_WIN_LOSS"]
# The lag is fitted per BET_GROUP_COLUMN (the operator) off its in-play bets
# (BET_IN_PLAY_COLUMN = 'Yes'); the report cuts the margin by BET_VIP_COLUMN
# and without BET_VIP_VALUE.
BET_GROUP_COLUMN = "OPERATOR_NAME"
BET_IN_PLAY_COLUMN = "BET_IN_PLAY"
BET_VIP_COLUMN = "CUSTOMER_TEMPERATURE"
BET_VIP_VALUE = "VIP"
# `bets totals-moves` reads BET_VIP_COLUMN = BET_RESTRICTED_VALUE as a restricted account.
BET_RESTRICTED_VALUE = "Restricted"
# The book comparison (book_comparison.py, part of every `bets` run): the customer
# temperatures that bet for an edge and shop on price (everyone else bets as placed, the stake
# x (candidate odds / odds) ^ SIM_ELASTICITY), the most a sharp stake may grow, the bets a
# market's measured edge is shrunk toward its segment's by, and the bootstrap resamples.
SIM_SHARP_GROUPS = ("Restricted",)
SIM_ELASTICITY = 0.0
SIM_MAX_SCALE = 3.0
SIM_EDGE_PRIOR = 200
SIM_BOOT = 1000
# A gamer's session: a run of matches with no gap over this many minutes (the schedule
# runs a match every 35-110 minutes, then breaks for three hours or more).
SESSION_BREAK_MINUTES = 240
# The lags tried, seconds (negative: the operator's clock runs behind
# GAMEPLAI's), in steps of LAG_STEP_SECONDS.
LAG_RANGE = (-30, 60)
LAG_STEP_SECONDS = 1

# Bets marked pre-match after the start, marked in play before it, or placed
# after the match-over message are left out; this many seconds either side
# are allowed.
BET_PHASE_TOLERANCE = 10
# The operators suspend at two minutes left in Q4; bets accepted after that
# (and the tolerance) are left out too.
EXCLUDE_AFTER_TWO_MINUTES = True
# A model or candidate is read against a bet only where its price carries the
# same information as prod's: the feed did not move on between them, and at
# most this many of prod's messages in between are missing from SCOUTING_FULL.
MAX_SCOUTING_GAP = 0
# SCOUTING_FULL messages that do not move the game on: a price made before
# one of these still carries the same information after it.
NEUTRAL_FEED_MESSAGES = ("BET_SUSPEND", "BET_UNSUSPEND")

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
# home leads. The labels name the game state rather than the arithmetic --
# a score in this sport is 6-8 points, so a 3-8 point gap is one score
# behind and 9 or more is two.
SCORE_DIFF_BUCKETS = [
    (None, -9, "Away 2 score"),
    (-8, -3, "Away 1 score"),
    (-2, 2, "Tight"),
    (3, 8, "Home 1 score"),
    (9, None, "Home 2 score"),
]

# A match whose PLAYER_1 / PLAYER_2 handles swap sides part-way through has
# its score difference, possession flag and market outcomes all inverted from
# that point, so its snapshots land in the wrong buckets rather than in no
# bucket. Set True to drop those matches; the default reports them instead,
# so the size of the problem is visible before any data is thrown away.
EXCLUDE_FLIPPED_MATCHES = False

# A quote is live when IS_ACTIVE is 'true'. STATUS is NOT read: GAMEPLAI
# say that column is wrong, so it does not get a vote. It is still carried
# onto every pair and still broken out in the report, which is how the two
# columns can be checked against each other.
#
# The feed publishes four combinations -- open/true, UNDER SETTLEMENT/false,
# CLOSED/false and CLOSED/true -- and none of them is a suspension. Dropping
# STATUS from the test moves CLOSED/true (about 0.1% of rows) from dead to
# live and moves nothing the other way. `preflight` prints the real counts,
# so this stays a checked fact rather than an assumption.
LIVE_IS_ACTIVE = "true"

# Non-live quotes are carried onto the pairs and flagged, but kept out of
# every metric: a price nobody could have taken is not a price. Set False
# to score them anyway, which is almost certainly wrong and is there so the
# cost of excluding them can be measured.
REQUIRE_LIVE_QUOTE = True

# The longest distance-to-go a real snap ever shows. Kick mechanics and
# stale duplicates carry garbage here, which is what lets the snapshot walk
# past them to the drive's actual first play. A first-and-30 after stacked
# penalties is real; the values above this are not.
MAX_PLAUSIBLE_DISTANCE = 40

# Which time-axis to split on: "period" (quarter, available now) or
# "drive" (drive number within the match, pending drive reconciliation).
TIME_AXIS = "period"

# What the calibration pairs prod and the candidate at: "drive", one snapshot
# per drive start off the play table (about 10 a match), or "play_over", every
# SCOUTING_FULL PLAY_OVER (about 60 a match, the snapshots v4-v6 price off),
# each at the first message both streams quoted before the next play starts.
SNAPSHOTS = "drive"

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

# One selection per market, for the cross-sectional calibration view.
#
# Pooling both sides of a market destroys the measurement: the sides are
# complements, so every (p, y) comes with a mirror (1-p, 1-y) and BOTH the
# realized rate and the mean prediction average to exactly 0.5 whatever the
# model does. Taking one side loses nothing -- the other is its complement --
# and lets the calibration gap exist at all.
CANONICAL_SELECTIONS = {50: "Home", 52: "Home", 54: "Over"}

# How to resolve the spread's second selection.
#
# Market 52 reads "PLAYER 1 to score over L more than PLAYER 2" and 53 reads
# "PLAYER 2 to score over L more than PLAYER 1". Two readings are possible
# and they disagree:
#
#   "literal"     two separate propositions. 52 wins on margin_1 > L_52,
#                 53 wins on margin_2 > L_53. If both sides carry the SAME L
#                 they overlap -- at L = -2.5 both win for any margin in
#                 (-2.5, +2.5) -- so they are not two sides of one market.
#   "complement"  one proposition, 53 being the NO of 52: 53 wins on
#                 margin_1 <= L. Partitions by construction.
#
# `cross` prints a spread interpretation report that tests both against the
# data. The probabilities summing to 1 while the two sides carry the same L
# is proof the literal reading is wrong, because complementary probabilities
# require complementary events.
SPREAD_RESOLUTION = "literal"

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
