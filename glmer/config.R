# Settings shared by backtest.R, fit.R and predict.R. Edit here; most of them
# can also be overridden per run from the command line (see README.md).

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Tried in order when --history isn't given (relative to the repo root).
# eAMFCalibrator's `history` command writes the first; nb2/AMFELO.csv is the
# snapshot Adrian's NB2 was built on.
HISTORY_CANDIDATES <- c("eAMFCalibrator/out/match_history.csv", "history.csv", "nb2/AMFELO.csv")

SPORT_CODE <- "AF"

# ---------------------------------------------------------------------------
# Per-player models
# ---------------------------------------------------------------------------

# A player gets their own models only with at least this many matches (after
# the weighting's window / burn-in has dropped rows). Everyone else, and any
# player whose fit fails, is priced by the global model.
MIN_PLAYER_MATCHES <- 60

# Each qualifying player gets two models, fitted on different rows of the
# same long-format data, with the same formula:
#   attack  -- the rows where they are the scorer (points they score)
#   defence -- the rows where they are the opponent (points they concede)
# A side's points in a match can then come from the global model, the
# scorer's attack model, the opponent's defence model, both averaged on the
# log scale ("pair"), or "pair" shrunk toward the global model ("blend":
# BLEND_WEIGHT on the per-player side, the rest on global). PREDICT_MODES
# are the ones backtest.R scores.
PREDICT_MODES <- c("global", "attack", "defence", "pair", "blend")
BLEND_WEIGHT <- 0.5

# Guard rails on a player's own models. At prediction time every numeric
# covariate is held to the range that model saw in training (a slope fitted
# on a feature that barely moved can't extrapolate), and a player's own
# linear predictor may sit at most PLAYER_MAX_SHIFT from the global model's
# on the log scale (0.5 = expected points x/÷ 1.65). Sides that hit the cap
# are counted in the backtest's fits.csv and predict.R's log.
PLAYER_MAX_SHIFT <- 0.5

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
#
# Each set is two right-hand sides in lme4 syntax: `global` for the global
# model (every side of every match) and `player` for each per-player model.
# Every column is known before kick-off (see to_long() and role_view() in
# common.R).
#
# Global model -- each row is one side's points, from the scorer's side:
#   Player, OpponentPlayer        gamer handles (scorer / opponent)
#   Team, OpponentTeam            NFL teams (scorer's / opponent's)
#   TeamPair                      "Team v OpponentTeam"
#   IsHome                        1 for PLAYER_1's side, 0 for PLAYER_2's
#   RestLog, OppRestLog           log1p(hours since the side's previous match), capped at a week
#   Session, OppSession           matches the side played in the previous SESSION_HOURS
#   FormFor, FormAgainst          side's recency-weighted points scored / conceded over
#   OppFormFor, OppFormAgainst      its previous matches, as a log ratio to the league
#   ExpLog, OppExpLog             log1p(previous matches played)
#   SessBand, OppSessBand         position in the side's session: s1, s2, s3_4, s5_6, s7_8, s9p
#   SessFormFor, SessFormAgainst  how the side has scored / conceded so far this session against
#   OppSessFormFor, ...             a pre-match expectation from the form features (log ratio,
#                                   shrunk toward 0 for a short session); 0 on a session's first match
#
# Per-player models -- the same rows seen from the modelled player's side,
# so one formula means the same in their attack model (points they score)
# and their defence model (points they concede):
#   Rival                         who they're playing
#   OwnTeam, RivalTeam            their NFL team / the rival's
#   Matchup                       "OwnTeam v RivalTeam"
#   OwnHome                       1 when they're PLAYER_1
#   OwnFormFor, OwnFormAgainst,   their own and the rival's form, rest, session
#   RivalFormFor, RivalFormAgainst, OwnRestLog, RivalRestLog, OwnSession,
#   RivalSession, OwnExpLog, RivalExpLog, OwnSessBand, RivalSessBand, OwnSessFormFor,
#   OwnSessFormAgainst, RivalSessFormFor, RivalSessFormAgainst
#
# Both: Stream, HourBlock (UTC 4-hour block, "h00".."h20"), Weekday ("d1".."d7").
#
# Extra flags:
#   match_re      global model gets (1|MatchId), one draw shared by both sides
#                 of a match -- the pace/tempo correlation between the scores
#   olre          (1|ObsID): one level per row, NB-like overdispersion on top
#                 of the Poisson (per-player models always get it; this flag
#                 is for the global model)
#   player_offset per-player models fit offset(GlobalEta): the global model's
#                 prediction for the row, so the per-player model only learns
#                 that player's deviations from it (intercept = over/under
#                 performance, REs = matchups the global model misses)
#
# Categorical terms in `player` are safest as random effects: a fixed factor
# needs every level seen in that player's rows, and rows with a new level fall
# back to the global model. Terms whose grouping has fewer than two levels in
# a player's rows are dropped for that player automatically.

GLOBAL_BASE <- "(1|Player) + (1|OpponentPlayer) + (1|Team) + (1|OpponentTeam) + (1|Stream)"
PLAYER_BASE <- "(1|Rival) + (1|OwnTeam) + (1|RivalTeam) + (1|Stream)"

FEATURE_SETS <- list(
  # NB2's own structure: player attack/defence, team attack/defence, stream.
  nb2 = list(
    global = GLOBAL_BASE,
    player = PLAYER_BASE,
    match_re = TRUE, olre = TRUE, player_offset = FALSE),

  # + home/away side.
  home = list(
    global = paste("IsHome +", GLOBAL_BASE),
    player = paste("OwnHome +", PLAYER_BASE),
    match_re = TRUE, olre = TRUE, player_offset = FALSE),

  # + when and how rested: time of day, rest since the last match, how many
  # matches this session.
  context = list(
    global = paste("IsHome + RestLog + OppRestLog + Session + OppSession + (1|HourBlock) +", GLOBAL_BASE),
    player = paste("OwnHome + OwnRestLog + RivalRestLog + OwnSession + RivalSession + (1|HourBlock) +",
                   PLAYER_BASE),
    match_re = TRUE, olre = TRUE, player_offset = FALSE),

  # + recent form of both sides and experience (the early-days learning curve).
  form = list(
    global = paste("IsHome + FormFor + FormAgainst + OppFormFor + OppFormAgainst + ExpLog + OppExpLog +",
                   GLOBAL_BASE),
    player = paste("OwnHome + OwnFormFor + OwnFormAgainst + RivalFormFor + RivalFormAgainst +",
                   PLAYER_BASE),
    match_re = TRUE, olre = TRUE, player_offset = FALSE),

  # form + how both sides are doing this session, and where in it they are.
  form_session = list(
    global = paste("IsHome + FormFor + FormAgainst + OppFormFor + OppFormAgainst + ExpLog + OppExpLog +",
                   "SessFormFor + SessFormAgainst + OppSessFormFor + OppSessFormAgainst +",
                   "SessBand + OppSessBand +", GLOBAL_BASE),
    player = paste("OwnHome + OwnFormFor + OwnFormAgainst + RivalFormFor + RivalFormAgainst +",
                   "OwnSessFormFor + RivalSessFormAgainst +", PLAYER_BASE),
    match_re = TRUE, olre = TRUE, player_offset = FALSE),

  # + player-specific team preference and team-vs-team matchups.
  matchup = list(
    global = paste("IsHome +", GLOBAL_BASE,
                   "+ (1|Player:Team) + (1|OpponentPlayer:OpponentTeam) + (1|TeamPair)"),
    player = paste("OwnHome +", PLAYER_BASE, "+ (1|Matchup)"),
    match_re = TRUE, olre = TRUE, player_offset = FALSE),

  # Per-player models as corrections to the global `home` model.
  home_offset = list(
    global = paste("IsHome +", GLOBAL_BASE),
    player = "(1|Rival) + (1|OwnTeam) + (1|RivalTeam)",
    match_re = TRUE, olre = TRUE, player_offset = TRUE)
)

# ---------------------------------------------------------------------------
# Row weightings
# ---------------------------------------------------------------------------
#
# type:
#   none         every row weight 1
#   exp_days     2^(-age_days / half_life)            (NB2 uses 60)
#   exp_matches  2^(-k / half_life), k = how many of that player's matches
#                are more recent (global model: counted per scorer). Players
#                who play a lot decay faster in calendar time.
#   window       1 within the last `days`, dropped otherwise
#   step         1 within the last `days`, `older` otherwise
# Optional on any type:
#   scalar       multiplies every weight after normalising them to mean 1.
#                Not cosmetic in a mixed model: it moves the variance
#                components, i.e. how hard every rating is shrunk.
#   drop_before  "YYYY-MM-DD": rows before it are dropped (e.g. the launch
#                period, where NBRatingTrial.py notes scoring ran wild)

WEIGHTINGS <- list(
  flat         = list(type = "none"),
  hl30         = list(type = "exp_days", half_life = 30),
  hl60         = list(type = "exp_days", half_life = 60),
  hl120        = list(type = "exp_days", half_life = 120),
  m150         = list(type = "exp_matches", half_life = 150),
  win120       = list(type = "window", days = 120),
  hl60_burnin  = list(type = "exp_days", half_life = 60, drop_before = "2026-01-15"),
  hl60_x0.5    = list(type = "exp_days", half_life = 60, scalar = 0.5),
  hl60_x2      = list(type = "exp_days", half_life = 60, scalar = 2),
  hl60_x4      = list(type = "exp_days", half_life = 60, scalar = 4)
)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FORM_HALF_LIFE_MATCHES <- 10   # FormFor / FormAgainst EWMA half-life, in matches
SESSION_HOURS <- 6             # Session: matches in the previous this-many hours
REST_CAP_HOURS <- 168          # RestLog: hours since the previous match, capped
SESSION_GAP_HOURS <- 2         # a gap longer than this between a side's matches starts a new session
SESSION_SHRINK_MATCHES <- 2    # SessFormFor / SessFormAgainst: mean x k / (k + this), k earlier matches

# ---------------------------------------------------------------------------
# Fitting and pricing
# ---------------------------------------------------------------------------

# glmer's optimiser: "nloptwrap" (lme4's own default, NLopt's BOBYQA with
# looser stopping rules -- much faster on the big global model) or "bobyqa"
# (minqa's, what nb2/glmer_poisson_settings.R uses: slower, a little tighter).
GLMER_OPTIMIZER <- "nloptwrap"
# glmer's nAGQ: 1 = Laplace (default, better); 0 = faster and a little rougher
# -- handy for a first wide sweep of the grid.
GLMER_NAGQ <- 1
GLMER_MAXFUN <- 100000

N_SIMS <- 10000                # Monte Carlo draws per match for moneyline probabilities
SEED <- 260909
