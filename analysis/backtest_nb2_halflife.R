#!/usr/bin/env Rscript
#
# Backtest different recency half-lives for an lme4::glmer.nb (negative
# binomial) mixed-effects version of the NB2 pre-match rating model, to find
# how heavily recent matches should be weighted -- the Python model hardcodes
# a 60-day half-life (weight = 2^(-age_days / half_life)); this fits the SAME
# structure once per candidate half-life on an identical chronological TRAIN
# split, and checks out-of-sample calibration on the held-out TEST split for
# each, so the choice of half-life is picked by backtest rather than guessed.
# HALF_LIVES below is that "weighting as a setting" -- edit the vector to try
# other candidates.
#
# Uses lme4::glmer.nb rather than glmmTMB: lme4 ships precompiled Windows
# binaries on CRAN, so it installs without Rtools/a C++ toolchain. The
# tradeoff is a single global dispersion (theta) fit across all players --
# glmmTMB defaults to the same thing anyway unless given a dispformula, so
# this loses nothing relative to what was actually being compared.
#
# Structure mirrors backtest_nb2_calibration.py's Python NB2 model, but as a
# mixed-effects model: attack/defense/team/stream are all random intercepts
# rather than hand-regularized fixed effects with sum-to-zero constraints --
# the more idiomatic mixed-model way to get the same shrinkage behavior, and
# what makes swapping the weighting scheme this cheap.
#
#   log(mu) = beta0 + (1|Player) + (1|OpponentPlayer) + (1|Team)
#             + (1|OpponentTeam) + (1|Stream)
#
# (1|Player) plays the role of the Python model's player_attack: a player's
# own scoring random intercept. (1|OpponentPlayer) plays the role of
# player_defense: it's the SAME player identities, but in the "who did this
# player concede to" role, fit as a separate random-effect distribution
# because it's a different column -- a good defender's fitted value there
# comes out negative on its own, no explicit subtraction needed.
#
# Score correlation between the two players in a match (the Python model's
# shared-Gamma joint layer) is NOT modelled here -- moneyline probabilities
# are simulated from independent per-side NB2 draws. That's a real
# simplification, acceptable for comparing half-life choices (which mostly
# affects mu1/mu2, not their covariance) but should be revisited before this
# is used for anything beyond that comparison.
#
# Usage:
#   Rscript backtest_nb2_halflife.R
#
# Requires: lme4, dplyr, tidyr
#   install.packages(c("lme4", "dplyr", "tidyr"))

suppressMessages({
  library(lme4)
  library(dplyr)
  library(tidyr)
})

TRAIN_FRACTION <- 0.75
HALF_LIVES <- c(14, 21, 30, 45, 60, 90, 120, 180, 365, Inf)
N_SIMS <- 20000
SEED <- 260909

set.seed(SEED)

script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  getwd()
}

cat("=== Loading and cleaning AMFELO.csv ===\n")
df <- read.csv(file.path(script_dir(), "AMFELO.csv"), stringsAsFactors = FALSE)

required <- c("SCHEDULED_START_TIME_UTC", "STREAM_NUMBER",
              "PLAYER_1_HANDLE", "PLAYER_2_HANDLE",
              "PLAYER_1_TEAM", "PLAYER_2_TEAM",
              "PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE")
missing_cols <- setdiff(required, names(df))
if (length(missing_cols) > 0) stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))

df <- df[complete.cases(df[required]), ]
df$SCHEDULED_START_TIME_UTC <- as.POSIXct(df$SCHEDULED_START_TIME_UTC, tz = "UTC")
df <- df[!is.na(df$SCHEDULED_START_TIME_UTC), ]
df$PLAYER_1_FINAL_SCORE <- as.integer(df$PLAYER_1_FINAL_SCORE)
df$PLAYER_2_FINAL_SCORE <- as.integer(df$PLAYER_2_FINAL_SCORE)
df <- df[df$PLAYER_1_FINAL_SCORE >= 0 & df$PLAYER_2_FINAL_SCORE >= 0, ]
df$PLAYER_1_HANDLE <- toupper(trimws(df$PLAYER_1_HANDLE))
df$PLAYER_2_HANDLE <- toupper(trimws(df$PLAYER_2_HANDLE))
df$PLAYER_1_TEAM <- trimws(df$PLAYER_1_TEAM)
df$PLAYER_2_TEAM <- trimws(df$PLAYER_2_TEAM)
df$STREAM_NUMBER <- as.character(df$STREAM_NUMBER)
df <- df[order(df$SCHEDULED_START_TIME_UTC), ]
df$MatchId <- seq_len(nrow(df))

n <- nrow(df)
split <- floor(n * TRAIN_FRACTION)
train_ids <- df$MatchId[seq_len(split)]
test_ids <- df$MatchId[(split + 1):n]
cat(sprintf("Total matches: %d\n", n))
cat(sprintf("Train: %d (%s -> %s)\n", length(train_ids),
            min(df$SCHEDULED_START_TIME_UTC[df$MatchId %in% train_ids]),
            max(df$SCHEDULED_START_TIME_UTC[df$MatchId %in% train_ids])))
cat(sprintf("Test:  %d (%s -> %s)\n", length(test_ids),
            min(df$SCHEDULED_START_TIME_UTC[df$MatchId %in% test_ids]),
            max(df$SCHEDULED_START_TIME_UTC[df$MatchId %in% test_ids])))

to_long <- function(d) {
  a <- data.frame(
    MatchId = d$MatchId, Score = d$PLAYER_1_FINAL_SCORE,
    Player = d$PLAYER_1_HANDLE, OpponentPlayer = d$PLAYER_2_HANDLE,
    Team = d$PLAYER_1_TEAM, OpponentTeam = d$PLAYER_2_TEAM,
    Stream = d$STREAM_NUMBER, MatchDate = d$SCHEDULED_START_TIME_UTC
  )
  b <- data.frame(
    MatchId = d$MatchId, Score = d$PLAYER_2_FINAL_SCORE,
    Player = d$PLAYER_2_HANDLE, OpponentPlayer = d$PLAYER_1_HANDLE,
    Team = d$PLAYER_2_TEAM, OpponentTeam = d$PLAYER_1_TEAM,
    Stream = d$STREAM_NUMBER, MatchDate = d$SCHEDULED_START_TIME_UTC
  )
  rbind(a, b)
}

train_df <- df[df$MatchId %in% train_ids, ]
test_df <- df[df$MatchId %in% test_ids, ]
train_long <- to_long(train_df)

# Exclude test matches with a player/team unseen in TRAIN -- no rating exists
# for them, same skip-and-report behaviour as the Python backtest.
known_players <- unique(c(train_long$Player, train_long$OpponentPlayer))
known_teams <- unique(c(train_long$Team, train_long$OpponentTeam))
seen <- test_df$PLAYER_1_HANDLE %in% known_players & test_df$PLAYER_2_HANDLE %in% known_players &
        test_df$PLAYER_1_TEAM %in% known_teams & test_df$PLAYER_2_TEAM %in% known_teams
cat(sprintf("Test matches with all players/teams seen in TRAIN: %d / %d\n", sum(seen), nrow(test_df)))
test_df <- test_df[seen, ]

latest_train_time <- max(train_long$MatchDate)
age_days_train <- as.numeric(difftime(latest_train_time, train_long$MatchDate, units = "days"))

evaluate_halflife <- function(half_life) {
  w <- if (is.infinite(half_life)) rep(1, length(age_days_train)) else 2^(-age_days_train / half_life)
  w <- w / mean(w)

  fit <- tryCatch(
    glmer.nb(
      Score ~ (1 | Player) + (1 | OpponentPlayer) + (1 | Team) + (1 | OpponentTeam) + (1 | Stream),
      data = train_long, weights = w
    ),
    error = function(e) { message(sprintf("  half_life=%s: fit failed: %s", half_life, e$message)); NULL }
  )
  if (is.null(fit)) return(NULL)

  theta <- getME(fit, "glmer.nb.theta")  # Var = mu + mu^2/theta  <=>  alpha = 1/theta, same convention as before

  p1_pred <- data.frame(Player = test_df$PLAYER_1_HANDLE, OpponentPlayer = test_df$PLAYER_2_HANDLE,
                         Team = test_df$PLAYER_1_TEAM, OpponentTeam = test_df$PLAYER_2_TEAM,
                         Stream = test_df$STREAM_NUMBER)
  p2_pred <- data.frame(Player = test_df$PLAYER_2_HANDLE, OpponentPlayer = test_df$PLAYER_1_HANDLE,
                         Team = test_df$PLAYER_2_TEAM, OpponentTeam = test_df$PLAYER_1_TEAM,
                         Stream = test_df$STREAM_NUMBER)
  mu1 <- predict(fit, newdata = p1_pred, type = "response", allow.new.levels = TRUE)
  mu2 <- predict(fit, newdata = p2_pred, type = "response", allow.new.levels = TRUE)

  n_test <- nrow(test_df)
  p1_fair_ml <- numeric(n_test)
  for (i in seq_len(n_test)) {
    s1 <- rnbinom(N_SIMS, size = theta, mu = mu1[i])
    s2 <- rnbinom(N_SIMS, size = theta, mu = mu2[i])
    p1_fair_ml[i] <- mean(s1 > s2) + 0.5 * mean(s1 == s2)
  }

  p1_won <- as.numeric(test_df$PLAYER_1_FINAL_SCORE > test_df$PLAYER_2_FINAL_SCORE)
  brier <- mean((p1_fair_ml - p1_won)^2)
  eps <- 1e-9
  p_clip <- pmin(pmax(p1_fair_ml, eps), 1 - eps)
  logloss <- -mean(p1_won * log(p_clip) + (1 - p1_won) * log(1 - p_clip))

  pred_total <- mu1 + mu2
  actual_total <- test_df$PLAYER_1_FINAL_SCORE + test_df$PLAYER_2_FINAL_SCORE
  total_corr <- cor(pred_total, actual_total)
  total_bias <- mean(actual_total - pred_total)

  list(half_life = half_life, theta = theta, brier = brier, logloss = logloss,
       total_corr = total_corr, total_bias = total_bias, n = n_test,
       p1_fair_ml = p1_fair_ml, p1_won = p1_won)
}

cat("\n=== Fitting one glmer.nb model per candidate half-life on TRAIN, evaluating on TEST ===\n")
results <- list()
for (hl in HALF_LIVES) {
  cat(sprintf("\nHalf-life = %s days...\n", ifelse(is.infinite(hl), "Inf (no decay)", hl)))
  res <- evaluate_halflife(hl)
  if (!is.null(res)) {
    results[[length(results) + 1]] <- res
    cat(sprintf("  theta=%.3f  Brier=%.4f  LogLoss=%.4f  TotalCorr=%.3f  TotalBias=%+.2f\n",
                res$theta, res$brier, res$logloss, res$total_corr, res$total_bias))
  }
}

cat("\n=== Summary across half-lives (sorted by Brier score, best first) ===\n")
summary_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(HalfLifeDays = ifelse(is.infinite(r$half_life), NA, r$half_life),
             Theta = r$theta, Brier = r$brier, LogLoss = r$logloss,
             TotalCorr = r$total_corr, TotalBias = r$total_bias, N = r$n)
}))
summary_df <- summary_df[order(summary_df$Brier), ]
print(summary_df, row.names = FALSE)

cat("\n(Lower Brier/LogLoss = better calibrated moneyline. Higher TotalCorr = better totals signal.\n")
cat(" Brier=0.25 is the uninformative always-50% baseline for reference.)\n")
