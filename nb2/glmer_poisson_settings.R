#!/usr/bin/env Rscript
#
# Grid-test glmer() settings for the pre-match model: recency half-life
# CROSSED with a separate weight scalar (per the discussion on why the
# scalar isn't cosmetic for a mixed model -- it shifts the estimated
# variance components, i.e. how much shrinkage every player/team rating
# gets, not just standard errors), using plain glmer() rather than
# glmer.nb() so both settings and the random-effects structure are fully
# under our control.
#
# Overdispersion: plain glmer() can't profile over an NB theta the way
# glmer.nb() does, so instead this uses the standard observation-level
# random effect (OLRE) trick -- family=poisson with an extra (1|ObsID)
# term, one level per row. That approximates NB-like overdispersion,
# fits faster than glmer.nb's profiling loop, and -- the actual point --
# is trivial to extend with more random-effect structure.
#
# Also adds (1|MatchId), shared by both players' rows in a match, to
# capture pace/tempo correlation between the two scores -- fixing the
# independence simplification in backtest_nb2_halflife.R and getting
# closer to Adrian's joint Gamma-Poisson layer. Simulation draws the
# match-level effect ONCE per Monte Carlo replicate and applies it to
# both players, so the induced correlation carries through to the
# simulated moneyline/totals probabilities, not just the point mu's.
#
#   log(mu) = beta0 + (1|Player) + (1|OpponentPlayer) + (1|Team)
#             + (1|OpponentTeam) + (1|Stream) + (1|MatchId) + (1|ObsID)
#
# Weight = scalar * (recency decay, normalized to mean 1). Normalizing
# the decay shape first, then applying the scalar as a separate
# multiplier, decouples "how much to prefer recent matches" (half-life)
# from "how much to trust the data overall vs. let it shrink less"
# (scalar) -- the two things you said you want to tune independently.
#
# Grids below are kept modest for runtime (glmer with two per-row-scale
# random effects on ~36K long-format rows is not fast) -- widen
# HALF_LIVES / SCALARS once you've seen how long one pass takes.
#
# Usage:
#   Rscript glmer_poisson_settings.R
#
# Requires: lme4, dplyr, tidyr
#   install.packages(c("lme4", "dplyr", "tidyr"))

suppressMessages({
  library(lme4)
  library(dplyr)
  library(tidyr)
})

TRAIN_FRACTION <- 0.75
HALF_LIVES <- c(30, 60, 90, Inf)
SCALARS <- c(0.5, 1, 2, 5)
N_SIMS <- 10000
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
cat(sprintf("Train: %d  Test: %d\n", length(train_ids), length(test_ids)))

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
  out <- rbind(a, b)
  out$ObsID <- factor(seq_len(nrow(out)))
  out$MatchId <- factor(out$MatchId)
  out
}

train_df <- df[df$MatchId %in% train_ids, ]
test_df <- df[df$MatchId %in% test_ids, ]
train_long <- to_long(train_df)

known_players <- unique(c(train_long$Player, train_long$OpponentPlayer))
known_teams <- unique(c(train_long$Team, train_long$OpponentTeam))
seen <- test_df$PLAYER_1_HANDLE %in% known_players & test_df$PLAYER_2_HANDLE %in% known_players &
        test_df$PLAYER_1_TEAM %in% known_teams & test_df$PLAYER_2_TEAM %in% known_teams
cat(sprintf("Test matches with all players/teams seen in TRAIN: %d / %d\n", sum(seen), nrow(test_df)))
test_df <- test_df[seen, ]

latest_train_time <- max(train_long$MatchDate)
age_days_train <- as.numeric(difftime(latest_train_time, train_long$MatchDate, units = "days"))
KNOWN_RE_FORM <- ~ (1 | Player) + (1 | OpponentPlayer) + (1 | Team) + (1 | OpponentTeam) + (1 | Stream)

evaluate_settings <- function(half_life, scalar) {
  raw <- if (is.infinite(half_life)) rep(1, length(age_days_train)) else 2^(-age_days_train / half_life)
  w <- scalar * (raw / mean(raw))

  # withCallingHandlers logs warnings (e.g. convergence notices, common and
  # usually not fatal for GLMMs with many random-effect groupings) without
  # aborting the fit -- unlike a tryCatch warning= handler, which would
  # discard the whole result the moment lme4 emits its first warning.
  fit <- tryCatch(
    withCallingHandlers(
      glmer(
        Score ~ (1 | Player) + (1 | OpponentPlayer) + (1 | Team) + (1 | OpponentTeam) +
          (1 | Stream) + (1 | MatchId) + (1 | ObsID),
        data = train_long, family = poisson, weights = w,
        control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 100000))
      ),
      warning = function(w) {
        message(sprintf("  half_life=%s scalar=%s: warning: %s", half_life, scalar, conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      message(sprintf("  half_life=%s scalar=%s: fit failed: %s", half_life, scalar, conditionMessage(e)))
      NULL
    }
  )
  if (is.null(fit)) return(NULL)

  vc <- as.data.frame(VarCorr(fit))
  sigma_match <- vc$sdcor[vc$grp == "MatchId"]
  sigma_obs <- vc$sdcor[vc$grp == "ObsID"]
  if (length(sigma_match) == 0 || length(sigma_obs) == 0) {
    message("  could not extract MatchId/ObsID variance components, skipping"); return(NULL)
  }

  p1_pred <- data.frame(Player = test_df$PLAYER_1_HANDLE, OpponentPlayer = test_df$PLAYER_2_HANDLE,
                         Team = test_df$PLAYER_1_TEAM, OpponentTeam = test_df$PLAYER_2_TEAM,
                         Stream = test_df$STREAM_NUMBER)
  p2_pred <- data.frame(Player = test_df$PLAYER_2_HANDLE, OpponentPlayer = test_df$PLAYER_1_HANDLE,
                         Team = test_df$PLAYER_2_TEAM, OpponentTeam = test_df$PLAYER_1_TEAM,
                         Stream = test_df$STREAM_NUMBER)
  eta1_base <- predict(fit, newdata = p1_pred, type = "link", re.form = KNOWN_RE_FORM, allow.new.levels = TRUE)
  eta2_base <- predict(fit, newdata = p2_pred, type = "link", re.form = KNOWN_RE_FORM, allow.new.levels = TRUE)

  n_test <- nrow(test_df)
  p1_fair_ml <- numeric(n_test)
  pred_total <- numeric(n_test)
  for (i in seq_len(n_test)) {
    m <- rnorm(N_SIMS, 0, sigma_match)       # shared match-level draw -> induces score correlation
    e1 <- rnorm(N_SIMS, 0, sigma_obs)
    e2 <- rnorm(N_SIMS, 0, sigma_obs)
    mu1 <- exp(pmin(pmax(eta1_base[i] + m + e1, -15), 15))
    mu2 <- exp(pmin(pmax(eta2_base[i] + m + e2, -15), 15))
    s1 <- rpois(N_SIMS, mu1)
    s2 <- rpois(N_SIMS, mu2)
    p1_fair_ml[i] <- mean(s1 > s2) + 0.5 * mean(s1 == s2)
    pred_total[i] <- mean(s1 + s2)
  }

  p1_won <- as.numeric(test_df$PLAYER_1_FINAL_SCORE > test_df$PLAYER_2_FINAL_SCORE)
  brier <- mean((p1_fair_ml - p1_won)^2)
  eps <- 1e-9
  p_clip <- pmin(pmax(p1_fair_ml, eps), 1 - eps)
  logloss <- -mean(p1_won * log(p_clip) + (1 - p1_won) * log(1 - p_clip))

  actual_total <- test_df$PLAYER_1_FINAL_SCORE + test_df$PLAYER_2_FINAL_SCORE
  total_corr <- cor(pred_total, actual_total)
  total_bias <- mean(actual_total - pred_total)

  list(half_life = half_life, scalar = scalar, sigma_match = sigma_match, sigma_obs = sigma_obs,
       brier = brier, logloss = logloss, total_corr = total_corr, total_bias = total_bias, n = n_test)
}

cat("\n=== Fitting one glmer(poisson) model per (half-life, scalar) setting on TRAIN, evaluating on TEST ===\n")
cat(sprintf("Grid: %d half-lives x %d scalars = %d fits. This will take a while.\n",
            length(HALF_LIVES), length(SCALARS), length(HALF_LIVES) * length(SCALARS)))

results <- list()
for (hl in HALF_LIVES) {
  for (sc in SCALARS) {
    cat(sprintf("\nhalf_life=%s  scalar=%s ...\n", ifelse(is.infinite(hl), "Inf", hl), sc))
    res <- evaluate_settings(hl, sc)
    if (!is.null(res)) {
      results[[length(results) + 1]] <- res
      cat(sprintf("  sigma_match=%.3f  sigma_obs=%.3f  Brier=%.4f  LogLoss=%.4f  TotalCorr=%.3f  TotalBias=%+.2f\n",
                  res$sigma_match, res$sigma_obs, res$brier, res$logloss, res$total_corr, res$total_bias))
    }
  }
}

cat("\n=== Summary across settings (sorted by Brier score, best first) ===\n")
summary_df <- do.call(rbind, lapply(results, function(r) {
  data.frame(HalfLifeDays = ifelse(is.infinite(r$half_life), NA, r$half_life), Scalar = r$scalar,
             SigmaMatch = r$sigma_match, SigmaObs = r$sigma_obs,
             Brier = r$brier, LogLoss = r$logloss, TotalCorr = r$total_corr, TotalBias = r$total_bias, N = r$n)
}))
summary_df <- summary_df[order(summary_df$Brier), ]
print(summary_df, row.names = FALSE)

cat("\n(SigmaMatch: how much the fit thinks matches vary in pace/tempo -- 0 means it found no shared correlation.\n")
cat(" Lower Brier/LogLoss = better calibrated moneyline. Brier=0.25 is the always-50% baseline.)\n")
