# Shared code for the glmer pre-match model: loading the match history,
# pre-match features, row weights, fitting the global and per-player models,
# predicting, pricing and scoring. Sourced by backtest.R, fit.R and
# predict.R; not meant to be run on its own.
#
# The model, per side of a match (the "scorer" against the "opponent"):
#
#   Score ~ Poisson(exp(eta + m + e))
#     eta  fixed + random effects from the feature set
#     m    (1|MatchId): shared by both sides of the match (global model only)
#     e    (1|ObsID):   one per row, overdispersion
#
# so a side's expected points are exp(eta + (sd_m^2 + sd_e^2) / 2), and the
# moneyline is simulated drawing m once per match for both sides.

suppressMessages(library(lme4))

`%||%` <- function(a, b) if (is.null(a)) b else a

# ---------------------------------------------------------------------------
# Command line / RStudio
# ---------------------------------------------------------------------------

# Merge "--key=value" / "--key value" / "--flag" command-line arguments over
# a list of defaults. In RStudio there are no arguments, so the defaults
# (edited at the top of each script) are what runs. Dashes in keys become
# underscores: --feature-sets=a,b -> args$feature_sets == "a,b".
parse_args <- function(defaults) {
  argv <- commandArgs(trailingOnly = TRUE)
  out <- defaults
  i <- 1
  while (i <= length(argv)) {
    a <- argv[i]
    if (!startsWith(a, "--")) stop(sprintf("unexpected argument '%s' (use --key=value)", a))
    a <- sub("^--", "", a)
    if (grepl("=", a, fixed = TRUE)) {
      key <- sub("=.*$", "", a)
      val <- sub("^[^=]*=", "", a)
    } else if (i < length(argv) && !startsWith(argv[i + 1], "--")) {
      key <- a
      val <- argv[i + 1]
      i <- i + 1
    } else {
      key <- a
      val <- "TRUE"
    }
    key <- gsub("-", "_", key)
    if (!key %in% names(defaults)) {
      stop(sprintf("unknown option --%s (known: %s)", gsub("_", "-", key),
                   paste0("--", gsub("_", "-", names(defaults)), collapse = ", ")))
    }
    out[[key]] <- val
    i <- i + 1
  }
  out
}

split_list <- function(x) {
  if (is.null(x) || !nzchar(x)) return(character())
  trimws(strsplit(x, ",", fixed = TRUE)[[1]])
}

as_flag <- function(x) isTRUE(x) || (is.character(x) && toupper(x) %in% c("TRUE", "T", "1", "YES"))

# The history CSV: --history if given, else the first of HISTORY_CANDIDATES
# that exists under the repo root.
resolve_history <- function(path, repo_root) {
  if (!is.null(path) && nzchar(path)) {
    if (!file.exists(path)) stop(sprintf("history file not found: %s", path))
    return(normalizePath(path))
  }
  for (p in HISTORY_CANDIDATES) {
    full <- file.path(repo_root, p)
    if (file.exists(full)) return(normalizePath(full))
  }
  stop(sprintf("no history file found; pass --history (looked for %s under %s)",
               paste(HISTORY_CANDIDATES, collapse = ", "), repo_root))
}

log_line <- function(...) {
  cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), sprintf(...), "\n", sep = "")
  flush.console()
}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS <- c("MATCH_CODE", "STREAM_NUMBER", "SCHEDULED_START_TIME_UTC",
                      "PLAYER_1_HANDLE", "PLAYER_1_TEAM", "PLAYER_2_HANDLE", "PLAYER_2_TEAM")

parse_time <- function(x) {
  x <- sub("T", " ", substr(as.character(x), 1, 19), fixed = TRUE)
  as.POSIXct(x, format = "%Y-%m-%d %H:%M:%S", tz = "UTC")
}

# A cut-off from the command line: "YYYY-MM-DD", "YYYY-MM-DD HH:MM[:SS]" or
# the same with a "T", in UTC.
parse_cutoff <- function(x) {
  t <- parse_time(x)
  if (is.na(t)) t <- as.POSIXct(sub("T", " ", x, fixed = TRUE), format = "%Y-%m-%d %H:%M", tz = "UTC")
  if (is.na(t)) t <- as.POSIXct(x, format = "%Y-%m-%d", tz = "UTC")
  if (is.na(t)) stop(sprintf("can't read the date '%s' (use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)", x))
  t
}

# A flat named list as a JSON object, for the Python side to read.
write_json <- function(x, path) {
  value <- function(v) {
    if (is.null(v) || (length(v) == 1 && (is.na(v) || (is.numeric(v) && is.infinite(v))))) return("null")
    if (is.logical(v)) return(tolower(as.character(v)))
    if (is.numeric(v)) return(format(v, digits = 10))
    v <- gsub("\\", "/", as.character(v), fixed = TRUE)   # Windows paths
    sprintf('"%s"', gsub('"', '\\"', v, fixed = TRUE))
  }
  body <- paste0('  "', names(x), '": ', vapply(x, value, ""), collapse = ",\n")
  writeLines(c("{", body, "}"), path)
}

# A match-history (or schedule) CSV shaped like eAMFCalibrator's
# match_history.csv / nb2/AMFELO.csv, one row per match, cleaned. Final
# scores are optional (NA for a match not yet played).
load_matches <- function(path, sport = SPORT_CODE) {
  df <- read.csv(path, stringsAsFactors = FALSE, colClasses = "character")
  missing <- setdiff(REQUIRED_COLUMNS, names(df))
  if (length(missing)) stop(sprintf("%s lacks %s", path, paste(missing, collapse = ", ")))
  if ("SPORT_CODE" %in% names(df)) df <- df[df$SPORT_CODE %in% c(sport, "", NA), ]
  for (s in c("PLAYER_1_FINAL_SCORE", "PLAYER_2_FINAL_SCORE")) {
    df[[s]] <- if (s %in% names(df)) suppressWarnings(as.integer(df[[s]])) else NA_integer_
  }
  out <- data.frame(
    MATCH_CODE = trimws(df$MATCH_CODE),
    Time = parse_time(df$SCHEDULED_START_TIME_UTC),
    Stream = trimws(df$STREAM_NUMBER),
    P1 = toupper(trimws(df$PLAYER_1_HANDLE)), P1Team = trimws(df$PLAYER_1_TEAM),
    P2 = toupper(trimws(df$PLAYER_2_HANDLE)), P2Team = trimws(df$PLAYER_2_TEAM),
    P1Score = df$PLAYER_1_FINAL_SCORE, P2Score = df$PLAYER_2_FINAL_SCORE,
    stringsAsFactors = FALSE)
  bad_score <- (!is.na(out$P1Score) & out$P1Score < 0) | (!is.na(out$P2Score) & out$P2Score < 0)
  half_score <- xor(is.na(out$P1Score), is.na(out$P2Score))
  out$P1Score[bad_score | half_score] <- NA
  out$P2Score[bad_score | half_score] <- NA
  keep <- nzchar(out$MATCH_CODE) & !is.na(out$Time) & nzchar(out$P1) & nzchar(out$P2) &
          nzchar(out$P1Team) & nzchar(out$P2Team) & out$P1 != out$P2
  out <- out[keep, ]
  # One row per match code: the settled one if there is one, else the last.
  out <- out[order(!is.na(out$P1Score), out$Time), ]
  out <- out[!duplicated(out$MATCH_CODE, fromLast = TRUE), ]
  out <- out[order(out$Time, out$MATCH_CODE), ]
  rownames(out) <- NULL
  out
}

# History and schedule together (the schedule's row wins a shared match
# code), so pre-match features see everything that happened before each
# scheduled match.
combine_matches <- function(history, schedule) {
  if (is.null(schedule) || !nrow(schedule)) return(history)
  both <- rbind(history[!history$MATCH_CODE %in% schedule$MATCH_CODE, ], schedule)
  both <- both[order(both$Time, both$MATCH_CODE), ]
  rownames(both) <- NULL
  both
}

# ---------------------------------------------------------------------------
# Long format and pre-match features
# ---------------------------------------------------------------------------

# Two rows per match, one per side, with every pre-match feature. A row's
# features use only matches that started strictly before it, so a side's own
# result (or anything later) never leaks into its features.
to_long <- function(m, form_half_life = FORM_HALF_LIFE_MATCHES,
                    session_hours = SESSION_HOURS, rest_cap_hours = REST_CAP_HOURS,
                    session_gap_hours = SESSION_GAP_HOURS, session_shrink = SESSION_SHRINK_MATCHES,
                    exp_cap = EXP_CAP_MATCHES, exp_floor = 0) {
  home <- data.frame(MatchId = m$MATCH_CODE, Time = m$Time, Side = "H", IsHome = 1,
                     Player = m$P1, OpponentPlayer = m$P2, Team = m$P1Team, OpponentTeam = m$P2Team,
                     Stream = m$Stream, Score = m$P1Score, OppScore = m$P2Score,
                     stringsAsFactors = FALSE)
  away <- data.frame(MatchId = m$MATCH_CODE, Time = m$Time, Side = "A", IsHome = 0,
                     Player = m$P2, OpponentPlayer = m$P1, Team = m$P2Team, OpponentTeam = m$P1Team,
                     Stream = m$Stream, Score = m$P2Score, OppScore = m$P1Score,
                     stringsAsFactors = FALSE)
  d <- rbind(home, away)
  d <- d[order(d$Time, d$MatchId, d$Side), ]
  rownames(d) <- NULL
  n <- nrow(d)
  tnum <- as.numeric(d$Time)
  done <- !is.na(d$Score)

  # League average points per side over every settled row before this one's
  # start time (the whole-history mean for the very first rows).
  first_at <- match(tnum, tnum)
  cs <- c(0, cumsum(ifelse(done, d$Score, 0)))
  cn <- c(0, cumsum(done))
  prior_sum <- cs[first_at]
  prior_n <- cn[first_at]
  overall <- if (any(done)) mean(d$Score[done]) else 17
  league <- ifelse(prior_n > 0, prior_sum / pmax(prior_n, 1), overall)

  alpha <- 1 - 2^(-1 / form_half_life)
  form_for <- form_against <- rest <- session <- exper <- numeric(n)
  for (idx in split(seq_len(n), d$Player)) {
    idx <- idx[order(tnum[idx])]
    s_for <- s_against <- NA_real_
    played <- 0
    prev_t <- NA_real_
    for (k in seq_along(idx)) {
      i <- idx[k]
      if (is.na(s_for)) {
        s_for <- league[i]
        s_against <- league[i]
      }
      form_for[i] <- log((s_for + 1) / (league[i] + 1))
      form_against[i] <- log((s_against + 1) / (league[i] + 1))
      exper[i] <- min(max(played, exp_floor), exp_cap)
      rest[i] <- if (is.na(prev_t)) rest_cap_hours else min(rest_cap_hours, (tnum[i] - prev_t) / 3600)
      earlier <- tnum[idx[seq_len(k - 1)]]
      session[i] <- sum(earlier >= tnum[i] - session_hours * 3600 & earlier < tnum[i])
      prev_t <- tnum[i]
      if (done[i]) {
        s_for <- s_for + alpha * (d$Score[i] - s_for)
        s_against <- s_against + alpha * (d$OppScore[i] - s_against)
        played <- played + 1
      }
    }
  }
  d$FormFor <- form_for
  d$FormAgainst <- form_against
  d$RestLog <- log1p(rest)
  d$Session <- session
  d$ExpLog <- log1p(exper)

  opp <- match(paste(d$MatchId, ifelse(d$Side == "H", "A", "H")), paste(d$MatchId, d$Side))
  d$OppFormFor <- d$FormFor[opp]
  d$OppFormAgainst <- d$FormAgainst[opp]
  d$OppRestLog <- d$RestLog[opp]
  d$OppSession <- d$Session[opp]
  d$OppExpLog <- d$ExpLog[opp]

  # Session form. A session is a side's run of matches with gaps under
  # session_gap_hours. Each settled match in it is scored against a rough
  # pre-match expectation from the form features -- the side's own scoring
  # (conceding) form times the opponent's conceding (scoring) form -- and a
  # match's SessFormFor / SessFormAgainst is the mean log ratio over the
  # session's earlier matches, shrunk toward 0 by session_shrink matches.
  # Earlier in the session means the side's previous matches, which have all
  # finished before this one starts.
  expect_for <- (league + 1) * exp(d$FormFor + d$OppFormAgainst)
  expect_against <- (league + 1) * exp(d$FormAgainst + d$OppFormFor)
  res_for <- ifelse(done, log((d$Score + 1) / expect_for), 0)
  res_against <- ifelse(done, log((d$OppScore + 1) / expect_against), 0)
  sess_pos <- sess_for <- sess_against <- numeric(n)
  gap <- session_gap_hours * 3600
  for (idx in split(seq_len(n), d$Player)) {
    idx <- idx[order(tnum[idx])]
    prev_t <- -Inf
    pos <- k <- 0
    sum_for <- sum_against <- 0
    for (i in idx) {
      if (tnum[i] - prev_t > gap) {
        pos <- k <- 0
        sum_for <- sum_against <- 0
      }
      pos <- pos + 1
      sess_pos[i] <- pos
      if (k > 0) {
        sess_for[i] <- sum_for / (k + session_shrink)        # = mean * k / (k + shrink)
        sess_against[i] <- sum_against / (k + session_shrink)
      }
      if (done[i]) {
        sum_for <- sum_for + res_for[i]
        sum_against <- sum_against + res_against[i]
        k <- k + 1
      }
      prev_t <- tnum[i]
    }
  }
  d$SessPos <- sess_pos
  d$SessBand <- as.character(cut(sess_pos, c(0, 1, 2, 4, 6, 8, Inf),
                                 labels = c("s1", "s2", "s3_4", "s5_6", "s7_8", "s9p")))
  d$SessFormFor <- sess_for
  d$SessFormAgainst <- sess_against
  d$OppSessBand <- d$SessBand[opp]
  d$OppSessFormFor <- d$SessFormFor[opp]
  d$OppSessFormAgainst <- d$SessFormAgainst[opp]

  hour <- as.integer(format(d$Time, "%H", tz = "UTC"))
  d$HourBlock <- sprintf("h%02d", 4L * (hour %/% 4L))
  d$Weekday <- paste0("d", format(d$Time, "%u", tz = "UTC"))
  d$TeamPair <- paste(d$Team, "v", d$OpponentTeam)
  d$ObsID <- as.character(seq_len(n))
  d
}

# The long rows as one player's own model sees them. Long rows are from the
# scorer's side; a player's attack model reads the rows they score in and
# their defence model the rows they concede in, so "Own" is the scorer in
# the first and the opponent in the second. That lets one per-player formula
# mean the same thing in both models: (1|Rival) is who they're up against,
# OwnHome is whether they're at home, OwnFormFor is their scoring form, etc.
role_view <- function(d, role) {
  att <- role == "attack"
  pick <- function(scorer, opponent) if (att) scorer else opponent
  d$Rival <- pick(d$OpponentPlayer, d$Player)
  d$OwnTeam <- pick(d$Team, d$OpponentTeam)
  d$RivalTeam <- pick(d$OpponentTeam, d$Team)
  d$Matchup <- paste(d$OwnTeam, "v", d$RivalTeam)
  d$OwnHome <- pick(d$IsHome, 1 - d$IsHome)
  for (f in c("FormFor", "FormAgainst", "RestLog", "Session", "ExpLog", "SessBand", "SessFormFor",
               "SessFormAgainst")) {
    opp <- paste0("Opp", f)
    d[[paste0("Own", f)]] <- pick(d[[f]], d[[opp]])
    d[[paste0("Rival", f)]] <- pick(d[[opp]], d[[f]])
  }
  d
}

# ---------------------------------------------------------------------------
# Row weights
# ---------------------------------------------------------------------------

# Weights for the rows of `d` as of `as_of`, normalised to mean 1 then times
# the spec's scalar. Rows the spec drops come back as 0. `group` is the
# column exp_matches counts matches within.
row_weights <- function(d, spec, as_of, group = "Player") {
  age_days <- as.numeric(difftime(as_of, d$Time, units = "days"))
  w <- switch(spec$type,
    none = rep(1, nrow(d)),
    exp_days = 2^(-age_days / spec$half_life),
    exp_matches = {
      k <- ave(-as.numeric(d$Time), d[[group]], FUN = function(t) rank(t, ties.method = "first") - 1)
      2^(-k / spec$half_life)
    },
    window = as.numeric(age_days <= spec$days),
    step = ifelse(age_days <= spec$days, 1, spec$older),
    stop(sprintf("unknown weighting type '%s'", spec$type)))
  if (!is.null(spec$drop_before)) {
    w[d$Time < as.POSIXct(spec$drop_before, tz = "UTC")] <- 0
  }
  if (any(w > 0)) w <- w / mean(w[w > 0])
  w * (spec$scalar %||% 1)
}

# ---------------------------------------------------------------------------
# Formulas
# ---------------------------------------------------------------------------

# Split a right-hand side on the "+" signs outside parentheses.
split_terms <- function(rhs) {
  chars <- strsplit(rhs, "")[[1]]
  depth <- 0
  cur <- ""
  out <- character()
  for (ch in chars) {
    if (ch == "(") depth <- depth + 1
    if (ch == ")") depth <- depth - 1
    if (ch == "+" && depth == 0) {
      out <- c(out, trimws(cur))
      cur <- ""
    } else {
      cur <- paste0(cur, ch)
    }
  }
  out <- c(out, trimws(cur))
  out[nzchar(out)]
}

# Drop terms the rows can't support: a random effect whose grouping has
# fewer than two levels, a fixed factor with one level, a constant numeric.
prune_terms <- function(terms, d) {
  keep <- vapply(terms, function(tm) {
    bar <- regmatches(tm, regexec("^\\((.*)\\|(.*)\\)$", tm))[[1]]
    if (length(bar)) {
      grp <- trimws(strsplit(bar[3], ":", fixed = TRUE)[[1]])
      if (!all(grp %in% names(d))) stop(sprintf("no column for grouping in term %s", tm))
      return(length(unique(do.call(paste, d[grp]))) >= 2)
    }
    vars <- all.vars(parse(text = tm)[[1]])
    for (v in vars) {
      if (!v %in% names(d)) stop(sprintf("no column '%s' (term %s)", v, tm))
      x <- d[[v]]
      if (length(unique(x)) < 2) return(FALSE)
    }
    TRUE
  }, logical(1))
  terms[keep]
}

build_formula <- function(rhs, d, extra = character(), offset = FALSE) {
  terms <- c(prune_terms(split_terms(rhs), d), extra)
  if (offset) terms <- c(terms, "offset(GlobalEta)")
  if (!length(terms)) terms <- "1"
  as.formula(paste("Score ~", paste(terms, collapse = " + ")), env = globalenv())
}

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

# glmer(poisson) with warnings collected rather than printed (singular fits
# and convergence notes are routine with this many variance components)
# and errors turned into NULL.
fit_glmer <- function(formula, d, start = NULL) {
  warns <- character()
  fit <- tryCatch(
    withCallingHandlers(
      glmer(formula, data = d, family = poisson, weights = .w, nAGQ = GLMER_NAGQ, start = start,
            control = glmerControl(optimizer = GLMER_OPTIMIZER, calc.derivs = FALSE,
                                   check.conv.singular = "ignore",
                                   optCtrl = if (GLMER_OPTIMIZER == "nloptwrap") {
                                     list(maxeval = GLMER_MAXFUN)
                                   } else {
                                     list(maxfun = GLMER_MAXFUN)
                                   })),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }),
    error = function(e) {
      warns <<- c(warns, paste("ERROR:", conditionMessage(e)))
      NULL
    })
  list(fit = fit, warnings = unique(warns))
}

re_sd <- function(fit, group) {
  vc <- as.data.frame(VarCorr(fit))
  s <- vc$sdcor[vc$grp == group & is.na(vc$var2) & vc$var1 == "(Intercept)"]
  if (length(s)) s[1] else 0
}

# New rows with every numeric fixed-effect covariate held to the range the
# fit saw in training, so a slope estimated on a feature that barely moved
# (a new player's form, say) can't be extrapolated into absurd predictions.
clamp_to_frame <- function(fit, newdata) {
  vars <- setdiff(all.vars(nobars(formula(fit))[[3]]), "GlobalEta")
  fr <- fit@frame
  for (v in vars) {
    if (v %in% names(fr) && is.numeric(fr[[v]]) && v %in% names(newdata)) {
      newdata[[v]] <- pmin(pmax(newdata[[v]], min(fr[[v]])), max(fr[[v]]))
    }
  }
  newdata
}

# The fit's linear predictor for new rows, without the per-row and per-match
# noise terms. Random-effect levels the fit hasn't seen count as 0.
fit_eta <- function(fit, newdata) {
  newdata <- clamp_to_frame(fit, newdata)
  bars <- findbars(formula(fit))
  keep <- Filter(function(b) !deparse(b[[3]]) %in% c("ObsID", "MatchId"), bars)
  re <- if (length(keep)) {
    as.formula(paste("~", paste(vapply(keep, function(b) paste0("(", deparse(b), ")"), ""),
                                collapse = " + ")))
  } else NA
  as.numeric(predict(fit, newdata = newdata, re.form = re, type = "link", allow.new.levels = TRUE))
}

# Which new rows a fit can price: every fixed-effect factor level must have
# been seen in its training rows.
fixed_levels_ok <- function(fit, newdata) {
  vars <- setdiff(all.vars(nobars(formula(fit))[[3]]), "GlobalEta")
  fr <- fit@frame
  ok <- rep(TRUE, nrow(newdata))
  for (v in vars) {
    if (v %in% names(fr) && (is.factor(fr[[v]]) || is.character(fr[[v]]))) {
      ok <- ok & as.character(newdata[[v]]) %in% as.character(unique(fr[[v]]))
    }
  }
  ok
}

# The global model on every settled row before as_of. `start` (optional): the
# variance parameters (theta) of an earlier fit of the same formula, a warm
# start for a refit on a little more data.
fit_global <- function(train, fs, wspec, as_of, start = NULL) {
  train$.w <- row_weights(train, wspec, as_of, group = "Player")
  train <- train[train$.w > 0, ]
  extra <- c(if (isTRUE(fs$match_re)) "(1|MatchId)", if (isTRUE(fs$olre)) "(1|ObsID)")
  f <- build_formula(fs$global, train, extra)
  res <- fit_glmer(f, train, start = if (is.null(start)) NULL else list(theta = start))
  if (is.null(res$fit) && !is.null(start)) res <- fit_glmer(f, train)   # the warm start didn't fit
  if (is.null(res$fit)) stop(sprintf("global model failed: %s", paste(res$warnings, collapse = "; ")))
  list(fit = res$fit, formula = deparse1(f), n = nrow(train),
       sigma_obs = re_sd(res$fit, "ObsID"), sigma_match = re_sd(res$fit, "MatchId"),
       warnings = res$warnings)
}

# One player's attack (role = "attack": rows they score in) or defence
# (role = "defence": rows they concede in) model. NULL when they have too
# few rows or the fit fails.
fit_player <- function(train, player, role, fs, wspec, as_of, min_matches) {
  col <- if (role == "attack") "Player" else "OpponentPlayer"
  d <- role_view(train[train[[col]] == player, ], role)
  d$.w <- row_weights(d, wspec, as_of, group = col)
  d <- d[d$.w > 0, ]
  if (nrow(d) < min_matches) return(NULL)
  f <- build_formula(fs$player, d, "(1|ObsID)", offset = isTRUE(fs$player_offset))
  res <- fit_glmer(f, d)
  if (is.null(res$fit)) return(list(fit = NULL, n = nrow(d), warnings = res$warnings))
  list(fit = res$fit, formula = deparse1(f), n = nrow(d), sigma_obs = re_sd(res$fit, "ObsID"),
       warnings = res$warnings)
}

# Everything: the global model, then every qualifying player's attack and
# defence models. `train` is the long data (settled rows before as_of).
# `cluster` (optional) spreads the per-player fits over parallel workers
# that have already sourced this file.
fit_bundle <- function(train, fs, wspec, as_of, min_matches = MIN_PLAYER_MATCHES,
                       cluster = NULL, verbose = TRUE, fs_name = NA, w_name = NA,
                       with_players = TRUE, form_half_life = FORM_HALF_LIFE_MATCHES,
                       start = NULL) {
  t0 <- Sys.time()
  global <- fit_global(train, fs, wspec, as_of, start = start)
  if (verbose) {
    log_line("  global: %d rows, sd(match)=%.3f sd(obs)=%.3f, %.0fs%s", global$n,
             global$sigma_match, global$sigma_obs, as.numeric(difftime(Sys.time(), t0, units = "secs")),
             if (length(global$warnings)) sprintf(" (%d warnings)", length(global$warnings)) else "")
  }
  if (isTRUE(fs$player_offset)) train$GlobalEta <- fit_eta(global$fit, train)

  players <- sort(unique(c(train$Player, train$OpponentPlayer)))
  if (!with_players) players <- character()   # global model only
  jobs <- expand.grid(player = players, role = c("attack", "defence"), stringsAsFactors = FALSE)
  one <- function(j) fit_player(train, jobs$player[j], jobs$role[j], fs, wspec, as_of, min_matches)
  t1 <- Sys.time()
  fits <- if (is.null(cluster) || !nrow(jobs)) {
    lapply(seq_len(nrow(jobs)), one)
  } else {
    # The workers get the data once, as globals; the task function itself
    # carries no environment (else every task would ship the global fit).
    parallel::clusterExport(cluster, c("train", "jobs", "fs", "wspec", "as_of", "min_matches"),
                            envir = environment())
    environment(one) <- globalenv()
    parallel::parLapplyLB(cluster, seq_len(nrow(jobs)), one)
  }
  per_player <- list()
  for (j in seq_len(nrow(jobs))) {
    f <- fits[[j]]
    if (is.null(f) || is.null(f$fit)) next
    per_player[[jobs$player[j]]][[jobs$role[j]]] <- f
  }
  failed <- sum(vapply(fits, function(f) !is.null(f) && is.null(f$fit), logical(1)))
  if (verbose && !with_players) {
    log_line("  per-player: not fitted (global model only)")
  } else if (verbose) {
    n_att <- sum(vapply(per_player, function(p) !is.null(p$attack), logical(1)))
    n_def <- sum(vapply(per_player, function(p) !is.null(p$defence), logical(1)))
    log_line("  per-player: %d attack + %d defence models of %d players (>= %d matches), %d failed, %.0fs",
             n_att, n_def, length(players), min_matches, failed,
             as.numeric(difftime(Sys.time(), t1, units = "secs")))
  }
  list(global = global, players = per_player, feature_set = fs_name, fs = fs,
       weighting = w_name, wspec = wspec, as_of = as_of, min_matches = min_matches,
       n_train_rows = nrow(train), failed_player_fits = failed,
       features = list(form_half_life = form_half_life, session_hours = SESSION_HOURS,
                       rest_cap_hours = REST_CAP_HOURS, session_gap_hours = SESSION_GAP_HOURS,
                       session_shrink = SESSION_SHRINK_MATCHES, exp_cap = EXP_CAP_MATCHES),
       fitted_at = Sys.time())
}

# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------

# Every source's linear predictor and overdispersion sd for each row of
# `newdata` (long format): the global model, the scorer's attack model and
# the opponent's defence model (NA where there is none).
predict_sources <- function(bundle, newdata) {
  g <- bundle$global
  newdata$GlobalEta <- fit_eta(g$fit, newdata)
  s_match2 <- g$sigma_match^2
  out <- data.frame(MatchId = newdata$MatchId, Side = newdata$Side,
                    eta_global = newdata$GlobalEta, s_global = g$sigma_obs,
                    eta_attack = NA_real_, s_attack = NA_real_, capped_attack = FALSE,
                    eta_defence = NA_real_, s_defence = NA_real_, capped_defence = FALSE,
                    stringsAsFactors = FALSE)
  for (role in c("attack", "defence")) {
    col <- if (role == "attack") "Player" else "OpponentPlayer"
    for (p in intersect(unique(newdata[[col]]), names(bundle$players))) {
      m <- bundle$players[[p]][[role]]
      if (is.null(m)) next
      rows <- which(newdata[[col]] == p)
      view <- role_view(newdata[rows, , drop = FALSE], role)
      ok <- fixed_levels_ok(m$fit, view)
      rows <- rows[ok]
      if (!length(rows)) next
      eta <- fit_eta(m$fit, view[ok, , drop = FALSE])
      # Held within PLAYER_MAX_SHIFT of the global model's prediction.
      lo <- out$eta_global[rows] - PLAYER_MAX_SHIFT
      hi <- out$eta_global[rows] + PLAYER_MAX_SHIFT
      out[[paste0("capped_", role)]][rows] <- eta < lo | eta > hi
      out[[paste0("eta_", role)]][rows] <- pmin(pmax(eta, lo), hi)
      # A per-player model has no match term, so its row sd also carries the
      # match-level swing; take that out, it's added back once per match.
      out[[paste0("s_", role)]][rows] <- sqrt(max(0, m$sigma_obs^2 - s_match2))
    }
  }
  out
}

# One linear predictor + sd per row for a prediction mode, and where it
# came from. Anything missing falls back to the global model.
combine_mode <- function(src, mode) {
  a <- !is.na(src$eta_attack)
  d <- !is.na(src$eta_defence)
  eta <- src$eta_global
  s <- src$s_global
  source <- rep("global", nrow(src))
  if (mode == "attack") {
    eta[a] <- src$eta_attack[a]; s[a] <- src$s_attack[a]; source[a] <- "attack"
  } else if (mode == "defence") {
    eta[d] <- src$eta_defence[d]; s[d] <- src$s_defence[d]; source[d] <- "defence"
  } else if (mode %in% c("pair", "blend")) {
    both <- a & d
    eta[both] <- (src$eta_attack[both] + src$eta_defence[both]) / 2
    s[both] <- sqrt((src$s_attack[both]^2 + src$s_defence[both]^2) / 2)
    source[both] <- "pair"
    oa <- a & !d
    eta[oa] <- src$eta_attack[oa]; s[oa] <- src$s_attack[oa]; source[oa] <- "attack"
    od <- d & !a
    eta[od] <- src$eta_defence[od]; s[od] <- src$s_defence[od]; source[od] <- "defence"
    if (mode == "blend") {
      own <- a | d
      w <- BLEND_WEIGHT
      eta[own] <- w * eta[own] + (1 - w) * src$eta_global[own]
      s[own] <- sqrt(w * s[own]^2 + (1 - w) * src$s_global[own]^2)
      source[own] <- paste0("blend_", source[own])
    }
  } else if (mode != "global") {
    stop(sprintf("unknown prediction mode '%s'", mode))
  }
  data.frame(eta = eta, s = s, source = source, stringsAsFactors = FALSE)
}

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

# Per match: each side's expected points and the home side's moneyline
# (ties split half/half, as in the NB2 backtests), simulated with one match
# draw shared by both sides. Chunked so memory stays small. n_sims = 0 skips
# the simulation: expected points only, moneyline NA.
price_pairs <- function(eta_h, eta_a, s_h, s_a, s_match, n_sims = N_SIMS, seed = SEED, chunk = 100) {
  n <- length(eta_h)
  mu_h <- exp(eta_h + (s_h^2 + s_match^2) / 2)
  mu_a <- exp(eta_a + (s_a^2 + s_match^2) / 2)
  p_home <- numeric(n)
  if (n == 0) return(data.frame(mu_h = mu_h, mu_a = mu_a, p_home = p_home))
  if (n_sims <= 0) return(data.frame(mu_h = mu_h, mu_a = mu_a, p_home = NA_real_))
  set.seed(seed)
  for (start in seq(1, n, by = chunk)) {
    i <- start:min(n, start + chunk - 1)
    k <- length(i)
    m <- matrix(rnorm(k * n_sims, 0, s_match), k)
    lh <- pmin(pmax(eta_h[i] + m + matrix(rnorm(k * n_sims), k) * s_h[i], -15), 15)
    la <- pmin(pmax(eta_a[i] + m + matrix(rnorm(k * n_sims), k) * s_a[i], -15), 15)
    yh <- matrix(rpois(k * n_sims, exp(lh)), k)
    ya <- matrix(rpois(k * n_sims, exp(la)), k)
    p_home[i] <- rowMeans(yh > ya) + 0.5 * rowMeans(yh == ya)
  }
  data.frame(mu_h = mu_h, mu_a = mu_a, p_home = p_home)
}

# Match-level predictions for one mode: one row per match with both sides'
# expected points, the home win probability and each side's source.
price_matches <- function(bundle, src, mode, n_sims = N_SIMS, seed = SEED) {
  cm <- combine_mode(src, mode)
  cm$MatchId <- src$MatchId
  cm$Side <- src$Side
  h <- cm[cm$Side == "H", ]
  a <- cm[cm$Side == "A", ]
  a <- a[match(h$MatchId, a$MatchId), ]
  pr <- price_pairs(h$eta, a$eta, h$s, a$s, bundle$global$sigma_match, n_sims, seed)
  data.frame(MATCH_CODE = h$MatchId, Pred_P1_Points = pr$mu_h, Pred_P2_Points = pr$mu_a,
             P1_Win_Prob = pr$p_home, Pred_Total = pr$mu_h + pr$mu_a,
             Pred_Spread = pr$mu_h - pr$mu_a, P1_Source = h$source, P2_Source = a$source,
             stringsAsFactors = FALSE)
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# The same headline numbers as nb2/backtest_nb2_calibration.py (Brier, log
# loss, totals bias / RMSE / correlation) plus per-side and margin errors.
score_predictions <- function(p) {
  won <- ifelse(p$P1Score > p$P2Score, 1, ifelse(p$P1Score < p$P2Score, 0, 0.5))
  prob <- pmin(pmax(p$P1_Win_Prob, 1e-9), 1 - 1e-9)
  total <- p$P1Score + p$P2Score
  margin <- p$P1Score - p$P2Score
  side_err <- c(p$P1Score - p$Pred_P1_Points, p$P2Score - p$Pred_P2_Points)
  data.frame(
    N = nrow(p),
    Brier = mean((prob - won)^2),
    LogLoss = -mean(won * log(prob) + (1 - won) * log(1 - prob)),
    TotalBias = mean(total - p$Pred_Total),
    TotalRMSE = sqrt(mean((total - p$Pred_Total)^2)),
    TotalCorr = suppressWarnings(cor(total, p$Pred_Total)),
    MarginRMSE = sqrt(mean((margin - p$Pred_Spread)^2)),
    MarginCorr = suppressWarnings(cor(margin, p$Pred_Spread)),
    SideRMSE = sqrt(mean(side_err^2)),
    SideMAE = mean(abs(side_err)))
}

# Moneyline calibration in equal-count buckets of the predicted home win
# probability, as nb2/backtest_nb2_calibration.py prints it: gap_pp is
# realised minus predicted, in percentage points.
calibration_table <- function(p, n_buckets = 10) {
  won <- ifelse(p$P1Score > p$P2Score, 1, ifelse(p$P1Score < p$P2Score, 0, 0.5))
  breaks <- unique(quantile(p$P1_Win_Prob, seq(0, 1, length.out = n_buckets + 1)))
  bucket <- cut(p$P1_Win_Prob, breaks, include.lowest = TRUE)
  out <- data.frame(Bucket = levels(bucket), N = as.vector(table(bucket)),
                    Predicted = as.vector(tapply(p$P1_Win_Prob, bucket, mean)),
                    Realized = as.vector(tapply(won, bucket, mean)), stringsAsFactors = FALSE)
  out$GapPP <- 100 * (out$Realized - out$Predicted)
  out
}
