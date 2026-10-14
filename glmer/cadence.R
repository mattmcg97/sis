# How often should glmer be refitted? The same model refitted once a day against refitted after
# every result, out of sample, match by match.
#
# Form already follows every game without a refit: FormFor / FormAgainst are recomputed from every
# result before a match whenever it is priced. What a refit adds is fresh player and team effects
# and coefficients. This measures what refreshing them after every result is worth.
#
# The production model (form_clock, with the per-match and per-row noise terms) takes minutes to
# fit, far too slow to refit after every game. A trimmed one drops those terms (and Stream and
# HourBlock) and fits in seconds from the fit before. Three arms:
#   full_daily   the production model (FEATURE_SETS$form_clock), refitted at each midnight
#   trim_daily   the trimmed model, refitted at each midnight
#   trim_game    the trimmed model, refitted before every kick-off on every result in by then
#                (RESULT_LAG minutes after a match's kick-off); a kick-off with no new result in
#                since the last fit is priced by that fit
# full_daily against trim_daily is what the trimming costs; trim_daily against trim_game is what
# refitting after every result gains.
#
# Every arm prices the same matches with the same features. Each side is priced with the
# production floor on experience (EXP_FLOOR_MATCHES). The moneyline is a normal around the
# expected margin (--margin-sd), as eAMFModel's shrink check does, with ties left out.
# One caveat: like backtest.R, a match's form covariates read every match that kicked off before
# it, even one whose result wasn't in yet. Every arm reads the same covariates, so the comparison
# is unaffected.
#
#   Rscript glmer/cadence.R                                   # the last 14 full days of the history
#   Rscript glmer/cadence.R --history=eAMFCalibrator/out/match_history.csv --days=14 --cores=4
#   Rscript glmer/cadence.R --arms=trim_daily,trim_game       # skip the slow production fits
#
# Writes (to --out, default glmer/out/cadence/):
#   cadence_matches.csv   every test match: the result and each arm's expected points
#   cadence_fits.csv      every fit: arm, cut-off, rows, seconds, variance parameters
#   cadence_report.txt    the comparison (also printed)

DEFAULTS <- list(
  history = "",         # "" = first of HISTORY_CANDIDATES in config.R
  out = "",             # "" = glmer/out/cadence
  test_from = "",       # "YYYY-MM-DD"; "" = --days before --test-until
  test_until = "",      # "YYYY-MM-DD" (exclusive); "" = the last match's day (a part day is left out)
  days = "14",
  arms = "full_daily,trim_daily,trim_game",
  trim = paste("IsHome + FormFor + FormAgainst + OppFormFor + OppFormAgainst + ExpLog + OppExpLog +",
               "SessLate + OppSessLate + (1|Player) + (1|OpponentPlayer) + (1|Team) + (1|OpponentTeam)"),
  trim_nagq = "0",      # the trimmed fits' nAGQ (0: quickest; README's Speed)
  weighting = "hl60",
  result_lag = "36",    # minutes from kick-off until a result is in (eAMFModel's stream)
  margin_sd = "9.35",
  cores = ""            # "" = all but one
)

.here <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) return(dirname(normalizePath(f[1])))
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (!is.null(of)) return(dirname(normalizePath(of)))
  }
  getwd()
})
source(file.path(.here, "config.R"))
source(file.path(.here, "common.R"))
args <- parse_args(DEFAULTS)
repo_root <- normalizePath(file.path(.here, ".."))
out_dir <- if (nzchar(args$out)) args$out else file.path(.here, "out", "cadence")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

arms <- split_list(args$arms)
known_arms <- c("full_daily", "trim_daily", "trim_game")
if (!length(arms) || !all(arms %in% known_arms)) stop(sprintf("--arms: any of %s", paste(known_arms, collapse = ", ")))
lag <- as.numeric(args$result_lag) * 60
wspec <- WEIGHTINGS[[args$weighting]]
if (is.null(wspec)) stop(sprintf("no weighting '%s' in config.R", args$weighting))
trim_formula <- as.formula(paste("Score ~", args$trim), env = globalenv())
trim_nagq <- as.integer(args$trim_nagq)

history_path <- resolve_history(args$history, repo_root)
matches <- load_matches(history_path)
matches <- matches[!is.na(matches$P1Score), ]
long <- to_long(matches)
day0 <- function(t) as.POSIXct(format(t, "%Y-%m-%d", tz = "UTC"), tz = "UTC")
test_until <- if (nzchar(args$test_until)) as.POSIXct(args$test_until, tz = "UTC") else day0(max(long$Time))
test_from <- if (nzchar(args$test_from)) as.POSIXct(args$test_from, tz = "UTC") else
  test_until - as.numeric(args$days) * 86400
test <- long[long$Time >= test_from & long$Time < test_until, ]
log_line("history %s: %d settled matches; test %s -> %s: %d matches", history_path, nrow(matches),
         format(test_from), format(test_until), nrow(test) / 2)

# ---------------------------------------------------------------------------
# The fits each arm makes: (cut-off, the kick-offs it prices)
# ---------------------------------------------------------------------------

kickoffs <- sort(unique(test$Time))
done_at <- sort(long$Time[!is.na(long$Score)]) + lag      # when each settled row's result is in
n_in <- function(t) findInterval(as.numeric(t), as.numeric(done_at))

schedule <- function(arm) {
  if (arm == "trim_game") {
    # one fit per kick-off with a new result in since the one before; the rest reuse it
    k <- n_in(kickoffs)
    cut <- kickoffs[!duplicated(k)]
    data.frame(cut = cut, first = cut, last = c(cut[-1], test_until))
  } else {
    days <- seq(day0(test_from), day0(test_until - 1), by = 86400)
    data.frame(cut = days, first = days, last = days + 86400)
  }
}

# ---------------------------------------------------------------------------
# One run: a stretch of an arm's fits in order, each warm-started from the one before
# ---------------------------------------------------------------------------

run_chunk <- function(task) {
  arm <- task$arm
  sched <- task$sched
  start <- NULL
  preds <- list()
  fits <- list()
  for (i in seq_len(nrow(sched))) {
    cut <- sched$cut[i]
    train <- long[!is.na(long$Score) & long$Time + lag <= cut, ]
    rows <- floor_experience(test[test$Time >= sched$first[i] & test$Time < sched$last[i], ])
    if (!nrow(rows)) next
    t0 <- Sys.time()
    if (arm == "full_daily") {
      g <- fit_global(train, FEATURE_SETS$form_clock, wspec, cut, start = start)
      fit <- g$fit
      mu <- exp(fit_eta(fit, rows) + (g$sigma_match^2 + g$sigma_obs^2) / 2)
    } else {
      train$.w <- row_weights(train, wspec, cut)
      train <- train[train$.w > 0, ]
      res <- fit_glmer(trim_formula, train, start = if (is.null(start)) NULL else list(theta = start),
                       nagq = trim_nagq)
      if (is.null(res$fit) && !is.null(start)) res <- fit_glmer(trim_formula, train, nagq = trim_nagq)
      if (is.null(res$fit)) stop(sprintf("%s fit at %s failed: %s", arm, format(cut), paste(res$warnings, collapse = "; ")))
      fit <- res$fit
      mu <- exp(fit_eta(fit, rows))
    }
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    start <- lme4::getME(fit, "theta")
    preds[[length(preds) + 1]] <- data.frame(arm = arm, MatchId = rows$MatchId, Side = rows$Side, mu = mu)
    fits[[length(fits) + 1]] <- data.frame(arm = arm, cut = format(cut, "%Y-%m-%d %H:%M:%S"), rows = nrow(train),
                                           seconds = round(secs, 2), theta = paste(round(start, 4), collapse = " "))
  }
  list(preds = do.call(rbind, preds), fits = do.call(rbind, fits))
}

cores <- if (nzchar(args$cores)) as.integer(args$cores) else max(1L, parallel::detectCores() - 1L)
# a stretch per core for the per-game fits; the daily arms' days are split in two at most, so each
# stretch still warm-starts from the day before
tasks <- list()
for (arm in arms) {
  s <- schedule(arm)
  parts <- if (arm == "trim_game") cores else min(2L, cores)
  parts <- max(1L, min(parts, nrow(s)))
  for (idx in split(seq_len(nrow(s)), ceiling(seq_len(nrow(s)) * parts / nrow(s)))) {
    tasks[[length(tasks) + 1]] <- list(arm = arm, sched = s[idx, ])
  }
  log_line("%s: %d fits in %d stretch(es)", arm, nrow(s), parts)
}
# the slow production stretches first, so they don't finish last
tasks <- tasks[order(vapply(tasks, function(t) t$arm != "full_daily", logical(1)))]

t_all <- Sys.time()
results <- if (cores > 1 && length(tasks) > 1) {
  cl <- parallel::makeCluster(min(cores, length(tasks)), outfile = "")
  parallel::clusterExport(cl, c(".here"), envir = environment())
  parallel::clusterEvalQ(cl, { source(file.path(.here, "config.R")); source(file.path(.here, "common.R")); NULL })
  parallel::clusterExport(cl, c("long", "test", "lag", "wspec", "trim_formula", "trim_nagq", "test_until",
                                "run_chunk"), envir = environment())
  res <- parallel::parLapplyLB(cl, tasks, function(task) run_chunk(task))
  parallel::stopCluster(cl)
  res
} else {
  lapply(tasks, run_chunk)
}
log_line("all fits in %.0f min", as.numeric(difftime(Sys.time(), t_all, units = "mins")))

preds <- do.call(rbind, lapply(results, `[[`, "preds"))
fits <- do.call(rbind, lapply(results, `[[`, "fits"))
write.csv(fits, file.path(out_dir, "cadence_fits.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# One row per match: the result and each arm's expected points
# ---------------------------------------------------------------------------

home <- test[test$Side == "H", c("MatchId", "Time", "Player", "OpponentPlayer", "Score", "OppScore")]
names(home) <- c("MatchId", "Time", "P1", "P2", "S1", "S2")
# each side's game of the day: the later of the two gamers' (how stale a day's fit has grown)
gday <- ave(as.numeric(long$Time), long$Player, format(long$Time, "%Y-%m-%d", tz = "UTC"),
            FUN = function(t) rank(t, ties.method = "first"))
key <- paste(long$MatchId, long$Side)
home$GameOfDay <- pmax(gday[match(paste(home$MatchId, "H"), key)], gday[match(paste(home$MatchId, "A"), key)])
for (arm in arms) {
  p <- preds[preds$arm == arm, ]
  pk <- paste(p$MatchId, p$Side)
  home[[paste0(arm, "_1")]] <- p$mu[match(paste(home$MatchId, "H"), pk)]
  home[[paste0(arm, "_2")]] <- p$mu[match(paste(home$MatchId, "A"), pk)]
}
priced <- complete.cases(home[, grep("_[12]$", names(home))])
home <- home[priced, ]
home <- home[order(home$Time), ]
out <- home
out$Time <- format(out$Time, "%Y-%m-%d %H:%M:%S")
write.csv(out, file.path(out_dir, "cadence_matches.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------

sd_m <- as.numeric(args$margin_sd)
margin <- home$S1 - home$S2
total <- home$S1 + home$S2
decided <- margin != 0
won <- as.numeric(margin > 0)
scores <- function(arm) {
  m <- home[[paste0(arm, "_1")]] - home[[paste0(arm, "_2")]]
  tt <- home[[paste0(arm, "_1")]] + home[[paste0(arm, "_2")]]
  p <- pmin(pmax(1 - pnorm(0, m, sd_m), 1e-9), 1 - 1e-9)
  data.frame(ll = ifelse(decided, -(won * log(p) + (1 - won) * log(1 - p)), NA),
             brier = ifelse(decided, (p - won)^2, NA),
             m_se = (margin - m)^2, t_se = (total - tt)^2, t_err = total - tt)
}
sc <- setNames(lapply(arms, scores), arms)
day <- format(home$Time, "%Y-%m-%d", tz = "UTC")

lines <- c(sprintf("%d matches, %s to %s (%d decided); history %s", nrow(home), format(min(home$Time)),
                   format(max(home$Time)), sum(decided), basename(history_path)),
           sprintf("trimmed model: %s (nAGQ %d)", args$trim, trim_nagq), "")
fit_line <- function(arm) {
  f <- fits[fits$arm == arm, ]
  sprintf("  %-11s %5d fits, %6.1f s a fit (median %.1f), %5.1f min in all", arm, nrow(f), mean(f$seconds),
          median(f$seconds), sum(f$seconds) / 60)
}
lines <- c(lines, "fits:", vapply(arms, fit_line, ""), "")
w <- 13
lines <- c(lines, sprintf("  %-22s%s", "", paste(sprintf("%*s", w, arms), collapse = "")))
metric_line <- function(label, f, fmt) {
  sprintf("  %-22s%s", label, paste(vapply(arms, function(a) sprintf("%*s", w, sprintf(fmt, f(sc[[a]]))), ""),
                                    collapse = ""))
}
lines <- c(lines,
           metric_line("moneyline log loss", function(s) mean(s$ll, na.rm = TRUE), "%.4f"),
           metric_line("moneyline Brier", function(s) mean(s$brier, na.rm = TRUE), "%.4f"),
           metric_line("margin RMSE", function(s) sqrt(mean(s$m_se)), "%.3f"),
           metric_line("total RMSE", function(s) sqrt(mean(s$t_se)), "%.3f"),
           metric_line("total: real less priced", function(s) mean(s$t_err), "%+.2f"), "")

# paired, match by match: b less a, with its t (match-level se) and the days it was better
paired <- function(a, b, label, keep = rep(TRUE, nrow(home))) {
  row <- function(name, x, lower_better = TRUE) {
    ok <- keep & !is.na(x)
    d <- x[ok]
    t <- mean(d) / (sd(d) / sqrt(length(d)))
    by_day <- tapply(d, day[ok], mean)
    sprintf("    %-14s %+9.5f  t %+5.1f  better on %d of %d days", name, mean(d), t,
            sum(if (lower_better) by_day < 0 else by_day > 0), length(by_day))
  }
  c(sprintf("  %s (%s less %s, %d matches):", label, b, a, sum(keep)),
    row("log loss", sc[[b]]$ll - sc[[a]]$ll),
    row("Brier", sc[[b]]$brier - sc[[a]]$brier),
    row("margin sq err", sc[[b]]$m_se - sc[[a]]$m_se),
    row("total sq err", sc[[b]]$t_se - sc[[a]]$t_se))
}
if (all(c("full_daily", "trim_daily") %in% arms)) {
  lines <- c(lines, paired("full_daily", "trim_daily", "what trimming costs"), "")
}
if (all(c("trim_daily", "trim_game") %in% arms)) {
  lines <- c(lines, paired("trim_daily", "trim_game", "what refitting after every result gains"), "")
  bands <- list(`games 1-2` = c(1, 2), `games 3-5` = c(3, 5), `games 6-8` = c(6, 8), `games 9+` = c(9, Inf))
  lines <- c(lines, "  ... by how far into the day (the later gamer's game of the day):")
  for (b in names(bands)) {
    keep <- home$GameOfDay >= bands[[b]][1] & home$GameOfDay <= bands[[b]][2]
    if (sum(keep) < 30) next
    lines <- c(lines, paired("trim_daily", "trim_game", b, keep))
  }
  # how far the per-game refits moved each match's price from the day's fit
  dm <- (home$trim_game_1 - home$trim_game_2) - (home$trim_daily_1 - home$trim_daily_2)
  dt <- (home$trim_game_1 + home$trim_game_2) - (home$trim_daily_1 + home$trim_daily_2)
  lines <- c(lines, "", sprintf("  per-game refits moved the expected margin by %.2f points on average (95%% within %.2f), the total by %.2f (%.2f)",
                                mean(abs(dm)), quantile(abs(dm), 0.95), mean(abs(dt)), quantile(abs(dt), 0.95)))
}
if (all(c("full_daily", "trim_game") %in% arms)) {
  lines <- c(lines, "", paired("full_daily", "trim_game", "the trimmed model refitted after every result against production daily"))
}
writeLines(lines, file.path(out_dir, "cadence_report.txt"))
cat(paste(lines, collapse = "\n"), "\n")
log_line("-> %s", out_dir)
