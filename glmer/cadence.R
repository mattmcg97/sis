# How often should glmer be refitted, and with what? Models, row weightings and refit cadences
# side by side, out of sample, match by match.
#
# Form already follows every game without a refit: FormFor / FormAgainst are recomputed from every
# result before a match whenever it is priced. What a refit adds is fresh player and team effects
# and coefficients. Refitting after every result needs a model that fits in seconds: the production
# one (form_clock, with the per-match and per-row noise terms) takes minutes.
#
# Each arm is MODEL[:WEIGHTING][:CADENCE]:
#   MODEL      full        the production model (FEATURE_SETS$form_clock); daily only
#              trim        production without the per-match and per-row noise terms, the stream and
#                          the time of day (--trim): fits in seconds
#              trim_pair   trim + gamer v gamer, (1|Player:OpponentPlayer)
#              trim_team   trim + each gamer with the team they play as, (1|Player:Team) +
#                          (1|OpponentPlayer:OpponentTeam)
#              trim_both   trim + both
#   WEIGHTING  a name in config.R's WEIGHTINGS, or built from parts (default hl60):
#              hl<D>       2^(-days ago / D)
#              m<N>        2^(-k / N), k = how many of the scorer's own matches are more recent
#              win<D>      the last D days only, flat
#              flat        every match alike
#              any of them + _x<S>: every weight times S after normalising to mean 1 -- how much
#              the data counts, and so how hard every rating is shrunk
#   CADENCE    daily       refitted at each midnight (the default)
#              game        refitted before every kick-off on every result in by then (RESULT_LAG
#                          minutes after a match's kick-off); a kick-off with no new result since
#                          the last fit is priced by that fit
#
# The first arm is the baseline: every other arm is compared with it, match by match. A game arm
# whose daily twin (same model and weighting) is also run is compared with that too, by how far
# into the day each match was.
#
# Every arm prices the same matches with the same features. Each side is priced with the
# production floor on experience (EXP_FLOOR_MATCHES). The moneyline is a normal around the
# expected margin (--margin-sd), as eAMFModel's shrink check does, with ties left out.
# One caveat: like backtest.R, a match's form covariates read every match that kicked off before
# it, even one whose result wasn't in yet. Every arm reads the same covariates, so the comparison
# is unaffected.
#
#   Rscript glmer/cadence.R                                   # full, trim, trim:game; last 14 full days
#   Rscript glmer/cadence.R --history=eAMFCalibrator/out/match_history.csv --days=28 --cores=4 \
#       --arms=trim:hl60:daily,trim:hl60:game
#   Rscript glmer/cadence.R --days=56 --arms=trim,trim:hl30,trim:hl60_x0.5,trim:m150,trim_pair,trim_team
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
  arms = "full:hl60:daily,trim:hl60:daily,trim:hl60:game",
  trim = paste("IsHome + FormFor + FormAgainst + OppFormFor + OppFormAgainst + ExpLog + OppExpLog +",
               "SessLate + OppSessLate + (1|Player) + (1|OpponentPlayer) + (1|Team) + (1|OpponentTeam)"),
  trim_nagq = "0",      # the trimmed fits' nAGQ (0: quickest; README's Speed)
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

# ---------------------------------------------------------------------------
# The arms
# ---------------------------------------------------------------------------

MODELS <- list(
  trim = args$trim,
  trim_pair = paste(args$trim, "+ (1|Player:OpponentPlayer)"),
  trim_team = paste(args$trim, "+ (1|Player:Team) + (1|OpponentPlayer:OpponentTeam)"),
  trim_both = paste(args$trim, "+ (1|Player:OpponentPlayer) + (1|Player:Team) + (1|OpponentPlayer:OpponentTeam)")
)

# A weighting by name: config.R's, else hl<D> / m<N> / win<D> / flat, each optionally _x<scalar>.
weighting_spec <- function(name) {
  if (!is.null(WEIGHTINGS[[name]])) return(WEIGHTINGS[[name]])
  parts <- regmatches(name, regexec("^(flat|hl|m|win)([0-9.]*)(_x([0-9.]+))?$", name))[[1]]
  if (!length(parts)) stop(sprintf("weighting '%s': a name in config.R, or hl<days>, m<matches>, win<days>, flat, each optionally _x<scalar>", name))
  n <- as.numeric(parts[3])
  spec <- switch(parts[2],
    flat = list(type = "none"),
    hl = list(type = "exp_days", half_life = n),
    m = list(type = "exp_matches", half_life = n),
    win = list(type = "window", days = n))
  if (nzchar(parts[5])) spec$scalar <- as.numeric(parts[5])
  spec
}

parse_arm <- function(s) {
  p <- strsplit(s, ":", fixed = TRUE)[[1]]
  cadence <- if (length(p) > 1 && p[length(p)] %in% c("daily", "game")) p[length(p)] else "daily"
  if (length(p) > 1 && p[length(p)] %in% c("daily", "game")) p <- p[-length(p)]
  model <- p[1]
  weighting <- if (length(p) > 1) p[2] else "hl60"
  if (!model %in% c("full", names(MODELS))) stop(sprintf("arm '%s': model is one of full, %s", s, paste(names(MODELS), collapse = ", ")))
  if (model == "full" && cadence == "game") stop("the full model takes minutes a fit: daily only")
  weighting_spec(weighting)                      # fail early on a bad name
  list(label = paste(model, weighting, cadence, sep = ":"), model = model, weighting = weighting, cadence = cadence)
}
arm_specs <- lapply(split_list(args$arms), parse_arm)
arms <- vapply(arm_specs, `[[`, "", "label")
names(arm_specs) <- arms
if (anyDuplicated(arms)) stop("--arms: an arm twice")
lag <- as.numeric(args$result_lag) * 60
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

schedule <- function(cadence) {
  if (cadence == "game") {
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
  spec <- task$spec
  wspec <- weighting_spec(spec$weighting)
  formula <- if (spec$model == "full") NULL else as.formula(paste("Score ~", MODELS[[spec$model]]), env = globalenv())
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
    if (spec$model == "full") {
      g <- fit_global(train, FEATURE_SETS$form_clock, wspec, cut, start = start)
      fit <- g$fit
      mu <- exp(fit_eta(fit, rows) + (g$sigma_match^2 + g$sigma_obs^2) / 2)
    } else {
      train$.w <- row_weights(train, wspec, cut)
      train <- train[train$.w > 0, ]
      res <- fit_glmer(formula, train, start = if (is.null(start)) NULL else list(theta = start), nagq = trim_nagq)
      if (is.null(res$fit) && !is.null(start)) res <- fit_glmer(formula, train, nagq = trim_nagq)
      if (is.null(res$fit)) stop(sprintf("%s fit at %s failed: %s", spec$label, format(cut), paste(res$warnings, collapse = "; ")))
      fit <- res$fit
      mu <- exp(fit_eta(fit, rows))
    }
    secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
    start <- lme4::getME(fit, "theta")
    preds[[length(preds) + 1]] <- data.frame(arm = spec$label, MatchId = rows$MatchId, Side = rows$Side, mu = mu)
    fits[[length(fits) + 1]] <- data.frame(arm = spec$label, cut = format(cut, "%Y-%m-%d %H:%M:%S"),
                                           rows = nrow(train), seconds = round(secs, 2),
                                           theta = paste(sprintf("%s=%.4f", sub("\\.\\(Intercept\\)$", "", names(start)), start),
                                                         collapse = " "))
  }
  list(preds = do.call(rbind, preds), fits = do.call(rbind, fits))
}

cores <- if (nzchar(args$cores)) as.integer(args$cores) else max(1L, parallel::detectCores() - 1L)
# stretches of an arm's fits, each warm-starting from the fit before: a per-game arm in one stretch
# per core, a daily arm in fortnights
tasks <- list()
for (spec in arm_specs) {
  s <- schedule(spec$cadence)
  parts <- if (spec$cadence == "game") cores else ceiling(nrow(s) / 14)
  parts <- max(1L, min(parts, nrow(s)))
  for (idx in split(seq_len(nrow(s)), ceiling(seq_len(nrow(s)) * parts / nrow(s)))) {
    tasks[[length(tasks) + 1]] <- list(spec = spec, sched = s[idx, ])
  }
  log_line("%s: %d fits in %d stretch(es)", spec$label, nrow(s), parts)
}
# the slowest stretches first, so they don't finish last
cost <- vapply(tasks, function(t) if (t$spec$model == "full") 1e6 else nrow(t$sched) * (1 + (t$spec$model != "trim")), 0)
tasks <- tasks[order(-cost)]

t_all <- Sys.time()
results <- if (cores > 1 && length(tasks) > 1) {
  cl <- parallel::makeCluster(min(cores, length(tasks)), outfile = "")
  parallel::clusterExport(cl, c(".here"), envir = environment())
  parallel::clusterEvalQ(cl, { source(file.path(.here, "config.R")); source(file.path(.here, "common.R")); NULL })
  parallel::clusterExport(cl, c("long", "test", "lag", "MODELS", "weighting_spec", "trim_nagq", "run_chunk"),
                          envir = environment())
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
col1 <- function(arm) paste0(arm, "|1")
col2 <- function(arm) paste0(arm, "|2")
for (arm in arms) {
  p <- preds[preds$arm == arm, ]
  pk <- paste(p$MatchId, p$Side)
  home[[col1(arm)]] <- p$mu[match(paste(home$MatchId, "H"), pk)]
  home[[col2(arm)]] <- p$mu[match(paste(home$MatchId, "A"), pk)]
}
priced <- complete.cases(home[, c(col1(arms), col2(arms))])
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
  m <- home[[col1(arm)]] - home[[col2(arm)]]
  tt <- home[[col1(arm)]] + home[[col2(arm)]]
  p <- pmin(pmax(1 - pnorm(0, m, sd_m), 1e-9), 1 - 1e-9)
  data.frame(ll = ifelse(decided, -(won * log(p) + (1 - won) * log(1 - p)), NA),
             brier = ifelse(decided, (p - won)^2, NA),
             m_se = (margin - m)^2, t_se = (total - tt)^2, t_err = total - tt)
}
sc <- setNames(lapply(arms, scores), arms)
day <- format(home$Time, "%Y-%m-%d", tz = "UTC")
lw <- max(24, max(nchar(arms)) + 2)

lines <- c(sprintf("%d matches, %s to %s (%d decided); history %s", nrow(home), format(min(home$Time)),
                   format(max(home$Time)), sum(decided), basename(history_path)),
           sprintf("trim: %s (nAGQ %d)", args$trim, trim_nagq), "")

# each arm alone: its fits and its scores
lines <- c(lines, sprintf("  %-*s %6s %8s %9s %8s %8s %8s %8s %7s", lw, "arm", "fits", "s a fit", "min all",
                          "log loss", "Brier", "mrg RMSE", "tot RMSE", "tot err"))
for (arm in arms) {
  f <- fits[fits$arm == arm, ]
  s <- sc[[arm]]
  lines <- c(lines, sprintf("  %-*s %6d %8.1f %9.1f %8.4f %8.4f %8.3f %8.3f %+7.2f", lw, arm, nrow(f), mean(f$seconds),
                            sum(f$seconds) / 60, mean(s$ll, na.rm = TRUE), mean(s$brier, na.rm = TRUE),
                            sqrt(mean(s$m_se)), sqrt(mean(s$t_se)), mean(s$t_err)))
}

# the fitted spread of each random effect (the last fit's, on the log scale): how big a matchup term is
lines <- c(lines, "", "  random-effect sds, each arm's last fit (log scale):")
for (arm in arms) {
  f <- fits[fits$arm == arm, ]
  lines <- c(lines, sprintf("  %-*s %s", lw, arm, f$theta[nrow(f)]))
}

# paired, match by match: b less a, with its t (match-level se) and the days it was better
diff_cells <- function(a, b, keep = rep(TRUE, nrow(home))) {
  one <- function(x) {
    ok <- keep & !is.na(x)
    d <- x[ok]
    t <- mean(d) / (sd(d) / sqrt(length(d)))
    by_day <- tapply(d, day[ok], mean)
    c(mean = mean(d), t = t, better = sum(by_day < 0), days = length(by_day))
  }
  list(ll = one(sc[[b]]$ll - sc[[a]]$ll), brier = one(sc[[b]]$brier - sc[[a]]$brier),
       m = one(sc[[b]]$m_se - sc[[a]]$m_se), t = one(sc[[b]]$t_se - sc[[a]]$t_se), n = sum(keep))
}
diff_head <- sprintf("  %-*s %6s %20s %20s %16s %16s", lw, "", "matches", "log loss (t, days)", "Brier (t)",
                     "margin sq err (t)", "total sq err (t)")
diff_line <- function(label, d) {
  sprintf("  %-*s %6d %+9.5f (%+4.1f, %2d/%-2d) %+9.5f (%+4.1f) %+9.3f (%+4.1f) %+9.3f (%+4.1f)", lw, label, d$n,
          d$ll["mean"], d$ll["t"], d$ll["better"], d$ll["days"], d$brier["mean"], d$brier["t"],
          d$m["mean"], d$m["t"], d$t["mean"], d$t["t"])
}
base <- arms[1]
if (length(arms) > 1) {
  lines <- c(lines, "", sprintf("  each arm less the baseline (%s), match by match; days: days it was better on log loss", base),
             diff_head)
  for (arm in arms[-1]) lines <- c(lines, diff_line(arm, diff_cells(base, arm)))
}

# per-game refits against their daily twin, by how far into the day
bands <- list(`games 1-2` = c(1, 2), `games 3-5` = c(3, 5), `games 6-8` = c(6, 8), `games 9+` = c(9, Inf))
for (arm in arms[vapply(arm_specs, function(s) s$cadence == "game", TRUE)]) {
  twin <- sub(":game$", ":daily", arm)
  if (!twin %in% arms) next
  lines <- c(lines, "", sprintf("  %s less %s: refitting after every result, by the later gamer's game of the day", arm, twin),
             diff_head, diff_line("all", diff_cells(twin, arm)))
  for (b in names(bands)) {
    keep <- home$GameOfDay >= bands[[b]][1] & home$GameOfDay <= bands[[b]][2]
    if (sum(keep) >= 30) lines <- c(lines, diff_line(b, diff_cells(twin, arm, keep)))
  }
  dm <- (home[[col1(arm)]] - home[[col2(arm)]]) - (home[[col1(twin)]] - home[[col2(twin)]])
  dt <- (home[[col1(arm)]] + home[[col2(arm)]]) - (home[[col1(twin)]] + home[[col2(twin)]])
  lines <- c(lines, sprintf("  the refits moved the expected margin by %.2f points on average (95%% within %.2f), the total by %.2f (%.2f)",
                            mean(abs(dm)), quantile(abs(dm), 0.95), mean(abs(dt)), quantile(abs(dt), 0.95)))
}
writeLines(lines, file.path(out_dir, "cadence_report.txt"))
cat(paste(lines, collapse = "\n"), "\n")
log_line("-> %s", out_dir)
