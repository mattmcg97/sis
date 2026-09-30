#!/usr/bin/env Rscript
#
# Backtest the glmer pre-match model over a grid of feature sets x row
# weightings: fit the global model and every player's attack/defence models
# on the matches before the test period, price the test matches every
# prediction mode (global only, per-player attack, per-player defence, both),
# and score them out of sample.
#
# By default it is one chronological split (first 75% of matches train, the
# rest test, the same split as nb2/backtest_nb2_calibration.py, so the Brier
# / log loss / totals numbers line up with NB2's). With --refit-days N it
# walks forward instead: refit every N days through the test period, each
# fit seeing only matches before its window -- slower, closer to production.
#
# Usage (from the repo root):
#   Rscript glmer/backtest.R
#   Rscript glmer/backtest.R --feature-sets=home,form --weightings=hl30,hl60 --cores=6
#   Rscript glmer/backtest.R --test-from=2026-07-01 --refit-days=14
#   Rscript glmer/backtest.R --feature-sets=form --weightings=hl30,hl60,hl120 \
#       --form-half-lives=5,10,20 --players=FALSE --refit-days=14   # half-life sweep
#   Rscript glmer/backtest.R --feature-sets=form --weightings=hl60 --scalars=0.5,1,2 \
#       --players=FALSE --refit-days=14                               # weight-scalar sweep
#
# In RStudio: set the defaults just below and Source the file.
#
# Writes (to --out, default glmer/out/backtest/):
#   summary.csv      one row per feature set x weighting x mode x subset
#   calibration.csv  moneyline calibration in probability deciles, per configuration
#   predictions.csv.gz  every priced test match, every configuration (gzipped)
#   by_player.csv    per-player errors, and how each mode compares with global
#   fits.csv         what each fit saw: rows, every variance component, warnings, time

DEFAULTS <- list(
  history = "",             # "" = first of HISTORY_CANDIDATES in config.R
  out = "",                 # "" = glmer/out/backtest
  feature_sets = "",        # comma-separated names from config.R; "" = all
  weightings = "",          # comma-separated names from config.R; "" = all
  modes = "",               # "" = PREDICT_MODES
  test_from = "",           # "YYYY-MM-DD"; "" = split by train_fraction
  test_until = "",          # "YYYY-MM-DD"; "" = end of history
  train_fraction = "0.75",
  refit_days = "0",         # 0 = one fit at the start of the test period
  form_half_lives = "",     # comma-separated FormFor / FormAgainst half-lives in matches;
                            #   "" = FORM_HALF_LIFE_MATCHES. More than one adds "_f<n>" to the
                            #   feature set's name in the outputs.
  players = "TRUE",         # FALSE = global model only (no per-player fits; mode global)
  scalars = "",             # comma-separated weight scalars, each applied to every weighting
                            #   (in place of its own); "" = each weighting's own. The outputs
                            #   call them "<weighting>_x<scalar>".
  min_matches = "",         # "" = MIN_PLAYER_MATCHES
  nagq = "",                # "" = GLMER_NAGQ
  optimizer = "",           # "" = GLMER_OPTIMIZER (nloptwrap or bobyqa)
  n_sims = "",              # "" = N_SIMS
  cores = ""                # "" = all but one
)

.here <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  if (length(f)) return(dirname(normalizePath(f[1])))
  for (i in rev(seq_len(sys.nframe()))) {
    o <- sys.frame(i)$ofile
    if (!is.null(o)) return(dirname(normalizePath(o)))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getSourceEditorContext()$path
    if (nzchar(p)) return(dirname(normalizePath(p)))
  }
  if (file.exists("common.R")) return(normalizePath("."))
  if (file.exists("glmer/common.R")) return(normalizePath("glmer"))
  stop("can't find the glmer/ folder: run from the repo root or setwd() to glmer/")
})
source(file.path(.here, "config.R"))
source(file.path(.here, "common.R"))

args <- parse_args(DEFAULTS)
repo_root <- dirname(.here)
out_dir <- if (nzchar(args$out)) args$out else file.path(.here, "out", "backtest")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (nzchar(args$min_matches)) MIN_PLAYER_MATCHES <- as.integer(args$min_matches)
if (nzchar(args$nagq)) GLMER_NAGQ <- as.integer(args$nagq)
if (nzchar(args$optimizer)) GLMER_OPTIMIZER <- args$optimizer
if (nzchar(args$n_sims)) N_SIMS <- as.integer(args$n_sims)
fs_names <- if (nzchar(args$feature_sets)) split_list(args$feature_sets) else names(FEATURE_SETS)
w_names <- if (nzchar(args$weightings)) split_list(args$weightings) else names(WEIGHTINGS)
scalars <- if (nzchar(args$scalars)) as.numeric(split_list(args$scalars)) else NA_real_
modes <- if (nzchar(args$modes)) split_list(args$modes) else PREDICT_MODES
with_players <- as_flag(args$players)
if (!with_players) modes <- "global"
form_hls <- if (nzchar(args$form_half_lives)) as.numeric(split_list(args$form_half_lives)) else FORM_HALF_LIFE_MATCHES
bad <- c(setdiff(fs_names, names(FEATURE_SETS)), setdiff(w_names, names(WEIGHTINGS)),
         setdiff(modes, PREDICT_MODES))
if (length(bad)) stop(sprintf("not in config.R: %s", paste(bad, collapse = ", ")))

# ---------------------------------------------------------------------------
# Data and folds
# ---------------------------------------------------------------------------

history_path <- resolve_history(args$history, repo_root)
log_line("history: %s", history_path)
matches <- load_matches(history_path)
matches <- matches[!is.na(matches$P1Score), ]
log_line("%d settled matches, %s -> %s, %d players, %d teams", nrow(matches),
         format(min(matches$Time)), format(max(matches$Time)),
         length(unique(c(matches$P1, matches$P2))), length(unique(c(matches$P1Team, matches$P2Team))))
# The long data once per form half-life: form is the only feature that
# depends on it.
longs <- setNames(lapply(form_hls, function(h) to_long(matches, form_half_life = h)),
                  as.character(form_hls))

test_from <- if (nzchar(args$test_from)) {
  as.POSIXct(args$test_from, tz = "UTC")
} else {
  matches$Time[floor(nrow(matches) * as.numeric(args$train_fraction)) + 1]
}
test_until <- if (nzchar(args$test_until)) as.POSIXct(args$test_until, tz = "UTC") else max(matches$Time) + 1
refit_days <- as.numeric(args$refit_days)
starts <- if (refit_days > 0) seq(test_from, test_until, by = refit_days * 86400) else test_from
starts <- starts[starts < test_until]
folds <- data.frame(fold = seq_along(starts), start = starts,
                    end = c(starts[-1], test_until))
test_n <- sum(matches$Time >= test_from & matches$Time < test_until)
log_line("train before %s (%d matches); test %s -> %s (%d matches) in %d fold(s)",
         format(test_from), sum(matches$Time < test_from), format(test_from),
         format(test_until), test_n, nrow(folds))

jobs <- expand.grid(fold = folds$fold, form_hl = form_hls, scalar = scalars, weighting = w_names,
                    feature_set = fs_names, stringsAsFactors = FALSE)
jobs <- jobs[, c("feature_set", "form_hl", "weighting", "scalar", "fold")]
jobs$label <- if (length(form_hls) > 1) sprintf("%s_f%g", jobs$feature_set, jobs$form_hl) else jobs$feature_set
jobs$wlabel <- if (all(is.na(scalars))) jobs$weighting else sprintf("%s_x%g", jobs$weighting, jobs$scalar)
log_line("%d feature sets x %d form half-lives x %d weightings x %d scalars x %d folds = %d fits (each: global%s)",
         length(fs_names), length(form_hls), length(w_names), length(scalars), nrow(folds), nrow(jobs),
         if (with_players) " + per-player" else " only")

# ---------------------------------------------------------------------------
# One job: fit before the fold, price the fold
# ---------------------------------------------------------------------------

run_job <- function(j) {
  job <- jobs[j, ]
  fold <- folds[folds$fold == job$fold, ]
  fs <- FEATURE_SETS[[job$feature_set]]
  wspec <- WEIGHTINGS[[job$weighting]]
  if (!is.na(job$scalar)) wspec$scalar <- job$scalar
  t0 <- Sys.time()
  long <- longs[[as.character(job$form_hl)]]
  train <- long[!is.na(long$Score) & long$Time < fold$start, ]
  test <- long[long$Time >= fold$start & long$Time < fold$end, ]
  res <- tryCatch({
    b <- fit_bundle(train, fs, wspec, fold$start, verbose = FALSE,
                    fs_name = job$label, w_name = job$weighting,
                    with_players = with_players, form_half_life = job$form_hl)
    src <- predict_sources(b, test)
    # A test match is "seen" when both players and both teams appear in
    # this fold's training rows -- NB2's backtest only prices those.
    h <- test[test$Side == "H", ]
    seen <- h$Player %in% train$Player & h$OpponentPlayer %in% train$Player &
            h$Team %in% train$Team & h$OpponentTeam %in% train$Team
    preds <- do.call(rbind, lapply(modes, function(mode) {
      p <- price_matches(b, src, mode)
      p$Seen <- seen[match(p$MATCH_CODE, h$MatchId)]
      cbind(FeatureSet = job$label, Weighting = job$wlabel, Mode = mode,
            Fold = job$fold, p, stringsAsFactors = FALSE)
    }))
    info <- data.frame(
      FeatureSet = job$label, Weighting = job$wlabel, Fold = job$fold,
      FoldStart = format(fold$start), GlobalRows = b$global$n,
      GlobalFormula = b$global$formula,
      SigmaMatch = b$global$sigma_match, SigmaObs = b$global$sigma_obs,
      VarComp = with(as.data.frame(VarCorr(b$global$fit)),
                     paste(sprintf("%s %.4f", grp, sdcor)[is.na(var2)], collapse = "; ")),
      GlobalWarnings = paste(b$global$warnings, collapse = " | "),
      AttackModels = sum(vapply(b$players, function(p) !is.null(p$attack), logical(1))),
      DefenceModels = sum(vapply(b$players, function(p) !is.null(p$defence), logical(1))),
      FailedPlayerFits = b$failed_player_fits,
      PlayerFitsWithWarnings = sum(unlist(lapply(b$players, function(p)
        vapply(p, function(m) length(m$warnings) > 0, logical(1))))),
      CappedPlayerPreds = sum(src$capped_attack) + sum(src$capped_defence),
      Seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs"))),
      Error = "", stringsAsFactors = FALSE)
    list(preds = preds, info = info)
  }, error = function(e) {
    list(preds = NULL, info = data.frame(
      FeatureSet = job$label, Weighting = job$wlabel, Fold = job$fold,
      FoldStart = format(fold$start), GlobalRows = NA, GlobalFormula = NA, SigmaMatch = NA,
      SigmaObs = NA, VarComp = NA, GlobalWarnings = NA, AttackModels = NA, DefenceModels = NA,
      FailedPlayerFits = NA, PlayerFitsWithWarnings = NA, CappedPlayerPreds = NA,
      Seconds = round(as.numeric(difftime(Sys.time(), t0, units = "secs"))),
      Error = conditionMessage(e), stringsAsFactors = FALSE))
  })
  cat(sprintf("[%s] done %s / %s / fold %d in %ss%s\n", format(Sys.time(), "%H:%M:%S"),
              job$label, job$wlabel, job$fold, res$info$Seconds,
              if (nzchar(res$info$Error)) paste(" -- FAILED:", res$info$Error) else ""))
  flush.console()
  res
}

cores <- if (nzchar(args$cores)) as.integer(args$cores) else max(1, parallel::detectCores() - 1)
cores <- max(1, min(cores, nrow(jobs)))
log_line("running on %d core(s)%s", cores,
         if (cores > 1) " -- per-job progress lines appear as each finishes" else "")
t_all <- Sys.time()
if (cores > 1) {
  cl <- parallel::makeCluster(cores, outfile = "")
  results <- tryCatch({
    parallel::clusterCall(cl, function(dir) {
      source(file.path(dir, "config.R"))
      source(file.path(dir, "common.R"))
      NULL
    }, .here)
    parallel::clusterExport(cl, c("longs", "jobs", "folds", "modes", "with_players", "run_job",
                                  "MIN_PLAYER_MATCHES", "GLMER_NAGQ", "GLMER_OPTIMIZER", "N_SIMS"))
    parallel::parLapplyLB(cl, seq_len(nrow(jobs)), function(j) run_job(j))
  }, finally = parallel::stopCluster(cl))
} else {
  results <- lapply(seq_len(nrow(jobs)), run_job)
}
log_line("all fits done in %.1f min", as.numeric(difftime(Sys.time(), t_all, units = "mins")))

# ---------------------------------------------------------------------------
# Collect and score
# ---------------------------------------------------------------------------

fits <- do.call(rbind, lapply(results, `[[`, "info"))
write.csv(fits, file.path(out_dir, "fits.csv"), row.names = FALSE)
if (any(nzchar(fits$Error))) {
  cat("\nFAILED fits:\n")
  print(fits[nzchar(fits$Error), c("FeatureSet", "Weighting", "Fold", "Error")], row.names = FALSE)
}
preds <- do.call(rbind, lapply(results, `[[`, "preds"))
if (is.null(preds) || !nrow(preds)) stop("no predictions: every fit failed (see fits.csv)")

mm <- matches[match(preds$MATCH_CODE, matches$MATCH_CODE), ]
preds$Time <- format(mm$Time)
preds$P1 <- mm$P1
preds$P2 <- mm$P2
preds$P1Team <- mm$P1Team
preds$P2Team <- mm$P2Team
preds$P1Score <- mm$P1Score
preds$P2Score <- mm$P2Score
gz <- gzfile(file.path(out_dir, "predictions.csv.gz"), "w")
write.csv(preds, gz, row.names = FALSE)
close(gz)

cfg <- paste(preds$FeatureSet, preds$Weighting, preds$Mode, sep = "\r")
summary_rows <- list()
for (rows in split(seq_len(nrow(preds)), cfg)) {
  p <- preds[rows, ]
  for (subset in c("all", "seen")) {
    q <- if (subset == "all") p else p[p$Seen, ]
    if (!nrow(q)) next
    own <- mean(c(q$P1_Source, q$P2_Source) != "global")
    summary_rows[[length(summary_rows) + 1]] <- cbind(
      FeatureSet = p$FeatureSet[1], Weighting = p$Weighting[1], Mode = p$Mode[1],
      Subset = subset, OwnModelShare = own, score_predictions(q), stringsAsFactors = FALSE)
  }
}
summary_df <- do.call(rbind, summary_rows)
summary_df <- summary_df[order(summary_df$Subset, summary_df$LogLoss), ]
write.csv(summary_df, file.path(out_dir, "summary.csv"), row.names = FALSE)

calibration <- do.call(rbind, lapply(split(seq_len(nrow(preds)), cfg), function(rows) {
  p <- preds[rows, ]
  cbind(FeatureSet = p$FeatureSet[1], Weighting = p$Weighting[1], Mode = p$Mode[1],
        calibration_table(p), stringsAsFactors = FALSE)
}))
write.csv(calibration, file.path(out_dir, "calibration.csv"), row.names = FALSE)

# Per player: errors on the points they scored and conceded, per mode, and
# the change against the same fit's global-only prediction.
side_key <- function(player, role) paste(preds$FeatureSet, preds$Weighting, preds$Mode, player, role, sep = "\r")
key <- c(side_key(preds$P1, "scored"), side_key(preds$P2, "scored"),
         side_key(preds$P2, "conceded"), side_key(preds$P1, "conceded"))
err <- c(preds$P1Score - preds$Pred_P1_Points, preds$P2Score - preds$Pred_P2_Points,
         preds$P1Score - preds$Pred_P1_Points, preds$P2Score - preds$Pred_P2_Points)
sums <- rowsum(cbind(N = 1, Err = err, SqErr = err^2), key)
parts <- do.call(rbind, strsplit(rownames(sums), "\r", fixed = TRUE))
by_player <- data.frame(FeatureSet = parts[, 1], Weighting = parts[, 2], Mode = parts[, 3],
                        Player = parts[, 4], Role = parts[, 5], N = sums[, "N"],
                        RMSE = sqrt(sums[, "SqErr"] / sums[, "N"]), Bias = sums[, "Err"] / sums[, "N"],
                        stringsAsFactors = FALSE)
g <- by_player[by_player$Mode == "global", ]
key_g <- paste(g$FeatureSet, g$Weighting, g$Player, g$Role)
by_player$RMSE_vs_global <- by_player$RMSE -
  g$RMSE[match(paste(by_player$FeatureSet, by_player$Weighting, by_player$Player, by_player$Role), key_g)]
by_player <- by_player[order(by_player$FeatureSet, by_player$Weighting, by_player$Player,
                             by_player$Role, by_player$Mode), ]
rownames(by_player) <- NULL
write.csv(by_player, file.path(out_dir, "by_player.csv"), row.names = FALSE)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

options(width = 200)
show <- function(df) {
  num <- vapply(df, is.numeric, logical(1))
  df[num] <- lapply(df[num], function(x) round(x, 4))
  print(df, row.names = FALSE)
}
cols <- c("FeatureSet", "Weighting", "Mode", "N", "OwnModelShare", "Brier", "LogLoss",
          "TotalBias", "TotalRMSE", "TotalCorr", "MarginRMSE", "SideRMSE")
all_rows <- summary_df[summary_df$Subset == "all", ]
cat("\n=== Out of sample, every test match, best 25 by log loss ===\n")
show(head(all_rows[, cols], 25))
cat("\n=== Best configuration for each mode ===\n")
show(do.call(rbind, lapply(split(all_rows, all_rows$Mode), function(x) x[1, cols])))
cat("\n=== Same, only matches whose players and teams were all in training (NB2's backtest subset) ===\n")
seen_rows <- summary_df[summary_df$Subset == "seen", ]
show(do.call(rbind, lapply(split(seen_rows, seen_rows$Mode), function(x) x[1, cols])))

best <- all_rows[1, ]
cat(sprintf("\n=== Moneyline calibration, %s / %s / %s (gap = realised - predicted, points) ===\n",
            best$FeatureSet, best$Weighting, best$Mode))
show(calibration[calibration$FeatureSet == best$FeatureSet & calibration$Weighting == best$Weighting &
                 calibration$Mode == best$Mode, c("Bucket", "N", "Predicted", "Realized", "GapPP")])

bp <- by_player[by_player$FeatureSet == best$FeatureSet & by_player$Weighting == best$Weighting &
                by_player$Mode != "global" & by_player$N >= 30, ]
if (nrow(bp)) {
  cat(sprintf("\n=== %s / %s: per-player RMSE change against global (negative = own model helps) ===\n",
              best$FeatureSet, best$Weighting))
  wide <- reshape(bp[, c("Player", "Role", "Mode", "RMSE_vs_global")], idvar = c("Player", "Role"),
                  timevar = "Mode", direction = "wide")
  names(wide) <- sub("RMSE_vs_global.", "", names(wide), fixed = TRUE)
  cat("players where each mode helps / hurts, on points scored and conceded:\n")
  for (m in setdiff(names(wide), c("Player", "Role"))) {
    x <- wide[[m]]
    cat(sprintf("  %-8s helps %3d  hurts %3d  median %+.3f\n", m, sum(x < 0, na.rm = TRUE),
                sum(x > 0, na.rm = TRUE), median(x, na.rm = TRUE)))
  }
}

cat(sprintf("\n(Brier 0.25 = always 50%%. Lower Brier / LogLoss / RMSE is better. OwnModelShare = share\n"))
cat(sprintf(" of sides priced by a per-player model rather than the global one.)\n"))
cat(sprintf("\nWritten to %s: summary.csv, calibration.csv, predictions.csv.gz, by_player.csv, fits.csv\n", out_dir))
