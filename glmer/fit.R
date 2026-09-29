#!/usr/bin/env Rscript
#
# Fit the glmer pre-match model for real: the global model plus every
# qualifying player's attack and defence models, on every settled match
# before --before (default: all of them), with one feature set and one
# weighting (pick them with backtest.R). Saves the lot for predict.R.
#
# Usage (from the repo root):
#   Rscript glmer/fit.R --feature-set=home --weighting=hl60 --mode=pair
#   Rscript glmer/fit.R --before=2026-09-24 --out=glmer/out/model_0924
#
# In RStudio: set the defaults just below and Source the file.
#
# Writes (to --out, default glmer/out/model/):
#   model.rds            everything predict.R needs
#   player_ratings.csv   per player: global attack / defence effects, and their own models' summaries
#   effects.csv          every random effect of the global model (teams, streams, ...)
#   fit_summary.txt      formulas, variance components, fixed effects, warnings

DEFAULTS <- list(
  history = "",             # "" = first of HISTORY_CANDIDATES in config.R
  out = "",                 # "" = glmer/out/model
  feature_set = "home",
  weighting = "hl60",
  mode = "pair",            # predict.R's default mode for this model
  before = "",              # "YYYY-MM-DD"; "" = fit on every settled match
  min_matches = "",         # "" = MIN_PLAYER_MATCHES
  nagq = "",                # "" = GLMER_NAGQ
  optimizer = "",           # "" = GLMER_OPTIMIZER (nloptwrap or bobyqa)
  cores = ""                # for the per-player fits; "" = all but one
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
out_dir <- if (nzchar(args$out)) args$out else file.path(.here, "out", "model")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
if (nzchar(args$min_matches)) MIN_PLAYER_MATCHES <- as.integer(args$min_matches)
if (nzchar(args$nagq)) GLMER_NAGQ <- as.integer(args$nagq)
if (nzchar(args$optimizer)) GLMER_OPTIMIZER <- args$optimizer
if (!args$feature_set %in% names(FEATURE_SETS)) stop(sprintf("no feature set '%s' in config.R", args$feature_set))
if (!args$weighting %in% names(WEIGHTINGS)) stop(sprintf("no weighting '%s' in config.R", args$weighting))
if (!args$mode %in% PREDICT_MODES) stop(sprintf("mode must be one of %s", paste(PREDICT_MODES, collapse = ", ")))
fs <- FEATURE_SETS[[args$feature_set]]
wspec <- WEIGHTINGS[[args$weighting]]

history_path <- resolve_history(args$history, repo_root)
log_line("history: %s", history_path)
matches <- load_matches(history_path)
settled <- matches[!is.na(matches$P1Score), ]
as_of <- if (nzchar(args$before)) {
  as.POSIXct(args$before, tz = "UTC")
} else {
  max(settled$Time) + 1
}
long <- to_long(matches)
train <- long[!is.na(long$Score) & long$Time < as_of, ]
log_line("fitting %s / %s on %d settled matches before %s (%s -> %s)", args$feature_set, args$weighting,
         nrow(train) / 2, format(as_of), format(min(train$Time)), format(max(train$Time)))

cores <- if (nzchar(args$cores)) as.integer(args$cores) else max(1, parallel::detectCores() - 1)
cl <- NULL
if (cores > 1) {
  cl <- parallel::makeCluster(cores)
  invisible(parallel::clusterCall(cl, function(dir, nagq, optimizer) {
    source(file.path(dir, "config.R"))
    source(file.path(dir, "common.R"))
    GLMER_NAGQ <<- nagq
    GLMER_OPTIMIZER <<- optimizer
    NULL
  }, .here, GLMER_NAGQ, GLMER_OPTIMIZER))
}
bundle <- tryCatch(
  fit_bundle(train, fs, wspec, as_of, cluster = cl, fs_name = args$feature_set,
             w_name = args$weighting),
  finally = if (!is.null(cl)) parallel::stopCluster(cl))
bundle$default_mode <- args$mode
bundle$history_path <- history_path
saveRDS(bundle, file.path(out_dir, "model.rds"))
log_line("saved %s (%.0f MB)", file.path(out_dir, "model.rds"),
         file.size(file.path(out_dir, "model.rds")) / 1e6)

# ---------------------------------------------------------------------------
# Readable outputs
# ---------------------------------------------------------------------------

gfit <- bundle$global$fit
re <- ranef(gfit)
effects <- do.call(rbind, lapply(names(re), function(g) {
  x <- re[[g]]
  if (!"(Intercept)" %in% names(x) || g %in% c("ObsID", "MatchId")) return(NULL)
  data.frame(Group = g, Level = rownames(x), Effect = x[["(Intercept)"]],
             Multiplier = exp(x[["(Intercept)"]]), stringsAsFactors = FALSE)
}))
effects <- effects[order(effects$Group, -effects$Effect), ]
write.csv(effects, file.path(out_dir, "effects.csv"), row.names = FALSE)

players <- sort(unique(c(train$Player, train$OpponentPlayer)))
eff <- function(group, lvl) {
  x <- re[[group]]
  if (is.null(x)) return(rep(NA_real_, length(lvl)))
  x[["(Intercept)"]][match(lvl, rownames(x))]
}
own <- function(p, role, what) {
  m <- bundle$players[[p]][[role]]
  if (is.null(m)) return(NA_real_)
  switch(what, n = m$n, sd = m$sigma_obs,
         intercept = unname(fixef(m$fit)["(Intercept)"]),
         warnings = length(m$warnings))
}
ratings <- data.frame(
  Player = players,
  Matches = vapply(players, function(p) sum(train$Player == p), numeric(1)),
  LastMatch = vapply(players, function(p) format(max(train$Time[train$Player == p])), ""),
  # Global model: + scores more than average, - concedes less than average.
  GlobalAttack = eff("Player", players),
  GlobalDefence = eff("OpponentPlayer", players),
  AttackModelRows = vapply(players, own, numeric(1), role = "attack", what = "n"),
  AttackIntercept = vapply(players, own, numeric(1), role = "attack", what = "intercept"),
  AttackSdObs = vapply(players, own, numeric(1), role = "attack", what = "sd"),
  DefenceModelRows = vapply(players, own, numeric(1), role = "defence", what = "n"),
  DefenceIntercept = vapply(players, own, numeric(1), role = "defence", what = "intercept"),
  DefenceSdObs = vapply(players, own, numeric(1), role = "defence", what = "sd"),
  stringsAsFactors = FALSE)
ratings$NetRating <- ratings$GlobalAttack - ratings$GlobalDefence
ratings <- ratings[order(-ratings$NetRating), ]
write.csv(ratings, file.path(out_dir, "player_ratings.csv"), row.names = FALSE)

sink(file.path(out_dir, "fit_summary.txt"))
cat(sprintf("glmer pre-match model -- feature set %s, weighting %s, default mode %s\n",
            args$feature_set, args$weighting, args$mode))
cat(sprintf("history %s\nfitted on %d settled matches before %s; fitted %s\n\n", history_path,
            nrow(train) / 2, format(as_of), format(bundle$fitted_at)))
cat("Weighting:", paste(names(wspec), unlist(wspec), sep = "=", collapse = ", "), "\n\n")
cat("GLOBAL MODEL\n", bundle$global$formula, "\n\n", sep = "")
print(summary(gfit)$varcor)
cat("\nFixed effects:\n")
print(round(summary(gfit)$coefficients, 4))
if (length(bundle$global$warnings)) cat("\nWarnings:\n", paste(" -", bundle$global$warnings, collapse = "\n"), "\n")
cat(sprintf("\nPER-PLAYER MODELS (>= %d matches): %d attack, %d defence; %d fits failed\n",
            bundle$min_matches, sum(!is.na(ratings$AttackModelRows)), sum(!is.na(ratings$DefenceModelRows)),
            bundle$failed_player_fits))
ex <- bundle$players[[1]]$attack %||% bundle$players[[1]]$defence
if (!is.null(ex)) cat("e.g.", names(bundle$players)[1], ":", ex$formula, "\n")
nw <- sum(unlist(lapply(bundle$players, function(p) vapply(p, function(m) length(m$warnings) > 0, logical(1)))))
cat(sprintf("%d per-player fits carried warnings (usually singular fits: a variance component at 0)\n", nw))
sink()

cat("\n")
print(summary(gfit)$varcor)
cat("\nTop and bottom of the global ratings (log scale; attack + = scores more, defence + = concedes more):\n")
show <- ratings[, c("Player", "Matches", "GlobalAttack", "GlobalDefence", "NetRating")]
show[3:5] <- lapply(show[3:5], round, 3)
print(head(show, 10), row.names = FALSE)
cat("...\n")
print(tail(show, 5), row.names = FALSE)
cat(sprintf("\nWritten to %s: model.rds, player_ratings.csv, effects.csv, fit_summary.txt\n", out_dir))
