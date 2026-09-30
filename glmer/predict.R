#!/usr/bin/env Rscript
#
# Price matches pre-match with a model saved by fit.R: each side's expected
# points (what eAMFModel's v4 / v5 / v6 take from NB2 as their pre-match
# prior), the home moneyline, total and spread, and which model priced each
# side (a player's own model, or the global one when they have too little
# data or aren't known at all).
#
# --schedule is a CSV shaped like the history (MATCH_CODE, STREAM_NUMBER,
# SCHEDULED_START_TIME_UTC, PLAYER_1_HANDLE, PLAYER_1_TEAM, PLAYER_2_HANDLE,
# PLAYER_2_TEAM; final scores optional). Without it, every history match
# from the model's cut-off on is priced -- an out-of-sample check of a model
# fitted with fit.R --before. The history is also needed for the form /
# rest / session features, which look back at each player's earlier matches.
#
# Usage (from the repo root):
#   Rscript glmer/predict.R --schedule=schedule.csv
#   Rscript glmer/predict.R --model=glmer/out/model_0924/model.rds --mode=global
#
# In RStudio: set the defaults just below and Source the file.
#
# Writes --out (default glmer/out/predictions.csv). Pred_P1_Points,
# Pred_P2_Points and Prediction_Status match NB2_schedule_predict.py's
# output columns.

DEFAULTS <- list(
  model = "",               # "" = glmer/out/model/model.rds
  schedule = "",            # "" = history matches from the model's cut-off on
  history = "",             # "" = the history the model was fitted on
  mode = "",                # "" = the model's default (fit.R --mode)
  out = "",                 # "" = glmer/out/predictions.csv
  n_sims = ""               # "" = N_SIMS; 0 = expected points only (no moneyline)
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
if (nzchar(args$n_sims)) N_SIMS <- as.integer(args$n_sims)
model_path <- if (nzchar(args$model)) args$model else file.path(.here, "out", "model", "model.rds")
if (!file.exists(model_path)) stop(sprintf("no model at %s: run fit.R first", model_path))
bundle <- readRDS(model_path)
mode <- if (nzchar(args$mode)) args$mode else (bundle$default_mode %||% "pair")
if (!mode %in% PREDICT_MODES) stop(sprintf("mode must be one of %s", paste(PREDICT_MODES, collapse = ", ")))
log_line("model %s: %s / %s, fitted before %s, mode %s", model_path, bundle$feature_set,
         bundle$weighting, format(bundle$as_of), mode)

history_path <- if (nzchar(args$history)) {
  resolve_history(args$history, repo_root)
} else if (!is.null(bundle$history_path) && file.exists(bundle$history_path)) {
  bundle$history_path
} else {
  resolve_history("", repo_root)
}
history <- load_matches(history_path)

if (nzchar(args$schedule)) {
  schedule <- load_matches(args$schedule)
  early <- sum(schedule$Time < bundle$as_of)
  if (early) {
    log_line("note: %d scheduled match(es) start before the model's cut-off -- priced in-sample", early)
  }
} else {
  schedule <- history[history$Time >= bundle$as_of, ]
  if (!nrow(schedule)) stop("no --schedule, and no history matches after the model's cut-off to price")
  log_line("no --schedule: pricing the %d history matches from %s on", nrow(schedule), format(bundle$as_of))
}

# Features are built over history + schedule together so every scheduled
# match sees each player's earlier matches, using the model's own settings.
FORM_HALF_LIFE_MATCHES <- bundle$features$form_half_life
SESSION_HOURS <- bundle$features$session_hours
REST_CAP_HOURS <- bundle$features$rest_cap_hours
long <- to_long(combine_matches(history, schedule))
rows <- long[long$MatchId %in% schedule$MATCH_CODE, ]

src <- predict_sources(bundle, rows)
pred <- price_matches(bundle, src, mode)
sch <- schedule[match(pred$MATCH_CODE, schedule$MATCH_CODE), ]
known <- rownames(ranef(bundle$global$fit)$Player)
out <- data.frame(
  MATCH_CODE = pred$MATCH_CODE, SCHEDULED_START_TIME_UTC = format(sch$Time, "%Y-%m-%d %H:%M:%S"),
  STREAM_NUMBER = sch$Stream, PLAYER_1_HANDLE = sch$P1, PLAYER_1_TEAM = sch$P1Team,
  PLAYER_2_HANDLE = sch$P2, PLAYER_2_TEAM = sch$P2Team,
  Pred_P1_Points = round(pred$Pred_P1_Points, 4), Pred_P2_Points = round(pred$Pred_P2_Points, 4),
  Pred_Total = round(pred$Pred_Total, 4), Pred_Spread = round(pred$Pred_Spread, 4),
  P1_Win_Prob = round(pred$P1_Win_Prob, 4),
  P1_Source = pred$P1_Source, P2_Source = pred$P2_Source,
  P1_Known = sch$P1 %in% known, P2_Known = sch$P2 %in% known,
  Prediction_Status = "OK", stringsAsFactors = FALSE)
if (any(!is.na(sch$P1Score))) {
  out$PLAYER_1_FINAL_SCORE <- sch$P1Score
  out$PLAYER_2_FINAL_SCORE <- sch$P2Score
}
out <- out[order(out$SCHEDULED_START_TIME_UTC, out$MATCH_CODE), ]

out_path <- if (nzchar(args$out)) args$out else file.path(.here, "out", "predictions.csv")
dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write.csv(out, out_path, row.names = FALSE)

capped <- sum(src$capped_attack) + sum(src$capped_defence)
if (capped) {
  log_line("note: %d per-player prediction(s) hit PLAYER_MAX_SHIFT and were held to within %.2f of the global model",
           capped, PLAYER_MAX_SHIFT)
}
srcs <- table(c(out$P1_Source, out$P2_Source))
log_line("priced %d matches; sides by source: %s; %d side(s) with a player the model has never seen",
         nrow(out), paste(names(srcs), srcs, sep = " ", collapse = ", "),
         sum(!out$P1_Known) + sum(!out$P2_Known))
if (!is.null(out$PLAYER_1_FINAL_SCORE)) {
  settled <- !is.na(out$PLAYER_1_FINAL_SCORE)
  if (any(settled)) {
    p <- pred[match(out$MATCH_CODE[settled], pred$MATCH_CODE), ]
    p$P1Score <- out$PLAYER_1_FINAL_SCORE[settled]
    p$P2Score <- out$PLAYER_2_FINAL_SCORE[settled]
    cat("\nAgainst the results already in:\n")
    sc <- score_predictions(p)
    if (N_SIMS <= 0) sc <- sc[setdiff(names(sc), c("Brier", "LogLoss"))]
    print(round(sc, 4), row.names = FALSE)
  }
}
cat(sprintf("\nWritten to %s\n", out_path))
