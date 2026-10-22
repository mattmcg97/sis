source("/home/user/sis/glmer/config.R"); source("/home/user/sis/glmer/common.R")
m <- load_matches("/home/user/sis/eAMFCalibrator/out/match_history.csv"); m <- m[!is.na(m$P1Score),]
d <- to_long(m)
spec <- WEIGHTINGS$hl60
for (asof in c("2026-04-01","2026-07-01","2026-10-01","2026-10-19")) {
  a <- as.POSIXct(asof, tz="UTC"); tr <- d[d$Time < a,]
  w <- row_weights(tr, spec, a)
  ess <- sum(w)^2/sum(w^2)
  cat(sprintf("as_of %s rows %d days %.0f  weight of a match that day %.2f  ESS/n %.2f  -> scalar to count data at Kish weight %.2f\n",
      asof, nrow(tr), as.numeric(difftime(a, min(tr$Time), units="days")), max(w), ess/nrow(tr), ess/sum(w)))
}
