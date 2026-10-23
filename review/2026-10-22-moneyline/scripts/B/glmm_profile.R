# Profile the Laplace deviance of the gamer-only difference GLMM over the gamer sd on a few cut days.
suppressMessages(library(lme4))
D <- read.csv("../in.csv", stringsAsFactors = FALSE)
D$start <- as.POSIXct(D$start, tz = "UTC"); D$settled <- D$start + 36 * 60; D$stf <- factor(D$st, levels = 0:2)
NG <- 89
diffZt <- function(a, b, nl) { n <- length(a)
  sparseMatrix(i = c(a, b), j = c(seq_len(n), seq_len(n)), x = c(rep(1, n), rep(-1, n)), dims = c(nl, n)) }
for (dd in c("2026-09-09", "2026-09-29", "2026-10-09")) {
  cut <- as.POSIXct(paste(dd, "00:00:00"), tz = "UTC")
  tr <- D[D$settled <= cut & !is.na(D$y), ]
  w <- 0.5^(as.numeric(difftime(max(tr$start), tr$start, units = "days")) / 30); w <- w * sum(w) / sum(w^2)
  tr$w <- w; tr$gdum <- factor(rep_len(seq_len(NG), nrow(tr)), levels = seq_len(NG))
  lf <- suppressWarnings(glFormula(y ~ 0 + stf + (1 | gdum), data = tr, family = binomial, weights = w))
  lf$reTrms$Zt <- as(diffZt(tr$g1 + 1, tr$g2 + 1, NG), "CsparseMatrix")
  devf <- suppressWarnings(do.call(mkGlmerDevfun, lf))
  o0 <- optimizeGlmer(devf)
  devf2 <- updateGlmerDevfun(devf, lf$reTrms)
  o2 <- suppressWarnings(optimizeGlmer(devf2, stage = 2))
  cat(dd, "n", nrow(tr), "ESS", round(sum(w)), "stage1 theta", round(o0$par, 3), "stage2 theta", round(o2$par[1], 3), "dev", round(o2$fval, 2), "\n")
  for (th in c(0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8)) {
    v <- devf2(c(th, o2$par[-1]))
    cat("   theta", th, "deviance", round(v, 2), "\n")
  }
  # how many distinct gamers carry weight in the window
  ww <- tapply(c(tr$w, tr$w), c(tr$g1, tr$g2), sum)
  cat("   gamers with weight > 20:", sum(ww > 20), "\n")
}
