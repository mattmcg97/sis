# Binomial GLMM, difference design, walk-forward daily (or every k days) refits with lme4's modular functions.
# logit P(home win) = stream home + (u[g1] - u[g2]) [+ (t[k1] - t[k2])] [+ (v[g1,k1] - v[g2,k2])],
# u ~ N(0, s_g^2), t ~ N(0, s_t^2), v ~ N(0, s_gt^2): variances by Laplace marginal likelihood (glmer).
# Usage: Rscript glmm_wf.R <in.csv> <out.csv> <start> <end> <terms: g|gt|gtx> <train_days|0> <half_life|0> <refit_days>
suppressMessages(library(lme4))
args <- commandArgs(TRUE)
inf <- args[1]; outf <- args[2]; d0 <- args[3]; d1 <- args[4]; terms <- args[5]
train_days <- as.numeric(args[6]); hl <- as.numeric(args[7]); refit <- as.integer(args[8])
D <- read.csv(inf, stringsAsFactors = FALSE)
D$start <- as.POSIXct(D$start, tz = "UTC")
D$settled <- D$start + 36 * 60
D$stf <- factor(D$st, levels = 0:2)
days <- sort(unique(D$day[D$day >= d0 & D$day < d1]))
NG <- 89; NT <- 10

diffZt <- function(a, b, nl) {   # nl x n sparse: +1 at level a, -1 at level b (1-based levels)
  n <- length(a)
  sparseMatrix(i = c(a, b), j = c(seq_len(n), seq_len(n)), x = c(rep(1, n), rep(-1, n)), dims = c(nl, n))
}

fit_one <- function(tr) {
  n <- nrow(tr)
  tr$gdum <- factor(rep_len(seq_len(NG), n), levels = seq_len(NG))
  tr$tdum <- factor(rep_len(seq_len(NT), n), levels = seq_len(NT))
  tr$vdum <- factor(rep_len(seq_len(NG * NT), n), levels = seq_len(NG * NT))
  form <- switch(terms,
    g = y ~ 0 + stf + (1 | gdum),
    gt = y ~ 0 + stf + (1 | gdum) + (1 | tdum),
    gtx = y ~ 0 + stf + (1 | gdum) + (1 | tdum) + (1 | vdum))
  if (length(unique(tr$st)) < 3) form <- update(form, . ~ . - stf + 1)
  w <- if (hl > 0) 0.5^(as.numeric(difftime(max(tr$start), tr$start, units = "days")) / hl) else rep(1, n)
  w <- w * sum(w) / sum(w^2)   # scaled to sum to the Kish effective sample size
  tr$w <- w
  lf <- suppressWarnings(glFormula(form, data = tr, family = binomial, weights = w,
                                   control = glmerControl(check.nlev.gtreq.5 = "ignore", check.nobs.vs.nRE = "ignore",
                                                          check.nobs.vs.nlev = "ignore", check.nlev.gtr.1 = "ignore")))
  rt <- lf$reTrms
  # replace each term's indicator block with the difference block (terms in rt$cnms order)
  blocks <- list()
  for (nm in names(rt$cnms)) {
    blocks[[nm]] <- switch(nm,
      gdum = diffZt(tr$g1 + 1, tr$g2 + 1, NG),
      tdum = diffZt(tr$k1 + 1, tr$k2 + 1, NT),
      vdum = diffZt(tr$g1 * NT + tr$k1 + 1, tr$g2 * NT + tr$k2 + 1, NG * NT))
  }
  Zt <- do.call(rbind, blocks)
  rt$Zt <- as(Zt, "CsparseMatrix")
  rt$Ztlist <- NULL
  lf$reTrms <- rt
  devf <- suppressWarnings(do.call(mkGlmerDevfun, lf))
  opt <- optimizeGlmer(devf)
  devf <- updateGlmerDevfun(devf, lf$reTrms)
  opt <- suppressWarnings(optimizeGlmer(devf, stage = 2))
  m <- suppressWarnings(mkMerMod(environment(devf), opt, lf$reTrms, fr = lf$fr))
  b <- getME(m, "b")[, 1]
  Gp <- getME(m, "Gp")
  re <- list()
  for (k in seq_along(names(rt$cnms))) re[[names(rt$cnms)[k]]] <- b[(Gp[k] + 1):Gp[k + 1]]
  fe <- fixef(m)
  th <- getME(m, "theta"); names(th) <- names(rt$cnms)
  list(fe = fe, re = re, theta = th)
}

pred_one <- function(M, te) {
  fe <- M$fe
  if (all(c("stf0", "stf1", "stf2") %in% names(fe))) z <- fe[paste0("stf", te$st)] else z <- rep(fe[1], nrow(te))
  re <- M$re
  if (!is.null(re$gdum)) z <- z + re$gdum[te$g1 + 1] - re$gdum[te$g2 + 1]
  if (!is.null(re$tdum)) z <- z + re$tdum[te$k1 + 1] - re$tdum[te$k2 + 1]
  if (!is.null(re$vdum)) z <- z + re$vdum[te$g1 * NT + te$k1 + 1] - re$vdum[te$g2 * NT + te$k2 + 1]
  1 / (1 + exp(-z))
}

res <- list(); thetas <- list(); M <- NULL
for (k in seq_along(days)) {
  dd <- days[k]
  if (is.null(M) || (k - 1) %% refit == 0) {
    cut <- as.POSIXct(paste(dd, "00:00:00"), tz = "UTC")
    tr <- D[D$settled <= cut & !is.na(D$y), ]
    if (train_days > 0) tr <- tr[tr$start >= cut - train_days * 86400, ]
    t0 <- Sys.time()
    M <- fit_one(tr)
    thetas[[dd]] <- data.frame(day = dd, n = nrow(tr), t(M$theta), secs = as.numeric(Sys.time() - t0, units = "secs"))
    cat(dd, nrow(tr), paste(names(M$theta), round(M$theta, 4)), round(M$fe, 3), round(as.numeric(Sys.time() - t0, units = "secs"), 1), "\n")
  }
  te <- D[D$day == dd, ]
  res[[dd]] <- data.frame(code = te$code, p = pred_one(M, te))
}
write.csv(do.call(rbind, res), outf, row.names = FALSE)
th <- do.call(rbind, lapply(thetas, function(x) { names(x)[3:(ncol(x) - 1)] <- paste0("theta", 1:(ncol(x) - 3)); x }))
write.csv(th, sub(".csv$", "_theta.csv", outf), row.names = FALSE)
