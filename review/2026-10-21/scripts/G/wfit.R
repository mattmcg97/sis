source("/home/user/sis/glmer/config.R"); source("/home/user/sis/glmer/common.R")
m <- load_matches("/home/user/sis/eAMFCalibrator/out/match_history.csv"); m <- m[!is.na(m$P1Score),]
d <- to_long(m)
f <- as.formula(paste("Score ~ IsHome + FormFor + FormAgainst + OppFormFor + OppFormAgainst + ExpLog + OppExpLog +",
  "SessLate + OppSessLate + (1|Player) + (1|OpponentPlayer) + (1|Team) + (1|OpponentTeam)"), env=globalenv())
a <- as.POSIXct("2026-10-01", tz="UTC"); lag <- 36*60
te <- floor_experience(d[d$Time >= a & d$Time < as.POSIXct("2026-10-19", tz="UTC"),])
res <- list()
for (from in c("2025-12-01","2026-04-01")) {
  tr <- d[d$Time + lag <= a & d$Time >= as.POSIXct(from, tz="UTC"),]
  tr$.w <- row_weights(tr, list(type="exp_days", half_life=60, scalar=0.5), a)
  t0 <- Sys.time(); fit <- fit_glmer(f, tr, nagq=0)$fit
  vc <- as.data.frame(VarCorr(fit))
  mu <- exp(fit_eta(fit, te)); h <- te$Side=="H"
  H <- data.frame(id=te$MatchId[h], mh=mu[h], s1=te$Score[h], s2=te$OppScore[h]); A <- data.frame(id=te$MatchId[!h], ma=mu[!h])
  H$ma <- A$ma[match(H$id, A$id)]
  em <- H$mh-H$ma; mg <- H$s1-H$s2; dec <- mg!=0; y <- as.numeric(mg>0)
  p <- pnorm(em/9.35); ll <- -(y*log(p)+(1-y)*log(1-p))
  cat(sprintf("history from %s: rows %d, max weight %.2f, %0.fs | sd Player %.3f Opp %.3f Team %.3f OppTeam %.3f | sd(pred margin) %.3f | Oct 1-18 ll %.5f, margin RMSE %.3f\n",
      from, nrow(tr), max(tr$.w), as.numeric(difftime(Sys.time(),t0,units="secs")),
      vc$sdcor[vc$grp=="Player"], vc$sdcor[vc$grp=="OpponentPlayer"], vc$sdcor[vc$grp=="Team"], vc$sdcor[vc$grp=="OpponentTeam"],
      sd(em), mean(ll[dec]), sqrt(mean((mg-em)^2))))
  res[[from]] <- list(em=em, ll=ll, dec=dec)
}
dl <- res[[1]]$ll - res[[2]]$ll; dec <- res[[1]]$dec
cat(sprintf("ll full-history minus truncated: %+.5f (t %.2f); mean |margin diff| %.3f\n", mean(dl[dec]), mean(dl[dec])/(sd(dl[dec])/sqrt(sum(dec))), mean(abs(res[[1]]$em-res[[2]]$em))))
