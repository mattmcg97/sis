source("/home/user/sis/glmer/config.R"); source("/home/user/sis/glmer/common.R")
args <- commandArgs(TRUE)
hist <- if (length(args)) args[1] else "/home/user/sis/nb2/AMFELO.csv"
cutday <- if (length(args) > 1) args[2] else "2026-08-27"
ndays <- if (length(args) > 2) as.numeric(args[3]) else 14
m <- load_matches(hist); m <- m[!is.na(m$P1Score), ]
d <- to_long(m)
# Version B: form features that read only results in by (kick-off - delta); lag 36 min
lag <- 36*60; delta <- 10*60
alpha <- 1 - 2^(-1/FORM_HALF_LIFE_MATCHES)
tnum <- as.numeric(d$Time)
# league average as in to_long
first_at <- match(tnum, tnum); done <- !is.na(d$Score)
cs <- c(0, cumsum(ifelse(done, d$Score, 0))); cn <- c(0, cumsum(done))
league <- ifelse(cn[first_at] > 0, cs[first_at]/pmax(cn[first_at],1), mean(d$Score))
ffB <- faB <- numeric(nrow(d)); expB <- numeric(nrow(d))
for (idx in split(seq_len(nrow(d)), d$Player)) {
  idx <- idx[order(tnum[idx])]
  sf <- sa <- NA; sfA <- saA <- numeric(length(idx))  # state after each match
  for (k in seq_along(idx)) {
    i <- idx[k]
    if (is.na(sf)) { sf <- league[i]; sa <- league[i] }
    sf <- sf + alpha*(d$Score[i]-sf); sa <- sa + alpha*(d$OppScore[i]-sa)
    sfA[k] <- sf; saA[k] <- sa
  }
  for (k in seq_along(idx)) {
    i <- idx[k]
    avail <- which(tnum[idx[seq_len(k-1)]] + lag <= tnum[i] - delta)
    j <- if (length(avail)) max(avail) else 0
    if (j == 0) { ffB[i] <- 0; faB[i] <- 0 } else {
      ffB[i] <- log((sfA[j]+1)/(league[i]+1)); faB[i] <- log((saA[j]+1)/(league[i]+1)) }
    expB[i] <- j
  }
}
opp <- match(paste(d$MatchId, ifelse(d$Side=="H","A","H")), paste(d$MatchId, d$Side))
dB <- d; dB$FormFor <- ffB; dB$FormAgainst <- faB; dB$OppFormFor <- ffB[opp]; dB$OppFormAgainst <- faB[opp]
dB$ExpLog <- log1p(expB); dB$OppExpLog <- dB$ExpLog[opp]
cut <- as.POSIXct(cutday, tz="UTC"); until <- cut + ndays*86400
f <- as.formula(paste("Score ~ IsHome + FormFor + FormAgainst + OppFormFor + OppFormAgainst + ExpLog + OppExpLog +",
  "SessLate + OppSessLate + (1|Player) + (1|OpponentPlayer) + (1|Team) + (1|OpponentTeam)"), env=globalenv())
train <- d[d$Time + lag <= cut, ]
train$.w <- row_weights(train, list(type="exp_days", half_life=60, scalar=0.5), cut)
t0 <- Sys.time(); res <- fit_glmer(f, train, nagq=0); cat("fit secs", as.numeric(difftime(Sys.time(),t0,units="secs")), "\n")
fit <- res$fit; print(round(fixef(fit),4))
score <- function(dd) {
  te <- floor_experience(dd[dd$Time >= cut & dd$Time < until, ])
  mu <- exp(fit_eta(fit, te))
  h <- te$Side=="H"; key <- te$MatchId
  H <- data.frame(MatchId=key[h], muH=mu[h], S1=te$Score[h], S2=te$OppScore[h], Time=te$Time[h])
  A <- data.frame(MatchId=key[!h], muA=mu[!h]); H$muA <- A$muA[match(H$MatchId, A$MatchId)]
  H
}
A <- score(d); B <- score(dB)
B <- B[match(A$MatchId, B$MatchId),]
mg <- A$S1 - A$S2; dec <- mg != 0; y <- as.numeric(mg > 0)
ll <- function(H) { p <- pmin(pmax(pnorm((H$muH-H$muA)/9.35),1e-9),1-1e-9); -(y*log(p)+(1-y)*log(1-p)) }
la <- ll(A)[dec]; lb <- ll(B)[dec]; df <- la - lb
# was the previous match of either gamer within lag+delta before kick-off?
prevgap <- sapply(seq_len(nrow(A)), function(k) {
  rows <- d[d$MatchId == A$MatchId[k], ]
  g <- sapply(rows$Player, function(p) { tt <- tnum[d$Player==p]; tk <- as.numeric(A$Time[k]); e <- tt[tt < tk]; if (length(e)) (tk-max(e))/60 else Inf })
  min(g) })
near <- (prevgap < (lag+delta)/60)[dec]
cat(sprintf("matches %d decided %d; share with a previous result not in at kick-off-10min: %.3f\n", nrow(A), sum(dec), mean(near)))
cat(sprintf("log loss kick-off form %.5f vs as-of-quote form %.5f: diff %+.5f (t %.2f)\n", mean(la), mean(lb), mean(df), mean(df)/(sd(df)/sqrt(length(df)))))
cat(sprintf("  among affected matches: diff %+.5f (n %d, t %.2f)\n", mean(df[near]), sum(near), mean(df[near])/(sd(df[near])/sqrt(sum(near)))))
mA <- A$muH-A$muA; mB <- B$muH-B$muA
cat(sprintf("margin RMSE A %.3f B %.3f; mean |margin change| on affected %.3f\n", sqrt(mean((mg-mA)^2)), sqrt(mean((mg-mB)^2)), mean(abs(mA-mB)[prevgap < (lag+delta)/60])))
