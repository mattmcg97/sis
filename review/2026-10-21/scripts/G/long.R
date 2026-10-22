source("/home/user/sis/glmer/config.R"); source("/home/user/sis/glmer/common.R")
m <- load_matches("/home/user/sis/nb2/AMFELO.csv")
cat("matches", nrow(m), "settled", sum(!is.na(m$P1Score)), "\n")
t0 <- Sys.time()
d <- to_long(m[!is.na(m$P1Score),])
cat("to_long secs", as.numeric(difftime(Sys.time(), t0, units="secs")), "\n")
saveRDS(d, "long.rds")
mon <- format(d$Time, "%Y-%m")
print(round(cbind(league=tapply(log(d$Score+1) , mon, function(x) NA),
  meanScore=tapply(d$Score, mon, mean),
  FormFor=tapply(d$FormFor, mon, mean), FormAgainst=tapply(d$FormAgainst, mon, mean),
  ExpLog=tapply(d$ExpLog, mon, mean), SessLate=tapply(d$SessLate, mon, mean)),3))
