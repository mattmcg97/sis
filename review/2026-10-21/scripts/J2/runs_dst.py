"""bet_runs.records: naive-UTC bet times turned into epoch seconds with .timestamp(), which reads a
naive datetime as LOCAL time. On a machine in a DST zone, the hour of the clock change breaks the
120 s windows."""
import datetime as dt, os, time
from eAMFCalibrator import bet_runs as br, bets, config
print("TZ", os.environ.get("TZ"), time.tzname)
COL = "CUSTOMER_ID"
import sys; T = dt.datetime.fromisoformat(sys.argv[1])
def b(i, c, secs, sel):
    return bets.Bet(bet_id=i, match_code="M1", time=T + dt.timedelta(seconds=secs), market_type=1, selection=sel,
                    odds=2.0, stake=10.0, revenue=0.0,
                    extra={config.BET_IN_PLAY_COLUMN: "Yes", COL: c, config.BET_GROUP_COLUMN: "FD"})
every = [b("a", "X", 0, 1), b("b", "Y", 20, 1), b("c", "Z", 40, 1), b("d", "W", 60, 1)]
recs = br.follow(br.records(every, [{}] * 4, {}, COL), after=120)
for x in recs:
    print(x["customer"], dt.datetime.utcfromtimestamp(x["t"]), "t-diff to first", x["t"] - recs[0]["t"],
          "after", x["after"], "followers", x["followers"])
