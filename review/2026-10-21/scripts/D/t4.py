import os, sys, numpy as np
sys.path.insert(0, "/home/user/sis")
D = os.path.dirname(os.path.abspath(__file__))
from eAMFModel import table_prior, v21_report, markets, books
from eAMFCalibrator import score_model
p = os.path.join(D, "prior_normal.csv")
with open(p, "w") as fh:
    fh.write("MATCH_CODE,home_pts,away_pts,score_model,score_params\n")
    fh.write("AF1,20,17,normal,\n")
    fh.write("AF2,20,17,nb2,0.10 0.12 0.30\n")
t = table_prior.TablePrior(p)
print("score kept:", t.score)
rows = [dict(MATCH_CODE="AF1", SCHEDULED_START_TIME_UTC="2026-10-02 10:00:00"),
        dict(MATCH_CODE="AF2", SCHEDULED_START_TIME_UTC="2026-10-02 10:00:00")]
means, bk = v21_report.prior_inputs(p, rows, [])
for c in ("AF1", "AF2"):
    mp, tp = bk[c]
    k = np.arange(len(mp)) - markets.MARGIN_MAX
    mu = (k * mp).sum(); sd = np.sqrt(((k - mu) ** 2 * mp).sum())
    kt = np.arange(len(tp)); tsd = np.sqrt(((kt - (kt * tp).sum()) ** 2 * tp).sum())
    print(c, "graded margin sd %.2f total sd %.2f tie %.3f" % (sd, tsd, mp[markets.MARGIN_MAX]))
print("walkthrough's own normal for AF1: margin sd", score_model.MARGIN_SD, "total sd", score_model.TOTAL_SD)
