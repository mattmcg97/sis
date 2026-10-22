import pandas as pd, numpy as np
for wt in ["walkthrough","walkthrough_ho"]:
  for k in ["nb2","avg"]:
    d=pd.read_csv(f"/home/user/sis/eAMFCalibrator/out/{wt}/priors/{k}.csv")
    d=d[d.score_model=="nb2"].copy()
    p=d.score_params.str.split(" ",expand=True).astype(float); a1,a2,sh=p[0],p[1],p[2]
    mu1,mu2=d.Pred_P1_Points,d.Pred_P2_Points
    v1=mu1+a1*mu1**2; v2=mu2+a2*mu2**2; cov=sh*mu1*mu2*np.minimum(a1,a2)
    tv=v1+v2+2*cov; mv=v1+v2-2*cov
    t=d.home_score+d.away_score; m=d.home_score-d.away_score
    te=t-(mu1+mu2); me=m-(mu1-mu2)
    # PIT-ish: share of totals beyond +-1 predicted sd
    z=te/np.sqrt(tv); zm=me/np.sqrt(mv)
    print(f"{wt:15s} {k:5s} n={len(d)} pred total sd={np.sqrt(tv.mean()):.2f} realized total rmse={np.sqrt((te**2).mean()):.2f} var ratio={np.mean(te**2)/tv.mean():.2f} | pred margin sd={np.sqrt(mv.mean()):.2f} realized={np.sqrt((me**2).mean()):.2f} ratio={np.mean(me**2)/mv.mean():.2f} | |z|>1.645 total {np.mean(abs(z)>1.645):.3f} margin {np.mean(abs(zm)>1.645):.3f}")
