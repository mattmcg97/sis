import pandas as pd, numpy as np
for wt in ["walkthrough","walkthrough_ho"]:
  for k in ["nb2","glmer","avg","avg-plain","nb2_infer"]:
    d=pd.read_csv(f"/home/user/sis/eAMFCalibrator/out/{wt}/priors/{k}.csv")
    m=d.home_score-d.away_score; e=m-d.exp_margin
    dec=m!=0; y=(m[dec]>0).astype(float); p=d.p_home[dec].clip(1e-6,1-1e-6)
    ll=-(y*np.log(p)+(1-y)*np.log(1-p)).mean()
    # best constant logit shift
    best=min(((-(y*np.log(q:=1/(1+np.exp(-(np.log(p/(1-p))+s))))+(1-y)*np.log(1-q)).mean()),s) for s in np.arange(-0.15,0.151,0.005))
    print(f"{wt:15s} {k:10s} margin bias {e.mean():+.3f} (se {e.std()/np.sqrt(len(e)):.3f}) home win {y.mean():.4f} vs pred {p.mean():.4f}  ll {ll:.5f} best shift {best[1]:+.3f} -> {best[0]:.5f}")
