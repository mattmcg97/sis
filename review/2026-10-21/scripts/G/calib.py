import pandas as pd, numpy as np
base="/home/user/sis/eAMFCalibrator/out/walkthrough/priors/%s.csv"
for k in ["nb2","glmer","avg-plain"]:
    d=pd.read_csv(base%k); d=d[~d.burn_in.astype(bool)] if False else d
    m=d.home_score-d.away_score; t=d.home_score+d.away_score
    em=d.exp_margin; et=d.exp_total
    bm=np.polyfit(em,m,1)[0]; bt=np.polyfit(et,t,1)[0]
    # logit slope of p_home
    from numpy.linalg import lstsq
    dec=m!=0; y=(m[dec]>0).astype(float); p=d.p_home[dec].clip(1e-4,1-1e-4); x=np.log(p/(1-p))
    # logistic regression via IRLS
    X=np.c_[np.ones(len(x)),x]; b=np.zeros(2)
    for _ in range(30):
        mu=1/(1+np.exp(-X@b)); W=mu*(1-mu); b=b+np.linalg.solve(X.T@(X*W[:,None]),X.T@(y-mu))
    print(f"{k:10s} n={len(d)} margin bias={np.mean(m-em):+.2f} slope={bm:.2f} | total bias={np.mean(t-et):+.2f} slope={bt:.2f} RMSE={np.sqrt(np.mean((t-et)**2)):.2f} | ML logit slope={b[1]:.3f} int={b[0]:+.3f} sd(p)={d.p_home.std():.3f}")
