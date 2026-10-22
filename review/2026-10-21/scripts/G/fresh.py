import pandas as pd, numpy as np
base="/home/user/sis/eAMFCalibrator/out/walkthrough/priors/%s.csv"
d={k:pd.read_csv(base%k).set_index("MATCH_CODE") for k in ["nb2","glmer","avg","avg-plain","nb2_infer","glmer_both","nb2_both"]}
idx=d["nb2"].index
for k in d: idx=idx.intersection(d[k].index)
N=d["nb2"].loc[idx]
for since in ["2026-09-01","2026-09-10"]:
    keep=(N.start>=since)&(~N.burn_in.astype(bool))&(N.home_score!=N.away_score)
    y=(N.home_score>N.away_score).astype(float)[keep]
    def ll(k):
        p=d[k].loc[idx].p_home[keep].clip(1e-6,1-1e-6); return -(y*np.log(p)+(1-y)*np.log(1-p))
    base_ll=ll("nb2")
    print(since, "n", keep.sum(), "nb2 ll", round(base_ll.mean(),5))
    for k in ["glmer","glmer_both","nb2_infer","nb2_both","avg-plain","avg"]:
        df=ll(k)-base_ll; print(f"   {k:10s} {df.mean():+.5f} (t {df.mean()/(df.std()/np.sqrt(len(df))):+.2f})")
