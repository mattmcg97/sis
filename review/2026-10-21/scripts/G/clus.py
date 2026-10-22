import pandas as pd, numpy as np, sys
base="/home/user/sis/eAMFCalibrator/out/%s/priors/%s.csv"
for wt in ["walkthrough","walkthrough_ho"]:
    d={k:pd.read_csv(base%(wt,k)) for k in ["nb2","glmer","avg","avg-plain","nb2_infer","glmer_both"]}
    g=d["glmer"]; print(wt, len(g), g.start.min(), g.start.max(), "burn_in", g.burn_in.sum())
    for a,b in [("nb2","glmer"),("nb2","avg"),("glmer","avg"),("nb2","avg-plain"),("nb2_infer","avg")]:
        A=d[a].set_index("MATCH_CODE"); B=d[b].set_index("MATCH_CODE")
        idx=A.index.intersection(B.index)
        A=A.loc[idx]; B=B.loc[idx]
        keep=(~A.burn_in.astype(bool)) & (A.home_score!=A.away_score)
        A=A[keep]; B=B[keep]
        y=(A.home_score>A.away_score).astype(float)
        def ll(p): p=p.clip(1e-6,1-1e-6); return -(y*np.log(p)+(1-y)*np.log(1-p))
        diff=ll(B.p_home)-ll(A.p_home)
        n=len(diff); t=diff.mean()/(diff.std()/np.sqrt(n))
        day=A.start.str[:10]
        def clustered(groups):
            # cluster-robust SE of mean
            e=diff-diff.mean()
            s=e.groupby(groups.values).sum()
            G=len(s); se=np.sqrt((s**2).sum())/n*np.sqrt(G/(G-1))
            return diff.mean()/se, G
        tday,Gd=clustered(day)
        # gamer-day: cluster on day only is coarser; also gamer clusters (two-way not easy) -> home gamer
        tg,Gg=clustered(A.home+"|"+day)
        print(f"  {b:10s} - {a:10s} n={n} mean={diff.mean():+.5f} t_match={t:+.2f} t_day={tday:+.2f} (G={Gd}) t_homegamerday={tg:+.2f}")
