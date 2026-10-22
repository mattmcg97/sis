import csv, sys, collections, numpy as np
f=sys.argv[1]; cs=sys.argv[2].split(','); pairs=[p.split(':') for p in sys.argv[3].split(',')]
inplay=sys.argv[4] if len(sys.argv)>4 else "all"
mkt=sys.argv[5] if len(sys.argv)>5 else "all"
per=collections.defaultdict(lambda: np.zeros(2+len(cs)))
for row in csv.DictReader(open(f)):
    if row['excluded']=='True' or row['BET_CASHED_OUT']=='Yes': continue
    if row['result'] not in ('won','lost','push'): continue
    if inplay!="all" and row["in_play"]!=inplay: continue
    if mkt!="all" and row["market"]!=mkt: continue
    if any(row.get(f'candidate_revenue_{c}') in ('',None) for c in cs): continue
    v=per[row['match_code']]; v[0]+=float(row['stake']); v[1]+=float(row['revenue'])
    for i,c in enumerate(cs): v[2+i]+=float(row[f'candidate_revenue_{c}'])
M=np.array(list(per.values())); n=len(M)
rng=np.random.default_rng(0); W=rng.multinomial(n,[1/n]*n,size=2000).astype(float)
def chg(i,w=None):
    S=M if w is None else None
    st=M[:,0]; return None
tot=M.sum(0)
print('matches',n,'stake',round(tot[0]))
for i,c in enumerate(cs):
    print(c,'change %+.2f'%(100*(tot[2+i]-tot[1])/tot[0]))
for a,b in pairs:
    ia,ib=cs.index(a),cs.index(b)
    d=100*(M[:,2+ia]-M[:,2+ib])
    est=d.sum()/tot[0]
    bs=(W@d)/(W@M[:,0])
    print('%s - %s: %+.2f [%+.2f, %+.2f]'%(a,b,est,np.percentile(bs,2.5),np.percentile(bs,97.5)))
