import csv, sys, datetime as dt, collections
hist=list(csv.DictReader(open('eAMFCalibrator/out/match_history.csv',encoding='utf-8')))
games=collections.defaultdict(list); start={}
for r in hist:
    t=dt.datetime.fromisoformat(r['SCHEDULED_START_TIME_UTC'][:19]); start[r['MATCH_CODE']]=t
    for k in (1,2): games[r[f'PLAYER_{k}_HANDLE'].strip().upper()].append((t,r['MATCH_CODE']))
LAG=dt.timedelta(minutes=int(sys.argv[3]))
prev={}
for g,ms in games.items():
    ms=sorted(set(ms))
    for (s0,_),(s1,c1) in zip(ms,ms[1:]):
        if s1>s0 and (s1-s0)<=dt.timedelta(hours=3):
            prev[c1]=max(prev.get(c1,s0+LAG),s0+LAG)
f=sys.argv[1]; cands=sys.argv[2].split(',')
agg=collections.defaultdict(lambda: collections.defaultdict(float))
for row in csv.DictReader(open(f)):
    if row['in_play']!='False' or row['excluded']=='True': continue
    if row['BET_CASHED_OUT']=='Yes': continue
    code=row['match_code']
    bt=dt.datetime.fromisoformat(row['bet_time'][:19])
    grp='clean' if (code not in prev or bt>=prev[code]) else 'unclean'
    # minutes before kickoff
    if not all(row.get(f'candidate_revenue_{c}') not in ('',None) for c in cands): continue
    a=agg[(grp,row['market'])]; b=agg[(grp,'all')]
    for x in (a,b):
        x['n']+=1; x['stake']+=float(row['stake']); x['rev']+=float(row['revenue'])
        for c in cands: x[c]+=float(row[f'candidate_revenue_{c}'])
for k in sorted(agg):
    x=agg[k]
    print(k, int(x['n']), round(x['stake']), 'prod %.2f%%'%(100*x['rev']/x['stake']), ' '.join('%s %+.2f'%(c,100*(x[c]-x['rev'])/x['stake']) for c in cands))
