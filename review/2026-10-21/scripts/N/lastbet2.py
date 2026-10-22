import csv, sys, datetime as dt, collections
hist=list(csv.DictReader(open('eAMFCalibrator/out/match_history.csv',encoding='utf-8')))
games=collections.defaultdict(list); start={}
for r in hist:
    t=dt.datetime.fromisoformat(r['SCHEDULED_START_TIME_UTC'][:19]); start[r['MATCH_CODE']]=t
    for k in (1,2): games[r[f'PLAYER_{k}_HANDLE'].strip().upper()].append((t,r['MATCH_CODE']))
LAG=dt.timedelta(minutes=28); prev={}
for g,ms in games.items():
    ms=sorted(set(ms))
    for (s0,_),(s1,c1) in zip(ms,ms[1:]):
        if s1>s0: prev[c1]=max(prev.get(c1,s0+LAG),s0+LAG)
last={}
for r in csv.DictReader(open(sys.argv[1])):
    if r['in_play']!='False' or r['market']!='moneyline' or not r['stream_prob']: continue
    bt=dt.datetime.fromisoformat(r['bet_time'][:19]); c=r['match_code']
    if c not in last or bt>last[c]: last[c]=bt
n=len(last); unc=sum(1 for c,bt in last.items() if c in prev and bt<prev[c])
mins=sorted((start[c]-bt).total_seconds()/60 for c,bt in last.items() if c in start)
print('matches',n,'last pre-match ML bet before a previous result was in:',unc, '%.0f%%'%(100*unc/n))
print('minutes before scheduled start of last pre-match ML bet: median %.1f, p25 %.1f p75 %.1f'%(mins[len(mins)//2],mins[len(mins)//4],mins[3*len(mins)//4]))
