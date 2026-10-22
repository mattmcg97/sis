import csv, sys, datetime as dt, collections
hist=list(csv.DictReader(open('eAMFCalibrator/out/match_history.csv',encoding='utf-8')))
games=collections.defaultdict(list); start={}
for r in hist:
    t=dt.datetime.fromisoformat(r['SCHEDULED_START_TIME_UTC'][:19]); start[r['MATCH_CODE']]=t
    for k in (1,2): games[r[f'PLAYER_{k}_HANDLE'].strip().upper()].append((t,r['MATCH_CODE']))
prevstart={}
for g,ms in games.items():
    ms=sorted(set(ms))
    for (s0,_),(s1,c1) in zip(ms,ms[1:]):
        if s1>s0: prevstart[c1]=max(prevstart.get(c1,s0),s0)
st_all=0; buckets=collections.Counter(); mins=collections.Counter()
for r in csv.DictReader(open(sys.argv[1])):
    if r['in_play']!='False' or r['excluded']=='True': continue
    c=r['match_code']; bt=dt.datetime.fromisoformat(r['bet_time'][:19]); s=float(r['stake']); st_all+=s
    if c in start: mins[min(60,int((start[c]-bt).total_seconds()//300)*5)]+=s
    if c in prevstart:
        m=(bt-prevstart[c]).total_seconds()/60
        for L in (25,28,30,33,36):
            if m<L: buckets[L]+=s
print('pre-match stake',round(st_all))
for L in sorted(buckets): print('placed < %d min after a gamer\'s previous kick-off: %.0f%%'%(L,100*buckets[L]/st_all))
print('stake by minutes before scheduled start (5-min bins):',{k:'%.0f%%'%(100*v/st_all) for k,v in sorted(mins.items())})
