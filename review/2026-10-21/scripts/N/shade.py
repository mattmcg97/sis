import csv, sys, collections
f=sys.argv[1]; cs=sys.argv[2].split(','); mkt=sys.argv[3]
agg=collections.defaultdict(lambda: collections.defaultdict(float))
for row in csv.DictReader(open(f)):
    if row['excluded']=='True' or row['BET_CASHED_OUT']=='Yes' or row['market']!=mkt: continue
    if row['result'] not in ('won','lost'): continue
    if any(row.get(f'candidate_revenue_{c}') in ('',None) for c in cs): continue
    if mkt!='moneyline' and any(row.get(f'same_line_{c}')!='True' for c in cs): continue
    try: p=float(row['stream_prob']); op=float(row['odds'])
    except ValueError: continue
    st=float(row['stake']); ip=row['in_play']
    x=agg[ip]; x['stake']+=st; x['rev']+=float(row['revenue'])
    for c in cs:
        q=float(row[f'candidate_prob_{c}']); oc=float(row[f'candidate_odds_{c}'])
        x[c+'_real']+=float(row[f'candidate_revenue_{c}'])-float(row['revenue'])
        x[c+'_eprod']+=st*p*(op-oc)          # E[change] if prod's prob is the truth
        x[c+'_ecand']+=st*q*(op-oc)          # E[change] if the candidate's prob is the truth
        x[c+'_dq']+=st*(q-p)
for ip,x in agg.items():
    print('in_play',ip,'stake',round(x['stake']))
    for c in cs:
        print('   %-12s realized %+.2f  E|prod true %+.2f  E|cand true %+.2f   mean(q-p) on backed side %+.4f'%(c,100*x[c+'_real']/x['stake'],100*x[c+'_eprod']/x['stake'],100*x[c+'_ecand']/x['stake'],x[c+'_dq']/x['stake']))
