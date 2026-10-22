import csv, sys, collections
f=sys.argv[1]; cs=sys.argv[2].split(',')
agg=collections.defaultdict(lambda: collections.defaultdict(float))
for row in csv.DictReader(open(f)):
    if row['excluded']=='True' or row['BET_CASHED_OUT']=='Yes': continue
    if any(row.get(f'candidate_revenue_{c}') in ('',None) for c in cs): continue
    mk=row['market']; c0=cs[0]
    mv='ml' if mk=='moneyline' else ('same' if row[f'same_line_{c0}']=='True' else 'moved')
    for k in (mv,'all'):
        x=agg[k]; x['stake']+=float(row['stake']); x['rev']+=float(row['revenue']); x['n']+=1
        for c in cs: x[c]+=float(row[f'candidate_revenue_{c}'])-float(row['revenue'])
T=agg['all']
for k,x in agg.items():
    print(k, int(x['n']), round(x['stake']), ' '.join('%s chg %+.0f (%+.2f pts of all stake)'%(c,x[c],100*x[c]/T['stake']) for c in cs))
