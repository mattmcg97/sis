import csv, sys, collections
f=sys.argv[1]; c=sys.argv[2]
agg=collections.defaultdict(lambda: collections.defaultdict(float))
for row in csv.DictReader(open(f)):
    if row['excluded']=='True' or row['BET_CASHED_OUT']=='Yes': continue
    rc=row.get(f'candidate_revenue_{c}')
    if rc in ('',None): continue
    mk=row['market']
    try:
        bl=float(row['bet_line']) if row['bet_line'] else None
        cl=float(row[f'candidate_line_{c}']) if row[f'candidate_line_{c}'] else None
    except ValueError: bl=cl=None
    same=row[f'same_line_{c}']=='True'
    if mk=='moneyline': mv='ml'
    elif same or bl is None or cl is None: mv='same'
    else:
        d=cl-bl
        mv='moved'
    sel=row['selection']
    k=(mk,sel,mv,row['in_play'])
    x=agg[k]; st=float(row['stake'])
    x['n']+=1; x['stake']+=st; x['rev']+=float(row['revenue']); x['crev']+=float(rc)
    if mv=='moved': x['dline']+=st*(cl-bl) if mk=='total' else st*(cl-bl)
    # expected under prod: only same-line
    try:
        p=float(row['stream_prob']); oc=float(row[f'candidate_odds_{c}']); op=float(row['odds'])
        if mv in ('same','ml'):
            x['exp_prod']+=st*(1-p*op); x['exp_cand']+=st*(1-p*oc); x['m']+=st
    except (ValueError,KeyError): pass
tot=collections.defaultdict(float)
for k in sorted(agg):
    x=agg[k]
    s='%-40s n=%6d stake=%9.0f prod %+6.2f%% chg %+6.2f' % (k, x['n'], x['stake'], 100*x['rev']/x['stake'], 100*(x['crev']-x['rev'])/x['stake'])
    if x['dline']: s+=' mean dline %+.2f' % (x['dline']/x['stake'])
    if x['m']: s+=' expchg(prod truth) %+.2f' % (100*(x['exp_cand']-x['exp_prod'])/x['m'])
    print(s)
