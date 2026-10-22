import csv
a={}
for r in csv.DictReader(open(f'{"%s"}/form_ho/bets_sim.csv' % "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/N")):
    if r['in_play']=='False' and r['market']=='moneyline' and r['candidate_prob_v21-avgf4']:
        a[r['bet_id']]=(r['match_code'],float(r['candidate_prob_v21-avgf4']),r['candidate_prob_v21-avgf4x'])
n=same=0; diffs=[]
for r in csv.DictReader(open('eAMFCalibrator/out_bets_drives/bets_sim.csv')):
    if r['bet_id'] in a and r['candidate_prob_v21']:
        n+=1; d=float(r['candidate_prob_v21'])-a[r['bet_id']][1]; diffs.append(d)
        if abs(d)<1e-6: same+=1
        if n<6: print(r['match_code'], r['candidate_prob_v21'], a[r['bet_id']][1:], r['stream_prob'])
import statistics
print(n, same, statistics.mean(abs(x) for x in diffs) if diffs else None)
