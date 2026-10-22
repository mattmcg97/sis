import csv, copy, datetime as dt
from eAMFCalibrator import walkthrough
h=list(csv.DictReader(open('eAMFCalibrator/out/match_history.csv',encoding='utf-8')))
# the same history with the last fortnight's finals missing (as a history "short of the finals")
h2=copy.deepcopy(h); n=0
for r in h2:
    if r['SCHEDULED_START_TIME_UTC']>='2026-10-01':
        r['PLAYER_1_FINAL_SCORE']=r['PLAYER_2_FINAL_SCORE']=''; n+=1
a=walkthrough.history_stamp(h, dt.date(2026,9,1), dt.date(2026,10,18))
b=walkthrough.history_stamp(h2, dt.date(2026,9,1), dt.date(2026,10,18))
print('finals blanked:', n, '| stamps equal:', a==b, a)
