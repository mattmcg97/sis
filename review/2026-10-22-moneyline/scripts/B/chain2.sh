#!/bin/bash
# after the BT grids, run Elo batch 2 and kal_team_pair
cd /tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/B
while [ ! -f ../../out/B/tune_bt_x2.csv ]; do sleep 15; done
python tune_dyn_b2.py elo > ../../out/B/tune_dyn_elo.log 2>&1
python -c "
import json, tune_dyn as T, common as C
T.tune('kal_team_pair', init=json.load(open(C.OUT + '/tune_dyn_kal_team.json'))['kw'])
" > ../../out/B/tune_dyn_pair.log 2>&1
