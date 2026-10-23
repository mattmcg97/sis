#!/bin/bash
cd /tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/out/B/glmm
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
while [ ! -f daily_gt_td0_hl30.csv ]; do sleep 20; done
for cfg in "g 90 0" "gtx 0 30"; do
  set -- $cfg
  f=daily_${1}_td${2}_hl${3}.csv
  [ -f $f ] && continue
  echo "== $cfg $(date)"
  /usr/bin/Rscript ../../../scripts/B/glmm_wf.R in.csv $f 2026-06-01 2026-10-19 $1 $2 $3 1
done
echo done $(date)
