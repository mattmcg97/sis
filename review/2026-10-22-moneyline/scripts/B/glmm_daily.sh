#!/bin/bash
# GLMM daily walk-forward refits over 2026-06-01..10-18 (select on the TUNE part only), one R process at a time.
cd /tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/out/B/glmm
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for cfg in "g 0 30" "g 0 60" "gt 0 30" "gtx 0 30" "g 90 0"; do
  set -- $cfg
  f=daily_${1}_td${2}_hl${3}.csv
  [ -f $f ] && continue
  echo "== $cfg $(date)"
  /usr/bin/Rscript ../../../scripts/B/glmm_wf.R in.csv $f 2026-06-01 2026-10-19 $1 $2 $3 1
done
echo done $(date)
