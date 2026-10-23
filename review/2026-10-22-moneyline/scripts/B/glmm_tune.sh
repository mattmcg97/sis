#!/bin/bash
# GLMM tuning on TUNE with weekly refits (refit every 7 days), one R process at a time.
cd /tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/out/B/glmm
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
for terms in g gt gtx; do
  for td in 0 120 60; do
    for hl in 0 45; do
      if [ "$td" != "0" ] && [ "$hl" != "0" ]; then continue; fi
      f=tune_${terms}_td${td}_hl${hl}.csv
      [ -f $f ] && continue
      echo "== $terms td=$td hl=$hl $(date)"
      /usr/bin/Rscript ../../../scripts/B/glmm_wf.R in.csv $f 2026-06-01 2026-08-01 $terms $td $hl 7
    done
  done
done
echo done $(date)
