#!/bin/bash
# gamer x team GLMM: TUNE only first (daily refits); full window only if it wins TUNE.
cd /tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/out/B/glmm
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
while [ ! -f daily_g_td90_hl0.csv ]; do sleep 20; done
echo "== gtx TUNE only $(date)"
/usr/bin/Rscript ../../../scripts/B/glmm_wf.R in.csv tuneonly_gtx_td0_hl30.csv 2026-06-01 2026-08-01 gtx 0 30 1
echo done $(date)
