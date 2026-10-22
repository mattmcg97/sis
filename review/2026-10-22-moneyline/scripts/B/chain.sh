#!/bin/bash
# wait for batch 1 (pid $1) then run batch 2
while kill -0 $1 2>/dev/null; do sleep 10; done
cd /tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/B
python tune_dyn_b2.py kal > ../../out/B/tune_dyn_kal.log 2>&1
