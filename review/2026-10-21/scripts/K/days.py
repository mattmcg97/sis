import sys
from eAMFCalibrator import config, money_layer as ml
config.CUTOFF_START = "2026-10-01 00:00:00"
config.CUTOFF_END = "2026-10-13 00:00:00"
ml.N_FLOW_BOOT = 50
ml.N_BOOT = 200
out = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/sweep/K/money_prod"
ml.run(out, sys.argv[1], history="eAMFCalibrator/out/match_history.csv",
       bets=["eAMFCalibrator/out_bets_v21_4k/bets_sim.csv"], grid=False)
