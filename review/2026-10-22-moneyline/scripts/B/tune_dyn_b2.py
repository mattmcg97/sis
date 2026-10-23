"""Batch 2: Kalman / Elo variants, started from the tuned base settings (still TUNE only)."""
import json, sys
import tune_dyn as T
import common as C

def load(name):
    return json.load(open(f"{C.OUT}/tune_dyn_{name}.json"))["kw"]

which = sys.argv[1]
if which == "kal":
    base = load("kal_team")
    for name in ["kal_team_sh", "kal_team_gt", "kal2", "mov_kal_team", "kal_team_probit", "kal_team_noint"]:
        T.tune(name, init=base)
    T.tune("kfull", init=base, maxfev=200)
elif which == "elo":
    base = load("elo_team") if __import__("os").path.exists(f"{C.OUT}/tune_dyn_elo_team.json") else {}
    for name in ["elo_team_gt", "elo_team_tau", "elo_team_sh", "mov_elo", "mov_elo_ac"]:
        T.tune(name, init=base)
