"""Same-line breakdown: both selections of a market are tallied, so each decision counts twice."""
import random
from eAMFCalibrator import directional as d, metrics
rnd = random.Random(3)
pairs = []
for m in range(40):                       # 40 matches, 5 drive snapshots each, moneyline only
    home_won = rnd.random() < 0.5
    for k in range(5):
        pp, pc = rnd.uniform(0.2, 0.8), None
        pc = min(0.99, max(0.01, pp + rnd.uniform(-0.05, 0.06)))
        for mid, p_, c_, won in ((50, pp, pc, home_won), (51, 1 - pp, 1 - pc, not home_won)):
            pairs.append(d.PairedObservation(match_code=f"M{m}", drive_number=k + 1, period_number=1 + k % 4,
                score_p1=0, score_p2=0, offensive_team=None, market_id=mid, message_count=10 * k,
                message_gap=0, prod_probability=p_, candidate_probability=c_, prod_line=None,
                candidate_line=None, prod_outcome=won, candidate_outcome=won, realized=None))
both = d.tally(pairs)
one = d.tally([p for p in pairs if p.market_id == 50])
print("both selections: n", both["n"], "cand", both["candidate"], "prod", both["prod"], "p", round(both["p_value"], 5))
print("home side only : n", one["n"], "cand", one["candidate"], "prod", one["prod"], "p", round(one["p_value"], 5))
v = d.match_level_votes(pairs)
print("match vote     :", v["candidate"], "-", v["prod"], "p", round(v["p_value"], 4))
