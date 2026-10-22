from eAMFCalibrator import totals_lines as tl
row = dict(match_code="M", drive_number="3", period_number="2", score_p1="14", score_p2="10", offensive_team="Home Team",
           prod_line="30", candidate_line="31.5", realized="30", market_id="54", prod_probability="0.5", candidate_probability="0.5")
s = tl.Snap(row)
print("prod need", s.prod_need, "cand need", s.cand_need, "real", s.real, "straddled", s.straddled())
print("\n".join(tl.straddles([s])[:3]))
print("\n".join(tl.by_need([s])[2:4]))
