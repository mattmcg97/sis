"""drives.py's snap and drive-start stats on feed-convention fakes (PLAY_OVER = closed play's kind + next snap's state)."""
from collections import Counter
from eAMFModel.tests.fakes import _matches
from eAMFModel import feed
from xgb import drives as dv

tot = Counter()
for code, rows in _matches(n=40, seed=3).items():
    rows = [dict(r, score_p1_at_start=r["score_p1"], score_p2_at_start=r["score_p2"], final_p1=rows[-1].get("final_p1", ""),
                 final_p2=rows[-1].get("final_p2", ""), home_handle="H", away_handle="A") for r in rows]
    st = dv.match_stats(code, [dict(r) for r in rows])
    recs = feed.snap_records(sorted(rows, key=lambda r: int(r["message"])))
    tot["drives.py snaps"] += st["h_snaps"] + st["a_snaps"]
    tot["sim snaps"] += len(recs)
    tot["sim snaps after KICKOFF/PUNT row"] += sum(1 for r in recs if r["fresh"])
    tot["drives.py drives"] += st["h_drives"] + st["a_drives"]
    tot["drives.py start_field"] += st["h_start_field"] + st["a_start_field"]
    tot["kickoff/punt next-state field"] += sum(int(r["field_position"]) for r in rows if r["play_kind"] in ("KICKOFF", "PUNT"))
    tot["kickoffs+punts"] += sum(1 for r in rows if r["play_kind"] in ("KICKOFF", "PUNT"))
for k, v in tot.items():
    print(f"{k:38s}{v}")
print("drives.py mean start", round(tot["drives.py start_field"] / tot["drives.py drives"], 1),
      "| fakes' real start after a kick/punt", round(tot["kickoff/punt next-state field"] / tot["kickoffs+punts"], 1))
