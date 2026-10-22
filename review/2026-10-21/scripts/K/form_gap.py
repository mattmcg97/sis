import csv, os, tempfile
from eAMFCalibrator import totals_calibrate as tc
from eAMFCalibrator.totals_signals import form_by_match
hist = list(csv.DictReader(open("nb2/AMFELO.csv", encoding="utf-8")))
form = form_by_match(hist)
full = list(csv.DictReader(open("eAMFCalibrator/out/match_history.csv", encoding="utf-8")))
late = [r["MATCH_CODE"] for r in full if "2026-09-11" <= r["SCHEDULED_START_TIME_UTC"][:10] <= "2026-09-30"]
early = [r["MATCH_CODE"] for r in full if "2026-08-21" <= r["SCHEDULED_START_TIME_UTC"][:10] <= "2026-09-09"]
print("form known: Aug21-Sep9 %d/%d, Sep11-Sep30 %d/%d" % (sum(c in form for c in early), len(early), sum(c in form for c in late), len(late)))
# what read_export does with a match the history lacks
d = tempfile.mkdtemp(dir=os.environ.get("SCR"))
p = os.path.join(d, "x.csv")
with open(p, "w", newline="") as fh:
    w = csv.DictWriter(fh, ["match_code", "message", "file_time", "period", "clock_seconds", "scrimmage", "score_p1", "score_p2", "line_54", "prob_54", "live_54", "final_p1", "final_p2", "prematch_line_54"])
    w.writeheader()
    w.writerow(dict(match_code=late[0], message=1, file_time="2026-09-20 10:00:00", period=1, clock_seconds=10, scrimmage=1, score_p1=7, score_p2=0, line_54=47.5, prob_54=0.5, live_54=1, final_p1=30, final_p2=14, prematch_line_54=44.5))
m = tc.read_export(p, form)
print("form read for a match after the history ends:", m[0][2]["q1"]["form"], "| usable:", tc.usable(m[0][2]["q1"], ["form"]))
