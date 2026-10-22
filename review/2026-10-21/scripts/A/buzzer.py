"""decision_points at a half's end with the 4th-down play's own row missing: a made kick is kept, every
other outcome dropped."""
from eAMFModel import v21

def row(msg, period, clock, kind, off, down, dist, y, a=0, b=0, messages=""):
    return dict(match_code="AF900", message=str(msg), period=str(period), clock_seconds=str(clock), play_kind=kind,
                offense=off, down=str(down), distance=str(dist), field_position=str(y), score_p1=str(a),
                score_p2=str(b), play_messages=messages, team_a_side="home", opening_offense="TEAM_A")

for label, after_score, lead in (("made kick, level", (3, 0), 0), ("went for it, no score, level", (0, 0), 0),
                                  ("went for it, 4 behind, TD", (6, 0), -4), ("went for it, 4 behind, stopped", (0, 0), -4)):
    a0, b0 = (0, 0) if lead == 0 else (0, 4)
    rows = [row(7, 2, 30, "SCRIMMAGE", "TEAM_A", 3, 8, 70, a0, b0),
            row(14, 2, 4, "SCRIMMAGE", "TEAM_A", 4, 5, 73, a0, b0),        # 4th and 5 at the 27, 0:04 left
            row(21, 3, 236, "KICKOFF", "TEAM_A", 1, 10, 25, a0 + after_score[0], b0 + after_score[1])]
    pts = v21.decision_points(rows)
    print(f"{label:32s} 4th-down records: {[(q['clock'], ('go','fg','punt')[q['choice']]) for q in pts['fourth']]}")
