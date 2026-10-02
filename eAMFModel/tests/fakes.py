"""Synthetic PLAY_OVER exports shaped like eAMFCalibrator scouting's, shared by the version tests."""

import random

MARKET_COLUMNS = {f"{k}_{m}": "" for m in (50, 51, 52, 53, 54, 55)
                  for k in ("line", "prob", "live", "outcome")}


def _fake_match(code, rng, lead_seconds=28.0, strength=(0.0, 0.0)):
    """A toy game on the real clock. The offense ahead in the second half
    uses `lead_seconds` a snap (it milks the clock), everyone else 16."""
    rows = []
    msg = [0]
    score = {"TEAM_A": 0, "TEAM_B": 0}
    other = {"TEAM_A": "TEAM_B", "TEAM_B": "TEAM_A"}

    def emit(period, clock, kind, offense, down, dist, field, messages=""):
        msg[0] += 7
        r = dict(MARKET_COLUMNS, match_code=code, message=str(msg[0]), period=str(period),
                 clock_seconds=str(max(0, int(clock))), play_kind=kind, offense=offense,
                 down=str(down), distance=str(dist), field_position=str(field),
                 score_p1=str(score["TEAM_A"]), score_p2=str(score["TEAM_B"]),
                 play_messages=messages, team_a_side="home", opening_offense="TEAM_A",
                 prematch_line_52="0.5", prematch_prob_52="0.5", prematch_line_54="30.5",
                 prematch_prob_54="0.5", prematch_prob_50="0.5")
        rows.append(r)

    offense = "TEAM_A"
    for period in (1, 2, 3, 4):
        clock = 240.0
        if period in (1, 3):
            offense = "TEAM_A" if period == 1 else "TEAM_B"
            clock -= 4
            emit(period, clock, "KICKOFF", offense, 1, 10, 25)
        y, down, dist = 25, 1, 10
        while clock > 0:
            margin = score[offense] - score[other[offense]]
            if down == 4:
                if y >= 60:
                    clock -= 5
                    good = rng.random() < 0.8
                    if good:
                        score[offense] += 3
                    emit(period, clock, "FIELD_GOAL", offense, 1, 10, 35,
                         ("FIELD_GOAL_GOOD_" if good else "FIELD_GOAL_MISSED_") + offense)
                else:
                    clock -= 10
                    offense = other[offense]
                    y, down, dist = 30, 1, 10
                    emit(period, clock, "PUNT", offense, 1, 10, y, "POSSESSION_" + offense)
                    continue
                if clock <= 0:
                    break
                clock -= 4
                kicker = offense
                offense = other[offense]
                y, down, dist = 25, 1, 10
                emit(period, clock, "KICKOFF", offense, 1, 10, y, f"KICKOFF_{kicker}|POSSESSION_{offense}")
                continue
            q = strength[0] if offense == "TEAM_A" else strength[1]
            gain = int(rng.choice([-2, 0, 0, 3, 4, 6, 8, 12, 15, 25]) * (1 + q))
            clock -= lead_seconds if (period >= 3 and margin > 0) else 16.0
            if rng.random() < 0.02:
                offense = other[offense]
                y, down, dist = 100 - y, 1, 10
                emit(period, clock, "SCRIMMAGE", offense, down, dist, y, "POSSESSION_" + offense)
                continue
            y += gain
            if y >= 100:
                score[offense] += 6
                emit(period, clock, "TOUCHDOWN", offense, 1, 10, 85, "TOUCHDOWN_" + offense)
                score[offense] += 1
                emit(period, clock, "CONVERSION", offense, 1, 10, 35, "EXTRA_POINT_GOOD_" + offense)
                clock -= 4
                kicker = offense
                offense = other[offense]
                y, down, dist = 25, 1, 10
                emit(period, clock, "KICKOFF", offense, 1, 10, y, f"KICKOFF_{kicker}|POSSESSION_{offense}")
                continue
            y = max(1, y)
            if gain >= dist:
                down, dist = 1, min(10, 100 - y)
            else:
                down, dist = down + 1, dist - gain
            emit(period, clock, "SCRIMMAGE", offense, down, dist, y)
    for r in rows:
        r["final_p1"], r["final_p2"] = str(score["TEAM_A"]), str(score["TEAM_B"])
    return rows




def _matches(n=120, seed=3, **kw):
    rng = random.Random(seed)
    return {f"AF{i:03d}": _fake_match(f"AF{i:03d}", rng, **kw) for i in range(n)}


def _with_handles(matches):
    rng = random.Random(7)
    names = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
    for rows in matches.values():
        home, away = rng.sample(names, 2)
        for r in rows:
            r["home_handle"], r["away_handle"] = home, away
    return matches
