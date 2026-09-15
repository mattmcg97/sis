"""Classify how each cleaned drive ended, using the same score-anchored
cleaning as clean_possession_sequence.py plus the field-position jump at
the handoff to the next drive.

There's no explicit play-result flag in INPLAY_FIELD_POSITION_PERIOD, so
the outcome is inferred:

  TD               SCORE_CHANGES delta=6 by the drive's own team, in the
                   window between the drive's last clean play and the next
                   drive's first (score-anchored, same as the TD-cleaning
                   logic already validated).
  FG               delta=3 by the drive's own team in that window.
  Safety           delta=2 scored by the OTHER team (the defense) in that
                   window -- distinguishes it from a 2pt conversion, which
                   is delta=2 by the SAME team that just scored the TD (and
                   so gets attached to a TD drive, not a live/failed drive).
  Turnover on downs 4th-down last play AND the next drive resumes at
                   (~)exactly the complementary field position, i.e. no
                   meaningful yardage was gained between the failed snap
                   and the new team's first play.
  Punt             no score, and the next drive resumes far down the field
                   from the complementary spot (consistent with a kick
                   flight of 30-50+ yards).
  Interception/Fumble  no score, not a 4th-down/zero-gap handoff, and the
                   field-position jump is within a plausible return
                   distance rather than a punt's.
  (end of half)    no score, and the next drive starts in a later period
                   than this one ended in, out of Q2 -- the team change is
                   just the kickoff to start the second half, not a play
                   outcome, so the gap logic above would misread it as a
                   punt/turnover and must be skipped.
  (end of period)  same idea for any other period boundary (e.g. Q4 -> OT).
  (end of match)   the last drive has no following drive to hand off to.

FIELD_POSITION is each play's own-team frame (yards from that team's own
goal line, 0-100). To compare the end of one team's drive with the start
of the other team's next drive on a common frame, mirror one side:
  complementary_position = 100 - end_pos_of_old_team
A handoff with (next_start_pos - complementary_position) near 0 didn't
move the ball; a large positive gap means the ball travelled downfield
(a kick), consistent with a punt.

Same two sample matches used throughout this thread, for direct
before/after comparison against clean_possession_sequence.py's output.
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
N_MATCHES = 2

TOD_GAP_TOLERANCE = 5     # yards -- "exactly at the point after 4th down"
PUNT_GAP_THRESHOLD = 25   # yards -- above this, treat the jump as a kick


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def clean_and_build_drives(plays, scores):
    """Same TD-anchored + stale-row-dropped cleaning as
    clean_possession_sequence.clean_drives(), but keeps each drive's
    message-count boundaries so scores and handoff gaps can be attached
    to the right drive afterwards."""

    td_scorer_team = {}
    for msg, period, p1_delta, p2_delta, p1_cum, p2_cum in scores:
        delta = p1_delta if p1_delta else p2_delta
        if delta == 6:
            td_scorer_team[msg] = "Home Team" if p1_delta else "Away Team"

    noise_msgs = set()
    resume_msgs = set()
    td_events = sorted(td_scorer_team.items())

    for td_msg, scoring_team in td_events:
        receiving_team = "Away Team" if scoring_team == "Home Team" else "Home Team"
        resume_msg = None
        for msg, period, team, down, dist, pos in plays:
            if msg <= td_msg:
                continue
            if team == receiving_team and down == 1 and dist == 10:
                resume_msg = msg
                break
        if resume_msg is None:
            continue
        resume_msgs.add(resume_msg)
        for msg, period, team, down, dist, pos in plays:
            if td_msg < msg < resume_msg:
                noise_msgs.add(msg)

    prev_team = None
    for msg, period, team, down, dist, pos in plays:
        if prev_team is not None and team != prev_team and msg not in resume_msgs:
            noise_msgs.add(msg)
        prev_team = team

    clean_plays = [p for p in plays if p[0] not in noise_msgs]

    drives = []
    current = None
    for msg, period, team, down, dist, pos in clean_plays:
        if current is None or team != current["team"]:
            if current is not None:
                drives.append(current)
            current = {
                "team": team, "n": 0,
                "start_pos": pos, "end_pos": pos,
                "start_msg": msg, "end_msg": msg,
                "last_down": down,
                "plays": [],
            }
        current["n"] += 1
        current["end_pos"] = pos
        current["end_msg"] = msg
        current["last_down"] = down
        current["plays"].append((msg, period, team, down, dist, pos))
    if current is not None:
        drives.append(current)

    return drives, len(noise_msgs)


def classify_drives(drives, scores):
    for i, d in enumerate(drives):
        window_start = d["end_msg"]
        window_end = drives[i + 1]["start_msg"] if i + 1 < len(drives) else None

        window_scores = [
            s for s in scores
            if s[0] > window_start and (window_end is None or s[0] < window_end)
        ]

        outcome = None
        for msg, period, p1_delta, p2_delta, p1_cum, p2_cum in window_scores:
            delta = p1_delta if p1_delta else p2_delta
            scoring_team = "Home Team" if p1_delta else "Away Team"
            if delta == 6 and scoring_team == d["team"]:
                outcome = "TD"
                break
            if delta == 3 and scoring_team == d["team"]:
                outcome = "FG"
                break
            if delta == 2 and scoring_team != d["team"]:
                outcome = "Safety"
                break

        if outcome is None:
            if i + 1 >= len(drives):
                outcome = "(end of match)"
            else:
                nxt = drives[i + 1]
                last_period = d["plays"][-1][1]
                next_period = nxt["plays"][0][1]
                if next_period != last_period:
                    # A period boundary with no score in between is the clock
                    # expiring, not a play outcome -- the "team change" here is
                    # just the kickoff to start the next half (or OT), so the
                    # field-position jump is meaningless and must not be run
                    # through the punt/turnover-on-downs/INT-fumble gap logic.
                    outcome = "(end of half)" if last_period == 2 else "(end of period)"
                else:
                    complementary_pos = 100 - d["end_pos"]
                    gap = nxt["start_pos"] - complementary_pos
                    if d["last_down"] == 4 and abs(gap) <= TOD_GAP_TOLERANCE:
                        outcome = "Turnover on downs"
                    elif gap > PUNT_GAP_THRESHOLD:
                        outcome = "Punt"
                    else:
                        outcome = "Interception/Fumble"

        d["outcome"] = outcome
    return drives


LONG_DRIVE_THRESHOLD = 15  # plays -- a real drive essentially never gets this long;
                           # flag it and dump the raw rows so we can see what got merged


def print_drives(drives):
    print(f"  {len(drives)} drives")
    for i, d in enumerate(drives, 1):
        flag = "  !!! SUSPICIOUSLY LONG -- likely several merged possessions !!!" if d["n"] > LONG_DRIVE_THRESHOLD else ""
        print(f"    {i:>2}. {d['team']:<11} {d['n']:>3} plays  "
              f"field {d['start_pos']:>3} -> {d['end_pos']:>3} (net {d['end_pos'] - d['start_pos']:+d})  "
              f"last_down={d['last_down']}  ==> {d['outcome']}{flag}")

    for i, d in enumerate(drives, 1):
        if d["n"] > LONG_DRIVE_THRESHOLD:
            print(f"\n  --- raw rows inside drive {i} ({d['team']}, {d['n']} plays) ---")
            print(f"  {'MSG#':>5} {'PER':>3} {'DN':>2} {'DIST':>4} {'FLD_POS':>7}")
            for msg, period, team, down, dist, pos in d["plays"]:
                note = " <- 1st down" if down == 1 else ""
                print(f"  {msg:>5} {period:>3} {down:>2} {dist:>4} {pos:>7}{note}")


def inspect_match(cur, match_code):
    print(f"\n{'=' * 100}\nMATCH_CODE = {match_code}\n{'=' * 100}")

    _, plays = fetch_all(cur, f"""
        SELECT EVENT_MESSAGE_COUNT, PERIOD_NUMBER, OFFENSIVE_TEAM, DOWN_NUMBER, DISTANCE, FIELD_POSITION
        FROM {DATABASE}.SHARED.INPLAY_FIELD_POSITION_PERIOD
        WHERE MATCH_CODE = %s
        ORDER BY EVENT_MESSAGE_COUNT
    """, (match_code,))

    _, scores = fetch_all(cur, f"""
        SELECT EVENT_MESSAGE_COUNT, PERIOD_NUMBER, PLAYER_1_SCORE_CHANGE, PLAYER_2_SCORE_CHANGE,
               PLAYER_1_SCORE_CUMULATIVE, PLAYER_2_SCORE_CUMULATIVE
        FROM {DATABASE}.SHARED.SCORE_CHANGES
        WHERE MATCH_CODE = %s
        ORDER BY EVENT_MESSAGE_COUNT
    """, (match_code,))

    print(f"{len(plays)} raw play rows, {len(scores)} score events\n")

    drives, n_noise = clean_and_build_drives(plays, scores)
    print(f"({n_noise} rows discarded as vision-recognition noise)")
    drives = classify_drives(drives, scores)
    print_drives(drives)

    print("\n  Outcome counts:")
    counts = {}
    for d in drives:
        counts[d["outcome"]] = counts.get(d["outcome"], 0) + 1
    for outcome, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {outcome:<20} {n}")


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print(f"=== Picking {N_MATCHES} sample settled AF matches with field position data ===")
            _, rows = fetch_all(cur, f"""
                SELECT e.MATCH_CODE, COUNT(*) AS N_PLAYS
                FROM {DATABASE}.SHARED.EVENT e
                JOIN {DATABASE}.SHARED.INPLAY_FIELD_POSITION_PERIOD fp ON fp.MATCH_CODE = e.MATCH_CODE
                WHERE e.SPORT_CODE = 'AF' AND e.INPLAY_EVENT_STATUS = 'SETTLED'
                GROUP BY e.MATCH_CODE, e.SCHEDULED_START_TIME_UTC
                ORDER BY e.SCHEDULED_START_TIME_UTC DESC
                LIMIT {N_MATCHES}
            """)
            for match_code, n_plays in rows:
                print(f"  {match_code}: {n_plays} plays")

            for match_code, _ in rows:
                inspect_match(cur, match_code)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
