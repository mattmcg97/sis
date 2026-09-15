"""Use scoring events to deterministically clean the vision-derived
play-by-play stream around touchdowns.

Football has a fixed sequence: TD -> PAT/2pt attempt -> kickoff -> the
OTHER team's offense. Since there are only two teams, the receiving
team after any score is always the opponent of whoever scored --
PLAYER_1 = Home Team, PLAYER_2 = Away Team (confirmed earlier via
GAMEPLAI_STREAM market descriptions). That lets us discard the
vision-recognition artifacts around a TD (the extra kickoff-mechanic
rows with garbage distance values, and the lagging team label) without
guessing: anchor on the SCORE_CHANGES message count, and walk forward
until the KNOWN correct receiving team shows a fresh 1st-and-10.

General rule (all transitions, not just scores): the first play row
immediately after any team-label change duplicates the previous team's
down/distance/field-position -- confirmed by direct inspection earlier
-- so it's also treated as noise.

Prints raw vs. cleaned drive breakdowns side by side for the same two
sample matches used in inspect_possession_sequence.py / _with_scores.py.
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
N_MATCHES = 2


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def raw_drives(plays):
    drives = []
    current = None
    for msg, period, team, down, dist, pos in plays:
        if current is None or team != current["team"]:
            if current is not None:
                drives.append(current)
            current = {"team": team, "n": 0, "start_pos": pos, "end_pos": pos}
        current["n"] += 1
        current["end_pos"] = pos
    if current is not None:
        drives.append(current)
    return drives


def clean_drives(plays, scores):
    # TD score events: message -> scoring player team
    td_scorer_team = {}
    for msg, period, p1_delta, p2_delta, p1_cum, p2_cum in scores:
        delta = p1_delta if p1_delta else p2_delta
        if delta == 6:
            td_scorer_team[msg] = "Home Team" if p1_delta else "Away Team"

    noise_msgs = set()
    resume_msgs = set()  # TD-anchored, verified-fresh-1st-down rows -- must never be treated as noise
    td_events = sorted(td_scorer_team.items())

    for td_msg, scoring_team in td_events:
        receiving_team = "Away Team" if scoring_team == "Home Team" else "Home Team"
        # find first play after the TD where receiving_team shows a fresh 1st-and-10
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

    # General rule: first row after ANY team-label change is noise (stale duplicate) --
    # EXCEPT a TD-anchored resume row, which was already verified as a genuine fresh
    # 1st-and-10 for the correct receiving team, not a stale duplicate.
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
            current = {"team": team, "n": 0, "start_pos": pos, "end_pos": pos}
        current["n"] += 1
        current["end_pos"] = pos
    if current is not None:
        drives.append(current)
    return drives, len(noise_msgs)


def print_drives(label, drives):
    print(f"  {label}: {len(drives)} drives")
    for i, d in enumerate(drives, 1):
        print(f"    {i:>2}. {d['team']:<11} {d['n']:>3} plays  "
              f"field {d['start_pos']:>3} -> {d['end_pos']:>3} (net {d['end_pos'] - d['start_pos']:+d})")


def inspect_match(cur, match_code):
    print(f"\n{'=' * 90}\nMATCH_CODE = {match_code}\n{'=' * 90}")

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

    print_drives("RAW (naive team-change count)", raw_drives(plays))

    cleaned, n_noise = clean_drives(plays, scores)
    print(f"\n  ({n_noise} rows discarded as vision-recognition noise)")
    print_drives("CLEANED (score-anchored + stale-row-dropped)", cleaned)


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
