"""Interleave AF play-by-play (INPLAY_FIELD_POSITION_PERIOD) with actual
scoring events (SCORE_CHANGES), merged by EVENT_MESSAGE_COUNT (both
tables share the same vendor message sequence per match), to check
whether the "field position near 100, then a few messages of unusual
distance values while OFFENSIVE_TEAM hasn't flipped yet" pattern seen
in inspect_possession_sequence.py really lines up with touchdowns, and
to measure exactly how many messages the team-label lags a real score.

Same two sample matches as inspect_possession_sequence.py (most recent
settled AF matches with field position data), for direct comparison.
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
N_MATCHES = 2

SCORE_TYPE = {6: "TD", 3: "FG", 1: "XP", 2: "SAFETY/2PT"}


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


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

    # Merge into one timeline: (msg_count, kind, payload), kind in {"PLAY", "SCORE"}
    timeline = []
    for row in plays:
        timeline.append((row[0], "PLAY", row))
    for row in scores:
        timeline.append((row[0], "SCORE", row))
    timeline.sort(key=lambda t: (t[0], 0 if t[1] == "SCORE" else 1))  # scores shown before plays at same msg#

    print(f"{len(plays)} plays, {len(scores)} score-change events\n")
    header = f"{'MSG#':>5} {'PER':>3} {'EVENT'}"
    print(header)

    prev_team = None
    last_score_msg = None
    for msg, kind, row in timeline:
        if kind == "SCORE":
            _, period, p1_delta, p2_delta, p1_cum, p2_cum = row
            delta = p1_delta if p1_delta else p2_delta
            who = "PLAYER_1" if p1_delta else "PLAYER_2"
            score_type = SCORE_TYPE.get(delta, f"?({delta})")
            print(f"{msg:>5} {period:>3}  $$$ SCORE: {who} +{delta} [{score_type}]  "
                  f"cumulative {p1_cum}-{p2_cum} $$$")
            last_score_msg = msg
        else:
            _, period, team, down, dist, pos = row
            note = ""
            if prev_team is not None and team != prev_team:
                note = "*** NEW POSSESSION ***"
                if last_score_msg is not None:
                    note += f"  (lag since last score: {msg - last_score_msg} messages)"
            print(f"{msg:>5} {period:>3}  play: {team:<11} down={down} dist={dist:>3} field_pos={pos:>3}  {note}")
            prev_team = team


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
