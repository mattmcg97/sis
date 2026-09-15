"""Print the full row-by-row play-by-play sequence for a few AF matches
so the possession-change inference (OFFENSIVE_TEAM switching) can be
eyeballed against down/distance/field-position, since there's no
explicit turnover/play-result flag in INPLAY_FIELD_POSITION_PERIOD to
confirm it directly.

Flags anomalies that would suggest the naive "team changed = new
possession" rule is being fooled by a data gap or quirk rather than a
real punt/turnover/score:
  - DOWN_NUMBER decreasing WITHOUT a team change (should only happen
    via a first down, i.e. DOWN_NUMBER resets to 1)
  - DOWN_NUMBER resetting to 1 WITHOUT the team change, when the prior
    distance-to-go doesn't look covered (possible missed first down or
    penalty not captured as its own row)
  - the same team continuing past a 4th down without possession
    changing (should essentially never happen barring a 4th-down
    conversion, which would reset to 1st down, not continue at 4th)
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
N_MATCHES = 2


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def inspect_match(cur, match_code):
    print(f"\n{'=' * 90}\nMATCH_CODE = {match_code}\n{'=' * 90}")
    _, rows = fetch_all(cur, f"""
        SELECT EVENT_MESSAGE_COUNT, PERIOD_NUMBER, OFFENSIVE_TEAM, DOWN_NUMBER, DISTANCE, FIELD_POSITION
        FROM {DATABASE}.SHARED.INPLAY_FIELD_POSITION_PERIOD
        WHERE MATCH_CODE = %s
        ORDER BY EVENT_MESSAGE_COUNT
    """, (match_code,))
    print(f"{len(rows)} plays\n")

    header = f"{'MSG#':>5} {'PER':>3} {'TEAM':<11} {'DN':>2} {'DIST':>4} {'FLD_POS':>7}  {'NOTE'}"
    print(header)

    prev = None
    drive_start_idx = 0
    drive_plays = []

    def flush_drive(drive_plays, drive_num):
        if not drive_plays:
            return
        team = drive_plays[0][2]
        start_pos = drive_plays[0][5]
        end_pos = drive_plays[-1][5]
        print(f"    -- drive {drive_num}: {team}, {len(drive_plays)} plays, "
              f"field position {start_pos} -> {end_pos} (net {end_pos - start_pos:+d}) --")

    drive_num = 0
    for row in rows:
        msg, period, team, down, dist, pos = row
        note = ""

        if prev is None:
            note = "(match start)"
            drive_num += 1
        else:
            pmsg, pperiod, pteam, pdown, pdist, ppos = prev
            team_changed = team != pteam

            if team_changed:
                flush_drive(drive_plays, drive_num)
                drive_plays = []
                drive_num += 1
                note = "*** NEW POSSESSION ***"
            else:
                if down < pdown:
                    if down == 1:
                        note = "(first down)"
                    else:
                        note = "!! DOWN DECREASED WITHOUT FIRST DOWN OR TEAM CHANGE !!"
                elif down == pdown and dist == pdist and pos == ppos:
                    note = "(no change vs prev row -- duplicate/no-op message?)"
                elif pdown == 4:
                    note = "!! SAME TEAM CONTINUED PAST 4TH DOWN, NO POSSESSION CHANGE !!"

            if period != pperiod and not team_changed:
                note += " [period boundary, same team retained?]"

        print(f"{msg:>5} {period:>3} {team:<11} {down:>2} {dist:>4} {pos:>7}  {note}")
        drive_plays.append(row)
        prev = row

    flush_drive(drive_plays, drive_num)
    print(f"\nTotal drives inferred: {drive_num}")


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
