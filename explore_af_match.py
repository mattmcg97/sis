"""Deep-dive into American football raw feed + one full settled match.

Pulls the AF market/selection catalog (to see every moneyline/total/
handicap variant) and traces one complete match: the full raw
GAMEPLAI_STREAM message history alongside the actual score progression
and final settlement stats, so raw messages can be mapped to outcomes.
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def print_rows(cols, rows, limit=None):
    for row in (rows[:limit] if limit else rows):
        print("  " + " | ".join(f"{c}={v}" for c, v in zip(cols, row)))


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("=== Distinct MARKET_TYPE / MARKET_ID / MARKET_DESCRIPTION for AF (GAMEPLAI_STREAM) ===")
            cols, rows = fetch_all(cur, f"""
                SELECT DISTINCT MARKET_TYPE, MARKET_ID, MARKET_DESCRIPTION
                FROM {DATABASE}.SHARED.GAMEPLAI_STREAM
                WHERE MATCH_CODE LIKE 'AF%'
                ORDER BY MARKET_TYPE, MARKET_ID
                LIMIT 100
            """)
            print(f"({len(rows)} distinct combos, showing up to 100)")
            print_rows(cols, rows)

            print("\n=== SHARED.MARKET for SPORT_CODE = 'AF' ===")
            cols, rows = fetch_all(cur, f"""
                SELECT MARKET_TYPE_ID, MARKET_TYPE_NAME
                FROM {DATABASE}.SHARED.MARKET
                WHERE SPORT_CODE = 'AF'
                ORDER BY MARKET_TYPE_ID
            """)
            print_rows(cols, rows)

            print("\n=== SHARED.SELECTION for SPORT_CODE = 'AF' ===")
            cols, rows = fetch_all(cur, f"""
                SELECT MARKET_TYPE_ID, SELECTION_ID, SELECTION_DESCRIPTION
                FROM {DATABASE}.SHARED.SELECTION
                WHERE SPORT_CODE = 'AF'
                ORDER BY MARKET_TYPE_ID, SELECTION_ID
            """)
            print_rows(cols, rows)

            print("\n=== Picking a sample settled AF match with stream data ===")
            cols, rows = fetch_all(cur, f"""
                SELECT e.MATCH_CODE, e.SCHEDULED_START_TIME_UTC, e.INPLAY_EVENT_STATUS,
                       MAX(g.EVENT_MESSAGE_COUNT) AS MSG_COUNT
                FROM {DATABASE}.SHARED.EVENT e
                JOIN {DATABASE}.SHARED.GAMEPLAI_STREAM g ON g.MATCH_CODE = e.MATCH_CODE
                WHERE e.SPORT_CODE = 'AF' AND e.INPLAY_EVENT_STATUS = 'SETTLED'
                GROUP BY 1, 2, 3
                ORDER BY MSG_COUNT DESC
                LIMIT 5
            """)
            print_rows(cols, rows)

            if not rows:
                print("No settled AF match with stream data found.")
                return

            match_code = rows[0][0]
            print(f"\nUsing MATCH_CODE = {match_code}")

            print(f"\n=== GAMEPLAI_STREAM for {match_code}, ordered by PUBLISH_TIME (first 40 shown) ===")
            cols, rows = fetch_all(cur, f"""
                SELECT PUBLISH_TIME, MARKET_TYPE, MARKET_ID, MARKET_DESCRIPTION,
                       STATUS, IS_ACTIVE, SUSPEND_FLAG, DECIMAL_ODD, PROBABILITY, RESULT
                FROM {DATABASE}.SHARED.GAMEPLAI_STREAM
                WHERE MATCH_CODE = %s
                ORDER BY PUBLISH_TIME, MARKET_ID
            """, (match_code,))
            print(f"({len(rows)} total rows for this match)")
            print_rows(cols, rows, limit=40)

            print(f"\n=== SCORE_CHANGES for {match_code} ===")
            cols, rows = fetch_all(cur, f"""
                SELECT PERIOD_NUMBER, PLAYER_1_SCORE_CHANGE, PLAYER_2_SCORE_CHANGE,
                       PLAYER_1_SCORE_CUMULATIVE, PLAYER_2_SCORE_CUMULATIVE, FILE_TIME
                FROM {DATABASE}.SHARED.SCORE_CHANGES
                WHERE MATCH_CODE = %s
                ORDER BY FILE_TIME
            """, (match_code,))
            print_rows(cols, rows)

            print(f"\n=== SCORE_ENDPERIOD for {match_code} ===")
            cols, rows = fetch_all(cur, f"""
                SELECT * FROM {DATABASE}.SHARED.SCORE_ENDPERIOD
                WHERE MATCH_CODE = %s ORDER BY PERIOD_NUMBER
            """, (match_code,))
            print_rows(cols, rows)

            print(f"\n=== SCORE_ENDGAME for {match_code} ===")
            cols, rows = fetch_all(cur, f"""
                SELECT * FROM {DATABASE}.SHARED.SCORE_ENDGAME WHERE MATCH_CODE = %s
            """, (match_code,))
            print_rows(cols, rows)

            print(f"\n=== STATS_AMERICAN_FOOTBALL for {match_code} ===")
            cols, rows = fetch_all(cur, f"""
                SELECT * FROM {DATABASE}.SHARED.STATS_AMERICAN_FOOTBALL WHERE MATCH_CODE = %s
            """, (match_code,))
            print_rows(cols, rows)

            print(f"\n=== TRADING.EVENT_INPLAY_SUMMARY_NFL for {match_code} ===")
            cols, rows = fetch_all(cur, f"""
                SELECT * FROM {DATABASE}.TRADING.EVENT_INPLAY_SUMMARY_NFL WHERE MATCH_CODE = %s
            """, (match_code,))
            print_rows(cols, rows)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
