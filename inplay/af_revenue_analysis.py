"""Where is the book losing money on AF markets, and is there a session
effect across a gamer's consecutive matches?

Margin % = SUM(REVENUE_GBP) / SUM(STAKE_GBP) * 100. Positive = book
profit, negative = book loss (customers winning more than expected).

AF market types (SHARED.MARKET): 1=Moneyline, 2=Team Handicap,
3=Total Points.
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"

MARKET_TYPE_NAMES = {1: "Moneyline", 2: "Handicap", 3: "Totals"}


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def print_table(cols, rows):
    widths = [max(len(str(c)), *(len(str(r[i])) for r in rows)) if rows else len(str(c)) for i, c in enumerate(cols)]
    print("  " + "  ".join(str(c).ljust(w) for c, w in zip(cols, widths)))
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("=== A. Overall AF margin by market type ===")
            cols, rows = fetch_all(cur, f"""
                SELECT MARKET_TYPE_ID, COUNT(*) AS N_ROWS,
                       SUM(STAKE_GBP) AS STAKE_GBP, SUM(REVENUE_GBP) AS REVENUE_GBP,
                       ROUND(100 * SUM(REVENUE_GBP) / NULLIF(SUM(STAKE_GBP), 0), 2) AS MARGIN_PCT
                FROM {DATABASE}.SHARED.CUSTOMER_REVENUE
                WHERE SPORT_CODE = 'AF' AND MARKET_TYPE_ID IN (1, 2, 3)
                GROUP BY MARKET_TYPE_ID
                ORDER BY MARKET_TYPE_ID
            """)
            for row in rows:
                mtid = row[0]
                print(f"  [{mtid}] {MARKET_TYPE_NAMES.get(mtid, mtid)}: n={row[1]} stake={row[2]:.0f} revenue={row[3]:.0f} margin={row[4]}%")

            print("\n=== B. Margin by market type x period placed (0 = pre-match) ===")
            cols, rows = fetch_all(cur, f"""
                SELECT MARKET_TYPE_ID, BET_PLACED_PERIOD_NUMBER,
                       COUNT(*) AS N_ROWS, SUM(STAKE_GBP) AS STAKE_GBP,
                       ROUND(100 * SUM(REVENUE_GBP) / NULLIF(SUM(STAKE_GBP), 0), 2) AS MARGIN_PCT
                FROM {DATABASE}.SHARED.CUSTOMER_REVENUE
                WHERE SPORT_CODE = 'AF' AND MARKET_TYPE_ID IN (1, 2, 3)
                GROUP BY MARKET_TYPE_ID, BET_PLACED_PERIOD_NUMBER
                ORDER BY MARKET_TYPE_ID, BET_PLACED_PERIOD_NUMBER
            """)
            print_table(["MARKET_TYPE", "PERIOD", "N_ROWS", "STAKE_GBP", "MARGIN_PCT"],
                         [(MARKET_TYPE_NAMES.get(r[0], r[0]), r[1], r[2], f"{r[3]:.0f}", r[4]) for r in rows])

            print("\n=== C. Margin by market type x in-play time band ===")
            cols, rows = fetch_all(cur, f"""
                SELECT MARKET_TYPE_ID, TIME_BET_PLACED_BANDING_IN_PLAY,
                       COUNT(*) AS N_ROWS, SUM(STAKE_GBP) AS STAKE_GBP,
                       ROUND(100 * SUM(REVENUE_GBP) / NULLIF(SUM(STAKE_GBP), 0), 2) AS MARGIN_PCT
                FROM {DATABASE}.SHARED.CUSTOMER_REVENUE
                WHERE SPORT_CODE = 'AF' AND MARKET_TYPE_ID IN (1, 2, 3)
                GROUP BY MARKET_TYPE_ID, TIME_BET_PLACED_BANDING_IN_PLAY
                ORDER BY MARKET_TYPE_ID, TIME_BET_PLACED_BANDING_IN_PLAY
            """)
            print_table(["MARKET_TYPE", "TIME_BAND", "N_ROWS", "STAKE_GBP", "MARGIN_PCT"],
                         [(MARKET_TYPE_NAMES.get(r[0], r[0]), r[1], r[2], f"{r[3]:.0f}", r[4]) for r in rows])

            print("\n=== D. Session effect: margin by nth match of the day for a gamer ===")
            print("  (unifies a handle's appearances as player 1 or player 2, ranked within each day)")
            cols, rows = fetch_all(cur, f"""
                WITH appearances AS (
                    SELECT MATCH_CODE, PLAYER_1_HANDLE AS HANDLE, SCHEDULE_EVENT_START_TIME_UTC
                    FROM {DATABASE}.SHARED.CUSTOMER_REVENUE
                    WHERE SPORT_CODE = 'AF' AND PLAYER_1_HANDLE IS NOT NULL
                    GROUP BY 1, 2, 3
                    UNION
                    SELECT MATCH_CODE, PLAYER_2_HANDLE AS HANDLE, SCHEDULE_EVENT_START_TIME_UTC
                    FROM {DATABASE}.SHARED.CUSTOMER_REVENUE
                    WHERE SPORT_CODE = 'AF' AND PLAYER_2_HANDLE IS NOT NULL
                    GROUP BY 1, 2, 3
                ),
                ranked AS (
                    SELECT MATCH_CODE, HANDLE,
                           ROW_NUMBER() OVER (
                               PARTITION BY HANDLE, DATE(SCHEDULE_EVENT_START_TIME_UTC)
                               ORDER BY SCHEDULE_EVENT_START_TIME_UTC
                           ) AS MATCH_NUM_IN_DAY
                    FROM appearances
                ),
                match_rev AS (
                    SELECT MATCH_CODE, MARKET_TYPE_ID, SUM(STAKE_GBP) AS STAKE_GBP, SUM(REVENUE_GBP) AS REVENUE_GBP
                    FROM {DATABASE}.SHARED.CUSTOMER_REVENUE
                    WHERE SPORT_CODE = 'AF' AND MARKET_TYPE_ID IN (1, 2, 3)
                    GROUP BY MATCH_CODE, MARKET_TYPE_ID
                )
                SELECT LEAST(r.MATCH_NUM_IN_DAY, 5) AS MATCH_NUM_CAPPED,
                       mr.MARKET_TYPE_ID,
                       COUNT(DISTINCT r.MATCH_CODE) AS N_MATCHES,
                       SUM(mr.STAKE_GBP) AS STAKE_GBP,
                       ROUND(100 * SUM(mr.REVENUE_GBP) / NULLIF(SUM(mr.STAKE_GBP), 0), 2) AS MARGIN_PCT
                FROM ranked r
                JOIN match_rev mr ON mr.MATCH_CODE = r.MATCH_CODE
                GROUP BY MATCH_NUM_CAPPED, mr.MARKET_TYPE_ID
                ORDER BY mr.MARKET_TYPE_ID, MATCH_NUM_CAPPED
            """)
            print("  (MATCH_NUM_CAPPED=5 means '5th or later')")
            print_table(["NTH_MATCH", "MARKET_TYPE", "N_MATCHES", "STAKE_GBP", "MARGIN_PCT"],
                         [(r[0], MARKET_TYPE_NAMES.get(r[1], r[1]), r[2], f"{r[3]:.0f}", r[4]) for r in rows])
    finally:
        conn.close()


if __name__ == "__main__":
    main()
