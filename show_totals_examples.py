"""Concrete, named examples: Q4 scoring events where the cushion (points
still needed to reach the totals line) is large -- i.e. far from the
line -- but GAMEPLAI's own PROBABILITY is still close to 50/50.

Every row is a real (MATCH_CODE, timestamp) pulled directly from
Snowflake, with the matched quote's own PUBLISH_TIME and the gap in
seconds so freshness can be checked by eye, plus the final score so the
actual outcome is visible. Uses the same matching logic as
unconditional_calibration.py (nearest quote at or after the scoring
event, capped at 30s staleness) so these examples are directly
representative of what feeds that script's Q4 buckets.
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
MAX_GAP_SECONDS = 30
MIN_CUSHION = 20  # "far from the line"
PROB_BAND = 5  # PROBABILITY within 50 +/- this


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print(f"=== Q4 scoring events, cushion > {MIN_CUSHION}, GAMEPLAI PROBABILITY within "
                  f"{50 - PROB_BAND}-{50 + PROB_BAND}%, quote fresh within {MAX_GAP_SECONDS}s ===")
            _, rows = fetch_all(cur, f"""
                WITH af_matches AS (
                    SELECT MATCH_CODE FROM {DATABASE}.SHARED.EVENT
                    WHERE SPORT_CODE = 'AF' AND INPLAY_EVENT_STATUS = 'SETTLED'
                ),
                score_events AS (
                    SELECT ROW_NUMBER() OVER (ORDER BY sc.MATCH_CODE, sc.FILE_TIME) AS EVENT_ID,
                           sc.MATCH_CODE, sc.PERIOD_NUMBER, sc.FILE_TIME,
                           sc.PLAYER_1_SCORE_CUMULATIVE, sc.PLAYER_2_SCORE_CUMULATIVE
                    FROM {DATABASE}.SHARED.SCORE_CHANGES sc
                    JOIN af_matches m ON m.MATCH_CODE = sc.MATCH_CODE
                    WHERE sc.PERIOD_NUMBER = 4
                ),
                model_at_event AS (
                    SELECT e.EVENT_ID, g.MARKET_ID, g.PROBABILITY, g.PUBLISH_TIME, g.MARKET_DESCRIPTION,
                           DATEDIFF('second', e.FILE_TIME, g.PUBLISH_TIME) AS GAP_SECONDS,
                           TRY_CAST(REGEXP_SUBSTR(g.MARKET_DESCRIPTION, '[0-9]+.[0-9]+') AS FLOAT) AS LINE_VALUE
                    FROM score_events e
                    JOIN {DATABASE}.SHARED.GAMEPLAI_STREAM g
                        ON g.MATCH_CODE = e.MATCH_CODE AND g.MARKET_ID IN (54, 55)
                        AND g.STATUS = 'open' AND g.IS_ACTIVE = 'true' AND g.PROBABILITY > 0
                        AND g.PUBLISH_TIME >= e.FILE_TIME
                        AND DATEDIFF('second', e.FILE_TIME, g.PUBLISH_TIME) <= {MAX_GAP_SECONDS}
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY e.EVENT_ID, g.MARKET_ID ORDER BY g.PUBLISH_TIME ASC, g.MODIFIED_EPOCH ASC
                    ) = 1
                )
                SELECT e.MATCH_CODE, e.FILE_TIME, e.PLAYER_1_SCORE_CUMULATIVE, e.PLAYER_2_SCORE_CUMULATIVE,
                       mo.LINE_VALUE,
                       mo.LINE_VALUE - (COALESCE(e.PLAYER_1_SCORE_CUMULATIVE, 0) + COALESCE(e.PLAYER_2_SCORE_CUMULATIVE, 0)) AS CUSHION,
                       mo.MARKET_ID, mo.MARKET_DESCRIPTION, mo.PROBABILITY, mo.PUBLISH_TIME, mo.GAP_SECONDS,
                       f.PLAYER_1_SCORE AS FINAL_P1, f.PLAYER_2_SCORE AS FINAL_P2
                FROM score_events e
                JOIN model_at_event mo ON mo.EVENT_ID = e.EVENT_ID
                JOIN {DATABASE}.SHARED.SCORE_ENDGAME f ON f.MATCH_CODE = e.MATCH_CODE
                WHERE mo.LINE_VALUE - (COALESCE(e.PLAYER_1_SCORE_CUMULATIVE, 0) + COALESCE(e.PLAYER_2_SCORE_CUMULATIVE, 0)) > {MIN_CUSHION}
                      AND mo.PROBABILITY BETWEEN {50 - PROB_BAND} AND {50 + PROB_BAND}
                ORDER BY e.MATCH_CODE, e.FILE_TIME
            """)
            print(f"  {len(rows)} matching rows\n")

            cols = ["MATCH_CODE", "FILE_TIME", "P1_SCORE", "P2_SCORE", "LINE_VALUE", "CUSHION",
                    "MARKET_ID", "MARKET_DESCRIPTION", "PROBABILITY", "PUBLISH_TIME", "GAP_SECONDS",
                    "FINAL_P1", "FINAL_P2"]
            for r in rows:
                print("  " + " | ".join(f"{c}={v}" for c, v in zip(cols, r)))
                if r[11] is not None and r[12] is not None:
                    current_total = (r[2] or 0) + (r[3] or 0)
                    final_total = r[11] + r[12]
                    print(f"    -> current_total={current_total}, final_total={final_total}, "
                          f"line={r[4]}, {'UNDER' if final_total < r[4] else 'OVER'} actually won")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
