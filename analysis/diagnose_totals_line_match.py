"""Verify the totals calibration isn't an artifact of timestamp mismatch.

unconditional_calibration.py computes "cushion" from the score at a
scoring event's FILE_TIME, but matches GAMEPLAI's PROBABILITY from the
nearest quote at or after that moment. If that nearest quote is actually
seconds or minutes later (this feed has known message gaps), we'd be
comparing a cushion computed from a stale score against a probability
GAMEPLAI calculated for a more advanced game state -- making the model
look unresponsive when it's really a matching bug.

This checks the raw time gap distribution, and dumps full detail for
the specific suspicious cell the user flagged: Q4+OT, cushion (15.5,
17.5], Under -- 12 observations across 10 matches with MODEL_%=49.36
despite REALIZED_%=91.67.
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("=== Time gap between scoring event (FILE_TIME) and matched GAMEPLAI quote (PUBLISH_TIME) ===")
            print("  (totals markets, period >= 4, across all settled AF matches)")
            _, rows = fetch_all(cur, f"""
                WITH af_matches AS (
                    SELECT MATCH_CODE FROM {DATABASE}.SHARED.EVENT
                    WHERE SPORT_CODE = 'AF' AND INPLAY_EVENT_STATUS = 'SETTLED'
                ),
                score_events AS (
                    SELECT ROW_NUMBER() OVER (ORDER BY sc.MATCH_CODE, sc.FILE_TIME) AS EVENT_ID,
                           sc.MATCH_CODE, sc.PERIOD_NUMBER, sc.FILE_TIME
                    FROM {DATABASE}.SHARED.SCORE_CHANGES sc
                    JOIN af_matches m ON m.MATCH_CODE = sc.MATCH_CODE
                    WHERE sc.PERIOD_NUMBER >= 4
                ),
                matched AS (
                    SELECT e.EVENT_ID, e.FILE_TIME, g.PUBLISH_TIME,
                           DATEDIFF('second', e.FILE_TIME, g.PUBLISH_TIME) AS GAP_SECONDS
                    FROM score_events e
                    JOIN {DATABASE}.SHARED.GAMEPLAI_STREAM g
                        ON g.MATCH_CODE = e.MATCH_CODE AND g.MARKET_ID IN (54, 55)
                        AND g.STATUS = 'open' AND g.IS_ACTIVE = 'true' AND g.PROBABILITY > 0
                        AND g.PUBLISH_TIME >= e.FILE_TIME
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY e.EVENT_ID, g.MARKET_ID ORDER BY g.PUBLISH_TIME ASC, g.MODIFIED_EPOCH ASC
                    ) = 1
                )
                SELECT GAP_SECONDS FROM matched
            """)
            gaps = sorted(r[0] for r in rows if r[0] is not None)
            if gaps:
                n = len(gaps)
                print(f"  n={n} min={gaps[0]} median={gaps[n // 2]} p90={gaps[int(n * 0.9)]} "
                      f"p99={gaps[int(n * 0.99)]} max={gaps[-1]}")
                over_30s = sum(1 for g in gaps if g > 30)
                over_120s = sum(1 for g in gaps if g > 120)
                print(f"  gap > 30s: {over_30s} ({100 * over_30s / n:.1f}%)   "
                      f"gap > 120s: {over_120s} ({100 * over_120s / n:.1f}%)")

            print("\n=== Full detail: Q4+OT, cushion 15.5-17.5, Under (the flagged suspicious cell) ===")
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
                    WHERE sc.PERIOD_NUMBER >= 4
                ),
                model_at_event AS (
                    SELECT e.EVENT_ID, g.MARKET_ID, g.PROBABILITY, g.PUBLISH_TIME, g.MARKET_DESCRIPTION,
                           TRY_CAST(REGEXP_SUBSTR(g.MARKET_DESCRIPTION, '[0-9]+.[0-9]+') AS FLOAT) AS LINE_VALUE
                    FROM score_events e
                    JOIN {DATABASE}.SHARED.GAMEPLAI_STREAM g
                        ON g.MATCH_CODE = e.MATCH_CODE AND g.MARKET_ID = 55
                        AND g.STATUS = 'open' AND g.IS_ACTIVE = 'true' AND g.PROBABILITY > 0
                        AND g.PUBLISH_TIME >= e.FILE_TIME
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY e.EVENT_ID, g.MARKET_ID ORDER BY g.PUBLISH_TIME ASC, g.MODIFIED_EPOCH ASC
                    ) = 1
                )
                SELECT e.MATCH_CODE, e.FILE_TIME, e.PLAYER_1_SCORE_CUMULATIVE, e.PLAYER_2_SCORE_CUMULATIVE,
                       mo.PUBLISH_TIME, DATEDIFF('second', e.FILE_TIME, mo.PUBLISH_TIME) AS GAP_SECONDS,
                       mo.LINE_VALUE, mo.PROBABILITY, mo.MARKET_DESCRIPTION,
                       f.PLAYER_1_SCORE AS FINAL_P1, f.PLAYER_2_SCORE AS FINAL_P2
                FROM score_events e
                JOIN model_at_event mo ON mo.EVENT_ID = e.EVENT_ID
                JOIN {DATABASE}.SHARED.SCORE_ENDGAME f ON f.MATCH_CODE = e.MATCH_CODE
                WHERE mo.LINE_VALUE - (COALESCE(e.PLAYER_1_SCORE_CUMULATIVE, 0) + COALESCE(e.PLAYER_2_SCORE_CUMULATIVE, 0)) > 15.5
                      AND mo.LINE_VALUE - (COALESCE(e.PLAYER_1_SCORE_CUMULATIVE, 0) + COALESCE(e.PLAYER_2_SCORE_CUMULATIVE, 0)) <= 17.5
                ORDER BY e.MATCH_CODE, e.FILE_TIME
            """)
            for r in rows:
                print("  " + " | ".join(str(v) for v in r))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
