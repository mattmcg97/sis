"""Concrete, named examples: late-game (period >= 4, i.e. Q4 or OT)
scoring events where the cushion (points still needed to reach the
totals line) is large -- i.e. far from the line -- but GAMEPLAI's own
PROBABILITY is still close to 50/50.

Prints a funnel first (how many events survive each filter) so it's
clear whether a filter combination returning zero rows means "this
doesn't happen" or "the filters are individually fine but their
intersection is just rare/empty." Every example row is a real
(MATCH_CODE, timestamp) pulled directly from Snowflake, with the
matched quote's own PUBLISH_TIME and gap in seconds so freshness can be
checked by eye, plus the final score. Uses the same matching logic as
unconditional_calibration.py (nearest quote at or after the scoring
event, capped at MAX_GAP_SECONDS staleness).
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
MAX_GAP_SECONDS = 30
MIN_CUSHION = 10  # "far from the line"
PROB_BAND = 7  # PROBABILITY within 50 +/- this
MAX_EXAMPLES = 40


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            base_cte = f"""
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
                           DATEDIFF('second', e.FILE_TIME, g.PUBLISH_TIME) AS GAP_SECONDS,
                           TRY_CAST(REGEXP_SUBSTR(g.MARKET_DESCRIPTION, '[0-9]+.[0-9]+') AS FLOAT) AS LINE_VALUE
                    FROM score_events e
                    JOIN {DATABASE}.SHARED.GAMEPLAI_STREAM g
                        ON g.MATCH_CODE = e.MATCH_CODE AND g.MARKET_ID IN (54, 55)
                        AND g.STATUS = 'open' AND g.IS_ACTIVE = 'true' AND g.PROBABILITY > 0
                        AND g.PUBLISH_TIME >= e.FILE_TIME
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY e.EVENT_ID, g.MARKET_ID ORDER BY g.PUBLISH_TIME ASC, g.MODIFIED_EPOCH ASC
                    ) = 1
                ),
                joined AS (
                    SELECT e.MATCH_CODE, e.PERIOD_NUMBER, e.FILE_TIME,
                           e.PLAYER_1_SCORE_CUMULATIVE, e.PLAYER_2_SCORE_CUMULATIVE,
                           mo.MARKET_ID, mo.MARKET_DESCRIPTION, mo.PROBABILITY, mo.PUBLISH_TIME, mo.GAP_SECONDS,
                           mo.LINE_VALUE,
                           mo.LINE_VALUE - (COALESCE(e.PLAYER_1_SCORE_CUMULATIVE, 0) + COALESCE(e.PLAYER_2_SCORE_CUMULATIVE, 0)) AS CUSHION,
                           f.PLAYER_1_SCORE AS FINAL_P1, f.PLAYER_2_SCORE AS FINAL_P2
                    FROM score_events e
                    JOIN model_at_event mo ON mo.EVENT_ID = e.EVENT_ID
                    JOIN {DATABASE}.SHARED.SCORE_ENDGAME f ON f.MATCH_CODE = e.MATCH_CODE
                )
            """

            print("=== Funnel: period >= 4 totals scoring-event x market rows, filter by filter ===")
            _, rows = fetch_all(cur, base_cte + "SELECT COUNT(*) FROM joined")
            print(f"  any match (any gap, any cushion, any probability): {rows[0][0]}")

            _, rows = fetch_all(cur, base_cte + f"SELECT COUNT(*) FROM joined WHERE GAP_SECONDS <= {MAX_GAP_SECONDS}")
            print(f"  + gap <= {MAX_GAP_SECONDS}s: {rows[0][0]}")

            _, rows = fetch_all(
                cur,
                base_cte + f"SELECT COUNT(*) FROM joined WHERE GAP_SECONDS <= {MAX_GAP_SECONDS} AND CUSHION > {MIN_CUSHION}",
            )
            print(f"  + cushion > {MIN_CUSHION}: {rows[0][0]}")

            _, rows = fetch_all(
                cur,
                base_cte + f"""
                SELECT COUNT(*) FROM joined
                WHERE GAP_SECONDS <= {MAX_GAP_SECONDS} AND CUSHION > {MIN_CUSHION}
                      AND PROBABILITY BETWEEN {50 - PROB_BAND} AND {50 + PROB_BAND}
                """,
            )
            print(f"  + probability within {50 - PROB_BAND}-{50 + PROB_BAND}%: {rows[0][0]}")

            print(f"\n=== Up to {MAX_EXAMPLES} example rows ===")
            _, rows = fetch_all(
                cur,
                base_cte + f"""
                SELECT MATCH_CODE, PERIOD_NUMBER, FILE_TIME, PLAYER_1_SCORE_CUMULATIVE, PLAYER_2_SCORE_CUMULATIVE,
                       LINE_VALUE, CUSHION, MARKET_ID, MARKET_DESCRIPTION, PROBABILITY, PUBLISH_TIME, GAP_SECONDS,
                       FINAL_P1, FINAL_P2
                FROM joined
                WHERE GAP_SECONDS <= {MAX_GAP_SECONDS} AND CUSHION > {MIN_CUSHION}
                      AND PROBABILITY BETWEEN {50 - PROB_BAND} AND {50 + PROB_BAND}
                ORDER BY MATCH_CODE, FILE_TIME
                LIMIT {MAX_EXAMPLES}
                """,
            )
            print(f"  {len(rows)} rows shown\n")

            cols = ["MATCH_CODE", "PERIOD", "FILE_TIME", "P1_SCORE", "P2_SCORE", "LINE_VALUE", "CUSHION",
                    "MARKET_ID", "MARKET_DESCRIPTION", "PROBABILITY", "PUBLISH_TIME", "GAP_SECONDS",
                    "FINAL_P1", "FINAL_P2"]
            for r in rows:
                print("  " + " | ".join(f"{c}={v}" for c, v in zip(cols, r)))
                if r[12] is not None and r[13] is not None:
                    current_total = (r[3] or 0) + (r[4] or 0)
                    final_total = r[12] + r[13]
                    print(f"    -> current_total={current_total}, final_total={final_total}, "
                          f"line={r[5]}, {'UNDER' if final_total < r[5] else 'OVER'} actually won")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
