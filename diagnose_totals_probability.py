"""Root-cause why totals MODEL_% (Over + Under) isn't summing to ~100%
even after filtering to STATUS='open' AND IS_ACTIVE='true'.

Two checks:
1. Kickoff overround for totals (mirrors the moneyline check in
   audit_af_data.py, which found money_line summed to exactly 100.00%
   at kickoff) -- does totals behave the same way at t=0?
2. Concurrency check -- are there ever MULTIPLE rows with
   STATUS='open' AND IS_ACTIVE='true' for the same MARKET_ID at the
   same PUBLISH_TIME? That would mean the single-active-line
   assumption behind the as-of join is wrong.

Plus a full raw dump of one match's totals rows (unfiltered by status)
to see the actual open/close transition pattern directly.
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
SAMPLE_SIZE = 300


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print(f"=== Sampling last {SAMPLE_SIZE} settled AF matches ===")
            _, rows = fetch_all(cur, f"""
                SELECT MATCH_CODE FROM {DATABASE}.SHARED.EVENT
                WHERE SPORT_CODE = 'AF' AND INPLAY_EVENT_STATUS = 'SETTLED'
                ORDER BY SCHEDULED_START_TIME_UTC DESC
                LIMIT {SAMPLE_SIZE}
            """)
            match_codes = [r[0] for r in rows]
            match_list_sql = ",".join(f"'{m}'" for m in match_codes)
            print(f"  {len(match_codes)} matches")

            print("\n=== 1. Totals overround at kickoff (Over% + Under%), STATUS='open' AND IS_ACTIVE='true' ===")
            _, rows = fetch_all(cur, f"""
                WITH first_ts AS (
                    SELECT MATCH_CODE, MIN(PUBLISH_TIME) AS FIRST_TIME
                    FROM {DATABASE}.SHARED.GAMEPLAI_STREAM
                    WHERE MATCH_CODE IN ({match_list_sql}) AND MARKET_TYPE = 'totals'
                          AND STATUS = 'open' AND IS_ACTIVE = 'true'
                    GROUP BY MATCH_CODE
                )
                SELECT g.MATCH_CODE, SUM(g.PROBABILITY) AS PROB_SUM, COUNT(*) AS N_ROWS
                FROM {DATABASE}.SHARED.GAMEPLAI_STREAM g
                JOIN first_ts f ON f.MATCH_CODE = g.MATCH_CODE AND f.FIRST_TIME = g.PUBLISH_TIME
                WHERE g.MARKET_TYPE = 'totals' AND g.STATUS = 'open' AND g.IS_ACTIVE = 'true'
                GROUP BY g.MATCH_CODE
            """)
            prob_sums = [r[1] for r in rows if r[1] is not None]
            n_rows_per_match = [r[2] for r in rows]
            if prob_sums:
                print(f"  n={len(prob_sums)} min={min(prob_sums):.2f} max={max(prob_sums):.2f} "
                      f"avg={sum(prob_sums)/len(prob_sums):.2f}")
            print(f"  rows-at-kickoff-snapshot per match: min={min(n_rows_per_match)} "
                  f"max={max(n_rows_per_match)} (expect 2: one Over, one Under)")
            weird = [r for r in rows if r[2] != 2]
            print(f"  matches with != 2 rows at kickoff snapshot: {len(weird)}/{len(rows)}")
            for m, prob_sum, n in weird[:10]:
                print(f"    {m}: n_rows={n} prob_sum={prob_sum}")

            print("\n=== 2. Concurrency check: multiple 'open'+'active' rows for same MARKET_ID at same PUBLISH_TIME? ===")
            _, rows = fetch_all(cur, f"""
                SELECT MATCH_CODE, MARKET_ID, PUBLISH_TIME, COUNT(*) AS N_CONCURRENT
                FROM {DATABASE}.SHARED.GAMEPLAI_STREAM
                WHERE MATCH_CODE IN ({match_list_sql}) AND MARKET_ID IN (54, 55)
                      AND STATUS = 'open' AND IS_ACTIVE = 'true'
                GROUP BY MATCH_CODE, MARKET_ID, PUBLISH_TIME
                HAVING COUNT(*) > 1
                LIMIT 20
            """)
            print(f"  {len(rows)} (match, market_id, publish_time) groups with concurrent open+active rows (showing up to 20)")
            for match_code, market_id, publish_time, n in rows:
                print(f"    {match_code} MARKET_ID={market_id} PUBLISH_TIME={publish_time} n_concurrent={n}")

            if match_codes:
                sample_match = match_codes[0]
                print(f"\n=== 3. Full raw totals row history for {sample_match} (unfiltered by status, first 60 rows) ===")
                _, rows = fetch_all(cur, f"""
                    SELECT PUBLISH_TIME, MARKET_ID, MARKET_DESCRIPTION, STATUS, IS_ACTIVE,
                           PROBABILITY, MODIFIED_EPOCH, MESSAGE_ID
                    FROM {DATABASE}.SHARED.GAMEPLAI_STREAM
                    WHERE MATCH_CODE = %s AND MARKET_ID IN (54, 55)
                    ORDER BY PUBLISH_TIME, MARKET_ID
                    LIMIT 60
                """, (sample_match,))
                for r in rows:
                    print("  " + " | ".join(str(v) for v in r))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
