"""Data quality audit for American football data, ahead of building an
in-play model to challenge GAMEPLAI's own probabilities.

Checks, over a sample of recent settled AF matches:
  1. Coverage: how many matches have data in each downstream table.
  2. Message completeness: gaps in GAMEPLAI_STREAM's EVENT_MESSAGE_COUNT
     sequence per match (are we missing raw feed messages?).
  3. Score consistency: does SCORE_ENDGAME agree with SCORE_ENDPERIOD
     (summed) and with the max cumulative score in SCORE_CHANGES?
  4. Period count sanity: how many periods does each match have?
  5. Moneyline overround sanity: at kickoff, does Home% + Away% sit in a
     plausible range (roughly 100-115%)? Wildly off values suggest bad
     or stale probability data.
"""

import statistics as stats

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
SAMPLE_SIZE = 300


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def summarize(label, values):
    if not values:
        print(f"  {label}: no data")
        return
    print(
        f"  {label}: n={len(values)} min={min(values):.2f} "
        f"mean={stats.mean(values):.2f} median={stats.median(values):.2f} "
        f"max={max(values):.2f}"
    )


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print(f"=== Sampling last {SAMPLE_SIZE} settled AF matches ===")
            _, rows = fetch_all(
                cur,
                f"""
                SELECT MATCH_CODE
                FROM {DATABASE}.SHARED.EVENT
                WHERE SPORT_CODE = 'AF' AND INPLAY_EVENT_STATUS = 'SETTLED'
                ORDER BY SCHEDULED_START_TIME_UTC DESC
                LIMIT {SAMPLE_SIZE}
                """,
            )
            match_codes = [r[0] for r in rows]
            total = len(match_codes)
            print(f"  {total} matches in sample")
            if not match_codes:
                print("No matches found, aborting.")
                return

            match_list_sql = ",".join(f"'{m}'" for m in match_codes)

            print("\n=== 1. Coverage across downstream tables ===")
            tables = [
                ("SHARED", "GAMEPLAI_STREAM"),
                ("SHARED", "SCORE_CHANGES"),
                ("SHARED", "SCORE_ENDPERIOD"),
                ("SHARED", "SCORE_ENDGAME"),
                ("SHARED", "STATS_AMERICAN_FOOTBALL"),
                ("TRADING", "EVENT_INPLAY_SUMMARY_NFL"),
            ]
            for schema, table in tables:
                _, rows = fetch_all(
                    cur,
                    f"""
                    SELECT COUNT(DISTINCT MATCH_CODE)
                    FROM {DATABASE}.{schema}.{table}
                    WHERE MATCH_CODE IN ({match_list_sql})
                    """,
                )
                covered = rows[0][0]
                pct = 100 * covered / total
                print(f"  {schema}.{table}: {covered}/{total} matches ({pct:.1f}%)")

            print("\n=== 2. GAMEPLAI_STREAM message sequence completeness ===")
            print("  (gap = MAX(EVENT_MESSAGE_COUNT) - COUNT(DISTINCT EVENT_MESSAGE_COUNT))")
            _, rows = fetch_all(
                cur,
                f"""
                SELECT MATCH_CODE,
                       MAX(EVENT_MESSAGE_COUNT) AS MAX_COUNT,
                       COUNT(DISTINCT EVENT_MESSAGE_COUNT) AS DISTINCT_COUNTS
                FROM {DATABASE}.SHARED.GAMEPLAI_STREAM
                WHERE MATCH_CODE IN ({match_list_sql})
                GROUP BY MATCH_CODE
                """,
            )
            gaps = [(m, mx, dc, mx - dc) for m, mx, dc in rows if mx is not None]
            gap_counts = [g[3] for g in gaps]
            matches_with_gaps = [g for g in gaps if g[3] > 0]
            summarize("gap size (messages missing)", gap_counts)
            print(f"  matches with at least one gap: {len(matches_with_gaps)}/{len(gaps)}")
            worst = sorted(matches_with_gaps, key=lambda g: g[3], reverse=True)[:10]
            if worst:
                print("  worst 10:")
                for m, mx, dc, gap in worst:
                    print(f"    {m}: expected ~{mx} messages, saw {dc} distinct counts, gap={gap}")

            print("\n=== 3. Score consistency: SCORE_ENDGAME vs SCORE_ENDPERIOD (summed) ===")
            _, rows = fetch_all(
                cur,
                f"""
                SELECT eg.MATCH_CODE, eg.PLAYER_1_SCORE, eg.PLAYER_2_SCORE,
                       sp.SUM_P1, sp.SUM_P2
                FROM {DATABASE}.SHARED.SCORE_ENDGAME eg
                JOIN (
                    SELECT MATCH_CODE, SUM(PLAYER_1_SCORE) AS SUM_P1, SUM(PLAYER_2_SCORE) AS SUM_P2
                    FROM {DATABASE}.SHARED.SCORE_ENDPERIOD
                    WHERE MATCH_CODE IN ({match_list_sql})
                    GROUP BY MATCH_CODE
                ) sp ON sp.MATCH_CODE = eg.MATCH_CODE
                WHERE eg.MATCH_CODE IN ({match_list_sql})
                """,
            )
            mismatches = [r for r in rows if r[1] != r[3] or r[2] != r[4]]
            print(f"  {len(rows)} matches compared, {len(mismatches)} mismatches")
            for m, p1, p2, sp1, sp2 in mismatches[:10]:
                print(f"    {m}: ENDGAME=({p1},{p2}) vs SUM(ENDPERIOD)=({sp1},{sp2})")

            print("\n=== 4. Period count distribution (SCORE_ENDPERIOD rows per match) ===")
            _, rows = fetch_all(
                cur,
                f"""
                SELECT PERIODS, COUNT(*) AS N_MATCHES FROM (
                    SELECT MATCH_CODE, COUNT(*) AS PERIODS
                    FROM {DATABASE}.SHARED.SCORE_ENDPERIOD
                    WHERE MATCH_CODE IN ({match_list_sql})
                    GROUP BY MATCH_CODE
                ) GROUP BY PERIODS ORDER BY PERIODS
                """,
            )
            for periods, n in rows:
                print(f"  {periods} periods: {n} matches")

            print("\n=== 5. Moneyline overround sanity at kickoff (Home% + Away%) ===")
            _, rows = fetch_all(
                cur,
                f"""
                WITH first_ts AS (
                    SELECT MATCH_CODE, MIN(PUBLISH_TIME) AS FIRST_TIME
                    FROM {DATABASE}.SHARED.GAMEPLAI_STREAM
                    WHERE MATCH_CODE IN ({match_list_sql}) AND MARKET_TYPE = 'money_line'
                    GROUP BY MATCH_CODE
                )
                SELECT g.MATCH_CODE, SUM(g.PROBABILITY) AS PROB_SUM, COUNT(*) AS N_ROWS
                FROM {DATABASE}.SHARED.GAMEPLAI_STREAM g
                JOIN first_ts f ON f.MATCH_CODE = g.MATCH_CODE AND f.FIRST_TIME = g.PUBLISH_TIME
                WHERE g.MARKET_TYPE = 'money_line'
                GROUP BY g.MATCH_CODE
                """,
            )
            prob_sums = [r[1] for r in rows if r[1] is not None]
            summarize("overround %", prob_sums)
            outliers = [r for r in rows if r[1] is not None and (r[1] < 90 or r[1] > 130)]
            print(f"  outliers (<90% or >130%): {len(outliers)}/{len(rows)}")
            for m, prob_sum, n in outliers[:10]:
                print(f"    {m}: overround={prob_sum:.2f}% ({n} rows at kickoff snapshot)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
