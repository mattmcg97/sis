"""Derive possession counts per team from AF play-by-play data
(INPLAY_FIELD_POSITION), and sanity-check the "shared possession pool"
hypothesis behind a possessions x points-per-possession model: does one
team's possession count move inversely with the other's (a real shared
resource), and does possession count actually predict points scored?

A new possession = a play whose OFFENSIVE_TEAM differs from the
previous play in the same match (standard "gaps and islands" SQL
pattern via LAG + running SUM).
"""

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
SAMPLE_SIZE = 1000


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print(f"=== Coverage: settled AF matches with INPLAY_FIELD_POSITION data ===")
            _, rows = fetch_all(cur, f"""
                SELECT COUNT(DISTINCT e.MATCH_CODE) AS TOTAL,
                       COUNT(DISTINCT fp.MATCH_CODE) AS WITH_DATA
                FROM {DATABASE}.SHARED.EVENT e
                LEFT JOIN {DATABASE}.SHARED.INPLAY_FIELD_POSITION fp ON fp.MATCH_CODE = e.MATCH_CODE
                WHERE e.SPORT_CODE = 'AF' AND e.INPLAY_EVENT_STATUS = 'SETTLED'
            """)
            total, with_data = rows[0]
            print(f"  {with_data}/{total} settled AF matches have field position data ({100*with_data/total:.1f}%)")

            print(f"\n=== Deriving possession counts for last {SAMPLE_SIZE} settled AF matches with data ===")
            _, rows = fetch_all(cur, f"""
                WITH sample_matches AS (
                    SELECT e.MATCH_CODE, e.PLAYER_1_HANDLE, e.PLAYER_2_HANDLE
                    FROM {DATABASE}.SHARED.EVENT e
                    WHERE e.SPORT_CODE = 'AF' AND e.INPLAY_EVENT_STATUS = 'SETTLED'
                    ORDER BY e.SCHEDULED_START_TIME_UTC DESC
                    LIMIT {SAMPLE_SIZE}
                ),
                plays AS (
                    SELECT fp.MATCH_CODE, fp.EVENT_MESSAGE_COUNT, fp.OFFENSIVE_TEAM,
                           LAG(fp.OFFENSIVE_TEAM) OVER (
                               PARTITION BY fp.MATCH_CODE ORDER BY fp.EVENT_MESSAGE_COUNT
                           ) AS PREV_TEAM
                    FROM {DATABASE}.SHARED.INPLAY_FIELD_POSITION fp
                    JOIN sample_matches m ON m.MATCH_CODE = fp.MATCH_CODE
                ),
                flagged AS (
                    SELECT *, CASE WHEN PREV_TEAM IS NULL OR OFFENSIVE_TEAM <> PREV_TEAM THEN 1 ELSE 0 END AS NEW_POSSESSION
                    FROM plays
                ),
                grouped AS (
                    SELECT *, SUM(NEW_POSSESSION) OVER (
                        PARTITION BY MATCH_CODE ORDER BY EVENT_MESSAGE_COUNT
                    ) AS POSSESSION_ID
                    FROM flagged
                ),
                per_team AS (
                    SELECT MATCH_CODE, OFFENSIVE_TEAM,
                           COUNT(DISTINCT POSSESSION_ID) AS POSSESSIONS, COUNT(*) AS PLAYS
                    FROM grouped
                    GROUP BY MATCH_CODE, OFFENSIVE_TEAM
                )
                SELECT pt.MATCH_CODE, pt.OFFENSIVE_TEAM, pt.POSSESSIONS, pt.PLAYS,
                       f.PLAYER_1_SCORE, f.PLAYER_2_SCORE
                FROM per_team pt
                JOIN {DATABASE}.SHARED.SCORE_ENDGAME f ON f.MATCH_CODE = pt.MATCH_CODE
            """)
            print(f"  {len(rows)} (match, team) rows pulled")

            by_match = {}
            for match_code, off_team, possessions, plays, p1_score, p2_score in rows:
                d = by_match.setdefault(match_code, {})
                side = "home" if off_team == "Home Team" else "away"
                d[side] = {"possessions": possessions, "plays": plays}
                d["p1_score"] = p1_score
                d["p2_score"] = p2_score

            complete = {m: d for m, d in by_match.items() if "home" in d and "away" in d}
            print(f"  {len(complete)} matches with both sides represented")

            home_poss = [d["home"]["possessions"] for d in complete.values()]
            away_poss = [d["away"]["possessions"] for d in complete.values()]
            totals = [h + a for h, a in zip(home_poss, away_poss)]
            home_pts = [d["p1_score"] for d in complete.values()]
            away_pts = [d["p2_score"] for d in complete.values()]

            def stats(label, vals):
                vals = sorted(vals)
                n = len(vals)
                print(f"  {label}: n={n} min={vals[0]} median={vals[n//2]} mean={sum(vals)/n:.2f} max={vals[-1]}")

            print("\n=== Possession count distribution ===")
            stats("Home possessions", home_poss)
            stats("Away possessions", away_poss)
            stats("Total possessions (both teams)", totals)

            def corr(xs, ys):
                n = len(xs)
                mx, my = sum(xs) / n, sum(ys) / n
                cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
                sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
                sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
                return cov / (sx * sy) if sx > 0 and sy > 0 else float("nan")

            print("\n=== Sanity checks ===")
            print(f"  Corr(home possessions, away possessions) within a match: {corr(home_poss, away_poss):.3f}")
            print("    (negative => shared/zero-sum pool: one team's extra possessions come at the other's expense)")
            print(f"  Corr(home possessions, home points scored): {corr(home_poss, home_pts):.3f}")
            print(f"  Corr(away possessions, away points scored): {corr(away_poss, away_pts):.3f}")
            print("    (positive => possession count genuinely predicts scoring, supporting the decomposition)")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
