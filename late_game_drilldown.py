"""Drill into late-game (period >= 4, i.e. Q4 + OT) AF moneyline and
totals bets: join each bet to the real score state at the moment it was
placed (via an as-of join on SCORE_CHANGES) and the match's final score,
then check calibration -- does the book's quoted implied probability
(100 / ODDS) match what actually happened, split by game state?

This is the concrete mechanism check behind the margin collapse found
in af_revenue_analysis.py: is GAMEPLAI mispricing specific late-game
states rather than being uniformly weak?

BET_DATE_UTC lags the true feed/game state by an operator-dependent
delay (network + operator processing before the bet is logged). As a
generic first-pass correction, LAG_SECONDS is subtracted from
BET_DATE_UTC before matching to the score state, so a bet is compared
against what the feed actually showed ~10s before it was logged --
closer to the price the customer actually saw and acted on. If losses
concentrate right after a scoring event within that window, it points
to stale-price arbitrage (bettors getting a bet in before the price
catches up to a score change) rather than a general model weakness.
"""

from collections import defaultdict

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
LAG_SECONDS = 10


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def moneyline_bucket(margin):
    if margin <= -15:
        return "1_trailing_15+"
    if margin <= -4:
        return "2_trailing_4-14"
    if margin <= 3:
        return "3_close_-3..3"
    if margin <= 14:
        return "4_leading_4-14"
    return "5_leading_15+"


def totals_bucket(cushion):
    if cushion <= -15:
        return "1_exceeded_by_15+"
    if cushion <= -4:
        return "2_exceeded_by_4-14"
    if cushion <= 3:
        return "3_close_to_line"
    if cushion <= 14:
        return "4_needs_4-14_more"
    return "5_needs_15+_more"


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print(f"=== Pulling late-game (period >= 4) AF moneyline + totals bets, joined to score state as of BET_DATE_UTC - {LAG_SECONDS}s ===")
            _, rows = fetch_all(cur, f"""
                WITH late_bets AS (
                    SELECT ROW_NUMBER() OVER (ORDER BY MATCH_CODE, BET_DATE_UTC) AS BET_ID,
                           MATCH_CODE, BET_DATE_UTC, MARKET_TYPE_ID, SELECTION_ID, ODDS,
                           STAKE_GBP, REVENUE_GBP, BET_PLACED_PERIOD_NUMBER, MARKET_LINE, BET_TYPE
                    FROM {DATABASE}.SHARED.CUSTOMER_REVENUE
                    WHERE SPORT_CODE = 'AF' AND MARKET_TYPE_ID IN (1, 3)
                          AND BET_PLACED_PERIOD_NUMBER >= 4
                ),
                score_at_bet AS (
                    SELECT b.BET_ID, sc.PLAYER_1_SCORE_CUMULATIVE, sc.PLAYER_2_SCORE_CUMULATIVE
                    FROM late_bets b
                    JOIN {DATABASE}.SHARED.SCORE_CHANGES sc
                        ON sc.MATCH_CODE = b.MATCH_CODE
                        AND sc.FILE_TIME <= DATEADD('second', -{LAG_SECONDS}, b.BET_DATE_UTC)
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY b.BET_ID ORDER BY sc.FILE_TIME DESC) = 1
                )
                SELECT b.BET_ID, b.MARKET_TYPE_ID, b.SELECTION_ID, b.ODDS, b.STAKE_GBP,
                       b.REVENUE_GBP, b.BET_PLACED_PERIOD_NUMBER, b.MARKET_LINE, b.BET_TYPE,
                       s.PLAYER_1_SCORE_CUMULATIVE, s.PLAYER_2_SCORE_CUMULATIVE,
                       f.PLAYER_1_SCORE AS FINAL_P1, f.PLAYER_2_SCORE AS FINAL_P2
                FROM late_bets b
                LEFT JOIN score_at_bet s ON s.BET_ID = b.BET_ID
                LEFT JOIN {DATABASE}.SHARED.SCORE_ENDGAME f ON f.MATCH_CODE = b.MATCH_CODE
            """)
            print(f"  {len(rows)} bet rows pulled")

            ml_buckets = defaultdict(lambda: {"n": 0, "stake": 0.0, "revenue": 0.0,
                                               "implied_sum": 0.0, "implied_n": 0, "wins": 0})
            tot_buckets = defaultdict(lambda: {"n": 0, "stake": 0.0, "revenue": 0.0,
                                                "implied_sum": 0.0, "implied_n": 0, "wins": 0})

            for (bet_id, mtid, sel_id, odds, stake, revenue, period, mline, bet_type,
                 p1, p2, final_p1, final_p2) in rows:
                if final_p1 is None or final_p2 is None:
                    continue
                p1 = p1 or 0
                p2 = p2 or 0
                stake = float(stake or 0)
                revenue = float(revenue or 0)

                if mtid == 1:
                    sel_score, opp_score = (p1, p2) if sel_id == 1 else (p2, p1)
                    sel_final, opp_final = (final_p1, final_p2) if sel_id == 1 else (final_p2, final_p1)
                    margin = sel_score - opp_score
                    won = sel_final > opp_final
                    bucket = moneyline_bucket(margin)
                    d = ml_buckets[bucket]
                elif mtid == 3 and mline is not None:
                    current_total = p1 + p2
                    cushion = float(mline) - current_total
                    final_total = final_p1 + final_p2
                    won = final_total > float(mline) if sel_id == 1 else final_total < float(mline)
                    bucket = totals_bucket(cushion)
                    d = tot_buckets[bucket]
                else:
                    continue

                d["n"] += 1
                d["stake"] += stake
                d["revenue"] += revenue
                d["wins"] += 1 if won else 0
                if bet_type == "Single" and odds:
                    d["implied_sum"] += 100.0 / float(odds)
                    d["implied_n"] += 1

            def print_buckets(title, buckets):
                print(f"\n=== {title} ===")
                header = f"  {'BUCKET':<22}{'N':>8}{'STAKE_GBP':>14}{'MARGIN_%':>10}{'IMPLIED_%':>11}{'REALIZED_%':>12}{'GAP':>8}"
                print(header)
                for bucket in sorted(buckets):
                    d = buckets[bucket]
                    margin_pct = 100 * d["revenue"] / d["stake"] if d["stake"] else float("nan")
                    implied = d["implied_sum"] / d["implied_n"] if d["implied_n"] else float("nan")
                    realized = 100 * d["wins"] / d["n"] if d["n"] else float("nan")
                    gap = realized - implied if d["implied_n"] else float("nan")
                    print(f"  {bucket:<22}{d['n']:>8}{d['stake']:>14.0f}{margin_pct:>10.2f}{implied:>11.2f}{realized:>12.2f}{gap:>8.2f}")
                print("  (IMPLIED_% = avg 100/ODDS on Single bets only; REALIZED_% = actual win rate, all bet types;")
                print("   GAP = REALIZED - IMPLIED: positive means the book priced this selection too generously)")

            print_buckets("Moneyline: bucketed by selection's score margin at bet time", ml_buckets)
            print_buckets("Totals: bucketed by points still needed to reach the line at bet time", tot_buckets)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
