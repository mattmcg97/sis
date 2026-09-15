"""Drill into AF moneyline and totals bets from Q2 onward: join each bet
to the real score state at the moment it was placed (via an as-of join
on SCORE_CHANGES) and the match's final score, then check calibration --
does the book's quoted implied probability (100 / ODDS) match what
actually happened?

Widened from the original Q4+OT-only cut: period floor is now 2 (roughly
6-7x more data), period itself is kept as an explicit cross-section
dimension instead of being filtered away, and margin/cushion are
quantile-binned (equal population per bucket) instead of fixed
thresholds, so small buckets stop being unreliable. Crossed with
selection (Home/Away, Over/Under) to isolate exactly which game states
and which side of the bet the mispricing concentrates in.

BET_DATE_UTC lags the true feed/game state by an operator-dependent
delay. LAG_SECONDS is subtracted from BET_DATE_UTC before matching to
the score state as a generic first-pass correction (confirmed to barely
move the numbers in the Q4+OT-only pass -- kept for consistency).

IMPLIED_% (from customer ODDS) includes the operator's overround, so it
is not directly comparable to a fair probability -- a healthy ~50/50
market prices around 54%, not 50%, and GAP against it is expected to
sit near -(target margin), not near zero. MODEL_% is GAMEPLAI's own
quoted PROBABILITY (confirmed de-vigged: Home%+Away% summed to exactly
100.00% at kickoff in the earlier audit), pulled via the same as-of
join pattern against GAMEPLAI_STREAM, matched to the MARKET_ID for that
market type + selection (money_line: 50/51, totals: 54/55). GAP_MODEL
(REALIZED - MODEL_%) is the cleaner calibration read since it isn't
confounded by the operator's margin.
"""

import statistics as stats
from collections import defaultdict

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
LAG_SECONDS = 10
PERIOD_FLOOR = 2
N_QUANTILE_BINS = 8

ML_SELECTION_NAMES = {1: "Home", 2: "Away"}
TOT_SELECTION_NAMES = {1: "Over", 2: "Under"}


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def period_group(period):
    if period is None:
        return "0_unknown"
    if period <= 2:
        return "1_Q1-Q2"
    if period == 3:
        return "2_Q3"
    return "3_Q4+OT"


def quantile_cutpoints(values, n_bins):
    values = sorted(v for v in values if v is not None)
    if len(values) < n_bins * 5:
        return []
    return stats.quantiles(values, n=n_bins)


def bucket_index(x, cutpoints):
    idx = 0
    for c in cutpoints:
        if x > c:
            idx += 1
        else:
            break
    return idx


def bucket_label(idx, cutpoints):
    lo = "-inf" if idx == 0 else f"{cutpoints[idx - 1]:.1f}"
    hi = "+inf" if idx == len(cutpoints) else f"{cutpoints[idx]:.1f}"
    return f"({lo}, {hi}]"


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print(f"=== Pulling AF moneyline + totals bets, period >= {PERIOD_FLOOR}, "
                  f"joined to score state + GAMEPLAI's own probability as of BET_DATE_UTC - {LAG_SECONDS}s ===")
            print("  (this covers ~6-7x more bets than the Q4+OT-only pass, plus a second heavy join, expect a longer run)")
            _, rows = fetch_all(cur, f"""
                WITH late_bets AS (
                    SELECT ROW_NUMBER() OVER (ORDER BY MATCH_CODE, BET_DATE_UTC) AS BET_ID,
                           MATCH_CODE, BET_DATE_UTC, MARKET_TYPE_ID, SELECTION_ID, ODDS,
                           STAKE_GBP, REVENUE_GBP, BET_PLACED_PERIOD_NUMBER, MARKET_LINE, BET_TYPE,
                           CASE
                               WHEN MARKET_TYPE_ID = 1 AND SELECTION_ID = 1 THEN 50
                               WHEN MARKET_TYPE_ID = 1 AND SELECTION_ID = 2 THEN 51
                               WHEN MARKET_TYPE_ID = 3 AND SELECTION_ID = 1 THEN 54
                               WHEN MARKET_TYPE_ID = 3 AND SELECTION_ID = 2 THEN 55
                           END AS FEED_MARKET_ID
                    FROM {DATABASE}.SHARED.CUSTOMER_REVENUE
                    WHERE SPORT_CODE = 'AF' AND MARKET_TYPE_ID IN (1, 3)
                          AND BET_PLACED_PERIOD_NUMBER >= {PERIOD_FLOOR}
                ),
                score_at_bet AS (
                    SELECT b.BET_ID, sc.PLAYER_1_SCORE_CUMULATIVE, sc.PLAYER_2_SCORE_CUMULATIVE
                    FROM late_bets b
                    JOIN {DATABASE}.SHARED.SCORE_CHANGES sc
                        ON sc.MATCH_CODE = b.MATCH_CODE
                        AND sc.FILE_TIME <= DATEADD('second', -{LAG_SECONDS}, b.BET_DATE_UTC)
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY b.BET_ID ORDER BY sc.FILE_TIME DESC) = 1
                ),
                model_prob_at_bet AS (
                    SELECT b.BET_ID, g.PROBABILITY AS MODEL_PROBABILITY
                    FROM late_bets b
                    JOIN {DATABASE}.SHARED.GAMEPLAI_STREAM g
                        ON g.MATCH_CODE = b.MATCH_CODE
                        AND g.MARKET_ID = b.FEED_MARKET_ID
                        AND g.STATUS = 'open' AND g.IS_ACTIVE = 'true' AND g.PROBABILITY > 0
                        AND g.PUBLISH_TIME <= DATEADD('second', -{LAG_SECONDS}, b.BET_DATE_UTC)
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY b.BET_ID ORDER BY g.PUBLISH_TIME DESC, g.MODIFIED_EPOCH DESC
                    ) = 1
                )
                SELECT b.BET_ID, b.MARKET_TYPE_ID, b.SELECTION_ID, b.ODDS, b.STAKE_GBP,
                       b.REVENUE_GBP, b.BET_PLACED_PERIOD_NUMBER, b.MARKET_LINE, b.BET_TYPE,
                       s.PLAYER_1_SCORE_CUMULATIVE, s.PLAYER_2_SCORE_CUMULATIVE,
                       f.PLAYER_1_SCORE AS FINAL_P1, f.PLAYER_2_SCORE AS FINAL_P2,
                       m.MODEL_PROBABILITY
                FROM late_bets b
                LEFT JOIN score_at_bet s ON s.BET_ID = b.BET_ID
                LEFT JOIN {DATABASE}.SHARED.SCORE_ENDGAME f ON f.MATCH_CODE = b.MATCH_CODE
                LEFT JOIN model_prob_at_bet m ON m.BET_ID = b.BET_ID
            """)
            print(f"  {len(rows)} bet rows pulled")

            ml_rows = []
            tot_rows = []
            for (bet_id, mtid, sel_id, odds, stake, revenue, period, mline, bet_type,
                 p1, p2, final_p1, final_p2, model_prob) in rows:
                if final_p1 is None or final_p2 is None:
                    continue
                p1 = p1 or 0
                p2 = p2 or 0
                stake = float(stake or 0)
                revenue = float(revenue or 0)
                implied = 100.0 / float(odds) if (bet_type == "Single" and odds) else None
                model_prob = float(model_prob) if model_prob is not None else None

                if mtid == 1:
                    sel_score, opp_score = (p1, p2) if sel_id == 1 else (p2, p1)
                    sel_final, opp_final = (final_p1, final_p2) if sel_id == 1 else (final_p2, final_p1)
                    margin = sel_score - opp_score
                    won = sel_final > opp_final
                    ml_rows.append((period, sel_id, margin, won, implied, model_prob, stake, revenue))
                elif mtid == 3 and mline is not None:
                    current_total = p1 + p2
                    cushion = float(mline) - current_total
                    final_total = final_p1 + final_p2
                    won = final_total > float(mline) if sel_id == 1 else final_total < float(mline)
                    tot_rows.append((period, sel_id, cushion, won, implied, model_prob, stake, revenue))

            def analyze(label, data, selection_names):
                print(f"\n=== {label}: {len(data)} bets, quantile-binned into "
                      f"{N_QUANTILE_BINS} equal-population buckets, crossed with period group and selection ===")
                cutpoints = quantile_cutpoints([d[2] for d in data], N_QUANTILE_BINS)
                if not cutpoints:
                    print("  Not enough data to bin.")
                    return

                buckets = defaultdict(lambda: {"n": 0, "stake": 0.0, "revenue": 0.0,
                                                "implied_sum": 0.0, "implied_n": 0,
                                                "model_sum": 0.0, "model_n": 0, "wins": 0})
                for period, sel_id, x, won, implied, model_prob, stake, revenue in data:
                    idx = bucket_index(x, cutpoints)
                    key = (period_group(period), idx, sel_id)
                    d = buckets[key]
                    d["n"] += 1
                    d["stake"] += stake
                    d["revenue"] += revenue
                    d["wins"] += 1 if won else 0
                    if implied is not None:
                        d["implied_sum"] += implied
                        d["implied_n"] += 1
                    if model_prob is not None:
                        d["model_sum"] += model_prob
                        d["model_n"] += 1

                header = (f"  {'PERIOD_GROUP':<12}{'RANGE':<16}{'SEL':<7}{'N':>8}{'STAKE_GBP':>13}"
                          f"{'MARGIN_%':>10}{'IMPLIED_%':>11}{'MODEL_%':>9}{'REALIZED_%':>12}{'GAP':>8}{'GAP_MODEL':>11}")
                print(header)
                for (pg, idx, sel_id) in sorted(buckets):
                    d = buckets[(pg, idx, sel_id)]
                    margin_pct = 100 * d["revenue"] / d["stake"] if d["stake"] else float("nan")
                    implied = d["implied_sum"] / d["implied_n"] if d["implied_n"] else float("nan")
                    model_pct = d["model_sum"] / d["model_n"] if d["model_n"] else float("nan")
                    realized = 100 * d["wins"] / d["n"] if d["n"] else float("nan")
                    gap = realized - implied if d["implied_n"] else float("nan")
                    gap_model = realized - model_pct if d["model_n"] else float("nan")
                    rng = bucket_label(idx, cutpoints)
                    sel_name = selection_names.get(sel_id, sel_id)
                    print(f"  {pg:<12}{rng:<16}{sel_name:<7}{d['n']:>8}{d['stake']:>13.0f}"
                          f"{margin_pct:>10.2f}{implied:>11.2f}{model_pct:>9.2f}{realized:>12.2f}{gap:>8.2f}{gap_model:>11.2f}")

                reliable = [(k, v) for k, v in buckets.items() if v["model_n"] >= 30]
                worst = sorted(
                    reliable,
                    key=lambda kv: abs(kv[1]["model_sum"] / kv[1]["model_n"] - 100 * kv[1]["wins"] / kv[1]["n"]),
                    reverse=True,
                )[:8]
                print("\n  Worst cells by |GAP_MODEL| (min 30 matched model probabilities):")
                for (pg, idx, sel_id), d in worst:
                    model_pct = d["model_sum"] / d["model_n"]
                    realized = 100 * d["wins"] / d["n"]
                    print(f"    {pg} {bucket_label(idx, cutpoints)} {selection_names.get(sel_id, sel_id)}: "
                          f"n={d['n']} model={model_pct:.2f}% realized={realized:.2f}% "
                          f"gap_model={realized - model_pct:+.2f}pp stake={d['stake']:.0f}")

            analyze("Moneyline (margin = selection's score lead)", ml_rows, ML_SELECTION_NAMES)
            analyze("Totals (cushion = points still needed to reach line)", tot_rows, TOT_SELECTION_NAMES)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
