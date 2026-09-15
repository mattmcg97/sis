"""Bet-free calibration check: does GAMEPLAI's own probability match
realized outcomes across ALL settled AF matches, not just states where
someone happened to place a bet?

late_game_drilldown.py's calibration numbers are conditioned on bets
being placed, which likely skews toward states sharp bettors already
spotted as mispriced -- not a neutral sample of game states. This
script instead samples one snapshot per scoring play per match (from
SCORE_CHANGES), across the full population of settled AF matches, and
compares GAMEPLAI_STREAM's PROBABILITY at that moment to the actual
final outcome.

Anchoring to scoring plays (rather than every GAMEPLAI_STREAM tick)
matters: sampling every price tick would produce millions of highly
autocorrelated rows from the same match's smoothly-evolving win
probability, inflating apparent sample size without adding independent
information. The real sample size for calibration purposes is bounded
by the number of distinct MATCHES, not rows -- both are reported per
bucket below so that's visible rather than hidden behind a big N.
"""

import statistics as stats
from collections import defaultdict

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
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


def analyze(label, data, selection_names):
    # data: list of (period, sel_id, x, won, model_prob, match_code)
    print(f"\n=== {label}: {len(data)} scoring-play observations, "
          f"quantile-binned into {N_QUANTILE_BINS} equal-population buckets ===")
    cutpoints = quantile_cutpoints([d[2] for d in data], N_QUANTILE_BINS)
    if not cutpoints:
        print("  Not enough data to bin.")
        return

    buckets = defaultdict(lambda: {"n": 0, "matches": set(), "model_sum": 0.0, "model_n": 0, "wins": 0})
    for period, sel_id, x, won, model_prob, match_code in data:
        idx = bucket_index(x, cutpoints)
        key = (period_group(period), idx, sel_id)
        d = buckets[key]
        d["n"] += 1
        d["matches"].add(match_code)
        d["wins"] += 1 if won else 0
        if model_prob is not None:
            d["model_sum"] += model_prob
            d["model_n"] += 1

    header = (f"  {'PERIOD_GROUP':<12}{'RANGE':<16}{'SEL':<7}{'N':>8}{'MATCHES':>9}"
              f"{'MODEL_%':>9}{'REALIZED_%':>12}{'GAP_MODEL':>11}")
    print(header)
    for (pg, idx, sel_id) in sorted(buckets):
        d = buckets[(pg, idx, sel_id)]
        model_pct = d["model_sum"] / d["model_n"] if d["model_n"] else float("nan")
        realized = 100 * d["wins"] / d["n"] if d["n"] else float("nan")
        gap_model = realized - model_pct if d["model_n"] else float("nan")
        rng = bucket_label(idx, cutpoints)
        sel_name = selection_names.get(sel_id, sel_id)
        print(f"  {pg:<12}{rng:<16}{sel_name:<7}{d['n']:>8}{len(d['matches']):>9}"
              f"{model_pct:>9.2f}{realized:>12.2f}{gap_model:>11.2f}")

    reliable = [(k, v) for k, v in buckets.items() if v["model_n"] >= 30 and len(v["matches"]) >= 20]
    worst = sorted(
        reliable,
        key=lambda kv: abs(kv[1]["model_sum"] / kv[1]["model_n"] - 100 * kv[1]["wins"] / kv[1]["n"]),
        reverse=True,
    )[:8]
    print("\n  Worst cells by |GAP_MODEL| (min 30 observations, min 20 distinct matches):")
    for (pg, idx, sel_id), d in worst:
        model_pct = d["model_sum"] / d["model_n"]
        realized = 100 * d["wins"] / d["n"]
        print(f"    {pg} {bucket_label(idx, cutpoints)} {selection_names.get(sel_id, sel_id)}: "
              f"n={d['n']} matches={len(d['matches'])} model={model_pct:.2f}% "
              f"realized={realized:.2f}% gap_model={realized - model_pct:+.2f}pp")


def main():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            print("=== Counting settled AF matches (the true population for this analysis) ===")
            _, rows = fetch_all(cur, f"""
                SELECT COUNT(*) FROM {DATABASE}.SHARED.EVENT
                WHERE SPORT_CODE = 'AF' AND INPLAY_EVENT_STATUS = 'SETTLED'
            """)
            print(f"  {rows[0][0]} settled AF matches total")

            print(f"\n=== Pulling moneyline calibration: one snapshot per scoring play (period >= {PERIOD_FLOOR}), "
                  f"across ALL settled AF matches ===")
            print("  (unconditional on bets -- this may take a while)")
            _, ml_raw = fetch_all(cur, f"""
                WITH af_matches AS (
                    SELECT MATCH_CODE FROM {DATABASE}.SHARED.EVENT
                    WHERE SPORT_CODE = 'AF' AND INPLAY_EVENT_STATUS = 'SETTLED'
                ),
                score_events AS (
                    SELECT ROW_NUMBER() OVER (ORDER BY sc.MATCH_CODE, sc.FILE_TIME) AS EVENT_ID,
                           sc.MATCH_CODE, sc.PERIOD_NUMBER,
                           sc.PLAYER_1_SCORE_CUMULATIVE, sc.PLAYER_2_SCORE_CUMULATIVE, sc.FILE_TIME
                    FROM {DATABASE}.SHARED.SCORE_CHANGES sc
                    JOIN af_matches m ON m.MATCH_CODE = sc.MATCH_CODE
                    WHERE sc.PERIOD_NUMBER >= {PERIOD_FLOOR}
                ),
                model_at_event AS (
                    SELECT e.EVENT_ID, g.MARKET_ID, g.PROBABILITY
                    FROM score_events e
                    JOIN {DATABASE}.SHARED.GAMEPLAI_STREAM g
                        ON g.MATCH_CODE = e.MATCH_CODE AND g.MARKET_ID IN (50, 51)
                        AND g.STATUS = 'open' AND g.IS_ACTIVE = 'true'
                        AND g.PUBLISH_TIME >= e.FILE_TIME
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY e.EVENT_ID, g.MARKET_ID ORDER BY g.PUBLISH_TIME ASC, g.MODIFIED_EPOCH ASC
                    ) = 1
                )
                SELECT e.MATCH_CODE, e.PERIOD_NUMBER, e.PLAYER_1_SCORE_CUMULATIVE, e.PLAYER_2_SCORE_CUMULATIVE,
                       mo.MARKET_ID, mo.PROBABILITY, f.PLAYER_1_SCORE AS FINAL_P1, f.PLAYER_2_SCORE AS FINAL_P2
                FROM score_events e
                JOIN model_at_event mo ON mo.EVENT_ID = e.EVENT_ID
                JOIN {DATABASE}.SHARED.SCORE_ENDGAME f ON f.MATCH_CODE = e.MATCH_CODE
            """)
            print(f"  {len(ml_raw)} moneyline (event x market) rows pulled")

            ml_data = []
            for match_code, period, p1, p2, market_id, prob, final_p1, final_p2 in ml_raw:
                if final_p1 is None or final_p2 is None:
                    continue
                p1 = p1 or 0
                p2 = p2 or 0
                sel_id = 1 if market_id == 50 else 2
                sel_score, opp_score = (p1, p2) if sel_id == 1 else (p2, p1)
                sel_final, opp_final = (final_p1, final_p2) if sel_id == 1 else (final_p2, final_p1)
                margin = sel_score - opp_score
                won = sel_final > opp_final
                model_prob = float(prob) if prob is not None else None
                ml_data.append((period, sel_id, margin, won, model_prob, match_code))

            analyze("Moneyline (margin = selection's score lead)", ml_data, ML_SELECTION_NAMES)

            print(f"\n=== Pulling totals calibration: one snapshot per scoring play (period >= {PERIOD_FLOOR}), "
                  f"across ALL settled AF matches ===")
            print("  (unconditional on bets -- this may take a while)")
            _, tot_raw = fetch_all(cur, f"""
                WITH af_matches AS (
                    SELECT MATCH_CODE FROM {DATABASE}.SHARED.EVENT
                    WHERE SPORT_CODE = 'AF' AND INPLAY_EVENT_STATUS = 'SETTLED'
                ),
                score_events AS (
                    SELECT ROW_NUMBER() OVER (ORDER BY sc.MATCH_CODE, sc.FILE_TIME) AS EVENT_ID,
                           sc.MATCH_CODE, sc.PERIOD_NUMBER,
                           sc.PLAYER_1_SCORE_CUMULATIVE, sc.PLAYER_2_SCORE_CUMULATIVE, sc.FILE_TIME
                    FROM {DATABASE}.SHARED.SCORE_CHANGES sc
                    JOIN af_matches m ON m.MATCH_CODE = sc.MATCH_CODE
                    WHERE sc.PERIOD_NUMBER >= {PERIOD_FLOOR}
                ),
                model_at_event AS (
                    SELECT e.EVENT_ID, g.MARKET_ID, g.PROBABILITY,
                           TRY_CAST(REGEXP_SUBSTR(g.MARKET_DESCRIPTION, '[0-9]+.[0-9]+') AS FLOAT) AS LINE_VALUE
                    FROM score_events e
                    JOIN {DATABASE}.SHARED.GAMEPLAI_STREAM g
                        ON g.MATCH_CODE = e.MATCH_CODE AND g.MARKET_ID IN (54, 55)
                        AND g.STATUS = 'open' AND g.IS_ACTIVE = 'true'
                        AND g.PUBLISH_TIME >= e.FILE_TIME
                    QUALIFY ROW_NUMBER() OVER (
                        PARTITION BY e.EVENT_ID, g.MARKET_ID ORDER BY g.PUBLISH_TIME ASC, g.MODIFIED_EPOCH ASC
                    ) = 1
                )
                SELECT e.MATCH_CODE, e.PERIOD_NUMBER, e.PLAYER_1_SCORE_CUMULATIVE, e.PLAYER_2_SCORE_CUMULATIVE,
                       mo.MARKET_ID, mo.PROBABILITY, mo.LINE_VALUE,
                       f.PLAYER_1_SCORE AS FINAL_P1, f.PLAYER_2_SCORE AS FINAL_P2
                FROM score_events e
                JOIN model_at_event mo ON mo.EVENT_ID = e.EVENT_ID
                JOIN {DATABASE}.SHARED.SCORE_ENDGAME f ON f.MATCH_CODE = e.MATCH_CODE
            """)
            print(f"  {len(tot_raw)} totals (event x market) rows pulled")

            tot_data = []
            for match_code, period, p1, p2, market_id, prob, line_value, final_p1, final_p2 in tot_raw:
                if final_p1 is None or final_p2 is None or line_value is None:
                    continue
                p1 = p1 or 0
                p2 = p2 or 0
                sel_id = 1 if market_id == 54 else 2
                current_total = p1 + p2
                cushion = line_value - current_total
                final_total = final_p1 + final_p2
                won = final_total > line_value if sel_id == 1 else final_total < line_value
                model_prob = float(prob) if prob is not None else None
                tot_data.append((period, sel_id, cushion, won, model_prob, match_code))

            analyze("Totals (cushion = points still needed to reach line)", tot_data, TOT_SELECTION_NAMES)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
