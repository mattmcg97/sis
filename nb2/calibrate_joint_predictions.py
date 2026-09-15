#!/usr/bin/env python3
"""Calibrate the "Mid-Year" joint NB2 model (AMF_Mid_Year_Model.docx) against
reality, using its own logged pre-match predictions.

NB2_joint_schedule_predictions.csv is not historical training data -- it is
the model's actual live output for real matches scheduled Thu 10 - Fri 11
Sep 2026, i.e. genuine out-of-sample, before-the-fact predictions (exactly
what the doc's Elo-vs-model logloss comparison is built on). Those matches
are now 4-5 days old, so by the time this runs they should be settled. This
script pulls the real final scores from Snowflake and checks:

  - Moneyline:  P1_Fair_ML_Prob        vs actual P1 win
  - Totals:     Fair_Over_Prob_At_Line vs actual (P1+P2) > Fair_Total_Line
  - Handicap:   Fair_P1_Cover_Prob_At_Line vs actual margin+handicap > 0
  - Continuous: Pred_P1_Points/Pred_Total_Points vs actual, for bias/RMSE

Sample size here is small (~80 matches, one live day), so treat this as a
first read rather than a definitive verdict -- rerun as more days of logged
predictions accumulate.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "analysis"))

import numpy as np
import pandas as pd

from snowflake_connect import get_connection

DATABASE = "SIS_PROD_CG_CURATED"
CSV_PATH = Path(__file__).parent / "NB2_joint_schedule_predictions.csv"


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def load_predictions():
    df = pd.read_csv(CSV_PATH)
    df = df[df["Prediction_Status"] == "OK"].copy()
    df = df.dropna(subset=["Match Id"]).copy()
    df = df[df["Match Id"] != "-"].copy()
    return df


def brier_logloss(p, y):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-9
    pc = np.clip(p, eps, 1 - eps)
    logloss = float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc)))
    return brier, logloss


def calibration_table(df, prob_col, outcome_col, n_bins=8):
    d = df[[prob_col, outcome_col]].dropna().copy()
    n_bins = min(n_bins, d[prob_col].nunique())
    if n_bins < 2:
        return None
    d["bucket"] = pd.qcut(d[prob_col], n_bins, duplicates="drop")
    table = d.groupby("bucket", observed=True).agg(
        n=(outcome_col, "size"),
        predicted=(prob_col, "mean"),
        realized=(outcome_col, "mean"),
    ).reset_index()
    table["gap_pp"] = 100 * (table["realized"] - table["predicted"])
    return table


def main():
    preds = load_predictions()
    match_codes = preds["Match Id"].unique().tolist()
    print(f"Loaded {len(preds)} live 'OK' predictions covering {len(match_codes)} matches "
          f"from {CSV_PATH.name}")
    print(f"Date range in file: {preds['Date'].min()} -> {preds['Date'].max()}")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ", ".join(["%s"] * len(match_codes))
            _, rows = fetch_all(cur, f"""
                SELECT e.MATCH_CODE, e.INPLAY_EVENT_STATUS, s.PLAYER_1_SCORE, s.PLAYER_2_SCORE
                FROM {DATABASE}.SHARED.EVENT e
                LEFT JOIN {DATABASE}.SHARED.SCORE_ENDGAME s ON s.MATCH_CODE = e.MATCH_CODE
                WHERE e.MATCH_CODE IN ({placeholders})
            """, tuple(match_codes))
    finally:
        conn.close()

    actual = pd.DataFrame(rows, columns=["MATCH_CODE", "STATUS", "P1_ACTUAL", "P2_ACTUAL"])
    settled = actual[actual["STATUS"] == "SETTLED"].dropna(subset=["P1_ACTUAL", "P2_ACTUAL"]).copy()
    print(f"\n{len(settled)}/{len(match_codes)} matches settled with a final score so far.")
    if len(actual) > len(settled):
        unsettled = actual[~actual["MATCH_CODE"].isin(settled["MATCH_CODE"])]
        status_counts = unsettled["STATUS"].value_counts(dropna=False)
        print("Not yet usable, by status:")
        print(status_counts.to_string())

    df = preds.merge(settled, left_on="Match Id", right_on="MATCH_CODE", how="inner")
    print(f"\n{len(df)} predictions joined to a settled result -- proceeding with those.")
    if len(df) == 0:
        print("Nothing settled yet -- rerun this script again in a day or two.")
        return

    df["actual_total"] = df["P1_ACTUAL"] + df["P2_ACTUAL"]
    df["actual_margin"] = df["P1_ACTUAL"] - df["P2_ACTUAL"]
    df["p1_won"] = (df["actual_margin"] > 0).astype(float)
    tie_mask = df["actual_margin"] == 0
    if tie_mask.any():
        print(f"({tie_mask.sum()} regulation ties in the settled sample -- scored as 0.5 for ML)")
        df.loc[tie_mask, "p1_won"] = 0.5

    # --- Moneyline ---
    print("\n" + "=" * 70)
    print("MONEYLINE  (P1_Fair_ML_Prob vs actual P1 win)")
    print("=" * 70)
    brier, logloss = brier_logloss(df["P1_Fair_ML_Prob"], df["p1_won"].round())
    print(f"Brier: {brier:.4f}   Log loss: {logloss:.4f}   (0.25 Brier = uninformative 50/50 baseline)")
    table = calibration_table(df, "P1_Fair_ML_Prob", "p1_won")
    if table is not None:
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Totals ---
    print("\n" + "=" * 70)
    print("TOTALS  (Fair_Over_Prob_At_Line vs actual total > Fair_Total_Line)")
    print("=" * 70)
    df["total_push"] = df["actual_total"] == df["Fair_Total_Line"]
    tot = df[~df["total_push"]].copy()
    print(f"({df['total_push'].sum()} pushes excluded, {len(tot)} remaining)")
    tot["over_hit"] = (tot["actual_total"] > tot["Fair_Total_Line"]).astype(float)
    brier, logloss = brier_logloss(tot["Fair_Over_Prob_At_Line"], tot["over_hit"])
    print(f"Brier: {brier:.4f}   Log loss: {logloss:.4f}")
    table = calibration_table(tot, "Fair_Over_Prob_At_Line", "over_hit")
    if table is not None:
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Handicap ---
    print("\n" + "=" * 70)
    print("HANDICAP  (Fair_P1_Cover_Prob_At_Line vs actual margin + Fair_P1_Handicap > 0)")
    print("=" * 70)
    df["cover_margin"] = df["actual_margin"] + df["Fair_P1_Handicap"]
    df["handicap_push"] = df["cover_margin"] == 0
    hcap = df[~df["handicap_push"]].copy()
    print(f"({df['handicap_push'].sum()} pushes excluded, {len(hcap)} remaining)")
    hcap["p1_covered"] = (hcap["cover_margin"] > 0).astype(float)
    brier, logloss = brier_logloss(hcap["Fair_P1_Cover_Prob_At_Line"], hcap["p1_covered"])
    print(f"Brier: {brier:.4f}   Log loss: {logloss:.4f}")
    table = calibration_table(hcap, "Fair_P1_Cover_Prob_At_Line", "p1_covered")
    if table is not None:
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Continuous score error ---
    print("\n" + "=" * 70)
    print("CONTINUOUS SCORE ERROR")
    print("=" * 70)
    total_err = df["actual_total"] - df["Pred_Total_Points"]
    margin_err = df["actual_margin"] - df["Pred_P1_Margin"]
    print(f"Total:  mean predicted={df['Pred_Total_Points'].mean():.2f}  mean actual={df['actual_total'].mean():.2f}  "
          f"mean error={total_err.mean():+.3f}  RMSE={np.sqrt((total_err ** 2).mean()):.3f}  "
          f"corr={np.corrcoef(df['Pred_Total_Points'], df['actual_total'])[0, 1]:.3f}")
    print(f"Margin: mean predicted={df['Pred_P1_Margin'].mean():.2f}  mean actual={df['actual_margin'].mean():.2f}  "
          f"mean error={margin_err.mean():+.3f}  RMSE={np.sqrt((margin_err ** 2).mean()):.3f}  "
          f"corr={np.corrcoef(df['Pred_P1_Margin'], df['actual_margin'])[0, 1]:.3f}")

    print(f"\nMean predicted margin SD: {df['Pred_Margin_SD'].mean():.2f}   "
          f"realized |margin error| mean: {margin_err.abs().mean():.2f}")
    print(f"Mean predicted total SD:  {df['Pred_Total_SD'].mean():.2f}   "
          f"realized |total error| mean:  {total_err.abs().mean():.2f}")
    print("(if realized |error| is well below the predicted SD across the board, the model may be overstating uncertainty; "
          "well above it, understating -- either is only indicative at this sample size)")

    # --- By stream, since the doc flags stream-level scoring differences ---
    if df["Model_Stream"].nunique() > 1:
        print("\n" + "=" * 70)
        print("MONEYLINE BRIER/LOGLOSS BY STREAM")
        print("=" * 70)
        for stream, g in df.groupby("Model_Stream"):
            b, ll = brier_logloss(g["P1_Fair_ML_Prob"], g["p1_won"].round())
            print(f"  Stream {stream:>4}: n={len(g):>3}  Brier={b:.4f}  LogLoss={ll:.4f}")


if __name__ == "__main__":
    main()
