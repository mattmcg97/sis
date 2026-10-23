"""Walk-forward stacks of the agents' predictions: logistic(a + sum b_k logit p_k), refitted each day on the
last WINDOW days of the members' own out-of-sample predictions (settled by the day's midnight), with a light
ridge on b. Members chosen up front (no TEST selection); reported against the best single members."""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

WINDOW = 60
RIDGE = 1.0

STACKS = {
    "S_long": ["pred_A_link_ridge", "pred_C_strength", "pred_B_mov_kal", "pred_B_glicko", "pred_D_lr4"],
    "S_long3": ["pred_A_link_ridge", "pred_B_glicko", "pred_D_lr4"],
    "S_table": ["pred_A_raw_tbl_avg", "pred_A_link_ridge", "pred_B_glicko", "pred_D_lr4"],
    "S_all": ["pred_A_raw_tbl_avg", "pred_A_link_ridge", "pred_C_strength", "pred_B_mov_kal", "pred_B_glicko",
              "pred_D_lr4", "pred_D_xgb_paired"],
}


def fit(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    pen = np.r_[0.0, np.full(X.shape[1], RIDGE)]

    def f(b):
        z = X1 @ b
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(pen * b * b)

    def g(b):
        return X1.T @ (H.sigmoid(X1 @ b) - y) + pen * b

    b0 = np.r_[0.0, np.full(X.shape[1], 1.0 / X.shape[1])]
    return minimize(f, b0, jac=g, method="L-BFGS-B").x


def run(df, cols, start="2026-07-01", end="2026-10-19"):
    out = pd.Series(np.nan, index=df.index)
    have = df[cols].notna().all(axis=1)
    L = {c: H.logit(df[c].to_numpy()) for c in cols}
    X = np.column_stack([L[c] for c in cols])
    settled = df.start + H.RESULT_LAG
    weights = []
    for d in sorted(df.loc[(df.day >= start) & (df.day < end), "day"].unique()):
        cut = pd.Timestamp(d)
        tr = (settled <= cut) & (df.start >= cut - pd.Timedelta(days=WINDOW)) & have & df.y.notna()
        if tr.sum() < 300:
            continue
        b = fit(X[tr.to_numpy()], df.y[tr].to_numpy())
        weights.append((d, *np.round(b, 3)))
        te = (df.day == d) & have
        out[te] = H.sigmoid(np.column_stack([np.ones(te.sum()), X[te.to_numpy()]]) @ b)
    return out, weights


def main():
    df = H.read_preds(H.load())
    for name, cols in STACKS.items():
        df[name], w = run(df, cols)
        print(name, "last weights (a, " + ", ".join(c[5:] for c in cols) + "):", w[-1][1:] if w else None)
        H.write_preds(df, name, name)
    singles = ["pred_A_best", "pred_A_link_ridge", "pred_A_raw_tbl_avg", "pred_C_strength", "pred_B_mov_kal",
               "pred_B_glicko", "pred_D_lr4", "pred_D_xgb_paired"]
    for w in (H.TEST, H.SEP, ("2026-09-22", "2026-10-19")):
        print(H.report(df, list(STACKS) + singles, window=w, refs=("prod", "pred_A_best")))


if __name__ == "__main__":
    main()
