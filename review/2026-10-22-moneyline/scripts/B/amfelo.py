"""AMFELO's starting ratings as a moneyline: raw Elo logistic, and a walk-forward fitted link a + b*diff."""
import common as C
import numpy as np, pandas as pd
from scipy.optimize import minimize

H = C.H
d = C.slim()
d["elo_x"] = (d.elo_p1 - d.elo_p2) / 400 * np.log(10)
d["amfelo_raw"] = pd.Series(H.sigmoid(d.elo_x), index=d.index).where(H.in_window(d, C.FULLWIN))


def make_link(train_days=None):
    def fit(tr):
        tr = tr[tr.elo_x.notna()]
        if train_days:
            tr = tr[tr.tday >= tr.tday.max() - train_days]
        x, y = tr.elo_x.to_numpy(), tr.y.to_numpy()

        def f(b):
            z = b[0] + b[1] * x
            r = H.sigmoid(z) - y
            return np.sum(np.logaddexp(0, z) - y * z), np.array([r.sum(), (r * x).sum()])
        return minimize(f, np.array([0.0, 1.0]), jac=True, method="BFGS").x

    def predict(b, day):
        return H.sigmoid(b[0] + b[1] * day.elo_x.to_numpy())
    return fit, predict


if __name__ == "__main__":
    cols = ["code", "start", "day", "y", "tday", "elo_x"]
    rows = []
    for td in [None, 14, 30, 60, 120]:
        f, p = make_link(td)
        pr = H.walk_forward(d[cols], f, p, *H.TUNE)
        w = H.in_window(d, H.TUNE) & d.y.notna() & pr.notna()
        m = H.metrics(d.y[w], pr[w]); m["train_days"] = td
        rows.append(m)
    r = pd.DataFrame(rows); print(r.to_string())
    best = r.sort_values("logloss").iloc[0]
    td = None if pd.isna(best.train_days) else int(best.train_days)
    print("chosen train_days", td)
    f, p = make_link(td)
    d["amfelo_link"] = H.walk_forward(d[cols], f, p, *C.FULLWIN)
    # coefficient path (weekly) for the notes
    coefs = []
    for day in sorted(d.loc[H.in_window(d, C.FULLWIN), "day"].unique())[::14]:
        tr = d[(d.start + H.RESULT_LAG <= pd.Timestamp(day)) & d.y.notna()]
        coefs.append((day, *np.round(f(tr), 3)))
    print("link a, b by fortnight:", coefs)
    H.write_preds(d, "amfelo_link", "B_amfelo_link")
    H.write_preds(d, "amfelo_raw", "B_amfelo_raw")
    for win in [H.TUNE, H.TEST, H.SEP, C.LATE]:
        print(H.report(d, ["amfelo_raw", "amfelo_link"], window=win, refs=("prod",)))
    r.to_csv(f"{C.OUT}/amfelo_tune.csv", index=False)
