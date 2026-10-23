"""Item 4: is the margin worth modelling jointly with the total?
Margin expectation: the per-side ridge's (b_side_margin, walk-forward, same fit as b_tot). Total: MEAN (blend).
  - residual correlation of margin and total errors; corr of |margin error| and the total error / realized total
  - the margin's spread by fifth of the expected total and of expected drives
  - walk-forward margin variance function log Vm = a + b log(mu/35) + c log(muD/10.3) (+ d m_hat^2), and the
    moneyline it gives, P(home) = Phi(m_hat / sd_m), against a constant sd (log loss, day-resampled interval)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, special

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flib as F  # noqa: E402

MEAN = sys.argv[1] if len(sys.argv) > 1 else "blend3"
MCOL = sys.argv[2] if len(sys.argv) > 2 else "b_side_margin"
df = F.panel()
m = pd.read_pickle(os.path.join(F.OUT, "means_all.pkl"))
d = df[["code", "day", "start", "settled", "total", "margin", "y", "prod", "D"]].merge(
    m[["code", MEAN, MCOL, "drives"]], on="code")
d = d.rename(columns={MEAN: "mu", MCOL: "mh", "drives": "muD"})
d["em"] = d.margin - d.mh
d["et"] = d.total - d.mu
lines = [f"margin: {MCOL}; total: {MEAN}; drives: tuned pace ridge"]
for wn in ("TUNE", "TEST", "SEP"):
    w = F.WINDOWS[wn]
    x = d[(d.day >= w[0]) & (d.day < w[1])].dropna(subset=["em", "et", "muD"])
    c1 = np.corrcoef(x.em, x.et)[0, 1]
    c2 = np.corrcoef(np.abs(x.em), x.et)[0, 1]
    c3 = np.corrcoef(x.em ** 2, x.total)[0, 1]
    # sign-aware: does the favourite's edge grow when the total runs high?
    fav = np.sign(x.mh)
    c4 = np.corrcoef(fav * x.em, x.et)[0, 1]
    lines.append(f"\n{wn} ({len(x):,}): corr(margin err, total err) {c1:+.3f}; corr(|margin err|, total err) {c2:+.3f}; "
                 f"corr(margin err^2, realized total) {c3:+.3f}; corr(favourite's margin err, total err) {c4:+.3f}")
    lines.append(f"  margin RMSE {np.sqrt((x.em ** 2).mean()):.3f}, total RMSE {np.sqrt((x.et ** 2).mean()):.3f}")
    for key, lab in (("mu", "expected total"), ("muD", "expected drives")):
        q = pd.qcut(x[key], 5, labels=False)
        g = x.groupby(q).agg(k=(key, "mean"), sd_m=("em", "std"), sd_t=("et", "std"), absm=("margin", lambda s: s.abs().mean()))
        lines.append(f"  by fifth of {lab}: " + "  ".join(f"{r.k:5.1f}: m sd {r.sd_m:5.2f} t sd {r.sd_t:5.2f}"
                                                            for r in g.itertuples()))
    q = pd.qcut(x.total, 5, labels=False)
    g = x.groupby(q).agg(k=("total", "mean"), sd_m=("em", "std"))
    lines.append("  by fifth of the REALIZED total: " + "  ".join(f"{r.k:5.1f}: m sd {r.sd_m:5.2f}" for r in g.itertuples()))

# walk-forward margin variance function -> moneyline
FORMS = {"const": [], "mu": ["lmu"], "D": ["lD"], "muD": ["lmu", "lD"], "muD_m2": ["lmu", "lD", "m2"]}
d["lmu"] = np.log(d.mu / 35)
d["lD"] = np.log(d.muD / 10.3)
d["m2"] = (d.mh / 10) ** 2


def fit_var(tr, cols):
    Z = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
    r2 = tr.em.values ** 2
    nll = lambda b: 0.5 * np.sum(Z @ b + r2 * np.exp(-(Z @ b)))
    grad = lambda b: 0.5 * Z.T @ (1 - r2 * np.exp(-(Z @ b)))
    b0 = np.zeros(Z.shape[1])
    b0[0] = np.log(r2.mean())
    return optimize.minimize(nll, b0, jac=grad, method="BFGS").x


ok = d.dropna(subset=["mh", "mu", "muD"])
pred = {f: np.full(len(d), np.nan) for f in FORMS}
coef = {}
for k, day in enumerate(F.days_between(d, "2026-06-01", "2026-10-19")):
    cut = pd.Timestamp(day)
    if k % 7 == 0:
        tr = ok[(ok.settled <= cut) & (ok.start >= cut - pd.Timedelta(days=60)) & ok.margin.notna()]
        bs = {f: fit_var(tr, c) for f, c in FORMS.items()}
        coef[day] = bs
    rows = ok[ok.day == day]
    for f, c in FORMS.items():
        Z = np.column_stack([np.ones(len(rows))] + [rows[cc].values for cc in c])
        sd = np.sqrt(np.exp(Z @ bs[f]))
        pred[f][rows.index.values] = special.ndtr(rows.mh.values / sd)
for f in FORMS:
    d[f"ml_{f}"] = pred[f]
lines.append("\nmargin variance function at 10-12 (log Vm = a + b log(mu/35) + c log(muD/10.3) [+ d (m/10)^2]):")
last = list(coef)[-1]
for f, b in coef[last].items():
    lines.append(f"  {f:8s} " + " ".join(f"{v:+.3f}" for v in b))
for wn in ("TUNE", "TEST", "SEP"):
    lines.append(F.H.report(d, [f"ml_{f}" for f in FORMS], window=F.WINDOWS[wn], refs=("ml_const", "prod")))
txt = "\n".join(lines)
print(txt)
open(os.path.join(F.OUT, "margin_joint.txt"), "w").write(txt + "\n")
