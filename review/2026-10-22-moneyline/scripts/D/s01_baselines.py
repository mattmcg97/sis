"""Simple baselines and the offset bases: (1) a walk-forward logistic link on rating_margin, training window
chosen on TUNE; (2) a time-decayed ridge strength margin (half-life, lambda chosen on TUNE) through a
walk-forward logistic link fitted on its previous 60 days' out-of-sample margins."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *

log = Log(os.path.join(OUT, "s01_baselines.log"))
df = load()
res = {}
for td in (60, 120, 240, None):
    p = wf_link(df, "rating_margin", RUN_FROM, RUN_TO, train_days=td)
    df[f"lk{td}"] = p
    res[f"lk{td}"] = ll(df, f"lk{td}", H.TUNE)
    log(f"link rating_margin train_days={td}: TUNE ll {res[f'lk{td}'][0]:.5f} n={res[f'lk{td}'][1]}")
best_lk = min((k for k in res if k.startswith("lk")), key=lambda k: res[k][0])
log(f"chosen link window: {best_lk}")

for team in (False,):
    for hl in (10, 20, 45, 90):
        for lam in (2, 5, 20):
            m = wf_ridge_strength(df, RUN_FROM, RUN_TO, half_life=hl, lam=lam, train_days=None, team=team)
            p = wf_on_oos(df, m, "2026-05-15", RUN_TO, kind="link", hist_days=60)
            k = f"rg_hl{hl}_l{lam}" + ("_t" if team else "")
            df[k + "_m"] = m
            df[k] = p
            res[k] = ll(df, k, H.TUNE)
            log(f"ridge hl={hl} lam={lam} team={team}: TUNE ll {res[k][0]:.5f} n={res[k][1]}")
best_rg = min((k for k in res if k.startswith("rg")), key=lambda k: res[k][0])
log(f"chosen ridge: {best_rg}")
# the chosen ones over the whole span (offset models need walk-forward bases on their training rows too)
td = {"lk60": 60, "lk120": 120, "lk240": 240, "lkNone": None}[best_lk]
hl, lam = best_rg.split("_")[1][2:], best_rg.split("_")[2][1:]
df["link_rm"] = wf_link(df, "rating_margin", "2025-12-01", RUN_TO, train_days=td)
df["ridge_m"] = wf_ridge_strength(df, "2025-12-01", RUN_TO, half_life=float(hl), lam=float(lam))
df["ridge_p"] = wf_on_oos(df, df.ridge_m, "2025-12-01", RUN_TO, kind="link", hist_days=60)
save_series(df, df.link_rm, "link_rm")
save_series(df, df.ridge_m, "ridge_margin")
save_series(df, df.ridge_p, "ridge_p")
log(f"full-span coverage: link {df.link_rm.notna().mean():.4f}, ridge {df.ridge_p.notna().mean():.4f} "
    f"(from {df.loc[df.ridge_p.notna(), 'day'].min()})")
for w in (H.TUNE, H.TEST, H.SEP):
    log(H.report(df, ["link_rm", "ridge_p"], window=w, refs=("prod", "link_rm")))
