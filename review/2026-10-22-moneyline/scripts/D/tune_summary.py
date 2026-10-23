"""Print tune_results.csv with each config's TUNE log loss less a reference config's (day-resampled 95%)."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml/scripts/D")
from common import *
ref = sys.argv[1] if len(sys.argv) > 1 else None
pat = sys.argv[2] if len(sys.argv) > 2 else ""
df = load()
r = pd.read_csv(os.path.join(OUT, "tune_results.csv"))
r = r[r.name.str.contains(pat)]
w = df[H.in_window(df, H.TUNE) & df.y.notna()].copy()
w["link_rm"] = load_series(df, "link_rm")
for n in r.name:
    p = pd.read_csv(os.path.join(OUT, "tune", f"{n}.csv")).set_index("code").p
    w[n] = p.reindex(w.code).to_numpy()
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
rows = []
for _, x in r.iterrows():
    d = (np.nan,) * 3
    if ref and ref in w:
        d = H.boot_diff(w, x["name"], ref, n_boot=1000)
    dl = H.boot_diff(w, x["name"], "link_rm", n_boot=1000)
    rows.append(dict(name=x["name"], ll=round(x.tune_ll, 5), slope=round(x.slope, 2), rounds=x.rounds_med,
                     d_ref=f"{d[0]:+.4f} ({d[1]:+.4f},{d[2]:+.4f})", d_link=f"{dl[0]:+.4f} ({dl[1]:+.4f},{dl[2]:+.4f})",
                     secs=int(x.secs), params=x.params if "C" in x["name"][:1] or "M" in x["name"][:1] else ""))
print(pd.DataFrame(rows).sort_values("ll").to_string(index=False))
