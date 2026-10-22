"""Agent B shared helpers: paths, slim data, indices, event ordering."""
import os as _os
for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    _os.environ[_v] = "1"
import os, sys
import numpy as np
import pandas as pd

ML = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml"
sys.path.insert(0, ML)
import harness as H  # noqa: E402

OUT = os.path.join(ML, "out", "B")
os.makedirs(OUT, exist_ok=True)
SLIM = os.path.join(OUT, "slim.pkl")
FULLWIN = ("2026-06-01", "2026-10-19")   # preds coverage
LATE = ("2026-09-22", "2026-10-19")


def slim(refresh=False):
    """The columns B needs, with integer ids. Cached."""
    if os.path.exists(SLIM) and not refresh:
        return pd.read_pickle(SLIM)
    df = H.load(tables=True, elo=True)
    keep = ["code", "start", "day", "p1", "p2", "t1", "t2", "s1", "s2", "y", "margin", "stream", "prod",
            "rating", "elo_p1", "elo_p2", "p1_games", "p2_games", "hour", "weekday"] + \
           [c for c in df.columns if c.startswith("tbl_") and c.endswith("_p")]
    d = df[keep].copy()
    gamers = sorted(set(d.p1) | set(d.p2))
    teams = sorted(set(d.t1) | set(d.t2))
    gi = {g: i for i, g in enumerate(gamers)}
    ti = {t: i for i, t in enumerate(teams)}
    d["g1"] = d.p1.map(gi).astype(int)
    d["g2"] = d.p2.map(gi).astype(int)
    d["k1"] = d.t1.map(ti).astype(int)
    d["k2"] = d.t2.map(ti).astype(int)
    d["st"] = (d.stream.fillna(1).astype(int) - 1).clip(0, 2)
    d["tsec"] = (d.start - pd.Timestamp("2025-12-10")).dt.total_seconds()
    d["tday"] = d.tsec / 86400.0
    d.attrs["gamers"] = gamers
    d.attrs["teams"] = teams
    d.to_pickle(SLIM)
    return d


def ngamers(d):
    return int(max(d.g1.max(), d.g2.max()) + 1)


def nteams(d):
    return int(max(d.k1.max(), d.k2.max()) + 1)
