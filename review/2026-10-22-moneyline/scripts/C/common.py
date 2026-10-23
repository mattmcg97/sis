"""Agent C shared bits: the per-side stat observations off xgb/data/drive_stats.csv, aligned to the
harness's matches (drive_stats' home is PLAYER_1; joined by match code)."""

import os
import sys

import numpy as np
import pandas as pd

ML = "/tmp/claude-0/-home-user-sis/9e2e3773-70c3-55ea-8c3a-69bd3c7055ad/scratchpad/ml"
sys.path.insert(0, ML)
import harness as H  # noqa: E402

ROOT = "/home/user/sis"
DRIVES = os.path.join(ROOT, "xgb/data/drive_stats.csv")
OUT = os.path.join(ML, "out", "C")
os.makedirs(OUT, exist_ok=True)

# stat -> (numerator, denominator); denominator None = per match. "points" is the side's final score.
# Signs: + means more of it is good for the offence (used only to orient tables, not the models).
STATS = {
    "pts":       ("points", None, +1),        # points a match (the site's points for/against)
    "ppd":       ("points", "drives", +1),    # points a drive
    "td":        ("td", "drives", +1),        # touchdowns a drive
    "fg":        ("fg", "drives", 0),         # field goals a drive
    "punt":      ("punt", "drives", -1),
    "turnover":  ("turnover", "drives", -1),  # interceptions/fumbles (+ defensive TDs) a drive
    "downs":     ("downs", "drives", -1),     # turnovers on downs a drive
    "rz":        ("rz_trips", "drives", +1),  # red-zone trips a drive
    "rz_td":     ("rz_td", "rz_trips", +1),   # red-zone TD %
    "plays":     ("drive_plays", "drives", 0),
    "start":     ("start_field", "drives", +1),  # field position: where drives start
    "ypp":       ("yards", "gains", +1),      # yards a play
    "success":   ("success", "snaps", +1),    # line to gain (or TD) a snap
    "big20":     ("big20", "snaps", +1),
    "big40":     ("big40", "snaps", +1),
    "loss":      ("loss", "snaps", -1),
    "third":     ("third_conv", "third_n", +1),
    "fourth":    ("fourth_conv", "fourth_n", +1),
    "drives":    ("drives", None, 0),         # drives a match (pace)
    "pace":      ("seconds", "snaps", 0),     # clock a snap
    "tds_pm":    ("td", None, +1),            # touchdowns a match (site)
    "yds_pm":    ("yards", None, +1),         # total offence a match (site)
    "fd_pm":     ("success", None, +1),       # snaps reaching the line to gain a match (~ first downs + TDs)
    "to_snap":   ("turnover", "snaps", -1),   # turnovers a snap
}
STAT_NAMES = list(STATS)


def load_drives():
    d = pd.read_csv(DRIVES)
    d = d.rename(columns={"match_code": "code"})
    return d


def side_obs(df, d=None):
    """Per match (df's row order): for each stat, the home offence's and the away offence's observed
    numerator and denominator, NaN where the match has no play-by-play. Returns dict of arrays
    {stat: (y_home, n_home, y_away, n_away)} with y the rate (num/den) and n the denominator."""
    if d is None:
        d = load_drives()
    m = df[["code", "s1", "s2"]].merge(d, on="code", how="left")
    assert len(m) == len(df)
    has = m["h_drives"].notna().to_numpy()
    out = {}
    for k, (num, den, _) in STATS.items():
        res = []
        for tag, pts in (("h", "s1"), ("a", "s2")):
            nu = m[pts].to_numpy(float) if num == "points" else m[f"{tag}_{num}"].to_numpy(float)
            de = np.ones(len(m)) if den is None else m[f"{tag}_{den}"].to_numpy(float)
            nu = np.where(has, nu, np.nan)
            de = np.where(has, de, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                y = np.where(de > 0, nu / de, np.nan)
            de = np.where(de > 0, de, np.nan)
            res += [y, de]
        out[k] = tuple(res)
    return out, has
