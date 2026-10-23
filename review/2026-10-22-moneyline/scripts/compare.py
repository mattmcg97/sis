"""Every agent's walk-forward predictions (preds/*.csv) side by side, on common matches, per window.

    python compare.py [--window TEST|SEP|LATE|TUNE] [--refs prod,pred_A_best] [--only A_,B_]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402

WINDOWS = {"TUNE": H.TUNE, "TEST": H.TEST, "SEP": H.SEP, "LATE": ("2026-09-22", "2026-10-19")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="TEST")
    ap.add_argument("--refs", default="prod")
    ap.add_argument("--only", default="")
    ap.add_argument("--min-cover", type=float, default=0.9,
                    help="leave out a prediction pricing under this share of the window's matches")
    a = ap.parse_args()
    df = H.read_preds(H.load())
    w = WINDOWS[a.window]
    inw = H.in_window(df, w) & df.y.notna()
    cols = [c for c in df if c.startswith("pred_")]
    if a.only:
        cols = [c for c in cols if any(c[5:].startswith(k) for k in a.only.split(","))]
    cover = {c: float(df.loc[inw, c].notna().mean()) for c in cols}
    keep = [c for c in cols if cover[c] >= a.min_cover]
    dropped = sorted(set(cols) - set(keep))
    if dropped:
        print("left out (coverage): " + ", ".join(f"{c[5:]} {cover[c]:.0%}" for c in dropped))
    refs = tuple(r for r in a.refs.split(",") if r)
    print(H.report(df, keep + [r for r in refs if r not in keep and r != "prod"], window=w, refs=refs))


if __name__ == "__main__":
    main()
