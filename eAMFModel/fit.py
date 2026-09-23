"""Where the structural constants in params.py come from.

`finals`: maximum likelihood on real AF final scores (nb2/AMFELO.csv,
23,952 matches) under the possession model the pricer uses:

    drives in the game   ~ normal(2m, sd), rounded, split alternately
    each drive           0, 3 (FG), or a TD worth 6 / 7 / 8
    regulation tie       overtime, won by 3 (share ot_fg) or 6

Seven parameters, fitted by Nelder-Mead on the exact likelihood of every
(home, away) final. The result (m = 4.80, sd = 2.14, TD 45.1%, FG 10.3%,
6-only 5.1%, two-point 4.3%, ot_fg at its bound of 1) is what params.py
carries; the per-drive rates agree with the 46% / 11% read directly off the
drives in directional_pairs.csv, which is the check that a "drive" here is
a real possession.

A negative binomial drive count was tried first (the obvious choice); it
cannot go below Poisson dispersion, so it overstated the spread of the
total (sd 14.8 against 12.9 in the data) and lost 563 log-likelihood
points to the normal count.

Takes a few minutes in pure Python:  python -m eAMFModel fit-finals
"""

import collections
import csv
import math
import os

DEFAULT_FINALS = os.path.join(os.path.dirname(__file__), "..", "nb2", "AMFELO.csv")
MAX_POINTS = 110
START = [4.66, math.log(1.5), 0.4645, 0.1066, 0.05, 0.043, 0.9]
STEPS = [0.5, 0.5, 0.05, 0.03, 0.02, 0.02, 0.1]
NAMES = ["drives_per_team", "log_count_sd", "p_td", "p_fg", "q6", "q8", "ot_fg"]


def load_finals(path=DEFAULT_FINALS, sport="AF"):
    out = collections.Counter()
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            if r.get("SPORT_CODE") != sport or r["PLAYER_1_FINAL_SCORE"] in ("", None):
                continue
            out[(int(float(r["PLAYER_1_FINAL_SCORE"])),
                 int(float(r["PLAYER_2_FINAL_SCORE"])))] += 1
    return out


def _convolve(a, b):
    out = [0.0] * min(MAX_POINTS, len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x < 1e-15:
            continue
        for j, y in enumerate(b):
            k = i + j
            if k >= MAX_POINTS:
                break
            out[k] += x * y
    return out


def joint(theta):
    """{(home, away): probability} under the model, or None if out of bounds."""
    m, log_sd, p_td, p_fg, q6, q8, ot_fg = theta
    if not (2 < m < 20 and 0 < p_td and 0 < p_fg and p_td + p_fg < 0.95
            and 0 <= q6 < 0.3 and 0 <= q8 < 0.3 and 0 <= ot_fg <= 1):
        return None
    drive = [0.0] * 9
    drive[0] = 1 - p_td - p_fg
    drive[3] = p_fg
    drive[6] = p_td * q6
    drive[7] = p_td * (1 - q6 - q8)
    drive[8] = p_td * q8
    powers = [[1.0]]
    for _ in range(30):
        powers.append(_convolve(powers[-1], drive))
    sd = math.exp(log_sd)
    weights = {k: math.exp(-0.5 * ((k - 2 * m) / sd) ** 2) for k in range(2, 40)}
    total = sum(weights.values())
    counts = {k: w / total for k, w in weights.items() if w / total > 1e-12}
    out = collections.defaultdict(float)
    for n, pn in counts.items():
        hi, lo = (n + 1) // 2, n // 2
        for nh, na in ((hi, lo), (lo, hi)):
            a, b = powers[min(nh, 30)], powers[min(na, 30)]
            for i, x in enumerate(a):
                if x < 1e-12:
                    continue
                for j, y in enumerate(b):
                    if y >= 1e-12:
                        out[(i, j)] += 0.5 * pn * x * y
    for k in range(MAX_POINTS):
        tie = out.pop((k, k), 0.0)
        if tie:
            for d, w in ((3, ot_fg), (6, 1 - ot_fg)):
                out[(k + d, k)] += 0.5 * tie * w
                out[(k, k + d)] += 0.5 * tie * w
    return out


def neg_log_likelihood(theta, finals):
    probs = joint(theta)
    if probs is None:
        return 1e18
    return -sum(n * math.log(max(probs.get(k, 0.0), 1e-12)) for k, n in finals.items())


def nelder_mead(f, x0, steps, iterations=400, report=None):
    pts = [list(x0)] + [[x0[j] + (steps[j] if j == i else 0) for j in range(len(x0))]
                        for i in range(len(x0))]
    vals = [f(p) for p in pts]
    for it in range(iterations):
        order = sorted(range(len(pts)), key=lambda i: vals[i])
        pts, vals = [pts[i] for i in order], [vals[i] for i in order]
        c = [sum(p[j] for p in pts[:-1]) / (len(pts) - 1) for j in range(len(x0))]
        xr = [c[j] + (c[j] - pts[-1][j]) for j in range(len(x0))]
        fr = f(xr)
        if fr < vals[0]:
            xe = [c[j] + 2 * (c[j] - pts[-1][j]) for j in range(len(x0))]
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = [c[j] + 0.5 * (pts[-1][j] - c[j]) for j in range(len(x0))]
            fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, len(pts)):
                    pts[i] = [pts[0][j] + 0.5 * (pts[i][j] - pts[0][j]) for j in range(len(x0))]
                    vals[i] = f(pts[i])
        if report and it % 50 == 0:
            report(it, vals[0], pts[0])
    best = min(range(len(pts)), key=lambda i: vals[i])
    return pts[best], vals[best]


def fit_finals(path=DEFAULT_FINALS, iterations=400, verbose=True):
    finals = load_finals(path)
    report = None
    if verbose:
        def report(it, value, point):
            print(f"  iter {it:>3}  nll {value:,.1f}  "
                  + "  ".join(f"{n}={v:.4f}" for n, v in zip(NAMES, point)), flush=True)
    theta, value = nelder_mead(lambda t: neg_log_likelihood(t, finals), START, STEPS,
                               iterations, report)
    return dict(zip(NAMES, theta)), value, sum(finals.values())


def moments(theta_dict):
    """Mean and sd of total and margin, and the score correlation, implied."""
    probs = joint([theta_dict[n] for n in NAMES])
    def stats(f):
        mu = sum(f(k) * p for k, p in probs.items())
        return mu, math.sqrt(sum((f(k) - mu) ** 2 * p for k, p in probs.items()))
    total = stats(lambda k: k[0] + k[1])
    margin = stats(lambda k: k[0] - k[1])
    home = stats(lambda k: k[0])
    away = stats(lambda k: k[1])
    cov = sum((k[0] - home[0]) * (k[1] - away[0]) * p for k, p in probs.items())
    return {"total": total, "margin": margin, "corr": cov / (home[1] * away[1])}


def scoring_curve(playover_path, bucket_seconds=30, quarter_seconds=240):
    """Cumulative share of regulation scoring by game time, off PLAY_OVER
    snapshots (eAMFCalibrator scouting's export).

    Returns ([F(0), F(b), F(2b), ..., F(960)], matches): F(t) is the mean
    share of a match's regulation points scored by game second t. Points are
    read as the change in the scoreboard between consecutive PLAY_OVERs, so
    they land in the bucket of the PLAY_OVER that first shows them.
    """
    by_match = collections.defaultdict(list)
    with open(playover_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by_match[r["match_code"]].append(r)
    regulation = 4 * quarter_seconds
    buckets = regulation // bucket_seconds
    points = [0.0] * buckets
    matches = 0
    for rows in by_match.values():
        rows.sort(key=lambda r: int(r["message"]))
        previous = 0
        seen = False
        for r in rows:
            if not r["period"] or not r["clock_seconds"]:
                continue
            period = int(r["period"])
            if period > 4:
                continue
            t = (period - 1) * quarter_seconds + (quarter_seconds - float(r["clock_seconds"]))
            b = min(buckets - 1, max(0, int(t // bucket_seconds)))
            score = int(r["score_p1"] or 0) + int(r["score_p2"] or 0)
            points[b] += score - previous
            previous = score
            seen = True
        matches += seen
    total = sum(points)
    curve = [0.0]
    for p in points:
        curve.append(curve[-1] + p / total)
    return curve, matches
