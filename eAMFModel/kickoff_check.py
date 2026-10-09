"""One kick-off simulated many times, its final totals against the real ones.

    python -m eAMFModel kickoff-dist --model v12_1001 --means 15.5,15.5 \
        --history eAMFCalibrator/out_oct/match_history.csv --since 2026-09-07

The sim plays the game from kick-off as v12 prices a match (v12.price_kickoff) with both sides'
expected points set to --means and league-average players. The real side is every settled match
in the history between --since and --until. That mixes strong and weak match-ups, so it is wider
than any one game: compare where the spikes are and how sharp, more than the tails.
"""

import csv
import datetime as dt

import numpy as np

from . import players, v12, v12_stream


def sim_totals(model_dir, means, n_paths=20000, seed=0):
    """The probability of each final total from one kick-off with these expected points."""
    tables_path, grid_path = v12_stream.model_paths(model_dir)
    tables = v12.sim.Tables.load(tables_path)
    grid = v12.PriorGrid.load(grid_path)
    prof = (players.Profile(), players.Profile())
    theta = v12.prior_theta(grid, means, prof)
    _, total = v12.price_kickoff(tables, theta, v12.Variant("v12"), prof, n_paths,
                                 np.random.default_rng(seed), seed=seed)
    return total


def real_totals(history_path, since=None, until=None):
    """The share of each final total among the settled matches in [since, until)."""
    counts = np.zeros(v12.TOTAL_MAX + 1)
    with open(history_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            when = (r.get("SCHEDULED_START_TIME_UTC") or "")[:10]
            if (since and when < since) or (until and when >= until):
                continue
            try:
                t = int(float(r["PLAYER_1_FINAL_SCORE"])) + int(float(r["PLAYER_2_FINAL_SCORE"]))
            except (KeyError, TypeError, ValueError):
                continue
            counts[min(t, v12.TOTAL_MAX)] += 1
    n = int(counts.sum())
    return (counts / n if n else counts), n


def _moments(pmf):
    k = np.arange(len(pmf))
    mean = float((k * pmf).sum())
    return mean, float(np.sqrt(((k - mean) ** 2 * pmf).sum()))


def report(sim, real, n_real, means, lo=10, hi=60):
    """Lines: each total side by side, then the summary."""
    out = []
    say = out.append
    ms, ss = _moments(sim)
    mr, sr = _moments(real)
    say(f"Kick-off simulated at {means[0]:g} / {means[1]:g} expected points, against {n_real:,} real matches")
    say(f"  mean total  sim {ms:5.1f}  real {mr:5.1f}")
    say(f"  sd          sim {ss:5.1f}  real {sr:5.1f}   (real mixes every match-up, so is wider)")
    say("")
    say(f"{'total':>5}{'sim':>8}{'real':>8}{'diff':>8}   sim # / real |")
    top = max(sim[lo:hi + 1].max(), real[lo:hi + 1].max())
    for t in range(lo, hi + 1):
        s, r = sim[t], real[t]
        bar_s, bar_r = round(40 * s / top), round(40 * r / top)
        bar = "".join("#" if i < bar_s else " " for i in range(max(bar_s, bar_r)))
        bar = "".join("|" if i == bar_r - 1 else c for i, c in enumerate(bar))
        say(f"{t:>5}{100 * s:7.1f}%{100 * r:7.1f}%{100 * (s - r):+7.1f}   {bar}")
    say("")
    ladder = [17, 24, 31, 38, 45]
    say("TDs and one field goal (17, 24, 31, 38, 45): sim "
        f"{100 * sim[ladder].sum():.1f}%, real {100 * real[ladder].sum():.1f}%; "
        f"the share at 31 itself: sim {100 * sim[31]:.1f}%, real {100 * real[31]:.1f}%")
    for line in (30.5, 34.5, 38.5):
        say(f"  over {line}: sim {100 * sim[int(line) + 1:].sum():.1f}%  real "
            f"{100 * real[int(line) + 1:].sum():.1f}%")
    return out


def run(model_dir, means, history_path, since=None, until=None, n_paths=20000, seed=0):
    """Simulate, read the history, print and return the report."""
    sim = sim_totals(model_dir, means, n_paths, seed)
    real, n = real_totals(history_path, since, until)
    lines = report(sim, real, n, means)
    print("\n".join(lines))
    return lines


def default_since(days=30):
    """The start of the last `days` days, as YYYY-MM-DD."""
    return (dt.date.today() - dt.timedelta(days=days)).isoformat()
