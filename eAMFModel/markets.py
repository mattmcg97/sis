"""The markets off a simulation: final scores read as margin and total distributions, the lines a book sets
on them (even, key numbers, held, anchored to prod's) and each market's chance at a line. Shared by every
version's stream, report and trader (v14 re-exports it; v2.1 reads it directly).

Distributions: the margin's over -MARGIN_MAX..MARGIN_MAX (index k is a margin of k - MARGIN_MAX, home's
points less away's), the total's over 0..TOTAL_MAX. Markets by prod's ids: 50 / 51 the home / away
moneyline, 52 / 53 home / away on the handicap, 54 / 55 over / under on the total."""

import multiprocessing as mp

import numpy as np

MARGIN_MAX = 100
TOTAL_MAX = 160


def _distributions(home, away):
    """Margin and total distributions from simulated final scores."""
    m = np.clip(home - away, -MARGIN_MAX, MARGIN_MAX) + MARGIN_MAX
    t = np.clip(home + away, 0, TOTAL_MAX)
    mp_ = np.bincount(m, minlength=2 * MARGIN_MAX + 1) / len(m)
    tp = np.bincount(t, minlength=TOTAL_MAX + 1) / len(t)
    return mp_, tp


def _graded(side, at):
    """The chance of a side once a push is taken out."""
    rest = 1 - at
    return np.where(rest > 1e-12, side / np.maximum(1e-12, rest), 0.5)


def even_line(pmf, offset):
    """The half-point line nearest to even money."""
    cdf = np.cumsum(pmf)[:-1]
    gap = np.abs(cdf - 0.5)
    gap = np.where(cdf <= 1e-12, np.inf, gap) if (cdf > 1e-12).any() else gap
    best = np.flatnonzero(gap <= gap.min() + 1e-12)
    x = np.arange(len(pmf)) - offset
    mean = float((pmf * x).sum() / max(1e-300, pmf.sum()))
    k = best[np.argmin(np.abs(best - offset + 0.5 - mean))]
    return float(k - offset) + 0.5


# Line rules other than the even line (the stream's "anchored" and "hyst" modes). Anchored: start
# at prod's line and step a point at a time toward even until P(over) is inside 50% +- ANCHOR_BAND.
# Hysteresis: keep the line the market last had until the even line is HOLD_MOVE points away or
# P(over) at the kept line leaves 50% +- HOLD_BAND. Lines are on prob_above's scale: a total, or
# the margin home must beat.
ANCHOR_BAND = 0.10
HOLD_MOVE = 2.0
HOLD_BAND = 0.15


def anchored_line(pmf, offset, start, band=ANCHOR_BAND):
    """`start` (prod's line), stepped a point at a time toward even until P(above) is inside
    50% +- band, never past the distribution's ends."""
    lo, hi = -offset, pmf.shape[-1] - 1 - offset
    x = float(start)
    for _ in range(pmf.shape[-1]):
        q = float(prob_above(pmf, offset, x))
        if q > 0.5 + band and x + 1 < hi:
            x += 1
        elif q < 0.5 - band and x - 1 > lo:
            x -= 1
        else:
            break
    return x


def held_line(pmf, offset, kept, move=HOLD_MOVE, band=HOLD_BAND):
    """The kept line while the even line is under `move` points from it and P(above) at it is
    inside 50% +- band; the even line otherwise (and with nothing kept)."""
    even = even_line(pmf, offset)
    if kept is None:
        return even
    q = float(prob_above(pmf, offset, kept))
    return kept if abs(even - kept) < move and abs(q - 0.5) <= band else even


# Key numbers: how the stream sets its own lines. Points come in lumps of 3 and 7, so the distributions are spiky: a
# line next to a spike prices far from its neighbours, and a small error in the spike moves its
# price a lot. Among half-point lines with P(over) inside 50% +- KEY_BAND, take the one with the
# least mass on the two whole numbers either side (ties: nearest 50%), and keep the market's last
# line while it is still in the band with at most KEY_HOLD more mass next to it.
KEY_BAND = 0.10
KEY_HOLD = 0.02


def key_line(pmf, offset, kept=None, band=KEY_BAND, hold=KEY_HOLD):
    """The half-point line in the gap between the spikes (see KEY_BAND); the even line when none
    has P(above) in the band."""
    pmf = np.asarray(pmf, dtype=float)
    above = 1 - np.cumsum(pmf)[:-1]                 # line k - offset + 0.5: P(above)
    mass = pmf[:-1] + pmf[1:]                       # on the two whole numbers either side
    ok = np.abs(above - 0.5) <= band + 1e-12
    if not ok.any():
        return even_line(pmf, offset)
    m = np.where(ok, mass, np.inf)
    tie = ok & (m <= m.min() + 1e-12)
    k = int(np.argmin(np.where(tie, np.abs(above - 0.5), np.inf)))
    if kept is not None and float(kept) % 1 == 0.5:
        j = int(np.floor(kept)) + offset
        if 0 <= j < len(above) and ok[j] and mass[j] <= mass[k] + hold:
            return float(kept)
    return float(k - offset) + 0.5


def prob_above(pmf, offset, line):
    """P(X > line), pushes taken out."""
    x = np.arange(pmf.shape[-1]) - offset
    above = pmf[..., x > line].sum(-1)
    at = pmf[..., x == line].sum(-1) if float(line).is_integer() else 0.0
    return _graded(above, at)


def prob_below(pmf, offset, line):
    """P(X < line), pushes taken out."""
    x = np.arange(pmf.shape[-1]) - offset
    below = pmf[..., x < line].sum(-1)
    at = pmf[..., x == line].sum(-1) if float(line).is_integer() else 0.0
    return _graded(below, at)


def market_prob(market, line, margin_pmf, total_pmf):
    """The chance of a market's selection at a line."""
    if market == 50:
        return prob_above(margin_pmf, MARGIN_MAX, 0.0)
    if market == 51:
        return prob_below(margin_pmf, MARGIN_MAX, 0.0)
    if market == 52:
        return prob_above(margin_pmf, MARGIN_MAX, line)
    if market == 53:
        return prob_below(margin_pmf, MARGIN_MAX, -line)
    if market == 54:
        return prob_above(total_pmf, 0, line)
    if market == 55:
        return prob_below(total_pmf, 0, line)
    raise ValueError(market)


def pool_context():
    """The multiprocessing start method for this platform."""
    return mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
