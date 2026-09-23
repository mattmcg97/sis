"""Small probability-mass-function toolkit, pure Python.

A pmf over points is a list: pmf[k] = P(points == k). Everything the
pricer does is built from three operations on these -- convolve, mix and
read a tail -- so they live here, exactly, with no sampling anywhere. That
is the point of the exact approach: a price moves only when the game state
moves, never because a simulation drew a different set of random numbers.
"""

import math

EPS = 1e-13


def point(k=0):
    """All the mass on k."""
    out = [0.0] * (k + 1)
    out[k] = 1.0
    return out


def normalise(pmf):
    total = sum(pmf)
    return [p / total for p in pmf] if total > 0 else pmf


def convolve(a, b, cap=None):
    """Distribution of X + Y for independent X ~ a, Y ~ b."""
    size = len(a) + len(b) - 1
    if cap is not None:
        size = min(size, cap)
    out = [0.0] * size
    b_nonzero = [(j, y) for j, y in enumerate(b) if y > EPS]
    for i, x in enumerate(a):
        if x <= EPS:
            continue
        for j, y in b_nonzero:
            k = i + j
            if k >= size:
                break
            out[k] += x * y
    return out


def powers(pmf, n_max, cap=None):
    """[pmf^0, pmf^1, ..., pmf^n_max] under convolution."""
    out = [point(0)]
    for _ in range(n_max):
        out.append(convolve(out[-1], pmf, cap))
    return out


def mix(weighted):
    """Mixture of pmfs: [(weight, pmf), ...] -> pmf. Weights need not sum to 1."""
    size = max(len(p) for _, p in weighted)
    out = [0.0] * size
    for w, p in weighted:
        if w <= 0:
            continue
        for k, x in enumerate(p):
            out[k] += w * x
    return out


def mean(pmf):
    return sum(k * p for k, p in enumerate(pmf))


def variance(pmf):
    mu = mean(pmf)
    return sum((k - mu) ** 2 * p for k, p in enumerate(pmf))


def poisson(lam, tail=1e-10, n_max=80):
    """Poisson pmf, truncated where the upper tail is negligible, renormalised."""
    if lam <= 0:
        return [1.0]
    out = []
    p = math.exp(-lam)
    k = 0
    cumulative = 0.0
    while k <= n_max:
        out.append(p)
        cumulative += p
        if k > lam and 1 - cumulative < tail:
            break
        k += 1
        p *= lam / k
    return normalise(out)


class Signed:
    """A pmf over integers that can be negative (a margin), offset-indexed."""

    def __init__(self, low, masses):
        self.low = low
        self.masses = masses

    @classmethod
    def empty(cls, low, high):
        return cls(low, [0.0] * (high - low + 1))

    def add(self, value, mass):
        self.masses[value - self.low] += mass

    def items(self):
        for i, p in enumerate(self.masses):
            if p > 0:
                yield self.low + i, p

    def prob_above(self, line):
        """P(X > line)."""
        return sum(p for v, p in self.items() if v > line)

    def prob_below(self, line):
        return sum(p for v, p in self.items() if v < line)

    def prob_at(self, value):
        i = value - self.low
        return self.masses[i] if 0 <= i < len(self.masses) else 0.0

    def mean(self):
        return sum(v * p for v, p in self.items())

    def total(self):
        return sum(self.masses)
