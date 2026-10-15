"""The rematch layer: a pre-match price's margin and total moved by how the same two gamers' earlier
meetings that day went against that price.

Each meeting's error against the prior (its margin less the expected margin, from the alphabetically
first gamer's side; its total less the expected total) is read as the pair's effect that day, u,
plus noise. Two meetings the same day share u, so the covariance of their errors is u's variance,
tau^2 -- kept apart for meetings with the same teams and with teams changed -- and the rest of an
error's variance is the noise, sigma^2. At a rematch, u's best guess from the day's earlier errors
e (the posterior mean, a Gaussian random effect's BLUP) is c' S^-1 e, with c the covariance of this
meeting's error with each earlier one (tau^2 for its teams' relation to that meeting) and S the
earlier errors' covariance (sigma^2 + tau^2 on the diagonal, tau^2 off it for their relation). The
margin moves by u's guess from the home gamer's side, the total by its own.

Any prior works: expected points per side (glmer, NB2: margin and total off them), or a moneyline
and a total line (prod: the margin implied by its moneyline, MARGIN_SD * the normal quantile). A
moneyline is the normal about the margin, Phi(margin / MARGIN_SD), as eAMFModel's form layer and
glmer/cadence.R read one.

walk_forward() fits tau and sigma each UTC day on the days before it (every error centred on the
prior's mean error then, so a biased prior is not corrected on rematches alone), and moves each
rematch by the day's earlier meetings: nothing a match is priced with comes after its kick-off.
"""

import math
from collections import defaultdict
from statistics import NormalDist

MARGIN_SD = 9.35
RELATIONS = ("same teams", "teams changed")
MIN_PAIRS = 200          # fewer pairs of meetings than this for a relation: no move on it
_N = NormalDist()


def implied_margin(p, sd=MARGIN_SD):
    """The expected margin a moneyline implies, read as a normal about it."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return sd * _N.inv_cdf(p)


def win_chance(margin, sd=MARGIN_SD):
    return _N.cdf(margin / sd)


def relation(a, b):
    """same teams / teams changed between two meetings of a pair (each gamer on the same team)."""
    return RELATIONS[0] if a["teams"] == b["teams"] else RELATIONS[1]


def item(match, start, day_key, home, away, home_team, away_team, margin=None, total=None,
         actual_margin=None, actual_total=None, gamer_keys=None):
    """One priced match. day_key: the pair and its day (bet_rematches.day_meetings'), margin and
    total: the prior's expected home margin and total, actual_*: the result (None if not in),
    gamer_keys: each gamer's session, (home's, away's) -- for the inference layer; without them
    each gamer's session is taken as the pair's day."""
    first = min(home, away)
    keys = tuple(gamer_keys) if gamer_keys else ((home, day_key), (away, day_key))
    return dict(match=match, start=start, day_key=day_key, home=home, away=away,
                teams={home: home_team, away: away_team}, margin=margin, total=total,
                actual_margin=actual_margin, actual_total=actual_total,
                sign=1 if home == first else -1, home_key=keys[0], away_key=keys[1])


def _errors(it):
    """(margin error from the first gamer's side, total error), or None without a price or result."""
    if None in (it["margin"], it["total"], it["actual_margin"], it["actual_total"]):
        return None
    return it["sign"] * (it["actual_margin"] - it["margin"]), it["actual_total"] - it["total"]


class Layer:
    """tau^2 by relation and sigma^2, for the margin (0) and the total (1), off a training set."""

    def __init__(self, tau2=None, sigma2=None, means=(0.0, 0.0), pairs=None):
        self.tau2 = tau2 or {r: [0.0, 0.0] for r in RELATIONS}
        self.sigma2 = sigma2 or [MARGIN_SD ** 2, 144.0]
        self.means = list(means)
        self.pairs = pairs or {r: 0 for r in RELATIONS}

    @classmethod
    def fit(cls, items):
        """Off every training item with a price and a result: the mean errors, then the products of
        the centred errors of every two meetings the same day (tau^2 by relation), and sigma^2."""
        errs = [(it, _errors(it)) for it in items]
        errs = [(it, e) for it, e in errs if e is not None]
        if not errs:
            return cls()
        means = [sum(e[q] for _, e in errs) / len(errs) for q in (0, 1)]
        var = [sum((e[q] - means[q]) ** 2 for _, e in errs) / len(errs) for q in (0, 1)]
        days = defaultdict(list)
        for it, e in errs:
            days[it["day_key"]].append((it, [e[0] - means[0], e[1] - means[1]]))
        sums = {r: [0.0, 0.0] for r in RELATIONS}
        pairs = {r: 0 for r in RELATIONS}
        for rows in days.values():
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    r = relation(rows[i][0], rows[j][0])
                    pairs[r] += 1
                    for q in (0, 1):
                        sums[r][q] += rows[i][1][q] * rows[j][1][q]
        tau2 = {r: [max(0.0, sums[r][q] / pairs[r]) if pairs[r] >= MIN_PAIRS else 0.0 for q in (0, 1)]
                for r in RELATIONS}
        sigma2 = [max(var[q] - max(tau2[r][q] for r in RELATIONS), 1e-6) for q in (0, 1)]
        return cls(tau2, sigma2, means, pairs)

    def move(self, it, earlier):
        """(the home margin's move, the total's move) for `it` off the day's earlier meetings
        (items with results), the BLUP of the pair's effect that day."""
        prev = [(e, _errors(e)) for e in earlier]
        prev = [(e, x) for e, x in prev if x is not None]
        if not prev:
            return 0.0, 0.0
        out = []
        for q in (0, 1):
            n = len(prev)
            S = [[0.0] * n for _ in range(n)]
            for a in range(n):
                for b in range(n):
                    S[a][b] = (self.sigma2[q] + self.tau2[relation(prev[a][0], prev[b][0])][q] if a == b
                               else self.tau2[relation(prev[a][0], prev[b][0])][q])
            c = [self.tau2[relation(it, p[0])][q] for p in prev]
            e = [p[1][q] - self.means[q] for p in prev]
            w = solve(S, c)
            out.append(sum(wi * ei for wi, ei in zip(w, e)))
        return it["sign"] * out[0], out[1]

    def describe(self):
        """Lines: the fitted variances and what one earlier meeting moves a rematch by."""
        L = []
        for q, name in ((0, "margin"), (1, "total")):
            parts = []
            for r in RELATIONS:
                t2 = self.tau2[r][q]
                w = t2 / (self.sigma2[q] + t2) if t2 else 0.0
                parts.append(f"{r}: tau {math.sqrt(t2):.2f} ({self.pairs[r]:,} pairs), one earlier "
                             f"meeting moves it by {w:.3f} of its error")
            L.append(f"  {name}: sigma {math.sqrt(self.sigma2[q]):.2f}, mean error {self.means[q]:+.2f}; "
                     + "; ".join(parts))
        return L


def solve(A, b):
    """x with A x = b, by Gaussian elimination (A small and positive definite)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                f = M[r][col] / M[col][col]
                for k in range(col, n + 1):
                    M[r][k] -= f * M[col][k]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0 for i in range(n)]


def walk_forward(items, min_days=7):
    """Each item with the layer's move: fitted each UTC day on every item before that day (no move
    until `min_days` days have been seen), applied off the day's earlier meetings of the pair.
    Returns (items with margin_adj / total_adj / move_margin / move_total, the last layer)."""
    items = sorted(items, key=lambda it: it["start"])
    by_key = defaultdict(list)
    out, layer, day, seen = [], Layer(), None, 0
    first_day = items[0]["start"].date() if items else None
    for it in items:
        d = it["start"].date()
        if d != day:
            day = d
            seen = (d - first_day).days
            train = [x for x in out if x["start"].date() < d]
            layer = Layer.fit(train) if seen >= min_days else Layer()
        earlier = by_key[it["day_key"]]
        dm, dt_ = layer.move(it, earlier) if seen >= min_days else (0.0, 0.0)
        row = dict(it, move_margin=dm, move_total=dt_, n_earlier=len(earlier),
                   margin_adj=None if it["margin"] is None else it["margin"] + dm,
                   total_adj=None if it["total"] is None else it["total"] + dt_)
        out.append(row)
        by_key[it["day_key"]].append(it)
    return out, layer


def score(rows, margin_key="margin", total_key="total", sd=MARGIN_SD):
    """{log_loss, brier, margin_rmse, total_rmse, n} of one reading of the rows (draws left out of
    the moneyline)."""
    ll, br, me, te, n = [], [], [], [], 0
    for r in rows:
        m, t = r.get(margin_key), r.get(total_key)
        if m is not None and r["actual_margin"] is not None:
            me.append((r["actual_margin"] - m) ** 2)
            if r["actual_margin"] != 0:
                p = min(max(win_chance(m, sd), 1e-6), 1 - 1e-6)
                y = 1.0 if r["actual_margin"] > 0 else 0.0
                ll.append(-(y * math.log(p) + (1 - y) * math.log(1 - p)))
                br.append((p - y) ** 2)
        if t is not None and r["actual_total"] is not None:
            te.append((r["actual_total"] - t) ** 2)
        n += 1
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return dict(n=n, log_loss=mean(ll), brier=mean(br),
                margin_rmse=math.sqrt(mean(me)) if me else None, total_rmse=math.sqrt(mean(te)) if te else None,
                ll=ll, br=br, me=me, te=te)


def paired(a, b):
    """(mean of b - a, its t) over matched lists."""
    d = [y - x for x, y in zip(a, b)]
    if len(d) < 3:
        return None, None
    m = sum(d) / len(d)
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1))
    return m, (m / (sd / math.sqrt(len(d))) if sd else None)
