"""The inference layer: the rematch layer widened to every match the two gamers have played that
session, against anyone. A beat B and C beat D: at A v D the price moves for what A's and D's games
said about their form that night, not only for what A and D did against each other.

A match's error against the prior (its margin less the expected margin, from the home side; its
total less the expected total) is read as

    margin error = v_home - v_away + u_pair + noise        total error = w_home + w_away + z_pair + noise

v / w: each gamer's form that session (one effect per gamer and session, rematch_layer.item's
gamer_keys), u / z: the pair's own effect that day (the rematch layer's), noise: the rest. So two
matches sharing a gamer covary by that gamer's form (with the sign of his side in each, on the
margin), and two meetings of a pair by both gamers' form and the pair's effect.

The variances are fitted by the method of moments on the prior's centred errors, as the rematch
layer's are:
  gamer   tau_v^2 / tau_w^2: the mean product of the errors of two matches the same session that share
          one gamer and not the pair (the margins each from that gamer's side), kept apart by whether he
          was on the same team in both
  pair    tau_u^2 / tau_z^2: two meetings of a pair the same day, less what their shared gamers' form
          explains, kept apart by whether the teams were the same (rematch_layer.RELATIONS)
  noise   sigma^2: an error's variance less both gamers' form and the pair's effect
A raw estimate below zero (a gamer who won big doing worse next time than the prior says) is
reported as it is and used as zero. Fewer than MIN_PAIRS pairs: zero. At a rematch the two gamers'
form and the pair's effect add up to what the rematch layer reads off the same meetings, so the
inference layer moves a rematch about as the rematch layer does, and moves a first meeting as well.

At a match, the move is the Gaussian BLUP of its error, c' S^-1 e, off the earlier errors e of
  - every earlier match that session of either gamer (they are finished: a gamer plays one match at
    a time), and
  - every earlier match of those matches' other gamers that session whose result was in
    (RESULT_LAG_MINUTES after its kick-off) -- they tell a gamer's form apart from his opponent's,
at most MAX_EARLIER of them (the gamers' own first), with c and S from the fitted variances.
walk_forward() fits each UTC day on the days before it, as rematch_layer.walk_forward does.
"""

import datetime as dt
import math
from collections import defaultdict

import numpy as np

from .rematch_layer import MARGIN_SD, MIN_PAIRS, RELATIONS, relation

GAMER_RELATIONS = ("same team", "team changed")
RESULT_LAG_MINUTES = 36
MAX_EARLIER = 60


def _keys(it):
    return it["home_key"], it["away_key"]


def _side(it, key):
    """+1 if the gamer of this session key is home in the match, -1 if away."""
    return 1 if key == it["home_key"] else -1


def _team(it, key):
    return it["teams"].get(key[0])


def gamer_relation(a, b, key):
    """same team / team changed: the gamer of `key` in the two matches."""
    return GAMER_RELATIONS[0] if _team(a, key) == _team(b, key) else GAMER_RELATIONS[1]


def _errors(it):
    """(margin error from the home side, total error), or None without a price or result."""
    if None in (it["margin"], it["total"], it["actual_margin"], it["actual_total"]):
        return None
    return it["actual_margin"] - it["margin"], it["actual_total"] - it["total"]


def _use(tau2, order):
    """Variances as used: below zero is zero. A 'changed' one may be the bigger (the pair's often is,
    once the gamers' form takes its share of a same-teams rematch): the noise on the diagonal keeps
    the covariance matrices proper, and move() checks."""
    return {r: [max(v, 0.0) for v in tau2[r]] for r in order}


class InferenceLayer:
    """The gamers' form and the pairs' effects (raw estimates by relation, [margin, total]), the
    noise and the mean errors, off a training set."""

    def __init__(self, gamer=None, pair=None, sigma2=None, means=(0.0, 0.0), counts=None):
        self.gamer = gamer or {r: [0.0, 0.0] for r in GAMER_RELATIONS}
        self.pair = pair or {r: [0.0, 0.0] for r in RELATIONS}
        self.sigma2 = sigma2 or [MARGIN_SD ** 2, 144.0]
        self.means = list(means)
        self.counts = counts or {}
        self.use_gamer = _use(self.gamer, GAMER_RELATIONS)
        self.use_pair = _use(self.pair, RELATIONS)

    @classmethod
    def fit(cls, items):
        errs = [(it, _errors(it)) for it in items]
        errs = [(it, e) for it, e in errs if e is not None]
        if not errs:
            return cls()
        means = [sum(e[q] for _, e in errs) / len(errs) for q in (0, 1)]
        var = [sum((e[q] - means[q]) ** 2 for _, e in errs) / len(errs) for q in (0, 1)]
        c = {id(it): (e[0] - means[0], e[1] - means[1]) for it, e in errs}
        by_key = defaultdict(list)
        for it, _ in errs:
            for k in _keys(it):
                by_key[k].append(it)
        gs, gn = {r: [0.0, 0.0] for r in GAMER_RELATIONS}, {r: 0 for r in GAMER_RELATIONS}
        pairs, seen = [], set()
        for k, its in by_key.items():
            for i in range(len(its)):
                for j in range(i + 1, len(its)):
                    a, b = its[i], its[j]
                    if a["day_key"] == b["day_key"]:
                        pk = (id(a), id(b)) if id(a) < id(b) else (id(b), id(a))
                        if pk not in seen:                  # found once through each gamer
                            seen.add(pk)
                            pairs.append((a, b))
                        continue
                    r = gamer_relation(a, b, k)
                    ea, eb = c[id(a)], c[id(b)]
                    gs[r][0] += _side(a, k) * _side(b, k) * ea[0] * eb[0]
                    gs[r][1] += ea[1] * eb[1]
                    gn[r] += 1
        gamer = {r: [gs[r][q] / gn[r] if gn[r] >= MIN_PAIRS else 0.0 for q in (0, 1)] for r in GAMER_RELATIONS}
        ps, pn = {r: [0.0, 0.0] for r in RELATIONS}, {r: 0 for r in RELATIONS}
        for a, b in pairs:
            ea, eb = c[id(a)], c[id(b)]
            shared = set(_keys(a)) & set(_keys(b))
            r = relation(a, b)
            # both from the pair's first gamer's side: each shared gamer's form counts +1
            ps[r][0] += a["sign"] * b["sign"] * ea[0] * eb[0] - sum(gamer[gamer_relation(a, b, k)][0] for k in shared)
            ps[r][1] += ea[1] * eb[1] - sum(gamer[gamer_relation(a, b, k)][1] for k in shared)
            pn[r] += 1
        pair = {r: [ps[r][q] / pn[r] if pn[r] >= MIN_PAIRS else 0.0 for q in (0, 1)] for r in RELATIONS}
        layer = cls(gamer, pair, None, means, {**{f"gamer {r}": n for r, n in gn.items()},
                                               **{f"pair {r}": n for r, n in pn.items()}})
        same_g, same_p = layer.use_gamer[GAMER_RELATIONS[0]], layer.use_pair[RELATIONS[0]]
        layer.sigma2 = [max(var[q] - 2 * same_g[q] - same_p[q], 1e-6) for q in (0, 1)]
        return layer

    def covariances(self, its):
        """(margin, total) covariance matrices of these matches' errors (each margin from its home
        side), the noise on the diagonal."""
        n = len(its)
        key_id, team_id = {}, {}
        k = lambda x: key_id.setdefault(x, len(key_id))
        t = lambda x: team_id.setdefault(x, len(team_id))
        hk = np.array([k(it["home_key"]) for it in its])
        ak = np.array([k(it["away_key"]) for it in its])
        ht = np.array([t(it["teams"].get(it["home"])) for it in its])
        at = np.array([t(it["teams"].get(it["away"])) for it in its])
        home = np.array([k((it["home"],)) for it in its])          # the gamer, for the pair's teams
        pk = np.array([k(("pair",) + tuple(it["day_key"])) for it in its])
        sg = np.array([it["sign"] for it in its], dtype=float)
        out = []
        for q in (0, 1):
            same, changed = self.use_gamer[GAMER_RELATIONS[0]][q], self.use_gamer[GAMER_RELATIONS[1]][q]
            M = np.zeros((n, n))
            for ka, kb, ta, tb, s in ((hk, hk, ht, ht, 1.0), (hk, ak, ht, at, -1.0),
                                      (ak, hk, at, ht, -1.0), (ak, ak, at, at, 1.0)):
                hit = ka[:, None] == kb[None, :]
                tau = np.where(ta[:, None] == tb[None, :], same, changed)
                M += hit * tau * (s if q == 0 else 1.0)
            p_same, p_changed = self.use_pair[RELATIONS[0]][q], self.use_pair[RELATIONS[1]][q]
            hit = pk[:, None] == pk[None, :]
            same_home = home[:, None] == home[None, :]
            teams_same = np.where(same_home, (ht[:, None] == ht[None, :]) & (at[:, None] == at[None, :]),
                                  (ht[:, None] == at[None, :]) & (at[:, None] == ht[None, :]))
            tau = np.where(teams_same, p_same, p_changed)
            M += hit * tau * (np.outer(sg, sg) if q == 0 else 1.0)
            M[np.diag_indices(n)] += self.sigma2[q]
            out.append(M)
        return out

    def move(self, it, earlier):
        """(the home margin's move, the total's move) for `it` off earlier matches (items with
        results): the BLUP of its error."""
        prev = [(e, _errors(e)) for e in earlier]
        prev = [(e, x) for e, x in prev if x is not None]
        if not prev:
            return 0.0, 0.0
        mats = self.covariances([p for p, _ in prev] + [it])
        n = len(prev)
        out = []
        for q in (0, 1):
            S, c = mats[q][:n, :n], mats[q][:n, n]
            if not np.any(c):
                out.append(0.0)
                continue
            e = np.array([x[q] - self.means[q] for _, x in prev])
            low = float(np.linalg.eigvalsh(S).min())
            if low <= 1e-6:                             # not a proper covariance: lift it until it is
                S = S + (1e-6 - low) * np.eye(n)
            try:
                w = np.linalg.solve(S, c)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(S, c, rcond=None)[0]
            out.append(float(w @ e))
        return out[0], out[1]

    def describe(self):
        """Lines: the fitted variances (raw, then as used) and what one earlier match moves a price by."""
        L = []
        for q, name in ((0, "margin"), (1, "total")):
            s2 = self.sigma2[q]
            g_same = self.use_gamer[GAMER_RELATIONS[0]][q]
            parts = []
            for r in GAMER_RELATIONS:
                raw, used = self.gamer[r][q], self.use_gamer[r][q]
                parts.append(f"{r} {_sd(raw)} ({self.counts.get(f'gamer {r}', 0):,} pairs)"
                             + ("" if abs(raw - used) < 1e-12 else f", used {_sd(used)}"))
            pp = []
            for r in RELATIONS:
                raw, used = self.pair[r][q], self.use_pair[r][q]
                pp.append(f"{r} {_sd(raw)} ({self.counts.get(f'pair {r}', 0):,} pairs)"
                          + ("" if abs(raw - used) < 1e-12 else f", used {_sd(used)}"))
            one = g_same / (s2 + 2 * g_same + self.use_pair[RELATIONS[0]][q]) if g_same else 0.0
            L.append(f"  {name}: sigma {math.sqrt(s2):.2f}, mean error {self.means[q]:+.2f}; a gamer's form "
                     f"that session: {'; '.join(parts)}; the pair's own: {'; '.join(pp)}; one earlier match "
                     f"of one gamer against someone else moves it by about {one:.3f} of that match's error")
        return L


def _sd(v):
    """A variance as a signed sd (a negative raw estimate shows as -sqrt(-v))."""
    return f"{math.copysign(math.sqrt(abs(v)), v):+.2f}"


def earlier_for(it, by_key, lag=dt.timedelta(minutes=RESULT_LAG_MINUTES), max_earlier=MAX_EARLIER):
    """(the earlier matches to read `it` off, how many of them are the gamers' own): every earlier
    match of either gamer that session, and their opponents' earlier matches that session whose
    result was in by `it`'s kick-off -- at most max_earlier, the gamers' own kept first (the most
    recent of them if there are more), the opponents' most recent filling the rest."""
    near = {}
    for k in _keys(it):
        for e in by_key.get(k, ()):
            near[id(e)] = e
    far = {}
    for e in near.values():
        for k in _keys(e):
            for f in by_key.get(k, ()):
                if id(f) not in near and f["start"] + lag <= it["start"]:
                    far[id(f)] = f
    # the gamers' own matches first (the pair's earlier meetings among them), their opponents' to fill
    own = sorted(near.values(), key=lambda x: x["start"])[-max_earlier:]
    rest = sorted(far.values(), key=lambda x: x["start"])[-(max_earlier - len(own)):] if len(own) < max_earlier else []
    return sorted(own + rest, key=lambda x: x["start"]), len(near)


def walk_forward(items, min_days=7):
    """Each item with the inference layer's move (infer_margin, infer_total), fitted each UTC day on
    every item before that day (no move until `min_days` days have been seen), and how many of
    the gamers' own earlier matches that session it read (n_near) and how many against others
    (n_others). Returns (rows, the last layer)."""
    items = sorted(items, key=lambda it: it["start"])
    by_key = defaultdict(list)
    out, layer, day, seen = [], InferenceLayer(), None, 0
    first_day = items[0]["start"].date() if items else None
    for it in items:
        d = it["start"].date()
        if d != day:
            day = d
            seen = (d - first_day).days
            train = [x for x in out if x["start"].date() < d]
            layer = InferenceLayer.fit(train) if seen >= min_days else InferenceLayer()
        earlier, n_near = earlier_for(it, by_key)
        own = [e for e in earlier if set(_keys(e)) & set(_keys(it))]
        dm, dt_ = layer.move(it, earlier) if seen >= min_days else (0.0, 0.0)
        out.append(dict(it, infer_margin=dm, infer_total=dt_, n_near=n_near,
                        n_others=sum(1 for e in own if e["day_key"] != it["day_key"])))
        for k in _keys(it):
            by_key[k].append(it)
    return out, layer
