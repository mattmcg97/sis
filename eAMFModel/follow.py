"""A pre-match model that follows each gamer's results since its fit: a second layer on NB2 or glmer.

A build fits its pre-match model once, at its cut-off, and its ratings stay put while gamers' levels
move. Each gamer's offset -- the sum of their errors against the model's own prices over their
matches settled since the fit, divided by that many matches plus PRIOR -- is added to their later
matches: the margin errors to the expected margin, the total errors to the expected total. A gamer
2 points worse than rated over 40 matches is priced 40 * 2 / (40 + 80) = 0.67 points worse.

Out of sample on AMFELO (12 fortnights, each priced off a fit at its start, errors restarting at each
fit), the layer took moneyline log loss 0.6682 -> 0.6656 for glmer (better in 9 of 12 fortnights)
and 0.6692 -> 0.6653 for NB2 (12 of 12). It gains nothing in the first days after a fit and more
the older the fit, so a rolling prior refitted every day (rolling_prior.py) is left as it is. There
is no same-session effect to catch beyond this: a gamer's errors correlate as much across sessions
as within one (see the README's "Following the results since the fit").

The offset of a match uses only the gamer's matches that started before it, so a quote priced as of
an earlier time (stream.known_states) reads the results known then.
"""

import datetime as dt
from bisect import bisect_left
from collections import defaultdict

from . import nb2_prior

ON = True
PRIOR = 80                       # matches' worth of zero offset each gamer starts from


def _gamers(row):
    return [str(row.get(k) or "").strip().upper() for k in ("PLAYER_1_HANDLE", "PLAYER_2_HANDLE")]


def _finals(row):
    try:
        return float(row["PLAYER_1_FINAL_SCORE"]), float(row["PLAYER_2_FINAL_SCORE"])
    except (KeyError, TypeError, ValueError):
        return None


class Following:
    """A fitted pre-match model plus each gamer's offset since its cut-off (`since`)."""

    FOLLOWS_RESULTS = True           # means() reads the results known when pricing

    def __init__(self, pre, since, prior=PRIOR):
        self.pre, self.since, self.prior = pre, since, prior

    def __getattr__(self, name):
        return getattr(self.pre, name)

    @property
    def AS_OF(self):
        return getattr(self.pre, "AS_OF", True)

    def describe(self):
        base = self.pre.describe() if hasattr(self.pre, "describe") else type(self.pre).__name__
        return f"{base}; following each gamer's results since {self.since:%Y-%m-%d} (prior {self.prior} matches)"

    def offsets(self, settled, priced):
        """gamer -> (sorted starts, cumulative margin errors, cumulative total errors) over `settled`
        matches priced at `priced` (code -> expected points)."""
        errors = defaultdict(list)
        for r in settled:
            p, f, t = priced.get(r["MATCH_CODE"]), _finals(r), nb2_prior._start(r)
            if p is None or f is None or t is None:
                continue
            em, et = (f[0] - f[1]) - (p[0] - p[1]), (f[0] + f[1]) - (p[0] + p[1])
            g1, g2 = _gamers(r)
            errors[g1].append((t, em, et))
            errors[g2].append((t, -em, et))
        out = {}
        for g, es in errors.items():
            es.sort(key=lambda e: e[0])
            cm, ct = [0.0], [0.0]
            for _, em, et in es:
                cm.append(cm[-1] + em)
                ct.append(ct[-1] + et)
            out[g] = ([e[0] for e in es], cm, ct)
        return out

    def means(self, schedule, results=None, **kw):
        """Each match's expected (home, away) points: the model's own, the margin moved by the two
        gamers' margin offsets and the total by their total offsets, each read off their matches
        settled since the fit that started before this one."""
        settled = [r for r in results or () if _finals(r) is not None
                   and nb2_prior._start(r) is not None and nb2_prior._start(r) >= self.since]
        if not settled:
            return self.pre.means(schedule, results=results, **kw)
        asked = {r["MATCH_CODE"] for r in schedule}
        extra = [r for r in settled if r["MATCH_CODE"] not in asked]
        got = self.pre.means(list(schedule) + extra, results=results, **kw)
        cum = self.offsets(settled, got)

        def offset(gamer, start):
            if gamer not in cum:
                return 0.0, 0.0
            starts, cm, ct = cum[gamer]
            k = bisect_left(starts, start)              # only matches that started before this one
            return cm[k] / (k + self.prior), ct[k] / (k + self.prior)

        out = {}
        for r in schedule:
            code = r["MATCH_CODE"]
            if code not in got:
                continue
            h, a = got[code]
            start = nb2_prior._start(r)
            if start is not None:
                (m1, t1), (m2, t2) = (offset(g, start) for g in _gamers(r))
                dm, dt_ = m1 - m2, (t1 + t2) / 2
                h, a = max(0.1, h + (dt_ + dm) / 2), max(0.1, a + (dt_ - dm) / 2)
            out[code] = (h, a)
        return out


def wrap(pre, rolling=False):
    """The pre-match model following the results since its fit (ON), or as it is: when it is a
    rolling prior refitted every day, or its cut-off is not recorded."""
    if not ON or rolling or pre is None:
        return pre
    before = (getattr(pre, "meta", None) or {}).get("before")
    if not before:
        return pre
    return Following(pre, dt.datetime.fromisoformat(before[:19]))
