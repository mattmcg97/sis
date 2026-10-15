"""Each gamer's stats as of each day: the pre-match features, off player-stats' per-match rows
(player_match_stats.csv), recalculated each UTC day from the matches before that day only.

  a stat      a ratio of two per-match counts (yards over the plays with a yardage, 3rd downs made over
              run, red-zone touchdowns over trips) or a per-match count (turnovers, possession, plays,
              points)
  recency     each gamer's matches weighted by how recent they are among the gamer's own: 2^(-k /
              HALF_LIFE) for the k-th most recent, so the match HALF_LIFE back counts half. Every
              match the gamer plays ages the rest, whether or not it carried the stat
  shrinkage   towards the league before the day, by PRIOR_MATCHES matches' worth of its average
              counts: a gamer with few matches reads near the league, one with many as their own
  detail      a stat that needs a field position is read only off matches with one on 80%+ of the
              snaps (player_stats' rule): through the 23 Sep - 9 Oct gap such a stat stays where the
              gamer's earlier matches left it
  a match     each gamer's values as of the start of the match's UTC day: for the feature layer, the
              home less the away (the margin's) and the two added (the total's)

Nothing a match is read with comes from its own day or later.
"""

import datetime as dt
from collections import defaultdict

from .bet_totals import _start

HALF_LIFE = 50              # matches
PRIOR_MATCHES = 20
DETAIL_SHARE = 0.8


def _f(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _detail(r):
    snaps = _f(r.get("snaps")) or 0
    return bool(snaps) and (_f(r.get("detail_snaps")) or 0) >= DETAIL_SHARE * snaps


def _scouted(r):
    return bool(_f(r.get("scouting")))


def _total(*cells):
    """The sum of (row, field) cells, or None if any is missing."""
    vals = [_f(row.get(k)) for row, k in cells]
    return None if any(v is None for v in vals) else sum(vals)


# name -> (what it reads off a match's row r and its opponent's row o: (numerator, denominator) or
# None where the match can't say)
FEATURES = (
    ("ypp_for", lambda r, o: (_f(r.get("yards")), _f(r.get("yards_known"))) if _detail(r) else None),
    ("ypp_against", lambda r, o: (_f(r.get("opp_yards")), _f(r.get("opp_yards_known"))) if _detail(r) else None),
    ("third_pct", lambda r, o: (_f(r.get("third_conv")), _f(r.get("third_att"))) if _detail(r) else None),
    ("rz_td_pct", lambda r, o: (_f(r.get("rz_td")), _f(r.get("rz_trips"))) if _detail(r) else None),
    ("giveaways", lambda r, o: (_total((r, "turnovers"), (r, "tod")), 1.0) if _detail(r) else None),
    ("takeaways", lambda r, o: (_total((o, "turnovers"), (o, "tod")), 1.0) if o and _detail(o) else None),
    ("top_min", lambda r, o: ((_f(r.get("top_s")) or 0) / 60.0, 1.0) if _scouted(r) and _f(r.get("top_s")) else None),
    ("plays", lambda r, o: (_f(r.get("plays")), 1.0) if _detail(r) else None),
    ("points_for", lambda r, o: (_f(r.get("points")), 1.0)),
    ("points_against", lambda r, o: (_f(r.get("points_against")), 1.0)),
)
NAMES = tuple(name for name, _ in FEATURES)


def _reads(r, o):
    """{feature: (numerator, denominator)} a match's row says, the unreadable left out."""
    out = {}
    for name, fn in FEATURES:
        got = fn(r, o)
        if got and got[0] is not None and got[1] is not None and got[1] > 0:
            out[name] = got
    return out


class Book:
    """Running weighted sums: per gamer and stat, the numerator and denominator over the gamer's
    matches (each older match aged by one step on every match the gamer plays), and the league's plain
    totals."""

    def __init__(self, half_life=HALF_LIFE, prior_matches=PRIOR_MATCHES):
        self.decay = 2.0 ** (-1.0 / half_life)
        self.prior = prior_matches
        self.gamer = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
        self.matches = defaultdict(float)             # each gamer's weighted count of matches
        self.league = defaultdict(lambda: [0.0, 0.0, 0])

    def add(self, gamer, reads):
        g = self.gamer[gamer]
        for s in g.values():
            s[0] *= self.decay
            s[1] *= self.decay
        self.matches[gamer] = self.matches[gamer] * self.decay + 1.0
        for name, (num, den) in reads.items():
            g[name][0] += num
            g[name][1] += den
            lg = self.league[name]
            lg[0] += num
            lg[1] += den
            lg[2] += 1

    def value(self, gamer, name):
        """The gamer's stat, shrunk towards the league's by PRIOR_MATCHES of its average counts; the
        league's alone for a gamer it has nothing on; None before the league has any."""
        lg = self.league[name]
        if not lg[2] or not lg[1]:
            return None
        pn, pd = self.prior * lg[0] / lg[2], self.prior * lg[1] / lg[2]
        s = self.gamer[gamer].get(name, (0.0, 0.0)) if gamer in self.gamer else (0.0, 0.0)
        return (s[0] + pn) / (s[1] + pd)

    def values(self, gamer):
        return {name: self.value(gamer, name) for name in NAMES}


def _day(text):
    t = _start(text)
    return None if t is None else t.date()


def daily(rows, half_life=HALF_LIFE, prior_matches=PRIOR_MATCHES, since=None):
    """{match: {"home": {stat: value}, "away": {...}, "n_home": weighted matches, "n_away": ...}} for
    every match in the rows (player_match_stats.csv's, two a match), each read as of the start of its
    UTC day off every match before that day; with `since` (a date), only the matches from it are kept
    (the earlier still build the book)."""
    by = defaultdict(dict)
    for r in rows:
        if r.get("match_code") and r.get("side") in ("home", "away"):
            by[r["match_code"]][r["side"]] = r
    matches = [(m, _start(p.get("home", p.get("away", {})).get("start")), p) for m, p in by.items()]
    matches = sorted(((m, t, p) for m, t, p in matches if t is not None), key=lambda x: (x[1], x[0]))
    days = defaultdict(list)
    for m, t, p in matches:
        days[t.date()].append((m, p))
    book = Book(half_life, prior_matches)
    out = {}
    for day in sorted(days):
        todays = days[day]
        if since is None or day >= since:
            for m, p in todays:
                h, a = p.get("home"), p.get("away")
                gh = (h or {}).get("player", "").strip().upper()
                ga = (a or {}).get("player", "").strip().upper()
                out[m] = {"home": book.values(gh), "away": book.values(ga),
                          "n_home": book.matches.get(gh, 0.0), "n_away": book.matches.get(ga, 0.0)}
        for m, p in todays:                           # then the day goes into the book
            h, a = p.get("home"), p.get("away")
            for r, o in ((h, a), (a, h)):
                if r and r.get("player"):
                    book.add(r["player"].strip().upper(), _reads(r, o))
    return out


def margin_x(f):
    """The margin's inputs off a match's features: home less away, stat by stat (None if any is)."""
    if not f:
        return None
    xs = [None if f["home"].get(n) is None or f["away"].get(n) is None else f["home"][n] - f["away"][n]
          for n in NAMES]
    return None if None in xs else xs


def total_x(f):
    """The total's inputs: the two gamers' stats added, stat by stat (None if any is)."""
    if not f:
        return None
    xs = [None if f["home"].get(n) is None or f["away"].get(n) is None else f["home"][n] + f["away"][n]
          for n in NAMES]
    return None if None in xs else xs


def since_day(text):
    """A --since timestamp's date, or None."""
    t = _start(text) if text else None
    return t.date() if isinstance(t, dt.datetime) else None
