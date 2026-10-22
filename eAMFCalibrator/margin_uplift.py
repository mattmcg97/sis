"""Margin uplift, two books: prod against each candidate over the report window, every live in-play
price tick a dummy bet -- staked to win 100 at the margin chart's price -- given to whichever book pays
more, each book then scored on what it is left with. Every match counts the same.

Why not the partner's single-book version (each book taking every dummy bet at its own price): with every
bet staked to win 100, a winner always costs the book 100 and a loser pays it its whole stake, and short
prices carry the big stakes, so a book that prices its favourites too short keeps big stakes when they
lose and pays the same 100 when they win. Overconfidence scored as margin: on simulated ticks with the
truth known, a calibrated book made 8.4%, an overconfident one (logit x 1.15) 10.5%, x 1.3 12.2%, an
underconfident one (x 0.85) 6.1% -- and flat stakes turn it round (underconfident 10.7%). Two books
leave no such reward: a book too sure, or not sure enough, offers the better price where it is wrong and
is taken there (against a flat prod: calibrated 8.2% on what it takes, x 1.15 2.5%, x 1.3 -3.0%). Every
tick of a match settles on the one result, so the matches, not the ticks, are the sample: each match
weighs the same (each tick 1 / its match's ticks in the market) and every interval resamples days (a
day's matches share gamers and form, so they go together).

The ticks (unchanged): minutes from the match's scheduled start, in play once past half a minute, scaled
to a 60-minute game clock (normalised = round(minutes x factor), the factor set so prod's matches end at
GAME_MINUTES on average); the quarter off the minutes left. Matched ticks only: prod's ticks where every
candidate has a live price for the same match, market and selection no older than PAIR_MAX_AGE minutes,
and every feed's probability within PROB_RANGE; the same moments for every feed.

The dummy bets, per matched tick and selection:

  price       each book's margin chart price at its probability rounded to 3 places (their
              FD_109_BALANCED.csv: probability, numerator, denominator -> numerator/denominator + 1);
              without a chart a proportional 109% book
  the bet     to the book whose price is longer (prod on a tie); a spread or total only where the
              candidate's line is prod's (the same bet); stake 100 / (price - 1); GGR -100 if the selection
              won, else the stake, settled as the calibrator settles every market (markets.resolve); a push
              is left out
  kept        more than CUT_LAST game minutes left, a fair price above FAIR_FLOOR for both books

Per candidate and market -- moneyline, spread and totals (each with its 95% interval over days,
SIM_BOOT resamples):

  to candidate      the share of the dummy handle the candidate's prices take
  candidate takes   the candidate's margin on the bets it takes; prod keeps: prod's on the rest
  candidate - prod  the two against each other: above zero, the candidate's prices hold up better where
                    the two differ -- the headline
  both books        the two together against prod taking every bet alone, which a second book only
                    lowers (the bettor always gets the better price); the less, the more the candidate
                    is picked off
  log loss          a proper score on the same ticks and weights, candidate less prod: a cross-check
                    (lower is better)

by quarter and by prod's price on the selection (moneyline), and every match's part of the headline.
"""

import csv
import datetime as dt
from collections import defaultdict

import numpy as np

from . import config, markets
from .metrics import day_of
from .pipeline import to_unit_probability

GAME_MINUTES = 60                        # the clock the minutes are scaled to (four 15-minute quarters)
QUARTER_CUTS = (45, 30, 15)              # remaining > 45 Q1, > 30 Q2, > 15 Q3, else Q4
CUT_LAST = 6                             # the dummy bets leave out the last minutes (their 2 of 20)
FAIR_FLOOR = 1.001
STAKE_TO_WIN = 100.0
FALLBACK_BOOK = 1.09
PAIR_MAX_AGE = 2.0                       # matched ticks: the candidate's price at most this many minutes old
PROB_RANGE = (0.05, 0.95)                # matched ticks: every feed's probability within this
PRICE_BANDS = (0.2, 0.35, 0.5, 0.65, 0.8)   # moneyline cuts: prod's chance of the selection backed
LINE_EPS = 1e-9
MONEYLINE, SPREAD, TOTALS = "Moneyline", "Spread", "Totals"
GROUPS = (MONEYLINE, SPREAD, TOTALS)
MARKET_OF = {50: (0, 0), 51: (0, 1), 52: (1, 0), 53: (1, 1), 54: (2, 0), 55: (2, 1)}   # id -> (group, selection)


# ---------------------------------------------------------------- the margin chart

def load_chart(path):
    """{probability in thousandths: marginated decimal price} off a chart CSV: the probability in
    the first column, then `numerator` and `denominator` (fractional odds) -- named in a header, or,
    with no header (FD_109_BALANCED.csv), the second and third columns."""
    rows = None
    for encoding in ("utf-8-sig", "cp1252"):            # a chart saved from Excel may not be UTF-8
        try:
            with open(path, newline="", encoding=encoding) as fh:
                rows = list(csv.reader(fh))
            break
        except UnicodeDecodeError:
            continue
    if not rows:
        raise SystemExit(f"{path}: the margin chart could not be read")
    head = [h.strip().lower() for h in rows[0]]
    if _number(head[0] if head else "") is not None and len(head) >= 3:
        num, den, body = 1, 2, rows                    # no header: probability, numerator, denominator
    else:
        try:
            num, den = head.index("numerator"), head.index("denominator")
        except ValueError:
            raise SystemExit(f"{path}: a margin chart needs numerator and denominator columns, or no header"
                             f" and probability, numerator, denominator in that order (it has"
                             f" {', '.join(head)})")
        body = rows[1:]
    got = []
    for r in body:
        try:
            p, n, d = float(r[0].strip().rstrip("%")), float(r[num]), float(r[den])
        except (ValueError, IndexError):
            continue
        if d:
            got.append((p, n / d + 1.0))
    scale = 100.0 if got and max(p for p, _ in got) > 1.5 else 1.0     # probabilities written as percents
    out = {int(round(p / scale * 1000)): price for p, price in got}
    if not out:
        raise SystemExit(f"{path}: no prices in the margin chart")
    return out


def _number(v):
    try:
        return float(str(v).strip().rstrip("%"))
    except ValueError:
        return None


def fallback_chart(book=FALLBACK_BOOK):
    """A proportional `book` (109%) where no chart is given: 1 / (probability x book), at least 1.01."""
    return {k: max(1.01, 1.0 / (k / 1000 * book)) for k in range(1, 1000)}


def chart_array(chart):
    """The chart as an array over thousandths, NaN where it has no price."""
    a = np.full(1001, np.nan)
    for k, price in chart.items():
        if 0 <= k <= 1000:
            a[k] = price
    return a


# ---------------------------------------------------------------- ticks

def _when(v):
    if v is None or v == "":
        return None
    if isinstance(v, dt.datetime):
        return v.replace(tzinfo=None) if v.tzinfo is None else \
            v.astimezone(dt.timezone.utc).replace(tzinfo=None)
    try:
        return dt.datetime.fromisoformat(str(v)[:26].replace("T", " "))
    except ValueError:
        return None


class Ticks:
    """One feed's live in-play ticks on the moneyline, spread and totals, as arrays."""

    def __init__(self, name):
        self.name = name
        self.parts = defaultdict(list)
        self.matches = {}
        self.finals = {}
        self.at_bets = False             # a model priced only at the moments the bets read
        self.max_age = None              # its own matching age (None: PAIR_MAX_AGE); 0: its own ticks alone

    def add(self, quote_rows, starts, finals):
        """Add snowflake_io.fetch_quotes rows; `starts` {match: scheduled start}, `finals` {match:
        (player 1, player 2)}. A match without either is left out."""
        cols = defaultdict(list)
        for r in quote_rows:
            code, market, publish, prob, _, desc, _, _, active = r[:9]
            m = MARKET_OF.get(int(market)) if market is not None else None
            if m is None or prob is None or str(active).lower() != config.LIVE_IS_ACTIVE:
                continue
            start, fin = starts.get(code), finals.get(code)
            t = _when(publish)
            if start is None or fin is None or t is None or None in fin:
                continue
            minutes = (t - start).total_seconds() / 60
            if minutes <= 0.5:                      # their round(minutes) > 0
                continue
            p = to_unit_probability(prob)
            line = markets.parse_line(desc) if m[0] else 0.0
            if p is None or line is None:
                continue
            won = markets.resolve(int(market), line if m[0] else None, *fin)
            if won is None:                         # a push
                continue
            idx = self.matches.setdefault(code, len(self.matches))
            self.finals[code] = tuple(fin)
            for key, v in (("match", idx), ("group", m[0]), ("sel", m[1]), ("minutes", minutes),
                           ("prob", p), ("line", line), ("won", won)):
                cols[key].append(v)
        for key, v in cols.items():
            self.parts[key].append(np.asarray(v))
        return self

    def array(self, key):
        parts = self.parts.get(key)
        return np.concatenate(parts) if parts else np.zeros(0)

    def subset(self, codes):
        """The ticks of these matches only."""
        keep = [i for code, i in self.matches.items() if code in codes]
        mask = np.isin(self.array("match"), keep)
        out = Ticks(self.name)
        out.at_bets, out.max_age = self.at_bets, self.max_age
        for key in self.parts:
            out.parts[key] = [self.array(key)[mask]]
        out.matches = {code: i for code, i in self.matches.items() if code in codes}
        out.finals = {code: f for code, f in self.finals.items() if code in codes}
        return out

    def __len__(self):
        return int(sum(len(p) for p in self.parts.get("prob", [])))

    def mean_end(self):
        """The mean over matches of each one's last in-play tick, in minutes from the start."""
        match, minutes = self.array("match").astype(int), self.array("minutes")
        if not len(match):
            return None
        last = np.full(max(self.matches.values()) + 1 if self.matches else 0, -np.inf)
        np.maximum.at(last, match, minutes)
        return float(last[np.isfinite(last)].mean())


def _latest(prod, cand, max_age=PAIR_MAX_AGE):
    """(ok, at): for each prod tick, whether the candidate has a tick for the same match, market and
    selection at or before it (no older than `max_age` minutes), and that tick's index in `cand`."""
    to_prod = np.full(max(cand.matches.values(), default=-1) + 1, -1)
    for c, i in cand.matches.items():
        to_prod[i] = prod.matches.get(c, -1)
    c_match = cand.array("match").astype(int)
    c_match = to_prod[c_match] if len(c_match) else c_match
    key = lambda m, t: (m * len(GROUPS) + t.array("group").astype(int)) * 2 + t.array("sel").astype(int)
    pk, ck = key(prod.array("match").astype(int), prod), key(c_match, cand)
    pm, cm = prod.array("minutes"), cand.array("minutes")
    order = np.lexsort((cm, ck))
    ck, cm = ck[order], cm[order]
    span = (max(pm.max(initial=0), cm.max(initial=0)) + 1.0) * 2
    at = np.searchsorted(ck * span + cm, pk * span + pm, side="right") - 1
    ok, at = at >= 0, np.clip(at, 0, None)
    hit = order[at] if len(order) else np.zeros(len(pk), dtype=int)
    if len(order):
        ok &= (ck[at] == pk) & (pm - cm[at] <= max_age) & (c_match[hit] >= 0)
    return ok, hit


def matched(feeds, max_age=PAIR_MAX_AGE, prob_range=PROB_RANGE):
    """`feeds` [(label, Ticks)], prod first, cut to prod's ticks that every candidate has a price for
    (its latest, no older than `max_age` minutes, or the candidate's own max_age: 0 for a model priced
    at the bets' moments alone, its price at prod's very tick) and where every feed's probability is
    within `prob_range`. A candidate's tick keeps its own probability, line and result, at prod's match and
    minute; the matches are the ones with a tick left."""
    prod = feeds[0][1]
    lo, hi = prob_range
    inside = lambda p: (p >= lo) & (p <= hi)
    ok = inside(prod.array("prob"))
    hits = []
    for _, t in feeds[1:]:
        ok_c, hit = _latest(prod, t, max_age if t.max_age is None else t.max_age)
        ok &= ok_c & inside(t.array("prob")[hit] if len(t) else np.zeros(len(ok)))
        hits.append(hit)
    left = set(np.unique(prod.array("match")[ok]).astype(int).tolist())
    codes = {c: i for c, i in prod.matches.items() if i in left}
    out = []
    for n, (label, t) in enumerate(feeds):
        o = Ticks(t.name)
        for key in prod.parts:
            own = n and key not in ("match", "minutes")
            o.parts[key] = [t.array(key)[hits[n - 1][ok]] if own else prod.array(key)[ok]]
        o.matches = dict(codes)
        o.finals = {c: prod.finals[c] for c in codes if c in prod.finals}
        out.append((label, o))
    return out


def factor_for(ticks):
    """The scale that ends the feed's matches at GAME_MINUTES on average."""
    end = ticks.mean_end()
    return GAME_MINUTES / end if end else 1.0


# ---------------------------------------------------------------- the dummy bets



# ---------------------------------------------------------------- the two books

def quarter(remaining):
    for q, cut in enumerate(QUARTER_CUTS, 1):
        if remaining > cut:
            return q
    return 4


def price_band(prob):
    """Prod's chance of the selection, cut at PRICE_BANDS (0 the longest)."""
    return int(np.searchsorted(PRICE_BANDS, prob, side="right"))


def band_name(i):
    edges = (PROB_RANGE[0],) + PRICE_BANDS + (PROB_RANGE[1],)
    return f"{edges[i]:.2f}-{edges[i + 1]:.2f}"


def _books(prod, cand, factor, chart):
    """Every matched tick's dummy bet between prod and one candidate: {keep, take, s_t, g_t, s_p, g_p,
    ll_p, ll_c, w, key, group, quarter, band} as arrays. take: the candidate pays more; s_t/g_t the
    stake and GGR at the price taken (by whichever book), s_p/g_p at prod's; w each tick's weight (1 over
    its match's kept ticks in the market)."""
    chart = chart_array(chart)
    price = lambda p: chart[np.clip(np.round(p * 1000).astype(int), 0, 1000)]
    group, match = prod.array("group").astype(int), prod.array("match").astype(int)
    pp, pc = prod.array("prob"), cand.array("prob")
    won = prod.array("won").astype(bool)
    remaining = GAME_MINUTES - np.round(prod.array("minutes") * factor)
    xp, xc = price(pp), price(pc)
    ok = lambda x, p: np.isfinite(x) & (x > 1.0) & (1.0 / np.clip(p, 1e-9, None) > FAIR_FLOOR)
    same = (group == 0) | (np.abs(prod.array("line") - cand.array("line")) < LINE_EPS)
    keep = (remaining > CUT_LAST) & same & ok(xp, pp) & ok(xc, pc)
    take = keep & (xc > xp)
    xt = np.where(take, xc, xp)
    stake = lambda x: np.where(keep, STAKE_TO_WIN / np.where(keep, x - 1.0, 1.0), 0.0)
    s_t, s_p = stake(xt), stake(xp)
    g_t = np.where(keep, np.where(won, -STAKE_TO_WIN, s_t), 0.0)
    g_p = np.where(keep, np.where(won, -STAKE_TO_WIN, s_p), 0.0)
    right = lambda p: np.clip(np.where(won, p, 1.0 - p), 1e-6, 1.0)
    key = match * len(GROUPS) + group
    n = np.bincount(key[keep], minlength=int(key.max(initial=0)) + 1) if len(key) else np.zeros(0)
    w = np.where(keep, 1.0 / np.maximum(n[key] if len(key) else 0, 1), 0.0)
    q = np.array([quarter(r) for r in remaining], int) if len(remaining) else np.zeros(0, int)
    band = np.searchsorted(PRICE_BANDS, pp, side="right")
    return dict(keep=keep, take=take, s_t=s_t, g_t=g_t, s_p=s_p, g_p=g_p, ll_p=-np.log(right(pp)),
                ll_c=-np.log(right(pc)), w=w, match=match, group=group, quarter=q, band=band)


def _per_match(b, mask, n_matches):
    """Each match's weighted sums in the cut: the candidate's handle and GGR on what it takes, prod's on
    what it keeps and on every bet alone, the two log losses and the weight they are over."""
    m = b["match"][mask]
    w = b["w"][mask]
    t = b["take"][mask]
    sums = lambda v: np.bincount(m, weights=v, minlength=n_matches)
    return dict(h_t=sums(w * b["s_t"][mask] * t), g_t=sums(w * b["g_t"][mask] * t),
                h_k=sums(w * b["s_p"][mask] * ~t), g_k=sums(w * b["g_p"][mask] * ~t),
                h_a=sums(w * b["s_p"][mask]), g_a=sums(w * b["g_p"][mask]),
                l_p=sums(w * b["ll_p"][mask]), l_c=sums(w * b["ll_c"][mask]), wt=sums(w), n=int(mask.sum()))


def _ratio(num, den, W=None):
    """100 x num / den over the matches, and over each resample (W: resamples x matches)."""
    tot = 100 * num.sum() / den.sum() if den.sum() else None
    if W is None:
        return tot
    d = W @ den
    return tot, np.where(d > 0, 100 * (W @ num) / np.where(d > 0, d, 1.0), np.nan)


def _ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (None, None)


def stats_of(pm, W):
    """The cut's numbers, each difference with its 95% interval (W: resampled days, spread over their matches)."""
    if not pm["n"]:
        return None
    t, td = _ratio(pm["g_t"], pm["h_t"], W)
    k, kd = _ratio(pm["g_k"], pm["h_k"], W)
    both, bd = _ratio(pm["g_t"] + pm["g_k"], pm["h_t"] + pm["h_k"], W)
    alone, ad = _ratio(pm["g_a"], pm["h_a"], W)
    hb = pm["h_t"].sum() + pm["h_k"].sum()
    used = pm["wt"] > 0
    lp = pm["l_p"][used] / pm["wt"][used]
    lc = pm["l_c"][used] / pm["wt"][used]
    Wu = W[:, used]
    cnt = Wu.sum(1)
    ll_d = (Wu @ (lc - lp)) / np.where(cnt > 0, cnt, 1.0)
    return dict(bets=pm["n"], matches=int(used.sum()), to_cand=100 * pm["h_t"].sum() / hb if hb else None,
                takes=t, keeps=k, head=(None if t is None or k is None else t - k, *_ci(td - kd)),
                both=both, alone=alone, gain=(None if both is None or alone is None else both - alone, *_ci(bd - ad)),
                ll_prod=float(lp.mean()) if len(lp) else None, ll_cand=float(lc.mean()) if len(lc) else None,
                ll_diff=(float((lc - lp).mean()) if len(lp) else None, *_ci(ll_d)))


def compute(feeds, chart=None, chart_name=None, factor=None, window=None, boot=None, seed=0):
    """`feeds` [(label, Ticks)] over the report window, prod first: each cut to the matches every feed
    has ticks for, then to the matched ticks (matched()). One factor, fitted on prod's ticks before the
    matching unless given, scales every feed's minutes. Each candidate against prod, two books."""
    common = set.intersection(*(set(t.matches) for _, t in feeds)) if feeds else set()
    if not common or len(feeds) < 2:
        return None
    feeds = [(label, t.subset(common)) for label, t in feeds]
    factor = factor or factor_for(feeds[0][1])
    raw = feeds
    feeds = matched(feeds)
    prod = feeds[0][1]
    if not len(prod):
        return None
    chart = chart or fallback_chart()
    n_matches = max(prod.matches.values()) + 1
    boot = config.SIM_BOOT if boot is None else boot
    days = sorted({day_of(code) for code in prod.matches})
    day_ix = np.zeros(n_matches, int)
    for code, i in prod.matches.items():
        day_ix[i] = days.index(day_of(code))
    Wd = np.random.default_rng(seed).multinomial(len(days), [1 / len(days)] * len(days), size=boot).astype(float)
    W = Wd[:, day_ix]                       # each match weighs its day's draw: the days are resampled, not matches
    cands = []
    for label, t in feeds[1:]:
        b = _books(prod, t, factor, chart)
        groups = {}
        for gi, g in enumerate(GROUPS):
            base = b["keep"] & (b["group"] == gi)
            if not base.any():
                continue
            cut = lambda mask: stats_of(_per_match(b, base & mask, n_matches), W)
            groups[g] = dict(all=cut(np.ones_like(base)),
                             quarters={q: cut(b["quarter"] == q) for q in (1, 2, 3, 4)},
                             bands={i: cut(b["band"] == i) for i in range(len(PRICE_BANDS) + 1)} if gi == 0 else {},
                             priced=int(((b["group"] == gi)).sum()), same_line=int(base.sum()))
        cands.append(dict(label=label, groups=groups, books=b))
    out = dict(feeds=[dict(label=label, ticks=len(t), of=len(r), mean_end=r.mean_end(), at_bets=r.at_bets)
                      for (label, t), (_, r) in zip(feeds, raw)],
               matches=len(prod.matches), factor=factor, window=window, boot=boot, cands=cands,
               chart=chart_name or f"proportional {100 * FALLBACK_BOOK:.0f}% (no chart given)")
    out["by_match"] = by_match(out, prod)
    for c in cands:
        c.pop("books")
    return out


def by_match(data, prod):
    """[{code, final, cands: [{group: {...}}]}]: each match's dummy bets against each candidate and its
    part of the headline -- its GGR on what the candidate takes over the candidate's whole taken handle,
    less its GGR on what prod keeps over prod's whole kept handle (the parts add up to the headline)."""
    codes = sorted(prod.matches, key=prod.matches.get)
    n_matches = max(prod.matches.values()) + 1
    per = []
    for c in data["cands"]:
        b = c["books"]
        got = {}
        for gi, g in enumerate(GROUPS):
            pm = _per_match(b, b["keep"] & (b["group"] == gi), n_matches)
            ht, hk = pm["h_t"].sum(), pm["h_k"].sum()
            n = np.bincount(b["match"][b["keep"] & (b["group"] == gi)], minlength=n_matches)
            got[g] = (pm, ht, hk, n)
        per.append(got)
    out = []
    for code in codes:
        i = prod.matches[code]
        row = {"code": code, "final": prod.finals.get(code), "cands": []}
        for got in per:
            cell = {}
            for g, (pm, ht, hk, n) in got.items():
                if not n[i]:
                    continue
                hb = pm["h_t"][i] + pm["h_k"][i]
                cell[g] = dict(ticks=int(n[i]), to_cand=100 * pm["h_t"][i] / hb if hb else None,
                               takes=100 * pm["g_t"][i] / pm["h_t"][i] if pm["h_t"][i] else None,
                               keeps=100 * pm["g_k"][i] / pm["h_k"][i] if pm["h_k"][i] else None,
                               part=100 * ((pm["g_t"][i] / ht if ht else 0.0) - (pm["g_k"][i] / hk if hk else 0.0)),
                               ll_diff=(pm["l_c"][i] - pm["l_p"][i]) / pm["wt"][i] if pm["wt"][i] else None)
            row["cands"].append(cell)
        out.append(row)
    return out


def starts_of(history):
    """{match: scheduled start} off fetch_history rows."""
    return {r["MATCH_CODE"]: _when(r.get("SCHEDULED_START_TIME_UTC")) for r in history}


def names_of(history):
    """{match: (start, "player 1 v player 2")} off fetch_history rows, for the by-match tables."""
    out = {}
    for r in history:
        when = _when(r.get("SCHEDULED_START_TIME_UTC"))
        out[r["MATCH_CODE"]] = (f"{when:%m-%d %H:%M}" if when else "",
                                f"{r.get('PLAYER_1_HANDLE') or '?'} v {r.get('PLAYER_2_HANDLE') or '?'}")
    return out


# ---------------------------------------------------------------- the reports

def _num(v, places=2, signed=False):
    if v is None:
        return "-"
    return f"{v:+.{places}f}" if signed else f"{v:.{places}f}"


def _iv(x, places=2):
    """'+1.23 [-0.45, +2.91]' off (value, lo, hi)."""
    if not x or x[0] is None:
        return "-"
    lo, hi = x[1], x[2]
    return f"{x[0]:+.{places}f} [{_num(lo, places, True)}, {_num(hi, places, True)}]"


def _final(f):
    return "-" if not f else f"{f[0]:g}-{f[1]:g}"


def _mins(v):
    return "-" if v is None else f"{v:.1f}"


ROWS = (("bets", "dummy bets"), ("matches", "matches"), ("to_cand", "to the candidate %"),
        ("takes", "candidate takes %"), ("keeps", "prod keeps %"), ("head", "candidate - prod"),
        ("both", "both books %"), ("alone", "prod alone %"), ("gain", "both - prod alone"),
        ("ll_diff", "log loss, cand - prod"))


def _value(st, key):
    v = st.get(key) if st else None
    if key in ("head", "gain"):
        return _iv(v)
    if key == "ll_diff":
        return _iv(v, 4)
    if key in ("bets", "matches"):
        return "-" if v is None else f"{v:,}"
    return _num(v)


def _cut_lines(label, cuts, names):
    """A table of one cut set (quarters or price bands): the headline and its interval per cut."""
    lines = [f"    {label:22s} {'bets':>7s} {'to cand':>8s} {'takes':>7s} {'keeps':>7s} {'candidate - prod':>24s}"
             f" {'both - prod alone':>24s}"]
    for key, st in cuts.items():
        if not st:
            continue
        lines.append(f"    {names(key):22s} {st['bets']:>7,d} {_num(st['to_cand']):>8s} {_num(st['takes']):>7s}"
                     f" {_num(st['keeps']):>7s} {_iv(st['head']):>24s} {_iv(st['gain']):>24s}")
    return lines


def text(data):
    """Lines of text: each candidate against prod, two books, with the cuts."""
    if not data:
        return []
    lines = ["", "  MARGIN UPLIFT -- two books: every in-play tick a dummy bet to win 100, given to whichever book"
                 " pays more; every match counts the same",
             f"  prod against each candidate, {data.get('window') or 'the report window'}, on {data['matches']:,}"
             f" matches: prod's ticks with a price in every feed no more than {PAIR_MAX_AGE:g} minutes old, every"
             f" probability {PROB_RANGE[0]:g}-{PROB_RANGE[1]:g}",
             f"  margin chart: {data['chart']}; clock: minutes x {data['factor']:.3f} -> {GAME_MINUTES} game minutes;"
             f" the last {CUT_LAST} left out; 95% over days ({data['boot']:,} resamples)",
             "  " + "; ".join(f"{f['label']} {f['ticks']:,} of {f['of']:,} ticks matched" for f in data["feeds"])]
    if _at_bets(data):
        lines.append(f"  {_at_bets(data)} priced only at the moments the bets read: the dummy bets are prod's ticks"
                     " there alone. Punters bet where prod is wrong, so these ticks favour the candidate (a model"
                     " no better than prod comes out ahead): run bets --all-moments for a fair test")
    for c in data["cands"]:
        for g, gd in c["groups"].items():
            st = gd["all"]
            lines += ["", f"  {c['label']} against prod, {g}" + (
                f" ({gd['same_line']:,} of {gd['priced']:,} matched ticks at prod's line)" if g != MONEYLINE else "")]
            for key, name in ROWS:
                lines.append(f"    {name:24s} {_value(st, key):>28s}")
            lines += _cut_lines("by quarter", gd["quarters"], lambda q: f"Q{q}")
            if gd["bands"]:
                lines += _cut_lines("by prod's chance", gd["bands"], band_name)
    lines += ["", "  candidate - prod: the candidate's margin on the bets its prices take less prod's on the ones it"
                  " keeps -- above zero, the candidate's prices hold up better where they differ (the headline);",
              "  both - prod alone: the two books together against prod taking every bet, which a second book"
              " only lowers; log loss: the same ticks scored on the result (lower is better)"]
    lines += match_summary(data)
    return lines


TOP = 10


def _at_bets(data):
    """The feeds priced only at the bets' moments, named, or ''."""
    return ", ".join(f["label"] for f in data["feeds"] if f.get("at_bets"))


def match_summary(data, top=TOP):
    """Lines: per candidate and market, how the headline splits over the matches."""
    rows = data.get("by_match")
    if not rows:
        return []
    lines = ["", "  BY MATCH -- each match's part of candidate - prod (the parts add up to it; every match in"
                 " margin_uplift_matches.txt)"]
    for f, c in enumerate(data["cands"]):
        for g in c["groups"]:
            part = {r["code"]: r["cands"][f][g]["part"] for r in rows if g in r["cands"][f]}
            if not part:
                continue
            order = sorted(part, key=part.get)
            worst, best = order[:top], order[::-1][:top]
            lines += ["", f"  {g}, {c['label']}: the {top} matches that cost it most {sum(part[x] for x in worst):+.2f};"
                          f" the {top} that gain it most {sum(part[x] for x in best):+.2f}; over {len(part):,} matches"]
            by = {r["code"]: r for r in rows}
            lines.append(f"    {'match':13s} {'final':>7s} {'ticks':>6s} {'to cand':>8s} {'takes':>8s} {'keeps':>8s}"
                         f" {'part':>7s}")
            for x in worst:
                cell = by[x]["cands"][f][g]
                lines.append(f"    {x:13s} {_final(by[x]['final']):>7s} {cell['ticks']:>6,d} {_num(cell['to_cand']):>8s}"
                             f" {_num(cell['takes']):>8s} {_num(cell['keeps']):>8s} {cell['part']:>+7.2f}")
    return lines


def match_text(data, names=None):
    """Lines of text: every match, each candidate and market, the matches that cost the first candidate
    most first."""
    rows = data.get("by_match") or []
    if not rows or not data["cands"]:
        return []
    names = names or {}
    lines = ["", "  MARGIN UPLIFT BY MATCH -- two books, every match counting the same",
             f"  {data.get('window') or 'the report window'}, {len(rows):,} matches; ticks: the match's dummy bets;"
             " to cand: the share of its handle the candidate takes;",
             "  takes / keeps: the candidate's margin on what it takes, prod's on what it keeps; part: the match's"
             " share of candidate - prod (the parts add up to it); cum: the running sum"]
    for f, c in enumerate(data["cands"]):
        for g in c["groups"]:
            have = [r for r in rows if g in r["cands"][f]]
            order = sorted(have, key=lambda r: r["cands"][f][g]["part"])
            lines += ["", f"  {c['label']}, {g}: candidate - prod {_iv(c['groups'][g]['all']['head'])}"]
            lines.append(f"    {'match':13s} {'start':11s} {'players':24s} {'final':>7s} {'ticks':>6s} {'to cand':>8s}"
                         f" {'takes':>8s} {'keeps':>8s} {'part':>8s} {'cum':>7s} {'log loss less prod':>18s}")
            cum = 0.0
            for r in order:
                cell = r["cands"][f][g]
                cum += cell["part"]
                when, who = names.get(r["code"], ("", ""))
                lines.append(f"    {r['code']:13s} {when:11s} {who[:24]:24s} {_final(r['final']):>7s} {cell['ticks']:>6,d}"
                             f" {_num(cell['to_cand']):>8s} {_num(cell['takes']):>8s} {_num(cell['keeps']):>8s}"
                             f" {cell['part']:>+8.3f} {cum:>+7.2f} {_num(cell['ll_diff'], 4, True):>18s}")
    return lines


def write_match_csv(path, data, names=None):
    """One row per match, candidate and market."""
    names = names or {}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_code", "start", "players", "final_home", "final_away", "candidate", "market", "ticks",
                    "to_candidate_pct", "candidate_takes_pct", "prod_keeps_pct", "part", "log_loss_less_prod"])
        for r in data.get("by_match") or []:
            when, who = names.get(r["code"], ("", ""))
            fin = r["final"] or ("", "")
            for f, c in enumerate(data["cands"]):
                for g, cell in r["cands"][f].items():
                    rnd = lambda v, k=4: "" if v is None else round(v, k)
                    w.writerow([r["code"], when, who, fin[0], fin[1], c["label"], g, cell["ticks"], rnd(cell["to_cand"]),
                                rnd(cell["takes"]), rnd(cell["keeps"]), rnd(cell["part"], 5), rnd(cell["ll_diff"], 5)])


def write_csv(path, data):
    """One row per candidate, market and cut (all, each quarter, each price band): every number."""
    keys = ("bets", "matches", "to_cand", "takes", "keeps", "head", "head_lo", "head_hi", "both", "alone",
            "gain", "gain_lo", "gain_hi", "ll_prod", "ll_cand", "ll_diff", "ll_diff_lo", "ll_diff_hi")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["candidate", "market", "cut"] + list(keys))
        for c in data["cands"]:
            for g, gd in c["groups"].items():
                cuts = [("all", gd["all"])] + [(f"Q{q}", s) for q, s in gd["quarters"].items()] + \
                       [(band_name(i), s) for i, s in gd["bands"].items()]
                for name, st in cuts:
                    if not st:
                        continue
                    flat = dict(st)
                    for k in ("head", "gain", "ll_diff"):
                        flat[k], flat[k + "_lo"], flat[k + "_hi"] = st[k]
                    w.writerow([c["label"], g, name] + ["" if flat.get(k) is None else
                                                        (round(flat[k], 5) if isinstance(flat[k], float) else flat[k])
                                                        for k in keys])


def _cell_iv(x, places=2):
    if not x or x[0] is None:
        return "<td>&mdash;</td>"
    sure = x[1] is not None and x[2] is not None and (x[1] > 0 or x[2] < 0)
    cls = ("good" if x[0] > 0 else "bad") if sure else ""
    return (f'<td class="{cls}">{x[0]:+.{places}f} <span class="dim">[{_num(x[1], places, True)},'
            f' {_num(x[2], places, True)}]</span></td>')


def _cell_ll(x):
    """Log loss less prod: lower is better, so the colour the other way."""
    if not x or x[0] is None:
        return "<td>&mdash;</td>"
    sure = x[1] is not None and x[2] is not None and (x[1] > 0 or x[2] < 0)
    cls = ("bad" if x[0] > 0 else "good") if sure else ""
    return f'<td class="{cls}">{x[0]:+.4f} <span class="dim">[{_num(x[1], 4, True)}, {_num(x[2], 4, True)}]</span></td>'


def html(data, esc):
    """The panel body: each candidate's headline per market, the cuts folded away, then the matches."""
    if not data or not data.get("cands"):
        return '<p class="dim">No margin uplift: no matched in-play ticks in every feed.</p>'
    head = ("<tr><th>Candidate</th><th>Market</th><th>Dummy bets</th><th>Matches</th><th>To candidate</th>"
            "<th>Candidate takes</th><th>Prod keeps</th><th>Candidate &minus; prod</th><th>Both books</th>"
            "<th>Prod alone</th><th>Both &minus; prod alone</th><th>Log loss, candidate &minus; prod</th></tr>")
    rows, folds = [], []
    for c in data["cands"]:
        for g, gd in c["groups"].items():
            st = gd["all"]
            if not st:
                continue
            rows.append(f"<tr><th>{esc(c['label'])}</th><th>{esc(g)}</th><td>{st['bets']:,}</td><td>{st['matches']:,}</td>"
                        f"<td>{_num(st['to_cand'])}%</td><td>{_num(st['takes'])}%</td><td>{_num(st['keeps'])}%</td>"
                        f"{_cell_iv(st['head'])}<td>{_num(st['both'])}%</td><td>{_num(st['alone'])}%</td>"
                        f"{_cell_iv(st['gain'])}{_cell_ll(st['ll_diff'])}</tr>")
            cut_rows = []
            for title, cuts, name in (("By quarter", gd["quarters"], lambda q: f"Q{q}"),
                                      ("By prod's chance of the selection", gd["bands"], band_name)):
                for key, s in cuts.items():
                    if s:
                        cut_rows.append(f"<tr><th>{esc(title)}</th><th>{esc(name(key))}</th><td>{s['bets']:,}</td>"
                                        f"<td>{_num(s['to_cand'])}%</td><td>{_num(s['takes'])}%</td><td>{_num(s['keeps'])}%</td>"
                                        f"{_cell_iv(s['head'])}{_cell_iv(s['gain'])}{_cell_ll(s['ll_diff'])}</tr>")
            line_note = (f" &middot; {gd['same_line']:,} of {gd['priced']:,} matched ticks at prod's line"
                         if g != MONEYLINE else "")
            folds.append(f"""
    <details>
      <summary>{esc(c['label'])}, {esc(g)}: by quarter and by prod's price{line_note}</summary>
      <table class="reach">
        <thead><tr><th>Cut</th><th></th><th>Dummy bets</th><th>To candidate</th><th>Candidate takes</th>
          <th>Prod keeps</th><th>Candidate &minus; prod</th><th>Both &minus; prod alone</th><th>Log loss less prod</th></tr></thead>
        <tbody>{''.join(cut_rows)}</tbody>
      </table>
    </details>""")
    by = ""
    if data.get("by_match"):
        by = f"""
    <details>
      <summary>By match: the matches that move candidate &minus; prod most (every match in
        margin_uplift_matches.txt and .csv)</summary>
      <pre class="dim" style="overflow-x:auto">{esc(chr(10).join(match_summary(data)))}</pre>
    </details>"""
    feeds = " &middot; ".join(f"{esc(f['label'])}: {f['ticks']:,} of {f['of']:,} ticks matched" for f in data["feeds"])
    at_bets = (f'\n    <p class="dim">{esc(_at_bets(data))} priced only at the moments the bets read: the dummy bets'
               " are prod's ticks there alone. Punters bet where prod is wrong, so these ticks favour the"
               " candidate (a model no better than prod comes out ahead): run bets --all-moments for a fair"
               " test.</p>" if _at_bets(data) else "")
    return f"""
    <p class="dim">Two books, {esc(data.get('window') or 'the report window')}, {data['matches']:,} matches: every
      live in-play price tick is a dummy bet to win {STAKE_TO_WIN:.0f} at the margin chart's price
      ({esc(data['chart'])}), given to whichever book pays more, settled on the result. Each book is scored on
      what it is left with, so a book too sure (or not sure enough) is taken where it is wrong. Every match
      counts the same (each tick one over its match's ticks in the market) and every interval is 95% over
      days (a day's matches resampled together). Matched ticks only: prod's ticks where every candidate has a live price for the same match,
      market and selection no more than {PAIR_MAX_AGE:g} minutes old, every probability
      {PROB_RANGE[0]:g}&ndash;{PROB_RANGE[1]:g}; the spread and totals only where the candidate's line is
      prod's. Minutes from the scheduled start &times; {data['factor']:.3f} make a {GAME_MINUTES}-minute game;
      the last {CUT_LAST} are left out.</p>
    <p class="dim">{feeds}</p>{at_bets}
    <table class="reach">
      <thead>{head}</thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="dim">Candidate &minus; prod: the candidate's margin on the bets its prices take less prod's on the
      ones it keeps &mdash; above zero, the candidate's prices hold up better where the two differ. Both &minus;
      prod alone: the two books together against prod taking every bet, which a second book only lowers (the
      bettor always gets the better price). Log loss: a proper score on the same ticks, lower is better.</p>
    {''.join(folds)}{by}"""


def run(out_dir, feeds, chart_path=None, factor=None, names=None):
    """Grade the report window's feeds, `feeds` [(label, Ticks)] with prod first; writes
    margin_uplift.txt and .csv, and margin_uplift_matches.txt and .csv (every match, `names`
    {match: (start, players)} off names_of()). The summary for html(), or None when no match has
    every feed."""
    import os
    chart = load_chart(chart_path) if chart_path else None
    window = f"{str(config.CUTOFF_START)[:10]} to {str(config.CUTOFF_END or 'latest')[:10]}"
    data = compute(feeds, chart, os.path.basename(chart_path) if chart_path else None, factor, window)
    if data is None:
        print("\n  margin uplift: no matched in-play ticks in every feed; skipped")
        return None
    lines = text(data)
    print("\n".join(lines))
    with open(os.path.join(out_dir, "margin_uplift.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    write_csv(os.path.join(out_dir, "margin_uplift.csv"), data)
    with open(os.path.join(out_dir, "margin_uplift_matches.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(match_text(data, names)) + "\n")
    write_match_csv(os.path.join(out_dir, "margin_uplift_matches.csv"), data, names)
    return data
