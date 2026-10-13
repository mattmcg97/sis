"""Margin uplift, the partner's way: their eBasketball before/after notebook's dummy bets, applied
to AF as a direct comparison -- prod against each candidate over the report window, on prod's ticks
that every candidate has a price for (see "Matched ticks only" below) -- weaknesses and all.

Every live in-play price tick is a dummy bet. Per tick:

  minutes     from the match's scheduled start (not the first play), in play once past half a
              minute; scaled to a game clock (normalised = round(minutes x factor)), the factor set
              so prod's matches end, on average, at GAME_MINUTES -- the partner scaled by
              (1 - 0.146) so a 20-minute game ended at 20; AF's four quarters make 60
  remaining   GAME_MINUTES - normalised; the quarter off it (> 45 Q1, > 30 Q2, > 15 Q3, else Q4)
  price       the margin chart's price at the probability rounded to 3 places (their
              FD_109_BALANCED.csv: probability, numerator, denominator -> numerator/denominator + 1);
              without a chart a proportional 109% book stands in
  stake       100 / (price - 1): every bet wins 100, so shorter prices carry bigger stakes
  expected    1 - price / fair, fair = 1 / probability; expected revenue = expected x stake
  GGR         stake - price x stake if the selection won, else stake

Kept for the money: a chart price, a fair price above 1.001, and more than CUT_LAST game minutes
left (the partner's 2 of 20). Margin % = GGR / handle, expected % = expected revenue / handle;
the uplift is a candidate's margin less prod's.

Moneyline: home is their side A, away B; a selection wins when its side scores more (a tie loses
both). Totals: over wins above the line, under below it (a push loses both). Spreads are left out,
as the partner had none.

Calibration, their weighted MAE: per game minute left, selection and band (moneyline: the price's
tenth of probability; totals: the line's band, TOTAL_BANDS here for AF's lines), |mean probability
- win rate| weighted by the ticks. The checkpoints (54, 45, 36, 30, 21 and 15 minutes left: their
18, 15, 12, 10, 7 and 5 of 20) show each band, calibration where every feed has 25 ticks and
margins where a feed has more than 50.

Matched ticks only: the feeds don't tick at the same moments (prod far more often than a candidate
priced at the PLAY_OVER snapshots), so every table bets at prod's ticks alone. At each one, each
candidate's latest live price for the same match, market and selection (no older than PAIR_MAX_AGE
minutes); a prod tick without one in every candidate is dropped from every feed, as is one where any
feed's probability is outside PROB_RANGE. The same moments and bets for all, only the probabilities
differ.
"""

import csv
import datetime as dt
import math
from collections import defaultdict

import numpy as np

from . import config, markets
from .pipeline import to_unit_probability

GAME_MINUTES = 60                        # the clock the minutes are scaled to (four 15-minute quarters)
QUARTER_CUTS = (45, 30, 15)              # remaining > 45 Q1, > 30 Q2, > 15 Q3, else Q4
CUT_LAST = 6                             # the money leaves out the last minutes (their 2 of 20)
CHECKPOINTS = ((54, 1), (45, 2), (36, 2), (30, 3), (21, 3), (15, 4))
FAIR_FLOOR = 1.001
STAKE_TO_WIN = 100.0
FALLBACK_BOOK = 1.09
MIN_CAL = 25                             # a checkpoint band's calibration: this many ticks in every feed
MIN_MARGIN = 50                          # a checkpoint band's margin: more ticks than this in a feed
PAIR_MAX_AGE = 2.0                       # matched ticks: the candidate's price at most this many minutes old
PROB_RANGE = (0.05, 0.95)                # matched ticks: every feed's probability within this

MONEYLINE, TOTALS = "Moneyline", "Totals"
GROUPS = (MONEYLINE, TOTALS)
SELECTIONS = {MONEYLINE: ("A", "B"), TOTALS: ("Over", "Under")}
MARKET_OF = {50: (0, 0), 51: (0, 1), 54: (1, 0), 55: (1, 1)}      # market id -> (group, selection)
ML_BANDS = [f"{i / 10:.1f} - {(i + 1) / 10:.1f}" for i in range(10)]
# AF totals bands (the partner's were basketball's, <100 to 130.5+): a line falls in the first
# band whose top it does not pass
TOTAL_BANDS = [(35.5, "<=35.5"), (40.5, "36-40.5"), (45.5, "41-45.5"), (50.5, "46-50.5"),
               (55.5, "51-55.5"), (math.inf, "56+")]
BANDS = {MONEYLINE: ML_BANDS, TOTALS: [b[1] for b in TOTAL_BANDS]}
NB = max(len(ML_BANDS), len(TOTAL_BANDS))
FIELDS = ("n_cal", "sum_p", "sum_won", "n", "handle", "expected", "ggr")


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


def _won(group, sel, line, finals):
    s1, s2 = finals
    if group == 0:
        return (s1 > s2) if sel == 0 else (s2 > s1)
    total = s1 + s2
    return (total > line) if sel == 0 else (total < line)


def _band(group, prob, line):
    if group == 0:
        return max(0, min(9, math.ceil(prob * 10) - 1))
    for i, (top, _) in enumerate(TOTAL_BANDS):
        if line <= top:
            return i
    return None


class Ticks:
    """One feed's live in-play ticks on the moneyline and totals, as arrays."""

    def __init__(self, name):
        self.name = name
        self.parts = defaultdict(list)
        self.matches = {}
        self.finals = {}

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
            line = markets.parse_line(desc) if m[0] == 1 else 0.0
            if p is None or line is None:
                continue
            band = _band(m[0], p, line)
            if band is None:
                continue
            idx = self.matches.setdefault(code, len(self.matches))
            self.finals[code] = tuple(fin)
            for key, v in (("match", idx), ("group", m[0]), ("sel", m[1]), ("minutes", minutes),
                           ("prob", p), ("band", band), ("won", _won(m[0], m[1], line, fin))):
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
    key = lambda m, t: (m * 2 + t.array("group").astype(int)) * 2 + t.array("sel").astype(int)
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
    (its latest, no older than `max_age` minutes) and where every feed's probability is within
    `prob_range`. A candidate's tick keeps its own probability, band and result, at prod's match and
    minute; the matches are the ones with a tick left."""
    prod = feeds[0][1]
    lo, hi = prob_range
    inside = lambda p: (p >= lo) & (p <= hi)
    ok = inside(prod.array("prob"))
    hits = []
    for _, t in feeds[1:]:
        ok_c, hit = _latest(prod, t, max_age)
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

def _bets(ticks, factor, chart):
    """Every tick's dummy bet: {remaining, cal, money, stake, expected, ggr} as arrays."""
    minutes, prob = ticks.array("minutes"), ticks.array("prob")
    won = ticks.array("won").astype(float)
    remaining = GAME_MINUTES - np.round(minutes * factor)
    price = chart_array(chart)[np.clip(np.round(prob * 1000).astype(int), 0, 1000)]
    fair = 1.0 / np.clip(prob, 1e-9, None)
    money = (remaining > CUT_LAST) & np.isfinite(price) & (price > 1.0) & (fair > FAIR_FLOOR)
    stake = np.where(money, STAKE_TO_WIN / np.where(money, price - 1.0, 1.0), 0.0)
    return dict(remaining=remaining, cal=remaining > 0, money=money, stake=stake,
                expected=np.where(money, (1.0 - price / fair) * stake, 0.0),
                ggr=np.where(money, np.where(won > 0, stake - price * stake, stake), 0.0))


def grade(ticks, factor, chart):
    """{(group, remaining, selection, band): {FIELDS}} for one feed: the partner's sums."""
    prob = ticks.array("prob")
    if not len(prob):
        return {}
    group, sel = ticks.array("group").astype(int), ticks.array("sel").astype(int)
    band, won = ticks.array("band").astype(int), ticks.array("won").astype(float)
    b = _bets(ticks, factor, chart)
    remaining, cal, money, stake, expected, ggr = (b[k] for k in ("remaining", "cal", "money", "stake",
                                                                  "expected", "ggr"))
    keep = cal | money
    rem = np.clip(remaining, -1, GAME_MINUTES).astype(int) + 1     # 0 holds anything past the end
    key = ((group * (GAME_MINUTES + 2) + rem) * 2 + sel) * NB + band
    size = 2 * (GAME_MINUTES + 2) * 2 * NB
    sums = {f: np.bincount(key[keep], weights=w[keep], minlength=size) for f, w in (
        ("n_cal", cal.astype(float)), ("sum_p", prob * cal), ("sum_won", won * cal),
        ("n", money.astype(float)), ("handle", stake), ("expected", expected), ("ggr", ggr))}
    out = {}
    for k in np.flatnonzero(sums["n_cal"] + sums["n"]):
        b, rest = k % NB, k // NB
        s, rest = rest % 2, rest // 2
        r, g = rest % (GAME_MINUTES + 2) - 1, rest // (GAME_MINUTES + 2)
        out[(GROUPS[g], int(r), SELECTIONS[GROUPS[g]][s], BANDS[GROUPS[g]][b])] = \
            {f: float(sums[f][k]) for f in FIELDS}
    return out


MATCH_FIELDS = ("n", "handle", "expected", "ggr", "scored", "brier", "logloss")


def by_match(feeds, factor, chart):
    """[{code, final, feeds: [{group: sums}]}] in the matches' order: each feed's dummy bets per match
    and market (ticks bet, handle, expected revenue, GGR), with a proper score on the same ticks
    (Brier and log loss on every tick with more than CUT_LAST game minutes left, bet or not) and
    `top_loser`, the highest probability the feed gave a selection that lost."""
    t0 = feeds[0][1]
    codes = sorted(t0.matches, key=t0.matches.get)
    size = (max(t0.matches.values()) + 1) * 2 if t0.matches else 0
    per_feed = []
    for _, t in feeds:
        match, group = t.array("match").astype(int), t.array("group").astype(int)
        prob, won = t.array("prob"), t.array("won").astype(float)
        b = _bets(t, factor, chart)
        scored = b["remaining"] > CUT_LAST
        right = np.clip(np.where(won > 0, prob, 1.0 - prob), 1e-6, 1.0)
        key = match * 2 + group
        sums = {f: np.bincount(key, weights=w, minlength=size) for f, w in (
            ("n", b["money"].astype(float)), ("handle", b["stake"]), ("expected", b["expected"]),
            ("ggr", b["ggr"]), ("scored", scored.astype(float)), ("brier", (prob - won) ** 2 * scored),
            ("logloss", -np.log(right) * scored))}
        top = np.full(size, np.nan)
        lost = scored & (won == 0)
        np.fmax.at(top, key[lost], prob[lost])
        per_feed.append((sums, top))
    out = []
    for code in codes:
        i = t0.matches[code]
        row = {"code": code, "final": t0.finals.get(code), "feeds": []}
        for sums, top in per_feed:
            row["feeds"].append({g: dict({f: float(sums[f][i * 2 + gi]) for f in MATCH_FIELDS},
                                         top_loser=None if np.isnan(top[i * 2 + gi]) else float(top[i * 2 + gi]))
                                 for gi, g in enumerate(GROUPS)})
        out.append(row)
    return out


def match_totals(rows, n_feeds, skip=()):
    """[{group: summed MATCH_FIELDS}] per feed over the rows, leaving out the `skip` codes."""
    out = [{g: dict.fromkeys(MATCH_FIELDS, 0.0) for g in GROUPS} for _ in range(n_feeds)]
    for r in rows:
        if r["code"] in skip:
            continue
        for f, feed in enumerate(r["feeds"]):
            for g in GROUPS:
                for k in MATCH_FIELDS:
                    out[f][g][k] += feed[g][k]
    return out


def parts(rows, n_feeds):
    """{(feed, group): {code: points}}: each match's share of a candidate's uplift on prod, its
    GGR over the candidate's whole handle less prod's GGR there over prod's; they add up to the
    uplift."""
    tot = match_totals(rows, n_feeds)
    out = {}
    for f in range(1, n_feeds):
        for g in GROUPS:
            hc, hp = tot[f][g]["handle"], tot[0][g]["handle"]
            if hc and hp:
                out[(f, g)] = {r["code"]: 100 * (r["feeds"][f][g]["ggr"] / hc - r["feeds"][0][g]["ggr"] / hp)
                               for r in rows}
    return out


def uplift_of(totals, f, g):
    """A candidate's margin less prod's off match_totals, None where either staked nothing."""
    m, base = margins(totals[f][g])[0], margins(totals[0][g])[0]
    return None if m is None or base is None else m - base


def scores(totals):
    """[{group: (Brier, log loss)}] per feed off match_totals."""
    return [{g: ((s[g]["brier"] / s[g]["scored"], s[g]["logloss"] / s[g]["scored"]) if s[g]["scored"]
                 else (None, None)) for g in GROUPS} for s in totals]


def names_of(history):
    """{match: (start, "player 1 v player 2")} off fetch_history rows, for the by-match tables."""
    out = {}
    for r in history:
        when = _when(r.get("SCHEDULED_START_TIME_UTC"))
        out[r["MATCH_CODE"]] = (f"{when:%m-%d %H:%M}" if when else "",
                                f"{r.get('PLAYER_1_HANDLE') or '?'} v {r.get('PLAYER_2_HANDLE') or '?'}")
    return out


def quarter(remaining):
    for q, cut in enumerate(QUARTER_CUTS, 1):
        if remaining > cut:
            return q
    return 4


def overall(cells):
    """{group: sums} over every minute, selection and band."""
    out = defaultdict(lambda: dict.fromkeys(FIELDS, 0.0))
    for (g, _, _, _), c in cells.items():
        for f in FIELDS:
            out[g][f] += c[f]
    return dict(out)


def margins(s):
    """(margin %, expected %) of summed cells, None where nothing was staked."""
    h = s.get("handle", 0.0)
    return (100 * s["ggr"] / h, 100 * s["expected"] / h) if h else (None, None)


def wmae(cells):
    """{(group, selection): (ticks, weighted MAE %)} and per minute {(group, selection, remaining):
    (ticks, WMAE %)}, their ip_minute_errors: per minute, selection and band, |mean probability -
    win rate| weighted by the ticks."""
    err, n = defaultdict(float), defaultdict(float)
    for (g, r, s, b), c in cells.items():
        if r <= 0 or not c["n_cal"]:
            continue
        e = abs(c["sum_p"] / c["n_cal"] - c["sum_won"] / c["n_cal"]) * c["n_cal"]
        for key in ((g, s), (g, s, r)):
            err[key] += e
            n[key] += c["n_cal"]
    return {k: (n[k], 100 * err[k] / n[k]) for k in n if n[k]}


def checkpoint(cells_by_feed, remaining, group):
    """The partner's checkpoint view for one game minute left and market: per selection and band,
    each feed's (priced, won, ticks) and (expected %, actual %, ticks)."""
    out = []
    for s in SELECTIONS[group]:
        for b in BANDS[group]:
            row = {"selection": s, "band": b, "feeds": []}
            for cells in cells_by_feed:
                c = cells.get((group, remaining, s, b))
                if not c:
                    row["feeds"].append(None)
                    continue
                m, e = margins(c)
                row["feeds"].append(dict(
                    priced=c["sum_p"] / c["n_cal"] if c["n_cal"] else None,
                    won=c["sum_won"] / c["n_cal"] if c["n_cal"] else None, n_cal=int(c["n_cal"]),
                    margin=m, expected=e, n=int(c["n"])))
            if any(row["feeds"]):
                out.append(row)
    return out


def checkpoint_wmae(rows, i):
    """Feed i's weighted MAE (%) over the bands every feed has MIN_CAL ticks in, per selection."""
    out = {}
    for s in sorted({r["selection"] for r in rows}):
        err = n = 0.0
        for r in rows:
            if r["selection"] != s or not all(f and f["n_cal"] >= MIN_CAL for f in r["feeds"]):
                continue
            f = r["feeds"][i]
            err += abs(f["priced"] - f["won"]) * f["n_cal"]
            n += f["n_cal"]
        out[s] = 100 * err / n if n else None
    return out


def starts_of(history):
    """{match: scheduled start} off fetch_history rows."""
    return {r["MATCH_CODE"]: _when(r.get("SCHEDULED_START_TIME_UTC")) for r in history}


# ---------------------------------------------------------------- the whole thing

def compute(feeds, chart=None, chart_name=None, factor=None, window=None):
    """`feeds` [(label, Ticks)] over the report window, prod first: each cut to the matches every
    feed has ticks for, then to the matched ticks (matched()). One factor, fitted on prod's ticks
    before the matching unless given, scales every feed's minutes, as the partner's one 0.146 did."""
    common = set.intersection(*(set(t.matches) for _, t in feeds)) if feeds else set()
    if not common:
        return None
    feeds = [(label, t.subset(common)) for label, t in feeds]
    factor = factor or factor_for(feeds[0][1])
    raw = feeds
    feeds = matched(feeds)
    if not len(feeds[0][1]):
        return None
    chart = chart or fallback_chart()
    cells = [grade(t, factor, chart) for _, t in feeds]
    out = dict(feeds=[dict(label=label, ticks=len(t), of=len(r), mean_end=r.mean_end())
                      for (label, t), (_, r) in zip(feeds, raw)],
               matches=len(feeds[0][1].matches), factor=factor, window=window, cells=cells,
               chart=chart_name or f"proportional {100 * FALLBACK_BOOK:.0f}% (no chart given)")
    out["overall"] = [overall(c) for c in cells]
    out["by_match"] = by_match(feeds, factor, chart)
    out["scores"] = scores(match_totals(out["by_match"], len(feeds)))
    out["wmae"] = [wmae(c) for c in cells]
    out["checkpoints"] = [(r, q, g, checkpoint(cells, r, g)) for r, q in CHECKPOINTS for g in GROUPS]
    return out


def headline(data):
    """[(group, [(margin %, expected %, ticks) per feed])]."""
    out = []
    for g in GROUPS:
        row = []
        for o in data["overall"]:
            s = o.get(g, {})
            row.append(margins(s) + (int(s.get("n", 0)),))
        out.append((g, row))
    return out


def _mins(v):
    return "-" if v is None else f"{v:.1f}"


def _pct(v, signed=False):
    return "-" if v is None else (f"{v:+.2f}" if signed else f"{v:.2f}%")


def text(data):
    """Lines of text: the partner's prints, a column per feed."""
    if not data:
        return []
    labels = [f["label"] for f in data["feeds"]]
    w = max(14, max(len(x) for x in labels) + 2)
    lines = ["", "  MARGIN UPLIFT -- the partner's dummy-bet method (every in-play tick a bet to win 100)",
             f"  prod against each candidate, {data.get('window') or 'the report window'}, on"
             f" {data['matches']:,} matches: prod's ticks with a price in every feed no more than"
             f" {PAIR_MAX_AGE:g} minutes old, every probability {PROB_RANGE[0]:g}-{PROB_RANGE[1]:g}",
             f"  margin chart: {data['chart']}; clock: minutes x {data['factor']:.3f} -> {GAME_MINUTES}"
             f" game minutes; money leaves out the last {CUT_LAST}", ""]
    lines.append(f"  {'':22s}" + "".join(f"{x:>{w}s}" for x in labels))
    lines.append(f"  {'ticks matched':22s}" + "".join(f"{f['ticks']:>{w},d}" for f in data["feeds"]))
    lines.append(f"  {'ticks in all':22s}" + "".join(f"{f['of']:>{w},d}" for f in data["feeds"]))
    lines.append(f"  {'mean match (minutes)':22s}" + "".join(f"{_mins(f['mean_end']):>{w}s}" for f in data["feeds"]))
    for g in GROUPS:
        lines.append(f"  {g}")
        sums = [o.get(g, {}) for o in data["overall"]]
        for key, lab in (("expected", "expected revenue"), ("ggr", "revenue (GGR)"), ("handle", "handle")):
            lines.append(f"    {lab:20s}" + "".join(f"{s.get(key, 0.0):>{w},.0f}" for s in sums))
        ms = [margins(s) for s in sums]
        lines.append(f"    {'margin %':20s}" + "".join(f"{_pct(m[0]):>{w}s}" for m in ms))
        lines.append(f"    {'expected margin %':20s}" + "".join(f"{_pct(m[1]):>{w}s}" for m in ms))
        base = ms[0][0]
        lines.append(f"    {'uplift on prod':20s}" + "".join(
            f"{_pct(None if m[0] is None or base is None else m[0] - base, True):>{w}s}" for m in ms))
        sc = [x[g] for x in data.get("scores") or []]
        if sc:
            lines.append(f"    {'Brier, same ticks':20s}" + "".join(f"{_num(b, 4):>{w}s}" for b, _ in sc))
            lines.append(f"    {'log loss, same ticks':20s}" + "".join(f"{_num(x, 4):>{w}s}" for _, x in sc))
    if data.get("scores"):
        lines += ["", "  margin % and uplift: the partner's dummy bets, every tick a bet to win 100, so a"
                      " short price carries a big stake;",
                  "  Brier and log loss: the same ticks scored on the result, every tick alike (lower is"
                      " better)"]
    lines += match_summary(data)
    lines += ["", "  calibration, weighted MAE % (per game minute left, selection and band):"]
    for g in GROUPS:
        for s in SELECTIONS[g]:
            vals = [x.get((g, s)) for x in data["wmae"]]
            lines.append(f"    {g + ' ' + s:20s}" + "".join(
                f"{'-' if v is None else f'{v[1]:.3f}':>{w}s}" for v in vals))
    for r, q, g, rows in data["checkpoints"]:
        if not rows:
            continue
        lines += ["", f"  Q{q}, {r} game minutes left -- {g}"]
        cal = [checkpoint_wmae(rows, i) for i in range(len(labels))]
        for s in SELECTIONS[g]:
            lines.append(f"    WMAE {s:17s}" + "".join(
                f"{'-' if c.get(s) is None else f'{c[s]:.3f}':>{max(w, 36)}s}" for c in cal))
        cw = max(w, 36)
        lines.append(f"    {'band':22s}" + "".join(f"{x[:cw - 2]:>{cw}s}" for x in labels)
                     + "   (priced/won/ticks | expected/actual margin)")
        for row in rows:
            cells = []
            for f in row["feeds"]:
                if not f:
                    cells.append(f"{'-':>{cw}s}")
                    continue
                cal_ = f"{f['priced']:.3f}/{f['won']:.3f}/{f['n_cal']}" if f["n_cal"] else "-"
                mar = f"{_pct(f['expected'])}/{_pct(f['margin'])}" if f["n"] > MIN_MARGIN else "-"
                cells.append(f"{cal_ + ' | ' + mar:>{cw}s}")
            lines.append(f"    {row['selection'] + ' ' + row['band']:22s}" + "".join(cells))
    return lines


def _num(v, places=2):
    return "-" if v is None else f"{v:.{places}f}"


def _final(f):
    return "-" if not f else f"{f[0]:g}-{f[1]:g}"


TOP = 10


def match_summary(data, top=TOP):
    """Lines: per candidate and market, how the uplift splits over the matches."""
    rows = data.get("by_match")
    if not rows:
        return []
    labels = [f["label"] for f in data["feeds"]]
    n = len(labels)
    tot = match_totals(rows, n)
    lines = ["", "  BY MATCH -- each match's part of the uplift: its GGR over the candidate's whole handle"
                 " less prod's over prod's;",
             "  the parts add up to the uplift (every match: margin_uplift_matches.txt)"]
    for (f, g), part in parts(rows, n).items():
        order = sorted(part, key=part.get)
        worst, best = order[:top], order[::-1][:top]
        hp = sorted((r["feeds"][0][g]["handle"] for r in rows), reverse=True)
        half = next((k + 1 for k in range(len(hp)) if sum(hp[:k + 1]) >= sum(hp) / 2), 0)
        lines += ["", f"  {g}, {labels[f]} on prod: {_pct(uplift_of(tot, f, g), True)} over {len(rows):,} matches",
                  f"    the {top} matches that cost {labels[f]} most {sum(part[c] for c in worst):+.2f};"
                  f" the {top} that gain it most {sum(part[c] for c in best):+.2f}",
                  f"    without the {top} that cost most {_pct(uplift_of(match_totals(rows, n, set(worst)), f, g), True)};"
                  f" without those {2 * top} {_pct(uplift_of(match_totals(rows, n, set(worst) | set(best)), f, g), True)}",
                  f"    half of prod's handle is in its {half} biggest matches"]
        by = {r["code"]: r for r in rows}
        lines.append(f"    {'match':13s} {'final':>7s} {'ticks':>6s} {'prod margin':>12s} {labels[f] + ' margin':>12s}"
                     f" {'part':>7s}   {'prod most sure of a loser':>25s} {labels[f] + ' most sure':>14s}")
        for c in worst:
            r = by[c]
            pf, cf = r["feeds"][0][g], r["feeds"][f][g]
            lines.append(f"    {c:13s} {_final(r['final']):>7s} {int(pf['n']):>6,d}"
                         f" {_pct(margins(pf)[0]):>12s} {_pct(margins(cf)[0]):>12s} {part[c]:>+7.2f}"
                         f"   {_num(pf['top_loser'], 3):>25s} {_num(cf['top_loser'], 3):>14s}")
    return lines


def match_text(data, names=None):
    """Lines of text: every match, each market, sorted by the first candidate's part of the uplift."""
    rows = data.get("by_match") or []
    if not rows:
        return []
    names = names or {}
    labels = [f["label"] for f in data["feeds"]]
    n = len(labels)
    tot = match_totals(rows, n)
    pt = parts(rows, n)
    lines = ["", "  MARGIN UPLIFT BY MATCH -- the partner's dummy bets (every in-play tick a bet to win 100)",
             f"  {data.get('window') or 'the report window'}, {len(rows):,} matches, prod's ticks every feed"
             f" priced (matched; probabilities {PROB_RANGE[0]:g}-{PROB_RANGE[1]:g}); margin chart: {data['chart']}",
             "  ticks: dummy bets (both selections count); wt: the match's share of the feed's handle; GGR: the"
             " book's revenue on it;",
             "  Brier: the feed's ticks scored on the result (lower is better); sure: the highest probability"
             " the feed gave a selection that lost;",
             f"  part: the match's GGR over the candidate's whole handle less prod's over prod's, in margin"
             " points -- the parts add up to the uplift;",
             "  sorted by the first candidate's part, the matches that cost it most first; cum: the running sum"]
    for g in GROUPS:
        if (1, g) not in pt:
            continue
        order = sorted(rows, key=lambda r: pt[(1, g)][r["code"]])
        lines += ["", f"  {g}: " + "; ".join(
            f"{labels[f]} {_pct(margins(tot[f][g])[0])}" for f in range(n)) + "; uplift " + ", ".join(
            f"{labels[f]} {_pct(uplift_of(tot, f, g), True)}" for f in range(1, n))]
        head = f"    {'match':13s} {'start':11s} {'players':24s} {'final':>7s} {'ticks':>6s}"
        for lab in labels:
            head += f" | {lab[:10]:>10s} {'wt':>7s} {'GGR':>9s} {'margin':>8s} {'Brier':>6s} {'sure':>5s}"
        for f in range(1, n):
            head += f" | {labels[f][:8] + ' part':>13s} {'cum':>7s}"
        lines.append(head)
        cum = [0.0] * n
        for r in order:
            when, who = names.get(r["code"], ("", ""))
            line = (f"    {r['code']:13s} {when:11s} {who[:24]:24s} {_final(r['final']):>7s}"
                    f" {int(r['feeds'][0][g]['n']):>6,d}")
            for f in range(n):
                x = r["feeds"][f][g]
                h = tot[f][g]["handle"]
                br = x["brier"] / x["scored"] if x["scored"] else None
                line += (f" | {'':10s} {_num(100 * x['handle'] / h if h else None, 2):>6s}%"
                         f" {x['ggr']:>9,.0f} {_pct(margins(x)[0]):>8s} {_num(br, 3):>6s}"
                         f" {_num(x['top_loser'], 2):>5s}")
            for f in range(1, n):
                v = pt[(f, g)][r["code"]] if (f, g) in pt else 0.0
                cum[f] += v
                line += f" | {v:>+13.3f} {cum[f]:>+7.2f}"
            lines.append(line)
    return lines


def write_match_csv(path, data, names=None):
    """One row per match: each feed's dummy bets and scores per market, and each candidate's part."""
    rows = data.get("by_match") or []
    names = names or {}
    labels = [f["label"] for f in data["feeds"]]
    n = len(labels)
    pt = parts(rows, n)
    fields = ("n", "handle", "expected", "ggr", "margin", "brier", "logloss", "top_loser")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_code", "start", "players", "final_home", "final_away"]
                   + [f"{lab}_{g.lower()}_{k}" for lab in labels for g in GROUPS for k in fields]
                   + [f"{labels[f]}_{g.lower()}_part" for f in range(1, n) for g in GROUPS])
        for r in rows:
            when, who = names.get(r["code"], ("", ""))
            fin = r["final"] or ("", "")
            out = [r["code"], when, who, fin[0], fin[1]]
            for f in range(n):
                for g in GROUPS:
                    x = r["feeds"][f][g]
                    m = margins(x)[0]
                    out += [int(x["n"]), round(x["handle"], 2), round(x["expected"], 2), round(x["ggr"], 2),
                            "" if m is None else round(m, 3),
                            round(x["brier"] / x["scored"], 5) if x["scored"] else "",
                            round(x["logloss"] / x["scored"], 5) if x["scored"] else "",
                            "" if x["top_loser"] is None else round(x["top_loser"], 4)]
            out += [round(pt[(f, g)][r["code"]], 5) if (f, g) in pt else "" for f in range(1, n) for g in GROUPS]
            w.writerow(out)


def write_csv(path, data):
    """One row per feed, market, minute left, selection and band: the sums every table is off."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["feed", "market", "remaining", "quarter", "selection", "band"] + list(FIELDS))
        for f, cells in zip(data["feeds"], data["cells"]):
            for (g, r, s, b), c in sorted(cells.items(), key=lambda kv: (kv[0][0], -kv[0][1], kv[0][2],
                                                                         BANDS[kv[0][0]].index(kv[0][3]))):
                w.writerow([f["label"], g, r, quarter(r) if r > 0 else "", s, b]
                           + [round(c[k], 4) for k in FIELDS])


def html(data, esc):
    """The panel body: the headline, the calibration, then each checkpoint folded away."""
    if not data:
        return '<p class="dim">No margin uplift: no matched in-play ticks in every feed.</p>'
    labels = [f["label"] for f in data["feeds"]]
    head = "".join(f"<th>{esc(x)}</th>" for x in labels)
    rows = []
    sc = data.get("scores") or []
    for g, cells in headline(data):
        base = cells[0][0]
        rows.append(f"<tr class=\"split\"><th class=\"state\" colspan=\"{len(labels) + 1}\">{esc(g)}</th></tr>")
        rows.append("<tr><th>Margin</th>" + "".join(f"<td>{_pct(m)}</td>" for m, _, _ in cells) + "</tr>")
        rows.append("<tr><th>Expected margin</th>" + "".join(f"<td>{_pct(e)}</td>" for _, e, _ in cells) + "</tr>")
        rows.append("<tr><th>Actual less expected</th>" + "".join(
            _cell(None if m is None else m - e) for m, e, _ in cells) + "</tr>")
        rows.append("<tr><th>Uplift on prod</th>" + "".join(
            _cell(None if m is None or base is None else m - base) if i else "<td>&mdash;</td>"
            for i, (m, _, _) in enumerate(cells)) + "</tr>")
        rows.append("<tr><th>Bets (ticks)</th>" + "".join(f"<td>{n:,}</td>" for _, _, n in cells) + "</tr>")
        if sc:
            rows.append("<tr><th>Brier, same ticks</th>" + "".join(f"<td>{_num(x[g][0], 4)}</td>" for x in sc) + "</tr>")
            rows.append("<tr><th>Log loss, same ticks</th>" + "".join(f"<td>{_num(x[g][1], 4)}</td>" for x in sc) + "</tr>")
    cal = []
    for g in GROUPS:
        for s in SELECTIONS[g]:
            vals = [x.get((g, s)) for x in data["wmae"]]
            cal.append(f"<tr><th>{esc(g)} {esc(s)}</th>"
                       + "".join(f"<td>{'&mdash;' if v is None else f'{v[1]:.2f}%'}</td>" for v in vals) + "</tr>")
    minute_rows = []
    for g in GROUPS:
        for s in SELECTIONS[g]:
            minute_rows.append(f"<tr class=\"split\"><th class=\"state\" colspan=\"{len(labels) + 1}\">"
                               f"{esc(g)} {esc(s)}</th></tr>")
            for r in range(GAME_MINUTES, 0, -1):
                vals = [x.get((g, s, r)) for x in data["wmae"]]
                if any(vals):
                    minute_rows.append(f"<tr><th>{r} (Q{quarter(r)})</th>" + "".join(
                        f"<td>{'&mdash;' if v is None else f'{v[1]:.2f}% <span class=dim>{int(v[0]):,}</span>'}</td>"
                        for v in vals) + "</tr>")
    checks = []
    for r, q, g, crow in data["checkpoints"]:
        if not crow:
            continue
        wm = [checkpoint_wmae(crow, i) for i in range(len(labels))]
        body = [f"<tr><th>WMAE {esc(s)}</th>" + "".join(
            f"<td colspan=\"2\">{'&mdash;' if c.get(s) is None else f'{c[s]:.2f}%'}</td>" for c in wm) + "</tr>"
            for s in SELECTIONS[g]]
        for row in crow:
            tds = []
            for f in row["feeds"]:
                if not f:
                    tds.append("<td>&mdash;</td><td>&mdash;</td>")
                    continue
                cal_ = (f"{100 * f['priced']:.1f} / {100 * f['won']:.1f} <span class=dim>{f['n_cal']:,}</span>"
                        if f["n_cal"] >= MIN_CAL else "&mdash;")
                mar = (f"{_pct(f['expected'])} / {_pct(f['margin'])} <span class=dim>{f['n']:,}</span>"
                       if f["n"] > MIN_MARGIN else "&mdash;")
                tds.append(f"<td>{cal_}</td><td>{mar}</td>")
            body.append(f"<tr><th>{esc(row['selection'])} {esc(row['band'])}</th>{''.join(tds)}</tr>")
        sub = "".join(f"<th>Priced / won %</th><th>Expected / actual margin</th>" for _ in labels)
        checks.append(f"""
      <h4>Q{q}, {r} game minutes left &middot; {esc(g)}</h4>
      <table class="reach">
        <thead><tr><th></th>{''.join(f'<th colspan="2">{esc(x)}</th>' for x in labels)}</tr>
          <tr><th>Band</th>{sub}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>""")
    by = []
    if data.get("by_match"):
        lines = match_summary(data)
        by.append(f"""
    <details>
      <summary>By match: the matches that move the uplift most (every match in
        margin_uplift_matches.txt and .csv)</summary>
      <pre class="dim" style="overflow-x:auto">{esc(chr(10).join(lines))}</pre>
    </details>""")
    feeds = " &middot; ".join(f"{esc(f['label'])}: {f['ticks']:,} of {f['of']:,} ticks matched, the last {_mins(f['mean_end'])}"
                              " minutes after the scheduled start on average" for f in data["feeds"])
    return f"""
    <p class="dim">The partner's method: every live in-play price tick is a dummy bet to win
      {STAKE_TO_WIN:.0f} at the margin chart's price ({esc(data['chart'])}), settled on the result.
      Prod against each candidate, {esc(data.get('window') or 'the report window')}, on matched ticks
      only: prod's ticks where every candidate has a live price for the same match, market and
      selection no more than {PAIR_MAX_AGE:g} minutes old, and every feed's probability is
      {PROB_RANGE[0]:g}&ndash;{PROB_RANGE[1]:g} ({data['matches']:,} matches). The same moments and bets for
      all; only the probabilities differ. Minutes from the scheduled start &times;
      {data['factor']:.3f} make a {GAME_MINUTES}-minute game; the last {CUT_LAST} are left out.</p>
    <p class="dim">{feeds}</p>
    <table class="reach">
      <thead><tr><th></th>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="dim">Margin and uplift weight every tick by its stake to win {STAKE_TO_WIN:.0f}, so short
      prices dominate; Brier and log loss score the same ticks on the result, every tick alike (lower is
      better).</p>{''.join(by)}
    <h3>Calibration <span class="dim">weighted MAE, per game minute left, selection and band</span></h3>
    <table class="reach">
      <thead><tr><th></th>{head}</tr></thead>
      <tbody>{''.join(cal)}</tbody>
    </table>
    <details>
      <summary>By game minute left</summary>
      <table class="reach">
        <thead><tr><th>Minutes left</th>{head}</tr></thead>
        <tbody>{''.join(minute_rows)}</tbody>
      </table>
    </details>
    <details>
      <summary>Checkpoints: each band's calibration (every feed &ge; {MIN_CAL} ticks) and margin
        (&gt; {MIN_MARGIN} ticks)</summary>
      {''.join(checks)}
    </details>"""


def _cell(v):
    if v is None:
        return "<td>&mdash;</td>"
    cls = "good" if v > 0.05 else "bad" if v < -0.05 else ""
    return f'<td class="{cls}">{v:+.2f}</td>'


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
