"""Who starts a run of bets: by customer id, whether the money that comes after a customer's bet follows it.

Every bet with a customer id (config.BET_CUSTOMER_COLUMN, --customer-column) is a possible leader. After it,
the other customers' money on the same match and market toward its side (home or over +1, away or under -1),
stake on it less on the other over both, -1 to +1, over the next window (in play `after` seconds, pre-match
PRE_AFTER), on prod's clock (bet time less the operator's lag) -- the customer's own bets left out, so one
punter splitting a stake is not a run. The money comes in runs anyway: after a score, after a price move,
everyone bets the same side. So each bet's follow is set against what the money before it (the other
customers', the same window back) and a score in the last 30 seconds say it should be, by market, pre-match and
in play apart. What is left, a customer's lead: above 0, the money after their bets goes their way more than
the moment says.

  - The lead of each customer: over their matches (each match their bets' mean), with its interval, and
    shrunk toward 0 by how much customers really differ (the spread of the leads beyond their noise).
  - Is it a trait? Each customer's matches split alternately in two, the leads of the halves correlated across
    customers. Near 0, nobody leads; who looked like it, was luck.
  - Runs started: bets that were first on their side (no other customer's bet that side in the window
    before) and had RUN_MIN or more other customers on that side in the window after.
  - Do they know? Their own edge (won less prod's chance of the side, in points) and the book's margin on
    them; and the same on the followers' bets (the other customers' bets on that side in the window after).
  - Pairs: one customer betting the side another just bet, within the window, on PAIR_MIN or more matches --
    a tip shared, or one punter on two accounts -- with the matches both bet and the other way round."""

import datetime as dt
from collections import Counter, defaultdict

import numpy as np

from . import bets, config

GROUP = {50: "moneyline", 51: "moneyline", 52: "spread", 53: "spread", 54: "total", 55: "total"}
SIGN = {50: 1, 51: -1, 52: 1, 53: -1, 54: 1, 55: -1}
AFTER = 120.0
PRE_AFTER = 1800.0
RECENT = 30.0
RUN_MIN = 3
MIN_MATCHES = 10
PAIR_MIN = 3
TOP = 25
BOOT = 1000
CUSTOMER_WORDS = ("CUSTOMER", "ACCOUNT", "PUNTER", "USER", "CLIENT")


def customer_of(b, column):
    """The bet's customer id as text, or None (missing or blank)."""
    v = b.extra.get(column)
    v = None if v is None else str(v).strip()
    return v or None


def records(every, joined, lags, column, checks=None):
    """Every bet on a side as a record: its cell (match, market, in play), its time on prod's clock, its side,
    stake and customer; settled, its result against prod's chance of its side and its revenue."""
    from .bet_money import since_score
    out = []
    for b, r in zip(every, joined):
        g = GROUP.get(b.feed_market)
        if g is None or b.time is None or not b.stake:
            continue
        seen = b.time - dt.timedelta(seconds=bets.bet_lag(b, lags))
        settled = r.get("result") in (bets.WON, bets.LOST) and not r.get("excluded") and r.get("stream_prob") is not None
        recent = 0.0
        if b.in_play and checks is not None:
            s = since_score(checks, b.match_code, seen)
            recent = float(s is not None and s <= RECENT)
        out.append(dict(match=b.match_code, group=g, in_play=bool(b.in_play), t=bets.epoch(seen), side=SIGN[b.feed_market],
                        stake=float(b.stake), customer=customer_of(b, column), operator=b.operator,
                        temperature=str(b.extra.get(config.BET_VIP_COLUMN) or ""), recent=recent, settled=settled,
                        won=(1.0 if r.get("result") == bets.WON else 0.0) if settled else None,
                        p=r.get("stream_prob") if settled else None, revenue=float(r.get("revenue") or 0.0)))
    return out


def follow(recs, after=AFTER, pre_after=PRE_AFTER):
    """Each record gains the other customers' money toward its side over the window after it and the one
    before (None with no stake there), their stake after, whether it was first on its side, the other
    customers on its side after, and the indices of their bets (its followers)."""
    cells = defaultdict(list)
    for i, x in enumerate(recs):
        cells[(x["match"], x["group"], x["in_play"])].append(i)
    for (_, _, in_play), ix in cells.items():
        ix.sort(key=lambda i: recs[i]["t"])
        w = after if in_play else pre_after
        t = np.array([recs[i]["t"] for i in ix])
        side = np.array([recs[i]["side"] for i in ix], float)
        stake = np.array([recs[i]["stake"] for i in ix])
        cust = [recs[i]["customer"] for i in ix]
        for k, i in enumerate(ix):
            x = recs[i]
            me = x["customer"]
            a0, a1 = np.searchsorted(t, x["t"], "right"), np.searchsorted(t, x["t"] + w, "right")
            b0, b1 = np.searchsorted(t, x["t"] - w, "left"), np.searchsorted(t, x["t"], "left")
            others_a = [j for j in range(a0, a1) if me is None or cust[j] != me]
            others_b = [j for j in range(b0, b1) if me is None or cust[j] != me]
            toward = x["side"] * side
            sa = stake[others_a].sum() if others_a else 0.0
            sb = stake[others_b].sum() if others_b else 0.0
            x["after"] = float((toward[others_a] * stake[others_a]).sum() / sa) if sa > 0 else None
            x["before"] = float((toward[others_b] * stake[others_b]).sum() / sb) if sb > 0 else 0.0
            x["after_stake"] = float(sa)
            x["first"] = not any(toward[j] > 0 for j in others_b)
            mine = [j for j in others_a if toward[j] > 0]
            x["followers"] = len({cust[j] if cust[j] is not None else ("?", j) for j in mine})
            x["follow_ix"] = [ix[j] for j in mine] if me is not None else []
    return recs


def residuals(recs):
    """Each record with money after gains its lead: the follow less what the money before and a recent score
    say, fitted by market, pre-match and in play apart."""
    by = defaultdict(list)
    for i, x in enumerate(recs):
        x["lead"] = None
        if x.get("after") is not None:
            by[(x["group"], x["in_play"])].append(i)
    for ix in by.values():
        if len(ix) < 10:
            continue
        y = np.array([recs[i]["after"] for i in ix])
        X = np.c_[np.ones(len(ix)), [recs[i]["before"] for i in ix], [recs[i]["recent"] for i in ix]]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        for i, e in zip(ix, y - X @ coef):
            recs[i]["lead"] = float(e)
    return recs


def _per_match(rows):
    """{match: the mean lead of the rows on it}."""
    m = defaultdict(list)
    for x in rows:
        if x["lead"] is not None:
            m[x["match"]].append(x["lead"])
    return {k: float(np.mean(v)) for k, v in m.items()}


def _tau2(means, ses):
    """DerSimonian-Laird: the variance of the true leads beyond the noise."""
    means, ses = np.asarray(means), np.asarray(ses)
    w = 1 / np.maximum(ses ** 2, 1e-12)
    mu = (w * means).sum() / w.sum()
    q = (w * (means - mu) ** 2).sum()
    c = w.sum() - (w ** 2).sum() / w.sum()
    return max(0.0, (q - (len(means) - 1)) / c) if c > 0 else 0.0


def customers(recs, min_matches=MIN_MATCHES):
    """Each customer with min_matches or more matches with a lead: bets, matches, stake, the lead [95%] and
    shrunk, runs started of firsts, their own edge and the book's margin on them, and their followers'."""
    by = defaultdict(list)
    for i, x in enumerate(recs):
        if x["customer"] is not None:
            by[x["customer"]].append(i)
    out = []
    for c, ix in by.items():
        rows = [recs[i] for i in ix]
        pm = _per_match(rows)
        if len(pm) < min_matches:
            continue
        v = np.array(list(pm.values()))
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        settled = [x for x in rows if x["settled"]]
        fol = sorted({j for x in rows for j in x["follow_ix"]})
        fs = [recs[j] for j in fol if recs[j]["settled"]]
        firsts = [x for x in rows if x["first"]]
        stake = sum(x["stake"] for x in rows)
        out.append(dict(
            customer=c, bets=len(rows), matches=len({x["match"] for x in rows}), stake=stake,
            temperature=Counter(x["temperature"] for x in rows).most_common(1)[0][0],
            operator=Counter(str(x["operator"]) for x in rows).most_common(1)[0][0],
            in_play=sum(x["in_play"] for x in rows) / len(rows), lead=float(v.mean()), se=se,
            lead_matches=len(v), firsts=len(firsts), runs=sum(x["followers"] >= RUN_MIN for x in firsts),
            after_stake=float(np.mean([x["after_stake"] for x in rows])),
            edge=100 * float(np.mean([x["won"] - x["p"] for x in settled])) if settled else float("nan"),
            margin=100 * sum(x["revenue"] for x in rows) / stake if stake else float("nan"),
            followers=len(fs), follower_edge=100 * float(np.mean([x["won"] - x["p"] for x in fs])) if fs else float("nan"),
            follower_margin=(100 * sum(x["revenue"] for x in fs) / sum(x["stake"] for x in fs)) if fs else float("nan")))
    ok = [r for r in out if np.isfinite(r["se"]) and r["se"] > 0]
    tau2 = _tau2([r["lead"] for r in ok], [r["se"] for r in ok]) if len(ok) >= 3 else 0.0
    for r in out:
        k = tau2 / (tau2 + r["se"] ** 2) if np.isfinite(r["se"]) and tau2 + r["se"] ** 2 > 0 else 0.0
        r["shrunk"] = r["lead"] * k
    return out, tau2


def trait(recs, min_half=5, B=BOOT, seed=0):
    """Each customer's matches split alternately (in the order they came) into two halves: the correlation
    of the halves' leads across the customers with min_half or more matches in each, and its 95% interval
    over customers."""
    by = defaultdict(list)
    for x in recs:
        if x["customer"] is not None and x["lead"] is not None:
            by[x["customer"]].append(x)
    a, b = [], []
    for rows in by.values():
        pm = _per_match(rows)
        first = {}
        for x in rows:
            first[x["match"]] = min(first.get(x["match"], x["t"]), x["t"])
        order = sorted(pm, key=first.get)
        h0, h1 = [pm[m] for m in order[0::2]], [pm[m] for m in order[1::2]]
        if len(h0) >= min_half and len(h1) >= min_half:
            a.append(np.mean(h0))
            b.append(np.mean(h1))
    n = len(a)
    if n < 5:
        return dict(n=n, r=float("nan"), lo=float("nan"), hi=float("nan"))
    a, b = np.array(a), np.array(b)
    r = float(np.corrcoef(a, b)[0, 1])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        k = rng.integers(0, n, n)
        if a[k].std() > 0 and b[k].std() > 0:
            draws.append(np.corrcoef(a[k], b[k])[0, 1])
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (float("nan"), float("nan"))
    return dict(n=n, r=r, lo=float(lo), hi=float(hi))


def pairs(recs, min_matches=PAIR_MIN):
    """(leader, follower): the matches where the follower bet the side the leader just had (within the
    window), with min_matches or more of them; the matches both bet, and the matches the other way round."""
    led = defaultdict(set)
    for x in recs:
        if x["customer"] is None:
            continue
        for j in x["follow_ix"]:
            f = recs[j]["customer"]
            if f is not None and f != x["customer"]:
                led[(x["customer"], f)].add(x["match"])
    keep = {k: v for k, v in led.items() if len(v) >= min_matches}
    if not keep:
        return []
    played = defaultdict(set)
    for x in recs:
        if x["customer"] is not None:
            played[x["customer"]].add(x["match"])
    out = [dict(leader=l, follower=f, followed=len(v), together=len(played[l] & played[f]),
                back=len(led.get((f, l), ()))) for (l, f), v in keep.items()]
    out.sort(key=lambda r: (-r["followed"], -r["followed"] / max(r["together"], 1)))
    return out


def coverage(every, column):
    """Bets with a customer id, by operator and by day: (bets, with an id, distinct ids)."""
    by_op, by_day = defaultdict(lambda: [0, 0, set()]), defaultdict(lambda: [0, 0, set()])
    for b in every:
        c = customer_of(b, column)
        for cell in (by_op[str(b.operator)], by_day[b.time.date() if b.time else None]):
            cell[0] += 1
            if c is not None:
                cell[1] += 1
                cell[2].add(c)
    return ({k: (v[0], v[1], len(v[2])) for k, v in by_op.items()},
            {k: (v[0], v[1], len(v[2])) for k, v in by_day.items()})


def analyse(every, joined, lags, column, checks=None, after=AFTER, min_matches=MIN_MATCHES):
    """Everything the section reports, off the window's bets."""
    by_op, by_day = coverage(every, column)
    if not any(v[1] for v in by_op.values()):
        return dict(column=column, by_op=by_op, by_day=by_day, ids=0)
    recs = residuals(follow(records(every, joined, lags, column, checks), after))
    custs, tau2 = customers(recs, min_matches)
    leads = [x["lead"] for x in recs if x["lead"] is not None]
    firsts = [x for x in recs if x["customer"] is not None and x["first"]]
    return dict(column=column, by_op=by_op, by_day=by_day, after=after,
                ids=len({x["customer"] for x in recs if x["customer"] is not None}),
                bets_with=sum(x["customer"] is not None for x in recs), with_lead=len(leads),
                run_rate=(sum(x["followers"] >= RUN_MIN for x in firsts) / len(firsts)) if firsts else float("nan"),
                customers=custs, tau=float(np.sqrt(tau2)), trait=trait(recs), pairs=pairs(recs),
                min_matches=min_matches, recs=recs)


def _short(c):
    return c if len(c) <= 12 else c[:10] + ".."


def _f(v, fmt):
    """v formatted, or '-' (a trailing % in fmt is kept as a sign, not a scaling)."""
    if v is None or not np.isfinite(v):
        return "-"
    return format(v, fmt[:-1]) + "%" if fmt.endswith("%") else format(v, fmt)


def report(out):
    """The section's lines."""
    col = out["column"]
    L = [f"  WHO STARTS A RUN, by customer id ({col}): after each bet, the other customers' money on its side over the "
         f"next {out.get('after', AFTER):g}s in play ({PRE_AFTER:g}s pre-match), against what the money before and a "
         "recent score say; the customer's own bets left out"]
    days = sorted((d for d in out["by_day"] if d is not None))
    with_id = [d for d in days if out["by_day"][d][1]]
    L.append("    with an id, by operator: " + "; ".join(
        f"{op} {v[1]:,} of {v[0]:,} bets ({v[2]:,} ids)" for op, v in sorted(out["by_op"].items())))
    if not out["ids"]:
        L.append(f"    no bet in the window has a {col}: nothing to probe (bets probe lists the bet table's columns)")
        return L
    L.append(f"    days with ids: {with_id[0]} to {with_id[-1]} ({len(with_id)} of {len(days)} days)" if with_id else "")
    L.append(f"    {out['bets_with']:,} bets with an id from {out['ids']:,} customers; {out['with_lead']:,} bets with other"
             f" customers' money after them")
    t = out["trait"]
    L.append(f"    is it a trait? the leads of alternate matches correlated over {t['n']:,} customers: "
             f"r {_f(t['r'], '+.3f')} [{_f(t['lo'], '+.3f')}, {_f(t['hi'], '+.3f')}]"
             + ("  -> a trait: some customers' bets are followed" if np.isfinite(t["lo"]) and t["lo"] > 0 else
                "  -> not a trait: who looks like a leader is luck" if np.isfinite(t["r"]) else ""))
    L.append(f"    the leads' spread beyond their noise (tau) {out['tau']:.3f}; a first bet on its side starts a run "
             f"({RUN_MIN}+ other customers after it) {_f(100 * out['run_rate'], '.1f')}% of the time")
    cs = sorted(out["customers"], key=lambda r: -r["shrunk"])
    L.append(f"    the {min(TOP, len(cs))} customers led the most, shrunk ({len(cs):,} with {out['min_matches']}+ matches):")
    L.append(f"      {'customer':12s} {'temp':>10s} {'bets':>6s} {'matches':>7s} {'stake':>9s} {'in play':>7s}"
             f" {'lead [95%]':>22s} {'shrunk':>7s} {'runs/firsts':>11s} {'edge':>6s} {'margin':>7s}"
             f" {'followers':>9s} {'f edge':>6s} {'f margin':>8s}")
    for r in cs[:TOP]:
        lo, hi = r["lead"] - 1.96 * r["se"], r["lead"] + 1.96 * r["se"]
        runs = f"{r['runs']}/{r['firsts']}"
        L.append(f"      {_short(r['customer']):12s} {r['temperature'][:10]:>10s} {r['bets']:6,d} {r['matches']:7,d}"
                 f" {r['stake']:9,.0f} {100 * r['in_play']:6.0f}% {r['lead']:+.3f} [{_f(lo, '+.3f')}, {_f(hi, '+.3f')}]"
                 f" {r['shrunk']:+7.3f} {runs:>11s}"
                 f" {_f(r['edge'], '+.1f'):>6s} {_f(r['margin'], '+.1f%'):>7s} {r['followers']:9,d}"
                 f" {_f(r['follower_edge'], '+.1f'):>6s} {_f(r['follower_margin'], '+.1f%'):>8s}")
    good = [r for r in cs if np.isfinite(r["edge"])]
    if len(good) >= 10:
        rho = np.corrcoef([r["shrunk"] for r in good], [r["edge"] for r in good])[0, 1]
        L.append(f"    do the leaders know? the shrunk lead against the customer's own edge, over {len(good):,} customers: "
                 f"r {rho:+.3f}")
    ps = out["pairs"]
    L.append(f"    pairs: one customer on the side another just bet, on {PAIR_MIN}+ matches ({len(ps):,} pairs); the top 15:")
    if ps:
        L.append(f"      {'leader':12s} {'follower':12s} {'followed':>8s} {'both bet':>8s} {'back':>5s}")
        for r in ps[:15]:
            L.append(f"      {_short(r['leader']):12s} {_short(r['follower']):12s} {r['followed']:8,d} {r['together']:8,d}"
                     f" {r['back']:5,d}")
    L.append("    lead: the other customers' money toward the bet's side after it, less what the moment says (-2 to +2), the"
             " mean over the customer's matches; shrunk: toward 0 by how much leads really differ; runs/firsts: of their"
             " bets first on a side, those 3+ other customers followed; edge: won less prod's chance, points; followers:"
             " the other customers' bets on that side after theirs, f edge and f margin on those; followed: matches the"
             " follower bet the leader's side within the window after; both bet: matches both bet; back: the other way")
    return L


def customer_rows(out):
    """The customers' table, for the CSV."""
    return sorted(out.get("customers", []), key=lambda r: -r["shrunk"])


def find_column(cols, column):
    """The configured column if the bet table has it; else None and the columns that look like a customer's."""
    if column in cols:
        return column, []
    return None, [c for c in cols if any(w in str(c).upper() for w in CUSTOMER_WORDS)]
