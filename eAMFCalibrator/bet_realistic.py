"""A betting simulation closer to how bettors behave than re-pricing the same bets: who would still
have bet against the candidate's price, which book a bettor would take when both quote, and how
sure each margin is.

  same bets   the existing re-sim: every bet placed again at the candidate's price (the operator's
              margin over the price kept: odds x prod / candidate), at the same stake
  acceptance  sharp accounts (config.SIM_SHARP_GROUPS, Restricted by default) bet for an edge: each
              is given its segment's measured edge over prod's price by market (bettors' return on
              the bets, shrunk toward the segment's), so a belief of (1 + edge) / odds. At the
              candidate's odds the edge is belief x candidate odds - 1: none left, no bet; else the
              stake moves with the edge (stake x edge now / edge then, at most SIM_MAX_SCALE).
              Everyone else (VIP, Standard, none) has little edge and does not shop on price: the
              bet stands, its stake x (candidate odds / odds) ^ SIM_ELASTICITY (0: as placed)
  two books   prod and the candidate both quote; a bettor takes whichever pays more on the bet's
              selection, where both quote the same line (moneyline always; a line market only
              where the candidate's line is the bet's). Each book is scored on what it is left
              with: a book generous in the wrong places is picked off
  intervals   95% bootstrap over matches (SIM_BOOT resamples) on each margin and its change

Revenue is the book's (positive: the book won).
"""

from collections import defaultdict

from . import bets, config

SHARP, RECREATIONAL = "sharp", "recreational"


def segment(row):
    """sharp or recreational off the customer temperature."""
    t = str(row.get(config.BET_VIP_COLUMN) or "").strip().lower()
    return SHARP if t in {g.lower() for g in config.SIM_SHARP_GROUPS} else RECREATIONAL


def _settled(r):
    return r["result"] in (bets.WON, bets.LOST, bets.PUSH)


def edges(rows, prior=None):
    """{(segment, market): bettors' return over prod's price on their settled, re-priced bets},
    each market shrunk toward its segment's by `prior` bets (config.SIM_EDGE_PRIOR)."""
    prior = config.SIM_EDGE_PRIOR if prior is None else prior
    seg = defaultdict(lambda: [0, 0.0, 0.0])
    cell = defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        if not r["simulated"] or not _settled(r):
            continue
        for key, d in ((segment(r), seg), ((segment(r), r["market"]), cell)):
            c = d[key]
            c[0] += 1
            c[1] += r["stake"]
            c[2] -= r["revenue"]            # the bettors' return
    out = {}
    for (sg, mk), (n, stake, ret) in cell.items():
        s = seg[sg]
        base = s[2] / s[1] if s[1] else 0.0
        own = ret / stake if stake else base
        out[(sg, mk)] = (n * own + prior * base) / (n + prior)
    for sg, (n, stake, ret) in seg.items():
        out[(sg, None)] = ret / stake if stake else 0.0
    return out


def scale(row, edge):
    """The candidate's stake over the real one for this bet: 0 where the bettor would not bet."""
    o, oc = row["odds"], row["candidate_odds"]
    if not o or not oc:
        return 0.0
    if segment(row) == SHARP and edge > 0:
        ev_c = (1 + edge) / o * oc - 1
        return 0.0 if ev_c <= 0 else min(config.SIM_MAX_SCALE, ev_c / edge)
    return (oc / o) ** config.SIM_ELASTICITY


def accepted(rows, edge_by):
    """The re-priced bets as the candidate would have taken them: (row, stake, revenue) for each,
    the revenue the candidate's at the scaled stake."""
    out = []
    for r in rows:
        if not r["simulated"] or not _settled(r):
            continue
        e = edge_by.get((segment(r), r["market"]), edge_by.get((segment(r), None), 0.0))
        k = scale(r, e)
        out.append((r, r["stake"] * k, r["candidate_revenue"] * k))
    return out


def two_books(rows):
    """Each re-priced bet quoted by both books at the same line, given to the one that pays more
    (prod on a tie): (row, book, stake, revenue) with book 'prod' or 'candidate'."""
    out = []
    for r in rows:
        if not r["simulated"] or not _settled(r) or not r.get("same_line"):
            continue
        if r["candidate_odds"] and r["candidate_odds"] > r["odds"]:
            out.append((r, "candidate", r["stake"], r["candidate_revenue"]))
        else:
            out.append((r, "prod", r["stake"], r["revenue"]))
    return out


def _per_match(items):
    """{match: [stake, revenue]} off (match, stake, revenue) items."""
    out = defaultdict(lambda: [0.0, 0.0])
    for m, s, v in items:
        out[m][0] += s
        out[m][1] += v
    return out


def boot(series, n=None, seed=0):
    """95% intervals over matches for several margins at once. `series` is {name: {match: [stake,
    revenue]}}; every name is resampled on the same matches, so a difference keeps its pairing.
    Returns {name: (margin %, lo, hi)}, with the draws under "_draws" for diff()."""
    import numpy as np
    n = config.SIM_BOOT if n is None else n
    matches = sorted(set().union(*[set(s) for s in series.values()]))
    weights = np.random.default_rng(seed).multinomial(len(matches), [1 / len(matches)] * len(matches),
                                                       size=n).astype(float)
    out, draws = {}, {}
    for k, s in series.items():
        st = np.array([s[m][0] if m in s else 0.0 for m in matches])
        rv = np.array([s[m][1] if m in s else 0.0 for m in matches])
        ws, wr = weights @ st, weights @ rv
        d = np.where(ws > 0, 100 * wr / np.where(ws > 0, ws, 1), np.nan)
        draws[k] = d
        ok = np.sort(d[~np.isnan(d)])
        out[k] = (100 * rv.sum() / st.sum() if st.sum() else float("nan"),
                  float(ok[int(0.025 * len(ok))]) if len(ok) else float("nan"),
                  float(ok[int(0.975 * len(ok)) - 1]) if len(ok) else float("nan"))
    out["_draws"] = draws
    return out


def diff(b, a, c):
    """(c's margin - a's, 95% interval) off a boot result."""
    import numpy as np
    d = b["_draws"][c] - b["_draws"][a]
    d = np.sort(d[~np.isnan(d)])
    if not len(d):
        return float("nan"), float("nan"), float("nan")
    return b[c][0] - b[a][0], float(d[int(0.025 * len(d))]), float(d[int(0.975 * len(d)) - 1])


def _cuts(rows):
    """(label, keep) cuts: all, each segment, in play / pre-match, each market."""
    out = [("all", lambda r: True), ("sharp", lambda r: segment(r) == SHARP),
           ("recreational", lambda r: segment(r) == RECREATIONAL),
           ("pre-match", lambda r: not r.get("in_play")), ("in play", lambda r: bool(r.get("in_play")))]
    for mk in sorted({str(r["market"]) for r in rows if r.get("market")}):
        out.append((mk, lambda r, mk=mk: str(r["market"]) == mk))
    return out


def _m(x):
    return f"{x[0]:6.2f}% [{x[1]:+6.2f}, {x[2]:+6.2f}]"


def report(results):
    """Lines of text: for each candidate, the same-bets re-sim, the acceptance sim and the
    two-book test, by cut, each with its 95% interval over matches."""
    L = ["\n  realistic betting simulation (95% intervals over matches; margin = the book's)",
         f"  sharp: {', '.join(config.SIM_SHARP_GROUPS)} (bet for an edge, shop on price); "
         f"everyone else bets as placed (elasticity {config.SIM_ELASTICITY:g}); "
         f"stake at most x{config.SIM_MAX_SCALE:g}"]
    for name, rows in results:
        e = edges(rows)
        acc = accepted(rows, e)
        two = two_books(rows)
        sharp_edges = ", ".join(f"{mk or 'all'} {100 * v:+.1f}%" for (sg, mk), v in sorted(
            e.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))) if sg == SHARP)
        L.append(f"\n  == {name} ==")
        L.append(f"  sharp accounts' edge over prod's price: {sharp_edges or 'none re-priced'}")
        L.append(f"\n  {'':14s} {'bets':>7s} {'stake':>11s}   {'prod (as bet)':>26s}   "
                 f"{'same bets':>26s}   {'acceptance':>26s}  {'kept':>6s}   {'change (acceptance)':>26s}")
        for label, keep in _cuts(rows):
            base = [r for r in rows if r["simulated"] and _settled(r) and keep(r)]
            if not base:
                continue
            a = [x for x in acc if keep(x[0])]
            series = {"prod": _per_match((r["match_code"], r["stake"], r["revenue"]) for r in base),
                      "same": _per_match((r["match_code"], r["stake"], r["candidate_revenue"]) for r in base),
                      "acc": _per_match((r["match_code"], s, v) for r, s, v in a)}
            b = boot(series)
            stake = sum(r["stake"] for r in base)
            kept = sum(s for _, s, _ in a) / stake if stake else 0.0
            L.append(f"  {label[:14]:14s} {len(base):7,d} {stake:11,.0f}   {_m(b['prod']):>26s}   "
                     f"{_m(b['same']):>26s}   {_m(b['acc']):>26s}  {100 * kept:5.1f}%   "
                     f"{_m(diff(b, 'prod', 'acc')):>26s}")
        L.append("\n  two books: each bet both quote at the same line goes to the one that pays more")
        L.append(f"  {'':14s} {'bets':>7s} {'stake':>11s}  {'to cand.':>8s}   {'prod keeps':>26s}   "
                 f"{'candidate takes':>26s}   {'both books':>26s}   {'prod alone':>26s}   "
                 f"{'change (both - alone)':>26s}")
        for label, keep in _cuts(rows):
            t = [x for x in two if keep(x[0])]
            if not t:
                continue
            series = {"prod": _per_match((r["match_code"], s, v) for r, bk, s, v in t if bk == "prod"),
                      "cand": _per_match((r["match_code"], s, v) for r, bk, s, v in t if bk == "candidate"),
                      "both": _per_match((r["match_code"], s, v) for r, bk, s, v in t),
                      "alone": _per_match((r["match_code"], r["stake"], r["revenue"]) for r, _, _, _ in t)}
            series = {k: v for k, v in series.items() if v}
            b = boot(series)
            stake = sum(s for _, _, s, _ in t)
            to_c = sum(s for _, bk, s, _ in t if bk == "candidate") / stake if stake else 0.0
            cell = lambda k: _m(b[k]) if k in b else f"{'-':>26s}"
            L.append(f"  {label[:14]:14s} {len(t):7,d} {stake:11,.0f}  {100 * to_c:7.1f}%   "
                     f"{cell('prod'):>26s}   {cell('cand'):>26s}   {cell('both'):>26s}   "
                     f"{cell('alone'):>26s}   {_m(diff(b, 'alone', 'both')):>26s}")
    L.append("\n  same bets: every bet again at the candidate's price; acceptance: sharp bets dropped or "
             "resized by the edge left at the candidate's odds; kept: the acceptance stake over the real "
             "stake; two books: 'both books' is the two books' margin together on the bets, set against "
             "prod taking them all alone")
    return L
