"""The pre-match bets replayed at each prior's prices: what the book would have made had it priced
the pre-match markets with glmer or NB2 (each with and without the rematch layer) instead of prod.

    python -m eAMFCalibrator prior-replay --since "2026-09-01 00:00:00" --until "2026-10-15 00:00:00" \\
        --arm "glmer-daily=glmer/out/cadence/cadence_matches.csv@trim:hl60_x0.5:daily" \\
        --arm "nb2=rolling:eAMFModel/out/nb2_daily"

The arms are prior-backtest's (NAME=SOURCE[#MODE]) and are read the same way: each match's expected
points, its score model, and the layers (prior_backtest.LAYERS: +layer the rematch layer, +infer the
inference layer, +both the two as one; with the gamers' stats, +feat the feature layer and +both+feat
it on +both) fitted walk-forward on the arm's own errors. prod -- a third
party's price -- is the benchmark: the bets' own prices.

Every pre-match bet (by the operator's in-play flag, and placed before kick-off) on the matches every
arm priced, after the layer's burn-in, is re-priced as `bets prematch` re-prices a candidate stream
(bets.join and bets.reprice): the same selection and stake at the same line, prod's probability at
bet time (its quote then, whatever its message count) swapped for the arm's at that line, the
operator's margin kept (odds x prod / arm), settled as the bet settled. An arm's moneyline or spread
is its home margin over the line, and its total its total over the line, pushes left out
(score_model's p_margin and p_over). An arm's chance is held within [PROB_FLOOR, 1 - PROB_FLOOR], so
a far line can't make the odds run away.

An arm prices a match as of its kick-off, off every result before it. A bet placed before either
gamer's previous result was in (bets.RESULT_IN after that match's kick-off) would be re-priced with
results neither the bettor nor prod's price had then. Those bets are reported apart. The headline is
the clean bets, placed with both gamers' earlier results in. (The inference layer also reads the
opponents' other games whose results were in by kick-off; for a clean bet placed before one of those
came in, that is a little later than the bettor had.)

The bets are the ones that were placed, at their stakes: nobody bets differently at the arm's prices.
Only the bets that won move the book: a shorter price on one is revenue kept, a longer one revenue
given away, and a lost bet loses its stake at any price. Bets that would not have been placed at the
arm's price, and bets that would have been placed instead, are not modelled.

Reported: prod's margin on the bets (revenue / stake) and each variant's change against it, in
points of stake, with a 95% interval from resampling matches (config.SIM_BOOT) and the weeks it came
out ahead. This is given overall and by market for the clean bets, then for the clean bets by meeting
(first / 2nd / 3rd+), operator, VIP and the rest, and week; then every pre-match bet, and the stake by
minutes before kick-off.
-> prior_replay.txt, prior_replay.csv (a row per bet with each variant's probability, odds and
revenue)
"""

import os
from bisect import bisect_left
from collections import Counter, defaultdict
import numpy as np

from . import bets, config, markets, prior_backtest as pb, score_model
from .snowflake_io import fetch_all

PROB_FLOOR = 0.005
MINUTE_BANDS = ((5, "under 5 min"), (15, "5-15 min"), (60, "15-60 min"), (None, "an hour or more"))


def _hold(p):
    return min(max(p, PROB_FLOOR), 1 - PROB_FLOOR)


def arm_chance(book, market, line, dm=0.0, dt_=0.0):
    """An arm's chance of a selection at a line, pushes left out, with its layer's move (dm on the
    margin, dt_ on the total); None for a line market without a line."""
    group = markets.market_group(market)
    if group == markets.MONEYLINE:
        p = book.p_margin(0.0, dm)
        return None if p is None else p if market == 50 else 1 - p
    if line is None:
        return None
    if group == markets.SPREAD:
        if market == 52:
            return book.p_margin(line, dm)
        # 53: the away margin over the line (literal), or the NO of 52 (complement)
        p = book.p_margin(line if markets.spread_resolution() == "complement" else -line, dm)
        return None if p is None else 1 - p
    p = book.p_over(line, dt_)
    return None if p is None else p if market == 54 else 1 - p


def price_bets(rows, price, n):
    """Each bet (bets.join's row against prod's quote at the bet) re-priced by the n variants a price
    function gives -- price(match, market, line, prod's chance) -- at the line of the prod quote the
    bet saw, whatever its message count (prod's pre-match quotes carry one), the operator's margin
    kept and settled as the bet settled (bets.reprice). Returns n lists, one dict per bet: simulated,
    candidate_prob, candidate_odds, candidate_revenue. The bets are walked match by match, so a price
    can keep one match's work."""
    out = [[None] * len(rows) for _ in range(n)]
    for i in sorted(range(len(rows)), key=lambda i: str(rows[i]["match_code"])):
        r = rows[i]
        p = r["stream_prob"]
        got = price(r["match_code"], r["feed_market"], r["stream_line"], p) if p is not None else (None,) * n
        for k, c in enumerate(got):
            if c is None:
                out[k][i] = dict(simulated=False, candidate_prob=None, candidate_odds=None, candidate_revenue=None)
                continue
            c = _hold(c)
            out[k][i] = dict(candidate_prob=c,
                             **bets.reprice(dict(r, candidate_prob=c, candidate_result=r["result"])))
    return out


def variants(ctx, arms, draws):
    """[([variant names], price)] -- each arm as it is and with each layer (prior_backtest.active_layers),
    all read off the same simulated scores -- with {match} every one of them priced after the burn-in,
    {match: what its gamers had played that session}, and the layers. prod is the benchmark: the
    bets' own prices, not re-priced."""
    prod_rows = None
    found, layers = [], {}
    for name, spec in [(pb.PROD, None)] + list(arms):
        mode, own, rows, layer = pb.fit_arm(name, spec, ctx)
        layers[name] = layer
        by = {r["match"]: r for r in rows}
        if name == pb.PROD:
            prod_rows = by
            continue
        fits = pb.fitted_by_day(rows) if mode == "fitted" else {}
        found.append((name, mode, own, fits, by))
    burn = ctx["burn_from"]
    codes = {c for c, r in prod_rows.items() if r["start"] >= burn}
    for _, _, _, _, by in found:
        codes &= set(by)

    out, layered = [], pb.active_layers(ctx)
    for name, mode, own, fits, by in found:
        memo = {}                                       # one match's book and its prices by line

        def price(match, market, line, p, by=by, mode=mode, own=own, fits=fits, memo=memo):
            if memo.get("match") != match:
                memo.clear()
                memo.update(match=match, book=pb.score_book(by[match], mode, own, fits, draws))
            key = (market, line)
            if key not in memo:
                r = by[match]
                memo[key] = (arm_chance(memo["book"], market, line),) + tuple(
                    arm_chance(memo["book"], market, line, r[mk], r[tk]) for _, mk, tk in layered)
            return memo[key]
        out.append(([name] + [name + tag for tag, _, _ in layered], price))
    played = {c: played_label(r) for c, r in prod_rows.items()}
    return out, codes, played, layers


def played_label(r):
    """What the match's gamers had played that session before it: a rematch, a first meeting after
    either had played someone else, or a first meeting with neither having played anyone else."""
    if r.get("n_earlier"):
        return "rematch"
    return "first, after others" if r.get("n_others") else "first, no others"


def previous_kickoffs(history):
    """gamer -> sorted kick-offs of their settled matches."""
    return {g: starts for g, (starts, _) in bets.gamer_games(history).items()}


def results_in(meet, bet_time, kickoffs):
    """Whether both gamers' previous results were in when the bet was placed (bets.RESULT_IN after
    each previous match's kick-off)."""
    if bet_time is None:
        return False
    for g in (meet["home"], meet["away"]):
        starts = kickoffs.get(g) or []
        i = bisect_left(starts, meet["start"]) - 1
        if i >= 0 and starts[i] + bets.RESULT_IN > bet_time:
            return False
    return True


def minute_band(minutes):
    if minutes is None:
        return "unknown"
    for top, text in MINUTE_BANDS:
        if top is None or minutes < top:
            return text
    return MINUTE_BANDS[-1][1]


class Table:
    """The re-priced bets as arrays: stake, prod's revenue and each variant's, by bet; with the match
    each bet is on, for resampling."""

    def __init__(self, base, names, revenue):
        self.base, self.names = base, names
        self.stake = np.array([r["stake"] for r in base], dtype=float)
        self.rev = np.array([r["revenue"] for r in base], dtype=float)
        self.cand = np.array(revenue, dtype=float).reshape(len(names), len(base))
        codes = sorted({r["match_code"] for r in base})
        at = {c: i for i, c in enumerate(codes)}
        self.match = np.array([at[r["match_code"]] for r in base], dtype=int)
        self.n_matches = len(codes)

    def margins(self, idx):
        """(stake, prod's margin %, [each variant's change, points]) over these bets."""
        stake = float(self.stake[idx].sum())
        if not stake:
            return stake, None, [None] * len(self.names)
        prod = 100 * float(self.rev[idx].sum()) / stake
        return stake, prod, [100 * float(c[idx].sum()) / stake - prod for c in self.cand]

    def interval(self, idx, k, n_boot, rng):
        """The 95% interval of variant k's change over these bets, resampling their matches."""
        idx = np.asarray(idx, dtype=int)
        m = self.match[idx]
        ms, inv = np.unique(m, return_inverse=True)
        stake = np.bincount(inv, self.stake[idx], len(ms))
        diff = np.bincount(inv, self.cand[k][idx] - self.rev[idx], len(ms))
        draws = rng.integers(0, len(ms), size=(n_boot, len(ms)))
        s, d = stake[draws].sum(axis=1), diff[draws].sum(axis=1)
        ok = s > 0
        if not ok.any():
            return None, None
        x = np.sort(100 * d[ok] / s[ok])
        return float(x[int(0.025 * len(x))]), float(x[min(len(x) - 1, int(0.975 * len(x)))])

    def revenue_change(self, idx, k):
        return float((self.cand[k][idx] - self.rev[idx]).sum())


def headline(title, table, idx, n_boot, seed=0):
    """Lines: prod's margin over these bets and each variant's change with its interval, revenue
    change and the weeks it came out ahead."""
    idx = list(idx)
    stake, prod, changes = table.margins(idx)
    n_m = len({table.base[i]["match_code"] for i in idx})
    L = [f"\n  {title}: {len(idx):,} bets on {n_m:,} matches, stake {stake:,.0f}"
         + (f", prod's margin {prod:.2f}%" if prod is not None else "")]
    if prod is None:
        return L
    weeks = defaultdict(list)
    for i in idx:
        weeks[bets._iso_week(table.base[i]["bet_time"])].append(i)
    rng = np.random.default_rng(seed)
    L.append(f"  {'':22s} {'change':>7s} {'95% interval':>17s} {'revenue':>12s} {'weeks ahead':>12s}")
    for k, name in enumerate(table.names):
        lo, hi = table.interval(idx, k, n_boot, rng)
        ahead = sum(1 for w in weeks.values() if table.revenue_change(w, k) > 0)
        band = "-" if lo is None else f"[{lo:+.2f}, {hi:+.2f}]"
        L.append(f"  {name[:22]:22s} {changes[k]:+7.2f} {band:>17s} {table.revenue_change(idx, k):+12,.0f} "
                 f"{f'{ahead} of {len(weeks)}':>12s}")
    return L


def cut(title, table, idx, key):
    """Lines: prod's margin and each variant's change (points) by a key of the bet."""
    groups = defaultdict(list)
    for i in idx:
        groups[key(table.base[i])].append(i)
    L = [f"\n  {title}",
         f"  {'':20s} {'bets':>7s} {'stake':>12s} {'prod':>7s} " + " ".join(f"{n[:11]:>11s}" for n in table.names)]
    for g in sorted(groups, key=str):
        stake, prod, changes = table.margins(groups[g])
        if prod is None:
            continue
        L.append(f"  {str(g)[:20]:20s} {len(groups[g]):7,d} {stake:12,.0f} {prod:6.2f}% "
                 + " ".join(f"{c:+11.2f}" for c in changes))
    return L


def report(table, idx_all, clean, info, n_boot, played=None):
    """Lines of text: the clean bets, then every pre-match bet and when the money went down. `played`:
    {match: what its gamers had played that session}."""
    played = played or {}
    idx_clean = [i for i in idx_all if clean[i]]
    vip = config.BET_VIP_VALUE.lower()
    L = headline("Clean bets (both gamers' earlier results in when placed)", table, idx_clean, n_boot)
    for g in (markets.MONEYLINE, markets.SPREAD, markets.TOTAL):
        sub = [i for i in idx_clean if table.base[i]["market"] == g]
        if sub:
            L += headline(f"Clean bets, {g}", table, sub, n_boot)
    L.append("\n  (change: each variant's margin less prod's, points of stake; negative is worse for the book)")
    L += cut("Clean bets by meeting", table, idx_clean, lambda r: info[r["match_code"]]["meeting"])
    L += cut("Clean bets by what the gamers had played that session (the inference layer moves all but the "
             "last)", table, idx_clean, lambda r: played.get(r["match_code"], "unknown"))
    L += cut("Clean bets by market and what the gamers had played", table, idx_clean,
             lambda r: f"{r['market']}, {played.get(r['match_code'], 'unknown')}")
    L += cut("Clean bets by market and meeting", table, idx_clean,
             lambda r: f"{r['market']} {info[r['match_code']]['meeting']}")
    L += cut("Clean bets by operator", table, idx_clean, lambda r: r.get(config.BET_GROUP_COLUMN))
    L += cut(f"Clean bets, {config.BET_VIP_VALUE} and the rest", table, idx_clean,
             lambda r: config.BET_VIP_VALUE if str(r.get(config.BET_VIP_COLUMN, "")).strip().lower() == vip
             else "the rest")
    L += cut("Clean bets week by week (placed)", table, idx_clean, lambda r: bets._iso_week(r["bet_time"]))
    L += headline("Every pre-match bet (the rest re-priced with results the bettor didn't have)", table,
                  idx_all, n_boot)
    L += cut("Every pre-match bet by when it was placed (minutes before kick-off) and whether clean", table,
             idx_all, lambda r: f"{r['_band']}, {'clean' if r['_clean'] else 'not'}")
    return L


def run(cur, out_dir, arms, since, until=None, draws=score_model.DRAWS, n_boot=None, features=None):
    """Replay the window's pre-match bets at every arm's prices, as it is and with each layer
    (features: player-stats' per-match stats, for the feature layer). Returns the report's path."""
    n_boot = config.SIM_BOOT if n_boot is None else n_boot
    ctx = pb.prepare(cur, since, until, features)
    window, tl = ctx["window"], ctx["tl"]
    found, codes, played, layers = variants(ctx, arms, draws)
    sql, params, _ = bets.bets_sql()
    cols, raw = fetch_all(cur, sql, tuple(params))
    every = bets.to_bets(cols, raw)
    flagged = [b for b in every if not b.in_play and b.feed_market is not None and b.match_code in codes]
    pre = [b for b in flagged if b.time is not None and b.time < window[b.match_code]["start"]]
    late = len(flagged) - len(pre)
    print(f"  {len(pre):,} pre-match bets on {len({b.match_code for b in pre}):,} of the {len(codes):,} "
          f"matches every arm priced after the burn-in ({len(every):,} bets in the window)", flush=True)
    if not pre:
        raise SystemExit("no pre-match bets on those matches: check --since / --until")
    finals = {c: (window[c].get("home_score"), window[c].get("away_score")) for c in codes}
    signs = bets.fit_line_signs(pre, tl, {})
    kickoffs = previous_kickoffs(ctx["history"])
    base = bets.join(pre, {}, signs, tl, {}, None, finals)          # each bet against prod's quote at the bet
    results = []
    for names, price in found:
        print(f"  pricing the bets with {' and '.join(names)}", flush=True)
        results += list(zip(names, price_bets(base, price, len(names))))
    for b, r in zip(pre, base):
        m = window[b.match_code]
        minutes = None if b.time is None else (m["start"] - b.time).total_seconds() / 60
        r.update(kickoff=m["start"], minutes_before=None if minutes is None else round(minutes, 1),
                 _band=minute_band(minutes), _clean=results_in(m, b.time, kickoffs))
    idx = [i for i in range(len(base)) if all(rows[i]["simulated"] for _, rows in results)]
    names = [n for n, _ in results]
    table = Table([base[i] for i in idx], names,
                  [rows[i]["candidate_revenue"] for _, rows in results for i in idx])
    clean = [base[i]["_clean"] for i in idx]
    why = Counter(bets.why_not(dict(r, **results[0][1][i])) for i, r in enumerate(base))
    held = sum(1 for _, rows in results for r in rows if r["candidate_prob"] in (PROB_FLOOR, 1 - PROB_FLOOR))
    L = [f"  {len(pre):,} pre-match bets, stake {sum(b.stake for b in pre):,.0f}; re-priced by every variant: "
         f"{len(idx):,} bets, stake {table.stake.sum():,.0f}",
         "  not re-priced: " + (", ".join(f"{k} {v:,}" for k, v in why.most_common() if k != "simulated")
                                 or "none"),
         f"  left out: {late:,} bets flagged pre-match but placed at or after kick-off",
         f"  a variant's chance held within [{PROB_FLOOR}, {1 - PROB_FLOOR}]: {held:,} bet prices",
         f"  clean (both gamers' earlier results in, {int(bets.RESULT_IN.total_seconds() // 60)} min after "
         f"kick-off): {sum(clean):,} of {len(idx):,} bets, stake "
         f"{sum(table.stake[i] for i in range(len(idx)) if clean[i]):,.0f}",
         "  each bet at the variant's chance at its line, the operator's margin kept, settled as it settled; "
         "the same bets and stakes (nobody bets differently)"]
    L += report(table, list(range(len(idx))), clean, window, n_boot, played)
    L += pb.layer_lines(layers)
    print("\n".join(L))
    os.makedirs(out_dir, exist_ok=True)
    out_rows = []
    for i in idx:
        r = base[i]
        row = {k: r.get(k) for k in ("bet_id", "match_code", "bet_time", "kickoff", "minutes_before", "market",
                                     "selection", "bet_line_prod_side", "odds", "stake", "revenue", "result",
                                     "stream_prob", config.BET_GROUP_COLUMN, config.BET_VIP_COLUMN)}
        row.update(meeting=window[r["match_code"]]["meeting"], played=played.get(r["match_code"]), clean=r["_clean"])
        for name, rows in results:
            row.update({f"{name}_prob": rows[i]["candidate_prob"], f"{name}_odds": rows[i]["candidate_odds"],
                        f"{name}_revenue": rows[i]["candidate_revenue"]})
        out_rows.append(row)
    bets.write_csv(os.path.join(out_dir, "prior_replay.csv"), out_rows)
    path = os.path.join(out_dir, "prior_replay.txt")
    bets.write_text(path, L)
    return path
