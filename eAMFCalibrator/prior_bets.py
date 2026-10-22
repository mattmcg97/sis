"""The pre-match report, no simulation of our own: prod against the priors' own prices -- each walkthrough
table priced off its own score distribution (priors/VARIANT.csv: NB2's joint shared-gamma model for nb2
and the averages, NB2's construction fitted to glmer's errors for glmer) -- and against a sim's
pre-match price from a betting-sim run (its candidate columns: v21 on a glmer build, say).

    python -m eAMFCalibrator prior-bets --bets eAMFCalibrator/out_bets_v21_4k/bets_sim.csv --sim-name glmer-sim \\
        --table nb2=eAMFCalibrator/out/walkthrough/priors/nb2.csv \\
        --table glmer=eAMFCalibrator/out/walkthrough/priors/glmer.csv \\
        --table avg=eAMFCalibrator/out/walkthrough/priors/avg.csv

  the prices   a table's chance of a selection at a line: its score model rebuilt from the row
               (score_model, score_params about home_pts / away_pts), read as prior-replay reads an
               arm (pushes left out). prod's: its probability at the bet. The sim's: its pre-match
               probability in the bets (its kick-off), its own re-priced bets as the betting sim made them
  the grades   the moneyline on every match with a result (no tie), prod's pre-match price (at the
               match's last pre-match moneyline bet), the sim's and every table's: Brier and log loss,
               each less prod's with a 95% interval over days; the same by how heavily the match was
               bet (quarters of its moneyline stake) and weighted by that stake; the result on prod's
               logit plus each price's disagreement with it (1: take the price's whole view). The
               spread and the total: the selection of each match's last pre-match bet in the market,
               at prod's line then, Brier against prod's
  the bets     every pre-match bet (moneyline, spread, total) at prod's line re-priced at each price as
               the betting sim re-prices (odds x prod / price, the operator's margin kept, settled as it
               settled); the same bets and two books (each bet to the price that pays more), 95% over
               matches. Clean bets head it: placed once both gamers' previous results were in
               (RESULT_IN after that match's kick-off), as a table's price, made with every result
               before kick-off, would otherwise know a result the bettor and prod did not. Then the
               in-play moneyline bets struck while prod's price was still within FLAT of its pre-match
               one (the kick-off money; those in the game's first KICKOFF_SECONDS apart), each table's
               price moved by prod's small move since (in logit), the sim at its own in-play price

Writes (to --out): prior_bets.txt and prior_bets.csv (a row a match: the result, the stake, every
moneyline price).
"""

import csv
import datetime as dt
import math
import os
import random
from collections import defaultdict

from . import markets, score_model
from .metrics import day_of
from .money_layer import SIDE, _logit, _num, _sigmoid, _time, read_sim_bets, read_table, sim_columns
from .prior_replay import arm_chance

FLAT = 0.02                  # prod's in-play price this near its pre-match one is still the pre-match price
KICKOFF_SECONDS = 15         # the game's first seconds: no play yet
QUARTER_SECONDS = 240.0
RESULT_IN = dt.timedelta(minutes=36)
N_BOOT = 1000
P_CLIP = 0.005
PROD = "prod"
MARKET_ORDER = (markets.MONEYLINE, markets.SPREAD, markets.TOTAL)


def _clip(p):
    return min(max(p, P_CLIP), 1 - P_CLIP)


def _in_play(r):
    return str(r.get("in_play")).strip().lower() == "true"


def _feed(r):
    return int(_num(r.get("feed_market")) or 0)


def candidate_columns(cols):
    """(suffix, label suffix) of each candidate in the bets: '' for candidate_prob, '_v21' for candidate_prob_v21."""
    return [(c[len("candidate_prob"):], c[len("candidate_prob_"):] if c != "candidate_prob" else "")
            for c in cols if c == "candidate_prob" or c.startswith("candidate_prob_")]


# --------------------------------------------------------------------------------------------- the tables

class Table:
    """A walkthrough table: each match's score model, kick-off and gamers."""

    def __init__(self, path, draws=score_model.DRAWS):
        rows, cols = read_table(path)
        need = {"MATCH_CODE", "home_pts", "away_pts"}
        if not need <= set(cols):
            raise SystemExit(f"{path}: a prior table here needs {', '.join(sorted(need))} (a walkthrough's priors/VARIANT.csv)")
        self.rows = {r["MATCH_CODE"].strip(): r for r in rows if r.get("MATCH_CODE")}
        self.draws, self._books = draws, {}

    def book(self, code):
        """The match's score book (score_model), or None."""
        if code in self._books:
            return self._books[code]
        r = self.rows.get(code)
        bk = None
        if r is not None:
            h, a = _num(r.get("home_pts")), _num(r.get("away_pts"))
            kind = (r.get("score_model") or "normal").strip() or "normal"
            params = [_num(x) for x in str(r.get("score_params") or "").split()]
            if h is not None and a is not None:
                if kind == "normal" or any(x is None for x in params):
                    bk = score_model.Normal(h - a, h + a)
                else:
                    bk = score_model.book(kind, (h, a) + tuple(params), code, self.draws)
        self._books[code] = bk
        return bk

    def p_home(self, code):
        r = self.rows.get(code)
        p = _num(r.get("p_home")) if r else None
        if p is not None:
            return p
        bk = self.book(code)
        return bk.p_home() if bk else None

    def chance(self, code, feed, line):
        bk = self.book(code)
        return None if bk is None else arm_chance(bk, feed, line)


def results_in(tables):
    """match -> when both gamers' previous results were in, off the tables' kick-offs and gamers."""
    games = defaultdict(list)
    for t in tables.values():
        for code, r in t.rows.items():
            start = _time(r.get("start"))
            if start is None:
                continue
            for g in (r.get("home"), r.get("away")):
                if g:
                    games[str(g).strip().upper()].append((start, code))
    prev = {}
    for g, ms in games.items():
        ms = sorted(set(ms))
        for (s0, _), (s1, c1) in zip(ms, ms[1:]):
            if s1 > s0:
                prev[c1] = max(prev.get(c1, s0 + RESULT_IN), s0 + RESULT_IN)
    return prev


# --------------------------------------------------------------------------------------------- the grades

def prematch_prices(rows, col):
    """match -> the home probability `col` gave at the match's last pre-match moneyline bet."""
    last = {}
    for r in rows:
        if r.get("market") != markets.MONEYLINE or _in_play(r) or SIDE.get(r.get("selection")) is None:
            continue
        p = _num(r.get(col))
        if p is None:
            continue
        key, code = str(r.get("bet_time") or ""), str(r.get("match_code")).strip()
        if code not in last or key >= last[code][0]:
            last[code] = (key, p if SIDE[r["selection"]] == 0 else 1 - p)
    return {c: v[1] for c, v in last.items()}


def results(rows):
    """match -> (home won 1/0, the day, its moneyline stake), ties and matches without a final left out."""
    out, stake = {}, defaultdict(float)
    for r in rows:
        if r.get("market") == markets.MONEYLINE:
            stake[str(r.get("match_code")).strip()] += _num(r.get("stake")) or 0.0
    for r in rows:
        code = str(r.get("match_code")).strip()
        a, b = _num(r.get("final_p1")), _num(r.get("final_p2"))
        if code in out or a is None or b is None or a == b:
            continue
        out[code] = (1.0 if a > b else 0.0, code[-6:], stake[code])
    return out


def _scores(p, y):
    p = _clip(p)
    return (p - y) ** 2, -(y * math.log(p) + (1 - y) * math.log(1 - p))


def _interval(groups, n_boot=N_BOOT, seed=0):
    """The mean over the members of `groups` ({group: [values]}) and its 95% interval resampling groups."""
    vals = [v for v in groups.values() if v]
    n = sum(len(v) for v in vals)
    if not n:
        return None, None, None
    sums = [(sum(v), len(v)) for v in vals]
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        s = c = 0.0
        for _ in sums:
            a, b = sums[rng.randrange(len(sums))]
            s += a
            c += b
        draws.append(s / c)
    draws.sort()
    return sum(a for a, _ in sums) / n, draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot) - 1]


def disagreement(codes, res, prod, price):
    """(weight on prod's logit, its se, weight on the price's disagreement with prod, its se): the result's
    logistic regression on the two, by Newton's method; None if it will not fit."""
    xs = [(_logit(_clip(prod[c])), _logit(_clip(price[c])) - _logit(_clip(prod[c])), res[c][0]) for c in codes]
    b = [1.0, 0.0]
    det = h = None
    for _ in range(30):
        g = [0.0, 0.0]
        h = [[0.0, 0.0], [0.0, 0.0]]
        for x0, x1, y in xs:
            p = _sigmoid(b[0] * x0 + b[1] * x1)
            w = p * (1 - p)
            g[0] += (y - p) * x0
            g[1] += (y - p) * x1
            h[0][0] += w * x0 * x0
            h[0][1] += w * x0 * x1
            h[1][1] += w * x1 * x1
        h[1][0] = h[0][1]
        det = h[0][0] * h[1][1] - h[0][1] * h[1][0]
        if det <= 0:
            return None
        step = ((h[1][1] * g[0] - h[0][1] * g[1]) / det, (h[0][0] * g[1] - h[1][0] * g[0]) / det)
        b = [b[0] + step[0], b[1] + step[1]]
        if max(abs(step[0]), abs(step[1])) < 1e-8:
            break
    return b[0], math.sqrt(h[1][1] / det), b[1], math.sqrt(h[0][0] / det)


def grade_lines(codes, res, prices):
    """Lines: every moneyline price's Brier and log loss on the matches, each less prod's, by how heavily bet."""
    L = [f"\n  the moneyline on {len(codes):,} matches with a result, prod's pre-match price and every other's"
         f" (less prod: below zero the price closer, 95% over days)",
         "  price              Brier   less prod                      log loss  less prod"]
    for name, pr in prices.items():
        b, lo, db, dl = (defaultdict(list) for _ in range(4))
        for c in codes:
            y, day, _ = res[c]
            sb, sl = _scores(pr[c], y)
            pb, pl = _scores(prices[PROD][c], y)
            b[day].append(sb)
            lo[day].append(sl)
            db[day].append(sb - pb)
            dl[day].append(sl - pl)
        cb, cl = _interval(db), _interval(dl, seed=1)
        if name == PROD:
            L.append(f"  {name:16s} {_interval(b)[0]:.4f}                                  {_interval(lo)[0]:.4f}")
        else:
            L.append(f"  {name:16s} {_interval(b)[0]:.4f}  {cb[0]:+.4f} [{cb[1]:+.4f}, {cb[2]:+.4f}]   {_interval(lo)[0]:.4f}"
                     f"   {cl[0]:+.4f} [{cl[1]:+.4f}, {cl[2]:+.4f}]")
    ranked = sorted(codes, key=lambda c: res[c][2])
    parts = [ranked[k * len(ranked) // 4:(k + 1) * len(ranked) // 4] for k in range(4)]
    tot = sum(res[c][2] for c in codes) or 1.0
    L.append("\n  by how heavily the match was bet (quarters of its moneyline stake): prod's Brier, each other's less prod's")
    L.append("  " + "".ljust(16) + "".join(f"{lab:>12s}" for lab in ("least bet", "2nd", "3rd", "most bet")) + "   by stake")
    L.append("  " + "(stake)".ljust(16) + "".join(f"{sum(res[c][2] for c in part):>12,.0f}" for part in parts))
    for name, pr in prices.items():
        def d(c):
            own = _scores(pr[c], res[c][0])[0]
            return own if name == PROD else own - _scores(prices[PROD][c], res[c][0])[0]
        fmt = "{:>12.4f}" if name == PROD else "{:>+12.4f}"
        L.append(f"  {name:16s}" + "".join(fmt.format(sum(d(c) for c in part) / len(part)) for part in parts)
                 + ("   {:.4f}" if name == PROD else "   {:+.4f}").format(sum(res[c][2] * d(c) for c in codes) / tot))
    L.append("\n  the result on prod's logit and each price's disagreement with it (1: the price's whole view, 0: none):")
    for name, pr in prices.items():
        if name != PROD:
            fit = disagreement(codes, res, prices[PROD], pr)
            if fit:
                L.append(f"  {name:16s} prod's logit {fit[0]:+.2f} (se {fit[1]:.2f}), the disagreement {fit[2]:+.2f} (se {fit[3]:.2f})")
    return L


def line_grades(rows, tables, codes):
    """Lines: the spread and the total -- each match's last pre-match bet in the market, its selection at
    prod's line then -- prod's Brier and each table's less it."""
    L = []
    for market, feeds in ((markets.SPREAD, (52, 53)), (markets.TOTAL, (54, 55))):
        last = {}
        for r in rows:
            code = str(r.get("match_code")).strip()
            if (_feed(r) not in feeds or _in_play(r) or code not in codes or r.get("result") not in ("won", "lost")
                    or _num(r.get("stream_prob")) is None or _num(r.get("stream_line")) is None):
                continue
            key = str(r.get("bet_time") or "")
            if code not in last or key >= last[code][0]:
                last[code] = (key, r)
        if not last:
            continue
        L.append(f"\n  the {market} at prod's pre-match line ({len(last):,} matches; each match's last pre-match bet's selection):")
        prod = defaultdict(list)
        for code, (_, r) in last.items():
            prod[code[-6:]].append(_scores(_num(r["stream_prob"]), 1.0 if r["result"] == "won" else 0.0)[0])
        L.append(f"  {PROD:16s} Brier {_interval(prod)[0]:.4f}")
        for name, t in tables.items():
            d = defaultdict(list)
            for code, (_, r) in last.items():
                c = t.chance(code, _feed(r), _num(r["stream_line"]))
                if c is None:
                    continue
                y = 1.0 if r["result"] == "won" else 0.0
                d[code[-6:]].append(_scores(_clip(c), y)[0] - _scores(_num(r["stream_prob"]), y)[0])
            m = _interval(d)
            if m[0] is not None:
                L.append(f"  {name:16s} less prod {m[0]:+.4f} [{m[1]:+.4f}, {m[2]:+.4f}]")
    return L


# --------------------------------------------------------------------------------------------- the bets

def _game_seconds(r):
    q = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}.get(r.get("quarter"))
    left = _num(r.get("clock_left"))
    return None if q is None or left is None else (q - 1) * QUARTER_SECONDS + (QUARTER_SECONDS - left)


def _revenue(stake, odds, result):
    return stake - (stake * odds if result == "won" else stake if result == "push" else 0.0)


def bet_book(rows, codes, tables, sims, prod_pre, prev_in):
    """Every pre-match bet and every kick-off moneyline bet on these matches, each with prod's revenue and
    each price's odds and revenue: [dict(group, market, match, stake, rev, prices={name: (odds, revenue)})]."""
    out = []
    for r in rows:
        code = str(r.get("match_code")).strip()
        market, feed = r.get("market"), _feed(r)
        p, odds, stake, rev = _num(r.get("stream_prob")), _num(r.get("odds")), _num(r.get("stake")), _num(r.get("revenue"))
        if (code not in codes or market not in MARKET_ORDER or r.get("result") not in ("won", "lost", "push")
                or not p or not odds or not stake or rev is None or str(r.get("on_prod_line")) not in ("True", "true", "1")):
            continue
        p = _clip(p)
        if not _in_play(r):
            t = _time(r.get("bet_time"))
            clean = code not in prev_in or (t is not None and t >= prev_in[code])
            group = "pre-match, clean" if clean else "pre-match, a previous result not yet in"
            moved = 0.0
        elif market == markets.MONEYLINE and abs(p - (prod_pre[code] if SIDE[r["selection"]] == 0 else 1 - prod_pre[code])) <= FLAT:
            gs = _game_seconds(r)
            group = "kick-off" + (f", first {KICKOFF_SECONDS}s" if gs is not None and gs <= KICKOFF_SECONDS else ", later")
            pre = _clip(prod_pre[code] if SIDE[r["selection"]] == 0 else 1 - prod_pre[code])
            moved = _logit(p) - _logit(pre)
        else:
            continue
        got = {}
        for name, t in tables.items():
            c = t.chance(code, feed, _num(r.get("stream_line")))
            if c is None:
                continue
            c = _clip(_sigmoid(_logit(_clip(c)) + moved))
            oc = odds * p / c
            got[name] = (oc, _revenue(stake, oc, r["result"]))
        for name, suffix in sims:
            oc, rc = _num(r.get("candidate_odds" + suffix)), _num(r.get("candidate_revenue" + suffix))
            if oc is not None and rc is not None and str(r.get("simulated" + suffix)) in ("True", "true", "1"):
                got[name] = (oc, rc)
        out.append(dict(group=group, market=market, match=code, stake=stake, rev=rev, odds=odds, prices=got))
    return out


def _stats(sums):
    """[prod's margin, the price's, the change, two books: takes, keeps, takes less keeps] off per-match sums
    [stake, prod's revenue, the price's, stake it takes, its revenue there, prod's revenue on the rest]."""
    t = [sum(v[i] for v in sums) for i in range(6)]
    if not t[0]:
        return None
    m = [100 * t[1] / t[0], 100 * t[2] / t[0]]
    takes = 100 * t[4] / t[3] if t[3] else None
    keeps = 100 * t[5] / (t[0] - t[3]) if t[0] > t[3] else None
    return m + [m[1] - m[0], takes, keeps, None if takes is None or keeps is None else takes - keeps]


def _pct(x, fmt="{:+7.2f}%"):
    return "      - " if x is None else fmt.format(x)


def bet_lines(book, names, n_boot=N_BOOT, seed=0):
    """Lines: the bets by group and market, each price's margin, change and two books against prod."""
    L = ["\n  the bets re-priced at each price (odds x prod / price, the operator's margin kept; two books: each bet to"
         " the price that pays more), the change and takes-less-keeps with 95% over days"]
    cuts = []
    for group in ("pre-match, clean", "pre-match, a previous result not yet in", f"kick-off, first {KICKOFF_SECONDS}s",
                  "kick-off, later"):
        g = [b for b in book if b["group"] == group]
        if not g:
            continue
        if group.startswith("pre-match"):
            cuts.append((f"{group}: every market", g))
            cuts += [(f"{group}: {m}", [b for b in g if b["market"] == m]) for m in MARKET_ORDER]
        else:
            cuts.append((f"{group} (the moneyline in play, prod still within {100 * FLAT:.0f} points of its pre-match price)", g))
    for label, sel in cuts:
        if not sel:
            continue
        stake = sum(b["stake"] for b in sel)
        L.append(f"\n  {label}: {len(sel):,} bets, {stake:,.0f} staked on {len({b['match'] for b in sel}):,} matches;"
                 f" prod as placed {100 * sum(b['rev'] for b in sel) / stake:+.2f}%")
        L.append("  price              margin   change [95%]               takes   keeps   takes less keeps [95%]      share")
        for name in names:
            sums = defaultdict(lambda: [0.0] * 6)
            n = 0
            for b in sel:
                if name not in b["prices"]:
                    continue
                n += 1
                oc, rc = b["prices"][name]
                a = sums[b["match"]]
                a[0] += b["stake"]
                a[1] += b["rev"]
                a[2] += rc
                if oc > b["odds"]:
                    a[3] += b["stake"]
                    a[4] += rc
                else:
                    a[5] += b["rev"]
            if not n:
                continue
            days = defaultdict(lambda: [0.0] * 6)          # a day's matches resampled together
            for k, v in sums.items():
                days[day_of(k)] = [a + x for a, x in zip(days[day_of(k)], v)]
            keys = sorted(days)
            full = _stats([days[k] for k in keys])
            rng = random.Random(seed)
            draws = [d for d in (_stats([days[keys[rng.randrange(len(keys))]] for _ in keys]) for _ in range(n_boot)) if d]
            ch = sorted(d[2] for d in draws)
            tk = sorted(d[5] for d in draws if d[5] is not None)
            lo, hi = int(0.025 * len(ch)), int(0.975 * len(ch)) - 1
            share = sum(v[3] for v in sums.values()) / sum(v[0] for v in sums.values())
            note = "" if n == len(sel) else f"  ({n:,} of the bets)"
            two = "         -          " if full[5] is None or not tk else \
                f"{full[5]:+7.2f} [{tk[int(0.025 * len(tk))]:+.2f}, {tk[int(0.975 * len(tk)) - 1]:+.2f}]"
            L.append(f"  {name:16s} {full[1]:+7.2f}%  {full[2]:+6.2f} [{ch[lo]:+.2f}, {ch[hi]:+.2f}]   {_pct(full[3])} {_pct(full[4])}"
                     f"  {two}   {100 * share:5.0f}%{note}")
    return L


# --------------------------------------------------------------------------------------------- the run

def run(out_dir, bets_paths, tables, sim_name="sim", draws=score_model.DRAWS):
    """The whole report: `tables` [(name, path)]. Returns prior_bets.txt's path."""
    rows = read_sim_bets(bets_paths)
    cols = sim_columns(bets_paths)
    res = results(rows)
    loaded = {name: Table(path, draws) for name, path in tables}
    sims = [(sim_name + ("-" + lab if lab else ""), suffix) for suffix, lab in candidate_columns(cols)]
    prices = {PROD: prematch_prices(rows, "stream_prob")}
    for name, suffix in sims:
        prices[name] = prematch_prices(rows, "candidate_prob" + suffix)
    for name, t in loaded.items():
        prices[name] = {c: p for c in res if (p := t.p_home(c)) is not None}
    codes = sorted(c for c in res if all(c in pr for pr in prices.values()))
    if not codes:
        raise SystemExit("no match is priced by every source: do the tables cover the bets' window?")
    missing = {name: sum(1 for c in res if c not in pr) for name, pr in prices.items()}
    L = [f"\nThe pre-match report: prod against the priors' own prices and a sim's, no simulation here"
         f" ({', '.join(os.path.basename(p) for p in bets_paths)})",
         "  " + "; ".join(f"{name}: {path}" for name, path in tables) + (f"; {', '.join(n for n, _ in sims)}: the bets' candidate" if sims else ""),
         f"  {len(res):,} settled matches without a tie in the bets; {len(codes):,} priced by all"
         + (" (" + ", ".join(f"{n} lacks {m:,}" for n, m in missing.items() if m) + ")" if any(missing.values()) else "")]
    L += grade_lines(codes, res, prices)
    L += line_grades(rows, loaded, set(codes))
    book = bet_book(rows, set(codes), loaded, sims, prices[PROD], results_in(loaded))
    L += bet_lines(book, [n for n, _ in sims] + list(loaded))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "prior_bets.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["match_code", "home_won", "moneyline_stake"] + [f"p_home_{n}" for n in prices])
        for c in codes:
            w.writerow([c, int(res[c][0]), round(res[c][2], 2)] + [round(prices[n][c], 5) for n in prices])
    path = os.path.join(out_dir, "prior_bets.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n  -> {path} and prior_bets.csv")
    return path
