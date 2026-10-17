"""The moneyline by game situation: what prod and a candidate price the pre-match favourite at in a
situation, against how often the favourite goes on to win -- and what the bets struck in it did.

    python -m eAMFCalibrator ml-states --pairs eAMFCalibrator/out_cal_v21_glmer/directional_pairs.csv \\
        --bets eAMFCalibrator/out_bets_v21_4k/bets_sim.csv

  the matches  the report's pairs (directional_pairs.csv: prod's and the candidate's moneyline at each of
               its snapshots, with the score, the quarter and who has the ball), the home moneyline (market
               50), each match from its first snapshot at 0-0. The pre-match favourite: the side prod's price
               was on there (the opening kick-off), those within EVENS of 50% left out
  situations   the first score of the game: who (the favourite or the underdog), a touchdown or a field
               goal, on the opening drive (before the ball first changed hands), later in the first quarter or
               after, read at the first snapshot after it with the other side on the ball (the try and the
               kick-off in); and the favourite's lead at the start of each later quarter
  the prices   each book's price on the favourite there against how often the favourite won, each match
               once a situation, 95% intervals over matches; the opening-drive touchdowns by how strong a
               favourite (prod's kick-off price)
  the bets     (--bets, a betting sim's bets_sim.csv) each settled in-play moneyline bet placed in the
               situation (from the snapshot it is read at to the next score): the stake, the book's margin
               and what prod's price says it should be (expected), the backed side's result less prod's
               price and less the candidate's (stake-weighted), and the change in the book's margin with
               the candidate re-priced (the betting sim's own re-price, on the bets it re-priced); then
               every in-play moneyline bet by the quarter, the favourite's lead and the side backed

Writes (to --out): ml_states.txt and ml_states.csv (a row a match: the favourite, the kick-off prices,
the first score, the prices after it, the result).
"""

import csv
import os
import random
from bisect import bisect_right
from collections import defaultdict

from . import markets
from .money_layer import SIDE, _num, _time, read_sim_bets, sim_columns

EVENS = 0.01               # a kick-off price this near 50% has no favourite
MIN_MATCHES = 15
N_BOOT = 1000
LEAD_BANDS = ((-99, -9, "2+ scores behind"), (-8, -3, "1 score behind"), (-2, 2, "level (within 2)"),
              (3, 8, "1 score ahead"), (9, 99, "2+ scores ahead"))
WHEN = ("opening drive", "later in Q1", "Q2 or later")
FIELDS = ("match_code", "fav", "prod_kickoff", "cand_kickoff", "first_score", "prod_after", "cand_after", "fav_won")


def _flag(v):
    return str(v).strip().lower() in ("1", "true", "1.0", "yes")


def lead_band(lead):
    """The favourite's lead in points as one of LEAD_BANDS."""
    for lo, hi, name in LEAD_BANDS:
        if lo <= lead <= hi:
            return name
    return None


def read_pairs(paths):
    """match -> [snapshots in time order: dict(t, drive, period, p1, p2, offense, prod, cand)], home's
    moneyline (market 50); and match -> home won (1/0, ties left out)."""
    by, won = defaultdict(list), {}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if str(r.get("market_id")).strip() != "50" or not _flag(r.get("prod_live")):
                    continue
                prod, t = _num(r.get("prod_probability")), _time(r.get("publish_time"))
                p1, p2 = _num(r.get("score_p1")), _num(r.get("score_p2"))
                if prod is None or t is None or p1 is None or p2 is None:
                    continue
                code = r["match_code"].strip()
                cand = _num(r.get("candidate_probability")) if _flag(r.get("candidate_live")) else None
                by[code].append(dict(t=t, drive=_num(r.get("drive_number")), period=int(_num(r.get("period_number")) or 0),
                                     p1=p1, p2=p2, offense=str(r.get("offensive_team") or ""), prod=prod, cand=cand))
                y = _num(r.get("prod_outcome"))
                if y in (0.0, 1.0):
                    won[code] = y
    for rows in by.values():
        rows.sort(key=lambda x: (x["t"], x["drive"] or 0))
    return by, won


def match_states(rows, home_won):
    """One match's situations, or None (not from its opening drive, no favourite, no result): dict(fav, the
    kick-off prices on the favourite, fav_won, the first score (situation, the row it is read at, the prices
    on the favourite there, the row of the next score), [(quarter, lead band, prod, cand)] at each later
    quarter's first snapshot, and the rows)."""
    if not rows or home_won is None or rows[0]["period"] != 1 or rows[0]["p1"] + rows[0]["p2"] > 0:
        return None
    kick = rows[0]
    if abs(kick["prod"] - 0.5) <= EVENS:
        return None
    fav = 1 if kick["prod"] > 0.5 else 2

    def on_fav(p):
        return None if p is None else p if fav == 1 else 1 - p

    def lead(r):
        return (r["p1"] - r["p2"]) * (1 if fav == 1 else -1)
    out = dict(fav=fav, rows=rows, prod_kickoff=on_fav(kick["prod"]), cand_kickoff=on_fav(kick["cand"]),
               fav_won=home_won if fav == 1 else 1 - home_won, first=None, quarters=[])
    first = next((i for i, r in enumerate(rows) if r["p1"] + r["p2"] > 0), None)
    if first is not None:
        r = rows[first]
        scorer = 1 if r["p1"] > r["p2"] else 2 if r["p2"] > r["p1"] else None
        side = "home" if scorer == 1 else "away"
        # read once the other side has the ball: the try and the kick-off are in
        back = next((j for j in range(first, len(rows)) if not rows[j]["offense"].lower().startswith(side)), None)
        if scorer is not None and back is not None:
            at = rows[back]
            pts = abs(at["p1"] - at["p2"])
            kind = "TD" if pts >= 6 else "FG" if pts == 3 else None
            opening = rows[0]["offense"]
            opened = all(x["offense"] == opening for x in rows[:first + 1]) and opening.lower().startswith(side)
            when = WHEN[0] if opened else WHEN[1] if r["period"] == 1 else WHEN[2]
            nxt = next((j for j in range(back + 1, len(rows)) if rows[j]["p1"] + rows[j]["p2"] > at["p1"] + at["p2"]), len(rows))
            if kind:
                who = "the favourite" if scorer == fav else "the underdog"
                out["first"] = dict(situation=f"{who} {kind}, {when}", who=who, kind=kind, when=when, row=back,
                                    next=nxt, prod=on_fav(at["prod"]), cand=on_fav(at["cand"]))
    seen = set()
    for r in rows:
        if r["period"] in (2, 3, 4) and r["period"] not in seen:
            seen.add(r["period"])
            out["quarters"].append((r["period"], lead_band(lead(r)), on_fav(r["prod"]), on_fav(r["cand"])))
    return out


def _miss(pairs, n_boot=N_BOOT, seed=0):
    """(mean of won - price, its 95% interval over matches) of [(won, price)]."""
    d = [w - p for w, p in pairs if p is not None]
    if not d:
        return None, None, None
    rng = random.Random(seed)
    draws = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return sum(d) / len(d), draws[int(0.025 * n_boot)], draws[int(0.975 * n_boot) - 1]


def _cell(m):
    return "        -          " if m[0] is None else f"{m[0]:+.3f} [{m[1]:+.3f}, {m[2]:+.3f}]"


def situation_lines(states, label):
    """Lines: each first-score situation and each later quarter's start, the books' prices on the favourite
    against its win rate."""
    L = ["\n  the first score of the game, once the other side has the ball after it: each book's price on the pre-match favourite,"
         " and how often the favourite won (won less price, 95% over matches)",
         f"  {'first score':34s} {'matches':>7s}  {'kick-off':>8s}  {'prod':>6s} {label[:6]:>6s}  {'won':>5s}"
         f"   {'won less prod':22s} won less {label}"]
    groups = defaultdict(list)
    for s in states:
        if s["first"]:
            groups[(s["first"]["who"], s["first"]["kind"], s["first"]["when"])].append(s)
    for key in sorted(groups, key=lambda k: (k[1], k[0], WHEN.index(k[2]))):
        g = groups[key]
        if len(g) < MIN_MATCHES:
            continue
        L.append(_line(f"{key[0]} {key[1]}, {key[2]}", g, lambda s: s["first"]["prod"], lambda s: s["first"]["cand"]))
    dog = groups.get(("the underdog", "TD", WHEN[0]), [])
    if dog:
        L.append("\n  the underdog's touchdown on the opening drive, by how strong a favourite (prod's kick-off price):")
        for lo, hi in ((0.5, 0.55), (0.55, 0.6), (0.6, 1.0)):
            g = [s for s in dog if lo < s["prod_kickoff"] <= hi]
            if g:
                L.append(_line(f"  favourite {lo:.2f}-{hi:.2f}", g, lambda s: s["first"]["prod"], lambda s: s["first"]["cand"]))
    L.append("\n  the start of each later quarter, by the favourite's lead then:")
    q = defaultdict(list)
    for s in states:
        for period, band, p, c in s["quarters"]:
            q[(period, band)].append((s, p, c))
    for period in (2, 3, 4):
        for _, _, band in LEAD_BANDS:
            g = q.get((period, band), [])
            if len(g) >= MIN_MATCHES:
                L.append(_line(f"Q{period}, the favourite {band}", [x[0] for x in g], None, None,
                               prices=[(x[1], x[2]) for x in g]))
    return L


def _line(name, g, prod_of, cand_of, prices=None):
    prices = prices or [(prod_of(s), cand_of(s)) for s in g]
    won = [s["fav_won"] for s in g]
    pp = [p for p, _ in prices]
    cc = [c for _, c in prices if c is not None]
    mp = _miss(list(zip(won, pp)))
    mc = _miss([(w, c) for w, (_, c) in zip(won, prices) if c is not None], seed=1)
    kick = sum(s["prod_kickoff"] for s in g) / len(g)
    return (f"  {name:34s} {len(g):>7,}  {kick:>8.3f}  {sum(pp) / len(pp):>6.3f} "
            f"{(sum(cc) / len(cc)) if cc else float('nan'):>6.3f}  {sum(won) / len(won):>5.3f}   {_cell(mp)}   {_cell(mc)}")


# --------------------------------------------------------------------------------------------- the bets

def read_bets(paths):
    """The settled in-play moneyline bets: dict(match, t, side, stake, odds, revenue, won, prod, cand, cand_rev)."""
    rows, cols = read_sim_bets(paths), sim_columns(paths)
    suffix = next((c[len("candidate_prob"):] for c in cols if c == "candidate_prob" or c.startswith("candidate_prob_")), None)
    out = []
    for r in rows:
        if (r.get("market") != markets.MONEYLINE or str(r.get("in_play")).strip().lower() != "true"
                or r.get("result") not in ("won", "lost") or SIDE.get(r.get("selection")) is None):
            continue
        p, odds, stake, rev, t = (_num(r.get("stream_prob")), _num(r.get("odds")), _num(r.get("stake")),
                                  _num(r.get("revenue")), _time(r.get("bet_time")))
        if not p or not odds or not stake or rev is None or t is None:
            continue
        cand = cand_rev = None
        if suffix is not None:
            cand = _num(r.get("candidate_prob" + suffix))
            if _flag(r.get("simulated" + suffix)):
                cand_rev = _num(r.get("candidate_revenue" + suffix))
        out.append(dict(match=str(r.get("match_code")).strip(), t=t, side=SIDE[r["selection"]] + 1, stake=stake, odds=odds,
                        revenue=rev, won=1.0 if r["result"] == "won" else 0.0, prod=p, cand=cand, cand_rev=cand_rev))
    return out


def place_bets(bets, by_match):
    """Each bet with its match's situation: (bet, state, the index of the latest snapshot before it)."""
    out = []
    times = {code: [r["t"] for r in s["rows"]] for code, s in by_match.items()}
    for b in bets:
        s = by_match.get(b["match"])
        if s is None:
            continue
        i = bisect_right(times[b["match"]], b["t"]) - 1
        if i >= 0:
            out.append((b, s, i))
    return out


def bet_stats(placed):
    """dict: bets, stake, the book's margin and prod's expected, won less prod and less the candidate
    (stake-weighted), on the favourite's share, and the candidate's re-priced margin less prod's on the bets it
    re-priced."""
    if not placed:
        return None
    st = sum(b["stake"] for b, _, _ in placed)
    rev = sum(b["revenue"] for b, _, _ in placed)
    exp = sum(b["stake"] * (1 - b["prod"] * b["odds"]) for b, _, _ in placed)
    wp = sum(b["stake"] * (b["won"] - b["prod"]) for b, _, _ in placed) / st
    with_c = [b for b, _, _ in placed if b["cand"] is not None]
    wc = sum(b["stake"] * (b["won"] - b["cand"]) for b in with_c) / sum(b["stake"] for b in with_c) if with_c else None
    sim = [b for b, _, _ in placed if b["cand_rev"] is not None]
    ss = sum(b["stake"] for b in sim)
    change = 100 * (sum(b["cand_rev"] for b in sim) - sum(b["revenue"] for b in sim)) / ss if ss else None
    fav = sum(b["stake"] for b, s, _ in placed if b["side"] == s["fav"]) / st
    return dict(bets=len(placed), stake=st, margin=100 * rev / st, expected=100 * exp / st, wp=wp, wc=wc,
                change=change, fav=fav, matches=len({b["match"] for b, _, _ in placed}))


def _bet_line(name, st):
    wc = "    -  " if st["wc"] is None else f"{st['wc']:+.3f}"
    ch = "    -  " if st["change"] is None else f"{st['change']:+6.2f}"
    return (f"  {name:44s} {st['bets']:>6,} {st['stake']:>10,.0f} {st['matches']:>5,}  {100 * st['fav']:>4.0f}%"
            f"  {st['margin']:+7.2f}% {st['expected']:+6.2f}%   {st['wp']:+.3f}   {wc}   {ch}")


BET_HEAD = (f"  {'':44s} {'bets':>6s} {'stake':>10s} {'mtchs':>5s}  {'on fav':>5s}  {'margin':>8s} {'expect':>7s}"
            f"   {'won-prod':>8s} {'won-cand':>8s} {'cand chg':>8s}")


def bet_lines(placed, label):
    L = [f"\n  the in-play moneyline bets in each first-score situation (from the snapshot it is read at to the next"
         f" score): margin and prod's expected, the backed side's result less prod's price and less {label}'s"
         f" (stake-weighted), {label}'s re-priced margin less prod's (points)", BET_HEAD]
    groups = defaultdict(list)
    for b, s, i in placed:
        f = s["first"]
        if f and f["row"] <= i < f["next"]:
            groups[f["situation"]].append((b, s, i))
            groups[f"{f['situation']} | on the {'favourite' if b['side'] == s['fav'] else 'underdog'}"].append((b, s, i))
    for key in sorted(groups):
        st = bet_stats(groups[key])
        if st and st["matches"] >= 5:
            L.append(_bet_line(key, st))
    L += ["\n  every in-play moneyline bet by the quarter, the favourite's lead at the snapshot before it and the side backed",
          BET_HEAD]
    q = defaultdict(list)
    for b, s, i in placed:
        r = s["rows"][i]
        lead = (r["p1"] - r["p2"]) * (1 if s["fav"] == 1 else -1)
        band = lead_band(lead)
        side = "favourite" if b["side"] == s["fav"] else "underdog"
        q[(r["period"], band, side)].append((b, s, i))
    order = [name for _, _, name in LEAD_BANDS]
    for key in sorted(q, key=lambda k: (k[0], order.index(k[1]) if k[1] in order else 9, k[2])):
        st = bet_stats(q[key])
        if st and st["matches"] >= 5:
            L.append(_bet_line(f"Q{key[0]}, favourite {key[1]}, on the {key[2]}", st))
    return L


# --------------------------------------------------------------------------------------------- the run

def run(out_dir, pairs_paths, bets_paths=None, label="candidate"):
    """The whole report. Returns ml_states.txt's path."""
    by, won = read_pairs(pairs_paths)
    states = {}
    for code, rows in by.items():
        s = match_states(rows, won.get(code))
        if s:
            states[code] = s
    if not states:
        raise SystemExit("no match in the pairs runs from 0-0 with a favourite and a result")
    firsts = sum(1 for s in states.values() if s["first"])
    L = [f"\nThe moneyline by game situation: {', '.join(os.path.basename(p) for p in pairs_paths)}",
         f"  {len(states):,} matches from 0-0 with a pre-match favourite (prod's kick-off price beyond"
         f" {100 * (0.5 + EVENS):.0f}%) and a result; {firsts:,} with a first score read; the candidate: {label}"]
    L += situation_lines(list(states.values()), label)
    if bets_paths:
        placed = place_bets(read_bets(bets_paths), states)
        L.append(f"\n  {len(placed):,} settled in-play moneyline bets on these matches")
        L += bet_lines(placed, label)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ml_states.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        for code, s in sorted(states.items()):
            f = s["first"] or {}
            w.writerow(dict(match_code=code, fav="home" if s["fav"] == 1 else "away", prod_kickoff=round(s["prod_kickoff"], 4),
                            cand_kickoff=None if s["cand_kickoff"] is None else round(s["cand_kickoff"], 4),
                            first_score=f.get("situation"), prod_after=f.get("prod"), cand_after=f.get("cand"),
                            fav_won=int(s["fav_won"])))
    path = os.path.join(out_dir, "ml_states.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n  -> {path} and ml_states.csv")
    return path
