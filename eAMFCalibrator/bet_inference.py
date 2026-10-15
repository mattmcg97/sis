"""Inference: the money when two gamers meet after either has played someone else that session, and
where it goes on what those games said. A beat B and C beat D: at A v D, is the money on A, does A
win more often than prod's price says, and does prod move?

  session   a gamer's session (bet_rematches.day_meetings: no gap over config.SESSION_BREAK_MINUTES)
  others    a gamer's earlier matches that session against anyone but this match's opponent; the
            last of them is his last other game: won, lost, drew (by how much) or none (fresh)
  form      the side the last other games favour: the gamer whose last other margin (from his side,
            0 when fresh) is the bigger. The state names both: "won v lost" is A v D above, "won v
            won" both won theirs, "fresh v lost" one hadn't played anyone else and the other had lost.
            gap: the form side's last margin less the other's (1-7, 8-14, 15-21, 22+); level: equal,
            no form side
  common    a common opponent: both gamers played the same third gamer earlier that session (the
            latest such); beat v lost to: one beat him and the other lost to him. Its side: the
            gamer who did better against him
  surprise  a gamer's last other margin less the margin prod's moneyline implied for him then
            (rematch_layer.implied_margin): what that game said beyond what prod already knew
  money     every single (moneyline, handicap, total; pre-match and in play), the moneyline and
            handicap read as on the form side or the other side
  residual  on the moneyline, won less prod's probability (the form side's at the match; a bet's at
            the bet): + where the side wins more than prod said (2se clustered by match)
  totals    the match's total less prod's line, against the two gamers' last other games' (their
            total less prod's line then)
  fits      how far to move a price per point of the last other games' surprises and total misses
  layers    prod alone, with the rematch layer and with the inference layer (inference_layer.py),
            walk-forward on prod's prices

    python -m eAMFCalibrator bets inference --since "2026-09-01 00:00:00"
"""

import math
import os
from collections import defaultdict

from . import bets, config, inference_layer, markets, rematch_layer, snowflake_io
from .bet_losers import prematch_p1
from .bet_rematches import (HOME_SIDE, _f, _fit_row, _group, _logit, _ordered, _pc, day_meetings,
                            logistic_offset, margin, meeting_lines, other_lines, prematch_line)
from .bet_sessions import group_of
from .bet_totals import _start, mean_se

STATES = ("won v lost", "won v drew", "drew v lost", "won v fresh", "fresh v lost", "drew v fresh",
          "fresh v drew", "won v won", "lost v lost", "drew v drew", "level", "fresh v fresh")
GAPS = ("1-7", "8-14", "15-21", "22+")
COMMON = ("beat v lost to", "beat v drew with", "drew with v lost to", "both beat", "both lost to", "level")
FORM, OTHER = "form side", "other side"
LAST_MISS = ("last others under 7+", "last others under 2-7", "last others within 2", "last others over 2-7",
             "last others over 7+")
PLAYED = ("rematch", "first, after others", "first, no others")
SAME_TEAM = ("same team as his last other", "team changed")
LAYER_MIN_DAYS = 7


def _word(m):
    return "fresh" if m is None else "won" if m > 0 else "lost" if m < 0 else "drew"


def gap_band(g):
    if g is None:
        return None
    return GAPS[0] if g <= 7 else GAPS[1] if g <= 14 else GAPS[2] if g <= 21 else GAPS[3]


def last_miss_band(x):
    if x is None:
        return None
    return (LAST_MISS[0] if x <= -7 else LAST_MISS[1] if x <= -2 else LAST_MISS[2] if x < 2
            else LAST_MISS[3] if x < 7 else LAST_MISS[4])


def session_games(meets):
    """gamer session key -> its matches (day_meetings' rows), in kick-off order."""
    out = defaultdict(list)
    for m in meets.values():
        for k in m.get("gamer_keys") or ():
            out[k].append(m)
    for v in out.values():
        v.sort(key=lambda m: m["start"])
    return out


def _from(m, g):
    """(g's margin, the total) in a meeting, or (None, None) without a result."""
    s1, s2 = m.get("home_score"), m.get("away_score")
    if s1 is None or s2 is None:
        return None, None
    return (s1 - s2) if m["home"] == g else (s2 - s1), s1 + s2


def others_of(m, g, key, games):
    """g's earlier matches that session against anyone but this match's opponent, with results."""
    opp = m["away"] if m["home"] == g else m["home"]
    return [x for x in games.get(key, ()) if x["start"] < m["start"] and opp not in (x["home"], x["away"])
            and _from(x, g)[0] is not None]


def read_game(x, g, prod_tl):
    """One of g's other games: his margin, his surprise against prod's moneyline then, the total's miss
    against prod's line then, and his team."""
    mg, total = _from(x, g)
    p1, line = prematch_p1(prod_tl, x["match"]), prematch_line(prod_tl, x["match"])
    implied = None if p1 is None else rematch_layer.implied_margin(p1 if x["home"] == g else 1 - p1)
    return dict(match=x["match"], margin=mg, surprise=None if implied is None else mg - implied,
                miss=None if line is None else total - line, team=x["team"].get(g),
                opponent=x["away"] if x["home"] == g else x["home"])


def common_opponent(m, home_games, away_games):
    """(label, the gamer who did better against him, the margin chain, the opponent) off the latest
    third gamer both played earlier that session, or Nones."""
    h_by, a_by = {}, {}
    for g, by, games in ((m["home"], h_by, home_games), (m["away"], a_by, away_games)):
        for x in games:
            by[x["opponent"]] = x                     # the latest against him
    both = [z for z in h_by if z in a_by]
    if not both:
        return None, None, None, None
    z = max(both, key=lambda z: max(h_by[z]["start"], a_by[z]["start"]))
    mh, ma = h_by[z]["margin"], a_by[z]["margin"]
    if mh == ma:
        return COMMON[-1], None, 0, z
    side, hi, lo = (m["home"], mh, ma) if mh > ma else (m["away"], ma, mh)
    word = lambda v: "beat" if v > 0 else "lost to" if v < 0 else "drew with"
    a, b = word(hi), word(lo)
    label = f"both {a}" if a == b else f"{a} v {b}"
    return label, side, hi - lo, z


def match_row(m, rows, prod_tl, games):
    """One match: what each gamer had played that session, the form side and its price, the common
    opponent, the result, the totals and the money."""
    hk, ak = m["gamer_keys"]
    out = dict(match_code=m["match"], start=m["start"], home=m["home"], away=m["away"], k=m["k"],
               meeting=m["meeting"], margin=m["margin"], total=m["total"])
    reads = {}
    for g, key in ((m["home"], hk), (m["away"], ak)):
        gs = []
        for x in others_of(m, g, key, games):
            r = read_game(x, g, prod_tl)
            r["start"] = x["start"]
            gs.append(r)
        reads[g] = gs
    last = {g: (gs[-1] if gs else None) for g, gs in reads.items()}
    lm = {g: (v["margin"] if v else None) for g, v in last.items()}
    out.update(home_others=len(reads[m["home"]]), away_others=len(reads[m["away"]]),
               home_last=lm[m["home"]], away_last=lm[m["away"]],
               home_last_surprise=last[m["home"]]["surprise"] if last[m["home"]] else None,
               away_last_surprise=last[m["away"]]["surprise"] if last[m["away"]] else None,
               home_last_miss=last[m["home"]]["miss"] if last[m["home"]] else None,
               away_last_miss=last[m["away"]]["miss"] if last[m["away"]] else None)
    out["played"] = PLAYED[0] if m["k"] > 1 else PLAYED[1] if any(reads.values()) else PLAYED[2]
    h, a = (lm[m["home"]] or 0), (lm[m["away"]] or 0)
    if not any(reads.values()):
        form, state = None, STATES[-1]
    elif h == a:
        form, state = None, STATES[-2]
    else:
        form = m["home"] if h > a else m["away"]
        other = m["away"] if form == m["home"] else m["home"]
        state = f"{_word(lm[form])} v {_word(lm[other])}"
    out.update(state=state, form=form, gap=None if form is None else abs(h - a),
               gap_band=None if form is None else gap_band(abs(h - a)))
    if form is not None:
        other = m["away"] if form == m["home"] else m["home"]
        sf = last[form]["surprise"] if last[form] else 0.0
        so = last[other]["surprise"] if last[other] else 0.0
        out["form_gap_surprise"] = None if sf is None or so is None else sf - so
        out["form_same_team"] = (None if last[form] is None else
                                 SAME_TEAM[0] if last[form]["team"] == m["team"].get(form) else SAME_TEAM[1])
    p1 = prematch_p1(prod_tl, m["match"])
    out["p1"] = p1
    win = None if m["home_score"] is None or m["away_score"] is None or m["home_score"] == m["away_score"] \
        else (m["home"] if m["home_score"] > m["away_score"] else m["away"])

    def side_of(g):
        if g is None or p1 is None:
            return None, None, None
        p = p1 if g == m["home"] else 1 - p1
        won = None if win is None else float(win == g)
        return p, won, None if won is None else won - p
    out["form_p"], out["form_won"], out["form_res"] = side_of(form)
    label, co, chain, z = common_opponent(m, reads[m["home"]], reads[m["away"]])
    out.update(common=label, common_side=co, common_chain=chain, common_opponent=z)
    out["co_p"], out["co_won"], out["co_res"] = side_of(co)
    line = prematch_line(prod_tl, m["match"])
    out["line"] = line
    out["miss"] = None if line is None or m["total"] is None else m["total"] - line
    lmiss = [v for v in (out["home_last_miss"], out["away_last_miss"]) if v is not None]
    out["last_miss"] = sum(lmiss) / len(lmiss) if lmiss else None
    out["last_miss_band"] = last_miss_band(out["last_miss"])
    if p1 is not None and m["home_score"] is not None and m["away_score"] is not None:
        out["home_margin_miss"] = (m["home_score"] - m["away_score"]) - rematch_layer.implied_margin(p1)
    for mk in (markets.MONEYLINE, markets.SPREAD, markets.TOTAL):
        rs = [r for r in rows if r["market"] == mk]
        out[f"{mk}_stake"] = sum(r["stake"] for r in rs)
        out[f"{mk}_revenue"] = sum(r["revenue"] for r in rs)
    out["moneyline_on_form"] = sum(r["stake"] for r in rows if r["market"] == markets.MONEYLINE
                                   and r["side"] == FORM)
    out["stake"] = sum(r["stake"] for r in rows)
    out["revenue"] = sum(r["revenue"] for r in rows)
    return out


def bet_row(b, r, mrow):
    """One bet on a match, read against the form side and the common opponent's side."""
    mk = markets.market_group(b.feed_market)
    side = HOME_SIDE.get(b.feed_market)
    backed = None if side is None else (mrow["home"] if side == 1 else mrow["away"])

    def rel(g):
        if side is None:
            return markets.selection_label(b.feed_market)
        return None if g is None else FORM if backed == g else OTHER
    won = {bets.WON: 1.0, bets.LOST: 0.0}.get(r.get("result"))
    p = r.get("stream_prob")
    return dict(match_code=b.match_code, market=mk, side=rel(mrow["form"]), co_side=rel(mrow["common_side"]),
                in_play=b.in_play, group=group_of(b), stake=b.stake, revenue=b.revenue,
                settled=r.get("result") in (bets.WON, bets.LOST, bets.PUSH), p=p, won=won,
                res=None if won is None or p is None or mk != markets.MONEYLINE else won - p,
                state=mrow["state"], gap_band=mrow["gap_band"], common=mrow["common"], played=mrow["played"],
                k=mrow["k"], form_same_team=mrow.get("form_same_team"), last_miss_band=mrow["last_miss_band"])


# --- reporting -------------------------------------------------------------------------------

ML_HEAD = (f"  {'':24s} {'matches':>7s} {'p side':>6s} {'won':>6s} {'resid':>6s} {'2se':>5s} {'ML stake':>10s} "
           f"{'on side':>7s} {'mgn':>6s} {'res':>6s} {'mgn other':>9s} {'res':>6s}")


def ml_lines(brows, mrows, key, order, title, prefix="form", side_key="side", min_matches=5):
    """Lines: matches by key on the moneyline -- prod's price for the side (the form side, or the
    common opponent's), how often it won against it, and the money on it and against it."""
    ms_all = [r for r in mrows if r.get(f"{prefix}_p") is not None]
    codes = {r["match_code"] for r in ms_all}
    gm = _group(ms_all, key)
    gb = _group([r for r in brows if r["market"] == markets.MONEYLINE and r.get(side_key) in (FORM, OTHER)
                 and r["match_code"] in codes], key)
    L = [f"\n  {title}", ML_HEAD]
    for k in _ordered(gm, order):
        ms, bs = gm[k], gb.get(k, [])
        if len(ms) < min_matches:
            continue
        stake = sum(r["stake"] for r in bs)
        on = [r for r in bs if r[side_key] == FORM]
        off = [r for r in bs if r[side_key] == OTHER]
        ps = [r[f"{prefix}_p"] for r in ms]
        wins = [r[f"{prefix}_won"] for r in ms if r[f"{prefix}_won"] is not None]
        _, res, se = mean_se([r for r in ms if r[f"{prefix}_res"] is not None], f"{prefix}_res")
        r_on = mean_se([r for r in on if r["res"] is not None], "res")[1]
        r_off = mean_se([r for r in off if r["res"] is not None], "res")[1]
        L.append(f"  {str(k)[:24]:24s} {len(ms):7,d} {_pc(sum(ps) / len(ps))} "
                 f"{_pc(sum(wins) / len(wins) if wins else None)} {_pc(res, 6, '+.1f')} "
                 f"{_pc(None if se is None else 2 * se, 5)} {stake:10,.0f} "
                 f"{_pc(sum(r['stake'] for r in on) / stake if stake else None, 7)} {_pc(margin(on))} "
                 f"{_pc(r_on, 6, '+.1f')} {_pc(margin(off), 9)} {_pc(r_off, 6, '+.1f')}")
    return L


def _ms(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    return m, 2 * math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def totals_lines(mrows):
    """Lines: the match's total against prod's line, by the gamers' last other games' misses."""
    g = _group([r for r in mrows if r["miss"] is not None], "last_miss_band")
    L = ["\n  Totals against prod's line, by the two gamers' last other games' total less prod's line then "
         "(their mean; miss: this total less its line; 2se by match)",
         f"  {'':24s} {'matches':>7s} {'last miss':>9s} {'line':>6s} {'miss':>6s} {'2se':>5s} {'over':>6s}"]
    for k in _ordered(g, LAST_MISS):
        rs = g[k]
        m, se = _ms([r["miss"] for r in rs])
        L.append(f"  {str(k)[:24]:24s} {len(rs):7,d} {_f(sum(r['last_miss'] for r in rs) / len(rs), 9, '+.2f')} "
                 f"{_f(sum(r['line'] for r in rs) / len(rs), 6, '.1f')} {_f(m, 6, '+.2f')} {_f(se, 5, '.2f')} "
                 f"{_pc(sum(1 for r in rs if r['miss'] > 0) / len(rs), 6)}")
    return L


def fit_lines(mrows):
    """Lines: how far to move a price per point of the gamers' last other games."""
    first = [r for r in mrows if r["k"] == 1]
    both = [r for r in first if r["home_last_surprise"] is not None and r["away_last_surprise"] is not None]
    L = ["\n  ==== How far to move a price: fits on the gamers' last other games (first meetings of the "
         "pair that session) ====",
         "  each slope: the points this match's miss moves per point of that game's (home's and away's "
         "each from their own side, so form carrying over shows as + on home's and - on away's)",
         f"\n  {'':46s} {'n':>6s}",
         "  Margin: the home margin less prod's (its moneyline read as a margin)"]
    L.append(_fit_row("both had played others", both, "home_margin_miss",
                      ["home_last_surprise", "away_last_surprise"]))
    L.append(_fit_row("home had (away fresh)", [r for r in first if r["home_last_surprise"] is not None
                                                and r["away_others"] == 0], "home_margin_miss", ["home_last_surprise"]))
    L.append(_fit_row("away had (home fresh)", [r for r in first if r["away_last_surprise"] is not None
                                                and r["home_others"] == 0], "home_margin_miss", ["away_last_surprise"]))
    L.append("  Total: the total less prod's line, on each gamer's last other game's total less its line")
    L.append(_fit_row("both had played others", [r for r in first if r["home_last_miss"] is not None
                                                 and r["away_last_miss"] is not None], "miss",
                      ["home_last_miss", "away_last_miss"]))
    L.append("  Moneyline: the form side winning, logistic with prod's price as the offset, on the form side's "
             "last surprise less the other's, per 10 points")
    for label, rs in (("every match with a form side", mrows), ("first meetings", first),
                      ("won v lost", [r for r in mrows if r["state"] == "won v lost"])):
        rs = [r for r in rs if r.get("form_won") is not None and r.get("form_gap_surprise") is not None
              and r.get("form_p") is not None]
        fit = logistic_offset([r["form_gap_surprise"] / 10 for r in rs], [r["form_won"] for r in rs],
                              [_logit(r["form_p"]) for r in rs]) if len(rs) >= 30 else None
        if fit is None:
            L.append(f"  {label:46s} {len(rs):6,d}  (too few, or no fit)")
            continue
        a, b, sa, sb = fit
        L.append(f"  {label:46s} {len(rs):6,d}  const {a:+.3f} ±{sa:.3f}  per 10 points {b:+.3f} ±{sb:.3f} "
                 f"(t {b / sb if sb else 0:+.1f}); at evens, 10 points of surprise gap -> "
                 f"{100 * (1 / (1 + math.exp(-(a + b))) - 0.5):+.1f} points")
    return L


def layer_lines(scored, layers, min_days):
    """Lines: prod alone, with the rematch layer and with the inference layer, walk-forward."""
    active = [r for r in scored if r.get("active")]
    L = ["\n  ==== The layers on prod's prices (refitted each day on the days before it; the first "
         f"{min_days} days are their burn-in) ====", "  the rematch layer's last fit:"]
    L += layers["rematch"].describe()
    L += ["  the inference layer's last fit:"] + layers["inference"].describe()
    L.append(f"\n  {'':30s} {'matches':>7s} {'log loss':>9s} {'+layer':>8s} {'t':>5s} {'+infer':>8s} {'t':>5s} "
             f"{'tot RMSE':>8s} {'+layer':>7s} {'t':>5s} {'+infer':>7s} {'t':>5s}")
    groups = [("every match", active)]
    groups += [(p, [r for r in active if r["played"] == p]) for p in PLAYED]
    groups += [(f"first, {s}", [r for r in active if r["played"] != PLAYED[0] and r["state"] == s]) for s in STATES]
    groups += [(f"common: {c}", [r for r in active if r["common"] == c]) for c in COMMON]
    for label, rs in groups:
        if len(rs) < 20:
            continue
        a = rematch_layer.score(rs)
        b = rematch_layer.score(rs, "margin_adj", "total_adj")
        c = rematch_layer.score(rs, "infer_margin_adj", "infer_total_adj")
        db, tb = rematch_layer.paired(a["ll"], b["ll"])
        dc, tc = rematch_layer.paired(a["ll"], c["ll"])
        _, ttb = rematch_layer.paired(a["te"], b["te"])
        _, ttc = rematch_layer.paired(a["te"], c["te"])
        L.append(f"  {label[:30]:30s} {len(rs):7,d} {_f(a['log_loss'], 9, '.4f')} {_f(db, 8, '+.4f')} "
                 f"{_f(tb, 5, '+.1f')} {_f(dc, 8, '+.4f')} {_f(tc, 5, '+.1f')} {_f(a['total_rmse'], 8, '.2f')} "
                 f"{_f(b['total_rmse'], 7, '.2f')} {_f(ttb, 5, '+.1f')} {_f(c['total_rmse'], 7, '.2f')} "
                 f"{_f(ttc, 5, '+.1f')}")
    L.append("  +layer / +infer: the log loss's change on prod alone (negative is better), t match by match; the "
             "total's RMSE with each, t on its squared error")
    return L


def report(brows, mrows):
    """Lines of text."""
    sided = [r for r in mrows if r["form"] is not None]
    first = [r for r in mrows if r["k"] == 1]
    sb = [r for r in brows if r["side"] in (FORM, OTHER) or r["market"] == markets.TOTAL]
    L = [f"  {len(mrows):,} settled matches in the window: {sum(r['played'] == PLAYED[1] for r in mrows):,} first "
         f"meetings of a pair that session after either gamer had played someone else, "
         f"{sum(r['played'] == PLAYED[0] for r in mrows):,} rematches, "
         f"{sum(r['played'] == PLAYED[2] for r in mrows):,} with neither having played anyone else; "
         f"{len(brows):,} bets, {sum(r['stake'] for r in brows):,.0f} stake",
         f"  a session: no gap over {config.SESSION_BREAK_MINUTES} minutes; others: a gamer's earlier matches "
         "that session against anyone but this opponent; the state: each gamer's last other game, the form "
         "side's first (won v lost: A v D after A beat B and C beat D)",
         "  p side: prod's pre-match moneyline for the side; won: how often it won; resid: won less p (2se by "
         "match); on side: the share of the moneyline stake on it; mgn: the book's margin on it and against "
         "it; res: won less prod's p at the bet, per bet"]
    L.append("\n  ==== How much is bet, by what the gamers had played that session ====")
    L += meeting_lines(mrows, "played", PLAYED, "Every match")
    L += meeting_lines(mrows, "state", STATES, "Every match by the state (each gamer's last other game)")
    L += meeting_lines([r for r in sided if r["state"] == "won v lost"], "gap_band", GAPS,
                       "Won v lost, by the gap between their last margins")
    L.append("\n  ==== Where the moneyline money goes, against the form side ====")
    L += ml_lines(sb, sided, "state", STATES, "Every match with a form side, by the state")
    L += ml_lines(sb, [r for r in sided if r["k"] == 1], "state", STATES,
                  "First meetings of the pair that session, by the state")
    L += ml_lines(sb, [r for r in sided if r["k"] > 1], "state", STATES,
                  "Rematches, by the state (the rematch layer's ground as well)")
    L += ml_lines(sb, sided, "gap_band", GAPS, "Every match with a form side, by the gap between the last margins")
    L += ml_lines([r for r in sb if r["state"] == "won v lost"], [r for r in sided if r["state"] == "won v lost"],
                  "gap_band", GAPS, "Won v lost, by the gap")
    L += ml_lines(sb, sided, "form_same_team", SAME_TEAM, "By whether the form side is on the team he had in his "
                  "last other game")
    L += ml_lines([r for r in sb if not r["in_play"]], sided, "state", STATES,
                  "Pre-match bets only, by the state (the match columns are every match)")
    L += ml_lines([r for r in sb if r["in_play"]], sided, "state", STATES,
                  "In-play bets only, by the state (the match columns are every match)")
    for grp in ("Restricted", "VIP"):
        L += ml_lines([r for r in sb if r["group"] == grp], sided, "state", STATES,
                      f"{grp} bets only, by the state (the match columns are every match)")
    L.append("\n  ==== A common opponent: both gamers played the same third gamer earlier that session ====")
    L += ml_lines(brows, [r for r in mrows if r["common_side"] is not None], "common", COMMON,
                  "By how each did against him (the side: the gamer who did better against him)", "co", "co_side")
    L += ml_lines(brows, [r for r in first if r["common_side"] is not None], "common", COMMON,
                  "First meetings of the pair, by how each did against him", "co", "co_side")
    L.append("\n  ==== The handicap and the total ====")
    L += other_lines(sb, "state", STATES, "Handicap: the money on the form side, by the state", markets.SPREAD,
                     (FORM, OTHER))
    L += other_lines(sb, "last_miss_band", LAST_MISS, "Total: over and under, by the gamers' last other games' "
                     "total less prod's line", markets.TOTAL, ("Over", "Under"))
    L += totals_lines([r for r in mrows if r["last_miss"] is not None and r["k"] == 1])
    L += fit_lines(mrows)
    big = sorted(sided, key=lambda r: -r["moneyline_stake"])[:25]
    L.append("\n  The 25 matches with a form side and the most moneyline stake (last: each gamer's last other "
             "margin; p: prod's for the form side; won: did it win)")
    L.append(f"  {'start (UTC)':16s} {'match':34s} {'mtg':4s} {'state':13s} {'last':9s} {'p':>4s} {'won':>3s} "
             f"{'ML stake':>9s} {'on form':>7s} {'ML rev':>8s}")
    for r in big:
        lasts = f"{_signed(r['home_last'])}/{_signed(r['away_last'])}"
        L.append(f"  {str(r['start'])[:16]:16s} {(r['home'] + ' v ' + r['away'])[:34]:34s} {r['meeting']:4s} "
                 f"{r['state'][:13]:13s} {lasts:9s} {_pc(r['form_p'], 4, '.0f')} "
                 f"{('-' if r['form_won'] is None else 'Y' if r['form_won'] else 'N'):>3s} "
                 f"{r['moneyline_stake']:9,.0f} "
                 f"{_pc(r['moneyline_on_form'] / r['moneyline_stake'] if r['moneyline_stake'] else None, 7, '.0f')} "
                 f"{r['moneyline_revenue']:8,.0f}")
    return L


def _signed(v):
    return "-" if v is None else f"{v:+d}"


# --- run -------------------------------------------------------------------------------------

def prod_layers(meets, codes, prod_tl, info, min_days=LAYER_MIN_DAYS):
    """The rematch and inference layers on prod's pre-match prices, walk-forward over these matches:
    (rows with both readings, {"rematch": its last fit, "inference": its last fit})."""
    items = []
    for code in codes:
        m = meets.get(code)
        p1, line = prematch_p1(prod_tl, code), prematch_line(prod_tl, code)
        if m is None or p1 is None or line is None:
            continue
        s1, s2 = m.get("home_score"), m.get("away_score")
        items.append(rematch_layer.item(code, m["start"], m["day_key"], m["home"], m["away"], m["team"][m["home"]],
                                        m["team"][m["away"]], rematch_layer.implied_margin(p1), line,
                                        None if s1 is None or s2 is None else s1 - s2,
                                        None if s1 is None or s2 is None else s1 + s2, m.get("gamer_keys")))
    rows, layer = rematch_layer.walk_forward(items, min_days)
    inferred, ilayer = inference_layer.walk_forward(items, min_days)
    by = {r["match"]: r for r in inferred}
    first = min((r["start"] for r in rows), default=None)
    for r in rows:
        x = by[r["match"]]
        r.update(infer_margin_adj=r["margin"] + x["infer_margin"], infer_total_adj=r["total"] + x["infer_total"],
                 active=first is not None and (r["start"].date() - first.date()).days >= min_days,
                 **{k: info.get(r["match"], {}).get(k) for k in ("played", "state", "common")})
    return rows, dict(rematch=layer, inference=ilayer)


def run(cur, out_dir):
    """Every settled match of the window with what its gamers had played that session, prod's
    pre-match prices and every bet on it; print the report, write the CSV."""
    history = snowflake_io.fetch_history(cur)
    meets = day_meetings(history)
    games = session_games(meets)
    since = _start(config.CUTOFF_START)
    until = _start(config.CUTOFF_END) if config.CUTOFF_END else None
    window = sorted((m for m in meets.values() if m["start"] >= since and (until is None or m["start"] < until)),
                    key=lambda m: m["start"])
    in_window = {m["match"] for m in window}
    before = {x["match"] for m in window for k in m["gamer_keys"] for x in games.get(k, ()) if x["start"] < m["start"]}
    codes = sorted(in_window | before)
    print(f"  {len(window):,} settled matches in the window; prod's prices for {len(codes):,} (with the gamers' "
          "earlier games those sessions)", flush=True)
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    every = [b for b in bets.to_bets(cols, raw) if b.feed_market is not None and b.match_code in in_window]
    prod_rows = []
    for start in range(0, len(codes), config.MATCH_CHUNK_SIZE):
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], codes[start:start + config.MATCH_CHUNK_SIZE])
        print(f"  prod's quotes: {min(start + config.MATCH_CHUNK_SIZE, len(codes)):,} of {len(codes):,} matches",
              flush=True)
    prod_tl = bets.timeline(prod_rows)
    del prod_rows
    # prod's pre-match prices: its closing quotes, before each match's first play (or its kick-off)
    pre_tl = bets.closing_timeline(prod_tl, snowflake_io.fetch_first_plays(cur, codes),
                                   {c: meets[c]["start"] for c in codes if c in meets})
    mrows = [match_row(m, [], pre_tl, games) for m in window]
    info = {r["match_code"]: r for r in mrows}
    lags = bets.fit_lags(every, prod_tl)
    joined = bets.join(every, lags, bets.fit_line_signs(every, prod_tl, lags), prod_tl, {}, None, {})
    brows = [bet_row(b, r, info[b.match_code]) for b, r in zip(every, joined)]
    by = defaultdict(list)
    for r in brows:
        by[r["match_code"]].append(r)
    mrows = [match_row(m, by.get(m["match"], []), pre_tl, games) for m in window]
    info = {r["match_code"]: r for r in mrows}
    scored, layers = prod_layers(meets, codes, pre_tl, info)
    scored = [r for r in scored if r["match"] in in_window]
    lines = report(brows, mrows) + layer_lines(scored, layers, LAYER_MIN_DAYS)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    bets.write_csv(os.path.join(out_dir, "bets_inference.csv"), mrows, list(dict.fromkeys(k for r in mrows for k in r)))
    path = os.path.join(out_dir, "bets_inference.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
