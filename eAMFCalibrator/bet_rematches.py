"""Rematches: the money when two gamers meet again the same day, and where it goes after the first
result.

  day       a gamer's session (no gap over config.SESSION_BREAK_MINUTES between his matches, so a
            night that runs past midnight UTC is one day); a pair's meetings in the first gamer's
            (alphabetically) session are numbered 1st, 2nd, 3rd, 4th+
  teams     a rematch against the day's first meeting: the same teams (each gamer on the team he had
            then: the exact matchup), swapped (each on the other's), or changed
  first     the day's first meeting: its winner, its margin of victory (draw, 1-3, 4-7, 8-13, 14-20,
            21+) and its total
  money     every single (moneyline, handicap, total; pre-match and in play) on each match, the
            moneyline and handicap read against the first meeting's winner (on him, or on his
            opponent) and the total as over or under
  price     prod's pre-match moneyline for the first meeting's winner, at the first meeting and at
            the rematch: whether prod moved after the result, and how often he won again against it
  residual  on the moneyline, won less prod's probability at the bet: 0 where prod is right, + where
            the side backed wins more than prod said (2se clustered by match)
  totals    prod's pre-match total line at the first meeting and at the rematch, and how far each
            match's total landed from it (its miss): does the first meeting's miss carry over?
  fits      how much to move a rematch's price per point of the first meeting's miss: the rematch's
            total miss on the first's (and the previous meeting's from the 3rd on), its margin miss
            on the first's (prod's moneyline read as a margin, MARGIN_SD), and the first winner
            winning again, logistic on prod's price and the first's margin miss
  layer     rematch_layer.py run on prod's prices walk-forward: what it moves and what that does to
            the log loss, Brier and total error of the rematches

    python -m eAMFCalibrator bets rematches --since "2026-10-01 00:00:00"
"""

import os
from collections import defaultdict

import math

from . import bets, config, markets, rematch_layer, snowflake_io
from .bet_losers import prematch_p1
from .bet_sessions import group_of
from .bet_totals import _start, mean_se
from .totals_signals import ols

MARGINS = ("draw", "1-3", "4-7", "8-13", "14-20", "21+")
MEETINGS = ("1st", "2nd", "3rd", "4th+")
TEAMS = ("same teams", "swapped teams", "teams changed")
TOTALS = ("first total <28", "first total 28-41", "first total 42+")
MISSES = ("first under 10+", "first under 3-10", "first within 3", "first over 3-10", "first over 10+")
WINNER, LOSER, DRAWN = "first winner", "first loser", "first drawn"
HOME_SIDE = {50: 1, 52: 1, 51: 2, 53: 2}
MIN_BETS = 30


def _i(v):
    try:
        return None if v is None or v == "" else int(float(v))
    except (TypeError, ValueError):
        return None


def margin_band(m):
    if m is None:
        return None
    m = abs(m)
    if m == 0:
        return MARGINS[0]
    return next(b for b, hi in zip(MARGINS[1:], (3, 7, 13, 20, 10 ** 6)) if m <= hi)


def total_band(t):
    if t is None:
        return None
    return TOTALS[0] if t < 28 else TOTALS[1] if t < 42 else TOTALS[2]


def miss_band(x):
    """The first meeting's total less prod's line, banded."""
    if x is None:
        return None
    return (MISSES[0] if x <= -10 else MISSES[1] if x <= -3 else MISSES[2] if x < 3
            else MISSES[3] if x < 10 else MISSES[4])


def prematch_line(tl, match, market=54, first_play=None, kickoff=None):
    """prod's last pre-match line on a market, or None. With the match's first play or its kick-off:
    the closing quote's (bets.closing_quote). Without: its last line with no message, else its first."""
    entry = tl.get((match, market))
    if not entry or not entry[1]:
        return None
    if first_play is not None or kickoff is not None:
        q = bets.closing_quote(tl, match, market, first_play, kickoff, need_line=True)
        return None if q is None else q[2]
    pre = [q for q in entry[1] if q[0] is None and q[2] is not None]
    q = pre[-1] if pre else next((q for q in entry[1] if q[2] is not None), None)
    return q[2] if q else None


def meeting_label(k):
    return MEETINGS[min(k, 4) - 1]


def summary(g):
    """A meeting's result: its winner and loser (None on a draw), margin, total, and who had which
    team."""
    s1, s2 = g["home_score"], g["away_score"]
    out = dict(match=g["match"], start=g["start"], home=g["home"], away=g["away"],
               team={g["home"]: g["home_team"], g["away"]: g["away_team"]},
               winner=None, loser=None, margin=None, total=None)
    if s1 is not None and s2 is not None:
        out.update(margin=abs(s1 - s2), total=s1 + s2)
        if s1 != s2:
            out["winner"], out["loser"] = (g["home"], g["away"]) if s1 > s2 else (g["away"], g["home"])
    return out


def teams_against(g, first):
    """same teams / swapped teams / teams changed: this meeting's teams against the first's."""
    now = {g["home"]: g["home_team"], g["away"]: g["away_team"]}
    was = first["team"]
    if all(now[h] == was.get(h) for h in now):
        return TEAMS[0]
    h1, h2 = g["home"], g["away"]
    if now[h1] == was.get(h2) and now[h2] == was.get(h1):
        return TEAMS[1]
    return TEAMS[2]


def sessions_of(games, gap):
    """(gamer, match) -> the gamer's session number, off games in start order: a new session after
    a gap over `gap` seconds."""
    out, last = {}, {}
    for g in games:
        for h in (g["home"], g["away"]):
            prev = last.get(h)
            if prev is None:
                n = 0
            else:
                n = out[(h, prev[1])] + ((g["start"] - prev[0]).total_seconds() > gap)
            out[(h, g["match"])] = n
            last[h] = (g["start"], g["match"])
    return out


def day_meetings(history, gap=None):
    """match -> its place among the day's meetings of its two gamers: the pair, the meeting number,
    the day's first meeting and the one before, the teams against the first meeting's, and each
    gamer's session (gamer_keys: (home, its session), (away, its session))."""
    gap = (config.SESSION_BREAK_MINUTES if gap is None else gap) * 60
    games = []
    for r in history:
        t = _start(r.get("SCHEDULED_START_TIME_UTC"))
        h1 = (r.get("PLAYER_1_HANDLE") or "").strip().upper()
        h2 = (r.get("PLAYER_2_HANDLE") or "").strip().upper()
        if t is None or not h1 or not h2:
            continue
        games.append(dict(match=r["MATCH_CODE"], start=t, home=h1, away=h2, home_team=r.get("PLAYER_1_TEAM"),
                          away_team=r.get("PLAYER_2_TEAM"), home_score=_i(r.get("PLAYER_1_FINAL_SCORE")),
                          away_score=_i(r.get("PLAYER_2_FINAL_SCORE"))))
    games.sort(key=lambda g: g["start"])
    session = sessions_of(games, gap)
    days = defaultdict(list)
    for g in games:
        a, b = sorted((g["home"], g["away"]))
        days[(a, b, session[(a, g["match"])])].append(g)
    out = {}
    for key, gs in days.items():
        a, b, _ = key
        first = summary(gs[0])
        for k, g in enumerate(gs, 1):
            out[g["match"]] = dict(summary(g), pair=f"{a} v {b}", k=k, meeting=meeting_label(k), first=first,
                                   prev=summary(gs[k - 2]) if k > 1 else None,
                                   teams=None if k == 1 else teams_against(g, first),
                                   day_key=key, earlier=[x["match"] for x in gs[:k - 1]],
                                   home_score=g["home_score"], away_score=g["away_score"],
                                   gamer_keys=tuple((h, session[(h, g["match"])]) for h in (g["home"], g["away"])))
    return out


def bet_row(b, r, m):
    """One bet on a match, read against the day's first meeting."""
    mk = markets.market_group(b.feed_market)
    side = HOME_SIDE.get(b.feed_market)
    fw = m["first"]["winner"]
    if side is None:
        rel = markets.selection_label(b.feed_market)
    else:
        backed = m["home"] if side == 1 else m["away"]
        rel = DRAWN if fw is None else WINNER if backed == fw else LOSER
    won = {bets.WON: 1.0, bets.LOST: 0.0}.get(r.get("result"))
    p = r.get("stream_prob")
    return dict(match_code=b.match_code, market=mk, side=rel, in_play=b.in_play, group=group_of(b),
                stake=b.stake, revenue=b.revenue, settled=r.get("result") in (bets.WON, bets.LOST, bets.PUSH),
                p=p, won=won, res=None if won is None or p is None or mk != markets.MONEYLINE else won - p,
                k=m["k"], meeting=m["meeting"], teams=m["teams"], first_band=margin_band(m["first"]["margin"]),
                first_total=total_band(m["first"]["total"]))


def match_row(m, rows, prod_tl):
    """One match: its place in the day, the first meeting, prod's price of the first winner then
    and now, the result, and the money."""
    fw = m["first"]["winner"]

    def p_of(match, g):
        p = prematch_p1(prod_tl, match)
        return None if p is None or fw is None else p if g["home"] == fw else 1 - p
    out = dict(match_code=m["match"], start=m["start"], pair=m["pair"], meeting=m["meeting"], k=m["k"],
               teams=m["teams"], home=m["home"], away=m["away"], margin=m["margin"], total=m["total"],
               first_match=m["first"]["match"], first_winner=fw, first_margin=m["first"]["margin"],
               first_band=margin_band(m["first"]["margin"]), first_total=total_band(m["first"]["total"]),
               fw_p_first=p_of(m["first"]["match"], m["first"]), fw_p_now=p_of(m["match"], m),
               fw_won=None if fw is None or m["winner"] is None else float(m["winner"] == fw))
    out["fw_res"] = None if out["fw_won"] is None or out["fw_p_now"] is None else out["fw_won"] - out["fw_p_now"]
    line0, line1 = prematch_line(prod_tl, m["first"]["match"]), prematch_line(prod_tl, m["match"])
    out.update(line_first=line0, line_now=line1,
               first_miss=None if line0 is None or m["first"]["total"] is None else m["first"]["total"] - line0,
               miss=None if line1 is None or m["total"] is None else m["total"] - line1)
    out["first_miss_band"] = miss_band(out["first_miss"])
    if m["prev"] is not None:
        line_p = prematch_line(prod_tl, m["prev"]["match"])
        out["prev_miss"] = None if line_p is None or m["prev"]["total"] is None else m["prev"]["total"] - line_p
    # the first winner's margin and prod's, read as margins (MARGIN_SD), at the first meeting and now
    if fw is not None and out["fw_p_first"] is not None and out["fw_p_now"] is not None and m["margin"] is not None:
        signed = m["margin"] if m["winner"] == fw else -m["margin"]
        out["fw_margin_miss"] = signed - rematch_layer.implied_margin(out["fw_p_now"])
        out["fw_first_surprise"] = m["first"]["margin"] - rematch_layer.implied_margin(out["fw_p_first"])
    for mk in (markets.MONEYLINE, markets.SPREAD, markets.TOTAL):
        rs = [r for r in rows if r["market"] == mk]
        out[f"{mk}_stake"] = sum(r["stake"] for r in rs)
        out[f"{mk}_revenue"] = sum(r["revenue"] for r in rs)
        for side in ((WINNER, LOSER) if mk != markets.TOTAL else ("Over", "Under")):
            out[f"{mk}_on_{side}"] = sum(r["stake"] for r in rs if r["side"] == side)
    out["stake"] = sum(r["stake"] for r in rows)
    out["revenue"] = sum(r["revenue"] for r in rows)
    return out


# --- reporting -------------------------------------------------------------------------------

def _pc(x, w=6, fmt=".1f"):
    return f"{'-':>{w}s}" if x is None else f"{100 * x:{fmt}}".rjust(w)


def _f(x, w=7, fmt=",.0f"):
    return f"{'-':>{w}s}" if x is None else f"{x:{fmt}}".rjust(w)


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        v = key(r) if callable(key) else r.get(key)
        if v is not None:
            out[v].append(r)
    return out


def _ordered(groups, order):
    return [o for o in order if o in groups] + sorted(k for k in groups if k not in order)


def margin(rs):
    """The book's margin on the settled bets."""
    st = sum(r["stake"] for r in rs if r["settled"])
    return sum(r["revenue"] for r in rs if r["settled"]) / st if st else None


def meeting_lines(mrows, key, order, title):
    """Lines: matches by key -- how many, how many bet on, the money per match, the book's margin."""
    groups = _group(mrows, key)
    L = [f"\n  {title}",
         f"  {'':24s} {'matches':>7s} {'bet on':>6s} {'stake/m':>8s} {'ML/m':>7s} {'hcp/m':>7s} "
         f"{'tot/m':>7s} {'ML mgn':>7s} {'all mgn':>7s}"]
    for k in _ordered(groups, order):
        rs = groups[k]
        n = len(rs)
        ml, st = sum(r["moneyline_stake"] for r in rs), sum(r["stake"] for r in rs)
        L.append(f"  {str(k)[:24]:24s} {n:7,d} {_pc(sum(1 for r in rs if r['stake']) / n, 6, '.0f')} "
                 f"{_f(st / n, 8)} {_f(ml / n)} {_f(sum(r['spread_stake'] for r in rs) / n)} "
                 f"{_f(sum(r['total_stake'] for r in rs) / n)} "
                 f"{_pc(sum(r['moneyline_revenue'] for r in rs) / ml if ml else None, 7)} "
                 f"{_pc(sum(r['revenue'] for r in rs) / st if st else None, 7)}")
    return L


ML_HEAD = (f"  {'':22s} {'matches':>7s} {'p then':>6s} {'p now':>6s} {'won':>6s} {'resid':>6s} {'2se':>5s} "
           f"{'ML stake':>10s} {'on winner':>9s} {'mgn':>6s} {'res':>6s} {'mgn loser':>9s} {'res':>6s}")


def ml_lines(brows, mrows, key, order, title, min_matches=5):
    """Lines: rematches by key on the moneyline -- the first winner's price at the first meeting and
    now, how often he won again against it, and the money on him and on his opponent."""
    gm, gb = _group(mrows, key), _group([r for r in brows if r["market"] == markets.MONEYLINE], key)
    L = [f"\n  {title}", ML_HEAD]
    for k in _ordered(gm, order):
        ms, bs = gm[k], gb.get(k, [])
        if len(ms) < min_matches:
            continue
        stake = sum(r["stake"] for r in bs)
        on_w = [r for r in bs if r["side"] == WINNER]
        on_l = [r for r in bs if r["side"] == LOSER]
        p0 = [r["fw_p_first"] for r in ms if r["fw_p_first"] is not None]
        p1 = [r["fw_p_now"] for r in ms if r["fw_p_now"] is not None]
        wins = [r["fw_won"] for r in ms if r["fw_won"] is not None]
        _, res, se = mean_se([r for r in ms if r["fw_res"] is not None], "fw_res")
        rw = mean_se([r for r in on_w if r["res"] is not None], "res")[1]
        rl = mean_se([r for r in on_l if r["res"] is not None], "res")[1]
        L.append(f"  {str(k)[:22]:22s} {len(ms):7,d} {_pc(sum(p0) / len(p0) if p0 else None)} "
                 f"{_pc(sum(p1) / len(p1) if p1 else None)} {_pc(sum(wins) / len(wins) if wins else None)} "
                 f"{_pc(res, 6, '+.1f')} {_pc(None if se is None else 2 * se, 5)} {stake:10,.0f} "
                 f"{_pc(sum(r['stake'] for r in on_w) / stake if stake else None, 9)} {_pc(margin(on_w))} "
                 f"{_pc(rw, 6, '+.1f')} {_pc(margin(on_l), 9)} {_pc(rl, 6, '+.1f')}")
    return L


def other_lines(brows, key, order, title, market, sides, min_bets=MIN_BETS):
    """Lines: the handicap (on the first winner or his opponent) or the total (over or under) by key."""
    a, b = sides
    gb = _group([r for r in brows if r["market"] == market], key)
    L = [f"\n  {title}",
         f"  {'':22s} {'matches':>7s} {'bets':>6s} {'stake':>10s} {'on ' + a:>16s} {'mgn':>6s} {'mgn ' + b:>16s}"]
    for k in _ordered(gb, order):
        bs = gb[k]
        if len(bs) < min_bets:
            continue
        stake = sum(r["stake"] for r in bs)
        on_a = [r for r in bs if r["side"] == a]
        L.append(f"  {str(k)[:22]:22s} {len({r['match_code'] for r in bs}):7,d} {len(bs):6,d} {stake:10,.0f} "
                 f"{_pc(sum(r['stake'] for r in on_a) / stake if stake else None, 16)} {_pc(margin(on_a))} "
                 f"{_pc(margin([r for r in bs if r['side'] == b]), 16)}")
    return L


def _ms(xs):
    """(mean, 2se) of a list."""
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    return m, 2 * math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def totals_lines(re_m):
    """Lines: prod's total line at the first meeting and now, and each rematch's miss against it, by
    the first meeting's total and by its own miss."""
    L = []
    for key, order, title in (("first_total", TOTALS, "by the first meeting's total"),
                              ("first_miss_band", MISSES, "by the first meeting's total less prod's line then")):
        g = _group([r for r in re_m if r["line_now"] is not None and r["miss"] is not None], key)
        L += [f"\n  Rematch totals against prod's line, {title} (miss: total less the line; 2se by match)",
              f"  {'':22s} {'matches':>7s} {'line then':>9s} {'line now':>8s} {'moved':>6s} {'first miss':>10s} "
              f"{'miss':>6s} {'2se':>5s} {'over':>6s}"]
        for k in _ordered(g, order):
            rs = g[k]
            l0 = [r["line_first"] for r in rs if r["line_first"] is not None]
            l1 = [r["line_now"] for r in rs]
            fm = [r["first_miss"] for r in rs if r["first_miss"] is not None]
            moved = [r["line_now"] - r["line_first"] for r in rs if r["line_first"] is not None]
            m, se = _ms([r["miss"] for r in rs])
            L.append(f"  {str(k)[:22]:22s} {len(rs):7,d} {_f(sum(l0) / len(l0) if l0 else None, 9, '.1f')} "
                     f"{_f(sum(l1) / len(l1), 8, '.1f')} {_f(sum(moved) / len(moved) if moved else None, 6, '+.2f')} "
                     f"{_f(sum(fm) / len(fm) if fm else None, 10, '+.2f')} {_f(m, 6, '+.2f')} {_f(se, 5, '.2f')} "
                     f"{_pc(sum(1 for r in rs if r['miss'] > 0) / len(rs), 6)}")
    return L


def logistic_offset(xs, ys, offsets, iters=50):
    """(a, b, se a, se b) of logit P(y) = offset + a + b x, by Newton; None if it cannot be fitted."""
    a = b = 0.0
    det = haa = hbb = 0.0
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y, o in zip(xs, ys, offsets):
            p = 1 / (1 + math.exp(-(o + a + b * x)))
            w = p * (1 - p)
            ga += y - p
            gb += (y - p) * x
            haa += w
            hab += w * x
            hbb += w * x * x
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            return None
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a + da, b + db
        if abs(da) < 1e-9 and abs(db) < 1e-9:
            break
    return a, b, math.sqrt(hbb / det), math.sqrt(haa / det)


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _fit_row(label, rs, ykey, xkeys):
    """One line: y on a constant and the x's (ols), the slopes with their se and t."""
    rs = [r for r in rs if r.get(ykey) is not None and all(r.get(x) is not None for x in xkeys)]
    if len(rs) < 30:
        return f"  {label[:46]:46s} {len(rs):6,d}  (too few)"
    fit = ols([[1.0] + [r[x] for x in xkeys] for r in rs], [r[ykey] for r in rs])
    if fit is None:
        return f"  {label[:46]:46s} {len(rs):6,d}  (singular)"
    beta, se = fit
    parts = "  ".join(f"{x}: {b:+.3f} ±{s:.3f} (t {b / s if s else 0:+.1f})"
                      for x, b, s in zip(xkeys, beta[1:], se[1:]))
    return f"  {label[:46]:46s} {len(rs):6,d}  const {beta[0]:+.2f}  {parts}"


def fit_lines(re_m):
    """Lines: how far to move a rematch per point of the earlier meetings' misses."""
    second = [r for r in re_m if r["k"] == 2]
    later = [r for r in re_m if r["k"] > 2]
    L = ["\n  ==== How far to move a rematch: fits on the earlier meetings' misses ====",
         "  each slope: the points the rematch's miss moves per point of that earlier miss (what a layer "
         "should move it by); const: the rematch's own bias against prod",
         f"\n  {'':46s} {'n':>6s}",
         "  Total: the rematch's total less prod's line now"]
    L.append(_fit_row("2nd meetings, on the first's miss", second, "miss", ["first_miss"]))
    for t in (TEAMS[0], TEAMS[2]):
        L.append(_fit_row(f"2nd meetings, {t}", [r for r in second if r["teams"] == t], "miss", ["first_miss"]))
    L.append(_fit_row("3rd+, on the first's and the previous one's", later, "miss", ["first_miss", "prev_miss"]))
    L.append("  Margin: the first winner's margin less prod's (its moneyline read as a margin), on the "
             "first meeting's margin less prod's then")
    L.append(_fit_row("2nd meetings", second, "fw_margin_miss", ["fw_first_surprise"]))
    L.append(_fit_row("3rd+ meetings", later, "fw_margin_miss", ["fw_first_surprise"]))
    L.append("  Moneyline: the first winner winning again, logistic with prod's price now as the offset, "
             "on the first meeting's margin surprise per 10 points")
    for label, rs in (("2nd meetings", second), ("3rd+ meetings", later), ("every rematch", re_m)):
        rs = [r for r in rs if r.get("fw_won") is not None and r.get("fw_first_surprise") is not None
              and r.get("fw_p_now") is not None]
        fit = logistic_offset([r["fw_first_surprise"] / 10 for r in rs], [r["fw_won"] for r in rs],
                              [_logit(r["fw_p_now"]) for r in rs]) if len(rs) >= 30 else None
        if fit is None:
            L.append(f"  {label:46s} {len(rs):6,d}  (too few, or no fit)")
            continue
        a, b, sa, sb = fit
        L.append(f"  {label:46s} {len(rs):6,d}  const {a:+.3f} ±{sa:.3f}  per 10 points {b:+.3f} ±{sb:.3f} "
                 f"(t {b / sb if sb else 0:+.1f}); at evens, 10 points of first surprise -> "
                 f"{100 * (1 / (1 + math.exp(-(a + b))) - 0.5):+.1f} points")
    return L


def layer_lines(scored, layer, min_days):
    """Lines: the layer on prod's prices, walk-forward, against prod alone."""
    active = [r for r in scored if r.get("active")]
    L = ["\n  ==== The rematch layer on prod's prices (rematch_layer.py, refitted each day on the days "
         f"before it; the first {min_days} days are its burn-in) ====", "  the last fit:"]
    L += layer.describe()
    L.append(f"\n  {'':26s} {'matches':>7s} {'moved':>7s} {'log loss':>9s} {'+layer':>9s} {'diff':>8s} {'t':>5s} "
             f"{'Brier':>7s} {'+layer':>7s} {'tot RMSE':>8s} {'+layer':>7s} {'diff':>7s} {'t':>5s}")
    groups = [("every match", active), ("rematches", [r for r in active if r["n_earlier"]]),
              ("2nd meetings", [r for r in active if r["n_earlier"] == 1]),
              ("3rd+ meetings", [r for r in active if r["n_earlier"] > 1])]
    for t in rematch_layer.RELATIONS:
        groups.append((f"rematches, {t}", [r for r in active if r["n_earlier"] and r.get("teams_label") == t]))
    for label, rs in groups:
        if not rs:
            continue
        a, b = rematch_layer.score(rs), rematch_layer.score(rs, "margin_adj", "total_adj")
        dl, tl = rematch_layer.paired(a["ll"], b["ll"])
        dt_, tt = rematch_layer.paired(a["te"], b["te"])
        moved = sum(1 for r in rs if abs(r["move_margin"]) > 1e-9 or abs(r["move_total"]) > 1e-9)
        L.append(f"  {label[:26]:26s} {len(rs):7,d} {moved:7,d} {_f(a['log_loss'], 9, '.4f')} "
                 f"{_f(b['log_loss'], 9, '.4f')} {_f(dl, 8, '+.4f')} {_f(tl, 5, '+.1f')} "
                 f"{_f(a['brier'], 7, '.4f')} {_f(b['brier'], 7, '.4f')} {_f(a['total_rmse'], 8, '.2f')} "
                 f"{_f(b['total_rmse'], 7, '.2f')} {_f(dt_, 7, '+.2f')} {_f(tt, 5, '+.1f')}")
    L.append("  diff: +layer less prod alone (negative is better), t on it match by match; the total's "
             "diff is in squared points")
    return L


def report(brows, mrows):
    """Lines of text."""
    re_m = [r for r in mrows if r["k"] > 1]
    re_b = [r for r in brows if r["k"] > 1 and r["side"] != DRAWN]
    first = lambda r: r["first_band"]
    meet_teams = lambda r: f"{r['meeting']} / {r['teams']}" if r.get("teams") else None
    mt_order = [f"{m} / {t}" for m in MEETINGS[1:] for t in TEAMS]
    L = [f"  {len(mrows):,} settled matches in the window, {len(re_m):,} rematches the same day "
         f"({sum(1 for r in re_m if r['teams'] == TEAMS[0]):,} with the same teams); "
         f"{len(brows):,} bets, {sum(r['stake'] for r in brows):,.0f} stake",
         f"  a day: the gamers' session (no gap over {config.SESSION_BREAK_MINUTES} minutes); first winner: "
         "the day's first meeting's winner (rematches after a drawn first meeting are left out of the sides)",
         "  p then / p now: prod's pre-match moneyline for the first winner at the first meeting and at this "
         "one; won: how often he won this one; resid: won less p now (2se by match)",
         "  on winner: the share of the stake on the first winner; mgn: the book's margin on each side; res: "
         "won less prod's p at the bet, per bet"]
    L.append("\n  ==== How much is bet on a meeting, by its place in the day ====")
    L += meeting_lines(mrows, "meeting", MEETINGS, "Every match by its meeting of the day")
    L += meeting_lines(re_m, "teams", TEAMS, "Rematches by the teams against the day's first meeting")
    L += meeting_lines(re_m, meet_teams, mt_order, "Rematches by meeting and teams")
    L += meeting_lines(re_m, first, MARGINS, "Rematches by the first meeting's margin of victory")
    L.append("\n  ==== Where the moneyline money goes in a rematch, against the day's first result ====")
    L += ml_lines(re_b, re_m, first, MARGINS[1:], "Every rematch, by the first meeting's margin of victory")
    for t in TEAMS:
        L += ml_lines([r for r in re_b if r["teams"] == t], [r for r in re_m if r["teams"] == t], first,
                      MARGINS[1:], f"{t.capitalize()} (against the first meeting), by its margin")
    L += ml_lines(re_b, re_m, "meeting", MEETINGS[1:], "Every rematch, by its meeting of the day")
    L += ml_lines(re_b, re_m, meet_teams, mt_order, "By meeting and teams")
    L += ml_lines([r for r in re_b if not r["in_play"]], re_m, first, MARGINS[1:],
                  "Pre-match bets only, by the first margin (the match columns are every rematch)")
    L += ml_lines([r for r in re_b if r["in_play"]], re_m, first, MARGINS[1:],
                  "In-play bets only, by the first margin (the match columns are every rematch)")
    for grp in ("Restricted", "VIP"):
        L += ml_lines([r for r in re_b if r["group"] == grp], re_m, first, MARGINS[1:],
                      f"{grp} bets only, by the first margin (the match columns are every rematch)")
    L.append("\n  ==== The handicap and the total in a rematch ====")
    L += other_lines(re_b, first, MARGINS[1:], "Handicap: the money on the first winner, by the first margin",
                     markets.SPREAD, (WINNER, LOSER))
    L += other_lines(re_b, "first_total", TOTALS, "Total: over and under, by the first meeting's total",
                     markets.TOTAL, ("Over", "Under"))
    L += other_lines(re_b, first, MARGINS[1:], "Total: over and under, by the first meeting's margin",
                     markets.TOTAL, ("Over", "Under"))
    L += totals_lines(re_m)
    L += fit_lines(re_m)
    big = sorted(re_m, key=lambda r: -r["moneyline_stake"])[:25]
    L.append("\n  The 25 rematches with the most moneyline stake (first: the day's first winner and his "
             "margin; fw p: prod's for him at the first meeting -> now; won: did he win again)")
    L.append(f"  {'start (UTC)':16s} {'pair':26s} {'mtg':4s} {'teams':13s} {'first':20s} {'fw p':9s} "
             f"{'won':>3s} {'ML stake':>9s} {'on fw':>6s} {'ML rev':>8s}")
    for r in big:
        fw = f"{r['first_winner']} by {r['first_margin']}" if r["first_winner"] else "draw"
        L.append(f"  {str(r['start'])[:16]:16s} {r['pair'][:26]:26s} {r['meeting']:4s} {str(r['teams'])[:13]:13s} "
                 f"{fw[:20]:20s} {_pc(r['fw_p_first'], 3, '.0f')}->{_pc(r['fw_p_now'], 3, '.0f')}  "
                 f"{('-' if r['fw_won'] is None else 'Y' if r['fw_won'] else 'N'):>3s} {r['moneyline_stake']:9,.0f} "
                 f"{_pc(r[f'moneyline_on_{WINNER}'] / r['moneyline_stake'] if r['moneyline_stake'] else None, 6, '.0f')} "
                 f"{r['moneyline_revenue']:8,.0f}")
    return L


# --- run -------------------------------------------------------------------------------------

LAYER_MIN_DAYS = 7


def prod_layer(meets, codes, prod_tl, min_days=LAYER_MIN_DAYS):
    """The rematch layer on prod's pre-match prices (its moneyline read as a margin, its total line),
    walk-forward over these matches: (rows, the last layer)."""
    items = []
    for code in codes:
        m = meets.get(code)
        p1, line = prematch_p1(prod_tl, code), prematch_line(prod_tl, code)
        if m is None or p1 is None or line is None:
            continue
        s1, s2 = m.get("home_score"), m.get("away_score")
        it = rematch_layer.item(code, m["start"], m["day_key"], m["home"], m["away"], m["team"][m["home"]],
                                m["team"][m["away"]], rematch_layer.implied_margin(p1), line,
                                None if s1 is None or s2 is None else s1 - s2,
                                None if s1 is None or s2 is None else s1 + s2)
        it["teams_label"] = m["teams"] if m["teams"] != TEAMS[1] else TEAMS[2]
        items.append(it)
    rows, layer = rematch_layer.walk_forward(items, min_days)
    first = min((r["start"] for r in rows), default=None)
    for r in rows:
        r["active"] = first is not None and (r["start"].date() - first.date()).days >= min_days
    return rows, layer

def run(cur, out_dir):
    """Every settled match of the window with its place in the day's meetings, prod's pre-match
    prices and every bet on it; print the report, write the CSV."""
    history = snowflake_io.fetch_history(cur)
    meets = day_meetings(history)
    since = _start(config.CUTOFF_START)
    until = _start(config.CUTOFF_END) if config.CUTOFF_END else None
    window = sorted((m for m in meets.values() if m["start"] >= since and (until is None or m["start"] < until)),
                    key=lambda m: m["start"])
    in_window = {m["match"] for m in window}
    codes = sorted(in_window | {x for m in window for x in m["earlier"]})
    print(f"  {len(window):,} settled matches in the window, {sum(m['k'] > 1 for m in window):,} rematches the "
          f"same day; prod's prices for {len(codes):,} (with the first meetings)", flush=True)
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
    lags = bets.fit_lags(every, prod_tl)
    joined = bets.join(every, lags, bets.fit_line_signs(every, prod_tl, lags), prod_tl, {}, None, {})
    brows = [bet_row(b, r, meets[b.match_code]) for b, r in zip(every, joined)]
    by = defaultdict(list)
    for r in brows:
        by[r["match_code"]].append(r)
    mrows = [match_row(m, by.get(m["match"], []), pre_tl) for m in window]
    scored, layer = prod_layer(meets, codes, pre_tl)
    scored = [r for r in scored if r["match"] in in_window]
    lines = report(brows, mrows) + layer_lines(scored, layer, LAYER_MIN_DAYS)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    bets.write_csv(os.path.join(out_dir, "bets_rematches.csv"), mrows,
                   list(dict.fromkeys(k for r in mrows for k in r)))
    path = os.path.join(out_dir, "bets_rematches.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
