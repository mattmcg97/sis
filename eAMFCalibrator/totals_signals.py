"""What separates the totals that go under: how each match was being played at the end of Q1 and
at half time, against prod's live total there and against where the totals money went next.

  play     every play off SCOUTING_FULL: its real start and end (FILE_TIME), the game clock (240
           seconds a quarter), the down, distance and field position before the snap, and what it
           was (scrimmage, kick-off, conversion; a punt, field goal or touchdown inside it)
  segment  Q1, and the first half: plays, how long the gamers take between plays (the real
           seconds from one PLAY_OVER to the next snap), how long a play runs, the game clock
           each play burns, yards per play, drives ended by punts, field goals, touchdowns and
           turnovers on downs, timeouts, points
  pre-match  prod's closing line (the last before Q1 starts) against the final, with the
           match's place in the gamers' sessions and their recent form and long-run level
           (walk-forward off the match history), and the pre-match totals money
  checkpoint  the message the next quarter starts on (Q2 for Q1, Q3 for the half): prod's live
           total line there, the points on the board, and what the rest of the match produced.
           over = final total less prod's line, so a feature prod already prices sits flat at 0
  money    every totals bet struck in the next segment (Q2 after Q1, the second half after the
           half), by CUSTOMER_TEMPERATURE: the share of the stake on the under and the book's
           margin on each side

Each feature is cut into quintiles over the matches; the report ranks the features by how far
over runs across them (prod missing it), and by how far the under share of the stake moves with
them (the money reading it).
"""

import csv
import datetime as dt
import math
import os
import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field

from . import bet_checks, bet_moments, bets, config, scouting, snowflake_io

SEGMENTS = (("q1", (1,), "SECOND_QUARTER_STARTED"), ("h1", (1, 2), "THIRD_QUARTER_STARTED"))
NEXT_START = {"q1": "THIRD_QUARTER_STARTED", "h1": None}
PRESNAP_MAX = 120.0          # a gap between plays longer than this is a break, not a huddle
GROUPS = ("All", "Restricted", "VIP", "Standard", "none")
PRE_FEATURES = (
    ("session_match", "match of the session (the later gamer's)"),
    ("last_both", "both gamers on their last match (1)"),
    ("form", "both gamers' recent form (pts vs league)"),
    ("longrun", "both gamers' long-run level"),
    ("form_vs_longrun", "recent form less the long-run level"),
    ("line", "prod's closing line"),
)
FORM_WEIGHT = 0.05           # EWMA weight: about the last 20 matches
FEATURES = (
    ("plays", "scrimmage plays"),
    ("presnap_s", "real seconds between plays (median)"),
    ("play_s", "real seconds a play runs (median)"),
    ("clock_per_play", "game-clock seconds per play (median)"),
    ("real_minutes", "real minutes the segment took"),
    ("yards_per_play", "yards per play"),
    ("drives", "drives ended (punt, FG, TD, downs)"),
    ("punts", "punts"),
    ("tds", "touchdowns"),
    ("fgs", "field goals"),
    ("timeouts", "timeouts called"),
    ("points", "points on the board"),
    ("needed", "prod's line less the points on the board"),
)


def _base(kind):
    return bet_moments._base(scouting._text(kind))


def _num(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class Play:
    period: int
    start_msg: int
    start_t: object
    start_clock: float
    team: object = None
    yard: float = None
    over_t: object = None
    kind: str = "scrimmage"
    events: set = field(default_factory=set)


def plays(rows):
    """[Play] of one match off its scouting rows (fetch_scouting's columns), in message order. A
    play is a kick-off or a conversion by its own messages, or by what came before it (the play
    after a touchdown is its conversion; the one after a conversion, field goal or safety is the
    kick-off)."""
    out, period, open_, pending = [], None, None, None
    for r in sorted(rows, key=lambda r: int(r[1])):
        status, kind = scouting._text(r[3]), _base(r[4])
        if status in scouting.PERIOD_START:
            period = scouting.PERIOD_START[status]
        if kind == "PLAY_STARTED":
            open_ = Play(period, int(r[1]), bet_moments._time(r[9]), _num(r[2]),
                         scouting._text(r[5]), _num(r[8]))
            if pending:
                open_.kind = pending
            continue
        if open_ is None or not kind:
            continue
        open_.events.add(kind)
        if kind.startswith(("KICKOFF", "ONSIDE_KICK")):
            open_.kind = "kickoff"
        elif kind.startswith(("EXTRA_POINT", "TWO_POINT")):
            open_.kind = "conversion"
        if kind == "PLAY_OVER":
            open_.over_t = bet_moments._time(r[9])
            ev = open_.events
            if "TOUCHDOWN" in ev:
                pending = "conversion"
            elif open_.kind == "conversion" or ev & {"FIELD_GOAL_GOOD", "SAFETY_AWARDED"}:
                pending = "kickoff"
            elif open_.kind == "kickoff":
                pending = None
            elif pending == "conversion" and open_.kind == "scrimmage":
                pending = None
            out.append(open_)
            open_ = None
    return out


def timeouts(rows):
    """{period: timeouts called}: TIMEOUT_CALLED messages, inside a play or between plays."""
    out, period = defaultdict(int), None
    for r in sorted(rows, key=lambda r: int(r[1])):
        status = scouting._text(r[3])
        if status in scouting.PERIOD_START:
            period = scouting.PERIOD_START[status]
        if (_base(r[4]) or "").startswith("TIMEOUT_CALLED"):
            out[period] += 1
    return out


def segment_features(ps, periods, called=None):
    """The features of one segment (the plays of these periods; `called`: timeouts() of the
    match)."""
    seg = [p for p in ps if p.period in periods]
    scrim = [p for p in seg if p.kind == "scrimmage"]
    presnap, run, clock, gains = [], [], [], []
    for a, b in zip(scrim, scrim[1:]):
        if a.period != b.period:
            continue
        if a.over_t and b.start_t:
            gap = (b.start_t - a.over_t).total_seconds()
            if 0 <= gap <= PRESNAP_MAX:
                presnap.append(gap)
        if a.start_clock is not None and b.start_clock is not None and a.start_clock >= b.start_clock:
            clock.append(a.start_clock - b.start_clock)
        if a.team and a.team == b.team and a.yard is not None and b.yard is not None \
                and not (a.events & {"PUNT", "TOUCHDOWN"}):
            gains.append(b.yard - a.yard)
    for p in scrim:
        if p.start_t and p.over_t:
            d = (p.over_t - p.start_t).total_seconds()
            if 0 <= d <= PRESNAP_MAX:
                run.append(d)
    times = [t for p in seg for t in (p.start_t, p.over_t) if t]
    count = lambda *kinds: sum(1 for p in seg if p.events & set(kinds))
    out = dict(plays=len(scrim) or None,
               presnap_s=statistics.median(presnap) if presnap else None,
               play_s=statistics.median(run) if run else None,
               clock_per_play=statistics.median(clock) if clock else None,
               real_minutes=(max(times) - min(times)).total_seconds() / 60 if len(times) > 1 else None,
               yards_per_play=sum(gains) / len(gains) if gains else None,
               punts=count("PUNT"), tds=count("TOUCHDOWN"), fgs=count("FIELD_GOAL_GOOD"),
               timeouts=sum((called or {}).get(q, 0) for q in periods))
    out["drives"] = out["punts"] + out["tds"] + count("FIELD_GOAL_GOOD", "FIELD_GOAL_MISSED") \
        + count("TURNOVER_ON_DOWNS")
    return out


def checkpoints(rows):
    """{status: its message} for the quarter starts of one match."""
    out = {}
    for r in sorted(rows, key=lambda r: int(r[1])):
        s = scouting._text(r[3])
        if s in scouting.PERIOD_START and s not in out:
            out[s] = int(r[1])
    return out


def group_of(b):
    t = str(b.extra.get(config.BET_VIP_COLUMN) or "").strip().lower()
    return next((g for g in GROUPS[1:4] if g.lower() == t), "none")


def money(match_bets, t0, t1):
    """{group: [bets, stake, under stake, under revenue, over revenue]} of the in-play totals bets
    struck in [t0, t1) (t1 None: to the end)."""
    out = {g: [0, 0.0, 0.0, 0.0, 0.0] for g in GROUPS}
    if t0 is None:
        return out
    for b in match_bets:
        if b.time is None or b.time < t0 or (t1 is not None and b.time >= t1):
            continue
        under = b.feed_market == 55
        for g in ("All", group_of(b)):
            c = out[g]
            c[0] += 1
            c[1] += b.stake
            c[2] += b.stake if under else 0.0
            c[3 if under else 4] += b.revenue
    return out


def form_by_match(history, weight=FORM_WEIGHT, shrink=20):
    """match -> (both gamers' recent form, both gamers' long-run level), each in points of total
    against the league, off the settled matches before it (walk-forward: nothing after the match
    is read). Form is an EWMA of the gamer's half of each total's residual; the long-run level its
    mean, shrunk towards 0 by `shrink` matches."""
    games = []
    for r in history:
        try:
            games.append((str(r["SCHEDULED_START_TIME_UTC"])[:19], r,
                          int(r["PLAYER_1_FINAL_SCORE"]) + int(r["PLAYER_2_FINAL_SCORE"])))
        except (KeyError, TypeError, ValueError):
            continue
    games.sort(key=lambda g: g[0])
    ew, lt, total, n, out = {}, defaultdict(lambda: [0.0, 0]), 0.0, 0, {}
    for _, r, tot in games:
        gs = (r["PLAYER_1_HANDLE"], r["PLAYER_2_HANDLE"])
        out[r["MATCH_CODE"]] = (sum(ew.get(g, 0.0) for g in gs),
                                sum(lt[g][0] / (lt[g][1] + shrink) for g in gs if lt[g][1] + shrink))
        res = tot - (total / n if n else tot)
        for g in gs:
            ew[g] = (1 - weight) * ew.get(g, 0.0) + weight * res / 2
            lt[g][0] += res / 2
            lt[g][1] += 1
        total += tot
        n += 1
    return out


def pre_features(match, info, pos, form):
    """The pre-match features of one match: its place in the gamers' sessions and their form."""
    out = {}
    i = info.get(match)
    if i:
        p = [pos.get((i.get(f"PLAYER_{s}_HANDLE"), match)) for s in "12"]
        if all(p):
            out["session_match"] = max(x[0] for x in p)
            out["last_both"] = int(all(x[1] == 1 for x in p))
    if match in form:
        f, lt = form[match]
        out.update(form=f, longrun=lt, form_vs_longrun=f - lt)
    return out


def match_row(match, rows, tl, prod_tl, scores, final, match_bets, pre_bets=(), pre=None):
    """One match: prod's closing pre-match line with the pre-match features and money, then every
    feature of each in-play segment, prod's line and the board at each checkpoint, the rest of
    the match against it, and the totals money in the next segment."""
    ps = plays(rows)
    called = timeouts(rows)
    cps = checkpoints(rows)
    out = dict(match_code=match, final_total=None if final is None else final[0] + final[1])

    def time_at(msg):
        if tl is None or msg is None:
            return None
        i = bisect_right(tl.msgs, msg) - 1
        if i < 0 or tl.times[i] == -math.inf:
            return None
        return dt.datetime.fromtimestamp(tl.times[i])     # bet_moments keeps naive .timestamp()

    start = time_at(cps.get("FIRST_QUARTER_STARTED"))
    q = bets.price_at_time(prod_tl, match, 54, start - dt.timedelta(milliseconds=1)) if start else None
    f = dict(pre or {})
    f["line"] = q[2] if q else None
    f["over"] = (None if f["line"] is None or out["final_total"] is None
                 else out["final_total"] - f["line"])
    m = money(pre_bets, dt.datetime.min, None)
    for g, (n, stake, us, ur, orv) in m.items():
        f[f"{g}_bets"], f[f"{g}_stake"], f[f"{g}_under_stake"] = n, stake, us
        f[f"{g}_under_rev"], f[f"{g}_over_rev"] = ur, orv
    out.update({f"pre_{k}": v for k, v in f.items()})
    for name, periods, status in SEGMENTS:
        f = segment_features(ps, periods, called)
        msg = cps.get(status)
        t = time_at(msg)
        q = bets.price_at_time(prod_tl, match, 54, t) if t else None
        line = q[2] if q else None
        board = sum(scores.score_at(match, msg)) if msg is not None else None
        f["points"] = board
        f["line"] = line
        f["needed"] = None if line is None or board is None else line - board
        f["over"] = (None if line is None or out["final_total"] is None
                     else out["final_total"] - line)
        nxt = NEXT_START[name]
        m = money(match_bets, t, time_at(cps.get(nxt)) if nxt else None)
        for g, (n, stake, us, ur, orv) in m.items():
            f[f"{g}_bets"], f[f"{g}_stake"], f[f"{g}_under_stake"] = n, stake, us
            f[f"{g}_under_rev"], f[f"{g}_over_rev"] = ur, orv
        out.update({f"{name}_{k}": v for k, v in f.items()})
    return out


# --- the report ------------------------------------------------------------------------------

def quintiles(xs, k=5):
    s = sorted(xs)
    return [s[min(len(s) - 1, int(len(s) * i / k))] for i in range(1, k)]


def edges_for(xs):
    """Quintile edges, or one bin per value where there are six values or fewer (a count)."""
    values = sorted(set(xs))
    return values[1:] if len(values) <= 6 else quintiles(xs)


def bin_of(x, edges):
    return bisect_right(edges, x)


def _mean_se(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None
    return m, 2 * statistics.pstdev(xs) / math.sqrt(len(xs))


def _corr(pairs):
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sxy / (sx * sy) if sx and sy else None


def _share(rs, seg, g):
    stake = sum(r[f"{seg}_{g}_stake"] for r in rs)
    return 100 * sum(r[f"{seg}_{g}_under_stake"] for r in rs) / stake if stake else None


def _under_margin(rs, seg, g="All"):
    us = sum(r[f"{seg}_{g}_under_stake"] for r in rs)
    return 100 * sum(r[f"{seg}_{g}_under_rev"] for r in rs) / us if us else None


def _f(x, fmt="+.2f", w=7):
    return f"{'-':>{w}s}" if x is None else f"{x:{fmt}}".rjust(w)


def features_of(seg):
    return PRE_FEATURES if seg == "pre" else FEATURES


def ranking(rows, seg):
    """[(feature, title, matches, corr with over, over in the bottom quintile, in the top,
    corr with the under share of the money, under share in the bottom and top quintile)],
    sorted by how far over moves across the quintiles."""
    out = []
    for key, title in features_of(seg):
        col = f"{seg}_{key}"
        rs = [r for r in rows if r.get(col) is not None and r.get(f"{seg}_over") is not None]
        if len(rs) < 25 or len({r[col] for r in rs}) < 2:
            continue
        edges = edges_for([r[col] for r in rs])
        filled = sorted({bin_of(r[col], edges) for r in rs})
        lo = [r for r in rs if bin_of(r[col], edges) == filled[0]]
        hi = [r for r in rs if bin_of(r[col], edges) == filled[-1]]
        c_over = _corr([(r[col], r[f"{seg}_over"]) for r in rs])
        shared = [(r[col], 100 * r[f"{seg}_All_under_stake"] / r[f"{seg}_All_stake"])
                  for r in rs if r[f"{seg}_All_stake"]]
        out.append((key, title, len(rs), c_over, _mean_se([r[f"{seg}_over"] for r in lo])[0],
                    _mean_se([r[f"{seg}_over"] for r in hi])[0], _corr(shared),
                    _share(lo, seg, "All"), _share(hi, seg, "All")))
    return sorted(out, key=lambda x: -abs((x[5] or 0) - (x[4] or 0)))


def report(rows):
    """Lines of text: per checkpoint, the features ranked, then each one by quintile."""
    L = []
    for seg in ("pre",) + tuple(s[0] for s in SEGMENTS):
        label = {"pre": "pre-match, prod's closing line (money: pre-match bets)",
                 "q1": "end of Q1 (money: bets struck in Q2)",
                 "h1": "half time (money: bets struck in the second half)"}[seg]
        rs = [r for r in rows if r.get(f"{seg}_over") is not None]
        if not rs:
            L.append(f"\n  {label}: no match with prod's line at the checkpoint")
            continue
        m, se = _mean_se([r[f"{seg}_over"] for r in rs])
        L.append(f"\n  ==== {label}: {len(rs):,} matches; over = final total less prod's line "
                 f"there, mean {m:+.2f} +-{se:.2f} ====")
        L.append(f"  features ranked by how far over moves from the lowest bin to the highest "
                 "(prod missing it), with the under share of the next segment's totals stake "
                 "(the money reading it); bins are quintiles, or one per value for a count")
        L.append(f"  {'feature':40s} {'matches':>7s} {'corr':>6s} {'over lo':>8s} {'over hi':>8s} "
                 f"{'corr$':>6s} {'under lo':>9s} {'under hi':>9s}")
        for key, title, n, c, lo, hi, cm, ulo, uhi in ranking(rs, seg):
            L.append(f"  {title[:40]:40s} {n:7,d} {_f(c, '+.2f', 6)} {_f(lo, '+.2f', 8)} "
                     f"{_f(hi, '+.2f', 8)} {_f(cm, '+.2f', 6)} {_f(ulo, '.1f', 8)}% {_f(uhi, '.1f', 8)}%")
        for key, title in features_of(seg):
            col = f"{seg}_{key}"
            have = [r for r in rs if r.get(col) is not None]
            if len(have) < 25:
                continue
            edges = edges_for([r[col] for r in have])
            L.append(f"\n  {title} at the {label.split(' (')[0]}")
            L.append(f"  {'bin':5s} {'range':>15s} {'matches':>7s} {'over':>7s} {'+-2se':>6s} "
                     f"{'needed':>7s} {'stake/m':>8s} "
                     + " ".join(f"{('under ' + g)[:12]:>12s}" for g in GROUPS[:4])
                     + f" {'mgn under':>9s}")
            filled = [q for q in range(len(edges) + 1) if any(bin_of(r[col], edges) == q for r in have)]
            for k, q in enumerate(filled):
                b = [r for r in have if bin_of(r[col], edges) == q]
                vals = [r[col] for r in b]
                om, ose = _mean_se([r[f"{seg}_over"] for r in b])
                nd = _mean_se([r[f"{seg}_needed"] for r in b if r.get(f"{seg}_needed") is not None])[0]
                stake = sum(r[f"{seg}_All_stake"] for r in b) / len(b)
                L.append(f"  {k + 1:<5d} {min(vals):7.1f}-{max(vals):<7.1f} {len(b):7,d} {_f(om)} "
                         f"{_f(ose, '.2f', 6)} {_f(nd, '.1f', 7)} {stake:8,.0f} "
                         + " ".join(f"{_f(_share(b, seg, g), '.1f', 11)}%" for g in GROUPS[:4])
                         + f" {_f(_under_margin(b, seg), '+.1f', 8)}%")
    return L


# --- run -------------------------------------------------------------------------------------

def run(cur, out_dir):
    """Every settled match of the window: its plays off SCOUTING_FULL, prod's total at the end of
    Q1 and at the half, the final, and the totals bets; print the report, write the CSV."""
    matches = snowflake_io.match_universe(cur, config.STREAMS["prod"])
    print(f"  {len(matches):,} settled matches in the window", flush=True)
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    by_match, pre_by = defaultdict(list), defaultdict(list)
    for b in bets.to_bets(cols, raw):
        if b.feed_market in (54, 55):
            (by_match if b.in_play else pre_by)[b.match_code].append(b)
    print(f"  {sum(map(len, by_match.values())):,} in-play and {sum(map(len, pre_by.values())):,} "
          "pre-match totals bets", flush=True)
    from .bet_totals import sessions
    history = snowflake_io.fetch_history(cur)
    info = {r["MATCH_CODE"]: r for r in history}
    pos, form = sessions(history), form_by_match(history)
    table = snowflake_io.scouting_table(cur)
    out = []
    chunk = config.MATCH_CHUNK_SIZE
    for start in range(0, len(matches), chunk):
        batch = matches[start:start + chunk]
        prod_rows = snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        finals = snowflake_io.fetch_final_scores(cur, batch)
        scores = bet_checks.Checks({}, {}, bet_checks.score_index(snowflake_io.fetch_scores(cur, batch)))
        srows = scouting.fetch_scouting(cur, table, batch, windowed=False)
        tls = bet_moments.build(srows, bet_checks.message_times(prod_rows))
        prod_tl = bets.timeline(prod_rows)
        by = defaultdict(list)
        for r in srows:
            by[r[0]].append(r)
        for m in batch:
            if by.get(m):
                out.append(match_row(m, by[m], tls.get(m), prod_tl, scores, finals.get(m),
                                     by_match.get(m, []), pre_by.get(m, []),
                                     pre_features(m, info, pos, form)))
        print(f"  {min(start + chunk, len(matches)):,} of {len(matches):,} matches read", flush=True)
    lines = report(out)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "totals_signals.csv")
    fields = list(dict.fromkeys(k for r in out for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(out_dir, "totals_signals.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
