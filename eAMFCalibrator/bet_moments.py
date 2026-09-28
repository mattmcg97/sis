"""Where in the game the bets are lost: every in-play bet tagged with what the scouting feed was
doing when it was struck, and the book's margin bucketed by each tag.

  moment      the feed's state at bet time: a play live, between plays, a kick-off or conversion
              to come, a timeout, a quarter break, a score in with the play not yet over
  suspended   the feed's own BET_SUSPEND .. BET_UNSUSPEND was on at bet time
  moved       what the feed did between the message the bet's price came from and bet time: a
              score, a play started, a play over, nothing (the price was current)
  next score  seconds from the bet to the board's next move
  price age   seconds from the price's message to the bet

Bet time is read against the feed's own clock: each message's FILE_TIME, moved onto prod's
publishing clock by the match's median gap between the two (prod publishes only on some
messages; the feed has them all). Expected margin is what the book keeps if prod's probability is
right (1 - odds x prod's probability); a bucket losing well past it is where bettors know more
than the price.
"""

import math
import statistics
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field

from .bet_checks import OVER_STATUS
from .scouting import PERIOD_START, _text

BEFORE, OVER, INTERRUPTION, BREAK, TIMEOUT = (
    "before the start", "match over", "interruption", "quarter break", "timeout")
LIVE, KICK_LIVE, CONV_LIVE, SCORED_LIVE = (
    "play live", "kick-off live", "conversion live", "score in, play not over")
CONV_NEXT, KICK_NEXT, BETWEEN = "conversion to come", "kick-off to come", "between plays"
MOMENTS = (BETWEEN, LIVE, SCORED_LIVE, CONV_NEXT, CONV_LIVE, KICK_NEXT, KICK_LIVE, TIMEOUT, BREAK,
           INTERRUPTION, BEFORE, OVER)
SCORES = ("TOUCHDOWN", "FIELD_GOAL_GOOD", "EXTRA_POINT_GOOD", "TWO_POINT_CONVERSION_SUCCESSFUL",
          "SAFETY")
MOVES = (("score", SCORES), ("kick-off", ("KICKOFF",)), ("play started", ("PLAY_STARTED",)),
         ("play over", ("PLAY_OVER",)), ("timeout", ("TIMEOUT_CALLED", "TIMEOUT_OVER")),
         ("possession", ("POSSESSION",)))
AGE_BANDS = (0, 2, 5, 10, 20, 60)
NEXT_BANDS = (10, 30, 60, 120)
MIN_BETS = 100


def _base(kind):
    """An in-play message without its team: TOUCHDOWN_TEAM_A -> TOUCHDOWN."""
    if not kind:
        return None
    for suffix in ("_TEAM_A", "_TEAM_B"):
        if kind.endswith(suffix):
            return kind[:-len(suffix)]
    return kind


def _num(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _time(v):
    """A naive datetime off a timestamp or its text."""
    import datetime as dt
    if v is None or v == "":
        return None
    if isinstance(v, dt.datetime):
        return v.replace(tzinfo=None)
    try:
        return dt.datetime.fromisoformat(str(v).replace("Z", "")).replace(tzinfo=None)
    except ValueError:
        return None


@dataclass
class Timeline:
    """One match's feed, message by message: its state after each message and its time on prod's
    clock."""
    msgs: list
    times: list
    moment: list
    suspended: list
    period: list
    clock: list
    down: list
    events: list = field(default_factory=list)


def _states(rows):
    """(moment, suspended, period, clock, down) after each of a match's scouting rows, in order."""
    started = ended = qbreak = interrupt = timeout = live = susp = scored = False
    pending = play = None
    period = down = None
    out = []
    for r in rows:
        status, kind = _text(r[3]), _base(_text(r[4]))
        if status in PERIOD_START:
            started, qbreak, period = True, False, PERIOD_START[status]
        elif status and OVER_STATUS.match(status):
            ended = True
        elif status and status.endswith("_ENDED"):
            qbreak = True
        elif status == "START_INTERRUPTION":
            interrupt = True
        elif status == "END_INTERRUPTION":
            interrupt = False
        elif status in ("BET_SUSPEND", "PERMANENT_BET_SUSPEND"):
            susp = True
        elif status == "BET_UNSUSPEND":
            susp = False
        if kind == "PLAY_STARTED":
            live, timeout, scored = True, False, False
            play = pending or "scrimmage"
        elif kind == "PLAY_OVER":
            live = False
            if play == "kickoff":
                pending = None
            play = None
        elif kind == "TOUCHDOWN":
            scored, pending = True, "conversion"
        elif kind and kind.startswith(("EXTRA_POINT", "TWO_POINT")):
            scored = scored or kind in SCORES
            pending = "kickoff"
        elif kind in ("FIELD_GOAL_GOOD", "SAFETY"):
            scored, pending = True, "kickoff"
        elif kind == "KICKOFF":
            if live:
                play = "kickoff"
            else:
                pending = None
        elif kind == "TIMEOUT_CALLED":
            timeout = True
        elif kind == "TIMEOUT_OVER":
            timeout = False
        d = _num(r[6])
        if d:
            down = int(d)
        if not started:
            m = BEFORE
        elif ended:
            m = OVER
        elif interrupt:
            m = INTERRUPTION
        elif qbreak:
            m = BREAK
        elif live:
            m = (SCORED_LIVE if scored else KICK_LIVE if play == "kickoff"
                 else CONV_LIVE if play == "conversion" else LIVE)
        elif timeout:
            m = TIMEOUT
        elif pending == "conversion":
            m = CONV_NEXT
        elif pending == "kickoff":
            m = KICK_NEXT
        else:
            m = BETWEEN
        out.append((m, susp, period, _num(r[2]), down, status, kind))
    return out


def build(scouting_rows, prod_times):
    """match -> Timeline off scouting.fetch_scouting rows (FILE_TIME last), each message's time put
    on prod's clock by the match's median gap between prod's first publish of a message
    (bet_checks.message_times) and the message's FILE_TIME."""
    by = defaultdict(list)
    for r in scouting_rows:
        if r[1] is not None:
            by[r[0]].append(r)
    out = {}
    for match, rs in by.items():
        rs.sort(key=lambda r: int(r[1]))
        msgs = [int(r[1]) for r in rs]
        files = [_time(r[9]) if len(r) > 9 else None for r in rs]
        pm, pt = prod_times.get(match, ([], []))
        at = dict(zip(pm, pt))
        gaps = [(at[m] - f).total_seconds() for m, f in zip(msgs, files) if f is not None and m in at]
        if gaps:
            shift = statistics.median(gaps)
            times = [None if f is None else f.timestamp() + shift for f in files]
        else:
            times = [None] * len(files)
        known = [t for t in times if t is not None]
        if not known and pm:
            times = [None] * len(msgs)
            for k, m in enumerate(msgs):
                j = bisect_right(pm, m) - 1
                times[k] = pt[j].timestamp() if j >= 0 else None
        last = -math.inf
        for k, t in enumerate(times):
            last = max(last, t) if t is not None else last
            times[k] = last
        states = _states(rs)
        out[match] = Timeline(msgs, times, [s[0] for s in states], [s[1] for s in states],
                              [s[2] for s in states], [s[3] for s in states],
                              [s[4] for s in states], [(s[5], s[6]) for s in states])
    return out


def _band(x, edges, unit="s"):
    """A number as a band: '<2s', '2-5s', ... '60s+'."""
    if x is None:
        return "unknown"
    lo = None
    for e in edges:
        if x < e:
            return f"<{e}{unit}" if lo is None else f"{lo}-{e}{unit}"
        lo = e
    return f"{lo}{unit}+"


def _moved(tl, i, j):
    """What the feed did in messages (i, j]: the weightiest of a score, a kick-off, a play started, a
    play over, a timeout, a possession; 'nothing' where none of them."""
    if j <= i:
        return "nothing"
    labels = set()
    for status, kind in tl.events[i + 1:j + 1]:
        for label, kinds in MOVES:
            if kind and kind.startswith(kinds):
                labels.add(label)
    for label, _ in MOVES:
        if label in labels:
            return label
    return "nothing"


def _quarter(period):
    if period is None:
        return "unknown"
    return f"Q{period}" if period <= 4 else "OT"


def _margin_band(score):
    d = abs(score[0] - score[1])
    return "level (0-2)" if d <= 2 else "1 score (3-8)" if d <= 8 else "2+ scores (9+)"


def moment_of(tl, checks, match, bet_time, price_message):
    """Every moment field for one bet."""
    out = dict(moment=None, feed_suspended=None, moved=None, price_age_seconds=None,
               price_age=None, next_score_seconds=None, next_score=None, quarter=None,
               clock_left=None, clock_band=None, down=None, score_margin=None)
    if tl is None or bet_time is None:
        return out
    t = bet_time.timestamp()
    j = bisect_right(tl.times, t) - 1
    if j < 0 or tl.times[j] == -math.inf:
        out["moment"] = BEFORE
        return out
    m = tl.msgs[j]
    clock = tl.clock[j]
    out.update(moment=tl.moment[j], feed_suspended="suspended" if tl.suspended[j] else "open",
               quarter=_quarter(tl.period[j]), clock_left=clock, down=tl.down[j])
    if clock is not None and tl.period[j] is not None:
        out["clock_band"] = (f"{_quarter(tl.period[j])} last 2:00" if clock <= 120
                             else f"{_quarter(tl.period[j])} before 2:00")
    score = checks.score_at(match, m)
    out["score_margin"] = _margin_band(score)
    if price_message is not None:
        p = int(price_message)
        i = bisect_right(tl.msgs, p) - 1
        out["moved"] = "price after the bet" if p > m else _moved(tl, i, j)
        if 0 <= i < len(tl.times) and tl.times[i] > -math.inf:
            out["price_age_seconds"] = round(t - tl.times[i], 2)
            out["price_age"] = _band(out["price_age_seconds"], AGE_BANDS)
    entry = checks.scores.get(match)
    if entry:
        smsgs, vals = entry
        k = bisect_right(smsgs, m)
        while k < len(smsgs) and vals[k] == score:
            k += 1
        if k < len(smsgs):
            n = bisect_right(tl.msgs, smsgs[k]) - 1
            if 0 <= n < len(tl.times) and tl.times[n] > -math.inf:
                out["next_score_seconds"] = round(tl.times[n] - t, 2)
    out["next_score"] = ("none after" if entry and out["next_score_seconds"] is None
                         else _band(out["next_score_seconds"], NEXT_BANDS))
    return out


def annotate(results, bets, checks, timelines):
    """Add the moment fields to every row of every candidate's rows (the same bets, in order)."""
    for k, b in enumerate(bets):
        if not b.in_play:
            extra = dict(moment_of(None, checks, b.match_code, None, None), moment="pre-match")
        else:
            msg = results[0][1][k].get("message")
            extra = moment_of(timelines.get(b.match_code), checks, b.match_code, b.time, msg)
        for _, rows in results:
            rows[k].update(extra)


def _in_scope(r):
    return (r.get("in_play") and not r.get("excluded") and r.get("result") in ("won", "lost", "push")
            and r.get("stake"))


def buckets(rows, by):
    """{bucket: (bets, stake, revenue, margin %, 2 x its standard error, expected margin %)} over the
    in-play bets the checks keep, settled won / lost / push."""
    groups = defaultdict(list)
    for r in rows:
        if _in_scope(r):
            groups[by(r) if callable(by) else r.get(by)].append(r)
    out = {}
    for g, rs in groups.items():
        stake = sum(r["stake"] for r in rs)
        rev = sum(r["revenue"] for r in rs)
        m = rev / stake
        se = math.sqrt(sum((r["revenue"] - r["stake"] * m) ** 2 for r in rs)) / stake
        priced = [r for r in rs if r.get("stream_prob") and r.get("odds")]
        exp_stake = sum(r["stake"] for r in priced)
        exp = (sum(r["stake"] * (1 - r["odds"] * r["stream_prob"]) for r in priced) / exp_stake
               if exp_stake else None)
        out[g] = (len(rs), stake, rev, 100 * m, 200 * se, None if exp is None else 100 * exp)
    return out


DIMENSIONS = (("moment", "the feed at bet time"), ("feed_suspended", "the feed's own suspension"),
              ("moved", "what the feed did between the price and the bet"),
              ("price_age", "the price's age at the bet"), ("next_score", "the board's next move"),
              ("clock_band", "quarter and clock"), ("score_margin", "the score"),
              ("market", "market"), ("selection", "selection"))


def _order(dim, g):
    if dim == "moment" and g in MOMENTS:
        return (0, MOMENTS.index(g))
    return (1, str(g))


def report(results, min_bets=MIN_BETS, candidates=True):
    """Lines of text: the book's margin bucketed by every moment field, worst first within each, and
    (with candidates) each candidate's change in margin on the bets it re-priced there."""
    from .bets import compare
    rows = results[0][1]
    if not candidates:
        results = []
    scope = [r for r in rows if _in_scope(r)]
    if not scope:
        return ["\n  losing bets by game moment: no in-play bets to bucket"]
    total = sum(r["stake"] for r in scope)
    names = [n for n, _ in results]
    compare = compare if results else (lambda *a: {})
    lines = [f"\n  where in the game the book loses: {len(scope):,} in-play bets, {total:,.0f} stake "
             "(margin = revenue / stake; +-2se; expected = what prod's probability says the book "
             "keeps; candidates: change in margin on the bets they all re-priced)"]
    head = (f"  {'':30s} {'bets':>7s} {'stake%':>7s} {'revenue':>11s} {'margin':>8s} {'+-2se':>6s} "
            f"{'expected':>9s} " + " ".join(f"{n[:9]:>9s}" for n in names))
    for dim, title in DIMENSIONS:
        table = buckets(rows, dim)
        cmp_ = compare(results, dim, _in_scope)
        lines.append(f"\n  by {title}")
        lines.append(head)
        for g in sorted(table, key=lambda g: _order(dim, g)):
            n, stake, rev, m, se2, exp = table[g]
            if n < min_bets:
                continue
            c = cmp_.get(g)
            cells = (" ".join(f"{x - c[2]:+9.2f}" for x in c[3]) if c else "")
            lines.append(f"  {str(g)[:30]:30s} {n:7,d} {100 * stake / total:6.1f}% {rev:+11,.0f} "
                         f"{m:7.2f}% {se2:6.2f} {'' if exp is None else f'{exp:8.2f}%'} {cells}")
    combos = buckets(rows, lambda r: (r.get("moment"), r.get("moved"), r.get("next_score")))
    worst = sorted(((g, v) for g, v in combos.items() if v[0] >= min_bets), key=lambda kv: kv[1][2])
    lines.append("\n  the costliest moments (feed at bet time / feed moved since the price / next "
                 "score), by revenue")
    lines.append(f"  {'':62s} {'bets':>7s} {'revenue':>11s} {'margin':>8s} {'expected':>9s}")
    for g, (n, stake, rev, m, se2, exp) in worst[:15]:
        text = " / ".join(str(x) for x in g)
        lines.append(f"  {text[:62]:62s} {n:7,d} {rev:+11,.0f} {m:7.2f}% "
                     f"{'' if exp is None else f'{exp:8.2f}%'}")
    return lines


def html_tables(results, esc, change_cell, min_bets=MIN_BETS):
    """The bucket tables as HTML, for the betting section of the calibration page."""
    from .bets import compare
    rows = results[0][1]
    scope = [r for r in rows if _in_scope(r)]
    if not scope:
        return ""
    total = sum(r["stake"] for r in scope)
    names = [n for n, _ in results]
    head = ("<tr><th>Bucket</th><th>Bets</th><th>Stake</th><th>Revenue</th><th>Margin</th>"
            "<th>&plusmn;2se</th><th>Expected</th>" + "".join(f"<th>{esc(n)}</th>" for n in names)
            + "</tr>")
    parts = []
    for dim, title in DIMENSIONS[:7]:
        table = buckets(rows, dim)
        cmp_ = compare(results, dim, _in_scope)
        body = []
        for g in sorted(table, key=lambda g: _order(dim, g)):
            n, stake, rev, m, se2, exp = table[g]
            if n < min_bets:
                continue
            c = cmp_.get(g)
            cls = "bad" if exp is not None and m < exp - se2 else ""
            body.append(f"<tr><th>{esc(g)}</th><td>{n:,}</td><td>{100 * stake / total:.1f}%</td>"
                        f"<td>{rev:+,.0f}</td><td class=\"{cls}\">{m:.2f}%</td><td>{se2:.2f}</td>"
                        f"<td>{'' if exp is None else f'{exp:.2f}%'}</td>"
                        + ("".join(change_cell(x - c[2]) for x in c[3]) if c else
                           "<td>&mdash;</td>" * len(names)) + "</tr>")
        parts.append(f"<h4>By {esc(title)}</h4><table class=\"reach\"><thead>{head}</thead>"
                     f"<tbody>{''.join(body)}</tbody></table>")
    return (f"<h3>Where in the game the book loses</h3><p class=\"dim\">{len(scope):,} in-play bets. "
            "Margin in red: more than two standard errors below what prod's own probability says the "
            "book keeps. Candidate columns: change in margin on the bets every candidate re-priced."
            "</p>" + "".join(parts))


NUMBERS = ("stake", "revenue", "odds", "stream_prob", "price_age_seconds", "next_score_seconds",
           "clock_left")


def from_csv(path):
    """bets_sim.csv or bets_checks.csv rows back as the report reads them."""
    import csv
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in NUMBERS:
                r[k] = _num(r.get(k))
            r["in_play"] = str(r.get("in_play")).lower() == "true"
            r["excluded"] = str(r.get("excluded")).lower() == "true"
            out.append(r)
    return out
