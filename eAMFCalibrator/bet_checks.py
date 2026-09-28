"""Checks on the bets against the match, before any is re-priced.

  phase   when in the match each bet was placed, off SCOUTING_FULL on prod's
          clock: pre-match, in play, after two minutes left in the fourth
          quarter, after the match-over message. A bet marked pre-match after
          kickoff, marked in play before it, or placed after the match was
          over is left out (a resettlement or a trader review, not a price
          anyone took)
  state   whether a model's game state carries the same information as
          prod's at the price the bet saw: the model knows the plays up to
          its latest PLAY_OVER, so SCOUTING_FULL must hold every message prod
          had from there to the bet's price, and no score may have changed in
          between
"""

import math
import re
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass

from . import config
from .scouting import PERIOD_START, _text

OVER_STATUS = re.compile(r"^(MATCH|GAME)_(OVER|END|ENDED|FINISHED|COMPLETE|COMPLETED)$|^FULL_TIME$|^FINAL$")
TWO_MINUTES = 120
PRE_MATCH, PRE_AFTER_KICKOFF = "pre-match", "pre-match after kickoff"
BEFORE_KICKOFF, IN_PLAY, AFTER_TWO, AFTER_OVER, NO_FEED = (
    "in play before kickoff", "in play", "after two minutes", "after match over", "no scouting")
EXCLUDED = {PRE_AFTER_KICKOFF, BEFORE_KICKOFF, AFTER_OVER}


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class MatchFeed:
    """One match's SCOUTING_FULL messages, as the checks read them."""
    messages: frozenset
    play_overs: list
    kickoff: int = None
    two_minutes: int = None
    over: int = None
    over_status: str = None
    last: int = None


def build_feeds(scouting_rows):
    """match -> MatchFeed off scouting.fetch_scouting rows."""
    by = defaultdict(list)
    for r in scouting_rows:
        if r[1] is not None:
            by[r[0]].append(r)
    out = {}
    for match, rs in by.items():
        rs.sort(key=lambda r: int(r[1]))
        period = kickoff = two = over = over_status = None
        msgs, overs = set(), []
        for r in rs:
            m = int(r[1])
            msgs.add(m)
            status, kind = _text(r[3]), _text(r[4])
            if status in PERIOD_START:
                period = PERIOD_START[status]
            if kind == "PLAY_STARTED" and kickoff is None:
                kickoff = m
            if kind == "PLAY_OVER":
                overs.append(m)
            clock = _num(r[2])
            if two is None and kickoff is not None and period == 4 and clock is not None \
                    and clock <= TWO_MINUTES:
                two = m
            if over is None and status and OVER_STATUS.match(status):
                over, over_status = m, status
        out[match] = MatchFeed(frozenset(msgs), overs, kickoff, two, over, over_status, int(rs[-1][1]))
    return out


def message_times(quote_rows):
    """match -> (sorted messages, publish times): prod's first row on each message."""
    from .bets import _naive
    first = {}
    for r in quote_rows:
        if r[6] is None or r[2] is None:
            continue
        key, t = (r[0], int(r[6])), _naive(r[2])
        if key not in first or t < first[key]:
            first[key] = t
    by = defaultdict(list)
    for (match, m), t in first.items():
        by[match].append((m, t))
    return {match: ([m for m, _ in sorted(v)], [t for _, t in sorted(v)]) for match, v in by.items()}


def score_index(score_rows):
    """match -> (sorted messages, (player 1, player 2) cumulative) off snowflake_io.fetch_scores."""
    by = defaultdict(list)
    for s in score_rows:
        if s[1] is not None:
            by[s[0]].append((int(s[1]), s[5], s[6]))
    out = {}
    for match, rs in by.items():
        rs.sort()
        p1 = p2 = 0
        msgs, vals = [], []
        for m, c1, c2 in rs:
            p1 = c1 if c1 is not None else p1
            p2 = c2 if c2 is not None else p2
            msgs.append(m)
            vals.append((p1, p2))
        out[match] = (msgs, vals)
    return out


class Checks:
    """The phase and state checks for every bet of a window."""

    def __init__(self, feeds, times, scores):
        self.feeds, self.times, self.scores = feeds, times, scores

    def time_of(self, match, message):
        """Publish time of prod's first row at or after the message (its last, past the end)."""
        entry = self.times.get(match)
        if entry is None or message is None:
            return None
        msgs, ts = entry
        i = bisect_left(msgs, message)
        return ts[i] if i < len(ts) else (ts[-1] if ts else None)

    def score_at(self, match, message):
        entry = self.scores.get(match)
        if entry is None:
            return (0, 0)
        msgs, vals = entry
        i = bisect_right(msgs, message)
        return vals[i - 1] if i else (0, 0)

    def phase(self, b):
        """(phase, seconds past the bound the phase is read against: after kickoff, before it, after
        two minutes left or after the match-over message)."""
        f = self.feeds.get(b.match_code)
        kick = self.time_of(b.match_code, f.kickoff) if f else None
        if kick is None or b.time is None:
            return NO_FEED, None
        two = self.time_of(b.match_code, f.two_minutes) if f.two_minutes is not None else None
        over = self.time_of(b.match_code, f.over if f.over is not None else f.last)
        tol = config.BET_PHASE_TOLERANCE
        after_two = (b.time - two).total_seconds() if two else None
        after_over = (b.time - over).total_seconds() if over else None
        kick_s = (b.time - kick).total_seconds()
        if not b.in_play:
            return (PRE_MATCH, None) if kick_s <= tol else (PRE_AFTER_KICKOFF, kick_s)
        if kick_s < -tol:
            return BEFORE_KICKOFF, -kick_s
        if after_over is not None and after_over > tol:
            return AFTER_OVER, after_over
        if after_two is not None and after_two >= 0:
            return AFTER_TWO, after_two
        return IN_PLAY, None

    def state(self, match, message):
        """The model's state at prod's message: its latest PLAY_OVER, how far behind, the messages
        prod had since that SCOUTING_FULL lacks, and whether the score moved in between."""
        out = dict(snapshot_message=None, snapshot_age_seconds=None, snapshot_age_messages=None,
                   scouting_missing=None, score_changed=None, state_ok=False)
        f = self.feeds.get(match)
        if f is None or message is None:
            return out
        message = int(message)
        i = bisect_right(f.play_overs, message) - 1
        if i < 0:
            return out
        s = f.play_overs[i]
        msgs = self.times.get(match, ([], []))[0]
        missing = sum(1 for m in msgs[bisect_right(msgs, s):bisect_right(msgs, message)]
                      if m not in f.messages)
        changed = self.score_at(match, message) != self.score_at(match, s)
        t0, t1 = self.time_of(match, s), self.time_of(match, message)
        out.update(snapshot_message=s, snapshot_age_messages=message - s,
                   snapshot_age_seconds=(t1 - t0).total_seconds() if t0 and t1 else None,
                   scouting_missing=missing, score_changed=changed,
                   state_ok=missing <= config.MAX_SCOUTING_GAP and not changed)
        return out

    def for_bet(self, b, message):
        """Every check field for one bet, read at prod's message."""
        phase, past = self.phase(b)
        out = dict(match_phase=phase, excluded=phase in EXCLUDED, phase_seconds=past)
        out.update(self.state(b.match_code, message))
        return out

    def completeness(self):
        """match -> (prod messages from kickoff to match over, the share SCOUTING_FULL holds)."""
        out = {}
        for match, (msgs, _) in self.times.items():
            f = self.feeds.get(match)
            if f is None or f.kickoff is None:
                out[match] = (0, 0.0)
                continue
            end = f.over if f.over is not None else max(msgs[-1] if msgs else 0, f.last)
            span = msgs[bisect_left(msgs, f.kickoff):bisect_right(msgs, end)]
            out[match] = (len(span), sum(m in f.messages for m in span) / len(span) if span else 0.0)
        return out


def _pct(xs, q):
    return xs[math.ceil(q * (len(xs) - 1))] if xs else None


def _secs(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return f"{'-':>7s} {'-':>7s} {'-':>7s}"
    return f"{statistics.median(xs):7.0f} {_pct(xs, 0.9):7.0f} {xs[-1]:7.0f}"


def _margin(rs):
    from .bets import LOST, PUSH, WON
    settled = [r for r in rs if r["result"] in (WON, LOST, PUSH)]
    stake = sum(r["stake"] for r in settled)
    return 100 * sum(r["revenue"] for r in settled) / stake if stake else float("nan")


def report(rows, checks):
    """Lines of text: the phases, the cash-outs, the match-over messages found, and the models'
    state against prod's, with SCOUTING_FULL's completeness by match."""
    from .bets import CASHED
    lines = ["\n  when the bets were placed, on prod's clock: seconds after kickoff (pre-match after "
             "kickoff), before it (in play before kickoff), after two minutes left in Q4, after the "
             f"match-over message; the (left out) ones allow {config.BET_PHASE_TOLERANCE}s",
             f"  {'':42s} {'bets':>7s} {'stake':>11s} {'margin':>8s}   seconds past the bound "
             "(median 90th max)"]
    order = [PRE_MATCH, PRE_AFTER_KICKOFF, BEFORE_KICKOFF, IN_PLAY, AFTER_TWO, AFTER_OVER, NO_FEED]
    groups = defaultdict(list)
    for r in rows:
        groups[(str(r.get(config.BET_GROUP_COLUMN)), r.get("match_phase"))].append(r)
    for (op, phase), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1])
                                                                  if kv[0][1] in order else 99)):
        stake = sum(r["stake"] for r in rs)
        mark = " (left out)" if phase in EXCLUDED else ""
        lines.append(f"  {(op[:20] + ' ' + str(phase) + mark)[:42]:42s} {len(rs):7,d} {stake:11,.0f} "
                     f"{_margin(rs):7.2f}%   {_secs(r['phase_seconds'] for r in rs)}")
    found = Counter(f.over_status or "none found: the last feed message" for f in checks.feeds.values())
    lines.append("  match-over message: " + ", ".join(f"{k} {v:,}" for k, v in found.most_common()))

    lines.append("\n  cash-outs (settled at the operator's cash-out offer, so left out: no candidate offer "
                 "to set against it)")
    lines.append(f"  {'':30s} {'bets':>7s} {'stake':>11s} {'of stake':>9s} {'revenue':>10s} {'margin':>8s}")
    by = defaultdict(list)
    for r in rows:
        by[(str(r.get(config.BET_GROUP_COLUMN)), "in play" if r["in_play"] else "pre-match")].append(r)
    for key, rs in sorted(by.items()):
        cashed = [r for r in rs if r["result"] == CASHED]
        stake_all = sum(r["stake"] for r in rs)
        stake = sum(r["stake"] for r in cashed)
        rev = sum(r["revenue"] for r in cashed)
        lines.append(f"  {' · '.join(key)[:30]:30s} {len(cashed):7,d} {stake:11,.0f} "
                     f"{100 * stake / stake_all if stake_all else 0:8.1f}% {rev:10,.0f} "
                     f"{100 * rev / stake if stake else 0:7.2f}%")

    live = [r for r in rows if r["in_play"] and not r.get("excluded") and r["stream_prob"] is not None]
    lines.append("\n  a model's game state against prod's at the price each in-play bet saw (the model "
                 "knows the plays to its latest PLAY_OVER)")
    if live:
        n = len(live)
        none = sum(r["snapshot_message"] is None for r in live)
        changed = sum(bool(r["score_changed"]) for r in live)
        gaps = Counter()
        for r in live:
            g = r["scouting_missing"]
            if g is not None:
                gaps["0" if g == 0 else "1" if g == 1 else "2-5" if g <= 5 else "6-20" if g <= 20
                     else "over 20"] += 1
        ok = sum(bool(r["state_ok"]) for r in live)
        lines.append(f"  {n:,} bets: same state {ok:,} ({100 * ok / n:.1f}%); before the first "
                     f"PLAY_OVER {none:,}; score changed since the PLAY_OVER {changed:,}")
        lines.append("  prod messages since the PLAY_OVER that SCOUTING_FULL lacks: "
                     + ", ".join(f"{k} {gaps[k]:,}" for k in ("0", "1", "2-5", "6-20", "over 20")
                                 if gaps[k]) + f" (allowed: {config.MAX_SCOUTING_GAP})")
        per = defaultdict(list)
        for r in live:
            per[r["period"]].append(r)
        lines.append(f"  {'':12s} {'bets':>7s} {'same':>7s}   seconds since the PLAY_OVER "
                     "(median 90th max)")
        for p, rs in sorted(per.items(), key=lambda kv: str(kv[0])):
            same = sum(bool(r["state_ok"]) for r in rs)
            lines.append(f"  period {str(p):5s} {len(rs):7,d} {100 * same / len(rs):6.1f}%   "
                         f"{_secs(r['snapshot_age_seconds'] for r in rs)}")

    comp = checks.completeness()
    stake_by = defaultdict(float)
    for r in live:
        stake_by[r["match_code"]] += r["stake"]
    short = sorted(((share, n, m) for m, (n, share) in comp.items() if share < 0.99), key=lambda x: x[0])
    lines.append(f"\n  SCOUTING_FULL against prod's messages, kickoff to match over: {len(comp):,} matches, "
                 f"{len(short):,} under 99% held")
    for share, n, m in short[:15]:
        lines.append(f"  {m:16s} {100 * share:5.1f}% of {n:,} messages   in-play stake {stake_by[m]:10,.0f}")
    return lines
