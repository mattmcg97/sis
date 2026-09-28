"""Checks on the bets against the match, before any is re-priced.

  phase   when in the match each bet was placed, off SCOUTING_FULL on prod's
          clock: pre-match, in play, after two minutes left in the fourth
          quarter, after the match-over message. A bet marked pre-match after
          the match started, marked in play before it, placed after two
          minutes left (the operators suspend there) or after the match was
          over is left out: a resettlement or a trader review, not a price
          anyone took
  state   the exact SCOUTING_FULL messages behind every price. Prod's price
          the bet saw was made at feed message m. A model prices off its
          latest PLAY_OVER s; it is the same information only where s is a
          PLAY_OVER the model can read (model_books: the same snapshots and
          the same reading as v4-v6, without simulating), TEAM_A's side is
          known, and the feed did not move on in (s, m]: no play started, no
          status beyond config.NEUTRAL_FEED_MESSAGES, no score change, and
          no message prod had that SCOUTING_FULL lacks. A candidate table's
          quote at message c is held to the same (c, m] window.
"""

import math
import re
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from . import config
from .scouting import PERIOD_START, _text

OVER_STATUS = re.compile(r"^ENDED$|^(MATCH|GAME)_(OVER|END|ENDED|FINISHED|COMPLETE|COMPLETED)$")
TWO_MINUTES = 120
PRE_MATCH, PRE_AFTER_START = "pre-match", "pre-match after the start"
BEFORE_START, IN_PLAY, AFTER_TWO, AFTER_OVER, NO_FEED = (
    "in play before the start", "in play", "after two minutes", "after match over", "no scouting")


def excluded_phases():
    """The phases left out of the re-pricing."""
    out = {PRE_AFTER_START, BEFORE_START, AFTER_OVER}
    if config.EXCLUDE_AFTER_TWO_MINUTES:
        out.add(AFTER_TWO)
    return out


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
    start: int = None
    two_minutes: int = None
    over: int = None
    over_status: str = None
    last: int = None
    late: dict = field(default_factory=dict)
    tail: tuple = ()
    moves: list = field(default_factory=list)
    move_labels: list = field(default_factory=list)


def build_feeds(scouting_rows):
    """match -> MatchFeed off scouting.fetch_scouting rows. The start is the first quarter's start
    message (the first PLAY_STARTED without one); `late` is each status's first message from two
    minutes left in Q4, and `tail` the last few statuses."""
    by = defaultdict(list)
    for r in scouting_rows:
        if r[1] is not None:
            by[r[0]].append(r)
    out = {}
    for match, rs in by.items():
        rs.sort(key=lambda r: int(r[1]))
        period = start = kick = two = over = over_status = None
        msgs, overs, late, statuses, moves = set(), [], {}, [], []
        neutral = {x.upper() for x in config.NEUTRAL_FEED_MESSAGES}
        for r in rs:
            m = int(r[1])
            msgs.add(m)
            status, kind = _text(r[3]), _text(r[4])
            for label in (kind, status):
                if label and label not in neutral:
                    moves.append((m, label))
            if status in PERIOD_START:
                period = PERIOD_START[status]
                if period == 1 and start is None:
                    start = m
            if kind == "PLAY_STARTED" and kick is None:
                kick = m
            if kind == "PLAY_OVER":
                overs.append(m)
            clock = _num(r[2])
            if two is None and kick is not None and period == 4 and clock is not None \
                    and clock <= TWO_MINUTES:
                two = m
            if status:
                statuses.append(status)
                if two is not None and status not in late:
                    late[status] = m
            if over is None and status and OVER_STATUS.match(status):
                over, over_status = m, status
        start = start if start is not None and (kick is None or start <= kick) else kick
        out[match] = MatchFeed(frozenset(msgs), overs, start, two, over, over_status, int(rs[-1][1]),
                               late, tuple(statuses[-3:]), [m for m, _ in moves],
                               [label for _, label in moves])
    return out


def _day(match):
    """The match's date off its code (AFnnnDDMMYY), as YYYY-MM-DD."""
    d = str(match)[-6:]
    return f"20{d[4:6]}-{d[2:4]}-{d[0:2]}" if d.isdigit() else "?"


def _missing(r):
    """Which of the state's fields a snapshot row lacks."""
    gone = [k for k in ("offense", "down", "distance", "field_position") if r.get(k) in ("", None)]
    return "+".join(gone) or "none missing"


def model_books(snapshots, diag=None):
    """(match -> the PLAY_OVER messages a model prices off, matches whose TEAM_A side is guessed,
    Counter of why the others cannot be priced), off the snapshots the models read
    (snowflake_io._play_over_snapshots), read as v4-v6's match_books read them. `diag` (a dict)
    collects the unreadable ones by play kind and missing field, and the days."""
    from eAMFModel import playover
    books, guessed, reasons = {}, set(), Counter()
    diag = diag if diag is not None else {}
    detail, days = diag.setdefault("detail", Counter()), diag.setdefault("days", defaultdict(Counter))
    for match, snaps in snapshots.items():
        rows = [{k: "" if v is None else str(v) for k, v in r.items()}
                for r in sorted(snaps, key=lambda r: int(r["message"]))]
        if not rows:
            continue
        day = days[_day(match)]
        day["matches"] += 1
        if any(r.get("team_a_side") not in ("home", "away") for r in rows):
            guessed.add(match)
            day["side not known"] += 1
            rows = [r if r.get("team_a_side") in ("home", "away") else dict(r, team_a_side="home")
                    for r in rows]
        msgs = []
        for r in rows:
            state, why = playover.state_for(r)
            day["play_overs"] += 1
            if state is None:
                reasons[why] += 1
                detail[(r.get("play_kind") or "?", why, _missing(r))] += 1
            else:
                msgs.append(int(r["message"]))
                reasons["priced"] += 1
                day["read"] += 1
        books[match] = msgs
    return books, guessed, reasons


OTHER_SPORT = re.compile(r"FREE_THROW|REBOUND|THREE_POINT|JUMP_BALL|DUNK|LAYUP|ASSIST|STEAL|"
                         r"SHOT|BASKET|GOAL_SCORED|CORNER|OFFSIDE|YELLOW_CARD|RED_CARD|INNING|WICKET")


def vocabulary(scouting_rows):
    """(Counter of (column, message) -> rows, Counter of (column, message) -> matches, match codes
    not starting with the AF prefix) over SCOUTING_FULL rows: the check that only Madden reached
    the join."""
    from .scouting import MATCH_PREFIX
    rows, matches, seen = Counter(), Counter(), set()
    odd = set()
    for r in scouting_rows:
        if not str(r[0]).upper().startswith(MATCH_PREFIX):
            odd.add(r[0])
        for col, value in (("status", _text(r[3])), ("in play", _text(r[4]))):
            if value:
                rows[(col, value)] += 1
                if (r[0], col, value) not in seen:
                    seen.add((r[0], col, value))
                    matches[(col, value)] += 1
    return rows, matches, odd


def side_diagnosis(scouting_rows, score_rows, matches):
    """For matches whose TEAM_A side is not known: Counter of why (no scoring message by a score
    change, or votes split), and Counter of the feed messages at their score changes."""
    from .scouting import team_a_votes
    matches = set(matches)
    rows, scores = defaultdict(list), defaultdict(list)
    for r in scouting_rows:
        if r[0] in matches:
            rows[r[0]].append(r)
    for sc in score_rows:
        if sc[0] in matches:
            scores[sc[0]].append(sc)
    why, near = Counter(), Counter()
    for match in matches:
        votes = team_a_votes(rows[match], scores[match])
        why["no scoring message by a score change" if not votes else
            "scoring messages split: " + ", ".join(f"{k} {v}" for k, v in sorted(votes.items()))
            if len(votes) > 1 else "one side only"] += 1
        kinds = sorted((int(r[1]), _text(r[4])) for r in rows[match] if r[1] is not None and _text(r[4]))
        msgs = [m for m, _ in kinds]
        for sc in scores[match]:
            if sc[1] is None or not ((sc[3] or 0) or (sc[4] or 0)):
                continue
            m = int(sc[1])
            for _, kind in kinds[bisect_left(msgs, m - 3):bisect_right(msgs, m)]:
                near[kind] += 1
    return why, near


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

    def __init__(self, feeds, times, scores, books=None, guessed=frozenset(), reasons=None, diag=None):
        self.feeds, self.times, self.scores = feeds, times, scores
        self.books, self.guessed, self.reasons = books, set(guessed), reasons or Counter()
        self.diag = diag or {}

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
        """(phase, seconds past the bound it is read against: after the start, before it, after
        two minutes left or after the match-over message)."""
        f = self.feeds.get(b.match_code)
        start = self.time_of(b.match_code, f.start) if f else None
        if start is None or b.time is None:
            return NO_FEED, None
        two = self.time_of(b.match_code, f.two_minutes) if f.two_minutes is not None else None
        over = self.time_of(b.match_code, f.over if f.over is not None else f.last)
        tol = config.BET_PHASE_TOLERANCE
        after_two = (b.time - two).total_seconds() if two else None
        after_over = (b.time - over).total_seconds() if over else None
        start_s = (b.time - start).total_seconds()
        if not b.in_play:
            return (PRE_MATCH, None) if start_s <= tol else (PRE_AFTER_START, start_s)
        if start_s < -tol:
            return BEFORE_START, -start_s
        if after_over is not None and after_over > tol:
            return AFTER_OVER, after_over
        if after_two is not None and after_two > tol:
            return AFTER_TWO, after_two
        return IN_PLAY, None

    def window(self, match, frm, to):
        """(reason, prod messages SCOUTING_FULL lacks) for the feed from message frm to `to`: None
        where it did not move on in (frm, to] -- no play or status beyond the neutral ones, no score
        change, every prod message held."""
        f = self.feeds.get(match)
        msgs = self.times.get(match, ([], []))[0]
        missing = sum(1 for m in msgs[bisect_right(msgs, frm):bisect_right(msgs, to)]
                      if m not in f.messages)
        i, j = bisect_right(f.moves, frm), bisect_right(f.moves, to)
        if j > i:
            return f"feed moved on: {f.move_labels[i]}", missing
        if self.score_at(match, to) != self.score_at(match, frm):
            return "score changed", missing
        if missing > config.MAX_SCOUTING_GAP:
            return "SCOUTING_FULL missing prod's messages", missing
        return None, missing

    def state(self, match, message):
        """The model against prod's message m: the feed messages behind its price (its latest
        PLAY_OVER s to m), and whether they carry the same information as prod's."""
        out = dict(feed_from=None, feed_to=message, snapshot_age_seconds=None,
                   scouting_missing=None, state_ok=False, state_reason=None)
        f = self.feeds.get(match)
        if f is None or message is None:
            out["state_reason"] = "no scouting"
            return out
        message = int(message)
        i = bisect_right(f.play_overs, message) - 1
        if i < 0:
            out["state_reason"] = "before the first PLAY_OVER"
            return out
        s = f.play_overs[i]
        t0, t1 = self.time_of(match, s), self.time_of(match, message)
        reason, missing = self.window(match, s, message)
        if match in self.guessed:
            reason = "TEAM_A's side not known"
        elif self.books is not None and s not in set(self.books.get(match, ())):
            reason = "the latest PLAY_OVER cannot be read"
        out.update(feed_from=s, snapshot_age_seconds=(t1 - t0).total_seconds() if t0 and t1 else None,
                   scouting_missing=missing, state_ok=reason is None, state_reason=reason)
        return out

    def candidate(self, match, cand_message, message):
        """A candidate's quote at feed message c against prod's at m: the same information only
        where the feed did not move on in (c, m]."""
        if cand_message is None or message is None:
            return dict(candidate_message=cand_message, candidate_ok=None, candidate_reason=None)
        f = self.feeds.get(match)
        if f is None:
            return dict(candidate_message=cand_message, candidate_ok=False,
                        candidate_reason="no scouting")
        c, m = int(cand_message), int(message)
        reason = "candidate's quote after prod's" if c > m else self.window(match, c, m)[0]
        return dict(candidate_message=c, candidate_ok=reason is None, candidate_reason=reason)

    def for_bet(self, b, message):
        """Every check field for one bet, read at prod's message."""
        phase, past = self.phase(b)
        out = dict(match_phase=phase, excluded=phase in excluded_phases(), phase_seconds=past)
        out.update(self.state(b.match_code, message))
        return out

    def completeness(self):
        """match -> (prod messages from the start to match over, the share SCOUTING_FULL holds);
        None for a match SCOUTING_FULL does not have."""
        out = {}
        for match, (msgs, _) in self.times.items():
            f = self.feeds.get(match)
            if f is None or f.start is None:
                out[match] = None
                continue
            end = f.over if f.over is not None else max(msgs[-1] if msgs else 0, f.last)
            span = msgs[bisect_left(msgs, f.start):bisect_right(msgs, end)]
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
    """Lines of text: the phases and the statuses around the end, the cash-outs, and the models'
    state against prod's, with SCOUTING_FULL's completeness by match."""
    from .bets import CASHED
    left_out = excluded_phases()
    lines = []
    sports = checks.diag.get("sports")
    vocab = checks.diag.get("vocab")
    if sports is not None or vocab is not None:
        lines.append("\n  sport check: only AF may reach the join")
        if sports is not None:
            lines.append("  EVENT's SPORT_CODE for the bets' matches: "
                         + (", ".join(f"{k} {v:,}" for k, v in sports.most_common()) or "none found"))
        if vocab is not None:
            counts, seen_in, odd = vocab
            lines.append(f"  SCOUTING_FULL match codes not starting AF: {len(odd):,}"
                         + (f" ({', '.join(sorted(map(str, odd))[:5])})" if odd else ""))
            flagged = [k for k in counts if OTHER_SPORT.search(k[1])]
            lines.append("  messages that look like another sport: "
                         + (", ".join(f"{v} {counts[(c, v)]:,}" for c, v in flagged) or "none"))
            lines.append("  every SCOUTING_FULL message (rows / matches):")
            for col in ("in play", "status"):
                items = sorted(((k, n) for k, n in counts.items() if k[0] == col), key=lambda kv: -kv[1])
                lines.append(f"    {col}: " + ", ".join(f"{v} {n:,}/{seen_in[(c, v)]:,}"
                                                     for (c, v), n in items))
    lines += ["\n  when the bets were placed, on prod's clock: seconds after the start (pre-match after "
             "the start), before it (in play before the start), after two minutes left in Q4, after "
             f"the match-over message; the (left out) ones allow {config.BET_PHASE_TOLERANCE}s",
             f"  {'':44s} {'bets':>7s} {'stake':>11s} {'margin':>8s}   seconds past the bound "
             "(median 90th max)"]
    order = [PRE_MATCH, PRE_AFTER_START, BEFORE_START, IN_PLAY, AFTER_TWO, AFTER_OVER, NO_FEED]
    groups = defaultdict(list)
    for r in rows:
        groups[(str(r.get(config.BET_GROUP_COLUMN)), r.get("match_phase"))].append(r)
    for (op, phase), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], order.index(kv[0][1])
                                                                  if kv[0][1] in order else 99)):
        stake = sum(r["stake"] for r in rs)
        name = f"{op[:20]} {phase}" + (" (left out)" if phase in left_out else "")
        lines.append(f"  {name[:44]:44s} {len(rs):7,d} {stake:11,.0f} "
                     f"{_margin(rs):7.2f}%   {_secs(r['phase_seconds'] for r in rs)}")
    found = Counter(f.over_status or "none found: the last feed message" for f in checks.feeds.values())
    lines.append("  match-over message: " + ", ".join(f"{k} {v:,}" for k, v in found.most_common()))
    late = defaultdict(list)
    for match, f in checks.feeds.items():
        two = checks.time_of(match, f.two_minutes)
        for status, m in f.late.items():
            t = checks.time_of(match, m)
            if two and t:
                late[status].append((t - two).total_seconds())
    if late:
        lines.append("  statuses from two minutes left in Q4 (matches; seconds after the mark, median "
                     "90th max):")
        for status, xs in sorted(late.items(), key=lambda kv: -len(kv[1]))[:10]:
            lines.append(f"    {status[:36]:36s} {len(xs):5,d}  {_secs(xs)}")
    tails = Counter(s for f in checks.feeds.values() for s in f.tail)
    lines.append("  statuses on the matches' last rows: "
                 + ", ".join(f"{k} {v:,}" for k, v in tails.most_common(8)))

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
    stake_by = defaultdict(float)
    for r in live:
        stake_by[r["match_code"]] += r["stake"]
    reasons = checks.reasons
    total = sum(reasons.values())
    if total:
        lines.append("\n  the models' PLAY_OVERs (v4-v6 price off the ones they can read; read here "
                     "without simulating)")
        lines.append(f"  {total:,} PLAY_OVERs: priced {reasons['priced']:,} "
                     f"({100 * reasons['priced'] / total:.1f}%); not: "
                     + ", ".join(f"{k} {v:,}" for k, v in reasons.most_common() if k != "priced"))
    detail = checks.diag.get("detail")
    if detail:
        lines.append("  the unreadable ones by play kind, reason and the fields missing:")
        for (kind, why, gone), n in detail.most_common(10):
            lines.append(f"    {kind[:14]:14s} {why[:10]:10s} {gone[:40]:40s} {n:7,d}")
    if checks.books is not None:
        lines.append(f"  matches with TEAM_A's side not known (no longer guessed; not priced): "
                     f"{len(checks.guessed):,} of {len(checks.books):,}, in-play stake "
                     f"{sum(stake_by[m] for m in checks.guessed):,.0f}")
    for key, title in (("side_why", "    why"), ("side_near", "    feed messages at their score changes")):
        c = checks.diag.get(key)
        if c:
            lines.append(f"{title}: " + ", ".join(f"{k} {v:,}" for k, v in c.most_common(10)))
    days = checks.diag.get("days")
    if days:
        lines.append(f"  by match day {'':3s} {'matches':>8s} {'side not known':>15s} {'PLAY_OVERs':>11s} "
                     f"{'read':>7s}")
        for day, c in sorted(days.items()):
            lines.append(f"  {day:16s} {c['matches']:8,d} {c['side not known']:15,d} "
                         f"{c['play_overs']:11,d} {100 * c['read'] / c['play_overs'] if c['play_overs'] else 0:6.1f}%")

    lines.append("\n  a model against prod's price at feed message m: priced off its latest PLAY_OVER s, "
                 "the same information only where the feed did not move on in (s, m]")
    if live:
        n = len(live)
        ok = sum(bool(r["state_ok"]) for r in live)
        why = Counter(r["state_reason"] for r in live if not r["state_ok"])
        lines.append(f"  {n:,} in-play bets: same information {ok:,} ({100 * ok / n:.1f}%)")
        for k, v in why.most_common(12):
            lines.append(f"    {str(k)[:60]:60s} {v:7,d}")
        lines.append(f"  (neutral feed messages: {', '.join(config.NEUTRAL_FEED_MESSAGES)}; SCOUTING_FULL "
                     f"may lack {config.MAX_SCOUTING_GAP} of prod's messages)")
        per = defaultdict(list)
        for r in live:
            per[r["period"]].append(r)
        lines.append(f"  {'':12s} {'bets':>7s} {'same':>7s}   seconds from s to m (median 90th max)")
        for p, rs in sorted(per.items(), key=lambda kv: str(kv[0])):
            same = sum(bool(r["state_ok"]) for r in rs)
            lines.append(f"  period {str(p):5s} {len(rs):7,d} {100 * same / len(rs):6.1f}%   "
                         f"{_secs(r['snapshot_age_seconds'] for r in rs if r['state_ok'])}")

    comp = checks.completeness()
    missing = [m for m, v in comp.items() if v is None]
    short = sorted(((v[1], v[0], m) for m, v in comp.items() if v is not None and v[1] < 0.99),
                   key=lambda x: x[0])
    lines.append(f"\n  SCOUTING_FULL against prod's messages, start to match over: {len(comp):,} matches, "
                 f"{len(missing):,} not in SCOUTING_FULL, {len(short):,} under 99% held")
    for share, n, m in short[:15]:
        lines.append(f"  {m:16s} {100 * share:5.1f}% of {n:,} messages   in-play stake {stake_by[m]:10,.0f}")
    return lines
