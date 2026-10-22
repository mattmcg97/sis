"""The matches the book loses on the moneyline: what kind of match is targeted, how it played out,
and every situation the money piles on in, read over every match of the window so a pattern is one
prod's price misses and not one picked out by the result.

  match     every settled match of the window: the book's result on the moneyline (and on the
            handicap and total), how the stake came in (in play, restricted, VIP, customers, the
            biggest one's share, the share on the eventual winner and the price it was taken at),
            prod's pre-match favourite and closing total against the result (an upset, a comeback,
            overtime, a low or high scorer against prod's line), how Q1 and the first half were
            played (totals_signals' pace), the gamers' places in their sessions, and a
            walk-forward Elo of every gamer's wins (slow: the long run; fast: recent form) against
            prod's pre-match price
  bet       every moneyline bet against prod's probability of its selection where it was priced
            (bet time less the operator's lag): the backed side's lead, quarter and clock, who had
            the ball and where (down, distance, field position off the latest PLAY_OVER at bet
            time), the feed's moment, the points against prod's pre-match total at that stage of the
            game, plays so far and how fast they came, prod's probability either side of the bet,
            and who scored next
  residual  won less prod's probability, bet by bet: 0 on average wherever prod's price is right,
            so a cut where it sits away from 0 is a situation prod misprices (2se clustered by
            match: one result settles every bet of a match)
  pile-on   a run of moneyline bets on one side of a match none more than BURST_GAP seconds after
            the last; a pile-on is a run with PILE_MIN_BETS bets or more whose stake is in the top
            (1 - PILE_QUANTILE) of runs: the situation at its first bet, what came next, the result

Picking out the losing matches picks on the result (the money was on the winner), so the losing
against the rest table says what those matches looked like; the residual tables, read over every
match, say whether a situation is one to target. Candidates (--candidate) add their own residual
beside prod's: a model that already prices the situation brings it back to 0.

    python -m eAMFCalibrator bets ml-losers --since "2026-10-01 00:00:00"
"""

import datetime as dt
import math
import os
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict

from . import bet_moments, bets, config, markets, snowflake_io, totals_signals
from .bet_favourite import phase_of
from .bet_sessions import group_of
from .bet_totals import _start, kickoff_time, mean_se, sessions

BURST_GAP = 45               # seconds between two bets of one run
PILE_QUANTILE = 0.95         # a pile-on's stake is in the top 5% of runs
PILE_MIN_BETS = 3
WORST_SHARE = 0.10           # the worst tenth of matches by the book's moneyline result
OFFSETS = (-60, -30, 30, 60, 120, 300)
MOVE_FLAT = 0.02             # a move of prod's probability smaller than this is no move
ELO_K = (("slow", 16.0), ("fast", 48.0))
QUARTER_SECONDS = 240.0
REGULATION = 4 * QUARTER_SECONDS
EVENS = 0.02                 # a pre-match price this near 50% has no favourite
MIN_BETS = 50
TOP = 25
HOME, AWAY = "home", "away"
SIDE = {50: HOME, 51: AWAY}
BACKED, OPPONENT = "backed side", "opponent"
SHORT = {"favourite": "fav", "underdog": "dog", "evens": "even"}
BALL_MOMENTS = (bet_moments.BETWEEN, bet_moments.LIVE, bet_moments.TIMEOUT)
KICK_MOMENTS = (bet_moments.SCORED_LIVE, bet_moments.CONV_NEXT, bet_moments.CONV_LIVE,
                bet_moments.KICK_NEXT, bet_moments.KICK_LIVE)
SEEN_STATUS = {"SECOND_QUARTER_STARTED": "q2", "THIRD_QUARTER_STARTED": "q3",
               "FOURTH_QUARTER_STARTED": "q4"}

LEADS = ("down 9+", "down 1-8", "level", "up 1-8", "up 9+")
P_EDGES = (0.10, 0.25, 0.40, 0.60, 0.75, 0.90)
RUNNING = ("under 7+", "under 3-7", "on pace", "over 3-7", "over 7+")
ZONES = ("own <20", "own 20-49", "opp 50-21", "red zone")
STAKES = (10, 50, 200, 1000)


def other(side):
    return AWAY if side == HOME else HOME


def tag(k):
    """-60 -> 'm60', 60 -> 'p60'."""
    return f"m{-k}" if k < 0 else f"p{k}"


def _int(v):
    try:
        return None if v is None or v == "" else int(float(v))
    except (TypeError, ValueError):
        return None


def band(x, edges, fmt="{:g}"):
    """A number as a band off ascending edges: '<10', '10-50', ... '1000+'."""
    if x is None:
        return None
    lo = None
    for e in edges:
        if x < e:
            return f"<{fmt.format(e)}" if lo is None else f"{fmt.format(lo)}-{fmt.format(e)}"
        lo = e
    return f"{fmt.format(lo)}+"


def p_band(p):
    return None if p is None else band(100 * p, [100 * e for e in P_EDGES], "{:.0f}") + "%"


def lead_band(lead):
    if lead is None:
        return None
    if lead <= -9:
        return LEADS[0]
    if lead < 0:
        return LEADS[1]
    if lead == 0:
        return LEADS[2]
    return LEADS[3] if lead <= 8 else LEADS[4]


def running_band(x):
    """Points on the board against prod's pre-match total at this stage of the game."""
    if x is None:
        return None
    if x <= -7:
        return RUNNING[0]
    if x <= -3:
        return RUNNING[1]
    if x < 3:
        return RUNNING[2]
    return RUNNING[3] if x < 7 else RUNNING[4]


def zone(field):
    """Field position (yards from the offence's own goal) as a zone of the field."""
    if field is None:
        return None
    return ZONES[0] if field < 20 else ZONES[1] if field < 50 else ZONES[2] if field < 80 else ZONES[3]


def down_distance(down, distance):
    if down is None or not 1 <= down <= 4:
        return None
    to_go = ("short" if distance is not None and distance <= 3 else
             "medium" if distance is not None and distance <= 6 else "long")
    return f"{('1st', '2nd', '3rd', '4th')[down - 1]} & {to_go}"


def favourite_of(pre_p):
    """favourite / underdog / evens off prod's pre-match probability of the side."""
    if pre_p is None:
        return None
    return "evens" if abs(pre_p - 0.5) < EVENS else "favourite" if pre_p > 0.5 else "underdog"


def elapsed(quarter, clock_left):
    """The share of regulation played, off the quarter label (Q1 .. Q4, OT) and the quarter's
    clock left."""
    if quarter == "OT":
        return 1.0
    if not quarter or not quarter.startswith("Q") or clock_left is None:
        return None
    q = int(quarter[1:])
    played = (q - 1) * QUARTER_SECONDS + (QUARTER_SECONDS - min(QUARTER_SECONDS, float(clock_left)))
    return max(0.0, min(1.0, played / REGULATION))


# --- prices and ratings ----------------------------------------------------------------------

def prematch_p1(tl, match, first_play=None, kickoff=None):
    """prod's last pre-match probability of player 1, or None. With the match's first play or its
    kick-off: the closing quote (bets.closing_quote -- a message below the first play counts as
    pre-match). Without: its last quote with no message at all, else its first quote."""
    entry = tl.get((match, 50))
    if not entry or not entry[1]:
        return None
    if first_play is not None or kickoff is not None:
        q = bets.closing_quote(tl, match, 50, first_play, kickoff)
        return None if q is None else q[1]
    quotes = entry[1]
    pre = [q for q in quotes if q[0] is None]
    return (pre[-1] if pre else quotes[0])[1]


def elo_by_match(history, ks=ELO_K):
    """match -> {name: player 1's win probability} off a walk-forward Elo of every gamer's wins (a
    draw half each), one rating per K: slow follows the long run, fast the recent form. Nothing
    after the match is read."""
    games = sorted(history, key=lambda r: str(r.get("SCHEDULED_START_TIME_UTC"))[:19])
    ratings = {name: defaultdict(float) for name, _ in ks}
    out = {}
    for r in games:
        s1, s2 = _int(r.get("PLAYER_1_FINAL_SCORE")), _int(r.get("PLAYER_2_FINAL_SCORE"))
        a, b = r.get("PLAYER_1_HANDLE"), r.get("PLAYER_2_HANDLE")
        if s1 is None or s2 is None or not a or not b:
            continue
        res = 1.0 if s1 > s2 else 0.0 if s1 < s2 else 0.5
        out[r["MATCH_CODE"]] = {}
        for name, k in ks:
            ra, rb = ratings[name][a], ratings[name][b]
            p = 1 / (1 + 10 ** ((rb - ra) / 400))
            out[r["MATCH_CODE"]][name] = p
            ratings[name][a] += k * (res - p)
            ratings[name][b] -= k * (res - p)
    return out


# --- the game at the bet ---------------------------------------------------------------------

def side_of_team(team, a_side):
    """TEAM_A / TEAM_B as home / away, given TEAM_A's side."""
    if team not in ("TEAM_A", "TEAM_B") or a_side not in (HOME, AWAY):
        return None
    return a_side if team == "TEAM_A" else other(a_side)


def snapshot_index(snapshots):
    """match -> (sorted messages, its PLAY_OVER snapshots in message order)."""
    out = {}
    for m, snaps in snapshots.items():
        s = sorted((x for x in snaps if x.get("message") is not None), key=lambda x: int(x["message"]))
        out[m] = ([int(x["message"]) for x in s], s)
    return out


def snapshot_at(index, match, message):
    """The latest PLAY_OVER snapshot at or before the message, or None."""
    entry = index.get(match)
    if entry is None or message is None:
        return None
    msgs, snaps = entry
    i = bisect_right(msgs, int(message))
    return snaps[i - 1] if i else None


def feed_message(tl, t):
    """The feed's latest message at time t (on prod's clock), or None."""
    if tl is None or t is None:
        return None
    j = bisect_right(tl.times, bets.epoch(t)) - 1
    return tl.msgs[j] if j >= 0 and tl.times[j] > -math.inf else None


def ball(snap, backed, moment):
    """Who had the ball at the bet and where, from the backed side's point of view: the next snap
    after the latest PLAY_OVER (the line of scrimmage on the screen, or the play live off it)."""
    out = dict(ball=None, zone=None, territory=None, has_ball=None, down=None, down_dist=None)
    if moment in KICK_MOMENTS:
        out["ball"] = "score, kick-off or conversion"
        return out
    if moment not in BALL_MOMENTS:
        out["ball"] = "break" if moment in (bet_moments.BREAK, bet_moments.INTERRUPTION) else "unknown"
        return out
    if snap is None:
        out["ball"] = "unknown"
        return out
    team = snap.get("next_offense") or snap.get("offense")
    field = _int(snap.get("next_field_position") if snap.get("next_offense") else snap.get("field_position"))
    down = _int(snap.get("next_down") if snap.get("next_offense") else snap.get("down"))
    dist = _int(snap.get("next_distance") if snap.get("next_offense") else snap.get("distance"))
    side = side_of_team(team, snap.get("team_a_side"))
    if side is None:
        out["ball"] = "unknown"
        return out
    mine = side == backed
    out.update(ball=BACKED if mine else OPPONENT, has_ball=int(mine), down=down,
               down_dist=down_distance(down, dist))
    if field is not None:
        out["zone"] = f"{'backed' if mine else 'opp'}: {zone(field)}"
        out["territory"] = field if mine else 100 - field
    return out


def score_change(checks, match, message, forward=True):
    """(side that scored: home / away / None, prod's message of it) of the next score change after
    this message (forward) or the last one at or before it; (None, None) where there is none."""
    entry = checks.scores.get(match)
    if entry is None or message is None:
        return None, None
    smsgs, vals = entry
    now = checks.score_at(match, message)
    k = bisect_right(smsgs, int(message))
    if forward:
        while k < len(smsgs) and vals[k] == now:
            k += 1
        if k >= len(smsgs):
            return None, None
        before, after = now, vals[k]
    else:
        k -= 1
        while k >= 0 and (vals[k - 1] if k else (0, 0)) == vals[k]:
            k -= 1
        if k < 0:
            return None, None
        before, after = (vals[k - 1] if k else (0, 0)), vals[k]
    side = HOME if after[0] > before[0] else AWAY if after[1] > before[1] else None
    return side, smsgs[k]


def pace_index(rows):
    """(scrimmage plays' start messages, real seconds before each snap or None) of one match off
    its scouting rows."""
    scrim = [p for p in totals_signals.plays(rows) if p.kind == "scrimmage"]
    gaps = [None]
    for a, b in zip(scrim, scrim[1:]):
        g = (b.start_t - a.over_t).total_seconds() if a.over_t and b.start_t and a.period == b.period else None
        gaps.append(g if g is not None and 0 <= g <= totals_signals.PRESNAP_MAX else None)
    return [p.start_msg for p in scrim], gaps


class Context:
    """What every bet and match is read against."""

    def __init__(self, prod_tl, checks, snapshots, scouting_rows, info, pos, elo, cands=()):
        self.tl, self.checks, self.info, self.pos, self.elo = prod_tl, checks, info, pos, elo
        self.snaps = snapshot_index(snapshots)
        self.cands = list(cands)
        by = defaultdict(list)
        for r in scouting_rows:
            if r[1] is not None:
                by[r[0]].append(r)
        self.rows = by
        self._pace, self._pre, self._line = {}, {}, {}

    def pre_p1(self, match):
        if match not in self._pre:
            self._pre[match] = prematch_p1(self.tl, match)
        return self._pre[match]

    def total_line(self, match):
        """prod's closing pre-match total line (its last before the kick-off), or None."""
        if match not in self._line:
            ko = kickoff_time(self.checks, match)
            q = bets.price_at_time(self.tl, match, 54, ko - dt.timedelta(milliseconds=1)) if ko else None
            self._line[match] = q[2] if q else None
        return self._line[match]

    def pace(self, match):
        if match not in self._pace:
            self._pace[match] = pace_index(self.rows.get(match, []))
        return self._pace[match]

    def handles(self, match, side):
        i = self.info.get(match) or {}
        n, o = ("1", "2") if side == HOME else ("2", "1")
        return (i.get(f"PLAYER_{n}_HANDLE") or None, i.get(f"PLAYER_{o}_HANDLE") or None,
                i.get(f"PLAYER_{n}_TEAM") or None, i.get(f"PLAYER_{o}_TEAM") or None)


def session_label(mine, theirs):
    """The match's place in the gamers' sessions, from the backed side's point of view."""
    if mine is None or theirs is None:
        return "unknown"
    a, b = mine[1] == 1, theirs[1] == 1
    return ("both on their last" if a and b else "backed on his last" if a
            else "opponent on his last" if b else "neither on his last")


def bet_row(b, r, ctx, cand_rows=()):
    """One moneyline bet: who and what it backed, prod's price around it, and the game at the
    moment it was struck (in play)."""
    side, match = SIDE[b.feed_market], b.match_code
    backed, opposed, team, opp_team = ctx.handles(match, side)
    p = r.get("stream_prob")
    won = {bets.WON: 1.0, bets.LOST: 0.0}.get(r.get("result"))
    out = dict(match_code=match, bet_id=b.bet_id, bet_time=b.time, side=side, group=group_of(b),
               operator=b.operator, customer=b.extra.get("CUSTOMER_NAME_HASH"), stake=b.stake,
               revenue=b.revenue, odds=b.odds, result=r.get("result"), won=won, in_play=b.in_play,
               phase=phase_of(b) or "unknown", match_phase=r.get("match_phase"),
               excluded=bool(r.get("excluded")), p=p, message=r.get("message"),
               res=None if won is None or p is None else won - p,
               backed=backed, opposed=opposed, backed_team=team, opposed_team=opp_team,
               stake_band=band(b.stake, STAKES), log_stake=math.log10(b.stake) if b.stake > 0 else None)
    out["expected"] = 1 - b.odds * p if p is not None and b.odds else None
    for name, cr in cand_rows:
        c = cr.get("candidate_prob")
        ok = c is not None and cr.get("candidate_ok") is not False \
            and not (cr.get("state_required") and not cr.get("state_ok"))
        out[f"p_{name}"] = c if ok else None
        out[f"res_{name}"] = won - c if ok and won is not None else None
    pre = ctx.pre_p1(match)
    pre_p = None if pre is None else pre if side == HOME else 1 - pre
    out.update(pre_p=pre_p, fav=favourite_of(pre_p),
               drift=None if p is None or pre_p is None else p - pre_p)
    e = ctx.elo.get(match)
    for name, _ in ELO_K:
        ep = None if not e else e[name] if side == HOME else 1 - e[name]
        out[f"elo_{name}_gap"] = None if ep is None or pre_p is None else ep - pre_p
    mine, theirs = ctx.pos.get((backed, match)), ctx.pos.get((opposed, match))
    out.update(backed_session_match=mine[0] if mine else None,
               opp_session_match=theirs[0] if theirs else None,
               session=session_label(mine, theirs))
    start = _start((ctx.info.get(match) or {}).get("SCHEDULED_START_TIME_UTC"))
    out["hour_utc"] = start.hour if start else None
    seen = None if b.time is None else b.time - dt.timedelta(seconds=r.get("latency_seconds") or 0)
    for k in OFFSETS:
        q = bets.price_at_time(ctx.tl, match, b.feed_market, seen + dt.timedelta(seconds=k)) if seen else None
        d = None if q is None or p is None else (p - q[1] if k < 0 else q[1] - p)
        out[f"{'into' if k < 0 else 'after'}_{tag(k)}"] = d
    out["before"] = _move(out["into_m60"], "followed a move", "faded a move")
    out["after"] = _move(out["after_p60"], "their way", "against them")
    if not b.in_play:
        ko = kickoff_time(ctx.checks, match)
        q = bets.price_at_time(ctx.tl, match, b.feed_market, ko - dt.timedelta(milliseconds=1)) \
            if ko and seen and ko > seen else None
        out["to_close"] = None if q is None or p is None else q[1] - p
        out["closing"] = _move(out["to_close"], "closed their way", "closed against them")
        return out
    tl = ctx.checks.timelines.get(match)
    mo = bet_moments.moment_of(tl, ctx.checks, match, b.time, r.get("message"))
    out.update(moment=mo["moment"] or "unknown", quarter=mo["quarter"], clock_left=mo["clock_left"],
               price_age_seconds=mo["price_age_seconds"])
    fm = feed_message(tl, b.time)
    at = fm if fm is not None else r.get("message")
    score = ctx.checks.score_at(match, at) if at is not None else None
    if score is not None:
        lead = (score[0] - score[1]) * (1 if side == HOME else -1)
        out.update(lead=lead, lead_band=lead_band(lead), board=score[0] + score[1])
    el = elapsed(mo["quarter"], mo["clock_left"])
    line = ctx.total_line(match)
    out["elapsed"] = el
    out["running"] = None if el is None or line is None or score is None else sum(score) - line * el
    out["running_band"] = running_band(out["running"])
    out.update(ball(snapshot_at(ctx.snaps, match, at), side, mo["moment"]))
    nxt, msg = score_change(ctx.checks, match, at)
    t_next = ctx.checks.time_of(match, msg) if msg is not None else None
    out["next_score_s"] = (t_next - b.time).total_seconds() if t_next and b.time else None
    if msg is None:
        out["next_scorer"] = "none" if at is not None and match in ctx.checks.scores else None
    else:       # a score taken off the board (a correction) has no scorer
        out["next_scorer"] = None if nxt is None else BACKED if nxt == side else OPPONENT
    out["next_score"] = next_label(out["next_scorer"], out["next_score_s"])
    _, last = score_change(ctx.checks, match, at, forward=False)
    t_last = ctx.checks.time_of(match, last) if last is not None else kickoff_time(ctx.checks, match)
    out["since_score_s"] = (b.time - t_last).total_seconds() if t_last and b.time else None
    starts, gaps = ctx.pace(match)
    n = bisect_right(starts, int(at)) if at is not None else 0
    so_far = [g for g in gaps[:n] if g is not None]
    out["plays_so_far"] = n if starts else None
    out["clock_per_play"] = (el * REGULATION / n) if el is not None and n else None
    out["presnap_s"] = statistics.median(so_far) if len(so_far) >= 3 else None
    return out


def _move(d, up, down):
    if d is None:
        return None
    return up if d >= MOVE_FLAT else down if d <= -MOVE_FLAT else "held"


def next_label(scorer, secs):
    if scorer is None:
        return None
    if scorer == "none" or secs is None:
        return "no more scoring"
    when = "within 30s" if secs < 30 else "30-120s" if secs < 120 else "after 120s"
    return f"{'backed' if scorer == BACKED else 'opponent'} {when}"


# --- runs and pile-ons -----------------------------------------------------------------------

def runs(rows, gap=BURST_GAP):
    """[[row]]: each match's moneyline bets on one side (pre-match and in play apart), in time
    order, split where one comes more than `gap` seconds after the last."""
    by = defaultdict(list)
    for r in rows:
        if r.get("bet_time") is not None:
            by[(r["match_code"], r["side"], bool(r["in_play"]))].append(r)
    out = []
    for rs in by.values():
        rs.sort(key=lambda r: r["bet_time"])
        run = [rs[0]]
        for a, b in zip(rs, rs[1:]):
            if (b["bet_time"] - a["bet_time"]).total_seconds() > gap:
                out.append(run)
                run = [b]
            else:
                run.append(b)
        out.append(run)
    return out


def pile_ons(all_runs, quantile=PILE_QUANTILE, min_bets=PILE_MIN_BETS):
    """(the runs that are pile-ons, the stake a run needs to be one)."""
    stakes = sorted(sum(r["stake"] for r in run) for run in all_runs)
    if not stakes:
        return [], None
    cut = stakes[min(len(stakes) - 1, int(len(stakes) * quantile))]
    return [run for run in all_runs if len(run) >= min_bets and sum(r["stake"] for r in run) >= cut], cut


PILE_KEYS = ("phase", "quarter", "clock_left", "lead", "lead_band", "fav", "pre_p", "ball", "zone",
             "down_dist", "moment", "running", "running_band", "session", "next_scorer",
             "next_score", "next_score_s", "after_p60", "after_p300", "into_m60", "backed",
             "opposed", "backed_team", "opposed_team")


def pile_row(run, n):
    """One pile-on: its first bet's situation, its money, what came next and the result."""
    first = run[0]
    stake = sum(r["stake"] for r in run)
    priced = [r for r in run if r["p"] is not None]
    ps = sum(r["stake"] for r in priced)
    out = dict(pile=n, match_code=first["match_code"], side=first["side"], in_play=first["in_play"],
               start=first["bet_time"], seconds=(run[-1]["bet_time"] - first["bet_time"]).total_seconds(),
               bets=len(run), customers=len({r["customer"] for r in run}), stake=stake,
               revenue=sum(r["revenue"] for r in run), won=first["won"], p_first=first["p"],
               p_avg=sum(r["stake"] * r["p"] for r in priced) / ps if ps else None,
               restricted=sum(r["stake"] for r in run if r["group"] == "Restricted") / stake if stake else None,
               vip=sum(r["stake"] for r in run if r["group"] == "VIP") / stake if stake else None)
    out.update({k: first.get(k) for k in PILE_KEYS})
    return out


# --- matches ---------------------------------------------------------------------------------

def lead_changes(vals):
    """How often the lead changed hands over a match's cumulative scores (a tie in between is not a
    change)."""
    last, n = 0, 0
    for p1, p2 in vals:
        s = (p1 > p2) - (p1 < p2)
        if s and last and s != last:
            n += 1
        if s:
            last = s
    return n


def match_row(match, final, ctx, ml, others):
    """One match: its moneyline money and the book's result, prod's prices against how it played
    out, the pace of Q1 and the first half, and the gamers."""
    f1, f2 = _int(final[0]), _int(final[1])
    winner = HOME if f1 > f2 else AWAY if f2 > f1 else None
    pre = ctx.pre_p1(match)
    fav = None if pre is None or abs(pre - 0.5) < EVENS else HOME if pre > 0.5 else AWAY
    line = ctx.total_line(match)
    rows = ctx.rows.get(match, [])
    cps = totals_signals.checkpoints(rows)
    i = ctx.info.get(match) or {}
    start = _start(i.get("SCHEDULED_START_TIME_UTC"))
    out = dict(match_code=match, start=start, hour_utc=start.hour if start else None,
               weekday=start.strftime("%a") if start else None, stream=i.get("STREAM_NUMBER"),
               home=i.get("PLAYER_1_HANDLE"), away=i.get("PLAYER_2_HANDLE"),
               home_team=i.get("PLAYER_1_TEAM"), away_team=i.get("PLAYER_2_TEAM"),
               final_home=f1, final_away=f2, final_total=f1 + f2, margin=abs(f1 - f2),
               pre_p1=pre, fav_p=None if pre is None else max(pre, 1 - pre),
               fav_won=None if fav is None or winner is None else int(fav == winner),
               total_line=line, total_vs_line=None if line is None else f1 + f2 - line,
               overtime=int("OVERTIME_1_STARTED" in cps), close=int(abs(f1 - f2) <= 3),
               blowout=int(abs(f1 - f2) >= 17))
    out["dog_won"] = None if out["fav_won"] is None else 1 - out["fav_won"]
    entry = ctx.checks.scores.get(match)
    vals = entry[1] if entry else []
    out["lead_changes"] = lead_changes(vals)
    half = cps.get("THIRD_QUARTER_STARTED")
    h = ctx.checks.score_at(match, half) if half is not None else None
    leader = None if h is None or h[0] == h[1] else HOME if h[0] > h[1] else AWAY
    out["half_points"] = sum(h) if h else None
    out["comeback"] = None if leader is None or winner is None else int(leader != winner)
    q2 = cps.get("SECOND_QUARTER_STARTED")
    out["q1_points"] = sum(ctx.checks.score_at(match, q2)) if q2 is not None else None
    q4 = cps.get("FOURTH_QUARTER_STARTED")
    if q4 is not None and entry:
        k = bisect_right(entry[0], q4)
        out["q4_lead_change"] = int(lead_changes(vals[max(0, k - 1):]) > 0)
    wm = {HOME: 50, AWAY: 51}.get(winner)
    for status, key in SEEN_STATUS.items():
        t = ctx.checks.time_of(match, cps[status]) if status in cps else None
        q = bets.price_at_time(ctx.tl, match, wm, t) if t and wm else None
        out[f"winner_p_{key}"] = q[1] if q else None
    tl_w = ctx.tl.get((match, wm)) if wm else None
    inplay = [q[1] for q in tl_w[1] if q[0] is not None and q[1] is not None] if tl_w else []
    out["winner_min_p"] = min(inplay) if inplay else None
    out["winner_pre_p"] = None if pre is None or winner is None else pre if winner == HOME else 1 - pre
    called = totals_signals.timeouts(rows)
    ps = totals_signals.plays(rows)
    for seg, periods in (("q1", (1,)), ("h1", (1, 2))):
        sf = totals_signals.segment_features(ps, periods, called)
        for k in ("plays", "presnap_s", "play_s", "clock_per_play", "yards_per_play", "punts", "tds"):
            out[f"{seg}_{k}"] = sf.get(k)
    e = ctx.elo.get(match)
    for name, _ in ELO_K:
        out[f"elo_{name}_p1"] = e[name] if e else None
        out[f"elo_{name}_vs_fav"] = (None if not e or fav is None
                                     else (e[name] if fav == HOME else 1 - e[name]) - out["fav_p"])
    p1, p2 = ctx.pos.get((out["home"], match)), ctx.pos.get((out["away"], match))
    out["session_match"] = max(p1[0], p2[0]) if p1 and p2 else None
    out["last_any"] = None if not (p1 and p2) else int(p1[1] == 1 or p2[1] == 1)
    out["last_both"] = None if not (p1 and p2) else int(p1[1] == 1 and p2[1] == 1)
    out.update(match_money(ml, winner, fav))
    for mk in (markets.SPREAD, markets.TOTAL):
        bs = [b for b in others if markets.market_group(b.feed_market) == mk]
        out[f"{mk}_stake"] = sum(b.stake for b in bs)
        out[f"{mk}_revenue"] = sum(b.revenue for b in bs)
    out["all_revenue"] = out["ml_revenue"] + out["spread_revenue"] + out["total_revenue"]
    return out


def match_money(ml, winner, fav):
    """The match's moneyline money: bets, stake, the book's revenue, who it came from and where."""
    stake = sum(r["stake"] for r in ml)
    out = dict(ml_bets=len(ml), ml_stake=stake, ml_revenue=sum(r["revenue"] for r in ml),
               ml_customers=len({r["customer"] for r in ml if r["customer"]}))
    out["ml_margin"] = out["ml_revenue"] / stake if stake else None
    share = lambda rs: sum(r["stake"] for r in rs) / stake if stake else None
    by_cust = Counter()
    for r in ml:
        by_cust[r["customer"]] += r["stake"]
    out.update(in_play_share=share([r for r in ml if r["in_play"]]),
               restricted_share=share([r for r in ml if r["group"] == "Restricted"]),
               vip_share=share([r for r in ml if r["group"] == "VIP"]),
               top_customer_share=(by_cust.most_common(1)[0][1] / stake) if stake else None,
               winner_share=share([r for r in ml if r["side"] == winner]) if winner else None,
               dog_share=share([r for r in ml if fav and r["side"] != fav]) if fav else None,
               pile_on_share=share([r for r in ml if r.get("pile_on") is not None]))
    on_w = [r for r in ml if r["side"] == winner and r["in_play"] and r["p"] is not None]
    ws = sum(r["stake"] for r in on_w)
    out["winner_bet_p"] = sum(r["stake"] * r["p"] for r in on_w) / ws if ws else None
    return out


def classify(rows, worst=WORST_SHARE):
    """Each match with moneyline stake marked 'worst', 'losing' (the book down on it, past the
    worst) or 'winning'; matches without any are 'no bets'."""
    bet = sorted((r for r in rows if r["ml_stake"]), key=lambda r: r["ml_revenue"])
    n = max(1, int(round(len(bet) * worst))) if bet else 0
    for k, r in enumerate(bet):
        r["class"] = "worst" if k < n and r["ml_revenue"] < 0 else "losing" if r["ml_revenue"] < 0 else "winning"
    for r in rows:
        r.setdefault("class", "no bets")
    return rows


# --- reporting -------------------------------------------------------------------------------

def _f(x, fmt="+.2f", w=7):
    return f"{'-':>{w}s}" if x is None else f"{x:{fmt}}".rjust(w)


def _pc(x, w=6, fmt=".1f"):
    return f"{'-':>{w}s}" if x is None else f"{100 * x:{fmt}}".rjust(w)


def stats(rs, cands=()):
    """The book's money and prod's price over some bets: bets, matches, stake, revenue, margin,
    expected margin, stake-weighted prod p and won, and the mean residual with 2se (by match)."""
    stake = sum(r["stake"] for r in rs)
    rev = sum(r["revenue"] for r in rs)
    priced = [r for r in rs if r["p"] is not None and r["won"] is not None and r["odds"]]
    ps = sum(r["stake"] for r in priced)
    n, m, se = mean_se(priced, "res")
    out = dict(bets=len(rs), matches=len({r["match_code"] for r in rs}), stake=stake, revenue=rev,
               margin=rev / stake if stake else None,
               expected=sum(r["stake"] * r["expected"] for r in priced) / ps if ps else None,
               p=sum(r["stake"] * r["p"] for r in priced) / ps if ps else None,
               won=sum(r["stake"] * r["won"] for r in priced) / ps if ps else None,
               res=m, se=None if se is None else 2 * se, priced=n)
    for c in cands:
        _, cm, cse = mean_se([r for r in priced if r.get(f"res_{c}") is not None], f"res_{c}")
        out[f"res_{c}"] = cm
    return out


def head(width=30, cands=()):
    return (f"  {'':{width}s} {'bets':>6s} {'mtch':>5s} {'stake':>10s} {'share':>6s} {'margin':>7s} "
            f"{'expect':>7s} {'prod p':>6s} {'won':>6s} {'resid':>6s} {'2se':>5s}"
            + "".join(f" {('res ' + c)[:7]:>7s}" for c in cands))


def stat_line(label, s, total, width=30, cands=()):
    return (f"  {str(label)[:width]:{width}s} {s['bets']:6,d} {s['matches']:5,d} {s['stake']:10,.0f} "
            f"{_pc(s['stake'] / total if total else None)} {_pc(s['margin'], 7)} {_pc(s['expected'], 7)} "
            f"{_pc(s['p'])} {_pc(s['won'])} {_pc(s['res'], 6, '+.1f')} {_pc(s['se'], 5)}"
            + "".join(f" {_pc(s.get(f'res_{c}'), 7, '+.1f')}" for c in cands))


def cut(rows, key, title, order=None, min_bets=MIN_BETS, cands=(), width=30, top=None):
    """Lines: the bets grouped by key (a field or a function), costliest to the book first unless
    an order is given."""
    groups = defaultdict(list)
    for r in rows:
        v = key(r) if callable(key) else r.get(key)
        if v is not None:
            groups[v].append(r)
    total = sum(r["stake"] for r in rows)
    keep = {k: stats(rs, cands) for k, rs in groups.items() if len(rs) >= min_bets}
    if not keep:
        return []
    if order:
        ks = [k for k in order if k in keep] + sorted((k for k in keep if k not in order), key=str)
    else:
        ks = sorted(keep, key=lambda k: keep[k]["revenue"])
    if top:
        ks = ks[:top]
    return [f"\n  {title}", head(width, cands)] + [stat_line(k, keep[k], total, width, cands) for k in ks]


NUMERIC = (
    ("p", "prod's probability of the selection"),
    ("pre_p", "prod's pre-match probability of the backed side"),
    ("drift", "prod's p less its pre-match p (+ backed side gone up)"),
    ("lead", "the backed side's lead"),
    ("elapsed", "share of regulation played"),
    ("running", "points against prod's pre-match total so far (+ over)"),
    ("has_ball", "the backed side has the ball (1)"),
    ("territory", "the backed side's field position (its own goal 0)"),
    ("down", "the down (either side)"),
    ("plays_so_far", "scrimmage plays so far"),
    ("clock_per_play", "game-clock seconds per play so far"),
    ("presnap_s", "real seconds between plays so far (median)"),
    ("since_score_s", "seconds since the last score"),
    ("into_m60", "prod's move into the bet over 60s (+ their way)"),
    ("into_m30", "prod's move into the bet over 30s (+ their way)"),
    ("after_p30", "prod's move in the 30s after (+ their way)"),
    ("after_p60", "prod's move in the 60s after (+ their way)"),
    ("price_age_seconds", "seconds from prod's price to the bet"),
    ("elo_fast_gap", "fast Elo less prod's pre-match p (backed side)"),
    ("elo_slow_gap", "slow Elo less prod's pre-match p (backed side)"),
    ("backed_session_match", "backed gamer's match of the session"),
    ("opp_session_match", "opponent's match of the session"),
    ("log_stake", "log10 stake"),
    ("hour_utc", "hour of the kick-off (UTC)"),
)
PRE_NUMERIC = ("p", "elo_fast_gap", "elo_slow_gap", "backed_session_match", "opp_session_match",
               "log_stake", "hour_utc", "to_close")


def ranking(rows, features):
    """[(key, title, bets, corr, resid in the bottom bin, its 2se, top bin, 2se, t, edges)] for each
    feature, sorted by how far the residual moves from the bottom bin to the top (t: the move over
    its standard error)."""
    out = []
    for key, title in features:
        have = [r for r in rows if r.get(key) is not None and r.get("res") is not None]
        if len(have) < 4 * MIN_BETS:
            continue
        edges = totals_signals.edges_for([r[key] for r in have])
        bins = defaultdict(list)
        for r in have:
            bins[totals_signals.bin_of(r[key], edges)].append(r)
        filled = sorted(bins)
        if len(filled) < 2:
            continue
        _, lo, lse = mean_se(bins[filled[0]], "res")
        _, hi, hse = mean_se(bins[filled[-1]], "res")
        t = None if lse is None or hse is None or not (lse or hse) else (hi - lo) / math.hypot(lse, hse)
        corr = totals_signals._corr([(r[key], r["res"]) for r in have])
        out.append((key, title, len(have), corr, lo, None if lse is None else 2 * lse, hi,
                    None if hse is None else 2 * hse, t, edges))
    return sorted(out, key=lambda x: -abs(x[8] or 0))


def ranking_lines(rows, features, label, detail=8, cands=()):
    ranked = ranking(rows, features)
    if not ranked:
        return []
    L = [f"\n  {label}: every feature cut in quintiles (one bin per value for a count); resid = won "
         "less prod's p, so 0 where prod is right; t = (top bin - bottom bin) over its se",
         f"  {'feature':52s} {'bets':>7s} {'corr':>6s} {'res lo':>7s} {'2se':>5s} {'res hi':>7s} "
         f"{'2se':>5s} {'t':>6s}"]
    for key, title, n, corr, lo, lse, hi, hse, t, _ in ranked:
        L.append(f"  {title[:52]:52s} {n:7,d} {_f(corr, '+.2f', 6)} {_pc(lo, 7, '+.1f')} {_pc(lse, 5)} "
                 f"{_pc(hi, 7, '+.1f')} {_pc(hse, 5)} {_f(t, '+.1f', 6)}")
    for key, title, n, _, _, _, _, _, t, edges in ranked[:detail]:
        have = [r for r in rows if r.get(key) is not None]

        def which(r, edges=edges, key=key):
            q = totals_signals.bin_of(r[key], edges)
            lo_ = f"{edges[q - 1]:.3g}" if q else "lowest"
            hi_ = f"{edges[q]:.3g}" if q < len(edges) else "highest"
            return f"{q + 1}: {lo_} to {hi_}"
        order = sorted({which(r) for r in have})
        L += cut(have, which, f"{title} (t {_f(t, '+.1f', 0)})", order=order, min_bets=1, cands=cands)
    return L


MATCH_FEATURES = (
    ("ml_stake", "moneyline stake"), ("ml_bets", "moneyline bets"), ("ml_customers", "customers"),
    ("in_play_share", "share of the stake in play"), ("restricted_share", "share restricted"),
    ("vip_share", "share VIP"), ("top_customer_share", "biggest customer's share"),
    ("pile_on_share", "share of the stake in pile-ons"),
    ("winner_share", "share on the eventual winner"), ("dog_share", "share on prod's underdog"),
    ("winner_bet_p", "prod's p of the winner when backed in play"),
    ("fav_p", "prod's pre-match favourite's p"), ("dog_won", "underdog won"),
    ("winner_pre_p", "prod's pre-match p of the winner"), ("winner_min_p", "winner's lowest p in play"),
    ("winner_p_q2", "winner's p at the start of Q2"), ("winner_p_q3", "winner's p at half time"),
    ("winner_p_q4", "winner's p at the start of Q4"),
    ("comeback", "half-time leader lost"), ("lead_changes", "lead changes"),
    ("q4_lead_change", "lead changed in Q4"), ("overtime", "overtime"), ("close", "decided by 3 or less"),
    ("blowout", "decided by 17+"), ("margin", "final margin"),
    ("final_total", "final total"), ("total_line", "prod's closing total line"),
    ("total_vs_line", "final total less prod's line (- low scorer)"),
    ("q1_points", "points in Q1"), ("half_points", "points at half time"),
    ("q1_plays", "Q1 scrimmage plays"), ("q1_presnap_s", "Q1 real seconds between plays"),
    ("q1_clock_per_play", "Q1 game clock per play"), ("q1_yards_per_play", "Q1 yards per play"),
    ("h1_plays", "H1 scrimmage plays"), ("h1_presnap_s", "H1 real seconds between plays"),
    ("h1_clock_per_play", "H1 game clock per play"), ("h1_yards_per_play", "H1 yards per play"),
    ("h1_punts", "H1 punts"), ("h1_tds", "H1 touchdowns"),
    ("elo_fast_vs_fav", "fast Elo less prod (favourite's side)"),
    ("elo_slow_vs_fav", "slow Elo less prod (favourite's side)"),
    ("session_match", "match of the session (later gamer)"), ("last_any", "a gamer on his last"),
    ("last_both", "both on their last"), ("hour_utc", "kick-off hour UTC"),
    ("spread_revenue", "the book on the handicap"), ("total_revenue", "the book on the total"),
)


def _ms(xs):
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    return m, (statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None), len(xs)


def compare_lines(columns, a, b, diff_label):
    """Lines: feature by feature, the mean over each (label, matches) column, and the columns a
    against b (their difference, and z: it over its se)."""
    L = [f"\n  {'feature':44s} " + " ".join(f"{label[:9]:>9s}" for label, _ in columns)
         + f" {diff_label[:9]:>9s} {'z':>6s}"]
    for key, title in MATCH_FEATURES:
        means = [_ms([r[key] for r in rs if r.get(key) is not None]) for _, rs in columns]
        (am, ase, _), (bm, bse, _) = means[a], means[b]
        z = None if am is None or bm is None or not ase or not bse else (am - bm) / math.hypot(ase, bse)
        L.append(f"  {title[:44]:44s} " + " ".join(_f(m, ',.2f', 9) for m, _, _ in means)
                 + f" {_f(None if am is None or bm is None else am - bm, '+,.2f', 9)} {_f(z, '+.1f', 6)}")
    return L


def match_cut(mrows, key, title, order=None, min_matches=10):
    """Lines: the matches with moneyline stake grouped by key: the book's result and how much of
    it the losing ones carry."""
    groups = defaultdict(list)
    for r in mrows:
        if r["class"] == "no bets":
            continue
        v = key(r) if callable(key) else r.get(key)
        if v is not None:
            groups[v].append(r)
    ks = [k for k in (order or []) if k in groups] or sorted(groups, key=lambda k: sum(r["ml_revenue"] for r in groups[k]))
    L = [f"\n  {title}",
         f"  {'':28s} {'matches':>7s} {'stake':>11s} {'revenue':>10s} {'margin':>7s} {'losing':>7s} "
         f"{'worst':>6s} {'stake/m':>8s}"]
    for k in ks:
        rs = groups[k]
        if len(rs) < min_matches:
            continue
        stake = sum(r["ml_stake"] for r in rs)
        rev = sum(r["ml_revenue"] for r in rs)
        L.append(f"  {str(k)[:28]:28s} {len(rs):7,d} {stake:11,.0f} {rev:10,.0f} {_pc(rev / stake if stake else None, 7)} "
                 f"{_pc(sum(r['class'] != 'winning' for r in rs) / len(rs), 7)} "
                 f"{_pc(sum(r['class'] == 'worst' for r in rs) / len(rs))} {stake / len(rs):8,.0f}")
    return L


def pile_lines(piles, all_rows, cut_stake, cands=()):
    """Lines: how much is piled on, the situations it is piled on in against all the money, and the
    biggest pile-ons one by one."""
    if not piles:
        return ["\n  no pile-ons"]
    rows = [r for r in all_rows if r.get("pile_on") is not None]
    inplay = [r for r in all_rows if r["in_play"]]
    pin = [r for r in rows if r["in_play"]]
    s = stats(rows, cands)
    L = [f"\n  ==== 6. Pile-ons: runs of moneyline bets on one side, none more than {BURST_GAP}s apart, "
         f"with {PILE_MIN_BETS}+ bets and {cut_stake:,.0f}+ stake (the top {100 * (1 - PILE_QUANTILE):.0f}% "
         f"of runs) ====",
         f"  {len(piles):,} pile-ons on {len({p[0]['match_code'] for p in piles}):,} matches: "
         f"{s['bets']:,} bets, {s['stake']:,.0f} stake ({_pc(s['stake'] / sum(r['stake'] for r in all_rows)).strip()}% "
         f"of the moneyline), the book's margin {_pc(s['margin']).strip()}% against "
         f"{_pc(s['expected']).strip()}% expected; resid {_pc(s['res'], 0, '+.1f')} +-{_pc(s['se'], 0)}"]
    L.append("\n  In play, the situation at a pile-on's first bet against all the in-play money "
             "(lift: the pile-on stake's share over all the stake's share)")
    for key, title, order in (("quarter", "quarter", ("Q1", "Q2", "Q3", "Q4", "OT")),
                              ("lead_band", "backed side's lead", LEADS),
                              ("fav", "pre-match", ("favourite", "evens", "underdog")),
                              ("ball", "ball", None), ("zone", "where the ball was", None),
                              ("down_dist", "down and distance", None), ("moment", "the feed", None),
                              ("running_band", "points against prod's pre-match pace", RUNNING),
                              ("session", "session", None), ("before", "prod over the 60s before", None)):
        L += lift_lines(pin, inplay, key, title, order)
    L += cut(pin, "next_score", "In play, pile-on bets by who scored next and when", min_bets=1, cands=cands)
    L += cut(pin, "after", "In play, pile-on bets by prod's move in the 60s after", min_bets=1, cands=cands)
    biggest = sorted((pile_row(p, k) for k, p in enumerate(piles)), key=lambda x: -x["stake"])[:40]
    L.append(f"\n  The {len(biggest)} biggest pile-ons (score: the backed side's lead; ball: who had it "
             "and where; p: prod's at the first bet; +60/+300: prod's move after, their way +; next: "
             "who scored next, seconds)")
    L.append(f"  {'match':16s} {'start (UTC)':16s} {'ph':3s} {'clk':>4s} {'lead':>4s} {'pre':4s} "
             f"{'ball':24s} {'d&d':14s} {'p':>5s} {'+60':>5s} {'+300':>5s} {'next':>16s} {'won':>3s} "
             f"{'bets':>4s} {'cust':>4s} {'stake':>9s} {'revenue':>9s} {'rest':>5s}")
    for p in biggest:
        nxt = (f"{'B' if p['next_scorer'] == BACKED else 'O'} {p['next_score_s']:.0f}s"
               if p.get("next_score_s") is not None and p["next_scorer"] in (BACKED, OPPONENT)
               else (p.get("next_scorer") or "-"))
        L.append(f"  {str(p['match_code'])[:16]:16s} {str(p['start'])[:16]:16s} "
                 f"{(p.get('quarter') or 'pre')[:3]:3s} {_f(p.get('clock_left'), '.0f', 4)} "
                 f"{_f(p.get('lead'), '+d', 4)} {SHORT.get(p.get('fav'), '-'):4s} "
                 f"{str(p.get('zone') or p.get('ball') or '-')[:24]:24s} {str(p.get('down_dist') or '-')[:14]:14s} "
                 f"{_pc(p['p_first'], 5, '.0f')} {_pc(p.get('after_p60'), 5, '+.0f')} {_pc(p.get('after_p300'), 5, '+.0f')} "
                 f"{nxt[:16]:>16s} {('-' if p['won'] is None else 'Y' if p['won'] else 'N'):>3s} "
                 f"{p['bets']:4d} {p['customers']:4d} {p['stake']:9,.0f} {p['revenue']:9,.0f} "
                 f"{_pc(p['restricted'], 5, '.0f')}")
    return L


def lift_lines(piled, every, key, title, order=None):
    """Lines: per value of key, the pile-on stake's share against all the stake's share, and the
    pile-ons' resid there."""
    ps = sum(r["stake"] for r in piled)
    es = sum(r["stake"] for r in every)
    pv, ev, pr = Counter(), Counter(), defaultdict(list)
    for r in piled:
        if r.get(key) is not None:
            pv[r[key]] += r["stake"]
            pr[r[key]].append(r)
    for r in every:
        if r.get(key) is not None:
            ev[r[key]] += r["stake"]
    ks = [k for k in (order or []) if k in ev] or sorted(ev, key=lambda k: -ev[k])
    L = [f"  {title + ':':40s} {'piled':>7s} {'all':>7s} {'lift':>5s} {'pile resid':>10s} {'2se':>5s}"]
    for k in ks[:12]:
        n, m, se = mean_se([r for r in pr.get(k, []) if r.get("res") is not None], "res")
        if n < 10:
            m = se = None
        a, b = pv[k] / ps if ps else 0, ev[k] / es if es else 0
        L.append(f"    {str(k)[:38]:38s} {_pc(a, 7)} {_pc(b, 7)} {_f(a / b if b else None, '.2f', 5)} "
                 f"{_pc(m, 10, '+.1f')} {_pc(None if se is None else 2 * se, 5)}")
    return L


def customer_lines(rows, n=20):
    """Lines: the customers who took the most off the book on the moneyline, and how they bet."""
    by = defaultdict(list)
    for r in rows:
        if r["customer"]:
            by[r["customer"]].append(r)
    total = sum(r["revenue"] for r in rows)
    ranked = sorted(by.items(), key=lambda kv: sum(r["revenue"] for r in kv[1]))
    winners = [c for c, rs in ranked if sum(r["revenue"] for r in rs) < 0]
    top = sum(sum(r["revenue"] for r in rs) for _, rs in ranked[:n])
    L = [f"\n  ==== Customers: {len(by):,} on the moneyline, {len(winners):,} up on it; the top {n} "
         f"took {-top:,.0f} (the book's whole moneyline result {total:+,.0f}) ====",
         f"  {'customer':14s} {'group':10s} {'operator':14s} {'mtch':>4s} {'bets':>5s} {'stake':>9s} "
         f"{'revenue':>9s} {'margin':>7s} {'resid':>6s} {'2se':>5s} {'inply':>5s} {'dog':>5s} {'p':>5s} "
         f"{'lead':>5s} {'piled':>5s} {'ball':>5s} {'phase':6s}"]
    for c, rs in ranked[:n]:
        s = stats(rs)
        inp = [r for r in rs if r["in_play"]]
        lead = [r["lead"] for r in inp if r.get("lead") is not None]
        balls = [r for r in inp if r.get("has_ball") is not None]
        g = Counter(r["group"] for r in rs).most_common(1)[0][0]
        op = Counter(str(r["operator"]) for r in rs).most_common(1)[0][0]
        ph = Counter(r["phase"] for r in rs).most_common(1)[0][0]
        L.append(f"  {str(c)[:14]:14s} {g[:10]:10s} {op[:14]:14s} {s['matches']:4d} {s['bets']:5d} "
                 f"{s['stake']:9,.0f} {s['revenue']:9,.0f} {_pc(s['margin'], 7)} {_pc(s['res'], 6, '+.1f')} "
                 f"{_pc(s['se'], 5)} {_pc(len(inp) / len(rs), 5, '.0f')} "
                 f"{_pc(sum(r['fav'] == 'underdog' for r in rs) / len(rs), 5, '.0f')} {_pc(s['p'], 5, '.0f')} "
                 f"{_f(sum(lead) / len(lead) if lead else None, '+.1f', 5)} "
                 f"{_pc(sum(r.get('pile_on') is not None for r in rs) / len(rs), 5, '.0f')} "
                 f"{_pc(sum(r['has_ball'] for r in balls) / len(balls) if balls else None, 5, '.0f')} {ph[:6]:6s}")
    return L


def report(rows, mrows, piles, cut_stake, cands=()):
    """Lines of text: the whole sweep."""
    scope = [r for r in rows if not r["excluded"] and r["won"] is not None]
    inplay = [r for r in scope if r["in_play"]]
    pre = [r for r in scope if not r["in_play"]]
    all_stake = sum(r["stake"] for r in rows)
    all_rev = sum(r["revenue"] for r in rows)
    bet_m = [r for r in mrows if r["class"] != "no bets"]
    losing = [r for r in bet_m if r["class"] != "winning"]
    worst = [r for r in bet_m if r["class"] == "worst"]
    L = [f"  {len(rows):,} moneyline bets, {all_stake:,.0f} stake, the book {all_rev:+,.0f} "
         f"({_pc(all_rev / all_stake if all_stake else None).strip()}%) on {len(bet_m):,} of "
         f"{len(mrows):,} settled matches",
         f"  losing matches: {len(losing):,} ({_pc(len(losing) / len(bet_m) if bet_m else None).strip()}%), "
         f"{sum(r['ml_revenue'] for r in losing):+,.0f}; the worst {len(worst):,} "
         f"{sum(r['ml_revenue'] for r in worst):+,.0f}; the winning ones "
         f"{sum(r['ml_revenue'] for r in bet_m if r['class'] == 'winning'):+,.0f}",
         "  margin: the book's (revenue over stake); expect: what prod's price leaves it (1 - odds x "
         "prod p); prod p / won: stake-weighted; resid: won less prod p per bet, 0 where prod is "
         "right, + where the backed side wins more than prod says (2se clustered by match)"]
    L.append("\n  ==== 1. Where the moneyline money goes and what it costs ====")
    L += cut(rows, lambda r: f"{r['match_phase']}{' *' if r['excluded'] else ''}",
             "Every bet by when it was struck against the feed (the checks leave the starred ones "
             "out of everything below)", min_bets=1, cands=cands)
    L += cut(scope, "phase", "By phase", order=("pre-match", "Q1", "Q2", "Q3", "Q4", "OT"),
             min_bets=1, cands=cands)
    L += cut(scope, "group", "By customer temperature", min_bets=1, cands=cands)
    L += cut(scope, lambda r: str(r["operator"]), "By operator", min_bets=1, cands=cands)
    L += cut(scope, lambda r: f"{r['phase']} / {r['fav']}", "By phase and the side's pre-match "
             "standing", order=[f"{p} / {f}" for p in ("pre-match", "Q1", "Q2", "Q3", "Q4", "OT")
                                for f in ("favourite", "evens", "underdog")], cands=cands)
    L.append("\n  ==== 2. The losing matches against the rest (picked on the result: what they "
             "looked like, not yet a signal) ====")
    L.append(f"  worst: the worst {100 * WORST_SHARE:.0f}% by the book's moneyline result; losing: "
             "every match the book lost on it (the worst too); z: losing less winning over its se")
    by_class = lambda *cs: [r for r in mrows if r["class"] in cs]
    L += compare_lines([("worst", by_class("worst")), ("losing", by_class("worst", "losing")),
                        ("winning", by_class("winning")), ("all bet", bet_m), ("no bets", by_class("no bets"))],
                       1, 2, "lose-win")
    stakes = sorted(r["ml_stake"] for r in bet_m)
    top_cut = stakes[int(len(stakes) * 0.8)] if stakes else 0
    most = [r for r in bet_m if r["ml_stake"] >= top_cut]
    rest = [r for r in bet_m if r["ml_stake"] < top_cut]
    L.append(f"\n  What the money targets: the most-bet fifth of matches (moneyline stake {top_cut:,.0f}+) "
             "against the rest, the result aside")
    L += compare_lines([("most-bet", most), ("the rest", rest)], 0, 1, "most-rest")
    L += match_cut(mrows, "class", "Matches by class", order=("worst", "losing", "winning"), min_matches=1)
    edges = totals_signals.quintiles(stakes) if len(stakes) >= 25 else []
    L += match_cut(mrows, lambda r: f"stake fifth {totals_signals.bin_of(r['ml_stake'], edges) + 1}",
                   "Matches by moneyline stake (fifths, 1 the least bet)",
                   order=[f"stake fifth {k}" for k in range(1, 6)], min_matches=1)
    L += match_cut(mrows, lambda r: None if r["fav_p"] is None else band(100 * r["fav_p"], (55, 60, 70, 80), "{:.0f}") + "% fav",
                   "Matches by prod's pre-match favourite", order=["<55% fav", "55-60% fav", "60-70% fav", "70-80% fav", "80+% fav"])
    L += match_cut(mrows, lambda r: None if r["total_vs_line"] is None else
                   "under 10+" if r["total_vs_line"] <= -10 else "under 3-10" if r["total_vs_line"] <= -3
                   else "within 3" if r["total_vs_line"] < 3 else "over 3-10" if r["total_vs_line"] < 10 else "over 10+",
                   "Matches by the final total against prod's closing line",
                   order=("under 10+", "under 3-10", "within 3", "over 3-10", "over 10+"))
    L += match_cut(mrows, lambda r: "dog won" if r["dog_won"] else "fav won" if r["dog_won"] == 0 else None,
                   "Matches by who won", order=("fav won", "dog won"))
    L += match_cut(mrows, lambda r: "comeback" if r["comeback"] else "held" if r["comeback"] == 0 else None,
                   "Matches by whether the half-time leader held on", order=("held", "comeback"))
    L += match_cut(mrows, "session_match", "Matches by match of the session (the later gamer)",
                   order=list(range(1, 30)), min_matches=5)
    L += match_cut(mrows, lambda r: None if r["hour_utc"] is None else f"{r['hour_utc']:02d}h",
                   "Matches by kick-off hour (UTC)", order=[f"{h:02d}h" for h in range(24)], min_matches=5)
    L += match_cut(mrows, lambda r: f"stream {r['stream']}" if r.get("stream") else None, "Matches by stream")
    L.append("\n  ==== 3. In play, every match: the residual by situation ====")
    for key, title, order in (
            ("quarter", "Quarter", ("Q1", "Q2", "Q3", "Q4", "OT")),
            ("lead_band", "The backed side's lead", LEADS),
            (lambda r: r.get("quarter") and r.get("lead_band") and f"{r['quarter']} / {r['lead_band']}",
             "Quarter and lead", [f"{q} / {ld}" for q in ("Q1", "Q2", "Q3", "Q4", "OT") for ld in LEADS]),
            (lambda r: p_band(r["p"]), "prod's probability of the selection",
             [p_band(x) for x in (0.05, 0.2, 0.3, 0.5, 0.7, 0.8, 0.95)]),
            ("fav", "Pre-match standing of the backed side", ("favourite", "evens", "underdog")),
            ("ball", "Who had the ball", None),
            ("zone", "Who had the ball and where", None),
            (lambda r: r.get("ball") in (BACKED, OPPONENT) and r.get("down_dist")
             and f"{'backed' if r['ball'] == BACKED else 'opp'}: {r['down_dist']}",
             "Who had the ball, the down and distance", None),
            ("moment", "The feed at the bet", None),
            ("running_band", "Points against prod's pre-match total so far", RUNNING),
            (lambda r: r.get("running_band") and r.get("fav") and f"{r['fav']} / {r['running_band']}",
             "Pre-match standing and pace (is a low scorer read early?)",
             [f"{f} / {x}" for f in ("favourite", "evens", "underdog") for x in RUNNING]),
            (lambda r: r.get("running_band") and r.get("lead_band") and f"{r['lead_band']} / {r['running_band']}",
             "Lead and pace", [f"{ld} / {x}" for ld in LEADS for x in RUNNING]),
            ("before", "prod over the 60s before the bet", ("followed a move", "held", "faded a move")),
            ("after", "prod over the 60s after the bet", ("their way", "held", "against them")),
            ("next_score", "Who scored next, and how soon", None),
            ("session", "The gamers' sessions", None),
            ("group", "Customer temperature", None),
            ("stake_band", "Stake", [band(x, STAKES) for x in (1, 20, 100, 500, 5000)]),
            (lambda r: None if r.get("price_age_seconds") is None else band(r["price_age_seconds"], (2, 5, 10, 30), "{:g}") + "s",
             "Age of prod's price at the bet", [band(x, (2, 5, 10, 30)) + "s" for x in (1, 3, 7, 20, 60)])):
        L += cut(inplay, key, title, order=order, cands=cands, width=34)
    for grp in ("Restricted", "VIP"):
        sub = [r for r in inplay if r["group"] == grp]
        L += cut(sub, "zone", f"{grp}: who had the ball and where", cands=cands, width=34)
        L += cut(sub, lambda r: r.get("quarter") and r.get("lead_band") and f"{r['quarter']} / {r['lead_band']}",
                 f"{grp}: quarter and lead", cands=cands, width=34)
    L.append("\n  ==== 4. In play, every match: the features ranked ====")
    L += ranking_lines(inplay, NUMERIC, "In play", cands=cands)
    L.append("\n  ==== 5. Pre-match, every match ====")
    for key, title, order in (
            ("fav", "Pre-match standing", ("favourite", "evens", "underdog")),
            (lambda r: p_band(r["p"]), "prod's probability", [p_band(x) for x in (0.05, 0.2, 0.3, 0.5, 0.7, 0.8, 0.95)]),
            ("closing", "prod from the bet to the kick-off", ("closed their way", "held", "closed against them")),
            ("session", "The gamers' sessions", None), ("group", "Customer temperature", None)):
        L += cut(pre, key, title, order=order, cands=cands, width=34)
    L += ranking_lines(pre, [(k, t) for k, t in NUMERIC + (("to_close", "prod's move to the kick-off"),)
                             if k in PRE_NUMERIC], "Pre-match", detail=4, cands=cands)
    L += pile_lines(piles, scope, cut_stake, cands)
    L.append("\n  ==== 7. Gamers and teams (every bet in scope, costliest to the book first) ====")
    for key, title in (("backed", "Backed gamer"), ("opposed", "Opposed gamer (the money against him)"),
                       ("backed_team", "Backed team"), ("opposed_team", "Opposed team"),
                       (lambda r: r["backed"] and r["backed_team"] and f"{r['backed']} / {r['backed_team']}",
                        "Backed gamer with his team")):
        L += cut(scope, key, title, cands=cands, width=34, top=TOP, min_bets=max(20, MIN_BETS // 2))
    L += customer_lines(scope)
    return L


# --- run -------------------------------------------------------------------------------------

def run(cur, out_dir):
    """Every bet and settled match of the window: the moneyline bets against prod's quotes and the
    feed, the matches against their results; print the report, write the CSVs."""
    cand_streams = list(config.CANDIDATES or [])
    bets.check_models(cand_streams)
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    every = [b for b in bets.to_bets(cols, raw) if b.feed_market is not None]
    universe = snowflake_io.match_universe(cur, config.STREAMS["prod"])
    matches = sorted(set(universe) | {b.match_code for b in every})
    ml = [b for b in every if b.feed_market in SIDE]
    print(f"  {len(every):,} bets ({len(ml):,} moneyline) on {len({b.match_code for b in every}):,} matches; "
          f"{len(matches):,} matches in the window", flush=True)
    prod_rows, finals = [], {}
    for start in range(0, len(matches), config.MATCH_CHUNK_SIZE):
        batch = matches[start:start + config.MATCH_CHUNK_SIZE]
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        finals.update(snowflake_io.fetch_final_scores(cur, batch))
        print(f"  prod's quotes: {min(start + config.MATCH_CHUNK_SIZE, len(matches)):,} of {len(matches):,} matches",
              flush=True)
    keep = {}
    checks = bets.fetch_checks(cur, matches, prod_rows, keep=keep)
    prod_tl = bets.timeline(prod_rows)
    del prod_rows
    lags = bets.fit_lags(every, prod_tl)
    base = bets.join(ml, lags, {}, prod_tl, {}, None, finals, checks)
    cand_joined = []
    for stream in cand_streams:
        name = bets.label(stream).replace(".", "_")
        print(f"  pricing the moneyline bets with {name}", flush=True)
        cand_rows = bets.candidate_quotes(cur, stream, sorted({b.match_code for b in ml}))
        cand_joined.append((name, bets.join(ml, lags, {}, prod_tl, bets.quote_index(cand_rows),
                                            bets.timeline(cand_rows), finals, checks,
                                            same_state=snowflake_io.is_model(stream))))
    history = snowflake_io.fetch_history(cur)
    info = {r["MATCH_CODE"]: r for r in history}
    missing = [m for m in matches if m not in info]
    for start in range(0, len(missing), config.MATCH_CHUNK_SIZE):
        for r in snowflake_io.fetch_match_info(cur, missing[start:start + config.MATCH_CHUNK_SIZE]):
            info[r["MATCH_CODE"]] = r
    ctx = Context(prod_tl, checks, keep.get("snapshots", {}), keep.get("scouting", []), info,
                  sessions(history), elo_by_match(history))
    cands = [name for name, _ in cand_joined]
    rows = [bet_row(b, base[k], ctx, [(name, j[k]) for name, j in cand_joined]) for k, b in enumerate(ml)]
    all_runs = runs(rows)
    piles, cut_stake = pile_ons(all_runs)
    for n, run_ in enumerate(piles):
        for r in run_:
            r["pile_on"] = n
    by_ml, by_other = defaultdict(list), defaultdict(list)
    for r in rows:
        by_ml[r["match_code"]].append(r)
    for b in every:
        if b.feed_market not in SIDE:
            by_other[b.match_code].append(b)
    mrows = classify([match_row(m, finals[m], ctx, by_ml.get(m, []), by_other.get(m, []))
                      for m in matches if m in finals and _int(finals[m][0]) is not None
                      and _int(finals[m][1]) is not None])
    cls = {r["match_code"]: r["class"] for r in mrows}
    for r in rows:
        r["match_class"] = cls.get(r["match_code"])
    print(f"  {len(rows):,} moneyline bets read; {len(piles):,} pile-ons; {len(mrows):,} settled matches",
          flush=True)
    lines = report(rows, mrows, piles, cut_stake or 0.0, cands)
    print("\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    bets.write_csv(os.path.join(out_dir, "bets_ml_losers.csv"), rows,
                   list(dict.fromkeys(k for r in rows for k in r)))
    bets.write_csv(os.path.join(out_dir, "bets_ml_matches.csv"), mrows,
                   list(dict.fromkeys(k for r in mrows for k in r)))
    prow = [pile_row(p, n) for n, p in enumerate(piles)]
    bets.write_csv(os.path.join(out_dir, "bets_ml_pileons.csv"), prow,
                   list(dict.fromkeys(k for r in prow for k in r)) or ["pile"])
    path = os.path.join(out_dir, "bets_ml_losers.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path
