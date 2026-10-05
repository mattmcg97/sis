"""What every version's price stream shares: quote rows shaped like GAMEPLAI_STREAM's,

    (MATCH_CODE, MARKET_ID, PUBLISH_TIME, PROBABILITY, DECIMAL_ODD,
     MARKET_DESCRIPTION, EVENT_MESSAGE_COUNT, STATUS, IS_ACTIVE)

so anything that can read a GAMEPLAI stream can read a version -- the calibrator swaps one in
for the candidate with `--candidate v9`. Here: the market descriptions, reading a line back, the
prod messages each PLAY_OVER's state is sure for, and what a pre-match model whose form follows
results knew when each pre-match quote was published.
"""

import datetime as dt
from bisect import bisect_left
from collections import defaultdict

from .state import ML_AWAY, ML_HOME, SPREAD_AWAY, SPREAD_HOME, TOTAL_OVER, TOTAL_UNDER

OPEN = "OPEN"
SUSPENDED = "SUSPENDED"

# A pre-match model whose form follows results (glmer) prices each pre-match quote off the results
# known when the quote was published. A match's result counts from its last play in the feed or,
# for a match the feed has none of, RESULT_MINUTES after its kick-off: one slot of the schedule,
# which runs a gamer's matches back to back 36 minutes apart.
RESULT_MINUTES = 36
FEED_END_WITHIN = dt.timedelta(hours=3)      # a feed end further than this from kick-off is not read


def _message(value):
    """A message count, or None."""
    try:
        return None if value in (None, "") else int(float(value))
    except (TypeError, ValueError):
        return None


def side_known(snaps):
    """Whether TEAM_A's side is tied to the scoreboard (home or away) for a match's snapshots."""
    return bool(snaps) and all(str(s.get("team_a_side") or "").lower() in ("home", "away")
                               for s in snaps)


def confident_windows(snaps, book_messages):
    """PLAY_OVER message -> the message its state lasts until (the next PLAY_STARTED, or the next
    PLAY_OVER): the prod messages a book priced at that PLAY_OVER is the state for. Only
    PLAY_OVERs with a book; none for a match whose TEAM_A side is not known."""
    if not side_known(snaps):
        return {}
    booked = set(book_messages)
    rows = sorted(snaps, key=lambda r: _message(r["message"]))
    out = {}
    for k, r in enumerate(rows):
        m = _message(r["message"])
        if m not in booked:
            continue
        ends = [e for e in (_message(r.get("next_start_message")),
                            _message(rows[k + 1]["message"]) if k + 1 < len(rows) else None)
                if e is not None]
        out[m] = min(ends) if ends else float("inf")
    return out


def description(market_id, line):
    """MARKET_DESCRIPTION text the calibrator's parse_line reads back."""
    if market_id == ML_HOME:
        return "PLAYER 1 to win"
    if market_id == ML_AWAY:
        return "PLAYER 2 to win"
    if market_id == SPREAD_HOME:
        return f"PLAYER 1 to score over {line:g} points more than PLAYER 2"
    if market_id == SPREAD_AWAY:
        return f"PLAYER 2 to score over {line:g} points more than PLAYER 1"
    if market_id == TOTAL_OVER:
        return f"Total points over {line:g}"
    if market_id == TOTAL_UNDER:
        return f"Total points under {line:g}"
    return None


def _parse_line(text):
    import re
    if not text:
        return None
    cleaned = re.sub(r"player\s*\d+", " ", text, flags=re.IGNORECASE)
    found = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(found.group()) if found else None


def when(value):
    """A time -- a datetime, or text as Snowflake and the CSVs write it -- as a naive UTC datetime,
    or None."""
    if value in (None, ""):
        return None
    if not isinstance(value, dt.datetime):
        text = str(value).strip().replace("T", " ")
        try:
            value = dt.datetime.fromisoformat(text)
        except ValueError:
            try:
                value = dt.datetime.fromisoformat(text[:19])
            except ValueError:
                return None
    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value


def match_ends(snapshots_by_match):
    """match -> when its last play in the feed was (its snapshots' latest file_time)."""
    out = {}
    for code, snaps in snapshots_by_match.items():
        times = [t for t in (when(s.get("file_time")) for s in snaps or ()) if t is not None]
        if times:
            out[code] = max(times)
    return out


def prematch_publish_times(prod_rows, first_play):
    """The publish times of a match's prod quotes from before its first play: no message, or one
    below first_play (when known)."""
    return {r[2] for r in prod_rows if r[2] is not None and r[3] is not None
            and (r[6] is None or (first_play is not None and r[6] < first_play))}


def _gamers(row):
    """A match row's two gamers, as the pre-match models read their handles."""
    return [g for g in (str(row.get(k) or "").strip().upper()
                        for k in ("PLAYER_1_HANDLE", "PLAYER_2_HANDLE")) if g]


def known_states(schedule, publish_times, history, ends=None, minutes=RESULT_MINUTES):
    """What a pre-match model whose form follows results knew at each pre-match quote.

    `schedule`: the matches to price (match_info rows); `publish_times`: match -> the publish times
    of its pre-match quotes; `history`: settled matches (each gamer's earlier starts); `ends`:
    match -> when it finished (read within FEED_END_WITHIN of its kick-off), else `minutes` after
    its kick-off. At a publish time the earliest of the two gamers' earlier matches still
    unfinished cuts what the model may know: the match is priced as if it started a second before
    that one, so its form reads only the results known then -- a row's features read only matches
    that started before it. Conservative when the two gamers' matches overlap: a result that came
    in behind one still going is left out.

    Returns (the rows to price: each match's own row, then a copy per cut, coded "<match>@<n>",
    with no finals; {(match, publish time): the code that prices it}). A publish time with nothing
    unfinished maps to the match's own row."""
    ends = ends or {}
    gamers = defaultdict(list)                       # gamer -> [(start, end, match)], by start
    for r in history or ():
        start = when(r.get("SCHEDULED_START_TIME_UTC"))
        if start is None:
            continue
        end = ends.get(r["MATCH_CODE"])
        if end is None or not start < end <= start + FEED_END_WITHIN:
            end = start + dt.timedelta(minutes=minutes)
        for g in _gamers(r):
            gamers[g].append((start, end, r["MATCH_CODE"]))
    for runs in gamers.values():
        runs.sort()
    starts = {g: [s for s, _, _ in runs] for g, runs in gamers.items()}
    rows, code_at = list(schedule), {}
    for r in schedule:
        code, start = r["MATCH_CODE"], when(r.get("SCHEDULED_START_TIME_UTC"))
        published = sorted((p for p in publish_times.get(code, ()) if when(p) is not None), key=when)
        if start is None or not published:
            code_at.update(((code, p), code) for p in published)
            continue
        since = when(published[0]) - dt.timedelta(days=1)
        earlier = sorted((s, e) for g in _gamers(r) if g in gamers
                         for s, e, c in gamers[g][bisect_left(starts[g], since):]
                         if s < start and c != code)
        cuts = {}
        for p in published:
            at = when(p)
            pending = [s for s, e in earlier if e > at]
            if not pending:
                code_at[(code, p)] = code
                continue
            cut = pending[0] - dt.timedelta(seconds=1)
            if cut not in cuts:
                cuts[cut] = f"{code}@{len(cuts) + 1}"
                rows.append(dict(r, MATCH_CODE=cuts[cut],
                                 SCHEDULED_START_TIME_UTC=cut.strftime("%Y-%m-%d %H:%M:%S"),
                                 PLAYER_1_FINAL_SCORE="", PLAYER_2_FINAL_SCORE=""))
            code_at[(code, p)] = cuts[cut]
    return rows, code_at
