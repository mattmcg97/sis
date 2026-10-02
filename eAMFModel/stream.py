"""What every version's price stream shares: quote rows shaped like GAMEPLAI_STREAM's,

    (MATCH_CODE, MARKET_ID, PUBLISH_TIME, PROBABILITY, DECIMAL_ODD,
     MARKET_DESCRIPTION, EVENT_MESSAGE_COUNT, STATUS, IS_ACTIVE)

so anything that can read a GAMEPLAI stream can read a version -- the calibrator swaps one in
for the candidate with `--candidate v9`. Here: the market descriptions, reading a line back, and
the prod messages each PLAY_OVER's state is sure for.
"""

from .state import ML_AWAY, ML_HOME, SPREAD_AWAY, SPREAD_HOME, TOTAL_OVER, TOTAL_UNDER

OPEN = "OPEN"
SUSPENDED = "SUSPENDED"


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
