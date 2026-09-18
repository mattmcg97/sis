"""The three main GAMEPLAI markets: what each selection means, what line
it carries, and whether it resulted true.

Market IDs are the routing key. 50/51 (moneyline) and 54/55 (totals) are
carried over from analysis/unconditional_calibration.py; 52/53 (spread)
come from the MARKET_DESCRIPTION text in GAMEPLAI_STREAM, e.g.
"PLAYER 2 to score over -2.5 points more than PLAYER 1".

PLAYER_1 is the home team and PLAYER_2 the away team, established in
analysis/clean_possession_sequence.py from these same descriptions.

Resolution returns True (won), False (lost) or None (push -- the final
landed exactly on the line). Pushes are counted and reported but kept out
of the calibration arithmetic, since a pushed selection has no realized
0/1 outcome to compare a probability against.
"""

import re

MONEYLINE = "moneyline"
SPREAD = "spread"
TOTAL = "total"

# market_id -> (market group, selection label, needs a line?)
MARKETS = {
    50: (MONEYLINE, "Home", False),
    51: (MONEYLINE, "Away", False),
    52: (SPREAD, "Home", True),
    53: (SPREAD, "Away", True),
    54: (TOTAL, "Over", True),
    55: (TOTAL, "Under", True),
}

MARKET_IDS = sorted(MARKETS)

# "PLAYER 1" / "PLAYER 2" carry digits that would otherwise be mistaken for
# the line, so they come out before the number hunt.
_PLAYER_TOKEN = re.compile(r"player\s*\d+", re.IGNORECASE)
_SIGNED_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_line(description):
    """Pull the handicap or total out of a MARKET_DESCRIPTION.

    Returns a float, or None when the description carries no number (which
    is the normal case for moneyline).
    """
    if not description:
        return None
    cleaned = _PLAYER_TOKEN.sub(" ", description)
    match = _SIGNED_NUMBER.search(cleaned)
    return float(match.group()) if match else None


def market_group(market_id):
    entry = MARKETS.get(market_id)
    return entry[0] if entry else None


def selection_label(market_id):
    entry = MARKETS.get(market_id)
    return entry[1] if entry else None


def needs_line(market_id):
    entry = MARKETS.get(market_id)
    return entry[2] if entry else False


def realized_value(market_id, final_p1, final_p2):
    """The quantity this market's line is compared against.

    Spread: the selection's own final margin. Total: the combined score.
    Moneyline carries no line, so None.
    """
    if final_p1 is None or final_p2 is None or market_id not in MARKETS:
        return None
    group = market_group(market_id)
    if group == SPREAD:
        return (final_p1 - final_p2) if market_id == 52 else (final_p2 - final_p1)
    if group == TOTAL:
        return final_p1 + final_p2
    return None


def resolve(market_id, line, final_p1, final_p2):
    """Did this selection result true, given the final score?

    None means push, or that the inputs could not settle it.
    """
    if final_p1 is None or final_p2 is None:
        return None
    if market_id not in MARKETS:
        return None

    group = market_group(market_id)

    if group == MONEYLINE:
        if final_p1 == final_p2:
            return None
        return final_p1 > final_p2 if market_id == 50 else final_p2 > final_p1

    if line is None:
        return None

    if group == SPREAD:
        margin = (final_p1 - final_p2) if market_id == 52 else (final_p2 - final_p1)
        if margin == line:
            return None
        return margin > line

    if group == TOTAL:
        total = final_p1 + final_p2
        if total == line:
            return None
        return total > line if market_id == 54 else total < line

    return None
