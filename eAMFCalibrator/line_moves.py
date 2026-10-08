"""Lines that move and then aren't reached: does a stream react too far to a score?

At every drive snapshot each stream quotes a total and a spread. Set against
the stream's previous quote in the same match, its line went up, down or
stayed. If a stream over-reacts -- lifts the total too far after a score, or
drops it too fast as the clock runs -- the games it moved on finish short of
(or past) its line more often than its own price said.

So, for each snapshot: did the game finish over the stream's line (the total)
or did the home side cover (the spread, read as the home margin the line asks
for), against the stream's own P(over) / P(home covers) at that line. A fair
middle line is 50-50; late in a game the points still to come are lumpy
(0, 3 or 7), the line nearest even money can sit well off 50%, and the price
is what a fair line has to match. Pushes are left out.

Split by how the line moved since the stream's last quote (and, for the
total, whether anyone scored in between), by quarter and by the score
difference at the quote.
"""

from collections import defaultdict

from . import buckets, markets

UP, DOWN, FLAT, FIRST = "up", "down", "flat", "first"
MOVES = (UP, DOWN, FLAT)
STEP = 0.25                 # a line moved if it moved by more than this
LEVEL, ONE, TWO = "level (0-2)", "1 score (3-8)", "2+ scores (9+)"
DIFFS = (LEVEL, ONE, TWO)
STREAMS = ("prod", "candidate")


def score_diff(p):
    """Level within 2, one score 3-8, two or more 9+."""
    d = abs(p.score_p1 - p.score_p2)
    return LEVEL if d <= 2 else ONE if d <= 8 else TWO


def _home_side(p, line, prob):
    """A spread pair as the home margin its line asks for and P(home covers),
    whichever side the pair is: 53 is the away side's own margin under
    literal resolution, the NO of 52's line under complement."""
    if line is None or prob is None:
        return None, None
    if p.market_id == 52:
        return line, prob
    if markets.spread_resolution() == "complement":
        return line, 1 - prob
    return -line, 1 - prob


def quotes(pairs, group):
    """{stream: [quote]} for one market group (markets.TOTAL or SPREAD): one
    quote per (match, drive), in drive order, each a dict with the line, the
    stream's P(over the line) -- for the spread, P(the home margin beats it) --
    the realized total or home margin, the board, the quarter, the score
    difference, the move since the stream's previous quote and whether the
    board changed in between."""
    seen, snaps = set(), []
    order = (52, 53) if group == markets.SPREAD else (54, 55)
    ordered = sorted(pairs, key=lambda p: (p.match_code, p.drive_number, p.message_count))
    for mid in order:                          # the home / over side first; a drive's first quote
        for p in ordered:
            if p.market_id != mid or (p.match_code, p.drive_number) in seen:
                continue
            seen.add((p.match_code, p.drive_number))
            snaps.append(p)
    snaps.sort(key=lambda p: (p.match_code, p.drive_number))
    out = {s: [] for s in STREAMS}
    for stream in STREAMS:
        last = {}
        for p in snaps:
            line = getattr(p, f"{stream}_line")
            prob = getattr(p, f"{stream}_probability")
            if line is None or prob is None or p.realized is None:
                continue
            if group == markets.SPREAD:
                line, prob = _home_side(p, line, prob)
                real = p.realized if p.market_id == 52 else -p.realized      # the home margin
            else:
                prob = prob if markets.selection_label(p.market_id) == "Over" else 1 - prob
                real = p.realized
            board = (p.score_p1, p.score_p2)
            prev = last.get(p.match_code)
            if prev is None:
                move, scored = FIRST, False
            else:
                delta = line - prev[0]
                move = UP if delta > STEP else DOWN if delta < -STEP else FLAT
                scored = board != prev[1]
            last[p.match_code] = (line, board)
            out[stream].append({"match": p.match_code, "drive": p.drive_number,
                                "line": line, "prob": prob, "real": real, "move": move,
                                "scored": scored, "quarter": buckets.period_bucket(p.period_number),
                                "diff": score_diff(p), "board": board})
    return out


def _cell(qs):
    """(n, share over the line, mean priced P(over)) over quotes that didn't push."""
    qs = [q for q in qs if q["real"] != q["line"]]
    if not qs:
        return (0, None, None)
    return (len(qs), sum(q["real"] > q["line"] for q in qs) / len(qs),
            sum(q["prob"] for q in qs) / len(qs))


def summarise(pairs, group):
    """{stream: {"moves": {label: cell}, "grid": {(quarter, diff): {move or "all": cell}}}}
    with each cell (n, share over / home covered, mean priced). The move
    labels split the total's up moves by whether anyone scored since the
    last quote, and the spread's by which way the line went (toward the side
    that scored)."""
    out = {}
    for stream, qs in quotes(pairs, group).items():
        moved = [q for q in qs if q["move"] != FIRST]
        moves = {
            "line up, after a score": _cell([q for q in moved if q["move"] == UP and q["scored"]]),
            "line up, no score": _cell([q for q in moved if q["move"] == UP and not q["scored"]]),
            "line down, after a score": _cell([q for q in moved if q["move"] == DOWN and q["scored"]]),
            "line down, no score": _cell([q for q in moved if q["move"] == DOWN and not q["scored"]]),
            "line held": _cell([q for q in moved if q["move"] == FLAT]),
            "all": _cell(qs),
        }
        grid = defaultdict(lambda: defaultdict(list))
        for q in qs:
            for key in ((q["quarter"], q["diff"]), (q["quarter"], "all")):
                grid[key]["all"].append(q)
                if q["move"] in MOVES:
                    grid[key][q["move"]].append(q)
        out[stream] = {"moves": moves,
                       "grid": {k: {m: _cell(v) for m, v in g.items()} for k, g in grid.items()}}
    return out


# The first touchdown: does a stream lift the total (and the scorer's spread) too far when a side
# scores early -- a quick opening drive -- and more so when the scorer was the favourite?
TD_WHEN = ("first minute", "1:00-2:00", "rest of Q1", "Q2", "second half", "clock unknown")
FAVOURITE, UNDERDOG, EVEN = "favourite", "underdog", "pick'em"
SCORERS = (FAVOURITE, UNDERDOG, EVEN)
EVEN_BAND = 0.05             # prod's first moneyline within this of 50%: neither side is favourite
QUARTER_SECONDS = 240


def td_when(period, clock):
    """When in the game a touchdown came: its quarter, and in Q1 the minute."""
    if period is None:
        return "clock unknown"
    if period >= 3:
        return "second half"
    if period == 2:
        return "Q2"
    if clock is None:
        return "clock unknown"
    gone = QUARTER_SECONDS - clock
    return "first minute" if gone <= 60 else "1:00-2:00" if gone <= 120 else "rest of Q1"


def favourites(pairs):
    """{match: "home" / "away" / None}: the side prod's first moneyline quote had above 50%,
    None within EVEN_BAND of it."""
    first = {}
    for p in sorted(pairs, key=lambda p: (p.match_code, p.drive_number, p.message_count)):
        if p.market_id not in (50, 51) or p.match_code in first or p.prod_probability is None:
            continue
        home = p.prod_probability if p.market_id == 50 else 1 - p.prod_probability
        first[p.match_code] = (None if abs(home - 0.5) < EVEN_BAND else "home" if home > 0.5 else "away")
    return first


def touchdowns(pairs):
    """{match: (first drive of the window, last, scoring side, when)}: each match's first
    touchdown -- the first snapshot where one side's score is 6 or more above the snapshot
    before -- and its window, every snapshot until the next score beyond the conversion."""
    boards = defaultdict(dict)
    for p in pairs:
        boards[p.match_code].setdefault(p.drive_number, (p.score_p1, p.score_p2, p.period_number,
                                                         p.clock_seconds))
    out = {}
    for m, by_drive in boards.items():
        drives = sorted(by_drive)
        for a, b in zip(drives, drives[1:]):
            (h0, a0, _, _), (h1, a1, period, clock) = by_drive[a], by_drive[b]
            if h1 - h0 >= 6 or a1 - a0 >= 6:
                side = "home" if h1 - h0 >= 6 else "away"
                cap = h1 + a1 + 2                          # the conversion stays in the window
                last = b
                for d in drives[drives.index(b) + 1:]:
                    if sum(by_drive[d][:2]) > cap:
                        break
                    last = d
                out[m] = (b, last, side, td_when(period, clock))
                break
    return out


def first_touchdowns(pairs):
    """{stream: {(when, scorer): {"total": cell, "spread": cell, "matches": n}}}: every quote in
    each match's first-touchdown window (touchdowns) -- the total's over, and the spread read from
    the scoring side (did it beat the margin its line asked for) -- against the stream's own
    price there; the scorer is the favourite or the underdog by prod's first moneyline."""
    fav, tds = favourites(pairs), touchdowns(pairs)
    by_market = {"total": quotes(pairs, markets.TOTAL), "spread": quotes(pairs, markets.SPREAD)}
    out = {}
    for stream in STREAMS:
        cells = defaultdict(lambda: {"total": [], "spread": [], "matches": set()})
        for market, qs in by_market.items():
            for q in qs[stream]:
                td = tds.get(q["match"])
                if td is None or not td[0] <= q["drive"] <= td[1]:
                    continue
                first, _, side, when = td
                f = fav.get(q["match"])
                scorer = EVEN if f is None else FAVOURITE if f == side else UNDERDOG
                if market == "spread" and side == "away":
                    q = dict(q, line=-q["line"], real=-q["real"], prob=1 - q["prob"])
                cells[(when, scorer)][market].append(q)
                cells[(when, scorer)]["matches"].add(q["match"])
        out[stream] = {k: {"total": _cell(v["total"]), "spread": _cell(v["spread"]),
                           "matches": len(v["matches"])} for k, v in cells.items()}
    return out
