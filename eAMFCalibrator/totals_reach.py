"""How far each total line sits above the score: within one score, two, or more.

At every drive snapshot a total line asks for (line - points already on
the board) more points. This counts how often that is one score or less
-- a single touchdown takes the game over -- two scores or less, or more,
for prod's line and each candidate's own line. Real is the same count for
the points the rest of the game really produced.

A score is a touchdown and its kick, 7, except where the scoreline is one
at which the trailing player goes for two after scoring: then it is 8.
Those are the margins at which most players in SCOUTING_FULL went for two
(behind by 1, 5, 8, 11 or 16), read at any time in the game.
"""

from . import markets

TWO_POINT_MARGINS = frozenset((1, 5, 8, 11, 16))
TOUCHDOWN = 7
TOUCHDOWN_AND_TWO = 8
WITHIN_ONE, WITHIN_TWO, BEYOND = "within_one", "within_two", "beyond"
CLASSES = (WITHIN_ONE, WITHIN_TWO, BEYOND)


def one_score(score_p1, score_p2):
    """Points a single score is worth from this scoreline."""
    return TOUCHDOWN_AND_TWO if abs(score_p1 - score_p2) in TWO_POINT_MARGINS else TOUCHDOWN


def reach(points_needed, score):
    """Which class `points_needed` more points falls in."""
    if points_needed <= score:
        return WITHIN_ONE
    if points_needed <= 2 * score:
        return WITHIN_TWO
    return BEYOND


def snapshots(pairs):
    """One total pair per snapshot (the over and under share a line)."""
    seen, out = set(), []
    for p in pairs:
        if markets.market_group(p.market_id) != markets.TOTAL:
            continue
        key = (p.match_code, p.drive_number)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _shares(counts, n):
    shares = {c: (counts[c] / n if n else None) for c in CLASSES}
    shares[WITHIN_TWO + "_or_less"] = ((counts[WITHIN_ONE] + counts[WITHIN_TWO]) / n) if n else None
    shares["n"] = n
    return shares


def summarise(pairs):
    """{"real", "prod", "candidate"} -> shares of the snapshots in each
    class, over the total snapshots where all three are known; and for
    "prod" and "candidate", under "over", how often the game went over that
    stream's line when the line sat in each class ({class: (over rate, n)}).

    The shares cannot match real's even for a perfect model: a line sits
    in the middle of the points still to come, so it is more than a score
    away more often than the result is. The over rates can: wherever a
    line sits, a line in the middle goes over half the time. A line held
    a score too far out shows as an over rate well under a half."""
    counts = {k: dict.fromkeys(CLASSES, 0) for k in ("real", "prod", "candidate")}
    overs = {k: {c: [0, 0] for c in CLASSES} for k in ("prod", "candidate")}
    n = 0
    for p in snapshots(pairs):
        if p.prod_line is None or p.candidate_line is None or p.realized is None:
            continue
        on_board = p.score_p1 + p.score_p2
        score = one_score(p.score_p1, p.score_p2)
        n += 1
        counts["real"][reach(p.realized - on_board, score)] += 1
        for stream, line in (("prod", p.prod_line), ("candidate", p.candidate_line)):
            where = reach(line - on_board, score)
            counts[stream][where] += 1
            if p.realized != line:
                overs[stream][where][0] += p.realized > line
                overs[stream][where][1] += 1
    out = {k: _shares(c, n) for k, c in counts.items()}
    for stream, by_class in overs.items():
        out[stream]["over"] = {c: ((k / m) if m else None, m) for c, (k, m) in by_class.items()}
    return out
