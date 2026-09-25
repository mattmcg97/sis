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

from . import buckets, markets

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
    stream's line when the line sat in each class ({class: (over rate, n)}),
    and, under "priced", the stream's own mean P(over) there.

    The shares cannot match real's even for a perfect model: a line sits
    in the middle of the points still to come, so it is more than a score
    away more often than the result is. The over rates can be judged:
    against the stream's own P(over) at its line -- a half for a middle
    line, but not late in a game, where the points still to come are lumpy
    (0, 3 or 7) and the line nearest even money can be well off it."""
    counts = {k: dict.fromkeys(CLASSES, 0) for k in ("real", "prod", "candidate")}
    overs = {k: {c: [0, 0, 0.0] for c in CLASSES} for k in ("prod", "candidate")}
    n = 0
    for p in snapshots(pairs):
        if p.prod_line is None or p.candidate_line is None or p.realized is None:
            continue
        on_board = p.score_p1 + p.score_p2
        score = one_score(p.score_p1, p.score_p2)
        n += 1
        counts["real"][reach(p.realized - on_board, score)] += 1
        for stream, line, prob in (("prod", p.prod_line, p.prod_probability),
                                   ("candidate", p.candidate_line, p.candidate_probability)):
            where = reach(line - on_board, score)
            counts[stream][where] += 1
            if p.realized != line:
                overs[stream][where][0] += p.realized > line
                overs[stream][where][1] += 1
                overs[stream][where][2] += over_probability(p, prob)
    out = {k: _shares(c, n) for k, c in counts.items()}
    for stream, by_class in overs.items():
        out[stream]["over"] = {c: ((k / m) if m else None, m) for c, (k, m, _) in by_class.items()}
        out[stream]["priced"] = {c: (q / m if m else None) for c, (_, m, q) in by_class.items()}
    return out


def over_probability(pair, probability):
    """The stream's P(over its own line), off whichever side of the total
    the snapshot's pair is."""
    return probability if markets.selection_label(pair.market_id) == "Over" else 1 - probability


ALL = "all"
STATES = (ALL, "level", "1 score, leader has ball", "1 score, trailer has ball",
          "2+ scores, leader has ball", "2+ scores, trailer has ball", "no ball")


def game_state(pair):
    """Where the game stands for the total: level (within 2), one score
    apart (3-8) or two or more (9+), and who has the ball."""
    margin = pair.score_p1 - pair.score_p2
    side = buckets.OFFENSIVE_TEAM_TO_SIDE.get(pair.offensive_team)
    if side is None:
        return "no ball"
    if abs(margin) <= 2:
        return "level"
    leader = buckets.HOME if margin > 0 else buckets.AWAY
    apart = "1 score" if abs(margin) <= 8 else "2+ scores"
    return f"{apart}, {'leader' if side == leader else 'trailer'} has ball"


def breakdown(pairs):
    """{(quarter, state): {"n", "real": (within 1, within 2), "prod" and
    "candidate": (line within 1, line within 2, over rate at the line, the
    stream's mean P(over) there)}}: summarise's figures for each quarter
    and game state, and for each quarter as a whole (state ALL)."""
    groups = {}
    for p in snapshots(pairs):
        if p.prod_line is None or p.candidate_line is None or p.realized is None:
            continue
        quarter = buckets.period_bucket(p.period_number)
        groups.setdefault((quarter, game_state(p)), []).append(p)
        groups.setdefault((quarter, ALL), []).append(p)
    out = {}
    for key, ps in groups.items():
        r = summarise(ps)

        def over(stream):
            k = sum(rate * n for rate, n in r[stream]["over"].values() if n)
            n = sum(n for _, n in r[stream]["over"].values())
            return k / n if n else None

        def priced(stream):
            q = sum(r[stream]["priced"][c] * m for c, (_, m) in r[stream]["over"].items() if m)
            n = sum(m for _, m in r[stream]["over"].values())
            return q / n if n else None
        out[key] = {"n": r["real"]["n"],
                    "real": (r["real"][WITHIN_ONE], r["real"][WITHIN_TWO + "_or_less"])}
        for stream in ("prod", "candidate"):
            out[key][stream] = (r[stream][WITHIN_ONE], r[stream][WITHIN_TWO + "_or_less"], over(stream),
                                priced(stream))
    return out
