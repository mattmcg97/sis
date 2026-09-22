"""Does the price move the right way when a play goes well for the offence?

The calibration work so far asks whether a probability was RIGHT. This
asks something cheaper to be sure about and harder to fake: whether the
model REACTS. A first down is good for the team with the ball. Its
moneyline should go up. If it does not, the number is not reading the
game, whatever its Brier score says.

The unit is a TRANSITION: two consecutive cleaned play rows, and what
happened between them. Each transition is classified from the play feed
alone -- down, distance, field position, and any points on the scoreboard
-- and each class carries an expected direction for the side in
possession. A selection's probability is then compared at the two
messages and scored on its SIGN, not its size.

Three things are excluded rather than guessed at:

  * A transition whose LINE moved between the two messages. A different
    line is a different question, exactly as it is on a pair, so the
    probability delta is not a reaction to the play.
  * A transition where either endpoint was not live. A price nobody could
    have taken did not move.
  * SHORT_GAIN. Three yards on 3rd and 8 helps nobody, so the class
    carries no directional expectation. It is counted and shown, and
    scored by nothing.

One bias is structural and worth naming before any number is read. A
failed third down usually ENDS the drive, so it is not an in-drive
transition at all -- restricted to plays inside one possession, the
sample leans towards things going well. That is why the drive-ending
transition is built too, and why in-drive and drive-ending are reported
apart rather than pooled.
"""

import collections
import random
from dataclasses import dataclass
from typing import Optional

from . import buckets, config, directional, drives, markets, metrics, snowflake_io

TOUCHDOWN_POINTS = 6
FIELD_GOAL_POINTS = 3
BIG_GAIN_YARDS = 5

# What a play did for the offence. The sign is the whole model: +1 means
# the side with the ball should be priced better after it than before, -1
# worse, 0 that nobody can say.
TOUCHDOWN = "touchdown"
FIELD_GOAL = "field_goal"              # exactly three
EXTRA_POINT = "extra_point"            # one or two: a PAT or a conversion
SCORE = "score"                        # any other points short of a touchdown
FIRST_DOWN = "first_down"
BIG_GAIN = "big_gain"                  # 5+ yards, short of the line to gain
SHORT_GAIN = "short_gain"              # 1-4 yards, short of it
NO_GAIN = "no_gain"
LOSS = "loss"
FAILED_CONVERSION = "failed_conversion"  # 3rd or 4th down, not converted
POSSESSION_LOST = "possession_lost"      # the drive ended, and not in points
POINTS_AGAINST = "points_against"        # a safety, or the defence scored

OUTCOME_SIGN = {
    TOUCHDOWN: +1,
    FIELD_GOAL: +1,
    EXTRA_POINT: +1,
    SCORE: +1,
    FIRST_DOWN: +1,
    BIG_GAIN: +1,
    SHORT_GAIN: 0,
    NO_GAIN: -1,
    LOSS: -1,
    FAILED_CONVERSION: -1,
    POSSESSION_LOST: -1,
    POINTS_AGAINST: -1,
}

# Strongest first, which is the order they are reported in. A model that
# gets the direction right should also move MORE on the ones near the top.
OUTCOME_ORDER = [TOUCHDOWN, FIELD_GOAL, EXTRA_POINT, SCORE,
                 FIRST_DOWN, BIG_GAIN, SHORT_GAIN,
                 NO_GAIN, LOSS, FAILED_CONVERSION, POSSESSION_LOST,
                 POINTS_AGAINST]

HOME, AWAY = drives.HOME_TEAM, drives.AWAY_TEAM

# What moved, and therefore what was scored. Where the line held, the
# probability carries the news; where it moved, the probability is not
# comparable across it and the line carries it instead.
PROBABILITY = "prob"
LINE = "line"


@dataclass(frozen=True)
class Transition:
    """Two consecutive play rows, and what happened between them."""
    match_code: str
    drive_number: int
    period_number: Optional[int]
    offensive_team: Optional[str]
    from_message: int
    to_message: int
    down_before: Optional[int]
    distance_before: Optional[int]
    field_before: Optional[int]
    field_after: Optional[int]
    yards: Optional[int]
    points: int
    outcome: str
    in_drive: bool

    @property
    def sign(self):
        return OUTCOME_SIGN.get(self.outcome, 0)

    @property
    def scorable(self):
        """Does this transition carry a direction anyone can check?"""
        return self.sign != 0 and self.offensive_team in (HOME, AWAY)


@dataclass(frozen=True)
class Move:
    """One selection's move across one transition, for one stream.

    `basis` says WHAT moved. Where the line held, the probability carries
    the news and is scored. Where the line moved, the probability is not
    comparable across it -- a different line is a different question, the
    same rule the pair comparison runs on -- so the LINE is scored
    instead. `before` and `after` are whichever of the two the basis names,
    which is why magnitudes are never pooled across bases.
    """
    transition: Transition
    stream: str
    market_id: int
    before: float
    after: float
    line_before: Optional[float]
    line_after: Optional[float]
    expected: int
    basis: str = PROBABILITY

    @property
    def delta(self):
        return self.after - self.before

    @property
    def flat(self):
        """The price did not move at all -- neither right nor wrong."""
        return self.delta == 0

    @property
    def hit(self):
        """Did it move the way the play says it should?"""
        if self.flat:
            return None
        return (self.delta > 0) == (self.expected > 0)

    @property
    def signed(self):
        """The move, oriented so that positive is the expected direction."""
        return self.delta * self.expected


def expected_line_sign(market_id, offensive_team, outcome_sign):
    """Which way this selection's LINE should move.

    Not the same question as the probability, and the difference is the
    whole reason a line move is worth scoring rather than dropping.

    A SPREAD line follows the side it names, exactly as that side's
    probability would: checked against the data, market 52's line tracks
    the home lead almost one for one and 53's is its mirror.

    A TOTAL line does NOT. Over and Under share ONE number -- 94% of
    snapshots quote the identical value on both -- so it rises on a good
    offensive play whichever selection is carrying it. Under's PROBABILITY
    should fall while Under's LINE goes up, and reading the line with the
    probability's expectation would score every one of them backwards.
    """
    if not outcome_sign or not markets.needs_line(market_id):
        return 0
    selection = markets.selection_label(market_id)
    if selection in ("Over", "Under"):
        return outcome_sign
    if offensive_team not in (HOME, AWAY) or selection not in ("Home", "Away"):
        return 0
    offence = "Home" if offensive_team == HOME else "Away"
    return outcome_sign if selection == offence else -outcome_sign


def expected_sign(market_id, offensive_team, outcome_sign):
    """Which way this selection's probability should move.

    A team-sided selection follows the side it names: the offence's own
    moneyline and spread rise on a good play, its opponent's fall. Totals
    are possession-blind -- points are points whoever scores them -- so
    Over follows the play and Under opposes it.
    """
    if not outcome_sign or offensive_team not in (HOME, AWAY):
        return 0
    selection = markets.selection_label(market_id)
    if selection in ("Over", "Under"):
        return outcome_sign if selection == "Over" else -outcome_sign
    if selection not in ("Home", "Away"):
        return 0
    offence = "Home" if offensive_team == HOME else "Away"
    return outcome_sign if selection == offence else -outcome_sign


def _points_between(scores, low, high, offensive_team):
    """Points the side in possession put on the board in (low, high].

    Signed to the offence: a score by the other side comes back negative,
    which is a defensive touchdown and is not a good play for the offence
    by any reading.
    """
    total = 0
    for row in scores:
        if not (low < row.event_message_count <= high):
            continue
        p1, p2 = row.p1_change or 0, row.p2_change or 0
        if offensive_team == HOME:
            total += p1 - p2
        elif offensive_team == AWAY:
            total += p2 - p1
    return total


def classify(before, after, points, in_drive, possession_kept):
    """Which outcome class this transition falls in.

    Order matters. Points settle it first; then a converted first down;
    then a down that ran out. Yardage only decides the cases none of those
    reached.
    """
    # Points settle it, whoever scored them. _points_between is signed to
    # the offence, so a negative total is a safety or a defensive score --
    # which is not a good play for the offence by any reading.
    if points >= TOUCHDOWN_POINTS:
        return TOUCHDOWN
    if points == FIELD_GOAL_POINTS:
        return FIELD_GOAL
    if points in (1, 2):
        # A PAT or a two-point conversion. Their own points, but the
        # touchdown they follow was already priced one transition ago, so
        # there is much less news in them than the class above.
        return EXTRA_POINT
    if points > 0:
        return SCORE
    if points < 0:
        return POINTS_AGAINST
    if not in_drive and not possession_kept:
        # The drive ended and nobody scored on it.
        return POSSESSION_LOST

    known = before.field_position is not None and after.field_position is not None
    yards = (after.field_position - before.field_position) if known else None

    converted = (after.down_number == 1 and before.down_number != 1
                 and (yards is None or yards > 0))
    if converted:
        return FIRST_DOWN
    if (after.down_number == 1 and before.down_number == 1
            and yards is not None and yards >= drives.FIRST_DOWN_YARDS):
        # 1st and 10 to 1st and 10 with the ten yards earned.
        return FIRST_DOWN
    if before.down_number in (3, 4) and after.down_number != 1:
        return FAILED_CONVERSION
    if yards is None:
        return SHORT_GAIN
    if yards >= BIG_GAIN_YARDS:
        return BIG_GAIN
    if yards > 0:
        return SHORT_GAIN
    if yards == 0:
        return NO_GAIN
    return LOSS


def transitions_for_match(match_code, plays, scores):
    """Every transition in one match, in message order.

    Two kinds. Inside a drive, one row to the next. At a drive's end, its
    last row to the first row of the next drive -- which is where a stop,
    a turnover and a touchdown all live, and leaving it out would keep
    almost every bad play out of the sample.
    """
    drive_list = drives.build_drives(match_code, plays, scores)
    scores = sorted(scores, key=lambda s: s.event_message_count)

    out = []
    for i, drive in enumerate(drive_list):
        steps = [(a, b, True) for a, b in zip(drive.plays, drive.plays[1:])]
        if i + 1 < len(drive_list):
            following = drive_list[i + 1]
            steps.append((drive.last, following.first,
                          following.offensive_team == drive.offensive_team))
        for before, after, same_drive in steps:
            in_drive = same_drive is True and after in drive.plays
            possession_kept = in_drive or same_drive
            points = _points_between(scores, before.event_message_count,
                                     after.event_message_count,
                                     drive.offensive_team)
            known = (before.field_position is not None
                     and after.field_position is not None)
            out.append(Transition(
                match_code=match_code,
                drive_number=drive.drive_number,
                period_number=before.period_number,
                offensive_team=drive.offensive_team,
                from_message=before.event_message_count,
                to_message=after.event_message_count,
                down_before=before.down_number,
                distance_before=before.distance,
                field_before=before.field_position,
                field_after=after.field_position if in_drive else None,
                yards=(after.field_position - before.field_position)
                      if (known and in_drive) else None,
                points=points,
                outcome=classify(before, after, points, in_drive,
                                 possession_kept),
                in_drive=in_drive,
            ))
    out.sort(key=lambda t: (t.from_message, t.to_message))
    return out


def moves_for_transitions(transitions, indexes, stats=None):
    """Price moves for every transition, selection and stream.

    A move is only built where the same stream had a LIVE quote at both
    messages and the LINE did not change between them. Anything else is
    counted and dropped: a different line is a different question, and a
    price nobody could have taken did not move.
    """
    stats = stats if stats is not None else collections.defaultdict(int)
    out = []
    for transition in transitions:
        if not transition.scorable:
            stats["transition_no_direction"] += 1
            continue
        stats["transition_scorable"] += 1
        for stream, index in indexes.items():
            for market_id in markets.MARKET_IDS:
                quotes = index.get((transition.match_code, market_id), {})
                before = quotes.get(transition.from_message)
                after = quotes.get(transition.to_message)
                if before is None or after is None:
                    stats["move_missing_quote"] += 1
                    continue
                if not (before.live and after.live):
                    stats["move_not_live"] += 1
                    continue
                line_before = markets.parse_line(before.description)
                line_after = markets.parse_line(after.description)
                moved = (markets.needs_line(market_id)
                         and line_before != line_after)
                if moved and (line_before is None or line_after is None):
                    # One end carried no readable number, so there is no
                    # move to measure, only a difference in the text.
                    stats["move_line_unreadable"] += 1
                    continue
                if moved:
                    # The line moved, so the probability is answering a
                    # different question at each end and cannot be
                    # differenced. The line itself carries the news.
                    expected = expected_line_sign(
                        market_id, transition.offensive_team, transition.sign)
                    basis, first, second = LINE, line_before, line_after
                else:
                    # GAMEPLAI publishes 0-100; everything in this suite
                    # works in 0-1, so a move reads on the same scale as a
                    # Brier delta rather than a hundred times larger.
                    expected = expected_sign(
                        market_id, transition.offensive_team, transition.sign)
                    basis = PROBABILITY
                    first = before.probability / 100.0
                    second = after.probability / 100.0
                if not expected:
                    continue
                stats[f"move_scored_on_{basis}"] += 1
                out.append(Move(
                    transition=transition, stream=stream, market_id=market_id,
                    before=first, after=second, line_before=line_before,
                    line_after=line_after, expected=expected, basis=basis))
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _clustered_rate(by_match, n_bootstrap=1000, seed=0):
    """Hit rate with a match-clustered interval.

    by_match maps a match code to (hits, decided). Resampling MATCHES
    rather than moves carries the same clustering the rest of the suite
    uses: plays inside one game are not independent draws.
    """
    codes = sorted(by_match)
    hits = sum(by_match[c][0] for c in codes)
    decided = sum(by_match[c][1] for c in codes)
    out = {"matches": len(codes), "hits": hits, "decided": decided,
           "rate": (hits / decided) if decided else None,
           "ci_low": None, "ci_high": None, "p_value": None}
    if not decided:
        return out
    # Against a coin: a model that reacted at random would sit at 0.500.
    out["p_value"] = metrics.sign_test(hits, decided - hits)
    if len(codes) < 2 or n_bootstrap <= 0:
        return out
    rnd = random.Random(seed)
    rates = []
    for _ in range(n_bootstrap):
        h = d = 0
        for _ in codes:
            ch, cd = by_match[codes[rnd.randrange(len(codes))]]
            h += ch
            d += cd
        if d:
            rates.append(h / d)
    if rates:
        rates.sort()
        out["ci_low"] = rates[int(0.025 * len(rates))]
        out["ci_high"] = rates[min(int(0.975 * len(rates)), len(rates) - 1)]
    return out


def block(moves, n_bootstrap=1000):
    """Everything worth saying about one bag of moves.

    The hit RATE pools both bases -- right is right, whether the news
    arrived as a probability or as a line. The MAGNITUDES never pool:
    probability points and handicap points are different units, and
    averaging them together would produce a number in no units at all.
    """
    by_match = collections.defaultdict(lambda: [0, 0])
    flat = 0
    sizes = {PROBABILITY: [], LINE: []}
    signed = {PROBABILITY: [], LINE: []}
    for move in moves:
        code = move.transition.match_code
        if move.flat:
            flat += 1
        else:
            by_match[code][1] += 1
            by_match[code][0] += int(move.hit)
        sizes[move.basis].append(abs(move.delta))
        signed[move.basis].append(move.signed)

    def mean(values):
        return (sum(values) / len(values)) if values else None

    out = _clustered_rate({k: tuple(v) for k, v in by_match.items()},
                          n_bootstrap)
    out["n"] = len(moves)
    out["flat"] = flat
    out["flat_share"] = (flat / len(moves)) if moves else None
    out["n_prob"] = len(sizes[PROBABILITY])
    out["n_line"] = len(sizes[LINE])
    out["mean_move"] = mean(sizes[PROBABILITY])
    out["mean_signed"] = mean(signed[PROBABILITY])
    out["mean_line_move"] = mean(sizes[LINE])
    out["mean_line_signed"] = mean(signed[LINE])
    return out


def by(moves, key, n_bootstrap=1000):
    """Blocks grouped by whatever key names the split."""
    grouped = collections.defaultdict(list)
    for move in moves:
        grouped[key(move)].append(move)
    return {k: block(v, n_bootstrap) for k, v in grouped.items()}


def period_label(move):
    return buckets.time_bucket(move.transition.period_number, None)


def selection_label(move):
    return (markets.market_group(move.market_id),
            markets.selection_label(move.market_id))


def head_to_head(moves, n_bootstrap=1000):
    """Where both streams scored the same move, which got it right?

    Paired on (transition, selection), so the two are answering exactly
    the same question about exactly the same play. A move both got right,
    or both got wrong, is a tie and decides nothing.
    """
    keyed = collections.defaultdict(dict)
    for move in moves:
        key = (move.transition.match_code, move.transition.from_message,
               move.transition.to_message, move.market_id)
        keyed[key][move.stream] = move

    by_match = collections.defaultdict(lambda: [0, 0])
    ties = both_flat = 0
    for key, sides in keyed.items():
        prod = sides.get(directional.PROD)
        candidate = sides.get(directional.CANDIDATE)
        if prod is None or candidate is None:
            continue
        if prod.flat and candidate.flat:
            both_flat += 1
            continue
        if prod.hit == candidate.hit:
            ties += 1
            continue
        by_match[key[0]][1] += 1
        by_match[key[0]][0] += int(bool(candidate.hit))
    out = _clustered_rate({k: tuple(v) for k, v in by_match.items()},
                          n_bootstrap)
    out["ties"] = ties
    out["both_flat"] = both_flat
    out["pairs"] = len(keyed)
    return out


def report(moves, n_bootstrap=1000):
    """The whole analysis, in the shape the console and HTML both read."""
    in_drive = [m for m in moves if m.transition.in_drive]
    ending = [m for m in moves if not m.transition.in_drive]
    out = {"moves": len(moves), "in_drive": len(in_drive), "ending": len(ending),
           "matches": len({m.transition.match_code for m in moves}),
           "transitions": len({(m.transition.match_code,
                                m.transition.from_message,
                                m.transition.to_message) for m in moves}),
           "streams": {}, "head_to_head": {}}
    for scope, subset in (("all", moves), ("in_drive", in_drive),
                          ("ending", ending)):
        out["head_to_head"][scope] = head_to_head(subset, n_bootstrap)
    for stream in (directional.PROD, directional.CANDIDATE):
        mine = [m for m in moves if m.stream == stream]
        out["streams"][stream] = {
            "overall": block(mine, n_bootstrap),
            "in_drive": block([m for m in mine if m.transition.in_drive],
                              n_bootstrap),
            "ending": block([m for m in mine if not m.transition.in_drive],
                            n_bootstrap),
            "by_outcome": by(mine, lambda m: m.transition.outcome, n_bootstrap),
            "by_period": by(mine, period_label, n_bootstrap),
            "by_selection": by(mine, selection_label, n_bootstrap),
            "by_market": by(mine, lambda m: markets.market_group(m.market_id),
                            n_bootstrap),
            "by_basis": by(mine, lambda m: m.basis, n_bootstrap),
        }
    return out


def outcome_census(transitions):
    """How many transitions of each class, and how many carry a direction.

    Printed before any hit rate, because a class with four transitions in
    it cannot say anything and the reader should see that first.
    """
    census = collections.OrderedDict()
    for outcome in OUTCOME_ORDER:
        rows = [t for t in transitions if t.outcome == outcome]
        census[outcome] = {
            "n": len(rows),
            "in_drive": sum(1 for t in rows if t.in_drive),
            "sign": OUTCOME_SIGN[outcome],
            "matches": len({t.match_code for t in rows}),
        }
    return census


def run(cur, match_codes, time_column, stats=None, n_bootstrap=1000,
        verbose=True):
    """Build every transition and every move across these matches."""
    stats = stats if stats is not None else collections.defaultdict(int)
    transitions, moves = [], []

    chunk = config.MATCH_CHUNK_SIZE
    for start in range(0, len(match_codes), chunk):
        batch = match_codes[start:start + chunk]
        if verbose:
            print(f"  chunk {start // chunk + 1}: {len(batch)} matches",
                  flush=True)
        play_rows = snowflake_io.fetch_plays(cur, batch, time_column)
        score_rows = snowflake_io.fetch_scores(cur, batch)
        indexes = {}
        for stream in (directional.PROD, directional.CANDIDATE):
            indexes[stream] = directional.index_by_message(
                snowflake_io.fetch_quotes(cur, config.STREAMS[stream], batch))

        plays_by_match = collections.defaultdict(list)
        for row in play_rows:
            plays_by_match[row[0]].append(row)
        scores_by_match = collections.defaultdict(list)
        for row in score_rows:
            scores_by_match[row[0]].append(row)

        for match_code in batch:
            plays = [drives.PlayRow(event_message_count=r[1], period_number=r[2],
                                    offensive_team=r[3], down_number=r[4],
                                    distance=r[5], field_position=r[6],
                                    play_time=r[7])
                     for r in plays_by_match.get(match_code, [])]
            scores = [drives.ScoreRow(event_message_count=r[1],
                                      period_number=r[2], p1_change=r[3],
                                      p2_change=r[4], p1_cumulative=r[5],
                                      p2_cumulative=r[6])
                      for r in scores_by_match.get(match_code, [])]
            if not plays:
                stats["match_no_plays"] += 1
                continue
            match_transitions = transitions_for_match(match_code, plays, scores)
            transitions.extend(match_transitions)
            moves.extend(moves_for_transitions(match_transitions, indexes,
                                               stats))
    return transitions, moves, stats


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

TRANSITION_FIELDS = [
    "match_code", "drive_number", "period_number", "offensive_team",
    "from_message", "to_message", "in_drive",
    "down_before", "distance_before", "field_before", "field_after",
    "yards", "points", "outcome", "sign",
]

MOVE_FIELDS = TRANSITION_FIELDS + [
    "stream", "market_id", "market", "selection", "basis",
    "line_before", "line_after",
    "before", "after", "delta", "expected", "signed", "flat", "hit",
]


def transition_row(t):
    return {
        "match_code": t.match_code, "drive_number": t.drive_number,
        "period_number": t.period_number, "offensive_team": t.offensive_team,
        "from_message": t.from_message, "to_message": t.to_message,
        "in_drive": int(t.in_drive), "down_before": t.down_before,
        "distance_before": t.distance_before, "field_before": t.field_before,
        "field_after": "" if t.field_after is None else t.field_after,
        "yards": "" if t.yards is None else t.yards,
        "points": t.points, "outcome": t.outcome, "sign": t.sign,
    }


def move_row(move):
    """One move, carrying its whole transition so the file needs no join."""
    row = dict(transition_row(move.transition))
    row.update({
        "stream": move.stream, "market_id": move.market_id,
        "market": markets.market_group(move.market_id),
        "selection": markets.selection_label(move.market_id),
        "basis": move.basis,
        "line_before": "" if move.line_before is None else move.line_before,
        "line_after": "" if move.line_after is None else move.line_after,
        "before": move.before, "after": move.after,
        "delta": round(move.delta, 6), "expected": move.expected,
        "signed": round(move.signed, 6), "flat": int(move.flat),
        "hit": "" if move.hit is None else int(move.hit),
    })
    return row
