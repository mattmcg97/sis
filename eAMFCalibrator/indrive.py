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
# Drive outcomes
# ---------------------------------------------------------------------------

NO_POINTS = "no_points"          # the drive ended and nobody scored on it

HANDOVER = "handover"            # the next drive belongs to the other side
SAME_TEAM = "same_team"          # the next drive carries the SAME label
MATCH_END = "match_end"          # there is no next drive

DRIVE_OUTCOME_ORDER = [TOUCHDOWN, FIELD_GOAL, EXTRA_POINT, SCORE,
                       POINTS_AGAINST, NO_POINTS]


@dataclass(frozen=True)
class DriveOutcome:
    """One drive, what it produced, and the scoreboard either side of it."""
    match_code: str
    drive_number: int
    period_number: Optional[int]
    offensive_team: Optional[str]
    first_message: int
    last_message: int
    n_plays: int
    start_field: Optional[int]
    end_field: Optional[int]
    score_p1_before: int
    score_p2_before: int
    score_p1_after: int
    score_p2_after: int
    points_for: int
    points_against: int
    outcome: str
    ended: str

    @property
    def score_diff_before(self):
        return self.score_p1_before - self.score_p2_before

    @property
    def score_diff_after(self):
        return self.score_p1_after - self.score_p2_after


def _classify_drive(points_for, points_against):
    if points_for >= TOUCHDOWN_POINTS:
        return TOUCHDOWN
    if points_for == FIELD_GOAL_POINTS:
        return FIELD_GOAL
    if points_for in (1, 2):
        return EXTRA_POINT
    if points_for > 0:
        return SCORE
    if points_against > 0:
        return POINTS_AGAINST
    return NO_POINTS


def drive_outcomes(match_code, plays, scores):
    """One row per drive: what it produced and what the scoreboard did.

    Every drive gets one, including the last of the match -- which has no
    following drive, so the transition view cannot see it at all.

    The scoring windows PARTITION the match. Drive N owns every score from
    where drive N-1's window ended up to the message drive N+1 opens on,
    the first drive owns everything before that and the last owns
    everything after. No score belongs to two drives and none belongs to
    none, which is what makes `reconcile` a real test rather than a
    restatement.
    """
    drive_list = drives.build_drives(match_code, plays, scores)
    if not drive_list:
        return []
    scores = sorted(scores, key=lambda s: s.event_message_count)

    # Each drive's window closes where the next one opens, so one drive's
    # upper bound is the next one's lower bound. None at either end means
    # unbounded: the first drive owns everything before it, the last owns
    # everything after.
    uppers = [(drive_list[i + 1].first.event_message_count
               if i + 1 < len(drive_list) else None)
              for i in range(len(drive_list))]
    lowers = [None] + uppers[:-1]
    boundaries = list(zip(drive_list, lowers, uppers))

    rows = []
    for index, (drive, low, high) in enumerate(boundaries):
        window = [s for s in scores
                  if (low is None or s.event_message_count >= low)
                  and (high is None or s.event_message_count < high)]
        p1 = sum(s.p1_change or 0 for s in window)
        p2 = sum(s.p2_change or 0 for s in window)
        if drive.offensive_team == HOME:
            points_for, points_against = p1, p2
        elif drive.offensive_team == AWAY:
            points_for, points_against = p2, p1
        else:
            points_for = points_against = 0

        before = score_before(scores, low)
        following = (drive_list[index + 1]
                     if index + 1 < len(drive_list) else None)
        if following is None:
            ended = MATCH_END
        elif following.offensive_team == drive.offensive_team:
            ended = SAME_TEAM
        else:
            ended = HANDOVER

        rows.append(DriveOutcome(
            match_code=match_code,
            drive_number=drive.drive_number,
            period_number=drive.period_number,
            offensive_team=drive.offensive_team,
            first_message=drive.first.event_message_count,
            last_message=drive.last.event_message_count,
            n_plays=len(drive.plays),
            start_field=drive.first.field_position,
            end_field=drive.last.field_position,
            score_p1_before=before[0], score_p2_before=before[1],
            score_p1_after=before[0] + p1, score_p2_after=before[1] + p2,
            points_for=points_for, points_against=points_against,
            outcome=_classify_drive(points_for, points_against),
            ended=ended,
        ))
    return rows


def score_before(scores, message):
    """The scoreboard as it stood entering a drive's scoring window."""
    if message is None:
        return 0, 0
    p1 = sum(s.p1_change or 0 for s in scores if s.event_message_count < message)
    p2 = sum(s.p2_change or 0 for s in scores if s.event_message_count < message)
    return p1, p2


def reconcile(outcomes, scores):
    """Do the points the drives claim add up to the points the feed sent?

    The one test that catches attribution silently going wrong. Every
    score row belongs to exactly one drive's window, so the totals have to
    match; if they do not, a drive boundary is in the wrong place or a
    score is being counted twice.
    """
    claimed_p1 = sum(o.score_p1_after - o.score_p1_before for o in outcomes)
    claimed_p2 = sum(o.score_p2_after - o.score_p2_before for o in outcomes)
    feed_p1 = sum(s.p1_change or 0 for s in scores)
    feed_p2 = sum(s.p2_change or 0 for s in scores)
    return {"claimed_p1": claimed_p1, "claimed_p2": claimed_p2,
            "feed_p1": feed_p1, "feed_p2": feed_p2,
            "ok": claimed_p1 == feed_p1 and claimed_p2 == feed_p2}


def drive_census(outcomes):
    """How many drives ended each way, and what they were worth."""
    out = collections.OrderedDict()
    for outcome in DRIVE_OUTCOME_ORDER:
        rows = [o for o in outcomes if o.outcome == outcome]
        out[outcome] = {
            "n": len(rows),
            "matches": len({o.match_code for o in rows}),
            "points": sum(o.points_for for o in rows),
            "against": sum(o.points_against for o in rows),
            "mean_plays": (sum(o.n_plays for o in rows) / len(rows)) if rows else None,
        }
    return out


DRIVE_FIELDS = [
    "match_code", "drive_number", "period_number", "offensive_team",
    "first_message", "last_message", "n_plays", "start_field", "end_field",
    "score_p1_before", "score_p2_before", "score_p1_after", "score_p2_after",
    "score_diff_before", "score_diff_after",
    "points_for", "points_against", "outcome", "ended",
]


def drive_row(o):
    return {
        "match_code": o.match_code, "drive_number": o.drive_number,
        "period_number": o.period_number, "offensive_team": o.offensive_team,
        "first_message": o.first_message, "last_message": o.last_message,
        "n_plays": o.n_plays,
        "start_field": "" if o.start_field is None else o.start_field,
        "end_field": "" if o.end_field is None else o.end_field,
        "score_p1_before": o.score_p1_before,
        "score_p2_before": o.score_p2_before,
        "score_p1_after": o.score_p1_after,
        "score_p2_after": o.score_p2_after,
        "score_diff_before": o.score_diff_before,
        "score_diff_after": o.score_diff_after,
        "points_for": o.points_for, "points_against": o.points_against,
        "outcome": o.outcome, "ended": o.ended,
    }

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


class Sink:
    """Collects the in-drive analysis off another pass's fetched rows.

    directional.build_pairs already holds the plays, the scores and both
    quote indexes for a chunk of matches, which is everything this
    analysis reads. Riding along costs one function call per match; a
    second pass would cost a second set of queries.
    """

    def __init__(self):
        self.transitions = []
        self.moves = []
        self.outcomes = []

    def add(self, match_code, plays, scores, indexes, stats=None):
        stats = stats if stats is not None else collections.defaultdict(int)
        transitions = transitions_for_match(match_code, plays, scores)
        self.transitions.extend(transitions)
        self.moves.extend(moves_for_transitions(transitions, indexes, stats))
        outcomes = drive_outcomes(match_code, plays, scores)
        self.outcomes.extend(outcomes)
        if not reconcile(outcomes, scores)["ok"]:
            stats["match_points_do_not_reconcile"] += 1

    def summary(self, n_bootstrap=300):
        """The high-level reading, for a report that is not about this."""
        if not self.moves:
            return None
        result = report(self.moves, n_bootstrap=n_bootstrap)
        outcomes = self.outcomes
        matches = len({o.match_code for o in outcomes}) or 1
        points = sum(o.points_for + o.points_against for o in outcomes)
        scored = sum(1 for o in outcomes if o.points_for or o.points_against)
        return {
            "result": result,
            "drives": len(outcomes),
            "drives_per_match": len(outcomes) / matches,
            "points": points,
            "points_per_drive": (points / len(outcomes)) if outcomes else None,
            "scoring_share": (scored / len(outcomes)) if outcomes else None,
            "census": drive_census(outcomes),
            "transitions": len(self.transitions),
        }


def run(cur, match_codes, time_column, stats=None, n_bootstrap=1000,
        verbose=True):
    """Build every transition and every move across these matches."""
    stats = stats if stats is not None else collections.defaultdict(int)
    transitions, moves, outcomes = [], [], []

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
            match_drives = drive_outcomes(match_code, plays, scores)
            outcomes.extend(match_drives)
            if not reconcile(match_drives, scores)["ok"]:
                stats["match_points_do_not_reconcile"] += 1
    return transitions, moves, outcomes, stats


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


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

ERROR = "error"
WARN = "warn"
NOTE = "note"

MOSTLY_FLAT = 0.5          # above this share, the rate is a minority report
THIN_DECIDED = 200         # below this, a rate says very little
COVERAGE_RATIO = 0.5       # of the median moves per transition


def checks(result, transitions, moves, outcomes=(), scores=()):
    """Findings about the ANALYSIS, not about the models.

    The distinction is the point. Two independently built models agreeing
    with each other in the WRONG direction, on thousands of plays, is not
    two broken models -- it is one miscoded expectation. Anything this
    turns up should be read as "the check is wrong" first and "the model
    is wrong" second, which is the opposite of how the rest of the suite
    reads its own output.
    """
    out = []
    by_stream = (result or {}).get("streams") or {}
    streams = [s for s in (directional.PROD, directional.CANDIDATE)
               if (by_stream.get(s) or {}).get("overall", {}).get("n")]

    # 1. A class below a coin on BOTH streams: the expectation is backwards.
    for outcome in OUTCOME_ORDER:
        rates = []
        for stream in streams:
            b = result["streams"][stream]["by_outcome"].get(outcome)
            if not b or not b["decided"] or b["ci_high"] is None:
                continue
            rates.append((stream, b))
        if len(rates) < len(streams) or not rates:
            continue
        if all(b["ci_high"] < 0.5 for _, b in rates):
            worst = min(b["rate"] for _, b in rates)
            out.append((ERROR, outcome,
                        f"below a coin on every stream ({_pct(worst)} at worst, "
                        f"interval entirely under 50%) -- two models agreeing "
                        f"this far in the wrong direction means the expected "
                        f"direction for this class is backwards, not that both "
                        f"models are"))

    # 2. A class that barely moves at all: the rate is a minority report.
    for outcome in OUTCOME_ORDER:
        for stream in streams:
            b = result["streams"][stream]["by_outcome"].get(outcome)
            if b and b["n"] and (b["flat_share"] or 0) > MOSTLY_FLAT:
                out.append((WARN, outcome,
                            f"{stream}: {_pct(b['flat_share'])} of {b['n']:,} "
                            f"moves were FLAT, so the rate is computed on "
                            f"{b['decided']:,} rows. The finding is that the "
                            f"price does not move, not the percentage"))
            break   # one stream is enough to make the point

    # 3. A class the quotes barely cover: it cannot be compared to the rest.
    per_transition = {}
    for outcome in OUTCOME_ORDER:
        rows = [t for t in transitions if t.outcome == outcome and t.scorable]
        if not rows:
            continue
        mine = sum(1 for m in moves if m.transition.outcome == outcome)
        per_transition[outcome] = mine / len(rows)
    if len(per_transition) > 2:
        median = sorted(per_transition.values())[len(per_transition) // 2]
        for outcome, rate in sorted(per_transition.items(), key=lambda kv: kv[1]):
            if rate < median * COVERAGE_RATIO:
                out.append((WARN, outcome,
                            f"{rate:.2f} moves per transition against a median "
                            f"of {median:.2f} -- the feed quotes these messages "
                            f"far less often, so this class is not measured the "
                            f"same way as the others"))

    # 4. Thin rows, so nobody reads a percentage off four plays.
    for outcome in OUTCOME_ORDER:
        for stream in streams[:1]:
            b = result["streams"][stream]["by_outcome"].get(outcome)
            if b and b["n"] and b["decided"] < THIN_DECIDED:
                out.append((NOTE, outcome,
                            f"only {b['decided']:,} decided moves"))

    # 5. An invariant. A line-basis move exists BECAUSE the line changed,
    #    so it cannot also have stood still. If this ever fires, the basis
    #    is being set somewhere it should not be.
    flat_lines = sum(1 for m in moves if m.basis == LINE and m.flat)
    if flat_lines:
        out.append((ERROR, "basis",
                    f"{flat_lines:,} line-basis moves are flat, which cannot "
                    f"happen: a move is scored on the line only where the line "
                    f"changed"))

    # 6. Do the points the drives claim add up to the points the feed sent?
    #    Every score belongs to exactly one drive's window, so a mismatch
    #    means a boundary is in the wrong place or a score is counted twice.
    if outcomes:
        by_match = collections.defaultdict(list)
        for o in outcomes:
            by_match[o.match_code].append(o)
        claimed = sum(o.score_p1_after - o.score_p1_before for o in outcomes)
        claimed += sum(o.score_p2_after - o.score_p2_before for o in outcomes)
        if scores:
            feed = sum((s.p1_change or 0) + (s.p2_change or 0) for s in scores)
            if claimed != feed:
                out.append((ERROR, "reconcile",
                            f"the drives claim {claimed:,} points and the feed "
                            f"sent {feed:,} -- a drive boundary is in the wrong "
                            f"place or a score is being counted twice"))
        no_outcome = sum(1 for o in outcomes if o.ended == SAME_TEAM)
        if no_outcome:
            out.append((NOTE, "drives",
                        f"{no_outcome:,} of {len(outcomes):,} drives are "
                        f"followed by another drive with the SAME team label, "
                        f"so the feed never said possession changed"))

    # 7. Both sides of a market should read the same where the feed
    #    publishes complements. Where they do not, it does not.
    for stream in streams:
        by_selection = result["streams"][stream]["by_selection"]
        grouped = collections.defaultdict(list)
        for (market, selection), b in by_selection.items():
            if b["rate"] is not None:
                grouped[market].append((selection, b["rate"]))
        for market, sides in grouped.items():
            if len(sides) != 2:
                continue
            gap = abs(sides[0][1] - sides[1][1])
            if gap > 0.02:
                out.append((NOTE, str(market),
                            f"{stream}: the two sides differ by {_pct(gap)} "
                            f"({sides[0][0]} {_pct(sides[0][1])}, "
                            f"{sides[1][0]} {_pct(sides[1][1])}), so the feed "
                            f"is not publishing exact complements here"))
    return out


def _pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"
