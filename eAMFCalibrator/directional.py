"""Paired head-to-head: at each snapshot, which model was closer to the truth?

The cross-sectional calibrator asks "is this model's 30% really 30%?", which
needs a lot of snapshots per cell before the realized rate means anything.
With a day of data most cells are too thin to read.

This asks a cheaper question that one day can answer: at the same snapshot,
on the same selection, |prod - y| against |candidate - y|. Whichever is
smaller wins that pair. The between-snapshot variance that swamps the
cross-sectional view cancels, because both models are scored on the exact
same game state and the exact same outcome.

Pairing is on EVENT_MESSAGE_COUNT, not on each stream's own nearest quote in
time. Both streams carry the same feed sequence, so the same message is the
same event for both. Matching each independently on time would let one land
0.1s from the snapshot and the other 2.5s away, scoring two different game
states against one outcome and flattering whichever got the closer quote.

Two inference levels are reported, because pairs inside a match are not
independent:

  row level    sign test over every pair. Sensitive, but a single match
               contributing 30 pairs counts 30 times.
  match level  each match votes once, by whichever model won more of its
               pairs. Far fewer observations, but they ARE independent.

When those two disagree, trust the match level.
"""

import math
import random
import statistics
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from . import buckets, config, handles, markets, metrics, snowflake_io
from .drives import PlayRow, ScoreRow, build_snapshots

CANDIDATE = "candidate"
PROD = "prod"
TIE = "tie"

# Two ways to decide a pair, because the streams do not always quote the
# same line.
PROBABILITY = "probability"   # whose stated probability was closer to its own 0/1
LINE = "line"                 # whose line was closer to what actually happened

LINE_EPSILON = 1e-6


@dataclass(frozen=True)
class PairedObservation:
    match_code: str
    drive_number: int
    period_number: Optional[int]
    score_diff: int
    offensive_team: Optional[str]
    market_id: int
    message_count: int
    message_gap: int
    prod_probability: float
    candidate_probability: float
    prod_line: Optional[float]
    candidate_line: Optional[float]
    # Each stream is graded against ITS OWN line, so the two outcomes can
    # differ: a total of 45 is over 44.5 but under 46.5.
    prod_outcome: Optional[bool]
    candidate_outcome: Optional[bool]
    realized: Optional[float]
    prod_decimal: Optional[float] = None
    candidate_decimal: Optional[float] = None
    publish_time: object = None

    @property
    def day(self):
        """Calendar day of the quote, for the per-day view."""
        return self.publish_time.date().isoformat() if self.publish_time else "unknown"

    @property
    def same_line(self):
        if self.prod_line is None and self.candidate_line is None:
            return True
        if self.prod_line is None or self.candidate_line is None:
            return False
        return abs(self.prod_line - self.candidate_line) < LINE_EPSILON

    @property
    def line_delta(self):
        if self.prod_line is None or self.candidate_line is None:
            return None
        return abs(self.prod_line - self.candidate_line)

    @property
    def prod_error(self):
        if self.prod_outcome is None:
            return None
        return abs(self.prod_probability - (1.0 if self.prod_outcome else 0.0))

    @property
    def candidate_error(self):
        if self.candidate_outcome is None:
            return None
        return abs(self.candidate_probability - (1.0 if self.candidate_outcome else 0.0))

    @property
    def prod_line_error(self):
        if self.prod_line is None or self.realized is None:
            return None
        return abs(self.prod_line - self.realized)

    @property
    def candidate_line_error(self):
        if self.candidate_line is None or self.realized is None:
            return None
        return abs(self.candidate_line - self.realized)

    def errors(self, mode):
        if mode == LINE:
            return self.prod_line_error, self.candidate_line_error
        return self.prod_error, self.candidate_error

    def comparable(self, mode):
        prod_error, candidate_error = self.errors(mode)
        return prod_error is not None and candidate_error is not None

    def winner(self, mode=PROBABILITY):
        prod_error, candidate_error = self.errors(mode)
        if prod_error is None or candidate_error is None:
            return None
        if candidate_error < prod_error:
            return CANDIDATE
        if prod_error < candidate_error:
            return PROD
        return TIE

    @property
    def disagreement(self):
        return abs(self.prod_probability - self.candidate_probability)


def index_by_message(quote_rows):
    """(match, market) -> {message: (probability, description, decimal, time)}."""
    out = defaultdict(dict)
    for match_code, market_id, publish_time, probability, decimal_odd, description, message in quote_rows:
        if message is None or probability is None:
            continue
        # A message carries one row per market; keep the first seen.
        out[(match_code, market_id)].setdefault(
            message,
            (float(probability), description,
             float(decimal_odd) if decimal_odd is not None else None,
             publish_time))
    return out


def common_messages(prod_index, candidate_index):
    """(match, market) -> sorted messages both streams quoted."""
    out = {}
    for key, prod_messages in prod_index.items():
        candidate_messages = candidate_index.get(key)
        if not candidate_messages:
            continue
        shared = sorted(set(prod_messages) & set(candidate_messages))
        if shared:
            out[key] = shared
    return out


def nearest_message(messages, target, max_gap):
    """Closest message to target, preferring an exact hit. None if too far."""
    if not messages or target is None:
        return None
    idx = bisect_left(messages, target)
    best = None
    for i in (idx - 1, idx):
        if 0 <= i < len(messages):
            gap = messages[i] - target
            if abs(gap) <= max_gap and (best is None or abs(gap) < abs(best[1])):
                best = (messages[i], gap)
    return best


def build_pairs(cur, match_codes, time_column, stats, scan=None):
    """Every paired observation for one chunk of matches.

    `scan` is the handle check, accumulating across chunks. Every match is
    scanned whether or not the result is acted on, so the report can say
    how big the problem is even when nothing is being excluded.
    """
    scan = scan if scan is not None else handles.Scan()
    prod_table = config.STREAMS[PROD]
    candidate_table = config.STREAMS[CANDIDATE]

    play_rows = snowflake_io.fetch_plays(cur, match_codes, time_column)
    score_rows = snowflake_io.fetch_scores(cur, match_codes)
    finals = snowflake_io.fetch_final_scores(cur, match_codes)
    prod_quotes = snowflake_io.fetch_quotes(cur, prod_table, match_codes)
    candidate_quotes = snowflake_io.fetch_quotes(cur, candidate_table, match_codes)

    prod_index = index_by_message(prod_quotes)
    candidate_index = index_by_message(candidate_quotes)
    shared = common_messages(prod_index, candidate_index)

    plays_by_match = defaultdict(list)
    for row in play_rows:
        plays_by_match[row[0]].append(row)
    scores_by_match = defaultdict(list)
    for row in score_rows:
        scores_by_match[row[0]].append(row)

    pairs = []
    for match_code in match_codes:
        plays = [
            PlayRow(event_message_count=r[1], period_number=r[2], offensive_team=r[3],
                    down_number=r[4], distance=r[5], field_position=r[6], play_time=r[7])
            for r in plays_by_match.get(match_code, [])
        ]
        scores = [
            ScoreRow(event_message_count=r[1], period_number=r[2], p1_change=r[3],
                     p2_change=r[4], p1_cumulative=r[5], p2_cumulative=r[6])
            for r in scores_by_match.get(match_code, [])
        ]
        final = finals.get(match_code)
        if not scan.add(match_code, scores, final, plays):
            stats["matches_with_flipped_handles"] += 1
            if config.EXCLUDE_FLIPPED_MATCHES:
                stats["matches_excluded_for_flipped_handles"] += 1
                continue

        if not plays:
            stats["matches_without_plays"] += 1
            continue

        if final is None or final[0] is None or final[1] is None:
            stats["matches_without_final"] += 1
            continue
        final_p1, final_p2 = final

        snapshots = build_snapshots(match_code, plays, scores)
        stats["snapshots"] += len(snapshots)

        for snap in snapshots:
            paired_any = False
            for market_id in markets.MARKET_IDS:
                messages = shared.get((match_code, market_id))
                if not messages:
                    stats["no_common_message_for_market"] += 1
                    continue

                hit = nearest_message(messages, snap.event_message_count,
                                      config.MAX_PAIR_MESSAGE_GAP)
                if hit is None:
                    stats["outside_message_gap"] += 1
                    continue
                message, gap = hit
                stats["exact_message_pair" if gap == 0 else "offset_message_pair"] += 1

                prod_probability, prod_description, prod_decimal, publish_time = \
                    prod_index[(match_code, market_id)][message]
                candidate_probability, candidate_description, candidate_decimal, _ = \
                    candidate_index[(match_code, market_id)][message]

                needs_line = markets.needs_line(market_id)
                prod_line = markets.parse_line(prod_description) if needs_line else None
                candidate_line = markets.parse_line(candidate_description) if needs_line else None
                if needs_line and (prod_line is None or candidate_line is None):
                    stats["unparsed_line"] += 1
                    continue

                # Each stream graded against its own line. Grading the
                # candidate against prod's line would score it on a question
                # it never asked.
                prod_outcome = markets.resolve(market_id, prod_line, final_p1, final_p2)
                candidate_outcome = markets.resolve(market_id, candidate_line, final_p1, final_p2)
                realized = markets.realized_value(market_id, final_p1, final_p2)

                if prod_outcome is None:
                    stats["prod_push"] += 1
                if candidate_outcome is None:
                    stats["candidate_push"] += 1
                if prod_outcome is None and candidate_outcome is None and realized is None:
                    stats["unresolvable"] += 1
                    continue

                observation = PairedObservation(
                    match_code=match_code,
                    drive_number=snap.drive_number,
                    period_number=snap.period_number,
                    score_diff=snap.score_diff,
                    offensive_team=snap.offensive_team,
                    market_id=market_id,
                    message_count=message,
                    message_gap=gap,
                    prod_probability=prod_probability / 100.0,
                    candidate_probability=candidate_probability / 100.0,
                    prod_line=prod_line,
                    candidate_line=candidate_line,
                    prod_outcome=prod_outcome,
                    candidate_outcome=candidate_outcome,
                    realized=realized,
                    prod_decimal=prod_decimal,
                    candidate_decimal=candidate_decimal,
                    publish_time=publish_time,
                )
                stats["same_line" if observation.same_line else "different_line"] += 1
                paired_any = True
                pairs.append(observation)
            if paired_any:
                stats["snapshots_paired"] += 1

    return pairs


def run(verbose=True):
    """Build every paired observation across the window."""
    stats = defaultdict(int)
    pairs = []
    header = {}
    scan = handles.Scan()

    conn = snowflake_io.get_connection()
    try:
        with conn.cursor() as cur:
            time_column, _ = snowflake_io.detect_play_time_column(cur)
            header["play_time_column"] = time_column

            for key in (PROD, CANDIDATE):
                n_rows, n_matches, first, last = snowflake_io.stream_window_summary(
                    cur, config.STREAMS[key])
                header[key] = {"rows": n_rows, "matches": n_matches,
                               "first": first, "last": last}

            # Only matches both streams cover can be paired at all.
            prod_matches = set(snowflake_io.match_universe(cur, config.STREAMS[PROD]))
            candidate_matches = set(snowflake_io.match_universe(cur, config.STREAMS[CANDIDATE]))
            match_codes = sorted(prod_matches & candidate_matches)
            header["prod_only_matches"] = len(prod_matches - candidate_matches)
            header["candidate_only_matches"] = len(candidate_matches - prod_matches)
            header["paired_matches"] = len(match_codes)
            if verbose:
                print(f"  {len(match_codes)} matches in both streams "
                      f"({len(prod_matches - candidate_matches)} prod-only, "
                      f"{len(candidate_matches - prod_matches)} candidate-only)")

            chunk = config.MATCH_CHUNK_SIZE
            for start in range(0, len(match_codes), chunk):
                batch = match_codes[start:start + chunk]
                if verbose:
                    print(f"  chunk {start // chunk + 1}: {len(batch)} matches", flush=True)
                pairs.extend(build_pairs(cur, batch, time_column, stats, scan))
    finally:
        conn.close()

    return pairs, stats, header, scan.summary()


def tally(pairs, mode=PROBABILITY):
    """Win/loss/tie counts and error means, under one decision mode.

    Pairs the mode cannot decide are excluded and counted separately, so a
    mode that only applies to part of the data says so rather than quietly
    shrinking the sample.
    """
    usable = [p for p in pairs if p.comparable(mode)]
    result = {"mode": mode, "n": len(usable), "n_offered": len(pairs),
              CANDIDATE: 0, PROD: 0, TIE: 0,
              "prod_error_sum": 0.0, "candidate_error_sum": 0.0,
              "prod_brier_sum": 0.0, "candidate_brier_sum": 0.0,
              "matches": set()}
    for pair in usable:
        prod_error, candidate_error = pair.errors(mode)
        result[pair.winner(mode)] += 1
        result["prod_error_sum"] += prod_error
        result["candidate_error_sum"] += candidate_error
        result["prod_brier_sum"] += prod_error ** 2
        result["candidate_brier_sum"] += candidate_error ** 2
        result["matches"].add(pair.match_code)
    return finalize(result)


def finalize(result):
    n = result["n"]
    decisive = result[CANDIDATE] + result[PROD]
    result["decisive"] = decisive
    result["candidate_win_rate"] = result[CANDIDATE] / decisive if decisive else None
    result["prod_mae"] = result["prod_error_sum"] / n if n else None
    result["candidate_mae"] = result["candidate_error_sum"] / n if n else None
    result["mae_delta"] = (result["prod_mae"] - result["candidate_mae"]) if n else None
    result["prod_brier"] = result["prod_brier_sum"] / n if n else None
    result["candidate_brier"] = result["candidate_brier_sum"] / n if n else None
    result["brier_delta"] = (result["prod_brier"] - result["candidate_brier"]) if n else None
    result["p_value"] = metrics.sign_test(result[CANDIDATE], result[PROD])
    result["n_matches"] = len(result["matches"])
    return result


def match_level_votes(pairs, mode=PROBABILITY):
    """One vote per match, by which model won more of that match's pairs.

    This is the clustering-robust reading: matches are independent, the
    pairs inside them are not.
    """
    per_match = defaultdict(lambda: {CANDIDATE: 0, PROD: 0, TIE: 0})
    for pair in pairs:
        won = pair.winner(mode)
        if won is not None:
            per_match[pair.match_code][won] += 1

    votes = {CANDIDATE: 0, PROD: 0, TIE: 0}
    for counts in per_match.values():
        if counts[CANDIDATE] > counts[PROD]:
            votes[CANDIDATE] += 1
        elif counts[PROD] > counts[CANDIDATE]:
            votes[PROD] += 1
        else:
            votes[TIE] += 1
    votes["n_matches"] = len(per_match)
    votes["decisive"] = votes[CANDIDATE] + votes[PROD]
    votes["candidate_win_rate"] = (votes[CANDIDATE] / votes["decisive"]
                                   if votes["decisive"] else None)
    votes["p_value"] = metrics.sign_test(votes[CANDIDATE], votes[PROD])
    return votes


# ---------------------------------------------------------------------------
# Paired error difference
# ---------------------------------------------------------------------------
#
# The win rate answers "which model was closer more often". It is also a weak
# test: if both models are unbiased around the same truth and differ only in
# noise, which one lands closer to the realized 0/1 is decided by which noise
# draw pointed at the outcome -- a coin flip regardless of variance. It
# discards magnitude, so it cannot see an improvement that consists of being
# less wrong. The paired difference in loss does see it, and computing it per
# match before averaging respects the clustering.

SQUARED = "brier"
ABSOLUTE = "mae"


def _loss(pair, kind, mode):
    prod_error, candidate_error = pair.errors(mode)
    if kind == SQUARED:
        return prod_error ** 2, candidate_error ** 2
    return prod_error, candidate_error


def per_match_deltas(pairs, kind=SQUARED, mode=PROBABILITY):
    """match -> mean(prod loss) - mean(candidate loss).

    Positive means the candidate carried less loss in that match.
    """
    totals = defaultdict(lambda: [0.0, 0.0, 0])
    for pair in pairs:
        if not pair.comparable(mode):
            continue
        prod_loss, candidate_loss = _loss(pair, kind, mode)
        bucket = totals[pair.match_code]
        bucket[0] += prod_loss
        bucket[1] += candidate_loss
        bucket[2] += 1
    return {match: (prod_sum - candidate_sum) / n
            for match, (prod_sum, candidate_sum, n) in totals.items()}


def paired_delta_summary(pairs, kind=SQUARED, mode=PROBABILITY,
                         n_bootstrap=2000, seed=0):
    """Match-clustered test on the paired loss difference.

    Bootstrap resamples MATCHES, not rows, so the confidence interval
    carries the same clustering assumption as the point estimate.
    """
    deltas = list(per_match_deltas(pairs, kind, mode).values())
    m = len(deltas)
    out = {"kind": kind, "mode": mode, "n_matches": m, "mean": None, "se": None,
           "t": None, "p_value": None, "ci_low": None, "ci_high": None,
           "sign_p_value": None,
           "matches_favouring_candidate": sum(1 for d in deltas if d > 0),
           "matches_favouring_prod": sum(1 for d in deltas if d < 0)}
    if m == 0:
        return out

    mean = sum(deltas) / m
    out["mean"] = mean
    out["sign_p_value"] = metrics.sign_test(out["matches_favouring_candidate"],
                                            out["matches_favouring_prod"])
    if m < 2:
        return out

    sd = statistics.stdev(deltas)
    se = sd / math.sqrt(m)
    out["se"] = se
    if se > 0:
        t = mean / se
        out["t"] = t
        # Normal approximation; fine at a few dozen matches and up.
        out["p_value"] = math.erfc(abs(t) / math.sqrt(2))

    rnd = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [deltas[rnd.randrange(m)] for _ in range(m)]
        means.append(sum(sample) / m)
    means.sort()
    out["ci_low"] = means[int(0.025 * n_bootstrap)]
    out["ci_high"] = means[min(int(0.975 * n_bootstrap), n_bootstrap - 1)]
    return out


# ---------------------------------------------------------------------------
# Line agreement
# ---------------------------------------------------------------------------

def line_agreement(pairs):
    """How often the two streams quote the same line, and by how much not.

    The whole comparison turns on this. If the candidate quotes a different
    line, its probability answers a different question, and only the
    line-closeness view compares the two fairly.
    """
    by_market = defaultdict(lambda: {"n": 0, "same": 0,
                                     "prod_whole": 0, "prod_half": 0,
                                     "candidate_whole": 0, "candidate_half": 0})
    deltas = []
    differing_deltas = []
    same = 0
    for pair in pairs:
        group = markets.market_group(pair.market_id)
        bucket = by_market[group]
        bucket["n"] += 1
        if pair.same_line:
            same += 1
            bucket["same"] += 1
        if pair.line_delta is not None:
            deltas.append(pair.line_delta)
            if not pair.same_line:
                differing_deltas.append(pair.line_delta)
        for side, line in (("prod", pair.prod_line), ("candidate", pair.candidate_line)):
            if line is None:
                continue
            whole = abs(line - round(line)) < LINE_EPSILON
            bucket[f"{side}_{'whole' if whole else 'half'}"] += 1

    out = {"n": len(pairs), "same": same, "different": len(pairs) - same,
           "same_rate": same / len(pairs) if pairs else None,
           "by_market": dict(by_market),
           "delta_median": None, "delta_mean": None, "delta_max": None,
           "differing_median": None, "differing_mean": None,
           "differing_max": None, "differing_n": 0}
    # Two distributions, because mixing them is misleading: including the
    # same-line zeros drags the median to 0 and hides how far apart the
    # lines actually are when they do differ.
    if deltas:
        ordered = sorted(deltas)
        out["delta_median"] = ordered[len(ordered) // 2]
        out["delta_mean"] = sum(ordered) / len(ordered)
        out["delta_max"] = ordered[-1]
    if differing_deltas:
        ordered = sorted(differing_deltas)
        out["differing_median"] = ordered[len(ordered) // 2]
        out["differing_mean"] = sum(ordered) / len(ordered)
        out["differing_max"] = ordered[-1]
        out["differing_n"] = len(ordered)
    return out


def line_delta_band(pair):
    delta = pair.line_delta
    if delta is None:
        return "no line"
    if delta < LINE_EPSILON:
        return "same line"
    if delta <= 0.5:
        return "0.5"
    if delta <= 1.0:
        return "1.0"
    if delta <= 2.0:
        return "2.0"
    return "> 2.0"


LINE_DELTA_ORDER = ["same line", "0.5", "1.0", "2.0", "> 2.0", "no line"]


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def group_by(pairs, key_function, mode=PROBABILITY):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[key_function(pair)].append(pair)
    return {key: tally(group, mode) for key, group in grouped.items()}


def split_by_line(pairs):
    """(same-line pairs, different-line pairs)."""
    same = [p for p in pairs if p.same_line]
    different = [p for p in pairs if not p.same_line]
    return same, different


def disagreement_band(pair):
    gap = pair.disagreement
    for low, high, label in config.DISAGREEMENT_BANDS:
        if gap >= low and (high is None or gap < high):
            return label
    return "unknown"


def probability_band(pair):
    """Banded on prod's probability, the incumbent's view of the state."""
    edge = min(int(pair.prod_probability * 10), 9)
    return f"{edge / 10:.1f}-{(edge + 1) / 10:.1f}"


def market_label(pair):
    return markets.market_group(pair.market_id)


def cell_key(pair):
    return (
        buckets.score_diff_bucket(pair.score_diff),
        buckets.time_bucket(pair.period_number, pair.drive_number),
        buckets.possession_bucket(pair.offensive_team),
    )


# ---------------------------------------------------------------------------
# Assembled summary
# ---------------------------------------------------------------------------

MARKET_ORDER = [markets.MONEYLINE, markets.SPREAD, markets.TOTAL]


def market_blocks(pairs, mode, n_bootstrap=2000):
    """Per-market tallies WITH their own match-clustered paired delta.

    The pooled figure averages over markets, so a real effect confined to
    one of them gets diluted by the flat ones. Each market needs its own
    clustered test to be read on its own.
    """
    out = {}
    for group in MARKET_ORDER:
        subset = [p for p in pairs if markets.market_group(p.market_id) == group]
        if not subset:
            continue
        out[group] = {
            "tally": tally(subset, mode),
            "votes": match_level_votes(subset, mode),
            "brier": paired_delta_summary(subset, SQUARED, mode, n_bootstrap),
            "mae": paired_delta_summary(subset, ABSOLUTE, mode, n_bootstrap),
        }
    return out


def build_summary(pairs, n_bootstrap=2000):
    """Everything both the console and the HTML report need.

    Assembled once so the two cannot drift apart.

    The split matters because the two streams do not always quote the same
    line. Where they do, their probabilities answer the same question and can
    be compared directly. Where they do not, the fair comparison is which
    LINE landed closer to what actually happened -- each probability still
    refers to its own line, so comparing them head to head would be scoring
    two different questions against one outcome.
    """
    same, different = split_by_line(pairs)

    def block(subset, mode):
        return {
            "n": len(subset),
            "overall": tally(subset, mode),
            "votes": match_level_votes(subset, mode),
            "brier": paired_delta_summary(subset, SQUARED, mode, n_bootstrap),
            "mae": paired_delta_summary(subset, ABSOLUTE, mode, n_bootstrap),
            "by_market": group_by(subset, market_label, mode),
            "markets": market_blocks(subset, mode, n_bootstrap),
        }

    return {
        "pairs": len(pairs),
        "lines": line_agreement(pairs),
        "same_line": block(same, PROBABILITY),
        "different_line": block(different, LINE),
        # Secondary view: on different-line pairs, each stream scored against
        # its own line. Fair, but it measures line and probability together.
        "different_line_probability": block(different, PROBABILITY),
        "all_probability": block(pairs, PROBABILITY),
        "by_line_delta": group_by(pairs, line_delta_band, LINE),
    }


# ---------------------------------------------------------------------------
# Cross-sectional calibration, under the line rule
# ---------------------------------------------------------------------------
#
# The line rule: a different line overrules the probability entirely, so a
# probability comparison is only made where both streams quoted the same
# line. Where they differ, the comparison is on the line.
#
# Restricting to same-line pairs buys a property that makes the cell tables
# far easier to read: the same line means the same question, so the two
# streams share ONE realized outcome per cell. The realized column is
# common to both and only the predicted values differ, which turns each
# cell into a direct "whose number was nearer the truth".

def calibration_cells(pairs, key_function, n_bootstrap=500, market_ids=None):
    """Per (cell, market), both streams' calibration on an identical population.

    Two restrictions, both load-bearing:

    Same line only, so the two streams answer the same question and share one
    realized outcome per cell.

    One selection per market (config.CANONICAL_SELECTIONS). Pooling both sides
    of a market cancels the measurement dead: the sides are complements, so
    every (p, y) arrives with a mirror (1-p, 1-y) and the realized rate and
    the mean prediction BOTH average to exactly 0.5 no matter how good or bad
    the model is. Splitting by market matters for the same reason at one
    remove -- moneyline, spread and total are different questions with
    different base rates, and averaging them gives a rate that describes none
    of them.

    Keyed by (cell, market group).
    """
    allowed = set(config.CANONICAL_SELECTIONS if market_ids is None else market_ids)
    same, _ = split_by_line(pairs)
    grouped = defaultdict(list)
    for pair in same:
        if pair.market_id in allowed and pair.comparable(PROBABILITY):
            grouped[(key_function(pair), markets.market_group(pair.market_id))].append(pair)

    cells = {}
    for key, subset in grouped.items():
        prod_points = [(p.prod_probability, p.prod_outcome) for p in subset]
        candidate_points = [(p.candidate_probability, p.candidate_outcome) for p in subset]
        prod_stats = metrics.summarize(prod_points, config.N_RELIABILITY_BINS)
        candidate_stats = metrics.summarize(candidate_points, config.N_RELIABILITY_BINS)
        brier = paired_delta_summary(subset, SQUARED, PROBABILITY, n_bootstrap)
        cells[key] = {
            "n": len(subset),
            "matches": len({p.match_code for p in subset}),
            "selection": config.CANONICAL_SELECTIONS.get(subset[0].market_id, ""),
            # Same line means same question, so both streams resolve to the
            # same outcome; realized is one number, not two.
            "realized": prod_stats["realized"],
            "realized_check": candidate_stats["realized"],
            "prod_predicted": prod_stats["mean_predicted"],
            "prod_gap": prod_stats["gap"],
            "prod_brier": prod_stats["brier"],
            "candidate_predicted": candidate_stats["mean_predicted"],
            "candidate_gap": candidate_stats["gap"],
            "candidate_brier": candidate_stats["brier"],
            "brier_delta": brier["mean"],
            "ci_low": brier["ci_low"],
            "ci_high": brier["ci_high"],
            "p_value": brier["p_value"],
            "winner": (CANDIDATE if (candidate_stats["gap"] is not None
                                     and abs(candidate_stats["gap"]) < abs(prod_stats["gap"]))
                       else PROD),
        }
    return cells


def line_cells(pairs, key_function, n_bootstrap=500):
    """Per cell, whose LINE landed closer, for the different-line pairs."""
    _, different = split_by_line(pairs)
    grouped = defaultdict(list)
    for pair in different:
        if pair.comparable(LINE):
            grouped[key_function(pair)].append(pair)

    cells = {}
    for key, subset in grouped.items():
        prod_errors = [p.prod_line_error for p in subset]
        candidate_errors = [p.candidate_line_error for p in subset]
        points = paired_delta_summary(subset, ABSOLUTE, LINE, n_bootstrap)
        cells[key] = {
            "n": len(subset),
            "matches": len({p.match_code for p in subset}),
            "mean_line_gap": sum(p.line_delta for p in subset) / len(subset),
            "prod_line_error": sum(prod_errors) / len(prod_errors),
            "candidate_line_error": sum(candidate_errors) / len(candidate_errors),
            "points_delta": points["mean"],
            "ci_low": points["ci_low"],
            "ci_high": points["ci_high"],
            "p_value": points["p_value"],
        }
    return cells


def possession_label(pair):
    return buckets.possession_bucket(pair.offensive_team)


def score_label(pair):
    return buckets.score_diff_bucket(pair.score_diff)


def quarter_label(pair):
    return buckets.time_bucket(pair.period_number, pair.drive_number)


CROSS_AXES = [
    ("Score difference", score_label,
     lambda: [label for _, _, label in config.SCORE_DIFF_BUCKETS] + ["unknown"]),
    ("Quarter", quarter_label, lambda: ["Q1", "Q2", "Q3", "Q4", "OT", "unknown"]),
    ("Possession", possession_label, lambda: ["Home", "Away", "unknown"]),
]


# ---------------------------------------------------------------------------
# Are the two sides of a market actually complements?
# ---------------------------------------------------------------------------
#
# The cross-sectional view calibrates one selection per market on the
# argument that the other side carries no extra information: if Over is
# overpriced by 11 points then Under is underpriced by 11, the same fact with
# its sign flipped. That argument holds only if the two sides really are
# complements -- probabilities summing to 1, outcomes partitioning.
#
# It is worth testing rather than assuming, for two reasons. If the
# probabilities carry an overround the sides are not quite mirrors and the
# discarded side holds a little information. And if the outcomes do NOT
# partition -- both sides winning, or both losing -- then the two are not
# opposite sides of one market at all and the resolution logic is wrong.
#
# Spread is the one to watch. Market 52 reads "PLAYER 1 to score over L more
# than PLAYER 2" and 53 reads "PLAYER 2 to score over L more than PLAYER 1".
# If both carry the SAME L, they overlap rather than partition: at L = -2.5
# both win whenever the margin sits between -2.5 and +2.5.

def complement_report(pairs):
    """Per market: do the two sides sum to 1, and do their outcomes partition?"""
    grouped = defaultdict(dict)
    for pair in pairs:
        group = markets.market_group(pair.market_id)
        grouped[(pair.match_code, pair.message_count, group)][pair.market_id] = pair

    per_market = defaultdict(lambda: {
        "both_sides": 0, "prob_sums": [], "line_equal": 0,
        "outcomes_partition": 0, "both_won": 0, "both_lost": 0,
        "outcome_unresolved": 0})

    for (_, _, group), sides in grouped.items():
        if len(sides) != 2:
            continue
        first, second = (sides[k] for k in sorted(sides))
        bucket = per_market[group]
        bucket["both_sides"] += 1
        bucket["prob_sums"].append(first.prod_probability + second.prod_probability)

        if first.same_line and second.same_line:
            # Both sides agreed with the candidate; compare the two sides'
            # own lines to each other.
            pass
        if (first.prod_line is None and second.prod_line is None) or (
                first.prod_line is not None and second.prod_line is not None
                and abs(first.prod_line - second.prod_line) < LINE_EPSILON):
            bucket["line_equal"] += 1

        a, b = first.prod_outcome, second.prod_outcome
        if a is None or b is None:
            bucket["outcome_unresolved"] += 1
        elif a != b:
            bucket["outcomes_partition"] += 1
        elif a:
            bucket["both_won"] += 1
        else:
            bucket["both_lost"] += 1

    out = {}
    for group, bucket in per_market.items():
        sums = bucket["prob_sums"]
        out[group] = {
            "both_sides": bucket["both_sides"],
            "prob_sum_mean": sum(sums) / len(sums) if sums else None,
            "prob_sum_min": min(sums) if sums else None,
            "prob_sum_max": max(sums) if sums else None,
            "line_equal": bucket["line_equal"],
            "outcomes_partition": bucket["outcomes_partition"],
            "both_won": bucket["both_won"],
            "both_lost": bucket["both_lost"],
            "outcome_unresolved": bucket["outcome_unresolved"],
        }
    return out


def both_sides_calibration(pairs, n_bootstrap=200):
    """Calibration for every selection, so mirroring can be eyeballed.

    Not an extra finding -- a consistency check. If the two sides of a market
    are complements, the second row's realized rate is one minus the first's
    and its gap is the first's negated. If they are not, the pipeline is
    measuring something other than what it claims.
    """
    same, _ = split_by_line(pairs)
    grouped = defaultdict(list)
    for pair in same:
        if pair.comparable(PROBABILITY):
            grouped[pair.market_id].append(pair)

    out = {}
    for market_id in markets.MARKET_IDS:
        subset = grouped.get(market_id)
        if not subset:
            continue
        prod_stats = metrics.summarize(
            [(p.prod_probability, p.prod_outcome) for p in subset],
            config.N_RELIABILITY_BINS)
        candidate_stats = metrics.summarize(
            [(p.candidate_probability, p.candidate_outcome) for p in subset],
            config.N_RELIABILITY_BINS)
        out[market_id] = {
            "market": markets.market_group(market_id),
            "selection": markets.selection_label(market_id),
            "n": len(subset),
            "matches": len({p.match_code for p in subset}),
            "realized": prod_stats["realized"],
            "prod_predicted": prod_stats["mean_predicted"],
            "prod_gap": prod_stats["gap"],
            "candidate_predicted": candidate_stats["mean_predicted"],
            "candidate_gap": candidate_stats["gap"],
            "canonical": market_id in config.CANONICAL_SELECTIONS,
        }
    return out


def spread_interpretation_report(pairs):
    """Which reading of the spread's second selection does the data support?

    Both readings are recomputed here from the line and the realized margin
    rather than taken from the stored outcome, so the answer does not depend
    on whichever resolution was active when the pairs were built.

    The decisive combination is probabilities summing to 1 while the two
    sides carry the SAME line. Complementary probabilities require
    complementary events, so that pairing is proof the literal reading --
    two independent propositions -- is wrong.
    """
    grouped = defaultdict(dict)
    for pair in pairs:
        if markets.market_group(pair.market_id) == markets.SPREAD:
            grouped[(pair.match_code, pair.message_count)][pair.market_id] = pair

    out = {"n": 0, "lines_equal": 0, "lines_mirrored": 0, "lines_other": 0,
           "prob_sums": [], "literal_partition": 0, "literal_both_won": 0,
           "literal_both_lost": 0, "readings_agree": 0, "unresolved": 0}

    for sides in grouped.values():
        if 52 not in sides or 53 not in sides:
            continue
        home, away = sides[52], sides[53]
        if (home.prod_line is None or away.prod_line is None
                or home.realized is None or away.realized is None):
            out["unresolved"] += 1
            continue

        out["n"] += 1
        out["prob_sums"].append(home.prod_probability + away.prod_probability)

        if abs(home.prod_line - away.prod_line) < LINE_EPSILON:
            out["lines_equal"] += 1
        elif abs(home.prod_line + away.prod_line) < LINE_EPSILON:
            out["lines_mirrored"] += 1
        else:
            out["lines_other"] += 1

        home_margin = home.realized                 # PLAYER 1 minus PLAYER 2
        away_margin = away.realized                 # the negation
        literal_home = home_margin > home.prod_line
        literal_away = away_margin > away.prod_line
        complement_away = home_margin < away.prod_line

        if literal_home != literal_away:
            out["literal_partition"] += 1
        elif literal_home:
            out["literal_both_won"] += 1
        else:
            out["literal_both_lost"] += 1

        if literal_away == complement_away:
            out["readings_agree"] += 1

    sums = out["prob_sums"]
    out["prob_sum_mean"] = sum(sums) / len(sums) if sums else None
    out["prob_sum_min"] = min(sums) if sums else None
    out["prob_sum_max"] = max(sums) if sums else None

    n = out["n"]
    if n:
        complementary_probabilities = abs((out["prob_sum_mean"] or 0) - 1.0) < 0.02
        same_line = out["lines_equal"] / n > 0.9
        mirrored_line = out["lines_mirrored"] / n > 0.9
        partitions = out["literal_partition"] / n > 0.99

        if mirrored_line and partitions:
            out["verdict"] = ("literal", "Sides carry mirrored lines and partition "
                                         "cleanly: two sides of one market, literal "
                                         "reading is correct.")
        elif same_line and complementary_probabilities and not partitions:
            out["verdict"] = ("complement", "Same line AND probabilities summing to 1, "
                                            "but the literal reading does not partition. "
                                            "53 is the NO of 52 -- set "
                                            "SPREAD_RESOLUTION = \"complement\".")
        elif partitions:
            out["verdict"] = ("literal", "The literal reading partitions, so it is "
                                         "self-consistent.")
        else:
            out["verdict"] = ("unclear", "Neither reading is clearly supported. "
                                         "Inspect a handful of spread rows directly.")
    else:
        out["verdict"] = ("unclear", "No spread pairs carrying both sides.")
    return out


def build_full_report(pairs, n_bootstrap=2000):
    """Everything the combined HTML report and both console views need.

    Assembled once from one pairing pass so the directional numbers, the
    cross-sectional cells and the per-pair table cannot disagree with each
    other.
    """
    axes = []
    for axis_name, key_function, order_factory in CROSS_AXES:
        axes.append({
            "name": axis_name,
            "order": order_factory(),
            "probability": calibration_cells(pairs, key_function),
            "line": line_cells(pairs, key_function),
        })

    full = calibration_cells(pairs, cell_key)
    return {
        "summary": build_summary(pairs, n_bootstrap),
        "complement": complement_report(pairs),
        "spread": spread_interpretation_report(pairs),
        "both_sides": both_sides_calibration(pairs),
        "axes": axes,
        "full_cell": full,
        "full_cell_order": sorted({k[0] for k in full}, key=buckets.sort_key),
        "daily": daily_breakdown(pairs),
    }


def daily_breakdown(pairs, n_bootstrap=300):
    """Per calendar day: how the comparison looked on that day's games.

    The window only grows, so the useful question as days accumulate is
    whether the answer is stable or drifting. A day that disagrees sharply
    with its neighbours is worth a look before it gets averaged away.
    """
    same, _ = split_by_line(pairs)
    by_day = defaultdict(list)
    for pair in same:
        by_day[pair.day].append(pair)

    out = {}
    for day, subset in sorted(by_day.items()):
        tallied = tally(subset, PROBABILITY)
        out[day] = {
            "pairs": tallied["n"],
            "matches": tallied["n_matches"],
            "win_rate": tallied["candidate_win_rate"],
            "brier": paired_delta_summary(subset, SQUARED, PROBABILITY, n_bootstrap),
        }
    return out


def sorted_pairs_by_disagreement(pairs):
    """Every pair, widest probability disagreement first.

    Sorted on the probability gap because that is what the reader is hunting
    for -- where the two models actually said different things. The line gap
    rides along as its own column, since on a different-line pair the
    probability gap is not what decided it.
    """
    return sorted(pairs, key=lambda p: (-p.disagreement, p.match_code,
                                        p.drive_number, p.market_id))
