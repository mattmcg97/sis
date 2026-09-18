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

from . import buckets, config, markets, metrics, snowflake_io
from .drives import PlayRow, ScoreRow, build_snapshots

CANDIDATE = "candidate"
PROD = "prod"
TIE = "tie"


@dataclass(frozen=True)
class PairedObservation:
    match_code: str
    drive_number: int
    period_number: Optional[int]
    score_diff: int
    offensive_team: Optional[str]
    market_id: int
    line: Optional[float]
    message_count: int
    message_gap: int
    prod_probability: float
    candidate_probability: float
    outcome: bool

    @property
    def target(self):
        return 1.0 if self.outcome else 0.0

    @property
    def prod_error(self):
        return abs(self.prod_probability - self.target)

    @property
    def candidate_error(self):
        return abs(self.candidate_probability - self.target)

    @property
    def disagreement(self):
        return abs(self.prod_probability - self.candidate_probability)

    @property
    def winner(self):
        if self.candidate_error < self.prod_error:
            return CANDIDATE
        if self.prod_error < self.candidate_error:
            return PROD
        return TIE


def index_by_message(quote_rows):
    """(match, market) -> {message_count: (probability, description)}."""
    out = defaultdict(dict)
    for match_code, market_id, publish_time, probability, decimal_odd, description, message in quote_rows:
        if message is None or probability is None:
            continue
        # A message carries one row per market; keep the first seen.
        out[(match_code, market_id)].setdefault(message, (float(probability), description))
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


def build_pairs(cur, match_codes, time_column, stats):
    """Every paired observation for one chunk of matches."""
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
        if not plays:
            stats["matches_without_plays"] += 1
            continue

        final = finals.get(match_code)
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

                prod_probability, prod_description = prod_index[(match_code, market_id)][message]
                candidate_probability, _ = candidate_index[(match_code, market_id)][message]

                line = markets.parse_line(prod_description) if markets.needs_line(market_id) else None
                if markets.needs_line(market_id) and line is None:
                    stats["unparsed_line"] += 1
                    continue

                outcome = markets.resolve(market_id, line, final_p1, final_p2)
                if outcome is None:
                    stats["pushes_or_unresolved"] += 1
                    continue

                paired_any = True
                pairs.append(PairedObservation(
                    match_code=match_code,
                    drive_number=snap.drive_number,
                    period_number=snap.period_number,
                    score_diff=snap.score_diff,
                    offensive_team=snap.offensive_team,
                    market_id=market_id,
                    line=line,
                    message_count=message,
                    message_gap=gap,
                    prod_probability=prod_probability / 100.0,
                    candidate_probability=candidate_probability / 100.0,
                    outcome=outcome,
                ))
            if paired_any:
                stats["snapshots_paired"] += 1

    return pairs


def run(verbose=True):
    """Build every paired observation across the window."""
    stats = defaultdict(int)
    pairs = []
    header = {}

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
                pairs.extend(build_pairs(cur, batch, time_column, stats))
    finally:
        conn.close()

    return pairs, stats, header


def tally(pairs):
    """Win/loss/tie counts and error means for a set of pairs."""
    result = {"n": len(pairs), CANDIDATE: 0, PROD: 0, TIE: 0,
              "prod_error_sum": 0.0, "candidate_error_sum": 0.0,
              "prod_brier_sum": 0.0, "candidate_brier_sum": 0.0,
              "matches": set()}
    for pair in pairs:
        result[pair.winner] += 1
        result["prod_error_sum"] += pair.prod_error
        result["candidate_error_sum"] += pair.candidate_error
        result["prod_brier_sum"] += pair.prod_error ** 2
        result["candidate_brier_sum"] += pair.candidate_error ** 2
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


def match_level_votes(pairs):
    """One vote per match, by which model won more of that match's pairs.

    This is the clustering-robust reading: matches are independent, the
    pairs inside them are not.
    """
    per_match = defaultdict(lambda: {CANDIDATE: 0, PROD: 0, TIE: 0})
    for pair in pairs:
        per_match[pair.match_code][pair.winner] += 1

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
# The win rate above answers "which model was closer more often", which is
# what was asked for and is worth seeing. It is also a weak test, and it is
# worth knowing why before reading it.
#
# If both models are unbiased around the same underlying truth and differ
# only in how noisy they are, then which one lands closer to the realized 0
# or 1 is decided by which noise draw happened to point at the outcome --
# and that is a coin flip regardless of variance. Simulated with the
# candidate at half prod's noise, the candidate's win rate is ~50% while its
# Brier score is clearly better. The win rate discards magnitude, so it
# cannot see an improvement that consists of being less wrong.
#
# The paired difference in loss does see it. Computed per match and then
# averaged across matches, it also respects the clustering: matches are
# independent, pairs inside them are not.

SQUARED = "brier"
ABSOLUTE = "mae"


def _loss(pair, kind):
    error = pair.candidate_error, pair.prod_error
    if kind == SQUARED:
        return pair.prod_error ** 2, pair.candidate_error ** 2
    return error[1], error[0]


def per_match_deltas(pairs, kind=SQUARED):
    """match -> mean(prod loss) - mean(candidate loss).

    Positive means the candidate carried less loss in that match.
    """
    totals = defaultdict(lambda: [0.0, 0.0, 0])
    for pair in pairs:
        prod_loss, candidate_loss = _loss(pair, kind)
        bucket = totals[pair.match_code]
        bucket[0] += prod_loss
        bucket[1] += candidate_loss
        bucket[2] += 1
    return {match: (prod_sum - candidate_sum) / n
            for match, (prod_sum, candidate_sum, n) in totals.items()}


def paired_delta_summary(pairs, kind=SQUARED, n_bootstrap=2000, seed=0):
    """Match-clustered test on the paired loss difference.

    Bootstrap resamples MATCHES, not rows, so the confidence interval
    carries the same clustering assumption as the point estimate.
    """
    deltas = list(per_match_deltas(pairs, kind).values())
    m = len(deltas)
    out = {"kind": kind, "n_matches": m, "mean": None, "se": None,
           "t": None, "p_value": None, "ci_low": None, "ci_high": None,
           "matches_favouring_candidate": sum(1 for d in deltas if d > 0),
           "matches_favouring_prod": sum(1 for d in deltas if d < 0)}
    if m == 0:
        return out

    mean = sum(deltas) / m
    out["mean"] = mean
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
    out["sign_p_value"] = metrics.sign_test(out["matches_favouring_candidate"],
                                            out["matches_favouring_prod"])
    return out


def group_by(pairs, key_function):
    grouped = defaultdict(list)
    for pair in pairs:
        grouped[key_function(pair)].append(pair)
    return {key: tally(group) for key, group in grouped.items()}


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


def cell_key(pair):
    return (
        buckets.score_diff_bucket(pair.score_diff),
        buckets.time_bucket(pair.period_number, pair.drive_number),
        buckets.possession_bucket(pair.offensive_team),
    )
