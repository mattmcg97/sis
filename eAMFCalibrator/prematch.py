"""Were the PRE-MATCH prices calibrated against what actually happened?

Everything else in this suite reads prices taken DURING a match, at a
drive start or between two plays. This reads the last price published
BEFORE the match got under way -- the closing line -- and asks the
oldest question there is: when the model said 62%, did it happen 62% of
the time?

Pre-match is defined from the play feed rather than from a schedule.
The play rows begin when the match does, so a quote is pre-match when it
carries no message count at all or a message count below the first play
row's. That needs no column the rest of the suite does not already read,
and it cannot drift out of step with a kickoff time nobody publishes.

One observation per (match, selection, stream): the CLOSING quote, the
last one before kickoff, which is the most informed price the model ever
published without seeing a snap. Taking the first instead would grade
the model on how it opened, which is a different and much easier
question.

A selection the final score cannot settle -- a push, or a match with no
final -- is counted and dropped. It has no realized 0/1 to compare a
probability against, exactly as in the calibration runs.
"""

import collections
import random
from dataclasses import dataclass
from typing import Optional

from . import config, directional, markets, metrics

PROD, CANDIDATE = directional.PROD, directional.CANDIDATE


@dataclass(frozen=True)
class Observation:
    """One closing price, and whether the selection came in."""
    match_code: str
    stream: str
    market_id: int
    probability: float
    line: Optional[float]
    outcome: bool
    publish_time: object
    live: bool
    n_quotes: int          # how many pre-match quotes this one was last of

    @property
    def error(self):
        return abs(self.probability - (1.0 if self.outcome else 0.0))

    @property
    def brier(self):
        return (self.probability - (1.0 if self.outcome else 0.0)) ** 2


def is_prematch(message, first_play_message):
    """Was this quote published before the match got under way?

    A quote with no message count at all has not been tied to a play, so
    it cannot be in-play. One with a message count below the first play
    row's was published before the feed had a play to point at.
    """
    if message is None:
        return True
    if first_play_message is None:
        return False
    return message < first_play_message


def closing_quotes(quote_rows, first_play_message):
    """market_id -> the last pre-match quote for it, and how many there were.

    Last by publish time: the closing line. The count travels with it so
    a market quoted once months out is not read as the same kind of thing
    as one quoted two hundred times up to kickoff.
    """
    grouped = collections.defaultdict(list)
    for row in quote_rows:
        (_, market_id, publish_time, probability, _decimal, description,
         message, status, is_active) = row
        if probability is None:
            continue
        if not is_prematch(message, first_play_message):
            continue
        grouped[market_id].append((publish_time, probability, description,
                                   status, is_active))
    out = {}
    for market_id, rows in grouped.items():
        rows.sort(key=lambda r: (r[0] is None, r[0]))
        publish_time, probability, description, status, is_active = rows[-1]
        out[market_id] = {
            "publish_time": publish_time,
            "probability": float(probability) / 100.0,
            "line": (markets.parse_line(description)
                     if markets.needs_line(market_id) else None),
            "live": directional.is_live(is_active),
            "n_quotes": len(rows),
        }
    return out


def observations_for_match(match_code, quotes_by_stream, first_play_message,
                           final, stats=None):
    """Every closing price for one match, resolved against the final score."""
    stats = stats if stats is not None else collections.defaultdict(int)
    if final is None or final[0] is None or final[1] is None:
        stats["prematch_no_final"] += 1
        return []
    final_p1, final_p2 = final

    out = []
    for stream, rows in quotes_by_stream.items():
        closing = closing_quotes(rows, first_play_message)
        if not closing:
            stats["prematch_no_quotes"] += 1
            continue
        for market_id, quote in closing.items():
            outcome = markets.resolve(market_id, quote["line"],
                                      final_p1, final_p2)
            if outcome is None:
                stats["prematch_push"] += 1
                continue
            if config.REQUIRE_LIVE_QUOTE and not quote["live"]:
                stats["prematch_not_live"] += 1
                continue
            stats["prematch_observations"] += 1
            out.append(Observation(
                match_code=match_code, stream=stream, market_id=market_id,
                probability=quote["probability"], line=quote["line"],
                outcome=outcome, publish_time=quote["publish_time"],
                live=quote["live"], n_quotes=quote["n_quotes"]))
    return out


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Pair:
    """One closing price from each stream, on the same question.

    Same question means the same match, the same selection AND the same
    line. Where the lines differ the two streams resolved against
    DIFFERENT outcomes, so neither the realized rate nor the Brier
    difference means anything -- the same rule the in-play comparison
    runs on, and the reason this is a pairing step rather than two
    independent tallies.
    """
    match_code: str
    market_id: int
    line: Optional[float]
    outcome: bool
    prod: Observation
    candidate: Observation


def build_pairs(observations, stats=None):
    """Every (match, selection) both streams closed on the same line."""
    stats = stats if stats is not None else collections.defaultdict(int)
    keyed = collections.defaultdict(dict)
    for o in observations:
        keyed[(o.match_code, o.market_id)][o.stream] = o

    out = []
    for (match_code, market_id), sides in keyed.items():
        prod, candidate = sides.get(PROD), sides.get(CANDIDATE)
        if prod is None or candidate is None:
            stats["prematch_one_stream_only"] += 1
            continue
        if markets.needs_line(market_id) and prod.line != candidate.line:
            stats["prematch_line_differs"] += 1
            continue
        stats["prematch_paired"] += 1
        out.append(Pair(match_code=match_code, market_id=market_id,
                        line=prod.line, outcome=prod.outcome,
                        prod=prod, candidate=candidate))
    out.sort(key=lambda p: (p.match_code, p.market_id))
    return out


def selection_key(pair):
    return (markets.market_group(pair.market_id),
            markets.selection_label(pair.market_id))


def line_key(pair):
    return selection_key(pair) + (pair.line,)


def cell(pairs, n_bootstrap=1000, seed=0):
    """Both streams against the SAME realized rate.

    The shape the cross-section uses, and for the same reason: one
    realized number with each stream's prediction beside it is the only
    way to see which one is off and in which direction.
    """
    out = {"n": len(pairs), "matches": len({p.match_code for p in pairs}),
           "realized": None, "prod_predicted": None, "candidate_predicted": None,
           "prod_gap": None, "candidate_gap": None, "prod_brier": None,
           "candidate_brier": None, "brier_delta": None, "ci_low": None,
           "ci_high": None, "p_value": None, "closer": None}
    if not pairs:
        return out
    n = len(pairs)
    out["realized"] = sum(1 for p in pairs if p.outcome) / n
    out["prod_predicted"] = sum(p.prod.probability for p in pairs) / n
    out["candidate_predicted"] = sum(p.candidate.probability for p in pairs) / n
    out["prod_gap"] = out["realized"] - out["prod_predicted"]
    out["candidate_gap"] = out["realized"] - out["candidate_predicted"]
    out["prod_brier"] = sum(p.prod.brier for p in pairs) / n
    out["candidate_brier"] = sum(p.candidate.brier for p in pairs) / n
    out["closer"] = ("cand" if abs(out["candidate_gap"]) < abs(out["prod_gap"])
                     else "prod")

    # Positive favours the candidate, clustered on matches.
    per_match = collections.defaultdict(list)
    for p in pairs:
        per_match[p.match_code].append(p.prod.brier - p.candidate.brier)
    deltas = [sum(v) / len(v) for v in per_match.values()]
    out["brier_delta"] = sum(deltas) / len(deltas)
    out["p_value"] = metrics.sign_test(sum(1 for d in deltas if d > 0),
                                       sum(1 for d in deltas if d < 0))
    if len(deltas) >= 2 and n_bootstrap > 0:
        rnd = random.Random(seed)
        means = []
        for _ in range(n_bootstrap):
            sample = [deltas[rnd.randrange(len(deltas))] for _ in deltas]
            means.append(sum(sample) / len(sample))
        means.sort()
        out["ci_low"] = means[int(0.025 * len(means))]
        out["ci_high"] = means[min(int(0.975 * len(means)), len(means) - 1)]
    return out


def cells(pairs, key, n_bootstrap=1000, min_n=1):
    grouped = collections.defaultdict(list)
    for p in pairs:
        grouped[key(p)].append(p)
    return {k: cell(v, n_bootstrap) for k, v in grouped.items()
            if len(v) >= min_n}


# How far from even money the closing prices actually sit. The whole
# pre-match reading turns on this: a book that never leaves 50/50 has no
# view to be calibrated, and its Brier will sit at 0.25 whatever else is
# true. Bands rather than a standard deviation, because the question is
# whether ANY price is ever confident, not what the average one looks
# like.
PRICE_BANDS = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10),
               (0.10, 0.20), (0.20, 0.50)]


def price_spread(observations):
    """Distribution of |closing price - 0.5|, and of how many quotes it
    was the last of."""
    out = {"n": len(observations), "bands": [], "quotes": collections.Counter(),
           "max_distance": None, "mean_distance": None}
    if not observations:
        return out
    distances = [abs(o.probability - 0.5) for o in observations]
    out["max_distance"] = max(distances)
    out["mean_distance"] = sum(distances) / len(distances)
    for low, high in PRICE_BANDS:
        n = sum(1 for d in distances if low <= d < high)
        out["bands"].append({"low": low, "high": high, "n": n,
                             "share": n / len(distances)})
    for o in observations:
        out["quotes"][min(o.n_quotes, 10)] += 1
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def block(observations, n_bins=None):
    """Calibration for one bag of closing prices."""
    n_bins = n_bins or config.N_RELIABILITY_BINS
    pairs = [(o.probability, o.outcome) for o in observations]
    out = dict(metrics.summarize(pairs, n_bins))
    out["matches"] = len({o.match_code for o in observations})
    out["bins"] = metrics.reliability_bins(pairs, n_bins) if pairs else []
    return out


def by(observations, key, n_bins=None):
    grouped = collections.defaultdict(list)
    for o in observations:
        grouped[key(o)].append(o)
    return {k: block(v, n_bins) for k, v in grouped.items()}


def selection_label(o):
    return (markets.market_group(o.market_id), markets.selection_label(o.market_id))


def head_to_head(pairs, n_bootstrap=1000, seed=0):
    """Paired Brier, prod against candidate, on the same closing prices.

    Takes PAIRS rather than observations, which is the fix this needed: a
    pair only exists where both streams closed on the same line, so the
    two Briers are squared against the same 0/1. Differencing them across
    a line change would have compared two different questions.
    """
    per_match = collections.defaultdict(list)
    for p in pairs:
        per_match[p.match_code].append(p.prod.brier - p.candidate.brier)
    deltas = [sum(v) / len(v) for v in per_match.values()]
    out = {"pairs": len(pairs), "matches": len(deltas), "mean": None,
           "ci_low": None, "ci_high": None, "p_value": None,
           "matches_favouring_candidate": sum(1 for d in deltas if d > 0),
           "matches_favouring_prod": sum(1 for d in deltas if d < 0)}
    if not deltas:
        return out
    out["mean"] = sum(deltas) / len(deltas)
    out["p_value"] = metrics.sign_test(out["matches_favouring_candidate"],
                                       out["matches_favouring_prod"])
    if len(deltas) < 2 or n_bootstrap <= 0:
        return out
    rnd = random.Random(seed)
    means = []
    for _ in range(n_bootstrap):
        sample = [deltas[rnd.randrange(len(deltas))] for _ in deltas]
        means.append(sum(sample) / len(sample))
    means.sort()
    out["ci_low"] = means[int(0.025 * len(means))]
    out["ci_high"] = means[min(int(0.975 * len(means)), len(means) - 1)]
    return out


def report(observations, n_bootstrap=1000, n_bins=None, stats=None,
           min_line_n=10):
    """The whole pre-match reading.

    Built on PAIRS, per SELECTION, against one shared realized rate.
    Pooling the two sides of a market would force the realized rate and
    both predictions to exactly 0.500 whatever the model does -- the
    sides are complements, so the mean of a price and one minus it is
    0.5 by arithmetic -- which is the same trap the cross-section avoids
    by reading one selection per market. The Brier survives that pooling;
    the gap does not.
    """
    if not observations:
        return None
    stats = stats if stats is not None else collections.defaultdict(int)
    pairs = build_pairs(observations, stats)
    lined = [p for p in pairs if markets.needs_line(p.market_id)]
    return {
        "n": len(observations),
        "matches": len({o.match_code for o in observations}),
        "pairs": len(pairs),
        "line_differs": stats.get("prematch_line_differs", 0),
        "one_stream_only": stats.get("prematch_one_stream_only", 0),
        "head_to_head": head_to_head(pairs, n_bootstrap),
        "by_selection": cells(pairs, selection_key, n_bootstrap),
        "by_line": cells(lined, line_key, n_bootstrap, min_n=min_line_n),
        "spread": price_spread(observations),
        "spread_by_market": {
            group: price_spread([o for o in observations
                                 if markets.market_group(o.market_id) == group])
            for group in sorted({markets.market_group(o.market_id)
                                 for o in observations})},
    }


class Sink:
    """Collects the pre-match view off another pass's fetched rows.

    directional.build_pairs already holds every quote for the chunk,
    including the pre-match ones it then drops, plus the plays that say
    where the match started and the final score to resolve against. So
    this rides along rather than paying for a second set of queries.
    """

    def __init__(self):
        self.observations = []

    def add(self, match_code, plays, final, quotes_by_stream, stats=None):
        first = min((p.event_message_count for p in plays), default=None)
        self.observations.extend(observations_for_match(
            match_code, quotes_by_stream, first, final, stats))

    def summary(self, n_bootstrap=300):
        return report(self.observations, n_bootstrap=n_bootstrap)

    def rows(self):
        return [row(o) for o in self.observations]


FIELDS = ["match_code", "stream", "market_id", "market", "selection",
          "publish_time", "probability", "line", "outcome", "error", "brier",
          "live", "n_quotes"]


def row(o):
    return {
        "match_code": o.match_code, "stream": o.stream,
        "market_id": o.market_id,
        "market": markets.market_group(o.market_id),
        "selection": markets.selection_label(o.market_id),
        "publish_time": "" if o.publish_time is None else str(o.publish_time),
        "probability": round(o.probability, 6),
        "line": "" if o.line is None else o.line,
        "outcome": int(o.outcome), "error": round(o.error, 6),
        "brier": round(o.brier, 6), "live": int(o.live),
        "n_quotes": o.n_quotes,
    }
