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


def head_to_head(observations, n_bootstrap=1000, seed=0):
    """Paired Brier, prod against candidate, on the same closing prices.

    Paired on (match, selection) so both streams are being graded on the
    same question about the same match, and clustered on matches because
    six selections inside one match are not six independent draws.
    """
    keyed = collections.defaultdict(dict)
    for o in observations:
        keyed[(o.match_code, o.market_id)][o.stream] = o

    per_match = collections.defaultdict(list)
    for (match_code, _), sides in keyed.items():
        prod, candidate = sides.get(PROD), sides.get(CANDIDATE)
        if prod is None or candidate is None:
            continue
        # Positive favours the candidate, matching the rest of the suite.
        per_match[match_code].append(prod.brier - candidate.brier)

    deltas = [sum(v) / len(v) for v in per_match.values()]
    out = {"pairs": sum(len(v) for v in per_match.values()),
           "matches": len(deltas), "mean": None, "ci_low": None,
           "ci_high": None, "p_value": None,
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


def report(observations, n_bootstrap=1000, n_bins=None):
    """The whole pre-match reading, in the shape the report renders."""
    if not observations:
        return None
    out = {"n": len(observations),
           "matches": len({o.match_code for o in observations}),
           "head_to_head": head_to_head(observations, n_bootstrap),
           "streams": {}}
    for stream in (PROD, CANDIDATE):
        mine = [o for o in observations if o.stream == stream]
        if not mine:
            continue
        out["streams"][stream] = {
            "overall": block(mine, n_bins),
            "by_market": by(mine, lambda o: markets.market_group(o.market_id),
                            n_bins),
            "by_selection": by(mine, selection_label, n_bins),
        }
    return out


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
