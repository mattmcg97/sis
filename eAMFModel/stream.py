"""Turn a match's feed into quote rows shaped like GAMEPLAI_STREAM's.

Output rows carry exactly the columns the calibrator fetches:

    (MATCH_CODE, MARKET_ID, PUBLISH_TIME, PROBABILITY, DECIMAL_ODD,
     MARKET_DESCRIPTION, EVENT_MESSAGE_COUNT, STATUS, IS_ACTIVE)

so anything that can read a GAMEPLAI stream can read the model -- the
calibrator swaps it in for the candidate with `--candidate v1`.

Lines. By default ("match") each spread and total is priced at the line
PROD was showing at that message, so the two probabilities answer the same
question and settle on the same outcome: the only fair way to compare
probabilities (a different line is a different question). "own" quotes
the model's own fair line instead, for use as a stream in its own right.

The pre-match prior is prod's last pre-match quote on each market (its
first in-play one if it never quoted before the kickoff): the stand-in for
the pre-match model's spread and total, fitted with its probabilities.
"""

from bisect import bisect_right
from collections import defaultdict

from . import feed
from .pricer import (ML_AWAY, ML_HOME, MARKET_IDS, SPREAD_AWAY, SPREAD_HOME,
                     TOTAL_OVER, TOTAL_UNDER, Model)

MATCH = "match"
OWN = "own"

OPEN = "OPEN"
SUSPENDED = "SUSPENDED"

# Elapsed messages are rounded to this before pricing, so a run of
# republished rows shares one book instead of re-solving the same state.
ELAPSED_STEP = 2


def description(market_id, line):
    """MARKET_DESCRIPTION text the calibrator's parse_line reads back."""
    if market_id == ML_HOME:
        return "PLAYER 1 to win"
    if market_id == ML_AWAY:
        return "PLAYER 2 to win"
    if market_id == SPREAD_HOME:
        return f"PLAYER 1 to score over {line:g} points more than PLAYER 2"
    if market_id == SPREAD_AWAY:
        return f"PLAYER 2 to score over {line:g} points more than PLAYER 1"
    if market_id == TOTAL_OVER:
        return f"Total points over {line:g}"
    if market_id == TOTAL_UNDER:
        return f"Total points under {line:g}"
    return None


def _parse_line(text):
    import re
    if not text:
        return None
    cleaned = re.sub(r"player\s*\d+", " ", text, flags=re.IGNORECASE)
    found = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(found.group()) if found else None


class ProdView:
    """Prod's lines and publish times by message, for one match."""

    def __init__(self, quote_rows, first_play_message):
        self.lines = defaultdict(list)       # market -> [(message, line)]
        self.times = []                      # [(message, publish_time)]
        self.prematch = {}                   # market -> (line, probability)
        first_inplay = {}
        rows = sorted(quote_rows, key=lambda r: (r[2] is None, r[2]))
        for (_, market_id, publish_time, probability, _, text,
             message, _, is_active) in rows:
            line = _parse_line(text) if market_id not in (ML_HOME, ML_AWAY) else None
            prob = None if probability is None else float(probability) / 100.0
            pre = message is None or (first_play_message is not None
                                      and message < first_play_message)
            if pre:
                if prob is not None:
                    self.prematch[market_id] = (line, prob)
                continue
            if market_id not in first_inplay and prob is not None:
                first_inplay[market_id] = (line, prob)
            if line is not None:
                self.lines[market_id].append((message, line))
            if publish_time is not None:
                self.times.append((message, publish_time))
        for market_id, value in first_inplay.items():
            self.prematch.setdefault(market_id, value)
        for market_id in self.lines:
            self.lines[market_id].sort(key=lambda x: x[0])
        self.times.sort(key=lambda x: x[0])
        self._line_keys = {m: [x[0] for x in v] for m, v in self.lines.items()}
        self._time_keys = [x[0] for x in self.times]

    def line_at(self, market_id, message):
        keys = self._line_keys.get(market_id)
        if not keys:
            return None
        i = bisect_right(keys, message)
        return self.lines[market_id][max(0, i - 1)][1]

    def time_at(self, message):
        if not self._time_keys:
            return None
        i = bisect_right(self._time_keys, message)
        return self.times[max(0, i - 1)][1]

    def messages(self):
        return sorted({m for m, _ in self.times})


def prior_for(model, prod):
    spread = prod.prematch.get(SPREAD_HOME)
    total = prod.prematch.get(TOTAL_OVER)
    ml = prod.prematch.get(ML_HOME)
    if not spread or not total or spread[0] is None or total[0] is None:
        return None
    return model.fit_prior(spread[0], total[0], ml_home=ml[1] if ml else None,
                           spread_home=spread[1], over=total[1])


def quote_rows(model, match_code, plays, scores, prod_quote_rows, line_mode=MATCH):
    """Every model quote for one match, GAMEPLAI-shaped.

    `plays` and `scores` are feed.Play / feed.Score; `prod_quote_rows` are
    prod's raw rows for this match. A market prod never lined gets no
    spread or total in "match" mode.
    """
    first_play = min((p.message for p in plays), default=None)
    prod = ProdView(prod_quote_rows, first_play)
    prior = prior_for(model, prod)
    if prior is None:
        return []

    ticks = feed.replay(plays, scores)
    tick_messages = [t.message for t in ticks]
    messages = sorted(set(tick_messages) | set(m for m in prod.messages()
                                               if first_play is None or m >= first_play))
    books = {}
    out = []
    for message in messages:
        i = bisect_right(tick_messages, message) - 1
        if i < 0:
            continue
        tick = ticks[i]
        live = tick.verdict == feed.LIVE and tick.state is not None
        book = None
        if tick.state is not None:
            state = tick.state
            elapsed = state.elapsed_in_period + (message - tick.message)
            elapsed = ELAPSED_STEP * round(elapsed / ELAPSED_STEP)
            age = state.drive_age
            if age is not None:
                age = ELAPSED_STEP * round((age + message - tick.message) / ELAPSED_STEP)
            key = (state.period, elapsed, state.home_score, state.away_score, state.offense,
                   state.down, state.field_position, state.distance, age, state.opening_receiver)
            if key not in books:
                books[key] = model.book(prior, feed.with_state(
                    tick, elapsed_in_period=elapsed, drive_age=age).state)
            book = books[key]
        publish_time = prod.time_at(message)
        for market_id in MARKET_IDS:
            if book is None:
                continue
            if market_id in (ML_HOME, ML_AWAY):
                line = None
            elif line_mode == MATCH:
                line = prod.line_at(market_id, message)
                if line is None:
                    continue
            else:
                line = book.fair_line(market_id)
            p = book.prob(market_id, line)
            if p is None:
                continue
            p = min(0.9999, max(0.0001, p))
            out.append((match_code, market_id, publish_time, round(100.0 * p, 2),
                        round(1.0 / p, 4), description(market_id, line), message,
                        OPEN if live else SUSPENDED, "true" if live else "false"))
    return out


def plays_from_rows(play_rows):
    """Calibrator/Snowflake play tuples -> feed.Play.

    (MATCH_CODE, EVENT_MESSAGE_COUNT, PERIOD_NUMBER, OFFENSIVE_TEAM,
     DOWN_NUMBER, DISTANCE, FIELD_POSITION, ...)
    """
    def num(v):
        return None if v is None else int(v)
    return [feed.Play(num(r[1]), num(r[2]), feed.side(r[3]), num(r[4]), num(r[5]), num(r[6]))
            for r in play_rows if r[1] is not None]


def scores_from_rows(score_rows):
    """(MATCH_CODE, MSG, PERIOD, P1_CHANGE, P2_CHANGE, P1_CUM, P2_CUM) -> feed.Score."""
    def num(v):
        return None if v is None else int(v)
    return [feed.Score(num(r[1]), num(r[2]), num(r[5]), num(r[6]))
            for r in score_rows if r[1] is not None]


def quotes_for_matches(model, match_codes, play_rows, score_rows, prod_quote_rows,
                       line_mode=MATCH):
    """quote_rows across many matches, from the calibrator's raw fetches."""
    by = defaultdict(lambda: ([], [], []))
    for row in play_rows:
        by[row[0]][0].append(row)
    for row in score_rows:
        by[row[0]][1].append(row)
    for row in prod_quote_rows:
        by[row[0]][2].append(row)
    out = []
    for match_code in match_codes:
        plays, scores, quotes = by.get(match_code, ([], [], []))
        if not plays or not quotes:
            continue
        out.extend(quote_rows(model, match_code, plays_from_rows(plays),
                              scores_from_rows(scores), quotes, line_mode))
    return out


_MODELS = {}


def model_for(version_name):
    from .params import version
    key = version_name.lower()
    if key not in _MODELS:
        _MODELS[key] = Model(version(key))
    return _MODELS[key]
