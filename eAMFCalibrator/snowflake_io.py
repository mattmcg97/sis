"""All Snowflake access for the calibration suite.

Credentials come from analysis/snowflake_connect.py so there is exactly
one place in the repo that knows how to authenticate. That module lives
in a sibling folder rather than a package, hence the path insert.

INPLAY_FIELD_POSITION_PERIOD has no timestamp column -- preflight
confirmed 7 columns, none of them a clock -- so a play's time is
reconstructed from the stream's own message clock instead (see clock.py).
detect_play_time_column() is kept anyway: if the feed ever gains a real
timestamp, it is preferred over the reconstruction automatically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from snowflake_connect import get_connection  # noqa: E402

from . import config  # noqa: E402
from .markets import MARKET_IDS  # noqa: E402

PLAY_TABLE = "INPLAY_FIELD_POSITION_PERIOD"
SCORE_TABLE = "SCORE_CHANGES"
FINAL_TABLE = "SCORE_ENDGAME"
EVENT_TABLE = "EVENT"

# Preferred clock for a play row, best first. Anything TIMESTAMP-typed is
# accepted as a fallback.
PLAY_TIME_PREFERENCE = ["FILE_TIME", "PUBLISH_TIME", "EVENT_TIME", "FILE_LOADED"]


def fetch_all(cur, sql, params=None):
    cur.execute(sql, params or ())
    cols = [d[0] for d in cur.description]
    return cols, cur.fetchall()


def qualified(table):
    return f"{config.DATABASE}.{config.SCHEMA}.{table}"


def describe_columns(cur, table):
    _, rows = fetch_all(cur, f"""
        SELECT column_name, data_type, is_nullable, ordinal_position
        FROM {config.DATABASE}.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (config.SCHEMA, table))
    return rows


def detect_play_time_column(cur):
    """Which column gives a play its wall-clock time, if any."""
    columns = describe_columns(cur, PLAY_TABLE)
    timestamped = [name for name, dtype, _, _ in columns if dtype.startswith("TIMESTAMP")]
    for preferred in PLAY_TIME_PREFERENCE:
        if preferred in timestamped:
            return preferred, columns
    return (timestamped[0] if timestamped else None), columns


def _in_clause(values):
    return ", ".join(["%s"] * len(values))


def window_predicate(column):
    """The cutoff, as a SQL fragment plus its bind parameters."""
    clauses = [f"{column} >= %s"]
    params = [config.CUTOFF_START]
    if config.CUTOFF_END:
        clauses.append(f"{column} <= %s")
        params.append(config.CUTOFF_END)
    return " AND ".join(clauses), params


def match_universe(cur, stream_table):
    """Settled sport matches carrying quotes inside the window.

    Restricted to matches with a final score, since a calibration
    observation needs a realized outcome to compare against.
    """
    predicate, params = window_predicate("g.PUBLISH_TIME")
    _, rows = fetch_all(cur, f"""
        SELECT DISTINCT e.MATCH_CODE
        FROM {qualified(EVENT_TABLE)} e
        JOIN {qualified(stream_table)} g ON g.MATCH_CODE = e.MATCH_CODE
        JOIN {qualified(FINAL_TABLE)} f ON f.MATCH_CODE = e.MATCH_CODE
        WHERE e.SPORT_CODE = %s
          AND e.INPLAY_EVENT_STATUS = 'SETTLED'
          AND {predicate}
        ORDER BY e.MATCH_CODE
    """, tuple([config.SPORT_CODE] + params))
    return [r[0] for r in rows]


def fetch_plays(cur, match_codes, time_column):
    time_select = time_column if time_column else "NULL"
    _, rows = fetch_all(cur, f"""
        SELECT MATCH_CODE, EVENT_MESSAGE_COUNT, PERIOD_NUMBER, OFFENSIVE_TEAM,
               DOWN_NUMBER, DISTANCE, FIELD_POSITION, {time_select} AS PLAY_TIME
        FROM {qualified(PLAY_TABLE)}
        WHERE MATCH_CODE IN ({_in_clause(match_codes)})
        ORDER BY MATCH_CODE, EVENT_MESSAGE_COUNT
    """, tuple(match_codes))
    return rows


def fetch_scores(cur, match_codes):
    _, rows = fetch_all(cur, f"""
        SELECT MATCH_CODE, EVENT_MESSAGE_COUNT, PERIOD_NUMBER,
               PLAYER_1_SCORE_CHANGE, PLAYER_2_SCORE_CHANGE,
               PLAYER_1_SCORE_CUMULATIVE, PLAYER_2_SCORE_CUMULATIVE
        FROM {qualified(SCORE_TABLE)}
        WHERE MATCH_CODE IN ({_in_clause(match_codes)})
        ORDER BY MATCH_CODE, EVENT_MESSAGE_COUNT
    """, tuple(match_codes))
    return rows


def fetch_final_scores(cur, match_codes):
    _, rows = fetch_all(cur, f"""
        SELECT MATCH_CODE, PLAYER_1_SCORE, PLAYER_2_SCORE
        FROM {qualified(FINAL_TABLE)}
        WHERE MATCH_CODE IN ({_in_clause(match_codes)})
    """, tuple(match_codes))
    return {r[0]: (r[1], r[2]) for r in rows}


def fetch_quotes(cur, stream_table, match_codes):
    """Every usable quote for these matches inside the window.

    Status filters mirror analysis/unconditional_calibration.py so the two
    are measuring the same notion of a live price.
    """
    predicate, params = window_predicate("PUBLISH_TIME")
    extra = " AND PROBABILITY > 0" if config.EXCLUDE_ZERO_PROBABILITY else ""
    _, rows = fetch_all(cur, f"""
        SELECT MATCH_CODE, MARKET_ID, PUBLISH_TIME, PROBABILITY,
               DECIMAL_ODD, MARKET_DESCRIPTION, EVENT_MESSAGE_COUNT
        FROM {qualified(stream_table)}
        WHERE MATCH_CODE IN ({_in_clause(match_codes)})
          AND MARKET_ID IN ({_in_clause(MARKET_IDS)})
          AND STATUS = 'open'
          AND IS_ACTIVE = 'true'
          {extra}
          AND {predicate}
        ORDER BY MATCH_CODE, MARKET_ID, PUBLISH_TIME
    """, tuple(list(match_codes) + list(MARKET_IDS) + params))
    return rows


def fetch_message_times(cur, stream_table, match_codes):
    """(match, message count, time) for rebuilding the play clock.

    Deliberately NOT filtered on STATUS or PROBABILITY the way quotes are:
    this is establishing when a feed message happened, and a suspended or
    zero-priced market timestamps that message just as well as a live one.
    """
    predicate, params = window_predicate("PUBLISH_TIME")
    _, rows = fetch_all(cur, f"""
        SELECT MATCH_CODE, EVENT_MESSAGE_COUNT, MIN(PUBLISH_TIME)
        FROM {qualified(stream_table)}
        WHERE MATCH_CODE IN ({_in_clause(match_codes)})
          AND EVENT_MESSAGE_COUNT IS NOT NULL
          AND {predicate}
        GROUP BY MATCH_CODE, EVENT_MESSAGE_COUNT
    """, tuple(list(match_codes) + params))
    return rows


def stream_window_summary(cur, stream_table):
    """Row and match counts inside the window, for the run header."""
    predicate, params = window_predicate("PUBLISH_TIME")
    _, rows = fetch_all(cur, f"""
        SELECT COUNT(*), COUNT(DISTINCT MATCH_CODE), MIN(PUBLISH_TIME), MAX(PUBLISH_TIME)
        FROM {qualified(stream_table)}
        WHERE {predicate}
    """, tuple(params))
    return rows[0]
