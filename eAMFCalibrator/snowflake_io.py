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

# A stream named MODEL:<version> is not a table: it is the eAMFModel pricer,
# run over prod's quotes and the play feed for the same matches. Everything
# that asks a stream WHICH matches or WHEN reads prod's table in its place.
MODEL_PREFIX = "MODEL:"

# Preferred clock for a play row, best first. Anything TIMESTAMP-typed is
# accepted as a fallback.
PLAY_TIME_PREFERENCE = ["FILE_TIME", "PUBLISH_TIME", "EVENT_TIME", "FILE_LOADED"]


def is_model(stream_table):
    return isinstance(stream_table, str) and stream_table.upper().startswith(MODEL_PREFIX)


def stream_name(value):
    """CLI value -> STREAMS entry: 'v1' means the model, a table stays a table."""
    if is_model(value):
        return MODEL_PREFIX + value.split(":", 1)[1].lower()
    if value.lower().startswith("v") and value[1:].isdigit():
        return MODEL_PREFIX + value.lower()
    return value


def source_table(stream_table):
    """The real table behind a stream: prod's, for a model stream."""
    return config.STREAMS["prod"] if is_model(stream_table) else stream_table


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
    stream_table = source_table(stream_table)
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


def rows_per_message(cur, stream_table):
    """How many rows a single (match, market, message) really carries.

    The pipeline long assumed one. For the moneyline that holds; for the
    spread and total, whose lines move, a message can carry the settlement
    of the old line beside the open quote for the new one. This says which,
    and how often a message offers both a live row and a dead one -- the
    case where picking the wrong row silently costs a pair.
    """
    stream_table = source_table(stream_table)
    predicate, params = window_predicate("PUBLISH_TIME")
    _, rows = fetch_all(cur, f"""
        WITH per_message AS (
            SELECT MARKET_ID, MATCH_CODE, EVENT_MESSAGE_COUNT,
                   COUNT(*) AS N_ROWS,
                   SUM(CASE WHEN IS_ACTIVE = %s THEN 1 ELSE 0 END) AS N_LIVE
            FROM {qualified(stream_table)}
            WHERE MARKET_ID IN ({_in_clause(MARKET_IDS)})
              AND EVENT_MESSAGE_COUNT IS NOT NULL
              AND {predicate}
            GROUP BY MARKET_ID, MATCH_CODE, EVENT_MESSAGE_COUNT
        )
        SELECT MARKET_ID,
               COUNT(*) AS MESSAGES,
               AVG(N_ROWS) AS MEAN_ROWS,
               MAX(N_ROWS) AS MAX_ROWS,
               SUM(CASE WHEN N_ROWS > 1 THEN 1 ELSE 0 END) AS MULTI,
               SUM(CASE WHEN N_ROWS > 1 AND N_LIVE > 0 AND N_LIVE < N_ROWS
                        THEN 1 ELSE 0 END) AS MIXED
        FROM per_message
        GROUP BY MARKET_ID
        ORDER BY MARKET_ID
    """, tuple([config.LIVE_IS_ACTIVE]
               + list(MARKET_IDS) + params))
    return rows


def team_vocabulary(cur, stream_table):
    """What the three team-naming columns actually contain.

    EVENT names the two sides one way, the play feed names them another,
    and the market descriptions name them a third. Whether those
    vocabularies meet decides whether PLAYER_1 = Home Team is a fact we can
    read per match or an assumption we are carrying.
    """
    stream_table = source_table(stream_table)
    _, rows = fetch_all(cur, f"""
        SELECT 'EVENT.PLAYER_1_TEAM' AS SRC, PLAYER_1_TEAM AS VALUE,
               COUNT(*) AS N, COUNT(DISTINCT MATCH_CODE) AS MATCHES
        FROM {qualified(EVENT_TABLE)}
        WHERE SPORT_CODE = %s AND INPLAY_EVENT_STATUS = 'SETTLED'
        GROUP BY 1, 2
        UNION ALL
        SELECT 'EVENT.PLAYER_2_TEAM', PLAYER_2_TEAM,
               COUNT(*), COUNT(DISTINCT MATCH_CODE)
        FROM {qualified(EVENT_TABLE)}
        WHERE SPORT_CODE = %s AND INPLAY_EVENT_STATUS = 'SETTLED'
        GROUP BY 1, 2
        UNION ALL
        SELECT 'PLAY.OFFENSIVE_TEAM', fp.OFFENSIVE_TEAM,
               COUNT(*), COUNT(DISTINCT fp.MATCH_CODE)
        FROM {qualified(PLAY_TABLE)} fp
        JOIN {qualified(EVENT_TABLE)} e ON e.MATCH_CODE = fp.MATCH_CODE
        WHERE e.SPORT_CODE = %s
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
    """, (config.SPORT_CODE, config.SPORT_CODE, config.SPORT_CODE))
    return rows


def market_descriptions(cur, stream_table):
    """One sample description per market ID, with how many forms exist.

    The mapping PLAYER_1 = Home Team was originally taken from this text.
    Printing it puts the evidence in front of the reader rather than in a
    comment in another file.
    """
    stream_table = source_table(stream_table)
    predicate, params = window_predicate("PUBLISH_TIME")
    _, rows = fetch_all(cur, f"""
        SELECT MARKET_ID, COUNT(DISTINCT MARKET_DESCRIPTION) AS FORMS,
               MIN(MARKET_DESCRIPTION) AS SAMPLE, COUNT(*) AS N
        FROM {qualified(stream_table)}
        WHERE MARKET_ID IN ({_in_clause(MARKET_IDS)})
          AND {predicate}
        GROUP BY MARKET_ID
        ORDER BY MARKET_ID
    """, tuple(list(MARKET_IDS) + params))
    return rows


def team_join_test(cur, stream_table):
    """Do a match's OFFENSIVE_TEAM values match its EVENT team names?

    If they do, EVENT gives a per-match mapping and nothing has to be
    assumed. If they do not, the two feeds are naming the same two sides in
    different vocabularies and the mapping stays positional.
    """
    stream_table = source_table(stream_table)
    _, rows = fetch_all(cur, f"""
        WITH per_match AS (
            SELECT fp.MATCH_CODE,
                   ARRAY_TO_STRING(ARRAY_SORT(ARRAY_AGG(
                       DISTINCT fp.OFFENSIVE_TEAM)), ' | ') AS PLAY_TEAMS,
                   ANY_VALUE(e.PLAYER_1_TEAM) AS P1_TEAM,
                   ANY_VALUE(e.PLAYER_2_TEAM) AS P2_TEAM
            FROM {qualified(PLAY_TABLE)} fp
            JOIN {qualified(EVENT_TABLE)} e ON e.MATCH_CODE = fp.MATCH_CODE
            WHERE e.SPORT_CODE = %s AND e.INPLAY_EVENT_STATUS = 'SETTLED'
            GROUP BY fp.MATCH_CODE
        )
        SELECT PLAY_TEAMS, P1_TEAM, P2_TEAM, COUNT(*) AS MATCHES
        FROM per_match
        GROUP BY PLAY_TEAMS, P1_TEAM, P2_TEAM
        ORDER BY MATCHES DESC
        LIMIT 20
    """, (config.SPORT_CODE,))
    return rows


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
    """Every quote for these matches inside the window, live or not.

    Liveness used to be a WHERE clause, which meant a market that was
    inactive at a snapshot produced no row and the snapshot simply
    vanished -- indistinguishable from a feed gap. Going inactive is not
    missing at random: it clusters on scoring plays and reviews, which are
    exactly the states where the two models differ most. Worse, if the two
    streams go inactive at different moments then pairing on "both had a
    live quote" silently drops the disagreements.

    So the filter moved out of SQL and into the caller, which keeps the
    same rows in the metrics (config.REQUIRE_LIVE_QUOTE, on by default) and
    can now count and show what it is dropping.
    """
    if is_model(stream_table):
        return _model_quotes(cur, stream_table, match_codes)
    predicate, params = window_predicate("PUBLISH_TIME")
    extra = " AND PROBABILITY > 0" if config.EXCLUDE_ZERO_PROBABILITY else ""
    _, rows = fetch_all(cur, f"""
        SELECT MATCH_CODE, MARKET_ID, PUBLISH_TIME, PROBABILITY,
               DECIMAL_ODD, MARKET_DESCRIPTION, EVENT_MESSAGE_COUNT,
               STATUS, IS_ACTIVE
        FROM {qualified(stream_table)}
        WHERE MATCH_CODE IN ({_in_clause(match_codes)})
          AND MARKET_ID IN ({_in_clause(MARKET_IDS)})
          {extra}
          AND {predicate}
        ORDER BY MATCH_CODE, MARKET_ID, PUBLISH_TIME
    """, tuple(list(match_codes) + list(MARKET_IDS) + params))
    return rows


def _model_quotes(cur, stream_table, match_codes):
    """Price the matches with an eAMFModel version, GAMEPLAI-shaped.

    Fetches what the model reads -- prod's quotes (for its pre-match prior
    and its lines), the plays and the scores -- and returns rows exactly as
    a stream table would.
    """
    from eAMFModel import stream as model_stream
    version = stream_table.split(":", 1)[1] or "v1"
    if version.lower() == "v3":
        return _v3_quotes(cur, match_codes)
    prod = fetch_quotes(cur, config.STREAMS["prod"], match_codes)
    plays = fetch_plays(cur, match_codes, None)
    scores = fetch_scores(cur, match_codes)
    return model_stream.quotes_for_matches(model_stream.model_for(version), match_codes,
                                           plays, scores, prod)


def _v3_quotes(cur, match_codes):
    """eAMFModel v3, GAMEPLAI-shaped: PLAY_OVER snapshots off SCOUTING_FULL
    (built exactly as `scouting` exports them), priced by simulation and
    held at prod's lines until the next PLAY_OVER."""
    from collections import defaultdict
    from eAMFModel import v3_stream
    from . import directional, scouting
    v3_stream.model_paths(config.V3_MODEL_DIR)          # fail early, with instructions
    table = scouting.locate(cur, config.SCOUTING_TABLE)
    snapshots, prod_all = {}, []
    chunk = config.MATCH_CHUNK_SIZE
    for start in range(0, len(match_codes), chunk):
        batch = list(match_codes[start:start + chunk])
        rows = scouting.fetch_scouting(cur, table, batch)
        scores = fetch_scores(cur, batch)
        finals = fetch_final_scores(cur, batch)
        prod = fetch_quotes(cur, config.STREAMS["prod"], batch)
        prod_all.extend(prod)
        index = directional.index_by_message(prod)
        by_match, scores_by, prod_by = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in rows:
            by_match[r[0]].append(r)
        for s in scores:
            scores_by[s[0]].append(s)
        for q in prod:
            prod_by[q[0]].append(q)
        for match_code in batch:
            match_rows = by_match.get(match_code)
            if not match_rows:
                continue
            first_play = next((r[1] for r in match_rows
                               if scouting._text(r[4]) == "PLAY_STARTED"), None)
            snaps, _ = scouting.snapshots_for_match(
                match_code, match_rows, scores_by.get(match_code, []), finals.get(match_code),
                index, scouting._prematch(prod_by.get(match_code, []), first_play))
            if snaps:
                snapshots[match_code] = snaps
    print(f"  v3: {sum(len(v) for v in snapshots.values()):,} PLAY_OVER snapshots across "
          f"{len(snapshots):,} of {len(match_codes):,} matches; simulating "
          f"{config.V3_PATHS:,} games each", flush=True)
    return v3_stream.quotes_for_matches(snapshots, prod_all, config.V3_MODEL_DIR,
                                        n_paths=config.V3_PATHS, workers=config.V3_WORKERS)


def status_profile(cur, stream_table):
    """What values STATUS and IS_ACTIVE actually take, and how often.

    The pipeline reads liveness from IS_ACTIVE alone. This profile is what
    keeps STATUS honest: it shows how the two columns line up, which is
    the evidence for the claim that STATUS cannot be trusted.
    """
    stream_table = source_table(stream_table)
    predicate, params = window_predicate("PUBLISH_TIME")
    _, rows = fetch_all(cur, f"""
        SELECT STATUS, IS_ACTIVE, COUNT(*) AS N,
               COUNT(DISTINCT MATCH_CODE) AS MATCHES,
               AVG(CASE WHEN PROBABILITY IS NULL THEN 1 ELSE 0 END) AS NULL_PROB,
               AVG(CASE WHEN PROBABILITY = 0 THEN 1 ELSE 0 END) AS ZERO_PROB
        FROM {qualified(stream_table)}
        WHERE MARKET_ID IN ({_in_clause(MARKET_IDS)})
          AND {predicate}
        GROUP BY STATUS, IS_ACTIVE
        ORDER BY N DESC
    """, tuple(list(MARKET_IDS) + params))
    return rows


def fetch_message_times(cur, stream_table, match_codes):
    """(match, message count, time) for rebuilding the play clock.

    Deliberately NOT filtered on STATUS or PROBABILITY the way quotes are:
    this is establishing when a feed message happened, and a suspended or
    zero-priced market timestamps that message just as well as a live one.
    """
    stream_table = source_table(stream_table)
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
    stream_table = source_table(stream_table)
    predicate, params = window_predicate("PUBLISH_TIME")
    _, rows = fetch_all(cur, f"""
        SELECT COUNT(*), COUNT(DISTINCT MATCH_CODE), MIN(PUBLISH_TIME), MAX(PUBLISH_TIME)
        FROM {qualified(stream_table)}
        WHERE {predicate}
    """, tuple(params))
    return rows[0]
