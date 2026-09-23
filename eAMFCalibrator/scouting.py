"""SCOUTING_FULL: the Madden scouting feed, and PLAY_OVER snapshots off it.

SCOUTING_FULL is the richer source behind the play table. Keyed like the
rest of the feed on (MATCH_CODE, EVENT_MESSAGE_COUNT), each message carries

    GAME_STATUS_MESSAGE     quarter starts / ends, NOT_STARTED, ENDED,
                            BET_SUSPEND / BET_UNSUSPEND / PERMANENT_BET_SUSPEND
    IN_PLAY_MESSAGE         PLAY_STARTED, PLAY_OVER, TOUCHDOWN_TEAM_A,
                            EXTRA_POINT_GOOD_TEAM_B, KICKOFF_TEAM_A, PUNT_...,
                            FIELD_GOAL_..., POSSESSION_TEAM_A, FOUL_..., ...
    IN_PLAY_CLOCK_SECONDS   the GAME CLOCK, counting down from 240 a quarter
    IN_PLAY_DETAIL_DATA_*   offensive team (TEAM_A / TEAM_B), down,
                            distance, field position

(column names and message vocabulary as read by MaddenScoutingAudit.py).

PLAY_OVER is the message to key snapshots on: the play is finished, the
state has settled, and the feed says so explicitly rather than leaving it
to be inferred from a change in down and distance. A snapshot here is a
PLAY_OVER message that GAMEPLAI_STREAM also quoted, with a probability on
at least one of the six markets.

Two commands use this module:

  probe    what the table is, what it carries, and how well PLAY_OVER
           lines up with GAMEPLAI_STREAM -- aggregate SQL, fast
  export   every such PLAY_OVER snapshot in the window, one row each, with
           the game state, the clock, the score, the final and prod's six
           quotes (line, probability, live, outcome). The offline input
           for eAMFModel's play-over backtest.

Duplicate (match, message) rows are collapsed to the latest loaded, as the
audit script does.
"""

import csv
import os
import re
from bisect import bisect_right
from collections import Counter, defaultdict

from . import config, directional, markets, snowflake_io
from .snowflake_io import fetch_all

DEFAULT_TABLE = "SCOUTING_FULL"
MATCH_PREFIX = "AF"
QUARTER_SECONDS = 240

TEAM = "IN_PLAY_DETAIL_DATA_OFFENSIVE_TEAM"
DOWN = "IN_PLAY_DETAIL_DATA_DOWN_NUMBER"
DIST = "IN_PLAY_DETAIL_DATA_DISTANCE"
FIELD = "IN_PLAY_DETAIL_DATA_FIELD_POSITION"
CLOCK = "IN_PLAY_CLOCK_SECONDS"
STATUS = "GAME_STATUS_MESSAGE"
MESSAGE = "IN_PLAY_MESSAGE"
NEEDED = ["MATCH_CODE", "EVENT_MESSAGE_COUNT", CLOCK, STATUS, MESSAGE,
          TEAM, DOWN, DIST, FIELD, "FILE_TIME"]

PERIOD_START = {"FIRST_QUARTER_STARTED": 1, "SECOND_QUARTER_STARTED": 2,
                "THIRD_QUARTER_STARTED": 3, "FOURTH_QUARTER_STARTED": 4,
                "OVERTIME_1_STARTED": 5, "OVERTIME_2_STARTED": 6}

# Play kinds, from the messages around a PLAY_STARTED .. PLAY_OVER pair
# (the audit script's classify_play, same order of precedence).
CONVERSION_MESSAGES = {f"{k}_TEAM_{t}" for t in "AB" for k in (
    "EXTRA_POINT_GOOD", "EXTRA_POINT_MISSED",
    "TWO_POINT_CONVERSION_SUCCESSFUL", "TWO_POINT_CONVERSION_FAILED")}
KICK_MESSAGES = {"KICKOFF_TEAM_A", "KICKOFF_TEAM_B", "ONSIDE_KICK_TEAM_A", "ONSIDE_KICK_TEAM_B"}
PUNT_MESSAGES = {"PUNT_TEAM_A", "PUNT_TEAM_B"}
FG_MESSAGES = {f"FIELD_GOAL_{r}_TEAM_{t}" for t in "AB" for r in ("GOOD", "MISSED")}
TD_MESSAGES = {"TOUCHDOWN_TEAM_A", "TOUCHDOWN_TEAM_B"}
SAFETY_MESSAGES = {"SAFETY_AWARDED_TEAM_A", "SAFETY_AWARDED_TEAM_B"}
DOWNS_MESSAGES = {"TURNOVER_ON_DOWNS_TEAM_A", "TURNOVER_ON_DOWNS_TEAM_B"}

# A snapshot on one of these is not a scrimmage state: the conversion and
# the kick. Exported (flagged) so the choice to drop them is visible.
NON_SCRIMMAGE = ("CONVERSION", "KICKOFF")

PROBABILITY_HINT = re.compile(r"PROB|PRICE|ODD|MARKET|SELECTION|HANDICAP|LINE", re.IGNORECASE)


def classify_play(pre_messages, in_messages):
    pre, inside = set(pre_messages), set(in_messages)
    everything = pre | inside
    if inside & CONVERSION_MESSAGES:
        return "CONVERSION"
    if everything & KICK_MESSAGES:
        return "KICKOFF"
    if everything & PUNT_MESSAGES:
        return "PUNT"
    if everything & FG_MESSAGES:
        return "FIELD_GOAL"
    if inside & TD_MESSAGES:
        return "TOUCHDOWN"
    if inside & SAFETY_MESSAGES:
        return "SAFETY"
    if inside & DOWNS_MESSAGES:
        return "TURNOVER_ON_DOWNS"
    return "SCRIMMAGE"


# ---------------------------------------------------------------------------
# Where the table is, and what it has
# ---------------------------------------------------------------------------

class Table:
    def __init__(self, qualified, columns):
        self.qualified = qualified
        self.columns = columns                      # {NAME: data_type}

    def has(self, name):
        return name in self.columns

    def time_expr(self, name="FILE_TIME"):
        kind = (self.columns.get(name) or "").upper()
        if kind.startswith(("TIMESTAMP", "DATE")):
            return name
        return f"TRY_TO_TIMESTAMP_NTZ(TO_VARCHAR({name}))"

    def latest_first(self):
        order = []
        if self.has("FILE_LOADED"):
            order.append(f"{self.time_expr('FILE_LOADED')} DESC NULLS LAST")
        order.append(f"{self.time_expr()} DESC NULLS LAST")
        return ", ".join(order)


def locate(cur, name=DEFAULT_TABLE):
    """Find the scouting table: a fully qualified name, or a search of the
    configured database for a table of that name (any schema, the
    configured schema preferred). Returns a Table, or raises."""
    parts = name.split(".")
    if len(parts) == 3:
        database, schema, table = parts
        candidates = [(database, schema, table)]
    else:
        _, rows = fetch_all(cur, f"""
            SELECT table_catalog, table_schema, table_name
            FROM {config.DATABASE}.INFORMATION_SCHEMA.TABLES
            WHERE UPPER(table_name) = UPPER(%s)
               OR UPPER(table_name) LIKE UPPER(%s)
            ORDER BY (UPPER(table_schema) = UPPER(%s)) DESC,
                     (UPPER(table_name) = UPPER(%s)) DESC
        """, (parts[-1], f"%{parts[-1]}%", config.SCHEMA, parts[-1]))
        candidates = [tuple(r) for r in rows]
    if not candidates:
        raise SystemExit(f"No table like {name!r} in {config.DATABASE}. "
                         "Pass --scouting-table DATABASE.SCHEMA.TABLE.")
    database, schema, table = candidates[0]
    _, cols = fetch_all(cur, f"""
        SELECT column_name, data_type
        FROM {database}.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """, (schema, table))
    found = Table(f"{database}.{schema}.{table}", {c.upper(): t for c, t in cols})
    found.others = [".".join(c) for c in candidates[1:]]
    missing = [c for c in NEEDED if not found.has(c)]
    if missing:
        raise SystemExit(f"{found.qualified} is missing {', '.join(missing)}; "
                         f"it has {', '.join(found.columns)}")
    return found


def scouting_rows_sql(table, columns, extra_where=""):
    """Deduplicated rows for AF matches inside the window."""
    predicate, params = snowflake_io.window_predicate(table.time_expr())
    sql = f"""
        SELECT {", ".join(columns)}
        FROM {table.qualified}
        WHERE MATCH_CODE LIKE %s AND {predicate} {extra_where}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY MATCH_CODE, EVENT_MESSAGE_COUNT
                                   ORDER BY {table.latest_first()}) = 1
    """
    return sql, [MATCH_PREFIX + "%"] + params


def _prod_table():
    return config.STREAMS[directional.PROD]


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe(cur, table, out):
    """Aggregate answers to: what is in the table, and does PLAY_OVER line
    up with GAMEPLAI_STREAM? Writes lines to `out` (a list)."""
    def say(line=""):
        out.append(line)
        print(line)

    say(f"Table: {table.qualified}")
    if getattr(table, "others", None):
        say(f"  other matches for the name: {', '.join(table.others)}")
    say(f"Columns ({len(table.columns)}):")
    for name, kind in table.columns.items():
        say(f"  {name:<44}{kind}")
    hinted = [c for c in table.columns if PROBABILITY_HINT.search(c)]
    say(f"\nColumns that look like markets or prices: {', '.join(hinted) or 'none'}")

    base_sql, base_params = scouting_rows_sql(
        table, ["MATCH_CODE", "EVENT_MESSAGE_COUNT", STATUS, MESSAGE, CLOCK, TEAM,
                DOWN, DIST, FIELD, table.time_expr() + " AS FT"])

    _, rows = fetch_all(cur, f"""
        WITH s AS ({base_sql})
        SELECT COUNT(*), COUNT(DISTINCT MATCH_CODE), MIN(FT), MAX(FT) FROM s
    """, tuple(base_params))
    n, matches, first, last = rows[0]
    say(f"\nWindow {config.CUTOFF_START} ->, MATCH_CODE LIKE '{MATCH_PREFIX}%': "
        f"{n:,} messages, {matches:,} matches, {first} -> {last}")

    for column in (MESSAGE, STATUS):
        _, rows = fetch_all(cur, f"""
            WITH s AS ({base_sql})
            SELECT {column}, COUNT(*) FROM s GROUP BY 1 ORDER BY 2 DESC LIMIT 80
        """, tuple(base_params))
        say(f"\n{column}:")
        for value, count in rows:
            say(f"  {str(value):<44}{count:>12,}")

    # Coverage: which scouting messages GAMEPLAI_STREAM quotes on.
    g_pred, g_params = snowflake_io.window_predicate("PUBLISH_TIME")
    market_list = ", ".join(str(m) for m in markets.MARKET_IDS)
    live = f"LOWER(TO_VARCHAR(IS_ACTIVE)) = '{config.LIVE_IS_ACTIVE}'"
    quotes_cte = f"""
        g AS (
            SELECT MATCH_CODE, EVENT_MESSAGE_COUNT,
                   COUNT(DISTINCT MARKET_ID) AS MARKETS,
                   COUNT(DISTINCT CASE WHEN {live} AND PROBABILITY > 0 THEN MARKET_ID END) AS LIVE_MARKETS
            FROM {snowflake_io.qualified(_prod_table())}
            WHERE MARKET_ID IN ({market_list}) AND PROBABILITY IS NOT NULL
              AND MATCH_CODE LIKE %s AND {g_pred}
            GROUP BY 1, 2)"""
    params = tuple(base_params + [MATCH_PREFIX + "%"] + g_params)
    _, rows = fetch_all(cur, f"""
        WITH s AS ({base_sql}), {quotes_cte}
        SELECT COALESCE(s.{MESSAGE}, '(none) ' || COALESCE(s.{STATUS}, '')) AS KIND,
               COUNT(*) AS N,
               COUNT(g.MATCH_CODE) AS QUOTED,
               SUM(CASE WHEN g.LIVE_MARKETS > 0 THEN 1 ELSE 0 END) AS LIVE_ANY,
               SUM(CASE WHEN g.LIVE_MARKETS = 6 THEN 1 ELSE 0 END) AS LIVE_ALL
        FROM s LEFT JOIN g ON g.MATCH_CODE = s.MATCH_CODE
                          AND g.EVENT_MESSAGE_COUNT = s.EVENT_MESSAGE_COUNT
        GROUP BY 1 ORDER BY 2 DESC LIMIT 60
    """, params)
    say(f"\nGAMEPLAI_STREAM ({_prod_table()}) quotes on the SAME message, by scouting message:")
    say(f"  {'message':<44}{'messages':>10}{'quoted':>9}{'1+ live':>9}{'6 live':>9}")
    for kind, n, quoted, live_any, live_all in rows:
        say(f"  {str(kind)[:43]:<44}{n:>10,}{_share(quoted, n):>9}"
            f"{_share(live_any, n):>9}{_share(live_all, n):>9}")

    _, rows = fetch_all(cur, f"""
        WITH s AS ({base_sql}),
        q AS (
            SELECT MATCH_CODE, EVENT_MESSAGE_COUNT, MARKET_ID,
                   MAX(CASE WHEN {live} AND PROBABILITY > 0 THEN 1 ELSE 0 END) AS LIVE
            FROM {snowflake_io.qualified(_prod_table())}
            WHERE MARKET_ID IN ({market_list}) AND MATCH_CODE LIKE %s AND {g_pred}
            GROUP BY 1, 2, 3)
        SELECT q.MARKET_ID, COUNT(*), SUM(q.LIVE)
        FROM s JOIN q ON q.MATCH_CODE = s.MATCH_CODE
                     AND q.EVENT_MESSAGE_COUNT = s.EVENT_MESSAGE_COUNT
        WHERE s.{MESSAGE} = 'PLAY_OVER'
        GROUP BY 1 ORDER BY 1
    """, tuple(base_params + [MATCH_PREFIX + "%"] + g_params))
    say("\nPLAY_OVER messages quoted, by market:")
    for market_id, quoted, live_n in rows:
        say(f"  {market_id:<6}{markets.market_group(market_id) or '':<11}"
            f"{markets.selection_label(market_id) or '':<7}{quoted:>10,} quoted  {live_n:>10,} live")

    # Does GAMEPLAI quote just after PLAY_OVER instead of on it?
    _, rows = fetch_all(cur, f"""
        WITH s AS ({base_sql}), {quotes_cte}
        SELECT d.SHIFT, COUNT(g.MATCH_CODE)
        FROM s CROSS JOIN (SELECT column1 AS SHIFT FROM VALUES (-2), (-1), (0), (1), (2), (3)) d
        LEFT JOIN g ON g.MATCH_CODE = s.MATCH_CODE
                   AND g.EVENT_MESSAGE_COUNT = s.EVENT_MESSAGE_COUNT + d.SHIFT
        WHERE s.{MESSAGE} = 'PLAY_OVER'
        GROUP BY 1 ORDER BY 1
    """, params)
    say("\nPLAY_OVER at message m: GAMEPLAI quotes at m + offset")
    for offset, quoted in rows:
        say(f"  {offset:+d}  {quoted:>10,}")

    # The clock at PLAY_OVER.
    _, rows = fetch_all(cur, f"""
        WITH s AS ({base_sql})
        SELECT {TEAM}, COUNT(*), MIN({CLOCK}), AVG({CLOCK}), MAX({CLOCK}),
               SUM(CASE WHEN {CLOCK} IS NULL THEN 1 ELSE 0 END),
               MIN({FIELD}), MAX({FIELD})
        FROM s WHERE {MESSAGE} = 'PLAY_OVER' GROUP BY 1 ORDER BY 2 DESC
    """, tuple(base_params))
    say("\nPLAY_OVER rows by offensive team: n, clock min/avg/max, clock missing, field min/max")
    for team, n, cmin, cavg, cmax, cnull, fmin, fmax in rows:
        say(f"  {str(team):<10}{n:>10,}  clock {cmin}/{_num(cavg)}/{cmax}  missing {cnull:,}"
            f"  field {fmin}..{fmax}")

    # Labels and field against the play table the suite has used so far.
    _, rows = fetch_all(cur, f"""
        WITH s AS ({base_sql})
        SELECT s.{TEAM}, p.OFFENSIVE_TEAM, COUNT(*),
               SUM(CASE WHEN s.{FIELD} = p.FIELD_POSITION THEN 1 ELSE 0 END),
               SUM(CASE WHEN s.{FIELD} = 100 - p.FIELD_POSITION THEN 1 ELSE 0 END),
               SUM(CASE WHEN s.{DOWN} = p.DOWN_NUMBER AND s.{DIST} = p.DISTANCE THEN 1 ELSE 0 END)
        FROM s JOIN {snowflake_io.qualified(snowflake_io.PLAY_TABLE)} p
          ON p.MATCH_CODE = s.MATCH_CODE AND p.EVENT_MESSAGE_COUNT = s.EVENT_MESSAGE_COUNT
        GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 12
    """, tuple(base_params))
    say(f"\nAgainst {snowflake_io.PLAY_TABLE} on the same message: "
        "scouting team, play-table team, rows, same field, mirrored field, same down+distance")
    for s_team, p_team, n, same, mirrored, dd in rows:
        say(f"  {str(s_team):<10}{str(p_team):<14}{n:>10,}{_share(same, n):>9}"
            f"{_share(mirrored, n):>9}{_share(dd, n):>9}")

    # TEAM_A / TEAM_B against the scoreboard's PLAYER_1 / PLAYER_2.
    _, rows = fetch_all(cur, f"""
        WITH s AS ({base_sql})
        SELECT s.{MESSAGE},
               SUM(CASE WHEN c.PLAYER_1_SCORE_CHANGE > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN c.PLAYER_2_SCORE_CHANGE > 0 THEN 1 ELSE 0 END),
               COUNT(DISTINCT s.MATCH_CODE)
        FROM s JOIN {snowflake_io.qualified(snowflake_io.SCORE_TABLE)} c
          ON c.MATCH_CODE = s.MATCH_CODE
         AND c.EVENT_MESSAGE_COUNT BETWEEN s.EVENT_MESSAGE_COUNT AND s.EVENT_MESSAGE_COUNT + 3
        WHERE s.{MESSAGE} IN ('TOUCHDOWN_TEAM_A', 'TOUCHDOWN_TEAM_B',
                              'FIELD_GOAL_GOOD_TEAM_A', 'FIELD_GOAL_GOOD_TEAM_B')
        GROUP BY 1 ORDER BY 1
    """, tuple(base_params))
    say("\nScoring messages against SCORE_CHANGES within 3 messages: "
        "PLAYER_1 (home) scored, PLAYER_2 (away) scored, matches")
    for message, p1, p2, n_matches in rows:
        say(f"  {message:<28}{p1:>8,}{p2:>8,}{n_matches:>8,}")
    say("\nIf TEAM_A always lines up with one PLAYER the mapping is fixed; if it splits,")
    say("it is per match, and the export resolves it match by match off these same scores.")
    return out


def _share(part, whole):
    return "-" if not whole else f"{100.0 * (part or 0) / whole:.1f}%"


def _num(value):
    return "-" if value is None else f"{float(value):.1f}"


# ---------------------------------------------------------------------------
# Export: one row per PLAY_OVER snapshot that GAMEPLAI quoted
# ---------------------------------------------------------------------------

EXPORT_FIELDS = [
    "match_code", "message", "file_time", "publish_time", "period", "clock_seconds",
    "bet_state", "play_kind", "scrimmage", "team_a_side",
    "offense", "down", "distance", "field_position",
    "next_start_message", "next_offense", "next_down", "next_distance", "next_field_position",
    "score_p1", "score_p2", "score_p1_at_start", "score_p2_at_start", "play_messages",
    "final_p1", "final_p2",
    "first_play_message", "opening_offense", "home_handle", "away_handle",
    "next_start_clock",
]
for _m in markets.MARKET_IDS:
    EXPORT_FIELDS += [f"line_{_m}", f"prob_{_m}", f"live_{_m}", f"outcome_{_m}"]
EXPORT_FIELDS += ["prematch_line_52", "prematch_prob_52", "prematch_line_54",
                  "prematch_prob_54", "prematch_prob_50", "live_markets"]


def scouting_matches(cur, table):
    sql, params = scouting_rows_sql(table, ["MATCH_CODE", MESSAGE])
    _, rows = fetch_all(cur, f"""
        WITH s AS ({sql}) SELECT DISTINCT MATCH_CODE FROM s WHERE {MESSAGE} = 'PLAY_OVER'
    """, tuple(params))
    return sorted(r[0] for r in rows)


def fetch_scouting(cur, table, match_codes):
    in_clause = ", ".join(["%s"] * len(match_codes))
    sql, params = scouting_rows_sql(
        table, ["MATCH_CODE", "EVENT_MESSAGE_COUNT", CLOCK, STATUS, MESSAGE, TEAM,
                DOWN, DIST, FIELD, "FILE_TIME"],
        extra_where=f"AND MATCH_CODE IN ({in_clause})")
    _, rows = fetch_all(cur, sql + " ORDER BY MATCH_CODE, EVENT_MESSAGE_COUNT",
                        tuple(params + list(match_codes)))
    return rows


def _text(value):
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _int(value):
    try:
        return None if value is None else int(float(value))
    except (TypeError, ValueError):
        return None


def team_a_side(rows, scores):
    """'home' / 'away' / None: which scoreboard side TEAM_A is in this match,
    read off its scoring messages against the next score change."""
    changes = sorted((s[1], s[3] or 0, s[4] or 0) for s in scores if s[1] is not None)
    votes = Counter()
    for r in rows:
        message = _text(r[4])
        team = None
        if message in TD_MESSAGES or message in {"FIELD_GOAL_GOOD_TEAM_A", "FIELD_GOAL_GOOD_TEAM_B"}:
            team = message[-6:]
        if team is None:
            continue
        m = r[1]
        hit = next(((p1, p2) for msg, p1, p2 in changes if m <= msg <= m + 3), None)
        if hit is None:
            continue
        scorer = "home" if hit[0] > 0 and not hit[1] else "away" if hit[1] > 0 and not hit[0] else None
        if scorer:
            a_side = scorer if team == "TEAM_A" else ("away" if scorer == "home" else "home")
            votes[a_side] += 1
    if not votes:
        return None
    side, n = votes.most_common(1)[0]
    return side if n >= 0.8 * sum(votes.values()) else None


def snapshots_for_match(match_code, rows, scores, final, quotes_index, prematch):
    """[dict] export rows for one match, and a Counter of what was dropped."""
    dropped = Counter()
    a_side = team_a_side(rows, scores)
    score_changes = sorted((s[1], s[5], s[6]) for s in scores if s[1] is not None)
    period = None
    bet_state = "UNKNOWN"
    open_start = None
    since_last_over = []
    first_play = None
    opening = None
    out = []
    starts = [(r[1], i) for i, r in enumerate(rows) if _text(r[4]) == "PLAY_STARTED"]
    start_msgs = [m for m, _ in starts]

    for i, r in enumerate(rows):
        message = r[1]
        status, kind = _text(r[3]), _text(r[4])
        if status in PERIOD_START:
            period = PERIOD_START[status]
        if status in ("BET_SUSPEND", "BET_UNSUSPEND", "PERMANENT_BET_SUSPEND"):
            bet_state = status
        if kind == "PLAY_STARTED":
            open_start = i
            if first_play is None:
                first_play = message
                opening = _text(r[5])
        if kind is not None:
            since_last_over.append(kind)
        if kind != "PLAY_OVER":
            continue

        pre = since_last_over[:-1]
        in_play = []
        start_message = None
        if open_start is not None:
            in_play = [_text(x[4]) for x in rows[open_start:i + 1] if _text(x[4])]
            start_message = rows[open_start][1]
        play_kind = classify_play(pre, in_play)
        since_last_over = []
        open_start = None

        quoted = {m: quotes_index.get((match_code, m), {}).get(message) for m in markets.MARKET_IDS}
        if not any(q is not None for q in quoted.values()):
            dropped["play_over_not_quoted"] += 1
            continue

        j = bisect_right(start_msgs, message)
        nxt = rows[starts[j][1]] if j < len(starts) else None
        p1, p2 = _score_at(score_changes, message)
        s1, s2 = _score_at(score_changes, start_message if start_message is not None else message)

        row = {
            "match_code": match_code, "message": message, "file_time": r[9],
            "publish_time": next((q.publish_time for q in quoted.values() if q), None),
            "period": period, "clock_seconds": r[2], "bet_state": bet_state,
            "play_kind": play_kind, "scrimmage": int(play_kind not in NON_SCRIMMAGE),
            "team_a_side": a_side,
            "offense": _text(r[5]), "down": _int(r[6]), "distance": _int(r[7]),
            "field_position": _int(r[8]),
            "next_start_message": nxt[1] if nxt else None,
            "next_offense": _text(nxt[5]) if nxt else None,
            "next_down": _int(nxt[6]) if nxt else None,
            "next_distance": _int(nxt[7]) if nxt else None,
            "next_field_position": _int(nxt[8]) if nxt else None,
            "next_start_clock": nxt[2] if nxt else None,
            "score_p1": p1, "score_p2": p2, "score_p1_at_start": s1, "score_p2_at_start": s2,
            "play_messages": "|".join(m for m in in_play if m not in ("PLAY_STARTED", "PLAY_OVER")),
            "final_p1": final[0] if final else None, "final_p2": final[1] if final else None,
            "first_play_message": first_play, "opening_offense": opening,
        }
        live_markets = 0
        for m, q in quoted.items():
            if q is None:
                continue
            line = markets.parse_line(q.description) if markets.needs_line(m) else None
            outcome = markets.resolve(m, line, *final) if final else None
            row[f"line_{m}"] = line
            row[f"prob_{m}"] = q.probability / 100.0
            row[f"live_{m}"] = int(q.live)
            row[f"outcome_{m}"] = "" if outcome is None else int(outcome)
            live_markets += int(q.live and q.probability > 0)
        row["live_markets"] = live_markets
        for key, value in prematch.items():
            row[key] = value
        out.append(row)
    return out, dropped


def _score_at(score_changes, message):
    p1 = p2 = 0
    for msg, c1, c2 in score_changes:
        if msg > message:
            break
        p1 = c1 if c1 is not None else p1
        p2 = c2 if c2 is not None else p2
    return p1, p2


def _prematch(prod_rows, first_play_message):
    """Prod's last quote before the first PLAY_STARTED, markets 50/52/54."""
    out = {}
    best = {}
    for (_, market_id, publish_time, probability, _, text, message, _, _) in prod_rows:
        if market_id not in (50, 52, 54) or probability is None:
            continue
        pre = message is None or (first_play_message is not None and message < first_play_message)
        if not pre:
            continue
        key = (publish_time is None, publish_time)
        if market_id not in best or key > best[market_id][0]:
            best[market_id] = (key, markets.parse_line(text), float(probability) / 100.0)
    for market_id, (_, line, prob) in best.items():
        if market_id == 50:
            out["prematch_prob_50"] = prob
        else:
            out[f"prematch_line_{market_id}"] = line
            out[f"prematch_prob_{market_id}"] = prob
    return out


def fetch_handles(cur, match_codes):
    """match -> (PLAYER_1_HANDLE, PLAYER_2_HANDLE) from the EVENT table.
    TEAM_A is PLAYER_1 (home) in this feed, so these are TEAM_A's and
    TEAM_B's gamers."""
    in_clause = ", ".join(["%s"] * len(match_codes))
    _, rows = fetch_all(cur, f"""
        SELECT MATCH_CODE, PLAYER_1_HANDLE, PLAYER_2_HANDLE
        FROM {snowflake_io.qualified(snowflake_io.EVENT_TABLE)}
        WHERE MATCH_CODE IN ({in_clause})
    """, tuple(match_codes))
    return {r[0]: (_text(r[1]), _text(r[2])) for r in rows}


def export(cur, table, path, limit=None, verbose=True):
    """Write every quoted PLAY_OVER snapshot in the window to `path`."""
    prod_matches = set(snowflake_io.match_universe(cur, _prod_table()))
    match_codes = [m for m in scouting_matches(cur, table) if m in prod_matches]
    if limit:
        match_codes = match_codes[-limit:]
    if verbose:
        print(f"\nExporting PLAY_OVER snapshots for {len(match_codes):,} matches "
              "(AF, settled, quoted by GAMEPLAI_STREAM)")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    totals = Counter()
    by_kind = Counter()
    by_market_live = Counter()
    written = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        chunk = config.MATCH_CHUNK_SIZE
        for start in range(0, len(match_codes), chunk):
            batch = match_codes[start:start + chunk]
            if verbose:
                print(f"  chunk {start // chunk + 1}: {len(batch)} matches", flush=True)
            rows = fetch_scouting(cur, table, batch)
            scores = snowflake_io.fetch_scores(cur, batch)
            finals = snowflake_io.fetch_final_scores(cur, batch)
            prod = snowflake_io.fetch_quotes(cur, _prod_table(), batch)
            handles = fetch_handles(cur, batch)
            index = directional.index_by_message(prod)
            by_match = defaultdict(list)
            for r in rows:
                by_match[r[0]].append(r)
            scores_by = defaultdict(list)
            for s in scores:
                scores_by[s[0]].append(s)
            prod_by = defaultdict(list)
            for q in prod:
                prod_by[q[0]].append(q)
            for match_code in batch:
                match_rows = by_match.get(match_code, [])
                if not match_rows:
                    continue
                first_play = next((r[1] for r in match_rows if _text(r[4]) == "PLAY_STARTED"), None)
                snaps, dropped = snapshots_for_match(
                    match_code, match_rows, scores_by.get(match_code, []),
                    finals.get(match_code), index, _prematch(prod_by.get(match_code, []), first_play))
                totals.update(dropped)
                totals["matches_with_snapshots"] += bool(snaps)
                home, away = handles.get(match_code, (None, None))
                for snap in snaps:
                    snap["home_handle"], snap["away_handle"] = home, away
                    writer.writerow(snap)
                    written += 1
                    by_kind[snap["play_kind"]] += 1
                    totals["team_a_unresolved"] += snap["team_a_side"] is None
                    for m in markets.MARKET_IDS:
                        by_market_live[m] += bool(snap.get(f"live_{m}"))
    if verbose:
        print(f"\n  {written:,} PLAY_OVER snapshots quoted by GAMEPLAI_STREAM -> {path}")
        print(f"  matches with a snapshot: {totals['matches_with_snapshots']:,}")
        print(f"  PLAY_OVER messages GAMEPLAI never quoted: {totals['play_over_not_quoted']:,}")
        print(f"  snapshots whose TEAM_A side is unresolved: {totals['team_a_unresolved']:,}")
        print("  by play kind: " + ", ".join(f"{k} {v:,}" for k, v in by_kind.most_common()))
        print("  live quotes by market: " + ", ".join(
            f"{m} {by_market_live[m]:,}" for m in markets.MARKET_IDS))
    return written


def write_sample(cur, table, match_codes, path):
    """Every column of SCOUTING_FULL for a few matches, with prod's six
    probabilities on the same message alongside, for reading by eye."""
    in_clause = ", ".join(["%s"] * len(match_codes))
    cols = list(table.columns)
    sql, params = scouting_rows_sql(table, cols, extra_where=f"AND MATCH_CODE IN ({in_clause})")
    _, rows = fetch_all(cur, sql + " ORDER BY MATCH_CODE, EVENT_MESSAGE_COUNT",
                        tuple(params + list(match_codes)))
    index = directional.index_by_message(snowflake_io.fetch_quotes(cur, _prod_table(), match_codes))
    i_mc, i_msg = cols.index("MATCH_CODE"), cols.index("EVENT_MESSAGE_COUNT")
    extra = []
    for m in markets.MARKET_IDS:
        extra += [f"PROD_LINE_{m}", f"PROD_PROB_{m}", f"PROD_LIVE_{m}"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(cols + extra)
        for r in rows:
            values = list(r)
            for m in markets.MARKET_IDS:
                q = index.get((r[i_mc], m), {}).get(r[i_msg])
                values += ([markets.parse_line(q.description), q.probability, int(q.live)]
                           if q else ["", "", ""])
            writer.writerow(values)
    return len(rows)
