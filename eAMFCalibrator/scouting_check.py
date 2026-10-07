"""Is SCOUTING_FULL complete? A day-by-day and match-by-match check, against a baseline of days
before a break (by default 23 Sep, when data was lost).

    python -m eAMFCalibrator scouting-check --since 2026-09-09            # to now, break 2026-09-23
    python -m eAMFCalibrator scouting-check --since 2026-09-09 --break 2026-09-23 --until 2026-10-06

Every expected match is one EVENT schedules for the sport (its result, where settled, off
SCORE_ENDGAME); every match SCOUTING_FULL has is checked on its own deduplicated messages:

    MISSING        scheduled and started, but no scouting rows at all
    NO_START       no FIRST_QUARTER_STARTED
    NO_END         no ENDED
    QUARTERS       fewer than four quarter starts
    GAPS           EVENT_MESSAGE_COUNT has holes between its first and last
    HEAD_MISSING   the first message count is later than the baseline's usual first
    FEW_MESSAGES   under half the baseline's median messages a match
    FEW_PLAYS      under half the baseline's median PLAY_OVERs a match
    NO_DETAIL      the PLAY_OVERs' field position mostly empty (where the baseline had it)
    NO_DOWN        the PLAY_OVERs' down and distance mostly missing (down 1-4, distance above 0)
    NO_CLOCK       the clock mostly empty (where the baseline had it)
    SCORE          the scoring messages do not add up to the settled final
    (info) NOT_SCHEDULED in scouting, not in EVENT; NO_FINAL no result yet; LIVE started in the
    last few hours, not judged; DUPLICATES the same message loaded more than once (collapsed)

Writes to --out:

    scouting_check.txt           the summary: per day, the columns and messages that changed
    scouting_check_days.csv      one row a day
    scouting_check_matches.csv   one row a match, with its flags
    scouting_check_columns.csv   every column's fill rate a day (all rows, and PLAY_OVER rows)
    scouting_check_messages.csv  every message kind's count a day
    scouting_check_detail_days.csv / _hours.csv
                                 down and distance (and field position, offensive team) a day / an
                                 hour: messages and PLAY_OVERs carrying them, the last such message
    scouting_check_detail_values.csv
                                 the commonest down, distance, field and team values a day
"""

import csv
import datetime as dt
import os
import re
import statistics
from collections import Counter, defaultdict

from . import config, scouting, snowflake_io
from .snowflake_io import fetch_all

BASELINE_DAYS = 14                 # default --since: this many days before the break
LIVE_HOURS = 4                     # a match started this recently is not judged yet
PAD_DAYS = 1                       # messages read this far either side, so edge matches are whole
FEW = 0.5                          # under this share of the baseline median is "few"
FILLED = 0.5                       # a column under this fill (where the baseline had 0.8+) is empty
COLUMN_SHIFT = 0.2                 # a column's daily fill this far off the baseline is reported
KIND_SHIFT = 0.5                   # a message kind's daily rate off the baseline by this factor
KIND_MIN = 5                       # ... on a day it was expected (or seen) at least this many times
KIND_DAYS = 3                      # ... on at least this many days

PROBLEMS = ("MISSING", "NO_START", "NO_END", "QUARTERS", "GAPS", "HEAD_MISSING", "NO_DOWN",
            "FEW_MESSAGES", "FEW_PLAYS", "NO_DETAIL", "NO_CLOCK", "SCORE")
INFO = ("NOT_SCHEDULED", "NO_FINAL", "LIVE", "DUPLICATES")

POINTS = {"TOUCHDOWN": 6, "EXTRA_POINT_GOOD": 1, "TWO_POINT_CONVERSION_SUCCESSFUL": 2,
          "FIELD_GOAL_GOOD": 3, "SAFETY_AWARDED": 2}
QUARTERS = ("FIRST_QUARTER_STARTED", "SECOND_QUARTER_STARTED",
            "THIRD_QUARTER_STARTED", "FOURTH_QUARTER_STARTED")
PLAIN = re.compile(r"^[A-Z_][A-Z0-9_$]*$")
JUMP = 0.3                         # an hour's down-and-distance share this far off the hour before
MIN_HOUR = 20                      # PLAY_OVERs an hour needs before its share is compared
TOP_VALUES = 12                    # values listed a day for each detail column


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

def _filled(column, kind):
    """1 where the column has a value (an empty or blank string counts as none)."""
    if (kind or "").upper().startswith(("TEXT", "VARCHAR", "STRING", "CHAR")):
        return f"IFF(NULLIF(TRIM({column}), '') IS NOT NULL, 1, 0)"
    return f"IFF({column} IS NOT NULL, 1, 0)"


def _down_ok():
    """1 where the message carries a usable down (1-4)."""
    return (f"IFF(TRY_TO_DOUBLE(TRIM(TO_VARCHAR({scouting.DOWN}))) BETWEEN 1 AND 4, 1, 0)")


def _dist_ok():
    """1 where the message carries a usable distance (above 0)."""
    return f"IFF(TRY_TO_DOUBLE(TRIM(TO_VARCHAR({scouting.DIST}))) > 0, 1, 0)"


def _dd_ok():
    """1 where the message carries both a usable down and distance."""
    return f"({_down_ok()} * {_dist_ok()})"


def _rows_cte(table, start, end):
    """raw: the table's AF rows with a message time in [start, end); s: one row a (match,
    message), the latest loaded."""
    ft = table.time_expr()
    loaded = table.time_expr("FILE_LOADED") if table.has("FILE_LOADED") else "NULL"
    sql = f"""
        raw AS (
            SELECT *, {ft} AS CHECK_FT, {loaded} AS CHECK_FL
            FROM {table.qualified}
            WHERE MATCH_CODE LIKE %s AND {ft} >= %s AND {ft} < %s),
        s AS (
            SELECT * FROM raw
            QUALIFY ROW_NUMBER() OVER (PARTITION BY MATCH_CODE, EVENT_MESSAGE_COUNT
                                       ORDER BY {table.latest_first()}) = 1)"""
    return sql, [scouting.MATCH_PREFIX + "%", start, end]


def fetch_expected(cur, start, end, sport=None):
    """EVENT's matches of the sport scheduled in [start, end), with the final where settled:
    {MATCH_CODE: {start, stream, final}}."""
    _, rows = fetch_all(cur, f"""
        WITH c AS (
            SELECT MATCH_CODE,
                   COALESCE(MAX(PLAYER_1_SCORE_CUMULATIVE), SUM(PLAYER_1_SCORE_CHANGE), 0) AS S1,
                   COALESCE(MAX(PLAYER_2_SCORE_CUMULATIVE), SUM(PLAYER_2_SCORE_CHANGE), 0) AS S2,
                   COUNT(*) AS N
            FROM {snowflake_io.qualified(snowflake_io.SCORE_TABLE)}
            WHERE MATCH_CODE LIKE %s
            GROUP BY 1)
        SELECT e.MATCH_CODE, MIN(e.SCHEDULED_START_TIME_UTC), MIN(e.STREAM_NUMBER),
               MAX(f.PLAYER_1_SCORE), MAX(f.PLAYER_2_SCORE), MAX(c.S1), MAX(c.S2), MAX(c.N),
               MAX(IFF(e.INPLAY_EVENT_STATUS = 'SETTLED', 1, 0))
        FROM {snowflake_io.qualified(snowflake_io.EVENT_TABLE)} e
        LEFT JOIN {snowflake_io.qualified(snowflake_io.FINAL_TABLE)} f ON f.MATCH_CODE = e.MATCH_CODE
        LEFT JOIN c ON c.MATCH_CODE = e.MATCH_CODE
        WHERE e.SPORT_CODE = %s AND e.SCHEDULED_START_TIME_UTC >= %s
          AND e.SCHEDULED_START_TIME_UTC < %s
        GROUP BY 1
    """, (scouting.MATCH_PREFIX + "%", sport or config.SPORT_CODE, start, end))
    out = {}
    for code, when, stream, p1, p2, c1, c2, n, settled in rows:
        final = (int(p1), int(p2)) if p1 is not None and p2 is not None else None
        changes = (int(c1), int(c2)) if n else None
        out[str(code)] = {"start": _time(when), "stream": "" if stream is None else str(stream),
                          "final": final, "score_changes": changes, "settled": bool(settled)}
    return out


def match_sql(table, start, end):
    """One row a scouting match: its messages, sequence, periods, plays, detail and points."""
    rows, params = _rows_cte(table, start, end)
    msg, status = scouting.MESSAGE, scouting.STATUS
    count = "TRY_TO_NUMBER(TO_VARCHAR(EVENT_MESSAGE_COUNT))"
    play_over = f"{msg} = 'PLAY_OVER'"
    cols = table.columns

    def detail(column):
        return f"SUM(IFF({play_over}, {_filled(column, cols.get(column))}, 0))"

    def points(team):
        return " + ".join(f"{pts} * SUM(IFF({msg} = '{kind}_TEAM_{team}', 1, 0))"
                          for kind, pts in POINTS.items())

    sql = f"""
        WITH {rows},
        r AS (SELECT MATCH_CODE, COUNT(*) AS RAW_ROWS FROM raw GROUP BY 1)
        SELECT s.MATCH_CODE AS MATCH_CODE,
               COUNT(*) AS MESSAGES,
               COUNT(DISTINCT {count}) AS DISTINCT_COUNTS,
               MIN({count}) AS FIRST_COUNT,
               MAX({count}) AS LAST_COUNT,
               MIN(CHECK_FT) AS FIRST_TIME,
               MAX(CHECK_FT) AS LAST_TIME,
               MAX(CHECK_FL) AS LAST_LOADED,
               ANY_VALUE(r.RAW_ROWS) AS RAW_ROWS,
               SUM(IFF({play_over}, 1, 0)) AS PLAY_OVERS,
               SUM(IFF({msg} = 'PLAY_STARTED', 1, 0)) AS PLAY_STARTS,
               MAX(IFF({status} = 'FIRST_QUARTER_STARTED', 1, 0)) AS STARTED,
               MAX(IFF({status} = 'ENDED', 1, 0)) AS ENDED,
               COUNT(DISTINCT IFF({status} IN ({", ".join(f"'{q}'" for q in QUARTERS)}),
                                  {status}, NULL)) AS QUARTERS,
               {detail(scouting.FIELD)} AS PO_FIELD,
               {detail(scouting.TEAM)} AS PO_TEAM,
               {detail(scouting.DOWN)} AS PO_DOWN,
               SUM(IFF({play_over}, {_dd_ok()}, 0)) AS PO_DD,
               SUM({_dd_ok()}) AS DD,
               SUM({_filled(scouting.CLOCK, cols.get(scouting.CLOCK))}) AS CLOCK_FILLED,
               {points("A")} AS POINTS_A,
               {points("B")} AS POINTS_B
        FROM s JOIN r ON r.MATCH_CODE = s.MATCH_CODE
        GROUP BY 1
    """
    return sql, params


def columns_sql(table, start, end):
    """Each day's rows, and each column's filled rows (all rows, and the PLAY_OVER rows).
    Returns (sql, params, the columns checked)."""
    rows, params = _rows_cte(table, start, end)
    checked = [c for c in table.columns if PLAIN.match(c)]
    play_over = f"{scouting.MESSAGE} = 'PLAY_OVER'"
    parts = []
    for k, c in enumerate(checked):
        f = _filled(c, table.columns[c])
        parts.append(f"SUM({f}) AS C{k}")
        parts.append(f"SUM(IFF({play_over}, {f}, 0)) AS P{k}")
    sql = f"""
        WITH {rows}
        SELECT TO_DATE(CHECK_FT) AS DAY, COUNT(*) AS N, SUM(IFF({play_over}, 1, 0)) AS N_PO,
               {", ".join(parts)}
        FROM s GROUP BY 1 ORDER BY 1
    """
    return sql, params, checked


def detail_sql(table, start, end, hourly=False):
    """Down and distance a day (or an hour): the messages and PLAY_OVERs carrying a usable down
    and distance, and each detail column on its own, with the first and last such message and the
    latest load."""
    rows, params = _rows_cte(table, start, end)
    po = f"IFF({scouting.MESSAGE} = 'PLAY_OVER', 1, 0)"
    cols = table.columns
    field = _filled(scouting.FIELD, cols.get(scouting.FIELD))
    team = _filled(scouting.TEAM, cols.get(scouting.TEAM))
    key = "DATE_TRUNC('HOUR', CHECK_FT)" if hourly else "TO_DATE(CHECK_FT)"
    sql = f"""
        WITH {rows},
        d AS (SELECT MATCH_CODE, CHECK_FT, CHECK_FL, {key} AS K, {po} AS PO, {_down_ok()} AS DOWN_OK,
                     {_dist_ok()} AS DIST_OK, {_dd_ok()} AS DD, {field} AS FIELD_OK, {team} AS TEAM_OK
              FROM s)
        SELECT K, COUNT(*) AS N, COUNT(DISTINCT MATCH_CODE) AS MATCHES, SUM(PO) AS N_PO,
               SUM(DOWN_OK) AS DOWN_OK, SUM(DIST_OK) AS DIST_OK, SUM(DD) AS DD,
               SUM(FIELD_OK) AS FIELD_OK, SUM(TEAM_OK) AS TEAM_OK,
               SUM(PO * DD) AS DD_PO, SUM(PO * FIELD_OK) AS FIELD_PO, SUM(PO * TEAM_OK) AS TEAM_PO,
               COUNT(DISTINCT IFF(DD = 1, MATCH_CODE, NULL)) AS MATCHES_DD,
               MIN(IFF(DD = 1, CHECK_FT, NULL)) AS FIRST_DD,
               MAX(IFF(DD = 1, CHECK_FT, NULL)) AS LAST_DD,
               MAX(CHECK_FL) AS LAST_LOADED
        FROM d GROUP BY 1 ORDER BY 1
    """
    return sql, params


def values_sql(table, start, end):
    """The commonest values a day of each detail column on the PLAY_OVERs (blank as '(none)')."""
    rows, params = _rows_cte(table, start, end)
    parts = [f"""SELECT TO_DATE(CHECK_FT) AS DAY, '{c}' AS COL,
                        COALESCE(NULLIF(TRIM(TO_VARCHAR({c})), ''), '(none)') AS VAL
                 FROM s WHERE {scouting.MESSAGE} = 'PLAY_OVER'"""
             for c in (scouting.DOWN, scouting.DIST, scouting.FIELD, scouting.TEAM)]
    sql = f"""
        WITH {rows},
        v AS ({" UNION ALL ".join(parts)})
        SELECT DAY, COL, VAL, COUNT(*) AS N FROM v GROUP BY 1, 2, 3
        QUALIFY ROW_NUMBER() OVER (PARTITION BY DAY, COL ORDER BY COUNT(*) DESC) <= {TOP_VALUES}
        ORDER BY 1, 2, 4 DESC
    """
    return sql, params


def messages_sql(table, start, end):
    """Each day's count of each message kind (in-play message, else the status)."""
    rows, params = _rows_cte(table, start, end)
    sql = f"""
        WITH {rows}
        SELECT TO_DATE(CHECK_FT) AS DAY,
               COALESCE(NULLIF(TRIM({scouting.MESSAGE}), ''),
                        'status ' || COALESCE(NULLIF(TRIM({scouting.STATUS}), ''), '(none)')) AS KIND,
               COUNT(*) AS N
        FROM s GROUP BY 1, 2
    """
    return sql, params


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _time(value):
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    try:
        return dt.datetime.fromisoformat(str(value)[:19].replace("T", " "))
    except ValueError:
        return None


def _num(value):
    return None if value is None or value == "" else int(float(value))


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def _share(part, whole):
    return part / whole if whole else None


def scouting_matches(cols, rows):
    """match_sql's rows as {MATCH_CODE: dict}."""
    out = {}
    for r in rows:
        d = dict(zip([c.upper() for c in cols], r))
        out[str(d["MATCH_CODE"])] = {
            "messages": _num(d["MESSAGES"]), "distinct": _num(d["DISTINCT_COUNTS"]),
            "first_count": _num(d["FIRST_COUNT"]), "last_count": _num(d["LAST_COUNT"]),
            "first_time": _time(d["FIRST_TIME"]), "last_time": _time(d["LAST_TIME"]),
            "last_loaded": _time(d["LAST_LOADED"]), "raw_rows": _num(d["RAW_ROWS"]),
            "play_overs": _num(d["PLAY_OVERS"]), "play_starts": _num(d["PLAY_STARTS"]),
            "started": bool(_num(d["STARTED"])), "ended": bool(_num(d["ENDED"])),
            "quarters": _num(d["QUARTERS"]), "po_field": _num(d["PO_FIELD"]),
            "po_team": _num(d["PO_TEAM"]), "po_down": _num(d["PO_DOWN"]),
            "po_dd": _num(d["PO_DD"]), "dd": _num(d["DD"]),
            "clock_filled": _num(d["CLOCK_FILLED"]),
            "points": (_num(d["POINTS_A"]), _num(d["POINTS_B"])),
        }
    return out


def assess(expected, found, since, until, brk, now):
    """One row a match (scheduled or in scouting) whose day falls in [since, until], each with
    its flags against the baseline (the days before `brk`). Returns (rows, baseline)."""
    rows = []
    for code in sorted(set(expected) | set(found)):
        e, s = expected.get(code), found.get(code)
        start = (e or {}).get("start") or (s or {}).get("first_time")
        if start is None or not since <= start.date() <= until:
            continue
        row = {"match_code": code, "day": start.date(), "start": start,
               "stream": (e or {}).get("stream", ""), "scheduled": e is not None,
               "final": (e or {}).get("final"), "in_scouting": s is not None,
               "score_changes": (e or {}).get("score_changes"), "settled": (e or {}).get("settled")}
        for key in ("messages", "distinct", "first_count", "last_count", "first_time", "last_time",
                    "last_loaded", "raw_rows", "play_overs", "play_starts", "started", "ended",
                    "quarters", "po_field", "po_team", "po_down", "po_dd", "dd", "clock_filled",
                    "points"):
            row[key] = (s or {}).get(key)
        if s:
            span = (s["last_count"] - s["first_count"] + 1) if s["first_count"] is not None else 0
            row["gaps"] = max(0, span - (s["distinct"] or 0))
            row["field_fill"] = _share(s["po_field"] or 0, s["play_overs"])
            row["dd_fill"] = _share(s["po_dd"] or 0, s["play_overs"])
            row["clock_fill"] = _share(s["clock_filled"] or 0, s["messages"])
            a, b = s["points"]
            row["score_agrees"] = (None if row["final"] is None or a is None else
                                   sorted((a, b)) == sorted(row["final"]))
        else:
            row.update(gaps=None, field_fill=None, dd_fill=None, clock_fill=None, score_agrees=None)
        row["live"] = (dt.timedelta(0) <= now - start < dt.timedelta(hours=LIVE_HOURS)
                       and not (s and s["ended"]))
        rows.append(row)

    base = [r for r in rows if r["in_scouting"] and r["day"] < brk and not r["live"]]
    firsts = Counter(r["first_count"] for r in base if r["first_count"] is not None)
    baseline = {
        "matches": len(base),
        "days": len({r["day"] for r in base}),
        "messages": _median(r["messages"] for r in base),
        "play_overs": _median(r["play_overs"] for r in base),
        "field_fill": _median(r["field_fill"] for r in base),
        "dd_fill": _median(r["dd_fill"] for r in base),
        "clock_fill": _median(r["clock_fill"] for r in base),
        "first_count": firsts.most_common(1)[0][0] if firsts else None,
        "score_agrees": _share(sum(1 for r in base if r["score_agrees"]),
                               sum(1 for r in base if r["score_agrees"] is not None)),
    }
    for r in rows:
        r["flags"] = _flags(r, baseline, now)
        r["problems"] = [f for f in r["flags"] if f in PROBLEMS]
    return rows, baseline


def _flags(r, base, now):
    flags = []
    if not r["scheduled"]:
        flags.append("NOT_SCHEDULED")
    if r["scheduled"] and r["final"] is None:
        flags.append("NO_FINAL")
    if r["raw_rows"] and r["messages"] and r["raw_rows"] > r["messages"]:
        flags.append("DUPLICATES")
    if r["live"]:
        return flags + ["LIVE"]
    if r["start"] > now:
        return flags
    if not r["in_scouting"]:
        return flags + ["MISSING"]
    if not r["started"]:
        flags.append("NO_START")
    if not r["ended"]:
        flags.append("NO_END")
    if (r["quarters"] or 0) < 4:
        flags.append("QUARTERS")
    if r["gaps"]:
        flags.append("GAPS")
    if base["first_count"] is not None and (r["first_count"] or 0) > base["first_count"]:
        flags.append("HEAD_MISSING")
    if base["messages"] and (r["messages"] or 0) < FEW * base["messages"]:
        flags.append("FEW_MESSAGES")
    if base["play_overs"] and (r["play_overs"] or 0) < FEW * base["play_overs"]:
        flags.append("FEW_PLAYS")
    if (base["field_fill"] or 0) >= 0.8 and (r["field_fill"] or 0) < FILLED:
        flags.append("NO_DETAIL")
    if (base["dd_fill"] or 0) >= 0.8 and (r["dd_fill"] or 0) < FILLED:
        flags.append("NO_DOWN")
    if (base["clock_fill"] or 0) >= 0.8 and (r["clock_fill"] or 0) < FILLED:
        flags.append("NO_CLOCK")
    if r["score_agrees"] is False:
        flags.append("SCORE")
    return flags


def days(rows, since, until, brk):
    """One summary row a day from `since` to `until`."""
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    out = []
    for k in range((until - since).days + 1):
        day = since + dt.timedelta(days=k)
        ms = by_day.get(day, [])
        judged = [m for m in ms if not m["live"] and m["start"] is not None]
        present = [m for m in ms if m["in_scouting"]]
        flags = Counter(f for m in ms for f in m["flags"])
        scored = [m for m in present if m["score_agrees"] is not None]
        loaded = [m["last_loaded"] for m in present if m["last_loaded"]]
        out.append({
            "day": day, "period": "baseline" if day < brk else "after",
            "scheduled": sum(1 for m in ms if m["scheduled"]),
            "settled": sum(1 for m in ms if m["final"] is not None),
            "event_settled": sum(1 for m in ms if m["settled"]),
            "with_score_changes": sum(1 for m in ms if m["score_changes"] is not None),
            "recovered": sum(1 for m in ms if m["final"] is None and m["settled"]
                             and m["score_changes"] is not None),
            "from_scouting": sum(1 for m in ms if m["final"] is None and m["settled"]
                                 and m["score_changes"] is None and m["in_scouting"] and m["ended"]),
            "scouting_agrees": _share(sum(1 for m in present if m["final"] and m["points"][0] is not None
                                          and tuple(m["points"]) == tuple(m["final"])),
                                      sum(1 for m in present if m["final"] and m["points"][0] is not None)),
            "sc_agrees": _share(sum(1 for m in ms if m["final"] and m["score_changes"]
                                    and tuple(m["score_changes"]) == tuple(m["final"])),
                                sum(1 for m in ms if m["final"] and m["score_changes"])),
            "in_scouting": len(present),
            "complete": sum(1 for m in judged if m["in_scouting"] and not m["problems"]),
            "incomplete": sum(1 for m in judged if m["in_scouting"] and m["problems"]),
            "missing": flags["MISSING"],
            "live": flags["LIVE"],
            "messages": _median(m["messages"] for m in present),
            "play_overs": _median(m["play_overs"] for m in present),
            "field_fill": _median(m["field_fill"] for m in present),
            "dd_fill": _median(m["dd_fill"] for m in present),
            "clock_fill": _median(m["clock_fill"] for m in present),
            "score_agrees": _share(sum(1 for m in scored if m["score_agrees"]), len(scored)),
            "last_loaded": max(loaded) if loaded else None,
            "flags": flags,
        })
    return out


def column_changes(cols, rows, checked, brk):
    """Per-day fill rates, and the columns whose fill moved off the baseline. Returns
    (long rows for the CSV, [(column, scope, baseline fill, {day: fill}) that changed])."""
    long_rows, fill = [], defaultdict(dict)
    for r in rows:
        d = dict(zip([c.upper() for c in cols], r))
        day = _time(d["DAY"]).date()
        n, n_po = _num(d["N"]) or 0, _num(d["N_PO"]) or 0
        for k, c in enumerate(checked):
            a, p = _num(d[f"C{k}"]) or 0, _num(d[f"P{k}"]) or 0
            long_rows.append({"day": day, "column": c, "rows": n, "filled": a,
                              "fill": _round(_share(a, n)), "play_over_rows": n_po,
                              "play_over_filled": p, "play_over_fill": _round(_share(p, n_po))})
            fill[(c, "all rows")][day] = _share(a, n)
            fill[(c, "PLAY_OVER")][day] = _share(p, n_po)
    changed = []
    for (c, scope), by_day in fill.items():
        base = _median(v for d, v in by_day.items() if d < brk)
        if base is None:
            continue
        after = {d: v for d, v in by_day.items() if d >= brk and v is not None}
        if any(abs(v - base) > COLUMN_SHIFT for v in after.values()):
            changed.append((c, scope, base, dict(sorted(by_day.items()))))
    return long_rows, sorted(changed)


def _kind_off(base, rate, matches):
    """Is a day's rate a match off the baseline, on enough messages to tell?"""
    expected, seen = base * matches, rate * matches
    if max(expected, seen) < KIND_MIN:
        return False
    return seen > 0 if base == 0 else rate < KIND_SHIFT * base or rate > base / KIND_SHIFT


def kind_changes(rows, matches_by_day, brk):
    """Per-day message counts, and the kinds whose rate a match moved off the baseline (or that
    vanished, or appeared) on KIND_DAYS+ days since the break. Returns (long rows,
    [(kind, baseline rate, {day: rate}, [days off])])."""
    long_rows, rate = [], defaultdict(dict)
    for day_value, kind, n in rows:
        day = _time(day_value).date()
        n = _num(n) or 0
        m = matches_by_day.get(day) or 0
        long_rows.append({"day": day, "kind": kind, "messages": n, "matches": m,
                          "per_match": _round(_share(n, m))})
        if m:
            rate[kind][day] = n / m
    all_days = sorted(matches_by_day)
    base_days = [d for d in all_days if d < brk and matches_by_day[d]]
    after_days = [d for d in all_days if d >= brk and matches_by_day[d]]
    changed = []
    for kind, by_day in rate.items():
        if not base_days or not after_days:
            continue
        base = _median([by_day.get(d, 0.0) for d in base_days])
        off = [d for d in after_days if _kind_off(base, by_day.get(d, 0.0), matches_by_day[d])]
        if len(off) >= KIND_DAYS:
            changed.append((kind, base, {d: by_day.get(d, 0.0) for d in all_days}, off))
    return long_rows, sorted(changed, key=lambda t: -t[1])


def detail_rows(cols, rows):
    """detail_sql's rows as dicts, with the shares worked out."""
    out = []
    for r in rows:
        d = {c.lower(): v for c, v in zip([c.upper() for c in cols], r)}
        row = {"when": _time(d["k"]), "first_dd": _time(d["first_dd"]),
               "last_dd": _time(d["last_dd"]), "last_loaded": _time(d["last_loaded"])}
        for k in ("n", "matches", "n_po", "down_ok", "dist_ok", "dd", "field_ok", "team_ok",
                  "dd_po", "field_po", "team_po", "matches_dd"):
            row[k] = _num(d[k]) or 0
        row["dd_share"] = _share(row["dd"], row["n"])
        row["dd_po_share"] = _share(row["dd_po"], row["n_po"])
        row["field_po_share"] = _share(row["field_po"], row["n_po"])
        row["team_po_share"] = _share(row["team_po"], row["n_po"])
        out.append(row)
    return out


def hour_jumps(hours):
    """The hours where the PLAY_OVERs' down-and-distance share moved more than JUMP from the
    hour before (both with MIN_HOUR+ PLAY_OVERs): [(before, after)]."""
    full = [h for h in hours if h["n_po"] >= MIN_HOUR]
    return [(a, b) for a, b in zip(full, full[1:])
            if abs(b["dd_po_share"] - a["dd_po_share"]) > JUMP]


def value_shares(rows, day_detail):
    """{(day, column): [(value, share of that day's PLAY_OVERs)]}, commonest first."""
    n_po = {d["when"].date(): d["n_po"] for d in day_detail}
    out = defaultdict(list)
    for day_value, column, value, n in rows:
        day = _time(day_value).date()
        out[(day, column)].append((str(value), _share(_num(n) or 0, n_po.get(day))))
    for k in out:
        out[k].sort(key=lambda t: -(t[1] or 0))
    return out


def detail_report(day_detail, hours, values, brk):
    """The down-and-distance section, as lines."""
    lines = []
    say = lines.append
    say("Down & distance (usable: down 1-4 and distance above 0), day by day:")
    head = (f"{'day':<12}{'msgs':>8}{'d&d':>8}{'':>5}{'PLAY_OVER':>10}{'d&d':>7}{'':>5}"
            f"{'field':>6}{'team':>6}{'matches':>10}  {'last d&d':<17}{'loaded':<17}")
    say(head)
    say("-" * len(head))
    for d in day_detail:
        day = d["when"].date()
        if day == brk:
            say(f"{'-- ' + brk.isoformat() + ' (break) ':-<{len(head)}}")
        say(f"{day.isoformat():<12}{d['n']:>8,}{d['dd']:>8,}{_pct(d['dd_share']):>5}{d['n_po']:>10,}"
            f"{d['dd_po']:>7,}{_pct(d['dd_po_share']):>5}{_pct(d['field_po_share']):>6}"
            f"{_pct(d['team_po_share']):>6}{d['matches_dd']:>5}/{d['matches']:<4}  "
            f"{_stamp(d['last_dd']):<17}{_stamp(d['last_loaded']):<17}")
    say("")
    say("msgs / d&d: messages, and those with a usable down and distance; PLAY_OVER / d&d: the same on "
        "the PLAY_OVERs; field, team: PLAY_OVERs with a field position / offensive team; matches: "
        "matches with any down and distance / matches; last d&d: the day's last message with them; "
        "loaded: the day's latest FILE_LOADED")
    last = max((d["last_dd"] for d in day_detail if d["last_dd"]), default=None)
    if last:
        since = [h for h in hours if h["when"] > last.replace(minute=0, second=0, microsecond=0)]
        say(f"Last message with a down and distance: {_stamp(last)}; since then "
            f"{sum(h['n_po'] for h in since):,} PLAY_OVERs in {sum(1 for h in since if h['n_po']):,} hours")
    else:
        say("No message in the window has a usable down and distance")
    say("")
    jumps = hour_jumps(hours)
    say(f"Hours where the PLAY_OVERs' down & distance share moved more than {100 * JUMP:.0f} points "
        f"from the hour before (hours with {MIN_HOUR}+ PLAY_OVERs):")
    if not jumps:
        say("  none")
    for a, b in jumps:
        say(f"  {_stamp(a['when'])} {_pct(a['dd_po_share']):>5} ({a['n_po']:,})  ->  "
            f"{_stamp(b['when'])} {_pct(b['dd_po_share']):>5} ({b['n_po']:,})")
    say("")
    say("Downs on the PLAY_OVERs, day by day (share of the day's PLAY_OVERs):")
    for d in day_detail:
        day = d["when"].date()
        shown = values.get((day, scouting.DOWN), [])[:7]
        say(f"  {day.isoformat():<12}" + "  ".join(f"{v} {_pct(sh)}" for v, sh in shown))
    return lines


def _stamp(value):
    return "-" if value is None else value.strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _round(v, places=3):
    return None if v is None else round(v, places)


def _pct(v):
    return "-" if v is None else f"{100 * v:.0f}%"


def _n(v):
    return "-" if v is None else f"{v:,.0f}"


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


MATCH_FIELDS = ["match_code", "day", "start", "stream", "scheduled", "in_scouting", "problems",
                "flags", "final", "points", "score_agrees", "messages", "raw_rows", "first_count",
                "last_count", "distinct", "gaps", "play_starts", "play_overs", "started", "ended",
                "settled", "score_changes", "quarters", "po_field", "po_team", "po_down", "po_dd", "dd",
                "field_fill", "dd_fill",
                "clock_fill",
                "first_time", "last_time", "last_loaded"]
DETAIL_FIELDS = ["when", "n", "matches", "n_po", "down_ok", "dist_ok", "dd", "dd_share", "dd_po",
                 "dd_po_share", "field_ok", "field_po", "field_po_share", "team_ok", "team_po",
                 "team_po_share", "matches_dd", "first_dd", "last_dd", "last_loaded"]
DAY_FIELDS = ["day", "period", "scheduled", "settled", "event_settled", "with_score_changes",
              "recovered", "sc_agrees", "from_scouting", "scouting_agrees", "in_scouting", "complete", "incomplete",
              "missing", "live", "messages", "play_overs", "field_fill", "dd_fill", "clock_fill",
              "score_agrees", "last_loaded"] + [f.lower() for f in PROBLEMS]


def report(table, since, until, brk, rows, baseline, day_rows, col_changed, kind_changed, now,
           detail=()):
    """The summary, as lines."""
    lines = []
    say = lines.append
    say(f"SCOUTING_FULL completeness: {table.qualified}")
    say(f"Matches scheduled (EVENT, sport {config.SPORT_CODE}) or in scouting from {since} to "
        f"{until} (UTC days); baseline: the days before {brk}")
    say("")
    if baseline["matches"]:
        say(f"Baseline: {baseline['matches']:,} matches over {baseline['days']} days; a match has a "
            f"median {_n(baseline['messages'])} messages and {_n(baseline['play_overs'])} PLAY_OVERs, "
            f"first message count {baseline['first_count']}, field position on "
            f"{_pct(baseline['field_fill'])} and down & distance on {_pct(baseline['dd_fill'])} "
            f"of PLAY_OVERs, the clock on {_pct(baseline['clock_fill'])} "
            f"of messages; scoring messages add up to the final in {_pct(baseline['score_agrees'])}")
    else:
        say(f"No baseline: no scouting matches before {brk} in the window (start --since earlier)")
    say("")
    head = (f"{'day':<12}{'sched':>6}{'final':>6}{'scout':>6}{'ok':>6}{'bad':>5}{'miss':>5}"
            f"{'msgs':>7}{'plays':>6}{'field':>6}{'d&d':>6}{'clock':>6}{'score':>6}  problems")
    say(head)
    say("-" * len(head))
    for d in day_rows:
        if d["day"] == brk:
            say(f"{'-- ' + brk.isoformat() + ' (break) ':-<{len(head)}}")
        probs = ", ".join(f"{f} {d['flags'][f]}" for f in PROBLEMS if d["flags"][f] and f != "MISSING")
        say(f"{d['day'].isoformat():<12}{d['scheduled']:>6}{d['settled']:>6}{d['in_scouting']:>6}"
            f"{d['complete']:>6}{d['incomplete']:>5}{d['missing']:>5}{_n(d['messages']):>7}"
            f"{_n(d['play_overs']):>6}{_pct(d['field_fill']):>6}{_pct(d['dd_fill']):>6}"
            f"{_pct(d['clock_fill']):>6}"
            f"{_pct(d['score_agrees']):>6}  {probs}" + (f"  ({d['live']} live)" if d["live"] else ""))
    say("")
    say("sched: EVENT's matches that day; final: settled; scout: in SCOUTING_FULL; ok / bad: in "
        "scouting with no problem / with one; miss: started, none in scouting; msgs, plays: median "
        "messages and PLAY_OVERs a match; field, d&d: median share of PLAY_OVERs with a field position, "
        "with a down and distance; "
        "clock: median share of messages with the clock; score: scoring messages add up to the final")

    after = [r for r in rows if r["day"] >= brk]
    total = Counter(f for r in after for f in r["problems"])
    judged = [r for r in after if not r["live"] and r["start"] <= now]
    say("")
    say(f"Since {brk}: {len(judged):,} matches started, {sum(1 for r in judged if not r['problems']):,} "
        f"complete, {sum(1 for r in judged if r['problems']):,} with a problem"
        + (": " + ", ".join(f"{k} {v}" for k, v in total.most_common()) if total else ""))
    bad_days = [d["day"] for d in day_rows if d["day"] >= brk and (d["incomplete"] or d["missing"])]
    if bad_days:
        say(f"Days with a missing or incomplete match: {', '.join(d.isoformat() for d in bad_days)}")
    clean = [d["day"] for d in day_rows if d["day"] >= brk and not (d["incomplete"] or d["missing"])
             and d["complete"]]
    if clean:
        say(f"Days fully complete: {', '.join(d.isoformat() for d in clean)}")
    streams = Counter(r["stream"] for r in after if "MISSING" in r["problems"])
    if streams:
        say("Missing matches by stream: " + ", ".join(f"{s or '?'} {n}" for s, n in streams.most_common()))

    say("")
    say(f"Columns whose daily fill moved more than {100 * COLUMN_SHIFT:.0f} points off the baseline "
        "(fill = share of rows with a value):")
    if not col_changed:
        say("  none")
    for column, scope, base, by_day in col_changed:
        changed = [(d, v) for d, v in by_day.items() if d >= brk and v is not None
                   and abs(v - base) > COLUMN_SHIFT]
        first = changed[0][0].isoformat() if changed else "-"
        recent = [v for d, v in by_day.items() if d >= brk and v is not None][-3:]
        say(f"  {column:<44}{scope:<11} baseline {_pct(base):>5}  first off {first}  "
            f"last days {' '.join(_pct(v) for v in recent)}  (off on {len(changed)} days)")

    say("")
    say(f"Message kinds whose daily rate a match fell under {KIND_SHIFT:.0%} or rose over "
        f"{1 / KIND_SHIFT:.0f}x of the baseline, or appeared, on {KIND_DAYS}+ days (days with "
        f"{KIND_MIN}+ expected or seen):")
    if not kind_changed:
        say("  none")
    for kind, base, by_day, off in kind_changed:
        recent = [v for d, v in by_day.items() if d >= brk][-3:]
        say(f"  {str(kind)[:43]:<44} baseline {base:>7.2f}/match  first off "
            f"{off[0].isoformat() if off else '-'}  last days {' '.join(f'{v:.2f}' for v in recent)}"
            f"  (off on {len(off)} days)")
    say("")
    say("Finals: SCORE_ENDGAME's, and what SCORE_CHANGES' last score would give a settled match without "
        "one (config.FINALS_FROM_SCORES):")
    head = (f"{'day':<12}{'sched':>6}{'settled':>9}{'endgame':>9}{'+changes':>10}{'changes=endgame':>17}"
            f"{'+scouting':>11}{'scouting=endgame':>18}")
    say(head)
    say("-" * len(head))
    for d in day_rows:
        say(f"{d['day'].isoformat():<12}{d['scheduled']:>6}{d['event_settled']:>9}{d['settled']:>9}"
            f"{d['recovered']:>10}{_pct(d['sc_agrees']):>17}{d['from_scouting']:>11}"
            f"{_pct(d['scouting_agrees']):>18}")
    say("settled: EVENT says SETTLED; endgame: SCORE_ENDGAME has a final; +changes: settled, no endgame "
        "final, but SCORE_CHANGES has its scores (recovered); changes=endgame: where both exist, the share "
        "whose SCORE_CHANGES last score is SCORE_ENDGAME's final; +scouting: settled, in neither, but in "
        "scouting to ENDED (config.FINALS_FROM_SCOUTING adds up its scoring messages); scouting=endgame: "
        "where both exist, the share whose scouting points, TEAM_A as PLAYER_1, are SCORE_ENDGAME's final "
        "exactly (the orientation check)")
    if detail:
        say("")
        lines.extend(detail)
    return lines


def run(cur, table, since, until, brk, out_dir, now=None):
    """Run the check over the match days [since, until]; write the files, print the summary.
    Returns the summary lines."""
    now = now or dt.datetime.utcnow()
    os.makedirs(out_dir, exist_ok=True)
    start = dt.datetime.combine(since, dt.time())
    end = dt.datetime.combine(until + dt.timedelta(days=1), dt.time())
    pad = dt.timedelta(days=PAD_DAYS)
    fmt = "%Y-%m-%d %H:%M:%S"

    print(f"  expected matches off {snowflake_io.EVENT_TABLE} ...", flush=True)
    expected = fetch_expected(cur, start.strftime(fmt), end.strftime(fmt))
    print(f"  {len(expected):,}; per-match scouting messages off {table.qualified} ...", flush=True)
    sql, params = match_sql(table, (start - pad).strftime(fmt), (end + pad).strftime(fmt))
    cols, raw = fetch_all(cur, sql, tuple(params))
    found = scouting_matches(cols, raw)
    rows, baseline = assess(expected, found, since, until, brk, now)
    day_rows = days(rows, since, until, brk)

    print(f"  {len(found):,} scouting matches; column fill a day ...", flush=True)
    sql, params, checked = columns_sql(table, start.strftime(fmt), end.strftime(fmt))
    ccols, crows = fetch_all(cur, sql, tuple(params))
    col_long, col_changed = column_changes(ccols, crows, checked, brk)

    print("  message kinds a day ...", flush=True)
    sql, params = messages_sql(table, start.strftime(fmt), end.strftime(fmt))
    _, krows = fetch_all(cur, sql, tuple(params))
    in_scouting = Counter(r["day"] for r in rows if r["in_scouting"])
    kind_long, kind_changed = kind_changes(krows, {d["day"]: in_scouting[d["day"]] for d in day_rows},
                                           brk)

    print("  down & distance a day and an hour ...", flush=True)
    window = (start.strftime(fmt), end.strftime(fmt))
    sql, params = detail_sql(table, *window)
    day_detail = detail_rows(*fetch_all(cur, sql, tuple(params)))
    sql, params = detail_sql(table, *window, hourly=True)
    hours = detail_rows(*fetch_all(cur, sql, tuple(params)))
    sql, params = values_sql(table, *window)
    _, vrows = fetch_all(cur, sql, tuple(params))
    values = value_shares(vrows, day_detail)
    detail = detail_report(day_detail, hours, values, brk)

    for d in day_rows:
        for f in PROBLEMS:
            d[f.lower()] = d["flags"][f]
    match_out = []
    for r in rows:
        r2 = dict(r)
        r2["flags"] = " ".join(r["flags"])
        r2["problems"] = " ".join(r["problems"])
        r2["final"] = "" if r["final"] is None else f"{r['final'][0]}-{r['final'][1]}"
        r2["score_changes"] = "" if not r["score_changes"] else f"{r['score_changes'][0]}-{r['score_changes'][1]}"
        r2["points"] = "" if not r["in_scouting"] else f"{r['points'][0]}-{r['points'][1]}"
        for k in ("field_fill", "dd_fill", "clock_fill"):
            r2[k] = _round(r[k])
        match_out.append(r2)
    match_out.sort(key=lambda r: (r["day"], r["start"] or dt.datetime.min, r["match_code"]))
    day_out = [dict(d, **{k: _round(d[k]) for k in ("field_fill", "dd_fill", "clock_fill",
                                                      "score_agrees")})
               for d in day_rows]

    _write_csv(os.path.join(out_dir, "scouting_check_matches.csv"), match_out, MATCH_FIELDS)
    _write_csv(os.path.join(out_dir, "scouting_check_days.csv"), day_out, DAY_FIELDS)
    _write_csv(os.path.join(out_dir, "scouting_check_columns.csv"), col_long,
               ["day", "column", "rows", "filled", "fill", "play_over_rows", "play_over_filled",
                "play_over_fill"])
    _write_csv(os.path.join(out_dir, "scouting_check_messages.csv"), kind_long,
               ["day", "kind", "messages", "matches", "per_match"])
    shares = ("dd_share", "dd_po_share", "field_po_share", "team_po_share")
    for name, rows_ in (("days", day_detail), ("hours", hours)):
        _write_csv(os.path.join(out_dir, f"scouting_check_detail_{name}.csv"),
                   [dict(r, **{k: _round(r[k]) for k in shares}) for r in rows_], DETAIL_FIELDS)
    _write_csv(os.path.join(out_dir, "scouting_check_detail_values.csv"),
               [{"day": day, "column": col, "value": v, "play_over_share": _round(sh)}
                for (day, col), vs in sorted(values.items()) for v, sh in vs],
               ["day", "column", "value", "play_over_share"])
    lines = report(table, since, until, brk, rows, baseline, day_rows, col_changed, kind_changed, now,
                   detail)
    with open(os.path.join(out_dir, "scouting_check.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print()
    print("\n".join(lines))
    print(f"\n  -> {out_dir}: scouting_check.txt, scouting_check_days.csv, "
          "scouting_check_matches.csv, scouting_check_columns.csv, scouting_check_messages.csv, "
          "scouting_check_detail_days.csv, scouting_check_detail_hours.csv, "
          "scouting_check_detail_values.csv")
    return lines
