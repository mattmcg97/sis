"""`python -m eAMFCalibrator run`: observations_for_chunk on fetch_quotes' 9-column rows."""
import datetime as dt
from collections import defaultdict
from unittest import mock
from eAMFCalibrator import pipeline, snowflake_io as io
T = dt.datetime(2026, 10, 1, 12)
plays = [("AF1", 10, 1, "Home Team", 1, 10, 25, T + dt.timedelta(seconds=10))]
quotes = [("AF1", 50, T + dt.timedelta(seconds=11), 55.0, 1.8, "Home", 10, "OPEN", "true")]  # fetch_quotes' SELECT
with mock.patch.object(io, "fetch_plays", lambda cur, codes, col: plays), \
     mock.patch.object(io, "fetch_scores", lambda cur, codes: []), \
     mock.patch.object(io, "fetch_final_scores", lambda cur, codes: {"AF1": (14, 7)}), \
     mock.patch.object(io, "fetch_quotes", lambda cur, table, codes: quotes):
    try:
        obs = pipeline.observations_for_chunk(None, "GAMEPLAI_STREAM", ["AF1"], "FILE_TIME", defaultdict(int))
        print("ok", obs)
    except Exception as e:
        print("crash:", type(e).__name__, e)
