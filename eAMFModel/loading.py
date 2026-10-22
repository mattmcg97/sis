"""A window's matches off the files: the match history, the window's settled matches, and the scouting
export's rows by match. Shared by every version's checks and reports (strength_check re-exports them)."""

import csv
import re

from . import books, nb2_prior, playover

# scouting-check's problems that leave a match's snaps wrong for a build: a quarter's start lost (the
# periods and clocks after it mislabelled) or holes in its messages (a lost PLAY_OVER merges two plays)
FEED_GATE = ("QUARTERS", "GAPS")


def feed_flags(path):
    """{match: its problems} off scouting-check's scouting_check_matches.csv ({} where there's no file)."""
    out = {}
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[r["match_code"]] = set(re.findall(r"[A-Z_]+", r.get("problems") or ""))
    except (OSError, KeyError):
        return {}
    return out


def gate(data, path):
    """(the export's matches less those scouting-check flagged with a FEED_GATE problem, how many went)."""
    flags = feed_flags(path) if path else {}
    drop = {c for c, f in flags.items() if f & set(FEED_GATE)}
    kept = {c: rows for c, rows in data.items() if c not in drop}
    return kept, len(data) - len(kept)


def window_codes(history, since, until):
    """The settled matches' codes in the window."""
    return {r["MATCH_CODE"] for r in history if books._finals(r) is not None
            and books._start(r) is not None and since <= books._start(r).date() <= until}


def load_snapshots(path, codes=None):
    """The export's rows by match in message order (playover.load), keeping only the `codes` given
    as it reads."""
    if codes is None:
        return playover.load(path)
    by_match = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["match_code"] in codes:
                by_match.setdefault(r["match_code"], []).append(r)
    for rows in by_match.values():
        rows.sort(key=lambda r: playover._int(r["message"]))
    return by_match


def history_rows(path):
    return nb2_prior.load_history(path)
