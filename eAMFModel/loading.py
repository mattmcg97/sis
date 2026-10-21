"""A window's matches off the files: the match history, the window's settled matches, and the scouting
export's rows by match. Shared by every version's checks and reports (strength_check re-exports them)."""

import csv

from . import books, nb2_prior, playover


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
