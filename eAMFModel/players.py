"""Player effects: how each gamer plays the clock and the 4th down.

Measured on SCOUTING_FULL PLAY_OVER snapshots (eAMFCalibrator scouting's
export) against the league, situation by situation, and shrunk toward the
league by how much evidence there is:

  pace        seconds of game clock the player's offense uses between one
              whistle and the next, over what the league uses in the same
              situation (score margin, whether it is the last two minutes of
              a half). 1.05 = 5% slower = fewer drives in his games. The
              players' means run from about 19 to 27 seconds against a
              league 23, and it is stable (split-half correlation 0.73).
  aggression  on 4th down, the shift in log odds of going for it against
              the league's fitted rate for that down, distance and field
              (drive.go_probability). Players run from about 16% to 93%
              where the league goes 44%: -1.4 to +2.8.
  milk        seconds per play when protecting a lead in the last two
              minutes of a half, over the league's in the same spot --
              how hard the player kills the clock.

Also here: the league's situation table of seconds per play, and the
running in-game pace estimate the pricer uses (Pace).
"""

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

from .drive import DriveParams, go_probability

# Evidence needed before a player's own number counts for half, set from the
# spread between players against the noise within them: pace varies ~6%
# between players and a single play's clock use ~50% (so ~70 plays); 4th-down
# go rates vary ~0.8 in log odds between players (a prior of ~1.6 in units
# of Fisher information, p(1-p) summed over decisions); milking is noisier.
PACE_PRIOR_PLAYS = 70
AGGRESSION_PRIOR = 2.0
MILK_PRIOR_PLAYS = 40

LATE_SECONDS = 120             # the "last two minutes" of a half


def margin_bucket(margin):
    if margin >= 9:
        return "lead9+"
    if margin > 0:
        return "lead1-8"
    if margin == 0:
        return "tied"
    if margin >= -8:
        return "trail1-8"
    return "trail9+"


def situation(period, clock_seconds, margin):
    """(late, margin bucket) for an offense at a whistle."""
    half_left = clock_seconds + (240 if period % 2 == 1 else 0)
    return (half_left <= LATE_SECONDS, margin_bucket(margin))


def _f(v):
    return float(v) if v not in ("", None) else None


def _i(v):
    return int(float(v)) if v not in ("", None) else None


def play_intervals(rows):
    """Consecutive whistles by the same offense in one period of one match:
    (offense TEAM_x, situation, seconds between them). `rows` are one
    match's export rows in message order."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if not a["period"] or a["period"] != b["period"] or int(a["period"]) > 4:
            continue
        if a["play_kind"] not in ("SCRIMMAGE", "KICKOFF", "PUNT"):
            continue
        if b["play_kind"] != "SCRIMMAGE" or b["offense"] != a["offense"] or not a["down"]:
            continue
        seconds = _f(a["clock_seconds"]) - _f(b["clock_seconds"])
        if not 0 <= seconds <= 60:
            continue
        margin = (_i(a["score_p1"]) - _i(a["score_p2"])) * (1 if a["offense"] == "TEAM_A" else -1)
        out.append((a["offense"], situation(int(a["period"]), _f(a["clock_seconds"]), margin),
                    seconds, int(b["message"])))
    return out


def fourth_downs(rows):
    """(offense, field, distance, went for it) for every non-desperate 4th
    down in one match."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["down"] != "4" or not a["field_position"] or not a["period"] or int(a["period"]) > 4:
            continue
        p = int(a["period"])
        half_left = _f(a["clock_seconds"]) + (240 if p % 2 == 1 else 0)
        margin = (_i(a["score_p1"]) - _i(a["score_p2"])) * (1 if a["offense"] == "TEAM_A" else -1)
        if (p >= 3 and half_left <= 240 and margin < 0) or half_left <= 30:
            continue
        went = b["play_kind"] not in ("PUNT", "FIELD_GOAL")
        out.append((a["offense"], int(a["field_position"]), max(1, int(a["distance"])), went))
    return out


@dataclass
class Profile:
    plays: int = 0
    pace: float = 1.0
    aggression: float = 0.0
    fourth_downs: int = 0
    milk: float = 1.0
    milk_plays: int = 0


@dataclass
class Book:
    """League situation table and every player's profile."""
    seconds: dict = field(default_factory=dict)      # "late|bucket" -> league mean
    players: dict = field(default_factory=dict)      # handle -> Profile

    def expected_seconds(self, sit):
        return self.seconds.get(f"{int(sit[0])}|{sit[1]}", 23.0)

    def profile(self, handle):
        return self.players.get((handle or "").upper(), Profile())

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"seconds": self.seconds,
                       "players": {h: p.__dict__ for h, p in self.players.items()}}, fh, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(raw["seconds"], {h: Profile(**p) for h, p in raw["players"].items()})


def load_handles(path):
    """match -> (home handle, away handle), from a CSV with MATCH_CODE,
    PLAYER_1_HANDLE, PLAYER_2_HANDLE (nb2/AMFELO.csv has them), or from the
    export's own home_handle / away_handle columns when present."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            code = r.get("MATCH_CODE") or r.get("match_code")
            home = r.get("PLAYER_1_HANDLE") or r.get("home_handle")
            away = r.get("PLAYER_2_HANDLE") or r.get("away_handle")
            if code and home and away:
                out[code] = (home.strip().upper(), away.strip().upper())
    return out


def build(matches, handles, drive_params=None):
    """A Book from {match: rows} (export rows, message order) and handles.

    TEAM_A is PLAYER_1 (home) throughout the feed -- the scouting probe
    showed it on every row -- so TEAM_A's handle is the home handle.
    """
    drive_params = drive_params or DriveParams()
    sums = defaultdict(lambda: [0.0, 0])
    intervals = []
    fourths = []
    for code, rows in matches.items():
        h = handles.get(code)
        for offense, sit, seconds, _ in play_intervals(rows):
            key = f"{int(sit[0])}|{sit[1]}"
            sums[key][0] += seconds
            sums[key][1] += 1
            if h:
                intervals.append((h[0] if offense == "TEAM_A" else h[1], sit, seconds))
        if h:
            for offense, y, t, went in fourth_downs(rows):
                fourths.append((h[0] if offense == "TEAM_A" else h[1], y, t, went))
    book = Book(seconds={k: s / n for k, (s, n) in sums.items() if n})

    obs = defaultdict(lambda: [0.0, 0.0, 0])            # seconds, expected, plays
    milk = defaultdict(lambda: [0.0, 0.0, 0])
    for who, sit, seconds in intervals:
        exp = book.expected_seconds(sit)
        if sit[0] and sit[1].startswith("lead"):
            m = milk[who]
            m[0] += seconds
            m[1] += exp
            m[2] += 1
        elif not sit[0]:
            o = obs[who]
            o[0] += seconds
            o[1] += exp
            o[2] += 1
    agg = defaultdict(lambda: [0.0, 0.0, 0])            # sum(go - p), sum p(1-p), n
    for who, y, t, went in fourths:
        p = go_probability(drive_params, y, t)
        a = agg[who]
        a[0] += went - p
        a[1] += p * (1 - p)
        a[2] += 1

    for who in set(obs) | set(agg) | set(milk):
        prof = Profile()
        s, e, n = obs.get(who, (0.0, 0.0, 0))
        if n:
            k = PACE_PRIOR_PLAYS * (e / n)
            prof.pace = (k + s) / (k + e)
            prof.plays = n
        s, e, n = milk.get(who, (0.0, 0.0, 0))
        if n:
            k = MILK_PRIOR_PLAYS * (e / n)
            prof.milk = (k + s) / (k + e)
            prof.milk_plays = n
        resid, info, n = agg.get(who, (0.0, 0.0, 0))
        if n:
            # One Newton step from 0 on the log-odds shift, shrunk.
            prof.aggression = resid / (info + AGGRESSION_PRIOR)
            prof.fourth_downs = n
        book.players[who] = prof
    return book


class Pace:
    """This match's pace so far, against what the players were expected to
    use: the running ratio of clock used to clock expected, shrunk to the
    players' own pace by `prior_seconds` of evidence.

    `ratio()` > 1: slower than the players' profiles said -> fewer drives
    to come than the pre-match price assumed.
    """

    def __init__(self, book, home_handle, away_handle, prior_seconds=600.0):
        self.book = book
        self.prior = {"TEAM_A": book.profile(home_handle).pace,
                      "TEAM_B": book.profile(away_handle).pace}
        self.prior_seconds = prior_seconds
        self.used = 0.0
        self.expected = 0.0

    def add(self, offense, sit, seconds):
        self.used += seconds
        self.expected += self.book.expected_seconds(sit) * self.prior.get(offense, 1.0)

    def ratio(self):
        """Observed over profile-expected, shrunk to 1."""
        k = self.prior_seconds
        return (k + self.used) / (k + self.expected)

    def profile_pace(self):
        return 0.5 * (self.prior["TEAM_A"] + self.prior["TEAM_B"])
