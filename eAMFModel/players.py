"""Player profiles: each player's pace, 4th-down aggression, clock milking and form on the day."""

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field

from .drive import DriveParams, go_probability

PACE_PRIOR_PLAYS = 70
AGGRESSION_PRIOR = 2.0
MILK_PRIOR_PLAYS = 40

LATE_SECONDS = 120


def margin_bucket(margin):
    """The offense's lead as a small bucket number."""
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
    """(late in a half, margin bucket) for an offense at a snap."""
    half_left = clock_seconds + (240 if period % 2 == 1 else 0)
    return (half_left <= LATE_SECONDS, margin_bucket(margin))


def _f(v):
    """A float, or None for a blank."""
    return float(v) if v not in ("", None) else None


def _i(v):
    """An int, or None for a blank."""
    return int(float(v)) if v not in ("", None) else None


def play_intervals(rows):
    """Seconds between consecutive snaps by the same offense, with the situation of each."""
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
    """Every ordinary 4th down: who had the ball, where, how far, and whether they went for it."""
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
    """One player's tendencies against the league."""
    plays: int = 0
    pace: float = 1.0
    aggression: float = 0.0
    fourth_downs: int = 0
    milk: float = 1.0
    milk_plays: int = 0
    form: float = None


@dataclass
class Book:
    """The league's snap timings and every player's profile."""
    seconds: dict = field(default_factory=dict)
    players: dict = field(default_factory=dict)

    def expected_seconds(self, sit):
        """The league's average seconds between snaps in this situation."""
        return self.seconds.get(f"{int(sit[0])}|{sit[1]}", 23.0)

    def profile(self, handle):
        """A player's profile, or a league-average one if unknown."""
        return self.players.get((handle or "").upper(), Profile())

    def save(self, path):
        """Write the book to JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"seconds": self.seconds,
                       "players": {h: p.__dict__ for h, p in self.players.items()}}, fh, indent=1)

    @classmethod
    def load(cls, path):
        """Read a book from JSON."""
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        return cls(raw["seconds"], {h: Profile(**p) for h, p in raw["players"].items()})


def load_handles(path):
    """Match code to (home handle, away handle) from a CSV."""
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
    """Build every player's profile from the export."""
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

    obs = defaultdict(lambda: [0.0, 0.0, 0])
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
    agg = defaultdict(lambda: [0.0, 0.0, 0])
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
            prof.aggression = resid / (info + AGGRESSION_PRIOR)
            prof.fourth_downs = n
        book.players[who] = prof
    return book


class Pace:
    """This match's pace so far against what the two players were expected to play at."""

    def __init__(self, book, home_handle, away_handle, prior_seconds=600.0):
        """Start with the two players' profiles and no snaps seen."""
        self.book = book
        self.prior = {"TEAM_A": book.profile(home_handle).pace,
                      "TEAM_B": book.profile(away_handle).pace}
        self.prior_seconds = prior_seconds
        self.used = 0.0
        self.expected = 0.0

    def add(self, offense, sit, seconds):
        """Count one more snap interval."""
        self.used += seconds
        self.expected += self.book.expected_seconds(sit) * self.prior.get(offense, 1.0)

    def ratio(self):
        """Observed pace over expected, shrunk toward 1."""
        k = self.prior_seconds
        return (k + self.used) / (k + self.expected)

    def profile_pace(self):
        """The two players' average profile pace."""
        return 0.5 * (self.prior["TEAM_A"] + self.prior["TEAM_B"])
