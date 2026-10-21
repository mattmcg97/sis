"""The trader: a local web page to test a model version by hand. Set up a match (the players, teams
and stream, and the pre-match model's expected points, which can be overwritten), then click through
the game a play at a time; after every play the version prices the state it leaves.

    python -m eAMFModel trader --model v14_model
    python -m eAMFModel trader --model v21_t10 --prior eAMFCalibrator/out/form_prior/priors/avg_form4_w05.csv

The version is the build's own (v14tables.npz: v14; v21model.pkl: v2.1, V21Trader). A version must keep
what INTERFACE lists; the tests check every version does. v2.1 prices as the betting sim's v21-drives
does: its offense nudged toward the prior's expected points and one tempo toward the match's expected
drives at the kick-off, the two carried into every state; a match picked off the prior table brings its
own points and drives.

The server keeps nothing between requests: the page holds the game and its history, and sends
the state with every request."""

import csv
import datetime as dt
import importlib
import inspect
import json
import os
import re
import threading
import webbrowser
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import game as gm, players

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trader.html")
DEFAULT_PATHS = 4000
DEFAULT_MARGIN = 0.05
MARGIN_SHOWN = 35                 # final margins shown either side of level
TOTAL_SHOWN = 75                  # points shown above the board
SPREAD_LADDER = 3                 # lines either side of the quoted one
TOTAL_LADDER = 4


# What the trader prices with, in each version (vN.py) and its stream (vN_stream.py).
INTERFACE = ("MARGIN_MAX", "PriorGrid", "Variant", "prior_theta", "price_states", "price_kickoff",
             "market_prob", "key_line", "even_line", "players_book", "prematch_model", "match_seed")
STREAM_INTERFACE = ("model_paths", "sim")
NOT_TRADED = {"v21"}             # the v2.1 snap sim: none of the trader's tables, priced by V21Trader
MATCHES_SHOWN = 600              # v2.1: the prior table's latest matches offered to pick from


def versions():
    """Every model version with a stream the trader prices with, oldest first (v14 now; v21 is priced apart)."""
    here = os.path.dirname(os.path.abspath(__file__))
    names = [f[:-len("_stream.py")] for f in os.listdir(here) if re.fullmatch(r"v\d+_stream\.py", f)]
    return sorted((n for n in names if n not in NOT_TRADED), key=lambda n: int(n[1:]))


def load_version(version):
    """(the version's module, its stream module); stops if it lacks anything the trader uses."""
    if version not in versions():
        raise SystemExit(f"no model version {version!r} (there are {', '.join(versions())})")
    model = importlib.import_module(f"{__package__}.{version}")
    stream = importlib.import_module(f"{__package__}.{version}_stream")
    missing = ([n for n in INTERFACE if not hasattr(model, n)]
               + [f"{version}_stream.{n}" for n in STREAM_INTERFACE if not hasattr(stream, n)])
    if missing:
        raise SystemExit(f"{version} lacks what the trader prices with: {', '.join(missing)}")
    return model, stream


def version_of(model_dir):
    """The version a build directory holds (its <version>tables.npz), newest first; the newest
    version when no directory is given."""
    for v in reversed(versions()):
        if model_dir is None or os.path.exists(os.path.join(model_dir, f"{v}tables.npz")):
            return v
    raise SystemExit(f"{model_dir} holds no model build (no {', '.join(v + 'tables.npz' for v in versions())})")


def odds(p, margin):
    """Decimal odds with the book's margin taken proportionally off a fair probability."""
    p = min(0.999, max(0.001, float(p)))
    return round(1.0 / (p * (1.0 + margin)), 2)


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


class Trader:
    """A model build, loaded once, that prices any game state."""

    def __init__(self, model_dir=None, paths=DEFAULT_PATHS, margin=DEFAULT_MARGIN, version=None):
        self.version = version or version_of(model_dir)
        self.m, stream = load_version(self.version)
        tables_path, grid_path = stream.model_paths(model_dir)
        self.dir = os.path.dirname(tables_path)
        self.tables = stream.sim.Tables.load(tables_path)
        self.grid = self.m.PriorGrid.load(grid_path)
        self.book = self.m.players_book(tables_path)
        self.big = self.m.big_ratings(tables_path) if hasattr(self.m, "big_ratings") else None   # v13
        self.pre = self.m.prematch_model(tables_path)
        self.variant = self.m.Variant(self.version)
        # v10 on: the prior's points are taken net of the players' pace
        self.prior_takes_pace = len(inspect.signature(self.m.prior_theta).parameters) >= 3
        self.paths, self.margin = paths, margin
        self.lock = threading.Lock()
        self._held = {}                                 # v14's held strengths by setup

    # -- what the page offers to choose from ---------------------------------------------------

    def prior_fit(self):
        """The pre-match fit pricing a match today: the build's own, or with a rolling prior attached
        (`eAMFModel prior-daily`) its latest day's fit."""
        inner = getattr(self.pre, "pre", self.pre)                     # under the prior shrink
        return inner.fit_for(dt.date.today()) if hasattr(inner, "fit_for") else inner

    def meta(self):
        """Players, teams and streams the pre-match model knows, its league average and the
        build's settings."""
        fit = self.prior_fit() if self.pre is not None else None
        prior_dir = getattr(fit, "directory", None)
        board_name = "NB2_player_leaderboardsept_team_joint.csv"
        nb2 = next((d for d in (prior_dir, os.path.join(self.dir, "nb2"))
                    if d and os.path.exists(os.path.join(d, board_name))), os.path.join(self.dir, "nb2"))
        board = _read_csv(os.path.join(nb2, board_name))
        names = [r["Player"] for r in board if r.get("Player")]
        known = set(names)
        names += sorted(h for h in (self.book.players if self.book else {}) if h not in known)
        teams = [r["Team"] for r in _read_csv(os.path.join(nb2, "NB2_team_ratingssept_joint.csv"))
                 if r.get("Team")]
        streams = [r["Stream"] for r in _read_csv(os.path.join(nb2, "NB2_stream_effectssept_team_joint.csv"))
                   if r.get("Stream")]
        league = list(self.pre.league) if self.pre is not None else [17.0, 17.0]
        built = (getattr(fit, "meta", None) or {}).get("before")
        inner = getattr(self.pre, "pre", self.pre)
        kind = {"nb2": "NB2"}.get(getattr(inner, "kind", None), getattr(inner, "kind", None)) or ("glmer" if fit is not None and "glmer" in type(fit).__module__ else "NB2")
        prior = None if fit is None else (
            f"{kind} refitted daily, latest fit before {str(built)[:10]}" if hasattr(inner, "fit_for")
            else f"{kind} fitted before {str(built)[:10]}")
        return dict(players=names, teams=sorted(teams), streams=streams or ["1", "2"],
                    league=[round(x, 2) for x in league], prematch=self.pre is not None,
                    built_before=built, prior=prior, model=self.dir, version=self.version,
                    paths=self.paths, margin=self.margin,
                    quarter=gm.QUARTER)

    def profile(self, handle):
        p = self.book.profile(handle) if self.book else players.Profile()
        return dict(plays=p.plays, pace=round(p.pace, 3), aggression=round(p.aggression, 3),
                    kick=round(p.kick, 3))

    def prematch(self, setup):
        """The pre-match model's expected (home, away) points for the match, and each player's
        profile."""
        out = dict(home_profile=self.profile(setup.get("home_player")),
                   away_profile=self.profile(setup.get("away_player")))
        if self.pre is None:
            out["error"] = "No pre-match model in this build"
            out["means"] = [17.0, 17.0]
            return out
        row = {"MATCH_CODE": "TRADER", "SPORT_CODE": "AF",
               "STREAM_NUMBER": str(setup.get("stream") or ""),
               "SCHEDULED_START_TIME_UTC": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
               "PLAYER_1_HANDLE": (setup.get("home_player") or "").upper(),
               "PLAYER_1_TEAM": setup.get("home_team") or "",
               "PLAYER_2_HANDLE": (setup.get("away_player") or "").upper(),
               "PLAYER_2_TEAM": setup.get("away_team") or ""}
        try:
            means = self.pre.means([row])["TRADER"]
        except (Exception, SystemExit) as e:          # pandas missing, an unknown player...
            out["error"] = f"Pre-match model failed: {e}"
            means = self.pre.league
        out["means"] = [round(float(m), 2) for m in means]
        out["league"] = list(self.pre.league)
        if tuple(out["means"]) == tuple(round(float(m), 2) for m in self.pre.league) and "error" not in out:
            out["note"] = "Unknown pairing: league average"
        return out

    # -- pricing ---------------------------------------------------------------------------

    def _books(self, setup, game):
        """The version's final margin and total distributions from this state (from the kick-off, with
        either side receiving, when `game` is None)."""
        home, away = setup.get("home_player"), setup.get("away_player")
        prof = ((self.book.profile(home), self.book.profile(away)) if self.book
                else (players.Profile(), players.Profile()))
        if self.big and home and away:
            prof = self.m.with_big(prof, self.big, (home.upper(), away.upper()),
                                   setup.get("home_team") or "", setup.get("away_team") or "")
        means = tuple(float(x) for x in setup["means"])
        if hasattr(self.m, "held_theta"):
            # v14: the match's strengths hold the prior's points with its sides' own rates, as the
            # stream prices it; the same for every state of a setup, so worked out once
            key = (home, away, means, setup.get("home_team"), setup.get("away_team"))
            if key not in self._held:
                self._held[key] = self.m.held_theta(self.tables, self.grid, means, prof, self.variant,
                                                    seed=self.m.match_seed(f"{home}|{away}|{means}"))
            theta0 = self._held[key]
        elif self.prior_takes_pace:
            theta0 = self.m.prior_theta(self.grid, means,
                                        prof if getattr(self.variant, "pace", False) else None)
        else:
            theta0 = self.m.prior_theta(self.grid, means)
        paths = int(setup.get("paths") or self.paths)
        seed = self.m.match_seed(f"{home}|{away}|{means}")          # the same paths every state
        rng = np.random.default_rng(seed)
        with self.lock:
            if game is None:
                return self.m.price_kickoff(self.tables, theta0, self.variant, prof, paths, rng, seed=seed)
            return self.m.price_states(self.tables, theta0, self.variant, [], True,
                                    [game.model_state()], [1], prof, paths, rng, seed=seed)[0]

    def price(self, setup, game=None, kept=None):
        """Every market at the state (pre-match with `game` None), at the version's own key-number lines
        (held from `kept`, the lines quoted before), with ladders, the expected score and the
        distributions."""
        margin = float(setup.get("margin", self.margin))
        if game is not None and game.phase == gm.FINAL:
            return self._settled(game)
        mpmf, tpmf = self._books(setup, game)
        kept = kept or {}
        prob = lambda market, line: float(self.m.market_prob(market, line, mpmf, tpmf))
        side = lambda p: dict(p=round(p, 4), odds=odds(p, margin))
        h = self.m.key_line(mpmf, self.m.MARGIN_MAX, kept.get("spread"))
        t = self.m.key_line(tpmf, 0, kept.get("total"))
        ml_home = prob(50, 0.0)
        x_m = np.arange(len(mpmf)) - self.m.MARGIN_MAX
        x_t = np.arange(len(tpmf))
        mean_margin, mean_total = float((mpmf * x_m).sum()), float((tpmf * x_t).sum())
        board = 0 if game is None else game.home + game.away
        lo = max(0, board)
        return dict(
            moneyline=dict(home=side(ml_home), away=side(1 - ml_home)),
            spread=dict(line=h, even=self.m.even_line(mpmf, self.m.MARGIN_MAX),
                        home=side(prob(52, h)), away=side(prob(53, -h)),
                        ladder=[dict(line=h + d, home=round(prob(52, h + d), 4))
                                for d in range(-SPREAD_LADDER, SPREAD_LADDER + 1)]),
            total=dict(line=t, even=self.m.even_line(tpmf, 0), over=side(prob(54, t)),
                       under=side(prob(55, t)),
                       ladder=[dict(line=t + d, over=round(prob(54, t + d), 4))
                               for d in range(-TOTAL_LADDER, TOTAL_LADDER + 1)]),
            expected=dict(home=round((mean_total + mean_margin) / 2, 2),
                          away=round((mean_total - mean_margin) / 2, 2),
                          total=round(mean_total, 2), margin=round(mean_margin, 2)),
            margin_dist=dict(start=-MARGIN_SHOWN, p=[round(float(p), 5) for p in
                                                     mpmf[self.m.MARGIN_MAX - MARGIN_SHOWN:
                                                          self.m.MARGIN_MAX + MARGIN_SHOWN + 1]]),
            total_dist=dict(start=lo, p=[round(float(p), 5) for p in tpmf[lo:lo + TOTAL_SHOWN + 1]]),
            kept=dict(spread=h, total=t), book_margin=margin)

    def _settled(self, game):
        m, t = game.home - game.away, game.home + game.away
        win = 1.0 if m > 0 else 0.0 if m < 0 else 0.5
        return dict(final=True, moneyline=dict(home=dict(p=win, odds=None), away=dict(p=1 - win, odds=None)),
                    expected=dict(home=game.home, away=game.away, total=t, margin=m))


class V21Trader(Trader):
    """v2.1 in the trader, as the betting sim prices it (v21_stream._kickoff and match_books): the build's
    snap sim, each side's offense nudged NUDGE_SHARE of the way to the prior's expected points and (v21-drives)
    one tempo on both sides' time tilts so the kick-off games' drives meet the match's expected drives, solved
    once a setup at the kick-off and carried into every state. The expected points and drives come off the
    prior table (a match picked off it, else the pair's latest meeting in it), or a build prior's own model, or
    typed in. The sim's final scores are read as markets.py's margin and total distributions and priced there."""

    def __init__(self, model_dir=None, paths=DEFAULT_PATHS, margin=DEFAULT_MARGIN, prior=None, nudge_share=1.0,
                 drives_share=1.0, pace=True):
        from . import markets, table_prior, v21, v21_stream
        self.v21, self.m = v21, markets
        model_file = v21_stream._model_file(model_dir)
        self.model = v21.Model.load(model_file)
        self.dir = os.path.dirname(model_file)
        self.prior = prior or v21_stream.default_prior()
        tables = v21_stream._prior_tables(self.prior) if self.prior else None
        self.is_table = table_prior.is_table(tables)
        self.pre = None if self.is_table else v21_stream.prematch_model(tables)
        self.rows = _read_csv(tables) if self.is_table else []
        self.by_code = {(r.get("MATCH_CODE") or "").strip(): r for r in self.rows}
        self.nudge_share, self.drives_share = float(nudge_share), float(drives_share)
        self.nudging = self.nudge_share > 0 and getattr(self.model, "nudge_dir", None) is not None
        self.pacing = bool(pace) and self.drives_share > 0 and v21.can_pace(self.model)
        self.version = "v21-drives" if self.pacing else "v21"
        # the pace as v21_stream reads it: each match's drives against its day's league, at the table's average
        # league, over the table's mean (drives_table); `adj` puts a typed match's (exp / league) on that scale
        self.drives, self.level, self.adj = {}, None, 1.0
        if self.pacing and self.is_table and any(r.get("exp_drives") for r in self.rows):
            self.drives = v21_stream.drives_table(tables)
            self.level = float(np.mean(list(self.drives.values())))
            leagues = [float(r["league_drives"]) for r in self.rows if r.get("league_drives") and r.get("exp_drives")]
            self.adj = float(np.mean(leagues)) / self.level if leagues else 1.0
        if self.pacing:
            self.sim_level = v21.drive_level(self.model)[0]          # the build's drives a game (worked out once)
        self.book, self.big = None, None
        self.paths, self.margin = paths, margin
        self.lock = threading.Lock()
        self._kick = {}

    # -- what the page offers to choose from ---------------------------------------------------

    def meta(self):
        pts = [(float(r["home_pts"]), float(r["away_pts"])) for r in self.rows if r.get("home_pts") and r.get("away_pts")]
        league = ([round(float(np.mean([p[0] for p in pts])), 2), round(float(np.mean([p[1] for p in pts])), 2)]
                  if pts else list(getattr(self.pre, "league", (17.0, 17.0))))
        latest = sorted((r for r in self.rows if r.get("MATCH_CODE")), key=lambda r: r.get("start") or "",
                        reverse=True)[:MATCHES_SHOWN]
        matches = [dict(code=r["MATCH_CODE"], label=f"{(r.get('start') or '')[:16]}  {r.get('away', '')} @ {r.get('home', '')}",
                        home=r.get("home", ""), away=r.get("away", ""), home_team=r.get("home_team", ""),
                        away_team=r.get("away_team", "")) for r in latest]
        if self.is_table:
            last = max((r.get("start") or "" for r in self.rows), default="")[:10]
            prior = f"{os.path.basename(self.prior)}, {len(self.rows):,} matches to {last}" + (
                f"; pace on {len(self.drives):,}" if self.drives else ("; no pace in it" if self.pacing else ""))
        else:
            prior = None if self.pre is None else f"{self.prior}'s pre-match model"
        how = (f"nudged {self.nudge_share:.0%} to the points" if self.nudging else "its own means") + (
            f", {self.drives_share:.0%} to the drives" if self.pacing else "")
        return dict(players=sorted(self.model.gamers), teams=sorted(self.model.teams), streams=[], league=league,
                    prematch=bool(self.rows) or self.pre is not None, built_before=None, prior=prior, model=self.dir,
                    version=self.version, how=how, paths=self.paths, margin=self.margin, quarter=gm.QUARTER,
                    matches=matches, pace=self.pacing)

    def profile(self, handle):
        return {}

    def _latest(self, home, away):
        """(the row, swapped) of the pair's latest meeting in the table, at the same ends first."""
        home, away = (home or "").strip().upper(), (away or "").strip().upper()
        same = [r for r in self.rows if r.get("home", "").upper() == home and r.get("away", "").upper() == away]
        flip = [r for r in self.rows if r.get("home", "").upper() == away and r.get("away", "").upper() == home]
        for rows, swapped in ((same, False), (flip, True)):
            if rows:
                return max(rows, key=lambda r: r.get("start") or ""), swapped
        return None, False

    def prematch(self, setup):
        """The match's expected (home, away) points and drives: a match picked off the table, else the pair's
        latest meeting in it, else the build prior's model, else the league's."""
        out = {}
        code = (setup.get("match") or "").strip()
        row, swapped = (self.by_code.get(code), False) if code in self.by_code else \
            self._latest(setup.get("home_player"), setup.get("away_player"))
        if row is not None and row.get("home_pts") and row.get("away_pts"):
            h, a = float(row["home_pts"]), float(row["away_pts"])
            out["means"] = [round(a, 2), round(h, 2)] if swapped else [round(h, 2), round(a, 2)]
            if code in self.by_code:
                out.update(match=code, home_player=row.get("home", ""), away_player=row.get("away", ""),
                           home_team=row.get("home_team", ""), away_team=row.get("away_team", ""))
                out["note"] = f"{code}, {(row.get('start') or '')[:16]}: the prior's own"
            else:
                out["note"] = (f"Their latest meeting in the prior, {row.get('MATCH_CODE')} ({(row.get('start') or '')[:10]})"
                               + (", the ends swapped" if swapped else ""))
            if self.pacing and row.get("exp_drives"):
                out["exp_drives"] = round(float(row["exp_drives"]), 2)
                out["league_drives"] = round(float(row.get("league_drives") or self.level or row["exp_drives"]), 2)
            return out
        meta = self.meta()
        if self.pre is not None:
            r = {"MATCH_CODE": "TRADER", "SPORT_CODE": "AF", "STREAM_NUMBER": str(setup.get("stream") or ""),
                 "SCHEDULED_START_TIME_UTC": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                 "PLAYER_1_HANDLE": (setup.get("home_player") or "").upper(), "PLAYER_1_TEAM": setup.get("home_team") or "",
                 "PLAYER_2_HANDLE": (setup.get("away_player") or "").upper(), "PLAYER_2_TEAM": setup.get("away_team") or ""}
            try:
                out["means"] = [round(float(x), 2) for x in self.pre.means([r])["TRADER"]]
                return out
            except (Exception, SystemExit) as e:
                out["error"] = f"Pre-match model failed: {e}"
        out["means"] = meta["league"]
        out.setdefault("note", "Not in the prior: the league's average (type the expected points"
                       + (" and drives" if self.pacing else "") + ")")
        return out

    # -- pricing ---------------------------------------------------------------------------

    def _pace(self, setup):
        """The match's expected drives over the table's mean (v21_stream's pace), or None: no pace nudge."""
        if not self.pacing:
            return None
        code = (setup.get("match") or "").strip()
        exp = setup.get("exp_drives")
        try:
            exp = float(exp)
        except (TypeError, ValueError):
            return None
        if not exp > 0:
            return None
        if code in self.drives and code in self.by_code and \
                abs(float(self.by_code[code].get("exp_drives") or 0) - exp) < 0.005:
            return self.drives[code] / self.level                    # the table's match as the sim reads it
        try:
            league = float(setup.get("league_drives"))
        except (TypeError, ValueError):
            league = None
        if league and league > 0:
            return exp / league * self.adj
        return exp / (self.level or self.sim_level)

    def _kickoff(self, setup):
        """The kick-off's books, nudge and tempo for a setup (each side receiving), solved once."""
        v21 = self.v21
        pair = tuple((setup.get(k) or "").strip().upper() for k in ("home_player", "away_player"))
        teams = tuple((setup.get(k) or "").strip() for k in ("home_team", "away_team"))
        means = tuple(round(float(x), 4) for x in setup["means"])
        rel = self._pace(setup)
        paths = int(setup.get("paths") or self.paths)
        key = (pair, teams, means, None if rel is None else round(rel, 6), paths)
        if key in self._kick:
            return self._kick[key]
        seed = zlib.crc32(f"{pair}|{teams}|{means}".encode())
        start = v21.kickoff_start(self.model, pair, teams)
        target = means if self.nudging else None
        info = dict(nudge=None, tempo=None)
        with self.lock:
            if rel is not None:
                goal = rel * self.sim_level
                (h, a), alpha, tempo, (own, got) = v21.drives_solve(self.model, start, target, goal, paths, seed=seed,
                                                                    share=self.nudge_share,
                                                                    drives_share=self.drives_share)
                info.update(tempo=round(float(tempo), 3), own_drives=round(float(own), 2),
                            goal_drives=round(float(own + self.drives_share * (goal - own)), 2),
                            drives=round(float(got), 2), pace=round(float(rel), 3))
            elif target is not None:
                (h, a), alpha = v21.nudge_solve(self.model, start, target, paths, seed=seed, share=self.nudge_share)
                tempo = None
            else:
                h, a = v21.simulate(self.model, start, paths, seed=seed)
                alpha = tempo = None
        alpha = alpha if target is not None else None
        if alpha is not None:
            info["nudge"] = [round(float(x), 3) for x in np.asarray(alpha, float)]
        out = dict(books=self.m._distributions(np.ravel(h), np.ravel(a)), alpha=alpha, tempo=tempo, seed=seed,
                   pair=pair, teams=teams, info=info)
        self._kick[key] = out
        return out

    def start_of(self, game, pair, teams):
        """The game's state as a v2.1 start of one (the scorer's try pending at a conversion)."""
        v21 = self.v21
        side = lambda s: 0 if s == gm.HOME else 1
        last = {"fresh": v21.FRESH, "nothing": v21.NOTHING, "short": v21.SHORT, "first": v21.FIRST,
                "loss": v21.LOSS, "timeout": v21.TIMEOUT}
        st = dict(period=[game.period], clock=[float(game.clock)], home=[game.home], away=[game.away],
                  tos=[[game.timeouts_home, game.timeouts_away]], kick2=[1 - side(game.opening_receiver)],
                  elapsed=[int(getattr(game, "snaps", 0) or 0)], down=[1], dist=[10], y=[25], prev=[v21.FRESH],
                  phase=[v21.KICK], team=[side(game.side)])
        if game.phase == gm.SCRIMMAGE:
            st.update(phase=[v21.SNAP], down=[game.down], dist=[max(1, min(game.distance, 100 - game.field))],
                      y=[game.field],
                      prev=[v21.FRESH if game.fresh else last.get(getattr(game, "last", "short"), v21.SHORT)])
        elif game.phase == gm.CONVERSION:                       # the scorer's try, then its kick-off
            st.update(team=[1 - side(game.side)], convert=[side(game.side)])
        out = {k: np.array(v) for k, v in st.items()}
        out.update(v21._start(self.model, 1, pair, teams))
        return out

    def _books(self, setup, game):
        kick = self._kickoff(setup)
        if game is None:
            return kick["books"]
        S = self.start_of(game, kick["pair"], kick["teams"])
        if kick["alpha"] is not None:
            S["nudge"] = np.asarray(kick["alpha"], float)[None, :]
        if kick["tempo"] is not None:
            S["tempo"] = np.array([float(kick["tempo"])])
        paths = int(setup.get("paths") or self.paths)
        with self.lock:
            h, a = self.v21.simulate(self.model, S, paths, seed=kick["seed"] + 1)
        return self.m._distributions(h[0], a[0])

    def price(self, setup, game=None, kept=None):
        out = super().price(setup, game, kept)
        if not out.get("final"):
            out["v21"] = self._kickoff(setup)["info"]
        return out


def is_v21(model_dir=None, version=None):
    """Whether the trader prices v2.1: asked for by name, or the build directory holds v2.1's model."""
    if version:
        return str(version).lower().startswith("v21")
    from . import v21
    return bool(model_dir) and os.path.exists(os.path.join(model_dir, v21.MODEL_FILE))


def respond(trader, path, body):
    """The API: (status, JSON-able reply) for a request."""
    if path == "/api/meta":
        return 200, trader.meta()
    if path == "/api/prematch":
        return 200, trader.prematch(body.get("setup") or {})
    setup = body.get("setup") or {}
    if "means" not in setup:
        return 400, {"error": "set the expected points first"}
    if path == "/api/start":
        g = gm.Game.start(setup.get("opening_receiver") or gm.HOME)
        pre = trader.price(setup)
        return 200, dict(game=g.to_dict(), text=g.describe(), prematch=pre,
                         price=trader.price(setup, g, pre["kept"]))
    if path in ("/api/play", "/api/price"):
        g = gm.Game.from_dict(body.get("game"))
        what = None
        if path == "/api/play":
            g, what = gm.apply(g, body.get("play") or {})
        return 200, dict(game=g.to_dict(), text=g.describe(), what=what,
                         price=trader.price(setup, g, body.get("kept")))
    return 404, {"error": f"no such call {path}"}


def handler_for(trader):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload, kind="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as fh:
                    return self._send(200, fh.read(), "text/html; charset=utf-8")
            if self.path == "/api/meta":
                return self._send(*respond(trader, self.path, {}))
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                self._send(*respond(trader, self.path, body))
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except Exception as e:                        # keep the server up; show the page why
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def serve(model_dir=None, port=8765, paths=DEFAULT_PATHS, margin=DEFAULT_MARGIN, browser=True,
          version=None, prior=None, nudge_share=1.0, drives_share=1.0, pace=True):
    if is_v21(model_dir, version):
        trader = V21Trader(model_dir, paths, margin, prior=prior, nudge_share=nudge_share,
                           drives_share=drives_share, pace=pace and version != "v21")
    else:
        trader = Trader(model_dir, paths, margin, version)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(trader))
    url = f"http://127.0.0.1:{port}/"
    print(f"  {trader.version} trader on {url} (model {trader.dir}, {paths:,} games a price); Ctrl+C to stop",
          flush=True)
    if browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
