"""The trader: a local web page to test a model version by hand. Set up a match (the players, teams
and stream, and the pre-match model's expected points, which can be overwritten), then click through
the game a play at a time; after every play the version prices the state it leaves.

    python -m eAMFModel trader --model v10_model

The version is the build's own (v10tables.npz: v10), so a build of any later version copied from
v10 runs as it is. A version must keep what INTERFACE lists; the tests check every version does.

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


def versions():
    """Every model version with a stream, oldest first: v8, v9, v10, ..."""
    here = os.path.dirname(os.path.abspath(__file__))
    names = [f[:-len("_stream.py")] for f in os.listdir(here) if re.fullmatch(r"v\d+_stream\.py", f)]
    return sorted(names, key=lambda n: int(n[1:]))


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
        self.pre = self.m.prematch_model(tables_path)
        self.variant = self.m.Variant(self.version)
        # v10 on: the prior's points are taken net of the players' pace
        self.prior_takes_pace = len(inspect.signature(self.m.prior_theta).parameters) >= 3
        self.paths, self.margin = paths, margin
        self.lock = threading.Lock()

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
        means = tuple(float(x) for x in setup["means"])
        if self.prior_takes_pace:
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
          version=None):
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
