"""The trader: a local web page to test v10 by hand. Set up a match (the players, teams and stream,
and the pre-match model's expected points, which can be overwritten), then click through the game a
play at a time; after every play v10 prices the state it leaves.

    python -m eAMFModel trader --model v10_model

The server keeps nothing between requests: the page holds the game and its history, and sends
the state with every request."""

import csv
import datetime as dt
import json
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import game as gm, players, sim10 as sim, v10
from .v10_stream import model_paths

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trader.html")
DEFAULT_PATHS = 4000
DEFAULT_MARGIN = 0.05
MARGIN_SHOWN = 35                 # final margins shown either side of level
TOTAL_SHOWN = 75                  # points shown above the board
SPREAD_LADDER = 3                 # lines either side of the quoted one
TOTAL_LADDER = 4


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
    """A v10 build, loaded once, that prices any game state."""

    def __init__(self, model_dir=None, paths=DEFAULT_PATHS, margin=DEFAULT_MARGIN):
        tables_path, grid_path = model_paths(model_dir)
        self.dir = os.path.dirname(tables_path)
        self.tables = sim.Tables.load(tables_path)
        self.grid = v10.PriorGrid.load(grid_path)
        self.book = v10.players_book(tables_path)
        self.pre = v10.prematch_model(tables_path)
        self.variant = v10.Variant("v10")
        self.paths, self.margin = paths, margin
        self.lock = threading.Lock()

    # -- what the page offers to choose from ---------------------------------------------------

    def meta(self):
        """Players, teams and streams the pre-match model knows, its league average and the
        build's settings."""
        prior_dir = getattr(self.pre, "directory", None) if self.pre is not None else None
        nb2 = prior_dir or os.path.join(self.dir, "nb2")
        board = _read_csv(os.path.join(nb2, "NB2_player_leaderboardsept_team_joint.csv"))
        names = [r["Player"] for r in board if r.get("Player")]
        known = set(names)
        names += sorted(h for h in (self.book.players if self.book else {}) if h not in known)
        teams = [r["Team"] for r in _read_csv(os.path.join(nb2, "NB2_team_ratingssept_joint.csv"))
                 if r.get("Team")]
        streams = [r["Stream"] for r in _read_csv(os.path.join(nb2, "NB2_stream_effectssept_team_joint.csv"))
                   if r.get("Stream")]
        league = list(self.pre.league) if self.pre is not None else [17.0, 17.0]
        built = None
        meta_path = os.path.join(nb2, "level.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                built = json.load(fh).get("before")
        return dict(players=names, teams=sorted(teams), streams=streams or ["1", "2"],
                    league=[round(x, 2) for x in league], prematch=self.pre is not None,
                    built_before=built, model=self.dir, paths=self.paths, margin=self.margin,
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
        """v10's final margin and total distributions from this state (from the kick-off, with
        either side receiving, when `game` is None)."""
        home, away = setup.get("home_player"), setup.get("away_player")
        prof = ((self.book.profile(home), self.book.profile(away)) if self.book
                else (players.Profile(), players.Profile()))
        means = tuple(float(x) for x in setup["means"])
        theta0 = v10.prior_theta(self.grid, means, prof if self.variant.pace else None)
        paths = int(setup.get("paths") or self.paths)
        seed = v10.match_seed(f"{home}|{away}|{means}")          # the same paths every state
        rng = np.random.default_rng(seed)
        with self.lock:
            if game is None:
                return v10.price_kickoff(self.tables, theta0, self.variant, prof, paths, rng, seed=seed)
            return v10.price_states(self.tables, theta0, self.variant, [], True,
                                    [game.model_state()], [1], prof, paths, rng, seed=seed)[0]

    def price(self, setup, game=None, kept=None):
        """Every market at the state (pre-match with `game` None), at v10's own key-number lines
        (held from `kept`, the lines quoted before), with ladders, the expected score and the
        distributions."""
        margin = float(setup.get("margin", self.margin))
        if game is not None and game.phase == gm.FINAL:
            return self._settled(game)
        mpmf, tpmf = self._books(setup, game)
        kept = kept or {}
        prob = lambda market, line: float(v10.market_prob(market, line, mpmf, tpmf))
        side = lambda p: dict(p=round(p, 4), odds=odds(p, margin))
        h = v10.key_line(mpmf, v10.MARGIN_MAX, kept.get("spread"))
        t = v10.key_line(tpmf, 0, kept.get("total"))
        ml_home = prob(50, 0.0)
        x_m = np.arange(len(mpmf)) - v10.MARGIN_MAX
        x_t = np.arange(len(tpmf))
        mean_margin, mean_total = float((mpmf * x_m).sum()), float((tpmf * x_t).sum())
        board = 0 if game is None else game.home + game.away
        lo = max(0, board)
        return dict(
            moneyline=dict(home=side(ml_home), away=side(1 - ml_home)),
            spread=dict(line=h, even=v10.even_line(mpmf, v10.MARGIN_MAX),
                        home=side(prob(52, h)), away=side(prob(53, -h)),
                        ladder=[dict(line=h + d, home=round(prob(52, h + d), 4))
                                for d in range(-SPREAD_LADDER, SPREAD_LADDER + 1)]),
            total=dict(line=t, even=v10.even_line(tpmf, 0), over=side(prob(54, t)),
                       under=side(prob(55, t)),
                       ladder=[dict(line=t + d, over=round(prob(54, t + d), 4))
                               for d in range(-TOTAL_LADDER, TOTAL_LADDER + 1)]),
            expected=dict(home=round((mean_total + mean_margin) / 2, 2),
                          away=round((mean_total - mean_margin) / 2, 2),
                          total=round(mean_total, 2), margin=round(mean_margin, 2)),
            margin_dist=dict(start=-MARGIN_SHOWN, p=[round(float(p), 5) for p in
                                                     mpmf[v10.MARGIN_MAX - MARGIN_SHOWN:
                                                          v10.MARGIN_MAX + MARGIN_SHOWN + 1]]),
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


def serve(model_dir=None, port=8765, paths=DEFAULT_PATHS, margin=DEFAULT_MARGIN, browser=True):
    trader = Trader(model_dir, paths, margin)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(trader))
    url = f"http://127.0.0.1:{port}/"
    print(f"  v10 trader on {url} (model {trader.dir}, {paths:,} games a price); Ctrl+C to stop",
          flush=True)
    if browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
