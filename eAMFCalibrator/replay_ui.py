"""The v21 replay: a local web page to step through a historic match a PLAY_OVER at a time, with
v2.1 pricing every state it can start from and prod's quote at that message beside it, and the
nudge -- the expected points v21 is pulled toward at the kick-off -- set by hand.

    python -m eAMFCalibrator replay-ui --v21-model v21_model --v21-prior-model eAMFCalibrator/out/walkthrough/priors/avg.csv
    python -m eAMFCalibrator replay-ui --export eAMFCalibrator/out/scouting_playover.csv --history AMFELO_20261013.csv

The match: pick a day and one of its matches (off the history), or give a match code. Its PLAY_OVER
snapshots are built as `scouting` exports them (snowflake_io._play_over_snapshots: the play feed
with each PLAY_OVER's prod quote, prod's last pre-match quote, the timeouts, who kicks off), from
Snowflake -- every answer kept in the replay's query cache, so a match read once costs nothing
again -- or, with --export, from a `scouting` export file (no Snowflake at all).

The prices (v21_stream's): the kick-off, each side receiving, with each side's offense nudged
along the build's strength direction so the kick-off means sit `share` of the way from the sim's
own to the expected points (v21.nudge_solve); every PLAY_OVER v21 can start from (a snap, a
kick-off after a score: v21_stream.state_starts) played out from there with the same nudge. A
PLAY_OVER with a conversion still to come has no v21 state, as in the price stream. Each price
reads the moneyline, the spread and the total at prod's lines (where prod quoted one) and at
v21's own key-number lines, the expected score and the final margin and total distributions.

The expected points start as the prior's (--v21-prior-model: a v14 build or a walkthrough
table; the calibrator's default otherwise) and can be changed on the page: the nudge is solved
again and every price follows. A nudge needs a tilt build (its strength direction); without one,
v21 prices off its own means.

The pace nudge (v21-drives, as the betting sim prices it): with the prior table's expected drives
(exp_drives; --v21-drives names another pace table), one tempo on both sides' time tilts is solved
with the points nudge so the kick-off games' drives go `drives share` of the way from the sim's own
to the match's expected drives, taken against the table's mean at the sim's own level
(v21.drives_solve), and carried into every state. The expected drives and the share can be changed
on the page too; blank drives, or --no-pace, price plain v21.
"""

import csv
import datetime as dt
import json
import os
import threading
import time
import webbrowser
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from . import config, snowflake_io
from .bet_totals import _start

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "replay_ui.html")
DEFAULT_PATHS = 1000
CHART_PATHS = 400              # games a state for the whole-match chart
DEFAULT_SHARE = 0.75
DEFAULT_DRIVES_SHARE = 1.0     # the pace nudge: all the way to the match's expected drives (v21-drives)
MARGIN_SHOWN = 35
TOTAL_SHOWN = 90
SPREAD_LADDER = 3
TOTAL_LADDER = 4
MARKETS = (50, 52, 54)


def _f(v):
    try:
        return None if v is None or v == "" else float(v)
    except (TypeError, ValueError):
        return None


def _i(v):
    x = _f(v)
    return None if x is None else int(x)


def _round(x, k=4):
    return None if x is None else round(float(x), k)


# --------------------------------------------------------------------------- where the matches come from

class SnowflakeSource:
    """Matches off the history, each match's PLAY_OVERs and prod's quotes off Snowflake (cached)."""
    kind = "Snowflake"

    def __init__(self, cursor, history_csv=None):
        self.cur = cursor
        self.history_csv = history_csv
        self._history = None
        self._lock = threading.Lock()

    def history(self):
        with self._lock:
            if self._history is None:
                if self.history_csv:
                    from eAMFModel import nb2_prior
                    self._history = nb2_prior.load_history(self.history_csv)
                else:
                    self._history = snowflake_io.fetch_history(self.cur)
            return self._history

    def describe(self):
        return f"Snowflake (cached in {config.FETCH_DISK_CACHE})" if config.FETCH_DISK_CACHE else "Snowflake"

    def snapshots(self, code, start):
        """The match's PLAY_OVER snapshot rows (scouting's export rows)."""
        was = config.CUTOFF_START, config.CUTOFF_END
        # prod's quotes are read inside the window: the match's day and the days either side
        lo = (start or dt.datetime(2026, 1, 1)) - dt.timedelta(days=1)
        config.CUTOFF_START = lo.strftime("%Y-%m-%d %H:%M:%S")
        config.CUTOFF_END = (lo + dt.timedelta(days=3)).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self._lock:
                snaps, _ = snowflake_io._play_over_snapshots(self.cur, [code], with_handles=True)
        finally:
            config.CUTOFF_START, config.CUTOFF_END = was
        return snaps.get(code, [])


class ExportSource:
    """Matches and their PLAY_OVERs off a `scouting` export file (and a history CSV for the gamers,
    teams and kick-offs): no Snowflake."""
    kind = "Export"

    def __init__(self, export_csv, history_csv=None):
        self.export_csv, self.history_csv = export_csv, history_csv
        by = defaultdict(list)
        with open(export_csv, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                by[r["match_code"]].append(r)
        self.rows = {c: sorted(rs, key=lambda r: _i(r["message"]) or 0) for c, rs in by.items()}
        self._history = None

    def history(self):
        if self._history is None:
            if self.history_csv:
                from eAMFModel import nb2_prior
                self._history = nb2_prior.load_history(self.history_csv)
            else:
                self._history = []
        return self._history

    def describe(self):
        return f"the export {self.export_csv} ({len(self.rows):,} matches)"

    def snapshots(self, code, start):
        return self.rows.get(code, [])

    def codes(self):
        return list(self.rows)


# --------------------------------------------------------------------------- the replay

def _side(offense, team_a_side):
    """'home' or 'away' for the play feed's TEAM_A / TEAM_B."""
    if offense not in ("TEAM_A", "TEAM_B"):
        return None
    a_home = (team_a_side or "home") != "away"
    return "home" if (offense == "TEAM_A") == a_home else "away"


def _clock(seconds):
    s = _f(seconds)
    if s is None:
        return ""
    s = max(0, int(round(s)))
    return f"{s // 60}:{s % 60:02d}"


def _prod(row, prefix=""):
    """Prod's quote at a PLAY_OVER (or its pre-match one, prefix 'prematch_'): each market's
    probability, line and whether it was live."""
    out = {}
    for m in MARKETS:
        p = _f(row.get(f"{prefix}prob_{m}"))
        if p is None:
            continue
        out[str(m)] = dict(p=_round(p), line=_f(row.get(f"{prefix}line_{m}")),
                           live=True if prefix else bool(_i(row.get(f"live_{m}"))))
    return out


def timeline_rows(rows, starts):
    """The page's timeline: every PLAY_OVER with what it says and prod's quote there."""
    out = []
    for r in rows:
        msg = _i(r.get("message"))
        side = _side(r.get("offense"), r.get("team_a_side"))
        y = _i(r.get("field_position"))
        out.append(dict(
            message=msg, period=_i(r.get("period")), clock=_clock(r.get("clock_seconds")),
            home=_i(r.get("score_p1")) or 0, away=_i(r.get("score_p2")) or 0, play=r.get("play_kind") or "",
            messages=(r.get("play_messages") or "").replace("|", " "), offense=side, down=_i(r.get("down")),
            distance=_i(r.get("distance")), field=y, bet_state=r.get("bet_state") or "",
            priced=msg in starts, prod=_prod(r)))
    return out


def _dist_slice(pmf, start, stop):
    return [round(float(x), 5) for x in pmf[max(0, start):stop]]


class Replay:
    """The v21 model, loaded once, and the matches loaded so far."""

    def __init__(self, source, model, prior=None, paths=DEFAULT_PATHS, share=DEFAULT_SHARE, model_file=None,
                 prior_name=None, drives=None, drives_share=DEFAULT_DRIVES_SHARE, drives_name=None):
        from eAMFModel import markets, v21, v21_stream
        self.mk, self.v21, self.stream = markets, v21, v21_stream
        self.source, self.model, self.prior = source, model, prior
        self.paths, self.share = paths, share
        self.model_file, self.prior_name = model_file, prior_name
        # the pace nudge: each match's expected drives (v21_stream.drives_table) over their mean, as the stream reads them
        self.drives = drives or {}
        self.pacing = bool(self.drives) and v21.can_pace(model)
        self.level = float(np.mean(list(self.drives.values()))) if self.drives else None
        self.drives_share, self.drives_name = float(drives_share), drives_name
        self.lock = threading.Lock()
        self.matches_ = {}                              # code -> loaded match
        self.nudges = {}                                # (code, means, share, paths, pace, drives share) -> the kick-off

    # -- what the page offers -----------------------------------------------------------------

    def meta(self):
        hist = self.source.history()
        days = sorted({(_start(r.get("SCHEDULED_START_TIME_UTC")) or dt.datetime.min).date().isoformat()
                       for r in hist if _start(r.get("SCHEDULED_START_TIME_UTC"))})
        return dict(model=self.model_file, prior=self.prior_name or (self.prior.describe()
                    if self.prior is not None and hasattr(self.prior, "describe") else None),
                    source=self.source.describe(), source_kind=getattr(self.source, "kind", None), paths=self.paths, share=self.share,
                    nudge=getattr(self.model, "nudge_dir", None) is not None,
                    mode=getattr(self.model, "mode", None), last_day=days[-1] if days else None,
                    first_day=days[0] if days else None, chart_paths=CHART_PATHS, pace=self.pacing,
                    drives_share=self.drives_share, drives_table=self.drives_name if self.pacing else None)

    def matches(self, day):
        """The day's matches off the history (or the export's, with no history)."""
        out = []
        for r in self.source.history():
            t = _start(r.get("SCHEDULED_START_TIME_UTC"))
            if t is None or t.date().isoformat() != day:
                continue
            out.append(dict(code=r["MATCH_CODE"], start=t.strftime("%H:%M"), home=r.get("PLAYER_1_HANDLE"),
                            away=r.get("PLAYER_2_HANDLE"), home_team=r.get("PLAYER_1_TEAM"),
                            away_team=r.get("PLAYER_2_TEAM"), final=[_i(r.get("PLAYER_1_FINAL_SCORE")),
                                                                     _i(r.get("PLAYER_2_FINAL_SCORE"))]))
        if not out and hasattr(self.source, "codes") and not self.source.history():
            out = [dict(code=c, start="", home="", away="", home_team="", away_team="", final=[None, None])
                   for c in self.source.codes()]
        return sorted(out, key=lambda m: (m["start"], m["code"]))

    # -- a match ------------------------------------------------------------------------------

    def _info(self, code):
        return next((r for r in self.source.history() if r.get("MATCH_CODE") == code), {})

    def load(self, code):
        """The match's PLAY_OVERs, the states v21 can price, the prior's expected points."""
        if code in self.matches_:
            return self.matches_[code]["view"]
        info = self._info(code)
        start = _start(info.get("SCHEDULED_START_TIME_UTC"))
        rows = self.source.snapshots(code, start)
        if not rows:
            raise ValueError(f"{code}: no PLAY_OVERs (no play feed for it, or not in the export)")
        rows = sorted(rows, key=lambda r: _i(r.get("message")) or 0)
        text = [self.stream._as_text(r) for r in rows]
        pair, teams = self.v21.match_people(code, text, {code: info} if info else {})
        with self.lock:
            starts = dict(self.stream.state_starts(self.model, text, pair, teams))
        prior, note = None, None
        if self.prior is not None:
            row = dict(MATCH_CODE=code, SPORT_CODE="AF", STREAM_NUMBER=info.get("STREAM_NUMBER") or "",
                       SCHEDULED_START_TIME_UTC=info.get("SCHEDULED_START_TIME_UTC") or "",
                       PLAYER_1_HANDLE=pair[0], PLAYER_1_TEAM=teams[0], PLAYER_2_HANDLE=pair[1],
                       PLAYER_2_TEAM=teams[1])
            try:
                hist = [r for r in self.source.history()
                        if start is None or (_start(r.get("SCHEDULED_START_TIME_UTC")) or start) < start]
                got = self.prior.means([row], results=hist)
                if code in got:
                    prior = [round(float(x), 2) for x in got[code]]
                else:
                    note = "Not in prior"
            except (Exception, SystemExit) as e:
                note = f"Prior failed: {e}"
        r0 = rows[0]
        final = [_i(r0.get("final_p1")), _i(r0.get("final_p2"))]
        if final[0] is None:
            final = [_i(info.get("PLAYER_1_FINAL_SCORE")), _i(info.get("PLAYER_2_FINAL_SCORE"))]
        pace = None
        if self.pacing and code in self.drives:
            pace = dict(exp=round(self.drives[code], 2), pace=round(self.drives[code] / self.level, 3),
                        level=round(self.level, 2))
        view = dict(code=code, home=pair[0], away=pair[1], home_team=teams[0], away_team=teams[1],
                    start=start.strftime("%Y-%m-%d %H:%M") if start else "", final=final, prior=prior,
                    prior_note=note, prematch=_prod(r0, "prematch_"),
                    timeline=timeline_rows(rows, starts), n_priced=len(starts), pace=pace,
                    pace_note=None if pace or not self.pacing else "Not in the pace table: plain v21")
        self.matches_[code] = dict(view=view, starts=starts, pair=pair, teams=teams)
        return view

    def _match(self, code):
        if code not in self.matches_:
            self.load(code)
        return self.matches_[code]

    # -- the nudge and the prices ---------------------------------------------------------------

    def pace_of(self, code, exp=None):
        """The match's expected drives over the table's mean (the stream's pace), `exp` typed on the page
        else the table's; None: no pace nudge."""
        if not self.pacing:
            return None
        own = self.drives.get(code)
        if exp is None or (own is not None and abs(round(own, 2) - exp) < 0.005):
            exp = own                                   # the table's, to the last digit (the page shows it rounded)
        return None if exp is None or not exp > 0 else float(exp) / self.level

    def nudge(self, code, means, share, paths, exp=None, drives_share=None):
        """(alpha or None, the kick-off books, the sim's own kick-off means, the nudged ones, the tempo or None,
        the pace's drives or None), solved once a (match, expected points, share, paths, pace, drives share)."""
        m = self._match(code)
        rel = self.pace_of(code, exp)
        dshare = self.drives_share if drives_share is None else float(drives_share)
        pacing = rel is not None and dshare > 0
        key = (code, None if means is None else tuple(round(float(x), 2) for x in means), round(float(share), 3),
               int(paths), round(rel, 6) if pacing else None, round(dshare, 3) if pacing else None)
        if key not in self.nudges:
            start = self.v21.kickoff_start(self.model, m["pair"], m["teams"])
            seed = self.stream._seed(code)
            nudging = means is not None and share > 0 and getattr(self.model, "nudge_dir", None) is not None
            tempo = pace = None
            with self.lock:
                h0, a0 = self.v21.simulate(self.model, start, int(paths), seed=seed)
                own = [float(h0.mean()), float(a0.mean())]
                if pacing:
                    goal = rel * self.v21.drive_level(self.model)[0]
                    (h, a), alpha, tempo, (own_d, got) = self.v21.drives_solve(
                        self.model, start, means if nudging else None, goal, int(paths), seed=seed, share=share,
                        drives_share=dshare)
                    alpha = np.asarray(alpha, float) if nudging else None
                    pace = dict(pace=round(rel, 3), own=round(float(own_d), 2),
                                goal=round(float(own_d + dshare * (goal - own_d)), 2), got=round(float(got), 2),
                                tempo=round(float(tempo), 3), share=dshare)
                elif nudging:
                    (h, a), alpha = self.v21.nudge_solve(self.model, start, means, int(paths), seed=seed, share=share)
                    alpha = np.asarray(alpha, float)
                else:
                    h, a, alpha = h0, a0, None
            books = self.mk._distributions(h.ravel(), a.ravel())
            self.nudges[key] = (alpha, books, own, [float(h.mean()), float(a.mean())], tempo, pace)
        return self.nudges[key]

    def _state(self, m, message, alpha, tempo=None):
        st = dict(m["starts"][message])
        if alpha is not None:
            st["nudge"] = np.tile(alpha, (len(st["period"]), 1))
        if tempo is not None:                           # the kick-off's pace nudge, as the stream carries it
            st["tempo"] = np.full(len(st["period"]), float(tempo))
        return st

    def price(self, code, message=None, means=None, share=None, paths=None, exp=None, drives_share=None):
        """Every market at the PLAY_OVER (the kick-off with message None), v21 and prod side by side."""
        share = self.share if share is None else float(share)
        paths = int(paths or self.paths)
        m = self._match(code)
        alpha, kick, own, nudged, tempo, pace = self.nudge(code, means, share, paths, exp, drives_share)
        nudge = dict(alpha=None if alpha is None else [round(float(x), 3) for x in alpha],
                     own=[round(x, 2) for x in own], nudged=[round(x, 2) for x in nudged],
                     target=None if means is None else [round(float(x), 2) for x in means], share=share,
                     available=getattr(self.model, "nudge_dir", None) is not None, pace=pace)
        view = m["view"]
        if message is None:
            prod = view["prematch"]
            mpmf, tpmf = kick
        else:
            message = int(message)
            row = next((t for t in view["timeline"] if t["message"] == message), None)
            if row is None:
                raise ValueError(f"{code}: no PLAY_OVER {message}")
            prod = row["prod"]
            if message not in m["starts"]:
                return dict(priced=False, message=message, prod=prod, nudge=nudge,
                            why="No v21 state")
            with self.lock:
                h, a = self.v21.simulate(self.model, self._state(m, message, alpha, tempo), paths,
                                         seed=self.stream._seed(code) + 1 + message)
            mpmf, tpmf = self.mk._distributions(h[0], a[0])
        out = self.markets(mpmf, tpmf, prod)
        out.update(priced=True, message=message, prod=prod, nudge=nudge, paths=paths)
        return out

    def markets(self, mpmf, tpmf, prod):
        mk = self.mk
        M = mk.MARGIN_MAX
        prob = lambda market, line: float(mk.market_prob(market, line, mpmf, tpmf))
        x_m, x_t = np.arange(len(mpmf)) - M, np.arange(len(tpmf))
        mean_m, mean_t = float((mpmf * x_m).sum()), float((tpmf * x_t).sum())
        h, t = mk.key_line(mpmf, M), mk.key_line(tpmf, 0)
        out = dict(
            moneyline=dict(home=_round(prob(50, 0.0)), away=_round(prob(51, 0.0))),
            spread=dict(line=h, home=_round(prob(52, h)), even=mk.even_line(mpmf, M),
                        ladder=[dict(line=h + d, home=_round(prob(52, h + d)))
                                for d in range(-SPREAD_LADDER, SPREAD_LADDER + 1)]),
            total=dict(line=t, over=_round(prob(54, t)), even=mk.even_line(tpmf, 0),
                       ladder=[dict(line=t + d, over=_round(prob(54, t + d)))
                               for d in range(-TOTAL_LADDER, TOTAL_LADDER + 1)]),
            expected=dict(home=round((mean_t + mean_m) / 2, 2), away=round((mean_t - mean_m) / 2, 2),
                          margin=round(mean_m, 2), total=round(mean_t, 2)),
            margin_dist=dict(start=-MARGIN_SHOWN, p=_dist_slice(mpmf, M - MARGIN_SHOWN, M + MARGIN_SHOWN + 1)),
            total_dist=dict(start=0, p=_dist_slice(tpmf, 0, TOTAL_SHOWN + 1)))
        at = {}                                         # v21 at prod's lines
        for k, q in (prod or {}).items():
            if k == "50":
                at[k] = out["moneyline"]["home"]
            elif q.get("line") is not None:
                at[k] = _round(prob(int(k), q["line"]))
        out["at_prod"] = at
        return out

    def price_all(self, code, means=None, share=None, paths=None, exp=None, drives_share=None):
        """Every PLAY_OVER v21 can price, in one simulation (CHART_PATHS games a state): the home
        side's chance and the expected margin and total, for the chart."""
        share = self.share if share is None else float(share)
        paths = int(paths or CHART_PATHS)
        m = self._match(code)
        alpha, kick, _, _, tempo, _ = self.nudge(code, means, share, paths, exp, drives_share)
        msgs = [t["message"] for t in m["view"]["timeline"] if t["message"] in m["starts"]]
        t0 = time.time()
        out = []
        mp0, tp0 = kick
        M = self.mk.MARGIN_MAX
        out.append(dict(message=None, home=_round(float(self.mk.market_prob(50, 0.0, mp0, tp0))),
                        margin=round(float((mp0 * (np.arange(len(mp0)) - M)).sum()), 2),
                        total=round(float((tp0 * np.arange(len(tp0))).sum()), 2)))
        if msgs:
            S = self.v21.concat_starts([self._state(m, msg, alpha, tempo) for msg in msgs])
            with self.lock:
                h, a = self.v21.simulate(self.model, S, paths, seed=self.stream._seed(code) + 1)
            for k, msg in enumerate(msgs):
                mp_, tp = self.mk._distributions(h[k], a[k])
                out.append(dict(message=msg, home=_round(float(self.mk.market_prob(50, 0.0, mp_, tp))),
                                margin=round(float(h[k].mean() - a[k].mean()), 2),
                                total=round(float(h[k].mean() + a[k].mean()), 2)))
        return dict(points=out, paths=paths, seconds=round(time.time() - t0, 1))


# --------------------------------------------------------------------------- the web side

def respond(replay, path, body):
    """The API: (status, JSON-able reply) for a request."""
    if path == "/api/meta":
        return 200, replay.meta()
    if path == "/api/matches":
        return 200, dict(matches=replay.matches(body.get("day") or ""))
    code = (body.get("code") or "").strip()
    if not code:
        return 400, {"error": "give a match code"}
    if path == "/api/load":
        return 200, replay.load(code)
    means = body.get("means")
    means = None if not means or any(x in (None, "") for x in means) else [float(x) for x in means]
    exp = _f(body.get("exp_drives"))                   # typed on the page; None: the table's
    if path == "/api/price":
        return 200, replay.price(code, body.get("message"), means, body.get("share"), body.get("paths"), exp,
                                 body.get("drives_share"))
    if path == "/api/price_all":
        return 200, replay.price_all(code, means, body.get("share"), body.get("chart_paths"), exp,
                                     body.get("drives_share"))
    return 404, {"error": f"no such call {path}"}


def handler_for(replay):
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
                return self._send(*respond(replay, self.path, {}))
            self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                self._send(*respond(replay, self.path, body))
            except ValueError as e:
                self._send(400, {"error": str(e)})
            except SystemExit as e:
                self._send(400, {"error": str(e)})
            except Exception as e:                        # keep the server up; show the page why
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def build(model_dir=None, prior=None, export=None, history=None, cursor=None, paths=DEFAULT_PATHS,
          share=DEFAULT_SHARE, drives_path=None, drives_share=DEFAULT_DRIVES_SHARE, pace=True):
    """The Replay off the command line's options: with `pace`, the pace nudge off `drives_path` (a pace table),
    else the prior table's own exp_drives (v21_stream.drives_table); plain v21 where neither has them."""
    from eAMFModel import v21, v21_stream
    model_file = v21_stream._model_file(model_dir)
    model = v21.Model.load(model_file)
    tables = v21_stream._prior_tables(prior)
    pre = v21_stream.prematch_model(tables) if tables else None
    source = ExportSource(export, history) if export else SnowflakeSource(cursor, history)
    drives, name = None, None
    if pace and v21.can_pace(model):
        try:
            drives = v21_stream.drives_table(drives_path, tables)
            name = drives_path or os.environ.get("EAMF_V21_DRIVES") or tables
        except SystemExit as e:
            print(f"  no pace nudge (plain v21): {e}", flush=True)
    return Replay(source, model, pre, paths, share, model_file=model_file, prior_name=tables, drives=drives,
                  drives_share=drives_share, drives_name=name)


def serve(replay, port=8766, browser=True):
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_for(replay))
    url = f"http://127.0.0.1:{port}/"
    pace = (f"pace nudge {replay.drives_share:.0%} off {replay.drives_name} ({len(replay.drives):,} matches)"
            if replay.pacing else "no pace nudge")
    print(f"  v21 replay on {url} (model {replay.model_file}; prior {replay.prior_name or 'none'}; {pace}; "
          f"{replay.source.describe()}; {replay.paths:,} games a price); Ctrl+C to stop", flush=True)
    if browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
