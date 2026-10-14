"""Pull every eAmerican football stat h2hggl.com shows straight from the API its pages read
(api-h2h.cdn-hudstats.com), into CSVs. No browser needed.

  players.csv       every player's stat line (participant/nfl): matches, wins, win %, streaks,
                    points for and against, touchdowns, red zone, third and fourth down, yards per
                    play ... every field the API gives, as it gives it
  matches.csv       every past match in the players' histories and the site's schedule
                    (schedule/past): start, stream, tournament, both players and teams, the score
  match_stats.csv   each match's team stats (match/stats), one row per match, period and side:
                    first downs, total offence, rushing, passing ... every field
  timelines/        with --timelines, each match's timeline as the API gives it (drives,
                    incidents, the half-time score)
  raw/              every match's response, kept: a rerun only fetches matches it lacks

The match's external_id is its match code in Snowflake (AF028141026), so the stats join straight
onto our matches. Side A and B are the API's teamA and teamB; match the players by name to put
them on our PLAYER_1 (home) and PLAYER_2.

How far back a player's history goes depends on the API: the first player read is used to find
out whether it takes a limit or pages (summary.txt says which it found, and how many matches each
player came back with).

    py analysis/h2hggl_api.py                        # players, histories, every match's stats
    py analysis/h2hggl_api.py --since 2026-09-01     # stats only for matches since then
    py analysis/h2hggl_api.py --players-only         # just players.csv, in seconds
"""

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

BASE = "https://api-h2h.cdn-hudstats.com/v1/"
HEADERS = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0 Safari/537.36"),
           "Origin": "https://h2hggl.com", "Referer": "https://h2hggl.com/", "Accept": "application/json"}
MATCH_FIELDS = ("externalId", "startDate", "isCancelled", "matchStatus", "streamName", "tournamentName",
                "participantAName", "teamAName", "teamAScore", "participantBName", "teamBName", "teamBScore")
PAGERS = (("page", lambda n, k: k + 1), ("offset", lambda n, k: n * k), ("skip", lambda n, k: n * k),
          ("pageNumber", lambda n, k: k + 1))


class Api:
    """GETs against the API: polite (a delay between calls), retried on errors, and, for match
    data, cached on disk."""

    def __init__(self, base, delay, cache, refresh=False):
        import requests
        self.base, self.delay, self.cache, self.refresh = base.rstrip("/") + "/", delay, cache, refresh
        self.http = requests.Session()
        self.http.headers.update(HEADERS)
        self.calls = self.hits = 0
        self.failed = []
        cache.mkdir(parents=True, exist_ok=True)

    def url(self, path, **params):
        params = {k: v for k, v in params.items() if v is not None}
        return self.base + path.lstrip("/") + (f"?{urlencode(params)}" if params else "")

    def get(self, path, cached=False, **params):
        """The parsed JSON at path, or None after four failed tries."""
        url = self.url(path, **params)
        file = self.cache / (hashlib.sha1(url.encode()).hexdigest()[:20] + ".json")
        if cached and file.exists() and not self.refresh:
            self.hits += 1
            return json.loads(file.read_text(encoding="utf-8"))
        last = "no answer"
        for attempt in range(4):
            time.sleep(self.delay if not attempt else 2 ** attempt)
            try:
                self.calls += 1
                r = self.http.get(url, timeout=30)
                if r.status_code == 404:
                    break
                if r.status_code in (429, 500, 502, 503, 504):
                    continue
                r.raise_for_status()
                data = r.json()
                if cached:
                    file.write_text(json.dumps(data), encoding="utf-8")
                return data
            except Exception as exc:                 # network, decoding: try again
                last = exc
                continue
        self.failed.append((url, last))
        return None


def flatten(obj, prefix=""):
    """A nested dict as one level, keys joined with '.'; lists kept as JSON text."""
    out = {}
    for k, v in (obj or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def ids(rows):
    return {r.get("externalId") for r in rows or [] if isinstance(r, dict) and r.get("externalId")}


def find_pager(api, path, params, first, most):
    """How this list goes further back than its first page: (the limit it honours or None, the
    page key that brings older rows or None, that key's value for page k)."""
    bigger = api.get(path, **params, limit=most) or []
    limit = len(bigger) if len(ids(bigger)) > len(ids(first)) else None
    page = bigger if limit else first
    if limit and limit < most or not limit:          # a capped limit, or none: try paging on top
        for key, step in PAGERS:
            nxt = api.get(path, **params, limit=limit, **{key: step(len(page), 1)}) or []
            if ids(nxt) - ids(page):
                return limit, key, step
    return limit, None, None


def pager_text(pager, first):
    limit, key, _ = pager
    if not limit and not key:
        return f"only the {len(first)} most recent matches come back (no limit or paging found)"
    return ", ".join(x for x in ((f"the API honours limit={limit}" if limit else ""),
                                 (f"paged by {key}" if key else "")) if x)


def history(api, path, params, pager, most, max_pages):
    """Every row of a paged list, by the pager found for it."""
    limit, key, step = pager
    rows = api.get(path, **params, limit=limit) or []
    if key is None:
        return rows
    seen, n = ids(rows), len(rows)
    for k in range(1, max_pages):
        more = api.get(path, **params, limit=limit, **{key: step(n, k)}) or []
        new = [r for r in more if isinstance(r, dict) and r.get("externalId") not in seen]
        if not new:
            break
        seen |= ids(new)
        rows += new
    return rows


def cancelled(m):
    return str(m.get("isCancelled")).strip().lower() in ("true", "1", "yes")


def write_csv(path, rows, first=()):
    fields = list(dict.fromkeys(list(first) + [k for r in rows for k in r]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def stat_rows(match, data):
    """match/stats as rows: one per period and side, every stat a column."""
    out = []
    for block in data if isinstance(data, list) else []:
        if not isinstance(block, dict):
            continue
        for side in ("A", "B"):
            stats = block.get(f"team{side}")
            if not isinstance(stats, dict):
                continue
            row = {"externalId": match.get("externalId"), "startDate": match.get("startDate"),
                   "period": block.get("periodType"), "side": side,
                   "participant": match.get(f"participant{side}Name"), "team": match.get(f"team{side}Name"),
                   "opponent": match.get(f"participant{'B' if side == 'A' else 'A'}Name")}
            row.update(flatten(stats))
            out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sport", default="nfl", help="the API's sport key (default nfl)")
    ap.add_argument("--since", help="match stats only for matches starting on or after this date (YYYY-MM-DD)")
    ap.add_argument("--max-matches", type=int, help="match stats for at most this many matches (newest first)")
    ap.add_argument("--players-only", action="store_true", help="only players.csv")
    ap.add_argument("--timelines", action="store_true", help="also every match's timeline")
    ap.add_argument("--most", type=int, default=5000, help="the limit asked for when the API takes one")
    ap.add_argument("--max-pages", type=int, default=400, help="most pages read of one history")
    ap.add_argument("--delay", type=float, default=0.25, help="seconds between calls (default 0.25)")
    ap.add_argument("--out", default="scrapes/h2hggl_api", help="output folder (a dated folder inside it)")
    ap.add_argument("--refresh", action="store_true", help="fetch matches again even where cached")
    ap.add_argument("--base", default=BASE, help=f"the API (default {BASE})")
    args = ap.parse_args(argv)
    if importlib.util.find_spec("requests") is None:
        sys.exit("needs requests:  py -m pip install requests")

    root = Path(args.out)
    out = root / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    api = Api(args.base, args.delay, root / "raw", args.refresh)
    sport = args.sport
    L = [f"h2hggl API pull, {dt.datetime.now():%Y-%m-%d %H:%M}, {api.base}"]

    players = api.get(f"participant/{sport}") or []
    if len(players) <= 15:                           # the page asks with a limit: ask for them all
        players = api.get(f"participant/{sport}", limit=args.most) or players
    rows = [flatten(p) for p in players if isinstance(p, dict)]
    write_csv(out / "players.csv", rows, ("participantName",))
    names = api.get(f"participant/{sport}/names") or []
    names = sorted({n for n in names if isinstance(n, str)} | {r.get("participantName") for r in rows
                                                                if r.get("participantName")})
    L.append(f"players.csv: {len(rows)} players, {len(rows[0]) if rows else 0} fields; "
             f"{len(names)} names known")
    print(L[-1], flush=True)
    if args.players_only:
        return finish(out, L, api)

    path = f"schedule/past/{sport}"
    first = api.get(path, participant=names[0]) if names else []
    pager = find_pager(api, path, {"participant": names[0]}, first or [], args.most) if first else (None, None, None)
    L.append(f"history: {pager_text(pager, first or [])}")
    print(L[-1], flush=True)
    matches, per_player = {}, {}
    for k, name in enumerate(names):
        hist = history(api, path, {"participant": name}, pager, args.most, args.max_pages)
        per_player[name] = len(hist)
        for m in hist:
            if isinstance(m, dict) and m.get("externalId"):
                matches[m["externalId"]] = m
        if (k + 1) % 10 == 0 or k + 1 == len(names):
            print(f"  histories: {k + 1} of {len(names)} players, {len(matches):,} matches", flush=True)
    first_all = api.get(path) or []
    every = history(api, path, {}, find_pager(api, path, {}, first_all, args.most), args.most, args.max_pages)
    added = len(ids(every) - set(matches))
    for m in every:
        if isinstance(m, dict) and m.get("externalId"):
            matches.setdefault(m["externalId"], m)
    mrows = sorted((flatten(m) for m in matches.values()), key=lambda m: str(m.get("startDate")), reverse=True)
    write_csv(out / "matches.csv", mrows, MATCH_FIELDS)
    counts = sorted(per_player.values())
    dates = [str(m.get("startDate"))[:10] for m in mrows if m.get("startDate")]
    L.append(f"matches.csv: {len(mrows):,} matches, {min(dates) if dates else '-'} to {max(dates) if dates else '-'} "
             f"({added:,} only in the site's schedule); matches per player: "
             + (f"min {counts[0]}, median {statistics.median(counts):g}, max {counts[-1]}" if counts else "none"))
    print(L[-1], flush=True)

    todo = [m for m in mrows if not cancelled(m) and (not args.since or str(m.get("startDate")) >= args.since)]
    if args.max_matches:
        todo = todo[:args.max_matches]
    srows, periods, missing = [], Counter(), 0
    if args.timelines:
        (out / "timelines").mkdir(exist_ok=True)
    for k, m in enumerate(todo):
        data = api.get(f"match/stats/{sport}", cached=True, external_id=m["externalId"])
        rows_ = stat_rows(m, data)
        missing += not rows_
        periods.update(r["period"] for r in rows_)
        srows += rows_
        if args.timelines:
            t = api.get("timeline/", cached=True, external_id=m["externalId"])
            if t is not None:
                (out / "timelines" / f"{m['externalId']}.json").write_text(json.dumps(t), encoding="utf-8")
        if (k + 1) % 100 == 0 or k + 1 == len(todo):
            print(f"  match stats: {k + 1:,} of {len(todo):,} ({api.hits:,} from the cache)", flush=True)
    write_csv(out / "match_stats.csv", srows, ("externalId", "startDate", "period", "side", "participant",
                                              "team", "opponent"))
    L.append(f"match_stats.csv: {len(srows):,} rows for {len(todo) - missing:,} of {len(todo):,} matches "
             f"({missing:,} with none); periods: " + ", ".join(f"{p} {n:,}" for p, n in periods.most_common()))
    L.append("stats per side: " + ", ".join(k for k in (srows[0] if srows else {})
                                            if k not in ("externalId", "startDate", "period", "side",
                                                         "participant", "team", "opponent")))
    L.append("matches per player: " + ", ".join(f"{n} {c}" for n, c in sorted(per_player.items(), key=lambda x: -x[1])))
    return finish(out, L, api)


def finish(out, L, api):
    L.append(f"{api.calls:,} calls, {api.hits:,} from the cache, {len(api.failed)} failed"
             + "".join(f"\n  failed {u}: {e}" for u, e in api.failed[:20]))
    L.append(f"Everything is in {out}")
    (out / "summary.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
