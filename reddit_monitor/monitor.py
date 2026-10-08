"""Find Reddit posts and subreddits tipping eFootball / eSoccer / eBasketball bets.

Two modes:
  * No login (default if no keys set): uses Reddit's public JSON endpoints.
    Slow (~1 request / 7s) and works best from a home/office connection.
        pip install requests
        python reddit_monitor/monitor.py --time year --out hits.csv
  * API keys: faster, higher limits.
        pip install praw
        export REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=...
        python reddit_monitor/monitor.py --time year --out hits.csv
"""
import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime, timezone

import config

PLAYER_CAPS = re.compile(r"\b(" + "|".join(config.PLAYERS) + r")\b")
PLAYER_ANY = re.compile(r"\b(" + "|".join(config.PLAYERS) + r")\b", re.I)


def _contains(text, terms):
    return [t for t in terms if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text)]


def score(text):
    """Return (score, details) for a block of text; score 0 means not relevant."""
    lower = text.lower()
    sports = _contains(lower, config.SPORT_TERMS)
    if not sports:
        return 0, {}
    bets = _contains(lower, config.BET_TERMS)
    books = _contains(lower, config.BOOKS)
    regions = _contains(lower, config.REGIONS)
    players = set(PLAYER_CAPS.findall(text))
    if bets or books:  # betting context makes mixed-case names count too
        players |= {p.upper() for p in PLAYER_ANY.findall(text)}
    s = 2 + sum(config.BET_TERMS[b] for b in bets) + 2 * len(books) + 3 * len(players) + len(regions)
    return s, {"sports": sports, "bets": bets, "books": books,
               "regions": regions, "players": sorted(players)}


# --- Backends. Both yield plain dicts so the rest of the script doesn't care. ---

def _post_dict(id, title, body, subreddit, author, created_utc, permalink):
    return {"id": id, "title": title or "", "body": body or "", "subreddit": subreddit,
            "author": author, "created_utc": created_utc, "permalink": permalink}


class PrawBackend:
    def __init__(self):
        import praw
        self.r = praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"],
                             client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                             user_agent="esports-integrity-monitor/0.2")

    def search(self, query, subreddit, time_filter, limit):
        for p in self.r.subreddit(subreddit).search(query, sort="new", time_filter=time_filter, limit=limit):
            yield _post_dict(p.id, p.title, p.selftext, p.subreddit.display_name,
                             str(p.author), p.created_utc, p.permalink)

    def comments(self, post):
        sub = self.r.submission(id=post["id"])
        sub.comments.replace_more(limit=0)
        return [c.body for c in sub.comments.list()]

    def search_subreddits(self, query):
        for s in self.r.subreddits.search(query, limit=100):
            yield s.display_name, s.title, s.public_description, s.subscribers


class JsonBackend:
    """Reddit's public .json endpoints - no login, ~10 requests/minute."""
    BASE = "https://www.reddit.com"

    def __init__(self, delay=7.0):
        import requests
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "esports-integrity-monitor/0.2 (research script)"
        self.delay = delay
        self._last = 0.0

    def _get(self, path, params):
        import requests
        for attempt in range(5):
            wait = self.delay - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                resp = self.s.get(self.BASE + path, params={**params, "raw_json": 1}, timeout=30)
            except requests.exceptions.ConnectionError as e:
                sys.exit(f"Could not reach Reddit ({e.__class__.__name__}). This network blocks it - "
                         "run from a home/office connection or use API keys.")
            if resp.status_code == 429:
                time.sleep(int(resp.headers.get("retry-after", 60)) or 60)
                continue
            if resp.status_code in (403, 451):
                sys.exit(f"Reddit refused the request ({resp.status_code}). This usually means the "
                         "network is blocked - run from a home/office connection or use API keys.")
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Gave up after repeated rate limits on {path}")

    def _listing(self, path, params, limit):
        after, got = None, 0
        while got < limit:
            data = self._get(path, {**params, "limit": 100, **({"after": after} if after else {})})["data"]
            for child in data["children"]:
                yield child["data"]
                got += 1
            after = data.get("after")
            if not after:
                break

    def search(self, query, subreddit, time_filter, limit):
        path = "/search.json" if subreddit == "all" else f"/r/{subreddit}/search.json"
        params = {"q": query, "sort": "new", "t": time_filter, "type": "link"}
        if subreddit != "all":
            params["restrict_sr"] = 1
        for p in self._listing(path, params, limit):
            yield _post_dict(p["id"], p.get("title"), p.get("selftext"), p.get("subreddit"),
                             p.get("author"), p.get("created_utc", 0), p.get("permalink"))

    def comments(self, post):
        data = self._get(post["permalink"].rstrip("/") + ".json", {"limit": 500})
        out, stack = [], list(data[1]["data"]["children"]) if len(data) > 1 else []
        while stack:
            c = stack.pop()
            if c.get("kind") != "t1":
                continue
            out.append(c["data"].get("body", ""))
            replies = c["data"].get("replies")
            if isinstance(replies, dict):
                stack.extend(replies["data"]["children"])
        return out

    def search_subreddits(self, query):
        for s in self._listing("/subreddits/search.json", {"q": query}, 100):
            yield s.get("display_name"), s.get("title"), s.get("public_description"), s.get("subscribers")


# --- Main logic ---

def iter_posts(backend, time_filter, limit):
    for q in config.QUERIES:
        yield from backend.search(q, "all", time_filter, limit)
    for name in config.WATCH_SUBREDDITS:
        for q in ("efootball", "esoccer", "ebasketball", "cyber"):
            yield from backend.search(q, name, time_filter, limit)


def find_subreddits(backend):
    seen = {}
    for q in config.SUBREDDIT_QUERIES:
        for name, title, desc, members in backend.search_subreddits(q):
            s, d = score(f"{name} {title or ''} {desc or ''} tips")  # descriptions rarely say "bet"
            if s and name not in seen:
                seen[name] = (s, members, d, desc)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="year", choices=["day", "week", "month", "year", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max results per query (default 1000 with keys, 250 without)")
    ap.add_argument("--comments", action="store_true", help="also scan comments of hits")
    ap.add_argument("--no-auth", action="store_true", help="force public JSON mode")
    ap.add_argument("--out", default="reddit_hits.csv")
    args = ap.parse_args()

    use_keys = not args.no_auth and os.environ.get("REDDIT_CLIENT_ID")
    backend = PrawBackend() if use_keys else JsonBackend()
    limit = args.limit or (1000 if use_keys else 250)
    print(f"Mode: {'API keys' if use_keys else 'public JSON (no login) - this will take a while'}")

    rows, seen = [], set()
    for post in iter_posts(backend, args.time, limit):
        if post["id"] in seen:
            continue
        seen.add(post["id"])
        s, d = score(f"{post['title']}\n{post['body']}")
        if not s:
            continue
        if args.comments:
            for body in backend.comments(post):
                _, cd = score(f"{post['title']}\n{body}")
                new = set(cd.get("players", [])) - set(d["players"])
                if new:
                    d["players"] = sorted(set(d["players"]) | new)
                    s += 3 * len(new)
        rows.append({
            "score": s, "subreddit": post["subreddit"], "author": post["author"],
            "title": post["title"],
            "created": datetime.fromtimestamp(post["created_utc"], timezone.utc).isoformat(),
            "url": f"https://reddit.com{post['permalink']}",
            **{k: "; ".join(v) for k, v in d.items()},
        })
        print(f"  hit [{s}] r/{post['subreddit']}: {post['title'][:70]}")

    rows.sort(key=lambda r: r["score"], reverse=True)
    fields = ["score", "subreddit", "author", "title", "created", "url",
              "sports", "bets", "books", "regions", "players"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} relevant posts (of {len(seen)} scanned) -> {args.out}")

    sub_counts = {}
    for r in rows:
        sub_counts[r["subreddit"]] = sub_counts.get(r["subreddit"], 0) + 1
    print("\nSubreddits with most hits:")
    for name, n in sorted(sub_counts.items(), key=lambda x: -x[1])[:25]:
        print(f"  r/{name}: {n}")
    print("\nCandidate dedicated subreddits:")
    for name, (s, members, d, desc) in sorted(find_subreddits(backend).items(), key=lambda x: -x[1][0])[:25]:
        print(f"  r/{name} ({members} members, score {s}): {(desc or '')[:80]}")


if __name__ == "__main__":
    main()
