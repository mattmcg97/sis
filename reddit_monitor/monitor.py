"""Find Reddit posts and subreddits tipping eFootball / eSoccer / eBasketball bets.

Setup:
    pip install praw
    export REDDIT_CLIENT_ID=... REDDIT_CLIENT_SECRET=...   # reddit.com/prefs/apps, "script" app
    python reddit_monitor/monitor.py --time year --out hits.csv
"""
import argparse
import csv
import os
import re
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


def reddit_client():
    import praw
    return praw.Reddit(client_id=os.environ["REDDIT_CLIENT_ID"],
                       client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                       user_agent="esports-integrity-monitor/0.1")


def iter_posts(reddit, time_filter, limit):
    for q in config.QUERIES:
        yield from reddit.subreddit("all").search(q, sort="new", time_filter=time_filter, limit=limit)
    for name in config.WATCH_SUBREDDITS:
        sub = reddit.subreddit(name)
        for q in ("efootball", "esoccer", "ebasketball", "cyber"):
            yield from sub.search(q, sort="new", time_filter=time_filter, limit=limit)


def find_subreddits(reddit):
    seen = {}
    for q in config.SUBREDDIT_QUERIES:
        for sub in reddit.subreddits.search(q, limit=100):
            text = f"{sub.display_name} {sub.title or ''} {sub.public_description or ''}"
            s, d = score(text + " tips")  # descriptions rarely say "bet" twice
            if s and sub.display_name not in seen:
                seen[sub.display_name] = (s, sub.subscribers, d, sub.public_description)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", default="year", choices=["day", "week", "month", "year", "all"])
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--comments", action="store_true", help="also scan comments of hits")
    ap.add_argument("--out", default="reddit_hits.csv")
    args = ap.parse_args()

    reddit = reddit_client()
    rows, seen = [], set()
    for post in iter_posts(reddit, args.time, args.limit):
        if post.id in seen:
            continue
        seen.add(post.id)
        s, d = score(f"{post.title}\n{post.selftext}")
        if not s:
            continue
        if args.comments:
            post.comments.replace_more(limit=0)
            for c in post.comments.list():
                cs, cd = score(f"{post.title}\n{c.body}")
                if cd.get("players"):
                    d["players"] = sorted(set(d["players"]) | set(cd["players"]))
                    s += 3 * len(cd["players"])
        rows.append({
            "score": s, "subreddit": post.subreddit.display_name,
            "author": str(post.author), "title": post.title,
            "created": datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat(),
            "url": f"https://reddit.com{post.permalink}",
            **{k: "; ".join(v) for k, v in d.items()},
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    fields = ["score", "subreddit", "author", "title", "created", "url",
              "sports", "bets", "books", "regions", "players"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} relevant posts -> {args.out}")

    subs = find_subreddits(reddit)
    sub_counts = {}
    for r in rows:
        sub_counts[r["subreddit"]] = sub_counts.get(r["subreddit"], 0) + 1
    print("\nSubreddits with most hits:")
    for name, n in sorted(sub_counts.items(), key=lambda x: -x[1])[:25]:
        print(f"  r/{name}: {n}")
    print("\nCandidate dedicated subreddits:")
    for name, (s, subs_n, d, desc) in sorted(subs.items(), key=lambda x: -x[1][0])[:25]:
        print(f"  r/{name} ({subs_n} members, score {s}): {(desc or '')[:80]}")


if __name__ == "__main__":
    main()
