"""Scrape every player stat h2hggl.com shows for eAmerican football.

The site builds its pages in the browser, so this drives a real (headless) Chromium. Starting at
the section's page it visits every page of the section and every player page linked from it,
politely (--delay seconds apart). On each page it:

  - keeps every JSON response the page loads (the site's own data feed): json/NNNNN.json, with
    each one's URL and shape in index.csv. Where the stats come from an API, this is the clean
    copy of them
  - keeps the websocket frames (live data), in websocket.jsonl
  - saves the page's tables as CSV and its visible text, after scrolling it to the bottom
  - clicks its stats-like tabs and "show more" buttons, and saves what they open

summary.txt then says what was found: the pages, the data endpoints and their fields, and the
tables with their headers. That tells us where the stats live, so a second pass can pull them
straight into one table.

Setup, once:

    py -m pip install playwright
    py -m playwright install chromium

Run:

    py analysis/scrape_h2hggl.py                 # everything, to scrapes/h2hggl/<time>/
    py analysis/scrape_h2hggl.py --show          # watch the browser do it
    py analysis/scrape_h2hggl.py --max-pages 20  # a quick look first
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from urllib.parse import parse_qsl, urldefrag, urlparse

START = "https://h2hggl.com/en/eamericanfootball"
SECTION = "eamericanfootball"
# a link from a section page is followed when its path says it is about a player or his stats
PLAYERISH = re.compile(r"player|participant|profile|gamer|team|stat|standing|ranking|leader", re.I)
TABS = re.compile(r"stat|player|standing|ranking|leader|table|histor|result|form|h2h|head|profile|record|match",
                  re.I)
MORE = re.compile(r"^(load|show|view|see) ?(more|all)\b|^more$", re.I)
ACCEPT = re.compile(r"^(accept|agree|allow|ok|got it|i agree)", re.I)
ID_SEGMENT = re.compile(r"^(\d+|[0-9a-f]{8,}|[0-9a-f-]{20,})$", re.I)
SKIP = re.compile(r"\.(png|jpe?g|gif|svg|webp|ico|css|js|woff2?|ttf|pdf|zip|mp4)$", re.I)

TABLES_JS = """() => Array.from(document.querySelectorAll('table')).map(t =>
    Array.from(t.querySelectorAll('tr')).map(r =>
        Array.from(r.querySelectorAll('th,td')).map(c => (c.innerText || '').trim())))"""
LINKS_JS = "() => Array.from(document.querySelectorAll('a[href]')).map(a => a.href)"
CLICKABLE_JS = """(re) => {
    const pat = new RegExp(re, 'i');
    const els = Array.from(document.querySelectorAll(
        '[role=tab], button, [role=button], li, a:not([href]), [class*=tab] > *'));
    const seen = new Set(), out = [];
    for (const el of els) {
        const text = (el.innerText || '').trim();
        if (!text || text.length > 30 || !pat.test(text) || seen.has(text)) continue;
        const r = el.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        seen.add(text);
        out.push(text);
    }
    return out;
}"""


def slug(text, n=60):
    """A string fit for a file name."""
    return (re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "root")[:n]


def pattern(url):
    """A URL's path with its ids as {id} and its query as keys only, so one endpoint called for
    every player reads as one line."""
    p = urlparse(url)
    path = "/".join("{id}" if ID_SEGMENT.match(seg) else seg for seg in p.path.split("/"))
    keys = sorted({k for k, _ in parse_qsl(p.query)})
    return f"{p.netloc}{path}" + (f"?{'&'.join(keys)}" if keys else "")


def fold(patterns, most=3):
    """pattern -> its label, with the last path segment as {name} wherever more than `most`
    patterns differ only there (one page or endpoint per player, keyed by his name)."""
    def split(pat):
        path, _, query = pat.partition("?")
        head, _, last = path.rstrip("/").rpartition("/")
        stem = re.sub(r"[^.]*", "", last, count=1)          # keep an extension: .json
        return (head, stem, path.endswith("/"), query), last
    siblings = defaultdict(set)
    for pat in patterns:
        key, last = split(pat)
        siblings[key].add(last)
    out = {}
    for pat in patterns:
        key, last = split(pat)
        head, stem, slash, query = key
        if len(siblings[key]) > most:
            out[pat] = f"{head}/{{name}}{stem}{'/' if slash else ''}" + (f"?{query}" if query else "")
        else:
            out[pat] = pat
    return out


def shape(data, depth=0):
    """A short description of a JSON value: its keys, and what its lists hold."""
    if isinstance(data, dict):
        if depth >= 2:
            return "{" + ", ".join(list(data)[:12]) + (", ..." if len(data) > 12 else "") + "}"
        parts = []
        for k, v in list(data.items())[:15]:
            parts.append(f"{k}: {shape(v, depth + 1)}" if isinstance(v, (dict, list)) else k)
        return "{" + ", ".join(parts) + (", ..." if len(data) > 15 else "") + "}"
    if isinstance(data, list):
        return f"list[{len(data)}] of " + (shape(data[0], depth + 1) if data else "nothing")
    return type(data).__name__


class Recorder:
    """Every JSON response and websocket frame the browser sees, written as it arrives."""

    def __init__(self, out, max_frames):
        self.dir = out / "json"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index, self.seen, self.page = [], set(), None
        self.ws = (out / "websocket.jsonl").open("w", encoding="utf-8")
        self.frames, self.max_frames, self.sockets = 0, max_frames, set()

    def response(self, resp):
        try:
            if "json" not in (resp.headers.get("content-type") or "").lower():
                return
            body = resp.text()
        except Exception:
            return
        digest = hashlib.sha1(f"{resp.url}\n{body}".encode("utf-8", "replace")).hexdigest()
        if digest in self.seen:
            return
        self.seen.add(digest)
        n = len(self.index)
        name = f"{n:05d}.json"
        (self.dir / name).write_text(body, encoding="utf-8")
        try:
            desc = shape(json.loads(body))
        except ValueError:
            desc = "not JSON"
        self.index.append(dict(n=n, file=name, status=resp.status, bytes=len(body), pattern=pattern(resp.url),
                               url=resp.url, page=self.page, shape=desc))

    def websocket(self, ws):
        self.sockets.add(ws.url)

        def frame(payload):
            if self.frames >= self.max_frames:
                return
            self.frames += 1
            text = payload if isinstance(payload, str) else payload.decode("utf-8", "replace")
            self.ws.write(json.dumps({"socket": ws.url, "page": self.page, "frame": text}) + "\n")
        ws.on("framereceived", frame)

    def close(self):
        self.ws.close()


def wanted(url, host, from_section):
    """Whether a link is one to visit: the section's own pages, and player-like pages linked
    from them, on the site's host."""
    url = urldefrag(url)[0]
    p = urlparse(url)
    if p.scheme not in ("http", "https") or p.netloc.lower().removeprefix("www.") != host or SKIP.search(p.path):
        return None
    if SECTION in p.path.lower() or (from_section and PLAYERISH.search(p.path)):
        return url
    return None


def settle(page, wait):
    """Let the page finish loading, then scroll it to the bottom to wake anything lazy."""
    try:
        page.wait_for_load_state("networkidle", timeout=wait * 1000)
    except Exception:
        pass
    for _ in range(6):
        before = page.evaluate("document.body ? document.body.scrollHeight : 0")
        page.evaluate("window.scrollTo(0, document.body ? document.body.scrollHeight : 0)")
        page.wait_for_timeout(600)
        if page.evaluate("document.body ? document.body.scrollHeight : 0") == before:
            break


def click_text(page, text):
    """Click the first visible element whose own text is exactly this; True if it was clicked."""
    loc = page.get_by_text(text, exact=True)
    for i in range(min(loc.count(), 3)):
        el = loc.nth(i)
        try:
            if el.is_visible():
                el.click(timeout=3000)
                return True
        except Exception:
            continue
    return False


def save_view(page, out, name, tables_log):
    """The page as it stands: its tables as CSV, its visible text, its HTML."""
    tables = page.evaluate(TABLES_JS) or []
    for k, rows in enumerate(t for t in tables if t):
        path = out / "tables" / f"{name}__{k}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)
        tables_log.append(dict(view=name, url=page.url, table=k, rows=len(rows),
                               header=" | ".join(rows[0])[:200] if rows else ""))
    (out / "text" / f"{name}.txt").write_text(page.inner_text("body") if page.query_selector("body") else "",
                                               encoding="utf-8")
    (out / "html" / f"{name}.html").write_text(page.content(), encoding="utf-8")


def visit(page, url, out, n, args, rec, tables_log):
    """One page: load it, save it, open its tabs and 'more' buttons and save those; its links."""
    rec.page = url
    name = f"{n:04d}_{slug(urlparse(url).path)}"
    page.goto(url, wait_until="domcontentloaded", timeout=args.wait * 3000)
    settle(page, args.wait)
    if n == 0:
        for text in page.evaluate(CLICKABLE_JS, ACCEPT.pattern) or []:
            click_text(page, text)
    save_view(page, out, name, tables_log)
    links = set(page.evaluate(LINKS_JS) or [])
    for k, text in enumerate((page.evaluate(CLICKABLE_JS, TABS.pattern) or [])[:args.max_tabs]):
        if not click_text(page, text):
            continue
        settle(page, args.wait)
        for _ in range(args.max_more):
            more = page.evaluate(CLICKABLE_JS, MORE.pattern) or []
            if not more or not click_text(page, more[0]):
                break
            settle(page, args.wait)
        save_view(page, out, f"{name}__tab{k}_{slug(text, 20)}", tables_log)
        links |= set(page.evaluate(LINKS_JS) or [])
        if urldefrag(page.url)[0] != url:          # the tab was a link: back to the page
            page.goto(url, wait_until="domcontentloaded", timeout=args.wait * 3000)
            settle(page, args.wait)
    return links


def summary(out, pages, failed, rec, tables_log):
    """Lines: what was found."""
    L = [f"h2hggl scrape, {dt.datetime.now():%Y-%m-%d %H:%M}: {len(pages)} pages visited, "
         f"{len(failed)} failed, {len(rec.index)} JSON responses, {rec.frames} websocket frames"]
    by_pattern = defaultdict(list)
    folded = fold({pattern(p) for p in pages})
    for p in pages:
        by_pattern[folded[pattern(p)]].append(p)
    L.append("\nPages, by URL pattern (ids as {id}):")
    for pat, ps in sorted(by_pattern.items(), key=lambda kv: -len(kv[1])):
        L.append(f"  {len(ps):5d}  {pat}   e.g. {ps[0]}")
    endpoints = defaultdict(list)
    folded = fold({r["pattern"] for r in rec.index})
    for r in rec.index:
        endpoints[folded[r["pattern"]]].append(r)
    L.append("\nJSON data endpoints, by URL pattern (largest first):")
    for pat, rs in sorted(endpoints.items(), key=lambda kv: -sum(r["bytes"] for r in kv[1])):
        big = max(rs, key=lambda r: r["bytes"])
        L.append(f"  {len(rs):5d} calls {sum(r['bytes'] for r in rs):>11,d} bytes  {pat}")
        L.append(f"         e.g. {big['url']}  ({big['file']})")
        L.append(f"         {big['shape'][:600]}")
    if rec.sockets:
        L.append("\nWebsockets: " + ", ".join(sorted(rec.sockets)))
    heads = defaultdict(list)
    for t in tables_log:
        heads[t["header"]].append(t)
    L.append("\nTables, by header row:")
    for head, ts in sorted(heads.items(), key=lambda kv: -len(kv[1])):
        L.append(f"  {len(ts):5d} tables, {sum(t['rows'] for t in ts):,d} rows  [{head or 'no header'}]")
        L.append(f"         e.g. {ts[0]['view']}__{ts[0]['table']}.csv  ({ts[0]['url']})")
    if failed:
        L.append("\nFailed:")
        L += [f"  {u}: {e}" for u, e in failed[:30]]
    L.append(f"\nEverything is in {out}")
    return L


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", action="append", help=f"page(s) to start from (default {START})")
    ap.add_argument("--out", default="scrapes/h2hggl", help="output folder (a dated folder inside it)")
    ap.add_argument("--max-pages", type=int, default=400, help="most pages to visit (default 400)")
    ap.add_argument("--depth", type=int, default=3, help="most links away from the start (default 3)")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between pages (default 1)")
    ap.add_argument("--wait", type=float, default=10.0, help="seconds to let a page load (default 10)")
    ap.add_argument("--max-tabs", type=int, default=8, help="stats-like tabs clicked per page (default 8)")
    ap.add_argument("--max-more", type=int, default=20, help="'show more' clicks per tab (default 20)")
    ap.add_argument("--max-frames", type=int, default=5000, help="websocket frames kept (default 5000)")
    ap.add_argument("--show", action="store_true", help="show the browser")
    ap.add_argument("--browser", help="path to a Chromium to use instead of Playwright's own")
    args = ap.parse_args(argv)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.exit("needs Playwright:  py -m pip install playwright  then  py -m playwright install chromium")

    starts = args.start or [START]
    host = urlparse(starts[0]).netloc.lower().removeprefix("www.")
    out = Path(args.out) / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    for sub in ("tables", "text", "html"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rec = Recorder(out, args.max_frames)
    queue = deque((u, 0, True) for u in starts)
    queued = set(starts)
    pages, failed, tables_log = [], [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.show, executable_path=args.browser)
        ctx = browser.new_context(viewport={"width": 1440, "height": 1000},
                                  user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        page = ctx.new_page()
        page.on("response", rec.response)
        page.on("websocket", rec.websocket)
        while queue and len(pages) < args.max_pages:
            url, depth, from_section = queue.popleft()
            try:
                links = visit(page, url, out, len(pages), args, rec, tables_log)
            except Exception as exc:
                failed.append((url, str(exc).splitlines()[0][:200]))
                print(f"  failed {url}: {failed[-1][1]}", flush=True)
                continue
            pages.append(url)
            print(f"  {len(pages):4d}  {len(rec.index):5d} json  {url}", flush=True)
            if depth < args.depth:
                in_section = SECTION in urlparse(url).path.lower()
                for link in sorted(links):
                    keep = wanted(link, host, in_section)
                    if keep and keep not in queued:
                        queued.add(keep)
                        queue.append((keep, depth + 1, in_section))
            time.sleep(args.delay)
        browser.close()
    rec.close()
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["n", "file", "status", "bytes", "pattern", "url", "page", "shape"])
        w.writeheader()
        w.writerows(rec.index)
    with (out / "tables.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["view", "url", "table", "rows", "header"])
        w.writeheader()
        w.writerows(tables_log)
    lines = summary(out, pages, failed, rec, tables_log)
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
