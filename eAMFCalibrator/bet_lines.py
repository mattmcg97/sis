"""Prod's spread and total lines through each match, against the lines customers bet.

If an operator only offers the lines GAMEPLAI sends, a bet on another line than prod's live one
was placed on a line prod showed a little earlier (the operator runs behind) or a little later
(its clock runs behind GAMEPLAI's). This reads which, how far, and whether the gap moves through
a match, and draws each match: prod's line as a step, every bet as a dot.
"""

import html
import math
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict

from . import bets, config, markets

WINDOW = 120.0
CURRENT, PAST, FUTURE, NEVER = "current", "past", "future", "never"
COLOURS = {CURRENT: "#2e7d32", PAST: "#e65100", FUTURE: "#1565c0", NEVER: "#c62828"}
PANELS = ((54, "total (over line; unders on the same line)", (54, 55)),
          (52, "spread, home side", (52,)),
          (53, "spread, away side", (53,)))


def spans(quote_rows):
    """(match, market) -> [(start, end, line, live)]: the stream's rows in time order, runs of the same
    line and liveness merged, each lasting until the next change (the last until its last row)."""
    rows = defaultdict(list)
    for r in quote_rows:
        match, market, publish, prob, _, desc, msg, status, active = r[:9]
        if publish is None:
            continue
        rows[(match, int(market))].append((bets._naive(publish), bets._live(active),
                                           markets.parse_line(desc)))
    out = {}
    for key, rs in rows.items():
        rs.sort(key=lambda x: (x[0], x[1]))
        merged = []
        for t, live, line in rs:
            if merged and merged[-1][2] == line and merged[-1][3] == live:
                merged[-1][1] = t
                continue
            if merged:
                merged[-1][1] = t
            merged.append([t, t, line, live])
        out[key] = [tuple(m) for m in merged]
    return out


def classify(t, line, market_spans, window=WINDOW):
    """(class, seconds) for a bet on `line` at time t: current (prod's live line then), past (prod
    last showed it live `seconds` before), future (prod showed it live `seconds` after), or never
    (not within `window` seconds either way)."""
    live = [s for s in market_spans if s[3]]
    starts = [s[0] for s in live]
    i = bisect_right(starts, t) - 1
    if i >= 0 and bets._same_line(line, live[i][2]) and line is not None and live[i][2] is not None:
        return CURRENT, 0.0
    past = [(t - s[1]).total_seconds() for s in live[:i + 1]
            if s[2] is not None and abs(s[2] - line) < 1e-6 and s[1] <= t]
    future = [(s[0] - t).total_seconds() for s in live[i + 1:]
              if s[2] is not None and abs(s[2] - line) < 1e-6]
    best_past = min(past) if past else None
    best_future = min(future) if future else None
    if best_past is not None and best_past <= window and (best_future is None or best_past <= best_future):
        return PAST, best_past
    if best_future is not None and best_future <= window:
        return FUTURE, best_future
    return NEVER, None


def classify_bets(all_bets, market_spans, signs):
    """[(bet, line on prod's side, class, seconds)] for every spread and total bet with a line."""
    out = []
    for b in all_bets:
        if b.market_type not in (2, 3) or b.line is None or b.time is None:
            continue
        sp = market_spans.get((b.match_code, b.feed_market))
        if not sp:
            continue
        sign = signs.get((b.operator, b.feed_market), (1,))[0]
        line = sign * b.line
        cls, secs = classify(b.time, line, sp)
        out.append((b, line, cls, secs))
    return out


def summary(classified):
    """Lines of text: how bets' lines sit against prod's, overall and by operator, market, pre-match
    or in play and period, with the seconds to the line prod showed."""
    lines = []

    def block(title, key):
        groups = defaultdict(list)
        for item in classified:
            groups[key(item)].append(item)
        lines.append(f"\n  {title}")
        lines.append(f"  {'':30s} {'bets':>7s} {'current':>8s} {'past':>7s} {'future':>7s} "
                     f"{'never':>7s}   past secs (median, 90th)   future secs (median, 90th)")
        for g, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
            g = " · ".join(str(x) for x in g) if isinstance(g, tuple) else g
            n = len(items)
            c = Counter(i[2] for i in items)
            past = sorted(i[3] for i in items if i[2] == PAST)
            fut = sorted(i[3] for i in items if i[2] == FUTURE)
            fmt = lambda xs: (f"{statistics.median(xs):6.1f} {xs[math.ceil(0.9 * (len(xs) - 1))]:6.1f}"
                              if xs else f"{'-':>6s} {'-':>6s}")
            lines.append(f"  {str(g)[:30]:30s} {n:7,d} {100 * c[CURRENT] / n:7.1f}% "
                         f"{100 * c[PAST] / n:6.1f}% {100 * c[FUTURE] / n:6.1f}% "
                         f"{100 * c[NEVER] / n:6.1f}%   {fmt(past)}                 {fmt(fut)}")
    block("by operator and market", lambda i: (i[0].operator, markets.market_group(i[0].feed_market)))
    block("by operator, pre-match or in play", lambda i: (i[0].operator,
                                                          "in play" if i[0].in_play else "pre-match"))
    block("in play, by operator and period (does the gap move through the match?)",
          lambda i: (i[0].operator, f"period {i[0].period}") if i[0].in_play else "pre-match")
    return lines


def _svg(match, market, title, market_ids, market_spans, classified, width=960, height=190):
    """One panel: prod's live line as a step, dead rows faint, the bets as dots."""
    sp = market_spans.get((match, market), [])
    pts = [(b, line, cls, secs) for b, line, cls, secs in classified
           if b.match_code == match and b.feed_market in market_ids]
    if not sp and not pts:
        return ""
    times = [s[0] for s in sp] + [s[1] for s in sp] + [b.time for b, *_ in pts]
    values = [s[2] for s in sp if s[2] is not None] + [line for _, line, _, _ in pts]
    if not values:
        return ""
    t0, t1 = min(times), max(times)
    span_s = max(1.0, (t1 - t0).total_seconds())
    lo, hi = min(values) - 1, max(values) + 1
    left, right, top, bottom = 50, 10, 22, 24
    x = lambda t: left + (width - left - right) * (t - t0).total_seconds() / span_s
    y = lambda v: top + (height - top - bottom) * (hi - v) / (hi - lo)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
             f'aria-label="{html.escape(match)} {html.escape(title)}">',
             f'<text x="{left}" y="14" class="t">{html.escape(title)}</text>']
    step = max(1, int((hi - lo) / 6))
    v = int(lo)
    while v <= hi:
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}" class="g"/>'
                     f'<text x="{left - 6}" y="{y(v) + 4:.1f}" class="a" text-anchor="end">{v:g}</text>')
        v += step
    for minute in range(0, int(span_s // 60) + 1, 5):
        xx = left + (width - left - right) * minute * 60 / span_s
        parts.append(f'<text x="{xx:.1f}" y="{height - 6}" class="a" text-anchor="middle">{minute}m</text>')
    for start, end, line, live in sp:
        if line is None:
            continue
        cls = "s" if live else "d"
        parts.append(f'<line x1="{x(start):.1f}" x2="{max(x(end), x(start) + 1):.1f}" '
                     f'y1="{y(line):.1f}" y2="{y(line):.1f}" class="{cls}"/>')
    for b, line, cls, secs in pts:
        tip = (f"{b.time:%H:%M:%S} {markets.selection_label(b.feed_market)} {line:g} "
               f"odds {b.odds} {b.operator}: {cls}" + (f" {secs:.0f}s" if secs else ""))
        parts.append(f'<circle cx="{x(b.time):.1f}" cy="{y(line):.1f}" r="3" '
                     f'fill="{COLOURS[cls]}" fill-opacity="0.75"><title>{html.escape(tip)}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def page(matches, market_spans, classified, summary_lines):
    """The HTML page: the summary, then each match's panels."""
    style = """
    :root { --bg:#fff; --ink:#1a1a1a; --muted:#666; --grid:#e6e6e6; --step:#1a1a1a; --dead:#bbb; }
    @media (prefers-color-scheme: dark) { :root { color-scheme:dark; --bg:#141414; --ink:#e8e8e8;
      --muted:#999; --grid:#2a2a2a; --step:#e8e8e8; --dead:#555; } }
    body { background:var(--bg); color:var(--ink); font-family:Helvetica,Arial,sans-serif; margin:0;
      padding:20px 16px; }
    main { max-width:1000px; margin:0 auto; }
    pre { font-size:12px; overflow-x:auto; }
    h2 { font-size:16px; margin:28px 0 4px; }
    .t { font-size:12px; fill:var(--ink); } .a { font-size:10px; fill:var(--muted); }
    .g { stroke:var(--grid); } .s { stroke:var(--step); stroke-width:2; }
    .d { stroke:var(--dead); stroke-width:4; stroke-opacity:0.6; }
    .key span { display:inline-block; margin-right:14px; font-size:13px; }
    .key i { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
    """
    key = "".join(f'<span><i style="background:{c}"></i>{k}</span>' for k, c in COLOURS.items())
    body = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Bet Lines</title><style>{style}</style></head><body><main>",
            "<h1>Prod's lines against the lines bet</h1>",
            f"<p class='key'>{key} &nbsp; prod's live line: dark step; inactive rows: grey</p>",
            "<pre>" + html.escape("\n".join(summary_lines)) + "</pre>"]
    for match in matches:
        body.append(f"<h2>{html.escape(match)}</h2>")
        for market, title, ids in PANELS:
            body.append(_svg(match, market, title, ids, market_spans, classified))
    body.append("</main></body></html>")
    return "\n".join(body)


def run(cur, out_dir, n_matches=12, only=None):
    """Fetch the window's bets and prod's quotes, classify every spread and total bet's line, print
    the summary and write out/bets_lines.html for the busiest matches (or `only`)."""
    import os
    from . import snowflake_io
    sql, params, _ = bets.bets_sql()
    cols, raw = bets.fetch_all(cur, sql, tuple(params))
    all_bets = bets.to_bets(cols, raw)
    counts = Counter(b.match_code for b in all_bets if b.market_type in (2, 3))
    matches = sorted(counts)
    prod_rows = []
    for start in range(0, len(matches), config.MATCH_CHUNK_SIZE):
        prod_rows += snowflake_io.fetch_quotes(cur, config.STREAMS["prod"],
                                               matches[start:start + config.MATCH_CHUNK_SIZE])
    market_spans = spans(prod_rows)
    signs = bets.fit_line_signs(all_bets, bets.timeline(prod_rows), {})
    classified = classify_bets(all_bets, market_spans, signs)
    lines = summary(classified)
    print("\n".join(lines))
    chosen = only or [m for m, _ in counts.most_common(n_matches)]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bets_lines.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(page(chosen, market_spans, classified, lines))
    return path, classified
