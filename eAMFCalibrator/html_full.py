"""The combined report: prod and every candidate against what happened.

One self-contained file -- no external fetches -- so it opens anywhere.
Every table reads the real outcome, prod, then one group of columns per
candidate (`--candidate v4,v5`; a single candidate is a group of one), so a
new version is compared by naming it on the command line.

The numbers come from multi.build(): each candidate paired with prod over
one common population, read at prod's line for the calibration tables and
at its own line for the line tables (see multi.py).
"""

import html
import os

from . import buckets, config, drives, handles, html_style, markets, totals_reach

MARKET_ORDER = [markets.MONEYLINE, markets.SPREAD, markets.TOTAL]
MARKET_TITLES = {markets.MONEYLINE: "Moneyline", markets.SPREAD: "Spread",
                 markets.TOTAL: "Total"}
LINE_MARKETS = [markets.SPREAD, markets.TOTAL]


def _pct(v, spec=".1f"):
    return "&mdash;" if v is None else f"{format(100 * v, spec)}%"


def _n(v, spec="+.4f"):
    return "&mdash;" if v is None else format(v, spec)


def _p(v):
    if v is None:
        return "&mdash;"
    return "&lt;0.001" if v < 0.001 else f"{v:.3f}"


def _ci(d, spec="+.4f"):
    if not d or d.get("ci_low") is None:
        return "&mdash;"
    return f"[{format(d['ci_low'], spec)}, {format(d['ci_high'], spec)}]"


def _cls(v, good_when_positive=True):
    if v is None:
        return ""
    if abs(v) < 1e-12:
        return ""
    positive = v > 0
    return "good" if positive == good_when_positive else "bad"


# Green near zero, red far from it. Every column headed "gap" answers the
# same question -- how far apart are two things that were meant to agree --
# so they share one ramp and differ only in where its steps fall. The steps
# are FIXED rather than taken from each run's own quantiles, so two reports
# can be read against each other.
PROB_GAP = (0.02, 0.05, 0.10, 0.20)      # realized minus predicted
PROB_DELTA = (0.02, 0.04, 0.06, 0.10)    # prod against candidate
LINE_GAP = (0.5, 1.0, 3.0, 6.0)          # handicap or total points
MESSAGE_GAP = (0, 1, 2, 3)               # feed messages


def _gap(value, cuts, extra=""):
    """The ramp class for a gap: magnitude, not sign."""
    if value is None or value == "":
        return extra
    magnitude = abs(float(value))
    step = next((i for i, cut in enumerate(cuts) if magnitude <= cut), len(cuts))
    return f"{extra} g{step}".strip()


def _outcome(value):
    if value is None:
        return '<span class="dim">push</span>'
    return '<span class="good">won</span>' if value else '<span class="bad">lost</span>'


# Display names for streams whose table names are long.
SHORT_NAMES = {"GAMEPLAI_STREAM_CANDIDATE": "Cand"}


def _name(side):
    name = str(side["name"])
    return html.escape(SHORT_NAMES.get(name.upper(), name))


def _dash(n):
    return "<td>&mdash;</td>" * n


# ---------------------------------------------------------------------------
# The verdict, as one line (kept for the console and the tests)
# ---------------------------------------------------------------------------

def _verdict(report):
    """The headline of one prod-vs-candidate report as a line of numbers."""
    same = report["summary"]["same_line"]
    decisive = report["summary"]["decisive"]
    brier = same["brier"]
    mean, lo, hi = brier.get("mean"), brier.get("ci_low"), brier.get("ci_high")
    votes = decisive["votes"]
    rate, p_value = votes.get("candidate_win_rate"), votes.get("p_value")

    def name(side):
        return "CANDIDATE" if side == "candidate" else "PROD"

    combined_side = None if rate is None or rate == 0.5 else (
        "candidate" if rate > 0.5 else "prod")
    same_side = None if mean is None else ("candidate" if mean > 0 else "prod")
    same_separates = mean is not None and lo is not None and not (lo <= 0 <= hi)
    combined_separates = p_value is not None and p_value <= 0.05 and combined_side

    if combined_separates:
        tone, head = ("good" if combined_side == "candidate" else "bad",
                      name(combined_side))
    elif same_separates:
        tone, head = ("good" if same_side == "candidate" else "bad",
                      f"{name(same_side)} on same line")
    else:
        tone, head = "neutral", "NO DIFFERENCE"

    parts = [f"<b>{head}</b>",
             f"overall {votes['candidate']}&ndash;{votes['prod']} "
             f"<span class=\"dim\">matches</span>, p {_p(p_value)}",
             f"{decisive['n']:,} pairs "
             f"<span class=\"dim\">({decisive['settled_on_line']:,} on line)</span>"]
    if mean is not None:
        parts.append(f"same line {mean:+.4f} &Delta;Brier, CI "
                     f"{'excludes' if same_separates else 'crosses'} 0")
    return tone, ' <span class="dim">&middot;</span> '.join(parts)


# ---------------------------------------------------------------------------
# Comparison tables
# ---------------------------------------------------------------------------
#
# Every comparison table reads the same way: N, then a group of columns per
# measure -- Mean (real, prod, then each stream), Gap (prod, then each
# stream), Brier (prod, then each stream) -- with each stream on the rows it
# was priced on. Streams whose N, real and prod agree on every row of a table
# share one N, Real and Prod column; where they differ, those columns repeat
# once per set of streams that shares them, labelled with the streams.

MEASURES = {
    # key: (group title, prod field, stream field, format, colouring)
    "mean": ("Mean", "prod", "cand", ".3f", None),
    "gap": ("Gap", "prod_gap", "cand_gap", "+.3f", "gap"),
    "brier": ("Brier", "prod_brier", "cand_brier", ".4f", "lower"),
    "line": ("Line error (points)", "prod_line", "cand_line", ".3f", "lower"),
    "same": ("Same line as prod", None, "same", "pct", None),
}


def _calibration_rec(r):
    """A calibration cell (cross-section, axis, pre-match) as a table record."""
    if not r or not r.get("n"):
        return None
    return {"n": r["n"], "real": r.get("realized"), "prod": r.get("prod_predicted"),
            "cand": r.get("candidate_predicted"), "prod_gap": r.get("prod_gap"),
            "cand_gap": r.get("candidate_gap"), "prod_brier": r.get("prod_brier"),
            "cand_brier": r.get("candidate_brier"), "matches": r.get("matches")}


def _brier_rec(t):
    """A tally (overall, market, selection, day) as a table record."""
    if not t or not t.get("n"):
        return None
    return {"n": t["n"], "prod_brier": t.get("prod_brier"), "cand_brier": t.get("candidate_brier")}


def _line_rec(c):
    """A line-error cell as a table record."""
    if not c or not c.get("n"):
        return None
    return {"n": c["n"], "prod_line": c.get("prod_line_error"),
            "cand_line": c.get("candidate_line_error"), "same": c.get("same_share")}


def _key(v):
    return round(v, 9) if isinstance(v, float) else v


def _groups(rows, n_sides, fields):
    """Sets of streams that share N, real and prod on every row of a table."""
    sigs = [tuple(None if recs[i] is None else tuple(_key(recs[i].get(f)) for f in fields)
                  for _, recs, _ in rows) for i in range(n_sides)]
    groups = []
    for i, sig in enumerate(sigs):
        for g in groups:
            if sigs[g[0]] == sig:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def _fmt(v, spec):
    if v is None:
        return "&mdash;"
    return _pct(v) if spec == "pct" else format(v, spec)


def _columns(sides, groups, measures):
    """(group title, header, cell) for every column; a cell maps a row's records to
    (text, class, sort value)."""
    names = [_name(s) for s in sides]
    many = len(groups) > 1
    tag = (lambda g: f' <span class="dim">&middot; {" ".join(names[i] for i in g)}</span>') if many \
        else (lambda g: "")

    def base(g, recs):
        return next((recs[i] for i in g if recs[i]), None)

    def shared(field, spec, bold=False, ramp=None):
        def cell(recs, g):
            r = base(g, recs)
            v = None if r is None else r.get(field)
            text = _fmt(v, spec)
            return (f"<b>{text}</b>" if bold and v is not None else text,
                    _gap(v, ramp) if ramp and v is not None else "", v)
        return cell

    def stream(i, field, prod_field, spec, colour):
        def cell(recs, g):
            r = recs[i]
            v = None if r is None else r.get(field)
            cls = ""
            if v is not None and colour == "gap":
                cls = _gap(v, PROB_GAP)
            elif v is not None and colour == "lower" and r.get(prod_field) is not None:
                cls = _cls(r[prod_field] - v)
            return _fmt(v, spec), cls, v
        return cell

    cols = [("", "N" + tag(g), lambda recs, g=g, c=shared("n", ","): c(recs, g)) for g in groups]
    for m in measures:
        title, prod_field, field, spec, colour = MEASURES[m]
        if m == "mean":
            cols += [(title, "Real" + tag(g), lambda recs, g=g, c=shared("real", spec, bold=True): c(recs, g))
                     for g in groups]
        if prod_field:
            ramp = PROB_GAP if colour == "gap" else None
            cols += [(title, "Prod" + tag(g), lambda recs, g=g, c=shared(prod_field, spec, ramp=ramp): c(recs, g))
                     for g in groups]
        cols += [(title, names[i], lambda recs, c=stream(i, field, prod_field, spec, colour): c(recs, None))
                 for i in range(len(sides))]
    return cols


def _table(sides, lead, rows, measures, table_class="", table_id=""):
    """A comparison table. `lead` is the leading headers (text, class); `rows` is
    (leading cells html, one record per side or None, row class), or a string for a
    full-width note row."""
    data = [r for r in rows if not isinstance(r, str) and any(r[1])]
    if not data:
        return ""
    fields = ["n"] + [MEASURES[m][1] for m in measures if MEASURES[m][1]]
    if "mean" in measures:
        fields.append("real")
    groups = _groups(data, len(sides), fields)
    cols = _columns(sides, groups, measures)
    top = "".join(f'<th class="{c}"></th>' for _, c in lead)
    i = 0
    while i < len(cols):
        j = i
        while j < len(cols) and cols[j][0] == cols[i][0]:
            j += 1
        title = cols[i][0] or ("N" if j - i > 1 else "")
        top += f'<th colspan="{j - i}" class="grp">{title}</th>'
        i = j
    starts = {i for i in range(len(cols)) if i == 0 or cols[i][0] != cols[i - 1][0]}
    grp = ' class="grp"'
    leaf = "".join(f'<th class="{c}">{t}</th>' for t, c in lead) + "".join(
        f"<th{grp if i in starts else ''}>{h}</th>" for i, (_, h, _) in enumerate(cols))
    body = []
    for r in rows:
        if isinstance(r, str):
            body.append(r)
            continue
        cells_html, recs, tr_class = r
        if not any(recs):
            continue
        tds = []
        for i, (_, _, cell) in enumerate(cols):
            text, cls, v = cell(recs)
            cls = f"{'grp ' if i in starts else ''}{cls}".strip()
            dv = "" if v is None else f' data-v="{v}"'
            tds.append(f'<td class="{cls}"{dv}>{text}</td>' if cls else f"<td{dv}>{text}</td>")
        body.append(f'<tr class="{tr_class}">{cells_html}{"".join(tds)}</tr>' if tr_class
                    else f"<tr>{cells_html}{''.join(tds)}</tr>")
    ident = f' id="{table_id}"' if table_id else ""
    cls = f' class="{table_class}"' if table_class else ""
    return f"""<table{cls}{ident}>
        <thead><tr>{top}</tr><tr>{leaf}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>"""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _headline(sides):
    """Brier at prod's line and line error at each side's own line, over every market."""
    brier = _table(sides, [("At prod's line", "")],
                   [("<th>All markets</th>",
                     [_brier_rec(s["prob_full"]["summary"]["same_line"]["overall"]) for s in sides],
                     "")], ["brier"])
    line = _table(sides, [("At its own line", "")],
                  [("<th>All lines</th>",
                    [_line_rec(s["line_full"]["line_error"]["all"].get("all")) for s in sides], "")],
                  ["line", "same"])
    return f"""
    <section class="panel" id="directional">
      <h2>Directional calibration</h2>
      {brier}
      {line}
    </section>"""


def _market_block(sides):
    """The same two readings by market."""
    rows = [(f"<th>{MARKET_TITLES[market]}</th>",
             [_brier_rec(s["prob_full"]["summary"]["same_line"]["by_market"].get(market))
              for s in sides], "") for market in MARKET_ORDER]
    brier = _table(sides, [("At prod's line", "")], rows, ["brier"])
    line_rows = [(f"<th>{MARKET_TITLES[market]}</th>",
                  [_line_rec(s["line_full"]["line_error"]["market"].get(market)) for s in sides], "")
                 for market in LINE_MARKETS]
    line = _table(sides, [("At its own line", "")], line_rows, ["line", "same"])
    return f"""
    <section class="panel" id="markets">
      <h2>By market</h2>
      {brier}
      {line}
    </section>"""


def _axis_section(sides, index):
    """One axis (score difference, quarter or possession) on its own: the
    calibration at prod's line by cell and selection, and each side's line
    error by cell."""
    axis0 = sides[0]["prob_full"]["axes"][index]
    tables = [s["prob_full"]["axes"][index]["probability"] for s in sides]
    keys = set().union(*tables)
    market_of = {}
    for t in tables:
        for k, r in t.items():
            market_of[k[1]] = r["market"]
    ids = sorted(market_of, key=lambda i: (MARKET_ORDER.index(market_of[i]), i))
    prob_rows = []
    for cell_label in axis0["order"]:
        for market_id in ids:
            if (cell_label, market_id) not in keys:
                continue
            row = [t.get((cell_label, market_id)) for t in tables]
            base = next((r for r in row if r and r["n"]), None)
            if base is None:
                continue
            prob_rows.append((f"<th>{html.escape(str(cell_label))}</th>"
                              f"<td>{MARKET_TITLES[base['market']]}</td>"
                              f"<td class=\"dim\">{base['selection']}</td>",
                              [_calibration_rec(r) for r in row], ""))
    line_tables = [s["line_full"]["axes"][index].get("line_error", {}) for s in sides]
    line_rows = [(f"<th>{html.escape(str(cell_label))}</th>",
                  [_line_rec(t.get(cell_label)) for t in line_tables], "")
                 for cell_label in axis0["order"]]
    name = html.escape(axis0["name"])
    prob = _table(sides, [(name, ""), ("Market", ""), ("Sel", "")], prob_rows,
                  ["mean", "gap", "brier"], "sortable")
    line = _table(sides, [(name, "")], line_rows, ["line", "same"])
    return f"""
    <section class="panel" id="axis{index}">
      <h2>{name}</h2>
      <h3>At prod's line</h3>
      <div class="scroll">{prob or '<p class="dim">no pairs</p>'}</div>
      <h3>Line error (points)</h3>
      {line or '<p class="dim">no pairs</p>'}
    </section>"""


def _full_cell(sides):
    """Quarter x possession x score difference, all three markets, at
    prod's line. Rows too thin to read are dimmed rather than dropped."""
    first = sides[0]["prob_full"]
    tables = [s["prob_full"]["full_cell"] for s in sides]
    order = first["full_cell_order"]
    for t in tables[1:]:
        order = order + [k for k in sorted({k[0] for k in t}, key=buckets.sort_key) if k not in order]
    rows = []
    for cell_label in order:
        for market in MARKET_ORDER:
            row = [t.get((cell_label, market)) for t in tables]
            base = next((r for r in row if r and r["n"]), None)
            if base is None:
                continue
            score, quarter, possession = (str(part) for part in cell_label)
            key = buckets.sort_key(cell_label)
            sparse = base["matches"] < config.MIN_CELL_MATCHES
            rows.append((f'<th class="ax" data-v="{key[0]}">{html.escape(score)}</th>'
                         f'<td class="ax" data-v="{key[1]}">{html.escape(quarter)}</td>'
                         f'<td class="ax" data-v="{key[2]}">{html.escape(possession)}</td>'
                         f"<td>{MARKET_TITLES[market]}</td>"
                         f'<td class="dim">{base["selection"]}</td>',
                         [_calibration_rec(r) for r in row], "thin" if sparse else ""))
    table = _table(sides, [("Score diff", "ax"), ("Quarter", "ax"), ("Possession", "ax"),
                           ("Market", ""), ("Sel", "")], rows, ["mean", "gap", "brier"],
                   "sortable", "crossTable")
    return f"""
    <section class="panel" id="cross">
      <h2>Cross-section calibration</h2>
      <div class="scroll">{table}</div>
    </section>"""


def _prematch_block(sides):
    """Calibration of the closing price, each candidate that quotes before
    kickoff against prod."""
    summaries = [s["line"].get("prematch") for s in sides]
    summaries = [x if x and x.get("n") else None for x in summaries]
    if not any(summaries):
        return ""
    base = next(x for x in summaries if x)

    def rows_for(field, label_of, sort_key):
        keys = set()
        for x in summaries:
            if x:
                keys |= set(x[field])
        return [(f"<th>{label_of(k)}</th>",
                 [_calibration_rec(x[field].get(k)) if x else None for x in summaries], "")
                for k in sorted(keys, key=sort_key)]

    measures = ["mean", "gap", "brier"]
    selections = _table(sides, [("Selection", "")],
                        rows_for("by_selection", lambda k: f"{k[0]} {k[1]}",
                                 lambda k: (str(k[0]), str(k[1]))), measures)
    lines = _table(sides, [("Line", "")],
                   rows_for("by_line", lambda k: f"{k[0]} {k[1]} {k[2]:+g}",
                            lambda k: (str(k[0]), str(k[1]), k[2] if k[2] is not None else 0)),
                   measures)
    spread = base["spread"]
    bands = "".join(
        f"<tr><th>{b['low']:.2f} to {b['high']:.2f}</th>"
        f"<td>{b['n']:,}</td><td>{_pct(b['share'])}</td></tr>"
        for b in spread["bands"])
    markets_rows = "".join(
        f"<tr><th>{group}</th><td>{s['n']:,}</td>"
        f"<td>{s['mean_distance']:.3f}</td><td>{s['max_distance']:.3f}</td>"
        f"<td>{_pct(s['bands'][0]['share']) if s['bands'] else '&mdash;'}</td></tr>"
        for group, s in base["spread_by_market"].items())
    line_rows = f"""
      <h3>By line</h3>
      {lines}""" if lines else ""
    return f"""
    <section class="panel" id="prematch">
      <h2>Pre-match calibration</h2>
      <div class="cols">
        <div>
          <table>
            <thead><tr><th>|p &minus; 0.500|</th><th>N</th><th>Share</th></tr></thead>
            <tbody>{bands}</tbody>
          </table>
        </div>
        <div>
          <table>
            <thead><tr><th>Market</th><th>N</th><th>Mean |p&minus;.5|</th>
              <th>Max</th><th>Within .02</th></tr></thead>
            <tbody>{markets_rows}</tbody>
          </table>
        </div>
      </div>
      <h3>By selection</h3>
      {selections}{line_rows}
    </section>"""


def _indrive_block(sides):
    """Does each price move the way the play says it should: prod, then
    each candidate, and each candidate head to head with prod."""
    summaries = [s["line"].get("indrive") for s in sides]
    if not any(summaries):
        return ""
    from . import directional, html_indrive
    rows = []

    def stream_rows(label, s):
        out = []
        for scope, key in (("Overall", "overall"), ("Moves of 1 point or more", "material"),
                           ("Inside a drive", "in_drive"), ("At a drive's end", "ending")):
            block = s.get(key)
            if not block or not block["n"]:
                continue
            out.append(f"""<tr>
                <th>{label}</th><td>{scope}</td>
                <td>{block['n']:,}</td><td>{block['decided']:,}</td>
                <td class="{html_indrive._rate_class(block['rate'])}"><b>{_pct(block['rate'])}</b></td>
                <td>{_pct(block['flat_share'])}</td>
            </tr>""")
        return out

    base = next(x for x in summaries if x)
    prod = base["result"]["streams"].get(directional.PROD)
    if prod and prod["overall"]["n"]:
        rows += stream_rows("Prod", prod)
    h2h_rows = []
    for side, x in zip(sides, summaries):
        if not x:
            continue
        cand = x["result"]["streams"].get(directional.CANDIDATE)
        if cand and cand["overall"]["n"]:
            rows += stream_rows(_name(side), cand)
        h = x["result"]["head_to_head"].get("all") or {}
        h2h_rows.append(f"""<tr><th>{_name(side)}</th>
          <td>{h.get('pairs', 0):,}</td><td>{h.get('decided', 0):,}</td>
          <td class="{html_indrive._rate_class(h.get('rate'))}"><b>{_pct(h.get('rate'))}</b></td>
          <td>{h.get('ties', 0):,}</td><td>{h.get('both_flat', 0):,}</td></tr>""")
    drive_rows = []
    for outcome, row in base["census"].items():
        if not row["n"]:
            continue
        drive_rows.append(f"""<tr>
            <th>{outcome}</th><td>{row['n']:,}</td>
            <td>{_pct(row['n'] / base['drives'])}</td>
            <td>{row['points']:,}</td>
        </tr>""")
    return f"""
    <section class="panel" id="indrive">
      <h2>In-drive reaction</h2>
      <div class="cols">
        <div>
          <table>
            <thead><tr><th>Stream</th><th>Scope</th><th>Moves</th><th>Decided</th>
              <th>Right</th><th>Flat</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div>
          <table>
            <thead><tr><th>Drive outcome</th><th>Drives</th><th>Share</th>
              <th>Points</th></tr></thead>
            <tbody>{''.join(drive_rows)}</tbody>
          </table>
        </div>
      </div>
      <h3>Head to head</h3>
      <table>
        <thead><tr><th>Candidate</th><th>Pairs</th><th>Decided</th><th>Candidate right</th>
          <th>Ties</th><th>Both flat</th></tr></thead>
        <tbody>{''.join(h2h_rows)}</tbody>
      </table>
    </section>"""




# ---------------------------------------------------------------------------
# Additional checks
# ---------------------------------------------------------------------------

def _run_block(sides, dropped):
    first = sides[0]
    stats, header = first["line"]["stats"], first["line"]["header"]
    exact = stats.get("exact_message_pair", 0)
    offset = stats.get("offset_message_pair", 0)
    rows = [
        ("Checks", _checks_summary(first["line_full"], first["line"]["scan"])),
        ("Window from", html.escape(str(config.CUTOFF_START))),
        ("Window to", html.escape(str(config.CUTOFF_END or "latest"))),
        ("Matches with pairs", f"{len({p.match_code for p in first['line_pairs']}):,}"),
        ("Of settled matches", f"{header.get('paired_matches', 0):,}"),
        ("Drive snapshots", f"{stats.get('snapshots', 0):,}"),
        ("Exact-message pairing", _pct(exact / (exact + offset)) if exact + offset else "&mdash;"),
        ("Pairs at prod's line", f"{len(first['prob_pairs']):,}"),
        ("Pairs at own lines", f"{len(first['line_pairs']):,}"),
    ]
    if len(sides) > 1:
        rows.append(("Dropped for want of every candidate",
                     f"{dropped.get('prob', 0):,} / {dropped.get('line', 0):,}"))
    for s in sides:
        lines = s["line_full"]["summary"]["lines"]
        rows.append((f"{_name(s)}: same line as prod", _pct(lines["same_rate"])))
    body = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return f"""
    <section class="panel">
      <h2>Run</h2>
      <table>
        <thead><tr><th>What</th><th>Value</th></tr></thead>
        <tbody>{body}</tbody>
      </table>
    </section>"""


# How far a stream's share of lines within one / two scores sits from the
# share of games that really had that little still to come, in points of
# share; and how far its over rate sits from its own priced P(over).
SHARE_GAP = (0.03, 0.06, 0.10, 0.15)


def _share_cell(value, real):
    """A stream's share with its gap to real's beside it, coloured by the gap."""
    if value is None:
        return "<td>&mdash;</td>"
    if real is None:
        return f"<td>{_pct(value, '.0f')}</td>"
    gap = value - real
    return (f'<td class="{_gap(gap, SHARE_GAP)}">{_pct(value, ".0f")}'
            f'<span class="pp">{100 * gap:+.0f}</span></td>')


def _over_gap_cell(over, priced, n=None):
    if over is None or priced is None:
        return "<td>&mdash;</td>"
    gap = over - priced
    count = f' <span class="dim">({n:,})</span>' if n else ""
    return f'<td class="{_gap(gap, PROB_GAP)}">{100 * gap:+.0f}{count}</td>'


def _quarter_order(label):
    """Q1..Q4, then overtime."""
    return {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "OT": 5}.get(label, 6)


def _totals_reach_block(sides):
    """The totals line test: how often each stream's line sits within one and
    two scores of the points already scored, beside how often the rest of
    the game really produced that little (totals_reach) -- overall, by
    quarter and game state, and each stream's over rate against its price."""
    per = [totals_reach.summarise(s["line_pairs"]) for s in sides]
    if not per or not per[0]["real"]["n"]:
        return ""
    names = ["Prod"] + [_name(s) for s in sides]
    streams = lambda r: [per[0]["prod"]] + [x["candidate"] for x in per]
    rows = []
    for label, key in (("Within 1 score", totals_reach.WITHIN_ONE),
                       ("Within 2 scores", totals_reach.WITHIN_TWO + "_or_less")):
        real = per[0]["real"][key]
        cells = "".join(_share_cell(st[key], real) for st in streams(per))
        rows.append(f"<tr><th>{label}</th><td><b>{_pct(real, '.0f')}</b></td>{cells}</tr>")
    head = "".join(f"<th>{n}</th>" for n in names)
    summary = f"""
      <table class="reach">
        <thead><tr><th>Line above the score</th><th>Real</th>{head}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="dim">{per[0]['real']['n']:,} snapshots</p>"""
    return f"""
  <section class="panel" id="totals-reach">
    <h2>Totals line within 1 and 2 scores</h2>
    {summary}
    {_totals_reach_breakdown(sides)}
    {_totals_over_block(sides, per)}
  </section>"""


def _totals_reach_breakdown(sides):
    """By quarter and game state: real's shares within one and two scores,
    and each stream's, with its gap to real."""
    per = [totals_reach.breakdown(s["line_pairs"]) for s in sides]
    keys = sorted(per[0], key=lambda k: (_quarter_order(k[0]), totals_reach.STATES.index(k[1])))
    if not keys:
        return ""
    names = ["Prod"] + [_name(s) for s in sides]
    group = (f'<th colspan="{1 + len(names)}" class="grp">Within 1 score</th>'
             f'<th colspan="{1 + len(names)}" class="grp">Within 2 scores</th>')
    sub = ('<th class="grp">Real</th>' + "".join(f"<th>{n}</th>" for n in names)) * 2
    rows = []
    for key in keys:
        first = per[0][key]
        lines = [first["prod"]] + [p.get(key, {}).get("candidate", (None,) * 4) for p in per]
        cells = ""
        for k in (0, 1):
            real = first["real"][k]
            cells += f'<td class="grp"><b>{_pct(real, ".0f")}</b></td>'
            cells += "".join(_share_cell(line[k], real) for line in lines)
        whole = key[1] == totals_reach.ALL
        label = "all" if whole else html.escape(key[1])
        tr = '<tr class="subtotal">' if whole else "<tr>"
        rows.append(f'{tr}<th>{key[0]}</th><td class="state">{label}</td>'
                    f'<td>{first["n"]:,}</td>{cells}</tr>')
    return f"""
    <h3>By quarter and game state</h3>
    <table class="reach">
      <thead><tr><th rowspan="2">Quarter</th><th rowspan="2" class="state">State</th><th rowspan="2">N</th>{group}</tr>
        <tr>{sub}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _totals_over_block(sides, per):
    """Each stream's over rate at its own line against its own priced
    P(over): overall by where the line sits, and by quarter."""
    names = ["Prod"] + [_name(s) for s in sides]
    streams = [per[0]["prod"]] + [x["candidate"] for x in per]
    rows = []
    for label, key in (("Within 1 score", totals_reach.WITHIN_ONE),
                       ("1 to 2 scores", totals_reach.WITHIN_TWO),
                       ("More than 2 scores", totals_reach.BEYOND)):
        rows.append(f"<tr><th>{label}</th>" + "".join(
            _over_gap_cell(st["over"][key][0], st["priced"][key], st["over"][key][1])
            for st in streams) + "</tr>")
    by_q = [totals_reach.breakdown(s["line_pairs"]) for s in sides]
    quarters = sorted((k for k in by_q[0] if k[1] == totals_reach.ALL), key=lambda k: _quarter_order(k[0]))
    for n, key in enumerate(quarters):
        lines = [by_q[0][key]["prod"]] + [b.get(key, {}).get("candidate", (None,) * 4) for b in by_q]
        tr = '<tr class="split">' if n == 0 else "<tr>"
        rows.append(f'{tr}<th>{key[0]}</th>'
                    + "".join(_over_gap_cell(l[2], l[3]) for l in lines) + "</tr>")
    head = "".join(f"<th>{n}</th>" for n in names)
    return f"""
    <h3>Over at the line against priced</h3>
    <table class="reach">
      <thead><tr><th>Line above the score / quarter</th>{head}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def _checks_summary(report, scan=None):
    """Whether the diagnostics passed, in a few words."""
    issues = []
    if scan and scan.get("matches_flipped"):
        issues.append(f"{scan['matches_flipped']:,} matches with flipped handles")
    state = report.get("market_state") or {}
    if state.get("share") and state["share"] > 0.05:
        issues.append(f"{_pct(state['share'])} of pairs not live")
    anchor = report.get("anchor") or {}
    if anchor.get("share_clean") is not None and anchor["share_clean"] < 0.9:
        issues.append(f"only {_pct(anchor['share_clean'])} of snapshots on "
                      "a drive's opening 1st &amp; 10")
    for market, row in (report.get("complement") or {}).items():
        both = row.get("both_sides") or 0
        if both and row.get("outcomes_partition", 0) / both < 0.999:
            issues.append(f"{market} does not partition")
    spread = report.get("spread") or {}
    verdict = (spread.get("verdict") or ("unclear", ""))[0]
    if verdict not in ("unclear", config.SPREAD_RESOLUTION):
        issues.append(f"spread reads {verdict}, configured {config.SPREAD_RESOLUTION}")
    if issues:
        return '<span class="bad">' + "; ".join(issues) + "</span>"
    return '<span class="good">all pass</span>'


def _handle_block(scan):
    """Did PLAYER_1 / PLAYER_2 stay pinned to the same team?"""
    if not scan or not scan["matches"]:
        return """
    <section class="panel">
      <h2>Handle check</h2>
    </section>"""
    kind_rows = []
    for kind in handles.KIND_ORDER:
        events = scan["counts"][kind]
        if not events:
            continue
        flips = "yes" if kind in handles.FLIP_KINDS else "no"
        kind_rows.append(f"""<tr>
            <th>{handles.KIND_TITLES[kind]}</th>
            <td>{events:,}</td>
            <td>{scan['matches_by_kind'][kind]:,}</td>
            <td class="{'bad' if kind in handles.FLIP_KINDS else 'dim'}">{flips}</td>
        </tr>""")
    event_rows = []
    for anomaly in scan["anomalies"][:200]:
        if anomaly.detail is not None:
            evidence = html.escape(anomaly.detail)
        else:
            joiner = "vs" if anomaly.is_final_check else "&rarr;"
            evidence = (f"{anomaly.before[0]}&ndash;{anomaly.before[1]} {joiner} "
                        f"{anomaly.after[0]}&ndash;{anomaly.after[1]}")
        event_rows.append(f"""<tr>
            <th>{html.escape(anomaly.match_code)}</th>
            <td>{anomaly.where}</td>
            <td class="{'bad' if anomaly.is_flip else 'dim'}">{handles.KIND_TITLES[anomaly.kind]}</td>
            <td>{evidence}</td>
        </tr>""")
    detail = "" if not event_rows else f"""
      <h3>Flagged events</h3>
      <div class="scroll">
      <table>
        <thead><tr><th>Match</th><th>Where</th><th>Kind</th><th>Evidence</th></tr></thead>
        <tbody>{''.join(event_rows)}</tbody>
      </table>
      </div>"""
    flipped = scan["matches_flipped"]
    verdict = ('<span class="good">clean</span>' if not flipped else
               f'<span class="bad">{flipped:,} of {scan["matches"]:,} flipped</span>')
    summary = [("Handles", verdict),
               ("Matches scanned", f"{scan['matches']:,}"),
               ("Flipped matches", "excluded" if config.EXCLUDE_FLIPPED_MATCHES
                else ("still in the numbers" if flipped else "none"))]
    summary_rows = "".join(f"<tr><th>{a}</th><td>{b}</td></tr>" for a, b in summary)
    return f"""
    <section class="panel">
      <h2>Handle check</h2>
      <div class="cols">
        <div><table><tbody>{summary_rows}</tbody></table></div>
        <div>
          <table>
            <thead><tr><th>Kind</th><th>Events</th><th>Matches</th><th>Flip</th></tr></thead>
            <tbody>{''.join(kind_rows) or '<tr><td colspan="4" class="dim">nothing flagged</td></tr>'}</tbody>
          </table>
        </div>
      </div>
      {detail}
    </section>"""


ANCHOR_TITLES = {
    drives.FIRST_DOWN: "Opening 1st &amp; 10",
    drives.MID_DRIVE: "Mid-drive snap",
    drives.NO_SNAP: "No real snap",
    drives.PLAY_OVER: "Every PLAY_OVER (SCOUTING_FULL)",
}


def _anchor_block(report):
    """Where in its drive each snapshot landed."""
    a = report.get("anchor")
    if not a or not a["pairs"]:
        return ""
    rows = []
    for kind in (drives.FIRST_DOWN, drives.PLAY_OVER, drives.MID_DRIVE, drives.NO_SNAP):
        n = a["counts"].get(kind, 0)
        if not n:
            continue
        good = kind in (drives.FIRST_DOWN, drives.PLAY_OVER)
        rows.append(f"""<tr>
            <th>{ANCHOR_TITLES[kind]}</th>
            <td>{n:,}</td>
            <td class="{'good' if good else 'bad'}">{_pct(n / a['pairs'])}</td>
        </tr>""")
    quarter_rows = []
    for key in sorted(a["by_quarter"]):
        total, off = a["by_quarter"][key]
        if not total:
            continue
        quarter_rows.append(f"""<tr>
            <th>{html.escape(str(key))}</th><td>{total:,}</td><td>{off:,}</td>
            <td class="{'bad' if off / total > 0.1 else ''}">{_pct(off / total)}</td>
        </tr>""")
    return f"""
    <section class="panel">
      <h2>Snapshot anchor</h2>
      <div class="cols">
        <div>
          <h3>Where it landed</h3>
          <table>
            <thead><tr><th>Anchor</th><th>Pairs</th><th>Share</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div>
          <h3>By quarter</h3>
          <table>
            <thead><tr><th>Quarter</th><th>Pairs</th><th>Off anchor</th>
              <th>Share</th></tr></thead>
            <tbody>{''.join(quarter_rows)}</tbody>
          </table>
        </div>
      </div>
    </section>"""


def _market_state_block(sides):
    """Non-live quotes, per candidate pass."""
    rows, state_rows = [], []
    for s in sides:
        state = s["line_full"].get("market_state")
        if not state or not state["pairs"]:
            continue
        rows.append(f"""<tr><th>{_name(s)}</th>
            <td>{state['pairs']:,}</td><td>{state['not_live']:,}</td>
            <td class="{'bad' if (state['share'] or 0) > 0.05 else ''}">{_pct(state['share'])}</td>
            <td>{state['prod_only']:,}</td><td>{state['candidate_only']:,}</td>
            <td>{state['both']:,}</td></tr>""")
        for (stream, label), n in sorted(state["by_state"].items(), key=lambda kv: -kv[1]):
            who = "Prod" if stream == "prod" else _name(s)
            state_rows.append(f"""<tr><td class="dim">{_name(s)}</td><td>{who}</td>
                <th>{html.escape(label)}</th><td>{n:,}</td></tr>""")
    if not rows:
        return ""
    by_state = "" if not state_rows else f"""
      <h3>By state</h3>
      <table>
        <thead><tr><th>Pass</th><th>Stream</th><th>Status / active</th><th>Pairs</th></tr></thead>
        <tbody>{''.join(state_rows)}</tbody>
      </table>"""
    return f"""
    <section class="panel">
      <h2>Market state</h2>
      <table>
        <thead><tr><th>Candidate</th><th>Pairs</th><th>Not live</th><th>Share</th>
          <th>Prod only</th><th>Candidate only</th><th>Both</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>{by_state}
    </section>"""


def _selection_block(sides):
    """Brier at prod's line per selection: both sides of every market."""
    first = sides[0]["prob_full"]["summary"]["same_line"].get("selections", {})
    ids = set(first)
    for s in sides[1:]:
        ids |= set(s["prob_full"]["summary"]["same_line"].get("selections", {}))
    info = {}
    for s in sides:
        for m, row in s["prob_full"]["summary"]["same_line"].get("selections", {}).items():
            info.setdefault(m, row)
    rows = []
    for market_id in sorted(ids, key=lambda m: (MARKET_ORDER.index(info[m]["market"]), m)):
        row = info[market_id]
        recs = [_brier_rec((s["prob_full"]["summary"]["same_line"].get("selections", {})
                            .get(market_id) or {}).get("tally")) for s in sides]
        rows.append((f"<th>{MARKET_TITLES[row['market']]}</th><td>{row['selection']}</td>"
                     f"<td class=\"dim\">{'&check;' if row['canonical'] else ''}</td>"
                     f'<td data-v="{market_id}" class="dim">{market_id}</td>', recs, ""))
    table = _table(sides, [("Market", ""), ("Sel", ""), ("Used", ""), ("ID", "")], rows, ["brier"])
    return f"""
    <section class="panel">
      <h2>By selection</h2>
      <div class="scroll">{table or '<p class="dim">no pairs</p>'}</div>
    </section>"""


def _both_sides_block(sides):
    tables = [s["prob_full"]["both_sides"] for s in sides]
    ids = set().union(*tables)
    info = {}
    for t in tables:
        for m, row in t.items():
            info.setdefault(m, row)
    by_market = {}
    for market_id in sorted(ids):
        by_market.setdefault(info[market_id]["market"], []).append(market_id)
    rows = []
    for market in MARKET_ORDER:
        group = by_market.get(market, [])
        for market_id in group:
            row = info[market_id]
            rows.append((f"<th>{MARKET_TITLES[market]}</th><td>{row['selection']}</td>"
                         f"<td class=\"dim\">{'&check;' if row['canonical'] else ''}</td>",
                         [_calibration_rec(t.get(market_id)) for t in tables], ""))
        if len(group) == 2:
            sums = []
            for t in tables:
                pair = [t.get(m) for m in group]
                if all(pair):
                    sums.append(sum(r["realized"] for r in pair if r["realized"] is not None))
            ok = bool(sums) and all(abs(x - 1.0) < 1e-9 for x in sums)
            realized = ", ".join(f"{x:.3f}" for x in sums) or "&mdash;"
            rows.append(f"""<tr class="subtotal"><th></th>
                <td colspan="100" class="{'good' if ok else 'bad'}">realized sums to {realized}:
                  {'mirror ok' if ok else 'NOT MIRRORED'}</td></tr>""")
    table = _table(sides, [("Market", ""), ("Sel", ""), ("Used", "")], rows, ["mean", "gap"])
    return f"""
    <section class="panel">
      <h2>Mirror check</h2>
      {table}
    </section>"""


def _daily(sides):
    dailies = [s["prob_full"].get("daily") or {} for s in sides]
    days = sorted(set().union(*dailies))
    if not days:
        return ""
    rows = [(f"<th>{html.escape(day)}</th>",
             [_brier_rec({"n": d[day]["pairs"], "prod_brier": d[day].get("prod_brier"),
                          "candidate_brier": d[day].get("candidate_brier")}) if day in d else None
              for d in dailies], "")
            for day in days]
    return f"""
    <section class="panel" id="daily">
      <h2>By day</h2>
      {_table(sides, [("Day", "")], rows, ["brier"])}
    </section>"""




def _integrity_block(sides):
    """Both sides of every market, per candidate pass: do the two
    probabilities sum to one, and do the two outcomes partition?"""
    rows = []
    for s in sides:
        report = s["line_full"]
        lines = report["summary"]["lines"]
        comp = report["complement"]
        for market in MARKET_ORDER:
            bucket = lines["by_market"].get(market)
            if not bucket or not bucket["n"]:
                continue
            c = comp.get(market, {})
            n = c.get("both_sides") or 0
            partition = (c.get("outcomes_partition", 0) / n) if n else None
            rows.append(f"""<tr><td>{_name(s)}</td>
                <th>{MARKET_TITLES[market]}</th>
                <td>{bucket['n']:,}</td>
                <td>{_pct(bucket['same'] / bucket['n'])}</td>
                <td>{_n(c.get('prob_sum_mean'), '.4f')}</td>
                <td class="{'' if partition is None or partition > 0.999 else 'bad'}">{_pct(partition)}</td>
                <td>{c.get('both_won', 0):,} / {c.get('both_lost', 0):,}</td>
            </tr>""")
    spread = sides[0]["line_full"]["spread"]
    spread_body = ""
    if spread["n"]:
        verdict, _ = spread["verdict"]
        spread_rows = [("Both sides present", f"{spread['n']:,}"),
                       ("Probabilities summed", _n(spread["prob_sum_mean"], ".4f")),
                       ("Lines mirrored", _pct(spread["lines_mirrored"] / spread["n"])),
                       ("Lines equal", _pct(spread["lines_equal"] / spread["n"])),
                       ("Literal reading partitions", _pct(spread["literal_partition"] / spread["n"])),
                       ("Verdict", f"<b>{verdict.upper()}</b>")]
        spread_body = f"""
      <h3>Spread reading</h3>
      <table><tbody>{''.join(f'<tr><th>{a}</th><td>{b}</td></tr>' for a, b in spread_rows)}</tbody></table>"""
    return f"""
    <section class="panel">
      <h2>Integrity checks</h2>
      <table>
        <thead><tr><th>Candidate</th><th>Market</th><th>Pairs</th><th>Same line</th>
          <th>P(both sides)</th><th>Partition</th><th>Both won / lost</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>{spread_body}
    </section>"""


# ---------------------------------------------------------------------------
# Every pair
# ---------------------------------------------------------------------------

def _down(pair):
    if pair.down_number is None and pair.distance is None:
        return "&mdash;"
    down = "?" if pair.down_number is None else pair.down_number
    distance = "?" if pair.distance is None else pair.distance
    return f"{down}&amp;{distance}"


def _joined(sides):
    """Every common snapshot once: prod's quote, and each candidate's quote
    as it was published (own line) with its probability at prod's line
    beside it. Widest disagreement with prod first."""
    from .multi import pair_key
    line_maps = [{pair_key(p): p for p in s["line_pairs"]} for s in sides]
    prob_maps = [{pair_key(p): p for p in s["prob_pairs"]} for s in sides]
    keys = list(line_maps[0])

    def spread_of(key):
        base = line_maps[0][key]
        gaps = [abs(m[key].candidate_probability - m[key].prod_probability)
                for m in prob_maps if key in m]
        return (-(max(gaps) if gaps else 0.0), base.match_code, base.drive_number, base.market_id)

    keys.sort(key=spread_of)
    return keys, line_maps, prob_maps


def _live_state(p_prod_live, not_live_names):
    if not not_live_names and p_prod_live:
        return "live"
    return ", ".join((["prod"] if not p_prod_live else []) + not_live_names)


def _won(outcome):
    return None if outcome is None else (1 if outcome else 0)


def _r(v, digits=4):
    return None if v is None else round(float(v), digits)


# The pair table is drawn in the browser from compact rows: tens of
# thousands of snapshots times a group of columns per candidate is too much
# markup to ship as HTML. Each column is (header, kind, starts a group);
# the kinds are formatted by report.js (PAIR_KINDS).
def _pair_data(sides):
    keys, line_maps, prob_maps = _joined(sides)
    base_map = line_maps[0]
    dual = [s["line"] is not s["prob"] for s in sides]
    last_quote = {}
    for p in base_map.values():
        if p.publish_time is None:
            continue
        seen = last_quote.get(p.match_code)
        if seen is None or p.publish_time > seen:
            last_quote[p.match_code] = p.publish_time
    columns = [("Time", "t", 0), ("Match", "t", 0), ("Drive", "i", 0), ("Msg", "i", 0),
               ("&plusmn;Msg", "mg", 0), ("Qtr", "t", 0), ("Home", "i", 0), ("Away", "i", 0),
               ("Diff", "sd", 0), ("Poss", "t", 0), ("Field", "i", 0), ("D&amp;D", "dd", 0),
               ("To end", "te", 0), ("Market", "t", 0), ("Sel", "t", 0), ("Result", "b", 0),
               ("Prod line", "l", 1), ("Prod prob", "p4", 0), ("Prod won", "o", 0),
               ("Prod err", "p4", 0)]
    for s in sides:
        n = _name(s)
        columns += [(f"{n} line", "l", 1), (f"{n} prob", "p4", 0),
                    (f"{n} at prod&#39;s line", "p4", 0), (f"{n} &Delta;prob", "dp", 0),
                    (f"{n} won", "o", 0), (f"{n} err", "p4", 0), ("Closer", "c", 0)]
    columns.append(("Live", "live", 0))

    rows = []
    for key in keys:
        p = base_map[key]
        cand_dead = [s["name"] for s, m in zip(sides, line_maps)
                     if key in m and not m[key].candidate_live]
        end = last_quote.get(p.match_code)
        to_end = (None if end is None or p.publish_time is None
                  else round((end - p.publish_time).total_seconds()))
        prod_error = None if (p.prod_outcome is None or not p.prod_live) else \
            abs(p.prod_probability - (1.0 if p.prod_outcome else 0.0))
        down = None
        if p.down_number is not None or p.distance is not None:
            down = [f"{'?' if p.down_number is None else p.down_number}&"
                    f"{'?' if p.distance is None else p.distance}",
                    p.down_number if p.down_number is not None else "",
                    0 if p.anchor in (drives.FIRST_DOWN, drives.PLAY_OVER) else 1]
        quarter = ("Q" + str(p.period_number) if p.period_number and p.period_number <= 4
                   else ("OT" if p.period_number else "?"))
        row = [None if p.publish_time is None else str(p.publish_time)[:19], p.match_code,
               p.drive_number, p.message_count, p.message_gap, quarter,
               p.score_p1, p.score_p2, p.score_diff,
               "H" if p.offensive_team == "Home Team" else ("A" if p.offensive_team == "Away Team" else "?"),
               p.field_position, down, to_end,
               MARKET_TITLES.get(markets.market_group(p.market_id), "?"),
               markets.selection_label(p.market_id) or "",
               None if p.realized is None else round(p.realized),
               _r(p.prod_line, 1), _r(p.prod_probability), _won(p.prod_outcome), _r(prod_error)]
        for side, lm, pm, both in zip(sides, line_maps, prob_maps, dual):
            c = lm.get(key)
            if c is None:
                row += [None] * 7
                continue
            dead = c.not_live
            error = None if (c.candidate_outcome is None or not c.candidate_live) else \
                abs(c.candidate_probability - (1.0 if c.candidate_outcome else 0.0))
            # its probability for prod's question: read at prod's line when
            # it has lines of its own, or its quote when that is prod's line
            at_prod = pm.get(key) if both else (c if c.same_line else None)
            q = None if (at_prod is None or dead) else at_prod.candidate_probability
            winner = c.decisive_winner if not dead else None
            closer = (["level", ""] if winner in (None, "tie") else
                      ["prod", "bad"] if winner == "prod" else [side["name"], "good"])
            row += [_r(c.candidate_line, 1), _r(c.candidate_probability), _r(q),
                    None if q is None else _r(q - p.prod_probability), _won(c.candidate_outcome),
                    _r(error), closer]
        row.append(_live_state(p.prod_live, cand_dead))
        rows.append(row)
    return columns, rows


def _pair_table(sides):
    import json
    columns, rows = _pair_data(sides)
    grp = ' class="grp"'
    head = "".join(f"<th{grp if g else ''}>{h}</th>" for h, _, g in columns)
    spec = json.dumps({"kinds": [k for _, k, _ in columns], "groups": [g for _, _, g in columns],
                       "prob_delta": PROB_DELTA, "message_gap": MESSAGE_GAP},
                      separators=(",", ":"))
    data = json.dumps(rows, separators=(",", ":")).replace("</", "<\\/")
    return f"""
    <section class="panel" id="pairs">
      <h2>Every pair</h2>
      <div class="filter">
        <input id="pairFilter" type="search" autocomplete="off" spellcheck="false"
               placeholder="filter by match id" aria-label="Filter rows by match id">
        <span id="pairCount" class="count">{len(rows):,} rows</span>
      </div>
      <div class="scroll">
      <table class="datatable" id="pairTable">
        <thead><tr>{head}</tr></thead>
        <tbody></tbody>
      </table>
      </div>
      <script type="application/json" id="pairSpec">{spec}</script>
      <script type="application/json" id="pairData">{data}</script>
    </section>"""


# ---------------------------------------------------------------------------

def _title(sides):
    return "eAMF prod vs " + " &middot; ".join(_name(s) for s in sides)


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.js"),
          encoding="utf-8") as _fh:
    _SCRIPT = _fh.read()


def render_sides(sides, dropped=None, extra=""):
    """The page for one or more candidates (multi.run + multi.build); `extra` is HTML for the
    bottom of the page (the betting simulation)."""
    dropped = dropped or {}
    first = sides[0]
    axes = first["prob_full"]["axes"]
    axis_sections = "".join(_axis_section(sides, i) for i in range(len(axes)))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_title(sides)}</title>
{html_style.CSS}
</head>
<body>
<div class="wrap">
  <header>
    <h1>{_title(sides)}</h1>
  </header>

  <div id="headline">{_headline(sides)}{_market_block(sides)}</div>
  {_full_cell(sides)}
  {axis_sections}
  {_prematch_block(sides)}
  {_indrive_block(sides)}
  {_totals_reach_block(sides)}
  <details class="panel" id="checks">
    <summary>Additional checks</summary>
    {_run_block(sides, dropped)}
    {_handle_block(first['line']['scan'])}
    {_anchor_block(first['line_full'])}
    {_market_state_block(sides)}
    {_selection_block(sides)}
    {_daily(sides)}
    {_integrity_block(sides)}
    {_both_sides_block(sides)}
  </details>
  {_pair_table(sides)}
  {extra}
</div>
{_SCRIPT}
</body>
</html>
"""


def single_side(report, header, stats, pairs, handle_scan, indrive_summary=None,
                prematch_summary=None, name="candidate"):
    """One prod-vs-candidate pass as a side, for callers with one report."""
    passed = {"pairs": pairs, "stats": stats, "header": header, "scan": handle_scan,
              "indrive": indrive_summary, "prematch": prematch_summary}
    return {"name": name, "stream": name, "line": passed, "prob": passed,
            "prob_pairs": pairs, "line_pairs": pairs,
            "prob_full": report, "line_full": report}


def render(report, header, stats, pairs, handle_scan,
           indrive_summary=None, prematch_summary=None):
    """One candidate: the same page with a single group of columns."""
    from . import labels
    return render_sides([single_side(report, header, stats, pairs, handle_scan,
                                     indrive_summary, prematch_summary,
                                     labels.candidate_label())])


def write_sides(path, sides, dropped=None, extra=""):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_sides(sides, dropped, extra))
    return path


def write(path, report, header, stats, pairs, handle_scan,
          indrive_summary=None, prematch_summary=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(report, header, stats, pairs, handle_scan,
                        indrive_summary, prematch_summary))
    return path
