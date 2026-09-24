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

from . import buckets, config, drives, handles, html_style, markets

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


def _name(side):
    return html.escape(side["name"])


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
# Sections
# ---------------------------------------------------------------------------

def _delta_cells(value, delta, spec_value, spec_delta, extra=""):
    """A candidate's value, its difference from prod (positive: the
    candidate did better), the 95% CI and p."""
    d = delta or {}
    return (f'<td class="grp{extra}">{_n(value, spec_value)}</td>'
            f'<td class="{_cls(d.get("mean"))}">{_n(d.get("mean"), spec_delta)}</td>'
            f'<td class="dim">{_ci(d, spec_delta)}</td>'
            f'<td>{_p(d.get("p_value"))}</td>')


def _group_head(sides, cols):
    """One group of headers per candidate, the first carrying its name."""
    return "".join(
        "".join(f'<th class="grp">{_name(s)} {c}</th>'.replace(" </th>", "</th>") if i == 0
                else f"<th>{c}</th>" for i, c in enumerate(cols))
        for s in sides)


def _headline(sides):
    """The comparison in five rows: Brier at prod's line, line error at each
    side's own line, how often the lines matched, and the match votes."""
    first = sides[0]
    same0 = first["prob_full"]["summary"]["same_line"]
    le0 = first["line_full"]["line_error"]["all"].get("all")

    rows = []
    cells = "".join(_delta_cells(s["prob_full"]["summary"]["same_line"]["overall"]["candidate_brier"],
                                 s["prob_full"]["summary"]["same_line"]["brier"], ".4f", "+.4f")
                    for s in sides)
    rows.append(f"""<tr><th>Brier at prod's line</th>
        <td>{same0['overall']['n']:,}</td><td>{_n(same0['overall']['prod_brier'], '.4f')}</td>{cells}</tr>""")

    if le0:
        cells = ""
        for s in sides:
            c = s["line_full"]["line_error"]["all"].get("all") or {}
            cells += _delta_cells(c.get("candidate_line_error"),
                                  {"mean": c.get("points_delta"), "ci_low": c.get("ci_low"),
                                   "ci_high": c.get("ci_high"), "p_value": c.get("p_value")},
                                  ".3f", "+.3f")
        rows.append(f"""<tr><th>Line error at own line (points)</th>
            <td>{le0['n']:,}</td><td>{le0['prod_line_error']:.3f}</td>{cells}</tr>""")
        cells = "".join(
            f'<td class="grp">{_pct((s["line_full"]["line_error"]["all"].get("all") or {}).get("same_share"))}</td>'
            + _dash(3) for s in sides)
        rows.append(f"<tr><th>Same line as prod</th><td>{le0['n']:,}</td><td>&mdash;</td>{cells}</tr>")

    def votes_row(label, get):
        cells = ""
        n = None
        for s in sides:
            v = get(s)
            n = n if n is not None else v.get("n_matches")
            cells += (f'<td class="grp {_cls((v.get("candidate_win_rate") or 0.5) - 0.5)}">'
                      f'{v.get("candidate", 0)}&ndash;{v.get("prod", 0)}</td>'
                      f'<td>{_pct(v.get("candidate_win_rate"))}</td><td></td>'
                      f'<td>{_p(v.get("p_value"))}</td>')
        return f"<tr><th>{label}</th><td>{(n or 0):,}</td><td>&mdash;</td>{cells}</tr>"

    rows.append(votes_row("Matches won at prod's line",
                          lambda s: s["prob_full"]["summary"]["same_line"]["votes"]))
    rows.append(votes_row("Matches won, line first",
                          lambda s: s["line_full"]["summary"]["decisive"]["votes"]))
    return f"""
    <section class="panel" id="directional">
      <h2>Directional calibration</h2>
      <table>
        <thead><tr><th>Reading</th><th>N</th><th>Prod</th>{_group_head(sides, ['', '&Delta;', '95% CI', 'p'])}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _market_block(sides):
    rows = []
    first = sides[0]["prob_full"]["summary"]["same_line"]
    for market in MARKET_ORDER:
        tallied = first["by_market"].get(market)
        if not tallied or not tallied["n"]:
            continue
        cells = ""
        for s in sides:
            block = s["prob_full"]["summary"]["same_line"]
            t = block["by_market"].get(market) or {}
            cells += _delta_cells(t.get("candidate_brier"),
                                  block.get("markets", {}).get(market, {}).get("brier"),
                                  ".4f", "+.4f")
        rows.append(f"""<tr><td>Brier at prod's line</td><th>{MARKET_TITLES[market]}</th>
            <td>{tallied['n']:,}</td><td>{tallied['n_matches']:,}</td>
            <td>{_n(tallied['prod_brier'], '.4f')}</td>{cells}</tr>""")
    first_le = sides[0]["line_full"]["line_error"]["market"]
    for market in LINE_MARKETS:
        c0 = first_le.get(market)
        if not c0:
            continue
        cells = ""
        for s in sides:
            c = s["line_full"]["line_error"]["market"].get(market) or {}
            cells += _delta_cells(c.get("candidate_line_error"),
                                  {"mean": c.get("points_delta"), "ci_low": c.get("ci_low"),
                                   "ci_high": c.get("ci_high"), "p_value": c.get("p_value")},
                                  ".3f", "+.3f")
        rows.append(f"""<tr><td>Line error (points)</td><th>{MARKET_TITLES[market]}</th>
            <td>{c0['n']:,}</td><td>{c0['matches']:,}</td>
            <td>{c0['prod_line_error']:.3f}</td>{cells}</tr>""")
    return f"""
    <section class="panel">
      <h2>By market</h2>
      <table>
        <thead><tr><th>Measure</th><th>Market</th><th>N</th><th>Matches</th><th>Prod</th>
          {_group_head(sides, ['', '&Delta;', '95% CI', 'p'])}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _calibration_cells(sides, row):
    """Real, prod and its gap, then each candidate's prediction, gap,
    Brier difference and p -- for one cell, off each side's own table."""
    base = next((r for r in row if r), None)
    if base is None:
        return None
    out = (f'<td data-v="{base["n"]}">{base["n"]:,}</td>'
           f'<td data-v="{base["matches"]}">{base["matches"]:,}</td>'
           f'<td data-v="{base["realized"] or 0}"><b>{_n(base["realized"], ".3f")}</b></td>'
           f'<td data-v="{base["prod_predicted"] or 0}">{_n(base["prod_predicted"], ".3f")}</td>'
           f'<td data-v="{base["prod_gap"] or 0}" class="{_gap(base["prod_gap"], PROB_GAP)}">'
           f'{_n(base["prod_gap"], "+.3f")}</td>')
    for r in row:
        if not r:
            out += '<td class="grp">&mdash;</td>' + _dash(3)
            continue
        out += (f'<td class="grp" data-v="{r["candidate_predicted"] or 0}">{_n(r["candidate_predicted"], ".3f")}</td>'
                f'<td data-v="{r["candidate_gap"] or 0}" class="{_gap(r["candidate_gap"], PROB_GAP)}">'
                f'{_n(r["candidate_gap"], "+.3f")}</td>'
                f'<td data-v="{r["brier_delta"] or 0}" class="{_cls(r["brier_delta"])}">{_n(r["brier_delta"])}</td>'
                f'<td data-v="{r["p_value"] if r["p_value"] is not None else 1}">{_p(r["p_value"])}</td>')
    return out


def _calibration_head(sides):
    return ("<th>N</th><th>Matches</th><th>Real</th><th>Prod</th><th>Gap</th>"
            + _group_head(sides, ["", "Gap", "&Delta;Brier", "p"]))


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
            row = [r if r and r["n"] else None for r in row]
            cells = _calibration_cells(sides, row)
            if cells is None:
                continue
            base = next(r for r in row if r)
            prob_rows.append(f"""<tr>
                <th>{html.escape(str(cell_label))}</th>
                <td>{MARKET_TITLES[base['market']]}</td>
                <td class="dim">{base['selection']}</td>{cells}</tr>""")

    line_tables = [s["line_full"]["axes"][index].get("line_error", {}) for s in sides]
    line_rows = []
    for cell_label in axis0["order"]:
        row = [t.get(cell_label) for t in line_tables]
        base = next((r for r in row if r), None)
        if base is None:
            continue
        cells = ""
        for r in row:
            if not r:
                cells += '<td class="grp">&mdash;</td>' + _dash(3)
                continue
            cells += (f'<td class="grp">{r["candidate_line_error"]:.3f}</td>'
                      f'<td>{_pct(r["same_share"])}</td>'
                      f'<td class="{_cls(r["points_delta"])}">{_n(r["points_delta"], "+.3f")}</td>'
                      f'<td>{_p(r["p_value"])}</td>')
        line_rows.append(f"""<tr><th>{html.escape(str(cell_label))}</th>
            <td>{base['n']:,}</td><td>{base['matches']:,}</td>
            <td>{base['prod_line_error']:.3f}</td>{cells}</tr>""")

    name = html.escape(axis0["name"])
    return f"""
    <section class="panel" id="axis{index}">
      <h2>{name}</h2>
      <h3>At prod's line</h3>
      <div class="scroll">
      <table class="sortable">
        <thead><tr><th>{name}</th><th>Market</th><th>Sel</th>{_calibration_head(sides)}</tr></thead>
        <tbody>{''.join(prob_rows) or '<tr><td colspan="8" class="dim">no pairs</td></tr>'}</tbody>
      </table>
      </div>
      <h3>Line error (points)</h3>
      <table>
        <thead><tr><th>{name}</th><th>N</th><th>Matches</th><th>Prod</th>
          {_group_head(sides, ['', 'Same line', '&Delta;points', 'p'])}</tr></thead>
        <tbody>{''.join(line_rows) or '<tr><td colspan="4" class="dim">no pairs</td></tr>'}</tbody>
      </table>
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
            row = [r if r and r["n"] else None for r in row]
            cells = _calibration_cells(sides, row)
            if cells is None:
                continue
            base = next(r for r in row if r)
            score, quarter, possession = (str(part) for part in cell_label)
            key = buckets.sort_key(cell_label)
            sparse = base["matches"] < config.MIN_CELL_MATCHES
            rows.append(f"""<tr class="{'thin' if sparse else ''}">
                <th class="ax" data-v="{key[0]}">{html.escape(score)}</th>
                <td class="ax" data-v="{key[1]}">{html.escape(quarter)}</td>
                <td class="ax" data-v="{key[2]}">{html.escape(possession)}</td>
                <td>{MARKET_TITLES[market]}</td>
                <td class="dim">{base['selection']}</td>{cells}</tr>""")
    return f"""
    <section class="panel" id="cross">
      <h2>Cross-section calibration</h2>
      <div class="scroll">
      <table class="sortable" id="crossTable">
        <thead><tr><th class="ax">Score diff</th><th class="ax">Quarter</th><th class="ax">Possession</th><th>Market</th><th>Sel</th>
          {_calibration_head(sides)}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      </div>
    </section>"""


def _prematch_block(sides):
    """Calibration of the closing price, each candidate that quotes before
    kickoff against prod."""
    summaries = [s["line"].get("prematch") for s in sides]
    summaries = [x if x and x.get("n") else None for x in summaries]
    if not any(summaries):
        return ""
    base = next(x for x in summaries if x)

    def cell_group(c):
        if not c:
            return '<td class="grp">&mdash;</td>' + _dash(3)
        return (f'<td class="grp">{_n(c["candidate_predicted"], ".3f")}</td>'
                f'<td class="{_gap(c["candidate_gap"], PROB_GAP)}">{_n(c["candidate_gap"], "+.3f")}</td>'
                f'<td class="{_cls(c["brier_delta"])}">{_n(c["brier_delta"])}</td>'
                f'<td>{_p(c["p_value"])}</td>')

    def rows_for(field, label_of, sort_key):
        keys = set()
        for x in summaries:
            if x:
                keys |= set(x[field])
        out = []
        for k in sorted(keys, key=sort_key):
            cells = [x[field].get(k) if x else None for x in summaries]
            c0 = next(c for c in cells if c)
            out.append(f"""<tr><th>{label_of(k)}</th>
                <td>{c0['n']:,}</td><td>{c0['matches']:,}</td>
                <td><b>{_n(c0['realized'], '.3f')}</b></td>
                <td>{_n(c0['prod_predicted'], '.3f')}</td>
                <td class="{_gap(c0['prod_gap'], PROB_GAP)}">{_n(c0['prod_gap'], '+.3f')}</td>
                {''.join(cell_group(c) for c in cells)}</tr>""")
        return "".join(out)

    head = f"<thead><tr><th>Selection</th>{_calibration_head(sides)}</tr></thead>"
    selections = rows_for("by_selection", lambda k: f"{k[0]} {k[1]}",
                          lambda k: (str(k[0]), str(k[1])))
    lines = rows_for("by_line", lambda k: f"{k[0]} {k[1]} {k[2]:+g}",
                     lambda k: (str(k[0]), str(k[1]), k[2] if k[2] is not None else 0))

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
    h2h = ""
    for s, x in zip(sides, summaries):
        if not x:
            continue
        h = x["head_to_head"]
        h2h += f"""<tr><th>{_name(s)}</th>
          <td>{h['pairs']:,}</td><td>{h['matches']:,}</td>
          <td class="{_cls(h['mean'])}">{_n(h['mean'])}</td>
          <td class="dim">{_ci(h)}</td>
          <td>{h['matches_favouring_candidate']:,}</td>
          <td>{h['matches_favouring_prod']:,}</td>
          <td>{_p(h['p_value'])}</td></tr>"""
    line_rows = (f"""
      <h3>By line</h3>
      <table>{head}<tbody>{lines}</tbody></table>""" if lines else "")
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
      <table>{head}<tbody>{selections}</tbody></table>{line_rows}
      <h3>Head to head</h3>
      <table>
        <thead><tr><th>Candidate</th><th>Pairs</th><th>Matches</th><th>&Delta;Brier</th>
          <th>95% CI</th><th>Matches closer</th><th>Prod closer</th><th>p</th></tr></thead>
        <tbody>{h2h}</tbody>
      </table>
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
                <td class="dim">{_ci(block, '.1%')}</td>
                <td>{_pct(block['flat_share'])}</td>
                <td>{_p(block['p_value'])}</td>
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
          <td class="dim">{_ci(h, '.1%')}</td>
          <td>{h.get('ties', 0):,}</td><td>{h.get('both_flat', 0):,}</td>
          <td>{_p(h.get('p_value'))}</td></tr>""")
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
              <th>Right</th><th>95% CI</th><th>Flat</th><th>p</th></tr></thead>
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
          <th>95% CI</th><th>Ties</th><th>Both flat</th><th>p</th></tr></thead>
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
}


def _anchor_block(report):
    """Where in its drive each snapshot landed."""
    a = report.get("anchor")
    if not a or not a["pairs"]:
        return ""
    rows = []
    for kind in (drives.FIRST_DOWN, drives.MID_DRIVE, drives.NO_SNAP):
        n = a["counts"].get(kind, 0)
        if not n:
            continue
        good = kind == drives.FIRST_DOWN
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
    rows = []
    for market_id in sorted(first, key=lambda m: (MARKET_ORDER.index(first[m]["market"]), m)):
        row = first[market_id]
        tallied = row["tally"]
        if not tallied["n"]:
            continue
        cells = ""
        for s in sides:
            r = s["prob_full"]["summary"]["same_line"].get("selections", {}).get(market_id)
            if not r:
                cells += '<td class="grp">&mdash;</td>' + _dash(3)
                continue
            cells += _delta_cells(r["tally"]["candidate_brier"], r.get("brier"), ".4f", "+.4f")
        rows.append(f"""<tr>
            <th>{MARKET_TITLES[row['market']]}</th>
            <td>{row['selection']}</td>
            <td class="dim">{'&check;' if row['canonical'] else ''}</td>
            <td data-v="{market_id}" class="dim">{market_id}</td>
            <td>{tallied['n']:,}</td><td>{tallied['n_matches']:,}</td>
            <td>{_n(tallied['prod_brier'], '.4f')}</td>{cells}</tr>""")
    return f"""
    <section class="panel">
      <h2>By selection</h2>
      <div class="scroll">
      <table>
        <thead><tr><th>Market</th><th>Sel</th><th>Used</th><th>ID</th><th>Pairs</th><th>Matches</th>
          <th>Prod</th>{_group_head(sides, ['', '&Delta;', '95% CI', 'p'])}</tr></thead>
        <tbody>{''.join(rows) or '<tr><td colspan="7" class="dim">no pairs</td></tr>'}</tbody>
      </table>
      </div>
    </section>"""


def _both_sides_block(sides):
    tables = [s["prob_full"]["both_sides"] for s in sides]
    rows = []
    by_market = {}
    for market_id, row in sorted(tables[0].items()):
        by_market.setdefault(row["market"], []).append((market_id, row))
    for market in MARKET_ORDER:
        group = by_market.get(market, [])
        for market_id, row in group:
            cells = ""
            for t in tables:
                r = t.get(market_id)
                if not r:
                    cells += '<td class="grp">&mdash;</td><td>&mdash;</td>'
                    continue
                cells += (f'<td class="grp">{_n(r["candidate_predicted"], ".3f")}</td>'
                          f'<td class="{_gap(r["candidate_gap"], PROB_GAP)}">{_n(r["candidate_gap"], "+.3f")}</td>')
            rows.append(f"""<tr>
                <th>{MARKET_TITLES[market]}</th><td>{row['selection']}</td>
                <td class="dim">{'&check;' if row['canonical'] else ''}</td>
                <td>{row['n']:,}</td><td>{row['matches']:,}</td>
                <td>{_n(row['realized'], '.3f')}</td>
                <td>{_n(row['prod_predicted'], '.3f')}</td>
                <td class="{_gap(row['prod_gap'], PROB_GAP)}">{_n(row['prod_gap'], '+.3f')}</td>
                {cells}</tr>""")
        if len(group) == 2:
            realized_sum = sum(r["realized"] for _, r in group if r["realized"] is not None)
            ok = abs(realized_sum - 1.0) < 1e-9
            rows.append(f"""<tr class="subtotal">
                <th></th><td colspan="4" class="dim">realized sums to</td>
                <td>{realized_sum:.3f}</td>
                <td colspan="{2 + 2 * len(tables)}" class="{'good' if ok else 'bad'}">
                  {'mirror ok' if ok else 'NOT MIRRORED'}</td>
            </tr>""")
    return f"""
    <section class="panel">
      <h2>Mirror check</h2>
      <table>
        <thead><tr><th>Market</th><th>Sel</th><th>Used</th><th>N</th><th>Matches</th>
          <th>Real</th><th>Prod</th><th>Gap</th>{_group_head(sides, ['', 'Gap'])}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _daily(sides):
    dailies = [s["prob_full"].get("daily") or {} for s in sides]
    days = sorted(set().union(*dailies))
    if not days:
        return ""
    rows = []
    for day in days:
        base = next(d[day] for d in dailies if day in d)
        cells = ""
        for d in dailies:
            row = d.get(day)
            if not row:
                cells += '<td class="grp">&mdash;</td>' + _dash(2)
                continue
            brier = row["brier"]
            cells += (f'<td class="grp {_cls(brier.get("mean"))}">{_n(brier.get("mean"))}</td>'
                      f'<td class="dim">{_ci(brier)}</td><td>{_p(brier.get("p_value"))}</td>')
        rows.append(f"""<tr><th>{html.escape(day)}</th>
            <td>{base['pairs']:,}</td><td>{base['matches']:,}</td>{cells}</tr>""")
    return f"""
    <section class="panel" id="daily">
      <h2>By day</h2>
      <table>
        <thead><tr><th>Day</th><th>Pairs</th><th>Matches</th>
          {_group_head(sides, ['&Delta;Brier', '95% CI', 'p'])}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
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
        n = html.escape(s["name"])
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
                    0 if p.anchor == drives.FIRST_DOWN else 1]
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


def render_sides(sides, dropped=None):
    """The page for one or more candidates (multi.run + multi.build)."""
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


def write_sides(path, sides, dropped=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_sides(sides, dropped))
    return path


def write(path, report, header, stats, pairs, handle_scan,
          indrive_summary=None, prematch_summary=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(report, header, stats, pairs, handle_scan,
                        indrive_summary, prematch_summary))
    return path
