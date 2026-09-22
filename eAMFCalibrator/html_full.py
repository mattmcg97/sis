"""The combined report: directional, cross-sectional, and every pair.

One self-contained file -- no external fetches -- so it opens anywhere. All
numbers come from directional.build_full_report(), which runs off a single
pairing pass, so the sections cannot disagree with each other.

The pair table at the bottom carries every paired observation, widest
probability disagreement first, with both streams' line, price, probability,
outcome and error, and which one finished closer. Column headers sort.
"""

import html
import os

from . import buckets, config, drives, handles, markets

MARKET_ORDER = [markets.MONEYLINE, markets.SPREAD, markets.TOTAL]
MARKET_TITLES = {markets.MONEYLINE: "Moneyline", markets.SPREAD: "Spread",
                 markets.TOTAL: "Total"}


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
# so they share one ramp and differ only in where its steps fall.
#
# The steps are FIXED rather than taken from each run's own quantiles, so
# two reports can be read against each other. Each tuple is four cut
# points: at or below the first is the greenest step, above the last the
# reddest. They were set from the observed distributions -- calibration
# gaps run p25 0.02, p50 0.07, p75 0.23; line gaps p75 1, p90 3, p95 6.
PROB_GAP = (0.02, 0.05, 0.10, 0.20)      # realized minus predicted
PROB_DELTA = (0.02, 0.04, 0.06, 0.10)    # prod against candidate
LINE_GAP = (0.5, 1.0, 3.0, 6.0)          # handicap or total points
MESSAGE_GAP = (0, 1, 2, 3)               # feed messages

GAP_BANDS = ["at or below", "to", "to", "to", "above"]


def _gap(value, cuts, extra=""):
    """The ramp class for a gap, on top of whatever else the cell wears.

    Magnitude, not sign: a gap is a distance from agreement, and +0.40 is
    no better than -0.40. (The sign still shows in the number, and the
    columns where the SIGN is the point -- Brier delta, points delta, win
    rate -- keep _cls.)
    """
    if value is None or value == "":
        return extra
    magnitude = abs(float(value))
    step = next((i for i, cut in enumerate(cuts) if magnitude <= cut), len(cuts))
    return f"{extra} g{step}".strip()


def _gap_key(cuts, unit, fmt=".2f"):
    """The ramp spelled out, so no cell has to be read by colour alone."""
    labels = [f"&le;{cuts[0]:{fmt}}"]
    labels += [f"&le;{cut:{fmt}}" for cut in cuts[1:]]
    labels.append(f"&gt;{cuts[-1]:{fmt}}")
    swatches = "".join(
        f'<span class="key g{i}">{label}</span>' for i, label in enumerate(labels))
    return f'<p class="gapkey"><span class="klab">{unit}</span>{swatches}</p>'


def _outcome(value):
    if value is None:
        return '<span class="dim">push</span>'
    return '<span class="good">won</span>' if value else '<span class="bad">lost</span>'


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _verdict(report):
    """The headline as a line of numbers, over EVERY pair.

    Two readings, and they can disagree. The same-line Brier is the
    sharper instrument but only covers pairs where both streams quoted
    the same line; the combined decision covers all of them under the
    rule that applies to each, so where it separates the two it overrules
    the same-line reading. Both are shown rather than reconciled into a
    sentence -- a candidate that prices better but lines worse is a real
    result and the numbers say so without being narrated.
    """
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


def _headline(report):
    """The directional result as two tables: the combined reading, then
    the two halves it is made of."""
    same = report["summary"]["same_line"]
    line = report["summary"]["different_line"]
    d = report["summary"]["decisive"]
    dv = d["votes"]

    views = []
    for block, label, metric, spec in ((same, "Same line", "brier", "+.4f"),
                                       (line, "Different line", "mae", "+.3f")):
        o, v, m = block["overall"], block["votes"], block[metric]
        views.append(f"""<tr>
            <th>{label}</th>
            <td>{o['n']:,}</td><td>{o['n_matches']:,}</td>
            <td class="{_cls((o['candidate_win_rate'] or 0.5) - 0.5)}">{_pct(o['candidate_win_rate'])}</td>
            <td>{v['candidate']}&ndash;{v['prod']}</td>
            <td class="dim">{v['tie']}</td>
            <td class="{_cls(m.get('mean'))}">{_n(m.get('mean'), spec)}</td>
            <td class="dim">{_ci(m, spec)}</td>
            <td>{_p(m.get('p_value'))}</td>
        </tr>""")

    return f"""
    <section class="panel" id="directional">
      <h2>Directional calibration</h2>
      <table>
        <thead><tr><th>Reading</th><th>Pairs</th><th>Matches</th>
          <th title="both streams quoted the same line">On prob</th>
          <th title="lines differ, so the closer line decides">On line</th>
          <th>Cand win</th><th>Match vote</th><th>Level</th>
          <th title="match-clustered sign test">p</th></tr></thead>
        <tbody><tr>
          <th>Overall</th>
          <td>{d['n']:,}</td><td>{d['matches']:,}</td>
          <td>{d['settled_on_probability']:,}</td>
          <td>{d['settled_on_line']:,}</td>
          <td class="{_cls((d['candidate_win_rate'] or 0.5) - 0.5)}">{_pct(d['candidate_win_rate'])}</td>
          <td>{dv['candidate']}&ndash;{dv['prod']}</td>
          <td class="dim">{dv['tie']}</td>
          <td>{_p(dv['p_value'])}</td>
        </tr></tbody>
      </table>
      <table>
        <thead><tr><th>Reading</th><th>Pairs</th><th>Matches</th>
          <th>Cand win</th><th>Match vote</th><th>Level</th>
          <th title="Brier on the same line, points on a different one">&Delta;</th>
          <th>95% CI</th><th>p</th></tr></thead>
        <tbody>{''.join(views)}</tbody>
      </table>
    </section>"""


def _market_block(report):
    rows = []
    for key, label, metric, spec in (("same_line", "Same line", "brier", "+.4f"),
                                     ("different_line", "Different line", "mae", "+.3f")):
        block = report["summary"][key]
        for market in MARKET_ORDER:
            tallied = block["by_market"].get(market)
            if not tallied or not tallied["n"]:
                continue
            clustered = block.get("markets", {}).get(market, {}).get(metric, {})
            rows.append(f"""<tr>
                <td>{label}</td><th>{MARKET_TITLES[market]}</th>
                <td>{tallied['n']:,}</td><td>{tallied['n_matches']:,}</td>
                <td class="{_cls((tallied['candidate_win_rate'] or 0.5) - 0.5)}">{_pct(tallied['candidate_win_rate'])}</td>
                <td class="{_cls(clustered.get('mean'))}">{_n(clustered.get('mean'), spec)}</td>
                <td class="dim">{_ci(clustered, spec)}</td>
                <td>{_p(clustered.get('p_value'))}</td>
            </tr>""")
    return f"""
    <section class="panel">
      <h2>By market</h2>
      <table>
        <thead><tr><th>View</th><th>Market</th><th>Pairs</th><th>Matches</th>
          <th title="candidate share of decisive pairs">Cand win</th><th>&Delta;</th><th>95% CI</th><th>p</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _integrity_block(report, stats):
    lines = report["summary"]["lines"]
    comp = report["complement"]
    spread = report["spread"]

    line_rows = []
    for market in MARKET_ORDER:
        bucket = lines["by_market"].get(market)
        if not bucket or not bucket["n"]:
            continue
        c = comp.get(market, {})
        n = c.get("both_sides") or 0
        partition = (c.get("outcomes_partition", 0) / n) if n else None
        line_rows.append(f"""<tr>
            <th>{MARKET_TITLES[market]}</th>
            <td>{bucket['n']:,}</td>
            <td>{_pct(bucket['same'] / bucket['n'])}</td>
            <td>{_n(c.get('prob_sum_mean'), '.4f')}</td>
            <td class="{'' if partition is None or partition > 0.999 else 'bad'}">{_pct(partition)}</td>
            <td>{c.get('both_won', 0):,} / {c.get('both_lost', 0):,}</td>
        </tr>""")

    verdict, explanation = spread["verdict"]
    mismatch = ("" if verdict in ("unclear", config.SPREAD_RESOLUTION)
                else '')
    spread_body = "<p class=\"dim\">No spread pairs carrying both sides.</p>"
    if spread["n"]:
        spread_body = f"""
        <dl class="stats">
          <div><dt>Both sides present</dt><dd>{spread['n']:,}</dd></div>
          <div><dt>Probabilities summed</dt><dd>{_n(spread['prob_sum_mean'], '.4f')}</dd></div>
          <div><dt>Lines mirrored</dt><dd>{_pct(spread['lines_mirrored'] / spread['n'])}</dd></div>
          <div><dt>Lines equal</dt><dd>{_pct(spread['lines_equal'] / spread['n'])}</dd></div>
          <div><dt>Literal reading partitions</dt><dd>{_pct(spread['literal_partition'] / spread['n'])}</dd></div>
          <div><dt>Verdict</dt><dd><b>{verdict.upper()}</b></dd></div>
        </dl>{mismatch}"""

    return f"""
    <section class="panel">
      <h2>Integrity checks</h2>
      <table>
        <thead><tr><th>Market</th><th>Pairs</th><th>Same line</th>
          <th>P(both sides)</th><th>Partition</th><th>Both won / lost</th></tr></thead>
        <tbody>{''.join(line_rows)}</tbody>
      </table>
      <h3>Spread reading</h3>
      {spread_body}
    </section>"""


def _handle_block(scan):
    """Did PLAYER_1 / PLAYER_2 stay pinned to the same team?

    Rendered above the other checks because it qualifies them: everything
    the calibrator buckets on is read in the PLAYER_1 frame, so a match
    whose handles swap lands in the wrong cells rather than in none.
    """
    if not scan or not scan["matches"]:
        return """
    <section class="panel">
      <h2>Handle check</h2>
      <p class="count">No matches scanned.</p>
    </section>"""

    flipped = scan["matches_flipped"]
    share = flipped / scan["matches"]
    if not flipped:
        state = ('<span class="good">clean</span> '
                 f'<span class="dim">{scan["matches"]:,} matches, no side\'s '
                 'total ever went down</span>')
    else:
        acted = ("excluded from the numbers above"
                 if config.EXCLUDE_FLIPPED_MATCHES
                 else "STILL IN the numbers above")
        state = (f'<span class="bad">{flipped:,} of {scan["matches"]:,} matches '
                 f'({share:.1%}) have flipped handles</span> '
                 f'<span class="dim">{acted}</span>')

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
    more = (f'<p class="count">{len(scan["anomalies"]) - 200:,} more not shown</p>'
            if len(scan["anomalies"]) > 200 else "")
    detail = "" if not event_rows else f"""
      <h3>Flagged events</h3>
      <div class="scroll">
      <table>
        <thead><tr><th>Match</th><th>Where</th>
          <th>Kind</th><th title="p1-p2 either side of the flag, or how many touchdowns agreed">Evidence</th></tr></thead>
        <tbody>{''.join(event_rows)}</tbody>
      </table>
      </div>{more}"""

    return f"""
    <section class="panel">
      <h2>Handle check</h2>
      <p class="count">{state}</p>
      <table>
        <thead><tr><th>Kind</th><th>Events</th><th>Matches</th>
          <th title="counted as a handle flip">Flip</th></tr></thead>
        <tbody>{''.join(kind_rows) or '<tr><td colspan="4" class="dim">nothing flagged</td></tr>'}</tbody>
      </table>
      {detail}
    </section>"""


ANCHOR_TITLES = {
    drives.FIRST_DOWN: "Opening 1st &amp; 10",
    drives.MID_DRIVE: "Mid-drive snap",
    drives.NO_SNAP: "No real snap",
}


def _anchor_block(report):
    """Where in its drive each snapshot landed.

    A snapshot is meant to be a drive's opening 1st-and-10. Anything else
    means the start was never found, so the score, possession and field
    position describe a different moment -- on a row that still looks
    perfectly well-formed.
    """
    a = report.get("anchor")
    if not a or not a["pairs"]:
        return ""
    rows = []
    for kind in (drives.FIRST_DOWN, drives.MID_DRIVE, drives.NO_SNAP):
        n = a["counts"].get(kind, 0)
        if not n:
            continue
        share = n / a["pairs"]
        good = kind == drives.FIRST_DOWN
        rows.append(f"""<tr>
            <th>{ANCHOR_TITLES[kind]}</th>
            <td>{n:,}</td>
            <td class="{'good' if good else 'bad'}">{_pct(share)}</td>
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
    tone = "good" if a["share_clean"] > 0.9 else "bad"
    return f"""
    <section class="panel">
      <h2>Snapshot anchor</h2>
      <p class="count"><span class="{tone}">{_pct(a['share_clean'])}</span>
        on the opening 1st &amp; 10 &middot; {a['off_anchor']:,} of
        {a['pairs']:,} pairs elsewhere &middot; {a['matches']:,} matches</p>
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


def _market_state_block(report):
    """What non-live quotes cost, and which state they were in.

    Lopsided between the streams is the thing to look for: pairs lost
    around scores are not lost at random, and that is where two models
    differ most.
    """
    state = report.get("market_state")
    if not state or not state["pairs"]:
        return ""
    if not state["not_live"]:
        return """
    <section class="panel">
      <h2>Market state</h2>
      <p class="count"><span class="good">all open and active</span></p>
    </section>"""

    state_rows = []
    for (stream, label), n in sorted(state["by_state"].items(),
                                     key=lambda kv: -kv[1]):
        state_rows.append(f"""<tr>
            <td class="dim">{html.escape(stream)}</td>
            <th>{html.escape(label)}</th><td>{n:,}</td>
        </tr>""")

    split_rows = []
    for label, table in (("Quarter", state["by_quarter"]),
                         ("Market", state["by_market"])):
        for key in sorted(table):
            total, dead = table[key]
            if not total:
                continue
            share = dead / total
            split_rows.append(f"""<tr>
                <td class="dim">{label}</td><th>{html.escape(str(key))}</th>
                <td>{total:,}</td><td>{dead:,}</td>
                <td class="{'bad' if share > 0.05 else ''}">{_pct(share)}</td>
            </tr>""")

    lopsided = abs(state["prod_only"] - state["candidate_only"])
    tone = "bad" if lopsided > max(10, 0.2 * max(1, state["not_live"])) else "dim"
    return f"""
    <section class="panel">
      <h2>Market state</h2>
      <p class="count">{state['not_live']:,} of {state['pairs']:,} pairs
        ({_pct(state['share'])}) &middot; {state['matches']:,} matches</p>
      <dl class="stats">
        <div><dt>Prod only</dt><dd class="{tone}">{state['prod_only']:,}</dd></div>
        <div><dt>Candidate only</dt><dd class="{tone}">{state['candidate_only']:,}</dd></div>
        <div><dt>Both</dt><dd>{state['both']:,}</dd></div>
      </dl>
      <div class="cols">
        <div>
          <h3>By state</h3>
          <table>
            <thead><tr><th>Stream</th>
              <th title="STATUS and IS_ACTIVE verbatim">Status / active</th>
              <th>Pairs</th></tr></thead>
            <tbody>{''.join(state_rows)}</tbody>
          </table>
        </div>
        <div>
          <h3>Where</h3>
          <table>
            <thead><tr><th>Split</th><th>Bucket</th><th>Pairs</th>
              <th>Not live</th><th>Share</th></tr></thead>
            <tbody>{''.join(split_rows)}</tbody>
          </table>
        </div>
      </div>
    </section>"""


def _selection_block(report):
    """The directional comparison per selection, not per market.

    "By market" pools each market's two sides, which is right for reading
    a result -- they are complements. It is wrong for checking one: a
    fault confined to one side averages away into a flat market row. Both
    sides side by side is what would show it.
    """
    rows = []
    for key, label, metric, spec in (("same_line", "Same line", "brier", "+.4f"),
                                     ("different_line", "Different line", "mae", "+.3f")):
        block = report["summary"][key]
        selections = block.get("selections", {})
        for market_id in sorted(selections, key=lambda m: (
                MARKET_ORDER.index(selections[m]["market"]), m)):
            row = selections[market_id]
            tallied = row["tally"]
            if not tallied["n"]:
                continue
            clustered = row.get(metric, {})
            used = "&check;" if row["canonical"] else ""
            rows.append(f"""<tr>
                <td>{label}</td>
                <th>{MARKET_TITLES[row['market']]}</th>
                <td>{row['selection']}</td>
                <td class="dim">{used}</td>
                <td data-v="{market_id}" class="dim">{market_id}</td>
                <td>{tallied['n']:,}</td><td>{tallied['n_matches']:,}</td>
                <td class="{_cls((tallied['candidate_win_rate'] or 0.5) - 0.5)}">{_pct(tallied['candidate_win_rate'])}</td>
                <td class="{_cls(clustered.get('mean'))}">{_n(clustered.get('mean'), spec)}</td>
                <td class="dim">{_ci(clustered, spec)}</td>
                <td>{_p(clustered.get('p_value'))}</td>
            </tr>""")
    return f"""
    <section class="panel">
      <h2>By selection</h2>
      <div class="scroll">
      <table>
        <thead><tr><th>View</th><th>Market</th><th>Sel</th>
          <th title="the side the pooled tables read; the other is its mirror">Used</th>
          <th>ID</th><th>Pairs</th><th>Matches</th>
          <th title="candidate share of decisive pairs">Cand win</th>
          <th>&Delta;</th><th>95% CI</th><th title="match-clustered">p</th></tr></thead>
        <tbody>{''.join(rows) or '<tr><td colspan="11" class="dim">no pairs</td></tr>'}</tbody>
      </table>
      </div>
    </section>"""


def _both_sides_block(report):
    cells = report["both_sides"]
    rows = []
    by_market = {}
    for market_id, row in sorted(cells.items()):
        by_market.setdefault(row["market"], []).append(row)
    for market in MARKET_ORDER:
        group = by_market.get(market, [])
        for row in group:
            used = "&check;" if row["canonical"] else ""
            rows.append(f"""<tr>
                <th>{MARKET_TITLES[market]}</th><td>{row['selection']}</td>
                <td class="dim">{used}</td>
                <td>{row['n']:,}</td><td>{row['matches']:,}</td>
                <td>{_n(row['realized'], '.3f')}</td>
                <td>{_n(row['prod_predicted'], '.3f')}</td>
                <td class="{_gap(row['prod_gap'], PROB_GAP)}">{_n(row['prod_gap'], '+.3f')}</td>
                <td>{_n(row['candidate_predicted'], '.3f')}</td>
                <td class="{_gap(row['candidate_gap'], PROB_GAP)}">{_n(row['candidate_gap'], '+.3f')}</td>
            </tr>""")
        if len(group) == 2:
            realized_sum = sum(r["realized"] for r in group if r["realized"] is not None)
            ok = abs(realized_sum - 1.0) < 1e-9
            rows.append(f"""<tr class="subtotal">
                <th></th><td colspan="4" class="dim">realized sums to</td>
                <td>{realized_sum:.3f}</td>
                <td colspan="4" class="{'good' if ok else 'bad'}">
                  {'mirror ok' if ok else 'NOT MIRRORED'}</td>
            </tr>""")
    return f"""
    <section class="panel">
      <h2>Mirror check</h2>
      {_gap_key(PROB_GAP, 'Gap', '.2f')}
      <table>
        <thead><tr><th>Market</th><th>Sel</th><th>Used</th><th>N</th><th>Matches</th>
          <th>Real</th><th>Prod</th><th title="realized minus predicted">Gap</th><th>Cand</th><th title="realized minus predicted">Gap</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _daily(report):
    daily = report.get("daily") or {}
    if not daily:
        return ""
    rows = []
    for day, row in sorted(daily.items()):
        brier = row["brier"]
        rows.append(f"""<tr>
            <th>{html.escape(day)}</th>
            <td>{row['pairs']:,}</td><td>{row['matches']:,}</td>
            <td class="{_cls((row['win_rate'] or 0.5) - 0.5)}">{_pct(row['win_rate'])}</td>
            <td class="{_cls(brier.get('mean'))}">{_n(brier.get('mean'))}</td>
            <td class="dim">{_ci(brier)}</td>
            <td>{_p(brier.get('p_value'))}</td>
        </tr>""")
    return f"""
    <section class="panel" id="daily">
      <h2>By day</h2>
      <table>
        <thead><tr><th>Day</th><th>Pairs</th><th>Matches</th><th title="candidate share of decisive pairs">Cand win</th>
          <th>&Delta;Brier</th><th>95% CI</th><th>p</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _cross_axis(axis):
    prob_rows = []
    # Keyed by market ID, so both sides of every market are here rather
    # than the canonical one. Ordered by market, then ID, so a market's two
    # sides sit together and can be read as the mirror pair they are.
    market_of = {key[1]: row["market"] for key, row in axis["probability"].items()}
    ids = sorted(market_of, key=lambda i: (MARKET_ORDER.index(market_of[i]), i))
    for cell_label in axis["order"]:
        for market_id in ids:
            row = axis["probability"].get((cell_label, market_id))
            if not row or not row["n"]:
                continue
            prob_rows.append(f"""<tr>
                <th>{html.escape(str(cell_label))}</th>
                <td>{MARKET_TITLES[row['market']]}</td>
                <td class="dim">{row['selection']}</td>
                <td>{row['n']:,}</td><td>{row['matches']:,}</td>
                <td><b>{_n(row['realized'], '.3f')}</b></td>
                <td>{_n(row['prod_predicted'], '.3f')}</td>
                <td class="{_gap(row['prod_gap'], PROB_GAP)}">{_n(row['prod_gap'], '+.3f')}</td>
                <td>{_n(row['candidate_predicted'], '.3f')}</td>
                <td class="{_gap(row['candidate_gap'], PROB_GAP)}">{_n(row['candidate_gap'], '+.3f')}</td>
                <td class="{_cls(row['brier_delta'])}">{_n(row['brier_delta'])}</td>
                <td>{_p(row['p_value'])}</td>
            </tr>""")

    line_rows = []
    for cell_label in axis["order"]:
        row = axis["line"].get(cell_label)
        if not row or not row["n"]:
            continue
        line_rows.append(f"""<tr>
            <th>{html.escape(str(cell_label))}</th>
            <td>{row['n']:,}</td><td>{row['matches']:,}</td>
            <td class="{_gap(row['mean_line_gap'], LINE_GAP)}">{row['mean_line_gap']:.2f}</td>
            <td>{row['prod_line_error']:.3f}</td>
            <td>{row['candidate_line_error']:.3f}</td>
            <td class="{_cls(row['points_delta'])}">{_n(row['points_delta'], '+.3f')}</td>
            <td class="dim">{_ci(row, '+.3f')}</td>
            <td>{_p(row['p_value'])}</td>
        </tr>""")

    return f"""
    <section class="panel">
      <h2>{html.escape(axis['name'])}</h2>
      {_gap_key(PROB_GAP, 'Gap', '.2f')}
      <h3>Same line</h3>
      <table>
        <thead><tr><th>Cell</th><th>Market</th><th>Sel</th><th>N</th><th>Matches</th>
          <th>Real</th><th>Prod</th><th title="realized minus predicted">Gap</th><th>Cand</th><th title="realized minus predicted">Gap</th><th title="prod minus candidate: positive favours the candidate">&Delta;Brier</th><th title="match-clustered">p</th></tr></thead>
        <tbody>{''.join(prob_rows) or '<tr><td colspan="12" class="dim">no pairs</td></tr>'}</tbody>
      </table>
      <h3>Different line</h3>
      <table>
        <thead><tr><th>Cell</th><th>N</th><th>Matches</th><th>Line gap</th>
          <th>Prod err</th><th>Cand err</th><th title="prod minus candidate line error, in points">&Delta;points</th><th>95% CI</th>
          <th>p</th></tr></thead>
        <tbody>{''.join(line_rows) or '<tr><td colspan="9" class="dim">no pairs</td></tr>'}</tbody>
      </table>
    </section>"""


def _full_cell(report):
    """Quarter x possession x score difference, all three markets.

    This is the cross-section: one bucket is the three axes together, not
    each in isolation. It is sparse by design and gets denser as matches
    accumulate, so rows too thin to read are dimmed rather than dropped --
    seeing that a bucket exists but has four matches in it is part of the
    information.
    """
    rows = []
    thin = 0
    for cell_label in report["full_cell_order"]:
        for market in MARKET_ORDER:
            row = report["full_cell"].get((cell_label, market))
            if not row or not row["n"]:
                continue
            score, quarter, possession = (str(part) for part in cell_label)
            order = buckets.sort_key(cell_label)
            sparse = row["matches"] < config.MIN_CELL_MATCHES
            thin += 1 if sparse else 0
            rows.append(f"""<tr class="{'thin' if sparse else ''}">
                <th class="ax" data-v="{order[0]}">{html.escape(score)}</th>
                <td class="ax" data-v="{order[1]}">{html.escape(quarter)}</td>
                <td class="ax" data-v="{order[2]}">{html.escape(possession)}</td>
                <td>{MARKET_TITLES[market]}</td>
                <td class="dim">{row['selection']}</td>
                <td data-v="{row['n']}">{row['n']:,}</td>
                <td data-v="{row['matches']}">{row['matches']:,}</td>
                <td data-v="{row['realized'] or 0}"><b>{_n(row['realized'], '.3f')}</b></td>
                <td data-v="{row['prod_predicted'] or 0}">{_n(row['prod_predicted'], '.3f')}</td>
                <td data-v="{row['prod_gap'] or 0}" class="{_gap(row['prod_gap'], PROB_GAP)}">{_n(row['prod_gap'], '+.3f')}</td>
                <td data-v="{row['candidate_predicted'] or 0}">{_n(row['candidate_predicted'], '.3f')}</td>
                <td data-v="{row['candidate_gap'] or 0}" class="{_gap(row['candidate_gap'], PROB_GAP)}">{_n(row['candidate_gap'], '+.3f')}</td>
                <td data-v="{row['brier_delta'] or 0}" class="{_cls(row['brier_delta'])}">{_n(row['brier_delta'])}</td>
                <td data-v="{row['p_value'] if row['p_value'] is not None else 1}">{_p(row['p_value'])}</td>
            </tr>""")
    return f"""
    <section class="panel" id="cross">
      <h2>Cross-section calibration <span class="tag">score diff &times; quarter &times; possession</span></h2>
      <p class="count">{len(rows):,} cells &middot; {thin:,} under
         {config.MIN_CELL_MATCHES} matches</p>
      {_gap_key(PROB_GAP, 'Gap', '.2f')}
      <div class="scroll">
      <table class="sortable" id="crossTable">
        <thead><tr><th class="ax">Score diff</th><th class="ax">Quarter</th><th class="ax" title="which side has the ball">Possession</th><th>Market</th><th>Sel</th><th>N</th><th>Matches</th>
          <th>Real</th><th>Prod</th><th title="realized minus predicted">Gap</th><th>Cand</th><th title="realized minus predicted">Gap</th><th title="prod minus candidate: positive favours the candidate">&Delta;Brier</th><th title="match-clustered">p</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      </div>
    </section>"""


def _state(pair):
    """Which side was not live, since either one blanks the comparison."""
    if not pair.not_live:
        return "live"
    if not pair.prod_live and not pair.candidate_live:
        return "both"
    return "prod" if not pair.prod_live else "cand"


def _down(pair):
    """Down and distance as "3&10", or a dash when the feed has neither."""
    if pair.down_number is None and pair.distance is None:
        return "&mdash;"
    down = "?" if pair.down_number is None else pair.down_number
    distance = "?" if pair.distance is None else pair.distance
    return f"{down}&amp;{distance}"


def _pair_rows(pairs):
    # There is no game clock in the feed, so "how close to the end" has to
    # come from the wall clock: seconds from this snapshot to the last quote
    # of its own match. A proxy, and labelled as one, but it is the only
    # answer available to "was this the dying seconds or the third quarter".
    last_quote = {}
    for p in pairs:
        if p.publish_time is None:
            continue
        seen = last_quote.get(p.match_code)
        if seen is None or p.publish_time > seen:
            last_quote[p.match_code] = p.publish_time

    out = []
    for p in pairs:
        basis = "line" if not p.same_line else "prob"
        state_class = ' class="notlive"' if p.not_live else ""
        anchor_class = "" if p.anchor == drives.FIRST_DOWN else "warn"
        end = last_quote.get(p.match_code)
        to_end = (None if end is None or p.publish_time is None
                  else (end - p.publish_time).total_seconds())
        # Line first, and probability only where the lines agree -- exactly
        # the rule the overall verdict is decided under.
        winner = p.decisive_winner
        winner_cell = ('<span class="dim">level</span>' if winner in (None, "tie")
                       else f'<span class="{"good" if winner == "candidate" else "bad"}">'
                            f'{"cand" if winner == "candidate" else "prod"}</span>')
        # A probability that refers to a different line is not comparable to
        # one that refers to another, and a line error is in points, so on a
        # different-line row the probability columns are blanked rather than
        # filled with a number that invites the wrong comparison.
        if p.not_live:
            # Not a price anyone could have taken, so nothing here is a
            # comparison. The two probabilities still show as facts.
            prod_error = candidate_error = None
            disagreement = '<td data-v="" class="dim">&mdash;</td>'
        elif p.same_line:
            prod_error, candidate_error = p.errors("probability")
            disagreement = (f'<td data-v="{p.disagreement}" '
                            f'class="{_gap(p.disagreement, PROB_DELTA)}">'
                            f'<b>{p.disagreement:.4f}</b></td>')
        else:
            prod_error = candidate_error = None
            disagreement = '<td data-v="" class="dim">&mdash;</td>' 
        out.append(
            f'<tr data-match="{html.escape(p.match_code).lower()}"{state_class}>'
            f'<td>{"" if p.publish_time is None else html.escape(str(p.publish_time)[:19])}</td>'
            f'<td>{html.escape(p.match_code)}</td>'
            f'<td data-v="{p.drive_number}">{p.drive_number}</td>'
            f'<td data-v="{p.message_count}">{p.message_count}</td>'
            f'<td data-v="{abs(p.message_gap)}" class="{_gap(p.message_gap, MESSAGE_GAP)}">'
            f'{p.message_gap:+d}</td>'
            f'<td>{"Q" + str(p.period_number) if p.period_number and p.period_number <= 4 else ("OT" if p.period_number else "?")}</td>'
            f'<td data-v="{p.score_p1}">{p.score_p1}</td>'
            f'<td data-v="{p.score_p2}">{p.score_p2}</td>'
            f'<td data-v="{p.score_diff}">{p.score_diff:+d}</td>'
            f'<td>{"H" if p.offensive_team == "Home Team" else ("A" if p.offensive_team == "Away Team" else "?")}</td>'
            f'<td data-v="{"" if p.field_position is None else p.field_position}">{"&mdash;" if p.field_position is None else p.field_position}</td>'
            f'<td data-v="{"" if p.down_number is None else p.down_number}" class="{anchor_class}">{_down(p)}</td>'
            f'<td data-v="{"" if to_end is None else to_end}" class="{"warn" if to_end is not None and to_end <= 120 else ""}">{"&mdash;" if to_end is None else f"{to_end:,.0f}"}</td>'
            f'<td>{MARKET_TITLES.get(markets.market_group(p.market_id), "?")}</td>'
            f'<td>{markets.selection_label(p.market_id) or ""}</td>'
            f'<td data-v="{"" if p.prod_line is None else p.prod_line}">{"&mdash;" if p.prod_line is None else format(p.prod_line, "+.1f")}</td>'
            f'<td data-v="{"" if p.candidate_line is None else p.candidate_line}">{"&mdash;" if p.candidate_line is None else format(p.candidate_line, "+.1f")}</td>'
            f'<td data-v="{"" if p.line_delta is None else p.line_delta}" class="{_gap(p.line_delta, LINE_GAP)}">{"&mdash;" if p.line_delta is None else format(p.line_delta, ".1f")}</td>'
            f'<td data-v="{"" if p.prod_decimal is None else p.prod_decimal}">{"&mdash;" if p.prod_decimal is None else format(p.prod_decimal, ".2f")}</td>'
            f'<td data-v="{"" if p.candidate_decimal is None else p.candidate_decimal}">{"&mdash;" if p.candidate_decimal is None else format(p.candidate_decimal, ".2f")}</td>'
            f'<td data-v="{p.prod_probability}">{p.prod_probability:.4f}</td>'
            f'<td data-v="{p.candidate_probability}">{p.candidate_probability:.4f}</td>'
            f'{disagreement}'
            f'<td data-v="{"" if p.realized is None else p.realized}">{"&mdash;" if p.realized is None else format(p.realized, ".0f")}</td>'
            f'<td>{_outcome(p.prod_outcome)}</td>'
            f'<td>{_outcome(p.candidate_outcome)}</td>'
            f'<td data-v="{"" if prod_error is None else prod_error}">{"&mdash;" if prod_error is None else format(prod_error, ".4f")}</td>'
            f'<td data-v="{"" if candidate_error is None else candidate_error}">{"&mdash;" if candidate_error is None else format(candidate_error, ".4f")}</td>'
            f'<td>{winner_cell}</td>'
            f'<td class="dim">{basis}</td>'
            f'<td data-v="{0 if p.not_live else 1}" class="{"bad" if p.not_live else "dim"}">{_state(p)}</td>'
            f'</tr>')
    return "".join(out)


def _pair_table(pairs):
    return f"""
    <section class="panel" id="pairs">
      <h2>Every pair <span class="tag">{len(pairs):,} rows</span></h2>
      <div class="filter">
        <input id="pairFilter" type="search" autocomplete="off" spellcheck="false"
               placeholder="filter by match id" aria-label="Filter rows by match id">
        <span id="pairCount" class="count">{len(pairs):,} rows</span>
        <span class="count" title="works in every table; Escape clears them all">pin</span>
      </div>
      {_gap_key(PROB_DELTA, '&Delta;prob', '.2f')}
      {_gap_key(LINE_GAP, 'Line gap', '.1f')}
      <div class="scroll">
      <table class="sortable" id="pairTable">
        <thead><tr>
          <th>Time</th><th>Match</th><th>Drive</th><th>Msg</th><th title="offset from the snapshot's own message; 0 is an exact hit">&plusmn;Msg</th>
          <th>Qtr</th><th title="PLAYER_1 score at the snapshot">Home</th><th title="PLAYER_2 score at the snapshot">Away</th><th title="home minus away">Diff</th><th>Poss</th>
          <th title="FIELD_POSITION at the snapshot">Field</th>
          <th title="down and distance at the snapshot; flagged when it is not the drive's opening 1st-and-10">D&amp;D</th>
          <th title="seconds from here to the match's last quote; the feed has no game clock, so this is a wall-clock proxy">To end</th>
          <th>Market</th><th>Sel</th>
          <th>Prod line</th><th>Cand line</th><th>&Delta;line</th>
          <th>Prod price</th><th>Cand price</th>
          <th>Prod prob</th><th>Cand prob</th><th>&Delta;prob</th>
          <th title="realized margin or total">Result</th><th title="won at its own line">Prod</th><th title="won at its own line">Cand</th>
          <th>Prod err</th><th>Cand err</th><th>Closer</th><th title="prob where both quoted the same line, line where they did not">Basis</th>
          <th title="whether both quotes were open and active; a non-live pair is shown but scored by nothing">Live</th>
        </tr></thead>
        <tbody>{_pair_rows(pairs)}</tbody>
      </table>
      </div>
    </section>"""


# ---------------------------------------------------------------------------

def _checks_summary(report, scan=None):
    """One line stating whether the diagnostics passed, for the collapsed block."""
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


def render(report, header, stats, pairs, handle_scan):
    verdict_class, verdict_text = _verdict(report)
    checks_summary = _checks_summary(report, handle_scan)
    # The settled universe and the matches that actually produced pairs are
    # not the same number: a match can be settled, carry quotes, and still
    # pair nothing. Report both rather than implying one.
    contributing = len({p.match_code for p in pairs})
    exact = stats.get("exact_message_pair", 0)
    offset = stats.get("offset_message_pair", 0)
    exact_rate = _pct(exact / (exact + offset)) if exact + offset else "&mdash;"
    lines = report["summary"]["lines"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eAMF Model Report</title>
<style>
  :root {{
    --bg:#f7f7f5; --panel:#fff; --ink:#1a1a18; --dim:#6b6b66; --line:#e2e2dd;
    --g0:#c9f6ca; --g1:#eccf92; --g2:#eeae82; --g3:#e49082; --g4:#d5766c;
    --good:#1c7c4a; --bad:#b3261e; --warn:#8a6d1f; --accent:#2d4a7c;
    --head:#f0f0ec; --axis:#eaeef4; --pick:#fdf0c8; --hover:#f2f2ef;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#17171a; --panel:#1f1f23; --ink:#ededea; --dim:#9a9a95; --line:#32323a;
      --g0:#132d15; --g1:#463305; --g2:#663c1c; --g3:#82463c; --g4:#9d534c;
      --good:#4cc281; --bad:#ef6f66; --warn:#d9b451; --accent:#8fb0e8;
      --head:#26262c; --axis:#232833; --pick:#4a3f1c; --hover:#26262c;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#17171a; --panel:#1f1f23; --ink:#ededea; --dim:#9a9a95; --line:#32323a;
    --g0:#132d15; --g1:#463305; --g2:#663c1c; --g3:#82463c; --g4:#9d534c;
    --good:#4cc281; --bad:#ef6f66; --warn:#d9b451; --accent:#8fb0e8;
    --head:#26262c; --axis:#232833; --pick:#4a3f1c; --hover:#26262c;
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--ink);padding:16px;
       font:13px/1.45 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}}
  .wrap{{max-width:1500px;margin:0 auto}}
  header{{display:flex;flex-wrap:wrap;gap:4px 18px;align-items:baseline;margin-bottom:10px}}
  h1{{font-size:18px;margin:0;letter-spacing:-0.01em}}
  h2{{font-size:14px;margin:0 0 3px;letter-spacing:-0.005em}}
  h3{{font-size:12px;margin:14px 0 5px;color:var(--dim);text-transform:uppercase;
      letter-spacing:0.05em}}
  .meta{{color:var(--dim);font-size:11.5px}}
  .meta b{{color:var(--ink);font-weight:600}}
  nav{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px;font-size:11.5px}}
  nav a{{color:var(--accent);text-decoration:none}}
  nav a:hover{{text-decoration:underline}}
  .verdict{{padding:9px 12px;border-radius:7px;margin-bottom:12px;background:var(--panel);
            border-left:3px solid var(--warn)}}
  .verdict.good{{border-left-color:var(--good)}}
  .verdict.bad{{border-left-color:var(--bad)}}
  .panel{{background:var(--panel);border:1px solid var(--line);border-radius:7px;
          padding:12px 14px;margin-bottom:12px}}
  .cols{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  @media (max-width:820px){{.cols{{grid-template-columns:1fr}}}}
  table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
  th,td{{text-align:right;padding:4px 6px;border-bottom:1px solid var(--line);
         white-space:nowrap}}
  thead th{{color:var(--dim);font-weight:500;font-size:10.5px;text-transform:uppercase;
            letter-spacing:0.04em;background:var(--head)}}
  tbody th,tfoot th{{text-align:left;font-weight:600}}
  tbody td:first-child,tbody th:first-child{{text-align:left}}
  tr.subtotal td,tr.subtotal th{{border-bottom:2px solid var(--line);font-size:11px}}
  .good{{color:var(--good);font-weight:600}}
  .bad{{color:var(--bad);font-weight:600}}
  .warn{{color:var(--warn)}}
  /* Gap columns: green near zero, red far from it, as asked for.
     Green and red are the one pair red-green colour blindness cannot
     separate, so the steps are NOT just five hues -- their OKLCH
     lightness falls monotonically, 0.93 -> 0.67 on light and 0.27 ->
     0.53 on dark, with every adjacent gap over the 0.06 floor. A reader
     who sees no hue at all still sees the cell darken as the gap grows.
     Each mode's steps are picked against its own surface rather than
     flipped, and every step clears 4.7:1 against the ink on it, so the
     number stays readable -- and the number is the real answer; the
     colour only says which band it is in, and the key names the bands. */
  td.g0{{background:var(--g0)}}
  td.g1{{background:var(--g1)}}
  td.g2{{background:var(--g2)}}
  td.g3{{background:var(--g3)}}
  td.g4{{background:var(--g4)}}
  .gapkey{{display:flex;flex-wrap:wrap;gap:3px;align-items:center;margin:0 0 8px;
           font-size:10.5px;font-variant-numeric:tabular-nums}}
  .gapkey .klab{{color:var(--dim);text-transform:uppercase;letter-spacing:0.04em;
                 margin-right:5px}}
  .gapkey .klab+.klab,.gapkey .key+.klab{{margin-left:6px;margin-right:0}}
  .key{{padding:1px 6px;border-radius:3px;border:1px solid var(--line);
        color:var(--ink)}}
  .key.g0{{background:var(--g0)}}
  .key.g1{{background:var(--g1)}}
  .key.g2{{background:var(--g2)}}
  .key.g3{{background:var(--g3)}}
  .key.g4{{background:var(--g4)}}
  .dim{{color:var(--dim);font-weight:400}}
  .stats{{display:flex;flex-wrap:wrap;gap:6px 20px;margin:0}}
  .stats dt{{color:var(--dim);font-size:10.5px;text-transform:uppercase;
             letter-spacing:0.04em}}
  .stats dd{{margin:1px 0 0;font-size:13px;font-variant-numeric:tabular-nums}}
  td.ax,th.ax{{background:var(--axis);text-align:left}}
  thead th.ax{{color:var(--ink)}}
  .tag{{color:var(--dim);font-weight:400;font-size:11px;letter-spacing:0}}
  .count{{color:var(--dim);font-size:11px;margin:0 0 8px}}
  .filter{{display:flex;gap:10px;align-items:center;margin:0 0 8px}}
  .filter .count{{margin:0}}
  .filter input{{font:inherit;font-size:12px;padding:4px 8px;min-width:220px;
                 color:var(--ink);background:var(--bg);border:1px solid var(--line);
                 border-radius:5px}}
  .filter input:focus{{outline:2px solid var(--accent);outline-offset:-1px}}
  th[title]{{cursor:help;border-bottom:1px dotted var(--dim)}}
  tbody tr:hover > *{{background:var(--hover)}}
  /* Click a row to pin it. Inset shadows rather than a border, so pinning
     never reflows the table. */
  tbody tr.picked > *{{background:var(--pick);font-weight:600}}
  tbody tr.picked > :first-child{{box-shadow:inset 4px 0 0 var(--warn)}}
  tbody tr.picked > :last-child{{box-shadow:inset -4px 0 0 var(--warn)}}
  tbody tr.thin.picked > *{{opacity:1}}
  tbody tr.notlive > *{{color:var(--dim);font-style:italic}}
  tbody tr.notlive.picked > *{{color:var(--ink);font-style:normal}}
  tbody tr.picked .dim{{color:var(--ink)}}
  code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px}}
  details.panel{{padding:0}}
  details.panel > summary{{cursor:pointer;padding:12px 14px;font-size:13px;
                           font-weight:600;list-style:none}}
  details.panel > summary::-webkit-details-marker{{display:none}}
  details.panel > summary::before{{content:"\25B8 ";color:var(--dim)}}
  details.panel[open] > summary::before{{content:"\25BE "}}
  details.panel[open] > summary{{border-bottom:1px solid var(--line)}}
  details.panel .panel{{border:none;margin:0;border-bottom:1px solid var(--line);
                        border-radius:0}}
  tr.thin td,tr.thin th{{opacity:0.45}}
  .scroll{{max-height:70vh;overflow:auto;border:1px solid var(--line);border-radius:5px}}
  .scroll thead th{{position:sticky;top:0;z-index:1}}
  .sortable thead th{{cursor:pointer;user-select:none}}
  .sortable thead th:hover{{color:var(--ink)}}
  .sortable thead th.asc::after{{content:" \\2191"}}
  .sortable thead th.desc::after{{content:" \\2193"}}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>eAMF &mdash; candidate vs prod</h1>
    <span class="meta">from <b>{html.escape(str(config.CUTOFF_START))}</b></span>
    <span class="meta"><b>{contributing}</b> matches with pairs</span>
    <span class="meta">of <b>{header.get('paired_matches', 0)}</b> settled</span>
    <span class="meta"><b>{report['summary']['pairs']:,}</b> pairs</span>
    <span class="meta"><b>{stats.get('snapshots', 0):,}</b> drive snapshots</span>
    <span class="meta">exact-message pairing <b>{exact_rate}</b></span>
    <span class="meta">same line <b>{_pct(lines['same_rate'])}</b></span>
  </header>

  <nav>
    <a href="#headline">Directional calibration</a>
    <a href="#cross">Cross-section calibration</a>
    <a href="#checks">Checks</a>
    <a href="#pairs">Every pair</a>
  </nav>

  <div class="verdict {verdict_class}">{verdict_text}</div>

  <div id="headline">{_headline(report)}{_market_block(report)}</div>
  {_full_cell(report)}
  <details class="panel" id="checks">
    <summary>Checks &mdash; {checks_summary}</summary>
    {_handle_block(handle_scan)}
    {_anchor_block(report)}
    {_market_state_block(report)}
    {_selection_block(report)}
    {_daily(report)}
    {_integrity_block(report, stats)}
    {_both_sides_block(report)}
    {''.join(_cross_axis(axis) for axis in report['axes'])}
  </details>
  {_pair_table(pairs)}
</div>
<script>
['pairTable', 'crossTable'].forEach(function (id) {{
  var table = document.getElementById(id);
  if (!table) return;
  var headers = table.tHead.rows[0].cells;
  var body = table.tBodies[0];
  Array.prototype.forEach.call(headers, function (header, index) {{
    header.addEventListener('click', function () {{
      var descending = !header.classList.contains('desc');
      Array.prototype.forEach.call(headers, function (other) {{
        other.classList.remove('asc', 'desc');
      }});
      header.classList.add(descending ? 'desc' : 'asc');
      var rows = Array.prototype.slice.call(body.rows);
      rows.sort(function (a, b) {{
        var x = a.cells[index], y = b.cells[index];
        var xv = x.dataset.v !== undefined ? x.dataset.v : x.textContent.trim();
        var yv = y.dataset.v !== undefined ? y.dataset.v : y.textContent.trim();
        var xn = parseFloat(xv), yn = parseFloat(yv);
        var numeric = !isNaN(xn) && !isNaN(yn);
        if (numeric) return descending ? yn - xn : xn - yn;
        // Blanks last either way.
        if (xv === '') return 1;
        if (yv === '') return -1;
        return descending ? yv.localeCompare(xv) : xv.localeCompare(yv);
      }});
      var fragment = document.createDocumentFragment();
      rows.forEach(function (row) {{ fragment.appendChild(row); }});
      body.appendChild(fragment);
    }});
  }});
}});

// Click any row to pin it, click again to unpin, Escape to clear. Survives
// sorting and filtering because the class rides on the row element itself.
(function () {{
  document.addEventListener('click', function (event) {{
    var cell = event.target.closest('td, th');
    if (!cell) return;
    var row = cell.parentElement;
    if (!row || row.parentElement.tagName !== 'TBODY') return;
    // Selecting text inside a row should not also pin it.
    if (String(window.getSelection())) return;
    row.classList.toggle('picked');
  }});
  document.addEventListener('keydown', function (event) {{
    if (event.key !== 'Escape') return;
    var picked = document.querySelectorAll('tbody tr.picked');
    for (var i = 0; i < picked.length; i++) picked[i].classList.remove('picked');
  }});
}})();

(function () {{
  var input = document.getElementById('pairFilter');
  var table = document.getElementById('pairTable');
  var label = document.getElementById('pairCount');
  if (!input || !table || !label) return;
  var rows = Array.prototype.slice.call(table.tBodies[0].rows);
  var total = rows.length;
  var pending = null;

  function apply() {{
    var needle = input.value.trim().toLowerCase();
    var shown = 0;
    // The row carries its match id, so this never touches the DOM text of
    // twenty-odd cells per row -- on a long window that is the difference
    // between a keystroke feeling instant and not.
    for (var i = 0; i < total; i++) {{
      var row = rows[i];
      var hit = !needle || row.dataset.match.indexOf(needle) !== -1;
      if (hit) shown++;
      var hidden = row.style.display === 'none';
      if (hit === hidden) row.style.display = hit ? '' : 'none';
    }}
    label.textContent = needle
      ? shown.toLocaleString() + ' of ' + total.toLocaleString() + ' rows'
      : total.toLocaleString() + ' rows';
  }}

  input.addEventListener('input', function () {{
    if (pending) clearTimeout(pending);
    pending = setTimeout(apply, 120);
  }});
  input.addEventListener('search', apply);
}})();
</script>
</body>
</html>
"""


def write(path, report, header, stats, pairs, handle_scan):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(report, header, stats, pairs, handle_scan))
    return path
