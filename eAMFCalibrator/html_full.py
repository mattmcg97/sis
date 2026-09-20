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

from . import config, markets

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


def _outcome(value):
    if value is None:
        return '<span class="dim">push</span>'
    return '<span class="good">won</span>' if value else '<span class="bad">lost</span>'


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _verdict(report):
    same = report["summary"]["same_line"]
    brier = same["brier"]
    mean, lo, hi = brier.get("mean"), brier.get("ci_low"), brier.get("ci_high")
    if mean is None or lo is None:
        return "neutral", "Not enough paired data to separate the two models."
    if lo <= 0 <= hi:
        lean = "the candidate" if mean > 0 else "prod"
        return ("neutral",
                f"No detectable difference. The point estimate leans to {lean} "
                f"({mean:+.4f} Brier) but the interval crosses zero.")
    side = "candidate" if mean > 0 else "prod"
    return (("good" if mean > 0 else "bad"),
            f"The {side} is better on same-line pairs, and the interval excludes zero.")


def _headline(report):
    same = report["summary"]["same_line"]
    line = report["summary"]["different_line"]
    o, v, b, m = same["overall"], same["votes"], same["brier"], same["mae"]
    lo, lv, lm = line["overall"], line["votes"], line["mae"]
    return f"""
    <section class="panel">
      <h2>Directional calibration</h2>
      <p class="sub">Paired at each drive-start snapshot. Where both streams
         quoted the same line the probability decides; where they differ the
         line does.</p>
      <div class="cols">
        <div>
          <h3>Same line &mdash; probability</h3>
          <dl class="stats">
            <div><dt>Pairs</dt><dd>{o['n']:,} <span class="dim">/ {o['n_matches']:,} matches</span></dd></div>
            <div><dt>Candidate win rate</dt><dd>{_pct(o['candidate_win_rate'])}</dd></div>
            <div><dt>Match vote</dt><dd>{v['candidate']}&ndash;{v['prod']} <span class="dim">({v['tie']} level)</span></dd></div>
            <div><dt>&Delta;Brier / match</dt><dd class="{_cls(b.get('mean'))}">{_n(b.get('mean'))}</dd></div>
            <div><dt>95% CI</dt><dd class="dim">{_ci(b)}</dd></div>
            <div><dt>p (clustered)</dt><dd>{_p(b.get('p_value'))}</dd></div>
            <div><dt>&Delta;MAE / match</dt><dd class="{_cls(m.get('mean'))}">{_n(m.get('mean'))}</dd></div>
          </dl>
        </div>
        <div>
          <h3>Different line &mdash; whose line was closer</h3>
          <dl class="stats">
            <div><dt>Pairs</dt><dd>{lo['n']:,} <span class="dim">/ {lo['n_matches']:,} matches</span></dd></div>
            <div><dt>Candidate win rate</dt><dd>{_pct(lo['candidate_win_rate'])}</dd></div>
            <div><dt>Match vote</dt><dd>{lv['candidate']}&ndash;{lv['prod']} <span class="dim">({lv['tie']} level)</span></dd></div>
            <div><dt>&Delta;points / match</dt><dd class="{_cls(lm.get('mean'))}">{_n(lm.get('mean'), '+.3f')}</dd></div>
            <div><dt>95% CI</dt><dd class="dim">{_ci(lm, '+.3f')}</dd></div>
            <div><dt>p (clustered)</dt><dd>{_p(lm.get('p_value'))}</dd></div>
          </dl>
        </div>
      </div>
      <p class="note">&Delta; is prod minus candidate, so positive favours the
         candidate. The win rate is a direction check only: it discards
         magnitude, so a genuinely better model can sit near 50%.</p>
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
      <p class="sub">Each market with its own match-clustered test. Pooling them
         dilutes an effect confined to one.</p>
      <table>
        <thead><tr><th>View</th><th>Market</th><th>Pairs</th><th>Matches</th>
          <th>Cand win</th><th>&Delta;</th><th>95% CI</th><th>p</th></tr></thead>
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
                else '<p class="note bad">Does not match the configured resolution '
                     f'(<code>{config.SPREAD_RESOLUTION}</code>). Every spread number '
                     'here is affected.</p>')
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
        </dl>
        <p class="note">{html.escape(explanation)}</p>{mismatch}"""

    return f"""
    <section class="panel">
      <h2>Integrity checks</h2>
      <p class="sub">Whether the measurement is sound before reading anything off it.</p>
      <table>
        <thead><tr><th>Market</th><th>Pairs</th><th>Same line</th>
          <th>P(both sides)</th><th>Partition</th><th>Both won / lost</th></tr></thead>
        <tbody>{''.join(line_rows)}</tbody>
      </table>
      <p class="note">P(both sides) of 1.0000 means a fair book with no overround,
         so each selection is exactly its opposite's complement and one can be
         dropped for free. Partition must be 100%: exactly one side wins.</p>
      <h3>Spread: two propositions, or one with a yes and a no?</h3>
      {spread_body}
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
                <td class="{_cls(row['prod_gap'])}">{_n(row['prod_gap'], '+.3f')}</td>
                <td>{_n(row['candidate_predicted'], '.3f')}</td>
                <td class="{_cls(row['candidate_gap'])}">{_n(row['candidate_gap'], '+.3f')}</td>
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
      <h2>Every selection, as a mirror check</h2>
      <p class="sub">Not an extra finding. If the sides are complements, each
         pair of rows has realized summing to 1.000 and gaps that cancel.</p>
      <table>
        <thead><tr><th>Market</th><th>Sel</th><th>Used</th><th>N</th><th>Matches</th>
          <th>Realized</th><th>Prod</th><th>Gap</th><th>Cand</th><th>Gap</th></tr></thead>
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
    means = [r["brier"].get("mean") for r in daily.values()
             if r["brier"].get("mean") is not None]
    note = ""
    if len(means) > 1:
        agree = len({m > 0 for m in means}) == 1
        note = (f"Daily &Delta;Brier spans {min(means):+.4f} to {max(means):+.4f}. "
                + ("All days agree on direction." if agree
                   else "Days disagree on direction &mdash; the effect is not stable yet."))
    return f"""
    <section class="panel" id="daily">
      <h2>By day</h2>
      <p class="sub">The window only grows, so the question as games accumulate is
         whether the answer is stable. A day out of step with its neighbours is
         worth a look before it gets averaged away.</p>
      <table>
        <thead><tr><th>Day</th><th>Pairs</th><th>Matches</th><th>Cand win</th>
          <th>&Delta;Brier</th><th>95% CI</th><th>p</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="note">{note}</p>
    </section>"""


def _cross_axis(axis):
    prob_rows = []
    for cell_label in axis["order"]:
        for market in MARKET_ORDER:
            row = axis["probability"].get((cell_label, market))
            if not row or not row["n"]:
                continue
            prob_rows.append(f"""<tr>
                <th>{html.escape(str(cell_label))}</th>
                <td>{MARKET_TITLES[market]}</td><td class="dim">{row['selection']}</td>
                <td>{row['n']:,}</td><td>{row['matches']:,}</td>
                <td><b>{_n(row['realized'], '.3f')}</b></td>
                <td>{_n(row['prod_predicted'], '.3f')}</td>
                <td class="{_cls(row['prod_gap'])}">{_n(row['prod_gap'], '+.3f')}</td>
                <td>{_n(row['candidate_predicted'], '.3f')}</td>
                <td class="{_cls(row['candidate_gap'])}">{_n(row['candidate_gap'], '+.3f')}</td>
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
            <td>{row['mean_line_gap']:.2f}</td>
            <td>{row['prod_line_error']:.3f}</td>
            <td>{row['candidate_line_error']:.3f}</td>
            <td class="{_cls(row['points_delta'])}">{_n(row['points_delta'], '+.3f')}</td>
            <td class="dim">{_ci(row, '+.3f')}</td>
            <td>{_p(row['p_value'])}</td>
        </tr>""")

    return f"""
    <section class="panel">
      <h2>{html.escape(axis['name'])}</h2>
      <h3>Same line &mdash; predicted against realized</h3>
      <table>
        <thead><tr><th>Cell</th><th>Market</th><th>Sel</th><th>N</th><th>Matches</th>
          <th>Realized</th><th>Prod</th><th>Gap</th><th>Cand</th><th>Gap</th>
          <th>&Delta;Brier</th><th>p</th></tr></thead>
        <tbody>{''.join(prob_rows) or '<tr><td colspan="12" class="dim">no pairs</td></tr>'}</tbody>
      </table>
      <h3>Different line &mdash; whose line was closer, in points</h3>
      <table>
        <thead><tr><th>Cell</th><th>N</th><th>Matches</th><th>Line gap</th>
          <th>Prod err</th><th>Cand err</th><th>&Delta;points</th><th>95% CI</th>
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
            label = " ".join(str(part) for part in cell_label)
            sparse = row["matches"] < config.MIN_CELL_MATCHES
            thin += 1 if sparse else 0
            rows.append(f"""<tr class="{'thin' if sparse else ''}">
                <th>{html.escape(label)}</th>
                <td>{MARKET_TITLES[market]}</td>
                <td class="dim">{row['selection']}</td>
                <td data-v="{row['n']}">{row['n']:,}</td>
                <td data-v="{row['matches']}">{row['matches']:,}</td>
                <td data-v="{row['realized'] or 0}"><b>{_n(row['realized'], '.3f')}</b></td>
                <td data-v="{row['prod_predicted'] or 0}">{_n(row['prod_predicted'], '.3f')}</td>
                <td data-v="{row['prod_gap'] or 0}" class="{_cls(row['prod_gap'])}">{_n(row['prod_gap'], '+.3f')}</td>
                <td data-v="{row['candidate_predicted'] or 0}">{_n(row['candidate_predicted'], '.3f')}</td>
                <td data-v="{row['candidate_gap'] or 0}" class="{_cls(row['candidate_gap'])}">{_n(row['candidate_gap'], '+.3f')}</td>
                <td data-v="{row['brier_delta'] or 0}" class="{_cls(row['brier_delta'])}">{_n(row['brier_delta'])}</td>
                <td data-v="{row['p_value'] if row['p_value'] is not None else 1}">{_p(row['p_value'])}</td>
            </tr>""")
    return f"""
    <section class="panel" id="cross">
      <h2>Cross-section calibration</h2>
      <p class="sub">One bucket is score difference &times; quarter &times;
         possession, all three together. {len(rows):,} cells, of which
         {thin:,} sit under {config.MIN_CELL_MATCHES} matches and are dimmed.
         Predicted against the shared realized rate, one selection per market.
         Click a header to sort.</p>
      <div class="scroll">
      <table class="sortable" id="crossTable">
        <thead><tr><th>Cell</th><th>Market</th><th>Sel</th><th>N</th><th>Matches</th>
          <th>Realized</th><th>Prod</th><th>Gap</th><th>Cand</th><th>Gap</th>
          <th>&Delta;Brier</th><th>p</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      </div>
      <p class="note">Gap is realized minus predicted: positive means that
         stream underpriced the selection. &Delta;Brier is prod minus candidate,
         match-clustered, so positive favours the candidate. With this many
         cells a couple will clear p&nbsp;&lt;&nbsp;0.05 by chance, so read a
         lone significant cell as noise unless its neighbours agree.</p>
    </section>"""


def _pair_rows(pairs):
    out = []
    for p in pairs:
        basis = "line" if not p.same_line else "prob"
        winner = p.winner("line" if not p.same_line else "probability")
        winner_cell = ('<span class="dim">level</span>' if winner in (None, "tie")
                       else f'<span class="{"good" if winner == "candidate" else "bad"}">'
                            f'{"cand" if winner == "candidate" else "prod"}</span>')
        prod_error, candidate_error = p.errors("line" if not p.same_line else "probability")
        out.append(
            f'<tr>'
            f'<td>{"" if p.publish_time is None else html.escape(str(p.publish_time)[:19])}</td>'
            f'<td>{html.escape(p.match_code)}</td>'
            f'<td data-v="{p.drive_number}">{p.drive_number}</td>'
            f'<td data-v="{p.message_count}">{p.message_count}</td>'
            f'<td data-v="{abs(p.message_gap)}" class="{"" if p.message_gap == 0 else "warn"}">'
            f'{p.message_gap:+d}</td>'
            f'<td>{"Q" + str(p.period_number) if p.period_number and p.period_number <= 4 else ("OT" if p.period_number else "?")}</td>'
            f'<td data-v="{p.score_diff}">{p.score_diff:+d}</td>'
            f'<td>{"H" if p.offensive_team == "Home Team" else ("A" if p.offensive_team == "Away Team" else "?")}</td>'
            f'<td>{MARKET_TITLES.get(markets.market_group(p.market_id), "?")}</td>'
            f'<td>{markets.selection_label(p.market_id) or ""}</td>'
            f'<td data-v="{"" if p.prod_line is None else p.prod_line}">{"&mdash;" if p.prod_line is None else format(p.prod_line, "+.1f")}</td>'
            f'<td data-v="{"" if p.candidate_line is None else p.candidate_line}">{"&mdash;" if p.candidate_line is None else format(p.candidate_line, "+.1f")}</td>'
            f'<td data-v="{"" if p.line_delta is None else p.line_delta}" class="{"" if not p.line_delta else "warn"}">{"&mdash;" if p.line_delta is None else format(p.line_delta, ".1f")}</td>'
            f'<td data-v="{"" if p.prod_decimal is None else p.prod_decimal}">{"&mdash;" if p.prod_decimal is None else format(p.prod_decimal, ".2f")}</td>'
            f'<td data-v="{"" if p.candidate_decimal is None else p.candidate_decimal}">{"&mdash;" if p.candidate_decimal is None else format(p.candidate_decimal, ".2f")}</td>'
            f'<td data-v="{p.prod_probability}">{p.prod_probability:.4f}</td>'
            f'<td data-v="{p.candidate_probability}">{p.candidate_probability:.4f}</td>'
            f'<td data-v="{p.disagreement}"><b>{p.disagreement:.4f}</b></td>'
            f'<td data-v="{"" if p.realized is None else p.realized}">{"&mdash;" if p.realized is None else format(p.realized, ".0f")}</td>'
            f'<td>{_outcome(p.prod_outcome)}</td>'
            f'<td>{_outcome(p.candidate_outcome)}</td>'
            f'<td data-v="{"" if prod_error is None else prod_error}">{"&mdash;" if prod_error is None else format(prod_error, ".4f")}</td>'
            f'<td data-v="{"" if candidate_error is None else candidate_error}">{"&mdash;" if candidate_error is None else format(candidate_error, ".4f")}</td>'
            f'<td>{winner_cell}</td>'
            f'<td class="dim">{basis}</td>'
            f'</tr>')
    return "".join(out)


def _pair_table(pairs):
    return f"""
    <section class="panel" id="pairs">
      <h2>Every pair &mdash; widest disagreement first</h2>
      <p class="sub">{len(pairs):,} paired observations. Sorted on the probability
         gap, because that is where the two models actually said different things.
         Click any header to re-sort. BASIS says which rule decided the row: the
         probability where both quoted the same line, the line where they did not.</p>
      <div class="scroll">
      <table class="sortable" id="pairTable">
        <thead><tr>
          <th>Time</th><th>Match</th><th>Drive</th><th>Msg</th><th>&plusmn;Msg</th>
          <th>Qtr</th><th>Score</th><th>Poss</th>
          <th>Market</th><th>Sel</th>
          <th>Prod line</th><th>Cand line</th><th>&Delta;line</th>
          <th>Prod price</th><th>Cand price</th>
          <th>Prod prob</th><th>Cand prob</th><th>&Delta;prob</th>
          <th>Result</th><th>Prod</th><th>Cand</th>
          <th>Prod err</th><th>Cand err</th><th>Closer</th><th>Basis</th>
        </tr></thead>
        <tbody>{_pair_rows(pairs)}</tbody>
      </table>
      </div>
      <p class="note"><b>&plusmn;Msg</b> is how far the paired message sits from the
         snapshot's own. Zero is an exact hit; anything else means the two
         streams never both quoted that message and the nearest common one was
         used. Worth checking on a large &Delta;prob, because late in a close
         game a one-message offset can straddle a decisive play and manufacture
         a disagreement that is not a model difference.</p>
      <p class="note">RESULT is the realized margin (spread) or combined total
         (total); moneyline has none. PROD and CAND under it are whether that
         stream's selection won <em>at its own line</em> &mdash; they can differ
         when the lines do. Errors are probability distance from the outcome on a
         same-line row, and points from the result on a different-line row.</p>
    </section>"""


# ---------------------------------------------------------------------------

def _checks_summary(report):
    """One line stating whether the diagnostics passed, for the collapsed block."""
    issues = []
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
    return ('<span class="good">all pass</span> '
            '<span class="dim">integrity, mirror, spread reading, by day, single axes</span>')


def render(report, header, stats, pairs):
    verdict_class, verdict_text = _verdict(report)
    checks_summary = _checks_summary(report)
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
    --good:#1c7c4a; --bad:#b3261e; --warn:#8a6d1f; --accent:#2d4a7c;
    --head:#f0f0ec;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#17171a; --panel:#1f1f23; --ink:#ededea; --dim:#9a9a95; --line:#32323a;
      --good:#4cc281; --bad:#ef6f66; --warn:#d9b451; --accent:#8fb0e8;
      --head:#26262c;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#17171a; --panel:#1f1f23; --ink:#ededea; --dim:#9a9a95; --line:#32323a;
    --good:#4cc281; --bad:#ef6f66; --warn:#d9b451; --accent:#8fb0e8; --head:#26262c;
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
  .sub{{color:var(--dim);font-size:11.5px;margin:0 0 9px}}
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
  .dim{{color:var(--dim);font-weight:400}}
  .stats{{display:flex;flex-wrap:wrap;gap:6px 20px;margin:0}}
  .stats dt{{color:var(--dim);font-size:10.5px;text-transform:uppercase;
             letter-spacing:0.04em}}
  .stats dd{{margin:1px 0 0;font-size:13px;font-variant-numeric:tabular-nums}}
  .note{{color:var(--dim);font-size:11.5px;margin:9px 0 0}}
  .note.bad{{color:var(--bad)}}
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
    <span class="meta" title="settled matches in both streams">of <b>{header.get('paired_matches', 0)}</b> settled</span>
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
    <summary>Checks and breakdowns &mdash; {checks_summary}</summary>
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
</script>
</body>
</html>
"""


def write(path, report, header, stats, pairs):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render(report, header, stats, pairs))
    return path
