"""One-screen HTML summary of a directional run.

Self-contained: no external CSS, fonts or scripts, so the file opens
anywhere. Built from directional.build_summary() so the numbers are the
same ones the console prints.
"""

import html
import os

from . import config, labels, markets

MARKET_TITLES = {
    markets.MONEYLINE: "Moneyline",
    markets.SPREAD: "Spread",
    markets.TOTAL: "Total",
}


def _pct(value):
    return "&mdash;" if value is None else f"{100 * value:.1f}%"


def _num(value, spec="+.4f"):
    return "&mdash;" if value is None else format(value, spec)


def _p(value):
    if value is None:
        return "&mdash;"
    return "&lt;0.001" if value < 0.001 else f"{value:.3f}"


def _ci(summary, spec="+.4f"):
    if summary.get("ci_low") is None:
        return "&mdash;"
    return f"[{format(summary['ci_low'], spec)}, {format(summary['ci_high'], spec)}]"


def _verdict(brier, votes):
    """A sentence, and a class that colours the strip."""
    mean = brier.get("mean")
    low, high = brier.get("ci_low"), brier.get("ci_high")
    if mean is None or low is None:
        return "neutral", "Not enough paired data to separate the two models."

    straddles = low <= 0 <= high
    better = mean > 0
    side = "candidate" if better else "prod"
    if straddles:
        return ("neutral",
                f"Points {'to the candidate' if better else 'to prod'}, but the "
                f"confidence interval crosses zero &mdash; not separated yet.")
    return (("good" if better else "bad"),
            f"The {side} is better, and the interval excludes zero.")


def _market_rows(block, metric_key, spec, clustered_key):
    """Per market, with its OWN match-clustered test.

    The pooled figure averages across markets, so an effect confined to one
    of them is diluted by the flat ones. Each market carries its own p.
    """
    rows = []
    for group in (markets.MONEYLINE, markets.SPREAD, markets.TOTAL):
        tallied = block["by_market"].get(group)
        if not tallied or not tallied["n"]:
            continue
        detail = block.get("markets", {}).get(group, {})
        clustered = detail.get(clustered_key, {})
        win = tallied["candidate_win_rate"]
        cls = "" if win is None else ("good" if win > 0.5 else "bad")
        p_value = clustered.get("p_value")
        p_cls = "good" if (p_value is not None and p_value < 0.05) else "dim"
        rows.append(f"""<tr>
            <th>{MARKET_TITLES[group]}</th>
            <td>{tallied['n']:,}</td>
            <td>{tallied['n_matches']:,}</td>
            <td class="{cls}">{_pct(win)}</td>
            <td>{_num(clustered.get('mean'), spec)}</td>
            <td class="dim">{_ci(clustered, spec)}</td>
            <td class="{p_cls}">{_p(p_value)}</td>
        </tr>""")
    if not rows:
        rows.append('<tr><td colspan="7" class="dim">no comparable pairs</td></tr>')
    return "\n".join(rows)


def _block_table(title, subtitle, block):
    """Line mode is measured in points, so it cannot share Brier's column."""
    overall = block["overall"]
    votes = block["votes"]
    line_mode = overall.get("mode") == "line"

    if line_mode:
        # Squaring a points error gives squared points, which reads as noise.
        summary = block["mae"]
        metric_key, spec, clustered_key = "mae_delta", "+.3f", "mae"
        column = "&Delta;points"
        stat_label = "Paired &Delta;points per match"
    else:
        summary = block["brier"]
        metric_key, spec, clustered_key = "brier_delta", "+.4f", "brier"
        column = "&Delta;Brier"
        stat_label = "Paired &Delta;Brier per match"

    return f"""
    <section class="panel">
      <h2>{title}</h2>
      <p class="sub">{subtitle}</p>
      <table>
        <thead><tr><th>Market</th><th>Pairs</th><th>Matches</th>
          <th>Cand win</th><th>{column}</th><th>95% CI</th><th>p</th></tr></thead>
        <tbody>{_market_rows(block, metric_key, spec, clustered_key)}</tbody>
        <tfoot><tr>
          <th>All</th>
          <td>{overall['n']:,}</td>
          <td>{overall['n_matches']:,}</td>
          <td>{_pct(overall['candidate_win_rate'])}</td>
          <td>{_num(summary.get('mean'), spec)}</td>
          <td class="dim">{_ci(summary, spec)}</td>
          <td>{_p(summary.get('p_value'))}</td>
        </tr></tfoot>
      </table>
      <dl class="stats">
        <div><dt>{stat_label}</dt>
             <dd>{_num(summary.get('mean'), spec)}
                 <span class="dim">{_ci(summary, spec)}</span></dd></div>
        <div><dt>p (match-clustered)</dt><dd>{_p(summary.get('p_value'))}</dd></div>
        <div><dt>Match vote</dt>
             <dd>{votes['candidate']}&ndash;{votes['prod']}
                 <span class="dim">({votes['tie']} level)</span></dd></div>
      </dl>
    </section>"""


def render(summary, header, stats):
    lines = summary["lines"]
    same = summary["same_line"]
    different = summary["different_line"]
    verdict_class, verdict_text = _verdict(same["brier"], same["votes"])

    exact = stats.get("exact_message_pair", 0)
    offset = stats.get("offset_message_pair", 0)
    exact_rate = f"{100 * exact / (exact + offset):.1f}%" if exact + offset else "&mdash;"

    line_rows = []
    for group in (markets.MONEYLINE, markets.SPREAD, markets.TOTAL):
        bucket = lines["by_market"].get(group)
        if not bucket or not bucket["n"]:
            continue
        same_rate = bucket["same"] / bucket["n"]
        line_rows.append(f"""<tr>
            <th>{MARKET_TITLES[group]}</th>
            <td>{bucket['n']:,}</td>
            <td>{100 * same_rate:.1f}%</td>
            <td>{bucket['prod_half']:,} / {bucket['prod_whole']:,}</td>
            <td>{bucket['candidate_half']:,} / {bucket['candidate_whole']:,}</td>
        </tr>""")

    window = html.escape(str(config.CUTOFF_START))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eAMF Model Check</title>
<style>
  :root {{
    --bg: #f7f7f5; --panel: #ffffff; --ink: #1a1a18; --dim: #6b6b66;
    --line: #e2e2dd; --good: #1c7c4a; --bad: #b3261e; --neutral: #8a6d1f;
    --accent: #2d4a7c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg: #17171a; --panel: #1f1f23; --ink: #ededea; --dim: #9a9a95;
      --line: #32323a; --good: #4cc281; --bad: #ef6f66; --neutral: #d9b451;
      --accent: #8fb0e8;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #17171a; --panel: #1f1f23; --ink: #ededea; --dim: #9a9a95;
    --line: #32323a; --good: #4cc281; --bad: #ef6f66; --neutral: #d9b451;
    --accent: #8fb0e8;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: var(--bg); color: var(--ink);
    font: 13px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 16px;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; }}
  header {{ display: flex; flex-wrap: wrap; gap: 4px 18px; align-items: baseline;
            margin-bottom: 10px; }}
  h1 {{ font-size: 17px; margin: 0; letter-spacing: -0.01em; }}
  .meta {{ color: var(--dim); font-size: 11.5px; }}
  .meta b {{ color: var(--ink); font-weight: 600; }}
  .verdict {{ padding: 9px 12px; border-radius: 7px; margin-bottom: 12px;
              border-left: 3px solid var(--neutral); background: var(--panel);
              font-size: 13px; }}
  .verdict.good {{ border-left-color: var(--good); }}
  .verdict.bad {{ border-left-color: var(--bad); }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .panel {{ background: var(--panel); border: 1px solid var(--line);
            border-radius: 7px; padding: 11px 13px; }}
  h2 {{ font-size: 13px; margin: 0 0 2px; letter-spacing: -0.005em; }}
  .sub {{ color: var(--dim); font-size: 11.5px; margin: 0 0 9px; }}
  table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: 4px 5px; border-bottom: 1px solid var(--line); }}
  thead th {{ color: var(--dim); font-weight: 500; font-size: 10.5px;
              text-transform: uppercase; letter-spacing: 0.04em; }}
  tbody th, tfoot th {{ text-align: left; font-weight: 600; }}
  tfoot th, tfoot td {{ border-bottom: none; border-top: 1px solid var(--line);
                        padding-top: 6px; }}
  .good {{ color: var(--good); font-weight: 600; }}
  .bad {{ color: var(--bad); font-weight: 600; }}
  .dim {{ color: var(--dim); font-weight: 400; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 4px 20px; margin: 9px 0 0; }}
  .stats div {{ min-width: 0; }}
  .stats dt {{ color: var(--dim); font-size: 10.5px; text-transform: uppercase;
               letter-spacing: 0.04em; }}
  .stats dd {{ margin: 1px 0 0; font-variant-numeric: tabular-nums; font-size: 13px; }}
  .wide {{ grid-column: 1 / -1; }}
  .note {{ color: var(--dim); font-size: 11.5px; margin: 10px 0 0; }}
  .note b {{ color: var(--ink); }}
  @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>eAMF &mdash; candidate vs prod</h1>
    <span class="meta">from <b>{window}</b></span>
    <span class="meta"><b>{header.get('paired_matches', 0)}</b> matches</span>
    <span class="meta"><b>{summary['pairs']:,}</b> pairs</span>
    <span class="meta"><b>{stats.get('snapshots', 0)}</b> drive snapshots</span>
    <span class="meta">exact-message pairing <b>{exact_rate}</b></span>
  </header>

  <div class="verdict {verdict_class}">{verdict_text}</div>

  <div class="grid">
    {_block_table(
        "Same line &mdash; whose probability was closer",
        "Both streams quoted the same line, so their probabilities answer the "
        "same question. This is the clean comparison.",
        same)}

    {_block_table(
        "Different line &mdash; whose line was closer",
        "Lines differ, so probabilities are not comparable. Scored on which "
        "line landed nearer the actual margin or total.",
        different)}

    <section class="panel wide">
      <h2>Line agreement</h2>
      <p class="sub">The two streams quoted the same line on
         <b>{_pct(lines['same_rate'])}</b> of pairs. On the
         <b>{lines.get('differing_n', 0):,}</b> that differ, the gap runs to a median of
         <b>{_num(lines.get('differing_median'), '.2f')}</b> and a max of
         <b>{_num(lines.get('differing_max'), '.2f')}</b>.</p>
      <table>
        <thead><tr><th>Market</th><th>Pairs</th><th>Same line</th>
          <th>Prod half / whole</th><th>Cand half / whole</th></tr></thead>
        <tbody>{''.join(line_rows) or '<tr><td colspan="5" class="dim">no lined markets</td></tr>'}</tbody>
      </table>
      <p class="note">
        <b>How to read this.</b> Each stream is graded against its
        <em>own</em> line &mdash; a total of 45 is over 44.5 but under 46.5, so
        scoring the candidate against prod&rsquo;s line would mark it on a
        question it never asked. &Delta;Brier is prod minus candidate, so
        positive favours the candidate. The paired figure is computed per match
        and then averaged, because drives inside a match are not independent;
        the interval is a bootstrap over matches. If it crosses zero, the data
        has not separated the two models. Win rate is a direction check only
        &mdash; it discards magnitude, so a genuinely better model can sit near
        50%.
      </p>
    </section>
  </div>
</div>
</body>
</html>
"""


def write(path, summary, header, stats):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(labels.relabel(render(summary, header, stats)))
    return path
