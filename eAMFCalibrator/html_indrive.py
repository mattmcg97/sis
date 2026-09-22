"""The in-drive reaction report, as one self-contained HTML file.

Same stylesheet as the main report, same rule: headings, tables and
numbers. What it answers is narrower -- when a play went well for the
side with the ball, did the price follow? -- so it is its own file rather
than a tenth panel on a page that is already long.
"""

import html

from . import directional, html_style, indrive

TITLES = {
    indrive.TOUCHDOWN: "Touchdown",
    indrive.FIELD_GOAL: "Field goal",
    indrive.EXTRA_POINT: "PAT or two-point",
    indrive.SCORE: "Other points",
    indrive.FIRST_DOWN: "First down",
    indrive.BIG_GAIN: f"Gain of {indrive.BIG_GAIN_YARDS}+",
    indrive.SHORT_GAIN: "Short gain",
    indrive.NO_GAIN: "No gain",
    indrive.LOSS: "Loss",
    indrive.FAILED_CONVERSION: "Failed conversion",
    indrive.POSSESSION_LOST: "Possession lost",
    indrive.POINTS_AGAINST: "Points against",
}

SIGN_TITLES = {1: "good", -1: "bad", 0: "--"}

# The ramp is centred on a COIN, not on zero. 50% is a model that reacted
# at random, so the green end is a long way above it and anything at or
# below it is the red end. The cut points are how far above a coin a rate
# has to sit to earn each step.
RATE_STEPS = (0.02, 0.06, 0.12, 0.20)


def _pct(value):
    return "&mdash;" if value is None else f"{100 * value:.1f}%"


def _n(value, spec=".4f"):
    return "&mdash;" if value is None else format(value, spec)


def _p(value):
    if value is None:
        return "&mdash;"
    return "&lt;0.001" if value < 0.001 else f"{value:.3f}"


def _ci(block):
    if block.get("ci_low") is None:
        return "&mdash;"
    return f"[{_pct(block['ci_low'])}, {_pct(block['ci_high'])}]"


def _rate_class(rate):
    """Green where the price followed the play, red where it did not.

    A rate barely above a coin and a rate below one both land on the red
    end, which is the point: a model that gets the direction right 51% of
    the time has not shown it is reading the game.
    """
    if rate is None:
        return ""
    if rate < 0.5:
        return "g4"
    step = next((i for i, cut in enumerate(RATE_STEPS)
                 if (rate - 0.5) <= cut), len(RATE_STEPS))
    return f"g{len(RATE_STEPS) - step}"


def _rate_key():
    """The ramp spelled out, so no cell is read by colour alone."""
    labels = [f"&ge;{100 * (0.5 + cut):.0f}%" for cut in reversed(RATE_STEPS)]
    labels.append(f"&lt;{100 * (0.5 + RATE_STEPS[0]):.0f}%")
    swatches = "".join(f'<span class="key g{i}">{label}</span>'
                       for i, label in enumerate(labels))
    return (f'<p class="gapkey"><span class="klab">Right</span>{swatches}'
            f'<span class="klab">50% is a coin</span></p>')


def _row(label, block, tag=""):
    return f"""<tr>
        <th>{label}{tag}</th>
        <td>{block['n']:,}</td>
        <td>{block['decided']:,}</td>
        <td>{block['matches']:,}</td>
        <td class="{_rate_class(block['rate'])}"><b>{_pct(block['rate'])}</b></td>
        <td class="dim">{_ci(block)}</td>
        <td>{_pct(block['flat_share'])}</td>
        <td>{block['n_prob']:,}</td>
        <td>{_n(block['mean_signed'], '+.4f')}</td>
        <td>{block['n_line']:,}</td>
        <td>{_n(block['mean_line_signed'], '+.2f')}</td>
        <td>{_p(block['p_value'])}</td>
    </tr>"""


HEAD = """<thead><tr><th>Split</th><th>Moves</th>
    <th title="moves that moved at all: a flat price is neither right nor wrong">Decided</th>
    <th>Matches</th>
    <th title="share of decided moves that went the expected way">Right</th>
    <th title="match-clustered">95% CI</th>
    <th title="share of moves where nothing changed">Flat</th>
    <th title="moves scored on the probability, because the line held">On prob</th>
    <th title="mean probability change, oriented so positive is the expected direction">Signed</th>
    <th title="moves scored on the line, because it moved">On line</th>
    <th title="mean line change, in points, oriented the same way">Signed</th>
    <th title="against a coin, match-clustered">p</th></tr></thead>"""


def _census_panel(census, transitions):
    total = len(transitions)
    rows = []
    for outcome, row in census.items():
        if not row["n"]:
            continue
        rows.append(f"""<tr>
            <th>{TITLES.get(outcome, outcome)}</th>
            <td class="dim">{SIGN_TITLES[row['sign']]}</td>
            <td>{row['n']:,}</td><td>{row['in_drive']:,}</td>
            <td>{row['matches']:,}</td><td>{_pct(row['n'] / total)}</td>
        </tr>""")
    in_drive = sum(1 for t in transitions if t.in_drive)
    scorable = sum(1 for t in transitions if t.scorable)
    return f"""
    <section class="panel" id="plays">
      <h2>What the plays were</h2>
      <p class="count">{total:,} transitions &middot; {in_drive:,} inside a
        drive &middot; {total - in_drive:,} at a drive's end &middot;
        {scorable:,} carrying a direction</p>
      <table>
        <thead><tr><th>Outcome</th>
          <th title="for the side in possession">For offence</th>
          <th>Transitions</th><th>In drive</th><th>Matches</th>
          <th>Share</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _stream_panel(stream, s):
    scope = "".join(_row(label, s[key])
                    for label, key in (("Overall", "overall"),
                                       ("Inside a drive", "in_drive"),
                                       ("At a drive's end", "ending"))
                    if s[key]["n"])
    outcomes = "".join(
        _row(TITLES.get(o, o), s["by_outcome"][o])
        for o in indrive.OUTCOME_ORDER
        if s["by_outcome"].get(o, {}).get("n"))
    periods = "".join(_row(str(k), s["by_period"][k])
                      for k in sorted(s["by_period"]))
    basis_rows = "".join(_row(str(k), s["by_basis"][k])
                        for k in sorted(s["by_basis"], key=str))
    market_rows = "".join(_row(str(k), s["by_market"][k])
                          for k in sorted(s["by_market"], key=str))
    selections = "".join(
        _row(f"{k[0]} {k[1]}", s["by_selection"][k])
        for k in sorted(s["by_selection"], key=lambda k: (str(k[0]), str(k[1]))))
    return f"""
    <section class="panel" id="{stream}">
      <h2>{html.escape(stream.capitalize())}</h2>
      {_rate_key()}
      <h3>Scope</h3>
      <table>{HEAD}<tbody>{scope}</tbody></table>
      <h3>By outcome</h3>
      <table>{HEAD}<tbody>{outcomes}</tbody></table>
      <h3>By period</h3>
      <table>{HEAD}<tbody>{periods}</tbody></table>
      <h3>By basis</h3>
      <table>{HEAD}<tbody>{basis_rows}</tbody></table>
      <h3>By market</h3>
      <table>{HEAD}<tbody>{market_rows}</tbody></table>
      <h3>By selection</h3>
      <table>{HEAD}<tbody>{selections}</tbody></table>
    </section>"""


def _head_to_head_panel(result):
    rows = []
    for label, key in (("Overall", "all"), ("Inside a drive", "in_drive"),
                       ("At a drive's end", "ending")):
        h = result["head_to_head"][key]
        if not h["pairs"]:
            continue
        rows.append(f"""<tr>
            <th>{label}</th>
            <td>{h['pairs']:,}</td><td>{h['decided']:,}</td>
            <td>{h['matches']:,}</td>
            <td class="{_rate_class(h['rate'])}"><b>{_pct(h['rate'])}</b></td>
            <td class="dim">{_ci(h)}</td>
            <td>{h['ties']:,}</td><td>{h['both_flat']:,}</td>
            <td>{_p(h['p_value'])}</td>
        </tr>""")
    return f"""
    <section class="panel" id="h2h">
      <h2>Head to head</h2>
      <table>
        <thead><tr><th>Split</th><th>Pairs</th>
          <th title="pairs where exactly one stream got the direction right">Decided</th>
          <th>Matches</th>
          <th title="candidate share of the decided pairs">Cand</th>
          <th>95% CI</th>
          <th title="both right or both wrong">Ties</th>
          <th title="neither price moved">Both flat</th>
          <th>p</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def _dropped_panel(stats):
    keys = [("move_scored_on_prob", "Scored on the probability"),
            ("move_scored_on_line", "Scored on the line"),
            ("transition_no_direction", "No direction to check"),
            ("move_missing_quote", "No quote at one end"),
            ("move_not_live", "Not live at one end"),
            ("move_line_unreadable", "No readable line at one end")]
    rows = "".join(f"<tr><th>{label}</th><td>{stats[key]:,}</td></tr>"
                   for key, label in keys if stats.get(key))
    if not rows:
        return ""
    return f"""
    <section class="panel">
      <h2>Where the moves went</h2>
      <table>
        <thead><tr><th>Reason</th><th>N</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>"""


def render(result, census, transitions, stats=None):
    streams = "".join(_stream_panel(stream, result["streams"][stream])
                      for stream in (directional.PROD, directional.CANDIDATE)
                      if result["streams"][stream]["overall"]["n"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eAMF &mdash; in-drive reaction</title>
{html_style.CSS}
</head>
<body>
<div class="wrap">
  <header>
    <h1>eAMF &mdash; in-drive reaction</h1>
    <span class="meta"><b>{result['matches']:,}</b> matches</span>
    <span class="meta"><b>{result['transitions']:,}</b> transitions</span>
    <span class="meta"><b>{result['moves']:,}</b> moves</span>
    <span class="meta"><b>{result['in_drive']:,}</b> in drive</span>
    <span class="meta"><b>{result['ending']:,}</b> drive end</span>
  </header>
  <nav>
    <a href="#plays">Plays</a>
    <a href="#{directional.PROD}">Prod</a>
    <a href="#{directional.CANDIDATE}">Candidate</a>
    <a href="#h2h">Head to head</a>
  </nav>
  {_census_panel(census, transitions)}
  {streams}
  {_head_to_head_panel(result)}
  {_dropped_panel(stats or {})}
</div>
</body>
</html>"""
