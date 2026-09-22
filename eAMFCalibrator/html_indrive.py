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

DRIVE_TITLES = {
    indrive.TOUCHDOWN: "Touchdown",
    indrive.FIELD_GOAL: "Field goal",
    indrive.EXTRA_POINT: "PAT or two-point",
    indrive.SCORE: "Other points",
    indrive.POINTS_AGAINST: "Defence scored",
    indrive.NO_POINTS: "No points",
}

ENDED_TITLES = {
    indrive.HANDOVER: "Handed over",
    indrive.SAME_TEAM: "Same team label after",
    indrive.MATCH_END: "Last of the match",
}

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
    <th title="share of decided moves that went the expected way &mdash; green from 70%, then 62%, 56%, 52%; below 52% is red, because 50% is a coin">Right</th>
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


def _drives_panel(census, outcomes):
    if not outcomes:
        return ""
    import collections as _c
    rows = []
    for outcome, row in census.items():
        if not row["n"]:
            continue
        plays = "&mdash;" if row["mean_plays"] is None else f"{row['mean_plays']:.1f}"
        rows.append(f"""<tr>
            <th>{DRIVE_TITLES.get(outcome, outcome)}</th>
            <td>{row['n']:,}</td><td>{_pct(row['n'] / len(outcomes))}</td>
            <td>{row['points']:,}</td>
            <td>{row['points'] / row['n']:.2f}</td>
            <td>{plays}</td><td>{row['matches']:,}</td>
        </tr>""")
    ended = _c.Counter(o.ended for o in outcomes)
    end_rows = "".join(
        f"<tr><th>{ENDED_TITLES.get(k, k)}</th><td>{ended[k]:,}</td>"
        f"<td>{_pct(ended[k] / len(outcomes))}</td></tr>"
        for k in (indrive.HANDOVER, indrive.SAME_TEAM, indrive.MATCH_END)
        if ended.get(k))
    points = sum(o.points_for + o.points_against for o in outcomes)
    matches = len({o.match_code for o in outcomes})
    return f"""
    <section class="panel" id="drives">
      <h2>What each drive produced</h2>
      <p class="count">{len(outcomes):,} drives &middot; {matches:,} matches
        &middot; {len(outcomes) / matches:.1f} per match &middot;
        {points:,} points &middot; {points / len(outcomes):.2f} per drive</p>
      <div class="cols">
        <div>
          <table>
            <thead><tr><th>Outcome</th><th>Drives</th><th>Share</th>
              <th>Points</th><th>Per drive</th><th>Plays</th>
              <th>Matches</th></tr></thead>
            <tbody>{''.join(rows)}</tbody>
          </table>
        </div>
        <div>
          <table>
            <thead><tr><th>How it ended</th><th>Drives</th>
              <th>Share</th></tr></thead>
            <tbody>{end_rows}</tbody>
          </table>
        </div>
      </div>
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


SEVERITY = {indrive.ERROR: ("bad", "Error"),
            indrive.WARN: ("warn", "Warn"),
            indrive.NOTE: ("dim", "Note")}


def _checks_panel(findings):
    if not findings:
        return """
    <section class="panel" id="checks">
      <h2>Checks</h2>
      <p class="count"><span class="good">all pass</span></p>
    </section>"""
    order = {indrive.ERROR: 0, indrive.WARN: 1, indrive.NOTE: 2}
    rows = []
    for severity, subject, message in sorted(findings,
                                             key=lambda f: (order[f[0]], f[1])):
        tone, label = SEVERITY[severity]
        rows.append(f"""<tr>
            <th class="{tone}">{label}</th>
            <td>{html.escape(str(subject))}</td>
            <td class="wrap">{html.escape(message)}</td>
        </tr>""")
    errors = sum(1 for f in findings if f[0] == indrive.ERROR)
    return f"""
    <section class="panel" id="checks">
      <h2>Checks</h2>
      <p class="count">{len(findings):,} findings &middot; {errors:,} errors
        &middot; an error is about the check, not the models</p>
      <table>
        <thead><tr><th>Severity</th><th>Subject</th><th class="wrap">Finding</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>"""


def render(result, census, transitions, stats=None, findings=(),
           drive_census=None, outcomes=()):
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
    <a href="#checks">Checks</a>
    <a href="#drives">Drives</a>
    <a href="#plays">Plays</a>
    <a href="#{directional.PROD}">Prod</a>
    <a href="#{directional.CANDIDATE}">Candidate</a>
    <a href="#h2h">Head to head</a>
  </nav>
  {_checks_panel(findings)}
  {_drives_panel(drive_census or {}, outcomes)}
  {_census_panel(census, transitions)}
  {streams}
  {_head_to_head_panel(result)}
  {_dropped_panel(stats or {})}
</div>
</body>
</html>"""
