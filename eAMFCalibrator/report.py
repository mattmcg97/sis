"""Console tables and CSV output for a calibration run.

The CSVs are the point of the suite being re-runnable: each run writes a
cell-level summary keyed identically every time, so two runs (prod vs
candidate, or the same stream a week apart) diff cleanly with `compare`.
"""

import csv
import os

from . import buckets, config, metrics

CELL_FIELDS = [
    "stream", "score_diff", "time_bucket", "possession", "market", "selection",
    "n", "matches", "mean_predicted", "realized", "gap", "brier", "log_loss", "ece",
]

OBS_FIELDS = [
    "match_code", "drive_number", "period_number", "score_diff", "offensive_team",
    "market_id", "line", "probability", "outcome", "gap_seconds",
]


def _fmt(value, spec=".4f"):
    return "n/a" if value is None else format(value, spec)


def print_header(header, stats):
    print(f"\n{'=' * 78}")
    print(f"Stream: {header.get('stream')}  ({header.get('table')})")
    print(f"{'=' * 78}")
    print(f"Window start        : {config.CUTOFF_START}"
          f"{'' if config.CUTOFF_END is None else ' -> ' + str(config.CUTOFF_END)}")
    print(f"Quotes in window    : {header.get('window_rows', 0):,} rows, "
          f"{header.get('window_matches', 0):,} matches")
    print(f"Actual quote span   : {header.get('window_first')}  ->  {header.get('window_last')}")
    print(f"{'Settled ' + config.SPORT_CODE + ' universe':<20}: {header.get('universe_matches', 0):,} matches")
    print(f"Snapshot clock      : {header.get('clock_mode')}")
    print(f"Match tolerance     : {config.MATCH_TOLERANCE_SECONDS}s ({config.QUOTE_DIRECTION})")
    print(f"Time axis           : {buckets.time_axis_name()}")

    print("\nCoverage:")
    print(f"  drive snapshots built            : {stats.get('snapshots', 0):,}")
    print(f"  snapshots with >=1 matched quote : {stats.get('snapshots_matched', 0):,}")

    exact = stats.get("clock_exact", 0)
    interpolated = stats.get("clock_interpolated", 0)
    unresolved = stats.get("clock_unresolved", 0)
    from_feed = stats.get("clock_from_play_feed", 0)
    timed = exact + interpolated + unresolved + from_feed
    if timed:
        print("\n  Snapshot clock provenance:")
        if from_feed:
            print(f"    from the play feed's own column: {from_feed:,}")
        if exact or interpolated or unresolved:
            print(f"    exact message hit              : {exact:,} ({100 * exact / timed:.1f}%)")
            print(f"    interpolated between messages  : {interpolated:,} ({100 * interpolated / timed:.1f}%)")
            print(f"    unresolved (dropped)           : {unresolved:,} ({100 * unresolved / timed:.1f}%)")
        if exact and 100 * exact / timed < 50:
            print("    ^ under half the play feed's message counts appear in the stream.")
            print("      Check that both are numbering the same sequence before trusting")
            print("      the sub-3-second gaps.")
    for label, key in [
        ("snapshots with no clock", "snapshots_without_time"),
        ("no quote at all for market", "no_quote_for_market"),
        ("nearest quote outside tolerance", "quote_outside_tolerance"),
        ("null probability", "null_probability"),
        ("line could not be parsed", "unparsed_line"),
        ("push / unresolved", "pushes_or_unresolved"),
        ("matches with no play rows", "matches_without_plays"),
        ("matches with no final score", "matches_without_final"),
    ]:
        if stats.get(key):
            print(f"  {label:<33}: {stats[key]:,}")


def print_overall(summary):
    print(f"\n{'-' * 78}\nOverall calibration\n{'-' * 78}")
    print(f"  {'GROUP':<12}{'N':>9}{'PRED':>9}{'REAL':>9}{'GAP':>9}"
          f"{'BRIER':>10}{'LOGLOSS':>10}{'ECE':>9}")
    for group in ["ALL"] + sorted(k for k in summary if k != "ALL"):
        s = summary[group]
        if not s["n"]:
            continue
        print(f"  {group:<12}{s['n']:>9,}{_fmt(s['mean_predicted'], '.3f'):>9}"
              f"{_fmt(s['realized'], '.3f'):>9}{_fmt(s['gap'], '+.3f'):>9}"
              f"{_fmt(s['brier']):>10}{_fmt(s['log_loss']):>10}{_fmt(s['ece']):>9}")
    print("\n  GAP is realized minus predicted. Positive means the model is")
    print("  underpricing the selection; negative means overpricing.")


def print_reliability(observations):
    pairs = [(o.probability, o.outcome) for o in observations]
    if not pairs:
        return
    print(f"\n{'-' * 78}\nReliability curve (all markets pooled)\n{'-' * 78}")
    print(f"  {'BIN':<14}{'N':>9}{'PRED':>9}{'REAL':>9}{'GAP':>9}")
    for b in metrics.reliability_bins(pairs, config.N_RELIABILITY_BINS):
        if not b["n"]:
            continue
        label = f"{b['lo']:.1f}-{b['hi']:.1f}"
        print(f"  {label:<14}{b['n']:>9,}{_fmt(b['predicted'], '.3f'):>9}"
              f"{_fmt(b['realized'], '.3f'):>9}{_fmt(b['gap'], '+.3f'):>9}")


def cell_rows(stream, cells, cell_matches):
    rows = []
    for key in sorted(cells, key=buckets.sort_key):
        pairs = cells[key]
        s = metrics.summarize(pairs, config.N_RELIABILITY_BINS)
        score_b, time_b, poss, market, selection = key
        rows.append({
            "stream": stream,
            "score_diff": score_b,
            "time_bucket": time_b,
            "possession": poss,
            "market": market,
            "selection": selection,
            "n": s["n"],
            "matches": len(cell_matches[key]),
            "mean_predicted": s["mean_predicted"],
            "realized": s["realized"],
            "gap": s["gap"],
            "brier": s["brier"],
            "log_loss": s["log_loss"],
            "ece": s["ece"],
        })
    return rows


def print_cells(rows):
    print(f"\n{'-' * 100}\nCalibration by cell "
          f"(score diff x {buckets.time_axis_name().lower()} x possession x selection)\n{'-' * 100}")
    print(f"  {'SCORE':<9}{buckets.time_axis_name():<12}{'POSS':<7}{'MARKET':<11}{'SEL':<7}"
          f"{'N':>8}{'MATCH':>7}{'PRED':>8}{'REAL':>8}{'GAP':>8}{'BRIER':>9}")
    for r in rows:
        print(f"  {r['score_diff']:<9}{r['time_bucket']:<12}{r['possession']:<7}"
              f"{r['market']:<11}{r['selection']:<7}{r['n']:>8,}{r['matches']:>7,}"
              f"{_fmt(r['mean_predicted'], '.3f'):>8}{_fmt(r['realized'], '.3f'):>8}"
              f"{_fmt(r['gap'], '+.3f'):>8}{_fmt(r['brier'], '.4f'):>9}")


def print_worst(rows):
    eligible = [r for r in rows
                if r["n"] >= config.MIN_CELL_OBSERVATIONS
                and r["matches"] >= config.MIN_CELL_MATCHES
                and r["gap"] is not None]
    if not eligible:
        print(f"\n  No cell reaches {config.MIN_CELL_OBSERVATIONS} observations across "
              f"{config.MIN_CELL_MATCHES} matches yet -- too little data to call anything mispriced.")
        return
    worst = sorted(eligible, key=lambda r: abs(r["gap"]), reverse=True)[:15]
    print(f"\n{'-' * 78}\nWorst cells by |gap| "
          f"(min {config.MIN_CELL_OBSERVATIONS} obs, {config.MIN_CELL_MATCHES} matches)\n{'-' * 78}")
    for r in worst:
        print(f"  {r['score_diff']:<9}{r['time_bucket']:<10}{r['possession']:<6}"
              f"{r['market']:<10}{r['selection']:<6} n={r['n']:<6} matches={r['matches']:<5} "
              f"pred={_fmt(r['mean_predicted'], '.3f')} real={_fmt(r['realized'], '.3f')} "
              f"gap={_fmt(r['gap'], '+.3f')}")


def write_cells_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CELL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  cell summary -> {path}")


def write_observations_csv(path, observations):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OBS_FIELDS)
        writer.writeheader()
        for o in observations:
            writer.writerow({
                "match_code": o.match_code,
                "drive_number": o.drive_number,
                "period_number": o.period_number,
                "score_diff": o.score_diff,
                "offensive_team": o.offensive_team,
                "market_id": o.market_id,
                "line": o.line,
                "probability": o.probability,
                "outcome": int(o.outcome),
                "gap_seconds": o.gap_seconds,
            })
    print(f"  observations -> {path}")


def read_cells_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def print_comparison(rows_a, rows_b, label_a, label_b, limit=40):
    """Diff two cell summaries on the cells they share.

    Ordered by how much the two disagree, capped at `limit` rows -- the
    full grid runs to hundreds of cells and the ones worth reading are
    always at the top.
    """
    key = lambda r: (r["score_diff"], r["time_bucket"], r["possession"], r["market"], r["selection"])
    index_b = {key(r): r for r in rows_b}

    shared = [(a, index_b[key(a)]) for a in rows_a if key(a) in index_b]
    print(f"\n{'=' * 104}\n{label_a} vs {label_b}: {len(shared)} shared cells\n{'=' * 104}")
    if not shared:
        print("  No cells in common -- were both runs built with the same buckets and time axis?")
        return

    print(f"  {'SCORE':<9}{'TIME':<10}{'POSS':<6}{'MARKET':<10}{'SEL':<6}"
          f"{'N_A':>7}{'N_B':>7}{'GAP_A':>9}{'GAP_B':>9}{'|A|-|B|':>10}{'BRIER_A':>10}{'BRIER_B':>10}")
    scored = []
    for a, b in shared:
        gap_a, gap_b = _float(a["gap"]), _float(b["gap"])
        if gap_a is None or gap_b is None:
            continue
        improvement = abs(gap_a) - abs(gap_b)
        scored.append((improvement, a, b, gap_a, gap_b))

    ranked = sorted(scored, key=lambda t: abs(t[0]), reverse=True)
    for improvement, a, b, gap_a, gap_b in ranked[:limit]:
        print(f"  {a['score_diff']:<9}{a['time_bucket']:<10}{a['possession']:<6}"
              f"{a['market']:<10}{a['selection']:<6}{int(a['n']):>7,}{int(b['n']):>7,}"
              f"{gap_a:>+9.3f}{gap_b:>+9.3f}{improvement:>+10.3f}"
              f"{_fmt(_float(a['brier'])):>10}{_fmt(_float(b['brier'])):>10}")

    if len(ranked) > limit:
        print(f"  ... {len(ranked) - limit} further cells not shown (smallest disagreement).")

    better = sum(1 for imp, *_ in scored if imp > 0)
    worse = sum(1 for imp, *_ in scored if imp < 0)
    print(f"\n  GAP_A is {label_a}, GAP_B is {label_b}. |A|-|B| positive means {label_b}")
    print(f"  sits closer to realized in that cell.")
    print(f"  {label_b} closer in {better} cells, {label_a} closer in {worse}, "
          f"tied in {len(scored) - better - worse}.")
