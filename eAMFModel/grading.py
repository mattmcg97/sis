"""Scoring a version's prices against prod's on the same PLAY_OVER quotes: one row per prod
quote, and the Brier score by cell with paired deltas against prod."""

import random
from collections import defaultdict
from dataclasses import dataclass

GROUPS = {50: "moneyline", 51: "moneyline", 52: "spread", 53: "spread",
          54: "total", 55: "total"}


@dataclass
class Row:
    """One prod quote: the market, its line, probability and outcome, and the snapshot it is at."""
    match_code: str
    message: int
    period: int
    home_score: int
    away_score: int
    market_id: int
    prod_line: float
    prod_probability: float
    prod_outcome: int
    prod_live: object = None        # 1 / 0, or None when the export predates the column


def _brier(p, y):
    return (p - y) ** 2


def summarise(graded, names, key=lambda row: GROUPS[row.market_id], n_boot=500, seed=0):
    """Brier by cell for prod and each version, with paired deltas against prod clustered on
    matches (all priced at prod's line)."""
    cells = defaultdict(list)
    for match_code, row, probs in graded:
        cells[key(row)].append((match_code, row, probs))
        cells["all"].append((match_code, row, probs))
    rng = random.Random(seed)
    out = {}
    for cell, items in cells.items():
        n = len(items)
        res = {"pairs": n, "matches": len({m for m, _, _ in items})}
        res["prod"] = sum(_brier(r.prod_probability, r.prod_outcome) for _, r, _ in items) / n
        per_match = defaultdict(lambda: defaultdict(float))
        for m, r, probs in items:
            base = _brier(r.prod_probability, r.prod_outcome)
            per_match[m]["n"] += 1
            for name in names:
                per_match[m][name] += base - _brier(probs[name], r.prod_outcome)
        matches = list(per_match)
        for name in names:
            res[name] = sum(_brier(probs[name], r.prod_outcome) for _, r, probs in items) / n
            deltas = []
            for _ in range(n_boot):
                sample = [rng.choice(matches) for _ in matches]
                tot = sum(per_match[m][name] for m in sample)
                cnt = sum(per_match[m]["n"] for m in sample)
                deltas.append(tot / cnt if cnt else 0.0)
            deltas.sort()
            res[name + "_delta"] = res["prod"] - res[name]
            res[name + "_ci"] = (deltas[int(0.025 * n_boot)], deltas[int(0.975 * n_boot) - 1])
            res[name + "_matches_better"] = sum(1 for m in matches if per_match[m][name] > 0)
        out[cell] = res
    return out


def print_summary(summary, names, title):
    """Print summarise's table."""
    print(f"\n  {title}")
    head = f"  {'cell':<12}{'pairs':>7}{'match':>6}{'prod':>8}"
    for name in names:
        head += f"{name:>8}"
    for name in names:
        head += f"  {'d ' + name + ' (95% CI)':<24}"
    print(head)
    order = sorted(summary, key=lambda c: (c == "all", str(c)))
    for cell in order:
        r = summary[cell]
        line = f"  {str(cell):<12}{r['pairs']:>7}{r['matches']:>6}{r['prod']:>8.4f}"
        for name in names:
            line += f"{r[name]:>8.4f}"
        for name in names:
            lo, hi = r[name + "_ci"]
            line += f"  {r[name + '_delta']:+.4f} [{lo:+.4f},{hi:+.4f}] "
        print(line)
    print("  d = prod Brier - version Brier: positive means the version beat prod.")
