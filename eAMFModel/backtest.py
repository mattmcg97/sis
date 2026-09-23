"""Score model versions against prod and candidate, offline, on a pairs CSV.

Input is the `directional_pairs.csv` the calibrator writes: one row per
(snapshot, market) with the game state, prod's and the candidate's line and
probability, and how each resolved. The model is Markov in the state the
row carries (score, period, message, possession, down, distance, field), so
every row can be re-priced exactly as the live stream would have priced it
at that message -- at PROD'S line, so its probability answers the same
question prod's did and settles on the same outcome.

The pre-match prior is prod's spread and total at the first snapshot of the
match (0-0, the kickoff): the stand-in for "the pre-match model's numbers".

Per match, the period starts are not in the file -- the snapshots are
sparse -- so each is placed midway between the last snapshot of the
period before and the first of its own. Live, the feed gives them exactly.
"""

import csv
import random
from collections import defaultdict
from dataclasses import dataclass

from . import pricer
from .pricer import HOME, AWAY, GameState
from .strength import Prior

GROUPS = {50: "moneyline", 51: "moneyline", 52: "spread", 53: "spread",
          54: "total", 55: "total"}


def _int(value):
    return int(float(value)) if value not in ("", None) else None


def _float(value):
    return float(value) if value not in ("", None) else None


def _side(label):
    if not label:
        return None
    return HOME if label.lower().startswith("home") else AWAY if label.lower().startswith("away") else None


@dataclass
class Row:
    match_code: str
    message: int
    period: int
    home_score: int
    away_score: int
    offense: str
    field_position: int
    down: int
    distance: int
    market_id: int
    prod_line: float
    candidate_line: float
    prod_probability: float
    candidate_probability: float
    prod_outcome: int
    candidate_outcome: int
    prod_live: object = None        # 1 / 0, or None when the file predates the column
    candidate_live: object = None

    @property
    def live(self):
        """Both streams tradeable, or unknown (older files)."""
        return self.prod_live is None or (self.prod_live and self.candidate_live)


def load_pairs(path):
    """Rows by match. Files written since the liveness columns went in say
    whether each quote was tradeable; older ones do not, and on those the
    late spread and total rows include prod quotes that were suspended --
    priced near 50% whatever the line -- which flatter any rival. Only the
    moneyline, which is live throughout, is a clean read on such a file."""
    rows = defaultdict(list)
    with open(path, newline="") as handle:
        for r in csv.DictReader(handle):
            rows[r["match_code"]].append(Row(
                match_code=r["match_code"],
                message=_int(r["message_count"]),
                period=_int(r["period_number"]),
                home_score=_int(r["score_p1"]) or 0,
                away_score=_int(r["score_p2"]) or 0,
                offense=_side(r["offensive_team"]),
                field_position=_int(r["field_position"]),
                down=_int(r["down_number"]),
                distance=_int(r["distance"]),
                market_id=_int(r["market_id"]),
                prod_line=_float(r["prod_line"]),
                candidate_line=_float(r["candidate_line"]),
                prod_probability=_float(r["prod_probability"]),
                candidate_probability=_float(r["candidate_probability"]),
                prod_outcome=_int(r["prod_outcome"]),
                candidate_outcome=_int(r["candidate_outcome"]),
                prod_live=_int(r.get("prod_live")),
                candidate_live=_int(r.get("candidate_live")),
            ))
    for match_rows in rows.values():
        match_rows.sort(key=lambda row: (row.message, row.market_id))
    return rows


@dataclass
class MatchContext:
    prior: Prior
    opening_receiver: str
    period_starts: dict


def context(match_rows, model=None):
    """Prior, opening receiver and period starts for one match, or None.

    With a model, the prior is fitted to prod's kickoff probabilities too
    (pricer.Model.fit_prior); without one it is read off the lines alone.
    """
    first_message = match_rows[0].message
    opening = [r for r in match_rows if r.message == first_message]
    lines = {r.market_id: r.prod_line for r in opening}
    probs = {r.market_id: r.prod_probability for r in opening}
    if lines.get(52) is None or lines.get(54) is None:
        return None
    if any(r.home_score or r.away_score for r in opening):
        return None
    if model is None:
        prior = Prior.from_lines(lines[52], lines[54])
    else:
        prior = model.fit_prior(lines[52], lines[54], ml_home=probs.get(50),
                                spread_home=probs.get(52), over=probs.get(54))

    by_period = defaultdict(list)
    for r in match_rows:
        by_period[r.period].append(r.message)
    starts = {}
    previous_last = 0
    for period in sorted(by_period):
        first = min(by_period[period])
        starts[period] = first if period == 1 else (previous_last + first) / 2.0
        previous_last = max(by_period[period])
    starts[1] = min(starts.get(1, first_message), first_message)
    return MatchContext(prior, opening[0].offense, starts)


def state_for(row, ctx):
    start = ctx.period_starts.get(row.period, row.message)
    half_first = 1 if row.period <= 2 else 3 if row.period <= 4 else row.period
    half_start = ctx.period_starts.get(half_first)
    if half_start is None:
        # The half's first quarter had no snapshot: an average one before this.
        half_start = start - (71.0 if half_first == 1 else 81.0)
    return GameState(
        period=row.period,
        elapsed_in_period=max(0.0, row.message - start),
        elapsed_in_half=max(0.0, row.message - half_start),
        home_score=row.home_score,
        away_score=row.away_score,
        offense=row.offense,
        down=row.down,
        field_position=row.field_position,
        distance=row.distance,
        opening_receiver=ctx.opening_receiver,
    )


def price_rows(match_rows, ctx, model):
    """{(message, market_id): probability} at prod's line."""
    out = {}
    books = {}
    for row in match_rows:
        key = row.message
        if key not in books:
            books[key] = model.book(ctx.prior, state_for(row, ctx))
        out[(row.message, row.market_id)] = books[key].prob(row.market_id, row.prod_line)
    return out


def has_liveness(path):
    with open(path, newline="") as handle:
        return "prod_live" in (csv.DictReader(handle).fieldnames or ())


def run(path, versions, limit=None, fit_prior=True, require_live=True):
    """[(match, row, {name: probability})] for every gradeable row.

    `require_live` drops pairs where either stream was suspended, as the
    calibrator does -- when the file says; see load_pairs.
    """
    data = load_pairs(path)
    models = {name: pricer.Model(params) for name, params in versions.items()}
    graded = []
    skipped = 0
    for i, (match_code, match_rows) in enumerate(sorted(data.items())):
        if limit is not None and i >= limit:
            break
        contexts = {name: context(match_rows, model if fit_prior else None)
                    for name, model in models.items()}
        if any(ctx is None for ctx in contexts.values()):
            skipped += 1
            continue
        priced = {name: price_rows(match_rows, contexts[name], model)
                  for name, model in models.items()}
        for row in match_rows:
            if row.prod_outcome is None or row.prod_probability is None:
                continue
            if require_live and not row.live:
                continue
            graded.append((match_code, row,
                           {name: priced[name][(row.message, row.market_id)] for name in models}))
    return graded, skipped


def _brier(p, y):
    return (p - y) ** 2


def summarise(graded, names, key=lambda row: GROUPS[row.market_id], n_boot=500, seed=0):
    """Brier by cell for prod, candidate and each version, with paired
    deltas against prod clustered on matches.

    The candidate is scored on its OWN line and outcome (as the calibrator
    does); prod and the model versions share prod's line.
    """
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
        cand = [(r.candidate_probability, r.candidate_outcome) for _, r, _ in items
                if r.candidate_outcome is not None and r.candidate_probability is not None]
        res["candidate"] = sum(_brier(p, y) for p, y in cand) / len(cand) if cand else None
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
    print(f"\n  {title}")
    head = f"  {'cell':<12}{'pairs':>7}{'match':>6}{'prod':>8}{'cand':>8}"
    for name in names:
        head += f"{name:>8}"
    for name in names:
        head += f"  {'d ' + name + ' (95% CI)':<24}"
    print(head)
    order = sorted(summary, key=lambda c: (c == "all", str(c)))
    for cell in order:
        r = summary[cell]
        line = f"  {str(cell):<12}{r['pairs']:>7}{r['matches']:>6}{r['prod']:>8.4f}"
        line += f"{r['candidate']:>8.4f}" if r["candidate"] is not None else f"{'-':>8}"
        for name in names:
            line += f"{r[name]:>8.4f}"
        for name in names:
            lo, hi = r[name + "_ci"]
            line += f"  {r[name + '_delta']:+.4f} [{lo:+.4f},{hi:+.4f}] "
        print(line)
    print("  d = prod Brier - version Brier: positive means the version beat prod.")
