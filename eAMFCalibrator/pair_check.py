"""Is prod's first quote on a message the one it stands by?

The directional report pairs a candidate with prod's FIRST live row on the first message after each
PLAY_OVER (directional.index_by_message: among live rows of a message the earliest wins). The
betting simulation reads prod's LATEST live row at the moment of the bet. Where prod publishes
several rows on one message and the first is a quick price it corrects a moment later, the report
grades the candidate against a price prod did not stand by, and a Brier edge there need not survive
at the moments bets are struck.

This reads prod's moneyline alone (no candidate, no simulation) and, at the message each PLAY_OVER
pairs on, sets prod's first live row against its last:

    python -m eAMFCalibrator pair-check --since "2026-10-01 00:00:00" --until "2026-10-13 00:00:00"

Brier first - last above zero: the last row was closer to the result, so the first was the staler
price. Set against the report's moneyline edge (prod's Brier less the candidate's per match), it
says how much of that edge the pairing alone could account for.
"""

import csv
import os
import random
import statistics
from bisect import bisect_left
from collections import defaultdict

from . import config, directional, markets, snowflake_io

MONEYLINE_IDS = [m for m in markets.MARKET_IDS if markets.market_group(m) == markets.MONEYLINE]
FIELDS = ("match_code", "market_id", "play_over_message", "message", "rows", "first_prob",
          "last_prob", "first_publish", "last_publish", "seconds_first_to_last", "outcome")


def live_rows_by_message(quote_rows):
    """(match, market) -> {message: [(publish time, probability 0-1)] in publish order} of the
    moneyline's live rows (fetch_quotes rows, ordered by publish time within a match and market)."""
    out = defaultdict(lambda: defaultdict(list))
    for match, market, publish, prob, _, _, msg, _, active in (r[:9] for r in quote_rows):
        if msg is None or prob is None or int(market) not in MONEYLINE_IDS or not directional.is_live(active):
            continue
        out[(match, int(market))][int(msg)].append((publish, float(prob) / 100.0))
    return out


def _seconds(a, b):
    try:
        return (b - a).total_seconds()
    except (TypeError, AttributeError):
        return None


def at_play_overs(quote_rows, play_overs, finals):
    """One row per PLAY_OVER and moneyline selection: prod's first and last live row on the first
    message it quoted in the PLAY_OVER's window (directional.first_message_in's rule), and the
    selection's result (pushes and matches without a final left out)."""
    by = live_rows_by_message(quote_rows)
    out = []
    for match, snaps in play_overs.items():
        final = finals.get(match)
        if not snaps or final is None or final[0] is None or final[1] is None:
            continue
        windows = directional.play_over_snapshots(match, snaps)
        for market in MONEYLINE_IDS:
            rows = by.get((match, market))
            if not rows:
                continue
            outcome = markets.resolve(market, None, final[0], final[1])
            if outcome is None:
                continue
            msgs = sorted(rows)
            for snap, end in windows:
                start = snap.event_message_count
                i = bisect_left(msgs, start)
                if i >= len(msgs) or msgs[i] >= end:
                    continue
                at = rows[msgs[i]]
                (t0, p0), (t1, p1) = at[0], at[-1]
                out.append(dict(match_code=match, market_id=market, play_over_message=start, message=msgs[i],
                                rows=len(at), first_prob=p0, last_prob=p1, first_publish=t0, last_publish=t1,
                                seconds_first_to_last=_seconds(t0, t1), outcome=int(bool(outcome))))
    return out


def brier_gap(rows, n_boot=2000, seed=0):
    """(mean over matches of Brier(first) - Brier(last), 95% bootstrap interval over matches, matches):
    above zero, prod's last row on the message was the closer to the result."""
    per = defaultdict(list)
    for r in rows:
        y = r["outcome"]
        per[r["match_code"]].append((r["first_prob"] - y) ** 2 - (r["last_prob"] - y) ** 2)
    deltas = [sum(v) / len(v) for v in per.values()]
    if not deltas:
        return None, None, None, 0
    rnd = random.Random(seed)
    m = len(deltas)
    means = sorted(sum(deltas[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
    return sum(deltas) / m, means[int(0.025 * n_boot)], means[min(int(0.975 * n_boot), n_boot - 1)], m


def summary(rows):
    """The lines pair-check prints."""
    n = len(rows)
    if not n:
        return ["  no PLAY_OVER moneyline quotes to compare"]
    multi = [r for r in rows if r["rows"] > 1]
    moved = [r for r in multi if abs(r["first_prob"] - r["last_prob"]) > 5e-4]
    gaps = sorted(r["seconds_first_to_last"] for r in moved if r["seconds_first_to_last"] is not None)
    L = [f"  {n:,} PLAY_OVER moneyline quotes on {len({r['match_code'] for r in rows}):,} matches "
         "(the message the directional report pairs on)",
         f"  more than one live row on that message: {len(multi):,} ({100 * len(multi) / n:.1f}%)",
         f"  ...and the first and last differ:      {len(moved):,} ({100 * len(moved) / n:.1f}%)"]
    if moved:
        diffs = [abs(r["first_prob"] - r["last_prob"]) for r in moved]
        L.append(f"  where they differ: mean |first - last| {statistics.mean(diffs):.4f}, "
                 f"90th percentile {sorted(diffs)[int(0.9 * (len(diffs) - 1))]:.4f}")
    if gaps:
        L.append(f"  seconds from the first row to the last: median {statistics.median(gaps):.1f}, "
                 f"90th percentile {gaps[int(0.9 * (len(gaps) - 1))]:.1f}")
    mean, lo, hi, m = brier_gap(rows)
    L.append(f"  Brier first - last, per match: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] over {m:,} matches")
    L.append("  (above zero: prod's last row was the closer, so the report's pairing on the first one "
             "flatters a candidate by about this much on the moneyline)")
    return L


def run(cur, out_dir):
    """Prod's moneyline at every PLAY_OVER of the window: the console summary, and
    pair_check.csv in out_dir."""
    codes = snowflake_io.in_start_order(cur, sorted(snowflake_io.match_universe(cur, config.STREAMS["prod"])))
    print(f"  {len(codes):,} settled matches in the window", flush=True)
    rows = []
    chunk = config.MATCH_CHUNK_SIZE
    for s in range(0, len(codes), chunk):
        batch = codes[s:s + chunk]
        quotes = snowflake_io.fetch_quotes(cur, config.STREAMS["prod"], batch)
        finals = snowflake_io.fetch_final_scores(cur, batch)
        prod_by = defaultdict(list)
        for r in quotes:
            prod_by[r[0]].append(r)
        play_overs, _ = snowflake_io._play_over_snapshots(cur, batch, prod_by_match=prod_by)
        rows += at_play_overs(quotes, play_overs, finals)
        print(f"  chunk {s // chunk + 1}: {len(batch)} matches, {len(rows):,} PLAY_OVER quotes so far", flush=True)
    lines = summary(rows)
    print("\n" + "\n".join(lines))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pair_check.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  -> {path}")
    return path
