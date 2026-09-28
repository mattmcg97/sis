"""The totals lines, value by value: where prod's and the candidate's lines sit against the
points still to come, off the report's directional_pairs*.csv (no Snowflake).

At a snapshot with `on_board` points scored, a total line L asks for N = L - on_board more. The
points the rest of the game really produced, R, are lumpy (0, 3, 6, 7, 8, 10, 14 ...), so two lines
a point apart can sit either side of a value R often takes: prod needs 6.5 (a touchdown goes over)
and the candidate 7.5 (a touchdown and its kick stays under). The values between the two lines are
the ones where they settle differently -- the "straddled" values.

  gaps        the candidate's N minus prod's, overall and by quarter
  straddles   the lines grouped by the values between them: how often the game landed on one,
              and which line that favoured
  by N        each stream's lines at each N: its mean P(over) against how often the game went over
  real        the points still to come, value by value, by quarter and game state, beside where
              each stream's lines sat
  one score   each line's reach (within one score, two, beyond: totals_reach) against the other's
"""

import csv
import glob
import math
import os
from collections import Counter, defaultdict

from . import buckets, markets, totals_reach

OVER_MARKET = 54
MAX_N = 21


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


class Snap:
    """One total snapshot: both lines and prices, the score and what the rest of the game made."""

    def __init__(self, row):
        self.match_code = row["match_code"]
        self.drive_number = row["drive_number"]
        self.period_number = int(_num(row["period_number"])) if _num(row["period_number"]) else None
        self.score_p1 = int(_num(row["score_p1"]) or 0)
        self.score_p2 = int(_num(row["score_p2"]) or 0)
        self.offensive_team = row.get("offensive_team") or None
        self.on_board = self.score_p1 + self.score_p2
        self.prod_need = _num(row["prod_line"]) - self.on_board
        self.cand_need = _num(row["candidate_line"]) - self.on_board
        self.real = _num(row["realized"]) - self.on_board
        over = int(row["market_id"]) == OVER_MARKET
        p, c = _num(row["prod_probability"]), _num(row["candidate_probability"])
        self.prod_over = p if over else 1 - p
        self.cand_over = c if over else 1 - c
        self.one = totals_reach.one_score(self.score_p1, self.score_p2)

    @property
    def quarter(self):
        return buckets.period_bucket(self.period_number)

    @property
    def state(self):
        return totals_reach.game_state(self)

    def straddled(self):
        """The whole values of points still to come between the two lines."""
        lo, hi = sorted((self.prod_need, self.cand_need))
        return tuple(k for k in range(math.ceil(lo), math.floor(hi) + 1) if lo < k < hi)


def read(path, live_only=True):
    """The total snapshots in a directional_pairs csv: one per (match, snapshot), the over's row
    where there is one."""
    chosen = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if markets.market_group(int(row["market_id"])) != markets.TOTAL:
                continue
            if None in (_num(row["prod_line"]), _num(row["candidate_line"]), _num(row["realized"]),
                        _num(row["prod_probability"]), _num(row["candidate_probability"])):
                continue
            if live_only and not (row.get("prod_live") == "1" and row.get("candidate_live") == "1"):
                continue
            key = (row["match_code"], row["drive_number"])
            if key not in chosen or int(row["market_id"]) == OVER_MARKET:
                chosen[key] = row
    return [Snap(r) for r in chosen.values()]


def _pct(x):
    return "     -" if x is None else f"{100 * x:5.1f}%"


def _need(n):
    return f"{n:+.1f}" if n is not None else "-"


def _brier(prob, went_over):
    return (prob - went_over) ** 2


def gaps(snaps):
    """Lines of text: the candidate's N minus prod's, overall and by quarter."""
    def bucket(d):
        return "same" if abs(d) < 1e-9 else f"{d:+.0f}" if abs(d) <= 3 else ("< -3" if d < 0 else "> +3")
    order = ["< -3", "-3", "-2", "-1", "same", "+1", "+2", "+3", "> +3"]
    by = defaultdict(Counter)
    for s in snaps:
        by["all"][bucket(s.cand_need - s.prod_need)] += 1
        by[s.quarter][bucket(s.cand_need - s.prod_need)] += 1
    lines = ["\n  the candidate's line minus prod's (points)",
             f"  {'':10s} {'n':>7s} " + " ".join(f"{k:>6s}" for k in order)]
    for key in ["all"] + sorted(k for k in by if k != "all"):
        c = by[key]
        n = sum(c.values())
        lines.append(f"  {str(key):10s} {n:7,d} " + " ".join(_pct(c[k] / n) for k in order))
    return lines


def straddles(snaps, top=14):
    """Lines of text: the lines grouped by the values between them."""
    groups = defaultdict(list)
    for s in snaps:
        st = s.straddled()
        if st:
            groups[st].append(s)
    same = len(snaps) - sum(len(v) for v in groups.values())
    lines = [f"\n  where the lines settle differently: the points still to come between them "
             f"({same:,} of {len(snaps):,} snapshots have no whole value between the lines)",
             f"  {'values between':16s} {'n':>6s} {'cand higher':>11s} {'prod P(over)':>12s} "
             f"{'cand P(over)':>12s} {'real over prod':>14s} {'real over cand':>14s} "
             f"{'Brier prod':>10s} {'Brier cand':>10s} {'landed on one':>13s} {'Brier prod':>10s} "
             f"{'Brier cand':>10s}"]
    for st, ss in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:top]:
        n = len(ss)
        higher = sum(s.cand_need > s.prod_need for s in ss)
        landed = [s for s in ss if s.real in st]
        po = sum(s.prod_over for s in ss) / n
        co = sum(s.cand_over for s in ss) / n
        ro_p = sum(s.real > s.prod_need for s in ss) / n
        ro_c = sum(s.real > s.cand_need for s in ss) / n
        bp = sum(_brier(s.prod_over, s.real > s.prod_need) for s in ss) / n
        bc = sum(_brier(s.cand_over, s.real > s.cand_need) for s in ss) / n
        if landed:
            lp = sum(_brier(s.prod_over, s.real > s.prod_need) for s in landed) / len(landed)
            lc = sum(_brier(s.cand_over, s.real > s.cand_need) for s in landed) / len(landed)
            tail = f"{_pct(len(landed) / n):>13s} {lp:10.4f} {lc:10.4f}"
        else:
            tail = f"{_pct(0.0):>13s} {'-':>10s} {'-':>10s}"
        name = ",".join(str(k) for k in st) if len(st) <= 4 else f"{st[0]}..{st[-1]}"
        lines.append(f"  {name:16s} {n:6,d} {_pct(higher / n):>11s} {_pct(po):>12s} {_pct(co):>12s} "
                     f"{_pct(ro_p):>14s} {_pct(ro_c):>14s} {bp:10.4f} {bc:10.4f} {tail}")
    lines.append("  (landed on one: the rest of the game made exactly a value between the lines, so the "
                 "two settle differently -- the lower line's over and the higher line's under both come "
                 "in; the Brier there says whose price was nearer. Brier on each stream's own line, "
                 "lower is better)")
    return lines


def by_need(snaps):
    """Lines of text: each stream's lines at each N, its mean P(over) against the real over rate."""
    def key(n):
        return min(max(n, 0.5), MAX_N + 0.5) if n is not None else None
    tab = {name: defaultdict(lambda: [0, 0.0, 0]) for name in ("prod", "cand")}
    for s in snaps:
        for name, need, prob in (("prod", s.prod_need, s.prod_over), ("cand", s.cand_need, s.cand_over)):
            if s.real == need:
                continue
            c = tab[name][key(need)]
            c[0] += 1
            c[1] += prob
            c[2] += s.real > need
    lines = ["\n  at each points-needed value: how many lines sat there, their mean P(over), and how "
             "often the game went over",
             f"  {'needs':>6s}   {'prod n':>7s} {'P(over)':>8s} {'real':>7s} {'gap':>7s}   "
             f"{'cand n':>7s} {'P(over)':>8s} {'real':>7s} {'gap':>7s}"]
    keys = sorted(set(tab["prod"]) | set(tab["cand"]))
    for k in keys:
        cells = []
        for name in ("prod", "cand"):
            n, q, o = tab[name].get(k, (0, 0.0, 0))
            if n:
                cells.append(f"{n:7,d} {_pct(q / n):>8s} {_pct(o / n):>7s} {100 * (o - q) / n:+6.1f}")
            else:
                cells.append(f"{'':7s} {'':8s} {'':7s} {'':6s}")
        label = f"{k:.1f}" if k < MAX_N + 0.5 else f"{MAX_N}.5+"
        lines.append(f"  {label:>6s}   {cells[0]}   {cells[1]}")
    lines.append("  (gap: real over rate minus mean P(over), in points of probability; a line where the "
                 "game went over far more or less often than priced)")
    return lines


def real_values(snaps, values=(0, 3, 6, 7, 8, 10, 13, 14, 15, 16, 17, 21)):
    """Lines of text: the points still to come, value by value, by quarter and state, beside the
    median of each stream's N."""
    groups = defaultdict(list)
    for s in snaps:
        groups[(s.quarter, s.state)].append(s)
        groups[(s.quarter, totals_reach.ALL)].append(s)
    lines = ["\n  the points the rest of the game made, share at each value, and where the lines sat "
             "(median points needed)",
             f"  {'':34s} {'n':>6s} " + " ".join(f"{('=' + str(v)):>5s}" for v in values)
             + f" {'>21':>5s}   {'prod N':>6s} {'cand N':>6s}"]
    for (q, st), ss in sorted(groups.items(), key=lambda kv: (str(kv[0][0]),
                                                              totals_reach.STATES.index(kv[0][1]))):
        n = len(ss)
        c = Counter(int(s.real) for s in ss)
        med = lambda xs: sorted(xs)[len(xs) // 2]
        lines.append(f"  {(str(q) + ' ' + st)[:34]:34s} {n:6,d} "
                     + " ".join(f"{100 * c[v] / n:5.1f}" for v in values)
                     + f" {100 * sum(v for k, v in c.items() if k > 21) / n:5.1f}"
                     + f"   {med([s.prod_need for s in ss]):6.1f} {med([s.cand_need for s in ss]):6.1f}")
    return lines


def one_score(snaps):
    """Lines of text: each line's reach against the other's, and the real reach there."""
    cls = totals_reach.CLASSES
    tab = defaultdict(Counter)
    for s in snaps:
        k = (totals_reach.reach(s.prod_need, s.one), totals_reach.reach(s.cand_need, s.one))
        tab[k][totals_reach.reach(s.real, s.one)] += 1
    names = {totals_reach.WITHIN_ONE: "within 1", totals_reach.WITHIN_TWO: "within 2",
             totals_reach.BEYOND: "beyond"}
    lines = ["\n  each line's reach (prod -> candidate) and where the rest of the game really landed",
             f"  {'prod -> cand':22s} {'n':>7s} " + " ".join(f"{'real ' + names[c]:>16s}" for c in cls)]
    for a in cls:
        for b in cls:
            c = tab.get((a, b))
            if not c:
                continue
            n = sum(c.values())
            lines.append(f"  {names[a] + ' -> ' + names[b]:22s} {n:7,d} "
                         + " ".join(f"{_pct(c[x] / n):>16s}" for x in cls))
    return lines


def write_csv(path, snaps):
    fields = ["match_code", "snapshot", "quarter", "state", "score_p1", "score_p2", "on_board",
              "one_score", "prod_need", "cand_need", "straddled", "real_remaining", "prod_p_over",
              "cand_p_over"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        for s in snaps:
            w.writerow([s.match_code, s.drive_number, s.quarter, s.state, s.score_p1, s.score_p2,
                        s.on_board, s.one, s.prod_need, s.cand_need,
                        "|".join(str(k) for k in s.straddled()), s.real,
                        round(s.prod_over, 4), round(s.cand_over, 4)])


def run(paths, out_dir, live_only=True):
    """Print every section for each pairs file and write totals_lines_<name>.csv."""
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "directional_pairs*.csv"))) if os.path.isdir(p) else [p]
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for path in files:
        snaps = read(path, live_only)
        name = os.path.basename(path)[len("directional_pairs"):-len(".csv")].lstrip("_") or "candidate"
        lines = [f"\n==== {name} against prod: {len(snaps):,} total snapshots from {path}"
                 + (" (both live)" if live_only else "")]
        if snaps:
            lines += gaps(snaps) + straddles(snaps) + by_need(snaps) + real_values(snaps) + one_score(snaps)
            csv_path = os.path.join(out_dir, f"totals_lines_{name}.csv")
            write_csv(csv_path, snaps)
            lines.append(f"\n  -> {csv_path}")
        print("\n".join(lines))
        out.append((name, snaps))
    return out
