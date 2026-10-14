"""v2.1's rules out of the data: every real decision in the scouting export swept for the situations
where an option never happens, or all but never (the few real ones that did are listed, to look at),
rather than rules written by hand.

    python -m eAMFModel v21-sweep --out eAMFCalibrator/out_v21_sweep

For each decision v2.1 makes (v21.DECISIONS) and each of its options, a tree splits the real decisions
on their full state -- the period, the seconds left in the half, the deciding side's lead, yards to the
goal, yards to go, the down, both sides' timeouts left and the play before (SWEEP_FEATURES) -- so as to
separate where the option was taken from where it wasn't. Its largest boxes of the state that hold at
least MIN_REGION real decisions, the option taken by RARE of them or fewer (and every part of the box
alike), are the situations it never or all but never happens in.

v21.build learns the same off its own matches (the decisions its sim makes) and the sim strikes those
options out there for every gamer, before the league's shares and any gamer's lean; real decisions
inside a box that took its option anyway are left out of the gamers' leans.

The sweep also checks the game's own end (v21.auto_end): the side ahead in Q4 whose downs left at a
40-second play clock outlast the clock, the defense out of timeouts -- the game is over.
"""

import csv

import numpy as np

from . import sim14 as s14, v21

SWEEP_FEATURES = {
    "fourth": ("period", "left", "lead", "togo", "dist", "own_to", "opp_to", "prev"),
    "early_fg": ("period", "left", "lead", "togo", "dist", "down", "own_to", "opp_to", "prev"),
    "kickoff": ("period", "left", "lead", "own_to", "opp_to"),
    "two": ("period", "left", "lead", "own_to", "opp_to"),
    "to_def": ("period", "left", "lead", "togo", "dist", "down", "own_to", "opp_to", "prev"),
    "to_off": ("period", "left", "lead", "togo", "dist", "down", "own_to", "opp_to", "prev"),
}
MIN_REGION = 30
RARE = 0.01
MAX_DEPTH = 8
EXAMPLES = 6
SHOWN = 25                    # situations listed for each option in the text (the CSV has them all)
# the sweep reads every kick on downs 1-3 in range, not only those in the half's last minute the sim decides at
LABELS = {"early_fg": "a field goal on downs 1-3, in range"}


def fine(name, pts):
    """{feature: int array} of real decision points' full state (SWEEP_FEATURES)."""
    out = {}
    for f in SWEEP_FEATURES[name]:
        v = np.array([q.get(f, -1) for q in pts], float)
        if f == "period":
            v = np.minimum(v, 5)
        out[f] = np.round(v).astype(np.int64)
    return out


# ---------------------------------------------------------------- the tree

def _h(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _best_split(F, y, min_leaf):
    """(feature, threshold) of the split (<= threshold left) that most lowers the entropy of y, with
    min_leaf or more each side; (None, None) for none."""
    n, k = len(y), y.sum()
    parent = n * _h(k / n)
    best, out = 1e-9, (None, None)
    for j in range(F.shape[1]):
        o = np.argsort(F[:, j], kind="stable")
        v, ck = F[o, j], np.cumsum(y[o])
        i = np.flatnonzero(v[:-1] != v[1:])
        i = i[(i + 1 >= min_leaf) & (n - i - 1 >= min_leaf)]
        if not len(i):
            continue
        nl, kl = i + 1, ck[i]
        cost = nl * _h(kl / nl) + (n - nl) * _h((k - kl) / (n - nl))
        b = int(np.argmax(parent - cost))
        if parent - cost[b] > best:
            best, out = parent - cost[b], (j, int(v[i[b]]))
    return out


def _grow(F, y, idx, box, depth, min_leaf, max_depth):
    n, k = len(idx), int(y[idx].sum())
    node = dict(box=box, n=n, k=k, idx=idx, kids=None)
    if k == 0 or k == n or depth >= max_depth or n < 2 * min_leaf:
        return node
    j, t = _best_split(F[idx], y[idx], min_leaf)
    if j is None:
        return node
    lo, hi = box.get(j, (None, None))
    left, right = idx[F[idx, j] <= t], idx[F[idx, j] > t]
    node["kids"] = (_grow(F, y, left, {**box, j: (lo, t)}, depth + 1, min_leaf, max_depth),
                    _grow(F, y, right, {**box, j: (t + 1, hi)}, depth + 1, min_leaf, max_depth))
    return node


def find_regions(F, y, min_region=MIN_REGION, rare=RARE, max_depth=MAX_DEPTH):
    """[(box, rows)] the largest boxes of the state ({column: (lo, hi)}, None unbounded) holding
    min_region or more decisions, the option (y) taken by `rare` of them or fewer, every part of the
    box alike; rows: the decisions inside."""
    F, y = np.asarray(F, np.int64), np.asarray(y, bool)
    if len(y) < min_region:
        return []
    root = _grow(F, y, np.arange(len(y)), {}, 0, max(5, min_region // 2), max_depth)

    def clean(node):                       # no part of min_region or more takes the option more than rarely
        if node["n"] >= min_region and node["k"] > rare * node["n"]:
            return False
        return node["kids"] is None or all(clean(c) for c in node["kids"])

    def ok(node):
        return node["n"] >= min_region and clean(node)

    out = []

    def collect(node):
        if ok(node):
            out.append((node["box"], node["idx"]))
        elif node["kids"] is not None:
            for c in node["kids"]:
                collect(c)
    collect(root)
    return out


# ---------------------------------------------------------------- the rules

def learn(points, min_region=MIN_REGION, rare=RARE):
    """{name: [(option, {feature: (lo, hi)})]} off real decision points ({name: [dict]})."""
    rules = {}
    for name, spec in v21.DECISIONS.items():
        pts = points.get(name) or []
        if len(pts) < min_region:
            continue
        cols = fine(name, pts)
        feats = SWEEP_FEATURES[name]
        F = np.stack([cols[f] for f in feats], axis=1)
        chosen = np.array([q["choice"] for q in pts])
        for opt in range(len(spec["options"])):
            for box, _ in find_regions(F, chosen == opt, min_region, rare):
                if box:                                   # a box with no bounds would strike the option out altogether
                    rules.setdefault(name, []).append((opt, {feats[j]: b for j, b in box.items()}))
    return rules


def mask(rules, name, cols):
    """(decisions, options) the options the learned rules leave (True) for decisions with this full
    state (cols: {feature: array}); None where no rule touches them. A decision every option of which
    is struck out keeps them all."""
    got = (rules or {}).get(name)
    if not got:
        return None
    n = len(np.asarray(next(iter(cols.values()))))
    ok = np.ones((n, len(v21.DECISIONS[name]["options"])), bool)
    for opt, box in got:
        inside = np.ones(n, bool)
        for f, (lo, hi) in box.items():
            v = np.asarray(cols[f])
            if lo is not None:
                inside &= v >= lo
            if hi is not None:
                inside &= v <= hi
        ok[inside, opt] = False
    ok[~ok.any(1)] = True
    return ok if not ok.all() else None


# ---------------------------------------------------------------- reading a rule

PERIODS = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4", 5: "OT"}
DOMAIN = {"period": (1, 5), "down": (1, 4), "own_to": (-1, 3), "opp_to": (-1, 3), "prev": (0, 5)}


def _mmss(s):
    return f"{int(s) // 60}:{int(s) % 60:02d}"


def _span(lo, hi, f=str, unit=""):
    if lo is None and hi is None:
        return "any"
    if lo is None:
        return f"{f(hi)}{unit} or less"
    if hi is None:
        return f"{f(lo)}{unit} or more"
    return f"{f(lo)}{unit}" if lo == hi else f"{f(lo)}-{f(hi)}{unit}"


def describe(box):
    """A rule's box in words."""
    parts = []
    for f in ("period", "left", "lead", "togo", "dist", "down", "own_to", "opp_to", "prev"):
        if f not in box:
            continue
        lo, hi = box[f]
        dlo, dhi = DOMAIN.get(f, (None, None))
        lo = dlo if lo is None else lo
        hi = dhi if hi is None else hi
        if f == "period":
            parts.append(PERIODS[lo] if lo == hi else f"{PERIODS[lo]}-{PERIODS[min(hi, 5)]}")
        elif f == "left":
            parts.append(f"{_span(lo, hi, _mmss)} left in the half")
        elif f == "lead":
            if hi is not None and hi < 0:
                parts.append(f"behind by {_span(-hi, None if lo is None else -lo)}")
            elif lo is not None and lo > 0:
                parts.append(f"ahead by {_span(lo, hi)}")
            elif lo == hi == 0:
                parts.append("level")
            elif lo is None or hi is None:
                parts.append(f"lead {_span(lo, hi, lambda v: f'{v:+d}')}")
            else:
                parts.append(f"lead {lo:+d} to {hi:+d}")
        elif f == "togo":
            parts.append(f"{_span(lo, hi)} yards from the goal")
        elif f == "dist":
            parts.append(f"{_span(lo, hi)} to go")
        elif f == "down":
            parts.append(f"down {_span(lo, hi)}")
        elif f in ("own_to", "opp_to"):
            who = "own" if f == "own_to" else "the other side's"
            if lo == hi:
                parts.append(f"{who} timeouts {'unknown' if lo < 0 else lo}")
            elif lo <= 0:
                parts.append(f"{who} timeouts 0" if hi <= 0 else f"{who} timeouts {hi} or fewer")
            elif hi >= 3:
                parts.append(f"{who} timeouts {lo} or more")
            else:
                parts.append(f"{who} timeouts {lo}-{hi}")
        elif f == "prev":
            parts.append("after " + " or ".join(v21.PREV_NAMES[v] for v in range(lo, hi + 1)))
    return ", ".join(parts) if parts else "anywhere"


# ---------------------------------------------------------------- the sweep

def sweep_points(matches, history=None):
    """Every real decision point of the matches ({code: rows}), each with its match and gamer."""
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    out = {k: [] for k in v21.DECISIONS}
    for code, rows in matches.items():
        rows = v21._with_side(rows)
        pair, _ = v21.match_people(code, rows, hist)
        got = v21.decision_points(rows)
        for name in v21.DECISIONS:
            for q in got[name]:
                out[name].append(dict(q, code=code, gamer=pair[q["side"]]))
    return out


def auto_end_check(matches):
    """The real states the game's own end (v21.auto_end) covers: (states, play carried on after it --
    another snap in Q4 with time on the clock --, the state's matches where it did)."""
    n = went = 0
    where = []
    for code, rows in matches.items():
        rows = v21._with_side(rows)
        for a, b in zip(rows, rows[1:]):
            if a["play_kind"] not in s14.SNAP_KINDS or a.get("period") != "4" or a.get("offense") not in v21.TEAMS \
                    or not a.get("down") or not a.get("clock_seconds"):
                continue
            m = s14._margin(a, a["offense"])
            _, opp = v21._tos_left(a, a["offense"], 4)
            c = s14._f(a["clock_seconds"])
            if m is None or opp != 0 or c <= 0 or not v21.auto_end(4, c, m, s14._i(a["down"]), 0):
                continue
            n += 1
            if b.get("period") == "4" and b["play_kind"] in ("SCRIMMAGE", "TOUCHDOWN", "TURNOVER_ON_DOWNS", "FIELD_GOAL",
                                                             "PUNT") and (s14._f(b.get("clock_seconds")) or 0) > 0:
                went += 1
                where.append((code, a["message"], c, m))
    return n, went, where


def sweep(matches, history=None, min_region=MIN_REGION, rare=RARE):
    """{name: [(option, box, n, taken, examples)]} the never and all-but-never situations of every
    decision over these matches, and the auto-end check."""
    pts = sweep_points(matches, history)
    found = {}
    for name, spec in v21.DECISIONS.items():
        got = pts[name]
        if len(got) < min_region:
            continue
        cols = fine(name, got)
        feats = SWEEP_FEATURES[name]
        F = np.stack([cols[f] for f in feats], axis=1)
        chosen = np.array([q["choice"] for q in got])
        rows = []
        for opt in range(len(spec["options"])):
            for box, idx in find_regions(F, chosen == opt, min_region, rare):
                hit = idx[chosen[idx] == opt]                 # an empty box: never or all but never anywhere
                rows.append((opt, {feats[j]: b for j, b in box.items()}, len(idx), len(hit),
                             [got[i] for i in hit[:EXAMPLES]]))
        found[name] = (len(got), rows)
    return found, auto_end_check(matches), {k: len(v) for k, v in pts.items()}


def _example(q):
    lead = q["lead"]
    return (f"{q['code']} {q['gamer'] or '?'} {PERIODS.get(min(q['period'], 5), q['period'])} {_mmss(q['clock'])}"
            f" {'+' if lead > 0 else ''}{lead}")


def report(found, auto, counts, min_region=MIN_REGION, rare=RARE):
    lines = [f"  v2.1 sweep: where each decision's options never happen, or all but never ({min_region}+ real"
             f" decisions, the option {100 * rare:.0f}% of them or fewer; every part of the situation alike)", ""]
    n, went, where = auto
    lines.append(f"  the game's own end -- the side ahead in Q4, its downs left x 40s more than the clock, the"
                 f" defense out of timeouts: {n:,} real states, play went on after {went:,}"
                 + (": " + ", ".join(f"{c} #{m} {_mmss(t)} +{d}" for c, m, t, d in where[:10]) if went else ""))
    for name, spec in v21.DECISIONS.items():
        title = LABELS.get(name, spec["label"])
        if name not in found:
            lines += ["", f"  {title}: {counts.get(name, 0):,} real -- too few to sweep"]
            continue
        total, rows = found[name]
        lines += ["", f"  {title} (by the {spec['who']}): {total:,} real"]
        for opt, label in enumerate(spec["options"]):
            mine = sorted((r for r in rows if r[0] == opt), key=lambda r: -r[2])
            if not mine:
                continue
            lines.append(f"    {label}: never or all but never in"
                         + (f" (the {SHOWN} largest of {len(mine)}; all in the CSV)" if len(mine) > SHOWN else ""))
            for _, box, n_, k_, ex in mine[:SHOWN]:
                tag = "never" if k_ == 0 else f"{k_} of them ({100 * k_ / n_:.1f}%)"
                lines.append(f"      {describe(box)}: {n_:,} real, {tag}")
                if k_:
                    lines.append("        who did: " + "; ".join(_example(q) for q in ex))
    lines += ["", "  (left: seconds left in the half -- in overtime, in the period; lead: the deciding side's; timeouts"
                  " unknown where the export has none. v21-build learns the same off its own matches and strikes"
                  " these options out there for every gamer)"]
    return lines


def write_csv(path, found):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["decision", "option", "situation", "real", "taken", "share", "who_did"])
        for name, (total, rows) in found.items():
            for opt, box, n_, k_, ex in rows:
                w.writerow([LABELS.get(name, v21.DECISIONS[name]["label"]), v21.DECISIONS[name]["options"][opt],
                            describe(box), n_, k_,
                            round(k_ / n_, 4), "; ".join(_example(q) for q in ex)])
