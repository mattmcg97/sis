"""v2.1's coverage: for every real state, how much real data the sim's draws there stand on.

    python -m eAMFModel v21-coverage --model v21_dec --since 2026-10-01 --until 2026-10-12 --out eAMFCalibrator/out_v21_coverage

For each real snap's state -- the time draw at the state, the distance draw at the state and the time
its real play took -- the fallback level the league's draw reaches (the features it keeps), how many
real snaps that group holds and, in full mode, whether the gamer's and team's own draws had enough
alike. For each real decision (4th downs, early kicks, conversions) the same for its cell. Written as
one row per state (coverage_states.csv, coverage_decisions.csv) and summed into coverage.txt:

  by the part of the game and the lead, and by the snaps left and the lead: the share of states drawn
  with the whole key (full), with the lead and the time of the game kept but coarser (situation),
  with the time but not the lead (no lead), with neither (bare), and the median group;
  the thinnest situations: where the most states lose the lead or the time of the game.
"""

import csv
import os

import numpy as np

from . import v21

LEAD_F = {"margin", "lead3", "fg4", "lead"}
END_F = {"endgame", "endgame3"}               # the lead (and snaps left) at the end of a half only
TIME_F = {"phase", "part", "half", "qtr", "remain", "rem4", "left", "cell"}
KINDS = ("full", "situation", "no lead", "bare")
PARTS = ("H1 to 2:00", "H1 last 2:00", "H2 to 2:00", "H2 last 2:00", "OT")
LEADS3 = ("behind", "level", "ahead")
DEC_NAMES = ("fourth", "early_fg", "two")


def level_kind(levels, end=None):
    """Each state's KINDS index at its level: 0 the whole key, 1 the lead and the time of the game kept,
    2 the time not the lead, 3 neither. endgame / endgame3 keep the lead only at the end of a half
    (`end`, per state; None: as if every state were there), and the time of the game everywhere."""
    lead = np.array([bool(set(lv) & LEAD_F) for lv in levels])
    endf = np.array([bool(set(lv) & END_F) for lv in levels])
    time_ = np.array([bool(set(lv) & TIME_F) for lv in levels]) | endf

    def kind(lv_idx, at_end=True):
        lv_idx = np.asarray(lv_idx)
        has_lead = lead[lv_idx] | (endf[lv_idx] & np.asarray(at_end))
        k = np.where(has_lead & time_[lv_idx], 1, np.where(time_[lv_idx], 2, 3))
        return np.where(lv_idx == 0, 0, k)
    if end is None:
        return kind(np.arange(len(levels)))
    return kind



def _own_found(model, f, recs):
    """{(kind, role, draw): bool array}: the gamer's or team's own draw had enough alike."""
    out = {}
    for (kind_, role, draw), layer in model.own.items():
        k = 0 if role == "off" else 1
        ids = np.array([model.gamer_index(r["gamer"][k]) if kind_ == "gamer" else model.team_index(r["team"][k])
                        for r in recs])
        out[(kind_, role, draw)] = layer.bin_of(dict(f, who=ids)) >= 0
    return out


def kind_weights(bins, f, end, blend=0.0):
    """((states, 4) each state's draws' weight in each of KINDS, (states,) the real snaps behind its
    draws -- blended, each level's group weighted by its chance -- and the level drawn from most)."""
    kind = level_kind(bins.levels, end)
    n = len(end)
    out = np.zeros((n, len(KINDS)))
    if blend:
        C = bins.chain(f)
        W = bins.blend_weights(C, blend)
        K = np.stack([kind(np.full(n, li), end) for li in range(len(bins.levels))], axis=1)
        for k in range(len(KINDS)):
            out[:, k] = (W * (K == k)).sum(1)
        n_eff = (W * np.where(C >= 0, bins.count[np.maximum(C, 0)], 0)).sum(1)
        return out, n_eff, W.argmax(1)
    g = bins.bin_of(f)
    lv = np.where(g >= 0, bins.level_of[np.maximum(g, 0)], -1)
    kk = np.where(lv >= 0, kind(np.maximum(lv, 0), end), 3)
    out[np.arange(n), kk] = 1.0
    return out, np.where(g >= 0, bins.count[np.maximum(g, 0)], 0).astype(float), lv


def match_coverage(model, code, rows, pair, teams):
    """(state rows, decision rows) of one match."""
    recs = [r for r in v21.match_snaps(rows, pair, teams) if r["period"] <= 4 + v21.MAX_OT]
    states, decisions = [], []
    if recs:
        f = v21._features(recs, model)
        end = f["endgame"] > 0
        blend = getattr(model, "blend", 0.0)
        tw, tn, tl = kind_weights(model.time, f, end, blend)
        dw, dn, dl = kind_weights(model.dist, f, end, blend)
        own = _own_found(model, f, recs) if model.mode == "full" and model.own else {}
        for i, r in enumerate(recs):
            row = dict(code=code, period=r["period"], clock=r["clock"], down=r["down"], distance=r["distance"],
                       field=r["field"], lead=r["margin"], prev=int(r["prev_cls"]), part=int(f["part"][i]),
                       remain=int(f["remain"][i]), endgame=int(f["endgame"][i]), tb=int(f["tb"][i]),
                       time_level=int(tl[i]), time_n=round(float(tn[i]), 1),
                       dist_level=int(dl[i]), dist_n=round(float(dn[i]), 1),
                       **{f"time_{k.replace(' ', '_')}": round(float(tw[i, j]), 4) for j, k in enumerate(KINDS)},
                       **{f"dist_{k.replace(' ', '_')}": round(float(dw[i, j]), 4) for j, k in enumerate(KINDS)})
            for (kind_, role, draw), got in own.items():
                row[f"{kind_}_{role}_{draw}"] = int(got[i])
            states.append(row)
    pts = v21.decision_points(rows, recs)
    for name in DEC_NAMES:
        got = [q for q, k in zip(pts[name], v21.in_draw_window(name, pts[name])) if k]
        ch = model.choices.get(name)
        if not got or ch is None or ch.bins is None:
            continue
        X = v21.decision_features(name, got)
        cols = {fn: X[:, j] for j, fn in enumerate(v21.DECISIONS[name]["features"])}
        b = ch.bins.bin_of(cols)
        for q, g in zip(got, b):
            decisions.append(dict(code=code, decision=name, period=q["period"], clock=q["clock"], lead=q["lead"],
                                  choice=q["choice"], level=int(ch.bins.level_of[g]) if g >= 0 else -1,
                                  n=int(ch.bins.count[g]) if g >= 0 else 0))
    return states, decisions


def run(model_dir, matches, history):
    model_path = os.path.join(model_dir, v21.MODEL_FILE) if os.path.isdir(model_dir) else model_dir
    model = v21.Model.load(model_path)
    hist = {r["MATCH_CODE"]: r for r in (history or [])}
    states, decisions = [], []
    for code, rows in sorted(matches.items()):
        rows = v21._with_side(rows)
        pair, teams = v21.match_people(code, rows, hist)
        s, d = match_coverage(model, code, rows, pair, teams)
        states += s
        decisions += d
    return dict(model=model, states=states, decisions=decisions, matches=len(matches))


def _pct(x, n):
    return 100.0 * x / n if n else float("nan")


def _kinds_cells(weights, sizes):
    n = len(weights)
    cells = [f"{100 * weights[:, k].mean() if n else float('nan'):5.1f}" for k in range(len(KINDS))]
    return cells + [f"{int(np.median(sizes)) if n else 0:>7d}"]


def _table(title, labels, keys, groups, s, model):
    """Rows by `groups` (an int array per state with `labels`), each draw's kinds and median group."""
    keys_ = [k.replace(" ", "_") for k in KINDS]
    tkind = np.array([[x[f"time_{k}"] for k in keys_] for x in s])
    dkind = np.array([[x[f"dist_{k}"] for k in keys_] for x in s])
    tn, dn = np.array([x["time_n"] for x in s]), np.array([x["dist_n"] for x in s])
    head = "".join(f"{k:>10s}" for k in KINDS)
    lines = ["", f"  {title}", f"  {'':26s}{'states':>8s}   time draw:{head}{'median':>8s}   distance draw:{head}{'median':>8s}"]
    for k, lab in zip(keys, labels):
        w = groups == k
        if not w.any():
            continue
        t, d = _kinds_cells(tkind[w], tn[w]), _kinds_cells(dkind[w], dn[w])
        lines.append(f"  {lab:26s}{int(w.sum()):>8d}{'':13s}" + "".join(f"{c:>10s}" for c in t[:-1]) + f"{t[-1]:>8s}"
                     f"{'':17s}" + "".join(f"{c:>10s}" for c in d[:-1]) + f"{d[-1]:>8s}")
    return lines, tkind, dkind


def report(out, top=20):
    model, s, dec = out["model"], out["states"], out["decisions"]
    if not s:
        return ["  no real states"]
    lead3 = np.sign(np.array([x["lead"] for x in s])).astype(np.int64) + 1
    part = np.array([x["part"] for x in s])
    remain = np.array([x["remain"] for x in s])
    lines = [f"  v2.1 coverage on {out['matches']:,} matches, {len(s):,} real states: the level each draw reaches"
             f" (full: the whole key; situation: the lead and the time of the game kept, coarser; no lead: the"
             f" time kept, not the lead; bare: neither), % of the states, and the median real snaps in the group"]
    t1, tkind, dkind = _table("by the part of the game and the lead", [f"{p}, {l}" for p in PARTS for l in LEADS3],
                              [3 * p + l for p in range(len(PARTS)) for l in range(3)], 3 * part + lead3, s, model)
    names = v21.REMAIN_NAMES
    t2, _, _ = _table("by the snaps left in the half and the lead", [f"{r} snaps left, {l}" for r in names for l in LEADS3],
                      [3 * r + l for r in range(len(names)) for l in range(3)], 3 * remain + lead3, s, model)
    lines += t1 + t2
    # in full mode: how often the gamers' and teams' own draws had enough alike
    own_keys = [k for k in s[0] if k.startswith(("gamer_", "team_"))]
    if own_keys:
        end = np.array([x["endgame"] > 0 for x in s])
        lines += ["", "  the gamers' and teams' own draws with enough alike (% of the states: the rest of the game /"
                      " the end of a half)"]
        for k in sorted(own_keys):
            v = np.array([x[k] for x in s], bool)
            lines.append(f"    {k.replace('_', ' '):24s}{_pct(v[~end].sum(), (~end).sum()):6.1f} / {_pct(v[end].sum(), end.sum()):5.1f}")
    # the thinnest situations: by part, lead (margin buckets), down and snaps left, most states losing the lead
    mb = v21.margin_b(np.array([x["lead"] for x in s]))
    down = np.array([x["down"] for x in s])
    key = ((part * 7 + mb) * 4 + np.clip(down - 1, 0, 3)) * len(names) + remain
    lost = dkind[:, 2] + dkind[:, 3]                  # each state's weight of draws losing the lead or the time
    rows = []
    for k in np.unique(key[lost > 0]):
        w = key == k
        p_, rest = divmod(int(k), 7 * 4 * len(names))
        m_, rest = divmod(rest, 4 * len(names))
        d_, r_ = divmod(rest, len(names))
        rows.append((int(round(lost[w].sum())), int(w.sum()), PARTS[p_], v21.MARGIN_NAMES[m_], d_ + 1, names[r_]))
    rows.sort(key=lambda r: -r[0])
    late = lambda r: "last 2:00" in r[2] or r[2] == "OT" or r[5] in names[:3]
    for title, pick in (("the end of a half (its last 2:00, or under 5 snaps left)", late),
                        ("the rest of the game", lambda r: not late(r))):
        lines += ["", f"  the thinnest situations, {title}: where the most states' distance draws lose the lead or"
                      f" the time of the game (of the states there)", f"  {'states losing it':>18s}{'of':>6s}   situation"]
        for n_lost, n_all, p_, m_, d_, r_ in [r for r in rows if pick(r)][:top]:
            lines.append(f"  {n_lost:>18d}{n_all:>6d}   {p_}, {m_}, {d_}{('st', 'nd', 'rd', 'th')[d_ - 1]} down,"
                         f" {r_} snaps left")
    # the decisions
    if dec:
        lines += ["", "  the decisions: the level each real decision's cell reaches (% full / with the lead kept / without),"
                      " median real decisions in the cell; Q4's last 2:00 (conversions: last 3:00) apart"]
        for name in DEC_NAMES:
            d = [q for q in dec if q["decision"] == name]
            ch = model.choices.get(name)
            if not d or ch is None or ch.bins is None:
                continue
            kinds = level_kind(ch.bins.levels)
            late_edge = 180.0 if name == "two" else v21.LATE
            for lab, keep in (("Q4 late", lambda q: q["period"] == 4 and q["clock"] <= late_edge),
                              ("the rest", lambda q: not (q["period"] == 4 and q["clock"] <= late_edge))):
                sel = [q for q in d if keep(q)]
                if not sel:
                    continue
                lv = np.array([q["level"] for q in sel])
                kk = np.where(lv >= 0, kinds[np.maximum(lv, 0)], 3)
                nn = np.array([q["n"] for q in sel])
                lines.append(f"    {v21.DECISIONS[name]['label'].split(' (')[0][:40]:40s} {lab:9s} {len(sel):>6d}:"
                             f" full {_pct((kk == 0).sum(), len(sel)):5.1f}, lead kept {_pct((kk == 1).sum(), len(sel)):5.1f},"
                             f" lost {_pct((kk >= 2).sum(), len(sel)):5.1f}; median {int(np.median(nn))}")
    return lines


def write(out, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for name, rows in (("coverage_states.csv", out["states"]), ("coverage_decisions.csv", out["decisions"])):
        if not rows:
            continue
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    lines = report(out)
    with open(os.path.join(out_dir, "coverage.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines
